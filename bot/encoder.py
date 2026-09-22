"""Sparse observation encoder for the RL policy: turns a parsed `proto.Turn` into feature indices.

Shared by the trainer and the bot (per the plan's "one module, numpy-free, Python 3.13-safe" requirement), so the
policy net always sees the same features it was trained on. Downstream, a numpy first layer does
`W[idx].sum(0) + Wd @ dense + b` (a gather-sum -- `nn.EmbeddingBag(mode="sum")` in torch); this module only produces
`idx` (a list of ints, one per active fact) and `dense` (a short list of floats), no numpy needed here since a plain
Python list is cheaper to build per turn than an array (see the Points lab notes in the plan).

Heading-relative frame: every position (window tiles, edges, dragon parts) is rotated so the dragon's current
heading is "local north" (up): a 90-degree turn maps compass north to compass east (proto.DX/DY), so converting an
absolute offset to heading-relative applies the inverse of that, `k` times, for `dir == k` (see `_rot`). This makes
the vocabulary heading-agnostic: a policy that has learned "kelp two ahead" does not need to relearn it for every
absolute compass direction. `dir == -1` (no heading yet, e.g. a fresh split child before its first move) falls back
to the unrotated absolute frame.

The four window and edge lookup tables below (`_WINDOW_PERM`, `_EDGE_H_CODE`, `_EDGE_V_CODE`) are that rotation,
precomputed once per heading -- but baked in as literal tuples rather than built by a Python loop at import time.
A Python loop costs thousands of CPU points per iteration on the judge (the Points lab measured helper.py's three
Enum classes alone at ~5.5M points for this reason); building ~600 table entries via `_rot`/`_edge_owner` calls at
import blew the per-turn budget by tens of millions of points in a `--sandbox` check. `tools/gen_encoder_tables.py`
regenerates these tuples from the (kept, for reference) `_rot`/`_edge_owner` functions if the geometry ever changes.

Local window cell id, for offset (dx, dy) with dx, dy in -3..3: `(dy + 3) * 7 + (dx + 3)` (0..48, row-major,
row 0 = local-forward-most). Local edge code, for owner-tile offset (dx, dy) with dx, dy in -3..4 (edges extend one
tile beyond the window on the far side): `(dy + 3) * 8 + (dx + 3)`, plus EDGE_CELLS (64) if the edge is vertical in
the local frame (0..63 local-horizontal, 64..127 local-vertical) -- folded into one code so the per-turn loop below
does not need to branch on edge orientation. "Owner" follows the proto convention: a horizontal edge belongs to the
tile south of it (it is that tile's north edge); a vertical edge belongs to the tile east of it (its west edge).

Vocabulary (contiguous ranges, sizes below; skip a fact for its "default" state instead of encoding it, so the
active count stays small: no pearl, no countdown yet known (-1), no wall/portal, is the usual case per tile):
    pearl_here    49              pearl sits on local cell c                      -> PEARL_BASE + c
    countdown     49*CD_BUCKETS   this tile's pearl timer is in bucket b (c>=0)    -> CD_BASE + c*CD_BUCKETS + b
    own_body      49              our own trailing segment (not the head) on c    -> OWN_BODY_BASE + c
    team_head     49              a teammate's head is on cell c                  -> TEAM_HEAD_BASE + c
    team_body     49              a teammate's trailing segment is on cell c      -> TEAM_BODY_BASE + c
    enemy_head    49              an enemy head is on cell c                      -> ENEMY_HEAD_BASE + c
    enemy_body    49              an enemy trailing segment is on cell c          -> ENEMY_BODY_BASE + c
    edge_kelp     128             kelp on the local edge with code e              -> EDGE_KELP_BASE + e
    edge_portal   128             a portal on that edge instead                   -> EDGE_PORTAL_BASE + e

VOCAB is the total, computed from the sizes above rather than hardcoded twice. Real-game sampling (two full arena
games, 29k turns) gave 50-105 active features per turn -- within the plan's ~100-200 estimate.

Dense scalars (fixed order, always present): length/64, units/limit, round/500, min(len(msgs), 8)/8, first sonar
value/65536 (0 if none). Sonar is not decoded here: no wire protocol has been designed yet (see the plan's
unknowns), so this is a placeholder signal, not a feature.

Not yet included (deferred, see the plan): remembered-map features (kelp/portals/countdowns seen on earlier turns
but outside the current window) and a decoded sonar payload.
"""
from proto import DOT7, DOT8

WINDOW = 49
CD_BUCKETS = 6
EDGE_SPAN = 8  # offsets -3..4
EDGE_CELLS = EDGE_SPAN * EDGE_SPAN

PEARL_BASE = 0
CD_BASE = PEARL_BASE + WINDOW
OWN_BODY_BASE = CD_BASE + WINDOW * CD_BUCKETS
TEAM_HEAD_BASE = OWN_BODY_BASE + WINDOW
TEAM_BODY_BASE = TEAM_HEAD_BASE + WINDOW
ENEMY_HEAD_BASE = TEAM_BODY_BASE + WINDOW
ENEMY_BODY_BASE = ENEMY_HEAD_BASE + WINDOW
EDGE_KELP_BASE = ENEMY_BODY_BASE + WINDOW
EDGE_PORTAL_BASE = EDGE_KELP_BASE + 2 * EDGE_CELLS
VOCAB = EDGE_PORTAL_BASE + 2 * EDGE_CELLS

DENSE_SIZE = 5


def _rot(dx, dy, k):
    """Absolute offset (dx, dy) -> heading-relative offset for a dragon facing compass step k (0=N, 1=E, ...):
    the inverse of rotating content N->E->S->W, so that e.g. absolute east maps to local "forward" when k=1 (E).
    Dev-time only (used by tools/gen_encoder_tables.py); the per-turn path uses the baked tables below instead."""
    for _ in range(k):
        dx, dy = dy, -dx
    return dx, dy


def _edge_owner(dir_, is_horiz, r, c):
    """Dev-time only, see `_rot`. Absolute edge (r, c as proto lays them out) -> (local_is_horiz, owner_dx, owner_dy)."""
    if is_horiz:  # r'th row = north edge of window row r: owner is the tile south of it, neighbour is north
        odx, ody = c - 3, r - 3
        ndx, ndy = odx, ody - 1
    else:  # west edge of window column c: owner is the tile east of it, neighbour is west
        odx, ody = c - 3, r - 3
        ndx, ndy = odx - 1, ody
    lox, loy = _rot(odx, ody, dir_) if dir_ >= 0 else (odx, ody)
    lnx, lny = _rot(ndx, ndy, dir_) if dir_ >= 0 else (ndx, ndy)
    ddx, ddy = lnx - lox, lny - loy  # one of (0,-1) (0,1) (1,0) (-1,0): local direction from owner to neighbour
    if ddx == 0:
        return (True, lox, loy) if ddy == -1 else (True, lnx, lny)  # whichever tile is local-south owns it
    return (False, lox, loy) if ddx == -1 else (False, lnx, lny)  # whichever tile is local-east owns it


# Baked by tools/gen_encoder_tables.py from _rot/_edge_owner above -- do not hand-edit, regenerate instead.
# absolute window index (row-major, row0=north) -> local cell id, per heading dir (N, E, S, W).
_WINDOW_PERM = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
     30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48),
    (42, 35, 28, 21, 14, 7, 0, 43, 36, 29, 22, 15, 8, 1, 44, 37, 30, 23, 16, 9, 2, 45, 38, 31, 24, 17, 10, 3, 46, 39,
     32, 25, 18, 11, 4, 47, 40, 33, 26, 19, 12, 5, 48, 41, 34, 27, 20, 13, 6),
    (48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21,
     20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
    (6, 13, 20, 27, 34, 41, 48, 5, 12, 19, 26, 33, 40, 47, 4, 11, 18, 25, 32, 39, 46, 3, 10, 17, 24, 31, 38, 45, 2, 9,
     16, 23, 30, 37, 44, 1, 8, 15, 22, 29, 36, 43, 0, 7, 14, 21, 28, 35, 42),
)
# absolute horizontal-edge (row r in 0..7, col c in 0..6, index r*7+c) -> local edge code, per heading dir.
_EDGE_H_CODE = (
    (0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 32, 33,
     34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 53, 54, 56, 57, 58, 59, 60, 61, 62),
    (112, 104, 96, 88, 80, 72, 64, 113, 105, 97, 89, 81, 73, 65, 114, 106, 98, 90, 82, 74, 66, 115, 107, 99, 91, 83,
     75, 67, 116, 108, 100, 92, 84, 76, 68, 117, 109, 101, 93, 85, 77, 69, 118, 110, 102, 94, 86, 78, 70, 119, 111,
     103, 95, 87, 79, 71),
    (62, 61, 60, 59, 58, 57, 56, 54, 53, 52, 51, 50, 49, 48, 46, 45, 44, 43, 42, 41, 40, 38, 37, 36, 35, 34, 33, 32,
     30, 29, 28, 27, 26, 25, 24, 22, 21, 20, 19, 18, 17, 16, 14, 13, 12, 11, 10, 9, 8, 6, 5, 4, 3, 2, 1, 0),
    (71, 79, 87, 95, 103, 111, 119, 70, 78, 86, 94, 102, 110, 118, 69, 77, 85, 93, 101, 109, 117, 68, 76, 84, 92,
     100, 108, 116, 67, 75, 83, 91, 99, 107, 115, 66, 74, 82, 90, 98, 106, 114, 65, 73, 81, 89, 97, 105, 113, 64, 72,
     80, 88, 96, 104, 112),
)
# absolute vertical-edge (row r in 0..6, col c in 0..7, index r*8+c) -> local edge code, per heading dir.
_EDGE_V_CODE = (
    (64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91,
     92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115,
     116, 117, 118, 119),
    (56, 48, 40, 32, 24, 16, 8, 0, 57, 49, 41, 33, 25, 17, 9, 1, 58, 50, 42, 34, 26, 18, 10, 2, 59, 51, 43, 35, 27,
     19, 11, 3, 60, 52, 44, 36, 28, 20, 12, 4, 61, 53, 45, 37, 29, 21, 13, 5, 62, 54, 46, 38, 30, 22, 14, 6),
    (119, 118, 117, 116, 115, 114, 113, 112, 111, 110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100, 99, 98, 97,
     96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69,
     68, 67, 66, 65, 64),
    (6, 14, 22, 30, 38, 46, 54, 62, 5, 13, 21, 29, 37, 45, 53, 61, 4, 12, 20, 28, 36, 44, 52, 60, 3, 11, 19, 27, 35,
     43, 51, 59, 2, 10, 18, 26, 34, 42, 50, 58, 1, 9, 17, 25, 33, 41, 49, 57, 0, 8, 16, 24, 32, 40, 48, 56),
)


def _cd_bucket(cd):
    """-1 (never / not yet known) returns -1 (no feature emitted); otherwise a bucket in 0..CD_BUCKETS-1."""
    if cd < 0:
        return -1
    if cd == 0:
        return 0
    if cd == 1:
        return 1
    if cd <= 3:
        return 2
    if cd <= 7:
        return 3
    if cd <= 15:
        return 4
    return 5


def encode(t, team, my_id, width, height, unit_limit):
    """(idx, dense): idx a list of unique active feature ints in [0, VOCAB); dense a list of DENSE_SIZE floats."""
    dir_ = t.dir if t.dir >= 0 else 0
    perm = _WINDOW_PERM[dir_]
    hcode = _EDGE_H_CODE[dir_]
    vcode = _EDGE_V_CODE[dir_]

    idx = set()
    for i in range(WINDOW):
        c = perm[i]
        if t.flags[i] == b"1":
            idx.add(PEARL_BASE + c)
        b = _cd_bucket(int(t.cds[i]))
        if b >= 0:
            idx.add(CD_BASE + c * CD_BUCKETS + b)

    for q in t.parts:
        pid = int(q[1])
        if pid == my_id:
            if q[5] == b"1":
                continue  # our own head: always at the centre, not worth a feature
            base = OWN_BODY_BASE
        else:
            is_head = q[5] == b"1"
            enemy = q[0] != team
            base = (ENEMY_HEAD_BASE if is_head else ENEMY_BODY_BASE) if enemy else (TEAM_HEAD_BASE if is_head else TEAM_BODY_BASE)
        x, y = int(q[2]), int(q[3])
        dx = ((x - t.hx + 3) % width) - 3
        dy = ((y - t.hy + 3) % height) - 3
        ldx, ldy = _rot(dx, dy, dir_) if dir_ else (dx, dy)
        idx.add(base + (ldy + 3) * 7 + (ldx + 3))

    ed = t.edges
    for r in range(8):
        line = ed[r]
        if line == DOT7:
            continue
        row = hcode[r * 7:r * 7 + 7]
        for c, tok in enumerate(line.split()):
            if tok != b".":
                idx.add((EDGE_KELP_BASE if tok == b"w" else EDGE_PORTAL_BASE) + row[c])
    for r in range(7):
        line = ed[8 + r]
        if line == DOT8:
            continue
        row = vcode[r * 8:r * 8 + 8]
        for c, tok in enumerate(line.split()):
            if tok != b".":
                idx.add((EDGE_KELP_BASE if tok == b"w" else EDGE_PORTAL_BASE) + row[c])

    msgs = t.msgs
    dense = [
        t.length / 64.0,
        t.units / unit_limit if unit_limit else 0.0,
        t.rnd / 500.0,
        min(len(msgs), 8) / 8.0,
        (msgs[0] % 65536) / 65536.0 if msgs else 0.0,
    ]
    return list(idx), dense
