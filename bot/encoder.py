"""Sparse observation encoder for the RL policy: turns a parsed `proto.Turn` into feature indices.

Shared by the trainer and the bot (per the plan's "one module, numpy-free, Python 3.13-safe" requirement), so the
policy net always sees the same features it was trained on. Downstream, a numpy first layer does
`W[idx].sum(0) + Wd @ dense + b` (a gather-sum -- `nn.EmbeddingBag(mode="sum")` in torch); this module only produces
`idx` (a list of ints, one per active fact) and `dense` (a short list of floats), no numpy needed here since a plain
Python list is cheaper to build per turn than an array (see the Points lab notes in the plan).

Heading-relative frame: every position (window tiles, edges, dragon parts) is rotated so the dragon's current
heading is "local north" (up), by the same compass-cycle rotation the engine uses for N/E/S/W (proto.DX/DY): a
90-degree turn maps compass north to compass east, so `_rot` cycles offsets the same way `k` times for `dir == k`.
This makes the vocabulary heading-agnostic: a policy that has learned "kelp two ahead" does not need to relearn it
for every absolute compass direction. `dir == -1` (no heading yet, e.g. a fresh split child before its first move)
falls back to the unrotated absolute frame.

Local window cell id, for offset (dx, dy) with dx, dy in -3..3: `(dy + 3) * 7 + (dx + 3)` (0..48, row-major,
row 0 = local-forward-most). Local edge id, for owner-tile offset (dx, dy) with dx, dy in -3..4 (edges extend one
tile beyond the window on the far side): `(dy + 3) * 8 + (dx + 3)` (0..63). "Owner" follows the proto convention:
a horizontal edge belongs to the tile south of it (it is that tile's north edge); a vertical edge belongs to the
tile east of it (it is that tile's west edge) -- see `_edge_owner`.

Vocabulary (contiguous ranges, sizes below; skip a fact for its "default" state instead of encoding it, so the
active count stays small: no pearl, no countdown yet known (-1), no wall/portal, is the usual case per tile):
    pearl_here    49   pearl sits on local cell c                              -> PEARL_BASE + c
    countdown     49*CD_BUCKETS   this tile's pearl timer is in bucket b (c>=0 only) -> CD_BASE + c*CD_BUCKETS + b
    own_body      49   our own trailing segment (not the head) is on cell c    -> OWN_BODY_BASE + c
    team_head     49   a teammate's head is on cell c                         -> TEAM_HEAD_BASE + c
    team_body     49   a teammate's trailing segment is on cell c             -> TEAM_BODY_BASE + c
    enemy_head    49   an enemy head is on cell c                             -> ENEMY_HEAD_BASE + c
    enemy_body    49   an enemy trailing segment is on cell c                 -> ENEMY_BODY_BASE + c
    kelp_h/v      64+64   kelp on the local horizontal/vertical edge owned by cell c
    portal_h/v    64+64   a portal on that edge instead

VOCAB is the total, computed from the sizes above rather than hardcoded twice.

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
EDGE_KELP_H_BASE = ENEMY_BODY_BASE + WINDOW
EDGE_KELP_V_BASE = EDGE_KELP_H_BASE + EDGE_CELLS
EDGE_PORTAL_H_BASE = EDGE_KELP_V_BASE + EDGE_CELLS
EDGE_PORTAL_V_BASE = EDGE_PORTAL_H_BASE + EDGE_CELLS
VOCAB = EDGE_PORTAL_V_BASE + EDGE_CELLS

DENSE_SIZE = 5


def _rot(dx, dy, k):
    """Absolute offset (dx, dy) -> heading-relative offset for a dragon facing compass step k (0=N, 1=E, ...):
    the inverse of rotating content N->E->S->W, so that e.g. absolute east maps to local "forward" when k=1 (E)."""
    for _ in range(k):
        dx, dy = dy, -dx
    return dx, dy


def _window_perm(dir_):
    """absolute window index (row-major, row0=north) -> local cell id, for a dragon facing `dir_`."""
    perm = [0] * WINDOW
    for i in range(WINDOW):
        adx, ady = i % 7 - 3, i // 7 - 3
        ldx, ldy = _rot(adx, ady, dir_) if dir_ >= 0 else (adx, ady)
        perm[i] = (ldy + 3) * 7 + (ldx + 3)
    return tuple(perm)


def _edge_owner(dir_, is_horiz, r, c):
    """Absolute edge (see the module docstring for r, c) -> (local_is_horiz, local_owner_dx, local_owner_dy)."""
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


_WINDOW_PERM = tuple(_window_perm(d) for d in range(4))
_EDGE_OWNER = []
for d in range(4):
    table = {}
    for r in range(8):
        for c in range(7):
            table[(True, r, c)] = _edge_owner(d, True, r, c)
    for r in range(7):
        for c in range(8):
            table[(False, r, c)] = _edge_owner(d, False, r, c)
    _EDGE_OWNER.append(table)
_EDGE_OWNER = tuple(_EDGE_OWNER)


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
    dir_ = t.dir
    perm = _WINDOW_PERM[dir_] if dir_ >= 0 else _WINDOW_PERM[0]
    eowner = _EDGE_OWNER[dir_] if dir_ >= 0 else _EDGE_OWNER[0]

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
        ldx, ldy = _rot(dx, dy, dir_) if dir_ >= 0 else (dx, dy)
        idx.add(base + (ldy + 3) * 7 + (ldx + 3))

    ed = t.edges
    for r in range(8):
        line = ed[r]
        if line == DOT7:
            continue
        for c, tok in enumerate(line.split()):
            if tok == b".":
                continue
            is_h, ox, oy = eowner[(True, r, c)]
            base_kelp = EDGE_KELP_H_BASE if is_h else EDGE_KELP_V_BASE
            base_portal = EDGE_PORTAL_H_BASE if is_h else EDGE_PORTAL_V_BASE
            idx.add((base_kelp if tok == b"w" else base_portal) + (oy + 3) * EDGE_SPAN + (ox + 3))
    for r in range(7):
        line = ed[8 + r]
        if line == DOT8:
            continue
        for c, tok in enumerate(line.split()):
            if tok == b".":
                continue
            is_h, ox, oy = eowner[(False, r, c)]
            base_kelp = EDGE_KELP_H_BASE if is_h else EDGE_KELP_V_BASE
            base_portal = EDGE_PORTAL_H_BASE if is_h else EDGE_PORTAL_V_BASE
            idx.add((base_kelp if tok == b"w" else base_portal) + (oy + 3) * EDGE_SPAN + (ox + 3))

    msgs = t.msgs
    dense = [
        t.length / 64.0,
        t.units / unit_limit if unit_limit else 0.0,
        t.rnd / 500.0,
        min(len(msgs), 8) / 8.0,
        (msgs[0] % 65536) / 65536.0 if msgs else 0.0,
    ]
    return list(idx), dense
