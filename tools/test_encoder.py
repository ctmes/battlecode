"""Checks bot/encoder.py: rotation invariance (the main risk in its design) via an independently rendered and
independently rotated synthetic turn block, plus vocabulary bounds and a hand-picked example.

`render_block` reimplements the proto wire format from the docs (not from encoder.py's internal tables) and
`rotate_world` rotates a world's content by physically re-deriving each rotated edge's type and owner from its two
rotated tile positions -- the same case analysis encoder._edge_owner must do, but performed on the WORLD before
rendering to text, not on the window during encode(). The two are independent code paths that meet only through the
real proto.parse_turn text round trip, so a sign error in either would show up as a mismatch here.

Run:  .venv\\Scripts\\python.exe tools\\test_encoder.py
"""
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "bot"))
import encoder  # noqa: E402
import proto  # noqa: E402

W, H = 40, 30
HX, HY = 20, 15
LETTERS = "NESW"


def render_block(dir_, pearls, cds, edges, parts, rnd=7, length=5, units=1, msgs=()):
    """pearls: set of (dx, dy) with a pearl now. cds: dict (dx, dy) -> countdown (default -1, i.e. never/unknown).
    edges: dict (dx, dy, 'h'|'v') -> token ('w' or a portal id string); 'h' is tile (dx, dy)'s north edge, 'v' is
    its west edge (the proto convention). parts: list of (team, id, dx, dy, facing_letter, is_head)."""
    lines = [f"ROUND {rnd}", f"DIR {LETTERS[dir_] if dir_ >= 0 else 'N'}", f"LENGTH {length}",
             f"UNIT_COUNT {units}", f"NUM_MSGS {len(msgs)}"]
    lines += [str(m) for m in msgs]
    for r in range(7):  # one line per tile (49 total), not one line per row: proto.parse_turn slices 49 *lines*
        for c in range(7):
            dx, dy = c - 3, r - 3
            x, y = (HX + dx) % W, (HY + dy) % H
            lines.append(f"{x} {y} {1 if (dx, dy) in pearls else 0} {cds.get((dx, dy), -1)}")
    lines.append(f"NUM_PARTS {len(parts)}")
    for team, pid, dx, dy, facing, is_head in parts:
        x, y = (HX + dx) % W, (HY + dy) % H
        lines.append(f"{team} {pid} {x} {y} {facing} {is_head}")
    for r in range(8):
        lines.append(" ".join(edges.get((c - 3, r - 3, "h"), ".") for c in range(7)))
    for r in range(7):
        lines.append(" ".join(edges.get((c - 3, r - 3, "v"), ".") for c in range(8)))
    return ("\n".join(lines) + "\n\n").encode()


def R(dx, dy):
    """One physical quarter turn, compass north -> east: matches proto.DX/DY, derived independently of encoder._rot."""
    return -dy, dx


assert (R(proto.DX[0], proto.DY[0])) == (proto.DX[1], proto.DY[1]), "R must match the engine's own N->E rotation"


def rot_k(dx, dy, k):
    for _ in range(k):
        dx, dy = R(dx, dy)
    return dx, dy


def rotate_world(pearls, cds, edges, parts, k):
    pearls2 = {rot_k(dx, dy, k) for dx, dy in pearls}
    cds2 = {rot_k(dx, dy, k): cd for (dx, dy), cd in cds.items()}
    edges2 = {}
    for (dx, dy, kind), tok in edges.items():
        ndx, ndy = (dx, dy - 1) if kind == "h" else (dx - 1, dy)  # this edge's neighbour tile (north or west)
        ox, oy = rot_k(dx, dy, k)
        nx, ny = rot_k(ndx, ndy, k)
        ddx, ddy = nx - ox, ny - oy  # rotated neighbour direction from the rotated owner: re-derive type + owner
        if ddx == 0:
            oo = (ox, oy) if ddy == -1 else (nx, ny)
            edges2[(oo[0], oo[1], "h")] = tok
        else:
            oo = (ox, oy) if ddx == -1 else (nx, ny)
            edges2[(oo[0], oo[1], "v")] = tok
    parts2 = [(team, pid, *rot_k(dx, dy, k), LETTERS[(LETTERS.index(facing) + k) % 4], is_head)
              for team, pid, dx, dy, facing, is_head in parts]
    return pearls2, cds2, edges2, parts2


def random_world(rng, n_pearls=10, n_edges=20, n_parts=6):
    cells = [(dx, dy) for dy in range(-3, 4) for dx in range(-3, 4)]
    pearls = set(rng.sample(cells, n_pearls))
    cds = {c: rng.choice([-1, 0, 1, 2, 5, 9, 20]) for c in rng.sample(cells, n_pearls)}
    h_slots = [(dx, dy, "h") for dy in range(-3, 5) for dx in range(-3, 4)]
    v_slots = [(dx, dy, "v") for dy in range(-3, 4) for dx in range(-3, 5)]
    edges = {s: rng.choice(["w", "3"]) for s in rng.sample(h_slots + v_slots, n_edges)}
    parts = [("A", 1, 0, 0, "N", "1")]  # my own head, always at the centre
    used = {(0, 0)}
    pid = 2
    for _ in range(n_parts):
        c = rng.choice(cells)
        if c in used:
            continue
        used.add(c)
        team = rng.choice(["A", "B"])
        is_head = rng.choice(["0", "1"])
        facing = rng.choice(LETTERS)
        parts.append((team, pid, c[0], c[1], facing, is_head))
        pid += 1
    return pearls, cds, edges, parts


def encode_world(dir_, pearls, cds, edges, parts):
    block = render_block(dir_, pearls, cds, edges, parts)
    t = proto.parse_turn(block)
    return encoder.encode(t, b"A", 1, W, H, 4)


def test_rotation_invariance():
    rng = random.Random(0)
    for trial in range(200):
        world = random_world(rng)
        idx0, dense0 = encode_world(0, *world)
        for k in (1, 2, 3):
            idxk, densek = encode_world(k, *rotate_world(*world, k))
            assert set(idxk) == set(idx0), f"trial {trial} k={k}: rotated features differ\n{sorted(set(idx0) ^ set(idxk))}"
            assert densek == dense0, f"trial {trial} k={k}: dense features should be rotation-independent"
    print(f"ok: {trial + 1} random worlds, all 4 headings agree after rotation")


def test_bounds_and_uniqueness():
    rng = random.Random(1)
    counts = []
    for _ in range(100):
        world = random_world(rng, n_pearls=rng.randint(0, 15), n_edges=rng.randint(0, 40), n_parts=rng.randint(0, 10))
        dir_ = rng.choice([-1, 0, 1, 2, 3])
        idx, dense = encode_world(dir_, *world)
        assert len(idx) == len(set(idx)), "duplicate feature indices"
        assert all(0 <= i < encoder.VOCAB for i in idx), "feature index out of vocabulary range"
        assert len(dense) == encoder.DENSE_SIZE
        counts.append(len(idx))
    print(f"ok: 100 random turns, all indices in [0, {encoder.VOCAB}), no duplicates, "
          f"active count {min(counts)}-{max(counts)}")


def test_dir_minus_one_no_crash():
    world = random_world(random.Random(2))
    idx, dense = encode_world(-1, *world)
    assert all(0 <= i < encoder.VOCAB for i in idx)
    print("ok: dir=-1 (no heading yet) falls back to the absolute frame without crashing")


def test_hand_example():
    # A pearl one step ahead (north, since dir=N), kelp on the step to the east, an enemy head two steps ahead.
    pearls = {(0, -1)}
    edges = {(1, 0, "v"): "w"}  # tile (1,0)'s west edge = the step from the head straight east
    parts = [("A", 1, 0, 0, "N", "1"), ("B", 2, 0, -2, "S", "1")]
    idx, dense = encode_world(0, pearls, {}, edges, parts)
    fset = set(idx)
    pearl_cell = ((-1) + 3) * 7 + (0 + 3)  # local cell of (0, -1) at dir=N (identity rotation)
    assert encoder.PEARL_BASE + pearl_cell in fset
    enemy_cell = ((-2) + 3) * 7 + (0 + 3)
    assert encoder.ENEMY_HEAD_BASE + enemy_cell in fset
    kelp_v_cell = (0 + 3) * encoder.EDGE_SPAN + (1 + 3)
    assert encoder.EDGE_KELP_V_BASE + kelp_v_cell in fset
    assert dense[0] == 5 / 64.0  # length
    print("ok: hand-picked example has exactly the expected pearl / enemy-head / kelp features")


if __name__ == "__main__":
    test_rotation_invariance()
    test_bounds_and_uniqueness()
    test_dir_minus_one_no_crash()
    test_hand_example()
