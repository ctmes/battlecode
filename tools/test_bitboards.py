"""Checks Brain.flood / Brain.layers (Python-int bitboards on a torus with kelp/portal edges) against a naive BFS.

Run:  python tools/test_bitboards.py
"""
import pathlib
import random
import sys
from collections import deque

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "bot"))
from brain import Brain  # noqa: E402


def naive_dist(w, h, free, wv, wh, sources, kmax):
    """Multi-source BFS distances (<= kmax) on the torus. wv/wh: cells whose WEST/NORTH edge is a wall."""
    dist = {s: 0 for s in sources}
    q = deque(sources)
    while q:
        x, y = q.popleft()
        d = dist[(x, y)]
        if d >= kmax:
            continue
        moves = (
            (((x + 1) % w, y), ((x + 1) % w, y) in wv),  # east: dest's west edge
            (((x - 1) % w, y), (x, y) in wv),  # west: source's west edge
            ((x, (y + 1) % h), (x, (y + 1) % h) in wh),  # south: dest's north edge
            ((x, (y - 1) % h), (x, y) in wh),  # north: source's north edge
        )
        for nxt, blocked in moves:
            if not blocked and nxt in free and nxt not in dist:
                dist[nxt] = d + 1
                q.append(nxt)
    return dist


def bits(cells, w):
    m = 0
    for x, y in cells:
        m |= 1 << (y * w + x)
    return m


def run(trials=400, seed=1):
    rng = random.Random(seed)
    for trial in range(trials):
        w = rng.randint(10, 24)
        h = rng.randint(10, 24)
        b = Brain(0, b"A", w, h, 64)
        cells = [(x, y) for y in range(h) for x in range(w)]
        density = rng.choice((0.5, 0.7, 0.9, 1.0))
        free = {c for c in cells if rng.random() < density}
        kelp_rate = rng.choice((0.0, 0.05, 0.15))
        kv_w = {c for c in cells if rng.random() < kelp_rate}
        kh_w = {c for c in cells if rng.random() < kelp_rate}
        pv_w = {c for c in cells if rng.random() < 0.02} - kv_w  # portal edges are walls too
        b.kv, b.kh, b.pv, b.ph = bits(kv_w, w), bits(kh_w, w), bits(pv_w, w), 0
        b.okv = b.full ^ (b.kv | b.pv)
        b.okh = b.full ^ (b.kh | b.ph)
        wv = kv_w | pv_w
        wh = kh_w
        free_mask = bits(free, w)
        if not free:
            continue
        start = rng.choice(sorted(free))
        want = naive_dist(w, h, free, wv, wh, [start], 10 ** 9)
        got = b.flood(free_mask, start[1] * w + start[0], 10 ** 9)
        assert got == len(want), f"flood mismatch trial {trial}: {got} != {len(want)} ({w}x{h})"
        # early exit never reports fewer than `need` when enough cells are reachable
        need = rng.randint(1, 40)
        got_n = b.flood(free_mask, start[1] * w + start[0], need)
        assert got_n >= min(need, len(want)), f"early exit under-reports trial {trial}"
        assert got_n <= len(want)
        # layers from several sources
        srcs = rng.sample(sorted(free), min(len(free), rng.randint(1, 5)))
        k = 6
        ref = naive_dist(w, h, free, wv, wh, srcs, k)
        lay = b.layers(bits(srcs, w), free_mask, k)
        for i in range(len(lay)):
            want_i = bits([c for c, dd in ref.items() if dd <= i], w)
            assert lay[i] == want_i, f"layer {i} mismatch trial {trial} ({w}x{h})"
        # layers stop only when nothing grows
        if len(lay) < k + 1:
            assert lay[-1] == bits(list(ref), w)
        # 7x7 window mask, including wrap-around
        x0, y0 = rng.randrange(w), rng.randrange(h)
        assert b.window(x0, y0) == bits([((x0 + c) % w, (y0 + r) % h) for r in range(7) for c in range(7)], w), \
            f"window mismatch trial {trial}"
    print(f"ok: {trials} random tori, flood + early exit + layers + window mask match naive references")


if __name__ == "__main__":
    run()
