"""Random symmetric maps in the engine's format, for tuning and testing on more than the 8 bundled boards.

    .venv\\Scripts\\python.exe tools\\mapgen.py --count 400 --seed 0 --out maps\\gen [--max-side 40] [--preview 3]

generate(seed) is deterministic and returns a valid, fully connected map: size 10..64 on each side, one of the three
symmetries, kelp walls (straight, bent, staircases, rooms with doors, dead-end pockets), portal pairs, a pearl supply
(uniform / rich patches on sparse floor / oases on barren floor / food-rich), and 1-4 mirrored dragons per team.
The mix is modelled on the bundled maps (open fields with a few walls, portals joining distant spots, sparse pearls
with rich patches, dragons in opposite regions); it is a proxy for the ladder's maps, not a copy of them.
Not generated: void tiles (arena.map) and walled-off decorative pockets.
"""
import argparse
import math
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from mapfile import GameMap, MAX_SIDE, MIN_SIDE, NORTH, SYMMETRIES, WEST  # noqa: E402

STYLES = {  # feature weights per wall style: (straight, bent, staircase, room, pocket)
    "open": (0, 0, 0, 0, 0),
    "walls": (5, 2, 3, 0, 0),
    "rooms": (3, 0, 0, 7, 0),
    "pockets": (4, 0, 0, 0, 6),
    "mixed": (3, 2, 2, 3, 2),
}
STYLE_ODDS = {"open": 1, "walls": 3, "rooms": 2, "pockets": 1, "mixed": 3}


def logu(rng, lo, hi):
    return int(round(math.exp(rng.uniform(math.log(lo), math.log(hi)))))


def pick_size(rng, lo, hi):
    """Small, medium and large boards in roughly the proportions of the bundled maps; squares about half the time."""
    r = rng.random()
    a, b = (lo, 20) if r < 0.30 else ((20, 40) if r < 0.75 else (40, hi))
    a, b = min(a, hi), min(max(a, b), hi)
    w = rng.randint(a, b)
    return w, (w if rng.random() < 0.5 else rng.randint(a, b))


# ---------------------------------------------------------------------- kelp features (edge lists, before mirroring)
def polyline(cx, cy, moves):
    """Edges along a path over tile corners; corner (cx, cy) is the top-left corner of tile (cx, cy), moves are N/E/S/W."""
    out = []
    for mv in moves:
        if mv == "E":
            out.append((cx, cy, NORTH))
            cx += 1
        elif mv == "W":
            cx -= 1
            out.append((cx, cy, NORTH))
        elif mv == "S":
            out.append((cx, cy, WEST))
            cy += 1
        else:
            cy -= 1
            out.append((cx, cy, WEST))
    return out


def straight(rng, w, h):
    return polyline(rng.randrange(w), rng.randrange(h), rng.choice("NESW") * rng.randint(2, max(3, min(w, h) // 2)))


def bent(rng, w, h):
    d, moves = rng.choice("NESW"), ""
    for _ in range(rng.randint(2, 3)):
        moves += d * rng.randint(1, max(2, min(w, h) // 4))
        d = rng.choice("EW" if d in "NS" else "NS")
    return polyline(rng.randrange(w), rng.randrange(h), moves)


def staircase(rng, w, h):
    a, b = rng.choice("EW"), rng.choice("NS")
    moves = "".join(a * rng.randint(1, 2) + b * rng.randint(1, 2) for _ in range(rng.randint(2, 8)))
    return polyline(rng.randrange(w), rng.randrange(h), moves)


def room(rng, w, h):
    """A rectangle of wall with one to three doors."""
    bw, bh = rng.randint(3, max(4, w // 3)), rng.randint(3, max(4, h // 3))
    edges = polyline(rng.randrange(w), rng.randrange(h), "E" * bw + "S" * bh + "W" * bw + "N" * bh)
    for door in rng.sample(range(len(edges)), rng.randint(1, 3)):
        edges[door] = None
    return [e for e in edges if e]


def pocket(rng, w, h):
    """Three walls around one tile: a dead end whose only way out is the fourth side."""
    x, y = rng.randrange(w), rng.randrange(h)
    sides = [(x, y, NORTH), (x, y, WEST), (x + 1, y, WEST), (x, y + 1, NORTH)]
    sides.pop(rng.randrange(4))
    return sides


FEATURES = (straight, bent, staircase, room, pocket)


def add_walls(rng, m):
    style = rng.choices(list(STYLES), [STYLE_ODDS[s] for s in STYLES])[0]
    weights = STYLES[style]
    if not any(weights):
        return style
    target = rng.uniform(0.06, 0.36) * m.w * m.h  # kelp edges, mirror images included: 3-18% of the 2*w*h edges
    for _ in range(2000):
        if len(m.kelp) >= target:
            break
        for x, y, d in rng.choices(FEATURES, weights)[0](rng, m.w, m.h):
            e = (x % m.w, y % m.h, d)
            m.kelp.update({e, m.mirror_edge(e)})
    return style


def labels(m):
    """Region number of every tile (moves through open edges and portals)."""
    lab, n = {}, 0
    for start in m.tiles:
        if start in lab:
            continue
        lab[start] = n
        todo = [start]
        while todo:
            x, y = todo.pop()
            for d in range(4):
                nxt = m.step(x, y, d)
                if nxt is not None and nxt not in lab:
                    lab[nxt] = n
                    todo.append(nxt)
        n += 1
    return lab, n


def open_doors(rng, m):
    """Joins the regions the walls cut off by deleting one separating kelp edge (and its mirror) per region and pass."""
    while True:
        lab, n = labels(m)
        if n == 1:
            return
        door = {}
        for x, y, d in m.kelp:
            a = (x, y)
            b = (x, (y - 1) % m.h) if d == NORTH else ((x - 1) % m.w, y)
            if lab[a] != lab[b]:
                door.setdefault(lab[a], []).append((x, y, d))
                door.setdefault(lab[b], []).append((x, y, d))
        for region in sorted(door)[:-1] if len(door) > 1 else door:
            e = rng.choice(sorted(door[region]))
            m.kelp.discard(e)
            m.kelp.discard(m.mirror_edge(e))


# ---------------------------------------------------------------------- portals
def add_portals(rng, m):
    want = min(rng.choice((0, 0, 1, 2, 2, 3, 4, 6)), m.w * m.h // 60)
    far = max(5, min(m.w, m.h) // 3)
    pid, made = 0, 0
    for _ in range(300):
        if made == want:
            break
        kind = rng.choice((NORTH, WEST))
        p = (rng.randrange(m.w), rng.randrange(m.h), kind)
        q = (rng.randrange(m.w), rng.randrange(m.h), kind)
        p2, q2 = m.mirror_edge(p), m.mirror_edge(q)
        four = {p, q, p2, q2}
        dx, dy = abs(p[0] - q[0]), abs(p[1] - q[1])
        if len(four) < 4 or four & m.kelp or four & set(m.portals) or min(dx, m.w - dx) + min(dy, m.h - dy) < far:
            continue
        m.portals.update({p: pid, q: pid, p2: pid + 1, q2: pid + 1})
        pid += 2
        made += 1


# ---------------------------------------------------------------------- pearls
def blob(rng, m):
    """Tiles of a random ellipse, wrapped on the torus."""
    r = max(2, min(m.w, m.h) // 4)
    cx, cy, rx, ry = rng.randrange(m.w), rng.randrange(m.h), rng.randint(2, r), rng.randint(2, r)
    return [(x % m.w, y % m.h) for x in range(cx - rx, cx + rx + 1) for y in range(cy - ry, cy + ry + 1)
            if ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1]


def add_pearls(rng, m):
    regime = rng.choice(("uniform", "uniform", "patches", "patches", "oases", "rich"))
    lo = rng.choice((1, 1, 1, 2, 5))
    if regime == "rich":
        base = (lo, max(lo + 1, logu(rng, 8, 60)))
    elif regime == "uniform":
        base = (lo, max(lo + 1, logu(rng, 30, 1500)))
    elif regime == "patches":
        base = (lo, logu(rng, 300, 2000))
    else:
        base = (0, 0)
    m.tiles = {(x, y): base for x in range(m.w) for y in range(m.h)}
    if regime in ("patches", "oases"):
        target = 0.12 if regime == "patches" else rng.uniform(0.10, 0.45)
        rich = (lo, max(lo + 1, logu(rng, 8, 100 if regime == "patches" else 400)))
        for _ in range(60):
            if sum(g != base for g in m.tiles.values()) >= target * m.w * m.h:
                break
            for t in blob(rng, m):
                m.tiles[t] = m.tiles[m.mirror_tile(*t)] = rich
    return regime


# ---------------------------------------------------------------------- dragons
def torus_dist(m, a, b):
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    return min(dx, m.w - dx) + min(dy, m.h - dy)


def grow_body(rng, m, length, used):
    """A head-first chain of adjacent tiles joined by open (not kelp, not portal) edges, or None."""
    body = [(rng.randrange(m.w), rng.randrange(m.h))]
    d = rng.randrange(4)
    while len(body) < length:
        x, y = body[-1]
        for k in ([d] if rng.random() < 0.7 else []) + rng.sample(range(4), 4):
            e = m.edge_ahead(x, y, k)
            nxt = m.step(x, y, k)
            if nxt is None or e in m.portals or nxt in used or nxt in body:
                continue
            body.append(nxt)
            d = k
            break
        else:
            return None
    return body if body[0] not in used else None


def add_dragons(rng, m):
    per_team = rng.choice((1, 2, 2, 3, 3, 4))
    length = rng.choice((3, 4, 4, 5, 6))
    apart = max(3, min(m.w, m.h) // 4)
    used, teams = set(), []
    for _ in range(per_team):
        for _ in range(200):
            a = grow_body(rng, m, length, used)
            if a is None:
                continue
            b = [m.mirror_tile(*c) for c in a]
            cells = set(a) | set(b)
            if len(cells) < 2 * length or cells & used:
                continue
            if any(torus_dist(m, p, q) < apart for p in a for q in b):
                continue
            used |= cells
            teams.append((a, b))
            break
        else:
            return False
    m.dragons = [(t, body) for a, b in teams for t, body in ((0, a), (1, b))]
    return True


# ---------------------------------------------------------------------- entry points
def generate(seed, min_side=MIN_SIDE, max_side=MAX_SIDE):
    """The map for `seed`; the same arguments always give the same map."""
    for attempt in range(100):
        rng = random.Random(f"battlecode-map/{seed}/{attempt}")
        w, h = pick_size(rng, min_side, max_side)
        m = GameMap(w, h, f"Gen {seed}", rng.choice(SYMMETRIES))
        style = add_walls(rng, m)
        add_pearls(rng, m)
        open_doors(rng, m)
        add_portals(rng, m)
        if add_dragons(rng, m) and not m.problems(connected=True):
            m.style = style
            return m
    raise RuntimeError(f"no valid map for seed {seed}")


def describe(m):
    gaps = sorted(set(m.tiles.values()))
    spawning = sum(hi > 0 for _, hi in m.tiles.values()) / len(m.tiles)
    return (f"{m.name}: {m.w}x{m.h} sym={m.sym:2s} {getattr(m, 'style', '?'):7s} kelp {len(m.kelp) / (2 * m.w * m.h):4.0%} "
            f"portals {len(m.portals) // 2} dragons {len(m.dragons) // 2}x{len(m.dragons[0][1])} "
            f"pearl tiles {spawning:4.0%} gaps {len(gaps)} kinds")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0, help="first seed; map i uses seed+i")
    ap.add_argument("--out", default=str(pathlib.Path(__file__).resolve().parents[1] / "maps" / "gen"))
    ap.add_argument("--min-side", type=int, default=MIN_SIDE)
    ap.add_argument("--max-side", type=int, default=MAX_SIDE)
    ap.add_argument("--preview", type=int, default=0, help="print this many maps as ASCII instead of writing files")
    args = ap.parse_args()
    if args.preview:
        for s in range(args.seed, args.seed + args.preview):
            m = generate(s, args.min_side, args.max_side)
            print(describe(m))
            print(m.render(), "\n")
        return
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for s in range(args.seed, args.seed + args.count):
        m = generate(s, args.min_side, args.max_side)
        (out / f"gen_{s:05d}.map").write_text(m.dumps(), newline="\n")
    print(f"wrote {args.count} maps to {out}")


if __name__ == "__main__":
    main()
