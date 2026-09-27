"""Many distinct games from the ladder's own maps (maps/ladder/, extracted by tools/replay_maps.py).

The engine takes no seed, and brain-vs-brain play is deterministic, so a map gives exactly one game per seat: ten
ladder maps would be only 20 games per opponent. A variant keeps a map's walls, portals and spawns but changes what
the engine's pearl randomness sees:

    variant 0        the map exactly as the ladder plays it (held out: never used for tuning)
    variant v > 0    mirror (v % 4: none, left-right, top-bottom, both) and, for v >= 4, every tile's pearl gaps
                     scaled by a factor in [0.9, 1.1] (the same factor for a tile and its mirror image, so the map
                     stays symmetric)

Mirroring alone changes which tile receives which random pearl draw (the engine ticks tiles row by row), so all four
orientations are different games. Variants from HOLDOUT up are reserved for accept/reject decisions.

A league map spec for a variant is ("ladder", name, v); league.map_bytes() builds it.
"""
import functools
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from mapfile import NORTH, GameMap  # noqa: E402

LADDER = pathlib.Path(__file__).resolve().parents[1] / "maps" / "ladder"
# The ten maps in the ladder's rotation as of 2026-09-27 (live bot's per-map record). Big Empty, Default Small and
# Stronghold showed one game each (an older pool); Stronghold has not been recovered from any replay.
POOL = ("autarky", "default", "devil", "portals", "prisoners_dilemma", "queen_of_spades", "schooltime",
        "slithery_fight", "trauma", "trophy")
HOLDOUT = 100_000


@functools.lru_cache(maxsize=None)
def original(name):
    return GameMap.loads((LADDER / f"{name}.map").read_text())


def flip(m, fx, fy):
    """m mirrored left-right (fx) and/or top-bottom (fy). Edge (x, y, NORTH) lies between rows y-1 and y, so a
    top-bottom flip moves it between rows h-y and h-1-y, i.e. to the north edge of row (h - y) % h; west edges
    likewise move to column (w - x) % w under a left-right flip."""
    w, h = m.w, m.h

    def tile(x, y):
        return (w - 1 - x if fx else x), (h - 1 - y if fy else y)

    def edge(e):
        x, y, d = e
        if d == NORTH:
            return (w - 1 - x if fx else x), ((h - y) % h if fy else y), d
        return ((w - x) % w if fx else x), (h - 1 - y if fy else y), d

    return GameMap(w, h, m.name, m.sym,
                   tiles={tile(x, y): gaps for (x, y), gaps in m.tiles.items()},
                   kelp={edge(e) for e in m.kelp},
                   portals={edge(e): pid for e, pid in m.portals.items()},
                   dragons=[(team, [tile(x, y) for x, y in body]) for team, body in m.dragons])


def jitter(m, seed, spread=0.1):
    """Every spawning tile's (min_gap, max_gap) scaled by one factor in [1 - spread, 1 + spread], shared with its
    mirror tile; tiles that never spawn (max_gap 0) are left alone."""
    tiles = {}
    for (x, y), (lo, hi) in m.tiles.items():
        if hi <= 0:
            tiles[(x, y)] = (lo, hi)
            continue
        key = min((x, y), m.mirror_tile(x, y))
        f = random.Random(f"ladder-jitter/{seed}/{key}").uniform(1 - spread, 1 + spread)
        nlo = max(1, round(lo * f)) if lo > 0 else lo
        tiles[(x, y)] = (nlo, max(nlo, round(hi * f)))
    return GameMap(m.w, m.h, m.name, m.sym, tiles=tiles, kelp=set(m.kelp), portals=dict(m.portals),
                   dragons=list(m.dragons))


def variant(name, v):
    m = original(name)
    if v == 0:
        return m
    out = flip(m, v % 4 in (1, 3), v % 4 in (2, 3))
    return jitter(out, v) if v >= 4 else out


def train_specs(n, generation=0):
    """n variants of every pool map for one tuning generation: fresh each generation, never variant 0 or a
    holdout variant."""
    return [("ladder", name, 1 + (generation * n + i) % (HOLDOUT - 1)) for name in POOL for i in range(n)]


def holdout_specs(n):
    """The pool maps as played on the ladder, plus n reserved variants of each: the decision set."""
    return [("ladder", name, v) for name in POOL for v in [0] + [HOLDOUT + i for i in range(n)]]
