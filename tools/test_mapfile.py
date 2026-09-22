"""Checks tools/mapfile.py against the bundled maps and against the real engine.

Run:  .venv\\Scripts\\python.exe tools\\test_mapfile.py

The engine probes are what pin the format down: a scripted dragon walks into a known edge and the engine reports where
and when it dies or lands, so a wrong reading of an edge index shows up as a wrong death round or position.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bot"))
sys.path.insert(0, str(ROOT / "tools"))
import mapfile  # noqa: E402
import opponents  # noqa: E402
import proto  # noqa: E402
from mapfile import GameMap, NORTH, WEST  # noqa: E402
from unswbc.engine import EngineModule  # noqa: E402

MAPS = sorted((ROOT / "maps").glob("*.map"))
MOVES = (b"MOVE N\n", b"MOVE E\n", b"MOVE S\n", b"MOVE W\n")
FLIP = {"xy": {0: 2, 1: 3, 2: 0, 3: 1}, "y": {0: 0, 1: 3, 2: 2, 3: 1}, "x": {0: 2, 1: 1, 2: 0, 3: 3}}


def test_bundled_maps_are_valid():
    """Every bundled symmetric map is closed under its own mirror: kelp, portal pairs, tile gaps and dragons."""
    for p in MAPS:
        m = GameMap.loads(p.read_text())
        if not m.sym:
            continue  # arena.map declares no symmetry
        assert m.problems() == [], f"{p.stem}: {m.problems()}"
    print("ok  bundled maps are symmetric under the decoded edge layout")


def test_roundtrip():
    """dumps(loads(text)) keeps the map; maps that list exactly 2*w*h edges come back byte for byte."""
    for p in MAPS:
        text = p.read_text().replace("\r\n", "\n")
        m = GameMap.loads(text)
        again = GameMap.loads(m.dumps())
        assert (again.w, again.h, again.sym, again.tiles, again.kelp, again.portals, again.dragons) == \
               (m.w, m.h, m.sym, m.tiles, m.kelp, m.portals, m.dragons), f"{p.stem} changed in a round trip"
        if f"EDGE_COUNT {2 * m.w * m.h}\n" in text:
            assert m.dumps() == text, f"{p.stem} is not byte-identical"
    print("ok  parse/write round trip")


def play(engine, data, pa, pb):
    owner, deaths = {}, []

    def spawn(did, init):
        owner[did] = pa if init.split(b"TEAM ")[1][:1] == b"A" else pb
        owner[did].spawn(did, init)

    res = engine.run(data, lambda did, block: owner[did].reply(did, block),
                     lambda did, rnd, why: deaths.append((owner[did].name, rnd, why)), spawn, lambda line: None, 0)
    return res, deaths


def test_engine_plays_rewritten_maps_identically(engine):
    """The same deterministic match on the original file and on the rewritten one ends the same way."""
    for p in MAPS:
        text = p.read_text()
        for a, b in (("chaser", "greedy"), ("hugger", "rammer")):
            results = []
            for data in (text, GameMap.loads(text).dumps()):
                res, deaths = play(engine, data.encode(), opponents.OPPONENTS[a](seed=0), opponents.OPPONENTS[b](seed=1))
                results.append((res.rounds, res.winner, res.a_length, res.b_length, deaths))
            assert results[0] == results[1], f"{p.stem} {a} vs {b}: rewriting the map changed the game"
    print("ok  the engine plays rewritten maps identically")


class Script:
    """Team A follows `moves` (one direction per round), team B the mirrored moves; afterwards both send no action."""

    def __init__(self, name, moves):
        self.name, self.moves, self.pos = name, moves, {}

    def spawn(self, did, init):
        pass

    def reply(self, did, block):
        t = proto.parse_turn(block)
        self.pos[t.rnd] = (t.hx, t.hy)
        return MOVES[self.moves[t.rnd]] if t.rnd < len(self.moves) else b""


def probe(engine, m, moves):
    """Runs both teams along mirrored scripts; returns (team A head position per round, team A death (round, reason))."""
    flip = FLIP[m.sym]
    a, b = Script("A", moves), Script("B", [flip[d] for d in moves])
    _, deaths = play(engine, m.dumps().encode(), a, b)
    mine = [(rnd, why) for name, rnd, why in deaths if name == "A"]
    return a.pos, (mine[0] if mine else None)


def blank(sym, dragon_a, w=12, h=12):
    m = GameMap(w, h, "probe", sym, tiles={(x, y): (1, 1000) for x in range(w) for y in range(h)})
    m.dragons = [(0, dragon_a), (1, [m.mirror_tile(*c) for c in dragon_a])]
    return m


def add_kelp(m, *edges):
    for e in edges:
        m.kelp.update({e, m.mirror_edge(e)})


def test_engine_kelp_edges(engine):
    """North edge of (x, y) lies between (x, y-1) and (x, y); west edge between (x-1, y) and (x, y)."""
    engine_cases = [
        # (kelp edge, head-first body, moves, expected death round)
        ((6, 3, NORTH), [(6, 5), (6, 6), (6, 7)], [0, 0, 0], 2),   # N,N: (6,4) then (6,3), then across the edge
        ((6, 4, NORTH), [(6, 5), (6, 6), (6, 7)], [0, 0, 0], 1),
        ((6, 3, NORTH), [(6, 1), (6, 0), (6, 11)], [2, 2, 2], 1),  # southwards: (6,2), then across the same edge
        ((6, 6, WEST), [(4, 6), (3, 6), (2, 6)], [1, 1, 1], 1),    # E: (5,6), then across the west edge of (6,6)
        ((6, 6, WEST), [(8, 6), (9, 6), (10, 6)], [3, 3, 3], 2),   # W: (7,6), (6,6), then across the same edge
    ]
    for edge, body, moves, want in engine_cases:
        m = blank("xy", body)
        add_kelp(m, edge)
        assert m.problems() == [], m.problems()
        _, death = probe(engine, m, moves)
        assert death == (want, "W"), f"kelp {edge} moves {moves}: expected death by kelp in round {want}, got {death}"
    print("ok  kelp on north and west edges blocks exactly where decoded")


def test_engine_portal_exit(engine):
    """Crossing a portal moving east / south lands on the partner's own tile; moving west / north on the tile before it."""
    m = blank("xy", [(1, 5), (0, 5), (11, 5)])
    p, q = (2, 5, WEST), (8, 9, WEST)
    for e, other, pid in ((p, q, 0), (m.mirror_edge(p), m.mirror_edge(q), 1)):
        m.portals[e] = m.portals[other] = pid
    assert m.problems() == [], m.problems()
    pos, death = probe(engine, m, [1, 1])  # east across p
    assert pos[1] == (8, 9), f"east through a west-edge portal: expected to land on (8, 9), got {pos}"
    assert m.step(1, 5, 1) == (8, 9)

    m = blank("xy", [(8, 9), (9, 9), (10, 9)])
    m.portals.update({p: 0, q: 0, m.mirror_edge(p): 1, m.mirror_edge(q): 1})
    pos, death = probe(engine, m, [3, 3])  # west across q, back out of p on its west side
    assert pos[1] == (1, 5), f"west through a west-edge portal: expected (1, 5), got {pos}"
    assert m.step(8, 9, 3) == (1, 5)

    m = blank("xy", [(3, 2), (3, 1), (3, 0)])
    p, q = (3, 4, NORTH), (9, 8, NORTH)
    m.portals.update({p: 0, q: 0, m.mirror_edge(p): 1, m.mirror_edge(q): 1})
    assert m.problems() == [], m.problems()
    pos, _ = probe(engine, m, [2, 2, 2])  # south: (3,3), then across p (north edge of (3,4)) to q's own tile
    assert pos[2] == (9, 8), f"south through a north-edge portal: expected (9, 8), got {pos}"
    assert m.step(3, 3, 2) == (9, 8)
    print("ok  portals exit where decoded, in both directions and on both orientations")


def test_engine_symmetry_names(engine):
    """SYMMETRY y mirrors x, x mirrors y, xy rotates: the mirrored tiles share one countdown at round 0."""
    want = {"y": lambda x, y: (11 - x, y), "x": lambda x, y: (x, 11 - y), "xy": lambda x, y: (11 - x, 11 - y)}
    for sym, mirror in want.items():
        seen = {}

        class Look:
            name = "A"

            def spawn(self, did, init):
                pass

            def reply(self, did, block):
                t = proto.parse_turn(block)
                if t.rnd == 0 and did == 0:
                    for i, c in enumerate(t.cds):
                        seen[((t.hx - 3 + i % 7) % 12, (t.hy - 3 + i // 7) % 12)] = int(c)
                return b""

        class Idle(Look):
            name = "B"

            def reply(self, did, block):
                return b""

        m = blank(sym, [(5, 5), (4, 5), (3, 5)] if sym != "x" else [(5, 5), (5, 4), (5, 3)])
        play(engine, m.dumps().encode(), Look(), Idle())
        shared = [(c, mirror(*c)) for c in seen if mirror(*c) in seen and mirror(*c) != c]
        assert len(shared) > 10, f"{sym}: too few mirrored pairs in view to test ({len(shared)})"
        agree = sum(seen[a] == seen[b] for a, b in shared)
        assert agree == len(shared), f"SYMMETRY {sym}: only {agree}/{len(shared)} mirrored tiles share a countdown"
        for other, om in want.items():
            if other != sym:
                pairs = [(c, om(*c)) for c in seen if om(*c) in seen and om(*c) != c]
                equal = sum(seen[a] == seen[b] for a, b in pairs)
                assert equal < len(pairs) // 2, f"SYMMETRY {sym} also mirrors like {other}"
    print("ok  SYMMETRY y / x / xy mirror x / y / both, as decoded")


def run():
    engine = EngineModule()
    test_bundled_maps_are_valid()
    test_roundtrip()
    test_engine_plays_rewritten_maps_identically(engine)
    test_engine_kelp_edges(engine)
    test_engine_portal_exit(engine)
    test_engine_symmetry_names(engine)


if __name__ == "__main__":
    run()
