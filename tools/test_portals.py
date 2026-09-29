"""Portal support in bot/brain.py (DEFAULTS "portals"): pair memory, link geometry, legality, flood fill, exploring.

Link geometry is checked against tools/mapfile.py GameMap.step, whose portal exits tools/test_mapfile.py checks
against the engine itself.

Run:  .venv\\Scripts\\python.exe tools\\test_portals.py
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bot"))
sys.path.insert(0, str(ROOT / "tools"))
from brain import Brain, DEAD, MOVES, OK, PORTAL  # noqa: E402
from mapfile import NORTH, WEST, GameMap  # noqa: E402
from proto import DX, DY, LETTERS  # noqa: E402

W, H = 40, 30


def block(hx, hy, dir_, edges=None, parts=None, pearls=(), cds=None, rnd=7, length=5, units=1):
    """A turn block with the head at (hx, hy). edges: {(x, y, 'h'|'v'): token} in absolute tile coordinates ('h' is
    the tile's north edge, 'v' its west edge); parts: [(team, id, x, y, facing, is_head)] absolute; pearls: absolute
    (x, y); cds: {(x, y): countdown}, default -1."""
    edges, parts, cds = edges or {}, parts or [], cds or {}
    lines = [f"ROUND {rnd}", f"DIR {LETTERS[dir_:dir_ + 1].decode()}", f"LENGTH {length}", f"UNIT_COUNT {units}",
             "NUM_MSGS 0"]
    for r in range(7):
        for c in range(7):
            x, y = (hx + c - 3) % W, (hy + r - 3) % H
            lines.append(f"{x} {y} {1 if (x, y) in pearls else 0} {cds.get((x, y), -1)}")
    lines.append(f"NUM_PARTS {len(parts)}")
    lines += [" ".join(str(v) for v in q) for q in parts]
    for r in range(8):
        lines.append(" ".join(edges.get(((hx + c - 3) % W, (hy + r - 3) % H, "h"), ".") for c in range(7)))
    for r in range(7):
        lines.append(" ".join(edges.get(((hx + c - 3) % W, (hy + r - 3) % H, "v"), ".") for c in range(8)))
    return ("\n".join(lines) + "\n\n").encode()


def brain(**params):
    b = Brain(0, b"A", W, H, 64, params={"portals": 1, **params})
    b.strict = b.debug = True
    return b


def me(x, y, dir_, length=2):
    """Our own parts: head at (x, y) facing dir_, the body straight behind it."""
    out = [("A", 0, x, y, LETTERS[dir_:dir_ + 1].decode(), 1)]
    for i in range(1, length):
        out.append(("A", 0, (x - DX[dir_] * i) % W, (y - DY[dir_] * i) % H, LETTERS[dir_:dir_ + 1].decode(), 0))
    return out


def test_link_geometry_matches_mapfile():
    """Every crossing of a pair lands where GameMap.step says, for west and north edges, across the wrap too."""
    pairs = [((21, 15, WEST), (5, 3, WEST)), ((0, 7, WEST), (33, 29, WEST)),
             ((10, 0, NORTH), (30, 12, NORTH)), ((4, 9, NORTH), (39, 0, NORTH))]
    for e1, e2 in pairs:
        m = GameMap(W, H, portals={e1: 0, e2: 0})
        b = brain()
        key = lambda e: (e[1] * W + e[0]) * 2 + (1 if e[2] == WEST else 0)  # noqa: E731
        b.note_portal(0, key(e1))
        b.note_portal(0, key(e1))  # seeing the same edge again completes nothing
        assert not b.link_dst
        b.note_portal(0, key(e2))
        n = 0
        for y in range(H):
            for x in range(W):
                for d in range(4):
                    if m.edge_ahead(x, y, d) in m.portals:
                        tx, ty = m.step(x, y, d)
                        assert b.link_dst.get((y * W + x) * 4 + d) == ty * W + tx, (e1, e2, x, y, d)
                        n += 1
        assert n == 4 and len(b.link_dst) == 4, (n, b.link_dst)
    print("ok: the four crossings of a pair land where mapfile.GameMap.step says, on both orientations")


def test_off_is_walls():
    """portals=0: a portal is still the old last-resort PORTAL status and ids are not even recorded."""
    b = Brain(0, b"A", W, H, 64, params={"portals": 0})
    b.strict = b.debug = True
    act = b.act(block(20, 15, 1, edges={(21, 15, "v"): "3"}, parts=me(20, 15, 1)))
    assert b.dbg["status"][1] == PORTAL and act != MOVES[1] and not b.pids, (act, b.dbg["status"])
    print("ok: with portals off a portal is a wall, as before")


def test_unknown_portal_is_explored_but_never_backwards():
    edges = {(21, 15, "v"): "3", (20, 16, "h"): "4"}  # portals on the head's east and south edges
    # facing north: south (behind us) is a portal we have never seen the far side of -- stepping back lands on the neck
    b = brain(portal_unknown=1e6)
    act = b.act(block(20, 15, 0, edges={(20, 16, "h"): "4"}, parts=me(20, 15, 0)[:1]))
    assert act != MOVES[2] and b.dbg["unknown"] == [], (act, b.dbg)
    b = brain(portal_unknown=1e6)
    act = b.act(block(20, 15, 0, edges=edges, parts=me(20, 15, 0)[:1]))
    assert act == MOVES[1] and b.dbg["unknown"] == [1] and b.dbg["risky"], (act, b.dbg)
    assert b.exploring
    # portal_hungry: with a pearl in view (three steps west) the unknown portal is no longer a bet worth taking
    b = brain(portal_unknown=1e6, portal_hungry=1)
    act = b.act(block(20, 15, 0, edges=edges, parts=me(20, 15, 0)[:1], pearls={(17, 15)}))
    assert act != MOVES[1] and not b.exploring, (act, b.dbg)
    b = brain(portal_unknown=1e6, portal_hungry=1)
    assert b.act(block(20, 15, 0, edges=edges, parts=me(20, 15, 0)[:1])) == MOVES[1]
    print("ok: an unknown portal is a candidate (exploring), except straight back against our facing")


def test_learning_stops_exploring_after_an_empty_pocket():
    """Autarky-style: the unknown portal leads into a closed 3x3 box with no pearl spawns -> stop exploring, even
    though the view reaches spawn tiles in the field beyond the box's walls. Landing on open ground with no spawn in
    view (a Portals child leaving a pearl room) or in a box with a spawn tile of its own does not."""
    box = {(x, 2, "h"): "w" for x in (4, 5, 6)} | {(x, 5, "h"): "w" for x in (4, 5, 6)}
    box |= {(4, y, "v"): "w" for y in (2, 3, 4)} | {(7, y, "v"): "w" for y in (2, 3, 4)}
    box[(5, 3, "v")] = "3"  # the partner, inside the box as on Autarky
    field = {(2, 1): 5, (8, 5): 9}  # spawn tiles in view, outside the box
    for far, cds, stops in ((box, {}, True), (box, field, True), ({(5, 3, "v"): "3"}, {}, False),
                            (box, {(6, 4): 4}, False)):
        b = brain(portal_unknown=1e6)
        b.act(block(20, 15, 1, edges={(21, 15, "v"): "3"}, parts=me(20, 15, 1)[:1], rnd=7))
        assert b.exploring
        b.act(block(5, 3, 1, edges=far, parts=me(5, 3, 1)[:1], cds=cds, rnd=8))  # landed on (5, 3) facing east
        assert b.explore_ok != stops and b.link_dst, (far is box, cds, b.explore_ok)
    print("ok: an empty walled-in pocket beyond an unknown portal stops exploring; open ground or a spawn tile does not")


def test_known_portal_lands_beyond_partner():
    """Pair known: the east portal lands on (5, 3). A remembered pearl there pulls the dragon through; our own body
    there (out of view) makes it fatal; an enemy head seen there in view makes it a trade, not a free move."""
    b = brain(portal_unknown=-1e9)
    b.note_portal(3, (15 * W + 21) * 2 + 1)
    b.note_portal(3, (3 * W + 5) * 2 + 1)
    b.pmem |= 1 << (3 * W + 5)  # a pearl seen at (5, 3) earlier, on a visit that also showed the room around it
    b.seen = b.full
    act = b.act(block(20, 15, 0, edges={(21, 15, "v"): "3"}, parts=me(20, 15, 0)))
    assert b.dbg["dest"][1] == 3 * W + 5 and b.dbg["blind"][1] and b.dbg["status"][1] == OK, b.dbg
    assert act == MOVES[1], act

    b = brain()
    b.note_portal(3, (15 * W + 21) * 2 + 1)
    b.note_portal(3, (3 * W + 5) * 2 + 1)
    b.body = [15 * W + 20, 3 * W + 5]  # our neck is on the far side: we came through it
    b.own = (1 << (15 * W + 20)) | (1 << (3 * W + 5))
    b.act(block(20, 15, 3, edges={(21, 15, "v"): "3"}, parts=me(20, 15, 3)[:1], length=2))
    assert b.dbg["status"][1] == DEAD, b.dbg["status"]

    b = brain()  # both ends in one window: the landing tile (23, 15) is in view, with an enemy head on it
    b.note_portal(3, (15 * W + 21) * 2 + 1)
    b.note_portal(3, (15 * W + 23) * 2 + 1)
    b.act(block(20, 15, 0, edges={(21, 15, "v"): "3", (23, 15, "v"): "3"},
                parts=me(20, 15, 0) + [("B", 1, 23, 15, "N", 1)]))
    assert b.dbg["dest"][1] == 15 * W + 23 and not b.dbg["blind"][1] and b.dbg["status"][1] == 2, b.dbg
    print("ok: a known portal lands beyond its partner: remembered pearls, own body and visible heads all count")


def test_flood_crosses_known_links():
    """A 1x2 pocket walled in by kelp, whose only way out is a known portal: the flood fill sees the room beyond."""
    b = brain()
    cell = lambda x, y: y * W + x  # noqa: E731
    # pocket (20, 15)-(21, 15): kelp all round except a portal on (21, 15)'s east edge = west edge of (22, 15)
    b.kh = (1 << cell(20, 15)) | (1 << cell(21, 15)) | (1 << cell(20, 16)) | (1 << cell(21, 16))
    b.kv = 1 << cell(20, 15)
    b.pv = 1 << cell(22, 15)
    b.okh = b.full ^ (b.kh | b.ph)
    b.okv = b.full ^ (b.kv | b.pv)
    assert b.flood(b.full, cell(20, 15), 100) == 2
    b.note_portal(8, cell(22, 15) * 2 + 1)
    b.note_portal(8, cell(5, 5) * 2 + 1)
    assert b.flood(b.full, cell(20, 15), 100) >= 100
    assert b.exits(cell(21, 15), 21, 15, 3, b.full) == 1  # the portal is its only way on
    print("ok: flood fill and dead-end exits go through known portal links")


def map_block(m, hx, hy, dir_=0, rnd=0):
    """The first turn block a dragon with its head on (hx, hy) of GameMap m would get (terrain only, no parts)."""
    edges = {}
    for (x, y, d) in m.kelp:
        edges[(x, y, "v" if d == WEST else "h")] = "w"
    for (x, y, d), pid in m.portals.items():
        edges[(x, y, "v" if d == WEST else "h")] = str(pid)
    global W, H
    W, H = m.w, m.h
    try:
        return block(hx, hy, dir_, edges=edges, parts=[("A", 0, hx, hy, "N", 1)], rnd=rnd)
    finally:
        W, H = 40, 30


def test_known_map_recognised_from_the_first_view():
    """Every founder of every ladder map recognises exactly its own map on its first turn (or waits, never guesses),
    loads all its portal pairs, and a map with one kelp edge moved is not mistaken for the original."""
    import ladder_maps
    import known_maps
    names = {c[0] for cs in known_maps.MAPS.values() for c in cs}
    resolved = total = 0
    for name in sorted(names):
        m = ladder_maps.original(name)
        for team, body in m.dragons:
            b = Brain(0, b"A", m.w, m.h, 64, params={"portals": 1, "map_oracle": 1})
            b.act(map_block(m, *body[0]))
            assert b.oracle in (name, None), (name, body[0], b.oracle)
            if b.oracle:
                assert len(b.link_dst) == 2 * len(m.portals), (name, len(b.link_dst))
            resolved += b.oracle == name
            total += 1
    print(f"   {resolved} of {total} founders knew their map on turn 0, the rest were still deciding")
    m = GameMap.loads(ladder_maps.original("trauma").dumps())  # a copy: original() is cached
    hx, hy = m.dragons[0][1][0]
    m.kelp.add((hx, hy, NORTH))  # one kelp edge the real map does not have, right by the head
    b = Brain(0, b"A", m.w, m.h, 64, params={"portals": 1, "map_oracle": 1})
    b.act(map_block(m, hx, hy))
    assert b.oracle is False, b.oracle
    print("ok: founders recognise their own ladder map from the first view; an altered map is not recognised")


if __name__ == "__main__":
    test_link_geometry_matches_mapfile()
    test_off_is_walls()
    test_unknown_portal_is_explored_but_never_backwards()
    test_learning_stops_exploring_after_an_empty_pocket()
    test_known_portal_lands_beyond_partner()
    test_flood_crosses_known_links()
    test_known_map_recognised_from_the_first_view()
