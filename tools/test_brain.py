"""Checks bot/brain.py's Brain.fallback() -- the emergency, learned-state-independent path taken when decide()
raises. Unlike decide()'s own legality check (which has its own PORTAL status specifically to avoid this), fallback()
re-implements legality from scratch and used to miss the portal case; see its docstring.

Run:  .venv\\Scripts\\python.exe tools\\test_brain.py
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bot"))
sys.path.insert(0, str(ROOT / "tools"))
from brain import Brain, MOVES  # noqa: E402
from test_encoder import H, W, render_block  # noqa: E402


def test_fallback_portal_edge_ignores_irrelevant_landing_tile_occupancy():
    # East edge (offset (1, 0)'s west edge, i.e. the head's own east edge) is a portal: token "7", any string
    # that's neither '.' nor 'w'. A portal step does not land on the physically-adjacent tile east of the head
    # (it exits the portal's paired edge instead, see brain.py's module docstring) -- an unrelated body sitting
    # on that irrelevant tile must not rule the direction out.
    edges = {(1, 0, "v"): "7", (0, 0, "h"): "w"}  # kelp to the north forces fallback() past its first choice
    parts = [("B", 2, 1, 0, "N", 0)]  # unrelated body on the (irrelevant) physically-adjacent tile east
    block = render_block(0, set(), {}, edges, parts, length=6)
    b = Brain(1, b"A", W, H, 8)
    action = b.fallback(block)
    assert action == MOVES[1], action  # MOVES[1] = b"MOVE E\n"
    print("ok: fallback() takes a portal move even when the physically-adjacent landing tile is occupied")


def test_fallback_kelp_still_blocks_and_normal_occupancy_still_blocks():
    # North kelp forces fallback() past its first (heading) choice in both cases below, so reaching South (still
    # open) actually demonstrates East was inspected and rejected -- not just skipped by ordering luck.
    edges = {(0, 0, "h"): "w", (1, 0, "v"): "w"}  # kelp both north and east -- East must still be excluded outright
    parts = [("B", 2, 1, 0, "N", 0)]
    block = render_block(0, set(), {}, edges, parts, length=6)
    b = Brain(1, b"A", W, H, 8)
    assert b.fallback(block) == MOVES[2], b.fallback(block)  # MOVES[2] = b"MOVE S\n"

    edges2 = {(0, 0, "h"): "w"}  # north kelp only; East is a plain edge with the same body occupying its tile
    block2 = render_block(0, set(), {}, edges2, parts, length=6)
    b2 = Brain(1, b"A", W, H, 8)
    assert b2.fallback(block2) == MOVES[2], b2.fallback(block2)
    print("ok: kelp still blocks outright, and a normal edge's occupied landing tile is still excluded")


def test_dive_takes_a_trapped_pearl_only_when_enabled_and_gated():
    # A 3-long dragon heading north; a pearl east of the head sits in a one-tile kelp pocket (kelp on the pocket's
    # north, south and east edges), so stepping onto it is a certain trap: the default Brain refuses it.
    edges = {(1, 0, "h"): "w", (1, 1, "h"): "w", (2, 0, "v"): "w"}
    parts = [("A", 1, 0, 0, "N", 1), ("A", 1, 0, 1, "N", 0), ("A", 1, 0, 2, "N", 0)]

    def act(params, cd=-1, units=5):
        block = render_block(0, {(1, 0)}, {(1, 0): cd}, edges, parts, length=3, units=units)
        return Brain(1, b"A", W, H, 64, params).act(block)

    dive = {"dive_len": 3, "dive_trap": 0.0, "dive_scope": 1}
    assert act({}) != MOVES[1], act({})
    assert act(dive) == MOVES[1], act(dive)
    assert act({**dive, "dive_len": 2}) != MOVES[1]  # too long to dive
    assert act({**dive, "dive_fountain": 1}, cd=7) != MOVES[1]  # not a fountain
    assert act({**dive, "dive_fountain": 1}, cd=1) == MOVES[1]  # a fountain always shows countdown 1
    assert act({**dive, "dive_units": 10}, units=5) != MOVES[1]  # team too small to spend a dragon
    assert act({**dive, "dive_units": 10}, units=10) == MOVES[1]
    print("ok: dive takes a trapped pearl only when enabled, short enough, on a fountain if asked, with team to spare")

    # a 5-long dragon relies on the boxed split to leave the pocket, so it dives only below the boxed split's cap
    # (64 minus the 4 slots boxed_reserve keeps for dragons shorter than boxed_long 8)
    long_parts = [("A", 1, 0, y, "N", 1 if y == 0 else 0) for y in range(5)]

    def act5(units):
        block = render_block(0, {(1, 0)}, {}, edges, long_parts, length=5, units=units)
        return Brain(1, b"A", W, H, 64, {**dive, "dive_len": 5}).act(block)

    assert act5(59) == MOVES[1], act5(59)
    assert act5(60) != MOVES[1], act5(60)
    print("ok: a long diver needs a legal boxed split to get back out")


def test_fountain_detection_needs_countdown_one_on_two_turns_running():
    # dive_scope 2 marks a tile as a fountain once it shows countdown 1 on consecutive turns: a (1,1) tile always
    # does, any other spawning tile resets to a fresh draw after hitting 1 (see the DEFAULTS comment)
    from test_encoder import HX, HY
    parts = [("A", 1, 0, 0, "N", 1), ("A", 1, 0, 1, "N", 0), ("A", 1, 0, 2, "N", 0)]
    b = Brain(1, b"A", W, H, 64, {"dive_len": 3, "dive_trap": 0.0, "dive_scope": 2})
    b.act(render_block(0, set(), {(2, -1): 1, (-2, 2): 1}, {}, parts, rnd=7, length=3))
    assert b.founts == [], b.founts
    b.act(render_block(0, set(), {(2, -1): 1, (-2, 2): 4}, {}, parts, rnd=8, length=3))
    assert b.founts == [((HX + 2) % W, (HY - 1) % H)], b.founts
    print("ok: a tile is a fountain only after countdown 1 on two turns running")


def test_boxed_r_end_keeps_only_rear_splits_late():
    # boxed in: kelp north, west and east of the head, own body south
    edges = {(0, 0, "h"): "w", (0, 0, "v"): "w", (1, 0, "v"): "w"}

    def act(params, length, rnd):
        parts = [("A", 1, 0, y, "N", 1 if y == 0 else 0) for y in range(length)]
        return Brain(1, b"A", W, H, 64, params).act(render_block(0, set(), {}, edges, parts, rnd=rnd, length=length))

    assert act({}, 4, 350) == b"SPLIT 2\n", act({}, 4, 350)  # mh3: a boxed split at any round
    assert act({}, 5, 350) == b"SPLIT 2\n"  # ... and a rear split only from round 433
    assert act({"boxed_rear_r": 0}, 5, 350) == b"SPLIT 3\n"
    late = {"boxed_r_end": 300, "boxed_rear_r": 0}
    assert act(late, 4, 299) == b"SPLIT 2\n"  # before boxed_r_end: unchanged
    assert not act(late, 4, 300).startswith(b"SPLIT")  # a short dragon boxed in late dies where it is
    assert act(late, 5, 300) == b"SPLIT 3\n"  # a rear split keeps the body, so it still happens
    print("ok: from boxed_r_end on a boxed-in dragon splits only for a rear split")


def test_ram_takes_a_head_on_trade_only_when_enabled_and_even():
    def act(params, length=2, enemy_len=3):
        parts = [("A", 1, 0, y, "N", 1 if y == 0 else 0) for y in range(length)]
        parts += [("B", 2, 1 + i, 0, "W", 1 if i == 0 else 0) for i in range(enemy_len)]  # enemy head east of ours
        return Brain(1, b"A", W, H, 64, params).act(render_block(0, set(), {}, {}, parts, length=length))

    ram = {"ram_len": 3, "ram": 1000.0}
    assert act({}) != MOVES[1], act({})  # mh3 never moves onto a head
    assert act(ram) == MOVES[1], act(ram)
    assert act(ram, enemy_len=1) != MOVES[1]  # it shows fewer segments than we have: not an even trade
    assert act(ram, length=4) != MOVES[1]  # too long to throw away
    assert act({**ram, "ram": 10.0}) != MOVES[1]  # a better move outscores a weak ram
    print("ok: a short dragon rams an adjacent enemy head at least as long, only when enabled")


def test_radar_parses_echoes_pings_ahead_and_guards_straight_dives():
    import proto
    # a 3-long dragon heading north; the pearl straight ahead sits in a one-tile kelp pocket (a certain trap)
    edges = {(0, -1, "h"): "w", (0, -1, "v"): "w", (1, -1, "v"): "w"}
    parts = [("A", 1, 0, 0, "N", 1), ("A", 1, 0, 1, "N", 0), ("A", 1, 0, 2, "N", 0)]

    def block(echo=None):
        b = render_block(0, {(0, -1)}, {}, edges, parts, length=3, units=5)
        if echo is None:
            return b
        ls = b.split(b"\n")
        return b"\n".join(ls[:5] + [b"ECHOES " + b" ".join(b"%d" % v for v in echo)] + ls[5:])

    t0, t1 = proto.parse_turn(block()), proto.parse_turn(block((1, 0, 0, 0, 0)))
    assert t0.echo is None and t1.echo == (1, 0, 0, 0, 0)
    assert (t1.hx, t1.hy, t1.flags, t1.cds, t1.parts, t1.edges) == (t0.hx, t0.hy, t0.flags, t0.cds, t0.parts, t0.edges)

    dive = {"dive_len": 3, "dive_trap": 0.0, "dive_scope": 1, "radar": 1, "radar_dive": 1}
    out = Brain(1, b"A", W, H, 64, dive).act(block((1, 0, 0, 0, 0)))  # the ray ahead hit kelp: the pocket is empty
    assert out == MOVES[0] + b"PROTOCOL 3\nSONAR N 0\n", out
    out = Brain(1, b"A", W, H, 64, dive).act(block((0, 0, 0, 1, 0)))  # it hit an enemy body: no dive
    assert not out.startswith(MOVES[0]) and out.endswith(b"0\n") and b"PROTOCOL 3\nSONAR " in out, out
    assert Brain(1, b"A", W, H, 64, {}).act(block()) != MOVES[0]  # radar off: no dive, no PROTOCOL line
    assert b"PROTOCOL" not in Brain(1, b"A", W, H, 64, {}).act(block())
    print("ok: radar parses ECHOES, pings along the move and keeps a dive out of an occupied line ahead")


def test_feed_dies_next_to_a_long_teammate_only_when_enabled_and_safe():
    feed = {"feed_r": 250, "feed_len": 8, "feed_dist": 3, "feed_min": 8, "feed_ratio": 1.5}
    me = [("A", 1, 0, 0, "N", 1), ("A", 1, 0, 1, "N", 0), ("A", 1, 0, 2, "N", 0)]

    def mate(dx, dy, n, pid):  # a teammate n long, head at (dx, dy), body trailing east
        return [("A", pid, dx + i, dy, "W", 1 if i == 0 else 0) for i in range(n)]

    def act(params, parts, rnd=300):
        return Brain(1, b"A", W, H, 64, params).act(render_block(0, set(), {}, {}, parts, rnd=rnd, length=3))

    near = me + mate(-3, -3, 3, 7) + mate(1, -2, 8, 9)  # teammate 9: head 3 away, 8 segments in view
    assert act({}, near) != b"SPLIT 1\n"  # off by default
    assert act(feed, near) == b"SPLIT 1\n", act(feed, near)
    assert act(feed, near, rnd=249) != b"SPLIT 1\n"  # before feed_r
    assert act({**feed, "feed_dist": 2}, near) != b"SPLIT 1\n"  # out of range
    assert act({**feed, "feed_min": 9}, near) != b"SPLIT 1\n"  # not long enough to feed
    assert act(feed, near + [("B", 20, -2, 1, "E", 1)]) != b"SPLIT 1\n"  # an enemy head as close would eat first
    print("ok: a short dragon feeds a long teammate within range, only when enabled, late, and with no enemy near")


def test_split_enemy_dist_only_blocks_splits_near_an_enemy_head():
    # a 4-long split child with a pearl in view (so it wants to split) and an enemy head 3 away
    parts = [("A", 1, 0, y, "N", 1 if y == 0 else 0) for y in range(4)] + [("B", 20, 3, 0, "S", 1), ("B", 20, 3, -1, "S", 0)]

    def act(params):
        return Brain(1, b"A", W, H, 64, params).act(render_block(0, {(-2, -2)}, {}, {}, parts, length=4, units=5))

    assert not act({}).startswith(b"SPLIT"), act({})  # default: any enemy head in view blocks it
    assert act({"split_enemy_dist": 0}) == b"SPLIT 2\n", act({"split_enemy_dist": 0})
    assert act({"split_enemy_dist": 2}) == b"SPLIT 2\n"  # the head is 3 away
    assert not act({"split_enemy_dist": 3}).startswith(b"SPLIT")
    print("ok: split_enemy_dist blocks a split only when an enemy head is that close")


if __name__ == "__main__":
    test_fallback_portal_edge_ignores_irrelevant_landing_tile_occupancy()
    test_fallback_kelp_still_blocks_and_normal_occupancy_still_blocks()
    test_dive_takes_a_trapped_pearl_only_when_enabled_and_gated()
    test_fountain_detection_needs_countdown_one_on_two_turns_running()
    test_radar_parses_echoes_pings_ahead_and_guards_straight_dives()
    test_boxed_r_end_keeps_only_rear_splits_late()
    test_ram_takes_a_head_on_trade_only_when_enabled_and_even()
    test_feed_dies_next_to_a_long_teammate_only_when_enabled_and_safe()
    test_split_enemy_dist_only_blocks_splits_near_an_enemy_head()
