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

Brain.strict = True  # a crash inside decide() fails the test instead of falling back
from test_encoder import H, HX, HY, W, render_block  # noqa: E402


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


def with_echo(block, echo):
    ls = block.split(b"\n")
    return b"\n".join(ls[:5] + [b"ECHOES " + b" ".join(b"%d" % v for v in echo)] + ls[5:])


def test_radar_parses_echoes_pings_ahead_and_guards_dives():
    import proto
    # a 3-long dragon heading north into a one-wide kelp corridor that runs on out of view (a trap as far as this
    # dragon knows), with a pearl at its mouth
    edges = {(0, y, "v"): "w" for y in (-1, -2, -3)} | {(1, y, "v"): "w" for y in (-1, -2, -3)}
    parts = [("A", 1, 0, 0, "N", 1), ("A", 1, 0, 1, "N", 0), ("A", 1, 0, 2, "N", 0)]
    block = render_block(0, {(0, -1)}, {}, edges, parts, length=3, units=5)

    t0, t1 = proto.parse_turn(block), proto.parse_turn(with_echo(block, (1, 0, 0, 0, 0)))
    assert t0.echo is None and t1.echo == (1, 0, 0, 0, 0)
    assert (t1.hx, t1.hy, t1.flags, t1.cds, t1.parts, t1.edges) == (t0.hx, t0.hy, t0.flags, t0.cds, t0.parts, t0.edges)

    dive = {"dive_len": 3, "dive_trap": 0.0, "dive_scope": 1, "radar": 1, "radar_dive": 1}

    def act(params, echo):
        b = Brain(1, b"A", W, H, 64, params)
        b.pinged = (0,)  # as if it had pinged north last turn
        return b, b.act(with_echo(block, echo))

    b, out = act(dive, (1, 0, 0, 0, 0))  # the ray left the view and stopped on kelp: nobody in the corridor
    assert b.lines == {0: 0} and out == MOVES[0] + b"PROTOCOL 3\nSONAR N 0\n", (b.lines, out)
    b, out = act(dive, (0, 0, 0, 1, 0))  # it stopped on an enemy body: no dive
    assert b.lines == {0: 3} and not out.startswith(MOVES[0]) and b"PROTOCOL 3\nSONAR " in out, (b.lines, out)
    assert Brain(1, b"A", W, H, 64, {}).act(block) != MOVES[0]  # radar off: no dive, no PROTOCOL line
    assert b"PROTOCOL" not in Brain(1, b"A", W, H, 64, {}).act(block)
    b = Brain(1, b"A", W, H, 64, {"radar": 1})
    b.act(block)
    assert b"PROTOCOL" not in b.act(block)  # sent once: the version sticks (and split children inherit it)
    print("ok: radar parses ECHOES, pings along the move and keeps a dive out of an occupied line")


def test_radar_three_rays_decode_what_the_view_cannot_explain():
    # heading north: kelp right beside the head on the east (that ray stops on it, in view), a teammate's body two
    # tiles west (that ray stops on it, in view), and the line north runs out of view
    edges = {(1, 0, "v"): "w"}
    parts = [("A", 1, 0, 0, "N", 1), ("A", 1, 0, 1, "N", 0), ("A", 1, 0, 2, "N", 0),
             ("A", 5, -2, 1, "N", 1), ("A", 5, -2, 0, "S", 0)]
    block = render_block(0, set(), {}, edges, parts, length=3, units=5)

    def lines(echo):
        b = Brain(1, b"A", W, H, 64, {"radar": 2})
        b.pinged = (0, 1, 3)
        b.act(with_echo(block, echo))
        return b.lines

    assert lines((1, 1, 0, 0, 1)) == {1: 0, 3: 1, 0: 4}  # left over after east and west: an enemy head north
    assert lines((2, 1, 0, 0, 0)) == {1: 0, 3: 1, 0: 0}  # ... kelp: the line north is clear
    assert lines((0, 1, 0, 0, 1)) == {1: 0, 3: 1}  # disagrees with the view: only the view is trusted
    b = Brain(1, b"A", W, H, 64, {"radar": 2})
    assert b.act(block).endswith(b"PROTOCOL 3\nSONAR N 0\nSONAR E 0\nSONAR W 0\n")
    print("ok: three rays decode: what the view explains is subtracted, the rest belongs to the unseen line")


def sprint_paths_of(parts, length, pearls=(), edges=None, smax=3):
    """Brain.sprint_paths for a window drawn with render_block (ours is id 1, team A): {"NNE": (length after,
    ram?), ...}."""
    import test_encoder
    b = Brain(1, b"A", W, H, 64, {})
    b.act(render_block(0, set(pearls), {}, edges or {}, parts, length=length))  # learns the edges, seeds the body
    hx, hy = test_encoder.HX, test_encoder.HY

    def cell(dx, dy):
        return ((hy + dy) % H) * W + (hx + dx) % W

    mine = others = 0
    heads = {}
    for team, pid, dx, dy, _, is_head in parts:
        if pid == 1:
            mine |= 1 << cell(dx, dy)
        else:
            others |= 1 << cell(dx, dy)
            if is_head:
                heads[cell(dx, dy)] = (team != "A", pid)
    pm = sum(1 << cell(dx, dy) for dx, dy in pearls)
    out = b.sprint_paths(hx, hy, length, others, mine, heads, pm, b.window((hx - 3) % W, (hy - 3) % H), smax)
    return {"".join("NESW"[d] for d in dirs): (n, ram >= 0) for dirs, cells, n, eaten, ram, bits, tl in out}


def test_sprint_paths_follow_the_engine_step_by_step():
    line = [("A", 1, 0, y, "N", 1 if y == 0 else 0) for y in range(3)]  # 3 long, heading north
    got = sprint_paths_of(line, 3)
    assert set(got) == {"NN", "NE", "NW", "EN", "EE", "ES", "WN", "WW", "WS"}, sorted(got)
    assert all(v == (2, False) for v in got.values()), got  # each extra step costs a segment; 2 long can't go on
    got = sprint_paths_of(line, 3, pearls={(0, -1)})
    assert got["NN"] == (3, False) and got["NNN"] == (2, False), got  # a pearl on the way pays for a step
    # a U: head (0,0), then (1,0), (1,1), and the tail at (0,1) just south of the head. Stepping onto the tail is
    # death as a first step (the tail has not moved yet), but the tail has left by the third step of W, S, E
    u = [("A", 1, 0, 0, "W", 1), ("A", 1, 1, 0, "W", 0), ("A", 1, 1, 1, "N", 0), ("A", 1, 0, 1, "E", 0)]
    got = sprint_paths_of(u, 4)
    assert not any(k.startswith("S") or k.startswith("E") for k in got), sorted(got)
    assert got["WSE"] == (2, False), got
    # kelp stops a path at any step
    got = sprint_paths_of(line, 3, edges={(0, -1, "h"): "w"})
    assert "NN" not in got and "NE" in got, sorted(got)
    # an enemy head two steps ahead: NN is a ram; a teammate's head there is just a wall
    foe = [("B", 2, 0, -2, "S", 1), ("B", 2, 0, -3, "S", 0), ("B", 2, 0, -4, "S", 0)]
    assert sprint_paths_of(line + foe, 3)["NN"] == (3, True)
    mate = [("A", 3, 0, -2, "S", 1), ("A", 3, 0, -3, "S", 0)]
    assert "NN" not in sprint_paths_of(line + mate, 3)
    print("ok: sprint paths pay per step, eat on the way, respect kelp and see the tail leave; rams are marked")


def test_sprint_ram_takes_an_enemy_head_two_steps_away():
    def act(params, enemy_len=3):
        parts = [("A", 1, 0, y, "N", 1 if y == 0 else 0) for y in range(3)]
        parts += [("B", 2, 0, -2 - i, "S", 1 if i == 0 else 0) for i in range(enemy_len)]
        return Brain(1, b"A", W, H, 64, params).act(render_block(0, set(), {}, {}, parts, length=3))

    ram = {"sprint_max": 3, "sprint_ram": 1000.0, "ram_len": 3, "ram": 1000.0}
    assert act(ram) == b"MOVE NN\n", act(ram)
    assert act({**ram, "sprint_ram": 0.0}) != b"MOVE NN\n"  # off
    assert act(ram, enemy_len=2) != b"MOVE NN\n"  # it shows fewer segments than we have
    assert act({**ram, "ram_len": 2}) != b"MOVE NN\n"  # too long to throw away
    print("ok: a short dragon sprints onto an enemy head two steps away, only when enabled and even")


def test_sprint_threat_keeps_a_long_dragon_out_of_an_enemy_sprint():
    # a 6-long dragon heading north, a 3-long enemy head at (2,-1): N and E end 2 steps from it (a 2-step sprint
    # away), W ends 4 away
    parts = [("A", 1, 0, y, "N", 1 if y == 0 else 0) for y in range(6)]
    parts += [("B", 2, 2, -1 + i, "N", 1 if i == 0 else 0) for i in range(3)]

    def act(params):
        return Brain(1, b"A", W, H, 64, params).act(render_block(0, set(), {}, {}, parts, length=6))

    assert act({}) == MOVES[0], act({})  # straight on by default
    assert act({"sprint_threat": 1.0}) == MOVES[3], act({"sprint_threat": 1.0})
    print("ok: sprint_threat steers a long dragon out of a short enemy's sprint reach")


def test_body_after_a_sprint_includes_every_tile_crossed():
    import test_encoder
    parts = [("A", 1, 0, y, "N", 1 if y == 0 else 0) for y in range(3)]
    b = Brain(1, b"A", W, H, 64, {"sprint_max": 2, "sprint_seg": 100.0})
    out = b.act(render_block(0, {(0, -1), (0, -2)}, {}, {}, parts, length=3))
    assert out == b"MOVE NN\n", out  # two pearls for one segment
    hx, hy = test_encoder.HX, test_encoder.HY
    try:
        test_encoder.HY = hy - 2  # the head is two tiles north now, 4 long (3 + 2 pearls - 1 segment)
        parts2 = [("A", 1, 0, y, "N", 1 if y == 0 else 0) for y in range(4)]
        b.act(render_block(0, set(), {}, {}, parts2, length=4))
    finally:
        test_encoder.HY = hy
    want = [((hy - 2 + y) % H) * W + hx for y in range(4)]
    assert b.body == want, (b.body, want)
    assert b.own == sum(1 << c for c in want)
    print("ok: after a sprint the tracked body includes every tile the sprint crossed")


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


def test_food_pull_steers_up_the_field_only_with_no_pearl_in_view():
    import known_maps
    sizes = {m[0]: w * h for (w, h), ms in known_maps.MAPS.items() for m in ms}
    assert known_maps.FIELDS and all(len(f) == sizes[name] for name, f in known_maps.FIELDS.items())
    assert "devil" in known_maps.FIELDS and "default" not in known_maps.FIELDS  # food lands anywhere on Default
    parts = [("A", 1, 0, y, "N", 1 if y == 0 else 0) for y in range(3)]  # 3 long, heading north
    east = (HY * W + (HX + 1) % W)
    field = bytes(200 if c == east else 100 for c in range(W * H))  # richer one step east

    def act(params, pearls=()):
        b = Brain(1, b"A", W, H, 64, params)
        b.field = field
        return b.act(render_block(0, set(pearls), {}, {}, parts, length=3))

    assert act({}) == MOVES[0], act({})  # off by default: straight on
    assert act({"food_pull": 100.0}) == MOVES[1], act({"food_pull": 100.0})
    assert act({"food_pull": 100.0}, {(-2, -3)}) != MOVES[1]  # a pearl in view: the pearl decides, not the field
    print("ok: food_pull steers a dragon with no pearl in view towards richer ground, only when enabled")


def test_rally_relays_the_king_and_pulls_short_dragons_towards_it():
    import proto
    for x, y, n, r in ((0, 0, 8, 0), (63, 63, 127, 500), (5, 9, 200, 301)):
        kind, ux, uy, un, ur = proto.unpack_sonar(proto.pack_king_sonar(x, y, n, r))
        assert (kind, ux, uy, un, ur) == (proto.SONAR_KING, x, y, min(n, 127), r & ~3), (x, y, n, r)
    assert proto.unpack_sonar(proto.pack_king_sonar(5, 9, 20, 300) ^ 1)[0] is None  # a bad check is dropped
    b_msg = proto.pack_king_sonar(5, 9, 20, 300, proto.king_salt(b"B"))
    assert proto.unpack_sonar(b_msg, proto.king_salt(b"B"))[0] == proto.SONAR_KING
    assert proto.unpack_sonar(b_msg, proto.king_salt(b"A"))[0] is None  # the other team's relay is not ours
    assert proto.unpack_sonar(2 ** 40)[0] is None  # a 64-bit value is nobody's king
    rally = {"rally_r": 350, "rally_pull": 200.0, "rally_age": 30, "feed_min": 8, "feed_len": 8}
    parts = [("A", 1, 0, y, "N", 1 if y == 0 else 0) for y in range(3)]  # 3 long, heading north
    A = proto.king_salt(b"A")  # the Brains below play team A
    king_east = proto.pack_king_sonar((HX + 12) % W, HY, 20, 400, A)

    def act(params, length=3, msgs=(), rnd=400, body=parts):
        return Brain(1, b"A", W, H, 64, params).act(
            render_block(0, set(), {}, {}, body, rnd=rnd, length=length, msgs=msgs))

    assert act({}, msgs=(king_east,)) == MOVES[0]  # off by default: straight on, no ping
    enemy_king = proto.pack_king_sonar((HX + 12) % W, HY, 20, 400, proto.king_salt(b"B"))
    assert act(rally, msgs=(enemy_king,)) == MOVES[0]  # the other team's king: ignored
    a = act(rally, msgs=(king_east,))
    assert a.startswith(MOVES[1]) and a.endswith(b"SONAR %d\n" % king_east), a  # east to it, relaying it
    assert act(rally, msgs=(king_east,), rnd=340).startswith(MOVES[0])  # before rally_r: relays, no pull
    assert act(rally, msgs=(proto.pack_king_sonar((HX + 12) % W, HY, 20, 360, A),)) == MOVES[0]  # 40 rounds stale
    long_body = [("A", 1, 0, y, "N", 1 if y == 0 else 0) for y in range(3)]
    a = act(rally, length=10, body=long_body)  # 10 long, no king heard of: it is the king and says so
    assert a.endswith(b"SONAR %d\n" % proto.pack_king_sonar(HX, HY, 10, 400, A)), a
    a = act({**rally, "radar": 2}, msgs=(king_east,))  # with the radar on, every ray carries the king (protocol 3)
    assert a.startswith(MOVES[1]) and b"PROTOCOL 3\n" in a, a
    assert [x for x in a.split(b"\n") if x.startswith(b"SONAR")] == \
        [b"SONAR %c %d" % (c, king_east) for c in b"ESN"], a  # ahead (east), then right and left
    print("ok: rally relays the longest teammate's head by sonar and pulls short dragons towards it")


def test_guard_keeps_the_king_off_a_pearl_next_to_an_enemy_head():
    me = [("A", 1, 0, y, "N", 1 if y == 0 else 0) for y in range(4)]  # heading north, 20 long in all
    foe = [("B", 9, 2, 0, "W", 1), ("B", 9, 3, 0, "W", 0)]  # a 2-long enemy head two east of ours

    def act(params):
        return Brain(1, b"A", W, H, 64, params).act(
            render_block(0, {(1, 0)}, {}, {}, me + foe, rnd=300, length=20))  # the pearl is next to its head

    guard = {"rally_r": 250, "guard_len": 12, "guard_care": 6.0}
    assert act({}).startswith(MOVES[1]), act({})  # unguarded: takes the pearl beside the enemy head
    assert not act(guard).startswith(MOVES[1]), act(guard)  # the king (no longer one heard of) keeps away
    assert act({"guard_len": 12, "guard_care": 6.0}).startswith(MOVES[1]) is False  # relay off: any 12+ is guarded
    assert act({**guard, "guard_len": 21}).startswith(MOVES[1])  # too short to be guarded
    print("ok: guard makes the king pass up a pearl next to an enemy head, only when enabled")


def test_escort_holds_a_ring_round_the_king_and_closes_on_enemies_near_it():
    import proto
    parts = [("A", 1, 0, y, "N", 1 if y == 0 else 0) for y in range(3)]  # 3 long, heading north
    esc = {"escort_r": 200, "rally_age": 30, "feed_min": 8, "escort_len": 4, "escort_ring": 4}

    def act(params, king, extra=()):
        return Brain(1, b"A", W, H, 64, params).act(render_block(
            0, set(), {}, {}, parts + list(extra), rnd=300, length=3, msgs=(proto.pack_king_sonar(*king, 20, 300, proto.king_salt(b"A")),)))

    far_east = ((HX + 12) % W, HY)
    assert act({}, far_east) == MOVES[0]  # off: straight on
    assert act({**esc, "escort_pull": 200.0}, far_east).startswith(MOVES[1])  # east, towards the ring
    south = (HX, (HY + 5) % H)  # the king 5 south; an enemy head 3 west and 2 south is within ring + 2 of it
    foe = [("B", 9, -3, 2, "E", 1), ("B", 9, -3, 3, "E", 0)]
    assert act({**esc, "escort_block": 300.0}, south, foe).startswith(MOVES[3])  # west, towards the enemy
    print("ok: escorts hold a ring round the king heard of and close on enemy heads near it, only when enabled")


def test_feed_ahead_dies_only_in_front_of_the_king():
    feed = {"feed_r": 250, "feed_len": 8, "feed_dist": 3, "feed_min": 8, "feed_ratio": 1.5}
    me = [("A", 1, 0, y, "N", 1 if y == 0 else 0) for y in range(3)]

    def king(facing):  # a teammate head two north of ours, with 8 segments in view
        body = [(0, -2), (1, -2), (1, -3), (2, -3), (2, -2), (3, -2), (3, -3), (-1, -3)]
        return [("A", 7, x, y, facing, 1 if i == 0 else 0) for i, (x, y) in enumerate(body)]

    def act(params, facing):
        return Brain(1, b"A", W, H, 64, params).act(render_block(0, set(), {}, {}, me + king(facing), rnd=300, length=3))

    assert act(feed, "N") == b"SPLIT 1\n"  # the plain rule feeds anywhere within feed_dist
    assert act({**feed, "feed_ahead": 3}, "S") == b"SPLIT 1\n"  # it faces us: we are in its path
    assert act({**feed, "feed_ahead": 3}, "N") != b"SPLIT 1\n"  # it faces away: not in its path
    print("ok: feed_ahead feeds only in front of the long teammate's head")


def test_hunt_closes_on_a_long_enemy_head():
    me = [("A", 1, 0, y, "N", 1 if y == 0 else 0) for y in range(3)]  # 3 long, heading north
    prey = [("B", 9, -3, 1 + i // 3, "E", 1 if i == 0 else 0) for i in range(12)]  # 12 segments, head 3 west

    def act(params):
        return Brain(1, b"A", W, H, 64, params).act(render_block(0, set(), {}, {}, me + prey, rnd=300, length=3))

    assert act({}) == MOVES[0], act({})  # off: straight on
    assert act({"hunt": 300.0, "hunt_len": 10, "ram_len": 3}) == MOVES[3], act({"hunt": 300.0, "hunt_len": 10, "ram_len": 3})  # west, at it
    assert act({"hunt": 300.0, "hunt_len": 13, "ram_len": 3}) == MOVES[0]  # not long enough to be worth it
    print("ok: hunt sends a short dragon at a long enemy head, only when enabled")


def test_map_params_apply_once_the_map_is_recognised():
    import brain as B
    fake = ("fakemap", 0, 0, 0, 0, (), 0)  # no kelp, no portals: agrees with a Brain that has learned none
    B.MAP_PARAMS["fakemap"] = {"trap": 7.0}
    try:
        b = Brain(1, b"A", W, H, 64, {"trap": 1.0})
        b.cands = [fake]
        b.recognise(0, 0)
        assert b.oracle == "fakemap" and b.p["trap"] == 7.0, b.p["trap"]
        off = Brain(1, b"A", W, H, 64, {"trap": 1.0, "map_params": 0})
        off.cands = [fake]
        off.recognise(0, 0)
        assert off.oracle == "fakemap" and off.p["trap"] == 1.0
        assert B.DEFAULTS["trap"] != 7.0  # the shared DEFAULTS are never written
    finally:
        del B.MAP_PARAMS["fakemap"]
    print("ok: MAP_PARAMS layer a map's own settings over the rest once the oracle recognises it")


if __name__ == "__main__":
    test_sprint_paths_follow_the_engine_step_by_step()
    test_sprint_ram_takes_an_enemy_head_two_steps_away()
    test_sprint_threat_keeps_a_long_dragon_out_of_an_enemy_sprint()
    test_body_after_a_sprint_includes_every_tile_crossed()
    test_fallback_portal_edge_ignores_irrelevant_landing_tile_occupancy()
    test_fallback_kelp_still_blocks_and_normal_occupancy_still_blocks()
    test_dive_takes_a_trapped_pearl_only_when_enabled_and_gated()
    test_fountain_detection_needs_countdown_one_on_two_turns_running()
    test_radar_parses_echoes_pings_ahead_and_guards_dives()
    test_radar_three_rays_decode_what_the_view_cannot_explain()
    test_boxed_r_end_keeps_only_rear_splits_late()
    test_ram_takes_a_head_on_trade_only_when_enabled_and_even()
    test_feed_dies_next_to_a_long_teammate_only_when_enabled_and_safe()
    test_split_enemy_dist_only_blocks_splits_near_an_enemy_head()
    test_food_pull_steers_up_the_field_only_with_no_pearl_in_view()
    test_rally_relays_the_king_and_pulls_short_dragons_towards_it()
    test_guard_keeps_the_king_off_a_pearl_next_to_an_enemy_head()
    test_escort_holds_a_ring_round_the_king_and_closes_on_enemies_near_it()
    test_feed_ahead_dies_only_in_front_of_the_king()
    test_hunt_closes_on_a_long_enemy_head()
    test_map_params_apply_once_the_map_is_recognised()
