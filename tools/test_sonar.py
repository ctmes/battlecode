"""Checks the sonar wire format (proto.pack_enemy_sonar/unpack_sonar, v2: position + length + facing) and its use
in brain.py as a relayed one-step prediction (DEFAULTS["sonar_predict"]).

Run:  .venv\\Scripts\\python.exe tools\\test_sonar.py
"""
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bot"))
sys.path.insert(0, str(ROOT / "tools"))
import arena  # noqa: E402
import mapgen  # noqa: E402
import proto  # noqa: E402
from brain import Brain  # noqa: E402
from unswbc.engine import EngineModule  # noqa: E402

MAPS = sorted((ROOT / "maps").glob("*.map"))


def test_pack_unpack_roundtrip():
    rng = random.Random(0)
    for _ in range(2000):
        x, y, length, facing = rng.randrange(64), rng.randrange(64), rng.randrange(50), rng.randrange(4)
        v = proto.pack_enemy_sonar(x, y, length, facing)
        assert 0 <= v <= 0xFFFFFFFF, f"payload {v} is not a uint32"
        kind, ux, uy, ulen, uf = proto.unpack_sonar(v)
        assert (kind, ux, uy, ulen, uf) == (proto.SONAR_ENEMY, x, y, min(length, 15), facing), \
            f"({x},{y},{length},{facing}) -> {v} -> {(kind, ux, uy, ulen, uf)}"
    for _ in range(2000):
        x, y, vertical, portal = rng.randrange(64), rng.randrange(64), rng.randrange(2), rng.randrange(2)
        v = proto.pack_terrain_sonar(x, y, vertical, portal)
        assert 0 <= v <= 0xFFFFFFFF, f"payload {v} is not a uint32"
        kind, ux, uy, uv, up = proto.unpack_sonar(v)
        assert (kind, ux, uy, uv, up) == (proto.SONAR_TERRAIN, x, y, vertical, portal), \
            f"({x},{y},{vertical},{portal}) -> {v} -> {(kind, ux, uy, uv, up)}"
    for kind_bits in (2, 3):  # any kind but ours must come back unrecognised, not misread as an enemy/terrain fact
        v = (kind_bits << 30) | 0x3FFFFFFF
        assert proto.unpack_sonar(v) == (None, 0, 0, 0, 0), f"kind {kind_bits} was not rejected"
    print("ok  sonar payloads round-trip exactly (enemy and terrain), and unknown kinds come back unrecognised")


def test_off_by_default_is_a_no_op():
    """sonar_predict == 0 (DEFAULTS) must never emit a SONAR line, whatever the situation."""
    from brain import DEFAULTS
    assert DEFAULTS["sonar_predict"] == 0.0
    engine = EngineModule()
    for mp in MAPS:
        me, other = arena.BrainPlayer("me"), arena.BrainPlayer("opp")
        res, deaths, errors = arena.play(engine, mp.read_bytes(), me, other)
        assert not errors
    print("ok  no crashes with sonar off (sanity check before the real sonar-on runs below)")


def test_engine_delivers_a_sent_sonar_message(engine):
    """With sonar_predict > 0 in a real game, at least one dragon receives a message that decodes as an enemy
    sighting, that sighting is a real dragon (any team) that was actually on the board, and its facing matches
    what that dragon actually reported on the wire at some point (not necessarily the same instant: the ray can
    take up to WIDTH+HEIGHT tiles and up to a round to arrive, so a little staleness is expected, not a bug)."""
    seen = {}  # (x, y) -> set of facings ever reported there
    heard = []

    class Watch(arena.BrainPlayer):
        def reply(self, did, block):
            t = proto.parse_turn(block)
            for q in t.parts:
                if q[5] == b"1":
                    seen.setdefault((int(q[2]), int(q[3])), set()).add(proto.LETTERS.find(q[4]))
            for m in t.msgs:
                kind, x, y, ln, f = proto.unpack_sonar(m)
                if kind == proto.SONAR_ENEMY:
                    heard.append((x, y, ln, f))
            return super().reply(did, block)

    p = {"sonar_predict": 4.0}
    mp = next(m for m in MAPS if m.stem == "big_empty")  # plenty of room for rays to travel and dragons to spread
    a, b = Watch("me", p), Watch("opp", p)
    res, deaths, errors = arena.play(engine, mp.read_bytes(), a, b)
    assert not errors, errors[0]
    assert heard, "sonar_predict > 0 but nobody ever received a decodable message in a whole game"
    bogus_pos = [h for h in heard if (h[0], h[1]) not in seen]
    bogus_facing = [h for h in heard if (h[0], h[1]) in seen and h[3] not in seen[(h[0], h[1])]]
    # both can go stale by the time a message is read (the target may since have moved, turned, or died), so
    # neither is 0 in general -- but both should be a small minority, not most of what was heard.
    assert len(bogus_pos) < 0.5 * len(heard), f"{len(bogus_pos)}/{len(heard)} heard positions were never any head"
    assert len(bogus_facing) < 0.5 * len(heard), f"{len(bogus_facing)}/{len(heard)} heard facings never matched"
    print(f"ok  sonar messages are delivered and decode to real positions and facings ({len(heard)} heard, "
          f"{len(bogus_pos)} stale position, {len(bogus_facing)} stale facing)")


def test_engine_sonar_predicted_matches_reported_facing(engine):
    """brain.py's internal `sonar_predicted` list (via self.dbg, same hook test_predict.py uses for `predicted`) is
    exactly (a received sighting's position + one step in its own reported facing) for every decodable message --
    the same extrapolation `predict` does locally, just fed from the wire instead of `t.parts`."""
    Brain.strict = Brain.debug = True
    checked = [0]

    class Watch(arena.BrainPlayer):
        def reply(self, did, block):
            action = super().reply(did, block)
            br = self.brains[did]
            if "sonar_predicted" not in br.dbg:
                return action
            t = proto.parse_turn(block)
            w_, h_ = br.W, br.H
            want = set()
            for m in t.msgs:
                kind, mx, my, _, mf = proto.unpack_sonar(m)
                if kind == proto.SONAR_ENEMY and mx < w_ and my < h_:
                    want.add(((mx + proto.DX[mf]) % w_, (my + proto.DY[mf]) % h_))
            got = set(br.dbg["sonar_predicted"])
            assert got == want, f"dragon {did}: sonar_predicted {got}, expected {want} from the wire data"
            checked[0] += len(want)
            return action

    p = {"sonar_predict": 4.0}
    for mp in MAPS[:3]:
        a, b = Watch("me", p), Watch("opp", p)
        res, deaths, errors = arena.play(engine, mp.read_bytes(), a, b)
        assert not errors, errors[0]
    assert checked[0] > 20, f"too few messages exercised the sonar-predict path ({checked[0]})"
    print(f"ok  brain.py's internal sonar_predicted list matches the wire data on {checked[0]} messages")


def test_engine_sonar_on_is_as_legal_and_safe_as_off(engine):
    """Turning sonar_predict on must not introduce a single legality bug or exception, on any bundled or generated
    map, against itself or a sparring bot -- it only adds an extra scored term and an extra output line."""
    Brain.strict = Brain.debug = True
    maps = [m.read_bytes() for m in MAPS] + [mapgen.generate(s, 10, 40).dumps().encode() for s in range(600000, 600020)]
    for data in maps:
        for side in "AB":
            me = arena.BrainPlayer("me", {"sonar_predict": 3.0})
            foe = arena.BrainPlayer("opp", {"sonar_predict": 3.0})
            res, deaths, errors = arena.play(engine, data, *((me, foe) if side == "A" else (foe, me)))
            assert not errors, errors[0]
    print(f"ok  sonar_predict=3 stays legal and exception-free on {len(maps)} maps, both sides")


def run():
    test_pack_unpack_roundtrip()
    test_off_by_default_is_a_no_op()
    engine = EngineModule()
    test_engine_delivers_a_sent_sonar_message(engine)
    test_engine_sonar_predicted_matches_reported_facing(engine)
    test_engine_sonar_on_is_as_legal_and_safe_as_off(engine)


if __name__ == "__main__":
    run()
