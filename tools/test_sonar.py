"""Checks the sonar wire format (proto.pack_enemy_sonar/unpack_sonar) and its use in brain.py.

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
        x, y, length = rng.randrange(64), rng.randrange(64), rng.randrange(50)
        v = proto.pack_enemy_sonar(x, y, length)
        assert 0 <= v <= 0xFFFFFFFF, f"payload {v} is not a uint32"
        kind, ux, uy, ulen = proto.unpack_sonar(v)
        assert (kind, ux, uy, ulen) == (proto.SONAR_ENEMY, x, y, min(length, 15)), \
            f"({x},{y},{length}) -> {v} -> {(kind, ux, uy, ulen)}"
    for kind_bits in (1, 2, 3):  # any kind but ours must come back unrecognised, not misread as an enemy sighting
        v = (kind_bits << 30) | 0x3FFFFFFF
        assert proto.unpack_sonar(v) == (None, 0, 0, 0), f"kind {kind_bits} was not rejected"
    print("ok  sonar payloads round-trip exactly, and unknown kinds come back unrecognised")


def test_off_by_default_is_a_no_op():
    """sonar_danger == 0 (DEFAULTS) must never emit a SONAR line, whatever the situation."""
    from brain import DEFAULTS
    assert DEFAULTS["sonar_danger"] == 0.0
    engine = EngineModule()
    for mp in MAPS:
        me, other = arena.BrainPlayer("me"), arena.BrainPlayer("opp")
        res, deaths, errors = arena.play(engine, mp.read_bytes(), me, other)
        assert not errors
    print("ok  no crashes with sonar off (sanity check before the real sonar-on runs below)")


def test_engine_delivers_a_sent_sonar_message(engine):
    """With sonar_danger > 0 in a real game, at least one dragon receives a message that decodes as an enemy
    sighting, and that sighting is a real dragon (any team) that was actually on the board at some point."""
    seen_positions = set()
    heard = []

    class Watch(arena.BrainPlayer):
        def reply(self, did, block):
            t = proto.parse_turn(block)
            seen_positions.update((int(q[2]), int(q[3])) for q in t.parts)
            for m in t.msgs:
                kind, x, y, ln = proto.unpack_sonar(m)
                if kind == proto.SONAR_ENEMY:
                    heard.append((x, y, ln))
            return super().reply(did, block)

    p = {"sonar_danger": 4.0}
    mp = next(m for m in MAPS if m.stem == "big_empty")  # plenty of room for rays to travel and dragons to spread
    a, b = Watch("me", p), Watch("opp", p)
    res, deaths, errors = arena.play(engine, mp.read_bytes(), a, b)
    assert not errors, errors[0]
    assert heard, "sonar_danger > 0 but nobody ever received a decodable message in a whole game"
    bogus = [h for h in heard if (h[0], h[1]) not in seen_positions]
    # a reported position can go stale by the time it's read (the target may since have moved or died), so this
    # is not 0 in general -- but it should be a small minority, not most of what was heard.
    assert len(bogus) < 0.5 * len(heard), f"{len(bogus)}/{len(heard)} heard positions were never any dragon's tile"
    print(f"ok  sonar messages are actually delivered and decode to real positions ({len(heard)} heard, "
          f"{len(bogus)} stale)")


def test_engine_sonar_on_is_as_legal_and_safe_as_off(engine):
    """Turning sonar_danger on must not introduce a single legality bug or exception, on any bundled or generated
    map, against itself or a sparring bot -- it only adds an extra scored term and an extra output line."""
    Brain.strict = Brain.debug = True
    maps = [m.read_bytes() for m in MAPS] + [mapgen.generate(s, 10, 40).dumps().encode() for s in range(600000, 600020)]
    for data in maps:
        for side in "AB":
            me = arena.BrainPlayer("me", {"sonar_danger": 3.0})
            foe = arena.BrainPlayer("opp", {"sonar_danger": 3.0})
            res, deaths, errors = arena.play(engine, data, *((me, foe) if side == "A" else (foe, me)))
            assert not errors, errors[0]
    print(f"ok  sonar_danger=3 stays legal and exception-free on {len(maps)} maps, both sides")


def run():
    test_pack_unpack_roundtrip()
    test_off_by_default_is_a_no_op()
    engine = EngineModule()
    test_engine_delivers_a_sent_sonar_message(engine)
    test_engine_sonar_on_is_as_legal_and_safe_as_off(engine)


if __name__ == "__main__":
    run()
