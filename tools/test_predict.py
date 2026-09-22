"""Checks the one-step enemy-facing prediction (DEFAULTS["predict"]) in brain.py.

Run:  .venv\\Scripts\\python.exe tools\\test_predict.py
"""
import pathlib
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


def test_off_by_default_is_a_no_op():
    from brain import DEFAULTS
    assert DEFAULTS["predict"] == 0.0
    engine = EngineModule()
    for mp in MAPS:
        me, other = arena.BrainPlayer("me"), arena.BrainPlayer("opp")
        res, deaths, errors = arena.play(engine, mp.read_bytes(), me, other)
        assert not errors
    print("ok  no crashes with prediction off (sanity check before the real runs below)")


def test_engine_prediction_matches_reported_facing(engine):
    """brain.py's own internal `predicted` list (exposed via self.dbg, the same hook other tests use) is exactly
    (enemy cell + one step in its own reported facing) for every visible enemy head, cross-checked against the raw
    wire data independently re-read from the turn block -- not a second copy of brain.py's formula."""
    Brain.strict = Brain.debug = True  # strict: a bug must raise here, not silently fall back and leave dbg stale
    checked = [0]

    class Watch(arena.BrainPlayer):
        def reply(self, did, block):
            action = super().reply(did, block)
            br = self.brains[did]
            if "predicted" not in br.dbg:
                return action  # boxed in this turn: decide() returned before computing any prediction
            t = proto.parse_turn(block)
            w_, h_ = br.W, br.H
            want = set()
            for q in t.parts:
                if q[5] == b"1" and int(q[1]) != did and q[0] != br.team:
                    f = proto.LETTERS.find(q[4])
                    if f < 0:
                        continue
                    ex, ey = int(q[2]), int(q[3])
                    want.add(((ex + proto.DX[f]) % w_, (ey + proto.DY[f]) % h_))
            got = set(br.dbg["predicted"])
            assert got == want, f"dragon {did}: predicted {got}, expected {want} from the wire data"
            checked[0] += len(want)
            return action

    p = {"predict": 40.0}
    for mp in MAPS[:3]:
        a, b = Watch("me", p), Watch("opp", p)
        res, deaths, errors = arena.play(engine, mp.read_bytes(), a, b)
        assert not errors, errors[0]
    assert checked[0] > 20, f"too few enemy sightings exercised the prediction path ({checked[0]})"
    print(f"ok  brain.py's internal predicted-cell list matches the wire data on {checked[0]} enemy sightings")


def test_engine_prediction_is_as_legal_and_safe_as_off(engine):
    """Turning predict on must not introduce a single legality bug or exception."""
    Brain.strict = Brain.debug = True
    maps = [m.read_bytes() for m in MAPS] + [mapgen.generate(s, 10, 40).dumps().encode() for s in range(610000, 610020)]
    for data in maps:
        for side in "AB":
            me = arena.BrainPlayer("me", {"predict": 100.0})
            foe = arena.BrainPlayer("opp", {"predict": 100.0})
            res, deaths, errors = arena.play(engine, data, *((me, foe) if side == "A" else (foe, me)))
            assert not errors, errors[0]
    print(f"ok  predict=100 stays legal and exception-free on {len(maps)} maps, both sides")


def run():
    test_off_by_default_is_a_no_op()
    engine = EngineModule()
    test_engine_prediction_matches_reported_facing(engine)
    test_engine_prediction_is_as_legal_and_safe_as_off(engine)


if __name__ == "__main__":
    run()
