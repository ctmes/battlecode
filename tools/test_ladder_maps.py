"""Checks tools/ladder_maps.py: every variant is a valid, symmetric map the engine plays, variants really are
different games, and tuning variants never overlap the held-out decision set.

Run:  .venv\\Scripts\\python.exe tools\\test_ladder_maps.py
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bot"))
sys.path.insert(0, str(ROOT / "tools"))
import arena  # noqa: E402
import ladder_maps  # noqa: E402
from brain import Brain  # noqa: E402
from unswbc.engine import EngineModule  # noqa: E402


def test_variants_are_valid_maps():
    for name in ladder_maps.POOL:
        base = ladder_maps.original(name)
        for v in (0, 1, 2, 3, 4, 7, ladder_maps.HOLDOUT):
            m = ladder_maps.variant(name, v)
            assert m.problems() == [], (name, v, m.problems())
            assert (len(m.kelp), len(m.portals), len(m.tiles)) == (len(base.kelp), len(base.portals), len(base.tiles))
            assert sorted(m.portals.values()) == sorted(base.portals.values()), (name, v)
    print("ok  every variant is a valid symmetric map with the same walls, portals and tiles")


def test_original_is_the_ladder_map():
    for name in ladder_maps.POOL:
        text = (ladder_maps.LADDER / f"{name}.map").read_text()
        assert ladder_maps.variant(name, 0).dumps().strip() == text.strip(), name
    print("ok  variant 0 is byte-for-byte the map the ladder plays")


def test_variants_are_different_games():
    Brain.strict, Brain.debug = True, False
    eng = EngineModule()
    for name in ("devil", "trophy"):
        outcomes = set()
        for v in range(8):
            res, deaths, errors = arena.play(eng, ladder_maps.variant(name, v).dumps().encode(),
                                             arena.BrainPlayer("me"), arena.BrainPlayer("opp"))
            assert not errors, errors[:1]
            outcomes.add((res.rounds, res.a_length, res.b_length, res.a_dragons, res.b_dragons, len(deaths)))
        assert len(outcomes) >= 6, f"{name}: only {len(outcomes)} distinct games from 8 variants"
        print(f"ok  {name}: 8 variants gave {len(outcomes)} distinct brain-vs-brain games")


def test_training_never_touches_the_holdout():
    held = set(ladder_maps.holdout_specs(20))
    for g in range(200):
        train = ladder_maps.train_specs(3, g)
        assert not held & set(train), g
        assert all(0 < v < ladder_maps.HOLDOUT for _, _, v in train)
    print("ok  200 generations of training variants never touch a held-out map")


def run():
    test_variants_are_valid_maps()
    test_original_is_the_ladder_map()
    test_training_never_touches_the_holdout()
    test_variants_are_different_games()


if __name__ == "__main__":
    run()
