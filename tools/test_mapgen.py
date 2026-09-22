"""Checks tools/mapgen.py: valid, deterministic, varied, and playable by the real engine.

Run:  .venv\\Scripts\\python.exe tools\\test_mapgen.py
"""
import pathlib
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bot"))
sys.path.insert(0, str(ROOT / "tools"))
import arena  # noqa: E402
import mapgen  # noqa: E402
import opponents  # noqa: E402
from brain import Brain  # noqa: E402
from mapfile import GameMap  # noqa: E402
from unswbc.engine import EngineModule  # noqa: E402

SEEDS = range(300)


def test_valid_after_a_text_round_trip():
    """Every generated map, written and read back, is symmetric, connected, in range, with a proper dragon start."""
    for s in SEEDS:
        m = GameMap.loads(mapgen.generate(s).dumps())
        assert m.problems(connected=True) == [], f"seed {s}: {m.problems(connected=True)}"
    print(f"ok  {len(SEEDS)} generated maps are valid")


def test_deterministic_and_bounded():
    for s in range(25):
        assert mapgen.generate(s).dumps() == mapgen.generate(s).dumps(), f"seed {s} is not reproducible"
    assert mapgen.generate(0).dumps() != mapgen.generate(1).dumps()
    for s in range(60):
        m = mapgen.generate(s, min_side=12, max_side=30)
        assert 12 <= m.w <= 30 and 12 <= m.h <= 30, f"seed {s}: {m.w}x{m.h} ignores the side limits"
    print("ok  reproducible per seed, side limits respected")


def test_variety():
    maps = [mapgen.generate(s) for s in SEEDS]
    assert {m.sym for m in maps} == set(mapgen.SYMMETRIES)
    assert {m.style for m in maps} == set(mapgen.STYLES)
    sides = [max(m.w, m.h) for m in maps]
    assert min(sides) <= 14 and max(sides) >= 55, f"sizes only span {min(sides)}..{max(sides)}"
    density = [len(m.kelp) / (2 * m.w * m.h) for m in maps]
    assert min(density) == 0 and max(density) > 0.10, f"kelp density spans {min(density):.2f}..{max(density):.2f}"
    with_portals = sum(bool(m.portals) for m in maps)
    assert 0.3 * len(maps) < with_portals < 0.9 * len(maps), f"{with_portals} of {len(maps)} maps have portals"
    per_team = Counter(len(m.dragons) // 2 for m in maps)
    assert set(per_team) == {1, 2, 3, 4}, f"dragons per team: {dict(per_team)}"
    spawning = [sum(hi > 0 for _, hi in m.tiles.values()) / len(m.tiles) for m in maps]
    assert min(spawning) < 0.5 < max(spawning), "no barren maps or no full ones"
    print(f"ok  varied: {dict(Counter(m.style for m in maps))}, {with_portals} with portals, "
          f"sides {min(sides)}..{max(sides)}, kelp up to {max(density):.0%}")


def test_engine_plays_generated_maps(engine):
    """The brain (strict, so an exception or a fatal move it could have avoided is an error) on 30 generated maps."""
    Brain.strict = Brain.debug = True
    played = Counter()
    for s in range(30):
        m = mapgen.generate(s, max_side=40)
        for name in ("chaser", "splitter"):
            for side in "AB":
                me, foe = arena.BrainPlayer("me"), opponents.OPPONENTS[name](seed=s)
                res, deaths, errors = arena.play(engine, m.dumps().encode(), *((me, foe) if side == "A" else (foe, me)))
                assert not errors, f"seed {s} vs {name} as {side}: {errors[0]}"
                played["games"] += 1
                played["rounds"] += res.rounds + 1
                played["deaths"] += sum(1 for n, *_ in deaths if n == "me")
    assert played["rounds"] > played["games"] * 20, "games end suspiciously early"
    print(f"ok  the engine plays generated maps ({played['games']} games, {played['rounds'] // played['games']} rounds "
          f"on average, no brain errors)")


def run():
    test_valid_after_a_text_round_trip()
    test_deterministic_and_bounded()
    test_variety()
    test_engine_plays_generated_maps(EngineModule())


if __name__ == "__main__":
    run()
