"""Checks the sparring bots in tools/opponents.py against the real engine.

Run:  .venv\\Scripts\\python.exe tools\\test_opponents.py
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bot"))
import opponents  # noqa: E402
import proto  # noqa: E402
from unswbc.engine import EngineModule  # noqa: E402

MAPS = sorted((ROOT / "maps").glob("*.map"))
SAFE = ("safe_random", "chaser", "rammer", "hugger", "splitter")  # the bots that only ever pick clean moves


class Watch:
    """Wraps a player and remembers, per dragon, whether its latest turn offered a clean move."""

    def __init__(self, inner):
        self.inner, self.name, self.clean = inner, inner.name, {}

    def spawn(self, did, init):
        self.inner.spawn(did, init)

    def reply(self, did, block):
        dragon = self.inner.dragons[did]
        self.clean[did] = bool(opponents.clean_dirs(proto.parse_turn(block), dragon.w, dragon.h))
        return self.inner.reply(did, block)


def play(engine, data, pa, pb):
    """Runs one game; returns (result, deaths) with deaths as (player, dragon id, round, reason letter)."""
    owner, deaths = {}, []

    def spawn(did, init):
        owner[did] = pa if init.split(b"TEAM ")[1][:1] == b"A" else pb
        owner[did].spawn(did, init)

    res = engine.run(data, lambda did, block: owner[did].reply(did, block),
                     lambda did, rnd, why: deaths.append((owner[did], did, rnd, why)), spawn, lambda line: None, 0)
    return res, deaths


def both_sides(engine, data, name, other, seed=0):
    """Plays `name` against `other` on each side; yields (side of name, name's player, result, deaths)."""
    for side in "AB":
        me, foe = opponents.OPPONENTS[name](seed=seed), opponents.OPPONENTS[other](seed=seed + 1)
        res, deaths = play(engine, data, *((me, foe) if side == "A" else (foe, me)))
        yield side, me, res, deaths


def test_valid_actions(engine):
    """Nothing but the dummy ever sends an action the engine rejects (death reason A)."""
    for name in opponents.OPPONENTS:
        for mp in MAPS:
            for _, me, _, deaths in both_sides(engine, mp.read_bytes(), name, "chaser"):
                bad = [d for d in deaths if d[0] is me and d[3] == "A"]
                assert bool(bad) == (name == "dummy"), f"{name} on {mp.stem}: {len(bad)} deaths by no valid action"
    print("ok  valid actions")


def test_safe_bots_never_step_into_known_death(engine):
    """A wall / self / body death is only allowed when the dragon had no clean move on its last turn."""
    for name in SAFE:
        for other in ("chaser", "greedy", "rammer", "splitter"):
            for mp in MAPS:
                data = mp.read_bytes()
                for side in "AB":
                    me = Watch(opponents.OPPONENTS[name](seed=0))
                    foe = opponents.OPPONENTS[other](seed=1)
                    _, deaths = play(engine, data, *((me, foe) if side == "A" else (foe, me)))
                    for pl, did, rnd, why in deaths:
                        assert not (pl is me and why in "WSO" and me.clean[did]), \
                            f"{name} vs {other} on {mp.stem}: dragon {did} died ({why}) in round {rnd} with a clean move"
    print("ok  safe bots only die boxed in, by a head-on, or through a portal")


def test_deterministic_and_seeded(engine):
    data = MAPS[0].read_bytes()
    for name in opponents.OPPONENTS:
        runs = [[(r.rounds, r.winner, r.a_length, r.b_length) for _, _, r, _ in both_sides(engine, data, name, "random")]
                for _ in range(2)]
        assert runs[0] == runs[1], f"{name} replays differently"
    outcomes = {tuple(r.rounds for _, _, r, _ in both_sides(engine, mp.read_bytes(), "random", "random", seed))
                for mp in MAPS for seed in range(3)}
    assert len(outcomes) > 1, "the seed does not change the random bot"
    print("ok  deterministic per seed, and the seed matters")


def test_dummy_always_loses(engine):
    for mp in MAPS:
        for side, _, res, _ in both_sides(engine, mp.read_bytes(), "dummy", "chaser"):
            assert res.winner == ("B" if side == "A" else "A"), f"dummy did not lose on {mp.stem} as {side}"
    print("ok  the dummy loses everywhere")


def test_splitter_splits(engine):
    data = next(mp for mp in MAPS if mp.stem == "big_empty").read_bytes()
    _, me, _, _ = next(both_sides(engine, data, "splitter", "chaser"))
    assert len(me.dragons) > 1, "the splitter never split"
    print(f"ok  the splitter split into {len(me.dragons)} dragons on big_empty")


def run():
    engine = EngineModule()
    for test in (test_valid_actions, test_safe_bots_never_step_into_known_death, test_deterministic_and_seeded,
                 test_dummy_always_loses, test_splitter_splits):
        test(engine)


if __name__ == "__main__":
    run()
