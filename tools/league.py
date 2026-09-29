"""Parallel matches: brain parameter sets against opponents on many maps, on both sides.

A map is a tuple (mapgen seed, min side, max side), ("ladder", name, variant) for a variant of a ladder map (see
tools/ladder_maps.py), or the path of a .map file. An opponent is ("brain", params) for
another brain, ("bot", name) for a sparring bot from tools/opponents.py, or ("frozen", dir_path) for a frozen bot/
snapshot (e.g. manual-heuristics/, see frozen_brain.py) whose code and params are pinned regardless of what
bot/brain.py currently contains. Games are deterministic, so one game per (map, side) is all there is to learn from
a pairing; more samples need more maps.

    with League() as lg:
        results = lg.run([Job(map, side, my_params, opponent), ...])
"""
import concurrent.futures
import functools
import os
import pathlib
import random
import sys
import time
from collections import Counter, namedtuple

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bot"))
sys.path.insert(0, str(ROOT / "tools"))
import arena  # noqa: E402
import frozen_brain  # noqa: E402
import ladder_maps  # noqa: E402
import mapgen  # noqa: E402
import opponents  # noqa: E402
from brain import Brain  # noqa: E402
from unswbc.engine import EngineModule  # noqa: E402

# seed: the engine's pearl sequence for this game (unswbc >= 1.0 seeds every match the way the judge does); None =
# the engine default, the only sequence unswbc 0.3.x has
Job = namedtuple("Job", "map side mine opp seed", defaults=(None,))
# score: 1 win / 0.5 draw / 0 loss for `mine`; the rest describes my team; cost is seconds of wall time.
# causes: {arena.DEATH name: count} for this job's deaths -- aggregate win rate alone can be too noisy to tell
# whether a change is doing what it's meant to (see the sonar work); the death-cause mix is the more direct signal.
Result = namedtuple("Result", "score deaths turns errors rounds my_len opp_len my_dragons opp_dragons cost causes")

_engine = None
# BC_UNMETERED=1 switches off every Brain's wall-clock self-metering (budget_ns). On the judge the meter reads CPU
# points, and a Brain turn costs at most ~16M of the 60M budget (sandbox, Portals, 19k turns), so it never fires there;
# locally it reads the wall clock and does fire under load, flipping whole games: the same 120 games scored 62.5% and
# 70.0% in two runs on 28 Sep. Unmetered, games are reproducible and match the judge.
UNMETERED = os.environ.get("BC_UNMETERED") == "1"


def metered(params):
    """`params` with the self-metering switched off when BC_UNMETERED=1 (see UNMETERED)."""
    return {**(params or {}), "budget_ns": 10 ** 15} if UNMETERED else params


def _init():
    global _engine
    Brain.strict, Brain.debug = True, False
    _engine = EngineModule()


@functools.lru_cache(maxsize=96)
def map_bytes(spec):
    if isinstance(spec, tuple) and spec[0] == "ladder":
        return ladder_maps.variant(spec[1], spec[2]).dumps().encode()
    if isinstance(spec, tuple):
        return mapgen.generate(*spec).dumps().encode()
    return pathlib.Path(spec).read_bytes()


def map_area(spec):
    """Cheap size estimate (no map is built) used to start the slowest games first."""
    if isinstance(spec, tuple) and spec[0] == "ladder":
        m = ladder_maps.original(spec[1])
        return m.w * m.h
    if isinstance(spec, tuple):
        seed, lo, hi = spec
        w, h = mapgen.pick_size(random.Random(f"battlecode-map/{seed}/0"), lo, hi)  # the first attempt, nearly always the one used
        return w * h
    _, w, h = pathlib.Path(spec).open().readline().split()
    return int(w) * int(h)


class CountingPlayer(arena.BrainPlayer):
    """A brain player that also counts dragon-turns, for a deaths-per-turn rate."""

    turns = 0

    def reply(self, did, block):
        self.turns += 1
        return super().reply(did, block)


class FrozenBrainPlayer:
    """Like arena.BrainPlayer, but backed by a Brain class loaded from a frozen snapshot rather than the live
    bot/brain.py import, so it plays exactly the code (not just the parameters) of whatever was pinned."""

    def __init__(self, name, brain_cls, params=None):
        self.name, self.brain_cls, self.params, self.brains = name, brain_cls, params, {}

    def spawn(self, did, init):
        self.brains[did] = self.brain_cls.from_init(init, self.params)

    def reply(self, did, block):
        return self.brains[did].act(block)


def _opponent(spec, seed):
    if spec[0] == "brain":
        return arena.BrainPlayer("opp", metered(spec[1]) or None)
    if spec[0] == "frozen":
        dir_path, params = spec[1]
        brain_cls, _ = frozen_brain.load(dir_path)
        return FrozenBrainPlayer("opp", brain_cls, metered(params) or None)
    return opponents.OPPONENTS[spec[1]](seed=seed)


def play_job(job):
    t0 = time.perf_counter()
    me = CountingPlayer("me", metered(job.mine) or None)
    foe = _opponent(job.opp, job.map[0] if isinstance(job.map, tuple) and isinstance(job.map[0], int) else 0)
    res, deaths, errors = arena.play(_engine, map_bytes(job.map), *((me, foe) if job.side == "A" else (foe, me)),
                                     seed=job.seed)
    a = job.side == "A"
    score = 0.5 if res.winner is None else float(res.winner == job.side)
    causes = dict(Counter(arena.DEATH.get(reason, reason) for n, _, _, reason in deaths if n == "me"))
    return Result(score, sum(causes.values()), me.turns, len(errors), res.rounds + 1,
                  res.a_length if a else res.b_length, res.b_length if a else res.a_length,
                  res.a_dragons if a else res.b_dragons, res.b_dragons if a else res.a_dragons,
                  time.perf_counter() - t0, causes)


class League:
    def __init__(self, workers=None):
        self.workers = workers or max(1, (os.cpu_count() or 2) - 2)
        self.pool = concurrent.futures.ProcessPoolExecutor(self.workers, initializer=_init)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.pool.shutdown(cancel_futures=True)

    def run(self, jobs):
        """Results in job order. The biggest boards start first so the batch has a short tail."""
        order = sorted(range(len(jobs)), key=lambda i: -map_area(jobs[i].map))
        out = [None] * len(jobs)
        for i, r in zip(order, self.pool.map(play_job, [jobs[i] for i in order], chunksize=1)):
            out[i] = r
        return out


def summarize(results):
    """Win rate (draw = half), death rate per 1,000 dragon-turns, death-cause mix and totals for a list of Results."""
    n = len(results)
    turns = sum(r.turns for r in results) or 1
    causes = Counter()
    for r in results:
        causes.update(r.causes)
    deaths = sum(causes.values())
    return {"games": n, "score": sum(r.score for r in results) / max(n, 1),
            "wins": sum(r.score == 1 for r in results), "draws": sum(r.score == 0.5 for r in results),
            "deaths_per_k": 1000 * deaths / turns, "errors": sum(r.errors for r in results),
            "cost": sum(r.cost for r in results), "causes": dict(causes),
            "head_on_share": causes.get("head-on", 0) / deaths if deaths else 0.0}


def wilson(score, n, z=1.96):
    """95% interval for a win rate (draws count as half a win, which is close enough for a guide)."""
    if n == 0:
        return 0.0, 1.0
    d = 1 + z * z / n
    centre = (score + z * z / (2 * n)) / d
    half = z * ((score * (1 - score) / n + z * z / (4 * n * n)) ** 0.5) / d
    return centre - half, centre + half
