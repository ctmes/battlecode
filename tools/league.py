"""Parallel matches: brain parameter sets against opponents on many maps, on both sides.

A map is a tuple (mapgen seed, min side, max side) or the path of a .map file. An opponent is ("brain", params) for
another brain or ("bot", name) for a sparring bot from tools/opponents.py. Games are deterministic, so one game per
(map, side) is all there is to learn from a pairing; more samples need more maps.

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
from collections import namedtuple

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bot"))
sys.path.insert(0, str(ROOT / "tools"))
import arena  # noqa: E402
import mapgen  # noqa: E402
import opponents  # noqa: E402
from brain import Brain  # noqa: E402
from unswbc.engine import EngineModule  # noqa: E402

Job = namedtuple("Job", "map side mine opp")
# score: 1 win / 0.5 draw / 0 loss for `mine`; the rest describes my team; cost is seconds of wall time
Result = namedtuple("Result", "score deaths turns errors rounds my_len opp_len my_dragons opp_dragons cost")

_engine = None


def _init():
    global _engine
    Brain.strict, Brain.debug = True, False
    _engine = EngineModule()


@functools.lru_cache(maxsize=96)
def map_bytes(spec):
    if isinstance(spec, tuple):
        return mapgen.generate(*spec).dumps().encode()
    return pathlib.Path(spec).read_bytes()


def map_area(spec):
    """Cheap size estimate (no map is built) used to start the slowest games first."""
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


def _opponent(spec, seed):
    if spec[0] == "brain":
        return arena.BrainPlayer("opp", spec[1] or None)
    return opponents.OPPONENTS[spec[1]](seed=seed)


def play_job(job):
    t0 = time.perf_counter()
    me = CountingPlayer("me", job.mine or None)
    foe = _opponent(job.opp, job.map[0] if isinstance(job.map, tuple) else 0)
    res, deaths, errors = arena.play(_engine, map_bytes(job.map), *((me, foe) if job.side == "A" else (foe, me)))
    a = job.side == "A"
    score = 0.5 if res.winner is None else float(res.winner == job.side)
    return Result(score, sum(1 for n, *_ in deaths if n == "me"), me.turns, len(errors), res.rounds + 1,
                  res.a_length if a else res.b_length, res.b_length if a else res.a_length,
                  res.a_dragons if a else res.b_dragons, res.b_dragons if a else res.a_dragons,
                  time.perf_counter() - t0)


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
    """Win rate (draw = half), death rate per 1,000 dragon-turns and totals for a list of Results."""
    n = len(results)
    turns = sum(r.turns for r in results) or 1
    return {"games": n, "score": sum(r.score for r in results) / max(n, 1),
            "wins": sum(r.score == 1 for r in results), "draws": sum(r.score == 0.5 for r in results),
            "deaths_per_k": 1000 * sum(r.deaths for r in results) / turns, "errors": sum(r.errors for r in results),
            "cost": sum(r.cost for r in results)}


def wilson(score, n, z=1.96):
    """95% interval for a win rate (draws count as half a win, which is close enough for a guide)."""
    if n == 0:
        return 0.0, 1.0
    d = 1 + z * z / n
    centre = (score + z * z / (2 * n)) / d
    half = z * ((score * (1 - score) / n + z * z / (4 * n * n)) ** 0.5) / d
    return centre - half, centre + half
