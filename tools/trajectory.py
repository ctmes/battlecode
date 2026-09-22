"""Trajectory collection for stage 5 (BC warm start -> PPO): runs bot/policy.py self-play on the real engine and
records a (features, action, log-prob, reward, done) Step per dragon-turn via Policy.decide(), ready for a GAE/PPO
update later -- the last piece of stage 4's environment before the learning algorithm itself.

Reward, per the plan's "Reward" section:
  step shaping = (length this turn - length last turn) / GROWTH_SCALE - STEP_COST
                 (growth == a pearl eaten; STEP_COST is a small per-turn cost so idling isn't free)
  death        = -(length at death) / DEATH_SCALE, added to that dragon's last step
  terminal     = +-1 / 0 (win/draw/loss), added to every surviving teammate's last step, plus a small bonus from
                 the team's OWN longest-living dragon and total length at game end (correlates with what the
                 round-500 tiebreaks reward, without literally reproducing their discrete A-vs-B comparison),
                 scaled by the map's tile count so it cannot swamp the win/draw/loss signal
Constants here are the *un-annealed* starting weights; annealing them down over training is a stage-5 concern, not
this module's. `MatchResult.a_length`/`a_dragons` turned out to be (approximately) the team's SUMMED alive length,
not its longest dragon's -- checked empirically, since neither field is documented -- so the longest-dragon term is
tracked here from each dragon's own last observed length instead of trusting a MatchResult field for it.
Not included (deferred): credit for causing a specific enemy's death -- on_death gives no clean per-dragon
attribution (a head-on kills both sides; a trap has no "who caused it" signal at all).
"""
import argparse
import concurrent.futures
import os
import pathlib
import sys
import time
from collections import namedtuple

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bot"))
sys.path.insert(0, str(ROOT / "tools"))
import mapgen  # noqa: E402
from policy import Policy, random_weights  # noqa: E402
from unswbc.engine import EngineModule  # noqa: E402

GROWTH_SCALE = 8.0     # length units per point of shaping reward for eating a pearl
STEP_COST = 1 / 256    # small per-turn cost so a policy can't stall for free shaping reward
DEATH_SCALE = 64.0     # matches encoder.py's own length normalisation
TIEBREAK_SCALE = 0.1   # weight of the longest-dragon + total-length terms, relative to the +-1 terminal outcome

Step = namedtuple("Step", "idx dense move_i move_logp split_i split_logp reward done")


class TrajPlayer:
    """One team's dragons for one game: wraps Policy.decide, keeps each dragon's growing step list plus its last
    known length (for shaping, and for the tiebreak terms at game end)."""

    def __init__(self, weights, seed):
        self.weights, self.seed = weights, seed
        self.policies = {}
        self.steps = {}     # did -> list[Step]
        self.last_len = {}  # did -> length as of its last recorded step
        self.alive = set()

    def spawn(self, did, init):
        self.policies[did] = Policy.from_init(init, self.weights, seed=self.seed * 1000 + did)
        self.steps[did] = []
        self.alive.add(did)

    def reply(self, did, block):
        d = self.policies[did].decide(block)
        prev = self.last_len.get(did, d["length"])
        shaped = (d["length"] - prev) / GROWTH_SCALE - STEP_COST
        self.last_len[did] = d["length"]
        self.steps[did].append(Step(d["idx"], d["dense"], d["move_i"], d["move_logp"],
                                     d["split_i"], d["split_logp"], shaped, False))
        return d["action"]

    def die(self, did):
        self.alive.discard(did)
        steps = self.steps.get(did)
        if steps:
            last = steps[-1]
            steps[-1] = last._replace(reward=last.reward - self.last_len.get(did, 0) / DEATH_SCALE, done=True)

    def finish(self, terminal_bonus):
        """Adds the shared terminal reward to every dragon still alive at game end, and marks those steps done."""
        for did in self.alive:
            steps = self.steps[did]
            if steps:
                last = steps[-1]
                steps[-1] = last._replace(reward=last.reward + terminal_bonus, done=True)

    def tiebreak_stats(self):
        lens = [self.last_len[d] for d in self.alive]
        return (max(lens) if lens else 0), sum(lens)


def collect_one(spec, weights, seed):
    seed_map, lo, hi = spec
    m = mapgen.generate(seed_map, lo, hi)
    data = m.dumps().encode()
    tile_scale = max(1, m.w * m.h)
    pa, pb = TrajPlayer(weights, seed), TrajPlayer(weights, seed + 1_000_000)
    owner = {}
    engine = EngineModule()

    def bot_spawn(did, init):
        team = next(line for line in init.split(b"\n") if line.startswith(b"TEAM")).split()[1]
        p = pa if team == b"A" else pb
        owner[did] = p
        p.spawn(did, init)

    def bot_reply(did, block):
        try:
            return owner[did].reply(did, block)
        except Exception:  # noqa: BLE001 - a crash shouldn't wedge the whole collection batch
            return b""

    def on_death(did, rnd, reason):
        owner[did].die(did)

    res = engine.run(data, bot_reply, on_death, bot_spawn, lambda line: None, 0)

    outcome_a = 0.0 if res.winner is None else (1.0 if res.winner == "A" else -1.0)
    longest_a, total_a = pa.tiebreak_stats()
    longest_b, total_b = pb.tiebreak_stats()
    bonus_a = TIEBREAK_SCALE * (longest_a + total_a) / tile_scale
    bonus_b = TIEBREAK_SCALE * (longest_b + total_b) / tile_scale
    pa.finish(outcome_a + bonus_a)
    pb.finish(-outcome_a + bonus_b)

    return pa.steps, pb.steps, res


def collect(specs, weights, seed):
    """Sequential (single-process) collection: returns a flat list of Step across every dragon in every game.
    Each game gets its own seed (seed + its index), not one shared seed -- otherwise every game's dragon-id N
    would draw from the identical RNG sequence, needlessly correlating the sampled actions across games."""
    out = []
    for i, spec in enumerate(specs):
        sa, sb = collect_one(spec, weights, seed + i)[:2]
        for steps in list(sa.values()) + list(sb.values()):
            out.extend(steps)
    return out


_pool_weights = None


def _pool_init(weights_seed):
    global _pool_weights
    _pool_weights = random_weights(weights_seed)


def _pool_job(spec_and_seed):
    spec, seed = spec_and_seed
    sa, sb = collect_one(spec, _pool_weights, seed)[:2]
    return list(sa.values()) + list(sb.values())


def collect_parallel(specs, weights_seed, seed, workers=None):
    """Same contract as collect(), but runs games across a worker pool (mirrors tools/rollout.py's pattern): each
    worker builds its own weights from `weights_seed` once, rather than pickling the arrays through IPC per job."""
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    jobs = [(spec, seed + i) for i, spec in enumerate(specs)]
    out = []
    with concurrent.futures.ProcessPoolExecutor(workers, initializer=_pool_init, initargs=(weights_seed,)) as pool:
        for step_lists in pool.map(_pool_job, jobs, chunksize=1):
            for steps in step_lists:
                out.extend(steps)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--min-side", type=int, default=10)
    ap.add_argument("--max-side", type=int, default=64)
    ap.add_argument("--seed", type=int, default=500000)
    ap.add_argument("--weights-seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0, help="0 = sequential (default), else a worker-pool size")
    args = ap.parse_args()

    specs = [(args.seed + i, args.min_side, args.max_side) for i in range(args.games)]
    t0 = time.perf_counter()
    if args.workers:
        steps = collect_parallel(specs, args.weights_seed, args.seed, workers=args.workers)
    else:
        steps = collect(specs, random_weights(args.weights_seed), args.seed)
    wall = time.perf_counter() - t0

    n = len(steps)
    rewards = [s.reward for s in steps]
    n_split = sum(1 for s in steps if s.split_i > 0)
    n_done = sum(1 for s in steps if s.done)
    mean_idx = sum(len(s.idx) for s in steps) / n
    mode = f"{args.workers} workers" if args.workers else "single-process"
    print(f"{args.games} games, {n:,} steps, {wall:.1f}s ({n / wall:,.0f} steps/s, {mode})")
    print(f"reward: min {min(rewards):.3f} mean {sum(rewards) / n:.4f} max {max(rewards):.3f}  "
          f"(sum {sum(rewards):.1f} over {n_done} terminal steps)")
    print(f"splits: {n_split} ({100 * n_split / n:.1f}%)  mean active features/step: {mean_idx:.1f}")


if __name__ == "__main__":
    main()
