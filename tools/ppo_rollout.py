"""PPO rollout collection: tools/ppo_policy.py self-play (or vs Brain) on the real engine, recording a
(features, action, log-prob, value, reward, done) PPOStep per dragon-turn for GAE + the PPO update in train_ppo.py.

Reuses tools/trajectory.py's reward formula unchanged (see its docstring) -- PPOTrajPlayer here is
trajectory.TrajPlayer's counterpart for the actor-critic policy, just carrying an extra `value` per step (from
PPOPolicy's critic).

League (a scoped-down version of the plan's "past checkpoints + the heuristic bot + occasional exploiters"): each
game is either mirror self-play (both sides run the identical weights) or one side vs Brain (the tuned heuristic),
chosen per game by `--brain-frac`. Past-checkpoint opponents and exploiters aren't implemented -- see
train_ppo.py's module docstring for why this is a deliberately scoped v1, not an oversight.
"""
import argparse
import concurrent.futures
import os
import pathlib
import sys
import time
from collections import namedtuple

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bot"))
sys.path.insert(0, str(ROOT / "tools"))
import mapgen  # noqa: E402
import torch_policy  # noqa: E402
from brain import Brain  # noqa: E402
from ppo_policy import PPOPolicy  # noqa: E402
from trajectory import GROWTH_SCALE, STEP_COST, DEATH_SCALE, TIEBREAK_SCALE  # noqa: E402
from unswbc.engine import EngineModule  # noqa: E402

PPOStep = namedtuple("PPOStep", "idx dense move_i move_logp split_i split_logp value reward done")


class PPOTrajPlayer:
    def __init__(self, weights, seed):
        self.weights, self.seed = weights, seed
        self.policies, self.steps, self.last_len, self.alive = {}, {}, {}, set()

    def spawn(self, did, init):
        self.policies[did] = PPOPolicy.from_init(init, self.weights, seed=self.seed * 1000 + did)
        self.steps[did] = []
        self.alive.add(did)

    def reply(self, did, block):
        d = self.policies[did].decide(block)
        prev = self.last_len.get(did, d["length"])
        # See trajectory.TrajPlayer.reply(): only reward growth, don't penalize a length drop (only ever a
        # deliberate split) -- splitting's payoff arrives far beyond GAE's credit horizon, and penalizing it here
        # measurably suppressed split rate over training (4.1% -> 1.2% over 25 PPO iterations).
        shaped = max(0.0, d["length"] - prev) / GROWTH_SCALE - STEP_COST
        self.last_len[did] = d["length"]
        self.steps[did].append(PPOStep(d["idx"], d["dense"], d["move_i"], d["move_logp"],
                                        d["split_i"], d["split_logp"], d["value"], shaped, False))
        return d["action"]

    def die(self, did):
        self.alive.discard(did)
        steps = self.steps.get(did)
        if steps:
            last = steps[-1]
            steps[-1] = last._replace(reward=last.reward - self.last_len.get(did, 0) / DEATH_SCALE, done=True)

    def finish(self, terminal_bonus):
        """See trajectory.TrajPlayer.finish(): every dragon that ever played gets the terminal reward on its own
        last step, not just ones still alive at game end -- fixes a real bug where dying early let a dragon dodge
        its team's eventual loss (measured: ~-0.05 avg reward dying early vs ~-0.4 surviving to a likely loss)."""
        for did, steps in self.steps.items():
            if steps:
                last = steps[-1]
                steps[-1] = last._replace(reward=last.reward + terminal_bonus, done=True)

    def tiebreak_stats(self):
        lens = [self.last_len[d] for d in self.alive]
        return (max(lens) if lens else 0), sum(lens)


class BrainOpponent:
    """A non-learning league opponent: no trajectory is recorded for it, only for the PPO side."""

    def __init__(self):
        self.brains = {}

    def spawn(self, did, init):
        self.brains[did] = Brain.from_init(init)

    def reply(self, did, block):
        return self.brains[did].act(block)


def collect_one(spec, weights, seed, vs_brain):
    seed_map, lo, hi = spec
    m = mapgen.generate(seed_map, lo, hi)
    data = m.dumps().encode()
    tile_scale = max(1, m.w * m.h)
    pa = PPOTrajPlayer(weights, seed)
    pb = BrainOpponent() if vs_brain else PPOTrajPlayer(weights, seed + 1_000_000)
    owner = {}

    def bot_spawn(did, init):
        team = next(line for line in init.split(b"\n") if line.startswith(b"TEAM")).split()[1]
        p = pa if team == b"A" else pb
        owner[did] = p
        p.spawn(did, init)

    def bot_reply(did, block):
        try:
            return owner[did].reply(did, block)
        except Exception:  # noqa: BLE001 - a crash shouldn't wedge the whole rollout batch
            return b""

    def on_death(did, rnd, reason):
        p = owner[did]
        if isinstance(p, PPOTrajPlayer):
            p.die(did)

    engine = EngineModule()
    res = engine.run(data, bot_reply, on_death, bot_spawn, lambda line: None, 0)

    outcome_a = 0.0 if res.winner is None else (1.0 if res.winner == "A" else -1.0)
    longest_a, total_a = pa.tiebreak_stats()
    pa.finish(outcome_a + TIEBREAK_SCALE * (longest_a + total_a) / tile_scale)
    steps = list(pa.steps.values())
    if isinstance(pb, PPOTrajPlayer):
        longest_b, total_b = pb.tiebreak_stats()
        pb.finish(-outcome_a + TIEBREAK_SCALE * (longest_b + total_b) / tile_scale)
        steps += list(pb.steps.values())
    return steps, outcome_a, res


_pool_weights = None


def _pool_init(weights):
    global _pool_weights
    _pool_weights = weights


def _pool_job(job):
    spec, seed, vs_brain = job
    return collect_one(spec, _pool_weights, seed, vs_brain)


def collect_parallel(specs, weights, seed, brain_frac=0.3, workers=None):
    """Returns (list of per-dragon trajectories, each a list[PPOStep] in order -- GAE needs each dragon's own
    sequence intact, not flattened across dragons/games; list of (vs_brain, outcome_a) per game, a cheap progress
    signal (win rate vs Brain) separate from the reward the PPO update actually trains on)."""
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    rng = np.random.default_rng(seed)
    jobs = [(spec, seed + i, bool(rng.random() < brain_frac)) for i, spec in enumerate(specs)]
    trajectories, outcomes = [], []
    with concurrent.futures.ProcessPoolExecutor(workers, initializer=_pool_init, initargs=(weights,)) as pool:
        for (spec, gseed, vs_brain), (steps, outcome_a, res) in zip(jobs, pool.map(_pool_job, jobs, chunksize=1)):
            trajectories.extend(steps)  # steps is already a list of per-dragon lists; keep each one intact
            outcomes.append((vs_brain, outcome_a))
    return trajectories, outcomes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--min-side", type=int, default=10)
    ap.add_argument("--max-side", type=int, default=64)
    ap.add_argument("--seed", type=int, default=800000)
    ap.add_argument("--brain-frac", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    torch.manual_seed(0)
    weights = torch_policy.export_numpy(torch_policy.PolicyNet())  # fresh random actor-critic weights
    specs = [(args.seed + i, args.min_side, args.max_side) for i in range(args.games)]
    t0 = time.perf_counter()
    trajectories, outcomes = collect_parallel(specs, weights, args.seed, args.brain_frac, args.workers)
    wall = time.perf_counter() - t0

    flat = [s for traj in trajectories for s in traj]
    n = len(flat)
    rewards = [s.reward for s in flat]
    vs_brain_outcomes = [o for vs_brain, o in outcomes if vs_brain]
    print(f"{args.games} games, {len(trajectories):,} dragon-trajectories, {n:,} steps, {wall:.1f}s ({n / wall:,.0f} steps/s)")
    print(f"reward: mean {sum(rewards) / n:.4f}  range [{min(rewards):.2f}, {max(rewards):.2f}]")
    if vs_brain_outcomes:
        wins = sum(o == 1 for o in vs_brain_outcomes)
        print(f"vs Brain: {len(vs_brain_outcomes)} games, {wins} wins ({wins / len(vs_brain_outcomes):.1%})")


if __name__ == "__main__":
    main()
