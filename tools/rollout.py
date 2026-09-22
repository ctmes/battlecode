"""Throughput harness for stage 4: bot/policy.py self-play on the real engine, run across many parallel worker
processes, to check the plan's dragon-turns/s throughput gate (~20k aggregate, a target the plan left unmeasured).

    .venv\\Scripts\\python.exe tools\\rollout.py [--games 64] [--max-side 64] [--seed 400000] [--workers N]

Throughput here is dominated by encode() + the policy's forward pass per bot_reply call, which costs the same
whether a game currently has 2 dragons or 40 (the engine calls bot_reply once per dragon-turn regardless of team
size, see context.txt's per-round order) -- so default-size generated maps are a fair proxy for the 30-60 dragon
games the plan names as the eventual training regime, not only games that happen to reach that population.
"""
import argparse
import concurrent.futures
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bot"))
sys.path.insert(0, str(ROOT / "tools"))
import mapgen  # noqa: E402
from policy import Policy, random_weights  # noqa: E402
from unswbc.engine import EngineModule  # noqa: E402

_engine = None
_weights = None


def _init(weights_seed):
    global _engine, _weights
    _engine = EngineModule()
    _weights = random_weights(weights_seed)  # same weights in every worker: deterministic, no IPC of the arrays


class PolicyPlayer:
    def __init__(self, seed):
        self.seed = seed
        self.dragons = {}
        self.turns = 0

    def spawn(self, did, init):
        self.dragons[did] = Policy.from_init(init, _weights, seed=self.seed * 1000 + did)

    def reply(self, did, block):
        self.turns += 1
        try:
            return self.dragons[did].act(block)
        except Exception:  # noqa: BLE001 - a crash shouldn't wedge the whole rollout batch
            return b""


def play_one(spec):
    seed, lo, hi = spec
    data = mapgen.generate(seed, lo, hi).dumps().encode()
    pa, pb = PolicyPlayer(seed), PolicyPlayer(seed + 1_000_000)
    owner = {}

    def bot_spawn(did, init):
        team = next(line for line in init.split(b"\n") if line.startswith(b"TEAM")).split()[1]
        p = pa if team == b"A" else pb
        owner[did] = p
        p.spawn(did, init)

    def bot_reply(did, block):
        return owner[did].reply(did, block)

    t0 = time.perf_counter()
    res = _engine.run(data, bot_reply, None, bot_spawn, lambda line: None, 0)
    cost = time.perf_counter() - t0
    return pa.turns + pb.turns, res.rounds + 1, cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=64)
    ap.add_argument("--min-side", type=int, default=10)
    ap.add_argument("--max-side", type=int, default=64)
    ap.add_argument("--seed", type=int, default=400000)
    ap.add_argument("--weights-seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    specs = [(args.seed + i, args.min_side, args.max_side) for i in range(args.games)]
    workers = args.workers or max(1, (os.cpu_count() or 2) - 2)
    t0 = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(workers, initializer=_init, initargs=(args.weights_seed,)) as pool:
        results = list(pool.map(play_one, specs, chunksize=1))
    wall = time.perf_counter() - t0

    turns = sum(r[0] for r in results)
    rounds = sum(r[1] for r in results)
    cpu_seconds = sum(r[2] for r in results)  # sum of each game's own wall time: a single-worker-equivalent rate
    print(f"{args.games} games, {workers} workers, {turns:,} dragon-turns, {rounds:,} rounds, wall {wall:.1f}s")
    print(f"aggregate throughput: {turns / wall:,.0f} dragon-turns/s  (gate: ~20,000)")
    print(f"per-worker rate: {turns / cpu_seconds:,.0f} dragon-turns/s (aggregate / {workers} ~= "
          f"{turns / wall / workers:,.0f} suggests pool overhead if far below the per-worker rate)")


if __name__ == "__main__":
    main()
