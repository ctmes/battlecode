"""Head-to-head evaluation for a BC-trained policy: win rate vs. an untrained random-weight policy (sanity floor)
and vs. Brain, the teacher it was trained to imitate (the real signal -- BC won't beat its own teacher, but how
close it gets, and whether it stops crashing/dying randomly, shows whether training actually worked).

    .venv\\Scripts\\python.exe tools\\eval_bc.py --weights tools\\bc\\bc.npz --games 100 --workers 14
"""
import argparse
import concurrent.futures
import os
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bot"))
sys.path.insert(0, str(ROOT / "tools"))
import mapgen  # noqa: E402
import proto  # noqa: E402
from brain import Brain  # noqa: E402
from policy import Policy, random_weights  # noqa: E402
from unswbc.engine import EngineModule  # noqa: E402

_bc_weights = None
_rand_weights = None


def _init(weights_path, rand_seed):
    global _bc_weights, _rand_weights
    _bc_weights = dict(np.load(weights_path))
    _rand_weights = random_weights(rand_seed)


class BCPlayer:
    def __init__(self, seed):
        self.seed = seed
        self.dragons = {}

    def spawn(self, did, init):
        self.dragons[did] = Policy.from_init(init, _bc_weights, seed=self.seed * 1000 + did)

    def reply(self, did, block):
        try:
            return self.dragons[did].act(block)
        except Exception:  # noqa: BLE001
            return b""


class RandomPlayer(BCPlayer):
    def spawn(self, did, init):
        self.dragons[did] = Policy.from_init(init, _rand_weights, seed=self.seed * 1000 + did)


class BrainPlayer:
    def __init__(self, seed):
        self.seed = seed
        self.dragons = {}

    def spawn(self, did, init):
        self.dragons[did] = Brain.from_init(init)

    def reply(self, did, block):
        return self.dragons[did].act(block)


def play_one(args):
    spec, side, kind, seed = args
    seed_map, lo, hi = spec
    data = mapgen.generate(seed_map, lo, hi).dumps().encode()
    bc = BCPlayer(seed)
    other = RandomPlayer(seed + 1) if kind == "random" else BrainPlayer(seed + 1)
    pa, pb = (bc, other) if side == "A" else (other, bc)
    owner = {}

    def bot_spawn(did, init):
        team = next(line for line in init.split(b"\n") if line.startswith(b"TEAM")).split()[1]
        p = pa if team == b"A" else pb
        owner[did] = p
        p.spawn(did, init)

    def bot_reply(did, block):
        return owner[did].reply(did, block)

    engine = EngineModule()
    res = engine.run(data, bot_reply, None, bot_spawn, lambda line: None, 0)
    win = res.winner
    outcome = 0.5 if win is None else float(win == side)
    my_len = res.a_length if side == "A" else res.b_length
    my_dr = res.a_dragons if side == "A" else res.b_dragons
    return outcome, my_len, my_dr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "tools" / "bc" / "bc.npz"))
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--min-side", type=int, default=10)
    ap.add_argument("--max-side", type=int, default=64)
    ap.add_argument("--seed", type=int, default=700000)  # held out from bc_data.py's 600000-600059
    ap.add_argument("--rand-seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    workers = args.workers or max(1, (os.cpu_count() or 2) - 2)
    n_maps = args.games // 2
    specs = [(args.seed + i, args.min_side, args.max_side) for i in range(n_maps)]

    for kind in ("random", "brain"):
        jobs = [(spec, side, kind, args.seed + i) for i, spec in enumerate(specs) for side in ("A", "B")]
        t0 = time.perf_counter()
        with concurrent.futures.ProcessPoolExecutor(workers, initializer=_init,
                                                      initargs=(args.weights, args.rand_seed)) as pool:
            results = list(pool.map(play_one, jobs, chunksize=1))
        wall = time.perf_counter() - t0
        outcomes = [r[0] for r in results]
        lens = [r[1] for r in results]
        wins = sum(o == 1 for o in outcomes)
        draws = sum(o == 0.5 for o in outcomes)
        losses = sum(o == 0 for o in outcomes)
        print(f"BC vs {kind}: {len(results)} games ({wall:.1f}s)  "
              f"win {wins} draw {draws} loss {losses}  score {sum(outcomes) / len(outcomes):.1%}  "
              f"mean my_len {sum(lens) / len(lens):.1f}")


if __name__ == "__main__":
    main()
