"""Head-to-head: the live Brain (bot/brain.py) vs. manual-heuristics/ (a previous from-scratch submission),
with the same stats eval_vs_policy.py reports -- including the longest currently-ALIVE dragon, the stat that
actually decides a round-500 game (not total length, see tools/tune.py's grower_brain benchmark notes).

    .venv\\Scripts\\python.exe tools\\eval_vs_manual_heuristics.py [--games 200] [--max-side 64] [--workers N]
    .venv\\Scripts\\python.exe tools\\eval_vs_manual_heuristics.py --brain-params tools\\tuned\\full_run1_avg8.json
"""
import argparse
import concurrent.futures
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bot"))
sys.path.insert(0, str(ROOT / "tools"))
import arena  # noqa: E402
import mapgen  # noqa: E402
import opponents  # noqa: E402  (registers "manual_heuristics" as a side effect of import)
from brain import Brain  # noqa: E402
from unswbc.engine import EngineModule  # noqa: E402

_engine = None
_brain_params = None


def _init(brain_params):
    global _engine, _brain_params
    _engine = EngineModule()
    _brain_params = brain_params


class BrainPlayer:
    name = "brain"

    def __init__(self, seed):
        self.dragons = {}
        self.cur_len = {}

    def spawn(self, did, init):
        self.dragons[did] = Brain.from_init(init, _brain_params)

    def reply(self, did, block):
        try:
            action = self.dragons[did].act(block)
            body = self.dragons[did].body
            self.cur_len[did] = len(body) if body else 0
            return action
        except Exception:  # noqa: BLE001
            return b""


class ManualHeuristicsPlayer:
    name = "manual_heuristics"

    def __init__(self, seed):
        self.dragons = {}
        self.cur_len = {}

    def spawn(self, did, init):
        cls = opponents.OPPONENTS["manual_heuristics"]._brain_cls
        self.dragons[did] = cls.from_init(init)

    def reply(self, did, block):
        try:
            action = self.dragons[did].act(block)
            body = self.dragons[did].body
            self.cur_len[did] = len(body) if body else 0
            return action
        except Exception:  # noqa: BLE001
            return b""


def alive_max(player, dead_ids):
    alive = [ln for did, ln in player.cur_len.items() if did not in dead_ids]
    return max(alive) if alive else 0


def play_one(args):
    spec, side, seed = args
    seed_map, lo, hi = spec
    data = mapgen.generate(seed_map, lo, hi).dumps().encode()
    brain, other = BrainPlayer(seed), ManualHeuristicsPlayer(seed + 1)
    pa, pb = (brain, other) if side == "A" else (other, brain)
    res, deaths, errors = arena.play(_engine, data, pa, pb)
    brain_dead = {did for name, did, _, _ in deaths if name == "brain"}
    other_dead = {did for name, did, _, _ in deaths if name == "manual_heuristics"}
    my_len = res.a_length if side == "A" else res.b_length
    op_len = res.b_length if side == "A" else res.a_length
    my_dr = res.a_dragons if side == "A" else res.b_dragons
    op_dr = res.b_dragons if side == "A" else res.a_dragons
    my_max = alive_max(brain, brain_dead)
    op_max = alive_max(other, other_dead)
    score = 0.5 if res.winner is None else float(res.winner == side)
    return score, my_len, op_len, my_dr, op_dr, my_max, op_max, len(errors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--max-side", type=int, default=64)
    ap.add_argument("--seed", type=int, default=700000)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--brain-params", default=None, help="tuned .json to give Brain (default: brain.DEFAULTS)")
    args = ap.parse_args()

    brain_params = json.loads(pathlib.Path(args.brain_params).read_text())["params"] if args.brain_params else None
    n_maps = (args.games + 1) // 2
    jobs = [((args.seed + i, 10, args.max_side), side, args.seed + i) for i in range(n_maps) for side in "AB"]
    workers = args.workers or max(1, (os.cpu_count() or 2) - 2)

    t0 = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(workers, initializer=_init, initargs=(brain_params,)) as pool:
        results = list(pool.map(play_one, jobs, chunksize=1))

    scores = [r[0] for r in results]
    wins, draws = sum(s == 1 for s in scores), sum(s == 0.5 for s in scores)
    losses = len(scores) - wins - draws
    score = sum(scores) / len(scores)
    my_len = sum(r[1] for r in results) / len(results)
    op_len = sum(r[2] for r in results) / len(results)
    my_dr = sum(r[3] for r in results) / len(results)
    op_dr = sum(r[4] for r in results) / len(results)
    my_max = sum(r[5] for r in results) / len(results)
    op_max = sum(r[6] for r in results) / len(results)
    errors = sum(r[7] for r in results)
    print(f"Brain{' (' + args.brain_params + ')' if args.brain_params else ' (DEFAULTS)'} "
          f"vs manual-heuristics/, {len(scores)} games, {time.perf_counter() - t0:.0f}s")
    print(f"score: {score:.1%}  W{wins} D{draws} L{losses}  errors {errors}")
    print(f"avg total length -- brain {my_len:.1f}  manual_heuristics {op_len:.1f}")
    print(f"avg dragons alive at end -- brain {my_dr:.1f}  manual_heuristics {op_dr:.1f}")
    print(f"avg longest ALIVE dragon (the round-500 tiebreak stat) -- brain {my_max:.1f}  manual_heuristics {op_max:.1f}")


if __name__ == "__main__":
    main()
