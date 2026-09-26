"""Head-to-head: the live Brain (bot/brain.py, what bot/main.py now runs) vs. the previously-shipped trained
Policy (bot/policy.py + bot/weights.bin, what bot/main.py used to run before the 2026-09-27 switch).

Loads weights.bin exactly the way bot/main.py does (flat frombuffer + reshape, not np.load), so this is a true
comparison against what was actually live, not a re-trained or re-shaped stand-in.

    .venv\\Scripts\\python.exe tools\\eval_vs_policy.py [--games 200] [--max-side 64] [--seed 500000] [--workers N]
    .venv\\Scripts\\python.exe tools\\eval_vs_policy.py --brain-params tools\\tuned\\split_volume_run1_avgN.json
"""
import argparse
import concurrent.futures
import json
import os
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bot"))
sys.path.insert(0, str(ROOT / "tools"))
import arena  # noqa: E402
import encoder  # noqa: E402
import mapgen  # noqa: E402
from brain import Brain  # noqa: E402
from policy import H1, H2, N_MOVE, SPLIT_FRACS, Policy  # noqa: E402
from unswbc.engine import EngineModule  # noqa: E402

# Ported from bot/main.py as it was before the 2026-09-27 switch to Brain (git history), so this loads
# bot/weights.bin exactly the way the previously-shipped bot did -- not a re-derived or re-shaped stand-in.
_SHAPES = [
    ("b1", (H1,)), ("b2", (H2,)), ("bm", (N_MOVE,)), ("bs", (len(SPLIT_FRACS) + 1,)),
    ("w1", (encoder.VOCAB, H1)), ("w2", (H1, H2)), ("wd", (encoder.DENSE_SIZE, H1)),
    ("wm", (H2, N_MOVE)), ("ws", (H2, len(SPLIT_FRACS) + 1)),
]


def _load_weights():
    path = ROOT / "bot" / "weights.bin"
    raw = path.read_bytes()
    flat = np.frombuffer(raw, dtype=np.float32)
    out, i = {}, 0
    for name, shape in _SHAPES:
        n = int(np.prod(shape))
        out[name] = flat[i:i + n].reshape(shape)
        i += n
    return out

_engine = None
_weights = None
_brain_params = None


def _init(brain_params):
    global _engine, _weights, _brain_params
    _engine = EngineModule()
    _weights = _load_weights()
    _brain_params = brain_params


class BrainPlayer:
    name = "brain"

    def __init__(self, seed):
        self.dragons = {}
        self.cur_len = {}  # dragon id -> most recently observed length

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


class PolicyPlayer:
    name = "policy"

    def __init__(self, seed):
        self.seed, self.dragons = seed, {}
        self.cur_len = {}  # dragon id -> most recently observed length

    def spawn(self, did, init):
        self.dragons[did] = Policy.from_init(init, _weights, seed=self.seed * 1000 + did)

    def reply(self, did, block):
        try:
            result = self.dragons[did].decide(block)  # .act() discards "length"; decide() has it for free
            self.cur_len[did] = result["length"]
            return result["action"]
        except Exception:  # noqa: BLE001
            return b""


def alive_max(player, dead_ids):
    alive = [ln for did, ln in player.cur_len.items() if did not in dead_ids]
    return max(alive) if alive else 0


def play_one(args):
    spec, side, seed = args
    seed_map, lo, hi = spec
    data = mapgen.generate(seed_map, lo, hi).dumps().encode()
    brain, policy = BrainPlayer(seed), PolicyPlayer(seed + 1)
    pa, pb = (brain, policy) if side == "A" else (policy, brain)
    res, deaths, errors = arena.play(_engine, data, pa, pb)
    brain_dead = {did for name, did, _, _ in deaths if name == "brain"}
    policy_dead = {did for name, did, _, _ in deaths if name == "policy"}
    my_len = res.a_length if side == "A" else res.b_length
    op_len = res.b_length if side == "A" else res.a_length
    my_dr = res.a_dragons if side == "A" else res.b_dragons
    op_dr = res.b_dragons if side == "A" else res.a_dragons
    my_max = alive_max(brain, brain_dead)
    op_max = alive_max(policy, policy_dead)
    score = 0.5 if res.winner is None else float(res.winner == side)
    return score, my_len, op_len, my_dr, op_dr, my_max, op_max, len(errors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--max-side", type=int, default=64)
    ap.add_argument("--seed", type=int, default=500000)
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
          f"vs Policy (bot/weights.bin), {len(scores)} games, {time.perf_counter() - t0:.0f}s")
    print(f"score: {score:.1%}  W{wins} D{draws} L{losses}  errors {errors}")
    print(f"avg total length -- brain {my_len:.1f}  policy {op_len:.1f}")
    print(f"avg dragons alive at end -- brain {my_dr:.1f}  policy {op_dr:.1f}")
    print(f"avg longest ALIVE dragon (the round-500 tiebreak stat) -- brain {my_max:.1f}  policy {op_max:.1f}")


if __name__ == "__main__":
    main()
