"""Behaviour-cloning demonstrations for stage 5: runs bot/brain.py (the tuned heuristic, DEFAULTS) as the teacher
in self-play on the real engine, and for every dragon-turn records the same (idx, dense) features
bot/encoder.py/policy.py would see, labelled with the teacher's actual choice translated into policy.py's action
space (3-way relative move {forward, left, right}; a split bucket 0..len(SPLIT_FRACS), 0 = "don't split").

A teacher move that reverses (relative "back", which policy.py always masks out -- Brain only ever does this when
totally boxed in with no other option, see its own `decide()`) has no representation in that 3-way space, so those
turns are dropped; they are rare and, by design, never something the trained policy could imitate anyway.
A teacher SPLIT is labelled by whichever of SPLIT_FRACS's buckets produces the closest child length to what Brain
actually chose -- Brain isn't restricted to those exact fractions, so this is a lossy relabelling, not an exact
match; tools/train_bc.py's reported split accuracy should be read with that in mind.

    .venv\\Scripts\\python.exe tools\\bc_data.py --games 200 --workers 14 --out tools\\bc\\demo.npz
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
import encoder  # noqa: E402
import mapgen  # noqa: E402
import proto  # noqa: E402
from brain import Brain  # noqa: E402
from policy import SPLIT_FRACS  # noqa: E402
from unswbc.engine import EngineModule  # noqa: E402

LETTERS = proto.LETTERS


def label_action(action, t):
    """Teacher action bytes -> (move_label or None, split_label). move_label is None either for a split turn
    (the move head wasn't acted on) or for an unrepresentable reversal -- label_action's caller tells these
    apart by split_label (>0 only for a real split)."""
    head, _, rest = action.partition(b" ")
    if head == b"SPLIT":
        k = int(rest)
        best = min(range(len(SPLIT_FRACS)),
                   key=lambda i: abs(max(2, min(round(t.length * SPLIT_FRACS[i]), t.length - 2)) - k))
        return None, best + 1
    d = LETTERS.find(rest.strip())
    d0 = t.dir if t.dir >= 0 else 0
    rel = (d - d0) % 4
    return {0: 0, 3: 1, 1: 2}.get(rel), 0  # forward, left, right; rel==2 (back) -> None, dropped by the caller


class Demo:
    """One dragon's demonstration rows for one game."""
    __slots__ = ("idx_lists", "dense", "move", "split")

    def __init__(self):
        self.idx_lists, self.dense, self.move, self.split = [], [], [], []

    def add(self, idx, dense, move_label, split_label):
        self.idx_lists.append(idx)
        self.dense.append(dense)
        self.move.append(-1 if move_label is None else move_label)
        self.split.append(split_label)


def collect_one(spec, seed):
    seed_map, lo, hi = spec
    m = mapgen.generate(seed_map, lo, hi)
    data = m.dumps().encode()
    brains, owner, demo = {}, {}, {}

    def bot_spawn(did, init):
        dragon_id, team, w, h, limit = proto.parse_init(init)
        brains[did] = Brain(dragon_id, team, w, h, limit)
        owner[did] = (team, w, h, limit)
        demo[did] = Demo()

    def bot_reply(did, block):
        t = proto.parse_turn(block)
        action = brains[did].act(block)
        move_label, split_label = label_action(action, t)
        if move_label is None and split_label == 0:
            return action  # an unrepresentable reversal: recorded nowhere, see the module docstring
        team, w, h, limit = owner[did]
        idx, dense = encoder.encode(t, team, did, w, h, limit)
        demo[did].add(idx, dense, move_label, split_label)
        return action

    engine = EngineModule()
    engine.run(data, bot_reply, None, bot_spawn, lambda line: None, 0)
    return list(demo.values())


def _job(spec_and_seed):
    spec, seed = spec_and_seed
    return collect_one(spec, seed)


def collect(specs, seed, workers=None):
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    jobs = [(spec, seed + i) for i, spec in enumerate(specs)]
    demos = []
    with concurrent.futures.ProcessPoolExecutor(workers) as pool:
        for game_demos in pool.map(_job, jobs, chunksize=1):
            demos.extend(game_demos)
    return demos


def save(demos, path):
    idx_lists, dense, move, split = [], [], [], []
    for d in demos:
        idx_lists.extend(d.idx_lists)
        dense.extend(d.dense)
        move.extend(d.move)
        split.extend(d.split)
    offsets = np.zeros(len(idx_lists) + 1, dtype=np.int64)
    for i, idx in enumerate(idx_lists):
        offsets[i + 1] = offsets[i] + len(idx)
    flat_idx = np.empty(offsets[-1], dtype=np.int32)
    for i, idx in enumerate(idx_lists):
        flat_idx[offsets[i]:offsets[i + 1]] = idx
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, flat_idx=flat_idx, offsets=offsets, dense=np.asarray(dense, dtype=np.float32),
                         move=np.asarray(move, dtype=np.int8), split=np.asarray(split, dtype=np.int8))
    return len(idx_lists)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--min-side", type=int, default=10)
    ap.add_argument("--max-side", type=int, default=64)
    ap.add_argument("--seed", type=int, default=600000)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "tools" / "bc" / "demo.npz"))
    args = ap.parse_args()

    specs = [(args.seed + i, args.min_side, args.max_side) for i in range(args.games)]
    t0 = time.perf_counter()
    demos = collect(specs, args.seed, args.workers)
    n = save(demos, pathlib.Path(args.out))
    wall = time.perf_counter() - t0

    move = np.load(args.out)["move"]
    split = np.load(args.out)["split"]
    move_counts = np.bincount(move[move >= 0], minlength=3)
    split_counts = np.bincount(split, minlength=len(SPLIT_FRACS) + 1)
    print(f"{args.games} games, {n:,} samples, {wall:.1f}s ({n / wall:,.0f} samples/s) -> {args.out}")
    print(f"move labels (forward/left/right, only on non-split turns): {move_counts.tolist()}")
    print(f"split labels (0=no-split, then each bucket): {split_counts.tolist()}")


if __name__ == "__main__":
    main()
