"""One-off migration: an evolved/BC/PPO checkpoint from before the sprint+sonar action-space change (bot/policy.py,
2026-09-25) -> a checkpoint compatible with the new one, warm-starting everything that still means the same thing
and freshly (randomly) initializing only what's genuinely new.

What changed and how each piece is handled:
  w1, b1, w2, b2, ws, bs   unchanged shapes, unchanged meaning -> copied verbatim.
  wm, bm   (move head) 3 options -> 3 * len(SPRINT_LENGTHS) (9), laid out `length_idx * 3 + rel_dir_idx` so
           indices 0-2 (length index 0, i.e. a single step) are *exactly* the old 3-option head, in the same
           order (see bot/policy.py's module docstring) -> old wm/bm copied into new columns 0-2; columns 3-8
           (the new sprint-length options) start at exactly ZERO, not bot/policy.py's usual random init scale.
  wd       (dense-feature weights) old dense features 0-3 (length, units, round, msg-count) are unchanged;
           old feature 4 was a crude, barely-used placeholder ("first sonar value / 65536") that the new encoder
           replaces outright with a properly decoded 5-scalar block (features 4-8) -> old wd rows 0-3 copied,
           row 4 (the placeholder) is NOT reused for anything, new rows 4-8 start at exactly ZERO too.

Why zero *weights*, not bot/policy.py's normal random-init scale (measured, not assumed -- a first version of
this script used the normal scale and it was a real bug): a fresh random projection of h2 produces logits that
compete for softmax probability mass against the trained ones with pure noise -- measured as a 57%->12% collapse
vs Brain even though every old column/row was copied through unchanged.

Why the new *biases* are a large negative constant (NEG_BIAS), not zero (a second real bug, caught the same
way): zero weights make a new option's logit exactly 0 regardless of the input -- but 0 is not automatically
"never selected". Measured on the actual checkpoint this migrates: bm's trained values are tiny (~+-0.05-0.07)
and wm's trained column norms sit close to bot/policy.py's own init scale (~1.0) -- training did not grow these
particular weights to a dramatically larger scale, so a competing logit of exactly 0 is not negligible next to
them, it is comparable. That gave a measurable second collapse (57%->31%) from zero-weight-but-zero-bias alone.
NEG_BIAS makes each new option's logit large and negative *before* CMA-ES ever touches it, so
exp(new - max) is negligible against any plausible trained logit, regardless of that logit's exact scale --
robust to not knowing the trained scale precisely, unlike hand-tuning a "small" weight scale would be. Zero
weights are kept alongside this (not just the negative bias alone): they also keep new columns from responding
to input at all until CMA-ES's own perturbations move them, which is the more predictable place for exploration
to start from.

Either way, CMA-ES (unlike a gradient method) has no trouble growing a parameter away from exactly zero or a
large negative constant -- tools/evolve.py's `initial_c` already gives these dimensions a sensible starting
variance to explore from.
No value head (wv/bv) is migrated -- evolutionary fine-tuning never uses one, see tools/evolve.py.

    .venv\\Scripts\\python.exe tools\\migrate_checkpoint.py tools\\evolved\\run1.npz tools\\evolved\\run1_migrated.npz
"""
import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bot"))
sys.path.insert(0, str(ROOT / "tools"))
import encoder  # noqa: E402
from policy import random_weights  # noqa: E402


NEG_BIAS = -10.0  # exp(-10) ~= 4.5e-5: negligible against any plausible trained logit -- see module docstring


def migrate(old, seed=0):
    new = random_weights(seed, encoder.VOCAB, encoder.DENSE_SIZE)  # only used for correctly-shaped containers
    old_h2, old_n_move = old["wm"].shape
    old_dense = old["wd"].shape[0]
    new["w1"], new["b1"], new["w2"], new["b2"] = old["w1"], old["b1"], old["w2"], old["b2"]
    new["ws"], new["bs"] = old["ws"], old["bs"]
    new["wm"] = np.zeros_like(new["wm"])  # zero weights -- see module docstring
    new["wm"][:, :old_n_move] = old["wm"]
    new["bm"] = np.full_like(new["bm"], NEG_BIAS)  # large-negative bias, NOT zero -- see module docstring
    new["bm"][:old_n_move] = old["bm"]
    keep_dense = min(old_dense, 4)  # only features 0-3 (length/units/round/msg-count) kept meaning; see docstring
    new["wd"] = np.zeros_like(new["wd"])  # zero, not random -- see module docstring
    new["wd"][:keep_dense, :] = old["wd"][:keep_dense, :]
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="the pre-migration checkpoint (.npz)")
    ap.add_argument("dst", help="where to write the migrated checkpoint")
    ap.add_argument("--seed", type=int, default=0, help="seed for the freshly-initialized (new) weights")
    args = ap.parse_args()

    old = dict(np.load(args.src))
    if old["wm"].shape[1] != 3 or old["wd"].shape[0] != 5:
        raise ValueError(f"{args.src!r} doesn't look like a pre-migration checkpoint "
                          f"(wm has {old['wm'].shape[1]} move options, wd has {old['wd'].shape[0]} dense "
                          f"features -- expected 3 and 5) -- already migrated?")
    new = migrate(old, args.seed)
    np.savez(args.dst, **new)
    print(f"migrated {args.src} ({old['wm'].shape[1]} move options, {old['wd'].shape[0]} dense features) "
          f"-> {args.dst} ({new['wm'].shape[1]} move options, {new['wd'].shape[0]} dense features)")


if __name__ == "__main__":
    main()
