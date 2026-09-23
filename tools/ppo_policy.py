"""Numpy actor-critic policy for fast PPO rollout collection. tools/-only: bot/policy.py stays actor-only for
deployment (the critic is never needed once training is done), but rollouts need it many times per second across
many parallel workers, so this stays numpy (like bot/policy.py) rather than torch, to avoid per-call framework
overhead -- tools/torch_policy.py's PolicyNet is used only for the backward pass during the PPO update itself.

Same architecture, masking and sampling as bot/policy.py's Policy (reusing its `choose()` so the tricky masking
logic has one definition), plus a value estimate from the shared trunk (v = h2 @ Wv + bv, matching
tools/torch_policy.py's value_head) for GAE. Weights come from torch_policy.export_numpy(), which always includes
"wv"/"bv" -- unlike a BC-era checkpoint, which bot/policy.py's Policy can still load since it never looks for them.

The raw h2 @ Wv + bv is denormalized using "value_mu"/"value_sigma" (PopArt running stats, see train_ppo.py's
popart_update()) before being returned as `value` -- this MUST stay raw-reward-scale, since it becomes `s.value`
in a PPOStep, and GAE (train_ppo.py's compute_gae) combines it directly with raw-scale rewards. Defaults to
mu=0/sigma=1 (identity) when absent, matching a BC-era or pre-PopArt checkpoint that was never normalized.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "bot"))
import encoder  # noqa: E402
import proto  # noqa: E402
from policy import choose  # noqa: E402


class PPOPolicy:
    def __init__(self, dragon_id, team, width, height, unit_limit, weights, seed=0):
        self.id, self.team = dragon_id, team
        self.W, self.H, self.limit = width, height, unit_limit
        self.w = weights
        self.rng = np.random.default_rng(seed)

    @classmethod
    def from_init(cls, init_bytes, weights, seed=0):
        dragon_id, team, w, h, limit = proto.parse_init(init_bytes)
        return cls(dragon_id, team, w, h, limit, weights, seed=seed)

    def forward(self, idx, dense):
        w = self.w
        h1 = w["w1"][idx].sum(0) + np.asarray(dense, dtype=np.float32) @ w["wd"] + w["b1"]
        np.maximum(h1, 0, out=h1)
        h2 = h1 @ w["w2"] + w["b2"]
        np.maximum(h2, 0, out=h2)
        raw_value = (h2 @ w["wv"] + w["bv"]).item()  # wv is (H2,1): a shape-(1,) result, not float()'s 0-d requirement
        value = raw_value * float(w.get("value_sigma", 1.0)) + float(w.get("value_mu", 0.0))
        return h2 @ w["wm"] + w["bm"], h2 @ w["ws"] + w["bs"], value

    def decide(self, block):
        t = proto.parse_turn(block)
        idx, dense = encoder.encode(t, self.team, self.id, self.W, self.H, self.limit)
        move_logits, split_logits, value = self.forward(idx, dense)
        action, move_i, move_logp, split_i, split_logp = choose(
            move_logits, split_logits, t, self.rng, self.W, self.H, self.limit)
        return {"action": action, "idx": idx, "dense": dense, "length": t.length, "round": t.rnd,
                "move_i": move_i, "move_logp": move_logp, "split_i": split_i, "split_logp": split_logp,
                "value": value}

    def act(self, block):
        return self.decide(block)["action"]
