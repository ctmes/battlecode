"""Smoke test for tools/ppo_policy.py: real engine games, self-play, untrained (fresh torch-exported) weights.
Checks it never crashes, every action is well-formed, and decide() agrees with bot/policy.py's Policy on the
action distribution for the same weights/inputs (they share `choose()`; this exercises the value head + forward()
path around it, which bot/policy.py's Policy doesn't have).

Run:  .venv\\Scripts\\python.exe tools\\test_ppo_policy.py
"""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bot"))
sys.path.insert(0, str(ROOT / "tools"))
import proto  # noqa: E402
import torch_policy  # noqa: E402
from ppo_policy import PPOPolicy  # noqa: E402
from unswbc.engine import EngineModule  # noqa: E402

MAPS = sorted((ROOT / "maps").glob("*.map"))
VALID_ACTIONS = {b"MOVE N", b"MOVE E", b"MOVE S", b"MOVE W"}


def play(engine, data, weights):
    dragons, errors, values = {}, [], []

    def bot_spawn(did, init):
        dragons[did] = PPOPolicy.from_init(init, weights, seed=did)

    def bot_reply(did, block):
        d = dragons[did].decide(block)
        values.append(d["value"])
        action = d["action"]
        head, _, rest = action.partition(b" ")
        if head == b"SPLIT":
            t = proto.parse_turn(block)
            k = int(rest)
            assert 2 <= k <= t.length - 2, f"split length {k} out of range for length {t.length}"
        elif action.rstrip(b"\n") not in VALID_ACTIONS:
            errors.append(f"dragon {did}: malformed action {action!r}")
        return action

    res = engine.run(data, bot_reply, None, bot_spawn, lambda line: errors.append(f"notice: {line}"), 0)
    return errors, values


def test_no_crashes_and_sane_values():
    engine = EngineModule()
    weights = torch_policy.export_numpy(torch_policy.PolicyNet())  # fresh random weights, with a real value head
    all_values = []
    for m in MAPS:
        errors, values = play(engine, m.read_bytes(), weights)
        assert not errors, f"{m.name}: {errors[:3]}"
        all_values.extend(values)
    assert all(v == v for v in all_values), "value estimate was NaN"  # v == v is False for NaN
    print(f"ok: {len(MAPS)} bundled maps, no crashes, every action well-formed, "
          f"{len(all_values)} value estimates, range [{min(all_values):.2f}, {max(all_values):.2f}]")


def test_decide_matches_policy_choose():
    """PPOPolicy and bot.policy.Policy must sample the identical action given identical logits + rng state --
    they share choose(), this just checks nothing was lost wiring the 3rd (value) output through."""
    from policy import Policy

    weights = torch_policy.export_numpy(torch_policy.PolicyNet())
    bc_style = {k: v for k, v in weights.items() if k not in ("wv", "bv")}  # what Policy actually needs
    engine = EngineModule()
    data = MAPS[0].read_bytes()

    def run_with(player_cls, w):
        dragons, actions = {}, []

        def spawn(did, init):
            dragons[did] = player_cls.from_init(init, w, seed=did)

        def reply(did, block):
            a = dragons[did].act(block)
            actions.append(a)
            return a

        engine.run(data, reply, None, spawn, lambda line: None, 0)
        return actions

    ppo_actions = run_with(PPOPolicy, weights)
    plain_actions = run_with(Policy, bc_style)
    assert ppo_actions == plain_actions, "PPOPolicy and Policy diverged despite identical logits/rng"
    print(f"ok: {len(ppo_actions)} actions identical between PPOPolicy and Policy given the same weights/seed")


def test_value_denormalization():
    """A checkpoint's "value_mu"/"value_sigma" (PopArt running stats, see train_ppo.py's popart_update) must be
    applied to the raw h2 @ wv + bv output before it's used as s.value -- GAE needs raw-reward-scale values, not
    whatever scale the value head was actually trained to output. Checks both that non-identity stats are
    applied, and that omitting them defaults to identity (mu=0, sigma=1), matching a BC-era or pre-PopArt
    checkpoint that was never normalized."""
    weights = torch_policy.export_numpy(torch_policy.PolicyNet())
    dense = [0.1, 0.2, 0.3, 0.0, 0.0]  # encoder.DENSE_SIZE == 5
    p = PPOPolicy(dragon_id=0, team=b"A", width=16, height=16, unit_limit=8, weights=weights, seed=0)
    _, _, raw_value = p.forward([], dense)  # no value_mu/value_sigma keys -> identity

    scaled_weights = dict(weights)
    scaled_weights["value_mu"] = np.float32(10.0)
    scaled_weights["value_sigma"] = np.float32(3.0)
    p_scaled = PPOPolicy(dragon_id=0, team=b"A", width=16, height=16, unit_limit=8, weights=scaled_weights, seed=0)
    _, _, scaled_value = p_scaled.forward([], dense)

    assert abs(scaled_value - (raw_value * 3.0 + 10.0)) < 1e-4, (scaled_value, raw_value)
    print(f"ok: value_mu/value_sigma denormalize the raw critic output ({raw_value:.3f} -> {scaled_value:.3f} "
          f"at mu=10, sigma=3); absent keys default to identity")


if __name__ == "__main__":
    test_no_crashes_and_sane_values()
    test_decide_matches_policy_choose()
    test_value_denormalization()
