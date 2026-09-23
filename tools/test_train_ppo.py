"""Unit tests for tools/train_ppo.py's compute_gae and popart_update -- the smoke test in train_ppo.py's own
main() only checks that a full training run doesn't crash and produces plausible-looking numbers; this checks
the underlying arithmetic against hand-computed values / known invariants, since a sign or indexing error in
either would silently produce a wrong policy gradient or a corrupted value scale without ever raising an
exception.

Run:  .venv\\Scripts\\python.exe tools\\test_train_ppo.py
"""
import pathlib
import sys
from collections import namedtuple

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from train_ppo import compute_gae, popart_update  # noqa: E402

# a minimal stand-in for ppo_rollout.PPOStep: compute_gae only reads .reward, .value, .done
Step = namedtuple("Step", "reward value done")


def traj(rewards, values, dones):
    return [Step(r, v, d) for r, v, d in zip(rewards, values, dones)]


def close(a, b, tol=1e-6):
    return abs(a - b) < tol


def test_matches_monte_carlo_when_lambda_gamma_one():
    # gamma=lam=1, values=0: GAE must reduce to plain discounted-by-nothing returns-to-go.
    t = traj([1, 2, 3], [0, 0, 0], [False, False, True])
    adv, ret = compute_gae(t, gamma=1.0, lam=1.0)
    assert all(close(a, b) for a, b in zip(ret, [6, 5, 3])), ret
    assert all(close(a, b) for a, b in zip(adv, [6, 5, 3])), adv  # adv == ret when the baseline is 0
    print("ok: gamma=lam=1, values=0 matches plain Monte Carlo returns-to-go [6, 5, 3]")


def test_returns_are_value_independent_when_lambda_gamma_one():
    # With gamma=lam=1 the value terms must telescope out of the *return* exactly, regardless of the value
    # function -- only the advantage (return - value) should change. A real invariant of GAE, not a coincidence.
    t0 = traj([1, 2, 3], [0, 0, 0], [False, False, True])
    t1 = traj([1, 2, 3], [5, 5, 5], [False, False, True])
    _, ret0 = compute_gae(t0, gamma=1.0, lam=1.0)
    _, ret1 = compute_gae(t1, gamma=1.0, lam=1.0)
    assert all(close(a, b) for a, b in zip(ret0, ret1)), (ret0, ret1)
    adv1, _ = compute_gae(t1, gamma=1.0, lam=1.0)
    assert all(close(a, b) for a, b in zip(adv1, [1, 0, -2])), adv1  # return - value at each step
    print("ok: returns are value-independent at gamma=lam=1; advantages correctly reflect the baseline")


def test_discounting():
    # gamma=0.5, lam=1, values=0, constant reward 1 -- hand-computed backward pass.
    t = traj([1, 1, 1], [0, 0, 0], [False, False, True])
    adv, ret = compute_gae(t, gamma=0.5, lam=1.0)
    assert close(adv[2], 1.0) and close(adv[1], 1.5) and close(adv[0], 1.75), adv
    assert all(close(a, b) for a, b in zip(ret, adv)), "values are 0, so returns == advantages here too"
    print("ok: discounting (gamma=0.5) matches hand-computed [1.75, 1.5, 1.0]")


def test_terminal_step_does_not_bootstrap_past_the_end():
    # The done flag on the LAST step must zero out ITS bootstrap (there's nothing beyond the trajectory to
    # continue into) -- regardless of how large that final value estimate is, done[-1]=True means next_v=0 for
    # it specifically, so its return is just its own reward (not reward + a phantom future value).
    t = traj([0, 0, 5], [0, 0, 999], [False, False, True])
    adv, ret = compute_gae(t, gamma=0.9, lam=0.9)
    assert close(adv[-1], 5.0 - 999), adv  # delta = reward - value (next_v forced to 0 by done)
    assert close(ret[-1], 5.0), ret  # return = adv + value telescopes back to just the terminal reward
    print("ok: the trajectory's last (done=True) step doesn't bootstrap past its own end")


def test_popart_rescale_invariant():
    # A stats update must leave the DENORMALIZED prediction (raw = normalized*sigma + mu) unchanged for any
    # input, or `s.value` recorded by tools/ppo_policy.py right after a checkpoint update would silently jump
    # scale -- exactly the class of bug GAE has no tolerance for (it mixes s.value directly with raw-scale
    # rewards; see train_ppo.py's module docstring, Phase 1 tuning note).
    torch.manual_seed(0)
    head = torch.nn.Linear(8, 1)
    h = torch.randn(20, 8)
    mu, sigma = 0.0, 1.0
    with torch.no_grad():
        raw_before = (head(h).squeeze(-1) * sigma + mu).numpy()

    returns = np.array([3.0, -1.0, 5.0, 2.0, 0.0, 4.0, -2.0, 1.0] * 4, dtype=np.float32)  # arbitrary, nonzero mean
    new_mu, new_sigma = popart_update(head, mu, sigma, returns, beta=0.5)
    assert not close(new_mu, mu) and not close(new_sigma, sigma), "test is vacuous unless stats actually moved"

    with torch.no_grad():
        raw_after = (head(h).squeeze(-1) * new_sigma + new_mu).numpy()
    assert np.allclose(raw_before, raw_after, atol=1e-4), (raw_before, raw_after)
    print(f"ok: popart_update moved (mu, sigma) ({mu}, {sigma}) -> ({new_mu:.3f}, {new_sigma:.3f}) but left "
          f"denormalized predictions unchanged (max diff {np.max(np.abs(raw_before - raw_after)):.2e})")


def test_popart_tracks_batch_statistics():
    # Sanity check on top of the invariant above: repeated updates on a stationary distribution of returns should
    # converge mu/sigma toward that distribution's real mean/std, not just preserve outputs while drifting away
    # from tracking anything meaningful.
    head = torch.nn.Linear(4, 1)
    rng = np.random.default_rng(0)
    mu, sigma = 0.0, 1.0
    true_mean, true_std = 7.0, 2.5
    for _ in range(200):
        returns = rng.normal(true_mean, true_std, size=512).astype(np.float32)
        mu, sigma = popart_update(head, mu, sigma, returns, beta=0.2)
    assert abs(mu - true_mean) < 0.5, mu
    assert abs(sigma - true_std) < 0.5, sigma
    print(f"ok: popart_update converges to the batch distribution's mean/std ({mu:.2f}, {sigma:.2f} "
          f"vs true {true_mean}, {true_std})")


if __name__ == "__main__":
    test_matches_monte_carlo_when_lambda_gamma_one()
    test_returns_are_value_independent_when_lambda_gamma_one()
    test_discounting()
    test_terminal_step_does_not_bootstrap_past_the_end()
    test_popart_rescale_invariant()
    test_popart_tracks_batch_statistics()
