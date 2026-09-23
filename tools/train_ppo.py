"""PPO fine-tuning: starts from a BC checkpoint (tools/bc/bc.npz), collects self-play (+ vs-Brain) rollouts with
tools/ppo_rollout.py, computes GAE per dragon-trajectory, and runs clipped-PPO minibatch updates with a
KL-to-teacher penalty (teacher = the frozen BC net, per the plan's "PPO fine-tune with a KL-to-teacher penalty").

Scope note on the plan's "League" (past checkpoints + the heuristic bot + occasional exploiters, local Elo): this
implementation only has self-play (mirror) + Brain, chosen per game by --brain-frac. Past-checkpoint opponents,
exploiters and Elo tracking are not built -- a deliberate v1 scope, not an oversight, since each is its own design
surface (a checkpoint pool policy, what makes something an "exploiter", a rating update rule) better done once
there's a first trained policy to build the pool from. --out saves periodic checkpoints, which is what a
past-checkpoint league would sample from later.

Critic caveat (also the plan's own words): the value head sees the same per-dragon egocentric view as the actor,
not "the union of the team's views" the plan describes as more accurate -- that needs a custom simulator to
assemble a full-team state. Revisit if value learning looks like the bottleneck (e.g. persistently high explained
variance error, or GAE producing very noisy advantages).

Two real bugs, found and fixed after runs that looked like "PPO destroys the policy" (both confirmed by
measurement, not just reasoning -- see the battlecode-stage4-rl-environment memory note for the full trail):
1. A credit-assignment bug in tools/ppo_rollout.py's (and trajectory.py's) `finish()`: it only applied the
   team's terminal win/loss outcome to dragons still alive at game end, so a dragon that died early dodged its
   team's eventual loss entirely, averaging ~-0.05 reward vs ~-0.4 for surviving to see a likely loss -- a huge,
   easily-learned incentive to suicide that had nothing to do with reward *shaping* constants. Now fixed: every
   dragon that ever played gets the terminal reward on its own last step.
2. The value head starts randomly initialised (a BC checkpoint has no critic) and, measured on real rollouts
   right before any training, has an *explained variance of -2.45* against the actual GAE returns -- worse than
   predicting a constant, because it over-predicts (mean 2.29 vs actual 1.35) with more spread than the returns
   themselves (std 1.74 vs 0.99). Since the PPO policy gradient is weighted by (return - value), the very first
   policy update was trained on close to noise. `value_warmup()` below fits the critic alone (actor frozen, via
   an optimizer over `net.value_head.parameters()` only) on an initial rollout before any policy gradient step
   touches it -- measured to bring explained variance from -2.45 to +0.015 after just one epoch of joint
   training, so a dedicated warmup should get it into a useful range before the actor is touched at all.

Phase 1 tuning (2026-09-23, see docs/rl-improvement-plan.md): with those two bugs and a third (split-discouraging
reward shaping, fixed in trajectory.py/ppo_rollout.py) out of the way, training is stable but not yet beating the
BC baseline it started from (27% vs Brain best checkpoint, vs BC's own 31%). Three changes here are grounded in
Yu et al., "The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games" (arXiv:2103.01955), the
standard empirical study of PPO hyperparameters in exactly this setting (parameter-shared actor-critic,
non-stationary multi-agent self-play):
1. `--batch-size` default raised 4096 -> 131072. The paper found going from 1 minibatch (full-batch updates) to
   just 4 minibatches per epoch made MAPPO fail to solve any of 23 SMAC maps; 1 minibatch was best on 22/23. The
   old default, sliced out of a rollout that can be hundreds of thousands of steps, plausibly produced hundreds
   of small, destabilizing minibatches per iteration. `n`/minibatch-count are now printed every iteration so this
   is visible, not assumed.
2. `--clip-eps` default lowered 0.2 -> 0.1. The paper found 0.2-0.5 measurably worse than smaller values under
   multi-agent non-stationarity (the mirror self-play opponent moving too); 0.2 was already at the bad edge.
3. Value-target (PopArt) normalization added, per the paper's finding that it "never hurts, often helps
   significantly" when return scale varies a lot across a batch (true here: episode length alone ranges from a
   handful of steps to 500 rounds). Implemented as the analytic-rescale form from Van Hasselt et al. 2016 (not a
   naive normalize-and-forget): `popart_update()` below adjusts `value_head`'s weight/bias in lockstep with every
   running-stats update so that its *denormalized* (raw-reward-scale) output for any given input is unchanged by
   the update. This specific form matters here: `tools/ppo_policy.py`'s numpy critic denormalizes its raw output
   using the same running mu/sigma (now saved in every checkpoint as "value_mu"/"value_sigma"), and that raw,
   denormalized number becomes `s.value` in a PPOStep -- which GAE combines directly with raw-scale rewards
   (`delta = reward + gamma*next_v - value`). A value head trained to output *normalized* numbers without the
   corresponding denormalization at collection time would silently feed GAE values on the wrong scale -- exactly
   the class of quiet, hard-to-notice bug this project has already hit three times via reward shaping. See
   `test_train_ppo.py`'s `test_popart_rescale_invariant` for the property this depends on.
   `explained_variance` is now also printed every iteration (before this iteration's update, reusing the
   already-collected rollout `s.value`s, and after, via one extra forward pass), not just at warmup -- cheap, and
   gives visibility into exactly when a run starts degrading instead of finding out only at the end.

    .venv\\Scripts\\python.exe tools\\train_ppo.py --init tools\\bc\\bc.npz --iters 50 --games-per-iter 64
"""
import argparse
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "bot"))
sys.path.insert(0, str(ROOT))
import torch_policy  # noqa: E402
from ppo_rollout import collect_parallel  # noqa: E402


def compute_gae(traj, gamma, lam):
    """traj: list[PPOStep] for one dragon, temporal order. Every trajectory from ppo_rollout.py ends with
    done=True (a dragon either dies or is force-marked done at game end -- see PPOTrajPlayer.die/finish), so the
    backward pass never needs to bootstrap past the end of what it collected."""
    n = len(traj)
    values = [s.value for s in traj]
    adv = [0.0] * n
    gae = 0.0
    for t in reversed(range(n)):
        next_v = 0.0 if traj[t].done else values[t + 1]
        delta = traj[t].reward + gamma * next_v - values[t]
        gae = delta + gamma * lam * (0.0 if traj[t].done else 1.0) * gae
        adv[t] = gae
    returns = [a + v for a, v in zip(adv, values)]
    return adv, returns


def build_batch(trajectories, gamma, lam):
    idx_lists, dense, move_i, move_logp, split_i, split_logp, adv, ret = [], [], [], [], [], [], [], []
    for traj in trajectories:
        if not traj:
            continue
        a, r = compute_gae(traj, gamma, lam)
        for s, ai, ri in zip(traj, a, r):
            idx_lists.append(s.idx)
            dense.append(s.dense)
            move_i.append(-1 if s.move_i is None else s.move_i)
            move_logp.append(0.0 if s.move_logp is None else s.move_logp)
            split_i.append(s.split_i)
            split_logp.append(s.split_logp)
            adv.append(ai)
            ret.append(ri)
    adv = np.asarray(adv, dtype=np.float32)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return {
        "idx_lists": idx_lists, "dense": np.asarray(dense, dtype=np.float32),
        "move_i": np.asarray(move_i, dtype=np.int64), "move_logp": np.asarray(move_logp, dtype=np.float32),
        "split_i": np.asarray(split_i, dtype=np.int64), "split_logp": np.asarray(split_logp, dtype=np.float32),
        "adv": adv, "ret": np.asarray(ret, dtype=np.float32),
    }


def iterate_minibatches(batch, batch_size, order, device):
    idx_lists = batch["idx_lists"]
    for start in range(0, len(order), batch_size):
        b = order[start:start + batch_size]
        lists = [idx_lists[i] for i in b]
        lengths = np.fromiter((len(x) for x in lists), dtype=np.int64, count=len(lists))
        offsets = np.concatenate(([0], np.cumsum(lengths)))[:-1]
        flat = np.concatenate(lists) if lists else np.zeros(0, dtype=np.int32)
        yield (
            torch.as_tensor(flat.astype(np.int64), device=device),
            torch.as_tensor(offsets, device=device),
            torch.as_tensor(batch["dense"][b], device=device),
            torch.as_tensor(batch["move_i"][b], device=device),
            torch.as_tensor(batch["move_logp"][b], device=device),
            torch.as_tensor(batch["split_i"][b], device=device),
            torch.as_tensor(batch["split_logp"][b], device=device),
            torch.as_tensor(batch["adv"][b], device=device),
            torch.as_tensor(batch["ret"][b], device=device),
        )


def popart_update(value_head, mu, sigma, returns, beta, eps=1e-4):
    """Moves running (mu, sigma) toward this batch's return statistics by `beta`, and analytically rescales
    value_head's weight/bias in place so its *denormalized* output (raw = normalized*sigma + mu) is UNCHANGED by
    the stats update for any given input -- the "ART" half of PopArt (Van Hasselt et al. 2016, "Learning values
    across many orders of magnitude"). Derivation: for W_new = W_old*sigma_old/sigma_new and
    b_new = (sigma_old*b_old + mu_old - mu_new)/sigma_new, denorm_new(W_new@h+b_new) == denorm_old(W_old@h+b_old)
    for any h. Without this, every stats update would discontinuously jump the value head's raw-scale
    predictions, corrupting `s.value` for every dragon-step recorded right after (see module docstring)."""
    batch_mean = float(np.mean(returns))
    batch_sq_mean = float(np.mean(np.square(returns)))
    new_mu = (1 - beta) * mu + beta * batch_mean
    new_second_moment = (1 - beta) * (sigma ** 2 + mu ** 2) + beta * batch_sq_mean
    new_sigma = max((new_second_moment - new_mu ** 2) ** 0.5, eps)
    with torch.no_grad():
        value_head.weight.mul_(sigma / new_sigma)
        value_head.bias.mul_(sigma / new_sigma).add_((mu - new_mu) / new_sigma)
    return new_mu, new_sigma


def predict_values(net, batch, batch_size, device):
    """Raw torch forward-pass value predictions (normalized-space if PopArt stats are non-identity -- caller
    denormalizes) for every step in batch, via one no-grad sweep. Used for explained-variance diagnostics."""
    n = len(batch["idx_lists"])
    with torch.no_grad():
        preds = [net(mb[0], mb[1], mb[2])[2].cpu().numpy()
                 for mb in iterate_minibatches(batch, batch_size, np.arange(n), device)]
    return np.concatenate(preds) if preds else np.zeros(0, dtype=np.float32)


def ppo_loss(net, teacher, mb, clip_eps, vf_coef, ent_coef, kl_coef, mu, sigma):
    flat_idx, offsets, dense, move_i, move_logp_old, split_i, split_logp_old, adv, ret = mb
    move_logits, split_logits, value = net(flat_idx, offsets, dense)
    with torch.no_grad():
        t_move_logits, t_split_logits, _ = teacher(flat_idx, offsets, dense)

    move_logp_all = F.log_softmax(move_logits, dim=-1)
    split_logp_all = F.log_softmax(split_logits, dim=-1)

    move_mask = move_i >= 0
    if move_mask.any():
        mi = move_i[move_mask].clamp(min=0)
        new_move_logp = move_logp_all[move_mask].gather(-1, mi[:, None]).squeeze(-1)
        ratio = torch.exp(new_move_logp - move_logp_old[move_mask])
        a = adv[move_mask]
        move_policy_loss = -torch.min(ratio * a, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * a).mean()
        move_clipfrac = ((ratio - 1).abs() > clip_eps).float().mean().item()
    else:
        move_policy_loss, move_clipfrac = torch.zeros((), device=move_i.device), float("nan")

    new_split_logp = split_logp_all.gather(-1, split_i[:, None]).squeeze(-1)
    s_ratio = torch.exp(new_split_logp - split_logp_old)
    split_policy_loss = -torch.min(s_ratio * adv, torch.clamp(s_ratio, 1 - clip_eps, 1 + clip_eps) * adv).mean()
    split_clipfrac = ((s_ratio - 1).abs() > clip_eps).float().mean().item()

    value_loss = F.mse_loss(value, (ret - mu) / sigma)

    move_entropy = -(move_logp_all.exp() * move_logp_all).sum(-1).mean()
    split_entropy = -(split_logp_all.exp() * split_logp_all).sum(-1).mean()
    entropy = move_entropy + split_entropy

    t_move_logp = F.log_softmax(t_move_logits, dim=-1)
    t_split_logp = F.log_softmax(t_split_logits, dim=-1)
    kl_move = (move_logp_all.exp() * (move_logp_all - t_move_logp)).sum(-1).mean()
    kl_split = (split_logp_all.exp() * (split_logp_all - t_split_logp)).sum(-1).mean()
    kl = kl_move + kl_split

    loss = move_policy_loss + split_policy_loss + vf_coef * value_loss - ent_coef * entropy + kl_coef * kl
    stats = {"move_pl": move_policy_loss.item(), "split_pl": split_policy_loss.item(), "vf": value_loss.item(),
              "ent": entropy.item(), "kl": kl.item(), "move_clip": move_clipfrac, "split_clip": split_clipfrac}
    return loss, stats


def explained_variance(returns, values):
    """1.0 = perfect prediction, 0.0 = no better than predicting the mean return, <0 = worse than that constant."""
    var_r = np.var(returns)
    return 1 - np.var(returns - values) / var_r if var_r > 0 else float("nan")


def value_warmup(net, specs, seed, brain_frac, workers, gamma, lam, epochs, batch_size, lr, device):
    """Fits the critic alone -- an optimizer over just net.value_head.parameters(), so gradients still flow into
    the shared trunk/actor heads but are never applied to them -- on one rollout collected with the actor's
    current (untouched) weights. See the module docstring: a fresh value head measured explained variance -2.45
    before any training, so the very first real policy update would otherwise be driven by that noise."""
    weights = torch_policy.export_numpy(net)
    trajectories, _ = collect_parallel(specs, weights, seed, brain_frac, workers)
    batch = build_batch(trajectories, gamma, lam)
    n = len(batch["idx_lists"])
    values_before = np.array([s.value for traj in trajectories for s in traj])
    ev_before = explained_variance(batch["ret"], values_before)

    opt = torch.optim.Adam(net.value_head.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        order = rng.permutation(n)
        for mb in iterate_minibatches(batch, batch_size, order, device):
            flat_idx, offsets, dense = mb[0], mb[1], mb[2]
            ret = mb[8]
            _, _, value = net(flat_idx, offsets, dense)
            loss = F.mse_loss(value, ret)
            opt.zero_grad()
            loss.backward()
            opt.step()

    ev_after = explained_variance(batch["ret"], predict_values(net, batch, batch_size, device))
    print(f"value warmup: {n:,} steps, {epochs} epochs, explained_variance {ev_before:.4f} -> {ev_after:.4f}")


def export_with_popart(net, mu, sigma):
    """torch_policy.export_numpy() plus the running PopArt stats, so tools/ppo_policy.py's numpy critic can
    denormalize to the same raw scale the value head was actually trained against. Missing/absent on BC-era or
    pre-PopArt checkpoints, which ppo_policy.py treats as mu=0, sigma=1 (identity -- matches this run's own state
    before the first popart_update() call)."""
    weights = torch_policy.export_numpy(net)
    weights["value_mu"] = np.float32(mu)
    weights["value_sigma"] = np.float32(sigma)
    return weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=str(ROOT / "bc" / "bc.npz"), help="starting weights (a BC checkpoint)")
    ap.add_argument("--out", default=str(ROOT / "ppo"))
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--games-per-iter", type=int, default=64)
    ap.add_argument("--min-side", type=int, default=10)
    ap.add_argument("--max-side", type=int, default=64)
    ap.add_argument("--brain-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=900000)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=4, help="PPO epochs per rollout batch")
    ap.add_argument("--batch-size", type=int, default=131072,
                     help="minibatch size; kept large on purpose -- see module docstring's Phase 1 tuning note")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip-eps", type=float, default=0.1)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--kl-coef", type=float, default=0.1)
    ap.add_argument("--popart-beta", type=float, default=0.2,
                     help="EMA rate for the value-target running mean/std (see popart_update); this project runs "
                          "only tens of iterations total, so this is deliberately much faster-moving than the "
                          "~3e-4 typical in literature written for per-minibatch updates over millions of steps")
    ap.add_argument("--warmup-epochs", type=int, default=10,
                     help="epochs to fit the critic alone before any policy update; 0 disables (see module docstring)")
    ap.add_argument("--warmup-lr", type=float, default=1e-3)
    ap.add_argument("--warmup-games", type=int, default=48)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    init_weights = dict(np.load(args.init))
    net = torch_policy.import_numpy(init_weights, device=args.device)
    teacher = torch_policy.import_numpy(init_weights, device=args.device)  # frozen, never updated
    for p in teacher.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    if args.warmup_epochs > 0:
        warmup_seed = args.seed + 9_000_000  # comfortably outside the main loop's seed range, still non-negative
        warmup_specs = [(warmup_seed + i, args.min_side, args.max_side) for i in range(args.warmup_games)]
        value_warmup(net, warmup_specs, warmup_seed, args.brain_frac, args.workers,
                     args.gamma, args.lam, args.warmup_epochs, args.batch_size, args.warmup_lr, args.device)

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"PPO from {args.init}, device={args.device}, {args.games_per_iter} games/iter, "
          f"brain_frac={args.brain_frac}, batch_size={args.batch_size}, clip_eps={args.clip_eps}")

    mu, sigma = 0.0, 1.0  # PopArt running stats; identity until the first popart_update() call below

    for it in range(args.iters):
        t0 = time.perf_counter()
        rollout_weights = export_with_popart(net, mu, sigma)
        specs = [(args.seed + it * 100000 + i, args.min_side, args.max_side) for i in range(args.games_per_iter)]
        trajectories, outcomes = collect_parallel(specs, rollout_weights, args.seed + it * 100000,
                                                    args.brain_frac, args.workers)
        t_collect = time.perf_counter() - t0

        batch = build_batch(trajectories, args.gamma, args.lam)
        n = len(batch["idx_lists"])
        n_mb_per_epoch = -(-n // args.batch_size)  # ceil div
        rng = np.random.default_rng(args.seed + it)

        # ev_pre: how well the critic that actually collected this rollout (pre-update) predicted it -- reuses
        # the already-recorded s.value, no extra forward pass. Then rescale value_head to the new stats.
        values_before = np.array([s.value for traj in trajectories for s in traj], dtype=np.float32)
        ev_pre = explained_variance(batch["ret"], values_before)
        mu, sigma = popart_update(net.value_head, mu, sigma, batch["ret"], args.popart_beta)

        stats_sum, n_mb = {}, 0
        for epoch in range(args.epochs):
            order = rng.permutation(n)
            for mb in iterate_minibatches(batch, args.batch_size, order, args.device):
                loss, stats = ppo_loss(net, teacher, mb, args.clip_eps, args.vf_coef, args.ent_coef, args.kl_coef,
                                        mu, sigma)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                for k, v in stats.items():
                    if v == v:  # skip nan (move stats when a minibatch has no move samples)
                        stats_sum[k] = stats_sum.get(k, 0.0) + v
                n_mb += 1

        raw_preds_after = predict_values(net, batch, args.batch_size, args.device) * sigma + mu
        ev_post = explained_variance(batch["ret"], raw_preds_after)
        t_update = time.perf_counter() - t0 - t_collect

        vs_brain = [o for vs_brain, o in outcomes if vs_brain]
        win_rate = sum(o == 1 for o in vs_brain) / len(vs_brain) if vs_brain else float("nan")
        mean_ret = float(np.mean(batch["ret"]))
        avg = {k: v / max(n_mb, 1) for k, v in stats_sum.items()}
        print(f"iter {it}: {n:,} steps ({n_mb_per_epoch}/epoch x {args.epochs} = {n_mb} minibatches), "
              f"ret {mean_ret:.3f}, ev {ev_pre:.3f}->{ev_post:.3f}, vs_brain_winrate {win_rate:.1%} "
              f"({len(vs_brain)} games), kl {avg.get('kl', 0):.4f}, ent {avg.get('ent', 0):.3f}, "
              f"clip m/s {avg.get('move_clip', 0):.2f}/{avg.get('split_clip', 0):.2f}  "
              f"({t_collect:.1f}s collect + {t_update:.1f}s update)")

        ckpt = out_dir / f"ppo_{it:04d}.npz"
        np.savez(ckpt, **export_with_popart(net, mu, sigma))
        if it == args.iters - 1:
            np.savez(out_dir / "latest.npz", **export_with_popart(net, mu, sigma))

    print(f"done, checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
