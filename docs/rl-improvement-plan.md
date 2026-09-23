# RL Improvement Plan: Getting PPO Past BC

**Status:** draft, 2026-09-23. Written after external research into how competitive/multi-agent game bots
are actually trained (AlphaStar league training, the MAPPO empirical-best-practices paper, Battlesnake and
Agar.io community/academic approaches) and a fresh checkpoint-by-checkpoint audit of the current `tools/ppo/`
run. See "Sources" at the end for everything cited.

## 0. Where things actually stand right now

- **Best real checkpoint: BC (`tools/bc/bc.npz`).** 72% vs an untrained random policy, 31% vs `Brain` (the
  CMA-ES-tuned heuristic teacher), per `tools/eval_bc.py`.
- **PPO is stable but not yet better than BC.** Three real bugs were found and fixed in `tools/trajectory.py` /
  `tools/ppo_rollout.py` / `train_ppo.py` (credit assignment in `finish()`, an unwarmed value head, and
  split-discouraging reward shaping — see `battlecode-stage5-ppo-still-collapsing` memory for the full trail).
  A 4th 25-iteration run with all three fixes no longer collapses: survival length stays flat (~200-220 turns)
  and vs-Brain win rate mildly improves (20% -> 27%, iter 0 -> iter 24, 95% Wilson CIs still overlapping).
  **But 27% is still below BC's own 31%.** PPO has not yet demonstrated it beats plain imitation.
- **Stage-5 gate (per the original plan): >=55% vs Brain over >=400 games.** Not close yet.
- **Conclusion driving this doc:** the training loop is no longer broken, but it also isn't clearly *working*.
  That's a different, more interesting problem — likely a mix of PPO hyperparameters that don't suit this
  setting, a self-play dynamic that isn't pushing skill upward, and reward shaping that's still ad hoc (fixed
  reactively three times now rather than designed to be safe by construction).

This plan is phased by expected-value-per-effort, cheapest/highest-confidence first. Each phase has its own
success criterion and should be validated independently with the existing `tools/eval_bc.py` + Wilson-CI
checkpoint sweep methodology before moving to the next phase — don't stack multiple untested changes into one
run, or a regression will be as hard to root-cause as the last three bugs were.

---

## Phase 1 — PPO hyperparameter fixes (cheap, high-confidence, do first)

Grounded in [Yu et al., "The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games"](https://arxiv.org/abs/2103.01955)
(NeurIPS 2022) — the standard empirical reference for PPO + parameter-sharing + a value function in a
non-stationary multi-agent setting, which is exactly this project's setup. All three findings below are
directly checkable against `tools/train_ppo.py`'s current defaults.

### 1.1 Minibatching is very likely far too aggressive

**Finding:** the MAPPO paper found that splitting a rollout batch into just 4 minibatches per epoch caused
training to **fail to solve any of 23 SMAC maps**; 1 minibatch (full-batch updates) was best on 22/23. Their
guidance: avoid splitting into minibatches at all if you can afford the memory.

**Current state:** `train_ppo.py --batch-size 4096` (default) is a *minibatch* size sliced out of
`iterate_minibatches()`, applied to a rollout that can be very large — an earlier (pre-fix) run collected up to
1.5M steps in a single iteration at `--games-per-iter 64`. Even at a more typical iteration size, this plausibly
means dozens to hundreds of minibatches per epoch x 4 epochs = hundreds of small, destabilizing gradient steps
per PPO iteration.

**Action:**
- Log `n` (total steps) and the resulting minibatch count every iteration (cheap, add to the existing print in
  `main()`'s training loop, `tools/train_ppo.py` ~line 274).
- Re-run a short (5-10 iteration) diagnostic with `--batch-size` raised by 10-50x (start by trying to fit the
  *entire* iteration's batch in one minibatch if GPU memory allows — the RTX 5080 has 16GB, and this is a small
  network, so this should be very feasible), or as few minibatches as memory allows otherwise.
- Validate with the existing `eval_bc.py` sweep before/after.

### 1.2 No value normalization

**Finding:** the MAPPO paper recommends PopArt-style running normalization of value targets (standardize with a
running mean/std, train against the normalized target, denormalize for GAE) and reports it "never hurts, often
improves final performance significantly" — particularly when return scale varies a lot across the batch.

**Current state:** `train_ppo.py`'s `ppo_loss()` does plain `F.mse_loss(value, ret)` on raw, unnormalized
returns. Return scale here plausibly varies a lot: episodes range from a handful of steps (early death) to 500
rounds, and reward mixes small per-step growth shaping with a terminal +-1 and a small tiebreak bonus.

**Action:**
- Add a running mean/std tracker (updated each iteration from `batch["ret"]`) to `train_ppo.py`. Normalize
  `ret` before computing `value_loss`; keep `value` itself trained in normalized space, denormalizing only where
  raw-scale values are needed (nowhere currently, since GAE already runs on raw `traj[t].value` from the
  policy's own forward pass — check this interacts correctly with `compute_gae()`, which expects the *same*
  scale for `values` and `rewards`; simplest correct approach is to normalize returns for the loss only, or
  normalize rewards themselves at the source in `ppo_rollout.py`/`trajectory.py`, whichever proves cleaner once
  implemented — don't guess, check `explained_variance` before/after to confirm it actually helps here before
  keeping it).

### 1.3 Clip epsilon is at the edge of what's already known to be bad

**Finding:** paper reports epsilon in the 0.2-0.5 range produces measurably worse final performance than smaller
values (~0.05-0.1) under multi-agent non-stationarity, because tighter clipping limits how far a policy can move
per update while other agents (here: the mirror self-play opponent) are simultaneously changing too.

**Current state:** `--clip-eps 0.2` (default) — right at the upper edge of the range flagged as suboptimal.

**Action:** sweep `--clip-eps` down (try 0.1, then 0.05) in the same diagnostic runs as 1.1/1.2. Cheap, no code
change needed, just a flag.

### 1.4 What NOT to change (already in the recommended range)

- `--epochs 4`: paper's guidance is 5-10 for hard problems, 15 max for easy ones. 4 is close enough; consider
  raising slightly (to ~6) *after* fixing minibatching, not before — more epochs over badly-chunked minibatches
  will just amplify the existing problem.
- `--gamma 0.99 --lam 0.95`: matches the paper's defaults, which it found transferred to multi-agent settings
  without modification. Leave as-is unless Phase 1.1/1.2/1.3 don't fully resolve the plateau.

### Phase 1 success criterion

A checkpoint-by-checkpoint `eval_bc.py` sweep (same methodology as the existing validated run) shows vs-Brain
win rate climbing with non-overlapping Wilson CIs between early and late checkpoints, ideally clearing BC's 31%
convincingly (not just a point estimate above it — the current 27% point estimate with overlapping CIs is not
good enough to call "better than BC," and neither would a new number be without the same rigor).

---

## Phase 2 — Reward shaping: switch to potential-based form

**Why:** three separate reward-shaping bugs have now been found and fixed by measurement (STEP_COST's unbounded
per-turn accumulation, `finish()`'s credit-assignment gap, and the split-discouraging growth shaping). That's a
pattern: ad hoc shaping terms are easy to get subtly wrong whenever a strategy's payoff (like a split's second
dragon) arrives outside GAE's effective credit horizon (~10-20 steps at gamma=0.99, lambda=0.95, per the
project's own prior investigation). Each bug so far has been caught reactively, after a full training run
degraded and someone dug in — an expensive way to find shaping errors.

**The fix, structurally:** [Ng, Harada & Russell's potential-based shaping](https://www.semanticscholar.org/paper/94066dc12fe31e96af7557838159bde598cb4f10)
proves that a shaping term of the form `F(s, s') = gamma * Phi(s') - Phi(s)`, for *any* potential function
`Phi(state)`, leaves the optimal policy unchanged. This means shaping can be freely tuned for faster learning
without ever risking a new perverse incentive, because the theorem guarantees policy invariance regardless of
what `Phi` looks like.

**Concrete plan:**
1. Define a potential function `Phi(state)` for a dragon — candidate terms: current length, a cheap board-control
   estimate (the project already has a fast bitboard flood-fill — reuse it), distance to nearest pearl. Start
   simple (just length) and validate before adding terms.
2. Rewrite the per-step shaping in `tools/trajectory.py` and `tools/ppo_rollout.py` as
   `gamma * Phi(next_state) - Phi(state)` instead of the current ad hoc `max(0.0, length_now - length_prev) /
   GROWTH_SCALE - STEP_COST` formula.
3. Keep the terminal (+-1 win/loss) and death penalty terms as true (non-shaped) rewards — potential-based
   shaping is specifically for the *dense* per-step signal, not the sparse outcome signal.
4. Validate: the split-discouraging pathology from before should be structurally impossible now (a split's
   effect on `Phi` — e.g. team-length-based potential — nets out correctly over time by construction), but
   confirm with the same eval sweep rather than assuming the theorem alone is enough (the theorem guarantees the
   *optimal* policy is unchanged in the limit; it says nothing about how fast PPO's practical, imperfect
   optimization gets there).

**Effort:** moderate — this touches the reward computation in two files and needs the same validation rigor as
any reward change has needed so far. Do this only after Phase 1 is validated, so any change in behavior is
attributable to one thing at a time.

---

## Phase 3 — Replace mirror self-play with a lightweight opponent pool

**Why:** pure mirror self-play (both sides run identical weights) is exactly the setup
[AlphaStar's league training](https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/)
was built to fix. Plain self-play can cycle through non-transitive strategies ("forgetting") instead of
monotonically improving, because an agent only ever has to beat *itself right now* — this was already on this
project's own suspect list (`--brain-frac 1.0` as an untried diagnostic) before this research pass.

**Full AlphaStar league (Main Agent + Main Exploiter + League Exploiter, with a persistent Elo-tracked pool) is
DeepMind-cluster-scale overkill here** — it's explicitly out of scope per the project's own plan, and rightly
so. A much cheaper, well-documented middle ground exists:
[Minimax Exploiter](https://arxiv.org/abs/2311.17190) uses just two archetypes (Main Agent + one Exploiter
trained against a frozen snapshot of it) and reports faster convergence than vanilla self-play on a single
GPU pair — a realistic scale for this project's hardware.

**Minimum viable version for this project (simpler than even Minimax Exploiter, a reasonable first step):**
1. Keep a small pool of the last N PPO checkpoints (`tools/ppo/ppo_*.npz` already exist — just stop discarding
   them) plus `Brain` (already available via `BrainOpponent` in `tools/ppo_rollout.py`).
2. In `collect_parallel`/`collect_one` (`tools/ppo_rollout.py`), replace "mirror self = current weights" with
   sampling an opponent from that pool per game. Start with uniform sampling from the last ~5 checkpoints (dead
   simple); if that alone breaks the mirror-cycling risk, consider win-rate-weighted sampling (PFSP's core idea:
   bias toward opponents you currently do worst against) as a follow-up refinement, not a prerequisite.
3. Keep `--brain-frac` for the Brain-opponent fraction as-is; it already exists and is a form of this idea, just
   currently the only form.

**Validation:** run the diagnostic `--brain-frac 1.0` (always vs. Brain, no self-play at all) as a quick,
cheap A/B against the pooled-opponent version — if collapse/stagnation patterns differ meaningfully between
"always vs Brain" and "pool of past selves + Brain," that confirms self-play dynamics (not reward shaping or
PPO hyperparameters) were a contributing factor.

**Effort:** moderate build (opponent sampling + checkpoint retention policy), no new ML machinery.

---

## Phase 4 — Team-level (pooled, centralized) critic

**Why:** the plan's original framing — that this "needs a custom simulator to assemble a full-team state" — is
worth re-checking against the actual code, not taken as given. `PPOTrajPlayer` (`tools/ppo_rollout.py`) already
holds every teammate's `Policy` instance and per-round features live in the same process during rollout
collection (`self.policies`, `self.steps` are keyed by dragon id, all populated within one `collect_one()` call
per game). The information needed for a joint critic input already exists in memory during collection — it just
isn't being gathered into one place. No new simulator is actually required.

**Design constraint:** team size is *variable* over an episode (splitting increases it, death decreases it), so
the joint critic input must be **permutation-invariant and variable-length-safe** — concatenation of fixed
slots won't work. Use mean- or max-pooling over teammates' hidden embeddings (or a small attention layer if
pooling proves too lossy), the standard MARL answer to variable-agent-count settings. This is genuinely an
active, not-fully-solved research niche in general (e.g. adaptive value decomposition work on varying-N urban
systems), but pooling is the well-established practical fix for the common case, and is a much smaller lift than
a general solution.

**Important:** keep the *actor* exactly as it is (per-dragon, egocentric, parameter-shared, decentralized
execution). Nothing about this phase should change what the deployed bot sees at inference — the pooled team
state is a training-time-only input to the critic, which is discarded at deployment (only the actor matters
once training is done, per `tools/torch_policy.py`'s existing docstring reasoning). This keeps the judge-side CPU
budget completely unaffected.

**Effort:** the largest phase so far — new bookkeeping in `ppo_rollout.py` to snapshot per-round team state,
a new pooling layer in `tools/torch_policy.py`'s `PolicyNet`, and re-validation of the whole pipeline (`wv`/`bv`
import/export logic will need to change shape, `test_torch_policy.py`'s parity tests will need updating). Do
this only after Phases 1-3 are validated and PPO is at least *matching* BC — don't spend the biggest phase on a
foundation that isn't proven to be worth building on yet.

**Success criterion:** explained variance of the critic (already computed by `explained_variance()` in
`train_ppo.py`, just needs printing every iteration, not just at warmup) should be visibly better with the
pooled team critic than the egocentric-only one on the same rollouts, before concluding it helped win rate too.

---

## Phase 5 (stretch, exploratory) — Search + learned value at inference

**Why:** every serious Battlesnake competition bot — the closest genre match with public strategy write-ups —
uses shallow minimax/expectimax search (~5 plies) with a flood-fill/area-control leaf heuristic, not a bare
reactive policy (see [coreyja.com's Battlesnake minimax writeup](https://coreyja.com/posts/BattlesnakeMinimax/Minimax%20in%20Battlesnake/)
and the wider [awesome-battlesnake](https://github.com/xtagon/awesome-battlesnake) list). This is the same
pattern that makes AlphaZero-style engines dominate pure policy networks in turn-based games generally: search
plus a learned evaluation beats either alone. This project already has the two ingredients a leaf evaluator
needs — a trained value head, and a cheap bitboard flood-fill (~2-6M CPU points, per prior measurement) — inside
a 100M-point turn budget that has shown headroom.

**The real complication (why this is Phase 5, not Phase 1):** Battlesnake is a perfect-information game; this
project's game is not (sonar-gated partial observability, no certain knowledge of enemy next moves). A clean
minimax needs either full information or a trustworthy opponent model. Two ways to get a workable approximation,
neither validated yet:
- Use `Brain` itself as a cheap, fast, already-available opponent-move proposer at search time (it's already a
  competitive heuristic bot — "assume the opponent plays roughly like Brain" is a much better prior than "assume
  it goes straight").
- Restrict search to *own*-move sequences only (multi-step self-lookahead against a static/frozen board
  snapshot, no opponent branching at all) — much cheaper, avoids the opponent-modeling problem entirely, at the
  cost of missing genuinely adversarial tactics. Worth trying first as the simplest version.

**Do not start this until PPO is clearly beating BC.** This phase adds real inference-time complexity and
CPU-budget risk (re-verify with `--sandbox -v` at every step, per this project's established practice) for a
policy that isn't yet the best one available. It's here because it's genuinely the field-standard next step for
a turn-based competitive game once the learning problem itself is solved, not because it's urgent.

---

## Validation discipline (applies to every phase above)

Lessons already paid for in this project, worth repeating as ground rules:

1. **One change at a time.** Every prior "fix" that turned out to be insufficient (STEP_COST alone, then
   value-warmup alone) was validated in isolation before the next bug was found underneath it. Don't skip this
   even though it's slower — stacking changes makes the next regression as hard to root-cause as this one was.
2. **Checkpoint-by-checkpoint eval with confidence intervals, not just the final checkpoint's win rate.** A
   flat or noisy in-training "mean return" number is not trustworthy on its own (already proven misleading once,
   per project memory) — always cross-check with `tools/eval_bc.py`'s real held-out-seed win rate, and use
   Wilson CIs to judge whether a change is a real effect or noise (as the already-validated 4th run did).
3. **Log every run to a file with `python -u` (or `PYTHONUNBUFFERED=1`).** A background training run's stdout
   fully buffers otherwise, so a `tail -f`/Monitor-based check sees nothing until the process exits — already
   hit once this project.
4. **Print `explained_variance` every iteration, not just at warmup.** It's already computed
   (`train_ppo.py`'s `explained_variance()`), just not logged during the main loop. Cheap, and would have given
   earlier visibility into exactly when a run started going wrong in past collapses.
5. **Treat "another session/window might be working on the same files" as live, not hypothetical.** This
   project has had genuinely concurrent Claude Code sessions editing `tools/train_ppo.py`/`tools/trajectory.py`
   during this exact investigation. Check `git status` and recent file mtimes before assuming a clean slate,
   the way this plan's own research phase had to.

---

## Recommended sequencing

1. **Phase 1** (hyperparameters) — cheapest, most directly evidenced, no architecture change. Do this first.
2. If still short of BC's 31%: **Phase 2** (potential-based shaping) and **Phase 3** (opponent pool) — can be
   validated in either order or as a combined A/B (pool-vs-mirror) since Phase 3 has its own quick diagnostic
   (`--brain-frac 1.0`) independent of Phase 2's reward rewrite.
3. Once PPO is reliably beating BC (not just a point estimate — non-overlapping CIs over a real held-out sweep,
   matching the stage-5 gate methodology): **Phase 4** (team-level critic) to push further past that baseline.
4. **Phase 5** (search + value hybrid) only once the above is solid and there's appetite for meaningfully more
   inference-time complexity and CPU-budget risk.

---

## Sources

- Yu et al., ["The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games"](https://arxiv.org/abs/2103.01955), NeurIPS 2022.
- DeepMind, ["AlphaStar: Grandmaster level in StarCraft II using multi-agent reinforcement learning"](https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/).
- ["Minimax Exploiter: A Data Efficient Approach for Competitive Self-Play"](https://arxiv.org/abs/2311.17190), arXiv:2311.17190.
- Ng, Harada, Russell, ["Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping"](https://www.semanticscholar.org/paper/94066dc12fe31e96af7557838159bde598cb4f10), ICML 1999.
- coreyja, ["Minimax in Battlesnake"](https://coreyja.com/posts/BattlesnakeMinimax/Minimax%20in%20Battlesnake/).
- ["awesome-battlesnake"](https://github.com/xtagon/awesome-battlesnake) curated resource list.
- ["Battlesnake Challenge: A Multi-agent RL Playground with Human-in-the-loop"](https://arxiv.org/pdf/2007.10504), arXiv:2007.10504.
- ["K-nearest Multi-agent Deep RL for Collaborative Tasks with a Variable Number of Agents"](https://arxiv.org/abs/2201.07092), arXiv:2201.07092.
- Project memory: `battlecode-stage4-rl-environment.md`, `battlecode-stage5-ppo-still-collapsing.md`,
  `battlecode-bot-strategy-findings.md`, `unswbc-engine-and-judge-budget.md`.
