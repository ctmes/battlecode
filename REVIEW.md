# Review: sea-dragon Battlecode bot, mid-project

Reviewer: M. Keane. Snapshot reviewed: HEAD `2b70325`, plus the working tree as it stood on 2026-09-27 at about 12:45 local time. A concurrent session reverted `grow_care` in `bot/brain.py` and started `tools/finetune_session.py` while I was reading; I did not touch either. Labels on each finding: **[V]** Verified (I ran it or read it unambiguously), **[L]** Likely, **[S]** Suspected.

Everything I ran lives in my scratchpad, not in the repo. The main experiments and their costs:

- **Budget probe:** replays recorded turn streams into the judge's metered sandbox (~2,000 turns, 10 s).
- **Host/judge decision parity:** 7,064 turns.
- **Seat effect and a reproduction of your 46% claim:** 300 games.
- **Shipped Policy vs Brain on the bundled maps:** 80 games.
- **Shipped Policy vs Brain by map size, on fresh seeds 5,000,000+:** 240 games.
- **Sealed-spawn census:** 750 generated maps.
- **Toy test of `tune.py`'s CMA boundary handling.**
- **The full test suite.**

I can add any of these scripts to `tools/` if you want them.

---

## 1. Verdict

Your best-evidenced bot is still the 22 Sep CMA-tuned heuristic, frozen as `manual-heuristics/`. The working tree now ships exactly that bot: it played 100/100 games move-for-move identically to the snapshot [V]. Five days of BC → PPO → evolution → Brain re-tuning have not produced a shipped improvement with evidence behind it.

The headline "evolved policy 57% vs Brain" is a map-size artefact. On fresh seeds the policy wins 89% on boards ≤20 wide and 8% on boards 49–64 wide [V]. It loses 40 of 40 games on the four largest bundled maps [V]. Your own ladder replays show a pool dominated by 25–64-wide maps, five of which you don't have locally [V].

Every result since Stage 3 is pairwise against your own bots at n ≤ 200. The one external number (58% "live") has no stated n. The Brain tuner has a boundary-handling bug that fully explains the "sigma runaway" [V]. Budget safety is good.

Confidence that you currently hold anything better than `manual-heuristics/`: low.

---

## 2. Claims audit

| # | Claim (source) | Evidence | Assessment | What would confirm it |
|---|---|---|---|---|
| 1 | Stage 3: `run1_avg5` won a 5-way, 3,160-game round robin at 59.5% and beats the old DEFAULTS 61–68% (`bot/brain.py:42-50`) | Held-out seeds 300000+ (`tools/tournament.py:32`). Per-participant SE ≈ 1.4pp. Best-of-5 null inflation ≈ 1.6pp | **Holds, within family.** The one properly designed selection in the project. Every participant is a Brain variant, and all maps are generated | The same round robin on ladder maps, with a non-Brain opponent in the field |
| 2 | BC reaches 31% vs Brain, 72% vs random (memory, `eval_bc.py`) | 100 games, Wilson [22.8, 40.6] | **Holds** (it's a floor, not a claim of strength) | n/a |
| 3 | PPO plateaus at the BC level (memory Stage 5) | Four runs; every checkpoint CI overlaps BC's | **Holds** | n/a |
| 4 | "Self-play mixing is protective, not dilutive" (memory) | One run per condition, no seed replication, KL drift confounded (0.06 vs 0.03–0.045) | **Unsupported.** Plausible, but n = 1 run per arm | 3 seeds × 2 conditions × ~1 h each |
| 5 | Evolution beats BC decisively (memory Stage 6) | 45.5% [38.7, 52.4] at gen 40, 57.0% [50.1, 63.7] at gen 274, vs BC 31% | **Holds** on generated maps. The pivot produced something real | n/a |
| 6 | Evolved policy at 57.0% "nominally clears the 55% gate" (memory) | n = 200 < the gate's 400. Seed block 100000+ reused for at least 3 checkpoints. On the full size range, in-training gens 300–307 score 49.2% over 528 games (`run1_meta.json`). The shipped checkpoint is gen 309 migrated, not gen 274, and was never validated | **Inflated, and it does not transfer.** By size on fresh seeds: 88.9% (≤20), 80.0% (21–32), 29.5% (33–48), 8.3% (49–64) [V] | Already refuted for the ladder use case. Retire the claim |
| 7 | "Bundled-map gap, 31.2% on 16 games; plausibly the `--max-side 32` training" (memory) | My run: 37.5% over 80 games. Arena 10/10, Colosseum 9/10, default_small 8/10; big_empty, default, queen_of_spades and schooltime 0/40 [V] | **Real, and misdescribed.** It's a board-size effect, and it holds even on `default.map`, which was a training anchor in all 300 generations | n/a |
| 8 | `grow_care` "validated": 92.0% vs grower_brain, 50.0% vs defaults (HEAD `brain.py` comment) | Two opponents, both built by you | **Holds against those opponents only.** It didn't transfer to manual-heuristics (claim 9): overfit to a constructed opponent | n/a |
| 9 | "Brain loses to manual_heuristics, 46.0% over 100 games" (`tools/campaign.py:26-27`) | I get 47.0% [37.5, 56.7] on the same seeds and parameters [V]. Not an exact reproduction, so your number came from different code or params | **Unsupported as a loss.** Indistinguishable from 50%. It still drove re-weighting the campaign and reverting DEFAULTS | ≥1,000 paired games |
| 10 | manual_heuristics has a "real, live 58% win rate" (`bot/brain.py:143-145`) | No n, no rating range, no dates in the repo | **Unverifiable.** Ladder win% isn't skill anyway: matchmaking pairs you near your rating, so win% trends to 50% at equilibrium. Elo trajectory is the metric | Export the battle history: version, date, opponent Elo, result |
| 11 | "full_run1 diverged; fix = bigger population and smaller sigma" (`tools/finetune_vs_manual.py:1-7`) | Toy reproduction [V], see §3.3 | **Misdiagnosed.** It's the boundary handling plus a sentinel cliff. `vs_manual_run1` repeats it (σ 0.074 → 0.32) | The toy test in §3.3 |
| 12 | Herding, prediction and sonar variants are negative results (`bot/brain.py:106-129`, memory) | 880–1,232 games per sweep, with firing-rate analysis | **Holds, within family.** Good practice | n/a |
| 13 | "Bots are deterministic, so one game per (map, side) is all there is" (`tools/league.py:6-7`) | 100/100 identical reproductions [V] | **Holds** for Brain vs Brain | n/a |
| 14 | The Brain fits the judge budget (implied by the design) | Metered-sandbox replay: later turns ≤12.6M points, p99 ≤10.8M, first turn ≤15.3M on 64×64, 60×40 and generated 64×64 maps [V] | **Holds, with ~8× headroom** | n/a |
| 15 | The shipped Policy fits the budget (`bot/main.py` before `f384a03`) | First turn ≈73M points on every dragon spawn, including every split child; later turns ≤9.9M [V] | **Holds, thin.** 27M of headroom on the first turn | n/a (only matters if re-shipped) |

---

## 3. Critical issues, ranked by impact on the final result

### 3.1 The learned policy is a small-map specialist, and the ladder plays mostly large maps [V]

- **What:** Shipped Policy (`bot/weights.bin`, byte-identical to `tools/evolved/run1_migrated.npz`, i.e. gen 309 with the sprint/sonar heads at bias −10) against the current Brain on fresh generated seeds 5,000,000+ (240 games, both seats):

  | max side | 10–20 | 21–32 | 33–48 | 49–64 |
  |---|---|---|---|---|
  | Policy score | 88.9% [77.8, 94.8] | 80.0% [68.2, 88.2] | 29.5% [20.5, 40.4] | 8.3% [3.3, 19.6] |

  On hand-made maps the mid band also collapses: default (32×32) 0/10 and trophy (25×25) 3/10.
- **Evidence it matters:** Your ladder replays (`replays/`, `leader-replays/`, 16 games) use 11 distinct maps: Autarky (54 wide), Big Empty (64), Default (32), Default Small (16), Devil (32×16), Prisoners Dilemma (32), Queen of Spades (25×35), Schooltime (60×40), Stronghold (48), Trauma (48), Trophy (25). Only Default Small is ≤20. Five are missing from `maps/`, and toolkit 0.3.6 doesn't bundle them.
- **Consequence:**
  - The 57% "gate pass" was a weighted average over the generator's size mix (`tools/mapgen.py:35-41`: 30% of maps ≤20 wide). The competition's mix is nothing like it.
  - Switching back to Brain on 27 Sep was the right call, but the commit gives no evidence for it, and eval_vs_policy's results were never persisted.
  - Three days of evolution compute (≈48 h for run1 plus 27 h in the discarded sprint/sonar prefix run) went into a policy that loses almost everything on 33+ boards.
- **Fix:** Retire the Policy as a ladder candidate unless it's retrained on large and ladder maps and re-validated per size band. See §7 for a cheap size-gated hybrid, which has small upside.

### 3.2 No evaluation resembles the ladder [V for the gaps; L for the impact]

- **What:** Every decision since Stage 3 was made against your own bots (Brain variants, your policy, your sparring bots, your own old snapshot) on your own generator's maps. No opponent in the pool was written by anyone else, and the ladder's map pool isn't represented.
- **Consequence:**
  - You're optimising a proxy with a known transfer failure (claims 6–8).
  - The `grow_care` episode is the template: 92% vs `grower_brain`, then no better than 47% vs the prior version.
- **Fix:**
  - Build a frozen ladder-proxy suite from:
    - the 11 ladder maps (recover the missing ones: try `unswbc maps` from a newer toolkit in a scratch venv, or decode them from the replays, where the map text is embedded almost verbatim in the header);
    - `manual-heuristics/`;
    - the evolved Policy;
    - splitter, grower and chaser;
    - behaviour notes from the five `leader-replays/`.
  - Reserve a seed block (e.g. 9,000,000+) used only for ship decisions.
  - Log every ranked battle: version, time, opponent rating, 5-game score.

### 3.3 The Brain tuner is broken in two independent ways [V]

- **Boundary handling:**
  - **What:** `SepCMA.tell()` clips the mean into [0, 1] (`tools/tune.py:171`) but feeds the unclipped weighted step `yw` into the evolution path `ps` (`tools/tune.py:173`). When the optimum lies on a bound, selection keeps pushing outward, `ps` keeps growing, and σ inflates every generation while the mean can't move.
  - **Toy test**, `tune.SepCMA` on 13 dims with a noisy quadratic, 5 seeds:

    | Optimum location | σ at gen 0 | 10 | 20 | 30 |
    |---|---|---|---|---|
    | Interior | 0.15 | 0.134 | 0.085 | 0.071 |
    | On the upper bound in 7 of 13 dims | 0.15 | 0.318 | 0.967 | 3.907 |

    `full_run1` went 0.138 → 1.989 with 7 of 13 final mean coordinates at 0 or 1.
  - **Per run:** sigma trends match how many mean coordinates are at a bound. `grow_care_run1` has none and σ falls 0.153 → 0.077. `run1`, `full_run1` and `vs_manual_run1` all have several and σ rises.
  - Once σ ≫ 1, candidates are random corners of the box, and the tuner is doing random search.
- **Sentinel cliffs:**
  - **What:** Parameters encode "off" as the lower bound of a magnitude. `split_len_max ∈ (0, 60)` (`tools/tune.py:58`) means 0 = off, but 1–3 disables splitting entirely (`bot/brain.py:330`).
  - **Evidence:** Every generation whose mean landed on `split_len_max` of 1–3 scored the mean at 0.000–0.086 against *every* opponent, with abnormally low deaths/k: `vs_manual_run1` gens 8 and 11 (0.000), `full_run1` gens 9 and 24 (0.057, 0.086).
  - The same shape affects `grow_care_len` (1 = always careful), `split_mate_radius` and `sprint_max`.
- **Also:**
  - "defaults" and "old" are defined relative to whatever `bot/brain.py` DEFAULTS say at import time (`tools/tune.py:68-71`). The opponent changed on 26 Sep (grow_care on), then again today (off). Every crash-resume re-imports it. Old and new numbers "vs defaults" aren't comparable.
  - `finetune_vs_manual.finalize()` picks the fallback generation by the noisy per-generation `best` (`:161`), whose null inflation at pop 20 × 36 games is ≈ +0.16.
- **Consequence:** Nothing tuned on 27 Sep (`full_run1`, `full_run1_recovered`, `vs_manual_run1`) can be trusted. The `session2_sprint` run launched at 12:40 uses the same code, with `sprint_max` sitting at its lower bound.
- **Fix:**
  - Reflect, or better, optimise in an unconstrained space mapped through a smooth transform (logit or cosine), plus a quadratic penalty on the pre-clip distance.
  - Remove sentinels from SPACE: decide on/off by direct A/B, and tune magnitudes only with the feature on.
  - Freeze opponents as snapshots (`frozen_brain.py` already exists; use it for "defaults" too).
  - Add the toy test above as a unit test.

### 3.4 Ship decisions are made at sample sizes that can't support them [V]

- **What:**
  - Your 46% (n = 100) was read as "loses".
  - The sonar_terrain sweep ships a weight if it beats w = 0 by > 0.02 on 80 games (`tools/campaign.py:152`). Per-arm SE is ≈ 0.056, so under independence most null sweeps would "win".
  - Validation seed blocks 100000+ and 700000+ have been reused across many candidates and checkpoints, so they're adaptively overfit.
- **Required n** (two-sided 95%, p ≈ 0.5):

  | Goal | Games |
  |---|---|
  | CI half-width 5pp | 385 |
  | CI half-width 3pp | 1,068 |
  | CI half-width 2pp | 2,401 |
  | Detect +5pp vs 50% at 80% power | 785 |
  | Detect +3pp vs 50% at 80% power | 2,181 |

  Paired seats on shared maps cut these substantially for A-vs-B comparisons against the same opponent.
- **Fix:**
  - Pre-register the rule: ship only if the paired score vs the incumbent is ≥ 53% with lower 95% bound > 50% over ≥ 1,000 games on untouched seeds and the ladder maps. Or run an SPRT (p0 = 0.50, p1 = 0.55, α = β = 0.05; expected ≈ 1,000 games).
  - A 14-worker league does ≈ 3–4 Brain games/s, so 1,000 games is about 5 minutes. There is no excuse for n = 100.

### 3.5 Provenance: you can't say from the repo what is live or why [V]

- **What:**
  - The submission history is only recoverable from comments: "grow_care_only (shipped v3)" (`tools/campaign.py:222`), "manual-heuristics … currently submitted and active" (`tools/frozen_brain.py:1-2`), and `tools/evolved/submitted/candidate_20260925_1549.npz`.
  - That candidate is *not* the weights in `bot/weights.bin`: L2 distance 0.82 from run1, with a different architecture.
  - `evolve.py` overwrites `args` in the meta file on every resume (`tools/evolve.py:372`). The run1 champion's actual recipe (pop 24 → 32, maps 10 → 25, max-side 32 → 64, vs brain → brain + splitter) survives only in logs and memory notes.
  - Evaluation outputs (46%, "33–46%", the grow_care validation, eval_vs_policy) were printed to a console and never saved (`tools/eval_vs_manual_heuristics.py:118-134`).
- **Fix:**
  - A `submissions.md` ledger: version, commit, params hash, weights hash, date, Elo before/after, number of ranked battles.
  - Every eval writes a JSON with commit, params, opponent-snapshot hash, seeds and per-game results.

### 3.6 Backup and repo hygiene [V/L]

- Local `main` is 11 commits ahead of `origin/main`. Commit `3e48ca1` adds `tools/bc/demo.npz` at 110,151,232 bytes, which is over GitHub's 100 MiB hard limit, so pushes will be rejected [L].
- The `.gitignore` rule `tools/bc/*.npz` came too late. The repo pack is 121 MB.
- `tools/evolved/` (≈75 compute-hours, including the CMA state needed to resume) is gitignored and exists on one disk.
- **Fix:** strip the blob from history (`git filter-repo --path tools/bc/demo.npz --invert-paths`), push, and copy `tools/evolved/` and `tools/bc/` to external storage.

### 3.7 Concurrent edits under running experiments [V]

- `bot/brain.py` DEFAULTS changed in the working tree today while `vs_manual_run1` (opponent mix includes `defaults:1`) was being crash-resumed. Each resume silently changes the opponent.
- Your own plan names concurrent sessions as a live risk (`docs/rl-improvement-plan.md:259-262`), and it happened again today.
- **Fix:** run tuning from a `git worktree` pinned at a tagged commit, with snapshot opponents.

### 3.8 Lower-impact findings

- **Sealed spawns in the generator** [V]:
  - 7/200 (max-side 32) to 12/200 (max-side 64) validation maps, and 7/100 eval maps, have a founder boxed in by kelp and its own body at round 0. Example: seed 700011 (11×11), where both teams die in round 0.
  - 2 of 100 eval maps are forced draws. Because the maps are symmetric, this is noise rather than bias, but the generator's "valid map" claim (`tools/mapgen.py:4`) doesn't hold for spawns.
- **Seat effect** [S]: Brain-vs-Brain mirrors on 50 maps gave side A 0.39 (≈[0.27, 0.53]). Your league already plays both seats, which is correct. A ranked battle keeps one colour for all 5 games (`context.txt:384`), so per-battle variance on the ladder is higher than per-game numbers suggest.
- **`mapgen.pick_size` ignores `lo` for the upper bands** [V] (`tools/mapgen.py:35-41`): `generate(s, 41, 64)` can return a 20-wide map. It's harmless for 10..64 but a trap for size-stratified training.
- **Founder status is set only after the legality early-return** [V] (`bot/brain.py:480-482` precedes `:549-550`). A founder with no OK move on round 0 that survives through a portal is permanently a non-founder. This is rare.
- **The shipped Policy has 6 action logits at −10** [V]. That's ≈10⁻⁴ probability per turn of an untrained sprint/sonar action. Moot while Brain ships.

---

## 4. Methodology critique

**Evaluation protocol.**
- The substrate is good: a deterministic in-process engine, both seats on shared maps, common random numbers in evolution, and exact reproducibility.
- The failures are upstream of the statistics:
  - wrong map distribution (§3.1);
  - in-family opponents (§3.2);
  - moving opponents (§3.3);
  - seed blocks reused for selection (§3.4).

**Statistics.**
- Wilson with draws counted as a half-win is fine as a guide. Draws are rare (1 in 50 mirrors).
- Games cluster in (map, seat) pairs. For HEAD-grow_care vs manual_heuristics, the per-map pair totals were 12 lost-both, 29 split, 9 won-both. That's slightly more splits than independence predicts (≈14/25/11), which means a seat effect, so Wilson on games is mildly conservative here.
- The CI method is not your problem. Selection and reuse are.

**Winner's curse.**
- Expected optimism of "best of pop" under the null: +0.16 at pop 20 × 36 games, +0.13 at pop 32 × 66 games, +0.16 at pop 14 × 28 games.
- You correctly report the CMA *mean*, not the best. Good.
- Hand-picking "gens 3–8, before the runaway" (`full_run1_recovered.json`) and choosing by `best` in `finalize()` reintroduce the curse.
- For the evolved policy, reporting the latest of three validations on the same seed block adds perhaps 1–2pp. That is small next to the size effect.

**Non-transitivity.**
- Stage 3 ran a real round robin. Nothing since has; you've gone back to pairwise-vs-one-reference.
- A 5–6-bot Elo or Bradley–Terry table over the ladder-proxy suite would have exposed claims 6–8 in an afternoon.

**Research arc.**
- **BC → PPO** was reasonable.
- **PPO was diagnosed well at the level of bugs**: STEP_COST suicide incentive, per-dragon GAE flattening, value warmup, split shaping, PopArt, minibatch count. All genuine catches.
- **But PPO's objective was never win rate:**
  - The ±1 outcome is added only to each dragon's last step (`tools/trajectory.py:95-106`), with γ = 0.99 and λ = 0.95 (`tools/train_ppo.py:276-277`).
  - A decision at round 50 sees the outcome through the value function at weight ≈ 0.99⁴⁵⁰ ≈ 1%. The critic target is itself a ~100-step discounted return.
  - So PPO optimised growth and survival, which is exactly what BC already imitates. A plateau at BC is the expected result, not a mystery.
  - Never tried: γ ≥ 0.999 with λ → 1, a team-outcome return, or a larger KL anchor.
- **PPO → evolution was justified in direction but unfair as a comparison:**
  - PPO got 155 iterations (≈4 wall-hours, ≈10k games).
  - Evolution got ≈48 h and ≈500k games, and optimised the actual objective (win rate vs Brain) directly.
  - "Evolution wins because it sidesteps credit assignment" is one hypothesis. "Evolution wins because it got ~50× the games and the right objective" is equally supported.
- **Separable CMA at n = 251k with λ = 32 is effectively a fixed-step (μ/μ_w, λ)-ES:**
  - σ went 0.0500 → 0.0494 over 309 generations (`run1_meta.json`).
  - c_σ ≈ 4·10⁻⁵ and c₁ ≈ 3·10⁻⁶, so neither step size nor covariance adapts.
  - That's fine, but call it what it is.
- **For the competition, the pivot that mattered was the one you didn't make:** improving the Brain against realistic opponents and maps. Your strategy notes already name the right target: head-on collisions are about half of deaths.

---

## 5. Engineering critique

**Correctness of the shipped logic.**
- Legality in `decide()` matches the rules I checked (`context.txt:256-260, 709-719`): kelp; any segment including your own not-yet-moved tail; head-on as a trade; portals treated as walls.
- Host (CPython 3.14) and judge sandbox (CPython 3.13) choose identical actions on 7,064/7,064 replayed turns [V], so local evaluation reflects the real bot.
- **Gap:** no unit test exercises `decide()`'s legality or `want_split()`'s legality. `test_brain.py` covers only `fallback()`.
- **Gap:** the runtime "LEGALITY BUG" detector in `tools/arena.py:62-64` needs `Brain.debug`, which the league turns off (`tools/league.py:42`), so hundreds of thousands of tuning games never check it. Turn it into a cheap counter and assert zero in every league run.

**Reproducibility.**
- Seeds and determinism are good [V].
- Provenance is weak (§3.5), and my reproduction of your 46% came out at 47%.
- There's no requirements file. The venv needs numpy 2.5.3, torch cu128, unswbc 0.3.6 and wasmtime 48, and your memory notes already record that this silently breaks a fresh venv.

**Budget safety** [V], from the metered-sandbox replay of the heaviest dragons:

| Bot | Map | First turn | Later max | p99 | Mean |
|---|---|---|---|---|---|
| Brain | big_empty 64×64 | 10.2–15.3M | 12.6M | 10.8M | 6–7.7M |
| Brain | schooltime 60×40 | 11.3–14.7M | 9.9M | 9.7M | 6–7M |
| Brain | generated 64×64 | 9.4–11.6M | 10.2M | 9.9M | 5.6–6.6M |
| Policy (old `main.py`) | big_empty | ≈73M | 9.9M | 9.6M | 8M |

- Brain has large headroom. `budget_ns = 60M` (`bot/brain.py:149`) never engages on the judge and never engages on the host either, so behaviour can't diverge between the two.
- Caveat: the probe replays blocks from Brain-vs-Brain games. A Policy-heavy opponent fielding ~8× the dragons puts more parts in view; re-probe if the opponent mix changes.

**Code health.**
- **Duplication:**
  - `eval_vs_policy.py` and `eval_vs_manual_heuristics.py` are near-copies.
  - `frozen_brain.load` and `opponents._load_snapshot_brain` are the same loader with different `strict` semantics (`frozen_brain.py:44` sets strict; `opponents.py:272-290` doesn't).
  - `campaign.py`, `finetune_vs_manual.py` and `finetune_session.py` copy-paste process supervision.
  - `bundled()` exists in both `tune.py` and `evolve.py`.
  - There are three `BrainPlayer` classes.
- **Dead code in the shipped bot:** about a third of `decide()` is behind flags that are off (predict, sonar_predict, sonar_terrain, squeeze, deny, sprint, explore, split caps, crowding, grow_care). `bot.toml` still ships the unused `policy.py`, `encoder.py` and `weights.bin`.
- **The fine-tune supervisor crash-looped 358 times in 36 minutes** (exit 1, then `0xC00000FD` stack overflow; `tools/tuned/finetune_vs_manual.log`) before a backoff was added. The root cause was never captured; stderr isn't logged.
- **Tests:** 15 standalone scripts, no runner, no CI. All 15 pass [V] (about 8 minutes, mostly mapgen, predict and sonar). They're good tests of what they cover. They don't cover the shipped decision path.

---

## 6. What's genuinely working

- **Bitboard Brain:** 6–13M points per turn on 64×64 boards, about 8× under the judge limit, with judge/host decision parity. This is the right architecture for a CPU-point judge, and it's why the Brain is robust across board sizes where the net isn't.
- **The in-process league:** deterministic, paired seats, common maps. It's a better experimental substrate than most teams have. It needs better inputs (maps, opponents, seed discipline), not replacement.
- **Stage 3:** CMA-tuned candidates picked by a round robin on held-out seeds, with the explicit note that single-opponent wins don't imply round-robin wins. That's exactly right, and it produced the bot that is still your best.
- **Negative-result discipline:** the prediction and sonar sweeps with firing-rate analysis ("decides only 0.7% of turns") are the kind of mechanism-level evidence most students skip.
- **PPO bug-hunting:** each fix had a mechanism and a test (GAE, PopArt invariance).
- **Budget engineering:** baked encoder tables, `frombuffer` weight loading, one write per turn.

---

## 7. Recommendations

### On your proposal: (manual heuristics) vs (manual heuristics + evolution)

Yes. Make that the main line, with these conditions:

1. **Base:** the Brain, not the net. It's budget-safe, size-robust and best-evidenced. `vs_manual_run1` is already a crude version of this. Its CMA mean averaged 52.9% against manual_heuristics over 32 non-degenerate generations (1,152 in-sample games; first 10 gens 48.6%, last 10 gens 55.3%). That hints at a few points of headroom, but it's unvalidated and run on the broken tuner. Fix §3.3 first.
2. **What to evolve:** not 251k weights. In order of preference:
   - (a) the Brain's ~30 scalars, with **separate parameter sets per board-size band** (≤20, 21–40, >40), since the right strategy evidently depends on board size;
   - (b) a small residual scorer (tens to low hundreds of weights) over features `decide()` already computes per move (pearl distance, trap area, dead end, head risk, voro, straight). Evolve it with the same ES and keep Brain's legality and trap checks as a hard mask.

   Both stay within about 15M points per turn.
3. **Opponents: a pool, not a mirror.**
   - Pure "MH vs MH+Δ" is a two-player loop against a deterministic opponent. It will find manual-heuristics-specific exploits (the grow_care precedent).
   - Pool: `manual-heuristics/` (≈50% weight), the evolved Policy (your only real swarm-style opponent, strong on small boards), splitter/grower, and every accepted champion frozen into the pool (PSRO-lite).
   - Self-play of the candidate against itself adds nothing for a deterministic bot.
4. **Maps:** the 11 ladder maps plus generated maps weighted toward ≥33 wide. Hold out 3 ladder maps and seed block 9,000,000+.
5. **Accept only if:** paired score vs manual-heuristics ≥ 53% with lower bound > 50% over ≥ 1,000 games on held-out seeds and ladder maps, *and* no pool member regresses by more than 3pp.
6. **Realistic expectation:** +2–5pp vs manual-heuristics in family; ladder transfer unknown until measured.

**Cheap side option, a size-gated hybrid:** run the Policy when `max(W, H) ≤ 20` and the Brain otherwise, decided at init.
- Only Default Small (1 of 11 observed ladder maps) qualifies, and the evidence there is 8/10 games.
- Upside ≈ +2–3pp overall if ladder opponents play like Brain, which they won't.
- Do it only after the suite exists to confirm it.

### Next day

| Action | Expected value | Cost | Success criterion |
|---|---|---|---|
| Fix `tune.py` boundary handling and remove sentinels (§3.3); add the toy-σ unit test | Makes every future tuning run meaningful | 2–3 h | Toy boundary case keeps σ ≤ 0.2 over 30 gens; no mean generation scores 0.000 |
| Stop or disregard the running `session2_sprint` and `vs_manual_run1` until the fix lands | Avoids another lost day | 0 | n/a |
| Recover the 5 missing ladder maps (newer toolkit, or decode from the replays) | Highest information per hour in this review | 1–3 h | 11 ladder maps in `maps/ladder/`, loadable by the engine |
| Push a backup (strip `demo.npz` from history); copy `tools/evolved/` off-disk | Insurance | 1 h | `origin/main == main`; checkpoints in two places |
| Write `submissions.md` and export the ladder battle history | Turns the only external signal into data | 1 h | Elo per version, with dates |

### Next week

| Action | Expected value | Cost | Success criterion |
|---|---|---|---|
| Build the ladder-proxy suite and a Bradley–Terry/Elo table over the pool; every eval writes provenance JSON | Replaces pairwise-vs-Brain as the decision metric | 1 day | One command, a ranked table with CIs, a per-map breakdown |
| Run the Brain-evolution programme above | The most likely real gain | 1–2 days compute | The acceptance rule in §7 |
| Failure analysis on ladder replays: death causes and tiebreak losses on large maps; what top teams do (sonar use, split cadence) | Targets the next heuristic | 0.5 day | A ranked list of loss modes with frequencies |
| Add `decide()`/`want_split()` legality unit tests and a zero-illegal-moves assertion in the league | Protects the shipped path | 2 h | Tests in the suite; counter at 0 over 1,000 games |

### Before submission

- Choose on the untouched seed block and ladder maps under the pre-registered rule.
- Re-run the sandbox budget probe on the exact submission directory (first turn and p99).
- Tag the commit and record the params and weights hashes.
- Run one unranked scrim on 3 ladder maps as a smoke test.

### Stop doing

- Tuning against "defaults"/"old" defined by whatever `bot/brain.py` currently says.
- Shipping or reverting on n ≤ 200, or on the seed blocks 100000+ and 700000+ (both are burned).
- Encoding "off" as the bottom of a CMA magnitude range.
- Editing `bot/brain.py` while experiments that import it are running or resuming.
- Adding mechanisms (sprint, terrain sonar, explore, crowding caps) without an A/B on the ladder suite. Each one enlarges the search space the broken tuner then wanders in.
- PPO, unless the thesis rather than the ladder is the goal. If it is the thesis, rerun with a team-outcome return and γ ≥ 0.999 before concluding anything about credit assignment.
- Further evolution of the 251k-weight policy at `--max-side 32`.

---

## 8. Open questions for you

1. What is live right now, and what was live when? For each version: dates, ranked battles, Elo. How many games does "58% live" rest on, and over what rating range?
2. When does the competition close, and how many days remain? The one-week plan assumes at least one.
3. Is the objective ladder placement, or a thesis about RL/ES for multi-agent games? The PPO advice differs.
4. Where do Devil, Autarky, Prisoners Dilemma, Stronghold and Trauma come from? Is the ladder pool fixed or growing?
5. Why did you switch from Policy to Brain on 27 Sep? Was there an eval_vs_policy or ladder result? None is in the repo.
6. What produced the `0xC00000FD` crashes and the 350 fast exit-1 restarts in `finetune_vs_manual`? Is stderr captured anywhere?
7. Why was the 114-generation sprint/sonar run discarded? If the portal-legality fix only changed the action mask, its weights may have been salvageable.
8. Two sessions are editing `bot/brain.py` and launching tuning runs concurrently. Who owns the working tree?

---

## Addendum (27 Sep, afternoon): ladder data, round robin, ladder maps

**Ladder record for the live bot** (v2 manual-heuristics, uploaded 25 Sep 16:23, from the user): 113–0–79 over 192 games, **59% [52, 66]**. This is the first external number, and it is clearly above 50%.

| Map | W–D–L | Win rate |
|---|---|---|
| Autarky | 16–0–5 | 76% |
| Queen Of Spades | 14–0–5 | 74% |
| Prisoners Dilemma | 13–0–7 | 65% |
| Trophy | 13–0–8 | 62% |
| Slithery Fight | 11–0–7 | 61% |
| Schooltime | 10–0–7 | 59% |
| Trauma | 13–0–9 | 59% |
| Default | 8–0–8 | 50% |
| Devil | 10–0–10 | 50% |
| Portals | 5–0–10 | **33%** |

Big Empty, Default Small and Stronghold have 1 game each, which looks like an older pool. The current pool is 10 maps, all 25–63 wide, with none small. That closes the size-gated-hybrid idea in §7.

**Ladder maps recovered.** Replays are Cap'n Proto "packed" messages. `tools/replay_maps.py` (a port of the official viewer's decoder) writes every map to `maps/ladder/`. The decoding is byte-exact: Queen Of Spades, Schooltime, Trophy and Default Small match the bundled files, and all 11 maps play in the engine. Two findings:

- The ladder's **Default** is not the local `default.map` (2,048 vs 2,112 edges), consistently across two replays.
- Stronghold is still missing, because it appears in no replay.

`maps/ladder/` is a subfolder, so the `maps/*.map` globs in `tune.py` and `evolve.py` do not pick it up.

**Round robin** (7 bots, 1,344 games; 8 bundled + 24 fresh generated maps, both seats):

| Bot | Overall | Large maps (33–64) |
|---|---|---|
| v3 Brain | 71% | 82% |
| Evolved policy | 71% | 46% |
| Today's tuned Brain | 71% | 82% |
| Shipped bot | 69% | 77% |
| BC policy | 42% | n/a |
| Splitter | 23% | n/a |
| Grower | 4% | n/a |

- The three Brains are tied head to head (43–57% per pairing).
- The evolved policy beats every Brain 77–80% on maps ≤20 wide and 5–10% on 33–64.

**New lead, Portals** [V mechanism; L cause of the 33%]: the Brain treats portals as walls (`bot/brain.py:14`). On `maps/ladder/portals.map` (40 portal edges, 286 kelp edges on 32×16), Brain vs Brain produced only 1,915 dragon-turns in 500 rounds. Total length ended at 8–10, and most deaths were walls. On Devil the same pairing gave 14,243 turns and length 88. Learning portal pairs (portal ids are visible, and a pair is known once both ends have been seen or relayed by sonar) is the most specific, testable improvement this review has found.
