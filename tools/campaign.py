"""Runs the 5-session tuning campaign end to end, unattended: each session auto-resumes on the WASM engine's
occasional bad_alloc crash (checkpointed every generation by tune.py itself, so a crash never loses more than the
generation in flight), waits for a target generation count, averages, and (where noted) validates and feeds its
result into the next session via --start. Safe to re-run: every step is idempotent (skips what is already done).

    .venv\\Scripts\\python.exe tools\\campaign.py [--workers 8]

Progress: tools/tuned/campaign.log (also printed). Each session's own generation-by-generation log is the usual
tools/tuned/<name>.json / tune.py stdout.
"""
import argparse
import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
TUNED = ROOT / "tools" / "tuned"
LOG = TUNED / "campaign.log"
VS = "defaults:2,old:1,grower_brain:2"  # session 1's benchmark (already running under this; kept for reference)
# Sessions 2+: broadened 2026-09-27 to add policy (the previously-shipped trained bot, ~8x Brain's dragon count
# -- session 1's grower_brain never tests being this outnumbered) and manual_heuristics (a prior from-scratch
# submission, a genuinely different design rather than a brain.py variant). manual_heuristics is weighted
# heaviest: per the user, it is the strongest opponent measured so far (Brain currently loses to it, 46.0% over
# 100 games, with total length/dragon count/longest-alive-dragon all roughly tied -- a tactical gap, not a
# numbers one), so beating it specifically should dominate what these sessions optimize for.
VS2 = "defaults:1,old:1,grower_brain:2,policy:2,manual_heuristics:3"

S1_PARAMS = ("grow_mod,founder_units,split_units,tiles_per_unit,split_r_end,grow_care_len,grow_care_mult,"
             "grow_care_margin,split_len_max,explore,split_mate_radius,split_mate_cap,split_food_ratio")
S2_PARAMS = "sprint_max,pearl_here,pearl_near,pearl_k"
# The original DEFAULTS search space (see tune.py SPACE's comment for what's deliberately left out) plus
# everything added this session, for the grand unified re-tune.
S4_PARAMS = (
    "pearl_here,pearl_near,pearl_k,trap,need_margin,area,head_risk,head_risk_small,trade_ratio,team_head_risk,"
    "straight,split_len,split_child,founder_split_len,grow_mod,split_r_end,split_min_exits,tiles_per_unit,"
    "split_pearls,split_units,founder_units,dead_end,need_floor,voro,voro_radius,grow_care_len,grow_care_mult,"
    "grow_care_margin,split_mate_radius,split_mate_cap,split_food_ratio,split_len_max,explore,sprint_max,"
    "sonar_terrain"
)


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def gen_of(name):
    f = TUNED / f"{name}.json"
    if not f.exists():
        return -1
    try:
        return json.loads(f.read_text())["state"]["gen"]
    except (json.JSONDecodeError, KeyError, OSError):
        return -1


def is_running(name):
    """True if some OTHER process (e.g. one launched before this campaign started) is already running
    `tune.py run --name <name>`. Checked via a real Windows process listing, not Git-Bash's `ps` (which misses
    processes outside its own tree -- the exact gap that caused the earlier orphaned-worker-pool incident)."""
    marker = f"--name {name}"
    out = subprocess.run(
        ["powershell", "-Command",
         "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Select-Object -Expand CommandLine"],
        capture_output=True, text=True).stdout
    return any(marker in line and "tune.py" in line for line in out.splitlines())


def run_until(name, target_gen, workers, fresh_args=None):
    """(Re)launches `tune.py run --name name --resume` until state.gen >= target_gen, tolerating crashes, and
    never launching a second concurrent process for the same name (waits instead if one is already running)."""
    if gen_of(name) >= target_gen:
        log(f"{name}: already at gen {gen_of(name)} >= {target_gen}, skipping")
        return
    attempt = 0
    while gen_of(name) < target_gen:
        if is_running(name):
            log(f"{name}: already running elsewhere (gen {gen_of(name)}), waiting rather than launching a second")
            time.sleep(30)
            continue
        attempt += 1
        resumed = (TUNED / f"{name}.json").exists()
        cmd = [PY, "tools/tune.py", "run", "--name", name, "--workers", str(workers)]
        cmd += ["--resume"] if resumed else fresh_args
        log(f"{name}: attempt {attempt} ({'resume' if resumed else 'fresh'}) toward gen {target_gen}")
        proc = subprocess.run(cmd, cwd=ROOT)
        g = gen_of(name)
        log(f"{name}: attempt {attempt} exited {proc.returncode}, now at gen {g}")
        if proc.returncode != 0 and g < target_gen:
            time.sleep(5)  # let any crashed worker pool fully release resources before retrying
    log(f"{name}: reached gen {gen_of(name)}")


def average(name, last=8):
    out = TUNED / f"{name}_avg{last}.json"
    if out.exists():
        log(f"{name}: average already exists ({out.name}), skipping")
        return out
    subprocess.run([PY, "tools/tune.py", "average", name, "--last", str(last)], cwd=ROOT, check=True)
    log(f"{name}: averaged -> {out.name}")
    return out


def validate(params_path, games=400, vs="defaults,old,grower_brain,splitter", workers=None):
    cmd = [PY, "tools/tune.py", "validate", str(params_path), "--games", str(games), "--vs", vs]
    if workers:
        cmd += ["--workers", str(workers)]
    log(f"validating {params_path.name} ({games} games vs {vs})")
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    log(proc.stdout)
    if proc.returncode != 0:
        log(f"validate FAILED: {proc.stderr[-2000:]}")


def merge_params(*paths):
    """Union of {name: value} from several tuned .json files, later files winning on overlap."""
    out = {}
    for p in paths:
        out.update(json.loads(pathlib.Path(p).read_text())["params"])
    return out


def sonar_terrain_sweep(workers):
    """A single dimension: a direct weight sweep is more diagnostic than full CMA-ES here, and matches how
    sonar_predict's negative result was established. Uses whatever session 1+2 have found so far as the base.
    Benchmarked directly against manual_heuristics (the strongest known opponent) rather than a self-play mirror,
    since what matters is whether terrain-sharing helps against the actual bar, not just against itself."""
    out_path = TUNED / "sonar_terrain_sweep.json"
    if out_path.exists():
        log("sonar_terrain sweep already done, skipping")
        return json.loads(out_path.read_text())
    base = merge_params(TUNED / "s1_avg8.json", TUNED / "s2_avg8.json")
    sys.path.insert(0, str(ROOT / "tools"))
    sys.path.insert(0, str(ROOT / "bot"))
    import league  # noqa: E402
    results = {}
    with league.League(workers) as lg:
        for w in (0, 15, 40, 80, 150):
            params = {**base, "sonar_terrain": w}
            maps = [(900000 + i, 10, 64) for i in range(40)]
            jobs = [league.Job(m, side, params, ("bot", "manual_heuristics")) for m in maps for side in "AB"]
            res = lg.run(jobs)
            s = league.summarize(res)
            results[w] = s["score"]
            log(f"sonar_terrain={w}: score {s['score']:.3f} over {s['games']} vs manual_heuristics")
    best = max(results, key=results.get)
    out = {**base, "sonar_terrain": best if results[best] > results[0] + 0.02 else 0}
    out_path.write_text(json.dumps({"params": out, "sweep": results}, indent=1))
    log(f"sonar_terrain sweep: {results} -> chose {out['sonar_terrain']}")
    return {"params": out}


def round_robin(candidates, games=200, workers=None):
    """candidates: {label: params_dict}. Every pair, both sides, on generated maps -- the final pick."""
    sys.path.insert(0, str(ROOT / "tools"))
    sys.path.insert(0, str(ROOT / "bot"))
    import league  # noqa: E402
    names = list(candidates)
    tally = {n: 0.0 for n in names}
    games_each = max(2, games // (len(names) - 1)) if len(names) > 1 else 0
    with league.League(workers) as lg:
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                maps = [(950000 + i * 1000 + k, 10, 64) for k in range(games_each // 2)]
                jobs = [league.Job(m, side, candidates[a], ("brain", candidates[b])) for m in maps for side in "AB"]
                res = lg.run(jobs)
                s = league.summarize(res)
                tally[a] += s["score"] * s["games"]
                tally[b] += (1 - s["score"]) * s["games"]
                log(f"round robin {a} vs {b}: {a} scored {s['score']:.3f} over {s['games']} games")
    log(f"round robin totals: {tally}")
    return tally


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    TUNED.mkdir(exist_ok=True)
    log("=== campaign start ===")

    # Session 1: splitting/circling/tiebreak (already running as full_run1 when this campaign was written --
    # this call just adopts and continues it under the same name).
    run_until("full_run1", 25, args.workers,
              fresh_args=["--generations", "35", "--pop", "14", "--maps", "6", "--max-side", "64",
                          "--vs", VS, "--params", S1_PARAMS])
    s1 = average("full_run1", 8)
    validate(s1, games=400, workers=args.workers)

    # Session 2: sprint + pearl recalibration, seeded from session 1.
    (TUNED / "s1_avg8.json").write_text(s1.read_text())
    run_until("sprint_run1", 20, args.workers,
              fresh_args=["--generations", "20", "--pop", "12", "--maps", "6", "--max-side", "64",
                          "--vs", VS2, "--params", S2_PARAMS, "--start", str(s1)])
    s2 = average("sprint_run1", 6)
    (TUNED / "s2_avg8.json").write_text(s2.read_text())
    validate(s2, games=400, workers=args.workers)

    # Session 3: sonar terrain-sharing, a direct sweep (one dimension).
    s3 = sonar_terrain_sweep(args.workers)
    start4 = TUNED / "campaign_start4.json"
    merged = merge_params(s1, s2)
    merged["sonar_terrain"] = s3["params"].get("sonar_terrain", 0)
    start4.write_text(json.dumps({"params": merged}, indent=1))

    # Session 4: grand unified re-tune, everything together, seeded from sessions 1-3 combined.
    run_until("grand_run1", 30, args.workers,
              fresh_args=["--generations", "30", "--pop", "16", "--maps", "8", "--max-side", "64",
                          "--vs", VS2, "--params", S4_PARAMS, "--start", str(start4)])
    s4 = average("grand_run1", 8)

    # Session 5: validation, round robin against what shipped before, and vs. the old trained Policy.
    validate(s4, games=400, vs="defaults,old,grower_brain,splitter,chaser,policy,manual_heuristics",
              workers=args.workers)
    candidates = {
        "defaults": {},
        "grow_care_only (shipped v3)": json.loads((TUNED / "grow_care_run1_avg8.json").read_text())["params"],
        "session1 (split/circle fix)": json.loads(s1.read_text())["params"],
        "grand_unified (final)": json.loads(s4.read_text())["params"],
    }
    round_robin(candidates, games=200, workers=args.workers)

    for script in ("eval_vs_manual_heuristics.py", "eval_vs_policy.py"):
        log(f"final detailed comparison: {script}")
        proc = subprocess.run([PY, f"tools/{script}", "--games", "150", "--brain-params", str(s4),
                                "--workers", str(args.workers)], cwd=ROOT, capture_output=True, text=True)
        log(proc.stdout)
        if proc.returncode != 0:
            log(f"{script} FAILED: {proc.stderr[-2000:]}")

    log("=== campaign done: final candidate at tools/tuned/grand_run1_avg8.json ===")
    print(json.dumps(json.loads(s4.read_text())["params"], indent=1))


if __name__ == "__main__":
    main()
