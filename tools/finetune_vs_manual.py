"""Focused session: fine-tune Brain specifically against manual-heuristics/ (the strongest opponent measured so
far), within a hard wall-clock budget. Corrected CMA-ES settings vs. the earlier full_run1 attempt, which
diverged (sigma grew every single generation, 0.138 -> 1.989 over 25 gens, while best/avg scores degraded):
bigger population (20 vs 14) for more robust covariance estimation in 13 dimensions, a smaller initial step size
(0.08 vs 0.15), and more maps per generation (10 vs 6) for a less noisy fitness signal. Also self-polices: if
sigma clearly runs away again, this stops itself and falls back to the last healthy point rather than grinding
to the generation ceiling on a diverged search.

    .venv\\Scripts\\python.exe tools\\finetune_vs_manual.py [--hours 6] [--workers 8]
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
LOG = TUNED / "finetune_vs_manual.log"
NAME = "vs_manual_run1"
PARAMS = ("grow_mod,founder_units,split_units,tiles_per_unit,split_r_end,grow_care_len,grow_care_mult,"
          "grow_care_margin,split_len_max,explore,split_mate_radius,split_mate_cap,split_food_ratio")
VS = "manual_heuristics:4,defaults:1"
SEED_FILE = TUNED / "full_run1_recovered.json"  # the healthy gen 3-8 point recovered from the diverged run


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def is_running(name):
    marker = f"--name {name}"
    out = subprocess.run(
        ["powershell", "-Command",
         "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Select-Object -Expand CommandLine"],
        capture_output=True, text=True).stdout
    return any(marker in line and "tune.py" in line for line in out.splitlines())


def kill_if_running(name):
    """Explicitly stops any process still running `tune.py run --name <name>`, whole tree. Needed because
    full_run1 showed subprocess.run() alone is not enough: a process launched independently of this script
    (or one that outlives a --generations ceiling mismatch) just keeps running past whatever target mattered."""
    out = subprocess.run(
        ["powershell", "-Command",
         "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
         "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"],
        capture_output=True, text=True).stdout
    try:
        rows = json.loads(out)
        rows = rows if isinstance(rows, list) else [rows]
    except (json.JSONDecodeError, TypeError):
        return
    marker = f"--name {name}"
    for row in rows:
        cmd = row.get("CommandLine") or ""
        if marker in cmd and "tune.py" in cmd:
            pid = row["ProcessId"]
            log(f"{name}: stopping still-running process {pid}")
            subprocess.run(["powershell", "-Command", f"taskkill /PID {pid} /T /F"], capture_output=True)


def state(name):
    f = TUNED / f"{name}.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def gen_of(name):
    s = state(name)
    return s["state"]["gen"] if s else -1


def diverging(name, window=4, sigma_ratio=1.8):
    """True if sigma has grown by more than `sigma_ratio`x over the last `window` generations with no
    improvement in best score -- the exact signature full_run1 showed for 25 straight generations."""
    s = state(name)
    if not s or len(s["history"]) < window + 1:
        return False
    hist = s["history"]
    now, then = hist[-1], hist[-1 - window]
    grew = now["sigma"] > then["sigma"] * sigma_ratio
    stalled = now["best"] <= then["best"] + 0.02
    monotonic = all(hist[-i]["sigma"] >= hist[-i - 1]["sigma"] for i in range(1, window + 1))
    return grew and stalled and monotonic


def run_session(deadline, workers):
    fresh_args = ["--generations", "60", "--pop", "20", "--maps", "10", "--max-side", "64",
                  "--sigma", "0.08", "--vs", VS, "--params", PARAMS, "--start", str(SEED_FILE)]
    attempt = 0
    stuck = 0  # consecutive attempts that both failed fast AND made no generation progress
    while time.time() < deadline:
        if gen_of(NAME) >= 0 and diverging(NAME):
            log(f"{NAME}: diverging (sigma runaway, no improvement) at gen {gen_of(NAME)} -- stopping early")
            kill_if_running(NAME)
            return "diverged"
        if is_running(NAME):
            time.sleep(20)
            continue
        attempt += 1
        gen_before = gen_of(NAME)
        resumed = (TUNED / f"{NAME}.json").exists()
        cmd = [PY, "tools/tune.py", "run", "--name", NAME, "--workers", str(workers)]
        cmd += ["--resume"] if resumed else fresh_args
        log(f"{NAME}: attempt {attempt} ({'resume' if resumed else 'fresh'})")
        remaining = deadline - time.time()
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, cwd=ROOT, timeout=max(60, remaining))
        except subprocess.TimeoutExpired:
            log(f"{NAME}: hit the wall clock budget mid-generation, stopping")
            kill_if_running(NAME)
            return "time_budget"
        elapsed = time.time() - t0
        g = gen_of(NAME)
        log(f"{NAME}: attempt {attempt} exited {proc.returncode} after {elapsed:.0f}s, now at gen {g}")
        if proc.returncode == 0:
            return "converged_or_ceiling"
        # A crash mid-generation (real progress happening, e.g. the earlier bad_alloc pattern) takes minutes.
        # A crash in well under a minute with zero generation progress is something else entirely -- exactly
        # the loop that burned ~1h retrying every 5s at gen 25 last time. Back off hard instead of repeating it.
        if elapsed < 45 and g <= gen_before:
            stuck += 1
            if stuck >= 8:
                log(f"{NAME}: stuck {stuck} attempts in a row with no progress (fast, repeated failures) -- "
                    f"stopping rather than looping further; last exit was {proc.returncode}")
                return "stuck"
            backoff = min(300, 10 * (2 ** (stuck - 1)))
            log(f"{NAME}: fast failure with no progress ({stuck} in a row), backing off {backoff}s")
            time.sleep(backoff)
        else:
            stuck = 0
            time.sleep(5)
        if diverging(NAME):
            log(f"{NAME}: diverging after crash-resume at gen {gen_of(NAME)} -- stopping early")
            kill_if_running(NAME)
            return "diverged"
    return "time_budget"


def finalize(reason):
    s = state(NAME)
    if not s or not s["history"]:
        log("no history to finalize")
        return
    hist = s["history"]
    if reason == "diverged":
        # back off to the healthiest recent point: best `best` score in the last 10 generations before the
        # flagged window, not the current (diverged) mean.
        window = hist[max(0, len(hist) - 14):max(1, len(hist) - 4)] or hist
        best_h = max(window, key=lambda h: h["best"])
        log(f"finalizing from gen {best_h['gen']} (best={best_h['best']:.3f}), pre-divergence, not the tail")
        out = {"params": best_h["mean_params"], "source": f"{NAME} gen {best_h['gen']} (pre-divergence fallback)"}
    else:
        last = min(8, len(hist))
        subprocess.run([PY, "tools/tune.py", "average", NAME, "--last", str(last)], cwd=ROOT, check=True)
        avg_path = TUNED / f"{NAME}_avg{last}.json"
        out = json.loads(avg_path.read_text())
        log(f"finalized via average of last {last} generations -> {avg_path.name}")
    final_path = TUNED / f"{NAME}_final.json"
    final_path.write_text(json.dumps(out, indent=1))
    log(f"final candidate -> {final_path}")

    # detailed before/after report against manual_heuristics specifically
    proc = subprocess.run([PY, "tools/eval_vs_manual_heuristics.py", "--games", "150",
                            "--brain-params", str(final_path), "--workers", "4"],
                           cwd=ROOT, capture_output=True, text=True)
    log("final vs manual_heuristics:\n" + proc.stdout)
    if proc.returncode != 0:
        log(f"eval_vs_manual_heuristics FAILED: {proc.stderr[-1500:]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    TUNED.mkdir(exist_ok=True)
    deadline = time.time() + args.hours * 3600
    log(f"=== vs-manual-heuristics finetune start, {args.hours}h budget, seeded from {SEED_FILE.name} ===")
    reason = run_session(deadline, args.workers)
    log(f"session ended: {reason}")
    finalize(reason)
    log("=== done ===")


if __name__ == "__main__":
    main()
