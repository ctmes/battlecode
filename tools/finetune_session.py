"""General-purpose focused tuning session: given a name, a parameter list, an opponent mix, and an optional seed,
runs tune.py to a hard wall-clock budget with the safety net built for the manual_heuristics session:
  - auto-resumes through crashes (the WASM engine's occasional bad_alloc)
  - a divergence guard (stops itself if sigma runs away with no score improvement, instead of grinding on)
  - a stuck-loop circuit breaker (stops itself after repeated fast failures with zero progress, with
    exponential backoff, instead of burning the whole budget retrying every few seconds)
  - ends with average-of-last-8 (or a pre-divergence fallback) plus a detailed before/after report vs
    manual_heuristics specifically (length, dragon count, and the longest-alive-dragon tiebreak stat)

    .venv\\Scripts\\python.exe tools\\finetune_session.py --name NAME --params a,b,c --vs "manual_heuristics:4,old:1"
        [--hours 2] [--workers 8] [--pop 20] [--maps 10] [--sigma 0.08] [--start path/to/seed.json]
"""
import argparse
import json
import pathlib
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
TUNED = ROOT / "tools" / "tuned"


def log(name, msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(TUNED / f"{name}.log", "a") as f:
        f.write(line + "\n")


def is_running(name):
    marker = f"--name {name}"
    out = subprocess.run(
        ["powershell", "-Command",
         "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Select-Object -Expand CommandLine"],
        capture_output=True, text=True).stdout
    return any(marker in line and "tune.py" in line for line in out.splitlines())


def kill_if_running(name):
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
            log(name, f"stopping still-running process {pid}")
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
    s = state(name)
    if not s or len(s["history"]) < window + 1:
        return False
    hist = s["history"]
    now, then = hist[-1], hist[-1 - window]
    grew = now["sigma"] > then["sigma"] * sigma_ratio
    stalled = now["best"] <= then["best"] + 0.02
    monotonic = all(hist[-i]["sigma"] >= hist[-i - 1]["sigma"] for i in range(1, window + 1))
    return grew and stalled and monotonic


def run_session(name, deadline, workers, fresh_args):
    attempt, stuck = 0, 0
    while time.time() < deadline:
        if gen_of(name) >= 0 and diverging(name):
            log(name, f"diverging (sigma runaway, no improvement) at gen {gen_of(name)} -- stopping early")
            kill_if_running(name)
            return "diverged"
        if is_running(name):
            time.sleep(20)
            continue
        attempt += 1
        gen_before = gen_of(name)
        resumed = (TUNED / f"{name}.json").exists()
        cmd = [PY, "tools/tune.py", "run", "--name", name, "--workers", str(workers)]
        cmd += ["--resume"] if resumed else fresh_args
        log(name, f"attempt {attempt} ({'resume' if resumed else 'fresh'})")
        remaining = deadline - time.time()
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, cwd=ROOT, timeout=max(60, remaining))
        except subprocess.TimeoutExpired:
            log(name, "hit the wall clock budget mid-generation, stopping")
            kill_if_running(name)
            return "time_budget"
        elapsed = time.time() - t0
        g = gen_of(name)
        log(name, f"attempt {attempt} exited {proc.returncode} after {elapsed:.0f}s, now at gen {g}")
        if proc.returncode == 0:
            return "converged_or_ceiling"
        if elapsed < 45 and g <= gen_before:
            stuck += 1
            if stuck >= 8:
                log(name, f"stuck {stuck} attempts in a row with no progress -- stopping, last exit {proc.returncode}")
                return "stuck"
            backoff = min(300, 10 * (2 ** (stuck - 1)))
            log(name, f"fast failure with no progress ({stuck} in a row), backing off {backoff}s")
            time.sleep(backoff)
        else:
            stuck = 0
            time.sleep(5)
        if diverging(name):
            log(name, f"diverging after crash-resume at gen {gen_of(name)} -- stopping early")
            kill_if_running(name)
            return "diverged"
    return "time_budget"


def finalize(name):
    reason_path = TUNED / f"{name}_reason.txt"
    reason = reason_path.read_text().strip() if reason_path.exists() else "time_budget"
    s = state(name)
    if not s or not s["history"]:
        log(name, "no history to finalize")
        return
    hist = s["history"]
    if reason == "diverged":
        window = hist[max(0, len(hist) - 14):max(1, len(hist) - 4)] or hist
        best_h = max(window, key=lambda h: h["best"])
        log(name, f"finalizing from gen {best_h['gen']} (best={best_h['best']:.3f}), pre-divergence, not the tail")
        out = {"params": best_h["mean_params"], "source": f"{name} gen {best_h['gen']} (pre-divergence fallback)"}
    else:
        last = min(8, len(hist))
        subprocess.run([PY, "tools/tune.py", "average", name, "--last", str(last)], cwd=ROOT, check=True)
        avg_path = TUNED / f"{name}_avg{last}.json"
        out = json.loads(avg_path.read_text())
        log(name, f"finalized via average of last {last} generations -> {avg_path.name}")
    final_path = TUNED / f"{name}_final.json"
    final_path.write_text(json.dumps(out, indent=1))
    log(name, f"final candidate -> {final_path}")

    proc = subprocess.run([PY, "tools/eval_vs_manual_heuristics.py", "--games", "150",
                            "--brain-params", str(final_path), "--workers", "4"],
                           cwd=ROOT, capture_output=True, text=True)
    log(name, "final vs manual_heuristics:\n" + proc.stdout)
    if proc.returncode != 0:
        log(name, f"eval_vs_manual_heuristics FAILED: {proc.stderr[-1500:]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--vs", required=True)
    ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pop", type=int, default=20)
    ap.add_argument("--maps", type=int, default=10)
    ap.add_argument("--sigma", type=float, default=0.08)
    ap.add_argument("--generations", type=int, default=60)
    ap.add_argument("--start", default=None)
    args = ap.parse_args()
    TUNED.mkdir(exist_ok=True)
    deadline = time.time() + args.hours * 3600
    fresh_args = ["--generations", str(args.generations), "--pop", str(args.pop), "--maps", str(args.maps),
                  "--max-side", "64", "--sigma", str(args.sigma), "--vs", args.vs, "--params", args.params]
    if args.start:
        fresh_args += ["--start", args.start]
    log(args.name, f"=== {args.name} start, {args.hours}h budget, params={args.params}, vs={args.vs}, "
                    f"start={args.start} ===")
    reason = run_session(args.name, deadline, args.workers, fresh_args)
    log(args.name, f"session ended: {reason}")
    (TUNED / f"{args.name}_reason.txt").write_text(reason)
    finalize(args.name)
    log(args.name, "=== done ===")


if __name__ == "__main__":
    main()
