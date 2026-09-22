"""Tuning loop for the brain's parameters: a separable CMA-ES over brain.DEFAULTS, scored by win rate in the league.

    .venv\\Scripts\\python.exe tools\\tune.py run --name run1 [--generations 20] [--pop 16] [--maps 10] [--max-side 32]
    .venv\\Scripts\\python.exe tools\\tune.py run --name run1 --resume            (continue an interrupted run)
    .venv\\Scripts\\python.exe tools\\tune.py average run1 [--last 5]             (steadier answer: mean of recent means)
    .venv\\Scripts\\python.exe tools\\tune.py validate tools\\tuned\\run1.json [--games 400] [--vs defaults,old,splitter]

Each generation draws fresh generated maps (plus the bundled ones that fit --max-side), plays every candidate on the
same maps against the opponents, both sides, and ranks the candidates by weighted win rate (draw = half). Because the
bots are deterministic the maps are the only noise, and sharing them across candidates removes most of it.
Validation plays on maps the tuner never saw (seeds 100000 up) and reports generated and bundled maps separately:
a gain on generated maps alone may be tuning to the generator rather than to the game.
"""
import argparse
import json
import math
import pathlib
import random
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from league import Job, League, summarize, wilson  # noqa: E402  (also puts bot/ on the path)
from brain import DEFAULTS  # noqa: E402

TUNED = ROOT / "tools" / "tuned"
MAPS = ROOT / "maps"

# name: (low, high, scale). "log" spreads a parameter over orders of magnitude, "int" rounds. Left out on purpose:
# squeeze / deny* (herding did not work, see the strategy notes), pessimistic, need_cap and budget_ns (not behaviour).
SPACE = {
    "pearl_here": (200, 3000, "log"), "pearl_near": (2, 100, "log"), "pearl_k": (2, 10, "int"),
    "trap": (300, 12000, "log"), "need_margin": (0, 8, "int"), "area": (0, 60, "lin"),
    "head_risk": (0, 1500, "lin"), "head_risk_small": (0, 600, "lin"), "trade_ratio": (0.2, 1.2, "lin"),
    "team_head_risk": (0, 800, "lin"), "straight": (0, 40, "lin"),
    "split_len": (3, 12, "int"), "split_child": (2, 4, "int"), "founder_split_len": (3, 12, "int"),
    "grow_mod": (0, 12, "int"), "split_r_end": (100, 500, "int"), "split_min_exits": (1, 3, "int"),
    "tiles_per_unit": (0, 150, "int"), "split_pearls": (0, 4, "int"), "split_units": (4, 64, "int"),
    "founder_units": (2, 64, "int"), "dead_end": (0, 400, "lin"), "need_floor": (0, 60, "int"),
    "voro": (0, 12, "lin"), "voro_radius": (2, 9, "int"),
}
# The brain as it was before the dead-end / room-floor / grower / territory work (git HEAD~ of brain.py's DEFAULTS).
OLD = {"grow_mod": 0, "split_r_end": 10 ** 9, "dead_end": 0.0, "need_floor": 0, "voro": 0.0, "voro_radius": 8}
OPPONENTS = {
    "defaults": ("brain", {}),
    "old": ("brain", OLD),
    "slow": ("brain", {"split_len": 8, "founder_split_len": 8}),
    "splitter": ("bot", "splitter"),
    "chaser": ("bot", "chaser"),
}


def opponent(name):
    """A registered name, or the path of a tuned-parameters JSON (a past result to beat)."""
    if name in OPPONENTS:
        return OPPONENTS[name]
    return "brain", json.loads(pathlib.Path(name).read_text())["params"]


def parse_vs(text):
    """'defaults:2,old' -> [('defaults', 2.0), ('old', 1.0)]. A trailing :number is the weight (paths may hold colons)."""
    out = []
    for item in filter(None, text.split(",")):
        name, sep, w = item.rpartition(":")
        try:
            out.append((name, float(w)) if sep and name else (item, 1.0))
        except ValueError:
            out.append((item, 1.0))
    return out


# ---------------------------------------------------------------------- parameter encoding
def decode(u, names):
    """Unit-box vector -> {name: value} (clipped to the box, ints rounded)."""
    out = {}
    for x, name in zip(u, names):
        lo, hi, scale = SPACE[name]
        x = min(1.0, max(0.0, x))
        v = lo * (hi / lo) ** x if scale == "log" else lo + x * (hi - lo)
        out[name] = int(round(v)) if scale == "int" else round(v, 4)
    return out


def encode(params, names):
    out = []
    for name in names:
        lo, hi, scale = SPACE[name]
        v = min(hi, max(lo, params.get(name, DEFAULTS[name])))
        out.append(math.log(v / lo) / math.log(hi / lo) if scale == "log" else (v - lo) / (hi - lo))
    return out


def changed(params):
    """Only what differs from DEFAULTS, so a result reads as a diff (a default outside SPACE counts as its clamp)."""
    out = {}
    for k, v in params.items():
        lo, hi, scale = SPACE[k]
        d = min(hi, max(lo, DEFAULTS[k]))
        if v != (int(round(d)) if scale == "int" else round(d, 4)):
            out[k] = v
    return out


# ---------------------------------------------------------------------- separable CMA-ES (maximizes)
class SepCMA:
    def __init__(self, x0, sigma, popsize):
        n = self.n = len(x0)
        self.lam, self.mu = popsize, popsize // 2
        raw = [math.log(self.mu + 0.5) - math.log(i + 1) for i in range(self.mu)]
        self.w = [r / sum(raw) for r in raw]
        self.mueff = 1 / sum(w * w for w in self.w)
        me = self.mueff
        self.cs = (me + 2) / (n + me + 5)
        self.ds = 1 + 2 * max(0.0, math.sqrt((me - 1) / (n + 1)) - 1) + self.cs
        self.cc = (4 + me / n) / (n + 4 + 2 * me / n)
        c1 = 2 / ((n + 1.3) ** 2 + me)
        cmu = 2 * (me - 2 + 1 / me) / ((n + 2) ** 2 + me)
        speed = (n + 2) / 3  # Ros & Hansen: a diagonal covariance can learn faster
        self.c1 = min(1.0, c1 * speed)
        self.cmu = min(1 - self.c1, cmu * speed)
        self.chi = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.m, self.sigma = list(x0), sigma
        self.C, self.ps, self.pc, self.gen = [1.0] * n, [0.0] * n, [0.0] * n, 0
        self.ys = []

    def ask(self, rng):
        """lam points (unclipped); the caller clips them when it evaluates."""
        self.ys = [[math.sqrt(c) * rng.gauss(0, 1) for c in self.C] for _ in range(self.lam)]
        return [[m + self.sigma * y for m, y in zip(self.m, ys)] for ys in self.ys]

    def tell(self, fitness):
        n, w = self.n, self.w
        best = sorted(range(self.lam), key=lambda k: -fitness[k])[: self.mu]
        yw = [sum(w[j] * self.ys[best[j]][i] for j in range(self.mu)) for i in range(n)]
        self.m = [min(1.0, max(0.0, m + self.sigma * y)) for m, y in zip(self.m, yw)]
        k = math.sqrt(self.cs * (2 - self.cs) * self.mueff)
        self.ps = [(1 - self.cs) * p + k * y / math.sqrt(c) for p, y, c in zip(self.ps, yw, self.C)]
        norm = math.sqrt(sum(p * p for p in self.ps))
        self.gen += 1
        hs = 1.0 if norm / math.sqrt(1 - (1 - self.cs) ** (2 * self.gen)) / self.chi < 1.4 + 2 / (n + 1) else 0.0
        kc = math.sqrt(self.cc * (2 - self.cc) * self.mueff)
        self.pc = [(1 - self.cc) * p + hs * kc * y for p, y in zip(self.pc, yw)]
        for i in range(n):
            rank_mu = sum(w[j] * self.ys[best[j]][i] ** 2 for j in range(self.mu))
            self.C[i] = ((1 - self.c1 - self.cmu) * self.C[i]
                         + self.c1 * (self.pc[i] ** 2 + (1 - hs) * self.cc * (2 - self.cc) * self.C[i])
                         + self.cmu * rank_mu)
        self.sigma *= math.exp(self.cs / self.ds * (norm / self.chi - 1))

    def state(self):
        return {k: getattr(self, k) for k in ("m", "sigma", "C", "ps", "pc", "gen")}

    def load(self, state):
        for k, v in state.items():
            setattr(self, k, v)


# ---------------------------------------------------------------------- evaluation
def bundled(max_side):
    out = []
    for p in sorted(MAPS.glob("*.map")):
        _, w, h = p.open().readline().split()
        if max(int(w), int(h)) <= max_side:
            out.append(str(p))
    return out


def play_all(lg, candidates, maps, vs):
    """candidates: list of param dicts. Returns per candidate {opponent name: [Result]}, in job order per opponent."""
    jobs, index = [], []
    for c, params in enumerate(candidates):
        for name, _ in vs:
            for m in maps:
                for side in "AB":
                    jobs.append(Job(m, side, params, opponent(name)))
                    index.append((c, name))
    per = [{name: [] for name, _ in vs} for _ in candidates]
    for (c, name), r in zip(index, lg.run(jobs)):
        per[c][name].append(r)
    return per


def fitness(per_opp, vs):
    total = sum(w for _, w in vs)
    return sum(w * summarize(per_opp[name])["score"] for name, w in vs) / total


def run(args):
    vs = parse_vs(args.vs)
    path = TUNED / f"{args.name}.json"
    TUNED.mkdir(exist_ok=True)
    saved = json.loads(path.read_text()) if args.resume else None
    names = saved["names"] if saved else (args.params.split(",") if args.params else list(SPACE))
    start = json.loads(pathlib.Path(args.start).read_text())["params"] if args.start else {}
    es = SepCMA(encode(start, names), args.sigma, args.pop)
    history = []
    if saved:
        es.load(saved["state"])
        history = saved["history"]
    anchors = bundled(args.max_side) if not args.no_bundled else []
    print(f"tuning {len(names)} parameters, population {args.pop}, {args.maps} generated + {len(anchors)} bundled maps "
          f"x 2 sides x {len(vs)} opponents = {(args.maps + len(anchors)) * 2 * len(vs)} games per candidate", flush=True)
    with League(args.workers) as lg:
        while es.gen < args.generations:
            t0 = time.perf_counter()
            g = es.gen
            maps = [(args.seed + g * args.maps + i, 10, args.max_side) for i in range(args.maps)] + anchors
            points = es.ask(random.Random(f"tune/{args.name}/{g}"))
            mean_params = decode(es.m, names)  # the last candidate is the mean this generation started from
            per = play_all(lg, [decode(p, names) for p in points] + [mean_params], maps, vs)
            fit = [fitness(p, vs) for p in per]
            es.tell(fit[:-1])
            mean_sum = {n: summarize(per[-1][n]) for n, _ in vs}
            games = sum(len(rs) for p in per for rs in p.values())
            history.append({"gen": g, "best": max(fit[:-1]), "avg": sum(fit[:-1]) / args.pop, "mean": fit[-1],
                            "sigma": es.sigma, "mean_params": changed(mean_params),
                            "mean_vs": {n: round(s["score"], 3) for n, s in mean_sum.items()},
                            "mean_deaths_per_k": round(sum(s["deaths_per_k"] for s in mean_sum.values()) / len(vs), 2),
                            "errors": sum(s["errors"] for s in mean_sum.values())})
            h = history[-1]
            print(f"gen {g:3d}  best {h['best']:.3f}  avg {h['avg']:.3f}  mean {h['mean']:.3f} "
                  f"{h['mean_vs']}  deaths/k {h['mean_deaths_per_k']}  sigma {es.sigma:.3f}  "
                  f"{time.perf_counter() - t0:.0f}s ({games / (time.perf_counter() - t0):.1f} games/s)", flush=True)
            path.write_text(json.dumps({"names": names, "params": changed(decode(es.m, names)), "state": es.state(),
                                        "history": history, "args": vars(args)}, indent=1))
    print(f"\nwrote {path}\nparameters that differ from DEFAULTS: {changed(decode(es.m, names))}")


def average(args):
    """The final mean of a noisy run is one draw; the mean of the last few generation means is steadier."""
    saved = json.loads((TUNED / f"{args.name}.json").read_text())
    names = saved["names"]
    rows = [encode(h["mean_params"], names) for h in saved["history"]][-(args.last - 1):] + [encode(saved["params"], names)]
    mean = [sum(col) / len(rows) for col in zip(*rows)]
    out = TUNED / f"{args.name}_avg{args.last}.json"
    out.write_text(json.dumps({"names": names, "params": changed(decode(mean, names)),
                               "source": f"mean of the last {len(rows)} means of {args.name}"}, indent=1))
    print(f"wrote {out}\n{changed(decode(mean, names))}")


def validate(args):
    params = json.loads(pathlib.Path(args.params).read_text())["params"]
    vs = parse_vs(args.vs)
    n_maps = math.ceil(args.games / 2)
    generated = [(args.seed + i, 10, args.max_side) for i in range(n_maps)]
    real = bundled(64)
    print(f"candidate {params or '(= DEFAULTS)'}\n{n_maps} generated maps x 2 sides = {2 * n_maps} games per opponent, "
          f"plus {len(real)} bundled maps x 2 sides\n")
    with League(args.workers) as lg:
        for name, _ in vs:
            for label, maps in (("generated", generated), ("bundled", real)):
                per = play_all(lg, [params], maps, [(name, 1.0)])[0][name]
                s = summarize(per)
                lo, hi = wilson(s["score"], s["games"])
                gate = ""
                if label == "generated" and name == "defaults":
                    gate = "  PASS (>= 55% over >= 400 games)" if s["score"] >= 0.55 and s["games"] >= 400 else \
                           "  (gate: >= 55% over >= 400 games)"
                print(f"vs {name:9s} {label:9s} {s['games']:4d} games: {s['score']:.1%} [{lo:.1%}, {hi:.1%}]  "
                      f"W{s['wins']} D{s['draws']} L{s['games'] - s['wins'] - s['draws']}  "
                      f"my deaths/1000 turns {s['deaths_per_k']:.1f}  errors {s['errors']}{gate}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--name", required=True)
    r.add_argument("--generations", type=int, default=20)
    r.add_argument("--pop", type=int, default=16)
    r.add_argument("--maps", type=int, default=10, help="fresh generated maps per generation")
    r.add_argument("--max-side", type=int, default=32)
    r.add_argument("--seed", type=int, default=1000, help="first map seed; validation uses 100000 up")
    r.add_argument("--sigma", type=float, default=0.15, help="initial step size in the unit box")
    r.add_argument("--vs", default="defaults:2,old:2", help="opponents and weights; a .json path adds a past result")
    r.add_argument("--params", default="", help="comma list of parameters to tune (default: all of SPACE)")
    r.add_argument("--start", default="", help="tuned .json to start from (default: brain.DEFAULTS)")
    r.add_argument("--no-bundled", action="store_true")
    r.add_argument("--resume", action="store_true")
    r.add_argument("--workers", type=int, default=None)
    a = sub.add_parser("average", help="average the last few generation means of a run into NAME_avgK.json")
    a.add_argument("name")
    a.add_argument("--last", type=int, default=5)
    v = sub.add_parser("validate")
    v.add_argument("params", help="tuned .json")
    v.add_argument("--games", type=int, default=400, help="generated-map games per opponent")
    v.add_argument("--vs", default="defaults,old,splitter")
    v.add_argument("--seed", type=int, default=100000)
    v.add_argument("--max-side", type=int, default=64)
    v.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    {"run": run, "average": average, "validate": validate}[args.cmd](args)


if __name__ == "__main__":
    main()
