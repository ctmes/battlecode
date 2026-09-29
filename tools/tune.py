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
from collections import namedtuple

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from league import Job, League, summarize, wilson  # noqa: E402  (also puts bot/ on the path)
import ladder_maps  # noqa: E402
from brain import DEFAULTS  # noqa: E402

TUNED = ROOT / "tools" / "tuned"
MAPS = ROOT / "maps"

# name: (low, high, scale). "log" spreads a parameter over orders of magnitude, "int" rounds. Left out on purpose:
# squeeze / deny* (herding did not work, see the strategy notes), pessimistic, need_cap and budget_ns (not behaviour).
SPACE = {
    # pearl_near, trap and straight were widened after run1 (see tuned/run1.json): its final mean pegged pearl_near
    # at 99% of (2, 100) and trap and straight exactly at the top of (300, 12000) and (0, 40) -- the optimizer wanted
    # to go further. The rest of the "at a bound" values in that run (split_len/split_child/founder_split_len at
    # their floor, split_min_exits=1, tiles_per_unit=0, split_pearls=0) are floors the game or want_split() already
    # impose regardless of SPACE (e.g. split_child's minimum of 2 is the engine's own minimum split length), so
    # widening them further would not change behaviour.
    "pearl_here": (200, 3000, "log"), "pearl_near": (2, 400, "log"), "pearl_k": (2, 10, "int"),
    "trap": (300, 40000, "log"), "need_margin": (0, 8, "int"), "area": (0, 60, "lin"),
    "head_risk": (0, 1500, "lin"), "head_risk_small": (0, 600, "lin"), "trade_ratio": (0.2, 1.2, "lin"),
    "team_head_risk": (0, 800, "lin"), "straight": (0, 120, "lin"),
    "split_len": (3, 12, "int"), "split_child": (2, 4, "int"), "founder_split_len": (3, 12, "int"),
    # grow_mod starts at 2: 0 (every child swarms) and 1 (no child ever splits) are opposite extremes one integer
    # apart, a cliff the search kept falling off (full_run1 gens 23-24).
    "grow_mod": (2, 16, "int"), "split_r_end": (100, 500, "int"), "split_min_exits": (1, 3, "int"),
    "tiles_per_unit": (0, 150, "int"), "split_pearls": (0, 4, "int"), "split_units": (4, 64, "int"),
    "founder_units": (2, 64, "int"), "dead_end": (0, 400, "lin"), "need_floor": (0, 60, "int"),
    "voro": (0, 12, "lin"), "voro_radius": (2, 9, "int"),
    # Added for the length-tiebreak weakness (a dragon that is already long plays more cautiously instead of
    # risking the investment): 0 / 1.0 / 0 are all no-ops, so the optimizer can tune this back off if it doesn't
    # help. See tools/tuned/ for grow_care-only runs (--params grow_care_len,grow_care_mult,grow_care_margin).
    # grow_care_len starts at 1, not 0 (= off): with grow_care_mult 1.0 and grow_care_margin 0 the mechanism is a
    # no-op at any length, so its strength is tuned smoothly through those two instead of an on/off step.
    "grow_care_len": (1, 30, "int"), "grow_care_mult": (1.0, 4.0, "lin"), "grow_care_margin": (0, 6, "int"),
    # Local crowding/food gate on splitting (see want_split): refuses to split into a spot already thick with
    # teammates, or short on pearls per nearby mouth, instead of only a flat team-size cap. 0 is off for both.
    "split_mate_radius": (0, 6, "int"), "split_mate_cap": (1, 6, "int"), "split_food_ratio": (0, 3, "lin"),
    # A length ceiling on splitting: the existing split_len/founder_split_len are floors (minimum length to be
    # split-eligible), never a ceiling, so nothing previously stopped an already-long dragon splitting itself
    # away the instant local conditions allowed. 0 = off, which OFF_AT_TOP encodes as the top of the range: values
    # 1-5 used to sit right next to "off" and disable splitting outright (bot/brain.py want_split), so every CMA
    # mean that drifted there scored 0.000 against every opponent (vs_manual_run1 gens 8 and 11).
    "split_len_max": (6, 64, "int"),
    # Explore: with no pearl signal nearby, reward a move onto never-seen ground instead of only "straight"
    # (which can loop a dragon back through already-searched-empty space). 0 = off.
    "explore": (0, 200, "lin"),
    # Sprint: cover up to this many tiles in one turn (costing length) instead of always 1. 1 = off.
    "sprint_max": (1, 3, "int"),
    # sonar_terrain is deliberately absent: brain.py only tests it for > 0, so its magnitude does nothing and CMA
    # would be tuning noise. Decide it by a direct on/off A/B instead.
    # mh4's mechanisms (29 Sep), each range clear of its 0 = off: topstyle's late consolidation and rams, the dive,
    # the turnaround split, and the portal exploration bets.
    "boxed_r_end": (150, 480, "int"), "ram": (100, 3000, "log"), "ram_len": (2, 5, "int"),
    "dive_len": (3, 40, "int"), "dive_trap": (0, 1, "lin"), "dive_radius": (2, 14, "int"),
    "boxed_rear_len": (4, 10, "int"), "boxed_reserve": (0, 8, "int"),
    "portal_unknown": (0, 400, "lin"), "portal_blind": (0, 300, "lin"),
}
# Parameters whose "off" value (0) lies outside their range: encoded as the range's top, decoded back to 0 there.
OFF_AT_TOP = {"split_len_max"}
# The brain as it was before the dead-end / room-floor / grower / territory work (git HEAD~ of brain.py's DEFAULTS).
OLD = {"grow_mod": 0, "split_r_end": 10 ** 9, "dead_end": 0.0, "need_floor": 0, "voro": 0.0, "voro_radius": 8}
OPPONENTS = {
    "defaults": ("brain", {}),
    "old": ("brain", OLD),
    "slow": ("brain", {"split_len": 8, "founder_split_len": 8}),
    "splitter": ("bot", "splitter"),
    "chaser": ("bot", "chaser"),
    # A disciplined opponent that splits a little early, then stops well before round 500 and just survives and
    # grows: this is what reproduces the real loss (more total length and dragons, still loses the round-500
    # longest-living-dragon tiebreak) that plain "defaults"/"old" self-play does not reliably exercise.
    "grower_brain": ("brain", {"founder_units": 10, "split_units": 10, "split_r_end": 150, "grow_mod": 0}),
    # The previously-shipped trained Policy (bot/weights.bin) and a previous from-scratch submission
    # (manual-heuristics/): both registered in tools/opponents.py since they need their own isolated code/weight
    # loading, not just a params dict. Added 2026-09-27 so tuning accounts for being heavily outnumbered (Policy
    # fields ~8x the dragons) and for a genuinely different heuristic design (manual-heuristics), not just
    # variations on the current brain.py.
    "policy": ("bot", "policy"),
    "manual_heuristics": ("bot", "manual_heuristics"),
    # Frozen opponents (code AND parameters pinned; see snapshots/README.md): unlike "defaults"/"old", which follow
    # whatever bot/brain.py DEFAULTS say at import time, these never move when the bot changes.
    "live": ("frozen", (str(ROOT / "manual-heuristics"), {})),  # v2, on the ladder 25-28 Sep
    "mh2": ("frozen", (str(ROOT / "snapshots" / "mh2_2026-09-28"), {})),  # v2 + boxed_split, submitted 28 Sep 06:40Z
    # mh2 + spare_team + boxed_reserve + rear split, submitted 28 Sep
    "mh3": ("frozen", (str(ROOT / "snapshots" / "mh3_2026-09-28"), {})),
    # mh3 whose short dragons ignore the trap check: forages like the ladder (7.7 pearls per 100 dragon-turns, 27
    # deaths per 1000, eats fountains) where every other local opponent eats 3-4.6 like we do (tools/forage.py)
    "farmer": ("frozen", (str(ROOT / "snapshots" / "farmer_2026-09-29"),
                          {"dive_len": 3, "dive_trap": 0.0, "dive_scope": 0})),
    # the farmer playing like the ladder's top 3 (tools/scout.py): stops splitting at round 300, turns around all
    # game, rams enemy heads with short dragons. 75.5% vs mh3 on held-out seeds (bot/brain.py DEFAULTS comment)
    "topstyle": ("frozen", (str(ROOT / "snapshots" / "topstyle_2026-09-29"),
                            {"split_r_end": 300, "boxed_r_end": 300, "boxed_rear_r": 0, "ram_len": 3, "ram": 1000.0,
                             "dive_len": 3, "dive_trap": 0.0, "dive_scope": 0})),
    # mh3 + portals (learned pairs, known-map oracle, oracle_seen), 29 Sep; not submitted as of this snapshot
    "mh3p": ("frozen", (str(ROOT / "snapshots" / "mh3p_2026-09-29"), {})),
    # mh3 + portals + topstyle + diving near fountains (any length); a submittable bot folder, DEFAULTS = mh4
    "mh4": ("frozen", (str(ROOT / "snapshots" / "mh4_2026-09-29"), {})),
    "v3": ("frozen", (str(ROOT / "snapshots" / "brain_2026-09-27"),
                      {"grow_care_len": 1, "grow_care_mult": 1.9346, "grow_care_margin": 1})),
    "tuned_0927": ("frozen", (str(ROOT / "snapshots" / "brain_2026-09-27"),
                              {"grow_mod": 9, "founder_units": 57, "split_units": 64, "tiles_per_unit": 1,
                               "split_r_end": 418, "grow_care_len": 13, "grow_care_mult": 1.3835,
                               "grow_care_margin": 3, "split_len_max": 41, "split_mate_cap": 3,
                               "split_food_ratio": 0.1705})),  # vs_manual_run1's final mean
    "grower": ("bot", "grower"),
    # the grower, but it walks through portals (see tools/opponents.py PortalGrower): the only local opponent that
    # reaches the pearls behind portals on Portals and Trauma, as the ladder's opponents do
    "portal_grower": ("bot", "portal_grower"),
    # Brains that use portals (bot/brain.py "portals"), following the live DEFAULTS like grower_brain does: the
    # sparring bot above is too weak to matter (mh3 beat it 99%), and every frozen Brain treats portals as walls
    # (portal params pinned to the first working version, so they stay put while the DEFAULTS are tuned)
    "portal_brain": ("brain", {"portals": 1, "portal_unknown": 150.0, "portal_blind": 100.0, "portal_learn": 0,
                               "map_oracle": 0}),
    "grower_portal": ("brain", {"founder_units": 10, "split_units": 10, "split_r_end": 150, "grow_mod": 0,
                                "portals": 1, "portal_unknown": 150.0, "portal_blind": 100.0, "portal_learn": 0,
                                "map_oracle": 0}),
    # grower_brain above follows the live DEFAULTS (so since 28 Sep it splits when boxed in too); this is the
    # original, pinned to the 27 Sep code, for comparisons across time.
    "grower_frozen": ("frozen", (str(ROOT / "snapshots" / "brain_2026-09-27"),
                                 {"founder_units": 10, "split_units": 10, "split_r_end": 150, "grow_mod": 0})),
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
def reflect(x):
    """Folds any real number into [0, 1] by mirroring at the bounds (period 2). Unlike clipping, this keeps the
    objective defined and continuous outside the box, so CMA-ES can sample and move there without the evolution
    path piling up against a wall -- the clip-the-mean version inflated sigma every generation whenever the optimum
    sat on a bound (0.15 -> 3.9 in 30 generations on a toy problem; full_run1 went 0.138 -> 1.989)."""
    x %= 2.0
    return 2.0 - x if x > 1.0 else x


def decode(u, names):
    """Unconstrained vector -> {name: value} (reflected into the box, ints rounded)."""
    out = {}
    for x, name in zip(u, names):
        lo, hi, scale = SPACE[name]
        x = reflect(x)
        v = lo * (hi / lo) ** x if scale == "log" else lo + x * (hi - lo)
        v = int(round(v)) if scale == "int" else round(v, 4)
        out[name] = 0 if name in OFF_AT_TOP and v >= hi else v
    return out


def encode(params, names):
    out = []
    for name in names:
        lo, hi, scale = SPACE[name]
        v = params.get(name, DEFAULTS[name])
        v = hi if name in OFF_AT_TOP and v == 0 else min(hi, max(lo, v))
        out.append(math.log(v / lo) / math.log(hi / lo) if scale == "log" else (v - lo) / (hi - lo))
    return out


def changed(params):
    """Only what differs from DEFAULTS, so a result reads as a diff (a default outside SPACE counts as its clamp)."""
    out = {}
    for k, v in params.items():
        lo, hi, scale = SPACE[k]
        d = DEFAULTS[k] if k in OFF_AT_TOP and DEFAULTS[k] == 0 else min(hi, max(lo, DEFAULTS[k]))
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
        """lam points in the unconstrained space; decode() reflects them into the box when they are evaluated."""
        self.ys = [[math.sqrt(c) * rng.gauss(0, 1) for c in self.C] for _ in range(self.lam)]
        return [[m + self.sigma * y for m, y in zip(self.m, ys)] for ys in self.ys]

    def tell(self, fitness):
        n, w = self.n, self.w
        best = sorted(range(self.lam), key=lambda k: -fitness[k])[: self.mu]
        yw = [sum(w[j] * self.ys[best[j]][i] for j in range(self.mu)) for i in range(n)]
        self.m = [m + self.sigma * y for m, y in zip(self.m, yw)]  # unclipped: see reflect()
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


# A map played with a given engine seed (unswbc >= 1.0 only: the ladder's own maps x the judge's per-match seeds)
SeededMap = namedtuple("SeededMap", "spec seed")


def play_all(lg, candidates, maps, vs):
    """candidates: list of param dicts; maps: map specs or SeededMaps. Returns per candidate {opponent name:
    [Result]}, in job order per opponent."""
    jobs, index = [], []
    for c, params in enumerate(candidates):
        for name, _ in vs:
            for m in maps:
                spec, seed = (m.spec, m.seed) if isinstance(m, SeededMap) else (m, None)
                for side in "AB":
                    jobs.append(Job(spec, side, params, opponent(name), seed))
                    index.append((c, name))
    per = [{name: [] for name, _ in vs} for _ in candidates]
    for (c, name), r in zip(index, lg.run(jobs)):
        per[c][name].append(r)
    return per


def fitness(per_opp, vs):
    total = sum(w for _, w in vs)
    return sum(w * summarize(per_opp[name])["score"] for name, w in vs) / total


RUN_DEFAULTS = {"generations": 20, "pop": 16, "maps": 10, "max_side": 32, "seed": 1000, "sigma": 0.15,
                "vs": "defaults:2,old:2", "params": "", "start": "", "no_bundled": False, "ladder": 0,
                "seeds": 0, "base": ""}


def run(args):
    path = TUNED / f"{args.name}.json"
    TUNED.mkdir(exist_ok=True)
    saved = json.loads(path.read_text()) if args.resume and path.exists() else None
    # Every setting that shapes the CMA-ES state or the evaluation (population size above all: SepCMA's weights
    # and learning rates are derived from it, and only m/sigma/C/ps/pc/gen round-trip through the saved state) must
    # match the original run to continue it rather than silently start a different search. Anything left unset on
    # the command line (still None) falls back to the resumed run's own settings, then to RUN_DEFAULTS.
    for key, fallback in RUN_DEFAULTS.items():
        if getattr(args, key, None) is None:  # absent (programmatic callers) counts as unset
            setattr(args, key, (saved["args"].get(key, fallback) if saved else fallback))  # .get: older runs
    vs = parse_vs(args.vs)
    names = saved["names"] if saved else (args.params.split(",") if args.params else list(SPACE))
    # base: fixed settings under every candidate (e.g. mh4's), so only `names` move; also the default start point
    base = json.loads(pathlib.Path(args.base).read_text())["params"] if args.base else {}
    start = json.loads(pathlib.Path(args.start).read_text())["params"] if args.start else base
    es = SepCMA(encode(start, names), args.sigma, args.pop)
    history = []
    if saved:
        es.load(saved["state"])
        history = saved["history"]
    anchors = bundled(args.max_side) if not args.no_bundled else []
    n_ladder = (args.ladder + args.seeds) * len(ladder_maps.POOL)
    print(f"tuning {len(names)} parameters, population {args.pop}, {args.maps} generated + {len(anchors)} bundled + "
          f"{n_ladder} ladder maps ({'engine seeds' if args.seeds else 'variants'}) x 2 sides x {len(vs)} opponents = "
          f"{(args.maps + len(anchors) + n_ladder) * 2 * len(vs)} games per candidate", flush=True)
    with League(args.workers) as lg:
        while es.gen < args.generations:
            t0 = time.perf_counter()
            g = es.gen
            maps = [(args.seed + g * args.maps + i, 10, args.max_side) for i in range(args.maps)] + anchors
            maps += ladder_maps.train_specs(args.ladder, g) if args.ladder else []  # fresh variants, never held-out ones
            # the maps as played with fresh engine seeds; ship decisions use 9,000,000 up (tools/bench1x.py)
            maps += [SeededMap(("ladder", n, 0), args.seed + g * args.seeds + i)
                     for n in ladder_maps.POOL for i in range(args.seeds)]
            points = es.ask(random.Random(f"tune/{args.name}/{g}"))
            mean_params = decode(es.m, names)  # the last candidate is the mean this generation started from
            per = play_all(lg, [{**base, **decode(p, names)} for p in points] + [{**base, **mean_params}], maps, vs)
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
                                        "history": history, "args": vars(args), "base": base}, indent=1))
    print(f"\nwrote {path}\nparameters that differ from DEFAULTS: {changed(decode(es.m, names))}")


def average(args):
    """The final mean of a noisy run is one draw; the mean of the last few generation means is steadier."""
    saved = json.loads((TUNED / f"{args.name}.json").read_text())
    names = saved["names"]
    history = saved["history"]
    # k=0 (--last 1) must mean "just the final mean", not "the whole history": history[-(0):] is history[0:], the
    # WHOLE list, not an empty slice -- Python has no negative zero, so a bare -(args.last - 1) silently averaged
    # over every generation ever run whenever --last was 1 (untested edge case; found by inspection, not a failure).
    k = max(args.last - 1, 0)
    tail = history[len(history) - k:] if k else []
    rows = [encode(h["mean_params"], names) for h in tail] + [encode(saved["params"], names)]
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
    ladder = ladder_maps.holdout_specs(args.ladder - 1) if args.ladder > 0 else []
    print(f"candidate {params or '(= DEFAULTS)'}\n{n_maps} generated maps x 2 sides = {2 * n_maps} games per opponent, "
          f"plus {len(real)} bundled maps x 2 sides, plus {len(ladder)} held-out ladder maps x 2 sides\n")
    sets = [("generated", generated), ("bundled", real), ("ladder", ladder)]
    with League(args.workers) as lg:
        for name, _ in vs:
            for label, maps in [(lab, ms) for lab, ms in sets if ms]:
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
    # All of these default to None, not RUN_DEFAULTS' values: on --resume, None means "keep the original run's
    # setting" (see run()); only an explicit flag overrides a resumed run. A fresh run falls back to RUN_DEFAULTS.
    r.add_argument("--name", required=True)
    r.add_argument("--generations", type=int, default=None)
    r.add_argument("--pop", type=int, default=None)
    r.add_argument("--maps", type=int, default=None, help="fresh generated maps per generation")
    r.add_argument("--max-side", type=int, default=None)
    r.add_argument("--seed", type=int, default=None, help="first map seed; validation uses 100000 up")
    r.add_argument("--sigma", type=float, default=None, help="initial step size in the unit box")
    r.add_argument("--vs", default=None, help="opponents and weights; a .json path adds a past result")
    r.add_argument("--params", default=None, help="comma list of parameters to tune (default: all of SPACE)")
    r.add_argument("--start", default=None, help="tuned .json to start from (default: brain.DEFAULTS)")
    r.add_argument("--no-bundled", action="store_true", default=None)
    r.add_argument("--ladder", type=int, default=None,
                   help="fresh variants of each ladder-pool map per generation (tools/ladder_maps.py); 0 = none")
    r.add_argument("--seeds", type=int, default=None,
                   help="engine seeds per ladder-pool map per generation, maps as played (needs unswbc >= 1.0: .venv-1x)")
    r.add_argument("--base", default=None, help="tuned-format .json of fixed params under every candidate (and the "
                                                "start point unless --start)")
    r.add_argument("--resume", action="store_true")
    r.add_argument("--workers", type=int, default=None, help="always auto-detected when omitted, even on --resume")
    a = sub.add_parser("average", help="average the last few generation means of a run into NAME_avgK.json")
    a.add_argument("name")
    a.add_argument("--last", type=int, default=5)
    v = sub.add_parser("validate")
    v.add_argument("params", help="tuned .json")
    v.add_argument("--games", type=int, default=400, help="generated-map games per opponent")
    v.add_argument("--vs", default="defaults,old,splitter")
    v.add_argument("--seed", type=int, default=100000)
    v.add_argument("--max-side", type=int, default=64)
    v.add_argument("--ladder", type=int, default=0,
                   help="held-out ladder maps: each pool map as played, plus this many - 1 reserved variants of it")
    v.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    {"run": run, "average": average, "validate": validate}[args.cmd](args)


if __name__ == "__main__":
    main()
