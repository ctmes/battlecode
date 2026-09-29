"""Brain parameter sets against frozen opponents on ladder-map variants, reporting what decides ladder games: win
rate, and each side's longest living dragon at round 500 (the first tiebreak), plus how the games were lost, where
each side's pearls came from (tools/forage.py), and the score on fountain maps vs the rest.

    .venv\\Scripts\\python.exe tools\\longest_bench.py candidates.json [--vs live,grower_brain,v3] [--variants 6]
        [--holdout] [--workers N] [--wall-budget]

candidates.json maps a label to a params dict (merged over bot/brain.py DEFAULTS), e.g.
    {"base": {}, "king": {"king_first": 2, "king_len": 6, "king_care": 2.0}}
By default the games use training variants 1..N of each ladder-pool map; --holdout uses the reserved decision set
(the maps as played on the ladder plus reserved variants) instead -- keep that for final accept/reject calls.

Every Brain-family player gets an effectively unlimited budget_ns unless --wall-budget: locally the budget is wall
time, so under CPU load (a busy pool, another session's run) the cutoff fires at random and the same game plays out
differently (13 of 40 mirrored pairs differed on 28 Sep). On the judge the Brain uses <= 12.6M of its 60M points,
so the cutoff is not what the ladder sees either.
"""
import argparse
import collections
import concurrent.futures
import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bot"))
sys.path.insert(0, str(ROOT / "tools"))

NO_BUDGET = {"budget_ns": 10 ** 13}
_engine = None


def _init():
    global _engine
    from brain import Brain
    from unswbc.engine import EngineModule
    Brain.strict, Brain.debug = True, False
    _engine = EngineModule()


def with_budget(spec, extra):
    """The opponent spec with `extra` merged into a Brain-family opponent's params (sparring bots unchanged)."""
    if not extra:
        return spec
    if spec[0] == "brain":
        return "brain", {**(spec[1] or {}), **extra}
    if spec[0] == "frozen":
        return "frozen", (spec[1][0], {**(spec[1][1] or {}), **extra})
    return spec


def play(job):
    import arena
    import forage
    import ladder_maps
    import league
    import replay_parse
    import tune
    label, params, opp, spec, side, seed, extra = job
    me = arena.BrainPlayer("me", {**(params or {}), **extra} or None)
    foe = league._opponent(with_budget(tune.opponent(opp), extra), 0)
    res, deaths, errors = arena.play(_engine, league.map_bytes(spec), *((me, foe) if side == "A" else (foe, me)),
                                     seed=seed)
    data = _engine.replay("a", "b")
    r = replay_parse.parse(data)
    f = forage.analyse(data, ladder_maps.variant(spec[1], spec[2]).dumps())
    them = "B" if side == "A" else "A"
    return {"label": label, "opp": opp, "map": spec[1], "score": 0.5 if res.winner is None else float(res.winner == side),
            "rounds": r["rounds"], "us": r["standing"][side], "them": r["standing"][them], "errors": len(errors),
            "food_us": dict(f[side]), "food_them": dict(f[them])}


def wilson(p, n, z=1.96):
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    import forage
    import ladder_maps
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates")
    ap.add_argument("--vs", default="live,grower_brain,v3")
    ap.add_argument("--variants", type=int, default=6)
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--wall-budget", action="store_true", help="keep the 60M wall-clock budget_ns (nondeterministic under load)")
    ap.add_argument("--maps", default="", help="comma list of ladder-pool map names to keep (default: all)")
    ap.add_argument("--seeds", type=int, default=0,
                    help="play the ladder maps as played, with this many engine seeds each, instead of variants "
                         "(unswbc >= 1.0; the judge seeds every match); --holdout takes the seeds from 1,000,000 up")
    args = ap.parse_args()
    cands = json.loads(pathlib.Path(args.candidates).read_text())
    if args.seeds:
        base = 1_000_000 if args.holdout else 0
        games = [(("ladder", n, 0), base + s) for n in ladder_maps.POOL for s in range(1, args.seeds + 1)]
    else:
        games = [(m, None) for m in (ladder_maps.holdout_specs(args.variants - 1) if args.holdout else
                                     [("ladder", n, v) for n in ladder_maps.POOL for v in range(1, args.variants + 1)])]
    if args.maps:
        keep = set(args.maps.split(","))
        games = [g for g in games if g[0][1] in keep]
    opps = [o for o in args.vs.split(",") if o]
    extra = {} if args.wall_budget else NO_BUDGET
    jobs = [(lab, p, o, m, s, seed, extra) for lab, p in cands.items() for o in opps for m, seed in games for s in "AB"]
    workers = args.workers or 12
    with concurrent.futures.ProcessPoolExecutor(workers, initializer=_init) as ex:
        res = list(ex.map(play, jobs, chunksize=2))
    kind = "engine seeds on the maps as played" if args.seeds else "variants"
    print(f"{len(res)} games, {len(games)} maps x 2 sides per pairing ({'holdout' if args.holdout else 'training'} "
          f"{kind}, {'wall-clock budget' if args.wall_budget else 'no budget cutoff'})\n")
    print(f"{'candidate':14s} {'vs':14s} {'score':>18s} {'fountain/other':>15s} {'longest@500 us/them':>20s} "
          f"{'tiebreak losses':>16s} {'eliminated':>11s} {'errors':>6s} | {'eat/100 us/them':>15s} "
          f"{'fountain/game':>14s} {'deaths/1000 us/them':>20s}")
    for lab in cands:
        for o in opps:
            rs = [r for r in res if r["label"] == lab and r["opp"] == o]
            sc = sum(r["score"] for r in rs) / len(rs)
            lo, hi = wilson(sc, len(rs))
            fm = [r["score"] for r in rs if r["map"] in forage.FOUNTAIN_MAPS]
            om = [r["score"] for r in rs if r["map"] not in forage.FOUNTAIN_MAPS]
            fin = [r for r in rs if r["rounds"] >= 499]
            lu = sum(r["us"][1] for r in fin) / max(len(fin), 1)
            lt = sum(r["them"][1] for r in fin) / max(len(fin), 1)
            tb = sum(r["score"] == 0 and r["rounds"] >= 499 and r["us"][1] < r["them"][1] for r in rs)
            el = sum(r["us"][0] == 0 for r in rs)
            fu, ft = collections.Counter(), collections.Counter()
            for r in rs:
                fu.update(r["food_us"])
                ft.update(r["food_them"])
            print(f"{lab:14s} {o:14s} {sc:6.1%} [{lo:4.0%}-{hi:4.0%}] {sum(fm) / max(len(fm), 1):6.0%} / "
                  f"{sum(om) / max(len(om), 1):<6.0%} {lu:9.1f} / {lt:<8.1f} {tb:16d} {el:11d} "
                  f"{sum(r['errors'] for r in rs):6d} | {100 * fu['eaten'] / max(fu['turns'], 1):6.1f} / "
                  f"{100 * ft['eaten'] / max(ft['turns'], 1):<6.1f} {fu['fountain'] / len(rs):5.0f} / "
                  f"{ft['fountain'] / len(rs):<6.0f} {1000 * fu['deaths'] / max(fu['turns'], 1):9.1f} / "
                  f"{1000 * ft['deaths'] / max(ft['turns'], 1):<8.1f}")
    by_map = collections.defaultdict(list)
    for r in res:
        by_map[(r["label"], r["map"])].append(r["score"])
    print("\nscore by map (all opponents pooled):")
    names = sorted({r["map"] for r in res})
    print(f"{'candidate':14s} " + " ".join(f"{n[:9]:>9s}" for n in names))
    for lab in cands:
        print(f"{lab:14s} " + " ".join(f"{sum(by_map[(lab, n)]) / len(by_map[(lab, n)]):9.0%}" for n in names))
    out = pathlib.Path(args.candidates).with_suffix(".results.json")
    out.write_text(json.dumps(res))


if __name__ == "__main__":
    main()
