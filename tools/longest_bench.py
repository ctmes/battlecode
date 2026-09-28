"""Brain parameter sets against frozen opponents on ladder-map variants, reporting what decides ladder games: win
rate, and each side's longest living dragon at round 500 (the first tiebreak), plus how the games were lost.

    .venv\\Scripts\\python.exe tools\\longest_bench.py candidates.json [--vs live,grower_brain,v3] [--variants 6]
        [--holdout] [--workers N]

candidates.json maps a label to a params dict (merged over bot/brain.py DEFAULTS), e.g.
    {"base": {}, "king": {"king_first": 2, "king_len": 6, "king_care": 2.0}}
By default the games use training variants 1..N of each ladder-pool map; --holdout uses the reserved decision set
(the maps as played on the ladder plus reserved variants) instead -- keep that for final accept/reject calls.
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

_engine = None


def _init():
    global _engine
    from brain import Brain
    from unswbc.engine import EngineModule
    Brain.strict, Brain.debug = True, False
    _engine = EngineModule()


def play(job):
    import arena
    import league
    import replay_parse
    import tune
    label, params, opp, spec, side = job
    me = arena.BrainPlayer("me", params or None)
    foe = league._opponent(tune.opponent(opp), 0)
    res, deaths, errors = arena.play(_engine, league.map_bytes(spec), *((me, foe) if side == "A" else (foe, me)))
    r = replay_parse.parse(_engine.replay("a", "b"))
    them = "B" if side == "A" else "A"
    return {"label": label, "opp": opp, "map": spec[1], "score": 0.5 if res.winner is None else float(res.winner == side),
            "rounds": r["rounds"], "us": r["standing"][side], "them": r["standing"][them], "errors": len(errors)}


def wilson(p, n, z=1.96):
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    import ladder_maps
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates")
    ap.add_argument("--vs", default="live,grower_brain,v3")
    ap.add_argument("--variants", type=int, default=6)
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    cands = json.loads(pathlib.Path(args.candidates).read_text())
    maps = ladder_maps.holdout_specs(args.variants - 1) if args.holdout else \
        [("ladder", n, v) for n in ladder_maps.POOL for v in range(1, args.variants + 1)]
    opps = [o for o in args.vs.split(",") if o]
    jobs = [(lab, p, o, m, s) for lab, p in cands.items() for o in opps for m in maps for s in "AB"]
    workers = args.workers or 12
    with concurrent.futures.ProcessPoolExecutor(workers, initializer=_init) as ex:
        res = list(ex.map(play, jobs, chunksize=2))
    print(f"{len(res)} games, {len(maps)} maps x 2 sides per pairing ({'holdout' if args.holdout else 'training variants'})\n")
    print(f"{'candidate':14s} {'vs':14s} {'score':>18s} {'longest@500 us/them':>20s} {'tiebreak losses':>16s} "
          f"{'eliminated':>11s} {'errors':>6s}")
    for lab in cands:
        for o in opps:
            rs = [r for r in res if r["label"] == lab and r["opp"] == o]
            sc = sum(r["score"] for r in rs) / len(rs)
            lo, hi = wilson(sc, len(rs))
            fin = [r for r in rs if r["rounds"] >= 499]
            lu = sum(r["us"][1] for r in fin) / max(len(fin), 1)
            lt = sum(r["them"][1] for r in fin) / max(len(fin), 1)
            tb = sum(r["score"] == 0 and r["rounds"] >= 499 and r["us"][1] < r["them"][1] for r in rs)
            el = sum(r["us"][0] == 0 for r in rs)
            print(f"{lab:14s} {o:14s} {sc:6.1%} [{lo:4.0%}-{hi:4.0%}] {lu:9.1f} / {lt:<8.1f} {tb:16d} {el:11d} "
                  f"{sum(r['errors'] for r in rs):6d}")
    by_map = collections.defaultdict(list)
    for r in res:
        by_map[(r["label"], r["map"])].append(r["score"])
    print("\nscore by map (all opponents pooled):")
    names = sorted({r["map"] for r in res})
    print(f"{'candidate':14s} " + " ".join(f"{n[:9]:>9s}" for n in names))
    for lab in cands:
        print(f"{lab:14s} " + " ".join(f"{sum(by_map[(lab, n)]) / len(by_map[(lab, n)]):9.0%}" for n in names))


if __name__ == "__main__":
    main()
