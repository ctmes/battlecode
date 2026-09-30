"""Local bench on the ladder's engine (unswbc 1.x): the real ladder maps x random seeds, both seats, frozen candidates
against frozen opponents -- including bench-only opponents that are handed the true map, so they use portals.

The ladder runs unswbc 1.x. The old .venv has 0.3.6, whose engine takes no seed, so every map gave the same game;
run this with the 1.x environment instead:

    .venv-1x\\Scripts\\python.exe tools\\bench1x.py candidates.json [--vs mh3,farmer,portal_farmer] [--seeds 10]
        [--seed-base 1] [--variants 0] [--maps trophy,default] [--workers 14] [--wall-budget] [--out results.json]

candidates.json maps a label to {"dir": snapshot folder, "params": {...}}, or to {} when the label names a frozen
opponent in tools/tune.py (e.g. {"mh3": {}}). Opponents: frozen tune.py names, or the bench-only ones in EXTRA.
Seeds: train and explore from 1 up; keep 9,000,000 up for ship decisions (never tune on those).
Each (map, seed) is played from both seats, so the two games share their pearl draws.
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

# Bench-only opponents: frozen snapshots handed the true map (oracle=True) through Brain.oracle_init().
EXTRA = {
    # farmer + oracle portals: moves through portals and floods across them, like the ladder teams on Portals
    "portal": (str(ROOT / "snapshots" / "portal_2026-09-29"), {"portal_use": 1}, True),
    # ... plus the farmer's fountain diving, the closest thing we have to a ladder team
    "portal_farmer": (str(ROOT / "snapshots" / "portal_2026-09-29"),
                      {"portal_use": 1, "dive_len": 3, "dive_trap": 0.0, "dive_scope": 0}, True),
}
ORACLE_ATTRS = ("kh", "kv", "ph", "pv", "okh", "okv", "seen", "links", "pl_pairs", "pl_entry")

_engine = None


def _init():
    global _engine
    import unswbc
    from unswbc.engine import EngineModule
    if int(unswbc.__version__.split(".")[0]) < 1:
        raise SystemExit(f"unswbc {unswbc.__version__} has no seeded engine: run this with .venv-1x")
    _engine = EngineModule()


class Seeded:
    """The engine with this game's seed filled in (arena.play only calls run())."""

    def __init__(self, engine, seed):
        self.engine, self.seed = engine, seed

    def run(self, *args):
        return self.engine.run(*args, seed=self.seed)


class Player:
    """A frozen Brain class; with a map, every dragon is given the true map (computed once per game)."""

    def __init__(self, name, cls, params, gm=None):
        self.name, self.cls, self.params, self.gm = name, cls, params or None, gm
        self.brains, self.oracle = {}, None

    def spawn(self, did, init):
        b = self.cls.from_init(init, self.params)
        if self.gm is not None:
            if self.oracle is None:
                b.oracle_init(self.gm)
                self.oracle = {k: getattr(b, k) for k in ORACLE_ATTRS}
            else:
                for k, v in self.oracle.items():
                    setattr(b, k, v)
        self.brains[did] = b

    def reply(self, did, block):
        return self.brains[did].act(block)


def resolve(name, spec=None):
    """-> (dir, params, oracle) for a candidate spec or an opponent name."""
    import tune
    if spec:
        return str(ROOT / spec["dir"]) if not pathlib.Path(spec["dir"]).is_absolute() else spec["dir"], \
            spec.get("params", {}), spec.get("oracle", False)
    if name in EXTRA:
        return EXTRA[name]
    kind, arg = tune.opponent(name)
    if kind != "frozen":
        raise SystemExit(f"{name}: only frozen snapshots can play here (it follows bot/brain.py otherwise)")
    return arg[0], arg[1], False


def play(job):
    import arena
    import frozen_brain
    import league
    import replay_parse
    from mapfile import GameMap
    label, mine, opp, (odir, oparams, oracle), spec, seed, side, no_budget = job
    mdir, mparams, moracle = mine
    data = league.map_bytes(spec)
    gm = GameMap.loads(data.decode()) if (oracle or moracle) else None
    if no_budget:
        mparams, oparams = {**mparams, "budget_ns": 10 ** 15}, {**oparams, "budget_ns": 10 ** 15}
    me = Player("me", frozen_brain.load(mdir)[0], mparams, gm if moracle else None)
    foe = Player("opp", frozen_brain.load(odir)[0], oparams, gm if oracle else None)
    res, deaths, errors = arena.play(Seeded(_engine, seed), data, *((me, foe) if side == "A" else (foe, me)))
    r = replay_parse.parse(_engine.replay("a", "b"))
    them = "B" if side == "A" else "A"
    return {"label": label, "opp": opp, "map": spec[1], "variant": spec[2], "seed": seed, "side": side,
            "score": 0.5 if res.winner is None else float(res.winner == side), "rounds": r["rounds"],
            "us": r["standing"][side], "them": r["standing"][them], "errors": len(errors),
            "error": errors[0][-400:] if errors else ""}


def wilson(p, n, z=1.96):
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    import ladder_maps
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates")
    ap.add_argument("--vs", default="mh3,farmer,portal_farmer")
    ap.add_argument("--seeds", type=int, default=10, help="seeds per map (each played from both seats)")
    ap.add_argument("--seed-base", type=int, default=1)
    ap.add_argument("--variants", type=int, default=0, help="also play ladder variants 1..N (0 = the real maps only)")
    ap.add_argument("--maps", default="", help="comma-separated ladder map names (default: the whole pool)")
    ap.add_argument("--workers", type=int, default=14)
    # budget_ns is wall-clock time here but CPU points on the judge, where the bot spends at most ~12.6M of its 60M,
    # so the budget almost never binds there. Locally, a loaded machine trips it at random, so it is off by default:
    # closer to the ladder, and every (map, seed) game exactly repeatable.
    ap.add_argument("--wall-budget", action="store_true", help="keep each bot's budget_ns as wall-clock time")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    cands = json.loads(pathlib.Path(args.candidates).read_text())
    mine = {lab: resolve(lab, spec) for lab, spec in cands.items()}
    opps = {o: resolve(o) for o in args.vs.split(",") if o}
    pool = args.maps.split(",") if args.maps else ladder_maps.POOL
    if set(pool) - set(ladder_maps.POOL):
        raise SystemExit(f"not ladder maps: {sorted(set(pool) - set(ladder_maps.POOL))}")
    maps = [("ladder", n, v) for n in pool for v in range(args.variants + 1)]
    seeds = range(args.seed_base, args.seed_base + args.seeds)
    jobs = [(lab, mine[lab], o, opps[o], m, s, side, not args.wall_budget)
            for lab in cands for o in opps for m in maps for s in seeds for side in "AB"]
    # a fresh pool every workers x MAX_TASKS games: the 1.x engine leaks ~21 MB a game in a long-lived worker, and a
    # 4,000-game run died of it (MemoryError, 30 Sep) -- the same fix as league.League.run
    import league
    res, step = [], args.workers * league.MAX_TASKS
    for i in range(0, len(jobs), step):
        with concurrent.futures.ProcessPoolExecutor(args.workers, initializer=_init) as ex:
            res += list(ex.map(play, jobs[i:i + step], chunksize=2))
    print(f"{len(res)} games: {len(maps)} maps x {args.seeds} seeds (from {args.seed_base}) x 2 seats per pairing\n")
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
        by_map[(r["label"], r["opp"], r["map"])].append(r["score"])
    names = sorted({r["map"] for r in res})
    print("\nscore by map:")
    print(f"{'candidate':14s} {'vs':14s} " + " ".join(f"{n[:9]:>9s}" for n in names))
    for lab in cands:
        for o in opps:
            print(f"{lab:14s} {o:14s} " + " ".join(
                f"{sum(by_map[(lab, o, n)]) / len(by_map[(lab, o, n)]):9.0%}" for n in names))
    errs = [r for r in res if r["errors"]]
    if errs:
        print(f"\n{len(errs)} games had bot errors; first:\n{errs[0]['error']}")
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(res))


if __name__ == "__main__":
    main()
