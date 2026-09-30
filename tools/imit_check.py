"""Plays a frozen bot (an imitation, say) against an opponent on the ladder maps and saves the replays as a
fetch_replays-style folder (index.csv seat = the first bot), so tools/scout.py can profile it next to the real team:

    .venv-1x\\Scripts\\python.exe tools\\imit_check.py snapshots\\imit_cheji --vs mh7 --seeds 3 --out replays\\imitcheck\\cheji
    .venv\\Scripts\\python.exe tools\\scout.py imitation=replays\\imitcheck\\cheji real=replays\\imit\\cheji_1001
"""
import argparse
import concurrent.futures
import csv
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bot"))
sys.path.insert(0, str(ROOT / "tools"))
import bench1x  # noqa: E402

NAMES = {"autarky": "Autarky", "default": "Default", "devil": "Devil", "portals": "Portals",
         "prisoners_dilemma": "Prisoners Dilemma", "queen_of_spades": "Queen Of Spades", "schooltime": "Schooltime",
         "slithery_fight": "Slithery Fight", "trauma": "Trauma", "trophy": "Trophy"}


def job(args):
    n, bot_dir, params, opp, mapname, seed, side, out = args
    import arena
    import frozen_brain
    import league
    odir, oparams, _ = bench1x.resolve(opp)
    me = bench1x.Player("me", frozen_brain.load(bot_dir)[0], params)
    foe = bench1x.Player("opp", frozen_brain.load(odir)[0], {**oparams, "budget_ns": 10 ** 15})
    res, deaths, errors = arena.play(bench1x.Seeded(bench1x._engine, seed), league.map_bytes(("ladder", mapname, 0)),
                                     *((me, foe) if side == "A" else (foe, me)))
    (pathlib.Path(out) / f"M{n}.replay").write_bytes(bench1x._engine.replay("a", "b"))
    result = "draw" if res.winner is None else ("win" if res.winner == side else "loss")
    return {"match": n, "at": f"2026-10-01T00:00:{n % 60:02d}Z", "map": NAMES[mapname], "seat": side,
            "opponent": opp, "opponent_elo": 0, "result": result, "ranked": False, "errors": len(errors)}


def main():
    import ladder_maps
    ap = argparse.ArgumentParser()
    ap.add_argument("bot")
    ap.add_argument("--params", default="{}", help="JSON params for the bot")
    ap.add_argument("--vs", default="mh7")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed-base", type=int, default=5000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    import json
    out = ROOT / args.out if not pathlib.Path(args.out).is_absolute() else pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bot = str(ROOT / args.bot) if not pathlib.Path(args.bot).is_absolute() else args.bot
    jobs, n = [], 0
    for m in ladder_maps.POOL:
        for s in range(args.seed_base, args.seed_base + args.seeds):
            for side in "AB":
                n += 1
                jobs.append((n, bot, json.loads(args.params), args.vs, m, s, side, str(out)))
    with concurrent.futures.ProcessPoolExecutor(args.workers, initializer=bench1x._init) as ex:
        rows = list(ex.map(job, jobs))
    with open(out / "index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["match", "at", "map", "seat", "opponent", "opponent_elo", "result", "ranked"])
        w.writeheader()
        w.writerows({k: v for k, v in r.items() if k != "errors"} for r in rows)
    wins = sum(r["result"] == "win" for r in rows)
    print(f"{len(rows)} games vs {args.vs}: {wins} wins ({100 * wins / len(rows):.0f}%), "
          f"{sum(r['errors'] for r in rows)} bot errors -> {out}")


if __name__ == "__main__":
    main()
