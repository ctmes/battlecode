"""How our ladder games were won and lost, from the replays tools/fetch_replays.py downloaded.

For every game in replays/ladder/index.csv: did we lose by elimination or at round 500 (and then on which
tiebreak: longest living dragon, or total length), how our dragons died, and how the final standings compared.
Parsed games are cached in replays/ladder/parsed.json, so a rerun only reads new replays.

    .venv\\Scripts\\python.exe tools\\ladder_report.py [--dir replays/ladder] [--since ISO] [--until ISO] [--maps A,B]
"""
import argparse
import collections
import concurrent.futures
import csv
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import replay_parse  # noqa: E402


def summarize(path):
    """The parts of a parsed replay the report needs, small enough to cache as JSON."""
    r = replay_parse.parse(path.read_bytes())
    deaths = collections.defaultdict(collections.Counter)
    for d in r["dragons"].values():
        if d["died"] is not None:
            deaths[d["team"]][d["reason"]] += 1
    born = collections.Counter(d["team"] for d in r["dragons"].values())
    return {"map": r["map"], "rounds": r["rounds"], "winner": r["winner"], "end_reason": r["end_reason"],
            "standing": r["standing"], "deaths": {t: dict(c) for t, c in deaths.items()}, "dragons": dict(born)}


def load(dirpath, rows):
    cache_path = dirpath / "parsed.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    todo = [r["match"] for r in rows if r["match"] not in cache and (dirpath / f"M{r['match']}.replay").exists()]
    if todo:
        with concurrent.futures.ProcessPoolExecutor(6) as ex:
            for m, s in zip(todo, ex.map(summarize, [dirpath / f"M{m}.replay" for m in todo])):
                cache[m] = s
        cache_path.write_text(json.dumps(cache))
    return cache


def loss_kind(g, us):
    them = "B" if us == "A" else "A"
    mine, theirs = g["standing"][us], g["standing"][them]
    if mine[0] == 0:
        return "eliminated"
    if theirs[0] == 0:
        return "we eliminated them"
    if mine[1] != theirs[1]:
        return "round 500: shorter longest dragon" if mine[1] < theirs[1] else "round 500: longer longest dragon"
    return "round 500: longest tied, less total length" if mine[2] < theirs[2] else \
        "round 500: longest tied, more total length"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "replays" / "ladder"))
    ap.add_argument("--since", default="")
    ap.add_argument("--until", default="")
    ap.add_argument("--maps", default="")
    args = ap.parse_args()
    dirpath = pathlib.Path(args.dir)
    rows = list(csv.DictReader((dirpath / "index.csv").open(encoding="utf-8")))
    keep = {m.strip() for m in args.maps.split(",") if m.strip()}
    rows = [r for r in rows if (not args.since or r["at"] >= args.since) and (not args.until or r["at"] < args.until)
            and (not keep or r["map"] in keep)]
    games = load(dirpath, rows)
    rows = [r for r in rows if r["match"] in games]
    print(f"{len(rows)} games, {rows[-1]['at'][:16] if rows else '-'} to {rows[0]['at'][:16] if rows else '-'} (UTC)\n")

    by_map = collections.defaultdict(list)
    for r in rows:
        by_map[r["map"]].append(r)
    print(f"{'map':18s} {'games':>5s} {'win%':>5s}   how we lost (count)")
    for name in sorted(by_map, key=lambda n: sum(x["result"] == "win" for x in by_map[n]) / len(by_map[n])):
        rs = by_map[name]
        wins = sum(x["result"] == "win" for x in rs)
        kinds = collections.Counter(loss_kind(games[x["match"]], x["seat"]) for x in rs if x["result"] == "loss")
        print(f"{name:18s} {len(rs):5d} {wins / len(rs):5.0%}   {dict(kinds.most_common())}")

    print("\nall losses by kind, and what the end standings looked like (ours vs theirs, averages):")
    kinds = collections.defaultdict(list)
    for r in rows:
        if r["result"] == "loss":
            kinds[loss_kind(games[r["match"]], r["seat"])].append(r)
    for kind, rs in sorted(kinds.items(), key=lambda kv: -len(kv[1])):
        def avg(side, i):
            vals = []
            for x in rs:
                g = games[x["match"]]
                team = x["seat"] if side == "us" else ("B" if x["seat"] == "A" else "A")
                vals.append(g["standing"][team][i])
            return sum(vals) / len(vals)
        print(f"  {kind:42s} {len(rs):3d}  dragons alive {avg('us', 0):5.1f} vs {avg('them', 0):5.1f}, "
              f"longest {avg('us', 1):5.1f} vs {avg('them', 1):5.1f}, total length {avg('us', 2):6.1f} vs "
              f"{avg('them', 2):6.1f}, ended round {sum(games[x['match']]['rounds'] for x in rs) / len(rs):5.0f}")

    print("\nhow our dragons died (per game), wins vs losses:")
    for res in ("win", "loss"):
        rs = [r for r in rows if r["result"] == res]
        c = collections.Counter()
        born = 0
        for r in rs:
            c.update(games[r["match"]]["deaths"].get(r["seat"], {}))
            born += games[r["match"]]["dragons"].get(r["seat"], 0)
        n = max(len(rs), 1)
        print(f"  {res:5s} ({len(rs)} games): dragons per game {born / n:5.1f}, deaths per game "
              f"{sum(c.values()) / n:5.1f}: " + ", ".join(f"{k} {v / n:.1f}" for k, v in c.most_common()))


if __name__ == "__main__":
    main()
