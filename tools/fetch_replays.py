"""Downloads a team's ladder games (replays + a CSV index) from game.battlecode.au, no login needed.

Games are public: the site's /games page lists every match (paged 10 at a time, filterable by team through its
SvelteKit data feed, /games/__data.json?teams=ID&page=N), and /api/matches/ID/replay returns the replay gzipped.
Saved replays are un-gzipped, so tools/replay_maps.py and the VS Code viewer read them as-is.

    .venv\\Scripts\\python.exe tools\\fetch_replays.py --team "Our Team" [--since 2026-09-25T08:23:32Z] [--limit 200]
        [--maps Portals,Devil] [--losses-only] [--out replays/ladder]

The index (index.csv in --out) has one row per game: match id, time, map, our seat, opponent, opponent rating,
result. Existing replay files are skipped, so a rerun only fetches new games.
"""
import argparse
import csv
import gzip
import json
import pathlib
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
SITE = "https://game.battlecode.au"
PAUSE = 0.5  # seconds between requests: be polite to a shared competition server


def fetch(path):
    req = urllib.request.Request(SITE + path, headers={"User-Agent": "unswbc-replay-fetch"})
    with urllib.request.urlopen(req, timeout=300) as reply:
        return reply.read()


def devalue(flat):
    """Decodes SvelteKit's flattened `devalue` data: every dict value / list item is an index into `flat`."""
    memo = {}

    def at(i):
        if i in memo:
            return memo[i]
        v = flat[i]
        if isinstance(v, dict):
            out = memo[i] = {}
            for k, j in v.items():
                out[k] = at(j) if isinstance(j, int) and j >= 0 else None
            return out
        if isinstance(v, list):
            if v and isinstance(v[0], str) and v[0] in ("Date", "Set", "Map", "BigInt", "RegExp"):
                return v[1] if len(v) > 1 else None
            out = memo[i] = []
            out.extend(at(j) for j in v)
            return out
        return v

    return at(0)


def games_page(team_id, page):
    data = json.loads(fetch(f"/games/__data.json?teams={team_id}&page={page}"))
    return devalue(data["nodes"][1]["data"])


def find_team(name_or_id):
    teams = games_page("", 1)["teams"]
    if str(name_or_id).isdigit():
        return int(name_or_id), next((t["name"] for t in teams if t["id"] == int(name_or_id)), "?")
    exact = [t for t in teams if t["name"] == name_or_id]
    loose = [t for t in teams if t["name"].strip().lower() == name_or_id.strip().lower()]
    hits = exact or loose
    if len(hits) != 1:
        close = [t["name"] for t in teams if name_or_id.strip().lower() in t["name"].lower()][:10]
        raise SystemExit(f"no unique team named {name_or_id!r}; similar: {close}")
    return hits[0]["id"], hits[0]["name"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True, help="team name (as on the leaderboard) or numeric id")
    ap.add_argument("--since", default="", help="ISO time; older games are skipped (e.g. a submission's upload time)")
    ap.add_argument("--limit", type=int, default=200, help="at most this many games (newest first)")
    ap.add_argument("--maps", default="", help="comma list of map names to keep (default: all)")
    ap.add_argument("--losses-only", action="store_true")
    ap.add_argument("--ranked-only", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "replays" / "ladder"))
    args = ap.parse_args()

    team_id, team_name = find_team(args.team)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    keep_maps = {m.strip() for m in args.maps.split(",") if m.strip()}
    print(f"team {team_name!r} (id {team_id}) -> {out}")

    rows, page = [], 1
    while len(rows) < args.limit:
        d = games_page(team_id, page)
        time.sleep(PAUSE)
        if not d["matches"]:
            break
        stop = False
        for m in d["matches"]:
            if args.since and m["at"] < args.since:
                stop = True  # the feed is newest first
                break
            if m["status"] != "completed" or not m["hasReplay"]:
                continue
            seat = "a" if m["a"]["id"] == team_id else "b"
            opp = m["b" if seat == "a" else "a"]
            result = "draw" if m["winner"] is None else ("win" if m["winner"] == seat else "loss")
            if (keep_maps and m["mapName"] not in keep_maps) or (args.losses_only and result != "loss") \
                    or (args.ranked_only and not m["ranked"]):
                continue
            rows.append({"match": m["id"], "at": m["at"], "map": m["mapName"], "seat": seat.upper(),
                         "opponent": opp["name"], "opponent_elo": opp["elo"], "result": result,
                         "ranked": m["ranked"]})
            if len(rows) >= args.limit:
                break
        if stop or page * d["perPage"] >= d["total"]:
            break
        page += 1

    got = 0
    for r in rows:
        dest = out / f"M{r['match']}.replay"
        if not dest.exists():
            body = fetch(f"/api/matches/{r['match']}/replay")
            dest.write_bytes(gzip.decompress(body) if body[:2] == b"\x1f\x8b" else body)
            got += 1
            time.sleep(PAUSE)
    # merge into the existing index (a fresh row replaces an old one with the same match id), so a narrow --since
    # or --maps fetch adds to the history instead of wiping it
    index = out / "index.csv"
    merged = {}
    if index.exists():
        with index.open(newline="", encoding="utf-8") as f:
            merged = {r["match"]: r for r in csv.DictReader(f)}
    merged.update({str(r["match"]): r for r in rows})
    fields = ["match", "at", "map", "seat", "opponent", "opponent_elo", "result", "ranked"]
    with index.open("w", newline="", encoding="utf-8") as f:  # team names include non-Latin scripts
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(sorted(merged.values(), key=lambda r: r["at"], reverse=True))
    wins = sum(r["result"] == "win" for r in rows)
    print(f"{len(rows)} games ({wins} wins), {got} replays downloaded, index ({len(merged)} games) -> {index}")


if __name__ == "__main__":
    main()
