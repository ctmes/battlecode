"""Where each team's pearls come from, read off a replay (a ladder .replay file or a local engine.replay() blob).

Per team: dragon-turns, pearls eaten, split by source -- spawned on a fountain tile, spawned elsewhere, dropped by a
dead teammate (own corpse), dropped by a dead enemy -- plus deaths, pearls dropped by its own deaths, and splits.
A fountain is a tile whose pearl gap is (1, 1)-ish (spawn rate >= 0.5/round): on six of the ten ladder maps a
handful of them are nearly the whole pearl supply, and ladder opponents eat 3-15x as many of them as we do.

Event order inside a dragon's turn is turnStart(id) -> dragonAction -> pearl off (if eaten) -> dragonUpdate, so a
pearl-off event is credited to the dragon whose turn it is (checked exactly against the turn blocks). Spawns are the
pearl-on events between roundStart and the round's first turn; a pearl-on right after a death event is that
dragon's drop.

Replays since 2026-09-28 ~06:44 UTC have their TILE gaps zeroed, so pass the map text (maps/ladder/<name>.map) for
fountain rates; without it the replay's own TILE lines are used (fine for local games).

    .venv\\Scripts\\python.exe tools\\forage.py [--since 2026-09-28T11:03] [--until ...]   # ladder, us vs them
"""
import argparse
import collections
import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import replay_parse as rp  # noqa: E402
from replay_maps import unpack  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
FOUNTAIN_RATE = 0.5
# ladder MAP_NAME -> maps/ladder file stem
LADDER_NAMES = {"Autarky": "autarky", "Default": "default", "Devil": "devil", "Portals": "portals",
                "Prisoners Dilemma": "prisoners_dilemma", "Queen Of Spades": "queen_of_spades",
                "Schooltime": "schooltime", "Slithery Fight": "slithery_fight", "Trauma": "trauma", "Trophy": "trophy"}
FOUNTAIN_MAPS = ("autarky", "devil", "portals", "prisoners_dilemma", "slithery_fight", "trauma")
KEYS = ("turns", "eaten", "fountain", "spawn", "own_corpse", "enemy_corpse", "deaths", "dropped", "splits")


def tile_rates(text):
    """(x, y) -> expected spawn attempts per round, from a map text's "TILE x y min_gap max_gap" lines."""
    rates = {}
    for line in text.split("\n"):
        if line.startswith("TILE "):
            f = line.split()
            a, b = int(f[3]), int(f[4])
            rates[(int(f[1]), int(f[2]))] = 2.0 / (a + b) if b > 0 else 0.0
    return rates


def analyse(data, map_text=None):
    """-> {"map": name, "rounds": n, "A": Counter, "B": Counter} with the KEYS above per team."""
    msg = rp.Message(unpack(data))
    root = msg.struct(0, 0)
    s, pw = root[0], root[3]
    text = msg.text(s, pw)
    rates = tile_rates(map_text or text)
    team_of = {i: "AB"[int(x.split()[1])] for i, x in enumerate(y for y in text.split("\n") if y.startswith("DRAGON "))}
    alive = set(team_of)
    st = {"A": collections.Counter(), "B": collections.Counter()}
    source = {}  # pearl position -> "spawn" or the team letter of the corpse it came from
    rnd, in_turn, cur, corpse = 0, False, None, None
    for ev in msg.list_structs(s, pw + 3):
        k = msg.u16(ev, 0)
        if k != 3 and k != 11:
            corpse = None
        if k == 0:
            rnd = msg.i32(msg.ptr(ev, 0), 0)
            in_turn = False
        elif k == 1:
            in_turn = True
            cur = msg.i32(msg.ptr(ev, 0), 0)
            if cur in alive:
                st[team_of[cur]]["turns"] += 1
            else:
                cur = None
        elif k == 3:
            e = msg.ptr(ev, 0)
            q = msg.ptr(e, 0)
            pos = (msg.i32(q, 0), msg.i32(q, 4))
            if msg.i32(e, 0) == 1:
                src = "spawn" if not in_turn else corpse
                source[pos] = src
                if src in st:
                    st[src]["dropped"] += 1
            else:
                src = source.pop(pos, None)
                if cur is None:
                    continue
                t = team_of[cur]
                c = st[t]
                c["eaten"] += 1
                if src == "spawn":
                    c["fountain" if rates.get(pos, 0.0) >= FOUNTAIN_RATE else "spawn"] += 1
                elif src == t:
                    c["own_corpse"] += 1
                elif src in st:
                    c["enemy_corpse"] += 1
        elif k == 10:
            e = msg.ptr(ev, 0)
            ch = msg.i32(e, 4)
            team_of[ch] = "AB"[msg.u16(e, 8)]
            alive.add(ch)
            st[team_of[ch]]["splits"] += 1
        elif k == 11:
            dead = msg.i32(msg.ptr(ev, 0), 0)
            corpse = team_of.get(dead)
            if dead in alive:
                alive.discard(dead)
                st[corpse]["deaths"] += 1
    name = next((x[9:].strip() for x in text.split("\n") if x.startswith("MAP_NAME ")), "?")
    return {"map": name, "rounds": rnd, "A": st["A"], "B": st["B"]}


def line(c, games):
    """One summary line for a team's pooled Counter over `games` games."""
    t = c["turns"] or 1
    return (f"eat/100 turns {100 * c['eaten'] / t:4.1f}  per game: fountain {c['fountain'] / games:5.0f}  other spawn "
            f"{c['spawn'] / games:4.0f}  own corpses {c['own_corpse'] / games:4.0f}  enemy corpses "
            f"{c['enemy_corpse'] / games:3.0f}  deaths/1000 turns {1000 * c['deaths'] / t:4.1f}  splits {c['splits'] / games:4.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="")
    ap.add_argument("--until", default="9999")
    ap.add_argument("--replays", default=str(ROOT / "replays" / "ladder"))
    args = ap.parse_args()
    lad = pathlib.Path(args.replays)
    rows = [r for r in csv.DictReader(open(lad / "index.csv", encoding="utf-8")) if args.since <= r["at"] < args.until]
    groups = collections.defaultdict(lambda: {"us": collections.Counter(), "them": collections.Counter(), "n": 0})
    for r in rows:
        path = lad / f"M{r['match']}.replay"
        stem = LADDER_NAMES.get(r["map"])
        if not path.exists() or stem is None:
            continue
        g = analyse(path.read_bytes(), (ROOT / "maps" / "ladder" / f"{stem}.map").read_text())
        us, them = r["seat"], "B" if r["seat"] == "A" else "A"
        for grp in ("all", "fountain maps" if stem in FOUNTAIN_MAPS else "other maps"):
            groups[grp]["us"].update(g[us])
            groups[grp]["them"].update(g[them])
            groups[grp]["n"] += 1
    for grp in ("all", "fountain maps", "other maps"):
        gg = groups[grp]
        if gg["n"]:
            print(f"{grp} ({gg['n']} games)")
            for side in ("us", "them"):
                print(f"  {side:4s} {line(gg[side], gg['n'])}")


if __name__ == "__main__":
    main()
