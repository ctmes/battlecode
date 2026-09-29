"""Scouting report: how a team plays, read off its ladder replays (fetched with tools/fetch_replays.py --team NAME).

Every dragon's body is rebuilt turn by turn from the action stream (moves, sprints, pearls eaten, splits) and checked
against each dragonUpdate's head and tail, so deaths can be pinned on whoever's body was hit. Per team it pools:

    results    win rate by map; games lost by elimination vs at round 500
    shape      dragon count, total and longest length at rounds 50..400 and at the end
    splitting  parent length before a split, child size, split rounds, how many times one dragon splits
    deaths     per 1000 dragon-turns by cause, and who they die into; enemy deaths they cause
    foraging   pearls per 100 dragon-turns by source (fountain / other spawn / own corpse / enemy corpse)
    movement   sprint rate and length, sonar use
    tiebreak   the final longest dragon: when it was born, at what length, from what parent

A directory is a fetch_replays --out folder (index.csv's seat column is that team's seat). Parsed games are cached
in <dir>/scout.json, so a rerun only reads new replays.

    .venv\\Scripts\\python.exe tools\\scout.py replays/top/cutlery replays/top/cheji replays/ladder [--since ISO]
    .venv\\Scripts\\python.exe tools\\scout.py mh3=replays/ladder@2026-09-28T11:03,2026-09-29T03:25 \\
        mh4=replays/ladder@2026-09-29T03:25     # two versions of one team, told apart by time
"""
import argparse
import collections
import concurrent.futures
import csv
import json
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import replay_parse as rp  # noqa: E402
from forage import FOUNTAIN_RATE, FOUNTAIN_MAPS, LADDER_NAMES, tile_rates  # noqa: E402
from mapfile import GameMap  # noqa: E402
from replay_maps import unpack  # noqa: E402

CHECK = (50, 100, 200, 300, 400)
REASONS = rp.REASONS
VERSION = 2  # bump to invalidate scout.json caches when extract() changes


def u16_list(msg, st, i):
    """The List(UInt16) behind pointer i of struct st (a move's directions)."""
    sg, _, _, pws, npc = st
    if i >= npc or rp._u64(msg.segs[sg], 8 * (pws + i))[0] == 0:
        return []
    s2, w2, p = msg._follow(sg, pws + i)
    return [rp._u16(msg.segs[s2], 8 * w2 + 2 * j)[0] for j in range(p >> 35)]


def positions(msg, seg, word):
    return [(msg.i32(q, 0), msg.i32(q, 4)) for q in msg.list_structs(seg, word)]


def new_team():
    return {"turns": 0, "moves": collections.Counter(), "sprint_eats": 0, "sonar": 0,
            "eat": collections.Counter(), "deaths": collections.Counter(), "death_len": [], "death_round": [],
            "died_into": collections.Counter(), "caused": collections.Counter(), "splits": [],
            "splits_per_dragon": [], "max_len": [], "shape": {}, "king": None, "dragons": 0}


def extract(data, map_text=None, hook=None):
    """One replay -> {"map", "rounds", "winner", "standing", "mismatch", "A": team stats, "B": team stats}.

    hook(kind, *args), if given, sees the game as it is tracked: ("init", state dict), ("eat", round, eater, pos,
    source), ("pearl_on", round, pos, source, id of the dragon whose death dropped it or None), ("split", round,
    parent, child), ("death", round, id, reason, killer or None, split size asked for or None), ("move", round, id,
    directions) before the move is applied, ("end", round)."""
    msg = rp.Message(unpack(data))
    root = msg.struct(0, 0)
    s, pw = root[0], root[3]
    text = msg.text(s, pw)
    gm = GameMap.loads(text)
    step = {}
    for x in range(gm.w):
        for y in range(gm.h):
            for d in range(4):
                step[x, y, d] = gm.step(x, y, d)
    rates = tile_rates(map_text or text)
    T = {"A": new_team(), "B": new_team()}
    team_of, bodies, occ, info = {}, {}, {}, {}
    for i, (team, body) in enumerate(gm.dragons):
        team_of[i] = "AB"[team]
        bodies[i] = list(body)
        info[i] = {"born": 0, "birth_len": len(body), "parent": None, "parent_len": None, "max": len(body),
                   "splits": 0, "len_at": {}}
        for c in body:
            occ[c] = i

    if hook:
        hook("init", {"bodies": bodies, "team_of": team_of, "info": info, "occ": occ, "step": step, "rates": rates,
                      "map": gm})

    def snapshot(label):
        for t in "AB":
            lens = [len(b) for i, b in bodies.items() if team_of[i] == t]
            T[t]["shape"][label] = (len(lens), sum(lens), max(lens, default=0))
        for i, b in bodies.items():
            info[i]["len_at"][label] = len(b)

    def remove(i):
        for c in bodies.pop(i):
            if occ.get(c) == i:
                del occ[c]

    source = {}  # pearl position -> "spawn" or the team letter of the corpse it came from
    rnd, in_turn, cur, corpse, dirs, eats, done, mismatch, updates = 0, False, None, None, None, 0, 0, 0, 0
    turn_no, last_dead, sonar_seen, split_req = 0, None, False, None
    for ev in msg.list_structs(s, pw + 3):
        k = msg.u16(ev, 0)
        if k != 3 and k != 11:
            corpse = None
        if k == 0:
            rnd = msg.i32(msg.ptr(ev, 0), 0)
            in_turn = False
            if rnd in CHECK:
                snapshot(rnd)
        elif k == 1:
            in_turn, cur, dirs, eats, done, split_req = True, msg.i32(msg.ptr(ev, 0), 0), None, 0, 0, None
            turn_no += 1
            if cur in bodies:
                T[team_of[cur]]["turns"] += 1
            sonar_seen = False
        elif k == 4:
            a = msg.ptr(msg.ptr(ev, 0), 0)
            if a is not None and msg.u16(a, 0) == 1:
                split_req = msg.i32(a, 4)
            if a is not None and msg.u16(a, 0) == 0:
                dirs = u16_list(msg, a, 0)
                if cur in bodies and dirs:
                    T[team_of[cur]]["moves"][len(dirs)] += 1
                    if hook:
                        hook("move", rnd, cur, dirs)
        elif k == 3:
            e = msg.ptr(ev, 0)
            q = msg.ptr(e, 0)
            pos = (msg.i32(q, 0), msg.i32(q, 4))
            if msg.i32(e, 0) == 1:
                src = "spawn" if not in_turn else corpse
                source[pos] = src
                if hook:
                    hook("pearl_on", rnd, pos, src, last_dead[1] if in_turn and last_dead and last_dead[0] == turn_no
                         else None)
            else:
                src = source.pop(pos, None)
                if hook and cur in bodies:
                    hook("eat", rnd, cur, pos, src)
                if cur in bodies:
                    t = team_of[cur]
                    eats += 1
                    if src == "spawn":
                        T[t]["eat"]["fountain" if rates.get(pos, 0.0) >= FOUNTAIN_RATE else "spawn"] += 1
                    elif src == t:
                        T[t]["eat"]["own_corpse"] += 1
                    elif src in T:
                        T[t]["eat"]["enemy_corpse"] += 1
                    else:
                        T[t]["eat"]["unknown"] += 1
        elif k == 9:
            e = msg.ptr(ev, 0)
            i = msg.i32(e, 0)
            h, tl = msg.ptr(e, 0), msg.ptr(e, 1)
            head, tail = (msg.i32(h, 0), msg.i32(h, 4)), (msg.i32(tl, 0), msg.i32(tl, 4))
            if i not in bodies or not dirs or i != cur:
                continue
            # one update per step of a sprint; the sprint's cost (a segment per extra step) comes off at the last
            old = bodies[i]
            new = [head] + (old if eats else old[:-1])
            if len(dirs) > 1 and eats:
                T[team_of[i]]["sprint_eats"] += eats
            eats = 0
            done += 1
            if done == len(dirs) and len(dirs) > 1:
                new = new[:len(new) - (len(dirs) - 1)]
            updates += 1
            if len(new) < 2 or new[-1] != tail:
                mismatch += 1
            for c in old:
                if occ.get(c) == i:
                    del occ[c]
            for c in new:
                occ[c] = i
            bodies[i] = new
            info[i]["max"] = max(info[i]["max"], len(new))
        elif k == 10:
            e = msg.ptr(ev, 0)
            p, ch, t = msg.i32(e, 0), msg.i32(e, 4), "AB"[msg.u16(e, 8)]
            pb, cb = positions(msg, e[0], e[3]), positions(msg, e[0], e[3] + 1)
            alive = sum(1 for j in bodies if team_of[j] == t)
            T[t]["splits"].append((rnd, len(pb) + len(cb), len(cb), alive))
            team_of[ch] = t
            if p in bodies:
                remove(p)
            bodies[p], bodies[ch] = pb, cb
            for c in pb:
                occ[c] = p
            for c in cb:
                occ[c] = ch
            info[p]["splits"] += 1
            info[ch] = {"born": rnd, "birth_len": len(cb), "parent": p, "parent_len": len(pb) + len(cb),
                        "max": len(cb), "splits": 0, "len_at": {}}
            if hook:
                hook("split", rnd, p, ch)
        elif k == 11:
            e = msg.ptr(ev, 0)
            dead, code = msg.i32(e, 0), msg.u16(e, 4)
            reason = REASONS[code] if code < len(REASONS) else "?"
            corpse = team_of.get(dead)
            if dead not in bodies:
                continue
            t = team_of[dead]
            T[t]["deaths"][reason] += 1
            T[t]["death_len"].append(len(bodies[dead]))
            T[t]["death_round"].append(rnd)
            killer = None
            if dead != cur:
                killer = cur  # the other half of a head-on
            elif reason == "head-on" and last_dead is not None and last_dead[0] == turn_no:
                killer = last_dead[1]  # the mover's own death comes after its victim's, whose body is gone by now
            elif reason in ("other body", "head-on") and dirs and done < len(dirs):
                nxt = step.get((*bodies[dead][0], dirs[done]))  # the fatal step is the one after the last update
                if occ.get(nxt, dead) != dead:
                    killer = occ[nxt]
            if killer is not None and killer in team_of:
                side = "team" if team_of[killer] == t else "enemy"
                T[t]["died_into"][f"{reason}/{side}"] += 1
                if side == "enemy":
                    T[team_of[killer]]["caused"][reason] += 1
            elif reason in ("other body", "head-on"):
                T[t]["died_into"][f"{reason}/?"] += 1
            last_dead = (turn_no, dead)
            if hook:
                hook("death", rnd, dead, reason, killer, split_req if dead == cur else None)
            remove(dead)
        elif k == 12:
            sender = msg.i32(msg.ptr(ev, 0), 0)
            if sender == cur and cur in bodies and not sonar_seen:
                sonar_seen = True
                T[team_of[cur]]["sonar"] += 1
    snapshot("end")
    if hook:
        hook("end", rnd)
    for t in "AB":
        mine = [i for i in info if team_of[i] == t]
        T[t]["dragons"] = len(mine)
        T[t]["splits_per_dragon"] = [info[i]["splits"] for i in mine]
        T[t]["max_len"] = [info[i]["max"] for i in mine]
        living = [i for i in bodies if team_of[i] == t]
        if living:
            king = max(living, key=lambda i: len(bodies[i]))
            ki = info[king]
            # a split that leaves the parent 2 long hands the body to the child (the old tail, facing the other
            # way): a turnaround, not a new dragon. Walk back through those to where the king's body really began
            chain = [king]
            while info[chain[-1]]["parent"] is not None and \
                    info[chain[-1]]["birth_len"] > info[chain[-1]]["parent_len"] / 2:
                chain.append(info[chain[-1]]["parent"])
            line, flips = chain[-1], len(chain) - 1
            li = info[line]
            line_len_at = {}
            for c in CHECK:
                holder = next((d for d in chain if info[d]["born"] < c or d == line), None)
                if c in info[holder]["len_at"]:
                    line_len_at[c] = info[holder]["len_at"][c]
            T[t]["king"] = {"len": len(bodies[king]), "born": ki["born"], "birth_len": ki["birth_len"],
                            "parent_len": ki["parent_len"], "splits": ki["splits"], "len_at": ki["len_at"],
                            "flips": flips, "line_born": li["born"], "line_birth_len": li["birth_len"],
                            "line_parent_len": li["parent_len"], "line_len_at": line_len_at,
                            "founder": li["parent"] is None}
    r = rp.parse(data)
    return {"map": r["map"], "rounds": r["rounds"], "winner": r["winner"], "end_reason": r["end_reason"],
            "standing": r["standing"], "mismatch": mismatch, "updates": updates, "A": T["A"], "B": T["B"]}


def work(args):
    path, stem = args
    mt = (ROOT / "maps" / "ladder" / f"{stem}.map").read_text() if stem else None
    return path.stem, {"v": VERSION, **extract(path.read_bytes(), mt)}


def load(folder, since="", until="9999", workers=14):
    """-> [(index row, extracted game)] for every game in folder/index.csv with a replay on disk."""
    folder = pathlib.Path(folder)
    rows = [r for r in csv.DictReader(open(folder / "index.csv", encoding="utf-8")) if since <= r["at"] < until]
    cache_path = folder / "scout.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    todo = []
    for r in rows:
        p = folder / f"M{r['match']}.replay"
        if p.exists() and cache.get(p.stem, {}).get("v") != VERSION:
            todo.append((p, LADDER_NAMES.get(r["map"])))
    if todo:
        with concurrent.futures.ProcessPoolExecutor(workers) as ex:
            for name, g in ex.map(work, todo, chunksize=2):
                cache[name] = g
        cache_path.write_text(json.dumps(cache))
    return [(r, cache[f"M{r['match']}"]) for r in rows if f"M{r['match']}" in cache]


# ---------------------------------------------------------------------------------------------------------- report
def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def pct(a, b):
    return 100.0 * a / b if b else float("nan")


def summarize(games, who):
    """Pooled numbers for one side ("us" = the folder's team, "them" = its opponents) over [(row, game)]."""
    out = collections.OrderedDict()
    n = len(games)
    sides = [(r["seat"] if who == "us" else ("B" if r["seat"] == "A" else "A"), r, g) for r, g in games]
    wins = sum(g["winner"] == t for t, r, g in sides)
    losses = sum(g["winner"] not in (None, t) for t, r, g in sides)
    out["games"] = f"{n}"
    out["win %"] = f"{pct(wins, n):.0f}"
    out["  win % fountain maps"] = f"{pct(*_wl(sides, True)):.0f}"
    out["  win % other maps"] = f"{pct(*_wl(sides, False)):.0f}"
    elim_loss = sum(g["standing"][t] is not None and g["standing"][t][0] == 0 for t, r, g in sides)
    elim_win = sum(g["standing"]["B" if t == "A" else "A"][0] == 0 for t, r, g in sides
                   if g["standing"]["B" if t == "A" else "A"] is not None)
    out["games ended by eliminating them %"] = f"{pct(elim_win, n):.0f}"
    out["games lost by being eliminated %"] = f"{pct(elim_loss, n):.0f}"
    # shape
    for c in CHECK + ("end",):
        snap = [g[t]["shape"].get(str(c)) or g[t]["shape"].get(c) for t, r, g in sides]
        snap = [x for x in snap if x]
        out[f"round {c}: dragons / total / longest"] = (f"{mean(x[0] for x in snap):5.1f} / {mean(x[1] for x in snap):5.0f}"
                                                         f" / {mean(x[2] for x in snap):4.1f}")
    # splitting: a split that leaves the parent 2 long is a turnaround (the child is the old tail, facing back)
    splits = [sp for t, r, g in sides for sp in g[t]["splits"]]
    flips = [sp for sp in splits if sp[1] >= 5 and sp[2] > sp[1] / 2]
    real = [sp for sp in splits if sp not in flips]
    out["splits per game (turnarounds among them)"] = f"{len(splits) / n:.0f} ({len(flips) / n:.0f})"
    if flips:
        out["  turnaround: length before, median"] = f"{statistics.median(sp[1] for sp in flips):.0f}"
        out["  turnaround: parent keeps 2 %"] = f"{pct(sum(sp[1] - sp[2] == 2 for sp in flips), len(flips)):.0f}"
    if real:
        before = collections.Counter(min(sp[1], 12) for sp in real)
        child = collections.Counter(min(sp[2], 8) for sp in real)
        out["other splits: parent length before (share)"] = " ".join(
            f"{k}{'+' if k == 12 else ''}:{pct(v, len(real)):.0f}" for k, v in sorted(before.items())
            if pct(v, len(real)) >= 2)
        out["other splits: child length (share)"] = " ".join(
            f"{k}{'+' if k == 8 else ''}:{pct(v, len(real)):.0f}" for k, v in sorted(child.items())
            if pct(v, len(real)) >= 2)
        rounds = [sp[0] for sp in real]
        out["other splits by round 0-49/50-149/150-299/300-399/400+ %"] = " / ".join(
            f"{pct(sum(lo <= x < hi for x in rounds), len(rounds)):.0f}"
            for lo, hi in ((0, 50), (50, 150), (150, 300), (300, 400), (400, 999)))
        last = [max((sp[0] for sp in g[t]["splits"] if not (sp[1] >= 5 and sp[2] > sp[1] / 2)), default=None)
                for t, r, g in sides]
        last = [x for x in last if x is not None]
        out["last non-turnaround split round, median"] = f"{statistics.median(last):.0f}" if last else "-"
        out["team size when splitting, median"] = f"{statistics.median(sp[3] for sp in real):.0f}"
    spd = [x for t, r, g in sides for x in g[t]["splits_per_dragon"]]
    out["dragons that ever split %"] = f"{pct(sum(x > 0 for x in spd), len(spd)):.0f}"
    out["splits by a dragon that splits, mean"] = f"{mean(x for x in spd if x > 0):.1f}"
    ml = [x for t, r, g in sides for x in g[t]["max_len"]]
    out["dragons ever reaching length 8 / 16 %"] = f"{pct(sum(x >= 8 for x in ml), len(ml)):.0f} / {pct(sum(x >= 16 for x in ml), len(ml)):.1f}"
    # deaths
    turns = sum(g[t]["turns"] for t, r, g in sides)
    deaths = collections.Counter()
    into = collections.Counter()
    caused = collections.Counter()
    for t, r, g in sides:
        deaths.update(g[t]["deaths"])
        into.update(g[t]["died_into"])
        caused.update(g[t]["caused"])
    out["dragon-turns per game"] = f"{turns / n:.0f}"
    out["deaths / 1000 turns"] = f"{1000 * sum(deaths.values()) / turns:.1f}"
    out["  by cause"] = " ".join(f"{k.replace('other body', 'body').replace('no action', 'noact')}:{1000 * deaths[k] / turns:.1f}"
                                 for k in REASONS if deaths[k])
    out["  body deaths into enemy / own team"] = (f"{1000 * into['other body/enemy'] / turns:.1f} / "
                                                  f"{1000 * into['other body/team'] / turns:.1f}")
    out["  head-ons vs enemy / own team"] = (f"{1000 * into['head-on/enemy'] / turns:.1f} / "
                                             f"{1000 * into['head-on/team'] / turns:.1f}")
    dl = [x for t, r, g in sides for x in g[t]["death_len"]]
    out["length at death, median / share <= 3"] = f"{statistics.median(dl) if dl else 0:.0f} / {pct(sum(x <= 3 for x in dl), len(dl)):.0f}%"
    out["enemy deaths caused per game (body / head-on)"] = f"{caused['other body'] / n:.1f} / {caused['head-on'] / n:.1f}"
    # foraging
    eat = collections.Counter()
    for t, r, g in sides:
        eat.update(g[t]["eat"])
    out["pearls / 100 turns"] = f"{100 * sum(eat.values()) / turns:.1f}"
    out["  fountain / spawn / own corpse / enemy corpse"] = " / ".join(
        f"{100 * eat[k] / turns:.1f}" for k in ("fountain", "spawn", "own_corpse", "enemy_corpse"))
    # movement
    moves = collections.Counter()
    sprint_eats = sonar = 0
    for t, r, g in sides:
        moves.update({int(k): v for k, v in g[t]["moves"].items()})
        sprint_eats += g[t]["sprint_eats"]
        sonar += g[t]["sonar"]
    nm = sum(moves.values())
    sprints = sum(v for k, v in moves.items() if k > 1)
    out["sprint turns %  (2 / 3 / 4+ steps)"] = (f"{pct(sprints, nm):.2f}  ({moves[2]} / {moves[3]} / "
                                                 f"{sum(v for k, v in moves.items() if k > 3)})")
    out["pearls eaten per sprint"] = f"{sprint_eats / sprints:.2f}" if sprints else "-"
    out["sonar sends per turn"] = f"{sonar / turns:.2f}"
    # tiebreak: the final longest dragon, traced back through turnarounds to where its body began
    kings = [g[t]["king"] for t, r, g in sides if g[t]["king"] and g["rounds"] >= 499]
    if kings:
        out["final longest (games to round 500)"] = f"{mean(k['len'] for k in kings):.1f}"
        out["  turnarounds in its line, mean"] = f"{mean(k['flips'] for k in kings):.1f}"
        out["  line began: round, median / founder %"] = (f"{statistics.median(k['line_born'] for k in kings):.0f} / "
                                                         f"{pct(sum(k['founder'] for k in kings), len(kings)):.0f}")
        out["  line began: length / parent length, median"] = (
            f"{statistics.median(k['line_birth_len'] for k in kings):.0f} / "
            f"{statistics.median([k['line_parent_len'] for k in kings if k['line_parent_len']] or [0]):.0f}")
        out["  line length at 100 / 200 / 300 / 400, mean"] = " / ".join(
            f"{mean(v for v in (k['line_len_at'].get(str(c), k['line_len_at'].get(c)) for k in kings) if v):.1f}"
            for c in (100, 200, 300, 400))
    return out


def _wl(sides, fountain):
    sub = [(t, g) for t, r, g in sides if (LADDER_NAMES.get(g["map"]) in FOUNTAIN_MAPS) == fountain]
    return sum(g["winner"] == t for t, g in sub), len(sub)


def by_map(games):
    rows = collections.defaultdict(lambda: [0, 0])
    for r, g in games:
        rows[g["map"]][0] += g["winner"] == r["seat"]
        rows[g["map"]][1] += 1
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="fetch_replays --out folders, each as [label=]dir[@since[,until]] (ISO "
                                            "times); the label defaults to the folder name")
    ap.add_argument("--since", default="", help="ISO time; applies to every folder unless given as dir@ISO")
    ap.add_argument("--opponents", action="store_true", help="also print each folder's opponents, pooled")
    args = ap.parse_args()
    cols = {}
    maps = {}
    for spec in args.dirs:
        label, _, rest = spec.rpartition("=")
        folder, _, window = rest.partition("@")
        since, _, until = window.partition(",")
        games = load(folder, since or args.since, until or "9999")
        label = label or pathlib.Path(folder).name
        bad = sum(g["mismatch"] for r, g in games)
        upd = sum(g["updates"] for r, g in games)
        print(f"{label}: {len(games)} games, body tracking mismatches {bad}/{upd}", file=sys.stderr)
        cols[label] = summarize(games, "us")
        if args.opponents:
            cols[label + " opp"] = summarize(games, "them")
        maps[label] = by_map(games)
    keys = list(next(iter(cols.values())).keys())
    for c in cols.values():
        keys += [k for k in c if k not in keys]
    w0 = max(len(k) for k in keys)
    widths = {c: max(len(c), max(len(v) for v in cols[c].values())) for c in cols}
    print(" " * w0 + "  " + "  ".join(c.rjust(widths[c]) for c in cols))
    for k in keys:
        print(k.ljust(w0) + "  " + "  ".join(cols[c].get(k, "-").rjust(widths[c]) for c in cols))
    print()
    names = sorted({m for v in maps.values() for m in v})
    print("win % by map (games)".ljust(w0) + "  " + "  ".join(c.rjust(12) for c in maps))
    for m in names:
        cells = []
        for c in maps:
            w, n = maps[c].get(m, (0, 0))
            cells.append((f"{pct(w, n):.0f} ({n})" if n else "-").rjust(12))
        print(m.ljust(w0) + "  " + "  ".join(cells))


if __name__ == "__main__":
    main()
