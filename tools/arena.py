"""Fast local matches on the official WASM engine, run in-process (no bot subprocesses, no sandbox).

Run with the venv interpreter (it has wasmtime + unswbc):
    .venv\\Scripts\\python.exe tools\\arena.py [--opponent greedy|self] [--maps maps] [--reps 1] [--split N]

This measures behaviour (deaths, wins), not judge points: use `unswbc run --sandbox -v` for points.
"""
import argparse
import pathlib
import sys
import time
import traceback
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bot"))
import proto  # noqa: E402
from brain import Brain, MOVES  # noqa: E402
from unswbc.engine import EngineModule  # noqa: E402

DEATH = {"W": "wall", "S": "self", "O": "other body", "H": "head-on", "A": "no action"}
DEBUG_DEATHS = False


class BrainPlayer:
    def __init__(self, name, params=None):
        self.name, self.params, self.brains = name, params, {}

    def spawn(self, did, init):
        self.brains[did] = Brain.from_init(init, self.params)

    def reply(self, did, block):
        return self.brains[did].act(block)


class GreedyPlayer:
    """Sparring partner: steps onto an adjacent pearl if it is safe, else any non-fatal move (prefers heading)."""

    def __init__(self, name="greedy"):
        self.name, self.dims = name, {}

    def spawn(self, did, init):
        _, _, w, h, _ = proto.parse_init(init)
        self.dims[did] = (w, h)

    def reply(self, did, block):
        w, h = self.dims[did]
        t = proto.parse_turn(block)
        occ = {(int(q[2]), int(q[3])) for q in t.parts}
        vt = t.edges[11].split()
        kelp = (t.edges[3].split()[3] == b"w", vt[4] == b"w", t.edges[4].split()[3] == b"w", vt[3] == b"w")
        win = (17, 25, 31, 23)  # window tile of the N, E, S, W neighbour
        safe = []
        for d in range(4):
            if kelp[d] or ((t.hx + proto.DX[d]) % w, (t.hy + proto.DY[d]) % h) in occ:
                continue
            safe.append(d)
        for d in safe:
            if t.flags[win[d]] == b"1":
                return MOVES[d]
        if t.dir in safe:
            return MOVES[t.dir]
        return MOVES[safe[0]] if safe else MOVES[0]


def play(engine, map_bytes, pa, pb):
    owner, deaths, errors = {}, [], []

    def bot_spawn(did, init):
        team = b"A"
        for line in init.split(b"\n"):
            if line.startswith(b"TEAM"):
                team = line.split()[1]
                break
        owner[did] = pa if team == b"A" else pb
        owner[did].spawn(did, init)

    def bot_reply(did, block):
        try:
            return owner[did].reply(did, block)
        except Exception:  # noqa: BLE001
            errors.append(traceback.format_exc())
            return b""

    def on_death(did, rnd, reason):
        deaths.append((owner[did].name, did, rnd, reason))
        if owner[did].name == "me":
            br = owner[did].brains.get(did)
            dbg = getattr(br, "dbg", None)
            # a fatal move while a safe one existed would be a legality bug (boxed-in deaths are not)
            if dbg is not None and reason in "WSO" and 0 in dbg["status"]:
                errors.append(f"LEGALITY BUG: dragon {did} round {rnd} {DEATH[reason]} with a safe move available: {dbg}")
            if DEBUG_DEATHS:
                print(f"   death: dragon {did} round {rnd} reason {DEATH.get(reason, reason)}  {dbg}")

    res = engine.run(map_bytes, bot_reply, on_death, bot_spawn, lambda line: None, 0)
    return res, deaths, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opponent", default="greedy", choices=("greedy", "self"))
    ap.add_argument("--maps", default=str(ROOT / "maps"))
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--split", type=int, default=0, help="enable splitting at this length (0 = off)")
    ap.add_argument("--lenient", action="store_true", help="swallow brain exceptions like the shipped bot does")
    ap.add_argument("--deaths", action="store_true", help="print the last decision context of each of my deaths")
    ap.add_argument("--only", default="", help="substring filter on map names")
    ap.add_argument("--params", default="", help="comma list of brain parameter overrides, e.g. pessimistic=0,timed=0")
    ap.add_argument("--quiet", action="store_true", help="summary only")
    args = ap.parse_args()

    global DEBUG_DEATHS
    DEBUG_DEATHS = args.deaths
    Brain.debug = True
    Brain.strict = not args.lenient
    params = {"split_len": args.split} if args.split else {}
    for kv in filter(None, args.params.split(",")):
        k, v = kv.split("=")
        params[k] = float(v) if "." in v else int(v)
    params = params or None
    engine = EngineModule()
    maps = [m for m in sorted(pathlib.Path(args.maps).glob("*.map")) if args.only in m.stem]
    total = Counter()
    t_all = time.perf_counter()
    print(f"{'map':22s} {'size':>7s} {'side':>4s} {'result':>7s} {'rounds':>6s} {'len me/opp':>11s} {'dr me/opp':>9s}  my deaths")
    for mp in maps:
        data = mp.read_bytes()
        size = data.split(b"\n", 1)[0].decode().replace("MAP ", "").replace(" ", "x")
        for rep in range(args.reps):
            for me_side in ("A", "B"):
                me = BrainPlayer("me", params)
                other = BrainPlayer("opp", params) if args.opponent == "self" else GreedyPlayer()
                pa, pb = (me, other) if me_side == "A" else (other, me)
                res, deaths, errors = play(engine, data, pa, pb)
                win = res.winner
                outcome = "draw" if win is None else ("win" if win == me_side else "loss")
                my_len = res.a_length if me_side == "A" else res.b_length
                op_len = res.b_length if me_side == "A" else res.a_length
                my_dr = res.a_dragons if me_side == "A" else res.b_dragons
                op_dr = res.b_dragons if me_side == "A" else res.a_dragons
                mine = Counter(DEATH.get(r, r) for n, _, _, r in deaths if n == "me")
                total[outcome] += 1
                for k, v in mine.items():
                    total["died:" + k] += v
                total["errors"] += len(errors)
                if not args.quiet:
                    print(f"{mp.stem:22s} {size:>7s} {me_side:>4s} {outcome:>7s} {res.rounds + 1:>6d} "
                          f"{my_len:>5d}/{op_len:<5d} {my_dr:>4d}/{op_dr:<4d}  {dict(mine) or '-'}")
                total["my_len"] += my_len
                total["my_dragons_end"] += my_dr
                if errors:
                    print(errors[0])
    print(f"\n{sum(total[k] for k in ('win', 'draw', 'loss'))} games in {time.perf_counter() - t_all:.1f}s: "
          f"{dict(total)}")


if __name__ == "__main__":
    main()
