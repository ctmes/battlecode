"""Fast local matches on the official WASM engine, run in-process (no bot subprocesses, no sandbox).

Run with the venv interpreter (it has wasmtime + unswbc):
    .venv\\Scripts\\python.exe tools\\arena.py [--opponent NAME|self|all] [--maps maps] [--reps 1] [--split N]

NAME is one of the sparring bots in tools/opponents.py (--list shows them), self plays the brain against itself,
and all runs every sparring bot in turn and ends with one summary table.
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
from brain import Brain  # noqa: E402
from opponents import OPPONENTS  # noqa: E402
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


def versus(kind, engine, maps, reps, params, opp_params, quiet):
    """Plays the brain against one kind of opponent on every map, on both sides, `reps` times; returns the tallies."""
    total = Counter()
    t_all = time.perf_counter()
    if not quiet:
        print(f"{'map':22s} {'size':>7s} {'side':>4s} {'result':>7s} {'rounds':>6s} {'len me/opp':>11s} {'dr me/opp':>9s}  my deaths")
    for mp in maps:
        data = mp.read_bytes()
        size = data.split(b"\n", 1)[0].decode().replace("MAP ", "").replace(" ", "x")
        for rep in range(reps):
            for me_side in ("A", "B"):
                me = BrainPlayer("me", params)
                other = BrainPlayer("opp", opp_params) if kind == "self" else OPPONENTS[kind](seed=rep)
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
                if not quiet:
                    print(f"{mp.stem:22s} {size:>7s} {me_side:>4s} {outcome:>7s} {res.rounds + 1:>6d} "
                          f"{my_len:>5d}/{op_len:<5d} {my_dr:>4d}/{op_dr:<4d}  {dict(mine) or '-'}")
                total["my_len"] += my_len
                total["my_dragons_end"] += my_dr
                if errors:
                    print(errors[0])
    print(f"\n{sum(total[k] for k in ('win', 'draw', 'loss'))} games in {time.perf_counter() - t_all:.1f}s: "
          f"{dict(total)}")
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opponent", default="greedy", choices=[*OPPONENTS, "self", "all"])
    ap.add_argument("--list", action="store_true", help="list the sparring bots and exit")
    ap.add_argument("--maps", default=str(ROOT / "maps"))
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--split", type=int, default=0, help="enable splitting at this length (0 = off)")
    ap.add_argument("--lenient", action="store_true", help="swallow brain exceptions like the shipped bot does")
    ap.add_argument("--deaths", action="store_true", help="print the last decision context of each of my deaths")
    ap.add_argument("--only", default="", help="substring filter on map names")
    ap.add_argument("--params", default="", help="comma list of brain parameter overrides, e.g. pessimistic=0,timed=0")
    ap.add_argument("--opp-params", default=None, help="parameter overrides for the opponent brain (with --opponent self)")
    ap.add_argument("--quiet", action="store_true", help="summary only")
    args = ap.parse_args()
    if args.list:
        for name, cls in OPPONENTS.items():
            print(f"{name:12s} {' '.join(cls.__doc__.split())}")
        return

    global DEBUG_DEATHS
    DEBUG_DEATHS = args.deaths
    Brain.debug = True
    Brain.strict = not args.lenient
    def parse_params(text, base=None):
        out = dict(base or {})
        for kv in filter(None, text.split(",")):
            k, v = kv.split("=")
            out[k] = float(v) if "." in v else int(v)
        return out or None

    params = parse_params(args.params, {"split_len": args.split} if args.split else None)
    opp_params = params if args.opp_params is None else parse_params(args.opp_params)
    engine = EngineModule()
    maps = [m for m in sorted(pathlib.Path(args.maps).glob("*.map")) if args.only in m.stem]
    kinds = list(OPPONENTS) if args.opponent == "all" else [args.opponent]
    results = {}
    for kind in kinds:
        if len(kinds) > 1:
            print(f"\n=== vs {kind} ===")
        results[kind] = versus(kind, engine, maps, args.reps, params, opp_params, args.quiet or len(kinds) > 1)
    if len(kinds) > 1:
        print(f"\n{'opponent':12s} {'win':>4s} {'draw':>4s} {'loss':>4s} {'my deaths':>9s} {'errors':>6s}")
        for kind, tot in results.items():
            died = sum(v for k, v in tot.items() if k.startswith("died:"))
            print(f"{kind:12s} {tot['win']:>4d} {tot['draw']:>4d} {tot['loss']:>4d} {died:>9d} {tot['errors']:>6d}")


if __name__ == "__main__":
    main()
