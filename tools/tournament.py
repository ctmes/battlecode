"""Round-robin: every named brain against every other, both sides, on shared maps. For deciding between tuned
candidates that each beat the reference opponents but whose ranking against each other isn't otherwise clear
(single-opponent validation is transitive-blind: A > C and B > C does not imply a result for A vs B).

    .venv\\Scripts\\python.exe tools\\tournament.py defaults tools\\tuned\\run1_avg5.json tools\\tuned\\run2_avg5.json ^
        [--games 300] [--max-side 64] [--seed 300000] [--no-bundled] [--workers N]

A participant is a name registered in tune.OPPONENTS ("defaults" = brain.DEFAULTS unchanged), or the path of a tuned
.json. Every unordered pair plays --games generated maps split evenly across both sides (plus the bundled maps,
reported separately), so it is (games // 2) maps x 2 sides per pair, same maps shared across all pairs.
"""
import argparse
import itertools
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from league import Job, League, summarize, wilson  # noqa: E402  (also puts bot/ on the path)
from tune import bundled, opponent  # noqa: E402


def load(name):
    return name, opponent(name)[1] if opponent(name)[0] == "brain" else None  # bots can't be tournament participants


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("participants", nargs="+", help="names from tune.OPPONENTS, or tuned .json paths (need >= 2)")
    ap.add_argument("--games", type=int, default=300, help="generated-map games per pair (split evenly, both sides)")
    ap.add_argument("--max-side", type=int, default=64)
    ap.add_argument("--seed", type=int, default=300000, help="held out from every tuning run and validate() by default")
    ap.add_argument("--no-bundled", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    if len(args.participants) < 2:
        ap.error("need at least 2 participants")

    entries = []
    for name in args.participants:
        label, params = load(name)
        if params is None:
            ap.error(f"{name}: only brain candidates ('defaults' or a tuned .json) can enter a tournament")
        entries.append((pathlib.Path(label).stem if label.endswith(".json") else label, params))
    labels = [e[0] for e in entries]
    if len(set(labels)) != len(labels):
        ap.error(f"duplicate labels after shortening to stems: {labels}")

    n_maps = args.games // 2
    generated = [(args.seed + i, 10, args.max_side) for i in range(n_maps)]
    real = [] if args.no_bundled else bundled(64)
    print(f"{len(entries)} participants, {len(list(itertools.combinations(entries, 2)))} pairs, "
          f"{n_maps} generated maps x 2 sides + {len(real)} bundled x 2 sides per pair, seed {args.seed}+", flush=True)

    jobs, index = [], []
    for (la, pa), (lb, pb) in itertools.combinations(entries, 2):
        for label, maps in (("generated", generated), ("bundled", real)):
            for m in maps:
                for side in "AB":
                    jobs.append(Job(m, side, pa, ("brain", pb)))
                    index.append((la, lb, label))
    with League(args.workers) as lg:
        results = lg.run(jobs)

    per_pair = {}
    for (la, lb, label), r in zip(index, results):
        per_pair.setdefault((la, lb, label), []).append(r)

    score = {l: 0.0 for l in labels}  # sum of match points across every pair this label played (generated maps only)
    played = {l: 0 for l in labels}
    print(f"\n{'A':12s} {'B':12s} {'maps':9s} {'games':>5s}  {'A score':>8s}  {'95% CI':>16s}")
    for la, lb in itertools.combinations(labels, 2):
        for label in ("generated", "bundled") if real else ("generated",):
            rs = per_pair[(la, lb, label)]
            s = summarize(rs)
            lo, hi = wilson(s["score"], s["games"])
            print(f"{la:12s} {lb:12s} {label:9s} {s['games']:5d}  {s['score']:7.1%}  [{lo:.1%}, {hi:.1%}]")
            if label == "generated":
                score[la] += s["score"] * s["games"]
                score[lb] += (1 - s["score"]) * s["games"]
                played[la] += s["games"]
                played[lb] += s["games"]

    print(f"\n{'rank':4s} {'label':12s} {'score vs field':>14s}  (generated maps only, ties broken by name)")
    for i, l in enumerate(sorted(labels, key=lambda l: (-score[l] / played[l], l)), 1):
        print(f"{i:4d} {l:12s} {score[l] / played[l]:13.1%}   ({int(score[l])}/{played[l]})")


if __name__ == "__main__":
    main()
