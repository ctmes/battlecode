"""Checks tools/tune.py and tools/league.py.

Run:  .venv\\Scripts\\python.exe tools\\test_tune.py
"""
import argparse
import json
import math
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import tune  # noqa: E402
from league import Job, League, summarize, wilson  # noqa: E402

SMALL = str(ROOT / "maps" / "default_small.map")


def test_encoding():
    names = list(tune.SPACE)
    assert tune.changed(tune.decode(tune.encode({}, names), names)) == {}, "DEFAULTS do not survive encode/decode"
    rng = random.Random(0)
    for _ in range(200):
        p = tune.decode([rng.uniform(-0.5, 1.5) for _ in names], names)
        for name, v in p.items():
            lo, hi, scale = tune.SPACE[name]
            assert lo <= v <= hi or (name in tune.OFF_AT_TOP and v == 0), f"{name}={v} escapes {lo}..{hi}"
            assert isinstance(v, int) == (scale == "int"), f"{name}={v!r} has the wrong type"
    assert set(tune.SPACE) <= set(tune.DEFAULTS), "SPACE names a parameter the brain does not have"
    print("ok  parameters encode and decode within bounds")


def test_sigma_stays_bounded_with_the_optimum_on_a_bound():
    """Regression test: SepCMA used to clip the mean into the box but feed the unclipped step into its evolution
    path, so with the optimum on a bound sigma grew every generation (0.15 -> 3.9 in 30 generations on exactly this
    problem; full_run1 went 0.138 -> 1.989). With reflect() it must shrink back instead."""
    n, rng = 13, random.Random(3)
    target = [1.0] * 7 + [0.3] * 6
    es = tune.SepCMA([0.5] * n, 0.15, 14)
    for g in range(30):
        pts = es.ask(random.Random(100 + g))
        es.tell([-sum((tune.reflect(x) - t) ** 2 for x, t in zip(p, target)) + rng.gauss(0, 0.05) for p in pts])
    assert es.sigma < 0.3, f"sigma ran away to {es.sigma:.2f} with the optimum on a bound"
    print(f"ok  sigma stays bounded ({es.sigma:.3f} after 30 generations) when the optimum sits on a bound")


def test_no_sentinel_cliffs():
    """Values next to an 'off' sentinel must not flip the behaviour: split_len_max 1-5 disabled splitting outright
    and grow_mod 0/1 are opposite extremes, and both used to be inside SPACE."""
    names = list(tune.SPACE)
    for u in (0.0, 0.01, 0.05, 0.99, 1.0):
        p = tune.decode([u] * len(names), names)
        assert p.get("split_len_max", 0) == 0 or p["split_len_max"] >= 6, p["split_len_max"]
        assert p["grow_mod"] >= 2, p["grow_mod"]
    assert "sonar_terrain" not in tune.SPACE, "sonar_terrain's magnitude is unused: it is an on/off A/B, not a CMA knob"
    print("ok  no parameter range sits next to an on/off cliff")


def test_parse_vs():
    assert tune.parse_vs("defaults:2,old") == [("defaults", 2.0), ("old", 1.0)]
    assert tune.parse_vs(r"D:\x\run1.json") == [(r"D:\x\run1.json", 1.0)]
    assert tune.parse_vs(r"D:\x\run1.json:3,splitter:0.5") == [(r"D:\x\run1.json", 3.0), ("splitter", 0.5)]
    print("ok  opponent lists parse, paths with colons included")


def test_cma_optimizes_a_noisy_objective():
    n, rng = 25, random.Random(1)
    target = [rng.uniform(0.1, 0.9) for _ in range(n)]
    dist = lambda u: math.sqrt(sum((a - b) ** 2 for a, b in zip(u, target)))  # noqa: E731
    es = tune.SepCMA([0.5] * n, 0.15, 16)
    start = dist(es.m)
    for g in range(80):
        pts = es.ask(random.Random(g))
        es.tell([-dist([min(1, max(0, x)) for x in p]) + rng.gauss(0, 0.03) for p in pts])
    assert dist(es.m) < 0.3 * start, f"the mean only moved from {start:.2f} to {dist(es.m):.2f} away from the optimum"
    print(f"ok  CMA-ES closes {start:.2f} -> {dist(es.m):.2f} on a noisy 25-dimensional objective")


def test_wilson():
    lo, hi = wilson(0.5, 400)
    assert 0.45 < lo < 0.46 and 0.54 < hi < 0.55
    assert wilson(0.5, 0) == (0.0, 1.0)
    print("ok  win-rate intervals")


def test_league_is_deterministic_and_side_symmetric():
    """Same parameters on both teams is one game seen from two seats: the two scores must add up to exactly 1."""
    # budget_ns is wall time locally, so with the machine under load the 60M default cuts turns short at random and
    # the test would measure the load, not the league
    nb = {"budget_ns": 10 ** 13}
    jobs = [Job(m, side, nb, ("brain", nb)) for m in (SMALL, (7, 10, 20), (8, 10, 20)) for side in "AB"]
    with League(3) as lg:
        first, second = lg.run(jobs), lg.run(jobs)
    assert [r[:9] for r in first] == [r[:9] for r in second], "league runs are not reproducible"
    assert [r.causes for r in first] == [r.causes for r in second], "death-cause tallies are not reproducible"
    for i in range(0, len(jobs), 2):
        assert first[i].score + first[i + 1].score == 1, f"seats are not symmetric on {jobs[i].map}"
        assert first[i].errors == 0
    s = summarize(first)
    assert s["games"] == 6 and s["score"] == 0.5 and s["errors"] == 0, s
    assert s["deaths_per_k"] == 1000 * sum(s["causes"].values()) / sum(r.turns for r in first), s
    print("ok  league games are reproducible and both seats see the same game")


def test_run_and_resume():
    name, path = "_smoke", ROOT / "tools" / "tuned" / "_smoke.json"
    args = argparse.Namespace(name=name, generations=2, pop=4, maps=1, max_side=16, seed=1000, sigma=0.15, vs="old",
                              params="pearl_near,straight,dead_end", start="", no_bundled=True, resume=False, workers=3)
    try:
        tune.run(args)
        saved = json.loads(path.read_text())
        assert len(saved["history"]) == 2 and saved["names"] == ["pearl_near", "straight", "dead_end"]
        args.generations, args.resume = 3, True
        tune.run(args)
        again = json.loads(path.read_text())
        assert len(again["history"]) == 3 and again["history"][:2] == saved["history"], "resume rewrote history"
        assert all(h["errors"] == 0 for h in again["history"])
    finally:
        path.unlink(missing_ok=True)
    print("ok  a tuning run writes its state and resumes where it stopped")


def test_resume_keeps_the_original_run_settings():
    """`run --name X --resume` with nothing else set must reuse X's population, map count, opponents, etc. -- not
    argparse's own defaults -- since SepCMA's learning rates are derived from the population size at construction
    and never round-trip through the saved state, so a size mismatch would silently start a different search."""
    name, path = "_smoke2", ROOT / "tools" / "tuned" / "_smoke2.json"
    blank = {f: None for f in tune.RUN_DEFAULTS}
    first = argparse.Namespace(name=name, resume=False, workers=3,
                                **{**blank, "generations": 2, "pop": 6, "maps": 1, "max_side": 16, "seed": 2000,
                                   "sigma": 0.15, "vs": "old", "params": "pearl_near,straight", "no_bundled": True})
    try:
        tune.run(first)
        before = json.loads(path.read_text())
        assert before["args"]["pop"] == 6 and before["args"]["maps"] == 1 and before["args"]["vs"] == "old"

        bare = argparse.Namespace(name=name, resume=True, workers=None, **{**blank, "generations": 3})
        tune.run(bare)
        after = json.loads(path.read_text())
        for key in ("pop", "maps", "max_side", "seed", "sigma", "vs", "params", "no_bundled"):
            assert after["args"][key] == before["args"][key], f"bare --resume changed {key}: " \
                f"{before['args'][key]!r} -> {after['args'][key]!r}"
        assert after["args"]["generations"] == 3 and len(after["history"]) == 3
        assert after["history"][:2] == before["history"], "a population-size mismatch would desync the CMA-ES state"

        override = argparse.Namespace(name=name, resume=True, workers=None, **{**blank, "generations": 4, "pop": 6})
        tune.run(override)  # an explicit --pop equal to the original is a no-op; nothing here checks a genuine change
        assert json.loads(path.read_text())["args"]["pop"] == 6
    finally:
        path.unlink(missing_ok=True)
    print("ok  a bare --resume keeps the original run's population, maps and opponents")


def test_average_last_one_uses_only_the_final_mean():
    """Regression test: history[-(args.last - 1):] used to silently average over the WHOLE history whenever
    --last was 1, because Python has no negative zero (-(1 - 1) == 0, and list[0:] is the entire list, not an
    empty slice) -- found by inspection, not a failing run, since average() had no test at all before this."""
    name = "_smoke_avg"
    path = ROOT / "tools" / "tuned" / f"{name}.json"
    out_path = ROOT / "tools" / "tuned" / f"{name}_avg1.json"
    names = ["pearl_near", "straight"]
    saved = {
        "names": names,
        "params": {"pearl_near": 50.0, "straight": 10.0},
        # wildly different from params, so a buggy average (pulling in the whole history) is obviously wrong
        "history": [{"mean_params": {"pearl_near": 999.0, "straight": 999.0}}] * 5,
    }
    path.write_text(json.dumps(saved))
    try:
        tune.average(argparse.Namespace(name=name, last=1))
        got = json.loads(out_path.read_text())["params"]
        assert abs(got["pearl_near"] - 50.0) < 1 and abs(got["straight"] - 10.0) < 1, got
    finally:
        path.unlink(missing_ok=True)
        out_path.unlink(missing_ok=True)
    print("ok  --last 1 uses only the final mean, not the whole history")


def run():
    test_encoding()
    test_sigma_stays_bounded_with_the_optimum_on_a_bound()
    test_no_sentinel_cliffs()
    test_parse_vs()
    test_cma_optimizes_a_noisy_objective()
    test_wilson()
    test_league_is_deterministic_and_side_symmetric()
    test_run_and_resume()
    test_resume_keeps_the_original_run_settings()
    test_average_last_one_uses_only_the_final_mean()


if __name__ == "__main__":
    run()
