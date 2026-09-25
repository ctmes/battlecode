"""Evolutionary fine-tuning of the RL policy network (bot/policy.py's Policy): a numpy port of tools/tune.py's
separable CMA-ES, scaled from Brain's ~20 hand-tuned scalars to the network's ~251k weights, scored by win rate
vs Brain -- same underlying algorithm this project already trusts (Stage 3: won a 5-way, 3160-game round robin at
59.5%), not a new one.

Why evolution, not more PPO: three full PPO experiments this stage (collapse -> fixed -> stable-but-flat; Phase 1
hyperparameter/PopArt tuning -> still flat; brain_frac=1.0 diagnostic -> actively worse) never once beat the BC
baseline it started from, despite healthy training diagnostics throughout. The recurring root cause has been
credit assignment: a split's payoff can arrive hundreds of turns later, past GAE's effective horizon (~10-20
steps at gamma=0.99, lambda=0.95) -- see the battlecode-stage5-ppo memory note for the full trail. Evolutionary
methods sidestep this entirely: fitness is the WHOLE GAME's outcome, there is no per-step value function, no GAE,
no reward shaping to get subtly wrong (the exact bug category that has cost three debugging rounds here). This
matches published results: OpenAI's ES paper (Salimans et al. 2017, arXiv:1703.03864) reports ES "does not suffer
in settings with sparse rewards" precisely because it never needs to propagate credit backward through time; Uber
AI's Deep Neuroevolution (Such et al. 2017, arXiv:1712.06567) trained networks with *millions* of parameters (our
network has ~251k) with a genetic algorithm simpler than what's implemented here and beat DQN/A3C/ES on many
Atari games. See docs/rl-improvement-plan.md and the battlecode-stage5-ppo-still-collapsing memory note for the
full reasoning trail that led here.

Why NOT reuse tools/tune.py's SepCMA class directly: it operates on a [0,1]-clipped unit box (Brain's parameters
are bounded and `encode()`/`decode()` map them there) -- network weights are unbounded real numbers, so this port
drops that clipping. It also uses plain Python lists/loops, fine at Brain's ~20 dimensions but far too slow at
~251k (profiled: unacceptable per-generation overhead) -- this port is numpy-vectorized, same update equations.

One more addition beyond tune.py's own methodology: tune.py already shares generated maps across all candidates
in a generation specifically to cancel map-difficulty noise ("because bots are deterministic the maps are the
only noise, and sharing them across candidates removes most of it" -- its own docstring). A trained Policy is
NOT deterministic even given fixed weights: `choose()` samples from a softmax, not argmax (see bot/policy.py).
So there is a second noise source tune.py's Brain-only world never had. This port extends the same
common-random-numbers principle to it: the Policy's RNG seed for a given (generation, map, side) slot is shared
across every candidate evaluated in that slot, so the *only* thing that can differ between two candidates' scores
on the same slot is their weights, not an unlucky sample.

Deployment note: the evolved weights dict has exactly bot/policy.py's Policy format (w1/wd/b1/w2/b2/wm/bm/ws/bs,
the same keys BC produces) -- no value head, no new deployment path, no change to the judge-side CPU budget.
Directly loadable by tools/eval_bc.py and bot/main.py exactly like a BC or PPO-actor checkpoint.

    .venv\\Scripts\\python.exe tools\\evolve.py run --name run1 --generations 40 --pop 24 --maps 10
    .venv\\Scripts\\python.exe tools\\evolve.py run --name run1 --resume
    .venv\\Scripts\\python.exe tools\\evolve.py validate tools\\evolved\\run1.npz --games 200
"""
import argparse
import concurrent.futures
import json
import os
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bot"))
sys.path.insert(0, str(ROOT / "tools"))
import encoder  # noqa: E402
import mapgen  # noqa: E402
from brain import Brain  # noqa: E402
from league import wilson  # noqa: E402  (generic, not Brain-specific -- see its docstring)
from opponents import OPPONENTS  # noqa: E402
from policy import Policy, random_weights  # noqa: E402
from unswbc.engine import EngineModule  # noqa: E402

EVOLVED = ROOT / "tools" / "evolved"

# Fixed, canonical key set (sorted) for bot/policy.py's Policy -- actor-only, no value head. A --start checkpoint
# may be a PPO checkpoint carrying extra keys ("wv", "bv", "value_mu", "value_sigma", the last two 0-d scalars
# initial_c() cannot derive a fan-in from); this project's evolutionary fine-tuning never needs a critic at all
# (see module docstring), so those are always dropped rather than accidentally evolved.
ACTOR_KEYS = ("b1", "b2", "bm", "bs", "w1", "w2", "wd", "wm", "ws")


# ---------------------------------------------------------------------- flatten / unflatten
def flatten(weights, keys):
    return np.concatenate([weights[k].ravel().astype(np.float64) for k in keys])


def unflatten(vector, keys, shapes):
    out, i = {}, 0
    for k in keys:
        shape = shapes[k]
        n = int(np.prod(shape))
        out[k] = vector[i:i + n].reshape(shape).astype(np.float32)
        i += n
    return out


def initial_c(keys, shapes):
    """Per-parameter starting variance for the diagonal covariance, matching each layer's own init scale
    (bot/policy.py's `layer()`: std = fan_in**-0.5, i.e. variance = fan_in**-1) rather than a single uniform
    value across the whole flat vector. Without this, one global sigma is either wildly too large for the
    ~845-wide embedding table or wildly too small for the output heads (fan_in 128) -- sep-CMA-ES's diagonal C
    *would* eventually adapt this on its own, but starting from a sane per-layer scale avoids wasting early
    generations on a badly-calibrated search before it does. Biases (initialized to 0, no natural scale to
    infer) get a small fixed variance instead."""
    parts = []
    for k in keys:
        shape = shapes[k]
        n = int(np.prod(shape))
        if k.startswith("b"):  # b1, b2, bm, bs: biases, no fan-in to derive a scale from
            parts.append(np.full(n, 0.01 ** 2))
        else:  # w1, wd, w2, wm, ws: weight matrices, shape (fan_in, fan_out)
            fan_in = shape[0]
            parts.append(np.full(n, 1.0 / fan_in))
    return np.concatenate(parts)


# ---------------------------------------------------------------------- numpy separable CMA-ES (maximizes, unbounded)
class NumpySepCMA:
    """Direct numpy port of tools/tune.py's SepCMA (same update equations, same trust), minus the [0,1] unit-box
    clipping that class needs for Brain's bounded parameters -- network weights are unbounded reals. c0 seeds the
    diagonal covariance per-dimension (see initial_c) instead of the uniform 1.0 tune.py's bounded domain uses."""

    def __init__(self, x0, sigma, popsize, c0=None):
        n = self.n = len(x0)
        self.lam, self.mu = popsize, popsize // 2
        raw = np.array([np.log(self.mu + 0.5) - np.log(i + 1) for i in range(self.mu)])
        self.w = raw / raw.sum()
        self.mueff = 1.0 / np.sum(self.w ** 2)
        me = self.mueff
        self.cs = (me + 2) / (n + me + 5)
        self.ds = 1 + 2 * max(0.0, np.sqrt((me - 1) / (n + 1)) - 1) + self.cs
        self.cc = (4 + me / n) / (n + 4 + 2 * me / n)
        c1 = 2 / ((n + 1.3) ** 2 + me)
        cmu = 2 * (me - 2 + 1 / me) / ((n + 2) ** 2 + me)
        speed = (n + 2) / 3  # Ros & Hansen: a diagonal covariance can learn faster
        self.c1 = min(1.0, c1 * speed)
        self.cmu = min(1 - self.c1, cmu * speed)
        self.chi = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.m = np.array(x0, dtype=np.float64)
        self.sigma = sigma
        self.C = np.array(c0, dtype=np.float64) if c0 is not None else np.ones(n)
        self.ps, self.pc, self.gen = np.zeros(n), np.zeros(n), 0
        self.ys = None

    def ask(self, rng):
        """lam points, unclipped."""
        self.ys = rng.standard_normal((self.lam, self.n)) * np.sqrt(self.C)
        return self.m[None, :] + self.sigma * self.ys

    def tell(self, fitness):
        n, w = self.n, self.w
        best = np.argsort(fitness)[::-1][: self.mu]
        yw = (w[:, None] * self.ys[best]).sum(axis=0)
        self.m = self.m + self.sigma * yw
        k = np.sqrt(self.cs * (2 - self.cs) * self.mueff)
        self.ps = (1 - self.cs) * self.ps + k * yw / np.sqrt(self.C)
        norm = np.sqrt(np.sum(self.ps ** 2))
        self.gen += 1
        hs = 1.0 if norm / np.sqrt(1 - (1 - self.cs) ** (2 * self.gen)) / self.chi < 1.4 + 2 / (n + 1) else 0.0
        kc = np.sqrt(self.cc * (2 - self.cc) * self.mueff)
        self.pc = (1 - self.cc) * self.pc + hs * kc * yw
        rank_mu = (w[:, None] * self.ys[best] ** 2).sum(axis=0)
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (self.pc ** 2 + (1 - hs) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * rank_mu)
        self.sigma *= np.exp(self.cs / self.ds * (norm / self.chi - 1))

    def arrays(self):
        """The large (n-length) numeric state -- saved as .npz, not JSON (n ~251k floats x4 arrays would bloat a
        JSON file badly; .npz is binary and both smaller and faster to read back)."""
        return {"m": self.m, "C": self.C, "ps": self.ps, "pc": self.pc}

    def scalars(self):
        return {"sigma": self.sigma, "gen": self.gen}

    def load(self, arrays, scalars):
        self.m, self.C, self.ps, self.pc = arrays["m"], arrays["C"], arrays["ps"], arrays["pc"]
        self.sigma, self.gen = scalars["sigma"], scalars["gen"]


# ---------------------------------------------------------------------- evaluation
_population = None
_keys = None
_shapes = None


def _pool_init(pop_bytes, keys, shapes):
    global _population, _keys, _shapes
    _population = pop_bytes  # list of raw flat float64 arrays; unflattened lazily per job (cheap)
    _keys, _shapes = keys, shapes


def _play_one(candidate_idx, map_bytes, side, opp_factory, seed):
    """opp_factory(seed) -> a team-level object with spawn(did, init)/reply(did, block) -- the same interface
    tools/opponents.py's Player base class and tools/arena.py's BrainPlayer already use throughout this project
    (one object manages every dragon on that side, not one instance per dragon), so a sparring bot's own
    constructor (`OPPONENTS[name]`) can be used as opp_factory directly with no adapter needed."""
    weights = unflatten(_population[candidate_idx], _keys, _shapes)
    opp = opp_factory(seed)
    mine = {}

    def bot_spawn(did, init):
        team = next(line for line in init.split(b"\n") if line.startswith(b"TEAM")).split()[1]
        if team == side.encode():
            mine[did] = Policy.from_init(init, weights, seed=seed * 1000 + did)
        else:
            opp.spawn(did, init)

    def bot_reply(did, block):
        try:
            return mine[did].act(block) if did in mine else opp.reply(did, block)
        except Exception:  # noqa: BLE001 - a crash shouldn't wedge the whole generation
            return b""

    engine = EngineModule()
    res = engine.run(map_bytes, bot_reply, None, bot_spawn, lambda line: None, 0)
    return 0.5 if res.winner is None else float(res.winner == side)


class _BrainOpponent:
    def __init__(self, seed):
        self.brains = {}

    def spawn(self, did, init):
        self.brains[did] = Brain.from_init(init)

    def reply(self, did, block):
        return self.brains[did].act(block)


class _RandomOpponent:
    """An untrained Policy opponent for validate()'s sanity-floor check, matching eval_bc.py's RandomPlayer."""

    _weights = None

    def __init__(self, seed):
        if _RandomOpponent._weights is None:
            _RandomOpponent._weights = random_weights(0, encoder.VOCAB, encoder.DENSE_SIZE)
        self.seed = seed
        self.policies = {}

    def spawn(self, did, init):
        self.policies[did] = Policy.from_init(init, _RandomOpponent._weights, seed=self.seed * 1000 + did)

    def reply(self, did, block):
        return self.policies[did].act(block)


# "brain" and "random" are handled here (not registered in tools/opponents.py's OPPONENTS, which is sparring
# bots only); any other name is looked up there directly, since its Player subclasses already match opp_factory's
# spawn/reply interface exactly.
def _opponent_factory(name):
    if name == "brain":
        return _BrainOpponent
    if name == "random":
        return _RandomOpponent
    return OPPONENTS[name]


def parse_vs(text):
    """'brain:2,splitter:1' -> [('brain', 2.0), ('splitter', 1.0)] -- same format as tools/tune.py's parse_vs,
    for the same reason: a weighted mix of training opponents, not just one."""
    out = []
    for item in filter(None, text.split(",")):
        name, sep, w = item.rpartition(":")
        try:
            out.append((name, float(w)) if sep and name else (item, 1.0))
        except ValueError:
            out.append((item, 1.0))
    return out


def _play_job(job):
    candidate_idx, map_bytes, side, opp_name, seed = job
    score = _play_one(candidate_idx, map_bytes, side, _opponent_factory(opp_name), seed)
    return candidate_idx, score


def _validate_job(job):
    """A module-level (picklable) dispatcher -- a closure over `kind` defined inside validate()'s own
    ProcessPoolExecutor block would NOT be picklable and crash the pool immediately, since local functions
    can't cross a process boundary."""
    kind, label, map_bytes, side, seed = job
    score = _play_one(0, map_bytes, side, _opponent_factory(kind), seed)
    return kind, label, score


def play_population(pop_flat, keys, shapes, maps, generation, vs, workers):
    """Every candidate in pop_flat plays every map in `maps` on both sides vs an opponent drawn from `vs`
    (a parse_vs()-style weighted list), sharing the maps, the opponent choice, AND the policy's rng seed per
    (generation, map, side) slot across all candidates (common random numbers -- see module docstring): the
    *same* opponent on a given slot for every candidate, so a weaker candidate can't get lucky with an easier
    draw. Returns a (len(pop_flat),) array of mean scores."""
    names = [n for n, _ in vs]
    probs = np.array([w for _, w in vs], dtype=np.float64)
    probs /= probs.sum()
    jobs = []
    for map_i, spec in enumerate(maps):
        data = mapgen.generate(*spec).dumps().encode() if isinstance(spec, tuple) else pathlib.Path(spec).read_bytes()
        for side_i, side in enumerate(("A", "B")):
            # Plain arithmetic, not Python's built-in hash(): hash() of a str is salted per-process (randomized
            # since 3.3), so it would silently differ across a --resume'd process from the original run -- not a
            # within-run correctness bug (the salt is stable for the run's whole lifetime), but it would break
            # exact reproducibility of a resumed run, which this project's other seeding deliberately preserves.
            seed = (generation * 1_000_003 + map_i * 2 + side_i) & 0x7FFFFFFF
            opp_name = names[0] if len(names) == 1 else names[np.random.default_rng(seed).choice(len(names), p=probs)]
            for c in range(len(pop_flat)):
                jobs.append((c, data, side, opp_name, seed))
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    scores = [[] for _ in pop_flat]
    with concurrent.futures.ProcessPoolExecutor(workers, initializer=_pool_init,
                                                  initargs=(pop_flat, keys, shapes)) as pool:
        for c, score in pool.map(_play_job, jobs, chunksize=4):
            scores[c].append(score)
    return np.array([sum(s) / len(s) for s in scores])


def bundled(max_side):
    out = []
    for p in sorted((ROOT / "maps").glob("*.map")):
        _, w, h = p.open().readline().split()
        if max(int(w), int(h)) <= max_side:
            out.append(str(p))
    return out


RUN_DEFAULTS = {"generations": 40, "pop": 24, "maps": 10, "max_side": 32, "seed": 1000, "sigma": 0.05,
                 "start": "", "no_bundled": False, "vs": "brain"}


def run(args):
    EVOLVED.mkdir(exist_ok=True)
    path = EVOLVED / f"{args.name}.npz"
    state_arrays_path = EVOLVED / f"{args.name}_state.npz"
    meta_path = EVOLVED / f"{args.name}_meta.json"
    saved_meta = json.loads(meta_path.read_text()) if args.resume and meta_path.exists() else None
    for key, fallback in RUN_DEFAULTS.items():
        if getattr(args, key) is None:
            setattr(args, key, (saved_meta["args"][key] if saved_meta else fallback))

    template = dict(np.load(args.start)) if args.start else random_weights(0, encoder.VOCAB, encoder.DENSE_SIZE)
    missing = [k for k in ACTOR_KEYS if k not in template]
    if missing:
        raise ValueError(f"--start {args.start!r} is missing required keys {missing} (expected {ACTOR_KEYS})")
    keys = ACTOR_KEYS
    shapes = {k: template[k].shape for k in keys}
    x0 = flatten(template, keys)
    c0 = initial_c(keys, shapes)

    es = NumpySepCMA(x0, args.sigma, args.pop, c0=c0)
    history = []
    if saved_meta:
        es.load(dict(np.load(state_arrays_path)), saved_meta["scalars"])
        history = saved_meta["history"]

    anchors = list(bundled(args.max_side)) if not args.no_bundled else []
    vs = parse_vs(args.vs)
    print(f"evolving {len(x0):,} parameters, population {args.pop}, {args.maps} generated + {len(anchors)} bundled "
          f"maps x 2 sides = {(args.maps + len(anchors)) * 2} games per candidate per generation, vs {vs}", flush=True)

    while es.gen < args.generations:
        t0 = time.perf_counter()
        g = es.gen
        maps = [(args.seed + g * args.maps + i, 10, args.max_side) for i in range(args.maps)] + anchors
        rng = np.random.default_rng(args.seed + g)
        points = es.ask(rng)
        pop_flat = list(points) + [es.m]  # the mean itself rides along, last, for its own tracked score
        scores = play_population(pop_flat, keys, shapes, maps, g, vs, args.workers)
        es.tell(scores[:-1])

        history.append({"gen": g, "best": float(scores[:-1].max()), "avg": float(scores[:-1].mean()),
                        "mean_score": float(scores[-1]), "sigma": float(es.sigma)})
        h = history[-1]
        dt = time.perf_counter() - t0
        games = len(pop_flat) * (args.maps + len(anchors)) * 2
        print(f"gen {g:3d}  best {h['best']:.3f}  avg {h['avg']:.3f}  mean_score {h['mean_score']:.3f}  "
              f"sigma {es.sigma:.4f}  {dt:.0f}s ({games / dt:.1f} games/s)", flush=True)

        weights = unflatten(es.m, keys, shapes)
        np.savez(path, **weights)
        np.savez(state_arrays_path, **es.arrays())
        meta_path.write_text(json.dumps({"scalars": es.scalars(), "history": history, "args": vars(args),
                                         "keys": list(keys), "shapes": {k: list(v) for k, v in shapes.items()}}))
    print(f"\nwrote {path}")


def validate(args):
    weights = dict(np.load(args.params))
    missing = [k for k in ACTOR_KEYS if k not in weights]
    if missing:
        raise ValueError(f"{args.params!r} is missing required keys {missing} (expected {ACTOR_KEYS})")
    n_maps = -(-args.games // 2)
    generated = [(args.seed + i, 10, args.max_side) for i in range(n_maps)]
    real = bundled(64)
    keys = ACTOR_KEYS
    shapes = {k: weights[k].shape for k in keys}
    flat = [flatten(weights, keys)]

    jobs = []
    for kind in ("random", "brain"):
        for label, maps in (("generated", generated), ("bundled", real)):
            for map_i, spec in enumerate(maps):
                data = mapgen.generate(*spec).dumps().encode() if isinstance(spec, tuple) else pathlib.Path(spec).read_bytes()
                for side in ("A", "B"):
                    jobs.append((kind, label, data, side, args.seed + map_i))
    workers = args.workers or max(1, (os.cpu_count() or 2) - 2)
    results = {}
    with concurrent.futures.ProcessPoolExecutor(workers, initializer=_pool_init,
                                                  initargs=(flat, keys, shapes)) as pool:
        for kind, label, score in pool.map(_validate_job, jobs, chunksize=4):
            results.setdefault((kind, label), []).append(score)
    for kind in ("random", "brain"):
        for label in ("generated", "bundled"):
            scores = results.get((kind, label), [])
            if not scores:
                continue
            s = sum(scores) / len(scores)
            lo, hi = wilson(s, len(scores))
            print(f"vs {kind:6s} {label:9s} {len(scores):4d} games: {s:.1%} [{lo:.1%}, {hi:.1%}]")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--name", required=True)
    r.add_argument("--generations", type=int, default=None)
    r.add_argument("--pop", type=int, default=None)
    r.add_argument("--maps", type=int, default=None)
    r.add_argument("--max-side", type=int, default=None)
    r.add_argument("--seed", type=int, default=None)
    r.add_argument("--sigma", type=float, default=None, help="initial per-dimension step size multiplier")
    r.add_argument("--start", default=None, help="a BC/PPO/evolved .npz to start from (default: fresh random_weights)")
    r.add_argument("--vs", default=None,
                    help="weighted training opponent mix, e.g. 'brain:2,splitter:1' -- names are 'brain' or any "
                         "tools/opponents.py sparring bot (default: brain only)")
    r.add_argument("--no-bundled", action="store_true", default=None)
    r.add_argument("--resume", action="store_true")
    r.add_argument("--workers", type=int, default=None)
    v = sub.add_parser("validate")
    v.add_argument("params", help="an evolved .npz")
    v.add_argument("--games", type=int, default=200)
    v.add_argument("--seed", type=int, default=100000)
    v.add_argument("--max-side", type=int, default=64)
    v.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    {"run": run, "validate": validate}[args.cmd](args)


if __name__ == "__main__":
    main()
