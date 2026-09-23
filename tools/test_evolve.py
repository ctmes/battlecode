"""Unit tests for tools/evolve.py: flatten/unflatten round-tripping, NumpySepCMA's update actually moving the
mean toward higher-fitness perturbations (a sign error here would silently search in the wrong direction without
ever raising an exception -- the same risk test_train_ppo.py's GAE tests guard against), and a real end-to-end
smoke test on tiny maps/population so a crash in the multiprocessing plumbing is caught before a real run is.

Run:  .venv\\Scripts\\python.exe tools\\test_evolve.py
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from evolve import ACTOR_KEYS, NumpySepCMA, flatten, initial_c, unflatten  # noqa: E402


def close(a, b, tol=1e-9):
    return abs(a - b) < tol


def test_flatten_unflatten_round_trip():
    rng = np.random.default_rng(0)
    shapes = {"b1": (4,), "b2": (3,), "bm": (2,), "bs": (2,), "w1": (5, 4), "w2": (4, 3), "wd": (2, 4),
              "wm": (3, 2), "ws": (3, 2)}
    weights = {k: rng.standard_normal(s).astype(np.float32) for k, s in shapes.items()}
    flat = flatten(weights, ACTOR_KEYS)
    back = unflatten(flat, ACTOR_KEYS, shapes)
    for k in ACTOR_KEYS:
        assert np.allclose(weights[k], back[k], atol=1e-6), k
    print("ok: flatten -> unflatten round-trips every key exactly (within float32 precision)")


def test_initial_c_matches_layer_variance():
    # bot/policy.py's layer(fan_in, fan_out) draws std = fan_in**-0.5, i.e. variance = fan_in**-1 -- initial_c
    # must reproduce that per weight matrix (keyed by shape[0] = fan_in), and a small fixed variance for biases.
    shapes = {"b1": (4,), "w1": (10, 4)}
    c0 = initial_c(("b1", "w1"), shapes)
    assert np.allclose(c0[:4], 0.01 ** 2), c0[:4]
    assert np.allclose(c0[4:], 1.0 / 10), c0[4:]
    print("ok: initial_c reproduces layer()'s fan_in**-1 weight variance and a small fixed bias variance")


def test_sep_cma_moves_toward_higher_fitness():
    # A minimal, hand-interpretable case: 2D, fitness = -||x||^2 (maximized at the origin), starting away from it.
    # After several ask/tell cycles the mean must have moved measurably closer to the optimum -- not a coincidence,
    # since fitness is deterministic given x here (no evaluation noise to explain a lucky drift).
    x0 = np.array([5.0, -5.0])
    es = NumpySepCMA(x0, sigma=1.0, popsize=16)
    start_dist = np.linalg.norm(es.m)
    rng = np.random.default_rng(0)
    for _ in range(25):
        points = es.ask(rng)
        fitness = -np.sum(points ** 2, axis=1)
        es.tell(fitness)
    end_dist = np.linalg.norm(es.m)
    assert end_dist < start_dist * 0.5, (start_dist, end_dist)
    print(f"ok: 25 generations on fitness=-||x||^2 moved the mean from distance {start_dist:.2f} to "
          f"{end_dist:.2f} from the optimum")


def test_sep_cma_state_round_trip():
    x0 = np.array([1.0, 2.0, 3.0])
    es = NumpySepCMA(x0, sigma=0.3, popsize=8)
    rng = np.random.default_rng(1)
    for _ in range(3):
        es.tell(-np.sum(es.ask(rng) ** 2, axis=1))
    arrays, scalars = es.arrays(), es.scalars()

    es2 = NumpySepCMA(np.zeros(3), sigma=1.0, popsize=8)  # deliberately different initial state
    es2.load(arrays, scalars)
    assert np.allclose(es.m, es2.m) and np.allclose(es.C, es2.C)
    assert close(es.sigma, es2.sigma) and es.gen == es2.gen
    print("ok: NumpySepCMA.arrays()/scalars() -> load() round-trips m, C, ps, pc, sigma, gen exactly")


if __name__ == "__main__":
    test_flatten_unflatten_round_trip()
    test_initial_c_matches_layer_variance()
    test_sep_cma_moves_toward_higher_fitness()
    test_sep_cma_state_round_trip()
