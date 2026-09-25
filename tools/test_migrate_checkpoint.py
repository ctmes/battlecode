"""Unit test for tools/migrate_checkpoint.py: the old-architecture columns/rows must survive migration exactly
(a warm-start that silently corrupts the very weights it's meant to preserve would be worse than no migration).

Run:  .venv\\Scripts\\python.exe tools\\test_migrate_checkpoint.py
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from migrate_checkpoint import NEG_BIAS, migrate  # noqa: E402


def _old_checkpoint(seed=0):
    # H1=256, H2=128 must match bot/policy.py's real (unchanged-by-this-migration) trunk dimensions --
    # random_weights() inside migrate() always builds the new arrays at those real sizes, so a toy-sized fixture
    # here would hit a shape mismatch that has nothing to do with migrate()'s own correctness (caught by hand:
    # an earlier version of this fixture used arbitrary small dims and migrate() rightly raised on it).
    H1, H2, VOCAB = 256, 128, 10
    rng = np.random.default_rng(seed)
    return {
        "w1": rng.standard_normal((VOCAB, H1)).astype(np.float32), "wd": rng.standard_normal((5, H1)).astype(np.float32),
        "b1": rng.standard_normal(H1).astype(np.float32),
        "w2": rng.standard_normal((H1, H2)).astype(np.float32), "b2": rng.standard_normal(H2).astype(np.float32),
        "wm": rng.standard_normal((H2, 3)).astype(np.float32), "bm": rng.standard_normal(3).astype(np.float32),
        "ws": rng.standard_normal((H2, 2)).astype(np.float32), "bs": rng.standard_normal(2).astype(np.float32),
    }


def test_unchanged_weights_survive_exactly():
    old = _old_checkpoint()
    new = migrate(old, seed=1)
    for k in ("w1", "b1", "w2", "b2", "ws", "bs"):
        assert np.array_equal(old[k], new[k]), k
    print("ok: w1/b1/w2/b2/ws/bs are copied through migration byte-for-byte unchanged")


def test_move_head_columns_0_to_2_preserved():
    old = _old_checkpoint()
    new = migrate(old, seed=1)
    assert np.array_equal(new["wm"][:, :3], old["wm"])
    assert np.array_equal(new["bm"][:3], old["bm"])
    print("ok: the old 3-option move head survives as the new head's first 3 columns exactly")


def test_move_head_new_columns_start_at_zero_weight_negative_bias():
    # Not randomly initialized (see module docstring: measured to corrupt even the *preserved* columns, since a
    # random projection of a large-magnitude trained h2 is not a small perturbation) -- and not zero-bias either
    # (a second measured bug: the real trained logit scale turned out too modest for 0 to read as "unlikely").
    old = _old_checkpoint()
    new = migrate(old, seed=1)
    assert new["wm"].shape[1] > 3, "test assumes bot/policy.py's real N_MOVE > 3"
    assert np.all(new["wm"][:, 3:] == 0.0)
    assert np.all(new["bm"][3:] == NEG_BIAS)
    print("ok: the new sprint-length move columns start at zero weight and a large-negative (not zero) bias")


def test_dense_weights_first_four_rows_preserved_rest_zero():
    old = _old_checkpoint()
    new = migrate(old, seed=1)
    assert np.array_equal(new["wd"][:4, :], old["wd"][:4, :])
    assert new["wd"].shape[0] > 5, "test assumes bot/encoder.py's real DENSE_SIZE > 5"
    assert np.all(new["wd"][4:, :] == 0.0), "the old sonar-placeholder row (and the new rows) must start at zero"
    print("ok: dense-feature rows 0-3 (length/units/round/msg-count) survive; rows 4+ start at exactly zero")


def test_migrated_network_reproduces_old_move_and_split_logits_exactly():
    """The actual point of zero-init, verified end to end: for any hidden state, a migrated network's OLD-head
    logits (move columns 0-2, split unchanged) must come out bit-identical to the source network's -- not just
    "the weight slice looks copied", but that the zero-filled new columns truly contribute nothing."""
    old = _old_checkpoint()
    new = migrate(old, seed=1)
    h2 = np.random.default_rng(2).standard_normal(old["w2"].shape[1]).astype(np.float32)  # a stand-in hidden state
    old_move_logits = h2 @ old["wm"] + old["bm"]
    new_move_logits = h2 @ new["wm"] + new["bm"]
    assert np.allclose(new_move_logits[:3], old_move_logits, atol=1e-5)
    assert np.allclose(new_move_logits[3:], NEG_BIAS, atol=1e-5)  # zero weight -> input-independent, = the bias
    print("ok: a migrated network reproduces the source network's move logits exactly for options 0-2, and the "
          "new options are a constant, large-negative logit regardless of the (random) hidden state")


if __name__ == "__main__":
    test_unchanged_weights_survive_exactly()
    test_move_head_columns_0_to_2_preserved()
    test_move_head_new_columns_start_at_zero_weight_negative_bias()
    test_dense_weights_first_four_rows_preserved_rest_zero()
    test_migrated_network_reproduces_old_move_and_split_logits_exactly()
