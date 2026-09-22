"""Parity check for tools/torch_policy.py: the torch net and bot/policy.py's numpy net must agree, since a trained
torch policy is only useful once its weights run identically through the numpy path that actually ships.

Run:  .venv\\Scripts\\python.exe tools\\test_torch_policy.py
"""
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "bot"))
import encoder  # noqa: E402
import policy as np_policy  # noqa: E402
import torch_policy  # noqa: E402


def test_export_matches_numpy_forward():
    torch.manual_seed(0)
    net = torch_policy.PolicyNet()
    weights = torch_policy.export_numpy(net)

    rng = np.random.default_rng(1)
    for trial in range(20):
        n_active = rng.integers(0, 30)
        idx = sorted(set(rng.integers(0, encoder.VOCAB, size=n_active).tolist()))
        dense = rng.standard_normal(encoder.DENSE_SIZE).astype(np.float32).tolist()

        dummy = np_policy.Policy(1, "A", 20, 20, 4, weights)
        np_move, np_split = dummy.forward(idx, dense)

        flat, offsets, dense_t = torch_policy.batch_to_tensors([idx], [dense])
        with torch.no_grad():
            t_move, t_split = net(flat, offsets, dense_t)
        t_move, t_split = t_move[0].numpy(), t_split[0].numpy()

        assert np.allclose(np_move, t_move, atol=1e-4), f"trial {trial}: move logits differ\n{np_move}\n{t_move}"
        assert np.allclose(np_split, t_split, atol=1e-4), f"trial {trial}: split logits differ\n{np_split}\n{t_split}"
    print(f"ok: {trial + 1} random inputs, torch and numpy forward passes agree within 1e-4")


def test_roundtrip_import_export():
    torch.manual_seed(2)
    net = torch_policy.PolicyNet()
    w1 = torch_policy.export_numpy(net)
    net2 = torch_policy.import_numpy(w1)
    w2 = torch_policy.export_numpy(net2)
    for k in w1:
        assert np.allclose(w1[k], w2[k], atol=1e-6), f"round trip changed {k}"
    print("ok: export -> import -> export round-trips exactly")


def test_random_weights_import_matches_numpy_shapes():
    weights = np_policy.random_weights(3)
    net = torch_policy.import_numpy(weights)
    exported = torch_policy.export_numpy(net)
    for k in weights:
        assert weights[k].shape == exported[k].shape, f"{k}: shape mismatch"
        assert np.allclose(weights[k], exported[k], atol=1e-6), f"{k}: values changed on import"
    print("ok: numpy random_weights() imports into PolicyNet and re-exports unchanged")


if __name__ == "__main__":
    test_export_matches_numpy_forward()
    test_roundtrip_import_export()
    test_random_weights_import_matches_numpy_shapes()
