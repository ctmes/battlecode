"""torch mirror of bot/policy.py's architecture, used only for training (BC, then PPO). Layer shapes and the single
shared first-layer bias match exactly, so a trained network's weights export straight into the numpy dict
bot/policy.py's Policy.forward() expects -- torch never ships (not in the judge's sandbox; bot.toml only includes
bot/*.py, and tools/ is dev-only); the deployed bot and tools/policy.py's untrained scaffold stay numpy-only.

Numpy forward (bot/policy.py): h1 = relu(W1[idx].sum(0) + dense @ Wd + b1); h2 = relu(h1 @ W2 + b2); heads = h2 @ W + b.
Torch forward here: h1 = relu(EmbeddingBag(idx, offsets) + Linear_d(dense)); h2 = relu(Linear_2(h1)); heads = Linear(h2).
EmbeddingBag has no bias of its own (mode="sum" matches W1[idx].sum(0) exactly) -- the dense path's Linear supplies
the single shared bias (its .bias is exactly b1), so there is only ever one first-layer bias term, as in the numpy
version. torch's nn.Linear.weight is (out_features, in_features); the numpy dict's convention is (in, out) (see
bot/policy.py's `layer()`), so export/import transpose every Linear weight -- EmbeddingBag.weight needs no
transpose, its shape (vocab, H1) already matches numpy's "w1" directly.

A `value_head` (h2 -> 1) shares the same trunk as the two action heads: a standard shared-parameter actor-critic,
needed once PPO (not BC) starts, for GAE. It sees the same per-dragon egocentric features as the actor -- the
plan's more accurate option (a critic over "the union of the team's views") needs a custom simulator to assemble
a full-team state and is explicitly deferred there ("revisit if value learning is the bottleneck"); this is the
simpler v1. A BC-era checkpoint (tools/bc/bc.npz) has no "wv"/"bv" -- import_numpy random-initialises just that
head rather than failing, so PPO can bootstrap the actor from BC without retraining it with a value head from
scratch.
"""
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "bot"))
import encoder  # noqa: E402
import policy as np_policy  # noqa: E402 - H1, H2, N_MOVE, SPLIT_FRACS live here, one definition shared both ways

H1, H2, N_MOVE = np_policy.H1, np_policy.H2, np_policy.N_MOVE
N_SPLIT = len(np_policy.SPLIT_FRACS) + 1


class PolicyNet(nn.Module):
    def __init__(self, vocab=encoder.VOCAB, dense_size=encoder.DENSE_SIZE, h1=H1, h2=H2):
        super().__init__()
        self.embed = nn.EmbeddingBag(vocab, h1, mode="sum")
        self.dense = nn.Linear(dense_size, h1, bias=True)  # its bias is the shared b1 (see module docstring)
        self.layer2 = nn.Linear(h1, h2, bias=True)
        self.move_head = nn.Linear(h2, N_MOVE, bias=True)
        self.split_head = nn.Linear(h2, N_SPLIT, bias=True)
        self.value_head = nn.Linear(h2, 1, bias=True)

    def forward(self, flat_idx, offsets, dense):
        h1 = torch.relu(self.embed(flat_idx, offsets) + self.dense(dense))
        h2 = torch.relu(self.layer2(h1))
        return self.move_head(h2), self.split_head(h2), self.value_head(h2).squeeze(-1)


def export_numpy(net):
    """PolicyNet -> the numpy weight dict bot/policy.py's Policy.forward() consumes. Copies each tensor to CPU
    individually rather than `net.to("cpu")`, which mutates the module's own device in place (a live problem
    mid-training on GPU: the caller's net would silently end up on CPU while its optimizer/data stayed on CUDA)."""
    def a(t):
        return t.detach().cpu().numpy().copy()

    with torch.no_grad():
        return {
            "w1": a(net.embed.weight),
            "wd": a(net.dense.weight).T.copy(), "b1": a(net.dense.bias),
            "w2": a(net.layer2.weight).T.copy(), "b2": a(net.layer2.bias),
            "wm": a(net.move_head.weight).T.copy(), "bm": a(net.move_head.bias),
            "ws": a(net.split_head.weight).T.copy(), "bs": a(net.split_head.bias),
            "wv": a(net.value_head.weight).T.copy(), "bv": a(net.value_head.bias),
        }


def import_numpy(weights, device="cpu"):
    """The reverse of export_numpy: load a numpy weight dict (e.g. from random_weights(), or a BC checkpoint) into
    a fresh PolicyNet -- used to resume/fine-tune (PPO) from a BC-trained policy. "wv"/"bv" (the value head) are
    optional: a BC-era checkpoint has no critic, so a missing value head is left at PolicyNet's fresh random init
    rather than treated as an error."""
    vocab, h1 = weights["w1"].shape
    dense_size = weights["wd"].shape[0]
    h2 = weights["w2"].shape[1]
    net = PolicyNet(vocab, dense_size, h1, h2)
    with torch.no_grad():
        net.embed.weight.copy_(torch.from_numpy(np.ascontiguousarray(weights["w1"])))
        net.dense.weight.copy_(torch.from_numpy(np.ascontiguousarray(weights["wd"].T)))
        net.dense.bias.copy_(torch.from_numpy(np.ascontiguousarray(weights["b1"])))
        net.layer2.weight.copy_(torch.from_numpy(np.ascontiguousarray(weights["w2"].T)))
        net.layer2.bias.copy_(torch.from_numpy(np.ascontiguousarray(weights["b2"])))
        net.move_head.weight.copy_(torch.from_numpy(np.ascontiguousarray(weights["wm"].T)))
        net.move_head.bias.copy_(torch.from_numpy(np.ascontiguousarray(weights["bm"])))
        net.split_head.weight.copy_(torch.from_numpy(np.ascontiguousarray(weights["ws"].T)))
        net.split_head.bias.copy_(torch.from_numpy(np.ascontiguousarray(weights["bs"])))
        if "wv" in weights and "bv" in weights:
            net.value_head.weight.copy_(torch.from_numpy(np.ascontiguousarray(weights["wv"].T)))
            net.value_head.bias.copy_(torch.from_numpy(np.ascontiguousarray(weights["bv"])))
    return net.to(device)


def batch_to_tensors(idx_lists, dense_rows, device="cpu"):
    """A list of per-sample idx lists + a list of dense rows -> (flat_idx, offsets, dense) for PolicyNet.forward,
    in the nn.EmbeddingBag(input, offsets) convention: offsets[i] is where sample i's indices start in flat_idx."""
    offsets = torch.zeros(len(idx_lists), dtype=torch.long)
    pos = 0
    for i, idx in enumerate(idx_lists):
        offsets[i] = pos
        pos += len(idx)
    flat = torch.zeros(pos, dtype=torch.long)
    pos = 0
    for idx in idx_lists:
        flat[pos:pos + len(idx)] = torch.as_tensor(idx, dtype=torch.long)
        pos += len(idx)
    dense = torch.as_tensor(np.asarray(dense_rows, dtype=np.float32))
    return flat.to(device), offsets.to(device), dense.to(device)
