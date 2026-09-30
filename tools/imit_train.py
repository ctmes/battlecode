"""Trains an imitation policy on tools/imit_data.py's samples and exports numpy weights for snapshots/imit_*/brain.py.

Network (same shape as bot/policy.py's, so the numpy forward is the same gather-sum): h1 = relu(W1[idx].sum(0) +
dense @ Wd + b1); h2 = relu(h1 @ W2 + b2); move = h2 @ Wm + bm (9: forward/left/right x 1-3 steps); split = h2 @ Ws +
bs (5: none / child 2 / turnaround / half / die). The split head learns from every sample, the move head only from
turns that moved. The last --val share of samples (whole replays, the order imit_data wrote them) is held out.

    .venv\\Scripts\\python.exe tools\\imit_train.py tools\\imit\\cheji.npz --out snapshots\\imit_cheji\\weights.npz
"""
import argparse
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bot"))
import encoder  # noqa: E402

H1, H2, N_MOVE, N_SPLIT = 256, 128, 9, 5


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.EmbeddingBag(encoder.VOCAB, H1, mode="sum")
        self.dense = nn.Linear(encoder.DENSE_SIZE, H1)  # its bias is b1
        self.l2 = nn.Linear(H1, H2)
        self.move = nn.Linear(H2, N_MOVE)
        self.split = nn.Linear(H2, N_SPLIT)

    def forward(self, idx, offsets, dense):
        h1 = F.relu(self.emb(idx, offsets) + self.dense(dense))
        h2 = F.relu(self.l2(h1))
        return self.move(h2), self.split(h2)


def batches(d, rows, bs, dev):
    for s in range(0, len(rows), bs):
        r = rows[s:s + bs]
        lens = d["lens"][r]
        starts = d["offsets"][r]
        idx = np.concatenate([d["idx"][a:a + n] for a, n in zip(starts, lens)])
        off = np.concatenate([[0], np.cumsum(lens)[:-1]])
        yield (torch.as_tensor(idx, dtype=torch.long, device=dev), torch.as_tensor(off, dtype=torch.long, device=dev),
               torch.as_tensor(d["dense"][r], device=dev), torch.as_tensor(d["move"][r], device=dev),
               torch.as_tensor(d["split"][r], device=dev))


def evaluate(net, d, rows, dev):
    net.eval()
    n_m = ok_m = 0
    conf = np.zeros((N_SPLIT, N_SPLIT), dtype=np.int64)
    nll = 0.0
    with torch.no_grad():
        for idx, off, dense, mv, sp in batches(d, rows, 8192, dev):
            lm, ls = net(idx, off, dense)
            nll += F.cross_entropy(ls, sp, reduction="sum").item()
            m = mv >= 0
            if m.any():
                ok_m += (lm[m].argmax(1) == mv[m]).sum().item()
                n_m += m.sum().item()
            np.add.at(conf, (sp.cpu().numpy(), ls.argmax(1).cpu().numpy()), 1)
    net.train()
    rec = conf.diagonal() / np.maximum(conf.sum(1), 1)
    return ok_m / max(n_m, 1), nll / len(rows), rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val", type=float, default=0.1)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    raw = np.load(args.data)
    d = {k: raw[k] for k in raw.files}
    n = len(d["split"])
    cut = int(n * (1 - args.val))
    train_rows, val_rows = np.arange(cut), np.arange(cut, n)
    net = Net().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    rng = np.random.default_rng(0)
    print(f"{n} samples ({cut} train / {n - cut} held out), device {dev}", flush=True)
    for ep in range(args.epochs):
        t0 = time.time()
        perm = rng.permutation(train_rows)
        tot = 0.0
        for idx, off, dense, mv, sp in batches(d, perm, args.bs, dev):
            lm, ls = net(idx, off, dense)
            loss = F.cross_entropy(ls, sp)
            m = mv >= 0
            if m.any():
                loss = loss + F.cross_entropy(lm[m], mv[m])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(sp)
        acc_m, nll_s, rec = evaluate(net, d, val_rows, dev)
        print(f"epoch {ep}: train loss {tot / cut:.4f}  held out: move acc {acc_m:.3f}  split nll {nll_s:.4f}  split "
              f"recall none/two/rear/half/die {' '.join(f'{x:.2f}' for x in rec)}  {time.time() - t0:.0f}s", flush=True)
    sd = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
    out = ROOT / args.out if not pathlib.Path(args.out).is_absolute() else pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, w1=sd["emb.weight"], wd=sd["dense.weight"].T, b1=sd["dense.bias"], w2=sd["l2.weight"].T,
             b2=sd["l2.bias"], wm=sd["move.weight"].T, bm=sd["move.bias"], ws=sd["split.weight"].T, bs=sd["split.bias"])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
