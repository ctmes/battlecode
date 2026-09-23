"""Behaviour-cloning training: fits tools/torch_policy.py's PolicyNet to tools/bc_data.py's demonstrations by
supervised cross-entropy, then exports numpy weights bot/policy.py's Policy can load directly.

Two losses, both from the plan's action space: split (always supervised -- every sample has a real split label,
0 meaning "correctly chose not to split") and move (only on samples where the teacher didn't split, since the
move head wasn't acted on those turns -- see bc_data.py's docstring).

    .venv\\Scripts\\python.exe tools\\train_bc.py --data tools\\bc\\demo.npz --epochs 6 --out tools\\bc\\bc.npz
"""
import argparse
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "bot"))
import torch_policy  # noqa: E402


def load(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def iterate_batches(d, batch_size, order, device):
    offsets, flat_idx, dense, move, split = d["offsets"], d["flat_idx"], d["dense"], d["move"], d["split"]
    for start in range(0, len(order), batch_size):
        batch = order[start:start + batch_size]
        idx_lists = [flat_idx[offsets[i]:offsets[i + 1]] for i in batch]
        lengths = np.fromiter((len(x) for x in idx_lists), dtype=np.int64, count=len(idx_lists))
        batch_offsets = np.concatenate(([0], np.cumsum(lengths)))[:-1]
        batch_flat = np.concatenate(idx_lists) if idx_lists else np.zeros(0, dtype=np.int32)
        yield (torch.as_tensor(batch_flat.astype(np.int64), device=device),
               torch.as_tensor(batch_offsets, device=device),
               torch.as_tensor(dense[batch], device=device),
               torch.as_tensor(move[batch].astype(np.int64), device=device),
               torch.as_tensor(split[batch].astype(np.int64), device=device))


def class_weights(labels, n_classes, device, power=1.0):
    """(Inverse frequency)**power weights for cross_entropy's `weight=`, for classes that actually occur (an
    absent class, e.g. bc_data.py's larger split bucket, would divide by zero -- its weight is never used since
    its label never appears as a target, so any placeholder is fine). power=1.0 (full inverse frequency) on a
    ~95%/5% split turned out to overshoot badly -- 93% recall but only 18% precision, i.e. it learned to predict
    "split" far too often. power=0.5 (sqrt-softened, a standard fix for exactly this) is the default used below."""
    counts = np.bincount(labels, minlength=n_classes)
    n = len(labels)
    present = counts > 0
    w = np.ones(n_classes, dtype=np.float32)
    w[present] = (n / (present.sum() * counts[present])) ** power
    return torch.as_tensor(w, device=device)


def step(net, batch, split_w=None, move_w=None):
    flat_idx, offsets, dense, move, split = batch
    move_logits, split_logits, _ = net(flat_idx, offsets, dense)  # BC has no return targets: the value head is untrained here
    split_loss = F.cross_entropy(split_logits, split, weight=split_w)
    move_mask = move >= 0
    if move_mask.any():
        move_loss = F.cross_entropy(move_logits[move_mask], move[move_mask], weight=move_w)
        move_acc = (move_logits[move_mask].argmax(-1) == move[move_mask]).float().mean().item()
    else:
        move_loss, move_acc = torch.zeros((), device=move.device), float("nan")
    split_acc = (split_logits.argmax(-1) == split).float().mean().item()
    return split_loss + move_loss, split_loss.item(), move_loss.item() if move_mask.any() else float("nan"), \
        split_acc, move_acc


def evaluate(net, d, order, batch_size, device):
    net.eval()
    tot_loss = tot_split_acc = tot_move_acc = n_move = n = 0.0
    with torch.no_grad():
        for batch in iterate_batches(d, batch_size, order, device):
            loss, _, _, sacc, macc = step(net, batch)
            bs = len(batch[2])
            tot_loss += loss.item() * bs
            tot_split_acc += sacc * bs
            if macc == macc:  # not nan
                tot_move_acc += macc * (batch[3] >= 0).sum().item()
                n_move += (batch[3] >= 0).sum().item()
            n += bs
    net.train()
    return tot_loss / n, tot_split_acc / n, (tot_move_acc / n_move if n_move else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(pathlib.Path(__file__).resolve().parent / "bc" / "demo.npz"))
    ap.add_argument("--out", default=str(pathlib.Path(__file__).resolve().parent / "bc" / "bc.npz"))
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    d = load(args.data)
    n = len(d["dense"])
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_val = max(1, int(n * args.val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    print(f"{n:,} samples ({len(train_idx):,} train / {len(val_idx):,} val), device={args.device}")

    net = torch_policy.PolicyNet().to(args.device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    # class weights for the TRAINING loss only (split is ~95%/5% imbalanced -- unweighted CE mostly just learns
    # to always predict "no split"; evaluate() stays unweighted so val_loss remains comparable across runs)
    split_w = class_weights(d["split"][train_idx], torch_policy.N_SPLIT, args.device, power=0.5)
    move_train = d["move"][train_idx]
    move_w = class_weights(move_train[move_train >= 0], 3, args.device, power=0.5)
    print(f"split class weights: {split_w.tolist()}  move class weights: {move_w.tolist()}")

    best_val = float("inf")
    best_weights = None
    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        order = rng.permutation(train_idx)
        tot_loss = tot_n = 0
        for batch in iterate_batches(d, args.batch_size, order, args.device):
            loss, sl, ml, sacc, macc = step(net, batch, split_w, move_w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(batch[2])
            tot_n += len(batch[2])
        val_loss, val_split_acc, val_move_acc = evaluate(net, d, val_idx, args.batch_size, args.device)
        print(f"epoch {epoch}: train_loss {tot_loss / tot_n:.4f}  val_loss {val_loss:.4f}  "
              f"val_split_acc {val_split_acc:.3%}  val_move_acc {val_move_acc:.3%}  ({time.perf_counter() - t0:.1f}s)")
        if val_loss < best_val:
            best_val = val_loss
            best_weights = torch_policy.export_numpy(net)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **best_weights)
    print(f"saved best-val checkpoint (val_loss {best_val:.4f}) to {out}")


if __name__ == "__main__":
    main()
