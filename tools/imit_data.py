"""Imitation data (30 Sep): turns a ladder team's replays into (what one of its dragons saw, what it did) samples,
for training a bot that plays like that team (tools/imit_train.py) as a sparring partner the bench and the tuner
can use -- every opponent they had so far was built from our own code, and mh9's tune overfit to it.

For every turn of every dragon of the team, the text block the engine would have sent it is rebuilt from the replay
(tools/scout.py tracks every body exactly) and run through the real proto.parse_turn and bot/encoder.py, so training
sees exactly what the deployed imitator (snapshots/imit_*/brain.py) sees. Two things a replay cannot give are
blanked on both sides: tile countdowns (-1) and sonar messages (none).

Labels (see LABELS): the move -- first step relative to the heading (forward / left / right) x steps (1-3; a
sprint's later turns are not modelled, the imitator sprints straight) -- or the split: child 2, a turnaround
(child = length - 2), any other child ("half"), or dying on purpose (an illegal SPLIT such as the top teams'
SPLIT 1, or no action at all). A move back onto the neck is dropped (suicide by another name, rare).

    .venv\\Scripts\\python.exe tools\\imit_data.py replays\\imit\\cheji_1001 --out tools\\imit\\cheji.npz [--workers 8]
"""
import argparse
import concurrent.futures
import csv
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bot"))
sys.path.insert(0, str(ROOT / "tools"))
import encoder  # noqa: E402
import proto  # noqa: E402
import scout  # noqa: E402
from forage import LADDER_NAMES  # noqa: E402
from mapfile import NORTH, WEST  # noqa: E402

LETTERS = "NESW"
REL = (0, 3, 1)  # heading-relative first step: forward, left, right (2 = back is dropped)
N_MOVE = 9  # (steps - 1) * 3 + REL.index(rel)
SPLIT_NONE, SPLIT_TWO, SPLIT_REAR, SPLIT_HALF, SPLIT_DIE = range(5)
N_SPLIT = 5


def edge_tokens(gm):
    """(x, y, 'h'|'v') -> token for every kelp ('w') or portal (its id) edge; h = the tile's north edge."""
    out = {}
    for (x, y, d) in gm.kelp:
        out[(x, y, "h" if d == NORTH else "v")] = "w"
    for (x, y, d), pid in gm.portals.items():
        out[(x, y, "h" if d == NORTH else "v")] = str(pid % 10)
    return out


def block(gm, edges, bodies, team_of, pearls, me, rnd, facing, units):
    """The engine's turn block for dragon `me` (protocol 2, countdowns blanked, no messages)."""
    w, h = gm.w, gm.h
    hx, hy = bodies[me][0]
    lines = [f"ROUND {rnd}", f"DIR {LETTERS[facing]}", f"LENGTH {len(bodies[me])}", f"UNIT_COUNT {units}",
             "NUM_MSGS 0"]
    for r in range(7):
        for c in range(7):
            x, y = (hx + c - 3) % w, (hy + r - 3) % h
            lines.append(f"{x} {y} {1 if (x, y) in pearls else 0} -1")
    parts = []
    window = {((hx + c - 3) % w, (hy + r - 3) % h) for r in range(7) for c in range(7)}
    for i, b in bodies.items():
        for k, cell in enumerate(b):
            if cell in window:
                parts.append(f"{team_of[i]} {i} {cell[0]} {cell[1]} N {1 if k == 0 else 0}")
    lines.append(f"NUM_PARTS {len(parts)}")
    lines += parts
    for r in range(8):
        lines.append(" ".join(edges.get(((hx + c - 3) % w, (hy + r - 3) % h, "h"), ".") for c in range(7)))
    for r in range(7):
        lines.append(" ".join(edges.get(((hx + c - 3) % w, (hy + r - 3) % h, "v"), ".") for c in range(8)))
    return ("\n".join(lines) + "\n\n").encode()


def facing_of(gm, body):
    """The direction from the neck to the head (through a portal too); 0 if it cannot be told."""
    if len(body) < 2:
        return 0
    for d in range(4):
        if gm.step(body[1][0], body[1][1], d) == body[0]:
            return d
    return 0


def one(args):
    path, seat, map_text = args
    data = pathlib.Path(path).read_bytes()
    st, pearls, pend = {}, set(), {}
    out_idx, out_dense, out_move, out_split = [], [], [], []

    def flush(i, split_label, move_label):
        idx, dense = pend.pop(i)
        out_idx.append(np.array(idx, dtype=np.int32))
        out_dense.append(dense)
        out_split.append(split_label)
        out_move.append(move_label)

    def hook(kind, *a):
        if kind == "init":
            st.update(a[0])
            st["edges"] = edge_tokens(st["map"])
        elif kind == "pearl_on":
            pearls.add(tuple(a[1]))
        elif kind == "eat":
            pearls.discard(tuple(a[2]))
        elif kind == "turn":
            rnd, cur = a
            if st["team_of"].get(cur) != seat or cur not in st["bodies"]:
                return
            gm, bodies = st["map"], st["bodies"]
            f = facing_of(gm, bodies[cur])
            units = sum(1 for i in bodies if st["team_of"][i] == seat)
            t = proto.parse_turn(block(gm, st["edges"], bodies, st["team_of"], pearls, cur, rnd, f, units))
            idx, dense = encoder.encode(t, seat.encode(), cur, gm.w, gm.h, 64)
            pend[cur] = (idx, dense)
            st.setdefault("face", {})[cur] = f
        elif kind == "move":
            rnd, cur, dirs = a
            if cur not in pend:
                return
            rel = (dirs[0] - st["face"][cur]) % 4
            if rel == 2:
                pend.pop(cur)  # back onto the neck: dropped
                return
            flush(cur, SPLIT_NONE, (min(len(dirs), 3) - 1) * 3 + REL.index(rel))
        elif kind == "split_req":
            rnd, cur, size = a
            if cur not in pend:
                return
            n = len(st["bodies"][cur])
            if size < 2 or n - size < 2:
                lab = SPLIT_DIE
            elif size == 2:
                lab = SPLIT_TWO
            elif size == n - 2:
                lab = SPLIT_REAR
            else:
                lab = SPLIT_HALF
            flush(cur, lab, -1)
        elif kind == "death":
            rnd, dead, reason, killer, _ = a
            if dead in pend and reason == "no action":
                flush(dead, SPLIT_DIE, -1)  # no action at all: dying on purpose (or out of time)
            pend.pop(dead, None)

    scout.extract(data, map_text, hook)
    return out_idx, out_dense, out_move, out_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="a fetch_replays --out folder (index.csv's seat = the team imitated)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="at most this many replays (0 = all)")
    args = ap.parse_args()
    folder = ROOT / args.folder if not pathlib.Path(args.folder).is_absolute() else pathlib.Path(args.folder)
    rows = [r for r in csv.DictReader(open(folder / "index.csv", encoding="utf-8"))
            if (folder / f"M{r['match']}.replay").exists() and r["map"] in LADDER_NAMES]
    if args.limit:
        rows = rows[:args.limit]
    jobs = [(str(folder / f"M{r['match']}.replay"), r["seat"],
             (ROOT / "maps" / "ladder" / f"{LADDER_NAMES[r['map']]}.map").read_text()) for r in rows]
    idx, dense, move, split = [], [], [], []
    with concurrent.futures.ProcessPoolExecutor(args.workers) as ex:
        for i, (a, b, c, d) in enumerate(ex.map(one, jobs)):
            idx += a
            dense += b
            move += c
            split += d
    lens = np.array([len(x) for x in idx], dtype=np.int64)
    flat = np.concatenate(idx) if idx else np.zeros(0, np.int32)
    out = ROOT / args.out if not pathlib.Path(args.out).is_absolute() else pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, idx=flat, offsets=np.concatenate([[0], np.cumsum(lens)[:-1]]), lens=lens,
                        dense=np.array(dense, dtype=np.float32), move=np.array(move, dtype=np.int64),
                        split=np.array(split, dtype=np.int64))
    sp = np.bincount(np.array(split), minlength=N_SPLIT)
    mv = np.bincount(np.array([m for m in move if m >= 0]), minlength=N_MOVE)
    print(f"{len(rows)} replays -> {len(move)} samples -> {out}\n  split labels (none/two/rear/half/die): {sp.tolist()}"
          f"\n  move labels (fwd/left/right x 1-3 steps): {mv.tolist()}")


if __name__ == "__main__":
    main()
