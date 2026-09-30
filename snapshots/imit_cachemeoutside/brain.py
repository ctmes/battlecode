"""Imitation sparring partner (30 Sep): plays like a ladder team, from a network trained on that team's replays
(tools/imit_data.py, tools/imit_train.py). Local-only -- it loads numpy and weights.npz from its own folder, never
ships. The same frozen-snapshot interface as every other bench/tune.py opponent (Brain.from_init, Brain.act).

Each turn: parse the block, blank tile countdowns and sonar messages (a replay cannot give them, so training never
saw them), encode with this folder's encoder.py, run the network, mask what is illegal or instantly fatal in view
(kelp or a body on a step, a sprint the dragon cannot pay for, a split the engine would refuse -- except the "die"
class, which is the top teams' deliberate SPLIT 1), and sample: split first (none / child 2 / turnaround / half /
die), then the move (forward / left / right x 1-3 steps, a sprint going straight). Deterministic per (seed, dragon,
round), so a (map, seed) game repeats exactly.
"""
import importlib.util
import pathlib
import random

import numpy as np

import proto

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(f"_imit_encoder_{abs(hash(str(HERE))) % 10 ** 8}", HERE / "encoder.py")
encoder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(encoder)

DEFAULTS = {"temp": 1.0, "temp_move": 0.25, "seed": 0}  # temp: the split head; temp_move: the move head
REL = (0, 3, 1)  # forward, left, right
DX, DY, LETTERS = proto.DX, proto.DY, proto.LETTERS
_W = {}


def weights():
    if not _W:
        d = np.load(HERE / "weights.npz")
        _W.update({k: d[k] for k in d.files})
    return _W


class Brain:
    strict = False
    debug = False

    def __init__(self, dragon_id, team, width, height, unit_limit, params=None):
        self.id, self.team, self.W, self.H, self.limit = dragon_id, team, width, height, unit_limit
        self.p = {**DEFAULTS, **(params or {})}
        self.w = weights()

    @classmethod
    def from_init(cls, init_bytes, params=None):
        return cls(*proto.parse_init(init_bytes), params=params)

    def act(self, block):
        try:
            return self.decide(proto.parse_turn(block))
        except Exception:  # noqa: BLE001 - a crash would kill the dragon
            if self.strict:
                raise
            return b"MOVE N\n"

    def blocked(self, t, d, steps, occ):
        """True if any of `steps` straight steps in absolute direction d crosses kelp or lands on a body in view."""
        x = y = 0
        ed = t.edges
        for _ in range(steps):
            # the edge crossed: N = our tile's north edge, S = the next tile's north edge, W / E likewise west edges
            if d in (0, 2):
                ex, ey = x, y if d == 0 else y + 1
                tok = ed[ey + 3].split()[ex + 3] if -3 <= ey <= 4 and -3 <= ex <= 3 else b"."
            else:
                ex, ey = x if d == 3 else x + 1, y
                tok = ed[8 + ey + 3].split()[ex + 3] if -3 <= ey <= 3 and -3 <= ex <= 4 else b"."
            if tok == b"w":
                return True
            x, y = x + DX[d], y + DY[d]
            if (x, y) in occ:
                return True
        return False

    def decide(self, t):
        t.cds = [b"-1"] * 49
        t.msgs = []
        w, rng = self.w, random.Random(f"{self.p['seed']}/{self.id}/{t.rnd}")
        idx, dense = encoder.encode(t, self.team, self.id, self.W, self.H, self.limit)
        h1 = np.maximum(w["w1"][idx].sum(0) + np.asarray(dense, dtype=np.float32) @ w["wd"] + w["b1"], 0)
        h2 = np.maximum(h1 @ w["w2"] + w["b2"], 0)
        temp = self.p["temp"]
        n, units = t.length, t.units
        room = units < self.limit
        split_ok = [True, room and n >= 4, room and n >= 5, room and n >= 4, True]
        sl = (h2 @ w["ws"] + w["bs"]) / temp  # full temperature: splits and deaths at the team's own rate
        choice = self.sample(sl, split_ok, rng)
        if choice == 1:
            return b"SPLIT 2\n"
        if choice == 2:
            return b"SPLIT %d\n" % (n - 2)
        if choice == 3:
            return b"SPLIT %d\n" % (n // 2)
        if choice == 4:
            return b"SPLIT 1\n"  # dying on purpose, as the top teams feed their king
        occ = set()
        for q in t.parts:
            dx = (int(q[2]) - t.hx + 3) % self.W - 3
            dy = (int(q[3]) - t.hy + 3) % self.H - 3
            occ.add((dx, dy))
        f = t.dir if t.dir >= 0 else 0
        move_ok = []
        for k in range(9):
            steps, rel = k // 3 + 1, REL[k % 3]
            move_ok.append(n > steps - 1 and (steps == 1 or n > steps) and not self.blocked(t, (f + rel) % 4, steps, occ))
        if not any(move_ok):
            move_ok = [True] * 3 + [False] * 6
        k = self.sample((h2 @ w["wm"] + w["bm"]) / self.p["temp_move"], move_ok, rng)  # sharper: sampled noise starves it
        d = (f + REL[k % 3]) % 4
        return b"MOVE " + LETTERS[d:d + 1] * (k // 3 + 1) + b"\n"

    @staticmethod
    def sample(logits, ok, rng):
        z = np.where(ok, logits, -1e9)
        z = np.exp(z - z.max())
        p = z / z.sum()
        r, acc = rng.random(), 0.0
        for i, pi in enumerate(p):
            acc += pi
            if r < acc:
                return i
        return int(np.argmax(p))
