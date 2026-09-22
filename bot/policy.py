"""Randomly-initialized numpy policy: the stage-4 scaffold for the RL environment. Consumes encoder.py's sparse
features and produces a masked relative move (forward/left/right; back is never legal) plus an optional split, in
the architecture the plan specifies (numpy gather-sum first layer, ~256/128 widths). Untrained -- this exists to
drive real engine + encode() + forward-pass rollouts for the throughput gate (see tools/rollout.py), not to play
well; stage 5 replaces `random_weights` with trained ones (same `forward`, so nothing else here changes).

Network: h1 = relu(W1[idx].sum(0) + dense @ Wd + b1); h2 = relu(h1 @ W2 + b2);
         move_logits = h2 @ Wm + bm   (3: forward, left, right)
         split_logits = h2 @ Ws + bs  (len(SPLIT_FRACS)+1: index 0 is "don't split")

Legal-move masking here is a simple immediate-neighbour check (kelp, any dragon segment -- the same shortcut
brain.py's own `fallback()` uses), not brain.py's full bitboard trap shield: enough to keep dragons alive for a
throughput test, not to play well or to ship. Split options are similarly masked to what the engine will actually
accept (team not at the unit limit, a valid child length) before sampling, so `decide()`'s recorded choice always
matches what happens -- see `decide()` for the trajectory-recording entry point tools/trajectory.py uses.
"""
import numpy as np

import encoder
import proto

H1, H2 = 256, 128
N_MOVE = 3  # forward, left, right
SPLIT_FRACS = (1 / 3, 1 / 2)  # "a few length-fraction buckets" (plan); split_logits[0] means "don't split"
MOVES = (b"MOVE N\n", b"MOVE E\n", b"MOVE S\n", b"MOVE W\n")
DX, DY = proto.DX, proto.DY


def random_weights(seed, vocab=encoder.VOCAB, dense_size=encoder.DENSE_SIZE):
    rng = np.random.default_rng(seed)

    def layer(fan_in, fan_out):
        return (rng.standard_normal((fan_in, fan_out)) * fan_in ** -0.5).astype(np.float32)

    return {
        "w1": layer(vocab, H1), "wd": layer(dense_size, H1), "b1": np.zeros(H1, np.float32),
        "w2": layer(H1, H2), "b2": np.zeros(H2, np.float32),
        "wm": layer(H2, N_MOVE), "bm": np.zeros(N_MOVE, np.float32),
        "ws": layer(H2, len(SPLIT_FRACS) + 1), "bs": np.zeros(len(SPLIT_FRACS) + 1, np.float32),
    }


def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def _legal(t, w, h):
    """Per absolute direction N,E,S,W: no kelp on the step and no dragon segment (any team) on the landing tile."""
    ed = t.edges
    vt = ed[11].split()
    kelp = (ed[3].split()[3] == b"w", vt[4] == b"w", ed[4].split()[3] == b"w", vt[3] == b"w")
    occ = {(int(q[2]), int(q[3])) for q in t.parts}
    return tuple(not kelp[d] and ((t.hx + DX[d]) % w, (t.hy + DY[d]) % h) not in occ for d in range(4))


class Policy:
    def __init__(self, dragon_id, team, width, height, unit_limit, weights, seed=0):
        self.id, self.team = dragon_id, team
        self.W, self.H, self.limit = width, height, unit_limit
        self.w = weights
        self.rng = np.random.default_rng(seed)  # rollouts sample (not argmax): PPO needs exploration + a log-prob

    @classmethod
    def from_init(cls, init_bytes, weights, seed=0):
        dragon_id, team, w, h, limit = proto.parse_init(init_bytes)
        return cls(dragon_id, team, w, h, limit, weights, seed=seed)

    def forward(self, idx, dense):
        w = self.w
        h1 = w["w1"][idx].sum(0) + np.asarray(dense, dtype=np.float32) @ w["wd"] + w["b1"]
        np.maximum(h1, 0, out=h1)
        h2 = h1 @ w["w2"] + w["b2"]
        np.maximum(h2, 0, out=h2)
        return h2 @ w["wm"] + w["bm"], h2 @ w["ws"] + w["bs"]

    def decide(self, block):
        """Everything a trajectory step needs: the action bytes, the (idx, dense) features it was chosen from,
        and each sampled head's index + log-prob under the distribution actually sampled from (so the log-prob
        matches what PPO's importance ratio needs later). Split options are masked *before* sampling so the
        recorded split_i is always exactly what happens -- unlike a "sample then validate" approach, which could
        record "chose to split" for a turn where a move happened instead because splitting was infeasible.
        move_i/move_logp are None on a turn that splits: the move head was never acted on that turn."""
        t = proto.parse_turn(block)
        idx, dense = encoder.encode(t, self.team, self.id, self.W, self.H, self.limit)
        move_logits, split_logits = self.forward(idx, dense)

        d0 = t.dir if t.dir >= 0 else 0
        legal = _legal(t, self.W, self.H)
        rel_abs = (d0, (d0 + 3) % 4, (d0 + 1) % 4)  # forward, left, right -> absolute N/E/S/W
        options = [i for i in range(3) if legal[rel_abs[i]]]
        if options:
            mp = _softmax(move_logits[options])
            oi = int(self.rng.choice(len(options), p=mp))
            move_i, move_logp, choice = options[oi], float(np.log(mp[oi])), rel_abs[options[oi]]
        else:  # boxed in on all 3 non-back directions: the back, else whatever isn't kelp; no move head sample
            choice = next((d for d in range(4) if legal[d]), d0)
            move_i = move_logp = None

        eligible = [True]  # index 0 ("don't split") is always available
        children = [0]
        for frac in SPLIT_FRACS:
            ok = t.units < self.limit and t.length >= 4
            child = max(2, min(round(t.length * frac), t.length - 2)) if ok else 0
            ok = ok and t.length - child >= 2
            eligible.append(ok)
            children.append(child)
        mask = np.where(eligible, 0.0, -1e30)
        sp = _softmax(split_logits + mask)
        split_i = int(self.rng.choice(len(sp), p=sp))
        split_logp = float(np.log(sp[split_i]))

        if split_i > 0:
            action, move_i, move_logp = b"SPLIT %d\n" % children[split_i], None, None
        else:
            action = MOVES[choice]
        return {"action": action, "idx": idx, "dense": dense, "length": t.length, "round": t.rnd,
                "move_i": move_i, "move_logp": move_logp, "split_i": split_i, "split_logp": split_logp}

    def act(self, block):
        return self.decide(block)["action"]
