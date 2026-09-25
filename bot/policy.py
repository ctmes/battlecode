"""Numpy policy for the RL environment. Consumes encoder.py's sparse features and produces a masked, sprint-aware
relative move (forward/left/right, 1-3 tiles; back is never legal), an optional split, and a deterministic sonar
broadcast, in the architecture the plan specifies (numpy gather-sum first layer, ~256/128 widths).

Network: h1 = relu(W1[idx].sum(0) + dense @ Wd + b1); h2 = relu(h1 @ W2 + b2);
         move_logits = h2 @ Wm + bm   (3 * len(SPRINT_LENGTHS): see choose())
         split_logits = h2 @ Ws + bs  (len(SPLIT_FRACS)+1: index 0 is "don't split")

Legal-move masking here is a simple immediate-neighbour check (kelp, any dragon segment -- the same shortcut
brain.py's own `fallback()` uses), not brain.py's full bitboard trap shield: enough to keep dragons alive for a
throughput test, not to play well or to ship. Split options are similarly masked to what the engine will actually
accept (team not at the unit limit, a valid child length) before sampling, so `decide()`'s recorded choice always
matches what happens -- see `decide()` for the trajectory-recording entry point tools/trajectory.py uses.

Sprinting (added after a top-of-leaderboard opponent's replay stats showed heavy sonar/sprint use we had zero of,
see the project's stage-6 notes): `MOVE NNN` moves 3 tiles in one turn, costing `steps - 1` body segments (see the
rules' "Sprinting" section) -- a genuinely different action from a plain MOVE, not a training nuance, so it needs
its own action-space slot, not just more training. The move head is extended from 3 options (forward/left/right,
1 tile) to `3 * len(SPRINT_LENGTHS)`, laid out `length_idx * 3 + rel_dir_idx` so indices 0-2 are *exactly* the old
3-option head (length=1) -- this ordering is deliberate: it lets an old checkpoint's wm/bm columns 0-2 warm-start
the new head's columns 0-2 unchanged, with only the new sprint columns needing fresh training (see
tools/migrate_checkpoint.py). Only the first step of a sprint is legality-checked here (matching this module's
existing "immediate neighbour only" philosophy, not brain.py's full trap shield) -- steps 2-3 are left for the
policy to learn to avoid, the same way it already has to learn not to run into a dead end 2 moves out.

Sonar (same motivation): broadcasting the nearest visible enemy's position/size/facing is deterministic and
already proven out in brain.py (`proto.pack_enemy_sonar`/`unpack_sonar`, "no addressing" wire format -- see that
module's docstring) -- ported here as `sonar_report()` rather than re-learned from scratch, since *whether this
particular broadcast is useful* isn't really in question (brain.py already validated the mechanism), only *what
a receiver does with it*, which is what encoder.py's new sonar-decode features let the network actually learn.
"""
import numpy as np

import encoder
import proto

H1, H2 = 256, 128
SPRINT_LENGTHS = (1, 2, 3)  # tiles moved in one turn; cost is `length - 1` body segments (see module docstring)
N_MOVE = 3 * len(SPRINT_LENGTHS)  # (length_idx * 3 + rel_dir_idx); indices 0-2 = length 1 = the pre-sprint head
SPLIT_FRACS = (1 / 3, 1 / 2)  # "a few length-fraction buckets" (plan); split_logits[0] means "don't split"
MOVES = (b"MOVE N\n", b"MOVE E\n", b"MOVE S\n", b"MOVE W\n")
LETTERS = proto.LETTERS  # b"NESW", indexed by absolute direction 0-3
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


def choose(move_logits, split_logits, t, rng, width, height, unit_limit):
    """The masking/sampling logic shared by Policy.decide() and tools/ppo_policy.py's PPOPolicy.decide() (which
    needs a 3rd, value output from forward() -- factored out here so the two don't duplicate this, the trickiest
    part of the module, see the class docstring). Returns (action, move_i, move_logp, split_i, split_logp); split
    options are masked *before* sampling so split_i always matches what happens (an infeasible sampled split would
    otherwise silently fall back to a move without the recorded action reflecting that). move_i/move_logp are None
    on a turn that splits: the move head was never acted on that turn.

    move_i indexes the N_MOVE-way head as `length_idx * 3 + rel_dir_idx` (see module docstring). A sprint option
    is eligible only if its first step is legal (the existing immediate-neighbour check, unchanged) AND the
    dragon can afford it (`length - (steps - 1) >= 2`, the same "stay at least 2 long" floor split uses) --
    steps 2+ are not legality-checked, see module docstring."""
    d0 = t.dir if t.dir >= 0 else 0
    legal = _legal(t, width, height)
    rel_abs = (d0, (d0 + 3) % 4, (d0 + 1) % 4)  # forward, left, right -> absolute N/E/S/W
    options = [li * 3 + ri for li, steps in enumerate(SPRINT_LENGTHS) if t.length - (steps - 1) >= 2
               for ri in range(3) if legal[rel_abs[ri]]]
    if options:
        mp = _softmax(move_logits[options])
        oi = int(rng.choice(len(options), p=mp))
        move_i, move_logp = options[oi], float(np.log(mp[oi]))
        li, ri = divmod(move_i, 3)
        direction, steps = rel_abs[ri], SPRINT_LENGTHS[li]
        move_action = b"MOVE " + LETTERS[direction:direction + 1] * steps + b"\n"
    else:  # boxed in on all 3 non-back directions: the back, else whatever isn't kelp; no move head sample
        direction = next((d for d in range(4) if legal[d]), d0)
        move_action = MOVES[direction]  # length 1, always affordable regardless of the sprint-cost floor above
        move_i = move_logp = None

    eligible = [True]  # index 0 ("don't split") is always available
    children = [0]
    for frac in SPLIT_FRACS:
        ok = t.units < unit_limit and t.length >= 4
        child = max(2, min(round(t.length * frac), t.length - 2)) if ok else 0
        ok = ok and t.length - child >= 2
        eligible.append(ok)
        children.append(child)
    mask = np.where(eligible, 0.0, -1e30)
    sp = _softmax(split_logits + mask)
    split_i = int(rng.choice(len(sp), p=sp))
    split_logp = float(np.log(sp[split_i]))

    if split_i > 0:
        return b"SPLIT %d\n" % children[split_i], None, None, split_i, split_logp
    return move_action, move_i, move_logp, split_i, split_logp


def sonar_report(t, team, width, height):
    """Deterministic (no learned parameters, see module docstring): the nearest visible enemy head's absolute
    position, its visible segment count as a coarse size hint, and its own reported facing, packed via
    proto.pack_enemy_sonar -- ported from brain.py's already-tested sonar broadcast, unchanged (same
    wrapped-Manhattan "nearest" tie-break, same visible-segment-count size hint). Returns None if no enemy head
    is visible this turn (nothing to report)."""
    segs = {}
    heads = []  # (pid, ex, ey, facing_letter)
    for q in t.parts:
        pid = q[1]
        segs[pid] = segs.get(pid, 0) + 1
        if q[0] != team and q[5] == b"1":
            heads.append((pid, int(q[2]), int(q[3]), q[4]))
    best, best_dist = None, None
    for pid, ex, ey, facing_letter in heads:
        dist = (min((ex - t.hx) % width, (t.hx - ex) % width)
                + min((ey - t.hy) % height, (t.hy - ey) % height))
        if best_dist is None or dist < best_dist:
            facing = LETTERS.find(facing_letter)
            best, best_dist = (ex, ey, segs.get(pid, 1), facing if facing >= 0 else 0), dist
    return proto.pack_enemy_sonar(*best) if best is not None else None


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
        matches what PPO's importance ratio needs later). See `choose()` for the masking/sampling rules. The
        sonar line (see `sonar_report()`, deterministic, not sampled) is appended to `action` but not itself
        recorded as a trajectory field -- there is nothing to learn on the sending side, see module docstring."""
        t = proto.parse_turn(block)
        idx, dense = encoder.encode(t, self.team, self.id, self.W, self.H, self.limit)
        move_logits, split_logits = self.forward(idx, dense)
        action, move_i, move_logp, split_i, split_logp = choose(
            move_logits, split_logits, t, self.rng, self.W, self.H, self.limit)
        sonar_msg = sonar_report(t, self.team, self.W, self.H)
        if sonar_msg is not None:
            action = action + b"SONAR %d\n" % sonar_msg
        return {"action": action, "idx": idx, "dense": dense, "length": t.length, "round": t.rnd,
                "move_i": move_i, "move_logp": move_logp, "split_i": split_i, "split_logp": split_logp}

    def act(self, block):
        return self.decide(block)["action"]
