"""Sea-dragon bot entry point: a thin turn loop around proto.py (I/O) and policy.py (decisions).

Loads weights.bin once per dragon-process (each dragon is its own OS process -- the rules: "individual dragons
do not share any program memory with other dragons"), then runs the trained Policy every turn.

Seeded from os.urandom, not a fixed constant: Policy.act() samples stochastically (every validated win-rate
number for this policy -- BC, PPO, evolved -- was measured with that same stochastic sampling; switching to
greedy/argmax at deployment would be an untested behaviour change on top of an already-consequential, one-way
submission). A fixed per-dragon seed would also make play fully predictable turn-for-turn against a scouting
opponent, since battles are publicly viewable.
"""
import os
import sys

import numpy as np

import encoder
import proto
from policy import H1, H2, N_MOVE, SPLIT_FRACS, Policy

# Raw open().read() + np.frombuffer, not np.load: np.load on a zip-based .npz measured ~1.2M CPU points under
# --sandbox (Points lab, see unswbc-engine-and-judge-budget project notes) vs ~0.15M for a flat binary blob --
# a real difference confirmed here (np.load's zip/per-array overhead was enough to blow the whole 100M/turn
# budget on round 0 in a --sandbox test; this fixed it). Shapes/order must exactly match tools/evolve.py's
# ACTOR_KEYS and how bot/weights.bin was produced (flat concatenation, float32, C order) -- see that module.
_SHAPES = [
    ("b1", (H1,)), ("b2", (H2,)), ("bm", (N_MOVE,)), ("bs", (len(SPLIT_FRACS) + 1,)),
    ("w1", (encoder.VOCAB, H1)), ("w2", (H1, H2)), ("wd", (encoder.DENSE_SIZE, H1)),
    ("wm", (H2, N_MOVE)), ("ws", (H2, len(SPLIT_FRACS) + 1)),
]


def _load_weights():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights.bin")
    with open(path, "rb") as f:
        raw = f.read()
    flat = np.frombuffer(raw, dtype=np.float32)
    out, i = {}, 0
    for name, shape in _SHAPES:
        n = int(np.prod(shape))
        out[name] = flat[i:i + n].reshape(shape)
        i += n
    return out


WEIGHTS = _load_weights()
_SEED = int.from_bytes(os.urandom(4), "little")


def main():
    policy = None
    while True:
        buf = proto.read_block()
        if not buf or buf.lstrip().startswith(b"ENDGAME"):
            return
        init, block = proto.split_payload(buf)
        if init is not None:
            policy = Policy.from_init(init, WEIGHTS, seed=_SEED)
        if policy is None:  # protocol surprise: never leave the turn without an action
            action = b"MOVE N\n"
        else:
            action = policy.act(block)
        # one write per turn: every write costs 2.5M points plus 4,000 per byte
        sys.stdout.write(action.decode() + "ENDTURN\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
