"""Sea-dragon bot entry point: a thin turn loop around proto.py (I/O) and brain.py (decisions)."""
import sys

import proto
from brain import Brain


def main():
    brain = None
    while True:
        buf = proto.read_block()
        if not buf or buf.lstrip().startswith(b"ENDGAME"):
            return
        init, block = proto.split_payload(buf)
        if init is not None:
            brain = Brain.from_init(init)
        if brain is None:  # protocol surprise: never leave the turn without an action
            action = b"MOVE N\n"
        else:
            action = brain.act(block)
        # one write per turn: every write costs 2.5M points plus 4,000 per byte
        sys.stdout.write(action.decode() + "ENDTURN\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
