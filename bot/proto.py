"""Lean UNSW Battlecode protocol layer: one read per turn, no helper.py, no enum.

The starter helper costs ~11.8M CPU points to import (three Enum classes) and ~10M per full parse; this
module parses one turn into plain ints/bytes for ~1M (see the Points lab results in the plan).

Turn block (text), one per dragon-turn:
    ROUND n / DIR d / LENGTH l / UNIT_COUNT u / NUM_MSGS k / k message lines
    49 tile lines "x y hasPearl pearlIn" (row-major 7x7 window, tile 24 = head)
    NUM_PARTS m / m lines "team id x y facing isHead"
    8 horizontal-edge rows of 7 tokens, then 7 vertical-edge rows of 8 tokens ('.' empty, 'w' kelp, n portal id)
A dragon's very first payload is prefixed by a 4-line init block (id, team, "MAP w h", unit limit).
"""
import sys

LETTERS = b"NESW"
DX = (0, 1, 0, -1)  # N, E, S, W; y grows downwards
DY = (-1, 0, 1, 0)
DOT7 = b". . . . . . ."
DOT8 = b". . . . . . . ."


def read_block():
    """Reads until the blank-line terminator; returns b'' at EOF."""
    rb = sys.stdin.buffer
    buf = rb.read1(65536)
    while buf and not buf.endswith(b"\n\n"):
        more = rb.read1(65536)
        if not more:
            break
        buf += more
    return buf


def split_payload(buf):
    """First payload of a dragon = init lines + first turn block -> (init or None, block)."""
    k = buf.find(b"ROUND")
    if k > 0:
        return buf[:k], buf[k:]
    return None, buf


def parse_init(text):
    """-> (dragon_id, team_letter_bytes, width, height, unit_limit)"""
    ls = [x for x in text.split(b"\n") if x]
    size = ls[2].split()
    return int(ls[0].split()[1]), ls[1].split()[1], int(size[1]), int(size[2]), int(ls[3].split()[1])


class Turn:
    __slots__ = ("rnd", "dir", "length", "units", "msgs", "hx", "hy", "flags", "cds", "parts", "edges")


def parse_turn(block):
    ls = block.strip().split(b"\n")
    t = Turn()
    t.rnd = int(ls[0].split()[1])
    t.dir = LETTERS.find(ls[1].split()[1])
    t.length = int(ls[2].split()[1])
    t.units = int(ls[3].split()[1])
    n = int(ls[4].split()[1])
    t.msgs = [int(x) for x in ls[5:5 + n]]
    j = 5 + n
    toks = b" ".join(ls[j:j + 49]).split()
    t.hx = int(toks[96])  # tile 24 (the head), token x
    t.hy = int(toks[97])
    t.flags = toks[2::4]  # b"1" where a pearl lies, per window tile
    t.cds = toks[3::4]  # pearl countdown per window tile (-1 = never)
    m = int(ls[j + 49].split()[1])
    t.parts = [x.split() for x in ls[j + 50:j + 50 + m]]
    e = j + 50 + m
    t.edges = ls[e:e + 15]
    return t
