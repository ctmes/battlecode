"""Checks bot/brain.py's Brain.fallback() -- the emergency, learned-state-independent path taken when decide()
raises. Unlike decide()'s own legality check (which has its own PORTAL status specifically to avoid this), fallback()
re-implements legality from scratch and used to miss the portal case; see its docstring.

Run:  .venv\\Scripts\\python.exe tools\\test_brain.py
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bot"))
sys.path.insert(0, str(ROOT / "tools"))
from brain import Brain, MOVES  # noqa: E402
from test_encoder import H, W, render_block  # noqa: E402


def test_fallback_portal_edge_ignores_irrelevant_landing_tile_occupancy():
    # East edge (offset (1, 0)'s west edge, i.e. the head's own east edge) is a portal: token "7", any string
    # that's neither '.' nor 'w'. A portal step does not land on the physically-adjacent tile east of the head
    # (it exits the portal's paired edge instead, see brain.py's module docstring) -- an unrelated body sitting
    # on that irrelevant tile must not rule the direction out.
    edges = {(1, 0, "v"): "7", (0, 0, "h"): "w"}  # kelp to the north forces fallback() past its first choice
    parts = [("B", 2, 1, 0, "N", 0)]  # unrelated body on the (irrelevant) physically-adjacent tile east
    block = render_block(0, set(), {}, edges, parts, length=6)
    b = Brain(1, b"A", W, H, 8)
    action = b.fallback(block)
    assert action == MOVES[1], action  # MOVES[1] = b"MOVE E\n"
    print("ok: fallback() takes a portal move even when the physically-adjacent landing tile is occupied")


def test_fallback_kelp_still_blocks_and_normal_occupancy_still_blocks():
    # North kelp forces fallback() past its first (heading) choice in both cases below, so reaching South (still
    # open) actually demonstrates East was inspected and rejected -- not just skipped by ordering luck.
    edges = {(0, 0, "h"): "w", (1, 0, "v"): "w"}  # kelp both north and east -- East must still be excluded outright
    parts = [("B", 2, 1, 0, "N", 0)]
    block = render_block(0, set(), {}, edges, parts, length=6)
    b = Brain(1, b"A", W, H, 8)
    assert b.fallback(block) == MOVES[2], b.fallback(block)  # MOVES[2] = b"MOVE S\n"

    edges2 = {(0, 0, "h"): "w"}  # north kelp only; East is a plain edge with the same body occupying its tile
    block2 = render_block(0, set(), {}, edges2, parts, length=6)
    b2 = Brain(1, b"A", W, H, 8)
    assert b2.fallback(block2) == MOVES[2], b2.fallback(block2)
    print("ok: kelp still blocks outright, and a normal edge's occupied landing tile is still excluded")


if __name__ == "__main__":
    test_fallback_portal_edge_ignores_irrelevant_landing_tile_occupancy()
    test_fallback_kelp_still_blocks_and_normal_occupancy_still_blocks()
