"""Smoke test for bot/policy.py: real engine games, self-play, untrained random weights. Checks it never crashes
and always returns a well-formed action -- not that it plays well (it won't; see the module docstring).

Run:  .venv\\Scripts\\python.exe tools\\test_policy.py
"""
import pathlib
import re
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bot"))
sys.path.insert(0, str(ROOT / "tools"))
import encoder  # noqa: E402
import proto  # noqa: E402
from policy import Policy, SPRINT_LENGTHS, choose, random_weights, sonar_report  # noqa: E402
from test_encoder import HX, HY, W, H, render_block  # noqa: E402
from unswbc.engine import EngineModule  # noqa: E402

MAPS = sorted((ROOT / "maps").glob("*.map"))
MOVE_RE = re.compile(rb"^MOVE ([NESW])\1*$")  # one letter, repeated (a sprint is the same direction every step)
MAX_STEPS = max(SPRINT_LENGTHS)


def play(engine, data, weights):
    dragons = {}
    errors = []
    splits = [0]
    sonars = [0]

    def bot_spawn(did, init):
        dragons[did] = Policy.from_init(init, weights, seed=did)

    def bot_reply(did, block):
        action = dragons[did].act(block)
        lines = action.split(b"\n")[:-1]  # drop the trailing "" from the final \n
        head, _, rest = lines[0].partition(b" ")
        if head == b"SPLIT":
            splits[0] += 1
            k = int(rest)
            t = proto.parse_turn(block)
            assert 2 <= k <= t.length - 2, f"split length {k} out of range for length {t.length}"
        else:
            m = MOVE_RE.match(lines[0])
            if not m or len(lines[0]) - 5 > MAX_STEPS:  # "MOVE " is 5 bytes; the rest is the repeated letter
                errors.append(f"dragon {did}: malformed move {lines[0]!r}")
        if len(lines) > 1:
            sonars[0] += 1
            head2, _, rest2 = lines[1].partition(b" ")
            if head2 != b"SONAR" or not (0 <= int(rest2) <= 0xFFFFFFFF):
                errors.append(f"dragon {did}: malformed sonar line {lines[1]!r}")
            if len(lines) > 2:
                errors.append(f"dragon {did}: unexpected extra line(s) {lines[2:]!r}")
        return action

    def on_notice(line):
        errors.append(f"engine notice: {line}")

    res = engine.run(data, bot_reply, None, bot_spawn, on_notice, 0)
    return res, errors, splits[0], sonars[0]


def test_no_crashes_and_valid_actions():
    engine = EngineModule()
    weights = random_weights(0)
    total_splits = total_sonars = 0
    for m in MAPS:
        res, errors, splits, sonars = play(engine, m.read_bytes(), weights)
        assert not errors, f"{m.name}: {errors[:3]}"
        total_splits += splits
        total_sonars += sonars
    print(f"ok: {len(MAPS)} bundled maps, no crashes, every action well-formed, "
          f"{total_splits} total splits, {total_sonars} total sonar pings")


def test_vocab_matches_encoder():
    w = random_weights(1)
    assert w["w1"].shape == (encoder.VOCAB, 256)
    assert w["wd"].shape == (encoder.DENSE_SIZE, 256)
    print("ok: random_weights defaults match encoder.VOCAB / DENSE_SIZE")


def _turn(dir_=0, length=5, units=1, parts=(), edges=None, msgs=(), rnd=7):
    return proto.parse_turn(render_block(dir_, set(), {}, edges or {}, list(parts), rnd=rnd,
                                          length=length, units=units, msgs=list(msgs)))


def _forced_move(move_logits, t, unit_limit=8):
    """split_logits forced to "never split" (index 0 given an overwhelming logit) so the returned action is
    always the move choice being tested, not a split."""
    split_logits = np.array([1000.0, -1000.0, -1000.0])
    rng = np.random.default_rng(0)
    return choose(np.asarray(move_logits, dtype=np.float64), split_logits, t, rng, W, H, unit_limit)


def test_sprint_unaffordable_lengths_excluded():
    # length=2: the cost floor (length - (steps-1) >= 2) affords only steps=1 -- length index 0.
    t = _turn(length=2)
    move_logits = np.full(9, -1000.0)
    move_logits[3] = 1000.0  # length index 1 (2 steps), forward: should be excluded despite the huge logit
    move_logits[0] = 500.0  # length index 0 (1 step), forward: the only real option
    action, move_i, _, split_i, _ = _forced_move(move_logits, t)
    assert split_i == 0 and move_i == 0 and action == b"MOVE N\n", (action, move_i, split_i)
    print("ok: an unaffordable sprint length is excluded from the move head regardless of its logit")


def test_sprint_length_and_direction_selected_correctly():
    # length=6 affords every sprint length (worst case costs 2 segments, 6-2=4 >= 2). No kelp/parts: all 4
    # absolute directions are legal, so every one of the 9 move options is genuinely eligible here.
    t = _turn(length=6)
    move_logits = np.full(9, -1000.0)
    move_logits[3 * 2 + 1] = 1000.0  # length index 2 (3 steps), rel_dir index 1 (left)
    action, move_i, _, split_i, _ = _forced_move(move_logits, t)
    assert move_i == 7 and split_i == 0, (move_i, split_i)
    assert action == b"MOVE WWW\n", action  # dir_=0 (N): left = W; 3 steps west
    print("ok: a 3-step sprint in a non-forward relative direction produces the right multi-letter MOVE")


def test_sprint_first_step_kelp_blocks_that_direction_at_every_length():
    # Kelp on the head's own north edge: every sprint length going forward (absolute N, since dir_=0) must be
    # excluded, since only the FIRST step is legality-checked (see policy.py's module docstring) but that alone
    # is already fatal here -- this is exactly the case that check exists for.
    t = _turn(length=6, edges={(0, 0, "h"): "w"})
    move_logits = np.full(9, -1000.0)
    for li in range(3):
        move_logits[li * 3 + 0] = 1000.0  # forward at every sprint length: must all be masked out
    move_logits[1] = 500.0  # length index 0, rel_dir index 1 (left): the fallback winner
    action, move_i, _, split_i, _ = _forced_move(move_logits, t)
    assert move_i == 1 and split_i == 0, (move_i, split_i)
    assert action == b"MOVE W\n", action  # dir_=0 (N): left = W
    print("ok: kelp on the first step blocks that direction at every sprint length, not just length 1")


def test_move_i_0_to_2_matches_the_pre_sprint_single_step_head():
    # Warm-start compatibility (see policy.py's module docstring): indices 0-2 must behave exactly like the old
    # 3-option head -- a single step, no more, regardless of how long the dragon is.
    t = _turn(length=20)
    for i, expected in enumerate((b"MOVE N\n", b"MOVE W\n", b"MOVE E\n")):  # forward, left, right at dir_=0
        move_logits = np.full(9, -1000.0)
        move_logits[i] = 1000.0
        action, move_i, _, split_i, _ = _forced_move(move_logits, t)
        assert move_i == i and action == expected, (i, action)
    print("ok: move indices 0-2 reproduce the pre-sprint head's single-step actions exactly")


def test_sonar_report_picks_nearest_enemy_and_packs_correctly():
    parts = [
        ("A", 1, 2, 0, "E", 1),    # enemy head, offset (2, 0) from the head -> wrapped distance 2
        ("A", 2, -3, -3, "S", 1),  # a second, farther enemy head -> wrapped distance 6
    ]
    t = _turn(length=5, parts=parts)
    msg = sonar_report(t, b"B", W, H)
    assert msg is not None
    kind, x, y, size, facing = proto.unpack_sonar(msg)
    assert kind == proto.SONAR_ENEMY
    assert (x, y) == ((HX + 2) % W, HY), (x, y)
    assert facing == proto.LETTERS.find(b"E"), facing
    print("ok: sonar_report picks the nearer of two visible enemies and packs its position/facing correctly")


def test_sonar_report_ignores_own_team_and_non_head_parts():
    parts = [("B", 1, 1, 0, "N", 0), ("A", 2, -1, 0, "N", 0)]  # own body segment; enemy BODY (not a head)
    t = _turn(length=5, parts=parts)
    assert sonar_report(t, b"B", W, H) is None
    print("ok: sonar_report reports nothing when no *enemy head* is visible (own parts and enemy bodies don't count)")


if __name__ == "__main__":
    test_vocab_matches_encoder()
    test_no_crashes_and_valid_actions()
    test_sprint_unaffordable_lengths_excluded()
    test_sprint_length_and_direction_selected_correctly()
    test_sprint_first_step_kelp_blocks_that_direction_at_every_length()
    test_move_i_0_to_2_matches_the_pre_sprint_single_step_head()
    test_sonar_report_picks_nearest_enemy_and_packs_correctly()
    test_sonar_report_ignores_own_team_and_non_head_parts()
