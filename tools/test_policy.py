"""Smoke test for bot/policy.py: real engine games, self-play, untrained random weights. Checks it never crashes
and always returns a well-formed action -- not that it plays well (it won't; see the module docstring).

Run:  .venv\\Scripts\\python.exe tools\\test_policy.py
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bot"))
import encoder  # noqa: E402
import proto  # noqa: E402
from policy import Policy, random_weights  # noqa: E402
from unswbc.engine import EngineModule  # noqa: E402

MAPS = sorted((ROOT / "maps").glob("*.map"))
VALID_ACTIONS = {b"MOVE N", b"MOVE E", b"MOVE S", b"MOVE W"}


def play(engine, data, weights):
    dragons = {}
    errors = []
    splits = [0]

    def bot_spawn(did, init):
        dragons[did] = Policy.from_init(init, weights, seed=did)

    def bot_reply(did, block):
        action = dragons[did].act(block)
        head, _, rest = action.partition(b" ")
        if head == b"SPLIT":
            splits[0] += 1
            k = int(rest)
            t = proto.parse_turn(block)
            assert 2 <= k <= t.length - 2, f"split length {k} out of range for length {t.length}"
        elif action.rstrip(b"\n") not in VALID_ACTIONS:
            errors.append(f"dragon {did}: malformed action {action!r}")
        return action

    def on_notice(line):
        errors.append(f"engine notice: {line}")

    res = engine.run(data, bot_reply, None, bot_spawn, on_notice, 0)
    return res, errors, splits[0]


def test_no_crashes_and_valid_actions():
    engine = EngineModule()
    weights = random_weights(0)
    total_splits = 0
    for m in MAPS:
        res, errors, splits = play(engine, m.read_bytes(), weights)
        assert not errors, f"{m.name}: {errors[:3]}"
        total_splits += splits
    print(f"ok: {len(MAPS)} bundled maps, no crashes, every action well-formed, {total_splits} total splits")


def test_vocab_matches_encoder():
    w = random_weights(1)
    assert w["w1"].shape == (encoder.VOCAB, 256)
    assert w["wd"].shape == (encoder.DENSE_SIZE, 256)
    print("ok: random_weights defaults match encoder.VOCAB / DENSE_SIZE")


if __name__ == "__main__":
    test_vocab_matches_encoder()
    test_no_crashes_and_valid_actions()
