"""Sparring partners for tools/arena.py: deliberately simple bots, roughly weakest first.

A player is what the arena drives: one instance per team, one call per dragon-turn.
    spawn(did, init)    a dragon appeared (init block: ID / TEAM / MAP w h / UNIT_LIMIT)
    reply(did, block)   -> action bytes such as b"MOVE N\\n" or b"SPLIT 3\\n"; b"" is no action, which kills the dragon
Randomness is seeded per (bot, seed, dragon), so a match replays identically.

Sensing is exact for the four neighbours of the head (kelp, dragon parts, portals), which is all any of these bots
looks at besides the pearls in the 7x7 window. None of them plans ahead, so all of them can be trapped.
"""
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "bot"))
import proto  # noqa: E402

DX, DY = proto.DX, proto.DY  # N, E, S, W; y grows downwards
MOVES = (b"MOVE N\n", b"MOVE E\n", b"MOVE S\n", b"MOVE W\n")
OPPONENTS = {}  # name -> class, in registration order


def look(t, w, h):
    """Per direction N, E, S, W: (kelp, blocked, portal) flags for the head's four neighbours.

    blocked = a dragon part (any team, including our own neck and tail) sits on the adjacent tile.
    """
    occ = {(int(q[2]), int(q[3])) for q in t.parts}
    vt = t.edges[11].split()
    edge = (t.edges[3].split()[3], vt[4], t.edges[4].split()[3], vt[3])
    kelp = tuple(e == b"w" for e in edge)
    portal = tuple(e not in (b".", b"w") for e in edge)
    blocked = tuple(((t.hx + DX[d]) % w, (t.hy + DY[d]) % h) in occ for d in range(4))
    return kelp, blocked, portal


def clean_dirs(t, w, h):
    """Directions whose step certainly survives: no kelp, no dragon part ahead, no portal (its far side is unseen)."""
    kelp, blocked, portal = look(t, w, h)
    return [d for d in range(4) if not (kelp[d] or blocked[d] or portal[d])]


def safe_dirs(t, w, h):
    """clean_dirs, else the portals (a gamble), else all four (the dragon is boxed in and doomed)."""
    return clean_dirs(t, w, h) or [d for d, p in enumerate(look(t, w, h)[2]) if p] or [0, 1, 2, 3]


def pearls(t):
    """Offsets (dx, dy) from the head of every pearl in the 7x7 window."""
    return [(i % 7 - 3, i // 7 - 3) for i, f in enumerate(t.flags) if f == b"1"]


def rel(t, w, h, x, y):
    """Offset of map cell (x, y) from the head, wrapped onto the short way round the torus."""
    dx, dy = (x - t.hx) % w, (y - t.hy) % h
    return (dx - w if dx > w // 2 else dx), (dy - h if dy > h // 2 else dy)


def toward(pool, targets, heading):
    """The direction in `pool` that ends nearest (Manhattan) to any target; ties keep the heading."""
    return min(pool, key=lambda d: (min(abs(px - DX[d]) + abs(py - DY[d]) for px, py in targets), d != heading, d))


def cruise(dragon, t, pool):
    """Keep the heading if the pool allows it, otherwise any move from the pool."""
    return MOVES[t.dir if t.dir in pool else dragon.rng.choice(pool)]


class Dragon:
    """A player's memory of one of its dragons."""

    def __init__(self, seed, init):
        _, self.team, self.w, self.h, self.limit = proto.parse_init(init)
        self.rng = random.Random(seed)
        self.mem = {}


class Player:
    """Base class: parses each turn and hands `act` the dragon's memory plus the parsed turn."""

    name = "?"

    def __init__(self, seed=0, name=None):
        self.seed = seed
        self.name = name or self.name
        self.dragons = {}

    def spawn(self, did, init):
        self.dragons[did] = Dragon(f"{self.name}/{self.seed}/{did}", init)

    def reply(self, did, block):
        return self.act(self.dragons[did], proto.parse_turn(block))

    def act(self, dragon, t):
        raise NotImplementedError


def register(cls):
    OPPONENTS[cls.name] = cls
    return cls


@register
class Dummy(Player):
    """Sends no action, so every dragon dies on its first turn. Only checks that you can win at all."""

    name = "dummy"

    def act(self, dragon, t):
        return b""


@register
class Random(Player):
    """A uniformly random move with no safety check; a quarter of its moves reverse into its own neck."""

    name = "random"

    def act(self, dragon, t):
        return MOVES[dragon.rng.randrange(4)]


@register
class Naive(Player):
    """A first bot: runs at the nearest pearl in view, otherwise straight on. Its only precaution is not
    reversing into its own neck; kelp and other bodies it never checks."""

    name = "naive"

    def act(self, dragon, t):
        ps = pearls(t)
        if not ps:
            return MOVES[t.dir]
        return MOVES[toward([d for d in range(4) if d != (t.dir + 2) % 4], ps, t.dir)]


@register
class Straight(Player):
    """Never turns. Dies at the first kelp or body in its path, and until then is a fast moving obstacle."""

    name = "straight"

    def act(self, dragon, t):
        return MOVES[t.dir]


@register
class Chaser(Player):
    """Greedy with eyes: heads for the nearest pearl anywhere in its 7x7 window, using only safe moves."""

    name = "chaser"

    def act(self, dragon, t):
        pool = safe_dirs(t, dragon.w, dragon.h)
        ps = pearls(t)
        return MOVES[toward(pool, ps, t.dir)] if ps else cruise(dragon, t, pool)


@register
class SafeRandom(Player):
    """A uniformly random move among those that do not kill it."""

    name = "safe_random"

    def act(self, dragon, t):
        return MOVES[dragon.rng.choice(safe_dirs(t, dragon.w, dragon.h))]


@register
class Rammer(Player):
    """Hunts the nearest enemy head in view and drives into it (head to head kills both). Chases pearls
    otherwise. A probe for how the other side handles being charged."""

    name = "rammer"

    def act(self, dragon, t):
        w, h = dragon.w, dragon.h
        heads = [rel(t, w, h, int(q[2]), int(q[3])) for q in t.parts if q[0] != dragon.team and q[5] == b"1"]
        pool = safe_dirs(t, w, h)
        if heads:
            px, py = min(heads, key=lambda p: abs(p[0]) + abs(p[1]))
            kelp, _, portal = look(t, w, h)
            for d in range(4):
                if (DX[d], DY[d]) == (px, py) and not (kelp[d] or portal[d]):
                    return MOVES[d]  # the head is next to us: ram it
            return MOVES[toward(pool, [(px, py)], t.dir)]
        ps = pearls(t)
        return MOVES[toward(pool, ps, t.dir)] if ps else cruise(dragon, t, pool)


@register
class Greedy(Player):
    """Steps onto an adjacent pearl if that is safe, else keeps its heading, else any non-fatal move."""

    name = "greedy"

    def act(self, dragon, t):
        kelp, blocked, _ = look(t, dragon.w, dragon.h)
        safe = [d for d in range(4) if not (kelp[d] or blocked[d])]
        win = (17, 25, 31, 23)  # window tile of the N, E, S, W neighbour
        for d in safe:
            if t.flags[win[d]] == b"1":
                return MOVES[d]
        if t.dir in safe:
            return MOVES[t.dir]
        return MOVES[safe[0]] if safe else MOVES[0]


@register
class Hugger(Player):
    """Right-hand rule: runs straight until something is on its right, then follows kelp and bodies round.
    Never goes for a pearl, so it lives long and grows slowly."""

    name = "hugger"

    def act(self, dragon, t):
        clean = clean_dirs(t, dragon.w, dragon.h)
        right, left = (t.dir + 1) % 4, (t.dir + 3) % 4
        hugging = dragon.mem.get("hug", False)
        if hugging and right in clean:  # the wall ended: curve round it, still hugging
            return MOVES[right]
        dragon.mem["hug"] = right not in clean
        for d in (t.dir, left, right):
            if d in clean:
                return MOVES[d]
        return MOVES[dragon.rng.choice(safe_dirs(t, dragon.w, dragon.h))]


@register
class Splitter(Chaser):
    """A chaser that splits in half whenever it is long enough, so a team grows into many small dragons."""

    name = "splitter"
    split_at = 6  # length at which to split (both halves are then at least 3 long)
    cap = 24  # ... while the team has fewer dragons than this

    def act(self, dragon, t):
        if t.length >= self.split_at and t.units < min(dragon.limit, self.cap):
            return b"SPLIT %d\n" % (t.length // 2)
        return super().act(dragon, t)


def _load_snapshot_brain(dirpath):
    """Loads a self-contained past submission's own brain.py + proto.py in isolation, so it plays exactly as it
    shipped (its own wire format, its own DEFAULTS) rather than accidentally picking up bot/'s current proto.py
    through Python's normal `import proto` module-cache lookup. Returns the snapshot's Brain class."""
    import importlib.util

    dirpath = pathlib.Path(dirpath)
    unique = f"_snapshot_{abs(hash(str(dirpath)))}"
    proto_spec = importlib.util.spec_from_file_location(f"{unique}_proto", dirpath / "proto.py")
    proto_mod = importlib.util.module_from_spec(proto_spec)
    sys.modules[f"{unique}_proto"] = proto_mod
    proto_spec.loader.exec_module(proto_mod)
    # brain.py does a plain `import proto`; point that name at the snapshot's own proto.py just while brain.py's
    # module body runs, then put whatever was there back so nothing else in this process is affected.
    prior = sys.modules.get("proto")
    sys.modules["proto"] = proto_mod
    try:
        brain_spec = importlib.util.spec_from_file_location(f"{unique}_brain", dirpath / "brain.py")
        brain_mod = importlib.util.module_from_spec(brain_spec)
        sys.modules[f"{unique}_brain"] = brain_mod
        brain_spec.loader.exec_module(brain_mod)
    finally:
        if prior is not None:
            sys.modules["proto"] = prior
        else:
            del sys.modules["proto"]
    return brain_mod.Brain


class SnapshotBrainPlayer:
    """Wraps a past submission's own Brain class (see _load_snapshot_brain) behind the arena/league Player
    interface (spawn/reply), so an old snapshot can be used as an opponent exactly like any sparring bot."""

    name = "?"
    _brain_cls = None  # set per-subclass by snapshot_opponent()

    def __init__(self, seed=0, name=None):
        self.name = name or self.name
        self.dragons = {}

    def spawn(self, did, init):
        self.dragons[did] = self._brain_cls.from_init(init)

    def reply(self, did, block):
        try:
            return self.dragons[did].act(block)
        except Exception:  # noqa: BLE001
            return b""


def snapshot_opponent(register_name, dirpath):
    """Registers a past submission's brain.py (at `dirpath`) as an opponent under `register_name`, playing
    exactly as it shipped -- own DEFAULTS, own wire format, isolated from bot/'s current brain.py/proto.py."""
    cls = type(f"Snapshot_{register_name}", (SnapshotBrainPlayer,),
                {"name": register_name, "_brain_cls": _load_snapshot_brain(dirpath)})
    OPPONENTS[register_name] = cls
    return cls


@register
class Grower(Chaser):
    """A chaser that splits in half early on, same as Splitter, but caps at a handful of dragons and stops
    splitting well before the round-500 tiebreak. The rest of the game is unopposed pearl-chasing, so
    whatever it has left grows long and safely -- a proxy for opponents that beat a big swarm on the
    longest-living-dragon tiebreak despite ending with far fewer units and less total length."""

    name = "grower"
    split_at = 6  # length at which to split (both halves are then at least 3 long)
    cap = 10  # ... while the team has fewer dragons than this
    split_deadline = 150  # ... and only before this round; after it, only chases pearls and stays alive

    def act(self, dragon, t):
        if t.length >= self.split_at and t.units < min(dragon.limit, self.cap) and t.rnd < self.split_deadline:
            return b"SPLIT %d\n" % (t.length // 2)
        return super().act(dragon, t)


_MANUAL_HEURISTICS = pathlib.Path(__file__).resolve().parents[1] / "manual-heuristics"
if _MANUAL_HEURISTICS.is_dir():
    snapshot_opponent("manual_heuristics", _MANUAL_HEURISTICS)


def _load_policy_weights():
    """Same flat frombuffer + reshape bot/main.py used before the 2026-09-27 switch to Brain (see
    tools/eval_vs_policy.py) -- the real, previously-shipped trained weights, not a re-derived stand-in."""
    import numpy as np
    import encoder as _encoder
    from policy import H1, H2, N_MOVE, SPLIT_FRACS

    shapes = [
        ("b1", (H1,)), ("b2", (H2,)), ("bm", (N_MOVE,)), ("bs", (len(SPLIT_FRACS) + 1,)),
        ("w1", (_encoder.VOCAB, H1)), ("w2", (H1, H2)), ("wd", (_encoder.DENSE_SIZE, H1)),
        ("wm", (H2, N_MOVE)), ("ws", (H2, len(SPLIT_FRACS) + 1)),
    ]
    raw = (pathlib.Path(__file__).resolve().parents[1] / "bot" / "weights.bin").read_bytes()
    flat = np.frombuffer(raw, dtype=np.float32)
    out, i = {}, 0
    for name, shape in shapes:
        n = int(np.prod(shape))
        out[name] = flat[i:i + n].reshape(shape)
        i += n
    return out


_POLICY_WEIGHTS = None


@register
class PolicyOpponent(Player):
    """The previously-shipped trained Policy (bot/policy.py + bot/weights.bin) as a sparring opponent, so tuning
    accounts for being heavily outnumbered (it typically fields ~8x the dragons Brain does), not just the
    disciplined-low-population case `grower` covers."""

    name = "policy"

    def __init__(self, seed=0, name=None):
        super().__init__(seed, name)
        global _POLICY_WEIGHTS
        if _POLICY_WEIGHTS is None:
            _POLICY_WEIGHTS = _load_policy_weights()
        from policy import Policy as _Policy
        self._Policy = _Policy

    def spawn(self, did, init):
        self.dragons[did] = self._Policy.from_init(init, _POLICY_WEIGHTS, seed=self.seed * 1000 + did)

    def reply(self, did, block):
        try:
            return self.dragons[did].act(block)
        except Exception:  # noqa: BLE001
            return b""
