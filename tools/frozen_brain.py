"""Loads a frozen bot/ snapshot -- e.g. manual-heuristics/, the exact code currently submitted and active on
the ladder -- as an isolated Brain class, so it can be used as a training/validation opponent that never drifts
as bot/brain.py keeps changing under active tuning. Without this, every "vs Brain" number in tune.py/evolve.py
tracks whatever bot/brain.py happens to contain right now, not what is actually out there being scrimmed.

brain.py does `import proto` at module scope, so each frozen snapshot's brain.py and proto.py are loaded
together under snapshot-specific module names (never "brain"/"proto", which stay bound to the live bot/ copy
everywhere else in the process) and wired to each other before either is cached.
"""
import importlib.util
import pathlib
import sys

_cache = {}


def _load(mod_name, path):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def load(dir_path):
    """(Brain class, DEFAULTS dict) for the brain.py + proto.py under `dir_path`. Cached per directory, so
    repeated calls (one per worker process, one per job) pay the import cost once."""
    key = str(pathlib.Path(dir_path).resolve())
    if key in _cache:
        return _cache[key]
    tag = f"_frozen{abs(hash(key)) % (1 << 32)}"

    proto_mod = _load(f"{tag}_proto", pathlib.Path(key) / "proto.py")
    prev_proto = sys.modules.get("proto")
    sys.modules["proto"] = proto_mod  # brain.py's `import proto` must resolve to this snapshot's copy
    try:
        brain_mod = _load(f"{tag}_brain", pathlib.Path(key) / "brain.py")
    finally:
        if prev_proto is None:
            sys.modules.pop("proto", None)
        else:
            sys.modules["proto"] = prev_proto

    brain_mod.Brain.strict, brain_mod.Brain.debug = True, False  # match league.py's _init() convention
    result = (brain_mod.Brain, brain_mod.DEFAULTS)
    _cache[key] = result
    return result
