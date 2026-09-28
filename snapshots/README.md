Frozen copies of bot code, used as tuning/validation opponents (tools/tune.py OPPONENTS, via tools/frozen_brain.py)
so that an opponent never changes when bot/brain.py does. Never edit a snapshot; add a new dated folder instead.

- `brain_2026-09-27/`: bot/brain.py + proto.py at commit 3bba468 (DEFAULTS = the live v2 manual-heuristics play,
  plus the switched-off grow_care / crowding / sprint / explore / terrain-sonar mechanisms that "v3" and
  "tuned_0927" turn on through params).
- The live v2 bot itself is `manual-heuristics/` at the repo root.
- `mh2_2026-09-28/`: bot/brain.py + proto.py at commit b9a62f4, submitted to the ladder as "mh2" (v4) on
  2026-09-28 06:40 UTC: v2 manual heuristics plus boxed_split=1.
- `mh3_2026-09-28/`: bot/brain.py + proto.py as submitted to the ladder as "mh3" on 2026-09-28: mh2 plus
  spare_team=1, boxed_reserve=4 and the rear split (boxed_rear_len=5, boxed_rear_r=433). Held-out ladder set,
  300 games per opponent: 71.2% vs mh2, 77.2% overall vs mh2 / live / v3 / grower_frozen.
