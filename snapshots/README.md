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
- `farmer_2026-09-29/`: mh3 plus the dive_* parameters (off by default, so with {} it plays exactly mh3). tune.py's
  "farmer" opponent plays it with dive_len=3, dive_trap=0, dive_scope=0: short dragons ignore the trap check, which
  reproduces the ladder opponents' foraging (fountain farming, ~3x our death rate) that no other local opponent has.
- `portal_2026-09-29/`: the farmer plus Brain.oracle_init() and a "portal_use" parameter (off by default, so with {}
  it plays exactly the farmer). A bench-only opponent: tools/bench1x.py hands each dragon the true map (kelp,
  portals, portal pairs), then with portal_use=1 portal moves are legal and every flood fill crosses portals.
  bench1x's "portal" and "portal_farmer" opponents. On Portals the halves never connect; 96 of the 104 spawning
  tiles are in pockets reachable only through portals, which is why the ladder teams out-eat us there.
- `topstyle_2026-09-29/`: the farmer plus boxed_r_end, ram_len and ram (off by default, so with {} it plays exactly
  mh3: 20 of 20 games identical). tune.py's "topstyle" opponent plays it the way the ladder's top 3 do
  (tools/scout.py): split_r_end=300, boxed_r_end=300, boxed_rear_r=0, ram_len=3, ram=1000 and the farmer's dive.
  bench1x, held-out seeds 9,000,000+, 200 games each: 75.5% vs mh3, 73.0% vs farmer, 58.2% vs portal_farmer.
- `mh3p_2026-09-29/`: mh3 plus portal support (DEFAULTS "portals", "map_oracle", "oracle_seen"), with its own copy of
  known_maps.py (brain.py reads it from its own folder). Unswbc 1.2.2, the real ladder maps x seeds 9,000,000-9,000,009,
  200 games per opponent: 72.2% vs mh3, 93.0% vs live, 74.0% vs farmer, 65.0% vs portal, 68.5% vs portal_farmer.
- `mh4_2026-09-29/`: a complete, submittable bot folder (main.py, brain.py, proto.py, known_maps.py, bot.toml) whose
  DEFAULTS are mh4: mh3 + portals (portals, map_oracle, oracle_seen) + topstyle (split_r_end 300, boxed_r_end 300,
  boxed_rear_r 0, ram_len 3, ram 1000) + diving near fountains at any length (dive_len 99, dive_trap 0, dive_scope 2,
  dive_radius 6). Built from the merged bot/brain.py (main + branch `portals`), which reproduces mh3, mh3p, topstyle
  and farmer game for game. Unswbc 1.2.2, the real ladder maps x fresh held-out seeds 9,100,000-9,100,009, both
  seats, 200 games per opponent (mh3 on the same games in brackets): mh3 91.0% (50), mh2 96.5 (70.5), live 97.5
  (87.0), mh3p 74.5 (29.5), topstyle 83.0 (21.5), farmer 92.0 (45.0), portal_farmer 89.0 (41.5), portal 82.0 (42.0),
  v3 98.0 (84.0), grower_frozen 92.5 (81.5). Longest dragon at round 500 about 21-27 vs mh3's 10-12. Every map >= 69%
  (Devil 69, Trauma 80). Judge sandbox: at most 18.4M points a turn (Portals, Schooltime).

