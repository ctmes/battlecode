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

- `mh4feed_2026-09-29/`: bot/brain.py + proto.py + known_maps.py with feed_r/feed_len/feed_min/feed_dist/feed_ratio
  (off by default; DEFAULTS are the base, not mh4). tune.py's "mh4feed" plays it as mh4 + feeding the king (feed_r 250,
  feed_len 8, feed_dist 3, feed_min 8, feed_ratio 1.5): a dragon at most 8 long within 3 of a teammate's head showing
  8+ segments (and 1.5x its length), with no enemy head that close, dies on the spot so the long one eats its pearls.
  Reproduces mh4 game for game with the feed off (20/20). Unswbc 1.2.2, real ladder maps x fresh held-out seeds
  9,200,000-9,200,009, both seats, 200 games per opponent (mh4 on the same games in brackets): mh4 58.0% (50),
  topstyle 86.0 (82.0), portal_farmer 91.0 (87.5); longest dragon at round 500 about 28-30 vs mh4's 22-24.
- `mh5_2026-09-29/`: a complete, submittable bot folder (main.py, brain.py, proto.py, known_maps.py, bot.toml) whose
  DEFAULTS are mh5 = mh4 + feeding the king (feed_r 250, feed_len 8, feed_dist 3, feed_min 8, feed_ratio 1.5; screened,
  not tuned). Built from mh4's own brain.py plus only the feed code and settings (proto.py, known_maps.py, main.py and
  bot.toml are mh4's), so it does not carry bot/brain.py's later work (radar). Plays mh4feed game for game (20/20) and,
  with feed_r 0, mh4 (20/20); its bench numbers are mh4feed's above.
- `mh6_2026-09-29/`: a complete, submittable bot folder whose DEFAULTS are mh6 = mh5 + split_enemy_dist 0 (a split is
  no longer refused just because an enemy head is in view). mh5's files plus only that change; plays mh5 game for game
  with split_enemy_dist 99 (20/20). Why: in the 25 ladder games the ~1600 teams eliminated mh4/mh5 (29 Sep), that rule
  blocked 65% of our split-ready turns before round 150 and we were outnumbered 9 to 34 by round 150; the top 3 make
  24-39% of their splits with an enemy head in view (we: 6%). Held out (seeds 9,300,000-9,300,009, 200 games each):
  54.5% vs mh5, topstyle 86.0 (mh5 85.5), portal_farmer 89.5 (91.0) -- level locally, where nobody crowds us early.
