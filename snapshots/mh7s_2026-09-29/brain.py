"""Heuristic brain for the sea-dragon bot.

Order of decisions each turn:
  1. Legality, exact from the visible 7x7 state (kelp, any segment, head-on). A legal move always exists
     unless the dragon is boxed in; then the least-bad option is taken.
  2. Among legal moves: reachable-area (trap) check by bitboard flood fill, pearl chasing, head-on risk.
  3. Optional split (off by default; thresholds are tunable parameters).
  4. Optional prediction, local and relayed: a visible (or sonar-reported) enemy head's own facing extrapolated
     one step, penalized like head_risk if a candidate move lands on or next to it.

Everything is plain Python ints / bitboards: on the judge a Python loop iteration costs thousands of CPU
points, whereas big-int shifts are cheap (Points lab: 32x32 flood fill 2.0M points vs 46M for a list BFS).
Cell index = y * W + x. Edge memory: a kh/kv bit set at a tile means kelp on that tile's NORTH/WEST edge
(ph/pv: portal). With "portals" off, portals are walls because their far side is unknown; on, a pair whose two
edges have both been seen becomes a link (see DEFAULTS "portals").
"""
import os
import time

import proto

_pc = time.perf_counter_ns
DX, DY, LETTERS = proto.DX, proto.DY, proto.LETTERS
DOT7, DOT8 = proto.DOT7, proto.DOT8
MOVES = (b"MOVE N\n", b"MOVE E\n", b"MOVE S\n", b"MOVE W\n")
WIN_R = [t // 7 for t in range(49)]
WIN_C = [t % 7 for t in range(49)]
NB_WIN = (17, 25, 31, 23)  # window tile of the head's N, E, S, W neighbour (the head is tile 24)
# Per direction N,E,S,W (matching the `kelp`/`port` tuples in decide()): the (tile, vertical) an edge in that
# direction is stored under, in the same terms learn_edges()/kh/kv/ph/pv already use (a kh/kv bit at a tile means
# kelp on that tile's own north/west edge) -- so a relayed edge lands on the exact bit a receiver's own vision
# would have set. E and S look one tile over because that neighbour "owns" the shared edge in this convention.
TERRAIN_EDGE_TILE = (
    lambda hx, hy, w_, h_: (hx, hy),                  # N: this tile's own north edge
    lambda hx, hy, w_, h_: ((hx + 1) % w_, hy),        # E: the east neighbour's west edge
    lambda hx, hy, w_, h_: (hx, (hy + 1) % h_),        # S: the south neighbour's north edge
    lambda hx, hy, w_, h_: (hx, hy),                  # W: this tile's own west edge
)
TERRAIN_EDGE_VERTICAL = (0, 1, 0, 1)

# legality classes, best first
OK, PORTAL, TRADE_ENEMY, TRADE_TEAM, DEAD = 0, 1, 2, 3, 4
# With "spare_team" on, a boxed-in dragon that cannot split prefers dying alone (DEAD) to a head-on with a teammate
# (TRADE_TEAM), which kills both: in local games a third of mh2's long-dragon deaths were doomed short teammates
# taking a long dragon down with them. A trade with an enemy still ranks above dying alone.
SPARE_RANK = {OK: 0, PORTAL: 1, TRADE_ENEMY: 2, DEAD: 3, TRADE_TEAM: 4}

DEFAULTS = {
    # Tuned 2026-09-22 by CMA-ES (tools/tune.py) over 25 of these parameters, in 2 rounds (the 2nd with a widened
    # search box after the 1st pegged 3 parameters at their bound), then picked by a 5-way, 3160-game round-robin
    # (tools/tournament.py) against DEFAULTS and 3 other tuned candidates on maps none of them were tuned or
    # validated on: this candidate (the mean of run1's last 5 generations, tools/tuned/run1_avg5.json) scored 59.5%
    # against the field and beat every other candidate head-to-head, despite 2 candidates from the wider search
    # beating the fixed reference opponents individually by more -- a reminder that single-opponent win rate doesn't
    # imply a round-robin win. Also beats these DEFAULTS 61-68% over 400+ held-out games. Parameters left out of
    # tools/tune.py's SPACE (pessimistic, need_cap, squeeze, deny*, budget_ns) were not tuned and keep their old
    # values and comments; squeeze/deny* are 0 because herding never worked (see the strategy notes).
    "pearl_here": 2348.8051,   # stepping onto a pearl
    "pearl_near": 64.2029,     # per step closer to the nearest visible pearl
    "pearl_k": 7,              # look-ahead distance (steps) of the pearl distance field
    "explore": 0.0,            # bonus for a move onto a tile never inside this dragon's own vision before, only
                                # scored with no pearl signal at all nearby (0 = off): otherwise "straight" is the
                                # only directional preference, which can loop a dragon back through searched-empty
                                # ground instead of pushing into new territory. Untuned.
    # Food field (29 Sep). mh6's ladder openings: in the 23 of 95 games we were eliminated, the enemy's nearest head
    # was closer than ours to 127 of the spawn pearls appearing before round 150 (ours: 55), and whoever is closer
    # eats it 75-93% of the time. With no pearl in view a dragon had nothing pulling it anywhere but "straight", so it
    # wandered empty ground (Devil's open ends) while theirs farmed the fountains. On a known map (map_oracle) this
    # adds food_pull per step up known_maps.FIELDS -- the spawn rate around each tile, decayed 0.75 per move of maze
    # distance, log-scaled so a step towards a lone fountain is one unit -- whenever no pearl is in view. 0 = off.
    # Tried alongside it and dropped: "spread" (with no pearl in view, a bonus per step away from each teammate head
    # in view, for maps where food lands anywhere) scored 40-52% vs mh6 at 15-120, Default included.
    "food_pull": 0.0,
    # Rallying on the king (29 Sep). mh6 lost 13 of 95 ladder games at round 500 on a shorter longest dragon while
    # holding more total length (113 vs 69); the top teams' kings get 57-75% of their food from teammates dying next
    # to them, but feeding (feed_r) only fires when a teammate happens to be within feed_dist. Sonar is the only way
    # to say where the king is: from rally_r - rally_age on, a dragon at least feed_min long that has heard of no longer
    # one pings its own head (proto.SONAR_KING) along its move, and every dragon relays the longest king it heard of
    # in the last rally_age rounds. From rally_r on, a dragon at most feed_len long with no pearl in view adds
    # rally_pull per step towards that head. 0 = off (no pings, the old protocol-2 bot).
    "rally_r": 0,
    "rally_pull": 0.0,
    "rally_age": 30,
    # Sprinting: MOVE with 2+ direction letters (e.g. MOVE NNE) takes that many steps in one turn; every step after
    # the first costs a tail segment unless it eats a pearl, and a dragon of length 2 cannot pay for one (it dies
    # with no valid action). sprint_paths() applies each candidate path step by step exactly as the engine does
    # (rules "Execution order"), so a sprint is as safe as a plain move, turns included. 1 = off (never sprint).
    # Ladder top 3 (tools/scout.py replays, 29 Sep): sprints are 0.3-1.3% of their turns and almost never kill the
    # sprinter by accident; 11-38% are rams (a 3-long dragon sprinting 2-3 tiles onto an enemy head), most of the
    # rest take pearls on the way. Their rams cost mh4 323 dragons in 55 ladder games (median length 6 vs their 3).
    "sprint_max": 1,
    "sprint_seg": 2348.8051,   # score per segment a sprint spends (a pearl eaten on the way pays one back)
    "sprint_contest": 0.0,     # bonus per pearl a sprint eats that an enemy head is within 2 steps of
    "sprint_ram": 0.0,         # score of a 2-3 step sprint onto an enemy head (same length gate as "ram") ...
    "sprint_ram_margin": 0,    # ... that shows at least this many segments more than we have
    "sprint_ram_r": 0,         # ... from this round on
    # Defence: an enemy head at least 3 long can sprint-ram any tile within min(3, its length - 1) steps of it, so a
    # move ending there is penalized like an adjacent enemy head (head_risk), times this (0 = off). Only enemies a
    # trade would hurt us against count, as for head_risk.
    "sprint_threat": 0.0,
    "sprint_threat_len": 0,    # ... for our dragons at least this long
    "trap": 8763.9753,         # penalty scale when the reachable area is smaller than needed
    "need_margin": 3,          # needed area = length + margin
    "need_cap": 120,           # ... capped, so the flood fill can exit early
    "area": 9.8743,            # bonus when not trapped
    "pessimistic": 1,          # trap check counts only tiles seen so far (unseen tiles may be kelp pockets)
    "head_risk": 553.4206,     # adjacent enemy head (we are the longer dragon: a trade hurts us)
    "head_risk_small": 236.0887,  # adjacent enemy head when we are much shorter (a trade helps us)
    "trade_ratio": 0.9463,     # we count as "much shorter" below this fraction of the enemy's visible length
    "team_head_risk": 596.8384,
    "straight": 30.3172,       # keep heading
    "split_len": 3,            # children (dragons born by a split) split when length >= this
    "split_len_max": 0,        # ... but never once length reaches this (0 = off): a floor alone never stops an
                                # already-long dragon splitting itself away just because room/pearls/cap allow it
    "split_child": 2,
    "split_units": 57,         # ... and the team has fewer dragons than this (the engine's unit limit also applies)
    # Founders (alive at round 0) seed the swarm, then stop splitting so one dragon grows: the round-500 tiebreak is
    # the longest living dragon, and a swarm of length-4 dragons loses it.
    "founder_split_len": 3,
    "founder_units": 49,       # founders split only while the team has fewer dragons than this
    "grow_mod": 7,             # children with id % grow_mod == 0 never split: they grow (0 = every child swarms)
    # Dynamic split conditions: split only when the situation supports another dragon.
    "tiles_per_unit": 0,       # team-size target = map tiles / this, at least 2 and at most the unit limit (0 = off)
    "split_pearls": 1,         # ... and at least this many pearls are in view (food for the extra mouth)
    "split_r_end": 433,        # ... and it is before this round (late children do not pay back)
    "split_min_exits": 1,      # ... and the parent has at least this many safe moves (not cornered)
    # ... and no enemy head is within this Manhattan distance (99 = none anywhere in view, as before). Ladder, 29 Sep:
    # in the 25 games the ~1600 teams eliminated mh4/mh5, an enemy head was in view on 41% of our turns before round
    # 150 and this rule blocked 65% of our split-ready turns; we split 2-5 times per 25 rounds to their 8-18 and were
    # outnumbered 9 to 34 by round 150. The top 3 make 24-39% of their splits with an enemy head in view (we: 6%).
    # 0 (never blocks), held out (seeds 9,300,000+, 200 games each): 54.5% vs mh5, topstyle 86.0 (mh5 85.5), portal_farmer
    # 89.5 (91.0) -- level locally, where no opponent crowds us early the way those ladder teams do.
    "split_enemy_dist": 99,
    # Local crowding/food, as an alternative to a flat team-size cap and pearl count that cannot tell a dragon
    # splitting into open space from one splitting into a knot of its own teammates (a big source of other-body
    # deaths in a large swarm). Both 0 = off (old behaviour): nearby teammates use `heads`, already built every
    # turn for head-risk scoring, so this costs nothing extra.
    "split_mate_radius": 0,    # Manhattan radius to count visible teammate heads in (0 = the check is off)
    "split_mate_cap": 2,       # refuse to split if this many teammates are already within split_mate_radius
    "split_food_ratio": 0.0,   # if > 0, require pearls-in-view >= this * (nearby teammates + 1), instead of
                                # the flat split_pearls count above
    "dead_end": 191.4698,      # penalty for stepping onto a cell with a single way on
    "need_floor": 8,           # room a move must leave, at least (short dragons otherwise pass tiny pockets)
    "squeeze": 0.0,            # bonus per safe move a nearby enemy head loses (herding towards walls and bodies)
    # Tron-style territory: weight of (cells I reach first - cells enemy heads reach first).
    "voro": 2.7837,
    "voro_radius": 5,          # ... measured this many steps out
    "deny": 0.0,               # bonus for a move that leaves a visible enemy head less room (area denial / herding)
    "deny_radius": 4,          # only enemy heads this close (Manhattan) are considered
    "deny_margin": 2,          # an enemy counts as enclosed when its region is smaller than its visible length + this
    # Prediction: a visible enemy head's own reported facing (free -- every part already reports it) extrapolated
    # one step, penalized like head_risk if a candidate move lands on or next to it. A 10-500 weight sweep over
    # 1,232 games found no benefit at any weight and real harm above ~250 (score down to 41-42%, CIs excluding 50%
    # on the losing side); deaths/1000 turns (38.6-41.1) and the head-on share of deaths (59.6-61.1%) never moved
    # from baseline (40.4, 60.7%) anywhere in the range, despite the extrapolation itself testing as exactly
    # correct against 32k+ live sightings. A firing-rate check explains why: a candidate prediction exists on 44%
    # of turns, but it's the deciding factor in only 1.5% of all turns -- the much larger existing terms (trap,
    # pearl_here, head_risk) already determine the outcome the rest of the time, so even a real effect on those
    # rare deciding turns gets diluted to noise in the aggregate. Also structural: local prediction only ever
    # covers an enemy already in this dragon's own 7x7 window, which head_risk's exact check mostly already
    # handles. Untuned as a mechanism, not just a weight; see tools/tune.py SPACE and tools/tuned/.
    "predict": 0.0,
    # Sonar-relayed prediction (proto.pack_enemy_sonar/unpack_sonar, carrying facing, not just position): a dragon
    # broadcasts its nearest visible enemy's position, size AND facing, so a teammate who has never seen that enemy
    # can run the exact same one-step extrapolation "predict" does above -- the one thing local prediction cannot
    # structurally do, since it needs the enemy in this dragon's own vision. Tested alone (predict=0) over a 25x
    # weight range (10-250, 880 games): still flat -- deaths/1000 turns moved by at most 0.4 (34.5-34.9) and the
    # head-on share by at most 0.4pp (58.9-59.3%) across the whole range, score never clearing 50% either way. A
    # firing-rate check found why: a decodable sighting exists on 24% of turns but changes the chosen move on only
    # 0.7% of all turns -- sonar's lack of any addressing means most broadcasts reach a dragon they cannot help,
    # or reach nobody. Two earlier, position-only payloads (fed into "voro" territory, then a distance penalty)
    # also measured as no help; all three sonar designs and both prediction mechanisms are now documented negative
    # results, not just unlucky weights. sonar_predict=0 also turns off sending (pointless with nobody listening).
    # Untuned; see tools/tune.py SPACE and tools/tuned/.
    "sonar_predict": 0.0,
    # Sonar (terrain): relay one of the current tile's own known kelp/portal edges instead of an enemy sighting.
    # A fact, not a sighting, so it never goes stale the way sonar_predict's payload does -- worth trying because
    # a top-of-leaderboard opponent's replay showed far heavier sonar use than the (negative-result) enemy-relay
    # weight sweep above ever tested for. 0 = off: no sending, and received terrain messages are ignored (not
    # decoded at all, so a teammate using this costs nothing extra for one that has it off). Untuned.
    "sonar_terrain": 0.0,
    # Once a dragon's own length reaches grow_care_len, it plays more cautiously: trap/dead-end/head-on
    # penalties are multiplied by grow_care_mult and need_margin gets grow_care_margin added, so a dragon that
    # is already a real investment (whether a surviving founder or a grow_mod-designated grower) protects that
    # length instead of taking the same risks a short, disposable swarm dragon would. Back to off (2026-09-27):
    # the 2026-09-26 tuned values (len=1, mult=1.9346, margin=1) fixed the round-500 tiebreak loss in isolation,
    # but as part of the combined bot they lost to manual_heuristics/ -- the exact same code from before this
    # mechanism existed -- 33-46% instead of manual_heuristics' real, live 58% win rate. DEFAULTS resets here to
    # match manual_heuristics exactly, and grow_care (plus split_len_max/explore/the crowding gates) goes back
    # on only if a fresh, properly-validated tuning pass earns it back. See tools/tuned/ for the history.
    "grow_care_len": 0,
    "grow_care_mult": 1.0,
    "grow_care_margin": 0,
    # King: a founder that never splits and plays with king_care x the usual caution, so the team ends the game
    # with one long dragon. 74% of the live bot's ladder losses (37 of 57, 26 Sep 18:00 - 27 Sep 08:43 UTC) were at
    # round 500 on the longest-dragon tiebreak: its longest dragon averaged 8.5 against the opponent's 15.0. A founder
    # is king if its id is below king_first (every ladder map lists the teams' dragons alternately, so 2 = each
    # team's first founder) or it starts at least king_len long (Autarky, Prisoners Dilemma and Slithery Fight hand
    # out 14-, 11- and 25-long founders, which the swarm otherwise splits straight into 2-long children). 0 = off.
    "king_first": 0,
    "king_len": 0,
    "king_care": 1.0,
    # Boxed in (no safe move): split instead of taking the least-bad move, since a split stands still. 0 = off.
    # On 2026-09-28, on the held-out ladder set (tools/longest_bench.py --holdout: the 10 ladder maps as played plus
    # 5 reserved variants each, both seats, 120 games per opponent), this one change took the live v2 bot from 50.0%
    # to 72.5% [64-80] against v2 itself, 51.7 -> 75.8% vs v3, 45.4 -> 64.2% vs tuned_0927, 65.0 -> 78.3% vs
    # grower_brain, 87.1 -> 97.5% vs the evolved policy and 95.0 -> 96.7% vs splitter, and cut games lost by
    # elimination from 23 to 0. No illegal-split deaths. The king_* mechanism above did not help and stays off.
    "boxed_split": 1,
    # boxed_reserve 4 + spare_team 1, 2026-09-28, held-out ladder set (tools/longest_bench.py --holdout, 120 games per
    # opponent): 65.4% [57-73] head-to-head vs mh2 (the same bot with both at 0), 62.5 -> 74.6% vs tuned_0927,
    # 74.2 -> 76.7% vs live v2, 76.7 -> 78.3% vs grower_frozen, 75.0 -> 73.8% vs v3 (noise), no other change.
    # Known cost it does NOT fix (found by the other session from mh2 replays): after split_r_end every split is a
    # boxed one, and a boxed long dragon still loses 2 length per SPLIT -- see "rear split" in the project notes.
    "boxed_reserve": 4,        # unit slots below the limit that boxed-in splits of short dragons may not use
    "boxed_long": 8,           # ... "short" meaning shorter than this
    "spare_team": 1,           # boxed in and unable to split: die alone rather than head-on a teammate (see SPARE_RANK)
    # Rear split, 2026-09-28. mh2's ladder replays: our longest dragon split after round 433 (so every split a boxed
    # one) in 31 of 68 games vs 1 of 292 for v2, e.g. leading 12:9 at round 480 and losing 8:9. Held-out ladder set,
    # 300 games per opponent, mh3 -> mh3 + rear split: 60.8 -> 71.2% vs mh2, 77.7 -> 81.7% vs live v2, 73.8 -> 77.2%
    # vs v3, 76.7 -> 78.7% vs grower_frozen (72.2 -> 77.2% overall), longest dragon at round 500 about +1.3, tiebreak
    # losses 303 -> 248, no eliminations, no errors. On from round 0 it lengthened the longest more but did not win more.
    "boxed_rear_len": 5,       # boxed-in split of a dragon at least this long gives the child all but 2 segments
    "boxed_rear_r": 433,       # ... from this round on (0 = always; boxed_rear_len 0 = off)
    # Dive: short dragons take trapped pearls. On six ladder maps nearly all the pearls spawn on "fountains" (pearl
    # gap 1-1: a pearl every round the tile is free), and all but Devil's sit on dead-end branches of the kelp graph,
    # so the trap check vetoes them (~90% of the adjacent pearls we pass up; 95-100% of those vetoes are real traps).
    # Ladder opponents eat 3-15x our fountain pearls, die 3x as often (92% of their dead are shorter than 4) and eat
    # 90% of their own dead back. A fountain always shows countdown 1 (other spawning tiles: ~1% of sightings).
    # Ladder opponents' long dragons also rear-split (child = all but 2) in 69% of their long-parent splits all game long:
    # a U-turn out of a dead end. With boxed_rear_r 0, a dragon 4+ long diving into a fountain branch gets out the
    # same way, so dive_len above 3 lets long dragons dive too (only while that boxed split is legal).
    "dive_len": 0,             # dragons at most this long dive (0 = off)
    "dive_trap": 0.0,          # ... with the trap penalty multiplied by this
    "dive_scope": 1,           # 0 = on every move (reckless), 1 = only a move onto a pearl, 2 = on every move while
                                # a fountain this dragon has seen is within dive_radius (fountain: a tile showing
                                # countdown 1 on two turns running, which no other tile did in 3M sightings)
    "dive_fountain": 0,        # scope 1 only: 1 = only onto a pearl on a tile showing countdown 1 (a fountain)
    "dive_units": 0,           # ... and only while the team has at least this many dragons
    "dive_radius": 6,          # scope 2: Manhattan distance to a known fountain
    # Radar (unswbc 1.x protocol 3, 29 Sep). The bot prints PROTOCOL 3 once (split children inherit it) and pings
    # after every action, from the head, along: 1 = the new facing only; 2 = the new facing and both sides (never
    # back: that ray stops on our own neck). Next turn's ECHOES line counts what the rays stopped on first -- kelp,
    # a teammate's body or head, an enemy's body or head -- pooled over the rays, with no direction and no
    # distance. radar_lines() decodes it: a ray whose first hit is already in view (kelp, or a dragon part) is
    # subtracted, and what is left belongs to the rays that left the view, so with one of those the kind is exact,
    # and with several the counts still say "all clear to kelp" or "all blocked". Pings cost nothing else; a bot
    # that ignores sonar plays the same games. 0 = off (protocol 2, as up to mh4). All of the top 3 ping every
    # turn by 29 Sep: cheji and forgot along all 4 directions, Cutlery along its open ones (ahead and both sides).
    "radar": 0,
    "radar_dive": 0,           # 1 = no dive into a line whose ray stopped on a dragon (an occupied corridor)
    "radar_head": 0.0,         # penalty for moving along a line whose ray stopped on an enemy head
    # Top-team style, from 450 replays of the ladder's top 3 (Cutlery, cheji bt, forgot to mention; tools/scout.py,
    # 29 Sep). All three stop ordinary splits around round 300, so attrition shrinks the swarm (cheji: 45 dragons at
    # round 300, 7 at 400) while the survivors eat the corpses: 57-75% of their longest dragon's pearls after round
    # 300 are their own dead, and it ends 32-43 long against our 10.5 -- though we hold more total length, in 33
    # dragons. They also ram enemy heads with 2-3-long dragons, 22-25 times a game, more often than they are rammed.
    # "topstyle" = split_r_end 300, boxed_r_end 300, boxed_rear_r 0, ram_len 3, ram 1000, dive_len 3, dive_trap 0,
    # dive_scope 0. On the 1.x engine, the 10 ladder maps x held-out seeds 9,000,000+ x both seats (200 games per
    # opponent, tools/bench1x.py): 75.5% [69-81] vs mh3, 73.0% vs farmer, 58.2% vs portal_farmer; longest dragon at
    # round 500 15.2 vs mh3's 11.0. Screened on seeds 1-6 (120 games vs mh3): each part alone 51-64%, round 300 beat
    # 250 and 350, ram 300-3000 all alike; a narrow "feed" (dragons <= 3 long, within 2) added nothing -- see "feed_r".
    "boxed_r_end": 0,          # from this round on a boxed-in dragon splits only for a rear split (0 = off)
    "ram_len": 0,              # dragons at most this long may move onto an adjacent enemy head, killing both (0 = off)
    "ram": 0.0,                # ... when that enemy shows at least as many segments: a move scored at this
    # Feeding the king. The ladder's top 3 grow their longest dragon on teammates who kill themselves next to it: after
    # round 300 their king's line eats 22-28 own-team corpse pearls a game, 80-94% of them from deliberate "no action"
    # deaths, the teammate's head a median 2-4 from the king's when it died, often 6+ long (cheji: 54%). mh4's king eats
    # 11, mostly its own turnaround stubs, and loses to turnarounds what it gains farming fountain slots (see the notes).
    # From feed_r on, a dragon at most feed_len long whose head is within feed_dist of the head of a teammate showing
    # at least max(feed_min, feed_ratio x its length) segments, with no enemy head that close, dies on the spot (an
    # illegal SPLIT 1, as Cutlery does) so the long one eats the ceil(length/2) pearls it drops. 0 = off.
    # mh4 + feed (feed_r 250, feed_len 8, feed_dist 3, feed_min 8, feed_ratio 1.5), 1.x engine, the 10 ladder maps x
    # fresh held-out seeds 9,200,000-9,200,009 x both seats: 58.0% [51-65] vs mh4 (200 games), longest dragon at round
    # 500 28.5 vs 23.8; best on Slithery Fight 85%, Portals 70%, Trauma 70%. Same seeds vs topstyle 86.0% (mh4 82.0),
    # vs portal_farmer 91.0% (mh4 87.5), tiebreak losses 21 -> 12 and 14 -> 6. Why (80 games vs mh4): not a richer diet
    # -- the king's line eats as much after round 300 (41.2 vs 43.5), with the same own-corpse share (36 vs 37%) -- but
    # fewer turnarounds (8.4 vs 12.5): the feeders are the short teammates crowding its pocket, so it is boxed in less.
    # Screened on seeds 1-6 (120 games each):
    # feeding from 300 with dragons <= 6 55.8%; pulling would-be feeders towards the long teammate ("gather") and
    # leaving it the pearls near its head ("yield") added nothing, nor did limiting diving to dragons <= 6 (48.5% held out).
    "feed_r": 0,
    "feed_len": 8,
    "feed_min": 8,
    "feed_dist": 3,
    "feed_ratio": 1.5,
    # Portals. 0 = off: portals are walls, as in every version up to mh3 ({"portals": 0} plays mh3 exactly: 80/80
    # games, unmetered). On, a dragon remembers each portal's id, and once it has seen both edges with one id (in
    # practice: by stepping through, since the far side then comes into view) the pair is a link in its legality, trap
    # flood fill, pearl field and dead-end checks. On Portals and Trauma each team starts in a region whose pearls are
    # mostly behind portals (Portals: 4 spawn tiles in our 208-tile half, 32 more in 2x2 rooms reached only by portal;
    # Trauma: 10 in our 264-tile field, 110 in the maze behind it). 2026-09-29, unswbc 1.2.2 engine, the real ladder
    # maps x seeds 1-10, 200 games per opponent, mh3 -> learned portals -> + map_oracle + oracle_seen (all three on):
    # vs mh3 50 -> 71.5 -> 72.5%, live 81.7 -> 93.5 -> 94.5%, farmer 48.5 -> 67.0 -> 69.0%, and against the opponents
    # that are handed the true map and use portals (tools/bench1x.py): portal 40.0 -> 58.0 -> 68.0%, portal_farmer
    # 42.8 -> 62.5 -> 72.0%. Judge cost: at most 18.0M points a turn (sandbox, Schooltime, 47,728 turns).
    "portals": 0,               # on in mh4 (snapshots/mh4_2026-09-29)
    "portal_unknown": 150.0,   # score of stepping into a portal whose far side is still unknown (exploring)
    "portal_blind": 100.0,     # penalty for a known portal whose landing tile is out of view (unseen occupancy)
    # A dragon that stepped into an unknown portal and landed in a walled-in pocket with no pearl-spawning tile stops
    # exploring unknown portals (1 = on): Autarky's portals lead into closed 3x3 boxes with no spawn tiles, while on
    # Portals, Trauma, Default and Queen of Spades the far sides are where the pearls are.
    "portal_learn": 1,
    # 1 = explore unknown portals only with no pearl in view (like "explore"). Fixed Autarky and Trophy but lost more on
    # Portals, Trauma and Default (vs portal users 49.6 -> 42.5%, 240 games each): off.
    "portal_hungry": 0,
    # Known maps: the ladder plays a fixed pool of maps, copied into bot/known_maps.py (tools/gen_known_maps.py). A
    # dragon keeps the candidates of its board size whose kelp and portals agree with every edge it has seen; once
    # exactly one is left it loads that whole map: every portal pair, kelp edge and spawn tile, so no portal is ever
    # a blind bet. No candidate left (a new or changed map, or a mirrored bench variant) = learn as usual. Every
    # founder of every ladder map, and every split child, knows its map on its first turn. Needs "portals".
    "map_oracle": 0,            # on in mh4
    # ... and count the whole known map as seen, so trap checks stop fearing ground this dragon has not looked at yet.
    # Without it the oracle did worse than learning (52 vs 58% vs portal): known portals led into "unseen" traps.
    "oracle_seen": 0,           # on in mh4
    "budget_ns": 60_000_000,   # self-metering: skip optional work past this (points on the judge)
}


_KNOWN = {}


def known_maps():
    """MAPS from the known_maps.py beside this file (not via sys.path, so a snapshot loaded from another folder
    reads its own copy); {} if there is none."""
    if not _KNOWN:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "known_maps.py")
        try:
            with open(path) as f:
                exec(f.read(), _KNOWN)
        except OSError:
            _KNOWN["MAPS"] = {}
    return _KNOWN.get("MAPS", {})


def known_field(name):
    """The known map's food field (known_maps.FIELDS, one byte per cell), or None if this copy has none."""
    known_maps()
    return _KNOWN.get("FIELDS", {}).get(name)


class Brain:
    strict = False  # tests set this so exceptions surface instead of falling back
    debug = False  # tests set this to keep the last decision's context in self.dbg

    def __init__(self, dragon_id, team, width, height, unit_limit, params=None):
        self.id = dragon_id
        self.team = team
        self.W, self.H, self.limit = width, height, unit_limit
        self.p = DEFAULTS if not params else {**DEFAULTS, **params}
        n = width * height
        self.N = n
        self.full = (1 << n) - 1
        self.first = self.full // ((1 << width) - 1)  # bit at x == 0 of every row
        self.last = self.first << (width - 1)
        self.nf = self.full ^ self.first
        self.nl = self.full ^ self.last
        self.kh = self.kv = self.ph = self.pv = 0
        self.okh = self.okv = self.full
        self.seen = 0  # tiles that have been inside some 7x7 window
        self.body = []  # own head-to-tail cells, reconstructed from the head trail
        self.own = 0  # bitboard of self.body
        self.tail = None
        self.path = None  # cells crossed by the sprint sent last turn, in order (update_body inserts them all)
        self.pinged = ()  # directions of the radar rays sent last turn (DEFAULTS "radar")
        self.proto3 = False  # PROTOCOL 3 already sent
        self.lines = {}  # this turn's decoded radar: direction -> what its ray stopped on (see radar_lines)
        tpu = self.p["tiles_per_unit"]
        self.target = unit_limit if tpu <= 0 else max(2, min(unit_limit, n // tpu))  # preferred team size
        self.cd1 = 0  # tiles that showed pearl countdown 1 last turn (for dive_scope 2's fountain detection)
        self.founts = []  # (x, y) of tiles seen at countdown 1 on two turns running: fountains
        self.founder = None  # True when alive at round 0 (decided on the first turn)
        self.king = None  # a founder that never splits (see DEFAULTS "king_first"); decided on the first turn
        # portals (DEFAULTS "portals"): an edge key is cell * 2 + (1 for the cell's west edge, 0 for its north edge)
        self.pids = {}  # portal id -> the one edge key seen with it so far, or -1 once both are known
        self.link_dst = {}  # source cell * 4 + direction -> landing cell, for every known crossing
        self.lmap = {}  # source cell -> landing cells (the flood fill's view of the same links)
        self.lsrc = 0  # bitboard of the source cells in lmap
        self.pmem = 0  # pearls as last seen, kept for tiles that are out of view now
        self.due = {}  # cell -> round its next pearl is due (from the countdowns seen)
        self.spawns = 0  # bitboard of the pearl-spawning tiles seen
        self.explore_ok = True  # False once an unknown portal led somewhere with no pearl spawns (portal_learn)
        self.exploring = False  # the last move stepped into an unknown portal
        self.oracle = None  # map_oracle: None while undecided, False for an unknown map, else the known map's name
        self.cands = None  # known maps still consistent with what this dragon has seen
        self.field = None  # food_pull: the known map's food field (bytes, one per cell)
        self.kinfo = None  # rally: (x, y, length, round) of the longest teammate heard of (maybe this dragon)
        self.king_msg = None  # rally: this turn's SONAR_KING payload
        self.hseen = self.vseen = 0  # cells whose north / west edge has been in view
        self.errors = 0
        self.t0 = 0

    @classmethod
    def from_init(cls, init_bytes, params=None):
        return cls(*proto.parse_init(init_bytes), params=params)

    # ------------------------------------------------------------------ entry point
    def act(self, block):
        """Returns the action bytes (e.g. b'MOVE N\\n'); never raises."""
        self.t0 = _pc()
        t = None
        self.king_msg = None
        try:
            t = proto.parse_turn(block)
            action = self.decide(t)
        except Exception:  # noqa: BLE001 - a crash would kill the dragon
            if self.strict:
                raise
            self.errors += 1
            action = self.fallback(block)
        if self.debug:
            self.dbg["act"] = action
        if self.p["radar"]:
            # rays leave the head after the action, so "ahead" is the new facing (a sprint's last step); a split or
            # anything else keeps the old one
            first = action[:action.index(b"\n")]
            f = LETTERS.find(first[-1:]) if first.startswith(b"MOVE ") else (t.dir if t is not None and t.dir >= 0
                                                                               else 0)
            self.pinged = (f,) if self.p["radar"] == 1 else (f, (f + 1) % 4, (f + 3) % 4)
            if not self.proto3:
                self.proto3 = True
                action += b"PROTOCOL 3\n"
            msg = self.king_msg or 0  # rally: each ray also carries the king's head
            action += b"".join(b"SONAR %c %d\n" % (LETTERS[d], msg) for d in self.pinged)
        elif self.king_msg is not None:
            action += b"SONAR %d\n" % self.king_msg  # the last SONAR line wins
        return action

    def over(self):
        return _pc() - self.t0 > self.p["budget_ns"]

    # ------------------------------------------------------------------ bitboard primitives
    def _step(self, reach, free):
        """One dilation step across passable edges, restricted to `free` cells (torus)."""
        w_, n_, full = self.W, self.N, self.full
        okv, okh = self.okv, self.okh
        e = (((reach & self.nl) << 1) | ((reach & self.last) >> (w_ - 1))) & okv
        src = reach & okv
        w = ((src & self.nf) >> 1) | ((src & self.first) << (w_ - 1))
        s = ((reach << w_) | (reach >> (n_ - w_))) & okh
        src2 = reach & okh
        n = ((src2 >> w_) | (src2 << (n_ - w_))) & full
        out = (reach | e | w | s | n) & free
        hit = reach & self.lsrc  # known portal crossings (none unless "portals" is on and a pair has been seen)
        while hit:
            low = hit & -hit
            for dst in self.lmap[low.bit_length() - 1]:
                out |= (1 << dst) & free
            hit ^= low
        return out

    def flood(self, free, start, need):
        """Cells reachable from `start` through `free`, counting stops once `need` is reached."""
        reach = 1 << start
        step = self._step
        while True:
            new = step(reach, free)
            if new == reach:
                return new.bit_count()
            reach = new
            cnt = reach.bit_count()
            if cnt >= need:
                return cnt

    def window(self, x0, y0):
        """Bitboard of the 7x7 window whose top-left tile is (x0, y0), wrapped."""
        w_, h_ = self.W, self.H
        m = 0
        for r in range(7):
            row = ((y0 + r) % h_) * w_
            if x0 + 7 <= w_:
                m |= 0x7F << (row + x0)
            else:
                m |= ((1 << (w_ - x0)) - 1) << (row + x0)
                m |= ((1 << (x0 + 7 - w_)) - 1) << row
        return m

    def layers(self, sources, free, k):
        """layers[i] = cells within i steps of any source (through free cells)."""
        out = [sources]
        reach = sources
        for _ in range(k):
            new = self._step(reach, free)
            if new == reach:
                break
            out.append(new)
            reach = new
        return out

    # ------------------------------------------------------------------ memory
    def learn_edges(self, t):
        w_, h_ = self.W, self.H
        x0 = (t.hx - 3) % w_
        y0 = (t.hy - 3) % h_
        ed = t.edges
        kh, ph, kv, pv = self.kh, self.ph, self.kv, self.pv
        old = (kh, ph, kv, pv)
        ids = self.p["portals"] and not self.oracle  # a loaded known map already has every pair
        for r in range(8):  # horizontal edges: north edge of window row r
            line = ed[r]
            if line != DOT7:
                row = ((y0 + r) % h_) * w_
                for c, tok in enumerate(line.split()):
                    if tok != b".":
                        cell = row + (x0 + c) % w_
                        if tok == b"w":
                            kh |= 1 << cell
                        else:
                            ph |= 1 << cell
                            if ids:
                                self.note_portal(int(tok), cell * 2)
        for r in range(7):  # vertical edges: west edge of window column c
            line = ed[8 + r]
            if line != DOT8:
                row = ((y0 + r) % h_) * w_
                for c, tok in enumerate(line.split()):
                    if tok != b".":
                        cell = row + (x0 + c) % w_
                        if tok == b"w":
                            kv |= 1 << cell
                        else:
                            pv |= 1 << cell
                            if ids:
                                self.note_portal(int(tok), cell * 2 + 1)
        if (kh, ph, kv, pv) != old:
            self.kh, self.ph, self.kv, self.pv = kh, ph, kv, pv
            self.okh = self.full ^ (kh | ph)
            self.okv = self.full ^ (kv | pv)

    def update_body(self, hidx, length, own_back):
        body = self.body
        if not body:
            chain = [hidx]
            cur = hidx
            while cur in own_back and len(chain) < length:
                cur = own_back[cur]
                chain.append(cur)
            self.body = body = chain
            own = 0
            for c in chain:
                own |= 1 << c
            self.own = own
        else:
            sprint = self.path and self.path[-1] == hidx and hidx != body[0]
            if sprint:
                # every tile the sprint crossed joins the body, and a later step may re-enter a tile the tail left on
                # an earlier one, so the trimmed tail can share a tile with the new front: rebuild the bits
                for c in self.path:
                    body.insert(0, c)
                del body[length:]
                own = 0
                for c in body:
                    own |= 1 << c
                self.own = own
            elif hidx != body[0]:
                body.insert(0, hidx)
                self.own |= 1 << hidx
            while len(body) > length:
                self.own &= self.full ^ (1 << body.pop())
        self.tail = body[length - 1] if len(body) >= length else None

    def note_portal(self, pid, key):
        """A portal edge seen with id `pid`; the second distinct edge with the same id completes a pair."""
        other = self.pids.get(pid)
        if other is None:
            self.pids[pid] = key
        elif other >= 0 and other != key:
            self.pids[pid] = -1
            self.add_pair(other, key)

    def add_pair(self, k1, k2):
        """Links for the four crossings of a portal pair. Rules: a portal leads out of its partner edge, keeping the
        direction of travel, so crossing eastwards lands on the tile east of the partner (engine-checked in
        tools/test_mapfile.py test_engine_portal_exit)."""
        if (k1 ^ k2) & 1:
            return  # the two edges of a pair always share an orientation
        w_, h_ = self.W, self.H
        for a, b in ((k1, k2), (k2, k1)):
            c, cb = a >> 1, b >> 1
            x, y, bx, by = c % w_, c // w_, cb % w_, cb // w_
            if a & 1:  # west edge of (x, y): crossed eastwards from (x-1, y), westwards from (x, y)
                links = ((y * w_ + (x - 1) % w_, 1, cb), (c, 3, by * w_ + (bx - 1) % w_))
            else:  # north edge of (x, y): crossed southwards from (x, y-1), northwards from (x, y)
                links = ((((y - 1) % h_) * w_ + x, 2, cb), (c, 0, ((by - 1) % h_) * w_ + bx))
            for src, d, dst in links:
                self.link_dst[src * 4 + d] = dst
                self.lmap.setdefault(src, []).append(dst)
                self.lsrc |= 1 << src

    def recognise(self, x0, y0):
        """Narrows bot/known_maps.py to the maps whose kelp and portals agree with every edge seen so far (the window
        at (x0, y0) included); with exactly one left, loads it whole (DEFAULTS "map_oracle")."""
        w_, h_ = self.W, self.H
        if self.cands is None:
            self.cands = known_maps().get((w_, h_), ())
        win = self.window(x0, y0)
        self.hseen |= win | self.window(x0, (y0 + 1) % h_)  # north edges of window rows 0..7
        self.vseen |= win | self.window((x0 + 1) % w_, y0)  # west edges of window columns 0..7
        hs, vs = self.hseen, self.vseen
        self.cands = [c for c in self.cands if (c[1] & hs) == self.kh and (c[2] & vs) == self.kv
                      and (c[3] & hs) == self.ph and (c[4] & vs) == self.pv]
        if len(self.cands) != 1:
            if not self.cands:
                self.oracle = False
            return  # several still fit: look again next turn
        name, kh, kv, ph, pv, pairs, spawns = self.cands[0]
        self.oracle = name
        self.kh, self.kv, self.ph, self.pv = kh, kv, ph, pv
        self.okh = self.full ^ (kh | ph)
        self.okv = self.full ^ (kv | pv)
        self.spawns |= spawns
        if self.p["food_pull"]:
            self.field = known_field(name)
        if self.p["oracle_seen"]:
            self.seen = self.full
        self.link_dst, self.lmap, self.lsrc = {}, {}, 0  # pairs learned so far are among these
        for k1, k2 in pairs:
            self.add_pair(k1, k2)

    # ------------------------------------------------------------------ decision helpers
    def nearby_mates(self, heads, hx, hy, w_, h_, radius):
        """Visible teammate heads (not self, not enemy) within Manhattan `radius` -- free: `heads` is already
        built every turn for head-risk scoring, so this is just a count over a handful of entries."""
        n = 0
        for hc, (enemy, pid) in heads.items():
            if enemy:
                continue
            ex, ey = hc % w_, hc // w_
            if min((ex - hx) % w_, (hx - ex) % w_) + min((ey - hy) % h_, (hy - ey) % h_) <= radius:
                n += 1
        return n

    def want_split(self, t, heads, n_ok, hx, hy, w_, h_):
        """Splitting means standing still this turn and giving up length, so it needs a reason."""
        p = self.p
        length = t.length
        child = p["split_child"]
        if child < 2 or length - child < 2 or t.units >= self.limit or t.rnd >= p["split_r_end"] or self.king:
            return False
        if p["split_len_max"] > 0 and length >= p["split_len_max"]:
            return False  # already a real investment: a length floor alone never stops a big dragon from
            # splitting itself away the instant local conditions (pearls, room, no cap yet) allow it
        if n_ok < p["split_min_exits"]:
            return False  # cornered
        rd = p["split_enemy_dist"]
        for hc, (enemy, _) in heads.items():  # an enemy head close enough to punish a turn spent standing still
            if enemy and min((hc % w_ - hx) % w_, (hx - hc % w_) % w_) + \
                    min((hc // w_ - hy) % h_, (hy - hc // w_) % h_) <= rd:
                return False
        # Local crowding/food: a flat pearl count and team-size cap don't know whether this particular spot
        # already has teammates piling into it (the source of most other-body deaths in a big swarm) or has
        # enough food nearby to feed one more mouth. Both are opt-in (0 = old behaviour, unaffected).
        mates = None
        if p["split_mate_radius"] > 0:
            mates = self.nearby_mates(heads, hx, hy, w_, h_, p["split_mate_radius"])
            if mates >= p["split_mate_cap"]:
                return False  # already crowded here: split somewhere else, or not at all
        if self.founder:  # founders seed the swarm, then grow
            return length >= p["founder_split_len"] and t.units < min(p["founder_units"], self.target)
        if p["grow_mod"] > 0 and self.id % p["grow_mod"] == 0:
            return False  # a designated grower
        if length < p["split_len"] or t.units >= min(p["split_units"], self.target):
            return False
        pearls = t.flags.count(b"1")
        if p["split_food_ratio"] > 0:
            if mates is None:
                mates = self.nearby_mates(heads, hx, hy, w_, h_, p["split_mate_radius"] or 3)
            return pearls >= p["split_food_ratio"] * (mates + 1)
        return pearls >= p["split_pearls"]

    def sprint_paths(self, hx, hy, length, others, mine, heads, pm, vis, smax):
        """Every multi-step move of 2..smax steps that the engine would carry out without killing us, applied the
        way it applies one (rules, "Execution order"): a step after the first must be paid for (a dragon of length
        2 cannot); the destination is checked against our own body as the last step left it, tail included, then
        against other dragons; the head moves, a pearl there is eaten, the tail advances unless it was, and a step
        after the first removes one more tail segment. A portal is crossed only when its pair is known and the
        landing tile is in view. A path whose last step lands on an enemy head is kept as a ram (both die).
        -> [(dirs, cells, length after, eaten pearls bitboard, rammed head cell or -1, own body bits after, tail
        after or -1)]. With the body only partly known (a long dragon's first turns) no tile is ever freed."""
        w_, h_ = self.W, self.H
        kh, kv, ph, pv, link = self.kh, self.kv, self.ph, self.pv, self.link_dst
        exact = len(self.body) >= length
        out = []

        def go(x, y, cell, L, body, bits, dirs, cells, eaten):
            k = len(dirs)
            if k and L <= 2:
                return  # cannot pay for another step
            for d in range(4):
                if d == 0:
                    nx, ny = x, (y - 1) % h_
                    ek, ep = (kh >> cell) & 1, (ph >> cell) & 1
                elif d == 1:
                    nx, ny = (x + 1) % w_, y
                    c2 = ny * w_ + nx
                    ek, ep = (kv >> c2) & 1, (pv >> c2) & 1
                elif d == 2:
                    nx, ny = x, (y + 1) % h_
                    c2 = ny * w_ + nx
                    ek, ep = (kh >> c2) & 1, (ph >> c2) & 1
                else:
                    nx, ny = (x - 1) % w_, y
                    ek, ep = (kv >> cell) & 1, (pv >> cell) & 1
                if ek:
                    continue
                dst = ny * w_ + nx
                if ep:
                    dst = link.get(cell * 4 + d)
                    if dst is None or not (vis >> dst) & 1:
                        continue
                    nx, ny = dst % w_, dst // w_
                b = 1 << dst
                if bits & b:
                    continue
                if others & b:
                    hd = heads.get(dst)
                    if k and hd is not None and hd[0]:
                        out.append((dirs + [d], cells + [dst], L, eaten, dst, bits, -1))
                    continue
                nb = [dst] + body if exact else body
                nbits, nL, ne = bits | b, L, eaten
                if (pm >> dst) & 1 and not (eaten >> dst) & 1:
                    nL += 1
                    ne |= b
                elif exact:
                    nbits &= ~(1 << nb.pop())
                if k:
                    nL -= 1
                    if exact:
                        nbits &= ~(1 << nb.pop())
                nd, nc = dirs + [d], cells + [dst]
                if k:
                    out.append((nd, nc, nL, ne, -1, nbits, nb[-1] if exact else -1))
                if k + 1 < smax:
                    go(nx, ny, dst, nL, nb, nbits, nd, nc, ne)

        body = self.body[:length] if exact else []
        bits = 0
        for c in body:
            bits |= 1 << c
        go(hx, hy, hy * w_ + hx, length, body, bits | mine | (0 if exact else self.own), [], [], 0)
        return out

    def radar_lines(self, echo, hx, hy, vis, occ, parts):
        """Last turn's rays (self.pinged, cast from this very head after our action) decoded against this turn's
        ECHOES counts -> {direction: kind}, kind 0 = kelp (the line is clear up to it, or the ray was lost), 1 ally,
        2 ally head, 3 enemy, 4 enemy head, 5 some dragon (which kind is ambiguous), for every pinged line whose first
        hit is known. A ray whose first hit lies in view is predicted from the view (this turn's, one round younger
        than the ray: if the two disagree only the view is trusted); the rays that leave the view share what the
        counts leave over."""
        w_, h_ = self.W, self.H
        kh, kv, ph, pv, link = self.kh, self.kv, self.ph, self.pv, self.link_dst
        known, hidden, kinds = {}, [], None
        for d in self.pinged:
            x, y = hx, hy
            cell = y * w_ + x
            kind = -1
            for _ in range(8):  # a straight line leaves the 7x7 view after 3 tiles; portals can bring it back
                if d == 0:
                    nx, ny = x, (y - 1) % h_
                    ek, ep = (kh >> cell) & 1, (ph >> cell) & 1
                elif d == 1:
                    nx, ny = (x + 1) % w_, y
                    ek, ep = (kv >> (ny * w_ + nx)) & 1, (pv >> (ny * w_ + nx)) & 1
                elif d == 2:
                    nx, ny = x, (y + 1) % h_
                    ek, ep = (kh >> (ny * w_ + nx)) & 1, (ph >> (ny * w_ + nx)) & 1
                else:
                    nx, ny = (x - 1) % w_, y
                    ek, ep = (kv >> cell) & 1, (pv >> cell) & 1
                if ek:
                    kind = 0
                    break
                nxt = ny * w_ + nx
                if ep:
                    nxt = link.get(cell * 4 + d, -1)
                    if nxt < 0:
                        break  # through a portal we have not mapped: out of sight
                    nx, ny = nxt % w_, nxt // w_
                if not (vis >> nxt) & 1:
                    break
                if (occ >> nxt) & 1:
                    if kinds is None:
                        kinds = {}
                        for q in parts:
                            enemy = q[0] != self.team
                            kinds[int(q[3]) * w_ + int(q[2])] = (4 if enemy else 2) if q[5] == b"1" else \
                                (3 if enemy else 1)
                    kind = kinds.get(nxt, 1)
                    break
                x, y, cell = nx, ny, nxt
            if kind >= 0:
                known[d] = kind
            else:
                hidden.append(d)
        self.rhid = hidden  # (diagnostics) the lines that left the view
        res = list(echo[:5])
        for kind in known.values():
            res[kind] -= 1
        if min(res) < 0 or not hidden:
            return known
        n = sum(res)
        if len(hidden) == 1:
            if n <= 1:
                known[hidden[0]] = res.index(1) if n else 0
        elif res[0] == len(hidden) and n == res[0]:
            for d in hidden:
                known[d] = 0
        elif res[0] == 0 and n == len(hidden):
            kind = next((k for k in range(1, 5) if res[k] == n), 5)
            for d in hidden:
                known[d] = kind
        return known

    def exits(self, tidx, tx, ty, back, free):
        """Free, passable neighbours of cell (tx, ty), not counting the way back."""
        w_, h_ = self.W, self.H
        okh, okv = self.okh, self.okv
        n = 0
        if back != 0 and (free >> (((ty - 1) % h_) * w_ + tx)) & 1 and (okh >> tidx) & 1:
            n += 1
        c = ty * w_ + (tx + 1) % w_
        if back != 1 and (free >> c) & 1 and (okv >> c) & 1:
            n += 1
        c = ((ty + 1) % h_) * w_ + tx
        if back != 2 and (free >> c) & 1 and (okh >> c) & 1:
            n += 1
        if back != 3 and (free >> (ty * w_ + (tx - 1) % w_)) & 1 and (okv >> tidx) & 1:
            n += 1
        if (self.lsrc >> tidx) & 1:  # known portals out of this cell (the way back through one lands on our head)
            for dst in self.lmap[tidx]:
                n += (free >> dst) & 1
        return n

    def territory(self, mine0, foes, free, radius):
        """Cells I reach strictly first minus cells the enemy heads reach strictly first (simultaneous dilation)."""
        free = free | mine0 | foes
        m, e = mine0, foes
        step = self._step
        for _ in range(radius):
            nm = step(m, free) & (self.full ^ e)
            ne = step(e, free) & (self.full ^ m)
            tie = nm & ne  # reached by both this step: nobody's
            m2 = m | (nm & (self.full ^ tie))
            e2 = e | (ne & (self.full ^ tie))
            if m2 == m and e2 == e:
                break
            m, e = m2, e2
        return m.bit_count() - e.bit_count()

    def safe_moves(self, ex, ey, occ, extra):
        """How many of an enemy head's moves would not kill a dragon that only looks at its four neighbours."""
        w_, h_ = self.W, self.H
        okh, okv = self.okh, self.okv
        e = ey * w_ + ex
        cs = (((ey - 1) % h_) * w_ + ex, ey * w_ + (ex + 1) % w_, ((ey + 1) % h_) * w_ + ex, ey * w_ + (ex - 1) % w_)
        passable = ((okh >> e) & 1, (okv >> cs[1]) & 1, (okh >> cs[2]) & 1, (okv >> e) & 1)
        n = 0
        for d in range(4):
            if passable[d] and not (occ >> cs[d]) & 1 and cs[d] != extra:
                n += 1
        return n

    # ------------------------------------------------------------------ decision
    def decide(self, t):
        p = self.p
        w_, h_ = self.W, self.H
        hx, hy = t.hx, t.hy
        hidx = hy * w_ + hx
        length = t.length
        self.learn_edges(t)
        x0 = (hx - 3) % w_
        y0 = (hy - 3) % h_
        vis = self.window(x0, y0)
        self.seen |= vis
        if self.oracle is None and p["map_oracle"] and p["portals"]:
            self.recognise(x0, y0)
        if p["dive_len"] and p["dive_scope"] == 2:  # before any early return, so no turn breaks the two-turn test
            cds, c1 = t.cds, 0
            for i in range(49):
                if cds[i] == b"1":
                    c1 |= 1 << (((y0 + WIN_R[i]) % h_) * w_ + (x0 + WIN_C[i]) % w_)
            new = c1 & self.cd1
            while new:
                low = new & -new
                c = low.bit_length() - 1
                if (c % w_, c // w_) not in self.founts:
                    self.founts.append((c % w_, c // w_))
                new ^= low
            self.cd1 = c1
        if self.king is None:  # before any early return, so a boxed-in first turn still decides it
            self.king = t.rnd == 0 and ((p["king_first"] > 0 and self.id < p["king_first"])
                                        or (p["king_len"] > 0 and length >= p["king_len"]))

        # visible dragons
        occ = 0
        mine = 0  # our own visible segments
        heads = {}  # cell -> (is_enemy, dragon id) for heads other than ours
        segs = {}  # dragon id -> visible segment count
        own_back = {}  # head-ward neighbour cell -> own segment cell (only needed to seed the body)
        enemy_facing = {}  # enemy head cell -> its own reported facing (free: q[4] on every part, usually unread)
        me, team = self.id, self.team
        seed = not self.body
        for q in t.parts:
            pid = int(q[1])
            x = int(q[2])
            y = int(q[3])
            cell = y * w_ + x
            occ |= 1 << cell
            segs[pid] = segs.get(pid, 0) + 1
            if pid == me:
                mine |= 1 << cell
            if q[5] == b"1":
                if pid != me:
                    enemy = q[0] != team
                    heads[cell] = (enemy, pid)
                    if enemy:
                        enemy_facing[cell] = LETTERS.find(q[4])
            elif seed and pid == me:
                f = LETTERS.find(q[4])
                own_back[((y + DY[f]) % h_) * w_ + (x + DX[f]) % w_] = cell
        self.update_body(hidx, length, own_back)
        self.path = None

        # pearls in view; with portals on, also remembered (with their countdowns) for when they are out of view
        pm = 0
        fl = t.flags
        portals = p["portals"]
        if portals:
            cds, due, rnd = t.cds, self.due, t.rnd
            for i in range(49):
                c = ((y0 + WIN_R[i]) % h_) * w_ + (x0 + WIN_C[i]) % w_
                if fl[i] == b"1":
                    pm |= 1 << c
                if cds[i] != b"-1":
                    due[c] = rnd + int(cds[i])
                    self.spawns |= 1 << c
            self.pmem = (self.pmem & (self.full ^ vis)) | pm
            if self.exploring:  # first look at the far side of the unknown portal we just stepped into
                self.exploring = False
                if p["portal_learn"]:
                    # the pocket we landed in, portals aside: an empty walled-in pocket (Autarky's boxes, whose view
                    # still shows spawn tiles in the field beyond the walls), not a bare patch of open ground (a child
                    # leaving a Portals pearl room lands in a field with no spawn tile in view)
                    lsrc, self.lsrc = self.lsrc, 0
                    pocket = self.layers(1 << hidx, self.seen, 16)[-1]
                    self.lsrc = lsrc
                    if pocket.bit_count() < 16 and not pocket & self.spawns:
                        self.explore_ok = False
        else:
            for i in range(49):
                if fl[i] == b"1":
                    pm |= 1 << (((y0 + WIN_R[i]) % h_) * w_ + (x0 + WIN_C[i]) % w_)

        # legality of the four moves, exact
        nidx = (((hy - 1) % h_) * w_ + hx, hy * w_ + (hx + 1) % w_, ((hy + 1) % h_) * w_ + hx, hy * w_ + (hx - 1) % w_)
        kh, kv, ph, pv = self.kh, self.kv, self.ph, self.pv
        kelp = ((kh >> hidx) & 1, (kv >> nidx[1]) & 1, (kh >> nidx[2]) & 1, (kv >> hidx) & 1)
        port = ((ph >> hidx) & 1, (pv >> nidx[1]) & 1, (ph >> nidx[2]) & 1, (pv >> hidx) & 1)
        status = [OK, OK, OK, OK]
        dest = list(nidx)  # landing cell per direction: a known portal lands beyond its partner edge
        blind = [False] * 4  # known portal whose landing cell is out of view: its occupancy is a guess
        unknown = []  # portals whose far side is still unknown: legal, but nothing about the landing is known
        for d in range(4):
            if kelp[d]:
                status[d] = DEAD
            elif port[d]:
                status[d] = PORTAL
                if portals:
                    dst = self.link_dst.get(hidx * 4 + d)
                    if dst is None:
                        # facing is the direction from our neck to our head, so stepping back against it lands on the
                        # neck, through a portal as anywhere else: a split child whose body straddles a portal it has
                        # never seen the far side of died this way (58 in 4 games on Portals)
                        if d != (t.dir + 2) % 4:
                            unknown.append(d)
                        continue
                    dest[d] = dst
                    if (vis >> dst) & 1:
                        hd = heads.get(dst)
                        status[d] = OK if not (occ >> dst) & 1 else \
                            DEAD if hd is None else (TRADE_ENEMY if hd[0] else TRADE_TEAM)
                    elif (self.own >> dst) & 1:
                        status[d] = DEAD  # our own trailing body, out of view on the other side
                    else:
                        status[d] = OK
                        blind[d] = True
            elif (occ >> nidx[d]) & 1:
                hd = heads.get(nidx[d])
                status[d] = DEAD if hd is None else (TRADE_ENEMY if hd[0] else TRADE_TEAM)
        ok = [d for d in range(4) if status[d] == OK]
        rams = []  # head-on trades a short dragon may take: onto an enemy head at least as long as it looks
        if p["ram_len"] and length <= p["ram_len"]:
            rams = [d for d in range(4) if status[d] == TRADE_ENEMY and segs.get(heads[dest[d]][1], 1) >= length]
        if self.debug:
            self.dbg = {"rnd": t.rnd, "pos": (hx, hy), "dir": t.dir, "len": length, "status": list(status),
                        "kelp": kelp, "port": port, "body": list(self.body[:8]), "tail": self.tail,
                        "parts": [tuple(q) for q in t.parts][:12], "dest": list(dest), "blind": list(blind),
                        "unknown": list(unknown)}
        if not ok:
            # Boxed in: every move is kelp, a body, a head-on trade or an unseen portal exit. A split is not a move,
            # so the parent stays put and survives the turn (the child takes the rear segments, facing away), which
            # beats any of those -- and it is how a long dragon spawned in a pocket gets out (see "king_first").
            child = p["split_child"]
            # near the unit limit, short dragons leave the last boxed_reserve slots to long ones: every boxed-in long
            # dragon that died in local mh2 games did so with the team at the 64-dragon cap, where no split is legal
            cap = self.limit - (p["boxed_reserve"] if length < p["boxed_long"] else 0)
            if p["boxed_split"] and child >= 2 and length - child >= 2 and t.units < cap:
                # rear split: leave 2 segments at the boxed head and hand the rest to the child, which leaves from the
                # old tail facing away, so a long dragon loses 2 once instead of 2 every round it stays boxed in
                rear = p["boxed_rear_len"] and length >= p["boxed_rear_len"] and t.rnd >= p["boxed_rear_r"]
                # late on, only a rear split (which keeps the body) is worth another dragon: a short one boxed in
                # dies where it is and its teammates eat it, so the swarm shrinks into fewer, longer dragons
                if rear or not p["boxed_r_end"] or t.rnd < p["boxed_r_end"]:
                    return b"SPLIT %d\n" % (length - 2 if rear else child)
            if p["spare_team"]:
                best = min(range(4), key=lambda d: (SPARE_RANK[status[d]], d != t.dir))
            else:
                best = min(range(4), key=lambda d: (status[d], d != t.dir))
            return MOVES[best]

        # prediction: a visible enemy head's own reported facing is a free, zero-cost, always-current guess at
        # where it goes next -- unlike sonar, no protocol needed, and it targets exactly the gap head_risk's
        # exact-adjacency check misses: a higher-id enemy hasn't moved yet this round, so head_risk is reacting to
        # its stale pre-move position, not where it is about to end up (see the plan notes: this is believed to be
        # a real share of head-on deaths, the largest cause by far).
        predicted = []  # [(x, y), ...] guessed next cells of visible enemy heads
        if p["predict"] > 0 and enemy_facing:
            for hc, f in enemy_facing.items():
                if f < 0:
                    continue
                ex, ey = hc % w_, hc // w_
                predicted.append(((ex + DX[f]) % w_, (ey + DY[f]) % h_))
        if self.debug:
            self.dbg["predicted"] = list(predicted)

        # sonar: broadcast the nearest visible enemy, own facing included (cheap: no flood fill), and decode
        # anything received this turn into the same kind of one-step guess "predicted" above makes
        sonar_msg, sonar_predicted = None, []  # sonar_predicted: [(x, y), ...] guessed cells from relayed sightings
        if p["sonar_predict"] > 0:
            if heads:
                best_e, best_dist = None, 1 << 30
                for hc, (enemy, pid) in heads.items():
                    if enemy:
                        ex, ey = hc % w_, hc // w_
                        dist = min((ex - hx) % w_, (hx - ex) % w_) + min((ey - hy) % h_, (hy - ey) % h_)
                        if dist < best_dist:
                            best_dist, best_e = dist, (ex, ey, segs.get(pid, 1), enemy_facing.get(hc, 0))
                if best_e is not None:
                    sonar_msg = proto.pack_enemy_sonar(*best_e)
            for m in t.msgs:  # untrusted: could be an enemy's sonar, or a stray value that happens to decode
                kind, mx, my, _, mf = proto.unpack_sonar(m)
                if kind == proto.SONAR_ENEMY and mx < w_ and my < h_:
                    sonar_predicted.append(((mx + DX[mf]) % w_, (my + DY[mf]) % h_))
        if self.debug:
            self.dbg["sonar_predicted"] = list(sonar_predicted)

        # sonar (terrain): unlike an enemy sighting, a learned kelp/portal edge never goes stale, so it is worth
        # relaying even with nobody obviously listening -- the payoff is a teammate (often a freshly split child
        # with none of this dragon's accumulated map memory) learning it for free instead of finding out by
        # stepping on it. Sends one of the current tile's own known edges (zero extra cost: `kelp`/`port` above
        # are already computed for this turn's legality check); merges any received fact straight into kh/kv/ph/pv,
        # exactly as learn_edges() would have, so it improves this dragon's own legality/trap checks immediately.
        if p["sonar_terrain"] > 0:
            if sonar_msg is None:
                for d in range(4):
                    if kelp[d] or port[d]:
                        tx_, ty_ = TERRAIN_EDGE_TILE[d](hx, hy, w_, h_)
                        sonar_msg = proto.pack_terrain_sonar(tx_, ty_, TERRAIN_EDGE_VERTICAL[d], 1 if port[d] else 0)
                        break
            learned = False
            for m in t.msgs:
                kind, mx, my, vertical, portal = proto.unpack_sonar(m)
                if kind == proto.SONAR_TERRAIN and mx < w_ and my < h_:
                    bit = 1 << (my * w_ + mx)
                    if portal:
                        self.pv |= bit if vertical else 0
                        self.ph |= bit if not vertical else 0
                    else:
                        self.kv |= bit if vertical else 0
                        self.kh |= bit if not vertical else 0
                    learned = True
            if learned:
                self.okh = self.full ^ (self.kh | self.ph)
                self.okv = self.full ^ (self.kv | self.pv)

        if self.founder is None:
            self.founder = t.rnd == 0
        if p["rally_r"] and t.rnd >= p["rally_r"] - p["rally_age"]:
            age = p["rally_age"]
            ki = self.kinfo if self.kinfo is not None and t.rnd - self.kinfo[3] <= age else None
            for m in t.msgs:  # untrusted: an enemy value that passes the check is at worst one bad hint
                kind, mx, my, ml, mr = proto.unpack_sonar(m)
                if kind == proto.SONAR_KING and mx < w_ and my < h_ and t.rnd - age <= mr <= t.rnd \
                        and (ki is None or (ml, mr) > (ki[2], ki[3])):
                    ki = (mx, my, ml, mr)
            # a dragon near the longest heard of claims it too (its own older, longer record may still be about)
            if length >= p["feed_min"] and (ki is None or 5 * length >= 4 * ki[2]):
                ki = (hx, hy, length, t.rnd)
            self.kinfo = ki
            if ki is not None:
                self.king_msg = proto.pack_king_sonar(*ki)
        if p["feed_r"] and t.rnd >= p["feed_r"] and length <= p["feed_len"]:
            fed, rd = False, p["feed_dist"]
            for hc, (enemy, pid) in heads.items():
                ex, ey = hc % w_, hc // w_
                dist = min((ex - hx) % w_, (hx - ex) % w_) + min((ey - hy) % h_, (hy - ey) % h_)
                if enemy and dist <= rd + 1:
                    fed = False  # an enemy would get to the pearls as soon as the teammate
                    break
                if not enemy and dist <= rd and segs.get(pid, 1) >= max(p["feed_min"], p["feed_ratio"] * length):
                    fed = True
            if fed:
                return b"SPLIT 1\n"
        if self.want_split(t, heads, len(ok), hx, hy, w_, h_):
            action = b"SPLIT %d\n" % p["split_child"]
            return action + b"SONAR %d\n" % sonar_msg if sonar_msg is not None else action
        # with one legal move a sprint can still ram, or get clear of an enemy's sprint reach
        sprinting = p["sprint_max"] > 1 and (p["sprint_ram"] > 0 or p["sprint_threat"] > 0) and \
            any(e for e, _ in heads.values())
        if (len(ok) + len(unknown) == 1 and not rams and not sprinting) or self.over():
            if self.debug:
                self.dbg["risky"] = blind[ok[0]]
            action = MOVES[ok[0]]
            return action + b"SONAR %d\n" % sonar_msg if sonar_msg is not None else action

        free = self.full ^ (occ | self.own)
        lay = None
        kk = p["pearl_k"]
        if pm and not self.over():
            lay = self.layers(pm, free, kk)
        care = 1.0
        margin = p["need_margin"]
        if p["grow_care_len"] > 0 and length >= p["grow_care_len"]:
            care = p["grow_care_mult"]
            margin += p["grow_care_margin"]
        if self.king:
            care = max(care, p["king_care"])
        need = min(max(length + margin, p["need_floor"]), p["need_cap"])
        tail = self.tail
        free_trap = free & self.seen if p["pessimistic"] else free
        dive = length <= p["dive_len"] and t.units >= p["dive_units"]
        lines = self.lines = self.radar_lines(t.echo, hx, hy, vis, occ, t.parts) \
            if p["radar"] and t.echo is not None and self.pinged else {}
        if dive and length >= 4:
            # a dragon this long can leave a dead end by the boxed split (with boxed_rear_r 0 a rear split: a U-turn
            # that costs a 2-long stub), so it dives only while that split is still legal
            dive = t.units < self.limit - (p["boxed_reserve"] if length < p["boxed_long"] else 0)
        if dive and p["dive_scope"] == 2:
            r = p["dive_radius"]
            dive = any(min((fx - hx) % w_, (hx - fx) % w_) + min((fy - hy) % h_, (hy - fy) % h_) <= r
                       for fx, fy in self.founts)

        # enemy heads close enough to squeeze: (cell, area they need, how short of it they already are)
        foes = []
        if p["deny"] > 0 and not self.over():
            for hc, (enemy, pid) in heads.items():
                if not enemy:
                    continue
                ex, ey = hc % w_, hc // w_
                if min((ex - hx) % w_, (hx - ex) % w_) + min((ey - hy) % h_, (hy - ey) % h_) <= p["deny_radius"]:
                    need_e = segs.get(pid, 1) + p["deny_margin"]
                    base = self.flood(free | (1 << hc), hc, need_e)
                    foes.append((hc, need_e, need_e - base if base < need_e else 0))

        # enemy heads to squeeze: (x, y, safe moves they have now)
        sq = []
        if p["squeeze"] > 0:
            for hc, (enemy, pid) in heads.items():
                if enemy:
                    ex, ey = hc % w_, hc // w_
                    if min((ex - hx) % w_, (hx - ex) % w_) + min((ey - hy) % h_, (hy - ey) % h_) <= p["deny_radius"]:
                        sq.append((ex, ey, self.safe_moves(ex, ey, occ, -1)))

        # tiles an enemy head can sprint-ram this turn beyond its four neighbours (those are head_risk's): through free
        # tiles for up to min(3, length - 1) steps, the last one onto the target -- only enemies a trade hurts us against
        threat = 0
        if p["sprint_threat"] > 0 and length >= p["sprint_threat_len"] and not self.over():
            freeo = self.full ^ occ
            step = self._step
            for hc, (enemy, pid) in heads.items():
                e_len = segs.get(pid, 1)
                if not enemy or e_len < 3 or length < e_len * p["trade_ratio"]:
                    continue
                near = step(1 << hc, self.full)
                reach = self.layers(1 << hc, freeo, min(3, e_len - 1) - 1)[-1]
                threat |= step(reach, self.full) & ~(near | (1 << hc))

        foe_heads = 0  # bitboard of enemy heads close enough to contest territory
        if p["voro"] > 0:
            for hc, (enemy, pid) in heads.items():
                if enemy:
                    ex, ey = hc % w_, hc // w_
                    if min((ex - hx) % w_, (hx - ex) % w_) + min((ey - hy) % h_, (hy - ey) % h_) <= p["voro_radius"]:
                        foe_heads |= 1 << hc

        fld = self.field if lay is None and not pm else None  # food_pull: only with no pearl in view
        if fld is not None:
            fpull, fhere = p["food_pull"] / 8, fld[hidx]
        kx = ky = None  # rally: the king's head to close in on
        if p["rally_r"] and t.rnd >= p["rally_r"] and length <= p["feed_len"] and lay is None and not pm \
                and self.kinfo is not None and self.kinfo[2] > length:
            kx, ky = self.kinfo[0], self.kinfo[1]
            kd = min((kx - hx) % w_, (hx - kx) % w_) + min((ky - hy) % h_, (hy - ky) % h_)
        best_d, best_s = ok[0], -1e18
        for d in ok:
            tidx = dest[d]
            tx = tidx % w_
            ty = tidx // w_
            s = 0.0
            if fld is not None:
                s += fpull * (fld[tidx] - fhere)
            if kx is not None:
                s += p["rally_pull"] * (kd - min((kx - tx) % w_, (tx - kx) % w_) - min((ky - ty) % h_, (ty - ky) % h_))
            on_pearl = (pm >> tidx) & 1
            if blind[d]:  # out of view beyond a known portal: its pearl as remembered, or due by its countdown, or
                # (a known map's spawn tile never seen yet) presumed there
                s -= p["portal_blind"]
                on_pearl = (self.pmem >> tidx) & 1 or self.due.get(tidx, 1 << 30) <= t.rnd or \
                    ((self.spawns >> tidx) & 1 and tidx not in self.due)
            if on_pearl:
                s += p["pearl_here"]
            elif lay is not None:
                for i in range(1, len(lay)):
                    if (lay[i] >> tidx) & 1:
                        s += p["pearl_near"] * (kk + 1 - i)
                        break
            elif p["explore"] > 0 and not (self.seen >> tidx) & 1:
                # no pearl signal at all nearby: with nothing else pulling a direction, "straight" alone lets a
                # dragon loop back through ground it has already searched empty. Reward pushing into unseen
                # territory instead -- spreads the swarm outward (more board claimed sooner) rather than pulling
                # everyone toward one shared point, which would just trade circling for a new collision magnet.
                s += p["explore"]
            if not self.over():
                fr = free_trap | (1 << tidx)
                if tail is not None and not on_pearl:
                    fr |= 1 << tail  # the tail cell is vacated by this move
                area = self.flood(fr, tidx, need)
                if area < need:
                    tm = care
                    guarded = p["radar_dive"] and lines.get(d, 0) >= 1  # a dragon down that line: occupied corridor
                    if dive and not guarded and (p["dive_scope"] != 1 or (on_pearl and (not p["dive_fountain"]
                                                                                        or t.cds[NB_WIN[d]] == b"1"))):
                        tm *= p["dive_trap"]
                    s -= p["trap"] * tm * (need - area) / need
                else:
                    s += p["area"]
                if p["dead_end"] and self.exits(tidx, tx, ty, (d + 2) % 4, free) < 2:
                    s -= p["dead_end"] * care
                for hc, need_e, base_short in foes:  # herding: leave visible enemies less room than they need
                    left = self.flood((free ^ (1 << tidx)) | (1 << hc), hc, need_e)
                    short = need_e - left if left < need_e else 0
                    if short > base_short:
                        s += p["deny"] * (short - base_short) / need_e
            if foe_heads and not self.over():
                s += p["voro"] * self.territory(1 << tidx, foe_heads, free, p["voro_radius"])
            for ex, ey, before in sq:
                after = self.safe_moves(ex, ey, occ, tidx)
                if after < before:  # fewer exits (0 = boxed in: it dies next turn)
                    s += p["squeeze"] * (before - after) * (3 - min(after, 2))
            for hc, (enemy, pid) in heads.items():
                ex = hc % w_
                ey = hc // w_
                if (ey == ty and (ex - tx) % w_ in (1, w_ - 1)) or (ex == tx and (ey - ty) % h_ in (1, h_ - 1)):
                    if enemy:
                        small = length < segs.get(pid, 1) * p["trade_ratio"]
                        s -= p["head_risk_small"] if small else p["head_risk"] * care
                    else:
                        s -= p["team_head_risk"] * care
            if (threat >> tidx) & 1:
                s -= p["sprint_threat"] * p["head_risk"] * care
            for px, py in predicted:  # a guess, not a fact: landing on or next to it is merely made less attractive
                if (py == ty and (px - tx) % w_ in (1, w_ - 1)) or (px == tx and (py - ty) % h_ in (1, h_ - 1)):
                    s -= p["predict"]
            for px, py in sonar_predicted:  # same idea, relayed: reaches an enemy this dragon never saw itself
                if (py == ty and (px - tx) % w_ in (1, w_ - 1)) or (px == tx and (py - ty) % h_ in (1, h_ - 1)):
                    s -= p["sonar_predict"]
            if d == t.dir:
                s += p["straight"]
            if lines.get(d) == 4:
                s -= p["radar_head"]
            if s > best_s:
                best_d, best_s = d, s
        for d in unknown:  # nothing is known beyond it until we step through: a flat bet on what lies there
            bet = p["portal_unknown"] if self.explore_ok and not (p["portal_hungry"] and lay) else -1e9
            s = bet + (p["straight"] if d == t.dir else 0.0)
            if s > best_s:
                best_d, best_s = d, s
        self.exploring = best_d in unknown
        if self.debug:
            self.dbg["risky"] = self.exploring or blind[best_d]
        action = MOVES[best_d]

        # sprints (DEFAULTS "sprint_max"): every legal path of 2..sprint_max steps, scored at its end tile on the same
        # core terms as a plain move (pearl field, trap room at the length left, dead end, adjacent heads, sprint
        # threat) plus the pearls eaten on the way, minus the segments spent. The flood fill runs only for a path
        # whose best case still beats the best move so far.
        path, sram = None, None  # sram: ((enemy segments, -steps), dirs) of the best sprint ram
        if p["sprint_max"] > 1 and not self.over():
            others = occ ^ mine
            for dirs, cells, L2, eaten, ram, bits, tl in self.sprint_paths(hx, hy, length, others, mine, heads, pm,
                                                                          vis, p["sprint_max"]):
                k = len(dirs)
                if ram >= 0:
                    e_segs = segs.get(heads[ram][1], 1)
                    if p["sprint_ram"] > 0 and length <= p["ram_len"] and e_segs >= length + p["sprint_ram_margin"] \
                            and t.rnd >= p["sprint_ram_r"]:
                        if sram is None or (e_segs, -k) > sram[0]:
                            sram = ((e_segs, -k), dirs)
                    continue
                tidx = cells[-1]
                tx, ty = tidx % w_, tidx // w_
                s = p["pearl_here"] * eaten.bit_count() - p["sprint_seg"] * (k - 1)
                if p["sprint_contest"] > 0 and eaten:
                    for c in cells:
                        if (eaten >> c) & 1:
                            cx, cy = c % w_, c // w_
                            if any(e and min((hc % w_ - cx) % w_, (cx - hc % w_) % w_)
                                   + min((hc // w_ - cy) % h_, (cy - hc // w_) % h_) <= 2
                                   for hc, (e, _) in heads.items()):
                                s += p["sprint_contest"]
                for hc, (enemy, pid) in heads.items():
                    ex, ey = hc % w_, hc // w_
                    if (ey == ty and (ex - tx) % w_ in (1, w_ - 1)) or (ex == tx and (ey - ty) % h_ in (1, h_ - 1)):
                        if enemy:
                            small = length < segs.get(pid, 1) * p["trade_ratio"]
                            s -= p["head_risk_small"] if small else p["head_risk"] * care
                        else:
                            s -= p["team_head_risk"] * care
                if (threat >> tidx) & 1:
                    s -= p["sprint_threat"] * p["head_risk"] * care
                if all(d == t.dir for d in dirs):
                    s += p["straight"]
                if s + p["pearl_near"] * kk + p["area"] <= best_s:
                    continue  # cannot win even with the best pearl field and no trap
                if self.over():
                    break
                free2 = self.full ^ (others | bits)
                rest = pm & ~eaten
                if rest:
                    lay2 = lay if not eaten and lay is not None else self.layers(rest, free2, kk)
                    for i in range(1, len(lay2)):
                        if (lay2[i] >> tidx) & 1:
                            s += p["pearl_near"] * (kk + 1 - i)
                            break
                need2 = min(max(L2 + margin, p["need_floor"]), p["need_cap"])
                fr = (free2 & self.seen if p["pessimistic"] else free2) | (1 << tidx)
                if tl >= 0:
                    fr |= 1 << tl  # vacated by the next move
                area = self.flood(fr, tidx, need2)
                if area < need2:
                    s -= p["trap"] * care * (need2 - area) / need2
                else:
                    s += p["area"]
                if p["dead_end"] and self.exits(tidx, tx, ty, (dirs[-1] + 2) % 4, free2) < 2:
                    s -= p["dead_end"] * care
                if s > best_s:
                    best_s, path = s, (dirs, cells)
        if path is not None:
            action = b"MOVE " + bytes(LETTERS[d] for d in path[0]) + b"\n"
            self.path = path[1]
            self.exploring = False

        if rams and p["ram"] > best_s:
            action = MOVES[rams[0]]
            self.path = None
        elif sram is not None and p["sprint_ram"] > best_s:
            action = b"MOVE " + bytes(LETTERS[d] for d in sram[1]) + b"\n"
            self.path = None
        if self.debug:
            self.dbg["sprint"] = path
            self.dbg["sram"] = sram
        return action + b"SONAR %d\n" % sonar_msg if sonar_msg is not None else action

    # ------------------------------------------------------------------ fallback
    def fallback(self, block):
        """Independent of learned state: first move that is not immediate death, preferring the heading.
        Portal edges (see decide()'s PORTAL status / module docstring) skip the occupancy check entirely -- a
        portal step does not land on the physically-adjacent tile, so that tile's occupancy is irrelevant, not
        "safe" or "unsafe" (this was a bug here: the occupancy check ran unconditionally, so an unrelated body on
        that irrelevant tile could wrongly rule out a portal move that decide()'s own legality check would allow)."""
        try:
            t = proto.parse_turn(block)
            occ = {(int(q[2]), int(q[3])) for q in t.parts}
            ed = t.edges
            vt = ed[11].split()
            tok = (ed[3].split()[3], vt[4], ed[4].split()[3], vt[3])
            kelp = tuple(x == b"w" for x in tok)
            portal = tuple(x != b"." and x != b"w" for x in tok)
            order = [t.dir] + [d for d in range(4) if d != t.dir] if t.dir >= 0 else range(4)
            for d in order:
                if kelp[d]:
                    continue
                if not portal[d] and ((t.hx + DX[d]) % self.W, (t.hy + DY[d]) % self.H) in occ:
                    continue
                return MOVES[d]
        except Exception:  # noqa: BLE001
            pass
        return MOVES[0]
