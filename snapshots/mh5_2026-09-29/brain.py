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
    # Sprinting: MOVE with a direction letter repeated `steps` times covers `steps` tiles in one turn instead of
    # 1, costing `steps - 1` body segments (the engine's own rule -- see the "Sprinting" rules section). 1 = off
    # (never sprint). Unlike the RL policy, which only legality-checks a sprint's first step, want_sprint below
    # walks the whole path with the same kh/kv/ph/pv edges a plain move already uses, so a sprint here is exactly
    # as safe as a plain move, never a blind gamble on steps 2-3. Untuned.
    "sprint_max": 1,
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
    # Top-team style, from 450 replays of the ladder's top 3 (Cutlery, cheji bt, forgot to mention; tools/scout.py,
    # 29 Sep). All three stop ordinary splits around round 300, so attrition shrinks the swarm (cheji: 45 dragons at
    # round 300, 7 at 400) while the survivors eat the corpses: 57-75% of their longest dragon's pearls after round
    # 300 are their own dead, and it ends 32-43 long against our 10.5 -- though we hold more total length, in 33
    # dragons. They also ram enemy heads with 2-3-long dragons, 22-25 times a game, more often than they are rammed.
    # "topstyle" = split_r_end 300, boxed_r_end 300, boxed_rear_r 0, ram_len 3, ram 1000, dive_len 3, dive_trap 0,
    # dive_scope 0. On the 1.x engine, the 10 ladder maps x held-out seeds 9,000,000+ x both seats (200 games per
    # opponent, tools/bench1x.py): 75.5% [69-81] vs mh3, 73.0% vs farmer, 58.2% vs portal_farmer; longest dragon at
    # round 500 15.2 vs mh3's 11.0. Screened on seeds 1-6 (120 games vs mh3): each part alone 51-64%, round 300 beat
    # 250 and 350, ram 300-3000 all alike; a "feed" mechanism (short dragons dying next to a long teammate) added nothing.
    "boxed_r_end": 0,          # from this round on a boxed-in dragon splits only for a rear split (0 = off)
    "ram_len": 0,              # dragons at most this long may move onto an adjacent enemy head, killing both (0 = off)
    "ram": 0.0,                # ... when that enemy shows at least as many segments: a move scored at this
    # Feeding the king: from feed_r on, a dragon at most feed_len long whose head is within feed_dist of the head of a
    # teammate showing at least max(feed_min, feed_ratio x its length) segments, with no enemy head that close, dies on
    # the spot (an illegal SPLIT 1, as the ladder's Cutlery does) so the long one eats the pearls it drops. 0 = off.
    # See bot/brain.py for the scouting behind it and the bench numbers.
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
# mh4 (29 Sep): mh3 + portals (learned pairs, known-map oracle, oracle_seen) + topstyle (stop splitting at 300,
# turnaround splits all game, short dragons ram) + dragons of any length diving near fountains they have seen,
# leaving dead ends by the turnaround split. mh5 (29 Sep): mh4 + feeding the king from round 250 (untuned settings).
# See snapshots/README.md for the bench numbers.
DEFAULTS.update({"portals": 1, "map_oracle": 1, "oracle_seen": 1,
                 "split_r_end": 300, "boxed_r_end": 300, "boxed_rear_r": 0, "ram_len": 3, "ram": 1000.0,
                 "dive_len": 99, "dive_trap": 0.0, "dive_scope": 2, "dive_radius": 6,
                 "feed_r": 250, "feed_len": 8, "feed_dist": 3, "feed_min": 8, "feed_ratio": 1.5})


def known_maps():
    """MAPS from the known_maps.py beside this file (not via sys.path, so a snapshot loaded from another folder
    reads its own copy); {} if there is none."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "known_maps.py")
    try:
        with open(path) as f:
            ns = {}
            exec(f.read(), ns)
        return ns["MAPS"]
    except OSError:
        return {}


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
        try:
            action = self.decide(proto.parse_turn(block))
        except Exception:  # noqa: BLE001 - a crash would kill the dragon
            if self.strict:
                raise
            self.errors += 1
            return self.fallback(block)
        if self.debug:
            self.dbg["act"] = action
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
            if hidx != body[0]:
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
        if n_ok < p["split_min_exits"] or any(e for e, _ in heads.values()):
            return False  # cornered, or an enemy head is in view
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

    def sprint_end(self, hx, hy, d, steps, occ, w_, h_):
        """Final cell of a `steps`-tile straight sprint in direction d, if every edge and cell along the way is
        clear (no kelp, no portal -- unlike a single plain move a sprint's far end is never checked by the engine
        beyond legality, so treating a portal as a wall here, same as elsewhere, is the only safe choice -- and
        no dragon part). None if any step is blocked."""
        kh, kv, ph, pv = self.kh, self.kv, self.ph, self.pv
        x, y = hx, hy
        for _ in range(steps):
            nx, ny = (x + DX[d]) % w_, (y + DY[d]) % h_
            idx = ny * w_ + nx
            if d == 0:
                blocked = (kh >> (y * w_ + x)) & 1 or (ph >> (y * w_ + x)) & 1
            elif d == 1:
                blocked = (kv >> idx) & 1 or (pv >> idx) & 1
            elif d == 2:
                blocked = (kh >> idx) & 1 or (ph >> idx) & 1
            else:
                blocked = (kv >> (y * w_ + x)) & 1 or (pv >> (y * w_ + x)) & 1
            if blocked or (occ >> idx) & 1:
                return None
            x, y = nx, ny
        return y * w_ + x

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
        if (len(ok) + len(unknown) == 1 and not rams) or self.over():
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

        foe_heads = 0  # bitboard of enemy heads close enough to contest territory
        if p["voro"] > 0:
            for hc, (enemy, pid) in heads.items():
                if enemy:
                    ex, ey = hc % w_, hc // w_
                    if min((ex - hx) % w_, (hx - ex) % w_) + min((ey - hy) % h_, (hy - ey) % h_) <= p["voro_radius"]:
                        foe_heads |= 1 << hc

        best_d, best_s = ok[0], -1e18
        for d in ok:
            tidx = dest[d]
            tx = tidx % w_
            ty = tidx // w_
            s = 0.0
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
                    if dive and (p["dive_scope"] != 1 or (on_pearl and (not p["dive_fountain"]
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
            for px, py in predicted:  # a guess, not a fact: landing on or next to it is merely made less attractive
                if (py == ty and (px - tx) % w_ in (1, w_ - 1)) or (px == tx and (py - ty) % h_ in (1, h_ - 1)):
                    s -= p["predict"]
            for px, py in sonar_predicted:  # same idea, relayed: reaches an enemy this dragon never saw itself
                if (py == ty and (px - tx) % w_ in (1, w_ - 1)) or (px == tx and (py - ty) % h_ in (1, h_ - 1)):
                    s -= p["sonar_predict"]
            if d == t.dir:
                s += p["straight"]
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

        # sprint: try covering 2-3 tiles this turn instead of 1, in any of the same legal-first-step directions.
        # Scored on the same core terms as a plain move (pearl value, trap room at the length left after paying
        # the sprint cost, dead end, head risk) so it only wins when it is genuinely better, not just faster --
        # the path is fully walked by sprint_end above, so this is never a blind gamble past the first tile.
        if p["sprint_max"] > 1 and not self.over():
            for d in ok:
                for steps in range(2, p["sprint_max"] + 1):
                    eff_len = length - (steps - 1)
                    if eff_len < 2:
                        break
                    tidx = self.sprint_end(hx, hy, d, steps, occ, w_, h_)
                    if tidx is None:
                        break  # further steps in this direction are blocked too
                    tx = (hx + DX[d] * steps) % w_
                    ty = (hy + DY[d] * steps) % h_
                    s = 0.0
                    on_pearl = (pm >> tidx) & 1
                    if on_pearl:
                        s += p["pearl_here"]
                    elif lay is not None:
                        for i in range(1, len(lay)):
                            if (lay[i] >> tidx) & 1:
                                s += p["pearl_near"] * (kk + 1 - i)
                                break
                    need_s = min(max(eff_len + margin, p["need_floor"]), p["need_cap"])
                    area = self.flood(free_trap | (1 << tidx), tidx, need_s)
                    if area < need_s:
                        s -= p["trap"] * care * (need_s - area) / need_s
                    else:
                        s += p["area"]
                    if p["dead_end"] and self.exits(tidx, tx, ty, (d + 2) % 4, free) < 2:
                        s -= p["dead_end"] * care
                    for hc, (enemy, pid) in heads.items():
                        ex, ey = hc % w_, hc // w_
                        if (ey == ty and (ex - tx) % w_ in (1, w_ - 1)) or \
                                (ex == tx and (ey - ty) % h_ in (1, h_ - 1)):
                            if enemy:
                                small = length < segs.get(pid, 1) * p["trade_ratio"]
                                s -= p["head_risk_small"] if small else p["head_risk"] * care
                            else:
                                s -= p["team_head_risk"] * care
                    if d == t.dir:
                        s += p["straight"]
                    if s > best_s:
                        best_s = s
                        action = b"MOVE " + LETTERS[d:d + 1] * steps + b"\n"

        if rams and p["ram"] > best_s:
            action = MOVES[rams[0]]
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
