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
(ph/pv: portal). Portals are treated as walls because their far side is unknown.
"""
import time

import proto

_pc = time.perf_counter_ns
DX, DY, LETTERS = proto.DX, proto.DY, proto.LETTERS
DOT7, DOT8 = proto.DOT7, proto.DOT8
MOVES = (b"MOVE N\n", b"MOVE E\n", b"MOVE S\n", b"MOVE W\n")
WIN_R = [t // 7 for t in range(49)]
WIN_C = [t % 7 for t in range(49)]

# legality classes, best first
OK, PORTAL, TRADE_ENEMY, TRADE_TEAM, DEAD = 0, 1, 2, 3, 4

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
    # Once a dragon's own length reaches grow_care_len, it plays more cautiously: trap/dead-end/head-on
    # penalties are multiplied by grow_care_mult and need_margin gets grow_care_margin added, so a dragon that
    # is already a real investment (whether a surviving founder or a grow_mod-designated grower) protects that
    # length instead of taking the same risks a short, disposable swarm dragon would. Tuned 2026-09-26 by CMA-ES
    # (tools/tune.py, --params grow_care_len,grow_care_mult,grow_care_margin, 14 generations, averaged over the
    # last 8 generation means: tools/tuned/grow_care_run1_avg8.json) against defaults/old/grower_brain (a
    # disciplined opponent that splits a little early then stops and just grows -- built to reproduce the
    # round-500 longest-living-dragon tiebreak loss). Validated on held-out maps/seeds: 92.0% [88.4%, 94.6%] vs
    # grower_brain (was a loss before this), 50.0% [44.4%, 55.6%] vs plain defaults (no regression, but not a
    # general win either -- this only activates once a dragon is already fairly long, so it is a narrow fix for
    # the tiebreak weakness, not a strict replacement for DEFAULTS the way the rest of these values are).
    "grow_care_len": 1,
    "grow_care_mult": 1.9346,
    "grow_care_margin": 1,
    "budget_ns": 60_000_000,   # self-metering: skip optional work past this (points on the judge)
}


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
        self.founder = None  # True when alive at round 0 (decided on the first turn)
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
        return (reach | e | w | s | n) & free

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
        for r in range(8):  # horizontal edges: north edge of window row r
            line = ed[r]
            if line != DOT7:
                row = ((y0 + r) % h_) * w_
                for c, tok in enumerate(line.split()):
                    if tok != b".":
                        bit = 1 << (row + (x0 + c) % w_)
                        if tok == b"w":
                            kh |= bit
                        else:
                            ph |= bit
        for r in range(7):  # vertical edges: west edge of window column c
            line = ed[8 + r]
            if line != DOT8:
                row = ((y0 + r) % h_) * w_
                for c, tok in enumerate(line.split()):
                    if tok != b".":
                        bit = 1 << (row + (x0 + c) % w_)
                        if tok == b"w":
                            kv |= bit
                        else:
                            pv |= bit
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
        if child < 2 or length - child < 2 or t.units >= self.limit or t.rnd >= p["split_r_end"]:
            return False
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
        self.seen |= self.window((hx - 3) % w_, (hy - 3) % h_)

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

        # legality of the four moves, exact
        nidx = (((hy - 1) % h_) * w_ + hx, hy * w_ + (hx + 1) % w_, ((hy + 1) % h_) * w_ + hx, hy * w_ + (hx - 1) % w_)
        kh, kv, ph, pv = self.kh, self.kv, self.ph, self.pv
        kelp = ((kh >> hidx) & 1, (kv >> nidx[1]) & 1, (kh >> nidx[2]) & 1, (kv >> hidx) & 1)
        port = ((ph >> hidx) & 1, (pv >> nidx[1]) & 1, (ph >> nidx[2]) & 1, (pv >> hidx) & 1)
        status = [OK, OK, OK, OK]
        for d in range(4):
            if kelp[d]:
                status[d] = DEAD
            elif port[d]:
                status[d] = PORTAL
            elif (occ >> nidx[d]) & 1:
                hd = heads.get(nidx[d])
                status[d] = DEAD if hd is None else (TRADE_ENEMY if hd[0] else TRADE_TEAM)
        ok = [d for d in range(4) if status[d] == OK]
        if self.debug:
            self.dbg = {"rnd": t.rnd, "pos": (hx, hy), "dir": t.dir, "len": length, "status": list(status),
                        "kelp": kelp, "port": port, "body": list(self.body[:8]), "tail": self.tail,
                        "parts": [tuple(q) for q in t.parts][:12]}
        if not ok:
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

        if self.founder is None:
            self.founder = t.rnd == 0
        if self.want_split(t, heads, len(ok), hx, hy, w_, h_):
            action = b"SPLIT %d\n" % p["split_child"]
            return action + b"SONAR %d\n" % sonar_msg if sonar_msg is not None else action
        if len(ok) == 1 or self.over():
            action = MOVES[ok[0]]
            return action + b"SONAR %d\n" % sonar_msg if sonar_msg is not None else action

        # pearls in view
        x0 = (hx - 3) % w_
        y0 = (hy - 3) % h_
        pm = 0
        fl = t.flags
        for i in range(49):
            if fl[i] == b"1":
                pm |= 1 << (((y0 + WIN_R[i]) % h_) * w_ + (x0 + WIN_C[i]) % w_)

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
        need = min(max(length + margin, p["need_floor"]), p["need_cap"])
        tail = self.tail
        free_trap = free & self.seen if p["pessimistic"] else free

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
            tidx = nidx[d]
            tx = (hx + DX[d]) % w_
            ty = (hy + DY[d]) % h_
            s = 0.0
            on_pearl = (pm >> tidx) & 1
            if on_pearl:
                s += p["pearl_here"]
            elif lay is not None:
                for i in range(1, len(lay)):
                    if (lay[i] >> tidx) & 1:
                        s += p["pearl_near"] * (kk + 1 - i)
                        break
            if not self.over():
                fr = free_trap | (1 << tidx)
                if tail is not None and not on_pearl:
                    fr |= 1 << tail  # the tail cell is vacated by this move
                area = self.flood(fr, tidx, need)
                if area < need:
                    s -= p["trap"] * care * (need - area) / need
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
        action = MOVES[best_d]
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
