"""Heuristic brain for the sea-dragon bot.

Order of decisions each turn:
  1. Legality, exact from the visible 7x7 state (kelp, any segment, head-on). A legal move always exists
     unless the dragon is boxed in; then the least-bad option is taken.
  2. Among legal moves: reachable-area (trap) check by bitboard flood fill, pearl chasing, head-on risk.
  3. Optional split (off by default; thresholds are tunable parameters).

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
    "pearl_here": 1000.0,   # stepping onto a pearl
    "pearl_near": 20.0,     # per step closer to the nearest visible pearl
    "pearl_k": 6,           # look-ahead distance (steps) of the pearl distance field
    "trap": 3000.0,         # penalty scale when the reachable area is smaller than needed
    "need_margin": 2,       # needed area = length + margin
    "need_cap": 120,        # ... capped, so the flood fill can exit early
    "area": 10.0,           # bonus when not trapped
    "pessimistic": 1,       # trap check counts only tiles seen so far (unseen tiles may be kelp pockets)
    "timed": 0,             # trap check vacates the tail one segment per step (measured worse than static)
    "seq_cap": 48,          # ... for at most this many tail segments
    "head_risk": 400.0,     # adjacent enemy head (we are the longer dragon: a trade hurts us)
    "head_risk_small": 100.0,  # adjacent enemy head when we are much shorter (a trade helps us)
    "trade_ratio": 0.5,     # we count as "much shorter" below this fraction of the enemy's visible length
    "team_head_risk": 250.0,
    "straight": 5.0,        # keep heading
    "split_len": 10 ** 9,   # split when length >= this (off by default)
    "split_child": 3,
    "split_units": 8,       # ... and the team has fewer dragons than this
    "budget_ns": 60_000_000,  # self-metering: skip optional work past this (points on the judge)
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
            return self.decide(proto.parse_turn(block))
        except Exception:  # noqa: BLE001 - a crash would kill the dragon
            if self.strict:
                raise
            self.errors += 1
            return self.fallback(block)

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

    def flood_t(self, free, start, need, seq, delay):
        """Like flood(), but tail cells `seq` (tail first) become free one per step, as the tail retreats."""
        reach = 1 << start
        step = self._step
        n = len(seq)
        i = 0
        while True:
            j = i - delay
            if 0 <= j < n:
                free |= 1 << seq[j]
            new = step(reach, free)
            i += 1
            cnt = new.bit_count()
            # a stalled frontier may resume when more tail cells vacate, but the head cannot wait: it must
            # keep moving, so it survives at most about `cnt` steps inside the region
            if cnt >= need or (new == reach and (i - delay >= n or i > cnt)):
                return cnt
            reach = new

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
                    heads[cell] = (q[0] != team, pid)
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

        # split, only when it is safe to stand still this turn (no enemy head next to us)
        if (length >= p["split_len"] and t.units < self.limit and t.units < p["split_units"]
                and p["split_child"] >= 2 and length - p["split_child"] >= 2
                and not any(e for e, _ in heads.values())):
            return b"SPLIT %d\n" % p["split_child"]
        if len(ok) == 1 or self.over():
            return MOVES[ok[0]]

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
        need = min(length + p["need_margin"], p["need_cap"])
        tail = self.tail
        free_trap = free & self.seen if p["pessimistic"] else free
        seq = None
        if p["timed"] and tail is not None:
            seq = self.body[::-1][:min(length - 1, p["seq_cap"])]  # tail first

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
                if seq is not None:
                    area = self.flood_t(fr, tidx, need, seq, 1 if on_pearl else 0)
                else:
                    if tail is not None and not on_pearl:
                        fr |= 1 << tail  # the tail cell is vacated by this move
                    area = self.flood(fr, tidx, need)
                if area < need:
                    s -= p["trap"] * (need - area) / need
                else:
                    s += p["area"]
            for hc, (enemy, pid) in heads.items():
                ex = hc % w_
                ey = hc // w_
                if (ey == ty and (ex - tx) % w_ in (1, w_ - 1)) or (ex == tx and (ey - ty) % h_ in (1, h_ - 1)):
                    if enemy:
                        small = length < segs.get(pid, 1) * p["trade_ratio"]
                        s -= p["head_risk_small"] if small else p["head_risk"]
                    else:
                        s -= p["team_head_risk"]
            if d == t.dir:
                s += p["straight"]
            if s > best_s:
                best_d, best_s = d, s
        return MOVES[best_d]

    # ------------------------------------------------------------------ fallback
    def fallback(self, block):
        """Independent of learned state: first move that is not immediate death, preferring the heading."""
        try:
            t = proto.parse_turn(block)
            occ = {(int(q[2]), int(q[3])) for q in t.parts}
            ed = t.edges
            vt = ed[11].split()
            kelp = (ed[3].split()[3] == b"w", vt[4] == b"w", ed[4].split()[3] == b"w", vt[3] == b"w")
            order = [t.dir] + [d for d in range(4) if d != t.dir] if t.dir >= 0 else range(4)
            for d in order:
                if kelp[d]:
                    continue
                if ((t.hx + DX[d]) % self.W, (t.hy + DY[d]) % self.H) in occ:
                    continue
                return MOVES[d]
        except Exception:  # noqa: BLE001
            pass
        return MOVES[0]
