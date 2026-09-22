"""Reader, writer and geometry for the engine's text map format.

    MAP w h
    SYMMETRY x|y|xy              (optional)
    MAP_NAME text
    TILE_COUNT n                 then n lines "TILE x y min_gap max_gap", row-major
    EDGE_COUNT m                 then m lines "EDGE index type portal_id"
    DRAGON_COUNT k               then k lines "DRAGON team length x y x y ..." (head first)

An edge is (x, y, d): d = 0 is the NORTH edge of tile (x, y), between (x, y-1) and (x, y); d = 1 is its WEST edge,
between (x-1, y) and (x, y). East and south borders are the wrapped copies of the west and north ones. The file index
of an edge is (2*y + d) * (w + 1) + x: rows alternate north / west edges and each row has one padding column, which is
why the indices have gaps. Type 0 = empty, 1 = kelp, 2 = portal (its id, shared by exactly two edges, follows; else -1).

SYMMETRY y mirrors x (x -> w-1-x), x mirrors y, xy is the 180 degree rotation; that is what the bundled maps do, and
tools/test_mapfile.py checks it against the engine. Directions are N, E, S, W = 0..3 and y grows downwards.
"""
import collections

NORTH, WEST = 0, 1  # edge kinds
DX = (0, 1, 0, -1)
DY = (-1, 0, 1, 0)
SYMMETRIES = ("x", "y", "xy")
MIN_SIDE, MAX_SIDE = 10, 64


class GameMap:
    def __init__(self, w, h, name="", sym="", tiles=None, kelp=None, portals=None, dragons=None):
        self.w, self.h, self.name, self.sym = w, h, name, sym
        self.tiles = tiles if tiles is not None else {}  # (x, y) -> (min_gap, max_gap)
        self.kelp = kelp if kelp is not None else set()  # {(x, y, d)}
        self.portals = portals if portals is not None else {}  # (x, y, d) -> id
        self.dragons = dragons if dragons is not None else []  # [(team, [(x, y), ...])], head first

    # ------------------------------------------------------------------ geometry
    def wrap(self, x, y):
        return x % self.w, y % self.h

    def mirror_tile(self, x, y):
        w, h = self.w, self.h
        if self.sym == "y":
            return w - 1 - x, y
        if self.sym == "x":
            return x, h - 1 - y
        return w - 1 - x, h - 1 - y

    def mirror_edge(self, e):
        """The image of an edge; the boundary between tiles c-1 and c maps to the boundary between w-c and w-1-c."""
        x, y, d = e
        w, h = self.w, self.h
        fx = self.sym in ("y", "xy")
        fy = self.sym in ("x", "xy")
        if d == NORTH:
            return (w - 1 - x if fx else x), ((h - y) % h if fy else y), NORTH
        return ((w - x) % w if fx else x), (h - 1 - y if fy else y), WEST

    def edge_ahead(self, x, y, d):
        """The edge crossed by stepping from tile (x, y) in direction d."""
        if d == 0:
            return x, y, NORTH
        if d == 3:
            return x, y, WEST
        if d == 2:
            return x, (y + 1) % self.h, NORTH
        return (x + 1) % self.w, y, WEST

    def far_side(self, e, d):
        """The tile on the far side of edge e when crossing it in direction d (used for portal exits)."""
        x, y, kind = e
        if d in (1, 2):  # crossing east / south lands on the tile that owns the edge
            return x, y
        return ((x, (y - 1) % self.h) if kind == NORTH else ((x - 1) % self.w, y))

    def partner(self, e):
        pid = self.portals[e]
        for f, q in self.portals.items():
            if q == pid and f != e:
                return f
        return None

    def step(self, x, y, d):
        """The tile a dragon arrives at when it moves from (x, y) in direction d, or None when kelp is in the way."""
        e = self.edge_ahead(x, y, d)
        if e in self.kelp:
            return None
        if e in self.portals:
            return self.far_side(self.partner(e), d)
        return (x + DX[d]) % self.w, (y + DY[d]) % self.h

    def components(self):
        """Number of connected components of the tile graph (moves through open edges and portals)."""
        seen, comps = set(), 0
        for start in self.tiles:
            if start in seen:
                continue
            comps += 1
            seen.add(start)
            todo = [start]
            while todo:
                x, y = todo.pop()
                for d in range(4):
                    nxt = self.step(x, y, d)
                    if nxt is not None and nxt not in seen and nxt in self.tiles:
                        seen.add(nxt)
                        todo.append(nxt)
        return comps

    # ------------------------------------------------------------------ validation
    def problems(self, connected=False):
        """Everything wrong with the map as a two-team symmetric game board; an empty list means it is valid.

        `connected` also demands one region. The engine does not (the bundled Queen of Spades and Schooltime have
        walled-off decorative pockets), but a generated map with unreachable tiles wastes pearls and board.
        """
        out = []
        w, h = self.w, self.h
        if not (MIN_SIDE <= w <= MAX_SIDE and MIN_SIDE <= h <= MAX_SIDE):
            out.append(f"size {w}x{h} outside {MIN_SIDE}..{MAX_SIDE}")
        if self.sym not in SYMMETRIES:
            out.append(f"symmetry {self.sym!r} is not one of {SYMMETRIES}")
            return out
        if len(self.tiles) != w * h:
            out.append(f"{len(self.tiles)} tiles for a {w}x{h} map")
        for (x, y), (lo, hi) in self.tiles.items():
            if self.tiles.get(self.mirror_tile(x, y)) != (lo, hi):
                out.append(f"tile ({x},{y}) and its mirror differ")
                break
            if lo < 0 or hi < lo:
                out.append(f"tile ({x},{y}) has gaps {lo}..{hi}")
                break
        for e in self.kelp:
            if self.mirror_edge(e) not in self.kelp:
                out.append(f"kelp {e} has no mirror image")
                break
        if self.kelp & set(self.portals):
            out.append("an edge is both kelp and a portal")
        out += self._portal_problems()
        out += self._dragon_problems()
        if connected and self.components() != 1:
            out.append(f"{self.components()} disconnected regions")
        return out

    def _portal_problems(self):
        out = []
        by_id = collections.defaultdict(list)
        for e, pid in self.portals.items():
            by_id[pid].append(e)
        for pid, es in by_id.items():
            if len(es) != 2:
                out.append(f"portal id {pid} appears {len(es)} times")
                continue
            p, q = es
            if p[2] != q[2]:
                out.append(f"portal {pid} joins a north edge to a west edge")
            fixed_p, fixed_q = self.mirror_edge(p) == p, self.mirror_edge(q) == q
            if fixed_p != fixed_q:
                out.append(f"portal {pid}: one end lies on the line of symmetry and the other does not")
            elif not fixed_p:
                mp, mq = self.mirror_edge(p), self.mirror_edge(q)
                ids = {self.portals.get(mp), self.portals.get(mq)}
                if None in ids or len(ids) != 1:
                    out.append(f"portal {pid}: the mirror image of the pair is not a pair")
        return out

    def _dragon_problems(self):
        out, cells = [], {}
        teams = collections.Counter(t for t, _ in self.dragons)
        if teams[0] != teams[1] or not teams[0]:
            out.append(f"teams have {teams[0]} and {teams[1]} dragons")
        mirrored = {(1 - t, tuple(self.mirror_tile(*c) for c in body)) for t, body in self.dragons}
        if mirrored != {(t, tuple(body)) for t, body in self.dragons}:
            out.append("team 1's dragons are not the mirror images of team 0's")
        for t, body in self.dragons:
            for i, c in enumerate(body):
                if c not in self.tiles:
                    out.append(f"dragon segment {c} is off the map")
                if c in cells:
                    out.append(f"dragon segments overlap at {c}")
                cells[c] = t
                if i and not any(self.step(*body[i - 1], d) == c for d in range(4)):
                    out.append(f"dragon segments {body[i - 1]} and {c} are not connected")
        return out

    # ------------------------------------------------------------------ text
    @classmethod
    def loads(cls, text):
        m = cls(0, 0)
        edges = []
        for line in text.splitlines():
            f = line.split()
            if not f:
                continue
            key = f[0]
            if key == "MAP":
                m.w, m.h = int(f[1]), int(f[2])
            elif key == "SYMMETRY":
                m.sym = f[1]
            elif key == "MAP_NAME":
                m.name = line.split(None, 1)[1] if len(f) > 1 else ""
            elif key == "TILE":
                m.tiles[(int(f[1]), int(f[2]))] = (int(f[3]), int(f[4]))
            elif key == "EDGE":
                edges.append((int(f[1]), int(f[2]), int(f[3])))
            elif key == "DRAGON":
                n = int(f[2])
                m.dragons.append((int(f[1]), [(int(f[3 + 2 * i]), int(f[4 + 2 * i])) for i in range(n)]))
        for idx, kind, pid in edges:
            r, x = divmod(idx, m.w + 1)
            e = (x % m.w, (r // 2) % m.h, r % 2)  # the padding column and the last row are wrapped copies
            if kind == 1:
                m.kelp.add(e)
            elif kind == 2:
                m.portals[e] = pid
        return m

    def dumps(self):
        w, h = self.w, self.h
        out = [f"MAP {w} {h}"]
        if self.sym:
            out.append(f"SYMMETRY {self.sym}")
        out += [f"MAP_NAME {self.name}", f"TILE_COUNT {len(self.tiles)}"]
        out += [f"TILE {x} {y} {lo} {hi}" for (y, x), (lo, hi) in sorted((c[::-1], g) for c, g in self.tiles.items())]
        out.append(f"EDGE_COUNT {2 * w * h}")
        for y in range(h):
            for d in (NORTH, WEST):
                for x in range(w):
                    e = (x, y, d)
                    idx = (2 * y + d) * (w + 1) + x
                    if e in self.kelp:
                        out.append(f"EDGE {idx} 1 -1")
                    elif e in self.portals:
                        out.append(f"EDGE {idx} 2 {self.portals[e]}")
                    else:
                        out.append(f"EDGE {idx} 0 -1")
        out.append(f"DRAGON_COUNT {len(self.dragons)}")
        for team, body in self.dragons:
            out.append(f"DRAGON {team} {len(body)} " + " ".join(f"{x} {y}" for x, y in body))
        out.append("END")
        return "\n".join(out) + "\n"

    def render(self):
        """ASCII picture: '#' kelp, letters portal ids, digits/letters for dragons (team 0 lower, team 1 upper)."""
        w, h = self.w, self.h
        g = [[" "] * (2 * w + 1) for _ in range(2 * h + 1)]
        for (x, y) in self.tiles:
            g[2 * y + 1][2 * x + 1] = "."
        for t, body in self.dragons:
            for i, (x, y) in enumerate(body):
                g[2 * y + 1][2 * x + 1] = ("Hb"[i > 0] if t == 0 else "Ao"[i > 0])
        for (x, y, d) in self.kelp:
            if d == NORTH:
                g[2 * y][2 * x + 1] = "="
            else:
                g[2 * y + 1][2 * x] = "|"
        for (x, y, d), pid in self.portals.items():
            ch = "0123456789abcdefghijklmnopqrstuvwxyz"[pid % 36]
            if d == NORTH:
                g[2 * y][2 * x + 1] = ch
            else:
                g[2 * y + 1][2 * x] = ch
        return "\n".join("".join(r) for r in g)
