"""Reads an official .replay file: who split, who died of what and when, and the final standings.

A replay is a Cap'n Proto message in packed encoding (tools/replay_maps.py unpacks it). The schema below was read
off the official replay viewer (unswbc's replay-viewer.vsix, webview.js):

    Replay      map :Text @p0, botA :Text @p1, botB :Text @p2, events :List(Event) @p3, result :GameResult @p4
    Event       union tag u16 @0, payload @p0: 0 roundStart{round i32@0}  1 turnStart{id i32@0}
                4 dragonAction  9 dragonUpdate{id i32@0, facing u16@4, head @p0, tail @p1}
                10 dragonSplit{parentId i32@0, childId i32@4, team u16@8, childFacing u16@10, parentBody @p0,
                childBody @p1}  11 dragonDeath{id i32@0, reason u16@4}  (2,3,5-8,12: pearls, tiles, logs, sonar)
    GameResult  terminated bit@0, endReason u16@2, winner tag u16@4 (1 = has one) + team u16@6,
                teamA/teamB @p0/@p1 = TeamStanding{dragonCount i32@0, longestDragon i32@4, totalLength i32@8}

Teams are 0 = A, 1 = B. Death reasons, in order: wall, self, other body, head-on, no action. The founding dragons are
the map's DRAGON lines, numbered in file order.

    .venv\\Scripts\\python.exe tools\\replay_parse.py replays\\ladder\\M364078.replay
"""
import struct
import sys

from replay_maps import unpack

REASONS = ("wall", "self", "other body", "head-on", "no action")
_u16, _i32, _u64 = struct.Struct("<H").unpack_from, struct.Struct("<i").unpack_from, struct.Struct("<Q").unpack_from


class Message:
    """Just enough of a Cap'n Proto reader for replays: structs, lists (incl. composite), text, far pointers."""

    def __init__(self, raw):
        n = struct.unpack_from("<I", raw, 0)[0] + 1
        sizes = struct.unpack_from(f"<{n}I", raw, 4)
        pos = 4 + 4 * n
        pos += (8 - pos % 8) % 8
        self.segs = []
        for s in sizes:
            self.segs.append(memoryview(raw)[pos:pos + 8 * s])
            pos += 8 * s

    def _follow(self, seg, word):
        """Resolves the pointer at (seg, word) to (seg, target word, pointer bits describing the target)."""
        p = _u64(self.segs[seg], 8 * word)[0]
        if p & 3 != 2:
            off = (p & 0xFFFFFFFF) >> 2
            off -= (1 << 30) if off >= 1 << 29 else 0
            return seg, word + 1 + off, p
        pad_seg, pad = p >> 32, (p & 0xFFFFFFFF) >> 3
        if not p & 4:  # single far: the landing pad is an ordinary pointer
            return self._follow(pad_seg, pad)
        far = _u64(self.segs[pad_seg], 8 * pad)[0]  # double far: content address, then a tag word
        tag = _u64(self.segs[pad_seg], 8 * pad + 8)[0]
        return far >> 32, (far & 0xFFFFFFFF) >> 3, tag

    def struct(self, seg, word):
        """(seg, data start, data words, pointer start, pointer count) of the struct the pointer points at."""
        if _u64(self.segs[seg], 8 * word)[0] == 0:
            return None
        s, w, p = self._follow(seg, word)
        dw, pc = (p >> 32) & 0xFFFF, p >> 48
        return s, w, dw, w + dw, pc

    def list_structs(self, seg, word):
        s, w, p = self._follow(seg, word)
        assert (p >> 32) & 7 == 7, "expected a composite (struct) list"
        tag = _u64(self.segs[s], 8 * w)[0]
        count, dw, pc = (tag & 0xFFFFFFFF) >> 2, (tag >> 32) & 0xFFFF, tag >> 48
        start = w + 1
        return [(s, start + i * (dw + pc), dw, start + i * (dw + pc) + dw, pc) for i in range(count)]

    def text(self, seg, word):
        if _u64(self.segs[seg], 8 * word)[0] == 0:
            return ""
        s, w, p = self._follow(seg, word)
        n = p >> 35
        return bytes(self.segs[s][8 * w:8 * w + n]).rstrip(b"\0").decode("utf-8", "replace")

    def u16(self, st, off):
        s, w, dw, _, _ = st
        return _u16(self.segs[s], 8 * w + off)[0] if off + 2 <= 8 * dw else 0

    def i32(self, st, off):
        s, w, dw, _, _ = st
        return _i32(self.segs[s], 8 * w + off)[0] if off + 4 <= 8 * dw else 0

    def ptr(self, st, i):
        s, w, dw, pw, pc = st
        return self.struct(s, pw + i) if i < pc else None


def parse(data):
    """-> {"map", "rounds", "winner" ('A'/'B'/None), "end_reason", "standing" {team: (dragons, longest, total)},
    "dragons" {id: {"team", "born", "parent", "died", "reason"}}}"""
    msg = Message(unpack(data))
    root = msg.struct(0, 0)
    s, w, dw, pw, pc = root
    map_text = msg.text(s, pw)
    dragons = {}
    for i, line in enumerate(x for x in map_text.split("\n") if x.startswith("DRAGON ")):
        dragons[i] = {"team": "AB"[int(line.split()[1])], "born": 0, "parent": None, "died": None, "reason": None}
    rnd = 0
    for ev in msg.list_structs(s, pw + 3):
        kind = msg.u16(ev, 0)
        if kind == 0:
            rnd = msg.i32(msg.ptr(ev, 0), 0)
        elif kind == 10:
            e = msg.ptr(ev, 0)
            dragons[msg.i32(e, 4)] = {"team": "AB"[msg.u16(e, 8)], "born": rnd, "parent": msg.i32(e, 0),
                                      "died": None, "reason": None}
        elif kind == 11:
            e = msg.ptr(ev, 0)
            d = dragons.setdefault(msg.i32(e, 0), {"team": "?", "born": None, "parent": None})
            d["died"], d["reason"] = rnd, REASONS[msg.u16(e, 4)] if msg.u16(e, 4) < len(REASONS) else "?"
    res = msg.ptr(root, 4)
    standing = {}
    for team, i in (("A", 0), ("B", 1)):
        t = msg.ptr(res, i)
        standing[team] = (msg.i32(t, 0), msg.i32(t, 4), msg.i32(t, 8)) if t else None
    winner = "AB"[msg.u16(res, 6)] if msg.u16(res, 4) == 1 else None
    name = next((x[9:].strip() for x in map_text.split("\n") if x.startswith("MAP_NAME ")), "?")
    return {"map": name, "rounds": rnd, "winner": winner, "end_reason": msg.u16(res, 2), "standing": standing,
            "dragons": dragons}


def main():
    for path in sys.argv[1:]:
        r = parse(open(path, "rb").read())
        dead = [d for d in r["dragons"].values() if d["died"] is not None]
        print(f"{path}: {r['map']}, {r['rounds']} rounds, winner {r['winner']}, end reason {r['end_reason']}, "
              f"standings {r['standing']}, {len(r['dragons'])} dragons, {len(dead)} deaths")
        for team in "AB":
            causes = {}
            for d in dead:
                if d["team"] == team:
                    causes[d["reason"]] = causes.get(d["reason"], 0) + 1
            print(f"   team {team} deaths: {causes}")


if __name__ == "__main__":
    main()
