"""Extracts the map from ladder replays into maps/ladder/, one .map per distinct map name.

A .replay is a Cap'n Proto message in "packed" encoding; the map is stored as plain text inside it. unpack() is a
port of the official replay viewer's decoder (sd() in unswbc's replay-viewer.vsix). Checked against the bundled
maps: Queen Of Spades, Schooltime, Trophy and Default Small come out byte-identical to maps/*.map.

    .venv\\Scripts\\python.exe tools\\replay_maps.py [replay files or directories ...]   (default: replays/ and leader-replays/)
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "maps" / "ladder"


def unpack(data):
    """Cap'n Proto packed -> unpacked bytes. A tag byte's set bits mark which of the next 8 bytes are non-zero;
    tag 0x00 is followed by a count of extra all-zero words, tag 0xFF by 8 raw bytes and a count of raw words."""
    out, i = bytearray(), 0
    while i < len(data):
        tag = data[i]
        i += 1
        if tag == 0:
            out += bytes(8 * (1 + data[i]))
            i += 1
        elif tag == 0xFF:
            out += data[i:i + 8]
            n = data[i + 8]
            out += data[i + 9:i + 9 + 8 * n]
            i += 9 + 8 * n
        else:
            for bit in range(8):
                if tag >> bit & 1:
                    out.append(data[i])
                    i += 1
                else:
                    out.append(0)
    return bytes(out)


def map_text(replay_bytes):
    raw = unpack(replay_bytes)
    start = raw.find(b"MAP ")
    end = raw.find(b"\nEND", start)
    if start < 0 or end < 0:
        raise ValueError("no map found in replay")
    return raw[start:end + 4] + b"\n"


def main():
    args = [pathlib.Path(a) for a in sys.argv[1:]] or [ROOT / "replays", ROOT / "leader-replays"]
    files = sorted(f for a in args for f in (a.glob("*.replay") if a.is_dir() else [a]))
    OUT.mkdir(parents=True, exist_ok=True)
    for f in files:
        text = map_text(f.read_bytes())
        name = re.search(rb"MAP_NAME ([^\n]*)", text).group(1).decode().strip()
        dest = OUT / (re.sub(r"\W+", "_", name.lower()).strip("_") + ".map")
        if dest.exists() and dest.read_bytes() != text:
            print(f"WARNING {f.name}: {name} differs from the {dest.name} already extracted (map changed on the ladder?)")
            continue
        dest.write_bytes(text)
        print(f"{f.name}: {name} -> {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
