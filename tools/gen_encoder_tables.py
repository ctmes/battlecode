"""Regenerates bot/encoder.py's baked _WINDOW_PERM / _EDGE_H_CODE / _EDGE_V_CODE tables from its _rot/_edge_owner
geometry functions, and by default checks the baked tables still match (run this after touching the geometry).

    .venv\\Scripts\\python.exe tools\\gen_encoder_tables.py          # verify (exit 1 and a diff on mismatch)
    .venv\\Scripts\\python.exe tools\\gen_encoder_tables.py --print  # print fresh literals to paste into encoder.py
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "bot"))
import encoder as e  # noqa: E402


def window_perm(dir_):
    perm = [0] * e.WINDOW
    for i in range(e.WINDOW):
        adx, ady = i % 7 - 3, i // 7 - 3
        ldx, ldy = e._rot(adx, ady, dir_)
        perm[i] = (ldy + 3) * 7 + (ldx + 3)
    return tuple(perm)


def edge_code(is_h, ox, oy):
    local = (oy + 3) * e.EDGE_SPAN + (ox + 3)
    return local if is_h else e.EDGE_CELLS + local


def edge_table(dir_, is_horiz, rows, cols):
    vals = []
    for r in range(rows):
        for c in range(cols):
            is_h, ox, oy = e._edge_owner(dir_, is_horiz, r, c)
            vals.append(edge_code(is_h, ox, oy))
    return tuple(vals)


def wrap(name, rows):
    lines = [f"{name} = ("]
    for row in rows:
        text = ", ".join(str(v) for v in row)
        lines.append(f"    ({text}),")
    lines.append(")")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", action="store_true", dest="print_")
    args = ap.parse_args()

    window = tuple(window_perm(d) for d in range(4))
    eh = tuple(edge_table(d, True, 8, 7) for d in range(4))
    ev = tuple(edge_table(d, False, 7, 8) for d in range(4))

    if args.print_:
        print(wrap("_WINDOW_PERM", window))
        print(wrap("_EDGE_H_CODE", eh))
        print(wrap("_EDGE_V_CODE", ev))
        return

    mismatches = []
    if window != e._WINDOW_PERM:
        mismatches.append("_WINDOW_PERM")
    if eh != e._EDGE_H_CODE:
        mismatches.append("_EDGE_H_CODE")
    if ev != e._EDGE_V_CODE:
        mismatches.append("_EDGE_V_CODE")
    if mismatches:
        print(f"MISMATCH in {mismatches}: re-run with --print and paste the fresh literals into encoder.py")
        sys.exit(1)
    print("ok: encoder.py's baked tables match _rot/_edge_owner")


if __name__ == "__main__":
    main()
