#!/usr/bin/env python3
"""Dump the internal data of every .agr file in THIS folder to a .txt file.

An .agr is a (usually gzip-compressed) Ableton groove/clip XML. This script
gunzips it and writes the raw XML (pretty-printed) so you can read the actual
contents — time signature, loop length, and the MidiNoteEvents with their
Time / Duration / Velocity, plus the KeyTrack pitches.

Usage:
    python3 agr_to_txt.py        # run from inside the agr_txt folder
Output:
    <name>.txt next to each <name>.agr
"""
import glob
import gzip
import os
import sys
import xml.dom.minidom as minidom

HERE = os.path.dirname(os.path.abspath(__file__))


def dump(path):
    data = open(path, "rb").read()
    gzipped = data[:2] == b"\x1f\x8b"
    try:
        raw = gzip.decompress(data) if gzipped else data
        text = raw.decode("utf-8", "replace")
        try:                                   # pretty-print if it parses as XML
            text = minidom.parseString(text).toprettyxml(indent="  ")
            text = "\n".join(ln for ln in text.splitlines() if ln.strip())   # drop blank lines
        except Exception:
            pass
    except Exception as e:                      # noqa: BLE001
        text = f"(failed to decode {os.path.basename(path)}: {e})"
    out = os.path.splitext(path)[0] + ".txt"
    with open(out, "w") as f:
        f.write(text)
    return out, len(text), gzipped


if __name__ == "__main__":
    agrs = sorted(glob.glob(os.path.join(HERE, "*.agr")))
    if not agrs:
        print("No .agr files found in", HERE)
        sys.exit(0)
    for p in agrs:
        out, n, gz = dump(p)
        print(f"wrote {os.path.basename(out)}  ({n} chars, {'gzipped' if gz else 'plain'} source)")
