#!/usr/bin/env python3
"""Generate Sky's Unicode tables (display-unsafe, invisible-in-identifiers).

All tables come from the same pinned Unicode version and SHA-256-pinned UCD
files, so the renderer's escaped set and the lexer's rejected set cannot be
generated from different data.

1. sky-diag, display-unsafe code points.

The escaped set (spec/diagnostics.md, "Terminal safety") is the union of:
  - Default_Ignorable_Code_Point      (DerivedCoreProperties.txt)
  - General_Category = Cf             (extracted/DerivedGeneralCategory.txt)
  - U+2028 LINE SEPARATOR, U+2029 PARAGRAPH SEPARATOR
  - C0 controls except tab, DEL, and C1 controls (U+0000-U+0008,
    U+000A-U+001F, U+007F-U+009F). Tab is handled by the renderer: kept
    in source lines, escaped everywhere else.

2. sky-syntax, characters the lexer rejects as invisible in identifiers:
     Default_Ignorable_Code_Point & (XID_Start | XID_Continue)
   (both from DerivedCoreProperties.txt). These are the default-ignorable
   characters that `unicode-ident` would otherwise let into an identifier,
   e.g. ZWJ, the Hangul fillers, and the variation selectors including
   U+E0100..U+E01EF. The lexer adds a few non-identifier default-ignorables
   of its own (see `INVISIBLE_EXTRA` in crates/sky-syntax/src/lexer.rs).

Outputs (all checked in; paths relative to the repository root):
  crates/sky-diag/src/display_unsafe_table.rs
      the merged, sorted range table
  crates/sky-diag/tests/data/ucd-display-unsafe-source.txt
      the verbatim UCD records used, parsed independently by
      crates/sky-diag/tests/display_unsafe_table.rs
  crates/sky-syntax/src/invisible_table.rs
      the merged, sorted range table for the lexer
  crates/sky-syntax/tests/data/ucd-default-ignorable.txt
      the verbatim Default_Ignorable_Code_Point records, which the lexer's
      exhaustive test combines with `unicode-ident` independently of the
      generated intersection

Usage (from crates/sky-diag; output paths do not depend on the cwd):
  python3 tools/gen_display_unsafe.py             # download and regenerate
  python3 tools/gen_display_unsafe.py --check     # fail if outputs are stale
  python3 tools/gen_display_unsafe.py --ucd-dir D # use local copies in D

Only the Python standard library is used. Downloads are verified against
the pinned SHA-256 digests below; bump UNICODE_VERSION and the digests
together.
"""

import argparse
import hashlib
import os
import re
import sys
import urllib.request

UNICODE_VERSION = (18, 0, 0)
BASE_URL = "https://www.unicode.org/Public/{v}/ucd/"
SOURCES = {
    # file name: (path under ucd/, sha256)
    "DerivedCoreProperties.txt": (
        "DerivedCoreProperties.txt",
        "09c928886a178fcafd93c29e4bd59073a058e5a100b716d425cb563ab50f68c9",
    ),
    "DerivedGeneralCategory.txt": (
        "extracted/DerivedGeneralCategory.txt",
        "d6b151d2d40ee9b1876d26f417980f45ffae47b6055ccf7203cb31f07a030f94",
    ),
}
EXTRA = [
    (0x0000, 0x0008, "C0 controls before tab"),
    (0x000A, 0x001F, "C0 controls after tab (LF, CR, ESC, ...)"),
    (0x007F, 0x009F, "DEL and C1 controls (CSI U+009B, ...)"),
    (0x2028, 0x2029, "LINE SEPARATOR, PARAGRAPH SEPARATOR"),
]

HERE = os.path.dirname(os.path.abspath(__file__))
CRATE = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(CRATE))
TABLE_RS = os.path.join(CRATE, "src", "display_unsafe_table.rs")
SOURCE_TXT = os.path.join(CRATE, "tests", "data", "ucd-display-unsafe-source.txt")
SYNTAX_CRATE = os.path.join(REPO, "crates", "sky-syntax")
INVISIBLE_RS = os.path.join(SYNTAX_CRATE, "src", "invisible_table.rs")
INVISIBLE_SOURCE_TXT = os.path.join(SYNTAX_CRATE, "tests", "data", "ucd-default-ignorable.txt")
RECORD = re.compile(r"^([0-9A-F]{4,6})(?:\.\.([0-9A-F]{4,6}))?\s*;\s*([A-Za-z_]+)\s*(#.*)?$")


def version():
    return ".".join(map(str, UNICODE_VERSION))


def load(name, ucd_dir):
    rel, digest = SOURCES[name]
    if ucd_dir:
        with open(os.path.join(ucd_dir, name), "rb") as f:
            data = f.read()
    else:
        url = BASE_URL.format(v=version()) + rel
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
    got = hashlib.sha256(data).hexdigest()
    if got != digest:
        sys.exit(f"{name}: sha256 {got} does not match pinned {digest}")
    text = data.decode("utf-8")
    first = text.splitlines()[0]
    if first != f"# {name[:-4]}-{version()}.txt":
        sys.exit(f"{name}: unexpected header {first!r}")
    return text


def records(text, prop):
    """Verbatim lines whose property field is `prop`, and their ranges."""
    lines, ranges = [], []
    for line in text.splitlines():
        m = RECORD.match(line)
        if m and m.group(3) == prop:
            lo = int(m.group(1), 16)
            hi = int(m.group(2) or m.group(1), 16)
            lines.append(line)
            ranges.append((lo, hi))
    if not ranges:
        sys.exit(f"no records for {prop}")
    return lines, ranges


def merge(ranges):
    out = []
    for lo, hi in sorted(ranges):
        if out and lo <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def render_table(merged):
    v = version()
    total = sum(hi - lo + 1 for lo, hi in merged)
    lines = [
        "// @generated by crates/sky-diag/tools/gen_display_unsafe.py. Do not edit.",
        f"//",
        f"// Unicode {v}:",
    ]
    for name, (rel, digest) in SOURCES.items():
        lines.append(f"//   {BASE_URL.format(v=v)}{rel}")
        lines.append(f"//     sha256 {digest}")
    lines += [
        "//",
        "// Default_Ignorable_Code_Point | General_Category=Cf | U+2028 | U+2029",
        "// | C0 controls except tab | DEL | C1 controls.",
        f"// {len(merged)} ranges, {total} code points.",
        "",
        "/// Unicode version the display-unsafe table was generated from.",
        f"pub const UNICODE_VERSION: (u8, u8, u8) = ({UNICODE_VERSION[0]}, {UNICODE_VERSION[1]}, {UNICODE_VERSION[2]});",
        "",
        "/// Sorted, non-overlapping, non-adjacent inclusive code point ranges.",
        "pub(crate) static RANGES: &[(u32, u32)] = &[",
    ]
    lines += [f"    (0x{lo:04X}, 0x{hi:04X})," for lo, hi in merged]
    lines += ["];", ""]
    return "\n".join(lines)


def render_source(dcp_lines, dgc_lines):
    v = version()
    out = [
        f"# Verbatim records from the Unicode {v} UCD used to generate",
        "# crates/sky-diag/src/display_unsafe_table.rs",
        "# (tools/gen_display_unsafe.py). Parsed independently by",
        "# tests/display_unsafe_table.rs as the reference definition.",
    ]
    for name, (rel, digest) in SOURCES.items():
        out.append(f"#   {rel} sha256 {digest}")
    out += ["", "# DerivedCoreProperties.txt: Default_Ignorable_Code_Point"]
    out += dcp_lines
    out += ["", "# extracted/DerivedGeneralCategory.txt: Cf"]
    out += dgc_lines
    out.append("")
    return "\n".join(out)


def expand(ranges):
    out = set()
    for lo, hi in ranges:
        out.update(range(lo, hi + 1))
    return out


def ranges_of(points):
    return merge([(c, c) for c in points])


def render_invisible_table(merged):
    v = version()
    rel, digest = SOURCES["DerivedCoreProperties.txt"]
    total = sum(hi - lo + 1 for lo, hi in merged)
    lines = [
        "// @generated by crates/sky-diag/tools/gen_display_unsafe.py. Do not edit.",
        "//",
        f"// Unicode {v}:",
        f"//   {BASE_URL.format(v=v)}{rel}",
        f"//     sha256 {digest}",
        "//",
        "// Default_Ignorable_Code_Point & (XID_Start | XID_Continue): the",
        "// invisible characters `unicode-ident` would otherwise accept in an",
        "// identifier. The lexer rejects every one of them.",
        f"// {len(merged)} ranges, {total} code points.",
        "",
        "/// Sorted, non-overlapping, non-adjacent inclusive code point ranges.",
        "pub(crate) static DEFAULT_IGNORABLE_IDENT: &[(u32, u32)] = &[",
    ]
    lines += [f"    (0x{lo:04X}, 0x{hi:04X})," for lo, hi in merged]
    lines += ["];", ""]
    return "\n".join(lines)


def render_invisible_source(dcp_lines):
    v = version()
    rel, digest = SOURCES["DerivedCoreProperties.txt"]
    out = [
        f"# Verbatim Default_Ignorable_Code_Point records from the Unicode {v} UCD",
        "# (tools/gen_display_unsafe.py in crates/sky-diag). The lexer's",
        "# exhaustive test combines them with unicode-ident's XID properties,",
        "# independently of crates/sky-syntax/src/invisible_table.rs.",
        f"#   {rel} sha256 {digest}",
        "",
    ]
    out += dcp_lines
    out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="fail if outputs are stale")
    ap.add_argument("--ucd-dir", help="directory with local copies of the source files")
    args = ap.parse_args()

    dcp = load("DerivedCoreProperties.txt", args.ucd_dir)
    dcp_lines, di = records(dcp, "Default_Ignorable_Code_Point")
    _, xid_start = records(dcp, "XID_Start")
    _, xid_continue = records(dcp, "XID_Continue")
    dgc_lines, cf = records(load("DerivedGeneralCategory.txt", args.ucd_dir), "Cf")
    merged = merge(di + cf + [(lo, hi) for lo, hi, _ in EXTRA])
    invisible = ranges_of(expand(di) & (expand(xid_start) | expand(xid_continue)))
    outputs = {
        TABLE_RS: render_table(merged),
        SOURCE_TXT: render_source(dcp_lines, dgc_lines),
        INVISIBLE_RS: render_invisible_table(invisible),
        INVISIBLE_SOURCE_TXT: render_invisible_source(dcp_lines),
    }
    stale = []
    for path, text in outputs.items():
        if args.check:
            try:
                with open(path, encoding="utf-8") as f:
                    current = f.read()
            except FileNotFoundError:
                current = None
            if current != text:
                stale.append(os.path.relpath(path, REPO))
        else:
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
    if stale:
        sys.exit("stale: " + ", ".join(stale) + " (rerun tools/gen_display_unsafe.py)")
    n_inv = sum(hi - lo + 1 for lo, hi in invisible)
    print(
        f"Unicode {version()}: display-unsafe {len(merged)} ranges;"
        f" invisible-in-identifiers {len(invisible)} ranges, {n_inv} code points"
        + (" (up to date)" if args.check else "")
    )


if __name__ == "__main__":
    main()
