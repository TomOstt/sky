#!/usr/bin/env python3
"""Generate Sky's Unicode tables (display-unsafe, invisible, NFKC).

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

3. sky-syntax, Unicode Normalization Form KC data for rejecting identifiers
   that are not in NFKC (the lexer never normalizes; the NFKC form is only
   used to write the error message):
     NFKC_QC=No and NFKC_QC=Maybe    (DerivedNormalizationProps.txt)
     Canonical_Combining_Class != 0  (UnicodeData.txt, field 3)
     full compatibility decompositions: UnicodeData.txt field 5, canonical
       and <tagged> mappings, applied recursively here (including the
       algorithmic Hangul syllable decomposition), so the lexer does one
       lookup per code point
     primary composites: two-code-point canonical decompositions that are
       not Full_Composition_Exclusion (DerivedNormalizationProps.txt)
   Hangul syllables themselves are composed and decomposed algorithmically
   by the lexer and are not in the tables. The NFKC conformance cases from
   NormalizationTest.txt are written out as a test fixture.

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
  crates/sky-syntax/src/nfkc_table.rs
      the NFKC tables above
  crates/sky-syntax/tests/data/nfkc-conformance.txt
      NFKC facts from NormalizationTest.txt: `source;nfkc` (or just `nfkc`
      when it is its own NFKC), one per line, as space-separated hex code
      points
  crates/sky-syntax/tests/data/nfkc-part1.txt
      the code points in Part 1 of NormalizationTest.txt, as hex ranges
      (`lo..hi`, or `cp`); every other code point is its own NFKC

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
# Inputs for the NFKC tables (sky-syntax only). Kept apart from SOURCES so the
# sky-diag outputs, whose headers list SOURCES, are unchanged by them.
NFKC_SOURCES = {
    "DerivedNormalizationProps.txt": (
        "DerivedNormalizationProps.txt",
        "98ac7f67d985fe781e317f6182e885e94cabb0c314769e6dd73e48b226931ccd",
    ),
    "UnicodeData.txt": (
        "UnicodeData.txt",
        "0736451de439ae7baf1425136617da495e09ee5afbe6e394374db7009ea08950",
    ),
    "NormalizationTest.txt": (
        "NormalizationTest.txt",
        "25a50d816764b04abfb4a646d3eb2b2a803284c3873d9a06757b94fe4513dde3",
    ),
}
# UnicodeData.txt has no "# Name-version.txt" header line; its pinned digest
# is the only version check.
NO_HEADER = {"UnicodeData.txt"}
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
NFKC_RS = os.path.join(SYNTAX_CRATE, "src", "nfkc_table.rs")
NFKC_CONFORMANCE_TXT = os.path.join(SYNTAX_CRATE, "tests", "data", "nfkc-conformance.txt")
NFKC_PART1_TXT = os.path.join(SYNTAX_CRATE, "tests", "data", "nfkc-part1.txt")
QC_RECORD = re.compile(r"^([0-9A-F]{4,6})(?:\.\.([0-9A-F]{4,6}))?\s*;\s*(NFKC_QC)\s*;\s*([NM])\s*(#.*)?$")
# Hangul syllable constants (Unicode section 3.12).
S_BASE, L_BASE, V_BASE, T_BASE = 0xAC00, 0x1100, 0x1161, 0x11A7
L_COUNT, V_COUNT, T_COUNT = 19, 21, 28
S_COUNT = L_COUNT * V_COUNT * T_COUNT
RECORD = re.compile(r"^([0-9A-F]{4,6})(?:\.\.([0-9A-F]{4,6}))?\s*;\s*([A-Za-z_]+)\s*(#.*)?$")


def version():
    return ".".join(map(str, UNICODE_VERSION))


def load(name, ucd_dir):
    rel, digest = {**SOURCES, **NFKC_SOURCES}[name]
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
    if name not in NO_HEADER and first != f"# {name[:-4]}-{version()}.txt":
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


def nfkc_qc(text, value):
    """Ranges with NFKC_QC=value ("N" or "M")."""
    out = []
    for line in text.splitlines():
        m = QC_RECORD.match(line)
        if m and m.group(4) == value:
            out.append((int(m.group(1), 16), int(m.group(2) or m.group(1), 16)))
    if not out:
        sys.exit(f"no NFKC_QC={value} records")
    return merge(out)


def unicode_data(text):
    """Nonzero combining classes, one-level decompositions (canonical and
    compatibility), and the canonical ones alone."""
    ccc, decomp, canonical = {}, {}, {}
    for line in text.splitlines():
        f = line.split(";")
        if len(f) != 15:
            sys.exit(f"UnicodeData.txt: bad line {line!r}")
        cp = int(f[0], 16)
        if int(f[3]):
            ccc[cp] = int(f[3])
        if f[5]:
            fields = f[5].split()
            tagged = fields[0].startswith("<")
            parts = [int(x, 16) for x in fields[1:] if tagged] or [int(x, 16) for x in fields]
            decomp[cp] = parts
            if not tagged:
                if not 1 <= len(parts) <= 2:
                    sys.exit(f"UnicodeData.txt: canonical decomposition of {cp:04X} has {len(parts)} parts")
                canonical[cp] = parts
    if not ccc or not decomp or not canonical:
        sys.exit("UnicodeData.txt: no combining classes or decompositions")
    return ccc, decomp, canonical


def full_decomposition(cp, decomp):
    """Full compatibility decomposition of one code point (NFKD, unordered)."""
    s = cp - S_BASE
    if 0 <= s < S_COUNT:
        out = [L_BASE + s // (V_COUNT * T_COUNT), V_BASE + (s % (V_COUNT * T_COUNT)) // T_COUNT]
        if s % T_COUNT:
            out.append(T_BASE + s % T_COUNT)
        return out
    if cp not in decomp:
        return [cp]
    return [x for part in decomp[cp] for x in full_decomposition(part, decomp)]


def ccc_ranges(ccc):
    """(lo, hi, class) runs of consecutive code points with the same class."""
    out = []
    for cp in sorted(ccc):
        if out and out[-1][1] + 1 == cp and out[-1][2] == ccc[cp]:
            out[-1] = (out[-1][0], cp, ccc[cp])
        else:
            out.append((cp, cp, ccc[cp]))
    return out


def render_nfkc_table(no, maybe, cccs, decomp, compose):
    v = version()
    lines = [
        "// @generated by crates/sky-diag/tools/gen_display_unsafe.py. Do not edit.",
        "//",
        f"// Unicode {v}:",
    ]
    for name in ("DerivedNormalizationProps.txt", "UnicodeData.txt"):
        rel, digest = NFKC_SOURCES[name]
        lines.append(f"//   {BASE_URL.format(v=v)}{rel}")
        lines.append(f"//     sha256 {digest}")
    count = lambda rs: sum(hi - lo + 1 for lo, hi in rs)
    full = {cp: full_decomposition(cp, decomp) for cp in sorted(decomp)}
    data, index = [], []
    for cp, d in full.items():
        if len(d) > 255:
            sys.exit(f"decomposition of {cp:04X} is too long")
        index.append((cp, len(data), len(d)))
        data.extend(d)
    if len(data) > 0xFFFF:
        sys.exit("decomposition data does not fit u16 offsets")
    lines += [
        "//",
        "// Data for checking that identifiers are in Unicode Normalization Form KC.",
        "// Hangul syllables are handled algorithmically and are not listed here.",
        f"// NFKC_QC=No: {len(no)} ranges, {count(no)} code points.",
        f"// NFKC_QC=Maybe: {len(maybe)} ranges, {count(maybe)} code points.",
        f"// {len(cccs)} combining-class runs, {len(index)} decompositions",
        f"// ({len(data)} code points), {len(compose)} primary composites.",
        "",
        "/// Unicode version these tables were generated from.",
        f"pub const UNICODE_VERSION: (u8, u8, u8) = ({UNICODE_VERSION[0]}, {UNICODE_VERSION[1]}, {UNICODE_VERSION[2]});",
        "",
        "/// NFKC_QC=No: sorted, non-overlapping, non-adjacent inclusive ranges.",
        "pub(crate) static NFKC_QC_NO: &[(u32, u32)] = &[",
    ]
    lines += [f"    (0x{lo:04X}, 0x{hi:04X})," for lo, hi in no]
    lines += [
        "];",
        "",
        "/// NFKC_QC=Maybe: sorted, non-overlapping, non-adjacent inclusive ranges.",
        "pub(crate) static NFKC_QC_MAYBE: &[(u32, u32)] = &[",
    ]
    lines += [f"    (0x{lo:04X}, 0x{hi:04X})," for lo, hi in maybe]
    lines += [
        "];",
        "",
        "/// Canonical_Combining_Class of every code point whose class is not 0:",
        "/// sorted, non-overlapping `(lo, hi, class)` runs.",
        "pub(crate) static CCC: &[(u32, u32, u8)] = &[",
    ]
    lines += [f"    (0x{lo:04X}, 0x{hi:04X}, {c})," for lo, hi, c in cccs]
    lines += [
        "];",
        "",
        "/// Full compatibility decomposition (canonical and compatibility mappings",
        "/// applied recursively, not yet reordered), sorted by code point:",
        "/// `(code point, offset, length)` into [`DECOMPOSITION_DATA`].",
        "pub(crate) static DECOMPOSITION: &[(u32, u16, u8)] = &[",
    ]
    lines += [f"    (0x{cp:04X}, {off}, {n})," for cp, off, n in index]
    lines += [
        "];",
        "",
        "/// The code points of every [`DECOMPOSITION`], concatenated.",
        "pub(crate) static DECOMPOSITION_DATA: &[u32] = &[",
    ]
    for i in range(0, len(data), 8):
        lines.append("    " + " ".join(f"0x{x:04X}," for x in data[i : i + 8]))
    lines += [
        "];",
        "",
        "/// Primary composites, sorted by `(first, second)`: `(first, second,",
        "/// composite)`.",
        "pub(crate) static COMPOSITION: &[(u32, u32, u32)] = &[",
    ]
    lines += [f"    (0x{a:04X}, 0x{b:04X}, 0x{c:04X})," for a, b, c in compose]
    lines += ["];", ""]
    return "\n".join(lines)


def render_nfkc_conformance(text):
    """NFKC facts from NormalizationTest.txt, deduplicated and sorted.

    Each test line c1;c2;c3;c4;c5 requires c4 == NFKC(c1) == NFKC(c2) ==
    NFKC(c3) == NFKC(c4) == NFKC(c5).
    """
    facts = set()
    for line in text.splitlines():
        if not line or line.startswith(("#", "@")):
            continue
        cols = [c.strip() for c in line.split("#")[0].split(";")[:5]]
        if len(cols) != 5 or not all(cols):
            sys.exit(f"NormalizationTest.txt: bad line {line!r}")
        for src in range(5):
            facts.add((cols[src], cols[3]))
    if not facts:
        sys.exit("NormalizationTest.txt: no test cases")
    v = version()
    rel, digest = NFKC_SOURCES["NormalizationTest.txt"]
    out = [
        f"# NFKC conformance facts from the Unicode {v} NormalizationTest.txt",
        "# (tools/gen_display_unsafe.py in crates/sky-diag). Each line is",
        "# `source;nfkc`, or just `nfkc` when the string is its own NFKC; strings",
        "# are space-separated hex code points.",
        f"#   {rel} sha256 {digest}",
        "",
    ]
    out += [a if a == b else f"{a};{b}" for a, b in sorted(facts)]
    out.append("")
    return "\n".join(out)


def render_nfkc_part1(text):
    """The code points of the Part 1 (character by character) test lines.

    UAX #15 conformance: every code point X not listed in Part 1 must satisfy
    X == NFKC(X) (and the same for the other forms).
    """
    points, part = set(), None
    for line in text.splitlines():
        if line.startswith("@Part"):
            part = line.split()[0]
            continue
        if part != "@Part1" or not line or line.startswith("#"):
            continue
        c1 = line.split(";")[0].split()
        if len(c1) != 1:
            sys.exit(f"NormalizationTest.txt: Part 1 line is not one code point: {line!r}")
        points.add(int(c1[0], 16))
    if not points:
        sys.exit("NormalizationTest.txt: no Part 1 lines")
    v = version()
    rel, digest = NFKC_SOURCES["NormalizationTest.txt"]
    out = [
        f"# Code points in Part 1 of the Unicode {v} NormalizationTest.txt",
        "# (tools/gen_display_unsafe.py in crates/sky-diag), as hex ranges",
        "# `lo..hi` or single code points. Every code point not listed is its",
        f"# own NFKC. {len(points)} code points.",
        f"#   {rel} sha256 {digest}",
        "",
    ]
    for lo, hi in ranges_of(points):
        out.append(f"{lo:04X}" if lo == hi else f"{lo:04X}..{hi:04X}")
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
    dnp = load("DerivedNormalizationProps.txt", args.ucd_dir)
    qc_no, qc_maybe = nfkc_qc(dnp, "N"), nfkc_qc(dnp, "M")
    _, fce = records(dnp, "Full_Composition_Exclusion")
    fce = expand(fce)
    ccc, decomp, canonical = unicode_data(load("UnicodeData.txt", args.ucd_dir))
    compose = sorted((d[0], d[1], cp) for cp, d in canonical.items() if len(d) == 2 and cp not in fce)
    nfkc_test = load("NormalizationTest.txt", args.ucd_dir)
    outputs = {
        TABLE_RS: render_table(merged),
        SOURCE_TXT: render_source(dcp_lines, dgc_lines),
        INVISIBLE_RS: render_invisible_table(invisible),
        INVISIBLE_SOURCE_TXT: render_invisible_source(dcp_lines),
        NFKC_RS: render_nfkc_table(qc_no, qc_maybe, ccc_ranges(ccc), decomp, compose),
        NFKC_CONFORMANCE_TXT: render_nfkc_conformance(nfkc_test),
        NFKC_PART1_TXT: render_nfkc_part1(nfkc_test),
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
        f" invisible-in-identifiers {len(invisible)} ranges, {n_inv} code points;"
        f" NFKC_QC=No {sum(hi - lo + 1 for lo, hi in qc_no)}, NFKC_QC=Maybe"
        f" {sum(hi - lo + 1 for lo, hi in qc_maybe)} code points, {len(compose)} composites"
        + (" (up to date)" if args.check else "")
    )


if __name__ == "__main__":
    main()
