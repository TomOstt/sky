#!/usr/bin/env python3
"""Lines of Rust per crate: production code vs comments vs in-src tests vs tests/ vs benches, for ~/sky/crates."""
import os, re, sys
root = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/sky/crates")
GENERATED = {"nfkc_table.rs", "display_unsafe_table.rs", "invisible_table.rs"}

def test_mod_files(src):
    """Files that are `#[cfg(test)] mod x;` declarations (test-only), resolved relative to the declaring file."""
    out = set()
    for dp, _, fs in os.walk(src):
        for f in fs:
            if not f.endswith(".rs"): continue
            p = os.path.join(dp, f)
            lines = open(p, encoding="utf-8").read().split("\n")
            for i, l in enumerate(lines):
                if l.strip() == "#[cfg(test)]" and i + 2 < len(lines):
                    pm = re.match(r'\s*#\[path\s*=\s*"([^"]+)"\]', lines[i + 1])
                    if pm and re.match(r"\s*mod\s+\w+\s*;", lines[i + 2]):
                        out.add(os.path.abspath(os.path.join(os.path.dirname(p), pm.group(1))))
                        continue
                if l.strip() == "#[cfg(test)]" and i + 1 < len(lines):
                    m = re.match(r"\s*(?:pub(?:\(crate\))?\s+)?mod\s+(\w+)\s*;", lines[i + 1])
                    if m:
                        name = m.group(1)
                        base = os.path.dirname(p) if f in ("lib.rs", "main.rs", "mod.rs") else os.path.join(os.path.dirname(p), f[:-3])
                        for cand in (os.path.join(base, name + ".rs"), os.path.join(base, name, "mod.rs")):
                            if os.path.exists(cand): out.add(os.path.abspath(cand))
                    # also #[path = "..."] patterns are ignored
    return out

def classify(path):
    """Return (code, comment, blank, inline_test) counts of a production file."""
    lines = open(path, encoding="utf-8").read().split("\n")
    code = com = blank = intest = 0
    i = 0; n = len(lines)
    while i < n:
        l = lines[i].strip()
        if l == "#[cfg(test)]" and i + 1 < n and re.match(r"\s*(pub\s+)?mod\s+\w+\s*\{", lines[i + 1]):
            # inline test module: count to the matching brace
            depth = 0; j = i + 1; started = False
            while j < n:
                depth += lines[j].count("{") - lines[j].count("}")
                if "{" in lines[j]: started = True
                j += 1
                if started and depth <= 0: break
            intest += j - i; i = j; continue
        if not l: blank += 1
        elif l.startswith("//"): com += 1
        else: code += 1
        i += 1
    return code, com, blank, intest

rows = []
for c in sorted(os.listdir(root)):
    cdir = os.path.join(root, c)
    if not os.path.isdir(cdir): continue
    src = os.path.join(cdir, "src")
    tmods = test_mod_files(src)
    tot = dict(code=0, com=0, blank=0, intest=0, testfiles=0, gen=0, testsdir=0, benches=0, src_all=0)
    for dp, _, fs in os.walk(src):
        for f in fs:
            if not f.endswith(".rs"): continue
            p = os.path.abspath(os.path.join(dp, f))
            nl = len(open(p, encoding="utf-8").read().split("\n")) - 1
            tot["src_all"] += nl
            if f in GENERATED: tot["gen"] += nl; continue
            if p in tmods: tot["testfiles"] += nl; continue
            a, b, d, e = classify(p)
            tot["code"] += a; tot["com"] += b; tot["blank"] += d; tot["intest"] += e
    for sub, key in (("tests", "testsdir"), ("benches", "benches")):
        d = os.path.join(cdir, sub)
        for dp, _, fs in os.walk(d):
            for f in fs:
                if f.endswith(".rs"):
                    tot[key] += len(open(os.path.join(dp, f), encoding="utf-8").read().split("\n")) - 1
    rows.append((c, tot))

hdr = f"{'crate':15} {'src(wc)':>8} {'prod code':>9} {'comments':>8} {'blank':>6} {'src tests':>9} {'generated':>9} {'tests/':>7} {'benches/':>8}"
print(hdr)
T = dict.fromkeys(rows[0][1], 0)
for c, t in rows:
    print(f"{c:15} {t['src_all']:8} {t['code']:9} {t['com']:8} {t['blank']:6} {t['testfiles']+t['intest']:9} {t['gen']:9} {t['testsdir']:7} {t['benches']:8}")
    for k in T: T[k] += t[k]
print(f"{'TOTAL':15} {T['src_all']:8} {T['code']:9} {T['com']:8} {T['blank']:6} {T['testfiles']+T['intest']:9} {T['gen']:9} {T['testsdir']:7} {T['benches']:8}")
