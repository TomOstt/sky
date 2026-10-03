#!/usr/bin/env python3
"""ft.py URL [-g REGEX ...] [-c N] [-n MAXLINES] [--head N]
Fetch a URL (cached under ./cache), convert HTML to plain text, and print either the head
or lines matching any regex (case-insensitive) with N lines of context."""
import sys, os, re, hashlib, subprocess, html, argparse

ap = argparse.ArgumentParser()
ap.add_argument("url")
ap.add_argument("-g", action="append", default=[])
ap.add_argument("-c", type=int, default=0)
ap.add_argument("-n", type=int, default=80)
ap.add_argument("--head", type=int, default=0)
ap.add_argument("--refresh", action="store_true")
a = ap.parse_args()

here = os.path.dirname(os.path.abspath(__file__))
cache = os.path.join(here, "cache"); os.makedirs(cache, exist_ok=True)
key = hashlib.sha1(a.url.encode()).hexdigest()[:16]
raw = os.path.join(cache, key + ".raw"); txt = os.path.join(cache, key + ".txt")
if a.refresh or not os.path.exists(raw) or os.path.getsize(raw) == 0:
    subprocess.run(["curl", "-sL", "--max-time", "90", "-A", "Mozilla/5.0 (research-bot)", a.url, "-o", raw], check=False)
data = open(raw, "rb").read()
if data[:4] == b"%PDF":
    sys.path.insert(0, os.path.join(here, "pylib"))
    import pypdf, io
    r = pypdf.PdfReader(io.BytesIO(data))
    t = "\n".join((p.extract_text() or "") for p in r.pages)
else:
    s = data.decode("utf-8", "replace")
    s = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", s)
    s = re.sub(r"(?i)<br\s*/?>|</(p|div|tr|li|h[1-6]|table|section|pre|td|th)>", "\n", s)
    s = re.sub(r"(?i)<(td|th)[^>]*>", " | ", s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s)
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"\n\s*\n+", "\n", s)
    t = s
open(txt, "w").write(t)
lines = t.split("\n")
print(f"[{a.url}] {len(lines)} lines, {len(t)} chars, cache={txt}")
if a.head:
    print("\n".join(lines[:a.head])); sys.exit()
if not a.g:
    print("\n".join(lines[:a.n])); sys.exit()
pats = [re.compile(g, re.I) for g in a.g]
shown = set(); out = 0
for i, l in enumerate(lines):
    if any(p.search(l) for p in pats):
        for j in range(max(0, i - a.c), min(len(lines), i + a.c + 1)):
            if j not in shown:
                shown.add(j); print(f"{j}: {lines[j][:400]}"); out += 1
        if out >= a.n: print("...[truncated]"); break
