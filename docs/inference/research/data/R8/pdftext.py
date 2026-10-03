import sys, pypdf, re
r = pypdf.PdfReader(sys.argv[1])
out = []
for i, p in enumerate(r.pages):
    try: t = p.extract_text() or ""
    except Exception as e: t = f"[page {i} error {e}]"
    out.append(f"\n=== PAGE {i+1} ===\n" + t)
open(sys.argv[2], "w").write("".join(out))
print(len(r.pages), "pages ->", sys.argv[2])
