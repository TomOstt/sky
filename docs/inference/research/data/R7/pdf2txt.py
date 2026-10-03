import sys, pypdf
r = pypdf.PdfReader(sys.argv[1])
out = open(sys.argv[2], "w")
for i, p in enumerate(r.pages):
    out.write(f"\n=== page {i+1} ===\n")
    out.write(p.extract_text() or "")
print(len(r.pages), "pages")
