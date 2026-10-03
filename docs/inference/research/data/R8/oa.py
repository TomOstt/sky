#!/usr/bin/env python3
# oa.py "query" [n] : OpenAlex relevance search, 2023+, prints date | arXiv id/doi | title
import sys, json, urllib.request, urllib.parse
q = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 15
url = "https://api.openalex.org/works?" + urllib.parse.urlencode({"search": q, "per-page": n, "filter": "from_publication_date:2023-01-01", "select": "title,publication_date,doi,cited_by_count"})
d = json.load(urllib.request.urlopen(url, timeout=60))
for w in d.get("results", []):
    doi = (w.get("doi") or "").replace("https://doi.org/10.48550/arxiv.", "arXiv:")
    print(w["publication_date"], "|", doi, "| cites", w.get("cited_by_count"), "|", (w["title"] or "")[:120])
