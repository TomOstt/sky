#!/usr/bin/env python3
# abs.py ID [ID...] : print title, dates and abstract from arxiv.org/abs/ID (curl, cached)
import sys, re, html, subprocess, os
here=os.path.dirname(os.path.abspath(__file__)); os.makedirs(f"{here}/cache",exist_ok=True)
for i in sys.argv[1:]:
    p=f"{here}/cache/abs_{i.replace('/','_')}.html"
    if not os.path.exists(p) or os.path.getsize(p)<1000:
        subprocess.run(["curl","-sL","--max-time","60",f"https://arxiv.org/abs/{i}","-o",p])
    s=open(p,encoding="utf-8",errors="replace").read()
    t=re.search(r'<h1 class="title[^"]*">(.*?)</h1>',s,re.S)
    a=re.search(r'<blockquote class="abstract[^"]*">(.*?)</blockquote>',s,re.S)
    d=re.search(r'<div class="submission-history">(.*?)</div>',s,re.S)
    clean=lambda x: re.sub(r"\s+"," ",html.unescape(re.sub(r"<[^>]+>"," ",x))).strip()
    print(f"### {i}: {clean(t.group(1)) if t else '?'}")
    if d: 
        dates=re.findall(r'\[v\d+\]</?[^>]*>?\s*([A-Z][a-z]{2}, \d+ [A-Z][a-z]{2} \d{4})',d.group(1)) or re.findall(r'(\w{3}, \d+ \w{3} \d{4})',d.group(1))
        print("dates:", dates[0] if dates else "?", "->", dates[-1] if dates else "?")
    print(clean(a.group(1)) if a else "NO ABSTRACT"); print()
