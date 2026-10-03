#!/bin/bash
# EXPERIMENTAL single-sample download throughput comparison:
#  (a) plain HTTPS resolve -> Xet bridge, single stream (what a naive Rust/curl client does)
#  (b) plain HTTPS resolve -> Xet bridge, 8 parallel Range requests
#  (c) huggingface_hub + hf_xet (native Xet reconstruction) into a throwaway cache
# Verifies sha256 of every result against the LFS oid from the tree API. No token is used.
set -u
S=/private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad
W=$S/research/R9-work/hub/dl
REPO=unsloth/Qwen3-0.6B-GGUF
FILE=Qwen3-0.6B-Q4_1.gguf
URL="https://huggingface.co/$REPO/resolve/main/$FILE"
rm -rf "$W"; mkdir -p "$W"; cd "$W" || exit 1
META=$(curl -sS -m 30 "https://huggingface.co/api/models/$REPO/tree/main" | python3 -c "
import sys,json
for e in json.load(sys.stdin):
  if e['path']=='$FILE': print(e['lfs']['oid'], e['size'], e.get('xetHash',''))")
SHA=$(echo $META | cut -d' ' -f1); SIZE=$(echo $META | cut -d' ' -f2)
echo "file=$FILE size=$SIZE lfs_sha256=$SHA"
now() { python3 -c 'import time; print(time.perf_counter())'; }
# warm-up (CDN edge), discarded
curl -sS -L -m 600 -o warm.bin "$URL"; rm -f warm.bin
# (a) single stream
t0=$(now); curl -sS -L -m 600 -o a.bin "$URL"; t1=$(now)
echo "a_single_stream_s=$(python3 -c "print(round($t1-$t0,2))") MBps=$(python3 -c "print(round($SIZE/1e6/($t1-$t0),1))") sha_ok=$([ "$(shasum -a 256 a.bin | cut -c1-64)" = "$SHA" ] && echo yes || echo NO)"
rm -f a.bin
# (b) 8 parallel ranges against the bridge URL (resolve redirect once)
LOC=$(curl -sS -I -m 30 "$URL" | tr -d '\r' | awk -F': ' 'tolower($1)=="location"{print $2}')
N=8; CH=$(( (SIZE + N - 1) / N ))
t0=$(now)
for i in $(seq 0 $((N-1))); do s=$((i*CH)); e=$(( (i+1)*CH - 1 )); [ $e -ge $SIZE ] && e=$((SIZE-1)); curl -sS -m 600 -r $s-$e -o part.$i "$LOC" & done; wait
cat $(for i in $(seq 0 $((N-1))); do echo part.$i; done) > b.bin; t1=$(now)
echo "b_8x_range_s=$(python3 -c "print(round($t1-$t0,2))") MBps=$(python3 -c "print(round($SIZE/1e6/($t1-$t0),1))") sha_ok=$([ "$(shasum -a 256 b.bin | cut -c1-64)" = "$SHA" ] && echo yes || echo NO)"
rm -f b.bin part.*
# (c) hf_xet via huggingface_hub into throwaway cache
t0=$(now)
P=$(HF_HUB_CACHE=$W/cache HF_HUB_DISABLE_PROGRESS_BARS=1 $S/venv/bin/python -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download('$REPO','$FILE'))" 2>/dev/null)
t1=$(now)
echo "c_hf_xet_s=$(python3 -c "print(round($t1-$t0,2))") MBps=$(python3 -c "print(round($SIZE/1e6/($t1-$t0),1))") sha_ok=$([ "$(shasum -a 256 "$P" | cut -c1-64)" = "$SHA" ] && echo yes || echo NO)"
find $W/cache -maxdepth 3 | sed "s|$W/||" | head -20
rm -rf "$W"
