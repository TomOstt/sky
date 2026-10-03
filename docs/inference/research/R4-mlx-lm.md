# R4: mlx-lm, the generation stack Sky must beat (code trace and end-to-end measurements)

Agent R4. Scope: the mlx-lm generation loop, the Qwen3 model code, KV caches, speculative and batch generation, the
server, wired memory and memory reporting, published numbers, and end-to-end measurements on the local M2 Pro.
MLX core (graph, eval, Metal backend, kernels) belongs to R3. I cite `R3-mlx-core.md`, `R3-kernels.md` and
`R3-issues.md` instead of repeating them.

Labels: **VERIFIED** (source path:line at the cloned commit, official URL, maintainer link, or our own command
plus raw output), **LIKELY** (reason given), **EXPERIMENTAL**, **MOONSHOT**.

---

## 0. Provenance and environment

- `$S` = `/private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad`.
- Source (both shallow clones):
  - mlx-lm `$S/src/mlx-lm` @ `53b9af378278d6dd5dacac447edeb0f523b16253` (2026-10-01, "Update acknowledgments…
    (#1936)").
  - mlx `$S/src/mlx` @ `0840c42a7aab95fcd4a39e8bc56aa6887470c043` (2026-10-01).
  - Paths below are relative to `$S/src/mlx-lm/mlx_lm/`, unless prefixed `mlx/`, which means `$S/src/mlx/`.
- Installed and measured: mlx 0.32.3, mlx-metal 0.32.3, mlx-lm 0.32.0 (`$S/venv`).
  - `generate.py`, `sample_utils.py`, `models/qwen3.py`, `models/base.py`, `models/cache.py`, `generate_utils.py`
    and `benchmark.py` are byte-identical between the install and the clone (checked with `diff -q`).
  - `server.py` differs only in tool-call JSON parsing (VERIFIED, `diff`).
- Machine: Apple M2 Pro, 16-core GPU, `applegpu_g14s`, 16 GB, macOS 26.6.1.
  - `max_recommended_working_set_size` = 12.71 GB (`mx.device_info()`).
  - Other agents ran CPU jobs concurrently. All timings ran under `bash $S/benchlock.sh`.
  - R3 notes that a 4K external display is attached, and its scan-out consumes some DRAM bandwidth.
- Models (HF cache): `mlx-community/Qwen3-0.6B-4bit` and `mlx-community/Qwen3-8B-4bit`.
  - Both are affine 4-bit, group 64, with bf16 scales, biases and norms (VERIFIED, safetensors headers, `R4/r4_bytes.out`).
- Every script, raw JSON and output file is in `$S/research/R4/` (list in §12).

---

## 1. Headline numbers (M2 Pro, mlx 0.32.3 / mlx-lm 0.32.0, greedy, batch 1)

All rows are VERIFIED by our measurements: medians of ≥5 warm runs, raw data in §7.

| Metric | Qwen3-0.6B-4bit | Qwen3-8B-4bit |
|---|---|---|
| Decode tok/s, 128-token prompt + 256 generated (stream_generate, wall) | **267.4** (ITL 3.76 ms) | **37.4** (ITL 26.4 ms) |
| Decode tok/s, `generate_step` steady state (same process as the ablations) | 281 (3.55 ms) | 38.8 (25.79 ms) |
| Prefill tok/s at 128 / 512 / 1024 / 2048 tokens | 2,350 / 2,714 / – / 2,635 | 200 / 220 / 217 / 218 |
| TTFT (call → first token, warm) at 128 / 512 / 1024 / 2048 tokens | 55 ms / 189 ms / – / 777 ms | 0.64 s / 2.33 s / 4.73 s / 9.41 s |
| TTFT at 4096 tokens | 1.82 s | 20.7 s |
| Peak MLX memory (decode / prefill 2048) | 0.59 / 1.46 GB | 4.92 / 5.61 GB |
| OS lifetime max phys_footprint (decode run / prefill run) | 0.89 / 2.04 GB | 5.25 / 6.29 GB |
| Bytes streamed per decode token (weights) | 335.4 MB (tied embedding = lm_head) | 4,257.7 MB |
| Effective bandwidth in decode, % of 182.5 GB/s achievable | 99.5 GB/s, **54.5%** | 166 GB/s, **91%** |
| Same, % of the 200 GB/s spec | 49.8% | 83% |
| GPU dispatches per decode token (exact, swizzled Metal encoder) | **541** (19.3/layer) | **693** (19.25/layer) |
| Memory barriers / compute encoders per token | 362 / 11 | 457 / 23 |
| Spec. decoding (8B + 0.6B draft), best k | – | **slower at every k** (0.37–0.74× of 38.9 tok/s) |

---

## 2. The generation loop (`generate.py`)

### 2.1 Prefill chunking (VERIFIED)

- `DEFAULT_PREFILL_STEP_SIZE = 2048` (generate.py:48). The CLI flag is `--prefill-step-size`
  (generate.py:171-177).
- `generate_step` (generate.py:304-473) processes all prompt tokens but the last in chunks of
  `min(prefill_step_size, remaining)` (generate.py:428-454). Each chunk does four things:
  - `model(chunk[None], cache)`, logits discarded;
  - `quantize_cache_fn` (KV quantization if enabled);
  - `mx.eval([c.state for c in prompt_cache])`, a full CPU↔GPU sync per chunk (generate.py:441);
  - `mx.clear_cache()`, which frees MLX's buffer cache (generate.py:454).
- The last prompt token always goes through the decode-style `_step` (generate.py:456). So a 2048-token prompt
  costs one 2047-token forward plus one 1-token forward. The LM head runs only on that last token.
- `speculative_generate_step` uses its own `_prefill` with a default chunk of **512** (generate.py:487, 578-586).
- `PromptProcessingBatch.prompt` in the batch path chunks the same way and right-pads ragged prompts
  (generate.py:1163-1209).

### 2.2 How async_eval pipelines step t+1 with step t (VERIFIED)

The decode loop is generate.py:456-473:

```
y, logprobs = _step(prompt_last)            # build graph for token 0
mx.async_eval(y, logprobs)                  # encode + commit, no wait
while True:
    next_y, next_logprobs = _step(y)        # build graph t+1 (y is still a lazy array)
    mx.async_eval(next_y, next_logprobs)    # encode + commit t+1 while the GPU runs t
    if n == 0: mx.eval(y)
    yield y.item(), logprobs                # wait for token t only
    if n % 256 == 0: mx.clear_cache()
    y, logprobs = next_y, next_logprobs
```

- Graph construction and Metal encoding of step t+1 happen on the Python thread while the GPU executes step t.
  R3 §1.2 verifies that all GPU encoding happens on the calling thread inside `eval_impl`.
- `y.item()` waits only on token t's event, because token t's command buffers were committed before t+1's.
- Steady-state inter-token latency (ITL) is max(GPU time, CPU build + encode) plus a small sync gap.
- `async_eval` is not fully asynchronous for large models (LIKELY; mechanism from R3 §1.2 step 5):
  - MLX blocks inside eval when more than `MAX_ACTIVE_TASKS = 10` command buffers are in flight.
  - Qwen3-8B uses 23 compute encoders per token (measured, §7.C).
  - So in a synchronous test, the `async_eval` call itself took 13.6 ms of the 27.3 ms step (§7.E). It was
    waiting for the GPU, not encoding.
  - Pipelining still works, because the queue never drains.

### 2.3 Per-token CPU synchronization points (VERIFIED)

- `generate_step`: one `y.item()` per token (generate.py:469), plus `mx.eval(y)` on the first token
  (generate.py:465).
- `stream_generate` (generate.py:658-760), per token:
  - `token in tokenizer.eos_token_ids`, a Python set (728);
  - `detokenizer.add_token(token)` (731);
  - `mx.get_peak_memory()` (744);
  - a `GenerationResponse` dataclass (735-746). It carries the `logprobs` mx.array lazily and never forces it.
- `speculative_generate_step` makes one full sync per draft/verify round: `mx.eval(tokens, draft_tokens)` and two
  `.tolist()` calls (generate.py:620-622). There is **no cross-round pipelining**: the GPU idles while Python
  compares tokens, rewinds caches and builds the next draft graph.
- `GenerationBatch._step` (batch and server path) also pipelines: `async_eval(next)` then `mx.eval(inputs,
  current_logprobs)` and `inputs.tolist()` (generate.py:1401-1411).
- The server additionally calls `r.logprobs[r.token].item()` per token (server.py:867, and 986 on the
  single-request path).
  - **Tested hypothesis, refuted:** this extra `.item()` does not cost throughput (§7.F; 25.71 vs 25.81 ms per
    token on 8B).

### 2.4 Sampling (`sample_utils.py`) (VERIFIED)

- Sampling runs on the GPU, inside the same lazy graph as the forward pass. There is no host round trip for
  sampling.
- `temp == 0` returns `greedy_sampler` = `mx.argmax(logprobs, -1)` (sample_utils.py:49-50, 133-134).
- The sampler is always applied to `logprobs = logits - logsumexp(logits)` (generate.py:418-419). So greedy decode
  still pays a full-vocabulary `LogSumExp` and `Subtract`: 2 dispatches, measured at 0.17 ms/token on 8B and
  0.10 ms on 0.6B (§7.E).
- With temperature, a chain of `mx.compile`d filters runs, then `categorical_sampling` =
  `mx.random.categorical(logits * (1/temp))` (sample_utils.py:271-273). All are
  `@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)`.
  - top-p (212-229): full `mx.sort` of the vocabulary, fp32 `exp`, reverse exclusive `cumsum`, threshold via
    `take_along_axis`, then `where`. This is O(V log V) on the GPU per token.
  - min-p (162-209): `max` plus a log-space threshold; `argpartition` only when `min_tokens_to_keep > 1`.
  - top-k (137-159): `argpartition(-logprobs, k-1)` plus `put_along_axis(-inf)`.
  - XTC (232-268): softmax and masks, plus `mx.random.uniform`.
- Order of the chain: top_p → min_p → xtc → top_k (sample_utils.py:55-64).

### 2.5 Logits processors, stop conditions, detokenizer (VERIFIED)

- Logits processors (sample_utils.py:76-130):
  - logit bias: `logits.at[:, idx].add`;
  - repetition penalty: gather the last 20 tokens' logits, a sign-aware scale, scatter back (276-306);
  - presence and frequency penalties (309-361).
- When processors are active, `generate_step` maintains a growing `tokens` array with `mx.concat` per step
  (generate.py:407-414). That is O(T) per step, on the GPU, lazily.
- Stop conditions in `stream_generate`: an EOS-id set check per token (728) and `max_tokens` (732).
- The batch/server path uses an Aho-Corasick automaton over stop-token sequences (`StopSequences`,
  generate.py:855-926). It also uses a text-level `TextStateMachine` for think/tool markers and stop words
  (929-1059).
- Detokenizer: Qwen3's `tokenizer.json` has a ByteLevel decoder, so `BPEStreamingDetokenizer` is used
  (tokenizer_utils.py:199-262, selected at 686-690).
  - Per token: a table lookup, `bytearray` decode, and a string append. That is O(1) amortized, microseconds.
  - It runs on the CPU while the GPU computes the next token, so it is hidden.
- The per-call detokenizer construction that cost 51–77 ms of TTFT in 0.28–0.31.3 (mlx-lm#1435, root cause in
  a commenter's bisect) is gone at HEAD.
  - `TokenizerWrapper.detokenizer` now returns `copy.copy(self._detokenizer)` + `reset()`
    (tokenizer_utils.py:511-519).
  - The token map is built once at load (tokenizer_utils.py:206-211).

### 2.6 How mlx-lm computes its numbers (VERIFIED, generate.py:721-760, benchmark.py)

- **prompt_tps** = `prompt.size / prompt_time`.
  - `prompt_time` runs from just before the first `next()` on the token generator to the first token yielded
    (722-726).
  - It includes prefill of N−1 tokens, the decode step on the last prompt token, and building/launching the
    *second* step.
  - It excludes tokenization, the chat template, and `wired_limit` setup (entered at 721).
  - So it is a TTFT-derived rate, not a pure prefill FLOP rate.
- **generation_tps** = `(n+1) / (now − tic)`, where `tic` is reset right after the first token (727, 743).
  - The numerator counts the first token, but its time is excluded.
  - It is biased high by (n+1)/n: +0.4% at 256 tokens, +14% at 8 tokens (we observed 327 vs 287 tok/s at 8
    tokens).
- **peak_memory** = `mx.get_peak_memory() / 1e9` (744, 758).
  - This is the MLX allocator's process-wide high-water mark, in decimal GB.
  - mlx-lm never resets it, so in a long-lived process it includes earlier work (load transients, earlier
    prompts).
  - It excludes Python, the tokenizer and Metal driver overheads. We report OS `phys_footprint` alongside it.
- **`mlx_lm.benchmark`**:
  - prompts are **random token ids** (benchmark.py:127-129);
  - EOS is emptied so generation always runs to `-g` (121);
  - one warm-up run, then `-n` trials (default 5);
  - it reports the **arithmetic mean** of `prompt_tps`, `generation_tps` and `peak_memory` (156-176);
  - defaults are `-p 512 -g 1024`, while BENCHMARKS.md uses `-p 2048 -g 128`.
- Our own measurements (§7) record both mlx-lm's self-reported values and wall-clock per-token timestamps,
  report medians, and reset peak memory per trial.

---

## 3. Qwen3 decode step: every op, in order (`models/qwen3.py`, `models/base.py`)

### 3.1 Op list per layer for one decode token (B = 1, L = 1)

Code: TransformerBlock 127-137, Attention 70-100, MLP 110-111. "Dispatches" was measured exactly per op class
(§7.C).

| # | Op (source) | MLX primitive | Dispatches |
|---|---|---|---|
| 1 | `input_layernorm(x)` (qwen3.py:133) | `RMSNorm` (`mx.fast.rms_norm`, mlx nn normalization.py:163-164) | 1 |
| 2-4 | `q_proj`, `k_proj`, `v_proj` (78) | 3 × `QuantizedMatmul` → `affine_qmv_fast` (M=1) | 3 |
| 5 | `q.reshape(B,L,H,D)`; `q_norm` (80-82) | Reshape (view) + `RMSNorm` over head_dim | 1 |
| 6 | `k.reshape`; `k_norm` (83-85) | Reshape + `RMSNorm` | 1 |
| 7 | `.transpose(0,2,1,3)` for q, k, v (80-86) | Transpose (views) | 0 |
| 8-9 | `rope(q, offset)`, `rope(k, offset)` (89-90) | 2 × `RoPE` (`mx.fast.rope`, traditional=False, base 1e6) | 2 |
| 10-11 | `cache.update_and_fetch(k, v)` (91; cache.py:347-375) | 2 × `SliceUpdate` (in place, buffer donated) + 2 × `Slice` views `[..., :offset, :]` | 2 |
| 12 | `scaled_dot_product_attention` (96-98; base.py:110-139) | `ScaledDotProductAttention` → `sdpa_vector` (1 pass); 2-pass when kL ≥ 1024 on `d`/`s` devices or GQA with kL ≥ 4096 (mlx/mlx/backend/metal/scaled_dot_product_attention.cpp:1560-1570) | 1 (2) |
| 13 | `output.transpose(0,2,1,3).reshape(B,L,-1)` (99) | views (no copy at L = 1) | 0 |
| 14 | `o_proj` (100) | `QuantizedMatmul` | 1 |
| 15 | `h = x + r` (134) | `Add` | 1 |
| 16 | `post_attention_layernorm(h)` (135) | `RMSNorm` | 1 |
| 17-18 | `gate_proj(x)`, `up_proj(x)` (111) | 2 × `QuantizedMatmul` | 2 |
| 19 | `swiglu(g, u)` (activations.py:9-11, `@partial(mx.compile, shapeless=True)`) | `Compiled` (silu(g)·u fused) | 1 |
| 20 | `down_proj` (111) | `QuantizedMatmul` | 1 |
| 21 | `out = h + r` (136) | `Add` | 1 |
| | **Per layer** | 7 qmv, 4 rms_norm, 2 rope, 2 KV writes, 1 sdpa, 2 add, 1 swiglu | **19** |

Per token, outside the layers:
- the embedding: `QuantizedEmbedding.__call__` gathers weight, scales and biases, then `Quantize` (dequantize):
  3 `Gather` + 1 kernel (mlx nn quantized.py:149-158);
- the final `RMSNorm`;
- the LM head: `lm_head` for 8B, or `embed_tokens.as_linear` for the tied 0.6B (qwen3.py:189-194), 1 qmv;
- sampling: `LogSumExp`, `Subtract`, `ArgReduce` (3).

Totals: 8B = 36 × 19 + 9 = **693**; 0.6B = 28 × 19 + 9 = **541**. Both match the exact Metal counts in §7.C
(VERIFIED).

### 3.2 Fusion and compilation status (VERIFIED)

- **No projection fusion.** q/k/v are three separate `nn.Linear`/`QuantizedLinear` modules (qwen3.py:55-58), and
  gate/up are separate (106-108).
  - Each is a separate qmv dispatch reading the same activation vector. Only the activation read is
    duplicated; the weights are distinct, so no bytes are wasted.
  - The cost is dispatch count, plus poor occupancy of the small k/v matrices.
  - R3 measured k/v_proj (4096→1024) at 54–63% of bandwidth (R3-mlx-core §7.B).
- **`mx.compile` in the decode path:** only the SwiGLU helper (activations.py:9-11) and the sampler helpers
  (sample_utils.py:137-273, active only for temp > 0).
  - The model forward and decode step are not compiled.
  - Nothing compiles across RMSNorm, rope, cache writes or residuals.
  - MLX's compiler fuses only elementwise ops, so it could not fuse norm→matmul or rope→cache anyway (R3 §1.4).
- **Masks:** a single-token decode with `KVCache` gets `mask=None` (cache.py:141-151 via `make_mask`). Prefill
  gets the string `"causal"`, which selects the fused causal kernel without materializing a mask (base.py:45-55).

---

## 4. KV caches (`models/cache.py`)

### 4.1 KVCache (VERIFIED, cache.py:339-420)

- Layout: `(B, n_kv_heads, capacity, head_dim)` in model dtype (bf16 here). `step = 256` (340).
- **Growth** (349-364): when `offset + new > capacity`, it allocates `zeros` of `ceil(new/256)·256` slots and
  `mx.concatenate([old, zeros])`.
  - That is a full copy of the layer's K and V every 256 tokens, which is O(n²/256) traffic.
  - Amortized it is negligible: about 26 µs/token at 4K context for 8B (LIKELY, arithmetic: 604 MB of K+V copied
    every 256 tokens).
  - For long prefills, each 2048-token chunk triggers one growth copy.
- **Write path:** `self.keys[..., prev:offset, :] = keys` (367-368) becomes a `SliceUpdate` primitive.
  - `SliceUpdate::eval_gpu` does `copy_gpu(in, out)`, and that copy is skipped when the input buffer is donatable
    (refcount 1). Then `copy_gpu_inplace` writes only the new slice (mlx/mlx/backend/metal/indexing.cpp:733-768;
    mlx/mlx/backend/metal/copy.cpp:13-24; mlx/mlx/backend/common/copy.h:25-48; mlx/mlx/backend/common/utils.h:218-223).
  - We verified the in-place path empirically: exactly **1 dispatch per SliceUpdate** in the Metal count (§7.C).
    A non-donated update would be 2 dispatches plus a full-cache copy.
- **Fetch:** `keys[..., :offset, :]` is a strided view (371-375). The SDPA vector path accepts it without a copy
  (`kv_copy_unless`, scaled_dot_product_attention.cpp:1519-1537).
- **trim(n)** only decrements `offset` (391-394). That makes rewind O(1) and is what speculative decoding and
  prompt-cache reuse rely on.
- **Copy-on-write hazard (VERIFIED code; LIKELY effect):**
  - `copy.deepcopy` of a cache shares buffers, because `mx.array.__deepcopy__` returns `mx::array(self)`, a new
    handle to the same `ArrayDesc` (mlx/python/src/array.cpp:559-563).
  - So the first write after a deepcopy cannot donate, and copies the whole buffer.
  - The server's `fetch_nearest_cache` deep-copies on every hit (cache.py:1656, 1662, 1670). So 2× the
    conversation's KV transiently exists while the cached entry is alive.
  - This matches the symptom in mlx-lm#1395, though its "deepcopy doubles at fetch" mechanism is slightly off:
    the duplication happens at the first write.

### 4.2 RotatingKVCache (VERIFIED, cache.py:423-593)

- Used when `max_kv_size` is set: `make_prompt_cache` builds `RotatingKVCache(max_size, keep=4)` per layer
  (35-38).
- Decode (S = 1) uses `_update_in_place`, a ring buffer. It grows in 256 steps up to `max_size`, then rotates
  `_idx` back to `keep` (482-520).
- Multi-token updates use `_update_concat`: temporal reorder, trim, concatenate (462-480), which copies.
- It is trimmable only while `offset < max_size` (544-545), so speculative decoding breaks once the window
  wraps. `to_quantized` raises NotImplementedError (553-554, mlx-lm#1573).
- The mask for S = 1 is needed only when `window_size < max_size` (556-580).

### 4.3 QuantizedKVCache and the quantized SDPA path (VERIFIED)

- Storage is a `(packed uint32, scales, biases)` tuple for K and for V. `step = 256`, and growth concatenates six
  arrays (251-302).
- Every update runs `mx.quantize` on the new K/V and does six slice writes (296-300).
- It is enabled by `--kv-bits` once `offset ≥ quantized_kv_start` (`maybe_quantize_kv_cache`, generate.py:296-301).
  - `DEFAULT_QUANTIZED_KV_START = 5000` (47).
  - The library default now matches the CLI (generate.py:317, 490) after mlx-lm#1651 / PR #1819, merged by
    michalk8 on 2026-09-09.
- Attention with a quantized cache is **not** fused (base.py:64-107):
  - `quantized_matmul(q, K) → where(mask) → softmax(precise) → quantized_matmul(scores, V)`, with GQA handled by
    `expand_dims`;
  - the full `(heads × L × kL)` fp32 scores tensor is materialized. In prefill that is chunk × context, which
    caused the peak-memory inversion in mlx-lm#1587 (§9).

### 4.4 Prompt cache save/load/trim, and server reuse (VERIFIED)

- `save_prompt_cache` (43-79) serializes each cache's `.state` plus class names and metadata into safetensors.
  `load_prompt_cache` (82-112) rebuilds them via `from_state`.
- `trim_prompt_cache` (122-138) requires all caches to be trimmable.
- CLI: `mlx_lm.cache_prompt` writes the cache. `mlx_lm.generate --prompt-cache-file` loads it and sends only the
  suffix after the cached prefix, by templating a `<query>` placeholder (generate.py:2009-2024, 2078-2088).
- Server: `LRUPromptCache` (1599-1741) keyed by a token trie (`PromptTrie`, 1508-1596).
  - `fetch_nearest_cache` (1652-1672) tries, in order:
    - an exact match: return a deepcopy;
    - a longer cached sequence with a trimmable cache: deepcopy, trim to the common prefix, process the
      remainder;
    - a shorter cached prefix: deepcopy, process the rest.
  - `insert_cache` drops prefix entries when the cache is trimmable (1697-1703).
  - Eviction is by count (`--prompt-cache-size`) and bytes (`--prompt-cache-bytes`) with type-aware LRU over
    assistant/user/system (1606-1635).
  - The batched server stores caches at segment boundaries and at completion (server.py:830-845, 879). The
    single-request path stores at 1006.
  - Open issues: streamed generations not stored (#1802); per-request `chat_template_kwargs` bypass the cache
    (#1803).

---

## 5. Speculative decoding, batch generation, server

### 5.1 `speculative_generate_step` (VERIFIED, generate.py:476-655)

- **Draft model:** any mlx-lm model with the same vocabulary size (CLI check at 2095-2096). It has its own cache
  list. With a supplied `prompt_cache`, the first `len(model.layers)` entries are the target's (527-531).
- `num_draft_tokens`: CLI default **3** (215-220), function default 2 (482).
- Each round:
  1. draft `k = min(remaining, num_draft_tokens)` tokens sequentially with the draft model. Each is
     `async_eval`'d, but the chain is serial (592-600);
  2. verify by running the target on `[y, d1..dk]` (k+1 tokens) in **one forward**, logits for all k+1 positions
     (557-576, 619);
  3. `mx.eval` and `.tolist()` (620-622);
  4. accept while `target_token[i] == draft_token[i]`, then emit the target's own token at the first mismatch,
     or the bonus token (624-635).
- **Cache rewind:** target trimmed by `k − n_accepted`, draft by `max(k − n − 1, 0)` (588-590). When all k are
  accepted, the last draft token is re-fed to the draft (646-649).
  - Requires trimmable caches (533-537), so hybrid/SSM `ArraysCache` models cannot use it (mlx-lm#1446).
- **Sampling equivalence:** at temp > 0 both models *sample* with the same sampler. A draft token is accepted only
  if the target's independent sample equals it.
  - This is **not** Leviathan/Chen rejection sampling.
  - Each emitted token is the target's own sample given a correct prefix, so it is exact in distribution
    (LIKELY, by construction).
  - Acceptance is Σ p·q instead of Σ min(p, q), i.e. lower at high temperature.
  - At temp 0 it is exact up to bf16 tie-breaking. Batched (M = k+1) and sequential (M = 1) logits differ in
    reduction order. Contributors' logit audits found exactly tied logits at divergence points (mlx-lm#1470,
    documented in #1592). We observed divergences in 3 of 9 speculative configurations (§7.D).
- Logits processors are supported, with `prev_tokens` bookkeeping (562-574).
- **Not present at HEAD:** prompt-lookup or n-gram drafting, EAGLE, Medusa, MTP heads.
  - MTP weights are dropped at load: qwen3_5.py:350, glm4_moe.py:334, deepseek_v32.py:486, kimi_linear.py:492
    (VERIFIED, grep).
  - Native MTP for Qwen3.5 is the open PR mlx-lm#990. zcbenz requested changes ("sampler is coupled with
    generation step…"; R3-issues E6).
  - n-gram self-speculation (PRs #1497/#1498) was closed: "At the moment we don't accept AI written architecture
    changes" (zcbenz, 2026-08-26).

### 5.2 BatchGenerator and the server (VERIFIED)

- `BatchGenerator` (1512-1884): continuous batching with `completion_batch_size = 32` and
  `prefill_batch_size = 8` defaults (1531-1532).
  - Each `_next()` first runs one decode step for the whole generation batch (1788-1795), then admits and
    prefills up to `prefill_step_size` tokens of waiting prompts (1801-1858).
  - `BatchKVCache` handles left padding and merges.
  - With 1 sequence it matches `generate_step`: 25.81 vs 25.59 ms ITL on 8B (§7.F).
- The server batches only if there is no draft model, `seed is None` and `kv_bits is None` (server.py:366-369,
  649-654). Otherwise `_serve_single` uses `stream_generate`, and that is the only route to speculative decoding
  in the server.

---

## 6. Wired memory and memory reporting (VERIFIED)

- `maybe_set_recommended_wired_limit()` sets `mx.set_wired_limit(max_recommended_working_set_size)` and returns
  the old limit (utils.py:73-82). On this machine that is 12.71 GB.
- `wired_limit(model, streams)` (generate.py:228-263) wraps every `stream_generate` call (721):
  - it sets the limit;
  - it warns if the model is > 0.9 × the recommended working set (246-254);
  - on exit it **synchronizes the generation stream** and restores the old limit (258-263).
- `BatchGenerator` sets the limit in `__init__` and restores it in `close()` (1564, 1570-1574). The server sets it
  once at startup (server.py:1932).
- The MLX side (residency sets, default limit 0) is in R3 §2.3.
- Per a commenter's measurement in mlx-lm#1435 (M5), `wired_limit` costs about 2.1 ms per call. Not verified
  here.
- `--verbose` prints `Peak memory: X GB` from `response.peak_memory` (generate.py:805). §2.6 explains what that
  number is and is not.

---

## 7. Measurements (all timing runs through `bash $S/benchlock.sh`; medians of ≥ 5 warm runs unless stated)

### 7.A Decode, prefill, TTFT, peak memory (VERIFIED, our measurement)

Script `R4/r4_gen.py`:
- it loads with `mlx_lm.load`, sets `tokenizer.eos_token_ids = []` so it always generates `max_tokens`, and
  calls `stream_generate` with token ids (no tokenization in the timing);
- the prompt is real text (mlx-lm/mlx docs) truncated to exactly N tokens;
- one warm-up, then 5 trials, with `mx.reset_peak_memory()` per trial.

Commands:
- `bash $S/benchlock.sh $S/venv/bin/python r4_gen.py <model> decode`
- `bash $S/benchlock.sh $S/venv/bin/python r4_gen.py <model> prefill 128 512 [1024] 2048`

Outputs: `A_06b.out`, `A_8b.out`, `raw_A_*.json`.

Decode, 128-token prompt, 256 tokens generated:

| Model | wall decode tok/s (5 runs) | median ITL | mlx-lm `generation_tps` | TTFT | peak (MLX) |
|---|---|---|---|---|---|
| 0.6B | 266.0, 261.4, **267.4**, 271.4, 270.5 | 3.762 ms | 268.5 | 55.2 ms | 0.588 GB |
| 8B | 37.74, 37.86, **37.44**, 36.13, 36.39 | 26.42 ms | 37.59 | 643 ms | 4.923 GB |

Prefill/TTFT (`max_tokens = 1`; TTFT = wall time from the call to the first response):

| Model | L = 128 | 512 | 1024 | 2048 |
|---|---|---|---|---|
| 0.6B TTFT | 54.5 ms | 188.6 ms | – | 777.3 ms |
| 0.6B prefill tok/s (L/TTFT; mlx-lm prompt_tps) | 2,350 (2,368) | 2,714 (2,723) | – | 2,635 (2,637) |
| 0.6B peak MLX memory | 0.588 GB | 0.896 GB | – | 1.460 GB |
| 8B TTFT | 0.639 s | 2.332 s | 4.730 s | 9.409 s |
| 8B prefill tok/s (L/TTFT; mlx-lm prompt_tps) | 200 (200.4) | 219.5 (219.6) | 216.5 (216.5) | 217.7 (217.7) |
| 8B peak MLX memory | 4.92 GB | 5.22 GB | 5.42 GB | 5.61 GB |

- OS `lifetime_max_phys_footprint`: 0.6B 0.89 GB (decode run), 2.04 GB (prefill run). 8B 5.25 GB (decode run),
  6.29 GB (prefill run, includes 2048-token chunk activations and the allocator cache).
- **8B prefill is compute-bound from 128 tokens up.** It achieves 3.3 TFLOPS on the linear layers alone
  (2 × 7.57 G × 218), which is:
  - 58% of the 5.73 TFLOPS theoretical FMA peak;
  - 61% of the 5.4 TFLOPS that R3's FMA probe reached;
  - ~75% of what MLX's qmm kernel reaches in isolation on 32B shapes (4.4 TFLOPS, R3 `r3_out_prefill_1.txt`).
- 0.6B prefill reaches 2.3–2.4 TFLOPS (41%). It is smaller-matrix bound.
- Small-prompt prefill is less efficient: 8B at 128 tokens is ~76 ms slower than a linear fit of the 512–2048
  points. There is no large fixed per-call overhead (LIKELY: the intercept is about −27 ms).

Decode vs context (`R4/r4_ctx.py`, `generate_step`, 48 tokens after 8 warm-up; `ctx_8b.out`, `ctx_06b.out`):

| Context | 8B ITL (tok/s) | 8B TTFT | 0.6B ITL (tok/s) | 0.6B TTFT |
|---|---|---|---|---|
| 128 | 25.54 ms (39.2) | 0.655 s | 3.75 ms (267) | 0.058 s |
| 1024 | 27.36 ms (36.6) | 4.93 s | 4.29 ms (233) | 0.355 s |
| 2048 | 28.50 ms (35.1) | 9.72 s | 4.82 ms (208) | 0.766 s |
| 4096 | 31.57 ms (31.7) | 20.69 s | 6.31 ms (158) | 1.82 s |
| 8192 | – | – | 8.64 ms (116) | 4.87 s |

- For 8B, the extra KV bytes from 128→4096 (585 MB/token) cost +6.04 ms. That is an effective **97 GB/s on the
  attention KV stream** (53% of achievable).
- For 0.6B (GQA 2), the same slope is 189 GB/s.
- The 8B (GQA 4) attention path is about 2× off the bandwidth line (VERIFIED time). The likely mechanism
  (LIKELY, R3 §7.F and R3-kernels §c) is that `sdpa_vector` re-reads K/V per query head, and the GQA-deduplicating
  variant only starts at kL ≥ 8192.

### 7.B Bandwidth efficiency (VERIFIED arithmetic on our measurements)

- Bytes per decode token come from the safetensors headers (`R4/r4_bytes.py` → `r4_bytes.out`).
  - We sum packed weights, scales and biases of all linear layers, plus the LM head and the norms.
  - The 8B embedding table is excluded (one row gathered).
  - For 0.6B the tied embedding *is* the LM head (`as_linear`), so it is included once.
  - Both checkpoints are 4.500 bits per quantized weight, including bf16 scales and biases.

| | 0.6B | 8B |
|---|---|---|
| Linear layers | 247.7 MB | 3,907.0 MB |
| LM head | 87.5 MB (tied embed) | 350.1 MB |
| Norms | 0.13 MB | 0.62 MB |
| **Weights per token** | **335.4 MB** | **4,257.7 MB** |
| KV read per context position | 114.7 KB | 147.5 KB |

Achievable bandwidth: R3 measured a 2 GiB fp16 all-reduce at a median of **182.5 GB/s** (best 184.2) and copies
at 181.5 GB/s, on this machine through the lock (`$S/research/r3_out_bw_1.txt`, VERIFIED R3 data).

| Run | Bytes/token (incl. KV) | ITL | GB/s | % of 200 spec | % of 182.5 achievable |
|---|---|---|---|---|---|
| 8B `stream_generate` (A, median ITL) | 4,295 MB | 26.42 ms | 162.6 | 81.3% | 89.1% |
| 8B `generate_step` (E) | 4,281 MB | 25.79 ms | 166.0 | 83.0% | **91.0%** |
| 8B, qmv-only chain (E) | 4,258 MB | 23.93 ms | 178.0 | 89.0% | **97.5%** |
| 8B at 4K context | 4,866 MB | 31.57 ms | 154.1 | 77.1% | 84.4% |
| 0.6B `stream_generate` (A) | 365 MB | 3.74 ms | 97.5 | 48.8% | 53.4% |
| 0.6B `generate_step` (E) | 354 MB | 3.55 ms | 99.5 | 49.8% | **54.5%** |
| 0.6B, qmv-only chain (E) | 335 MB | 2.41 ms | 139.4 | 69.7% | 76.4% |

**Negative result (VERIFIED on M2 Pro):** for an 8B 4-bit dense model at short context, mlx-lm decode is at about
91% of the achievable bandwidth roofline. Its quantized GEMVs, run as a real dependent chain, stream at 97.5% of
achievable.

R3 independently found 94–97% for 8B- and 32B-shaped layer stacks (R3-mlx-core §7.F).

### 7.C Dispatches per decode token (VERIFIED, exact)

Method:
- `R4/r4count.m` → `libr4count.dylib` swizzles the concrete compute-encoder class (`AGXG14XFamilyComputeContext`),
  counting `dispatchThreadgroups:`, `dispatchThreads:`, `memoryBarrierWithScope:` and `endEncoding`.
- `R4/r4_census.py` evaluates exactly one decode step (forward + logsumexp + argmax, warm 128-token cache), and
  also counts primitives in `mx.export_to_dot`.
- Validation: 10 dependent adds gave 10 dispatches and 9 barriers.
- Command: `$S/venv/bin/python r4_census.py <model> 128` (counting run, no timing). Output: `C_census.out`,
  `census_*.json`, `census_*.dot`.

| | 0.6B (28 layers) | 8B (36 layers) |
|---|---|---|
| Graph primitive nodes | 829 | 1,061 |
| QuantizedMatmul | 197 | 253 |
| RMSNorm | 113 | 145 |
| RoPE | 56 | 72 |
| SliceUpdate (KV writes) | 56 | 72 |
| Add | 56 | 72 |
| ScaledDotProductAttention | 28 | 36 |
| Compiled SwiGLU | 28 | 36 |
| Gather 3, Quantize (dequant) 1, LogSumExp 1, Subtract 1, ArgReduce 1 | 7 | 7 |
| Views, no dispatch (Reshape, Transpose, Slice, Squeeze, ExpandDims, Broadcast) | 288 | 368 |
| **Exact `dispatchThreadgroups` / `dispatchThreads`** | 225 / 316 | 289 / 404 |
| **Total GPU dispatches** | **541** | **693** |
| Memory barriers | 362 | 457 |
| Compute encoders (one per command buffer) | 11 | 23 |
| Forward only, without logsumexp/subtract/argmax | 538 | 690 |

- Every kernel-launching primitive is exactly one dispatch. qmv and SDPA use `dispatchThreadgroups`; the rest use
  `dispatchThreads`.
- That confirms in-place KV writes and single-pass SDPA at this context.
- Barriers are fewer than dispatches because q/k/v, gate/up and rope q/k are independent.
- Extrapolation (VERIFIED by R3's 64-layer census): Qwen3-32B has about **1,225 dispatches per token**, of which
  449 are qmv.

### 7.D Speculative decoding: Qwen3-8B-4bit target, Qwen3-0.6B-4bit draft (VERIFIED, our measurement)

Setup (`R4/r4_spec.py`):
- prompts use the chat template with `enable_thinking=False`; greedy; EOS on; `max_tokens = 192`;
- warm-up of both paths, then 5 trials per configuration;
- tok/s = (n−1)/(t_last − t_first), wall clock;
- accept = fraction of emitted tokens with `from_draft = True`.

Command: `bash $S/benchlock.sh $S/venv/bin/python r4_spec.py chat|code|summary`. Output: `D_spec.out`,
`raw_D_*.json`.

| Prompt (prompt tokens) | k = 0 (plain) | k = 2 | k = 4 | k = 6 |
|---|---|---|---|---|
| Chat answer (31) | **38.98** | 25.12 (0.64×), acc 0.51 | 19.96 (0.51×), acc 0.62 | 14.43 (0.37×), acc 0.64 |
| Code generation (44) | **38.87** | 28.82 (0.74×), acc 0.59 | 26.39 (0.68×), acc 0.71 | 20.71 (0.53×), acc 0.76 |
| Summarize passage (522) | **37.71** | 25.59 (0.68×), acc 0.55 | 22.65 (0.60×), acc 0.68 | 15.90 (0.42×), acc 0.69 |

- Greedy outputs matched the non-speculative baseline in 9 of 12 configurations. The three that diverged:
  chat k = 6 at token 26, and summary k = 2 and k = 4 at token 99. The cause is the bf16 tie-break issue in §5.1.
- **Why it loses on M2 Pro.** We measured the cost of one target forward on M tokens (`r4_spec.py verify`,
  `D_verify.out`, median of 15, synchronous, including about 1.5 ms of build and encode):

| M (tokens verified) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|
| 8B forward, ms | 27.6 | 47.8 | 69.4 | 91.8 | 113.0 | 170.0 | 174.6 | 178.7 | 181.8 | 182.3 |
| × of M = 1 | 1.00 | 1.73 | 2.52 | 3.33 | 4.10 | 6.16 | 6.33 | 6.48 | 6.59 | 6.61 |

  - On gen-14 GPUs, MLX routes affine M = 2..5 to `qmv` with one threadgroup row per input row. Weights are
    re-read and re-dequantized per row; `use_qmv_wide` requires gen ≥ 15 for affine (mlx/mlx/backend/metal/quantized.cpp:541-544).
  - M ≥ 6 crosses `get_qmv_batch_limit` into `qmm_splitk`/`qmm` (quantized.cpp:89-140, 1889-1914), a flat
    "dead zone" at ~6.5×.
  - Verifying k+1 tokens therefore costs about (k+1)×. Speculation cannot pay off here, whatever the acceptance.
  - R3 reproduced the same curve per kernel (R3-mlx-core §7.C).
- **Not representative of M5 Max (LIKELY).** gen-17 uses `qmv_wide` for M ≥ 2. A contributor's M5 Max data on a
  70B 4-bit forward (mlx#3791, merged 2026-08-08, via R3-issues E4) shows M = 3 at +12%, M = 5 at +48%, M = 8 at
  +125%. Speculative decoding with a 0.6B draft went from about 18 to 24.5 tok/s (k = 3) there.
- So on M5 Max, mlx-lm's draft-model speculation is probably a 1.2–1.4× win for 32B. It still has two
  inefficiencies, a per-round full sync and sample-and-match acceptance. Not measured by us.

### 7.E Per-token time budget for decode (VERIFIED, our measurement; ablation)

Method (`R4/r4_budget.py`):
- A forward function replicates qwen3.py op for op. Validated: `max|model() − fwd()| = 0.0`.
- Switches remove op classes while keeping every quantized matmul, with the same weights, shapes and dependency
  structure. k/v and `up` outputs are kept alive as extra outputs.
- "Pipelined" = mlx-lm's own build-next / async_eval / `.item()` loop, median of 64 tokens after 8 warm-up.
  "Sync" = one step at a time, median of 20.

Command: `bash $S/benchlock.sh $S/venv/bin/python r4_budget.py <model> 128`. Output: `E_budget.out`,
`raw_E_*.json`.

Qwen3-8B-4bit, context 128 (`generate_step` reference: **25.79 ms**, 38.8 tok/s):

| Variant | Pipelined ITL | Δ vs full | Dispatches | Barriers | Sync: build / async_eval / wait (ms) |
|---|---|---|---|---|---|
| Full (= qwen3.py) | **25.90** | – | 693 | 457 | 0.65 / 13.6* / 13.0 |
| No logsumexp (argmax on logits) | 25.72 | −0.17 | 691 | 455 | |
| No RMSNorms (input/post/q/k) | 25.14 | −0.76 | 548 | 348 | |
| No RoPE | 25.61 | −0.29 | 621 | 421 | |
| No attention (rope + KV write + SDPA) | 24.70 | −1.20 | 513 | 348 | |
| No residual adds | 25.72 | −0.18 | 621 | 385 | |
| No SwiGLU | 25.55 | −0.35 | 657 | 422 | |
| **qmv only (all non-matmul removed)** | **23.93** | **−1.97 (7.6%)** | 258 | 132 | 0.65 / 14.2* / 11.2 |

\* The `async_eval` call blocks on MLX's back-pressure of 10 in-flight command buffers (23 per token here), so in
the sync measurement this is mostly GPU wait (LIKELY, R3 §1.2).

Qwen3-0.6B-4bit, context 128 (`generate_step` reference: **3.55 ms**, 281 tok/s):

| Variant | Pipelined ITL | Δ | Dispatches | Barriers | Sync: build / async_eval / wait (ms) |
|---|---|---|---|---|---|
| Full | **3.51** | – | 541 | 362 | 0.54 / 1.40 / 3.01 |
| No logsumexp | 3.41 | −0.10 | 539 | 360 | |
| No RMSNorms | 3.30 | −0.21 | 428 | 278 | |
| No RoPE | 3.41 | −0.10 | 485 | 342 | |
| No attention | 2.91 | −0.60 | 401 | 283 | |
| No residual adds | 3.55 | ≈0 | 485 | 305 | |
| No SwiGLU | 3.47 | −0.04 | 513 | 333 | |
| **qmv only** | **2.41** | **−1.11 (31%)** | 202 | 113 | 0.23 / 0.81 / 2.08 |

Budget, 8B (VERIFIED arithmetic):

| Component | ms/token | Share |
|---|---|---|
| Ideal weight + KV streaming at 182.5 GB/s | 23.46 | 91% |
| qmv kernels' shortfall vs ideal | +0.47 | 2% |
| All non-matmul ops: 435 dispatches, attention KV reads, barriers | +1.97 | 7.6% |
| CPU (Python build 0.65 ms + encode), hidden by pipelining | ~0 exposed | – |

- Removing 435 small dependent dispatches saved 1.97 ms, about **4.5 µs each including their work**. For 0.6B:
  339 dispatches, 1.11 ms, **3.3 µs each**.
- R3's tiny-op probe gives 2.9–3.2 µs per dependent op (`r3_out_dispatch_1.txt`).
- The per-class deltas do not add up (2.95 ms vs 1.97 ms total). Removing one class also removes barrier waits
  that overlap with other classes.

Budget, 0.6B: the ideal is 1.94 ms; qmv shortfall +0.47 ms; non-matmul +1.11 ms. The CPU side, 1.94 ms of build
plus encode per token, is only just hidden behind ~3.5 ms of GPU time.
- On a GPU about 3× faster (M5 Max), this model would become **CPU-bound** in mlx-lm (LIKELY, from these
  numbers).
- That is consistent with BaseRT's M5 Pro result of 1.33× over mlx-lm on Qwen3-0.6B decode (§8).

### 7.F Extra checks (VERIFIED, our measurement)

- **Server per-token pattern** (`R4/r4_server_pattern.py`, `server_pattern.out`, median of 3 × 96 tokens):

| Loop | 0.6B ITL | 8B ITL |
|---|---|---|
| `generate_step` | 3.458 ms | 25.592 ms |
| `BatchGenerator` (1 sequence) | 3.523 ms | 25.810 ms |
| `BatchGenerator` + `r.logprobs[r.token].item()` per token (as server.py:867) | 3.514 ms | 25.714 ms |

  - Hypothesis refuted: the server's per-token logprob `.item()` does not break pipelining.
  - `BatchGenerator` is within 1–2% of `generate_step` at batch 1.
- `stream_generate` vs `generate_step` on 8B: 26.42 vs 25.79 ms ITL, from different runs about 4 h apart. This is
  within run-to-run spread (the A trials themselves spanned 36.1–37.9 tok/s). The per-token Python work in
  `stream_generate` is microseconds and pipelined (LIKELY).

---

## 8. Published numbers

- **Apple ML Research, "Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU"** (2025-11-19,
  https://machinelearning.apple.com/research/exploring-llms-mlx-m5; VERIFIED via fetch):
  - Hardware: base M5 MacBook Pro (24 GB, 153 GB/s) vs M4 (120 GB/s).
  - Method: `mlx_lm.generate` with a **4096-token prompt and 128 generated tokens**.
  - Models: Qwen 1.7B bf16, Qwen 8B bf16 and 4-bit, Qwen 14B 4-bit, Qwen 30B-A3B 4-bit MoE, GPT-OSS-20B MXFP4.
  - TTFT speedups M5 vs M4: 3.57×, 3.62×, 3.97× (8B 4-bit), 4.06× (14B 4-bit), 3.52× (30B MoE), 3.33×
    (GPT-OSS).
  - Generation: 1.19–1.27× ("bounded by memory bandwidth"), roughly tracking the 1.28× bandwidth ratio.
  - TTFT is under 10 s for dense 14B 4-bit at 4K, i.e. ≥ 410 prompt tok/s on a 10-core M5. Memory: 8B 4-bit
    5.61 GB, 14B 4-bit 9.16 GB.
  - Requires macOS 26.2+.
  - Absolute TTFT and tok/s values are only in charts. We did not extract them.
- **Apple Newsroom, M5 Pro/Max MacBook Pro** (2026-03-03,
  https://www.apple.com/newsroom/2026/03/apple-introduces-macbook-pro-with-all-new-m5-pro-and-m5-max/; VERIFIED):
  - "up to 4x faster LLM prompt processing than M4 Pro and M4 Max";
  - M5 Pro: up to 307 GB/s, 64 GB;
  - M5 Max: up to **614 GB/s**, 128 GB, 40-core GPU;
  - the footnotes do not give the LLM test configuration.
- **BaseRT paper** (arXiv 2607.19438, 2026-07-21, Base Compute, Melbourne; vendor self-report, so treat as
  advocacy, but the tables are complete):
  - Setup: M5 Pro 48 GB, mlx-lm 0.31.3 / MLX 0.32.0, llama.cpp b9960, tg128 and pp128–2048, mean of 5.
  - Decode:
    - Qwen3-0.6B Q4: BaseRT 530.8, **mlx-lm 398.1**, llama.cpp 386.1;
    - Qwen3.6-27B Q4 (the largest dense model): BaseRT 18.1, **mlx-lm 18.4** (MLX faster), llama.cpp 15.4;
    - MoE 35B-A3B: about 1.06–1.09× over mlx-lm.
  - Prefill vs mlx-lm:
    - Qwen3.6-27B: pp128 440 vs 305 (1.44×), pp2048 **497 vs 486 (1.02×)**;
    - Qwen3-0.6B: pp128 11,873 vs 4,600 (2.6×), pp2048 20,105 vs 16,198 (1.24×).
  - Its own conclusion: "BaseRT and MLX both operate close to what the hardware's bandwidth allows" on decode.
  - Text saved at `R4/baseRT_2607.19438.txt`.
- **mlx-lm BENCHMARKS.md** (VERIFIED, file in repo): 64 GB M4 Max, mlx 0.29.2.dev, mlx-lm 0.28.2.
  - Qwen3-4B-Instruct-2507 4-bit: prompt (2048) 1,622 tok/s, generation (128) 134.5 tok/s, 3.35 GB.
  - Qwen3-30B-A3B 4-bit: 1,754 / 113.3 tok/s, 18.2 GB.
  - Derived (LIKELY): about 2.3 GB/token × 134.5 = 300 GB/s, roughly 55% of the M4 Max's 546 GB/s. That suggests
    the small-model overhead regime is also present on Max-class chips.
- **Community, M4 Max / M5 Max, 32B-class 4-bit (low quality, flagged):**
  - llmcheck.net "Qwen 3 32B on M4 Max: 22 tok/s … Q4_K_M via MLX" mixes a GGUF quant name with MLX. Unreliable.
  - x.com/RobbiewOnline: about 23 tok/s for "Qwen3.5-27B" on M5 Max. No method given.
  - A search snippet claims "dense Qwen 32B, 1K→128K context: 31.2 → 8.5 tok/s at 4-bit". Source and hardware
    unverified.
  - Better third-party anchors are in R3-issues.md:
    - a contributor's M5 Max (40-core) bandwidth measurement of 530 GB/s;
    - mlx#3791 M5 Max 70B data (M = 1 forward 75.9 ms, i.e. about 13 tok/s on 70B).
- **Our own extrapolation to Qwen3-32B 4-bit on M5 Max (LIKELY, arithmetic):**
  - 17.99 GB of weights per token (31.98 B linear + LM-head parameters at 4.5 bits per weight). KV adds 262 KB per
    context position.
  - Ceiling is 34.1 tok/s at the 614 GB/s spec, or about 29.5 tok/s at the 530 GB/s contributor measurement.
  - mlx-lm at ~91% efficiency gives an expected **≈ 27 tok/s at short context**. About 1.07 GB extra per token at
    4K context gives about −5…−10%.

---

## 9. GitHub issues and PRs: findings and maintainer positions (VERIFIED via `gh api`, raw in `R4/issues/`)

- **#1651** (2026-08-03, Vlor999): `quantized_kv_start` default 0 in the library vs 5000 in the CLI.
  - Measured on M4 Pro, Qwen3-0.6B-4bit, kv_bits = 8: quantizing from step 0 costs **6–22% decode throughput** and
    saves memory only past about 256 tokens (169 MB at 1024).
  - The companion #1566 attributes the cost to the unfused quantized attention: 3 kernels plus a materialized
    scores tensor vs fused SDPA. On a 5,120-token prompt the quantized cache was *faster* than fp16: +3.7%
    (0.5B), +18.3% (Llama-1B).
  - Maintainers: zcbenz closed the reporter's PR #1652: "the number of PRs is way beyond our capacity to review
    so I'm closing the non-essential ones".
  - michalk8 merged the equivalent #1819 (2026-09-09), so library and CLI now default to 5000.
- **#1450** (2026-07-02, dahai80, non-member): "Decode performance optimizations".
  - Proposes mixed-bit recipes, skip-logsumexp, a fused single-kernel sampler, an FP8 KV cache, `mx.compile` of
    the decode step, and Python dispatch reduction.
  - It claims "up to 162% faster" on M2 Ultra. That comparison is mxfp8 vs 2-bit recipes, so it measures bytes,
    not the decode optimizations.
  - Closed by zcbenz (2026-08-26) without comment.
  - Spin-off PRs #1466 (quant2 recipes) and #1479 (skip-logsumexp, "~5-8%") were closed for review capacity
    (zcbenz, 2026-08-21). #1497/#1498 (n-gram self-speculation, +17% on Qwen3.6-27B at S = 2) were closed with
    "At the moment we don't accept AI written architecture changes".
  - **Our data:** skip-logsumexp saves 0.17 ms/token on 8B (0.7%) and 0.10 ms on 0.6B (3%), not 5–8%. The
    sampler is already on the GPU and is a negligible share for greedy.
- **#1587** (open): QuantizedKVCache **higher** peak memory than fp16, about +30–73% at 8K–32K.
  - Contributors showed the cause is prefill, not steady state. The unfused quantized attention materializes
    `(chunk × context)` fp32 scores: 1.208 GB predicted vs 1.216 GB measured when varying the chunk size. Decode
    peak is lower with the quantized cache. Quantized decode is slower (−3…−7% on dense 32B; "roughly halved" for
    long-context serving on M5).
  - Implication: mlx-lm has no fused quantized-KV attention kernel.
- **#763** (open): long-context decode about 50% of llama.cpp with flash attention (M3 Ultra, MiniMax-M2.1 4-bit,
  146K: 5.95 vs 12.12 tok/s).
  - awni asked for reproduction commands. No fix.
  - Consistent with our measured 53%-efficient attention KV stream on 8B (§7.A).
- **#1435** (closed by zcbenz): a +51–77 ms TTFT offset. A commenter bisected it to mlx-lm 0.28.0's per-call
  detokenizer construction (`get_vocab`). The HEAD code caches the map (§2.5).
- **#1425** (open): a 6–8% decode regression from 0.31.0 to 0.31.3 on Qwen3.5-35B-A3B (M3 Ultra). No maintainer
  response.
- **#1470 / #1423 / #846** (open): speculative output diverges at temp 0. Contributors traced it to exact bf16
  logit ties under different reduction orders; documented in PR #1592.
- **#1132** (open): speculative decoding is −25…−45% on a 397B-A17B MoE target with a 9B draft (Mac Studio). It
  asks for a warning.
- **#851, #872, #1135, #1350** (open, no maintainer response): n-gram drafting, internal MTP, DFlash, and
  speculative prefill (PFlash) are requested. None is implemented at HEAD.
- **#1395** (open): prompt-cache reuse doubles peak memory. The mechanism is copy-on-first-write (§4.1).
- **#250** (open, 2025): the speculative `_step` is slower than plain `_step` at the same shape.
- Maintainer stance overall (VERIFIED from the closures above, LIKELY as policy):
  - mlx-lm is capacity-constrained on reviews and conservative about architectural changes to the generation
    loop.
  - Performance work happens mainly in MLX core kernels (e.g. qmv_wide, the NAX paths, mlx#3791, mlx#3961).
  - Loop-level optimizations from outsiders (fused sampler, compiled decode, n-gram speculation) have not been
    accepted in 2026.

---

## 10. Conclusions

### 10.1 Where mlx-lm spends time per token (8B 4-bit, M2 Pro, short context; VERIFIED)

- **91%** is the unavoidable weight and KV stream at the achievable bandwidth (23.46 ms of 25.79).
- **~2%** is the quantized GEMVs running below the achievable bandwidth (178.0 vs 182.5 GB/s). The small k/v
  projections are the worst (R3: 54–63%).
- **~7.6%** (1.97 ms) is everything that is not a matmul: 435 extra dependent dispatches.
  - By ablation: attention block 1.20 ms, RMSNorms 0.76, SwiGLU 0.35, RoPE 0.29, adds 0.18, logsumexp 0.17. These
    overlap, so they do not add up.
- **~0%** is CPU. The Python graph build (0.65 ms) and Metal encoding are hidden by `async_eval` pipelining.
  Sampling is on the GPU. The detokenizer is microseconds and overlapped.
- **Long context:** attention grows at about 1.5 ms per 1K tokens on 8B (97 GB/s effective, about 2× the ideal).
  At 4K it is 19% of the ITL, versus about 10% if it streamed at the line.
- **0.6B (the draft-model or small-model regime):** only 54.5% of the roofline. 31% of the ITL is non-matmul
  dispatch overhead, and the qmv chain reaches only 76% of achievable. The CPU path (1.94 ms) nearly equals the
  GPU path (3.5 ms).

### 10.2 Fraction of roofline

- 8B decode: **91% of achievable (83% of spec)** at short context; 84% at 4K.
- 0.6B decode: **54.5%**.
- 8B prefill: **58% of the FP32/FP16 ALU peak** on M2 Pro, which has no NAX. On M5, MLX uses NAX for prefill
  (R3-kernels §d), and BaseRT shows MLX within 2% of the best hand-written runtime at pp2048 on a dense 27B (but
  1.44× behind at pp128).

### 10.3 Where a whole-model compiler could beat mlx-lm (evidence-ranked)

1. **Decode on 32B at short context: a small, real win of 5–10% (LIKELY).**
   - Evidence: 7.6% of the 8B ITL is non-matmul dispatches (§7.E).
   - Qwen3-32B has 1,225 dispatches per token (776 non-qmv). On M5 Max the per-token GPU time shrinks about 3×
     while per-dispatch latency stays similar, so the overhead share should grow, not shrink.
   - BaseRT on M5 Pro gets 0.98× of mlx-lm on 27B decode, so even an expert hand-tuned runtime has not beaten
     it there.
   - Techniques:
     - fuse RMSNorm into the following GEMV prologue;
     - concatenate q/k/v and gate/up into single GEMVs with a SwiGLU epilogue;
     - fuse residual add plus norm;
     - fuse rope plus KV write, or q/k-norm plus rope plus KV write;
     - a static, pre-recorded dispatch list.
   - Upper bound on M2 Pro for 8B: the qmv-only ITL (23.93 ms) is **1.08× faster** than mlx-lm (25.79 ms).
2. **Long-context decode attention: a large win at ≥ 4K context (LIKELY).**
   - Evidence: 97 GB/s effective on the KV stream (§7.A), mlx-lm#763 (≈ 50% of llama.cpp at 146K), and R3's
     finding that `sdpa_vector` re-reads K/V per q-head below kL = 8192.
   - A GQA-aware split-K decode attention kernel could make attention cost about half for Qwen3-32B (8 KV heads,
     GQA 8).
   - At 16K context that is about a 10–20% ITL improvement (LIKELY, arithmetic).
   - The KV-quantized path is unfused (#1587, #1651). A fused int8/fp8 KV attention kernel is a further
     long-context win: half the KV bytes at full fusion. That needs a quality check, and the stated constraints
     say same quantization, so it is optional.
3. **Speculative verification kernels (M = 2..8) and the loop (EXPERIMENTAL on M5 Max).**
   - Evidence: on M2 Pro, verify costs (k+1)×, so speculation loses 26–63% (§7.D). On M5 Max, the mlx#3791 data
     shows +12% at M = 3, +48% at M = 5, +125% at M = 8, against an ideal near flat to M ≈ 8 (R3 §7.C, §7.G: a
     60-line custom kernel got M = 3 at 1.30× vs MLX's 1.92× on M2).
   - In the loop: rejection-sampling acceptance (Σ min(p, q)) instead of sample-and-match, cross-round
     pipelining instead of a hard sync per round, and prompt-lookup or n-gram drafting (none in mlx-lm).
   - This is the only route found to a **>1.2× decode** gain under the same-weights constraint (LIKELY 1.3–2×
     on code and summarization prompts with a good verifier). It must be measured on M5 Max.
4. **Small models (drafts, ≤ 4B): 1.3–1.8× (LIKELY).**
   - Evidence: 0.6B at 54.5% of the roofline; BaseRT +33% on 0.6B decode on M5 Pro; mlx-lm's own BENCHMARKS imply
     about 55% on 4B on M4 Max.
   - This matters for Sky mainly as the draft model's speed inside speculative decoding.
5. **Prefill/TTFT on M5 Max: modest (LIKELY).**
   - BaseRT vs mlx-lm on dense 27B: 1.44× at pp128, falling to 1.02× at pp2048.
   - At the long prompts that dominate TTFT, mlx-lm is near the NAX ceiling. Short-prompt TTFT has the headroom.
6. **Peak RAM: small (LIKELY).**
   - mlx-lm peak = weights + KV + about 0.3–0.7 GB of transient activations (8B at a 2K prompt). A static memory
     plan can trim the transient part and avoid copy-on-write duplicates on prompt-cache reuse (§4.1), but weights
     dominate.
7. **Energy per token: unknown.** Not measured (needs powermetrics/sudo).

**Negative results to carry into the plan:**
- mlx-lm decode on a mid-size dense model is about 91% of the roofline. "Beat mlx-lm on decode tok/s" with the
  same 4-bit weights is capped at about 1.1× without algorithmic changes.
- On M2-class GPUs, speculative decoding in mlx-lm is strictly harmful (0.37–0.74×).
- The server's per-token host syncs and `BatchGenerator` do not cost throughput at batch 1.
- Skip-logsumexp and a "fused sampler" are worth ≤ 1% for greedy 8B.

---

## 11. Open questions

1. M5 Max measurements: everything here is M2 Pro, which has no NAX and no `qmv_wide` for affine. The top
   priorities:
   - the verify-cost curve (M = 1..8) for Qwen3-32B;
   - mlx-lm speculative decoding with a 0.6B or 1.7B draft;
   - decode at 4K/16K/32K;
   - TTFT at 128/2K/8K.
2. Per-dispatch GPU latency on M5 Max. It decides whether the non-matmul share there is 5% or 15%.
3. Energy per token, mlx-lm vs alternatives (powermetrics needs sudo).
4. Whether the 8B GQA attention inefficiency is entirely the K/V re-read per q-head (R3's hypothesis), or
   partly the 2-pass threshold on `s`/`d` devices at kL ≥ 1024.
5. How much of mlx-lm's 0.6B gap is CPU-bound on M5-class GPUs. Our CPU build plus encode is 1.94 ms per token
   on M2 Pro.

---

## 12. Reproduction: commands and raw files (all in `$S/research/R4/`)

| Item | Command | Output |
|---|---|---|
| Bytes per token | `$S/venv/bin/python r4_bytes.py` | `r4_bytes.out`, `r4_bytes.json` |
| A decode | `bash $S/benchlock.sh $S/venv/bin/python r4_gen.py Qwen3-{0.6B,8B}-4bit decode` | `A_06b.out`, `A_8b.out`, `raw_A_*decode*.json` |
| A prefill/TTFT | `bash $S/benchlock.sh $S/venv/bin/python r4_gen.py Qwen3-0.6B-4bit prefill 128 512 2048` (8B: `128 512 1024 2048`) | same, `raw_A_*prefill*.json` |
| Context sweep | `bash $S/benchlock.sh $S/venv/bin/python r4_ctx.py Qwen3-8B-4bit 128 1024 2048 4096` (0.6B: plus 8192) | `ctx_8b.out`, `ctx_06b.out`, `raw_ctx_*.json` |
| C dispatch census | `clang -dynamiclib -fobjc-arc -O2 -framework Metal -framework Foundation -o libr4count.dylib r4count.m`; `$S/venv/bin/python r4_census.py <model> 128` | `C_census.out`, `census_*.{json,dot}` |
| D speculative | `bash $S/benchlock.sh $S/venv/bin/python r4_spec.py {chat,code,summary,verify}` | `D_spec.out`, `D_verify.out`, `raw_D_*.json` |
| E budget | `bash $S/benchlock.sh $S/venv/bin/python r4_budget.py <model> 128` | `E_budget.out`, `raw_E_*.json` |
| Server pattern | `bash $S/benchlock.sh $S/venv/bin/python r4_server_pattern.py <model>` | `server_pattern.out`, `raw_server_pattern_*.json` |
| Issues | `gh api repos/ml-explore/mlx-lm/issues/N[/comments]` | `issues/i*.txt` |
| BaseRT paper text | WebFetch + pypdf | `baseRT_2607.19438.txt` |

---

## Architect brief

The full brief is in R4's hand-off message, mirrored here in condensed form.

- **Headline:** on M2 Pro, mlx-lm decodes Qwen3-8B-4bit at **91% of achievable bandwidth** (38.8 tok/s;
  25.79 ms ITL vs 23.46 ms ideal).
  - The remaining 7.6% is 435 small non-matmul dispatches; the GEMV chain alone is 97.5% efficient.
  - Fusion alone can win at most about 1.08–1.10× on short-context decode with the same weights.
- **Bigger levers:**
  1. long-context attention (KV streamed at ~53% efficiency);
  2. small-M verification kernels plus a better speculative loop. mlx-lm's speculation is 0.37–0.74× on M2 Pro
     because verify costs about (k+1)×;
  3. small and draft models (0.6B at 54.5% of the roofline, 31% dispatch overhead, nearly CPU-bound).
- **Prefill:** compute-bound, 58% of the ALU peak on M2 Pro. On M5, MLX already uses NAX (BaseRT: 1.02× at
  pp2048, 1.44× at pp128 on 27B).
- Everything must be re-measured on M5 Max (gen 17: `qmv_wide`, NAX, 614 GB/s).
