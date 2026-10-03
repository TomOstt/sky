# Part XVIII — TTFT optimization

Time to first token (TTFT) is the latency a user feels before anything happens. This part splits it into its components, says what MLX/mlx-lm and llama.cpp do for each, specifies Sky's design for each, and sets a target per component. It ends with the honest TTFT targets of the spine.

Prefill kernels are in [Part XVII](part17-prefill.md), the runtime in [Part XIII](part13-llm-runtime.md), memory and residency in [Part XIX](part19-memory.md), and the measurement protocol in [Part XXIII](part23-benchmarks.md).

Labels: `[VERIFIED: source]`, `[LIKELY: reason]`, `[EXPERIMENTAL: what decides]`. Physics numbers come from [`roofline.py`](roofline.py) (`python3 roofline.py --model qwen3-32b --hw m5-max-40`): 614 GB/s spec, 534 GB/s planning bandwidth, 65 TFLOPS fp16 on the neural accelerators, all `[LIKELY]` until measured on the M5 Max ([Part IV](part04-hardware-limits.md) §2).

## 1. Summary

- **Warm TTFT is prefill.** A 512-token Qwen3-32B prompt needs 0.50 s at 100% of the fp16 peak; MLX's published pp512 of 731 tok/s gives about 0.70 s. Sky wins warm TTFT by the prefill ratio, plus one structural saving: the first token comes out of the last prefill chunk, while mlx-lm runs the last prompt token as a separate decode step, which costs one more full weight pass, about 34–40 ms (R4 §2.1).
- **Cold TTFT is everything else**: process start, imports, model resolution, pipeline compilation, and making 18.4 GB of weights GPU-resident. mlx-lm copies the weights with `pread` into new buffers (R3 §4); llama.cpp compiles its shaders for 4.8 s on the first run after installation (R5 §1.6). Sky maps an aligned weight file with no copy, reuses compiled pipelines through Metal's OS shader cache (or a binary archive in an explicit `.skybin`), and overlaps the rest.
- **One cost is unknown, and it decides the cold target:** how long the GPU takes to make 18.4 GB of page-cache pages resident. An MLX contributor reports "~1–2 s" of mapping on an M5 Max (R3-issues G3). Sky overlaps it with prefill by making layer groups resident in order.
- **Idle gaps cost TTFT.** macOS un-wires GPU memory after about a second of idle; llama.cpp measured 16× slower first tokens before it added a residency heartbeat (R5 §9.5). Even with residency held, mlx-lm's TTFT rose 8.8% after an 8 s idle (R9 §6.4). Sky keeps residency during a configurable keep-alive and states the idle gap in every measurement.
- **A prefix cache removes prompt tokens from prefill.** It is the only way to an honest sub-second TTFT on long conversations. With batch- and chunk-invariant kernels (D6) its results are bit-identical to recomputation for prefixes written by prefill.

## 2. Definitions

From R9 §7.5 and [Part XXIII](part23-benchmarks.md) §6:

| Term | Start | End | Conditions |
|---|---|---|---|
| **TTFT (warm)** | prompt token ids handed to the engine | id of output token 0 available to the harness | process warm, model loaded, fixed 1.0 s idle gap before submission |
| **TTFT_text** | raw messages | same | adds template rendering and tokenization |
| **TTFT, cold (S5)** | `exec` of the engine process | same | (a) weights in the page cache; (b) cold page cache via a fresh APFS clone (`cp -c` gives a 0%-resident copy at no disk cost; R9 §6.5); (c) first run after install, including Sky's relayout-cache build |
| **TTFT, repeated prefix (S10)** | second request sharing a cached prefix | same | prefix caches enabled on both sides, cache state recorded |

Prefill tok/s is `n_prompt / TTFT`, headline only for n ≥ 2048 (R9 §7.5). mlx-lm's own `prompt_tps` is a TTFT-derived rate that excludes tokenization and the template (R4 §2.6).

## 3. Where the time goes

### 3.1 The physics floor (warm, M5 Max 40-core, Qwen3-32B)

| Prompt | Prefill at 100% of 65 TFLOPS | At 70% | MLX-like (731 tok/s) | One extra weight pass at 534 GB/s |
|---:|---:|---:|---:|---:|
| 128 | 123 ms | 176 ms | 175 ms | 34 ms |
| 512 | 496 ms | 708 ms | 700 ms | 34 ms |
| 1,024 | 1.00 s | 1.43 s | 1.40 s | 35 ms |
| 4,096 | 4.20 s | 6.01 s | 5.60 s | 36 ms |
| 32,768 | 48.8 s | 69.7 s | — | 50 ms |

Prefill columns: `roofline.py`, `prefill_ceiling_tps`. The MLX-like column applies the published pp512 rate (731 tok/s, R5, `[LIKELY: one community run]`) to all lengths, which overstates MLX at long prompts where attention grows. The last column is one decode-shaped forward pass (17.99 GB of weights plus the KV of the context) at the planning bandwidth.

The owner's "900 → 420 ms": 900 ms at 800 tok/s is a prompt of about 690 tokens (R8 §10). That prompt needs 670 ms even at 100% of the fp16 peak, so 420 ms is impossible from a cold prompt at fp16 and possible only when a cached prefix removes most of it ([Part IV](part04-hardware-limits.md) §1, §7).

### 3.2 Component by component

| # | Component | mlx-lm / MLX today | llama.cpp | Sky design (section) |
|---|---|---|---|---|
| 1 | Process start | Python interpreter, imports of `mlx`, `mlx_lm` and transformers' tokenizer stack | native binary | native Rust binary; Metal device created on a worker thread (§4.1) |
| 2 | Hub resolution | `snapshot_download`, which resolves the revision over the network unless offline `[LIKELY: R9 §5.4]` | `-hf` always resolves `main` over the network (R9 §5.4) | offline first: zero network calls for a complete snapshot ([Part XIII](part13-llm-runtime.md) §3.3) |
| 3 | Weights | `pread` of every byte into freshly allocated Metal buffers, 32 MiB batches on `clamp(cores/2, 4, 16)` threads (`mlx/io/load.cpp:340-446`, R3 §4) | `mmap` + `newBufferWithBytesNoCopy`; Qwen3-8B Q4_0 load 3,626 ms cold-ish, 422 ms warm (R5 §1.5, §8 D) | first run: relayout build or an aligned in-memory copy; later runs: `mmap` + no-copy + per-layer-group residency (§4.3) |
| 4 | Tokenizer and template | Python tokenizer load; the detokenizer map was rebuilt per call in mlx-lm 0.28.0–0.31.3 (+51–77 ms), fixed at HEAD (R4 §2.5) | C++ vocabulary from GGUF metadata | Rust `tokenizers` on a background thread; tables built once (§4.4) |
| 5 | Pipelines | precompiled `mlx.metallib` (190 MB, mmapped by Metal; R3 §5) plus JIT kernels for `mx.compile`, gathers and custom kernels (R3 §3) | 34 libraries compiled from embedded source at start: **4.841 s** the first time, **14–48 ms** later via the OS shader cache (R5 §1.6) | plan cache (data) → MSL regenerated deterministically → OS shader cache; an explicit `.skybin` may carry a binary archive (§4.5) |
| 6 | Prefill | chunks of 2048 (`DEFAULT_PREFILL_STEP_SIZE`), each ending with `mx.eval` of the cache and `mx.clear_cache()` (`generate.py:428-454`, R4 §2.1) | `n_batch` 2048, `n_ubatch` 512 (R5 §8 A) | chunk programs pre-committed back to back, no host sync between chunks ([Part XVII](part17-prefill.md)) |
| 7 | First token | the last prompt token always runs through the decode-style `_step`: a 2048-token prompt costs a 2047-token forward **plus a 1-token forward** (`generate.py:456`, R4 §2.1) | the last token is in the last ubatch; logits copied to the host (608 KB for Qwen3) and sampled on the CPU by default (R5 §6) | the last chunk computes the last row's logits and samples on the GPU; no extra pass (§4.6) |
| 8 | Host sync | `mx.eval(y)`: a blocking wait on an `MTLSharedEvent`; a synchronous round trip costs 149–158 µs on the M2 Pro (R3 §7.E) | `waitUntilCompleted` before sampling (R5 §6) | the streamer thread waits on a shared event; 84.8 µs from commit to CPU on the M2 Pro (R6 B3) |

## 4. Design, component by component

### 4.1 Process start

- `sky` is one native executable. Sky's AOT executables start about 4 ms after an empty process (spec, "Size and start-up"; R1 §5.6). The LLM path adds Metal device and queue creation, which runs on a worker thread while the main thread resolves the model.
- Target: ≤ 20 ms from `exec` to the point where resolution starts `[EXPERIMENTAL: measured by S5 with phase timestamps]`.

### 4.2 Hub resolution

- A complete snapshot at a known commit costs only `stat` calls on the snapshot files: zero network ([Part XIII](part13-llm-runtime.md) §3.3).
- A first-ever run that must download is outside every TTFT definition; `sky pull` exists so downloads happen ahead.
- Target: ≤ 5 ms offline.

### 4.3 Weights: the relayout cache vs mapping

**Steady state (every run after the first).** Sky maps `~/Library/Caches/sky/weights/<key>.skyweights` ([Part XII](part12-quantization.md) §5) with `mmap(PROT_READ, MAP_SHARED)` and wraps each region with `newBufferWithBytesNoCopy`. No byte is copied. What remains is making the pages GPU-resident:
- Metal wires the entire no-copy buffer on its first use by a kernel, and the wired ceiling counts mapped buffers (R3-issues G3, MLX contributor data).
- On an M5 Max, the same contributor reports mapped views running "at resident speed" after a "~1–2 s reversible mapping instead of a weights read" `[LIKELY: one contributor; model size not stated in our notes]`.

**Progressive residency.** The `.skyweights` file stores the embedding, then layer groups, then the head, each as its own page-aligned region ([Part XIII](part13-llm-runtime.md) §6.2), each region in its own residency set (≤ 5% of the working set each, the policy MLX adopted after macOS 27 began deciding residency per set; mlx#4211, R3-issues G4). Sky commits the sets and calls `requestResidency()` in layer order at load, and the prefill program's first chunk needs layer group 0 first. If residency proceeds layer by layer, prefill of early layers overlaps with wiring of later ones `[EXPERIMENTAL: M9 in Part XXIII measures time-to-resident per GB and whether requestResidency on later sets proceeds while kernels use earlier ones]`.

**The first run.** No `.skyweights` exists yet. Three options:

| Option | First-run TTFT | Later runs | Disk | Memory during the first run |
|---|---|---|---|---|
| A. Build the cache, then map it | one full read + one full write of 18.4 GB before the first token | mapped | +18.4 GB | page cache only |
| B. Load through an aligned in-memory copy (as MLX does, plus the repack), serve, and write the cache in the background | ≈ MLX's load | mapped | +18.4 GB | 18.4 GB anonymous, as MLX |
| C. Build in `sky pull` / `sky compile`; `sky run` uses B if the cache is missing | ≈ MLX's load | mapped | +18.4 GB | as B |

Proposal: **C**, with B as `sky run`'s fallback. B's background write uses `fcntl(F_NOCACHE)` on machines with less than about 2.5× the weight size in RAM, so the written pages do not double memory use; on larger machines it lets the page cache keep them, so the next run starts warm `[EXPERIMENTAL: M9 decides; Part VI §12 names this measurement]`. The relayout build time is always reported in S5 (c) (D9; R9 §9 Q6).

GGUF files skip all of this: they are mapped directly (R5 §1.5).

### 4.4 Tokenizer and template

- `tokenizer.json` (11.4 MB for Qwen3; [Part XIII](part13-llm-runtime.md) §7.2) is parsed on a background thread started at `exec`, in parallel with §4.3 and §4.5. Its cost is hidden unless it exceeds both `[EXPERIMENTAL: S5 phase timestamps]`.
- The id→bytes table, the special-token table and the stop-set bitmap are built once per process, at load, never per request. A `.skybin` may store them; the plan cache records only the hashes of the text files they come from.
- Rendering and tokenizing a 512-token prompt is CPU work of a few milliseconds or less `[EXPERIMENTAL: measured as TTFT_text − TTFT]`; target ≤ 2 ms.
- Lesson from mlx-lm#1435: a per-request rebuild of a vocabulary-sized table added 51–77 ms to every TTFT for a year before a user bisected it (R4 §2.5, §9). A test asserts that a second request builds no vocabulary-sized structure (§9, XVIII-T4).

### 4.5 Pipelines

Caches are data-only (spec non-negotiable 3; D19). The pipeline path therefore has three cases:

| Case | Where compiled code comes from | Expected cost |
|---|---|---|
| First run of a model on this machine (no plan cache) | Sky compiles the plan, generates MSL and compiles it with `makeLibrary(source:)` | seconds: llama.cpp's 34 libraries take **4.841 s** on first install (R5 §1.6); pipelines that use Metal 4 tensor operations take about **1–2 s** each to compile cold (llama.cpp PR #16634 comment; R6 A3) `[LIKELY]` |
| Later runs (`sky run`, plan cache hit) | the cached plan is data; MSL is regenerated from it deterministically, byte-identical to the first run, and Metal's OS shader cache returns the compiled pipelines | MSL generation plus cache lookups: llama.cpp's later starts take **14–48 ms** for its whole library set this way (R5 §1.6) |
| An explicit `.skybin` from `sky compile --archive` | a Metal binary archive harvested on the compiling machine (`MTL4PipelineDataSetSerializer` → `serializeAsArchiveAndFlush(url:)`; loaded with `MTL4Archive.makeComputePipelineState`); used only if `K_os` and `K_target` match ([Part XIII](part13-llm-runtime.md) §13) | archive load; also covers the first run after an OS update has cleared the OS shader cache `[LIKELY]` |

Rules:
- Sky never writes a binary archive implicitly; archives exist only inside `.skybin` build artifacts (D19).
- MSL generation must be deterministic so that the OS shader cache hits: the same plan produces byte-identical source (test XIII-T16).
- On a cache miss, pipelines compile on worker threads in the order they are needed: the first prefill chunk program, then decode.
- Targets `[EXPERIMENTAL: M10 in Part XXIII]`: MSL regeneration plus pipeline creation ≤ 50 ms on a plan-cache hit with a warm OS shader cache; the first-run compile time is reported separately in S5 (c).

### 4.6 Prefill and the first token

- **Chunks are submitted back to back.** mlx-lm ends every chunk with a full CPU↔GPU synchronization and a buffer-cache clear (R4 §2.1). Sky pre-commits all chunk programs; each reads its token ids from a buffer filled before the first commit. On the M2 Pro, a CPU wait between command buffers leaves the GPU idle ≈ 105 µs (R6 B3); for 32B prefill that is noise, but there is no reason to pay it.
- **The last chunk produces token 1.** The LM head runs on the last row only (0.6 MB of fp32 logits for Qwen3), followed by the GPU sampler, which writes token 1 and initializes `StepState` ([Part XIII](part13-llm-runtime.md) §11). mlx-lm instead feeds the last prompt token through a separate 1-token step (R4 §2.1). That step streams every weight once: ≈ 34 ms on the M5 Max at the planning bandwidth (§3.1). At a 128-token prompt this is about 20% of MLX's TTFT; at 512 tokens about 5%.
- **Exactness.** The first token's logits come from the prefill kernel, not the decode kernel. This is consistent with D6 because the guarantee that matters (speculative output = plain output; cached prefix = recomputed prefix) compares prefill with prefill. Chunk invariance makes the result independent of where the chunk boundaries fall.
- **Decode follows immediately.** Decode steps are committed right after the last chunk; they read `StepState`, which the prefill program wrote. No host round trip separates prefill from decode.
- Kernel-level prefill design and its targets (in the spine, ≥ 1.0× mlx-lm on the M5, and ≥ 1.25× mlx-lm and ≥ 1.0× llama.cpp on M1–M4) are [Part XVII](part17-prefill.md)'s. [Part IV](part04-hardware-limits.md) §5's 1.15–1.3× on the M5 Max is a best-case estimate of the prefill gain, not a target. The warm-TTFT target is ≥ 1.15× (§10).

### 4.7 Host synchronization

The first token must reach the host. The streamer thread waits on the per-step `MTLSharedEvent`; on the M2 Pro a Metal 4 `signalEvent` is visible to the CPU 84.8 µs after commit (R6 B3). Target ≤ 0.2 ms from GPU completion to the token being printed or sent.

## 5. Cold vs warm, idle gaps and residency

### 5.1 Two different effects

| Effect | Evidence | Size |
|---|---|---|
| **Un-wiring after idle.** macOS began un-wiring GPU memory after about 1 s idle | llama.cpp issue #10119: M2 Ultra, 8B, 26.34 ms/token became 429.08 ms after a 1.4 s pause; fixed by residency sets and a heartbeat (#11427 → #11648 → #17766 → #24074, 2025-01 → 2026-06) `[VERIFIED: R5 §9.5]` | **16×** on the first tokens |
| **Idle-gap ramp with residency held.** mlx-lm, warm process, Qwen3-0.6B-4bit, 1,097-token prompt, M2 Pro, 6 runs per gap, randomized | 386.5 ms back to back; 408.7 ms after 0.25 s; 407.9 ms after 1 s; 414.1 ms after 3 s; 420.6 ms after 8 s `[VERIFIED: R9 §6.4; R9-work/power/ttft_gap_out.txt]` | **+8.8%** at 8 s; +5.7% already at 0.25 s |

The second effect appears at 0.25 s, before un-wiring would start, so it is LIKELY a GPU power-state or clock ramp. It is not isolated, though: mlx-lm restores the previous wired limit at the end of every `stream_generate` call (`generate.py:258-263`; R4 §6), so re-wiring may contribute too `[LIKELY]`. Scenario S8 of [Part XXIII](part23-benchmarks.md) separates the two by comparing runs with Sky's residency held and released at gaps of 0–30 s.

### 5.2 Sky's residency policy

- **Weights and persistent buffers** are in residency sets, each at most 5% of `recommendedMaxWorkingSetSize`, at most 32 per queue (Feature Set Tables; R6 A4). Layer groups are sized to satisfy both ([Part XIX](part19-memory.md) §7).
- **Heartbeat.** While a session is "hot", a background thread calls `requestResidency()` on every set. llama.cpp uses 5 ms, after starting at 500 ms, and keeps it alive 180 s after the last graph (`ggml-metal-device.m:925-1031`; R5 §1.4). Sky starts with llama.cpp's values and tunes the period: the heartbeat costs CPU wake-ups, which matter for the energy target `[EXPERIMENTAL: sweep 5 / 50 / 500 ms against first-token latency after 1–60 s gaps and against idle power]`.
- **Keep-alive.** `sky serve --keep-alive 180s` (default). After the keep-alive, Sky calls `endResidency()` so the memory can serve other applications; the next request pays re-wiring and says so in its `timings`.
- **Pre-warm on arrival.** When a request arrives, before tokenization, Sky submits a tiny dispatch so any clock ramp starts while the CPU renders the template `[EXPERIMENTAL: S8 decides whether it removes part of the 5.7–8.8%]`.
- **Risk carried from MLX.** Wiring tens of GB has triggered kernel panics on M1–M5 (`completeMemory() prepare count underflow`, mlx#3186, FB22091885), which a user avoided only by not wiring at all (R3-issues G5). Sky's residency code therefore has a kill switch (`SKY_NO_RESIDENCY=1`) and logs every set operation in debug builds.

### 5.3 What cold TTFT compares

- **S5 (a), page cache warm, AOT plan present** (an explicit `.skybin`, or a plan cache with a warm OS shader cache): the spine's "cold ≥ 3× better" target. Sky pays process start, pipeline creation (§4.5), residency, and prefill; mlx-lm pays interpreter start and imports, a full `pread` copy of 18.4 GB, tokenizer load and prefill. The copy alone is a full pass over 18.4 GB through the CPU, and the imports are paid only by mlx-lm. Whether the ratio reaches 3× depends on the residency cost in §4.3, which applies to both engines. `[EXPERIMENTAL: S5 (a)]`
- **S5 (b), page cache cold:** every engine reads 18.4 GB from SSD. Seconds are unavoidable, and the report never compares a warm Sky with a cold competitor ([Part IV](part04-hardware-limits.md) §7).
- **S5 (c), first run after install:** includes Sky's relayout build (option A) or MLX-like load (B/C) and pipeline compiles. Reported, not a headline.

## 6. Prompt and prefix caching

### 6.1 Why it is the largest legitimate lever

TTFT then scales with the new tokens only (R8 §5.6, §8). For chat and agent loops, where every turn re-sends the whole conversation, this is the difference between seconds and tens of milliseconds.

The floor of a cached turn on the M5 Max, Qwen3-32B, with 4K cached tokens:
- **n new tokens ≤ about 8 (shader ALUs) to 33 (neural accelerators)**: the pass is bandwidth-bound (`roofline.py`, `verify_free_tokens`; [Part IV](part04-hardware-limits.md) §8), so TTFT ≈ one weight pass plus the cached KV: ≈ 36 ms at 534 GB/s.
- **n = 64**: ≈ 64 × (62.4 + 8.6) GFLOP ≈ 4.5 TFLOP, about 70 ms at 100% of 65 TFLOPS and 100 ms at 70% (arithmetic from R8 §1.4 and R9 §3.3: 2,097,152 FLOP of attention per token per context position).

### 6.2 Design

Requirements on the KV layout, which [Part XIV](part14-kv-cache.md) owns:
- KV is stored in fixed blocks of B positions, with B a multiple of the decode attention's fixed split size. SGLang had to align chunk boundaries to its split-KV size to keep its radix cache deterministic (R8 §3.3).
- Blocks are reference-counted, so reuse never copies. Only the last, partially filled block of a shared prefix is copied on write.

Sky's prefix cache (`sky-llm/src/prompt_cache.rs`):
- a radix tree keyed by token-id blocks, mapping each prefix to KV blocks (and, for hybrid models, to a DeltaNet state checkpoint at the block boundary);
- a byte budget carved from the memory plan's persistent arena ([Part XIX](part19-memory.md) §5), with LRU eviction;
- optional persistence: `sky run --prompt-cache <file>` and `sky serve --prompt-cache-dir` write blocks to disk and map them on load. For Qwen3-32B that is 0.26 MB per token at fp16, so a 4K-token system prompt is 1.07 GB (R8 §5.6); reading it at SSD speed is LIKELY much faster than the ≈ 6 s needed to recompute it at 70% of peak (§3.1) `[EXPERIMENTAL: M9]`;
- a cache key for persisted blocks: model key, normalized config hash (RoPE parameters included), KV format, numerics mode, kernel plan and tuning hash, and block size. A block computed by a different Sky build or tuning entry is not bit-identical and is rejected.

### 6.3 Exactness

- **Prefixes written by prefill** (system prompts, documents, tool outputs) are bit-identical to recomputation, because prefill is chunk-invariant (D6; for hybrid Gated-DeltaNet layers only with the recurrent-order prefill, since the labeled chunked-scan mode waives chunk invariance, VII-M7): `sky test` asserts equal logits with and without the cache for every prompt in the suite.
- **Prefixes written by decode** (earlier assistant turns) hold the KV the conversation actually produced. A cold re-prefill of the same text computes the same K/V with prefill kernels, whose reduction order differs from decode's, so the two can differ in the last bits. Sky does not claim cache-hit = cache-miss for these blocks unless the prefill K/V projection is restricted to decode's reduction order `[EXPERIMENTAL: measure that restriction's prefill cost; until then `sky serve --deterministic` re-prefills decode-written blocks]`.

### 6.4 The competitors' caches

| Engine | Mechanism | Known gaps |
|---|---|---|
| mlx-lm library (`stream_generate`) | none built in; users hand-roll prefix matching with `trim_prompt_cache` | mlx-lm#1194 (R4 issues) |
| `mlx_lm.server` | `LRUPromptCache` keyed by a token trie; exact, longer-trimmable and shorter-prefix matches; eviction by count and bytes (`cache.py:1599-1741`) | each hit deep-copies the cache, and the first write then copies the whole conversation's KV: 2× KV transiently, which killed a 16 GB server at about 12K tokens (#1395; mechanism in R4 §4.1); per-request `chat_template_kwargs` bypass the cache (#1803); streamed requests reported not to store entries (#1802, not reproduced by a second user) |
| `mlx_lm.generate --prompt-cache-file` | saves a cache to safetensors; loads it and sends only the suffix (R4 §4.4) | single prefix, manual |
| llama-server | `cache_prompt: true` by default; slot save/restore; `--cache-reuse` (R9 §6.1, R8 §5.6) | its README warns logits "are not guaranteed to be bit-for-bit identical for different batch sizes" |

One user report shows how much the trimming policy matters: the same 7K-token conversation had TTFT 0.406 s on llama.cpp (21 tokens decoded after a cache hit) and 3.184 s on MLX after trimming thousands of cached tokens (mlx-lm#480, LM Studio logs) `[LIKELY: a single report]`.

## 7. Chunked prefill

- At batch 1, chunking does not reduce TTFT. It bounds activation memory and enables interleaving with other requests, which year 1 does not do (R8 §8).
- Chunk size matters in three other ways:
  - the neural-accelerator GEMM wants large M; tiles of at least 64 × 64 (R6 A3, [Part XVII](part17-prefill.md));
  - the activation arena grows linearly with the chunk ([Part XIX](part19-memory.md) §9);
  - chunk boundaries must be multiples of the KV block size so a prefix cache can split anywhere (§6.2).
- Results do not depend on the chunk size (D6), so the autotuner may choose it per prompt length (D11 level (c)).
- mlx-lm uses 2048 for `generate_step` and 512 for speculative generation (R4 §2.1).

## 8. What the plan cache and the `.skybin` remove from TTFT

| Work | First `sky run` (nothing cached) | Later `sky run` (plan cache, data only) | Explicit `.skybin` |
|---|---|---|---|
| Config normalization, builder, TIR passes, scheduling, memory planning | full compile pipeline ([Part VI](part06-architecture.md) §5, steps 2–12) | none: the plan is loaded and validated | none |
| Autotuning | `-O2` DB lookup; `-O3` search (minutes) | none: entries recorded in the plan | none |
| MSL | generated | regenerated deterministically from the plan | stored |
| Compiled pipelines | compiled (seconds, §4.5) | Metal's OS shader cache | binary archive if present and matching; else OS shader cache or compile |
| Tokenizer tables | built | rebuilt from the tokenizer file (data) | stored |

`sky compile` exists for distribution, for paying `-O3` once, and for a binary archive that survives an OS shader-cache flush. The spine's cold target is stated "with AOT `.skybin`"; S5 (a) reports both the `.skybin` case and the plan-cache case.

## 9. Per-component targets

Qwen3-32B MLX 4-bit, M5 Max 40-core, 512-token prompt, page cache warm, an AOT plan (`.skybin`, or plan cache with a warm OS shader cache) and `.skyweights` present. All targets are `[EXPERIMENTAL]`, checked by phase timestamps in `sky run --trace-startup` and S5/S2/S10 of [Part XXIII](part23-benchmarks.md).

| # | Component | Cold target (S5 a) | Warm target (S2) | Overlapped with |
|---|---|---|---|---|
| 1 | `exec` → resolution starts | ≤ 20 ms | — | — |
| 2 | Hub resolution | ≤ 5 ms | — | 1 |
| 3 | Plan (cache or `.skybin`) open, validation and key check | ≤ 10 ms | — | — |
| 4 | MSL regeneration and pipeline states (OS shader cache or `.skybin` archive) | ≤ 50 ms | — | 5, 6 |
| 5 | Weights mapped, no-copy views created | ≤ 10 ms | — | — |
| 5b | Residency of 18.4 GB | **unknown: the deciding measurement**; overlapped with prefill by layer group | 0 (held by the heartbeat) | 7 |
| 6 | Tokenizer load (background) | hidden behind 4–5 | 0 | 4, 5 |
| 6b | Template render + tokenize 512 tokens | ≤ 2 ms | ≤ 2 ms (TTFT_text only) | — |
| 7 | Prefill, 512 tokens | ≤ 1/1.15 of MLX's measured time (≈ 0.61 s if MLX is at 0.70 s) | same | 5b |
| 8 | First token | no extra weight pass (≈ 34–40 ms saved vs mlx-lm) | same | — |
| 9 | GPU → host | ≤ 0.2 ms | ≤ 0.2 ms | — |
| | **Total** | prefill + residency not hidden + ≈ 0.1 s | prefill + < 1 ms | |

The cold total is dominated by prefill if residency overlaps well, and by residency if it does not. Sky's cold-TTFT advantage over mlx-lm is everything mlx-lm does before its prefill starts (interpreter, imports, copy) minus Sky's unhidden residency time.

## 10. Honest TTFT targets

From the spine (§3), for Qwen3-32B 4-bit on the M5 Max, relative to mlx-lm on the same machine and file:

| Metric | Owner's table | Real MLX today | Sky target | Condition |
|---|---|---|---|---|
| TTFT (owner's row) | 900 → 420 ms | — | warm ≥ 1.15× better (prefill gain plus the saved weight pass of §4.6); cold ≥ 3× better; repeated prefix ≥ 1.3× better with both prefix caches on | the spine's first table |
| TTFT, warm | — | ≈ 0.70 s at 512 tokens (731 tok/s pp512, `[LIKELY]`) | follows prefill (≥ 1.15× better) | prompt length stated; idle gap 1.0 s |
| TTFT, repeated prefix | — | unmeasured | ≥ 1.3× better (≈ 2× at ≤ 8 new tokens) | prefix caches enabled on both sides; ≥ 5× holds only against an engine without prefix reuse, and is reported as such (see §11) |
| TTFT, cold (process start → first token) | — | unmeasured | ≥ 3× better | weights in page cache; AOT `.skybin` |

What the owner's "900 → 420 ms" can honestly mean: for a ≈ 690-token prompt, 420 ms requires ≈ 1,640 tok/s of prefill, which is above the fp16 ceiling of the neural accelerators (≈ 1,030 tok/s at 100% of 65 TFLOPS). So it is possible only when a cached prefix removes most of the prompt, or under int8 activations, which change outputs (R8 §10; [Part IV](part04-hardware-limits.md) §1).

## 11. Disagreement with the spine

1. **"Warm ≈ parity" vs "warm ≥ 1.15×".** The spine's first table says warm TTFT is about parity; its target table says ≥ 1.15× better; [Part IV](part04-hardware-limits.md) §7 says warm TTFT improves as much as prefill (1.15–1.3×). This part follows the target table and Part IV, and adds the structural saving of §4.6: about one weight pass (≈ 34–40 ms), which is about 20% at 128 tokens and about 5% at 512. At long prompts warm TTFT converges to the prefill ratio.

   Resolved: the spine's first table now also says warm ≥ 1.15× better, citing the saved weight pass of §4.6 (SPINE §3).
2. **"Repeated prefix ≥ 5× better, with prefix caches enabled on both sides" is not supported by the arithmetic when mlx-lm's cache hits.** With both caches hitting:
   - for ≤ 8 new tokens, Sky's TTFT is one bandwidth-bound pass (≈ 36 ms), and mlx-lm's is two passes (the n − 1 new tokens, then the separate last-token step; R4 §2.1) plus per-call overhead: about **2×**;
   - for 64 new tokens (scenario S10 of [Part XXIII](part23-benchmarks.md)), both are compute-bound at ≈ 70–100 ms of prefill, and mlx-lm adds one ≈ 36 ms pass and its overhead: about **1.3–1.5×**.

   A ≥ 5× ratio is honest against mlx-lm's library path, which has no prefix reuse (#1194), or when its server cache misses (#1803, #480). We propose stating the target as: "≥ 5× against mlx-lm without prefix reuse; against a hitting mlx-lm prompt cache, report the measured ratio, expected ≈ 2× at ≤ 8 new tokens and ≈ 1.3–1.5× at 64". `[EXPERIMENTAL: S10 with both caches confirmed hitting]`

   Resolved: the spine now says repeated prefix ≥ 1.3× better (≈ 2× at ≤ 8 new tokens) with prefix caches enabled on both sides; ≥ 5× holds only against an engine without prefix reuse and is reported as such (SPINE §3).

## 12. Implementation tasks

| ID | Task | Crate / files | Deliverable | Acceptance test |
|---|---|---|---|---|
| XVIII-T1 | Startup trace | `sky-cli/src/llm/run.rs`, `sky-llm/src/trace.rs` | `--trace-startup` prints a timestamp per component of §9 (CLOCK_UPTIME_RAW) | Phases sum to the measured `exec` → token-0 time within 1 ms |
| XVIII-T2 | Parallel startup | `sky-llm/src/session.rs` | Metal init and pipeline creation, weight mapping and tokenizer load on separate threads, joined before prefill | Trace shows overlap; total ≤ the slowest phase + 10 ms (excluding residency and prefill) |
| XVIII-T3 | Progressive residency | `sky-metal/src/residency.rs` | per-layer-group sets committed in layer order; heartbeat with keep-alive and `endResidency` | S8: first-token latency after 1, 8 and 30 s gaps within 5% of back-to-back while hot; after keep-alive, re-wiring time reported |
| XVIII-T4 | One-time text tables | `sky-llm/src/text/*` | tables built once at load | A counting-allocator test shows no allocation of ≥ vocabulary size on the second request |
| XVIII-T5 | Pipelines without compiling | `sky-metal/src/{pipelines.rs,archive.rs}` | deterministic MSL regeneration from the plan cache; `sky compile --archive` harvests into the `.skybin` only | Second process start creates all pipelines in ≤ 50 ms on the M2 Pro through the OS shader cache; a `.skybin` archive is used only when `K_os` and `K_target` match; no archive file is ever written outside `sky compile` |
| XVIII-T6 | First token from the last chunk | `sky-tir` prefill specialization, `sky-llm/src/generate.rs` | LM head on the last row, GPU sampling, `StepState` init in the prefill program | No 1-token pass between prefill and token 0 (GPU timeline); token 0 equal to a reference that runs the last token separately within the D6 tolerance, and identical across chunk sizes |
| XVIII-T7 | First-run weight strategy | `sky-llm/src/load/skyweights.rs` | options B and C of §4.3, `F_NOCACHE` writer | S5 (c) first-run TTFT ≤ 1.1× mlx-lm's load-inclusive TTFT; second run maps the file |
| XVIII-T8 | Prefix cache | `sky-llm/src/prompt_cache.rs` | radix tree over KV blocks; LRU byte budget; refcounted reuse | Hit = miss bitwise for prefill-written prefixes over the test suite; no KV copy on a hit except the last partial block; S10 ratio reported |
| XVIII-T9 | Persistent prompt cache | same, plus `sky-llm/src/prompt_cache/disk.rs` | write/map blocks with the §6.2 key | A block from a different tuning entry is rejected; a 4K-token system prompt loads faster than it recomputes |
| XVIII-T10 | Pre-warm on arrival | `sky-serve` | tiny dispatch at request arrival | S8 with and without; kept only if it removes ≥ half of the gap effect |

## 13. Measurements that decide

- **Residency cost** of 18.4 GB of page-cache pages on the M5 Max, and whether per-layer-group residency overlaps with prefill (M9). This decides the cold target.
- **mlx-lm's cold breakdown** (S5 a) on the M5 Max: interpreter and imports, `pread` copy, tokenizer, prefill. Nothing in our notes measures it yet.
- **Idle-gap curve** (S8) for both engines, separating un-wiring from clock ramp.
- **Relayout build time** vs `pread` load (M9), which picks option A, B or C.
- **S10 with both caches confirmed hitting**, which settles §11 item 2.
- **Heartbeat period** vs idle power (energy target, [Part XXIII](part23-benchmarks.md) S6).

## 14. Risks and negative results

- **Residency may dominate cold start for both engines.** If wiring 18.4 GB costs seconds and cannot overlap with compute, Sky's cold advantage shrinks to the interpreter, imports and copy that mlx-lm pays; ≥ 3× may then not hold.
- **macOS changes residency behaviour between releases** (macOS 27 decides per residency set; mlx#4211). The policy must be re-validated on each major OS.
- **Wiring large memory has caused kernel panics** (mlx#3186). Sky inherits that risk with any residency policy.
- **A heartbeat costs energy.** If the period needed to keep TTFT flat is short, the energy target and the TTFT target pull against each other; the keep-alive bounds the cost to active sessions.
- **The first-token saving is only one weight pass.** It matters at short prompts and is noise at long ones; it is not a reason to expect large warm-TTFT gains.
- **Prefix-cache exactness covers prefill-written blocks only** unless prefill is restricted to decode's reduction order, at a cost we have not measured.
- **Prefill is near MLX's level on long prompts.** BaseRT, a hand-written Metal 4 runtime, is 1.44× MLX at 128-token prompts but only 1.02× at 2,048 (R4 §8). Warm-TTFT gains on long prompts depend on Part XVII beating that.
