# Part XXIII — Benchmark suite, methodology, demo, profiling and explain

This part specifies how Sky's inference performance is measured, compared, shown and explained. It turns two spine decisions into tools and procedures:
- D12, "explainability is a product feature": `sky explain` and `sky profile` for models;
- D13, "benchmarks are a separate deliverable": `sky bench llm` in the Sky repository and the public `sky-evaluation` repository.

It also fixes which claims Sky may make, on what evidence, and how a hostile reviewer can check them.

**Sources.** The research appendices used here:
- [R9](research/R9-formats-methodology.md): protocol (§7), results schema (§8), memory and power (§6.2–6.4), statistics (§6.5), quality (§6.6), packaging (§6.7), the hostile-reviewer list (§10);
- [R3](research/R3-mlx-core.md): MLX microbenchmark method and its floors; [R3-issues](research/R3-issues.md), [R3-kernels](research/R3-kernels.md);
- [R4](research/R4-mlx-lm.md): mlx-lm's measurement definitions, exact dispatch counts, ablations;
- [R5](research/R5-llamacpp.md): llama-bench definitions, published Apple numbers, llama.cpp dispatch counts;
- [R6](research/R6-apple-metal.md): Metal timing facilities, dispatch and bandwidth microbenchmarks;
- [R8](research/R8-algorithms.md): quality numbers and exactness;
- [R10](research/R10-sky-cpu-perf.md): the slope method and Sky's existing suite;
- [R2](research/R2-tinygrad.md) and [R7](research/R7-compilers.md): profiler and autotuner pitfalls, other runtimes.

Physics numbers come from [`roofline.py`](roofline.py).

---

## 1. Summary

1. **One harness, one clock, token ids in.** `sky-evaluation` drives every engine through its own driver code, feeds token ids, and timestamps every output token with `CLOCK_UPTIME_RAW`. Engine-reported rates are kept only as diagnostics. llama-bench, mlx-lm and llama.cpp's perf counters each measure something different [VERIFIED: R9 §6.1].
2. **Same-file pairs.** Sky is compared with mlx-lm on the identical MLX file and with llama.cpp on the identical GGUF file (D5). Bits per weight, weight bytes per decode token and KLD are printed next to every speed.
3. **Correctness before speed.** Four quality gates (G-tok, G-same, G-greedy, G-settings; §5.3) must pass for a file pair before any speed number for that pair is published.
4. **Microbenchmarks must close the token budget.** The suite covers every term of a per-token cost model. Their sum, at the plan's shapes and counts, must reproduce the measured token time within a stated error before any "why faster" attribution is published (§3.1).
5. **Pre-registered, interleaved, paired.** Every cell has N = 10 randomized blocks. A win is claimed only when the 95% bootstrap CI of the paired ratio excludes 1.0 [VERIFIED: R9 §6.5, §7.0].
6. **Honest memory and energy.** Peak RAM is the system-wide rise in unreclaimable memory. Energy is whole-system energy over loops of at least 60 s. Per-process footprint, RSS and chip power counters are diagnostics [VERIFIED: R9 §6.2, §6.3].
7. **Profiling uses only what Metal exposes without Xcode.** That means command-buffer GPU times, stage-boundary timestamps and Metal 4 counter heaps. Apple GPUs expose no per-dispatch hardware counters, so `sky profile` measures time and takes bytes and FLOPs from the execution plan [VERIFIED: R6 §A3, §B1].
8. **One instrument for every engine.** `skycount`, an extension of R4's Metal interposer, counts dispatches, barriers, encoders and command-buffer GPU times inside any engine's process. `sky explain`'s own counts must equal `skycount`'s on Sky.
9. **Exact counters gate pull requests; timings gate nightly runs.** Kernels, barriers, bytes and allocations per token are deterministic and can block a PR. Timing regressions are detected statistically every night.
10. **The demo shows only measured, published numbers.** Plain short-context decode is shown as parity, because that is what the evidence predicts (MLX runs at 91–97% of measured bandwidth; R3 §7.F, R4 §7.B). The scenes where Sky can lead are exact speculation, long-context decode, time to first token, small models and energy.

---

## 2. What the benchmarks must establish

The audience is a reviewer who knows MLX and llama.cpp well and assumes any speedup is a measurement error. Every target in SPINE §3 therefore becomes a fixed scenario, metric, file pair and decision rule, written down before the data exist (§5.0).

| SPINE §3 target (Qwen3-32B 4-bit, M5 Max) | Scenario (§4.3) | Metric (§6) | File pair | Target met when |
|---|---|---|---|---|
| Decode, plain, context ≤ 1K: ≥ 1.0× mlx-lm until the raw read is measured (≥ 1.05× needs ≈ 575 GB/s) | S1; S3 at depths 128 and 1K | decode tok/s | Sky vs mlx-lm, MLX 4-bit g64 | CI lower bound of Sky/mlx-lm ≥ 1.0 |
| Decode, plain, long context: ≥ 1.05× at 8K, ≥ 1.10× at 16K, ≥ 1.2× at 32K (bounds ≈ 1.08 / 1.14 / 1.24×) | S3 at 8K, 16K, 32K | decode tok/s | same | lower bound ≥ 1.05 / 1.10 / 1.2 at the respective depth |
| Verify cost (weight GEMVs): M ≤ 8 tokens at ≤ 1.3× the time of M = 1 | M3 (kernel); S9v (whole forward) | time(M) / time(1) | Sky alone; mlx-lm and llama.cpp curves shown beside it | upper bound ≤ 1.3 for every M ≤ 8 |
| Exact speculative decode: ≥ 1.8× mlx-lm plain, ≥ 1.2× mlx-lm speculative with the same draft | S9 | effective tok/s | Sky speculative vs mlx-lm plain; vs mlx-lm speculative | lower bounds ≥ 1.8 and ≥ 1.2; Sky's output byte-identical to Sky plain on 100% of prompts |
| Prefill: ≥ 1.0× on M5; on M1–M4 ≥ 1.25× mlx-lm and ≥ 1.0× llama.cpp (prompts 512–8K) | S2 at 512, 2K, 8K | prefill tok/s | same-file pairs | lower bound ≥ 1.0 (M5) / ≥ 1.25 vs mlx-lm and ≥ 1.0 vs llama.cpp (others) |
| TTFT, warm: ≥ 1.15× better | S2 | TTFT | same-file pairs | lower bound of baseline/Sky ≥ 1.15 |
| TTFT, repeated prefix: ≥ 1.3× better (≥ 5× only against an engine without prefix reuse, reported as such) | S10 | TTFT, prefix caches on in both engines | same-file pairs | lower bound ≥ 1.3 |
| TTFT, cold: ≥ 3× better | S5a | exec → first token | same-file pairs | lower bound ≥ 3 |
| Peak memory ≤ 0.9× mlx-lm at 32K | S7 at depth 32K | `peak_ram_bytes` | Sky vs mlx-lm | upper bound of Sky/mlx-lm ≤ 0.9 |
| Energy ≤ 0.85× mlx-lm, plain decode | S6 decode loop; S4 at 10 min | J/token, gross, whole system | Sky vs mlx-lm | upper bound ≤ 0.85 |

**Claim vocabulary.**
- For speeds the ratio is Sky ÷ baseline. For TTFT the speedup is baseline ÷ Sky. For RAM and energy the ratio is Sky ÷ baseline, and lower is better.
- **Win / tie / loss** (R9 §7.0): a win when the 95% CI of the paired ratio lies entirely on the better side of 1.0, a loss when it lies entirely on the worse side, and a tie otherwise.
- **Target met**: the CI bound named in the table clears the target. A median above the target with a CI that straddles it is reported as "not shown".
- **Multiple comparisons** (our proposal, not in R9): the headline claims form a fixed pre-registered list of about ten comparisons. They are tested with Holm's correction at a family-wise 5%, using bootstrap p-values (§5.7). All other cells are descriptive.

---

## 3. Microbenchmarks

### 3.1 The closure principle

A microbenchmark is useful only if it explains end-to-end time. The suite is therefore built around a per-token cost model that every microbenchmark feeds. The model extends R6's fitted cost model (R6, "Per-dispatch cost model"):

```text
t_token ≈ Σ_k t_k(shape_k, format_k, M)        kernel times at the plan's shapes, chain mode   (M3, M5, M6, M7)
        + n_dep · t_dep + n_ind · t_ind        dispatch boundaries                             (M2)
        + n_gap · t_gap                        command-buffer gaps on the GPU timeline          (M8)
        + t_host                               host work on the critical path                   (sky profile, mode run)

closure error  e = | t_token(measured, S1/S3) − t_token(predicted) | / t_token(measured)
```

- **Counts come from the plan.** `n_dep`, `n_ind` and `n_gap` come from Sky's execution plan, or from `skycount` for a competitor (§11.4).
- **Fitted values on M2 Pro** [VERIFIED: R6 §B3]:
  - `t_dep` ≈ 1.1–1.25 µs per dependent dispatch;
  - `t_ind` ≈ 0.14 µs per independent dispatch;
  - `t_gap` ≈ 0.4 µs when the next command buffer is committed in advance, ≈ 105 µs when the CPU waits for the previous one.
- **Chain-mode kernel times already contain one dependent bubble each.** The model then counts only the bubbles that are not inside a measured kernel.

**Rule.** `sky profile` and `sky explain --compare` publish an attribution only when e ≤ 5% for that model, chip and depth [EXPERIMENTAL: XXIII-T4 measures e on Qwen3-8B on the M2 Pro and on Qwen3-32B on the M5 Max]. For competitors the same prediction is computed from their kernels (§3.4, adapters) and their measured counts. The residual is printed, not forced to zero.

**Why ablation alone is not enough.** R4 removed op classes from mlx-lm's Qwen3-8B decode one at a time. The per-class savings added to 2.95 ms, but removing all non-matmul ops together saved only 1.97 ms. Removing one class also removes barrier waits that overlap other classes [VERIFIED: R4 §7.E]. Attribution needs a model with explicit bubbles; summing ablations does not work.

### 3.2 Method rules for every microbenchmark

| Rule | Evidence |
|---|---|
| Time on the GPU clock: `MTLCommandBuffer.gpuStartTime`/`gpuEndTime` (Metal 3) or `MTL4CommitFeedback.gpuStartTime`/`gpuEndTime` (Metal 4). Wall clock is used only for harness-level metrics. | One-op-per-eval wall timings measure a submit/sync floor: 149–170 µs in MLX on the M2 Pro, ≈ 0.28 ms on a machine inferred to be an M5 Max; raw Metal commit → completion is 106–125 µs [VERIFIED: R3 §7.B, R6 §B3; LIKELY for the M5 Max figure: R3-issues C5] |
| At least 64 calls and 2 ms of GPU work per command buffer. Per-call time is the slope between two call counts (K and 4K), which cancels command-buffer start and stop costs. | R6 reports marginal costs as the N = 100 → 1000 slope [VERIFIED: R6 §B3] |
| SLC-busting: rotate distinct weight (and KV) buffers totalling at least max(512 MiB, 16 × the cache knee measured by M1). | On the M2 Pro, re-reads stay cache-resident up to ≈ 8 MiB (400–490 GB/s) and are DRAM-bound from 16 MiB [VERIFIED: R6 §B2]. R3 rotated ≥ 320 MB [VERIFIED: R3 §7]. llama.cpp's `test-backend-ops` uses 31–33 MB matrices that may partly hit the cache [LIKELY: R5 §8.E]. The M5 Max SLC size is unknown [R6 open questions]. |
| Two modes per kernel: **throughput** (independent calls, concurrent encoder, no barriers) and **chain** (each call depends on the previous through a barrier). Both are published; chain is the one decode sees. | Dependent dispatch ≈ 1.1–1.25 µs vs independent ≈ 0.14 µs [VERIFIED: R6 §B3]. R3 notes its batched GEMV numbers are optimistic because independent calls overlap [VERIFIED: R3 §7.B]. |
| Warm the GPU clock for ≥ 300 ms before timing, and record the GPU frequency during the run (macmon `gpu_freq_mhz`). | DVFS ramp [VERIFIED: R6 microbench `common.swift`, `warmupGPU`]. Laptop GPU clocks drop to 0.84–1.52 GHz in some power modes [EXPERIMENTAL: R6 hardware model, FB23754032]. |
| Interleave variants in rounds inside one process. Take ≥ 15 repetitions per round and report the median of rounds, p10, p90 and min. | µs-scale dispatch numbers vary 10–20% between separate runs; bandwidth about 2% [VERIFIED: R6 Part B]. R6's `bw confirm` mode interleaves rounds [VERIFIED]. |
| Check every output against the CPU reference interpreter, or against the competitor's output for same-format kernels. | R3 §7.B and R6 §B5 check outputs; an unchecked fast kernel is the classic benchmark bug [LIKELY]. |
| M3 also checks **batch invariance**: output row 0 at M = k must equal the M = 1 output bit for bit. | D6. Exact speculation depends on it [VERIFIED: R8 §3.3–3.5]. |
| Count bytes as stored: packed values + scales + biases (or GGUF block bytes), plus activations and outputs. Count FLOPs as 2 per multiply-accumulate. | MLX 4-bit g64 is 0.5625 bytes per weight [VERIFIED: R3 §7.B] |
| Compare against competitor kernels in the same session, interleaved: MLX in the same Python process, llama.cpp `test-backend-ops` in alternating rounds. | R3 §7.G compared MLX and a custom kernel in one process [VERIFIED] |
| Never put each kernel in its own command buffer. | 14 µs GPU and 5.7 µs CPU per command buffer [VERIFIED: R6 §B3]. tinygrad's profiler does this and turns an 18.4 ms decode token into 29.4 ms [VERIFIED: R2 §8.3]. |
| Autotuner timings (`sky-tune`, Part X) follow the same rules, and the winner is re-checked inside the full plan. | tinygrad's BEAM chose a kernel it had timed at 206 µs that actually runs in ≈ 540 µs, because its Metal search never flushes caches [VERIFIED: R2 §6.4] |
| Use the **slope method** for fixed costs: two sizes separate the fixed part from the per-unit part. It applies to dispatch chains (N = 100 vs 1000), prefill (TTFT vs n_prompt) and process start. | Whole-process timings misstated Sky's CPU gaps in both directions until R10 took slopes [VERIFIED: R10 §0–1]. R6 used slopes for dispatch [VERIFIED]. |

### 3.3 The suite

The microbenchmarks live in a library module, `crates/sky-metal/src/bench/` (XXIII-T2), so that `sky-hw` calibration and `sky-tune` reuse them. They run through `sky bench kernel <name>`, the command [Part XII](part12-quantization.md) uses for its codec gates, and write JSON (`sky-microbench/1`). Competitor adapters live in `sky-evaluation/harness/micro/` (XXIII-T3).

| ID | Measures | Grid | Competitor equivalent | Term it explains | Gate |
|---|---|---|---|---|---|
| M1 stream | read, copy, write and triad GB/s; re-read bandwidth vs working-set size | 1 MiB – 4 GiB; load width; threadgroup size; grid; shared vs private storage | MLX reductions and copies (R3 §7.A) | `BW_eff` in every bandwidth term; the SLC knee | G0 |
| M2 dispatch | GPU cost per dispatch (dependent, independent, barrier, encoder per dispatch, command buffer per dispatch, ICB replay, MTL4 barriers); submit latency; command-buffer gaps | N = 1, 10, 100, 1000 | MLX tiny-op chain (R3 §7.E) | `t_dep`, `t_ind`, `t_gap` | G0 |
| M3 qgemv | µs, GB/s, % of `BW_eff`, ×(M = 1) per shape × M × format; batch-invariance bit check | shape catalog (§3.5); M ∈ {1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32}; formats MLX affine 4/6/8 (g64, also g32 and g128), bf16/fp16 dense, GGUF Q4_0, Q4_K, Q6_K, Q8_0, MXFP4 | `mx.quantized_matmul` (R3 §7.B–C); `test-backend-ops perf -o MUL_MAT` (R5 §8.E) | weight streaming; verification cost | G1 |
| M4 gemm | TFLOPS per shape, tile and dtype; `simdgroup_matrix` (ALU) vs MPP `matmul2d` (neural accelerators on M5) | M ∈ {16, 32, 64, 128, 256, 512, 2048, 8192} × linear shapes; f16, bf16, half × int4b, int8 × int8 (W8A8 mode); tiles 32×32 to 128×128 | `mx.matmul`, `mx.quantized_matmul`, PyTorch `torch.mm` on MPS, `MPSMatrixMultiplication`, `test-backend-ops` n = 512 | prefill linear layers; the ALU / neural-accelerator crossover for verification | G5 (G1 for ALU) |
| M5 attention | decode: KV GB/s vs depth × queries; prefill: TFLOPS vs prompt length | depth 128 … 128K; queries 1, 2, 4, 5, 8, 16; head configs of every model; KV f16, bf16, int8 | `mx.fast.scaled_dot_product_attention`; `test-backend-ops -o FLASH_ATTN_EXT` | attention term vs context | G5 |
| M6 small ops | RMSNorm, QK-norm, RoPE, residual add, SwiGLU/GeGLU, logsumexp, argmax, top-k/top-p/min-p, keyed Gumbel; standalone and fused into neighbours | rows 1–32 (decode, verify) and 512–8K (prefill); vocab 128,256, 151,936 and 248,320 | `mx.fast.rms_norm`, `mx.fast.rope`; llama.cpp backend ops; CPU sampling over a host copy of the logits | the non-matmul dispatches (7.6% of mlx-lm's 8B token, R4 §7.E) | G4 |
| M7 kv | KV append for 1 and M positions (with and without fused RoPE), rewind, prefix copy | positions 1–32; KV dtypes; all head configs | mlx-lm `SliceUpdate`; llama.cpp `SET_ROWS` | KV write; speculative rollback | G4 |
| M8 loop | GPU idle per token for: CPU waits then commits; pre-committed next step; K steps per command buffer; GPU-written token feedback | K ∈ {1, 2, 4, 8, 16} | MLX `async_eval` pipeline: 28.9 µs per pipelined eval vs 149–158 µs synchronous (R3 §7.E) | `t_gap`, `t_host` | G4 |
| M9 load | page-in GB/s for mmap + no-copy vs `pread` into new buffers; relayout-cache build; GGUF direct map; first token after idle gaps with and without the residency heartbeat | file sizes 0.5–40 GB; warm and cold page cache; idle 0.1–30 s | MLX `pread` loader (R3 §4); llama.cpp mmap + residency sets (R5 §1.4–1.5) | cold TTFT; idle-gap TTFT | G6 |
| M10 compile | MSL compile and pipeline creation per kernel family; binary archive and MTL4 archive load | every skeleton; tensor-op kernels | llama.cpp: 4.841 s for its first library compile after install, 14–48 ms on later starts (R5 §1.6) | cold start | G6 |

### 3.4 What each microbenchmark must show

**M1 stream.**
- It establishes `BW_eff`, the denominator of every "percent of bandwidth" in this report, on each machine, each day the machine is used for publication.
- Report two numbers: the best read kernel and the "typical vector load" kernel. On the M2 Pro they are 196 and ≈ 185 GB/s [VERIFIED: R6 §B2].
- The working-set sweep locates the cache knee that sets the rotation size for M3–M7.

**M2 dispatch.**
- It reproduces R6's table on each chip and adds the MTL4 path.
- On the M5 Max the dependent-dispatch bubble is unknown; R6 assumes ≈ 1 µs [EXPERIMENTAL: R6]. It decides how much fusion is worth there.
- R4 estimates that the non-matmul share of a token grows on the M5 Max, because GPU time per token falls about 3× while per-dispatch latency stays similar [LIKELY: R4 §10.3].

**M3 qgemv.**
- It is the G1 gate and the verify-cost target.
- For each shape it prints:
  - µs at M = 1 and GB/s as a percentage of `BW_eff`;
  - the ×(M = 1) curve for M = 2…32, in chain mode;
  - the bit check of row 0 against M = 1.
- **Competitor curves on the M2 Pro, which the adapters must reproduce before they are trusted:**
  - MLX: ×1.87 at M = 3 and ×4.43 at M = 6 on the Qwen3-32B q_proj shape [VERIFIED: R3 §7.C];
  - llama.cpp Q4_0: ×2.54 at n = 4 and ×4.49 at n = 8 on a 4096 × 14336 matrix [VERIFIED: R5 §8.E].
- **The M2 Pro curves do not predict the M5 Max.** MLX routes M ≥ 2 to `qmv_wide` only on GPU generation ≥ 15. A contributor measured a whole 70B forward on the M5 Max at +12% for M = 3, +48% for M = 5 and +125% for M = 8 [VERIFIED as contributor data: R3-issues E4]. M3 must be re-run on every chip.

**M4 gemm.**
- On the M2 Pro the anchors are:
  - MPP `matmul2d` f16: 5.42 TFLOPS;
  - register-only `simdgroup_matrix`: 5.44 TFLOPS;
  - MLX Steel fp16 GEMM: 4.84–5.03 TFLOPS;
  - MLX quantized `qmm_t`: 4.38–4.44 TFLOPS.

  [VERIFIED: R6 §B5, R3 §7.D]
- On the M5 Max the neural-accelerator FP16 peak is ≈ 64–65.5 TFLOPS and INT8 ≈ 122 TOPS [LIKELY: R6 §A2, two sources agree per core].
- Apple's 64×32 sample tile reaches about half of peak there; 64×64 or larger is needed [LIKELY: R6 §A2].
- The grid starts at M = 16 so that M4 and M3 overlap at M = 16–32. That overlap locates the crossover where neural-accelerator GEMM beats the ALU skinny GEMM for verification. R6 puts the ridge at ≈ 30 rows on the M5 Max neural accelerators and ≈ 8 on the ALU [LIKELY: R6 hardware model].
- **PyTorch MPS** is included here because MLX's own issue tracker reports PyTorch MPS ahead of MLX on some fp16/bf16 GEMM shapes:
  - MLX at ≈ 0.90–0.95× Torch on the tested shapes [VERIFIED as contributor data: R3-issues B6];
  - 1.1–1.2× slower than PyTorch MPS for 1280² bf16 on M5 [VERIFIED: R3-issues B7].

**M5 attention.**
- **Decode.** M5 reports KV bytes read per second against `BW_eff`. On the M2 Pro, mlx-lm's 8B attention streams KV at ≈ 97 GB/s, 53% of achievable [VERIFIED: R4 §7.A]. R3 measured ≈ 56 GB/s effective at 4K context for 8B-shaped layers [VERIFIED: R3 §7.F].
- **The likely cause** is that K and V are re-read per query head; MLX's GQA-deduplicating path starts only at 8K [LIKELY: R3 §7.F, R3-kernels].
- **Queries 5–8.** The grid includes 5 and 8 queries because MLX's fused decode attention handles at most 4 queries with GQA 8 and falls back above that [VERIFIED: R3 §9, R3-kernels].
- **Prefill.** The prefill half reports TFLOPS against the M4 peak.

**M6 small ops.**
- These kernels are launch- and barrier-bound at decode sizes.
- R4's ablation priced them on mlx-lm's 8B token: RMSNorms 0.76 ms, RoPE 0.29 ms, SwiGLU 0.35 ms, residual adds 0.18 ms, logsumexp 0.17 ms. The per-class figures overlap and do not add [VERIFIED: R4 §7.E].
- M6 measures each op standalone (chain mode) and fused into its neighbour. The difference is the measured value of each fusion rule in [Part IX](part09-scheduler.md).
- **Sampling** is measured on the GPU and on the CPU, including the logits copy. llama.cpp copies 151,936 × 4 B ≈ 608 KB of logits to the host per token by default [VERIFIED: R5 §6].

**M7 kv.**
- Append cost per position.
- Rewind cost. It should be a scalar update; the test proves no copy happens.
- Prefix copy GB/s, for the prompt cache.

**M8 loop.**
- The difference between "the CPU waits for token k, then commits k + 1" and "the next step is already committed and reads a GPU-written token" is ≈ 105 µs vs 0.4 µs of GPU idle per token on the M2 Pro [VERIFIED: R6 §B3].
- For a 0.6B draft model at ≈ 3.5 ms per token that gap is ≈ 3% (arithmetic).
- K steps per command buffer trade emission burstiness (ITL percentiles, §6.2) for fewer commits.

**M9 load.**
- MLX loads by `pread` into new buffers [VERIFIED: R3 §4]. llama.cpp maps GGUF without copying [VERIFIED: R5 §1.5].
- Without residency maintenance macOS unwires GPU memory after about 1 s idle. llama.cpp issue #10119 shows 26.34 ms/token becoming 429.08 ms after a 1.4 s pause [VERIFIED: R5 §9.5].
- M9's idle sweep is the microbenchmark behind S8.

**M10 compile.**
- Tensor-op pipelines take about 1–2 s to compile, which binary archives hide; Sky ships archives only inside an explicit `sky compile` `.skybin` (D19) [LIKELY: R6 §A3].
- The first execution of a freshly linked binary on macOS costs ≈ 350–400 ms of code-signing checks [VERIFIED: R10 §1].
- M10 separates both from steady-state cold start.

### 3.5 Shape catalog

Linear-layer shapes (input → output) for the models with entries in `roofline.py` `MODELS` (sourced from each `config.json`). "QKV" and "gate+up" are the horizontally fused shapes Sky's planner produces (D4, [Part VIII](part08-graph-optimizer.md)). M3 and M4 use both the separate and the fused shapes.

| Model | q | k, v | QKV fused | o | gate, up | gate+up fused | down | LM head |
|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | 1024 → 2048 | 1024 → 1024 | 1024 → 4096 | 2048 → 1024 | 1024 → 3072 | 1024 → 6144 | 3072 → 1024 | 1024 → 151936 (tied) |
| Qwen3-8B | 4096 → 4096 | 4096 → 1024 | 4096 → 6144 | 4096 → 4096 | 4096 → 12288 | 4096 → 24576 | 12288 → 4096 | 4096 → 151936 |
| Qwen3-32B | 5120 → 8192 | 5120 → 1024 | 5120 → 10240 | 8192 → 5120 | 5120 → 25600 | 5120 → 51200 | 25600 → 5120 | 5120 → 151936 |
| Qwen3-30B-A3B | 2048 → 4096 | 2048 → 512 | 2048 → 5120 | 4096 → 2048 | 2048 → 768 per expert (8 of 128 active) | 2048 → 1536 per expert | 768 → 2048 per expert | 2048 → 151936 |
| Llama 3.1 8B | 4096 → 4096 | 4096 → 1024 | 4096 → 6144 | 4096 → 4096 | 4096 → 14336 | 4096 → 28672 | 14336 → 4096 | 4096 → 128256 |
| Llama 3.x 70B | 8192 → 8192 | 8192 → 1024 | 8192 → 10240 | 8192 → 8192 | 8192 → 28672 | 8192 → 57344 | 28672 → 8192 | 8192 → 128256 |

Gemma 3, Mistral Small, gpt-oss and Qwen3.6-27B shapes are added with their `roofline.py` entries (XXIII-T20). The source configurations are in R9 §3 and R8 §1.6–1.7.

Attention head configurations for M5:

| Model | Query heads / KV heads | head_dim | Notes |
|---|---|---|---|
| Qwen3-0.6B | 16 / 8 | 128 | GQA 2 |
| Qwen3-8B, Llama 3.1 8B | 32 / 8 | 128 | GQA 4 |
| Qwen3-32B, Llama 3.x 70B | 64 / 8 | 128 | GQA 8 |
| Qwen3.6-27B (16 full-attention layers) | 24 / 4 | 256 | partial RoPE 0.25; plus 48 Gated DeltaNet layers [VERIFIED: R8 §1.7] |
| gpt-oss | 64 / 8 | 64 | learned sinks; sliding window 128 on alternate layers [VERIFIED: R9 §3] |
| Gemma 3 27B | 32 / 16 | 128 | 5 local (window 1024) : 1 global [VERIFIED: R9 §3] |

### 3.6 Calibration anchors

A microbenchmark is trusted on a new chip only after it reproduces the published numbers on the M2 Pro. These are the M2 Pro anchors, all measured by the research agents [VERIFIED]:

| Quantity | Anchor | Source |
|---|---|---|
| Best read; typical float4 read; copy; write | 196.3; 181–186; 183.0; 187.3 GB/s | R6 §B2 |
| MLX-visible streaming (2 GiB reduce) | 182.5–184 GB/s | R3 §7.A |
| Cache knee | resident ≤ 8 MiB (400–490 GB/s); DRAM from 16 MiB | R6 §B2 |
| Dependent / independent dispatch | 1.1–1.25 / 0.14 µs | R6 §B3 |
| Command buffer per dispatch | 13.9 µs GPU, 5.74 µs CPU | R6 §B3 |
| CPU→GPU→CPU round trip; command-buffer gap | 100–125 µs; 104.6 µs if the CPU waits, 0.4 µs if pre-committed | R6 §B3 |
| ICB replay | ≈ 4 µs CPU for any N; GPU per dispatch unchanged (1.38 µs with barriers) | R6 §B4 |
| Simple 4-bit affine GEMV (R6 kernel), 32B up/gate and down | 183–191 GB/s | R6 §B2 |
| MLX `affine_qmv_fast`, 32B shapes, batched | 162–177 GB/s; k/v_proj 114–115 GB/s | R3 §7.B |
| llama.cpp GEMV n = 1: Q4_0 / Q8_0 / Q6_K / Q4_K / MXFP4 | 177 / 176 / 172 / 147 / 138 GB/s | R5 §8.E |
| MPP f16 GEMM 4096³; MPP half × int4; MPS | 5.42; 4.90; 4.41 TFLOPS | R6 §B5 |
| MLX fp16 Steel GEMM; MLX quantized `qmm_t`, 32B MLP, M = 512 | 4.84–5.03; 4.38–4.44 TFLOPS | R3 §7.D |
| mlx-lm Qwen3-8B decode token: dispatches, barriers, encoders | 693, 457, 23 | R4 §7.C |

M5 Max values to compare on day one. None of these is our measurement:

| Quantity | Published value | Label |
|---|---|---|
| Achievable read bandwidth | 530 GB/s (one contributor, method not shown); R6 plans for 520–585 GB/s | LIKELY (R3-issues A4; R6 hardware model) |
| Neural-accelerator FP16 / INT8 peak | ≈ 64–65.5 TFLOPS / ≈ 122 TOPS | LIKELY (R6 §A2) |
| MLX fp16 GEMM | 52.7–59.6 TFLOPS (maintainer, mlx#3211) | VERIFIED statement (R3-issues B4) |
| MLX 70B 4-bit forward, M = 1 / 3 / 5 / 8 | 75.9 / 84.8 / 112.7 / 170.8 ms | VERIFIED as contributor data (R3-issues E4) |
| Qwen3-32B 4-bit: MLX tg128 / pp512; llama.cpp Q4_K_M tg128 / pp512 | 28.41 / 730.9; 26.38 / 667.0 | LIKELY (one community run, R5 §9.3) |
| LLaMA-2 7B Q4_0, llama.cpp `c1d0e7a`, pp512 / tg128 | 3219.99 / 119.92 | VERIFIED (discussion #4167, R5 §9.1) |

---

## 4. The end-to-end matrix

### 4.1 Models

| Model | Role | What it exercises | Decode weight bytes per token, 4-bit | KV per context position, 16-bit | Native context |
|---|---|---|---|---|---|
| Qwen3-0.6B | draft model; overhead-bound regime | tied LM head, small shapes | 0.34 GB (`roofline.py`) | 112 KiB | from config |
| Qwen3-8B | dev-box flagship (G3, G4) | GQA 4 | 4.26 GB | 144 KiB | from config |
| **Qwen3-32B** | **headline** | GQA 8, untied head, QK-norm | 17.99 GB | 256 KiB | 40,960 [VERIFIED: R9 §3.3] |
| **Qwen3.6-27B** | **second headline** (D10) | 48 Gated DeltaNet + 16 full-attention layers, head_dim 256, MTP head, vocab 248,320 | ≈ 14.4 GB [VERIFIED arithmetic: R8 §1.7] | 64 KiB, plus a 151 MB fixed recurrent state [R8 §1.7] | 262,144 [R8 §1.7] |
| Qwen3-30B-A3B | MoE | 128 experts, top-8 | 1.71 GB active | 96 KiB | from config |
| Llama 3.1 8B | breadth; long native context | llama3 RoPE scaling | 4.22 GB | 128 KiB | 131,072 (×8 of 8,192) [VERIFIED: R9 §3] |
| Llama 3.3 70B | large dense | GQA 8; same architecture as 3.1 [R9 §3] | 39.09 GB | 320 KiB | 131,072 |
| Gemma 3 27B (4B on the M2 Pro) | architecture breadth | sandwich and (1 + w) norms, GeGLU, 5:1 sliding windows, embedding scale | to be added to `roofline.py` (T20) | T20 | from config |
| Mistral Small 3.x 24B (7B v0.3 on the M2 Pro) | architecture breadth | θ 1e8 / 1e9; decoupled head_dim | T20 | T20 | from config |
| gpt-oss-20b, gpt-oss-120b | MXFP4 MoE | MXFP4 experts, q/k/v/o and expert biases, sinks, alternating window 128 | 3.71 / 5.00 GB native [R8 §1.6] | 24 / 36 KiB (full-attention layers) [R8 §1.6] | 131,072 (YaRN ×32 of 4,096) [R9 §3] |

Unlabelled byte and KV figures are computed by `roofline.py` at MLX 4-bit g64 and 16-bit KV.

**Gated models.** The gated `meta-llama` and `google` repositories were read by R9 through `unsloth` mirrors [VERIFIED: R9 §3]. The harness pins whichever repository the BF16 base comes from by commit.

### 4.2 Formats and same-file pairs

| File (all built from one pinned BF16 commit unless noted) | Nominal bpw | Built by | Pairs |
|---|---|---|---|
| MLX affine 4-bit, group 64, bf16 scales and biases | 4.5 | `mlx_lm.convert -q --q-bits 4 --q-group-size 64` | Sky vs mlx-lm |
| MLX affine 6-bit / 8-bit, group 64 | 6.5 / 8.5 | `--q-bits 6` / `--q-bits 8` | Sky vs mlx-lm |
| MLX 4-bit as published by `mlx-community` | 4.5 | downloaded, pinned by commit | Sky vs mlx-lm (secondary row: what users download) |
| HF BF16 safetensors (the base itself) | 16 | — | Sky vs mlx-lm; Sky vs PyTorch MPS (transformers) |
| GGUF BF16 | 16 | `convert_hf_to_gguf.py --outtype bf16` | Sky vs llama.cpp. Its values equal the HF BF16 tensors, proved by hashing the decoded tensors, so llama.cpp can be placed in the BF16 three-way table |
| GGUF Q4_0 / Q4_K_M / Q8_0, no imatrix | Q4_0 4.5 plus a Q6_K output tensor; Q4_K_M a Q4_K/Q6_K mix; Q8_0 8.5 | `llama-quantize` | Sky vs llama.cpp |
| GGUF from popular publishers (bartowski, unsloth) | as published | downloaded, pinned | Sky vs llama.cpp (secondary) |
| gpt-oss native MXFP4 (experts) + BF16 (rest) | mixed | original checkpoint; MLX and GGUF conversions | Sky vs mlx-lm; Sky vs llama.cpp, per file |

**Facts that shape the pairs** [VERIFIED: R9 §0, §2.3–2.5; R8 §1.3]:
- **No lossless conversion between the 4.5-bpw types.** MLX g64 ↔ Q4_0 or Q4_K has none. MLX g64 → Q4_1 is exact on Qwen3-8B but costs 5.0 bpw.
- **"Q4_0" files are mixed precision.** bartowski's Qwen3-8B Q4_0 is 4.67 bpw. For Qwen3-32B a Q6_K output tensor adds 1.1% bytes.
- **Q4_K_M is larger.** It streams 19.32 GB per Qwen3-32B token, against 17.99 GB for MLX g64: 7.4% more.
- **Group size differs between publishers.** Qwen's official `Qwen/Qwen3-32B-MLX-4bit` uses group size 128 (17.406 GB). "MLX 4-bit" therefore always names its group size.

**Cross-format rows** (Sky on an MLX file vs llama.cpp on a GGUF file) are descriptive only. They print bpw, bytes per token and KLD against BF16, and never use the words "faster" or "slower".

**Sky's relayout cache** (`.skyweights`, D9) counts as the same file only when the dequantized tensors are proved bit-identical to the source file's. The proof's hash is recorded in every result (§7.2). The cache build time is reported in S5 [VERIFIED as a proposal: R9 §9, question 6].

### 4.3 Contexts and scenarios

The "context" axis means the prompt length for prefill and TTFT, and the KV depth for decode.

| ID | Scenario | Context values | Notes |
|---|---|---|---|
| S1 | prompt 512 → decode 512 tokens | depth ≈ 512–1023 | headline short-context decode and TTFT@512 (R9 §7.4) |
| S2 | prompt n → 1 token | n ∈ {128, 512, 1K, 2K, 4K, 8K, 32K, 128K} | TTFT curve; prefill rate headline at n ≥ 2048 (§6.2) |
| S3 | depth d → decode 256 tokens | d ∈ {128, 1K, 4K, 8K, 32K, 128K} | decode vs context (the KV-bound regime) |
| S4 | S1 in a loop for 1, 10 and 30 minutes | 512 | thermal stability (§5.6) |
| S5 | exec → first token, prompt 512 | 512 | (a) warm page cache; (b) cold page cache via a fresh APFS clone; (c, optional) first run after install |
| S6 | energy loops of ≥ 60 s: decode, prefill (n = 2048), TTFT with a fixed 1.0 s idle gap; idle with the model loaded | 512 / 2048 | J/token (§6.4) |
| S7 | memory sampler at 10 Hz during S1 and S3 | 512 and 32K | peak RAM at load and in steady decode |
| S8 | TTFT and the first 8 inter-token latencies after an idle gap | gaps {0, 0.25, 1, 3, 8, 30} s, randomized, 6 per gap | residency and power-state ramp. R9 measured mlx-lm TTFT rising from 386.5 ms back-to-back to 407.9 ms after 1 s and 420.6 ms after 8 s [VERIFIED: R9 `power/ttft_gap_out.txt`] |
| S9 | speculative decoding: chat, code generation, code editing, summarization; T = 0 and T = 0.7 with seed | depth by prompt | effective tok/s, acceptance, exactness (§4.5) |
| S9v | one forward over M tokens at depth 512 and 4K | M ∈ {1…8, 12, 16, 32} | whole-model verify cost (R4 §7.D and R5 §8.G method) |
| S10 | second request sharing a 4K prefix, then 64 new tokens | 4K + 64 | repeated-prefix TTFT with prefix caches on in both engines |

S1–S7 are R9 §7.4's scenarios with the context grid widened to the assignment's values. S8–S10 are additions for the spine's TTFT and speculation targets.

**Context-extension rules.**
- **Qwen3 dense models and Qwen3-30B-A3B have a native maximum of 40,960 positions.**
  - The 128K cells need YaRN ×4. YaRN is enabled identically in every engine, asserted from engine logs and the normalized config, and labelled `context_extension: yarn ×4`.
  - Cells at ≤ 32K never enable YaRN. Qwen's card advises against static YaRN below 32K, and enabling it in one runtime only is a known trap [VERIFIED: R9 §3.1].
- **Llama 3.x, Qwen3.6-27B and gpt-oss reach 128K natively** through their configured scaling.

**Depth construction.**
- A decode cell at depth d first needs a prefill of d tokens.
- At ≤ 8K the protocol prefills fresh for every trial (`depth_mode: fresh`).
- At 32K and 128K a session prefills once. Each trial then truncates the KV cache back to d and decodes (`depth_mode: rewound`). mlx-lm exposes `cache.trim(n)` (R4 uses it in `r4_census.py`), llama-bench caches depth state for `-d` (R5 §8), and Sky rewinds natively (D10).
- One fresh trial per session checks the rewound trials. The method is valid only if the fresh and rewound decode rates agree within their CI [EXPERIMENTAL: XXIII-T16].

**Planning cost of long prompts.** At 70% of an assumed 64 TFLOPS, `roofline.py` puts the Qwen3-32B TTFT at:

| Prompt | 512 | 8K | 32K | 128K |
|---|---|---|---|---|
| TTFT | ≈ 0.7 s | ≈ 13 s | ≈ 70 s | ≈ 576 s |

A 128K prefill cell therefore costs about 10 minutes per trial and engine. 128K cells run at N = 3 (Tier B) [LIKELY, arithmetic on an assumed peak].

### 4.4 Chips and feasibility

| Chip | Role | Notes |
|---|---|---|
| M2 Pro, 16-core GPU, 16 GB | development; engineering numbers; publishable only for small models after passing `doctor` on a freshly rebooted box | GPU working set 11.84 GiB (12.71 GB) [VERIFIED: R6 §B1]. Today the box runs with 4.1 GB of swap and cannot publish [VERIFIED: R9 §6.5] |
| **M5 Max, 40-core GPU, 128 GB** | **headline** | 614 GB/s only on the 40-core part. The 32-core M5 Max is 460 GB/s and is sold only with 36 GB [VERIFIED: R6 §0, §A1]. Record chassis (14" or 16") |
| M4 Max, 40-core GPU, 128 GB | third chip (required for G8's ≥ 3 chips) | 546 GB/s; no neural accelerators; GPU generation ≥ 15, so MLX uses `qmv_wide` there [VERIFIED: R5 §9.2, R3-issues E2] |
| M3 Max (optional) | reproduce mlx-lm's published 1.82× speculative result: Qwen2.5-32B, 19.68 → 35.74 tok/s [VERIFIED: R5 §9.4] | |
| M5 Pro (optional) | compare with BaseRT's published M5 Pro numbers [VERIFIED: R7] | |

Which cells fit. Totals are `roofline.py` `memory_gb`, which assumes 0.3 GB of scratch and 0.2 GB of runtime. The M2 Pro limit is its measured GPU working set. The 128 GB limit assumes the M2 Pro's 74% working-set ratio holds, ≈ 95 GB [LIKELY: R6 hardware model]. Cells above it need `iogpu.wired_limit_mb`, which requires `sudo` and is applied identically to all engines.

| Model and format | 1K | 8K | 32K | 128K | M2 Pro (12.71 GB) | 128 GB Max |
|---|---|---|---|---|---|---|
| Qwen3-0.6B, any format | ≤ 1.8 GB | ≤ 2.6 | ≤ 5.5 | 15.9–16.7 | ≤ 32K | all |
| Qwen3-8B, 4-bit | 5.3 | 6.3 | 9.9 | 24.4 | ≤ 32K | all |
| Qwen3-8B, 8-bit | 9.4 | 10.4 | 14.0 | 28.5 | ≤ 8K, tight | all |
| Qwen3-8B, BF16 | 17.0 | 18.1 | 21.7 | 36.2 | none | all |
| Qwen3-32B, 4-bit | 19.2 | 21.1 | 27.5 | 53.3 | none | all |
| Qwen3-32B, 8-bit | 35.6 | 37.5 | 43.9 | 69.7 | none | all |
| Qwen3-32B, BF16 | 66.3 | 68.2 | 74.6 | 100.4 | none | ≤ 32K (128K needs the wired limit raised) |
| Qwen3-30B-A3B, 4-bit | 17.8 | 18.5 | 20.9 | 30.6 | none | all |
| Llama 3.x 70B, 4-bit | 40.5 | 42.9 | 50.9 | 83.1 | none | all |
| Llama 3.x 70B, 8-bit | 75.8 | 78.1 | 86.2 | 118.4 | none | ≤ 32K |
| Qwen3.6-27B, 4-bit | — | — | — | — | none: its streamed weights alone (≈ 14.4 GB) exceed 12.71 GB | all [LIKELY] |

A 64 GB machine has a working set of about 47 GB at the same 74% ratio [LIKELY]. It cannot run Qwen3-32B at 128K, Llama 70B at 32K, or the Qwen3-32B BF16 reference logits (65.5 GB of weights, R9 §9). That is why §14 asks for 128 GB.

### 4.5 Engines and settings

| Engine | Pinned by | Driver (mode A) | Reads | KV default | Sampling in the loop |
|---|---|---|---|---|---|
| Sky | release binary sha256 | Rust driver on the `sky-llm` API (`sky bench llm --driver`) | MLX (through the relayout cache), GGUF (mapped), BF16, MXFP4 | per pair: bf16 against mlx-lm, f16 against llama.cpp | GPU |
| mlx-lm + MLX | `uv.lock` (mlx, mlx-metal, mlx-lm) | Python, in process: `stream_generate` fed token ids, timestamps on each yielded token | MLX affine, mxfp4, BF16 | bf16 [VERIFIED: R9 §3.1] | GPU, inside the lazy graph [VERIFIED: R4 §2.4] |
| llama.cpp | commit plus CMake flags, or a release binary sha256 | C driver on `llama_decode` + `llama_synchronize` + host argmax, i.e. llama-bench's work plus sampling (R9 §7.4) | GGUF | f16 [VERIFIED: R9 §3.1] | CPU by default; GPU samplers exist but are off for plain generation [VERIFIED: R5 §6] |
| PyTorch MPS (transformers) | `uv.lock` (torch, transformers) | Python streamer callback | BF16 safetensors | model dtype | GPU argmax [LIKELY] |
| Optional adapters | when the code is public and can be pinned | — | — | — | — |

**Optional adapters:**
- Core AI on macOS 27, Apple's new inference framework. An Apple reviewer will ask about it, and R6 lists its competitiveness as open [VERIFIED: R6 §A3].
- BaseRT [VERIFIED: R7].
- mpk-apple (Monolith) [VERIFIED: R7].

**Settings profiles.** Every cell runs under the **matched** profile. Tier B adds an **engine-default** row that shows what a user gets out of the box.

The matched profile [VERIFIED as R9 §7.3 G4, extended]:
- The same KV dtype within a pair.
- Flash attention on: llama.cpp `-fa on`. The default became `auto` in #15434 [VERIFIED: R5 §9.5].
- The same KV capacity C = n_prompt + n_gen, rounded up to 256, in every engine:
  - llama.cpp `-c C`; its default is the trained context, auto-fitted to free memory [VERIFIED: R5 §11];
  - Sky compiles its plan for maximum context C;
  - mlx-lm grows its cache in 256-token steps [VERIFIED: R5 §12].
- Each engine's default prefill chunk, recorded:
  - mlx-lm `prefill_step_size` 2048;
  - llama.cpp `-b 2048 -ub 512`;
  - Sky's tuned chunk.

  [VERIFIED: R9 §6.1]
- No speculation, except in S9.
- Prompt and prefix caches off, except in S10. llama-server defaults to `cache_prompt: true` [VERIFIED: R9 §6.1].
- argmax sampling; EOS ignored; batch 1.
- Chat template in non-thinking mode (`enable_thinking=false`). mlx-lm turns Qwen3 thinking on by default [VERIFIED: R9 §10].

**Speculative scenario S9.**

Drafts:

| Method | Sky | mlx-lm | llama.cpp |
|---|---|---|---|
| Draft model: Qwen3-0.6B or 1.7B in the pair's format family | yes | yes | yes |
| EAGLE-3 public weights: `RedHatAI/Qwen3-32B-speculator.eagle3`, `AngelSlim/Qwen3-32B_eagle3` (D10) | yes | not supported | `draft-eagle3`, if a GGUF exists |
| MTP head of Qwen3.6-27B | yes | not merged | `draft-mtp` |
| Prompt lookup | yes | not supported | n-gram modes |

The support facts are VERIFIED in R4 §9 and R5 §7.

Protocol:
- **Draft length.** Each engine runs at its own default and at its best value from the pre-registered grid {2, 3, 4, 6, 8}. The best value is chosen on a separate tuning prompt set, so no competitor is handicapped by a bad choice.
- **Recorded per trial:**
  - effective tok/s;
  - acceptance rate and mean accepted length τ;
  - whether the output equals the same engine's plain output.
- **Expected divergence.** mlx-lm diverged from plain greedy in 3 of 12 configurations on the M2 Pro [VERIFIED: R4 §7.D] and at token 91 on the M5 Max in issue #1470 [VERIFIED: R8 §3.4].
- **Exactness requirement.** Sky's speculative output must equal Sky's plain output byte for byte, at T = 0 and at T = 0.7 with the same seed (D10).

### 4.6 Tiers and machine time

| Tier | Cells | N | Scenarios |
|---|---|---|---|
| **H** (headline) | Qwen3-32B: {MLX 4-bit g64 vs mlx-lm; Q4_K_M and Q4_0 vs llama.cpp}. Qwen3.6-27B: {MLX 4-bit, Q4_K_M}. On the M5 Max. | 10 (128K: 3) | S1–S10 |
| **A** (core) | Qwen3-0.6B, 8B, 30B-A3B; Llama 3.1 8B, 3.3 70B; Gemma 3 27B; Mistral Small 24B; gpt-oss-20b, 120b × {MLX 4-bit g64, Q4_K_M, Q4_0; native MXFP4 for gpt-oss}. On the M5 Max and M4 Max. | 10 | S1, S2 (≤ 8K), S3 (≤ 32K), S7 |
| **B** (breadth) | MLX 6/8-bit, BF16, Q8_0; all 128K cells; PyTorch MPS rows; engine-default rows; M2 Pro rows; optional chips | 5 (128K: 3) | S1–S3, S7 |

**Planning estimate** [LIKELY: arithmetic from the published MLX rates (28.4 tok/s, 731 tok/s; R5 §9.3), the `roofline.py` prefill times above, and an assumed 30 s mean cool-down].
- One S1 trial on Qwen3-32B takes ≈ 0.7 s TTFT plus 511 tokens at ≈ 28 tok/s ≈ 18 s, plus the idle gap and cool-down: ≈ 50 s.
- Tier H is therefore about two days of machine time, Tier A about one week per chip, and Tier B about one week.
- `benchmark.sh --dry-run` prints the schedule and its estimate before any run.

---

## 5. The protocol, step by step

This is R9 §7 adapted to the matrix above. Every step is executable by `benchmark.sh` (§8.2) and leaves a record in the results.

```text
PREREG commit ─► build and hash artefacts ─► prepare machine, run doctor ─► quality gates per file pair
      ─► speed / memory / energy blocks (seeded, interleaved) ─► invalidation ─► statistics ─► claims
      ─► publish everything, including warm-ups and invalid trials
```

### 5.0 Pre-registration

`PREREG.md` is committed, and its commit hash recorded, before any headline data are collected [VERIFIED: R9 §7.0]. It contains:
- the hypotheses (the table in §2) and the claim rules (win/tie/loss, target met, Holm);
- metric definitions (§6), scenarios and cells (§4), N, the block-order seed and the analysis code commit;
- quiet-gate thresholds, the cool-down rule and the invalidation rules (§5.5, §5.8);
- pinned versions of all engines, tools, models and prompts (§8.4);
- the KLD gate thresholds and their fallback if the BF16 noise floor is too high (§5.3).

A change after data collection creates a new version (`PREREG-v2`). Results cite the version they were collected under, and results collected under an earlier version stay published.

### 5.1 Artefacts: one BF16 base, every quantization built by us

Following R9 §7.1:
1. **Base.** Pin each BF16 base at a 40-hex commit and record the sha256 of every shard.
2. **MLX files.** Build with the pinned mlx-lm (§4.2 commands). Compare each with its `mlx-community` counterpart and record whether they are byte-identical.
   - If they differ, the headline uses ours, and the community file runs as a secondary row.
   - Whether `mlx_lm.convert` is deterministic is an open question; build twice and compare [R9 §9, question 2].
3. **GGUF files.** Run `convert_hf_to_gguf.py --outtype bf16`, then `llama-quantize` to Q4_0, Q4_K_M and Q8_0.
   - Use no imatrix, so the files have no data dependence.
   - Record each file's tensor-type histogram.
4. **Size metadata, from the files themselves, never from nominal bpw:**
   - `weight_bytes_total`;
   - `weight_bytes_read_per_decode_token`: all linear weights, the LM head and one embedding row;
   - effective bpw.

   `roofline.py`'s uniform-bpw formats approximate the mixed files (its `gguf-q4_0` has no Q6_K head). Cells use the file inspector's numbers (XXIII-T20).
5. **Sky's relayout cache.** For each MLX file, build `.skyweights` once and prove the dequantized tensors bit-equal with `sky weights verify` ([Part XII](part12-quantization.md) §5; D5, D9). Record the cache's sha256, the proof result and the build time.
6. **Prompts.**
   - Licence-clean English text, wrapped in each model's chat template in non-thinking mode.
   - Rendered once with pinned transformers and Jinja and a pinned date. gpt-oss and Mistral templates inject today's date [VERIFIED: R9 §10].
   - Stored as token-id arrays with a sha256, with exactly the model's BOS convention (Qwen has none).
   - Lengths as in §4.3, three texts per length (R9 §7.1).
   - A held-out set of the same shape is generated and hashed now and published after the headline run, to answer "tuned on the prompts" objections.
7. **Quality corpora** (R9 §6.6), as token ids with a sha256:
   - a Wikitext-2-style chunked corpus;
   - 100 chat prompts with BF16-reference greedy responses.

### 5.2 Machine preparation

Recorded in `machine` and `power_state` (§7). Following R9 §6.5 and §7.2:
- **Hardware and OS:** a pinned macOS build; freshly rebooted; logged in; no other user apps.
- **Power:**
  - AC power, with the adapter wattage recorded; battery 100% and not charging.
  - High Power Mode on (`pmset -g` `powermode` = 2 where the key exists). The M2 Pro 14" has no such key [VERIFIED: R9 §6.4].
  - Low Power Mode off (`notifyutil -g com.apple.system.lowpowermode` → 0).
- **Display:** lid open; fixed brightness, recorded from `ioreg -rc AppleARMBacklight`; auto-brightness and True Tone off; no external display.
- **Background services:**
  - models stored in a `*.noindex` directory, or Spotlight off;
  - Time Machine off; notifications off;
  - `caffeinate -i -s` for the whole session.
- **Ambient temperature** recorded with a thermometer.
- **`./benchmark.sh doctor`** refuses to continue if:
  - swap is in use;
  - thermal pressure is above 0;
  - Low Power Mode is on;
  - the machine is on battery (for any headline cell);
  - free RAM is below model + KV + margin. The margin is 8 GB on machines with ≥ 64 GB (R9 §7.2) and 3 GB on 16 GB machines, where R9's 8 GB would exclude the 8B model (our proposal).

  It prints `pmset -g assertions`; R9 found audio assertions that keep the SoC out of deep idle [VERIFIED: R9 §6.5].

The protocol needs no root, except optionally `pmset` (to set the power mode) and `iogpu.wired_limit_mb` (for over-size Tier B cells).

### 5.3 Correctness gates (per file pair, before any speed number)

R9 calls these G1–G4. They carry the names used in [Part XII](part12-quantization.md) §7.2 (G-tok, G-same, G-greedy, G-settings), which do not clash with the project gates G0–G9. Part XII owns the quality harness (`crates/sky-quant/src/quality/`, `sky bench llm --quality`); this part fixes when the gates run and what they block.

| Gate | Test | Pass |
|---|---|---|
| **G-tok** (tokenizer) | Sky's token ids vs golden ids from the Python `tokenizers` library, on the prompt texts and R9's 28-string adversarial corpus | identical. Engines are fed ids anyway; G-tok guards Sky's text API [VERIFIED: R9 §7.3] |
| **G-same** (identical-weights equivalence) | teacher-forced next-token distributions on the same file: 2,048 positions with the full vocabulary plus 64k positions with top-64 | KLD(Sky ‖ baseline) ≤ 0.1 × KLD(baseline ‖ BF16); top-1 agreement ≥ 99%; Sky's KLD vs BF16 ≤ baseline's + 2σ [VERIFIED: R9 §7.3; D13] |
| **G-greedy** (greedy smoke test) | 32 prompts × 128 tokens | each divergence must occur at a near-tie: the reference's top-2 logit margin < 0.1. A larger margin is a bug; stop and fix it [VERIFIED: R9 §7.3] |
| **G-settings** (settings parity) | asserted from engine logs and configs | the matched profile of §4.5, RoPE and YaRN configuration, and EOS handling |

**Calibrating G-same.**
- **The BF16 reference** is computed by two independent implementations: llama.cpp on the BF16 GGUF, and mlx-lm on BF16 MLX (or transformers). They must agree to KLD ≲ 1e-4 before either is trusted [LIKELY achievable: R9 §6.6].
- **Fallback.** If that noise floor exceeds 0.1 × the quantization KLD, the 0.1× threshold cannot be tested. PREREG fixes the fallback: the threshold becomes 2 × the measured BF16-vs-BF16 noise floor, and the change is printed in the report.
- **Expected size.** For Mistral-7B, Q4_K_M's median KLD against FP16 is 0.0075, with the top token differing at 5.8% of positions [VERIFIED: R8 §4.2b]. The MLX g64 KLD on Qwen3-32B has not been measured [R8 §4.2].
- **One logit dump format for every engine.** `llama-perplexity` stores base log-probs as uint16 and its files reach 37 GiB for Llama 3, so every engine's logits go into Part XII's compact `.skylogits` dump: per position the f32 logsumexp, the target logit, and the top-64 ids with their logits. The harness can also write llama.cpp's `.kld` format so llama.cpp's own tool can check our numbers [VERIFIED: R9 §6.6].
- **Quality runs never share a process with speed runs.**

**What is never claimed.** Bit-identity with MLX, llama.cpp or a BF16 reference is never claimed: BF16 and FP16 greedy outputs diverge on 49–100% of prompts [VERIFIED: R8 §3.4]. What Sky claims is quality parity (G-same) and byte-identity with its own plain decode (S9).

### 5.4 Two settings profiles

The **matched** profile (§4.5) is the headline.

The **engine-default** profile reruns S1–S3 with each engine's documented defaults:
- mlx-lm CLI defaults;
- `llama-cli`/`llama-server` defaults, with the context fitted automatically;
- `sky run` defaults.

It shows that no headline depends on handicapping a competitor's configuration. Its results are labelled `settings_profile: engine-default` and are never mixed into headline ratios.

### 5.5 Run order: sessions, blocks, gaps and gates

```text
for block b in 1..N:                                   # N = 10 (Tier B: 5; 128K: 3)
    order = shuffle(engines_of_pair, seed = H(seed, cell_group, b))
    for engine e in order:
        full_quiet_gate()                              # 30 s window; see below
        sample U(t) baseline for 5 s                   # memory baseline before spawn (§6.3)
        s = spawn_session(e, file, matched_settings)   # load weights from a warm page cache
        warmup(s)                                      # one trial, discarded, published
        for cell c in shuffle(cells_of_group, seed = H(seed, e, b)):
            short_quiet_gate()                         # 5 s window
            idle(1.0 s)                                # fixed idle gap before submission
            trial(s, c)                                # per-token timestamps; samplers running
            cool_down()
        stop(s)
```

- **The session is the sampling unit.** A trial value is one session's value for a cell. The CI therefore includes session-to-session variance (load, memory placement), which a user also sees.
- **Engines are never resident at the same time.** Two resident models would share wired memory, would distort the system-wide RAM metric, and would keep each other's background threads running during the other's trials. llama.cpp's residency heartbeat wakes every 5 ms, for example [VERIFIED: R5 §1.4, §6].
- **The full quiet gate** (R9 §6.5), logged per session:
  - CPU ≥ 97% idle over 30 s (`top -l 2 -s 5 -n 0`);
  - GPU < 1% active;
  - thermal pressure level 0 (notify key `com.apple.system.thermalpressurelevel`). `pmset -g therm` is useless on Apple silicon [VERIFIED: R9 §6.4];
  - PSTR within 10% of the idle baseline;
  - no change in `vm_stat` swap-ins or swap-outs.
- **The short gate** checks the same conditions over 5 s.
- **Cool-down.** Wait until thermal pressure is 0 and PSTR is within 10% of idle for 5 consecutive seconds, for at least 5 s and at most 120 s. A trial that starts after a 120 s timeout is marked `thermal-not-recovered`. R9 specifies the same conditions with a 60 s alternative; the bounds here are our proposal.
- **Warm-ups.** One discarded warm-up per process. JIT engines compile at first use, and Sky has a JIT [VERIFIED: R9 §6.5]. Warm-ups are published, because warm-up may never settle (Barrett et al., OOPSLA 2017, cited by R9).

### 5.6 Sustained thermal runs

S4 runs S1 back to back (1.0 s idle gap) for 1, 10 and 30 minutes, per engine, in randomized order.

Before each run the machine returns to idle: thermal pressure 0, PSTR within 10% of idle, and SoC temperature (macmon `temp`) within 2 °C of the first run's start.

Reported:
- the tok/s of every request over time;
- the minute-1 median;
- for 10 and 30 minutes, the last-5-minute median and its ratio to minute 1 [VERIFIED: R9 §6.4];
- the median GPU frequency;
- seconds at each thermal pressure level;
- maximum fan speed;
- chassis and power mode.

Why this matters:
- A sustained llama.cpp pp512 run fell 19% on a laptop [EXPERIMENTAL: R6, llama.cpp PR #16634].
- The 14" M5 Max reportedly loses ≈ 7.4% sustained GPU performance in High Power Mode and ≈ 25% in Automatic [LIKELY, unverified secondary sources: R9 §6.4].

### 5.7 Statistics

**Per cell** [VERIFIED as method: R9 §6.5, §7.6]:
- median, p10, p90, min, max, mean ± sd and n; p5/p95 are also kept for schema-v1 compatibility;
- the 95% bootstrap CI of the median: 10,000 resamples, percentile method.

**Per comparison**, paired by block:

```text
x_b, y_b      Sky's and the baseline's value in block b = 1..N (same cell, same file)
r            = median(x) / median(y)
r_i          = median(x[B_i]) / median(y[B_i]),  B_i = N block indices drawn with replacement, i = 1..10000
CI95         = [percentile_2.5(r_i), percentile_97.5(r_i)]
p_boot       = fraction of r_i on the wrong side of 1.0 (or of the target)
```

- **Holm's correction** applies to `p_boot` over the pre-registered headline list (§2).
- **Sample size.** With llama-bench's measured run-to-run CV of ≈ 1.1% (decode) and 0.5% (prefill), N = 10 gives about ±0.8% on the median [LIKELY: R9 §6.5].
- **Rule:** if the measured CV of a headline cell on the M5 Max exceeds 2%, that cell is re-run with N = 20 (pre-registered).
- **Coarseness.** With N = 10, a bootstrap CI of a median takes few distinct values, so cells whose ratio sits near a target also get N = 20 [LIKELY: a property of medians of small samples].

### 5.8 Invalidation and publication

A trial is invalid and re-queued when:
- a quiet gate fails;
- thermal pressure reaches 1 or more during the trial;
- `vm_stat` swap-outs, compressions or decompressions change;
- the battery discharges or charges during an AC run (`BatteryPower`, `IsCharging`). The 16" M5 Max may draw more than its 140 W adapter under full load [LIKELY: R9 §6.3];
- the engine errors or times out;
- a greedy trial's output ids differ from the session's first trial. That is nondeterminism: it is flagged, and for Sky it fails D6.

A cell with fewer than 8 valid trials of 10 is re-run as a new set of blocks. Every trial is published with its validity and reasons: valid, invalid, warm-up and re-queued [VERIFIED: R9 §7.6].

---

## 6. Metric definitions and their pitfalls

### 6.1 Clock and events

- **Clock:** `clock_gettime_nsec_np(CLOCK_UPTIME_RAW)`, read inside the engine process by the harness's driver code, not by the engine [VERIFIED: R9 §7.5].
- **`t_submit`:** immediately before the prompt ids are handed to the engine API (mode A), or before the request is written to the socket (mode B).
- **`t_tok[i]`:** when the id of output token i becomes available to the driver, for i = 0 … N − 1.
- **No I/O in the timed loop.** Timestamps go into a preallocated array and are written after the trial.

### 6.2 Latency and throughput

| Metric | Definition | Pitfall it avoids |
|---|---|---|
| **TTFT** | `t_tok[0] − t_submit`, warm process, model loaded, fixed 1.0 s idle gap, prompt cache off | mlx-lm's `prompt_tps` excludes tokenization but includes the last prompt token's decode step, its sampling and the launch of step 2 [VERIFIED: R4 §2.6]. The idle gap alone moves TTFT by up to 8.8% (S8) |
| TTFT_text (secondary) | also includes template rendering and tokenization from raw messages | a hidden tokenization boundary |
| **Prefill tok/s** | `n_prompt / TTFT`. The headline is n_prompt ≥ 2048; at 128–1K it is printed as "TTFT-based" | at 512 tokens on Qwen3-32B, the one decode step inside TTFT (≈ 35 ms at MLX's 28.4 tok/s) is ≈ 5% of the ≈ 0.7 s TTFT (arithmetic, R5 §9.3). This is why the SPINE §3 "pp512" row is re-measured with this definition |
| **Decode tok/s** | `(N − 1) / (t_tok[N−1] − t_tok[0])` at a stated depth | mlx-lm's `generation_tps` counts the first token in the numerator but not its time: +0.4% at 256 tokens, +14% at 8 tokens [VERIFIED: R4 §2.6]. llama-bench `tg128` decodes random tokens from an *empty* cache without sampling and averages per-repetition rates [VERIFIED: R5 §8, R9 §6.1] |
| ITL p50, p90, p99, max | over `t_tok[i] − t_tok[i−1]`, i ≥ 1; their mean is 1 / decode tok/s | an engine that emits K tokens per command buffer (Sky, D9) or speculative rounds produces bursts. Percentiles then measure burstiness, not throughput. K is printed with the ITL |
| E2E latency | `t_tok[N−1] − t_submit` | — |
| **Cold-start TTFT** | first token time − `exec` time of the engine process, with the page-cache state ((a) warm, (b) cold via APFS clone) and shader-cache state recorded | the first exec of a freshly linked binary pays ≈ 350–400 ms of code signing [VERIFIED: R10 §1]. llama.cpp's first-ever library compile takes 4.8 s [VERIFIED: R5 §1.6]. S5 discards the first exec after install |
| **Effective tok/s** (speculative) | the decode formula over a speculative run, plus acceptance rate, mean accepted length τ and `identical_to_plain` | speculative vs plain without the competitor's own speculative row; unexact speculation |
| Verify cost | time of one forward on M tokens ÷ time on 1 token, same depth | measuring it with one call per eval (R3 §7.C's single-eval rows compress the ratios) |
| Effective bandwidth (diagnostic) | `weight_bytes_read_per_decode_token × decode tok/s`, plus KV bytes at depth, against M1's `BW_eff` and the spec | exposes size games: a faster engine on a smaller file proves nothing (R9 §7.6) |

**The slope method on TTFT.**
- Fit TTFT(n) = a + b·n over n ≤ 2K, where attention is a small share. For Qwen3-32B attention is 6.4% of prefill FLOPs at 4K [VERIFIED: R8 §1.4].
- The intercept a is the fixed per-request cost; the slope b is the per-token prefill cost.
- Report both, and the full curve to 128K.
- R4 found mlx-lm's 8B intercept slightly negative (≈ −27 ms), i.e. no large fixed cost [LIKELY: R4 §7.A].

### 6.3 Memory

**Headline: `peak_ram_bytes`** [VERIFIED as a definition: R9 §6.2; D13].

```text
U(t)            = page_size × (wire_count + internal_page_count − purgeable_count + compressor_page_count)
                  read with host_statistics64(HOST_VM_INFO64) at 10 Hz by a sampler outside the engine
peak_ram_bytes  = max_t U(t) − median(U over the 5 s before the engine process is spawned)
```

Load peak and steady decode are reported separately. Copying loaders can peak at about 2× during load [LIKELY: R9 §6.2].

**Why per-process metrics fail.** R9's probe on the M2 Pro [VERIFIED: R9 §6.2]:

| Metric | Weights copied into buffers | mmap + no-copy weights read only by the GPU | GPU-private buffers |
|---|---|---|---|
| `phys_footprint` (Activity Monitor) | counted | **not counted** | counted |
| RSS, `ru_maxrss` | counted (pages the CPU touched) | **not counted** | not counted |
| `task_vm_info.external` | file pages the CPU touched | not counted | — |
| `MTLDevice.currentAllocatedSize` | counted | counted (buffer length, resident or not) | counted |
| `mx.get_peak_memory` | live MLX buffers only; excludes the cache, Python, driver and page cache | — | — |
| system wired / U(t) | counted | **counted** (+1085 MiB per GiB read) | counted |

llama.cpp's default path is exactly mmap + no-copy. Its footprint would look about 19 GB smaller than MLX's for the same 32B model, although both need the RAM [LIKELY: R9 §6.2].

**Guard rails** [VERIFIED as protocol: R9 §6.2]:
- **(a) Weights not counted.** If `peak_ram_bytes < 0.95 × weight_bytes_total`, the engine is reading unwired file pages through the CPU. Re-measure with `task_vm_info.external` added, and flag the result.
- **(b) Memory pressure.** `vm_stat` swap-outs, compressions and decompressions must not change during the run.
- **(c) Noise floor.** A null engine (`sleep`) through the same harness gives the noise floor, expected at ±100–300 MB [LIKELY]. Repeat 3× and take the median.

**Diagnostics, always printed beside the headline:**
- `ri_lifetime_max_phys_footprint` from `proc_pid_rusage(RUSAGE_INFO_V6)`;
- `ru_maxrss`;
- peak `external`;
- the engine's own report;
- `model_file_resident_bytes` from `mincore` on an observer mapping.

**Optional audit** [EXPERIMENTAL: R9 §6.2]: a balloon process occupies X GB and raises it until decode drops more than 5% or swap starts. RAM needed ≈ total − X_max. It cannot be gamed and runs once per headline cell.

### 6.4 Energy

**Headline: gross whole-system energy per token** [VERIFIED as a definition: R9 §6.3, §7.5; D13].

```text
E[a,b]          = Σ_i P̄_i · |window_i ∩ [a,b]|          window-averaged samples; no trapezoid
J/token decode  = E_system(decode loop ≥ 60 s) / (generated tokens − 1 per request)
J/token prefill = E_system(prefill loop ≥ 60 s, n = 2048) / total prompt tokens
ETTFT           = mean E_system[t_submit, t_tok[0]] over a ≥ 60 s TTFT loop, 1.0 s idle gap
net             = gross − P_idle · duration,   P_idle over ≥ 60 s with the model loaded and the engine idle
error bound     ≤ 2τ/T: ≤ 7% at τ = 2.1 s, T = 60 s; also the spread over ≥ 5 windows
```

**Source.**
- SMC `PSTR`, read without root at about 1 s cadence, or an external meter (preferred, `power.source`).
- R9 found PSTR and battery `SystemPowerIn` agree within 0.5–2.2%, which shows consistency, not independence.
- The external meter is a wall meter or an inline USB-C PD meter, with the battery at 100% and not charging, following MLPerf Power.
- macmon's `sys_power` is max(PSTR, CPU + GPU + ANE) [VERIFIED: R9 §6.3, macmon `metrics.rs`]. The sampler therefore reads PSTR directly.

**Pitfalls.**

| Pitfall | Evidence | Rule |
|---|---|---|
| `powermetrics` needs root, and Apple calls its values "estimated" and not for comparing devices | [VERIFIED: R9 §6.3] | not used for headline energy |
| Chip counters (CPU + GPU + ANE) exclude DRAM and the rest of the system | Qwen3-0.6B decode on mlx-lm: chip 20.7 W, chip + DRAM 25.0 W, PSTR 49.2 W. Per token 111 / 134 / 264 mJ: system energy is 2.4× the chip counters [VERIFIED: R9 §6.3] | chip and DRAM counters are diagnostics |
| Sub-second phases cannot be measured | one 0.58 s TTFT window gave 2.26 mJ/token from PSTR and 8.37 from chip counters [VERIFIED: R9 §6.3] | every energy phase loops for ≥ 60 s |
| M5 counters update in ≈ 2.1 s batches, and substring channel matching double-counts (106.7 W reported against a true 36.2 W) | [LIKELY: R9 §6.3, all-smi #410] | exact channel names per chip generation; PSTR or meter for the headline |
| Battery drain on AC | [LIKELY: R9 §6.3] | battery telemetry checked every second; the trial is invalidated |
| mlx-lm's Python host loop draws CPU power | 7.1 W of CPU during 0.6B decode on the M2 Pro [VERIFIED: R9 §6.3] | counted. It is part of mlx-lm's real cost and a fair place for Sky to win. The report names it as the source of the difference |

### 6.5 Columns printed with every speed

Every speed in every table carries these columns [VERIFIED as rule: R9 §7.6]:
- file format;
- bpw (effective, from the file);
- weight bytes per decode token;
- KV dtype;
- KLD against BF16;
- top-1 agreement;
- the pair's G-same result.

A speed without these columns does not appear in any report or slide.

---

## 7. Results JSON schema

### 7.1 Version 1 (R9 §8)

One file per (engine, model file, scenario) cell. Field names are normative. This is R9 §8 verbatim:

```json
{
  "schema_version": "sky-bench/1.0",
  "run_id": "uuid", "created_utc": "2026-10-02T12:00:00Z",
  "preregistration": {"git_sha": "40hex", "path": "PREREG.md"},
  "harness": {"name": "sky-evaluation", "version": "1.0.0", "git_sha": "40hex",
              "clock": "CLOCK_UPTIME_RAW", "driver_mode": "native|server"},
  "machine": {"model_identifier": "Mac17,x", "chassis": "14in|16in", "chip": "Apple M5 Max",
              "cpu_cores": {"super": 0, "performance": 0, "efficiency": 0}, "gpu_cores": 40,
              "memory_bytes": 0, "serial_sha256": "hex",
              "metal": {"device": "Apple M5 Max", "max_buffer_length": 0,
                        "recommended_max_working_set_size": 0},
              "page_size": 16384},
  "os": {"product_version": "26.x", "build": "25Gxx", "kernel": "Darwin 25.x"},
  "power_state": {"on_ac": true, "adapter_w": 140, "battery_percent": 100, "is_charging": false,
                  "powermode": 2, "lowpowermode": 0, "display_brightness_raw": 32768,
                  "external_displays": 0, "lid_open": true, "ambient_c": 22.0},
  "engine": {"name": "sky|mlx-lm|llama.cpp", "version": "x.y.z", "git_sha": "40hex",
             "binary_sha256": "hex", "build_flags": ["-DGGML_METAL=ON"],
             "python": "3.12.x", "lockfile_sha256": "hex",
             "settings": {"kv_cache_dtype": "bf16|f16", "flash_attention": true,
                          "prefill_chunk": 2048, "ubatch": 512, "threads": 0,
                          "prompt_cache": false, "speculative": false,
                          "wired_limit_bytes": 0, "mmap": true}},
  "model": {"base": {"repo": "Qwen/Qwen3-32B", "commit": "40hex"},
            "file_format": "mlx-safetensors|gguf",
            "source": {"repo": "org/name", "commit": "40hex", "derived_by": "cmd + tool versions"},
            "files": [{"path": "model-00001-of-00004.safetensors", "size": 0,
                       "sha256": "hex", "xet_hash": "hex"}],
            "quant": {"scheme": "mlx-affine|Q4_0|Q4_K_M", "bits": 4, "group_size": 64,
                      "tensor_type_histogram": {"Q4_0": 0, "Q6_K": 1, "F32": 0},
                      "bpw_effective": 4.5},
            "weight_bytes_total": 0, "weight_bytes_read_per_decode_token": 0,
            "tokenizer": {"files": [{"path": "tokenizer.json", "sha256": "hex"}],
                          "eos_ids": [151645, 151643]}},
  "workload": {"scenario": "S1|S2|S3|S4|S5a|S5b|S6-decode|S6-prefill|S6-ttft|S7",
               "prompt_id": "en-512-a", "prompt_ids_sha256": "hex", "n_prompt": 512,
               "n_gen": 512, "depth": 512, "batch": 1, "sampler": "argmax",
               "ignore_eos": true, "chat_template": {"mode": "non-thinking",
               "renderer": "transformers 5.18.0", "date_pinned": "2026-01-01"},
               "idle_gap_s": 1.0, "seed": 0},
  "trials": [{
    "index": 0, "block": 0, "position_in_block": 0, "warmup": false, "valid": true,
    "invalid_reasons": [],
    "quiet_gate": {"cpu_idle_pct": 98.7, "gpu_active_pct": 0.2, "thermal_pressure": 0,
                   "pstr_w": 9.1, "swap_delta_pages": 0},
    "timestamps_ns": {"t_submit": 0, "t_tok_first": 0, "t_tok_last": 0,
                      "per_token_file": "raw/tok_ts_0.bin"},
    "latency": {"ttft_ms": 0, "ttft_text_ms": null, "e2e_ms": 0,
                "itl_ms": {"p50": 0, "p90": 0, "p99": 0, "max": 0}},
    "throughput": {"prefill_tps": 0, "decode_tps": 0},
    "engine_reported": {"prompt_tps": 0, "generation_tps": 0, "peak_memory_gb": 0},
    "output": {"token_ids_sha256": "hex", "n_tokens": 512},
    "memory": {"peak_ram_bytes": 0, "steady_ram_bytes": 0, "load_peak_ram_bytes": 0,
               "baseline_used_bytes": 0, "sampler_hz": 10,
               "diagnostics": {"phys_footprint_peak": 0, "ru_maxrss": 0,
                               "task_external_peak": 0, "engine_self_report": 0,
                               "model_file_resident_bytes": 0, "metal_allocated_peak": null},
               "guard": {"weights_counted": true, "swapouts_delta": 0, "compressions_delta": 0}},
    "power": {"source": "pstr|wall_meter|usb_c_meter", "sampler": "macmon 0.8.2",
              "interval_ms": 200, "idle_w": 0,
              "energy_j": {"system": 0, "soc": 0, "soc_dram": 0},
              "j_per_token_gross": 0, "j_per_token_net": 0, "ettft_j": null,
              "battery_power_mw_max": 0},
    "thermal": {"pressure_max": 0, "seconds_at_level": {"0": 0, "1": 0},
                "thermal_state_max": 0, "gpu_freq_mhz_median": 0, "fan_rpm_max": 0}
  }],
  "summary": {"decode_tps": {"n": 10, "median": 0, "p5": 0, "p95": 0, "mean": 0, "stdev": 0,
                             "ci95_median": [0, 0], "method": "bootstrap-percentile-10000"},
              "prefill_tps": {}, "ttft_ms": {}, "peak_ram_bytes": {}, "j_per_token_gross": {},
              "sustained_ratio_last5_over_first1": null,
              "effective_bandwidth_gbps": 0},
  "quality": {"reference": {"impl": "llama.cpp-bf16+mlx-bf16", "agreement_kld": 0},
              "corpus": {"id": "wikitext2-chunks-512", "ids_sha256": "hex", "n_positions": 0},
              "vs_reference": {"kld_mean": 0, "kld_ci95": [0, 0], "kld_p99": 0, "kld_max": 0,
                               "top1_agreement": 0, "ppl": 0, "ppl_ratio": 0, "rms_dp": 0},
              "identical_weights_pair": {"other_engine": "mlx-lm", "kld_mean": 0,
                                         "top1_agreement": 0, "max_abs_logit_diff_p99": 0},
              "greedy_smoke": {"prompts": 32, "mean_match_prefix": 0, "max_divergence_margin": 0},
              "gate_pass": true},
  "comparisons": [{"metric": "decode_tps", "baseline": "mlx-lm", "candidate": "sky",
                   "ratio_of_medians": 0, "ci95": [0, 0], "pairing": "block",
                   "claim": "win|tie|loss"}],
  "raw_files": {"timestamps": "raw/", "power": "raw/power.jsonl", "memory": "raw/mem.jsonl",
                "env": "raw/env.json"}
}
```

### 7.2 Version 1.1 additions (this part's proposal)

Version 1.1 is a strict superset of version 1: no field is removed or renamed.

```json
{
  "schema_version": "sky-bench/1.1",
  "engine": {"name": "sky|mlx-lm|llama.cpp|torch-mps|<adapter>"},
  "workload": {
    "scenario": "S1|S2|S3|S4|S5a|S5b|S5c|S6-decode|S6-prefill|S6-ttft|S7|S8|S9|S9v|S10",
    "settings_profile": "matched|engine-default",
    "depth_mode": "fresh|rewound",
    "kv_capacity_tokens": 0,
    "context_extension": {"kind": "none|yarn", "factor": 1.0, "original_max_position": 0},
    "prefix_cache": {"enabled": false, "shared_prefix_tokens": 0},
    "idle_gap_sweep_s": null,
    "speculative": {"method": "none|draft-model|eagle3|mtp|prompt-lookup",
                    "draft": {"repo": "org/name", "commit": "40hex", "sha256": "hex"},
                    "draft_tokens": 0, "draft_tokens_choice": "default|tuned",
                    "acceptance_rule": "match|rejection|keyed-gumbel",
                    "temperature": 0.0, "seed": 0}
  },
  "trials": [{
    "depth_construction": {"prefill_s": 0, "rewound_from_tokens": null},
    "speculative": {"rounds": 0, "drafted": 0, "accepted": 0, "tau_mean": 0,
                    "identical_to_plain": true, "first_divergence_index": null},
    "verify_curve_ms": null,
    "metal_counters": {"instrument": "skycount 1.0", "overhead_checked": true,
                       "dispatches_per_token": 0, "barriers_per_token": 0,
                       "encoders_per_token": 0, "command_buffers_per_token": 0,
                       "icb_executes_per_token": 0, "gpu_busy_fraction": 0,
                       "cb_gap_us": {"p50": 0, "p90": 0},
                       "buffer_allocations_during_decode": 0},
    "emission": {"tokens_per_command_buffer": 1}
  }],
  "summary": {
    "decode_tps": {"p10": 0, "p90": 0, "min": 0, "max": 0, "cv": 0},
    "ttft_fit": {"intercept_ms": 0, "slope_ms_per_token": 0, "fit_range": [128, 2048]},
    "closure": {"predicted_ms_per_token": 0, "measured_ms_per_token": 0, "error": 0}
  },
  "comparisons": [{"p_boot": 0, "holm_adjusted": 0, "target": null, "target_met": null}],
  "sky": {"version": "x.y.z", "plan_sha256": "hex", "numerics": "inference",
          "opt_level": "O2", "tuning_db": {"sha256": "hex", "entries_used": 0},
          "relayout": {"cache_sha256": "hex", "dequant_bit_equal": true, "build_s": 0},
          "explain_file": "raw/explain.json", "profile_file": "raw/profile.json"}
}
```

| Field | Why it is needed |
|---|---|
| `settings_profile` | keeps engine-default rows out of headline ratios (§5.4) |
| `depth_mode`, `depth_construction` | rewound depth is a methodological choice a reader must see (§4.3) |
| `kv_capacity_tokens` | RAM depends on the allocated capacity, not only on depth. llama.cpp defaults to the trained context (R5 §11) |
| `context_extension` | YaRN in one engine only is a known trap (R9 §3.1) |
| `speculative.*`, `identical_to_plain` | the exactness claim (D10) and fairness of draft length |
| `metal_counters` | the single-instrument counts behind the "why" claims (§11.4) |
| `emission.tokens_per_command_buffer` | explains ITL percentiles for burst emitters (§6.2) |
| `ttft_fit`, `closure` | the slope method (§6.2) and the closure test (§3.1) |
| `sky.relayout` | D5's bit-equality proof for the relayout cache |
| `p_boot`, `holm_adjusted`, `target_met` | the claim rules of §2 |

### 7.3 Validation

- JSON Schema files `schema/sky-bench-1.0.schema.json` and `schema/sky-bench-1.1.schema.json` (draft 2020-12). They set `additionalProperties: false`, so a typo cannot silently drop a field.
- `harness/validate.py` checks every results file. A results directory with an invalid file is not published.
- Sky's existing `sky-bench` JSON (`"schema":1`, integer-nanosecond samples; `crates/sky-bench/src/report.rs`) stays as it is. LLM results use the new schema, and `sky bench llm` writes `sky-bench/1.1`.

---

## 8. The `sky-evaluation` repository

A public repository published at gate G8, separate from Sky so that anyone can audit the harness without trusting Sky's code. It is Apache-2.0, like Sky; results data are under CC BY 4.0 (proposal).

### 8.1 Layout

```text
sky-evaluation/
├── README.md                 what is claimed, how to run, links to published runs
├── PREREG.md                 pre-registration (versioned; tags PREREG-v1, -v2, …)
├── benchmark.sh              phases: doctor fetch build gates speed memory energy quality report
├── reproduce.sh              re-run a published run and compare with it (§8.3)
├── lock/
│   ├── uv.lock               mlx, mlx-metal, mlx-lm, transformers, tokenizers, huggingface_hub, hf_xet, numpy, torch
│   ├── llama.cpp.json        commit, CMake flags (-DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release, GGML_NATIVE, BLAS), binary sha256
│   ├── sky.json              release version, binary sha256
│   ├── tools.json            macmon version + sha256, skycount sha256
│   └── models.lock.json      every input and derived file: repo, 40-hex commit, path, size, sha256, xet hash, derived_by
├── prompts/                  token-id arrays + source text + template + renderer version + pinned date + sha256
├── corpora/                  quality corpora as ids; manifest of BF16 reference logits (hashes, download locations)
├── scenarios/                S1…S10 definitions; tiers.json (H, A, B, demo, quick)
├── harness/
│   ├── coordinator/          block scheduler, quiet gates, sessions, invalidation (Python 3.12, pinned)
│   ├── drivers/              sky (Rust binary), mlx_lm_driver.py, llama_driver.c, torch_mps_driver.py
│   ├── client/               OpenAI-compatible streaming client for mode B
│   ├── samplers/             mem (host_statistics64, 10 Hz), power (SMC PSTR; macmon pipe), thermal (notify key), battery
│   ├── quality/              .skylogits writers for every engine, KLD / top-1 / PPL, .kld export
│   ├── micro/                competitor microbenchmarks: mlx_ops.py, torch_mps.py, llama_ops.sh
│   └── stats/                bootstrap, paired ratio, claims, Holm
├── tools/skycount/           Metal call counter (dylib), its tests and validation data
├── schema/                   JSON Schema for sky-bench 1.0 and 1.1
├── report/                   Markdown/HTML generator
├── demo/                     terminal dashboard, replay player, run sheet (§9)
└── results/
    └── <date>_<chip>_<run_id>/
        ├── env.json  locks/  PREREG.sha
        ├── cells/<engine>__<file>__<scenario>.json
        ├── raw/      tok_ts_<trial>.bin  power.jsonl  mem.jsonl  thermal.jsonl  logs/
        ├── quality/  KLD summaries; top-64 dumps by hash
        ├── report.md  report.html
        └── SHA256SUMS
```

```text
harness architecture (mode A)

  coordinator (Python, pinned)  ── PREREG + scenarios → seeded block schedule → quiet gate → session → trials
        │  JSON over stdin/stdout                                        → validation → stats → report
        ├──────────────┬────────────────┬────────────────┬────────────────┐
        ▼              ▼                ▼                ▼                ▼
   sky driver     mlx_lm_driver.py  llama_driver.c   torch_mps_driver.py  (one engine process at a time)
   t_tok[i] taken in-process with CLOCK_UPTIME_RAW, buffered, written after the trial
        ▲ optional: skycount.dylib inside any engine process (§11.4)
  samplers (separate processes): memory 10 Hz · power (PSTR ≈ 1 s; IOReport 200 ms) · thermal 10 Hz · battery 1 Hz
```

### 8.2 `benchmark.sh`

Phases follow R9 §6.7:

| Phase | Does | Refuses / fails when |
|---|---|---|
| `doctor` | checks hardware, OS build, AC, power modes, free RAM and disk, swap, thermal state; records the environment | any §5.2 condition fails |
| `fetch` | resumable downloads with `HF_HUB_OFFLINE=0`; verifies the sha256 of every file *through* symlinks (`shasum -a 256 "$(realpath …)"`) | a hash differs. Hashing warms the page cache, so cold-start steps use APFS clones [VERIFIED: R9 §6.7] |
| `build` | builds llama.cpp at the pinned commit and flags, or downloads pinned binaries; installs `uv.lock`; builds the files of §5.1 | a derived file's hash differs from `models.lock.json` |
| `gates` | G-tok, G-same, G-greedy, G-settings per pair | any gate fails; that pair's speed cells are skipped and the failure is reported |
| `speed` | S1–S5, S8–S10 under §5.5 | — |
| `memory` | S7 | guard rail (b) fails |
| `energy` | S6 and S4, with an optional external meter | battery charges or discharges |
| `quality` | full quality tables | — |
| `report` | JSON validation, statistics, Markdown/HTML, SHA256SUMS | an invalid JSON file |

**Options:**
- `--tier H|A|B|demo|quick`, `--cells <glob>`, `--engines sky,mlx-lm,llama.cpp,torch-mps`, `--n`, `--seed`;
- `--dry-run`: print the schedule and the estimated machine time;
- `--offline`: `HF_HUB_OFFLINE=1`, snapshot paths only.

**Offline execution.** Every engine is pointed at snapshot paths, never at repository names:
- mlx-lm gets `--model <snapshot dir>`;
- llama.cpp gets `-m <snapshot>/<file>`, never `-hf`, which tracks `main` [VERIFIED: R9 §6.7].

**Disk.** The Qwen3-32B set alone (BF16, MLX 4-bit, two GGUFs, Sky's relayout cache) is about 140 GB [VERIFIED: R9 §6.7]. Llama 70B BF16 is 141 GB by itself (`roofline.py`). Tier H + A needs more than 1 TB free, and Tier B about 2 TB [LIKELY: sums of file sizes].

A **consumer mode** downloads the published quantized files by hash, skips the BF16 reference and uses the published reference logits [VERIFIED as proposal: R9 §6.7].

### 8.3 `reproduce.sh`

```text
./reproduce.sh results/<date>_<chip>_<run_id> [--cells headline|all|demo] [--n 10] [--verify-only]
  1. read env.json and locks/; print differences from this host (chip, GPU cores, RAM, chassis, OS build)
  2. install the pinned versions (uv.lock, llama.cpp commit, Sky binary by sha256, macmon, skycount)
  3. fetch and verify every model file and prompt by hash
  4. doctor
  5. re-run the selected cells with the published PREREG version and seed
  6. for each cell: ratio of reproduced to published median, with a bootstrap CI
  7. verdict per cell: "reproduced" if the ratio is within ±5% (the G8 criterion), else "differs" with both values
```

`--verify-only` stops after step 4: it checks pins, hashes and the machine without timing anything. A run on a different chip or OS build still executes, but its verdicts are labelled "different host".

### 8.4 Version pinning

| Item | Pinned by |
|---|---|
| Python packages | `uv.lock`, including `huggingface_hub`: its 1.x and 2.x behave differently, and 1.32 added a shared Xet-keyed blob store [VERIFIED: R9 §5.4, §6.7] |
| llama.cpp | 40-hex commit plus exact CMake flags, or an official release binary with its sha256 |
| Sky | release binary sha256, plus the tuning-DB sha256 recorded per run |
| macmon, skycount | version + sha256 |
| macOS | product version and build, recorded and compared by `reproduce.sh` |
| Models | `models.lock.json`: inputs and derived files with commands and tool versions |
| Prompts and corpora | sha256 of the id arrays; renderer version; pinned date |

### 8.5 Environment capture

Into `env.json` [VERIFIED as list: R9 §6.7]:
- `sw_vers`; `uname -a`;
- `sysctl hw.model machdep.cpu.brand_string hw.memsize hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu`;
- `system_profiler SPHardwareDataType SPPowerDataType SPDisplaysDataType -json`, with the serial number hashed;
- `pmset -g`, `pmset -g custom`, `pmset -g batt`, `pmset -g ac`, `pmset -g assertions`;
- thermal pressure and `thermalState`; backlight level;
- Metal device name, GPU family, `maxBufferLength`, `recommendedMaxWorkingSetSize`;
- `clang --version`, `rustc --version`, the Python version, the `uv.lock` hash;
- engine versions and git SHAs, the macmon and skycount versions, `HF_HUB_OFFLINE`;
- M1's `BW_eff` and the cache knee measured that day (§3.4).

### 8.6 Drivers and the timestamp contract

**Mode A (native, headline).** Each driver runs inside the engine's process and implements one request/response contract:
- **Input:** a JSON trial request (prompt-id file, `n_gen`, settings, scenario).
- **Output:** `t_submit`, `t_tok[0..N−1]`, the output ids and the engine-reported numbers, written after the trial.

| Driver | Engine call | Notes |
|---|---|---|
| `mlx_lm_driver.py` | `stream_generate` with token ids; `tokenizer.eos_token_ids = []`; argmax sampler; `mx.reset_peak_memory()` per trial | the method of R4's `r4_gen.py` [VERIFIED: R4 §7.A]. `generate_step` timings are a diagnostic variant (R4 measured 26.42 vs 25.79 ms ITL in different runs, within spread) |
| `llama_driver.c` | `llama_decode` per token, `llama_synchronize`, host argmax over the logits | the same work as llama-bench plus sampling (R9 §7.4). A variant with llama.cpp's GPU samplers (`-bs`) is a labelled extra row |
| Sky driver | `sky-llm` generation API with ids; `--ignore-eos` | built from the pinned release |
| `torch_mps_driver.py` | transformers `generate` with a streamer callback on `mps`, BF16, greedy | secondary baseline |

**Mode B (server, secondary).** `mlx_lm.server`, `llama-server` (with `cache_prompt=false`) and `sky serve` run on 127.0.0.1. One OpenAI-compatible streaming client sends the prompt as token ids and timestamps at the socket, using AIPerf definitions [VERIFIED: R9 §7.4].

Mode B exists to show that no engine wins by skipping work a real user pays for, such as detokenization and HTTP streaming. S10 runs in mode B, because prefix caches are a server feature in mlx-lm and llama.cpp.

### 8.7 Publishing

**Published:**
- every cell file;
- raw per-token timestamps;
- power, memory and thermal streams;
- `env.json`;
- all trials, including warm-ups and invalid ones with their reasons;
- quality summaries;
- the top-64 logit dumps, by hash.

The full-vocabulary dumps, about 1.2 GB per engine-file pair for 2,048 positions in fp16 [VERIFIED as estimate: R9 §6.6], go to release assets addressed by hash.

**Immutability.** A published run is never edited. A correction is a new run whose README points to the one it supersedes.

`SHA256SUMS` covers every file. Signing it is optional.

### 8.8 `sky bench llm` in the Sky repository

`sky bench llm` is the developer-facing command ([Part XIII](part13-llm-runtime.md) §15). It uses the same metric definitions (§6), writes `sky-bench/1.1` JSON, and computes medians with `crates/sky-bench/src/stats.rs`.

```text
sky bench llm hf://mlx-community/Qwen3-8B-4bit@<40-hex> --pinned --scenario S1,S3 --depth 128,4096 \
    --n 10 --json out/ [--engines sky,mlx-lm,llama.cpp] [--quality] [--quick] [--profile]
```

- **Sky only (the default).** It runs Sky's driver in process.
- **`--engines`.** It runs the pinned `sky-evaluation` drivers if they are installed, with the same block schedule. The results are labelled engineering numbers. Published comparisons come only from a `sky-evaluation` run, whose harness, pins and PREREG a reviewer can audit without trusting Sky's code.
- **`--quality`.** It runs Part XII's quality harness (the G-same and G-greedy gates).

A weekly CI job runs `sky bench llm` and `sky-evaluation` on the CI Mac and checks that their medians for Sky agree within the CI (XXIII-T10).


---

## 9. The demo

### 9.1 Purpose and rules

The owner's mission (§74) asks for a demo that would convince Apple's own engineers. It must run on one M5 Max, with Sky and the competitor side by side on the same file, prompt and settings, with live counters and profiler evidence of *why* Sky is faster.

The demo is a presentation of the protocol's results, not a separate measurement. Its scene list is fixed in PREREG before the headline data exist, so scenes are not chosen after seeing which cells Sky won.

| May be shown | May not be shown |
|---|---|
| Live runs on the stage machine, one engine at a time, each alone on the GPU | Two engines running at once. They share one memory system, so each gets an unknown fraction of the bandwidth and the comparison means nothing |
| A replay of two measured runs' per-token timestamps, side by side from t_submit = 0, labelled "replay of runs ⟨id⟩, ⟨id⟩" | Simulated, interpolated or sped-up token streams |
| Numbers from the published protocol run on this machine, each with its run_id, date, N and CI | Projected, roofline or "expected" tok/s; the owner's original table; another chip's numbers presented as this one's |
| Weight bytes per token, and achieved bandwidth as a percentage of the bandwidth M1 measured on this machine that day | `roofline.py` ceilings shown as tok/s |
| G-same results (KLD, top-1 agreement) for the files on screen | Matching greedy text offered as proof of quality |
| Sky's speculative decoding next to the competitor's own speculative mode with the same draft, and next to both engines' plain decode | Sky speculative against competitor plain alone |
| Energy per token from the S6 and S4 loops of the published run | Live J/token from windows shorter than 60 s; chip-only power presented as energy |
| Peak RAM as the system-wide rise in used memory | Activity Monitor's memory column, footprint or RSS for an mmap engine |
| A prompt the audience picks by number from the pre-registered list | A prompt chosen by us on the day |
| Scenes where Sky ties or loses, shown as ties or losses | Dropping a pre-registered scene because Sky lost it |

### 9.2 Setup

- **Machine:** the benchmark host itself, an M5 Max with the 40-core GPU and 128 GB, so the demo and the published run share hardware. The chassis is recorded.
- **Software:** exactly the pins of the published run. `reproduce.sh --verify-only` checks binaries, lockfiles and file hashes without timing anything.
- **Files:**
  - Qwen3-32B MLX 4-bit g64 (our build) for the mlx-lm scenes;
  - the Q4_K_M GGUF for the llama.cpp scene;
  - Qwen3.6-27B for its scene, if Tier H covers it by then.
- **Display:** the built-in display at fixed brightness.
  - Mirroring to a projector is allowed and labelled "stage conditions". It adds display scan-out traffic, which R3 notes costs a few GB/s for a 4K, 100 Hz display [LIKELY: R3 provenance note].
  - Both engines run under the same condition. The numbers cited on screen come from the protocol run, which had no external display.
- **Watchdog:** macOS kills long GPU work while a display is active [VERIFIED: R6 §A4]. No 128K prefill runs live. Sky's command buffers stay short: K decode steps, or one prefill chunk.
- **Dashboard:** a terminal UI, not a browser, refreshing at 2 Hz, so its own GPU and CPU load are small and equal for both engines. A richer view may run on a second machine fed over the network.

### 9.3 Screen

```text
┌─ Qwen3-32B · mlx.affine(b4, g64, bf16) · 4.50 bpw · weight bytes/token … · file sha256 … · prompt en-512-<k> sha256 … ─┐
│ mlx-lm <pinned version>                                │ Sky <pinned version>                                          │
│ <streamed text>                                        │ <streamed text>                                               │
│ TTFT … ms · decode … tok/s · tokens …/512              │ TTFT … ms · decode … tok/s · tokens …/512                     │
├────────────────────────────────────────────────────────┴───────────────────────────────────────────────────────────────┤
│ machine now: system W (PSTR) · GPU W · GPU MHz · GPU active % · memory used Δ · thermal level                          │
├────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ per token, both engines (skycount): dispatches · barriers · command buffers · GPU busy % · bandwidth % of M1 today      │
├────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ protocol run <run_id>, <date>, N = 10: Sky / mlx-lm = r [lo, hi] · G-same: KLD …, top-1 … · PREREG-v<k>                 │
└────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 9.4 Run sheet

```text
T − 1 day, on the stage machine
 1. Reboot. Log in. Quit every app. Connect the 140 W adapter. Battery → Energy Mode (on power adapter) = High Power.
 2. ./benchmark.sh doctor                              # refuses on swap, battery, thermal pressure ≥ 1, Low Power Mode
 3. ./reproduce.sh results/<headline run> --cells demo --verify-only
 4. ./benchmark.sh gates --tier demo                   # G-tok, G-same, G-greedy, G-settings for the demo files
 5. ./benchmark.sh speed --tier demo --n 10            # only if the pins changed since the headline run; prints run_id
 6. ./benchmark.sh energy --tier demo                  # S6 loops and a 10-minute S4
 7. ./benchmark.sh report --run <run_id>               # the JSON and report are published before the demo
 8. demo/rehearse.sh --run <run_id>                    # dress rehearsal; fails if any on-screen number lacks a run_id

T − 2 hours
 9. Reboot; repeat steps 1–2.
10. demo/run.sh warmup                                 # one discarded run per engine and file, shown as "warm-up"

Live
11. Scene 1, same file:     demo/run.sh files          # file sha256s; the `sky weights verify` bit-equality line
12. Scene 2, plain decode:  the audience picks k
      demo/run.sh plain --engine mlx-lm --prompt en-512-k --gen 512
      demo/run.sh plain --engine sky    --prompt en-512-k --gen 512
      demo/run.sh replay --last 2                      # the two measured streams side by side
13. Scene 3, long context:  demo/run.sh replay --run <run_id> --cell S3@32768
      # labelled replay: building 32K of depth takes about a minute per engine on Qwen3-32B
      # (roofline.py, 70% of an assumed 64 TFLOPS); run live only if time allows
14. Scene 4, exact speculation
      demo/run.sh spec --engine sky    --draft <pinned draft> --prompt code-k --gen 512
      demo/run.sh diff --engine sky                    # speculative vs plain output: the diff is empty
      demo/run.sh spec --engine mlx-lm --draft <same draft> --prompt code-k --gen 512
      demo/run.sh diff --engine mlx-lm                 # shown as measured, whatever it is
15. Scene 5, first token:   demo/run.sh cold           # exec → first token, warm page cache (S5a)
                            demo/run.sh prefix         # second request on a shared 4K prefix (S10)
16. Scene 6, why:           demo/why.sh --run <run_id> # sky explain --compare, skycount table, sky profile timeline
17. Scene 7, energy, RAM:   demo/show.sh --run <run_id> --metrics j_per_token,peak_ram   # published values only
```

### 9.5 Live counters

| Counter | Source | Refresh | Shown as |
|---|---|---|---|
| tok/s over the last 32 tokens; tokens so far; TTFT | driver timestamps (§6.1) | every token | measured |
| system power | SMC `PSTR` | ≈ 1 s | instantaneous power, labelled "not energy" |
| GPU power, GPU frequency, GPU active | macmon (IOReport) | 200 ms | channel names checked per chip generation (R9 §6.3) |
| memory used Δ | `host_statistics64` sampler | 10 Hz | rise over the pre-spawn baseline |
| thermal pressure level | notify key `com.apple.system.thermalpressurelevel` | 10 Hz | level 0–4 |
| dispatches, barriers and command buffers per token; GPU busy % | `skycount` in each engine's process (§11.4) | every token | measured, same instrument for both |
| bandwidth % | file weight bytes per token × tok/s ÷ M1's `BW_eff` | every token | measured inputs only |

`skycount`'s overhead is measured before the demo: S1 decode with and without it, N = 10. If the slowdown CI excludes 1.0 or the median slowdown exceeds 0.5%, the live screen drops the `skycount` row and shows the counts from the published counting run, labelled as such.

### 9.6 Evidence of why

Each scene's "why" panel shows measured quantities with the instrument that produced them.

| Scene | Evidence | Instrument |
|---|---|---|
| Plain decode | dispatches, barriers and command buffers per token; GPU busy fraction; command-buffer gaps. The closure-based attribution (§11.5) with its unexplained remainder. Expect parity or a small difference: MLX already runs at 91–97% of measured bandwidth (R3 §7.F, R4 §7.B) | `skycount`; `sky explain --compare` |
| Long context | KV bytes per token at 32K; KV GB/s of each engine's attention kernel at that depth; attention's share of Sky's token | M5 adapters (labelled microbenchmark); `sky profile` |
| Speculation | verify-cost curves (S9v) of both engines; acceptance; τ; the identity result | harness; S9v |
| First token | Sky's cold-start breakdown: exec, plan load, weight mapping, pipeline archive, prefill, first step. mlx-lm's exec → model-loaded → first-token split, from driver timestamps | `sky profile --mode run` host timeline; driver timestamps |
| Energy | CPU power during decode. mlx-lm's Python loop drew 7.1 W on the M2 Pro for Qwen3-0.6B [VERIFIED: R9 §6.3] | macmon `cpu_power`; S6 results |

### 9.7 When something goes wrong

- A live run that fails the short quiet gate, or reaches thermal pressure ≥ 1, is shown as failed. The scene then switches to the replay of the published trial, labelled "recorded ⟨date⟩, run ⟨id⟩".
- Each scene runs once. Nothing is re-run until a number looks better. The published CI is the claim; the live run only illustrates it.

---

## 10. Profiling: `sky profile`

### 10.1 What Apple exposes without Xcode

| Mechanism | Granularity | Availability | Cost | Use |
|---|---|---|---|---|
| `MTLCommandBuffer.gpuStartTime` / `gpuEndTime` | per command buffer | Metal 3, every chip [VERIFIED: R6 §A3] | none | mode `run`: GPU time per token, GPU busy fraction, gaps between command buffers |
| `MTL4CommitFeedback.gpuStartTime` / `gpuEndTime` | per commit | Metal 4, macOS 26 [VERIFIED: R6 §A3] | none | mode `run` on the MTL4 path |
| `MTLCounterSampleBuffer`, timestamp counter set | **stage boundaries only**, i.e. start and end of a compute pass. Dispatch-boundary sampling is unsupported | M2 Pro probe: `supportsCounterSampling(atStageBoundary)=true`, at dispatch, draw, tile and blit `false`; the only counter set is `GPUTimestamp` [VERIFIED: R6 §B1, `results/probe.txt`] | needs one pass per measured unit | modes `passes` and `split` |
| `MTL4CounterHeap` + `writeTimestamp(granularity: .relaxed / .precise, …)` on a compute encoder or command buffer, resolved with `resolveCounterRange` | between commands inside a pass | Metal 4, macOS 26 [VERIFIED: R6 §A3, Apple docs] | unknown [EXPERIMENTAL] | mode `mtl4ts` |
| One command buffer per kernel | per kernel | everywhere | +14 µs GPU and 5.7 µs CPU per dispatch [VERIFIED: R6 §B3]. tinygrad's profiler does this and turns an 18.4 ms decode token into 29.4 ms [VERIFIED: R2 §8.3] | **never** |
| Other GPU performance counters (bytes, occupancy, ALU) | — | not exposed: the timestamp set is the only counter set on the M2 Pro [VERIFIED: R6 §B1] | — | bytes and FLOPs come from the plan |
| Instruments Metal System Trace, `.gputrace` viewing, offline `metal` compiler | per kernel | need Xcode [VERIFIED: R6 §A3] | — | optional cross-check on a machine with Xcode |
| DRAM traffic counters | per agent | older `powermetrics` had a bandwidth sampler (asitop parses `GFX DCS RD/WR`), but macOS 26.6.1's `powermetrics` lists no bandwidth sampler [VERIFIED: R9 `power/powermetrics_help.txt`, asitop `parsers.py:5-24`]. Whether IOReport still carries the channels is unknown | — | [EXPERIMENTAL: XXIII-T12 probes IOReport. If present, measured DRAM bytes per decode window become a profile column] |

### 10.2 Modes

| Mode | Measures | How | Perturbation |
|---|---|---|---|
| `run` (default) | GPU start and end of every command buffer; host spans (encode, commit, waits, detokenizer thread) | properties above; host spans on `CLOCK_UPTIME_RAW` | none expected. Test: token time within 1% of an unprofiled run |
| `passes` | each compute pass of the production plan | stage-boundary samples on the plan's own passes | one sample pair per pass [EXPERIMENTAL: measured by T12] |
| `split` | each dispatch | one compute pass per dispatch, with stage-boundary samples | dependent chains cost about the same: R6 measured 1.16 µs per tiny dispatch with one encoder per dispatch vs 1.15 µs serial [VERIFIED: R6 §B3]. Independent dispatches lose their overlap (≈ 0.14 µs each when concurrent). The run prints Σ split − run as the perturbation |
| `mtl4ts` | each dispatch inside a pass | Metal 4 timestamps between commands | unknown; printed [EXPERIMENTAL] |
| `isolated` | each distinct kernel without its neighbours | replay each kernel instance K times in one command buffer with the plan's real arguments, rotating through all layers' weight buffers. A model's layers are far larger than any SLC, so this is naturally SLC-busting | none on the real run; this is the per-kernel "clean" time the closure test sums (§3.1) |

The default is `run` + `isolated`. Every number carries the mode that produced it.

### 10.3 Output

Per kernel, for one decode token at a stated depth (or one prefill chunk):

| Column | Source |
|---|---|
| kernel name: skeleton plus the L1 ops fused into it, e.g. `qgemv[rms_norm → qkv → qk_norm → rope → kv_store]` | plan |
| layers covered; count per token | plan |
| grid, threadgroup shape, threadgroup memory | kernel IR (L3) |
| bytes read and written per call, split into weights, KV and activations | plan: exact by construction from shapes and formats |
| FLOPs per call | plan |
| GPU time per call: median and p90 over tokens | measured (mode) |
| achieved GB/s and TFLOPS | derived |
| roofline time and fraction (formula below) | derived |
| predicted time, and predicted ÷ measured | cost model ([Part XXII](part22-performance-model.md)) |

```text
t_roof(k)     = max( bytes_k / BW_eff , flops_k / F_eff(path_k) )     BW_eff from M1, F_eff from M4 (ALU or neural accelerator),
roof_frac(k)  = t_roof(k) / t_measured(k)                             measured on this chip, this day, cached in sky-tune
```

**Totals per token:**
- Σ kernel time;
- dispatch bubbles (counts × `t_dep`, `t_ind`);
- command-buffer gaps;
- exposed host time;
- measured token time;
- closure error e.

A layout example. The values come from runs, so none are shown:

```text
$ sky profile hf://mlx-community/Qwen3-8B-4bit@<rev> --depth 4096 --tokens 64 --mode run,isolated
kernel                                   layers  /token  bytes/call  FLOP/call  µs (p50)  GB/s  roof%  pred/meas
qgemv[rms_norm→qkv→qk_norm→rope→kv]      0–35    36      …           …          …         …     …      …
attn_decode_gqa[split …]                 0–35    36      …           …          …         …     …      …
qgemv[o→+residual]                       0–35    36      …           …          …         …     …      …
qgemv[rms_norm→gate_up→swiglu]           0–35    36      …           …          …         …     …      …
qgemv[down→+residual]                    0–35    36      …           …          …         …     …      …
lm_head + sample                         —       1       …           …          …         …     …      …
per token: kernels … µs · bubbles … µs · CB gaps … µs · host … µs · measured … µs · closure e = …%
```

The kernel list shows the five-to-six-kernel layer that R5 and R4 recommend (R5 §11.4, R4 §10.3). The actual list is whatever the scheduler produced ([Part IX](part09-scheduler.md)).

### 10.4 Timeline export

`--timeline trace.json` writes the Chrome Trace Event Format (JSON). It opens in the Perfetto UI (https://ui.perfetto.dev) or `chrome://tracing`, with no Xcode.

| Track | Events |
|---|---|
| CPU: submit thread | encode, commit, wait (if any) |
| CPU: streamer | detokenization, output |
| GPU: command buffers | GPU start–end, gaps between buffers |
| GPU: passes | in `passes` mode |
| GPU: dispatches | in `split` or `mtl4ts` mode, named as in §10.3 |
| counters | system power, GPU power, GPU MHz (`"ph": "C"` events from the samplers) |

**Time base.**
- All GPU times are converted into the host's `CLOCK_UPTIME_RAW` domain.
- Metal 3 `gpuStartTime` is host time in seconds.
- Counter-sample timestamps convert through `MTLDevice.sampleTimestamps` pairs taken before and after the run. On the M2 Pro the probe returned equal CPU and GPU values, which suggests one domain [LIKELY: R6 `results/probe.txt`]. The conversion is checked on every chip anyway.

### 10.5 Validation (`crates/sky-metal/tests/profile_*.rs`)

1. `profile_run_overhead`: token time with mode `run` is within 1% of an unprofiled run, for Qwen3-0.6B and 8B.
2. `profile_bytes_exact`: Σ weight bytes over one token equals the file inspector's `weight_bytes_read_per_decode_token`, exactly.
3. `profile_flops_exact`: Σ FLOPs equals the analytic count, i.e. 2 × the parameters touched plus attention at the depth, as in `roofline.py`.
4. `profile_split_perturbation`: Σ split − run is printed and stable across 3 runs.
5. `profile_trace_schema`: the trace validates against the Trace Event schema, and has non-overlapping GPU spans per track.

### 10.6 Command line

```text
sky profile <model | .skybin> [--prompt-ids <file>] [--depth N] [--tokens N] [--phase decode|prefill|ttft]
            [--mode run,passes,split,mtl4ts,isolated] [--timeline trace.json] [--json profile.json]
```

---

## 11. `sky explain` for models

### 11.1 Today and what is added

- **Today:** Sky has no `sky explain` subcommand [VERIFIED: R1 §5.8]. Sky programs get `sky check --explain`, which merges in source order:
  - checker notes;
  - pass explanations with rule IDs (`FUSE-001`, `GVN-001`, `LICM-*`, `EG-COST-001`, …);
  - parallelism notes and buffer reuse;
  - numerics notes;
  - a per-function cost report;
  - citations of tuned values.
- **Added:** `sky explain <model>`, with the section flags of [Part XIII](part13-llm-runtime.md) §15 (`--config`, `--weights`, `--template`, `--kernels`, `--fusion`, `--memory`, `--ttft`) plus:
  - `--per-token`: the default view, described below;
  - `--context N`;
  - `--measure`: runs a short `sky profile` first;
  - `--from-profile <file>`;
  - `--compare <engine>=<skycount census>`;
  - `--format text|json`.
- **Sky programs:** `sky explain prog.sky` forwards to `sky check --explain`, so there is one verb for both.
- **Where the record comes from:** every pass records rule IDs and cost numbers into the plan (VI-T6, [Part VI](part06-architecture.md)). `sky explain` only reads that record. It never re-derives decisions.

### 11.2 The per-token report

| Section | Content | Source |
|---|---|---|
| Header | model; file and sha256; canonical format name (`mlx.affine(b4, g64, bf16)`, [Part XII](part12-quantization.md)); bpw; plan sha256; chip; OS; numerics contract; tuning entries; which of (chip, OS, Sky build, tuning entry) this run matches ([Part XIII](part13-llm-runtime.md) §13) | plan |
| Counts per token, exact | dispatches; dependent dispatches; barriers; passes; command buffers (or per K tokens); ICB executes; host waits on the GPU (target 0 in steady decode); CPU→GPU writes (patched scalars only); host allocations (target 0); Metal buffer creations (target 0); pipeline-state switches | plan; checked against `skycount` (§11.6) |
| Bytes per token | weights by tensor class; KV read at depth d; KV written; activation and scratch traffic (modeled); logits. Total, and the cost contract `cost bytes_per_token <= 18.2 GB` (D12) | plan, file inspector |
| Kernels | the §10.3 table; predicted time always; measured time with `--measure` | plan; profile |
| Memory plan | §11.3 | memory planner ([Part XIX](part19-memory.md)) |
| Fusion decisions | each fusion made or refused, with its rule ID, the cost-model numbers and, when measured, the M6 value of that fusion | explain records |
| Numerics | the contract per kernel (`inference`, `portable`, `relaxed`) and its reduction-order attribute (D6) | plan |

### 11.3 The memory plan, byte by byte

| Region | Contents | Size from | Checked against |
|---|---|---|---|
| weights | mapped `.skyweights` or GGUF views; the residency set | file sizes | `MTLDevice.currentAllocatedSize` delta; U(t) delta at load |
| KV arena | layout, capacity C, dtype | config × C | `kv_capacity_tokens` in the results |
| state | DeltaNet recurrent state; draft-model and MTP caches | config | |
| scratch arena | one row per tensor: offset, size, first use, last use | planner intervals | overlap test: no two live intervals share bytes |
| logits, sampling, emitted-token ring, patched scalars | | plan | |
| pipelines and archives | pipeline count and archive bytes (driver memory itself is not observable) | Metal | |

The total is compared with `currentAllocatedSize`, allowing 16 KiB of page rounding per buffer, and with S7's `peak_ram_bytes`. The difference is printed as "not in Sky's plan": driver memory, Metal libraries, the Python-free host heap.

### 11.4 One instrument for every engine: `skycount`

`skycount` extends R4's interposer `r4count.m`. R4's version swizzles the concrete compute-encoder class and counts dispatches, barriers and encoders. It was validated on a known chain: 10 dependent adds gave 10 dispatches and 9 barriers [VERIFIED: R4 §7.C].

`skycount` hooks:
- `dispatchThreadgroups:threadsPerThreadgroup:`, `dispatchThreads:threadsPerThreadgroup:` and the indirect variants;
- `memoryBarrierWithScope:` and `memoryBarrierWithResources:count:`;
- `endEncoding`;
- `executeCommandsInBuffer:withRange:`, plus ICB command encoding, so ICB replays are counted by their contents;
- `commit`, adding a completion handler that records `gpuStartTime` and `gpuEndTime`. This gives GPU busy fraction and command-buffer gaps without changing the work;
- the MTL4 compute encoder's dispatches and `barrierAfterEncoderStages…`, and `MTL4CommandQueue commit:count:` with a feedback handler;
- `newBufferWithLength:options:` and `newBufferWithBytesNoCopy:…`, for buffers created during decode.

**Injection.**
- Python drivers: `ctypes.CDLL` and an install call, as R4 did.
- C drivers and the Sky binary: `DYLD_INSERT_LIBRARIES` with a constructor. These are self-built binaries, so system-binary protections do not apply [LIKELY].

**Per-token counts.** Counts are taken over decodes of N1 and N2 tokens, and the slope is the per-token value. That separates prefill and set-up from steady decode, the same slope method `crates/sky-bench/tests/allocations.rs` uses for allocations per iteration.

**The same instrument for every engine.**
- MLX uses neither MTL4 command queues nor ICBs [VERIFIED: R3-issues A3].
- llama.cpp encodes through Metal 3 compute encoders and uses no ICBs [VERIFIED: R5 §1.1].
- `skycount` covers both API generations, so Sky's MTL4 path and replayed ICBs are counted by the same tool as the competitors' Metal 3 paths.

### 11.5 The "why faster" report

**Counts measured so far**, all on the M2 Pro at context 128:

| Per decode token | MLX / mlx-lm | llama.cpp | Source |
|---|---|---|---|
| Qwen3-8B: dispatches / barriers / encoders | **693 / 457 / 23** (exact, interposer) | **≈ 724 / ≈ 542** (688 encoded nodes + 36 attention reduces; 506 nodes after a barrier + 36 inside attention); 2 command buffers per graph | R4 §7.C; R5 §8.D2, §1.1 |
| Per layer | 19.25 dispatches | 19 nodes, ≈ 20 dispatches, ≈ 15 barriers | R4 §7.C; R5 §8.D2 |
| Qwen3-0.6B: dispatches / barriers / encoders | 541 / 362 / 11 | ≈ 564 / ≈ 394 | R4 §7.C; R5 §8.D2 |
| Qwen3-32B (64 layers) | ≈ 1,225 kernel-launching primitives (graph census). The exact count is taken with `skycount` on the M5 Max | not measured | R3 §7.F, R4 §7.C |
| Host synchronization | one `.item()` per token, pipelined behind the next step's `async_eval` | one `waitUntilCompleted` per token before host sampling | R3 §1.2, R4 §2.2; R5 §6 |
| Logits copied to the host | none for greedy (sampling on the GPU) | 151,936 × 4 B ≈ 608 KB | R4 §2.4; R5 §6 |
| Weight bytes per token (same file) | 4,257.7 MB (`mlx-community/Qwen3-8B-4bit`) | 4,431.32 MB (bartowski Q4_0, Q6_K head) | R4 §7.B; R5 §8.B |

**Attribution.** A "why faster" statement is a decomposition of a measured difference, not a list of features:

```text
Δt = t_token(baseline) − t_token(Sky)                      measured: S1 or S3, same file, same depth, same run

Δt = Δ_kernels + Δ_dispatch + Δ_gaps + Δ_host + unexplained
  Δ_kernels   Σ over op groups of t(baseline kernels) − t(Sky kernels)        M3/M5/M6/M7 adapters, chain mode, this chip
  Δ_dispatch  (n_dep,b − n_dep,s)·t_dep + (n_ind,b − n_ind,s)·t_ind          skycount counts; t_dep, t_ind from M2
  Δ_gaps      Σ command-buffer gaps per token, baseline − Sky                 skycount GPU times
  Δ_host      exposed host time per token, baseline − Sky                     driver timestamps, skycount
  unexplained Δt minus the four terms
```

- **Op groups** match by role: QKV projection, attention at depth d, output projection, FFN, LM head plus sampling. The engines fuse differently, so kernels cannot be matched one to one.
- **Threshold.** The report is printed only when |unexplained| ≤ 10% of Δt (proposal). Otherwise it prints "attribution incomplete" and the terms it has.

**Evidence that this decomposition works.** R4's budget for mlx-lm's 8B token on the M2 Pro closes the same way [VERIFIED: R4 §7.E]:
- 91% is weight and KV streaming at achievable bandwidth;
- about 2% is the GEMVs' shortfall;
- 7.6% is 435 non-matmul dispatches;
- about 0% is exposed host time.

**Layout example.** Sky's values come from runs, so none are shown:

```text
$ sky explain hf://mlx-community/Qwen3-8B-4bit@<rev> --context 128 --measure \
      --compare mlx-lm=census/mlx-lm_qwen3-8b_ctx128.json

Qwen3-8B · mlx.affine(b4, g64, bf16) · 4.50 bpw · Apple M2 Pro 16-core · numerics=inference · plan …
per decode token (depth 128)        Sky        mlx-lm 0.32.0    instrument
  GPU dispatches                    …          693              skycount, both engines
  memory barriers                   …          457              skycount
  compute encoders                  …          23               skycount
  command buffers                   …          …                skycount
  host waits on the GPU             0          1 (pipelined)    driver + skycount
  Metal buffers created in decode   0          …                skycount
  weight bytes                      4,257.7 MB (same file)      file inspector (R4 §7.B)
  token time, S1, N = 10            … ms       … ms             harness, run <id>
attribution of Δt = … ms   (closure: Sky e = …%, mlx-lm e = …%)
  kernels … ms · dispatch boundaries … ms · command-buffer gaps … ms · host … ms · unexplained … ms
```

### 11.6 Verification of `sky explain`

| Test | Checks |
|---|---|
| `explain_counts_match_skycount` (`crates/sky-cli/tests/`) | for Qwen3-0.6B and 8B at depths 128 and 4096: plan counts = `skycount` counts, exactly |
| `explain_zero_alloc_decode` | host allocations per token = 0 (counting allocator, slope over two decode lengths, as `crates/sky-bench/tests/allocations.rs`); Metal buffer creations during decode = 0 (`skycount`) |
| `explain_memory_plan_total` | memory-plan total = `currentAllocatedSize` delta ± 16 KiB per buffer |
| `explain_bytes_contract` | `cost bytes_per_token <= <bound>` holds for the golden models; a deliberately bad plan fails it |
| `explain_golden_qwen3_0_6b` | the text report of the Qwen3-0.6B plan matches a golden file, with numbers masked |

---

## 12. CI performance-regression tracking

### 12.1 Machines

- **Existing pattern.** The `Bench` workflow (`.github/workflows/bench.yml`) runs `sky-bench` weekly on `[self-hosted, sky-linux]`. It keeps inputs out of the script text, uses a `concurrency: group: bench`, and uploads JSON artifacts for 90 days [VERIFIED: repository file]. The new workflows copy that pattern.
- **`sky-macos-metal`.** The self-hosted macOS runner of VI-T2 ([Part VI](part06-architecture.md)) runs Metal correctness tests on every PR (D14).
- **Perf runner (recommendation).** A second, dedicated Mac runs nothing but nightly performance jobs. It is rebooted nightly and passes `doctor` before each run.
  - Correctness jobs heat a machine and leave caches warm, so sharing one runner makes timings noisier.
  - If only one macOS machine exists, perf jobs run in a nightly window with correctness jobs paused (shared concurrency group).
- **M5 Max benchmark host.** It runs the weekly job when it is not reserved for a protocol run.

### 12.2 What runs when

| Trigger | Workflow | Runs | Effect |
|---|---|---|---|
| PR touching `crates/sky-{tir,kernel,metal,llm,quant}` | `perf-pr.yml` | exact-counter check (§12.3); microbenchmark smoke: M1 1 GiB read, M2 1000-dispatch chain, M3 on two Qwen3-8B shapes at M = 1 and 4 | counters block the merge; timings post a comment only |
| Nightly | `perf-nightly.yml` | M1–M8 in full; S1 and S3 (depth 128, 4096) for Qwen3-0.6B and 8B, N = 5; `sky profile --mode isolated` for every kernel of both plans | opens an issue on a regression (§12.4) |
| Weekly | `perf-weekly.yml` | `sky-evaluation --tier quick` on the M5 Max host; the `sky bench llm` vs `sky-evaluation` consistency check (XXIII-T10) | issue |
| Release | manual | Tier H subset on the M5 Max | numbers for the release notes, with run_ids |

### 12.3 Exact-counter gates

- **Golden files.** `tests/perf/golden/<model>-<format>-ctx<d>.json` hold, per decode token:
  - dispatches, dependent dispatches, barriers, passes, command buffers;
  - host waits, host allocations, Metal buffer creations;
  - weight bytes and KV bytes at depth d;
  - pipelines compiled per plan.
- **Inputs.** They come from `sky explain --format json` and are cross-checked by `skycount` (§11.6).
- **Increases fail the PR.** Any increase fails unless the PR updates the golden file, and the PR description must cite the rule ID that caused the increase.
- **Decreases pass.** They pass with a note, and the golden file updates on merge.
- **Cost contracts** (`sky compile --contract`, [Part XIX](part19-memory.md)) are checked in the same job.

These numbers have no noise, so they can block a merge. Timings cannot.

### 12.4 Timing regressions

- **Baseline.** Per metric, the median of the previous 14 nightly medians.
- **Alarm.** The 3-night rolling ratio falls below 0.97 (above 1.03 for TTFT, RAM and energy), *and* the bootstrap CI of the ratio excludes 1.0.
- **Calibration.** Thresholds are set from two weeks of A/A runs on unchanged code, targeting at most one false alarm per month [EXPERIMENTAL: XXIII-T14 measures the false-alarm rate].
- **Attribution.** Each alarm carries the per-kernel deltas from `sky profile --mode isolated` and the changed closure terms. The issue then says which kernel or which count moved, not only that tok/s fell.

### 12.5 History and bisection

- Nightly results are appended as JSON lines (one per metric) to a `perf-history` repository. The raw JSON is also kept as artifacts for 90 days.
- A static HTML page plots each metric with its noise band.
- `sky-evaluation/tools/bisect.sh <good> <bad> --cell <cell>` drives `git bisect run` with the quick version of the regressing cell. A commit is "bad" when its ratio CI excludes 1.0 on the regressing side.

---

## 13. The hostile-review checklist

Each pitfall below is one a careful reviewer will look for. The R9 §10 list is extended with the pitfalls found by R2–R6.

| # | Pitfall | Evidence | How the protocol prevents it |
|---|---|---|---|
| 1 | Files of different size compared as "4-bit" | Q4_0 adds a Q6_K head (+1.1% bytes at 32B); Q4_K_M streams 7.4% more bytes than MLX g64; imatrix files differ [R9 §2.3, R8 §1.3] | same-file pairs; quants built from one BF16 commit without imatrix; bpw and bytes per token beside every speed (§4.2, §6.5) |
| 2 | KV dtype or flash-attention mismatch | mlx-lm defaults to bf16 KV, llama.cpp to f16 [R9 §3.1] | G-settings asserts the matched profile from logs (§4.5, §5.3) |
| 3 | llama-bench `tg128` presented as decode speed | random tokens, empty cache, no sampling, mean of rates [R9 §6.1] | decode at a stated depth on real prompts; harness timestamps; medians (§6.2) |
| 4 | Engine-reported rates | `generation_tps` counts token 0 without its time: +14% at 8 tokens [R4 §2.6] | harness timestamps; engine numbers are diagnostics (§6.2) |
| 5 | Prompt or prefix cache left on | llama-server defaults to `cache_prompt: true` [R9 §6.1] | off and recorded; S10 turns it on in both engines (§4.5) |
| 6 | TTFT with an unstated idle gap or tokenization boundary | the gap alone moves TTFT by up to 8.8% [R9 §6.4] | fixed 1.0 s gap; ids in; TTFT_text separate; S8 sweep (§6.2) |
| 7 | Footprint or RSS reported for an mmap engine; `mx.get_peak_memory` presented as RAM | GPU-read mmap weights are invisible to both [R9 §6.2] | system-wide `peak_ram_bytes` with guard rails (§6.3) |
| 8 | Different KV capacity between engines | llama.cpp allocates its trained context by default [R5 §11] | equal capacity C, recorded (§4.5, §7.2) |
| 9 | Chip power presented as energy | system energy was 2.4× the chip counters [R9 §6.3] | PSTR or an external meter; chip counters are diagnostics (§6.4) |
| 10 | Sub-second phases measured with 1–2 s counters | 2.26 vs 8.37 mJ/token on one 0.58 s window [R9 §6.3] | every energy phase loops for ≥ 60 s (§6.4) |
| 11 | Charging during the run, or the battery draining on a 140 W adapter | [LIKELY: R9 §6.3] | battery telemetry each second; trial invalidated (§5.8) |
| 12 | Thermal drift without interleaving; 14" vs 16"; High Power Mode off | sustained pp512 fell 19% on a laptop [EXPERIMENTAL: R6] | randomized blocks; cool-down rule; chassis and power mode recorded; S4 (§5.5, §5.6) |
| 13 | Greedy text match offered as quality proof | even Q6_K changes the top token at 3.9% of positions [R8 §4.2] | teacher-forced KLD gate G-same (§5.3) |
| 14 | Injected template dates; thinking on in one engine; different EOS lists | [R9 §10] | ids rendered once with a pinned date; non-thinking mode; EOS ignored (§4.5, §5.1) |
| 15 | Publishing from a swapping machine | the dev box ran with 4.1 GB of swap [R9 §6.5] | `doctor` refuses; swap Δ = 0 per trial (§5.2, §5.8) |
| 16 | One-op-per-eval microbenchmarks | 149–170 µs floor in MLX; ≈ 0.28 ms on an M5 Max [R3 §7.B; R3-issues C5] | ≥ 64 calls per command buffer; GPU clock; slopes (§3.2) |
| 17 | Cache-resident microbenchmark weights | cache knee ≤ 8 MiB on the M2 Pro [R6 §B2]; 31–33 MB matrices in `test-backend-ops` [R5 §8.E] | rotation ≥ max(512 MiB, 16 × knee) (§3.2) |
| 18 | Autotuner timings that never flush caches | tinygrad BEAM timed 206 µs for a ≈ 540 µs kernel [R2 §6.4] | same rules in `sky-tune`; winners re-checked inside the plan (§3.2) |
| 19 | A profiler that changes what it measures | one command buffer per kernel: 18.4 → 29.4 ms per token [R2 §8.3] | mode `run` is tested for no perturbation; `split` prints its perturbation (§10.2) |
| 20 | Cherry-picked prompts | — | pre-registered prompts; held-out set published afterwards; the audience picks in the demo (§5.1, §9) |
| 21 | Random-token prompts | distort MoE routing [LIKELY: R9 §6.1] | real text, fed as ids (§5.1) |
| 22 | Sky's speculative decode compared only with competitors' plain decode | mlx-lm has its own speculation: 1.82× on an M3 Max [R5 §9.4] | rows against competitor plain *and* competitor speculative with the same draft; draft length tuned per engine (§4.5) |
| 23 | Speculation that changes the output | mlx-lm diverged in 3 of 12 configurations on the M2 Pro [R4 §7.D] and at token 91 in #1470 [R8 §3.4] | `identical_to_plain` recorded for every engine; Sky must be 100% (§4.5, §7.2) |
| 24 | Stale competitor versions | llama.cpp's maintainers close known gaps within weeks [R5 §9.6] | dated pins; re-run at G5 and G8 and per Sky release with the latest competitor releases (§8.4) |
| 25 | Our conversion differs from the file users download | [R9 §7.1] | both reported (§5.1) |
| 26 | Wrong chip bin | the 32-core M5 Max has 460 GB/s, not 614 [R6 §0] | GPU core count recorded; 40-core host (§4.4) |
| 27 | YaRN on in one engine only | [R9 §3.1] | identical context extension, asserted (§4.3) |
| 28 | Cold start that includes first-exec signing or a first-ever shader compile for one engine only | ≈ 350–400 ms signing [R10 §1]; llama.cpp 4.8 s first compile [R5 §1.6] | first exec discarded; shader-cache state recorded; S5c optional and labelled (§6.2) |
| 29 | Idle unwiring after pauses | 26 → 429 ms/token after a 1.4 s pause without the heartbeat [R5 §9.5] | the S8 sweep reports it for every engine (§4.3) |
| 30 | External displays or other load | display scan-out takes DRAM bandwidth [LIKELY: R3 provenance note] | no external display; quiet gates; power assertions recorded (§5.2) |
| 31 | Means of per-run rates; best-of-N | llama-bench reports a mean of rates [R5 §8] | medians and CIs; every trial published (§5.7, §5.8) |
| 32 | Many comparisons, winners picked | — | pre-registered headline list with Holm's correction; every cell published (§2, §5.7) |
| 33 | Sky tuned on the benchmark prompts | — | tuning DB built on a separate prompt set; its hash and entries recorded per result; held-out prompts (§5.1, §7.2) |
| 34 | Engines interfering through residency or background threads | llama.cpp's heartbeat wakes every 5 ms [R5 §1.4] | one engine resident at a time (§5.5) |
| 35 | Sky's relayout cache treated as a different file | [R9 §9, question 6] | the bit-equality proof hash in every result; build time in S5 (§4.2, §7.2) |

---

## 14. Disagreement with the spine

1. **Energy method in the SPINE §3 target table.**
   - **The conflict.** The row "Energy per token ≤ 0.85× mlx-lm plain decode" names "powermetrics, sustained 10-minute run" as its condition. That contradicts D13, which uses whole-system energy from SMC `PSTR`.
   - **Why D13 is right.** R9 found that `powermetrics` needs root and Apple calls its values "estimated". Its "combined" power excludes DRAM. System energy was 2.4× the chip counters [VERIFIED: R9 §6.3].
   - **This chapter's choice.** The condition should read "whole-system energy (PSTR or an external meter), sustained 10-minute run". This chapter uses it.
   - Resolved: the spine now says whole-system energy (SMC `PSTR`, or an external meter), sustained ≥ 10-minute interleaved runs, never chip-only counters (SPINE §3, D13).
2. **"fp16 KV" in the §3 targets.**
   - **The conflict.** mlx-lm uses a bf16 KV cache by default [VERIFIED: R9 §3.1]. Settings parity within a pair requires one dtype.
   - **This chapter's choice.** The condition should read "16-bit KV in the baseline's dtype": bf16 against mlx-lm, f16 against llama.cpp.
   - **Effect.** Bytes per token, and so the speed targets, are unchanged. The numerics differ, which matters for G-same.
   - Resolved: the spine's decode rows now say 16-bit KV in the baseline's type (mlx-lm uses bf16) (SPINE §3).
3. **M5 Max memory (D14: "≥ 64 GB, 128 GB preferred").**
   - **What needs the room.** The matrix requested for this part includes:
     - Qwen3-32B at 128K: 53.3 GB;
     - Llama 70B 4-bit at 32K: 50.9 GB;
     - the Qwen3-32B BF16 reference logits: 65.5 GB of weights.

     These totals come from `roofline.py` and R9 §9.
   - **Why 64 GB is not enough.** On a 64 GB machine the GPU working set would be about 47 GB, if the M2 Pro's 74% ratio holds [LIKELY: R6].
   - **This chapter's requirement.** The evaluation host must have **128 GB and the 40-core GPU**; so must the M4 Max.
   - Resolved: the spine now requires an M5 Max with the 40-core GPU and 128 GB, and 128 GB for any M4 Max used as a third chip (SPINE D14).
4. **Minor: "Prefill, pp512" in §3.**
   - Under D13's definition the prefill headline is at n_prompt ≥ 2048.
   - At 512 tokens, one decode step is about 5% of TTFT on Qwen3-32B (§6.2), and 731 tok/s is a community `pp512` number.
   - This chapter reports a TTFT-based rate at 512, labelled as such, and makes the prefill headline at 2K and 8K.
   - Resolved: the spine now puts the prefill headline at prompts ≥ 2,048 (pp2048) and reports pp512 only to compare with published numbers (SPINE §3).
5. **Minor: G0 baselines "on the M2 Pro".**
   - The development box cannot produce publishable numbers as it is used today: 4.1 GB of swap [VERIFIED: R9 §6.5].
   - G0 numbers are engineering numbers unless the box is rebooted, quiet, and passes `doctor`.

---

## 15. Implementation tasks

| ID | Task | Crate / files / repo | Deliverable | Acceptance test | Gate |
|---|---|---|---|---|---|
| XXIII-T1 | GPU facts and calibration | `crates/sky-hw/src/gpu.rs` (`GpuFacts`: family, cores, neural accelerators, working set, `maxBufferLength`); `crates/sky-metal/src/bench/calibrate.rs`; results in the `sky-tune` DB | `sky hw` prints GPU facts and today's `BW_eff`, `t_dep`, `t_ind`, `F_eff` and cache knee; the fingerprint includes the GPU (R1 §9.2) | on the M2 Pro: best read within 3% of 196.3 GB/s; `t_dep` within 1.1–1.25 µs; knee between 8 and 16 MiB; the M2 Pro and the CI Mac get different fingerprints | G0 |
| XXIII-T2 | Microbenchmark suite M1–M10 | `crates/sky-metal/src/bench/{stream,dispatch,qgemv,gemm,attn,small,kv,loop,load,compile}.rs`; `sky bench kernel <id>` (`crates/sky-cli/src/bench_kernel.rs`) | the §3 suite under the §3.2 rules; `sky-microbench/1` JSON | reproduces the §3.6 M2 Pro anchors: within 3% for bandwidth, within 15% for µs-scale dispatch figures (R6 saw 10–20% run-to-run); every output checked against the reference; Sky's GEMV passes the batch-invariance bit check for M = 1…32 | G0 (M1, M2), G1 (M3), G4–G6 (rest) |
| XXIII-T3 | Competitor microbenchmark adapters | `sky-evaluation/harness/micro/{mlx_ops.py,torch_mps.py,llama_ops.sh}` | same shapes, byte accounting, rotation and modes for MLX, PyTorch MPS and llama.cpp `test-backend-ops` | on the M2 Pro: R3 §7.B batched GB/s and §7.C ratios within 5%; R5 §8.E within 10% | G0 |
| XXIII-T4 | Closure tool | `crates/sky-metal/src/bench/closure.rs`; `sky explain --closure` | predicted vs measured token time (§3.1) for Sky and, from `skycount` counts and adapters, for competitors | Qwen3-8B on the M2 Pro: e ≤ 5% for Sky; e reported for mlx-lm | G4 |
| XXIII-T5 | `sky-evaluation` skeleton | new repository: `benchmark.sh`, `reproduce.sh`, `lock/`, `prompts/`, `corpora/`, `PREREG.md` v0 | phases `doctor`, `fetch`, `build`; lockfiles for Qwen3-0.6B and 8B | `doctor` refuses on battery, swap, thermal pressure ≥ 1 and Low Power Mode (unit tests with mocked probes); `fetch` verifies sha256 through symlinks; `models.lock.json` complete for the G0 models | G0 internal, G8 public |
| XXIII-T6 | Mode-A drivers | `harness/drivers/{mlx_lm_driver.py,llama_driver.c,torch_mps_driver.py}`; Sky's driver in `crates/sky-cli/src/bench_llm.rs` | per-token `CLOCK_UPTIME_RAW` timestamps, ids in and out, buffered | mlx-lm driver rate = `generation_tps` × n/(n+1) within 2%; llama driver rate at depth d within 3% of llama-bench `-d`; repeated greedy runs give identical ids | G0 |
| XXIII-T7 | Samplers and gates | `harness/samplers/{mem.c,power.py,thermal.c,battery.py}`; `harness/coordinator/gates.py` | 10 Hz U(t); PSTR read directly; thermal level; battery; quiet gates; external-meter input | R9's memprobe table reproduced (mmap + GPU-only weights invisible to footprint, visible as wired); null-engine noise floor reported; PSTR and battery `SystemPowerIn` agree within R9's 0.5–2.2% on the M2 Pro | G0 |
| XXIII-T8 | Quality adapters | `harness/quality/` (`.skylogits` writers for mlx-lm, llama.cpp, PyTorch); Sky's side is XII-T10 | teacher-forced dumps for every engine; G-same and G-greedy tables | two BF16 references agree to KLD ≲ 1e-4 on Qwen3-0.6B, or the fallback rule triggers and is reported; our KLD of a Q4_0 file equals llama.cpp's own `.kld` tool within its uint16 noise | G3 |
| XXIII-T9 | Statistics and report | `harness/stats/{bootstrap.py,claims.py,holm.py}`, `report/` | §5.7 statistics; claim and target rules; Markdown/HTML | on synthetic lognormal data with a known ratio, 95% CI coverage is 93–97%; the report regenerates byte-identically from raw data | G0 |
| XXIII-T10 | `sky bench llm` | `crates/sky-cli/src/bench_llm.rs`, `crates/sky-bench/src/stats.rs` (CLI surface per [Part XIII](part13-llm-runtime.md) §15) | §6 metrics, `sky-bench/1.1` JSON, `--engines` through the pinned drivers | on the M2 Pro (Qwen3-8B) its Sky medians agree with `sky-evaluation`'s within the CI; JSON validates against the schema | G3 |
| XXIII-T11 | `skycount` | `sky-evaluation/tools/skycount/{skycount.m,README.md,tests/}` | Metal 3 and MTL4 hooks of §11.4; per-token counts by slope; command-buffer GPU times | reproduces R4's 693 / 457 / 23 for mlx-lm Qwen3-8B; counts llama.cpp's Qwen3-8B dispatches within 2% of R5's ≈ 724; a known 10-op chain gives 10 dispatches and 9 barriers; overhead on S1 ≤ 0.5% | G0 |
| XXIII-T12 | `sky profile` | `crates/sky-metal/src/profile/{mod.rs,timestamps.rs,trace.rs}`; `crates/sky-cli/src/profile.rs` | modes `run`, `passes`, `split`, `isolated` (+ `mtl4ts`, EXPERIMENTAL); JSON and Chrome trace; the IOReport DRAM-counter probe | the §10.5 tests pass; the trace loads in Perfetto; the IOReport probe's result (present or absent) is recorded | G4 |
| XXIII-T13 | `sky explain` for models | `crates/sky-cli/src/explain.rs`; explain records from VI-T6 | §11.2 report; memory plan; `--compare` attribution (§11.5) | the §11.6 tests pass; on the M2 Pro the attribution against mlx-lm on Qwen3-8B has |unexplained| ≤ 10% of Δt, or prints "attribution incomplete" | G4 (static), G8 (why-faster) |
| XXIII-T14 | Performance CI | `.github/workflows/{perf-pr.yml,perf-nightly.yml,perf-weekly.yml}`; `tests/perf/golden/*.json`; `perf-history` repository | §12 | a PR adding one dispatch per layer fails `perf-pr`; an injected 5% slowdown is flagged within 3 nights; ≤ 1 false alarm in 2 weeks of A/A runs | G4 |
| XXIII-T15 | Scenario and tier definitions | `sky-evaluation/scenarios/{tiers.json,S1…S10.json}`; `harness/coordinator/feasibility.py` (uses `roofline.py`) | §4 matrix; per-host feasibility; time estimate | `benchmark.sh --dry-run` lists only cells that fit the host's working set and prints a time estimate | G5 |
| XXIII-T16 | Rewound depth | drivers; Sky KV truncation | `depth_mode: rewound` for 32K and 128K cells | at 32K, fresh and rewound decode rates agree within their CI for every engine; otherwise rewound mode is disabled for that engine | G5 |
| XXIII-T17 | Speculative scenarios | `harness/scenarios/spec.py`; drivers | S9 and S9v, with per-engine draft-length tuning on a separate prompt set | Sky's `identical_to_plain` is 100% at T = 0 and T = 0.7 with a seed; the mlx-lm #1470 script is run on the M5 Max and its result recorded | G6 |
| XXIII-T18 | Publication and outside reproduction | `results/` layout, `SHA256SUMS`, `reproduce.sh` verdicts | published Tier H run, ≥ 3 chips | an outside party reproduces the headline cells within 5% with `reproduce.sh` | G8 |
| XXIII-T19 | Demo | `sky-evaluation/demo/{tui/,run.sh,why.sh,show.sh,rehearse.sh,RUN-SHEET.md}` | §9 | dress rehearsal on the M5 Max: every on-screen number traces to a run_id; a test fails if the UI's data source contains any field not read from a results file or a live sampler | G8 |
| XXIII-T20 | Model physics and file bytes | `docs/inference/roofline.py` (Gemma 3, Mistral Small, gpt-oss, Qwen3.6-27B); `sky-quant` file inspector | per-file `weight_bytes_read_per_decode_token` from the tensor-type histogram | parameter totals match the HF safetensors totals, as R8 §1.6 checked; computed bytes equal the files' tensor data sizes exactly | G0 |

---

## 16. Measurements that decide

1. **M5 Max achievable bandwidth and cache knee (M1).**
   - It sets `BW_eff` for every percentage in the report and the rotation size of every microbenchmark.
   - If achievable bandwidth is below 85% of spec, every ceiling in [Part IV](part04-hardware-limits.md) moves.
   - R6 plans for 520–585 GB/s; one contributor reported 530 GB/s.
2. **Run-to-run CV of the headline cells on the M5 Max under this protocol.**
   - It decides N: 10, or 20 if the CV exceeds 2%.
   - It decides whether the 1.0× decode parity target can be shown at all. With a CV of about 1%, the CI half-width is about ±0.8% (R9).
3. **BF16-vs-BF16 cross-runtime KLD.** It decides whether G-same's 0.1× threshold is testable or the pre-registered fallback applies (R9 §9, question 4).
4. **Closure error e** on the M2 Pro (Qwen3-8B) and the M5 Max (Qwen3-32B).
   - It decides whether microbenchmarks explain end-to-end time.
   - If e > 5%, the missing term is found before any attribution is published.
5. **Perturbation of `split` and `mtl4ts` profiling.** It decides whether per-dispatch times are publishable or stay internal.
6. **PSTR on the M5 Max.** Its presence, its cadence, and its agreement with an external meter decide the energy method. If PSTR is absent, an external meter is mandatory.
7. **Fresh vs rewound depth agreement.** It decides whether 32K and 128K decode cells can use rewound depth.
8. **Idle-gap curves (S8) for each engine.** They decide whether a 1.0 s gap is representative, and how much Sky's residency design is worth to users who pause between prompts.
9. **A/A false-alarm rate of the nightly CI.** It sets the regression thresholds.
10. **Whether `mlx_lm.convert` is byte-deterministic and equals `mlx-community`'s files.** It decides which MLX file is the headline (R9 §9, question 2).
11. **An outside party's reproduction within 5%.** It is the G8 exit criterion, and the strongest single piece of evidence the report can offer.

---

## 17. Risks and negative results

**Results the benchmarks will probably show, and that must be shown plainly.**
- **Plain short-context decode will show parity, not a lead.**
  - MLX's whole decode step runs at 94–97% of measured bandwidth, and mlx-lm at 91% on the 8B (R3 §7.F, R4 §7.B).
  - BaseRT, a hand-written Metal 4 runtime, tied MLX on a dense 27B (R7).
  - The suite is built so that a tie is reported as a tie.
- **The quality gate may be hard to pass with a strict reading.** If two BF16 references disagree more than expected, the 0.1× threshold is untestable, and the fallback weakens the claim.

**Risks to the measurement system itself.**
- **Instruments depend on private or undocumented surfaces.**
  - `skycount` swizzles concrete AGX encoder classes.
  - IOReport channel names change per generation; on M5, `PCPU` means the "super" cores [VERIFIED: R9 §6.3].
  - PSTR is unverified on the M5 Max (R9 §9).
  - A macOS update can break any of them. Mitigation: pin the OS build; every instrument has a self-test (a known chain, a null engine).
- **Apple GPUs expose no per-dispatch counters** [VERIFIED: R6 §B1]. `sky profile`'s bytes and FLOPs are modeled from the plan. They are exact for logical bytes but say nothing about actual DRAM traffic, which caching can make smaller or re-reads larger. The IOReport probe (T12) may or may not close that gap.
- **Watchdog.** macOS kills long GPU work while a display is active, and an M5 Max report shows kernel panics under sustained load with the display on [VERIFIED: R6 §A4]. Long prefill cells (128K) and the demo must keep command buffers short. A watchdog kill is recorded as an invalid trial, never silently retried.
- **Competitors move quickly.**
  - On the same M5 Max, llama.cpp's pp512 on LLaMA-2 7B Q4_0 went from 987 tok/s with its November 2023 build to 3,220 tok/s with its August 2026 build, largely through the Metal 4 tensor path (#16634, #20962) [VERIFIED: R5 §9.1, §10].
  - Its maintainers close obvious gaps within weeks once they are pointed out [LIKELY: R5 §9.6].
  - Published numbers therefore age fast. Every result is dated and pinned, and re-run at each gate.

**Costs.**
- **Machine time is large.** Tier H is about two days, Tier A about a week per chip [LIKELY, §4.6]. If the M5 Max is shared with development, the protocol and the work compete for the same machine. A reservation calendar and the nightly perf runner are mitigations; a second M5 Max would be the real fix (owner's decision, D14).
- **Disk.** More than 1 TB for Tier H + A [LIKELY, §8.2].

**Negative results already known**, which the design takes as given:
- One command buffer per kernel is not a profiling method: +14 µs per dispatch (R6), and +60% token time in tinygrad (R2).
- ICB replay does not reduce GPU time per dispatch; it saves CPU only (R6 §B4). llama.cpp's maintainers measured no gain from command reuse (#14570, R5).
- Small-M verification curves from the M2 Pro do not predict the M5 Max, because MLX's `qmv_wide` routing differs by GPU generation (R3 §7.C). Every M3 result is per chip.
- Every common per-process memory metric can be fooled, even by accident (R9 §6.2). Only the system-wide metric is a headline.
- `powermetrics` needs root and undercounts system energy (R9 §6.3). It is not used for energy claims.
