# Part XXII — Performance model

This part defines the analytic model that predicts how long a decode step, a prefill chunk or a speculative round takes on a given Apple GPU. The model is:
- the cost function that the scheduler ([Part IX](part09-scheduler.md)) minimizes;
- the pruner and the sanity check of the autotuner ([Part X](part10-autotuner.md));
- the "predicted" column of `sky explain`;
- the source of every projected number in this report.

It is deliberately small: a roofline per kernel, a fill term per kernel, and a bubble per kernel boundary. Every parameter is either measured (with its source) or marked as an assumption to be replaced at gate G0 (M2 Pro) or G5 (M5 Max).

The reference implementation is [`roofline.py`](roofline.py): `python3 roofline.py --part22` prints every table in this part. The production implementation is Rust (§10). Labels: `[VERIFIED: source]`, `[LIKELY: reason]`, `[EXPERIMENTAL: what decides]`, `[MOONSHOT]`. Research notes are cited as (R6 §B3) and live in [`research/`](research/).

## 1. What the model is for, and what it is not

The model answers three kinds of question:

| Question | Who asks | Example |
|---|---|---|
| Which of two plans is faster, and by how much? | scheduler | fuse RMSNorm into the QKV GEMV, or keep a separate statistics kernel? |
| Is this candidate worth timing, and is this timing believable? | autotuner | a GEMV candidate measured below its DRAM floor has hit the cache |
| Where does a step's time go? | `sky explain`, this report | 386 kernels, 1.27 ms of overhead, 34.0 ms of streaming |

It does not replace measurement. Choices that the model cannot separate (within about 3%) are decided by timing on the device ([Part X](part10-autotuner.md)). Nothing the model projects is shown as measured (SPINE D13).

## 2. The analytic model

### 2.1 One step

A step (one decode token, one verify pass, one prefill chunk) is an ordered list of kernels k = 1…N from the scheduled plan. Its GPU time is

```text
T_gpu = Σ_k [ max( B_k / (η_k · BW_stream),        DRAM bound
                   O_k / R_lane,                    ALU bound (lane instructions)
                   F_k / (φ · F_matrix) )           matrix bound (neural accelerators or MMA)
              + t_fill ]                            per-kernel ramp, tail and latency floor
        + N_dep · t_dep + N_indep · t_indep          kernel-boundary bubbles

T_step = T_gpu + T_cpu,exposed
T_cpu,exposed = max(0, T_cpu − T_gpu)               CPU work pipelined one step ahead (mlx-lm, Sky)
              = T_cpu + t_roundtrip                  CPU in the per-token dependency chain
```

| Symbol | Meaning | Unit |
|---|---|---|
| B_k | DRAM bytes the kernel reads and writes: weights, KV, activations, partials | bytes |
| η_k | fraction of `BW_stream` the kernel class reaches (1.0 for a GEMV; attention and gathers less) | — |
| BW_stream | the rate a well-written 4-bit GEMV streams from DRAM | GB/s |
| O_k | ALU lane instructions: dequantization (`c_deq` per weight), FMAs on the ALU path (one per weight per row), recomputed reductions | instructions |
| R_lane | cores × 128 lanes × clock | instructions/s |
| F_k, F_matrix, φ | FLOPs issued to the matrix units; their peak; the achieved fraction (depends on tile shape) | FLOPs, FLOP/s |
| t_fill | the time a real kernel spends beyond its roofline time: filling the GPU with enough bytes in flight, draining the tail, and for tiny kernels their whole latency-bound execution | µs |
| t_dep, t_indep | the bubble at a dependent kernel boundary (barrier or serial encoder); the cost of an independent dispatch in a concurrent encoder | µs |
| T_cpu, t_roundtrip | CPU time per step (graph build, encoding); the GPU idle gap when the CPU waits for completion before committing | µs |

On chips without neural accelerators (M1–M4) the matrix work runs on the same ALUs as the dequantization, so the model adds `F_k / (2 φ R_lane)` to the ALU term instead of taking a maximum.

### 2.2 Why a fill term is needed

The pure bubble at a dependent boundary, measured with a 32-thread kernel, is 1.14–1.25 µs on the M2 Pro (R6 §B3). Real kernels in a real chain cost more than that beyond their bandwidth time. Four independent measurements agree:

| Source | Measurement | Overhead per kernel |
|---|---|---|
| mlx-lm, Qwen3-8B, M2 Pro (R4 §7.E) | removing 435 small dispatches saves 1.97 ms | 4.5 µs, including their small work |
| llama.cpp, Qwen3-0.6B and 8B, M2 Pro (R5 §8.B, D2) | a two-point fit leaves 64 µs per layer not explained by bytes, over ≈ 20 dispatches and ≈ 15 barriers | ≈ 3 µs per dispatch plus barrier |
| tinygrad ICB replay, M2 Pro (R2 §7.5) | a 64-kernel chain, one barrier per command | 4.3 µs marginal per kernel |
| MLX, three Qwen3-32B-shaped layers, M2 Pro (R3 §7.F) | 7.15 ms against 6.92 ms of streaming at 182 GB/s, 66 kernels | 3.5 µs per kernel |

So the model splits the per-kernel overhead into the measured bubble (`t_dep` = 1.2 µs) and a fill term `t_fill` that every dispatch pays. `t_fill` = 2.3 µs is taken from llama.cpp's data: 64 µs per layer = 20 × t_fill + 15 × 1.2 + 5 × 0.14 gives 2.27 µs. It is therefore out of sample for the mlx-lm validation in §5. `[EXPERIMENTAL: the G0 chain microbenchmark in §4 measures it directly]`

The SPINE's short form "N_dep × 1.2 µs + N_indep × 0.14 µs" is the bubble part only. For dispatch-heavy plans it undercounts by about 3× (§5, and "Disagreement with the spine" at the end).

### 2.3 The ALU budget of a quantized GEMV

A 4-bit affine GEMV spends about `c_deq` = 2 lane instructions per weight to unpack it (one mask on a pre-scaled 16-bit word and one int-to-float conversion, R3 §3.3), then one FMA per weight per activation row. With M rows (M = 1 for decode, 2…32 for verification) a GEMV over W weights issues `(c_deq + M) · W` instructions. It stays bandwidth-bound while that is below the budget

```text
budget = R_lane / (BW_stream / bytes_per_weight)          lane instructions per weight

M2 Pro:  2.863e12 / (185e9 / 0.5625) = 8.7
M5 Max:  8.294e12 / (534e9 / 0.5625) = 8.7     (R_lane = 40 × 128 × 1.62 GHz, LIKELY)
```

`[VERIFIED arithmetic; R_lane for M5 Max LIKELY (R6, R8 §4.3)]`. The two chips have the same budget per weight, so the ALU verify knee sits at the same place on both: `c_deq + M ≤ 8.7`, i.e. M ≤ 6–7 rows before a perfect ALU kernel becomes compute-bound. `roofline.py`'s `verify_free_tokens(alu=True)` gives 8.2–8.7 because it ignores the dequantization. Every recomputed reduction a fused kernel adds (for example the RMSNorm statistics, §9.3 and [Part IX](part09-scheduler.md) §4) spends the same budget.

## 3. Parameters per chip

`roofline.py` holds these values in `GPU_TIMING` and `HARDWARE`.

| Parameter | M2 Pro 16-core | Source / label | M5 Max 40-core | Source / label |
|---|---:|---|---:|---|
| DRAM bandwidth, spec | 200 GB/s | VERIFIED (Apple) | 614 GB/s | VERIFIED (Apple, R6 §A1) |
| Raw streaming read | 196 GB/s | VERIFIED (R6 §B2) | 520–585 planned | LIKELY (R6) |
| BW_stream (4-bit GEMV) | **185 GB/s** | VERIFIED: 183.8–186.3 on 32B shapes, 191.4 on a 378 MB shape (R6 §B2) | **534 GB/s** (set A), 573 (set B, §6) | LIKELY: contributor's 530 GB/s (R3-issues); B fitted |
| Cache-resident re-read (≤ 8 MiB working set) | 400–490 GB/s | VERIFIED (R6 §B2) | unknown | measure |
| R_lane | 2.86 × 10¹² | VERIFIED: 16 × 128 × 1.398 GHz; FMA probe reaches 94% (R3 §7.D) | 8.29 × 10¹² | LIKELY (R6, R8 §4.3) |
| F_matrix (fp16) | 5.73 TFLOPS on the ALUs; MPP `matmul2d` reaches 5.42 (95%) and 4.90 with int4 operands | VERIFIED (R6 §B5) | ≈ 65 TFLOPS on the neural accelerators | LIKELY, two sources (R6 §A2) |
| t_dep | 1.2 µs | VERIFIED: 1.14–1.25 (R6 §B3) | ≈ 1.0 µs | EXPERIMENTAL (R6 assumes; measure) |
| t_indep | 0.14 µs | VERIFIED (R6 §B3) | 0.14 µs | EXPERIMENTAL (carried over) |
| t_fill | 2.3 µs | EXPERIMENTAL (llama.cpp fit, §2.2) | 2.3 µs (set A), 0.5 µs (set C) | EXPERIMENTAL |
| CPU encode, direct | 0.07–0.09 µs per dispatch, +0.15 per explicit barrier | VERIFIED (R6 §B3) | — | measure |
| ICB replay (CPU) | ≈ 4 µs per replay, any length | VERIFIED (R6 §B4) | — | measure |
| t_roundtrip | 104.6 µs (95–135) | VERIFIED (R6 §B3) | — | measure |
| Co-resident threads | 3 × 1024 per core | VERIFIED (R6 §A4) | — | measure |
| Sustained clock derating | — | — | up to ≈ 20% on laptops | EXPERIMENTAL (R6 §M5 Max table) |

The M5 Pro, M5 Ultra and M4 Max entries in `roofline.py` are scaled from these by core count and bandwidth. They are ESTIMATES until measured.

## 4. Calibration procedure

Calibration is one command, `sky hw --calibrate-gpu` (new; task XXII-T2). It runs fixed microbenchmarks and stores the fitted parameters. Every benchmark follows the timing rules of [Part X](part10-autotuner.md) §4:
- GPU timestamps, not wall clock;
- a 2-second warm-up load before timing, so the clock is up;
- working sets of at least 1 GiB rotated between repetitions, so nothing is served from the system-level cache;
- medians of at least 9 repetitions, interleaved with each other;
- the thermal state recorded before and after (R9 §6.4).

| # | Microbenchmark | Gives | Method |
|---|---|---|---|
| C1 | streaming read, 4-byte scalar loads, grid-stride, 1 GiB | raw read bandwidth | R6 `bin/bw` (R6 §B2) |
| C2 | Sky's own GEMV skeleton on three shapes (25600 × 5120, 5120 × 25600, 151936 × 5120), weights rotated | BW_stream | ratio to C1 is η of the GEMV class |
| C3 | 1,000 dependent 1 × 32-thread kernels, serial and with barriers; 1,000 independent ones | t_dep, t_indep | R6 `bin/dispatch` (R6 §B3) |
| C4 | 1,000 dependent *realistic small kernels*: one-threadgroup RMSNorm over 5120 values; a 64 × 256-thread elementwise kernel over 40 KB | t_dep + t_fill | subtract C3's t_dep |
| C5 | one layer's real kernels (K1…K5 of [Part IX](part09-scheduler.md) §3) replayed 64 times, weights rotated | per-kernel fill on real kernels | compare with C2 + C4 |
| C6 | 8 independent FMA chains per thread | R_lane | R3 `r3_bench_fma.py` (R3 §7.D) |
| C7 | MPP `matmul2d` by tile (32×32 … 128×128), dtype (f16, bf16, int4b) and M (16 … 4096) | F_matrix, φ(tile, M) | R6 `bin/gemm` (R6 §B5) |
| C8 | re-read of working sets 1 MiB … 64 MiB | BW_cache and the cache size | R6 §B2 |
| C9 | CPU encode per dispatch, ICB replay, MTL4 command-buffer reuse, commit-to-complete round trip | CPU terms | R6 §B3–§B4 |
| C10 | 10-minute sustained GEMV chain | clock derating | with R9's thermal logger |

**Fitting.**
- `t_dep` and `t_indep` are slopes of time against N (100 → 1,000 dispatches), so launch constants cancel.
- `t_fill` is the C4 slope minus `t_dep`, and C5 checks it on real kernels: the fitted model must predict C5 within 3%, or the fill term is refit per kernel class (GEMV, attention, small).
- `η` per kernel class is measured time ÷ (bytes ÷ BW_stream), from C2 and from the attention kernel at three contexts.

**Storage.** The fitted values are written as records of the existing tuning database, in its data-only text format: kernel `gpumodel.v1`, size class `calib`, integer parameters such as `stream_mbs:534000`, `tdep_ns:1000`, `tfill_ns:2300`, keyed by the GPU fingerprint of [Part X](part10-autotuner.md) §6.1. The format already bounds values to integers ≤ 2⁴⁰ and eight parameters per record, so the model uses two records. An OS update changes the fingerprint and therefore triggers recalibration.

## 5. Validation: predicting mlx-lm on the M2 Pro

The test: predict mlx-lm's measured decode speed from the model, with parameters that did not come from mlx-lm, and report the error.

Inputs:
- **Bytes per token.** Weights from the safetensors headers: 4,257.7 MB for Qwen3-8B, 335.4 MB for Qwen3-0.6B (R4 §7.B). KV read in the timed window: 23.3 MB and 18.6 MB (R4 §7.B, generate_step rows).
- **Plan.** mlx-lm's exact dispatch census (R4 §7.C):
  - 8B: 693 dispatches; 457 barriers plus 23 encoder boundaries = 480 dependent;
  - 0.6B: 541 dispatches; 362 + 11 = 373 dependent.
- **Chip parameters.** §3: BW_stream = 185 GB/s (R6), t_dep = 1.2 µs and t_indep = 0.14 µs (R6), t_fill = 2.3 µs (llama.cpp, §2.2).
- **Attention class efficiency.** MLX's `sdpa_vector` streams K/V at 97 GB/s on 8B (GQA 4) and 189 GB/s on 0.6B (GQA 2). These are the slopes of mlx-lm's context sweep (R4 §7.A), so η = 0.52 and 1.0.
- **CPU.** About 2.2 ms per token for 8B (0.65 ms of Python plus about 693 × 2.2 µs of encode; R3 §7.E, R4 §7.E) and 1.94 ms for 0.6B (R4 §7.E). mlx-lm pipelines one token ahead, so CPU time is exposed only if it exceeds GPU time.

Result (`roofline.py --part22`):

| Model | Streaming (weights + KV) | Fill (N × t_fill) | Bubbles | CPU exposed | Predicted | Measured | Error |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-8B | 23.33 ms | 1.59 ms | 0.61 ms | 0 | **25.53 ms (39.2 tok/s)** | 25.79 ms (38.8), `generate_step` | **−1.0%** |
| Qwen3-8B | | | | | | 26.42 ms (37.9), `stream_generate` | −3.4% |
| Qwen3-0.6B | 1.93 ms | 1.24 ms | 0.47 ms | 0 | **3.64 ms (274.5 tok/s)** | 3.55 ms (281.7), `generate_step` | **+2.6%** |
| Qwen3-0.6B | | | | | | 3.76 ms (266.0), `stream_generate` | −3.1% |

Two further checks isolate the terms:

| Check | Predicted | Measured (R4 §7.E) | Error |
|---|---:|---:|---:|
| 8B, GEMV-only chain (258 dispatches, 132 barriers) | 23.78 ms | 23.93 ms | −0.6% |
| 0.6B, GEMV-only chain (202 dispatches, 113 barriers) | 2.42 ms | 2.41 ms | +0.4% |
| 8B, full step **without** t_fill (bubble-only model) | 23.93 ms | 25.79 ms | **−7.2%** |
| 0.6B, full step **without** t_fill | 2.40 ms | 3.55 ms | **−32.4%** |

What this says:
- **The model predicts mlx-lm within −3.4% … +2.6%** on both a bandwidth-dominated model and a dispatch-dominated one. `[VERIFIED arithmetic on VERIFIED inputs; t_fill EXPERIMENTAL]`
- **The fill term is not optional.** A bubble-only model misses a third of the 0.6B step. That is the regime of draft models.
- **The CPU term is zero for both**, consistent with R4's finding that pipelining hides 1.94 ms of CPU behind 3.5 ms of GPU on the 0.6B. On a GPU three times faster the CPU would be exposed; §7 returns to this.
- The remaining error is within run-to-run spread: the five 8B trials span 36.1–37.9 tok/s (R4 §7.A).

Sky's own plan on the same chip, predicted by the same model, for reference: Qwen3-8B 42.0 tok/s at context 160 (1.08× mlx-lm's 38.8) and 36.3 tok/s at 4K (1.15× mlx-lm's 31.7, R4 §7.A); Qwen3-0.6B 412 tok/s (1.47×). `[EXPERIMENTAL: G4 measures them]`

## 6. Worked example: one Qwen3-32B decode step on the M5 Max

### 6.1 Sky's plan, kernel by kernel

The plan is [Part IX](part09-scheduler.md) §3: five fused kernels per layer, a deterministic split-combine kernel when the context spans more than one split (splits of 256 positions), and two head kernels. Parameter set A (534 GB/s, t_dep 1.0 µs, t_fill 2.3 µs), context 512, fp16 KV, MLX 4-bit group-64 weights:

| Kernel | Per token | MB each | MFLOP each | Bound | Time each | Per token, with boundary |
|---|---:|---:|---:|---|---:|---:|
| K1 RMSNorm + QKV GEMV + q/k-norm + RoPE + KV append | 64 | 29.54 | 104.9 | DRAM | 57.6 µs | 3.75 ms |
| K2 GQA attention, fixed-size splits | 64 | 2.18 | 16.9 | DRAM | 6.8 µs | 0.50 ms |
| K2c deterministic split combine | 64 | 0.08 | 0.1 | latency | 2.5 µs | 0.22 ms |
| K3 o_proj GEMV + residual | 64 | 23.65 | 83.9 | DRAM | 46.6 µs | 3.05 ms |
| K4 RMSNorm + gate/up GEMV + SwiGLU | 64 | 147.54 | 524.4 | DRAM | 278.6 µs | 17.89 ms |
| K5 down GEMV + residual | 64 | 73.82 | 262.1 | DRAM | 140.5 µs | 9.06 ms |
| H1 final norm + lm_head GEMV + partial argmax | 1 | 437.60 | 1,555.8 | DRAM | 821.8 µs | 0.82 ms |
| H2 sample combine: token, position, stop flag | 1 | 0.07 | — | latency | 2.4 µs | 0.003 ms |
| **Total** | **386** | | | | | **35.30 ms → 28.3 tok/s** |

Where the 35.30 ms go:
- 34.03 ms streaming 18.13 GB;
- 0.89 ms of fill (386 × 2.3 µs);
- 0.39 ms of bubbles (386 × 1.0 µs).

The step reaches 96% of the 29.5 tok/s ceiling at this stream rate.

Every GEMV is DRAM-bound at M = 1. The fused prologues and epilogues (norm statistics recomputed in every simdgroup, q/k-norm, RoPE, residual, SwiGLU) use under 40% of the ALU budget: K4, the largest, issues 852 M lane instructions against 2.31 G available in its 278 µs.

### 6.2 The MLX-like plan

The same model applied to mlx-lm's plan:
- 19 dispatches per layer plus 9 per token = 1,225 per token (R4 §3.1, R3 §7.F);
- 848 of them dependent, using the 8B ratio of 480/693;
- MLX's attention class efficiency of 0.52, which is the M2 Pro value (R4).

Its overhead is 3.72 ms per token: 2.82 ms of fill and 0.90 ms of bubbles. Sky's is 1.27 ms. The difference, 2.45 ms, is 6.5% of the step. Most of it is the 14 extra latency-bound kernels per layer.

### 6.3 Predicted tok/s against the ceiling, three parameter sets

The M5 Max values of t_fill and BW_stream are unmeasured, so the table brackets them:
- **A (planning):** BW_stream = 534 GB/s, t_fill = 2.3 µs.
- **B (fitted):** the stream rate at which the MLX-like plan reproduces the published 28.4 tok/s (R5 §9.3, one community run). It comes out at 573 GB/s, 93% of spec.
- **C (low overhead):** 534 GB/s with t_fill = 0.5 µs, as if the M5 Max had much smaller fill costs than the M2 Pro.

| Set | Context | Ceiling at BW_stream | Sky: dispatches, step | Sky tok/s | MLX-like: dispatches, step | MLX-like tok/s | Sky ÷ MLX-like |
|---|---:|---:|---|---:|---|---:|---:|
| A | 512 | 29.5 | 386, 35.30 ms | **28.3** | 1,225, 37.95 ms | 26.4 | **1.075** |
| A | 8,192 | 26.5 | 386, 39.74 ms | 25.2 | 1,225, 45.20 ms | 22.1 | 1.137 |
| A | 32,768 | 20.1 | 386, 53.95 ms | 18.5 | 1,225, 68.40 ms | 14.6 | 1.268 |
| B | 512 | 31.6 | 386, 32.97 ms | **30.3** | 1,225, 35.61 ms | 28.1 | **1.080** |
| B | 8,192 | 28.5 | 386, 37.11 ms | 26.9 | 1,225, 42.36 ms | 23.6 | 1.142 |
| B | 32,768 | 21.6 | 386, 50.35 ms | 19.9 | 1,225, 63.97 ms | 15.6 | 1.271 |
| C | 512 | 29.5 | 386, 34.60 ms | 28.9 | 1,225, 35.74 ms | 28.0 | **1.033** |
| C | 8,192 | 26.5 | 386, 39.05 ms | 25.6 | 1,225, 42.99 ms | 23.3 | 1.101 |
| C | 32,768 | 20.1 | 386, 53.26 ms | 18.8 | 1,225, 66.19 ms | 15.1 | 1.243 |

`[EXPERIMENTAL: every M5 Max number here is a model output; G5 measures them]`

Readings:
1. **Short context: Sky ÷ MLX-like = 1.03–1.08×.** This clears the SPINE's M5 Max target of ≥ 1.0× and brackets the ≥ 1.05× that applies once a raw read of ≈ 575 GB/s or more is measured. Where inside the bracket the result falls depends on the M5 Max's fill cost, which nobody has measured. Set B, the one consistent with MLX's published number, predicts 30.3 tok/s for Sky, inside the SPINE's "~30–31".
2. **The ratio is more robust than the absolute numbers.** Raising the stream rate from 534 to 573 GB/s moves both engines by 7%, but their ratio by 0.5%.
3. **Long context: 1.10–1.14× at 8K and 1.24–1.27× at 32K.** The gain comes from attention: at 32K the KV stream is 138 MB per layer. The ratio assumes MLX's attention stays at the M2 Pro efficiency (0.52) and Sky's GQA-aware kernel reaches 0.90. MLX switches to a GQA-deduplicating 2-pass kernel at ≥ 8K for GQA-8 models, which reads K/V twice instead of eight times (R3-kernels §c), so 0.52 may be pessimistic for MLX there. These ratios exceed [Part XV](part15-attention.md) §8.2's bounds (≈ 1.08× at 8K, ≈ 1.24× at 32K) because they compare whole plans: they include reading 1's dispatch and fill savings on top of attention, while Part XV's bound is attention alone with weight streaming and fixed costs at parity. `[EXPERIMENTAL: the G5 context sweep decides]`
4. **Sky's own long-context cost is the combine traffic.** At 32K, 128 splits × 64 heads of fp32 partials are 4.3 MB per layer, 3% of the KV bytes. A larger split size halves it, at the cost of parallelism at short context ([Part IX](part09-scheduler.md) §10).

## 7. Small and draft models

The same model, GPU side only, context 256:

| Chip | Model | Sky plan (tok/s) | MLX-like plan (tok/s) | Ratio | Roofline ceiling |
|---|---|---:|---:|---:|---:|
| M2 Pro | Qwen3-0.6B | 401 | 270 | 1.48 | 538 |
| M2 Pro | Qwen3-1.7B | 169 | 140 | 1.21 | 197 |
| M2 Pro | Qwen3-8B | 42 | 39 | 1.07 | 46 |
| M5 Max | Qwen3-0.6B | 862 | 429 | 2.01 | 1,465 |
| M5 Max | Qwen3-1.7B | 426 | 284 | 1.50 | 536 |
| M5 Max | Qwen3-8B | 115 | 98 | 1.18 | 124 |

Readings:
- **The 0.6B is a dispatch problem.** On the M5 Max, Sky's plan spends 0.69 ms streaming weights and KV and 0.47 ms on fill and bubbles (142 dispatches); the MLX-like plan spends 1.64 ms on fill and bubbles (541 dispatches).
- **mlx-lm's CPU may bind first on the M5 Max.** Its CPU cost on the M2 Pro is 1.94 ms per 0.6B token (R4 §7.E), more than the 1.16 ms the model predicts for Sky's whole GPU step on the M5 Max. Sky's static plan costs ≈ 12 µs of CPU per 0.6B step with direct re-encoding, the default, or ≈ 4 µs with an ICB (R6 §B3–§B4; [Part XX](part20-persistent-execution.md) §7).
- **The SPINE target of ≥ 1.5× on 0.6B–1.7B is met by the model on the M5 Max (1.50–2.01×), but not for the 1.7B on the M2 Pro (1.21×).** There the larger GEMVs dominate. `[EXPERIMENTAL: G4 measures 0.6B on the M2 Pro]`
- **For speculation this sets the draft cost.** At context 1024 a 0.6B draft token costs 1.44 ms in Sky's plan, c = 0.040 of the 35.6 ms Qwen3-32B target step, against 2.50 ms (c = 0.070) for the MLX-like plan (§9).

## 8. The prefill model

### 8.1 Form

A prompt of P tokens runs as chunks of T tokens (buckets in [Part IX](part09-scheduler.md) §10). Each chunk streams every weight once and does `2 · T` FLOPs per weight, plus causal attention against the `ctx0` positions already cached:

```text
T_chunk = Σ_layers Σ_k [ max( B_k / BW_stream,
                              F_k / (φ · F_matrix),
                              c_deq · W_k · ⌈T / BM⌉ / R_lane )  + t_fill ] + N · t_dep
F_attention = 4 · n_heads · head_dim · T · (ctx0 + (T + 1) / 2)
```

Three facts set the shape:
- **A chunk is compute-bound only above the ridge.** That is `T* = (F_matrix / BW_stream) · b / 16` tokens, the same expression as the free verify width. It is 34 tokens on the M5 Max neural accelerators and 8.7 on the M2 Pro ALUs. Below T*, a chunk streams 18 GB of weights for little arithmetic. That is why TTFT for short prompts is dominated by weight streaming plus overhead, and why BaseRT's 1.44× at pp128 shrinks to 1.02× at pp2048 (R7 §0.2).
- **Weights are dequantized once per M-tile.** MPP has no native group-scale format (R6 §A3), so each `BM × BK` weight tile is dequantized into threadgroup memory per output row-block. That is `c_deq · W · ⌈T/BM⌉` ALU instructions per chunk. On the M5 Max this runs on the ALUs while the matrix work runs on the neural accelerators. Whether the two overlap is unmeasured, so the model gives both bounds. On M1–M4 both run on the ALUs and add.
- **φ depends on the tile.** R6 measured MPP at 95% of peak with 32×32 and 64×32 tiles on the M2 Pro (5.42 TFLOPS) and found Apple's 64×32 sample tile at about half of peak on the M5 Max (R6 §A2). MLX's quantized `matmul2d` path reached 35.0 TFLOPS (0.54) and its fp16 GEMM 46.7 (0.72) on the M5 Max `[E: MLX #4411, R6]`.

### 8.2 Predictions

Chunk 512; overlap = dequantization overlaps the matrix work:

| Model, chip | Prompt | φ | Overlap | Predicted tok/s | Ceiling at 100% | Measured baseline |
|---|---:|---:|---|---:|---:|---|
| Qwen3-32B, M5 Max | 128 | 0.71 | yes | 729 | 1,039 | — |
| | 512 | 0.71 | yes | 731 | 1,033 | **MLX 731** (R5 §9.3, LIKELY) |
| | 512 | 0.85 | yes | **875** | 1,033 | — |
| | 512 | 0.85 | no | 793 | 1,033 | — |
| | 2,048 | 0.85 | yes / no | 854 / 776 | 1,007 | — |
| | 8,192 | 0.85 | yes / no | 777 / 712 | 915 | — |
| Qwen3-8B, M2 Pro | 512 | 0.85 | — | 337 | 408 | mlx-lm 220 (R4 §7.A); llama.cpp 312 (R5 §8.A) |
| | 512 | 0.95 | — | 376 | 408 | — |
| | 2,048 | 0.85 / 0.95 | — | 327 / 365 | 395 | mlx-lm 218 |

Readings:
- **φ = 0.71 reproduces MLX's published 731 tok/s.** This is a consistency check, not a validation: Part IV derived 0.71 from that same number.
- **On the M5 Max, the model puts Sky at 1.08–1.20× MLX** at pp512 if Sky's `matmul2d` kernels reach φ = 0.85. The two values bracket whether dequantization overlaps the matrix work. The SPINE target is ≥ 1.0×. `[EXPERIMENTAL: C7 and a G5 prefill run decide φ and the overlap]`
- **On the M2 Pro, the model puts Sky at 1.55–1.72× mlx-lm**, consistent with llama.cpp already reaching 1.42× mlx-lm on the same chip (R5). The SPINE target is ≥ 1.25× mlx-lm and ≥ 1.0× llama.cpp.
- **The int8 path is a different model.** With W4A8 the matrix term uses the int8 rate (≈ 122 TOPS on the M5 Max, R6 §A2, one source). The dequantization term then becomes the bound unless weights are repacked to int8 tiles. It stays a separately labeled mode (SPINE §3).

## 9. The speculative model

### 9.1 Form

One speculative round drafts γ tokens and verifies them in one target pass of width M = γ + 1 (for trees, M = number of nodes). Following R8 §2.1:

```text
S = τ / (1 + γ·c + γ·β + δ)

τ  = tokens emitted per round (measured, or (1 − α^(γ+1)) / (1 − α) for i.i.d. acceptance α)
c  = time of one draft token ÷ T₁        (T₁ = one plain target decode step)
β  = (T_verify(M) / T₁ − 1) / (M − 1)    marginal cost of one more verified row
δ  = per-round fixed cost ÷ T₁           (accept/reject, rewind, any CPU synchronization)
```

Leviathan et al.'s Theorem 3.8 is the special case β = δ = 0 (R8 §2.1). For trees, γ·β becomes (nodes − 1)·β and γ·c becomes depth·c.

### 9.2 β from measured verify-cost curves

| Engine, chip, model | T(M) ÷ T(1) | β | Source |
|---|---|---|---|
| mlx-lm, M2 Pro, Qwen3-8B forward | 1.73 (M=2), 3.33 (4), 6.48 (8) | 0.73–0.78 | R4 §7.D, VERIFIED |
| llama.cpp, M2 Pro, Qwen3-8B Q4_0 | 1.52 (2), 2.08 (4), 3.76 (8) | 0.36–0.52 | R5 §8.G, VERIFIED |
| MLX `qmv_wide`, M5 Max, 70B forward | 1.12 (3), 1.48 (5), 2.25 (8), 3.90 (16) | 0.06–0.19 | mlx#3791, contributor data (R3-issues E4), LIKELY |
| R3's 60-line multi-row GEMV, M2 Pro, 32B gate/up | 1.08 (2), 1.30 (3), 2.24 (4, register spill) | 0.08–0.41 | R3 §7.G, VERIFIED |
| **Sky model, ALU path, 32B, context 1024** | 1.00 (2), 1.01 (4), **1.16 (8)**, 2.03 (16), 3.79 (32) | 0.002–0.09 | `roofline.py --part22` |
| **Sky model, neural-accelerator path, M5 Max** | 1.01 (8), 1.03 (16), 1.19 (32) | 0.002–0.006 | same |

The model's β is a lower bound. It assumes a verify GEMV that keeps M activation rows in registers without spilling. R3's simple kernel spilled at M = 4, and its time jumped from 1.30× to 2.24× (R3 §7.G). Two model details matter:
- **At M = 8 the ALU path is compute-bound.** `(c_deq + 8) = 10 > 8.7` lane instructions per weight (§2.3), so even a perfect kernel costs 1.15× at M = 8.
- **Recomputing the RMSNorm statistics per simdgroup would push M = 8 to 1.28×.** The model's 1.16× assumes they are not recomputed per simdgroup: either a stored-statistics kernel (what `roofline.py` models) or [Part XVI](part16-decode.md) §3.5's producer partials, which cost about the same ([Part IX](part09-scheduler.md) §4.2). That choice is what keeps the model inside the SPINE target of ≤ 1.3×.

### 9.3 Speedup for Qwen3-32B on the M5 Max

Base T₁ = Sky's plan at context 1024, set A: 28.1 tok/s. c = 0.03 for EAGLE-3 (assumed, as in R8 and Part IV). δ ≈ 0: Sky's accept/reject and rewind are a few GPU kernels (about 10 µs, under 0.1% of T₁) with no CPU synchronization ([Part XXI](part21-speculative.md)). τ values are the published ones for `RedHatAI/Qwen3-32B-speculator.eagle3` (R8 Appendix B). Cells: β, then speedup S (effective tok/s).

| Drafter (τ) | γ | Sky model | Sky target β = 0.05 | MLX M5 Max curve | llama.cpp M2 Pro curve | mlx-lm M2 Pro curve |
|---|---:|---|---|---|---|---|
| EAGLE-3 HumanEval (2.29) | 3 | 0.002: 2.09× (59) | 1.85× (52) | 0.100: 1.65× (46) | 0.360: 1.06× (30) | 0.777: 0.67× (19) |
| EAGLE-3 GSM8K (2.49) | 3 | 0.002: 2.27× (64) | 2.01× (56) | 0.100: 1.79× (50) | 0.360: 1.15× (32) | 0.777: 0.73× (20) |
| EAGLE-3 GSM8K (3.08) | 7 | 0.022: 2.25× (63) | 1.97× (55) | 0.179: 1.25× (35) | 0.394: 0.78× (22) | 0.783: 0.46× (13) |
| DFlash-class (4.4; none public for 32B) | 8 | 0.002: 3.50× (98) | 2.68× (75) | 0.156: 1.77× (50) | 0.345: 1.10× (31) | 0.685: 0.65× (18) |

`[EXPERIMENTAL: τ is published and VERIFIED; every speedup is a model output]`

Readings:
1. **With Sky's target β ≤ 0.05, public EAGLE-3 drafters give 52–56 tok/s, 1.85–2.01×.** That is inside the SPINE's "47–59 tok/s typical". The idealized model reaches 59–64.
2. **With MLX's own M5 Max verify curve, the same drafter gives 1.25–1.79×.** This brackets mlx-lm's 1.82× on an M3 Max (R5 §9.4), which used a draft model rather than EAGLE-3. Long chains (γ = 7) lose most: β grows with M on that curve.
3. **With M2-class verify curves, speculation loses or barely wins.** That reproduces R4's 0.37–0.74× for mlx-lm and R5's 1.12× for llama.cpp in sign and size.
4. **Speculation multiplies the draft's per-token cost by γ.** With a 0.6B draft model instead of EAGLE-3, c = 0.040 in Sky's plan (§7) against 0.070 for an MLX-like draft. At γ = 3 that is the difference between 1 + 0.12 and 1 + 0.21 in the denominator.

## 10. How the model feeds the rest of the system

### 10.1 Rust types

| Item | Crate / file | Role |
|---|---|---|
| `GpuFacts` | `sky-hw/src/gpu.rs` (new) | family, cores, neural accelerators, GPU fingerprint; calibrated parameters from §4 |
| `KernelCost { dram_bytes, cache_bytes, lane_ops, matrix_flops, class }` | `sky-kernel/src/cost.rs` | exact counts per kernel instance, as `Poly` in the bucket parameters (context, chunk, M), computed from the kernel IR |
| `Prediction { time_ns, bound: Bound, fill_ns, bubble_ns }`, `enum Bound { Dram, Cache, Alu, Matrix, Latency }` | `sky-kernel/src/cost.rs` | the result of `predict(&KernelCost, &GpuFacts)` |
| `PlanPrediction` | `sky-kernel/src/schedule/predict.rs` | `predict_plan(&ScheduledPlan, &GpuFacts)`: per kernel, totals, and the step formula of §2.1 |
| `SpecPrediction` | `sky-llm/src/spec/model.rs` | §9's S, with β from `predict_plan` at widths 1…32 or from measured curves |

`roofline.py` is the executable specification. A test (`sky-kernel/tests/model_matches_roofline.rs`) checks that the Rust implementation reproduces the `--part22` tables to the printed precision.

### 10.2 Consumers

- **Scheduler ([Part IX](part09-scheduler.md)).** Every fusion and recompute-versus-store decision compares `predict_plan` of two candidate partitions. A decision is taken by the model only if the predicted difference exceeds a margin (default 3%). Otherwise both candidates go to the block-level tuner.
- **Autotuner ([Part X](part10-autotuner.md)).** Three uses:
  - *Pruning.* A candidate whose predicted time exceeds 1.25× the best prediction is not compiled.
  - *Ordering.* Candidates are timed best-predicted first, so a time budget spends itself on likely winners.
  - *Sanity.* A measured time more than 5% below the DRAM floor `B_k / BW_raw` is impossible for a streaming kernel, so the measurement is rejected as cache-contaminated. This is exactly the failure that made tinygrad's BEAM pick a 540 µs kernel it had timed at 206 µs (R2 §6.4).
- **`sky explain`.** For every kernel it prints the predicted time, the bound, and, under `sky profile`, the measured GPU time and the residual. A residual above 15% flags either a model error (refit) or a kernel defect (a register spill, a bad tile).
- **Cost contracts.** Bytes and dispatch counts are exact properties of a plan, so `sky check` proves contracts such as `cost bytes_per_token <= 18.2 GB` or `cost dispatches_per_token <= 400` statically. That extends the existing polynomial cost contracts (R1 §5.8). Time is never a contract: the spec deliberately excludes run-time time budgets (R1 §5.8), and the model is a prediction, not a proof.

## 11. Implementation tasks

| ID | Task | Crate / files | Deliverable | Acceptance test |
|---|---|---|---|---|
| XXII-T1 | Per-kernel cost from kernel IR | `sky-kernel/src/cost.rs` | `KernelCost` for every skeleton and generated kernel, as `Poly` in bucket parameters | Byte counts equal `roofline.py --part22`'s Part IX tables for Qwen3-8B and 32B; DRAM bytes of a decode plan equal the weights plus KV within 0.1% |
| XXII-T2 | GPU calibration | `sky-hw/src/gpu.rs`, `sky-cli` (`sky hw --calibrate-gpu`), microbenchmarks in `sky-metal/benches/calib/` | C1–C10 of §4; results stored as `gpumodel.v1` records in the tuning DB | On the M2 Pro, t_dep within 10% of R6 (1.14–1.25 µs) and BW_stream within 3% of R6's GEMV (183.8–186.3 GB/s); repeated runs agree within 3% |
| XXII-T3 | Plan predictor | `sky-kernel/src/schedule/predict.rs` | `predict_plan` implementing §2.1 | `sky-kernel/tests/model_matches_roofline.rs`: reproduces the §5 and §6 tables |
| XXII-T4 | Validation harness | `sky bench llm --predict` | For any engine with a dispatch census, predicts and reports error | Reproduces §5 within 1 percentage point; reruns on the M5 Max at G5 |
| XXII-T5 | Explain and profile integration | `sky-cli` (`explain`, `profile`) | Per-kernel predicted vs measured, bound, residual | Every kernel of a Qwen3-8B decode step appears with a residual; a deliberately mistuned GEMV shows a residual above 15% |
| XXII-T6 | Static cost contracts on plans | `sky-cli/src/cost.rs` | `bytes_per_token`, `dispatches_per_token`, `peak_memory` contracts on compiled models | `sky check` proves `bytes_per_token <= 18.2 GB` for Qwen3-32B 4-bit and refutes 17.9 GB with a witness |
| XXII-T7 | Speculative model | `sky-llm/src/spec/model.rs` | §9's S with β from predicted or measured curves; picks γ | On recorded acceptance traces, the chosen γ is within one step of the measured best |

## 12. Measurements that decide

- **t_fill and t_dep on the M5 Max (C3, C4).** They decide whether the short-context gain is 1.03× (set C) or 1.08× (sets A, B).
- **BW_stream on the M5 Max (C2).** It sets the ceiling. It is the only way to tell whether MLX's 28.4 tok/s is at 96% (534 GB/s) or 90% (573 GB/s) of what a GEMV can stream.
- **MLX's attention efficiency at 8K and 32K on the M5 Max.** It decides the long-context gain (1.10–1.27×).
- **The verify-cost curve of Sky's verify GEMV at M = 2…32** on both chips. It replaces the model's idealized β.
- **φ and dequantization overlap for `matmul2d` on the M5 Max (C7).** They decide prefill between 1.08× and 1.20× of MLX.
- **The model's residuals on the full Qwen3-8B step on the M2 Pro at G4.** If they exceed 5%, add a term before any M5 Max projection is quoted.

## 13. Risks and negative results

- **The model is calibrated on one GPU generation.** Every M5 Max parameter in sets A–C is an assumption or a fit to one community measurement. The Apple9/10 "dynamic caching" register allocation (R6 §A4) may change both t_fill and occupancy.
- **t_fill is a single number for very different kernels.** It fits the M2 Pro within ±3.4%, but one constant cannot be right for a 2 µs RMSNorm and a 280 µs GEMV. XXII-T2 fits it per kernel class if C5 demands.
- **The fill term flatters fusion.** Fusion removes dispatches, and the model charges each dispatch 3.3–3.5 µs. If the M5 Max's real cost is near the pure bubble (set C), fusion is worth 1.03×, not 1.08×. That is the honest bracket, and it is why speculation, not fusion, carries the decode targets (SPINE D18).
- **The verify β is idealized.** Register pressure, not bandwidth, decided R3's experiment. The model's 1.16× at M = 8 assumes a kernel nobody has written yet.
- **Thermal derating is outside the model.** Sustained laptop runs can lose about 20% (R6). The model predicts the cool-chip rate; benchmarks report sustained rates ([Part XXIII](part23-benchmarks.md)).
- **Negative result:** a bubble-only model, which the SPINE's short form suggests, misses 32% of a 0.6B decode step and 7% of an 8B step on the M2 Pro (§5).

## Disagreement with the spine

1. **The per-kernel overhead is about 3.5 µs, not 1.2 µs.** SPINE §2 and Part IV §4 price a dependent boundary at ≈ 1.2 µs. On that basis they estimate mlx-lm's 1,225 kernels at ≈ 1.5 ms per 32B token and Sky's plan at ≈ 0.4–0.5 ms.
   - The 1.2 µs is the bubble of an empty kernel. Real kernels in a chain cost about 3.5 µs each beyond their bandwidth time, on four independent measurements (§2.2). A model without that term misses 32% of a 0.6B step (§5).
   - With it, the overhead is ≈ 3.7 ms per token for the MLX-like plan and ≈ 1.3 ms for Sky's. [Part XVI](part16-decode.md) §2 and §4 still use the bubble-only figure (0.26 ms for 8B, ≈ 0.8 ms for 32B).
   - The SPINE's conclusions survive: about 1.05× at short context, larger for small models. The arithmetic behind them should use the fill term, and the dispatch-reduction estimate in Part IV §4 ("≤ 3–4%") becomes ≈ 3–7%.

   Resolved: the spine now says a dependent boundary costs ≈ 1.2 µs of idle GPU time and, with the ≈ 2.3 µs fill fitted here, ≈ 3.5 µs per dependent kernel (SPINE §2). [Part XVI](part16-decode.md) §4 now quotes the ≈ 3–7% figure next to its bubble-only estimate.
2. **The verify target holds only if the norm statistics are not recomputed per simdgroup at M ≥ 6.** SPINE D18 requires ≤ 1.3× of M = 1 at M ≈ 8. On the ALU path, the model says a perfect kernel already costs 1.15× there, because M = 8 is past the ALU budget. Recomputing the RMSNorm statistics per simdgroup, which is right at M = 1, would make it 1.28×, at the edge of the target. So the statistics must come from producer partials ([Part XVI](part16-decode.md) §3.5) or a stored-statistics kernel in the verify plans ([Part IX](part09-scheduler.md) §4 and §8). This is a refinement of D18, not a reversal.
