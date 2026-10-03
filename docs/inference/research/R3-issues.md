# R3-issues — Known MLX performance issues (Q6 of R3), primary sources

Compiled 2026-10-01 by an R3 sub-agent (fork). Scope: GitHub issues/PRs/discussions in
ml-explore/mlx, ml-explore/mlx-lm, ml-explore/mlx-examples (via `gh`), plus a few primary
web sources (Apple newsroom, Apple ML Research, independent microbenchmarks). No benchmarks
were run for this file. Raw `gh` dumps were produced with
`$S/research/r3_issues_raw/dump.py` and `maint.py` (same directory).

Labels: **VERIFIED** = link to a primary statement/measurement (I quote it; "VERIFIED" means
the statement exists as quoted, not that a third-party number is reproducible).
**LIKELY** = derived by me from verified numbers or a credible non-maintainer measurement.
**EXPERIMENTAL** / **MOONSHOT** = untested ideas. Maintainers referenced: awni (Awni Hannun),
angeloskath (Angelos Katharopoulos), jagrit06 (Jagrit Digani), zcbenz (Cheng Zhao) — all
`MEMBER` on ml-explore/mlx.

MLX release map (from `gh release view`): v0.30.0 2025-11-19 (NAX), v0.30.4 2026-01-27,
v0.31.1 2026-03-12, v0.31.2 2026-04-22, v0.32.0 2026-07-07, v0.32.1 2026-08-18,
v0.32.2 2026-08-25, v0.32.3 2026-09-29 (the R3 venv runs 0.32.3; the clone is at
0840c42, 0.32.4-dev).

---

## A. Hardware context (Apple primary sources)

A1. **Apple newsroom, 2026-03-03** (VERIFIED,
https://www.apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/):
"M5 Max supports up to 128GB of unified memory with higher unified memory bandwidth up to
614GB/s"; M5 Pro "up to 307GB/s"; "up-to-40-core GPU"; "a Neural Accelerator in each GPU
core"; "over 4x the peak GPU compute for AI compared to the previous generation."

A2. **Apple ML Research, "Exploring LLMs with MLX and the Neural Accelerators in the M5
GPU", 2025-11-19** (VERIFIED, https://machinelearning.apple.com/research/exploring-llms-mlx-m5;
mirrored as mlx discussion #2816): base M5 (24 GB) vs M4: TTFT speedups 3.33x–4.06x
(Qwen3-8B-4bit 3.97x, Qwen3-14B-4bit 4.06x); "Generating subsequent tokens is bounded by
memory bandwidth, rather than by compute ability ... the M5 provides 19-27% performance boost
compared to the M4, thanks to its greater memory bandwidth (120GB/s for the M4, 153GB/s for
the M5, which is 28% higher)." "MLX requires macOS 26.2 or later" to use the Neural
Accelerators. Significance: Apple itself says decode gains on M5 come only from bandwidth;
NAX is a prefill/TTFT lever.

A3. **NAX gating in code** (VERIFIED, `mlx/backend/metal/device.cpp:952-969` @0840c42):
`is_nax_available()` returns true only if `__builtin_available(macOS 26.2, ...)` (l.958) and
`gen >= (arch == 'p' ? 18 : 17)` (l.965); compile-time opt-out `MLX_METAL_NO_NAX`. Kernels
include `<MetalPerformancePrimitives/MetalPerformancePrimitives.h>`
(`kernels/steel/gemm/nax.h:12`, `kernels/steel/attn/nax.h:12`,
`kernels/gated_delta_nax_ops.h:6`). No `MTL4*` command-queue or indirect-command-buffer use
anywhere in `mlx/` (grep, VERIFIED) — MLX uses Metal 4 only for in-shader tensor ops.

A4. **Independent NA microbenchmarks** (VERIFIED as statements; numbers are third-party):
- tzakharko, 2025-10-15 (A19, https://tzakharko.github.io/apple-neural-accelerators-benchmark/):
  FP16 ≈1024 FLOP/cycle/core, INT8 ≈2048 OP/cycle/core; "The only supported way to access it
  is by using the high-level Metal Performance Primitives (MPP) framework in combination with
  the new Metal Tensor APIs"; best tile ≈32×32; operands must be in fast on-chip storage; no
  BF16 observed.
- Draw Things, 2026-04-01 (https://releases.drawthings.ai/p/metal-quantized-attention-pulling):
  M5 Max FP16 "about 60 TFLOPs", INT8 "around 110 TFLOPs"; INT8 matmul "1.61× to 1.87× faster
  than our FP16 baseline".
- Measured M5 Max (40-core) peak bandwidth "530 GB/s" (erwinzhang7, contributor, in
  mlx#3837, 2026-08-05) ≈ 86% of the 614 GB/s spec (LIKELY; method not shown).

---

## B. Neural Accelerators (NAX) in MLX — timeline, claims, gaps

B1. mlx#2693 (2025-10-22, closed) — awni: "Neural accelerator support is a work in progress.
We will be evaluating additional Metal 4 features on a case-by-case basis and include what
makes sense. FYI AMX != Neural Accelerators." https://github.com/ml-explore/mlx/issues/2693
(VERIFIED)

B2. mlx#2687 (closed 2025-12-04) — external "[Experiment] Use metal performance primitives"
(`mpp::tensor_ops` in quantized + SDPA kernels); awni: "Closing this given #2772".
https://github.com/ml-explore/mlx/pull/2687 (VERIFIED)

B3. **mlx#2772 "Add Neural Accelerator Support"** (jagrit06, merged 2025-11-19, shipped in
v0.30.0 whose highlight reads "Support for Neural Accelerators on M5 (macOS >= 26.2)").
Files: `quantized_nax.*`, `fp_quantized_nax.*`, `steel_attention_nax.*`,
`steel_gemm_fused_nax.*`, `steel_gemm_gather_nax.*`. Follow-ups: #2811 centralize NAX
condition (awni), #2916 JIT NAX (v0.30.1), #3018 NAX split-K (v0.30.4), #3211 M5 Pro/Max
tunings (v0.31.1), #3271 refactor, #3422 split-K routing fix (v0.31.2), #3622 build target
26.2 (v0.32.0), #4447 "M5 ultra tunings" (v0.32.3). https://github.com/ml-explore/mlx/pull/2772
(VERIFIED)

B4. **M5 Max dense GEMM throughput** — mlx#3211, jagrit06 (2026-03-10): fp16 4096³ nn 58.2
TFLOPS; 4096×14336×4096 59.6; 4096×4096×14336 54.9/52.7 (nn/nt). "Split-K [matmuls] need
separate work". https://github.com/ml-explore/mlx/pull/3211 (VERIFIED). Combined with A4
(FP16 NA ≈60 TFLOPS on M5 Max) ⇒ **MLX large fp16 GEMM is at or near the FP16 NA ceiling on
M5 Max** (LIKELY). Prefill headroom therefore lies in (a) shapes MLX tiles badly, (b) INT8,
(c) non-GEMM time — not in large-GEMM kernel quality.

B5. mlx#3422 (angeloskath, merged 2026-04-20): "Split-K matmuls were being routed to the non
NAX version"; "AddMM was completely broken on NAX"; M5 Max chained-matmul speedups up to 2.7x
on mid shapes (512×512…1024×512), none on large shapes. https://github.com/ml-explore/mlx/pull/3422
(VERIFIED) — NAX path was immature for ~5 months after launch.

B6. mlx#3018 (hxu296, contributor, merged 2026-01-26): NAX split-K up to ~1.6x on large-K;
after it, "For bf16 / fp16, MLX is ~0.90–0.95× Torch across the tested shapes" (PyTorch MPS
faster). https://github.com/ml-explore/mlx/pull/3018 (VERIFIED statement; contributor data)

B7. mlx#3196 (open, 2026-03-03): on M5, MLX ~1.1–1.2x slower than PyTorch MPS for 1280²
bf16 GEMM; jagrit06: "I will get to addressing this issue". Later comment (2026-08-11, M3
Ultra): MPSGraph up to 1.18x faster at M=15,731 large-K shapes, bit-identical output, cause
traced to `GEMM_TPARAM_MACRO` tile choice. https://github.com/ml-explore/mlx/issues/3196
(VERIFIED) ⇒ MLX GEMM is **not** uniformly at the limit; tile selection leaves 5–18% on some
shapes (LIKELY).

B8. mlx#3622 (zcbenz, merged 2026-06-13): "The neural accelerator requires macOS 26.2 and
with Xcode 26.5 we have to explicitly pass `-mmacosx-version-min=26.2` to metal toolchain to
compile NAX kernels correctly." Contributor: below 26.2 the cooperative-tensor path lowers
through a shim that "appears broken". https://github.com/ml-explore/mlx/pull/3622 (VERIFIED)
— toolchain fragility for anyone emitting MPP code. Also open: mlx#4533 "no longer builds from
source with Xcode 27 (macOS 27.0 SDK)"; #4594 (2026-09-30) cooperative-tensor build fix for
macOS 27.

B9. Precision trade: mlx#3534 — zcbenz (2026-05-13): M5 fp32 matmul/sdpa error is "because of
Neural Accelerators kicking in, in short it is Apple's equivalent of NVIDIA Tensor Cores which
trade precision for much better performance ... expected behavior"; `MLX_ENABLE_TF32=0`
disables. Documented via #3894 (mlx#3860). https://github.com/ml-explore/mlx/issues/3534
(VERIFIED). Matters for "same output quality" comparisons (fp16/bf16 LLM paths unaffected).

B10. Quantized prefill on NAX — split-K heuristic: mlx#3584 (M5 Max qmm 1.58–1.79x slower than
fp16 at M=96/128) was **not reproduced** on later builds by a contributor across M2/M3 Pro/M5
Max ("The existing `512 / current_tgs` heuristic looks safe-to-helpful"); zcbenz closed it.
https://github.com/ml-explore/mlx/issues/3584 (VERIFIED). mlx#4237 (M5 split-K "cliff" for
N=6656..8192) closed by zcbenz: "currently we are only focusing on real cases happened during
inference". https://github.com/ml-explore/mlx/pull/4237 (VERIFIED)

B11. INT8 on NAX: I found no MLX PR using INT8 NA matmul for LLM prefill (LIKELY absent; R3
code reading should confirm in `quantized_nax.h`). Given A4 (INT8 ≈1.8x FP16), an
INT8-activation prefill path is an **EXPERIMENTAL** lever (changes numerics; needs quality
evaluation), not available in MLX.

B12. llama.cpp parity on M5 (cross-reference, VERIFIED statements): Apple's
"Developer-Ecosystem-Engineering" account authored ggml-org/llama.cpp#20962 (merged
2026-04-25) "Optimize Metal Tensor API usage": +26% geomean prompt processing on M5 Max;
community M5 Max numbers in that thread: llama-7B Q4_0 pp512 3246 t/s, tg128 110.5 t/s.
https://github.com/ggml-org/llama.cpp/pull/20962 — Apple is improving both runtimes.

---

## C. Decode: how close MLX is to bandwidth

C1. **awni, mlx-examples#1155 (2024-12-14)**, M3 Max: `mlx-community/Qwen2.5-32B-Instruct-4bit`
"Generation: 256 tokens, 19.680 tokens-per-sec ... Peak memory: 18.573 GB"; with
`--draft-model mlx-community/Qwen2.5-0.5B-Instruct-8bit --num-draft-tokens 4`: 35.738 tok/s,
"The outputs are identical." https://github.com/ml-explore/mlx-examples/pull/1155 (VERIFIED).
Derived: Qwen2.5-32B and Qwen3-32B both read ≈18.0 GB/token at 4-bit g64 (31.2B layer
params + 0.78B lm_head at 4.5 bpw) ⇒ 19.68 tok/s ≈ 354 GB/s ≈ **89% of the 400 GB/s spec**
(LIKELY; it must be the 40-core/400 GB/s M3 Max, since 354 GB/s exceeds the 30-core part's
300 GB/s). Batch-1 dense decode of a 32B model in MLX is close to the bandwidth wall.

C2. Ceiling for the headline target (LIKELY, derived): Qwen3-32B 4-bit g64 ≈18.0 GB/token ⇒
614 GB/s → 34.1 tok/s theoretical; 530 GB/s (A4 measured) → ≈29.4 tok/s at short context,
before KV-cache reads. Any "beat MLX" decode claim at batch 1 lives inside the gap between
MLX's achieved rate and ~29 tok/s, unless bytes/token change (quantization) or tokens/pass
change (speculation).

C3. mlx#3313 (2026-03-25, user): M5 Air Fish-Speech 36-layer 4-bit model at "83.6 GB/s (55% of
peak)", attributed to "~10 separate Metal compute commands" per layer; asks for FlashFormer-style
megakernels. **zcbenz**: "The really hard part with this is to make the mega kernel work with
various transformer architectures ... not practical for generic machine learning frameworks or
even generic inference engines. So I'm closing this as won't fix, not because it is a bad idea,
but because it is engineering not practical at the moment."
https://github.com/ml-explore/mlx/issues/3313 (VERIFIED quote; the 55%/dispatch attribution is
the user's, unverified — that model is small, 4.7 GB, where fixed costs matter more).

C4. Negative result — mlx#3753 (closed 2026-06-23): doubling qmv `results_per_simdgroup`
(4→8) looked +24–35% in a batched microbench, but the author retracted: single-dispatch decode
was "neutral to slightly negative — fewer but larger threadgroups hurts occupancy at low load".
https://github.com/ml-explore/mlx/pull/3753 (VERIFIED). The base M5 numbers there (4096²
4-bit ≈0.063 ms batched ⇒ ≈150 GB/s vs 153 spec) suggest qmv itself is near bandwidth on
large shapes (LIKELY).

C5. mlx#3961 (merged 2026-08-09, v0.32.1): narrower NVFP4 qmv tile on M5 Max (+2.7% gen on
Qwen3.5-4B). **zcbenz**: "I can not observe performance improvement when benchmarking
individual matmul" (his per-call numbers, one eval per call, have a ≈0.28 ms floor, e.g.
1×4096×512 = 0.278 ms; machine inferred to be an M5 Max because the change is gated to
`applegpu_g17s` and he saw its effect) "But I can definitely observe the improvement on inference ... I hope there is a
easier way to measure improvements like this." https://github.com/ml-explore/mlx/pull/3961
(VERIFIED). Implication: one-op-per-eval microbenchmarks on M5 Max are dominated by a
≈0.28 ms submit/sync floor (LIKELY); R3's "batched in one eval" variant is the right method.

---

## D. Dispatch, command buffers, CPU-side overhead

D1. **mlx#4521 (open, 2026-09-16)**: the Device commits after `max_ops_per_buffer` or
`max_mb_per_buffer` (50/50 on Ultra); "the byte rule trips roughly once per layer, so a
single-token decode step spans 45–90 command buffers. Each boundary costs about 20 µs of GPU
time". Raising to `MLX_MAX_OPS_PER_BUFFER=1000 MLX_MAX_MB_PER_BUFFER=400`: +5% / +7.7% on a
launch-heavy MoE (~1,600 kernels/token) on M3 Ultra; "on a dense Qwen3.8-27B 8-bit model
(about 1,000 large matvecs per step) change nothing". No maintainer reply yet.
https://github.com/ml-explore/mlx/issues/4521 (VERIFIED statement; user data). For a dense
32B model the commit policy is worth ≈0% (LIKELY).

D2. mlx#4562 (open, 2026-09-25): proposes `mx.metal.set_command_buffer_limits`; Instruments
on GLM-5.3 (2× M3 Ultra): "~231 command buffers per decode token, each covering ~0.25 ms of
GPU work", "`eval` lets at most 10 command buffers be in flight", "GPU busy only 68% of the
time at 123k context"; raising limits process-wide made "a 30k-token prefill then held
**+88 GB instead of +8 GB**" because in-flight buffers keep intermediates alive.
https://github.com/ml-explore/mlx/pull/4562 (VERIFIED statement) — decode and prefill want
different commit policies; a static-schedule runtime can choose per phase.

D3. mlx#1963 (2025-03) — awni proposes `MLX_MAX_OPS_PER_BUFFER=8 MLX_MAX_MB_PER_BUFFER=1000000`
(fixed a 65% regression for the user) and says "We want to set these so they work reasonably
well for the given platform. They are tuned for different GPU sizes."
https://github.com/ml-explore/mlx/issues/1963 (VERIFIED). mlx#3192 (closed 2026-06-05): the
MB counter actually sums elements not bytes; zcbenz: "The `MLX_MAX_MB_PER_BUFFER` env is
already commonly used by users and apps so we are not going to merge this".
https://github.com/ml-explore/mlx/pull/3192 (VERIFIED). The variables are undocumented (D1).

D4. Per-op CPU bookkeeping matters only for small/fast models:
- mlx#4048 (closed for inactivity 2026-08-19): removing a per-op `addCompletedHandler` +
  `unordered_set` took Qwen3.5-0.8B-4bit decode "from 280 to 367 tok/s" (M3 Ultra);
  independent re-test on a 172 GB MoE: "no measurable change — 34.78 vs 34.76 tok/s".
  https://github.com/ml-explore/mlx/pull/4048 (VERIFIED)
- mlx#3882 (merged, v0.32.1): reusing WAR-tracking hash tables — "Qwen 35B hits this path
  1,056 times per decoded token", ≈+1.9%. https://github.com/ml-explore/mlx/pull/3882
- mlx#3920 (closed 2026-08-03): cheaper `eval_impl` walk "+2.0 to 2.8% decode" on
  Qwen3.6-35B-A3B (M3 Max); zcbenz: "the implementation is dirty enough that I can't even
  suggest changes". https://github.com/ml-explore/mlx/pull/3920 (VERIFIED)
- History: mlx#998 shared events + async_eval (Gemma on M1 Max 128→140 tok/s, 164 with 80
  ops/buffer); mlx#977 explicit barriers with concurrent dispatch (101.4→103.6 tok/s);
  mlx#1085 split encoders; mlx-examples#1040 removing accidental syncs in async generation
  "180 -> 230 toks/sec" on Llama-3.2-1B-4bit. (VERIFIED, awni PRs)
Conclusion (LIKELY): per-token CPU overhead is a few % for 30B-class dense decode but up to
~30% for sub-1B models; fixed-graph replay helps small models most.

D5. Concurrent dispatch semantics (VERIFIED via mlx#3630, merged 2026-06-12, v0.32.0):
"The compute encoder is created with `MTL::DispatchTypeConcurrent`, so consecutive dispatches
can overlap unless MLX inserts a `memoryBarrier` itself"; auto-barriers covered RAW/WAW only
until angeloskath's general WAR fix. Also mlx#3462: MLX allocates with
`MTLResourceHazardTrackingModeUntracked` and `commandBufferWithUnretainedReferences()`.
https://github.com/ml-explore/mlx/pull/3630 , https://github.com/ml-explore/mlx/pull/3462

D6. CPU↔GPU sync cost — mlx#4438 (open, 2026-08-31): a stream crossing costs "~180 µs" with
the default fence vs "~4 µs" with `MLX_METAL_FAST_SYNCH=1` (M3 Ultra). **zcbenz**: "The way
`MLX_METAL_FAST_SYNCH` is implemented is not guaranteed to work and not recommended by the
Metal team ... So we can not make it the default." https://github.com/ml-explore/mlx/issues/4438
(VERIFIED). Relevant to sampling on CPU or any host round-trip per token.

D7. mx.compile does not reduce kernel launches for a decode graph — **zcbenz** on mlx#4296
(shapeless whole-step compile): "This is not going to help because you can not fuse them
together so even with compilation you would end up launching exact same graph."
https://github.com/ml-explore/mlx/pull/4296 (VERIFIED). awni on mlx#2509: `nn.gelu` "is
already compiled by default" (compile's wins are elementwise fusion). mlx#3789 (decode ideas):
zcbenz on command-buffer reuse: "Sounds a plausible optimization but would need some real world
benchmarks." https://github.com/ml-explore/mlx/issues/3789

D8. Fusion economics (non-maintainer data, mlx#4259): fused add+rms_norm via `metal_kernel`
was **0.73x (slower) at 1×4096** because "dispatch dominates", 1.07–1.21x at prefill shapes;
end-to-end on Qwen2.5-0.5B prefill "roughly 0.5-2.5% of a real prefill". **zcbenz** policy:
"Currently for us to add a new builtin fast op, you have to deploy a custom kernel in a popular
inference engine to prove it is definitely helping first."
https://github.com/ml-explore/mlx/issues/4259 (VERIFIED quote; numbers are a user's). Also
mlx#4297 (fused RoPE+KV append) closed with the same policy. Negative result for "fuse the
small ops" as a primary decode lever on large models.

---

## E. Small-M (speculative verification) — the clearest kernel-level headroom

E1. mlx#3661 (2026-06-11, M1 Pro, 0.31.2): LM-head-sized weight (151936×1024): 4-bit qmm
"0.70 → 1.20 → 2.17 → 3.38 ms for M=1/2/4/8, i.e. the weight sweep is effectively re-paid per
row"; fp16 drops from "~161 GB/s" (GEMV, M=1) to "~56 GB/s" (GEMM) at M=2..64; batched decode
B=2 per-request 39.0 vs llama.cpp 82.4 tok/s. https://github.com/ml-explore/mlx/issues/3661
(VERIFIED statement; user data, pre-qmv_wide).

E2. **mlx#3764 `qmv_wide`** (jessegross, merged 2026-06-26, v0.32.0): for M in [2, limit):
"dequantizes each weight group once and reuses it across the tile ... adapted from llama.cpp's
`kernel_mul_mv_ext`". Affine is "gated to gen-15+" — **M2 Pro (gen 14) affine int4 stays on
plain qmv** (relevant to R3's local measurements). M5 Max total speedups over qmv, int4:
1.4x (M=2), 1.7x (M=4), 1.6x (M=8). angeloskath review: "mxfp4 Qwen 9B w batch size 4 from
253tps to 295tps on the M5 Max" from one hoisted multiply.
https://github.com/ml-explore/mlx/pull/3764 (VERIFIED). mlx#3888 adds `gemv_wide` for fp16/bf16
few-row matmuls (v0.32.1).

E3. mlx#3553 (M4 Pro, 0.31.x): M=3 costs +28–37% over M=1 on 5120→13824 / 13824→5120;
contributor: "`qmv_fast` maps each M row to independent threadgroups, so every extra row
re-reads the entire quantized weight stream". https://github.com/ml-explore/mlx/issues/3553

E4. **mlx#3791** (merged 2026-08-08, v0.32.1), M5 Max, 70B dense 4-bit g64 whole-model
forward vs rows M: M=1 75.9 ms, M=3 84.8, M=5 112.7, M=8 170.8, M=12 241.5, M=16 296.0 (after
PR; M=12 was 294.6 via qmm). Speculative decoding "improves from ~18 tok/s (release) to 24.5
tok/s (main + k=3 chain with a 0.6B draft)". zcbenz approved: "for small shapes I think it is
rather hard to get accurate numbers". https://github.com/ml-explore/mlx/pull/3791 (VERIFIED
statement; contributor data). Derived (LIKELY): on M5 Max, M=3 costs +12%, M=5 +48%, M=8
+125% vs M=1 — far from the ideal "flat until compute-bound" curve (a 4-bit 70B forward at M=8
is ≈1.1 TFLOP, i.e. still memory-bound).

E5. mlx#3863 (closed 2026-08-02): qmm "dead zone" — "flat ~0.51 ms for M=10→32 ... At M=32 the
op runs at ~92 GB/s, far under the ~400 GB/s roofline" (M3 Max, 5120→17408); BM=16 tile gave
0.51→0.31 ms. **zcbenz**: "For small-M we rely on qmv_wide for optimization but this PR tunes
for affine_qmm which I think is the wrong direction. There should be some optimization chances
since get_qmv_batch_limit does not consider use_qmv_wide." Follow-up #3987 closed.
https://github.com/ml-explore/mlx/pull/3863 (VERIFIED)

E6. mlx-lm#990 (open since 2026-03-13; zcbenz requested changes 2026-08-21: "sampler is
coupled with generation step, and module implementation is coupled with speculative decoding",
and on 2026-08-24 "I will see if I can implement this in a clean and progressive way"):
native MTP for Qwen3.5: "1.52x token generation on Qwen3.5-27B dense on M4 Pro"; M2 Ultra 27B
8-bit 1.32x, MoE 1.09–1.11x; "The fp16 27B was actually 0.61x (slower)".
https://github.com/ml-explore/mlx-lm/pull/990 (VERIFIED statements; user data). So MTP
speculative decoding is not in mlx-lm main as of 2026-10-01 (LIKELY; PR still open). Qwen3-32B has no MTP head ⇒ draft
model or n-gram/prompt-lookup only (DFlash-style drafters: mlx-lm#1135, open).

Conclusion (LIKELY): M = 2–16 is where MLX leaves the most kernel-level performance on the
table, and it is exactly the speculative-verification regime. A runtime that keeps the
verify cost ≈ M=1 up to M≈8 roughly doubles the value of speculation.

---

## F. Attention (decode at long context; prefill)

F1. mlx-lm#763 (2026-01-15, open): M3 Ultra MiniMax-M2.1-4bit 146k-context generation 5.95 t/s
(MLX) vs 12.12 t/s (llama.cpp FA). awni asked for repro, then landed mlx#3023 "Faster two pass
sdpa" (v0.30.4) — "Closes https://github.com/ml-explore/mlx-lm/issues/763" (cross-repo, so the
issue still shows open). https://github.com/ml-explore/mlx/pull/3023 (VERIFIED)

F2. mlx#3837 (closed 2026-08-16): decode SDPA with few KV heads has a fixed floor — on M5 Max
(erwinzhang7), n_q=32, L=32768, D=128: n_kv=1 10.3% of peak, n_kv=4 36.1%, **n_kv=8 56.2%**,
n_kv=32 88.3%; "at low KV head counts the kernel is not bandwidth bound at all, it is sitting
on a fixed cost of roughly 0.3 ms at this context length". Custom `metal_kernel` variants hit
~1.5x on M1 Ultra. zcbenz closed: "a vague issue about something is not good enough without a
solution". https://github.com/ml-explore/mlx/issues/3837 (VERIFIED)

F3. mlx-lm#1679 → **mlx#4077** (merged 2026-08-18, v0.32.2): "`sdpa_vector_2pass_1` has every
simdgroup of a threadgroup stream the same K/V block, so loads scale with query heads instead
of kv heads"; new `_gqa` kernel gated to "single-token queries without mask or sinks,
`gqa_factor == 8`, head_dim 64 or 128, `N >= 8192`"; M5 Max kernel speedups 1.08–1.31x.
zcbenz: "Nice improvement, thanks!" https://github.com/ml-explore/mlx/pull/4077 (VERIFIED).
**Qwen3-32B = 64q/8kv/D128 (gqa 8) qualifies only at ≥8K context**; below 8K the old kernel
runs. #4380 extends to gqa 12/16 (v0.32.3); #4596 (open) unrolls 4 keys/iter.

F4. Quantized-KV fused attention is **not in MLX**: mlx#3026 "Quantized SDPA" open since
2026-01-20 (awni: "The numbers seem quite good.. a little too good to be true").
https://github.com/ml-explore/mlx/pull/3026 (VERIFIED state). mlx-lm's QuantizedKVCache
dequantizes outside SDPA; mlx-lm#1587 reports it has *higher* peak memory than fp16 KV.

F5. Prefill is compute-bound — **awni**, mlx-lm#92 (2025-04-18): "For most models we have
optimized this substantially and it is mostly a hardware limitation. In general Apple silicon
is much more FLOP limited than comparable Nividia GPUs and prompt processing is a compute bound
computation." https://github.com/ml-explore/mlx-lm/issues/92 (VERIFIED; pre-M5 statement)

F6. MLX vs llama.cpp, maintainer claims: angeloskath, mlx-lm#858 (2026-02-11, M3 Ultra,
gpt-oss-20b/120b): "In all cases MLX is faster, for the 120b up to 30% for 8k context (1680 vs
1264 tps)"; zcbenz, mlx#2955 (2026-01-21): "after improvements MLX became faster than
llama.cpp". awni, mlx-examples#1029 (2024-12-17): "It looks like MLX 4-bit is about the same
quality as Q4_K_M which is what we expected." https://github.com/ml-explore/mlx-lm/issues/858 ,
https://github.com/ml-explore/mlx/issues/2955 ,
https://github.com/ml-explore/mlx-examples/issues/1029 (VERIFIED quotes)

---

## G. Memory, allocator, wiring, loading

G1. `get_peak_memory` — mlx#3896 (closed 2026-08-08): contributor shows `peak_memory_` tracks
`active_memory_` only (allocator.cpp:166-167 in v0.32.0) while freed buffers stay in the cache;
churn test: peak 1.00 GB vs OS footprint 60.19 GB. **zcbenz**: "`get_peak_memory()` is mostly
useful for measuring the memory consumption when ineferencing/training a single model, where
cache hit rate would be close to 100%. ... I suggest doing manual checking with
`get_active_memory() + get_cache_memory()` instead." https://github.com/ml-explore/mlx/issues/3896
(VERIFIED). Implication for "peak RAM" comparisons: use OS `phys_footprint`, not MLX counters.

G2. Buffer cache policy — mlx#3886: reuse window `[size, min(2*size, size + 2*page_size))`
misses on monotonically growing allocations; **zcbenz**: "the benefits of tuning cache policy
is not high, and I'm closing this as won't fix." https://github.com/ml-explore/mlx/issues/3886
(VERIFIED). mlx-lm preallocates KV in 256-step chunks to avoid it.

G3. **No mmap / zero-copy loading** — mlx#4249 (`mx.mmap_weights`, closed 2026-08-14):
**zcbenz**: "At the moment we are not considering using mmap, see
https://github.com/ml-explore/mlx/discussions/615#discussioncomment-13414338 for some past
experiments." Contributor data (M5 Max): mapped views run "at resident speed", "~1–2 s
reversible mapping instead of a weights read"; caveats (mlx#2878 comment): "Metal wires the
*entire* bytesNoCopy buffer on first kernel use" and "the wired ceiling counts mapped
buffers"; safetensors "tensors land at unaligned offsets, forcing a one-time repack". antbob
(discussion #615): zero-copy needs page-aligned bases, so "you either have to copy or convert
safetensor files to match". https://github.com/ml-explore/mlx/pull/4249 (VERIFIED). ⇒ A
runtime with an aligned weight container + `newBufferWithBytesNoCopy` can win load time /
TTFT-from-cold and avoid a second copy; steady-state footprint is the same (LIKELY).

G4. Wiring — mlx#1510 + mlx-examples#1069 (awni, 2024-10): residency sets/`set_wired_limit`;
Llama-3.1-70B fp16 on M2 Ultra: generation "0.23 toks/sec" not wired vs "4.7" wired.
https://github.com/ml-explore/mlx-examples/pull/1069 (VERIFIED). mlx#4211 (merged
2026-08-13, v0.32.1): "MacOS 27 appears to make GPU residency decisions per
`MTL::ResidencySet`"; one monolithic set caused re-wiring stalls: "0.46 -> 13.19 tokens/sec"
(GLM-5.1, M3 Ultra); now ≤5% sets via `MLX_RESIDENCY_SET_MAX_PCT`. zcbenz: "solid and awesome
work". https://github.com/ml-explore/mlx/pull/4211 (VERIFIED)

G5. Reliability — mlx#3186 (open since 2026-03-01): kernel panics "completeMemory() prepare
count underflow" @IOGPUMemory.cpp:550 (filed as FB22091885) across M1–M5; a user's controlled
A/B: not calling `mx.set_wired_limit` ("residency set stays empty") → "no panic, 10/10 runs";
still holding on macOS 27 beta / IOGPUFamily 162.11. https://github.com/ml-explore/mlx/issues/3186
(VERIFIED statements; user data). Any Metal runtime that wires tens of GB inherits this risk.

G6. Buffer-count limit: `[metal::malloc] Resource limit (499000) exceeded` (one MTLBuffer per
array). zcbenz on mlx#3849: "The code reading resource limit is bugged ... exceeding the limit
clearly indicates some fatal mistakes"; mlx-lm#1662 leak acknowledged by zcbenz as "a MLX bug
that we should fix in the framework side". https://github.com/ml-explore/mlx/issues/3849 ,
https://github.com/ml-explore/mlx-lm/issues/1662 (VERIFIED). Arena allocation sidesteps this
class of failure (LIKELY).

G7. Load-time watchdog: mlx-lm#1572 (open): one-shot `mx.eval(model.parameters())` of >300 GB
models triggers the GPU watchdog ("fixed by chunked eval"). https://github.com/ml-explore/mlx-lm/issues/1572

---

## H. Maintainer policy signals (what MLX will not do soon)

| Topic | Statement | Link |
|---|---|---|
| Megakernels / whole-layer fusion | "not practical for generic machine learning frameworks or even generic inference engines ... won't fix" (zcbenz) | mlx#3313 |
| New fused fast ops | must first be proven "in a popular inference engine" (zcbenz) | mlx#4259, #4297 |
| Shapeless whole-step compile | "you would end up launching exact same graph" (zcbenz) | mlx#4296 |
| mmap / zero-copy weights | "not considering using mmap" (zcbenz) | mlx#4249 |
| Allocator cache policy | "won't fix" (zcbenz) | mlx#3886 |
| Fast CPU↔GPU sync by default | "not recommended by the Metal team" (zcbenz) | mlx#4438 |
| Shape-specific tuning | "only focusing on real cases happened during inference" (zcbenz) | mlx#4237 |
| Metal 4 adoption | "case-by-case basis" (awni) | mlx#2693 |

These are structural choices of a general-purpose framework, i.e. the openings a
model-specialized compiler/runtime can exploit (LIKELY).

---

## I. What this means for "beat MLX on Qwen3-32B 4-bit, batch 1, M5 Max"

1. Decode at short context: MLX is within roughly 10–15% of the bandwidth wall (C1: ≈89% of
   spec on M3 Max; C2 ceiling ≈29 tok/s at measured M5 Max bandwidth). Dispatch/commit-policy
   work is worth ≈0–3% on dense 30B-class models (D1, D4). A compiler alone cannot produce a
   large batch-1 decode win here (LIKELY; negative result).
2. Decode at long context (≥8K): attention efficiency with 8 KV heads is the measurable gap
   (F2: 56% of peak at n_kv=8 on M5 Max; F3 only partly fixes it, and only ≥8K); fused
   quantized-KV attention is absent (F4). Moderate headroom (LIKELY).
3. Speculative decoding: the verify cost curve (E4: +48% at M=5, +125% at M=8 on M5 Max) and
   the M=10–32 qmm dead zone (E5) are the largest kernel-level inefficiencies found. Fixing the
   curve plus a good draft (C1: 1.82x with a 0.5B draft on code) is the highest-leverage
   decode lever (LIKELY / EXPERIMENTAL for magnitude).
4. Prefill: large fp16 NAX GEMM ≈ at the FP16 NA ceiling (B4); remaining levers are tile
   selection (B7, 5–18% on some shapes), quantized-prefill dispatch (B10), attention, and
   INT8 NA matmul (B11; EXPERIMENTAL, ≈1.6–1.9x on matmul per A4 but changes numerics).
5. Load/peak RAM/TTFT-from-cold: MLX copies weights (no mmap; G3) and its peak counters
   under-report (G1); an aligned zero-copy weight format is a concrete, verifiable win on load
   time, not on steady-state RAM (LIKELY).
6. Robustness differentiators: residency-set panics (G5), macOS-27 re-wiring stalls (G4),
   buffer-count limits (G6), toolchain fragility for MPP code (B8).

Open questions: (a) MLX's actual tok/s for Qwen3-32B-4bit on M5 Max (no primary source found;
third-party pages quote ~33 tok/s for Qwen3.5-27B-4bit, unverified); (b) whether MLX's
quantized prefill uses INT8 NA anywhere (code check); (c) whether qmv_wide's affine path is
enabled on gen-17 (M5) for M=2..8 at Qwen3-32B shapes and where `get_qmv_batch_limit` sets the
crossover (code check: `mlx/backend/metal/quantized.cpp`).

---

## Web sources used (besides GitHub)

- [Apple ML Research — Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU (2025-11-19)](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
- [Apple Newsroom — Apple debuts M5 Pro and M5 Max (2026-03-03)](https://www.apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/)
- [tzakharko — Investigating the GPU Neural Accelerators on Apple A19/M5 (2025-10-15)](https://tzakharko.github.io/apple-neural-accelerators-benchmark/)
- [Draw Things — Metal Quantized Attention: pulling M5 Max ahead with Int8 matrix multiplication (2026-04-01)](https://releases.drawthings.ai/p/metal-quantized-attention-pulling)
- Low-weight (third-party, not used for conclusions): [arXiv 2511.05502](https://arxiv.org/abs/2511.05502) (M2 Ultra runtime comparison: "MLX achieved highest sustained generation throughput"); [arXiv 2601.19139](https://arxiv.org/abs/2601.19139) (vllm-mlx claims 21–87% over llama.cpp on M4 Max); third-party pages quoting Qwen3.5-27B-4bit ≈33 tok/s on M5 Max (unverified).
