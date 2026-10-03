# The first 30 GitHub issues

These are the first thirty issues of the inference programme, in gate order from G0 to G6. Together they cover the ten implementations of [Part XXIV](part24-roadmap-now.md), split where the work divides cleanly, and then the next tasks G4–G6 need: the explain report and the performance model, prefill, the GGUF path, the autotuner, exact speculation, and time to first token. Each issue is implementation-ready. It names the crate and files, the steps, the tests, the benchmark and a definition of done, and it links the chapter task IDs it implements.

Conventions:
- **`I-NN`** is this file's issue number. Dependencies refer to these numbers. The filing script at the end replaces them with the real GitHub numbers once the issues exist. It does not use `#NN`, because the bodies also cite upstream issues such as mlx-lm #1470.
- **Gates** follow the task placement of [Part XXV](part25-roadmap-12-months.md) §10, which assigns every chapter task to the gate that owns it: G0 ground truth (weeks 0–2), G1 Metal foundation (1–5), G2 TIR and the oracle (1–6), G3 first tokens (5–9), G4 parity on the M2 Pro (9–14), G5 lead on the M5 Max (14–20), G6 exact speculation and TTFT (20–28). Where an issue carries a task owned by a later gate, the issue says so.
- **"Current Sky behavior"** states what exists in the repository today, from [Part II](part02-sky-today.md) and [Part III](part03-current-performance.md). In one line: Sky is CPU-only and f64-only, with no GPU, Metal, f16, bf16 or integer code, 75,005 production lines in 22 crates, and CI on a self-hosted Linux runner only (Part II, summary).
- Numbers in tests and benchmarks are copied from the cited chapters. A number marked "target" is a goal from the chapter, not a measurement.
- Housekeeping is not an issue here. Before I-01, the owner commits the ten in-flight worktrees to their branches (II-T1), freezes new scientific features through G4 (SPINE D17), and opens the ROADMAP amendment (VI-T4).

| Issue | Title | Gate | Depends on |
|---|---|---|---|
| I-01 | Benchmark harness: one external driver timestamps every token | G0 | — |
| I-02 | M2 Pro baselines, GPU calibration and model physics | G0 | I-01 |
| I-03 | Tensor-stack crates, the macOS Metal CI runner and IP checks | G1 | — |
| I-04 | M5 Max day-one kit, verify-cost curves and the competitor watch | G1 | I-01 |
| I-05 | `sky-metal` foundation: device, buffers, encoders, timing, microbenchmarks | G1 | I-03 |
| I-06 | Quantized GEMV `qgemv_q4g64` at M = 1 | G1 | I-05 |
| I-07 | Batch-invariant verify GEMV family, M = 2…16 | G1 | I-06 |
| I-08 | `sky-tir` core: graph, types and `QFormat`, symbolic dims, verifier, text | G2 | I-03 |
| I-09 | TIR reference interpreter, `dequant_ref`, composites and Philox | G2 | I-08 |
| I-10 | `sky-hub` and config normalization | G2 | I-03 |
| I-11 | MLX loader, the `.skyweights` relayout cache, the same-quantization proof, no-copy mapping | G2 | I-05, I-09, I-10 |
| I-12 | Qwen3 builder and oracle parity with transformers | G2 | I-09, I-10, I-11 |
| I-13 | Tokenizer, chat templates, stop set and streaming detokenizer | G3 | I-10 |
| I-14 | Unfused Metal executor, generation loop v0 and `sky run hf://` | G3 | I-05, I-06, I-12, I-13 |
| I-15 | Correctness gates: quality harness, KLD and top-1 against other engines | G3 | I-01, I-14 |
| I-16 | Decode fusion: five or six kernels per layer | G4 | I-07, I-11, I-14 |
| I-17 | Static memory planner and zero allocation per token | G4 | I-08, I-14 |
| I-18 | The decode program: K steps in flight, GPU-written state, stop flags | G4 | I-16, I-17, I-20 |
| I-19 | GQA decode attention with a fixed split | G4 | I-05, I-09, I-14 |
| I-20 | GPU sampler with keyed Gumbel noise | G4 | I-09, I-14 |
| I-21 | Exactness suite: batch, chunk and verify-width invariance | G4 | I-07, I-14, I-19, I-20 |
| I-22 | `sky explain` per token, the performance model, `sky profile`, performance CI | G4 | I-16, I-17, I-18 |
| I-23 | Prefill GEMM on MPP `matmul2d`, the `simdgroup_matrix` fallback, the `verify_mpp` probe | G5 | I-05, I-06, I-09 |
| I-24 | The GGUF path for same-file pairs with llama.cpp | G5 | I-06, I-09, I-11, I-23 |
| I-25 | Prefill pipeline and token 1 from the last chunk | G5 | I-16, I-19, I-23 |
| I-26 | Autotuner v1 | G5 | I-05, I-16, I-22 |
| I-27 | Exact speculation engine, with prompt-lookup drafting first | G6 | I-07, I-18, I-20, I-21 |
| I-28 | Draft-model speculation and the comparison with mlx-lm | G6 | I-27 |
| I-29 | EAGLE-3 drafter and tree verification | G6 | I-23, I-27, I-28 |
| I-30 | TTFT: exact prefix cache, data-only plan cache, AOT `.skybin` | G6 | I-11, I-18, I-21, I-25 |

The critical path is Part XXIV's: I-05 → I-06 → I-14 → I-16 → I-18. I-01 and I-03 start on day one; I-02 and I-04 follow I-01, and I-05, I-08 and I-10 follow I-03.

## G0: ground truth

### I-01 — Benchmark harness: one external driver timestamps every token for every engine

**Labels:** `gate:G0`, `kind:measurement`, `crate:sky-bench`, `crate:sky-cli`

**Gate:** G0.

**Motivation.** Nothing in the report is a result until it is measured under a protocol a hostile reviewer accepts ([Part XXIV](part24-roadmap-now.md) #1). The competitors' own tools measure different things: mlx-lm's `generation_tps` counts n+1 tokens over n intervals, llama-bench's `tg` decodes from an empty cache without sampling, footprint and RSS miss memory-mapped weights read by the GPU, and chip power counters undercount system energy by 2.4× (R9 §6). These differences matter because the margins Sky expects on plain decode are small: 1.05–1.14× on the M2 Pro ([Part IV](part04-hardware-limits.md) §4).

**Current Sky behavior.** `sky-bench compare` runs 13 CPU programs as subprocesses: one untimed warmup, five timed repetitions, the median wall time of the whole process, and a generated `RESULTS.md` (`crates/sky-bench/src/compare/`; [Part III](part03-current-performance.md) §1.1). Part III adds the slope method for per-unit costs (§1.4). There is no LLM driver, no per-token timestamp, and no memory, power or thermal sampler.

**Proposed architecture** ([Part XXIII](part23-benchmarks.md) §5–§8):
- The harness lives in the separate `sky-evaluation` repository, private from G0 and public at G8, so that it stays separate from Sky's code (D13; [Part XXV](part25-roadmap-12-months.md) §16.2). `sky bench llm` in the Sky repository drives the same drivers (XXIII-T10, at G3).
- One driver per engine emits each token id with a `CLOCK_UPTIME_RAW` timestamp (XXIII-T6).
- Samplers (XXIII-T7): system memory used = wired + anonymous − purgeable + compressed, at 10 Hz through `host_statistics64`; SMC `PSTR` read directly; thermal state; battery. Peak RAM uses R9's `peak_ram_bytes` with its guard rails (XIX-T11).
- A quiet-machine gate: AC power, High Power Mode, no swap, low load. Runs are interleaved in randomized blocks, at least five each.
- Metrics (SPINE D13): TTFT from token ids, with the idle gap stated; prefill = n_prompt / TTFT; decode = (N − 1) / (t_last − t_first); peak RAM as the rise over a pre-start baseline; J/token over loops of ≥ 60 s.
- Statistics: bootstrap confidence intervals and Holm correction (XXIII-T9). Results JSON per R9 §8; protocol per R9 §7.
- `skycount`, an encoder-method interposer that counts dispatches, barriers and encoders per token for any engine (XXIII-T11, V-T2).

**Implementation steps.**
1. Create the `sky-evaluation` skeleton: `benchmark.sh`, `reproduce.sh`, `lock/`, `prompts/`, `corpora/`, `PREREG.md` v0, with the phases `doctor`, `fetch` and `build` (XXIII-T5).
2. Write the mlx-lm and llama.cpp drivers, and a stub Sky driver for I-14.
3. Write the samplers and the gates.
4. Compute the metrics; write the statistics and the report generator.
5. Build `skycount`.

**Dependencies:** none.

**Tests.**
- A fake engine emitting tokens at a known rate checks the arithmetic of every metric.
- The memory probe: a memory-mapped file read only by the GPU is invisible to footprint and visible as wired memory (R9's memprobe table); a mapped-weights engine passes the 0.95 × weights guard; the null-engine noise floor is recorded (XIX-T11).
- The mlx-lm driver's rate equals `generation_tps` × n/(n+1) within 2%; the llama.cpp driver at depth d agrees with `llama-bench -d` within 3%.
- On synthetic lognormal data with a known ratio, the 95% interval covers it 93–97% of the time.
- `doctor` refuses to run on battery, with swap in use, at thermal pressure ≥ 1 or in Low Power Mode (mocked probes); `fetch` verifies sha256 through symlinks.

**Benchmark.** On the M2 Pro, reproduce Part XXIV #1's table: mlx-lm Qwen3-8B 4-bit at 37–39 tok/s decode and about 218 tok/s prefill at 2K; Qwen3-0.6B at about 267 tok/s; llama.cpp Qwen3-8B Q4_0 at 34.8 tok/s (tg128) and 312 tok/s (pp512). `skycount` reproduces mlx-lm's 693 dispatches, 457 barriers and 23 encoders per 8B token, and llama.cpp's ≈ 724 dispatches within 2%.

**Definition of done.** The numbers above are reproduced within 5%; the JSON is byte-stable in schema; a second run on another day agrees within the reported confidence interval; the report regenerates byte-identically from the raw data.

**Links.** [Part XXIV](part24-roadmap-now.md) #1; XXIII-T5, XXIII-T6, XXIII-T7, XXIII-T9, XXIII-T11; XIX-T11; V-T2; SPINE D13.

### I-02 — M2 Pro baselines, GPU calibration and model physics

**Labels:** `gate:G0`, `kind:measurement`, `crate:sky-bench`, `crate:sky-hw`

**Gate:** G0.

**Motivation.** G0's exit criterion is ground truth: mlx-lm and llama.cpp baselines and Metal microbenchmarks on the M2 Pro in JSON, the roofline calibrated with measured bandwidth, and the owner's decision on M5 Max and CI hardware (SPINE §6). Every later gate is measured against these numbers, and every risk has a kill criterion that needs a measurement (XXVIII-T1).

**Current Sky behavior.** `sky-hw` detects the CPU only: architecture, cores, caches and memory, but not the GPU family, cores, neural accelerators or bandwidth, and its fingerprint cannot tell two GPUs apart ([Part II](part02-sky-today.md) §9). The only GPU measurements are the research notes' (R3–R6), taken with ad-hoc scripts.

**Proposed architecture.**
- Competitor pins and adapters (`benches/compare/llm/pins.toml`, MLX, mlx-lm, llama.cpp and tinygrad commits, model commits and sha256), one command that reruns Part V §9's tables (V-T1).
- GPU facts in `sky-hw` (`GpuFacts`: family, cores, neural accelerators, working set, `maxBufferLength`) and the calibration numbers `BW_eff`, `t_dep`, `t_ind`, `F_eff` and the cache knee, measured with the research microbenchmarks until `sky-metal` takes them over in I-05 (XXIII-T1).
- Competitor microbenchmark adapters with the same shapes, byte accounting and rotation for MLX, PyTorch MPS and llama.cpp `test-backend-ops` (XXIII-T3).
- Model physics: `roofline.py` with the per-kernel functions Part XXII uses (IV-T1), extended to Gemma 3, Mistral Small, gpt-oss and Qwen3.6-27B, and a file inspector that computes bytes read per decode token from each file's tensor types (XXIII-T20).
- The risk register `docs/inference/risks.json`, reviewed at every gate (XXVIII-T1).

**Implementation steps.** V-T1, XXIII-T1, XXIII-T3, IV-T1, XXIII-T20, XXVIII-T1; then the baseline runs through I-01 and the JSON commit.

**Dependencies:** I-01.

**Tests.**
- The pinned adapters refuse to run on a commit or hash mismatch (V-T1).
- On the M2 Pro: best read within 3% of 196.3 GB/s; `t_dep` within 1.1–1.25 µs; the cache knee between 8 and 16 MiB; the M2 Pro and the CI Mac get different fingerprints (XXIII-T1).
- `roofline.py` reproduces every number in Part IV (IV-T1); parameter totals match the Hugging Face safetensors totals, and computed bytes equal the files' tensor data sizes exactly (XXIII-T20).

**Benchmark.** Reproduce every Part V §9 number on the M2 Pro within its original run spread (≤ 3%) (V-T1); R3 §7.B batched GB/s and §7.C ratios within 5% and R5 §8.E within 10% (XXIII-T3).

**Definition of done.** Baseline and calibration JSON committed; `roofline.py` calibrated with the measured bandwidth; the risk register exists, and a gate review cannot close while a kill criterion due at that gate lacks a measurement (XXVIII-T1).

**Links.** V-T1; XXIII-T1, XXIII-T3, XXIII-T20; IV-T1; XXVIII-T1; SPINE §6 (G0).

## G1: Metal foundation

### I-03 — Tensor-stack crates, the macOS Metal CI runner and IP checks

**Labels:** `gate:G1`, `kind:infra`, `crate:sky-tir`, `crate:sky-metal`

**Gate:** G1.

**Motivation.** Nothing Metal-specific can be tested in CI today ([Part II](part02-sky-today.md) §12), and G1's exit includes `cargo test -p sky-metal` on a macOS runner. SPINE D14 asks for a self-hosted macOS runner separate from the development laptop. The IP file needs notices, contributor sign-off and a guard against private Apple APIs before any kernel lands ([Part XXVII](part27-moat.md) §5.6).

**Current Sky behavior.** CI runs `cargo test --workspace --locked`, rustfmt, clippy, `cargo-deny`, a release-symbol scan and fuzzing on a self-hosted Linux runner only, with the comment "GitHub-hosted minutes ran out on this private repo" (Part II §12). There are 22 crates and no tensor crate. There is no `THIRD_PARTY_NOTICES.md` and no sign-off rule; `deny.toml` allows only crates.io and permissive licences (Part XXVII §5.6).

**Proposed architecture.**
- Empty crates `sky-{tir,kernel,metal,quant,llm,hub}` with the dependency graph of [Part VI](part06-architecture.md) §7, and a test that fails the build if a scientific crate depends on a tensor crate (VI-T1).
- `.github/workflows/metal.yml` on a self-hosted macOS runner (VI-T2, II-T7): test jobs under `MTL_DEBUG_LAYER=1`, timing jobs kept separate, the OS version pinned and recorded.
- `THIRD_PARTY_NOTICES.md` and an attribution check (XXVII-T1); a DCO sign-off rule with AI-tool trailers kept (XXVII-T2); a private-API lint for `sky-metal`'s build: no private symbols and no `air.*` assembly in default builds (XXVII-T6).
- `cargo deny` covers the new dependencies: `objc2-metal`, `tokenizers`, `minijinja`, `safetensors`, `memmap2`.

**Implementation steps.** VI-T1; provision the runner (owner decision, D14); VI-T2 and II-T7; XXVII-T1, XXVII-T2, XXVII-T6.

**Dependencies:** none (hardware: SPINE D14).

**Tests.**
- A deliberately failing macOS-only test fails CI; the Linux jobs are unchanged (II-T7).
- A deliberately broken kernel fails CI once `sky-metal` has one (VI-T2).
- CI fails on a source file with an unlisted foreign copyright header, on a commit without sign-off, and on a deliberately inserted private call (XXVII-T1, XXVII-T2, XXVII-T6).

**Benchmark.** None. Report the macOS job's wall time.

**Definition of done.** `metal.yml` runs on every pull request, the crates and their dependency test are merged, and the three IP checks run in CI.

**Links.** VI-T1, VI-T2; II-T7; XXVII-T1, XXVII-T2, XXVII-T6; SPINE D14, D15.

### I-04 — M5 Max day-one kit, verify-cost curves and the competitor watch

**Labels:** `gate:G1`, `kind:measurement`, `crate:sky-bench`, `crate:sky-hw`

**Gate:** G1 (an exit item, committed by week 5; [Part XXV](part25-roadmap-12-months.md) §7.2).

**Motivation.** The numbers that decide the thesis are the M5 Max's achievable bandwidth, MLX's verify curve on it, and `matmul2d` efficiency at M = 8–32. They decide R-3 and R-4 and can be measured in days ([Part XXVIII](part28-risks.md) §9). Plain decode ≥ 1.05× mlx-lm needs a raw read of at least about 575 GB/s ([Part XVI](part16-decode.md) §11). Without an M5 Max committed by the end of G0, R-19's STOP/PIVOT applies: suspend M5 work and re-plan around M1–M4. Competitors move monthly, so every claim is re-checked against their current releases (R-8).

**Current Sky behavior.** No GPU code and no M5 hardware; the research microbenchmarks in `research/microbench/` ran only on the M2 Pro ([Part XXVII](part27-moat.md) §5.8).

**Proposed architecture.**
- `research/microbench/run_m5.sh` (IV-T2): raw-Metal read, copy and write bandwidth and the cache knee; dependent and independent dispatch costs; `matmul2d` TFLOPS by tile (64×32, 64×64, 64×128, 128×128) and dtype, int8 included, and by M = 1…32 on Qwen3-32B shapes; sustained clocks over 10 minutes; system power through `PSTR`.
- First baselines through I-01 (IV-T3): mlx-lm and llama.cpp on Qwen3-32B at 512, 8K and 32K context; pp512 and pp8K; peak RAM by the honest metric.
- Verify-cost curves, time(M)/time(1) for M = 1–32 at kernel level and whole forward, on Qwen3-8B and 32B shapes, on the M2 Pro and the M5 Max (IV-T4, V-T4).
- Third-party claims reproduced (V-T11): the community MLX run (28.41 tok/s decode, 730.9 tok/s pp512), llama.cpp (26.38 / 667), MTPLX on Qwen3.6-27B (30.37 → 81.74). Core AI on OS 27 as a third baseline when available (R-6).
- The competitor watch: a weekly job on the macOS runner rerunning V-T1, V-T4 and V-T6 at llama.cpp master and MLX / mlx-lm main, and a monthly pinned re-measurement of MLX, llama.cpp, Monolith, Core AI and MTPLX (V-T10, XXVII-T8, XXVIII-T3).

**Implementation steps.** IV-T2, then IV-T3, IV-T4 and V-T4 on both chips, V-T11; then the watch jobs.

**Dependencies:** I-01; the M5 Max host (SPINE D14).

**Tests.** Results validate against the schema; repeated bandwidth runs agree within 3%; each third-party claim is reproduced within 5% or recorded as not reproducible, with the exact commands (V-T11); the curves reproduce R3/R4/R5 on the M2 Pro (V-T4).

**Benchmark.** The measurements above are the deliverable. On the M2 Pro the curves must show what R3–R5 measured: mlx-lm Qwen3-8B whole-forward verify at 3.33× for M = 4 and 6.48× for M = 8; llama.cpp at 2.08× and 3.76×.

**Definition of done.** Results JSON committed by week 5 (XXVIII-T2); `roofline.py`'s M5 Max entry moves from LIKELY to VERIFIED; R-3, R-4 and R-6 updated in the risk register; the watch has run for 4 consecutive weeks with archived results (V-T10).

**Links.** IV-T2, IV-T3, IV-T4; V-T4, V-T10, V-T11; XXVII-T8; XXVIII-T2, XXVIII-T3; [Part XXVIII](part28-risks.md) R-3, R-4, R-6, R-8, R-19.

### I-05 — `sky-metal` foundation: device, capability probe, buffers, arenas, encoders, timing and microbenchmarks

**Labels:** `gate:G1`, `kind:runtime`, `crate:sky-metal`, `crate:sky-hw`

**Gate:** G1.

**Motivation.** Every GPU byte goes through this crate (SPINE D3). G1's exit criterion is that the ported microbenchmarks reproduce R6 within 10% (Part XXIV #2).

**Current Sky behavior.** No GPU code anywhere ([Part II](part02-sky-today.md), summary); no Metal binding in `Cargo.lock`. Sky's caches are data-only, and `sky build` embeds SIR and its interpreter rather than caching code (Part II §6.4–§6.5).

**Proposed architecture** ([Part XI](part11-metal-backend.md) §3, §6, §8, §10):
- `objc2-metal` 0.3.2; `Gpu::open`; a capability probe (GPU families, Metal 4, threadgroup memory, `maxBufferLength`, recommended working set) recorded in `sky-hw` (XI-T1). A GPU fingerprint that changes with the OS build (X-T1).
- Raw values and guarded selector shims for the OS 26.4 and 27 enums the binding lacks (XI-T2).
- Arenas with a 256-byte planner contract; one argument struct per kernel carrying GPU addresses (XI-T3).
- Libraries compiled at run time with `makeLibrary(source:)`. The OS shader cache reuses compiled code; Sky writes no binary cache (D19). The data-only plan cache comes with I-30.
- Metal 3 encoders, concurrent, with explicit barriers only where the plan says; the Metal 4 path follows at G5 (XI-T6). Metal 4 is not faster per dispatch (R6).
- GPU timestamps per command buffer or stage boundary; Apple GPUs have no per-dispatch counters (R6).
- A residency stub; the heartbeat lands with the weights in I-11.
- Test infrastructure: poisoned outputs, canaries, a fuzz harness, validation jobs (XI-T16).
- The microbenchmark suite in `crates/sky-metal/src/bench/` (XXIII-T2) and GPU calibration `sky hw --calibrate-gpu`, stored as `gpumodel.v1` records in the tuning database (XXII-T2).

**Implementation steps.** XI-T1, XI-T2, XI-T3, XI-T6, XI-T16, X-T1, then XXIII-T2 and XXII-T2.

**Dependencies:** I-03.

**Tests.**
- Every buffer path is written and read back; an unaligned no-copy buffer is refused at creation.
- A 10-kernel chain with pointer arguments runs clean under `MTL_DEBUG_LAYER=1` (XI-T3).
- A 1,000-dispatch dependent chain with a different argument offset per dispatch produces the exact count; randomized dependency graphs with poisoned buffers pass (XI-T6).
- An injected out-of-bounds write and an injected unwritten element are both caught (XI-T16).
- The fingerprint is stable across runs, changes when the OS build string changes, and differs from the CPU fingerprint on every CI machine (X-T1).

**Benchmark.** On the M2 Pro: read bandwidth ≥ 190 GB/s (R6: 196), copy about 183 GB/s, a dependent dispatch about 1.2 µs, an independent one about 0.14 µs (Part XXIV #2). Calibration: `t_dep` within 10% of R6 (1.14–1.25 µs), streaming bandwidth within 3% of R6's GEMV (183.8–186.3 GB/s), repeated runs within 3% (XXII-T2).

**Definition of done.** The benchmarks reproduce R6 within 10%; the probe prints R6's `probe.txt` values on the M2 Pro (families 1–8 and Metal 4 true, 32 KiB threadgroup memory, 8.88 GiB `maxBufferLength`, 11.84 GiB working set); `cargo test -p sky-metal` is green on the macOS runner.

**Links.** XI-T1, XI-T2, XI-T3, XI-T6, XI-T16; X-T1; XXII-T2; XXIII-T2; [Part XXIV](part24-roadmap-now.md) #2; SPINE D3, D19.

### I-06 — Quantized GEMV `qgemv_q4g64` at M = 1, at ≥ 95% of the raw read

**Labels:** `gate:G1`, `kind:kernel`, `crate:sky-kernel`, `crate:sky-metal`

**Gate:** G1.

**Motivation.** Decode is weight streaming: 23.9 of the 25.8 ms of an 8B token in mlx-lm are the 4-bit matmuls (R4). G1's exit criterion is this kernel: ≥ MLX `quantized_matmul` on every Qwen3-8B and 32B shape, and ≥ 88% of the measured raw-Metal read bandwidth on the large shapes.

**Current Sky behavior.** No GPU kernels. On the CPU, Sky's fused elementwise kernels run at C speed (axpy 1.00–1.01×), but nothing is quantized and there is no integer type ([Part III](part03-current-performance.md), summary; [Part II](part02-sky-today.md) §3.2).

**Proposed architecture** ([Part XVI](part16-decode.md) §3.2; [Part XI](part11-metal-backend.md) §9.1):
- Format: MLX affine, 4 bits, group 64, bf16 scales and biases; scales may be negative (50.25% are, R9).
- Each simdgroup owns r output rows. Its 32 lanes split K into 16-value chunks in a fixed order that does not depend on M.
- Activations are pre-scaled by 1, 1/16, 1/256 and 1/4096, so nibbles are masked and never shifted (MLX's `qmv_fast` trick, also llama.cpp's Q4_0 kernel). Each group contributes `s · Σ(q·x) + b · Σx`, accumulated in fp32, with one `simd_sum` per row.
- Hand-written MSL template in `crates/sky-kernel/src/skeletons/qgemv.rs`, under the `inference` pragmas (`fp math_mode(safe)`, `contract(off)`, explicit `fma`; D6). It migrates into kernel IR later. Prologue and epilogue slots (`EPI` 0–3) are filled in I-16, and the row-tile relayout is measured there (XVI-T2).
- The MLX affine codecs for 4, 6 and 8 bits at groups 64 and 128 (XII-T7); the GGUF codecs follow in I-24.

**Implementation steps.** Write the template and its parameter space; port the benchmark to DRAM-sized rotating weight sets; add the tests; compare against MLX on every shape.

**Dependencies:** I-05.

**Tests.**
- Oracle: the output matches an fp32 reference computed from `mx.dequantize`'s values, within a tolerance derived from fp16 input rounding. TIR's `dequant_ref` (I-09) replaces the MLX export when it lands.
- Dequantization bit-exact against `mx.dequantize` on real Qwen3-8B tensors.
- Fuzzing: random shapes (K a multiple of 64, any N) with poisoned outputs.
- Run-to-run bit-identical; built with safe math (a test fails if reassociation changes bits between two instantiations).

**Benchmark.** Rotating DRAM-sized weight sets, many calls per command buffer, GPU timestamps, medians (this avoids tinygrad's BEAM mis-ranking, R2, and MLX's single-eval floor, R3). Targets on the M2 Pro: ≥ 186 GB/s on 32B shapes, 95% of the 196 GB/s raw read, where MLX reaches 162–177; ≥ 150 GB/s on k/v_proj (5120 → 1024), where MLX reaches 114 (XVI-T1; Part XXIV #3).

**Definition of done.** G1's criterion holds on the M2 Pro for every Qwen3-8B and 32B shape at M = 1.

**Links.** XVI-T1, XI-T10, XII-T7 (MLX affine codecs); [Part XXIV](part24-roadmap-now.md) #3.

### I-07 — Batch-invariant verify GEMV family, M = 2…16

**Labels:** `gate:G1`, `kind:kernel`, `crate:sky-kernel`, `crate:sky-tir`

**Gate:** G1.

**Motivation.** Everything distinctive in the plan rests on this kernel ([Part XXVIII](part28-risks.md) §7, condition 2). Exact speculation needs decode (M = 1) and verification (M = k) to share one reduction order (SPINE D6, D10), and cheap verification decides whether speculation gains about 2× or nothing: mlx-lm's verify costs about (k+1) forwards on the M2 Pro (R4), and MLX adds +48% at 5 tokens and +125% at 8 on the M5 Max (mlx#3791).

**Current Sky behavior.** None.

**Proposed architecture** ([Part XI](part11-metal-backend.md) §9.2; [Part XXI](part21-speculative.md) §3.1–§3.2):
- One template with instances MMAX = 2, 4, 8 and 16. VPT must equal `gemv_q4g64`'s, because it is part of the family's reduction schedule.
- The K partition and the reduction tree are identical for every M; only the number of accumulators changes. Registers stay bounded by keeping activations in fp16 and lowering r as M grows (R3's test kernel spilled from M = 4).
- A `reduction_schedule` attribute on every instance, checked by a verifier (`crates/sky-kernel/src/verify/schedule.rs`, XXI-T1).
- Never split M across launches: that re-streams the weights, ×2.5–2.7 at M = 6 ([Part XXI](part21-speculative.md) §3.2).

**Implementation steps.** Write the template, the attribute and the verifier check; the invariance tests; the cost curve on Qwen3-8B and 32B shapes.

**Dependencies:** I-06.

**Tests.** Bitwise M-invariance for M = 1…16 on 1,000 fuzzed shapes and on real Qwen3 weights: every row equals the M = 1 result for the same input row (XI-T11).

**Benchmark.** The verify-cost curve, M = 1…32 (V-T4). Targets: ≤ 1.15× at M = 4 and ≤ 1.3× at M = 8 relative to M = 1 on Qwen3-8B/32B shapes (XXI-T1). On the M2 Pro, G1 asks t(4)/t(1) ≤ 1.45 (β ≤ 0.15) and t(8)/t(1) ≤ 1.5 ([Part XXV](part25-roadmap-12-months.md) §7.2). Also the invariance cost at M = 1 against a specialized GEMV.

**Definition of done.** The invariance tests are green in CI; the invariance cost at M = 1 on the large 32B shapes is ≤ 3%, or the kernel is redesigned (R-11); the G1 verify-cost thresholds hold after at most two kernel iterations, or the question escalates to the M5 Max measurement ([Part XXVIII](part28-risks.md) §5).

**Links.** XI-T11, XXI-T1, V-T4; [Part XXIV](part24-roadmap-now.md) #3; [Part XXVIII](part28-risks.md) R-11.

## G2: TIR and the oracle

### I-08 — `sky-tir` core: graph, types and `QFormat`, symbolic dims, operations, verifier and text format

**Labels:** `gate:G2`, `kind:compiler`, `crate:sky-tir`, `crate:sky-quant`

**Gate:** G2.

**Motivation.** SPINE D1: tensors get a sibling IR. SIR is f64-only, rank ≤ 2, at most 2^24 entries per array, without a memory model, and one new operation touched 23 files ([Part II](part02-sky-today.md) §5.8). Quantized storage is a type, so that "same quantization" can be checked (D5).

**Current Sky behavior.**
- SIR has 47 operations, 13 default passes, a total verifier and a round-trip text format, all over f64 values; only level L1 exists (Part II §5).
- `Poly` provides exact symbolic sizes, unified by equality only, over a domain of sizes ≥ 1 (Part II §3.4, §14). It lives in `crates/sky-types/src/poly.rs` and imports `sky_ast`; `sky-types` depends on `sky-linalg`. `Explanation` and `RuleId` live in `crates/sky-ir/src/pass.rs` (Part II §14).
- The `Tensor` shape word parses, checks and lowers, but no tensor can exist at run time (Part II §3.5). There are no integer or quantized types (Part II §3.2).

**Proposed architecture** ([Part VII](part07-tensor-ir.md) §3–§10; [Part XII](part12-quantization.md) §2):
- First, `Poly` moves into a dependency-light crate and `Explanation`, `RuleId` and `PassCx` move to `sky-diag`, re-exported so that no SIR caller changes (II-T8, VII-T2).
- An arena-interned, hash-consed SSA graph with derived types, modeled on tinygrad's UOp (R2 §9.1).
- `Type { dtype, shape, layout }`. `DType` includes `Quant(QFormat)`: format, packing and layout as three descriptors, with canonical names, printers and parsers (XII-T1), and the type rules QT1–QT8 with their diagnostics (XII-T2).
- `Dim` is `Const(u64)` or `Sym(Poly)` with interval bounds; `SymTable` provides `prove_le`, `range` and `divides`, so that `pos: Int<0..MAX_CTX>` exists.
- Views as index functions with a simplifier and zero-copy detection; the operation table and type rules; the verifier V1–V24; the text format with a fuzz target.

**Implementation steps.** II-T8 and VII-T2; VII-T1; XII-T1 (descriptors); VII-T3; VII-T4; VII-T5 with XII-T2; VII-T6.

**Dependencies:** I-03.

**Tests.**
- SIR goldens and `--explain` goldens byte-identical after the moves; `cargo tree -p sky-tir` shows neither `sky-linalg` nor `sky-ast`.
- Interning is idempotent; `structural_hash` is stable across runs and thread counts; `prove_le` never answers `Proven` where an exact evaluation at random points within the bounds refutes it.
- One passing and one failing case per type rule; a failing fixture per verifier rule; UI goldens for the quantization diagnostics (`add` on a quantized tensor, a wrong blocked axis, K not a multiple of the block, `quantize` in an inference graph, int8 KV without a mode).
- A composed index function equals sequential application on random shapes; every view chain of the Qwen3 and Gemma 3 graphs ends affine, without `//` or `%`.
- Text round trip on every test graph and on the Qwen3-32B layer of §9.2; 60 s of fuzzing per pull request.

**Benchmark.** VII-M1: the time to build and verify the Qwen3-32B module, which should be negligible next to weight loading.

**Definition of done.** The acceptance tests of II-T8, VII-T1 to VII-T6, XII-T1 (descriptors) and XII-T2 pass.

**Links.** II-T8; VII-T1, VII-T2, VII-T3, VII-T4, VII-T5, VII-T6; XII-T1, XII-T2; [Part XXIV](part24-roadmap-now.md) #4.

### I-09 — TIR reference interpreter, `dequant_ref`, composites and Philox

**Labels:** `gate:G2`, `kind:test`, `crate:sky-tir`

**Gate:** G2.

**Motivation.** The reference interpreter is the oracle for every kernel and every fusion (SPINE D6), and the test oracles are among the few assets that are hard to copy ([Part XXVII](part27-moat.md) §3.9). Every GEMV, GEMM and relayout test needs a bit-exact dequantization reference (D5).

**Current Sky behavior.** `sky-interp` and `sky-ir-interp` interpret f64 `Value`s shared by three engines; Part II §14 rules them out for tensors.

**Proposed architecture** ([Part VII](part07-tensor-ir.md) §11; [Part XII](part12-quantization.md) §2.4):
- `Mode::F32` and `Mode::F64` interpreters (VII-T7).
- `dequant_ref` for the P0 formats: MLX affine b4, b6 and b8 at groups 64 and 128, with the scale dtype recorded; GGUF Q4_0, Q4_1, Q8_0, Q4_K and Q6_K (`crates/sky-tir/src/interp/dequant.rs`, XII-T1).
- Composites with reference decompositions: `rms_norm`, `rope` with all scalings, `softmax`, `attention` with all masks, gated linear units (VII-T8).
- Philox4x32-10, `rng.uniform` and `sample` (VII-T10).

**Implementation steps.** VII-T7; XII-T1 (`dequant_ref` and golden vectors); VII-T8; VII-T10.

**Dependencies:** I-08.

**Tests.**
- Bit-identical results with 1, 2 and 8 threads; per-operation known-answer tests.
- Golden vectors from `mx.dequantize` (MLX affine b4/b6/b8, g64/g128) and from ggml's `dequantize_row_*` (Q4_0, Q4_1, Q8_0, Q4_K, Q6_K), bit-exact in f32, including groups with negative scales.
- Native interpreter versions of composites equal their decompositions bit for bit in `Mode::F32`; a dev-only comparison with transformers' modules on random tensors.
- Philox known-answer vectors; a χ² test of `sample`; equal samples for equal (key, position) whatever the batch.
- A differential property test against a NumPy reference on random small graphs (dev-only).

**Benchmark.** VII-M4: interpreter speed on Qwen3-0.6B and 8B, which decides which oracle tests run per pull request and which nightly.

**Definition of done.** 100% of the operation set is verified and interpreted (Part XXIV #4), and XII-T1's golden-vector test passes.

**Links.** VII-T7, VII-T8, VII-T10; XII-T1; [Part XXIV](part24-roadmap-now.md) #4.

### I-10 — `sky-hub` and config normalization

**Labels:** `gate:G2`, `kind:runtime`, `crate:sky-hub`, `crate:sky-llm`

**Gate:** G2 (XIII-T1, XIII-T4); the download and provenance tasks finish at G3.

**Motivation.** `sky run hf://…` must work offline-first, share the Hugging Face cache safely and pin what it ran. Configuration errors must fail loudly: mlx-lm drops unknown keys, so a re-saved Llama 3.1 config loads with θ = 1e4 and Gemma 3's scale becomes 1.0 instead of 0.125 ([Part XIII](part13-llm-runtime.md) §4.1; R9 §3.1).

**Current Sky behavior.** No network or Hugging Face code. `sky-project` opens files safely by handle, which carries over; its no-symlink rule does not fit the Hugging Face cache ([Part II](part02-sky-today.md) §14).

**Proposed architecture** ([Part XIII](part13-llm-runtime.md) §3–§4):
- The `hf://org/model[:QUANT]` grammar; cache discovery honoring the Hugging Face environment variables in order (XIII-T1).
- A config normalizer producing a `ModelSpec`: accepts both `rope_scaling` and `rope_parameters`; an unknown key fails with `CFG-003` (XIII-T4).
- By G3: offline-first resolution and downloads with flock-compatible locking, never writing under the shared blob store (XIII-T2); provenance, the commit (`X-Repo-Commit`) and sha256 recorded in a models lock (XIII-T3).

**Implementation steps.** XIII-T1, XIII-T4; then XIII-T2, XIII-T3.

**Dependencies:** I-03.

**Tests.**
- Table-driven URI tests over every example and error; a fixture cache with shared-store symlinks resolves through the links.
- The 31 configs of R9's corpus and their transformers-5 re-saves give equal `ModelSpec`s; the Llama 3.1 re-save gets θ 5e5 and llama3 ×8; Gemma 3's global scale is 0.125.
- A pinned complete snapshot makes 0 network calls (a mock server counts requests); a concurrent `hf_hub_download` blocks on the same lock; no file is ever created under `hub/blobs/`.
- `sky verify` detects a modified byte in a shard through two symlinks.

**Benchmark.** Resolution within the TTFT budget of [Part XVIII](part18-ttft.md) §9 (hub ≤ 5 ms, target).

**Definition of done.** The acceptance tests of XIII-T1 to XIII-T4 pass.

**Links.** XIII-T1, XIII-T2, XIII-T3, XIII-T4; [Part XXIV](part24-roadmap-now.md) #5.

### I-11 — Weights: the MLX safetensors loader, the `.skyweights` relayout cache, the same-quantization proof, and no-copy mapping with residency

**Labels:** `gate:G2`, `kind:runtime`, `crate:sky-quant`, `crate:sky-llm`, `crate:sky-metal`

**Gate:** G2 (loader, cache, proof); the no-copy mapping and residency tasks finish at G3.

**Motivation.** mlx-community safetensors files are unaligned: data starts at an offset ≡ 6 (mod 8), and 0 of 907 tensors are even 4-byte aligned. On the M2 Pro an unaligned `uint` load silently returns the rounded-down word, so zero-copy use of those files is impossible (R9). SPINE D9 answers with a page-aligned, lossless relayout cache keyed by the source sha256 and mapped with no copy; D5 requires a test proving the dequantized values bit-equal. Without a residency heartbeat, llama.cpp's first tokens after an idle pause are 16× slower (#10119).

**Current Sky behavior.** No model loading. Sky's tuning database already lives under `~/Library/Caches/sky`, the precedent for the cache location ([Part II](part02-sky-today.md) §9).

**Proposed architecture** ([Part XII](part12-quantization.md) §4–§5; [Part XIII](part13-llm-runtime.md) §6; [Part XI](part11-metal-backend.md) §6.2–§6.4):
- A safetensors reader with MLX quantization inference that follows mlx-lm's per-module rules exactly; an ambiguous module fails by name (XIII-T6, XII-T3).
- The `.skyweights` writer and reader with the lossless recipes RL1–RL7: every plane 16 KiB-aligned; Q‖K‖V concatenation and gate‖up interleaving for I-16; atomic rename; `sky weights build`, `verify` and `gc` (XII-T5, XIII-T7).
- The same-quantization proof: an exhaustive CPU proof against `dequant_ref` with a manifest, plus a GPU codec leg in `crates/sky-metal/tests/codec_exact.rs` (XII-T6).
- By G3: `Buffer::no_copy(mmap, range)` views over `.skyweights` segments, split at `maxBufferLength`, and overlapping GGUF views (XI-T4); a residency set with a heartbeat keeper and a plan-fit check against the wired limit (XI-T5). XIX-T7 refines the residency policy at G5.
- If the disk is too small, fall back to copying into Metal buffers, as MLX does, with a warning.

**Implementation steps.** XII-T3 / XIII-T6; XII-T5 / XIII-T7; XII-T6; then XI-T4 and XI-T5.

**Dependencies:** I-05, I-09, I-10.

**Tests.**
- `relayout_roundtrip_bit_exact` on all tensors of Qwen3-8B-4bit; every tensor at least 256-byte aligned and regions 16 KiB-aligned (XIII-T7).
- 0 mismatches on Qwen3-0.6B and 8B 4-bit and on Qwen3-8B Q4_K_M; a deliberately MSB-first recipe and a sub-block scale bug both fail (XII-T6).
- A killed build leaves no `.skyweights` file; per-tensor formats equal mlx-lm's on the 0.6B, 8B and 32B 4-bit files (XII-T3).
- Mapping Qwen3-0.6B's `.skyweights` and a GGUF file does not grow RSS by the file size, and GPU dequantization equals CPU dequantization bit for bit (XI-T4).

**Benchmark.** Relayout of Qwen3-8B (4.6 GB) is SSD-bound, a few seconds, once; later loads are mmap only, milliseconds (Part XXIV #5). Record the build and proof times for Qwen3-32B: they decide whether the cache is built on first run or only by `sky compile` ([Part XII](part12-quantization.md) §14). Residency: decode, idle 2 s, decode, with the first token after the gap within 10% of steady state (XI-T5).

**Definition of done.** The same-quantization proof is green in CI for 0.6B and 8B; a second load maps the file without copies; XI-T4 and XI-T5 pass by G3.

**Links.** XII-T3, XII-T5, XII-T6; XIII-T6, XIII-T7; XI-T4, XI-T5; XIX-T7 (G5); [Part XXIV](part24-roadmap-now.md) #5; SPINE D5, D9.

### I-12 — The Qwen3 builder and oracle parity with Hugging Face transformers

**Labels:** `gate:G2`, `kind:compiler`, `crate:sky-llm`, `crate:sky-tir`

**Gate:** G2.

**Motivation.** G2's exit criterion: Qwen3-0.6B logits match HF transformers (fp32) within tolerance. The Rust builder is the first of D7's two front doors and the fast path to first tokens.

**Current Sky behavior.** The Sky language cannot express a Qwen3 layer; R1 counts 12 missing features ([Part XXVII](part27-moat.md) §3.5; [Part II](part02-sky-today.md) §3.7).

**Proposed architecture** ([Part VII](part07-tensor-ir.md) §12; [Part XXIV](part24-roadmap-now.md) #6):
- `build(config) -> Graph` emits the embedding gather, the decoder layers (64 for 32B, 36 for 8B, 28 for 0.6B), the final norm and the output head, with composites emitted directly.
- Two programs, prefill (symbolic T) and decode (T = 1), both taking KV state values.

**Implementation steps.** `crates/sky-tir/src/build.rs` (`TirBuilder`, `GraphBuilder`, `ModelGraph`); `crates/sky-llm/src/arch/qwen3.rs`; reference logits generated once with transformers and stored by sha256 under `tests/data/`.

**Dependencies:** I-09, I-10, I-11.

**Tests.** `crates/sky-llm/tests/oracle_qwen3.rs`: logits for 8 fixed prompts within a tolerance set by the weights' bf16 storage; greedy tokens agree for 32 steps, except at documented near-ties. Known causes of disagreement to check first: RoPE variants, the QK-norm placement, the EOS set.

**Benchmark.** None: the interpreter runs at seconds per token, and this issue is about correctness.

**Definition of done.** The oracle test is green.

**Links.** VII-T11, XIII-T5; [Part XXIV](part24-roadmap-now.md) #6; SPINE D7.

## G3: first tokens

### I-13 — Tokenizer, chat templates, stop set and streaming detokenizer

**Labels:** `gate:G3`, `kind:runtime`, `crate:sky-llm`

**Gate:** G3.

**Motivation.** G3 requires a tokenizer that is exact on a test corpus. Benchmarks exchange token ids, never text, because tokenizers differ: llama.cpp skips NFC normalization, so NFD "café naïve Å" is 5 ids in HF and 11 in llama.cpp (R9 §4.2).

**Current Sky behavior.** None. A CPU-side tokenizer written in Sky would be slow today: every operation on an array of more than 16 entries is a boxed runtime call, 148 ns per element access ([Part III](part03-current-performance.md) §3.1). The year-1 tokenizer is Rust (Part III §7.4).

**Proposed architecture** ([Part XIII](part13-llm-runtime.md) §7–§10):
- The `tokenizers` 0.23.2 crate, fancy-regex build (XIII-T9).
- minijinja with pycompat, a Python-exact `tojson` and `preserve_order` (XIII-T10).
- Qwen3's thinking mode made explicit (mlx-lm turns it on by default); exactly one BOS; the EOS set as the union of config, generation_config and tokenizer_config; content escaping, so that a typed `<|im_end|>` never becomes token 151645 (XIII-T11).
- A streaming detokenizer that emits only at UTF-8 boundaries (XIII-T13).

**Implementation steps.** XIII-T9, XIII-T10, XIII-T11, XIII-T13.

**Dependencies:** I-10.

**Tests.**
- 168/168 golden ids identical to Python; a nightly differential fuzz.
- R9's 30 template cases byte-identical, including Qwen3 with thinking on and off and a pinned `strftime_now`.
- Double-BOS cases for Llama 3.1, Gemma 3 and Mistral Small 3.2 give exactly one BOS; Qwen3's stop set is {151645, 151643}.
- `detok_stream_equals_decode` on the corpus and on Qwen3's id-61804 emoji case, with no U+FFFD in any chunk.

**Benchmark.** Render plus tokenize ≤ 2 ms in the TTFT budget ([Part XVIII](part18-ttft.md) §9, target).

**Definition of done.** The acceptance tests of XIII-T9, XIII-T10, XIII-T11 and XIII-T13 pass.

**Links.** XIII-T9, XIII-T10, XIII-T11, XIII-T13; [Part XXIV](part24-roadmap-now.md) #7.

### I-14 — Unfused Metal executor, generation loop v0 and `sky run hf://`

**Labels:** `gate:G3`, `kind:runtime`, `crate:sky-kernel`, `crate:sky-llm`, `crate:sky-cli`

**Gate:** G3.

**Motivation.** G3's exit criterion: `sky run hf://mlx-community/Qwen3-8B-4bit` on Metal, unfused; greedy output matching mlx-lm for 128 tokens on 20 prompts, near-tie divergences explained; decode ≥ 0.7× mlx-lm.

**Current Sky behavior.** `sky run` runs Sky source on the `ast`, `sir` and `native` engines; the `Engine` enum is at `crates/sky-cli/src/run.rs:67` and the subcommand match at `crates/sky-cli/src/main.rs:70` ([Part II](part02-sky-today.md) §14). There is no model path and no GPU code generator.

**Proposed architecture** ([Part XXIV](part24-roadmap-now.md) #8; [Part XI](part11-metal-backend.md) §7; [Part VIII](part08-graph-optimizer.md) §2–§4):
- The graph optimizer's base: the pass manager and pipeline, the rule language compiled into matchers, and `canon`, `fold`, `cse` and `views` (VIII-T1, VIII-T2, VIII-T3).
- The MSL emitter for generated kernels: the layout of Part XI §7.2, the `inference` pragmas, function constants, `min_lang` (XI-T7).
- `crates/sky-kernel/src/lower_unfused.rs`: every TIR operation lowers to a skeleton (the GEMV of I-06, a simple attention kernel, argmax) or a generated elementwise or reduction kernel.
- Memory plan v0: one buffer per value, then lifetime reuse. KV in per-layer slabs, 16 KiB-aligned and zero-filled, preallocated for the configured maximum context (XIV-T1).
- Greedy argmax on the GPU. Loop v0: one command buffer per token with a host wait; I-18 removes the wait.
- The CLI: `sky run` with `--stats` printing every field R9 requires (XIII-T17), and `sky bench llm` through I-01's drivers (XXIII-T10).

**Implementation steps.** VIII-T1 to VIII-T3; XI-T7; the unfused lowering; XIV-T1; `crates/sky-llm/src/generate.rs`; XIII-T17; XXIII-T10; the golden prompts.

**Dependencies:** I-05, I-06, I-12, I-13.

**Tests.**
- I-12's oracle comparison, now on Metal, within tolerance; golden prompts.
- MSL golden snapshots: the same IR gives identical bytes, and every snapshot compiles at its `min_lang` on macOS 26 (XI-T7).
- The pipeline verifies after every pass; rewriting terminates on random graphs, with identical output for 1 and 8 threads (VIII-T1, VIII-T2); interpreter results are bit-identical before and after `canon`, `fold`, `cse` and `views` (VIII-T3).
- Qwen3-32B at 32K allocates exactly 8,589,934,592 bytes of KV plus at most one page of padding per plane (XIV-T1).
- Divergences from mlx-lm must be explained by measured near-ties: bf16 and fp16 greedy outputs diverge on 49–100% of prompts across stacks (R8), so the gate counts explained divergences and never promises identity with MLX.

**Benchmark.** Qwen3-8B decode at about 0.7–0.9× mlx-lm on the M2 Pro (about 19 kernels per layer, as in mlx-lm, plus a host wait per token); 0.6B lower, because it is overhead-bound (Part XXIV #8). `sky bench llm`'s Sky medians agree with `sky-evaluation`'s within the confidence interval (XXIII-T10).

**Definition of done.** G3's criterion is met; `--stats` prints the R9-required fields.

**Links.** VIII-T1, VIII-T2, VIII-T3; XI-T7; XIV-T1; XIII-T17; XXIII-T10; [Part XXIV](part24-roadmap-now.md) #8; SPINE G3.

### I-15 — Correctness gates: the quality harness, KLD and top-1 agreement against other engines

**Labels:** `gate:G3`, `kind:test`, `crate:sky-quant`, `crate:sky-bench`

**Gate:** G3.

**Motivation.** No speed claim is made on a pair of files until tokenizer ids are identical, KLD(Sky ‖ other) ≤ 0.1 × KLD(other ‖ BF16) on identical weights, and teacher-forced top-1 agreement is ≥ 99% (SPINE D13). This is a kill criterion from G3 on ([Part XXVIII](part28-risks.md) R-15): if a pair fails, no speed claim on it until it is fixed. Every quantization is built from one pinned BF16 base, so that "same quantization" is checkable ([Part XXIII](part23-benchmarks.md) §5.1).

**Current Sky behavior.** None: until I-14, Sky produces no logits.

**Proposed architecture** ([Part XII](part12-quantization.md) §7; [Part XXIII](part23-benchmarks.md) §5.3):
- The quality harness in `crates/sky-quant/src/quality/` and `sky bench llm --quality`: `.skylogits` dumps, KLD, perplexity, top-1 and Δp statistics, and a `.kld` export (XII-T10).
- Quality adapters writing teacher-forced dumps for mlx-lm, llama.cpp and PyTorch, and the G-same and G-greedy tables (XXIII-T8).
- The artefacts: MLX 4-bit g64, GGUF Q4_0 and Q4_K_M built from the pinned BF16 bases of Qwen3-0.6B and 8B; whether `mlx_lm.convert` reproduces mlx-community's bytes decides the headline MLX file ([Part XXIII](part23-benchmarks.md) §16, item 10).

**Implementation steps.** XII-T10; XXIII-T8; build the artefacts; run G-same and G-greedy in CI on 0.6B.

**Dependencies:** I-01, I-14.

**Tests.**
- Two BF16 references agree to a mean KLD of about 1e-4 or less on Qwen3-0.6B, or the pre-registered fallback rule triggers and is reported (XXIII-T8, XII-T10).
- Our KLD of a GGUF pair equals llama.cpp's own perplexity tool within its uint16 quantization noise.

**Benchmark.** The quality line printed next to every speed number: KLD of MLX 4-bit g64, Q4_K_M and Q4_0 against BF16 ([Part XII](part12-quantization.md) §14).

**Definition of done.** G-same and G-greedy run in CI on Qwen3-0.6B and pass for Sky against mlx-lm on the same MLX file.

**Links.** XII-T10; XXIII-T8; [Part XXVIII](part28-risks.md) R-15; SPINE D5, D13.

## G4: parity on the M2 Pro

### I-16 — Decode fusion: horizontal Q‖K‖V and gate‖up, prologues and epilogues, five or six kernels per layer

**Labels:** `gate:G4`, `kind:compiler`, `crate:sky-tir`, `crate:sky-kernel`

**Gate:** G4.

**Motivation.** mlx-lm issues 693 dispatches (457 barriers) per Qwen3-8B token, and its non-matmul work is 7.6% of the token (R4). Sky's plan issues 182–219 per 8B token and 322–387 per 32B token; the model gives 1.03–1.08× on 32B at short context and 1.5–2.0× for a 0.6B draft ([Part IX](part09-scheduler.md) §3.3).

**Current Sky behavior.** Sky's CPU `fuse` pass fuses elementwise SIR chains into kernels that run at C speed; it is a precedent, not an implementation ([Part II](part02-sky-today.md) §7, §14). I-14 provides the first TIR passes.

**Proposed architecture.**
- The rest of the graph optimizer ([Part VIII](part08-graph-optimizer.md) §3): `shapes` with guards, `recognize`, `horizontal` with the relayout plan, `specialize`, `dead`, `layout`, `legality`, and the contract log (VIII-T4 to VIII-T9).
- The scheduler ([Part IX](part09-scheduler.md) §2–§6, §9): the plan data model, anchors and legality, partition search, recompute versus store (the norm statistics are recomputed in the K1 and K4 prologues, switching to stored at M = 6 on M5 Max parameters), guards G1–G8, barrier insertion (IX-T1 to IX-T5, IX-T9).
- The kernels ([Part XVI](part16-decode.md) §4):
  - K1: rms_norm prologue + Q‖K‖V GEMV + q/k-norm + RoPE + KV-append epilogue (XIV-T2);
  - K2: GQA decode attention (I-19);
  - K3: o_proj + residual;
  - K4: rms_norm + gate‖up GEMV + SwiGLU;
  - K5: down + residual (XVI-T3 to XVI-T5).
- The decode relayout: Q‖K‖V concatenation, gate‖up interleave, and the optional block order, kept only if ≥ 2% faster (XVI-T2).
- If K1's register pressure is too high, the occupancy guard splits it into two kernels; both are measured.

**Implementation steps.** VIII-T4 to VIII-T9; IX-T1 to IX-T5 and IX-T9; XVI-T2 to XVI-T5; XIV-T2; then the fused plan in I-14's executor.

**Dependencies:** I-07, I-11, I-14.

**Tests.**
- Fused and unfused plans agree within the `inference` tolerance.
- Concatenated and separate products give identical bits on the GEMV skeleton, and the relayout plan passes the same-quantization test (VIII-T5); `kv_append_invariance`: decode, verify (γ = 1…16) and prefill chunks (64…2048) write identical K/V bytes (XIV-T2).
- Changing a producer's threadgroup count must change the plan, never silently the result (XVI-T4).
- Greedy partition equals the exhaustive optimum on Qwen3, Llama 3 and Gemma 3 blocks, in < 10 ms per block (IX-T3); SwiGLU fused into down is refused by guard G6 (IX-T5).
- Barriers = dispatches in the decode plan; a plan with independent kernels gets no barrier between them; Metal validation reports no hazards (IX-T9).

**Benchmark.** `skycount`: ≤ 6 × L + 3 dispatches per token. Qwen3-8B decode ≥ 1.0× mlx-lm on the M2 Pro (G4), aiming at 1.05–1.10×; non-matmul time near 2% against mlx-lm's 7.6%. A fused GEMV must stream within 1% of the unfused one on the large shapes, or the norm moves back to a separate kernel there ([Part XVI](part16-decode.md) §11).

**Definition of done.** `sky explain` shows one Q‖K‖V and one gate‖up GEMV per layer; with I-17 and I-18, G4's decode criterion holds.

**Links.** VIII-T4 to VIII-T9; IX-T1 to IX-T5, IX-T9; XVI-T2, XVI-T3, XVI-T4, XVI-T5; XIV-T2; [Part XXIV](part24-roadmap-now.md) #9.

### I-17 — Static memory planner: lifetime intervals, arena offsets and zero allocation per token

**Labels:** `gate:G4`, `kind:compiler`, `crate:sky-tir`, `crate:sky-llm`

**Gate:** G4.

**Motivation.** Zero allocation in the hot path, and every byte owned and explainable (SPINE D9, D12). It avoids MLX's transients: KV growth by concatenation every 256 positions, 2× KV on prompt-cache reuse (#1395), and +1.41 / +2.38 GB for unfused quantized-KV scores at 8K / 32K (#1587) ([Part XIX](part19-memory.md) §8).

**Current Sky behavior.** SIR has `Lifetimes` and a buffer-reuse explanation (`MEM-001`) for CPU arrays ([Part II](part02-sky-today.md) §10, §14). There is no GPU planner.

**Proposed architecture** ([Part XIX](part19-memory.md) §3–§5):
- Lifetime intervals over schedules, with layer-body reuse and concurrency domains (XIX-T1).
- Best-fit offset assignment by size, with the in-place rules `TMEM-001/002/003` and a scheduler veto (XIX-T2).
- A plan verifier and a NaN-poison mode for dead slots (XIX-T3); may-be-zero sizes without zero-length buffers or dispatches (XIX-T4).
- Everything allocated at session start (XIX-T5).

**Implementation steps.** XIX-T1 to XIX-T5.

**Dependencies:** I-08, I-14.

**Tests.**
- A property test against brute force over 10⁵ random schedules; a deliberately bad plan is caught on the GPU.
- `currentAllocatedSize` is constant over 10,000 tokens; a counting allocator shows 0 allocations per token.
- An empty KV cache, a fully cached prompt and a zero-draft round run without allocation or dispatch.

**Benchmark.** Efficiency ≥ 0.95 on the Qwen3 layer body; planning 64 layers takes ≤ 10× the time of 1 layer. Peak RAM against mlx-lm: the target ≤ 0.9× at 32K needs mlx-lm to sit ≥ 3 GB above the 27.5 GB floor, which is unmeasured ([Part XXVIII](part28-risks.md) R-18).

**Definition of done.** XIX-T1 to XIX-T5 pass.

**Links.** XIX-T1 to XIX-T5; [Part XXIV](part24-roadmap-now.md) #9.

### I-18 — The decode program: direct re-encoding, K steps in flight, GPU-written `StepState`, stop flags, shared events, and the small-model plan

**Labels:** `gate:G4`, `kind:runtime`, `crate:sky-llm`, `crate:sky-metal`

**Gate:** G4.

**Motivation.** G4 requires Qwen3-0.6B decode ≥ 1.3× mlx-lm, which runs it at 54.5% of bandwidth with 1.94 ms of CPU per 3.5 ms of GPU (R4). The GPU idles about 105 µs each time the CPU waits before committing the next command buffer (R6). mlx-lm's Python loop drew 7.1 W of CPU during 0.6B decode (R9 §6.3).

**Current Sky behavior.** No GPU loop; I-14's loop waits on the host every token.

**Proposed architecture** ([Part XX](part20-persistent-execution.md) §3–§7):
- `DecodeProgram` as validated data (XX-T1).
- `StepState` and `sample_finish`: the GPU writes the next token, the position and the RNG counter for the next step (XX-T2).
- A Metal 3 direct-encoding submitter: the static plan is re-encoded each step (tens of µs, hidden behind the GPU), with K steps in flight and one command buffer per step (XX-T3). Metal 4 command reuse still re-encodes, so it is no cheaper (D9).
- An `MTLSharedEvent` and a token ring buffer feed the CPU streamer (XX-T4).
- A stop flag turns later steps into no-ops by early exit or indirect dispatch; cancellation keeps KV and state consistent (XX-T5). Indirect grids size attention by context (XX-T6).
- The GPU-resident generation loop (XIII-T14) and the decode plan of five or six kernels per layer (XVI-T6); the small-model plan with ≤ 5 dependent kernels per layer for Qwen3-0.6B and 1.7B (XX-T11, XVI-T8).

**Implementation steps.** XX-T1 to XX-T6, XX-T11; XIII-T14; XVI-T6, XVI-T8.

**Dependencies:** I-16, I-17, I-20.

**Tests.**
- 256 greedy tokens with K = 4 are bit-identical to K = 1 with a host sync after every step.
- Replaying 1,000 tokens equals 1,000 single-step submissions bit for bit.
- Under random CPU delays of 0–50 ms, no token is lost, duplicated or reordered over 10⁶ tokens.
- Generation stops at EOS with ≤ K − 1 wasted steps; a wasted 32B step costs < 3% of a real step; contexts 1…32K get the right split counts.

**Benchmark.** Median GPU gap between steps < 2 µs, against about 105 µs with a host sync (XX-T3). Qwen3-0.6B ≥ 1.3× mlx-lm at G4 and G5, with 1.5× as the target ([Part XXV](part25-roadmap-12-months.md) §16.1); Qwen3-8B ≥ 1.0×; ≤ 6 × L + 3 dispatches per token (XVI-T6).

**Definition of done.** G4's small-model criterion holds; XX-T1 to XX-T6, XX-T11, XIII-T14, XVI-T6 and XVI-T8 pass. CPU and energy instrumentation follows at G6 (XX-T12, in I-28).

**Links.** XX-T1 to XX-T6, XX-T11; XIII-T14; XVI-T6, XVI-T8; [Part XXIV](part24-roadmap-now.md) #9.

### I-19 — GQA decode attention (`flash_decode`) with a fixed split and a canonical combine

**Labels:** `gate:G4`, `kind:kernel`, `crate:sky-kernel`, `crate:sky-tir`

**Gate:** G4; the long-context matrix that checks it is owned by G5.

**Motivation.** MLX re-reads K/V for every query head below 8K context: Qwen3-8B's KV stream runs at about 97 GB/s, 53% efficient (R4), and MLX reaches 56% of peak at 8 KV heads at 32K on the M5 Max (mlx#3837). The long-context targets are ≥ 1.05× / 1.10× / 1.2× mlx-lm at 8K / 16K / 32K, bounded by about 1.08× / 1.14× / 1.24× ([Part XV](part15-attention.md) §8.2).

**Current Sky behavior.** None.

**Proposed architecture** ([Part XV](part15-attention.md) §4; [Part XI](part11-metal-backend.md) §9.5):
- The attention reference semantics and canonical reduction in the interpreter, with a row-independence probe per matrix path (XV-T1).
- Each KV head is read once for all of its query heads (G = 8 for Qwen3-32B).
- The context is split into blocks of a fixed size S, a plan constant shared by decode and verify. Each block produces a partial (max, sum, output), and `attn_combine` reduces the partials in block order, so results do not depend on how the context length is split (XV-T2).
- Two split sizes keyed by context bucket if one fixed S underuses the GPU at short context; the bucket is part of the plan.

**Implementation steps.** XV-T1; XV-T2 (`crates/sky-kernel/src/skeletons/flash_decode.rs`, `crates/sky-kernel/src/kernels/attn_combine.rs`); XI-T14.

**Dependencies:** I-05, I-09, I-14.

**Tests.** The interpreter matches a naive f64 softmax-attention within tolerance; `decode_verify_invariance`; bitwise invariance across M and across context lengths for a fixed query; each K/V byte read once per KV head, counted in a debug build (XI-T14).

**Benchmark.** On Qwen3-8B shapes on the M2 Pro: ≥ 85% of the measured read bandwidth at 4K–32K, and ≥ 1.5× MLX's attention speed at 4K (XV-T2). End to end: 8B decode at 4K context ≥ 1.05× mlx-lm, stretch 1.10× (Part XXIV #10). At G5, the long-context matrix at 4K–32K for both engines, with their KV-stream efficiency (XV-T12, XVI-T10, V-T6).

**Definition of done.** XV-T1 and XV-T2 pass; at G5, the per-context bounds of Part XV are met within 3%.

**Links.** XV-T1, XV-T2; XI-T14; XV-T12, XVI-T10, V-T6 (G5); [Part XXIV](part24-roadmap-now.md) #10.

### I-20 — GPU sampler: greedy, temperature, top-k, top-p, min-p, penalties, keyed Gumbel noise (Philox)

**Labels:** `gate:G4`, `kind:kernel`, `crate:sky-kernel`, `crate:sky-llm`

**Gate:** G4.

**Motivation.** Sampling on the CPU forces a host round trip per token, and in Sky's own CPU code it would cost 22.5 ms per token over Qwen3's 151,936 logits ([Part III](part03-current-performance.md) §7.3). Keyed Gumbel noise (seed, position) makes speculative output equal plain output at any temperature (SPINE D10).

**Current Sky behavior.** No sampler. Sky's CPU random numbers serve scientific code only.

**Proposed architecture** ([Part XIII](part13-llm-runtime.md) §9; [Part XVI](part16-decode.md) §6):
- An LM-head epilogue plus a finalize kernel; temperature, top-k, top-p, min-p, penalties and logit bias.
- top-p at k = 0 takes about 20 passes over the 152K floats instead of a sort per token; greedy skips log-softmax; ties go to the lowest id.
- Gumbel noise from Philox keyed by (seed, position), matching I-09's reference.

**Implementation steps.** XVI-T7 and XIII-T12 (`crates/sky-kernel/src/skeletons/sample.rs`, `crates/sky-llm/src/sample/`).

**Dependencies:** I-09, I-14.

**Tests.** χ² against the CPU reference over 10⁶ draws per filter; the same seed gives the same tokens; no CPU access to logits in the decode loop.

**Benchmark.** Sampling ≤ 1–3% of a token either way; fused sampling measured 0.7% on 8B in mlx-lm (R4).

**Definition of done.** The XVI-T7 and XIII-T12 acceptance tests pass.

**Links.** XVI-T7, XIII-T12, VII-T10; XXI-T4 (the acceptance kernels reuse this code at G6); [Part XXIV](part24-roadmap-now.md) #10.

### I-21 — Exactness suite: batch, chunk and verify-width invariance, and mlx-lm #1470 as an adversarial test

**Labels:** `gate:G4`, `kind:test`, `crate:sky-llm`, `crate:sky-tir`

**Gate:** G4; it becomes the G6 release gate with speculation on (XXI-T9).

**Motivation.** Exactness is the strongest technical lead available ([Part XXVII](part27-moat.md) §3.2). mlx-lm's speculative output departs from plain greedy at token 91 on Qwen3-32B with a 0.6B draft, on an exact logit tie, and the answer was documentation (#1470, #1592). The suite is the prerequisite for G6.

**Current Sky behavior.** Sky enforces byte-identical output across its three CPU engines (`engines_agree`, and differential fuzzing on 3,000 fixed seeds per `cargo test`; [Part II](part02-sky-today.md) §12). The discipline carries over; the rule becomes "deterministic per device, within tolerance of the oracle" (SPINE D6).

**Proposed architecture.**
- The logits of a sequence are bit-identical whether it is decoded alone, verified in a batch of M = 2…16, or prefilled in chunks of different sizes (IX-T7, `SCHED-INV-001`).
- The verifier rejects plans whose decode and verify instances declare different reduction schedules (V18, VII-T12).
- The spec section on numerics contracts, `inference` and `portable` (VI-T5).
- The #1470 reproduction (`benches/compare/llm/mlx_lm_1470.py`), started now and owned by G6: mlx-lm's divergence index is recorded, not asserted (V-T3).
- Speculative-equals-plain rows over prompts × draft lengths 1–16 × T ∈ {0, 0.6, 1.0}, written now and enabled by I-27 (XIII-T15, XXI-T9).

**Implementation steps.** IX-T7; VII-T12; VI-T5; the start of V-T3; CI wiring per pull request on the M2 Pro runner.

**Dependencies:** I-07, I-14, I-19, I-20.

**Tests.** The suite itself. For Qwen3-0.6B and 8B: logits at position p identical for M = 1…8 and for chunk sizes 16…2048 (IX-T7); the guard error and a mismatched-`ReduceSchedule` fixture as goldens (VII-T12).

**Benchmark.** The cost of invariance: end-to-end 8B decode with the invariant kernels against a specialized variant. Above 5%, exactness becomes opt-in (R-11, a G4 pivot).

**Definition of done.** The suite is green on the M2 Pro per pull request; VI-T5 is merged; the invariance cost is recorded in the risk register.

**Links.** IX-T7, VII-T12, VI-T5; V-T3, XIII-T15, XXI-T9 (G6); [Part XXIV](part24-roadmap-now.md) #10; [Part XXVIII](part28-risks.md) R-11.

### I-22 — `sky explain` per token, the performance model, `sky profile` and performance CI

**Labels:** `gate:G4`, `kind:compiler`, `crate:sky-cli`, `crate:sky-kernel`, `crate:sky-tir`

**Gate:** G4; the cost contracts and `sky explain --memory` are owned by G5, and the "why faster" comparison by G8.

**Motivation.** Explainability is a product feature (SPINE D12) and the source of the "why faster" evidence an outside reviewer needs ([Part XXVII](part27-moat.md) §5.5). The performance model decides where effort goes; it reproduces mlx-lm on the M2 Pro within −3.4% to +2.6% ([Part XXII](part22-performance-model.md) §5). Performance regressions must fail CI ([Part XXIII](part23-benchmarks.md) §12).

**Current Sky behavior.** There is no `sky explain` subcommand. `sky check --explain` merges rule-ID explanations from the checker and every pass. The cost model gives each function flops and bytes as intervals of `Poly`, with `ENTRY_BYTES = 8`, and contracts such as `cost flops <= 2*N^3` are proved, refuted with a witness, or warned about ([Part II](part02-sky-today.md) §10).

**Proposed architecture.**
- Explain records from every tensor pass (VI-T6); `sky explain --stage optimizer` (VIII-T10); the plan explain and the data-only plan record (IX-T10).
- Per-kernel cost from kernel IR as `Poly` in bucket parameters (XXII-T1); the plan predictor (XXII-T3); validation `sky bench llm --predict` (XXII-T4); predicted against measured per kernel in `explain` and `profile` (XXII-T5).
- `sky profile` with timestamps at encoder boundaries, its modes and a Chrome-trace export (XI-T15, XXIII-T12); the closure tool (XXIII-T4); the per-token report of Part XXIII §11.2 (XXIII-T13).
- Performance CI: `perf-pr`, `perf-nightly` and `perf-weekly` workflows with exact-counter gates and timing regressions (XXIII-T14).
- At G5: contracts on compiled models (`bytes_per_token`, `dispatches_per_token`, `memory`; XXII-T6, XIX-T9, XVI-T11), byte units in bounds (XIX-T10, a spec change), and `sky explain --memory` (XIX-T8).

**Implementation steps.** VI-T6; VIII-T10; IX-T10; XXII-T1, XXII-T3, XXII-T4, XXII-T5; XI-T15; XXIII-T4, XXIII-T12, XXIII-T13, XXIII-T14; at G5, XXII-T6, XIX-T8, XIX-T9, XIX-T10, XVI-T11.

**Dependencies:** I-16, I-17, I-18.

**Tests.**
- Byte counts equal `roofline.py --part22`'s tables for Qwen3-8B and 32B; the DRAM bytes of a decode plan equal the weights plus KV within 0.1% (XXII-T1).
- A deliberately mistuned GEMV shows a residual above 15% (XXII-T5); per-kernel attribution sums to within 5% of the step's command-buffer time (XI-T15).
- A pull request adding one dispatch per layer fails `perf-pr`; an injected 5% slowdown is flagged within 3 nights; ≤ 1 false alarm in 2 weeks of A/A runs (XXIII-T14).
- At G5: `sky check` proves `bytes_per_token <= 18.2 GB` for Qwen3-32B 4-bit and refutes 17.9 GB with a witness (XXII-T6); Part XIX §11.2's examples give their stated outcomes (proven at 4K with 19.67 GB planned; refuted at 32K with "holds for context ≤ 5,345").

**Benchmark.** Closure error e ≤ 5% for Sky on Qwen3-8B on the M2 Pro (XXIII-T4); the model reproduces Part XXII §5 within 1 percentage point (XXII-T4).

**Definition of done.** `sky explain` on Qwen3-8B lists every kernel with its reason, bytes, FLOPs, and predicted and measured time; performance CI gates pull requests.

**Links.** VI-T6; VIII-T10; IX-T10; XXII-T1, XXII-T3, XXII-T4, XXII-T5, XXII-T6; XI-T15; XXIII-T4, XXIII-T12, XXIII-T13, XXIII-T14; XIX-T8, XIX-T9, XIX-T10; XVI-T11.

## G5: lead on the M5 Max

### I-23 — Prefill GEMM on MPP `matmul2d` with dequantized tiles, the `simdgroup_matrix` fallback, and the `verify_mpp` probe

**Labels:** `gate:G5`, `kind:kernel`, `crate:sky-kernel`, `crate:sky-metal`

**Gate:** G5.

**Motivation.** On M1–M4, prefill has 1.25–1.4× of headroom over mlx-lm: MPP `matmul2d` reaches 5.42 TFLOPS fp16 (95% of peak) on the M2 Pro, where mlx-lm's end-to-end 8B prefill reaches 3.3 TFLOPS (58%) ([Part IV](part04-hardware-limits.md) §5). G5 gates prefill at ≥ 1.25× mlx-lm and ≥ 1.0× llama.cpp on M1–M4, and ≥ 1.0× mlx-lm on the M5, where both use the neural accelerators ([Part XXV](part25-roadmap-12-months.md) §16.1). The neural-accelerator path is fragile (R-10), so every pipeline is checked per OS and compiler.

**Current Sky behavior.** No GPU GEMM. Sky's CPU matmul loses 2× to the FMA ban and about 7.5× more to AMX ([Part III](part03-current-performance.md), summary); it is irrelevant to this work.

**Proposed architecture** ([Part XVII](part17-prefill.md) §3; [Part XI](part11-metal-backend.md) §8.4, §9.3–§9.4):
- `mpp_qgemm` with tiles of 64×64 or larger: Apple's sample 64×32 tile reaches about half of peak (R6) (XVII-T1).
- Codecs dequantize MLX affine (2–8 bits, g32/64/128), Q4_0, Q8_0, Q4_K, Q6_K and MXFP4 into fp16 threadgroup tiles, double-buffered (XVII-T2).
- A `half × int4b` variant with a per-group post-scale on OS 26.4+ (XVII-T3, XII-T11); the faster one becomes the default per chip.
- The `sg_qgemm` fallback on `simdgroup_matrix`, never selected on GPU generation ≥ 17 (XVII-T4, XI-T13).
- The `verify_mpp` probe: bitwise row invariance and small-M streaming on each chip, which IX-T8 and XXI-T2 consume (XI-T12).
- The compiler-regression self-check: digests per (OS build, compiler) (XI-T9).

**Implementation steps.** XVII-T1 to XVII-T4; XII-T11; XI-T12, XI-T13; XI-T9.

**Dependencies:** I-05, I-06, I-09.

**Tests.** Results within 1e-3 relative of an fp32 reference; a dequantized tile bit-equal to `dequant_ref`; the fallback runs with MPP disabled; the row-invariance probe reports pass or fail per chip; a deliberately perturbed kernel is caught by the self-check in ≤ 2 s for a Qwen3-32B plan.

**Benchmark.** M2 Pro: ≥ 5.2 TFLOPS fp16 at 4096³ (MPP's best is 5.42; XVII-T1); the quantized path ≥ 0.9× the fp16 rate at the same shapes (XVII-T2) and ≥ 85% of MPP fp16 throughput, at least 4.6 of 5.42 TFLOPS (XI-T12); the fallback ≥ 85% of peak, where MLX's Steel GEMM reaches 85–88% (XVII-T4). M5 Max: ≥ 80% of the neural-accelerator peak measured by I-04, on Qwen3-32B MLP shapes (XVII-T1).

**Definition of done.** The XVII-T1 to XVII-T4, XI-T9, XI-T12 and XI-T13 acceptance tests pass; the per-chip default is recorded.

**Links.** XVII-T1 to XVII-T4; XI-T9, XI-T12, XI-T13; XII-T11; [Part XXVIII](part28-risks.md) R-10.

### I-24 — The GGUF path for same-file pairs with llama.cpp: reader, direct mapping, and the Q4_0, Q8_0, Q4_K and Q6_K codecs

**Labels:** `gate:G5`, `kind:kernel`, `crate:sky-quant`, `crate:sky-llm`, `crate:sky-kernel`

**Gate:** G5 (moved from G7 for the Qwen3 files; [Part XXV](part25-roadmap-12-months.md) §16.1, item 2).

**Motivation.** G5's "≥ 1.0× llama.cpp" can only be measured on llama.cpp's own file, because every comparison is a same-file pair (D5); no lossless conversion exists between MLX 4-bit and the GGUF 4-bit files at equal size (R9). llama.cpp's K-quant decode is less efficient than its Q4_0 decode (Q4_K 147 GB/s and MXFP4 138 GB/s against 177), so Q4_K_M files leave about 1.15–1.2× of headroom ([Part V](part05-competitors.md) §12).

**Current Sky behavior.** No GGUF code.

**Proposed architecture** ([Part XII](part12-quantization.md) §4.2, §6.3; [Part XIII](part13-llm-runtime.md) §6.4):
- A GGUF reader enforcing exact padded offsets, with per-tensor formats and layout flags; GGUF is 32-byte aligned, so it is mapped directly with no relayout (SPINE D9) (XII-T4, XIII-T8). Llama GGUF files get `RopeLayout::Interleaved`.
- Decode codecs that read GGUF's native blocks in place (XVI-T9; XII-T7's GGUF codecs), on the `qgemv` skeleton of I-06.
- Prefill through I-23's tile loader, which already dequantizes Q4_0, Q8_0, Q4_K and Q6_K (XVII-T2).
- The plane-split relayout for K-quants (XII-T9) and the P1 codecs stay in G7.

**Implementation steps.** XII-T4 / XIII-T8; XVI-T9; then the same-file decode and prefill runs against llama.cpp.

**Dependencies:** I-06, I-09, I-11, I-23.

**Tests.** The reader reproduces R9 §2.3's tensor-type histograms for the three 8B and 0.6B files (4.67 and 4.90 bits per weight) and rejects IQ2 with Part XII §10.1's message; GPU dequantization equals `dequant_ref` bit for bit (`codec_exact.rs`); mapping a GGUF file does not grow RSS by its size.

**Benchmark.** Q4_0 ≥ llama.cpp's 177 GB/s on R5's shape; Q4_K ≥ 165 GB/s against llama.cpp's 147 (XVI-T9); same-file decode and pp2048 against llama.cpp on the M2 Pro and the M5 Max (XVII-T9).

**Definition of done.** XII-T4, XIII-T8 and XVI-T9 pass, and G5's llama.cpp prefill comparison runs on Q4_0 and Q4_K_M files.

**Links.** XII-T4, XII-T7 (GGUF codecs), XIII-T8, XVI-T9, XVII-T2; [Part XXV](part25-roadmap-12-months.md) §16.1; SPINE D5, D9.

### I-25 — Prefill pipeline: fused prefill layer, back-to-back chunks, `flash_prefill`, chunk invariance, and token 1 from the last chunk

**Labels:** `gate:G5`, `kind:runtime`, `crate:sky-llm`, `crate:sky-kernel`

**Gate:** G5.

**Motivation.** Most of the M1–M4 prefill gap lies outside the GEMM: chunk syncs, cache clears, attention and norms ([Part IV](part04-hardware-limits.md) §5). llama.cpp already prefills 1.35–1.42× faster than mlx-lm on the M2 Pro ([Part V](part05-competitors.md) §12). mlx-lm also runs the last prompt token as an extra decode step, a full weight pass worth about 20% of TTFT at 128 tokens (R4; [Part XVIII](part18-ttft.md) §4.6).

**Current Sky behavior.** None beyond I-14's unfused prefill.

**Proposed architecture** ([Part XVII](part17-prefill.md) §4–§8; [Part XV](part15-attention.md) §6):
- Five kernels per layer per chunk, with QK-norm, RoPE and the KV write in the Q‖K‖V epilogue (XVII-T5).
- Back-to-back chunk submission with no CPU wait; the LM-head row and sampling run in the last chunk, which also initializes `StepState` (XVII-T7, XVIII-T6).
- FA2 `flash_prefill` with causal skip and GQA grouping, in `simdgroup_matrix` and MPP variants (XV-T6).
- The chunk-invariance test (XVII-T6); a command-buffer time budget with re-chunking for the display watchdog (XI-T17).
- The explanation of the M1–M4 gap from per-op GPU time of mlx-lm and llama.cpp at 512 and 2,048 tokens guides the order of work (V-T7).

**Implementation steps.** V-T7; XVII-T5, XVII-T6, XVII-T7; XVIII-T6; XV-T6; XI-T17; then the benchmark matrix (XVII-T9).

**Dependencies:** I-16, I-19, I-23.

**Tests.**
- A 4,096-token prompt prefilled with chunk sizes {64, 192, 512, 2,048, 4,096} gives a bit-identical KV cache and last-row logits.
- No 1-token pass between prefill and token 0 on the GPU timeline; token 0 equals a reference that runs the last token separately, within tolerance, and is identical across chunk sizes.
- A real 32K-token prefill runs without watchdog errors with a display attached on the CI runner.

**Benchmark.** Gaps between chunks < 5 µs; TTFT at 128 tokens on Qwen3-8B / M2 Pro ≥ 1.2× mlx-lm (XVII-T7); `flash_prefill` ≥ 1.0× MLX `steel_attention` at 2K–8K on Qwen3-8B shapes (XV-T6); pp128, pp512, pp2K and pp8K against mlx-lm and llama.cpp on the M2 Pro (8B) and the M5 Max (32B) (XVII-T9). V-T7's breakdown must account for ≥ 90% of the measured 1.35–1.42× difference.

**Definition of done.** G5's prefill targets on M1–M4 hold at pp2048: ≥ 1.25× mlx-lm and ≥ 1.0× llama.cpp (the latter on I-24's GGUF file).

**Links.** XVII-T5, XVII-T6, XVII-T7, XVII-T9; XVIII-T6; XV-T6; XI-T17; V-T7.

### I-26 — Autotuner v1: GPU records in the tuning database, honest timing, verification and optimization levels

**Labels:** `gate:G5`, `kind:compiler`, `crate:sky-tune`, `crate:sky-hw`

**Gate:** G5.

**Motivation.** G5 includes "autotuner v1". Expected gains are 0–5% on bandwidth-bound GEMVs and 5–30% on small-M prefill GEMMs and attention (R7 §10.2). R-13's kill criterion applies here: if search improves the hot kernels by less than 5% over hand-picked defaults on small-M prefill and verify shapes, the general layers freeze.

**Current Sky behavior.** `sky-tune` has a data-only database (`sky-tune-db 1`) keyed by (fingerprint, kernel, size class), a 500 ms budget per tunable, median scoring, a ≥ 3% rule for winners, and a bound of 4,096 records and 1 MiB. It tunes CPU kernels only ([Part II](part02-sky-today.md) §9). I-05 adds the GPU fingerprint.

**Proposed architecture** ([Part X](part10-autotuner.md) §2–§7):
- GPU records with the new keys `hw`, `os`, `contract`, `order`, `check` and `err`; the format version stays 1 (X-T2).
- GPU tunables with order-preserving and order-changing parameters, invariance groups, chain-slope timing on GPU timestamps, buffer rotation, warm-up and interleaving (X-T3). This fixes tinygrad's BEAM mis-ranking: no cache flush, minimum-of-3 timing, device-name-only keys (R2 §6.4).
- A verification harness (X-T4); pruning with analytic guards, branch-and-bound and Ansor-style budget allocation (X-T5); a DRAM-floor check (X-T6).
- Block-level tuning between the head-aligned K1 and P1b partitions (X-T7).
- Optimization levels `-O1` (heuristics), `-O2` (cached database), `-O3` (tune missing entries) and `--autotune=overnight` (X-T8); attention tunables with a bit-neutrality test (XV-T10).

**Implementation steps.** X-T2 to X-T8; XV-T10.

**Dependencies:** I-05, I-16, I-22.

**Tests.**
- Today's `parse_result` loads a GPU line without warnings and never matches it; bounds fuzzed for 10⁶ inputs.
- Three re-tunings of one instance give the same winner, or winners within 3% of each other.
- A kernel that writes nothing, one that hard-codes outputs and one with an off-by-one tail are rejected; an order-changing candidate with M-dependent rows is rejected.
- Re-creating tinygrad's mistake, a 28 MB matrix timed warm, is flagged.
- `-O2` never writes the database (the file is unchanged byte for byte).

**Benchmark.** Tuning gain over analytic defaults per kernel class on the M2 Pro and the M5 Max ([Part X](part10-autotuner.md) §14); on a recorded search, pruning keeps the true winner in ≥ 95% of instances while compiling ≤ 30% of candidates (X-T5).

**Definition of done.** X-T2 to X-T8 and XV-T10 pass, and the R-13 measurement is recorded in the risk register.

**Links.** X-T2 to X-T8; XV-T10; [Part XXVIII](part28-risks.md) R-13.

## G6: exact speculation and TTFT

### I-27 — Exact speculation engine: verify plans, GPU accept/reject/rewind, keyed Gumbel and rejection sampling, rounds without a host sync; prompt-lookup drafting first

**Labels:** `gate:G6`, `kind:runtime`, `crate:sky-llm`, `crate:sky-kernel`

**Gate:** G6.

**Motivation.** Speculation is the pitch (SPINE D18). mlx-lm synchronizes the host every round (`mx.eval` plus two `.tolist()` calls, `generate.py:620-622`), re-feeds the last token, and accepts by sample-and-match (Σ p·q), which accepts 50% where rejection sampling (Σ min(p, q)) accepts 100% at (0.5, 0.5) ([Part XXI](part21-speculative.md) §6, §8; R4 §2.3). Prompt lookup is the cheapest drafter and exercises the whole engine without a draft model.

**Current Sky behavior.** None.

**Proposed architecture** ([Part XXI](part21-speculative.md) §6–§9; [Part XX](part20-persistent-execution.md) §6):
- Speculation hooks in the generation loop (XIII-T15).
- Acceptance kernels: keyed Gumbel and rejection sampling with residual resampling (XXI-T4).
- Commit kernels: the KV length update, tree-path compaction and draft-cache rewind; GPU-resident length counters (XXI-T5, XIV-T3, XIV-T7).
- Verify plans on the ALU family, M = 2…8 (I-07), with verify attention for chains (XV-T3).
- A round as a program: rounds pre-committed, stop flags, the ring buffer (XX-T10, XXI-T11).
- Prompt-lookup and suffix-tree drafting on the CPU, with one event-driven CPU step per round, about 0.1 ms (XXI-T7).

**Implementation steps.** XIII-T15; XXI-T4, XXI-T5; XIV-T3, XIV-T7; XV-T3 (chains); XX-T10; XXI-T11; XXI-T7; enable I-21's speculative rows (XXI-T9).

**Dependencies:** I-07, I-18, I-20, I-21.

**Tests.**
- Self-draft acceptance is 100% with keyed Gumbel at every temperature; rejection mode passes χ² on 10⁶ draws.
- The KV cache is byte-equal after n rounds of speculation and after plain decoding of the same tokens; `kv_rewind_exact` over 1,000 random accept patterns.
- Greedy output is bit-identical to plain decoding on 20 prompts × 128 tokens; with a random drafter, output is byte-identical to plain at T = 0, 0.6 and 1.0 over 100 prompts × draft lengths 1–16 (XIII-T15).

**Benchmark.** The GPU timeline shows < 5 µs between rounds, except CPU-drafter rounds (XXI-T11). Prompt lookup gives ≥ 2× effective tok/s on the summarization and code-editing workloads, greedy, and ≈ 1.0× (never < 0.97×) on fresh text (XXI-T7).

**Definition of done.** XIII-T15, XXI-T4, XXI-T5, XXI-T7, XXI-T9 and XX-T10 pass; the exactness suite is green with speculation on, per pull request on the M2 Pro and nightly on the M5 Max.

**Links.** XIII-T15; XXI-T4, XXI-T5, XXI-T7, XXI-T9, XXI-T11; XX-T10; XIV-T3, XIV-T7; XV-T3.

### I-28 — Draft-model speculation (Qwen3-0.6B) and the comparison with mlx-lm's speculative mode

**Labels:** `gate:G6`, `kind:runtime`, `crate:sky-llm`

**Gate:** G6.

**Motivation.** On the M5 Max, MLX's own verify cost is already modest: β ≈ 0.06–0.18 per drafted token, from mlx#3791's +12% / +48% / +125% at 3 / 5 / 8 tokens, so Sky's kernel advantage there is about 1.2×. The bar is ≥ 1.2× mlx-lm's own speculative mode with the same draft: below 1.2× the plan pivots, and below 1.1× the speculative engine stops being presented as a differentiator ([Part XXVIII](part28-risks.md) R-3). mlx-lm's 0.6B draft is CPU-bound at about 1.94 ms per draft token on the M2 Pro (R4 §7.E).

**Current Sky behavior.** None.

**Proposed architecture** ([Part XXI](part21-speculative.md) §4; [Part XX](part20-persistent-execution.md) §7):
- The draft model as a static plan with ≤ 5 dependent kernels per layer (I-18's small-model plan), an optional truncated draft vocabulary, and its own KV (XXI-T6).
- Drafter memory: the embedding shared with the target when the hashes match (XIX-T12).
- The acceptance study: τ per drafter, workload, temperature, thinking mode and acceptance rule, replacing every assumed τ (XXI-T15).
- CPU time per token, wakeups per second and J/token by R9's method (XX-T12), for both engines.

**Implementation steps.** XXI-T6; XIX-T12; XX-T12; XXI-T15; the R-3 measurement on the M5 Max.

**Dependencies:** I-27.

**Tests.** The exactness suite stays green; draft-step timing from GPU timestamps; the energy method matches R9's protocol on a known workload (XX-T12).

**Benchmark.** c ≤ 0.035 for the 0.6B draft against a 32B target on the M5 Max (XXI-T6). ≥ 1.2× mlx-lm's speculative mode with the same 0.6B draft (XXI-T11; SPINE §3), with the same prompts at T = 0 and 0.6. On the M2 Pro, mlx-lm's speculation is the baseline at 0.37–0.74× of plain (R4).

**Definition of done.** XXI-T6, XXI-T15 and XX-T12 pass; R-3 and R-12 are measured and recorded in the risk register.

**Links.** XXI-T6, XXI-T11, XXI-T15; XIX-T12; XX-T12; [Part XXVIII](part28-risks.md) R-3, R-12.

### I-29 — EAGLE-3 drafter: converter, runtime, and tree verification sized from measured β

**Labels:** `gate:G6`, `kind:runtime`, `crate:sky-llm`, `crate:sky-quant`, `crate:sky-kernel`

**Gate:** G6.

**Motivation.** mlx-lm cannot run EAGLE-3: it supports separate draft models only, and no MLX-format EAGLE-3 weights exist (R4, R8). The public `RedHatAI/Qwen3-32B-speculator.eagle3` and `AngelSlim/Qwen3-32B_eagle3` reach τ = 2.29–3.08 at T = 0.6 (R8 §2.3), which the model turns into 47–59 tok/s typical on Qwen3-32B / M5 Max at β ≤ 0.05 (SPINE §3).

**Current Sky behavior.** None.

**Proposed architecture** ([Part XXI](part21-speculative.md) §4–§5; [Part XV](part15-attention.md) §5):
- A converter and runtime for the Red Hat and AngelSlim checkpoints: hidden-state taps from three target layers, the draft KV, the 32K → 151K vocabulary map; drafter quantization in `sky-quant`, with its τ loss measured (XXI-T8).
- Verify attention with tree bitmasks and a per-node window gather. A mask-only tree is rejected, because a term's position within a block changes its sum (XV-T3, XXI-T3).
- A tree sizer from the measured β(M, context) and acceptance, never from the ridge number (XXI-T10, X-T9, XXII-T7).
- The neural-accelerator tier only if G5's `verify_mpp` probe (I-23) and XXI-T2 pass; otherwise ALU-width trees (IX-T8).

**Implementation steps.** XXI-T8; XV-T3 (trees) and XXI-T3; XXI-T10; X-T9; XXII-T7.

**Dependencies:** I-23, I-27, I-28.

**Tests.** `tree_verify_exact`; `spec_equals_plain` on Qwen3-0.6B and 8B; τ within 5% of the Red Hat card at the card's settings (bf16 drafter); on recorded acceptance traces, the chosen γ is within one step of the measured best.

**Benchmark.** ≥ 1.7× plain on the M5 Max at β ≤ 0.05 (XXI-T8, G6). Trees are kept on a chip only if they beat the best chain by ≥ 10% (XXI-T10). On the declared suite at T = 0.6 with EAGLE-3: a suite average below 1.5× of Sky's plain decode restricts claims to the classes where ≥ 1.7× holds; below 1.2× on every class, the speculative headline stops (R-12).

**Definition of done.** XXI-T8 passes on Qwen3-8B (M2 Pro) and Qwen3-32B (M5 Max); R-12 is measured and recorded.

**Links.** XXI-T3, XXI-T8, XXI-T10; XV-T3; X-T9; XXII-T7; inputs from G5: XXI-T2, IX-T8, XV-T5; [Part XXVIII](part28-risks.md) R-4, R-12.

### I-30 — TTFT: exact prefix cache, data-only plan cache and the AOT `.skybin`

**Labels:** `gate:G6`, `kind:runtime`, `crate:sky-llm`, `crate:sky-metal`

**Gate:** G6; the persistent prompt cache (XVIII-T9) is owned by G7.

**Motivation.** G6 includes "AOT `.skybin`; prompt cache". The TTFT targets are: cold start (process start to first token, weights in the page cache) ≥ 3× better than mlx-lm; a repeated prefix ≥ 1.3× better with both engines' caches on, about 2× at ≤ 8 new tokens; warm TTFT ≥ 1.15× (SPINE §3; [Part IV](part04-hardware-limits.md) §7). llama.cpp's first shader compile after install took 4.8 s (R5), and mlx-lm reads every weight byte with `pread` (R3).

**Current Sky behavior.** `sky build` embeds SIR and its interpreter in an executable, the precedent for `.skybin`; Sky has no code cache ([Part II](part02-sky-today.md) §6.4–§6.5). Sky's AOT executables start in about 4 ms ([Part XVIII](part18-ttft.md) §4.1).

**Proposed architecture** ([Part XVIII](part18-ttft.md) §4–§8; [Part XIII](part13-llm-runtime.md) §13; [Part XIV](part14-kv-cache.md) §10):
- The plan cache stores the execution plan as validated, bounded data: kernel choices, parameters and offsets, never MSL or binaries. MSL is regenerated deterministically on load, and the OS shader cache reuses the compiled code (XIII-T16, XVIII-T5; SPINE D19).
- `.skybin` carries Metal binary archives only when built by `sky compile --archive`, and uses them only when the OS and target keys match (XI-T8).
- A startup trace and parallel startup (XVIII-T1, XVIII-T2); progressive residency in layer order (XVIII-T3); one-time text tables (XVIII-T4); a first-run weight strategy (XVIII-T7).
- The prefix cache: a radix tree over KV blocks with an LRU byte budget and refcounted reuse (XVIII-T8, XIV-T8).

**Implementation steps.** XIII-T16; XI-T8; XVIII-T1, XVIII-T2, XVIII-T3, XVIII-T4, XVIII-T5, XVIII-T7, XVIII-T8; XIV-T8.

**Dependencies:** I-11, I-18, I-21, I-25.

**Tests.**
- A test scans every plan-cache record type for MSL or binaries; each key component invalidates exactly what Part XIII §13.2 says; a corrupted offset is rejected without a panic (fuzzed); a second run regenerates byte-identical MSL.
- Prefix hit = miss, bitwise, for prefill-written prefixes; restore-and-continue equals the uninterrupted session byte for byte on 32 multi-turn transcripts; a changed numerics key misses; a corrupted `.skykv` file is detected by its sha256 and ignored.
- A counting allocator shows no allocation of vocabulary size or more on the second request (XVIII-T4).

**Benchmark.** A second process start creates all pipelines in ≤ 50 ms on the M2 Pro (XVIII-T5); warm start ≤ 100 ms for a Qwen3-8B plan (XI-T8, target); cold TTFT ≥ 3× mlx-lm; repeated prefix against mlx-lm with both caches on; first-token latency after 1, 8 and 30 s gaps within 5% of back-to-back (XVIII-T3); first-run TTFT ≤ 1.1× mlx-lm's load-inclusive TTFT (XVIII-T7).

**Definition of done.** XIII-T16, XI-T8 and XVIII-T1 to XVIII-T5, XVIII-T7 and XVIII-T8 pass; the SPINE §3 TTFT targets are measured on the M2 Pro and the M5 Max.

**Links.** XIII-T16; XI-T8; XVIII-T1, XVIII-T2, XVIII-T3, XVIII-T4, XVIII-T5, XVIII-T7, XVIII-T8, XVIII-T9 (G7); XIV-T8.

## Filing these issues

The snippet below is for reference only; it has not been run. It does four things:
1. splits this file into one body per issue (`issues/I-01.md` … `issues/I-30.md`);
2. creates the labels;
3. files each issue with `gh issue create`, in gate order;
4. replaces the `I-NN` references with the real GitHub numbers, which exist only after filing.

The title comes from each `### I-NN — …` heading and the labels from its `**Labels:**` line.

```sh
#!/usr/bin/env bash
# File the 30 issues of docs/inference/issues.md. Review before running.
# Needs bash >= 4 (associative arrays): on macOS, `brew install bash`.
set -euo pipefail

repo="${REPO:?set REPO=owner/name}"
src="${1:-docs/inference/issues.md}"
dir="${2:-issues}"
mkdir -p "$dir"

# 1. Split: each "### I-NN — Title" heading starts a body file; any "## " heading ends one.
awk -v dir="$dir" '
  /^## /                  { out = "" }
  /^### I-[0-9][0-9] — /  { out = dir "/" substr($2, 1, 4) ".md" }
  out != ""               { print > out }
' "$src"

# 2. Labels (re-running is harmless).
for l in gate:G0 gate:G1 gate:G2 gate:G3 gate:G4 gate:G5 gate:G6 \
         kind:measurement kind:infra kind:kernel kind:compiler kind:runtime kind:test \
         crate:sky-bench crate:sky-metal crate:sky-kernel crate:sky-tir crate:sky-quant \
         crate:sky-llm crate:sky-hub crate:sky-cli crate:sky-tune crate:sky-hw; do
  gh label create "$l" --repo "$repo" >/dev/null 2>&1 || true
done

# 3. File the issues in order and remember their numbers.
declare -A num
for f in "$dir"/I-[0-9][0-9].md; do
  id=$(basename "$f" .md)
  title=$(sed -n '1s/^### I-[0-9][0-9] — //p' "$f")
  labels=$(sed -n 's/^\*\*Labels:\*\* //p' "$f" | tr -d '`' | sed 's/, */,/g')
  url=$(tail -n +2 "$f" | gh issue create --repo "$repo" --title "$title" \
          --label "$labels" --body-file -)
  num[$id]=${url##*/}
  echo "$id -> #${num[$id]}  $title"
done

# 4. Rewrite I-NN into real issue numbers. The tokens are fixed-width (I-01 … I-30),
#    so no token is a prefix of another.
for f in "$dir"/I-[0-9][0-9].md; do
  id=$(basename "$f" .md)
  body=$(tail -n +2 "$f")
  for k in "${!num[@]}"; do body=${body//"$k"/"#${num[$k]}"}; done
  printf '%s\n' "$body" | gh issue edit "${num[$id]}" --repo "$repo" --body-file -
done
```

Notes:
- `I-NN` tokens are used instead of `#NN` because the bodies also cite upstream issues, such as mlx-lm #1470 and llama.cpp #10119, which a blind `#NN` rewrite would corrupt.
- Step 3 files the issues in the order of this file, so GitHub's numbers follow the gate order.
