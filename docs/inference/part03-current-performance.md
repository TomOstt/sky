# Part III. Current performance: where Sky wins, where it loses, and why

This part measures the Sky compiler described in [Part II](part02-sky-today.md) on the CPU, against C, Rust, Apple's Accelerate and CPython. It explains each loss down to the code that causes it, lists the fixes ("Roadmap A", F1–F11) with their expected gains, gives an honest verdict on the owner's example CPU gates, and says which of this matters for LLM inference on Apple GPUs. Decision D16 ([Part VI](part06-architecture.md)) makes this CPU work secondary to the inference programme; this part supplies the evidence for that decision.

**Sources.** [R10](research/R10-sky-cpu-perf.md) (measurements and fixes; repository at `a99e7e3`, measured 2026-10-01/02), [R1](research/R1-sky-today.md) (probes), `benches/compare/RESULTS.md` (the project's published table) and [`roofline.py`](roofline.py) for decode-step times. No benchmark was re-run for this chapter. File names such as `r10/m5.log` are R10's raw logs and scripts, listed in R10 §9. Labels follow the report's rules.

**Machine and toolchain** (R10 §1): Apple M2 Pro, 10 cores (6P + 4E), 16 GB, macOS 26.6.1; Rust 1.85.1 (the repository's pinned toolchain), Apple clang 17.0.0. Every timed command ran under a lock against other measurement jobs; medians are reported, minimums where medians were noisy (flagged).

## Summary

| # | Finding | Evidence |
|---:|---|---|
| 1 | The published ratios time whole processes. Sky's fixed cost before any work is 4.7–5.7 ms (C: 1.45 ms), and native runs add Cranelift compile time. Per unit of work the picture changes: nbody is 3.2× C, not 7.5×; fib is 2.7× C, not 1.36×. | [VERIFIED: R10 §0, §3] |
| 2 | The worst gap is missing from the suite: `x[i]` on a large array costs **148 ns** in native code, **57× C**, because every operation on an array of more than 16 entries is a boxed call into the interpreters' runtime. | [VERIFIED: R10 §4.4] |
| 3 | Fused elementwise kernels already run at C speed (axpy 1.00–1.01×, hypot 1.00×; a six-operation chain with a division 1.14–1.17×). | [VERIFIED: R10 §4.5] |
| 4 | Native scalar loops lose to C on checks, SIMD lane shuffles and register spills, not on allocation: in nbody's loop 36% of instructions are checks, 19% lane shuffles, 14% spills, 23% arithmetic. | [VERIFIED: R10 §4.1, disassembly census] |
| 5 | Neither the SIR pipeline nor Cranelift 0.121.2 has an inliner; a native call costs 2.0 ns against C's 0.74 ns. | [VERIFIED: R10 §4.3] |
| 6 | Matmul loses 2× to the FMA ban and about 7.5× more to Apple's AMX unit; Sky's reproducible kernel is already at 89% of its own no-FMA ceiling. | [VERIFIED: R10 §4.7] |
| 7 | Accelerate's `cblas_dgemm` is bit-identical to the sequential k-order FMA chain on GEMM shapes on this machine, which opens a reproducible path to AMX speed; it fails for tiny products and matrix-vector. | [EXPERIMENTAL: one machine and OS; R10 §4.9] |
| 8 | Almost none of this matters for LLM tokens per second. At 148 ns per element, one CPU pass over Qwen3's 151,936 logits costs 22.5 ms per token: more than a whole decode step of Qwen3-8B on the M2 Pro at its bandwidth ceiling. Sampling must run on the GPU. | [VERIFIED: arithmetic on R10 §4.4 and `roofline.py`] |

---

## 1. The benchmark suite, and what its ratios mean

### 1.1 The suite

| Item | Value |
|---|---|
| Cases | 13, in `benches/compare/{sky,c,rust,python,numpy,lib}` |
| Harness | `crates/sky-bench/src/compare/{mod.rs,workloads.rs,tools.rs,report.rs}`; command `cargo run --release -p sky-bench -- compare` |
| Output | `benches/compare/RESULTS.md`, generated 2026-10-01 at revision `88edceb83a66` |
| Method | each cell is a separate program run as a subprocess: 1 untimed warmup, 5 timed repetitions, **median wall time of the whole process**; a cell whose result differs from the reference beyond the workload's tolerance is marked and excluded |
| C | Apple clang `-O3 -march=native`, not given `-ffp-contract=off`, so clang may contract `a*b+c` into FMA (C's mandelbrot prints 7971151 where Rust and Sky print 7992339) |
| Rust | `-C opt-level=3 -C target-cpu=native`; Rust never contracts |
| CPython | pure CPython 3.14.0; NumPy was not installed, so that column is empty |
| Libraries | Accelerate (`cblas_dgemm`, LAPACK `dpotrf`/`dpotrs`) |
| Sky | release `sky run` on engines ast, sir, native, auto |

[VERIFIED: R10 §2; `RESULTS.md` header]

So C is allowed FMA contraction in scalar code while Sky's default forbids it; this biases scalar floating-point rows slightly towards C [LIKELY: clang's default `-ffp-contract=on`; corroborated by the different mandelbrot output]. Rust has Sky's semantics, which makes it the fairer scalar baseline.

### 1.2 Published results

Median wall time of the whole process, in ms unless marked [VERIFIED: copied from `RESULTS.md` by R10 §2.1]:

| Case | C -O3 | Rust | BLAS/LAPACK | CPython | Sky ast | Sky sir | Sky native | Sky auto |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nbody (10^5 steps) | 4.58 | 5.97 | – | 610 | 2714 | 1536 | 34.3 | 34.3 |
| matmul-64 ×100 | 8.66 | 8.50 | 6.03 | 1162 | 7.95 | 8.80 | 9.82 | 9.77 |
| matmul-256 ×8 | 21.5 | 21.8 | 3.92 | 6134 | 11.3 | 12.2 | 13.5 | 13.5 |
| matmul-512 ×2 | 42.3 | 41.6 | 4.95 | 12.3 s | 14.1 | 15.7 | 16.1 | 16.1 |
| cholesky-256 ×10 | 16.5 | 18.7 | 8.17 | 1108 | 14.1 | 15.7 | 16.4 | 16.2 |
| cholesky-32 ×5000 | 15.5 | 17.7 | 10.1 | 2108 | 39.9 | 40.1 | 41.0 | 40.2 |
| elementwise-1e6 ×20 | 15.8 | 15.3 | – | 2110 | 241 | 25.2 | 24.4 | 24.1 |
| linear_chain-1e6 ×20 | 10.4 | 12.6 | – | 1571 | 156 | 19.4 | 19.4 | 19.5 |
| elementwise-1e7 ×2 | 41.1 | 40.1 | – | 2651 | 320 | 84.9 | 97.0 | 94.6 |
| linear_chain-1e7 ×2 | 36.5 | 37.7 | – | 2129 | 208 | 61.4 | 69.4 | 68.7 |
| mandelbrot 300², 500 iterations | 24.8 | 26.6 | – | 912 | 13.4 s | 11.7 s | 39.2 | 38.9 |
| rk4 2·10^5 | 14.3 | 14.4 | – | 234 | 1330 | 942 | 20.3 | 20.0 |
| fib-30 | 8.42 | 8.45 | – | 124 | 718 | 1622 | 11.5 | 11.9 |

Floors (an empty program): C process 1.45 ms; `sky run` ast 4.82, sir 5.15, native 5.58, auto 5.32 ms. Headline geometric means for the best Sky engine: 1.40× C, 1.32× Rust, 2.36× BLAS/LAPACK (five cases), 0.02× CPython. [VERIFIED: `RESULTS.md`]

### 1.3 What these ratios actually measure

A whole-process median mixes four things that are not the workload:

| Fixed cost | Size | Evidence |
|---|---|---|
| process start, lex, parse, check | 4.7–5.7 ms for `print(1)`; 5.3 ms for `nbody.sky`; **16.1 ms** for `elementwise-1e7.sky`, whose inputs are 290 KB of literals | [VERIFIED: R10 §4.8] |
| SIR lowering and the pass pipeline | about 5 ms for nbody | [VERIFIED: R10 §4.8] |
| CLIF generation and Cranelift compilation | **about 12 ms** for nbody (an energy function plus a fully unrolled, SLP-packed main loop of about 4,400 CLIF lines) | [VERIFIED: R10 §4.8] |
| building the inputs | about 70 of the 93 ms of elementwise-1e7 (front end plus three outer products) | [VERIFIED: R10 §3.2, `r10/m3.log`] |

C has fixed costs too: about 33 ms of elementwise-1e7's C time is initializing two 80 MB arrays with page faults [VERIFIED: R10 §3.2]. A ratio of two such totals says little about either compiler.

### 1.4 The slope method

R10 re-measured each workload at two sizes N1 < N2 and took the slope:

```text
per-unit cost  = (t(N2) − t(N1)) / (N2 − N1)
fixed cost     = t(N1) − N1 · per-unit cost
```

The slope cancels process start-up, the front end, compilation and input construction. It is the same idea as the decode metric of D13 ([Part XXIII](part23-benchmarks.md)): decode speed is `(N − 1) / (t_last − t_first)`, never total tokens over total time.

| Workload | `RESULTS.md` (whole process) | Slope (per unit of work) |
|---|---:|---:|
| nbody vs C | 7.5× | **3.2×** per step |
| fib vs C | 1.36× | **2.7×** per call |
| mandelbrot vs Rust (identical semantics) | 1.46× | **1.20×** per inner iteration |
| rk4 vs C | 1.39× | **1.89×** per step |
| elementwise-1e7 vs C | 2.06× | **0.6×** per pass on 10 threads; **1.17–1.36×** on 1 thread |

[VERIFIED: R10 §0, §3]

The table cuts both ways: whole-process ratios overstate nbody's gap (two thirds of its 32 ms is fixed cost) and understate fib's (the floor dilutes a 2.7× per-call gap to 1.36×).

### 1.5 The harness's generated explanations

`RESULTS.md` ends with an "Interpretation" per workload kind. The text is canned per kind (`crates/sky-bench/src/compare/report.rs`) and says it is "not profiler output". Several statements are wrong for the native and auto engines:

| `RESULTS.md` says | What the code and measurements show |
|---|---|
| Calls (fib): "every call is interpreted" | calls are native `call` instructions; 2.0 ns per call against C's 0.74 ns; fib's 1.36× is mostly the floor |
| Small vectors (nbody): "each vector operation allocates a small array and checks every element" | false in native code, where arrays of ≤ 16 entries are scalar-replaced (`crates/sky-jit/src/module/small.rs`); true for ast and sir. About 23 of nbody's 32 ms is fixed cost; per step it is 3.2× C |
| Scalar (mandelbrot): "each operation is dispatched by an interpreter" | the native loop is machine code, 1.20× Rust per iteration; the remainder is the per-operation checks |
| Array chains: "unfused operation allocates…" and "includes building the inputs" | the second clause is the story: input construction and the front end are about 70 of 93 ms at 1e7; the fused kernel runs at C speed |
| Cholesky: "unblocked factorisation with finiteness checks" | n = 32 does run the plain loops (`crates/sky-linalg/src/chol.rs:147-150`); about a third of each iteration is boxed dispatch for a `Vector[32]` |
| Matmul: "Sky beat C" | true, but C is a naive triple loop at 13.8 GFLOP/s |

[VERIFIED: R10 §2.2]

---

## 2. Per-unit results

### 2.1 Scalar code, calls, small vectors

Sky is the native engine with default threads. Medians in ms at the two sizes; per-unit columns from the slope [VERIFIED: R10 §3.1, logs `r10/m1_scalar.log`, `r10/m5.log`].

| Workload | N1 → N2 | C per unit | Rust per unit | Sky per unit | Sky / C | Sky / Rust |
|---|---|---:|---:|---:|---:|---:|
| fib, recursive calls | fib 30 → 32 (2.69M → 7.05M calls) | 0.74 ns | 0.53–0.72 ns | **2.00 ns** | **2.7×** | 2.8–3.8× |
| nbody, 5 bodies | 10^5 → 4·10^5 steps (minimums) | 27.6 ns | 38.9 ns | **87.1 ns** | **3.2×** | 2.2× |
| nbody, `SKY_SLP=off` | same | | | 115.7 ns | 4.2× | 3.0× |
| mandelbrot 300² | maxit 500 → 2000 | 67.96 ms per Δ | 71.84 ms | **86.49 ms** | 1.27× | **1.20×** |
| rk4 | 2·10^5 → 8·10^5 steps | 33.4 ns | 38.3 ns | **63.1 ns** | **1.89×** | 1.65× |
| index loop `s = s + x[i] + 1` over a 10^6 vector | 2.5·10^5 → 10^6 iterations | 2.6 ns | – | **148 ns** | **57×** | – |
| the same on the SIR interpreter | | | | 387 ns | 149× | |

Fixed parts of Sky's time from the same fits: nbody about 23 ms, fib about 6.7 ms (floor 5.7 plus about 1 ms of compile), rk4 about 8 ms, mandelbrot about 11 ms. `SKY_SLP=always` equals `auto` on nbody. [VERIFIED: R10 §3.1]

### 2.2 Arrays

| Workload | Unit | C, 1 thread | Sky native, 10 threads | Sky native, 1 thread | Sky sir, 10 threads | Sky sir, 1 thread | Accelerate |
|---|---|---:|---:|---:|---:|---:|---:|
| elementwise 10^7 (`sqrt(X*X+Y*Y)*0.5+D`) | ms per pass | 3.7 | **2.3** | 5.0 (1.36×) | 3.8 | 4.3 (1.17×) | – |
| elementwise 10^6 | ms per pass | 0.30 | **0.20** | 0.39 (1.31×) | 0.33 | 0.47 (1.56×) | – |
| linear_chain 10^6 | ms per pass | 0.21 | **0.134** | – | 0.28 | – | – |
| matmul 512 (product plus rescale) | ms per product (GFLOP/s) | 19.4 (13.8, naive loop) | 2.97 (90) | 11.0 (24.3) | – | – | 0.61 (438) |
| matmul 512, `--numerics=relaxed` (Accelerate inside Sky) | ms per product (GFLOP/s) | | 0.80 (338) | | | | |
| cholesky-32 (factor plus two solves) | µs per solve | 2.72 | 6.49 (2.4×) | – | – | – | 1.47 |
| cholesky-32, relaxed | µs per solve | | 5.49 | | | | |

[VERIFIED: R10 §3.2, `r10/m2_arrays.log`]

Memory, elementwise 10^7 on one thread (`/usr/bin/time -l`): peak RSS 526 MB native, 365 MB sir, 161 MB C. Page reclaims are identical at 2 and 6 passes, so the per-pass result allocation does not fault in fresh pages; the native/sir difference is the extra write stream into a new buffer. [VERIFIED: R10 §3.2, `r10/m4.log`]

### 2.3 Fused kernels

Nanoseconds per element, single thread, preallocated output, `cargo bench -p sky-jit --bench kernels` [VERIFIED: R10 §4.5, `r10/kernels_bench.log`; R1 §6.2 for the checked-C column]:

| Kernel | n | Sky (checked) | Sky, checks off | Sky scalar (no SIMD) | best C/Rust | C with the same checks | Sky / best |
|---|---:|---:|---:|---:|---:|---:|---:|
| axpy | 10^5 | 0.234 | 0.181 | 0.295 | 0.233 | 0.232 | **1.00×** |
| axpy | 10^7 | 0.263 | 0.261 | 0.312 | 0.262 | 0.266 | **1.01×** |
| hypot | 10^7 | 0.301 | 0.303 | 0.602 | 0.302 | 0.414 | **1.00×** |
| chain6 (six operations, one division) | 10^7 | 0.357 | 0.271 | 0.675 | 0.309 | 1.037 | **1.16×** |

Kernel compile time is 0.16–0.26 ms. The deferred finiteness check costs about 0% on memory-bound kernels and about 30% on the compute-bound chain with a division; Sky's checked kernels take 0.66–0.84× the time of C that makes the same checks. [VERIFIED]

### 2.4 Dense linear algebra

GFLOP/s, `sky-linalg` bench with `BENCH_SIZES=32,64,256,512,1024 BENCH_THREADS=1,10` [VERIFIED: R10 §4.7, `r10/blas_bench.log`].

One thread (Accelerate with `VECLIB_MAXIMUM_THREADS=1`):

| Operation | n | Sky reproducible | Sky FMA kernel | Sky relaxed | faer | matrixmultiply | Accelerate |
|---|---:|---:|---:|---:|---:|---:|---:|
| gemm | 64 | 21.8 | 38.6 | 217 | 43.3 | 43.9 | 296 |
| gemm | 256 | 24.95 | 48.4 | 339 | 48.3 | 48.3 | 378 |
| gemm | 512 | 24.98 | 47.1 | 639* | 48.8 | 47.7 | 366 |
| gemm | 1024 | 24.60 | 47.4 | 543* | 49.4 | 47.3 | 352 |
| potrf | 32 | 3.09 | 3.08 | 3.39 | 5.54 | – | 12.8 |
| potrf | 256 | 14.8 | 18.6 | 52.5 | 25.2 | – | 110.6 |
| potrf | 1024 | 20.7 | 32.5 | 100 | 37.1 | – | 154.5 |
| gesv | 1024 | 19.2 | 26.4 | 83.9 | 34.7 | – | 66.6 |

\* The in-process relaxed call does not cap Accelerate's threads, so these two cells are multi-threaded AMX.

Ten threads:

| Operation | n | Sky reproducible | Sky FMA kernel | faer | Accelerate |
|---|---:|---:|---:|---:|---:|
| gemm | 256 | 97 | 150 | 119 | 378 |
| gemm | 512 | 133 | 242 | 234 | 689 |
| gemm | 1024 | 137 | 239 | 267 | 628 |
| potrf | 256 | **12.3** (slower than 1 thread) | 13.3 | 22.9 | 113.5 |
| potrf | 1024 | 42.7 | 51.9 | 84.3 | 154.7 |

### 2.5 Against CPython

R10 did not slope-measure CPython. From the whole-process medians of §1.2, the best Sky engine is faster than pure CPython on every case: for example 17.8× on nbody (610 vs 34.3 ms), 10.8× on fib-30 (124 vs 11.5), 23× on mandelbrot (912 vs 38.9), 87× on elementwise-1e6 (2110 vs 24.1) and 870× on matmul-512 (12.3 s vs 14.1 ms); the geometric mean is 0.02× [VERIFIED: `RESULTS.md`]. Because these totals include Sky's fixed costs, the per-unit advantage is larger [LIKELY]. NumPy was not measured, so nothing here compares Sky with vectorized Python.

### 2.6 Where Sky wins

- **Fused elementwise kernels run at C speed**: axpy 1.00–1.01×, hypot 1.00× of the best C or Rust loop (§2.3).
- **Checks cost less than C's**: Sky's checked kernels take 0.66–0.84× the time of hand-written C with the same checks, because the check is deferred to one test per loop (§2.3).
- **Threads for free**: elementwise programs run 0.6–0.67× the time of single-threaded C per pass on 10 threads, with bit-identical results whatever the thread count (§2.2). This is a parallelism win, not a code-generation win.
- **Reproducible GEMM is near its ceiling**: 24.6–25.0 GFLOP/s per core is 89% of the no-FMA NEON peak, and 1.76× faster than the naive C triple loop on one thread (§2.4).
- **Scalar loops are close**: mandelbrot is 1.20× Rust per iteration, with output bit-identical to Rust's (§2.1).
- **Against an interpreter**: one to three orders of magnitude faster than pure CPython (§2.5).
- **f64 GEMV at about 105 GB/s on 10 threads** (2048×2048) [EXPERIMENTAL-quality: R1 §5.5].

### 2.7 Where Sky loses

| Gap, per unit of work | Size | Root cause (§3) |
|---|---:|---|
| element access on arrays of > 16 entries | 57× C | boxed runtime call per element (§3.1) |
| elementwise op on a 32-entry vector | about 75–100× the inline cost | same (§3.1) |
| nbody step | 3.2× C | checks, lane shuffles, spills (§3.2) |
| fib call | 2.7× C | call protocol, no inliner (§3.3) |
| rk4 step | 1.89× C | four calls per step, no inliner (§3.3) |
| elementwise pass, 1 thread | 1.36× C (native), 1.17× (sir) | no buffer reuse in native code (§3.4) |
| GEMM, 1 thread | 14–15× Accelerate | FMA ban × AMX (§3.5) |
| GEMM, 10 threads | 4.6–5.2× Accelerate | same |
| cholesky-32 | 2.4× C, 4.4× LAPACK | plain loops at small n; boxed `Vector[32]`; loop-invariant factor redone (§3.6) |
| fixed cost per run | 4.7–5.7 ms floor; about 12 ms of compile for nbody; 16 ms of front end for 290 KB of literals | no code cache; literal-heavy sources (§3.9) |

[VERIFIED: §§2.1–2.4]

---

## 3. Why Sky loses: root causes in the code

### 3.1 Boxed runtime calls for arrays of more than 16 entries

In native code, only arrays with a static shape and at most 16 entries are scalar-replaced (`MAX_SMALL = 16`, `crates/sky-jit/src/module/small.rs:53`). Every operation on any other array goes through one generic path:

```text
generated code (crates/sky-jit/src/module.rs:2229-2245: Index, Slice, MatMul, Dot, Fused, ... → generic)
  └─ generic (module.rs:2271-2284)
       ├─ sky_rt_push_ref / sky_rt_push_num / sky_rt_push_elem   one runtime call per operand
       └─ sky_rt_op(site)          crates/sky-jit/src/runtime.rs:583-606
            └─ eval_op             runtime.rs ~1100-1205: match on OpKind, run the interpreter's
                                   function on Rc<Value> (bounds checks in sky_interp::array::index)
            └─ allocate an Rc and a handle for the result
  └─ generated code unboxes the result, or keeps the handle and releases it at block end
```

For `x[i]` that is about five runtime calls and one or two allocations per element: **148 ns per access against 2.6 ns in C** (57×); the SIR interpreter takes 387 ns [VERIFIED: R10 §4.4]. For a 32-entry elementwise add it is about 291 ns per operation, against 3–4 ns for an 8-entry add that stays inline [VERIFIED: R1 App. B]. Small arrays read at a run-time index use an O(N) compare-and-select chain (`small.rs:221-230`).

The suite has no case for this pattern, so `RESULTS.md` never shows it. It is the item that matters most for LLM-related CPU code (§7). The in-flight `store` operation (arrays worktree, [Part II](part02-sky-today.md) §13.3) writes in place when the operand dies, but it is dispatched through the same `generic` path, so element writes will cost about what reads cost [LIKELY: same code path; not measured].

### 3.2 Checks, SIMD lane shuffles and spills in native loops

A static census of nbody's native main loop (1,765 instructions; `sky build --emit=obj`, `otool -tV`) [VERIFIED: R10 §4.1]:

| Category | Instructions | Share |
|---|---:|---:|
| finiteness checks (`fsub x, x`) | 204 | 12% |
| `fcmp` (finite, zero divisor, `sqrt` domain) | 215 | 12% |
| conditional branches | 215 | 12% |
| moves, lane inserts and extracts, `dup` (SLP packing) | 342 | 19% |
| stack spills and reloads | 252 | 14% |
| floating-point arithmetic (including the adds that combine check lanes) | 411 | 23% |
| unconditional branches, constant loads, other | about 126 | 7% |

Where they come from:

- **Checks.** Native scalar code checks every operation inline: `is_finite` is `fsub x, x; fcmp ord` (`crates/sky-jit/src/module.rs:1366-1377`) followed by a branch to a cold block (`finite_or`, `module.rs:1380-1390`), applied in `arith` (`module.rs:2357-2359`) unless the range analysis proves the operation safe (`module.rs:1357-1360`; `crates/sky-ir/src/exact/mod.rs`). Every division has a zero-divisor check (`module.rs:2348-2352`); every `sqrt` has a domain check before it and an `fadd 0.0` after it (normalizing −0 to +0, Sky semantics). Fused kernels defer the check to one test per loop (`crates/sky-jit/src/kernel.rs:31-58`); scalar code does not.
- **Lane shuffles.** SLP packs two of three vector components into `F64X2` (`mov.d v17[1], v31[0]`) and then extracts lanes again to check them (`mov d28, v24[1]`). SLP still pays: turning it off is 33% slower (115.7 against 87.1 ns per step).
- **Spills.** The loop carries 30 f64 values (ten `Vector[3]`) plus packed copies against 32 vector registers (regalloc2 0.12.2). A zero vector is reloaded from the stack (`ldr q31, [sp,#0x4d0]`) instead of rematerialized [LIKELY: `vconst`/splat is not covered by Cranelift's `opts/remat.isle`].

Units cost nothing in native code: dimensions are static and checked at compile time (`module.rs:2329-2339`). Allocation plays no part in the native nbody loop. [VERIFIED: R10 §4.1, §4.2]

### 3.3 No inliner, and the call protocol

- **Protocol.** Every function takes `(depth: i32, stack_limit: i64, site: i32, args…)` and returns `(status: i32, results…)`. The callee compares the stack pointer with the limit (`stack_check`, `module.rs:1393`); the caller checks `depth ≥ 1000` before the call and `status ≠ 0` after it (`call_op`, `module.rs:2932-2954`); the result is finiteness-checked. Measured: **2.0 ns per call against 0.74 ns in C**. The C version uses `long` while Sky numbers are f64, and clang partially turns the recursion into a loop [LIKELY: a standard LLVM transform]. [VERIFIED: R10 §4.3]
- **No inliner anywhere.** The SIR pipeline has none (`crates/sky-opt/src/pipeline.rs:10-60`), and neither has Cranelift 0.121.2: `Context::optimize` is legalization, unreachable-code removal, constant-phi removal and one e-graph pass (`cranelift-codegen-0.121.2/src/context.rs:150-189`). rk4's right-hand side `f(y)` is four native calls per step, which is the likely reason rk4 is 1.89× C while call-free mandelbrot is 1.27× [LIKELY]. [VERIFIED: R10 §4.3, §4.6]
- **The SIR interpreter is slower than the tree-walker on calls**: fib-30 takes 1622 ms against 718 ms, about 600 ns against 267 ns per call, because "a call still allocates its argument list and its frame" (`crates/sky-ir-interp/src/machine.rs:30`) and the value table is resized per activation (`machine.rs:1211-1248`). [VERIFIED: R10 §4.3; a negative result]

Cranelift has no auto-vectorizer, no loop unroller and no inliner. Sky does its own vectorization (explicit `F64X2` in kernels, SLP in scalar code), and `F64X2` is already the full NEON width for f64. The losses are register allocation and check branches, not instruction selection. [VERIFIED: R10 §4.6]

### 3.4 No buffer reuse in native code

The SIR interpreter writes an elementwise result into the buffer of an operand that dies there, using `sky_ir::Lifetimes` (`crates/sky-ir/src/lifetime.rs`). `sky-jit` does not use `Lifetimes` at all, so every native array operation writes a fresh buffer: an extra write stream per pass and a larger footprint. Measured on elementwise 10^7, one thread: 5.0 ms per pass native against 4.3 ms sir; peak RSS 526 MB against 365 MB. [VERIFIED: R10 §4.5, §3.2]

### 3.5 Matmul: the FMA ban times AMX

| Factor | Size | Why |
|---|---:|---|
| FMA ban | 2× | the reproducible rule adds `round(a·b)` to the sum in naive `k` order, so multiply and add are separate operations (`spec/v0.1.md:639-663`). The reproducible kernel runs at 24.6–25.0 GFLOP/s, 89% of the no-FMA peak of 27.9 GFLOP/s per P-core (4 FP pipes × 2 lanes × 3.49 GHz); Sky's own FMA kernel runs at 47–48 GFLOP/s, equal to faer and matrixmultiply and 86% of the FMA peak of 55.8 |
| AMX | about 7.5× on one thread | Accelerate drives Apple's matrix coprocessor: 352–378 GFLOP/s on one thread; 628–689 on all threads, about 1.8× its single-thread rate, likely because each P-cluster of the M2 Pro has an AMX unit |
| Everything else | none | blocking, packing, the 4×8 register tile and threading are fine for NEON; Sky's reproducible kernel scales 5.6× on 6P + 4E |

[VERIFIED: R10 §4.7; the per-core peak model and the per-cluster AMX are LIKELY, consistent with the measured ratios]

No NEON kernel can reach Accelerate here; only AMX (on M4, SME) can, and AMX is reachable only through Accelerate.

### 3.6 Cholesky at small n

- **Plain loops.** For n ≤ 40 Sky runs the plain reference loops (`crates/sky-linalg/src/chol.rs:27`, `:147-150`; `reference.rs:122-146`): 3.09 GFLOP/s at n = 32 against Accelerate's 12.8.
- **Boxed dispatch.** A `Vector[32]` exceeds `MAX_SMALL`, so its scaling, the solve, two `index` operations and `ones @ x` are runtime calls.
- **A loop-invariant factorization.** The benchmark's loop body is `%x_3 = solve[cholesky] %K, %50` with a constant `K`: the factorization is redone 5,000 times. C and LAPACK also refactor every iteration, so the benchmark is algorithm-parity.
- **Threading hurts** potrf at n = 256 (12.3 GFLOP/s on 10 threads against 14.8 on one): the thresholds are wrong for it.

Estimated split of the 6.49 µs per iteration: about 3.5 µs plain-loop factorization, about 1 µs for the two triangular solves, about 2 µs of boxed dispatch [LIKELY: split by subtraction, not profiled]. [VERIFIED otherwise: R10 §4.7]

### 3.7 libm and reproducibility

- `sky_interp::math` calls Rust's `f64::exp`, `ln`, `sin` and so on (`crates/sky-interp/src/math.rs:189-215`), which is the platform libm: Apple's on macOS, glibc or musl elsewhere. None of these guarantees correct rounding, so some results can differ in the last bit between operating systems [VERIFIED in code; LIKELY that results differ].
- The spec claims "the same program prints the same bits everywhere" (`spec/v0.1.md:640`) and also admits the platform's math library may be within four ulps (`spec/v0.1.md:805`). The two statements are in tension.
- `sky-ode` already uses the pure-Rust `libm` crate for its step controller (`crates/sky-ode/src/dopri.rs:248`), which is the cross-platform fix.
- Inside fused kernels, every math function other than `sqrt` and `abs` is a per-lane scalar call that builds two `Value`s and calls `sky_interp::math::apply` (`crates/sky-jit/src/kernel_rt.rs:30-41`) [VERIFIED]; about 5–10 ns per element for exp- or tanh-heavy bodies such as softmax, GELU and SiLU [LIKELY: an extern call, a match and a libm call; not measured].

[R10 §4.9c, §4.5]

Two further reproducibility facts were measured on this machine and matter for any fix that touches Accelerate [VERIFIED: R10 §4.9a–b, `r10/accdet/`]:

| Fact | Result |
|---|---|
| `cblas_dgemm`, `cblas_dgemv` across runs and `VECLIB_MAXIMUM_THREADS` ∈ {default, 1, 2, 4, 8} | identical bits |
| `dpotrf` across thread counts | identical across runs at a fixed count; **1 thread differs from ≥ 2 threads** at n = 512 and 1024; all agree at n = 64 |
| `cblas_dgemm` against the sequential FMA chain `c = fma(a[i,k], b[k,j], c)` from +0.0, k ascending | bit-identical on every entry at n = 16, 64, 256, 512, 1000, at 1023×1021×1019, at 64×64×8192, for 1×2048×2048 and with a transposed B |
| the same, for tiny products and matrix @ vector | **28 of 35** entries at 7×5×3; **83 of 2,048** at 2048×1×2048 |
| `cblas_dgemm` against Sky's current unfused chain | only 15–30% of entries equal |

Not tested: other Apple chips, other macOS versions, `beta = 1`. Apple documents no such guarantee [EXPERIMENTAL].

### 3.8 Where fusion stops

[VERIFIED: R10 §4.5, IR dumps of `micro/fuse_dag.sky`]

1. **Multi-use intermediates.** In `T = X*Y; Z = sqrt(T) + T*0.5`, `T` stays a materialized array.
2. **Reductions.** `dot`, `@` and `norm` consume a materialized array; there is no producer-to-reduction fusion and no `sum` builtin (sums are written `ones @ X`).
3. **Matmul epilogues** (`A@B + C`, scaling) are not fused.
4. **Broadcasts other than scalars** are not candidates. The benchmark itself needs a full array `D` of 0.25s, because Sky has no `array + scalar`.
5. **Math functions** are per-lane scalar calls (§3.7).
6. **Native code does not reuse buffers** (§3.4).

### 3.9 Fixed costs: start-up, front end, Cranelift

| What | ms |
|---|---:|
| C empty process | 1.45 |
| `sky check` on `print(1)` | 4.75 |
| `sky run` on `print(1)`: ast / sir / native / auto | 4.69 / 5.47 / 5.67 / 5.73 |
| `sky check nbody.sky` | 5.34 |
| `sky check --emit=clif nbody.sky` (CLIF generated, not compiled) | 12.6 |
| `sky run --engine=sir`, nbody with 1 step | 11.05 |
| `sky run --engine=native`, nbody with 1 step | **23.0** |
| `sky build --emit=obj nbody.sky` | 23.7 |
| `sky check elementwise-1e7.sky` (290 KB of literals) | 16.1 |

[VERIFIED: R10 §4.8, `r10/m1b.log`, `r10/m3.log`]

Of nbody's native 32 ms: about 5.3 ms process and front end, about 5 ms lowering and the SIR pipeline, **about 12 ms CLIF generation and Cranelift**, and 8.7 ms for the 10^5 steps. Nothing is cached across processes ([Part II](part02-sky-today.md) §6.5). One more fixed cost belongs to every macOS benchmark: the first execution of a freshly linked binary costs about 350–400 ms (code-signing check), so warmups are mandatory, also for `sky build` outputs [VERIFIED: R10 §1, `r10/compare_nosky.log`].

---

## 4. Negative results

All VERIFIED in R10 §8 unless marked.

- Native per-pass allocation does not cause extra page faults: page reclaims are identical at 2 and 6 passes; macOS malloc recycles the freed 80 MB block. The real cost is the write stream and the RSS.
- `SKY_SLP=always` gives nothing over `auto` on nbody.
- The SIR interpreter is slower than the tree-walker on call-heavy code (fib: 1622 against 718 ms).
- Threads make potrf slower at n = 256 (12.3 against 14.8 GFLOP/s).
- The fused kernel is not where elementwise programs lose.
- "Small vectors allocate" is false in native code.
- Accelerate `dpotrf` is not thread-count invariant.
- Several of the harness's generated explanations are wrong (§1.5).
- Sky's own NEON GEMM cannot be tuned to Accelerate's speed: it is at 89% of the no-FMA ceiling and its FMA variant at 86% of the FMA ceiling; the remaining factor is AMX.

---

## 5. Roadmap A: the fixes F1–F11

Difficulty and confidence are on a 1–5 scale (5 = hardest, most confident). Gains are per unit of work against the slopes of §2. [R10 §5]

| Fix | Design in one line | Expected gain | Diff. | Conf. | Files | Matters for LLM work? |
|---|---|---|---:|---:|---|---|
| **F1** inline element access for big arrays | a `Rep::Buf` representation (handle plus `(ptr, len)` from one call `sky_rt_buf`, hoisted out of loops); `index` becomes an inline integral-and-in-range test plus a load; no checks at all with a `Proof::InBounds` from `for i in range(len(x))` | 148 ns → about 1–3 ns per element, 50–100× on indexing loops [LIKELY] | 3 | 4 | `crates/sky-jit/src/module.rs` (`Rep`, `rep_of`, dispatch at `:2229-2245`, new `index_buf`); `runtime.rs` (`sky_rt_buf`); `aot.rs` (export); `crates/sky-ir/src/exact/mod.rs` (`InBounds` for big arrays) | **yes**, with element writes (below) |
| F1 follow-on: element writes | `x[i] = v` as an in-place store when the operand dies and its refcount is 1, copy-on-write otherwise; a language change | writes at load-store cost [LIKELY] | 3 | 3 | spec, checker, and the F1 files; builds on the arrays worktree | **yes** |
| **F2** deferred checks in native scalar code | per check region (straight-line, no effects), accumulate `v − v` of the values that reach a sink, one `fcmp uno` and a branch; the cold block replays the region with today's checks for the exact diagnostic | nbody 1.4–1.8× per step; mandelbrot 10–20%; rk4 10–20% [LIKELY] | 3 | 3 | `module.rs` (`arith`, `finite_or`, `math`, `pow_const`, `inverse`); `module/slp.rs`; new `module/defer.rs`; differential fuzzing with overflow inputs | no (scientific) |
| **F3** SIR inliner and a leaner call protocol | (a) inline small non-recursive callees without `io` before `specialize`; (b) skip stack checks in leaf functions, return failure as a NaN sentinel for single-number results, keep depth and limit in a pinned register | rk4 1.89× → about 1.2–1.3× C; fib 2.7× → about 1.8–2.0× [LIKELY] | 3 / 2 | 4 / 3 | new `crates/sky-opt/src/inline.rs`; `pipeline.rs`; `crates/sky-jit/src/module.rs` (`call_op` `:2932`, `define_thunk` `:872`) | moderate |
| **F4** hoist factorizations | lower `K⁻¹ @ b` to `factor[cholesky]` plus `solve_factored`; LICM hoists the factor under the existing guard rule; bits unchanged | cholesky-32 6.5 → about 2.5–3 µs (≤ C's 2.7); about 1.2 µs with F5 [LIKELY] | 2 | 4 | `crates/sky-ir/src/op.rs`; `crates/sky-opt/src/{cholesky,licm}.rs`; `crates/sky-interp/src/solve.rs`; `crates/sky-ir-interp/src/machine.rs`; `crates/sky-jit/src/runtime.rs` | no |
| **F5** mid-size arrays without boxing | static-shape values of ≤ about 512 entries that do not escape live in stack slots, with loops (not unrolled code) and a pointer ABI into `sky-linalg` | 5–20× on mid-size vector loops; cholesky-32 dispatch about 2 → 0.2 µs [LIKELY] | 4 | 3 | `module.rs`, `module/small.rs`, `runtime.rs`, `aot.rs`, `crates/sky-rt` | no |
| **F6** in-place reuse in native code | pass `Lifetimes::reuse` to the runtime (`sky_rt_op_consume`); write into the operand's buffer when `Rc::strong_count == 1` | about 15% per pass on 10^7 chains (5.0 → 4.3 ms); peak RSS about −30% (526 → 365 MB) [VERIFIED basis: the SIR numbers] | 2 | 4 | `module.rs` (`generic`, `push_operands`); `runtime.rs` (`sky_rt_op`, `fused`); `kernel.rs` (`Kernel::run`) | moderate (memory on 16 GB machines) |
| **F7** "reproducible FMA" edition plus a self-verified Accelerate path | define products and factorizations as k-ordered FMA chains (correctly rounded, so still the same bits everywhere); on macOS, run large GEMMs on Accelerate only in shape classes that pass a bit-for-bit self-test against Sky's FMA kernel | 2× everywhere (25 → 47 GFLOP/s per core) [VERIFIED by the FMA-kernel column]; 14× one thread, about 5× all threads on large GEMM if the self-test passes [EXPERIMENTAL] | 3 | 4 (2×); 2–3 (Accelerate) | `spec/v0.1.md:639-663` and `:665+`; `crates/sky-linalg/src/{micro,gemm,chol,lu,tri,inverse,numerics,relaxed,accelerate}.rs`; `crates/sky-jit/src/module/small.rs`; goldens | **no** (scientific only) |
| **F8** cut native compile time | a content-addressed on-disk cache of compiled modules keyed by SIR hash, Sky version and target; lower unroll and SLP budgets for straight-line code; cold code at `opt_level=none`; faster literal lexing | −10 to −20 ms per run on a cache hit [LIKELY] | 2 | 4 (cache); 3 | not listed by R10; likely `crates/sky-jit/src/{lib.rs,module.rs,aot.rs}`, `crates/sky-cli`, `crates/sky-syntax/src/lexer.rs` [LIKELY] | minor (CLI start-up) |
| **F9** threading fixes | raise potrf's parallel threshold; E-core-aware chunking (smaller chunks with stealing, or P-core-only QoS) | 10–30% at mid sizes [LIKELY] | 2 | 3 | `crates/sky-linalg/src/{tuning,threads,chol}.rs`; `crates/sky-ir/src/parallel.rs` | no |
| **F10** one libm, vector math in kernels | route `sky_interp::math::apply_f64` to one pure-Rust implementation (`libm` crate, or CORE-MATH for correct rounding), then lane-wise `F64X2` versions with the identical operation sequence | cross-OS reproducibility; 5–10× on transcendental-heavy kernels [LIKELY, not measured] | 3 | 3 | `crates/sky-interp/src/math.rs`; `crates/sky-jit/src/kernel.rs`; `kernel_rt.rs` | minor (CPU softmax only if sampling stayed on the CPU) |
| **F11** LLVM backend (inkwell) | not recommended now: the losses are Sky-level; LLVM cannot vectorize loops with Sky's early-exit checks either; it compiles far more slowly; it adds a large pinned dependency | perhaps 1.2–1.5× on scalar loops [EXPERIMENTAL] | 4–5 | 2 (net benefit) | — | no |

On F11, R7 reaches the same conclusion from published data: keep Cranelift for host and scalar code, and revisit only for SME matmuls or CPU kernels that must vectorize well beyond Cranelift ([R7](research/R7-compilers.md) Topic 11). D16 records it ([Part VI](part06-architecture.md)).

### 5.1 R10's order, if Roadmap A ran as its own programme

Gates are on per-unit slopes at two sizes, never on whole-process medians [R10 §7]:

| Weeks | Item | Gate | Realistic? |
|---|---|---|---|
| 1 | benchmark hygiene: slope mode in `sky-bench compare`, corrected explanations, new cases (big-array indexing, a top-k/top-p sampler over 128K entries, a call microbenchmark, 32–256-entry vectors), a fixed-cost split | `RESULTS.md` shows per-unit ratios | yes |
| 2–4 | F1 | indexing loop ≤ 1.5× C (today 57×) | yes [LIKELY] |
| 3–4 | F6 | elementwise 10^7, 1 thread, ≤ 1.15× C per pass (today 1.36×); RSS ≤ SIR's | yes |
| 4–5 | F4 | cholesky-32 ≤ 1.0× C per iteration (today 2.4×) | yes |
| 5–7 | F3 | rk4 ≤ 1.3× C per step (today 1.89×); fib ≤ 2.0× C per call (today 2.7×) | rk4 yes; fib LIKELY |
| 6–9 | F2 | mandelbrot ≤ 1.05× Rust per iteration (today 1.20×); nbody ≤ 2.0× C per step (today 3.2×) | mandelbrot LIKELY; nbody EXPERIMENTAL (spills remain) |
| 8–9 | F8 | nbody fixed cost ≤ 10 ms on a cache hit (today about 23) | yes |
| 9–12 | F7 | own kernel ≥ 45 GFLOP/s per core in the default mode; matmul-512 ≤ 1.3× Accelerate if the self-test passes on M1–M4, otherwise ≤ 3× (all threads) | 2×: yes; Accelerate gate EXPERIMENTAL |
| 10–12 | F5, if time | 32-entry vector operations in loops ≤ 3× C | EXPERIMENTAL |

### 5.2 What the inference programme schedules

D16 makes the CPU side secondary: only cheap fixes, and fixes that also help tokenizer and sampling code. Applied to the table:

| Group | Items | When |
|---|---|---|
| Do now (days, no risk) | III-T1 benchmark hygiene, including the sampler case; II-T6 (`sky hw` thread count) | before G1 |
| Language prerequisites for host code written in Sky | F1 with element writes, an integer type, F3 | with the Sky-source front door after G4, or by a scientific-Sky owner earlier; none moves a year-1 inference metric (§7.4) |
| Cheap, useful to scientific users | F6, F8 (F8 needs a decision under non-negotiable 3, §10) | whenever capacity allows |
| Scientific track only | F2, F4, F5, F7, F9, F10 | scheduled by scientific value, outside the inference plan |
| Not now | F11 | revisit after F1–F3, as an optional `sky build --release` backend |

---

## 6. The owner's example gates: an honest verdict

### 6.1 "Fused elementwise ≤ 1.1× C"

| Reading of the gate | Status today | After F6 |
|---|---|---|
| per element, fused kernel, one thread, preallocated output | **met** for axpy (1.00–1.01×) and hypot (1.00×); **not met** for chain6 (1.14–1.17×), whose divisor needs its own check (checks off: 0.88×) | unchanged |
| per pass, inside a Sky program, one thread | 1.36× native, 1.17× sir (elementwise 10^7) | about 1.1–1.15×: the benchmark's extra array `D` costs about one stream in four |
| per pass, with threads | 0.6× of single-threaded C | — |
| whole-process median (`RESULTS.md`) | 1.5–2.4× | still dominated by the 290 KB literal front end and input construction |

[VERIFIED: §2.2, §2.3; the F6 column LIKELY, R10 §7]

**Verdict: realistic if the gate is defined per element or per pass, single thread, on named kernels; not realistic on whole-process medians** while the benchmark builds its inputs from outer products and literals. Meeting it for chain6 needs a cheaper check on divisors, for instance a proof that the divisor is non-zero and finite [EXPERIMENTAL: whether the range analysis can prove it on the benchmark's inputs].

### 6.2 "Matmul ≤ 1.2× Accelerate"

| Mode | Status today | Best case after fixes |
|---|---|---|
| reproducible (default), one thread | 14–15× Accelerate (24.6–25.0 against 352–378 GFLOP/s) | F7's FMA chain: about 7.5×; still AMX-bound |
| reproducible, all threads | 4.6–5.2× (133–137 against 628–689 GFLOP/s) | F7's FMA chain: about 2.6–2.9× |
| reproducible with F7's self-verified Accelerate route | — | ≈ 1.0× on shape classes that pass the self-test [EXPERIMENTAL] |
| relaxed, kernel benchmark, one thread | 1.12× at n = 256 (339 against 378 GFLOP/s); 1.36× at n = 64 (217 against 296) | — |
| relaxed, inside a Sky program (matmul-512, product plus rescale) | 1.31× (0.80 against 0.61 ms per product) | — |

[VERIFIED: §2.2, §2.4; R10 §7]

**Verdict: not realistic in the default reproducible mode** under today's semantics: the own kernel is at 89% of its ceiling and the rest is AMX. It is met today only in relaxed mode at mid sizes on the kernel benchmark. In the default mode the only route is F7: redefine reproducible products as FMA chains (an edition change) and call Accelerate per shape class behind a bit-for-bit self-test. That rests on one machine and one OS so far, and must be re-verified on M1, M3 and M4 (M4 uses SME, whose `FMOPA` is architecturally a per-element fused multiply-add: promising, untested).

### 6.3 Recommended form of the gates

Replace both with gates that a benchmark can decide:

1. "Fused elementwise kernels (axpy, hypot, chain6) at ≤ 1.1× the best of C and Rust per element, one thread, output preallocated, measured by `cargo bench -p sky-jit --bench kernels`; and ≤ 1.15× C per pass inside a Sky program, by slope."
2. "Default mode: GEMM ≥ 45 GFLOP/s per P-core and ≤ 3× Accelerate on all threads at n ≥ 256. Relaxed mode: ≤ 1.2× Accelerate at n ≥ 256. If F7's self-test passes on M1–M4: ≤ 1.3× Accelerate in the default mode."

---

## 7. Relevance to the LLM direction

### 7.1 What CPU work an LLM runtime does

Decode is bound by GPU memory bandwidth and prefill by GPU matrix throughput ([Part IV](part04-hardware-limits.md)). On the CPU, a runtime does tokenization (byte-pair merges over hash maps and integer arrays), sampling (temperature, top-k/top-p, penalties over a 32K–152K vocabulary), scheduling and KV-cache bookkeeping (integers and indices), constrained-decoding masks, and small glue arithmetic. [R10 §6]

### 7.2 Which fixes matter

| Item | Matters? | Why |
|---|---|---|
| F1 inline element access, plus element writes | **yes, for any CPU code written in Sky** | sampling, tokenizer tables and logits processors are exactly "read and write one element of a large array in a loop" (§7.3) |
| an integer type | **yes, a prerequisite** | token ids, indices, bitmasks, hashing, packed bytes; not a performance fix but nothing above works without it ([Part II](part02-sky-today.md) §3.2) |
| F3 inlining, cheap calls | moderate | orchestration code is call-heavy; at 2 ns per call the per-token budget is not at risk |
| F6 in-place reuse | moderate | on 16 GB machines every redundant buffer competes with weights and KV cache |
| F8 compile and start-up | minor | model load takes seconds; matters for command-line feel |
| F10 vector exp | minor | CPU softmax only if sampling stayed on the CPU, which §7.3 rules out |
| F2 deferred checks | scientific | LLM host code is not floating-point-check-bound |
| F4, F5, F9 | scientific | — |
| F7 reproducible FMA and AMX GEMM | **scientific only** | LLMs run f16/bf16/int4 on the GPU; f64 GEMM is irrelevant |
| F11 LLVM | no | the code generator that matters for tokens per second is Metal |

[LIKELY: R10 §6]

### 7.3 The sampler-cost example: why sampling runs on the GPU

Qwen3's vocabulary has 151,936 entries ([`roofline.py`](roofline.py)). A sampler that reads each logit once, in Sky native code as it is today:

```text
151,936 logits × 148 ns per element = 22.5 ms per token        (SIR interpreter: 58.8 ms)
```

Top-k and top-p need more than one pass, so 22.5 ms is a lower bound. Against the decode step it would sit in [VERIFIED: arithmetic; step times from `roofline.py` at context 512 and spec bandwidth, and from the measured MLX baseline]:

| Model and chip | Decode step | Sampler / step | tok/s with a 22.5 ms CPU sampler |
|---|---:|---:|---:|
| Qwen3-32B 4-bit, M5 Max, at the bandwidth ceiling (33.9 tok/s) | 29.5 ms | 0.76 | 19.2 |
| Qwen3-32B 4-bit, M5 Max, MLX measured (28.4 tok/s; [Part V](part05-competitors.md)) | 35.2 ms | 0.64 | 17.3 |
| Qwen3-8B 4-bit, M2 Pro, at the ceiling (46.2 tok/s) | 21.6 ms | 1.04 | 22.7 |
| Qwen3-0.6B 4-bit, M2 Pro, at the ceiling (507.7 tok/s), the draft-model size | 1.97 ms | 11.4 | 40.9 |

Three conclusions:

1. **Today's Sky cannot host the sampler at all**: it would cost more than the whole decode step of an 8B model and eleven steps of a draft model.
2. **F1 alone does not fix it.** At F1's expected 1–3 ns per element, one pass costs 0.15–0.46 ms: negligible for 32B, but 8–23% of a 0.6B draft step at its ceiling, and speculation runs the draft every token [LIKELY: F1's gain is itself LIKELY].
3. **Any CPU sampler, however fast, forces a host round trip per token.** In MLX on the M2 Pro a blocking evaluation round trip costs 149–158 µs against 29 µs pipelined ([R3](research/R3-mlx-core.md) §7.E); llama.cpp's default CPU sampling copies 151,936 × 4 B ≈ 608 KB of logits per token and waits for the GPU before every sample, leaving the GPU idle while the CPU samples and encodes ([R5](research/R5-llamacpp.md) §6). mlx-lm avoids this by sampling inside its lazy GPU graph and queuing step N+1 before blocking on token N (R5 §6, `mlx_lm/generate.py:457-466`).

So sampling, accept/reject and the stop checks run on the GPU, with a counter-based generator for seeded reproducibility (D9, D10; [Part XIII](part13-llm-runtime.md), [Part XVI](part16-decode.md), [Part XXI](part21-speculative.md)). The CPU fix list does not change that decision; it only decides whether Sky could ever host CPU-side logits processors written by users.

### 7.4 What the year-1 runtime actually runs on the CPU

In the year-1 design the tokenizer, chat template, generation loop, KV manager and prompt cache are Rust in `sky-llm` ([Part XIII](part13-llm-runtime.md)), and sampling is on the GPU. **No CPU fix in this part moves a year-1 inference metric.** F1, element writes, an integer type and cheap calls are justified by the language: by D7's second front door (models and, later, host code in Sky source) and by user-written logits processors. They belong with that work, not on the G1–G4 critical path.

The one CPU lesson that does carry into the inference work is methodological: measure per unit of work by slope, report fixed costs separately, and never let a canned explanation stand in for a profile. [Part XXIII](part23-benchmarks.md) applies it to tokens per second.

---

## 8. Implementation tasks

**III-T1. Benchmark hygiene in `sky-bench compare`.**
- Files: `crates/sky-bench/src/compare/{mod.rs,workloads.rs,report.rs,tools.rs}`; new cases under `benches/compare/{sky,c,rust}`.
- Deliverable: a slope mode (two sizes per case; per-unit cost and fixed cost in `RESULTS.md`); a fixed-cost split (front end, lowering and passes, code generation, run); corrected "Interpretation" texts (§1.5); new cases for big-array indexing, a top-k/top-p sampler over 151,936 entries, a call microbenchmark and 32–256-entry vectors; a C column built with `-ffp-contract=off` beside the default one.
- Acceptance: regenerated `RESULTS.md` reproduces the per-unit ratios of §2.1 within ±10% on the M2 Pro; the sampler case shows Sky's per-token cost next to C's; the canned texts no longer contradict §1.5.

**III-T2. F1: inline element access for big arrays.**
- Files: `crates/sky-jit/src/module.rs` (`Rep::Buf`, `rep_of`, `index_buf`, dispatch at `:2229-2245`), `crates/sky-jit/src/runtime.rs` (`sky_rt_buf`), `crates/sky-jit/src/aot.rs`, `crates/sky-rt`, `crates/sky-ir/src/exact/mod.rs`.
- Deliverable: buffer representation for numeric arrays above `MAX_SMALL`, inline `index` and `len`, `Proof::InBounds` for big arrays.
- Acceptance: the indexing loop of §2.1 ≤ 1.5× C per iteration by slope; the differential fuzzer covers out-of-range, negative and non-integral indices with byte-identical diagnostics on all engines; `sky build` executables pass `crates/sky-cli/tests/build.rs`.

**III-T3. Element writes on the buffer representation.**
- Depends on: II-T4 (arrays worktree) and III-T2.
- Files: `crates/sky-jit/src/module.rs` (`store` on `Rep::Buf`), `runtime.rs` (`sky_rt_push_ref_move`), spec section "Building and changing arrays".
- Deliverable: `v[i] = x` compiled to an inline store when the operand dies and its refcount is 1, copy-on-write otherwise.
- Acceptance: a write loop over 10^6 entries ≤ 2× C per element by slope; no allocation per iteration (a case in `crates/sky-bench/tests/allocations.rs` for the native engine); engines agree on the arrays goldens.

**III-T4. An integer type: design.**
- Files: a new design note `docs/design/integers.md`; later `spec/v0.1.md` (edition-gated), `crates/sky-types`, `crates/sky-ir/src/ir.rs` (`NumberType`/`Repr`), both interpreters, `crates/sky-jit`.
- Deliverable: a proposal for integer scalars and arrays (width, overflow as an error in Sky's style, conversion rules, indices as integers, how literals type), reviewed against the spec's principle "reject ambiguity".
- Acceptance: the proposal answers how `for i in range(n)`, `v[i]` and `n = len(v)` type, and how an integer array of 151,936 token ids is declared; the owner approves it before any code is written.

**III-T5. F3: SIR inliner and a leaner call protocol.**
- Files: new `crates/sky-opt/src/inline.rs`; `crates/sky-opt/src/pipeline.rs`; `crates/sky-jit/src/module.rs` (`call_op` `:2932`, `define_thunk` `:872`).
- Deliverable: an inliner before `specialize` with an explanation rule per inlined call; leaf functions without stack checks; the NaN-sentinel return for single-number results if it passes the thunk test.
- Acceptance: rk4 ≤ 1.3× C per step and fib ≤ 2.0× C per call, by slope; `crates/sky-opt/tests/fuzz_*.rs` and the differential fuzzer unchanged; "recursion is too deep" still names the user's function at the same call depth.

**III-T6. F6: in-place reuse in native code.**
- Files: `crates/sky-jit/src/module.rs` (`generic`, `push_operands`), `runtime.rs` (`sky_rt_op`, `fused`), `kernel.rs` (`Kernel::run`).
- Deliverable: `sky_rt_op_consume(site, operand)` using `Lifetimes::reuse`.
- Acceptance: elementwise 10^7, one thread, ≤ 1.15× C per pass; peak RSS ≤ the SIR engine's on the same program; `MEM-001` notes in `--explain` now also hold for native code.

**III-T7. F8: compile cache (after a security decision).**
- Files: likely `crates/sky-jit/src/{lib.rs,module.rs,aot.rs}` and `crates/sky-cli` [LIKELY].
- Deliverable: first, a written decision on whether a cache of compiled machine code is compatible with non-negotiable 3 (`spec/v0.1.md:1036`); then the cache, keyed by SIR hash, Sky version and target, stored with owner and mode checks.
- Acceptance: nbody's fixed cost ≤ 10 ms on a cache hit; a corrupted or foreign cache entry is ignored and recompiled, never executed; deleting the cache changes nothing but time.

**III-T8. Scientific-track fixes, handed to their owner.**
- Items: F2, F4, F5, F7, F9, F10, with the designs, files and gates of §5 and §5.1.
- Deliverable: one issue per fix, carrying R10's design, gate and test plan; F7 starts with the self-test harness (`r10/accdet/fmaeq*.c` ported into `sky-linalg`'s tests) run on M1, M3 and M4.
- Acceptance: each issue exists with its gate stated per unit by slope; none is scheduled inside G1–G4 ([Part XXV](part25-roadmap-12-months.md)).

---

## 9. Measurements that decide

| Question | Measurement | Decides |
|---|---|---|
| How much of nbody's 3.2× is spills rather than checks? | prototype F2 behind a flag; repeat the instruction census and the slope | whether F2 alone reaches ≤ 2.0× C, or a register-allocation fix is needed |
| Does Accelerate equal the FMA chain beyond this machine? | `fmaeq` suite on M1, M3, M4 and two macOS versions; `beta = 1`, `alpha = −1`, k ≥ 16,384; find the shape-class thresholds | whether F7's Accelerate route ships, and for which classes |
| What do math functions cost inside kernels? | extend `crates/sky-jit/benches/kernels` with softmax-, GELU- and SiLU-like bodies | the value of F10 |
| Is the NaN-sentinel return compatible with multi-result functions and thread thunks? | F3 prototype under the differential fuzzer with threads on | whether F3(b) is safe |
| Are E-cores the stragglers in 10-thread kernels? | 6 against 10 threads at mid sizes | F9, and II-T6's choice |
| Why is the SIR interpreter's call path slower than the tree-walker's? | profile `crates/sky-ir-interp/src/machine.rs:1211-1248` on fib | an interpreter fix, or none |
| What does a Sky-written sampler cost after F1? | the III-T1 sampler case, before and after III-T2 | whether user-written CPU logits processors are viable at all |

[EXPERIMENTAL: all open in R10 §8]

---

## 10. Risks and negative results

The negative results are listed in §4.

| Risk | Likelihood | Consequence | Mitigation |
|---|---|---|---|
| Roadmap A work leaks into G1–G4 | LIKELY: the fixes are concrete and tempting | the inference critical path slips | §5.2: only III-T1 and II-T6 before G1; the rest after G4 or by another owner |
| F7's Accelerate route rests on undocumented behaviour | VERIFIED as undocumented | a macOS update silently changes bits | the self-test gates speed, never results: on a mismatch Sky's own FMA kernel runs and gives the same bits |
| F7 changes golden outputs | VERIFIED: products change bits once | a new edition and a golden refresh | edition-gated, announced in the spec |
| F8 stores compiled machine code on disk | LIKELY conflict with non-negotiable 3 | security objection; a cache that executes foreign code | III-T7 starts with the decision; entries keyed and validated, never trusted from other users |
| F2's region replay is subtle | LIKELY | a wrong first error, or a missing one | differential fuzzing with overflow inputs and diagnostics compared byte for byte, as for fused kernels |
| F1 changes the runtime ABI | VERIFIED: new exports in `aot.rs` and `sky-rt` | stale `libsky_rt.a` breaks `sky build` | build tests in CI on both operating systems (II-T7) |
| the slope method hides super-linear effects (caches, page faults) | LIKELY at sizes that cross a cache level | a misleading per-unit ratio | choose N1 and N2 on the same side of each cache boundary; report both raw times |
| the 22.5 ms sampler figure is read as an argument about the GPU | LIKELY misreading | confusion in reviews | state it as the cost of Sky CPU code today; the GPU decision stands on the round-trip argument as well (§7.3) |

---

## 11. Disagreement with the spine

One point, a correction of fact rather than of a decision.

**D6 describes Sky's default numerics as "the same bits everywhere".** That is the spec's claim (`spec/v0.1.md:640`), but it does not hold today for transcendental functions across operating systems, because `sky_interp::math` calls the platform libm (`crates/sky-interp/src/math.rs:189-215`; §3.7). It holds per operating system, and F10 makes it hold everywhere. Two consequences for the inference work:

- the decision D6 itself is unaffected: its `inference` contract already keys reproducibility on chip, OS, build and tuning entry;
- D6's proposed `--numerics=portable` mode (the same bits across Apple GPU generations) will meet the same problem on the GPU: `exp`, `rsqrt` and similar functions come from the operating system's Metal compiler and math library, so portable bits will need Sky-owned implementations of those functions in MSL, as F10 does for the CPU [LIKELY: Metal does not document bit-exact transcendental results across OS versions; EXPERIMENTAL: compare softmax and SiLU outputs bit for bit across two macOS versions and two GPU families].
