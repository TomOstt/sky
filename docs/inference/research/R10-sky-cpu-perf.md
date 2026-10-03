# R10: Sky's CPU performance today. Where it wins, where it loses, and why ("Roadmap A")

Author: research agent R10. Date of measurements: 2026-10-01/02. Repo: `~/sky` at `a99e7e3` (HEAD, read-only).
Raw logs and scripts: `$S/research/r10/` (where `S=/private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad`).

**Labels.**
- **VERIFIED**: backed by a code citation (`path:line`), a URL, or our own measurement with its command and raw numbers.
- **LIKELY**: reasoned, with the reason given.
- **EXPERIMENTAL**: needs a test.
- **MOONSHOT**: a long shot.

---

## 0. Summary: the ten things the architect needs to know

1. **The published table overstates most CPU gaps because it times whole processes.**
   - `benches/compare/RESULTS.md` reports the median wall time of a whole process.
   - Sky's fixed cost is 4.7 to 5.7 ms before any work: process start, then lex, parse and check. C's is 1.45 ms. Native runs add Cranelift compile time on top.
   - We re-measured each workload at two sizes and took the slope, which gives the true per-unit ratio:

     | Workload | RESULTS.md (whole process) | Slope (per unit of work) |
     |---|---:|---:|
     | nbody vs C | 7.5× | **3.2×** per step |
     | fib vs C | 1.36× | **2.7×** per call |
     | mandelbrot vs Rust (identical semantics) | 1.46× | **1.20×** per inner iteration |
     | rk4 vs C | 1.39× | **1.89×** per step |
     | elementwise-1e7 vs C | 2.06× | **0.6×** per pass on 10 threads; **1.17–1.36×** on 1 thread |

     VERIFIED: §3.
   - So the fib gap is *under*stated by RESULTS, and nbody's is *over*stated. Two thirds of nbody's 32 ms is fixed cost:
     - ~5 ms front end;
     - ~5 ms SIR lowering and optimisation;
     - **~12 ms of CLIF generation plus Cranelift compile**.

     VERIFIED: §4.8.
2. **The worst CPU gap is missing from the suite: reading one element of a large array in a native loop.**
   - `x[i]` on a non-small array costs **148 ns per access in native code, 57× C**. VERIFIED: §4.4.
   - Every non-small-array operation in native code is a call into a boxed-`Value` runtime interpreter (`crates/sky-jit/src/module.rs:2229-2245` → `module.rs:2271` `generic` → `runtime.rs:583` `sky_rt_op`).
   - This is the item that matters most for the LLM direction. Sampling, tokenizer tables and logits processors are exactly this pattern.
3. **The "small vectors allocate" weakness is already fixed in native code.** It is only true in the interpreters.
   - Native code scalar-replaces statically shaped arrays of 16 entries or fewer (`crates/sky-jit/src/module/small.rs:1-8`, `:53` `MAX_SMALL = 16`). VERIFIED.
   - The remaining cost in nbody's native loop is visible in a census of its 1765 static instructions (VERIFIED: disassembly, §4.1):

     | Share of loop instructions | What it is |
     |---:|---|
     | 36% | finiteness/zero/domain checks |
     | 19% | SLP lane shuffles |
     | 14% | spills |
     | 23% | actual floating-point arithmetic |

4. **Fused elementwise kernels already run at C speed.**
   - Single thread, output preallocated, ratio to the best of C/Rust (VERIFIED: `sky-jit` `kernels` bench, §4.5):
     - axpy: **1.00–1.01×**;
     - hypot: **1.00×**;
     - chain6 (with a division): **1.14–1.17×**.
   - The owner's gate "fused elementwise ≤ 1.1× C" is therefore *already met at kernel level* for two of three kernels.
   - Whole-program ArrayChain gaps come from front-end parsing of huge literals and from building the inputs (~50 ms of 93 ms at 1e7), not from the chain.
5. **Matmul is two separate deficits, and the reproducible kernel is already near its ceiling.**
   - Sky's reproducible GEMM runs at **24.6–25.0 GFLOP/s on one core**. That is ~89% of the theoretical *no-FMA* NEON peak (27.9 GFLOP/s at 3.49 GHz).
   - Sky's own FMA ("fused") kernel runs at **47–48 GFLOP/s**, the same as faer and matrixmultiply.
   - Accelerate runs at **352–378 GFLOP/s on 1 thread and 628–689 on all threads** (AMX).
   - So the loss decomposes as **2× from the FMA ban × ~7.5× from AMX**. Blocking, packing and threads are not the problem. VERIFIED: §4.7.
6. **Surprising reproducibility fact (EXPERIMENTAL, measured on this machine): Accelerate's `cblas_dgemm` is bit-identical to the sequential k-order FMA chain.**
   - The chain is `c = fma(a[i,k], b[k,j], c)` from +0.0.
   - All entries matched at n = 16, 64, 256, 512 and 1000, at odd sizes (1023×1021×1019), with long k (8192), for vector @ matrix and with a transposed B, on 1 thread and on the default thread count (row-major, beta = 0); see §4.9.
   - **It does not hold for tiny products (7×5×3: 28 of 35 entries) or matrix @ vector (2048×1×2048: 83 of 2048).**
   - Only 15–30% of entries match the unfused chain that Sky defines today.
   - If Sky redefined reproducible products as FMA chains, which its own `fused` kernel already computes deterministically, then a self-tested Accelerate path could give reproducible AMX speed on Apple machines.
   - This is the only route by which "matmul ≤ 1.2× Accelerate" is realistic in the default mode.
7. **Accelerate is not uniformly thread-invariant.**
   - `dgemm` and `dgemv` gave identical bits for 1/2/4/8/default threads.
   - `dpotrf` at n ≥ 512 gives different bits on 1 thread than on ≥ 2 threads. It is stable run to run at a fixed thread count. VERIFIED: §4.9.
   - So "call Accelerate for Cholesky" cannot be part of reproducible mode.
8. **Calls are compiled natively. The harness's "every call is interpreted" text is wrong for native and auto.**
   - The native call protocol costs 2.0 ns per call against C's 0.74 ns. It consists of a stack-budget check, a depth check, a status return with its compare, and a finiteness check of the result.
   - Neither the SIR pipeline nor Cranelift 0.121.2 has an inliner (`sky-opt/src/pipeline.rs:10-60`; `cranelift-codegen-0.121.2/src/context.rs:150-189`).
   - The SIR interpreter is **slower than the tree-walker on calls** (fib-30: 1622 vs 718 ms in RESULTS). This is a negative result.
9. **Cranelift has no auto-vectorizer, no unroller and no inliner. Sky does its own vectorization.**
   - Fused kernels use explicit F64X2 with 4× unroll; scalar code uses SLP pairs (`module/slp.rs`).
   - F64X2 is already the full NEON width for f64.
   - The codegen weaknesses we saw are register allocation (heavy spilling in nbody) and the check branches, not instruction selection.
   - **Recommendation: no LLVM backend in Roadmap A** (§5, F11).
10. **For the LLM-on-GPU pivot, almost none of the f64 CPU work matters for tokens per second.** What matters on the CPU side:
    - (a) inline element access and in-place mutation of arrays (sampling, tokenizer, KV-cache bookkeeping);
    - (b) an integer type (Sky has only f64 numbers);
    - (c) cheap calls and inlining for orchestration code;
    - (d) not adding latency per token.

    The f64 BLAS, FMA, AMX, check-deferral and libm items are *scientific-Sky* work (§6).

---

## 1. Provenance and method

**Machine.** VERIFIED: `RESULTS.md` header and `sky hw`.
- Apple M2 Pro: 10 cores (6P + 4E), 16 GB.
- macOS 26.6.1 (Darwin 25.6.0).

**Toolchain.**
- The repository pins Rust **1.85.1** (`~/sky/rust-toolchain.toml`).
- Homebrew's `rustc` on PATH is 1.84.1. We built with the rustup toolchain `~/.cargo/bin/cargo` (1.85.1), which matches `RESULTS.md` ("rustc 1.85.1").
- Apple clang 17.0.0.

**Build** (under the lock):

```
bash $S/benchlock.sh ~/.cargo/bin/cargo build --release -p sky-cli -p sky-bench --manifest-path ~/sky/Cargo.toml
```

- It finished in 14.9 s (incremental).
- Benches: `cargo bench -p sky-linalg --bench blas --no-run` and `cargo bench -p sky-jit --bench kernels --no-run`.

**Timing harness.**
- `$S/research/r10/timeit.py` runs a subprocess: 2 warmups (1 in some sets), then 7–9 timed runs. It reports median, min and max wall time.
- Every timed command ran under `bash $S/benchlock.sh`. Other agents were measuring GPU and memory work on the same machine, so we report medians and use mins where the medians were noisy (flagged).
- Gotcha: the *first* execution of a freshly linked binary on macOS costs ~350–400 ms (code-signing check). One `--warmup 0` probe run showed C nbody at "419 ms". Warmups are mandatory, and they matter for `sky build` outputs too. VERIFIED: `r10/compare_nosky.log`.

**Slope method.** Each workload was run at two sizes (N1, N2). The per-unit cost is Δt/ΔN, which cancels process start-up, the front end, compiles and input construction. The fixed cost is t(N1) − N1·slope.

**Templates.** Benchmark programs were rendered at full size with `sky-bench compare --no-sky --emit-sky $R/sky ...`. Size variants were made with `sed` into `$R/var/`.

---

## 2. The existing benchmark suite and its latest results (reproduced)

**Sources.**
- `benches/compare/{sky,c,rust,python,numpy,lib}` (13 cases).
- Harness: `crates/sky-bench/src/compare/{mod.rs,workloads.rs,tools.rs,report.rs}`.
- Engine benchmark: `crates/sky-bench/src/bench.rs` over `benches/programs/*.sky`.
- Docs: `docs/guide.md:166-190` ("Benchmarks (`sky-bench`)").
- Results: `benches/compare/RESULTS.md`, generated 2026-10-01 at Sky revision `88edceb83a66` (a PR-branch commit squash-merged as `b18f4d6`, #44). HEAD `a99e7e3` adds #45–#53 (dogfood fixes, docs).

**Configuration.**
- C is compiled with `-O3 -march=native` (`tools.rs:166-180`). It is **not** given `-ffp-contract=off`, so Apple clang may contract `a*b+c` into FMA.
  - LIKELY, from clang's default `-ffp-contract=on`.
  - Corroborated: C's mandelbrot prints 7971151 while Rust and Sky print 7992339 (`r10/m1_scalar.log`).
- Rust uses `-C opt-level=3 -C target-cpu=native`. Rust never contracts.

### 2.1 Key rows of `RESULTS.md`

Median wall time. VERIFIED: copied from `benches/compare/RESULTS.md`.

| case | C -O3 | Rust | BLAS/LAPACK | CPython | Sky ast | Sky sir | Sky native | Sky auto | Sky ast relaxed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nbody (1e5 steps) | 4.58 ms | 5.97 | – | 610 | 2714 | 1536 | 34.3 | 34.3 | 2676 |
| matmul-64 ×100 | 8.66 | 8.50 | 6.03 | 1162 | 7.95 | 8.80 | 9.82 | 9.77 | 5.88 |
| matmul-256 ×8 | 21.5 | 21.8 | 3.92 | 6134 | 11.3 | 12.2 | 13.5 | 13.5 | 7.82 |
| matmul-512 ×2 | 42.3 | 41.6 | 4.95 | 12.3 s | 14.1 | 15.7 | 16.1 | 16.1 | 10.9 |
| cholesky-256 ×10 | 16.5 | 18.7 | 8.17 | 1108 | 14.1 | 15.7 | 16.4 | 16.2 | 7.32 |
| cholesky-32 ×5000 | 15.5 | 17.7 | 10.1 | 2108 | 39.9 | 40.1 | 41.0 | 40.2 | 36.8 |
| elementwise-1e6 ×20 | 15.8 | 15.3 | – | 2110 | 241 | 25.2 | 24.4 | 24.1 | 245 |
| linear_chain-1e6 ×20 | 10.4 | 12.6 | – | 1571 | 156 | 19.4 | 19.4 | 19.5 | 170 |
| elementwise-1e7 ×2 | 41.1 | 40.1 | – | 2651 | 320 | 84.9 | 97.0 | 94.6 | 305 |
| linear_chain-1e7 ×2 | 36.5 | 37.7 | – | 2129 | 208 | 61.4 | 69.4 | 68.7 | 202 |
| mandelbrot 300², 500 it | 24.8 | 26.6 | – | 912 | 13.4 s | 11.7 s | 39.2 | 38.9 | 13.3 s |
| rk4 2e5 | 14.3 | 14.4 | – | 234 | 1330 | 942 | 20.3 | 20.0 | 1338 |
| fib-30 | 8.42 | 8.45 | – | 124 | 718 | 1622 | 11.5 | 11.9 | 730 |

**Floors** (empty program):
- C process: 1.45 ms;
- `sky run` ast: 4.82 ms; sir: 5.15 ms; native: 5.58 ms; auto: 5.32 ms.

**Headline geometric means** for the best Sky engine:
- vs C -O3: 1.40×;
- vs Rust: 1.32×;
- vs BLAS/LAPACK: 2.36× (5 cases: matmul-64 1.32, matmul-256 2.89, matmul-512 2.85, cholesky-256 1.72, cholesky-32 3.94);
- vs CPython: 0.02×.

### 2.2 What the harness's generated "Interpretation" gets wrong

These explanations are canned per workload kind (`compare/report.rs`). The text says so: "not profiler output".

| RESULTS.md says | Reality (with evidence) |
|---|---|
| Calls (fib): "every call is interpreted" | False for native and auto. Calls are native `call` instructions (CLIF in §4.3). The per-call cost is 2.0 ns vs C 0.74 ns (2.7×). fib's 1.36× ratio is mostly the floor. VERIFIED. |
| SmallVector (nbody): "each vector operation allocates a small array and checks every element" | False for native: arrays of ≤16 entries are scalar-replaced (`module/small.rs`). True for ast and sir. Of the 32 ms, ~23 ms is fixed cost (front end + ~12 ms Cranelift compile). Per step it is 3.2× C. VERIFIED. |
| Scalar (mandelbrot): "each operation is dispatched by an interpreter" | False for native: the loop is machine code. Per iteration it is 1.20× Rust. The remaining cost is the per-op checks. VERIFIED. |
| ArrayChain: "unfused operation allocates..." and "includes building the inputs" | The second clause is the real story: input construction plus front end ≈ 70 of 93 ms at 1e7. The fused kernel is C-speed. VERIFIED. |
| Cholesky: "unblocked factorisation with finiteness checks" | n = 32 does use the plain loops (`chol.rs:147-150`). About a third of the per-iteration time is boxed runtime dispatch for `Vector[32]`, which exceeds `MAX_SMALL`. VERIFIED/LIKELY (§4.7). |
| Matmul: "Sky beat C" | True, but C is a naive triple loop at 13.8 GFLOP/s. Single-threaded, Sky's packed kernel is 1.76× faster per product; with 10 threads, much more. VERIFIED. |

---

## 3. Re-measurement at HEAD: per-unit ratios

Commands: `r10/m1_scalar.sh`, `r10/m2_arrays.sh`, `r10/m5.sh`. Logs: `r10/m1_scalar.log`, `r10/m2_arrays.log`, `r10/m5.log`. Numbers are medians in ms; "min" marks a case where we used the minimum because the medians were noisy.

### 3.1 Scalar, calls, small vectors

Sky is the native engine, default threads.

| workload | N1 → N2 | C t1 / t2 | Rust t1 / t2 | Sky t1 / t2 | per-unit C | per-unit Rust | per-unit Sky | Sky/C | Sky/Rust |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| fib (calls) | fib 30 → 32 (2.69M → 7.05M calls) | 4.57 / 7.79 | 6.07 / 8.38 | 12.11 / 20.84 | 0.74 ns/call | 0.53–0.72 | **2.00** | **2.7×** | 2.8–3.8× |
| nbody (5 bodies) | 1e5 → 4e5 steps | 5.42 / 13.70 (min) | 6.03 / 17.70 (min) | 31.69 / 57.81 (min) | 27.6 ns/step | 38.9 | **87.1** | **3.2×** | 2.2× |
| nbody, `SKY_SLP=off` | same | | | 34.00 / 68.71 (min) | | | 115.7 | 4.2× | 3.0× |
| mandelbrot 300² | maxit 500 → 2000 | 25.34 / 93.30 | 26.57 / 98.41 | 40.17 / 126.66 | 67.96 ms per Δ | 71.84 | 86.49 | **1.27×** | **1.20×** |
| rk4 | 2e5 → 8e5 steps | 9.01 / 29.02 | 10.26 / 33.23 | 20.83 / 58.69 | 33.4 ns/step | 38.3 | **63.1** | **1.89×** | 1.65× |
| index loop `s = s + x[i] + 1` over a 1e6 vector | 2.5e5 → 1e6 iterations | 3.61 / 5.57 | – | 53.83 / 165.10 | 2.6 ns/iter | – | **148** | **57×** | – |
| same, SIR engine | | | | 106.6 / 397.0 | | | 387 | 149× | |

Notes:
- `SKY_SLP=always` equals `auto` on nbody (59.15 vs 59.25 ms at 4e5).
- C's and Rust's mandelbrot outputs differ (7971151 vs 7992339). C fused into FMA; Sky and Rust match bit for bit.
- The fixed parts of Sky's time:

  | workload | fixed cost | of which |
  |---|---:|---|
  | nbody | ~23 ms | §4.8 |
  | fib | ~6.7 ms | floor 5.7 + compile ~1 |
  | rk4 | ~8 ms | |
  | mandelbrot | ~11 ms | includes the exterior-point work |

### 3.2 Arrays

| workload | per unit | C (1 thread) | Sky native, 10 thr | Sky native, 1 thr | Sky sir, 10 thr | Sky sir, 1 thr | Accelerate |
|---|---|---:|---:|---:|---:|---:|---:|
| elementwise 1e7 (`sqrt(X*X+Y*Y)*0.5+D`) | ms per pass | 3.7 | **2.3** | 5.0 (1.36×) | 3.8 | 4.3 (1.17×) | – |
| elementwise 1e6 | ms per pass | 0.30 | **0.20** | 0.39 (1.31×) | 0.33 | 0.47 (1.56×) | – |
| linear_chain 1e6 | ms per pass | 0.21 | **0.134** | – | 0.28 | – | – |
| matmul 512 (whole repetition: product + rescale) | ms per product | 19.4 (13.8 GFLOP/s, naive loop) | 2.97 (90) | 11.0 (24.3) | – | – | 0.61 (438) |
| matmul 512, `--numerics=relaxed` (Accelerate inside Sky) | ms per product | | 0.80 (338) | | | | |
| cholesky-32 (factor + 2 solves) | µs per solve | 2.72 | 6.49 (2.4×) | – | – | – | 1.47 |
| cholesky-32, relaxed | µs per solve | | 5.49 | | | | |

**Fixed-cost decomposition of elementwise-1e7.** VERIFIED: `r10/m3.log`.

| Component | Time |
|---|---:|
| `sky check` (front end on the 290 KB source) | 16.1 ms |
| Build inputs only (front end + lowering + 3 outer products + print), 10 threads | 70.6 ms |
| Same, 1 thread | 97.5 ms |
| Same with R = 0 (adds the final `ROW @ X @ COL` reduction) | 75.2 ms |
| R = 2 (the benchmark) | 93.25 ms |

- So the 2 passes are ~4.6 ms of 93.
- C's fixed cost is ~33 ms: initialising 2 arrays of 80 MB with page faults.

**Memory.** `/usr/bin/time -l`, 1 thread, elementwise-1e7 (VERIFIED: `r10/m4.log`):

| | native | sir | C |
|---|---:|---:|---:|
| Peak RSS | 526 MB | 365 MB | 161 MB |
| Page reclaims, R = 2 and R = 6 | identical (32362) | identical (22545) | |

- The per-pass result allocation does **not** add page faults; macOS malloc recycles the freed 80 MB block. Negative result: my "fresh pages per pass" hypothesis was wrong.
- The native/sir difference per pass is the extra write stream into a new buffer (no in-place reuse in native).

---

## 4. Weakness by weakness: the code-level cause

### 4.1 Small vectors: "allocate arrays and repeat checks"

**Interpreters (ast and sir): true.**
- `Value::Array { dims: Vec<usize>, data: Vec<f64>, dim }` (`crates/sky-interp/src/value.rs:35-39`). Every array result is two heap allocations, `dims` and `data`.
- `array.rs:41-46` `all_finite` makes a second pass over every result.
- `array.rs:72-99` `scale` clones `dims` and collects a new `Vec` unless the operand is owned.
- The SIR interpreter moves operands that die at their operation, so steady-state loops allocate nothing:
  - `sky-ir/src/lifetime.rs:1-60`;
  - `sky-ir-interp/src/machine.rs:12-31`;
  - test `crates/sky-bench/tests/allocations.rs:242` (`nbody_loop_allocates_nothing_per_iteration_on_sir`).
- Dispatch on boxed `Value`s still costs ~15 µs per nbody step on SIR (1536 ms / 1e5).

**Native: false for ≤16 entries.** VERIFIED.
- Arrays with a statically known shape and at most 16 entries are scalar-replaced into N `f64` SSA values. They are passed to and from functions as N `f64` parameters and results, with no handle and no allocation (`crates/sky-jit/src/module.rs:16-40`, `module/small.rs:1-8`).
- `MAX_SMALL = 16` was chosen by measurement (`small.rs:42-53`): a 4×4 matmul loop takes 0.20 s as handles vs 0.01 s as scalars, but unrolling costs ~3 ms of compile per 4×4 product.

**What nbody's native loop actually executes.** VERIFIED: `sky build --emit=obj` of `$R/sky/nbody.sky`, `otool -tV`, census by `r10` script (`nbody_main.s`). Static census of the loop body (0x69c–0x2230; back-edge 0x221c → 0x520), 1765 instructions:

| category | count | share |
|---|---:|---:|
| finiteness checks (`fsub x,x`) | 204 | 12% |
| `fcmp` (finite, zero divisor, sqrt domain) | 215 | 12% |
| conditional branches | 215 | 12% |
| moves, lane inserts/extracts, `dup` (SLP packing) | 342 | 19% |
| stack spills and reloads | 252 | 14% |
| FP arithmetic (including the adds that combine check lanes) | 411 | 23% |
| unconditional branches, constant loads, other | ~126 | 7% |

Further observations in the same disassembly:
- The hot loop reloads a zero vector from the stack (`ldr q31, [sp,#0x4d0]`).
- It inserts lanes (`mov.d v17[1], v31[0]`) to pack two of three vector components into F64X2, then extracts lanes (`mov d28, v24[1]`) to check them.
- Every `sqrt` is followed by `fadd d, d, 0.0` (normalising −0 to +0, Sky semantics) and preceded by a `fcmp d, 0; b.mi` domain check.
- Every division has a `fcmp d, 0; b.eq` zero check.

Interpretation:
- The ~3× per-step gap vs C is the check machinery (~36% of instructions plus branch-predictor load), the SLP shuffles (~19%) and the spills (~14%).
- Allocation plays no part. SLP still pays overall: turning it off is 33% slower (115.7 vs 87.1 ns per step).

**Arrays of more than 16 entries** (`Vector[32]` in cholesky-32, any mid-size matrix) become runtime handles. Every operation on them is a runtime call (§4.3).

### 4.2 Finiteness, shape and unit checks: where, and whether they are in hot loops

| check | interpreters | native scalar code | native fused kernels |
|---|---|---|---|
| units (dimensions) | dynamic, per op (`sky_interp::arith`) | **static, zero cost**: computed from types at compile time (`module.rs:2329-2339`) | static: settled at kernel compile (`kernel.rs:24-30`) |
| shapes | dynamic, per op (`sky_interp::array`) | static for typed arrays (`small.rs`); dynamic inside the runtime for boxed arrays | static (`runtime.rs:1385-1440`, `fused` checks operand shapes once per call) |
| finiteness | `all_finite` pass per array op (`array.rs:41`); per scalar op in `arith` | **per operation, inline in hot loops**: `is_finite` = `fsub x,x; fcmp ord` (`module.rs:1366-1377`, fn at `:1373`) plus `brif` to a cold block (`finite_or`, `module.rs:1380-1390`). Applied in `arith` (`module.rs:2357-2359`) unless the range analysis proves the op safe (`proven(id)`, `module.rs:1357-1360`; analysis `sky-ir/src/exact/mod.rs:1-60`) | **deferred**: `acc += r - r` in vector registers, one test after the loop, replay via the reference path on failure (`kernel.rs:31-58`); only values reaching the result through a divisor or a math call are checked separately |
| zero divisor | per op | inline `fcmp eq b, 0` unless proven non-zero (`module.rs:2348-2352`) | via the deferred check (x/0 = inf) |
| stack budget, call depth | per call | per call (`module.rs:1393-1399`, `module.rs:2932-2954`) | – |

Check cost measured in kernels (`sky-jit` bench, `r10/kernels_bench.log`), `nochk/sky` time ratio:

| kernel | nochk / sky |
|---|---:|
| axpy at 1e5 | 0.78 |
| axpy at 1e7 | 1.00 (memory bound) |
| hypot | 1.00 |
| chain6 | 0.76–0.77 |

So with deferral the check costs 0–30%. Sky's deferred check is cheaper than naive checking in C: `sky / c+chk` = 0.66–0.84. VERIFIED.

In native **scalar** code the checks are not deferred, and they sit inside the hot loop (§4.1 census). This is the main fixable cost behind nbody, mandelbrot and rk4.

### 4.3 Calls, and what still runs interpreted under `auto` and native

**Native call protocol.** VERIFIED: `sky check --emit=clif` of fib, quoted here.
- Every function takes `(depth: i32, stack_limit: i64, site: i32, args...)` and returns `(status: i32, results...)`.
- The callee entry runs `get_stack_pointer; icmp ult sp, limit; brif` (`stack_check`, `module.rs:1393`).
- The caller does `icmp uge depth, 1000; brif` before the call (`call_op`, `module.rs:2932-2954`) and `icmp ne status, 0; brif` after it.
- The fib result add is finiteness-checked; `n - 1` and `n - 2` are proven safe by the range analysis.
- Measured: **2.0 ns per call vs 0.74 ns for C**.
  - The C version uses `long` (`benches/compare/c/fib.c`); Sky numbers are always f64.
  - Clang also partially converts the recursion into a loop. LIKELY: a standard LLVM transform; the C per-call time is consistent with it.

**No inlining anywhere.**
- The SIR pipeline is: specialize (clones callees by argument facts), fold, egraph, gvn, licm, contracts, dce, powers, fuse, cholesky, transposes (`crates/sky-opt/src/pipeline.rs:10-60`). There is no inliner.
- Cranelift 0.121.2 has none either (the `cranelift-codegen-0.121.2/src` tree has no inliner module; `Context::optimize` is legalize, unreachable-code removal, constant-phi removal and one e-graph pass, `context.rs:150-189`).
- So rk4's `f(y)` is 4 native calls per step. That is LIKELY the main reason rk4 is 1.89× C while mandelbrot, which has no calls, is 1.27×.

**SIR interpreter.**
- "A call still allocates its argument list and its frame" (`crates/sky-ir-interp/src/machine.rs:30`). The value table is resized per activation (`machine.rs:1211-1248`).
- fib-30 costs 1622 ms, about 600 ns per call. That is **slower than the tree-walker's 718 ms** (about 267 ns per call). Negative result.

**Tree-walker.** A `BTreeMap<String, Value>` frame per call, cloning parameter names, with name lookup by string (`crates/sky-interp/src/eval.rs:131-139`, `:168-172`, `:887-916`).

**Still interpreted under native and `auto`.** VERIFIED.
1. **Every operation on a non-small or `unknown`-typed array.** These all go to `generic` (`module.rs:2229-2245`):
   - `MatMul`, `Norm`, `Dot`, `Transpose`, `Solve`, `SolveRight`, `MatMulEx`, `DiagMul`, `Index`, `Slice`, `Zeros`, `Identity`, `Len`, `ArrayLiteral`, `ConstArray`, `Show`;
   - elementwise ops on big arrays;
   - `Fused` (`module.rs:2258-2265`).

   `generic` (`module.rs:2271-2284`) does the following:
   - pushes each operand with a runtime call (`sky_rt_push_num`, `push_ref`, `push_elem`);
   - calls `sky_rt_op(site)` (`runtime.rs:583-606`), which matches on `OpKind` and runs the interpreter's own functions on `Rc<Value>`s (`eval_op`, `runtime.rs` ~`1100-1205`);
   - allocates a handle for the result;
   - the generated code then unboxes it, or keeps it as a handle and releases it at the end of the block.

   Measured: **148 ns per `x[i]`** (§3.1).
2. **`fused` operations** are interpreted element by element until 512 elements have run for that op (`runtime.rs:1318-1322`, `1358-1375`). They are then compiled once and cached in a `HashMap` keyed by op address. Big kernels compile immediately when they qualify for threads.
3. **Values of type `unknown`** (unannotated, unspecialised parameters) run boxed in the runtime.
4. **Deoptimisation**: if a typed slot receives a mismatching value, the whole program re-runs on the SIR interpreter, skipping the output already written (`module.rs:93-107`).
5. `auto` falls back to SIR, then to the tree-walker, if native compilation is refused (`module.rs:261-266` `refuse` sites are mostly malformed-IR guards).

### 4.4 Big-array element access: the gap the suite does not measure

- **Microbenchmark** (`$R/micro/idx1e6.sky`): `for i in range(N): s = s + x[i] + 1.0` over a `zeros(1000000)`-based vector.
- **SIR**: `%21 = index %x, %i : num[1; ()]` inside the `while` body (`r10/micro`, `--emit-ir`).
- **Measured**: native 148 ns per iteration, SIR 387 ns, C 2.6 ns. VERIFIED: `r10/m5.log`.
- **Cause**: `Index` on a non-small array goes through `generic`. The sequence is push_ref, then push_num, then `sky_rt_op`, then `eval_op` (bounds checks in `sky_interp::array::index`), then `Rc` allocation, then the handle table, then unbox, then release. That is ~5 runtime calls and 1–2 allocations per element.
- **Also missing from the language**: element writes. "Arrays are immutable values (v0.1 has no element writes)" (`value.rs:30-34`).
- Small arrays with a *run-time* index use an O(N) `fcmp`/`select` chain (`module/small.rs:221-230`).

### 4.5 Array chains and fusion: what exists and where it stops

What exists:
- `sky-opt/src/fuse.rs:1-56` fuses **trees** of elementwise candidates: `add`, `sub`, `mul`, `div`, `neg`, math functions.
- A producer is absorbed only if its result has **exactly one use** and the consumer is a candidate in the same block. Scalars broadcast. Failing operations never move past effects (`fuse.rs:30-45`).
- Kernels (`crates/sky-jit/src/kernel.rs:1-100`):
  - explicit F64X2 with 4 vectors (8 elements) per iteration, then a scalar tail;
  - loop-invariant values computed once and splatted;
  - unaligned-tolerant loads; `MemFlags::new().with_notrap()` and `MemFlags::trusted()` loads with **no per-element bounds checks** (`kernel.rs:754`, `:865`);
  - deferred finiteness check;
  - threads via deterministic chunks;
  - output allocated uninitialised (`kernel.rs:1308-1315`).

Measured kernel speed, ns per element, single thread, preallocated output. VERIFIED: `r10/kernels_bench.log`, command `CARGO_TARGET_TMPDIR=$R/kbench RUSTC=~/.cargo/bin/rustc bash $S/benchlock.sh target/release/deps/kernels-6601d69be4736129 --bench`.

| kernel | n | sky | nochk | scalar | best C/Rust | sky/best |
|---|---:|---:|---:|---:|---:|---:|
| axpy | 1e5 | 0.234 | 0.181 | 0.295 | 0.233 | 1.00× |
| axpy | 1e7 | 0.263 | 0.261 | 0.312 | 0.262 | 1.01× |
| hypot | 1e7 | 0.301 | 0.303 | 0.602 | 0.302 | 1.00× |
| chain6 | 1e7 | 0.357 | 0.271 | 0.675 | 0.309 | **1.16×** |

Kernel compile time is 0.16–0.26 ms.

Where fusion stops. VERIFIED unless marked:
1. **Multi-use intermediates (DAGs).** `T = X*Y; Z = sqrt(T) + T*0.5` keeps `%T = mul ...` as a materialised array (`$R/micro/fuse_dag.sky`, `--emit-ir`). GVN even merged X and Y.
2. **Reductions.** `dot`, `@` and `norm` consume a materialised array (`%s = matmul %Z, %Z` after `%Z = fused ...`). No producer-to-reduction fusion. There is no `sum` builtin; sums are written `ones @ X`.
3. **Matmul epilogues** (`A@B + C`, scaling): not fused.
4. **Broadcasts other than scalars** (row or column vectors): not candidates. The benchmark itself needs a full array `D` of 0.25s because Sky has no `array + scalar`.
5. **Math functions inside kernels** are per-lane scalar calls into `kernel_rt::math`. That builds two `Value`s and calls `sky_interp::math::apply` (`kernel_rt.rs:30-41`). LIKELY ~5–10 ns per element for exp/tanh-heavy chains (softmax, GELU, SiLU): an extern call plus a match plus libm. Not measured here; EXPERIMENTAL.
6. **Native does not reuse buffers.** There is no use of `sky_ir::Lifetimes` in `sky-jit`; the SIR interpreter does reuse. Measured: native 1-thread 5.0 vs SIR 1-thread 4.3 ms per pass; RSS 526 vs 365 MB.

### 4.6 Cranelift codegen for loops

- **Vectorization.** Cranelift 0.121.2 has no auto-vectorizer, no loop unroller and no inliner. VERIFIED: `cranelift-codegen-0.121.2/src/context.rs:150-189`; the only "vectoriz" hits in `src/` are in NaN canonicalisation and a comment. Its optimiser is one e-graph pass: GVN, LICM-style placement during elaboration, constant folding, ISLE simplifications, rematerialisation of `iconst`/`f64const`/ALU-with-immediate (`opts/remat.isle`).
- **SIMD width.** NEON on aarch64 is 128-bit, so F64X2 is all there is for f64 on M1–M3. M4 adds SME (streaming SVE plus matrix tiles), which is a different programming model (MOONSHOT for f64 GEMM; §5 F7).
- **Bounds checks.** Fused kernels have none per element. Small arrays: compile-time positions, or a validated O(N) select chain. Big arrays: none inline, because the access is a runtime call (§4.4).
- **Register allocation** (regalloc2 0.12.2). In nbody, 14% of loop instructions are spills and reloads. The loop carries 30 f64s (10 `Vector[3]`) plus F64X2 packed copies, against 32 V registers. A zero vector constant is reloaded from the stack rather than rematerialised. LIKELY: `vconst`/splat not covered by `remat.isle`.
- **Settings**: `opt_level=speed`, `is_pic=false` (`crates/sky-jit/src/lib.rs:42-59`).

### 4.7 sky-linalg GEMM and Cholesky vs Accelerate

**Design** (`crates/sky-linalg/src/lib.rs:1-119`, `gemm.rs`, `micro.rs`, `tuning.rs`, `chol.rs`, `tri.rs`). VERIFIED.
- **GEMM: Goto/BLIS structure.**
  - Cache blocks `mc/kc/nc = 128/256/1024` (`tuning.rs:71-75`, tunable by `sky tune`).
  - B packed into 8-column slivers, shared by workers; A packed per worker into 4-row slivers (`gemm.rs:94-128`).
  - **4×8 register tile** (`micro.rs:20-22`). That is 16 NEON registers of accumulators; LLVM autovectorises the reproducible update (`micro.rs:27-45`, `lib.rs:87-92`).
  - **No FMA** in reproducible mode (spec `spec/v0.1.md:641-647`).
  - Threads: rayon over row chunks above `1 << 21` multiply-adds (`tuning.rs:102`, `gemm.rs:282-318`); a column split for wide, short products (`gemm.rs:320-361`).
  - Products of ≤16 columns, matvec and vecmat have their own unpacked kernels (`gemm.rs:582-890`).
- **Relaxed mode:**
  - NEON `vfmaq_laneq_f64` 4×8 tile (`micro.rs:66-112`);
  - reassociated dot (`reassoc.rs`);
  - Accelerate `cblas_dgemm`, `dpotrf`/`dpotrs` and `dgesv` above size thresholds (`relaxed.rs:34-64`, `accelerate.rs`).
- **Cholesky:** blocked right-looking with nb = 32 (n ≤ 128) or 64 (`tuning.rs:89-95`). The diagonal block and panel are done on transposed copies, and the trailing update uses the GEMM kernel on the lower triangle only (`chol.rs:145-188`). **n ≤ 40 runs the plain reference loops** (`chol.rs:27`, `:147-150`; `reference.rs:122-146`).

**Measured GFLOP/s.** `sky-linalg` bench (`BENCH_SIZES=32,64,256,512,1024 BENCH_THREADS=1,10`, `r10/blas_bench.log`). VERIFIED.

1 thread (the Accelerate column runs with `VECLIB_MAXIMUM_THREADS=1`):

| op | n | Sky repro | Sky fused (FMA) | relaxed | faer | matrixmultiply | Accelerate |
|---|---:|---:|---:|---:|---:|---:|---:|
| gemm | 64 | 21.8 | 38.6 | 217 | 43.3 | 43.9 | 296 |
| gemm | 256 | 24.95 | 48.4 | 339 | 48.3 | 48.3 | 378 |
| gemm | 512 | 24.98 | 47.1 | 639* | 48.8 | 47.7 | 366 |
| gemm | 1024 | 24.60 | 47.4 | 543* | 49.4 | 47.3 | 352 |
| potrf | 32 | 3.09 | 3.08 | 3.39 | 5.54 | – | 12.8 |
| potrf | 256 | 14.8 | 18.6 | 52.5 | 25.2 | – | 110.6 |
| potrf | 1024 | 20.7 | 32.5 | 100 | 37.1 | – | 154.5 |
| gesv | 1024 | 19.2 | 26.4 | 83.9 | 34.7 | – | 66.6 |

\* The in-process relaxed call does not cap Accelerate's threads, so the "1 thread" relaxed figure at 512 and 1024 is multi-threaded AMX.

10 threads:

| op | n | Sky repro | Sky fused | faer | Accelerate |
|---|---:|---:|---:|---:|---:|
| gemm | 256 | 97 | 150 | 119 | 378 |
| gemm | 512 | 133 | 242 | 234 | 689 |
| gemm | 1024 | 137 | 239 | 267 | 628 |
| potrf | 256 | **12.3** (slower than 1 thread) | 13.3 | 22.9 | 113.5 |
| potrf | 1024 | 42.7 | 51.9 | 84.3 | 154.7 |

Why matmul loses:
- **FMA ban, 2×.** The reproducible kernel is at **89%** of the no-FMA peak (27.9 GFLOP/s per P-core: 4 FP pipes × 2 lanes × 3.49 GHz, with multiply and add as separate µops). The fused kernel equals faer and matrixmultiply at 86% of the FMA peak (55.8). LIKELY on the peak model: the M2 P-core has four 128-bit FP/SIMD pipes at about 3.49 GHz, a widely reported microarchitecture figure we did not measure ourselves; the measured ratios agree with it.
- **AMX, ~7.5× on top**, single thread. Multi-threaded, Accelerate reaches ~1.8× its single-thread speed, LIKELY because the M2 Pro's two P-clusters each have an AMX unit (an inference from the jump from 366 to 689). Sky repro scales 5.6× on 6P + 4E.
- **Nothing else.** Blocking, packing, register tile and threads are all fine for NEON. Writing our own NEON kernel cannot reach Accelerate. Only AMX/SME can, and AMX is reachable only through Accelerate.

Why Cholesky loses:
- At n = 32 the plain loops run at 3 GFLOP/s vs Accelerate's 12.8.
- At n = 256 threading *hurts* (12.3 vs 14.8 GFLOP/s); the thresholds are wrong for potrf.
- Large n: 20.7 vs 154.5 GFLOP/s (7.5×), for the same FMA plus AMX reasons, because the trailing update is GEMM.

**The cholesky-32 program, specifically.** VERIFIED: `--emit-ir` of `$R/sky/cholesky-32.sky`.
- The loop body is `%x_3 = solve[cholesky] %K, %50`. The factorisation of the **loop-invariant** literal `K` is redone every iteration.
- C and LAPACK also refactor every iteration (`benches/compare/c/cholesky.c`, `lib/cholesky.c`), so the benchmark is algorithm-parity.
- Per iteration Sky costs 6.49 µs vs C 2.72 µs vs LAPACK 1.47 µs. Estimated split:
  - ~3.5 µs plain-loop factor (potrf at n = 32 is 3.09 GFLOP/s on 10.9 kflop);
  - ~1 µs for the two triangular solves;
  - ~2 µs boxed dispatch of the `Vector[32]` scaling, the solve, two `index` ops and `ones @ x`.

  LIKELY: split by subtraction; not profiled.

### 4.8 Fixed costs: start-up, front end, Cranelift compile

Sources: `r10/m1b.log`, `r10/m3.log`, `r10/m1_scalar.log`. VERIFIED.

| what | ms |
|---|---:|
| C empty process | 1.45 (RESULTS) |
| `sky check` `print(1)` | 4.75 |
| `sky run` `print(1)`: ast / sir / native / auto | 4.69 / 5.47 / 5.67 / 5.73 |
| `sky check nbody.sky` | 5.34 |
| `sky check --emit=clif nbody.sky` (CLIF generated, not compiled) | 12.6 |
| `sky run --engine=sir`, nbody with 1 step | 11.05 |
| `sky run --engine=native`, nbody with 1 step | **23.0** (SLP off: 23.5) |
| `sky build --emit=obj nbody.sky` | 23.7 |
| `sky check elementwise-1e7.sky` (290 KB of literals) | 16.1 |
| `sky check cholesky-32.sky` / `matmul-512.sky` | 6.3 / 7.7 |

Of nbody's native 32 ms:
- ~5.3 ms is process plus front end;
- ~5 ms is lowering plus the SIR pipeline;
- **~12 ms is CLIF generation plus Cranelift** (the energy function plus a fully unrolled, SLP-packed main loop, ~4400 CLIF lines);
- 8.7 ms is the 1e5 steps.

### 4.9 Reproducibility facts relevant to every fix

**(a) Accelerate run-to-run and thread-count determinism.** VERIFIED: `r10/accdet/accdet.c`, `run.sh`, `accdet.log`. FNV hashes of the output bytes, 3 runs each, `VECLIB_MAXIMUM_THREADS` ∈ {default, 1, 2, 4, 8}, n ∈ {64, 512, 1024}.
- `cblas_dgemm` and `cblas_dgemv`: identical bits across all runs and all thread counts.
- `dpotrf`: identical across runs at a fixed thread count. At n = 512 and 1024, **1 thread differs from ≥ 2 threads** (512: `5be98922…` vs `637222ac…`; 1024: `e9f3f6c2…` vs `fb904e7d…`). At n = 64 all thread counts agree.
- Not tested: other Apple chips and other macOS versions. Apple documents no guarantee. EXPERIMENTAL.

**(b) Accelerate dgemm equals the sequential FMA chain.** VERIFIED on this machine and OS; EXPERIMENTAL in general.
- `r10/accdet/fmaeq.c` computes, per entry, `f = fma(a[i,k], b[k,j], f)` for k ascending from +0.0 (scalar, `-ffp-contract=off`), plus the unfused chain `p = p + a*b`. It compares both with `cblas_dgemm(RowMajor, NoTrans, NoTrans, alpha=1, beta=0)`.
- Results:

  | n | threads | entries equal to the FMA chain | entries equal to the unfused chain |
  |---:|---|---:|---:|
  | 16 | default | 256 / 256 | 100 |
  | 64 | default | 4096 / 4096 | 1178 |
  | 256 | default | 65536 / 65536 | 12141 |
  | 256 | 1 | 65536 / 65536 | 12141 |
  | 512 | default | 262144 / 262144 | 39981 |

- So AMX's fp64 accumulate is (on this chip) a correctly rounded FMA applied in k order, and Accelerate does not split k across partial sums at these sizes.

**(b2) Rectangular, odd, long-k, thin and transposed shapes.** VERIFIED: `r10/accdet/fmaeq2.c`, `fmaeq2.log`. Same comparison, `cblas_dgemm` RowMajor, alpha = 1, beta = 0.

| m × n × k | B | entries equal to the FMA chain |
|---|---|---:|
| 1000 × 1000 × 1000 | NoTrans | 1,000,000 / 1,000,000 |
| 1023 × 1021 × 1019 | NoTrans | 1,044,483 / 1,044,483 |
| 64 × 64 × 8192 (long k) | NoTrans | 4096 / 4096 |
| 1 × 2048 × 2048 (vector @ matrix) | NoTrans | 2048 / 2048 |
| 512 × 512 × 512 | **Trans** | 262144 / 262144 |
| **7 × 5 × 3** (tiny) | NoTrans | **28 / 35** |
| **2048 × 1 × 2048** (matrix @ vector) | NoTrans | **83 / 2048** |

- Equality holds for real GEMM shapes, odd sizes, long k, vector @ matrix and a transposed B.
- It **fails for tiny products and for matrix @ vector**. Those take other code paths inside Accelerate, LIKELY a reassociated dot-product kernel for gemv-shaped calls.
- So a self-tested Accelerate route must be **per shape class**, e.g. "all of m, n, k ≥ 16 and n > 1", with Sky's own kernels for the rest. Sky already routes products by shape class (spec `v0.1.md:637`, "Variant choice by shape"), so this fits.

**(c) Transcendental functions are not cross-platform reproducible today.** VERIFIED in code.
- `sky_interp::math` calls Rust `f64::exp/ln/sin/...` (`crates/sky-interp/src/math.rs:189-215`), which is the platform libm: Apple libm on macOS, glibc or musl elsewhere. These differ in the last ulp for some inputs (LIKELY: none of them guarantees correct rounding).
- The spec admits "the platform's math library is within four ulps" (`spec/v0.1.md:805`), while it also claims "the same program prints the same bits everywhere" (`spec/v0.1.md:640`).
- `sky-ode` already uses the pure-Rust `libm` crate for its step controller (`crates/sky-ode/src/dopri.rs:248`), which is the cross-platform fix.

---

## 5. Fixes: designs, expected gains, difficulty and confidence

Difficulty and confidence are both on a 1–5 scale (5 = hardest / most confident). Gains are per unit of work, measured against today's §3 slopes.

### F1. Inline element access for big arrays (buffer representation)

**Problem.** §4.4: 148 ns per `x[i]`, 57× C.

**Design.**
- Add `Rep::Buf` for any value whose static type is a numeric array larger than `MAX_SMALL`. It keeps the handle (ownership is unchanged) plus `(ptr: i64, len: i64)`, obtained from one runtime call, `sky_rt_buf(h) -> (ptr, len)`. Arrays are immutable and refcounted, so the pointer is stable while the handle is live.
- The e-graph's LICM hoists the extraction out of loops, since the handle is loop-invariant.
- `index` with a run-time f64 index becomes inline code:

```text
k  = fcvt_to_sint_sat.i64 i
ok = (fcvt_from_sint.f64 k == i) & (k u< len)   # integral and in range
brif ok, fast, cold                             # cold: sky_rt_fail_op(site, ...) builds the interpreter's error
fast: x = load.f64 notrap aligned ptr + 8*k
```

- With the range analysis, `for i in range(len(x))` (or a constant bound equal to the static length) makes `i` an integer in [0, len). That is `Proof::InBounds`, which `sky-ir/src/exact` already has for small arrays (`exact/mod.rs:50-56`). With the proof there are no checks at all.
- `len` becomes a constant or the loaded `len`.

**Expected gain.** 148 ns → ~1–3 ns per element, i.e. **50–100× on indexing loops**. That puts them at 1–1.5× C, bounded by the serial f64 add chain. LIKELY: the access is then a load plus at most a compare and a branch.

**Difficulty / confidence.** 3 / 4.

**Files.**
- `crates/sky-jit/src/module.rs`: `Rep`, `rep_of`, the dispatch at `:2229-2245`, a new `index_buf`.
- `crates/sky-jit/src/runtime.rs`: `sky_rt_buf`.
- `crates/sky-jit/src/aot.rs`: export the symbol.
- `crates/sky-ir/src/exact/mod.rs`: `InBounds` for big arrays.
- Tests: the differential fuzzer must cover out-of-range and non-integer indices.

**Follow-on (needed for the LLM direction).** Element writes with value semantics: `x = x.with(i, v)`, or `x[i] = v` as sugar. Implement as an in-place store when the operand dies there (the `Lifetimes::reuse` rule) and the runtime refcount is 1; otherwise copy-on-write. Difficulty 3 / confidence 3. It is a language change: spec plus checker.

### F2. Deferred finiteness checks in native scalar code (poison accumulation plus cold replay)

**Problem.** §4.1: ~36% of nbody's loop instructions, and 215 branches, are checks.

**Design.** Port the kernel strategy (`kernel.rs:31-58`) to scalar code at the granularity of a *check region*: a maximal straight-line run inside a block, with no op that has an effect or can observe a non-finite value.
- **Sinks** must be checked before use:
  - divisors;
  - comparisons and branches;
  - math calls (exp(−inf) = 0 launders);
  - index conversion;
  - call arguments;
  - `print`;
  - yielded and loop-carried values;
  - function results.
- **Propagating ops** need no check of their own: `+`, `−`, `*`, `neg`, `abs`, `sqrt`, and a numerator. A non-finite input yields a non-finite result.
- Per region, accumulate only the values that reach a sink: `acc = acc + (v - v)`, as a tree, two lanes at a time.
- Then one `fcmp uno acc, acc; brif cold`.
- The cold block **replays the region** from its inputs, which are SSA values still live at region entry, with today's per-op checks. That reproduces the exact first error, message and spans.
- Regions contain no effects, so a replay is invisible.

Example, mandelbrot's iteration:
- today: ~7 checked ops;
- with F2: 1 check on `zr2 + zi2` before the compare, plus 1 combined check of `(zi, zr)` at the back-edge.

Alternative: the AArch64 FPSR sticky flags (IOC/DZC/OFC) checked once per region. Cranelift has no FPSR access in its IR (VERIFIED: no such instruction in 0.121), so it would need a patch or a call. Not now.

**Expected gain.**
- nbody: **1.4–1.8× per step**. LIKELY: it removes most of the 36% check instructions and the branch pressure; some checks remain at divisors and back-edges.
- mandelbrot: **10–20%**, which closes most of the 1.20× gap to Rust.
- rk4: ~10–20%.

**Difficulty / confidence.** 3 / 3. The correctness argument already exists and is tested for kernels; the subtle parts are region formation and replay codegen.

**Files.**
- `crates/sky-jit/src/module.rs` (`arith`, `finite_or`, `math`, `pow_const`, `inverse`);
- `crates/sky-jit/src/module/slp.rs` (pair checks);
- a new `crates/sky-jit/src/module/defer.rs`;
- fuzz targets (differential, with overflow inputs).

**Pseudocode.**

```text
for block in function:
  regions = split(block, at = ops with effects | calls | prints | branches)
  for r in regions:
    sinks = { v in r : v used by sink-op | v live-out of r }
    need  = backward closure of sinks through propagating ops,
            stopping at sink inputs:
            each v in need is "covered" if some covered descendant propagates it
    emit r's ops without per-op checks; acc = tree_add({ v - v : v in uncovered_roots(need) })
    brif ordered(acc), cont, cold_r
cold_r: re-emit r with today's checks (fail_with → sky_rt_fail*), then jump to the error exit
```

### F3. SIR inliner and a leaner call protocol

**Problem.** §4.3: fib is 2.7× C per call. rk4 makes 4 calls per step and is 1.89× C.

**Design.**
- **(a)** A SIR inliner pass that runs first, before `specialize`. It inlines callees that are non-recursive, below N ops, with ≤ 16 scalar-or-small parameters, and have no `io` effect. Then fold, GVN, fuse and the check deferral see across the call.
- **(b)** For calls that remain:
  - skip the stack check in functions whose frame is bounded and which make no calls;
  - return failure in the f64 result as a NaN sentinel for single-number-result functions. A NaN is never a legal Sky value, so this is sound. It drops the status register and replaces `icmp ne + brif` with `fcmp uno + brif` on a value already in a register.
  - keep depth and stack limit in a pinned register instead of passing them as arguments. Cranelift 0.121.2 supports `enable_pinned_reg` on aarch64 (VERIFIED: `cranelift-codegen-0.121.2/src/isa/aarch64/lower.rs:130-131`, `abi.rs:1090`; `settings.rs:530`). Whether it interacts cleanly with the thread thunks and AOT objects is EXPERIMENTAL.

**Expected gain.** rk4 1.89× → **~1.2–1.3× C**. fib 2.7× → ~1.8–2.0× (the f64 vs `long` difference remains).

**Difficulty / confidence.** Inliner 3 / 4; protocol 2 / 3.

**Files.** A new `crates/sky-opt/src/inline.rs`, `crates/sky-opt/src/pipeline.rs`, `crates/sky-jit/src/module.rs` (`call_op` `:2932`, `define_thunk` `:872`, signatures).

### F4. Hoist factorisations out of loops (split `solve[cholesky]` into factor + solve)

**Problem.** §4.7: cholesky-32 refactors a constant K 5000 times.

**Design.**
- Lower `K⁻¹ @ b` to `L = factor[cholesky] K; x = solve_factored[cholesky] L, b`. `sky-linalg` already exposes `cholesky` and `cholesky_solve` separately (`chol.rs:191`, `:234`).
- LICM then hoists `factor` when K is invariant. It needs the existing guard rule for fallible ops (`sky-opt/src/licm.rs`): the error must surface in the first iteration exactly as before. Hoist into a peeled first iteration, or guard on trip count > 0.
- Bits are unchanged, because the factor is deterministic.
- The same applies to LU (`solve[lu]`) and the inverse.

**Expected gain.** cholesky-32 goes from 6.5 µs to ~2.5–3 µs per iteration (the solves plus dispatch), **at or below C's 2.7 µs**. With F5 it falls to ~1.2 µs, beating LAPACK's refactor-every-time 1.47 µs. LIKELY.

**Honesty note.** This changes the algorithm relative to the C row. Report it as "Sky hoists, C as written does not".

**Difficulty / confidence.** 2 / 4.

**Files.**
- `crates/sky-ir/src/op.rs` (new `OpKind`s);
- `crates/sky-lower/src/*` or `crates/sky-opt/src/cholesky.rs`;
- `crates/sky-opt/src/licm.rs`;
- `crates/sky-interp/src/solve.rs`;
- `crates/sky-ir-interp/src/machine.rs`;
- `crates/sky-jit/src/runtime.rs` (via `generic`).

### F5. Mid-size arrays without boxing (escape analysis plus stack buffers)

**Problem.** Arrays of 17 to a few hundred entries pay boxed runtime dispatch (~2 µs per iteration in cholesky-32; LIKELY 0.1–0.3 µs per op).

**Design.**
- For values of static shape with ≤ ~512 entries (4 KB) that do not escape, allocate a Cranelift stack slot. "Escape" means returned, stored into an `unknown` slot, captured by a handle, or passed to a runtime-only op that keeps it.
- Emit **loops**, not unrolled code, for elementwise ops, dot, matvec and index. This avoids the compile-time blowup that set `MAX_SMALL = 16`.
- Call `sky-linalg` through a **pointer ABI** (`extern "C" fn sky_rt_chol_solve(ptr_k, n, ptr_b, out) -> status`) instead of `sky_rt_op` with boxed `Value`s.
- Materialise into a handle only on escape.

**Expected gain.** 5–20× on mid-size vector loops; cholesky-32 dispatch drops from ~2 µs to ~0.2 µs per iteration. LIKELY.

**Difficulty / confidence.** 4 / 3.

**Files.** `crates/sky-jit/src/module.rs`, `module/small.rs` (generalise `Opnd`), `runtime.rs` (pointer-ABI entries), `aot.rs`, `crates/sky-rt` (exports for `sky build`).

### F6. In-place buffer reuse in native code

**Problem.** §4.5: an extra write stream per op and +44% peak RSS vs SIR.

**Design.**
- Pass the lifetime bit (`sky_ir::Lifetimes::reuse`, `lifetime.rs:44-55`) to the runtime: `sky_rt_op_consume(site, operand_index)`.
- The runtime takes the operand's `Rc<Value>`. If `Rc::strong_count == 1` it writes the result into that buffer: `Kernel::run(inputs, &mut data)`, and `sky_interp::array::arith_owned` already accepts `Operand::Owned`.

**Expected gain.** ~15% per pass on 1e7-element chains (5.0 → ~4.3 ms on 1 thread). Peak RSS falls ~30% (526 → ~365 MB). VERIFIED basis: the SIR numbers in §3.2.

**Difficulty / confidence.** 2 / 4.

**Files.** `crates/sky-jit/src/module.rs` (`generic`, `push_operands`), `runtime.rs` (`sky_rt_op`, `fused`), `kernel.rs` (a run-into-buffer path, which exists as `Kernel::run`).

### F7. "Reproducible FMA" numerics plus a self-verified Accelerate path

**Problem.** §4.7: 2× lost to the FMA ban and ~7.5× to AMX.

**Design.**
- **Spec change** (a new edition): matrix-product entries are defined as `c = fma(a[i,k], b[k,j], c)`, k ascending, from +0.0. Likewise for the Cholesky, LU, triangular and Gram chains (`s = fma(-l, l, s)`).
- This is still **bit-reproducible on every machine**:
  - `fma` is correctly rounded by IEEE 754-2008;
  - it is native on aarch64 and on x86-64 with FMA3;
  - elsewhere, use a correctly rounded software `fma`, which is slow but gives the same bits.
- Sky's own `fused` kernel already computes exactly this order, deterministic across block sizes and threads (`micro.rs:10-17`). Measured 47–48 GFLOP/s per core.
- **Accelerate route**, on macOS only:
  - At first use, per process, or cached by `sky tune` keyed by OS build plus chip, run a self-test. It compares `cblas_dgemm` with the reference FMA kernel bit for bit on a fixed suite: square, odd, rectangular, long-k, thin, transposed.
  - The self-test runs **per shape class**. On this machine, tiny products (7×5×3) and matrix @ vector fail it (§4.9 b2), while all larger GEMM shapes pass.
  - Large reproducible GEMMs (beta = 0) in a passing class go to Accelerate. The rest stay on Sky's kernels.
  - **The self-test gates speed, never results.** On a mismatch Sky uses its own kernel, which gives the same bits.
  - Keep Sky's own Cholesky and LU structure. Do **not** use `dpotrf`, which varies with thread count (§4.9a).
  - The trailing updates `C -= L·Lᵀ` continue each entry's chain *from C*. That equals Accelerate only if Accelerate accumulates into C in k order with beta = 1. Not tested: EXPERIMENTAL. The fallback is to keep them on the own fused kernel.

**Expected gain.**
- 2× everywhere: 25 → 47 GFLOP/s per core; 137 → 240 GFLOP/s on 10 threads. VERIFIED by the `fused` column.
- On Apple, if the self-test passes: 14× single-thread and ~5× multi-thread on large GEMM (352–689 GFLOP/s). EXPERIMENTAL: one machine and one OS so far.
- This is the **only** path to "matmul ≤ 1.2× Accelerate" in the default mode.

**Costs.**
- Golden outputs involving products change once.
- Scalar `a*b + c` in user code stays unfused unless the spec also adopts contraction for scalars. Do not: it would make scalar code depend on what the compiler chooses to fuse, unless the rule is a syntactic one.

**Difficulty / confidence.**
- 3 / 4 for the 2× (the kernel exists; the work is spec, editions and the x86 fallback).
- 3 / 2–3 for the Accelerate route: it depends on undocumented behaviour and must be re-verified on M1, M3 and M4 (M4 uses SME, whose `FMOPA` is architecturally a per-element fused multiply-add, which is promising but untested).

**Files.**
- `spec/v0.1.md` (the reproducible arithmetic section `:639-663`, the relaxed section `:665+`);
- `crates/sky-linalg/src/{micro.rs,gemm.rs,chol.rs,lu.rs,tri.rs,inverse.rs,numerics.rs,relaxed.rs,accelerate.rs}`;
- `crates/sky-jit/src/module/small.rs` (inline small products must also fuse);
- golden tests.

### F8. Cut native compile time (fixed cost)

**Problem.** §4.8: ~12 ms of CLIF generation plus Cranelift for nbody; 5–16 ms of front end on literal-heavy files.

**Design.**
- **(a)** A content-addressed cache of compiled native modules on disk, keyed by a hash of the SIR plus the Sky version plus the target. Repeated `sky run`s skip codegen. Cranelift supports this: `cranelift-codegen` has an incremental cache (`src/incremental_cache.rs` in 0.121.2), and AOT objects already exist (`aot.rs`).
- **(b)** Lower the unroll and SLP budget for straight-line code that runs once.
- **(c)** Compile `<main>`'s hot loop with `opt_level=speed` and cold code with `none`.
- **(d)** Speed up the lexer and parser on numeric literals (16 ms for 290 KB).

**Expected gain.** −10 to −20 ms per run on programs like nbody and elementwise-1e7 (cache hit). This matters for scripting UX and for benchmark optics. LIKELY.

**Difficulty / confidence.** Cache 2 / 4; others 2 / 3.

### F9. Threading fixes

The pieces already exist: rayon in GEMM, kernels and elementwise ops, plus independent pure calls (`module.rs:120-180`).

**Fixes.**
- Raise or tune potrf's parallel threshold. n = 256 is slower on 10 threads than on 1 (12.3 vs 14.8 GFLOP/s).
- Make chunking E-core-aware: smaller chunks with work stealing, or P-core-only QoS for latency-sensitive kernels. Equal splits on 6P + 4E make E-cores the stragglers. LIKELY.

**Gain.** 10–30% at mid sizes. **Difficulty / confidence.** 2 / 3.

**Files.** `crates/sky-linalg/src/{tuning.rs,threads.rs,chol.rs}`, `crates/sky-ir/src/parallel.rs`.

### F10. One libm for all platforms, and vector math in kernels

**Problem.** §4.9c: platform libm breaks cross-OS reproducibility; §4.5: per-lane math calls.

**Design.**
- Route `sky_interp::math::apply_f64` to a single pure-Rust implementation: the `libm` crate (musl port; deterministic, not correctly rounded), or CORE-MATH (correctly rounded).
- Then write lane-wise F64X2 versions **with the identical operation sequence**, so kernels vectorise exp, log and tanh with the same bits.

**Gain.** It makes the reproducibility claim true across OSes. 5–10× on transcendental-heavy kernels: LIKELY, not measured.

**Difficulty / confidence.** 3 / 3.

**Files.** `crates/sky-interp/src/math.rs`, `crates/sky-jit/src/kernel.rs` (math lowering), `kernel_rt.rs`.

### F11. LLVM backend (inkwell) vs staying on Cranelift

**Recommendation: stay on Cranelift for Roadmap A.** Reasons:
1. The measured losses are Sky-level, not instruction selection: boxed runtime element access (57×), per-op checks (36% of instructions), no inlining, the FMA ban, fixed costs. Fused kernels already match C and Rust (1.00×). mandelbrot is within 1.20× of Rust.
2. LLVM's auto-vectorizer cannot vectorise loops with Sky's per-op early-exit checks. F2 is needed either way. LIKELY.
3. LLVM `-O2` compiles typically 10–50× slower than Cranelift. LIKELY: widely reported, e.g. wasmtime and rustc_codegen_cranelift comparisons; not measured here. Native compile is already Sky's biggest fixed cost (§4.8).
4. Build weight and version pinning of LLVM (hundreds of MB) vs pure-Rust Cranelift.
5. For the GPU pivot, the codegen that matters is Metal. A CPU LLVM backend does not help tokens per second.

Where LLVM would help: register allocation (nbody's 14% spills), scheduling and unrolling, perhaps 1.2–1.5× on scalar loops (EXPERIMENTAL). Revisit as an optional `sky build --release` AOT backend after F1–F3.

**Difficulty / confidence of an LLVM backend.** 4–5 / 2 for net benefit.

---

## 6. Relevance to the LLM-on-Apple-GPU direction

Decode is bound by GPU memory bandwidth; prefill by GPU matmul throughput. On the CPU side, an LLM runtime does the following work:
- tokenisation (byte-pair merges over hash maps and integer arrays);
- sampling (temperature, top-k/top-p, repetition penalties over a 32k–152k vocabulary);
- scheduling and KV-cache bookkeeping (integers, indices);
- constrained-decoding masks;
- small glue arithmetic.

| item | matters for LLM? | why |
|---|---|---|
| F1 inline element access (+ writes) | **Yes, critical** | At 148 ns per element, one top-p pass over a 152k vocab costs **~22 ms per token**, more than a whole GPU decode step. Either fix F1 or keep all sampling on the GPU. |
| Integer type (not in Sky: every number is f64) | **Yes** | Token ids, indices, bitmasks and hashing. Not a perf fix, a prerequisite. |
| F3 inlining and cheap calls | Yes (moderate) | Orchestration code is call-heavy. Per-token budgets are fine at 2 ns per call. |
| F8 compile and start-up | Minor | Model load (seconds) dominates; matters for CLI UX. |
| F6 in-place reuse / memory footprint | Yes (moderate) | 16 GB machines: every redundant buffer competes with weights and KV cache. |
| F10 vector exp | Minor | CPU softmax only if sampling stays on the CPU. |
| F2 check deferral | Scientific mostly | LLM CPU code is not FP-check-bound. |
| F4 factorisation hoisting, F5 mid-size arrays | Scientific | |
| F7 reproducible FMA / AMX GEMM | **Scientific only** | LLMs run fp16/bf16/int4 on the GPU. f64 GEMM is irrelevant. |
| F9 BLAS threading | Scientific | |
| LLVM backend | No | |

**Blunt conclusion.** Roadmap A as a *CPU f64 performance* programme makes Sky a better scientific language. It does not move Sky toward beating MLX or llama.cpp. The CPU-side items that do matter (F1 plus writes, integers, cheap calls, low per-token overhead) are cheap enough to fold into Roadmap A. Everything else should be scheduled by its scientific value.

---

## 7. Recommended Roadmap A (12 weeks), ordered, with gates

Gates are on **per-unit slopes** (two sizes), not whole-process medians. The first deliverable makes that the reporting standard.

| wk | item | gate (measured as in §3) | realistic? |
|---|---|---|---|
| 1 | **Benchmark hygiene.** Slope mode in `sky-bench compare` (two sizes per case); fix the canned explanations; add cases for big-array indexing, a top-k/top-p sampler over 128k, a call microbenchmark and mid-size (32–256) vectors; print the fixed-cost split (front end, lowering, codegen, run). | Report exists; RESULTS shows per-unit ratios. | Yes |
| 2–4 | **F1** buffer rep and inline index (+ InBounds) | indexing loop ≤ **1.5× C** (today 57×) | Yes (LIKELY) |
| 3–4 | **F6** native in-place reuse | elementwise 1e7, 1 thread: ≤ **1.15× C per pass** (today 1.36×); RSS ≤ SIR's | Yes |
| 4–5 | **F4** factorisation hoisting | cholesky-32 per iteration ≤ **1.0× C** (today 2.4×) | Yes |
| 5–7 | **F3** SIR inliner + call protocol | rk4 ≤ **1.3× C** per step (today 1.89×); fib ≤ **2.0× C** per call (today 2.7×) | rk4 yes; fib LIKELY |
| 6–9 | **F2** deferred checks | mandelbrot ≤ **1.05× Rust** per iteration (today 1.20×); nbody ≤ **2.0× C** per step (today 3.2×) | mandelbrot LIKELY; nbody EXPERIMENTAL (spills remain) |
| 8–9 | **F8** compile cache + literal parsing | nbody fixed cost ≤ 10 ms on a cache hit (today ~23) | Yes |
| 9–12 | **F7** reproducible-FMA edition + self-verified Accelerate | own kernel ≥ 45 GFLOP/s per core in the default mode; matmul-512 ≤ **1.3× Accelerate** *if* the self-test passes on M1–M4, else ≤ **3× Accelerate** (all threads) | 2× yes; the Accelerate gate EXPERIMENTAL |
| 10–12 | **F5** mid-size stack buffers (if time) | 32-vector loop ops ≤ 3× C | EXPERIMENTAL |

On the owner's example gates:
- **"Fused elementwise ≤ 1.1× C"**:
  - already true at kernel level for axpy and hypot (1.00–1.01×); chain6 is 1.16×;
  - per pass in programs, after F6, ~1.1–1.15× on 1 thread (the benchmark's extra `D` stream costs ~1 stream in 4);
  - with threads Sky is already faster than single-threaded C.

  **Realistic** if defined per pass. **Not realistic** on whole-process medians while inputs are built from outer products and 290 KB literals.
- **"Matmul ≤ 1.2× Accelerate"**:
  - **not realistic** under today's no-FMA semantics: the gap is 14–15× on one thread and 4.6–5.2× on all threads, and the own kernel is already at 89% of its ceiling;
  - with the FMA redefinition, the best NEON code is still ~2.6–2.9× behind (AMX);
  - only the self-verified Accelerate route (F7) can meet it. Today that rests on a single machine/OS measurement.

---

## 8. Negative results and open questions

**Negative results** (all VERIFIED):
- The native per-pass allocation does not cause extra page faults (identical page reclaims at R = 2 and R = 6); the allocator recycles. The real cost is the write stream plus RSS.
- `SKY_SLP=always` gives no gain over `auto` on nbody.
- The SIR interpreter is slower than the tree-walker on call-heavy code (fib: 1622 vs 718 ms).
- Threading hurts potrf at n = 256 (12.3 vs 14.8 GFLOP/s).
- The fused kernel is not where elementwise loses.
- "Small vectors allocate" is not true in native code.
- Accelerate `dpotrf` is not thread-count invariant.
- Several harness explanations are wrong (§2.2).

**Open questions** (EXPERIMENTAL):
1. Does Accelerate dgemm equal the FMA chain on M1, M3 and M4, on other macOS versions, for beta = 1 and alpha = −1, and for very long k (k ≥ 16384)? Where exactly are the shape-class thresholds below which it diverges? On this machine 7×5×3 and 2048×1×2048 diverge, while 64×64×8192 and every larger GEMM shape tried agree (§4.9 b2).
2. How much of nbody's 3.2× is spills vs checks? Answer by prototyping F2 behind a flag and re-running the census.
3. What do math functions inside kernels actually cost per element (exp/tanh)? Extend the `sky-jit` kernels bench with a `softmax`-like body.
4. Is a NaN-sentinel return ABI compatible with the multi-result and thread-thunk machinery (`define_thunk`, `module.rs:872`)?
5. The E-core straggler hypothesis for 10-thread kernels: measure 6 threads vs 10 at mid sizes.
6. Why is the SIR interpreter's call path slower than the tree-walker's? Profile `machine.rs:1211-1248`.

---

## 9. Appendix: commands and raw logs

All under `$S/research/r10/`:
- Build: `build.log`; `build_blas.log`; `build_kernels.log`.
- Baselines: `bin/` (C, `-ffast-math` C, Rust and Accelerate programs compiled from `benches/compare`).
- Rendered programs: `sky/*.sky`; size variants in `var/*.sky`.
- `compare_nosky.log`, `compare_nosky_quickcheck.md`: warmup-0 probe, numbers invalid (first-exec penalty); used only to render templates.
- `m1_scalar.sh` / `.log`: floors, fib, nbody with three SLP modes, mandelbrot, rk4.
- `m1b.sh` / `.log`: check, emit-clif, steps = 1 fixed costs, build obj.
- `m2_arrays.sh` / `.log`: elementwise, linear chain, matmul, Cholesky; 1 thread and 10 threads; relaxed.
- `m3.sh` / `.log`: fixed-cost split of elementwise-1e7.
- `m4.sh` / `.log`: `/usr/bin/time -l` page reclaims and RSS.
- `m5.sh` / `.log`, `micro/idx*.sky`, `micro/idx.c`: indexing loop.
- `micro/fuse_dag.sky`: fusion boundary IR.
- `blas_bench.log`: `BENCH_SIZES=32,64,256,512,1024 BENCH_THREADS=1,10 target/release/deps/blas-947639d8b35aa9eb --bench`.
- `kernels_bench.log`: `CARGO_TARGET_TMPDIR=$R/kbench RUSTC=~/.cargo/bin/rustc target/release/deps/kernels-6601d69be4736129 --bench`.
- `nbody.clif`, `nbody_main.clif`, `nbody.o`, `nbody.s`, `nbody_main.s`: native code for the census (`sky check --emit=clif`, `sky build --emit=obj`, `otool -tV`).
- `accdet/accdet.c`, `run.sh`, `accdet.log`: Accelerate determinism.
- `accdet/fmaeq.c`, `fmaeq.log`, `fmaeq2.c`, `fmaeq2.log`: Accelerate vs the FMA chain.
