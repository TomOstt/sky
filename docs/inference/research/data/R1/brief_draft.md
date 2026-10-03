# R1 architect brief: Sky today (main a99e7e3)

## Bottom line

Today Sky is a scientific language that runs only on the CPU, only in f64, and only on vectors and matrices (rank ≤ 2). Its strengths are:
- the checker: physical dimensions, symbolic shapes, matrix tags, cost contracts;
- its engineering discipline: a verifier, three engines that must print byte-identical output, and `--explain` rule IDs.

None of its performance machinery transfers to Metal with f16 or int4 weights. That machinery is `sky-linalg`, `sky-jit` and the fused kernels. An LLM runtime would be a new compiler and runtime. It can borrow Sky's front end, its IR patterns, its tuning database and its test discipline. My estimate is that about 10–20% of the ~75k production lines are reusable (LIKELY).

## How a program runs today (VERIFIED)

```
.sky → sky-syntax → sky-ast → sky-types check (units, shapes, tags, @-chain plans)
     → sky-lower (+sky-ad) → SIR (47 ops, structured SSA) → sky-opt, 13 passes, verify after each
     → native Cranelift JIT ▸ else SIR interpreter ▸ else AST tree-walker   (run.rs:190-268)
All engines run array ops through ONE value layer: sky_interp::{Value,arith,array,solve}
     → sky-linalg (f64, no FMA) → [--numerics=relaxed] Accelerate.   CPU only.
```

## Key facts

1. **Data model** (VERIFIED)
   - f64 only. SIR has a `Repr` slot that can say F32, but no engine reads it (`sky-ir/src/ir.rs:42`).
   - No integer type: loop indices are f64.
   - Arrays are dense, immutable, never empty and rank ≤ 2. A rank-3 literal is rejected.
   - At most 2^24 entries per array (`sky-types/src/constant.rs:23`). One Qwen3-32B MLP weight has about 131M entries (LIKELY, from the public config), so it is about 7.8× too big.
   - Every number and array carries a 112-byte `Dimension`.
2. **Shapes**: a dimension is either a known number or a variable, matched by equality only. There is no size arithmetic such as `N+1` (`sky-types/src/shape.rs:1-6,98`). Sizes must be compile-time constants. Probe: `zeros(len(h))` is rejected.
3. **Gaps for tensor code** (VERIFIED by probes)
   - No broadcasting between arrays, by design; on main not even a scalar with a vector.
   - No reductions, no `ones`, `arange` or `concat`, no element writes, no reshape or views.
   - No structs, tuples, imports or closures.
   - A 64-wide MLP with RMSNorm runs identically on all three engines, but only with workarounds: `exp(zeros(64))` stands in for ones, and `y@y` stands in for a sum.
   - By my count, writing a Qwen3 layer in Sky needs at least 12 missing features.
4. **SIR**
   - 47 ops, a total verifier, a round-trip text format, fuzzing.
   - No memory model: no buffers, dtypes, devices or streams. The design's lower "strategy" and "kernel" levels exist only on paper.
   - Every `match` on the op enum is exhaustive, so adding ops is expensive: one in-flight branch added 5 ops and touched 23 files.
5. **Native code**
   - Cranelift compiles only scalars, Bools and arrays of at most 16 entries with a static shape inline.
   - Every other array operation is a call into the runtime: about 290 ns per op, against 3–4 ns inline (measured).
   - Vectors are at most 128-bit `F64X2`.
   - `sky build` executables are about 5.7 MB and embed the SIR and the SIR interpreter.
6. **Fusion** covers only elementwise trees in which each intermediate is used once. It does not fuse reductions or matmul epilogues. Math functions inside a fused kernel are scalar library calls, one lane at a time.
7. **`sky-linalg`**
   - Goto/BLIS-style f64 GEMM with a 4×8 register tile and no FMA by default; Cholesky, LU and inverse.
   - The algorithm choice by matrix tag happens in the checker.
   - Measured on a 2048² f64 GEMV: 0.32 ms per product, about 105 GB/s effective on 10 threads. The Accelerate path takes 0.46 ms.
   - Published: a 512³ product takes 16.1 ms in Sky and 4.95 ms in Accelerate.
8. **Numerics policy**
   - Reproducible by default: no FMA, fixed summation order, bit-identical results across engines, threads and machines.
   - `--numerics=relaxed` is opt-in and applies to CPU kernels only.
   - Every non-finite value is a runtime error.
   - Measured cost of that check in compiled kernels: about 0% when memory-bound, and +30% on a compute-bound six-op chain with a division.
9. **Tests**
   - 2,628 `#[test]` functions, 164 UI goldens, 22 IR golden programs, 5 fuzz targets.
   - Engine equality is enforced by the CLI run tests, the examples test, 3,000 differential-fuzz seeds per `cargo test`, and a test comparing `sky build` executables with the JIT.
   - There is no gate script. CI runs on Linux only.
10. **`sky-hw` / `sky-tune`**
    - `sky-hw` knows CPU facts only: no GPU, no memory bandwidth.
    - The tuning database is data-only text keyed by (machine fingerprint, kernel, size class). It is directly reusable for tuning Metal tile sizes.

## Negative findings

- There is no GPU, Metal or ANE code or dependency anywhere. The ROADMAP planned GPUs "through MLIR"; I believe upstream MLIR has no direct Metal target (LIKELY).
- `sky hw` says "parallel kernels use 6 threads", but nothing reads that value. The kernels use 10 threads, efficiency cores included (`sky-ir/src/parallel.rs:267`).
- Hooks with no consumer:
  - `specialize_for` and `SpecializationCache`;
  - `MatMulPolicy` (only `syrk` changes execution);
  - the e-graph's `Relaxed` contract.
- What the README calls "loop fusion" is fusion of elementwise expressions only.
- **The 10 in-flight branches have zero commits.** All of the work is uncommitted in `.claude/worktrees/*`, untouched since 17:20.
  - The production code of all ten compiles.
  - Three have test targets that no longer compile: tuples, readable output and ODE events.
  - Sizes-from-variables changes only the checker.
- **Policy conflicts with competitive GPU inference:**
  - reproducible-by-default forbids FMA, reordered sums and `simdgroup` matrix accumulation;
  - "never NaN or inf" conflicts with −inf attention masks;
  - "no broadcasting between arrays" conflicts with ordinary model code;
  - byte-identical engines cannot hold between a GPU and an f64 CPU oracle.

## In-flight work that matters for tensors and LLMs

- **Prerequisites for writing tensors in Sky:**
  - arrays-you-can-build: an SSA `store` that writes in place when the old array is dead, which suits KV-cache appends, plus `arange` and `concat`;
  - sizes-from-variables: run-time sizes and `a·N + b`, in the checker only;
  - reductions: vectors only, and the most complete of the ten.
- **Partial:**
  - broadcasting: a scalar with a vector only;
  - tuples;
  - data-in: a deterministic SplitMix64 random-number generator, useful for sampling, but CSV input only.
- **Irrelevant:** units, eigenvalues and tridiagonal solves, ODE events, readable output.

## Recommendations

1. **Build the LLM path as new crates beside SIR**, not by extending `OpKind` or `sky_interp::Value`:
   - `sky-tir`: a tensor IR with dtypes including quantized blocks, N-d symbolic shapes using `sky_types::poly::Poly`, views, axis reductions, broadcasting, attention and KV append;
   - `sky-metal`: Metal Shading Language kernels plus a runtime;
   - `sky-llm`: GGUF loading via mmap, a tokenizer, a sampler and the generation loop.
2. **Reuse:** `sky-diag`; `sky-tune` with a GPU-aware fingerprint added to `sky-hw`; the pass-manager, verifier and printer patterns of `sky-ir`; the `egg` scaffolding; the cost model, made aware of bytes per dtype; and the differential-testing discipline, with tolerance-based oracles.
3. **Milestone 1:** build the Qwen3 graph from GGUF metadata in Rust. Teach the Sky language to target that IR later.
4. **Decide the LLM numerics policy.** Suggested: deterministic for a given device and build, checked against an f32 CPU reference and by agreement on greedy tokens, with finiteness checks as a debug mode.
5. **Commit or park the worktrees now.** Repair the three stale test files. Merge arrays-you-can-build, sizes and reductions first if authoring in Sky matters.
6. **Verify the Metal toolchain on this machine, which has only Command Line Tools:** in particular, whether runtime compilation with `newLibraryWithSource:` works without Xcode. `deny.toml` allows crates.io packages under MIT, Apache or BSD licenses. On that basis, porting MLX or llama.cpp Metal kernels (MIT) is LIKELY permitted, with license notices.

## Open questions

1. Is the LLM written in Sky, or built by Sky's toolchain?
2. Does "reproducible by default" apply to the LLM product?
3. Should the tensor IR be a level of SIR or a sibling crate?
4. What happens to the uncommitted worktrees?
5. Can MSL be compiled at run time without Xcode? (EXPERIMENTAL)

## Notes

Full notes: `/private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad/research/R1-sky-today.md` (§1–§9, appendices). Raw data, scripts and logs: `…/scratchpad/research/r1/`.
