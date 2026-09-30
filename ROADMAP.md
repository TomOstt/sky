# Sky roadmap

Sky's long-term goal: **scientists write the mathematics, the constraints, how accurate the answer must be, and what to optimize. The compiler handles the machine.** Future compilers and future hardware should keep making the same Sky source faster without anyone rewriting it.

This file turns that vision into an order of work. It is a plan, not a promise. Each phase must ship something scientists can use on its own. The detailed design of the next foundation is in [`docs/design/ir.md`](docs/design/ir.md).

## Principles

These decide every scheduling question below.

1. **More knowledge, more optimization.** Units, shapes, matrix properties, precision, effects and physical constraints are kept as long as possible and never thrown away early. That knowledge is Sky's only real advantage over C++ with good libraries.
2. **One semantic IR, many passes.** Almost every idea in the vision is an analysis or rewrite over a single intermediate representation (IR): a graph of mathematical operations that carries all that knowledge. Build the IR once, then add capabilities as passes. Never add a capability as a special path from syntax to machine code.
3. **A small, stable core; rich libraries.** New science goes into libraries that teach the compiler facts, such as "FFT is linear, O(N log N), invertible". It does not go into syntax. The core language changes rarely and only through versioned editions.
4. **Correct first, then fast, and always explained.** Every transformation keeps the program's contract: its meaning, its units, and its declared error bounds. Every non-trivial decision can be shown with `--explain`. Optimizations are checked against a reference interpreter.
5. **Reuse existing work; don't reinvent it.** Most pieces already exist in libraries, compilers and papers. Sky's contribution is the semantics that connect them. See "Building on existing work" for how to reuse code safely under Sky's Apache-2.0 license.
6. **Intuitive to scientists.** When the syntax or meaning is ambiguous, report a clear error with both possible spellings rather than guess silently. Diagnostics speak physics ("expected acceleration, found velocity"), not compiler internals.

## Phases

Numbers in brackets refer to the vision's list of 100 ideas ("Idea catalog" below).

### Phase 0: v0.1, a checker plus an interpreter (in progress)
Straight-line code, blocks and functions; units with SI prefixes and physical constants; symbolic shapes; matrix-property tags declared in annotations; the Cholesky rewrite, shown by `--explain`; `sky check`, `sky run` and `sky repl`.
**Done when:** a scientist can write, check and run a small physics script with units, and every example in `spec/v0.1.md` has a golden test.

### Phase 1: the foundation
The semantic IR [2]: lowering from the checked AST, a verifier, a stable text format, and an interpreter that serves as the reference semantics.
- Language editions and versioned semantics [75, 76]: `edition = "2026"` in `sky.toml`, and a rule that the meaning of old code never changes silently.
- A framework for library-provided facts [70, 77], starting with properties of built-in operations.
- Explanations as a first-class output of every pass [88, 89], generalising today's `RewriteSite`.

**Done when:** every v0.1 program lowers to IR and verifies, the IR interpreter matches the tree-walking interpreter on every test (differential fuzzing included), and the Cholesky rewrite is an IR pass rather than a special case in the checker.

### Phase 2: the first optimizations
These come cheaply once the IR exists:
- Dimensional and shape derivation and debuggers [4, 93, 94];
- Exact rationals and symbolic constants (π, √2) kept until approximation is needed [9];
- Identity rewriting on an e-graph, each rewrite rule identified and justified [5], and hybrid symbolic-numeric evaluation [99];
- Dead-computation elimination, lazy evaluation and fusion [18, 27–30];
- Automatic differentiation as an IR transform [32].

**Done when:** `--explain` shows fused kernels and rewrites with reasons, and `grad(f)` works on checked code.

### Phase 3: numerics as semantics
- Error budgets: `error <= 1e-12` leads to precision chosen per region [7, 8, 31], using interval and affine analysis on the IR. Built so far: the roundoff analysis (`sky_ir::roundoff`) and the contract (`accuracy x within 1e-12`, checked by `sky check`); choosing the precision needs code generation (Phase 4). See `docs/design/ir.md`, "Error budgets and roundoff".
- Uncertainty-carrying values [10], starting with linear propagation. Done: `5.31 ± 0.02 kg` with correlations tracked, on the tree-walker (`spec/v0.1.md`, "Uncertainty"); arrays, SIR and non-linear propagation are next.
- `equation` / `solve` for ordinary differential equations (ODEs), with a transparent solver-selection table [11, 12]: stiffness, symplectic structure, tolerances.
- Condition-number warnings [42] and numerical-instability hints with rewrites [92].
- Contracts such as `ensures` and `conserve energy`, monitored at run time [3, 29, 81]. `assert`, `requires` and `ensures` are in (spec "Contracts"); `conserve` comes with `equation`/`solve`.

**Done when:** an orbit script declares its tolerance and conserved quantity, and Sky picks and explains the integrator.

### Phase 4: speed on one machine
- Native code through Cranelift, with LLVM later [61].
- Runtime specialisation on shapes and values [62, 63].
- Automatic parallelism from the dependency graph [22].
- Algorithm variants chosen by size and hardware [15], and discovery of hardware capabilities [20]. Built so far: `sky hw` and the `sky-hw` crate (SIMD, cores, caches, memory, a machine fingerprint); the variants themselves need the fast kernels.
- Autotuning with a local performance database [16, 17]. Built so far: `sky tune` and the `sky-tune` crate (a tuner, a data-only database keyed by the fingerprint, a registry for kernels to plug into); see the spec's "Hardware and tuning".
- Allocation and copy elimination guided by lifetimes [25, 26, 85].
- Performance contracts [86].

**Done when:** dense linear algebra and fused elementwise code in Sky are within 2× of hand-tuned libraries.

### Phase 5: accelerators and scale
- GPUs through MLIR [19, 21, 55].
- Native sparse and graph types [34, 35].
- Communication avoidance and moving compute to the data [23, 24].
- Deterministic parallel execution [52].
- Distributed arrays: one program runs from a laptop to a cluster [100].
- Reproducibility packages (`sky freeze`) [74].

### Research track (prototype after publishing or reading the literature)
- Choosing algorithms from library semantics, e.g. N-body direct → TreePM [13, 14].
- Proof-carrying optimization and translation validation [6, 60, 82].
- Detecting conservation laws and structure-preserving (Noether-aware) integrators [39, 40].
- Automatic nondimensionalization and preconditioners [41, 43].
- Geometry, manifolds and index-notation tensor calculus [36–38, 13 in the original vision].
- Surrogate models with error bounds [46, 47].
- Physics-regime checks and model switching [96–98].
- Worst-case execution time analysis [50, 51].
- Generic code constrained by algebraic laws (Field, LieGroup) [79, 80].

### Not language features
Keep the IR abstract enough to allow these later, but don't build them into the language:
- carbon-aware and cost-aware scheduling [48, 49, 87];
- reversible, quantum, interplanetary and relativistic computing [53, 54, 56–58];
- shared cross-user optimization databases [69], which raise privacy and trust questions.

## Building on existing work

Most of the pieces already exist somewhere: in a library, a compiler, or a paper. Sky's job is to connect them through one semantic IR. So for every capability, look for prior work first and write new code last. There are three ways to reuse, in order of preference:

1. **Depend on it** as a crate or library. This is the cheapest option and gets upstream fixes for free. `cargo deny` checks every dependency's license.
2. **Port or copy it** when a dependency doesn't fit, for example when the code is in Julia, Racket, C++ or Python, or is a small piece of a large system. This is allowed only from permissively licensed code: MIT, BSD, Apache-2.0 (including the LLVM exception), ISC or Zlib, the same list as `deny.toml`. Keep the original copyright notice in the file header, name the source and version (or commit), and add an entry to `THIRD_PARTY_NOTICES.md`.
3. **Learn from it** by reading papers and designs and writing our own implementation. This is the only option for GPL, LGPL, AGPL, proprietary or unlicensed code: never copy, paste or translate that code line by line.

When unsure, treat code as case 3 and ask. A license mistake is far more expensive to undo than to avoid.

| Need (phase) | Prior work | License | How to reuse |
|---|---|---|---|
| Rewrites, identities, extraction (2) | `egg`, `egglog` | MIT | depend |
| CPU code generation, JIT (4) | Cranelift | Apache-2.0 WITH LLVM-exception | depend |
| GPUs, sparse tensors, accelerators (5) | MLIR (`linalg`, `sparse_tensor`, `gpu`), `melior` bindings | Apache-2.0 WITH LLVM-exception | depend |
| Dense linear algebra (2–4) | `faer`, `nalgebra` | MIT, Apache-2.0 | depend |
| ODE and DAE solving (3) | SUNDIALS (CVODE, ARKODE, IDA) | BSD-3-Clause | depend through bindings |
| Solver selection heuristics (3) | DifferentialEquations.jl, e.g. its default algorithm choice | MIT | port the heuristics |
| Unit systems and dimensional analysis (0–2) | Unitful.jl, Pint, `uom` | MIT, BSD-3-Clause, Apache/MIT | port tables and ideas |
| Symbolic simplification (2) | SymPy, Symbolics.jl | BSD-3-Clause, MIT | port rule sets into e-graph rewrites |
| Floating-point accuracy rewrites (3) | Herbie, FPTaylor | MIT | port rules and techniques |
| Automatic differentiation (2, 4) | Enzyme, JAX | Apache-2.0 (+LLVM-exception) | depend on Enzyme later; port JAX's transform design |
| Tensor algebra compilation (4–5) | TACO, Halide | MIT | port scheduling and format ideas |
| Uncertainty propagation (3) | Measurements.jl, `uncertainties` | MIT, BSD-3-Clause | port |
| Autotuning (4) | Ansor/TVM, OpenTuner | Apache-2.0, MIT | port search strategies |

Check each project's license again at the time of reuse; this table can go stale.

## Idea catalog

The full vision lists 100 ideas, numbered as in the phases above: semantic compilation (1), a mathematics-aware IR (2), physical-law contracts (3), dimensional analysis (4), theorem-aware and proof-carrying optimization (5–6), error budgets, mixed precision and exact arithmetic (7–9), uncertainty (10), differential equations and solver synthesis (11–12), algorithm substitution and complexity awareness (13–15), profile-guided and learned optimization (16–18), hardware abstraction and discovery (19–21), asynchrony, communication avoidance and locality (22–24), zero-copy and lifetimes (25–26), fusion, whole-program and lazy optimization (27–30), demand-driven precision (31), differentiable programming and adjoints (32–33), sparse, graph and geometric types (34–38), conservation-law discovery (39–40), nondimensionalization, conditioning and preconditioners (41–43), multiscale and adaptive computing (44–45), surrogate models (46–47), energy and carbon objectives (48–49), real-time and worst-case timing (50–51), determinism (52), future hardware (53–58), hardware digital twins and superoptimization (59–60), machine-specific and runtime specialisation (61–64), compression-aware and approximate computing (65–67), semantic caching and shared optimization data (68–69), libraries as semantic knowledge (70), executable papers and mathematics (71–73), reproducibility and versioned semantics (74–75), a small, extensible core (76–78), algebraic generics (79–80), contracts and verification (81–82), race freedom and memory predictability (83–85), performance and cost contracts (86–87), explainability and counterexamples (88–90), scientific debuggers (91–94), generated scientific tests (95), physics sanity and regime checks (96–98), hybrid symbolic-numeric solving (99), and one language from notebook to supercomputer (100).
