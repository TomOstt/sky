# Sky

Sky is a scientific programming language that checks physical units, tensor shapes, and matrix properties together at compile time, in syntax that reads like Python.

See `spec/v0.1.md` for the v0.1 scope.

## A quick tour

[`examples/showcase.sky`](examples/showcase.sky) is a short program that shows what Sky does today. Run it with `sky run examples/showcase.sky`.

- **Units are types.** `42.195 km / (2 h + 35 min)` is a speed, and `print(pace as km/h)` shows it in the unit you ask for. Adding a length to a time is an error at check time, not a wrong number at run time.
- **Physical constants are built in.** `c`, `G`, `g_n`, `ħ` and others are predefined, with CODATA values.
- **Derivatives are exact and typed.** `derivative(position, t)(2 s)` (or `(∂position/∂t)(2 s)`) is computed by automatic differentiation, so there is no finite-difference error. Its type is a velocity. `-grad(U)(1 m, 2 m)` of an energy is a force vector in N.
- **Matrix properties are proven, then used.** Sky checks that a `PositiveDefinite` matrix literal really is positive definite. `K⁻¹ @ f` then runs as a Cholesky solve, so the inverse is never formed; other tags pick other algorithms (division for `Diagonal`, a transpose for `Orthogonal`, LU for any square matrix), and an unparenthesized `A @ B @ C @ v` is associated by the compiler to cost the fewest flops. `--explain` shows each choice with its estimated flops.
- **Loops and functions stay checked.** A `while` loop over a falling projectile keeps its time, height and speed in their units through every step.

[`examples/orbit.sky`](examples/orbit.sky) shows the numerics built so far. It is a year of Earth's orbit, with the force written as `-grad(U)(r)` of a potential. The script states its tolerance and that energy is conserved. Sky picks a symplectic integrator and explains the choice, and energy drifts by about 4e-12. The same file checks a contract (`requires`, `assert`) and carries a measurement's uncertainty (`1.000 ± 0.005 m`) through a formula.

The expected output of each example is next to it (`showcase.out`, `orbit.out`). A test runs every example on every engine and compares the output byte for byte.

## Status

`sky check` lexes, parses and type-checks a file, and reports every error with source spans. It checks units (with SI prefixes and physical constants), symbolic tensor shapes, and matrix properties, through functions, `if` and `while`. `sky run` runs a file that checks cleanly, and `sky repl` is an interactive session; both print values in SI units, or in a unit you choose with `as` (`print(v as km/h)`). Vectors and matrices are written as literals (`[1 m, 2 m]`, `[[1, 2], [3, 4]]`), indexed and sliced, multiplied with `@`, inverted with `⁻¹`, and printed as Sky.

```sky
def kinetic(m: kg, v: m/s) -> J:
    return 0.5 * m * v^2

print(kinetic(2 kg, 3 m/s))      # 9 J
E = 1 g * c^2                    # c, h, ħ, k_B, G, ... are predefined
y = 3 m + 2 s                    # error: cannot add length and time

K: PositiveDefinite Matrix[2, 2] N/m = [[2000 N/m, -1000 N/m], [-1000 N/m, 1000 N/m]]
f: Vector[2] N = [0 N, 100 N]
u = K⁻¹ @ f                      # --explain: rewritten to a Cholesky solve
print(u)                         # about [0.1 m, 0.2 m]
v = 90 km/h
print(v)                         # 25 m/s: values print in SI
print(v as km/h, u as mm)        # 90 km/h [100 mm, 200 mm]: `as` shows a value in a unit
print(v as kg)                   # error: cannot show a velocity in kg
```

Differential equations read like the blackboard: `x'` is the derivative, dimensions are checked (`v'` must be an acceleration), and the solver picks its method from what the script declares, and explains it.

```sky
GM = 1.327e20 m^3/s^2
equation orbit(t: s):
    r: Vector[2] m
    v: Vector[2] m/s
    r' = v
    v' = -GM * r / norm(r)^3

sol = solve orbit from 0 s to 1 yr with r = [1 AU, 0 m], v = [0 m/s, 29.8 km/s]:
    tolerance 1e-9
    conserve energy = 0.5 * norm(v)^2 - GM / norm(r) within 1e-6
print(sol.r, sol.v)              # the final state, in SI: about where the year began
print(explain(sol))              # which method ran, and why
```

`sky check --explain` says what the checker found in each `solve` (a second-order system, so a symplectic integrator applies; what is to be conserved) without running it. Equations run on the tree-walking interpreter (SIR cannot lower them yet, so `--engine=auto` falls back). See "Differential equations" in `spec/v0.1.md`.

A function states what it costs in its header, next to `requires` and `ensures`, and `sky check` proves it (or refutes it, as an error) from the program, without running it. The cost is a polynomial in the sizes of the arrays, in floating-point operations, bytes moved and allocations; `--explain` reports it for every function, even without contracts, with the hot spots.

```sky
def gram(A: Matrix[N, M]) -> Matrix[M, M]:
    cost flops <= 2 * N * M^2       # proven: `gram` costs 2·M²·N flops
    return Aᵀ @ A

def mm(A: Matrix[N, K], B: Matrix[K, M]) -> Matrix[N, M]:
    cost flops <= 2 * N^3           # error: `A @ B` costs 2·K·M·N flops
    return A @ B
```

A loop the model cannot bound is a warning, not silence (`unbounded (loop at line 7)`). Time is not a contract: it is noise, and a contract that is sometimes false is not one. See "Performance contracts" in `spec/v0.1.md`.

```
sky check file.sky            # diagnostics on stderr; exit 1 on any error
sky check --explain file.sky  # also show the edition, the checker's rewrites (A⁻¹ @ b → Cholesky solve), what it found in each `solve`, and what the optimization passes did
sky check --emit-ir file.sky  # print the program's IR (SIR text) on stdout; --emit-ir=lower, =after:fold|egraph|dce|cholesky, =final
sky run file.sky              # check, then run; only a clean file runs. print() writes to stdout
sky repl                      # expressions echo their value; blocks (def, if, while, for) continue on `...` lines until an empty line
sky run --engine=sir file.sky # pick the engine: ast (tree-walker, the default), sir (lower to SIR, run it), native (compile the SIR to machine code), auto
sky run --engine=sir --threads=4 file.sky  # SIR runs big elementwise kernels and independent pure calls on 4 threads (SKY_THREADS; 1 is sequential); the output never depends on it
sky run --numerics=relaxed file.sky  # opt in to fast, not bit-reproducible arrays: fused multiply-add, reassociated sums, Accelerate on macOS (or `numerics = "relaxed"` in sky.toml); says so on stderr
sky check --emit=clif file.sky # print the Cranelift IR the native engine compiles
sky build file.sky -o prog    # compile to a standalone native executable that behaves like `sky run --engine=native`; --emit=obj writes the object file only
sky repl                      # one line at a time; expressions echo their value
sky hw                        # what this machine can do (SIMD, cores, caches, memory) and what Sky uses each for
sky tune                      # measure kernel parameters (block sizes, unroll factors) here and remember the winners
sky tune --show               # list the remembered results; --reset forgets them
```

Phase 4 has started with the machine. `sky hw` discovers the CPU, its SIMD features, core counts (performance and efficiency), cache sizes and memory, with no `unsafe` code and no dependencies, and a stable fingerprint of the machine. `sky tune` times each tunable kernel's candidate parameters on this machine and stores the winners in a small text file in your cache directory (`SKY_TUNE_DB` overrides it), keyed by that fingerprint. Tuning changes speed, never results, and the file is data only: it is written only when you run `sky tune`, and it is not a saved session. The kernels use it: the matrix product's cache blocks, the Cholesky block size and the thread threshold (`sky-linalg`), the native-kernel threshold and the parallel thresholds (`sky-ir-interp`) are tuned, each read once per process and clamped, with the old constants as the fallback (`sky tune linalg_gemm` tunes one family; `sky check --explain` cites what was tuned where it picks a kernel).

```
$ sky repl
sky> x = 10 km
sky> x / 2 s
5000 m/s
sky> 5 kg * 2 m/s^2
10 N
sky> 1 m / 0 s
error: division by zero   (rendered with the source span, like any diagnostic)
sky> v = 90 km/h
sky> v as km/h
90 km/h
sky> def area(w: m, h: m) -> m^2:
...      return w * h
...
sky> area(3 m, 4 m)
12 m^2
sky> derivative(area, w)(3 m, 4 m)
4 m
```

The derivative of a function of a vector is its gradient, with the unit of the result over the unit of the vector: the gradient of a potential of a position is a force, and `-grad(U)(p)` points down the potential.

```
M = 5.972e24 kg
mass = 1000 kg

def U(p: Vector[3] m) -> J:
    return -G * M * mass / norm(p)          # G is the constant of gravitation

p = [7000 km, 0 km, 0 km]
print(-grad(U)(p))                          # [-8134.473387755101 N, 0 N, 0 N]: toward the origin
```

A line that ends in `:`, or leaves a `(` or `[` open, continues on a `...` line; a block ends at an empty line. A failing input changes nothing, however many lines it has. `:help` and `:quit` (or Ctrl-D) are the only commands.

`as` is only for showing: `v as km/h` is a display value that `print` and the REPL write, not a number to compute with, so `(v as km/h) + 1` is an error. To compute with kilometres per hour, divide by the unit: `v / (1 km/h)`. See "Showing values in a unit" in `spec/v0.1.md`.

`--emit-ir` prints the program after lowering to SIR (`=lower`), after one pass of the default pipeline (`=after:fold`, `egraph`, `dce`, `cholesky`), or at the end (the default). With it, stdout is only the IR (the `edition:` and `ok` lines go to stderr). A program that checks but cannot be lowered yet (arrays) is still `ok`, with a note saying why there is no IR. `--explain` merges each pass's explanations (rule id, before and after, the reasons) with the checker's notes in source order. For example, `x: m` then `y = x * 1` gives:

```
f.sky:2:5: note: [ALG-001] `x * 1` is rewritten to `x`
  = note: applied by the `egraph` pass
  = note: because multiplying a finite float by 1 gives it back bit for bit, and cannot overflow
```

`sky run` has three engines that print the same bytes, on stdout and stderr, with the same exit status. The default, `--engine=ast`, is the tree-walking interpreter, which also stays as the oracle the other engines are tested against. `--engine=sir` lowers the program to SIR, runs the optimization pipeline with verification, and executes it on the SIR reference interpreter. `--engine=native` compiles the same optimized SIR to machine code through Cranelift (`crates/sky-jit`) and runs that; it is over 100 times faster than the interpreters on a loop of scalar arithmetic (see Benchmarks). `--engine=auto` picks native code, then SIR, then the tree-walker: a program native code cannot take runs on SIR, and a program SIR cannot express yet (iterating over an array, an input declaration inside a function, ...) silently runs on the tree-walking interpreter instead. `--engine=sir` and `--engine=native` never fall back and report such a program as an error (`cannot run on SIR: ...`, `cannot run natively: ...`). A runtime error found on SIR is re-run on the tree-walker, with its output discarded, so that it is pointed at and labelled exactly as before (see "Running programs" in `spec/v0.1.md`).

`sky build file.sky -o prog` compiles a program ahead of time: the same SIR-to-Cranelift translation the native engine uses writes an object file (Mach-O on macOS, ELF on Linux), and the system C compiler links it with the runtime library `libsky_rt.a` (the `sky-rt` crate: printing, arrays, errors, math, the kernels, and the SIR interpreter that finishes a run when native code deoptimizes) into an executable. The executable prints the same stdout and the same diagnostics, and exits with the same status, as `sky run --engine=native` on the file; it carries the source text and the SIR, needs nothing from the machine that built it but the C library, and is about 6 MB, starting in about 4 ms. `cargo build --workspace` builds `libsky_rt.a` next to `sky`, where `sky build` looks for it (`--runtime=<path>` or `SKY_RT_LIB` name another). A program native code cannot take (an `equation`, for example) is refused (`cannot build: ...`). `--target=<triple>` builds an object for any x86-64 or arm64 target from any machine (`--emit=obj`), and an executable for either architecture on macOS. See "Building executables" in `spec/v0.1.md`.

Numbers are always finite: division by zero and overflow are runtime errors, not `inf` or `NaN`. See "Running programs" in `spec/v0.1.md`. The REPL keeps no session file.

Exit status: `0` ok; `1` diagnostics with at least one error (including a refused project root or invalid `sky.toml`, or a runtime error in `sky run`); `2` usage error or input that can't be read.

A project can declare its language edition in `sky.toml`, so the meaning of old code never changes silently:

```
edition = "2026"     # the only key; the file is a tiny data-only subset of TOML
```

Without a `sky.toml` a file uses the latest edition. An edition newer than the installed Sky is an error. See "Editions" in `spec/v0.1.md`.

Not there yet:
- `for` loops and `import`
- rank-3 tensor literals, negative indices, and array writes (arrays are immutable)
- inputs: a declaration without a value (`K: PositiveDefinite Matrix[N, N]`) is checked but has nothing to read it from at run time
- multi-line input in the REPL
- everything in [`ROADMAP.md`](ROADMAP.md) beyond v0.1

## Layout

```
crates/
  sky-diag/     diagnostics and the terminal-safe renderer
  sky-syntax/   lexer: tokens, indentation, Unicode safety checks; the Edition type
  sky-ast/      typed AST with source spans, and the parser
  sky-units/    dimensions and the unit table
  sky-types/    type checker (units, shapes, matrix properties)
  sky-linalg/   dense linear algebra kernels (GEMM, Cholesky, LU, inverse) with a fixed order of arithmetic
  sky-project/  project root discovery, sky.toml manifest, import resolution
  sky-interp/   tree-walking interpreter
  sky-ir/       SIR: data model, verifier, effects, text format, pass manager
  sky-lower/    checked AST to SIR
  sky-opt/      passes (fold, dce, cholesky) and the default pipeline
  sky-egraph/   e-graph rewriting as a pass
  sky-ir-interp/  reference interpreter for SIR
  sky-cli/      the `sky` command: check, run, repl
  sky-bench/    benchmark runner (not shipped; nothing depends on it)
spec/           grammar, typing rules, diagnostic format
tests/ui/       golden diagnostic tests for `sky check`
tests/ir/       golden tests for `sky check --emit-ir` and the pass explanations
examples/       runnable programs, each with its expected output (`.out`)
benches/        benchmark workloads: `benches/programs/*.sky`
fuzz/           cargo-fuzz targets for the lexer, parser and manifest
```

## Benchmarks

`sky-bench` times each program in `benches/programs/` through four engines: (a) the tree-walking interpreter (`sky run`), (b) lowering plus the SIR interpreter with no passes, (c) lowering plus `sky_opt::default_pipeline` plus the SIR interpreter, and (d) the same optimized SIR compiled to machine code (`sky-jit`). It also times the compile-side phases (lex, parse, check, lower, pipeline, verify, and the native compile), and it fails if the four engines do not print identical output. Every workload prints a checksum, so none can be skipped.

```
cargo run --release -p sky-bench                              # all workloads, a table on stdout
cargo run --release -p sky-bench -- --filter fib              # workloads whose name contains the text
cargo run --release -p sky-bench -- --json out.json           # also write the results as JSON
cargo run --release -p sky-bench -- --reps 9 --warmup 2       # more repetitions (default 5, warmup 1)
cargo run --release -p sky-bench -- --quick                   # tiny sizes, a smoke run
```

Always use `--release`: debug timings mean nothing. Each engine runs `--reps` times, interleaved, and the table shows the median with the range as a percentage of it; the JSON keeps every sample (integer nanoseconds, schema version 1) so runs can be compared over time. On a busy machine the median drifts upward; the minimum is the steadier number. Set `SKY_BENCH_REV` to record a commit in the JSON.

`sky-bench par` measures automatic parallelism instead (`cargo run --release -p sky-bench -- par [--threads 1,2,4,8] [--reps N] [--quick]`): the speedup of a fused kernel, an elementwise `add` and two or four independent calls over threads and sizes, and the break-even sizes the default thresholds (`sky_ir::parallel::Config`) are set from.

The workloads are `scalar_loop` (10^6 iterations of arithmetic with units), `recursion` (`fib(25)`), `math_calls` (`for` over `range` with math functions), `array_chain` (elementwise vector chains), `matmul` (`@` on small matrices), `chain_order` (`A @ B @ C @ v` with 500×500 matrices, which the compiler associates as `A @ (B @ (C @ v))`) against `chain_order_forced` (the same with `((A @ B) @ C) @ v`, forced by parentheses), `cholesky` (`K⁻¹ @ f` on a PositiveDefinite literal), `derivative` (`derivative(f, t)(x)` in a loop) and `gradient` (`grad(U)(p)` of a spring-network potential of a position vector, by reverse mode, in a loop), `licm_inverse` (`inv(K)` of a 30 × 30 matrix inside a time-stepping loop, which `licm` computes once) and `cse_kernel` (the same subexpressions in the arms of an `if` and after it, which `gvn` and `licm` merge and hoist). To add one, put a `.sky` file in `benches/programs/` whose first line is `# bench: N=<full> quick=<small>` and which sets its size with a line `N = <full>`, then list it in `crates/sky-bench/src/bench.rs` (a test fails if a file is not listed).

The chain workloads show what choosing the association buys (spec, "Algorithm choice"): on one release run, `chain_order` took about 3 ms and `chain_order_forced` about 540 ms per engine (roughly 165×, the flop model predicts 334×: 500,500,000 flops as written, 1,500,000 after), with the outputs identical across the three engines. The machine was busy, so read it as an order of magnitude and re-run it on a quiet one. The `+` in their setup lines (`identity(500) + identity(500)`) drops the tags `identity` carries, so the matrices are dense and untagged, and the lines end in `# N` so `--quick` can swap the literal size.

`nbody` (an N-body-like time-stepping loop on vectors of 32 bodies, `v = v + dt * a`) is the loop the buffer reuse of `sky_ir::lifetime` is for; `crates/sky-bench/tests/allocations.rs` counts the allocations per iteration of every workload on the three engines with a counting allocator.

The `bench` workflow (`.github/workflows/bench.yml`) runs the suite in release mode weekly and on manual dispatch, never on pull requests, and uploads the JSON as an artifact.

## Development

The toolchain is pinned in `rust-toolchain.toml`.

```
cargo test --workspace --locked
SKY_BLESS=1 cargo test -p sky-cli --test ui   # regenerate goldens, then review the diff
```

Licensed under Apache-2.0.
