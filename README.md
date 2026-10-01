# Sky

**A scientific programming language that knows physics.**

Sky reads like Python, but its compiler understands what your numbers *mean*: their physical units, the shapes of your arrays, the properties of your matrices, and how much your code costs. It uses that knowledge twice: to catch mistakes before the program runs, and to pick faster algorithms than you wrote. It compiles to native machine code through Cranelift.

```python
fuel = 120 kg
burn = 3 min
rate = fuel / burn
left = fuel - rate      # oops: forgot to multiply by a time
```

```
$ sky check rocket.sky
rocket.sky:4:13: error: cannot subtract M·T⁻¹ from mass
  |
4 | left = fuel - rate      # oops: forgot to multiply by a time
  |             ^ mismatched dimensions
  |        ---- mass
  |               ---- M·T⁻¹
  = note: `rate` is of dimension M·T⁻¹ because `rate = fuel / burn` (line 3), where `fuel`
          is a mass (`fuel = 120 kg`, line 1) and `burn` is a time (`burn = 3 min`, line 2)
  = help: these differ by a factor of time: `fuel` is `rate` times a time
```

Python would have printed a wrong number. C has no idea what a kilogram is. Sky tells you which line is wrong, why, and how to fix it.

---

## What makes Sky different

### Units are types, all the way down

Every value carries its dimension through arithmetic, functions, loops, branches, matrices, and derivatives. SI prefixes, derived units, and CODATA constants are built in. `as` shows a value in any unit you like.

```python
def fall_time(height: m) -> s:
    return sqrt(2 * height / g_n)

print(fall_time(1 km))                     # 14.280869812290344 s
print(42.195 km / (2 h + 35 min) as km/h)  # 16.333548387096776 km/h
E = 1 g * c^2                              # c, G, h, ħ, k_B, N_A, … are predefined
```

### Measurements carry their uncertainty

Write `±` and Sky propagates the uncertainty through every formula. It tracks *which* measurements each value came from, so correlated errors cancel instead of compounding. Most hand-written error propagation gets this wrong.

```python
L = 1.000 ± 0.005 m
g = 9.81 ± 0.05 m/s^2
print(2 * π * sqrt(L / g))   # 2.006 ± 0.007 s
print(L - L)                 # 0 ± 0 m   (the same measurement, perfectly correlated)
```

### Exact derivatives, with the right units

`derivative` and `grad` use automatic differentiation, so there is no finite-difference step to tune and no truncation error. The result has the correct type: the derivative of a position is a velocity, and minus the gradient of an energy is a force.

```python
def U(x: m, y: m) -> J:
    return 0.5 * 3 N/m * (x^2 + y^2)

F = -grad(U)(1 m, 2 m)       # [-3 N, -6 N]: a force vector, in newtons
```

### The compiler proves matrix properties, then exploits them

Declare a matrix `PositiveDefinite`, `Symmetric`, `Diagonal` or `Orthogonal`. Sky verifies the declaration, then chooses the algorithm. `K⁻¹ @ f` becomes a Cholesky solve, and the inverse is never formed. A matrix chain gets the cheapest association. `--explain` shows every decision:

```
$ sky check --explain chain.sky
chain.sky:5:5: note: [LA-CHAIN-001] `A @ B @ C @ v` is rewritten to `A @ (B @ (C @ v))`
  = note: A is Matrix[500,500], B is Matrix[500,500], C is Matrix[500,500], v is Vector[500]
  = note: estimated cost: 500,500,000 flops as written, 1,500,000 flops after the rewrite, about 334x fewer
```

Shapes are checked symbolically too. `Matrix[N, K] @ Matrix[K, M]` is a `Matrix[N, M]`, and a mismatched product is a compile error naming both dimensions.

### Differential equations that explain themselves

Write the equations as you would on a blackboard and say what you need: a tolerance, and what must be conserved. Sky picks the integrator, detects stiffness, and tells you exactly what it did and why.

```python
k1 = 0.04 s^-1                   # a classic stiff chemical kinetics problem
k2 = 3e7 m^3/mol/s
k3 = 1e4 m^3/mol/s

equation robertson(t: s):
    y1: mol/m^3
    y2: mol/m^3
    y3: mol/m^3
    y1' = -k1 * y1 + k3 * y2 * y3
    y2' = k1 * y1 - k3 * y2 * y3 - k2 * y2^2
    y3' = k2 * y2^2

sol = solve robertson from 0 s to 40 s with y1 = 1 mol/m^3, y2 = 0 mol/m^3, y3 = 0 mol/m^3:
    tolerance 1e-8
print(explain(sol))
```

```
… the step was limited by stability, not accuracy (the estimated h*lambda stayed near 3.42
for 15 steps, at Dormand-Prince's stability limit of about 3.3), which means the system is stiff
so from t = 0.0418 s the integration continued from that state with Rodas3, an L-stable
Rosenbrock method …
took 494 steps in all: 49 with Dormand-Prince (13 rejected) before the switch and 445 with Rodas3
```

Give it a second-order system with `conserve energy`, and it picks a symplectic integrator on its own. Over a year of Earth's orbit, the energy drifts by about 4e-12.

### Performance contracts, proven at compile time

A function can state what it costs. `sky check` proves the bound from the program, symbolically, for every input size. It never measures and never guesses.

```python
def gram(A: Matrix[N, M]) -> Matrix[M, M]:
    cost flops <= 2 * N * M^2       # proven
    return Aᵀ @ A

def mm(A: Matrix[N, K], B: Matrix[K, M]) -> Matrix[N, M]:
    cost flops <= 2 * N^3           # error: `mm` costs 2·K·M·N flops, which is more than `2 * N^3`
    return A @ B                    #   help: the cost is exactly 2·K·M·N: write `cost flops <= 2 * K * M * N`
```

### Results you can quote

- **Bit-for-bit reproducible by default.** Floating-point operations run in a fixed order, so the same program prints the same digits on every machine. Fast math (FMA, reassociation, Apple Accelerate) is an explicit opt-in: `--numerics=relaxed`.
- **Never a silent NaN.** Division by zero and overflow are errors that point at the line, not an `inf` that poisons the rest of the run.
- **Three engines that agree.** A tree-walking interpreter, an IR interpreter, and native code produce byte-identical output, including error messages. Tests and a differential fuzzer enforce it.
- **Editions.** A project pins its language edition in `sky.toml`, so old code never silently changes meaning.

---

## Fast

Sky compiles to native code with [Cranelift](https://cranelift.dev/). Its optimizer runs type specialization, constant folding, e-graph rewriting, GVN, LICM, loop fusion, and SIMD kernels, and it parallelizes big kernels and independent calls across cores automatically.

Median wall time on an Apple M2 Pro. Sky's times include starting up, type-checking, optimizing, and compiling the program on every run, about 5 ms on its own.

| Workload | C `-O3` | Rust `-O3` | Python | **Sky** |
|---|---:|---:|---:|---:|
| Matrix multiply 512×512 (×2) | 42.3 ms | 41.6 ms | 12.3 s | **16.1 ms** |
| Cholesky solve n=256 (×10) | 16.5 ms | 18.7 ms | 1.11 s | **16.4 ms** |
| `fib(30)`, recursive | 8.4 ms | 8.5 ms | 124 ms | **11.5 ms** |
| RK4 pendulum, 200k steps | 14.3 ms | 14.4 ms | 234 ms | **20.0 ms** |
| Elementwise math, 1M entries ×20 | 15.8 ms | 15.3 ms | 2.11 s | **24.4 ms** |
| Mandelbrot 300×300 | 24.8 ms | 26.6 ms | 912 ms | **39.2 ms** |
| N-body, 100k steps | 4.6 ms | 6.0 ms | 610 ms | **34.3 ms** |

Sky is 10–760× faster than CPython and within 1–2× of C on most workloads, without type annotations on hot loops or hand-tuning. Sky beats the hand-written C matrix code because it uses cache-blocked, autotuned kernels (`sky tune`); a vendor BLAS is still faster there. N-body is mostly fixed compile cost on a 30 ms run. The full table, method, and every ratio are in [`benches/compare/RESULTS.md`](benches/compare/RESULTS.md).

`sky build` compiles a program ahead of time into a standalone executable (about 6 MB, starting in about 4 ms) for x86-64 or arm64.

---

## How it works

```
 program.sky
     │   lex · parse · check: units, symbolic shapes, matrix properties, cost contracts
     ▼
 typed AST ─────────────────────────────► tree-walking interpreter  (the reference oracle)
     │   lower
     ▼
 SIR: an SSA IR whose types carry units, shapes and matrix tags
     │   specialize · fold · e-graph rewrites · GVN · LICM · fusion · algorithm choice · DCE
     ▼
 optimized SIR ──► SIR interpreter
               └─► Cranelift ──► JIT machine code          (sky run)
                             └─► object file + runtime ──► executable  (sky build)
```

The type checker is the heart of it. Units, shapes, and tags are inferred through functions and loops, and every rewrite the optimizer performs is justified by a fact the checker proved. `--explain` prints those justifications, with a rule ID for each.

It's written in Rust in 22 crates, with golden tests for every diagnostic and fuzzing of the lexer, parser, and engines against each other.

---

## Try it

You need [Rust](https://rustup.rs/); the toolchain version is pinned in `rust-toolchain.toml`.

```sh
git clone https://github.com/TomOstt/sky && cd sky
cargo build --release --workspace
export PATH="$PWD/target/release:$PATH"

sky run examples/showcase.sky        # a tour: units, derivatives, matrices, a simulation
sky run examples/orbit.sky           # a year of Earth's orbit, with an explained integrator
sky check --explain examples/showcase.sky   # see every decision the compiler made
sky repl                             # an interactive session
```

| Command | What it does |
|---|---|
| `sky check file.sky` | type-check: units, shapes, matrix properties, cost contracts |
| `sky check --explain file.sky` | also show every rewrite and algorithm choice, and why |
| `sky run file.sky` | check, compile to native code, run |
| `sky build file.sky -o prog` | compile to a standalone executable |
| `sky repl` | interactive session |
| `sky hw` / `sky tune` | inspect this machine / autotune kernels for it |

Everything else is in the guide: engines, flags, numerics modes, the IR, and benchmarks.

---

## Status

Sky is a young language, built in the open by one developer. Its core works well: the type system, units, uncertainty, automatic differentiation, linear algebra, ODE solving, the optimizer, and native code. In [dogfooding](examples/dogfood/), five realistic projects were written in it: heat diffusion, curve fitting with uncertainties, finite-element structures, orbital mechanics, and a newcomer's first hour. They are kept in the repo with an honest log of everything that got in the way.

These are in progress right now:
- **Arrays:** element writes, `ones`/`linspace`/concatenation, sizes from variables.
- **Reductions:** `sum`, `mean`, `std`, `min`/`max`.
- **Broadcasting:** a scalar combined with an array.
- **Output:** formatted printing and named derived units in output.
- **Tuples:** multiple return values.
- **Data and randomness:** reading CSV files, command-line arguments, reproducible random numbers.
- **More units:** imperial and lab units, °C/°F, user-defined units.
- **More linear algebra:** eigenvalues, banded solves, reusing a factorization.
- **ODE features:** events, solutions sampled at chosen times.

After that: GPUs and accelerators, distributed arrays, and more research directions. See [`ROADMAP.md`](ROADMAP.md).

## Learn more

- [`docs/guide.md`](docs/guide.md): every command, engine, and flag, and what works today
- [`spec/v0.1.md`](spec/v0.1.md): the language specification
- [`examples/`](examples/): runnable programs, each with its expected output
- [`docs/design/`](docs/design/): the IR and automatic differentiation designs
- [`ROADMAP.md`](ROADMAP.md): where Sky is going, and the research behind it

## Repository layout

```
crates/
  sky-syntax, sky-ast         lexer (Unicode-safe, indentation) and parser
  sky-units                   dimensions, the unit table, physical constants
  sky-types                   the type checker: units, shapes, matrix properties, cost contracts
  sky-interp                  tree-walking interpreter (the reference oracle)
  sky-ir, sky-lower           SIR, its verifier and text format; AST → SIR lowering
  sky-opt, sky-egraph         optimization passes and e-graph rewriting
  sky-ad, sky-derive          automatic differentiation
  sky-ode                     ODE integrators and method selection
  sky-linalg                  dense linear algebra kernels with a fixed order of arithmetic
  sky-ir-interp               SIR interpreter
  sky-jit, sky-rt             Cranelift native code (JIT and AOT) and the runtime library
  sky-hw, sky-tune            hardware discovery and autotuning
  sky-cli                     the `sky` command
  sky-bench                   benchmarks and the C/Rust/Python comparison
spec/  docs/  examples/  benches/  tests/  fuzz/
```

## Development

```sh
cargo test --workspace --locked
SKY_BLESS=1 cargo test -p sky-cli --test ui   # regenerate goldens, then review the diff
```

Licensed under [Apache-2.0](LICENSE).
