# Sky

Sky is a scientific programming language that checks physical units, tensor shapes, and matrix properties together at compile time, in syntax that reads like Python.

See `spec/v0.1.md` for the v0.1 scope.

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

```
sky check file.sky            # diagnostics on stderr; exit 1 on any error
sky check --explain file.sky  # also show the edition, the checker's rewrites (A⁻¹ @ b → Cholesky solve) and what the optimization passes did
sky check --emit-ir file.sky  # print the program's IR (SIR text) on stdout; --emit-ir=lower, =after:fold|egraph|dce|cholesky, =final
sky run file.sky              # check, then run; only a clean file runs. print() writes to stdout
sky repl                      # expressions echo their value; blocks (def, if, while, for) continue on `...` lines until an empty line
sky run --engine=ast file.sky # pick the engine: sir (lower to SIR, run it), ast (tree-walker), auto (the default)
sky repl                      # one line at a time; expressions echo their value
```

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

A line that ends in `:`, or leaves a `(` or `[` open, continues on a `...` line; a block ends at an empty line. A failing input changes nothing, however many lines it has. `:help` and `:quit` (or Ctrl-D) are the only commands.

`as` is only for showing: `v as km/h` is a display value that `print` and the REPL write, not a number to compute with, so `(v as km/h) + 1` is an error. To compute with kilometres per hour, divide by the unit: `v / (1 km/h)`. See "Showing values in a unit" in `spec/v0.1.md`.

`--emit-ir` prints the program after lowering to SIR (`=lower`), after one pass of the default pipeline (`=after:fold`, `egraph`, `dce`, `cholesky`), or at the end (the default). With it, stdout is only the IR (the `edition:` and `ok` lines go to stderr). A program that checks but cannot be lowered yet (arrays) is still `ok`, with a note saying why there is no IR. `--explain` merges each pass's explanations (rule id, before and after, the reasons) with the checker's notes in source order. For example, `x: m` then `y = x * 1` gives:

```
f.sky:2:5: note: [ALG-001] `x * 1` is rewritten to `x`
  = note: applied by the `egraph` pass
  = note: because multiplying a finite float by 1 gives it back bit for bit, and cannot overflow
```

`sky run` has two engines that print the same bytes, on stdout and stderr, with the same exit status. `--engine=auto` (the default) lowers the program to SIR, runs the optimization pipeline with verification, and executes it on the SIR reference interpreter; a program SIR cannot express yet (iterating over an array, a `range` bound computed at run time, an input declaration inside a function, ...) silently runs on the tree-walking interpreter instead. `--engine=sir` never falls back and reports such a program as an error (`cannot run on SIR: ...`), and `--engine=ast` always uses the tree-walker, which stays as the oracle the SIR engine is tested against. A runtime error found on SIR is re-run on the tree-walker, with its output discarded, so that it is pointed at and labelled exactly as before (see "Running programs" in `spec/v0.1.md`).

Numbers are always finite: division by zero and overflow are runtime errors, not `inf` or `NaN`. See "Running programs" in `spec/v0.1.md`. The REPL keeps no session file.

Exit status: `0` ok; `1` diagnostics with at least one error (including a refused project root or invalid `sky.toml`, or a runtime error in `sky run`); `2` usage error or input that can't be read.

A project can declare its language edition in `sky.toml`, so the meaning of old code never changes silently:

```
edition = "2026"     # the only key; the file is a tiny data-only subset of TOML
```

Without a `sky.toml` a file uses the latest edition. An edition newer than the installed Sky is an error. See "Editions" in `spec/v0.1.md`.

Not there yet:
- `for` loops and `import`
- transpose (`Aᵀ`), rank-3 tensor literals, negative indices, and array writes (arrays are immutable)
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
benches/        benchmark workloads: `benches/programs/*.sky`
fuzz/           cargo-fuzz targets for the lexer, parser and manifest
```

## Benchmarks

`sky-bench` times each program in `benches/programs/` through three engines: (a) the tree-walking interpreter (`sky run`), (b) lowering plus the SIR interpreter with no passes, and (c) lowering plus `sky_opt::default_pipeline` plus the SIR interpreter. It also times the compile-side phases (lex, parse, check, lower, pipeline, verify), and it fails if the three engines do not print identical output. Every workload prints a checksum, so none can be skipped.

```
cargo run --release -p sky-bench                              # all workloads, a table on stdout
cargo run --release -p sky-bench -- --filter fib              # workloads whose name contains the text
cargo run --release -p sky-bench -- --json out.json           # also write the results as JSON
cargo run --release -p sky-bench -- --reps 9 --warmup 2       # more repetitions (default 5, warmup 1)
cargo run --release -p sky-bench -- --quick                   # tiny sizes, a smoke run
```

Always use `--release`: debug timings mean nothing. Each engine runs `--reps` times, interleaved, and the table shows the median with the range as a percentage of it; the JSON keeps every sample (integer nanoseconds, schema version 1) so runs can be compared over time. On a busy machine the median drifts upward; the minimum is the steadier number. Set `SKY_BENCH_REV` to record a commit in the JSON.

The workloads are `scalar_loop` (10^6 iterations of arithmetic with units), `recursion` (`fib(25)`), `math_calls` (`for` over `range` with math functions), `array_chain` (elementwise vector chains), `matmul` (`@` on small matrices), `cholesky` (`K⁻¹ @ f` on a PositiveDefinite literal) and `derivative` (`derivative(f, t)(x)` in a loop). To add one, put a `.sky` file in `benches/programs/` whose first line is `# bench: N=<full> quick=<small>` and which sets its size with a line `N = <full>`, then list it in `crates/sky-bench/src/bench.rs` (a test fails if a file is not listed).

The `bench` workflow (`.github/workflows/bench.yml`) runs the suite in release mode weekly and on manual dispatch, never on pull requests, and uploads the JSON as an artifact.

## Development

The toolchain is pinned in `rust-toolchain.toml`.

```
cargo test --workspace --locked
SKY_BLESS=1 cargo test -p sky-cli --test ui   # regenerate goldens, then review the diff
```

Licensed under Apache-2.0.
