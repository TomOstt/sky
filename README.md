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
```

`as` is only for showing: `v as km/h` is a display value that `print` and the REPL write, not a number to compute with, so `(v as km/h) + 1` is an error. To compute with kilometres per hour, divide by the unit: `v / (1 km/h)`. See "Showing values in a unit" in `spec/v0.1.md`.

`--emit-ir` prints the program after lowering to SIR (`=lower`), after one pass of the default pipeline (`=after:fold`, `egraph`, `dce`, `cholesky`), or at the end (the default). With it, stdout is only the IR (the `edition:` and `ok` lines go to stderr). A program that checks but cannot be lowered yet (arrays) is still `ok`, with a note saying why there is no IR. `--explain` merges each pass's explanations (rule id, before and after, the reasons) with the checker's notes in source order. For example, `x: m` then `y = x * 1` gives:

```
f.sky:2:5: note: [ALG-001] `x * 1` is rewritten to `x`
  = note: applied by the `egraph` pass
  = note: because multiplying a finite float by 1 gives it back bit for bit, and cannot overflow
```

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
spec/           grammar, typing rules, diagnostic format
tests/ui/       golden diagnostic tests for `sky check`
tests/ir/       golden tests for `sky check --emit-ir` and the pass explanations
fuzz/           cargo-fuzz targets for the lexer, parser and manifest
```

## Development

The toolchain is pinned in `rust-toolchain.toml`.

```
cargo test --workspace --locked
SKY_BLESS=1 cargo test -p sky-cli --test ui   # regenerate goldens, then review the diff
```

Licensed under Apache-2.0.
