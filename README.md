# Sky

Sky is a scientific programming language that checks physical units, tensor shapes, and matrix properties together at compile time, in syntax that reads like Python.

See `spec/v0.1.md` for the v0.1 scope.

## Status

`sky check` lexes, parses and type-checks a file, and reports every error with source spans. It checks units (with SI prefixes and physical constants), symbolic tensor shapes, and matrix properties, through functions, `if` and `while`. `sky run` runs a file that checks cleanly, and `sky repl` is an interactive session; both print values in SI units. Vectors and matrices are written as literals (`[1 m, 2 m]`, `[[1, 2], [3, 4]]`), indexed and sliced, multiplied with `@`, inverted with `⁻¹`, and printed as Sky.

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
```

```
sky check file.sky            # diagnostics on stderr; exit 1 on any error
sky check --explain file.sky  # also show rewrite decisions (A⁻¹ @ b → Cholesky solve)
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
```

Numbers are always finite: division by zero and overflow are runtime errors, not `inf` or `NaN`. See "Running programs" in `spec/v0.1.md`. The REPL keeps no session file.

Exit status: `0` ok; `1` diagnostics with at least one error (including a refused project root, or a runtime error in `sky run`); `2` usage error or input that can't be read.

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
  sky-syntax/   lexer: tokens, indentation, Unicode safety checks
  sky-ast/      typed AST with source spans, and the parser
  sky-units/    dimensions and the unit table
  sky-types/    type checker (units, shapes, matrix properties)
  sky-project/  project root discovery and import resolution
  sky-interp/   tree-walking interpreter
  sky-cli/      the `sky` command: check, run, repl
spec/           grammar, typing rules, diagnostic format
tests/ui/       golden diagnostic tests for `sky check`
fuzz/           cargo-fuzz targets for the lexer and parser
```

## Development

The toolchain is pinned in `rust-toolchain.toml`.

```
cargo test --workspace --locked
SKY_BLESS=1 cargo test -p sky-cli --test ui   # regenerate goldens, then review the diff
```

Licensed under Apache-2.0.
