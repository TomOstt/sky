# Sky

Sky is a scientific programming language that checks physical units, tensor shapes, and matrix properties together at compile time, in syntax that reads like Python.

See `spec/v0.1.md` for the v0.1 scope.

## Status

`sky check` works on straight-line code: it lexes, parses, and type-checks a file and reports every error with source spans. `sky run` runs a file that checks cleanly, and `sky repl` is an interactive session; both print values in SI units.

```sky
v = 3 m / 2 s          # velocity
g = 9.81 m/s^2
y = 3 m + 2 s          # error: cannot add length and time
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

Not there yet:
- blocks: `def`, `if`, `for`, `import`, and indented code
- syntax for matrix tags (`PositiveDefinite`, ...) and for the unit of an array's elements, which the checker supports internally
- physical constants and unit prefixes (the unit table has 17 units)
- arrays at run time (the interpreter runs scalars, Bools and strings), and multi-line input in the REPL

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
