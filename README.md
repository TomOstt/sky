# Sky

Sky is a scientific programming language that checks physical units, tensor shapes, and matrix properties together at compile time, in syntax that reads like Python.

See `spec/v0.1.md` for the v0.1 scope.

## Status

`sky check` works on straight-line code: it lexes, parses, and type-checks a file and reports every error with source spans.

```sky
v = 3 m / 2 s          # velocity
g = 9.81 m/s^2
y = 3 m + 2 s          # error: cannot add length and time
```

```
sky check file.sky            # diagnostics on stderr; exit 1 on any error
sky check --explain file.sky  # also show rewrite decisions (A⁻¹ @ b → Cholesky solve)
```

Not there yet:
- blocks: `def`, `if`, `for`, `import`, and indented code
- syntax for matrix tags (`PositiveDefinite`, ...) and for the unit of an array's elements, which the checker supports internally
- physical constants and unit prefixes (the unit table has 17 units)
- `sky run` and `sky repl`, since the interpreter is still empty

## Layout

```
crates/
  sky-diag/     diagnostics and the terminal-safe renderer
  sky-syntax/   lexer: tokens, indentation, Unicode safety checks
  sky-ast/      typed AST with source spans, and the parser
  sky-units/    dimensions and the unit table
  sky-types/    type checker (units, shapes, matrix properties)
  sky-project/  project root discovery and import resolution
  sky-interp/   tree-walking interpreter (not started)
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
