# sky-units golden tests

Run by `tests/golden.rs` (`cargo test -p sky-units --test golden`). These are
crate-local until the parser and the `tests/ui/` harness exist.

Each `NAME.case` has a sibling `NAME.expected` holding the exact output.

- `pass_*.case` must finish without an error. An empty `.expected` means plain
  success with nothing shown.
- `fail_*.case` must end in an error. Its `.expected` ends with
  `error: <message>`, the `Display` of `DimMismatch`.

Case files are line-oriented. Blank lines and lines starting with `#` are
ignored. All lines in a file share one unifier, and the first error stops the
case.

```
unify <expr> with <expr>   # unify expected with found
show <expr>                # prints "<expr> = <expr with solutions applied>"
```

Expressions:

| Syntax                   | Meaning                                    |
|--------------------------|--------------------------------------------|
| `L M T I Θ N J`          | SI base dimensions (`N` is amount of substance, `J` luminous intensity) |
| `1`                      | dimensionless                              |
| `[km]`                   | the dimension of a unit from the unit table (`[N]` is the newton, `[J]` the joule) |
| `?a`                     | a dimension variable (fresh on first use)  |
| `a * b`, `a · b`, `a / b`| product, quotient (left-associative)       |
| `a^2`, `a^-1`, `a^(1/2)` | integer or rational power                  |
| `( ... )`                | grouping                                   |

To regenerate every `.expected` from the current output, run
`SKY_BLESS=1 cargo test -p sky-units --test golden`, then review the diff.

Exponent overflow (a numerator or denominator beyond `i64`) is reported as
`error: dimension exponent too large`, from either building an expression or
unifying. It is never a panic.

These cases exercise the `sky-units` library directly. User-facing cases move
to `tests/ui/` once `sky check` parses units; the cases here then stay only
as library-level tests.
