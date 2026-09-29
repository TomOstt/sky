# sky-types golden tests

Run by `tests/golden.rs` (`cargo test -p sky-types --test golden`). These are
**library-level** tests: they exercise the shape, tag, and rewrite rules
directly, with a tiny case language of their own, not Sky source.

User-facing behaviour is pinned by the checker goldens that walk the real AST
(`crates/sky-types/tests/ui/`, added by #19), and those move to `tests/ui/`
once `sky check` type-checks. Rules that only exist at the checker level live
there, not here: scalar scaling (`c * A`, `A * c`, `A / c`), unary minus
(`-A`), the `A ± c` and `c / A` errors, the `A^-1`-on-a-matrix error, and
source spans in diagnostics.

Each `NAME.case` has a sibling `NAME.expected` holding the exact output.

- `pass_*.case` must finish without an error. An empty `.expected` means plain
  success with nothing shown.
- `fail_*.case` must end in an error. Its `.expected` ends with
  `error: <message>`, the `Display` of `ShapeError`.

Case files are line-oriented. Blank lines and lines starting with `#` are
ignored. All lines in a file share one `ShapeUnifier`, and the first error
stops the case.

```
let NAME = TYPE            # declare a variable with a type
unify SHAPE with SHAPE     # shape unification (expected, found)
show EXPR                  # prints "EXPR : TYPE" with solved variables applied
explain LHS @ RHS          # prints "explain: <rewrite explanation>" or "explain: no rewrite"
```

- **SHAPE** is `Scalar`, `Vector[d]`, `Matrix[d,d]`, or `Tensor[d,...]`, with
  no spaces inside the brackets. Each `d` is an integer or a shape-variable
  name (fresh on first use, shared within the file).
- **TYPE** is a SHAPE followed by zero or more tags: `Symmetric`,
  `PositiveDefinite`, `Diagonal`, `Orthogonal`.
- **EXPR** combines names with `A @ B` (matmul), `A + B` and `A - B`
  (elementwise), `inv(A)` and `A⁻¹` (inverse), `transpose(A)` and `Aᵀ`, and
  parentheses.
- In `explain`, `inv(X)` or `X⁻¹` on the left is the explicit inverse. `X^N`
  is a power, which is never rewritten, even for `N = -1`. This is a
  library-level guarantee of `cholesky_rewrite` (`pass_power_no_rewrite`).
  In Sky source the checker never gets that far: `A^-1` on a matrix is an
  error there (#19).

To regenerate every `.expected` from the current output, run
`SKY_BLESS=1 cargo test -p sky-types --test golden`, then review the diff.
