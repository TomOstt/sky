# sky-types golden tests

Run by `tests/golden.rs` (`cargo test -p sky-types --test golden`). These
exercise the shape, tag, and rewrite rules directly, because the parser
(#9) and the `tests/ui/` harness don't exist yet. **User-facing cases move to
`tests/ui/` once `sky check` parses shapes and tags.** The cases here then
stay only as library-level tests.

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
  is a power, which is never rewritten, even for `N = -1`.

To regenerate every `.expected` from the current output, run
`SKY_BLESS=1 cargo test -p sky-types --test golden`, then review the diff.
