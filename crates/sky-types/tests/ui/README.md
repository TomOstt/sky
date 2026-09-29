# sky-types checker goldens

Run by `crates/sky-types/tests/ui_check.rs` (`cargo test -p sky-types --test ui_check`).

These follow the `tests/ui/` conventions (see `tests/ui/README.md` at the workspace root) and the parser goldens in `crates/sky-ast/tests/ui/`:
- `NAME.sky` has a sibling `NAME.stderr` holding the exact diagnostics, in the `spec/diagnostics.md` format. An empty `.stderr` means the file checks cleanly.
- `ok_*` files have no errors, and `err_*` files have at least one.

For each file, the runner lexes it with `sky_syntax::lex`, parses it with `sky_ast::parse_module`, and, if both succeed without errors, checks it with `sky_types::check_module_with`.

Extra siblings, both optional:
- `NAME.types`: every name bound after the last statement, as `name: type` in name order. A type is the shape, then the tags if any, then the element dimension in brackets, e.g. `Matrix[3,3] {Symmetric} [1]` or `Scalar [L·T⁻¹]`.
- `NAME.explain`: the Cholesky rewrite decisions, rendered as notes (what `sky check --explain` will print). It is required whenever the file has a rewrite, so every rewrite is pinned. An empty `.explain` pins "no rewrite".

## Seeding names: `# env:`

The language has no syntax yet for property tags or for the unit of an array's elements. A comment line of the form

```text
# env: NAME: SHAPE [TAG ...] [unit UNIT]
```

binds `NAME` before checking, for example `# env: A: Matrix[N,N] PositiveDefinite` or `# env: v: Vector[3] unit m/s`. `SHAPE` is `Scalar`, `Vector[d]`, `Matrix[m,n]`, or `Tensor[d1,d2,...]`, where each dimension is an integer or a shape-variable name. `TAG` is `Symmetric`, `PositiveDefinite`, `Diagonal`, or `Orthogonal`. `UNIT` uses table symbols joined by `*` and `/`, each with an optional integer power (`kg*m^2/s^2`). This is a test-harness convention, not Sky syntax, and it goes away once annotations can declare tags and units.

A line `# prelude` starts from `Env::with_prelude()`, so the physical constants (`c`, `h`, `ħ`, ...) are in scope, as they are under `sky check`. Without it the environment is empty.

## Why they live here

`sky check` runs the checker, and `tests/ui/check/` has end-to-end cases. Most files here seed names with `# env:` lines, which only this runner understands, so they stay here until annotations can declare tags and units.

To regenerate every `.stderr` and the existing `.types`/`.explain` files, run `SKY_BLESS=1 cargo test -p sky-types --test ui_check`, then review the diff.
