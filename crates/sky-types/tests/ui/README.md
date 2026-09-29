# sky-types checker goldens

Run by `crates/sky-types/tests/ui_check.rs` (`cargo test -p sky-types --test ui_check`).

These follow the `tests/ui/` conventions (see `tests/ui/README.md` at the workspace root) and the parser goldens in `crates/sky-ast/tests/ui/`:
- `NAME.sky` has a sibling `NAME.stderr` holding the exact diagnostics, in the `spec/diagnostics.md` format. An empty `.stderr` means the file checks cleanly.
- `ok_*` files have no errors, and `err_*` files have at least one.

For each file, the runner lexes it with `sky_syntax::lex`, parses it with `sky_ast::parse_module`, and, if both succeed without errors, checks it with `sky_types::check_module_with`.

Extra siblings, both optional:
- `NAME.types`: every name bound after the last statement, as `name: type` in name order. A type is the shape, then the tags if any, then the element dimension in brackets, e.g. `Matrix[3,3] {Symmetric} [1]` or `Scalar [L·T⁻¹]`.
- `NAME.explain`: the Cholesky rewrite decisions, rendered as notes. It is required whenever the file has a rewrite, so every rewrite is pinned; an empty `.explain` pins "no rewrite". (No case here has a rewrite now: the rewrite cases need a declared positive-definite matrix, so they live in `tests/ui/types/`, where the CLI harness pins them through `sky check --explain`.)

## Declarations

Names with property tags or element units are declared in the file itself, with real Sky annotations (`A: PositiveDefinite Matrix[N, N]`, `v: Vector[3] m/s`), so there is no test-only seeding: the old `# env:` comment lines are gone. Every case that used them moved to `tests/ui/types/` at the workspace root, where `sky check` runs it end to end. This runner still reads that directory, but only for the `.types` files, which the CLI harness can't see.

## Why the rest live here

These cases need nothing but the checker (unit and shape errors, Bool rules, name resolution), and pin the checker's own output next to its `.types`. Parse errors of annotations (`x: 5 m`, `Matrix[N+1, N]`) are in `crates/sky-ast/tests/ui/`.

To regenerate every `.stderr` and the existing `.types` files (here and in `tests/ui/types/`), run `SKY_BLESS=1 cargo test -p sky-types --test ui_check`, then review the diff. The `tests/ui/types/` `.stderr` and `.explain` files are blessed by `SKY_BLESS=1 cargo test -p sky-cli --test ui`.

A line `# prelude` starts from `Env::with_prelude()`, so the physical constants (`c`, `h`, `ħ`, ...) are in scope, as they are under `sky check`. Without it the environment is empty.
