# sky-ast parser goldens

Run by `crates/sky-ast/tests/ui_parser.rs` (`cargo test -p sky-ast --test ui_parser`).

These follow the `tests/ui/` conventions (see `tests/ui/README.md` at the workspace root):
- `NAME.sky` has a sibling `NAME.stderr` holding the exact diagnostics, in the `spec/diagnostics.md` format. An empty `.stderr` means the file checks cleanly.
- `ok_*` files have no errors, and `err_*` files have at least one.

For each file, the runner lexes it with `sky_syntax::lex`, parses it with `sky_ast::parse_module`, and renders the diagnostics with sky-diag.

## Why they live here for now

The goldens are crate-local, like `crates/sky-units/tests/golden/`. `sky check` does not run the parser yet, and the `sky-cli` harness runs `sky check` on everything under `tests/ui/`, so parser error cases there would fail it. They move to `tests/ui/parser/` once `sky check` parses. At that point only the path in each header changes, so re-bless them.

To regenerate every `.stderr`, run `SKY_BLESS=1 cargo test -p sky-ast --test ui_parser`, then review the diff.
