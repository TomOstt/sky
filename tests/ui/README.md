Golden tests: each `.sky` file here has an expected-output file with its diagnostics. Passing and failing examples both live here.

- `foo.sky` is checked with `sky check tests/ui/.../foo.sky` (run from the workspace root), and its stderr must equal `foo.stderr` byte for byte. An empty `.stderr` means the file must check cleanly. Diagnostics use the format in `crates/sky-diag` (see `spec/diagnostics.md`); several diagnostics are separated by one blank line.
- The exit status must be 1 if `foo.stderr` contains an `error:` header and 0 otherwise (warnings alone do not fail a check).
- The harness is `crates/sky-cli/tests/ui.rs`; it walks every subdirectory. Regenerate expectations with `SKY_BLESS=1 cargo test -p sky-cli --test ui` and review the diff.
- Files here are stored byte for byte (`.gitattributes`: `-text`), because some tests depend on CRLF, tabs, or bidi/control characters.
- `lexer/`: tokens, indentation, and the lexer's security checks (`ok_*` clean, `warn_*` warnings only, `err_*` errors). Later phases (parser, type checker) will add diagnostics to some `ok_*` files; re-bless them then.
