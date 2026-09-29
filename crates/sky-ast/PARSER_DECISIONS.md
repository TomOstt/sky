# Parser decisions (PR 1: AST + Pratt expression parser)

Each item is either **signed off by Eng Lead** or **needs sign-off** (a default chosen by Parser Coder).

## Signed off by Eng Lead

**Token interface.** The parser uses the real lexer types from `sky_syntax` (#4): `Token`, `TokenKind`, `Kw`, and `Span` (a re-export of `sky_diag::Span`). It takes `&[Token]` plus the source `&str` and reads token text as `source[span]`.
- There is no `±` token. The lexer rejects `±`, and `+-` is `Plus`, `Minus`.
- `d`, `inv`, `grad` are plain names, so `d = 3` and `inv(A)` parse.

**Unit literals.** The normative grammar is in `spec/v0.1.md`, "Unit literals".
- A unit follows a numeric literal, with a space between them. The lexer rejects `10km`. The parser itself treats an `Int`/`Float` directly followed by an `Ident` token as a unit either way.
- A name is a unit only when it directly follows a number. `x km` is a parse error ("a unit may only follow a numeric literal"), with a help line suggesting `x * km`.
- `*`, `/`, `^` join the unit only when they are directly adjacent to both neighbours (compared by span). Otherwise they are ordinary operators: `10 km / 30 min` is `Div(10 km, 30 min)`, and `10 m / t` is `Div(10 m, t)`.
- `5 meters + 20 kelvin` parses fine. The dimension mismatch is the checker's error.

**Eng Lead's additions:**
1. Compound units use `*`, never juxtaposition. `5 kg m` is a parse error with the help `` `kg*m` ``.
2. Exponents are a signed integer (`s^-2`) or a parenthesized rational (`m^(1/2)`, `m^(-3/2)`), with all tokens adjacent.
   - They are stored as `Pow { num: i64, den: u64 }` with `den > 0` (1 for integers), exactly as written and not reduced.
   - A zero denominator, or an integer that doesn't fit (`num` in `i64`, `den` in `u64`), is a parse error, never a panic.
3. The grammar is written into `spec/v0.1.md` (part of this PR).

**Also signed off by Eng Lead** (review of #9's open questions):
- **Keywords as attribute names.** Any keyword is allowed as the name right after `.`, so `linalg.solve(A, b)` and `x.if` parse. This applies to attribute access only: a bare keyword (`solve = 1`) is still an error. The lexer is unchanged; the parser accepts `Keyword(_)` after `Dot`.
- **`10 m/2` is an error.** Once `*`, `/` or `^` touches both neighbours, it belongs to the unit, so what follows must be unit syntax. The help line says to put spaces around the operator to divide the quantity.
- **`a = b = c` (chained assignment) is an error.**

## Needs sign-off

**Precedence.** The table is in `src/precedence.rs`, lowest to highest:

| level | operators | associativity |
|---|---|---|
| 1 | `== != < <= > >=` (and `≠ ≤ ≥`) | **non-associative:** `a < b < c` is an error, reported and then recovered as left-assoc |
| 2 | `+ -` | left |
| 3 | `* / @` | left |
| 4 | prefix `- + ∂ ∇` | prefix |
| 5 | `^` | right; its right operand may be a prefix expression (`2^-1`) |
| 6 | postfix `⁻¹`, call `f(..)`, index `M[..]`, attribute `.` | tightest |

- `-x^2` is `-(x^2)`, and `-A⁻¹` is `-(A⁻¹)`.
- A parenthesized comparison is not a chain: `(a < b) == c` parses.

**AST shape:**
- `Expr` is a struct holding `kind: ExprKind` and `span`; likewise `Stmt`/`StmtKind`.
- `Literal` and `UnitExpr` are enums that carry spans in each variant.
- `Paren` is kept as a node.
- Literals are not decoded: a literal node stores only its span.

**Operators:**
- `⁻¹` is its own node, `ExprKind::Inverse`, and is not desugared to `^ -1`, because the checker rewrites `A⁻¹ @ b` into a Cholesky solve.
- `@` is `BinaryOp::MatMul`.
- `∂` and `∇` are plain prefix nodes (`UnaryOp::Partial`/`Nabla`), with no derivative-form semantics.

**Other choices:**
- Type annotations are ordinary expressions (`Matrix[3, 4]` is an `Index`). The checker rejects `Matrix[N+1]`.
- Assignment and annotation targets must be plain names. `A[0] = 1` and `a.b = 1` are errors.
- `10 km⁻¹` is `Inverse(10 km)`: postfix applies to the quantity, not to the unit.
- An `Indent` at the start of a line is reported ("indented blocks are not supported yet"), and the line is still parsed. `Dedent` is ignored until blocks exist (PR 2). A keyword where an expression is expected gets "statements and blocks are not supported by the parser yet".
- `MAX_DEPTH = 256` bounds recursion and AST height. It also counts operators chained at one level, so a flat sum of more than ~255 terms is an error. This is what keeps recursive `Drop` and printing safe.
- `MAX_UNIT_FACTORS = 64` limits the number of names in one unit literal.
- `ParseError` has an optional `help` line in addition to the message, primary span, and secondary labels.

## Robustness (fuzz target)
- The parser never panics. All token access is bounds-checked, text goes through `str::get`, and integer conversions are checked.
- It never calls `Span::new` on token-derived offsets, because that debug-asserts.
- A missing `Eof` is fine, and tokens after the first `Eof` are ignored.
- Error recovery: at top level, an error skips to the next `Newline`.

## Errors
- `ParseError { message, span, secondary: Vec<sky_diag::Label>, help: Option<String> }`.
- It converts losslessly into `sky_diag::Diagnostic` (severity `error`) via `into_diagnostic()` or `From`.

## Golden tests (`crates/sky-ast/tests/ui/`)
- `crates/sky-ast/tests/ui_parser.rs` lexes each `*.sky` with `sky_syntax::lex` and parses it with `parse_module`.
- It renders the lexer's diagnostics and the parse errors with sky-diag's renderer (`spec/diagnostics.md` format), passing the raw source to `SourceFile::new`. Diagnostics are joined by blank lines.
- It compares the result byte for byte with the sibling `.stderr` (empty means clean). `ok_*` files must have no error, and `err_*` files at least one.
- They are crate-local (like `crates/sky-units/tests/golden/`) because `sky check` doesn't run the parser yet, and the sky-cli harness runs `sky check` on everything under `tests/ui/`. They move to `tests/ui/parser/` when `sky check` parses.
- `sky-cli` and `sky check` are unchanged in this PR.
- Regenerate with `SKY_BLESS=1 cargo test -p sky-ast --test ui_parser`.

## Open questions for Eng Lead
1. Should the parser eventually move into `sky-syntax` (its crate description says "lexer and parser") or stay in `sky-ast`?
