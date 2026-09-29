# Parser decisions (PR 1: AST + Pratt expression parser)

Status of each item: **signed off** (Eng Lead agreed) or **needs Eng Lead sign-off** (a default chosen by Parser Coder).

## Token interface (signed off, temporary mirror)
- `src/token.rs` mirrors `crates/sky-syntax/src/token.rs` from PR #4 (the lexer, not merged) exactly: variant set, order, derives, and the `Kw::from_ident`/`Kw::as_str` helpers. It is marked `// TEMPORARY` and becomes `pub use sky_syntax::{Token, TokenKind, Kw};` once #4 merges.
- `Span` is `sky_diag::Span` (merged in #1), used directly and not redefined.
- The parser takes `&[Token]` plus the source `&str`, and reads token text as `source[span]`.

## Precedence (needs Eng Lead sign-off)
The table lives in `src/precedence.rs`, lowest to highest:

| level | operators | associativity |
|---|---|---|
| 1 | `== != < <= > >=` | **non-associative:** `a < b < c` is a parse error (reported, then recovered as left-assoc) |
| 2 | `+ -` | left |
| 3 | `* / @` | left |
| 4 | prefix `- + ∂ ∇` | prefix |
| 5 | `^` | right; its right operand may be a prefix expression (`2^-1`) |
| 6 | postfix `⁻¹`, call `f(..)`, index `M[..]`, attribute `.` | tightest |

- `-x^2` is `-(x^2)`, and `-A⁻¹` is `-(A⁻¹)`.
- A parenthesized comparison is not a chain: `(a < b) == c` parses.

## AST shape (needs Eng Lead sign-off)
- `Expr` is a struct holding `kind: ExprKind` and `span: Span` (likewise `Stmt`/`StmtKind`), so every node has a span without per-variant boilerplate. `Literal` and `UnitExpr` are enums that carry spans in each variant.
- `⁻¹` is its own node, `ExprKind::Inverse`, and is not desugared to `^ -1`, because the checker rewrites `A⁻¹ @ b` into a Cholesky solve.
- `@` is `BinaryOp::MatMul`.
- `∂` and `∇` are plain prefix nodes (`UnaryOp::Partial`/`Nabla`), with no derivative-form semantics.
- ASCII `d`, `inv`, `grad` are ordinary names: `d = 3` and `inv(A)` parse normally.
- There is no `±` operator: PR #4's lexer rejects `±` ("uncertainty (±) is not supported in v0.1") and lexes `+-` as `Plus`, `Minus`, so `a +- b` is `a + (-b)`.
- `ExprKind::Paren` is kept as a node for span fidelity and a future CST.
- Literals are not decoded: an int, float, or string node stores only its span. The only integer text the parser converts is unit exponents, and it uses checked conversion.
- Type annotations are ordinary expressions (`Matrix[3, 4]` is an `Index`). The checker, not the grammar, rejects things like `Matrix[N+1]`.

## Unit literals (signed off by Eng Lead, 2026-09-29)
The normative grammar is in `spec/v0.1.md`, "Unit literals".
- After a numeric literal (`Int`/`Float`), a following `Ident` starts a unit, with or without a space before it (`10 km`, `10km`).
- `*`, `/`, `^` join the unit only when directly adjacent to both neighbours (compared by span). Otherwise they are ordinary operators: `10 km / 30 min` is `Div(10 km, 30 min)`, and `10 m / t` is `Div(10 m, t)`.
- Once an adjacent operator is part of the unit, what follows must be valid unit syntax, or it is an error. For example, `10 m/2` gets a help line saying to put spaces around `/` to divide the quantity.
- Compound units use `*`, never juxtaposition. `5 kg m` is a parse error with the help `` `kg*m` ``.
- Exponents are a signed integer (`s^-2`) or a parenthesized rational (`m^(1/2)`, `m^(-3/2)`), with all tokens adjacent. They are stored as `Pow { num: i64, den: u64 }` with `den > 0` (1 for integers), exactly as written and not reduced. A zero denominator, or an integer that does not fit (`num` in `i64`, `den` in `u64`), is a parse error, never a panic.
- `m^2^3` (chained unit exponents) is an error.
- A name after a non-number (`x km`) is a parse error: "a unit may only follow a numeric literal", with a help line suggesting `x * km`.
- `5 meters + 20 kelvin` parses fine. The dimension mismatch is the checker's error.
- Postfix operators after a unit literal apply to the quantity: `10 km⁻¹` is `Inverse(10 km)`, not a unit `km⁻¹` (needs sign-off).

## Statements (needs Eng Lead sign-off)
- Top-level simple statements are separated by `Newline`: expression statements, `x = e`, `A: T`, and `A: T = e`.
- Assignment and annotation targets must be a plain name. `A[0] = 1` and `a.b = 1` are errors in v0.1.
- `a = b = c` (chained assignment) is an error.
- Blocks (`def`/`if`/`for`) are not in this PR. A keyword where an expression is expected gets "statements and blocks are not supported by the parser yet".
- `Indent` at the start of a line is reported ("indented blocks are not supported yet"), and the indented line is still parsed. `Dedent` is ignored.
- **Keywords cannot be attribute names:** `linalg.solve(A, b)` is an error because `solve` is a keyword. Should attribute position accept keywords? (Question for Eng Lead.)
- `Newline` inside `(...)`/`[...]` would be an error, but PR #4's lexer never emits one there (it suppresses newlines inside brackets, as Python does).

## Robustness (fuzz target)
- The parser never panics. All token access is bounds-checked, and text goes through `str::get`. It never calls `Span::new` on token-derived offsets, because that debug-asserts.
- A missing `Eof` is fine, and tokens after the first `Eof` are ignored.
- `MAX_DEPTH = 256` bounds recursion and AST height. It counts both nesting and operators folded into one chain, so a flat sum of more than ~255 terms is also an error. That bound is what keeps recursive `Drop` and printing safe. (Needs sign-off: we could raise it or make `Drop` iterative later.)
- `MAX_UNIT_FACTORS = 64` limits the number of names in one unit literal.
- Error recovery: at top level, an error skips to the next `Newline`.

## Errors
- `ParseError { message, span, secondary: Vec<sky_diag::Label>, help: Option<String> }`.
- It converts losslessly into `sky_diag::Diagnostic` (severity `error`) via `into_diagnostic()` or `From`, and renders in the `spec/diagnostics.md` format.

## Golden tests
- None are added in `tests/ui/` yet. They need the real lexer, and `sky check` does not call the parser yet.
- For now, tests use a clearly marked test-only tokenizer (`tests/common/mod.rs`) and S-expression snapshots (`Expr::to_sexpr`).
- When the lexer lands, `tests/ui/<name>.sky` + `.stderr` goldens will be added for every case here (the spec examples, the unit-literal examples and counter-examples, and the error cases).

## Open questions for Eng Lead
1. May keywords be attribute names? Today `linalg.solve(A, b)` is a parse error because `solve` is a keyword.
2. Should the parser eventually live in `sky-syntax` (its crate description says "lexer and parser") or stay in `sky-ast`?

Answered by PR #4: the lexer suppresses `Newline` inside brackets, and `Int` text is ASCII digits with `_` only between digits (the parser's checked exponent conversion accepts that).
