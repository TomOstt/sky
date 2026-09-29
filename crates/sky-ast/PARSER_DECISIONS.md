# Parser decisions (sky-ast: AST + Pratt expression parser)

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

**Also signed off by Eng Lead** (review of #9):
- **`⁻¹` directly after a unit is an error.** `10 km⁻¹` would otherwise read as `(10 km)⁻¹` = 0.1/km, while readers expect 10/km; both have the same dimension, so the checker could not catch it. This also fires after the last factor of a compound unit (`9.8 m/s⁻¹`, `1 kg*m⁻¹`) and with a space before the glyph (`10 km ⁻¹`). The error is on the `⁻¹` token, with the unit as a secondary label and the help line "write `10 km^-1` for per-km, or `(10 km)⁻¹`" (number and unit taken from the source; compound units say "to invert `s`"). `(10 km)⁻¹` and `A⁻¹` stay valid. Superscript exponents (`10 m²`, `10 m⁻²`) never reach the parser: the lexer rejects superscript digits and a lone `⁻`.
- **`⁻¹` is never lowered to `^`.** `A⁻¹ @ b` is `MatMul(Inverse(A), b)`, and `A^-1` stays `Pow(A, Neg(1))`; the `ok_inverse_node` golden pins both trees.
- **`x**2` is an error:** two adjacent `*` tokens give "`**` is not an operator" with help "Sky writes powers as `^`" (also inside units: `10 m**2`). Separated stars (`x * *2`) are a plain missing-operand error.
- **`i64::MIN` exponents are accepted:** `m^-9223372036854775808` parses (the magnitude is read as `u64` and negated with a checked subtraction); one below it is an error.

**Boolean operators** (Eng Lead's decision, `sky-ast/bool-ops`):
- `and`, `or`, `not` are keywords (`Kw::And`, `Kw::Or`, `Kw::Not` in `sky-syntax`). As with every keyword, they are still allowed as attribute names: `x.and`, `np.logical.not(a)`.
- Precedence is Python's: `or` < `and` < prefix `not` < comparisons < everything else (full table below). `not a < b` is `not (a < b)`, `a < b and b < c` needs no parentheses, `not not x` is fine, and `a and b or c` is `(a and b) or c`. `and`/`or` are left-associative.
- The AST has a separate node for `and`/`or` (they short-circuit), and the S-expressions are `(and a b)`, `(or a b)`, `(not a)`.
- C-style spellings are lexer errors with a help line: `&` → "for logical and, use `and`", `|` → "for logical or, use `or`", `!` → "for logical negation, use `not`; for \"not equal\", use `!=` or `≠`" (`!=` is unchanged).

**`with_deep_stack`** (approved by Eng Lead): `sky_ast::with_deep_stack(f)` runs `f` on a scoped thread named `sky-deep-stack` with `DEEP_STACK_BYTES` (8 MiB) of stack and returns its result; a panic in `f` is resumed with its original payload. Every entry point that parses, checks, or interprets must go through it (`sky check`/`run`/`repl`, the fuzz harness). The closure and its result must be `Send`, so `Rc`-based state is created inside the closure.

## Needs sign-off

**Precedence.** The table is in `src/precedence.rs`, lowest to highest:

| level | operators | associativity |
|---|---|---|
| 1 | `or` | left (Eng Lead) |
| 2 | `and` | left (Eng Lead) |
| 3 | prefix `not` | prefix (Eng Lead) |
| 4 | `== != < <= > >=` (and `≠ ≤ ≥`) | **non-associative:** `a < b < c` is an error, reported and then recovered as left-assoc |
| 5 | `+ -` | left |
| 6 | `* / @` | left |
| 7 | prefix `- + ∂ ∇` | prefix |
| 8 | `^` | right; its right operand may be a prefix expression (`2^-1`) |
| 9 | postfix `⁻¹`, call `f(..)`, index `M[..]`, attribute `.` | tightest |

- `-x^2` is `-(x^2)`, and `-A⁻¹` is `-(A⁻¹)`.
- A parenthesized comparison is not a chain: `(a < b) == c` parses.

**Boolean operators** (`sky-ast/bool-ops`; the placement is Eng Lead's, these details are Parser Coder's):
- **A run of `and` (or of `or`) is one flat node:** `ExprKind::BoolOp { op: BoolOp, operands: Vec<Expr>, op_spans: Vec<Span> }`, with two or more operands in source order and one span per keyword. `a and b` prints `(and a b)` as specified, but `a and b and c` prints `(and a b c)`, not `(and (and a b) c)`. The meaning is the same (left to right, stopping at the first operand that decides), and it is what Python's AST does. A binary node would make a 10k-operand chain a 10k-deep tree: either it counts against `MAX_DEPTH` (so a long chain is an error), or it doesn't (so `Drop`, the printer, the checker, and the interpreter all recurse 10k deep). Flat runs parse in a loop, add one nesting level however long they are, and keep the tree height bounded. Mixed runs nest by precedence: `a and b or c` is `Or[And[a, b], c]`, and a parenthesized run stays a separate `Paren` operand.
- **`not` is `UnaryOp::Not`** inside the existing `ExprKind::Unary`, not a new node. It is an ordinary one-operand prefix operator (it doesn't short-circuit), so it fits the prefix table, carries `op_span` like `-x`, and needs no new printer or checker plumbing.
- **`not` under a tighter operator is an error.** `not` may start an operand only where `and`, `or`, `not`, `=`, `(`, `[`, `,`, or the start of the line would allow a whole comparison. So `-not x`, `a < not b`, `a + not b`, and `2^not x` are errors, as in Python. The message is "`not` binds more loosely than `-`, so it needs parentheses here", on `not`, with the operator as a secondary label and the help "write `(not x)`" (the source text, up to 40 characters).
- **`and`/`or` where an operand should be** (`x = and b`) get "expected an expression, found keyword `and`" (without the "statements and blocks" note other keywords get).
- **Depth:** a `not` counts one level, like `-`. An `and`/`or` run counts one level in total, however many operands it has.

**Bool literals** (`sky-ast/bool-ops`; the Bool typing rules are in `spec/v0.1.md`, "Booleans and conditions"):
- `true` and `false` are keywords (`Kw::True`, `Kw::False`, lowercase, to match the team's existing golden `true or (3 m + 2 s)`).
- **AST:** `Literal::Bool { value: bool, span: Span }`, a new variant of the existing `Literal` enum, not a new `ExprKind`. All literal leaves stay in one place, `Literal::span()` covers it, and the checker types literals in one `match`. Unlike numbers and strings it carries its value, because there's nothing to decode. S-expressions print `true`/`false`.
- **Downstream:** adding a variant breaks exhaustive `match`es on `Literal`. `sky-types/check-walk` has one (`check.rs`, `Literal::Int(_) | Literal::Float(_)` / `Literal::Str(_)`), which needs a `Literal::Bool { .. }` arm when it merges main.
- A unit can't follow a Bool: the unit-literal rule only starts after `Int`/`Float`, so `true m` is "unexpected name `m` after an expression; a unit may only follow a numeric literal".
- They stay attribute names after `.` (`y = x.true` parses). Since `sky-ast/assign-targets`, `x.true = 1` parses as an attribute target, exactly like `x.flag = 1` (it was a parse error while targets were plain names; the golden is now `ok_assign_attribute`). The v0.1 checker rejects both writes.
- `true = 1`, `false = 0`, and `true: Bool` are errors on the literal: "cannot assign to `true`" / "cannot annotate `true`", with the help "`true` and `false` are Bool literals, not names". Other non-name targets keep the generic "invalid assignment target" message.
- `True`/`False` are ordinary identifiers, and the parser does nothing special with them. The checker should give an undefined `True`/`False` the help "Sky writes booleans as `true`/`false`" (Dinesh's side).

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
- Assignment targets: see "Assignment targets and compound assignment" below. Annotation targets must be plain names.
- An `Indent` at the start of a line is reported ("indented blocks are not supported yet"), and the line is still parsed. `Dedent` is ignored until blocks exist (PR 2). A keyword where an expression is expected gets "statements and blocks are not supported by the parser yet".
- **Depth semantics.** `MAX_DEPTH = 256` bounds recursion and AST height, which also keeps recursive `Drop` and printing safe. Each nesting level counts exactly once: a `(`...`)` group, a prefix operator (including `not`), a call, an index, an attribute, a `⁻¹`, each infix operator folded into an operand, and each `and`/`or` run. So 256 nested parens, calls (`f(f(...))`), indexes (`a[a[...]]`), prefix operators, or a 256-operator `^` chain parse, and 257 give "expression is nested too deeply (the limit is 256 levels)". Operators chained at one level count too, so a flat sum of 257 terms (256 `+`) is fine and one more `+` is an error. Error paths and node building are kept out of the recursive functions, so a level costs about 3 KiB of stack in debug builds and under 1 KiB in release; a test parses, prints, and drops 256 levels of each kind on a 1 MiB thread.
- `MAX_UNIT_FACTORS = 64` limits the number of names in one unit literal.
- `ParseError` has an optional `help` line in addition to the message, primary span, and secondary labels.

**Assignment targets and compound assignment** (`sky-ast/assign-targets`; the AST shape is approved by Eng Lead, the details below are Parser Coder's):
- **AST:** `StmtKind::Assign { target: AssignTarget, value }` and a new `StmtKind::CompoundAssign { target, op: CompoundOp, op_span, value }`. `AssignTarget` is `Name(Ident)`, `Index { base, items, bracket_span, span }`, or `Attribute { base, name, span }`. `Binding` is unchanged: only a plain name can be annotated.
- **Parsing:** the left side is parsed as an ordinary expression, then converted. A name, index, or attribute becomes a target; anything else is an error on the target's span, with the operator as a secondary label ("because of this") and the help "only a name (`x`), an index (`A[i]`), or an attribute (`a.b`) can be assigned to". Messages name the kind of target: "cannot assign to a function call `f(x)`", "... a parenthesized expression `(x)`" (help also says to remove the parentheses), "... a literal `1`", "... a unit literal `10 m`", "... an operator expression `a + b`". `true = 1`/`true += 1` keep #24's "cannot assign to `true`".
- **The parser accepts index and attribute writes.** In v0.1 the checker rejects them: arrays are immutable (decided by Eng Lead), and nothing has assignable attributes yet.
- **Compound assignment is never desugared.** `x += e` is `CompoundAssign`, printed `(+= x e)`, never `(= x (+ x e))`: the target is evaluated once and diagnostics can point at the `+=` token (`op_span` covers both characters). `CompoundOp` has `Add Sub Mul Div Pow MatMul` (`+= -= *= /= ^= @=`), `symbol()`, and `binary()` (the `BinaryOp` it applies). Comparisons and `and`/`or` have no compound form.
- **Tokens:** `PlusAssign`, `MinusAssign`, `StarAssign`, `SlashAssign`, `CaretAssign`, `AtAssign` in `sky-syntax`, lexed greedily: `x^=2` is `x ^= 2`, while `a+ =b` is `+` then `=` (a parse error: "expected an expression, found `=`"). `**=` lexes as `*`, `*=`, and the parser reports "`**=` is not an operator" with the help "Sky writes powers as `^`; the compound assignment is `^=`".
- **Chains are errors:** `a = b = c` (unchanged message), `x += y += z` and `a = b += c` ("chained assignment (`x += y += z`) is not supported", on the second operator). `x: T += 1` is "`+=` cannot follow a type annotation".
- **Slices:** `ExprKind::Index { base, items: Vec<IndexItem> }` (was `indices: Vec<Expr>`). `IndexItem` is `Expr(Expr)` or `Slice { start: Option<Expr>, stop: Option<Expr>, colon_span, span }`; `IndexItem::as_expr()` returns `None` for a slice (for readers of generic arguments like `Matrix[3, 4]`, where a slice is a checker error). Slices parse on the read side too: `A[0, :]` → `(index A 0 (: _ _))`, `A[1:3]` → `(index A (: 1 3))`. There is no step (the spec has none): `A[1:2:3]` and `A[::2]` are "slice steps (`start:stop:step`) are not supported in Sky v0.1", on the second `:`.
- **Depth:** an index counts one level whatever its items are; slice bounds are parsed at the index's level, so `a[1:a[1:...]]` nests exactly like `a[a[...]]` (256 parse, 257 is the depth error). Index items are parsed from one frame with a single `expr_bp` call site, so an index level costs no more stack than before (debug: 256 levels of `a[...]` or `a[1:...]` parse, print, and drop in about 832 KiB, like parens).
- **Quoted source text is bounded.** Diagnostics quote at most 64 characters (`MAX_QUOTED_CHARS`) of the first line of any source text, then `…`, so a 100k-character name gives a short message. This applies to every message that quotes a name, number, or unit, not just the new target errors.
- **Downstream:** exhaustive matches on `StmtKind` need a `CompoundAssign` arm, code matching `Assign { target, .. }` now gets an `AssignTarget` (`AssignTarget::Name(id)` is the old shape), and code destructuring `ExprKind::Index { base, indices }` must use `items` (`items.iter().filter_map(IndexItem::as_expr)` for the old behaviour). The `parse_module` fuzz harness (`fuzz/harness/parse_module.rs`, from #26) is updated in this PR: its iterative span walk covers targets, index items, and slices, and its token list has the six new kinds. The open PR that still needs the change is `sky-types/check-walk` (#19, `check.rs`). Seeds `nest_index_256` (256 nested index levels, must parse) and `compound_assign` (all six operators on name, index, and attribute targets) are in `fuzz/corpus/parse_module/`, pinned by `crates/sky-ast/tests/fuzz_parse_module.rs`.

## Robustness (fuzz target)
- The parser never panics. All token access is bounds-checked, text goes through `str::get`, and integer conversions are checked.
- It never calls `Span::new` on token-derived offsets, because that debug-asserts.
- A missing `Eof` is fine: the end of the slice acts as `Eof`, a zero-width span at the end of the last token (`Span::point(0)` for an empty slice). Tokens after the first `Eof` are ignored.
- Error recovery: at top level, an error skips to the next `Newline`.

## Errors
- `ParseError { message, span, secondary: Vec<sky_diag::Label>, help: Option<String> }`.
- It converts losslessly into `sky_diag::Diagnostic` (severity `error`) via `into_diagnostic()` or `From`.

## Golden tests (`crates/sky-ast/tests/ui/`)
- `crates/sky-ast/tests/ui_parser.rs` lexes each `*.sky` with `sky_syntax::lex` and parses it with `parse_module`.
- It renders the lexer's diagnostics and the parse errors with sky-diag's renderer (`spec/diagnostics.md` format), passing the raw source to `SourceFile::new`. Diagnostics are joined by blank lines.
- It compares the result byte for byte with the sibling `.stderr` (empty means clean). `ok_*` files must have no error, and `err_*` files at least one.
- A file may also have a `.ast` sibling holding `Module::to_sexpr` (one statement per line), which pins the tree shape and not just the absence of errors (e.g. `ok_inverse_node.ast`: `(= x (@ (inverse A) b))`).
- They are crate-local (like `crates/sky-units/tests/golden/`) because `sky check` doesn't run the parser yet, and the sky-cli harness runs `sky check` on everything under `tests/ui/`. They move to `tests/ui/parser/` when `sky check` parses.
- The runner parses inside `with_deep_stack`, like every entry point should.
- Regenerate with `SKY_BLESS=1 cargo test -p sky-ast --test ui_parser`.

**`with_deep_stack` details** (Parser Coder):
- If the thread can't be spawned, it panics with "failed to spawn 8 MiB parser thread: {e}" rather than running `f` inline. Running inline on a stack of unknown size would quietly drop the guarantee that deep-but-legal input can't overflow. Spawn failure means the OS is out of threads or memory, which is fatal for a CLI anyway.
- One thread per call. The cost is about 40 µs (measured, debug and release), negligible next to a `sky check` run; the REPL can wrap each input (or its whole loop).
- Nothing is wired into `sky-cli` or the fuzz harness in this PR; those entry points belong to other open PRs.

## Open questions for Eng Lead
1. Should the parser eventually move into `sky-syntax` (its crate description says "lexer and parser") or stay in `sky-ast`?
