# Sky IR (SIR): design

Status: draft for Phase 1 of [`ROADMAP.md`](../../ROADMAP.md). Owner: Eng Lead. Date: 2026-09-29.

## Why an IR, and why now

Today the checker computes units, shapes and matrix tags, uses them for diagnostics and one rewrite, and then discards them. Every later capability needs that knowledge in one place it can analyse and transform: fusion, autodiff, error budgets, solver choice, native code, GPUs. SIR is that place. If Sky instead added each capability as its own path from the AST, it would end up with a dozen half-compilers that disagree with each other.

SIR must be in place before any optimizer or backend work (Phase 2 onwards). Its first version only has to represent v0.1 programs faithfully. It does not need to be fast.

## Goals

1. **Keep meaning.** Every value carries its dimension, symbolic shape, closed tag set, numeric representation, and (later) error bound. Nothing is lowered to loops until a pass decides to.
2. **Make every capability a pass.** Analyses read SIR; transforms rewrite it; backends consume it. There are no side channels.
3. **Explain everything.** Every transform records *what* it changed, *why* (the facts it relied on), and *which rule*. `sky check --explain` prints these records.
4. **Be verifiable.** A verifier checks the IR's invariants after every pass in debug builds and tests. A reference interpreter defines the semantics, and every optimization is differentially tested against it.
5. **Be extensible without changing the core.** Libraries add operations, facts and rewrite rules as data (roadmap ideas 70 and 77).
6. **Stay stable and versioned.** SIR has a data-only text format with an edition header (spec non-negotiable 3; roadmap idea 75), so IR can be stored in golden tests, caches and reproducibility packages.

## Non-goals for version 1
- Loop-level, memory-layout or hardware representation. Those are later levels (see "Levels").
- Performance of the IR itself, beyond "compiles a 10 000-line program in well under a second".
- Replacing the checker. v1 is lowered *from* the checked AST and reuses sky-types' results. Moving checking onto SIR is a later decision.

## Levels

| Level | Name | Contains | Produced by |
|---|---|---|---|
| L1 | **Semantic IR** | mathematical operations on typed values, structured control flow, calls | lowering from the checked AST |
| L2 | Strategy IR | chosen algorithms (e.g. `cholesky_solve`), chosen precision per value, storage formats (dense, diagonal, CSR) | strategy passes over L1 |
| L3 | Kernel IR | explicit iteration spaces, fused kernels, memory and lifetimes | kernel lowering |
| L4 | Backend | Cranelift or LLVM functions, MLIR modules, or direct interpretation | code generators |

Each level is the same data structure with a different set of allowed operations. The verifier enforces which operations are legal at which level. Transitions are one-way, so later levels never need to recover meaning. This is the "progressive lowering" approach proven by MLIR; if Sky adopts MLIR for L3 and L4, L2 maps onto its `linalg` and `sparse_tensor` dialects.

## L1 data model

```text
Module     { edition, functions: [Function], facts: FactTable }
Function   { name, params: [Value], results: [Type], body: Region, effects: EffectSet }
Region     { blocks: [Block] }            -- structured: one entry block, results via yield
Block      { args: [Value], ops: [Op] }
Op         { id, kind: OpKind, operands: [Value], results: [Value], regions: [Region],
             attrs: AttrMap, span: Span }
Value      { id, ty: Type }               -- SSA: defined once
```

- **Structured control flow** (in the style of MLIR's `scf`): `if` and `while` are operations with nested regions, not unstructured jumps. Structured regions make fusion, autodiff and parallelisation much simpler, and v0.1 has no `goto`.
- **SSA values** make dataflow explicit. That is what dead-computation elimination, lazy evaluation and dependency-graph parallelism (ideas 22, 29, 30) need.
- **Spans on every operation** keep diagnostics and explanations pointing at the user's source after any number of passes. A transform that merges operations joins their spans. An operation also has **operand spans** (`operand_spans`, empty when unknown): the source span of each operand at the place it is read, which a runtime error's labels point at (`this is zero`). A pass keeps the span of an operand that survives, joins where it merges, or leaves the field empty, and the interpreter then uses the operation's span.

### Types

```text
Type = Number { dim: Dimension, shape: Shape, tags: TagSet, repr: Repr }
     | Bool             -- a scalar condition (v1 has no Bool arrays)
     | Str
     | Nothing          -- what `print` and a function without a return value give
     | Unknown          -- the checker couldn't type it; passes treat it conservatively

Repr = Unchosen         -- L1 default: a strategy pass decides
     | Exact            -- rational or symbolic until approximation is needed (idea 9)
                        -- reserved: no pass chooses it (see "Exact and symbolic values")
     | F32 | F64        -- chosen by precision passes (F16, BF16, F128 come with Phase 3)
```

Functions are not first-class values in v1: calls name their callee (`call @f`). A function's parameter types are also the types of its entry block's arguments, and the verifier checks that the two agree. `if`, `while` and `fused` report no effects of their own; a pass computes their effects from their regions.

`Dimension`, `Shape` and `TagSet` are the existing types from sky-units and sky-types, reused as they are. SIR depends on those crates and does not copy them. Tags stay *closed* (implications applied), as the checker already guarantees.

Planned extension points, which v1 leaves out: an error bound per value (interval or affine form, Phase 3), a distribution for uncertain values (idea 10), and a coordinate frame (original vision §12).

### Operations (L1, v1)

| Group | Operations |
|---|---|
| Constants | `const.rational`, `const.float`, `const.bool`, `const.str`, `const.symbol` (π, e, √2) is deferred (see "Exact and symbolic values"); `literal.error[too_large\|overflow]`, a literal with no finite `f64` value, which fails when it runs (as `sky run` does, only where it is evaluated) |
| Arithmetic | `add`, `sub`, `mul`, `div`, `neg`, `pow` (with a constant rational exponent attribute when known), `inverse` |
| Linear algebra | `matmul`, `transpose`, `solve` (produced by rewrites, never written by users in v1) |
| Vector builtins | `norm` (one operand of any shape; the result is a scalar of its dimension: the Euclidean or Frobenius length) and `dot` (two vectors of one length; a scalar of the product of the dimensions). Both have effect `fail` (a sum of squares or products can overflow) and evaluate through `sky_interp::vecfn::norm` and `dot`, the functions `sky run` calls, with the call's span and the operands' spans, so the results and diagnostics are byte-identical. `dot` is commutative; the passes treat both as opaque operations that can fail (`fold` evaluates `norm` of a constant scalar) |
| Arrays | `array.literal`, `index`, `slice`, `zeros`, `identity`, `len`; `add`, `sub`, `mul` (scaling), `div`, `neg`, `inverse` and `matmul` also work on arrays |
| Comparison and logic | `cmp.{eq,ne,lt,le,gt,ge}` (on Bools and strings only `eq`/`ne`), `not`; `and`/`or` short-circuit, so they lower to `if`; `expect.bool[and\|or\|not\|if\|while]`, the run-time Bool check of an operand the checker could not type (an unannotated parameter's result), which fails with `sky run`'s wording and on the operand's span; `expect.number`, the same for the operand of a unary `+` (which lowers to nothing else), whose result is the operand |
| Contracts | `assert[assert\|requires\|ensures] "text" [fn "f"] [msg "m"] names("a", ...) %cond, %a, ...`: no results, effect `fail`; operand 0 is the condition (a Bool, or `unknown`, checked when it runs) and the others are the values of `names`, which only the failure message uses; it fails with `sky_interp::contract`'s message on the condition's span (spec "Contracts"). Not allowed in a `fused` body |
| Control | `if` (regions), `while` (regions), `yield`, `call`, `return` |
| Fusion | `fused` (one region: a scalar body run per element; made by the `fuse` pass, see "Fusion") |
| Display | `show "unit"` (`v as km/h`: the operand written in a unit, a `str`, only ever printed) |
| Environment | `param`, `input "name"` (a value-less declaration: no operands, a result of the declared type; nothing supplies it in v0.1, so it fails when run with "`name` is declared but has no value", at the declaration), `print` (effectful) |

Every operation kind is declared once, in a table with:
- its operand and result type rule (a function reusing `sky_types::rules`);
- its effects;
- its algebraic properties, such as commutative, associative, linear or idempotent, which rewrites and the e-graph consume;
- its reference-interpreter semantics.

Library-defined operations register in the same table (goal 5).

### Effects

`EffectSet ⊆ {Pure, Read, Write, IO, Random, Sync}`. `Pure` operations may be reordered, duplicated, eliminated, cached or run in parallel. Everything else keeps its relative order. Effects are inferred for functions (and can later be declared, as in `pure fn`, original vision §23). This is the minimum needed for safe reordering, communication avoidance and caching (ideas 22, 23, 68).

v1 implements three of them (`io`, `fail`, `call`; the rest come with the operations that need them). `sky_ir::effects` infers a function's set as the union over its operations and nested regions, with a `call` taking the callee's effects (and keeping `call` as the mark that a call can also exhaust the call depth). Recursion is a fixed point: a caller is revisited whenever a callee's set grows. Lowering runs the inference instead of computing effects itself, and the verifier rejects a function whose declared effects miss an inferred one. Declaring more is allowed, so a pass that removes code leaves a valid module, and the `effects` pass can tighten the declarations afterwards.

### Facts

`FactTable` holds knowledge that isn't part of a type, keyed by value or function: "this matrix is reused 400 times", "estimated condition number", "sparsity pattern", "shape variable N = 4096 at run time". Each fact records its provenance: declared, proven by a pass, or observed by profiling. Passes may rely only on proven or declared facts for correctness. Observed facts may guide *choices* but never *meaning*.

## Passes

A pass is:

```rust
trait Pass {
    fn name(&self) -> &'static str;
    fn level(&self) -> Level;                                  // where it may run
    fn run(&self, m: &mut Module, cx: &mut PassCx) -> Changed; // cx collects explanations
}
```

The pass manager runs a declared pipeline and calls the verifier before the first pass and after every pass (always in tests and in `sky check`, in debug builds elsewhere); a failure names the pass that produced the invalid IR. It can print the IR after each pass (`sky check --emit-ir[=<stage>]`, below) for debugging and golden tests. The `level()` method is left out until there is a second level.

v1 ships these passes:
1. **lower**: checked AST to L1. Uses the checker's types; never re-derives them. It infers function effects with the `effects` pass's rule instead of computing them itself.
2. **verify**: SSA dominance, type rules per operation, closed tags, legality by level, and effect consistency (declared effects include the inferred ones).
3. **effects** (`sky_ir::effects`): the union over operations and regions, calls taking the callee's effects, recursion by fixed point. Not part of the default pipeline (lowering already ran it); available as a pass that also tightens declarations.
4. **fold**: constant folding, exact on rationals and bit-identical to the reference interpreter on floats.
5. **egraph**: rewriting with `egg` under the `Reproducible` contract (exact and conditional rules only).
6. **contracts**: removes an `assert` whose condition the exactness analysis proves true (`CONTRACT-001` for `assert`, `-002` for `requires`, `-003` for `ensures`), from constants and from the arguments of every call of the function (the module is closed, so the hull of the call sites' enclosures bounds a parameter); it says "this `requires` always holds". Only the failure is removed: the operations that compute the condition stay. See "Contracts" in the spec.
7. **dce**: removes pure operations whose results are never used, and operations that can fail by kind but provably cannot here (`DCE-002`), with an explanation when user-written code is removed (idea 30).
8. **fuse**: trees of elementwise array operations become one `fused` kernel (`FUSE-001`); see "Fusion" below.
9. **cholesky**: `matmul(inverse(A), b)` with PositiveDefinite A becomes `solve[cholesky](A, b)`. Its explanation is the checker's sentence for the same rewrite, word for word (`sky_opt::cholesky_text`).

Alongside the passes, `sky_ir::exact` (see "Exact and symbolic values") is an *analysis*: it changes nothing, so it is not in the pipeline, and each pass that wants it (`fold` for explanations, `dce`, the e-graph) computes it again rather than caching it across a change.

The default pipeline (`sky_opt::default_pipeline`) is 4, 5, 6, 7, 8, 9 in that order: fold first so the e-graph sees constants; contracts after the e-graph (it sees the simplified conditions) and before dce (which removes the comparison a dropped check leaves unused); the e-graph before dce because it leaves the operations it replaced behind; dce before fuse because a dead operation left behind counts as a second use of its operands and would keep a temporary alive; fuse after the e-graph because a `fused` body is opaque to fold and the e-graph (they would otherwise rewrite it as scalar code, and `neg` differs, below); cholesky last so its `solve` is never rewritten (fuse and cholesky touch different operations, so their relative order does not matter). `sky-opt`'s passes and `sky-egraph`'s rewriting each keep their own record type (`Rewrite`), and convert to `Explanation` in one place each (`Rewrite::explain`, through `PassCx::explain`), so rule ids and texts are unchanged.

### Fusion

A chain such as `c = (a + b) * 2 - d` over arrays builds a temporary array per operation. The `fuse` pass turns a chain (in general a tree) of elementwise operations into one `fused` operation that computes each element once. It records the loop and the composition, not machine code: there is no code generation yet, and a later backend gets one kernel per `fused`. The payoff now is explicit structure, and an explanation of each kernel in `--explain`.

```text
%c = fused %a, %b, %two, %d : num[1; (3)] {
  ^1(%x: num[1; ()], %y: num[1; ()], %s: num[1; ()], %e: num[1; ()]):
    %t1 = add %x, %y : num[1; ()] @ 47..52
    %t2 = mul %t1, %s : num[1; ()] @ 46..57
    %r = sub %t2, %e : num[1; ()] @ 46..61
    yield %r @ 46..61
} @ 46..61
```

**The operation** (`OpKind::Fused`, specified in `op.rs`):

* Operands: `k >= 1` values, arrays of one shape and scalars, at least one array. A scalar operand is broadcast, as in the scaling `A * s`.
* One result: an array of that shape, whose dimension is that of the value the body yields.
* One region with one block. Its arguments are the per-element values, one per operand in order (a scalar of the operand's dimension). Its operations are scalar and it ends in `yield` of one value.
* The verifier checks that the body is scalar-only (every result a scalar number or a Bool), pure or `Fail` only (no `print`, no `call`), has no nested regions (no control flow in v1), and is self-contained (it uses only its arguments and its own results); that the argument count and dimensions match the operands; and that the result's shape is the operands' and its dimension the yield's.
* Effects are those of the body: `fail` if any operation in it can fail. `properties()` says pure, as for `if` and `while`, and effect inference and the passes look inside. The text format needs nothing new (regions already print and parse), and canonicalization treats the body like any region.
* Element `i` of the result is what the body computes with element `i` of each array operand. Each body operation is evaluated by the code an unfused operation uses (`sky_interp::arith`), so results are bit-identical, with one exception: `neg` in a body is `0 - x`, as `neg` on an array is (a zero stays `+0`, where scalar `neg` gives `-0`). For that reason fold and the e-graph leave `fused` bodies alone.

**Failure semantics.** The unfused chain runs one operation over all elements, then the next, and reports the *first operation in program order* that fails at any element (division by zero, overflow); an error carries its operation's message and span, never an element. Per-element evaluation could meet a later operation's failure at an earlier element first. Two designs were possible: fuse only when at most one operation can fail, or define the fused operation's error to be the unfused one. We chose the second, so no chain is refused: **a `fused` reports the error of the earliest failing operation in body order, over all elements.** The reference interpreter runs the elements in order; when body operation `k` fails, it remembers the error and runs the remaining elements only up to operation `k`, so an earlier operation failing at a later element replaces it. A backend may visit elements in any order as long as it reports that error. The body keeps the operations in the unfused program order and their own spans, so the error is the same message at the same span. This is tested three ways: hand-built kernels where the naive order gives a different span (in `sky-opt` and `sky-ir-interp`), a differential property test over random elementwise programs with division by zero and overflow (the same output, error message and span before and after `fuse`), and the pipeline test.

**Native kernels.** For long arrays the SIR interpreter compiles the body once (`sky_jit::kernel`, Cranelift) and runs it as an unrolled 128-bit vector loop, when the interpreter has spent about the compile time (512 elements) on that `fused` in the run; `SKY_JIT=off` and `SKY_JIT_MIN_ELEMENTS=<n>` turn it off or move the threshold. The bytes and the error are the interpreter's: arithmetic is plain IEEE (never contracted to a fused multiply-add), `neg` is `0 - x`, math functions call `sky_interp::math::apply`, and the loop only *detects* a failure, by accumulating a NaN for every non-finite value that a divisor or a math call could hide, plus the result. Only then does the kernel run the interpreter's own algorithm (the rule above) to produce the exact diagnostic, so "the first failing operation in body order" is defined by one piece of code. A body the kernel does not implement, or whose dimensions do not agree, is refused and stays on the interpreter. Proved by a differential property test (native against interpreted on random bodies and overflowing data, in `sky-ir-interp`), and against the reference path in `sky-jit`; measured by `cargo bench -p sky-jit --bench kernels`.

**What is fused.** A candidate is `add`, `sub`, `mul`, `div` or `neg` on arrays of known type, laid out as the checker allows (two arrays of one shape for `add`/`sub`, an array and a scalar for `mul`, an array over a scalar for `div`). A candidate is absorbed into its consumer when its result has exactly one use, which is an operand of a candidate in the same block. A *tree* is a candidate with everything absorbed into it, and a tree of two or more operations becomes one `fused` at the root's position. Scalar operands stay outside as operands. The kernel keeps the root's result value (name and users) and the join of the operations' spans, and the operations move into the body with their own spans and names.

Moving the operations to the root's position must not reorder effects: an operation that can fail is never moved past another operation that can fail, print, call, loop, or touch an `Unknown`-typed value (which may fail dynamically). If a tree would move past one, the tree is not fused as a whole; its subtrees are tried, and the pass repeats until nothing is left, so it is idempotent.

**Explanation.** One `FUSE-001` per kernel, at the joined span. For the chain above, `sky check --explain` says:

```text
note: [FUSE-001] `(a + b) * 2 - d` is rewritten to `fused(a, b, 2, d)`
  = note: applied by the `fuse` pass
  = note: because fused 3 elementwise operations into one kernel; saves 2 temporary arrays of 3 elements
  = note: because every intermediate result is used exactly once, by the next operation, and every array operand is a Vector[3]
  = note: because 3 of the operations can fail (`add`, `mul`, `sub`); the kernel reports the failure of the first of them in program order that fails at any element, which is the error the separate operations gave
```

It also names the user's variables that stop being materialized (`` `t` is only used here ``), and says "no operation can fail" or "only `div` can fail, so a failure has the same message and place as before" when that is the case.

### `sky check` and the passes

* `sky check --emit-ir[=<stage>]` prints the SIR text to stdout: `lower` (straight out of lowering), `after:<pass>` (or just the pass name: `fold`, `egraph`, `contracts`, `dce`, `fuse`, `cholesky`), or `final` (the default, the end of the pipeline). With the flag, stdout is only the IR, so it can be piped into a file and parsed back; the `edition:` and `ok` lines go to stderr.
* `sky check --explain` also shows the passes' explanations, merged with the checker's notes in source order. The checker's Cholesky note is kept as it is (it names the operands by their source text, which SIR does not keep), and the pass's explanation of the same site is dropped, so it is never shown twice. A site only the IR sees (`B = inv(A)`, then `B @ b`) is shown once, from the pass, in the same words.
* A program that checks but cannot be lowered yet (`sky_lower::LowerError`, today: array literals) is still `ok`. `--emit-ir` prints a note saying why there is no IR and nothing on stdout; `--explain` alone is unchanged.

### Explanations

```text
Explanation { pass, rule: RuleId, span: Span, before: String, after: String,
              because: [Fact or type property], effect_on_semantics: Unchanged | WithinBound(err) }
```

This generalises the checker's `RewriteSite`. Every explanation is a note diagnostic in `--explain`. A rewrite that removes code has an empty `after`. `RuleId`s are stable (e.g. `LA-CHOLESKY-001`), so users and documentation can refer to them.

## Rewrites and the e-graph (Phase 2, designed now)

Pure L1 expression trees are loaded into an e-graph (the `egg` crate, MIT). Rewrite rules come from three places: the operation property table, built-in identity sets (trigonometry, linear algebra, exponent laws), and library modules. Extraction picks the cheapest equivalent form under a cost model. Each applied rule becomes an explanation.

A rewrite rule must state its side conditions in terms of facts: for example, `Aᵀ → A` requires `Symmetric ∈ tags(A)`, and `x/x → 1` requires `x ≠ 0` proven. That keeps identity-based optimization safe (ideas 5 and 6). Rules that are only valid over the reals, not over floats (for instance reassociation), are tagged and allowed only when the program's numerical contract permits them (`@fast` vs `@reproducible`, original vision §28).

## Exact and symbolic values

Roadmap idea 9 asks for "exact rationals and symbolic constants (π, √2) kept until approximation is needed", and the design's first goal is that an optimization never changes what `sky run` prints. These two pull against each other: `1/10 + 2/10` is exactly `3/10`, and `0.1 + 0.2` is not `0.3` in floats. This section says how they fit together.

### Decision: the meaning stays f64; exactness is an analysis

**SIR means what `sky run` computes**: every scalar is an `f64`, each operation rounds once, in the order written, and an operation whose result would leave the finite numbers fails. This does not change. A `const.float` is the binary number it holds (`0.1` is `3602879701896397 / 2⁵⁵`, not `1/10`), a `const.rational` is converted with one division, and no pass may substitute a value that differs in any bit unless the program opted into a looser contract.

**Exactness is a fact about a value, not a second meaning of it.** `sky_ir::exact` analyses a function and tells each scalar apart along two axes, and keeps them apart:

| | Answers | Domain | May justify a rewrite? |
|---|---|---|---|
| **ideal value** | "which real number does this expression denote?" | a `Sym`: a rational, or a sum of at most four terms `q·1`, `q·π`, `q·e`, `q·√n` (`n` squarefree) with rational coefficients; or unknown | no |
| **range** | "which `f64`s can the interpreter produce here?" | an interval `[lo, hi]` of finite `f64`s; parameters are the whole finite range, constants are points | yes |

The split is the whole design. A fact about the ideal value is a fact about mathematics, and a rewrite justified by it is right for the reals and can be wrong for the float: `sqrt(2) * sqrt(2) - 2` is exactly `0` over the reals and `4.4e-16` in `f64`, so dividing by it is fine at run time, and treating it as zero would be wrong. A fact about the range is a fact about the number the program computes, so it is safe to build on. Passes rely only on ranges. The ideal value is used to *say* things (below) and is the fact a looser contract would use, if a program ever opts into one.

The analysis is total and never guesses. Every transfer function follows the interpreter's own rounding: `+ - * /` and `sqrt` are correctly rounded and monotone, so the range of the result is read off the corners of the operand box with the same `f64` operations, exactly; the libm functions (`exp`, `log`, `sin`, ...) and `powi` get a small relative slack (about 450 ulps for libm), so that a platform's libm cannot make a proof wrong. What it does not know it says: `None` for the ideal value, the whole finite range for the range. The soundness of every transfer function is a property test against `sky_interp`'s own arithmetic and math functions (`crates/sky-opt/src/exact_tests.rs`), and the analysis trusts the same things the other passes trust: a value of type `Number` scalar is a finite number of that dimension, and a value of type `Unknown` is not analysed at all.

### The symbolic domain

`Sym` is a sorted sum of at most `MAX_TERMS = 4` terms. Each atom is `1`, `π`, `e` or `√n`; `√8` is stored as `2√2`, so **two forms are equal as data exactly when they are equal as real numbers**. That needs the atoms to be linearly independent over the rationals, which holds for the square roots of distinct squarefree integers together with `1`, and for `π` or `e` next to them (both are transcendental). Whether `1, π, e` are jointly independent is an open problem, so a form never holds both `π` and `e`, and `π - e` is "not known".

It is closed under `+`, `-`, negation, scaling by a rational, products of algebraic forms (`√2·√3 = √6`, `(1+√2)² = 3+2√2`), division by a rational or a single `q·√n`, small integer powers, and `sqrt` of a rational. A few functions have a closed form at a few points: `exp(1) = e`, `atan(1) = π/4`, `acos(-1) = π`, `floor` and `hypot` of rationals. Everything else (`π²`, `1/π`, `sin 1`) is `None`, and `None` is never wrong, only silent. Coefficients are `i64` rationals with checked arithmetic, so overflow is `None` as well.

Nothing in SIR constructs `π` from a name yet, so it appears only through the functions above and in tests; see "`const.symbol`" below.

### What the analysis is used for

None of these change a printed digit. Each is covered by a differential property test against the reference interpreter.

1. **Discharging `Fail`.** An operation that can fail *by its kind* (`div`, `add`, `mul`, `sqrt`, `log`, ...) but provably cannot with these operands is treated as pure. `x / 2` and `x * 0.5` cannot fail for any finite `x` (the result is at most half of `x` in size); `sqrt(2)` and `log(3)` cannot; `1 / x`, `x / 0.5`, `x * 2`, `x + x` and `sqrt(x)` can, and stay. The analysis repeats the operand-dimension rules instead of trusting the verifier, because a proof of "cannot fail" is only as good as its premises. Consumers:
   * `dce` removes an unused such operation, and says so with rule **`DCE-002`** and the proof ("`div` can fail in general, but not here: the divisor is 2 (never zero) and the result is at most 9e307 in size, which fits an f64").
   * the e-graph treats such a node as unable to fail, so a rule that stops evaluating it may fire: `sqrt(3) - sqrt(3)` becomes `0` (`ALG-003` needs its operand droppable).
2. **Nonzero and positive facts for conditional rules.** A class whose range excludes zero is *nonzero*, and one whose range is above zero is *positive*, as `f64`s. New rule **`ALG-010`**, `x / x → 1`, needs nonzero (`x / x` is exactly `1.0` for every finite nonzero `x`, and fails at zero): `sqrt(5) / sqrt(5)` becomes `1`. Positivity strengthens the sign fact the real-only rules (`EXP-001`, `ALG-004`) ask for. The facts enter the e-graph as fields on the nodes made from the program's own operations, and only under the `Reproducible` contract: a `Fast` rewrite can change a value in its last bit, which could carry a razor-thin fact over its edge. The e-graph's public API is unchanged.
3. **Exact `--explain` text.** A folded float says what it is over the reals: "over the reals the result is exactly 1/2, and the f64 result is exactly that value", or "over the reals the result is exactly 1/3; the f64 result 0.3333333333333333 is that value rounded (`sky run` rounds at every step)". It is text only: what is folded in is the interpreter's number. An integer the float has exactly says nothing, and neither does a form longer than 30 characters (a `const.float` `0.1` is a rational with a 16-digit numerator, which is the truth about the program and not worth printing).

### What it is deliberately not used for

* **Deciding a rewrite from the ideal value.** See above.
* **`x - x == 0` for an unknown `x`.** That is the e-graph's job and needs no analysis (`ALG-003`); the analysis only widens what counts as droppable.
* **Tightening declared function effects.** `infer_effects` stays a function of operation kinds: a declaration is a contract the verifier checks, and a proof is a fact about operands, so a function keeps `fail` in its effects even when every failing operation in it is harmless. A later pass may tighten declarations from proofs.
* **`FoldMode::Exact`.** It stays the opt-in that keeps exact rationals at the cost of the last printed bit. It changes what is computed; the analysis never does. The analysis is what such a contract would consult to decide when the exact value may replace the float.

### `const.symbol`: not now

The operations table has long listed `const.symbol` (π, e, √2). It is **not added in this version**, for three reasons.

1. It would change no output and enable no rewrite. Its `f64` meaning is the nearest float, which is what `const.float` already holds for `pi` today, and the analysis proves everything the float supports from the float (the range of `pi` is the point `3.141592653589793`: positive, nonzero). A symbol adds a name, and a name is not a fact the optimizer may use without letting the ideal value into a decision, which the rule above forbids.
2. It would promise more than SIR means. `sin(pi)` is exactly `0` for a symbol and `1.2246e-16` for the float the program computes; an operation called `pi` invites the rewrite that turns one into the other. Until a contract says "this program means the reals", a symbol is a trap.
3. It touches every match over `OpKind` (verifier, printer, parser, interpreter, lowering, e-graph, AD) while other work is changing that enum, for no immediate benefit.

When a program can opt into a real-valued contract (an `@exact` region, or a `Repr::Exact` chosen by a precision pass), `const.symbol` arrives with it, with the ideal value as its meaning and `f64` as its approximation, and `Sym` is what it plugs into. Until then the way to put an exact `π` in front of the analysis is `acos(-1)`.

One follow-up would make `--explain` say "exactly 3/10" for a decimal literal the user wrote: lowering emitting `const.rational 3/10` for decimal literals whose numerator and denominator are exact floats. That is bit-identical by construction (`n as f64 / d as f64` is the correctly rounded decimal), but it is a lowering change with many golden diffs, and is left alone here.

## Error budgets and roundoff

ROADMAP Phase 3 lists error budgets (ideas 7, 8, 31): `error <= 1e-12` leading to a precision chosen per region. Choosing a precision needs code generation, which is Phase 4. This section is what is built before it: **the analysis** (`sky_ir::roundoff`) and **the contract** (`accuracy x within tol`, spec "Accuracy budgets"). No precision is chosen, and no program changes.

### The question

`exact` answers "which `f64`s can this value be?". `roundoff` answers "how far is that `f64` from the value the mathematics gives?": for each scalar, a range of the computed number and bounds on `|computed - exact|`, where *exact* is the same program with every operation exact and inputs and literals the reals they hold. Two bounds are kept, absolute and relative, because each is lost where the other survives (a subtraction of nearly equal numbers keeps the absolute bound and loses the relative one; a product keeps the relative bound whatever the size), and each is derived from the other whenever the range allows.

### Method

An abstract interpretation with one rule per operation (`roundoff/transfer.rs`):

* the **rounding** an operation adds is `u = 2^-53` times the result for a correctly rounded `+ - * / sqrt`, an assumed few ulps for a math function (`LIBM_ULPS = 4`; `erf`, which Sky implements, 16; `asinh` and `acosh` 8, and `atanh` and `acosh` carry an extra absolute term because Rust composes them from `ln_1p` and `ln` and they lose digits near their singularities: a property test found that), and zero for `neg`, `abs`, `min`, `max`, `floor`, ... which round nothing;
* the incoming errors are **propagated** through the operation's derivative: a Lipschitz constant over the range (absolute channel) or a condition number (relative channel);
* every bound is an inequality valid for any size of error, not a first-order expansion (`(1+p)(1+u) - 1`, not `p + u`; a mean-value bound over the whole segment between the computed and the exact operand). That is what makes the analysis *sound* and not only accurate for small errors;
* every bound is computed with each `f64` step rounded up.

When every input is a constant, an enclosure of the exact value in double-double arithmetic (`roundoff/hp.rs`, radius rigorous, about 27 digits) is carried alongside. It tightens the upper bounds to the actual error (`0.1 + 0.2` is off by exactly `2.78e-17`) and gives the one thing an upper bound cannot: a **lower bound** on the error, which is how a budget is proven *violated*. `sqrt(x^2 + 1) - x` at `x = 1e8` computes `0`; the exact value is `5.0e-9`; the error is 100%.

Control flow is followed, not approximated away: an `if` whose condition the rounding error could flip (the exact computation may take the other arm) gives **unbounded** results; a `while` is unrolled while its condition is known at every iteration and is otherwise unbounded; a call is analysed in the context of its arguments, so a budget inside a function is checked for the values the program calls it with (a never-called function is analysed for arbitrary finite inputs). Arrays and untyped values are not modelled. "Unbounded" is reported, never guessed.

### Soundness argument and tests

Each transfer rule is an inequality with a short derivation in its comment. The test that matters compares every intermediate value of random programs (all operations and every math function except `erf`) with an independent double-double reference (`sky-opt/src/roundoff_ref.rs`), with and without the enclosure, for programs of constants and for programs over parameter ranges: the interpreter's `f64` is inside the claimed absolute and relative bounds and never closer than the claimed lower bounds. Mutating a transfer formula (dropping a second-order term, halving a slope, a smaller `LIBM_ULPS`) makes it fail. It found real bugs while being written: the exponent of `x ^ (1/3)` the interpreter passes is `0.3333...` and not `1/3`; the mean-value segment must include the computed point; `cbrt` is Hölder with constant 2^(2/3) and not 1 across zero; a computed exponent that is a small integer runs `powi`; and Rust's `atanh` is inaccurate near -1. The trusted assumptions are the ones the module lists: correct rounding of `+ - * / sqrt`, the library error bound, and `powi` within `(|n| + 2)` ulps.

### The contract, the report

`accuracy y within 1e-12` (relative) or `within 10 nm` (absolute) is a statement (`StmtKind::Accuracy`). Lowering emits no operation for it: it records a `Goal` (the SSA value the name holds at that point, the function, the tolerance in SI) beside the module (`sky_lower::lower_with_accuracy`), and `sky check` analyses the module *as lowered, before any pass*, so the accuracy is that of the program as written. A pass could not be trusted to keep an operation with no result and no effect, and the goal's values would not survive rewrites. (A consequence: a value that is only mentioned in an `accuracy` statement is still dead code to `dce`.) The outcomes, proven / unproven with blame / violated, are in the spec. The blame (`roundoff/blame.rs`) is a backward pass over the values the goal depends on: operations whose result is unbounded though their operands are bounded, additions and subtractions that may cancel and lose relative accuracy, other operations that magnify relative error, and otherwise the largest rounding contributions weighted by their sensitivity. It links to the checker's cancellation warning when that reports the same expression.

### How this drives precision in Phase 4

`NumberType` already has a `Repr` slot (`Unchosen`, `F32`, `F64`, `Exact`). A precision pass will, for each region carrying a budget, ask the analysis for the error the region *would* have with a representation of unit roundoff `u_r` instead of `2^-53`: `2^-53` enters the transfer rules only through `transfer::U` and the ulp counts, so making those per-value inputs lets the analysis be run again with `f32`'s `2^-24`, or the double-double's `2^-105`. The pass then picks, per value, the cheapest `Repr` for which the budget is still proven, working backwards from the goals along the blame's sensitivities (a value whose sensitivity is tiny may drop to `f32`; the operand of a cancellation must be raised); values the analysis could not bound stay `f64`. The analysis says *where* precision is needed; the pass says how much; the goals remain the contract that is re-checked after the choice, so a wrong choice is a failed budget, never a silent loss. What is missing is the rewrite that lowers a value's `Repr`, the code generator that honours it, and that per-value `u` (a small change).

### What it is not

Not a change of meaning: `sky run` still computes `f64`, one rounding per operation. Not interval arithmetic on the inputs: an input is an exact value (the range says which, not how uncertain). Not a proof of the whole program's correctness: a budget is proven for the *runs that produce a number* (overflow and domain errors are not analysed) and under the library assumption.

## Reference semantics and testing

- **The IR interpreter is the specification.** It evaluates L1 directly. Once it exists, the tree-walking interpreter in sky-interp either becomes lowering plus the IR interpreter, or stays as a second implementation for differential testing. The Phase 1 decision is to keep both until the IR interpreter passes every run test. It does, so `sky run` executes lowered, optimized SIR by default (`--engine=auto`), and the tree-walker stays as the oracle of the differential tests (`--engine=ast`) and as the fallback for what lowering cannot express yet. The run tests in `sky-cli` compare the two engines' stdout, stderr and exit status byte for byte. One SIR limit is still worked around there, and is what to fix before the tree-walker can retire: SIR has no operation that stops a run with a message (so a `range` bound computed at run time stays on the tree-walker). An operation also carries the source span of each operand at its use (`OpData::operand_spans`), so a runtime error found on SIR has the tree-walker's labels, secondary labels and notes, and `sky run` prints SIR's diagnostic as it is.
- **Differential testing:** for every program in `tests/`, and for fuzz-generated programs, run the unoptimized IR against the optimized IR (and later each backend). They must agree within the declared error bound. This is the practical form of translation validation, and it comes long before formal proofs (idea 6).
- **Golden IR files:** `tests/ir/*.sky` with `.sir` expectations, blessed like the other goldens.
- **Fuzzing:** the verifier and the IR text parser each get a fuzz target.

## Text format

A human-readable, data-only format, round-trip exact (`parse(print(m)) == m`, after ids are renumbered into the printer's canonical pre-order). `crates/sky-ir/src/parse.rs` documents the grammar; the printer is `print.rs`.

```text
sir edition 2026

fn @main() -> () effects(io, fail) {
  %0 = const.float 10000 : num[L; ()]
  %1 = const.rational 1/1000 : num[1; ()]
  %x = mul %0, %1 : num[L; ()] @ 34..52
  print %x @ 60..68
  return
}
```

* Types are `num[<dim>; <shape>; {<tags>}; <repr>]` (tags and repr only when present; repr only when not `unchosen`), `bool`, `str`, `nothing` and `unknown`. Dimensions print as `Dimension` displays them (`L²·M·T⁻²`), shapes as `(3, N#0)` (a shape variable prints with its id, because identity is the id).
* Regions nest as `{ ... }` after the operation, with block labels and arguments (`^1(%i: num[1; ()]):`); the function body's entry block is implicit.
* Spans are byte ranges, not line:column, because the parser can only restore what does not need the source file. `0..0` is omitted.
* Floats print in the shortest round-trip form, or as `0x` plus the bits when not finite.

It is parsed by a small hand-written parser that never panics, limits nesting to `MAX_REGION_DEPTH` and input size to `sky_syntax::MAX_SOURCE_BYTES`, and has a fuzz target. It never carries executable content or serialized closures (non-negotiable 3).

## Crates

| Crate | Contents | Depends on |
|---|---|---|
| `sky-ir` | data model, operation table, verifier, effect inference, exactness analysis, text printer and parser, pass manager | sky-units, sky-types (types only), sky-ast (reads the unit text of `show`), sky-diag |
| `sky-lower` | checked AST to L1 | sky-ast, sky-types, sky-ir |
| `sky-opt` | passes: fold, dce, fuse, cholesky, and the default pipeline (which includes `sky-egraph`'s pass); later autodiff | sky-ir, sky-egraph |
| `sky-egraph` | identity-aware e-graph rewriting (`egg`) as a pass | sky-ir |
| `sky-ir-interp` | reference interpreter | sky-ir, sky-units |

## Phase 1 milestones
1. `sky-ir` data model, operation table, verifier, and text print/parse round-trip. Done.
2. `sky-lower` for every v0.1 construct; `--emit-ir`; golden IR tests. `--emit-ir` and the goldens (`tests/ir/`, run by `crates/sky-cli/tests/emit_ir.rs`) are done; lowering covers everything `sky run` executes except arrays (`LowerError`).
3. Reference interpreter; differential tests against sky-interp on every run test. Done for everything that lowers; `crates/sky-opt/tests/pipeline.rs` adds "the same output before and after the pipeline".
4. The Cholesky rewrite and DCE as passes with explanations; delete `cholesky_rewrite` from the checker. The passes and their explanations are done and run in the default pipeline. **The deletion is not done, on purpose** (see below).
5. `edition` in `sky.toml` and in SIR; the spec defines what an edition may and may not change. (`sky.toml` half done: spec "Editions"; `Edition` lives in `sky-syntax` and `sky-ir` re-exports it.)

### Milestone 4: why `cholesky_rewrite` stays for now

Deleting the checker's `cholesky_rewrite`, `RewriteSite`, `Checked::rewrites` and `Checked::cholesky_solves` needs `--explain` and `sky run` to work without them. Today neither does:

* **`sky run` reads `Checked::cholesky_solves`.** The tree-walking interpreter (and the REPL) does the Cholesky solve at exactly the sites the checker chose. The design keeps the tree-walker until the IR interpreter passes every run test; `sky run` executes SIR by default now (`--engine=auto`); the site set stays part of the tree-walker's input, which remains as the fallback and the oracle.
* **Programs with arrays do not lower.** `ok_cholesky_literal` and `ok_literal_tag_proofs` have `--explain` goldens, and SIR has no array values, so only the checker can explain them.
* **The wording needs source text.** The checker's note names the operands as written (`(K⁻¹)⁻¹ @ (f + f)`). SIR values have names, not use-site spans, so the pass can reproduce the sentence for simple names only (that is what `cholesky_text` does) and the goldens would change for compound operands.
* **The two see different sites.** The checker rewrites the syntactic `A⁻¹ @ b` only. The pass also rewrites `B = inv(A); B @ b`, which `sky run` computes with an explicit inverse. That is a behaviour difference to decide on when `sky run` moves to SIR, not one to introduce through a deletion.

What was done instead: `--explain` runs the pass pipeline and shows its explanations, the Cholesky pass's explanation joins into the checker's sentence word for word, and the two are deduplicated by span, so both sources agree wherever they overlap. What is left before the checker's rewrite can go: (1) arrays lowered to SIR; (2) `sky run` and the REPL executing SIR, so `solve[cholesky]` is what runs; (3) SIR keeping operand source spans (or explanations taking the source text from the span) so the goldens stay byte for byte. Then `cholesky_rewrite`, `RewriteSite` and `cholesky_solves` are deleted in one change, with `cholesky_text` as the only source of the note.

## Open questions
- **Build the IR ourselves or use MLIR directly?** The proposal is our own L1 and L2 in Rust (fast to build, keeps the Rust-only toolchain, Sky-specific types), with MLIR as a possible L3/L4 target in Phase 5 through `melior`. This needs a decision before milestone 1.
- **Should checking move onto SIR?** One source of truth is attractive, but AST-level diagnostics are easier to make good. Revisit after Phase 2.
- **Arrays at run time:** v1 arrays are dense and row-major (vectors and matrices, never empty), and the reference interpreter runs them with the same code as `sky run`. Are sparse formats L2 facts?
- **Symbolic shapes at run time:** specialise per concrete shape (idea 62), or compile shape-polymorphic code with checks?
