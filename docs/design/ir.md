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
- **Spans on every operation** keep diagnostics and explanations pointing at the user's source after any number of passes. A transform that merges operations joins their spans.

### Types

```text
Type = Number { dim: Dimension, shape: Shape, tags: TagSet, repr: Repr }
     | Bool             -- a scalar condition (v1 has no Bool arrays)
     | Str
     | Nothing          -- what `print` and a function without a return value give
     | Unknown          -- the checker couldn't type it; passes treat it conservatively

Repr = Unchosen         -- L1 default: a strategy pass decides
     | Exact            -- rational or symbolic until approximation is needed (idea 9)
     | F32 | F64        -- chosen by precision passes (F16, BF16, F128 come with Phase 3)
```

Functions are not first-class values in v1: calls name their callee (`call @f`). A function's parameter types are also the types of its entry block's arguments, and the verifier checks that the two agree. `if` and `while` report no effects of their own; a pass computes their effects from their regions.

`Dimension`, `Shape` and `TagSet` are the existing types from sky-units and sky-types, reused as they are. SIR depends on those crates and does not copy them. Tags stay *closed* (implications applied), as the checker already guarantees.

Planned extension points, which v1 leaves out: an error bound per value (interval or affine form, Phase 3), a distribution for uncertain values (idea 10), and a coordinate frame (original vision §12).

### Operations (L1, v1)

| Group | Operations |
|---|---|
| Constants | `const.rational`, `const.float`, `const.bool`, `const.str`, `const.symbol` (π, e, √2) |
| Arithmetic | `add`, `sub`, `mul`, `div`, `neg`, `pow` (with a constant rational exponent attribute when known), `inverse` |
| Linear algebra | `matmul`, `transpose`, `solve` (produced by rewrites, never written by users in v1) |
| Arrays | `array.literal`, `index`, `slice`, `zeros`, `identity`, `len`; `add`, `sub`, `mul` (scaling), `div`, `neg`, `inverse` and `matmul` also work on arrays |
| Comparison and logic | `cmp.{eq,ne,lt,le,gt,ge}` (on Bools and strings only `eq`/`ne`), `not`; `and`/`or` short-circuit, so they lower to `if` |
| Control | `if` (regions), `while` (regions), `yield`, `call`, `return` |
| Display | `show "unit"` (`v as km/h`: the operand written in a unit, a `str`, only ever printed) |
| Environment | `param`, `print` (effectful) |

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
6. **dce**: removes pure operations whose results are never used, with an explanation when user-written code is removed (idea 30).
7. **cholesky**: `matmul(inverse(A), b)` with PositiveDefinite A becomes `solve[cholesky](A, b)`. Its explanation is the checker's sentence for the same rewrite, word for word (`sky_opt::cholesky_text`).

The default pipeline (`sky_opt::default_pipeline`) is 4, 5, 6, 7 in that order: fold first so the e-graph sees constants; the e-graph before dce because it leaves the operations it replaced behind; cholesky last so its `solve` is never rewritten. `sky-opt`'s passes and `sky-egraph`'s rewriting each keep their own record type (`Rewrite`), and convert to `Explanation` in one place each (`Rewrite::explain`, through `PassCx::explain`), so rule ids and texts are unchanged.

### `sky check` and the passes

* `sky check --emit-ir[=<stage>]` prints the SIR text to stdout: `lower` (straight out of lowering), `after:<pass>` (or just the pass name: `fold`, `egraph`, `dce`, `cholesky`), or `final` (the default, the end of the pipeline). With the flag, stdout is only the IR, so it can be piped into a file and parsed back; the `edition:` and `ok` lines go to stderr.
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

## Reference semantics and testing

- **The IR interpreter is the specification.** It evaluates L1 directly. Once it exists, the tree-walking interpreter in sky-interp either becomes lowering plus the IR interpreter, or stays as a second implementation for differential testing. The Phase 1 decision is to keep both until the IR interpreter passes every run test, then retire the tree-walker.
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
| `sky-ir` | data model, operation table, verifier, effect inference, text printer and parser, pass manager | sky-units, sky-types (types only), sky-ast (reads the unit text of `show`), sky-diag |
| `sky-lower` | checked AST to L1 | sky-ast, sky-types, sky-ir |
| `sky-opt` | passes: fold, dce, cholesky, and the default pipeline (which includes `sky-egraph`'s pass); later autodiff | sky-ir, sky-egraph |
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

* **`sky run` reads `Checked::cholesky_solves`.** The tree-walking interpreter (and the REPL) does the Cholesky solve at exactly the sites the checker chose. The design keeps the tree-walker until the IR interpreter passes every run test; until `sky run` executes SIR, the site set is part of its input.
* **Programs with arrays do not lower.** `ok_cholesky_literal` and `ok_literal_tag_proofs` have `--explain` goldens, and SIR has no array values, so only the checker can explain them.
* **The wording needs source text.** The checker's note names the operands as written (`(K⁻¹)⁻¹ @ (f + f)`). SIR values have names, not use-site spans, so the pass can reproduce the sentence for simple names only (that is what `cholesky_text` does) and the goldens would change for compound operands.
* **The two see different sites.** The checker rewrites the syntactic `A⁻¹ @ b` only. The pass also rewrites `B = inv(A); B @ b`, which `sky run` computes with an explicit inverse. That is a behaviour difference to decide on when `sky run` moves to SIR, not one to introduce through a deletion.

What was done instead: `--explain` runs the pass pipeline and shows its explanations, the Cholesky pass's explanation joins into the checker's sentence word for word, and the two are deduplicated by span, so both sources agree wherever they overlap. What is left before the checker's rewrite can go: (1) arrays lowered to SIR; (2) `sky run` and the REPL executing SIR, so `solve[cholesky]` is what runs; (3) SIR keeping operand source spans (or explanations taking the source text from the span) so the goldens stay byte for byte. Then `cholesky_rewrite`, `RewriteSite` and `cholesky_solves` are deleted in one change, with `cholesky_text` as the only source of the note.

## Open questions
- **Build the IR ourselves or use MLIR directly?** The proposal is our own L1 and L2 in Rust (fast to build, keeps the Rust-only toolchain, Sky-specific types), with MLIR as a possible L3/L4 target in Phase 5 through `melior`. This needs a decision before milestone 1.
- **Should checking move onto SIR?** One source of truth is attractive, but AST-level diagnostics are easier to make good. Revisit after Phase 2.
- **Arrays at run time:** v1 arrays are dense and row-major (vectors and matrices, never empty), and the reference interpreter runs them with the same code as `sky run`. Are sparse formats L2 facts?
- **Symbolic shapes at run time:** specialise per concrete shape (idea 62), or compile shape-polymorphic code with checks?
