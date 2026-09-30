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
| Linear algebra | `matmul`, `transpose`; produced by rewrites, never written by users in v1: `solve[cholesky\|lu\|diagonal]` (`A⁻¹ @ b`), `solve.right[...]` (`x @ A⁻¹`), `matmul[ta,tb,alg=...]` (transposed operands, an algorithm class), `diag.mul[left\|right]` (see "Algorithm choice") |
| Vector builtins | `norm` (one operand of any shape; the result is a scalar of its dimension: the Euclidean or Frobenius length) and `dot` (two vectors of one length; a scalar of the product of the dimensions). Both have effect `fail` (a sum of squares or products can overflow) and evaluate through `sky_interp::vecfn::norm` and `dot`, the functions `sky run` calls, with the call's span and the operands' spans, so the results and diagnostics are byte-identical. `dot` is commutative; the passes treat both as opaque operations that can fail (`fold` evaluates `norm` of a constant scalar) |
| Arrays | `array.literal`, `index`, `slice`, `zeros`, `identity`, `len`; `add`, `sub`, `mul` (scaling), `div`, `neg`, `inverse` and `matmul` also work on arrays |
| Comparison and logic | `cmp.{eq,ne,lt,le,gt,ge}` (on Bools and strings only `eq`/`ne`), `not`; `and`/`or` short-circuit, so they lower to `if`; `expect.bool[and\|or\|not\|if\|while]`, the run-time Bool check of an operand the checker could not type (an unannotated parameter's result), which fails with `sky run`'s wording and on the operand's span; `expect.number`, the same for the operand of a unary `+` (which lowers to nothing else), whose result is the operand |
| Range checks | `expect.range[bound]` and `expect.range[step]`: one number (or `unknown`) operand, one dimensionless scalar result. The run-time checks of a `range` argument that is not a literal (the checker checks literals): `bound` is a dimensionless whole number of magnitude at most 2^53, `step` is not zero; each fails with the interpreter's message on the argument's span. Lowering makes the `bound` check right after each argument is evaluated, left to right, and the `step` check after the last; effect `fail`; the `fold` pass removes a check whose operand is a constant that passes |
| Tag promises | `assume`: one number operand, one result of the same dimension and shape with the tags the callee's parameter promises (`K: PositiveDefinite Matrix[N, N]`); the value is passed on as it is, nothing is checked and it cannot fail. The checker takes a parameter's tags on trust at a call and `sky run` checks them where the callee relies on them (a Cholesky solve fails with "the matrix is not positive definite"), so lowering wraps an argument that does not guarantee them (a computed matrix) in `assume` and the callee's own operation fails, with the tree-walker's message |
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
8. **powers**: `x ^ n` for a constant integer `2 <= n <= 14` becomes the multiplications `f64::powi` performs, bit for bit (`POW-CHAIN-001`); see "Algorithm choice" below.
9. **fuse**: trees of elementwise array operations become one `fused` kernel (`FUSE-001`); see "Fusion" below.
10. **cholesky**: `matmul(inverse(A), b)` with PositiveDefinite A becomes `solve[cholesky](A, b)`. Its explanation is the checker's sentence for the same rewrite, word for word (`sky_opt::cholesky_text`).
11. **transposes**: `matmul(transpose(A), B)` becomes `matmul[ta](A, B)` (`LA-TRANSPOSE-001`), exact.
12. **structure**: Gram products (`LA-GRAM-001`), Diagonal scaling (`LA-DIAGONAL-001`) and the transpose of a Symmetric matrix (`LA-SYMMETRIC-001`), all exact.
13. **algorithms**: records the algorithm class a policy chooses for each `matmul` (`ALG-MATMUL-001`); the default policy chooses nothing.
14. **specialize**: a call whose arguments have facts the callee's signature lacks (a concrete shape for `Vector[N]`, a constant, a proven sign) gets a clone of the callee that knows them, simplified; see "Specialization" below.
15. **gvn**: global value numbering, CSE across blocks and regions: an operation identical to one that dominates it (same kind, attributes, operands and result types; commutative operands ordered) reuses the earlier result (`GVN-001`). Scoping is dominance (an `if` arm's values are not visible after the `if`, a `while`'s `before` values not in its `after`). Never for `io`, and a call only if the callee cannot print. An operation that can fail is reused too: the earlier one has failed already if this one would, and Sky reports the first failure, so the later one's span is never the reported one.
16. **licm**: loop-invariant code motion out of `while` bodies (`LICM-001`, `LICM-002`). An invariant operation that cannot fail (by kind, or proved by `sky_ir::exact`) moves to the pre-header. One that can fail moves only where the loop reaches it before anything observable: from the `before` region behind only clean operations, or from the `after` region when the loop's first test is known true; otherwise, when the `before` region is straight-line and clean, into a guard (`if <first test> { hoisted; while ... } else { forwarded values }`) so a loop that runs zero times never fails. Never a call or anything with `io`. See `sky_opt::licm` for the argument.

Alongside the passes, `sky_ir::exact` (see "Exact and symbolic values") is an *analysis*: it changes nothing, so it is not in the pipeline, and each pass that wants it (`fold` for explanations, `dce`, the e-graph) computes it again rather than caching it across a change.

The default pipeline (`sky_opt::default_pipeline`) is 14, then 4, 5, 15, 16, then 6 to 13 in that order. Specialize first, because it reads the program as lowered (its constants are `const` operations or provable by the exactness analysis) and hands every later pass functions that already know their constants and shapes (and, since `contracts` reasons from the calls of a function, it now does so per clone); fold next so the e-graph sees constants; contracts after the e-graph (it sees the simplified conditions) and before dce (which removes the comparison a dropped check leaves unused); the e-graph before dce because it leaves the operations it replaced behind; dce before fuse because a dead operation left behind counts as a second use of its operands and would keep a temporary alive; fuse after the e-graph because a `fused` body is opaque to fold and the e-graph (they would otherwise rewrite it as scalar code, and `neg` differs, below); cholesky last so its `solve` is never rewritten (fuse and cholesky touch different operations, so their relative order does not matter). `sky-opt`'s passes and `sky-egraph`'s rewriting each keep their own record type (`Rewrite`), and convert to `Explanation` in one place each (`Rewrite::explain`, through `PassCx::explain`), so rule ids and texts are unchanged.

Within 4 to 13 (and 15, 16): fold first so the e-graph sees constants; gvn after the e-graph, which makes equal expressions syntactically equal, and before licm, so an operation equal to an outer one is reused instead of hoisted under a guard; licm after both and before contracts, dce, powers and fuse: fuse then sees the operations that stay in the loop (a partly invariant chain would otherwise fuse into a kernel that recomputes its invariant part), and specialize has already unrolled the loops whose trip count is constant; contracts after the e-graph (it sees the simplified conditions) and before dce (which removes the comparison a dropped check leaves unused); the e-graph before dce because it leaves the operations it replaced behind; dce before fuse because a dead operation left behind counts as a second use of its operands and would keep a temporary alive; fuse after the e-graph because a `fused` body is opaque to fold and the e-graph (they would otherwise rewrite it as scalar code, and `neg` differs, below); powers before fuse so its multiplications can fuse into a kernel (`pow.const` does not) and after the e-graph, which knows `pow.const`; cholesky after fuse so its `solve` is never rewritten (fuse and cholesky touch different operations, so their relative order does not matter); transposes before structure, because it is what turns `Aᵀ @ A` into a product of one value by its own transpose; algorithms last, so a policy sees the operations the exact rewrites left. `sky-opt`'s passes and `sky-egraph`'s rewriting each keep their own record type (`Rewrite`), and convert to `Explanation` in one place each (`Rewrite::explain`, through `PassCx::explain`), so rule ids and texts are unchanged.

### Algorithm choice

Phase 4 wants Sky to beat hand-written code by picking better algorithms that only Sky can prove valid, because it knows shapes, dimensions and matrix tags. The spec's "Algorithm choice" section is the normative text; this is how it is built.

**The operations.** All are L1, produced by rewrites, never written by users:

| Operation | Meaning | Exact? |
|---|---|---|
| `solve[cholesky\|lu\|diagonal] A, b` | `A⁻¹ @ b`, `A⁻¹` never formed. `b` is a vector or `n` rows of right-hand sides; the result has `b`'s shape and the dimension of `b` over `A`'s. `lu` is Gaussian elimination with partial pivoting whose forward phase is the Gauss-Jordan `inverse`'s, operation for operation, so singular means the same thing; `diagonal` is `bᵢ / aᵢᵢ`; `cholesky` needs PositiveDefinite. The tag of `cholesky` and `diagonal` is a promise the interpreter verifies. | no: differs from `inverse` then `matmul` in the last bits, at least as accurate (spec) |
| `solve.right[...] x, A` | `x @ A⁻¹`, the `y` with `y @ A = x`, applied to `A`'s factors without forming `Aᵀ` | no, likewise |
| `matmul[ta,tb,alg=...] A, B` | `matmul` with the flagged matrix operand transposed **without materializing the transpose**, and an algorithm class (`auto`, `dot`, `matvec`, `vecmat`, `gemm`, `syrk`). The reference interpreter transposes with the function `transpose` calls and multiplies with the one `matmul` calls, so it is *defined* as `matmul(transpose(A), B)`. This is the contract a fast kernel implements. A flag is only legal on a rank-2 operand; a `matmul[]` with no attribute is plain `matmul`. | yes |
| `matmul[alg=syrk] A, A` (with `ta` xor `tb`) | a Gram product: one triangle is computed and mirrored; `aₚᵢ·aₚⱼ` equals `aₚⱼ·aₚᵢ` bit for bit | yes |
| `diag.mul[left\|right] D, X` / `X, D` | `D @ X` or `X @ D` as a row or column scaling, `0.0 + dᵢ·xᵢⱼ` (the other terms of `matmul`'s sum are `±0` added to a sum started at `+0.0`; the `0.0 +` reproduces the sign of a zero). O(n²). The Diagonal tag is verified. | yes |

Every `alg=` names a shape class or a structure, and **must give the reference result bit for bit** (`k` accumulated in index order from `+0.0`): that is what lets kernels and tuning plug in per class without changing an output digit. A kernel that reorders the sum is a rewrite, not a variant.

`SolveMethod`, `Side` and `Algorithm` are defined in `sky_types::algo`, below the interpreters, and re-exported by `sky_ir`. The numerics and the value layer (shapes, dimensions, errors) are `sky_interp::solve`, called by both the SIR interpreter and the tree-walker.

**The chain planner lives in the checker, not in a pass.** `sky_types::chain::plan_chain` takes the factors of an unparenthesized `@` chain (shape, dimension, closed tags, whether a factor is `A⁻¹`) and returns a tree: the association by the matrix-chain dynamic program over a flop model (`sky_types::flops`: polynomials in the shape variables, compared for every size, so symbolic shapes are only reordered when provably no worse), with each `A⁻¹` factor next to a value turned into a solve by tag (Orthogonal: `Aᵀ`; Diagonal: division; PositiveDefinite: Cholesky; else LU), `A⁻¹ @ B⁻¹` into `(B @ A)⁻¹` where cheaper. It cannot be a SIR pass for two reasons. First, **parentheses exist only in the syntax tree**: `(A @ B) @ C` and `A @ B @ C` lower to the same SIR, and a pass could not tell an association the user forced from one it may change. Second, **the reassociation and the solves change rounding, so the tree-walker must make the same choice**, or `--engine=ast` and `--engine=sir` would disagree in the last bits. So the checker records the plan under the span of the whole expression (`Checked::chains`, like `Checked::cholesky_solves`), the tree-walker evaluates it (`sky_interp`'s `chain_exec`) and lowering emits it (`sky_lower`'s `chain`), both in the order the module documentation of `chain` spells out: each factor once, left to right, then the tree, children first. The consequence is that these rewrites fire only where the checker sees the chain written out; `M = inv(A)` then `M @ b` is left as it is (the older `cholesky` pass remains the one exception, for PositiveDefinite). A nested chain's plan is withdrawn when the enclosing chain is planned, and a plain PositiveDefinite `A⁻¹ @ b` stays with the older per-site Cholesky rewrite so its explanation is unchanged.

**The exact rewrites are SIR passes** (`sky-opt`), because they do not change a bit and the engines need not follow them: transpose elimination (`matmul(transpose(A), B)` to `matmul[ta]`, rule `LA-TRANSPOSE-001`; `transpose(S)` of a Symmetric `S` to `S`, `LA-SYMMETRIC-001`), the structure passes (Diagonal `diag.mul`, `LA-DIAGONAL-001`; Gram `syrk`, `LA-GRAM-001`) and the power chain (`x ^ n`, `2 ≤ n ≤ 14`, as the multiplications `f64::powi` performs, `POW-CHAIN-001`; the binary method is optimal in multiplication count up to 14 and first suboptimal at 15, where the optimal tree would round differently from `powi`, so larger and negative exponents stay `pow`). A policy hook (`sky_opt::MatMulPolicy`, run by the `algorithms` pass) chooses the `alg=` of a `matmul` from its shape class; `syrk` is chosen by `structure`, where the operand being one value is visible, and the default policy leaves everything else `auto`. Kernel and tuning code register their own policy. Pass order is documented in `sky_opt::pipeline`.

**Explanations.** The checker's chain notes (`LA-CHAIN-001`) are in `Checked::chain_notes` and print with `sky check --explain`, with the estimated flops before and after; the exact rewrites explain through `PassCx` as every pass does.

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

**Native code for a whole module.** `sky-jit` (`module.rs`) also compiles every function of a module to Cranelift IR and calls `<main>` (`sky run --engine=native`; `sky check --emit=clif` prints the IR). It is the second reference-checked backend: the same stdout and the same diagnostics as this interpreter, tested against it by the differential tests and by the fuzz oracle on the module as lowered and after the default pipeline. A scalar is an `f64` (its dimension is the type's), a Bool an `i8`, a *small array* (a vector or matrix of a shape known at compile time with at most 16 entries) is its entries as separate `f64`s with no handle and no runtime call, and everything else, including a value of type `unknown`, a handle to a value the runtime owns; a value of unknown type that turns out not to be what a declaration says stops native code and the run continues on this interpreter (see the spec, "Native code").

**Small arrays in native code.** `array.literal`, `add`, `sub`, `neg`, scaling, `index` (a run-time position gets the interpreter's checks: whole, non-negative, in range), `dot`, `norm`, `matmul` and `matmul[ta,tb]`, `transpose`, `fused`, `zeros`, `identity` and `len` on small arrays are inline scalar code (`sky-jit`, `module/small.rs`), each entry computed in the order of the shared kernels (naive sums for products; fewer than 8 terms left to right, else 8 interleaved partial sums for `dot`, `@` of vectors and the squares of `norm`; no fused multiply-add). A failure (a non-finite entry, a zero divisor, a bad index) is detected inline and the whole operation is then run by the runtime for the interpreter's exact diagnostic; a `fused` body runs one operation at a time over all entries, which is the rule "the earliest failing operation in body order, at its first failing entry". **ABI:** a small array crosses function calls, `if` results and `while` loop values as its `N` entries, row-major, as `N` parameters, results or block parameters; a function with more than one result value returns them through a caller-provided result area. It becomes a handle only where a handle is needed (`print`, `assert` names, runtime-only operations such as solves, slots of type `unknown`), and a handle flowing into a small-array slot is unpacked with a shape check that deoptimizes on a mismatch. The limit of 16 entries covers 2- to 4-vectors and matrices up to 4 by 4.

**Native kernels.** For long arrays the SIR interpreter compiles the body once (`sky_jit::kernel`, Cranelift) and runs it as an unrolled 128-bit vector loop, when the interpreter has spent about the compile time (512 elements) on that `fused` in the run; `SKY_JIT=off` and `SKY_JIT_MIN_ELEMENTS=<n>` turn it off or move the threshold. The bytes and the error are the interpreter's: arithmetic is plain IEEE (never contracted to a fused multiply-add), `neg` is `0 - x`, math functions call `sky_interp::math::apply`, and the loop only *detects* a failure, by accumulating a NaN for every non-finite value that a divisor or a math call could hide, plus the result. Only then does the kernel run the interpreter's own algorithm (the rule above) to produce the exact diagnostic, so "the first failing operation in body order" is defined by one piece of code. A body the kernel does not implement, or whose dimensions do not agree, is refused and stays on the interpreter. Proved by a differential property test (native against interpreted on random bodies and overflowing data, in `sky-ir-interp`), and against the reference path in `sky-jit`; measured by `cargo bench -p sky-jit --bench kernels`.

**What is fused.** A candidate is `add`, `sub`, `mul`, `div`, `neg` or a math function (`math.sqrt`, `math.exp`, `math.atan2`, ...) on arrays of known type, laid out as the checker allows (two arrays of one shape for `add`/`sub`; two arrays of one shape, or an array and a scalar, for `mul`; two arrays of one shape, or an array over a scalar, for `div`; for a math function, arrays of one shape and scalars, at least one an array, each scalar broadcast). So `sqrt(x*x + y*y) * c + d` over vectors is one kernel of five operations. A math function of arrays is the scalar function applied to each entry (`sky_interp::math::apply`, whose core `apply_f64` a body operation calls), so a fused body and the unfused array operation give the same bits and the same diagnostic, and the failure rule above covers them: a domain error is an operation that can fail like `div`. A candidate is absorbed into its consumer when its result has exactly one use, which is an operand of a candidate in the same block. A *tree* is a candidate with everything absorbed into it, and a tree of two or more operations becomes one `fused` at the root's position. Scalar operands stay outside as operands. The kernel keeps the root's result value (name and users) and the join of the operations' spans, and the operations move into the body with their own spans and names.

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

### Specialization

A generic function is compiled once for every caller. `def dot(a: Vector[N], b: Vector[N])` loops over `len(a)`, checks every index, and cannot know that all its callers pass 3-vectors. The `specialize` pass (`sky_opt::specialize`, ROADMAP Phase 4, "runtime specialisation on shapes and values") gives such a call a clone of the callee that knows what the call proves.

**Facts.** For each `call`, per parameter the callee reads:

* *shape*: the parameter has a size variable (`Vector[N]`) and the argument a concrete size: `N = 3`;
* *value*: a scalar or Bool parameter and an argument that is a constant: a literal, or a number the exactness analysis proves is one `f64` (`2 * 3`; a proven zero is not used, since the range cannot tell `+0` from `-0`);
* *sign*: a scalar parameter and an argument proven strictly positive or negative. Kept only if it decides something in the clone (a comparison, or a check the analysis then proves), otherwise the clone would be a copy.

* *type*: a parameter declared `unknown` (unannotated) and an argument whose type is known exactly (a number of a dimension and a concrete shape of at most two axes, a Bool, a string; no tags, default representation): the clone's parameter has that type. See "Types" below.

An `unknown` parameter gets no constant or sign fact: its operations check their operands at run time, and a constant of a more precise type could change which check fails. Facts are trusted as far as the types they come from: a `Vector[3]` argument has three entries, which lowering already assumes when it writes a constant for `len`.

**The clone** is a copy of the callee named `f[N=3]`, `f[x=2]`, `f[N=3,x>0]` (deterministic: sizes by variable, then parameters in order), with the sizes substituted in every type, constant parameters removed from the signature and replaced by a constant in the body (the call passes fewer arguments), and the body simplified: `len` of a static shape becomes its constant, comparisons a sign decides become Bools, `[a, b, c][1]` becomes `b`, constants fold (exactly, `fold`), dead code goes (`dce`), and `while` loops whose trip count is then a small constant are **unrolled**. Every step keeps the operations' kinds, spans and operand spans, and only removes or replaces an operation when it computes what the interpreter would, or provably cannot fail, so outputs, errors and diagnostics are identical (the property test runs random programs through the pipeline with and without the pass). The call depth is unchanged, and "recursion is too deep" names `f`, not `f[N=3]` (`sky_ir::source_name`).

**Types.** An unannotated parameter is `unknown` in SIR, so every operation on it is a run-time check (is it a number? of this dimension? a Bool?) and the native engine runs it on boxed values: `fib(30)` took 2.5 s natively, slower than the tree-walker. A call that passes values of known types gets a clone (`fib[n:Scalar]`, `sq[x:Scalar(length)]`, `total[v:Vector[10]]`, `same[x:Bool]`) whose parameters have those types, and `retype` (in `specialize`, and `sky_ir::infer` for the result-type rules) types the body again from them: every value that was `unknown` gets the type the verifier's rules give it (`n - 1` a number, `n < 2` a Bool), the `expect.bool` and `expect.number` checks that are now satisfied disappear, and the clone's declared result is the type of what it returns. The rules:

* *Validity.* The clone is for calls whose argument types are known *exactly*; the generic function stays for the others. The verifier is the judge of the clone: an operation whose operands its rules reject (`n < 2` for a Bool `n`, `n + 1 m` for a dimensionless `n`) is a run-time error in the generic function, so the clone is not made and the call that reaches it keeps the generic function and its error. Only checks that the argument's type already answers are removed; every operation that can fail (division, overflow, indexing) is kept, so errors and their spans are the generic function's. A clone that removes no run-time type check and passes no known type on to a call is dropped.
* *Dimensions.* One clone per exact type, dimension included: `fib(3)` and `fib(3 m)` are different clones (`fib[n:Scalar]`, `fib[n:Scalar(length)]`; the second is not valid for `fib`, whose `n < 2` compares with a dimensionless number). A dimension-polymorphic clone is not made: a scalar's dimension is part of the type in native code, and every operation's dimension rule is decided per dimension.
* *Deduplication and recursion.* The name is the identity, so a recursive call (`fib(n - 1)` inside `fib[n:Scalar]` passes a `Scalar`) and every other caller with the same argument types find the clone by name, and recursion ends. The per-function and per-module limits apply, so a function called with many types keeps the generic body for the rest.
* *Early returns.* Lowering compiles an early `return` to a flag and a returned value that is a stand-in (`nothing`) when the flag is false. That stand-in is never read, and a value that is a number on one path and `nothing` on the other would make everything after it `unknown`. `retype` finds the stand-ins by their uses: a value an `if` yields next to a flag that is a different constant in each arm is *dead* in one arm when every use of the result is in a region where the flag is known (by the conditions of the `if`s around the use, through `not`) to be the other arm's constant, or feeds a dead slot. It types the result as the live arm's and writes a stand-in of that type (`0`, `false`, `""`) into the dead arm. A function that can fall off its end (a path with no `return`) has no such proof and returns `unknown`.
* *Results and mutual recursion.* A call's result type is the callee's declared result; for the clones being made together it is iterated from "no value yet" to a fixed point, so `fib` (`n` or a sum of two `fib`s) is a number and `is_even`/`is_odd` are Bools. A clone that does not verify with the settled results is left as its first typing made it, with an `unknown` result. Callers outside the clones keep their `unknown` call results (a result that turned out typed is boxed on return); only the clones are retyped.

**Unrolling.** A loop is unrolled when running it symbolically, by inlining its `before` and `after` regions and folding, decides its condition every time: true on at most 8 checks, then false. The trial is a copy of the function; if any check is not decided, or the copies would add more than 256 operations, the loop stays. The unrolled code is the loop's operations in the order the loop runs them, each with its own span, so a failure in iteration 3 is the same failure. Loops nested in one another are tried innermost first, and again in each copy of the outer body.

**Checks removed** is what the explanation counts: operations that can fail (division, overflow, `index`, contracts, ...) that the clone no longer has, or that the exactness analysis now proves cannot fail, against the generic function, counted once per source operation (an unrolled body has the checks of one copy). The analysis gained `Proof::InBounds` for this: `index` and `slice` with constant positions inside a static shape, and `len` of a typed array (which also gets a value when its first axis has a static length).

**Limits**, all documented on the module and tested: at most 8 clones per function (counting every fact set), bodies of at most 300 operations, at most 64 clones and 4096 cloned operations per module, unrolling at most 8 iterations and 256 added operations per clone. A call past a limit keeps calling the generic function, and the first says so (`SPEC-004`). **Recursion terminates**: a clone's own calls are specialized in turn, a recursive call with the facts of the clone it is in finds itself by name (`g[N=3]` calls `g[N=3]`), and recursion on a constant (`fact(n - 1)` from `fact(5)`) makes one clone per value until the base case or the limit, past which the deepest clone calls the generic function, as before. A generic function that no call reaches any more (and that was called before) is removed, so the program does not carry both.

**Explanations** (rule ids `SPEC-001` to `-005`): `[SPEC-001] specialized `dot` for N = 3 at this call: 2 checks removed`, with the facts and what they simplified as reasons; `SPEC-002` a call reuses an existing clone; `SPEC-003` a loop of a clone was unrolled; `SPEC-004` a limit stopped a specialization; `SPEC-005` a clone with a type fact: `[SPEC-005] specialized `fib` for n: Scalar (dimensionless): 4 run-time type checks removed`, where the checks are the operations of the generic function that had to find out at run time what a value is.

**Run-time hook.** The pass uses what a call *proves*. An engine that learns the argument values while it runs (the native engine, compiling for the first call) calls `specialize_for(module, "f", &[Some(value), None, ...])`, which returns the module plus the clone (named by `specialized_name`, with the signature unchanged so it is called exactly as `f` is) and the specializations of the calls inside it. The clone computes what `f` computes only for arguments equal to those values (bitwise for numbers, equal shapes for arrays): the guard, and the fallback to `f`, are the engine's. `SpecializationCache` keeps the results for one module, keyed by function and facts, at most 64.

### Lifetimes and buffer reuse

Arrays are immutable values, so an executor that gives every array operation a new buffer allocates and copies for nothing whenever the operand is never read again. `sky_ir::lifetime` finds those places. It is an *analysis* (it changes no IR): `Lifetimes::analyze(&Function)` says, for every operand of every operation, whether it **dies** there, that is, is not used again on any execution path.

* It is backward liveness over the structured regions. In straight-line code the last use dies. In an `if`, each arm is followed by what follows the `if`, so a value used in one arm and nowhere after dies in that arm, and one used after the `if` dies in neither. A `while` runs its regions any number of times: a value defined *outside* the loop and read inside it never dies inside it, an initial loop-carried operand dies at the `while` only if the loop does not read it, and the loop-carried block arguments die at `condition` and `yield`, which is what hands a buffer from one iteration to the next. A call argument dies at the call if the caller is done with it, and the callee owns its parameters.
* Where it is unsure it says no: the same value twice in one operation (`a + a`) dies at neither use; a function with an id out of range, regions that contain themselves, or nesting past `MAX_REGION_DEPTH` gets the empty answer (nothing dies); a function with no array-typed (or `Unknown`) value is not walked at all.
* `Lifetimes::reuse(op)` names, for an elementwise operation (`add`, `sub`, `mul`, `div` by a scalar, `neg`, `fused`), the operand whose buffer can hold the result: it dies there and has the result's shape and element representation. This is on the types only; the interpreter also reuses at run time when it sees the shapes.

The SIR interpreter uses it directly. `Value::Array` owns its `Vec`, so no value is ever shared: an operand is either cloned or **moved** out of its slot, and the analysis says which. An owned array of the result's shape holds the result (`sky_interp::array::arith_owned`), also for a `fused` kernel (element `i` is read from every operand before element `i` of the result is written), and a value that dies is dropped right after its operation instead of at the end of the function. A slot read after a move reports `used before it is defined`, so a mistake in the analysis fails loudly instead of printing a wrong number. The rest of the interpreter avoids per-operation allocation too (results are defined without a temporary `Vec`, the values a region hands out travel in a reused buffer, a `fused` body runs in a reused value table), so the time-stepping loop `v = v + dt * a` runs with **no allocation per iteration** in steady state (`crates/sky-bench/tests/allocations.rs` counts them with a counting allocator, in a test binary only).

A `fused` body made only of `add`, `sub`, `mul`, `div`, `neg` and constants is compiled by the interpreter (`sky-ir-interp/src/kernel.rs`) to steps on `f64` registers. The first element runs the ordinary way, which checks the dimensions and finds the result's; if it succeeds, the other elements run on the registers, with each failure built by the same code (`arith::overflow`, `div_zero`, the same spans) and the "earliest failing operation in body order" rule applied to the step index. Any other body runs the ordinary way throughout.

How this meets the native and parallel paths of `fused`: the order is the native kernel (`native.rs`, once it is compiled), then parallel chunks for a big kernel, then the scalar kernel and the interpreted loop, which write into a dying operand. The compiled loop may not overlap its input, so the native path builds its result in the buffer of an array that died earlier (the interpreter keeps a few), and a loop that kills as many arrays as it makes still allocates nothing; on the N-body loop (32 elements) it is 3.3 times faster than the scalar kernel, which is why it comes first. Nothing is moved while a call group runs (`sky_ir::parallel`): its operations run out of program order, so a deferred operation could read what an in-place one had moved. The activation is frozen for the group and the operands that die in it are dropped when it is done. The machines of a parallel run are made inside the thread that uses them and share only the (plain data) analysis.

What it does not do, and why:

* **Views.** `transpose` followed by `@` could read strided, and a `slice` that is only read could be a view. `Value::Array` has no strides or offsets, and adding them touches every consumer of `Value` (and the kernels, which another change owns), so both still copy. The copy is O(size), against O(size·k) for the `@`.
* **Calls** allocate their argument list and frame, and a call in a loop is not allocation-free.
* Arrays are still two allocations (entries and shape), for a result that has no buffer to reuse.

The tree-walker has no analysis; it applies the same idea where it is cheap. A binary operation owns its operands (temporaries or clones), so an array among them holds the result. And in `x = x + dt * v` (also `x += ...` and a binding) the value is *moved* out of `x` at its read instead of cloned, when the right side reads `x` exactly once and can run no user code (a call of a user function could read `x` as a module-level name, so any call but a math function disqualifies it, as do `and`, `or` and conditional expressions, which read conditionally).

`sky check --explain` says where a buffer is reused, one `MEM-001` note per site, quoting the expression the user wrote (a fused kernel by the expression it yields, not by the statements it merged): "`v + dt * a` reuses `v`'s buffer: no allocation in this loop".

### `sky check` and the passes

* `sky check --emit-ir[=<stage>]` prints the SIR text to stdout: `lower` (straight out of lowering), `after:<pass>` (or just the pass name: `specialize`, `fold`, `egraph`, `gvn`, `licm`, `contracts`, `dce`, `fuse`, `cholesky`), or `final` (the default, the end of the pipeline). With the flag, stdout is only the IR, so it can be piped into a file and parsed back; the `edition:` and `ok` lines go to stderr.
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

## The cost model

`sky_ir::cost` gives every function a **cost**: flops, bytes read, bytes written, allocations and bytes allocated, each an `Amount`: a lower and an upper bound (`lo <= actual <= hi`, for every run and every size from 1 up) as polynomials in the symbolic shape sizes (`sky_types::poly::Poly`: exact rational coefficients over monomials in named sizes, capped in terms and degree, every operation checked, `None` on overflow). It is a static analysis over L1, one rule per operation (`cost/ops.rs`, with the weights in the module documentation and the spec's "Performance contracts"), computed on the module after the default pipeline, since the contract is about the program that runs (fusion removes allocations, the Cholesky rewrite changes a solve's flops).

- **Straight-line code is exact** (`lo == hi`). An `if` joins its branches: the greater `hi`, the lesser `lo`, using the sound polynomial order (`x = 1 + y`, all coefficients non-negative), and where two polynomials are not ordered, the coefficient-wise max or min, which is above (below) both wherever sizes are non-negative.
- **Loops** multiply the body by a `Trips` interval. A `while` is counted when its condition is `i < stop` (or `<=`, `>`, `>=`) on a carried index that the body advances by a constant (`add %i, c` yielded at the same position, which is how `for i in range(...)` lowers) and `start` and `stop` are sizes the model can name (constants, `len` of an array, sums and products of those). The count is exact for a step of 1 and `[d/c, d/c + 1]` for another step. A condition of the form `if not broke { i < stop } else { false }` (a `break`) gives an upper bound only. The index's own step is left out of the flops. Any other loop makes the amount unbounded, with a `Reason` carrying the loop's span.
- **Calls** substitute the callee's size variables by the polynomials of the argument shapes (a callee size the arguments do not fix, or a cycle in the call graph, makes the amount unbounded), in an order where callees come first, found without recursion over the call graph.
- **Sites**: every costly operation, call and loop keeps its contribution (loops around it included), which is how `sky check` says which operation dominates a refuted contract and where the hot spots are.

`sky check` (`sky-cli/src/cost.rs`) proves a `cost` line when `hi` is at most the bound for all sizes from 1 up (sound, not complete), refutes it with a witness point where `lo` exceeds the bound in exact arithmetic, and otherwise warns. The unit of cost is the model's, not a machine's: it counts what the reference implementation does. Making a machine-specific estimate (idea 87) means replacing the weights with a hardware description, not changing the analysis.

## Automatic parallelism

The SIR interpreter runs some work on several threads without changing anything a program can observe: the same stdout bytes, the same error (message, spans, labels, notes) and the same exit status on 1 thread and on 64. The analysis is `sky_ir::parallel`; the machinery is in `sky-ir-interp` (`rayon`, one shared pool per thread count). The tree-walker stays sequential, so `--threads` matters for `--engine=sir` and `--engine=auto`.

### The rule for errors

Sky's behaviour on failure is "the first failure in program order wins, after everything printed before it". Work may run concurrently only if it has no `io` and one of two things holds:

* it **cannot fail** (an elementwise kernel whose operations the exactness analysis proves in range; most kernels cannot be proven, so this is the rarer case), or
* its failures are **resolved deterministically after the fact**: run everything in parallel, and if anything failed, report the failure the sequential order would have met first. Across elements that is the lowest index; across operations, the earliest in program order.

Anything that prints stays where it is: a `print` after a call that might fail must not run before the call is known to have succeeded, so an operation with `io` ends a window of concurrent calls, and `print` never runs on a worker.

### What runs in parallel

* **`fused` kernels** with at least `fused_work` element-operations (elements times body operations). Elements are cut into contiguous chunks, each a function of the element count and the thread count only. Each chunk runs the same loop as the sequential interpreter on a value table of its own and ends with its data and its failure, if any, as *(index of the failing body operation, error at the lowest failing element)*. The kernel's error is the smallest operation index, and among equal indices the lowest chunk: exactly the element the sequential loop meets first (`fused` reports the first failing operation in body order; see "Fusion"). Chunking cannot change a value: elements are independent and each is the same `f64` computation.
* **Elementwise array operations** (`add`, `sub`, `mul`, `div`, `neg` on arrays, `mul` and `div` of two arrays of one shape included; a `div` with a zero entry in the divisor goes through the sequential code, which reports the lowest failing entry) with at least `array_elements` entries. A lone math function of an array is not split (a chain that contains one is a `fused` kernel, which is). The only failure left is one overflow for the whole operation, which is found after the entries are computed and is the same error whichever thread found it. Cases that fail *before* the entries are computed (mismatched shapes or dimensions, a zero divisor) go through the sequential code, which produces the error.
* **Independent pure calls.** A window of operations in one block that starts at a call to a function with no `io` (transitively), no recursion and enough estimated cost. The expensive calls that do not read the result of another are started on the pool, the operations between them that do not wait for a result run in place, and the operations that do (`total + f(a)`) are deferred until the calls are done. When the started calls are done the window is replayed in program order (a call gives its result or its error, a deferred operation runs), so the first failure in program order is the error, and a failing in-place operation is reported only if nothing before it failed. A callee runs on a worker with the caller's call depth; its output goes nowhere because it has none.

Loops are not parallelized: the values a `while` carries from one iteration to the next make its iterations a chain (the work inside an iteration is what the three kinds above cover).

### The stack

The one resource a worker changes is the stack: a worker starts on an empty one, and `sky run` stops with "the program nests too deeply" at a byte budget. A callee is started elsewhere only if its whole call tree is a shallow chain of calls and regions with no recursion (`MAX_STACK_WEIGHT`), and only while the caller has used at most half of the budget. A run that would have hit the limit therefore hits it whether or not a call was moved.

### Thresholds

Measured with `cargo run --release -p sky-bench -- par` (see the "Benchmarks" section of the README); the defaults live in `sky_ir::parallel::Config`:

| threshold | default | what it is |
|---|---|---|
| `fused_work` | 4096 element-operations | a 4-operation kernel is split from 1024 elements; 4 threads first win at 1024 to 2048 elements, and are 3.1x faster from 10^4 |
| `array_elements` | 262144 | a plain `add` of two arrays is memory-bound: 4 threads first win at about 131072 entries (about 2x), and reach about 2.5x on millions |
| `call_cost` | 400 | estimated interpreted operations (a loop counts 100 iterations) a callee must cost; starting a group costs about 13 microseconds, which is about 85 operations at the interpreter's price per operation, so 400 keeps the overhead under a quarter of a call |
| `chunk_min` | 512 | fewest elements in a chunk; at most 4 chunks per thread |

`Config::eager` sets every threshold to its minimum; the determinism tests use it so that small programs exercise every parallel path.

### Controls

`sky run --threads=N` and `sky check --threads=N` (or `SKY_THREADS=N`; the default is the number of cores) set the number of threads; 1 is sequential, exactly the interpreter without this analysis. `sky check --explain` says what would run in parallel and, for what would not, why:

```text
note: [PAR-001] this elementwise kernel over 1e6 elements runs on 8 threads
note: [PAR-002] these 2 calls are independent (pure): run concurrently on up to 8 threads
note: [PAR-003] this elementwise kernel over 3 elements is not parallelized: it may fail and is small
note: [PAR-003] the call to `noisy` is not run concurrently with the calls after it
```

The notes depend on the thread count only through the number they print; what is parallel does not.

## Reference semantics and testing

- **The IR interpreter is the specification.** It evaluates L1 directly. Once it exists, the tree-walking interpreter in sky-interp either becomes lowering plus the IR interpreter, or stays as a second implementation for differential testing. The Phase 1 decision is to keep both until the IR interpreter passes every run test. It does, so `sky run` executes lowered, optimized SIR by default (`--engine=auto`), and the tree-walker stays as the oracle of the differential tests (`--engine=ast`) and as the fallback for what lowering cannot express yet. The run tests in `sky-cli` compare the two engines' stdout, stderr and exit status byte for byte. A `range` bound or step computed at run time is checked by `expect.range[bound|step]` (a whole number of sane size, a step that is not zero) with the interpreter's own errors. An operation also carries the source span of each operand at its use (`OpData::operand_spans`), so a runtime error found on SIR has the tree-walker's labels, secondary labels and notes, and `sky run` prints SIR's diagnostic as it is.
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
| `sky-ir` | data model, operation table, verifier, effect inference, exactness analysis, parallelism analysis, roundoff analysis, static cost model, text printer and parser, pass manager | sky-units, sky-types (types only), sky-ast (reads the unit text of `show`), sky-diag |
| `sky-lower` | checked AST to L1 | sky-ast, sky-types, sky-ir |
| `sky-opt` | passes: fold, dce, gvn, licm, fuse, cholesky, the exact algorithm rewrites (transposes, structure, powers), and the default pipeline (which includes `sky-egraph`'s pass); later autodiff | sky-ir, sky-egraph |
| `sky-egraph` | identity-aware e-graph rewriting (`egg`) as a pass | sky-ir |
| `sky-ir-interp` | reference interpreter, and its parallel runtime (`rayon`) | sky-ir, sky-units |

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
