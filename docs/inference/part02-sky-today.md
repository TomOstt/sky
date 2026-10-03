# Part II. Sky as it exists today

This part describes the Sky compiler as it stands at `main` = `a99e7e3` (2026-10-01), plus ten uncommitted feature worktrees. It is the factual basis for decision D1 in [Part VI](part06-architecture.md): tensors get a sibling IR, `sky-tir`, instead of a bigger SIR. The CPU performance of the same code is measured in [Part III](part03-current-performance.md).

**Sources.** [R1](research/R1-sky-today.md) (architecture map), [R10](research/R10-sky-cpu-perf.md) (CPU performance), `docs/design/ir.md`, `ROADMAP.md`, `spec/v0.1.md`, and three probes run for this chapter with the release `sky` binary built from `a99e7e3`; each probe is shown where it is used. Repository paths are relative to the repository root. Labels follow the report's rules: `[VERIFIED: source]`, `[LIKELY: reason]`, `[EXPERIMENTAL: what decides]`, `[MOONSHOT]`.

## Summary

Sky today is a CPU-only, f64-only scientific language. Its type checker is the most developed part: it tracks physical dimensions, symbolic shapes (by equality only), four matrix-property tags and cost polynomials. Three engines must print byte-identical output:

- an AST tree-walker, kept as the oracle;
- a reference interpreter for SIR, Sky's SSA IR;
- Cranelift native code for SIR, just-in-time or ahead of time.

All three share one value layer, `sky_interp::{Value, arith, array, solve, math}`. Arrays are dense, row-major, immutable, never empty, of rank at most 2, and hold at most 2^24 f64 entries. Native code compiles scalars, Bools and arrays of at most 16 entries inline; every other array operation is a call into the interpreters' runtime. [VERIFIED: R1 §0]

| Fact | Today | Evidence |
|---|---|---|
| Numeric types | f64 only; no integer type | [VERIFIED: `crates/sky-interp/src/value.rs:16-54`; R1 §3.6] |
| Rank of any value that can exist at run time | at most 2 | [VERIFIED: `crates/sky-ir/src/op.rs:381-382`; probe in §3.5] |
| Entries per array | at most 2^24 = 16,777,216 | [VERIFIED: `crates/sky-types/src/constant.rs:23`] |
| Size arithmetic (`N+1`, `2*N`) | none; sizes unify by equality only | [VERIFIED: `crates/sky-types/src/shape.rs:1-6`] |
| SIR operations | 47 `OpKind` variants | [VERIFIED: `crates/sky-ir/src/op.rs:212-668`] |
| Optimization passes in the default pipeline | 13 | [VERIFIED: `crates/sky-opt/src/pipeline.rs:109-123`] |
| IR levels implemented, of L1–L4 | L1 only | [VERIFIED: `docs/design/ir.md:25-34`; no `Level` type in `crates/sky-ir`] |
| Values native code keeps in registers | f64 scalars, Bools, arrays of ≤ 16 entries | [VERIFIED: `crates/sky-jit/src/module/small.rs:53`] |
| Cost of one boxed array operation in native code | 148 ns per `x[i]`; about 291 ns per 32-entry add | [VERIFIED: R10 §4.4; R1 App. B] |
| GPU, Metal, f16/bf16/int code | none | [VERIFIED: grep of `crates/`, R1 §9.3c] |
| Production Rust | 75,005 lines in 22 crates | [VERIFIED: R1 §2.1] |
| Tests | 2,628 `#[test]` functions, 164 UI goldens, 22 IR golden programs, 5 fuzz targets | [VERIFIED: R1 §6.3] |
| Continuous integration | self-hosted Linux runner only | [VERIFIED: `.github/workflows/ci.yml`] |
| In-flight features | 10 worktrees, all uncommitted | [VERIFIED: `git rev-list --count main..worktree-agent-*` = 0 for all ten, re-checked for this chapter] |

The rest of this part explains each row, what it costs the inference work, and what carries over (§14).

---

## 1. The pipeline from source to hardware

### 1.1 Stages

| Stage | Entry point | Output | Notes |
|---|---|---|---|
| Project discovery | `sky_project::discover_opened`, called at `crates/sky-cli/src/run.rs:110` | `Discovery{edition, numerics}` | walks up to `sky.toml` under the rules of spec "Imports and the project root" |
| Lex | `sky_syntax::lex`, `crates/sky-syntax/src/lexer.rs:117` | `Lexed{tokens, warnings}` | `INDENT`/`DEDENT`; rejects bidi controls; NFKC identifier checks; at most 16 MiB of source (`crates/sky-syntax/src/lib.rs:37`) |
| Parse | `sky_ast::parse_module`, `crates/sky-ast/src/parser.rs:138` | `sky_ast::Module` plus errors | Pratt expression parser; never panics |
| Check | `sky_types::check_module_with`, `crates/sky-types/src/check.rs:495` | `Checked` (`check.rs:409`) | units, shapes, tags, chain plans, Cholesky sites, `expr_types: SpanMap<Ty>`, functions |
| Driver | `analyze_ir` (`crates/sky-cli/src/frontend.rs:98`); `run_source` (`crates/sky-cli/src/run.rs:163`) | — | merges lexer warnings and checker diagnostics, sorted by span |
| Lower | `sky_lower::lower_for_run`, `crates/sky-lower/src/lib.rs:360` | `sky_ir::Module` (L1) | uses the checker's types and never re-derives them; derivative functions come from `sky-ad` |
| Optimize | `sky_opt::default_pipeline().verify_each(true)`, `crates/sky-cli/src/ir.rs:209` | optimized SIR | 13 passes; the verifier runs before the first and after every pass |
| Execute natively | `sky_jit::compile_par` (`crates/sky-jit/src/module.rs:449`), then `Compiled::run_or_rerun` | stdout | Cranelift, whole module, in memory |
| Execute SIR | `sky_ir_interp::run_with`, `crates/sky-ir-interp/src/lib.rs:149` | stdout | reference interpreter |
| Execute AST | `sky_interp::run_module_with_derivatives`, `crates/sky-interp/src/eval.rs:89` | stdout | tree-walker, the oracle |
| Ahead of time | `sky_jit::aot::compile_object` (`crates/sky-jit/src/aot.rs:127`) + `libsky_rt.a` + the system `cc` | executable | carries the SIR text and the SIR interpreter, so it can deoptimize |

[VERIFIED: R1 §1.1]

### 1.2 Diagram: Sky today

```text
                              Sky today   (main a99e7e3, 2026-10-01)

 prog.sky
   │  sky-project: finds sky.toml (edition, numerics); reads the file by handle, ≤ 16 MiB
   ▼
 sky-syntax::lex ─────────► tokens: INDENT/DEDENT; bidi controls rejected; NFKC identifiers
   ▼
 sky-ast::parse_module ───► AST: Module{Stmt}, Expr, TypeExpr{tags, shape, unit}
   ▼
 sky-types::check_module_with ──► Checked{expr_types, chains, cholesky_solves, funcs}
   │   units: Dimension = 7 rational exponents    shapes: Known(u64) | Var, equality only
   │   tags: Symmetric, PositiveDefinite,          chain planner for @ chains and A⁻¹ @ b
   │         Diagonal, Orthogonal                  cost polynomials (Poly); numerics warnings
   │
   │   --engine=auto:   lower ──ok──► compile ──ok──► native
   │                      │              └── CompileError ──► sir
   │                      └── LowerError ───────────────────► ast
   │   Both steps finish before the program starts, so a fallback never repeats output.
   │
   ├── ast ──► sky-interp: tree-walker, BTreeMap<String, Value> environment ──────────────────────┐
   │           derivative(f, x): sky-derive ─► sky-lower ─► sky-ad ─► sky-ir-interp               │
   ▼                                                                                              │
 sky-lower::lower_for_run (+ derivative functions from sky-ad)                                    │
   ▼                                                                                              │
 SIR, level L1 only: typed SSA in arenas, structured regions, 47 OpKinds, f64 meaning             │
   ▼   sky-opt::default_pipeline, verifier before the first pass and after every pass:            │
   │   specialize, fold, egraph, gvn, licm, contracts, dce, powers, fuse,                         │
   │   cholesky, transposes, structure, algorithms                                                │
   │                                                                                              │
   ├── sir ──► sky-ir-interp: one value table per call; moves operands that die ──────────────────┤
   │           compiles a `fused` body with sky-jit::kernel after 512 elements                    │
   ▼                                                                                              │
 sky-jit: Cranelift 0.121.2, whole module compiled in memory (native)                             │
   │   inline machine code: f64 scalars, i8 Bools, arrays of ≤ 16 entries (F64X2 SLP)             │
   │   any other value is a u32 handle; each operation on it calls sky_rt_op(site) ───────────────┤
   │   deoptimization: the program re-runs from the start on sky-ir-interp                        │
   │   sky build: sky-jit::aot ─► .o + libsky_rt.a (runtime, SIR text, SIR interpreter)           │
   │                                                                                              │
   ▼                                                                                              ▼
 ══ ONE VALUE LAYER FOR ALL ENGINES: sky_interp::{Value, arith, array, solve, math, vecfn} ══
                                              ▼
 sky-linalg: f64 only. GEMM (packed, blocked, 4×8 register tile), Cholesky, LU, inverse,
             syrk, triangular solves, dot in a fixed 8-lane order; rayon; no FMA by default
             └── --numerics=relaxed: NEON FMA kernels; Accelerate dgemm, dpotrf, dgesv (AMX)
                                              ▼
 HARDWARE: CPU cores only. f64 scalar code and 128-bit NEON. No GPU, Metal, ANE or SME code;
           no f32, f16, bf16 or integer arithmetic anywhere.
```

[VERIFIED: R1 §1.3, redrawn; every box is a function named in §1.1]

### 1.3 How `--engine=auto` chooses, and when it falls back

`auto` is the default. It prefers native code, then SIR, then the tree-walker, and it decides before anything runs (`crates/sky-cli/src/run.rs:190-268`):

1. **Lowering runs first** (`run.rs:190-216`). A `LowerError` sends the program to the tree-walker, silently. Under `--engine=sir` or `--engine=native` the same error is fatal: "cannot run on SIR", "cannot run natively".
2. **Native compilation runs next** (`run.rs:219-239`). A `CompileError` sends the program to the SIR interpreter, silently under `auto`.
3. **Deoptimization is not an error.** When a value the checker typed `unknown` turns out not to match a declared type, native code stops and the program re-runs from the start on the SIR interpreter, skipping the bytes native code already wrote (`run.rs:250-258`).
4. **A verifier failure before the first pass** counts as "unsupported" and falls back to the tree-walker. A failure after a pass is an internal error (`crates/sky-cli/src/ir.rs:211-221`).

[VERIFIED: R1 §1.2]

### 1.4 What never lowers

These constructs raise `LowerError` and always run on the tree-walker [VERIFIED: R1 §1.2]:

| Construct | Where lowering refuses it |
|---|---|
| `±` uncertainty | `crates/sky-lower/src/expr.rs:93,422` |
| `equation` / `solve` | `crates/sky-lower/src/stmt.rs:248` |
| attributes; `∂` and `∇` operators | `expr.rs:100`; `expr.rs:192` |
| functions used as values | `expr.rs:383,469` |
| input declarations below the top level | `stmt.rs:332` |
| `for x in <array of unknown type>` | `crates/sky-lower/src/for_loop.rs:261` |
| arrays whose size is known only at run time | `crates/sky-lower/src/ctx.rs:263` |

The last row matters for inference: a sequence length or a KV-cache length is a run-time size, so any Sky program that depended on one would run on the slowest engine.

### 1.5 One value layer under three engines

`sky-jit` depends on `sky-interp`, and so does `sky-opt` (it folds constants with the interpreter's own arithmetic). The tree-walker's `Value` is therefore the semantic root of every engine: native code calls `sky_interp::{arith, array, solve, math}` for every operation it does not inline, and the SIR interpreter evaluates with the same functions [VERIFIED: R1 §2.2, from the `Cargo.toml` files]. This is how Sky keeps three engines byte-identical. It is also why a new data model, such as typed tensors, cannot be added "in SIR only": it would have to enter `sky_interp::Value` and `sky_interp::array`, which every engine and the optimizer share [LIKELY: R1 §2.2].

---

## 2. Crates

The workspace has 22 crates. Totals: 133,545 lines in `src/`, of which 75,005 are production code (the rest are 17,734 comment lines, 5,555 blank lines, 25,474 lines of in-source tests and 10,023 generated lines; these categories overlap by a few hundred lines); 47,606 lines in `tests/`; 988 in `benches/`; 5,225 in `fuzz/` [VERIFIED: R1 §2.1, `loc.py`]. "Prod" below counts non-blank, non-comment lines outside test modules and generated tables. Line references in the last columns are inside `crates/<crate>/src/`.

| Crate | Purpose | src lines | Prod | Test lines (src + tests/) | Key types and entry points | Workspace deps | External deps |
|---|---|---:|---:|---:|---|---|---|
| `sky-diag` | spans, diagnostics, renderer | 1,745 | 441 | 961 + 76 | `Span`, `Diagnostic`, `SourceFile`, `with_deep_stack` | — | unicode-ident |
| `sky-syntax` | lexer, tokens, NFKC and confusables, `Edition` (9,953 lines are a generated NFKC table) | 14,775 | 1,490 | 2,780 | `lex` (`lexer.rs:117`), `Token`, `TokenKind`, `MAX_SOURCE_BYTES` | diag | unicode-ident |
| `sky-ast` | AST and Pratt parser | 5,695 | 4,551 | 101 + 4,781 | `parse_module` (`parser.rs:138`), `Stmt`, `Expr`, `TypeExpr` (`ast.rs:212`), `ShapeAnn` | diag, syntax | — |
| `sky-units` | dimensions, units, SI prefixes, CODATA constants, math-function dimension rules | 3,345 | 1,490 | 1,199 + 844 | `Dimension` (`dimension.rs:109`), `Rational` (`rational.rs:16`), `MathFn` (`math.rs:37`) | — | — |
| `sky-types` | the checker: units, shapes, tags, chain planner, flop polynomials, cost syntax, numerics warnings, ODE typing | 16,813 | 11,872 | 1,684 + 3,773 | `check_module_with` (`check.rs:495`), `Checked` (`:409`), `Ty` (`:161`), `ValueTy` (`:123`), `Shape`/`ShapeDim` (`shape.rs:138/98`), `TensorType` (`rules.rs:18`), `plan_chain` (`chain.rs:555`), `poly::Poly` | ast, diag, linalg, units | — |
| `sky-linalg` | reproducible f64 dense linear algebra; relaxed and Accelerate paths | 4,511 | 3,180 | 171 + 1,291 | `gemm`, `gemm_into_with`, `syrk`, `cholesky*`, `lu`, `inverse`, `dot`, `Numerics`, `tuning::register` | tune | rayon |
| `sky-interp` | tree-walker and the shared value layer | 10,491 | 4,966 | 4,038 + 52 | `Value` (`value.rs:16`), `run_module_with_derivatives` (`eval.rs:89`), `arith`, `array`, `solve`, `math::apply`, `vecfn` | ast, diag, ode, syntax, types, units | — |
| `sky-ode` | ODE integrators and method choice (RK4, DOPRI5, Verlet, Yoshida4, Rodas3) | 2,593 | 1,774 | 163 + 1,758 | `System`, `Options`, `Method`, `solve` | — | libm |
| `sky-ir` | SIR data model, op table, verifier, text format, pass manager, analyses (effects, exact, roundoff, cost, lifetime, parallel) | 14,665 | 10,737 | 702 + 5,359 | `Module`/`Function` (`ir.rs:356/306`), `Type`/`NumberType`/`Repr` (`ir.rs:74/53/42`), `OpKind` (`op.rs:212`), `verify` (`verify.rs:122`), `Pass`/`PassManager` (`pass.rs:197/253`), `cost::analyze` (`cost/mod.rs:236`), `Lifetimes` (`lifetime.rs:111`) | ast, diag, syntax, tune, types, units | — |
| `sky-lower` | checked AST to SIR L1 | 4,873 | 3,631 | 0 + 5,643 | `lower` (`lib.rs:310`), `lower_for_run` (`:360`), `LowerError` (`:252`) | ad, ast, diag, interp, ir, syntax, types, units | — |
| `sky-ad` | automatic differentiation as a SIR transform | 5,356 | 3,260 | 1,255 | `derivative`, `derivative_functions` | diag, ir, types, units | — |
| `sky-derive` | `derivative(f, x)` for the tree-walker, computed on SIR | 225 | 154 | 0 | `SirDerivatives` | ad, ast, diag, interp, ir, ir-interp, lower, types | — |
| `sky-egraph` | e-graph rewriting of SIR expression segments | 3,393 | 2,603 | 0 + 2,409 | `EgraphPass`, `RewriteContract`, `rule_table`, `CostModel` | diag, ir, types, units | egg 0.11.0, quanta |
| `sky-opt` | passes and the default pipeline | 14,036 | 5,863 | 6,106 + 5,801 | `default_pipeline` (`pipeline.rs:362`), `DEFAULT_PASSES` (`:109`), `MatMulPolicy`, `specialize_for` | diag, egraph, interp, ir, types, units | — |
| `sky-ir-interp` | SIR reference interpreter, parallel runtime, compiled `fused` kernels | 5,901 | 2,184 | 2,996 + 6,563 | `run` (`lib.rs:140`), `run_with` (`:149`), `ParConfig`, `JitOptions` | diag, interp, jit, ir, tune, units | rayon |
| `sky-jit` | Cranelift native code: whole-module JIT, fused kernels, AOT objects, `extern "C"` runtime | 10,730 | 7,348 | 1,044 | `compile` (`module.rs:424`), `compile_par` (`:449`), `Compiled` (`:273`), `kernel::compile_kernel` (`kernel.rs:1147`), `aot::compile_object` (`aot.rs:127`), `Slp` (`module/slp.rs:74`) | diag, ir, interp, types, units, tune | cranelift-* 0.121.2, target-lexicon, rayon |
| `sky-rt` | static runtime library for `sky build` executables | 249 | 131 | 31 | `sky_rt_start`, `Meta` | diag, ir-interp, jit | — |
| `sky-hw` | CPU and OS discovery, machine fingerprint | 1,011 | 509 | 320 | `HardwareInfo` (`lib.rs:32`), `detect`, `fingerprint`, `kernel_threads` | — | — |
| `sky-tune` | autotuner and data-only performance database | 2,090 | 965 | 652 + 61 | `Tunable` (`measure.rs:31`), `lookup` (`lib.rs:138`), `Db`, `Registry`, `Store` | hw | — |
| `sky-project` | project root, `sky.toml`, import resolution (not yet wired into the language) | 2,754 | 1,699 | 487 + 1,407 | `discover`, `discover_opened`, `resolve_import`, `parse_manifest`, `Numerics` | diag, syntax | libc |
| `sky-cli` | the `sky` binary: check, run, build, repl, hw, tune | 3,865 | 2,929 | 110 + 7,452 | `main` (`main.rs:68`), `run::run`, `ir::prepare_run`, `frontend::analyze_ir` | 17 workspace crates | — |
| `sky-bench` | engine benchmarks, parallel break-even study, cross-language comparison | 4,429 | 3,228 | 674 + 336 | `bench.rs`, `compare/` (writes `benches/compare/RESULTS.md`), `par.rs`, `native_par.rs` | 12 workspace crates | — |

[VERIFIED: R1 §2.1]

The largest production crates are `sky-types` (11.9k), `sky-ir` (10.7k), `sky-jit` (7.3k), `sky-opt` (5.9k) and `sky-interp` (5.0k).

### 2.1 Dependency layers

```text
L0  sky-diag     sky-hw      sky-units     sky-ode (libm)
L1  sky-syntax (diag)                      sky-tune (hw)
L2  sky-ast (diag, syntax)   sky-linalg (tune; rayon)   sky-project (diag, syntax)
L3  sky-types (ast, diag, linalg, units)
L4  sky-interp (ast, diag, ode, syntax, types, units)
    sky-ir (ast, diag, syntax, tune, types, units)
L5  sky-ad (diag, ir, types, units)        sky-egraph (diag, ir, types, units; egg)
    sky-jit (diag, ir, interp, types, units, tune; cranelift, rayon)
L6  sky-lower (ad, ast, diag, interp, ir, syntax, types, units)
    sky-opt (diag, egraph, interp, ir, types, units)
    sky-ir-interp (diag, interp, jit, ir, tune, units; rayon)
L7  sky-derive (ad, ast, diag, interp, ir, ir-interp, lower, types)
    sky-rt (diag, ir-interp, jit)
L8  sky-cli (17 crates)      sky-bench (12 crates)
```

[VERIFIED: R1 §2.2, from the `Cargo.toml` files]

Three edges shape the inference work:

- **`sky-jit` → `sky-interp`** and **`sky-opt` → `sky-interp`**: the value layer is below the native backend and the optimizer (§1.5).
- **`sky-types` → `sky-linalg`**: the checker proves positive definiteness and computes condition-number warnings with the real kernels (`crates/sky-types/src/linalg.rs:1-25`). A consequence: any crate that wants `sky_types::poly::Poly`, the symbolic-size polynomial D1 reuses, also links the checker and `sky-linalg` (§14).
- **`sky-ir` → `sky-tune`**: the parallel thresholds of SIR are tuned values.

---

## 3. The type system and its limits

### 3.1 Three type layers

| Layer | Type | Contents |
|---|---|---|
| Checker | `Ty` (`crates/sky-types/src/check.rs:161`) | `Value(ValueTy{dim, ty: TensorType{shape, tags}})`, `Bool`, `Str`, `Func`, `Ode`, `Unknown` (an unannotated parameter, or the result of an earlier error, so one error never cascades) |
| SIR | `Type` (`crates/sky-ir/src/ir.rs:74`) | `Number(NumberType{dim, shape, tags, repr})`, `Bool`, `Str`, `Nothing`, `Unknown` |
| Run time | `Value` (`crates/sky-interp/src/value.rs:16-54`) | §4 |

`Repr` (`crates/sky-ir/src/ir.rs:42`) is `Unchosen | Exact | F32 | F64`. **No pass chooses a `Repr`, and no engine reads one: every number is computed as f64.** The only non-test uses of `Repr::F32` and `Repr::Exact` are the printer, the parser and display code. `docs/design/ir.md:64` plans F16, BF16 and F128 for "Phase 3". [VERIFIED: R1 §3.1, grep]

### 3.2 f64 only, and no integer type

| Feature | Status | Evidence |
|---|---|---|
| Element type | f64 everywhere: `Value::Number.value: f64`, `Value::Array.data: Vec<f64>`, every kernel takes `&[f64]` | [VERIFIED: R1 §3.6] |
| Bytes per entry in the cost model | `ENTRY_BYTES = 8` | [VERIFIED: `crates/sky-ir/src/cost/ops.rs:16`] |
| Integers | none. An integer is an f64 that happens to be whole. Loop indices are f64 and bounded by `MAX_RANGE_BOUND = 2^53` so that `i + 1` never stalls | [VERIFIED: `crates/sky-types/src/constant.rs:25-30`] |
| Index operands | dimensionless scalars checked at run time to be whole and in range (`expect.range`, `index`) | [VERIFIED: R1 §3.6; `docs/design/ir.md`, "Range checks"] |

Token ids up to 151,936 are exact in f64, so the missing integer type is not a correctness problem for ids. It is a problem for everything around them: bit operations and hashing (tokenizers), byte strings, integer tensors for gather and scatter, 4-bit and 8-bit packed weights, and the cost of converting and range-checking an f64 at every index [LIKELY: these operations have no f64 formulation, or a slow one].

### 3.3 Rank at most 2, at most 2^24 entries

- **The checker's `Shape` has any rank** (`Vec<ShapeDim>`, `crates/sky-types/src/shape.rs:138`), but nothing at run time does:
  - array literals have at most two levels ("array literals have at most two levels", `crates/sky-types/src/array.rs:146-152`);
  - the SIR array operations are for vectors and matrices only ("a vector has one axis and a matrix two (v0.1 has no higher ranks)", `crates/sky-ir/src/op.rs:381-382`);
  - `Value::Array` is documented as "a vector (one entry in `dims`) or matrix (two)" (`crates/sky-interp/src/value.rs:30-34`).
- **`MAX_ARRAY_ELEMENTS = 1 << 24`** (`crates/sky-types/src/constant.rs:23`): 16,777,216 entries, 128 MiB of f64, enforced by both the checker and the interpreter. Spec: "an array may have at most 2^24 elements" (`spec/v0.1.md:596`).

For scale, using the public Qwen3-32B configuration as encoded in [`roofline.py`](roofline.py) (hidden size 5,120, MLP width 25,600, vocabulary 151,936):

| Qwen3-32B tensor | Entries | Multiple of the cap |
|---|---:|---:|
| one MLP projection, 5,120 × 25,600 | 131,072,000 | 7.8× |
| embedding or output head, 151,936 × 5,120 | 777,912,320 | 46.4× |

[VERIFIED: arithmetic on the configuration in `roofline.py`]

### 3.4 Sizes are matched by equality only

- **Representation.** `ShapeDim = Known(u64) | Var(ShapeVar)` (`crates/sky-types/src/shape.rs:98-104`). "There is no dimension arithmetic (`N+1`, `2*N`), so `ShapeDim` has no way to express it" (`shape.rs:1-6`). `Matrix[N+1, N]` is a parse error with help (spec "Type annotations"), and "dimension arithmetic such as `N+1`" is listed as out of scope (`spec/v0.1.md:1029`). [VERIFIED]
- **Unification** is equality only, with provenance recorded for diagnostics (`ShapeUnifier`). It is monomorphic within a module and generic per `def`: each call gets fresh copies of the callee's size variables. [VERIFIED: R1 §3.2]
- **Sizes must be compile-time constants.**
  - `zeros(len(h))` inside a generic function is an error: "`zeros` needs a constant size, found `len(h)`" (R1 probe `p2_zeros_n.sky`).
  - A slice with a computed bound gets a fresh rigid size variable that equals only itself; lowering refuses it where a stand-in value is needed (`crates/sky-lower/src/ctx.rs:263`).
  - An array whose size is known only at run time does not lower (§1.4).
- **Specialization** clones a generic function per concrete shape or constant (`f[N=3]`), at most 8 clones per function and 64 per module (`docs/design/ir.md`, "Specialization").

What this rules out for a transformer, all LIKELY from the rules above: a KV cache whose length grows by one per token (`T + 1`), a head split (`D = heads · head_dim`, a product of two sizes), a concatenation (`T1 + T2`), and a prompt of run-time length.

### 3.5 The `Tensor` shape word parses, checks and lowers, but no tensor can exist

The annotation grammar has three shape words (`spec/v0.1.md:543`; `crates/sky-ast/src/parser.rs:2345`):

```text
shape := ( "Vector" | "Matrix" | "Tensor" ) "[" dim ( "," dim )* [ "," ] "]"
```

The spec's own example is `M: Tensor[B, T, D] kg`. The checker accepts any number of dimensions for `Tensor` (`ShapeKind::Tensor => None` in `crates/sky-types/src/check.rs:1357`). Probe for this chapter:

```text
$ cat t1.sky                      $ cat t3.sky
M: Tensor[2, 3, 4]                def g(T: Tensor[2, 2, 2]):
print(1)                              return T * 2
                                  print(1)

$ sky check t1.sky                -> t1.sky: ok
$ sky check --emit-ir t1.sky      -> %M = input "M" : num[1; (2, 3, 4)] @ 0..18
$ sky run t1.sky                  -> error: `M` is declared but has no value
$ sky run --engine=native t1.sky  -> error: `M` is declared but has no value
$ sky check --emit-ir t3.sky      -> %2 = mul %T, %1 : num[1; (2, 2, 2)]
$ sky run t3.sky                  -> 1          (g is never called)
```

[VERIFIED: probe with the release binary built from `a99e7e3`]

So a rank-3 type is accepted by the parser, the checker, lowering and the SIR verifier, but **no rank-3 value can ever be made**: there is no rank-3 literal, no builtin returns one, and the only remaining source is a value-less declaration (an "input"), which v0.1 has nothing to read from. The word is a placeholder. It is the natural attachment point for the tensor surface proposed in D8 (`x: Tensor[T, 5120] in f16`; [Part VII](part07-tensor-ir.md)), but nothing below the checker supports it, and the verifier's acceptance of rank-3 types is incidental rather than designed [LIKELY: no test or design text mentions rank 3 except to exclude it].

### 3.6 Units, tags and uncertainty

These are Sky's scientific core and are irrelevant to LLM inference, where tensors are dimensionless. Three facts matter because of their cost or their reach:

- **Dimension** is 7 rational exponents, each two `i64`s: 112 bytes, stored in every run-time `Value` (`crates/sky-units/src/dimension.rs:109-111`, `rational.rs:16-19`). [VERIFIED: R1 §3.3]
- **Tags** (`Symmetric`, `PositiveDefinite`, `Diagonal`, `Orthogonal`) are a closed `u8` lattice fixed in code (`crates/sky-types/src/tags.rs:32-140`). Literal tags are proven from the entries; tags on parameters are run-time-checked promises. [VERIFIED: R1 §3.3]
- **Uncertainty** (`x ± dx`) is not part of the type; SIR cannot represent it, so such programs run on the tree-walker. [VERIFIED: R1 §3.4]

### 3.7 Other language features, and what a Qwen3 layer would need

| Feature | Status on `main` | In flight (§13) |
|---|---|---|
| records / structs | none | — |
| tuples | none | tuples worktree |
| modules / `import` | parses, "not supported yet"; the resolver in `sky-project` is not wired in (`spec/v0.1.md:1031`) | — |
| generics | size variables in signatures; unannotated parameters (`unknown`), specialized by the `specialize` pass | — |
| closures, first-class functions | none; `call @f` names a fixed callee | — |
| strings | literals, `print`, `==`/`!=` only | readable-output worktree (f-strings) |
| element writes | check errors: "arrays are immutable in v0.1" (`spec/v0.1.md:593`) | arrays worktree (`store`) |
| reductions (`sum`, `max`, `argmax`, ...) | none ("cannot find `sum`", R1 probe `p3_reduce.sky`) | reductions worktree, vectors only |
| broadcasting | number with array only; array with array forbidden by design (`spec/v0.1.md:608`) | broadcasting worktree, number with vector only |
| file input | none | data-in worktree (`read_csv`) |

[VERIFIED: R1 §3.6, §7.1]

Writing one Qwen3 decoder layer in Sky source needs twelve things that `main` lacks [VERIFIED as missing: R1 §9.3b]:

| # | Needed | Today |
|---:|---|---|
| 1 | N-d tensors of rank 3–4, reshape, axis permutation | rank ≤ 2, no reshape |
| 2 | f16/bf16 activations; int4/int8 block-quantized weights with scales; casts | f64 only |
| 3 | integer scalars and tensors (ids, positions, lengths) | none |
| 4 | broadcasting between arrays, at least by name | forbidden |
| 5 | axis reductions, softmax, argmax | none; in-flight work is vector-only |
| 6 | gather (embedding lookup) and scatter | none |
| 7 | KV-cache update; run-time sizes | `store` and checker-only sizes in flight |
| 8 | batched matmul and attention over heads with GQA sharing | none |
| 9 | records or modules for layer parameters; a list of layers | none |
| 10 | binary input of about 18 GB of weights (mmap); bytes and strings for the tokenizer | no file I/O |
| 11 | IEEE −inf and NaN in masked softmax | every non-finite value is an error |
| 12 | a cost model with bytes per dtype | `ENTRY_BYTES = 8` |

This list is why D7 makes a Rust model builder in `sky-llm` the first front door, and Sky source the second ([Part VI](part06-architecture.md)).

---

## 4. Values and memory

### 4.1 Tree-walker

- `Value` (`crates/sky-interp/src/value.rs:16-54`) has the variants `Number{value: f64, dim}`, `Uncertain`, `Array{dims: Vec<usize>, data: Vec<f64>, dim}`, `Bool`, `Str`, `Shown`, `None`, `Func(Arc<Function>)` (a clone of the checked AST) and `Ode`. [VERIFIED: R1 §4.1]
- A `Value` is at least about 160 bytes: the `Array` variant holds two `Vec`s and a 112-byte `Dimension` [LIKELY: layout reasoning, not measured].
- The environment is `RuntimeEnv{vars: BTreeMap<String, Value>, chains}` (`crates/sky-interp/src/env.rs:1-20`): every variable read is a string-keyed B-tree lookup. [VERIFIED]
- Arrays are owned values, cloned on copy, never shared, with no views. The tree-walker moves a variable at `x = x + …` when the right-hand side reads `x` once and can run no user code (`docs/design/ir.md:272`). [VERIFIED]

### 4.2 SIR interpreter

- The same `sky_interp::Value`, kept in a value table per activation (`Vec<Option<Value>>`, `crates/sky-ir-interp/src/machine.rs:1-120`). [VERIFIED]
- `sky_ir::Lifetimes` (backward liveness, `crates/sky-ir/src/lifetime.rs:111,125`) marks where each operand dies. The interpreter moves dying operands and writes elementwise results into their buffers (`sky_interp::array::arith_owned`). The loop `v = v + dt * a` allocates nothing per iteration, tested by a counting allocator in `crates/sky-bench/tests/allocations.rs`. [VERIFIED: R1 §4.2]
- Every array is still two allocations (`dims` and `data`), and slices and transposes copy: "`Value::Array` has no strides or offsets" (`docs/design/ir.md:268,270`). A call allocates its argument list and frame (`machine.rs:30`). [VERIFIED]

### 4.3 Native code

- A scalar is an `f64` (its dimension is static and not stored) and a Bool an `i8`.
- A **small array** (static shape, at most 16 entries, `MAX_SMALL` at `crates/sky-jit/src/module/small.rs:53`) is scalar-replaced into N f64 SSA values and crosses calls, `if` and `while` as N separate values. The limit was chosen by measurement: a 4×4 matrix product in a loop took 0.20 s as handles and 0.01 s as scalars, but unrolling costs about 3 ms of compile per 4×4 product (`small.rs:42-53`). [VERIFIED]
- **Everything else** (arrays over 16 entries, strings, `unknown` values) is a handle: a `u32` index into a thread-local table of `Cell{rc, value: Option<Rc<Value>>}` (`crates/sky-jit/src/runtime.rs:152-156,192`). Generated code retains and releases handles; an array operation pushes its operands and calls `sky_rt_op(site)` (`crates/sky-jit/src/module.rs:2229-2284`), which runs the interpreters' functions. [VERIFIED: R1 §4.3]

### 4.4 Allocation, alignment, aliasing

| Property | Today | Evidence |
|---|---|---|
| Allocator | the system allocator; no `#[global_allocator]` outside two test binaries | [VERIFIED: R1 §4.4, grep] |
| Alignment | `Vec<f64>`, 8 bytes; the kernel code relies on nothing more (`crates/sky-jit/src/kernel.rs:74`) | [VERIFIED] |
| Code memory | one 256 MiB arena per module, 16 MiB per compiled kernel (`crates/sky-jit/src/lib.rs:78-93`) | [VERIFIED] |
| Aliasing and mutation | none: arrays are immutable values, so no memory model is needed | [VERIFIED: spec, `value.rs:30-34`] |
| GEMM packing | temporary buffers per call | [LIKELY: R1, from `gemm.rs` structure] |

For inference this means there is nothing to build on below the value level: no buffers, dtypes, devices, page alignment, views, in-place update or arena planning. The tensor stack needs its own memory model: the static memory plan, no-copy weight buffers and preallocated KV of D9 ([Part XIII](part13-llm-runtime.md), [Part XIX](part19-memory.md)).

---

## 5. SIR

### 5.1 Data model

- `Module{edition, functions}`. A `Function` holds flat arenas of values, ops, blocks and regions addressed by `u32` ids (`crates/sky-ir/src/ir.rs:306-319`).
- Regions are structured (MLIR `scf` style) and hold one block in v1; blocks end in `yield`, `condition` or `return`. Mutable variables become loop-carried block arguments.
- Every op carries a `span` and `operand_spans`, so a runtime error found on SIR has the tree-walker's labels.

[VERIFIED: R1 §5.1; `docs/design/ir.md`, "L1 data model"]

### 5.2 The 47 operations

| Group | Operations (text-format mnemonics) | Count |
|---|---|---:|
| Constants and inputs | `const.float`, `const.rational`, `const.bool`, `const.str`, `const.nothing`, `literal.error`, `input`, `const.array` (literals of ≥ 64 entries) | 8 |
| Arithmetic | `add`, `sub`, `mul`, `div`, `neg`, `pow.const`, `pow`, `inverse`, `math.*` (one variant covering 30 functions: sqrt … erf, sign) | 9 |
| Linear algebra | `matmul`, `transpose`, `norm`, `dot`; produced only by rewrites: `solve[cholesky\|lu\|diagonal]`, `solve.right[…]`, `matmul[ta,tb,alg=…]`, `diag.mul[left\|right]` | 8 |
| Arrays | `array.literal`, `index`, `slice`, `zeros`, `identity`, `len` | 6 |
| Kernels | `fused` (one region; a scalar body run per element) | 1 |
| Tags | `assume` | 1 |
| Comparison and dynamic checks | `cmp.{eq,ne,lt,le,gt,ge}`, `not`, `expect.bool`, `expect.number`, `expect.range` | 5 |
| Control | `if`, `while`, `yield`, `condition`, `call`, `return` | 6 |
| Contracts and output | `assert`, `print`, `show` | 3 |
| **Total** | | **47** |

[VERIFIED: variant list of `OpKind`, `crates/sky-ir/src/op.rs:212-668`; mnemonics from `op.rs:694-742`]

Effects are `{io, fail, call}` (`crates/sky-ir/src/ir.rs:105-116`). Almost every arithmetic op has `fail`, because overflow to infinity is an error. Calls are first-order (`call @name`); a function receives an extra parameter per module-level name it reads, and early `return`, `break` and `continue` become Bool flags. [VERIFIED: R1 §5.1]

**There is no memory model**: no buffers, loads, stores, pointers, views, dtypes, devices, streams or asynchronous operations. The design defers all of them to L2–L4 (`docs/design/ir.md:25-34`). [VERIFIED]

### 5.3 Verifier and text format

- **Verifier** (`crates/sky-ir/src/verify.rs:122`): arena integrity, SSA and scoping, region shape, per-op typing (it calls `sky_types::rules`, so checker and verifier cannot drift), closed tags, effect consistency, and the rules for `fused` bodies. It is total and never panics. [VERIFIED: R1 §5.1]
- **Text format** (`crates/sky-ir/src/print.rs`, `parse.rs`): data-only, round-trip exact (`parse(print(m)) == canonicalize(m)`), parsed by a hand-written parser that never panics and limits nesting and size; fuzz target `fuzz/fuzz_targets/sir_parse.rs`. [VERIFIED]
- **Pass manager** (`crates/sky-ir/src/pass.rs`): runs a declared pipeline, verifies before the first pass and after every pass with `verify_each(true)` (which `sky run` always sets), can print after each pass (`sky check --emit-ir=<stage>`), and collects `Explanation{pass, rule, span, before, after, because, related}` records. [VERIFIED]

### 5.4 The 13 passes

| # | Pass | What it does | Explanation rules |
|---:|---|---|---|
| 1 | `specialize` | clones a callee per call with known shapes, constants, signs or types (`f[N=3]`, `fib[n:Scalar]`); unrolls constant loops of ≤ 8 iterations and ≤ 256 added ops | `SPEC-001`…`005` |
| 2 | `fold` | constant folding, bit-identical to the interpreter | — |
| 3 | `egraph` | `egg` saturation of pure segments under the `Reproducible` contract (§5.7) | `EG-COST-001`, `ALG-*`, `LA-*` |
| 4 | `gvn` | global value numbering across regions, scoped by dominance | `GVN-001` |
| 5 | `licm` | loop-invariant code motion out of `while`, guarded for ops that can fail | `LICM-001/002` |
| 6 | `contracts` | removes `assert`/`requires`/`ensures` checks the exactness analysis proves true | `CONTRACT-001…003` |
| 7 | `dce` | removes dead pure ops, and failing-by-kind ops that provably cannot fail | `DCE-*` |
| 8 | `powers` | `x^n` for 2 ≤ n ≤ 14 as the multiplication chain of `powi`, bit-identical | `POW-CHAIN-001` |
| 9 | `fuse` | a single-use tree of elementwise ops in one block becomes one `fused` op | `FUSE-001` |
| 10 | `cholesky` | `matmul(inverse(A), b)` with PositiveDefinite `A` becomes `solve[cholesky]` | `LA-CHOLESKY-*` |
| 11 | `transposes` | `matmul(transpose(A), B)` becomes `matmul[ta]` | `LA-TRANSPOSE-001` |
| 12 | `structure` | Gram products (`syrk`), Diagonal scaling (`diag.mul`), `transpose(S)` of a Symmetric `S` | `LA-GRAM-001`, `LA-DIAGONAL-001`, `LA-SYMMETRIC-001` |
| 13 | `algorithms` | records an algorithm class per `matmul` through a `MatMulPolicy`; the default policy chooses nothing | `ALG-MATMUL-001` |

[VERIFIED: `crates/sky-opt/src/pipeline.rs:9-97,109-123,362`; R1 §5.2]

Alongside the passes there are analyses that rewrite nothing: `effects`, `exact` (interval and symbolic "ideal value"), `roundoff` (sound error bounds for `accuracy` budgets), `cost`, `lifetime` and `parallel`. Rewrites that change rounding are not passes: the checker decides `@`-chain association and solves, so that the tree-walker makes the same choice (`docs/design/ir.md`, "Algorithm choice"). `run_reporting` clones the whole module before every pass to decide "changed" (`crates/sky-opt/src/pipeline.rs:128-138`), which is fine for small programs [LIKELY: quadratic-ish for large generated graphs].

### 5.5 Levels: only L1 exists

`docs/design/ir.md:25-34` defines four levels as "the same data structure with a different set of allowed operations":

| Level | Name in `ir.md` | Meant to contain | Exists? | Where the tensor stack puts it (D1) |
|---|---|---|---|---|
| L1 | Semantic IR | mathematical operations on typed values, structured control flow, calls | **yes** | TIR graph, `sky-tir` |
| L2 | Strategy IR | chosen algorithms, precision per value, storage formats | no | TIR strategy, `sky-tir` |
| L3 | Kernel IR | iteration spaces, fused kernels, memory and lifetimes | no | kernel IR, `sky-kernel` |
| L4 | Backend | Cranelift or LLVM functions, MLIR modules, interpretation | Cranelift only, consuming L1 directly | MSL for Metal (`sky-metal`); TIR reference interpreter |

There is no `Level` type and no `fn level` in `crates/sky-ir` (grep), and `ir.md:121` says `level()` "is left out until there is a second level". `sky-jit` compiles L1 directly. [VERIFIED]

### 5.6 Hooks without consumers

- `sky_opt::specialize_for` and `SpecializationCache`, documented as the native engine's run-time specialization hook, have no caller outside `sky-opt`.
- `MatMulPolicy`: no kernel registers one. Of the algorithm classes, only `syrk` changes execution (`crates/sky-interp/src/solve.rs:527`).

[VERIFIED: R1 §5.2, grep]

### 5.7 `sky-egraph`

- Runs as pass 3 on each block's pure expression segments; effectful ops are barriers.
- Limits: 12 iterations, 20,000 nodes, 20,000 ops per segment, and no time limit by default, so the result does not depend on machine speed (`crates/sky-egraph/src/lib.rs:140-150`).
- Rule classes Exact, Conditional, Relaxed and RealOnly; contracts `Reproducible` (default), `Relaxed`, `Fast`. **No pass enables `Relaxed` or `Fast`**; `--numerics=relaxed` does not change the e-graph contract.
- 27 rules (`crates/sky-egraph/src/rules.rs:869-999`): scalar algebra, exponent laws, linear-algebra identities. None concerns tensors.
- Extraction cost: each e-node is priced by `sky_ir::cost` on a one-op function, `8·flops + bytes + 8·64·allocations`; symbolic sizes default to 64.

[VERIFIED: R1 §5.3]

### 5.8 What one new operation costs

Every `match` over `OpKind` is exhaustive on purpose: the verifier, printer, parser, both interpreters, the JIT, lowering, the e-graph, AD and the cost model. This is the stated reason `const.symbol` was deferred (`docs/design/ir.md:347`). Measured footprints:

| Evidence | Footprint |
|---|---|
| `OpKind::DiagMul`, a rare rewrite-only op | named in 10 non-test files in 4 crates |
| `OpKind::Fused` | named in 31 non-test files in 8 crates |
| reductions worktree: **one** new op, `Reduce(Reduction)` | 23 tracked source files modified in 9 crates, plus 3 new source files |
| arrays worktree: five new ops (`Store`, `Fill`, `Linspace`, `Arange`, `Concat`) | 23 tracked files modified in 7 crates (+610/−151), plus 5 new files (1,732 lines) |

[VERIFIED: grep and `git status` in the worktrees, re-run for this chapter; R1 §9.3b]

Each worktree also adds language features, so not every modified file is an exhaustive match [LIKELY]. The order of magnitude stands: an operation in SIR costs edits across most of the compiler. About twenty tensor operations would multiply that, and each would have to keep three engines byte-identical. This is the cost D1 avoids by giving tensors their own IR and adding a single SIR operation, `tensor.call @graph`.

---

## 6. Native code (`sky-jit`)

### 6.1 What compiles inline

- scalar f64 arithmetic, comparisons, `if`/`while`, calls between SIR functions;
- small-array operations (≤ 16 entries): literal, add, sub, neg, scaling, index, dot, norm, matmul, transpose, `fused`, zeros, identity, len;
- `powi` chains; inline `sqrt`, `abs`, `floor`, `ceil`, `min`, `max`;
- finiteness and zero-divisor checks; call-depth and stack-budget checks.

Every function returns `(status, results…)`; a failed check calls a runtime function that rebuilds the interpreters' exact diagnostic. [VERIFIED: R1 §5.6]

### 6.2 What stays a runtime call, and what it costs

Left as runtime calls (`crates/sky-jit/src/module.rs:2229-2284`): `show`, `matmul`/`norm`/`dot`/`transpose` on handles, `assume`, the solves, `matmul[…]`, `diag.mul`, `array.literal`, `const.array`, `index`, `slice`, `zeros`, `identity`, `len`, `fused` on handles (`fn fused` calls `generic`, `module.rs:2264`), `print`, `assert`, all non-exact math functions (through `sky_interp::math::apply_f64`), and anything typed `unknown`. [VERIFIED: R1 §5.6]

`generic` (`module.rs:2271-2284`) pushes each operand with a runtime call, calls `sky_rt_op(site)` (`crates/sky-jit/src/runtime.rs:583-606`), which matches on `OpKind` and runs the interpreter's function on `Rc<Value>`s, then allocates a handle for the result [VERIFIED: R10 §4.3]. Measured on the M2 Pro (medians, slopes; [Part III](part03-current-performance.md) §1 explains the method):

| Operation in a native loop | Cost | Evidence |
|---|---:|---|
| elementwise add of two 8-entry vectors (small array, inline) | about 3–4 ns per op | [VERIFIED: R1 App. B; EXPERIMENTAL-quality: load average 4.8] |
| elementwise add of two 32-entry vectors (handles, runtime call) | about 291 ns per op (SIR interpreter: about 330 ns) | same |
| `x[i]` on a 10^6-entry vector | 148 ns per access (SIR: 387 ns; C: 2.6 ns) | [VERIFIED: R10 §3.1, `r10/m5.log`] |

Crossing the 16-entry line costs about 75–100× per elementwise operation [LIKELY: R1 App. B]. This is the main reason nbody runs slower than C in native code, and [Part III](part03-current-performance.md) §3.1 traces it.

### 6.3 Calls, SIMD and threads

- **Call protocol.** Every function takes `(depth: i32, stack_limit: i64, site: i32, args…)` and returns `(status, results…)`; the callee checks the stack pointer, the caller checks depth before and status after the call. Measured cost: 2.0 ns per call against C's 0.74 ns. No inliner exists in the SIR pipeline or in Cranelift 0.121.2. [VERIFIED: R10 §4.3]
- **SIMD.** SLP pairs f64 operations into `F64X2`; the default `Auto` mode packs only loops that are not latency-bound (`crates/sky-jit/src/module/slp.rs:1-30,74-100`). Fused kernels use explicit `F64X2` loops. Nothing is wider than 128 bits. [VERIFIED: R1 §5.6]
- **Threads.** Big `fused` kernels and elementwise operations run in deterministic chunks; independent pure calls run on workers in windows from `sky_ir::parallel` (`module.rs:120-180`). Native thresholds are higher than the interpreter's: 64× for calls, 128× for kernels (524,288 element-operations) (`docs/design/ir.md`, "Automatic parallelism"). [VERIFIED]

### 6.4 `sky build`: executables that embed SIR and the interpreter

- The same `translate_module` emits a Mach-O or ELF object through `cranelift-object` for the target's baseline CPU (`crates/sky-jit/src/aot.rs:1-90`).
- The object holds an image with **the SIR text**, the dimension and site tables, the source and the warnings, plus a C `main` that calls `sky_rt_start`.
- It links against `libsky_rt.a` (crate `sky-rt`), which contains the runtime, **the SIR interpreter**, `sky-linalg` and Cranelift (for kernels compiled while running).
- Why the interpreter is inside: deoptimization (§1.3) is decided at run time, and a native runtime error is built by re-running the failing operation, which needs the operation (`spec/v0.1.md`, "Building executables").
- About 5.7 MB per executable whatever the program; it starts about 4 ms after an empty process; executables are single-threaded (no windows, no chunks); `crates/sky-cli/tests/build.rs` requires byte-identical behaviour to `sky run --engine=native`.

[VERIFIED: R1 §5.6; spec "Building executables"]

The pattern (ship the IR and its reference semantics inside the artifact) is a precedent for the AOT `.skybin` of `sky compile`, which could carry the TIR text for `sky explain` and validation [LIKELY; the design belongs to [Part XVIII](part18-ttft.md)].

### 6.5 No code cache

Nothing is cached across processes. Compiled kernels are cached per run, keyed by op address (`crates/sky-jit/src/runtime.rs:177-181`); the module is recompiled on every `sky run`. The floor for `print(1)` is 5.58 ms native against 1.45 ms for an empty C process (`benches/compare/RESULTS.md`, "Floors"). [VERIFIED]

---

## 7. Fusion and its limits

**Where it lives.** `crates/sky-opt/src/fuse.rs` forms `fused` ops; `crates/sky-ir/src/op.rs:467-507` defines their semantics; `crates/sky-jit/src/kernel.rs` compiles a body to a Cranelift loop; `crates/sky-ir-interp/src/native.rs` decides when the interpreter compiles; `crates/sky-ir-interp/src/kernel.rs` is a register kernel for `+ − × ÷ neg` bodies; `crates/sky-jit/src/module/small.rs` inlines fused ops on small arrays. [VERIFIED: R1 §5.4]

**How fusion is decided.** A purely syntactic, single-use tree rule inside one block: a candidate (`add`, `sub`, `mul`, `div`, `neg`, a math function) is absorbed into its consumer when its result has exactly one use, by a candidate in the same block, with the same array shape. Scalars are broadcast leaves. An operation that can fail never moves past another that can fail, print, call, loop or touch an `unknown` value (`fuse.rs:1-55`). [VERIFIED]

**Code generation** (`crates/sky-jit/src/kernel.rs:1-110,150-200`):

- loop-invariant values computed once and splatted;
- `F64X2` loop unrolled 4× (8 elements per iteration) and a scalar tail; unaligned-tolerant loads with no per-element bounds checks (`kernel.rs:754,865`);
- math functions other than `sqrt`/`abs` are scalar calls, lane by lane, into `sky_interp::math::apply`;
- failure detection by NaN accumulation (`acc += r − r`), one test after the loop, then a replay of the reference path to build the exact diagnostic;
- relaxed mode may contract `a*b + c` to FMA;
- large kernels run in deterministic chunks on rayon.

[VERIFIED: R1 §5.4; R10 §4.5]

**When it compiles.** In the SIR interpreter after 512 elements have run through the op (compiling costs 120–180 µs, `crates/sky-ir-interp/src/native.rs:11-30`); in native code through the runtime's kernel cache, immediately for kernels big enough for threads. [VERIFIED]

**Limits** [VERIFIED: R1 §5.4; R10 §4.5]:

| Limit | Consequence for transformer code |
|---|---|
| no reductions inside kernels | softmax, RMSNorm and argmax cannot fuse |
| no matmul prologue or epilogue fusion | no fused dequantization, bias, activation or residual around a product |
| single use only; no multi-output kernels; no recomputation | a value used twice (`T` in `sqrt(T) + T*0.5`) is materialized |
| no control flow in bodies | no masking by condition |
| broadcast only of scalars | no row or column vectors (norm weights, biases) |
| f64 and 128-bit SIMD only | no f16/bf16/int paths |
| non-trivial math functions are per-lane scalar calls | exp/tanh-heavy bodies (SiLU, softmax) pay a call per element [LIKELY: 5–10 ns per element, not measured] |
| native code does not reuse buffers | an extra write stream per op; +44% peak RSS against SIR (R10 §3.2) |

**Measured.** Kernels run at C speed: axpy 1.00–1.01× and hypot 1.00× of the best C/Rust loop, a six-operation chain with a division 1.14–1.17× (single thread, preallocated output) [VERIFIED: R10 §4.5]. Whole elementwise programs are 1.5–2.4× slower than C in `RESULTS.md`, mostly because of fixed costs ([Part III](part03-current-performance.md) §1.3).

The fusion rule is the precedent for kernel formation in TIR, but nothing of its implementation transfers: TIR's fusion must handle reductions, epilogues of quantized GEMV, broadcasts along named axes and multi-output kernels ([Part VIII](part08-graph-optimizer.md), [Part IX](part09-scheduler.md)).

---

## 8. `sky-linalg`

| Aspect | Today | Evidence |
|---|---|---|
| GEMM structure | Goto/BLIS: cache blocks over `j` (1024), `k` (256), `i` (128), tunable per size class (`le64`, `le256`, `le1024`, `gt1024`); `B` packed in 8-column slivers, `A` in 4-row slivers; 4×8 register tile; special unpacked kernels for ≤ 16 columns, matvec and vecmat; output split across rayon threads above 2^21 multiply-adds | [VERIFIED: `crates/sky-linalg/src/lib.rs:1-171`, `micro.rs:1-60`, `tuning.rs:71-102`] |
| Reproducible rule | each output entry summed in naive `k` order, no FMA; dot products in a fixed 8-lane interleaved order | [VERIFIED: spec "Arrays: reproducible arithmetic", `spec/v0.1.md:639-663`] |
| SIMD | reproducible kernels are plain Rust "written so LLVM vectorizes them" (NEON; AVX2 without FMA on x86); relaxed kernels use NEON `vfmaq_laneq_f64` | [VERIFIED: R1 §5.5] |
| Element type | f64 only; no f32 or lower-precision kernel anywhere | [VERIFIED] |
| Factorizations | blocked right-looking Cholesky (block 32 or 64; plain loops for n ≤ 40), LU as the Gauss-Jordan forward phase, blocked inverse, triangular solves, `syrk` | [VERIFIED: R10 §4.7] |
| Relaxed mode | Accelerate `cblas_dgemm`, `cblas_dsyrk`, `dpotrf`/`dpotrs`, `dgesv`, above size thresholds, trusted only away from the singularity boundary | [VERIFIED: spec "Relaxed numerics"] |
| Speed (one P-core) | reproducible GEMM 24.6–25.0 GFLOP/s (89% of the no-FMA NEON peak); its own FMA kernel 47–48; Accelerate (AMX) 352–378 | [VERIFIED: R10 §4.7] |

Algorithm choice by matrix properties happens above the kernels: the checker's chain planner (`crates/sky-types/src/chain.rs:555`) picks `Qᵀ`, division, Cholesky or LU by tag, and SIR passes add `syrk`, `diag.mul` and `matmul[ta]`.

An f64 matrix-vector product was also measured as a reference point for bandwidth: 200 GEMVs on a 2048×2048 matrix (33.5 MB per pass) took 0.318 ms each on 10 threads, about 105 GB/s [EXPERIMENTAL-quality: R1 §5.5; load average about 4.3; part of the matrix may have been served from the system-level cache]. At 8 bytes per weight against about 0.56 for a 4-bit format, an f64 path moves about 14× more bytes per weight than a 4-bit runtime. Nothing in `sky-linalg` is on the inference path; [Part III](part03-current-performance.md) §3.5 explains its own deficits.

---

## 9. `sky-hw` and `sky-tune`: CPU-only today, but the database is reusable

**`sky-hw` detects** (`crates/sky-hw/src/lib.rs:30-70`): architecture, OS and model; SIMD flags (NEON; SSE…AVX-512 and FMA on x86); physical, logical, P- and E-core counts; L1d, L2 and L3 sizes, cache line, memory and page size. It reads `sysctl` on macOS and `/proc`, `/sys` on Linux, with no `unsafe` code. On the development machine: Apple M2 Pro, NEON, 10 cores (6P + 4E), L1d 128 KiB, L2 16 MiB, L3 unknown, 16 GiB, fingerprint `448c927493c66f47`. The fingerprint is FNV-1a over architecture, model, features and cache sizes, deliberately excluding core counts and memory (`lib.rs:76-104`). [VERIFIED: R1 §5.7]

**Not detected:** the GPU (cores, Metal family, features), memory bandwidth, AMX/SME, the Neural Engine. [VERIFIED]

**A negative finding.** `sky hw` prints "10 cores (6 performance): parallel kernels use 6 threads" from `HardwareInfo::kernel_threads` (`crates/sky-hw/src/lib.rs:135`; text at `report.rs:92`), but no crate outside `sky-hw` calls `kernel_threads`. `sky-linalg` (`threads.rs`) and `sky_ir::parallel::Config::default_threads` (`crates/sky-ir/src/parallel.rs:267-275`) use `available_parallelism()`, which is 10 on this machine, E-cores included; `sky check --explain` says "runs on 10 threads". [VERIFIED: R1 §5.7]

**`sky-tune`** (`crates/sky-tune/src/{lib.rs,db.rs}`):

- A `Tunable` is a kernel name, a size class, a grid of integer parameters (the first is the default) and a work measure.
- `sky tune` times each candidate in rotating rounds within a 500 ms budget and scores by median; a winner must be at least 3% faster than the default.
- The database is data-only text (`sky-tune-db 1`, then `result fp=… kernel=… size=… params=k:v,…` lines), keyed by (fingerprint, kernel, size class), stored at `~/Library/Caches/sky/tune.db` or `SKY_TUNE_DB`, bounded to 1 MiB and 4,096 records, written atomically, validated line by line.
- Registered tunables: `linalg_gemm_{le64,le256,le1024,gt1024}` (`mc`, `kc`, `nc`), `linalg_cholesky_*` (`nb`), `linalg_threads`, and the interpreter thresholds.
- **"Tuning changes speed, never results"** is a normative guarantee (`spec/v0.1.md:1022`).

[VERIFIED: R1 §5.7; spec "Hardware and tuning"]

Both crates carry over ([Part X](part10-autotuner.md)): the database format and the ≥ 3% rule fit Metal tile and threadgroup parameters as they are. What changes: the fingerprint must separate an M2 Pro from an M5 Max (GPU family, core count, neural accelerators, measured bandwidth); the 4,096-record bound may be too small once kernels × shapes × verify widths × chips are tuned [LIKELY: an estimate of about a thousand records per model and chip, not counted]; and the "never results" guarantee does not hold for GPU tiles whose reduction order differs (§11).

---

## 10. Cost model, cost contracts and `--explain`

**The cost model** (`sky_ir::cost::analyze`, `crates/sky-ir/src/cost/mod.rs:1-80,236`; weights in `cost/ops.rs`) gives every function its flops, bytes read, bytes written, allocations and allocated bytes, each an interval `[lo, hi]` of polynomials in the symbolic sizes (`sky_types::poly::Poly`: exact rational coefficients, at most 256 terms and degree 64). Weights: 1 flop for `+ − × ÷` and cheap math functions, 20 for transcendentals, `2nkm` for a product, `2n³` for an inverse, `n³/3 + 2n²k` for a Cholesky solve; a fused kernel costs its body per entry. Loops count when they are a `for range` with a nameable bound; any other loop is "unbounded", with the loop's span as the reason. It runs on the module after the pipeline. [VERIFIED: R1 §5.8; `docs/design/ir.md`, "The cost model"]

**Contracts.** A `def` header may say `cost flops <= 2*N^3`, with measures `flops`, `bytes_read`, `bytes_written`, `bytes`, `allocations` and `allocated_bytes`. `sky check` proves a line when `hi` is within the bound for every size ≥ 1, refutes it with a witness point (an error), or warns. A run-time time budget is deliberately excluded ("time is noise"). [VERIFIED: spec "Performance contracts"]

**`--explain`.** There is no `sky explain` subcommand. `sky check --explain` merges, in source order: the checker's notes (`LA-CHAIN-001`, Cholesky, ODE structure); pass explanations (`FUSE-001`, `GVN-001`, `LICM-*`, `SPEC-*`, `POW-CHAIN-001`, `LA-*`, `EG-COST-001`, `DCE-*`, `CONTRACT-*`); parallelism notes (`PAR-001…003`); buffer reuse (`MEM-001`); relaxed-numerics notes (`NUM-001/002`); a per-function cost report; and citations of tuned values (`crates/sky-cli/src/frontend.rs:157-205`, `crates/sky-cli/src/ir.rs:158-168,230-445`). [VERIFIED: R1 §5.8]

What D12 needs on top ([Part XXII](part22-performance-model.md), [Part XXIII](part23-benchmarks.md)): bytes per dtype instead of `ENTRY_BYTES = 8`; a "per decode token" scope for `cost bytes_per_token <= 18.2 GB`; predicted against measured time per kernel; and a `sky explain` command. The method (sound polynomial bounds with sites that name the costly operation) carries over unchanged.

---

## 11. Numerics policy, and where it conflicts with GPU inference

**Reproducible by default** (`spec/v0.1.md:639-663`): every engine calls the same kernels; products sum in naive `k` order, dots and norms in a fixed 8-lane tree; no FMA in kernels or in Cranelift code; threads split outputs only. The spec's claim is "the same program prints the same bits everywhere" (`spec/v0.1.md:640`). For transcendental functions nothing guarantees it across operating systems, because `sky_interp::math` calls the platform libm ([Part III](part03-current-performance.md) §3.7). [VERIFIED in code; LIKELY that some results differ, R10 §4.9c]

**Relaxed** (`spec/v0.1.md:664-699`): opt-in by `--numerics=relaxed` or `numerics = "relaxed"` in `sky.toml`, never by environment variable or per-file pragma, because "the mode is a property of a run". It permits FMA, reassociated reductions (16 partial sums), Accelerate on macOS, and thread-count-dependent sums at about 10^6 terms or more. It still guarantees units and shapes, finite results, the same errors in the same order, and agreement within 1e-12 in tests. It announces itself on stderr. The REPL is always reproducible. [VERIFIED]

**Finiteness** (`spec/v0.1.md:978`): "a Sky number is always finite". Division by zero, overflow and non-real powers are runtime errors at the failing expression. The checks sit after every operation in the interpreters, inline after each operation in native scalar code (a compare and a branch), and deferred to one test per loop in fused kernels (NaN accumulation). Measured in fused kernels: about 0% on memory-bound kernels, about +30% on a compute-bound six-operation chain with a division [VERIFIED: R1 §6.2, `sky-jit` `kernels` bench, EXPERIMENTAL-quality load].

**Three engines, one output** (`spec/v0.1.md:914-927`): identical means the same stdout, the same stderr (message, span, labels, notes) and the same exit status, with one intended exception for resource-limit messages. Mechanisms: one value layer, `operand_spans`, deoptimization instead of divergence, failure-order rules for `fused` ops and parallel chunks, and AOT executables that carry the SIR interpreter. [VERIFIED: R1 §6.3]

### 11.1 Conflicts with GPU inference

| Scientific-Sky rule | Stated in | Why tensor programs on a GPU cannot keep it | What replaces it (D6, D8; [Part VI](part06-architecture.md)) |
|---|---|---|---|
| Same bits on every machine; no FMA; fixed summation order | `spec/v0.1.md:639-663` | `simdgroup_matrix` and Metal 4 tensor ops accumulate in a hardware-defined order; FMA is the native operation; bandwidth needs split-K and tree reductions | contract `inference`: fp32 accumulation, FMA allowed, a fixed order chosen by the kernel configuration; bit-reproducible run to run per (chip, OS, Sky build, tuning entry); batch- and chunk-invariant (hybrid DeltaNet prefill excepted, VII-M7) |
| A Sky number is always finite | `spec/v0.1.md:978` | attention masks and softmax use −inf; a per-operation check is not free on the GPU | −inf allowed inside kernels; a finiteness check on outputs at layer boundaries is a validation mode |
| No broadcasting between arrays | `spec/v0.1.md:608` | bias add, norm weights, masks and per-row softmax broadcast | explicit broadcast views in TIR; on the surface, broadcasting only by matching size names (D8) |
| Engines byte-identical | `spec/v0.1.md:914-927` | an f16 GPU kernel cannot match an f64 CPU oracle | deterministic per (device, build, tuning entry); tolerance against an f32 reference; greedy-token agreement with mlx-lm and Hugging Face |
| Tuning changes speed, never results | `spec/v0.1.md:1022` | two tile configurations of one GPU kernel sum in different orders | the tuning entry becomes part of the reproducibility key; `--numerics=portable` restricts tiles to one canonical order (§17) |
| The numerics mode is a property of a run, set only by flag or manifest | `spec/v0.1.md:664-699` | D6 attaches `inference` to f16/bf16/quantized storage as well as to `--numerics=inference` | open: whether storage types imply the contract, or only the flag does ([Part VII](part07-tensor-ir.md)) |

[VERIFIED as policy: spec lines cited; LIKELY as conflicts: R1 §9.4]

### 11.2 Other policies the inference work touches

| Policy | Stated in | Where the inference work meets it |
|---|---|---|
| Caches use a data-only format, no pickle-style serialization (non-negotiable 3) | `spec/v0.1.md:1036` | Metal binary archives and pipeline caches, an AOT `.skybin`, and a native-code cache (Part III, F8) all hold compiled code. SPINE D19 now decides this for the tensor stack: implicit caches hold validated data only, and binary archives appear only inside explicit `sky compile` outputs (`.skybin`), which are build artifacts ([Part XI](part11-metal-backend.md) §8.2, [Part XVIII](part18-ttft.md)). The native-code cache (F8) still needs an explicit decision from the project's security owner |
| Imports never resolve symbolic links below the project root | spec "Imports and the project root" | the Hugging Face cache stores each snapshot file as a symbolic link into `blobs/` (seen on the development machine: `snapshots/<rev>/model.safetensors -> ../../blobs/<sha256>`); `sky-hub` must not reuse the import rule for model files |
| Arrays are never empty; `Poly` comparisons hold for sizes ≥ 1 | spec "Arrays"; `crates/sky-types/src/poly.rs` module docs | a KV cache is empty before the first token; a prompt-cache hit can have length 0. TIR either keeps sizes ≥ 1 by construction or extends `Poly`'s domain ([Part VII](part07-tensor-ir.md)) |

[VERIFIED: the policies and the cache layout; LIKELY: the consequences]

---

## 12. Tests and CI

| What | Count |
|---|---:|
| `#[test]` functions in the workspace (static grep) | 2,628 |
| … largest: `sky-ir-interp` 300, `sky-opt` 289, `sky-cli` 256, `sky-types` 244, `sky-ast` 228, `sky-interp` 211, `sky-lower` 207, `sky-ir` 187 | |
| … smallest: `sky-tune` 31, `sky-jit` 24, `sky-hw` 15, `sky-rt` 2, `sky-derive` 0 | |
| files with `proptest!` blocks | 18 |
| UI goldens (`tests/ui/**/*.sky`, each with `.stderr`) | 164 |
| IR golden programs (`tests/ir/*.sky`; 43 `.sir` files plus `.lower.sir` and `.explain`) | 22 |
| fuzz targets: lex, parse_module, sir_parse, manifest, differential | 5 |

[VERIFIED: R1 §6.3]

**Where engine equality is enforced:**

- `crates/sky-cli/tests/run_repl.rs`: `engines_agree` runs `--engine={ast,sir,native}` and compares bytes;
- `crates/sky-cli/tests/examples.rs`: every `examples/*.sky` must print its `.out` file on every engine;
- `crates/sky-ir-interp/tests/differential.rs` (2,775 lines) and `fuzz_differential.rs`: the fuzz generator and oracle on 3,000 fixed seeds per `cargo test`, AST against SIR against native, lowered and optimized, with SLP off, auto and always (`fuzz/harness/differential.rs:4055-4093`);
- `crates/sky-cli/tests/build.rs`: AOT executables against `--engine=native`;
- `sky-bench` fails if its four engines disagree; `crates/sky-opt/tests/fuzz_*.rs` require the same output before and after each pass.

`sky-jit` has only 24 tests of its own: its correctness rests on the differential tests in other crates [VERIFIED: counts above]. The `GATE_*` constants that steer the fuzz generator around known bugs are all `false` (`fuzz/harness/differential.rs:415-443`): every known bug is fixed. There is no gate script in the repository (`tools/` holds only `check_release_symbols.sh`). [VERIFIED: R1 §6.3]

**CI** (`.github/workflows/ci.yml`): `cargo test --workspace --locked` on a **self-hosted Linux runner only**, with the comment "GitHub-hosted minutes ran out on this private repo; macOS is checked locally before every push"; plus `rustfmt`, `clippy -D warnings`, `cargo-deny`, a release-symbol scan, a Unicode-table check, 60 s per fuzz target on every pull request and a nightly fuzz run. Toolchain Rust 1.85.1. [VERIFIED]

Consequence: nothing Metal-specific can be tested in CI today. D14 asks for a self-hosted macOS runner, separate from the development laptop (task II-T7).

---

## 13. The ten in-flight worktrees

### 13.1 State

`git worktree list` shows ten worktrees under `.claude/worktrees/agent-*`, on branches `worktree-agent-*`, all at `b4fa49c` (`main~1`; `a99e7e3` only touched the README, the guide and two help strings). `agent-afa55e105ab8ee772` is locked. **Every branch has zero commits**: `git rev-list --count main..<branch>` is 0 for all ten, and all of the work is uncommitted in the worktree directories. The files were last modified between 17:16 and 17:20 on 2026-10-01. [VERIFIED: R1 §8.1; re-checked for this chapter, with the same diff statistics]

### 13.2 What each contains

| Worktree | Feature | Tracked diff | New files | New `#[test]` | Spec/docs | What it adds | Tests compile? |
|---|---|---|---|---:|---|---|---|
| `a0952514…` | **arrays you can build** | 23 files, +610/−151 | 5 files, 1,732 lines | 5 | no; code cites a spec section "Building and changing arrays" that does not exist | `v[i] = x`, `A[i,j] += x`, `v[a:b] = w` as SSA functional update (`store`), in place when the old value dies; `ones`, `full`, `linspace`, `arange`, `concat`, `append`; ops `Store{axes}`, `Fill{ones}`, `Linspace`, `Arange`, `Concat{append}` through checker, lowering, verifier, both interpreters and the JIT runtime | yes |
| `abf8af99…` | **sizes from variables** | 10 files, +1,081/−149 | `sky-types/src/sizes.rs`, 284 lines | 0 | no | `Linear` sizes `a·root + b` in **one** size variable (`zeros(n)` with `n = 21` is `Vector[21]`; `v[1:N−1]` is `Vector[N−2]`); rigid run-time size variables | yes, but checker-only: no lowering, SIR or engine changes, so the engines cannot run what it accepts |
| `a3be8cf3…` | **reductions** | 29 files, +408/−21 | 13 files, 1,117 lines | 14 | spec +29 lines, `ir.md`, README | `sum`, `prod`, `mean`, `var`, `pvar`, `std`, `pstd`, `min`, `max`, `argmin`, `argmax`, `median`, `cumsum`, **vectors only**; one op `Reduce(Reduction)`; `sum` uses `dot`'s fixed 8-lane order; one runtime twin `sky_interp::reduce` for all engines | yes |
| `a383df0a…` | broadcasting | 22 files, +479/−112 | 1 file, 241 lines | 9 | spec, `ir.md`, README | number ⊕ vector under `+ − * /`; number ⊕ matrix stays `*` and `/` only (`A − λ` reads as `A − λI`); array–array broadcasting stays forbidden | yes |
| `a1dadc02…` | readable output | 40 files, +704/−122 | 4 files, 1,143 lines | 4 | no | f-strings with format specs, `print(sep=, end=)`, named derived units in output | **no** |
| `ad1ea58e…` | tuples | 34 files, +941/−128 | 3 files, 535 lines | 0 | no | `a, b`, unpacking, `t[0]`, tuple types, multiple returns; elements are numbers, arrays or Bools, not nested | **no** |
| `a0bbbe93…` | data in | 21 files, +359/−32 | 3 files, 1,417 lines | 7 | no | `read_csv` with run-time sizes; `random`, `uniform`, `normal`, `seed` on SplitMix64 with a portable `ln`; command-line inputs `sky run prog.sky L="2.5 m"` | yes |
| `add6061b…` | units | 19 files, +691/−90 | 2 files, 414 lines, plus a stray `.t/` directory | 4 | no | `unit name = value unit`; °C/°F readings vs differences; imperial and lab units | yes |
| `a7e29e8b…` | linear algebra | 21 files, +525/−16 | 4 files, 1,337 lines | 9 | no | `eigvals`, `eigvecs` (symmetric, deterministic sign and order; generalized `Kx = λMx`), `outer`, `solve_tridiagonal`; op `OpKind::Linalg` | yes |
| `afa55e10…` (locked) | ODE events | 18 files, +905/−58 | 10 files, 1,145 lines | 21 | spec, README | events located by Illinois regula falsi on the dense interpolant; dense output `y(t)`; a new example | **no** |

[VERIFIED: R1 §8.2, §8.4; diff statistics re-checked for this chapter]

**Compilation** (R1 §8.4; `cargo check --workspace --tests --locked`, toolchain 1.85.1, a separate target directory with the `sky-*` fingerprints cleared before each worktree): the production code of all ten compiles. Three test targets do not, each because an exhaustive AST match in a test file was not updated:

| Worktree | Errors | Failing targets |
|---|---:|---|
| readable output | 2 | `crates/sky-types/tests/expr_types.rs:40`; `fuzz/harness/parse_module.rs:311` (`ExprKind::FString`, `ExprKind::Keyword`) |
| tuples | 4 | `expr_types.rs:40,110`; `parse_module.rs:311,396` (`ExprKind::Tuple`, `AssignTarget::Tuple`) |
| ODE events | 1 | `expr_types.rs:198` (`SolveOption::Until`) |

No worktree's test suite has been run [VERIFIED: R1 §8.4]: "compiles" is a weak completeness signal.

### 13.3 The three prerequisites for writing tensors in Sky

D17 names three worktrees that tensor authoring in Sky source needs. Each is a partial fix:

| Worktree | What tensor code needs from it | Gap |
|---|---|---|
| **arrays you can build** | in-place `store` (KV-cache append, building RoPE tables); `arange` (positions); `concat` | no spec text, 5 tests, no goldens. In native code a dying operand is moved when its refcount is 1 (`sky_rt_push_ref_move`, added in the worktree's `crates/sky-jit/src/runtime.rs`), so writes are in place, but each `store` is still dispatched through `generic`, the boxed path that costs 148 ns per `x[i]` [LIKELY: same code path; not measured] |
| **reductions** | softmax, RMSNorm, argmax sampling | vectors only; tensors need reductions along a named axis (`mean(x * x, over: D)`, D8) |
| **sizes from variables** | sequence and KV lengths are run-time sizes | checker-only, no tests; `a·N + b` in one variable cannot express `heads · head_dim`; it introduces a size representation (`Linear`) beside `Poly`, which D1 uses for TIR |

[VERIFIED: worktree contents; LIKELY: the needs, R1 §8.3]

The three overlap. Files modified by more than one of them (`git status` in each worktree):

| Pair | Shared files |
|---|---:|
| arrays ∩ reductions | 12 (`sky-ir/src/{op,parse,verify,cost/ops}.rs`, `sky-jit/src/{module,runtime}.rs`, `sky-ir-interp/src/machine.rs`, `sky-interp/src/{eval,lib}.rs`, `sky-lower/src/expr.rs`, `sky-types/src/{check,lib}.rs`) |
| arrays ∩ sizes | 4 (`sky-types/src/{array,check,flow,lib}.rs`) |
| sizes ∩ reductions | 2 (`sky-types/src/{check,lib}.rs`) |

[VERIFIED: for this chapter]

Landing them is therefore sequential work with conflict resolution, not three merges.

### 13.4 Medium and low relevance

- **Broadcasting** (medium): bias add, norm weights, masks and per-row softmax need array–array broadcasting, which the language forbids by policy. The worktree keeps that rule; D8's broadcasting by matching size names is a different design.
- **Tuples** (medium): returning `(logits, kv)`.
- **Data in** (medium): the deterministic SplitMix64 generator is useful for CPU-side reproducibility; CSV is useless for weights of about 18 GB. GPU sampling uses a counter-based generator (Philox, D9).
- **Low or none:** readable output (useful for tooling), units, linear algebra (`outer` is marginal), ODE events.

[LIKELY: R1 §8.3]

### 13.5 What to do with them

Following D17: commit every worktree's state to its branch now (task II-T1), so that a `git worktree prune`, a disk cleanup or an agent restart cannot lose about 16,000 lines of work (6,703 lines inserted into tracked files plus 9,365 lines in new files, from the table in §13.2); fix the three stale test targets while committing, which is a few match arms each (II-T2); land the three prerequisites (II-T3 to II-T5); park the other seven, branches kept, until after gate G4. The recommended landing order is reductions (most complete: spec, goldens, IR goldens, CLI tests), then arrays (largest overlap with reductions), then sizes, re-based on `Poly` (§17).

---

## 14. What carries over to the inference work

| Component | Verdict | Reason | Lands in |
|---|---|---|---|
| `sky-diag` (`Span`, `Diagnostic`, renderer) | **reuse as is** | no dependencies; every new crate needs spans and diagnostics | all new crates |
| `sky-syntax`, `sky-ast`, editions | **reuse; extend after G4** | the second front door (D7) needs tensor syntax. "A lone name is always a unit" (`spec/v0.1.md:557`) means a dtype cannot follow a shape as a bare word, hence D8's `in f16` | `sky-ast` |
| `sky-project` | **reuse the handle-based reading; not the import rules** | safe opening by handle with owner and mode checks fits model paths; the no-symlink rule does not fit the Hugging Face cache (§11.2) | `sky-hub`, `sky-llm` |
| `sky_types::poly::Poly` | **reuse, after a small extraction** | exact symbolic-size polynomials with sound comparison, already the cost model's currency. But `poly.rs` (728 lines) imports `sky_ast` to parse contract bounds, and `sky-types` depends on `sky-linalg`, so `sky-tir` would link the checker; and its domain is sizes ≥ 1 | `sky-tir` (D1) |
| `sky-types` checker (shape unifier, diagnostics) | **reuse for Sky-source authoring only** | units and tags are irrelevant to LLMs; there are no dtypes | `sky-types`, after G4 |
| `sky-ir` infrastructure: arenas with `u32` ids, total verifier, round-trip text format with a fuzz target, `PassManager` (`verify_each`, `print_after`), `Explanation`, effect inference, `Lifetimes` | **reuse as patterns; share `Explanation` as code** | `Pass::run` takes `&mut sky_ir::Module` (`crates/sky-ir/src/pass.rs`), so the manager is not generic. The pieces are small: `pass.rs` 348 lines, `print.rs` 364, `parse.rs` 1,354, `verify.rs` 1,871. Sharing the `Explanation` type lets `sky explain` merge SIR and TIR explanations without conversion | `sky-tir`, `sky-kernel` |
| `sky-ir` op set and `Type` | **replace for tensors; extend by one op** | f64 only, rank ≤ 2 at run time, no memory model, exhaustive matches (§5.8); D1 adds `tensor.call @graph` | `sky-ir` |
| `sky_ir::cost` (intervals of `Poly`, sites) | **reuse the method** | `ENTRY_BYTES = 8`, no dtype, no per-token scope; the method maps directly onto roofline estimates where bytes dominate | TIR cost model ([Part XXII](part22-performance-model.md)) |
| `sky-opt` passes | **mostly replace** | `gvn`, `licm`, `dce`, `fold` are coded against `OpKind` and Sky's failure order; `fuse` is a precedent, not an implementation | TIR passes ([Part VIII](part08-graph-optimizer.md)) |
| `sky-egraph` | **partial reuse, later** | the `egg` integration, contract classes and cost hook carry over; the 27 rules do not | block-level search (G9+) |
| `sky-interp`, `sky-ir-interp` | **do not use for tensors** | f64 `Value` with a 112-byte `Dimension`, shared by every engine; routing tensors through it drags scientific semantics into the hot path | a separate TIR reference interpreter (f32/f64) |
| `sky-jit`, `sky-rt` | **keep for CPU Sky; not a GPU path** | Cranelift targets the CPU and the runtime operates on `sky_interp::Value`; the "embed the IR and its interpreter" pattern of `sky build` is a precedent for `.skybin` | — |
| `sky-linalg` | **irrelevant to tokens per second** | f64 CPU kernels; its "fix the order, then make that order fast" rule informs D6 | — |
| `sky-hw` | **reuse and extend** | add GPU family, core count, neural accelerators, measured bandwidth; the fingerprint must separate chips | [Part X](part10-autotuner.md), [Part XI](part11-metal-backend.md) |
| `sky-tune` | **reuse** | data-only database keyed by (fingerprint, kernel, size class) with integer parameters and a ≥ 3% rule; raise the 4,096-record bound if needed; restate its "never results" guarantee for tensor kernels | [Part X](part10-autotuner.md) |
| cost contracts, `--explain`, stable rule ids | **reuse the concept** | D12 adds `cost bytes_per_token` and a `sky explain` command | `sky-cli` |
| differential-testing discipline (goldens, fuzzed oracles) | **reuse the discipline** | replace byte identity by deterministic-per-device and tolerance oracles | every new crate |
| `sky-bench compare` | **reuse the pattern** | subprocess runs, medians, spreads, floors, a generated `RESULTS.md`; needs slope mode ([Part III](part03-current-performance.md)) | `sky bench llm` ([Part XXIII](part23-benchmarks.md)) |
| `sky-cli` | **reuse and extend** | `Engine` enum at `crates/sky-cli/src/run.rs:67`; subcommand match at `crates/sky-cli/src/main.rs:70` | `run hf://`, `compile`, `bench llm`, `explain` |
| `deny.toml` policy | **applies as is** | crates.io only; Apache-2.0, MIT, BSD, ISC, Zlib, Unicode licences; any Metal binding must pass it | [Part XI](part11-metal-backend.md) |
| units, uncertainty, ODEs, tags, Cholesky/LU/inverse, `accuracy`, `exact`/`roundoff`, AD, REPL, tree-walker | **irrelevant to year-1 inference** | LLM tensors are dimensionless; training is out of scope in year 1 | — |

[VERIFIED: the facts in the Reason column, cited in §§1–13; LIKELY: the verdicts]

**How much carries over.** Code that carries over directly is about 10–20% of the 75,005 production lines: the front end, `sky-diag`, `sky-project`, `sky-tune`, `sky-hw`, the e-graph scaffolding and the IR infrastructure patterns [LIKELY: R1 §9.2]. The performance machinery (`sky-linalg`, `sky-jit`, fused kernels) does not transfer to Metal, f16 or int4. Nothing should be deleted: these subsystems are the scientific product.

**Why these facts lead to D1.** Extending SIR to carry tensors would mean, in order: a dtype and a memory model in a type system that has neither (§3, §5.2); arrays above 2^24 entries and of rank above 2 in a value layer shared by three engines and the optimizer (§1.5, §3.3); about twenty operations, each touching most of the compiler (§5.8); and a rule, byte-identical engines, that a GPU cannot meet against an f64 oracle (§11.1). A sibling IR reuses the parts that are generic (spans, diagnostics, `Poly`, the tuning database, the verifier and pass-manager patterns) and leaves scientific Sky's semantics untouched, at the price of one SIR operation. [Part VI](part06-architecture.md) states the decision and its amendment of ROADMAP principle 2.

---

## 15. Implementation tasks

**II-T1. Commit the ten worktrees to their branches.**
- Files: `.claude/worktrees/agent-*` on branches `worktree-agent-*`.
- Deliverable: one commit per branch, with a message naming the feature, its state from §13.2 and any stale test target; the branches pushed to the remote. Remove the stray `.t/` directory in the units worktree first.
- Acceptance: `git rev-list --count main..worktree-agent-<id>` ≥ 1 for all ten; `git -C .claude/worktrees/agent-<id> status --porcelain` is empty; `git diff --shortstat main...worktree-agent-<id>` matches the tracked-diff column of §13.2 plus the new files.

**II-T2. Repair the three stale test targets.**
- Files: `crates/sky-types/tests/expr_types.rs` (lines 40, 110, 198) and `fuzz/harness/parse_module.rs` (lines 311, 396) in the readable-output, tuples and ODE-events worktrees.
- Deliverable: match arms for `ExprKind::FString`, `ExprKind::Keyword`, `ExprKind::Tuple`, `AssignTarget::Tuple` and `SolveOption::Until`.
- Acceptance: `cargo check --workspace --tests --locked` exits 0 in all ten worktrees.

**II-T3. Land reductions.**
- Files: the reductions worktree (`crates/sky-interp/src/reduce.rs`, `crates/sky-types/src/reduce.rs`, `crates/sky-units/src/reduce.rs`, `OpKind::Reduce` in `crates/sky-ir/src/op.rs`, `tests/ir/reductions.*`, `crates/sky-cli/tests/reductions.rs`).
- Deliverable: a pull request rebased on `main`, with the spec text it already has.
- Acceptance: `cargo test --workspace --locked` passes on Linux CI and on macOS; `engines_agree` holds on `tests/ir/reductions.sky`; 60 s of every fuzz target finds nothing new.

**II-T4. Land arrays you can build.**
- Files: the arrays worktree (`crates/sky-types/src/array_build.rs`, `crates/sky-interp/src/build.rs`, `crates/sky-interp/src/build_eval.rs`, `crates/sky-lower/src/array_build.rs`, `crates/sky-ir/src/verify/build.rs`, `sky_rt_push_ref_move` in `crates/sky-jit/src/runtime.rs`), plus a new spec section "Building and changing arrays" in `spec/v0.1.md`.
- Deliverable: the spec section; IR goldens `tests/ir/array_build.*`; UI goldens for the new errors; a differential-fuzz generator extension for `store` and `concat`.
- Acceptance: engines agree on the new goldens; a loop `v[i] = x` over a 10^4-entry vector allocates nothing per iteration on SIR (a new case in `crates/sky-bench/tests/allocations.rs`); the native engine moves the operand when its refcount is 1 (a unit test on `sky_rt_push_ref_move`).

**II-T5. Re-base sizes from variables on `Poly` and carry it through the engines.**
- Files: `crates/sky-types/src/{shape.rs,sizes.rs}` (add `ShapeDim::Expr(Poly)` in place of the one-variable `Linear` form), `crates/sky-lower/src/ctx.rs:263`, `crates/sky-ir-interp`, `crates/sky-jit`.
- Deliverable: run-time sizes that lower and run; sizes that are products of variables.
- Acceptance: `zeros(n)` with a run-time `n` runs on all three engines; `v[1:N-1]` types as `Vector[N-2]`; a function on `Matrix[H*D, N]` unifies with an argument `Matrix[8*128, N]`; the worktree's examples pass `engines_agree`.
- Schedule: with the Sky-source front door, after G4 (§17).

**II-T6. Make `sky hw` tell the truth about threads.**
- Files: `crates/sky-hw/src/lib.rs:135`, `crates/sky-hw/src/report.rs:92`, `crates/sky-linalg/src/threads.rs`, `crates/sky-ir/src/parallel.rs:267-275`.
- Deliverable: either the kernels use `kernel_threads()` by default, or `sky hw` prints the count they actually use. Choose by measuring 6 against 10 threads (Part III, F9).
- Acceptance: `sky hw` and `sky check --explain` name the same thread count; the `sky-linalg` and `par` benchmarks do not regress.

**II-T7. Add a macOS job to CI (with D14).**
- Files: `.github/workflows/ci.yml`; a self-hosted macOS runner separate from the development laptop.
- Deliverable: `cargo test --workspace --locked` on macOS for every pull request, the job that later runs the Metal tests of `sky-metal`.
- Acceptance: a deliberately failing macOS-only test fails CI; the existing Linux jobs are unchanged.

**II-T8. Extract `Poly` and share `Explanation` before `sky-tir` starts.**
- Files: `crates/sky-types/src/poly.rs` into a new dependency-light crate (for example `crates/sky-poly`, depending only on `sky-diag` and `sky_units::Rational`), with the contract-bound parser (`sky_ast::Expr` → `Poly`) staying in `sky-types`; `Explanation` and `RuleId` from `crates/sky-ir/src/pass.rs` into a module `sky-tir` can depend on without the SIR op set.
- Deliverable: both moves, with re-exports so that no SIR caller changes.
- Acceptance: SIR goldens and `--explain` goldens are byte-identical; `cargo tree -p sky-tir` shows neither `sky-linalg` nor `sky-ast`; `crates/sky-types/tests/poly_properties.rs` passes unchanged. [Part VII](part07-tensor-ir.md) decides whether sizes may be 0.

---

## 16. Risks and negative results

**Negative results** (all VERIFIED, cited above):

- `Repr` exists, but nothing chooses or reads it (§3.1).
- The `Tensor` shape word checks and lowers, but no rank-3 value can exist (§3.5).
- The SIR levels L2–L4 exist only in `docs/design/ir.md` (§5.5).
- `specialize_for` and `SpecializationCache` have no caller; the default `MatMulPolicy` chooses nothing; no pass enables the e-graph's `Relaxed` or `Fast` contracts (§5.6, §5.7).
- `sky hw` claims 6 kernel threads while the kernels use 10 (§9).
- `sky-jit` has 24 tests of its own; there is no gate script; CI never runs on macOS (§12).
- All ten feature branches have zero commits (§13).

**Risks:**

| Risk | Likelihood | Consequence | Mitigation |
|---|---|---|---|
| uncommitted worktree work is lost (prune, cleanup, restart) | LIKELY over months if nothing is done | about 16,000 lines of features lost | II-T1, now |
| the three prerequisites conflict when landed | VERIFIED overlap of 12 files between two of them | weeks of rebase work during G1–G4 | land sequentially in the order of §13.5, each with full tests |
| someone extends `sky_interp::Value` for tensors to "reuse the engines" | LIKELY temptation, because it looks like the shortest path | destabilizes three engines and the optimizer; drags units and the finiteness rule into the hot path | D1; review rule: tensor code never imports `sky_interp` |
| D17's freeze slips and scientific features land during G1–G4 | LIKELY without an explicit rule | review load and merge conflicts on the critical path | the freeze is stated in [Part XXV](part25-roadmap-12-months.md) |
| the `Tensor` word misleads users | LIKELY: the spec shows `M: Tensor[B, T, D] kg` | programs that check but can never run | a check-time note that rank-3 values are not supported yet, until D8 lands |
| the reuse estimate (10–20%) is optimistic | EXPERIMENTAL: decided when `sky-tir` reaches G2 | more new code than planned | plan `sky-tir` as new code; count reused lines at G2 |
| compiled-code caches conflict with non-negotiable 3 | LIKELY | a security objection late in G6 (`.skybin`, pipeline caches) | decided for the tensor stack by SPINE D19 (data-only caches; archives only in an explicit `.skybin`); decide F8's native-code cache before G5 (§11.2) |

---

## 17. Disagreement with the spine

Three points refine the spine; none reverses a decision.

1. **D17, "sizes from variables".** The spine says to finish and land it with the other two prerequisites. As written it cannot serve tensor authoring: it is checker-only with no tests, and it represents sizes as `a·N + b` in a single variable, which cannot express `heads · head_dim`, while D1 makes `Poly` the size representation of TIR. Landing it as is would add a third size representation (`Known | Var`, `Linear`, `Poly`) that the tensor stack does not use. Recommendation: commit it now (II-T1), re-base it on `Poly` and take it through lowering and the engines (II-T5) as part of the Sky-source front door after G4, not during G1–G4. Reductions and arrays can land earlier.
2. **D6, the list of rules that do not carry over.** D6 names three scientific-Sky rules that tensor programs replace (finiteness, no broadcasting, byte-identical engines). There is a fourth: "tuning changes speed, never results" (`spec/v0.1.md:1022`). D6 already makes the tuning entry part of the reproducibility key, which contradicts that guarantee for tensor kernels; the spec text must say so explicitly, or `sky-tune` will be held to a promise its GPU tunables cannot keep. Resolved: the spine now says (D6, "Tuning and results") that under `inference` tuning may change the last bits, so results are bit-reproducible *given* (chip, OS/Metal compiler, Sky build, tuning entry), and that `portable` keeps "tuning changes speed, never results".
3. **D1, reuse of `sky_types::poly::Poly`.** Reuse is right, but `Poly` sits in the checker crate (which links `sky-linalg` and `sky-ast`) and is sound only for sizes ≥ 1. Either extract it (II-T8) or accept the dependency, and decide whether TIR sizes may be 0 before the KV-cache design depends on it.

For the record, the spine's statement that one new SIR operation touched 23 files is consistent with the evidence: the reductions worktree adds exactly one operation and modifies 23 tracked source files in 9 crates (§5.8).
