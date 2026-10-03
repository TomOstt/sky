# R1 — Sky as it exists today: architecture map

Research agent R1. Subject: `~/sky` at `main` = `a99e7e3` ("README that shows what Sky is…", 2026-10-01), working tree clean. Ten unfinished agent worktrees sit on `b4fa49c` (= `main~1`) and are covered in §8.

How I gathered this: I read the code, `spec/v0.1.md`, `docs/guide.md`, `docs/design/{ir,ad}.md`, `README.md`, `ROADMAP.md`, `benches/compare/RESULTS.md`, the examples and the tests. I ran small probe programs with the release binary `~/sky/target/release/sky` (built from `main` at 20:04). I took timings through `benchlock.sh`, and compile-checked the worktrees into a separate target directory. All raw files are under `$S/research/r1/`.

Evidence labels, as the brief defines them:
- **VERIFIED**: backed by `path:line`, our own command output, or an official document.
- **LIKELY**: inferred, with the reason given.
- **EXPERIMENTAL**: needs a test.
- **MOONSHOT**: high risk, high upside.

Paths below are relative to `~/sky/crates/` unless they start with `spec/`, `docs/`, `tests/`, `benches/`, `examples/` or `fuzz/`.

---

## 0. One-paragraph summary

Sky today is a **CPU-only, f64-only scientific language**. Its type checker is the most developed part. It tracks:
- physical dimensions;
- symbolic shapes, with equality only;
- four matrix-property tags;
- cost contracts.

It has three engines that must print byte-identical output: an AST tree-walker, a reference interpreter for SIR (its SSA IR), and a Cranelift JIT/AOT for SIR.

**All three engines share one value layer: `sky_interp::{Value, arith, array, solve, math}`.**
- Arrays are dense, row-major, immutable, rank ≤ 2 and never empty.
- They hold f64 entries, at most 2^24 of them.
- Each `Value` carries a 112-byte `Dimension`.

Native code compiles three kinds of value inline: scalars, Bools, and arrays of at most 16 entries whose shape is known at compile time. Every other array operation is a call into the runtime, which runs the interpreter's array functions. Those functions in turn call `sky-linalg`: reproducible f64 GEMM, Cholesky and LU, with no FMA by default and optional Accelerate. There is no GPU code, no Metal, no f16/bf16/int8, no integer type, no N-d tensor, no binary file I/O, no tokenizer and no notion of device memory. (VERIFIED, details below.)

---

## 1. Pipeline: source → hardware

### 1.1 The real path (VERIFIED)

| Stage | Entry point | Output | Notes |
|---|---|---|---|
| Project discovery | `sky_project::discover_opened` (called at `sky-cli/src/run.rs:110`) | `Discovery{edition, numerics}` | walks up to `sky.toml`; security rules in spec §"Imports and the project root" |
| Lex | `sky_syntax::lex` `sky-syntax/src/lexer.rs:117` | `Lexed{tokens, warnings}` | `INDENT`/`DEDENT`; rejects bidi controls; NFKC identifier checks; max 16 MiB (`sky-syntax/src/lib.rs:37`) |
| Parse | `sky_ast::parse_module` `sky-ast/src/parser.rs:138` | `sky_ast::Module` (+ errors) | Pratt expression parser; never panics |
| Check | `sky_types::check_module_with` `sky-types/src/check.rs:495` | `Checked` `check.rs:409` | units, shapes, tags, chain plans, Cholesky sites, `expr_types: SpanMap<Ty>` (types keyed by span), funcs |
| Driver | `sky-cli/src/frontend.rs:98` `analyze_ir`; `sky-cli/src/run.rs:163` `run_source` | — | lexer warnings plus checker diagnostics, sorted by span |
| Lower | `sky_lower::lower_for_run` `sky-lower/src/lib.rs:360` | `sky_ir::Module` (L1) | uses the checker's types and never re-derives them; derivative functions are added through `sky-ad` |
| Optimize | `sky_opt::default_pipeline().verify_each(true)` `sky-cli/src/ir.rs:209` | optimized SIR | 13 passes, with the verifier run before the first and after every pass |
| Execute: native | `sky_jit::compile_par` `sky-jit/src/module.rs:449`, then `Compiled::run_or_rerun` | stdout | Cranelift JIT, whole module, in memory |
| Execute: SIR | `sky_ir_interp::run_with` `sky-ir-interp/src/lib.rs:149` | stdout | reference interpreter |
| Execute: AST | `sky_interp::run_module_with_derivatives` `sky-interp/src/eval.rs:89` | stdout | tree-walker, the oracle |
| AOT | `sky_jit::aot::compile_object` `sky-jit/src/aot.rs:127` + `libsky_rt.a` + system `cc` | executable | carries the SIR text and the SIR interpreter, so it can deoptimize |

### 1.2 `--engine=auto` selection and fallback (VERIFIED, `sky-cli/src/run.rs:190-268`)

1. Lowering and the pipeline run **before** anything executes (`run.rs:190-216`).
   - A `LowerError` sends the run to the **tree-walker**, silently.
   - Under `--engine=sir` or `--engine=native` the same `LowerError` is a hard error: "cannot run on SIR" / "cannot run natively".
2. If lowering succeeded, native compilation is attempted (`run.rs:219-239`). A `CompileError` falls back to **SIR**, silently under `auto`.
3. Native code that *deoptimizes* does not error. This happens when a value the checker typed `unknown` turns out not to match a declared type at run time. The program then re-runs from the start on the SIR interpreter, and the bytes native code already wrote are skipped (`run.rs:250-258`; `sky-jit/src/module.rs` module docs, "Values whose type is unknown").
4. A SIR verifier failure **before** the first pass counts as "unsupported", so the run falls back to the AST (`sky-cli/src/ir.rs:211-221`). A failure **after** a pass is an internal error.

What does not lower today (`LowerError`) and therefore always runs on the tree-walker:
- `±` uncertainty (`sky-lower/src/expr.rs:93,422`).
- `equation`/`solve` (`sky-lower/src/stmt.rs:248`).
- Attributes (`expr.rs:100`) and `∂`/`∇` operators (`expr.rs:192`).
- Functions used as values (`expr.rs:383,469`).
- Input declarations below the top level (`stmt.rs:332`).
- `for x in <array whose type is unknown>` (`for_loop.rs:261`).
- Arrays whose size is known only at run time (`sky-lower/src/ctx.rs:263`).

Derivatives always run on SIR. Under the AST engine, `sky-derive` lowers the program's `def`s, builds `df/dx` with `sky-ad`, and runs it on the SIR interpreter (`sky-derive/src/lib.rs:1-40`). (VERIFIED)

### 1.3 Diagram — "Sky today"

```
                                    Sky today   (main a99e7e3)
 prog.sky
   │  sky-project: sky.toml (edition, numerics), entry file read via handle (≤16 MiB)
   ▼
 sky-syntax::lex ─────► tokens (+INDENT/DEDENT)  [bidi rejected, NFKC idents, confusables warned]
   ▼
 sky-ast::parse_module ► AST  Module{Stmt} · Expr · TypeExpr{tags, shape, unit}
   ▼
 sky-types::check_module_with ──► Checked{expr_types: SpanMap<Ty>, chains, cholesky_solves, funcs, diags}
   │   sky-units (Dimension = 7 rational exponents, unit table, CODATA, MathFn rules)
   │   shapes: ShapeDim = Known(u64) | Var  (equality unification only)
   │   tags: {Symmetric, PositiveDefinite, Diagonal, Orthogonal} closed lattice (u8)
   │   chain planner: A@B@C@v association + A⁻¹@b → Cholesky/LU/Diag/Qᵀ  (changes rounding,
   │                  so it lives in the checker and BOTH the tree-walker and lowering obey it)
   │   numerics warnings (cond numbers via sky-linalg), uncertainty = run-time only
   │
   ├──[--engine=ast | LowerError]──► sky-interp tree-walker (BTreeMap env, Value) ──► stdout
   │                                   └─ derivative(f,x) ─► sky-derive ─► sky-lower ─► sky-ad ─► sky-ir-interp
   ▼
 sky-lower::lower_for_run (+ sky-ad derivative fns)
   ▼
 SIR L1: typed SSA, arenas, structured regions (if/while/fused), 47 OpKinds, f64 meaning
   ▼  sky-opt::default_pipeline, verify after every pass
   │  specialize → fold → egraph(egg, Reproducible) → gvn → licm → contracts → dce
   │  → powers → fuse → cholesky → transposes → structure → algorithms
   │  analyses (no rewriting): lifetime · parallel · cost · exact · roundoff · effects
   │
   ├──[--engine=sir | native CompileError]──► sky-ir-interp
   │        value table per activation (Vec<Option<Value>>), moves at last use,
   │        in-place elementwise, rayon chunks, fused bodies JIT-compiled (sky-jit::kernel)
   │        once ≥512 elements have run through that op
   │
   └──[--engine=native | auto]──► sky-jit (Cranelift 0.121, opt_level=speed, whole module)
            f64 / i8 / small arrays (≤16 entries, static shape) = inline machine code (+SLP F64X2)
            everything else = handle (u32 index into thread-local table of Rc<Value>)
                              → sky_rt_op(site) ─┐       deopt ⇒ re-run on sky-ir-interp
            sky build: sky-jit::aot ─► .o + libsky_rt.a (sky-rt) ─► cc ─► exe (~5.7 MB)
                                                  ▼
   VALUE LAYER SHARED BY ALL ENGINES: sky-interp::{Value, arith, array, solve, math, vecfn}
                                                  ▼
   sky-linalg: f64 GEMM (Goto/BLIS 4x8 micro-tile, packed, cache-blocked), Cholesky, LU, inverse,
               dot (fixed 8-lane order), syrk, tri-solves; rayon threads; NO FMA by default
               └─[--numerics=relaxed]─► FMA micro-kernels (NEON vfmaq) / Accelerate cblas_dgemm,
                                         dpotrf/dpotrs, dgesv (→ AMX) on macOS
   ▼
 HARDWARE: CPU cores only. f64 scalar, NEON 128-bit (F64X2 in Cranelift; LLVM auto-vectorized in
 sky-linalg), rayon thread pools. No GPU / Metal / ANE / SME. No f32/f16/bf16/int compute.
```

---

## 2. Crates

### 2.1 Size, purpose, key types and dependencies

How the line counts were made: `tokei` is not installed. `src(wc)` is `wc -l` over `src/`. The other columns come from `$S/research/r1/loc.py`:
- **prod** = non-blank, non-comment lines outside test modules and generated tables;
- **tests** = test modules inside `src` plus the crate's `tests/` directory.

Workspace totals:
- 133,545 lines in `src`: 75,005 prod, 17,734 comments, 5,555 blank, 25,474 in-src tests and 10,023 generated.
- 47,606 lines in `tests/` and 988 in `benches/`.
- 182,139 lines of Rust under `crates/`, plus 5,225 in `fuzz/`.

(VERIFIED: `python3 $S/research/r1/loc.py ~/sky/crates`, raw output in `$S/research/r1/loc_main.txt`.)

| crate | purpose | src(wc) | prod | tests (src+tests/) | key types / entry points | workspace deps (Cargo.toml) | external deps |
|---|---|---:|---:|---:|---|---|---|
| sky-diag | spans, diagnostics, renderer, deep-stack helper | 1,745 | 441 | 961+76 | `Span`, `Diagnostic`, `SourceFile`, `with_deep_stack` | — | unicode-ident |
| sky-syntax | lexer, tokens, NFKC/confusables, `Edition` | 14,775 | 1,490 | 2,780 | `lex` (`lexer.rs:117`), `Token`, `TokenKind`, `Kw`, `MAX_SOURCE_BYTES` | diag | unicode-ident |
| sky-ast | AST plus Pratt parser (also `equation`/`solve` syntax) | 5,695 | 4,551 | 101+4,781 | `parse_module` (`parser.rs:138`), `Module`, `Stmt/StmtKind`, `Expr/ExprKind`, `TypeExpr` (`ast.rs:212`), `ShapeAnn`, `DimExpr` | diag, syntax | — |
| sky-units | dimensions, unit table, SI prefixes, CODATA constants, math-function dimension rules | 3,345 | 1,490 | 1,199+844 | `Dimension` (`dimension.rs:109`), `Rational` (`rational.rs:16`), `Unit`, `lookup`, `CONSTANTS`, `MathFn` (`math.rs:37`), `DimUnifier` | — | — |
| sky-types | the checker: units, symbolic shapes, tags, chain planner, flops polynomials, cost syntax, numerics warnings, ODE typing | 16,813 | 11,872 | 1,684+3,773 | `check_module_with` (`check.rs:495`), `Checked` (`:409`), `Ty` (`:161`), `ValueTy` (`:123`), `Env` (`:223`), `Shape`/`ShapeDim` (`shape.rs:138/98`), `TagSet` (`tags.rs:94`), `TensorType` (`rules.rs:18`), `ChainPlan`/`plan_chain` (`chain.rs:555`), `poly::Poly`, `algo::{SolveMethod, Algorithm}` | ast, diag, **linalg**, units | — |
| sky-linalg | reproducible f64 dense linear algebra (+ relaxed/Accelerate) | 4,511 | 3,180 | 171+1,291 | `gemm`, `gemm_into_with`, `syrk`, `cholesky*`, `lu`, `Lu::solve`, `inverse`, `dot`, `Numerics`, `set_threads`, `tuning::register` | tune | rayon (dev: faer, matrixmultiply, proptest) |
| sky-interp | tree-walking interpreter **and the shared value layer** | 10,491 | 4,966 | 4,038+52 | `Value` (`value.rs:16`), `run_module_with_derivatives` (`eval.rs:89`), `RuntimeEnv`, `arith`, `array`, `solve`, `math::apply`, `vecfn::{norm,dot}`, `uncertain::{Uncertain,Lin}`, `RuntimeError` | ast, diag, ode, syntax, types, units | — |
| sky-ode | ODE integrators and automatic method choice (RK4, DOPRI5, Verlet, Yoshida4, Rodas3) | 2,593 | 1,774 | 163+1,758 | `System`, `Options`, `Method`, `solve`, `Solution` | — | libm |
| sky-ir | SIR data model, op table, verifier, text format, pass manager, analyses (effects, exact, roundoff, cost, lifetime, parallel), type inference | 14,665 | 10,737 | 702+5,359 | `Module`/`Function` (`ir.rs:356/306`), `Type`/`NumberType`/`Repr` (`ir.rs:74/53/42`), `OpKind` (`op.rs:212`), `verify` (`verify.rs:122`), `Pass`/`PassManager` (`pass.rs:197/253`), `print`/`parse`, `cost::analyze` (`cost/mod.rs:236`), `Lifetimes` (`lifetime.rs:111`), `parallel::Config` (`parallel.rs:188`) | ast, diag, syntax, **tune**, types, units | — |
| sky-lower | checked AST → SIR L1 | 4,873 | 3,631 | 0+5,643 | `lower` (`lib.rs:310`), `lower_for_run` (`:360`), `lower_defs(_partial)`, `LowerError` (`:252`) | ad, ast, diag, interp, ir, syntax, types, units | — |
| sky-ad | AD as a SIR transform: forward mode, plus reverse mode for straight-line code, `if` and calls | 5,356 | 3,260 | 1,255 | `derivative`, `derivative_functions`, `differentiate_into`, `AdError` | diag, ir, types, units | — |
| sky-derive | `derivative(f,x)(args)` for the tree-walker, computed through SIR | 225 | 154 | 0 | `SirDerivatives` (implements `sky_interp::DerivativeHook`) | ad, ast, diag, interp, ir, ir-interp, lower, types | — |
| sky-egraph | e-graph rewriting of SIR expression segments (egg) | 3,393 | 2,603 | 0+2,409 | `EgraphPass`, `RewriteConfig`, `RewriteContract`, `rule_table`, `CostModel` | diag, ir, types, units | egg =0.11.0, quanta |
| sky-opt | optimization passes and the default pipeline | 14,036 | 5,863 | 6,106+5,801 | `default_pipeline` (`pipeline.rs:362`), `DEFAULT_PASSES` (`:109`), `fold`, `dce`, `gvn`, `licm`, `fuse`, `cholesky`, `transposes`, `structure`, `powers`, `algorithms`/`MatMulPolicy`, `specialize`, `specialize_for` | diag, egraph, interp, ir, types, units | — |
| sky-ir-interp | SIR reference interpreter; parallel runtime; JIT-compiled fused kernels | 5,901 | 2,184 | 2,996+6,563 | `run` (`lib.rs:140`), `run_with` (`:149`), `ParConfig`, `JitOptions` | diag, interp, **jit**, ir, tune, units | rayon |
| sky-jit | Cranelift native code: whole-module JIT, fused kernels, AOT objects, `extern "C"` runtime | 10,730 | 7,348 | 1,044 | `compile` (`module.rs:424`), `compile_par` (`:449`), `Compiled` (`:273`), `emit_clif` (`:469`), `kernel::compile_kernel` (`kernel.rs:1147`), `aot::compile_object` (`aot.rs:127`), `Slp` (`module/slp.rs:74`) | diag, ir, **interp**, types, units, tune | cranelift-{codegen,frontend,jit,module,native,object} 0.121.2, target-lexicon, rayon |
| sky-rt | static runtime library for `sky build` executables | 249 | 131 | 31 | `sky_rt_start`, `Meta` | diag, ir-interp, jit | — (crate-type staticlib+rlib) |
| sky-hw | CPU/OS hardware discovery and fingerprint | 1,011 | 509 | 320 | `HardwareInfo` (`lib.rs:32`), `detect`, `fingerprint`, `kernel_threads` | — | — |
| sky-tune | autotuner plus a data-only performance database | 2,090 | 965 | 652+61 | `Tunable` (`measure.rs:31`), `lookup` (`lib.rs:138`), `Db`, `Record`, `Registry`, `Store` | hw | — |
| sky-project | project root, `sky.toml`, import resolution (not wired into the language) | 2,754 | 1,699 | 487+1,407 | `discover(_opened)`, `resolve_import`, `parse_manifest`, `Numerics` | diag, syntax | libc (unix) |
| sky-cli | the `sky` binary: check/run/build/repl/hw/tune | 3,865 | 2,929 | 110+7,452 | `main` (`main.rs:68`), `run::run`, `ir::build`/`prepare_run`, `frontend::analyze_ir`, `cost`, `accuracy`, `numerics` | ast, derive, diag, hw, interp, ir, ir-interp, jit, linalg, lower, opt, project, rt, syntax, tune, types, units | — |
| sky-bench | benchmarks, parallel break-even study, cross-language comparison | 4,429 | 3,228 | 674+336 | `bench.rs`, `compare/` (writes `benches/compare/RESULTS.md`), `par.rs`, `native_par.rs` | ast, derive, diag, interp, ir, ir-interp, jit, lower, opt, syntax, types, units | — |

Notes on size (VERIFIED from the table):
- `sky-syntax` is mostly a generated NFKC table: 9,953 lines in `nfkc_table.rs`. Its real code is about 1.5k lines.
- The largest production crates are sky-types (11.9k), sky-ir (10.7k), sky-jit (7.3k), sky-opt (5.9k) and sky-interp (5.0k).

### 2.2 Crate dependency layers (VERIFIED from the Cargo.toml files)

```
L0  sky-diag      sky-hw       sky-units      sky-ode(libm)
L1  sky-syntax(diag)          sky-tune(hw)
L2  sky-ast(diag,syntax)      sky-linalg(tune; rayon)      sky-project(diag,syntax)
L3  sky-types(ast,diag,linalg,units)
L4  sky-interp(ast,diag,ode,syntax,types,units)       sky-ir(ast,diag,syntax,tune,types,units)
L5  sky-ad(diag,ir,types,units)   sky-egraph(diag,ir,types,units; egg)
    sky-jit(diag,ir,interp,types,units,tune; cranelift, rayon)
L6  sky-lower(ad,ast,diag,interp,ir,syntax,types,units)
    sky-opt(diag,egraph,interp,ir,types,units)
    sky-ir-interp(diag,interp,jit,ir,tune,units; rayon)
L7  sky-derive(ad,ast,diag,interp,ir,ir-interp,lower,types)     sky-rt(diag,ir-interp,jit)
L8  sky-cli(everything)    sky-bench(most)
```

Structural observations (VERIFIED from the dependency lists; the consequences are LIKELY):
- **`sky-interp` is a dependency of the native backend** (`sky-jit` → `sky-interp`) and of the optimizer (`sky-opt` → `sky-interp`, which folds constants with the interpreter's arithmetic). The tree-walker's `Value` type is therefore the semantic root of every engine. A new data model, such as typed tensors, cannot be added "in SIR only": it has to be added to `sky_interp::Value` and `sky_interp::array`, or the new path has to be decoupled from them.
- `sky-types` → `sky-linalg`: the checker proves positive definiteness and computes condition-number warnings on constant matrices with the real kernels (`sky-types/src/linalg.rs:1-25`).
- `sky-ir` → `sky-tune`: the parallel thresholds are tuned (`ParConfig::tuned`).
- `sky-ir-interp` → `sky-jit`: the SIR interpreter JIT-compiles `fused` bodies.

---

## 3. Type system

### 3.1 Representation

There are three type layers, all VERIFIED.

1. **Checker `Ty`** (`sky-types/src/check.rs:161`):
   - `Value(ValueTy)`, where `ValueTy` is `{dim: Dimension, ty: TensorType}` (`check.rs:123`);
   - `Bool`;
   - `Str`;
   - `Func(Arc<FuncSig>)`;
   - `Ode(Arc<OdeTy>)`;
   - `Unknown`, used for an unannotated parameter or the result of an earlier error, so that one error never cascades.

   `TensorType` is `{shape: Shape, tags: TagSet}`, with the invariant that tags are closed and appear only on a square rank-2 shape (`sky-types/src/rules.rs:13-53`).

2. **SIR `Type`** (`sky-ir/src/ir.rs:74`): `Number(NumberType{dim, shape, tags, repr})`, `Bool`, `Str`, `Nothing` or `Unknown`.
   - `Repr` is `Unchosen | Exact | F32 | F64` (`ir.rs:42`).
   - **No pass ever chooses a `Repr`, and no engine reads one: every number is computed as f64.** The only non-test references to `Repr::F32` and `Repr::Exact` are in the printer, the parser and the display code. A grep of `crates/` confirms this.
   - The design doc lists F16/BF16/F128 as "Phase 3" plans (`docs/design/ir.md:64`).

3. **Runtime `Value`** (§4).

### 3.2 Sizes and shape generics (VERIFIED)

- `ShapeDim = Known(u64) | Var(ShapeVar)` (`sky-types/src/shape.rs:98-104`), and `Shape = Vec<ShapeDim>` of any rank (`shape.rs:138`).
- **"There is no dimension arithmetic (`N+1`, `2*N`), so `ShapeDim` has no way to express it"** (`shape.rs:1-6`; spec §"Out of scope": "dimension arithmetic such as `N+1`").
- Unification is equality only, with provenance recorded for diagnostics (`ShapeUnifier`). It is monomorphic per module, and generic per `def`, where each call gets fresh copies (spec §"Shape variables across branches and functions").
- Annotations: `Scalar`, `Vector[n]`, `Matrix[m, n]` and `Tensor[...]`. `Tensor[2,2,2]` is accepted as an annotation; probe `p6_tensor_ann.sky` gave `ok`. Nothing produces rank 3, though: a rank-3 literal is an error ("array literals have at most two levels"; probe `p5_rank3.sky`), and the SIR array ops are vector/matrix only (`sky-ir/src/op.rs:381-387`: "a vector has one axis and a matrix two (v0.1 has no higher ranks)").
- **Sizes must be compile-time constants.**
  - `zeros(len(h))` inside a generic function is an error: "`zeros` needs a constant size, found `len(h)`" (probe `p2_zeros_n.sky`).
  - A slice with a computed bound gets a fresh rigid size variable. Lowering then refuses it if a stand-in is needed (`sky-lower/src/ctx.rs:263`).
  - There is no runtime-size path (the `sizes` worktree, §8, starts one).
- Specialization clones a generic function per concrete shape or constant (`f[N=3]`). See §5.2.
- `MAX_ARRAY_ELEMENTS = 1<<24` (16 Mi entries, 128 MiB of f64) per array (`sky-types/src/constant.rs:23`). The checker and the interpreter both enforce it. **A single Qwen3-32B MLP weight (5120×25600 ≈ 131M entries) is 7.8× over this cap.** (The limit is VERIFIED; the Qwen shape is LIKELY, from the public config.)

### 3.3 Units and tags (VERIFIED)

- **Dimension**: 7 rational exponents (`[Rational; 7]`, where each `Rational` is two `i64`s), so 112 bytes per dimension (`sky-units/src/dimension.rs:109-111`, `rational.rs:16-19`). Units convert to SI at the literal, and values never remember a unit.
- **Tags**: `Symmetric`, `PositiveDefinite`, `Diagonal`, `Orthogonal`, stored as a `u8` bitset with closure rules PD⇒Sym and Diag⇒Sym (`sky-types/src/tags.rs:32-140`). The four are fixed in code and not extensible (`sky_ast::TAG_NAMES`).
  - Literal tags are proven from the constant entries (PD by a Cholesky factorization of up to 32×32, at the checker).
  - Tags on parameters are runtime-checked promises.
  - Orthogonal is never checked at run time and must be known statically.

### 3.4 Uncertain numbers (VERIFIED)

Uncertainty is **not part of the type**. `x ± dx` types as `x + dx` does, and only scalars may carry it (`sky-types/src/uncertain.rs:1-25`, `:63-80`).

At run time the value is `Value::Uncertain{value, dim, lin: Lin}`: linear propagation over named error sources, which tracks correlations (`sky-interp/src/uncertain.rs:1-40`, `:177`).

**SIR cannot represent it**: any `±` triggers a `LowerError`, and the program runs on the tree-walker.

### 3.5 Compile time vs run time (VERIFIED unless marked)

**Known at compile time:**
- dimensions (always static in typed code);
- shapes (constant or symbolic);
- tags (with promises checked at run time);
- `@`-chain association and solve method;
- costs, as polynomials;
- which operations can fail, as effects.

**Known at run time:**
- values;
- the actual sizes behind symbolic variables;
- an `unknown`-typed parameter's kind and dimension, all checked dynamically;
- uncertainty.

### 3.6 dtypes, integers, records, modules, generics, closures (VERIFIED)

| Feature | Status today |
|---|---|
| dtypes | **f64 only.** `Value::Number.value: f64`, `Value::Array.data: Vec<f64>`, the cost model's `ENTRY_BYTES = 8` (`sky-ir/src/cost/ops.rs:16`), and every kernel takes `&[f64]` |
| integer types | **None.** Integers are f64 values that happen to be whole numbers. Loop indices are f64 ("Loop indices are `f64` numbers", `sky-types/src/constant.rs:25-30`) and are checked at run time by `expect.range`; index operands are "dimensionless scalars" checked as whole numbers |
| structs/records | none |
| tuples | none on main (in flight, §8) |
| modules/imports | `import` parses but is "not supported yet". The resolver exists in `sky-project` but is not wired into the language (spec §"Out of scope", line 1031) |
| generics | size variables in `def` signatures, and unannotated parameters (type `unknown`, checked dynamically and specialized by the `specialize` pass) |
| closures / first-class functions | **None.** A `def` cannot take a function argument; nested defs cannot read enclosing locals; SIR calls name a fixed callee (`call @f`) (spec "Limits of the SIR lowering", line 171; `sky-lower/src/lib.rs` module docs) |
| strings | literals, print and `==`/`!=` only; no formatting on main (in flight) |

---

## 4. Values and memory

### 4.1 Tree-walker, `sky-interp` (VERIFIED)

- `Value` (`sky-interp/src/value.rs:16-54`) has these variants:
  - `Number{value: f64, dim: Dimension}`
  - `Uncertain(Uncertain)`
  - `Array{dims: Vec<usize>, data: Vec<f64>, dim: Dimension}`
  - `Bool`, `Str(String)`, `Shown(String)`, `None`
  - `Func(Arc<Function>)`, where the body is a clone of the checked AST
  - `Ode(Arc<OdeValue>)`
- Size is LIKELY ≥160 bytes per `Value`: two `Vec`s plus a 112-byte `Dimension` in the `Array` variant. This is layout reasoning, not a measurement.
- The environment is `RuntimeEnv{vars: BTreeMap<String, Value>, chains}` (`sky-interp/src/env.rs:1-20`), so every variable read is a string-keyed B-tree lookup.
- Arrays are **owned values**: cloned on copy, with no sharing and no views.
  - The tree-walker *moves* a variable at `x = x + …` when the right-hand side reads `x` exactly once and can run no user code (`docs/design/ir.md:272`).
  - A binary operation writes into an owned temporary operand.

### 4.2 SIR interpreter, `sky-ir-interp` (VERIFIED, `sky-ir-interp/src/machine.rs:1-120`)

- Values are the same `sky_interp::Value`, kept in a value table per activation (`Vec<Option<Value>>`, indexed by `ValueId`).
- `sky_ir::Lifetimes` (a backward liveness analysis, `sky-ir/src/lifetime.rs:111,125`) marks where each operand dies. The interpreter **moves** dying operands and writes elementwise results into their buffers (`arith_owned`).
  - Dead arrays are dropped immediately, and spent buffers are recycled.
  - "`v = v + dt*a` in a loop allocates nothing", which `crates/sky-bench/tests/allocations.rs` tests with a counting global allocator.
- Every array is still **two allocations**, one for `dims` and one for `data` (`docs/design/ir.md:270`). Slices and transposes copy (`ir.md:268`: "`Value::Array` has no strides or offsets").

### 4.3 Native code, `sky-jit` (VERIFIED, `sky-jit/src/module.rs:1-40`, `runtime.rs:1-60,152-192`)

- A scalar is an `f64` (its dimension is static, not stored) and a Bool is an `i8`.
- A **small array**, meaning a static shape with at most 16 entries (`sky-jit/src/module/small.rs:53`, `MAX_SMALL = 16`), is scalar-replaced into N `f64` SSA values. It is passed through calls, `if` and `while` as N separate values.
- Everything else, including all arrays of more than 16 entries, strings and `unknown` values, is a **handle**: a `u32` index into a thread-local table of `Cell{rc, value: Option<Rc<Value>>}` (`runtime.rs:152-156,192`).
  - Generated code retains and releases handles. Each block owns its values, and operands are borrowed.
  - Array operations push their operands and call `sky_rt_op(site)` (`module.rs:2229-2284`).
  - The runtime then runs the same `sky_interp::{arith,array,solve,…}` functions the interpreters use.
- Probe `p7_mlp.sky`, a 64-wide MLP-like function: its Cranelift IR is 251 lines dominated by runtime calls (`call fn1/fn2/...`). Raw file: `$S/research/r1/probe/p7.clif`.

### 4.4 Allocators, copying, aliasing (VERIFIED)

- **Allocator:** the system allocator. There is no `#[global_allocator]` outside test binaries (grep: only `sky-bench/tests/allocations.rs` and `sky-syntax/src/nfkc/tests.rs`).
- **Alignment:** arrays are `Vec<f64>` with 8-byte alignment. The kernel docs say so explicitly (`sky-jit/src/kernel.rs:74`).
- **Code memory:** the JIT reserves one 256 MiB arena for a module's code and data, and 16 MiB per compiled kernel (`sky-jit/src/lib.rs:78-93`).
- **Semantics:** arrays are immutable values in the language. There is **no aliasing and no mutation**, so no memory model is needed. The flip side is that there is **no in-place update**: element writes are check errors on main.
- The GEMM packs into temporary per-call buffers, which is LIKELY from the `gemm.rs` structure (not traced line by line).

---

## 5. SIR, optimizer and backends

### 5.1 SIR (VERIFIED, `sky-ir/src/{ir.rs,op.rs,verify.rs}`, `docs/design/ir.md`)

**Data model.**
- `Module{edition, functions}`. A `Function` has flat arenas of values, ops, blocks and regions addressed by `u32` ids (`ir.rs:306-319`).
- Regions are structured and hold exactly one block in v1. Each block ends in a terminator: `yield`, `condition` or `return`.
- SSA throughout. Mutable variables become loop-carried block arguments, MLIR `scf.while` style.
- Every op carries a `span` plus `operand_spans`, so runtime errors point at the source.

**Instruction set: 47 `OpKind` variants** (`op.rs:212-668`; enumerated with awk). By group:
- **Constants:** `ConstFloat{bits}`, `ConstRational`, `ConstBool`, `ConstStr`, `ConstNothing`, `LiteralError`, `Input{name}`, `ConstArray{cols, bits: Arc<[u64]>}` (written for literals of ≥64 numeric entries).
- **Arithmetic:** `Add`, `Sub`, `Mul`, `Div` (elementwise on equal shapes; `mul`/`div` by a scalar), `Neg`, `PowConst{exponent: Rational}`, `Pow`, `Inverse`, and `Math(MathFn)`, which covers 30 functions applied entry by entry to arrays: sqrt, cbrt, abs, exp, log, log10, log2, sin, cos, tan, asin, acos, atan, atan2, hypot, min, max, floor, ceil, round, expm1, log1p, sinh, cosh, tanh, asinh, acosh, atanh, erf, sign (`sky-units/src/math.rs:37-68`).
- **Linear algebra:**
  - `MatMul`, `Transpose`, `Norm`, `Dot`.
  - Produced only by rewrites: `Solve{Cholesky|Lu|Diagonal}`, `SolveRight{…}`, `MatMulEx{ta, tb, algorithm: Auto|Dot|MatVec|VecMat|Gemm|Syrk}`, `DiagMul{Left|Right}`.
- **Arrays:** `ArrayLiteral{cols}`, `Index`, `Slice{axes}`, `Zeros`, `Identity`, `Len`.
- **Kernel:** `Fused`, one region with a scalar body that runs once per element.
- **Tags:** `Assume{tags}`.
- **Logic:** `Cmp(Eq|Ne|Lt|Le|Gt|Ge)`, `Not`, and the dynamic checks `ExpectBool`, `ExpectNumber`, `ExpectRange{Bound|Step}`.
- **Control:** `If`, `While`, `Yield`, `Condition`, `Call{callee}`, `Return`.
- **Contracts and I/O:** `Assert{…}`, `Print`, `Show{unit}`.

**Effects** are `{Io, Fail, Call}` (`ir.rs:105-116`). Almost every arithmetic op has `Fail`, because overflow to infinity is an error. Op properties record `commutative` and `associative_real` (`op.rs:673-845`).

**Calls:** `call @name` only, first order. A function takes an extra trailing parameter for each module-level name it reads (`sky-lower/src/lib.rs` docs, "Globals"). Early `return`, `break` and `continue` become Bool flags.

**Arrays:** dense, row-major, never empty, rank 1–2, f64, one dimension shared by all entries (`op.rs:379-387`).

**Memory model: none.** There are no buffers, loads, stores, pointers, views, dtypes, devices, streams or async operations. All of that is left to "L2 Strategy IR / L3 Kernel IR / L4 Backend" (`docs/design/ir.md:25-34`). **Those levels do not exist in code**: there is no `Level` enum and no `fn level`, and grep finds no match.

**Verification** (`verify.rs:1-60`, `verify.rs:122`): arena integrity, SSA and scoping, region shape, per-op typing (it reuses `sky_types::rules` so the checker and verifier cannot drift), closed tags, effect consistency, and `fused`-body rules. The verifier is total and never panics.

**Text format:** round-trips (`parse(print(m)) == canonicalize(m)`) and has a fuzz target (`fuzz/fuzz_targets/sir_parse.rs`). The pass manager checks the module before the first pass and after every pass when `verify_each(true)` is set, and `sky run` always sets it (`sky-cli/src/ir.rs:209`).

### 5.2 The full `sky-opt` pipeline (VERIFIED, `sky-opt/src/pipeline.rs:9-97,109-123,362`)

Each pass is listed with a one-line description.
1. **specialize**: clones a callee for call sites with known shapes, constants, signs or types (`f[N=3]`, `fib[n:Scalar]`), retypes `unknown` parameters, simplifies, and unrolls constant loops of ≤8 iterations and ≤256 added ops. Limits: 8 clones per function and 64 per module (`docs/design/ir.md:218-250`).
2. **fold**: constant folding, bit-identical to the interpreter (`FoldMode::BitIdentical`).
3. **egraph**: `egg` saturation of pure expression segments under the `Reproducible` contract (§5.3).
4. **gvn**: global value numbering (CSE across regions, scoped by dominance).
5. **licm**: loop-invariant code motion out of `while`, with a guard when an op that can fail would be hoisted from a loop that may run zero times.
6. **contracts**: removes `assert`/`requires`/`ensures` checks that the exactness analysis proves true.
7. **dce**: removes dead pure ops, and ops that can fail by kind but provably cannot here (`DCE-002`).
8. **powers**: `x^n` for 2 ≤ n ≤ 14 becomes the multiplication chain `powi` uses, bit-identical.
9. **fuse**: a tree of single-use elementwise ops (`add`/`sub`/`mul`/`div`/`neg`/math) in one block becomes one `fused` op.
10. **cholesky**: `matmul(inverse(A), b)` with PD `A` becomes `solve[cholesky]`, at sites the checker did not already rewrite.
11. **transposes**: `matmul(transpose(A), B)` becomes `matmul[ta]`, with no materialized transpose.
12. **structure**: Gram products (`syrk`, one triangle mirrored), Diagonal scaling (`diag.mul`), and `transpose(S)` of a Symmetric `S` becomes `S`.
13. **algorithms**: records an algorithm class per `matmul` through a `MatMulPolicy`. **The default policy chooses nothing** (`sky-opt/src/pipeline.rs:313-316`).

Also present:
- Analyses with no rewriting: `effects`, `exact` (interval plus symbolic "ideal value"), `roundoff` (sound error bounds for `accuracy` budgets), `cost`, `lifetime` and `parallel`.
- `run_reporting` clones the whole module before every pass to decide "changed" (`pipeline.rs` `run_reporting`; `EgraphPass::run`, `sky-egraph/src/lib.rs:274-280`; `sky-opt/src/pipeline.rs:128-138`). That is LIKELY fine for small programs and LIKELY quadratic-ish for big generated graphs.

Rewrites that change rounding are **not** passes. The checker decides chain association and solves, so that the tree-walker makes the same choice (`sky-opt/src/lib.rs:43-46`; `docs/design/ir.md:165`).

Hooks without consumers (VERIFIED by grep; these are negative findings):
- `sky_opt::specialize_for` and `SpecializationCache`, the "run-time specialization hook for the native engine", have **no caller outside `sky-opt`**.
- `MatMulPolicy`: no kernel registers one. Of the algorithm classes, only `Syrk` changes execution (`sky-interp/src/solve.rs:527`).

### 5.3 `sky-egraph` (VERIFIED, `sky-egraph/src/{lib.rs,rules.rs,cost.rs}`)

- **Where it runs:** pass 3 of the default pipeline, on each block's pure expression segments. Effectful ops and ops that can fail for reasons the e-graph does not model are barriers.
- **Limits:** 12 iterations, 20,000 nodes, 20,000 ops per segment, and **no time limit by default**, so the result does not depend on machine speed (`lib.rs:140-150`).
- **Rule classes:** Exact, Conditional, Relaxed and RealOnly. The contracts are `Reproducible` (Exact + Conditional, the default), `Relaxed` (adds Relaxed rules) and `Fast` (adds RealOnly rules). **No pass enables `Relaxed` or `Fast`**: `--numerics=relaxed` does not change the e-graph contract, and a grep finds no `RewriteContract::Relaxed` outside the crate.
- **Rules:** 27 in all (`rules.rs:869-999`):
  - ALG-001…013 and 015…018, for example `x*1`, `x+0` (only if `x` provably ≠ −0), `x−x`, `x/x` (only if nonzero), `x+x → 2*x`, distribution and reassociation (Relaxed);
  - EXP-001/002 (RealOnly);
  - LA-001…007: `Aᵀ → A` if Symmetric, `(Aᵀ)ᵀ`, `(A⁻¹)⁻¹` (RealOnly), transposes of sums, and matmul reassociation (Relaxed);
  - TRIG-001 (RealOnly).
- **Extraction cost** (`cost.rs:1-60`): each e-node is priced by asking `sky_ir::cost` about a one-op function. The price is `8·flops + bytes + 8·64·allocations`, in eighths of a flop. Symbolic sizes default to 64. Ties break on a latency table (add 1, mul 2, div 8, transcendental 24, matmul 16). When an array-valued root gets cheaper, the pass explains it with rule `EG-COST-001`.

### 5.4 "Compiled fused elementwise kernels" (VERIFIED)

**Where they live:**
- `sky-opt/src/fuse.rs`, the pass that forms `fused` ops;
- the `OpKind::Fused` semantics (`sky-ir/src/op.rs:467-507`);
- `sky-jit/src/kernel.rs`, which compiles a fused body into a Cranelift loop;
- `sky-ir-interp/src/native.rs`, where the interpreter decides when to compile;
- `sky-ir-interp/src/kernel.rs`, a scalar register "kernel" for bodies made only of `+ − × ÷ neg`;
- `sky-jit/src/module/small.rs`, which inlines fused ops on small arrays.

**How fusion is decided:** a purely syntactic, single-use tree rule inside one block. A candidate is absorbed into its consumer when its result has exactly one use and both ops have the same array shape. Scalars are broadcast leaves. The pass never moves an op that can fail past another op that can fail, print, call, loop or touch `Unknown` (`fuse.rs:1-55`).

**Codegen** (`sky-jit/src/kernel.rs:1-110,150-200`):
- Values that do not depend on the element are computed once, outside the loop.
- The loop runs in `F64X2` (128-bit, "Cranelift's widest vector"), unrolled 4× (8 elements per iteration), with a scalar tail.
- Math functions other than `sqrt`/`abs` are **scalar calls, lane by lane**, into `sky_interp::math::apply`.
- Failure detection is a NaN-accumulating check per value. On a hit, the kernel re-runs the interpreter's reference path to build the exact diagnostic.
- Relaxed mode can contract `a*b+c` into an FMA.
- Large kernels run in deterministic chunks on rayon. In native code the thresholds are 128× higher (524,288 element-ops).

**When it compiles:** in the SIR interpreter, after 512 elements have run through that op (compiling costs 120–180 µs, `native.rs:11-30`). In native code, through the runtime's kernel cache.

**Limits** (VERIFIED):
- No reductions inside kernels, so softmax and norm cannot fuse.
- No matmul epilogue or prologue fusion.
- No multi-output kernels, and no recomputation of multi-use intermediates.
- No control flow in bodies.
- No fusion across statements whose intermediate has two uses.
- f64 only, 128-bit SIMD only.
- Every non-trivial math function is a scalar libm call.

Measured by the project: kernels reach about 1.4× of C. Whole elementwise programs on 1e6–1e7 elements run 1.5–2.4× slower than C (native `elementwise-1e7` 97.0 ms vs C 41.1 ms; `benches/compare/RESULTS.md`).

### 5.5 `sky-linalg` (VERIFIED, `sky-linalg/src/lib.rs:1-171`, `micro.rs:1-60`, `accelerate.rs:38-90`)

- **GEMM:** Goto/BLIS structure.
  - Cache blocks over `j` (1024), `k` (256) and `i` (128), all tunable per size class: `le64`, `le256`, `le1024`, `gt1024`.
  - `B` is packed into 8-column slivers and `A` into 4-row slivers. The 4×8 accumulator tile (`MR=4`, `NR=8`) stays in registers.
  - Special unpacked kernels handle ≤16 columns, matvec and vecmat.
  - The output is split across rayon threads above 2^21 multiply-adds (tunable).
- **Reproducible rule:** each output entry is summed in naive `k` order with **no FMA**. Dot products use a fixed 8-lane interleaved order.
- **SIMD:**
  - The reproducible kernels are plain Rust loops "written so LLVM vectorizes them" (NEON); x86 adds AVX2 without FMA.
  - The relaxed kernels use NEON `vfmaq_laneq_f64` intrinsics.
  - Everything is f64. There is no f32 or lower-precision kernel anywhere.
- **Factorizations:** blocked right-looking Cholesky (block of 32 or 64 columns); LU done as the Gauss-Jordan forward phase, recursive on columns; a blocked Gauss-Jordan inverse; triangular solves; `syrk`.
- **Algorithm choice by matrix properties** is made *above* the kernels:
  - The checker's chain planner (`sky-types/src/chain.rs:555`) picks Orthogonal → `Qᵀ`, Diagonal → division, PositiveDefinite → Cholesky, otherwise LU.
  - The SIR passes add `syrk`, `diag.mul` and `matmul[ta]`.
- **Relaxed numerics:** Accelerate `cblas_dgemm`, `cblas_dsyrk`, `dpotrf`/`dpotrs`, `dgesv`, but only for sizes that pay off. Vendor results are trusted only when clearly away from the singularity boundary.
- **Numbers** (`benches/compare/RESULTS.md`, M2 Pro): matmul-512×2 takes Sky native 16.1 ms vs Accelerate 4.95 ms vs naive C 42.3 ms, and relaxed Sky 10.9 ms.
- **Our probe** (EXPERIMENTAL-quality measurement; noisy because the load average was about 4.3 from other agents' unlocked processes): 200 f64 GEMVs on a 2048×2048 matrix (33.5 MB per pass), `sky run --engine=native`, median of 5 runs minus setup.

  | configuration | per GEMV | effective bandwidth |
  |---|---:|---:|
  | 10 threads, reproducible | 0.318 ms | ~105 GB/s |
  | relaxed (Accelerate path) | 0.455 ms | ~74 GB/s |
  | `--threads=1` | 0.863 ms | ~39 GB/s |
  | SIR interpreter | 0.351 ms | ~95 GB/s |

  Raw data: `$S/research/r1/probe/gemv_results.txt`. Commands: `bash benchlock.sh bash run_gemv.sh`, which wraps `python3 timeit.py 5 sky run <flags> p8_gemv.sky` and `p8_setup.sky`. Part of the 33.5 MB may be served from the SoC system cache; the size of the M2 Pro SLC was not verified, so these bandwidth numbers are an upper-bound reading. **Take-away:** Sky's CPU f64 matvec is competent, but at 8 bytes per weight versus about 0.56 bytes for Q4_0 it moves about 14× more bytes per weight than a 4-bit runtime.

### 5.6 `sky-jit` (VERIFIED, `sky-jit/src/{lib.rs,module.rs,kernel.rs,runtime.rs,aot.rs,module/slp.rs,module/small.rs}`)

**Lowered to Cranelift:**
- scalar f64 arithmetic, compares, `if`/`while`, and calls between SIR functions;
- small-array ops (≤16 entries): literal, add/sub/neg/scale, index, dot, norm, matmul, transpose, fused, zeros, identity, len;
- `powi` chains;
- inline `sqrt`, `abs`, `floor`, `ceil`, `min`, `max`;
- finiteness and zero-divisor checks;
- call-depth and stack-budget checks.

Every function returns `(status, results…)`, and a failing check calls a runtime function to rebuild the exact diagnostic.

**Left as runtime calls** (`module.rs:2229-2284`): `show`, `matmul` on handles, `norm`/`dot` on handles, `transpose`, `assume`, the solves, `matmul[…]`, `diag.mul`, `array.literal`, `const.array`, `index`, `slice`, `zeros`, `identity`, `len`, `fused` on handles (`fn fused` just calls `generic`, `module.rs:2264`), `print`, `assert`, all non-exact math functions (through `sky_interp::math::apply_f64`), and anything typed `unknown`.

**SIMD:**
- SLP pairs of f64 ops into `F64X2`. The default `Auto` mode packs only loops that are not latency-bound (`slp.rs:1-30,74-100`).
- Fused kernels use `F64X2` loops.
- Nothing is wider than 128 bits, because Cranelift's NEON support stops there.

**Threads:** big `fused` kernels and elementwise ops run in deterministic chunks. Independent pure calls run on workers in "windows" from the `sky_ir::parallel` analysis (`module.rs:120-180`). Executables built with `sky build` are single-threaded.

**AOT** (`aot.rs:1-90`):
- The same `translate_module` emits an object file (Mach-O or ELF) through `cranelift-object`, for the baseline target CPU.
- The object contains an image holding the SIR text, the dimension and site tables, and the source and warnings, plus a C `main` that calls `sky_rt_start`.
- It links against `libsky_rt.a`, which contains the runtime, the SIR interpreter, `sky-linalg` and Cranelift (for kernels compiled while running).
- Result: about 5.7 MB per executable, starting in about 4 ms (spec, line 1001).

**Code caching:** none across processes. Compiled kernels are cached per run, keyed by op address (`runtime.rs:177-181`). The module is recompiled on every `sky run`; the floor for `print(1)` is 5.58 ms native (`RESULTS.md`, "Floors").

### 5.7 `sky-hw` and `sky-tune` (VERIFIED)

**`sky-hw` detects** (`sky-hw/src/lib.rs:30-70`):
- arch, OS and model;
- SIMD feature flags (NEON on arm64; SSE…AVX-512, FMA on x86);
- physical, logical, P- and E-core counts;
- L1d, L2 and L3 sizes, cache line size, memory size and page size.

It reads these through `sysctl` on macOS and `/proc` and `/sys` on Linux, with no `unsafe` code. `sky hw` on this machine prints: Apple M2 Pro, neon, 10 cores (6P+4E), L1d 128 KiB, L2 16 MiB, L3 unknown, 16 GiB, fingerprint `448c927493c66f47`.

**Not detected:** GPU (cores, family, Metal features), memory bandwidth, AMX/SME, ANE.

The fingerprint is FNV-1a over arch, model, features and cache sizes. Core counts and memory are deliberately left out (`lib.rs:76-104`).

**Negative finding (VERIFIED): `sky hw` overstates what Sky does.**
- `sky hw` reports "10 cores (6 performance): parallel kernels use 6 threads", from `HardwareInfo::kernel_threads` (`sky-hw/src/lib.rs:135`; report text at `sky-hw/src/report.rs:92`).
- No crate outside `sky-hw` calls `kernel_threads`.
- `sky-linalg` (`threads.rs`, `default_threads`) and `sky_ir::parallel::Config::default_threads` (`sky-ir/src/parallel.rs:267-275`) both use `available_parallelism()`, which is 10 on this machine, E-cores included. `sky check --explain` on `p8_gemv.sky` says "runs on 10 threads".

**`sky-tune`** (`sky-tune/src/{lib.rs,db.rs}`):
- A `Tunable` is a kernel name, a size class, a grid of integer parameters (the first is the default), and a work measure.
- `sky tune` times each candidate (rotating rounds, median, a 500 ms budget). A winner must be at least 3% faster.
- The database is data-only text: `sky-tune-db 1`, then `result fp=… kernel=… size=… params=k:v,… default=… best_ns default_ns when`.
- It is keyed by (fingerprint, kernel, size class), lives at `~/Library/Caches/sky/tune.db` (or `SKY_TUNE_DB`), and is bounded (1 MiB, 4096 records).
- Registered tunables:
  - `linalg_gemm_{le64,le256,le1024,gt1024}` (mc, kc, nc);
  - `linalg_cholesky_*` (nb);
  - `linalg_threads`;
  - the interpreter thresholds (native-kernel threshold, parallel thresholds).
- **Tuning may change speed, never results.** It is a hard rule.

### 5.8 Cost model, cost contracts, `--explain` (VERIFIED)

**Cost model:** `sky_ir::cost::analyze` (`sky-ir/src/cost/mod.rs:1-80,236`, `cost/ops.rs`).
- Per function it computes flops, bytes read, bytes written, allocations and allocated bytes.
- Each measure is an interval `[lo, hi]` of polynomials in the symbolic sizes (`sky_types::poly::Poly`: exact rational coefficients, capped degree).
- Weights:
  - 1 flop for `+ − × ÷` and the cheap math functions;
  - 20 for transcendentals;
  - `2nkm` for matmul;
  - `2n³` for an inverse;
  - `n³/3 + 2n²k` for a Cholesky solve;
  - a fused kernel costs its body per entry.
- Loops are counted when they are a `for range` with a nameable bound; any other loop is "unbounded". Calls substitute the callee's polynomial. It is computed on the module *after* the pipeline.

**Contracts:** `cost flops <= 2*N^3` lines in a `def` header (`StmtKind::Cost`). `sky check` proves a contract if `hi ≤ bound` for every size ≥ 1, refutes it with a witness, or warns (`sky-cli/src/cost.rs`). A run-time time budget is deliberately excluded.

**`sky explain`:** there is no such subcommand.
- It is `sky check --explain`, which merges in source order:
  - the checker notes (`LA-CHAIN-001`, Cholesky, ODE structure);
  - pass explanations through `PassCx` (`FUSE-001`, `GVN-001`, `LICM-*`, `SPEC-*`, `POW-CHAIN-001`, `LA-*`, `EG-COST-001`, `DCE-*`, `CONTRACT-*`);
  - parallelism notes (`PAR-001/2/3`);
  - buffer reuse (`MEM-001`);
  - relaxed numerics (`NUM-001/2`);
  - a per-function cost report;
  - citations of tuned values (`sky-cli/src/frontend.rs:157-205`, `sky-cli/src/ir.rs:158-168,230-445`).
- Separately, the builtin `explain(sol)` prints an ODE solver's reasoning.

---

## 6. Numerics policy and tests

### 6.1 Reproducible by default vs `--numerics=relaxed` (VERIFIED, spec lines 639-699; `sky-cli/src/numerics.rs:1-110`)

**Default (reproducible).**
- Every engine calls the same kernels.
- Fixed summation orders: naive `k` order for products, the 8-lane tree for dots and norms.
- No FMA, neither in the kernels nor in Cranelift code.
- Threads split outputs only.
- Results are bit-identical across machines, thread counts and engines.

**Relaxed.** This is opt-in, through `--numerics=relaxed` or `numerics = "relaxed"` in `sky.toml`. There is deliberately no environment variable and no per-file pragma. It permits:
- FMA in the kernels and in scalar native `a*b+c`;
- reassociated reductions (16 partial sums);
- Accelerate on macOS;
- thread-count-dependent sums at about 1M terms or more.

Relaxed mode still guarantees the same units and shapes, finite results, the same errors in the same order, and agreement within a 1e-12 tolerance in tests. It announces itself on stderr. The e-graph contract does **not** change under relaxed (§5.3). The REPL is always reproducible.

### 6.2 Finiteness checks (VERIFIED)

**Rule:** "a Sky number is always finite". Division by zero, overflow and a non-real power are runtime errors at the failing expression, never inf or NaN (spec line 978).

**Where the checks sit:**
- **Interpreters:** after every arithmetic operation (`sky_interp::arith`). This is where the order is defined: dimension checks first, then a zero divisor, then finiteness.
- **Native scalar code:** an inline check after each op, with a cold path into the runtime to rebuild the diagnostic.
- **Fused kernels:** one vector add per checked value (`acc += r − r`), collected once after the loop. Only values reaching the result through a divisor or a math call need their own check (`sky-jit/src/kernel.rs:31-60`).
- **Array kernels:** a whole-result finiteness check.
- **Relaxed mode:** the check applies to the fused value. Native code replays the operation unfused before failing.

**Cost:**
- In fused kernels, about one extra vector op per checked value; the project describes it as "pays for one vector add per checked value".
- In scalar native code, a compare and branch per operation.
- The project does not publish a with/without comparison for whole programs. `kernel::Options::check_finite = false` exists "for measuring only" (`kernel.rs:175`).
- **Our measurement of the fused-kernel check cost.** I ran `sky-jit`'s own bench from `main` into a scratch target dir:
  ```
  bash $S/benchlock.sh cargo bench -p sky-jit --bench kernels --locked
  ```
  Script: `$S/research/r1/run_kernel_bench.sh`; raw output: `$S/research/r1/kernel_bench/run.log`. Each value is the best of 9 batches, in ns per element, with a preallocated output. The load average was about 11 while it ran, because other agents had processes running outside the lock. Best-of-9 dampens that, but treat the numbers as EXPERIMENTAL-quality.

  | kernel | n | `sky` (checked) | `nochk` | `scalar` (checked, no SIMD) | best plain C/Rust | C with the same checks | sky / best | nochk / best |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | axpy | 1e3 | 0.162 | 0.143 | 0.310 | 0.121 | 0.232 | 1.34× | 1.19× |
  | axpy | 1e5 | 0.236 | 0.236 | 0.297 | 0.233 | 0.232 | 1.01× | 1.01× |
  | axpy | 1e7 | 0.266 | 0.265 | 0.315 | 0.264 | 0.266 | 1.00× | 0.99× |
  | hypot | 1e7 | 0.305 | 0.309 | 0.601 | 0.305 | 0.414 | 1.00× | 1.01× |
  | chain6 (6 ops, one division) | 1e3 | 0.356 | 0.274 | 0.688 | 0.304 | 1.031 | 1.17× | 0.90× |
  | chain6 | 1e7 | 0.355 | 0.275 | 0.681 | 0.312 | 1.037 | 1.14× | 0.88× |

  - **Check cost:** about 0% on memory-bound kernels (axpy and hypot at ≥1e5), and **about +30% on a compute-bound six-operation chain with a division**, where the divisor needs a check of its own.
  - Sky's checked kernels beat hand-written C that makes the same checks (0.66–0.84× of its time) and sit within 1.0–1.34× of unchecked C and Rust.
  - Kernel compile time is 0.16–0.25 ms.
  - All of this is 128-bit f64 NEON on the CPU. The numbers say nothing about the cost of the same checks in GPU kernels; I LIKELY expect it to be similar for bandwidth-bound decode GEMV and higher for fused attention (EXPERIMENTAL).

**For LLM work** (LIKELY):
- Attention masks use −inf.
- Softmax and log-softmax rely on IEEE inf/NaN semantics.
- An error on every non-finite value is incompatible with standard model code, and a per-op reduction on the GPU is not free.

### 6.3 The three-engine byte-identical rule and how it is enforced (VERIFIED)

**The rule:** spec lines 914-927. Identical means the same stdout, the same stderr (message, span, labels, notes) and the same exit status. One intended exception: resource-limit messages ("recursion is too deep" vs "nests too deeply").

**Mechanisms:**
- one value layer for all engines (§2.2);
- `operand_spans` in SIR;
- deoptimization rather than divergence;
- failure-order rules for `fused` ops and parallel chunks;
- AOT executables that carry the SIR interpreter.

**Test counts:**

| what | count |
|---|---:|
| `#[test]` functions (static grep, workspace) | 2,628 |
| … sky-ir-interp | 300 |
| … sky-opt | 289 |
| … sky-cli | 256 |
| … sky-types | 244 |
| … sky-ast | 228 |
| … sky-interp | 211 |
| … sky-lower | 207 |
| … sky-ir | 187 |
| … sky-syntax | 118 |
| … sky-project | 104 |
| … sky-units | 99 |
| … sky-egraph | 73 |
| … sky-ode | 66 |
| … sky-diag | 60 |
| … sky-linalg | 40 |
| … sky-bench | 40 |
| … sky-ad | 34 |
| … sky-tune | 31 |
| … sky-jit | 24 |
| … sky-hw | 15 |
| … sky-rt | 2 |
| … sky-derive | 0 |
| files with `proptest!` blocks | 18 |
| UI goldens (`tests/ui/**/*.sky`, each with `.stderr`) | 164 |
| IR golden programs (`tests/ir/*.sky`; 43 `.sir` files, plus `.lower.sir` and `.explain` files) | 22 |
| fuzz targets (`fuzz/fuzz_targets/`): lex, parse_module, sir_parse, manifest, differential | 5 |

**Where the engine-equality rule is enforced:**
- `sky-cli/tests/run_repl.rs`: `engines_agree` runs `sky run --engine={ast,sir,native}` and compares bytes. `grep` counts 51 lines that mention `engines_agree` or `run_script(`, which includes their definitions.
- `sky-cli/tests/examples.rs`: every `examples/*.sky` must print its `.out` file on ast, auto, sir and native, unless SIR declines.
- `sky-ir-interp/tests/differential.rs` (2,775 lines) plus `fuzz_differential.rs`. The latter runs the fuzz generator and oracle on 3,000 fixed seeds per `cargo test`: AST vs SIR vs native, lowered and optimized, with SLP off, auto and always (`fuzz/harness/differential.rs:4055-4093`).
- `sky-cli/tests/build.rs`: AOT executable vs `--engine=native`.
- `sky-bench` fails if its four engines disagree.
- `sky-opt/tests/fuzz_*.rs`: output is the same before and after each pass.

**Gate script:** there is **none in the repository**. `tools/` holds only `check_release_symbols.sh`.

**CI** (`.github/workflows/ci.yml`):
- `cargo test --workspace --locked` on a **self-hosted Linux runner only**; "macOS is checked locally before every push".
- fmt, clippy `-D warnings`, cargo-deny, a release-symbol scan, a Unicode-table check.
- 60 s per fuzz target on every PR, plus a nightly fuzz run.

**Fuzz gates:** the `GATE_*` constants in `fuzz/harness/differential.rs:415-443` steer the generator around known bugs. All are `false` today, meaning every known bug is fixed.

---

## 7. Language syntax relevant to tensors

### 7.1 Summary

| Area | What exists |
|---|---|
| Functions | `def f(a: Matrix[N, K] kg, b) -> Vector[N]:`; annotations combine tags, shape and unit; `requires`/`ensures`/`cost` lines go in the header; recursion is allowed; there are no default arguments, no keyword arguments and no first-class functions |
| Loops | `while c:`, `for i in range(a, b, s):` (indices are f64 integers), `for x in v:` (tree-walker only), `break`, `continue` |
| Arrays | `[a, b]` literals (a vector) and `[[…], […]]` literals (a matrix, written as rows); at most 2 levels; never empty; one dimension per array; immutable |
| Indexing | `v[i]`, `A[i, j]`, `A[i]` (a row), slices `A[0, :]`, `v[1:3]`, `A[:n, 2:]`; no step; no negative indices; writes are check errors |
| Operators | `+ −` (equal shapes only); `* /` (elementwise on equal shapes, or scaling by a number); `@` (matmul / matvec / vecmat / dot); `ᵀ` / `transpose`; `⁻¹` / `inv`; `^` (scalars only; `v^2` is an error by design) |
| Broadcasting | number × array and array ÷ number only; **`c + v`, `c − v`, `c / v` and array–array broadcasting are errors** (spec line 608; probes `p1`, `p4`) |
| Size generics | `Vector[N]`, `Matrix[M, N]`; equality unification; fresh variables per call; no arithmetic |
| Structs/records/tuples | none (tuples are in flight) |
| Modules/imports | none |
| Builtins: arrays | `zeros(n)`, `zeros(m, n)` (constant sizes only), `identity(n)`, `len(x)`, `norm(x)`, `dot(a, b)`, `transpose(A)`, `inv(A)` |
| Builtins: math (entrywise on arrays) | sqrt, cbrt, abs, exp, log, log10, log2, sin, cos, tan, asin, acos, atan, atan2, hypot, min, max, floor, ceil, round, expm1, log1p, sinh, cosh, tanh, asinh, acosh, atanh, erf, sign |
| Builtins: other | `print`, `derivative(f, x)`, `grad(f)`, `∂f/∂x`, `nominal`, `uncertainty`, `explain(sol)`, `range` (only as the iterable of a `for`) |
| Missing on main (VERIFIED by probe `p3_reduce`: "cannot find `sum`") | sum/mean/max/argmax and every other reduction, `ones`, `arange`, `linspace`, `concat`, element writes, reshape, any rank-3 operation, integer dtypes, casts |

### 7.2 Three real examples (VERIFIED, copied verbatim)

1. `examples/showcase.sky:43-48`: proven tags and an algorithm choice.
   ```python
   K: PositiveDefinite Matrix[2, 2] N/m = [[4 N/m, 1 N/m],
                                           [1 N/m, 3 N/m]]
   f = [2 N, 1 N]
   x = K⁻¹ @ f                                      # solved via Cholesky, no inverse formed
   print("displacement:", x as mm)
   ```
2. `tests/ir/chain_order.sky`: shapes and chain association. Its SIR golden `tests/ir/chain_order.sir` shows `matmul %C, %v`, then `%B`, then `%A`.
   ```python
   A: Matrix[500, 500]
   B: Matrix[500, 500]
   C: Matrix[500, 500]
   v: Vector[500]
   x = A @ B @ C @ v
   y = ((A @ B) @ C) @ v
   print(x, y)
   ```
3. `examples/dogfood/heat/rod1d.sky` (excerpt, lines 31-48): how array code must be written today. There are no `ones` and no element writes, so the boundary is embedded with an identity slice.
   ```python
   idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
   x = idx * dx                   # array * scalar scales every entry
   ones21 = exp(zeros(21))
   w = (x - 0.5 m * ones21) / (0.1 m)
   bc = T_hot * identity(21)[:, 0] + T_cold * identity(21)[:, 20]
   E = identity(21)[:, 1:20]
   while change > 1e-9 K:
       interior = u[1:20] + r * (u[0:19] - 2 * u[1:20] + u[2:21])
       unew = E @ interior + bc
   ```

### 7.3 Probe: an MLP/RMSNorm-shaped function (VERIFIED, `$S/research/r1/probe/p7_mlp.sky`)

A 64-wide `W @ x`, then SiLU, then RMSNorm, checks and runs identically on all three engines (`0.9807678960254922 0.9807678960254922`). It has to be written with three workarounds:
- `one = exp(zeros(64))` instead of `ones`;
- `y @ y` instead of a reduction;
- every size fixed at 64.

`fuse` turns SiLU into one kernel (`FUSE-001`). The `zeros`/`exp` constants are **not** folded or hoisted to compile time. The generic version (`Vector[N]`) cannot build its ones vector at all (probe `p2`).

---

## 8. In-flight work: 10 agent worktrees

### 8.1 What git says (VERIFIED)

`git -C ~/sky worktree list`:
- `main` is at `a99e7e3`;
- ten worktrees `.claude/worktrees/agent-*` are on branches `worktree-agent-*`, **all at `b4fa49c`**, which is `main~1`; `a99e7e3` only touched the README, the guide and two help strings;
- `agent-afa55e105ab8ee772` is `locked`.

For every branch, `git log --oneline main..<branch>` printed **nothing** and `git diff --stat main...<branch> | tail -3` printed **nothing**. The branches have **zero commits**: all the work is **uncommitted** in the worktree directories.

I measured the changes read-only with `git --no-optional-locks status --porcelain` and `diff --stat HEAD`. Every file was last modified between 17:16 and 17:20 on 2026-10-01, and nothing has been touched since, so the agents appear to have stopped.

### 8.2 Per worktree

| worktree | feature | tracked diff | new untracked | new `#[test]` | spec/docs touched | what it adds | completeness (my read) |
|---|---|---|---|---:|---|---|---|
| a0952514… | **arrays you can build** | 23 files, +610/−151 | 5 files, 1,732 lines (`sky-types/src/array_build.rs` 731, `sky-interp/src/build.rs` 505, `build_eval.rs`, `sky-lower/src/array_build.rs` 172, `sky-ir/src/verify/build.rs` 264) | 5 | no (code cites a spec section "Building and changing arrays" that does not exist yet) | element and slice writes `v[i] = x`, `A[i,j] += x`, `v[a:b] = w` as **SSA functional update** (`store`), done in place when the old value dies (`Lifetimes`); `ones`, `full`, `linspace`, `arange`, `concat`, `append`; new SIR ops `Store{axes}`, `Fill{ones}`, `Linspace`, `Arange`, `Concat{append}`, threaded through checker, lowering, verifier, both interpreters and the JIT runtime | broad and cross-engine; few tests; no goldens; no spec text |
| abf8af99… | **sizes from variables** | 10 files, +1,081/−149 (`shape.rs` +443, `module_reads.rs` +100) | `sky-types/src/sizes.rs` 284 | 0 | no | `Linear` sizes (`a·root + b` in **one** size variable): `zeros(n)` with `n = 21` gives `Vector[21]`, `v[1:N−1]` gives `Vector[N−2]`; **rigid run-time size variables** for data-dependent `n` | **checker-only**: no lowering, SIR, interpreter or JIT changes, and no tests; the engines cannot run what it accepts |
| a3be8cf3… | **reductions** | 29 files, +408/−21 | 13 files, 1,117 lines (`reduce.rs` ×3, `tests/ir/reductions.*`, `tests/ui/types/*reductions*`, `sky-cli/tests/reductions.rs`) | 14 | spec +29 lines, `ir.md`, README | `sum`, `prod`, `mean`, `var`, `pvar`, `std`, `pstd`, `min`, `max`, `argmin`, `argmax`, `median`, `cumsum`, **vectors only**; `sum` uses `dot`'s fixed 8-lane order; one shared runtime twin (`sky_interp::reduce`) for all engines, constant folding included | the most complete: spec, goldens, IR goldens, CLI tests; no axis/matrix reductions |
| a383df0a… | **broadcasting** | 22 files, +479/−112 | `sky-cli/tests/broadcast.rs` 241 | 9 | spec, `ir.md`, README | **number ⊕ vector only**, under `+ − * /` including `c − v` and `c / v`; number ⊕ matrix stays limited to `*` and `/` by the number, by design (`A − λ` reads as `A − λI`); **array–array broadcasting stays forbidden**; fuse pass and small-array JIT updated | coherent, and deliberately minimal |
| a1dadc02… | **readable output** | 40 files, +704/−122 | 4 files, 1,143 lines (`sky-types/src/format.rs` 153, …) | 4 | no | f-strings with format specs (`{x:.2f}`, `{v as km/h}`), `print(sep=, end=)`, named derived units in output | wide diff (40 files); spec not updated; **its test targets do not compile** (§8.4) |
| ad1ea58e… | **tuples** | 34 files, +941/−128 | 3 files, 535 lines (`tuple.rs` in types, interp and lower) | 0 | no | `a, b`, unpacking, `t[0]`, tuple types in signatures, multiple returns; elements are numbers, arrays or Bools only, no nesting | cross-engine code; no tests; no spec; **its test targets do not compile** (§8.4) |
| a0bbbe93… | **data in** | 21 files, +359/−32 | 3 files, 1,417 lines (`sky-interp/src/data.rs` 826, `inputs.rs` 246, `sky-types/src/data.rs` 345) | 7 | no | `read_csv(path[, column])` gives a dimensionless matrix or vector whose sizes are **run-time sizes** (`ShapeUnifier::fresh_runtime`); `random`, `uniform`, `normal`, `seed` on **SplitMix64**, seed 0 by default, fdlibm-style `ln`, so results match on every machine; `sky run prog.sky L="2.5 m"` command-line inputs | substantial; tests in-file only; spec not updated |
| add6061b… | **units** | 19 files, +691/−90 | 2 files, 414 lines (`parse_units.rs`, `temperature.rs`) plus a stray `.t/` dir | 4 | no | `unit name = value unit` declarations; °C/°F readings vs differences; imperial and lab units | narrow; few tests |
| a7e29e8b… | **linear algebra** | 21 files, +525/−16 | 4 files, 1,337 lines (`sky-linalg/src/eig.rs` 509, `tridiag.rs` 159, `linfn.rs` ×2) | 9 | no | `eigvals`, `eigvecs` (symmetric, `tred2`/`tql2`, deterministic sign and ordering; also the generalized `K x = λ M x`), `outer`, `solve_tridiagonal` (O(n) `dgtsv`-style); new `OpKind::Linalg` | kernels present; spec and goldens not updated |
| afa55e10… (locked) | **ODE** | 18 files, +905/−58 | 10 files, 1,145 lines (`sky-ode/src/events.rs` 376, tests, `examples/projectile.*`, UI goldens) | 21 | spec +, README | events (stop when …, located by Illinois regula falsi on the dense interpolant), dense output `y(t)`, a new example | well tested and the best-documented after reductions, **but one test target does not compile** (§8.4) |

**Do they compile?** Yes for production code; partly for tests. All ten worktrees' libraries and binaries pass `cargo check`. Seven also pass with `--tests`.

Three have test targets that do not compile: tuples, readable output and ODE events. Each one failed to update an exhaustive AST match in `sky-types/tests/expr_types.rs` or `fuzz/harness/parse_module.rs`.

I did not run the test suites, because that is too heavy for the shared machine. Method and logs are in §8.4 at the end of this file. Scripts: `$S/research/r1/check_worktrees{2,3}.sh`; logs: `$S/research/r1/wtcheck2/` and `wtcheck3/`.

### 8.3 Which of them matter for tensor/LLM work (LIKELY, from the content above)

**High relevance**, but each is a partial fix:
- **Arrays you can build.** The `store` op with in-place reuse is the right semantic primitive for KV-cache appends and building RoPE tables. `arange` is needed for positions.
- **Sizes from variables.** Sequence length and KV length are run-time sizes. The worktree only touches the checker, and allows `a·N + b` in one variable. Shapes such as `heads·head_dim` need products of several variables; `sky_types::poly::Poly` already exists and could serve.
- **Reductions.** Softmax, RMSNorm and argmax sampling need them, but they must work along an axis of an N-d tensor (rows of logits, per-head scores). This branch is vector-only.

**Medium relevance:**
- **Broadcasting.** Bias add, the norm weight over rows, masks and per-row softmax all need array–array broadcasting, **which the language forbids by policy**. This is a design conflict, not a missing feature.
- **Tuples.** Returning `(logits, kv)`.
- **Data in.** The deterministic RNG is directly useful for reproducible sampling. CSV is useless for weights of about 18 GB: GGUF or safetensors and mmap are needed.

**Low or no relevance:** readable output (useful for tooling), units, linear algebra (`outer` is marginal), ODE events.

---

## 9. Assessment for the LLM-on-Apple-GPU direction (blunt)

### 9.1 The core mismatch (VERIFIED facts, with LIKELY implications)

| Need for Qwen3-32B Q4 on Metal | Sky today |
|---|---|
| dtypes f16/bf16/int8/int4 plus block scales | f64 only (`Repr` slot exists but no engine reads it) |
| N-d tensors (batch × seq × heads × head_dim) | rank ≤ 2; rank-3 literals rejected; ops vector/matrix only |
| arrays with 10^8+ entries (≈131M for the largest weight) | at most 2^24 entries per array |
| views, strides, reshape, in-place KV-cache updates | no views (slices and transposes copy); immutable; no reshape |
| broadcasting, axis reductions, softmax | forbidden or missing (in-flight work is vector-only) |
| integer tensors (token ids, positions, indices) | no integer type; indices are f64 |
| a GPU backend, device buffers, command queues | none; Cranelift is CPU-only; no Metal bindings in `Cargo.lock` |
| fused attention, quantized GEMV/GEMM kernels | the fuser is elementwise only; kernels are f64, 128-bit NEON |
| binary weight loading, mmap, tokenizer, sampler | none; the language cannot read files on main |
| −inf masks, IEEE inf/NaN semantics | every non-finite value is a runtime error by design |
| speed with FMA, `simdgroup_matrix`, reordered sums | reproducible-by-default forbids FMA and reordering; relaxed mode exists but only for CPU kernels |

### 9.2 Reuse, replace, ignore

**Reuse as-is or nearly so** (LIKELY):
- **`sky-diag`, `sky-syntax`, `sky-ast`, `sky-project`, editions**: the front-end infrastructure, if the LLM model is written in Sky source at all. About 8k production lines.
- **`sky-tune`**: the data-only database keyed by fingerprint × kernel × size class, with integer parameters and a winner-must-be-≥3%-faster rule. It is a ready-made Metal tile and threadgroup tuner store. It needs a GPU-aware fingerprint, which means extending `sky-hw`.
- **`sky-hw`**: keep the CPU facts. Add GPU core count, Metal GPU family, unified-memory bandwidth (measured), max threadgroup memory and SIMD-group width.
- **The `sky-ir` engineering patterns**, reused as code templates rather than as the same op set:
  - arena IR with `u32` ids;
  - a total verifier;
  - a round-trip text format plus a fuzz target;
  - `PassManager` with `verify_each` and `print_after` and an `Explanation{rule, span, because}` record;
  - effect inference;
  - a lifetime analysis that lets operations reuse buffers in place.
- **`sky-egraph`**: the `egg` integration, rule-contract classes and cost-model-driven extraction could host tensor-algebra rewrites (layout, epilogue folding). The 27 current rules are scalar and linear-algebra identities and are not reusable.
- **The cost-model concept** (`sky_ir::cost`: bytes and flops as polynomials of symbolic sizes): it maps naturally onto roofline estimates for decode, where bytes moved dominate. It must become dtype-aware (`ENTRY_BYTES = 8` today).
- **The differential-testing discipline:** golden IR, fuzzed engine-vs-engine oracles, explanations with rule IDs. Keep the discipline, but replace "byte-identical" with tolerance-based or deterministic-per-device oracles for GPU paths.
- **The data-in worktree's SplitMix64 RNG**, for deterministic sampling.

**Must be replaced or written new** (VERIFIED that none of it exists):
- the tensor data model (dtypes, N-d, strides, views, device placement);
- a tensor IR;
- a Metal codegen or kernel library;
- a GPU runtime (buffers, heaps, command buffers, synchronization);
- a weights loader (GGUF/safetensors, mmap, zero-copy `MTLBuffer`);
- a tokenizer (byte-level BPE for Qwen);
- a chat template;
- a KV cache;
- a sampler;
- a generation loop and server/CLI;
- a benchmark harness against MLX, mlx-lm and llama.cpp (tok/s, TTFT, RAM, energy).

**Irrelevant to the LLM headline** (LIKELY): units and dimensions, uncertainty, ODEs, matrix tags (PD/Sym/Diag/Orth), Cholesky/LU/inverse/eig, contracts, the exact and roundoff analyses, AD (inference only), the tree-walker, the REPL, `sky build` AOT for CPU, and `sky-linalg`'s f64 kernels.

**Per-crate verdict for the LLM-on-Metal path** (LIKELY, my judgment from §§2–6):

| crate | verdict | why |
|---|---|---|
| sky-diag | **reuse** | spans and diagnostics, no dependencies |
| sky-syntax, sky-ast | reuse if the model is authored in Sky | front end; needs dtype, tensor, tuple and import syntax |
| sky-project | reuse (small) | safe file opening, manifest, editions |
| sky-hw | **reuse plus extend** | add GPU family, cores and measured bandwidth to the fingerprint |
| sky-tune | **reuse** | data-only database of (fingerprint, kernel, size class) → integer parameters |
| sky-types | partial (only for Sky authoring) | the shape unifier and its diagnostics are good, and `poly::Poly` is useful; units and tags are irrelevant to LLMs; no dtypes |
| sky-units | irrelevant | LLM tensors are dimensionless |
| sky-ir | patterns reuse; op set no | arena, verifier, text format and pass manager are good templates; the f64 L1 op set and the missing memory model do not fit |
| sky-lower | no | AST → SIR for Sky's semantics |
| sky-opt | mostly no | gvn, licm, dce and fold are generic in spirit but coded against `OpKind` and Sky's failure order; `fuse` is a precedent for kernel formation |
| sky-egraph | partial | the `egg` scaffolding and the contract classes and cost hook are reusable; the rules are not |
| sky-ad | irrelevant for inference | keep for later fine-tuning |
| sky-interp, sky-ir-interp | no (CPU reference at most) | the f64 `Value` with a `Dimension` is the wrong data model |
| sky-jit, sky-rt | no | Cranelift targets the CPU only; the runtime operates on `sky_interp::Value` |
| sky-linalg | no | f64 CPU kernels; its fixed reduction-order idea could inform a deterministic mode |
| sky-ode, sky-derive | irrelevant | — |
| sky-cli | reuse (entry point) | add `sky llm …` subcommands and an engine |
| sky-bench | reuse the pattern | the `compare` harness (subprocess, median, spread, floors, RESULTS.md) |

**Overall reuse estimate** (LIKELY): code that carries over directly is about 10–20% of the 75k production lines — the front end, diag, project, tune, hw, egraph scaffolding and IR infrastructure patterns. The performance-critical machinery (`sky-linalg`, `sky-jit`, fused kernels) does not transfer to Metal, f16 or int4.

**Delete?** Not from the repository: these subsystems are the scientific product. Small dead hooks could go: `specialize_for`/`SpecializationCache`, which no engine uses; the empty `MatMulPolicy` default; the duplicated `cholesky_rewrite` the design doc already marks for deletion. **Do not route the LLM path through `sky_interp::Value`**: it is f64 plus a 112-byte `Dimension`, shared by every engine, and coupling to it would drag the scientific semantics into the hot path.

### 9.3 Concrete extension points

#### (a) An N-d tensor type with dtypes

| Where | What changes |
|---|---|
| `sky-ir/src/ir.rs:42` `Repr` (`Unchosen\|Exact\|F32\|F64`) | Extend to `F16`, `BF16`, `I8`, `I32`, and quantized block formats such as `Q4_0`, `Q4_K`, `Q8_0` with group size. This is the designated slot ("chosen by precision passes"), and nothing consumes it today |
| `sky-ir/src/ir.rs:53` `NumberType{dim, shape, tags, repr}` | Already has rank-N `Shape`. Needs strides/layout and device, either here or in a new tensor IR |
| `sky-types/src/check.rs:123` `ValueTy{dim, ty: TensorType}` and `sky-types/src/rules.rs:18` `TensorType` | Add an element type; it would come from the AST |
| `sky-ast/src/ast.rs:212` `TypeExpr{tags, shape, unit}` | Needs dtype syntax. **Conflict:** "a lone name is always a unit", so the token after a shape is a unit (`ast.rs:200-210`). A dtype needs its own syntax, e.g. `Tensor[N, D] of f16` or `f16[N, D]` |
| `sky-types/src/shape.rs:98` `ShapeDim` | Add an `Expr(Poly)` arm for size arithmetic; `sky_types::poly::Poly` already exists |
| Limits | Lift `MAX_ARRAY_ELEMENTS` (`sky-types/src/constant.rs:23`) and make `ENTRY_BYTES` (`sky-ir/src/cost/ops.rs:16`) dtype-aware |
| Runtime value | Do **not** extend `sky_interp::Value::Array` (`sky-interp/src/value.rs:31-35`). Add a separate tensor runtime type: a buffer handle plus dtype, shape, strides and device |

#### (b) A tensor IR above or beside SIR

- The design doc already describes L2 "Strategy" and L3 "Kernel" levels as "the same data structure with a different set of allowed operations" (`docs/design/ir.md:25-34`). None of it is implemented.
- **Option 1: add tensor ops to `OpKind`** (`op.rs:212`). Cost: every `match` over `OpKind` is exhaustive on purpose: the verifier, printer, parser, both interpreters, the JIT, lowering, the e-graph, AD and cost. Adding about 20 tensor ops would touch about 10 crates. That cost is VERIFIED; it is the precise reason `const.symbol` was deferred (`docs/design/ir.md:347`).
  - Measured precedent: the arrays-you-can-build worktree added 5 SIR ops (`store`, `fill`, `linspace`, `arange`, `concat`). Doing so touched 23 tracked files across sky-cli, sky-interp, sky-ir-interp, sky-ir (cost, lifetime, op, parse, print, verify), sky-jit (module, runtime), sky-lower and sky-types, and added 5 new files (+610/−151 tracked, +1,732 new lines). VERIFIED from `git status` and `git diff --stat` in that worktree.
  - Even the rare op `DiagMul` is matched by name in 10 non-test files across 4 crates (grep).
- **Option 2 (recommended, LIKELY): a new crate (for example `sky-tir`)** beside SIR, with its own op set:
  - matmul and quantized matmul, rmsnorm, rope, softmax, attention, gather/embedding, elementwise, reduce over an axis, reshape/transpose views, kv-append, sample;
  - the verifier, printer and pass-manager patterns of `sky-ir`;
  - lowering into a kernel IR or MSL.
- **How the graph gets built:** for the Qwen3 milestone, from GGUF metadata (a Rust builder) rather than from Sky source, which cannot express a transformer today. A Sky-language front end can target the same IR later.
- **Precedent to generalize:** `OpKind::Fused`, a region with a scalar body (`op.rs:467-507`), and the `fuse` pass (`sky-opt/src/fuse.rs`).

#### (c) A GPU (Metal) backend

- No GPU code exists. Grep finds no match for metal, MTL, objc or gpu in `crates/`, apart from README and ROADMAP prose ("GPUs through MLIR", Phase 5).
- **New crate** `sky-metal`: Metal device, queue and buffers through Rust bindings (for example `objc2-metal` or `metal-rs`), kernel source in MSL, a pipeline cache, and command-buffer scheduling.
- **Engine registration:** add a variant to `sky-cli/src/run.rs:67` `Engine` and a branch in the dispatch at `run.rs:219-268`.
- **Kernel tuning:** register tunables with `sky-tune` (`sky-tune/src/lib.rs` `Tunable`/`Registry`) and extend `sky-hw` (`HardwareInfo`, `sky-hw/src/lib.rs:32`) with GPU facts so the fingerprint separates M2 Pro from M5 Max.
- **Toolchain** (EXPERIMENTAL): this machine has Command Line Tools only, so the offline `metal`/`metallib` compiler (part of Xcode) is LIKELY unavailable. Runtime compilation through `newLibraryWithSource:` should work without Xcode. Verify before planning.
- **The ROADMAP's existing GPU plan is "GPUs through MLIR [19, 21, 55]" via `melior`** (`ROADMAP.md`, Phase 5, and the "Building on existing work" table). Going directly to Metal departs from that plan, and the report should say so explicitly. LIKELY: upstream MLIR has no Metal or MSL code generator, so an MLIR route to Apple GPUs would mean SPIR-V plus a translator, for example SPIRV-Cross. That is an extra hop with weaker control over `simdgroup` matrix intrinsics.
- **Dependency policy** (VERIFIED, `deny.toml`):
  - registries: crates.io only (`unknown-git = "deny"`);
  - licenses allowed: Apache-2.0 (with or without the LLVM exception), MIT, BSD-2/3-Clause, ISC, Zlib, Unicode.
  - A Metal binding from crates.io is admissible if its license is on that list; verify the specific crate.
  - The ROADMAP permits porting MIT-licensed code with notices. On that basis, porting MLX (MIT) or llama.cpp/ggml (MIT) Metal kernels is LIKELY compatible; verify the licenses at the time of reuse.

#### (d) A model runtime

- None of it exists: binary file reading, mmap, tokenizer, chat template, generation loop, sampling, KV-cache management, or a streaming CLI.
- **New crate** `sky-llm`, plus a CLI subcommand in the `sky-cli/src/main.rs:70` match, for example `sky llm run model.gguf -p …`.
- **Benchmarking:** reuse the `sky-bench` "compare" harness pattern (`crates/sky-bench/src/compare/`). It already runs subprocesses, takes medians and spreads, records floors and writes a RESULTS.md, and can be adapted to tok/s, TTFT and RSS.
- **Reusable pieces:** the data-in worktree's deterministic RNG; `sky-project`'s safe file opening (open by handle, owner and mode checks) for model paths.

### 9.3b What the Sky *language* would need before a Qwen3 decoder layer could be written in it (LIKELY; each item is VERIFIED missing on main unless noted)

1. N-d tensors of rank 3 to 4 (`[seq, heads, head_dim]`), with reshape and axis permutation. Today: rank ≤ 2, no reshape.
2. dtypes: f16/bf16 activations, int4/int8 block-quantized weights with scales, and casts. Today: f64 only.
3. Integer scalars and tensors, for token ids, positions and lengths. Today: none.
4. Broadcasting between arrays, at least explicit or named. Today: forbidden by the spec.
5. Axis reductions (sum and max over an axis), softmax and argmax. The in-flight work covers vectors only.
6. Gather (embedding lookup by an integer tensor) and its inverse scatter.
7. KV-cache update and run-time sizes. In flight: `store`, and sizes that the checker alone understands.
8. Batched matmul or attention over heads, with grouped-query head sharing (GQA).
9. Records or modules for layer parameters, and an indexable collection of layers (a list of tensors). Today: no structs, lists or imports.
10. Binary input of about 18 GB of weights (mmap), plus byte and string handling for the tokenizer. Today: no file I/O.
11. IEEE −inf and NaN semantics in masked softmax. Today: forbidden.
12. A performance model of bytes per dtype. Today: `ENTRY_BYTES = 8`.

So the "≥8 missing features" in the brief is conservative: by this list there are 12.

### 9.4 Policy conflicts the report must decide (VERIFIED as policy, LIKELY as conflicts)

1. **"Bit-for-bit reproducible by default"** forbids FMA, reordered reductions and vendor GEMMs. Competitive Metal kernels need `simdgroup_matrix` (hardware accumulation order), FMA and split-K or tree reductions. Recommended reading for the LLM path: "deterministic run-to-run on one device and build", not "identical across engines and machines". Otherwise ship relaxed as the LLM default, which the spec currently forbids from environment variables or pragmas.
2. **"Never a silent NaN or inf"**: masked attention and log-softmax need −inf and IEEE semantics. Make the check a debug or sanitizer mode on GPU.
3. **"No broadcasting between arrays"** and **"`^` on arrays is an error"**: tensor code needs NumPy-style broadcasting, or at least explicit, named broadcasting.
4. **Three engines must agree byte for byte**: a GPU engine cannot meet this against an f64 CPU tree-walker. Define a new oracle relationship, such as an f32 CPU reference within a tolerance, or token-level agreement on greedy decoding.

### 9.5 Open questions

- **Q1:** Is the LLM runtime meant to be written *in Sky*, which is a language project needing tensors, dtypes, broadcasting, integers and I/O first, or *by Sky's toolchain*, as a Rust runtime plus compiler infrastructure? This decides whether the front-end work in §8 is on the critical path.
- **Q2:** Does the reproducibility promise apply to the LLM product? See §9.4.
- **Q3:** Should the tensor IR be a level of SIR, as `ir.md` "Levels" envisions, or a sibling crate? Exhaustive `OpKind` matches make the first option expensive.
- **Q4:** Should the ten uncommitted worktrees be committed or abandoned? None is merged, and three of them (arrays-you-can-build, sizes, reductions) are the foundations a Sky-language tensor front end would need.
- **Q5 (EXPERIMENTAL):** Metal toolchain availability without Xcode, and the reach of runtime MSL compilation.

---

## Appendix A — probes and measurements (raw files under `$S/research/r1/probe/`)

| probe | command | result |
|---|---|---|
| `sky hw` | `~/sky/target/release/sky hw` | M2 Pro, neon, 10 cores (6P/4E), L1d 128 KiB, L2 16 MiB, L3 unknown, 16 GiB, fingerprint 448c927493c66f47 |
| `p1_silu_naive.sky` | `sky check` | `error: cannot add a vector to a scalar … no broadcasting between a scalar and an array` |
| `p2_zeros_n.sky` | `sky check` | ``error: `zeros` needs a constant size, found `len(h)` `` |
| `p3_reduce.sky` | `sky check` | ``error: cannot find `sum` in this scope`` |
| `p4_broadcast.sky` | `sky check` | `error: cannot add a vector to a matrix … rank 2 vs rank 1` |
| `p5_rank3.sky` | `sky check` | `error: array literals have at most two levels` |
| `p6_tensor_ann.sky` | `sky check` | `ok` (a `Tensor[2,2,2]` annotation is accepted) |
| `p7_mlp.sky` | `sky check --explain`, `--emit-ir`, `--emit=clif`; `sky run --engine={ast,sir,native}` | identical output on all three engines; SiLU fused; CLIF of 251 lines dominated by runtime calls |
| `p8_gemv.sky` vs `p8_setup.sky` | `bash $S/benchlock.sh bash run_gemv.sh` (5 runs each, median) | see §5.5; raw in `gemv_results.txt` |
| `p9_*` per-op overhead | `bash $S/benchlock.sh bash run_ops.sh` | see Appendix B |
| `sky-jit` kernels bench (finiteness-check cost) | `bash $S/research/r1/run_kernel_bench.sh` (`cargo bench -p sky-jit --bench kernels`, scratch target dir) | see §6.2; raw in `$S/research/r1/kernel_bench/run.log` |
| worktree compile checks | `check_worktrees2.sh`, `check_worktrees3.sh` | see §8.4; logs in `wtcheck2/` and `wtcheck3/` |
| lines of code | `python3 $S/research/r1/loc.py ~/sky/crates` | §2.1; raw in `$S/research/r1/loc_main.txt` |

## Appendix B — per-op overhead probe

EXPERIMENTAL-quality measurement: median of 5 runs, release `sky` from `main`. Load average was about 4.8 because other agents had processes running outside the lock. Raw data: `$S/research/r1/probe/ops_results.txt`.

The loop is `for i in range(1000000): v = v + w`.

| program | engine | median wall | per iteration (minus setup ≈ 9.8 ms) |
|---|---|---:|---:|
| `p9_op32_0` (setup plus 1 iteration, 32-entry vectors) | native | 9.84 ms | — |
| `p9_op32` (1e6 iterations, **32 entries**: a handle, so each op is a runtime call) | native | 301.1 ms | **≈291 ns per array op** |
| `p9_op8` (1e6 iterations, **8 entries**: a small array, scalar-replaced inline) | native | 13.07 ms | **≈3–4 ns per op** (8 adds plus checks) |
| `p9_scalar` (1e6 scalar adds) | native | 6.90 ms | ≈ the process floor (the loop costs ~1–2 ms) |
| `p9_op32` | SIR interpreter | 342.1 ms | ≈330 ns per op |

What the numbers show (LIKELY):
- Crossing the 16-entry `MAX_SMALL` line costs about 75–100× per elementwise op, because the op becomes a runtime call on `Rc<Value>` handles. This explains why `nbody` (5-body state arrays of more than 16 entries) runs 7.5× slower than C (`ROADMAP.md`, Phase 4 status).
- Used as a host-side dispatcher, about 300 ns per op × about 1,300 ops per token would be about 0.4 ms per token. That is not a bottleneck for GPU decode, but these runtime paths are CPU f64 compute, not a dispatcher.

## §8.4 — worktree compile check results (VERIFIED, our runs)

**Method.** I ran `cargo check --manifest-path <worktree>/Cargo.toml --workspace --tests --locked` with toolchain 1.85.1 and `CARGO_TARGET_DIR=$S/research/r1/wtcheck-target`, under `benchlock`. That directory and the bench target directory were deleted afterwards to save disk space; the logs are kept. The worktrees were not modified. Script: `$S/research/r1/check_worktrees2.sh`; logs: `$S/research/r1/wtcheck2/`.

**A first attempt was invalid; its logs are kept for the record.** Cargo hashes path packages relative to the workspace root and judges freshness by mtime. With a shared target directory, it treated every worktree's sources (dated 17:20) as already fresh against main's newer artifacts, and reported a false "OK" in 0.2 s. The corrected script deletes the `sky-*` fingerprints before each worktree, which forces all 22 sky crates to be re-checked. The invalid logs are in `$S/research/r1/wtcheck-invalid-shared-fingerprints/`.

Baseline: `main` checks cleanly (rc=0, 0 errors, 0 warnings).

| worktree | feature | `check --workspace --tests` | first error |
|---|---|---|---|
| a0952514 | arrays you can build | **OK** (22 crates, 0 warnings) | — |
| abf8af99 | sizes from variables | **OK** | — (but it touches only the checker) |
| a3be8cf3 | reductions | **OK** | — |
| a383df0a | broadcasting | **OK** | — |
| a0bbbe93 | data in | **OK** | — |
| a7e29e8b | linear algebra | **OK** | — |
| add6061b | units | **OK** | — |
| a1dadc02 | readable output | **FAIL** | `fuzz/harness/parse_module.rs:311`: non-exhaustive match, the new `ExprKind::FString` and `ExprKind::Keyword` are not covered; the test target `sky-ast` `fuzz_parse_module` was not updated |
| ad1ea58e | tuples | **FAIL** | the same harness, at `:311` (`ExprKind::Tuple`) and `:396` (`AssignTarget::Tuple`) |
| afa55e10 | ODE (locked) | **FAIL** | `crates/sky-types/tests/expr_types.rs:198`: non-exhaustive, `SolveOption::Until` not covered |

All three failures are **test or fuzz-harness targets that were not updated for new AST variants**. That is a sign the work stopped part-way.

**Follow-up run** (script `$S/research/r1/check_worktrees3.sh`; logs `$S/research/r1/wtcheck3/`):
- Libraries and binaries only (`cargo check --workspace`): **all three compile cleanly** (rc=0, 0 errors).
- Tests with `--keep-going`:

  | worktree | errors | failed test targets |
  |---|---:|---|
  | a1dadc02 (readable output) | 2 | `sky-types` `expr_types.rs:40` and the fuzz harness `parse_module.rs:311` (`FString`, `Keyword`) |
  | ad1ea58e (tuples) | 4 | `expr_types.rs:40,110` and `parse_module.rs:311,396` (`ExprKind::Tuple`, `AssignTarget::Tuple`) |
  | afa55e10 (ODE) | 1 | `expr_types.rs:198` (`SolveOption::Until`) |

So the production code of all ten worktrees compiles. What is stale is the exhaustive AST matches in two test files (`sky-types/tests/expr_types.rs` and `fuzz/harness/parse_module.rs`).

**Not done:** running the test suites of the worktrees. That would be too heavy for the shared machine. "Compiles" is therefore a weak completeness signal.
