# Part VII — Tensor IR (TIR)

This part specifies TIR, the intermediate representation of Sky's tensor programs:
- its data model, types, operations and their type rules;
- how state (the KV cache, recurrent state) is represented and lowered in place;
- what is static, symbolic or bucketed;
- the verifier, the text format and provenance;
- the reference interpreter and the numerics of every operation under the `inference` contract (SPINE D6);
- the Rust model-builder API;
- the Sky language surface that produces TIR (SPINE D8), and its lowering.

The graph optimizer that rewrites TIR is [Part VIII](part08-graph-optimizer.md). Fusion profitability, kernel boundaries and the memory plan are [Part IX](part09-scheduler.md) and [Part XIX](part19-memory.md). The quantized-format descriptors carried by TIR types are owned by [Part XII](part12-quantization.md); this part owns the type system and text syntax around them. The builders that emit TIR for each model family are [Part XIII](part13-llm-runtime.md).

TIR is Sky's *Tensor IR*. It is not TVM's TensorIR, the loop-level IR of Apache TVM; the two share an acronym and nothing else.

Evidence labels follow SPINE §8. Research notes are cited as (R2 §3.1) and ship as [`research/R2-tinygrad.md`](research/R2-tinygrad.md) and so on. Repository paths are relative to the Sky root.

## 0. Summary

| # | Decision | Section |
|---|---|---|
| VII-D1 | TIR lives in the new crate `sky-tir`, as a sibling of SIR. SIR reaches it only through one operation, `tensor.call`. The Rust model builders (`sky-llm::arch`) emit TIR directly. | §1, §14 |
| VII-D2 | A graph is an **arena-interned (hash-consed) SSA DAG**. A node is `(op, srcs, attrs)` plus a span. Structurally equal nodes are one node. Types are **derived** by a total function at construction; everything else (uses, ranges, provenance) lives in side tables. | §3 |
| VII-D3 | `Type { dtype, shape, layout }`. `DType::Quant` carries Part XII's `QFormat` and `QPacking`. A dimension is `Const(u64)` or `Sym(Poly)` over a table of **bounded symbols** (interval plus divisibility), so `P: 0..40960` is a type-level fact. | §4 |
| VII-D4 | Views (reshape, permute, slice, broadcast, pad, concat) are operations whose meaning is an **index function**, as in tinygrad's rangeify. L1 has no strides. Physical layouts exist only on parameters and state, after layout assignment. | §4.4, §4.5 |
| VII-D5 | TIR is **strictly typed**: no implicit broadcasting and no implicit dtype promotion. The language's broadcasting by size names lowers to explicit `broadcast` views. | §5.1 |
| VII-D6 | About 50 primitive operations, plus **composites** (`rms_norm`, `rope`, `softmax`, `attention`, `swiglu`, `sample`, Gated DeltaNet), each *defined* by a reference decomposition into primitives. | §5 |
| VII-D7 | State is **functional and linear**: `update` returns a new version of a state value, every old version is read before the update, and the final version is returned. It is lowered in place. Graphs have no other effects; guards on run-time sizes run before a step. | §6 |
| VII-D8 | The token count `T` is a **plan symbol** (specialized per plan). The position `P` is a **run-time symbol** with bounds. The precision plan (every narrowing cast) is part of the L1 graph, so every plan rounds at the same places, which batch and chunk invariance need. | §7 |
| VII-D9 | The **reference interpreter** (f32, and f64 for checks) defines the meaning of every graph. §11 gives each operation's numerics and tolerance class under `inference`. | §11 |
| VII-D10 | The **language surface** is edition-gated: storage formats (`in f16`, `in q4(group: 64)`), broadcasting by matching size names, reductions `over:` a named axis, records for weights, weights as value-less declarations, `Index[N]` and size products. Named arguments have one spelling, `name: value`. | §13 |

## 1. Position

### 1.1 Where TIR sits

```text
  models/qwen3.sky ──► sky-syntax ─► sky-ast ─► sky-types (tensor types, edition 2027)
                                                    │
                                     sky-lower ─────┼──► SIR ──tensor.call @logits──┐
                                                    │                               │
                                                    └──► TIR (lower_tensor) ──┐     │ runs
  hf://org/model ─► sky-llm::arch::<family>::build ──► TIR ────────────────────┤     │
                                                                               ▼     ▼
                      ┌──────────────────────── sky-tir ─────────────────────────────────┐
                      │ L1 graph: Module { syms, qtypes, graphs }                        │
                      │   verify · text (print/parse) · interp (f32/f64 oracle)          │
                      │   passes (Part VIII) ──► L2 strategy annotations (Parts IX, XII) │
                      └──────────────────────────────┬───────────────────────────────────┘
                                                     ▼
                         sky-kernel (L3 kernel IR) ─► sky-metal (L4: MSL, runtime)
```

The levels mirror `docs/design/ir.md` ("Levels"): one data structure per level family, with the verifier deciding which operations and attributes are legal at which level. TIR has two levels, both in `sky-tir`:
- **L1 (graph):** typed operations on tensors; no algorithm, layout or kernel choices.
- **L2 (strategy):** the same graph with strategy attributes on matmuls, attention and reductions (algorithm variant, reduction schedule, storage and KV formats). Those attributes are chosen by Parts IX, XII, XIV and XVI; §7.3 and §8 say what the verifier requires of them.

L3 (kernel IR) and L4 (MSL) are different data structures in `sky-kernel` and `sky-metal`. TIR deliberately does not do what tinygrad does with its single UOp type, which also represents the linearized kernel and the host-side submission code (R2 §0 item 1). Kernel-level and host-level concerns have their own IRs here, so the tensor graph stays small and its invariants stay checkable.

### 1.2 Why a sibling and not more SIR

SPINE D1 and [Part VI](part06-architecture.md) give the decision. The facts behind it, all [VERIFIED: R1 §5.1, §9.3]:
- SIR values are f64 only, rank ≤ 2 and at most 2^24 entries; one Qwen3-32B MLP weight is 7.8× over that cap.
- SIR has no memory model: no buffers, dtypes, views or devices.
- Every array operation in every engine runs through `sky_interp::Value`.
- `OpKind` is matched exhaustively in about 10 crates. Adding five array operations on the arrays-you-can-build branch touched 23 files.
- The three-engine byte-identical rule cannot hold between a GPU and an f64 oracle.

### 1.3 What TIR takes from SIR

| SIR infrastructure | Where | TIR counterpart |
|---|---|---|
| Arena IR with `u32` ids | `crates/sky-ir/src/ir.rs` | `NodeId(u32)` arenas per graph (§3) |
| Total verifier, never panics, errors with spans | `crates/sky-ir/src/verify.rs:1-60` | `sky_tir::verify` (§8), same `VerifyError` shape |
| Round-trip text format with a fuzz target | `crates/sky-ir/src/{print,parse}.rs`, `fuzz/fuzz_targets/sir_parse.rs` | `sky_tir::text` (§9), fuzz target `tir_parse` |
| `Pass`, `PassManager`, `verify_each`, `print_after` | `crates/sky-ir/src/pass.rs` | `sky_tir::pass` with the same API (Part VIII) |
| `Explanation { pass, rule, span, before, after, because, related }` and `PassCx::explain` | `crates/sky-ir/src/pass.rs` | the same type, moved to `sky-diag` and re-exported by both IRs (VII-T2) |
| Spans and operand spans on every operation | `docs/design/ir.md` "L1 data model" | spans plus a provenance side table (§10) |
| `sky_types::poly::Poly` for symbolic sizes | `crates/sky-types/src/poly.rs` | dimensions and symbolic sizes (§4.3) |
| Lifetime analysis that reuses dying buffers | `crates/sky-ir/src/lifetime.rs` | state linearity and in-place `update` (§6) |
| Rule IDs and `--explain` notes | `sky check --explain` | the same notes from `sky explain` (Part VIII §8) |

What TIR does *not* take: `sky_interp::Value`, the f64 value layer, the "every non-finite value is an error" rule inside graphs, and the byte-identical-engines rule. SPINE D6 replaces those three for tensor programs.

## 2. Lessons taken from other systems

| Lesson | Evidence | What TIR does |
|---|---|---|
| One hash-consed SSA graph with **derived** dtype and shape, and a uniform rewrite engine, makes CSE free and lets every cache key on a structural hash | tinygrad `UOp(op, src, arg, tag)`, dtype from the total function `dtype_from_uop`, hash-consing in `UOpMetaClass.__call__` (`uop/ops.py:122-213`, `240-251`) (R2 §3.1) | §3: interned `Node`, `infer` at construction, `Graph::structural_hash` |
| Provenance belongs in a side table, or hash-consing would keep two nodes apart because of their source positions | tinygrad keeps `metadata` in a side `WeakKeyDictionary`, not in the node (`uop/ops.py:216`) (R2 §3.1) | §10: the span is not part of node identity; merged nodes keep every origin |
| Views as index functions ("rangeify") make views free and give one fusion rule: a node fuses into its consumers when they agree on its index | `apply_movement_op` (`schedule/indexing.py:169-187`); consumer agreement (`schedule/indexing.py:230-277`) (R2 §3.3, §5.1) | §4.5 here; the legality rule is Part VIII §6 |
| Symbolic integers need intervals and divisibility, not just names | `UOp.variable(name, min, max)`, `_min_max`, `divides`, `multiple_of` (`uop/ops.py:1014-1165`) (R2 §3.6) | §4.3: `SymTable` with bounds and `multiple_of` |
| Per-layer bodies should be built and scheduled once | `@function(precompile=True)`: 28 layers share one schedule (`llm/model.py:176-183`) (R2 §4) | §5.11: one layer graph, called once per layer |
| A quantized weight must be a **type**, not a decode expression fused by a generic code generator | tinygrad's lazy GGUF decode expressions run its fused Q4_0 GEMV at 53 GB/s, about 26% of the M2 Pro's peak; dequantizing to fp16 at load was faster (76.3 vs 54.3 tok/s on Qwen3-0.6B) (R2 §8.2–§8.4) | `DType::Quant` with Part XII's descriptors; quantized operands only in `qmatmul` and `gather_rows`, which map to hand-designed skeletons |
| Implicit broadcasting in the IR hides intent | tinygrad broadcasts implicitly in elementwise ops (`uop/ops.py:440-447`) (R2 §3.1); Sky's spec forbids array broadcasting because each case "would be a guess about which axis is meant" (`spec/v0.1.md`, "Arrays: entry by entry") | TIR has explicit `broadcast` only; the language broadcasts by names (§13.4) |
| Fusing only elementwise operations at run time leaves the important fusions on the table | `mx.compile` fuses only unary, binary, ternary and broadcast ops, at most 11 deep; reductions, matmuls, norms, RoPE and SDPA never fuse; mlx-lm compiles only the SwiGLU (R3 §1.4) | composites are first-class, so Part VIII and Part IX can fuse norms into GEMV prologues and RoPE plus KV append into epilogues |
| Symbolic shapes only where something is truly dynamic | Relax's general symbolic-shape machinery is "overkill when shapes come from types" (R7 Topic 2, Borrow) | only `P` (and derived `S = P + T`) is a run-time symbol; `T` is bucketed (§7) |
| Layout assignment and buffer assignment are passes | XLA (R7 §4a) | Part VIII §3.11 and Part XIX |

What we do not copy from tinygrad (R2 §9.2, §9.3):
- **Run-time laziness.** It exists to discover fusion through an eager Python API and to capture graphs for its JIT. Sky sees the whole program at compile time. [Part VIII](part08-graph-optimizer.md) §1 gives the argument and the costs measured.
- **Heuristic recompute-versus-store rules** ("> 3 buffers", "never recompute a reduction", `schedule/rangeify.py:84-96`). TIR records legality only; profitability is a cost-model decision in Part IX.
- **Python compile latency.** One Qwen3-0.6B block costs 76–151 ms to schedule on a cache miss in tinygrad (R2 §3.7). A Rust implementation of the same rewrites is expected to be 10–100× faster [LIKELY: typical Rust vs Python for pointer-chasing code (R2 §9.2 item 6); measured by VII-M3].

## 3. Data model

### 3.1 Module, graphs and nodes

```rust
// crates/sky-tir/src/graph.rs

/// A compilation unit: the symbols, quantized types and graphs of one model
/// (or one Sky file's tensor functions).
pub struct Module {
    pub version: u32,                       // text format version, printed `tir version 1`
    pub syms: SymTable,                     // §4.3
    pub requires: Vec<Constraint>,          // `require P + T <= C`
    pub qtypes: Vec<QType>,                 // interned (QFormat, QPacking) pairs, §4.2
    pub graphs: Vec<Graph>,                 // `@qwen3_layer`, `@qwen3_step`, ...
}

/// One function: parameters, a DAG of nodes, results.
pub struct Graph {
    pub name: Arc<str>,
    nodes: Vec<Node>,                       // arena; NodeId(i) indexes it; srcs < self
    tys: Vec<NodeTy>,                       // derived, parallel to `nodes` (§3.3)
    intern: HashMap<NodeKey, NodeId>,       // hash-consing table (§3.2)
    prov: Vec<Provenance>,                  // side table, parallel to `nodes` (§10)
    pub params: Vec<NodeId>,                // the `Op::Param` nodes, in signature order
    pub results: Vec<NodeId>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct NodeId(pub u32);

pub struct Node {
    pub op: Op,                             // fieldless #[repr(u16)] enum: fast dispatch, rule indexing
    pub srcs: SmallVec<[NodeId; 4]>,
    pub attrs: Attrs,                       // per-op payload: axes, eps, shapes (Dims), dtypes, ...
    pub span: Span,                         // first origin; NOT part of identity (§3.2, §10)
}

/// A node's derived type: one tensor, or several for multi-result nodes
/// (`call`, `topk`, `deltanet.scan`), read through `proj[i]`.
pub enum NodeTy { One(Type), Many(SmallVec<[Type; 2]>) }

pub struct Type {
    pub dtype: DType,                       // §4.1
    pub shape: Shape,                       // §4.3
    pub layout: Layout,                     // §4.4; `Layout::Logical` at L1
}
```

Every value that is not a tensor is a rank-0 tensor (`i32[]` for a position, `u32[2]` for a sampling key). There is no separate scalar type.

`Op` is a fieldless enum so that the rewrite engine can index rules by the root operation (the PatternMatcher lesson, [Part VIII](part08-graph-optimizer.md) §4). Payloads live in `Attrs`, an enum with one variant per payload shape; floating-point attributes are stored as bits (`F64Bits(u64)`) so that `Attrs: Eq + Hash`.

### 3.2 Interning

`GraphBuilder::node(op, srcs, attrs, span) -> Result<NodeId, TypeError>`:
1. builds the key `NodeKey { op, srcs, attrs }` (the span is excluded);
2. if the key is in `intern`, appends `span` to that node's provenance and returns its id;
3. otherwise calls `infer` (§3.3); on success pushes the node, its type and its provenance, and returns the new id; on failure returns a `TypeError` that carries the span.

Consequences:
- **CSE is a construction invariant.** Two `rms_norm`s of the same value with the same weight are one node. Part VIII's `cse` pass only re-establishes the invariant after bulk rewrites.
- **The graph is acyclic by construction.** A node's sources exist before it, so `srcs[i] < id`. The verifier checks it anyway (rule V1).
- **Ids are deterministic.** They follow construction order, which builders and lowering make deterministic. No pass iterates a `HashMap` to decide anything; worklists run in `NodeId` order. Printing and hashing therefore do not depend on hash seeds.
- **Parameters are nodes** (`Op::Param` with `Attrs::Param { index, name, role }`), so rules see them like any other node, and two parameters are never merged because their `index` differs.

`Graph::structural_hash()` is a sha256 over the canonical text of §9 without provenance. Part X uses it as the tuning-database key of a block, and Part XIII uses it as the plan-cache key, as tinygrad keys its schedule cache (R2 §4).

### 3.3 Derived types

`crates/sky-tir/src/infer.rs`:

```rust
/// The type of `op` applied to sources of types `srcs` with `attrs`, or why not.
/// Total: every combination returns Ok or Err; nothing panics.
pub fn infer(op: Op, srcs: &[&NodeTy], attrs: &Attrs, cx: &InferCx) -> Result<NodeTy, TypeError>;

pub struct InferCx<'m> { pub syms: &'m SymTable, pub qtypes: &'m [QType], pub requires: &'m [Constraint] }

pub struct TypeError { pub rule: &'static str, pub message: String, pub span: Span, pub notes: Vec<String> }
```

`infer` holds the type rules of §5 and nothing else. The verifier recomputes every node's type and compares it with the cached one (rule V3), so a pass that edits a node behind the builder's back is caught. Unlike tinygrad's `dtype_from_uop`, `infer` can fail: TIR is strictly typed, and a failure is a builder or lowering bug reported at the source span.

### 3.4 Side tables and analyses

Analyses are computed on demand and cached against `Graph::version` (incremented by every mutation):

| Analysis | Type | Used by |
|---|---|---|
| `uses(id)` | `SmallVec<[NodeId; 4]>` | DCE, fusion legality, state linearity |
| `kind(id)` | `ValueKind::{Value, Weight, State}`: `Weight` for a weight parameter and any view of one; `State` for a state parameter and every `update` result | relayout legality (weights are constants), V11 |
| `int_range(id)` | `Option<(i64, i64)>` for integer values: from `range` on parameters, `iota`, `symval`, `argmax`, arithmetic | `gather_rows` bounds (V13), index width (V22) |
| `index_fn(consumer, producer)` | `IndexFn` (§4.5) | view folding, fusion legality (Part VIII §6) |
| `provenance(id)` | `&Provenance` | explanations, `sky explain`, kernel names |
| `liveness` | first and last use per node | the interpreter frees values; Part XIX plans memory |

### 3.5 Rewriting an immutable graph

Nodes are never edited in place. A rewrite records `subst: NodeId → NodeId` and calls `Graph::rebuild(&subst)`, which walks the results in id order, re-interns every reachable node with substituted sources, and drops unreachable nodes. This is tinygrad's `graph_rewrite` discipline (R2 §3.7) with SIR's arena discipline: no dangling ids, no stale analyses (the version changes), and CSE is restored by the re-interning. Part VIII §4 describes the driver.

### 3.6 Crate layout

```text
crates/sky-tir/src/
  lib.rs
  graph.rs        Module, Graph, Node, NodeId, interning, rebuild
  dtype/mod.rs    DType, Type, Layout
  dtype/qformat.rs  QFormat, QPacking (owned by Part XII), QType
  sym.rs          SymTable, SymInfo, Constraint; prove_le, range, divides
  shape.rs        Dim, Shape
  op.rs           Op, Attrs, OpInfo table (arity, effects, composite flag)
  infer.rs        type rules
  index.rs        IndexFn, view composition, the index simplifier
  composite/      rms_norm.rs rope.rs softmax.rs attention.rs glu.rs sample.rs deltanet.rs
  verify/         mod.rs (V1–V24), quant.rs (QT1–QT8 of Part XII), state.rs, plan.rs
  text/           print.rs parse.rs  (grammar in parse.rs docs)
  build.rs        TirBuilder, GraphBuilder (§12)
  interp/         mod.rs eval.rs dequant.rs (Part XII) composite.rs rng.rs
  pass.rs         Pass, PassManager (as sky-ir), re-exports sky_diag::Explanation
  passes/         Part VIII
  plan/           PlanSpec, Specialization, ContractLog, memory.rs (Part XIX)
  cost.rs         bytes and FLOPs as Poly in the plan's symbols
```

Dependencies: `sky-diag`, `sky-types` (for `Poly`), `smallvec`. `sky-quant` depends on `sky-tir`, not the reverse ([Part XII](part12-quantization.md) §2.2; [Part VI](part06-architecture.md) §7). `sky-egraph` is added only when the e-graph work of Part VIII §5 starts.

## 4. Types

### 4.1 Element types

```rust
// crates/sky-tir/src/dtype/mod.rs
pub enum DType {
    F32, F16, BF16,
    F64,              // reference interpreter only; the verifier rejects it at L2 for Metal
    I32, U32, I16, U16, I8, U8,
    Bool,
    Quant(QTypeId),   // index into Module::qtypes: a (QFormat, QPacking) pair, §4.2
}
```

| DType | Bytes | Where it is used | Notes |
|---|---|---|---|
| `f32` | 4 | activations between kernels where precision matters, accumulators, logits, DeltaNet state | the compute type of every floating operation (§11.2) |
| `f16` | 2 | KV cache (SPINE §2), operands of neural-accelerator GEMM | max 65504 |
| `bf16` | 2 | checkpoint norms, scales, activations fed to matmuls | Qwen3's release dtype (R9 §3) |
| `i32` | 4 | token ids, positions, indices, `argmax` results | every index proven to fit (V22) |
| `u32` | 4 | RNG keys and bits, packed MLX codes inside `QPacking` | |
| `i8`, `u8` | 1 | int8 KV (labeled mode), raw blocks | |
| `bool` | 1 | masks, comparison results, `check.finite` | |
| `Quant(q)` | per format | weights; quantized KV in a labeled mode | §4.2 |

There is no `f64` on the GPU (Metal has no double) and no `i64`. Integer values that could exceed `i32` are rejected by V22 rather than silently widened.

### 4.2 Quantized types

[Part XII](part12-quantization.md) §2.2 defines `QFormat` (the numeric meaning: code, block, sub-blocks, scale, offset) and `QPacking` (where the bits sit), both as plain data in `crates/sky-tir/src/dtype/qformat.rs`. TIR adds the pair and the rules that use it:

```rust
pub struct QType { pub format: QFormat, pub packing: QPacking }   // interned in Module::qtypes

impl QType {
    /// The axis that is blocked, the block length, and the logical element type
    /// of a dequantized element (f32 in the interpreter; Part XII §2.4).
    pub fn block_axis(&self) -> BlockAxis;
    pub fn block(&self) -> u32;
    pub fn bits_per_weight(&self) -> Ratio;               // exact: 9/2 for mlx.affine(b4,g64,*)
    /// The storage planes a parameter of this type binds to, with their shapes:
    /// MLX: codes u32[N, K*b/32], scales and offsets [N, K/g]; GGUF: one block plane.
    pub fn planes(&self, logical: &Shape) -> SmallVec<[PlaneDecl; 3]>;
}
```

- A tensor of type `mlx.affine(b4,g64,bf16)[8192, 5120]` has the **logical** shape `[8192, 5120]`. Its storage is the planes `codes u32[8192, 640]`, `scales bf16[8192, 80]`, `biases bf16[8192, 80]` [VERIFIED: R9 §1.2 naming and shapes].
- A parameter of a quantized type binds to all of its planes at once; the loader supplies them ([Part XIII](part13-llm-runtime.md) §6).
- Part XII's rules QT1–QT8 (§2.5 there) are TIR verifier rules, implemented in `crates/sky-tir/src/verify/quant.rs`. In TIR terms:
  - a quantized value may be an operand only of `qmatmul` (as the weight), `gather_rows` (as the table), `dequant`, `relayout`, `concat` and `slice` along the non-blocked axis, and `call` (into a `weight` parameter);
  - `quantize` may not appear in an inference graph (QT5); `quantize_dynamic` and an `update` into a quantized state are legal only under a labeled mode recorded in the plan (QT7).
- **Printing.** The type prints as the format's canonical name followed by the logical shape: `mlx.affine(b4,g64,bf16)[8192, 5120]`, `gguf.q4_k[25600, 5120]`. A packing other than the format's native one is printed after the shape: `... packing(split_planes, align 16384)`.

With a coarser type the reader would lose bytes: the cost model reads `bits_per_weight` (4.5 for MLX affine 4-bit group 64, Part XII §2.3), so the bytes a step streams are a property of the graph, not of a kernel.

### 4.3 Shapes and symbols

```rust
// crates/sky-tir/src/shape.rs
pub enum Dim {
    Const(u64),
    Sym(Poly),        // sky_types::poly::Poly over symbol names; never a constant polynomial
}
pub struct Shape(pub SmallVec<[Dim; 4]>);

// crates/sky-tir/src/sym.rs
pub struct SymInfo {
    pub name: Arc<str>,
    pub kind: SymKind,
    pub lo: u64,                 // inclusive
    pub hi: u64,                 // inclusive
    pub multiple_of: u64,        // 1 = no divisibility fact
}
pub enum SymKind {
    Static(u64),                 // a named constant, e.g. C = 40960 (printed by name for readability)
    Plan,                        // bound to a constant per plan (T); symbolic in the generic graph
    Runtime,                     // known when a step runs (P); a kernel argument
}
pub struct Constraint { pub lhs: Poly, pub rel: Rel, pub rhs: Poly }   // `P + T <= C`
```

`Poly` is `sky_types::poly::Poly`: exact rational coefficients over monomials in named sizes, capped at 256 terms and degree 64, with overflow reported as `None` [VERIFIED: `crates/sky-types/src/poly.rs:1-30`]. Its comparison `le_for_all_sizes` assumes every size is at least 1 (`poly.rs:326`). A position can be 0, so TIR does not use that function. `SymTable` implements its own sound tests with the public `Poly::subst` and `Poly::all_coefficients_nonnegative`, and sky-types does not change (as [Part XXIV](part24-roadmap-now.md) #4 asks):

| Function | Method | Answers |
|---|---|---|
| `prove_le(p, q)` | (1) substitute `x = lo_x + y_x` for every symbol and require every coefficient of `q - p` in the `y` to be non-negative; (2) else interval arithmetic: `max(p) <= min(q)` over `[lo, hi]`; (3) else, for each `require a <= b`, retry (1) and (2) on `q - p - (b - a)` | `Proven` or `Unknown` (never a wrong `Proven`) |
| `range(p)` | interval arithmetic over the bounds | `(min, max)` as exact rationals |
| `divides(k, p)` | every term `c·∏xᵉ` has `k` dividing `c·∏ multiple_of(x)ᵉ` | `true` only if proven |
| `numel_fits_i32(shape)` | `range(∏ dims).max < 2^31` | used by V22 |

Bounded integers are the type-level form of tinygrad's `UOp.variable(name, min, max)` with `multiple_of` (R2 §3.6). Example: with `P: 0..40960`, `T: 1..=8192` and `require P + T <= 40960`, `prove_le(P + T, C)` succeeds by step (3), so slicing the cache to `P + T` positions needs no run-time check in the graph; the guard runs once per step on the CPU (§6.3).

Symbols in the language surface are the same objects: `pos: Index[C]` declares a run-time symbol with bounds `0..C` (§13.3).

### 4.4 Layout

```rust
pub enum Layout {
    Logical,                         // L1: no physical layout chosen (row-major if materialized)
    RowMajor { align: u32 },         // a materialized buffer: parameters, state, plan outputs
    Strided { strides: SmallVec<[Dim; 4]>, offset: Dim },   // state views kept by layout assignment
    Packed(RelayoutId),              // a weight after a lossless relayout recipe (Part XII §5.3)
}
```

- L1 values are `Logical`. Intermediate values never get a physical layout in TIR: Part IX decides which intermediates exist in memory at all, and Part XIX places them.
- Parameters and state carry a layout from the start (`RowMajor` as loaded), because they are real buffers.
- Layout assignment (Part VIII §3.11) changes weight layouts to `Packed(..)` only through Part XII's lossless recipes (RL1–RL7), and chooses the KV layout once for all plans.

### 4.5 Views as index functions

A view operation does not move data. Its meaning is a function from the coordinates of its output to the coordinates of its input, plus an optional validity predicate. This is tinygrad's rangeify representation (R2 §3.3) with Sky's symbolic sizes.

```rust
// crates/sky-tir/src/index.rs
pub enum Ix {
    Coord(u8),                       // output coordinate r_k
    Sym(SymId),                      // P, T, ... (run-time or plan symbols)
    Const(i64),
    Add(Box<Ix>, Box<Ix>),
    Mul(Box<Ix>, i64),
    FloorDiv(Box<Ix>, i64),
    Mod(Box<Ix>, i64),
}
pub struct IndexFn {
    pub coords: SmallVec<[Ix; 4]>,   // one input coordinate per input axis
    pub valid: Option<Pred>,         // pad and concat pieces: where the input is read at all
}
```

| View | Output coordinates `r` → input coordinates | Valid |
|---|---|---|
| `reshape[s']` | linearize `r` row-major over `s'`, then split by the input shape with `//` and `%` | always |
| `permute[p]` | `in[p[k]] = r[k]` | always |
| `slice[axis a, start s, len n]` | `in[a] = r[a] + s`; other axes unchanged | always (bounds proven by V8) |
| `broadcast[shape, new_axes]` | drops the coordinates of new or expanded axes | always |
| `pad[axis a, before b, after e, value v]` | `in[a] = r[a] - b` | `b <= r[a] < b + n`; else the value `v` |
| `concat[axis a]`, source `i` | `in[a] = r[a] - off_i` | `off_i <= r[a] < off_i + n_i` |

Composition is substitution: `f ∘ g` replaces each `Coord(k)` in `f` by `g.coords[k]`. The simplifier (`index.rs::simplify`) uses the bounds of §4.3 for `//` and `%`: for example `(r₁·128 + r₂) // 128 → r₁` when `0 <= r₂ < 128`. tinygrad's comment on its reshape case says the simplifier does "a lot of heavy lifting" (`schedule/indexing.py:166`, R2 §3.3); the same will be true here, and VII-T4 tests it on every view chain of the Qwen3 and Gemma 3 graphs.

A view chain over a parameter or state buffer whose composed index function is affine (an offset plus one stride per axis, unit stride on the innermost axis) is marked **zero-copy**: it becomes a buffer offset and strides passed to the consuming kernel, never a kernel of its own. If the strides also make it one contiguous byte range, it is tinygrad's `contiguous_view` case (R2 §3.3). The KV read `slice[axis=1, start=0, len=P+T]` of a `[8, C, 128]` cache is affine but not contiguous (head stride `C·128`); attention consumes it in place, with no copy.

## 5. Operations

### 5.1 Conventions

- An operation is written `op[attrs] %src, ... : result-type`, as in the text format (§9).
- In the type rules, `x: dt[s]` means "a source of dtype `dt` and shape `s`". Dims are compared as canonical `Poly`s, so `Hkv·G·Dh` and `8192` are equal when the symbols are bound to 8, 8 and 128. QTypes are compared by `QFormat` only; packings never make two tensors incompatible (QT8, Part XII §2.5).
- **No implicit broadcasting.** Every elementwise operation needs operands of one shape. A smaller operand is made to fit with an explicit `broadcast` view.
- **No implicit promotion.** Elementwise operands have one dtype. Narrowing and widening are explicit `cast`s; the precision plan is therefore visible in the graph (§7.3).
- Floating elementwise and reduction operations **compute in f32** whatever their storage dtype (f64 in the interpreter's F64 mode). The result dtype is the operand dtype for elementwise operations and the accumulator dtype for sums; a narrower result needs a `cast`. §11.2 gives the numerics.
- An operation's effects are none. State is threaded functionally (§6).

### 5.2 Leaves

| Op | Sources | Attrs | Result | Rule |
|---|---|---|---|---|
| `param` | — | `index`, `name`, `role`: `input`, `weight`, `state` or `sym S`; optional `range lo..hi` for integers | the declared type (`i32[]` for `sym S`) | names unique per graph (V4); a `sym` parameter binds the run-time symbol `S` (V5) |
| `const` | — | dtype, shape, data | `dt[s]` | data printed in shortest round-trip decimal, or as hex bits when large or not finite |
| `symval` | — | symbol `S` | `i32[]` | the value of `S`; `S` must be bound in this graph (V5) |
| `iota` | — | shape `s`, axis `a` | `i32[s]` | element `r` is `r[a]` |

### 5.3 Elementwise, casts

| Op | Sources | Result | Notes |
|---|---|---|---|
| `neg abs sqrt rsqrt recip exp exp2 log log2 sin cos tanh sigmoid erf floor ceil round` | `x: f[s]` | `f[s]` | `f` ∈ {f32, f16, bf16}; computed in f32 |
| `add sub mul div max min pow` | `a, b: f[s]` | `f[s]` | `pow` with a constant exponent is canonicalized to multiplications or `sqrt` where exact (Part VIII) |
| `add sub mul max min and or xor shl shr idiv imod` | `a, b: i[s]` | `i[s]` | integers; `idiv`/`imod` round toward −∞; overflow is excluded by V22 |
| `cmp.eq cmp.ne cmp.lt cmp.le cmp.gt cmp.ge` | `a, b: dt[s]` | `bool[s]` | mnemonics as in SIR |
| `not` | `x: bool[s]` | `bool[s]` | |
| `select` | `c: bool[s]`, `a, b: dt[s]` | `dt[s]` | tinygrad's `WHERE` |
| `cast[to]` | `x: dt[s]` | `to[s]` | float→float rounds to nearest even; int→float rounds to nearest even; float→int truncates toward zero and is defined only for in-range values (the interpreter reports an out-of-range value as an error) |
| `bitcast[to]` | `x: dt[s]` | `to[s']` | equal byte size; last axis rescaled when element sizes differ |

### 5.4 Reductions

| Op | Sources | Attrs | Result | Notes |
|---|---|---|---|---|
| `reduce.sum`, `reduce.mean` | `x: dt[s]` | `axes`, `acc` (f32 for floats, i32 for integers) | `acc[s without axes]` | `mean` is `sum` followed by one division by the element count, in `acc` |
| `reduce.max`, `reduce.min` | `x: dt[s]` | `axes` | `dt[s without axes]` | −inf is a valid element |
| `argmax`, `argmin` | `x: dt[s]` | `axis` | `i32[s without axis]` | ties go to the **lowest index** (R8 §3.5 rule 5); result range `0..len` |
| `topk` | `x: dt[s]` | `axis`, `k` | `Many(dt[s with axis→k], i32[s with axis→k])` | values descending; ties by lowest index; read with `proj[0]`, `proj[1]` |

There is no `keepdims`. A reduction removes its axes; `broadcast` puts them back. Reductions over more than one axis are allowed when the axes are adjacent.

### 5.5 Matrix products

| Op | Sources | Attrs | Result | Rules |
|---|---|---|---|---|
| `matmul` | `a: f[B…, M, K]`, `b: f'[B…, K, N]` | `tb` (b is `[B…, N, K]`), `acc=f32`, `out` | `out[B…, M, N]` | batch dims equal (use `broadcast` for GQA-style sharing); `f`, `f'` floats, each widened exactly to `acc` |
| `qmatmul` | `x: f[B…, M, K]`, `w: Quant(q)[N, K]` | `acc=f32`, `out` | `out[B…, M, N]` | `w` is always used transposed, as checkpoints store `[out, in]`; the blocked axis of `q` is `K` and `K` is a multiple of the block (QT3); `w` is a `Weight` value |

`matmul(x, transpose(w))` is canonicalized to `matmul[tb] x, w`, and to `qmatmul` when `w` is quantized ([Part VIII](part08-graph-optimizer.md) `TIR-CANON-004`); the Rust sketch in [Part XXIV](part24-roadmap-now.md) #4 is written that way.

The result dtype is an explicit attribute. Part XII's QT4 writes `qmatmul(x: f16 or bf16, W) -> x's dtype`; in TIR that is `out` equal to `x`'s dtype. The default precision plan of §7.3 uses `out=f32`, which keeps one rounding fewer before the residual add.

### 5.6 Views

| Op | Source | Attrs | Result | Rule (V8) |
|---|---|---|---|---|
| `reshape` | `x: dt[s]` | target dims `s'` | `dt[s']` | `∏s = ∏s'` as polynomials |
| `permute` | `x: dt[s]` | permutation `p` | `dt[s∘p]` | `p` is a permutation of `0..rank` |
| `slice` | `x: dt[s]` | `axis`, `start`, `len` (Dims, may be symbolic) | `s` with `s[axis] = len` | `start + len <= s[axis]` proven, or guarded (§6.3) |
| `broadcast` | `x: dt[s]` | target shape `t`, `map`: output axis of each source axis | `dt[t]` | `map` strictly increasing; `s[i] = t[map[i]]` or `s[i] = 1` |
| `pad` | `x: dt[s]` | `axis`, `before`, `after`, `value` | `s` with `s[axis] + before + after` | `value` has `x`'s dtype |
| `concat` | `x₁ … xₙ: dt[sᵢ]` | `axis` | sum along `axis` | other dims equal; quantized sources only along a non-blocked axis |

Views of a `Weight` are `Weight`s; views of a `State` value are plain values (a read), never states (V11).

### 5.7 Gather

| Op | Sources | Attrs | Result | Rule |
|---|---|---|---|---|
| `gather_rows` | `table: dt[R, D]`, `idx: i32[s]` | `out` | `out[s…, D]` | `int_range(idx) ⊆ [0, R)` proven (V13); a quantized table is dequantized row by row (Part XII's `gather_rows`) |

Embedding lookup is `gather_rows[out=f32] %embed, %tokens`. General gathers along other axes, and the expert-indexed matmul of mixture-of-experts models, are added with the MoE builders (G7, §5.13.8).

### 5.8 State update

| Op | Sources | Attrs | Result | Rules |
|---|---|---|---|---|
| `update` | `st: dt[s]` (a `State` value), `v: dt[s with axis→n]`, `off: i32[]` | `axis` | `dt[s]`, `State` | `off + n <= s[axis]` proven or guarded (V12); `st` has no other `update` consumer and all its other readers precede this one (V11) |

`update` is the KV append: elements `[off, off + n)` along `axis` are replaced by `v`; all others are kept. Part XII's `kv_append` is `update` on a KV state; when the state has a quantized dtype it is legal only under a labeled mode (QT7). A recurrent state (DeltaNet) is not appended to but replaced: a graph may return any value of the state's type as the new version, under the same linearity rules (§6.2).

### 5.9 Quantization operations

| Op | Sources | Attrs | Result | Where legal |
|---|---|---|---|---|
| `dequant` | `w: Quant(q)[s]` | `out` | `out[s]` | anywhere; materializing a dequantized weight is reported by `sky explain` (`TIR-QUANT-001`) because it multiplies the bytes read |
| `relayout` | `w: Quant(q)[s]` or float `Weight` | `recipe` (RL1–RL7, Part XII §5.3) | same `QFormat`, new `QPacking` or layout | weights only; evaluated when `.skyweights` is built, never per step (QT6) |
| `quantize` | `x: f[s]` | `format` | `Quant(q)[s]` | only in `sky quant` pipelines; rejected in inference graphs (QT5) |
| `quantize_dynamic` | `x: f[s]` | `format` | `Quant(q)[s]` | only under a labeled mode recorded in the plan (QT7) |

### 5.10 Counter-based random numbers

| Op | Sources | Attrs | Result | Meaning |
|---|---|---|---|---|
| `rng.philox` | `key: u32[2]`, `c0: i32[s]`, `c1: i32[s]` | `stream` (u32) | `u32[s]` | element `r` is word 0 of Philox4x32-10 with key `key` and counter `(c0[r], c1[r], stream, 0)` |
| `rng.uniform` | `bits: u32[s]` | — | `f32[s]` | `((bits >> 8) + 0.5) · 2⁻²⁴`: exact in f32, strictly inside (0, 1), so `log` never sees 0 or 1 |

Philox4x32-10 is the counter-based generator of Salmon et al. ("Parallel random numbers: as easy as 1, 2, 3", SC 2011) [LIKELY: standard algorithm; cuRAND ships it; the round constants are taken from the paper at implementation, VII-T10]. It is a pure function: no hidden state, so `rng.*` nodes hash-cons and replay like any other node, and a sample depends only on (seed, counters). The sampler keys the counters by (position, token id) (§5.13.6), which is what keyed Gumbel sampling needs for speculation to reproduce plain decoding exactly at any temperature (R8 §3.2; SPINE D10).

### 5.11 Calls and projections

| Op | Sources | Attrs | Result | Rules (V15) |
|---|---|---|---|---|
| `call` | arguments | callee `@g` | `Many(results of @g)` | arity and types match `@g`'s signature; a `weight` parameter takes a `Weight` value, a `state` parameter takes a `State` value and its result version replaces it; the call graph is acyclic |
| `proj` | `x: Many(...)` | index `i` | the `i`-th type | |

A model's step graph calls one layer graph once per layer (64 calls for Qwen3-32B). The layer graph is built, optimized, scheduled and tuned once; the 64 calls differ only in their arguments. This is tinygrad's per-block function (R2 §4) as a typed construct, and it is what lets Part X tune one block for all layers (SPINE D11).

### 5.12 Validation

| Op | Sources | Attrs | Result | Use |
|---|---|---|---|---|
| `check.finite` | `x: f[s]` | `label` | `bool[]` | true iff every element of `x` is finite; inserted only by `--check-finite` at layer boundaries and on logits; its result is an extra graph output, so no pass may remove it |

Inside a graph, −inf and NaN follow IEEE 754 (SPINE D6: "every non-finite value is an error" does not hold inside tensor programs; masks use −inf). Finiteness is checked at the boundary with scientific Sky (§14) and, in validation mode, by `check.finite`.

### 5.13 Composites

A composite is an operation whose meaning is **defined** by a fixed decomposition into primitives, stored in `crates/sky-tir/src/composite/` as a function `decompose(attrs, srcs, &mut GraphBuilder) -> NodeId`. Three things follow:
- the interpreter may evaluate a composite through its decomposition, and a native implementation in the interpreter is tested against the decomposition on random inputs (VII-T8);
- composite recognition ([Part VIII](part08-graph-optimizer.md) §3.7) is exact when the matched primitives are the decomposition, and is a recorded contract rewrite when they differ in rounding;
- a kernel that implements a composite is tested against the decomposition at the tolerance of §11.2.

Canonical operand layouts keep the composites small: token axis first for activations (`[T, H, Dh]`), head axis first for the cache (`[Hkv, S, Dh]`).

| Composite | Sources | Attrs | Result |
|---|---|---|---|
| `rms_norm` | `x: f[…, D]`, `w: f'[D]` | `eps`, `out`, `offset` (0, or 1 for Gemma's `(1+w)`), `round_before_weight` (none or a dtype) | `out[…, D]` |
| `rope` | `x: f32[T, H, Dh]`, `pos: i32[T]` | `style` (`neox` or `normal`), `theta`, `rot` (even, ≤ Dh), `scaling`, `angle` (`f32`) | `f32[T, H, Dh]` |
| `softmax` | `x: f32[…, S]` (+ `sinks: f32[H]`) | `axis`, `sinks` | `f32[…, S]` |
| `attention` | `q: f[T, Hq, Dh]`, `k, v: f'[Hkv, S, Dh]`, `qpos: i32[T]` (+ `parent: i32[T]`, `sinks: f32[Hq]`) | `mask` (`causal`, `sliding(W)`, `tree(prefix)`), `scale`, `softcap` | `f32[T, Hq, Dh]` |
| `swiglu`, `geglu` | `g, u: f32[…, F]` | `out`; `geglu`: `approx` (`tanh` or `erf`); `swiglu`: optional `clamp`, `alpha`, `up_offset` (gpt-oss) | `out[…, F]` |
| `silu`, `gelu` | `x: f32[s]` | `gelu`: `approx` | `f32[s]` |
| `sample` | `logits: f32[V]`, `key: u32[2]`, `pos: i32[]` | `temperature`, `top_k`, `top_p`, `min_p` | `i32[]`, range `0..V` |
| `deltanet.scan` | `q, k: f32[T, Hk, Dk]`, `v: f32[T, Hv, Dv]`, `g, beta: f32[T, Hv]`, `state: f32[Hv, Dk, Dv]` | `form` (`recurrent` or `chunked(c)`), `checkpoints` | `Many(f32[T, Hv, Dv], f32[Hv, Dk, Dv] [, f32[T, Hv, Dk, Dv]])` |
| `conv.causal` | `x: f[T, Ch]`, `w: f'[Ch, K]`, `state: f[Ch, K−1]` | `act` (none or `silu`) | `Many(f32[T, Ch], f[Ch, K−1])` |

#### 5.13.1 `rms_norm`

```text
# rms_norm[eps=e, out=dt, offset=o] %x, %w   over the last axis (size D)
%xf = cast[f32] %x
%sq = mul %xf, %xf
%ms = reduce.mean[axes=last, acc=f32] %sq           : f32[…]
%e  = const f32[] e
%v  = add %ms, broadcast(%e)                         : f32[…]
%r  = rsqrt %v
%y  = mul %xf, broadcast(%r)                         : f32[…, D]
%y' = cast[f32] (cast[rb] %y)                        # only if round_before_weight = rb (the transformers form)
%wf = add (cast[f32] %w), broadcast(const f32[] o)   # offset 0 is folded away
%z  = mul %y', broadcast(%wf)
%o  = cast[dt] %z
```

`round_before_weight` exists because the transformers implementation casts the normalized value to the input dtype before multiplying by the weight [LIKELY: `Qwen3RMSNorm.forward` in transformers; to be confirmed by VII-T8's comparison against transformers]. The builder chooses the form; the two forms are different composites (different attributes), so recognition never silently swaps them.

#### 5.13.2 `rope`

The inverse frequencies are a compile-time table: `inv[i] = θ^(−2i/rot)` for `i in 0..rot/2`, computed in f64 and rounded once to f32, then scaled:

| `scaling` | Table transform | Source |
|---|---|---|
| `none` | — | Qwen3 (θ = 1e6), Mistral (R9 §3) |
| `linear(f)` | `inv[i] / f` | Gemma 3 global layers, ×8 (R9 §3) |
| `llama3(f, low, high, L)` | wavelength `λ = 2π/inv[i]`; keep if `λ < L/high`; `inv[i]/f` if `λ > L/low`; otherwise blend with `s = (L/λ − low)/(high − low)`: `(1 − s)·inv[i]/f + s·inv[i]` | Llama 3.1/3.2 (R9 §3) [LIKELY: transformers `_compute_llama3_parameters`] |
| `yarn(f, L, β_fast, β_slow, attn_factor, truncate)` | correction range from `β_fast`, `β_slow` (floored and ceiled when `truncate`, as llama.cpp and mlx-lm always do, R9 §3.1); a linear ramp between interpolated `inv[i]/f` and extrapolated `inv[i]`; `cos` and `sin` multiplied by `attn_factor` (default `0.1·ln f + 1`) | Qwen3 at 128K (opt-in), gpt-oss (`truncate=false`) [LIKELY: transformers `_compute_yarn_parameters`] |
| `longrope(short, long, L)` | per-dimension factors, chosen by context length | Phi-4-mini (tier 1); runtimes disagree on the choice (R9 §3.1), so the rule is an explicit attribute |

Then, per token `t` and frequency `i`: `a = f32(pos[t]) · inv[i]` (one f32 multiply), `c = cos(a)`, `s = sin(a)`, and

```text
neox   (rotate-half):   x₁ = x[…, 0:rot/2],  x₂ = x[…, rot/2:rot]
                        y₁ = x₁·c − x₂·s,    y₂ = x₂·c + x₁·s
normal (interleaved):   pairs (x[2i], x[2i+1]) rotated by the same formulas
partial (rot < Dh):     y[…, rot:Dh] = x[…, rot:Dh]
```

The angle is computed in f32 as transformers does, so at position 131071 it carries the f32 spacing of that range, 2⁻⁷ rad (2⁻⁸ at 40959). `angle=f64` is an attribute value reserved for a long-context accuracy experiment [EXPERIMENTAL: VII-M5 measures whether it changes logits at 128K]. GGUF files for Llama and Mistral have Q and K permuted at conversion and use `normal` (R9 §3.1); the loader sets `style` per file, never per model family.

#### 5.13.3 `softmax`

```text
%m = reduce.max[axes=a] %x                       # with sinks: max(%m, broadcast(%sinks))
%e = exp (sub %x, broadcast(%m))
%s = reduce.sum[axes=a, acc=f32] %e              # with sinks: add exp(sinks − m)
%y = div %e, broadcast(%s)
```

#### 5.13.4 `attention`

Grouped-query attention with `G = Hq / Hkv` query heads per KV head. The verifier requires `divides(Hkv, Hq)` (V14).

```text
%q4 = reshape[T, Hkv, G, Dh] %q ; %qg = permute[1, 2, 0, 3] %q4 ; %q3 = reshape[Hkv, G·T, Dh] %qg
%s  = matmul[tb, acc=f32, out=f32] %q3, %k        : f32[Hkv, G·T, S]
%s  = mul %s, broadcast(const scale)              # softcap c: mul(c, tanh(div(%s, c)))
%vis = mask(qpos, S, prefix, parent)              : bool[T, S], broadcast to [Hkv, G, T, S]
%s  = select %vis, %s, broadcast(const −inf)
%p  = softmax[axis=2] %s                          # sinks: per query head, softmax[sinks]
%o  = matmul[acc=f32, out=f32] %p, %v             : f32[Hkv, G·T, Dh]
%out = reshape → permute → reshape                : f32[T, Hq, Dh]
```

| `mask` | Key slot `j` is visible to query `t` iff | Used for |
|---|---|---|
| `causal` | `j <= qpos[t]` | decode, prefill, chain verification |
| `sliding(W)` | `qpos[t] − W < j <= qpos[t]` | Gemma 3 local layers, gpt-oss [LIKELY: matches the transformers sliding mask; the boundary is checked per model by the G7 correctness suite] |
| `tree(prefix)` | `j < prefix`, or `j − prefix` is `t` or an ancestor of `t` in `parent` | draft-tree verification (SPINE D10); the traversal order is unchanged, only the mask differs (R8 §3.5 rule 2) |

Every mode keeps at least the query's own slot visible, so no row is entirely −inf and the softmax denominator is never zero (V14). The decomposition materializes `[Hkv, G·T, S]` scores; kernels never do (flash-decoding and flash prefill, [Part XV](part15-attention.md)), and the difference in summation order is within the `inference` contract (§11.2).

#### 5.13.5 Gated linear units

`silu(x) = x · sigmoid(x)`; `gelu[tanh](x) = 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`; `gelu[erf](x) = 0.5·x·(1 + erf(x/√2))`. `swiglu(g, u) = silu(g)·u`, `geglu(g, u) = gelu(g)·u`. gpt-oss's clamped form is `swiglu[clamp=7, alpha=1.702, up_offset=1]`: `(clip(u, ±7) + 1) · g′ · sigmoid(1.702·g′)` with `g′ = min(g, 7)` (R9 §3).

#### 5.13.6 `sample`

```text
1. temperature = 0:  return argmax(logits)                     (ties: lowest token id)
2. z = logits / temperature                                    (f32)
3. top_k:   (vals, ids) = topk[k](z)                           (descending; ties by lowest id)
4. top_p:   keep the shortest prefix of vals whose softmax mass reaches top_p (at least one)
   min_p:   drop entries with p < min_p · p_max
5. noise:   g = −log(−log(rng.uniform(rng.philox[stream=1](key, pos, ids))))
6. return   ids[argmax(vals + g)]  over the kept entries
```

Step 6 is the Gumbel-max trick: the argmax of logits plus independent Gumbel noise is distributed as softmax of the logits over the kept set. The noise for token `x` at position `t` depends only on `(key, t, x)`, so speculative decoding that accepts a drafted token iff it equals this sample emits exactly the plain-decoding sequence, provided the logits are bit-identical (R8 §3.2, §3.3; SPINE D10). Standard rejection sampling (accept with `min(1, p/q)`) is a separate `sample.verify` composite owned by [Part XXI](part21-speculative.md).

#### 5.13.7 Gated DeltaNet and the causal convolution

Qwen3.5/3.6/3.8-27B interleave 48 Gated DeltaNet layers with 16 attention layers, with 16 QK heads and 48 V heads of 128, a convolution of width 4, and an fp32 recurrent state (R8 §1.7). The recurrent form, per position `t` and V head `h` (K head `hk = h / (Hv/Hk)`):

```text
S  ← exp(g[t,h]) · S                         S: f32[Dk, Dv] for head h
u  ← beta[t,h] · (v[t,h] − Sᵀ k[t,hk])        f32[Dv]
S  ← S + k[t,hk] ⊗ u
o[t,h] ← Sᵀ q[t,hk]                           q, k L2-normalized and q scaled by Dk^(−1/2) upstream
```

[LIKELY: this is the recurrent reference of flash-linear-attention's gated delta rule, which MLX's `gated_delta_update` kernel also implements; our research recorded that kernel's shapes and chunk sizes but not its source (R3-kernels §gated_delta_update). VII-T9 checks the decomposition against transformers' `qwen3_next` on random inputs before any kernel work.]

- `form=recurrent` is the definition. `form=chunked(c)` computes the same mathematics with a chunked parallel scan in a different floating-point order. MLX uses chunks of 16 (with neural accelerators) or 8 for T > 8, and the sequential form for T ≤ 8 (R3-kernels §gated_delta_update).
- `checkpoints` makes the scan also return the state after every position, for exact rollback (§6.4).
- `conv.causal[act]` is `y[t, c] = act(Σⱼ w[c, j] · xx[t + j, c])`, with `xx` the concatenation of the stored `K − 1` previous inputs and `x` along the token axis; its new state is the last `K − 1` rows of `xx`.

The state is `48 × 48 × 128 × 128 × 4 B = 151 MB` for the whole model, read and written every token (R8 §1.7).

#### 5.13.8 Deferred: mixture-of-experts

Qwen3-MoE and gpt-oss (G7) need a router composite (`moe.route[score=softmax|sigmoid, top_k, renorm]`, with the per-family order of softmax and top-k from R9 §3) and an expert-indexed product (`qmatmul.gather`: weights `Quant(q)[E, N, K]` and expert ids). They are specified with the G7 builders; nothing in TIR's data model has to change for them.

## 6. Effects and state

### 6.1 Graphs are pure functions

A TIR graph is a pure function from its parameters to its results. Nothing in a graph prints, fails, calls the host or keeps hidden state:
- random numbers are keyed (§5.10);
- run-time size conditions are guards evaluated before the graph runs (§6.3);
- the KV cache and recurrent state are **values**: a state parameter passes version *n* in, and the graph returns version *n + 1*.

So the effect system of SIR (`io`, `fail`, `call`; `docs/design/ir.md` "Effects") has nothing to describe inside TIR, and every node may be reordered, duplicated (recomputed), merged or removed, subject only to the state rules below. The single place where TIR meets effects is `tensor.call` in SIR (§14).

### 6.2 Linearity, and how state is lowered in place

For every state value `s` (a `state` parameter, an `update` result, or a value returned as a state's next version):
1. At most one consumer produces the next version: an `update`, or the node returned in that state's result position.
2. Every other consumer of `s` is a read, and no read of `s` may depend on the next version. Then a schedule exists in which all reads of version *n* happen before version *n + 1* is written.
3. The final version is returned at the result position of its state parameter, with the same type and layout.

These are rules V11 and V12. They make the in-place lowering always legal:
- Part IX orders every read of version *n* before the write of version *n + 1* (a write-after-read edge);
- Part XIX gives both versions one buffer;
- an `update` writes only its slab `[off, off + n)`.

This is SIR's buffer-reuse argument (`docs/design/ir.md`, "Lifetimes and buffer reuse": an operand that dies at an operation can hold its result), applied to values that are too big to copy: Qwen3-32B's fp16 KV cache is 10.74 GB at 40,960 positions (R8 §1.5). A graph that reads an old version after the update is rejected. In Sky source that takes an alias of the old value, since a write rebinds the name (§13.6):

```text
cache.sky:9:9: error: the old value of `kv.k` is read after `kv.k` is updated
  |
7 |     before = kv.k
  |              ---- the old value is kept here
8 |     kv.k[:, pos : pos + T] = k
  |     ---------------------- and the cache is updated in place here
9 |     d = before[:, : pos] - kv.k[:, : pos]
  |         ^^^^^^ so this read would see the new values
  = note: keeping the old value would need a copy of the cache: 83.9 MB per layer for Qwen3-32B's K
  = help: compute what you need from `before` before the update
```

(83.9 MB is the K half of one layer's cache: 8 heads × 40,960 positions × 128 × 2 B.) A recurrent state that is *replaced* (DeltaNet) is lowered in place only if the replacing kernel reads each tile before writing it. Otherwise Part IX uses two buffers and alternates them, which doubles the state's memory (302 MB instead of 151 MB for Qwen3.6-27B).

In Qwen3 the cache is written before attention reads it, and attention reads the new version (§9.2), as batch-invariant attention requires: one cache layout and the update before the attention kernel, for prefill and decode alike (R8 §3.3).

### 6.3 Guards

A constraint the verifier cannot prove from the symbol bounds (V17) becomes a guard of the plan. The runtime evaluates guards on the CPU from the step's run-time symbol values before it submits the step. Guards never run inside a kernel, so a graph never fails halfway through. A failing guard is a run-time error at the constraint's origin:

```text
models/qwen3.sky:69:5: error: the context is full
   |
69 |     requires pos + T <= C
   |     ^^^^^^^^^^^^^^^^^^^^^ 40704 + 512 is more than 40960
   = note: this model was compiled for at most 40960 positions (C)
   = help: compile with a larger context, `sky compile --context 65536`, or start a new conversation
```

### 6.4 Rollback for speculative decoding

Exact speculation (SPINE D10) needs every state to be restored to the accepted prefix:
- **KV cache.** Rollback is truncation. Rejected draft tokens leave values in slots at and beyond the new `P`, and attention never reads past `P + T`, so those slots are dead until overwritten. Draft-tree verification additionally compacts the accepted path into consecutive slots; that is a small graph (`gather_rows` along the slot axis, then `update`) owned by [Part XXI](part21-speculative.md).
- **Recurrent state (DeltaNet).** The state cannot be truncated. Two exact options, both with `form=recurrent` so verification reproduces plain decoding bit for bit (R8 §3.5 rule 4):

  | Option | How | Extra state traffic per verify step (Qwen3.6-27B, T = 5) |
  |---|---|---|
  | (a) checkpoints | `deltanet.scan[checkpoints]` writes the state after every position; the runtime keeps the accepted one | about 5 × 151 MB = 755 MB written, 5.2% of the 14.4 GB step |
  | (b) replay | verification does not write the state; it saves each position's small inputs (`q, k, v, g, beta`); after acceptance, a replay graph applies the accepted positions to the stored state | about 151 MB read + 151 MB written = 302 MB, 2.1% of the step |

  The figures are arithmetic from R8 §1.7 (151 MB of state, 14.4 GB streamed per token). [EXPERIMENTAL: VII-M6 measures both; (b) is preferred if its replay kernel stays at the bandwidth roofline.]

## 7. Static, symbolic and bucketed sizes

### 7.1 What is known when

| Quantity (Qwen3-32B) | Kind | Value or range | Where it comes from |
|---|---|---|---|
| hidden 5120, layers 64, heads 64 / 8, head_dim 128, MLP 25600, vocabulary 151,936 | static | constants | `config.json` (R8 §1.1) |
| weight formats and layouts | static per artifact | `mlx.affine(b4,g64,bf16)` | file header (Part XII) |
| `C`, maximum context | static per artifact | 40,960 by default (`max_position_embeddings`); chosen at compile time | `sky compile --context` |
| `T`, tokens per step | plan | decode `1..=1`; verify widths; prefill buckets such as `33..=128` | Parts XVI, XVII |
| `P`, position of the first token | run-time | `0..C` | the runtime, every step |
| `S = P + T`, keys visible | derived run-time | `1..=C` | — |
| context bucket | plan sub-key | ranges of `S`, for example `S <= 4096` | selects attention splits ([Part XV](part15-attention.md)); the graph is the same |
| batch | static | 1 in year 1 | SPINE §1 |
| device | static per artifact | the `sky-hw` fingerprint | Part X |

Only `P` (and through it `S`) is unknown when kernels are generated. tinygrad has the same structure, with `start_pos` and the chunk length as bounded variables patched into the argument buffer at replay (R2 §7.5, §8.1); TIR states it in the types.

### 7.2 Plans and buckets

```rust
// crates/sky-tir/src/plan/mod.rs
pub struct PlanSpec {
    pub name: Arc<str>,                         // "decode", "verify5", "prefill128"
    pub binds: BTreeMap<SymId, (u64, u64)>,     // T ↦ 1..=1, or T ↦ 33..=128
    pub context: Option<(u64, u64)>,            // optional S bucket
    pub outputs: OutputSet,                     // which results this plan needs (logits of the last row only, ...)
}
```

Part VIII's `specialize` pass derives one L1 graph per plan from the generic graph. A plan that binds `T` to a range is compiled for the range's maximum, with `T` as a run-time value checked against the range by a guard. The runtime picks the plan for each step from `(T, S)`. Nothing is compiled at run time: plans are built ahead of time (`sky compile`) or at first use and cached, keyed by `Graph::structural_hash` and the `PlanSpec`.

### 7.3 Invariance across plans

SPINE D6 requires batch invariance (a sequence's logits do not depend on the verify width) and chunk invariance (prefill results do not depend on the chunk size). TIR contributes two things.
- **The precision plan is part of L1.** Every narrowing is an explicit `cast` in the generic graph, so all plans round at the same points. The default plan of the Qwen3 builder: residual stream in f32; matmul inputs in the checkpoint dtype (bf16); matmul accumulation and outputs in f32; KV in the configured KV dtype; queries cast to the KV dtype before attention, because the M5 neural-accelerator prefill path takes f16 or bf16 operands only (SPINE §2); logits in f32 (R8 §3.4 note 1: f32 logits make exact top ties rare).
- **Reduction schedules are L2 attributes that must agree.** At L2 every reduction (matmul K-sums, norms, softmax and attention over keys, `deltanet.scan`) carries a `ReduceSchedule`: a canonical summation order for its reduction length. It may depend on the reduction length and the kernel configuration, never on `T`, on the plan, or on how other axes are tiled. Rule V18 requires the schedules of a node to be equal in every plan derived from one generic graph. This is R8 §3.5 rule 1 made checkable; attention's version is a fixed split *size* over keys (R8 §3.3).

Part IX and the kernels must honour the schedule. TIR can only state and check the requirement.

## 8. Verifier rules

`sky_tir::verify::verify(&Module) -> Result<(), Vec<VerifyError>>` is total and never panics, like SIR's (`crates/sky-ir/src/verify.rs:1-60`). The pass manager runs it before the first pass and after every pass in tests and debug builds. Each rule has a test in `crates/sky-tir/tests/verify.rs` named after it (`v07_elementwise_shapes_differ`, ...).

| Rule | Checks |
|---|---|
| V1 | Arena integrity: ids in range; `srcs[i] < id` (acyclic); params and results name live nodes |
| V2 | Interning: no two live nodes with equal `(op, srcs, attrs)` |
| V3 | Derived types: each cached type equals `infer(op, srcs, attrs)` |
| V4 | Parameters: unique names; valid roles; a `weight` is never the state of an `update` |
| V5 | Symbols: every symbol in a type or attribute is declared; run-time symbols are bound by a `sym` parameter or passed through a call |
| V6 | Dimensions: every dim is provably ≥ 1 |
| V7 | Elementwise: equal shapes and dtypes; integer-only and float-only operations respect their domains |
| V8 | Views: reshape keeps the element count; permute is a permutation; slice and pad bounds proven or guarded; broadcast map valid |
| V9 | Matrix products: contraction dims equal; batch dims equal; float operands; `acc` valid |
| V10 | Quantized types: QT1–QT8 of Part XII §2.5 (`verify/quant.rs`) |
| V11 | State linearity: §6.2 rules 1–3 |
| V12 | `update`: offset plus length within the axis, proven or guarded |
| V13 | `gather_rows`: index range within the table, proven from `int_range` |
| V14 | Composites: valid attributes (`rot` even and ≤ Dh; `Hkv` divides `Hq`; masks keep the own slot visible; `eps` > 0) |
| V15 | Calls: callee exists; arity, types and roles match; the call graph is acyclic |
| V16 | Results: types match the signature; each state parameter's final version is returned in its position |
| V17 | Constraints: each `require` is proven or recorded as a guard of every plan that uses it |
| V18 | Levels and plans: no strategy attributes at L1; at L2 every matmul, attention and reduction has a strategy, and its `ReduceSchedule` is equal across the plans of one generic graph |
| V19 | Layouts: non-`Logical` layouts only on parameters and state at L1; `Packed` only through a recipe of Part XII §5.3 |
| V20 | Labeled modes: `quantize_dynamic`, quantized-state `update` and contract rewrites appear only when the plan's `ContractLog` names their mode |
| V21 | Provenance: every node has at least one origin |
| V22 | Index width: every tensor's element count fits in `i32` (< 2³¹); a byte offset that can reach 2³² is marked 64-bit |
| V23 | Constants: no NaN in `const` data except where the op allows it; −inf allowed |
| V24 | Validation: `check.finite` results are graph outputs |

V22 matters for real shapes: the Qwen3 embedding has 151,936 × 5120 = 777,912,320 elements, under 2³¹; f32 logits for an 8192-token prefill would be 1,244,659,712 elements, under 2³¹, but 4.98 GB, over 2³² bytes, so their byte offsets must be 64-bit. Part VIII's dead-output pass removes those logits anyway.

## 9. Text format

### 9.1 Grammar

Data only, round-trip exact (`parse(print(m)) == m` after canonical renumbering), parsed by a hand-written parser that never panics, with a fuzz target, as SIR's (`docs/design/ir.md`, "Text format").

```text
module    := "tir" "version" INT NL { decl NL }
decl      := sym | require | graph
sym       := "sym" NAME ( "static" "=" INT | "plan" range | "runtime" range ) [ "multiple_of" INT ]
range     := INT ".." INT | INT "..=" INT                         (exclusive, inclusive)
require   := "require" poly ( "<=" | "==" ) poly
graph     := "graph" "@" NAME "(" [ param { "," param } ] ")" "->" "(" types ")" "{" { stmt } "}"
param     := "%" NAME ":" [ role ] type [ "range" range ] [ STRING ]         (STRING: the module path)
role      := "input" | "weight" | "state" | "sym" NAME
stmt      := "%" NAME "=" OP [ "[" attrs "]" ] [ operands ] ":" rtype [ "@" origin ] NL
           | "return" operands NL
rtype     := [ "state" ] type | "(" types ")"
type      := dtype "[" [ dim { "," dim } ] "]" [ packing ] [ layout ]
dtype     := "f32" | "f16" | "bf16" | "f64" | "i32" | "u32" | "i16" | "u16" | "i8" | "u8" | "bool"
           | QNAME                                     (Part XII canonical names: mlx.affine(b4,g64,bf16), gguf.q4_k, ...)
dim       := INT | poly                                (polynomials over symbol names: P+T, Hkv*G*Dh)
origin    := INT ".." INT | STRING                      (a Sky source span, or a builder path)
```

Comments start with `#`. Floats print in shortest round-trip form, or as `0x` and the bits when not finite.

### 9.2 A Qwen3-32B decoder layer

As emitted by the Qwen3 builder (§12) at L1, before Part VIII runs, with the default precision plan of §7.3 and an fp16 KV cache:

```text
tir version 1

sym C static = 40960                 # maximum context of this artifact
sym T plan 1..=8192                  # tokens processed by one step; bound per plan
sym P runtime 0..40960               # position of the step's first token
require P + T <= C

graph @qwen3_layer(
    %h:     f32[T, 5120],
    %pos:   sym P,
    %qpos:  i32[T] range 0..40960,
    %k_in:  state f16[8, C, 128]                         "kv.k",
    %v_in:  state f16[8, C, 128]                         "kv.v",
    %ln1:   weight bf16[5120]                            "w.input_layernorm",
    %wq:    weight mlx.affine(b4,g64,bf16)[8192, 5120]   "w.self_attn.q_proj",
    %wk:    weight mlx.affine(b4,g64,bf16)[1024, 5120]   "w.self_attn.k_proj",
    %wv:    weight mlx.affine(b4,g64,bf16)[1024, 5120]   "w.self_attn.v_proj",
    %qn_w:  weight bf16[128]                             "w.self_attn.q_norm",
    %kn_w:  weight bf16[128]                             "w.self_attn.k_norm",
    %wo:    weight mlx.affine(b4,g64,bf16)[5120, 8192]   "w.self_attn.o_proj",
    %ln2:   weight bf16[5120]                            "w.post_attention_layernorm",
    %wg:    weight mlx.affine(b4,g64,bf16)[25600, 5120]  "w.mlp.gate_proj",
    %wu:    weight mlx.affine(b4,g64,bf16)[25600, 5120]  "w.mlp.up_proj",
    %wd:    weight mlx.affine(b4,g64,bf16)[5120, 25600]  "w.mlp.down_proj"
) -> (f32[T, 5120], state f16[8, C, 128], state f16[8, C, 128]) {
  # attention
  %x     = rms_norm[eps=1e-6, out=bf16] %h, %ln1               : bf16[T, 5120]   @ "input_layernorm"
  %q     = qmatmul[out=f32] %x, %wq                            : f32[T, 8192]    @ "self_attn.q_proj"
  %k     = qmatmul[out=f32] %x, %wk                            : f32[T, 1024]    @ "self_attn.k_proj"
  %v     = qmatmul[out=f32] %x, %wv                            : f32[T, 1024]    @ "self_attn.v_proj"
  %q3    = reshape[T, 64, 128] %q                              : f32[T, 64, 128] @ "self_attn"
  %k3    = reshape[T, 8, 128] %k                               : f32[T, 8, 128]  @ "self_attn"
  %v3    = reshape[T, 8, 128] %v                               : f32[T, 8, 128]  @ "self_attn"
  %qn    = rms_norm[eps=1e-6, out=f32] %q3, %qn_w              : f32[T, 64, 128] @ "self_attn.q_norm"
  %kn    = rms_norm[eps=1e-6, out=f32] %k3, %kn_w              : f32[T, 8, 128]  @ "self_attn.k_norm"
  %qr    = rope[neox, theta=1000000, rot=128] %qn, %qpos       : f32[T, 64, 128] @ "rotary_emb"
  %kr    = rope[neox, theta=1000000, rot=128] %kn, %qpos       : f32[T, 8, 128]  @ "rotary_emb"
  %kt    = permute[1, 0, 2] %kr                                : f32[8, T, 128]  @ "self_attn"
  %vt    = permute[1, 0, 2] %v3                                : f32[8, T, 128]  @ "self_attn"
  %kh    = cast[f16] %kt                                       : f16[8, T, 128]  @ "kv.k"
  %vh    = cast[f16] %vt                                       : f16[8, T, 128]  @ "kv.v"
  %p     = symval P                                            : i32[]           @ "kv"
  %k_out = update[axis=1] %k_in, %kh, %p                       : state f16[8, C, 128] @ "kv.k"
  %v_out = update[axis=1] %v_in, %vh, %p                       : state f16[8, C, 128] @ "kv.v"
  %ks    = slice[axis=1, start=0, len=P+T] %k_out              : f16[8, P+T, 128] @ "kv.k"
  %vs    = slice[axis=1, start=0, len=P+T] %v_out              : f16[8, P+T, 128] @ "kv.v"
  %qh    = cast[f16] %qr                                       : f16[T, 64, 128] @ "self_attn"
  %a     = attention[causal, scale=0.088388346] %qh, %ks, %vs, %qpos : f32[T, 64, 128] @ "self_attn"
  %a2    = reshape[T, 8192] %a                                 : f32[T, 8192]    @ "self_attn"
  %ab    = cast[bf16] %a2                                      : bf16[T, 8192]   @ "self_attn"
  %o     = qmatmul[out=f32] %ab, %wo                           : f32[T, 5120]    @ "self_attn.o_proj"
  %h1    = add %h, %o                                          : f32[T, 5120]    @ "residual"
  # MLP
  %x2    = rms_norm[eps=1e-6, out=bf16] %h1, %ln2              : bf16[T, 5120]   @ "post_attention_layernorm"
  %g     = qmatmul[out=f32] %x2, %wg                           : f32[T, 25600]   @ "mlp.gate_proj"
  %u     = qmatmul[out=f32] %x2, %wu                           : f32[T, 25600]   @ "mlp.up_proj"
  %m     = swiglu[out=bf16] %g, %u                             : bf16[T, 25600]  @ "mlp.act_fn"
  %d     = qmatmul[out=f32] %m, %wd                            : f32[T, 5120]    @ "mlp.down_proj"
  %h2    = add %h1, %d                                         : f32[T, 5120]    @ "residual"
  return %h2, %k_out, %v_out
}
```

Notes on the listing:
- 16 parameters and 32 operation nodes; 7 of the nodes are `qmatmul`s, which read 274,268,160 B of quantized weights per layer (arithmetic from R8 §1.2's shapes at 4.5 bits per weight: 29.5 MB for Q, K and V, 23.6 MB for O, 147.5 MB for gate and up, 73.7 MB for down). Over 64 layers plus the 437.6 MB output head that is the 17.99 GB per token of SPINE §2.
- `0.088388346` is 1/√128 as the shortest decimal that round-trips to the same f32.
- `qpos` is a parameter, not `P + iota(T)`, so the same layer serves tree verification, where positions are not consecutive (§5.13.4). The step graph computes it once for all 64 layers.
- The cache is written before attention reads it, and attention reads the new version (§6.2).

### 9.3 The logits and step graphs (abbreviated)

The builder emits two top-level graphs. `@qwen3_logits` is the forward pass; `@qwen3_step` adds sampling. The D7 comparison (§12) is on `@qwen3_logits`, which is what the Sky source of §13.9 defines.

```text
graph @qwen3_logits(
    %tokens: input i32[T] range 0..151936,
    %pos:    sym P,
    %emb:    weight mlx.affine(b4,g64,bf16)[151936, 5120]  "weights.model.embed_tokens",
    # ... 64 × 11 layer weights, named "weights.model.layers.{i}.<path>" ...
    %normw:  weight bf16[5120]                             "weights.model.norm",
    %head:   weight mlx.affine(b4,g64,bf16)[151936, 5120]  "weights.lm_head",
    %k0:     state f16[8, C, 128]                          "cache.0.k",
    %v0:     state f16[8, C, 128]                          "cache.0.v",
    # ... "cache.63.k", "cache.63.v" ...
) -> (f32[151936], state f16[8, C, 128], ...) {
  %t     = iota[shape=[T], axis=0]                             : i32[T]
  %p     = symval P                                            : i32[]
  %pb    = broadcast[shape=[T], map=[]] %p                     : i32[T]
  %qpos  = add %pb, %t                                         : i32[T]
  %x0    = gather_rows[out=f32] %emb, %tokens                  : f32[T, 5120]     @ "model.embed_tokens"
  %l0    = call @qwen3_layer(%x0, %pos, %qpos, %k0, %v0, %ln1_0, %wq_0, ...) : (f32[T, 5120], state f16[8, C, 128], state f16[8, C, 128]) @ "model.layers.0"
  %x1    = proj[0] %l0                                         : f32[T, 5120]
  # ... 63 more calls ...
  %last  = slice[axis=0, start=T-1, len=1] %x64                : f32[1, 5120]
  %xn    = rms_norm[eps=1e-6, out=bf16] %last, %normw          : bf16[1, 5120]    @ "model.norm"
  %lg    = qmatmul[out=f32] %xn, %head                         : f32[1, 151936]   @ "lm_head"
  %lg1   = reshape[151936] %lg                                 : f32[151936]
  return %lg1, (proj[1] %l0), (proj[2] %l0), ...
}

graph @qwen3_step(%tokens: input i32[T] range 0..151936, %pos: sym P, %key: input u32[2], ...) -> (i32[], state ...) {
  %r     = call @qwen3_logits(%tokens, %pos, ...)              : (f32[151936], state f16[8, C, 128], ...)
  %lg    = proj[0] %r                                          : f32[151936]
  %p     = symval P                                            : i32[]
  %tv    = symval T                                            : i32[]
  %lp    = add %p, %tv                                         : i32[]            # position of the sampled token, P + T
  %tok   = sample[temperature=0.6, top_k=20, top_p=0.95] %lg, %key, %lp : i32[] @ "sampler"
  return %tok, (proj[1] %r), ...
}
```

- The sampling parameters shown are the Qwen3 model card's thinking-mode values (R8 §1.1).
- The noise is keyed by the position of the token being sampled, `P + T`. A verify step keys row *i* by `P + i + 1`, the position that row predicts, so plain and speculative decoding draw the same noise for the same position.
- The builder slices the last row before the final norm. The Sky source computes logits for every row and selects the last one, and Part VIII's dead-output pass (`TIR-DEAD-002`, which moves the row selection before the norm and the product) turns that into the same graph.

## 10. Spans and provenance

```rust
// crates/sky-tir/src/graph.rs
pub struct Provenance {
    pub origins: SmallVec<[Origin; 1]>,     // every place this node came from
    pub derived: Option<ExplanationId>,     // set when a rewrite created the node
}
pub enum Origin {
    Source { file: SourceId, span: Span },                          // Sky source, from lowering
    Builder { path: Arc<str>, site: &'static Location<'static> },   // Rust builder: model path + call site
}
```

- **Merging.** When interning finds an existing node, the new origin is appended. A rewrite gives its result the union of the origins of the nodes it replaced and a link to its explanation. This is SIR's rule that "a transform that merges operations joins their spans" (`docs/design/ir.md`, "L1 data model"), with a set instead of a joined range, because TIR nodes can come from distant places.
- **Paths.** The builder sets a path prefix with `g.at("self_attn.q_proj")` and records the Rust call site with `#[track_caller]`. A call node prefixes its callee's paths (`model.layers.17`), so `sky explain` names `model.layers.17.self_attn.q_proj` or, for all layers at once, `model.layers.*.self_attn.q_proj`.
- **Parameter names.** A weight or state parameter carries its module path (the `STRING` in §9.1), which is its source path: the parameter's or input's name followed by field names and vector indices (`w.self_attn.q_proj`, `kv.k`, `weights.model.embed_tokens`, `cache.17.k`). A call substitutes the argument's path for the callee's parameter name, so `w.self_attn.q_proj` in the layer graph is `weights.model.layers.17.self_attn.q_proj` at the 18th call. The loader maps a weight path to checkpoint tensors by dropping the root name and appending `.weight`, plus `.scales` and `.biases` for an MLX quantized tensor ([Part XIII](part13-llm-runtime.md) §6). The builder uses the same names as the Sky source of §13.9, so the relayout plan, `sky explain` and the D7 comparison all see one name per tensor.
- **Kernels.** Part XI names generated kernels after the provenance of their anchor node (`qwen3_layer.self_attn.qkv_proj.qgemv`), so GPU profiles and `sky explain` speak in model terms, not node ids.
- **Printing.** The text format prints the first origin (`@ 120..160` or `@ "self_attn.q_proj"`); `--provenance=all` prints all of them. **Canonical text**, used for hashing and for the D7 comparison (§12), prints no origins and numbers values by position, as SIR's canonical numbering does (`crates/sky-ir/src/canon.rs`); it keeps parameter names.

## 11. Reference interpreter and numerics

### 11.1 The interpreter

```rust
// crates/sky-tir/src/interp/mod.rs
pub enum Mode { F32, F64 }
pub struct Interp<'m> { pub module: &'m Module, pub mode: Mode, pub threads: usize }

impl Interp<'_> {
    /// Runs `graph` on bound parameters. Deterministic for a given mode,
    /// whatever `threads` is.
    pub fn run(&self, graph: &str, args: &Bindings) -> Result<Outputs, InterpError>;
}
```

| Aspect | `Mode::F32` | `Mode::F64` |
|---|---|---|
| Floating values | every op computes in f32, rounds to nearest even, no FMA contraction | every value that is f32 in the graph is computed in f64 |
| Narrowing casts (to f16, bf16) | applied | applied: they are part of the program's meaning |
| Reductions | sequential in index order from +0.0 (the order of Sky's reproducible matmul) | the same order, in f64 |
| Transcendentals | the pure-Rust `libm` crate (a port of musl's libm), so the oracle gives the same bits on every machine; `sky-ode` already depends on it (R1 §2.1) | `libm`'s f64 functions |
| Dequantization | exact in f64, rounded once to f32 (Part XII §2.4) | exact in f64 |
| Composites | their decomposition, or a native implementation tested equal to it | the same |
| Parallelism | output rows split across threads; each element computed by one thread in a fixed order | the same |

- **Memory.** Values are freed after their last use. A quantized weight is dequantized one row at a time inside `qmatmul` and `gather_rows`, never materialized. Qwen3-0.6B and 8B (4.6 GB at 4 bits, R9 §1.2) run whole on the 16 GB development Mac; Qwen3-32B runs one layer, or one kernel, at a time with real weights (VII-T8).
- **Speed.** It is an oracle, not an engine. [Part XXIV](part24-roadmap-now.md) #6 expects seconds per token on Qwen3-0.6B; that is acceptable.
- **Errors.** The interpreter reports what the verifier cannot exclude statically: an out-of-range float→int cast, a guard violated by bound values, an index outside a table.

### 11.2 Numerics of each operation under `inference`

Kernels may differ from `Mode::F32` only as this table allows. A difference outside it is a kernel bug. `u = 2⁻²⁴` is the f32 unit roundoff.

| Operation class | Kernels may | Kernels must | Tolerance against the reference |
|---|---|---|---|
| Views, `concat`, `slice`, `gather_rows` on float tables, `update`, `select`, comparisons, integer ops, `iota`, `symval`, `cast` (any), `bitcast`, `argmax`, `topk`, `rng.*` | — | produce identical bits given identical inputs; ties by lowest index | exact, bit for bit (class E) |
| Dequantization (inside `qmatmul`, `gather_rows`, `dequant`) | pre-scale activations (Part XII §6.2), which changes the rounding of the products | decode every element to the reference value (Part XII §5.5, GPU leg) | codec exact; products in class S |
| `add`, `sub`, `mul`, `div`, `neg`, `max`, `min`, `abs` | contract `a·b + c` into one FMA | round to nearest even | ≤ `u·|a·b|` per contraction, against `Mode::F32` (class R) |
| `exp`, `log`, `sin`, `cos`, `tanh`, `sigmoid`, `erf`, `rsqrt`, `sqrt`, `pow` | use the backend's precise or fast function; the choice is recorded per kernel | use the same function in every plan | the function's error bound as recorded by Part XI, against `Mode::F64` (class T) |
| Sums: `reduce.sum`, `reduce.mean`, matmul K-sums, softmax denominators, attention over keys, norms | any fixed order chosen by the kernel configuration (`ReduceSchedule`); split-K with a fixed split size; online softmax | accumulate in f32; use the same order in every plan (V18); no atomics on floats | `|k − r₆₄| ≤ γ_K · Σ|tᵢ|`, `γ_K = K·u / (1 − K·u)`, against `Mode::F64` (class S) |
| −inf, NaN | use −inf for masks | follow IEEE 754; keep at least one visible key per query (V14) | non-finite results must match in position and kind |

The class S bound is the standard a priori bound for floating-point sums and dot products of length K [VERIFIED: the standard bound; Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., chapters 3–4]. It is a worst case; VII-M2 also reports the observed distribution, so a kernel that is merely inside the bound but far worse than its peers is visible.

These are per-operation rules. A narrowing cast amplifies a difference: an f32 difference of one unit in the last place can move a bf16 rounding by one bf16 unit. So tolerances are checked operation by operation, on identical inputs. Whole-model agreement is checked with SPINE D13's gates (KLD, top-1 agreement), never by bits against another engine (R8 §3.4).

### 11.3 Oracle tests

| Test | Compares | Where |
|---|---|---|
| per-op property tests | each kernel against `Mode::F32`/`F64` on random and adversarial inputs (−inf masks, ties, negative scales) at the §11.2 class | `crates/sky-metal/tests/oracle_ops.rs` |
| composite decomposition | the interpreter's native composite against its decomposition | `crates/sky-tir/tests/composites.rs` |
| model logits (G2) | Qwen3-0.6B logits from the interpreter against transformers in fp32 | `crates/sky-llm/tests/oracle_qwen3.rs` ([Part XXIV](part24-roadmap-now.md) #6) |
| plan invariance | decode, verify and prefill plans produce bit-identical logits and KV for the same positions | `crates/sky-llm/tests/invariance.rs` (Parts XVI, XXI) |

## 12. The Rust model builder

[Part XIII](part13-llm-runtime.md) §5.1 defines the `ArchBuilder` trait. The Qwen3 implementation in `crates/sky-llm/src/arch/qwen3.rs`:

```rust
use sky_tir::build::{GraphBuilder, GraphId, ModelGraph, TirBuilder};
use sky_tir::{AttnMask, DType::*, RopeSpec, RopeStyle, Sym};

pub struct Qwen3;

impl ArchBuilder for Qwen3 {
    fn build(&self, spec: &ModelSpec, m: &mut TirBuilder) -> Result<ModelGraph, BuildError> {
        build(spec, m)
    }
    // family(), consumed_config_keys(), parameter_names() as in Part XIII
}

pub fn build(spec: &ModelSpec, m: &mut TirBuilder) -> Result<ModelGraph, BuildError> {
    let cfg = &spec.config;                                     // normalized config.json (Part XIII §4)
    let c = m.sym_static("C", spec.max_context);
    let t = m.sym_plan("T", 1..=spec.max_tokens_per_step);
    let p = m.sym_runtime("P", 0..spec.max_context);
    m.require((p + t).le(c))?;                                  // becomes a guard if unprovable

    let layer = build_layer(m, spec, c, t, p)?;
    let step = build_step(m, spec, layer, c, t, p)?;
    m.finish(step)                                              // verifies (V1–V24); returns ModelGraph
}

fn build_layer(m: &mut TirBuilder, spec: &ModelSpec, c: Sym, t: Sym, p: Sym)
    -> Result<GraphId, BuildError>
{
    let cfg = &spec.config;
    let (d, hq, hkv, dh, f) = (cfg.hidden, cfg.heads, cfg.kv_heads, cfg.head_dim, cfg.ffn);
    let prec = &spec.precision;                                 // §7.3: act, kv, matmul_in, logits
    let mut g = m.graph("qwen3_layer");

    let h    = g.param("h", F32, [t.into(), d.into()]);
    let pos  = g.sym_param("pos", p);
    let qpos = g.param("qpos", I32, [t]).with_range(0..spec.max_context);
    let k_in = g.state("kv.k", prec.kv, [hkv.into(), c.into(), dh.into()]);
    let v_in = g.state("kv.v", prec.kv, [hkv.into(), c.into(), dh.into()]);
    // `weight` looks up the tensor's QType and shape in the file header (Part XIII §6)
    // and fails with the checkpoint name if the shape is not the one given here.
    let hdr  = &spec.weights;
    let ln1  = g.weight(hdr, "w.input_layernorm", [d])?;
    let wq   = g.weight(hdr, "w.self_attn.q_proj", [hq * dh, d])?;
    let wk   = g.weight(hdr, "w.self_attn.k_proj", [hkv * dh, d])?;
    let wv   = g.weight(hdr, "w.self_attn.v_proj", [hkv * dh, d])?;
    let qn_w = g.weight(hdr, "w.self_attn.q_norm", [dh])?;
    let kn_w = g.weight(hdr, "w.self_attn.k_norm", [dh])?;
    let wo   = g.weight(hdr, "w.self_attn.o_proj", [d, hq * dh])?;
    let ln2  = g.weight(hdr, "w.post_attention_layernorm", [d])?;
    let wg   = g.weight(hdr, "w.mlp.gate_proj", [f, d])?;
    let wu   = g.weight(hdr, "w.mlp.up_proj", [f, d])?;
    let wd   = g.weight(hdr, "w.mlp.down_proj", [d, f])?;

    // attention
    let x  = g.at("input_layernorm").rms_norm(h, ln1, cfg.eps, prec.matmul_in);
    let q  = g.at("self_attn.q_proj").qmatmul(x, wq, F32);
    let k  = g.at("self_attn.k_proj").qmatmul(x, wk, F32);
    let v  = g.at("self_attn.v_proj").qmatmul(x, wv, F32);
    let q  = g.reshape(q, [t, hq, dh]);
    let k  = g.reshape(k, [t, hkv, dh]);
    let v  = g.reshape(v, [t, hkv, dh]);
    let q  = g.at("self_attn.q_norm").rms_norm(q, qn_w, cfg.eps, F32);
    let k  = g.at("self_attn.k_norm").rms_norm(k, kn_w, cfg.eps, F32);
    let rope = RopeSpec { style: RopeStyle::Neox, theta: cfg.rope_theta, rot: dh, scaling: cfg.rope_scaling.clone() };
    let q  = g.at("rotary_emb").rope(q, qpos, &rope);
    let k  = g.at("rotary_emb").rope(k, qpos, &rope);
    let kt = g.permute(k, [1, 0, 2]);
    let kt = g.at("kv.k").cast(kt, prec.kv);
    let vt = g.permute(v, [1, 0, 2]);
    let vt = g.at("kv.v").cast(vt, prec.kv);
    let off   = g.symval(p);
    let k_out = g.at("kv.k").update(k_in, kt, off, 1);
    let v_out = g.at("kv.v").update(v_in, vt, off, 1);
    let ks = g.slice(k_out, 1, 0, p + t);
    let vs = g.slice(v_out, 1, 0, p + t);
    let qh = g.cast(q, prec.kv);
    let a  = g.at("self_attn").attention(qh, ks, vs, qpos, AttnMask::Causal, 1.0 / (dh as f64).sqrt());
    let a  = g.reshape(a, [t, hq * dh]);
    let a  = g.cast(a, prec.matmul_in);
    let o  = g.at("self_attn.o_proj").qmatmul(a, wo, F32);
    let h1 = g.at("residual").add(h, o);

    // MLP
    let x2 = g.at("post_attention_layernorm").rms_norm(h1, ln2, cfg.eps, prec.matmul_in);
    let gt = g.at("mlp.gate_proj").qmatmul(x2, wg, F32);
    let up = g.at("mlp.up_proj").qmatmul(x2, wu, F32);
    let mm = g.at("mlp.act_fn").swiglu(gt, up, prec.matmul_in);
    let dn = g.at("mlp.down_proj").qmatmul(mm, wd, F32);
    let h2 = g.at("residual").add(h1, dn);

    g.ret([h2, k_out, v_out]);
    m.add(g)
}
```

- Every `GraphBuilder` method interns through §3.2 and returns `Result` internally; a type error panics in debug builds of the builder and is returned as `BuildError` with the builder path and the Rust call site in release builds.
- `GraphBuilder::weight` reads the tensor's `QType` and shape from the file header ([Part XIII](part13-llm-runtime.md) §6), so the same builder emits `mlx.affine(b4,g64,bf16)` for an MLX file and `gguf.q4_k` or `gguf.q6_k` per tensor for a Q4_K_M GGUF, whose recipes mix formats per tensor (R8 §1.3).
- The builder is deterministic: same `ModelSpec`, same node ids, same text.
- **The D7 test.** `crates/sky-llm/tests/front_doors.rs` lowers the function `logits` of `models/qwen3.sky` (§13.9) and builds `@qwen3_logits` for the same `ModelSpec`. It runs both through the target-independent passes of [Part VIII](part08-graph-optimizer.md) (`canon`, `fold`, `cse`, `views`, `shapes`, `recognize`, and `dead` for the plans' output sets) and requires byte-identical canonical text (§10). The comparison is after these passes because the two front doors legitimately spell the same computation differently: the Sky source writes the RMS norm in primitives and lets `recognize` find it, and computes logits for every row before selecting one, while the builder emits the composite and slices first. [Part XIII](part13-llm-runtime.md)'s test XIII-T5 asks for "byte-identical L1 text"; this is the text it compares.

## 13. The Sky language surface (SPINE D8)

This section is a proposed addition to `spec/v0.1.md`, written in its style; "must" is normative. It is a proposal for after gate G4: the Rust builders of §12 are the path to first tokens (SPINE D7), and nothing on the critical path waits for it. It follows the spec's rule for new features, "intuitive, reject ambiguity": every place where two readings type-check is an error that shows both spellings, as `*` versus `@` and `v^2` are today (`spec/v0.1.md`, "Arrays: entry by entry").

### 13.1 The edition gate, and what a tensor function is

**Edition.** Everything in this section needs the next edition, proposed as `2027` (`Edition::E2027` in `crates/sky-syntax/src/edition.rs`; `edition = "2027"` in `sky.toml`). The spec's edition rules allow it: an edition may add syntax and keywords, and it never changes the meaning of a program that compiled before (`spec/v0.1.md`, "Editions"). Under 2026, each construct below is an error that names the edition:

```text
prog.sky:3:17: error: storage formats need edition 2027
  |
3 | x: Tensor[T, D] in f16
  |                 ^^^^^^ `in f16` is a storage format
  = note: this project uses edition 2026 (sky.toml)
  = help: set `edition = "2027"` in sky.toml; programs written for 2026 keep their meaning under 2027
```

In 2027, `record` is a reserved word (a 2026 program that uses it as a name gets a migration error, as the spec allows) and `of` is a contextual word, meaningful only after a shape in a type.

**Tensor values and tensor functions.**
- A *tensor value* is a value whose type has a storage format (§13.2), is an `Index` (§13.3), or is a record (§13.6), or a value computed from tensor values.
- A *tensor function* is a `def` with a tensor parameter or a tensor result. Its body is lowered to a TIR graph (§14) and follows the `inference` contract (SPINE D6): arithmetic is computed in f32; −inf and NaN follow IEEE 754 inside the function; finiteness is checked where values leave it (§14.3).
- A tensor function may call tensor functions and the builtins of §13.8. It may not `print`, use `±`, `equation` or `solve`, or call a function that is not a tensor function:

  ```text
  prog.sky:12:5: error: a tensor function cannot print
     |
  12 |     print(x)
     |     ^^^^^^^^ `layer` is a tensor function: it runs on the GPU and has no output
     = help: return the value, and print it where `layer` is called
  ```
- Tensor values are dimensionless. A unit together with a storage format is an error ("storage formats are for dimensionless tensors"): mixing units with narrow formats would make SI conversion round, which nothing in this proposal needs.

### 13.2 Storage formats

```text
type     := tag* shape [ unit ] [ "in" storage ]
          | "Index" "[" size "]"
          | record_type
          | shape "of" record_type                                   (§13.6)
storage  := "f32" | "f16" | "bf16" | "i32" | "Index" "[" size "]"
          | ( "q4" | "q8" ) "(" "group" ":" INT [ "," "scale" ":" ( "bf16" | "f16" ) ] ")"
          | QNAME                                                    (gguf.q4_k, gguf.q6_k, mx.fp4(g32), ...)
```

```sky
x: Tensor[T, 5120] in f16
W: Matrix[25600, 5120] in q4(group: 64)
tokens: Tensor[T] in Index[V]
```

- **`in` names how values are stored, not how they are computed.** Arithmetic is in f32 whatever the storage (§11.2). `in` is already a keyword (`for x in v`), so it can never be read as a unit; the units branch spells the inch `inch` for the same reason.
- **`q4(group: 64)` is `mlx.affine(b4, g64, bf16)`** ([Part XII](part12-quantization.md) §2.5). `scale: f16` selects fp16 scales and offsets, which MLX's format also allows. `group` must be 32, 64 or 128, the group sizes MLX's quantizer accepts (R9 §1.4). GGUF formats are written by their canonical names.
- **Conversion into a format happens only at a binding or a write.** `x: Tensor[T, D] in bf16 = expr` rounds `expr` to bf16, to nearest even. Writing into a slice of a stored tensor (`kv.k[:, a : b] = k`) rounds to that tensor's format. Reading a stored value widens it exactly. There is no other implicit conversion; `f16(x)`, `bf16(x)` and `f32(x)` convert explicitly.
- **Quantized formats are for weights.** A quantized tensor can be the right operand of `@` (written `Wᵀ` because checkpoints store `[out, in]`), have rows looked up (`E[tokens]`), or be dequantized with `dequant(W)`. Anything else is an error:

  ```text
  prog.sky:9:11: error: cannot add a quantized matrix
    |
  9 |     y = W + B
    |           ^ `W` is stored as q4(group: 64)
    |         - this is Matrix[F,D] in q4(group: 64)
    = note: a quantized matrix is storage for a weight: it can be the right side of `@`, have its rows looked up, or be dequantized
    = help: write `dequant(W) + B` to compute with its values; that reads and expands every weight
  ```
- An unknown format is an error that lists the known ones; a bad `group` names the allowed values (`` `q4` takes `group: 32`, `group: 64` or `group: 128`, found `group: 48` ``).

### 13.3 Sizes: products, `Index[N]`, and sizes as values

- **Products in declared sizes.** `dim := INT | NAME | dim "*" dim`. `Matrix[Hkv * G * Dh, D]` is legal; sums and differences in a declared size are not (`Matrix[N + 1, N]` stays a parse error, with the help that they are allowed in slices and `requires`). Declared sizes are products of primitive sizes so that the binder can solve them from tensor shapes (§13.7). This needs `ShapeDim::Expr(Poly)` in `crates/sky-types/src/shape.rs`, which R1 §9.3 identifies as the extension point; `Poly` already exists.
- **`Index[N]`** is the type of an integer `i` with `0 <= i < N`, known when the program runs: `pos: Index[C]`. As a storage format, `Tensor[T] in Index[V]` is a tensor of such integers, stored as i32. A token id is an index into the vocabulary axis, so `E[tokens]` with `E: Matrix[V, D]` needs no run-time bounds check.
- **Sizes as values.** Inside a tensor function, the size names of its signature are in scope as dimensionless integers: `T - 1`, `pos + T`, `iota(T)`, `reshape(o, T, Hkv * G * Dh)`.
- **Run-time sizes.** An `Index` used as a slice bound makes a run-time size: `kv.k[:, : pos + T]` is `Tensor[Hkv, pos + T, Dh]`. This is the sizes-from-variables branch's "rigid run-time size", with the bounds of TIR's symbol table (§4.3).
- **`requires` over sizes.** `requires pos + T <= C` is checked at compile time when the bounds prove it, and otherwise before every call, as a guard (§6.3).

### 13.4 Broadcasting by matching size names

Today the spec has no broadcasting between arrays, because each case "would be a guess about which axis is meant". Named axes remove the guess. The rules:

1. Two tensor operands of an entrywise operation (`+ - * /`, the two-argument math functions, comparisons) with **equal shapes** combine entry by entry, as today.
2. Otherwise they **broadcast by name**. Every axis of the operand with fewer axes must carry a size name (a size variable, or a product of size variables). Each name must occur exactly once among the other operand's axes, and in the same order. The result has the shape of the operand with more axes; the smaller operand is repeated along the axes it lacks.
3. **Names, never numbers.** An axis written as an integer (`Tensor[7, 5120]`) has no name and never broadcasts. Two axes whose sizes happen to be equal are not the same axis.
4. A number combines with a tensor of any rank (the broadcasting branch's number ⊕ vector rule, extended inside tensor functions).
5. Names are compared after following equalities between names and before replacing names by numbers. `T` and `D` remain different axes at a call where both are 128, so a function body means the same thing at every call.

With `x: Tensor[T, D]` and `w: Vector[D]`, `x * w` multiplies every row of `x` by `w`. Every way it can fail is an error with the spellings that would be valid:

```text
prog.sky:7:11: error: `w` matches two axes of `x`
  |
7 |     y = x * w
  |           ^ `w`'s axis D matches axis 1 and axis 2 of `x`
  |         - this is Tensor[T,D,D] [dimensionless]
  |             - this is Vector[D] [dimensionless]
  = note: broadcasting matches size names, and `x` has two axes named D
  = help: to multiply along axis 1, write `x * expand(w, x, at: 1)`
  = help: to multiply along axis 2, write `x * expand(w, x, at: 2)`
```

```text
prog.sky:5:7: error: cannot broadcast `w` against `x`: their axes have no names
  |
5 | y = x * w
  |       ^ `x` is Tensor[7,5120] and `w` is Vector[5120]
  = note: broadcasting matches size names, never sizes
  = help: name the axes (`x: Tensor[T, D]`, `w: Vector[D]`), or write `x * expand(w, x, at: 1)`
```

```text
prog.sky:9:11: error: `w`'s axes are in a different order than in `x`
  |
9 |     y = x * w
  |           ^ `x` is Tensor[T,H,D], `w` is Tensor[D,H]
  = help: write `x * permute(w, H, D)`
```

```text
prog.sky:4:7: error: cannot apply `*` to Vector[T] and Vector[D]: neither has all of the other's axes
  |
4 | p = a * b
  |       ^ broadcasting repeats the smaller operand; here neither is smaller
  = help: for the outer product use `outer(a, b)`
```

`expand(w, x, at: k)` is the positional escape: it places `w`'s axes at `x`'s axes `k, k+1, …`, requires the sizes there to be equal, and has `x`'s shape. (`outer` comes from the linear-algebra branch, R1 §8.2.)

### 13.5 Reductions over a named axis

- `sum(x, over: D)`, `mean`, `max`, `min`, `argmax` and `argmin` remove the axis named `D`; `softmax(x, over: S)` keeps it. `axis: k` names an axis by position instead, for the rare tensor whose axes are unnamed or share a name.
- A vector may omit the axis (`sum(v)`, as on the reductions branch). A tensor of rank 2 or more must say which axis:

  ```text
  prog.sky:4:9: error: `mean` of a Tensor[T,D] needs an axis
    |
  4 |     m = mean(x)
    |         ^^^^^^^ `x` has two axes
    = help: write `mean(x, over: D)` for one mean per row, or `mean(x, over: T)` for one per column
  ```
- `over:` takes one axis; reduce twice for two. An axis that does not exist is an error that lists the axes; a name that matches two axes is an error that shows `axis: 1` and `axis: 2`.
- Sums of f16 or bf16 values accumulate in f32 and give f32. `argmax` gives an `Index[N]` for the reduced axis's size `N`, and breaks ties toward the lowest index.

**One spelling for named arguments.** `over: D`, `theta: 1e6`, `group: 64` and `positions: qpos` are named arguments, written `name: value`. The readable-output branch spells its keyword arguments `print(x, sep=", ")` (R1 §8.2; `ExprKind::Keyword`). Sky should have one spelling, and SPINE D8 fixes `over: D` and `q4(group: 64)`, so that branch should change to `sep: ", "` before it lands. `name=value` in a call then becomes a parse error with the help "write `name: value`": `=` is assignment in Sky, and an assignment is never an expression.

### 13.6 Records

```text
record_stmt := "record" NAME [ "[" NAME { "," NAME } "]" ] ":" NEWLINE INDENT field { field } DEDENT
field       := NAME ":" type NEWLINE
record_type := NAME [ "[" dim { "," dim } "]" ]           (NAME a declared record)
```

```sky
record Mlp[D, F]:
    gate_proj: Matrix[F, D]
    up_proj: Matrix[F, D]
    down_proj: Matrix[D, F]
```

- A record is a value with named fields; its bracketed parameters are sizes. Fields are read with `.` (`w.mlp.up_proj`).
- `Vector[L] of Layer[D, F, …]` is a vector of `L` records, indexed (`layers[l]`) and iterated (`for layer in layers`) like any vector.
- Records are values. `r.f = v` rebinds `r` to a record whose field `f` is `v`, and `kv.k[:, a : b] = t` rebinds `kv` with that slice of `k` replaced. This is the arrays-you-can-build branch's functional `store`, which updates in place when the old value is not used again (R1 §8.2), and which TIR lowers to `update` (§14.2).
- A lone name in a type is a record if a record of that name is declared, and a unit otherwise. Declaring a record with a unit's name (`record m:`) is an error, so a type never has two readings.
- No recursion, no methods, no default values. A duplicate field, an unknown field (with a did-you-mean) and a field with a value are errors.
- TIR has no records: lowering flattens a record into its fields (§14.2).

### 13.7 Weights as value-less declarations

- A module-level declaration with no value is an input (`spec/v0.1.md`, "Type annotations": "Declaration without a value is an input"). `weights: CausalLM[V, D, F, Hkv, G, Dh, L]` declares the whole checkpoint.
- `sky run models/qwen3.sky weights=hf://Qwen/Qwen3-32B` binds it, using the data-in branch's `name=value` command-line inputs (R1 §8.2). `sky run hf://Qwen/Qwen3-32B --arch models/qwen3.sky` runs [Part XIII](part13-llm-runtime.md)'s generation loop with the file's `logits` function in place of the built-in builder.
- **Binding** happens before anything runs, and any failure stops the run (exit status 1):
  - each field path names checkpoint tensors: drop the root name, append `.weight`, and take `.scales` and `.biases` for an MLX quantized tensor (`weights.model.layers.17.self_attn.q_proj` → `model.layers.17.self_attn.q_proj.weight`, `.scales`, `.biases`);
  - the storage format comes from the file, unless the declaration names one, which must then match;
  - sizes are solved from the tensors' shapes. A product is solved when all its factors but one are known: `Dh = 128` from `q_norm`, then `Hkv = 8` from `k_proj`'s 1024 rows, then `G = 8` from `q_proj`'s 8192 rows;
  - every checkpoint tensor must be used, as [Part XIII](part13-llm-runtime.md) §5.1 requires of the builders, unless the declaration's skip list names it.

```text
error: the checkpoint has no tensor for `weights.model.layers.3.self_attn.q_norm`
  = note: looked for `model.layers.3.self_attn.q_norm.weight` in hf://Qwen/Qwen3-32B
  = help: check the field's name in `record Attention`; or this checkpoint is not a Qwen3 model

error: `weights.model.layers.0.self_attn.k_proj` is [1024, 5120] in the checkpoint, but its type is Matrix[Hkv * Dh, D] = [512, 5120]
  = note: Hkv = 4 was given on the command line (`--size Hkv=4`); Dh = 128 comes from `self_attn.q_norm`
  = help: the checkpoint needs Hkv = 8

error: cannot determine Hkv and G from the checkpoint
  = note: `q_proj` has Hkv * G * Dh = 8192 rows and Dh = 128, so Hkv * G = 64; no tensor fixes Hkv alone
  = help: pass the size explicitly: `--size Hkv=8`
```

### 13.8 Builtins of tensor functions

| Builtin | TIR | Notes |
|---|---|---|
| `rsqrt`, `sigmoid`, `silu`, `gelu(x, approx: tanh)` | elementwise op or composite | with the existing math functions, applied entrywise |
| `sum`, `mean`, `max`, `min`, `argmax`, `argmin` with `over:` or `axis:` | `reduce.*`, `argmax`, `argmin` | §13.5 |
| `softmax(x, over: S)` | `softmax` | |
| `reshape(x, dims…)`, `permute(x, names…)`, `expand(w, x, at: k)` | views | `permute` names the axes in their new order |
| `iota(N)` | `iota` | an `Index[N]` tensor `0, 1, …, N−1` |
| `f32(x)`, `f16(x)`, `bf16(x)`, `dequant(W)` | `cast`, `dequant` | |
| `E[idx]` with `idx: Tensor[…] in Index[R]` | `gather_rows` | rows of a matrix, quantized or not |
| `rope(x, positions, theta:, style:, rot:, scaling:)` | `rope` | `style: neox` is the default |
| `attention(q, k, v, positions:, window:, sinks:, scale:)` | `attention` | causal; `window: W` gives a sliding window; `scale` defaults to `1/sqrt(Dh)` |
| `sample(logits, key:, position:, temperature:, top_k:, top_p:)` | `sample` | keyed by `(key, position)` |

`rms_norm` is deliberately not a builtin: three lines of Sky say exactly what it computes, and Part VIII's recognition finds it (§13.9, line 40).

### 13.9 Qwen3 in Sky source

The whole dense Qwen3 family, from 0.6B to 32B, is one file: the sizes come from the checkpoint (§13.7). [Part VIII](part08-graph-optimizer.md) §8 cites its line numbers.

```sky
# models/qwen3.sky: the Qwen3 dense decoder. Needs `edition = "2027"` in sky.toml.
# The record tree is the checkpoint's module tree: `model.layers.17.self_attn.q_proj`.

record Attention[D, Hkv, G, Dh]:
    q_proj: Matrix[Hkv * G * Dh, D]
    k_proj: Matrix[Hkv * Dh, D]
    v_proj: Matrix[Hkv * Dh, D]
    o_proj: Matrix[D, Hkv * G * Dh]
    q_norm: Vector[Dh]
    k_norm: Vector[Dh]

record Mlp[D, F]:
    gate_proj: Matrix[F, D]
    up_proj: Matrix[F, D]
    down_proj: Matrix[D, F]

record Layer[D, F, Hkv, G, Dh]:
    input_layernorm: Vector[D]
    self_attn: Attention[D, Hkv, G, Dh]
    post_attention_layernorm: Vector[D]
    mlp: Mlp[D, F]

record Decoder[V, D, F, Hkv, G, Dh, L]:
    embed_tokens: Matrix[V, D]
    layers: Vector[L] of Layer[D, F, Hkv, G, Dh]
    norm: Vector[D]

record CausalLM[V, D, F, Hkv, G, Dh, L]:
    model: Decoder[V, D, F, Hkv, G, Dh, L]
    lm_head: Matrix[V, D]

record Cache[Hkv, C, Dh]:
    k: Tensor[Hkv, C, Dh] in f16
    v: Tensor[Hkv, C, Dh] in f16

weights: CausalLM[V, D, F, Hkv, G, Dh, L]        # no value: bound when the program runs
eps = 1e-6

def rms_norm(x: Tensor[T, D], w: Vector[D]) -> Tensor[T, D]:
    return x * rsqrt(mean(x * x, over: D) + eps) * w

def head_norm(x: Tensor[T, H, Dh], w: Vector[Dh]) -> Tensor[T, H, Dh]:
    return x * rsqrt(mean(x * x, over: Dh) + eps) * w

def layer(h: Tensor[T, D], pos: Index[C], qpos: Tensor[T] in Index[C],
          w: Layer[D, F, Hkv, G, Dh], kv: Cache[Hkv, C, Dh]) -> (Tensor[T, D], Cache[Hkv, C, Dh]):
    requires pos + T <= C
    a = w.self_attn
    x: Tensor[T, D] in bf16 = rms_norm(h, w.input_layernorm)
    q = reshape(x @ a.q_projᵀ, T, Hkv * G, Dh)
    k = reshape(x @ a.k_projᵀ, T, Hkv, Dh)
    v = reshape(x @ a.v_projᵀ, T, Hkv, Dh)
    q = rope(head_norm(q, a.q_norm), qpos, theta: 1e6)
    k = rope(head_norm(k, a.k_norm), qpos, theta: 1e6)
    kv.k[:, pos : pos + T] = permute(k, Hkv, T, Dh)
    kv.v[:, pos : pos + T] = permute(v, Hkv, T, Dh)
    qh: Tensor[T, Hkv * G, Dh] in f16 = q
    o = attention(qh, kv.k[:, : pos + T], kv.v[:, : pos + T], positions: qpos)
    ob: Tensor[T, Hkv * G * Dh] in bf16 = reshape(o, T, Hkv * G * Dh)
    h = h + ob @ a.o_projᵀ
    x2: Tensor[T, D] in bf16 = rms_norm(h, w.post_attention_layernorm)
    m = w.mlp
    gu: Tensor[T, F] in bf16 = silu(x2 @ m.gate_projᵀ) * (x2 @ m.up_projᵀ)
    h = h + gu @ m.down_projᵀ
    return h, kv

def logits(tokens: Tensor[T] in Index[V], pos: Index[C],
           cache: Vector[L] of Cache[Hkv, C, Dh]) -> (Tensor[V], Vector[L] of Cache[Hkv, C, Dh]):
    requires pos + T <= C
    qpos = pos + iota(T)
    h = weights.model.embed_tokens[tokens]
    for l in range(L):
        h, cache[l] = layer(h, pos, qpos, weights.model.layers[l], cache[l])
    x: Tensor[T, D] in bf16 = rms_norm(h, weights.model.norm)
    return (x @ weights.lm_headᵀ)[T - 1], cache
```

How it type-checks, at the lines that use the new rules:
- **Lines 40 and 43.** `mean(x * x, over: D)` is a `Tensor[T]`; `+ eps` broadcasts a number; `x * rsqrt(…)` broadcasts the `Tensor[T]` along `D` by name; `* w` broadcasts `Vector[D]` along `T`. In `head_norm` the same text works on `Tensor[T, H, Dh]` because the names, not the positions, decide.
- **Line 49.** The annotation `in bf16` rounds the f32 result once, to the checkpoint's dtype, before the matrix products: the precision plan of §7.3, written in the source.
- **Lines 50–52.** `x @ a.q_projᵀ` multiplies by a quantized weight; the type of `a.q_proj` comes from the checkpoint at binding. `reshape` splits `Hkv * G * Dh` into three named axes; the element counts agree as polynomials.
- **Lines 55–56.** Writes into the f16 cache round to f16 and rebind `kv` (§13.6). The slice length `(pos + T) − pos` simplifies to `T`, the length of the written value.
- **Line 58.** `kv.k[:, : pos + T]` has the run-time size `pos + T`. `attention` requires the query head count `Hkv * G` to be a multiple of the key head count `Hkv`, which holds as polynomials, whatever the sizes.
- **Line 63.** `silu(…) * (…)` multiplies two `Tensor[T, F]`s entry by entry; Part VIII recognizes it as `swiglu`.
- **Line 75.** The logits of every row are computed and the last one is selected. Part VIII's dead-output pass computes one row instead (`TIR-DEAD-002`).

## 14. Lowering from Sky to TIR, and `tensor.call` in SIR

### 14.1 Checker changes

All in `sky-types`, all edition-gated:

| Change | Where |
|---|---|
| storage formats on tensor types | `TensorType` gains `storage: Option<Storage>` (`crates/sky-types/src/rules.rs:18`); `TypeExpr` gains `storage` (`crates/sky-ast/src/ast.rs:212`) |
| size products | `ShapeDim::Expr(Poly)` (`crates/sky-types/src/shape.rs:98`) |
| `Index[N]`, records, vectors of records | new `Ty` variants (`crates/sky-types/src/check.rs:161`), new `crates/sky-types/src/record.rs` |
| named broadcasting and its errors | new `crates/sky-types/src/broadcast.rs`, called by the entrywise rules in `rules.rs` |
| named reductions | the reductions branch's `reduce.rs`, extended with `over:` and `axis:` |
| tensor-function rules (§13.1) | new `crates/sky-types/src/tensor_fn.rs` |
| `MAX_ARRAY_ELEMENTS` | not applied to tensor values (`crates/sky-types/src/constant.rs:23`) |

### 14.2 Lowering

`crates/sky-lower/src/tensor.rs` (new) lowers each tensor function to a TIR graph with `sky_tir::build::GraphBuilder`, giving every node an `Origin::Source` with the expression's span. It reads the checker's types and never re-derives them, as SIR lowering does (`crates/sky-lower/src/lib.rs:360`).

| Sky | TIR |
|---|---|
| a parameter `x: Tensor[T, D] in f16` | `param %x: f16[T, D]` (`T`, `D` become symbols or constants) |
| a record parameter `w: Layer[…]` | one parameter per field, named by its path (`w.self_attn.q_proj`) |
| `pos: Index[C]` | `param %pos: sym P` with `P: 0..C` |
| a module-level weight input read by the function | one `weight` parameter per field it reads (`weights.model.norm`) |
| entrywise operations | the primitive op, with `cast[f32]` on narrower operands and `broadcast` views for named broadcasting |
| `x @ Wᵀ` | `qmatmul` (quantized `W`) or `matmul[tb]` |
| `sum(x, over: D)`, `mean`, … | `reduce.*[axes=(position of D)]` |
| an annotated binding `y: … in bf16 = e` | `cast[bf16]` |
| `kv.k[:, a : b] = t` | `cast` to `kv.k`'s format, then `update[axis=1]`; `kv.k` becomes a `state` result of the graph |
| `E[tokens]` | `gather_rows` |
| `for l in range(L):` over a vector of records | unrolled into `L` `call`s of the callee's graph, each with that record's fields |
| a call of a tensor function | `call @f`, with records flattened |
| `requires` over sizes | a `require` constraint, proven or turned into a guard |
| `return h, kv` | the graph's results; record fields flattened |

### 14.3 `tensor.call` in SIR

One new SIR operation (VI-T3 in [Part VI](part06-architecture.md)) runs a compiled tensor program from ordinary Sky code:

```text
%r = tensor.call @logits %pos, %tokens : (num[1; (151936)]) effects(call, fail) @ 812..860
```

- **Kind.** `OpKind::TensorCall { graph: Arc<str> }` in `crates/sky-ir/src/op.rs`, with its verifier rule, printer, parser and interpreter arms. Like every new `OpKind`, it touches every exhaustive match (R1 §9.3 counts about 10 crates); it is one operation, added once.
- **Operands** are the SIR values for the graph's `input` and `sym` parameters, in order. They must be numbers or arrays of SIR's kinds: rank ≤ 2, f64, at most 2²⁴ entries (R1 §3.2). Each is converted to the parameter's dtype by rounding to nearest even. A value that does not fit (beyond f16's 65504, a non-integer for an `Index`, an index out of range) is a run-time error naming the parameter.
- **Weights and state never cross the boundary.** They live in the tensor runtime (`sky_tir::runtime::TensorRuntime`) and are bound by name: weights from the module's value-less declarations (§13.7), state created on first use and kept across calls.
- **Results** are converted to f64, which is exact from f32, f16 and bf16. A non-finite result is a run-time error ("the tensor function `logits` returned a non-finite value at [1234]"), so scientific Sky keeps its rule that a number is always finite (`spec/v0.1.md`, "Arithmetic"); the rule moves from every operation to the boundary.
- **Effects** are `call` and `fail`: a guard can fail, and so can a conversion.
- **Engines.** The tree-walker, the SIR interpreter and native code all call the same `TensorRuntime::call`, so they still print identical bytes. What may differ between tensor backends (the Metal kernels against the reference interpreter) is governed by the tolerances of §11.2, not by the three-engine rule.
- **`sky check` never runs anything** (non-negotiable 1). Checking a tensor function type-checks it and builds its TIR; `tensor.call` executes only under `sky run`.

## 15. What Sky lacks today

R1 §9.3b lists what the language would need before a Qwen3 decoder layer could be written in it. R1 counts 12 items, all VERIFIED missing on `main`; this part adds three it needs on the way.

| # | Missing on `main` (R1 §9.3b) | In-flight branch (R1 §8.2) | This part |
|---|---|---|---|
| 1 | Tensors of rank 3 and 4, reshape and axis permutation | — | `reshape`, `permute`, rank-N types (§13.8) |
| 2 | dtypes: f16/bf16 activations, quantized weights, casts | — | storage formats, `in` (§13.2) |
| 3 | integer scalars and tensors | — | `Index[N]`, `in Index[N]`, `in i32` (§13.3) |
| 4 | broadcasting between arrays | broadcasting (a383df0a): number ⊕ vector only; arrays still forbidden | broadcasting by size names (§13.4) |
| 5 | axis reductions, softmax, argmax | reductions (a3be8cf3): vectors only, 13 functions, spec and goldens | `over:` and `axis:` (§13.5) |
| 6 | gather and scatter | — | `E[idx]` gather; slice writes cover the scatter LLMs need |
| 7 | KV-cache update and run-time sizes | arrays-you-can-build (a0952514): functional `store`, in place; sizes-from-variables (abf8af99): checker only | `update` (§5.8); `Index` slice bounds (§13.3) |
| 8 | batched matmul and attention with GQA | — | `attention` builtin; `@` with quantized weights |
| 9 | records for layer parameters; a list of layers | tuples (ad1ea58e): multiple returns, no nesting | `record`, `Vector[L] of R` (§13.6) |
| 10 | binary input of weights; bytes for the tokenizer | data in (a0bbbe93): `name=value` command-line inputs, CSV | weights as value-less declarations (§13.7); the tokenizer stays in Rust (Part XIII) |
| 11 | IEEE −inf and NaN in masked softmax | — | inside tensor functions only; finiteness at the boundary (§13.1, §14.3) |
| 12 | a model of bytes per dtype | — | TIR's cost model reads dtypes and `bits_per_weight` (§4.2, Part XXII) |
| 13 | size products (`Hkv * Dh`) | sizes-from-variables: `a·N + b` in one variable | `ShapeDim::Expr(Poly)` (§13.3) |
| 14 | named arguments | readable output (a1dadc02): `sep=` keyword arguments | one spelling, `name: value` (§13.5) |
| 15 | a way to say "this code is a tensor program" | — | tensor functions and the 2027 edition (§13.1) |

SPINE D17 lands three branches first: arrays-you-can-build, sizes-from-variables and reductions. Rows 5 and 7 build on them directly; row 13 extends sizes-from-variables from one variable to products. The tuples branch (row 9) is needed for `return h, kv`; it is among the parked branches, and one of the three whose test targets do not compile (R1 §8.4), so it needs that fix before §13 can land.

## 16. Disagreement with the spine

Two points. Neither changes a decision.

1. **Chunk invariance and recurrent layers (SPINE D6).** D6 requires prefill results not to depend on the chunk size. For attention layers that is a kernel property Part XVII can deliver. For Gated DeltaNet layers (Qwen3.5/3.6/3.8, SPINE D10) it conflicts with the fast prefill form: the chunked parallel scan sums in a different order than the one-token recurrence (R8 §3.5 rule 4), so a prompt prefilled in chunks leaves a recurrent state that is not bit-identical to the state token-by-token processing would leave, and decode continues from that state. R8 resolves this only for verification ("run the recurrent update per position inside the verify pass"). Two exact options exist, and one inexact one:
   - recurrent form in prefill too: exact, slower prefill [EXPERIMENTAL: VII-M7 measures how much];
   - chunked form with chunk boundaries at fixed absolute positions: invariant across prefill chunkings that are multiples of the scan chunk, but still not equal to token-by-token decode;
   - chunked form as a recorded contract rewrite (`TIR-DELTA-001`), giving up D6's chunk invariance for those layers, labeled in the plan.

   This part makes `form=recurrent` the definition (§5.13.7) and leaves the choice to the measurement. D6 should be read as holding exactly for attention models, and for hybrid models only with the recurrent prefill. Resolved: the spine now says (D6) that hybrid Gated-DeltaNet layers are the one exception to chunk invariance: they use the slower recurrent-order prefill or a labeled mode that waives chunk invariance for those layers, and measurement VII-M7 decides.
2. **"Identical L1" (SPINE D7).** The two front doors produce identical text only after the target-independent normalization passes of Part VIII (§12), because the Sky source writes in primitives what the builder writes as composites. The test is exact after those passes; it is not exact on the raw lowering. Resolved: the spine now says (D7) that both front doors must produce identical TIR after Part VIII's canonicalization passes.

## 17. Implementation tasks

| ID | Task | Crate / files | Deliverable | Acceptance test |
|---|---|---|---|---|
| VII-T1 | Graph, interning, types, dims, symbols | `crates/sky-tir/src/{graph.rs, dtype/mod.rs, shape.rs, sym.rs}` | `Module`, `Graph`, `Node`, `Type`, `Dim`, `SymTable` with `prove_le`, `range`, `divides` | property tests: interning is idempotent; `structural_hash` stable across runs and thread counts; `prove_le` never says `Proven` when an exact evaluation at random points within the bounds refutes it |
| VII-T2 | Shared explanations and pass manager | move `Explanation`, `RuleId`, `PassCx` from `crates/sky-ir/src/pass.rs` to `sky-diag`, re-exported by `sky-ir`; `crates/sky-tir/src/pass.rs` | one explanation type for SIR and TIR | every `tests/ir/*.explain` golden is byte-identical before and after |
| VII-T3 | Operation table and type rules | `crates/sky-tir/src/{op.rs, infer.rs}` | §5's rules | one passing and one failing case per rule of §5, each failure with its message |
| VII-T4 | Index functions and the simplifier | `crates/sky-tir/src/index.rs` | composition, simplification with bounds, zero-copy detection | property test: a composed function equals sequential application on random shapes; every view chain of the Qwen3 and Gemma 3 graphs ends affine without `//` or `%` |
| VII-T5 | Verifier V1–V24 | `crates/sky-tir/src/verify/` (with XII-T2 for `quant.rs`) | total verifier | a failing fixture per rule (`tests/verify.rs::vNN_*`); the fuzz target never panics |
| VII-T6 | Text format | `crates/sky-tir/src/text/{print,parse}.rs`, `fuzz/fuzz_targets/tir_parse.rs` | print, parse, canonical text | round trip on every test graph and on the §9.2 layer; 60 s of fuzzing per PR in CI, as for SIR |
| VII-T7 | Reference interpreter | `crates/sky-tir/src/interp/` | `Mode::F32`, `Mode::F64` | bit-identical results for 1, 2 and 8 threads; per-op known-answer tests |
| VII-T8 | Composites and their decompositions | `crates/sky-tir/src/composite/` | `rms_norm`, `rope` (all scalings), `softmax`, `attention` (all masks), GLUs | native interpreter versions equal their decompositions bit for bit in `Mode::F32`; a dev-only test against transformers' modules on random tensors, within §11.2 |
| VII-T9 | Gated DeltaNet and causal convolution | `crates/sky-tir/src/composite/deltanet.rs` | `deltanet.scan` (recurrent, chunked, checkpoints), `conv.causal` | dev-only comparison with transformers' `qwen3_next` layer; chunked equals recurrent within class S |
| VII-T10 | Counter-based RNG and `sample` | `crates/sky-tir/src/interp/rng.rs`, `composite/sample.rs` | Philox4x32-10, `rng.uniform`, `sample` | Philox known-answer vectors; a χ² test of `sample`'s distribution; equal samples for equal (key, position) whatever the batch |
| VII-T11 | Builder API and the Qwen3 builder | `crates/sky-tir/src/build.rs`; `crates/sky-llm/src/arch/qwen3.rs` (XIII-T5) | `TirBuilder`, `GraphBuilder`, `ModelGraph` | G2: Qwen3-0.6B logits from the interpreter match transformers in fp32 (`crates/sky-llm/tests/oracle_qwen3.rs`) |
| VII-T12 | Plans, guards, plan invariance | `crates/sky-tir/src/plan/`, `verify/plan.rs` | `PlanSpec`, guards, V17, V18 | the guard error of §6.3 as a UI golden; a fixture with mismatched `ReduceSchedule`s fails V18 |
| VII-T13 | Language surface in the checker | `sky-syntax` (edition 2027), `sky-ast` (`TypeExpr.storage`, records, named arguments), `sky-types` (§14.1) | §13 | a UI golden for every error in §13; every existing golden unchanged; `models/qwen3.sky` checks |
| VII-T14 | Lowering and `tensor.call` | `crates/sky-lower/src/tensor.rs`; `crates/sky-ir` (with VI-T3) | §14.2, §14.3 | D7: `crates/sky-llm/tests/front_doors.rs` byte-identical canonical text; `tests/ir/tensor_call.sky` golden; the three engines print identical bytes for a program that uses `tensor.call` |

## 18. Measurements that decide

| ID | Measurement | Decides |
|---|---|---|
| VII-M1 | Time to build and verify the Qwen3-32B TIR module (64 calls of one layer), on the M2 Pro | Whether verification stays on in release builds; target: negligible next to weight loading |
| VII-M2 | Observed per-op error distributions of the first Metal kernels against `Mode::F64`, against the §11.2 bounds | Whether the class S bound is a useful test threshold or a tighter empirical one is needed |
| VII-M3 | Time of Part VIII's pipeline on the 0.6B, 8B and 32B graphs, against tinygrad's 76–151 ms per block schedule on a miss (R2 §3.7) | Whether the interning-and-rebuild design needs incremental rewriting |
| VII-M4 | Interpreter speed on Qwen3-0.6B and 8B | Which oracle tests run per PR and which nightly |
| VII-M5 | RoPE with f32 angles against f64 angles at 32K and 128K positions: logits KLD | Whether `angle=f64` becomes the default for long contexts |
| VII-M6 | DeltaNet rollback: checkpoints versus replay on Qwen3.6-27B, time and bytes per verify step | §6.4's choice |
| VII-M7 | Gated DeltaNet prefill: recurrent versus chunked form, tokens per second | §16 item 1 |
| VII-M8 | Count of `//` and `%` left in the index expressions of the generated Qwen3 kernels | The simplifier's quality; tinygrad's Q4_0 GEMV kept a run-time division for the nibble shift (R2 §7.3) |

## 19. Risks and negative results

- **Generic code generation from a good IR is not enough.** tinygrad has the IR this part copies and still reaches only 19–43% of peak bandwidth on fused quantized GEMVs, because its code generator treats a dequantized weight as an expression (R2 §8.4). TIR prevents that failure by typing quantized weights, but only hand-designed skeletons (SPINE D4) avoid it in the kernels. The IR is necessary, not sufficient.
- **The composite set can grow without bound.** Every new architecture wants a composite (M-RoPE, MLA, sinks, softcaps). Policy: a new composite only when a kernel skeleton or a fusion needs to see it whole; otherwise primitives.
- **`Poly` cannot express `min`, `max` or division.** A sliding-window KV read of length `min(S, W)` cannot be a dim. It is expressed through the `sliding(W)` mask over the full `[0, S)` range, with a rotating cache of constant size `W` for local layers. If a model needs a true `min` in a shape, `Dim` needs a new variant.
- **Bit-exactness depends on kernels TIR cannot see.** V18 checks that plans *declare* equal reduction schedules; whether kernels honour them is tested only by Parts XVI and XXI's invariance tests.
- **The language surface may go unused.** If the Rust builders cover every model, few will write models in Sky. §13 is after G4 for that reason, and the D7 test keeps the two front doors honest.
- **The interpreter may be too slow for large oracles in CI.** Qwen3-0.6B runs per PR, 8B nightly, 32B one layer at a time (VII-M4).
- **A spelling conflict with an in-flight branch.** Named arguments as `name: value` (§13.5) require a small change to the readable-output branch before it lands.
- **Negative result to remember:** MLX's `mx.compile` fuses only elementwise chains, at most 11 deep (R3 §1.4), and mlx-lm uses it only for the SwiGLU. A fusion design that cannot see norms, RoPE and attention as units cannot beat it; this is why composites are first-class here.
