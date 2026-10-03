# Part VIII — Graph optimizer: rewrites, fusion and lazy execution

This part specifies the passes that run over TIR between the model builders ([Part VII](part07-tensor-ir.md)) and the scheduler ([Part IX](part09-scheduler.md)):
- why Sky needs no run-time laziness, and what is still decided at run time;
- the ordered pass pipeline, with each pass's purpose, legality, rule IDs and explanation text;
- the rewrite engine (declarative patterns compiled to matchers) and the limited role of the e-graph;
- which rewrites are exact and which are allowed only under the `inference` contract, and how the latter are recorded;
- fusion legality (the rangeify rule) and the interface through which the scheduler decides profitability;
- what `sky explain` prints for a Qwen3 layer.

All passes live in `crates/sky-tir/src/passes/` and run under `sky_tir::pass::PassManager`, which has the API of SIR's (`crates/sky-ir/src/pass.rs`): `verify_each`, `print_after`, and explanations through `PassCx::explain`. Evidence labels follow SPINE §8.

## 0. Summary

| # | Decision | Section |
|---|---|---|
| VIII-D1 | **No run-time laziness.** A model's graphs are built, optimized and specialized once per (model, plans, device), ahead of time or at first use, and cached by structural hash. Per step, the runtime only supplies the position, picks a prebuilt plan and binds the true inputs. | §1 |
| VIII-D2 | An **ordered, deterministic pipeline** of twelve passes, verified after each pass. Every change is an explanation with a stable rule ID. | §2, §3 |
| VIII-D3 | Rewrites are **declarative rules** in a data-only pattern language, compiled at build time into Rust matchers indexed by root operation (tinygrad's PatternMatcher lesson). Side conditions are named Rust predicates. | §4 |
| VIII-D4 | Rewrites are **exact by default**. Rewrites that change rounding are allowed only under the `inference` contract and are recorded in the plan's `ContractLog`. Rewrites that change values (re-quantization) are forbidden in inference graphs (SPINE D5). | §7 |
| VIII-D5 | **Horizontal fusion** (Q/K/V concatenation, gate/up interleaving) is a graph rewrite plus a **weight relayout plan**, which the `.skyweights` build executes with Part XII's lossless recipes. It is exact. | §3.8 |
| VIII-D6 | **Legality here, profitability in Part IX.** The optimizer emits `FusionLegality`: the producer–consumer pairs that may fuse, by the rangeify rule ("consumers agree on the index"), and the skeleton templates the graph matches. The scheduler chooses among them with its cost model. | §6 |
| VIII-D7 | The **e-graph** (`sky-egraph`, egg) is used only for block-level algebra on one layer template, with DAG-aware roofline extraction. It is EXPERIMENTAL and off by default. | §5 |
| VIII-D8 | **Weight and state layouts are chosen once for all plans**, because decode, verify and prefill share one copy of the weights and one KV cache. Intermediate layouts are Part IX's. | §3.11 |

## 1. Why Sky needs no run-time laziness

Lazy evaluation exists in tinygrad and MLX for reasons that do not apply to Sky.

**tinygrad.** Every tensor operation builds a hash-consed UOp, and nothing runs until a value is needed (`tensor.py:104-113`, `218-225`). Laziness serves two purposes: to see fusion opportunities through an eager Python API, and to capture a graph for the JIT (R2 §9.3). Its costs, measured on our M2 Pro:
- a trivial `(x+1).realize()` with every cache warm costs 453 µs;
- a schedule-cache hit for one transformer block costs about 1.1 ms of Python, so an eager 28-layer decode step pays ≳ 30 ms before any GPU work [LIKELY: per-block hits summed];
- TinyJit, which exists to remove that, still costs 155 µs per call (R2 §4, §7.5).

**MLX.** The graph is rebuilt in Python for every token. Building the Qwen3 decode graph costs 1.2–1.4 ms per token for 64 layers (1,872 nodes, about 1,225 of them launching kernels), and the framework floor is about 3.5 µs per kernel. mlx-lm hides this behind the GPU with `async_eval` for large models, but the floor caps small draft models: a 28-layer Qwen3-0.6B drafter is limited to about 1.9 ms per token before any arithmetic on the M2 Pro (R3 §1.2, §7.E, §7.F) [LIKELY: R3 scales this from its measured 36- and 64-layer runs]. `mx.compile` fuses only elementwise chains (R3 §1.4).

**Sky.** The compiler sees the whole program before anything runs: the model is a TIR module from a builder, or a Sky file whose tensor functions are lowered as a whole. There is no Python to trace, no graph break, and no guard that triggers a recompilation: shapes, dtypes and formats are in the types (R7 Topic 4, "What Sky gains by owning the language"; Dynamo needs a mean of 21 graphs per TorchBench model and only 70% of models have none broken). So the graph is built, optimized and specialized once, and the plan is cached under `Graph::structural_hash` and the `PlanSpec` ([Part VII](part07-tensor-ir.md) §3.2, §7.2).

**What is still decided when the program runs**, and how, with no compilation:

| Run-time fact | How it enters | Where specified |
|---|---|---|
| position `P` (and `S = P + T`) | a run-time symbol: a kernel argument in the uniform buffer, written by the GPU in the resident decode loop | Part VII §4.3; Part XX |
| which plan runs (decode, a verify width, a prefill bucket) | the runtime picks a prebuilt plan from `(T, S)` | Part VII §7.2 |
| context bucket (attention split parameters) | a plan sub-key selected by `S` | Part XV |
| addresses of the true inputs and outputs (token ids, logits) | bound per step; everything else lives at fixed offsets of the static memory plan | Part XIX |
| guards that the bounds did not prove | evaluated on the CPU before submission | Part VII §6.3 |
| accepted draft length, stop conditions | data on the GPU; selects which state version to keep | Parts XXI, XX |
| device and tuning entry | fixed per artifact; the tuning database is consulted at plan time | Part X |
| weight format and layout | fixed per artifact, from the file header | Part XII |

An eager, interactive mode (a REPL over tensors) is not a year-1 goal. The reference interpreter of Part VII §11 already evaluates any graph directly when one is needed.

## 2. The pipeline

```text
 builder / lowering
        │   L1 module (generic: T is a plan symbol, P a run-time symbol)
        ▼
 verify ─► canon ─► fold ─► cse ─► views ─► shapes ─► recognize ─► horizontal          once per model
                                                                       │
                                     normalized L1 (the D7 comparison) ◄┤
                                                                       ▼
 for each plan:            specialize ─► dead ──┐                                       once per plan
                                                ▼
 for all plans together:   layout (weights and state)                                   once per model
                                                │
 for each plan:            legality ◄───────────┘                                       once per plan
                                                │
                                                ▼
              OptimizedTir { plans, FusionLegality, LayoutPlan, RelayoutPlan, ContractLog }
                                                │
                                                ▼
                         Part IX: partition, schedule, memory plan
```

| # | Pass (`name()`) | File | Purpose | Rule IDs |
|---|---|---|---|---|
| 1 | `verify` | `verify/mod.rs` | reject invalid input before any pass (V1–V24) | — |
| 2 | `canon` | `passes/canon.rs` | one canonical form per computation | `TIR-CANON-001`…`007` |
| 3 | `fold` | `passes/fold.rs` | evaluate constant and weight-only subgraphs | `TIR-FOLD-001`…`003` |
| 4 | `cse` | `passes/cse.rs` | re-intern after rewrites; report merges of distinct source sites | `TIR-CSE-001` |
| 5 | `views` | `passes/views.rs` | compose views into index functions; move views toward sources; mark zero-copy reads | `TIR-VIEW-001`…`004` |
| 6 | `shapes` | `passes/shapes.rs` | prove constraints, or turn them into guards; index width; divisibility facts | `TIR-SHAPE-001`…`005` |
| 7 | `recognize` | `passes/composite.rs` | composites from primitives | `TIR-NORM-001/002`, `TIR-ROPE-001`, `TIR-SOFTMAX-001`, `TIR-ATTN-001`, `TIR-GLU-001/002` |
| 8 | `horizontal` | `passes/horizontal.rs` | concatenate products that share an input; relayout plan | `TIR-HCAT-001`…`003` |
| 9 | `specialize` | `passes/specialize.rs` | one graph per plan; simplify with the bound `T` | `TIR-PLAN-001`…`004`, `TIR-DELTA-001` |
| 10 | `dead` | `passes/dead.rs` | remove outputs a plan does not need; push row selections upward | `TIR-DEAD-001`, `TIR-DEAD-002` |
| 11 | `layout` | `passes/layout.rs` | weight and state layouts, once for all plans | `TIR-LAYOUT-001`…`003` |
| 12 | `legality` | `passes/legality.rs` (analysis) | `FusionLegality` for Part IX; reports | `TIR-FUSE-001`…`004`, `TIR-QUANT-001` |

**Why this order.** It follows the reasoning `docs/design/ir.md` gives for SIR's pipeline.
- `canon` first, so every later rule sees one form.
- `fold` after `canon`, so constants are in canonical position; `cse` after `fold`, because folding makes equal constants equal.
- `views` before `shapes`, because bounds are proven on composed index functions.
- `shapes` before `recognize`, because the patterns test shapes, such as "the weight's axis is the axis averaged over".
- `recognize` before `horizontal`, because gate/up interleaving needs to know that `swiglu` is the only consumer.
- `horizontal` before `specialize`, because weights are shared by every plan, so concatenation and interleaving must be decided once, for all plans.
- `specialize` before `dead`, because what is dead depends on the plan (a prefill plan needs only the last row of logits).
- `layout` after `dead`, because dead values get no layout.
- `legality` last; it changes nothing.

Passes 1–8 are target-independent and run once per model; their output is the **normalized L1** that the D7 test compares between the two front doors ([Part VII](part07-tensor-ir.md) §12). Passes 9, 10 and 12 run once per plan; pass 11 runs once over all plans together, because weights and state are shared.

## 3. The passes

Each subsection gives the pass's purpose, its legality condition, its rule IDs and a sample explanation in `sky explain`'s format (§8).

### 3.1 `verify`

The verifier of [Part VII](part07-tensor-ir.md) §8 runs before the first pass and, in tests and debug builds, after every pass. A failure before the first pass is a builder or lowering bug; a failure after a pass names that pass, as SIR's `PassError::InvalidAfter` does.

### 3.2 `canon`: one form per computation

| Rule | Rewrite | Exact because |
|---|---|---|
| `TIR-CANON-001` | operands of commutative operations (`add`, `mul`, `max`, `min`, `and`, `or`, `xor`, `cmp.eq`, `cmp.ne`) ordered by `NodeId` | IEEE addition and multiplication are commutative bit for bit |
| `TIR-CANON-002` | identity views removed: reshape to the same shape, identity permutation, full slice, broadcast to the same shape | no data moves |
| `TIR-CANON-003` | cast chains: `cast[a](cast[b](x))` → `cast[a](x)` when `b` holds every value of `x`'s dtype exactly (f16 → f32 → bf16 is one rounding from exact values either way); `cast` to the operand's own dtype removed | a widening that is exact cannot change the final rounding |
| `TIR-CANON-004` | `matmul(a, permute(b))` with the last two axes swapped → `matmul[tb] a, b`; `qmatmul` when `b` is quantized | the same products and sums (SIR's `LA-TRANSPOSE-001` in tensor form) |
| `TIR-CANON-005` | `sub(x, const c)` → `add(x, const −c)` | IEEE: `x − c` equals `x + (−c)` bit for bit, signed zeros included |
| `TIR-CANON-006` | `div(x, const c)` → `mul(x, const 1/c)` when `c` is a power of two | `1/c` is exact and both round once from the same exact value |
| `TIR-CANON-007` | `select(c, x, x)` → `x`; `select` with a constant condition → the chosen operand | — |

`TIR-CANON-005` and `TIR-CANON-006` exist so that later patterns need one form only. Neither changes a bit; a rewrite that would (`div(x, 3)` to `mul(x, 1/3)`) is not canonicalization (§7).

### 3.3 `fold`: constants and weight-only subgraphs

| Rule | What | Exact? |
|---|---|---|
| `TIR-FOLD-001` | a subgraph with no parameters is evaluated by the reference interpreter (`Mode::F32`) and replaced by a `const` | exact with respect to the reference: the folded value *is* the reference value, where a kernel would only approximate it |
| `TIR-FOLD-002` | symbolic dims and slice bounds simplified as polynomials; `slice[start=0, len=S]` of an axis of size `S` becomes an identity view (then `TIR-CANON-002`) | exact |
| `TIR-FOLD-003` | a subgraph whose only inputs are weights (`cast[f32]` of a norm weight, Gemma's `1 + w`) is moved into the relayout plan and evaluated when `.skyweights` is built, if the result is no larger than twice its inputs | exact: the same f32 arithmetic, done once; GGUF's Gemma files already store `1 + w` (R9 §3.1) |

RoPE's frequency tables are not folded by this pass: they are attributes of the `rope` composite, computed once when the composite is decomposed or a kernel is generated ([Part VII](part07-tensor-ir.md) §5.13.2).

### 3.4 `cse`

Interning makes structurally equal nodes one node at construction ([Part VII](part07-tensor-ir.md) §3.2). After a pass rewrites, `Graph::rebuild` re-interns and restores the invariant. The pass explains a merge only when it joins nodes from two different source sites, which is something the user wrote twice:

```text
note: [TIR-CSE-001] `rms_norm(h, w)` is computed once for its two uses
  = note: applied by the `cse` pass
  = note: because both compute the same function of the same values (lines 12 and 15)
```

### 3.5 `views`: index functions

| Rule | What |
|---|---|
| `TIR-VIEW-001` | a chain of views becomes one view: `reshape∘reshape`, `permute∘permute`, `slice∘slice`, and mixed chains whose composed index function ([Part VII](part07-tensor-ir.md) §4.5) is again a single view |
| `TIR-VIEW-002` | views move toward the sources, through casts, unary operations and binary operations whose operands carry the same view: `cast(permute(x))` → `permute(cast(x))`. They stop at reductions, products, composites and state |
| `TIR-VIEW-003` | a view of a parameter or state whose index function is affine is marked zero-copy: it becomes an offset and strides for the consuming kernel |
| `TIR-VIEW-004` | a view whose index function keeps a `//` or `%` after simplification is reported, because it will cost integer division in every kernel that reads it |

All four are exact: views move no data. `TIR-VIEW-004` is a report, not a rewrite. It exists because tinygrad's quantized GEMV kept a run-time division to select a nibble, one cause of its 26% of peak bandwidth on Q4_0 (R2 §7.3, §8.4); a reported division is a defect to fix in the builder or the simplifier, not a cost to accept.

### 3.6 `shapes`: constraints, guards, index width

| Rule | What |
|---|---|
| `TIR-SHAPE-001` | a `require` or bounds condition proven for every run of a plan (by Part VII §4.3's `prove_le`); no run-time check |
| `TIR-SHAPE-002` | a condition the bounds do not prove becomes a guard of the plan, checked on the CPU before each step |
| `TIR-SHAPE-003` | element offsets proven to fit in `i32`, or marked 64-bit (V22) |
| `TIR-SHAPE-004` | divisibility facts recorded for Part IX: for Qwen3-32B, `K = 5120` is a multiple of the 64-element group and of the 256-element k-quant super-block (5120 = 20 × 256), and `Dh = 128` of every vector width |
| `TIR-SHAPE-005` | gather indices proven in range from `int_range` (token ids below the vocabulary size) |

The same condition can be proven in one plan and guarded in another (§8: `pos + T <= C` holds for every decode step but needs a check for a 512-token prefill chunk).

### 3.7 `recognize`: composites from primitives

The builders emit composites directly; Sky source writes them in primitives, and this pass finds them. A pattern matches the composite's reference decomposition ([Part VII](part07-tensor-ir.md) §5.13), casts included.

| Rule | Recognizes | Class |
|---|---|---|
| `TIR-NORM-001` | `x * rsqrt(mean(x*x) + eps) * w`, mean over the weight's axis, with the decomposition's casts | exact |
| `TIR-NORM-002` | `x / sqrt(mean(x*x) + eps) * w` | under contract: `x / sqrt(m)` and `x * rsqrt(m)` round differently; recorded |
| `TIR-ROPE-001` | the rotate-half (or interleaved) pattern with cos/sin of `positions × table`, when the table equals the composite's table bit for bit | exact |
| `TIR-SOFTMAX-001` | max, subtract, exp, sum, divide over one axis | exact |
| `TIR-ATTN-001` | scaled `q·kᵀ`, mask by `select` with −inf, softmax over keys, product with `v`, with the GQA reshapes | exact |
| `TIR-GLU-001` | `silu(g) * u` and `gelu(g) * u` (with an optional cast) | exact |
| `TIR-GLU-002` | `g / (1 + exp(−g)) * u` as `swiglu` | under contract: `silu` is defined as `g · sigmoid(g)`; recorded |

When a pattern's root matches but a condition fails, the pass explains why at `--explain=all` and nothing changes. Near misses are where users lose performance without knowing it:

```text
note: [TIR-NORM-001] not recognized as `rms_norm`: the mean is over T, but the weight's axis is D
```

Exactness is by construction: a composite *is* its decomposition, so replacing the matched primitives by the composite changes nothing until a kernel implements it, and kernels are held to the composite's tolerance class ([Part VII](part07-tensor-ir.md) §11.2).

### 3.8 `horizontal`: shared-input products, and the relayout plan

| Rule | Rewrite | Legal when |
|---|---|---|
| `TIR-HCAT-001` | `qmatmul(x, W₁) … qmatmul(x, Wₙ)` → one `qmatmul(x, W)` with `W` the row concatenation, followed by `slice` views of the output | the same activation node `x`; every `Wᵢ` a weight parameter (or a weight view) with equal `QFormat`, equal `K` and the blocked axis `K` (QT3); equal `acc` and `out`; the decision is the same in every plan |
| `TIR-HCAT-002` | `swiglu(qmatmul(x, W_g), qmatmul(x, W_u))` → `swiglu` of the two halves of one `qmatmul(x, W_gu)`, with `W_gu` interleaving row tiles of `W_g` and `W_u` | as above, and `swiglu` is the only consumer of both products |
| `TIR-HCAT-003` | (refusal) the products share `x` but not a format | — |

Both rewrites are exact: each output element is the same dot product over `K` as before. One condition sits outside TIR: the GEMV skeleton's summation order for a row must depend only on `K` and the kernel configuration, not on `N`. [Part XVI](part16-decode.md) states that requirement for the verify GEMV, and a test (VIII-T5) runs the concatenated and the separate products through the same skeleton and compares bits.

**The relayout plan.** A weight produced by concatenation or interleaving does not exist in the checkpoint. The pass records how to build it:

```rust
// crates/sky-tir/src/plan/relayout.rs
pub struct RelayoutStep {
    pub new_param: Arc<str>,                  // "w.self_attn.qkv_proj"
    pub recipe: RecipeId,                     // RL3 (concatenate rows) or RL4 (interleave row tiles), Part XII §5.3
    pub parts: Vec<(Arc<str>, Range<u64>)>,   // ("w.self_attn.q_proj", 0..8192), ("…k_proj", 8192..9216), ("…v_proj", 9216..10240)
    pub tile: Option<u32>,                    // RL4: rows per tile, set by the consuming skeleton (Part IX)
}
pub struct RelayoutPlan { pub steps: Vec<RelayoutStep>, pub id: [u8; 32] }   // id: sha256 of the steps
```

The `.skyweights` builder of [Part XII](part12-quantization.md) §5.4 executes the plan (its step 2, "the graph optimizer decides which tensors to concatenate or interleave"), and the plan's id is part of the cache key. The same-quantization test proves the dequantized values unchanged. The layer graph is per layer *type*, so 64 layers produce 64 copies of each step, one per layer path. For an MLX file each step rewrites bytes it already copies for alignment (RL1); for a GGUF file, which is otherwise mapped directly, horizontal fusion is what would route it through the relayout cache, so for GGUF the pass is subject to Part XII's measurement (XII-T9), and off until that decides.

**Formats that differ by layer.** A Q4_K_M file stores `attn_v` and `ffn_down` as Q6_K in 32 of Qwen3-32B's 64 layers and Q4_K elsewhere (R8 §1.3). In those layers `TIR-HCAT-001` concatenates Q and K only and explains why V stays separate:

```text
note: [TIR-HCAT-003] `v_proj` is not concatenated with `q_proj` and `k_proj` in 32 of 64 layers
  = note: applied by the `horizontal` pass
  = note: because in those layers `v_proj` is gguf.q6_k and `q_proj`, `k_proj` are gguf.q4_k: rows of two formats cannot share one matrix
```

The builder then emits two layer graphs, one per combination of formats, and each is optimized, scheduled and tuned once. SPINE D11's "all layers share shapes" still holds; formats can differ, and the tuner sees two blocks.

**Checkpoints that are already fused.** Phi-4 ships `qkv_proj` and `gate_up_proj` as halves, and gpt-oss interleaves `gate_up` even/odd (R9 §3.1). The builder declares the existing arrangement; the pass relayouts only to the tile the skeleton wants.

**Why it pays.** At decode, Qwen3-32B's K and V products read 2,949,120 B each against Q's 23,592,960 B (arithmetic from R8 §1.2 at 4.5 bits per weight). R3 measured about 2.1–2.4 µs of fixed cost per MLX operation, about 9% of a 32B K or V GEMV's 26 µs (R3 §7.E). Small GEMVs are also occupancy-bound: MLX's separate K and V products (N = 1024, 128 threadgroups) reach only 54–63% of bandwidth on the M2 Pro, against 94–97% for the large ones (R3 §7.B); one 10,240-row product has ten times the threadgroups. Neither MLX nor llama.cpp merges Q/K/V or gate/up for files that store them separately (R3 §1.4; R5 §1.3, `ggml-metal-fusion.cpp:674-701`). Whether it pays is still Part IX's decision: the rewrite is legal, and the scheduler may split the product again if its cost model prefers.

### 3.9 `specialize`: one graph per plan

The pass copies the generic graph once per `PlanSpec` ([Part VII](part07-tensor-ir.md) §7.2), binds `T`, re-runs `fold`, `canon` and `views` on the copy, and applies:

| Rule | What | Class |
|---|---|---|
| `TIR-PLAN-001` | shapes with `T` bound; size-1 axes folded | exact |
| `TIR-PLAN-002` | with `T = 1`, the causal mask keeps every key in `[0, P + 1)` visible, so it is removed | exact |
| `TIR-PLAN-003` | each product's row count recorded as a strategy hint for Part IX: GEMV for `T = 1`, the verify GEMV for small `T`, GEMM above | a hint, no rewrite |
| `TIR-PLAN-004` | a plan with `T` bound to a range gets a guard `T` within the range | — |
| `TIR-DELTA-001` | in a labeled prefill mode, `deltanet.scan[form=recurrent]` becomes `form=chunked(c)` | under contract; breaks chunk invariance for those layers (Part VII §16) |

Specialization never moves a rounding point: it binds symbols and removes operations that are provably identities. So the plans derived from one generic graph round at the same places, which is the L1 half of SPINE D6's batch and chunk invariance (Part VII §7.3); the reduction schedules checked by V18 are the L2 half.

### 3.10 `dead`: outputs a plan does not need

| Rule | What |
|---|---|
| `TIR-DEAD-001` | results outside the plan's `OutputSet` are removed, and everything only they used |
| `TIR-DEAD-002` | a selection of rows (`slice` along the token axis) moves before the operations that compute each row on its own: elementwise operations, reductions over other axes, products whose row axis is the token axis, `gather_rows` |

Both are exact: each kept row is computed exactly as before. The selection stops at the first operation that mixes rows, which in a decoder is attention; it never crosses a state update, because the KV writes of every row are outputs.

For Qwen3-32B, a 512-token prefill that selects the last row of logits computes one row of the output head instead of 512, saving 511 × 2 × 151,936 × 5120 = 795 GFLOP, about 2.4% of the step's arithmetic, and the 311 MB f32 logits buffer (arithmetic from R8 §1.1, §1.4).

### 3.11 `layout`: weights and state, once for all plans

| Rule | What |
|---|---|
| `TIR-LAYOUT-001` | each weight gets one layout that every consuming skeleton accepts, in every plan |
| `TIR-LAYOUT-002` | each state value gets one layout for every plan |
| `TIR-LAYOUT-003` | (report) a weight for which no single layout suits all its consumers |

- **Weights.** Every skeleton declares the layouts it accepts (Part IX's `LayoutReq`): the quantized GEMV takes MLX planes at 16 KiB alignment (RL1), and the M5 `int4b` prefill GEMM wants codes in `EvenLow` order with 128-byte-aligned rows (RL5; Part XII §5.1, §6.4). A weight consumed by both the decode and the prefill plan gets one layout both accept. If none exists, the pass reports `TIR-LAYOUT-003` and the scheduler chooses between the slower codec and a second copy of the weight, which costs memory and is never the default (SPINE §3's memory target).
- **State.** Decode, verify and prefill share one KV cache, so it has one layout. The default is head-major `[Hkv, C, Dh]` per layer, in which each KV head's keys are contiguous, so decode attention can read them once for all `G` query heads ([Part XV](part15-attention.md)); MLX re-reads K and V once per query head at decode, which leaves its KV reads about 2× below bandwidth at 4K context end to end on Qwen3-8B on the M2 Pro (≈ 97 GB/s, R4), and up to 3× in R3's 8-layer microbenchmark (R3 §7.F). The attention skeleton may require token-major instead; the pass then chooses that, and the append writes it.
- **Intermediates** keep `Layout::Logical`. Which of them exist in memory at all is the scheduler's decision.

### 3.12 `legality`: what may fuse

An analysis that changes nothing. It produces the `FusionLegality` of §6 and the reports `TIR-FUSE-001`…`004` (why a pair may not fuse) and `TIR-QUANT-001` (a `dequant` that would materialize a weight).

## 4. The rewrite engine

### 4.1 Rules are data

Passes 2–8 are mostly rewrite rules. A rule is written in a small, data-only pattern language, in `crates/sky-tir/rules/*.tirr`, one file per pass:

```text
# crates/sky-tir/rules/canon.tirr
rule TIR-CANON-006 exact "division by a power of two is multiplication by its exact inverse"
  div(x, c: const) where is_pow2(c)  =>  mul(x, recip_const(c))

rule TIR-CANON-004 exact "a transposed weight is read in place"
  matmul(a, permute[swap_last2](b))                =>  matmul[tb](a, b)
  matmul(a, permute[swap_last2](b)) where quant(b) =>  qmatmul(a, b)

# crates/sky-tir/rules/composite.tirr
rule TIR-GLU-001 exact "silu(g) * u is swiglu"
  cast[t](mul(silu(g), u))  =>  swiglu[out=t](g, u)
  mul(silu(g), u)           =>  swiglu[out=f32](g, u)
```

- A pattern names operations, binds sources to variables, and may test attributes. A source list can be an exact tuple or, for commutative operations, any order, as in tinygrad's `UPat` (R2 §3.7).
- **Side conditions are named predicates**, written in Rust and registered in `crates/sky-tir/src/rewrite/preds.rs` (`is_pow2`, `quant`, `same_qformat`, `rows_independent`, …). They can ask the analyses of [Part VII](part07-tensor-ir.md) §3.4: bounds, `int_range`, value kind, index functions.
- **Right-hand sides** are operation templates, or a named Rust builder for the complicated ones (`build_qkv_concat`).
- The class (`exact`, or `contract` with its mode) is part of the rule. The driver refuses a `contract` rule unless the plan's contract allows it, and records every firing in the `ContractLog` (§7).

Keeping rules as data follows two Sky principles: rewrite rules as data that libraries can add (`docs/design/ir.md`, goal 5), and the security rule that stored artifacts carry no executable content (non-negotiable 3); the executable parts, the predicates and builders, are ordinary compiled Rust.

### 4.2 Rules are compiled, not interpreted

`crates/sky-tir/build.rs` parses the rule files and generates `rules_gen.rs`: for each pass, a `match` on the root `Op` (a fieldless enum, [Part VII](part07-tensor-ir.md) §3.1), then nested checks on the sources, with the rules that share a root merged into one decision tree. A rule that needs a source operation the graph does not contain is never tried. These are tinygrad's three PatternMatcher mechanisms (rules indexed by root op, `early_reject`, patterns compiled to code; `uop/ops.py:1595-1619`, `uop/upat.py`, R2 §3.7), done once at Sky's build time instead of at every start of a Python process.

### 4.3 The driver

```rust
// crates/sky-tir/src/rewrite/driver.rs
pub fn rewrite_to_fixpoint(g: &mut Graph, rules: &CompiledRules, cx: &mut PassCx,
                           contract: &ContractPolicy, log: &mut ContractLog) -> Changed;
```

- **Order.** A worklist in `NodeId` order, bottom-up: sources before users. A successful rewrite records the substitution and enqueues the users of the replaced node. At the end of a sweep, `Graph::rebuild` re-interns ([Part VII](part07-tensor-ir.md) §3.5).
- **Determinism.** No decision depends on hash-map iteration order or on timing, so the same input gives the same output and the same explanations on every machine and thread count. SIR's e-graph pass has the same rule: no time limit by default, so results do not depend on machine speed (R1 §5.3).
- **Termination.** Every rule must decrease a well-founded measure (node count, then a fixed rank of operations, then `NodeId`). A property test applies the rule set to random graphs and checks the measure (VIII-T2). The driver also stops after `MAX_SWEEPS` (16) and reports an internal error that names the rules still firing, the analogue of tinygrad's `REWRITE_STACK_LIMIT` (R2 §3.7).
- **Explanations.** Every firing calls `PassCx::explain(rule, span, before, after, because)`, so `sky explain` needs no side channel, exactly as in SIR (`crates/sky-ir/src/pass.rs`).

### 4.4 Observability and regression tests

- `sky explain --rewrites` prints every match with the rule, the node and the time spent, as tinygrad's `TRACK_MATCH_STATS` and `VIZ` do (R2 §3.7).
- **Rewrite replay.** `tests/tir/rewrites/*.tir` stores pairs of (input, expected output) graphs per pass, blessed like SIR's `.sir` goldens. A rule change that alters any output fails the test with a diff, which is the role of tinygrad's `PROCESS_REPLAY`.
- **Differential.** For every pass, a property test runs random graphs through the interpreter before and after the pass: bit-identical in `Mode::F32` for exact rules, within the recorded class for contract rules (VIII-T3).

### 4.5 Expected cost

tinygrad's Python rewriting costs 76–151 ms to schedule one Qwen3-0.6B block on a cache miss (R2 §3.7). The same work in Rust over interned `u32` ids is expected to be 10–100× faster [LIKELY: R2 §9.2 item 6]. Since a model has one layer graph per combination of formats (usually one), the optimizer's time should be small next to loading weights and compiling MSL [EXPERIMENTAL: VIII-M1].

## 5. When to use the e-graph

The pipeline of §3 is a sequence of rewrites, each applied where it is legal. That is enough when the useful rewrites are known and do not compete. Where they compete, and the right choice depends on cost and on sharing, an e-graph can explore the alternatives. R7's survey gives the evidence (R7 §6):
- Tensat ran equality saturation in egg with multi-pattern rules (needed to merge two products that share an input) and ILP extraction. It found graphs up to 16% faster than the state of the art, with 48× less optimization time than TASO.
- E-graphs blow up with associativity and multi-pattern rules; a 2024 study hit memory limits and needed MCTS to choose rules.
- In production, Cranelift's e-graph gained about 0.1% from keeping several versions of a value, with an average e-class size of 1.13.
- Mirage's large wins come from kernel-level search costing up to 4 hours per pattern.
- An LLM decode graph is small and repetitive, and its useful rewrites are a handful of known identities (R7 §6.3).

**So Sky uses the e-graph narrowly** [EXPERIMENTAL; SPINE §6 lists "e-graph block search" under G9+]:
- **Scope.** One layer template (Qwen3's has 32 operations, [Part VII](part07-tensor-ir.md) §9.2), saturated once and replicated, never the whole model (R7 §6.5).
- **Rules.** The block algebra of R7 §6.5: shared-input concatenation (as a multi-pattern rule), scale folding into float weights (under contract only), transposes into operand layouts, GQA packing, gate/up interleaving. These are the cases where choices compete: with mixed formats, Q+K together and V alone, or Q alone and K+V together; which norm absorbs a scale; where a permutation is cheapest.
- **Language.** `sky-egraph` gains a TIR language, `TirLang`, whose e-class analysis carries TIR types (dtype, shape, `QFormat`, value kind), so that rules can test legality. Composites stay opaque but algebraic (R7 §6.5).
- **Contracts.** `sky-egraph`'s classes map directly: `Exact` and `Conditional` rules are exact here; `Relaxed` rules are `inference`-contract rewrites and are recorded; `RealOnly` rules are not used for tensors. Nothing that changes a quantized weight's values exists in the rule set.
- **Extraction** is DAG-aware: a shared subterm is priced once, which tree-cost extraction gets wrong (today's `sky-egraph` counts "a subterm used twice … twice", R7 §6.4). At 30–60 operations an exact ILP or branch-and-bound is affordable (R7 §6.5). The cost is a roofline per candidate fusion group: bytes over measured bandwidth, FLOPs over peak, plus a dispatch cost, with device numbers from `sky-hw`. Part IX's cost model provides it, because the real cost of a graph is the cost of its kernels after fusion.
- **Default.** Off. `-O3` and `--autotune=overnight` may run it. Its plan is kept only if Part IX's model rates it faster than the deterministic pipeline's and a measurement confirms it, and it must pass the same oracle tests.

## 6. Fusion legality, and the interface to Part IX

### 6.1 The rangeify rule

Every value has an iteration domain: its shape. A consumer reads a producer through an **index function** ([Part VII](part07-tensor-ir.md) §4.5), the composition of the views between them. tinygrad decides fusion with one rule on these functions: a node fuses into its consumers when they agree on its index (`schedule/indexing.py:230-277`, R2 §5.1). TIR states the rule and its consequences as legality conditions. Whether a legal fusion is *profitable* (recompute or store, launch cost, on-chip capacity) is not decided here; tinygrad decides it with fixed heuristics (more than 3 buffers, never recompute a reduction; `schedule/rangeify.py:84-96`) that its own documentation says are not systematic (R2 §5.2, §5.3). Sky's scheduler decides it with a cost model (Part IX).

| Rule | A producer `p` may share a kernel with a consumer `c` … |
|---|---|
| F1 inline | … if `p` and the views between them contain no reduction: `p` is computed at index `f(c)` inside `c`'s loops. With several consumers this is **recomputation** unless F2 applies. |
| F2 shared | … once for all its consumers if all of them read `p` through the same index function, ignoring validity masks (rangeify's "consumers agree"). |
| F3 reduction prologue | … if `p` contains a reduction over axes `R` (the mean of `rms_norm`) and either `c`'s tile holds whole rows of `R` (row-local), or every tile recomputes the reduction (row-local recompute), always with `p`'s `ReduceSchedule`, so every tile gets bit-identical values in every plan. |
| F4 reduce, then broadcast | … if a value reduced over `R` and broadcast back over `R` (inside `softmax`, `rms_norm`) is held whole in one threadgroup; otherwise this is a kernel boundary (tinygrad's "ended ranges", R2 §5.1 rule 4). |
| F5 product prologue and epilogue | … if `p` is an elementwise or F3 producer of a product's activation operand (prologue), or `c` is an elementwise consumer indexed by the product's output index (epilogue). A row-local epilogue (`rms_norm` over `Dh`, RoPE pairs within a head) needs a tile that covers whole rows of its own axis: for Q/K norms and RoPE, an N-tile that is a multiple of 128 and aligned to heads. |
| F6 state | … if an `update` is the epilogue of the group that computes its value, writing in place; every read of the old version is in an earlier group or precedes the write in this one; an `update` is never recomputed. |
| F7 quantized weights | … only inside product and gather skeletons. A `dequant` that would materialize a weight is reported (`TIR-QUANT-001`). |
| F8 composites | … if the group takes a composite whole (attention), or its elementwise prologue or epilogue parts, never part of a composite's internal reduction. |

A pair that no rule allows is a **barrier**, reported with the rule that forbids it (`TIR-FUSE-001`…`004`: a reduction without a row-local tile, a state read after its write, a materialized dequant, a split composite).

**Multi-output groups.** A group may have several outputs, as long as each output element is produced exactly once in the group's iteration space. The Q/K/V group of a Qwen3 layer writes `q` to scratch memory and K and V into the cache; outputs that are consumed inside the group are forwarded in registers or threadgroup memory instead of being reloaded.

### 6.2 Templates

The kernel library is a set of skeletons with prologue and epilogue slots (SPINE D4). `legality` matches each skeleton's anchor (a `qmatmul`, an `attention`, a reduction, `sample`) and grows its prologue and epilogue greedily along legal edges, giving the maximal legal group around each anchor. For the Qwen3-32B decode plan:

| Group | Anchor (skeleton) | Prologue | Epilogue | Outputs | Rules |
|---|---|---|---|---|---|
| G1 | `qmatmul`, N = 10240 (quantized GEMV) | `rms_norm` of `h`, cast to bf16 | split into Q, K, V; Q/K `rms_norm` over 128; RoPE; casts; K and V `update` | `q` (scratch), K and V (state) | F3, F5, F6, multi-output |
| G2 | `attention` (flash-decode, GQA) | cast of `q` | reshape, cast to bf16 | `a` | F8 |
| G3 | `qmatmul`, N = 5120 | — | residual `add` | `h1` | F5 |
| G4 | `qmatmul`, N = 51200 interleaved | `rms_norm` of `h1`, cast | `swiglu`, cast | `m` | F3, F5 |
| G5 | `qmatmul`, N = 5120 | — | residual `add` | `h2` | F5 |

Five groups per layer, six if attention needs a separate fixed-order combine of its partial results ([Part XV](part15-attention.md)). Gate G4's target is at most six kernels per layer (SPINE §6). MLX launches about 1,225 kernels per 64-layer token, about 19 per layer (R3 §7.F), and tinygrad about 17 per layer (R2 §5.3). Fewer kernels saves dispatches, not bytes; at short context MLX already reaches 94–97% of the bandwidth ceiling, so this matters most for small draft models and long verify chains (R3 §8.3).

### 6.3 The interface

```rust
// crates/sky-tir/src/passes/legality.rs — produced here, consumed by the scheduler (Part IX)
pub struct OptimizedTir {
    pub module: Module,                                // normalized L1: one graph per plan
    pub plans: Vec<PlanSpec>,
    pub legality: BTreeMap<PlanId, FusionLegality>,
    pub layouts: LayoutPlan,                           // weights and state, shared by every plan (§3.11)
    pub relayout: RelayoutPlan,                        // built into .skyweights (§3.8)
    pub contract: ContractLog,                         // every rounding-changing rewrite (§7)
    pub explanations: Vec<Explanation>,
}

pub struct FusionLegality {
    pub edges: Vec<FuseEdge>,                          // pairs that may share a kernel
    pub barriers: Vec<Barrier>,                        // pairs that may not, with the rule (TIR-FUSE-*)
    pub templates: Vec<TemplateMatch>,                 // maximal legal groups around skeleton anchors
}

pub struct FuseEdge { pub producer: NodeId, pub consumer: NodeId, pub how: FuseHow, pub recompute: Option<RecomputeCost> }
pub enum FuseHow { Inline, Shared, RowLocal, RowLocalRecompute, Prologue, Epilogue, StateEpilogue }

/// What recomputation would add, as polynomials in the plan's symbols, per unit of the consumer's tiling.
pub struct RecomputeCost { pub extra_onchip_bytes: Poly, pub extra_flops: Poly, pub per: TileUnit }

pub struct TemplateMatch {
    pub skeleton: SkeletonKind,     // QGemv | VerifyGemv | Gemm | NaGemm | FlashDecode | FlashPrefill | Reduce | Sample | Elementwise
    pub anchor: NodeId,
    pub prologue: Vec<NodeId>,
    pub epilogue: Vec<NodeId>,
    pub outputs: Vec<NodeId>,
    pub tile_reqs: Vec<TileReq>,    // e.g. N-tile a multiple of 128 and aligned to heads
}
```

The contract between the two parts:
- **The scheduler may fuse only along `edges` and within `templates`.** The kernel-IR verifier in `sky-kernel` checks that every fused group is a union of legal edges, so a scheduler bug cannot produce an illegal kernel.
- **The scheduler decides everything about profitability:** which legal groups to form, recompute or store (with `RecomputeCost` as input), split or keep concatenated products, launch order and barriers. An example of a decision left to it: the G1 prologue recomputes `rms_norm`'s sum of squares in every threadgroup, reading the 5120 values of `h` (20,480 B in f32) from on-chip caches each time, against a separate small kernel that costs one more dispatch. Part IX decides with its cost model and the autotuner's block-level search (SPINE D11 (b)).
- **The scheduler guarantees back:** the `ReduceSchedule`s that V18 compares; write-after-read order for state; each `update` executed once per element; no floating-point atomics under `inference` (SPINE D6: run-to-run bit reproducibility).

## 7. Exact rewrites and rewrites under contract

| Rewrite | Rule | Class | Notes |
|---|---|---|---|
| canonical operand order, identity views, cast chains, transposed operands | `TIR-CANON-*` | exact | §3.2 |
| constant folding; weight-only subgraphs into the relayout plan | `TIR-FOLD-*` | exact with respect to the reference | §3.3 |
| merging equal nodes | `TIR-CSE-001` | exact | |
| view composition and motion; zero-copy views; GQA packing (a reshape) | `TIR-VIEW-*` | exact | views move no data |
| recognizing a composite's exact decomposition | `TIR-NORM-001`, `TIR-ROPE-001`, `TIR-SOFTMAX-001`, `TIR-ATTN-001`, `TIR-GLU-001` | exact | a composite *is* its decomposition |
| Q/K/V concatenation; gate/up interleaving | `TIR-HCAT-001`, `TIR-HCAT-002` | exact | given row order over K independent of N (§3.8) |
| specialization; dead outputs; row selection moved upward | `TIR-PLAN-*`, `TIR-DEAD-*` | exact | |
| weight and state layouts | `TIR-LAYOUT-*` | exact | lossless recipes only |
| `x / sqrt(m)` computed as `x * rsqrt(m)` | `TIR-NORM-002` | contract | last-bit differences |
| `g / (1 + exp(−g))` computed as `g · sigmoid(g)` | `TIR-GLU-002` | contract | |
| softmax scale folded into the f32 copy of a float norm weight | `TIR-SCALE-001` | contract | rounds `w·s` once more |
| an RMS norm weight folded into the next product's **float** weights | `TIR-SCALE-002` | contract | `(x·γ)@W → x@(diag(γ)·W)`; never for quantized `W` |
| `exp` as `exp2` with a `log2(e)` prescale | `TIR-EXP2-001` | contract | common in attention kernels |
| an intermediate kept in f16 or bf16 instead of f32 | `TIR-NARROW-001` | contract, labeled precision mode only | changes values, not only rounding order |
| chunked DeltaNet scan in prefill | `TIR-DELTA-001` | contract, labeled mode only | gives up chunk invariance for those layers |
| reordering a sum (split-K, trees, online softmax) | — | contract | a kernel choice, recorded as a `ReduceSchedule`, not a graph rewrite |
| folding anything into quantized weights; a new group size; quantizing activations or KV outside a labeled mode | — | **forbidden** | changes the model's values (SPINE D5; QT5, QT7) |

Every contract rewrite is recorded:

```rust
pub struct ContractEntry { pub rule: RuleId, pub sites: Vec<Origin>, pub mode: Option<LabeledMode>, pub detail: String }
pub struct ContractLog { pub entries: Vec<ContractEntry> }   // printed by `sky explain`; hashed into the plan id
```

The `ContractLog` is part of the plan's identity. Together with the device, the Sky build and the tuning entry, it fixes every bit of the output: SPINE D6's "run-to-run bit-reproducible on the same chip, OS, Sky build and tuning entry". For Qwen3, from either front door, the default pipeline records no contract rewrite: every rewrite it applies is exact.

## 8. `sky explain` on a Qwen3 layer

The optimizer's part of [Part XXIII](part23-benchmarks.md)'s `sky explain`, for Qwen3-32B with MLX 4-bit weights, using the Sky-source front door of [Part VII](part07-tensor-ir.md) §13.9 (line numbers refer to that listing), for the decode plan and a 512-token prefill plan. It is illustrative output of this design: the byte and FLOP figures are computed from R8's shapes, nothing in it is measured.

```text
$ sky explain hf://mlx-community/Qwen3-32B-4bit --arch models/qwen3.sky --stage optimizer --plans decode,prefill512
model: models/qwen3.sky on hf://mlx-community/Qwen3-32B-4bit; 64 layers; weights mlx.affine(b4,g64,bf16), 4.50 bits per weight
sizes: V = 151936, D = 5120, F = 25600, Hkv = 8, G = 8, Dh = 128, L = 64, C = 40960
plans: decode (T = 1), prefill512 (T in 1..=512); run-time: pos in 0..40960
numerics: inference; contract rewrites: none

models/qwen3.sky:40:12: note: [TIR-NORM-001] `x * rsqrt(mean(x * x, over: D) + eps) * w` is recognized as `rms_norm(x, w, eps=1e-6)`
   |
40 |     return x * rsqrt(mean(x * x, over: D) + eps) * w
   |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   = note: applied by the `recognize` pass
   = note: because the operations are exactly the decomposition of `rms_norm`: a mean of squares over D accumulated in f32, `rsqrt`, then the two products in this order; exact
   = note: because it is called at lines 49, 61 and 74, with D = 5120

models/qwen3.sky:43:12: note: [TIR-NORM-001] `x * rsqrt(mean(x * x, over: Dh) + eps) * w` is recognized as `rms_norm(x, w, eps=1e-6)` over Dh
   |
43 |     return x * rsqrt(mean(x * x, over: Dh) + eps) * w
   |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   = note: applied by the `recognize` pass
   = note: because the axis averaged over is the weight's axis, Dh = 128; called at lines 53 and 54, for 64 query heads and 8 key heads

models/qwen3.sky:50:17: note: [TIR-HCAT-001] `x @ a.q_projᵀ`, `x @ a.k_projᵀ` and `x @ a.v_projᵀ` become one product with a 10240-row weight
   |
50 |     q = reshape(x @ a.q_projᵀ, T, Hkv * G, Dh)
   |                 ^^^^^^^^^^^^^ 8192 rows
51 |     k = reshape(x @ a.k_projᵀ, T, Hkv, Dh)
   |                 ------------- 1024 rows
52 |     v = reshape(x @ a.v_projᵀ, T, Hkv, Dh)
   |                 ------------- 1024 rows
   = note: applied by the `horizontal` pass
   = note: because the three products read the same `x` (line 49), and their weights have one format, mlx.affine(b4,g64,bf16), and one K = 5120
   = note: because concatenating rows keeps every group of 64 whole: relayout recipe RL3 (Part XII §5.3), applied once when .skyweights is built; the same-quantization test proves the values unchanged
   = note: because each output row is the same sum over K as before; exact, given a GEMV whose order over K does not depend on N (Part XVI)
   = note: 29,491,200 bytes per layer either way; 3 products become 1

models/qwen3.sky:63:32: note: [TIR-GLU-001] `silu(x2 @ m.gate_projᵀ) * (x2 @ m.up_projᵀ)` rounded to bf16 is recognized as `swiglu(g, u, out=bf16)`
   |
63 |     gu: Tensor[T, F] in bf16 = silu(x2 @ m.gate_projᵀ) * (x2 @ m.up_projᵀ)
   |                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   = note: applied by the `recognize` pass
   = note: because `silu(g) * u`, then the binding's `in bf16`, is the decomposition of `swiglu`; exact

models/qwen3.sky:63:37: note: [TIR-HCAT-002] `x2 @ m.gate_projᵀ` and `x2 @ m.up_projᵀ` become one product with interleaved rows
   |
63 |     gu: Tensor[T, F] in bf16 = silu(x2 @ m.gate_projᵀ) * (x2 @ m.up_projᵀ)
   |                                     ^^^^^^^^^^^^^^^^^
   = note: applied by the `horizontal` pass
   = note: because both read `x2` (line 61), have one format and K = 5120, and are consumed only by `swiglu`
   = note: because interleaving row tiles of the two weights (recipe RL4, tile set by the GEMV skeleton) puts gate row i and up row i in one threadgroup, so `swiglu` can run in the product's epilogue
   = note: 147,456,000 bytes per layer either way; 2 products become 1; exact (a layout change)

models/qwen3.sky:58:9: note: [TIR-PLAN-002] in plan `decode`, the causal mask of `attention` is dropped
   |
58 |     o = attention(qh, kv.k[:, : pos + T], kv.v[:, : pos + T], positions: qpos)
   |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   = note: applied by the `specialize` pass
   = note: because with T = 1 the only query is at position pos, and every key in 0..pos + 1 is visible to it; exact

models/qwen3.sky:58:23: note: [TIR-VIEW-003] `kv.k[:, : pos + T]` is read in place
   |
58 |     o = attention(qh, kv.k[:, : pos + T], kv.v[:, : pos + T], positions: qpos)
   |                       ^^^^^^^^^^^^^^^^^^
   = note: applied by the `views` pass
   = note: because the slice of the cache is an offset and strides (head stride C·Dh = 5,242,880 elements); attention reads it with no copy

models/qwen3.sky:69:5: note: [TIR-SHAPE-001] in plan `decode`, `pos + T <= C` always holds
   |
69 |     requires pos + T <= C
   |     ^^^^^^^^^^^^^^^^^^^^^
   = note: applied by the `shapes` pass
   = note: because pos is at most 40959 and T is 1, so pos + T is at most 40960 = C; no check is made at run time

models/qwen3.sky:69:5: note: [TIR-SHAPE-002] in plan `prefill512`, `pos + T <= C` is checked before each step
   |
69 |     requires pos + T <= C
   |     ^^^^^^^^^^^^^^^^^^^^^
   = note: applied by the `shapes` pass
   = note: because pos can be up to 40959 and T up to 512, which the bounds alone do not exclude

models/qwen3.sky:75:12: note: [TIR-DEAD-002] in plan `prefill512`, `(x @ weights.lm_headᵀ)[T - 1]` computes 1 row of logits instead of 512
   |
75 |     return (x @ weights.lm_headᵀ)[T - 1], cache
   |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   = note: applied by the `dead` pass
   = note: because `[T - 1]` keeps one row, and the norm, the cast and the product with `lm_head` compute each row on its own, so the selection moves before them (line 74)
   = note: saves 511 × 1.56 GFLOP = 795 GFLOP, about 2.4% of the step's arithmetic, and a 311 MB f32 logits buffer

models/qwen3.sky:33:5: note: [TIR-LAYOUT-002] the KV cache is laid out head-major, `[Hkv, C, Dh]`, in every plan
   |
33 |     k: Tensor[Hkv, C, Dh] in f16
   |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   = note: applied by the `layout` pass
   = note: because decode attention reads each KV head's keys once for its G = 8 query heads, and they are contiguous in this layout (Part XV)
   = note: because prefill, verify and decode share one cache, so one layout serves every plan

optimizer summary, per layer (all 64 layers have one format, so one layer graph):
                                   before    decode    prefill512
  operations (views excluded)          23        20            20
  quantized products                    7         4             4    q, k, v merged; gate, up merged
  candidate kernel groups               —         5             5    Part IX decides; G4's target is at most 6
  weight bytes per layer      274,268,160 B, unchanged by every rewrite
  weight bytes per token   17,990,737,920 B with the output head
relayout plan: 128 steps (64 × RL3 qkv_proj, 64 × RL4 gate_up_proj); built into the .skyweights cache
```

What a reader can check against the model:
- Each rule names the source it changed, and the `because` lines give the facts it relied on, as SIR's explanations do (`docs/design/ir.md`, "Explanations").
- The byte totals agree with SPINE §2's 17.99 GB per token; horizontal fusion changes kernels, not bytes.
- The plan-specific notes (`TIR-PLAN-002`, `TIR-SHAPE-001/002`, `TIR-DEAD-002`) show why one source gives different programs for decode and prefill, and that the difference is exact.
- The kernel list, predicted against measured times, and the memory plan follow in the scheduler's and runtime's parts of the same command (SPINE D12; Parts IX, XIX, XXIII).

## 9. Implementation tasks

| ID | Task | Crate / files | Deliverable | Acceptance test |
|---|---|---|---|---|
| VIII-T1 | Pipeline and pass manager | `crates/sky-tir/src/{pass.rs, passes/mod.rs}` | `default_pipeline()`, `--emit-tir=after:<pass>` | verification after every pass in tests; goldens `tests/tir/qwen3_layer.after_<pass>.tir` |
| VIII-T2 | Rule language, build-time compiler, driver | `crates/sky-tir/{build.rs, rules/*.tirr}`, `src/rewrite/{driver,preds}.rs` | compiled matchers indexed by root op | a termination property test on random graphs; identical output for 1 and 8 threads and across runs; a fuzz target for the driver never panics |
| VIII-T3 | `canon`, `fold`, `cse`, `views` | `crates/sky-tir/src/passes/{canon,fold,cse,views}.rs` | §3.2–§3.5 | rewrite-replay goldens; interpreter results bit-identical before and after on random graphs |
| VIII-T4 | `shapes` and guards | `passes/shapes.rs` | §3.6 | the decode plan proves `P + T <= C`; the prefill plan emits a guard; V22 offsets on a large-logits fixture |
| VIII-T5 | `recognize` and `horizontal`, relayout plan | `passes/{composite,horizontal}.rs`, `plan/relayout.rs` | §3.7, §3.8 | D7: normalized L1 from `models/qwen3.sky` equals the builder's; concatenated and separate products give identical bits on the GEMV skeleton (with Part XVI); the relayout plan, executed by XII-T5, passes the same-quantization test; a Q4_K_M fixture gives `TIR-HCAT-003` and two layer graphs |
| VIII-T6 | `specialize` and `dead` | `passes/{specialize,dead}.rs` | §3.9, §3.10 | decode, verify and prefill plans from one graph; logits and KV bit-identical across plans for the same positions (with Parts XVI, XVII); the prefill output head computes one row |
| VIII-T7 | `layout` | `passes/layout.rs` | §3.11 | one KV layout for every plan; every skeleton's `LayoutReq` met; a fixture with conflicting requirements gives `TIR-LAYOUT-003` |
| VIII-T8 | `legality` and the Part IX interface | `passes/legality.rs` | `OptimizedTir`, `FusionLegality` | the Qwen3 decode plan gives groups G1–G5 of §6.2; `sky-kernel`'s verifier rejects a schedule that uses an illegal edge |
| VIII-T9 | `ContractLog` and plan identity | `plan/contract.rs` | §7 | a contract rewrite changes the plan id and appears in `sky explain`; a forbidden rewrite is rejected by V20 |
| VIII-T10 | `sky explain --stage optimizer` | `crates/sky-cli`, `crates/sky-tir/src/explain.rs` | §8 | golden `tests/tir/qwen3.explain` reproduces §8 on Qwen3-0.6B shapes |
| VIII-T11 | E-graph prototype [EXPERIMENTAL] | `crates/sky-egraph/src/tir/` | `TirLang`, the R7 §6.5 rule set, DAG-aware extraction with Part IX's cost | on the Qwen3 and Q4_K_M layer templates it reproduces the deterministic pipeline's decisions in under one second; it never changes interpreter results beyond the recorded contract |

## 10. Measurements that decide

| ID | Measurement | Decides |
|---|---|---|
| VIII-M1 | Pipeline time for Qwen3-0.6B, 8B and 32B, against tinygrad's 76–151 ms per block (R2 §3.7) | whether incremental rewriting is ever needed |
| VIII-M2 | Decode time with and without `TIR-HCAT-001/002`, Qwen3-8B on the M2 Pro and 32B on the M5 Max (needs Part IX's kernels) | whether horizontal fusion is on by default; for GGUF files, together with XII-T9 |
| VIII-M3 | `rms_norm` as a recomputed GEMV prologue against a separate kernel, groups G1 and G4 | input to Part IX's cost model |
| VIII-M4 | KV layout, head-major against token-major: decode attention bandwidth at 4K and 32K; append cost in prefill | §3.11's default |
| VIII-M5 | E-graph plans against the deterministic pipeline: modeled and measured time, compile time | whether §5 ever leaves EXPERIMENTAL |
| VIII-M6 | Prefill with and without `TIR-DEAD-002`: time and peak memory at 512 and 8K tokens | confirms §3.10's 2.4% and 311 MB |
| VIII-M7 | Count of `TIR-VIEW-004` reports over every G7 architecture | the index simplifier's quality; target zero |

## 11. Risks and negative results

- **Concatenation is exact only if kernels cooperate.** If a GEMV skeleton chose its split over K by N, the merged Q/K/V product would round differently from separate ones and `TIR-HCAT-001` would become a contract rewrite. VIII-T5 tests it; Part XVI's batch-invariance requirement already forbids N-dependent splits.
- **Relayout for GGUF.** Horizontal fusion requires new weight tensors, so for GGUF files it means a relayout cache that Part XII otherwise avoids (§3.8). It stays off for GGUF until XII-T9 and VIII-M2 show it pays.
- **Recognition is brittle.** A Sky-source variant that does not match a pattern falls back to primitives and loses its fused kernel without an error. Near-miss explanations (§3.7) and the builders' direct emission of composites limit the damage; VIII-M7-style counts over real models catch regressions.
- **Rule interaction.** Fixed-point rewriting can cycle or depend on order. The decreasing measure, the deterministic worklist and the sweep cap (§4.3) turn that into a test failure rather than a hang.
- **The e-graph may never pay.** Production evidence is about 0.1% for multi-versioning (Cranelift) and up to 16% on graph-level rewrites of convolutional models (Tensat); the useful LLM rewrites are already in §3. It stays EXPERIMENTAL and off.
- **Weight layout conflicts** between decode and prefill skeletons force a choice between a slower codec and a second copy of the weights (§3.11), and the second copy costs the memory that SPINE §3's target protects.
- **Legality is not profitability.** A legal fusion can be slower. The split makes that Part IX's problem, with its cost model and the autotuner's block search behind it (SPINE D11).
- **Negative results that shaped this part.** MLX's compiler fuses only elementwise chains, at most 11 operations deep, and mlx-lm uses it only for SwiGLU (R3 §1.4). tinygrad's fusion and recompute decisions are fixed heuristics that its maintainers call unsystematic (R2 §5.2–§5.3), and its generic fusion of dequantization into products runs its Q4_0 GEMV at 26% of peak (R2 §8.4). llama.cpp's fixed fusion table has no product epilogues or prologues and no Q/K/V or gate/up merge (R5 §1.3). Each is a design this part avoids: composites are units, legality is separate from a cost-model decision, quantized weights are typed, and fusions are rules with explanations.
