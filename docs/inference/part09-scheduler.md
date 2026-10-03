# Part IX — Global scheduler

The scheduler turns an optimized tensor program into a list of kernels. It decides:
- which TIR operations share a kernel (the fusion partition);
- which intermediate values are stored and which are recomputed;
- the order of dispatches and where barriers go;
- which specialized plans exist (decode, verify widths, prefill chunks, context buckets).

It hands each kernel to the kernel IR ([Part XI](part11-metal-backend.md)), its buffers to the memory planner ([Part XIX](part19-memory.md)) and its open parameters to the autotuner ([Part X](part10-autotuner.md)). Every decision is priced by the performance model of [Part XXII](part22-performance-model.md) and recorded with a rule ID for `sky explain`.

Labels: `[VERIFIED: source]`, `[LIKELY: reason]`, `[EXPERIMENTAL: what decides]`, `[MOONSHOT]`. Research notes are cited as (R5 §8) and live in [`research/`](research/). Byte and FLOP counts come from `python3 roofline.py --part22`.

## 0. Summary

- **Kernels form around anchors.** An anchor is a hand-written skeleton: GEMV, verify GEMV, GEMM, neural-accelerator GEMM, decode attention, prefill attention, sampling. Producers fuse into its prologue and consumers into its epilogue. Everything without an anchor becomes a generated elementwise or reduction kernel.
- **A Qwen3 layer decodes in five kernels.** That is the minimum for a dense transformer layer at batch 1: each boundary is a reduction over values that different threadgroups produced. A sixth kernel combines attention splits at longer contexts. mlx-lm uses 19 dispatches per layer and llama.cpp about 20 (R4, R5).
- **Recompute or store is a cost comparison, not a rule.** Recomputing the RMSNorm statistic in every simdgroup is free at M = 1 but makes the GEMV compute-bound at M ≥ 6. Storing per-threadgroup partials in the producer and reducing them in each consumer ([Part XVI](part16-decode.md)'s design) is cheaper at every width, so the cost function selects it. The same function rejects combining attention splits in the o_proj prologue.
- **Fusion stops at hard limits**: register pressure (detected from the compiled pipeline), 32 KB of threadgroup memory, the ALU budget per weight, occupancy, and reduction-order invariance.
- **Plans are specialized** by verify width, prefill chunk and context bucket. Only order-preserving choices may vary between specializations, so batch and chunk invariance (SPINE D6) hold by construction.

## 1. Position, inputs and outputs

```text
 Part VIII                     Part IX (this part)                                     Parts X, XI, XIX
 ─────────                     ───────────────────                                     ────────────────
 optimized TIR graph  ──┐
 (composites: rms_norm,  │      1 anchor selection      ┐
  rope, attention,       ├──►   2 partition search      │ cost: Part XXII predict()
  swiglu; QKV and        │      3 recompute vs store    │ guards: §5
  gate/up concatenated)  │      4 specialization (§10)  ┘
 strategy annotations ──┘       5 order + barriers  ◄────────── memory plan (Part XIX)
 (formats, algorithm                 │
  variants, KV format,               ▼
  speculation)                 ScheduledPlan ──► sky-tune lookup / search (Part X)
                                     │
                                     ▼
                               kernel IR per KernelGroup (Part XI) ──► MSL
```

The types live in `sky-kernel/src/schedule/` (SPINE D20: the scheduler owns profitability and calls `sky-tir`'s `passes/legality.rs`) and elsewhere in `sky-kernel/src/`:

```rust
pub struct ScheduledPlan {
    pub spec: Specialization,           // Decode { m }, Prefill { chunk }, context bucket
    pub kernels: Vec<KernelGroup>,      // in dispatch order
    pub epochs: Vec<Epoch>,             // barrier-delimited groups of kernels
    pub buffers: Vec<LogicalBuffer>,    // handed to the memory planner (§11)
    pub decisions: Vec<Decision>,       // rule ID, predicted costs, alternatives (§12)
}

pub struct KernelGroup {
    pub anchor: Anchor,                 // Gemv | VerifyGemv { m } | Gemm | NaGemm | AttnDecode
                                        // | AttnPrefill | Sample | Generated(GenKind)
    pub prologue: Vec<NodeId>,          // TIR nodes evaluated on the anchor's inputs
    pub epilogue: Vec<NodeId>,          // TIR nodes evaluated on the anchor's results
    pub outputs: Vec<BufferId>,         // more than one for multi-output groups
    pub params: ParamSet,               // skeleton parameters, from sky-tune or the default
    pub order: ReductionSchedule,       // the order-defining part of params (§8, Part X §3)
    pub cost: KernelCost,               // Part XXII §10
}
```

`ScheduledPlan` is plain data. That is what lets the plan cache store it as data and regenerate MSL on load (SPINE D19, §12).

## 2. Kernel formation

### 2.1 Anchors

| Anchor | Skeleton (Part XI) | Used for | Iteration space and tiling constraint |
|---|---|---|---|
| `Gemv` | quantized GEMV, M = 1 | decode projections, lm_head | rows × K; a simdgroup owns r rows, a threadgroup s simdgroups (R6 §B2: r = 4 with 16 values per lane streams at 183–191 GB/s) |
| `VerifyGemv { m }` | the same skeleton with M activation rows in registers | verification, M = 2…8 | identical per-row reduction order to `Gemv` (§8) |
| `Gemm` | MPP `matmul2d` on the ALUs (macOS 26+); `simdgroup_matrix` as the fallback | prefill and wide verification on M1–M4 | BM × BN × BK tiles; weights dequantized per tile |
| `NaGemm` | MPP `matmul2d` on the neural accelerators | prefill and wide verification on M5 | BM ≥ 16; tiles of 64 × 64 or larger for efficiency (R6 §A2) |
| `AttnDecode` | flash-decoding with fixed-size splits | decode and verify attention | (KV head, split); GQA-packed ([Part XV](part15-attention.md)) |
| `AttnPrefill` | flash attention, causal | prefill chunks | (query block, head) |
| `Sample` | partial reduction plus fixed-order combine | argmax, keyed Gumbel, top-k | (vocabulary block) |
| `Generated` | none; generated whole by `sky-kernel` | elementwise, row reductions, gathers | from the op's ranges |

An anchor is chosen per TIR op from its strategy annotation ([Part VIII](part08-graph-optimizer.md)): a `qmatmul` with M = 1 becomes `Gemv`, with M ≤ 8 `VerifyGemv`, and above that `Gemm` or `NaGemm` by chip (§8).

### 2.2 Prologue fusion: producers into the anchor

A producer P may be fused into the prologue of anchor A if all four conditions hold:
1. **P's output is consumed only through A's input loads.** Each element is read where A would have loaded it.
2. **P can be evaluated on A's ranges.** The index function of P's output, written in A's iteration variables, must not need values that A's threadgroup does not have. This is rangeify's consumer-range agreement (R2 §5.1), with one addition: when P contains a reduction over an axis A already streams (the RMSNorm sum of squares over the hidden axis that a GEMV reads in full), each simdgroup may *recompute* the reduction while it streams. Whether that pays is §4's decision.
3. **Every replication of P is priced.** If A has n threadgroups that each need all of P's input, P is evaluated n times, and the cost is charged to the ALU and cache terms of A.
4. **The guards of §5 pass.**

Examples:
- `rms_norm(h) → qkv_gemv`: legal. The statistic Σh² needs the whole vector, which every GEMV threadgroup reads anyway. Three placements exist, and §4.2(a) prices them:
  1. each simdgroup recomputes Σh² over the cached vector before its K loop;
  2. the kernel that wrote h stores one fp32 partial sum per threadgroup in its epilogue, and every consumer threadgroup adds the partials in index order (the default of [Part XVI](part16-decode.md) §3.5);
  3. a separate statistics kernel.

  In all three the prologue forms `x̂ = h · r · γ` while loading, with `r = rsqrt(Σh²/H + ε)`, before the matmul, so decode, verify and prefill share one formulation ([Part XVI](part16-decode.md) §3.5). Applying r after the dot product instead (`W·(γ ⊙ h) · r`) would save the pass over x in placement 1, but would round differently from prefill.
- `embedding gather + dequant → layer 0 qkv_gemv`: legal. The token id comes from the GPU-resident uniform buffer ([Part XX](part20-persistent-execution.md)), and every threadgroup dequantizes the same 2.9 KB row from cache.
- `swiglu → down_gemv`: illegal. `down` reads all 25,600 values of m in every threadgroup, but m's elements come from different threadgroups of the gate/up GEMV.

### 2.3 Epilogue fusion: consumers into the anchor

A consumer C may be fused into the epilogue of A if:
1. C's inputs are A's outputs on the same rows, plus broadcast values (weights, scalars, tables).
2. Any reduction in C runs over rows a single threadgroup owns. This may impose a tiling constraint on A.
3. Each element C writes is written by exactly one thread, in place or into a separate buffer.
4. The guards of §5 pass.

| Epilogue | Requirement | Constraint it adds |
|---|---|---|
| residual add `h += y` | the output row's thread also owns `h[row]` | none; in place on the fp32 residual buffer |
| SwiGLU `m = silu(g) · u` | g_i and u_i in the same simdgroup | gate/up rows **interleaved** in the weight layout: a relayout without requantization, allowed by SPINE D5 (R7 rule R6) |
| q-norm, k-norm (RMSNorm over a 128-wide head) | all 128 rows of a head in one threadgroup | **head-aligned tiling**: rows per threadgroup a multiple of 128 for the q and k row ranges |
| RoPE (rotate-half pairs `(i, i + 64)`) | both rows of a pair in one threadgroup | implied by head-aligned tiling |
| KV append | k and v rows written to the cache at `pos` | none beyond the above; multi-output (§2.5) |
| partial argmax or Gumbel-max over logits | per-threadgroup partial; a fixed-order combine follows | writes partials for the `Sample` combine |

### 2.4 Generated kernels

Operations with no anchor are generated whole by `sky-kernel`:
- elementwise groups (any chain of unary and binary operations with agreeing ranges);
- row reductions (a statistics kernel, softmax over a row);
- gathers and scatters.

They follow rangeify: one kernel per group of operations whose consumers agree on ranges, a new kernel wherever a reduction's result is broadcast back over the reduced axis (R2 §5.1, rule 4). In a fused decode plan almost nothing is left for them: the RMSNorm statistics kernel at verify widths M ≥ 6 (§4) and the MoE router are the main cases.

### 2.5 Multi-output fusion

A group may write several outputs when they come from disjoint row ranges of one anchor, or from one designated threadgroup:
- the QKV GEMV writes q to scratch, and k and v into the KV cache;
- layer 0's QKV GEMV also writes h₀ = embed(token), once, from threadgroup 0, because the residual path needs it;
- the sampling combine writes the token id, the next position, the stop flag and the next token's attention grid size (§10).

Multi-output groups are legal only when no output is read by another threadgroup of the same kernel.

### 2.6 What the graph optimizer must already have done

The scheduler relies on four rewrites from [Part VIII](part08-graph-optimizer.md):
- q, k and v projections concatenated into one weight;
- gate and up concatenated, with rows interleaved;
- the softmax scale 1/√d folded into the attention kernel's constant;
- GQA query heads grouped by KV head.

Without them the anchors are smaller and more numerous. MLX's separate k and v GEMVs reach only 54–63% of bandwidth on these shapes because there are too few threadgroups (R3 §7.B).

## 3. The decode plan for a Qwen3 layer

### 3.1 The five kernels

```text
 h (fp32 residual, in place)
  │
  ├─► K1  Gemv over [Wq; Wk; Wv]   prologue: r from the Σh² partials (fixed order); x̂ = h · r · γ_attn
  │        epilogue: × rsqrt(Σh²/H+ε); per q/k head: q-norm / k-norm, RoPE (table at pos); v unchanged
  │        outputs: q → scratch;  k, v → KV cache[layer][·][pos]
  │ ── barrier (RAW: q, K[pos], V[pos])
  ├─► K2  AttnDecode, one threadgroup per (KV head g, split s of b_kv positions), all G = 8 q heads of g
  │        writes normalized output (one split) or fp32 partials (O, m, l) per split
  │ ── barrier            [K2c: combine partials per head, left fold s = 0, 1, …, S−1 — only if S > 1]
  ├─► K3  Gemv over Wo             epilogue: h[row] += y[row]; one Σh² partial per threadgroup
  │ ── barrier (RAW: all of h)
  ├─► K4  Gemv over Wgu (rows interleaved g_i, u_i)  prologue: r from partials; x̂ = h · r · γ_mlp   epilogue: m_i = silu(g_i)·u_i
  │ ── barrier (RAW: all of m)
  └─► K5  Gemv over Wdown          epilogue: h[row] += y[row]; Σh² partials
     ── barrier (RAW: all of h) ──► next layer's K1
```

Per token, before layer 0: an embedding gather that writes h₀ and its Σh² partials ([Part XVI](part16-decode.md) §4). Fusing the gather into layer 0's K1 saves that dispatch, but then every K1 threadgroup must compute Σh₀² itself (placement 1). The scheduler prices both; the difference is one dispatch per token.

Per token, after the last layer:
- **H1:** final RMSNorm (prologue, from the last K5's partials), the lm_head GEMV, and an epilogue that writes per-threadgroup partial maxima of `logit + Gumbel(seed, position, token id)` (or of the logit, for greedy).
- **H2:** a fixed-order combine of those partials. It writes the token id, `pos + 1`, the stop flag and the next step's indirect-dispatch arguments into the uniform buffer the next token reads ([Part XVI](part16-decode.md), [Part XX](part20-persistent-execution.md)).

**Why five is the floor.** Each boundary is a global dependence: K2 needs every row of q and the new K/V; K3 needs all 64 heads' outputs; K4 needs all of h for its norm; K5 needs all of m; the next K1 needs all of h. Removing one needs a grid-wide synchronization inside a kernel, which costs the same as a boundary on Apple GPUs (1.1–1.3 µs measured against 1.1–1.25 µs) and has no forward-progress guarantee (R6 §A4, §B6). So the megakernel is rejected (R6; SPINE §2) and survives only as a G9 feasibility spike ([Part XX](part20-persistent-execution.md) §8).

### 3.2 Bytes, FLOPs and boundaries per kernel

One layer, decode M = 1, MLX 4-bit group-64 weights (4.5 bits per weight), fp16 KV, fp32 residual (`roofline.py --part22`):

| Kernel | Qwen3-32B weights | other DRAM | MFLOP | Qwen3-8B weights | other DRAM | MFLOP | Boundary after |
|---|---:|---:|---:|---:|---:|---:|---|
| K1 RMSNorm + QKV + q/k-norm + RoPE + KV append | 29.49 MB | 0.05 MB | 104.9 | 14.16 MB | 0.04 MB | 50.4 | dependent |
| K2 attention, context 512 | — | 2.18 MB | 16.9 | — | 2.14 MB | 8.5 | dependent |
| K2 attention, context 32,768 | — | 138.5 MB | 1,084 | — | 136.4 MB | 542 | dependent |
| K2c split combine, context 512 / 32,768 | — | 0.08 / 4.28 MB | 0.1 / 6.3 | — | 0.04 / 2.14 MB | 0.0 / 3.1 | dependent |
| K3 o_proj + residual | 23.59 MB | 0.06 MB | 83.9 | 9.44 MB | 0.04 MB | 33.6 | dependent |
| K4 RMSNorm + gate/up + SwiGLU | 147.46 MB | 0.08 MB | 524.4 | 56.62 MB | 0.05 MB | 201.4 | dependent |
| K5 down + residual | 73.73 MB | 0.09 MB | 262.1 | 28.31 MB | 0.06 MB | 100.7 | dependent |
| **Layer, weights** | **274.27 MB** | | | **108.53 MB** | | | |

Per token, Qwen3-32B: 64 layers × 274.27 MB plus the 437.6 MB lm_head = 17.99 GB, which matches SPINE §2 and R7 §0.1. Qwen3-8B: 36 × 108.53 + 350.1 = 4,257 MB, which matches R4's header count. `[VERIFIED arithmetic]`

All GEMVs run at 3.55 FLOP per byte. That is far below the ridge (122 FLOP/byte on the M5 Max neural accelerators, 29 on the M2 Pro), so each is DRAM-bound at M = 1, with the ALU budget to spare for the fused work. K4, the largest, issues 852 M lane instructions against 2.31 G available in its 278 µs on the M5 Max ([Part XXII](part22-performance-model.md) §6.1).

### 3.3 Dispatches and barriers against MLX and llama.cpp

| Engine | Dispatches per layer | Dependent boundaries per layer | Per token, Qwen3-8B | Per token, Qwen3-32B |
|---|---:|---:|---:|---:|
| mlx-lm | 19 | ≈ 12 barriers + an encoder boundary every ≈ 1.6 layers | 693 (457 barriers, 23 encoders) | ≈ 1,225 |
| llama.cpp | ≈ 20 (19 nodes + the attention reduce) | ≈ 15 | ≈ 724 (≈ 542 barriers) | ≈ 1,290 (extrapolated) |
| tinygrad | ≈ 17 kernels (0.6B) | every kernel (one barrier per ICB command) | — | ≈ 1,100 (extrapolated, R2 §9.2) |
| **Sky** | **5** (6 with K2c) | **5** (6) | **182–219** | **322–387** |

Sources: mlx-lm `[VERIFIED: R4 §7.C, exact Metal counts]`; llama.cpp `[VERIFIED: R5 §8.D2]`; tinygrad `[VERIFIED: R2 §5.3]`; Sky's counts follow from §3.1.

What the difference is worth:
- With the fill term of [Part XXII](part22-performance-model.md) §2.2, the MLX-like plan pays 3.72 ms of fixed cost per Qwen3-32B token on the M5 Max (2.82 ms fill, 0.90 ms bubbles) and Sky's plan 1.27 ms.
- At short context that is 1.03–1.08× in tokens per second ([Part XXII](part22-performance-model.md) §6.3).
- For a 0.6B draft model it is 1.5–2.0× ([Part XXII](part22-performance-model.md) §7).

`[EXPERIMENTAL: G4 measures the 8B and 0.6B on the M2 Pro; G5 the 32B on the M5 Max]`

What MLX and llama.cpp cannot fuse, and why:
- MLX's compiler fuses only elementwise operations: no reductions, matmuls, norms, RoPE or attention (R3 §1.4). Its maintainers closed a fused RoPE + KV-append request (mlx#4297, R3-issues).
- llama.cpp has a fixed table of nine fusion patterns: norm + multiply, add chains, MoE router and combine, SSM. It has no GEMV prologue or epilogue fusion and no QKV or gate/up merge (R5 §1.3). Its fusion is worth 4.5% on 8B decode (R5 §8.F).

### 3.4 K1's tiling constraint, and the alternative partition

The q/k-norm and RoPE epilogue requires each threadgroup to own whole 128-row heads. For Qwen3-32B that gives 80 threadgroups (64 q heads + 8 k + 8 v), 2 per core on a 40-core M5 Max. MLX's GEMV uses 8 rows per threadgroup (R3 §3.3) and would launch 1,280. Two tilings satisfy the constraint:
- 32 simdgroups × 4 rows (1,024 threads; R6's best rows-per-simdgroup);
- 16 simdgroups × 8 rows.

R6 measured 8 rows per simdgroup at 150–173 GB/s instead of 183–191 when each lane held 32 values, so the first is the default.

If either tiling loses more than the boundary it saves, the alternative partition **P1b** moves q/k-norm and RoPE into K2's prologue:
- K1 writes raw q, k and v.
- K2's threadgroups normalize and rotate q on load. q is recomputed per split, which is trivial.
- The threadgroup that owns the newest split also normalizes and rotates the new k, appends it, and uses it.

P1b costs nothing in kernels but complicates attention. The block-level tuner measures both ([Part X](part10-autotuner.md) §2.2). `[EXPERIMENTAL: decided at G4 on the M2 Pro and at G5 on the M5 Max]`

## 4. Recompute versus store

### 4.1 The cost function

For a producer P with consumers C₁…Cₙ, compare plan S (store: P runs as its own kernel or in another kernel's epilogue, writes its output, and the consumers read it) with plan R (recompute: P is evaluated inside each consumer). With the kernel time T(·) of [Part XXII](part22-performance-model.md) §2.1:

```text
ΔT = T_R − T_S
   = Σᵢ [ T(Cᵢ ⊕ P × rᵢ) − T(Cᵢ) ]                        recomputing P rᵢ times inside Cᵢ
   − [ T(P) + t_dep + t_fill + bytes(out_P)/BW_stream      the stored kernel, its boundary and output
       + Σᵢ bytes(out_P)/BW(Cᵢ) ]                          the consumers' reads of it

T(Cᵢ ⊕ P × rᵢ): adds rᵢ · ops(P) to Cᵢ's ALU term and rᵢ · bytes(in_P) to its cache term
               (cache bandwidth 400–490 GB/s on the M2 Pro for working sets ≤ 8 MiB, R6 §B2)
```

rᵢ is the replication factor: how many threadgroups or simdgroups of Cᵢ each need all of P.

The decision:
- recompute if ΔT < −ε;
- store if ΔT > ε;
- otherwise keep both partitions for the block-level tuner, with ε = 3% of the group's time.

The scheduler evaluates ΔT per specialization. The same producer can be recomputed in the decode plan and stored in a verify plan. Rule ID: `SCHED-RECOMP-001`.

### 4.2 Examples

**(a) RMSNorm statistics in K1 and K4.** Qwen3-32B on the M5 Max, set A of [Part XXII](part22-performance-model.md). The model prices placement 1 (recompute per simdgroup) and placement 3 (statistics kernel) directly:

| Plan | Verify width M | K1 | K4 | Statistics kernels | Per layer |
|---|---:|---:|---:|---:|---:|
| recompute (Σh² in every simdgroup) | 1 | 57.6 µs, DRAM-bound | 278.6 µs, DRAM-bound | — | **336 µs** |
| store (statistics kernel S1, S2) | 1 | 57.6 µs | 278.6 µs | 2 × (2.3 + 1.0) µs | 343 µs |
| recompute | 8 | 78.2 µs, ALU-bound | 381.6 µs, ALU-bound | — | 460 µs |
| store | 8 | 65.5 µs, ALU-bound | 318.3 µs, ALU-bound | 2 × (2.6 + 1.0) µs | **391 µs** |

At M = 1 the recomputation (65.5 M lane instructions in K4: 12,800 simdgroups × 5,120) hides under the DRAM time, and a statistics kernel would cost a boundary. At M = 8, each simdgroup recomputes eight sums: 524 M extra instructions, on a kernel that is already past its ALU budget ([Part XXII](part22-performance-model.md) §2.3). The crossover for K4 is M ≈ 5–6: `524 M + 327.7 M · M ≤ 2,288 M`.

Placement 2, Part XVI's producer partials, is cheaper than both at every width on these shapes:
- the producer (K3 or K5, 640 threadgroups of 8 rows) writes 640 fp32 partials per activation row, 2.5 KB;
- each consumer threadgroup reads and adds 640 × M of them: for K4 at M = 8, 6,400 threadgroups × 5,120 adds = 33 M instructions, about 4 µs, against 63 µs for placement 1 and 7.2 µs for placement 3;
- it adds no kernel and no second pass over x.

So the scheduler selects placement 2 by default, and placement 3 where the producer cannot write partials (the first layer after a generated kernel, or a producer whose rows are split across threadgroups differently).

One constraint makes any switch safe. Batch invariance requires that the statistic have the same bits whichever placement computes it, and at every M. So the reduction order of Σh² is a property of the TIR `rms_norm` composite, not of the fusion choice (§5, guard G7). With placement 2 that order includes the producer's partition of rows into partials (8 rows per partial here). That partition is therefore an order-changing parameter of K3 and K5, shared by decode, verify and prefill ([Part X](part10-autotuner.md) §3.2).

**(b) RoPE tables.** Computing cos and sin per pair costs 64 pairs × 72 heads per layer, which is trivial either way. Reading a table costs 512 B per position. The decision is about numerics, not time:
- a table built once in f64 on the CPU and rounded to fp32 gives the same rotation inputs on every chip, in prefill and decode, and in the reference interpreter;
- MLX computes `fast::cos/sin` in the kernel (R3-kernels §c), whose results can differ between GPU generations and OS releases.

So the scheduler stores. The table holds 512 B per position: 20 MB for Qwen3's native 40,960 positions, or one table per context bucket. Rule ID: `SCHED-RECOMP-ROPE-001`.

**(c) The attention split combine.** At S > 1 splits, K3's prologue could combine the partials itself, since every o_proj threadgroup reads all 8,192 attention outputs anyway. [Part XI](part11-metal-backend.md) §9.5 offers that variant and leaves the choice to this part. But each of K3's 640 threadgroups would read S × 33 KB of partials: 42 MB of cache traffic at S = 2, about 90 µs at the M2 Pro's 400–490 GB/s cache bandwidth. A separate combine kernel costs 2.5 µs plus a boundary. At 32K context with 512-position splits (S = 64) it would be 1.35 GB. Store wins at every S > 1, so the o_proj-prologue combine is never selected for Qwen3 shapes. It could pay only if o_proj ran on very few, wide threadgroups, which guard G5 forbids. The cheaper store is a combine inside K2 by the last threadgroup to finish:
- each split's threadgroup writes its partial, issues a device-scope fence and increments a per-head counter with an atomic read-modify-write;
- the threadgroup that sees the count reach S − 1 combines all partials in index order (deterministic whichever threadgroup is last) and resets the counter.

Nothing spins, so there is no forward-progress risk. R6 found that RMW atomics with device fences observe other threadgroups' writes, while relaxed loads do not (R6 §B6). `[EXPERIMENTAL: a visibility stress test on the M2 Pro and the M5 Max decides between in-kernel and separate combine (IX-T6)]`

**(d) SwiGLU.** With interleaved rows, g and u are never stored. The epilogue writes m directly, which saves a kernel and two 51 KB round trips per layer.

**(e) Dequantization.** Always recomputed. Storing dequantized weights would read 16/4.5 = 3.6× the bytes at decode. In prefill, recomputing per M-tile costs `c_deq · W · ⌈T/BM⌉` ALU instructions ([Part XXII](part22-performance-model.md) §8.1), against writing and re-reading an fp16 copy of 18 GB of weights per chunk.

**(f) Embedding and final-norm statistics.** Recomputed: a 2.9 KB row per layer-0 threadgroup, and Σh² per lm_head threadgroup (97 M lane instructions against an 822 µs kernel).

## 5. When to stop fusing

A fusion is refused if any guard fails. Each refusal is recorded with its rule ID.

| Guard | Test | Evidence | Rule ID |
|---|---|---|---|
| G1 registers | After compiling a candidate, `MTLComputePipelineState.maxTotalThreadsPerThreadgroup` must be at least the requested threadgroup size. Before compiling, `sky-kernel`'s live-value estimate × threads per core must fit the register file (≈ 208 KB per core on Apple7/8). | A bfloat MPP pipeline dropped to 768 threads (R6 §B1). R3's multi-row GEMV spilled at M = 4 and ran 2.2–5× slower (R3 §7.G). 8 rows per simdgroup with 32 values per lane lost 10–20% of bandwidth (R6 §B2). | `SCHED-REFUSE-REG-001` |
| G2 threadgroup memory | ≤ 32 KB per threadgroup, including epilogue staging | still 32 KB on M5 (R6 §A4) | `SCHED-REFUSE-TGMEM-001` |
| G3 bindings | ≤ 31 buffer slots and ≤ 4 KB of `setBytes` on the Metal 3 path; argument tables on Metal 4 | R6 §A4 | `SCHED-REFUSE-BIND-001` |
| G4 ALU budget | fused lane instructions ≤ the budget per weight (8.7 on both chips) × weights, unless the cost model shows the fused kernel still wins | [Part XXII](part22-performance-model.md) §2.3 | `SCHED-REFUSE-ALU-001` |
| G5 parallelism | at least two threadgroups per core, or the measured minimum from calibration C5 | Metal has no occupancy query (R6 §A4); MLX's k/v GEMVs reach 54–63% with 128 threadgroups (R3 §7.B) | `SCHED-REFUSE-OCC-001` |
| G6 cross-threadgroup reduction | a consumer reducing over values that other threadgroups produced cannot join the kernel | a global barrier costs as much as a boundary and can deadlock past 48 resident threadgroups on the M2 Pro (R6 §B6) | `SCHED-REFUSE-XTG-001` |
| G7 order invariance | a fusion may not make one row's reduction order depend on M, the chunk size T, or the bucket | SPINE D6; R8 §3.3, §3.5 | `SCHED-REFUSE-ORDER-001` |
| G8 compile cost | the plan's distinct pipelines × measured compile time stays under the level's budget ([Part X](part10-autotuner.md) §7) | tensor-op pipelines take ≈ 1–2 s to compile (R6 §A3) | `SCHED-REFUSE-COMPILE-001` |

G1 is enforced twice: by estimate during partition search, then by the compiled pipeline when the autotuner builds it. A pipeline that fails the check sends the partition back with that group split at its epilogue.

## 6. The partition algorithm

The partition is searched per **block template**, not per model. All 64 layers of Qwen3-32B have the same shapes and the same plan up to buffer offsets (SPINE D11). Layer 0 (embedding fusion) and the head (final norm, lm_head, sampling) get their own small searches.

```text
partition(block, spec):
  anchors   ← choose_anchor(op, spec) for every matmul / attention / sample op       (§2.1)
  frontier  ← singleton groups (every anchor; every other op on its own)
  loop:
    cands ← { (g₁ → g₂) legal by §2.2–§2.5 and guards G2–G8 }
    for c in cands: Δ(c) ← predict(plan with c) − predict(plan)                       (Part XXII)
    best  ← argmin Δ
    if Δ(best) > −ε: break
    apply best; record Decision { rule, Δ, alternatives within ε }
  ties     ← decisions whose runner-up was within ε → candidate partitions for Part X §2.2
  return plan, ties
```

This is XLA's priority fusion: the analytical model orders fusions by predicted saving (R7 §4a). The layer graph has 25–40 operations, so exhaustive enumeration of anchor-consistent partitions is also cheap. The test suite runs both and requires the greedy result to equal the exhaustive optimum on Qwen3, Llama 3 and Gemma 3 blocks (IX-T3).

## 7. The prefill plan

A prompt runs in chunks of T tokens from the buckets of §10. Per layer and chunk:

| Kernel | Anchor | Prologue | Epilogue | Notes |
|---|---|---|---|---|
| P0 row statistics (optional) | `Generated` reduction | — | writes T values of `rsqrt(Σh²/H+ε)` | chosen by §4's cost function |
| P1 QKV GEMM | `NaGemm` (M5) / `Gemm` (M1–M4) | γ ⊙ h; statistics recomputed if no P0 | row scale; q/k-norm per head (BN multiple of 128); RoPE from the table; KV append for T positions; q to scratch | MPP outputs are cooperative tensors; epilogues use their element iteration (R6 §A3) |
| P2 causal flash attention | `AttnPrefill` | — | — | KV blocks aligned to the decode split size b_kv (chunk invariance, R8 §3.3) |
| P3 o_proj GEMM | `NaGemm` / `Gemm` | — | residual add | |
| P4 gate/up GEMM | `NaGemm` / `Gemm` | γ ⊙ h; statistics | row scale; SwiGLU | interleaved rows, as in decode |
| P5 down GEMM | `NaGemm` / `Gemm` | — | residual add | |
| last chunk only: H1, H2 on the final row | `Gemv`, `Sample` | | | logits only for the last position unless log-probabilities are requested |

Three differences from decode:
- **The statistics often move to P0.** In a GEMM, the replication factor is the number of N-tiles: 80 for QKV with BN = 128, so recomputing Σh² costs 80 × T × H ALU instructions. For T = 512 on Qwen3-32B that is 210 M instructions (25 µs on the M5 Max's ALUs if not overlapped with the matrix work, nothing if overlapped), against about 23 µs for P0 (it reads 10.5 MB of fp32 activations) and its boundary. The margin is small, so the tuner decides.
- **Boundaries are cheap and the GEMMs are not.** A chunk of 512 tokens takes about 0.6 s on the M5 Max; 320 boundaries add about 1 ms. Prefill partitions are chosen for matrix efficiency (tile size, φ in [Part XXII](part22-performance-model.md) §8), not dispatch count.
- **Chunk size has a floor.** Every chunk streams all 18 GB of weights, so a chunk is compute-bound only above `T* = (F_matrix / BW_stream) · b / 16`: 34 tokens on the M5 Max and 9 on the M2 Pro. Below that, prefill is weight streaming. The default chunk is 512; the smallest bucket used for prompts is 64 on M5 and 16 on M1–M4.

**Chunk pipelining** is a 2-D wavefront over (chunk c, layer l): (c, l) needs (c, l − 1) and the KV of (c − 1, l). Kernels (c, l) and (c − 1, l + 1) are therefore independent and can share a barrier epoch, so one fills the other's tail. It cannot share weight streams, because they are different layers. The gain is bounded by the fill and bubble time, about 0.2% of a 512-token chunk, so it is off by default. `[EXPERIMENTAL: worth measuring only for 64–128-token chunks]`

## 8. Verify plans for M = 2 … 32

Verification runs the target model on M = γ + 1 positions (a chain) or M tree nodes ([Part XXI](part21-speculative.md)). The plan is the decode plan with three changes:
- **Anchors by width and chip:**

  | Chip | M = 2…8 | M = 9…32 |
  |---|---|---|
  | M1–M4 | `VerifyGemv { m }` on the ALUs; knee at M ≈ 6–7, about 1.16× at M = 8 by the model | `Gemm` on the ALUs; cost grows about linearly (the model gives 2.0× at 16, 3.8× at 32) |
  | M5 | `VerifyGemv { m }` | `NaGemm`: flat to M ≈ 34 by the ridge (1.03× at 16, 1.19× at 32 by the model) |

  The knees come from the ALU budget (c_deq + M ≤ 8.7 instructions per weight) and the neural-accelerator ridge (34 rows on the M5 Max at 534 GB/s) ([Part XXII](part22-performance-model.md) §2.3, §9.2).
- **Fusion choices change** where the cost function says so: the RMSNorm statistics are stored at M ≥ 6 (§4.2a).
- **Attention packs M query rows** with the G query heads of each KV head into one threadgroup per (KV head, split), so K/V are read once whatever M is. The mask is causal within the M new positions, or the tree mask with per-node positions for trees ([Part XV](part15-attention.md)). K1 appends M tentative positions; rejection truncates them ([Part XIV](part14-kv-cache.md)).

**Batch invariance by construction.** For every weight, the decode instantiation (M = 1) and each verify instantiation must have the same `ReductionSchedule`: lane-to-K mapping, K order, and the `simd_sum` tree. `VerifyGemv { m }` is the `Gemv` skeleton with M as a compile-time constant and more activation registers. The scheduler compares the schedules and refuses a plan where they differ (`SCHED-INV-001`). The test suite compares bits, row by row (IX-T7). This follows R8 §3.5's rule that "M = 1 is the M = k kernel with masked rows".

**The neural-accelerator path breaks that construction.** MPP `matmul2d` is lowered by Apple's backend, and its internal order is opaque (R6 §A2). Nothing guarantees that its per-row result equals the ALU GEMV's. So:
- **ALU-family plans** (the default) verify up to M = 8 on every chip, and plain decode and every verify width give identical bits.
- **NA-family plans** (M5, trees of 16–32 nodes) run *every* target pass on the `NaGemm` skeleton, including plain M = 1 decode, padded to the tile height. Bit-identity then holds within the family.

The cost is plain decode on a kernel that is slower at M = 1 (R6: a one-row `matmul2d` is slower than a GEMV). `[EXPERIMENTAL: IX-T8 measures the M = 1 penalty and whether matmul2d's per-row result is M-invariant. If it is M-invariant and within 1.1× of the GEMV, one family suffices]` See "Disagreement with the spine".

## 9. Dispatch order and barrier minimization

At batch 1, decode is a chain, so the order is fixed by dependence. The scheduler's job is not to add barriers beyond the true dependences, and to put independent work into the same epoch.

- **Barrier insertion.** Every kernel carries its read and write sets as byte ranges `(arena, offset, length)` from the memory plan (§11). A barrier precedes kernel k if k reads a range written in the current epoch (RAW), or writes a range read or written in it (WAR, WAW). Read-read overlap is free. This is llama.cpp's tracker (R5 §1.2), computed once at compile time instead of per token.
- **Encoding.** One compute encoder per command buffer, of the concurrent dispatch type. Barriers are `memoryBarrier(scope: .buffers)` on Metal 3, `barrierAfterEncoderStages` on Metal 4, or `setBarrier` on an ICB command; all cost the same on the GPU (R6 §B3). K steps in flight, one command buffer per step ([Part XX](part20-persistent-execution.md) §5). Never one command buffer per kernel: 14 µs of GPU time each (R6 §B3).
- **No barrier on independent commands.** tinygrad sets a barrier on every ICB command (R2 §7.5). Sky sets them only where an epoch ends.
- **Reordering.** A list scheduler hoists independent kernels, looking ahead up to 64 nodes, into the current epoch. llama.cpp's equivalent pass gains about 0 on Qwen3 decode (R5 §8.F), because nothing in a fused decode chain is independent. It matters for MoE (expert GEMVs are mutually independent), for prefill wavefronts (§7) and for the draft-and-verify round, where accept/reject bookkeeping can overlap the next round's first draft kernel.

The decode plan has five or six dependent boundaries per layer and no independent dispatches. The barrier count per token is the dispatch count, 322 to 386 for Qwen3-32B.

## 10. Specialization buckets

| Dimension | How it varies at run time | Buckets | What may differ between buckets |
|---|---|---|---|
| Position / context | `pos` is a GPU-resident scalar; K2's split count S = ⌈(pos + M) / b_kv⌉ comes from an indirect dispatch whose arguments H2 wrote at the end of the previous token (R6 §A4) | tuning buckets ≤ 1K, ≤ 4K, ≤ 16K, > 16K (llama.cpp's attention table uses the same, R5 §4) | order-preserving parameters only: GQA packing (query-head-parallel at short context, KV-head-packed at long), threadgroups per split |
| Verify width | per speculative round | M ∈ {1, 2, …, 8}; {16, 32} on M5 NA-family plans | anchor (§8); stored or recomputed statistics |
| Prefill chunk | per prompt | T ∈ {16, 32, 64, 128, 256, 512, 1024, 2048}; a prompt is the largest full chunks plus a greedy decomposition of the remainder | tiles BM, BN (order-preserving: K order fixed); P0 on or off |
| KV format | per model configuration | fp16 (default), int8 (labeled mode) | attention skeleton ([Part XIV](part14-kv-cache.md)) |

Rules:
- **The split size b_kv is fixed per (model, chip, tuning entry)**, never per bucket ([Part XV](part15-attention.md) AT-D2). [Part XI](part11-metal-backend.md) §9.5's skeleton uses 512 as its example; [Part XXII](part22-performance-model.md)'s worked example uses 256. Changing it would change the attention reduction tree, and so the logits, as the context grows. A larger b_kv halves the combine traffic at long context (3% of KV bytes at 32K with b_kv = 256, [Part XXII](part22-performance-model.md) §6.3) but leaves fewer threadgroups at short context. The tuner chooses it once, as an order-changing parameter ([Part X](part10-autotuner.md) §3).
- **Chunk invariance** requires that each GEMM row's K order not depend on T or on the tile's M, and that prefill attention blocks align with b_kv (R8 §3.3). The NA path's per-row M-invariance is the same open question as in §8.
- **Bucket choice at submission.** The CPU commits K steps ahead (one command buffer each) before it knows exactly how many will be accepted. It uses the bucket of the largest reachable position, `pos + K · (γ + 1)`. Bucket parameters are order-preserving, so any bucket gives the same bits.

Plan count for Qwen3-32B: 1 decode + 7 verify (+ 2 NA-family) + 8 prefill chunks, each with about 8 distinct pipelines, many shared across plans. That is about 100–150 pipelines. Tensor-op pipelines take about 1–2 s each to compile cold (R6 §A3), so the prefill buckets dominate first-run compile time. Mitigations: fewer prefill buckets at `-O1`/`-O2` (64, 512, 2048), and pipeline archives inside explicit `sky compile` outputs (SPINE D19).

## 11. Interface to the memory planner (Part XIX)

The scheduler gives the planner logical buffers with lifetimes in **epoch units**. A buffer read in epoch e is live until the end of e, because kernels inside an epoch may overlap.

```rust
pub struct LogicalBuffer {
    pub id: BufferId,
    pub class: BufferClass,     // Weight | Kv | State | Scratch | Io | Uniform | Table
    pub bytes: Poly,            // in bucket parameters: m, chunk, context bucket
    pub align: u32,             // 16 KiB for no-copy weights; 128 B for MPP tensors (R6 §A3)
    pub first_epoch: EpochId,
    pub last_epoch: EpochId,
    pub writers: Vec<KernelIdx>,
    pub readers: Vec<KernelIdx>,
}
```

The planner returns `MemoryPlan { arenas, placement: Map<BufferId, (ArenaId, offset)> }` ([Part XIX](part19-memory.md)). The scheduler then recomputes barriers on the concrete byte ranges and asserts that aliasing added none. If one appears, the planner reruns with those two buffers forbidden to alias.

**In-place updates.** The residual h is one fp32 buffer updated in place by K3 and K5. That is legal because each element is read and written by the thread that owns its row, and no reader of the old h follows; the scheduler checks the last-reader condition.

**The KV cache** belongs to [Part XIV](part14-kv-cache.md). The scheduler declares K1's write range (positions [pos, pos + M)) and K2's read range ([0, pos + M)) for barrier tracking.

**Decode scratch is small.** For Qwen3-32B:
- h, 20 KB;
- q, 16 KB;
- attention output, 16 KB;
- m, 51 KB;
- sampling partials, about 150 KB;
- attention partials, up to 4.3 MB at 32K with b_kv = 256, reused across layers.

The planner's real work is prefill (T × (H + 10,240 + 25,600) activations per chunk) and the arena for verify widths.

## 12. Explanations and the plan cache

Every decision carries a rule ID, the predicted costs of the chosen and the best rejected alternative, and the guard that refused anything. `sky explain --plan` prints them per block template:

```text
$ sky explain hf://mlx-community/Qwen3-32B-4bit --chip m5-max-40 --plan decode
block qwen3.layer (×64), decode m=1, context 512              predicted 539 µs/layer (set A)
 K1  Gemv[qkv 10240×5120 q4g64]  r=4 s=32 head-aligned      57.6 µs  DRAM   29.54 MB
     SCHED-PRO-001   rms_norm(attn) → prologue: r from 640 Σh² partials written by the previous K5 (+0.8 M adds;
                     per-simdgroup recompute +13 M lane ops; statistics kernel +3.3 µs)
     SCHED-EPI-001   q_norm, k_norm, rope → epilogue (head-aligned tiling, 80 threadgroups)
     SCHED-MULTI-001 outputs: q → scratch; k, v → kv[layer][pos]
 K2  AttnDecode  b_kv=256 GQA-packed (8 q heads / threadgroup)  6.8 µs  DRAM   2.18 MB
 K2c combine (S=2, left fold)                                 2.5 µs  latency
     SCHED-RECOMP-003 combine in o_proj prologue refused: +42 MB cache traffic (~90 µs) vs 3.5 µs
 K3  Gemv[o 5120×8192]  + residual (in place)                46.6 µs  DRAM   23.65 MB
 K4  Gemv[gate|up interleaved 51200×5120] + SwiGLU           278.6 µs DRAM  147.54 MB
     SCHED-REFUSE-XTG-001 swiglu → down prologue refused: down reads all of m (cross-threadgroup)
 K5  Gemv[down 5120×25600] + residual (in place)             140.5 µs DRAM   73.82 MB
 boundaries 6 × 1.0 µs bubble (kernel times above include 2.3 µs fill each)   6.0 µs
```

The plan cache stores `ScheduledPlan` as validated, bounded data: anchor and skeleton IDs with template versions, integer parameters, fusion recipes as TIR node lists, buffer placements. It never stores MSL or binaries. On load, `sky-kernel` regenerates the MSL deterministically, and Metal's OS-level shader cache supplies compiled code (SPINE D19). A plan is reused only if its key matches: model hash, TIR hash, strategy, chip fingerprint, OS build, Sky build and the tuning entries it cites.

## 13. Implementation tasks

| ID | Task | Crate / files | Deliverable | Acceptance test |
|---|---|---|---|---|
| IX-T1 | Plan data model | `sky-kernel/src/schedule/{plan.rs,buffer.rs}` | `ScheduledPlan`, `KernelGroup`, `LogicalBuffer`, `Decision`; text printer and bounded parser | Round-trip test; parser rejects oversized or malformed plans; fuzzed for 10⁶ iterations without a panic |
| IX-T2 | Anchor selection, calling the legality analysis | `sky-kernel/src/schedule/anchor.rs`; legality stays in `sky-tir/src/passes/legality.rs` ([Part VIII](part08-graph-optimizer.md) VIII-T8; SPINE D20) | §2.1–§2.5 rules, including head-aligned tiling and multi-output | Unit tests per rule; a Qwen3 block yields exactly K1–K5 + K2c and the refusals listed in §12 |
| IX-T3 | Partition search | `sky-kernel/src/schedule/partition.rs` | §6's priority fusion plus exhaustive mode for tests | Greedy equals exhaustive optimum on Qwen3, Llama 3 and Gemma 3 blocks; search under 10 ms per block |
| IX-T4 | Recompute versus store | `sky-kernel/src/schedule/recompute.rs` | §4.1 cost function | Reproduces the §4.2(a) table from `roofline.py`; switches K1/K4 statistics to stored at M = 6 for Qwen3-32B on M5 Max parameters |
| IX-T5 | Guards | `sky-kernel/src/guards.rs`, `sky-metal` pipeline check | G1–G8 | A deliberately over-fused kernel (SwiGLU fused into down) is refused by G6; a register-heavy epilogue is refused after compile by G1 |
| IX-T6 | Split combine variants | `sky-kernel` skeleton hooks | separate K2c and in-kernel last-threadgroup combine | Visibility stress test: 10⁸ combines with random split timing, zero mismatches against the separate kernel, on the M2 Pro and the M5 Max |
| IX-T7 | Invariance checker | `sky-kernel/src/schedule/invariance.rs`; `sky-llm/tests/batch_invariance.rs` | `SCHED-INV-001`; bitwise tests | For Qwen3-0.6B and 8B: logits of position p identical at M = 1…8 and across chunk sizes 16…2048; mlx-lm #1470's script as an adversarial test |
| IX-T8 | NA-family plan selection | `sky-kernel/src/schedule/family.rs`; consumes [Part XI](part11-metal-backend.md)'s `verify_mpp` probe (XI-T12) and [Part XXI](part21-speculative.md)'s XXI-T2 | Builds NA-family plans only when the probe reports per-row M-invariance and the M = 1 kernel streams at ≥ 95% of the raw read (the condition of [Part XXI](part21-speculative.md) §3) | On the M5 Max: an NA-family plan is produced exactly when the probe passes; the ALU-family plan otherwise; bitwise plain-vs-speculative test passes in both |
| IX-T9 | Barrier insertion and encoding order | `sky-kernel/src/schedule/barrier.rs` | byte-range epochs; reorder pass | Decode plan: barriers = dispatches; a constructed plan with independent kernels gets no barrier between them; a GPU race detector run (Metal validation) reports no hazards |
| IX-T10 | Plan explain and plan cache | `sky-cli` (`explain --plan`), `sky-kernel/src/schedule/cache.rs` | §12 output; data-only cache | Every kernel and decision listed with a rule ID; a cache entry with a changed OS build is not reused |

## 14. Measurements that decide

- **Per-kernel fill and the boundary cost on the M5 Max** ([Part XXII](part22-performance-model.md) C3–C5). They decide whether fusion is worth 1.03× or 1.08× on Qwen3-32B decode.
- **Head-aligned K1 (80 threadgroups) against P1b** on the M2 Pro and the M5 Max (§3.4).
- **Recomputed versus stored statistics at M = 1…8** on the M5 Max: confirms or moves the crossover at M ≈ 6.
- **In-kernel versus separate split combine:** correctness under stress, then time.
- **`matmul2d` per-row M-invariance and its M = 1 penalty** (IX-T8). This decides whether NA-family plans are needed.
- **Fusion gain layer by layer**, against llama.cpp's ≈ 64 µs per layer that its bytes do not explain (R5), on the M2 Pro at G4.

## 15. Risks and negative results

- **Fusion does not move bytes.** At M = 1 every GEMV is DRAM-bound, and fusion only removes fixed costs. The predicted gain is 1.03–1.08× on Qwen3-32B decode; the rest of the case rests on attention, small models and speculation (SPINE D18).
- **Head-aligned tiling may cost bandwidth.** 80 threadgroups on 40 cores may not keep enough bytes in flight. P1b exists for this case.
- **The last-threadgroup combine depends on memory-model behavior Apple does not document** beyond the MSL 4.1 specification. If the stress test ever fails, the separate combine kernel stays, at about 0.2 ms per Qwen3-32B token.
- **Opaque MPP lowering limits exactness** at wide verification (§8) and makes NA-path epilogues depend on cooperative-tensor element access, which is young (MLX's NAX build fixes for the macOS 27 SDK, R3-kernels §d).
- **Compile time grows with buckets.** About 100–150 pipelines per model is fine warm (15–50 ms each from the OS cache, R5 §1.6) but costly cold.
- **Negative result from llama.cpp:** its reorder pass gains about 0 on Qwen3 decode, and its fusion table 4.5% (R5 §8.F). Sky's decode gain comes from the epilogue and prologue fusions that table lacks, not from reordering.

## Disagreement with the spine

**Batch invariance across verify widths holds per kernel family, not across the ALU and neural-accelerator paths.** SPINE D6 requires that logits not depend on the verify width, and D10 sizes draft trees to k\* ≈ 32 on the M5 neural accelerators. These two hold together only if the neural-accelerator verify path and the plain M = 1 decode path give identical per-row results.
- MPP `matmul2d` is lowered opaquely by Apple's backend (R6 §A2), and nothing guarantees that.
- So exact speculation with 16–32-node trees requires an NA-family plan, in which plain decode also runs on `matmul2d`, at a cost R6 says is real (a one-row `matmul2d` is slower than a GEMV).
- The default ALU-family plan gives exactness with chains and trees of up to 8 nodes on every chip.

The SPINE's targets are unaffected. The ≥ 1.7× speculative target assumes public EAGLE-3 chains with γ = 3–7, which fit the ALU family. But D10.4's "~32 on M5 Max" should read "up to ~32 in an NA-family plan, whose plain-decode speed is measured separately (IX-T8)".

Resolved: the spine now says (D10) that byte-identity requires one kernel family for decode and verify: wide verification on the neural accelerators forces decode onto the same path, or byte-identity holds only for ALU-width verification (M ≤ ~8); measuring the accelerator GEMM at M = 1 decides. It also gives k\* as ~32 in theory and 16–24 after tile efficiency on the M5 Max.
