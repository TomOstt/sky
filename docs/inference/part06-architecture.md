# Part VI — Sky's proposed architecture

This part says what Sky becomes: which pieces are new, which stay, how they connect at compile time and at run time, and which alternatives were rejected and why. Every later design part (VII–XXI) details one box of the diagram in §10.

Evidence labels follow the report's convention: `[VERIFIED: source]`, `[LIKELY: reason]`, `[EXPERIMENTAL: what decides]`, `[MOONSHOT]`. Research notes are cited as (R1 §3) and live in [`research/`](research/).

## 1. Summary

Sky gains a **tensor stack beside its scientific stack**:
- a tensor IR (TIR);
- a graph optimizer and a global scheduler with a measured cost model;
- an autotuner;
- a kernel IR with a library of hand-designed skeletons;
- a Metal backend;
- an LLM runtime.

The scientific stack (checker, SIR, three engines, `sky-linalg`, Cranelift) stays as it is. The two meet in one place: a SIR operation, `tensor.call`, that runs a compiled tensor program. Sky source can therefore embed tensor code, and models can be written in Sky once the front end has tensor types.

Three things in the current plan change:

1. **ROADMAP principle 2 ("one semantic IR") is amended.** Tensors get a sibling IR instead of a bigger SIR. The principle's intent is kept: capabilities are passes over an IR, never a special path from syntax to machine code. Only its letter is dropped. §8.1 gives the evidence.
2. **ROADMAP Phase 5 "GPUs through MLIR" is replaced** by a native Metal backend that emits Metal Shading Language (MSL) text (§8.2).
3. **The numerics policy gains an `inference` contract** for tensor programs. It is deterministic and batch-invariant, but uses FMA, fp32 accumulation and fixed-order reductions. The scientific default (`reproducible`: f64, no FMA, the same bits everywhere) is unchanged. See [Part VII](part07-tensor-ir.md) §numerics and SPINE D6.

## 2. Sky today and Sky tomorrow

Today (details in [Part II](part02-sky-today.md)):

```text
.sky ─► sky-syntax ─► sky-ast ─► sky-types ─► sky-lower ─► SIR ─► sky-opt (13 passes)
                                    (units, shapes,              │
                                     tags, @-chains)             ├─► native: Cranelift JIT / AOT (sky-jit, sky-rt)
                                                                 ├─► SIR interpreter (sky-ir-interp)
                                                                 └─► tree-walker (sky-interp)
             all engines run array ops through sky_interp::Value ─► sky-linalg (f64, no FMA) ─► CPU
```

Tomorrow, shown at the level of crates. The full diagram with every subsystem is in §10.

```text
                    scientific Sky (unchanged)                     tensor stack (new)
 .sky ─► front end ─► SIR ─► sky-opt ─► engines ──tensor.call──►┐
                                                                ▼
 hf://org/model ─► sky-hub ─► sky-llm::arch::<family>::build ─► TIR (sky-tir) ─► graph optimizer ─► scheduler
 model.gguf ─────────────────────────────────────────────────────┘     ▲            (sky-tir)       (sky-tir +
                                                                        │                             sky-kernel)
                                                  sky-hw + sky-tune ─► cost model / autotuner ◄───────┘
                                                                                     │
                                                    kernel IR + skeletons (sky-kernel) ─► MSL (sky-metal)
                                                                                     │
                                       execution plan / .skybin ─► runtime (sky-llm + sky-metal) ─► Apple GPU
```

## 3. Principles of the tensor stack

These decide the design questions in Parts VII–XXI.

1. **The whole program is visible before it runs.** Sky has static types, and models have static structure, so tracing, fusion, memory planning and kernel choice all happen at compile time. There is no runtime laziness (R2 §5).

   Only these are specialized at run time:
   - the cache position and chunk length, as bounded scalar parameters with buckets;
   - per-device tuning entries;
   - buffer addresses.

   `[VERIFIED: R2 traces tinygrad's laziness as a cost (11–14 s warm-up for a 0.6B model); R3 measures MLX's per-primitive CPU cost at ≈ 3 µs (graph build plus encode)]`
2. **Static plans, replayed.** A decode step is the same work every token. It is planned once, as dispatch list, barriers, memory offsets and pipelines, then directly re-encoded each step with a few patched scalars (position, token, RNG counter), with K steps in flight and one command buffer per step (SPINE D9). Nothing is allocated or built per token.
3. **Hand-designed skeletons, generated fusion code.** Quantized GEMV, small-M verify GEMV, GEMM, attention and sampling are skeletons written by people and parameterized for the tuner. The code fused into their prologues and epilogues (norms, RoPE, KV append, residuals, activations) is generated. Elementwise and reduction kernels are generated whole.

   The evidence:
   - tinygrad's fully generic codegen reaches 10–43% of bandwidth on 4-bit decode (R2 §3);
   - every engine near the roofline on Apple uses hand-written kernels: MLX, llama.cpp, BaseRT (R3, R5, R7).
4. **Measure, then believe.** The cost model is a roofline estimate calibrated by measurements per chip (`sky-hw`, `sky-tune`). Every fusion and kernel choice the model makes can be checked by timing on the device.
5. **Deterministic by contract.** Kernels are batch-invariant and chunk-invariant with fixed reduction orders (hybrid DeltaNet layers are the one chunk-invariance exception, decided by VII-M7). A run is bit-reproducible on a given (chip, OS, Sky build, tuning entry). This is what makes exact speculative decoding testable byte for byte ([Part XXI](part21-speculative.md)).
6. **Every byte explained.** `sky explain` shows the plan:
   - every kernel per token;
   - its bytes, FLOPs, predicted and measured time;
   - every fusion made or refused, with a rule ID;
   - the memory plan, byte by byte.

   This extends Sky's existing `--explain` and cost contracts (R1 §5).
7. **Two front doors, one IR.** A Rust model builder comes first. Sky source comes once the language has tensor types. Both produce the same TIR, and a test checks this.

## 4. The levels of the tensor stack

The levels mirror `docs/design/ir.md` (L1 semantic, L2 strategy, L3 kernel, L4 backend). SIR implemented only L1 (R1 §5); the tensor stack implements all four.

| Level | Holds | Crate | Part |
|---|---|---|---|
| TIR graph | hash-consed SSA graph; tensor types (dtype incl. quantized blocks, symbolic shapes on `Poly` with bounded integers, views as index functions); ops; state values (KV, recurrent) | `sky-tir` | [VII](part07-tensor-ir.md) |
| TIR strategy | per op: storage format, precision, algorithm variant (GEMV / verify-GEMV / GEMM / neural-accelerator GEMM; attention algorithm); KV format; speculation strategy; weight layouts | `sky-tir` | [VIII](part08-graph-optimizer.md), [XII](part12-quantization.md) |
| Scheduled plan | kernel boundaries (fusion groups); dispatch order and barriers; memory plan (arena offsets) | `sky-tir` + `sky-kernel` | [IX](part09-scheduler.md), [XIX](part19-memory.md) |
| Kernel IR | per kernel: grid, threadgroup shape, memory spaces (device, threadgroup, simdgroup registers, cooperative tensors), tiles, loops; a skeleton plus generated prologue/epilogue; simplified index expressions | `sky-kernel` | [IX](part09-scheduler.md), [XI](part11-metal-backend.md) |
| Backend | MSL text, compiled at run time and cached; the Metal runtime | `sky-metal` | [XI](part11-metal-backend.md), [XX](part20-persistent-execution.md) |
| Oracle | the TIR reference interpreter, in f32 with f64 checks | `sky-tir` | [VII](part07-tensor-ir.md) |

## 5. Compile time: from a model to an execution plan

`sky compile` runs this pipeline ahead of time. `sky run` runs the same pipeline on first use and caches the result keyed by its inputs.

```text
 1. resolve      sky-hub: hf://org/model[:QUANT] → pinned commit + sha256; shared HF cache        (Part XIII)
 2. configure    normalize config.json (rope_scaling | rope_parameters; fail on unknown keys)     (Part XIII)
 3. build        sky-llm::arch::<family>::build(config) → TIR graph   (or: Sky source → TIR)      (Part VII)
 4. bind         weights: GGUF mapped directly; MLX safetensors → aligned relayout cache          (Parts XII, XIX)
 5. optimize     canonicalize, fold, CSE, views, composites, horizontal fusion, layouts            (Part VIII)
 6. specialize   decode program (T = 1 … k verify widths) and prefill programs (chunk buckets)     (Parts XVI, XVII)
 7. schedule     fusion partition by cost model; recompute vs store; barrier minimization          (Part IX)
 8. plan memory  lifetime intervals → arena offsets; KV, scratch, state; residency set             (Part XIX)
 9. tune         sky-tune lookup; on miss: heuristic (-O2) or search (-O3, overnight)              (Part X)
10. lower        kernel IR: skeleton + generated prologue/epilogue; index simplification           (Part XI)
11. emit         MSL text → pipelines (runtime compile; archive only in a .skybin)                 (Part XI)
12. assemble     execution plan: dispatch lists, barrier lists, argument layout, memory plan       (Part XX)
13. package      .skybin: plan + MSL + pipeline archive + tokenizer + template + config + hashes   (Part XIII)
```

Every step records what it did and why (rule IDs, cost-model numbers). `sky explain` prints that record.

## 6. Run time: one process, one GPU queue

```text
 CPU                                                     GPU (one queue, in order)
 ───                                                     ────────────────────────
 main thread: CLI / server, tokenizer, chat template
   │ prompt token ids
   ▼
 prefill: submit chunk programs ──────────────────────►  prefill kernels (GEMM / NA-GEMM, flash attention, KV append)
   │                                                      │
   ▼                                                      ▼
 decode: keep K steps in flight (1 cmd buffer each) ──►  step t: fused layer kernels ×L → lm_head → sample
   │      (patched scalars live in a GPU buffer;          writes token[t+1], pos+1, RNG counter, stop flag
   │       the GPU writes them itself)                    step t+1 reads them; a stop flag turns later steps into no-ops
   │
   │  speculation (optional): draft program → verify program (M = 1+γ) → accept/reject/rewind on GPU
   ▼
 streamer thread: reads emitted token ids from a ring buffer (shared memory), detokenizes on UTF-8 boundaries, prints/serves
 residency: residency set + heartbeat keep weights wired across idle gaps (R5 §5)
```

Per token the CPU does nothing except submit the next batch of steps and read finished tokens. Both are asynchronous. This differs from each competitor:
- MLX builds and encodes about 1,225 kernels per 64-layer token on the CPU (R3 §7).
- llama.cpp re-encodes about 724 dispatches per 8B token and waits for the GPU at the end of each token (R5 §1).
- mlx-lm's Python loop drew about 7 W of CPU during 0.6B decode (R9 §8).

The static plan is re-encoded directly each step (SPINE D9): that costs tens of µs and hides behind the GPU with K steps in flight, and ICBs are used only where the GPU must choose the command count or CPU wake-ups cost energy. Whether ICB replay would be faster than re-encoding is doubtful for dense 32B models. llama.cpp's maintainers measured no gain from indirect command buffers (#14570, R5); R3 measured 0% at real sizes for one command buffer per token. The gain is certain for small models such as speculative drafts, where the fixed cost per kernel dominates (R3: about 3.5 µs per kernel caps a 28-layer 0.6B model near 500–550 tok/s on the M2 Pro). `[EXPERIMENTAL: Part XX measures both]`

## 7. Crates and dependencies

| Crate | New / existing | Role | Depends on |
|---|---|---|---|
| `sky-tir` | new | TIR data model, verifier, text format, reference interpreter, graph passes, strategy, memory planner | `sky-types` (`Poly`), `sky-diag`, `sky-egraph` (later) |
| `sky-kernel` | new | kernel IR, skeleton library, index simplifier, fusion code generation, cost-model hooks | `sky-tir` |
| `sky-metal` | new | MSL emission, Metal runtime: device, buffers (shared / no-copy), residency, pipelines and archives, command encoding and replay, GPU timestamps | `sky-kernel`, `objc2-metal` (Part XI) |
| `sky-quant` | new | quantization formats as data: MLX affine, GGUF k-quants, MXFP4/NVFP4, int8; pack/unpack/relayout; quality tools (KLD, perplexity) | `sky-tir` |
| `sky-llm` | new | config normalization, architecture builders, safetensors/GGUF loading, relayout cache, tokenizer, chat templates, sampler config, KV/state manager, prompt cache, generation loop, speculation | `sky-tir`, `sky-metal`, `sky-quant`, `sky-hub`, `tokenizers`, `minijinja` |
| `sky-hub` | new | Hugging Face resolution and downloads, shared cache, pinning | — |
| `sky-serve` | new (late year 1) | OpenAI-compatible HTTP server | `sky-llm` |
| `sky-hw` | existing, extended | adds GPU facts: family, cores, neural accelerators, measured bandwidth, dispatch cost | — |
| `sky-tune` | existing, extended | kernel tuning DB keyed by (fingerprint, OS / Metal compiler, template version, shape bucket, format) | `sky-hw` |
| `sky-ir`, `sky-lower` | existing, extended later | the `tensor.call` op; lowering tensor-typed Sky code to TIR | `sky-tir` |
| `sky-cli` | existing, extended | `sky run hf://…`, `sky compile`, `sky bench llm`, `sky explain`, `sky profile` for models | all |

The dependency direction is one-way. The scientific crates never depend on the tensor crates except through `tensor.call`, and the tensor crates never depend on `sky-interp`'s `Value`.

## 8. Decisions and the alternatives we rejected

### 8.1 A sibling tensor IR, not an extended SIR

| Option | For | Against |
|---|---|---|
| **Sibling TIR (chosen)** | Freedom to choose dtypes, rank, memory model and invariants for tensors. No risk to scientific Sky. Fast to build. | A second IR to maintain. Principle 2 amended. |
| Extend SIR | One IR, as ROADMAP principle 2 says | SIR is f64-only, rank ≤ 2, ≤ 2^24 entries per array, with no memory model (buffers, dtypes, devices). Every array op runs through `sky_interp::Value`. Adding one op touched 23 files on an in-flight branch. Three-engine byte equality cannot hold between a GPU and an f64 oracle. `[VERIFIED: R1 §4–§6, §9]` |

The shared parts are infrastructure, not data structures:
- the pass manager and verifier style;
- the text format conventions;
- spans and explanations with rule IDs;
- `Poly` for symbolic sizes;
- `sky-diag`;
- the `egg` scaffolding;
- the tuning DB.

### 8.2 Our own stack emitting MSL; no MLIR

R7 evaluated four options for the GPU path:
- **A:** our own stack;
- **B:** lower all of Sky to MLIR;
- **C:** use selected MLIR dialects;
- **D:** use MLIR only for future non-Apple backends.

| Evidence | Label |
|---|---|
| Upstream MLIR has no Apple/Metal dialect. IREE's Apple target declares `mmaCount=0`. SPIRV-Cross maps cooperative matrices only to 8×8 float `simdgroup_matrix`. | VERIFIED (R7 §5) |
| Every competitive Apple-GPU engine emits MSL text: MLX, llama.cpp, tinygrad, BaseRT, the PyTorch MPS paths. The community Triton backend dropped its AIR path because AIR is undocumented. | VERIFIED (R7 §3) |
| The M5 neural accelerators are reachable from MSL through Metal 4 tensor ops / Metal Performance Primitives, which is the path MLX uses. | VERIFIED (R3, R7) |
| Mojo reaches the M5's matrix units only through undocumented `air` intrinsics, and still runs W4A16 at 0.85–0.96× MLX. | VERIFIED (R7 §5) |
| `melior` (Rust MLIR bindings) needs system LLVM/MLIR and is pre-1.0. | VERIFIED (R7) |

**Decision: option A for Apple.** MLIR, Triton or Mojo remain candidates only for non-Apple backends later (option D). The `melior` row in ROADMAP's "Building on existing work" table is removed for the GPU need.

### 8.3 MSL text, compiled at run time

The alternatives are:
- AIR (undocumented; R7);
- offline `metallib` (needs the Xcode Metal toolchain, absent on the development machine; R1 §10, R6);
- the private `MTLCompiler` service that tinygrad uses (R2 §1). It is private API and is not acceptable for a product Apple would review.

We emit MSL and compile with `makeLibrary(source:)`. The plan cache holds only data (D19): MSL is regenerated on load and Metal's OS shader cache supplies the compiled pipelines, so a warm start pays almost no compile cost. llama.cpp's measurements show what is at stake: 4.8 s for its first compile after installation, 15–50 ms thereafter (R5 §2). The `.skybin` carries the MSL and, where the OS allows, the compiled pipeline archive. The Metal-API binding choice (`objc2-metal`) and the Metal 4 specifics are in [Part XI](part11-metal-backend.md).

### 8.4 Hand-designed skeletons plus generated fusion, not generic codegen

| Option | Evidence |
|---|---|
| Fully generated kernels (tinygrad style) | On M2 Pro, fused Q4_0 decode reached ~10% of peak bandwidth for Qwen3-0.6B. A single Q4_0 GEMV reached 53 GB/s, against 177 GB/s for llama.cpp's hand-written kernel. BEAM search mis-ranked Q4_0 candidates. `[VERIFIED: R2 §2–§3, R5]` |
| Hand-written kernels only (MLX / llama.cpp style) | Near the bandwidth roofline on decode: MLX 94–97% of achievable, llama.cpp Q4_0 GEMV ~97% (R3, R5). But nothing is fused into the GEMV: llama.cpp has a fixed list of 9 fusion patterns, and MLX compiles only SwiGLU (R3, R5). |
| **Skeletons + generated prologue/epilogue (chosen)** | Gets the hand-written inner loop, and fuses the surrounding ops by construction. This is the Triton / ThunderKittens / Inductor-template pattern (R7 §1, §8). |

### 8.5 Static plans and replay, not per-token graph building

See §6. The data say replay will not by itself win on dense 32B decode. It removes per-token CPU work, which matters for:
- draft models;
- energy (R9 §8);
- jitter.

It is also the basis of GPU-resident sampling and speculation.

### 8.6 Weights: aligned relayout cache plus mmap, not copies

mlx-community safetensors files cannot be used zero-copy:
- their data starts at offset ≡ 6 (mod 8), so no tensor is 4-byte aligned;
- on the M2 Pro an unaligned `uint` load silently returns the rounded-down word (R9 §1–§2, VERIFIED).

MLX copies the weights with `pread` (R3). Sky instead builds an aligned, lossless relayout cache once, keyed by the source sha256, then mmaps it into no-copy buffers. GGUF is 32-byte aligned and is mapped directly, as llama.cpp does. Details are in [Part XII](part12-quantization.md) and [Part XIX](part19-memory.md).

### 8.7 CPU code generation: Cranelift stays; no LLVM now

- Cranelift's output is roughly at LLVM -O0 speed and its compiles are only 20–35% faster (R7 §11).
- CPU codegen is not on the LLM critical path: sampling moves to the GPU, and tokenization is Rust.
- Sky's CPU gaps come from IR and runtime design, not instruction selection. Boxed array access costs 148 ns per element; deferred checks and in-place reuse are missing (R10).

Roadmap A's CPU fixes are in [Part III](part03-current-performance.md). An LLVM ahead-of-time backend can be revisited later.

### 8.8 Numerics: contracts, not one global rule

| Contract | Where | Guarantees |
|---|---|---|
| `reproducible` (existing default) | scientific Sky | f64; no FMA; fixed order; same bits on every machine, engine and thread count |
| `relaxed` (existing, opt-in) | scientific Sky | may use FMA and vendor kernels |
| `inference` (new default for tensor programs) | TIR | fp32 accumulation, FMA allowed, fixed-order reductions. Bit-reproducible per (chip, OS, build, tuning entry). Batch-invariant and chunk-invariant (except hybrid DeltaNet prefill, VII-M7). −inf allowed inside kernels (masks); output finiteness checked in validation mode. |
| `portable` (proposed) | TIR | the same bits across Apple GPU generations, by restricting tile choices to a canonical reduction order. `[EXPERIMENTAL: measure the cost]` |

Rules are recorded per kernel in the execution plan, so `sky explain` can say which contract each result obeys.

**Tuning and results.** The spec guarantees today that "tuning changes speed, never results" (spec, "Hardware and tuning — Guarantees"). For tensor programs that guarantee holds only under `portable`.

Under `inference`, three tuned choices can change the last bits:
- tile shapes;
- K-splits;
- the internal order of Metal 4 `matmul2d`, which is opaque to us (R6).

So the `inference` contract states the weaker property explicitly:
- results are bit-reproducible *given* (chip, OS and Metal compiler, Sky build, tuning entry);
- every tuning candidate is tested to stay within the contract's tolerance of the reference before it can be stored;
- the plan records which entry it used.

`portable` keeps the original guarantee by allowing only order-preserving parameters. Any TIR program runs under `inference` unless `portable` is requested. Scientific f64 code is unaffected.

### 8.9 Caches and non-negotiable 3

The spec's non-negotiable 3 requires caches in a data-only format. The tensor stack keeps it:

| Artifact | Treatment |
|---|---|
| Weight relayout cache (`.skyweights`) | Data |
| Plan cache | Data: kernel choices, parameters and memory offsets, bounded and validated like the tuning DB. It never holds MSL or binaries. On load, MSL is regenerated by the deterministic generator, and Metal's own OS-level shader cache supplies the compiled code. |
| Metal binary archives | Appear only in explicit `sky compile` outputs (`.skybin`). These are build artifacts in the same class as `sky build` executables, not caches. |
| Tuning DB | Keeps its data-only text format, with keys extended for GPU kernels ([Part X](part10-autotuner.md)). |

The Hugging Face cache is made of symlinks. `sky-hub` reads it as an external store; the `sky-project` rule against symlinks governs project imports only.

## 9. Amendments to ROADMAP.md and the spec

These are proposed as separate PRs after this report is accepted.

1. **Principle 2:** add "Tensor programs use a sibling IR (TIR) that shares SIR's infrastructure and conventions; capabilities remain passes over an IR, never special paths from syntax to machine code."
2. **Phase 5:** replace "GPUs through MLIR [19, 21, 55]" with "Apple GPUs through a native Metal backend (TIR → kernel IR → MSL); MLIR, Triton or Mojo only for future non-Apple targets."
3. **"Building on existing work":**
   - drop the MLIR/`melior` row for GPUs;
   - add rows for MLX, llama.cpp, tinygrad, Metal FlashAttention and ThunderMittens. All are MIT- or Apache-licensed. Policy: learn from them, write our own kernels, and port only with notices.
4. **A new phase, "Inference on Apple silicon"**, with the gates G0–G9 from [Part XXV](part25-roadmap-12-months.md).
5. **Spec:**
   - a new section "Numerics contracts", with `inference` and `portable`;
   - tensor types and storage formats (`in f16`, `in q4(group: 64)`), edition-gated, from [Part VII](part07-tensor-ir.md);
   - a byte unit for memory contracts (`cost memory <= 20 GB`, `cost bytes_per_token <= 18.2 GB`); Sky has no byte unit today ([Part XIX](part19-memory.md));
   - the tuning-database rule. Today only an explicit `sky tune` writes the DB. `sky compile -O3` / `--autotune` must be allowed to write it too, as an explicit opt-in, with the same data-only format (`sky-tune-db 1` with new keys, which today's parser already ignores; [Part X](part10-autotuner.md)).

## 10. The architecture in one diagram

```text
 ┌──────────────────────────────────────── FRONT DOORS ─────────────────────────────────────────┐
 │  Sky source (.sky)                     Hugging Face checkpoint              GGUF file          │
 │  sky-syntax → sky-ast → sky-types      hf://org/model[:QUANT]               (mmap, 32-B        │
 │  (units, shapes, dtypes, storage       sky-hub: pin commit + sha256,         aligned)           │
 │   formats, named-axis broadcast)       shared ~/.cache/huggingface                              │
 │  sky-lower → SIR ──tensor.call──┐      config normalize → sky-llm::arch::<family>::build        │
 └─────────────────────────────────┼──────────────────────────┬───────────────────────────────────┘
                                   ▼                          ▼
 ┌──────────────────── TIR GRAPH (sky-tir) ─────────────────────────────────────────────────────┐
 │ hash-consed SSA · dtypes incl. Quant(QFormat) · symbolic dims (Poly, bounded ints) · views as  │
 │ index functions · state values (KV, DeltaNet) · composites with reference decompositions       │
 │ verifier · text format · spans · REFERENCE INTERPRETER (f32/f64 oracle)                        │
 └───────────────────────────────────────────────┬───────────────────────────────────────────────┘
                                                 ▼
 ┌──────────────────── GRAPH OPTIMIZER (Part VIII) ──────────────────────────────────────────────┐
 │ canonicalize · fold · CSE · view folding · composite recognition (rms_norm, rope, attention,   │
 │ swiglu) · horizontal fusion (QKV, gate/up) + weight relayout plan · decode/prefill             │
 │ specialization · layout assignment · e-graph block algebra (sky-egraph, later)                 │
 │ every rewrite: rule ID + explanation; contract rewrites recorded                               │
 └───────────────────────────────────────────────┬───────────────────────────────────────────────┘
                                                 ▼
 ┌──────────── STRATEGY (L2) ────────────┐   ┌──────── HARDWARE MODEL (Part XXII) ────────┐
 │ formats · precision · GEMV / verify-  │   │ sky-hw: GPU family, cores, neural accel.,  │
 │ GEMV / GEMM / NA-GEMM · attention     │   │ measured bandwidth, dispatch + barrier     │
 │ algorithm · KV format · speculation   │   │ cost · roofline per op · calibrated by      │
 └──────────────────┬────────────────────┘   │ measurements                               │
                    ▼                        └───────────────┬─────────────────────────────┘
 ┌──────────────────── GLOBAL SCHEDULER (Part IX) ──────────────────────▼────────────────────────┐
 │ fusion partition (bytes, FLOPs, launches, barriers, registers, occupancy) · recompute vs store │
 │ static MEMORY PLAN (lifetimes → arena offsets: scratch, KV, state) · dispatch order · barriers  │
 └───────────────────────────────────────────────┬───────────────────────────────────────────────┘
                                                 ▼
 ┌──────────────────── AUTOTUNER (Part X) ───────────────────────────────────────────────────────┐
 │ kernel: BEAM over skeleton parameters · block: partition search (one layer tuned, all layers   │
 │ reuse it) · model: speculation tree, KV format, chunk sizes · honest timing (DRAM-sized sets,  │
 │ GPU timestamps, medians) · sky-tune DB (fingerprint, OS/Metal compiler, template, shape, format)│
 └───────────────────────────────────────────────┬───────────────────────────────────────────────┘
                                                 ▼
 ┌──────────────────── KERNEL IR (sky-kernel, L3) ───────────────────────────────────────────────┐
 │ SKELETONS: quant GEMV · verify GEMV (M ≤ 32, batch-invariant) · GEMM via MPP matmul2d (all     │
 │ chips, macOS 26+; NAs on M5; simdgroup_matrix fallback) · flash-decode (GQA, fixed splits) ·   │
 │ flash-prefill · reduce · sample                                                                 │
 │ + GENERATED prologue/epilogue (norm, RoPE, QK-norm, KV append, residual, SwiGLU) · index simpl.│
 └───────────────────────────────────────────────┬───────────────────────────────────────────────┘
                                                 ▼
 ┌──────────────────── BACKENDS (L4) ────────────────────────────────────────────────────────────┐
 │ sky-metal: MSL text → makeLibrary(source:) → pipelines + OS shader cache; archives in .skybin  │
 │ CPU: TIR reference interpreter (oracle) · host code: SIR + Cranelift (existing)                 │
 └───────────────────────────────────────────────┬───────────────────────────────────────────────┘
                                                 ▼
 ┌──────────────────── EXECUTION PLAN / .skybin (sky compile) ───────────────────────────────────┐
 │ decode program (static dispatch + barrier list) · verify programs (M = 2…k) · prefill programs │
 │ (chunk buckets) · memory plan · MSL + pipeline archive · relayout cache ref (.skyweights) ·     │
 │ tokenizer · chat template · sampler · speculation config · hashes of every input               │
 └───────────────────────────────────────────────┬───────────────────────────────────────────────┘
                                                 ▼
 ┌──────────────────── PERSISTENT RUNTIME (sky-llm + sky-metal) ─────────────────────────────────┐
 │ GPU-resident decode loop: token / position / RNG counter / stop flag live on the GPU; K steps  │
 │ in flight; static plan directly encoded · speculation: draft (EAGLE-3, small model, prompt lookup) → verify │
 │ → accept / reject / rewind on GPU · sampling on GPU (keyed Gumbel, top-k/p, min-p, penalties)  │
 │ weights: mmapped no-copy + residency set + heartbeat · KV + prefix cache (disk, mmapped)       │
 │ streaming detokenizer (CPU thread) · OpenAI-compatible server (sky-serve)                      │
 └───────────────────────────────────────────────┬───────────────────────────────────────────────┘
                                                 ▼
 ┌──────────────────── OBSERVABILITY ────────────────────────────────────────────────────────────┐
 │ sky explain: kernels/token, bytes/token, barriers, memory plan byte by byte, rule IDs ·         │
 │ sky profile: per-dispatch GPU time vs roofline · cost contracts (bytes_per_token, memory) ·     │
 │ sky bench llm + sky-evaluation (Part XXIII)                                                    │
 └───────────────────────────────────────────────┬───────────────────────────────────────────────┘
                                                 ▼
          Apple GPU, M1–M5: shader cores · simdgroup matrices · M5 neural accelerators · unified memory
```

## 11. Implementation tasks

| ID | Task | Crate / files | Deliverable | Acceptance test |
|---|---|---|---|---|
| VI-T1 | Create the tensor-stack crates with CI and `cargo deny` | `crates/sky-{tir,kernel,metal,quant,llm,hub}` | Empty crates with the documented dependency graph | `cargo test` green; a test fails the build if a scientific crate depends on a tensor crate |
| VI-T2 | Set up a macOS CI runner for Metal tests | `.github/workflows/metal.yml`, self-hosted runner | Metal tests run on every PR | A deliberately broken kernel fails CI |
| VI-T3 | Add `tensor.call` to SIR | `crates/sky-ir/src/op.rs`, verifier, text format | SIR can call a TIR program by name | Round-trip text test; verifier rejects mismatched types |
| VI-T4 | Amend ROADMAP.md (§9 items 1–4) | `ROADMAP.md` | Merged amendment | Review by the owner |
| VI-T5 | Add a spec section on numerics contracts | `spec/v0.1.md` (or the next edition's spec) | The `inference` and `portable` contracts | Spec examples become golden tests once the code exists |
| VI-T6 | Explain-record plumbing across the tensor pipeline | `sky-tir` (passes), `sky-cli` | Every pass records rule IDs and costs | `sky explain` on a Qwen3 model lists every kernel with its reason |

## 12. Measurements that decide

- Replay versus re-encoding on the M5 Max, for the 32B model and for the 0.6B draft model ([Part XX](part20-persistent-execution.md)). This decides how much engineering the persistent runtime deserves.
- Relayout-cache cold build time against MLX's `pread` load time ([Part XVIII](part18-ttft.md)). This decides whether the cache is built on first run or by `sky compile` only.
- Fusion gains over unfused kernels, layer by layer, against the about 64 µs per layer that llama.cpp's weight bytes do not explain (R5). This decides how far fusion goes ([Part IX](part09-scheduler.md)).

## 13. Risks and negative results

- **The architecture does not by itself beat MLX on dense decode.** MLX is at 94–97% of measured bandwidth on short-context decode (R3). Hand-written runtimes such as BaseRT tie it (R7). The architecture earns its keep through:
  - speculation;
  - long-context attention;
  - time to first token and loading;
  - RAM and energy;
  - determinism;
  - reach across chips.

  It does not earn it on raw dense decode. Part XXVIII states this risk in full.
- **Two IRs cost maintenance.** Mitigation: shared infrastructure crates and conventions, and no scientific crate depends on the tensor stack.
- **The skeleton library is hand-written code that has to keep up with new chips.** Mitigation: per-family tuning tables (as llama.cpp keeps for attention, R5) and the autotuner. The M5 neural-accelerator path is fragile everywhere today: llama.cpp silently disabled it in some builds (R5), and the Metal 4.1 tensor path has undocumented behaviors (R7).
- **The runtime-compile path depends on the OS's Metal compiler.** A macOS update can change the generated code, its speed, or (rarely) its results. Tuning entries and bit-reproducibility are therefore keyed by OS and Metal compiler version (Part X).
