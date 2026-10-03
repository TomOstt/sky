# Part XX — Persistent execution and the compiled decode loop

This part specifies how Sky runs decode once the plan exists: the decode program as data, how it is replayed, how K steps stay in flight with the GPU writing its own scalars, how tokens reach the CPU, how speculation rounds run without a host sync, how a draft model runs at low overhead, and why Sky does not build a persistent megakernel. It ends with what this buys, stated with its limits, and a measurement plan for the M5 Max.

"Persistent" in Sky means that the **plan** persists on the GPU side between tokens: argument structs, pipelines, memory and, optionally, encoded command lists. It does not mean a kernel that runs forever.

It builds on the Metal backend ([Part XI](part11-metal-backend.md): kernel ABI, encoders, synchronization), the scheduler that produces the dispatch list ([Part IX](part09-scheduler.md)), the memory plan ([Part XIX](part19-memory.md)), the generation loop and sampler ([Part XIII](part13-llm-runtime.md)) and speculation ([Part XXI](part21-speculative.md)).

Evidence labels: `[VERIFIED: source]`, `[LIKELY: reason]`, `[EXPERIMENTAL: what decides]`, `[MOONSHOT]`. Research notes are cited as (R6 §B4) and ship as appendices: [R2](research/R2-tinygrad.md), [R3](research/R3-mlx-core.md), [R4](research/R4-mlx-lm.md), [R5](research/R5-llamacpp.md), [R6](research/R6-apple-metal.md) (primary), [R7](research/R7-compilers.md), [R8](research/R8-algorithms.md), [R9](research/R9-formats-methodology.md). The microbenchmarks are in [`research/microbench/`](research/microbench/). "Measured" means on the M2 Pro (16-core GPU, macOS 26.6.1) unless stated.

## 1. Summary

| # | Decision | Why |
|---|---|---|
| 1 | Decode is a **static program**: a dispatch list per step variant, immutable argument structs, and a small `StepState` buffer of per-token scalars that the **GPU itself** updates (position, token, RNG counter, verify width, stop flag). | Nothing is built or allocated per token (Part VI §3, principle 2). |
| 2 | **K steps are kept in flight.** The CPU commits step t+K when step t completes; it never sits between two steps. | When the CPU waits for one command buffer before committing the next, the GPU idles 104.6 µs; when both are committed up front, 0.4 µs (R6 §B3) `[VERIFIED]`. |
| 3 | **Replay mechanism: direct encoding of the static list is the default; ICBs are used where the GPU must choose how many commands run.** | Encoding is 0.07–0.09 µs per dispatch on the CPU, so a fused 32B step costs tens of µs to encode and hides behind a 30 ms GPU step. ICB replay costs about 4 µs of CPU for any length but no GPU time is saved (R6 §B4) `[VERIFIED]`. |
| 4 | **Position-dependent grids are indirect dispatches** whose arguments the GPU writes; **stopped steps are no-ops** (every kernel exits early; on the ICB path the execution range drops to zero). | Steps are encoded before their position is known. |
| 5 | **`MTLSharedEvent` per step + a token ring buffer in shared memory**; a CPU streamer thread detokenizes. | The CPU learns about tokens asynchronously; it never gates the GPU. |
| 6 | **A speculation round (draft → verify → accept → rewind → emit) is one static program** whose varying quantities (accepted count, positions) live in GPU state. No host sync between rounds. | mlx-lm syncs fully once per round and the GPU idles while Python rewinds and rebuilds (R4 §2.3) `[VERIFIED]`. |
| 7 | **No persistent megakernel.** It is rejected for production; only a G9 feasibility spike with explicit triggers remains (§8.4). | No forward-progress guarantee, deadlock past co-residency, stale relaxed loads, and a global barrier costs as much as a kernel boundary (R6 §A4, §B6) `[VERIFIED]`. |
| 8 | **Honest gains:** ≤ 3% for dense 32B plain decode; large for small and draft models; lower CPU energy; speculation without host syncs. | §9 |

## 2. The costs that decide the design

| Quantity | Measured | Source |
|---|---|---|
| Dependent kernel boundary (GPU) | 1.1–1.25 µs, whether serial encoder, explicit barrier or Metal 4 barrier | R6 §B3 `[VERIFIED]` |
| Independent dispatch (GPU) | 0.14 µs | R6 §B3 |
| CPU encode per dispatch, Metal 3 serial / Metal 4 without barriers | 0.081 / 0.070 µs | R6 §B3 |
| CPU encode per dispatch with explicit barriers, Metal 3 concurrent / Metal 4 | 0.233 / 0.194 µs | R6 §B3 |
| 1,000 dependent dispatches, direct serial encoding | GPU 1,399 µs, CPU 72.0 µs | R6 §B4, [`results/icb.txt`](research/microbench/results/icb.txt) |
| Same, CPU-encoded ICB with `setBarrier`, replayed | GPU 1,383 µs, CPU **3.8 µs** | R6 §B4 |
| ICB one-time encoding | 0.163 µs per command | R6 §B4 |
| GPU-encoded ICB (encode kernel + replay, 1,000-slot range) | 161.6 µs for 10 live commands; ≈ 0.15 µs per reset slot; +0.5 ms per 1,000 commands | R6 §B4 |
| One command buffer per dispatch | 13.9 µs GPU, 5.74 µs CPU each | R6 §B3 |
| Commit → completion seen by the CPU, one tiny dispatch | 106–125 µs (Metal 3); 84.8 µs median (Metal 4 commit + `signalEvent`) | R6 §B3, [`results/mtl4.txt`](research/microbench/results/mtl4.txt) |
| GPU gap between two command buffers: CPU waits then commits / both pre-committed | **104.6 µs / 0.4 µs** | R6 §B3, [`results/dispatch.txt`](research/microbench/results/dispatch.txt) |
| Device-wide barrier inside one kernel (working variants) | 1.11–1.43 µs | R6 §B6, [`results/persist.txt`](research/microbench/results/persist.txt) |
| MLX framework floor | ≈ 3.5 µs per kernel; ≈ 1,225 kernels and 4.2–4.4 ms per 64-layer token with no math | R3 §7.F |
| mlx-lm, Qwen3-0.6B 4-bit | 541 dispatches, 362 barriers, 3.51 ms per token; CPU build + encode 1.94 ms; 54.5% of achievable bandwidth | R4 §1, §7.E |
| llama.cpp, Qwen3-8B | ≈ 724 dispatches, ≈ 542 barriers per token; ≈ 64 µs per layer not explained by weight bytes | R5 §8 D2 |
| llama.cpp command replay (#14570, never merged) | "does not lead to any measurable improvement" | R5 §9.5 `[VERIFIED]` |

Two conclusions follow. First, the GPU-side cost of a dispatch does not depend on how it was submitted, so replay cannot make the GPU faster; only fewer dependent dispatches can ([Part IX](part09-scheduler.md)). Second, the expensive things are a CPU in the dependency chain (≈ 100 µs per round trip) and a CPU that builds a graph per token (MLX's ≈ 3.5 µs per kernel). Both are removed by a static program with steps pre-committed, whatever the replay mechanism.

## 3. The decode program

### 3.1 Data

The decode program is part of the plan, so it is data (SPINE D19): it is stored in the plan cache and validated on load; pipelines are rebuilt from regenerated MSL ([Part XI](part11-metal-backend.md) §8.2).

```rust
// crates/sky-llm/src/program.rs (sketch)
pub struct DecodeProgram {
    pub steps: Vec<StepTemplate>,      // [0] = plain decode (M = 1); others: verify widths, draft steps (Part XXI)
    pub args: ArgArenaPlan,            // immutable argument structs: (arena, offset) per pointer, plan constants
    pub state: StepStateLayout,        // offsets of the GPU-written scalars
    pub ring: RingLayout,              // token ring buffer (§5.6)
    pub k_in_flight: u32,              // §5.2
}

pub struct StepTemplate {
    pub name: Sym,                     // "decode", "verify_m5", "draft_step"
    pub dispatches: Vec<Dispatch>,
    pub signal_after: bool,            // signal the shared event after this step
}

pub struct Dispatch {
    pub pipeline: PipelineId,          // index into the plan's pipeline table
    pub args_off: u32,                 // 256-byte-aligned offset in the argument arena (Part XI §3.6)
    pub grid: Grid,
    pub threads: [u16; 3],
    pub tg_mem: [u32; 4],              // threadgroup memory per index
    pub barrier_before: bool,          // from the scheduler's barrier list
    pub label: Sym,                    // for `sky explain` and `sky profile`
}

pub enum Grid {
    Fixed([u32; 3]),
    Indirect { state_off: u32 },       // MTLDispatchThreadgroupsIndirectArguments inside StepState
}
```

`DecodeProgram::validate` checks, before anything runs: every argument offset is in range and aligned; every pointer in an argument struct lands inside its arena and satisfies the element alignment of [Part XI](part11-metal-backend.md) §6.1; every `Indirect` offset points at an indirect-arguments slot; the barrier list is consistent with the dependencies the scheduler recorded (a dispatch that reads a buffer written by an earlier dispatch of the same step has a barrier between them). A plan that fails validation is rebuilt, never run.

### 3.2 `StepState`

One per sequence, in shared storage, read by every kernel through the `st` pointer of its argument struct.

```metal
struct StepState {
  // inputs of the next step, written by the previous step's last kernel (or by the CPU before step 0)
  uint  pos;              // position of the first token of the step
  uint  token;            // input token (plain decode)
  uint  verify_m;         // tokens processed in the step: 1, or 1 + γ when verifying
  uint  stop;             // 0 = run; otherwise the reason: 1 EOS, 2 max tokens, 3 CPU request, 4 error
  ulong rng_counter;      // keyed sampling: noise = Philox(seed, position) (Part XIII)
  uint  n_emitted;        // tokens written to the ring so far; monotone
  uint  max_tokens;
  uint  cancel;           // written by the CPU (§5.7); read by step_begin
  // speculation (Part XXI)
  uint  n_draft, n_accepted;
  uint  draft_tokens[16];
  // indirect arguments for position-dependent dispatches, written for the next step
  uint  attn_grid[3];     // MTLDispatchThreadgroupsIndirectArguments: {n_splits, n_kv_heads, 1}
  uint  icb_range[2];     // MTLIndirectCommandBufferExecutionRange {location, length}: ICB path only
};
```

### 3.3 One step

```text
 step t (one command buffer on the default path)
 ┌───────────────────────────────────────────────────────────────────────────────────────────────┐
 │ embed(token)                                    reads st.token                                 │
 │ for each layer (5 dependent kernels, 6 with the split combine; Part IX §3):                    │
 │   qkv_gemv  [norm prologue, head epilogue: q/k-norm, RoPE at st.pos, K/V append]               │
 │   attn_decode_split   grid = INDIRECT(st.attn_grid)   fixed split size (Part XI §9.5)          │
 │   attn_combine  fixed-order merge of split partials, only when > 1 split (Part IX §3)          │
 │   o_gemv    [residual epilogue]                                                                │
 │   gate_up_gemv [norm prologue, SwiGLU epilogue]                                                │
 │   down_gemv [residual epilogue]                                                                │
 │ lm_head_gemv [final-norm prologue]                                                             │
 │ sample (2 kernels: partial top-k / argmax per threadgroup, then a fixed-order final pass)      │
 │   final pass, one thread:  ring[n_emitted] = tok; n_emitted++; token = tok; pos++;             │
 │                            stop = EOS? max_tokens? cancel?; attn_grid = f(pos); icb_range = …  │
 └───────────────────────────────────────────────────────────────────────────────────────────────┘
   then: signal shared event (value t+1); fence → step t+1
```

Every kernel begins with `if (a.st->stop != 0) return;` ([Part XI](part11-metal-backend.md) §9.1). Only the sampler's final pass writes `StepState`, and it is the last dispatch of the step, so no kernel of step t sees a half-updated state.

The sampler's final pass, as generated for greedy decode (keyed-Gumbel sampling replaces the argmax, [Part XIII](part13-llm-runtime.md)):

```metal
[[kernel]] void sample_finish(constant SampleArgs& a [[buffer(0)]], ushort tid [[thread_index_in_threadgroup]],
                              ushort ln [[thread_index_in_simdgroup]])
{
  device StepState& s = *a.st;
  if (s.stop != 0) return;
  // reduce the per-threadgroup winners in index order: ties go to the lowest token id (fixed order)
  const uint tok = reduce_winners_fixed_order(a, tid, ln);
  if (tid != 0) return;
  a.ring->tokens[s.n_emitted % RING_CAP] = tok;
  s.n_emitted += 1;
  s.token = tok;
  s.pos += 1;
  s.rng_counter += 1;
  uint reason = 0;
  if (is_eos(a, tok))                 reason = 1;
  else if (s.n_emitted >= s.max_tokens) reason = 2;
  else if (s.cancel != 0)             reason = 3;
  s.stop = reason;
  s.attn_grid[0] = (s.pos + SPLIT) / SPLIT;          // splits covering positions 0 .. pos (next step's query)
  s.attn_grid[1] = N_KV_HEADS;
  s.attn_grid[2] = 1;
  s.icb_range[0] = ICB_STEP_FIRST;
  s.icb_range[1] = reason ? 0u : ICB_STEP_COUNT;     // ICB path: a stopped step executes nothing
}
```

### 3.4 What stays host-driven

Prefill is not part of the persistent loop. It is a sequence of chunk programs ([Part XVII](part17-prefill.md)) submitted back to back, each command buffer bounded in GPU time ([Part XI](part11-metal-backend.md) §11.1). Its last chunk writes `StepState` for the first decode step, and decode steps are committed behind it without a wait, so time to first token includes no extra round trip.

## 4. Replay mechanisms

### 4.1 Options

| Option | How | CPU per step (fused 32B: ≈ 330 dispatches; 0.6B draft: ≈ 145) | GPU per step | Can the GPU choose what runs? | Status |
|---|---|---|---|---|---|
| **A. Direct encoding, Metal 3** | serial compute encoder per step from the `DispatchList` | ≈ 27 µs / ≈ 12 µs at 0.081 µs per dispatch | baseline | grid sizes via indirect dispatch; command count no | `[VERIFIED costs]`; default |
| **B. Re-encoding into a reused `MTL4CommandBuffer`** | `beginCommandBufferWithAllocator` … `endCommandBuffer`, argument table, intra-pass barriers; allocators recycled | ≈ 64 µs / ≈ 28 µs at 0.194 µs per dispatch with barriers | same (1.184 vs 1.223 µs per dependent dispatch) | as A | `[VERIFIED costs]`; the object is reused, the commands are re-encoded ("reuse and repurpose each command buffer indefinitely by starting over, encoding new commands", Apple) |
| **C. CPU-encoded ICB, replayed** | ICB built once at plan load; `executeCommandsInBuffer` per step | ≈ 4 µs, any length | same (1.38 µs per dispatch with barriers) | yes: `executeCommandsInBuffer` with an indirect range buffer | `[VERIFIED costs]` |
| **D. GPU-encoded ICB** | a kernel writes `compute_command`s each step | ≈ 4 µs | + an encode kernel; ≈ 0.15 µs per reset slot | fully | `[VERIFIED]` correct; too costly to re-encode a whole step |
| **E. Recommitting an ended `MTL4CommandBuffer` without re-encoding** | commit the same encoded buffer again | ≈ commit cost only | same | as A | `[EXPERIMENTAL]`: not documented; XX-T9 probes it in a subprocess with a bounded wait |

The CPU figures for A and B are the measured per-dispatch costs multiplied by the dispatch counts of a fused plan (64 layers × 5 + head and sampler ≈ 330 for Qwen3-32B; 28 × 5 + 5 ≈ 145 for Qwen3-0.6B) `[LIKELY: arithmetic]`.

### 4.2 Decision

All options produce the same GPU work, so the choice is about CPU time, CPU wakeups and which decisions the GPU can make.

- **Default: A.** With K ≥ 2 steps in flight, 27 µs of encoding for a 30 ms step is invisible, and A is the path every production engine uses (R3 §2.1, R5 §1.1). It needs no ICB-specific pipeline flags or workarounds.
- **C where the GPU must decide the command count:** stopping (§5.4), adaptive draft lengths (§6.4), and small models where waking the CPU once per step costs energy (§7). Built in G6, adopted by measurement (§10).
- **B when the Metal 4 path is the default** ([Part XI](part11-metal-backend.md) §3.4). Its argument tables and queue barriers fit the static plan well, but it is not cheaper.
- **D only for the few commands whose grids depend on GPU state on the ICB path** (§4.3), never for whole steps.

The three encoders consume one `DispatchList`, so the replay mode is a runtime option (`--replay=direct|icb|mtl4`) and a test checks that all modes produce identical tokens.

### 4.3 Encoding sketches

Direct encoding, Metal 3 (the serial encoder orders dependent dispatches; fences order steps, [Part XI](part11-metal-backend.md) §3.5):

```rust
// crates/sky-metal/src/encode/mtl3.rs (sketch)
pub fn encode_step(cb: &ProtocolObject<dyn MTLCommandBuffer>, p: &Prepared, step: &StepTemplate,
                   wait: &ProtocolObject<dyn MTLFence>, update: &ProtocolObject<dyn MTLFence>) {
    let enc = cb.computeCommandEncoderWithDispatchType(MTLDispatchType::Serial).expect("encoder");
    enc.waitForFence(wait);
    for d in &step.dispatches {
        enc.setComputePipelineState(&p.pipelines[d.pipeline.0]);
        unsafe { enc.setBuffer_offset_atIndex(Some(&p.arg_arena), d.args_off as usize, 0) };
        for (i, &len) in d.tg_mem.iter().enumerate() {
            if len > 0 { unsafe { enc.setThreadgroupMemoryLength_atIndex(len as usize, i) } }
        }
        let tpt = MTLSize { width: d.threads[0] as usize, height: d.threads[1] as usize, depth: d.threads[2] as usize };
        match d.grid {
            Grid::Fixed(g) => unsafe {
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: g[0] as usize, height: g[1] as usize, depth: g[2] as usize }, tpt)
            },
            Grid::Indirect { state_off } => unsafe {
                enc.dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(
                    &p.state_buf, state_off as usize, tpt)
            },
        }
    }
    enc.updateFence(update);
    enc.endEncoding();
}
```

ICB built once at plan load (`MTLIndirectCommandBufferDescriptor` with `ConcurrentDispatch`, as in R6 [`icb.swift`](research/microbench/icb.swift)):

```rust
// crates/sky-metal/src/encode/icb.rs (sketch)
pub fn build_step_icb(gpu: &Gpu, p: &Prepared, step: &StepTemplate)
    -> Retained<ProtocolObject<dyn MTLIndirectCommandBuffer>> {
    let desc = MTLIndirectCommandBufferDescriptor::new();
    desc.setCommandTypes(MTLIndirectCommandType::ConcurrentDispatch);
    desc.setInheritBuffers(false);
    desc.setInheritPipelineState(false);
    desc.setMaxKernelBufferBindCount(1);                       // the argument struct only
    let icb = unsafe {
        gpu.device.newIndirectCommandBufferWithDescriptor_maxCommandCount_options(
            &desc, step.dispatches.len(), MTLResourceOptions::StorageModeShared)
    }.expect("icb");
    for (i, d) in step.dispatches.iter().enumerate() {
        let c = icb.indirectComputeCommandAtIndex(i);
        c.setComputePipelineState(&p.pipelines_icb[d.pipeline.0]); // created with supportIndirectCommandBuffers = true
        unsafe { c.setKernelBuffer_offset_atIndex(&p.arg_arena, d.args_off as usize, 0) };
        for (j, &len) in d.tg_mem.iter().enumerate() { if len > 0 { c.setThreadgroupMemoryLength_atIndex(len as usize, j) } }
        if d.barrier_before { c.setBarrier() }
        let g = match d.grid { Grid::Fixed(g) => g, Grid::Indirect { .. } => p.max_grid(d) }; // see below
        c.concurrentDispatchThreadgroups_threadsPerThreadgroup(g.into(), d.threads.into());
    }
    icb
}

// Per step: one call. The range comes from StepState, so a stopped step runs nothing.
unsafe { enc.executeCommandsInBuffer_indirectBuffer_indirectBufferOffset(&icb, &p.state_buf, p.icb_range_off) };
```

ICB commands carry fixed grid sizes. A position-dependent dispatch (decode attention) therefore uses either its maximum grid, with threadgroups beyond the current split count exiting at once, or a tiny GPU kernel that rewrites only that command's grid each step with MSL `compute_command` (option D for one or two commands; R6 verified GPU-side encoding) `[EXPERIMENTAL: XX-T7 measures both]`.

ICB rules collected from the evidence:

| Rule | Source |
|---|---|
| Pipelines used in an ICB need `supportIndirectCommandBuffers = true` | R6 §A4; [`common.swift`](research/microbench/common.swift) `makePipeline(icb:)` |
| Resources referenced by ICB commands must be resident: `useResource` per encoder, or residency sets | R6 [`icb.swift`](research/microbench/icb.swift); residency sets cover it ([Part XI](part11-metal-backend.md) §6.3) |
| `setBarrier` on a command orders it after all earlier commands; GPU cost per dispatch is the same as direct encoding | R6 §B4 |
| `executeCommands` itself belongs to no stage; each command keeps its own stages | Apple, `MTL4ComputeCommandEncoder` "Command stages" |
| On GPUs before Apple9, tinygrad sets each pipeline on the encoder and issues a zero-size dispatch before executing an ICB ("before apple9 the encoder must use the pipelines") | `runtime/ops_metal.py:111-115` (R2 §7.5). R6's single-pipeline ICB tests on the M2 Pro passed without it; Sky implements it behind a flag and XX-T7 tests whether multi-pipeline ICBs need it |
| No `setBytes` in ICB commands | hence the argument-struct ABI ([Part XI](part11-metal-backend.md) §3.6) |
| Successive replays of one ICB with shared argument memory cannot overlap | tinygrad serializes replays for this reason (R2 §7.5); Sky's steps are fenced anyway |

## 5. K steps in flight

### 5.1 Timeline

```text
 CPU (submitter)   commit s1 s2 s3 s4 ......... (event 1) commit s5 ...... (event 2) commit s6 ......
 CPU (streamer)                                  read ring[0], detokenize ..  read ring[1] ..........
                         │  │  │  │                   ▲                          ▲
 GPU               ──────[ s1 ][ s2 ][ s3 ][ s4 ][ s5 ][ s6 ] ...   gaps between steps ≈ 0.4 µs
                          each step: kernels → sample_finish writes StepState + ring → signal(t)
```

Each step is one command buffer on the default path; one command buffer holding several steps is equivalent on the GPU (0.4 µs between pre-committed command buffers, R6 §B3), and per-step buffers give per-step events, timing and error attribution. The Metal 3 queue is created with room for the in-flight window (`newCommandQueueWithMaxCommandBufferCount`; tinygrad uses 1,024, R2 §7.5).

### 5.2 Choosing K

K must cover the CPU's reaction time with margin, and should not be much larger, because a CPU-side stop takes effect only after the steps already in flight (§5.7).

| Model on M5 Max | Step time | K | GPU time queued |
|---|---|---|---|
| Qwen3-32B 4-bit | ≈ 33–35 ms (SPINE §2) | 2 | ≈ 70 ms |
| Qwen3-8B 4-bit | ≈ 8 ms `[LIKELY: roofline, 4.26 GB/token]` | 4 | ≈ 32 ms |
| Qwen3-0.6B 4-bit | ≈ 0.9 ms (§7) | 32 | ≈ 29 ms |

Rule: K = max(2, ⌈T_queue / T_step⌉) with T_queue ≈ 30 ms by default `[EXPERIMENTAL: §10 sweeps K]`. The CPU refills in batches when half the window has drained, so for small models it wakes about 60 times a second, not once per token (§9, energy).

### 5.3 Scalars patched by the GPU

The GPU writes every scalar that changes per token (§3.2). The CPU writes `StepState` only before the first step of a request and when it handles a stop or an error. This is what makes the encoded steps position-independent: argument structs hold addresses of `StepState` fields, never their values.

What depends on the position:

| Kernel | What changes with `pos` | Mechanism |
|---|---|---|
| QKV epilogue | RoPE angle, KV write address | reads `st.pos` |
| Decode attention | number of key splits, causal mask | indirect grid `st.attn_grid`; reads `st.pos` |
| Sampler | RNG counter, ring slot | reads and writes `StepState` |
| Everything else | nothing | — |

### 5.4 Stopping, and no-op steps

A generation stops on EOS, on `max_tokens`, on a CPU request (cancel, stop string) or on an error. The sampler sets `st.stop`; every later step in flight must then do nothing and leave the KV cache and `StepState` untouched.

| Mechanism | Path | Cost of a no-op step |
|---|---|---|
| Early exit: every kernel starts with `if (st->stop) return;` | all | every dispatch still launches its threadgroups, which exit at once: about (dependent dispatches) × (bubble) ≈ 0.3–0.4 ms for a 32B step `[EXPERIMENTAL: XX-T5 measures it]` |
| Zero-size indirect grids written by the sampler | direct | ≈ the bubbles alone; needs every dispatch indirect |
| ICB execution range of length 0 (`st.icb_range[1] = 0`) | ICB | one `executeCommands` with nothing to run `[EXPERIMENTAL]` |

Early exit is always present: it is the safety net and costs one load per thread. The other mechanisms are adopted only if the measured no-op cost matters; for a 32B model at most K − 1 = 1 step is wasted per generation, so it does not. It matters for speculation rounds with adaptive draft lengths (§6.4).

### 5.5 `MTLSharedEvent` signaling

| Path | Signal | Source |
|---|---|---|
| Metal 3 | `commandBuffer.encodeSignalEvent_value(event, t + 1)` after the step's encoder | R6 [`dispatch.swift`](research/microbench/dispatch.swift) |
| Metal 4 | `queue.signalEvent_value(event, t + 1)` after committing the step | R6 [`mtl4.cpp`](research/microbench/mtl4.cpp) |

The submitter waits for values with an `MTLSharedEventListener` notification (`notifyListener:atValue:block:`) or `waitUntilSignaledValue:timeoutMS:` (tinygrad's call, R2 §7.4), not with a spin: a spinning CPU thread is exactly the energy cost this part removes. Measured commit-to-observation latencies include submission (106 µs Metal 3, 84.8 µs Metal 4 for one tiny dispatch, R6 §B3); the latency from the GPU's signal to the CPU's wakeup alone is unmeasured `[EXPERIMENTAL: E4 in §10]`. It affects only when the user sees a token, not throughput, because the GPU keeps running the steps in flight.

### 5.6 The token ring buffer

```text
 shared storage, owned by the sequence
 ┌───────────────┬────────────────────────────────────────────┬──────────────────────┐
 │ header        │ tokens[RING_CAP]    (u32)                  │ meta[RING_CAP]       │
 │ (CPU-written: │ written by sample_finish / accept           │ (logprob, step index,│
 │  read_idx)    │ at n_emitted % RING_CAP                    │  flags: drafted …)   │
 └───────────────┴────────────────────────────────────────────┴──────────────────────┘
 GPU writes slot n, then increments StepState.n_emitted.
 CPU reads slots [read_idx, n_emitted) only after observing the event value of the step that wrote them.
```

Protocol:
1. **Visibility.** The CPU reads `n_emitted` and the slots only after it has observed the step's event value. Metal makes GPU writes to shared memory visible to the CPU at that point; reading earlier is unspecified `[LIKELY: Metal's completion and event semantics]`.
2. **Capacity.** `RING_CAP` ≥ (K + 1) × (tokens per step), where a speculation round emits up to γ + 1 tokens. The submitter never commits a step that could overwrite unread slots, so the ring needs no GPU-side backpressure.
3. **Order.** Tokens are appended in order; one sequence has one writer at a time.
4. **Streamer.** A CPU thread drains the ring, detokenizes on UTF-8 boundaries, checks stop strings, and prints or serves ([Part XIII](part13-llm-runtime.md)). It never touches the GPU.

### 5.7 Stop strings, cancellation and errors

Stop strings need detokenized text, so they are checked by the streamer on the CPU. On a match, and on user cancellation, the CPU writes `st.cancel = 1`. The GPU reads it in the sampler of the next step to run. Whether a CPU write to shared memory reaches a command buffer that is already committed is not guaranteed by Metal `[EXPERIMENTAL: XX-T5 measures how often it does]`, so the protocol does not depend on it:

1. the submitter stops committing;
2. the steps in flight run (at most K) and their tokens are discarded by the streamer;
3. after the last in-flight event, the CPU sets the authoritative `pos` and `n_emitted` to what the user received; KV entries beyond `pos` are dead and are overwritten by the next request (logical truncation, [Part XIV](part14-kv-cache.md)).

On a command-buffer error ([Part XI](part11-metal-backend.md) §11.2): the CPU drains the queue, restores `StepState` from its copy taken at the last completed step (it reads the state at every observed event), truncates KV logically to that position, re-plans if the error was a timeout, and resumes. No token is emitted twice, because the ring index is restored too.

## 6. Speculation rounds without a host sync

### 6.1 The round as a program

```text
 round r                                                    (all state on the GPU)
 ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
 │ draft_1 … draft_γ      draft program steps (small model, EAGLE-3 head, or prompt lookup):      │
 │                        each writes st.draft_tokens[i] and its draft probabilities             │
 │ verify                 target program at M = 1 + γ over [st.token, draft_1 … draft_γ];       │
 │                        the verify family of Part XI §9.6: same bits per token as plain decode │
 │ accept                 greedy: longest prefix where target argmax = draft;                    │
 │                        sampling: keyed-Gumbel equality or rejection sampling (Part XXI)       │
 │                        writes n_accepted, the bonus or correction token, appends             │
 │                        1 + n_accepted tokens to the ring; pos += 1 + n_accepted;             │
 │                        token = last emitted; stop if EOS among them or max_tokens reached     │
 │ rewind                 target and draft KV beyond pos are dead (logical); hybrid models       │
 │                        restore recurrent state from the snapshot at the accepted position     │
 │ signal(r)                                                                                     │
 └──────────────────────────────────────────────────────────────────────────────────────────────┘
 round r+1 reads st.token and st.pos: no CPU between rounds
```

The dispatch list of a round is fixed; only `StepState` varies. The CPU keeps K rounds in flight exactly as it keeps K decode steps (§5).

### 6.2 What makes it exact

Exactness, meaning speculative output byte-identical to Sky's plain output, needs three properties, all owned elsewhere:
- the verify pass computes each token's logits with the same bits as plain decode: one kernel family for all widths ([Part XI](part11-metal-backend.md) §9.2, §9.6) and fixed-split attention (§9.5);
- KV written during verification equals KV that plain decode would write for the same position, and rollback truncates rather than recomputes (R8 §3.5 rule 3);
- keyed sampling: the noise for a position depends on (seed, position), not on the step that drew it ([Part XXI](part21-speculative.md)).

This part's contribution is that the round runs entirely on the GPU, so nothing in the host loop can reorder or re-run work.

### 6.3 Why the host sync matters

mlx-lm's `speculative_generate_step` performs `mx.eval(tokens, draft_tokens)` and two `.tolist()` calls per round; "the GPU idles while Python compares tokens, rewinds caches and builds the next draft graph" (generate.py:620-622, R4 §2.3) `[VERIFIED]`. The cost of one synchronous round trip alone is 149–158 µs in MLX (R3 §7.E) and 104.6 µs of GPU idle in raw Metal (R6 §B3). On top of that, each draft step in mlx-lm is a Python-built graph: 1.94 ms of CPU per Qwen3-0.6B token on the M2 Pro (R4 §7.E). In Sky a round costs the CPU only its commit (tens of µs, §4.1).

### 6.4 Adaptive draft length

Some drafting policies stop drafting early when the draft's confidence drops ([Part XXI](part21-speculative.md)). The round then has a variable number of draft steps and a variable verify width:
- **direct path:** encode γ_max draft steps; the draft sampler sets a per-round flag after which the remaining draft kernels exit early (§5.4), and the verify kernels read `st.verify_m`;
- **ICB path:** the draft steps are one ICB executed with an indirect range whose length the GPU sets, so skipped draft steps cost nothing `[EXPERIMENTAL: XX-T8]`.

Verify kernels of the family take M from `StepState` at run time, so a variable width needs no re-encoding and does not change any token's bits ([Part XI](part11-metal-backend.md) §9.2).

## 7. Draft-model execution at low overhead

A draft model is the case where fixed costs dominate. For Qwen3-0.6B 4-bit, roofline.py gives 0.39 GB per token at 512 context; at the planning bandwidth of 534 GB/s on the M5 Max that is 0.73 ms, and at the M2 Pro's 196 GB/s 1.99 ms.

| Cost per 0.6B token | mlx-lm (M2 Pro, measured) | Sky plan (arithmetic) |
|---|---|---|
| Dispatches | 541, with 362 barriers (R4 §7.C) | ≈ 145 dependent: 28 layers × 5 + embedding, final norm, LM head, 2 sampler kernels |
| GPU dispatch bubbles | part of the 1.11 ms "non-matmul" share (R4 §7.E) | ≈ 145 × 1.1–1.25 µs ≈ 0.17 ms on the M2 Pro; ≈ 0.15 ms on the M5 Max assuming ≈ 1 µs (R6 hardware model) `[LIKELY]` |
| CPU per token | 1.94 ms build + encode (R4 §7.E) | ≈ 12 µs direct encoding, ≈ 4 µs with an ICB (§4.1) |
| Token time | 3.51–3.76 ms (267–281 tok/s) | M2 Pro: 1.99 / 0.92 (streaming efficiency assumed 92%) + 0.17 ≈ 2.33 ms → ≈ 430 tok/s, 1.5–1.6× mlx-lm `[LIKELY]`; M5 Max: 0.73 / 0.92 + 0.15 ≈ 0.94 ms → ≈ 1,060 tok/s `[LIKELY]` |

Assumptions are stated in the table; the 92% streaming efficiency is below what the raw GEMV reached on large matrices (94–98%, R6 §B2) but small matrices stream worse (MLX's k/v projections ran at 54–63%, R3 §7.B), so it is a planning number to be replaced by measurement. [Part XXII](part22-performance-model.md) §7, which adds a fitted fill cost per kernel (≈ 3.5 µs effective per dependent kernel, SPINE §2), predicts less: ≈ 401 tok/s (1.48×) on the M2 Pro and ≈ 862 tok/s on the M5 Max. The targets this supports are G4's 0.6B decode ≥ 1.3× mlx-lm on the M2 Pro and SPINE §3's ≥ 1.5× for small and draft models. BaseRT's 1.33× over MLX on Qwen3-0.6B on an M5 Pro (R7 §8.1) shows the gap is real on M5 hardware too.

How the draft model is run:
- **Its own static program** (`StepTemplate` "draft_step"), compiled with the same skeletons, with its own KV and arenas, resident beside the target's.
- **≤ 5 dependent kernels per layer**: the norm prologues and head-block epilogue of [Part XI](part11-metal-backend.md) §9.1, in [Part IX](part09-scheduler.md) §3's plan. At short context the draft's attention runs as one split, so Part IX's separate split-combine kernel is not needed.
- **Draft steps back to back inside the round** with no CPU between them, sampled on the GPU.
- **Draft cost ratio.** A 0.6B draft token at ≈ 0.94 ms against a 32B step of ≈ 34 ms gives a ratio of ≈ 0.028 (≈ 0.04 in Part XXII's model, which includes the fill term), inside the 0.02–0.05 range SPINE §2 assumes for its speculation table `[LIKELY: arithmetic]`. In mlx-lm on the M2 Pro the CPU cost alone of a 0.6B draft step (1.94 ms) exceeds its GPU time.

## 8. Why we do not build a megakernel

A persistent megakernel keeps one grid of threadgroups resident and moves through the model's operations with device-memory flags instead of kernel boundaries. Hazy Research's Llama-1B decode megakernel reached 78% of bandwidth on an H100; Mirage's MPK took Qwen3-8B on an A100 from 14.5 to 12.5 ms per token (1.16×) against a ≈ 10 ms bound (R7 §7.1) `[VERIFIED]`. On Apple GPUs the preconditions fail.

### 8.1 The evidence

| Precondition (R7 §7.3) | Apple GPU, measured or documented | Source |
|---|---|---|
| Forward progress for spinning threadgroups | **None documented.** "Forward progress", "deadlock" and "spin" appear nowhere in the MSL 4.1 spec | R6 §A4 `[VERIFIED]` |
| Co-residency of the whole grid | **48 threadgroups × 1,024 threads co-resident on the M2 Pro; with 64, the extra 16 were never scheduled until others exited, and a grid barrier deadlocked (48 timeouts).** No API reports residency or occupancy | R6 §A4, §B6 `[VERIFIED]` |
| Ordered device-memory flags | **A relaxed `atomic_load` spin never observed other threadgroups' increments**, with or without `coherent(device)`: every threadgroup hit the spin bound. Working spins need a device-scope fence in the loop (1.26 µs per barrier) or a read-modify-write (1.11 µs). Acquire/release orders arrive only with MSL 4.1 on macOS 27 | R6 §A4, §B6 `[VERIFIED]` |
| A global sync cheaper than a kernel boundary | **No: 1.11–1.43 µs per device-wide barrier, the same as a dependent dispatch boundary (1.1–1.25 µs)** | R6 §B3, §B6 `[VERIFIED]` |
| Long-running kernels allowed | **The display watchdog kills long GPU work** (`kIOGPUCommandBufferCallbackErrorImpactingInteractivity`); on an M5 Max sustained load escalated to kernel panics with a display attached | R6 §A4 `[VERIFIED]` |
| Prefetch across operation boundaries | no TMA; the undocumented `simdgroup_async_copy` helps on M1/M2 and hurts on Apple9 and later (Metal FlashAttention) | R7 §7.3 `[VERIFIED]` |
| Registers | a megakernel's register allocation is the maximum over its operation types, which lowers occupancy (MPK) | R7 §7.1, §7.3 |
| Static schedule | available: Sky's plan is static | §3 |

### 8.2 The upside, if it worked

| Model | Dependent boundaries per token (fused plan) | Bubble time | Share of the step |
|---|---|---|---|
| Qwen3-32B, M5 Max | ≈ 330 | ≈ 0.33 ms at ≈ 1 µs (assumed) | ≈ 1% of ≈ 34 ms |
| Qwen3-0.6B, M5 Max | ≈ 145 | ≈ 0.15 ms | ≈ 15% of ≈ 0.94 ms |

A megakernel that replaced each boundary with a global barrier would save nothing, because the barrier costs the same. It could save time only by replacing full-grid syncs with finer producer/consumer flags (start the MLP on the first finished chunk of the attention output, for example), which needs ordered flags and guaranteed co-residency, the two things Metal does not give. For dense 32B, MLX already runs the whole step at 94–97% of its measured bandwidth (R3 §7.F), which bounds the upside of any scheduling trick at a few percent; SPINE D18 states it as ≤ 3%.

### 8.3 What others on Apple chose

Monolith (`jiazhihao/mpk-apple`, opened 2026-09-24 by MPK's author) describes its design as "the whole generation loop compiled into one GPU-resident static program of fused whole-GPU dispatches, replayed from an indirect command buffer with no CPU work on the critical path" (R7 §7.1) `[VERIFIED]`. That is a static dispatch sequence with ICB replay, the design of this part with its ICB option (§4), not a spin-waiting kernel. Its decode target is ≥ 1.10× over the better of MLX and llama.cpp.

### 8.4 What would reopen the question

A feasibility spike (XX-T13, gate G9) runs only when all of these hold:
1. macOS 27 is available on the benchmark machine and acquire/release atomics compile and order correctly in R6's barrier test (`persist.swift`, rerun with `memory_order_acquire`/`release`);
2. a calibrated co-residency probe gives a stable capacity on the M5 Max across power modes;
3. after fusion, measured dispatch bubbles exceed 10% of the draft model's step;
4. the watchdog's limit on a headless machine and with a display is known.

Even then the spike measures fine-grained flags against kernel boundaries before anything is built on it `[MOONSHOT]`.

## 9. Expected gains, stated honestly

| Workload | What this part changes | Expected | Basis |
|---|---|---|---|
| Qwen3-32B plain decode, M5 Max | no CPU in the chain; replay instead of per-token graph build | **≤ 3%, likely ≈ 0–1% from this part alone.** The decode headroom comes from fusion and kernels (Parts IX, XI, XVI), not from replay | MLX pipelines and is at 94–97% of measured bandwidth (R3 §7.F); one command buffer per token changed nothing at real sizes (R3 §7.F); llama.cpp's replay attempt gained nothing (#14570) |
| Qwen3-8B plain decode, M2 Pro | same | ≤ 3% from this part | same |
| Small and draft models (0.6B–1.7B) | no per-token graph build; ≈ 145 dependent dispatches instead of 541 | **≥ 1.5× mlx-lm** target; ≈ 1.5–1.6× by arithmetic on the M2 Pro (§7) | mlx-lm at 54.5% of bandwidth with 1.94 ms of CPU per token (R4); BaseRT 1.33× on 0.6B (R7) |
| Speculation | rounds without host syncs; draft steps without CPU | removes ≥ 0.1 ms of idle per round plus the CPU cost of draft steps (1.94 ms per 0.6B draft step in mlx-lm on the M2 Pro); the enabling condition for exact, cheap rounds | R4 §2.3, §7.E; R6 §B3 |
| Energy per token | CPU near idle during decode; wakeups batched (§5.2) | up to ≈ 14% less system power at equal speed for 0.6B on the M2 Pro: mlx-lm's host loop drew 7.1 W of a 49.2 W system `[LIKELY: arithmetic]`; the SPINE target is ≤ 0.85× mlx-lm J/token | R9 §6.3 |
| Latency jitter (p99 inter-token time) | no Python, no garbage collection, no graph build on the path | reported, no target | — |
| Time to first token | none directly: prefill is host-driven (§3.4) | — | — |

## 10. Plan to measure on the M5 Max

Pre-registered before data collection (R9 §7.0), on the benchmark host of SPINE D14: M5 Max 40-core, 128 GB; latest macOS 26 and, when available, macOS 27; AC power, High Power Mode, quiet-machine gate, interleaved randomized blocks; one external harness timestamps tokens (SPINE D13). All runs pair Sky with mlx-lm on the same files the same day.

| ID | Experiment | Conditions | Metrics |
|---|---|---|---|
| E1 | Dispatch costs on Apple10 | R6's `dispatch`, `mtl4`, `icb` binaries rerun unchanged | t_dep, t_indep, CPU encode per dispatch, ICB replay cost, inter-CB gap |
| E2 | Replay matrix | modes {direct with host sync per step (K = 1, wait), direct K ∈ {2, 4, 8, 32}, Metal 4 re-encode, CPU-ICB, ICB + indirect range} × models {0.6B, 1.7B, 8B, 32B} × context {512, 4K, 32K} | tok/s; GPU busy fraction from command-buffer times; gaps between steps; CPU time per token (thread CPU time); p50/p99 inter-token time |
| E3 | No-op steps | early exit vs zero-size indirect grids vs zero ICB range, 0.6B and 32B | GPU time per stopped step |
| E4 | Event latency | GPU signal → CPU wakeup: listener vs `waitUntilSignaledValue` vs spin | µs, CPU wakeups per second |
| E5 | Speculation rounds | 32B target + 0.6B draft, γ ∈ {2, 4, 8}: rounds with a host sync vs GPU-resident rounds; same draft, same acceptance | tok/s, idle time per round, tokens bit-identical to plain decode |
| E6 | Energy | 10-minute sustained decode, 0.6B and 32B; Sky modes vs mlx-lm | J/token from SMC `PSTR`; CPU and GPU channels from IOReport (R9 §6.3) |
| E7 | Overrun | cancellation latency and wasted steps vs K | steps run after a CPU stop request; how often an in-flight command buffer sees `cancel` |

Decision rules, fixed in advance:
- The ICB path becomes the default for a model class if it gives ≥ 3% tok/s or ≥ 10% lower J/token at equal tok/s on that class; otherwise direct encoding stays the default.
- The Metal 4 path becomes the default only if it beats Metal 3 by ≥ 2% on some model and loses nowhere ([Part XI](part11-metal-backend.md) §15 item 5).
- K defaults come from the smallest K whose tok/s is within 1% of K = 32.
- The megakernel spike (§8.4) is scheduled only if E1 and E2 show dispatch bubbles above 10% of the draft step after fusion.

Gates: G4 checks the M2 Pro numbers (0.6B ≥ 1.3× mlx-lm); G5 runs E1–E2 on the M5 Max; G6 runs E5 and E6 and must show speculative output byte-identical to plain output.

## 11. Disagreement with the spine

1. **D9 says "dispatches are replayed (ICB or Metal 4 command reuse)".** Metal 4 command buffers are reused objects, but their commands are re-encoded each time; recommitting an encoded buffer without re-encoding is undocumented (§4.1, option E). And direct encoding of a static plan from Rust costs tens of microseconds per 32B step, already hidden when two steps are in flight. This part therefore makes **direct encoding of the static plan the default** and uses ICBs where the GPU must choose how many commands run and where CPU wakeups cost energy. The spine's intent (no per-token graph work, no CPU in the chain) is unchanged.

   Resolved: the spine now says the static plan is directly encoded each step (tens of µs, hidden behind the GPU) and that ICBs are used only where the GPU must choose the command count (stop, adaptive γ) or where CPU wake-ups cost energy (SPINE D9).
2. **D9 says "K tokens are submitted per command buffer".** The design keeps K steps in flight with one command buffer per step by default. Pre-committed command buffers are separated by 0.4 µs on the GPU (R6 §B3), so the difference is a CPU-side detail; per-step buffers give per-step events, timing and error attribution. K steps in one buffer remain an option.

   Resolved: the spine now says K steps in flight, one command buffer per step (SPINE D9).

## 12. Implementation tasks

| ID | Task | Crate / files | Deliverable | Acceptance test |
|---|---|---|---|---|
| XX-T1 | Decode program as data | `sky-llm/src/program.rs` | `DecodeProgram`, `validate`, text dump for `sky explain` | round-trip through the plan cache; a corrupted offset or missing barrier is rejected |
| XX-T2 | `StepState`, `sample_finish`, GPU-written scalars | `sky-llm/src/state.rs`, sampler skeletons (Part XIII) | §3.2–3.3 | 256 greedy tokens with K = 4 are bit-identical to K = 1 with a host sync after every step |
| XX-T3 | Direct-encoding submitter, Metal 3 | `sky-metal/src/encode/mtl3.rs`, `sky-llm/src/decode_loop.rs` | serial encoder per step, fences, K in flight | median GPU gap between steps < 2 µs (vs ≈ 105 µs with a host sync) from command-buffer timestamps |
| XX-T4 | Shared events, ring buffer, streamer | `sky-llm/src/stream.rs` | §5.5–5.6 | under random CPU delays (0–50 ms) no token is lost, duplicated or reordered over 10⁶ tokens |
| XX-T5 | Stop flags and no-op steps; cancellation | kernels' early exit; `decode_loop.rs` | §5.4, §5.7 | generation stops at EOS with ≤ K − 1 wasted steps; a wasted 32B step costs < 3% of a real step; cancellation leaves KV and `StepState` consistent (next request's output equals a fresh process's) |
| XX-T6 | Indirect grids for attention | `sample_finish`, attention dispatch | `st.attn_grid` | contexts 1…32K give the right split counts; output equals a fixed-grid run bit for bit |
| XX-T7 | CPU-encoded ICB path | `sky-metal/src/encode/icb.rs` | build once, replay per step; attention grid by maximum grid or GPU patch; pre-Apple9 pipeline-use flag | tokens identical to the direct path; CPU time ≤ 10 µs per step; both attention variants timed |
| XX-T8 | ICB indirect ranges | `icb.rs` | stopped steps and variable draft steps skipped by range | a stopped step costs ≤ 10 µs of GPU time `[target]` |
| XX-T9 | Metal 4 submitter; recommit probe | `sky-metal/src/encode/mtl4.rs` | argument tables, queue barriers, allocator ring; a bounded subprocess probe for recommitting an ended buffer | tokens identical to the direct path; the probe reports works / fails / undefined without hanging the machine |
| XX-T10 | Speculation round without host sync | `sky-llm/src/spec/round.rs` (with Part XXI) | §6 | greedy output bit-identical to plain decode on 20 prompts × 128 tokens with a 0.6B draft; instrumented run shows no CPU wait inside a round |
| XX-T11 | Draft-model plan | `sky-llm` arch builders, Part IX fusion rules | ≤ 5 dependent kernels per layer for Qwen3-0.6B/1.7B | 0.6B decode ≥ 1.3× mlx-lm on the M2 Pro (G4); dispatch census printed by `sky explain` |
| XX-T12 | CPU and energy instrumentation | `sky-cli` (`sky bench llm`) | CPU time per token, wakeups per second, J/token per R9 | numbers reported with method; matches R9's protocol on a known workload |
| XX-T13 | Megakernel feasibility spike (G9, only on the §8.4 triggers) | `research/spikes/persist41/` | R6's barrier test with acquire/release on macOS 27; co-residency calibration; flag-based producer/consumer microbenchmark | a written go/no-go with measurements; no product code |

## 13. Measurements that decide

1. **Replay mode vs tok/s and CPU energy** across model sizes (E2, E6). Decides whether ICBs become the default for small models and how much engineering the persistent runtime deserves (Part VI §12).
2. **The dependent-dispatch bubble on the M5 Max** (E1). Sets the fixed cost per kernel in the cost model ([Part XXII](part22-performance-model.md)) and the draft model's ceiling.
3. **No-op step cost** (E3). Decides whether early exit suffices or zero grids or ICB ranges are needed, mainly for adaptive draft lengths.
4. **Speculation rounds with and without host syncs** (E5). Decides how much of Sky's speculative speedup comes from the loop rather than the kernels, and is part of the G6 claim.
5. **Event-to-CPU latency and wakeup rate** (E4). Decides the streamer's wakeup policy for interactive use.
6. **Overrun after a CPU stop** (E7). Decides the default K for interactive serving.

## 14. Risks and negative results

**Negative results this design accepts:**
- Replay does not reduce GPU time per dispatch: 1.38 µs with an ICB against 1.40 µs direct (R6 §B4).
- llama.cpp's maintainers found no gain from command replay on top of pipelined encoding (#14570), and R3 found 0% from one command buffer per token at real sizes (R3 §7.F).
- A device-wide barrier inside a kernel costs as much as a kernel boundary; relaxed spins never see other threadgroups' writes; grids beyond co-residency deadlock (R6 §B6).
- The GPU cannot create command buffers; it can only fill pre-allocated ICB slots, pick indirect grid sizes and indirect ICB ranges (R6 §A4).
- GPU-encoded ICBs cost ≈ 0.15 µs per empty slot plus the encode kernel, too much to re-encode a whole step each token (R6 §B4).

**Risks:**

| Risk | Effect | Mitigation |
|---|---|---|
| The gain on dense 32B is within noise | the persistent runtime looks like effort without return on the headline model | it is justified by small models, speculation and energy (§9), and the report says so; E2 measures it |
| A CPU write (`cancel`) is not seen by command buffers already committed | up to K extra steps after a stop | the protocol discards them and restores state (§5.7); K is bounded by E7 |
| An error in one step leaves later in-flight steps running on bad state | corrupted tokens or KV | CPU snapshot of `StepState` at each observed event; logical KV truncation; replay from the snapshot (§5.7) |
| ICB quirks differ by GPU family (pre-Apple9 pipeline use) | wrong results or validation errors on M1/M2 | flag-guarded workaround; XX-T7 tests on Apple8 and Apple10 |
| Metal 4 is newer and less exercised than Metal 3 | driver bugs | Metal 3 stays the default until measurements and the test suite say otherwise |
| Debugging is harder without the CPU in the loop | slower diagnosis | `SKY_METAL_SYNC=1` and a debug mode that waits and checks `StepState` after every step ([Part XI](part11-metal-backend.md) §11.3) |
| The watchdog kills a long command buffer | lost work | decode steps are short; prefill chunks are bounded ([Part XI](part11-metal-backend.md) §11.1) |
| Energy measurement is noisy on M5 (counters update in ≈ 2.1 s batches; channel double-counting) | the energy claim is not demonstrable | R9's protocol: ≥ 60 s loops, `PSTR` as the headline, wall-power meter as the gold standard (R9 §6.3) |
