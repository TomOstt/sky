# R2 — tinygrad, traced through the code (evidence notes for Sky)

- Source: `$S/src/tinygrad` (shallow clone) at commit `512e9b230c6294e8999a0ab265254460727758ce` (2026-10-02 00:43 +0800, "Revert "broken test" (#18564)"). All `path:line` citations refer to this commit. `S=/private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad`.
- Machine: Apple M2 Pro (16-core GPU, 16 GB, nominal 200 GB/s ≈ 204.8 GB/s LPDDR5-6400 × 256-bit), macOS 26.6.1, Metal family Apple8 (the generated host code takes the "before apple9" branch, see §7.3). Python 3.12.12 venv.
- Raw logs, scripts, and outputs live in `$S/research/R2/`. All timings ran through `bash $S/benchlock.sh …`. Commands are listed in §11.
- Labels: **VERIFIED** (code at this commit, our own measurement, or an official doc) · **LIKELY** (strong inference, reason given) · **EXPERIMENTAL** (needs a test) · **MOONSHOT**.
- Scope warning: tinygrad moves fast. Much of what is "known" about tinygrad (ShapeTracker, LazyBuffer, `Ops.VIEW`, `DEFINE_GLOBAL`, OptOps UPCAST/LOCAL/GROUP/UNROLL, `runtime/graph/metal.py`, MetalProgram) is **gone** at this commit. §1 maps old names to new ones.

---

## 0. Summary

1. **One IR for everything.** VERIFIED. A hash-consed `UOp(op, src, arg, tag)` with **derived** dtype and shape represents:
   - the tensor graph and the kernel AST;
   - the schedule (`LINEAR` of `CALL`s) and the linearized program;
   - the host-side GPU submission code;
   - even the pattern-matcher compiler.
   Every transformation is a `graph_rewrite` with a `PatternMatcher`.
2. **ShapeTracker is gone; "rangeify" is current.** VERIFIED. Movement ops are UOps whose effect is applied as index arithmetic on `RANGE` loop variables (`schedule/indexing.py:169-187`). Fusion means sharing ranges. Materialization means fresh ranges plus `STAGE`→`STORE`.
3. **Fusion and recompute decisions are heuristics, not a cost model.** VERIFIED. The realize/fuse decision comes from consumer-range agreement (`schedule/indexing.py:230-277`). Recompute-vs-store is a 3-rule test (`schedule/rangeify.py:84-96`). The maintainers say so: "tinygrad doesn't yet have a systematic way to do it" (`docs/developer/speed.md:43`).
4. **OptOps are now `TC`, `SPLIT`, `PADTO`, `SWAP`** (`codegen/opt/__init__.py:6-8`). VERIFIED. UPCAST/LOCAL/GROUP/UNROLL are all `SPLIT(axis, (amount, AxisType, top))`.
5. **Metal tensor cores = 8×8×8 `simdgroup_matrix` only.** VERIFIED. No Metal 4 tensor ops, no Metal Performance Primitives, nothing M5/Apple10-specific (grep negative, §5.3).
6. **Metal runtime is now "HCQ2" over Indirect Command Buffers.** VERIFIED. There is no per-kernel Metal program object (`runtime/ops_metal.py:213`, runtime=None). Every launch is batched into an ICB with **one argument buffer**. Submission is a **compiled C function on the CPU** that calls `objc_msgSend` (`runtime/ops_metal.py:91-125`; generated code in `R2/debug4_matmul_relu.txt`).
7. **Measured on our M2 Pro**, using tinygrad's own LLM stack (`python -m tinygrad.llm`):
   - Qwen3-0.6B GGUF (Q4_0 blocks, Q6_K tied LM head) decodes at **54.3 tok/s** (18.41 ms/token median, n=26). That is ≈**21.6 GB/s ≈ 10.5% of peak** bandwidth.
   - The same model with weights dequantized to fp16 at load (`REALIZE=1`) runs **faster** (76.3 tok/s) despite 3.2× the bytes. The fused-dequant GEMV codegen is the bottleneck.
   - One decode step is **488 kernels** (~17 per layer).
8. **Fused quantized GEMV is the weak spot.** VERIFIED, measured at 12288×4096. Effective bandwidth on packed bytes:
   - Q4_0: 53 GB/s;
   - Q8_0: 86 GB/s;
   - Q6_K: 38.5 GB/s;
   - dense fp16 (heuristic): 134 GB/s.

   Root cause, from the generated MSL:
   - packed bytes are read as scalar `uchar` loads, since coalescing excludes uint8 (`codegen/late/coalesce.py:143`);
   - nibbles are extracted with a runtime integer **division**;
   - each thread computes 3 whole rows with no SIMD-cooperative reduction, because the matvec heuristic requires both multiplicands to be direct loads (`codegen/opt/heuristic.py:62-64`).
9. **BEAM**: ~4 s per GEMV kernel. It found a ~190 GB/s fp16 GEMV kernel-only (good). For Q4_0 it reported 206 µs but picked a kernel that really takes ~540 µs (the heuristic's takes ~316–510 µs). Metal BEAM never flushes caches (`codegen/opt/search.py:144`; only NV implements `invalidate_caches`, `runtime/ops_nv.py:721`).
10. **Dispatch.**
    - TinyJit round trip on M2 Pro: **155 µs/call** for 1 tiny kernel.
    - **+4.3 µs** marginal per additional kernel inside one ICB batch.
    - Eager realize of a trivial op (all caches warm): 453 µs.
    - Every ICB command has `setBarrier()` (`runtime/ops_metal.py:263`), so there is no kernel-level concurrency.
11. **Published Apple number (maintainer statement).** The tiny corp claims Qwen3.5-0.8B on M3 Max: baseline **178 tok/s** (vs LMStudio 160, llama.cpp 140), and **195.7 tok/s** with its "custom kernel feature" ([X post](https://x.com/__tinygrad__/status/2046973311977230768)). Quant format is not stated in the post. tinygrad's CLI maps `qwen3.5:0.8b` to a Q8_0 GGUF (`tinygrad/llm/cli.py:94`), so LIKELY Q8_0. This is consistent with our ~40% of peak for fused Q8_0 GEMV (§8.4).

---

## 1. Old name → current mechanism (VERIFIED by grep: 0 hits in `tinygrad/` for each old name)

| Old tinygrad concept | At this commit |
|---|---|
| `ShapeTracker` / `View` / `Ops.VIEW` | Movement-op UOps (`RESHAPE/EXPAND/PERMUTE/PAD/SHRINK/FLIP`) with shape-as-UOp srcs (`uop/ops.py:810-835`). Applied as index math by **rangeify** (`schedule/indexing.py:169-187`). No `tinygrad/shape/` dir. |
| `LazyBuffer` | The Tensor holds a `UOp` (`tensor.py:55`, `tensor.py:104-113`) |
| `DEFINE_GLOBAL` / `DEFINE_LOCAL` / `DEFINE_REG` | `PARAM` (`ParamArg.addrspace=GLOBAL`) / `BUFFER`/`ALLOC` with `addrspace=LOCAL/REG` (`uop/ops.py:24-42`, `1197-1221`) |
| `DEFINE_VAR` / `BIND` | `PARAM` with `addrspace=ALU` and `vmin_vmax`. Binding sets `ParamArg.val` (`uop/ops.py:1014-1048`). |
| `VECTORIZE` / `GEP` | `STACK` (a vector is an extra leading shape dim, `uop/ops.py:355-357`) plus `INDEX` by constant (`uop/movement.py:20-22`) |
| `UNROLL` / `CONTRACT` ops | `AxisType.UPCAST` ranges, expanded by the `expander` (`codegen/__init__.py:78-84`) |
| `CONTIGUOUS` / `BUFFERIZE` | `STAGE` (`uop/__init__.py:91`, `schedule/prepare.py:58-59`, `schedule/indexing.py:92`) |
| `ASSIGN` | `STORE` + `AFTER` (`tensor.py:258-259`) |
| `KERNEL` / `ScheduleItem` list | `CALL` (body = `SINK` with `KernelInfo`) inside a `LINEAR` (`docs/developer/developer.md:20`, `schedule/__init__.py:79`) |
| OptOps UPCAST/LOCAL/GROUP/GROUPTOP/UNROLL | `SPLIT(axis, (amt, AxisType.UPCAST|LOCAL, top))` (`codegen/opt/postrange.py:94-116`) |
| `runtime/graph/metal.py` (MetalGraph) | HCQ2 batches: `runtime/support/hcq2.py` + `MetalQueue` (`runtime/ops_metal.py:127-166`) |
| `MetalProgram` per kernel | None. `MetalDevice(..., runtime=None)` (`runtime/ops_metal.py:213`). Kernels exist only as ICB commands. |

---

## 2. Pipeline map: Tensor op → GPU work on Metal

| Stage | Where (VERIFIED) | What happens |
|---|---|---|
| Tensor op | `tensor.py:104-116`, `mixin/*.py` | `Tensor._apply_uop` builds a new UOp node. Nothing runs (lazy). |
| Realize trigger | `tensor.py:218-225` (`realize`), `tensor.py:276-285` (`_buffer` for `.data/.numpy/.item`) | `linear_with_vars` → `run_linear` |
| Bufferize outputs | `tensor.py:167-210` | Outputs get storage via `BUFFER` + `STORE` + `AFTER`. Live Tensors are rewired (`_apply_map_to_tensors`, `tensor.py:18-32`). |
| Callify + cache-key normalization | `schedule/__init__.py:260-269`, `251-258` | Input buffers become `PARAM`s. **Bound Variables become value-stripped PARAMs** ("so different values hit same cache"). |
| Schedule cache | `schedule/__init__.py:121-150` | Key = `function.key` (sha256 over op/dtype/arg/src keys, `uop/ops.py:267-269`). `SCACHE>=2` persists to disk. |
| Prepare | `schedule/prepare.py:271-278` | Multi-device, inline calls, `split_reduceop` (`103-125`), copy/stage/store rules (`199-269`) |
| Rangeify (fusion) | `schedule/indexing.py:189-320`, `schedule/rangeify.py:350-376` | Assign ranges, decide realize points, convert movement ops to index math, remove/keep `STAGE`, limit buffers, `STAGE`→`STORE`, split into kernels |
| Schedule linearize | `schedule/__init__.py:31-79` | RAW/WAR deps between kernels; Kahn toposort into `LINEAR` |
| Memory plan | `schedule/memory.py:20-62` | Lifetimes + TLSF suballocation into one arena per (device, copy/compute lane) |
| Optimize kernel AST | `codegen/opt/postrange.py:239-256` | `opts_to_apply` / BEAM (`codegen/opt/search.py`) / `hand_coded_optimizations` (`codegen/opt/heuristic.py`) |
| Lower | `codegen/__init__.py:275-397` | **30 `graph_rewrite` passes** (count of `graph_rewrite(` in that range). Expander, reduce→accumulator, locals, gpudims, devectorize, coalescing, dtype/transcendental decomps, control flow. |
| Linearize | `codegen/late/linearizer.py:8-51` | Priority toposort: loads early, stores late, ranges placed by run-count |
| Render (Metal) | `renderer/cstyle.py:358-409` | MSL text. One `args_t` argument-buffer struct per kernel. |
| Compile | `runtime/ops_metal.py:29-71` | Private `MTLCompiler.framework` code-gen service (request type 13) → `MTLB` metallib. Disk-cached (`device.py:339-344`). |
| Pipeline | `runtime/ops_metal.py:241-247` | `newLibraryWithData` + `newComputePipelineStateWithDescriptor` with `setSupportIndirectCommandBuffers(True)` |
| Batch + encode | `runtime/support/hcq2.py:305-325`, `443-491`, `runtime/ops_metal.py:127-166`, `249-266` | Kernels go into an `MTLIndirectCommandBuffer` (one ConcurrentDispatch command per kernel, `setBarrier()` each). Args go into one shared argument buffer. Host submit code is a UOp program. |
| Host submit compile | `device.py:18` (`HCQ_RUNTIME_DEV=CPU`), `runtime/ops_cpu.py:90` | The submit program is rendered to **C** and compiled by clang (CPU device). It calls `objc_msgSend`. |
| Link | `runtime/support/hcq2.py:637-648` | Placeholders become real buffers. GPU addresses become constants (link-time patches) or a per-call input table. |
| Execute | `engine/realize.py:183-201`, `283-289` | `exec_hcq` writes input addresses into the table, then calls the compiled CPU submit function |
| Sync | `runtime/ops_metal.py:268-283` | `MTLSharedEvent waitUntilSignaledValue:timeoutMS:` |

---

## 3. UOps (Question 1)

### 3.1 Representation (VERIFIED)
- `@dataclass(eq=False, slots=True) class UOp: op, src, arg, tag` (`uop/ops.py:240-247`). **dtype is not a field.** It is a `recursive_property` computed by a total function `dtype_from_uop(op, src, arg)` (`uop/ops.py:122-195`, `246-247`). Examples:
  - CONST dtype comes from the Python value's type (int→`weakint`, float→`weakfloat`, `uop/ops.py:184-190`);
  - CAST/BITCAST take the dtype from `arg`;
  - PARAM/BUFFER take it from `ParamArg.dtype`.
- **Shape is derived too** (`_shape`, `uop/ops.py:329-450`). Broadcasting is implicit in elementwise ops (`uop/ops.py:440-447`).
- **Hash-consing**: `UOpMetaClass.__call__` looks up `(op, src, arg, tag, type(arg))` in a weakref cache, so structurally equal UOps are the same object (`uop/ops.py:197-213`, freed in `__del__` at `248-251`). Equality is identity (`_UOpTuple`, `uop/ops.py:229-234`).
- `key` = sha256 of `(op, dtype, arg)` plus the src keys (`uop/ops.py:267-269`). It is used for all caches: schedule, compile, BEAM, runtime.
- `ParamArg` (`uop/ops.py:24-42`) fields:
  - `slot`, `dtype`, `size` (always a concrete int; symbolic shapes become max-size + SHRINK, `uop/ops.py:1228-1240`);
  - `vmin_vmax`, `multiple_of` (divisibility hint);
  - `name`, `addrspace` (GLOBAL/LOCAL/REG/ALU), `device`, `volatile`, `image`;
  - `buffer` (the UOp owns the device Buffer 1:1), `bind_on_realize`, `val` (bound Variable value).
- `AxisType` (`uop/ops.py:17-22`): `DEVICE, GLOBAL, LOCAL, WARP, WEAK, LOOP, UPCAST, PLACEHOLDER`. A `RANGE`'s arg is `(axis_type, axis_id…)` (`uop/ops.py:498-506`).
- `metadata` (source-level provenance) is kept in a side `WeakKeyDictionary`, not in the node (`uop/ops.py:216`).
- `SPEC>1` runs a full type/shape verifier on every created node (`uop/ops.py:204-212`, `uop/spec.py`).

### 3.2 Ops vocabulary (`uop/__init__.py:13-102`, groups at `104-135`) (VERIFIED)
1. **Defines/special**: `SPECIAL` (GPU dims), `BUFFER`, `ALLOC`.
2. **Non-op**: `NOOP`, `REWRITE_ERROR`, `PARAM`, `CALL`, `PROGRAM`, `LINEAR`, `SOURCE`, `BINARY`, `SINK`, `AFTER` (ordering edge: "consumers of the AFTER run after src[1:]"), `GROUP`, `STACK`, `GETADDR` (hcq).
3. **Load/store**: `INDEX` ("a BinaryOp similar to ADD, but it operates on pointers"), `SHRINK`, `LOAD`, `STORE`.
4. **Math**: `WMMA`; Unary (`CAST BITCAST EXP2 LOG2 SIN SQRT RECIPROCAL NEG TRUNC`); Binary (`ADD MUL SHL SHR CDIV MAX CMOD CMPLT CMPNE CMPEQ XOR OR AND THREEFRY SUB FDIV POW FLOORDIV FLOORMOD`); Ternary (`WHERE MULACC`).
5. **Control/const/custom**: `BARRIER RANGE IF END ENDIF BACKEDGE` (while loop), `CONST`, `CUSTOM/CUSTOMI` (inline strings), `INS` (machine instruction for ISA renderers).
6. **Tensor-graph only**: `CONTIGUOUS_BACKWARD DETACH STAGE COPY MSELECT MSTACK CUSTOM_FUNCTION`; movement `RESHAPE PERMUTE EXPAND PAD FLIP` (+`SHRINK`), `UNSHARD`; `REDUCE ALLREDUCE`.
7. **Pattern-compiler IR**: `PYLITERAL`.

Note the toposort order: "the order of these Ops controls the order of the toposort" (`uop/__init__.py:12`).

### 3.3 Movement ops / views: rangeify is current (VERIFIED)
- Movement ops are UOps whose shape args are UOp srcs (`marg`, `uop/ops.py:810-816`; `_mop`, `818-835`). There is no view object.
- **Rangeify** gives every node a tuple of input/output index expressions over `RANGE` UOps. It walks the tensor graph consumers-first (`schedule/indexing.py:207-313`). A movement op transforms the index expressions (`apply_movement_op`, `schedule/indexing.py:169-187`):
  - SHRINK adds an offset;
  - PERMUTE reorders;
  - FLIP gives `s-1-r`;
  - EXPAND drops ranges;
  - PAD produces `(r-off).valid(r>=off & r<sh+off)`, i.e. a WHERE with `Invalid`;
  - RESHAPE linearizes to a flat index and re-splits with `//` and `%`, then simplifies. The comment there: "this simplify is doing a lot of heavy lifting. this is the replacement for the reshape view merging code" (`schedule/indexing.py:166`).
- After rangeify, movement ops are deleted (`remove_movement_op_after_rangeify`, `schedule/indexing.py:133-147`). Kernels contain only `INDEX(buf, index_expr)`.
- Contiguous views that collapse to a byte range become zero-copy buffer views (`UOp.contiguous_view`, `uop/ops.py:931-950`; `Buffer.view`, `device.py:252-254`).

### 3.4 Reductions and memory ops (VERIFIED)
- Tensor-level `REDUCE` has `arg=(op, n_axes)`, with the reduced axes permuted to the front (`uop/ops.py:652-661`). After rangeify it becomes `REDUCE(src, *ranges)` with `arg=(op, 0)` (`schedule/indexing.py:107-113`).
- In codegen, a reduce becomes a **REG accumulator buffer**: `acc.after(init).store(acc OP inp).end(ranges)` (`reduce_ranges_to_acc`, `codegen/__init__.py:197-204`).
- Grouped reductions across threads (LOCAL/WARP ranges) become a LOCAL `STAGE` (shared memory) plus a second reduce (`fix_group_for_reduce`, `codegen/__init__.py:160-175`). Barriers are inserted automatically (`codegen/__init__.py:249-273`).
- Memory:
  - kernel params are `PARAM` (GLOBAL);
  - `INDEX(param, idx)` is a pointer;
  - `LOAD(INDEX)` and `STORE(INDEX, val[, gate])`;
  - the gate is folded via `Invalid` (`uop/symbolic.py:61-102`);
  - `STAGE` marks "materialize here";
  - `AFTER` orders effects.
- Kernel I/O inference: `ProgramInfo.from_sink` (`uop/ops.py:1370-1388`).

### 3.5 Vectorization (VERIFIED)
- No vector dtype in the IR. **A vector is a shape dimension** (`STACK` adds a leading dim, `uop/ops.py:355-357`).
- `UPCAST` ranges are replaced by constant vectors reshaped into position (`expander`, `codegen/__init__.py:78-84`).
- `devectorizer2` unpacks elementwise ops over vector shapes into per-lane ops plus `STACK` (`codegen/__init__.py:117-158`).
- `memory_coalescing` then groups adjacent scalar loads/stores with the same base into `SHRINK(buf, offset, 4|2)` vector accesses (`codegen/late/coalesce.py:109-168`). This applies **only for float/half/int/uint/fp8** (`coalesce.py:143-145`; half8 only with `ALLOW_HALF8`). uint8 is not vectorized; see §8.4 for the consequence.
- The renderer emits `float4`/`half4`/`float2` and `(device float2*)` pointer casts.

### 3.6 Symbolic integers and the simplifier (VERIFIED)
- `Variable = UOp`: `UOp.variable(name, min, max)` is a `PARAM` with `addrspace=ALU` and `vmin_vmax` (`uop/ops.py:1014-1018`). `bind()` stores the value in the arg (`1032-1038`).
- Interval analysis: `_min_max` (`uop/ops.py:1105-1165`) covers ADD/SUB/MUL/SHL/SHR/CMOD/CDIV/FLOORDIV/FLOORMOD/AND/XOR/MAX/CMP/WHERE/RANGE/CAST and constant-table LOADs.
- Divisibility: `const_factor`, `divides`, `divide_exact`, `gcd` (`uop/ops.py:1060-1100`).
- `sym_infer` renders the expression to Python and `exec`s it (`uop/ops.py:1167-1181`). Launch dims are evaluated this way at runtime (`ProgramInfo.launch_dims`, `1361-1364`).
- Rule sets (all `PatternMatcher`s):
  - `symbolic_simple` (folding, `uop/symbolic.py:115-199`);
  - `symbolic` (term combining, where-folding, lt-folding, simplex canonicalization, `235-324`, plus `div_and_mod_symbolic`, `uop/divandmod.py`);
  - `uop_given_valid` / `simplify_valid` (simplify an index *given* its validity predicate, by substituting fake bounded variables, `uop/symbolic.py:342-386`);
  - `sym` (the full set, `440-475`).

### 3.7 PatternMatcher / graph_rewrite (VERIFIED)
- Rules are `(UPat, fn)` pairs. UPat supports:
  - op/dtype/arg/name/tag;
  - src as a tuple (exact), a list (all permutations, used for commutative ops), or a single UPat (repeat);
  - `UPat.any`, and operator sugar (`UPat.var("x") * 1`).
  - Code: `uop/ops.py:1453-1562`.
- The callback receives named captures (and `ctx`). Returning `None` means no match.
- Matching:
  - `PatternMatcher.pdict` indexes rules by root op;
  - an `early_reject` set skips rules whose required child ops are absent (`uop/ops.py:1595-1619`);
  - with `UPAT_COMPILE=1` (default) each UPat is **compiled to Python source**. The compiler is itself written as UOp rewrites (`uop/upat.py:11-186`; deferred compile at `uop/ops.py:1588-1593`).
- Drivers (`uop/ops.py:1773-1890`):
  - `unified_rewrite`: a stack-based, waitlist-driven rewrite to a fixed point, bottom-up (`bpm`) and/or top-down (`pm`);
  - `walk_rewrite`: "MLIR-style Walk Pattern Rewrite Driver: single-pass".
  - CALL bodies are opaque unless `enter_calls=True`. Infinite-loop guards: `REWRITE_STACK_LIMIT=250000`, a seen-set in the fixed-point loop.
- Observability: `TRACK_MATCH_STATS` / `VIZ` record every match with timing (`uop/ops.py:1706-1756`). `PROCESS_REPLAY` captures rewrite inputs/outputs for regression testing (`1646-1652`, `1693-1701`).
- **Performance** (VERIFIED by our measurement, §11):
  - Scheduling 7 kernels (rand + 256² matmul + relu): 61.6 ms on a cache miss.
  - One Qwen3-0.6B transformer block body: 76–151 ms on a schedule-cache miss, ~1.1 ms on a hit.
  - The outer prefill graph (294 kernels): 785 ms on a miss.
  - A cold compile of a never-seen tiny kernel costs 3.7–7.9 ms end to end (schedule, 30-pass codegen, MTLCompiler, host program), parallelized across a worker pool.
  - The maintainers: compile speed is "limited largely by the runtime of the Python doing UOp rewrites. Currently it's a bit slow, but on par with torch.compile" (`docs/developer/speed.md:14-16`).

---

## 4. Lazy execution (Question 2) (VERIFIED)
- **When realization happens**: explicit `Tensor.realize()` (`tensor.py:218-225`), or any data access (`_buffer` → `contiguous().realize()`, `tensor.py:276-285`; `item/tolist/numpy`). Tensor construction from Python/numpy/disk creates realized buffers immediately (`tensor.py:57-99`).
- **How the graph is built**: each op creates a hash-consed UOp (`tensor.py:104-113`). All live Tensors are tracked (`all_tensors`, `tensor.py:17`) so that after scheduling, every Tensor referencing replaced nodes is rewired to the realized buffers (`_apply_map_to_tensors`, `tensor.py:18-32`).
- **Avoiding re-tracing**, three cache layers:
  1. the **schedule cache**, keyed by the normalized CALL body (buffers and bound Variables become PARAMs; `schedule/__init__.py:121-150`, `251-258`);
  2. the **to_program cache**, keyed by (AST key, renderer, target, config ContextVars) (`codegen/__init__.py:511-521`), plus the compiler disk cache (`device.py:339-344`) and runtime/pipeline caches (`engine/realize.py:107-112`, `runtime/ops_metal.py:241-247`);
  3. **TinyJit**, which skips Python graph building entirely on replay (§7.4).

  `@function(precompile=True)` turns a Python function into a CALL whose body is scheduled once per structural hash (`function.py:39-98`). The LLM code wraps every transformer block this way, so 28 layers share one schedule (`llm/model.py:176-183`; "cache hit" lines in `R2/llm_q4_0_debug1.txt`).
- **Per-call cost without JIT** (VERIFIED, measured):
  - a trivial `(x+1).realize()` with everything cached costs **453 µs**;
  - a schedule-cache hit for a transformer block costs ~1.1 ms of Python;
  - so an eager 28-layer decode step pays ≳30 ms of Python before any GPU work (LIKELY, by summing per-block hits).

  The JIT exists to remove this.

---

## 5. Scheduler (Question 3)

### 5.1 Where kernel boundaries are decided (VERIFIED)
The code is `schedule/indexing.py:run_rangeify` (`189-320`) plus the cleanups in `schedule/rangeify.py:get_kernel_graph` (`350-376`). The rules:

1. **Always realized** (`pm_generate_realize_map`, `schedule/indexing.py:45-54`):
   - every `STORE` (explicit assign/contiguous outputs);
   - sources of `MSELECT`/`MSTACK` (multi-device);
   - non-contiguous inputs of custom-kernel CALLs (also marked non-removable);
   - the src of a STORE whose destination appears in its own source (WAR hazard, `realize_store_after_src`).
2. **Single consumer** → the node inherits the consumer's ranges, i.e. it is fused (`schedule/indexing.py:242-244`).
3. **Multiple consumers** → if all consumers index it with identical range expressions (ignoring valid-masks), it is fused and the valid becomes the OR of the consumers' valids. Otherwise it gets fresh ranges on the mismatching axes and is **partially realized** on those axes (`schedule/indexing.py:245-268`).
4. **Ending ranges / reduce→broadcast**. When a REDUCE's output is broadcast back over the reduced axes, or an EXPAND does not just inject a range (the comment: "this is why convs are realized"), the consumer ranges are "ended". An elementwise or REDUCE node with ended ranges is fully realized (`schedule/indexing.py:218-223`, `270-277`, `294-297`). This is the classic "reduce then expand" kernel boundary (softmax = 2+ kernels; RMSNorm = reduce kernel + apply).
5. **Too many buffers**: with `MAX_KERNEL_BUFFERS` (WEBGPU=8 built-in), elementwise inputs are bufferized (`schedule/rangeify.py:170-198`).
6. **Split reduce** for large reductions with few outputs (`split_reduceop`, `schedule/prepare.py:103-125`): if `numel(x)/numel(out) ≥ 32768`, it splits into two kernels.
7. A kernel is formed wherever a `STORE`/`END` has no open (non-DEVICE) ranges (`split_store`, `schedule/rangeify.py:335-348`).

### 5.2 Recompute vs store (VERIFIED: `schedule/rangeify.py:50-102`)
Each candidate materialization (`STAGE` with `removable=True`) is tested by `remove_bufferize`. **It is kept (stored)** if any of these holds:
- it is user `contiguous()` or non-removable;
- the computation reads **>3 distinct buffers**;
- **any REDUCE inside it reads a buffer** (i.e. never recompute a reduction).

Otherwise it is removed: the producer is **recomputed** in the consumer by substituting the consumer's indices for the producer's ranges. There is no FLOP/byte cost model. The docs say: "when is it worth recomputing an arithmetic operation instead of storing… tinygrad doesn't yet have a systematic way to do it" (`docs/developer/speed.md:28-43`).

### 5.3 Known weaknesses, in the maintainers' words (VERIFIED, `docs/developer/speed.md`)
- "Model Speed (scheduler)… This is currently a big bottleneck of training speed." (`:26`)
- "tinygrad doesn't yet have a systematic way to do it [recompute vs store]." (`:43`)
- "we never copy the inputs to on chip SRAM, but this is often quite helpful for kernel speed. Also, we aren't doing a good job with L2 cache awareness" (`:59`). Confirmed in the matmul MSL: fragments load straight from `device` memory, with no `threadgroup` staging (§6.3).
- "Compile Speed (Python)… a bit slow… gets even slower if you are using BEAM" (`:14`).
- "Execution Speed (driver)… very little to improve here" (`:20-22`). Our measurements disagree for Metal at batch-1 decode granularity: 155 µs per JIT round trip, plus 4.3 µs per kernel, times 488 kernels per token (§7.5).

Observed in the LLM: **~17 kernels per transformer layer** for decode (`R2/llm_q4_0_debug1.txt`: "scheduled 17 kernels" per block). Attention, norms and RoPE produce many ~10 µs kernels (§8.3).

---

## 6. Kernel optimization (Question 4)

### 6.1 Scheduler object and OptOps (VERIFIED)
- `OptOps = {TC, SPLIT, PADTO, SWAP}` (`codegen/opt/__init__.py:6-8`). `Opt(op, axis, arg)`.
- `SPLIT(axis, (amt, new_type, top))` splits a RANGE `r` into `r_outer*amt + r_new` (or `r_new*old + r_outer` if `top`) with the new range typed `UPCAST` or `LOCAL` (`codegen/opt/postrange.py:65-73`, `94-116`). Allowed sources: `split_targets` (`postrange.py:14-15`). Meaning of each combination:
  - `UPCAST` on an output axis = old UPCAST;
  - `UPCAST` on a reduce axis = old UNROLL;
  - `LOCAL` on an output axis = old LOCAL;
  - `LOCAL` on a reduce axis = old GROUP; with `top=True`, old GROUPTOP.
- Limits:
  - upcast ≤16 on output axes and unroll ≤32 (`postrange.py:103-105`);
  - for GROUP, a shared-memory size check "prevents METAL compiler hangs" (`postrange.py:106-110`);
  - `PADTO` pads a const axis to a multiple, bounded at 4× the work (`postrange.py:127-143`);
  - `SWAP` swaps two GLOBAL axes (`postrange.py:144-150`).
- `TC` matches `REDUCE(ADD, MUL(a,b))` with matching dtypes. It splits the N/M/K axes into 2-wide pieces according to the TensorCore fragment bit layout and substitutes a `WMMA` UOp (`postrange.py:157-223`).
- Mode selection (`postrange.py:239-256`):
  1. explicit `opts_to_apply`;
  2. else BEAM if `beam>=1`;
  3. else `hand_coded_optimizations` (skipped if the kernel contains a `STAGE`).

### 6.2 hand_coded_optimizations (VERIFIED: `codegen/opt/heuristic.py`)
1. **Tensor cores first** (`:28-44`). Try TC on axis choices 0..2. Then upcast M/N by 5/4/3/2 and set LOCAL on N to 4/2. `TC_MIN_GLOBALS` variant.
2. **Matvec path** (`:60-78`). Only if `ren.has_local`, the reduce is ADD, `has_shared`, and **`reduceop.src[0]` is `MUL(INDEX, INDEX)`**, i.e. both operands are direct loads, and the first reduce range appears in idx0. Then: `SPLIT(reduce, (MV_THREADS_PER_ROW=8, LOCAL))` (group), `SPLIT(global, (MV_BLOCKSIZE=4, LOCAL))`, `SPLIT(global, (MV_ROWS_PER_THREAD=4, UPCAST))`. **A fused-dequant weight is an expression, not an INDEX, so this path never fires for quantized GEMV** (VERIFIED by the generated kernel, §8.4).
3. **Grouping** (`:80-89`). If the output size ≤2048 (≤240 on QCOM), `SPLIT(first_reduce, (16, LOCAL, top=True))`, then stop.
4. **Upcast small masked axes** (≤7, from `Tensor.stack`/pad) (`:93-109`).
5. **Upcast output axes by 3 or 4**, while prod ≥1024 and upcast_size <32, preferring axes where some buffer has stride 0 (broadcast reuse), sorted by (#strides, sum of strides) (`:111-137`).
6. **Unroll the last reduce axis**: fully if ≤32, else by 4 (`:139-150`). If nothing is upcast yet, upcast 4 (`:152-154`).
7. **Locals** (`:156-186`). Rank axes by "is an expand axis". Choose local sizes from [32 (axis 0 only), 16, 8, 4, 3, 2] with a total ≤128. At most 3 axes.

### 6.3 Tensor-core abstraction (VERIFIED)
- `TensorCore(dtype_in, dtype_out, frag_a, frag_b, frag_c)` (`renderer/tc.py:6-48`). Each fragment is (lane bits, element bits) over tile bits `m<i>/n<i>/k<i>`. Dims, thread count, and the relabeling are derived from the bit lists. This is a compact, declarative MMA layout spec reused for NVIDIA mma, AMD WMMA/MFMA and Metal.
- **Metal**: one shape **(8,8,8)** with dtype pairs (f32,f32), (f16,f32), (f16,f16), (bf16,f32), (bf16,bf16) (`renderer/tc.py:137-141`). Enabled only for `Apple>=7` (`renderer/cstyle.py:362`).
- Rendered as a helper that loads 2 elements per thread into `simdgroup_float8x8`, calls `simdgroup_multiply_accumulate`, and extracts the result (`renderer/cstyle.py:394-401`). Accumulators live in a per-thread register array and are repacked every K-step (see the MSL in `R2/debug4_matmul_relu.txt`, lines ~25-210: `float buf0[32]`, 16 WMMA calls per K iteration, `float2` direct device loads).
- **Metal 4 tensor ops / MPP / M5 neural accelerators: NOT USED.** VERIFIED. `grep -i "tensor_ops|matmul2d|MetalPerformancePrimitives|mpp::|cooperative_tensor|Apple10|MTL4"` finds only:
  - the `-std=metal4.0` compiler flag on macOS ≥26 (`runtime/ops_metal.py:56`);
  - a comment "MTL4 solves that dance" about timestamps (`runtime/ops_metal.py:121`);
  - the `MTLGPUFamilyApple10` enum value in autogen (`runtime/autogen/metal.py:1022`), unused by the renderer.

### 6.4 BEAM search (VERIFIED: `codegen/opt/search.py`)
- **Candidates** (`:15-23`):
  - SPLIT-UPCAST amt∈{0,2,3,4,5,7} × axis 0..9;
  - SPLIT-LOCAL amt∈{0,2,3,4,8,13,16,29} × axis 0..7;
  - SPLIT-LOCAL-top (GROUPTOP) amt∈{13,16,28,29,32,49,64,256} × axis 0..7;
  - optional PADTO 32 (`BEAM_PADTO`);
  - LOCAL 32 on axis 0;
  - TC (11 variants);
  - SWAP over pairs of axes 0..4.

  That is ~210 actions per expansion. Filtered by validity, `BEAM_UPCAST_MAX=256`, `BEAM_LOCAL_MAX=1024` (`:85-103`).
- **Loop** (`:113-163`). Expand every beam member by every action. Compile candidates in a **process pool** (`:116`, `:132`) with a per-compile SIGALRM timeout `BEAM_TIMEOUT_SEC=10` and a `BEAM_UOPS_MAX=3000` limit (`:55-78`). Deduplicate by **compiled binary** (`:135`). Drop candidates using >1000× the min estimated FLOPs (`:137-141`). Keep the top `amt` by time.
- **Timing** (`_time_program`, `:35-48`; `engine/realize.py:291-300`):
  - real execution with profiling timestamps;
  - `cnt=3` runs, **min** taken;
  - early stop if a run exceeds 3× the current best;
  - global size reduced to ≤65536 work items and the time extrapolated (`get_test_global_size`, `:25-33`);
  - **L2/SLC flush only if the device has `invalidate_caches`, which only NV has** (`:144`, `runtime/ops_nv.py:721`).
- **Stop** when the best improves by less than `BEAM_MIN_PROGRESS` = 0.01 µs (`:118`, `:161`).
- **Cache**: `diskcache("beam_search", {"ast": ast.key, "amt", "allow_test_size", "device", "suffix"})` → applied opts list (`:107-111`, `:171`). The key has no GPU model or driver version (LIKELY a hazard across M2→M5).
- JIT usage: `JITBEAM`/`BEAM` env sets the beam on CALLs before compile (`engine/realize.py:216-220`, `273-277`; `engine/jit.py:36`).

**Measured BEAM (§11.6):**
- **4.0–4.35 s per GEMV kernel** (cold, M2 Pro, worker pool).
- fp16 12288×4096: the heuristic kernel took 677–790 µs; the BEAM kernel took **530–535 µs** (≈189 GB/s ≈ 92% of nominal peak). Per JIT call: 751 → 709 µs (−6%; the per-call time includes the ~155 µs round trip).
- Q4_0 (fused dequant): BEAM *reported* a best of 205.96 µs, but the chosen kernel measured **~540 µs** (DEBUG=2, two runs). That is slower than the heuristic's kernel (315–510 µs across runs). The per-call DEBUG=0 median is unchanged (536 vs 530–544 µs).
- **Why** (EXPERIMENTAL/LIKELY): BEAM's cache-warm min-of-3 timing (no flush on Metal) lets a 28 MB packed-weight working set sit partly in the system-level cache. Timing then rewards kernels that are fast on cached data, not on DRAM streaming. The 100 MB fp16 case does not fit, and there BEAM's estimate matched reality (530 µs). Test: rotate among more than SLC-size distinct weight buffers, or flush with a large dummy copy between runs.

### 6.5 Cost model (VERIFIED)
`Estimates(ops, lds, mem)` is computed from the linearized UOps (`renderer/__init__.py:15-61`). It is used only for stats (GFLOPS/GB/s lines), the BEAM 1000× compute filter, and the LLM benchmark's "GB/s" print. **No analytic performance model chooses schedules or opts.** Selection is heuristics or measured search.

---

## 7. Codegen and runtime (Questions 5 and 6)

### 7.1 Lowering passes (VERIFIED: `codegen/__init__.py:275-397`)
In order:
1. multi_pm, early movement ops, load-collapse, split ranges, initial `sym`, simplify/merge ranges (`codegen/simplify.py:21-64`);
2. `apply_opts`;
3. post-opt symbolic, **expander** (UPCAST→vectors), remove reduces (→REG accumulators / LOCAL group buffers), add local buffers, **group GPU dims** (`codegen/gpudims.py:28-101`: merge/split dims to fit hardware limits, map to gid/lid);
4. expand broadcast + add LOADs, **devectorize**, symbolic, **memory coalescing**, images;
5. extra symbolic + `indexing_simplify`, lower weak index dtypes (weakint→int), final symbolic;
6. decompositions (early/dtype/late/transcendental; fast integer division unless `DISABLE_FAST_IDIV`), move gates from INDEX, renderer `extra_matcher`, cast consts, implicit barriers, RANGE→SPECIAL, **add control flow**, number params.

In total: 30 `graph_rewrite` calls plus substitutes. `to_program` caches by key (`:511-521`). `do_to_program` → linearize → render → compile (`:432-509`).

### 7.2 Metal renderer and a real example (VERIFIED)
- `MetalRenderer` (`renderer/cstyle.py:358-409`):
  - `kernel void`, `device` pointers, `threadgroup __attribute__((aligned(16)))` locals;
  - `threadgroup_barrier(mem_flags::mem_threadgroup)`;
  - `gid` = `[[threadgroup_position_in_grid]]`, `lid` = `[[thread_position_in_threadgroup]]`;
  - `precise::sin`; bf16 via `bfloat`, with SQRT/EXP2/LOG2/SIN upcast to float.
  - **All buffer pointers and scalar Variables are fields of one `args_t` struct passed as `constant args_t& args [[buffer(0)]]`** (`:402-405`). That is what lets an ICB command bind a single argument-buffer offset.
- Example 1, the command from the task (`PYTHONPATH=$S/src/tinygrad DEBUG=4 … (Tensor.rand(256,256)@Tensor.rand(256,256)).relu().realize()`; full log in `R2/debug4_matmul_relu.txt`):
  - "scheduled 7 kernels in 61.56 ms | CACHE MISS". The 7 are threefry RNG plus copies plus matmul.
  - Matmul: `r_8_2_4_32_2_4_4_32 opts: (TC axis0, SPLIT(0,4,UPCAST), SPLIT(1,4,UPCAST), SPLIT(1,4,LOCAL))`. The MSL defines `__WMMA_8_8_8_float_float` (simdgroup_float8x8), `struct args_t { device float* data0_65536; … }`, and `float buf0[32]` accumulators.
  - The K loop `for (int Lidx0 = 0; Lidx0 < 32; …)` issues 8 `float2` device loads and 16 WMMAs, then writes the accumulators back.
  - **ReLU is fused into the epilogue** (`((0.0f<val)?val:0.0f)`, stored as `float2`).
  - The kernel took 70 µs ≈ 480 GFLOPS for a tiny 256³ fp32 problem.
- Example 2, fused Q4_0 GEMV (`R2/d4_q4_0_gemv.log`), quoted in §8.4.

### 7.3 Index simplification quality (VERIFIED with examples)
- Good: matmul index math is clean shift/and (`int alu0 = (((lidx0>>3)&1)<<2);`, `int alu38 = (alu5+alu0+(Lidx0<<3)+alu1);`, one base plus constant offsets per load).
- Minor misses: unsigned constant chains are not folded in the threefry kernel (`…+val1.x+4u+1u`, `+1u+1u`). The Metal compiler will fold these, so they are harmless. LIKELY cause: committed (CAST) consts escape the bare-CONST folding rules late in the pipeline (`uop/symbolic.py:323` comment).
- Bad (in effect): in the Q4_0 GEMV the nibble selector becomes a runtime `(Lidx!=0)?16:1` **divisor** (`val/alu5`). The 2-wide nibble axis is a loop rather than unrolled, so `x / 2^(4*i)` in `q_to_uint8` (`llm/gguf.py:44-47`, implemented with `div(shift_tensor)`) cannot strength-reduce to a constant shift.

### 7.4 Runtime: compile, buffers, sync (VERIFIED: `runtime/ops_metal.py`)
- **Compile** (`:29-71`). It does not use `newLibraryWithSource`. It calls the **private `MTLCompiler.framework`** `MTLCodeGenServiceCreate("tinygrad")` and `MTLCodeGenServiceBuildRequest(…, REQUEST_TYPE_COMPILE=13, …)` with flags `-fno-fast-math -std=metal4.0 (macOS≥26) --driver-mode=metal -x metal -fmodules-cache-path=<cache_dir>`. The result is an `MTLB…ENDT` metallib, disk-cached by source (`device.py:339-344`, `CCACHE`).
- **Load**: `newLibraryWithData` → `newFunctionWithName` → pipeline descriptor with `setSupportIndirectCommandBuffers(True)` (`:241-247`).
- **Buffers**: `newBufferWithLength:options:MTLResourceStorageModeShared` (unified memory; `.contents()` gives a host mapping, `:171-183`).
  - Residency: an `MTLResidencySet` added to the queue when supported (`:202-205`, `:226-235`); otherwise `useResources:count:usage:` per encoder (`:108-109`).
  - The generic `Allocator` keeps an LRU free-list keyed by (size, spec) (`device.py:259-291`). `free` syncs the device (`runtime/ops_metal.py:178-182`).
  - Copies on unified memory are host memcpy (`runtime/support/hcq2.py:46`, `engine/realize.py:153-155`).
- **Sync**: one `MTLSharedEvent` per device. `synchronize` waits `waitUntilSignaledValue:timeoutMS:` on the timeline value (`:268-283`; `device.py:437-455`). Encoders use an `MTLFence` (`waitForFence`/`updateFence`) to serialize command buffers (`:107`, `:118`).

### 7.5 "Graph" runtime: HCQ2 + ICB (VERIFIED: `runtime/support/hcq2.py`, `runtime/ops_metal.py:100-166`, `249-266`)
- **Batching**. `sched_batches` groups consecutive enqueueable CALLs per device peer group into one batch (`hcq2.py:305-325`). Dependencies within a batch are tracked by byte ranges (`DepsTracker`, `:167-198`). Waits and signals are emitted as `INS` UOps (`:233-274`).
- **Encoding for Metal** (`MetalQueue.exec`, `runtime/ops_metal.py:133-145`). Each kernel's args (GPU addresses of buffers via `GETADDR`, plus Variable values) are laid out in **one shared argument buffer** at 256-byte-aligned offsets. Symbolic launch dims are written there too.
- `submit` (`:151-166`) creates a placeholder buffer tagged `mtl_icb`. At link time `MetalDevice.new_icb` (`:249-266`) builds an `MTLIndirectCommandBuffer`:
  - `MTLIndirectCommandTypeConcurrentDispatch`, `maxKernelBufferBindCount=1`;
  - one command per kernel: pipeline, `setKernelBuffer(argbuf, off, 0)`, static threadgroup sizes, and **`setBarrier()` on every command** (so kernels never overlap).
- Symbolic sizes are re-set per run by the host code calling `concurrentDispatchThreadgroups:threadsPerThreadgroup:` on that ICB command (`:157-160`).
- **Replay / host code**. `mtl_run` (`:100-125`) is itself a UOp function (`@uopfunc`) whose body is a sequence of `objc_msgSend` calls:
  - `commandBuffer` → `computeCommandEncoder` → `waitForFence` → [pre-Apple9 only: `setComputePipelineState` + a zero-size `dispatchThreadgroups` per distinct pipeline, "before apple9 the encoder must use the pipelines", `:111-115`] → `executeCommandsInBuffer:withRange:` → `updateFence` → `endEncoding` → `encodeSignalEvent:value:` → `commit`.
  - The whole batch submit (`hcq_submit`) is **rendered to C and compiled by clang** for `HCQ_RUNTIME_DEV=CPU` (`device.py:18`; `runtime/ops_cpu.py:90`). Real generated code is in `R2/debug4_matmul_relu.txt` lines ~218-360. For the 0.6B LLM decode batch the host C was large enough to trigger 2,199 clang warnings (`R2/llm_q4_0_default.txt`).
- **Input patching**. GPU addresses of JIT inputs are not baked in. `resolve_getaddrs` turns them into reads from an "inputs" table (`hcq2.py:511-524`). `exec_hcq` writes the current input addresses into that table every call (`engine/realize.py:185-186`). Non-input addresses are link-time constants (`hcq2.py:599-603`). Variables (e.g. `start_pos`) are passed as scalar `vals` to the host function, which writes them into the argument buffer (`engine/realize.py:160-170`, `hcq2.py:527-545`).
- **Serialization**. `hcq_fence` (`hcq2.py:409-416`): "wait for the previous run of this schedule, then announce and record this one". The generated C spins on `signaledValue` until the previous replay of the **same** batch has finished, because the argument buffer and ICB are reused. **Successive replays of one JIT cannot overlap.**
- **Limits**:
  - one argument buffer per batch;
  - ICB `maxCommandCount = len(cmds)`;
  - `MTLCommandQueue` with max 1024 command buffers (`:199`);
  - local size checked against `maxTotalThreadsPerThreadgroup` (`:260`);
  - `rtalloc_size` 64 MB bump pool for per-linear scratch (`device.py:383`, `429-434`);
  - small schedules (<`HCQ_CACHE_THRESH`=64 calls) use runtime address patches so the compiled form is cacheable (`hcq2.py:19`, `566-575`).
- **Profiling** (`PROFILE` or `DEBUG>=2`) puts every kernel in its own command buffer to read `GPUStartTime/GPUEndTime` (`runtime/ops_metal.py:162-166`, `273-279`). This inflates time: the 0.6B decode goes from 18.4 ms to 29.4 ms per token.
- **Measured dispatch cost** (`R2/overhead_and_beam.txt`):
  - TinyJit with 1 tiny kernel: **155.2 µs/call** (round trip: Python + table write + C submit + GPU launch + fence);
  - 64-kernel chain: 426.7 µs, i.e. **4.31 µs marginal per kernel** inside one ICB;
  - eager realize: 452.6 µs.

### 7.6 TinyJit (VERIFIED: `engine/jit.py`)
- Call 0: run eagerly (and realize the returned tensors) (`:128-133`).
- Call 1: **capture**. A global `capturing` list makes `create_linear_with_vars` append the scheduled `LINEAR` to the JIT instead of running it (`schedule/__init__.py:293-296`). After the function returns, all captured linears are concatenated, optionally pruned (`prune_linear` keeps only kernels depending on inputs, `:14-22`, `:152-156`), the input buffers are parametrized as PARAM slots, then memory-planned and compiled with `JITBEAM`/`BEAM` (`jit_lower`, `:29-38`). Held buffers are excluded from planning (`:158-163`).
- Call ≥2: **replay**. Check that input names and (view, variables, dtype, device) match (`:169-175`; `_prepare_jit_inputs`, `:84-102`), then `run_linear(linear, var_vals, input_uops, jit=True)` (`:70-75`).
  - An input that the JIT itself writes is copied first (`_copy_input`, `:24-27`, `:71`).
  - Returned tensors with symbolic shapes are rebound to the new Variable values (`:66-68`, `:74`).
- **Symbolic variables** (KV position): `var_vals` come from bound Variables in the inputs or passed as UOp args (`:98-100`). They flow `run_linear` → `exec_hcq` → host submit `vals` → argument buffer. One captured program serves all values of `start_pos` in `[0, max_context-1]`.

### 7.7 Memory planner (VERIFIED: `schedule/memory.py:20-62`)
- For every plannable `BUFFER` in a `LINEAR`, record first and last use (copy buffers get extra lifetime).
- Sort open/close events and suballocate with a **TLSF allocator** (`runtime/support/memory.py:23`; 256-byte blocks) into one arena per (device, copy/compute lane).
- Each buffer becomes `arena[offset:offset+nbytes].bitcast(dtype)`, a zero-copy view.
- Measured (`R2/llm_q4_0_debug1.txt`):
  - decode JIT: "memory reduced from 6.92 MB -> 1.48 MB, 457 -> 2 bufs";
  - eager prefill: "260.47 MB -> 5.11 MB, 708 -> 2 bufs".

---

## 8. LLMs on tinygrad (Question 7)

### 8.1 KV cache (VERIFIED: `tinygrad/llm/model.py`)
- The cache is an fp16 tensor `(2, B, n_kv_heads, max_context, head_dim)`, zero-initialized ("so the flash kernels can safely read whole tiles past the valid region") (`:246-252`).
- Append is an explicit STORE into a symbolic slice followed by `AFTER`: `self.cache_kv[:, :, :, start_pos:start_pos+T, :].uop.store(Tensor.stack(k, v)…)`, then `assigned_kv = Tensor(self.cache_kv.uop.after(store))` (`:217-218`). Read back with a **symbolic-length** slice `assigned_kv[0, :, :, 0:start_pos+T, :]` (`:224-225`).
- `start_pos = UOp.variable("start_pos", 0, max_context-1)` and `toks = UOp.variable("toks", 1, chunk_size)` are bound per step (`:531-532`, `:542`). Two JITs, `prefill_jit` and `rollout_jit` (`:410-412`, `:422-423`).
- The causal mask is built only when `T != 1` (`:233-236`). Attention is plain `scaled_dot_product_attention` with GQA (`:242`); no flash attention on Metal. Custom flash attention exists only for AMD RDNA3/4 hybrid models (`:219-223`).
- Sampling: Gumbel-max argmax on the GPU (`:418-420`). The CPU syncs every token via `.item()` (`:547`).

### 8.2 Quantized weights: fused, not dequantized at load (by default) (VERIFIED)
- `gguf_load` copies the packed GGUF bytes to the device and returns **lazy decode expressions** per tensor (`llm/gguf.py:232-250`). Examples: Q4_0 = `(q_to_uint8(blocks[:,2:],4).bitcast(int8)-8) * blocks[:,:2].bitcast(f16).cast(f32)` (`llm/gguf.py:52`). Formats: Q4_0/1, Q5_0/1, Q8_0, Q2_K…Q6_K, the IQ family, MXFP4, Q1_0 (`:22-38`).
- `from_gguf` lazily casts all to fp16 (`llm/model.py:432`). With `REALIZE=0` (the default, `:427`) weights stay lazy. Each block is a `@function(precompile=True, allow_implicit=True)` whose implicit inputs are the packed buffers: "we pass in the weights implicitly so we unpack the GGUF on the fly" (`:178-180`). So **dequant is fused into every consuming kernel on every token**. The code comments: "without this contiguous, it unpacks the weights from the model every time… for now it's faster" (`:512`).
- **Custom quant kernels exist only for AMD RDNA3/4**: q8 activations with `sudot4` dp4a and nontemporal loads (`llm/kernels/amd.py:27-35`, `86-105`, `107-119`). On Metal, `Linear.__call__` falls through to generic `nn.Linear` (`:105`).
- Custom-kernel escape hatch: `Tensor.custom_kernel` / `UOp.custom_kernel` (kernels written as UOps) (`tensor.py:159-165`, `uop/ops.py:1313-1316`).

### 8.3 Our decode measurement on M2 Pro (VERIFIED; commands in §11.2–11.5)
Model: `$S/models/Qwen3-0.6B-Q4_0.gguf`. 28 layers, dim 1024, FFN 3072, 16 q / 8 kv heads, head_dim 128. Tensor types: 193 × Q4_0, 3 × Q4_1, 1 × Q6_K (`token_embd`, tied as the LM head, 127.6 MB), 113 × f32. `--max_context 1024`, prompt = 1 token.

| Config | Steady decode (median) | tok/s | Weights resident | Notes |
|---|---|---|---|---|
| default (`REALIZE=0`, fused dequant) | **18.41 ms** (n=26, min 18.11, max 19.69) | **54.3** | 376 MB | est. 392 MB traffic/token → 21.6 GB/s |
| `REALIZE=1` (fp16 dequant at load) | **13.10 ms** (n=10, 13.01–13.44) | **76.3** | 1,192 MB | est. ~1.2 GB/token → ~92 GB/s |
| default under `DEBUG=2` profiling | 29.4 ms | 34 | — | one command buffer per kernel |

- First-call costs: prefill-JIT eager 7.6–10.0 s, rollout eager 2.3–3.2 s, rollout capture 1.1–1.3 s. That makes ~11–14 s to the first steady token for a 0.6B model, with compile caches partly warm (VERIFIED, `R2/llm_q4_0_*.txt`).
- **488 kernels per decode token** (`R2/prof_decode_debug2.txt`, `GlobalCounters.kernel_count`). Per-kernel breakdown of one token (`DEBUG=2`; absolute times inflated by per-kernel command buffers, 21.15 ms summed vs 18.4 ms real):

| kernel | count | total µs | % | what (LIKELY, from shapes) |
|---|---|---|---|---|
| `r_1187_32_4_4_2_2_2_32` | 1 | 5738 | 27.1 | LM head GEMV over Q6_K 151936×1024 (127.6 MB) at ~22 GB/s |
| `r_32_32_3_32_2_16` | 56 | 4483 (80/ea) | 21.2 | FFN gate/up GEMV (Q4_0 3072×1024) at ~22 GB/s |
| `r_2_8_2_64_16_2_2_16` | 28 | 1876 | 8.9 | FFN down (Q4_0/Q4_1 1024×3072) |
| `r_2048_16_2_2_16` | 28 | 1778 | 8.4 | Q projection (2048×1024) |
| `r_1024_16_{6,2,4}_2_16` | 3×28 | 3061 | 14.5 | K/V/O projections |
| `r_32_4_1187` | 2 | 761 | 3.6 | argmax/sampling over 151936 |
| ~290 kernels < 20 µs | — | ~3,100 | ~15 | norms, RoPE, attention pieces (`r_4_4_(start_pos+1)` etc.), KV store |

### 8.4 Why the fused quant GEMV is slow (VERIFIED by MSL + micro-benchmarks)
Micro-benchmark `R2/bench_gemv.py`, `y[1,12288] = x[1,4096] @ W^T`, TinyJit, median of 7×50 calls at DEBUG=0. Per-call times include the ~155 µs JIT round trip.

| W format | bytes | per call | eff. GB/s (bytes/time) | kernel-only (DEBUG=2) |
|---|---|---|---|---|
| fp16 dense, heuristic (MV path) | 100.7 MB | 751 µs | 134 | 677–790 µs (128–149 GB/s) |
| fp16 dense, BEAM=2 | 100.7 MB | 709 µs | 142 | 530–535 µs (~189 GB/s) |
| Q8_0 fused | 53.5 MB | 623 µs | 86 | (noisy) |
| Q4_0 fused, heuristic | 28.3 MB | 530–544 µs | 52–53 | 315–510 µs (varies run-to-run) |
| Q4_0 fused, BEAM=2 | 28.3 MB | 536 µs | 53 | ~540 µs |
| Q6_K fused | 41.3 MB | 1072 µs | 38.5 | ~1.1–1.2 ms |

Generated MSL for the Q4_0 kernel (`r_128_32_3_128_2_16`, opts `SPLIT(0,(3,UPCAST))`, `SPLIT(4,(0,UPCAST))`, `SPLIT(0,(32,LOCAL))`; `R2/d4_q4_0_gemv.log`):
- **One thread computes 3 full output rows over all K.** Threads in a SIMD group sit 6,912 bytes apart: `alu3 = gidx0*221184 + lidx0*6912 + Lidx0_0_0*18`. There is no cooperative reduction across lanes, so loads are uncoalesced.
- **54 scalar `unsigned char` loads per iteration**: `unsigned char val0 = *(data2+(alu3+1)); …`. uint8 is excluded from vector coalescing (`codegen/late/coalesce.py:143`).
- **Runtime integer division for nibble extraction**: `unsigned char alu5 = ((Lidx0_0_1!=0)?16u:1u); … ((val9/alu5)&15u)`. The nibble loop (`Lidx0_0_1 < 2`) is not unrolled.
- Each weight goes through 4 conversions: `int8 → float → *scale → half → *x(half) → float → +=`.
- The heuristic MV path that gives fp16 its coalesced, grouped GEMV is skipped because the weight is not a direct `INDEX` (`codegen/opt/heuristic.py:62-64`).

Cross-check with the published number (LIKELY). 178 tok/s for Qwen3.5-0.8B (Q8_0 per the CLI default) on M3 Max (≈400 GB/s) implies ~0.85–1 GB/token × 178 ≈ 150–180 GB/s, i.e. ~40% of peak. That matches our Q8_0 GEMV at 42% of the M2 Pro peak. Q4_0 (26%) and Q6_K (19%) are worse, so tinygrad's best published Apple number is likely on its *best* quant path. Their 195.7 tok/s came from the custom-kernel feature, not the generic compiler.

### 8.5 Published Apple numbers
- Maintainer statement ([the tiny corp on X](https://x.com/__tinygrad__/status/2046973311977230768)): "We set out to replicate Kimi's 193 tok/s Qwen3.5-0.8B on M3 Max. Our baseline is already 178 tok/s, beating LMStudio (160) and llama.cpp (140) out of the box, but with tinygrad's custom kernel feature Claude cranked it to 195.7!" Config not given in the snippet: quant, context, and whether llama.cpp used the same quant. Treat as a **claim**, not a verified benchmark.
- No tok/s numbers in the repo docs (grep `tok/s` in `docs/`, `README.md`: none).

---

## 9. Verdict for Sky (Question 8)

### 9.1 Copy these concepts (with the specific mechanism)
1. **One hash-consed SSA graph IR with derived types/shapes and a uniform rewrite engine.** UOp (`uop/ops.py:197-251`), PatternMatcher with root-op index, early-reject, and compiled matchers (`uop/ops.py:1595-1619`, `uop/upat.py`).

   For Sky (Rust): intern nodes in an arena (`u32` ids), derive dtype/shape/ranges as cached analyses, and write passes as declarative rules compiled to match automata. Keep tinygrad's "everything is the same IR" property, which lets the schedule, the kernels and even the command stream be rewritten and cached by structural hash.
2. **Rangeify as the fusion model.** Movement ops become index functions; consumers propagate RANGE vectors upward; a node fuses iff its consumers agree on its index expression; materialization = fresh ranges + STAGE (`schedule/indexing.py:189-320`). This is simple, general, and makes views free. Copy it, then add what tinygrad lacks: a **cost model** for recompute-vs-store and multi-output fusion. The current rule is >3 buffers / reduce-reads-buffer (`schedule/rangeify.py:84-96`).
3. **Symbolic integer layer.** Variables with `[min,max]` and `multiple_of`, interval arithmetic, `uop_given_valid`, and Invalid/valid gating instead of branches (`uop/ops.py:1014-1165`, `uop/symbolic.py`). Sky already has types; make `pos: Int<0..MAX_CTX>` a first-class compile-time-bounded runtime value that feeds exactly this machinery (bounds-check elimination, 32-bit index proofs, divisibility for vector loads).
4. **Loop-transform optimization vocabulary**: SPLIT-to-{UPCAST, LOCAL, top}, TC, PADTO, SWAP over typed RANGE axes (`codegen/opt/postrange.py`). This is a small, composable, searchable space. Keep the persistent tuning cache keyed by kernel hash, but **add GPU model, OS and driver version to the key** (tinygrad's key has device name only, `codegen/opt/search.py:107`).
5. **Declarative MMA fragment layouts** (`TensorCore` lane/element bit maps, `renderer/tc.py:6-48`). This is a good template for 8×8 simdgroup MMA. For Metal 4 `matmul2d` / M5 neural accelerators the layout may be opaque (cooperative tensors), so this needs a different abstraction (EXPERIMENTAL; not in tinygrad).
6. **Runtime replay design**: prebuilt ICB + **single argument buffer** + per-call patching of only (input addresses, scalar Variables, symbolic dims) (`runtime/ops_metal.py:127-166`, `249-266`; `hcq2.py:511-524`). Residency sets. Shared-storage buffers. Host submission as compiled code. Sky can generate this command stream AOT or at model load.
7. **Static memory planning** (lifetime + TLSF into arenas, `schedule/memory.py`) and **schedule caching by structural hash with Variables normalized away** (`schedule/__init__.py:251-258`). Plus `@function(precompile)`-style per-layer bodies, so N layers are compiled once (`function.py`, `llm/model.py:176-183`).
8. **Pattern-based debugging infrastructure**: VIZ match traces, `PROCESS_REPLAY` (golden rewrite replay), `SPEC` verifiers at every stage (`uop/spec.py`). These are cheap to copy and valuable for a compiler.

### 9.2 Avoid / where tinygrad is weak (and Sky can win) — batch-1 decode on Apple GPUs
1. **Quantized GEMV via generic fusion.** VERIFIED. 19–43% of peak bandwidth (Q6_K 38.5, Q4_0 53, Q8_0 86 GB/s vs ~190 achievable). Their own fastest Apple result needed a custom kernel. Sky needs **first-class quantized matvec**:
   - vectorized uint32/uint4 loads of packed blocks;
   - constant-shift nibble unpack;
   - lane-parallel K-reduction with `simd_sum`;
   - several rows per simdgroup;
   - fp16/fp32 mixed math with minimal conversions;
   - a layout transform at load time (repack GGUF/MLX blocks into a GPU-friendly interleave) where it helps.

   The decode roofline at Qwen3-32B Q4 (~18–19 GB of weights) is bandwidth. Every % of streaming efficiency is a % of tok/s.
2. **Too many kernels.** VERIFIED. 488 kernels/token at 28 layers (~17/layer) at ≥4.3 µs serialized each, ≈2.1 ms/token of pure dispatch + barrier cost on M2 Pro (LIKELY; marginal-cost × count). At 64 layers (Qwen3-32B) a similar structure gives ~1,100 kernels and ~5 ms/token (LIKELY extrapolation). That is ~10–15% of a ~35–45 ms bandwidth floor (LIKELY). Target ≤6–8 kernels/layer for Sky:
   - RMSNorm folded into the next GEMV prologue;
   - QKV as one GEMV;
   - RoPE + KV-append fused;
   - one decode-attention kernel (split-K over context when long);
   - gate+up+SiLU fused;
   - down + residual.
3. **Every ICB command has a barrier** (`runtime/ops_metal.py:263`), so there is no concurrency between independent kernels (e.g. K and V projections, attention heads). Sky should emit barriers only on true dependencies (`MTLIndirectCommandBuffer` + `setBarrier` selectively, or concurrent dispatch with `memoryBarrierWithScope` only where needed). EXPERIMENTAL gain.
4. **Replay serialization** (`hcq_fence`) and a ~155 µs JIT round trip on M2 Pro. VERIFIED. Decode is serial anyway (sampling feeds the next token), but Sky should:
   - double-buffer argument buffers so token t+1's submission (and CPU-side sampling/tokenizer work) overlaps token t;
   - keep sampling on the GPU and feed the token id GPU-side (tinygrad already samples on the GPU but syncs via `.item()` every token, `llm/model.py:547`).
5. **BEAM methodology on Apple.** VERIFIED that there is no cache flush on Metal and that the result was mis-ranked in our Q4_0 test. Timing uses min-of-3, test-size extrapolation, no DVFS control, and costs ~4 s per kernel. Sky's autotuner should:
   - stream DRAM-sized working sets (rotate buffers or flush);
   - report medians;
   - pin a warm GPU clock (run a warmup load);
   - prune with an analytic bandwidth/occupancy model before timing.
6. **Python compile latency.** VERIFIED: 7.6–10 s first prefill call, 76–151 ms per block-schedule miss, ~1.1 ms per cache hit, 785 ms outer schedule, 3.7–7.9 ms per kernel cold. A Rust compiler doing the same rewrites should be 10–100× faster (LIKELY; typical Python vs Rust for pointer-chasing graph code). TTFT can be dominated by compile unless Sky AOT-compiles per (model, target).
7. **No threadgroup staging, no Metal 4 tensor ops.** VERIFIED (§6.3; `docs/developer/speed.md:59`). Prefill (compute-bound) on M5 needs threadgroup-tiled GEMM and, if available, M5 neural-accelerator matmul (`mpp::tensor_ops::matmul2d`). tinygrad offers nothing here, so it is an open field for Sky (EXPERIMENTAL; verify API availability via other agents).
8. **Recompute-vs-store and fusion are heuristics** with documented gaps (`speed.md:26-43`). Sky sees the whole program and can solve fusion globally: a cost model with bytes moved vs FLOPs, on-chip capacity, and launch cost.

### 9.3 Mapping tinygrad's ideas onto a statically typed, whole-program compiled language
- **Runtime laziness is not needed** for the model graph. In tinygrad, laziness exists (a) to see fusion opportunities through an eager Python API and (b) to capture graphs for the JIT. Sky sees the whole dataflow at compile time, so it can build the UOp-like graph AOT, rangeify/fuse AOT, generate kernels and the command stream AOT, and memory-plan statically (decode shapes are fixed except for position).
- **What still needs runtime specialization:**
  1. **Cache position / sequence length** (`start_pos`, `T`, attention length `pos+T`). tinygrad uses bounded symbolic Variables passed as kernel args plus symbolic launch dims patched per replay (VERIFIED, §7.5–7.6). Sky: compile kernels parametric in `pos ∈ [0, max_ctx)`, with a small set of specializations:
     - decode `T=1` vs prefill chunk sizes (bucketed, e.g. 32/128/512);
     - attention split-K variants by context bucket (more parallelism at long context);
     - mask-free decode path (tinygrad already skips the mask when `T==1`).
  2. **Device-specific tuning** (M2 vs M5; Apple8 vs Apple10 features). Either AOT per target, or a first-run tuning step with a persistent cache keyed by kernel hash + GPU identity.
  3. **Weight format/layout.** Known at model-load time. Make it a compile-time parameter of the model artifact; repack at load if needed.
  4. **Buffer addresses.** Patch per call only for true inputs/outputs; everything else is baked at load (tinygrad's link-time patches).
- **Static types help where tinygrad pays at runtime**:
  - shape/dtype errors at compile time (tinygrad's `spec` verifiers and shape checks run in Python per node);
  - index-width proofs (tinygrad needs `vmin/vmax` analysis and `pm_lower_weak` to pick int32);
  - guaranteed fusion boundaries;
  - no per-call graph tracing.

### 9.4 Idea → concrete evidence in the code

| Idea | Evidence (VERIFIED) | Verdict for Sky |
|---|---|---|
| UOp graph IR | `uop/ops.py:240-247` (fields), `122-195` (derived dtype), `197-213` (hash-consing), `uop/__init__.py` (Ops) | Copy the design (arena-interned SSA, derived analyses) |
| Lazy execution | `tensor.py:104-113`, `218-225`, `_apply_map_to_tensors` `18-32` | Not needed for the static model graph; keep only for an interactive/REPL mode |
| Scheduler | `schedule/indexing.py:189-320`, `schedule/rangeify.py:50-102`, `335-376`, `schedule/__init__.py:31-79` | Copy rangeify; replace heuristics with a cost model |
| BEAM | `codegen/opt/search.py:15-173`; no Metal cache flush `:144` | Copy the search space + persistent cache; fix timing; add an analytic pre-filter |
| TinyJit | `engine/jit.py:104-178`; capture hook `schedule/__init__.py:293-296` | Replace with AOT command-stream generation; keep the replay-with-patching runtime |
| Kernel codegen | `codegen/__init__.py:275-397` (30 passes), `renderer/cstyle.py:358-409` | Copy the staged lowering; add quant-aware templates and threadgroup tiling |
| Symbolic shape/index math | `uop/ops.py:1014-1181`, `uop/symbolic.py`, `schedule/indexing.py:154-187` | Copy; tie to Sky's type system (bounded ints) |
| Tensor-core abstraction | `renderer/tc.py:6-48`, `137-141`; `codegen/opt/postrange.py:157-223` | Copy for simdgroup 8×8; add a Metal 4 / M5 path tinygrad lacks |
| Recompute vs store | `schedule/rangeify.py:84-96`; `docs/developer/speed.md:28-43` | Improve (global cost model) |
| Backend/runtime separation | `device.py:375-455` (Compiled/Allocator/Compiler), `runtime/ops_*.py`, `Renderer` | Copy the separation; Sky's backends are Cranelift (CPU) plus an MSL/AIR emitter |
| Views vs buffers | `uop/ops.py:781-801` (`base`, `storage_base`), `931-997` (`contiguous_view`, `buffer`), `device.py:252-254` | Copy: views are index math; byte-range views are zero-copy |
| Memory planning | `schedule/memory.py:20-62`, TLSF `runtime/support/memory.py:23` | Copy (static at compile time for decode) |
| Zero-copy | Shared storage `runtime/ops_metal.py:174`; buffer views; memory-planner arenas are views; host-memcpy copies on unified memory (`hcq2.py:46`); GGUF bytes kept packed on device | Copy. Also mmap→`newBufferWithBytesNoCopy` for weights, which tinygrad does **not** do (its `external_ptr` path wraps an existing MTLBuffer, `runtime/ops_metal.py:173`; model load ran at 0.15–0.32 GB/s, LIKELY a cold-file plus copy path) |

---

## 10. Open questions / EXPERIMENTAL items
1. Exact cause of the BEAM Q4_0 mis-ranking (SLC residency vs DVFS vs test-size). Test: flush between timed runs, rotate buffers larger than the SLC, use median not min.
2. How much the 488→~200 kernel reduction (fusion) would save on Apple GPUs. Test: hand-fuse in a micro-harness and measure the marginal per-kernel cost on M5 (it may differ from M2 Pro's 4.3 µs).
3. Whether selective ICB barriers give real overlap on Apple GPUs for small decode kernels (EXPERIMENTAL).
4. tinygrad on M5/Apple10: no special path exists, so performance would follow the generic codegen. Not measurable here (no M5).
5. The tiny corp's 178 / 195.7 tok/s claim: quant type and settings unverified. It could be reproduced on an M3/M4 Max with `python -m tinygrad.llm --model qwen3.5:0.8b --benchmark` (Q8_0 by default).
6. Qwen3-8B Q4_0 on tinygrad (local file exists: `$S/models/Qwen3-8B-Q4_0.gguf`, 4.8 GB). Not run: the 16 GB machine and benchlock time budget, plus the expected ~4 s+ first call and long capture. Expected from the per-format GEMV bandwidth: Q4_0 at ~53 GB/s on ~4.6 GB of weights gives ~90 ms/token ≈ 11 tok/s. This is LIKELY only; needs a measurement.

---

## 11. Measurements: exact commands and raw output
Common environment:
```
S=/private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad
export PYTHONPATH=$S/src/tinygrad
```
From §11.3 onward, also `export XDG_CACHE_HOME=$S/research/R2/xdg` (fresh compile/BEAM cache). Runs §11.1–11.2 used the default `~/Library/Caches/tinygrad` (a cache.db already existed, so the compile cache may have been partly warm). All runs went through `bash $S/benchlock.sh …`.

### 11.1 DEBUG=4 matmul+relu (codegen example)
```
cd $S/research/R2 && PYTHONPATH=$S/src/tinygrad DEBUG=4 timeout 170 bash $S/benchlock.sh $S/venv/bin/python -c "from tinygrad import Tensor; (Tensor.rand(256,256)@Tensor.rand(256,256)).relu().realize()" > debug4_matmul_relu.txt 2>&1
```
Raw (excerpt):
```
scheduled     7 kernels in    61.56 ms | CACHE MISS 53252200 |     773 uops in cache
r_8_2_4_32_2_4_4_32       opts: (Opt(op=OptOps.TC, axis=0, arg=(-1, 0, 1)), Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.LOCAL)))
float2 __WMMA_8_8_8_float_float(float2 a, float2 b, float2 c){ simdgroup_float8x8 mat_a, mat_b; ... simdgroup_multiply_accumulate(mat_c, mat_a, mat_b, mat_c); ...}
struct args_t { device float* data0_65536; device float* data1_65536; device float* data2_65536; };
kernel void r_8_2_4_32_2_4_4_32(constant args_t& args [[buffer(0)]], uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  ... float buf0[32]; ... for (int Lidx0 = 0; Lidx0 < 32; Lidx0++) { ... float2 val0 = (*((device float2*)((data1_65536+(alu38+2048))))); ... float2 wmma0 = __WMMA_8_8_8_float_float(val0, val4, float2((*(buf0+17)),(*(buf0+21)))); ...
  float alu73 = ((0.0f<val8)?val8:0.0f); ... *((device float2*)((data0_65536+(alu105+16)))) = float2(alu81,alu82);
void hcq_submit(volatile unsigned long* restrict slots_0_24, unsigned long* restrict inputs_1_6, volatile unsigned char* restrict None_2_1152, ...) {
  ... hcq_fence(slots_0_24, timeline_3_2, val6, handles_4_5); ... *(((unsigned long*)(None_2_1152))+0) = val0; ...
  for (unsigned long Lidx4 = 0; Lidx4 < 4ul; Lidx4++) { mtl_run(None_2_1152, Lidx4, ...); }  mtl_run_1(None_2_1152, alu15, ...); }
*** METAL      7 r_8_2_4_32_2_4_4_32   arg  3 mem   0.00 GB tm     70.12us/     3.83ms (    480 GFLOPS   11|64     GB/s)
```

### 11.2 LLM decode benchmark, default (fused dequant)
```
cd $S/research/R2 && ( time PYTHONPATH=$S/src/tinygrad timeout 175 bash $S/benchlock.sh $S/venv/bin/python -m tinygrad.llm --model $S/models/Qwen3-0.6B-Q4_0.gguf --max_context 1024 --benchmark 30 ) > llm_q4_0_default.txt 2>&1
```
Raw (excerpt; `real` includes lock wait; `user 0m17.359s`):
```
loaded model "Qwen3-0.6B" at 0.15 GB/s with 376,204,288 bytes and 751,632,384 params, max context 1024 on METAL
10020.02 ms,   0.10 tok/s,    0.06 GB/s, 564/561 MB
3158.64 ms,   0.32 tok/s,    0.12 GB/s, 391/561 MB
1259.40 ms,   0.79 tok/s,    0.31 GB/s, 391/562 MB
 19.84 ms,  50.39 tok/s,   19.76 GB/s, 392/562 MB
 18.50 ms,  54.06 tok/s,   21.21 GB/s, 392/562 MB
 18.16 ms,  55.07 tok/s,   21.62 GB/s, 392/562 MB
 ... (26 steady samples: median 18.41 ms, min 18.11, max 19.69 → 54.3 tok/s)
```

### 11.3 LLM decode, REALIZE=1 (fp16 weights realized at load)
```
cd $S/research/R2 && bash $S/benchlock.sh bash -c "REALIZE=1 timeout 170 $S/venv/bin/python -m tinygrad.llm --model $S/models/Qwen3-0.6B-Q4_0.gguf --max_context 1024 --benchmark 14 > llm_q4_0_realize1.txt 2>&1"
```
Raw:
```
loaded model "Qwen3-0.6B" at 0.32 GB/s with 1,192,099,840 bytes and 751,632,384 params, max context 1024 on METAL
6527.41 ms ... 1232.90 ms ... 952.36 ms
 13.18 ms,  75.87 tok/s,   91.63 GB/s, 1207/1378 MB
 13.01 ms,  76.89 tok/s,   92.88 GB/s, 1208/1378 MB
 ... (10 steady samples: median 13.10 ms, 13.01–13.44 → 76.3 tok/s)
```

### 11.4 Schedule/JIT timings (DEBUG=1)
```
cd $S/research/R2 && bash $S/benchlock.sh bash -c "DEBUG=1 timeout 170 $S/venv/bin/python -m tinygrad.llm --model $S/models/Qwen3-0.6B-Q4_0.gguf --max_context 1024 --benchmark 5 > llm_q4_0_debug1.txt 2>&1"
```
Raw (excerpt):
```
scheduled   310 kernels in   107.21 ms | CACHE MISS f59e6f94     (weight load)
scheduled    25 kernels in   151.07 ms | CACHE MISS 2af0e17a     (prefill block body)
scheduled    25 kernels in     1.12 ms |  cache hit 2af0e17a
scheduled    25 kernels in    95.50 ms | CACHE MISS c5a5170b     (2nd distinct block body)
scheduled    25 kernels in     1.09 ms |  cache hit c5a5170b      (x26)
scheduled   294 kernels in   784.81 ms | CACHE MISS 014eb050     (outer prefill graph)
memory reduced from 260.47 MB -> 5.11 MB, 708 -> 2 bufs
7566.56 ms,   0.13 tok/s
scheduled    17 kernels in   106.05 ms | CACHE MISS 7996e8e6     (decode block body)
scheduled    17 kernels in     1.37 ms |  cache hit 7996e8e6
scheduled    17 kernels in    76.29 ms | CACHE MISS 374e9be4
scheduled    39 kernels in    61.12 ms | CACHE MISS 42eed0fd     (outer decode graph)
memory reduced from 6.31 MB -> 0.61 MB, 456 -> 2 bufs
2339.24 ms,   0.43 tok/s
JIT captured 1 linears with 2 inputs
memory reduced from 6.92 MB -> 1.48 MB, 457 -> 2 bufs
1130.30 ms,   0.88 tok/s
 18.08 ms,  55.31 tok/s
 17.80 ms,  56.18 tok/s
```

### 11.5 Per-kernel decode profile (DEBUG=2)
```
cd $S/research/R2 && PYTHONPATH=$S/src/tinygrad DEBUG=2 bash $S/benchlock.sh timeout 170 $S/venv/bin/python prof_decode.py $S/models/Qwen3-0.6B-Q4_0.gguf 6 > prof_decode_debug2.txt 2>&1
```
Raw:
```
LOAD_S 1.076
TOKEN 0 wall_ms 7906.80 kernels 969 global_mem_MB 564.2 gpu_sum_ms 62.94
TOKEN 1 wall_ms 2324.62 kernels 487 global_mem_MB 391.4 gpu_sum_ms 40.73
TOKEN 2 wall_ms 1220.96 kernels 487 global_mem_MB 391.7 gpu_sum_ms 40.57
TOKEN 3 wall_ms 29.34 kernels 488 global_mem_MB 392.0 gpu_sum_ms 27.62
TOKEN 4 wall_ms 29.49 kernels 488 global_mem_MB 392.4 gpu_sum_ms 27.66
TOKEN 5 wall_ms 29.39 kernels 488 global_mem_MB 392.7 gpu_sum_ms 27.71
```
Aggregation of the final token is in the table in §8.3 (script inline in this session; 488 kernels parsed; METAL sum 21.15 ms; p10/p50/p90 = 9.3/11.4/77.2 µs; 287 kernels < 20 µs).

### 11.6 GEMV micro-benchmarks and BEAM (`bench_gemv.py`)
```
# default heuristics
bash $S/benchlock.sh bash -c "for m in fp16 q4_0; do timeout 80 $S/venv/bin/python bench_gemv.py \$m 12288 4096 50 2>&1 | grep -a RESULT; done; timeout 60 $S/venv/bin/python bench_gemv.py fp16 1024 1024 200 2>&1 | grep -a RESULT"
RESULT mode=fp16 N=12288 K=4096 warmup_s=0.56 per_call_us_median=751.3 min=744.0 max=824.7 weight_MB=100.66 implied_GBps=134.0
RESULT mode=q4_0 N=12288 K=4096 warmup_s=0.35 per_call_us_median=544.4 min=464.4 max=909.4 weight_MB=28.31 implied_GBps=52.0
RESULT mode=fp16 N=1024 K=1024 warmup_s=0.20 per_call_us_median=183.3 min=181.5 max=234.3 weight_MB=2.10 implied_GBps=11.4
# cold BEAM search (DEBUG=2 for progress; per-call numbers here are inflated by DEBUG=2)
bash $S/benchlock.sh bash -c "for m in q4_0 fp16; do JITBEAM=2 IGNORE_BEAM_CACHE=1 DEBUG=2 timeout 170 $S/venv/bin/python bench_gemv.py \$m 12288 4096 50 > beam2_\$m.log 2>&1; done"
RESULT mode=q4_0 ... warmup_s=4.35 ...   (BEAM final: "3.78s: 205.96us from 51 -> 51 actions")
RESULT mode=fp16 ... warmup_s=4.02 ...   (BEAM final: "3.76s: 530.58us from 26 -> 24 actions")
# cached BEAM at DEBUG=0, and kernel-only times at DEBUG=2
RESULT jit1_us=155.2 jit64_us=426.7 marginal_per_kernel_us=4.31 eager1_us=452.6           (bench_overhead.py)
RESULT cold64 trial=0 build_ms=1.59 realize_ms=504.7 per_kernel_ms=7.89
RESULT cold64 trial=1 build_ms=1.71 realize_ms=238.3 per_kernel_ms=3.72
RESULT cold64 trial=2 build_ms=1.71 realize_ms=486.2 per_kernel_ms=7.60
BEAMCACHED RESULT mode=fp16 N=12288 K=4096 warmup_s=0.20 per_call_us_median=709.1 ... implied_GBps=142.0
BEAMCACHED RESULT mode=q4_0 N=12288 K=4096 warmup_s=0.30 per_call_us_median=536.3 ... implied_GBps=52.8
== d2_default_fp16   *** METAL r_128_32_3_1024_4      tm 789.75us / 709.92us / 677.29us
== d2_default_q4_0   *** METAL r_128_32_3_128_2_16    tm 317.79us / 318.25us / 315.00us
== d2_beam_fp16      *** METAL r_192_16_16_2_2_64_4   tm 531.62us / 534.83us / 530.00us
== d2_beam_q4_0      *** METAL r_768_16_16_8_2_16     tm 542.62us / 540.00us / 540.38us
# other formats (ggml type ids: 8=Q8_0, 2=Q4_0, 14=Q6_K; random packed bytes)
bash $S/benchlock.sh bash -c "for m in ggml8 ggml2 ggml14; do timeout 60 $S/venv/bin/python bench_gemv.py \$m 12288 4096 50 2>&1 | grep -a RESULT; DEBUG=2 ...; done"
RESULT mode=ggml8 N=12288 K=4096 warmup_s=0.49 per_call_us_median=623.2 min=606.2 max=750.5 weight_MB=53.48 implied_GBps=85.8
RESULT mode=ggml2 N=12288 K=4096 warmup_s=0.28 per_call_us_median=530.4 min=461.0 max=907.3 weight_MB=28.31 implied_GBps=53.4
RESULT mode=ggml14 N=12288 K=4096 warmup_s=0.70 per_call_us_median=1071.5 min=1070.3 max=1171.9 weight_MB=41.29 implied_GBps=38.5
   (DEBUG=2 kernel-only in this run: ggml8 1208-1216us, ggml2 507-511us, ggml14 1098-1179us — note ggml2 differs from the 315us seen in the earlier run: single-kernel DEBUG=2 timings vary ~±50% between runs, LIKELY GPU DVFS while the CPU idles between synchronized calls; use the DEBUG=0 per-call medians for comparisons)
# MSL of the fused Q4_0 GEMV
bash $S/benchlock.sh bash -c "DEBUG=4 timeout 60 $S/venv/bin/python bench_gemv.py q4_0 12288 4096 1 > d4_q4_0_gemv.log 2>&1"
```
Caveats:
- The GEMV per-call numbers include ~155 µs of JIT round trip. That is negligible at 100 MB and significant at 28 MB.
- DEBUG=2 kernel times include per-command-buffer overheads and DVFS effects.
- Other agents may run non-benchmark GPU work outside the lock.
- Side effect: the §11.1–11.2 runs used `~/Library/Caches/tinygrad/cache.db`, which existed before our runs.
