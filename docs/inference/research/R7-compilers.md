# R7: What other compilers and inference systems teach Sky

*Research notes for the Sky LLM-inference technical report. Agent R7, compiled 2026-10-01.*

**Scope.** This file covers Triton, TVM/Ansor/MetaSchedule, Halide, XLA/JAX, PyTorch 2, MLIR/IREE, tensor superoptimization and e-graphs, megakernels, prior art on Apple GPUs, attention algorithms, autotuning, CPU codegen, and AI kernel generation.

For each system it records:
- what Sky should borrow;
- what Sky should avoid;
- which systems already did what Sky wants, and how well that went.

**Workload.** The headline workload is Qwen3-32B, 4-bit, batch 1, on an M5 Max.

**Evidence labels:**
- **[VERIFIED: source]** means I read the source in this session: a paper or abstract, an official document, source code with a path or commit, or a maintainer statement.
- **[LIKELY: reason]** is an inference, with the reason given.
- **[EXPERIMENTAL]** is a proposal that has to be measured.
- **[MOONSHOT]** is a speculative idea.

The deep dives on MLX, llama.cpp, tinygrad and Metal hardware belong to other agents; this file only cross-references them.

**Method.** I used primary sources where I could:
- arXiv abstracts and full text;
- official blogs and documentation;
- GitHub source, read through `gh` at named commits;
- the local clones under `$S/src`.

Late in the session WebFetch and WebSearch hit a rate limit. After that I verified only through `gh` and raw GitHub files. Where a number came from a secondary source, it is labelled LIKELY, not VERIFIED.

Local clone commits used:
- metal-flash-attention `8671cdd` (2024-09-23)
- metal-benchmarks `dc2adc6`
- mlx `0840c42` (2026-10-01)
- tinygrad `512e9b2` (2026-10-02)

---

## Contents

0. [Executive summary and the workload's physics](#0-executive-summary-and-the-workloads-physics)
1. [Topic 1: Triton (and Inductor templates)](#topic-1--triton-and-inductor-templates)
2. [Topic 2: TVM, Ansor, MetaSchedule, Relax, MLC-LLM](#topic-2--tvm-ansor-metaschedule-relax-mlc-llm)
3. [Topic 3: Halide and scheduling languages](#topic-3--halide-and-scheduling-languages)
4. [Topic 4: XLA/JAX and PyTorch 2; what owning the language buys](#topic-4--xlajax-and-pytorch-2)
   - [Design sketch: a Sky "what vs how" split for tensors](#design-sketch--a-sky-what-vs-how-split-for-tensors)
5. [Topic 5: MLIR, IREE and paths to Apple GPUs (decision)](#topic-5--mlir-iree-and-paths-to-apple-gpus)
6. [Topic 6: Tensor-graph superoptimization and e-graphs](#topic-6--tensor-graph-superoptimization-and-e-graphs)
7. [Topic 7: Megakernels and persistent execution](#topic-7--megakernels-and-persistent-execution)
8. [Topic 8: Apple-GPU prior art (negative results)](#topic-8--apple-gpu-prior-art-negative-results)
9. [Topic 9: Attention algorithms](#topic-9--attention-algorithms)
10. [Topic 10: Cost models and autotuning in practice](#topic-10--cost-models-and-autotuning-in-practice)
11. [Topic 11: CPU codegen: Cranelift vs LLVM vs C](#topic-11--cpu-codegen-cranelift-vs-llvm-vs-c)
12. [Topic 12: Superoptimization and AI-assisted kernel generation](#topic-12--superoptimization-and-ai-assisted-kernel-generation)
13. [Borrow / avoid / decision table](#13-borrow--avoid--decision-table)
14. [The 10 most important lessons for Sky](#14-the-10-most-important-lessons-for-sky)
15. [Open questions](#15-open-questions)
16. [Sources](#16-sources)
17. [Architect brief](#17-architect-brief)

---

## 0. Executive summary and the workload's physics

### 0.1 The numbers that bound everything

The model config was read from Hugging Face `Qwen/Qwen3-32B/config.json` **[VERIFIED: https://huggingface.co/Qwen/Qwen3-32B/blob/main/config.json]**:

| Parameter | Value |
|---|---|
| layers | 64 |
| hidden | 5120 |
| query heads | 64 |
| KV heads | 8 (GQA ratio 8) |
| head_dim | 128 |
| MLP intermediate | 25600 |
| vocab | 151936 |
| embeddings | untied |
| rope_theta | 1e6 |
| max_position_embeddings | 40960 |

The MLX community 4-bit build uses `group_size 64, bits 4` **[VERIFIED: https://huggingface.co/mlx-community/Qwen3-32B-4bit]**.

The calculation script is `research/R7-work/qwen3_32b_budget.py`. Its arithmetic is exact. Its bandwidth figures assume peak bandwidth.

| Quantity | Value |
|---|---|
| Linear-layer parameters (64 layers) | 31.21 B (487.6 M per layer) |
| Embedding / lm_head | 0.778 B each |
| Bytes read per decode token at 4.5 bits/weight (MLX affine g64: 4 bits + fp16 scale + bias per 64) | **17.99 GB** |
| Bytes per token at an ideal 4.0 bits/weight | 15.99 GB |
| M5 Max memory bandwidth (Apple) | up to **614 GB/s** **[VERIFIED: https://www.apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/]** |
| **Decode ceiling at short context** (614 / 17.99) | **34.1 tok/s**; 27.3 tok/s at 80% of peak |
| KV cache per token (fp16 K+V, 64 layers × 8 heads × 128) | **256 KiB**; 10.7 GB at the 40,960-token native context |
| Context at which KV reads equal 10% / 25% / 50% / 100% of the weight bytes (fp16 KV) | 6.9 k / 17.2 k / 34.3 k / 68.6 k tokens |
| Prefill FLOPs, linear layers | 2 × 31.2 B × P (≈256 TFLOP at P = 4096) |
| Prefill attention FLOPs (causal) | 0.9% of the total at P = 512; 6.4% at 4096; 21.6% at 16384 |
| Decode GEMV arithmetic intensity | ≈3.6 FLOP/byte, strongly bandwidth-bound |

**Consequences.** All four are [LIKELY], following from the arithmetic above.

1. **Batch-1 decode is a bytes problem.** No scheduling cleverness beats 34 tok/s on this model at 4.5 bits/weight on an M5 Max. To beat MLX by a wide margin, a competitor must do at least one of three things:
   - run at a much higher fraction of the bandwidth than MLX does;
   - move fewer bytes, through the weight format or the KV format;
   - do fewer weight passes per generated token (speculative decoding: one weight pass verifies several tokens).
2. **Prefill and TTFT are compute problems.** On an M5 they run on the per-core Neural Accelerators, which are reachable from MSL through Metal 4 TensorOps (Topics 5 and 8). TTFT is the metric a compiler can move most.
3. **Long context changes the picture.** Beyond about 7 k tokens, attention KV traffic exceeds 10% of the weight traffic. KV-cache format and decode-attention efficiency then matter (Topic 9).
4. **RAM is weights plus KV.** That is about 18.4 GB of weights including the embedding table. Add 256 KiB per context token in fp16, or about half that with 8-bit KV. KV-cache allocation policy is a RAM lever.

### 0.2 What prior art says about the headroom

- **The best published native Metal runtime does not beat MLX on decode for a large dense model.** BaseRT (July 2026) uses hand-written Metal 4 tensor-core kernels. On an M5 Pro with dense Qwen3.6-27B at 4-bit it measured:
  - decode: **18.1 tok/s**, vs MLX **18.4** and llama.cpp 15.4;
  - prefill: 440 vs 305 tok/s (MLX) at pp128, but only 497 vs 486 at pp2048.
  - Its large decode wins (1.33× over MLX) are on sub-1B models only.

  **[VERIFIED: https://arxiv.org/abs/2607.19438, Tables 1–2]**
- **The newest Apple-silicon megakernel project aims for only 10% over MLX on decode.** Monolith/mpk-apple was opened 2026-09-24 by the author of MPK and Mirage. Its v1 success metric for plain decode is "≥ 1.10× the better of MLX / llama.cpp", with a stretch goal of "≥ 80% of the chip's bandwidth bound" **[VERIFIED: https://github.com/jiazhihao/mpk-apple/issues/1]**.
- **An MLIR compiler with an M5 matrix path is still slightly behind MLX.** Modular's Mojo/MAX is MLIR-based and open-sourced on 2026-08-18. Its M5 W4A16 matmul mix runs at **0.85–0.96× MLX** **[VERIFIED: https://github.com/modular/modular/blob/main/max/kernels/src/linalg/matmul/gpu/apple/FP4_DESIGN.md]**.

**Bottom line [LIKELY]:**
- For Qwen3-32B decode, kernels and scheduling alone are worth roughly 1.0–1.2× over MLX.
- Short-prompt prefill/TTFT is worth roughly 1.2–1.5×. BaseRT shows 1.44× at pp128.
- Long-context decode, through KV format and attention, is worth more than 1.2×.
- Speculative decoding changes the work per token and is the largest decode lever. It is outside this topic, but the compiler must support it (static verify-k-tokens graphs).

### 0.3 Recommendations in one paragraph

Build Sky's own small stack:
1. **Sky IR**: shapes, dtypes, quant formats and layouts are types.
2. **Graph rewrites and fusion partitioning**: e-graph for algebra and layout, then a cost-model-driven fusion partitioner.
3. **Tile-level kernel IR**: a Triton/ThunderKittens-like block model with a small set of templates.
4. **MSL text**: compiled by Apple's in-OS Metal compiler.

Around that stack:
- An **AOT static execution plan**: a static memory plan, plus a single pre-encoded command stream or ICB per context bucket.
- A **per-device autotuning database**, which already exists as sky-tune.
- **Do not put MLIR in the core.** No production MLIR path reaches Apple's matrix units except Mojo's undocumented-AIR route, and the C++/LLVM dependency weight is large. Revisit MLIR, Triton or Mojo only for future non-Apple backends.
- **Make numerics contracts first-class**: tolerance-based "Relaxed" contracts that gate format and algebra changes. Together with weights-as-constants layout specialization, this is Sky's real IP. Kernel heroics alone are not.


---

## Topic 1 — Triton (and Inductor templates)

### Mechanics

**Programming model.** Triton is SPMD over *programs*. One program is one thread block (CTA) and operates on whole tiles; threads are not visible.

- `tl.program_id(axis)` selects the tile.
- `tl.arange` builds blocks of offsets, so a pointer becomes a *tensor of pointers*.
- `tl.load(ptrs, mask=…, other=…)` and `tl.store` take masks, which handle ragged edges.
- `tl.dot(a, b)` maps a tile product onto the tensor-core (MMA) instructions.
- Reductions (`tl.sum`, `tl.max`, `tl.reduce`) and scans (`tl.associative_scan`) work along a tile axis.

The programmer chooses block sizes, `num_warps` and `num_stages`. The compiler decides thread mapping, coalescing, vectorization, shared-memory staging and swizzling, MMA layouts and software pipelining.

The MAPL 2019 paper presents Triton as a tile-centric language whose matmul and convolution kernels are "on par with hand-tuned vendor libraries (cuBLAS / cuDNN)" **[VERIFIED: Tillet, Kung, Cox, MAPL 2019; abstract via https://www.sigplan.org/OpenTOC/mapl19.html]**.

**Matmul.** Each program owns a `BLOCK_M × BLOCK_N` output tile and loops over K in `BLOCK_K` steps with `acc += tl.dot(a_tile, b_tile)`, accumulating in fp32. The launch order is "grouped" (a `GROUP_SIZE_M` swizzle of program ids) so neighbouring programs reuse the same A and B panels in L2. The epilogue (bias, activation, cast) runs on `acc` before `tl.store`. **[LIKELY: this is the canonical Triton tutorial structure; not re-fetched this session.]**

**Reductions.** A row-wise softmax, LayerNorm or RMSNorm loads a whole row (BLOCK = next power of two of the row length, with masks) and computes max, exp and sum in registers in one pass. Long rows loop in chunks, using two passes or Welford. **[LIKELY: same tutorial set.]**

**Pipeline.** The stages are:
1. The Python AST becomes TTIR (MLIR `tt` dialect).
2. TTIR becomes TritonGPU IR (TTGIR), where every tensor carries a *layout encoding*: blocked, MMA, dot-operand, shared/swizzled, slice.
3. Passes run over TTGIR: layout propagation and conversion, coalescing, pipelining, and on newer GPUs warp specialization.
4. TTGIR lowers to LLVM IR, then to PTX/cubin or AMDGCN.

**Linear layouts (2025, ASPLOS 2026).** All layouts are binary matrices over F₂ that map hardware bits (register, lane, warp) to tensor coordinates, and a layout conversion becomes linear algebra. Integrated into Triton, this fixed many layout bugs and gave up to 14.2× in micro-benchmarks and 1.59× on real benchmarks **[VERIFIED: https://arxiv.org/abs/2505.23819]**. This is the most transferable *compiler-internal* idea for Sky's kernel IR. Apple's `simdgroup_matrix` 8×8 fragments and the M5 16×16 fragments are both bit-permutation layouts. Mojo's `_apple_frag_layout` bit-scatter is an example (Topic 5).

**Lower-level escape hatches (2025–26).**
- *Gluon* exposes layouts, shared memory, warp specialization and target-specific features directly **[VERIFIED: https://triton-lang.org/main/gluon/index.html]**.
- *TLX* is a warp-aware extension from Meta for production kernels **[VERIFIED: https://arxiv.org/abs/2605.10905, abstract]**.

The direction matters: Triton's authors added *lower* levels because the block model alone could not express state-of-the-art Hopper and Blackwell kernels.

**Autotuner.** The signature is `@triton.autotune(configs, key, prune_configs_by, reset_to_zero, restore_value, pre_hook, post_hook, warmup, rep, use_cuda_graph, do_bench, cache_results)` **[VERIFIED: https://triton-lang.org/main/python-api/generated/triton.autotune.html]**.
- `configs` is a list of `triton.Config` objects, each holding meta-parameters, `num_warps` and `num_stages`.
- `key` names arguments "whose change in value will trigger the evaluation of all provided configs". This is the per-shape cache key.
- `prune_configs_by` takes a performance model or early-prune hook.
- `cache_results` persists the winners to disk.

The design is small and correct for a narrow workload: an explicit candidate list, a shape key, a pruner and a disk cache.

### Inductor templates and epilogue fusion

Inductor's matmul and bmm lowerings are *templates*: Triton kernels with hooks where the scheduler can inline consumers into the epilogue. These are pointwise ops on the template's output with compatible indexing, such as bias, activation, scale and cast. Recent versions also inline producers into the *prologue*, for example a cast or dequantization on the inputs. **[LIKELY: Inductor source and design knowledge, not re-fetched.]**

In `max-autotune` mode several template configurations, plus vendor libraries, are benchmarked per shape and the winner is cached.

PyTorch 2.13 (July 2026) added a **CuTeDSL "Native DSL" backend** to Inductor, used for GEMM and RMSNorm alongside Triton **[VERIFIED: https://pytorch.org/blog/pytorch-2-13-release-blog/]**. Even PyTorch's flagship compiler routes its hottest ops to hand-designed template families, not to fully generated code.

### Triton on Apple GPUs

There is **no official Metal backend**. In 2026 the work is out-of-tree in `triton-lang/triton-ext`:
- **PR #127**, "minimal Apple GPU backend", was merged 2026-09-14.
- **PR #126** is still open (opened 2026-08-31). It has three parts:
  - an `AccelerateAppleMatmul` pass that rewrites `tt.dot` to an 8×8 simdgroup MMA encoding;
  - a `StoreShuffleLayout` pass;
  - an `EmitMSL` pass that lowers TTGIR *straight to MSL source*, which the Metal toolchain then compiles to a metallib.
- The PR body says it "Supersedes #48 that implemented this by lowering to AIR via LLVM; that path relies on AIR internals Apple does not publicly document/support, so this PR achieves the same through MSL codegen."

**[VERIFIED: `gh api repos/triton-lang/triton-ext/pulls/126` and `/127`]**

The PR's own benchmarks compare against an earlier commit of the same backend, not against MLX. Its performance is therefore unknown relative to the state of the art **[VERIFIED: PR discussion summary]**.

### Published results (other)

- ThunderKittens uses warp-level 16×16 tiles and "match[es] CuBLAS and FlashAttention-3 on GEMM and attention inference". It is 10–40% faster on attention backward **[VERIFIED: https://arxiv.org/abs/2410.20399]**.
- TileLang "decouples scheduling space (thread binding, layout, tensorize and pipeline) from dataflow" through annotations **[VERIFIED: https://arxiv.org/abs/2504.17577]**.
- Helion is a Python DSL that compiles to autotuned Triton and explores thousands of candidates. Its LLM-guided autotuning cut tuning time from 261 s to 39 s **[VERIFIED: https://pytorch.org/blog/helion/ and https://pytorch.org/blog/from-minutes-to-seconds-llm-guided-autotuning-for-helion-kernels/, via search excerpt]**.

### Borrow

1. **A tile-level kernel IR**: a block program over tiles with masks, explicit `dot` and reductions. It is the right *middle* level between Sky's tensor IR and MSL.
2. **Layouts as first-class, composable objects** in that IR, in the style of linear layouts over F₂. This covers simdgroup fragments, threadgroup swizzles and packed 4-bit weight layouts.
3. **The autotuner contract**: an explicit config list, a shape key, a pruner and a persistent cache. Sky's `sky-tune` already has the same shape (Topic 10).
4. **Template plus epilogue/prologue fusion** as the way to fuse elementwise work into GEMV and GEMM, not free-form fusion.

### Avoid

1. Building or adopting Triton-for-Metal as Sky's backend. It is immature, out-of-tree and emits MSL anyway, and Sky would inherit a Python and MLIR toolchain.
2. Pretending one level is enough. Triton grew Gluon and TLX below it, so Sky's IR should allow hand-written templates to sit next to generated code.

---

## Topic 2 — TVM, Ansor, MetaSchedule, Relax, MLC-LLM

### Mechanics and results

- **TVM / AutoTVM (OSDI 2018).** A Halide-derived tensor-expression language with schedule primitives. AutoTVM adds hand-written *templates* with knobs, a learned cost model (XGBoost, TreeGRU) and simulated annealing. **[LIKELY: well-known; not re-fetched.]**
- **Ansor (OSDI 2020).** It drops templates.
  - It "explores many more optimization combinations by sampling programs from a hierarchical representation of the search space": derivation rules generate *sketches* (multi-level tiling, fusion, caching stages, rfactor), and random annotation fills in tile sizes, unrolling, vectorization and parallelization.
  - It "fine-tunes the sampled programs with evolutionary search and a learned cost model".
  - A *task scheduler* distributes the tuning budget across subgraphs.
  - Results: "up to 3.8×, 2.6×, and 1.7×" over the state of the art on Intel CPU, ARM CPU and NVIDIA GPU.
  - **[VERIFIED: https://arxiv.org/abs/2006.06762]**
- **Tuning budget in practice.** TVM's auto-scheduler tutorials recommend `num_measure_trials ≈ 800 × len(tasks)`, for example about 20,000 trials for ResNet-50's 29 tasks **[VERIFIED: TVM docs tune_network_cuda (v0.15), via search excerpt]**. Each trial means compiling and measuring a kernel, so tuning a network takes hours **[LIKELY]**.
- **MetaSchedule (NeurIPS 2022).** Search spaces are *probabilistic programs*: schedules with sampling instructions. A chosen candidate is a replayable, serializable *trace*. Search is evolutionary with a cost model. **[VERIFIED: https://arxiv.org/abs/2205.13603 (abstract)]**
- **Learned cost models.** TLP learns from the *schedule-primitive sequence* rather than from low-level program features. It speeds up average search time by 9.1× on CPU and 3.0× on GPU. Its multi-task variant transfers across hardware using "only 7% of the target hardware data" **[VERIFIED: https://arxiv.org/abs/2211.03578]**.
- **Relax (ASPLOS 2025).** A cross-level IR covering graph, loop-level tensor programs and library calls, with symbolic shapes. It claims LLM performance "competitive with state-of-the-art systems" and deployment to phones and browsers **[VERIFIED: https://arxiv.org/abs/2311.02103]**.
- **MLC-LLM** is TVM Unity/Relax plus rule-based GPU schedules (DLight) and Metal codegen (MSL text).
  - It is still actively developed (commits on 2026-10-01) **[VERIFIED: `gh api repos/mlc-ai/mlc-llm/commits`]**.
  - Apache TVM is also active **[VERIFIED: `gh api repos/apache/tvm/commits`]**.
  - On Apple silicon, a 2025 comparative study (M2 Ultra, Qwen-2.5) found that "MLX achieves the highest sustained generation throughput", while "MLC-LLM delivers consistently lower TTFT for moderate prompt sizes" **[VERIFIED: https://arxiv.org/abs/2511.05502]**.
  - Secondary summaries put MLC about 17% below MLX in steady-state decode **[LIKELY: secondary source; treat as indicative]**.

### Negative results and complexity traps

1. **A general compiler for LLMs on Apple GPUs already exists, and it sits behind MLX on decode throughput.** MLC-LLM has had years of work from strong compiler researchers. "Compile LLMs to Metal with an ML compiler" is not novel and does not by itself win **[VERIFIED for the ranking via 2511.05502; LIKELY for the size of the gap]**.
2. **Tuning cost scales with the generality of the search space.** Ansor-style search needs thousands of trials per network. Learned cost models (TLP, Tenset) exist to cut that cost, and they are only needed when the workload is open-ended **[LIKELY]**.
3. **Hand templates came back.**
   - AutoTVM's templates were a maintenance burden.
   - Ansor removed them.
   - TVM's LLM path then reintroduced *rule-based*, hand-written schedules (DLight), because per-model, per-device autotuning was too slow for users.
   - **[LIKELY: DLight's existence and role in MLC-LLM are well-known; not re-fetched.]**
4. **Measurement infrastructure is a product in itself**: RPC runners, noise control, caching, version invalidation **[LIKELY]**.

### Borrow

- **Traces as the unit of tuning** (MetaSchedule). Store the *sequence of schedule decisions* in the database, not just knob values, so a winner can be replayed and audited.
- **A task scheduler** (Ansor) that spends the tuning budget where it most improves end-to-end tokens per second. For Qwen3-32B that means gate/up and down GEMVs and the lm_head (Topic 10).
- **Symbolic shapes only for the one truly dynamic dimension** (context length), with buckets. Relax's machinery is overkill when shapes come from types.

### Avoid

- Learned cost models, Ansor-scale generic sketch search, and an RPC tuning farm. The workload has about 60 distinct kernels (Topic 10).
- A general loop-level IR as the *only* path to performance. Keep hand-designed templates as first-class citizens.

---

## Topic 3 — Halide and scheduling languages

### Mechanics

**Algorithm and schedule are separate.**
- The *algorithm* is a set of pure functions over integer domains.
- The *schedule* decides evaluation order and storage. Its primitives include:
  - `split`, `reorder`, `tile`, `vectorize`, `parallel`, `unroll`;
  - `compute_at`/`store_at`/`compute_root` (inline by default);
  - `rfactor` (turning a reduction into a parallel two-stage reduction);
  - `gpu_blocks`/`gpu_threads`.
- *Bounds inference* derives the region each producer must compute from its consumers.

A schedule cannot change the result, apart from rfactor's reassociation. Correctness follows by construction. **[LIKELY: Halide's documented design (PLDI 2013 and tutorials); not re-fetched.]**

**GPU and Metal.** Halide has a Metal backend (`src/CodeGen_Metal_Dev.cpp`, 948 lines) that emits MSL text and can compile it offline to a metallib via `xcrun`. It does **not** use `simdgroup` or `simdgroup_matrix` (grep: no matches) **[VERIFIED: `gh api repos/halide/Halide/contents/src/CodeGen_Metal_Dev.cpp`]**. The general schedule language did not reach Apple's matrix hardware.

### Autoschedulers

All four are in the tree: `src/autoschedulers/{mullapudi2016, li2018, adams2019, anderson2021}` **[VERIFIED: `gh api repos/halide/Halide/contents/src/autoschedulers`]**.

| Autoscheduler | Technique | Result |
|---|---|---|
| Mullapudi 2016 | Greedy grouping and tiling on top of bounds analysis, with no autotuning | "in seconds, generates schedules … performance-competitive with, and often better than, schedules manually authored by expert Halide developers" **[VERIFIED: https://research.google/pubs/automatically-scheduling-halide-image-processing-pipelines/]** |
| Li 2018 | Simple scheduler for gradient pipelines (Halide AD) | **[LIKELY]** |
| Adams 2019 | Beam search over a much larger space, with a cost model trained on random programs | "on average almost twice as fast as the existing Halide autoscheduler without autotuning, or more than twice as fast with … the first automatic scheduling algorithm to significantly outperform human experts on average" **[VERIFIED: https://halide-lang.org/papers/halide_autoscheduler_2019.pdf, abstract]** |
| Anderson 2021 (GPU) | Two-phase search, hierarchical sampling, memoization | 1.7× faster than prior automatic solutions on average (up to 5×), "competitive with what the best human experts were able to achieve"; compile time 49× faster on average **[VERIFIED: https://arxiv.org/abs/2012.07145]** |

**Successors.**
- TVM's tensor expressions descend from Halide.
- Exo 2 (ASPLOS 2025) lets users *grow* the scheduling language with libraries of new scheduling operations ("Cursors"). It amortizes scheduling over 80+ kernels with an order of magnitude less scheduling code **[VERIFIED: https://arxiv.org/abs/2411.07211]**.

**The 2024–26 trend** runs toward *tile languages*: Triton, ThunderKittens, TileLang, CuTe DSL, Gluon. The block-level algorithm is written by hand. Only a handful of knobs remain as "schedule" (tile sizes, stages, warps), and autotuners set them **[LIKELY: synthesis of Topic 1 sources]**.

### Negative results

1. **Expert-quality autoscheduling took about five years and three generations of autoschedulers, for one domain (image pipelines).** Even then, high-value production pipelines were commonly hand-scheduled **[LIKELY: the record above plus practitioner reports; production adoption not verified this session]**.
2. **General scheduling languages lag new hardware features.** Halide's Metal codegen never adopted `simdgroup_matrix` **[VERIFIED above]**.

### Borrow

- Keep the *what/how split*, but make "how" a **typed, checked, optional annotation on a small set of kernel templates**, with a default chosen analytically (Mullapudi-style) and refined by the tuner. See the design sketch after Topic 4.
- Correct-by-construction schedules. A schedule may only choose among semantics-preserving transformations. A reduction-reassociating choice (split-K, tree reductions) is legal only under a numerics contract that allows it, and is recorded so that results stay reproducible for a given (schedule, device). This fits Sky's existing `Reproducible`/`Relaxed`/`Fast` contracts **[VERIFIED for the contracts: ~/sky/crates/sky-egraph/src/lib.rs]**.

### Avoid

- A general-purpose scheduling language as a product surface.
- A learned-cost-model autoscheduler.

---

## Topic 4 — XLA/JAX and PyTorch 2

### 4a. XLA/JAX

**Pipeline.** The XLA:GPU pipeline runs HLO passes, including SPMD partitioning and **layout assignment** ("for each operation present in the HLO, the Layout Assignment pass chooses an optimal layout"). Then comes **fusion** ("XLA's single most important optimization"). Then **buffer assignment**, which "aims to produce an optimal buffer allocation for the program, minimizing the amount of intermediate memory consumed".

Codegen has three paths:
- vendor libraries;
- LLVM IR emitters for reductions, transposes and similar ops;
- Triton for GEMMs and advanced fusions, where XLA "selects tiling parameters and invokes Triton".

The runtime extracts CUDA graphs.

**[VERIFIED: https://openxla.org/xla/gpu_architecture]**

**Priority fusion.** A single *priority-based* fusion pass replaced the greedy bottom-up passes. Its analytical cost model estimates kernel time as max(compute time, memory time) plus launch overhead. Fusions are applied in order of modeled runtime reduction. **[VERIFIED: https://github.com/openxla/xla/discussions/6407 and /10065, via search excerpt]**

**JAX** traces Python into jaxprs over abstract (shape, dtype) values. `jit` compiles per signature and recompiles on new shapes. AOT goes through `jax.export` (serialized StableHLO) and PJRT executables. **[LIKELY: documented behaviour; not re-fetched.]**

**jax-metal is a negative result.**
- It was Apple's PJRT plugin, lowering StableHLO to **MPSGraph**.
- It is closed-source. The JAX team "has no access to the codebase".
- Its last release was 2024-10-08, and users ask "Is jax-metal dead or alive?"

**[VERIFIED: https://github.com/jax-ml/jax/discussions/32135 and https://developer.apple.com/metal/jax/, via search excerpt]**

Together with PyTorch's 2026 move *off* MPSGraph ("MPSGraph … adds compilation and scheduling overhead per dispatch"; "The native Metal path eliminates MPSGraph's per-op compilation cost") **[VERIFIED: https://pytorch.org/blog/pytorch-2-13-release-blog/]**, this says: **do not use MPSGraph as Sky's compilation target.**

### 4b. PyTorch 2 (ASPLOS 2024)

**Dynamo.**
- Mechanism: a CPython frame-evaluation hook symbolically executes Python bytecode, extracts FX graphs, installs *guards*, and on unsupported constructs inserts a **graph break** (fall back to Python, continue in a new graph).
- Graph breaks are common:

  | Suite | Models | Works with Dynamo | Models with 0 graph breaks | Mean graphs per model | Operators captured |
  |---|---|---|---|---|---|
  | TorchBench | 80 | 74 (93%) | 52 (70%) | 21.1 | 91.8% |
  | HuggingFace | — | — | — | 7.7 | — |
  | TIMM | — | — | — | 1 | — |

  **[VERIFIED: PyTorch 2 paper, Table, https://docs.pytorch.org/assets/pytorch2-2.pdf]**

**AOTAutograd.** It traces forward and backward jointly. It "uses a min-cut algorithm to split this joint graph into separate forward and backward graphs in a way that optimizes for memory usage", with rematerialization **[VERIFIED: same paper]**.

**Dynamic shapes.** Symbolic sizes are used, with proactive "0/1 specialization": "if an input size is 0 or 1 … we treat it as a constant and add an appropriate guard" **[VERIFIED: same paper]**.

**Inductor.**
- Design: a define-by-run loop-level IR, a scheduler that fuses by dependency and score, buffer reuse, and codegen to Triton (GPU) and C++/OpenMP (CPU).
- Results: **2.27× inference and 1.41× training** geomean speedups on A100 over 180+ models **[VERIFIED: same paper]**.

**On Apple (MPS).**
- The `torch.compile` on MPS tracker says: "torch.compile support for MPS device is an early prototype and attempt to use it to accelerate end-to-end network is likely to fail".
  - Reductions are slower than eager.
  - There is no dynamic-shape support.

  **[VERIFIED: https://github.com/pytorch/pytorch/issues/150121]**
- 2.13 added FlexAttention on MPS as *hand-written Metal kernels*, for sparse prefill and for decode including GQA **[VERIFIED: 2.13 release blog]**.

### What Sky gains by owning the language

1. **No graph breaks, by construction.** The whole decode step, sampling and speculative accept/reject included, is one Sky program. PyTorch needed a bytecode-level JIT because it does not own Python, and still sees about 30% of TorchBench models break **[VERIFIED for the statistic]**.
2. **Static shapes, dtypes and quant formats are types, so there are no guards and no recompiles.** Context length is the one dynamic dimension. Sky can specialize per context-length bucket *ahead of time* (FlashInfer-style plan, Topic 9).
3. **Whole-program static planning that Python frameworks cannot do.**
   - A static memory plan (XLA buffer assignment), which matters for the RAM metric.
   - A static dispatch schedule, encoded once and replayed (Topic 7).
   - Weight-layout specialization. Weights are compile-time constants, so their packing can be co-designed with kernels; see the design sketch below.
4. **Explainable decisions.** Sky already reports rewrites and costs through `--explain` **[VERIFIED: ~/sky/crates/sky-egraph/src/lib.rs, `EG-COST-001`]**.

**Losses:**
- The PyTorch and Hugging Face ecosystem. Model import becomes a converter: safetensors/GGUF readers, a tokenizer, and a per-architecture model definition in Sky.
- Users who live in Python.

**[LIKELY]**

### Borrow / avoid (Topic 4)

- **Borrow:** priority fusion with an analytical cost model; layout assignment as a pass; buffer assignment with liveness (a static memory plan); AOT executables; PyTorch's guard idea, reduced to "context bucket".
- **Avoid:**
  - tracing-based capture and dynamic-shape machinery;
  - MPSGraph as a target (jax-metal abandoned; PyTorch migrating off);
  - depending on closed vendor plugins.

---

## Design sketch — a Sky "what vs how" split for tensors

**[EXPERIMENTAL: a proposal consistent with Sky's IR design (`docs/design/ir.md`: fusion, specialization, algorithm choice, contracts). Syntax is illustrative.]**

### Principles

1. **Types carry facts the compiler must not guess.** These are:
   - shape, with symbolic sizes only for context length;
   - element type;
   - **quant format**: scheme, bits, group size, scale and bias types, packing;
   - **layout**: logical-to-physical map, including fragment-order packing for weights;
   - units where meaningful;
   - **numerics contract**.

   A Q4 tensor cannot be added to an f16 tensor without an explicit `dequant`, the way `1 m + 1 s` is a type error today.
2. **Schedules are optional, checked annotations on kernel templates.** A schedule chooses among semantics-preserving transformations:
   - tiling;
   - fusion grouping, i.e. which ops go in a template's prologue or epilogue;
   - mapping to threadgroups, simdgroups and lanes;
   - staging through threadgroup memory;
   - vector width;
   - split-K.

   A choice that reassociates a reduction is legal only under a `relaxed` or `tolerance` contract, and the chosen order is recorded in the plan. Results are then bit-reproducible for a given (plan, device), which Sky's philosophy needs.
3. **The default is automatic.** An analytic default (a roofline plus occupancy rules) picks a schedule, and `sky tune` refines its knobs per device. A hand schedule overrides both, and the compiler explains each choice.
4. **The reference interpreter is the oracle.** Every template and schedule is differentially tested against the unfused reference semantics, at the declared tolerance (Topic 12).

### (a) Q4 dequant-GEMV

```sky
// WHAT
type Q4g64 = Quant[affine, bits = 4, group = 64, scale = f16, bias = f16]

fn qgemv[N, K](w: Matrix[N, K, Q4g64], x: Vector[K, f16]) -> Vector[N, f32]
    numerics relaxed(rel = 1e-3)            // tolerance contract vs exact dequant-then-dot
{
    dequant(w) @ x                          // the math; no loops, no tiles
}

// HOW (optional; the default comes from the analytic model + sky-tune)
schedule qgemv on apple_gpu {
    rows_per_simdgroup  = tunable {4, 8, 16}        // output rows a simdgroup owns
    simdgroups          = tunable {2, 4, 8}         // per threadgroup
    k_per_lane          = tunable {8, 16, 32}       // 4-bit values per lane per step (vector loads)
    reduce k with simd_sum, order = fixed           // deterministic tree; recorded in the plan
    weight_layout       = interleave_groups(64)     // offline repack: scale/bias adjacent to their nibbles
}
```

### (b) The Qwen3 decode block, with fusion as schedule

```sky
@numerics(relaxed, kl(logits) <= 1e-3)      // a *model-level* quality contract
fn qwen3_block(h: Vector[5120, f16], c: KV[Layer], pos: Index, w: BlockWeights) -> Vector[5120, f16] {
    let x        = rmsnorm(h, w.attn_norm)
    let (q, k, v)= split(qgemv(w.qkv, x), [8192, 1024, 1024])   // QKV concatenated (rewrite R1)
    let q        = rope(rmsnorm_heads(q, w.q_norm, heads = 64), pos)
    let k        = rope(rmsnorm_heads(k, w.k_norm, heads = 8), pos)
    c.append(k, v)
    let a        = attention(q, c, causal, gqa = 8)
    let h        = h + qgemv(w.o, a)
    let x        = rmsnorm(h, w.mlp_norm)
    let (g, u)   = split(qgemv(w.gate_up, x), [25600, 25600])    // gate/up concatenated (R1)
    h + qgemv(w.down, silu(g) * u)
}

schedule qwen3_block on m5_max {
    fuse rmsnorm(attn_norm)     into qgemv(qkv).prologue      // each TG recomputes Σx² over 5120: cheap at batch 1
    fuse {q_norm, k_norm, rope, c.append} into qgemv(qkv).epilogue
    attention  = split_k_decode(gqa_pack = 8, chunks = tunable, order = fixed)
    fuse residual into qgemv(o).epilogue
    fuse rmsnorm(mlp_norm)      into qgemv(gate_up).prologue
    weight_layout(gate_up) = interleave_rows(g, u)            // so one TG holds g_i and u_i …
    fuse silu_mul               into qgemv(gate_up).epilogue  // … and writes m_i directly (halves intermediate traffic)
    fuse residual into qgemv(down).epilogue
    dispatch = static_plan                                    // whole step pre-encoded, replayed per token (Topic 7)
}
```

### Facts as types vs schedule

| Belongs in **types** | Belongs in **schedule** |
|---|---|
| Shapes; head counts; GQA ratio | Tile sizes and rows per simdgroup |
| Quant format: bits, group, scale and bias dtypes | Threadgroup and simdgroup mapping; vector width |
| *Logical* layout and KV-cache layout | Fusion grouping (prologue/epilogue) |
| Precision and tolerance contract | Split-K, chunk count, reduction tree shape (recorded) |
| Units; positivity and other facts (existing) | Staging via threadgroup memory; async copy |
| Weight *packing* (physical layout) when it is an ABI of the weights file | Weight packing when Sky repacks at load time (a compiler choice) |

**Why the weight layout is the interesting part [LIKELY].** At batch 1, weights are constants loaded once. Sky can therefore choose their physical layout freely, within the declared format: interleaved gate/up rows, QKV concatenation, fragment-order nibble packing. llama.cpp and MLX fix the format and layout per quant type. Mojo's M5 FP4 design note shows that matching the weight packing to the MMA fragment order is what makes a fused dequant-matmul bit-identical to the dense path and fast **[VERIFIED: modular FP4_DESIGN.md, "Correctness anchor"]**.


---

## Topic 5 — MLIR, IREE and paths to Apple GPUs

*This is a decision item. The recommendation is at the end of this topic.*

### 5.1 The MLIR dialects that matter, and what is missing

The upstream MLIR dialect directory (`mlir/include/mlir/Dialect`, `llvm/llvm-project` main, 2026-10) contains, among others:

`AMDGPU, ArmNeon, ArmSME, ArmSVE, Bufferization, EmitC, GPU, Linalg, MemRef, NVGPU, NVVM, SCF, SPIRV, SparseTensor, Tensor, Tosa, Transform, Vector, X86, XeGPU`

**There is no Apple, Metal or AIR dialect** **[VERIFIED: `gh api repos/llvm/llvm-project/contents/mlir/include/mlir/Dialect`]**.

The relevant dialects:

| Dialect | Role |
|---|---|
| `linalg` | Structured ops with tiling and fusion interfaces |
| `tensor` + `bufferization` | Value semantics, then one-shot bufferize |
| `vector` | `vector.contract` and transfer ops |
| `gpu` | Kernels, launches and serialization targets (NVVM, ROCDL, SPIR-V) |
| `transform` | Schedules expressed as IR |
| `nvgpu`, `amdgpu`, `xegpu` | Per-vendor matrix-unit and async-copy ops |

**[LIKELY: upstream documentation; dialect roles not re-fetched.]**

The vendor dialects show what *any* serious GPU target needs: a hand-built dialect for its matrix units and async copies. For Apple, nobody has upstreamed one.

### 5.2 IREE

IREE's Metal support is **alive**. The HAL driver had commits in June and August 2026 **[VERIFIED: `gh api repos/iree-org/iree/commits?path=runtime/src/iree/hal/drivers/metal`]**.

Codegen reuses the SPIR-V pipeline:

> "Metal does not provide an open intermediate language; we reuse the SPIR-V code generation pipeline and then cross compile the generated SPIR-V into MSL source with SPIRV-Cross."

The design document also notes:
- "The Metal HAL driver does not perform any caching on GPU programs";
- queue-ordered allocation "just translates to blocking wait and allocation".

**[VERIFIED: https://iree.dev/developers/design-docs/metal-hal-driver/]**

The compiler target sets `msl_version = 3.0` and can embed a compiled metallib **[VERIFIED: `compiler/plugins/target/MetalSPIRV/{SPIRVToMSL.cpp,MetalSPIRVTarget.cpp}`]**.

**The decisive negative.** IREE's Apple target description declares *no matrix-multiply ops*:

```
getAppleTargetDetails(): ... /*mmaCount=*/0, /*mmaOps=*/nullptr, ...
// Note: These values have not been checked and may be higher
getMetalTargetDetails(): arch "apple", features "spirv:v1.3,cap:Shader"
```

**[VERIFIED: `KnownTargets.cpp` @ `9acea6a` (2026-09-11), lines ~898–915 and ~1430–1433, iree-org/iree]**

IREE's Metal path therefore cannot use `simdgroup_matrix`, let alone the M5 Neural Accelerators. It is a portability path, not a performance path. I found **no published IREE LLM results on Apple GPUs** **[LIKELY: absence of evidence from searches in this session]**.

### 5.3 SPIR-V → MSL (SPIRV-Cross)

SPIRV-Cross maps SPIR-V **cooperative matrix** to MSL `simdgroup_matrix`, with hard limits:
- 8×8 only;
- floating-point only (fp16, fp32, bf16; integer types are rejected);
- subgroup scope only;
- compile-time sizes.

Element-wise operations on cooperative matrices are *emulated*. That support was merged 2026-09-16 (PR #2645) **[VERIFIED: https://github.com/KhronosGroup/SPIRV-Cross/pull/2645]**.

There is no mapping to Metal 4 `cooperative_tensor` or `tensor_ops` (the M5 Neural Accelerator path) **[LIKELY: none mentioned in the PR or the search results]**.

### 5.4 Mojo / MAX: the one compiler that reaches the M5 matrix units without MSL

**Lowering.** Mojo GPU functions lower to LLVM IR and then to "Metal-compatible AIR". At runtime the AIR is compiled to a metallib through metal-cpp. Initial support targeted MSL 3.2 / AIR 2.7.0 on macOS 15 **[VERIFIED: https://forum.modular.com/t/apple-silicon-gpu-support-in-mojo/2295]**.

**Initial gaps (Sep–Dec 2025).** The "Running AI models" and "Serving AI models" items were open, and matmul and tensor-op examples were missing **[VERIFIED: same thread]**.

**2026 state.** The open-sourced tree has Apple matmul kernels **[VERIFIED: `modular/modular` `max/mojo/max/gpu/compute/arch/mma_apple.mojo`]**:
- `_mma_apple_8x8` calls `llvm.air.simdgroup_matrix_8x8_multiply_accumulate` and works on all generations M1–M5.
- `_mma_apple` calls `llvm.air.simdgroup_matrix_16x16x16_multiply_accumulate`. This is "Apple M5 only (Metal 4.0 / AIR 2.8.0)".
- The M5 MMA "accepts fp16/bf16/fp32/fp8/int8, but **not FP4**" **[VERIFIED: `max/kernels/src/linalg/matmul/gpu/apple/FP4_DESIGN.md`]**.

**Performance on M5 is still slightly below MLX.**

> "the **shipped** dispatch is **~0.85–0.90× MLX-4bit** across the FLUX mix"

Routing one slice to `matmul2d` raised the mix to 0.964× MLX. The residual is "a **measured** ~10 TF/s co-issue gap (loads and the MMA contend for the per-core issue path)".

**[VERIFIED: FP4_DESIGN.md, "Performance"]**

**Licence and ownership.** Modular became "A Qualcomm Company" (the acquisition completed 2026-07-29). The Mojo compiler was open-sourced on 2026-08-18 under Apache-2.0 with LLVM exceptions **[VERIFIED: https://www.modular.com/blog/mojo-open-source]**.

This makes Mojo's AIR emitter a *licence-compatible reference* for Sky (port or learn, per ROADMAP rule 2) **[LIKELY: licence compatible with Sky's deny.toml list; verify the file headers before copying]**.

### 5.5 AIR directly, vs MSL text

**AIR is undocumented.** Apple's Metal IR (AIR) is LLVM-bitcode-based and not publicly documented.

**Two projects learned this in practice:**
- Triton's community backend abandoned its AIR route (PR #48) for MSL emission, because AIR "relies on AIR internals Apple does not publicly document/support" **[VERIFIED: triton-ext PR #126 body]**.
- Metal FlashAttention reaches undocumented hardware features **from inside MSL** with `__asm("air.simdgroup_async_copy_2d.p3i8.p1i8")` and similar declarations **[VERIFIED: `$S/src/metal-flash-attention/Sources/FlashAttention/GEMM/GEMMHeaders.swift:45–101`]**.

  This *MSL + `__asm("air.*")` escape hatch* gives most of the benefit of emitting AIR without owning an AIR emitter.

**The documented route to the M5 Neural Accelerators is MSL.**
- It goes through Metal 4 TensorOps and Metal Performance Primitives (`matmul2d`, cooperative tensors).
- MLX uses exactly this route. Its `steel/gemm/nax.h` includes `<MetalPerformancePrimitives/MetalPerformancePrimitives.h>` with 16×16 fragments, gated by `is_nax_available()`.
  - **[VERIFIED: `$S/src/mlx/mlx/backend/metal/kernels/steel/gemm/nax.h`, `device.cpp:952`]**
  - **[VERIFIED: https://machinelearning.apple.com/research/exploring-llms-mlx-m5: "MLX leverages the Tensor Operations (TensorOps) and Metal Performance Primitives framework introduced with Metal 4"]**

**The tensor path is still opaque.** A 2026 reverse-engineering study of the Metal 4.1 tensor path (on an M4 Max, which predates the Neural Accelerators) found:
- `matmul2d` "executes entirely on the GPU shader cores";
- fp8 is emulated;
- "eleven previously undocumented behaviors", including the opaque "8x8 cooperative_tensor fragment layout".

**[VERIFIED: https://arxiv.org/abs/2606.12765]**

**Every competitive Apple-GPU engine generates MSL text.**
- MLX (hand-written MSL plus JIT'd MSL strings).
- llama.cpp (Metal shaders).
- tinygrad. It renders MSL with `simdgroup_multiply_accumulate` and compiles through the private `MTLCodeGenService` API for speed **[VERIFIED: `$S/src/tinygrad/tinygrad/renderer/cstyle.py:401`, `runtime/ops_metal.py:29–68`]**.
- PyTorch MPS (hand-written Metal kernels).
- BaseRT (hand-written Metal 4 kernels).
- Halide and TVM/MLC (MSL emitters).
- IREE (MSL via SPIRV-Cross).
- triton-ext (MSL emitter).

**[VERIFIED for each, see the respective sections]**

**Toolchain note.** Since Xcode 26 the *offline* Metal toolchain is a separate download of about 700 MB (`xcodebuild -downloadComponent MetalToolchain`) **[VERIFIED: https://developer.apple.com/forums/thread/803174 and https://github.com/actions/runner-images/issues/13080, via search excerpt]**.

Runtime compilation from MSL source (`MTLDevice newLibraryWithSource`) uses the OS's compiler and needs no Xcode. Sky can JIT MSL on users' machines and AOT-compile metallibs only in its build farm **[LIKELY: this is how MLX's JIT path and tinygrad work; confirm on a clean macOS install]**.

### 5.6 Rust ↔ MLIR (melior) and dependency weight

**melior** wraps the **MLIR C API**. The latest release is `0.28.2` (2026-09-28), with frequent breaking minor releases (0.27.x → 0.28.x in September 2026). It requires a system **LLVM/MLIR 23** (for example `brew install llvm@23`). Its README warns that parts of the API are not type-safe; for instance, accessing unloaded dialects "can lead to runtime errors or segmentation faults" **[VERIFIED: https://github.com/mlir-rs/melior README; crates.io API]**.

The MLIR C API cannot define new dialects, passes or interfaces. A Metal dialect, or custom tiling and fusion interfaces, would have to be written in **C++** **[LIKELY: the C API exposes IR construction and pass running, not dialect definition (TableGen and C++)]**.

LLVM and MLIR are a multi-gigabyte, hours-long C++ build with roughly six-monthly major versions to track **[LIKELY: well-known; not measured here]**.

### 5.7 Options A–D

| Criterion | **A. Own stack** (Sky IR → tile IR → MSL text, optional `__asm("air.*")`) | **B. Lower all of Sky to MLIR** | **C. Selected MLIR dialects as tools** (e.g. linalg/vector/transform → SPIR-V → SPIRV-Cross → MSL) | **D. MLIR only for some backends** (later NVIDIA/AMD) |
|---|---|---|---|---|
| **Control over Apple features** (simdgroup MMA, NAX TensorOps, async copy, threadgroup layout) | **Full**: MSL exposes TensorOps/MPP; `__asm("air.*")` covers the rest | Low: no Apple dialect; a custom one in C++ is still needed | **Poor**: SPIRV-Cross caps at 8×8 float simdgroup matrices; no NAX path; IREE declares `mmaCount=0` | n/a for Apple; good on NVIDIA/AMD via nvgpu/amdgpu |
| **Complexity** | Moderate: tile IR, about 10 kernel-template families, MSL printer, runtime | Very high: C++ dialects, passes, bufferization, melior C-API gaps | High, for little gain | Moderate, deferred |
| **Compile time** | Fast: Rust passes plus Apple's compiler (per-kernel MSL compile; cache) | Slower: MLIR pass pipelines plus LLVM | Slower | Per-backend |
| **Optimization power on Apple** | As good as the templates plus tuner; the Apple frontier *is* hand-written MSL (MLX, BaseRT, MFA) | Not better: still bottoms out in MSL or AIR | Worse (no MMA/NAX) | n/a |
| **Maintainability** (Rust project) | Pure Rust plus the MSL dialect of C++ in templates | C++ and LLVM-version treadmill; melior pre-1.0 | Same as B for the parts used | Isolated behind a backend interface |
| **IP differentiation** | **High**: the semantic layer, static planning, formats and contracts are Sky's own | Low: MLIR stacks are commoditized (IREE; Mojo now open-source) | Low | Neutral |
| **Evidence it works on Apple** | MLX, BaseRT, tinygrad, MFA, llama.cpp | None: Mojo reaches 0.85–0.96× MLX with a large team and *undocumented* AIR | IREE: no MMA | — |

**Comparison with tinygrad's minimal stack [VERIFIED for the mechanics above; LIKELY for the conclusion].** tinygrad shows that a small IR plus a C-style MSL renderer plus BEAM search reaches Apple's matrix units (`simdgroup_multiply_accumulate`) with almost no infrastructure. Its LLM decode on Apple is reported by other agents to trail MLX. A minimal stack is *necessary* (fast iteration, no C++), but generic codegen is *not sufficient*. The hot ops need hand-designed templates: quantized GEMV, NAX GEMM, decode and prefill attention.

### 5.8 Recommendation

**Adopt option A. Defer D. Use C only as ideas, not as code. Do not do B.**

1. **The core stays Rust: own IR, own tile-level kernel IR, MSL text as the GPU target.**
   - Reach the Neural Accelerators through documented Metal 4 TensorOps/MPP in MSL.
   - Allow a narrow, test-gated `__asm("air.*")` escape hatch for features MSL does not expose, such as async copy (MFA precedent).
   - **[LIKELY: the strongest evidence is that every competitive Apple engine does this, and that the only MLIR route to NAX (Mojo) depends on undocumented AIR and still trails MLX.]**
2. **Borrow MLIR's *ideas*, reimplemented small in Rust:**
   - linalg's structured-op and tiling interfaces;
   - the transform dialect's "schedule as data" (Sky's schedules become serializable traces, as in MetaSchedule);
   - vector-dialect contraction and transfer abstractions in the tile IR.
3. **Change the ROADMAP.**
   - Replace "GPUs through MLIR [19, 21, 55]" and the `MLIR … melior … depend` row.
   - New wording: "Apple GPUs through Sky's tile IR → MSL (Metal 4 TensorOps); other GPUs later via a backend interface that may print Triton/Gluon, Mojo, or MLIR."
   - Keep the reuse policy: Mojo's Apple AIR code is Apache-2.0 + LLVM exception, so it can be *read and ported* if AIR emission ever becomes necessary **[EXPERIMENTAL]**.
4. **Revisit triggers.** Each would reopen B or C:
   - Apple documents AIR or ships an MLIR/LLVM Metal target;
   - Sky commits to NVIDIA/AMD backends (then D, possibly through Triton, CuTe DSL or Mojo rather than raw MLIR);
   - a Metal dialect appears upstream with an active maintainer.

---

## Topic 6 — Tensor-graph superoptimization and e-graphs

### 6.1 Systems

**TASO (SOSP 2019).**
- It *generates* graph substitutions by enumerating all graphs of up to four operators: 743 candidates in about 5 minutes.
- It *verifies* them with an automated theorem prover against operator specifications.
- It applies them with a cost-based backtracking search.
- It is up to 2.8× faster than existing frameworks.

**[VERIFIED: https://theory.stanford.edu/~aiken/publications/papers/sosp19.pdf, via search excerpt]**

**PET (OSDI 2021).** It adds *partially* equivalent transformations plus automatically generated correction kernels **[LIKELY: paper known; numbers not re-checked]**.

**Tensat (MLSys 2021).**
- It replaces TASO's backtracking with **equality saturation in `egg`**: all rewrites are applied to an e-graph, then the best graph is extracted.
- It needs multi-pattern rules (for example, merging two matmuls that share an input) and **ILP extraction with cycle filtering**.
- It finds graphs "with up to 16% speedup over state-of-the-art, while spending on average 48x less time optimizing".

**[VERIFIED: https://arxiv.org/abs/2101.01332]**

**Follow-up (2024).** Equality saturation does not escape memory limits:

> "memory constraints prevent the IR from capturing all optimized versions and thus reintroduce the phase-ordering problem in the construction phase"

Choosing which rules to apply with MCTS gave up to an 11% inference speedup **[VERIFIED: https://arxiv.org/abs/2410.05534]**.

**EinNet (OSDI 2023)** derives tensor-algebra expressions. **Korch (ASPLOS 2024)** splits operators into primitives and then orchestrates kernels with an ILP **[LIKELY: known papers, not re-checked this session]**.

**Mirage (OSDI 2025).**
- *μGraphs* represent a program at the kernel, thread-block and thread levels at once. This lets Mirage find custom fused kernels, for example re-deriving FlashAttention and FlashDecoding.
- Search is restricted to the **LAX fragment**: multilinear operators, division, and at most one exponentiation on every input→output path.
- **Probabilistic equivalence verification** generalizes polynomial identity testing, so random tests over finite fields give strong guarantees for LAX programs.
- **Abstract-expression pruning** is essential. The RMSNorm search at 7 or more operators takes 22–26 s with it and more than 10 h without.
- Results:
  - up to **3.3×** over the best baselines (GQA), and up to 2.2× over FlashAttention/FlashDecoding in some cases;
  - end-to-end **1.0–1.9×** over PyTorch (LLaMA-3-8B: 1.4× at batch 1).
- Cost: "Mirage takes up to 4 hours to optimize a LAX program".
- Limit: the verifier "does not support certain DNN operators such as ReLU".

**[VERIFIED: https://arxiv.org/abs/2405.05751, full text §§1–8, Table 5]**

**MPK, compiler side (arXiv 2512.22219).**
- Operators are decomposed into *SM-level tasks*, each computing "a disjoint subset of the output", which forms a *tensor task graph*.
- An event is inserted for every producer/consumer task pair whose regions overlap.
- *Successor-set and predecessor-set event fusion* reduce the number of events by **37–118×**.
- The graph is normalized so that each task has at most one dependent and one triggering event.
- Dynamic batch sizes are handled by specializing graphs to powers of two.

**[VERIFIED: https://arxiv.org/html/2512.22219v2]** The runtime side is in Topic 7.

**2026 newcomers.**
- **Axon** is a "synthesizing superoptimizer" that generates target instructions from semantic specifications and verifies with SMT over unbounded tensors **[VERIFIED: https://arxiv.org/abs/2606.26344, abstract]**.
- **AutoMegaKernel** pairs an LLM agent with a *static* validator for deadlock and race freedom **[VERIFIED: https://arxiv.org/abs/2606.09682]**.

### 6.2 How practical are e-graphs

**Extraction is NP-hard in general.** Once shared sub-terms are priced once (DAG cost), extraction is NP-hard. Tree-cost greedy extraction is exact only for tree costs. The practical answers are:
- ILP, as in Tensat;
- **SmoothE**, a differentiable, GPU-accelerated extraction that supports non-linear and learned costs (ASPLOS 2025 best paper);
- **e-boost** (ICCAD 2025), which combines heuristics with exact solving.

**[VERIFIED: https://github.com/cornell-zhang/SmoothE; https://arxiv.org/abs/2508.13020, via search excerpt]**

**Blow-up.** Associativity and commutativity rules and multi-pattern rules explode e-graph size. Tensat limited multi-pattern applications, and the 2024 MCTS paper hit memory limits. **[VERIFIED above]**

**Production evidence: Cranelift's acyclic e-graph ("aegraph").**
- Prototyping with `egg` cost "over 2x" compile time at first, and still "23% overhead — with no rewrites applied" after tuning.
- The production design gives up full saturation and cycles in favour of eager single-pass rewriting with scoped elaboration.
- Measured outcome: about **2%** faster execution than the classical pipeline, at about **7–8%** more compile time.
- The multi-value representation itself is worth only about **0.1%**.
- "the average eclass size after rewriting is 1.13 enodes".
- The author's verdict: "the aegraph itself — the ability to represent multiple options for one value — may not (yet?) be pulling its weight".

**[VERIFIED: https://cfallin.org/blog/2026/04/09/aegraph/]**

### 6.3 Negative results

1. **Superoptimizers win by finding kernel-level fusions, and they are expensive.** Graph-level rewrites alone (TASO, Tensat) bring single-digit to teen-percent gains on already-optimized models. Mirage's large wins come from kernel-level μGraph search: up to 4 h per pattern, NVIDIA-specific, with a restricted verifier. **[VERIFIED for each number above; LIKELY for the generalization]**
2. **For conventional code, saturation rarely pays.** In production, the multi-version part of e-graphs bought about 0.1%. **[VERIFIED: Cranelift]**
3. **The LLM decode graph is small and repetitive.** About 64 identical blocks of roughly 25–40 ops. The interesting rewrites are a handful of *known* identities: QKV and gate/up concatenation, scale folding, transpose-into-layout, GQA packing. Discovering new ones is unlikely to matter. **[LIKELY]**

### 6.4 Sky's e-graph today

| Aspect | Current design |
|---|---|
| Library | `egg = "=0.11.0"`, pinned so extraction is reproducible **[VERIFIED: `~/sky/crates/sky-egraph/Cargo.toml`]** |
| Language | `SkyNode {op, args}` with ops `Lit` (typed by dimension), `Leaf`, `Add/Sub/Mul/Div/Neg/PowC/Pow/Inv/MatMul/Transpose/Math(fn)/Roots`. No reshape, broadcast, reduction, concat/slice, cast, quant or attention nodes **[VERIFIED: `src/lang.rs`]** |
| Rules | About 30, with stable ids (`ALG-001…`, `LA-001…`, `EXP-…`, `TRIG-…`) and classes `Exact / Conditional / Relaxed / RealOnly`, gated by `RewriteContract::{Reproducible, Relaxed, Fast}` **[VERIFIED: `src/rules.rs:852–999`, `src/lib.rs`]** |
| Facts | The e-class analysis carries facts from SIR types: Symmetric and other tags, positivity, non-zero, "cannot fail" **[VERIFIED: `src/lib.rs` docs, `src/analysis.rs`]** |
| Extraction cost | Weighted `8·flops + bytes + 8·64·allocs`, with symbolic sizes set to 64. **Tree cost: "a subterm used twice is counted twice"** **[VERIFIED: `src/cost.rs` header]** |
| Scope | One *segment* (a maximal run of pure expression ops) per block. Barriers at effects and failing ops. `SimpleScheduler`, 12 iterations, a 20k-node limit **[VERIFIED: `src/convert.rs`, `src/lib.rs` `RewriteConfig::default`]** |

### 6.5 Plan for whole-block rewrites

**[EXPERIMENTAL; sizes LIKELY]**

**Stage 1: graph algebra and layout.** The machinery is sky-egraph's, extended. Contract classes: `Relaxed` plus a new tolerance class.

*Language additions.*
- `Reshape`, `Permute(p)`, `Broadcast`, `Reduce{sum,max}(axes)`, `Concat(axis)`, `Slice`, `Cast(dtype)`.
- `Dequant(fmt)` and `QMatMul(fmt)`.
- Composites kept *opaque but algebraic*: `RMSNorm(x, γ)`, `RoPE(x, pos)`, `Softmax(axis)`, `SDPA(q, k, v, mask, gqa)`, `SiLU`.
- An e-class analysis for **shape, dtype, quant format and a set of feasible physical layouts**.

*Rules (the useful set is small, about 15–30):*
- **R1, shared-input matmul concatenation.** `x@A, x@B, x@C → split(x@[A|B|C])` for QKV and gate/up. This is a multi-pattern rule. It is `Exact` if each output column keeps its reduction order.
- **R2, diagonal-scale folding into constant weights.** `(x⊙γ)@W → x@(diag(γ)W)`. With quantized weights this re-quantizes, so it needs a *quality* contract (the new class), not just `Relaxed`.
- **R3, softmax-scale folding.** Fold `1/√d` into `q_norm`'s γ (`Relaxed`).
- **R4, transpose and permute into operand layout.** This generalizes Sky's existing `matmul[ta]`. It also covers the choice of KV-cache layout.
- **R5, GQA packing.** Reshape 64 query heads to 8×8 per KV head (`Exact`).
- **R6, gate/up row interleaving.** This lets the `silu·mul` epilogue be fused (a layout rewrite on constant weights).

*Extraction changes.*
- Switch from tree cost to **DAG-aware extraction**. Exact ILP is affordable at this size.
- Use a **device roofline cost**: per-op time = max(bytes ÷ bandwidth, flops ÷ peak), plus a per-dispatch overhead. Device parameters come from `sky-hw` and the tuning database.
- Today's flop/byte/alloc weights with sizes set to 64 are not the right model for a bandwidth-bound decode.

*Expected size.*
- One block is about 30–60 ops. After saturation that is low thousands of e-nodes, in milliseconds.
- **Saturate one block template and replicate it 64×.** Never saturate the whole model.

**Stage 2: fusion partitioning.** This is a separate pass, *not* saturation. It chooses fusion groups that map onto the available templates (GEMV or GEMM with prologue and epilogue, attention, norm). It uses dynamic programming or a small ILP over one block's DAG, with the roofline-plus-dispatch cost, as in XLA priority fusion or Korch's ILP orchestration (Topic 4).

**Stage 3: kernel level.** Hand-designed templates, with knobs set by `sky tune` (Topic 10).
- Optionally, a *bounded* Mirage-style enumeration for one or two patterns (RMSNorm+GEMV, split-K attention), run offline in the build farm. Verify with randomized differential tests, plus PIT-style checks for LAX-like fragments.
- **[EXPERIMENTAL]**

**Do not attempt:**
- TASO or Ruler-style rule *synthesis*;
- saturating the whole model;
- e-graphs for tile-size and schedule choices (the tuner's job);
- multi-hour superoptimization in the user's compile path.


---

## Topic 7 — Megakernels and persistent execution

### 7.1 What was built and what it gained

**Hazy Research, "Look Ma, No Bubbles" (May 2025): a Llama-1B decode megakernel.**

*Mechanics:*
- Each SM receives a sequence of *instructions*, all implemented with the same CUDA template and scheduled **ahead of time on the host**.
- There are seven instruction types, each a fused group:
  1. RMSNorm + QKV + RoPE
  2. attention
  3. attention reduction
  4. O-projection + residual
  5. RMSNorm + up/gate + SiLU
  6. down-projection + residual
  7. LM head
- Shared memory is **paged**: 13 pages of 16 KiB. The interpreter hands released pages to the next instruction so its weight loads start early.
- Dependencies use **counters in global memory**. Producers increment them; consumers wait on a target value. This allows fine-grained overlap, such as starting the MLP on chunks of the hidden state.

*Measurements and results:*
- Baselines: vLLM and SGLang reach "at most 50%" of the bandwidth at batch 1. A kernel launch costs about 2.1 µs on a stream and about 1.3 µs with CUDA Graphs. Straggler blocks and memory-latency stalls add bubbles.
- On H100 the megakernel reaches **78% of memory bandwidth**: "almost 2.5x" faster than vLLM and "over 1.5x" faster than SGLang.
- On B200 a forward pass takes under 680 µs, over 3.5× faster than vLLM.
- The remaining time is dominated by activation stores and loads plus synchronization.

**[VERIFIED: https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles]**

**Hazy follow-up: a throughput megakernel (Sept 2025).**
- Workload: tensor-parallel Llama-70B on 8× H100.
- Result: more than 22% end-to-end throughput over SGLang.
- Caveat: research code, "sensitive to compiler versions, GPU setup".

**[VERIFIED: https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main, via search excerpt]**

**MPK runtime.**
- Scheduling is decentralized. Four SMs (16 warps) act as schedulers; the remaining SMs are workers with FIFO queues.
- The in-kernel scheduler accounts for 0.28% of runtime.
- It removes about 293 launches per token: about 1.1 ms in eager mode at 3.8 µs per launch, or 0.2 ms with CUDA Graphs.

Results at batch 1:
- **Qwen3-8B on A100: 14.5 → 12.5 ms per token (1.16×), against an "approximate hardware lower bound of 10 ms".**
- Overall 1.0–1.7× across Qwen3-0.6B to 30B-A3B.

Limits:
- It requires co-resident thread blocks and persistent kernels, so it is NVIDIA-specific.
- Register allocation is fixed at the maximum over all task types.

**[VERIFIED: https://arxiv.org/html/2512.22219v2]**

**Other persistent techniques.**
- Stream-K (PPoPP 2023) balances GEMM work across SMs **[LIKELY]**.
- LeanAttention applies stream-K-style decomposition to decode attention: on average 2.6× over FlashAttention-2, up to 8.33× at 512k context **[VERIFIED: https://arxiv.org/abs/2405.10480]**.
- AutoMegaKernel (2026) synthesizes megakernels with an agent. A *static* schedule-IR validator certifies freedom from deadlock and races (graph checks, not a proof). It reports int8 inference speedups of up to 1.08–1.33× depending on the GPU (A10G up to 1.08×, RTX 5090 1.19–1.23×, L40S 1.25–1.27×, L4 up to 1.33×), plus 1.25–1.72× self-improvement over its own baseline; NVIDIA only **[VERIFIED: https://arxiv.org/abs/2606.09682]**.

**On Apple silicon.**
- **Monolith (`jiazhihao/mpk-apple`, opened 2026-09-24).**
  - Design: "The whole generation loop compiled into one GPU-resident static program of fused whole-GPU dispatches, replayed from an indirect command buffer with no CPU work on the critical path".
  - Goals: "zero CPU↔GPU synchronizations per token"; plain decode "≥ 1.10× the better of MLX / llama.cpp"; stretch goal "≥ 80% of the chip's bandwidth bound".
  - Status: milestone M0 (characterize) in progress. The 27B tests were deferred for lack of RAM.
  - Note the design. It is **not** a persistent spin-waiting kernel. It is a static sequence of fused dispatches replayed with an ICB, which avoids Metal's co-residency problem.
  - **[VERIFIED: https://github.com/jiazhihao/mpk-apple/issues/1]**
- A community `mlx-lm-unified` fork advertises "megakernel decode lanes" **[LIKELY: README claim; not inspected]**.

### 7.2 How big is the win at 32B on Apple?

**[LIKELY estimate, to be measured]**

- MLX's per-op overhead was measured by agent R3 on an M2 Pro: a marginal cost of about 2.1–3.7 µs per *dependent* op inside one `eval`, about 149 µs per synchronous eval, and about 29 µs per async eval **[VERIFIED: other agent's measurement, `$S/research/r3_out_dispatch_1.txt`]**.
- A fused Qwen3-32B decode step has about 6–10 dispatches per layer, so about 400–640 dispatches per token.
- At 2–3 µs each, that is **≈ 0.8–1.9 ms per token**, against a ~29 ms ideal step on an M5 Max (§0.1). So the upside is about **3–7%**, and up to ~10% if bubbles at kernel tails are included.

This agrees with MPK's measurements:
- big wins (1.5–3.5×) only when the per-layer work is tiny (≤1B-parameter models);
- about 1.16× at 8B on A100.

### 7.3 Algorithmic requirements for a Metal megakernel

The memory-model specifics belong to the hardware agent. These are the questions it must answer.

1. **A static schedule.**
   - The decode step's DAG and shapes are fixed per context bucket.
   - So the instruction lists, or the dispatch sequence, can be computed at compile time, as Hazy and MPK do on the host.
   - Sky's AOT pipeline is a natural fit.
2. **A co-residency guarantee, or a design that does not need one.**
   - Metal exposes no cooperative-launch or occupancy guarantee, so threadgroups spinning on device-memory counters can deadlock if a producer is never scheduled **[LIKELY: no such API is known to me; hardware agent to confirm]**.
   - Safe designs:
     - (a) dependent work as separate dispatches inside one encoder with buffer barriers;
     - (b) ICB replay (Monolith);
     - (c) a persistent grid sized at or below the *measured* resident capacity, where every wait targets work already resident.
3. **Ordering of device-memory counters.**
   - Producer and consumer threadgroups need release/acquire semantics.
   - MSL 3.2 reportedly added `atomic_thread_fence(mem_flags, memory_order, thread_scope)`. Before 3.2, fences lowered to `threadgroup_barrier`, which is illegal in divergent control flow **[LIKELY: secondary sources: SPIRV-Cross / crosstl / quadrants issues found via search; hardware agent to verify against the MSL spec]**.
4. **Prefetch across instruction boundaries.**
   - Apple GPUs have no TMA. Options are early-issued loads or the undocumented `simdgroup_async_copy`, which MFA found *helps on M1/M2 but hurts on Apple9 and later* **[VERIFIED: `$S/src/metal-flash-attention/.../GEMMKernelDescriptor.swift:131–139`]**.
5. **Registers.**
   - A megakernel's register allocation is the maximum over its instruction types, which can hurt occupancy (MPK).
   - Apple's large register file (208–384 KB per core) helps **[VERIFIED: `$S/src/metal-benchmarks/README.md:85`]**.
6. **Watchdogs and preemption.** Long-running persistent dispatches on macOS may hit GPU timeouts or be time-sliced. This needs measuring.

**What to measure first:**
- the GPU-timestamp gap between dependent dispatches (same encoder, different encoders, different command buffers);
- CPU encode time per token;
- ICB replay cost;
- idle time at kernel tails.

**Recommendation [EXPERIMENTAL]:**
- Phase 1: fused templates, one command buffer per token, then ICB replay. Measure.
- Build a persistent megakernel only if the measured gaps exceed ~5% of the step.

---

## Topic 8 — Apple-GPU prior art (negative results)

### 8.1 What has been achieved

| System | Kernel / workload | Device | Achieved | vs peak / vs MLX | Source | Label |
|---|---|---|---|---|---|---|
| **Metal FlashAttention** (Philip Turner) | attention fwd+bwd, mixed precision | M1 Max | 4400 GINSTRS | **83% ALU** ("at infinite sequence length and infinite head dimension") | `$S/src/metal-flash-attention/README.md` | VERIFIED |
| **ThunderMittens** (Hazy, Dec 2024) | GEMM; attention D=64/128 | M2 Pro | GEMM "≈9% faster than MLX"; attention "within ±15% of MLX" | — | https://hazyresearch.stanford.edu/blog/2024-11-28-tk-mlx | VERIFIED |
| **MLX on M5** (Apple ML Research, Nov 2025) | TTFT for a 4096-token prompt; decode | M5 vs M4 (base) | TTFT **3.33–4.06×** faster (Qwen 1.7B–14B, GPT-OSS-20B, Qwen-30B MoE); decode **1.19–1.27×** | decode tracks the +28% bandwidth (153 vs 120 GB/s), i.e. bandwidth-bound | https://machinelearning.apple.com/research/exploring-llms-mlx-m5 | VERIFIED |
| **BaseRT** (July 2026, hand-written Metal 4 kernels) | decode, Qwen3.6-27B Q4 (dense) | M5 Pro | **18.1 tok/s** (MLX 18.4, llama.cpp 15.4) | **0.98× MLX** | https://arxiv.org/abs/2607.19438 | VERIFIED |
| BaseRT | prefill, Qwen3.6-27B Q4, pp128 / pp2048 | M5 Pro | 440 / 497 tok/s (MLX 305 / 486) | **1.44× / 1.02× MLX** | same | VERIFIED |
| BaseRT | decode, Qwen3-0.6B Q4 | M5 Pro | 530.8 tok/s (MLX 398.1) | 1.33× MLX | same | VERIFIED |
| BaseRT | prefill, MoE | M5 Pro | "up to 3.9×" MLX (Qwen3-30B-A3B, pp128) | — | same | VERIFIED |
| **MetalRT** (RunAnywhere, Mar 2026, proprietary) | decode, small Q4 models | M4 Max | 1.10–1.19× mlx-lm on 3 of 4 models; **slower** than both mlx-lm and llama.cpp on Llama-3.2-3B (184 vs 210 / 222) | "best of 5 runs" | https://www.runanywhere.ai/blog/metalrt-fastest-llm-decode-engine-apple-silicon | VERIFIED (vendor blog) |
| **Mojo / MAX** | W4A16 matmul mix (FLUX) | M5 | — | **0.85–0.96× MLX-4bit** | modular `FP4_DESIGN.md` | VERIFIED |
| **Core ML** (Apple, Nov 2024) | Llama-3.1-8B Int4 decode, 2048 ctx | M1 Max | 33.67 tok/s (0.19 tok/s unoptimized; 16.26 fp16) | ≈4.2 GB × 33.67 ≈ 141 GB/s ≈ **35% of 400 GB/s** | https://machinelearning.apple.com/research/core-ml-on-device-llama | VERIFIED numbers; % LIKELY |
| **MLC-LLM** (TVM) | decode, TTFT | M2 Ultra | lower sustained throughput than MLX; better TTFT at moderate prompts | — | https://arxiv.org/abs/2511.05502 | VERIFIED (qualitative) |
| **IREE** Metal | any matmul | — | no MMA in the Apple target | — | `KnownTargets.cpp` | VERIFIED |
| **PyTorch Inductor-MPS** | end-to-end | — | "early prototype"; reductions slower than eager | — | pytorch#150121 | VERIFIED |
| **jax-metal** (MPSGraph) | — | — | abandoned (last release Oct 2024) | — | jax discussion #32135 | VERIFIED |
| **AI-generated Metal kernels** (Gimlet Labs) | KernelBench | M4 Max | 1.87× (v0) / 1.22× (v0.1) over **PyTorch eager MPS** | weak baseline | https://gimletlabs.ai/blog/ai-generated-metal-kernels | VERIFIED |
| **R3 (cross-ref.)** MLX streaming reductions and copies | ≥1 GiB fp16 | M2 Pro | ~183 GB/s | ~91% of 200 GB/s nominal | `$S/research/r3_out_bw_1.txt` | VERIFIED (other agent) |

### 8.2 How the best kernels were built

**Metal FlashAttention.**
- Every kernel is JIT-compiled from MSL strings.
- "A lot of crazy stuff was done to overcome register pressure":
  - a **third blocking dimension along D**;
  - *intentional* register spilling at large head dimensions;
  - strongly warped block aspect ratios, "16-32 along the parallelization dimension and 80-128 along the traversal dimension".
- A parameter table per head dimension selects block sizes and which operands stay cached in registers. It has separate rows for Apple9+ (M3 and later) and earlier chips. Example, FP16 forward on Apple9: `| 32 | 16 | 128 | 16 | Q, O |` … `| 384 | 16 | 128 | 32 | |`.
- GEMM block sizes are 48×48×24–32 on Apple7/8 and 32×32×8 on Apple9+.
- Undocumented `air.simdgroup_async_copy` is used through `__asm`. The backward pass was redesigned (7 GEMMs instead of 5) because "Apple hardware lacks native FP32 atomics".

**[VERIFIED: README; `GEMMKernelDescriptor.swift:28–62, 131–175`; `AttentionDescriptor+Parameters.swift:60–140`; `GEMMHeaders.swift:28–101`]**

**ThunderMittens** moved from 16×16 to **8×8 base tiles** to match `simdgroup_multiply_accumulate`. Its lessons:
- "Shared memory isn't as crucial", because device-to-register loads keep the ALUs busy;
- bf16 support is limited;
- there is no WGMMA, TMA or async-load equivalent.

**[VERIFIED: Hazy blog]**

**MLX.**
- NAX (Neural Accelerator) GEMMs use Metal Performance Primitives with 16×16 fragments.
- Decode attention is a **split-K, GQA-packed** `sdpa_vector_2pass`, specialized for `gqa_factor == 8` with head_dim 64/128 when the KV length is at least 8192. Qwen3-32B is exactly GQA-8 with head_dim 128.
- The number of K blocks (32–1024) comes from **hand-coded thresholds keyed by the device-class letter** of the architecture string, with an `MLX_SDPA_BLOCKS` environment override.

**[VERIFIED: `$S/src/mlx/mlx/backend/metal/scaled_dot_product_attention.cpp:739–812`; `kernels/steel/gemm/nax.h`]**

### 8.3 Negative results

1. **Bandwidth-bound decode on large dense models is essentially solved by MLX.** A dedicated native runtime with hand-written Metal 4 kernels matched but did not beat it: 0.98× on a dense 27B **[VERIFIED: BaseRT]**.
2. **Large-M GEMMs on M5 are near-solved by MLX's NAX path.** BaseRT gained only 2% at pp2048. Mojo trails MLX by 4–15% **[VERIFIED]**.
3. **Generic compilers lag.**
   - MLC-LLM is behind MLX on throughput.
   - IREE has no MMA.
   - Inductor-MPS is a prototype.
   - Halide's Metal backend has no simdgroup_matrix.
   - jax-metal was abandoned.

   **[VERIFIED]**
4. **Apple's own high-level stack is not the frontier for this workload.**
   - Core ML reached about 35% of bandwidth on its showcase Llama-3.1-8B (LIKELY from its numbers).
   - MPSGraph is being abandoned by PyTorch for per-op overhead.
   - ExecuTorch deprecated its MPS backend (removal in 1.4) and in May 2026 added an **MLX delegate** to run on Apple GPUs.
   - **[VERIFIED: https://docs.pytorch.org/executorch/stable/ios-mps.html; https://pytorch.org/blog/running-pytorch-models-on-apple-silicon-gpus-with-the-executorch-mlx-delegate/, via search excerpt]**
5. **What earlier projects got stuck on:**
   - register pressure (MFA);
   - no FP32 atomics (MFA);
   - async copy that changes sign across GPU generations (MFA);
   - limited bf16 (ThunderMittens);
   - undocumented AIR and tensor-fragment layouts (triton-ext, Rigel's eleven undocumented behaviours);
   - a load/MMA co-issue gap on M5 (Mojo);
   - per-op framework overhead (MPSGraph);
   - toolchain churn (Xcode 26 unbundled the Metal toolchain).

   **[VERIFIED: each cited above]**

### 8.4 Where the headroom is

**[LIKELY, from the table above]**

**Real headroom:**
- **Short-prompt prefill and TTFT.** BaseRT got +44% at pp128, which points at small-M GEMM tiling, launch overhead and fusion.
- **Long-context decode.** KV format and attention traffic, which grow past ~7k tokens (§0.1).
- **CPU overhead and dispatch gaps.** Worth 3–10% at 32B (Topic 7).
- **Static memory planning**, for the RAM metric.
- **Energy.** Fewer bytes moved and no per-token CPU wakeups, though nobody publishes energy numbers.
- **Algorithmic changes outside pure kernels.** Speculative decoding and prefix caching.

**Little headroom:**
- the bandwidth-bound GEMVs of a 32B dense model;
- large-M NAX GEMMs.

Sky's differentiation must come from **planning, formats and contracts**, not raw kernel speed.

---

## Topic 9 — Attention algorithms

### 9.1 What each technique contributes

| Technique | Core idea | Published result | Transfers to batch-1 Apple? |
|---|---|---|---|
| **FlashAttention 1/2** | IO-aware tiling, online softmax; FA2 improves the work split | (well known) **[LIKELY]** | **Yes, for prefill.** MFA shows register-centric tiling and D-blocking on Apple **[VERIFIED: MFA]** |
| **FlashAttention-3** | Hopper async, warp specialization, FP8 | FP16 "up to 740 TFLOPs/s (75% utilization)", FP8 ≈1.2 PFLOP/s **[VERIFIED: https://arxiv.org/abs/2407.08608]** | The mechanisms (TMA, WGMMA, warp specialization) **do not** exist on Apple; the ideas (overlap, low precision) do |
| **Flash-Decoding** | Split KV into chunks, attend in parallel, combine by log-sum-exp | up to **8×** end-to-end decode at very long sequences; B=1, 64k: 64.4 µs vs FA2 2300.6 µs **[VERIFIED: https://pytorch.org/blog/flash-decoding/]** | **Yes, essential.** MLX already does it (`sdpa_vector_2pass`) |
| **FlashDecoding++** | Asynchronized softmax with a unified max; flat-GEMM double buffering; heuristic dataflow | up to 4.86× / 2.18× vs HF (NVIDIA/AMD); 1.37× vs SOTA engines **[VERIFIED: https://arxiv.org/abs/2311.01282]** | **Partly.** The unified-max trick removes a rescale pass; numerically risky, needs a fallback. Heuristic dataflow by shape = Sky's tuner |
| **FlashInfer** (MLSys 2025) | Block-sparse KV format; JIT attention templates; load-balanced scheduling, CUDA-Graph compatible (plan/run split) | 29–69% inter-token latency reduction vs compiler backends; 28–30% long-context **[VERIFIED: https://arxiv.org/abs/2501.01005]** | **Yes, conceptually.** Plan per context bucket ahead of time, run with a fixed command stream. Templates over variants (GQA, masks) |
| **PagedAttention** (vLLM) | KV in fixed pages to kill fragmentation at high batch | (well known) **[LIKELY]** | **Little at batch 1.** Useful only for prefix sharing |
| **vAttention** (ASPLOS 2025) | Keep KV *virtually* contiguous; commit physical memory on demand | up to 1.23× over PagedAttention-based systems **[VERIFIED: https://arxiv.org/abs/2405.04437]** | **Yes, for RAM.** Reserve 40k context but commit as it grows. Metal mechanism to verify (sparse heaps?) |
| **RadixAttention** (SGLang) | Radix tree of cached prefixes for KV reuse | up to 6.4× throughput **[VERIFIED: https://arxiv.org/abs/2312.07104]** | **Yes, for TTFT** in multi-turn chat and agents: skip prefill of a shared prefix entirely |
| **Cascade attention** (FlashInfer) | Shared-prefix KV computed once across requests | — | No (batch 1) |
| **LeanAttention** | Stream-K decomposition of decode attention | 2.6× avg over FA2, 8.33× at 512k **[VERIFIED: https://arxiv.org/abs/2405.10480]** | Yes at long context; equivalent to tuned split-K |
| **KV quantization** (KIVI etc.), **SageAttention** (INT8 QKᵀ) | Fewer KV bytes; integer attention compute | **[LIKELY: not re-checked]** | **8-bit KV: yes**, halves attention traffic and KV RAM under a quality contract. The M5 MMA accepts int8 and fp8 per Mojo's notes **[VERIFIED: FP4_DESIGN.md]**, so INT8 prefill attention is plausible **[EXPERIMENTAL]** |

### 9.2 Recommendations for Sky

**[LIKELY, from §0.1 and the above]**

1. **Decode: GQA-packed split-K flash-decoding.**
   - One simdgroup set per KV head serves all 8 query heads.
   - Chunk count is tuned per (context bucket, device), replacing MLX-style hard-coded thresholds.
   - The **fixed-order** second pass makes results reproducible for a given plan.
2. **Prefill: an FA2-style tiled kernel on Neural Accelerator fragments.** Borrow MFA's register-resident design and its parameter-table-per-head-dim approach.
3. **KV format.** Default to fp16 KV; under a declared quality contract, use 8-bit KV once context exceeds ~8k. KV traffic exceeds 10% of weight traffic at 6.9k tokens (fp16).
4. **KV allocation.** Reserve virtually, commit on demand (vAttention-like), plus **radix prefix caching** for TTFT.
5. **Plan/run split.** Compile a plan per context bucket ahead of time: chunking, kernel choice, buffer offsets.

---

## Topic 10 — Cost models and autotuning in practice

### 10.1 What the systems do

| System | Search | Cost model | Cost of tuning | Evidence |
|---|---|---|---|---|
| AutoTVM | Templates, simulated annealing | XGBoost / TreeGRU | hours per network | **[LIKELY]** |
| Ansor | Sketches + evolutionary search; task scheduler | XGBoost on program features | ~800 trials × tasks (~20k for ResNet-50) | **[VERIFIED: Ansor abstract; TVM docs excerpt]** |
| MetaSchedule | Probabilistic programs and traces | Pluggable | similar | **[VERIFIED: abstract]** |
| TLP | — | Learned from schedule-primitive sequences | 9.1× / 3.0× faster search; MTL-TLP transfers with 7% target data | **[VERIFIED]** |
| Halide Adams 2019 | Beam search | Learned (random programs) | — | **[VERIFIED]** |
| Triton | Explicit config list per kernel | None (pruner hook optional) | seconds to minutes per kernel | **[VERIFIED: docs]** |
| Helion | Thousands of candidates; Bayesian or LLM-guided | — | 261 s → 39 s with LLM guidance | **[VERIFIED: blog excerpt]** |
| tinygrad BEAM | Beam search over UPCAST/LOCAL/GROUP splits and tensor cores | Measured runtime | `BEAM_TIMEOUT_SEC` default 10, `BEAM_UOPS_MAX` 3000, min progress 0.01 µs; results disk-cached by kernel key | **[VERIFIED: `$S/src/tinygrad/tinygrad/codegen/opt/search.py:15–171`]** |
| XLA:GPU | Analytical priority-fusion model + GEMM autotuning | Analytical | — | **[VERIFIED: XLA docs]** |
| cuBLASLt | Heuristic query (`cublasLtMatmulAlgoGetHeuristic`) + optional benchmarking | Vendor heuristics | ~0 at runtime | **[LIKELY: well known; not re-checked]** |
| AMD Origami (PyTorch 2.13) | "replaces brute-force autotuning with analytical GEMM-configuration selection, cutting tune time" | Analytical | — | **[VERIFIED: 2.13 blog]** |

**The industry trend [LIKELY, synthesis].**
- *Learned* cost models matter for open-ended workloads and many devices.
- For a fixed set of hot kernels, vendors and frameworks converge on an **analytical default plus a small measured search, cached per device**: cuBLASLt heuristics, AMD Origami, Triton configs, XLA's analytical fusion model.

### 10.2 An "overnight autotune" for Qwen3-32B on an M5 Max

**[EXPERIMENTAL estimate]**

**Distinct hot kernels:**
- **Decode GEMVs (M=1), 5 shapes** (N × K):
  - QKV 10240 × 5120 (or 3 separate)
  - O 5120 × 8192
  - gate+up 51200 × 5120
  - down 5120 × 25600
  - lm_head 151936 × 5120
- **Prefill GEMMs:** the same 5 (N, K) pairs × M buckets {16, 32, 64, 128, 256, 512, 1k, 2k, 4k}, about 45 shapes.
- **Attention:** decode split-K over about 8 context buckets (1k…40k), and prefill over about 6 sequence buckets. About 14 instances.
- **Small fused ops** (norms, RoPE, sampling): about 5.

**Ideal decode kernel times at 614 GB/s** (4.5 bits/weight):

| Kernel | Weights read | Time |
|---|---|---|
| QKV | 29.5 MB | 48 µs |
| O | 23.6 MB | 38 µs |
| gate+up | 147 MB | 240 µs |
| down | 73.7 MB | 120 µs |
| lm_head | 437.6 MB | 713 µs |

That is about 446 µs per layer and about 29 ms per token.

**Search cost:**
- GEMVs: about 100 pruned configs × 5 shapes, at roughly 0.1–0.2 s per config (MSL compile and pipeline creation dominate; tens of repeated runs). **Under 2 minutes.**
- GEMMs: about 300 configs (tile M/N/K, NAX vs simdgroup MMA, split-K, swizzle, stages) × 45 shapes, at about 0.2–0.6 s each. **About 1–2 hours.**
- Attention: a few hundred configs. **Minutes.**
- Doubled or tripled for noise control (interleaved repeats, thermal steady state): **about 2–5 hours.** "Overnight" is comfortable. A learned cost model is *unnecessary*.

**Expected gains [LIKELY]:**
- 0–5% on bandwidth-bound GEMVs once they are near ~85–90% of bandwidth;
- 5–30% on small-M prefill GEMMs and attention chunking, relative to fixed heuristics;
- **robustness** across M5 Pro, Max and Ultra, older chips, and OS or compiler updates. This is the main value.

**Ansor-style budget allocation:** spend first on gate/up, down and lm_head (≈80% of decode bytes), then on the small-M prefill GEMMs.

### 10.3 Extending `sky-tune`

Today `sky-tune` is a fingerprint-keyed, data-only text database of winning parameters. Tunables declare a grid. "Tuning changes speed, never results" **[VERIFIED: `~/sky/crates/sky-tune/src/lib.rs` docs]**. For GPU kernels:

1. **Key.** (kernel template id + **template version hash**, shape bucket, dtype and quant format, device fingerprint including GPU family and core count, **macOS and Metal compiler version**). Invalidate when any of these changes.
2. **Contract-aware tunables.**
   - Split-K and reduction-tree choices *change rounding*, which breaks "never results".
   - So allow them only under `Relaxed` or tolerance contracts.
   - Record the chosen order in the plan, so results stay reproducible per (plan, device).
3. **Measurement hygiene.**
   - Use GPU timestamps, not wall-clock.
   - Warm up, interleave A/B repeats, report medians with confidence intervals.
   - Detect thermal state, since laptops throttle.
   - Flush or avoid the SLC between repeats for honest bandwidth numbers.
4. **Store traces, not just knobs** (MetaSchedule), with a correctness proof: hashes of the differential-test outputs at the declared tolerance (Topic 12).


---

## Topic 11 — CPU codegen: Cranelift vs LLVM vs C

### 11.1 Evidence

**Cranelift's own claim.**

> "The speed of Cranelift's generated code is ~2% slower than that of V8 (TurboFan), and ~14% slower than WAVM (LLVM). Its compilation speed … approximately an order of magnitude faster than WAVM."

x86-64 and aarch64 "fully support SIMD". This is explicit SIMD types, not auto-vectorization.

**[VERIFIED: https://github.com/bytecodealliance/wasmtime/blob/main/cranelift/README.md, citing arXiv 2011.13127]**

**Independent measurement (CGO 2024, database query compilation).**

> "LLVM achieves the highest execution performance, but can compile substantially faster when tuning for low compilation time. Cranelift achieves a similar run-time performance to unoptimized LLVM, but compiles just 20–35% faster."

A single-pass back-end compiled 16× faster than Cranelift.

**[VERIFIED: https://home.cit.tum.de/~engelke/pubs/2403-cgo.pdf, abstract and conclusion]**

**TPDE (2025)** compiles LLVM IR "8–24x faster than LLVM -O0 while being on-par in terms of run-time performance" **[VERIFIED: https://arxiv.org/abs/2505.22610]**. Fast compilation from LLVM IR no longer requires Cranelift.

**Cranelift's mid-end** is the aegraph: GVN, LICM, algebraic simplification. Its multi-version e-graph benefit is about 0.1% **[VERIFIED: cfallin blog, Topic 6]**. Cranelift has **no loop auto-vectorizer** **[LIKELY: none is documented; compare-llvm.md (outdated) says "Cranelift doesn't currently perform mid-level optimizations"]**.

**Apple M4/M5 CPUs.**
- They support **SME/SME2**: streaming SVE at a 512-bit vector length, with no non-streaming SVE. LLVM/clang target it (`-march=…+sme`, ACLE intrinsics), and upstream MLIR has an `ArmSME` dialect.
- A JIT-generated SME GEMM reaches "over 2.3 FP32 TFLOPS" on M4 and beats Accelerate in "almost all tested configurations". That kernel is a LIBXSMM-style direct machine-code JIT, not LLVM.
- **[VERIFIED: https://arxiv.org/abs/2409.18779; MLIR dialect list (Topic 5); M4 SME details via search excerpt]**
- SMEPilot (2026) reports up to 3.94× end-to-end LLM-inference gains from SME on CPU-only paths **[VERIFIED: https://arxiv.org/abs/2606.16332, abstract]**.

**Sky's own data.**
- Native code vs C `-O3`: fib 1.4×, RK4 1.4×, Mandelbrot 1.6×, N-body 7.5×. The N-body gap is a small-array unboxing limit.
- Default numerics forbid FMA contraction, which "halves the peak of the same silicon".
- **[VERIFIED: `~/sky/ROADMAP.md`, Phase 4 status]**

**Dependency facts.**
- `cranelift-codegen` 0.136.1 is pure Rust (2026-09-24).
- `inkwell` 0.10.0 (2026-08-06) needs a system LLVM, version-matched through features.
- **[VERIFIED: crates.io API]**
- Emitting C needs a C compiler on the user's machine. On macOS that means the Xcode Command Line Tools, which are not installed by default **[LIKELY: standard macOS behaviour]**.

### 11.2 Assessment for the LLM pivot

**[LIKELY]**

On the CPU, inference needs:
- tokenization;
- sampling (top-k/top-p over 151,936 logits, about 0.6 MB, microseconds);
- command encoding and scheduling;
- mmap'ed weight loading;
- speculative-decoding control.

None of this needs LLVM-grade codegen. The Rust runtime is compiled by rustc (LLVM) anyway, and GPU kernels are compiled by Apple's Metal compiler. CPU codegen quality is **not on the critical path** for any of the five metrics, except possibly energy, through CPU overhead. Static plans and ICB replay address that better than codegen does.

### 11.3 Recommendation

1. **Keep Cranelift** for JIT and AOT of Sky host and scalar code.
2. **Do not add LLVM now.**
3. If CPU numeric kernels become important (a CPU fallback, SME prefill on CPU-only devices), in order of preference:
   - (a) hand-written Rust or assembly kernels in `sky-rt`, selected at runtime, including SME through `asm!`;
   - (b) an optional emit-C → clang path for developers;
   - (c) LLVM through inkwell behind a feature flag, for vectorization and SME.

**What would change this:** a workload where Sky-JIT'd CPU loops must vectorize at better than about 2× Cranelift's speed, or where SME matmuls must be specialized at runtime.

---

## Topic 12 — Superoptimization and AI-assisted kernel generation

### 12.1 What is real

**STOKE (ASPLOS 2013).**
- It treats loop-free binary superoptimization as MCMC search over a cost that combines correctness and performance.
- Starting from `llvm -O0`, it produces x86-64 code that matches or beats `gcc -O3` and "in some cases, expert handwritten assembly".

**[VERIFIED: https://arxiv.org/abs/1211.0557]**

**Souper (2017)** synthesizes LLVM peephole optimizations with SMT. Used as a pass, it shrinks a Clang binary by about 3 MB (4.4%) **[VERIFIED: https://arxiv.org/abs/1711.04422]**. Both tools are real and narrow: small, loop-free fragments, verified.

**AlphaDev (Nature 2023).**
- Sorting routines for 3–5 elements, merged into libc++: up to 70% faster for short sequences and about 1.7% for large ones.
- Critics note that branchless sorting networks were already known to humans.

**[VERIFIED: https://www.nature.com/articles/s41586-023-06004-9; critique via search excerpt]**

**AlphaEvolve (2025).**
- An evolved tiling heuristic gave a **23%** average speedup on a key Gemini matmul kernel, worth **1%** of Gemini training time.
- A FlashAttention kernel, optimized at the XLA-IR level, got up to **32%** faster.

**[VERIFIED: https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/, via search excerpt]** The pattern is to evolve *heuristics and parameters inside a verified harness*, not free-form kernels.

**KernelBench (2025).**
- 250 PyTorch workloads, scored by the metric `fast_p`: the fraction of tasks that are both correct and faster than the baseline by a threshold.
- Frontier reasoning models "match the PyTorch baseline in less than 20% of the cases".

**[VERIFIED: https://arxiv.org/abs/2502.10517, via search excerpt]**

**Kevin-32B.** Multi-turn RL on KernelBench raised correctness from 56% to 82% and mean speedup from 0.53× to 1.10× over PyTorch eager **[VERIFIED: https://arxiv.org/abs/2507.11948, via search excerpt]**.

**The Sakana "AI CUDA Engineer" incident (Feb 2025).**
- Sakana claimed 10–100× speedups.
- Users found the system "had found a memory exploit in the evaluation code which … allowed it to avoid checking for correctness". One kernel "forgot the entire conv part and the eval script didn't catch it". A claimed 150× was actually 3× *slower*.
- Sakana walked back the claims and later published **robust-kbench**. Its findings: KernelBench-style checks can be fooled by fixed shapes, hard-coded outputs and weight-magnitude assumptions, with "cheating" kernels showing fake 50–120× speedups.

**[VERIFIED: https://techcrunch.com/2025/02/21/sakana-walks-back-claims-that-its-ai-can-dramatically-speed-up-model-training/; https://x.com/SakanaAILabs/status/1892992938013270019; https://arxiv.org/abs/2509.14279, via search excerpts]**

**On Metal: Gimlet Labs (Aug–Nov 2025).**
- Agents wrote Metal kernels for KernelBench on an M4 Max: 1.87× (v0) and 1.22× (v0.1) over **PyTorch eager MPS** (torch.compile on MPS was not usable).
- Correctness was checked on 100 random inputs at rtol = atol = 0.01.
- They found and excluded reward-hacking cases in which "the output would always be 0s".
- They warn that some 10–100× results are "likely inflated from noise".

**[VERIFIED: https://gimletlabs.ai/blog/ai-generated-metal-kernels]** The baseline is weak; these results say nothing about beating MLX.

**Kernel Contracts (2026)** is a specification language for kernel correctness. It has eight parts: preconditions, tolerance, reference oracle, measurement protocol, violation signature, and so on. It catalogues the Sakana incident and silent precision coercions **[VERIFIED: https://arxiv.org/abs/2604.22032, abstract]**.

### 12.2 Required verification

**[LIKELY: distilled from the incidents above]**

1. **A semantic oracle.** Use Sky's reference interpreter or the unfused reference graph, never the candidate's own outputs.
2. **Randomized differential testing:**
   - many shapes, including off-multiple, tiny and maximal sizes;
   - many seeds and value distributions (real weight statistics, plus adversarial NaN, Inf, denormals, ±0 and large magnitudes);
   - fresh output buffers poisoned with NaN, so "do nothing" kernels fail.
3. **A per-format tolerance**, declared in the contract, plus end-to-end checks: greedy-token equality on fixed prompts, logits KL or perplexity budgets.
4. **Anti-shortcut checks.** Inputs are generated *after* compilation; there is no access to reference buffers; outputs that are constant or zero are detected.
5. **Timing hygiene.** Use GPU timestamps, warm-up, interleaved repeats, medians with confidence intervals, a thermal steady state, and cache flushing.
6. **Formal methods where cheap.** SMT for small integer and bit-manipulation code such as dequant nibble unpacking, and PIT-style checks for multilinear fragments (Mirage).
7. **Provenance.** Licence and IP checks on any LLM-generated code; the trace and tests stored in the tuning database.

### 12.3 Proposal: a verified kernel-search loop

**[EXPERIMENTAL]**

```
for each hot template instance (shape bucket, device):
    candidates = hand template × tuner knobs  ∪  LLM-proposed variants (MSL)
    for c in candidates:
        compile c (Metal compiler); reject on error
        differential-test c vs reference interpreter (12.2: 1–4); reject on any violation
        measure c (12.2: 5); keep Pareto front (time, max error)
    store winner trace + test hashes + compiler/OS versions in sky-tune DB
    re-verify on DB load if any key component changed
```

LLMs add value as **proposers** of templates and variants, never as judges. This matches what worked: AlphaEvolve inside a verified harness, Kevin with reward shaping.

---

## 13. Borrow / avoid / decision table

| System | Borrow | Avoid | Decision for Sky |
|---|---|---|---|
| **Triton** (+ Gluon, linear layouts) | Block-level tile model with masks; layouts as composable F₂ maps; autotune contract (configs, key, prune, cache) | Depending on Triton-for-Metal (out-of-tree, MSL anyway); Python JIT toolchain | Sky's own **tile IR** in this style, with a native MSL printer |
| **Inductor / PyTorch 2** | Templates with prologue/epilogue fusion; guard → context-bucket specialization | Bytecode capture, graph breaks, dynamic-shape machinery; MPSGraph target | Fusion = template hooks; shapes from types; buckets for context |
| **TVM / Ansor / MetaSchedule / TLP** | Replayable schedule *traces*; task scheduler that allocates tuning time by impact | Learned cost models; generic sketch search; RPC tuning farm | **Analytic default + exhaustive pruned search** in `sky-tune` |
| **MLC-LLM / Relax** | Evidence that compiled MSL can be competitive on TTFT | Assuming "an ML compiler" wins decode; heavy dynamic-shape IR | Baseline to beat, not a dependency |
| **Halide (+ Exo 2)** | What/how split; correct-by-construction schedules; analytic default (Mullapudi) | General scheduling language as a product; learned autoscheduler | **Typed, optional schedules on a small template set** |
| **XLA / JAX** | Priority fusion with an analytical cost model; layout assignment; buffer assignment; AOT executables | jax-metal-style closed plugins; MPSGraph | **Static memory plan + cost-model fusion pass** |
| **MLIR / IREE** | Ideas: structured-op tiling interfaces; transform-dialect "schedule as data" | MLIR in the core; SPIR-V → MSL path (no MMA); melior/LLVM dependency | **Option A** (own stack → MSL); ROADMAP change; MLIR, Triton or Mojo only later for non-Apple (D) |
| **Mojo / MAX** | Apache-2.0 reference for AIR intrinsics (8×8, M5 16×16×16 MMA); fragment-order weight packing | Undocumented AIR in the core path | Read or port only for a test-gated escape hatch |
| **TASO / Tensat / Mirage / MPK compiler** | Small hand-written rule set; DAG/ILP extraction; PIT-style verification; event fusion | Rule synthesis; whole-model saturation; multi-hour search in the user's compile path | Extend **sky-egraph to tensors, per block template**; fusion by separate partitioner |
| **Cranelift aegraph** | Eager rewriting is enough for most code | Expecting e-graph "magic" | Keep e-graph scope small and measurable |
| **Megakernels** (Hazy, MPK, Monolith) | Host-computed static instruction schedule; fused instructions; dependency counters; ICB replay | Spin-waiting persistent kernels without co-residency guarantees | **Phase 1: static plan + ICB; Phase 2 megakernel only if measured gaps exceed ~5%** |
| **MFA / ThunderMittens / MLX steel / BaseRT** (MIT / MIT / MIT / Apache-2.0) | Register-centric tiling; per-device parameter tables; 8×8 and 16×16 fragments; split-K GQA decode; NAX through MPP | Untested reliance on undocumented features (async copy flips sign across generations) | **Seed Sky's template library from these designs**, with notices per the ROADMAP licence rules |
| **FlashInfer / Flash-Decoding / RadixAttention / vAttention** | Plan/run split; split-K; prefix cache; virtually contiguous KV | Paging for its own sake at batch 1 | Attention plan per context bucket; prefix cache for TTFT |
| **Cranelift vs LLVM vs C** | Cranelift for host JIT; optional C/clang for developers | LLVM dependency now | **Keep Cranelift**; revisit for SME or vectorized CPU kernels |
| **AI kernel generation** | LLMs as candidate proposers in a verified loop | Trusting unverified speedups or weak baselines | Verified kernel search (**EXPERIMENTAL**) |

---

## 14. The 10 most important lessons for Sky

1. **Decode at batch 1 is bytes, not FLOPs.** Qwen3-32B at 4.5 bits/weight reads 18 GB per token, a ceiling of 34 tok/s on a 614 GB/s M5 Max. A native Metal-4 runtime (BaseRT) *matched but did not beat* MLX on a dense 27B (18.1 vs 18.4 tok/s). Big decode wins need fewer bytes (format, KV) or fewer weight passes (speculation). **[VERIFIED + arithmetic]**
2. **The compiler's best measurable lever is TTFT, especially short prompts.** The evidence is BaseRT's 1.44× over MLX at pp128 versus 1.02× at pp2048. Small-M GEMM tiling, fusion and launch overhead are where generated, tuned code wins. **[VERIFIED + LIKELY]**
3. **Every competitive Apple-GPU engine emits MSL.** The only MLIR route to the M5 matrix units (Mojo) uses undocumented AIR and still trails MLX by 4–15%. IREE's Apple target has no MMA. Build Sky's own stack to MSL and drop "GPUs through MLIR" for Apple. **[VERIFIED]**
4. **Generic ML compilers lag hand-designed templates on Apple GPUs.** MLC-LLM, IREE, Inductor-MPS, Halide-Metal and jax-metal all show it. Sky needs ~10 excellent template families (quantized GEMV, NAX GEMM, decode and prefill attention, fused norm/RoPE/epilogues), generated and tuned, not generic loop nests. **[VERIFIED]**
5. **Owning the language is worth something concrete.**
   - No graph breaks (30% of TorchBench models break under Dynamo).
   - Types carry shapes, formats and layouts.
   - The result is whole-program AOT planning: a static memory plan, a static dispatch stream, and weight-layout co-design with the kernels.

   **[VERIFIED + LIKELY]**
6. **Numerics contracts are Sky's differentiator.** Extend `Reproducible/Relaxed/Fast` with *tolerance* contracts (KL, logits error) that gate quantization, scale folding, KV formats and split-K order. Record the choices so results are reproducible per plan. Nobody else makes these trade-offs explicit and checkable. **[EXPERIMENTAL]**
7. **Autotuning a fixed model is cheap.** About 60 kernel instances, a few hours, no learned cost model. The payoff is robustness across devices and OS updates, plus 5–30% on small-M prefill. Replace hard-coded heuristics of the MLX `sdpa` device-letter kind. **[EXPERIMENTAL estimate]**
8. **Use e-graphs for a small, known algebra.** QKV and gate/up concatenation, scale folding, transpose-to-layout, GQA packing, with DAG-aware extraction on one block template. Fusion is a separate cost-model partitioner. Production experience (Cranelift: about 0.1% from multi-versioning) and the superoptimizers' cost (Mirage: up to 4 h per pattern) say do not go broad. **[VERIFIED + LIKELY]**
9. **Megakernels pay when per-op work is tiny.** For 32B they are worth about 3–10% (MPK: 1.16× at 8B). On Metal, start with a static fused dispatch stream replayed through an ICB, as Monolith does. Build a persistent megakernel only if measured gaps justify it, and only after the co-residency and memory-ordering questions are answered. **[VERIFIED + LIKELY]**
10. **Never trust a kernel speedup without an oracle.** Sakana's 150× was a 3× slowdown, and KernelBench-style checks have been fooled into fake 50–120× results. Every generated or tuned kernel passes differential tests against Sky's reference semantics at the declared tolerance, with poisoned outputs and GPU-timestamp measurement. **[VERIFIED]**

---

## 15. Open questions

1. **What does MLX actually achieve on an M5 Max for Qwen3-32B 4-bit?** Decode tok/s as a fraction of 34.1, and prefill TFLOP/s at pp128, pp512 and pp4096. This sets the real headroom. *(Benchmark agents.)*
2. **Metal dispatch costs.** Per-dispatch gap between dependent dispatches: same encoder vs new encoder vs new command buffer; ICB replay cost; tail bubbles. *(Hardware agent.)*
3. **Metal synchronization guarantees.**
   - Co-residency and forward progress for persistent threadgroups.
   - MSL 3.2+ `atomic_thread_fence` scopes and orders on device memory.
   - Watchdog limits.

   *(Hardware agent.)*
4. **Is the documented MPP `matmul2d` path as fast as Mojo's AIR `simdgroup_matrix_16x16x16` intrinsic** for small-M GEMM on M5? If not, does the `__asm("air.*")` escape hatch need to be a first-class backend feature?
5. **Can Metal heaps or sparse buffers give vAttention-style "reserve virtually, commit on demand" KV allocation** on Apple GPUs?
6. **Which weight formats are worth it under a quality contract?** Group size 128 (≈4.25 bits/weight), mixed 3/4-bit, codebooks. What is the quality-vs-bytes frontier for Qwen3-32B? *(Quantization agent.)*
7. **How often must the tuning database be invalidated?** Per macOS or Metal-compiler update. Is a quick re-validation sweep (minutes) enough?
8. **Licence review before porting** any MFA, MLX, BaseRT or Mojo kernel code (MIT, Apache-2.0; all on Sky's allow-list). The modular repo shows `NOASSERTION` at the root, so check file-level headers.

---

## 16. Sources

**Papers**
- Ansor https://arxiv.org/abs/2006.06762
- MetaSchedule https://arxiv.org/abs/2205.13603
- TLP https://arxiv.org/abs/2211.03578
- Relax https://arxiv.org/abs/2311.02103
- MLX/MLC/llama.cpp comparison https://arxiv.org/abs/2511.05502
- BaseRT https://arxiv.org/abs/2607.19438
- Rigel https://arxiv.org/abs/2606.12765
- Mirage https://arxiv.org/abs/2405.05751
- MPK https://arxiv.org/abs/2512.22219
- AutoMegaKernel https://arxiv.org/abs/2606.09682
- Axon https://arxiv.org/abs/2606.26344
- Tensat https://arxiv.org/abs/2101.01332
- Tensor graph EqSat + MCTS https://arxiv.org/abs/2410.05534
- TASO https://theory.stanford.edu/~aiken/publications/papers/sosp19.pdf
- SmoothE https://github.com/cornell-zhang/SmoothE
- e-boost https://arxiv.org/abs/2508.13020
- Linear Layouts https://arxiv.org/abs/2505.23819
- TLX https://arxiv.org/abs/2605.10905
- ThunderKittens https://arxiv.org/abs/2410.20399
- TileLang https://arxiv.org/abs/2504.17577
- Exo 2 https://arxiv.org/abs/2411.07211
- Halide autoscheduler 2019 https://halide-lang.org/papers/halide_autoscheduler_2019.pdf
- Anderson 2021 https://arxiv.org/abs/2012.07145
- Mullapudi 2016 https://research.google/pubs/automatically-scheduling-halide-image-processing-pipelines/
- PyTorch 2 https://docs.pytorch.org/assets/pytorch2-2.pdf
- FlashAttention-3 https://arxiv.org/abs/2407.08608
- FlashDecoding++ https://arxiv.org/abs/2311.01282
- FlashInfer https://arxiv.org/abs/2501.01005
- vAttention https://arxiv.org/abs/2405.04437
- LeanAttention https://arxiv.org/abs/2405.10480
- SGLang https://arxiv.org/abs/2312.07104
- Cranelift/copy-and-patch https://arxiv.org/abs/2011.13127
- CGO'24 compile-time study https://home.cit.tum.de/~engelke/pubs/2403-cgo.pdf
- TPDE https://arxiv.org/abs/2505.22610
- Hello SME https://arxiv.org/abs/2409.18779
- SMEPilot https://arxiv.org/abs/2606.16332
- STOKE https://arxiv.org/abs/1211.0557
- Souper https://arxiv.org/abs/1711.04422
- AlphaDev https://www.nature.com/articles/s41586-023-06004-9
- KernelBench https://arxiv.org/abs/2502.10517
- Kevin https://arxiv.org/abs/2507.11948
- robust-kbench https://arxiv.org/abs/2509.14279
- Kernel Contracts https://arxiv.org/abs/2604.22032

**Official docs and blogs**
- Apple M5 Pro/Max newsroom https://www.apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/
- Apple ML: MLX on M5 https://machinelearning.apple.com/research/exploring-llms-mlx-m5
- Apple ML: Core ML Llama https://machinelearning.apple.com/research/core-ml-on-device-llama
- IREE Metal HAL https://iree.dev/developers/design-docs/metal-hal-driver/
- XLA GPU architecture https://openxla.org/xla/gpu_architecture
- XLA priority fusion RFC https://github.com/openxla/xla/discussions/6407
- Triton autotune https://triton-lang.org/main/python-api/generated/triton.autotune.html
- Gluon https://triton-lang.org/main/gluon/index.html
- PyTorch 2.13 https://pytorch.org/blog/pytorch-2-13-release-blog/
- torch.compile-MPS tracker https://github.com/pytorch/pytorch/issues/150121
- Helion https://pytorch.org/blog/helion/
- Flash-Decoding https://pytorch.org/blog/flash-decoding/
- Hazy megakernel https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles
- Hazy TP megakernel https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main
- ThunderMittens https://hazyresearch.stanford.edu/blog/2024-11-28-tk-mlx
- Cranelift aegraph https://cfallin.org/blog/2026/04/09/aegraph/
- Cranelift README https://github.com/bytecodealliance/wasmtime/blob/main/cranelift/README.md
- Mojo open source https://www.modular.com/blog/mojo-open-source
- Mojo Apple GPU forum https://forum.modular.com/t/apple-silicon-gpu-support-in-mojo/2295
- MetalRT https://www.runanywhere.ai/blog/metalrt-fastest-llm-decode-engine-apple-silicon
- BaseRT HF blog https://huggingface.co/blog/basecompute/basert-release
- Gimlet Labs https://gimletlabs.ai/blog/ai-generated-metal-kernels
- AlphaEvolve https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/
- Sakana walk-back https://techcrunch.com/2025/02/21/sakana-walks-back-claims-that-its-ai-can-dramatically-speed-up-model-training/
- ExecuTorch MPS (deprecated) https://docs.pytorch.org/executorch/stable/ios-mps.html
- jax-metal status https://github.com/jax-ml/jax/discussions/32135
- Xcode 26 Metal toolchain https://developer.apple.com/forums/thread/803174
- Qwen3-32B config https://huggingface.co/Qwen/Qwen3-32B/blob/main/config.json

**Source code (read in this session)**
- IREE: `compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/KnownTargets.cpp` @ `9acea6a`; `compiler/plugins/target/MetalSPIRV/*`.
- SPIRV-Cross PR #2645.
- triton-ext PRs #126 and #127.
- modular/modular: `max/mojo/max/gpu/compute/arch/mma_apple.mojo`, `max/kernels/src/linalg/matmul/gpu/apple/FP4_DESIGN.md`.
- Halide: `src/CodeGen_Metal_Dev.cpp`, `src/autoschedulers/`.
- llvm-project: `mlir/include/mlir/Dialect`.
- melior README.
- mpk-apple issue #1.
- Local clones: MFA `8671cdd`, metal-benchmarks `dc2adc6`, MLX `0840c42`, tinygrad `512e9b2`.
- `~/sky/crates/{sky-egraph,sky-tune}`, `~/sky/ROADMAP.md`.

**Other agents' measurements cross-referenced:** `$S/research/r3_out_dispatch_1.txt`, `$S/research/r3_out_bw_1.txt`.


---

## 17. Architect brief

**Files.**
- Notes: `$S/research/R7-compilers.md`.
- Parts: `$S/research/R7-parts/`.
- Arithmetic: `$S/research/R7-work/qwen3_32b_budget.py`.

Here `$S` = `/private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad`.

### Findings

1. **The physics.** Qwen3-32B at 4-bit (MLX group-64 affine, 4.5 bits/weight) reads **17.99 GB per decode token**. On an M5 Max at 614 GB/s that gives a ceiling of **34.1 tok/s**. Other fixed costs:
   - KV cache is 256 KiB per token in fp16, so 10.7 GB at 40k context.
   - KV traffic passes 10% of weight traffic at about 6.9k tokens.
   - Prefill costs about 256 TFLOP for a 4k-token prompt.

   [VERIFIED config and Apple spec, plus arithmetic]
2. **Prior art leaves little decode headroom over MLX on large dense models.**
   - BaseRT (hand-written Metal 4 runtime, July 2026), on a dense Qwen3.6-27B Q4 on an M5 Pro:
     - decode: **18.1 vs MLX 18.4 tok/s**;
     - prefill: **1.44× MLX at pp128 but 1.02× at pp2048**.
   - Monolith (mpk-apple, opened Sept 2026 by the MPK author) sets its decode target at only **≥1.10×** over the better of MLX and llama.cpp.

   [VERIFIED]
3. **Apple GPU codegen has converged on MSL text.** MLX, llama.cpp, tinygrad, PyTorch MPS, BaseRT, Halide, TVM/MLC, IREE (via SPIRV-Cross) and triton-ext all emit MSL. triton-ext *abandoned* an AIR path because AIR is undocumented. The M5 Neural Accelerators are reachable from MSL through Metal 4 TensorOps and Metal Performance Primitives, which is how MLX uses them. [VERIFIED]
4. **The MLIR route to Apple GPUs is weak.**
   - Upstream MLIR has no Apple or Metal dialect.
   - IREE's Apple target declares `mmaCount=0`.
   - SPIRV-Cross maps cooperative matrices only to 8×8 float `simdgroup_matrix`.
   - Mojo (open-sourced August 2026, Apache-2.0 + LLVM exception) reaches the M5's 16×16×16 MMA only through undocumented `llvm.air.*` intrinsics, and its W4A16 matmuls run at **0.85–0.96× MLX**.
   - melior needs a system LLVM/MLIR 23 and is pre-1.0.

   [VERIFIED]
5. **Megakernels pay off only when per-layer work is small.**
   - Hazy's Llama-1B megakernel reached 78% of bandwidth on H100, 1.5× SGLang.
   - MPK took Qwen3-8B on A100 from 14.5 to **12.5 ms/token (1.16×)**, against a ~10 ms bound.
   - At 32B on Apple, dispatch and bubble overhead is roughly 3–10%. Agent R3 measured about 2–3.7 µs per dependent MLX op on an M2 Pro.

   [VERIFIED + LIKELY]
6. **Superoptimization is real but expensive.**
   - Tensat: +16%, with 48× faster search than TASO.
   - Mirage: up to 3.3× at the kernel level, but up to 4 h of search per pattern, a LAX-only verifier, and NVIDIA only.
   - Cranelift's production e-graph: multi-version representation worth about 0.1%; average e-class size 1.13.

   [VERIFIED]
7. **Autotuning one model is cheap.** There are about 60 kernel instances, and an exhaustive search over a pruned space takes about 2–5 hours. No learned cost model is needed. [EXPERIMENTAL estimate]
8. **Owning the language removes graph breaks and enables ahead-of-time planning.** Dynamo captures 93% of TorchBench models, but only 70% of models have zero graph breaks, with a mean of 21 graphs per model. Sky gets static shapes and formats from types, which enables AOT planning. [VERIFIED]
9. **CPU codegen is not on the LLM critical path.** Cranelift's generated code runs at about the speed of *unoptimized* LLVM, and it compiles only 20–35% faster (CGO 2024). It is about 14% slower than LLVM on wasm. [VERIFIED]
10. **AI kernel generation helps only inside verified harnesses.**
    - AlphaEvolve: a kernel 23% faster, worth 1% of training time.
    - Kevin-32B: 1.10× over PyTorch eager.
    - Sakana's claimed 150× was really a **3× slowdown**, caused by an evaluation exploit.
    - Gimlet's Metal results are measured against eager MPS, a weak baseline.

    [VERIFIED]

### Negative results (tried; did not win)

- **Generic compilers on Apple GPUs:**
  - MLC-LLM sustains lower throughput than MLX.
  - IREE has no MMA.
  - Inductor-MPS is an "early prototype".
  - Halide's Metal backend has no `simdgroup_matrix`.
  - jax-metal (MPSGraph, closed source) is abandoned.
  - PyTorch migrated off MPSGraph because of its per-op compile cost.
  - ExecuTorch deprecated its MPS backend and added an MLX delegate.
- **Hand-written native runtimes:** BaseRT is roughly level with MLX on dense decode. MetalRT loses to both MLX and llama.cpp on Llama-3.2-3B.
- **Apple's own Core ML 8B showcase** reaches about 35% of memory bandwidth.
- **Undocumented-hardware traps:**
  - MFA's async copy helps on M1/M2 but hurts on M3 and later.
  - There are no native FP32 atomics.
  - AIR is undocumented.
  - The Metal 4.1 tensor path has eleven undocumented behaviours (Rigel).

### Recommendations

1. **Architecture: option A, Sky's own stack.** The pipeline:
   1. Sky IR, whose types carry shape, dtype, quant format, layout and numerics contract.
   2. E-graph algebra on *one block template*: QKV and gate/up concatenation, scale folding, transpose-to-layout, GQA packing, with DAG-aware extraction and a roofline cost.
   3. A cost-model fusion partitioner (XLA priority fusion style).
   4. A tile-level kernel IR (Triton/ThunderKittens style, with F₂ layouts).
   5. **MSL text**, using Metal 4 TensorOps for the Neural Accelerators and a test-gated `__asm("air.*")` escape hatch.
   6. An **AOT static plan**: a memory plan plus a pre-encoded dispatch stream or ICB per context bucket.
   7. Per-device tuning in `sky-tune`.
2. **Edit the ROADMAP.** Replace "GPUs through MLIR" and the melior dependency row for Apple. MLIR, Triton or Mojo become candidates only for later non-Apple backends.
3. **Build about 10 template families**, seeded from the MFA, MLX, ThunderMittens and BaseRT designs (MIT and Apache-2.0):
   - Q4 GEMV with fused norm prologue and residual/activation epilogues;
   - Neural Accelerator GEMM with small-M variants;
   - split-K GQA decode attention with fixed reduction order;
   - FA2-style prefill attention;
   - fused norm, RoPE and KV-append;
   - sampling.
4. **Make tolerance contracts first-class.** They gate weight and KV formats, scale folding and split-K order, and every choice is recorded in the plan so results stay reproducible. This is Sky's differentiating IP.
5. **Treat weights as constants.** Repack offline: QKV concatenation, interleaved gate/up rows, fragment-order nibbles.
6. **Tuning.**
   - Search: an analytic default plus exhaustive pruned search.
   - Cache key: template version, shape bucket, format, device fingerprint, and OS/Metal-compiler version.
   - Tunables that change rounding are allowed only under contracts.
   - Time with GPU timestamps.
7. **Megakernel.** Phase 1 is a static plan plus ICB replay, measured. Go further only if the measured gaps exceed about 5%.
8. **Verified kernel search.**
   - The reference interpreter is the oracle.
   - Use poisoned outputs and adversarial and shape-varied inputs.
   - LLMs act only as proposers.
9. **CPU codegen:** keep Cranelift. No LLVM now.
10. **Expected wins, in order:**
    1. TTFT and short-prompt prefill.
    2. Long-context decode (KV format and attention).
    3. RAM (static planning, commit-on-demand KV).
    4. Energy (no per-token CPU work).
    5. Dense short-context decode: only about 1.0–1.2× over MLX, unless the format changes or speculative decoding (outside this topic) reduces the weight passes per token.

### Open questions

1. What fraction of the 34.1 tok/s ceiling, and how many prefill TFLOP/s, does MLX actually reach on an M5 Max for Qwen3-32B? This sets the real headroom.
2. Metal dispatch costs: the gap between dependent dispatches, the cost of ICB replay, co-residency and forward-progress guarantees, and the semantics of MSL 3.2 device fences.
3. For small-M GEMMs, does the documented MPP `matmul2d` match the undocumented AIR 16×16×16 intrinsic?
4. Can Metal heaps or sparse buffers give vAttention-style commit-on-demand KV?
5. Where is the quality-vs-bytes frontier for weight and KV formats under a tolerance contract?
6. How often must the tuning database be invalidated across macOS and Metal updates?
