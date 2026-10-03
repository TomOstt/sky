# Part XXVII — Moat, IP and the acquisition package

This part asks three questions:
- What would Sky own that Apple could not rebuild cheaply?
- Which techniques are worth showing to patent counsel?
- What would an acquirer be shown?

The owner's stated ambition is that Sky becomes valuable enough that a company such as Apple would rather acquire it, at a scale the owner puts on the order of $100M, than rebuild it. This report gives no valuation, and our research found no sourced price for a comparable transaction. What it can assess is the technical question underneath: how long Apple would take to reproduce each thing Sky would have, and what evidence would make the conversation credible.

Labels follow the report's convention: `[VERIFIED: source]`, `[LIKELY: reason]`, `[EXPERIMENTAL: what decides]`, `[MOONSHOT]`. Research notes are cited as (R3 §7) and live in [`research/`](research/). §4 and §5.6 contain questions for counsel, never legal conclusions.

## 1. Summary

1. **Apple already owns every layer Sky builds on, and most of the layers Sky would build.**
   - MLX with its neural-accelerator path (NAX).
   - Core AI (OS 27): an ahead-of-time compiling, device-specializing inference framework with in-place KV "states" and custom Metal kernels.
   - Metal 4 and Metal Performance Primitives, whose lowering only Apple can see.
   - The neural accelerators, pre-release hardware, and engineers who already upstream M5 kernels to llama.cpp.

   `[VERIFIED: R3-issues §B, R6 §A2–A3]`
2. **What Apple's stack lacks today are choices, not capabilities.** It has no byte-identical speculation: mlx-lm #1470 was answered with documentation. It has no cheap verification of 5–32 tokens, no EAGLE or MTP drafting in mlx-lm, no zero-copy loading, no CPU-free token loop, and no accounting of every byte. MLX's maintainers have declined several of these on the record (§2.2).
3. **No single candidate moat survives more than months against Apple.**
   - The longest-lived are exact speculation on batch-invariant kernels and the numerics contracts, at about 3–6 months each. They take that long because they cut across every kernel, and they count only if Sky ships them first, with proof.
   - Tuning data, portability and zero-copy loading take Apple days to weeks.
   - The language is Sky's most original asset and the one least likely to interest an inference buyer.
4. **On the headline chip, the kernel-level speculative advantage over MLX is smaller than the M2 Pro data suggest.** MLX's own contributor data on an M5 Max imply a per-draft verify cost of β ≈ 0.06–0.18, not the β ≈ 0.5–1 measured on M2-class GPUs. With the public EAGLE-3 drafter, the model gives Sky's verify kernel about 1.2× over an MLX-class one (§3.2). The rest of any speculative lead must come from drafting, acceptance and attention. `[EXPERIMENTAL arithmetic on contributor data]`
5. **Code volume is not a barrier in 2026.** Sky's public history shows about 75k production lines of Rust written over three calendar days by one person directing AI agents (§3.9). The scarce assets are:
   - validated designs and negative results;
   - measured data and test oracles;
   - a reproducible public lead;
   - people and users.
6. **Seven mechanisms are candidates for patent counsel (§4).** Every one has close published prior art, which we name.
7. **The acquisition package (§5)** consists of:
   - a one-page summary in which every number is `TBD (measured at G8)`;
   - a live demo;
   - the `sky-evaluation` repository;
   - the "why faster" report.

   IP hygiene is clean today: Apache-2.0, a `cargo deny` allow list, no ported code. One material diligence question remains: AI-assisted authorship.
8. **Today Sky has no inference code, a private repository with no users, and one author.** An acquisition conversation needs the owner's Milestones 5–7, a team and adoption. Until those exist, open-source adoption, upstream partnership and other buyers are the likelier outcomes (§5.8).

## 2. What Apple already has

### 2.1 Inventory

| Asset | What it does for this workload | Evidence | What it means for Sky |
|---|---|---|---|
| **MLX and mlx-lm** | Apple's open-source (MIT) research and local-inference framework. Its decode step runs at 94–97% of MLX's achievable bandwidth on the M2 Pro (R3 §8.1). It decodes Qwen3-32B at 28.4 tok/s on an M5 Max (R5 §9.3). NAX handles prefill GEMM and attention since v0.30.0 (2025-11-19). | `[VERIFIED: R3, R5; LIKELY for the single community M5 Max run]` | The baseline sits near the roofline and is maintained by Apple. |
| **MLX's speed on new chips** | M5 Pro/Max tunings shipped in v0.31.1 (2026-03-12), nine days after the chips were announced (2026-03-03). "M5 ultra tunings" shipped in v0.32.3 (2026-09-29), seven days after the M5 Ultra shipped (2026-09-22). | `[VERIFIED: R3-issues §B3 release map; R6 §A1]` | Apple has tuned kernels ready at launch, which Sky cannot do without the hardware. |
| **Core AI** (WWDC26, OS 27) | "The inference framework powering on-device Apple Intelligence … fast inference across the CPU, GPU, and Neural Engine". It offers PyTorch conversion (`coreai-torch`), ahead-of-time compilation, per-device "specialization" with a model cache, "states" for in-place KV caches, custom Metal 4 kernels (`TorchMetalKernel`), Instruments support and a debugger. | `[VERIFIED: R6 §A3, WWDC26-102/324/325/326]` | The same product category as Sky Inference, built by Apple. Nobody we found has measured its LLM decode or prefill speed (R6, open questions). |
| **Core ML** | Apple's earlier framework. Its 2024 Llama-3.1-8B showcase decoded at about 35% of memory bandwidth. | `[VERIFIED numbers; LIKELY percentage: R7 §8.1]` | Historically not the frontier for LLM decode. |
| **Metal 4, MPP, MPS** | TensorOps are the only route to the neural accelerators. MPP delegates each matmul to Apple-internal `air.externally_defined` functions, so "we cannot see or control how MPP is lowered". MPP fp16 GEMM reaches 95% of peak on the M2 Pro. | `[VERIFIED: R6 §A2, B5]` | Every competitive Sky kernel on an M5 runs through code only Apple can tune. |
| **The neural accelerators** | One per GPU core. On an M5 Max, about 65 TFLOPS fp16 and 122 TOPS int8. Apple calls them a prefill feature and advises moving `simdgroup_matrix` code to TensorOps. | `[VERIFIED for design and quotes; LIKELY for the rates: R6 §A2]` | Apple designs the hardware and knows its behavior. Outsiders have catalogued eleven undocumented behaviors of the tensor path (Rigel, R7 §5.5). |
| **Compilers** | Apple ships the Metal shader compiler that turns Sky's MSL into GPU code, the MPP backend, and Core AI's ahead-of-time compiler. | `[VERIFIED: R6]` (team sizes are not in our evidence) | Sky's value can only lie above Apple's compilers, never below them. |
| **Apple ML research on this workload** | ReDrafter ("up to 2.3×" in MLX on Apple silicon), Mirror-SD (drafting on the Neural Engine), MTP fine-tuning with gated LoRA, KV Prediction. The Foundation Models use 2-bit QAT and 8-bit KV on device, and 3.56-bpw ASTC weights on the server. | `[VERIFIED: R8 §2.6, §4.2]` | Apple has in-house expertise in speculative decoding and weight compression. |
| **Engineers working upstream** | Apple's Developer-Ecosystem-Engineering account wrote llama.cpp's Metal tensor-API kernel (#20962, merged 2026-04-25): +26% geomean prompt processing on an M5 Max. | `[VERIFIED: R3-issues §B12, R5 §9.5]` | Apple improves the open runtimes Sky competes with. |
| **The MLX contributor community** | `qmv_wide` (mlx#3764), the M5 Max verify-cost data (mlx#3791), GQA decode attention (mlx#4077), a measured 530 GB/s on M5 Max (mlx#3837). | `[VERIFIED: R3-issues §C–F]` | Sky competes with a crowd, not only with Apple's staff. |
| **Hardware** | Every chip, before launch. | `[LIKELY: tunings shipped within days of each launch]` | Sky has one M2 Pro with 16 GB (SPINE D14). |

### 2.2 What Apple's stack does not do today, and what its maintainers said

| Not done today | Evidence | Stated position |
|---|---|---|
| Speculative output identical to plain output | mlx-lm#1470: on an M5 Max, Qwen3-32B with a 0.6B draft diverges from plain greedy at token 91. The cause is a bf16 logit tie under a different reduction order (R8 §3.4). | PR #1592 documents it: "lossless up to floating-point associativity, not bit-exact". No code fix. |
| Cheap verification beyond 3 tokens | M5 Max, whole 70B forward: +12% at 3 tokens, +48% at 5, +125% at 8 (mlx#3791). `qmv` for M < 13 never uses the neural accelerators. With GQA 8, 5–8 queries fall back to unfused attention, and 9–48 hit a flat plateau (R3-kernels §d). | #3826, closed: "likely not going to happen in production". #4198: "only focusing on real cases happened during inference". |
| EAGLE-3 or MTP drafting in mlx-lm | mlx-lm drops Qwen3.5 MTP tensors. The community PRs #1468/#1469 were closed (R8 §2.3). | #1497/#1498: "At the moment we don't accept AI written architecture changes". #1652: "the number of PRs is way beyond our capacity to review" (R4 §9). |
| Zero-copy weight loading | MLX copies weights with `pread` (R3 §4). | mlx#4249: "At the moment we are not considering using mmap". |
| Megakernels and whole-step compilation | — | mlx#3313: "won't fix". mlx#4296: "you would end up launching exact same graph". |
| New fused operations | — | mlx#4259: the kernel must first be deployed "in a popular inference engine to prove it is definitely helping". |
| A token loop with no CPU work | mlx-lm's Python loop drew about 7 W of CPU during 0.6B decode (R9 §6.3). | — |
| A byte-level account of each token | None. Maintainers point users to Metal GPU captures (R3-kernels §d). | — |

### 2.3 Reading

The gaps reflect the priorities of a general-purpose framework, not limits on ability [LIKELY: each item has a known fix, and MLX ships comparable kernel work in weeks; see §3.2]. Apple could close any of them if it chose to.

The same policy record also shows a partnership path. MLX's maintainers ask that a new kernel be proven in a popular engine before it becomes a builtin, and Sky could be that engine (§5.8).

## 3. The candidate moats

Each candidate below is assessed on five points:
- what it is, and its status in Sky today;
- who else has it;
- what it is worth;
- how long Apple would need to reproduce it, and why;
- what would make it stronger.

The time estimates are judgments, labeled LIKELY, and anchored to how fast comparable work actually shipped.

### 3.1 Search-based block compilation and measured tuning data across chips

- **What it is.** Three levels of autotuning (SPINE D11; [Part X](part10-autotuner.md)):
  - BEAM search over kernel-skeleton parameters;
  - a search over block fusion partitions, where one tuned transformer block is reused for every layer;
  - model-level choices.

  A per-chip cost model drives it. The tuning database is keyed by chip, OS, Metal compiler, template version, shape and format.
- **Status.** `sky-tune` exists for CPU kernels. It is a data-only database keyed by fingerprint × kernel × size class, and its rule is that "tuning may change speed, never results" (R1 §5.7). For GPU tensor programs that rule carries over only to `portable` mode; under `inference` a tuning entry may change the last bits (SPINE D6). The GPU search, the cost model and the data do not exist yet.
- **Who else has it.**
  - llama.cpp tunes its decode attention offline per GPU family. The community gathered 3,884 rows across 19 SKUs in about two weeks, reduced to 1,033 rows across 4 families (R5 §9.5).
  - MLX picks attention block counts by hand-coded thresholds keyed by device-class letter (R7 §8.2).
  - tinygrad has BEAM search (R2 §6.4).
  - TVM/Ansor, Triton and XLA are the published prior art (R7 §10).
- **What it is worth.** For Qwen3-32B, R7 estimates about 60 kernel instances and 2–5 hours of tuning per chip. The expected gains are:
  - 0–5% on bandwidth-bound GEMVs;
  - 5–30% on small-M prefill GEMMs and attention chunking;
  - mainly, robustness across chips and OS updates (R7 §10.2).

  `[EXPERIMENTAL]`
- **Apple's time to reproduce: 1–3 months for the infrastructure, days per chip for the data.** `[LIKELY: the methods are published; MLX already keeps per-device tables and shipped M5 Ultra tunings within a week of launch; Apple has pre-release silicon]` Tuning data is no asset against Apple. Apple can regenerate any table Sky has, on chips Sky cannot buy yet.
- **What would strengthen it.** Evidence that the search finds configurations hand-tuning missed on shapes that matter (verify widths 2–32, short prompts), and that it re-tunes after an OS update with no regression.
- **Verdict: weak against Apple; moderate against third parties without hardware.**

### 3.2 Batch-invariant verify kernels plus exact speculation

- **What it is.**
  - One GEMV/GEMM family in which the decode instance (M = 1) and the verify instances (M = 2…32) share one reduction order.
  - Fixed-split attention.
  - Keyed Gumbel sampling.

  Together these make speculative output byte-identical to Sky's plain output at any temperature. The target is that verifying up to about 8 tokens costs ≤ 1.3× one token (SPINE D6, D10, §3; [Part XXI](part21-speculative.md); [Part XXIV](part24-roadmap-now.md) #3 and #10).
- **Status.** Not built. A 60-line test kernel reached ×1.30 at M = 3 on the M2 Pro, but spilled registers from M = 4 (R3 §7.G).
- **Who else has it.** The ideas are published:
  - batch-invariant kernels (Thinking Machines, 2025-09);
  - seeded-hash Gumbel sampling in SGLang's deterministic mode (2025-09-22);
  - drafter-invariant speculation (Daliri, Musco & Suresh, arXiv 2408.07978);
  - verify-and-replay determinism (LLM-42, arXiv 2601.17768).

  All of these are on NVIDIA (R8 §3.2–3.5). We found nobody shipping them on Apple GPUs. Competitors are, however, working on the same regime:
  - **MLX on an M5 Max** already verifies 3 tokens for +12% (β ≈ 0.06 per drafted token), 5 for +48% (β ≈ 0.12) and 8 for +125% (β ≈ 0.18). Its speculative mode reached 24.5 tok/s on a 70B with a 0.6B draft. The same PR measures a plain forward at 75.9 ms (≈ 13.2 tok/s). `[VERIFIED statements, contributor data: mlx#3791 via R3-issues §E4; β and the ≈ 1.86× ratio are our derivation]`
  - **llama.cpp** has a Q = 2 verify attention kernel measured at 1.75× on an M5 Max (#23114). An open PR claims speculative decoding rising from 1.84× to 2.27× on an M4 Max (#27441, UNVERIFIED) (R5 §9.4–9.5).
- **What it is worth.**
  - **Speed: about 1.2× from the kernel alone.** R8's model is S = τ / (1 + γc + γβ). With the published EAGLE-3 acceptance (τ = 2.29 at 3 drafts and T = 0.6) and c = 0.03, it gives:
    - ≈ 1.65× for an MLX-class verify (β ≈ 0.10, interpolated from mlx#3791);
    - ≈ 1.99× at Sky's target β = 0.02.

    That is a kernel advantage of about 1.2× on the M5 Max. `[EXPERIMENTAL arithmetic]` Anything beyond 1.2× must come from:
    - drafting: EAGLE-3 support, which mlx-lm lacks, and a cheaper draft step;
    - acceptance: rejection sampling, Σ min(p, q) instead of Σ p·q;
    - attention for 5–16 queries;
    - longer drafts, where MLX's cost rises.
  - **Exactness is worth more as a property than as speed.** It makes speculative decoding testable byte for byte, and mlx-lm cannot claim it today.
- **Apple's time to reproduce: 2–6 months.** `[LIKELY]` The reasons:
  - a multi-row decode kernel (`qmv_wide`, adapted from llama.cpp) went from a contributor PR to a release within weeks;
  - the batch-invariance recipe is public;
  - the real cost is applying one reduction order to every kernel (GEMV, GEMM, attention, norms, sampling), against MLX's stated view that bit-exactness is not a goal.
- **What would strengthen it.**
  - Being first, with proof. Publish the #1470 script as an adversarial test that Sky passes and others fail.
  - Make "speculative == plain" a CI invariant on every chip.
  - Publish verify-cost curves per chip.
  - Extend exact rollback to the hybrid Gated-DeltaNet models (R8 §3.5), which nobody ships.
- **Verdict: the strongest technical lead available, measured in months. It lasts only as "first, with proof", or as a test standard others adopt.**

### 3.3 Determinism and reproducibility contracts

- **What it is.** Named numerics contracts, recorded per kernel in the execution plan and shown by `sky explain` (SPINE D6; [Part VI](part06-architecture.md) §8.8):
  - `reproducible` (existing);
  - `inference`: bit-reproducible per chip, OS, build and tuning entry, and batch- and chunk-invariant (chunk invariance is waived only for hybrid Gated-DeltaNet layers in a labeled mode, VII-M7);
  - `portable` (proposed);
  - `relaxed`.
- **Status.** Exists for scientific Sky: f64, no FMA, the same bits on every machine (R1 §6.1). Not yet for tensors.
- **Who else has it.**
  - SGLang and vLLM ship deterministic modes on NVIDIA (R8 §3.3).
  - Kernel Contracts (arXiv 2604.22032) is a specification language for kernel correctness (R7 §12.1).
  - MLX has no contracts, and enables TF32 by default on the neural accelerators (R3-kernels §d).
  - R7 calls tolerance contracts "Sky's differentiating IP" (R7 §17).
- **Apple's time to reproduce: 3–6 months for a deterministic mode in MLX or Core AI; longer as a language-level contract.** `[LIKELY: the same cross-cutting cost as §3.2; contracts also need a compiler that records each decision, which MLX's lazy graph does not do]`
- **What would strengthen it.** Users who need reproducibility, for example evaluation, audited deployments or research. Also a `portable` mode with the same bits across chip generations, if its cost is acceptable `[EXPERIMENTAL]`.
- **Verdict: a product differentiator consistent with Sky's scientific roots; moderate durability; close prior art.**

### 3.4 Explainability, every byte accounted for, and cost contracts

- **What it is.** `sky explain` prints, for each decode token (SPINE D12):
  - every kernel, with predicted and measured time, bytes and FLOPs;
  - the memory plan, byte by byte;
  - every fusion made or refused, with a rule ID.

  Compile-time contracts such as `cost bytes_per_token <= 18.2 GB` are checked by `sky check`.
- **Status.** Sky already records rule-ID explanations from every pass. It already has cost contracts over symbolic sizes (`cost flops <= 2*N^3`), which `sky check` proves, refutes with a witness, or warns about (R1 §5.8). The tensor version does not exist.
- **Who else has it.**
  - Apple has Instruments, GPU captures, and Core AI's Instruments support and debugger (R6 §A3).
  - MLX has nothing comparable.
  - Apple GPUs expose no per-dispatch timestamps, only stage boundaries (R6), so "measured per kernel" needs care ([Part XXIII](part23-benchmarks.md) §6.1).
- **Apple's time to reproduce: 3–6 months for a comparable report in Core AI's tooling.** `[LIKELY: Apple owns the profilers; the "predicted" half needs a cost model]`
- **Verdict: a strong developer feature and the source of the "why faster" evidence (§5.5). Not durable.**

### 3.5 The language

- **What it is.** Models written as typed Sky (SPINE D7, D8):
  - storage formats in the types: `x: Tensor[T, 5120] in f16`, `W: Matrix[25600, 5120] in q4(group: 64)`;
  - broadcasting only by matching size names;
  - reductions over named axes.

  Shapes and formats are checked before the program runs. A test requires the Sky-source model and the Rust builder to produce identical TIR.
- **Status.** Sky today is f64-only and rank ≤ 2. It has no integers and no reductions on `main`, and it forbids broadcasting. R1 counts 12 missing features before a Qwen3 layer can be written in Sky (R1 §9.3b). The Rust builder comes first; Sky-source models come after G4.
- **Is "about 200 lines" an advantage? Not by length.** mlx-lm's `models/qwen3.py` is 234 lines, and llama.cpp's `src/models/qwen3.cpp` is 159 lines. `[VERIFIED: wc -l at mlx-lm 53b9af3 and llama.cpp 81e39ad]` The difference would be that shapes, storage formats and broadcasting are checked before the program runs, and that the source feeds a whole-program planner. `[EXPERIMENTAL: no Sky model exists]`
- **Apple's position.** Apple owns Swift and gives Core AI a Swift API. Its model authors convert from PyTorch (R6 §A3). We found no evidence that Apple wants a new modelling language. `[LIKELY]`
- **Apple's time to reproduce: not applicable.** Apple would either want the language or ignore it. A language takes years to gain users, which is a barrier only if it gains them.
- **Verdict: the most original asset, and the least relevant to an inference buyer** ([Part XXVIII](part28-risks.md), R-22). The inference engine must stand on its own without the language.

### 3.6 Performance portability across M1–M5 and the M5 Ultra

- **What it is.**
  - One compiler with per-chip tuning.
  - MPP `matmul2d` as a portable GEMM path: 95% of peak on the M2 Pro (R6 §B5).
  - Verify widths sized per chip: about 8 tokens on shader ALUs, and 33–35 in theory on the M5 neural accelerators if `matmul2d` is efficient at those widths ([Part IV](part04-hardware-limits.md) §8), realistically 16–24 ([Part XXI](part21-speculative.md) §3.3).
- **Who else has it.** MLX has per-device tunings, including the M5 Ultra. llama.cpp keeps per-family tables. Apple designed TensorOps to be portable from M1 to M5 (R6 §A2).
- **Apple's time to reproduce: already done for MLX; days per new chip.** `[VERIFIED: tunings shipped 7–9 days after launches]`
- **Verdict: not a moat against Apple.** Against others it is a moat of hardware access, which Sky does not have ([Part XXVIII](part28-risks.md), R-19).

### 3.7 Zero-copy aligned relayout loading

- **What it is.** mlx-community safetensors files are unaligned: data starts at offset ≡ 6 (mod 8), and 0 of 907 tensors are even 4-byte aligned. On the M2 Pro, an unaligned `uint` load silently returns the rounded-down word (R9 §0). Sky therefore (SPINE D9):
  - builds a page-aligned, lossless relayout cache keyed by the source sha256;
  - proves that the dequantized values are bit-equal;
  - maps the cache into no-copy buffers.

  GGUF files are mapped directly.
- **Who else has it.** llama.cpp maps GGUF with no copy (R5 §1.5). MLX copies its weights. A contributor's mmap prototype for MLX ran "at resident speed", and MLX declined it (mlx#4249; R3-issues §G3).
- **What it is worth.** Load time and cold TTFT. Not steady-state RAM, because Metal wires the buffers on use (R3 §8.3). The ≥ 3× cold-TTFT target is EXPERIMENTAL (SPINE §3).
- **Apple's time to reproduce: weeks.** `[LIKELY: the design is known; MLX's obstacle is policy]`
- **Verdict: a visible user win; not a moat.**

### 3.8 Summary

| Candidate | New relative to prior art | Apple's time to reproduce `[LIKELY]` | Durable against Apple? | What would strengthen it |
|---|---|---|---|---|
| Block-compilation search and tuning data | Low | 1–3 months (infrastructure); days per chip (data) | No | proof the search beats hand-tuning on verify and short-prompt shapes |
| Batch-invariant verify kernels and exact speculation | Medium: first on Apple; ideas published for NVIDIA | 2–6 months | Months | being first with proof; public adversarial tests; exact rollback for hybrid models |
| Determinism contracts | Medium | 3–6 months for a mode; longer as a language contract | Months | users who need reproducibility |
| Explainability and cost contracts | Medium (bytes-per-token contracts at compile time) | 3–6 months | No | the "why faster" report |
| The language | High as originality, unproven as value | not applicable | not applicable | adoption by model authors |
| Portability from M1 to the M5 Ultra | Low | already done | No | — |
| Zero-copy relayout loading | Low | weeks | No | — |

### 3.9 What is actually hard to copy

- **The combination, shipped first and proven.** Each piece takes Apple months. All of them together, verified against an oracle on four chips and reproduced by an outsider, is a lead of perhaps a year, but only while Apple's priorities stay elsewhere. `[LIKELY: the sum of the parts' effort]`
- **Evidence and negative results.** This report, R1–R10 and `sky-evaluation` record what does not work on Apple GPUs:
  - persistent megakernels;
  - command replay as a decode win for dense 32B models;
  - 2-bit codebook formats on Apple's ALU budget;
  - plain decode beyond about 1.05–1.14× of MLX.

  That saves an acquirer months of measurement.
- **Test oracles.** The TIR reference interpreter, the batch-invariance and exactness suites, the adversarial #1470 test, and the correctness gates of R9 §7.3.
- **Trained artefacts.** A DFlash-class drafter for Qwen3-32B (gate G9) needs GPU training time and data, and nobody has published one (R8 §2.9). It is the one asset whose cost is not just engineering time. `[EXPERIMENTAL: needs training compute the project does not have]`
- **People and adoption** (§5.8).

**Code volume is not a barrier.** The public history of the Sky repository is:
- 50 commits between 2026-09-29 and 2026-10-01;
- one human author;
- AI co-author trailers throughout the commit messages.

`[VERIFIED: git log, 2026-10-02]` In that time the repository reached about 75k production lines of Rust (R1 §2.1). Whatever that says about Sky's productivity, it says the same about any rebuilder with the same tools. Line count and implementation effort will not deter an acquirer. `[LIKELY]`

## 4. Candidates for patent counsel

Patent consultation is the owner's call (SPINE D15). The table lists mechanisms that are technically new relative to the prior art our research found, together with the prior art counsel should see. It is a technical list. It contains no legal conclusion about patentability, infringement or anything else.

| # | Mechanism (where in the report) | What may be technically new | Prior art for counsel (from R1–R10) |
|---|---|---|---|
| P1 | A **reduction-schedule invariant in the IR**: the decode (M = 1) and verify (M = k) instances of each GEMV, GEMM and attention kernel must carry the same reduction-schedule attribute, enforced by the verifier. With keyed Gumbel sampling, speculative output becomes byte-identical to plain output (SPINE D6, D10; [Part XXI](part21-speculative.md)) | Enforcement at compile time in an IR, on GPUs whose matrix-unit accumulation order is opaque; works together with tree-verification masks | Thinking Machines batch-invariant kernels (2025-09); SGLang deterministic mode (2025-09-22); Daliri et al. arXiv 2408.07978; LLM-42 arXiv 2601.17768; Leviathan et al. arXiv 2211.17192 |
| P2 | **Exact speculative rollback for hybrid recurrent layers** (Gated DeltaNet): the recurrent update runs per position inside the verify pass, with a state checkpoint per draft position (R8 §3.5) | Bit-exact verification for linear-attention layers | MTPLX's claimed "exact rejection sampling with residual correction" (UNVERIFIED); SGLang and vLLM MTP support for Qwen3-Next (R8 §2.7). The literature on linear-attention speculation was not searched. |
| P3 | **Speculation sized from measured hardware data**: verify width and draft tree chosen at compile time from the measured ridge point per chip (k\* = R·b/16), separately for ALU and neural-accelerator paths (R8 §2.2) | A closed-form sizing rule tied to measured device data inside a compiler | Sequoia arXiv 2402.12374 (hardware-aware trees); SpecInfer; EAGLE-2 dynamic trees (R8 §2.4, §2.8) |
| P4 | A **proof-carrying aligned relayout cache**: page-aligned relayout of unaligned checkpoints, keyed by source sha256, with a bit-equality test of dequantized values, mapped with no copy into GPU buffers ([Part XII](part12-quantization.md), [Part XIX](part19-memory.md)) | The verified cache as a load path | GGUF's 32-byte alignment; llama.cpp's mmap + `bytesNoCopy`; MLX discussion #615 ("convert safetensor files to match"); the mlx#4249 prototype |
| P5 | **Compile-time contracts on memory traffic** (`cost bytes_per_token <= …`), proved over symbolic sizes for a whole model (SPINE D12) | Contract checking of bytes per generated token | Sky's existing flops contracts (R1 §5.8); Kernel Contracts arXiv 2604.22032 |
| P6 | **One tuned block reused for every layer**: fusion-partition search on one transformer block, applied to all layers, keyed by chip, OS, Metal compiler and template version (SPINE D11; [Part X](part10-autotuner.md)) | The keying and the per-layer reuse | TVM/Ansor/MetaSchedule, XLA priority fusion, tinygrad BEAM, llama.cpp's per-family tables (R7 §10, R5 §9.5) |
| P7 | A **GPU-resident decode loop on a GPU that cannot submit work**: pre-committed per-token command buffers; the GPU writes the token, position and a stop flag that turns later steps into no-ops; GPU sampling; shared-event signalling ([Part XX](part20-persistent-execution.md)) | The stop-flag scheme on Metal | Monolith/mpk-apple (ICB replay, 2026-09); tinygrad's HCQ2 ICB replay; MPK; CUDA Graphs; Hazy's megakernels (R7 §7) |

Technical observations for counsel's briefing:
- All seven mechanisms have close published prior art, mostly on NVIDIA. The Apple-specific parts are largely implementation. The most distinctive item is probably the combination of P1, P3 and P7 in one compiler. `[LIKELY]`
- Publishing this report, a public repository and the G8 paper will disclose all seven.
- Sky's license, Apache-2.0, contains in §3 a patent license grant from each contributor for their contributions to the Work. `[VERIFIED: ~/sky/LICENSE]`

Questions for counsel:
1. Is any of P1–P7 worth a filing given the prior art listed, or is defensive publication preferable?
2. How does publishing this report, the repository or a paper before any filing affect the options? By when must the owner decide, relative to G8?
3. How does Apache-2.0's patent license interact with filings on techniques implemented in released code?
4. Freedom to operate: could Sky's plan touch patents, Apple's or others', on speculative decoding, batch-invariant kernels, quantization formats or GPU command replay? We have not searched.

The authorship questions are in §5.6.

## 5. The acquisition package

### 5.1 Contents

1. A one-page summary (§5.2), with the box explaining the owner's original table (§5.3).
2. A live demo on the benchmark host (§5.4; [Part XXIII](part23-benchmarks.md)).
3. `sky-evaluation`, the public benchmark repository: pinned competitor versions, the protocol, raw JSON, and an outside reproduction (§5.5).
4. The "why faster" report: a `sky explain` decomposition of each headline number (§5.5).
5. This report, with R1–R10 as appendices.
6. The IP file: license, notices, authorship records, and counsel's answers (§5.6).
7. Team and traction facts (§5.8).

### 5.2 One-page summary template

The page is generated from the `sky-evaluation` results JSON (task XXVII-T3), never typed by hand: a number missing from the JSON cannot appear on the page. Until gate G8, every cell is `TBD (measured at G8)`.

---

*Template begins.*

**Sky Inference: one-page summary**

**What it is.** A compiler and runtime that turns a Hugging Face checkpoint (MLX safetensors or GGUF) into a hardware-specialized execution plan for Apple GPUs, from M1 to M5 Ultra, and runs it with no per-token CPU work. Rust, Apache-2.0. `sky run hf://org/model "prompt"`.

**What is different.**
1. Exact speculative decoding: output byte-identical to plain decoding at any temperature.
2. Every decode byte explained by `sky explain`.
3. One compiler from M1 to M5 Ultra, with measured per-chip tuning.

**Headline: Qwen3-32B.** File: MLX 4-bit group 64, built by us from `Qwen/Qwen3-32B@TBD`, sha256 TBD. Machine: M5 Max, 40-core GPU, TBD GB, macOS TBD. Runs: 10-minute sustained, AC power, High Power Mode, the protocol of [Part XXIII](part23-benchmarks.md).

| Metric | mlx-lm (version TBD) | Sky (version TBD) | Sky ÷ mlx-lm (95% CI) | Condition |
|---|---|---|---|---|
| Decode, plain, 512 context | TBD (measured at G8) | TBD (measured at G8) | TBD (measured at G8) | greedy; fp16 KV |
| Decode, plain, 32K context | TBD (measured at G8) | TBD (measured at G8) | TBD (measured at G8) | fp16 KV |
| Decode, exact speculation, code/math suite | TBD (measured at G8) | TBD (measured at G8) | TBD (measured at G8) | T = 0.6; drafter TBD |
| Decode, exact speculation, chat suite | TBD (measured at G8) | TBD (measured at G8) | TBD (measured at G8) | T = 0.6 |
| Decode, exact speculation, creative suite | TBD (measured at G8) | TBD (measured at G8) | TBD (measured at G8) | T = 0.6 |
| Speculative vs mlx-lm's own speculative mode, same draft model | TBD (measured at G8) | TBD (measured at G8) | TBD (measured at G8) | same prompts |
| Prefill, 2,048-token prompt | TBD (measured at G8) | TBD (measured at G8) | TBD (measured at G8) | — |
| TTFT, cold (process start → first token) | TBD (measured at G8) | TBD (measured at G8) | TBD (measured at G8) | weights in page cache |
| TTFT, repeated prefix | TBD (measured at G8) | TBD (measured at G8) | TBD (measured at G8) | prefix caches on both sides |
| Peak RAM at 32K | TBD (measured at G8) | TBD (measured at G8) | TBD (measured at G8) | system-wide metric (Part XXIII §6.3) |
| Energy per token | TBD (measured at G8) | TBD (measured at G8) | TBD (measured at G8) | whole system, ≥ 60 s loops |
| Same rows: Qwen3.6/3.8-27B hybrid with MTP | TBD (measured at G8) | TBD (measured at G8) | TBD (measured at G8) | second headline (SPINE D10) |
| Same rows: GGUF file vs llama.cpp (build TBD) | TBD (measured at G8) | TBD (measured at G8) | TBD (measured at G8) | same-file pair |

**Correctness gates.** These must pass before the table above is shown.

| Gate | Result |
|---|---|
| Tokenizer ids identical on the test corpus | TBD (measured at G8) |
| KLD(Sky ‖ mlx-lm) ÷ KLD(mlx-lm ‖ BF16), same file (≤ 0.1 required) | TBD (measured at G8) |
| Teacher-forced top-1 agreement (≥ 99% required) | TBD (measured at G8) |
| Speculative output = plain output, byte for byte (prompts × temperatures) | TBD (measured at G8) |
| Bits per weight / bytes per decode token | TBD (measured at G8) |

**Across chips.**

| Chip | Plain decode ÷ mlx-lm | Speculative ÷ mlx-lm | Prefill ÷ mlx-lm |
|---|---|---|---|
| M2 Pro | TBD (measured at G8) | TBD (measured at G8) | TBD (measured at G8) |
| M4 Max | TBD (measured at G8) | TBD (measured at G8) | TBD (measured at G8) |
| M5 Max | TBD (measured at G8) | TBD (measured at G8) | TBD (measured at G8) |
| M5 Ultra | TBD (measured at G8) | TBD (measured at G8) | TBD (measured at G8) |

**Evidence.**
- `sky-evaluation` commit: TBD.
- Reproduced by: TBD, within TBD%.
- "Why faster" report: TBD.
- Paper: TBD.

**Models supported:** TBD (the G7 list). **Adoption:** TBD, from public sources only (users, outside contributors, integrations). **Team:** TBD. **IP:** Apache-2.0; third-party notices TBD; patent filings TBD (counsel).

*Template ends.*

---

### 5.3 Why the owner's original table cannot be right

> **Box: the owner's table ("MLX 35 → Sky 60") against the physics** (SPINE §2–§3; [Part IV](part04-hardware-limits.md) §1)
>
> | Owner's claim | Why it cannot be right as written | Honest form |
> |---|---|---|
> | MLX decodes Qwen3-32B 4-bit at 35 tok/s on an M5 Max | 35 tok/s is above the datasheet ceiling of 33.9 tok/s for these weights: 17.99 GB per token at 614 GB/s. The published MLX number is 28.4 tok/s (R5, one community run). | Re-measure MLX on our M5 Max at G0/G5. |
> | Sky decodes it at 60 tok/s, plain | Needs 1.09 TB/s on a 614 GB/s machine. | Plain decode target: ≥ 1.0× mlx-lm on the M5 Max until its raw read is measured; ≥ 1.05× (≈ 30–31 tok/s) needs ≈ 575 GB/s. "60" is honest only as *effective* throughput from exact speculation: 47–59 tok/s typical with the public EAGLE-3 drafter, ≥ 60 on input-grounded or greedy work. A 60+ *average* needs a drafter nobody has published (G9). |
> | Prefill 800 → 1,600 tok/s | 1,600 × ≈ 63 GFLOP per token exceeds the fp16 peak of the neural accelerators (≈ 1,030 tok/s ceiling). It is possible only with int8 activations, and W4A8 cost Qwen3-8B 11.5 MMLU points (R8). The published MLX pp512 is 731 tok/s. | ≥ 1.0× mlx-lm on M5; ≥ 1.25× mlx-lm and ≥ 1.0× llama.cpp on M1–M4; int8 prefill only as a labeled mode with its quality delta. |
> | RAM 24 → 20 GB | Meaningless without a context length. The floor is 20.0 GB at 4K with fp16 KV, and 27.5 GB at 32K. | ≤ 0.9× mlx-lm at 32K with fp16 KV, measured system-wide. |
> | TTFT 900 → 420 ms | A cold 700-token prompt needs ≥ 0.68 s even at 100% of fp16 peak. | Warm TTFT ≥ 1.15× better, following prefill; cold start (process to first token) ≥ 3× better; repeated prefix ≥ 1.3× better with both prefix caches on (≥ 5× only against an engine without prefix reuse). |
>
> Every target is relative to mlx-lm measured on the same machine with the same file. Nothing projected is ever shown as measured.

### 5.4 The live demo

[Part XXIII](part23-benchmarks.md) owns the demo design (its §9) and the rule that the demo shows only measured, published numbers (its §1, item 10). An acquirer-facing demo adds these requirements:

| Scene | What it shows | Rule |
|---|---|---|
| 1. Cold start | `sky run hf://…` against `mlx_lm.generate`, timed from process start | file sha256 and page-cache state on screen |
| 2. Plain decode, side by side | tok/s, bytes per token, % of bandwidth, live whole-system power | shown as parity if it is parity |
| 3. Exact speculation | Sky plain and Sky speculative at T = 0.6, then `cmp` of the two outputs; then mlx-lm's #1470 script diverging | the prompts are declared before the demo |
| 4. Long context | decode at 32K context | context length stated |
| 5. `sky explain` | one token's kernels, bytes, and predicted vs measured times | the same build as scenes 1–4 |
| 6. Reproduce | `git clone sky-evaluation && ./benchmark.sh` started live; results compared with the published JSON | pinned versions on screen |
| 7. Where Sky does not win | plain short-context decode near parity; creative-text speculation around 1.2–1.4× (R8 §9) | showing the limits is part of the credibility |

General rules:
- no projected number on any slide;
- workloads declared in advance;
- sustained numbers, not bursts;
- the same benchmark host for every scene.

### 5.5 The benchmark repository and the "why faster" evidence

**`sky-evaluation`** is public at gate G8 (SPINE D13; [Part XXIII](part23-benchmarks.md) §8). It contains:
- pinned competitor versions and model hashes;
- the protocol of R9 §7: pre-registration, quantizations built from one pinned BF16 base, correctness gates, interleaved randomized blocks, bootstrap confidence intervals;
- the results JSON schema;
- per-token timestamps taken by one external harness for every engine.

The G8 exit criterion is that **an outside party reproduces the headline within 5%**.

**The "why faster" report.** For each headline metric, a `sky explain` decomposition:
- bytes per token, kernels per token, barriers;
- time per kernel class against the roofline ([Part XXII](part22-performance-model.md)).

The same decomposition is given for the competitor where it can be measured, for example the dispatch census of R4 §7.C and R5 §8.D2. A claim without a mechanism is not shown.

### 5.6 IP hygiene

| Item | Today (2026-10-02) | Required before G8 |
|---|---|---|
| License | `LICENSE` is Apache-2.0; `license = "Apache-2.0"` in `Cargo.toml`; GitHub detects apache-2.0 `[VERIFIED]` | unchanged unless the open-source decision (§5.7) changes it; decide before the first public release |
| Dependency licenses | `cargo deny` allow list (Apache-2.0, MIT, BSD-2/3, ISC, Zlib, Unicode); crates.io only; NCSA only for the fuzzing harness `[VERIFIED: deny.toml]` | the new crates (`objc2-metal`, `tokenizers`, `minijinja`, `safetensors`, `memmap2`) checked by the same tool; R9 lists them as Apache-2.0 or MIT |
| Ported code | none: a grep of `crates/` for "ported from", "adapted from" and foreign copyright lines finds nothing; no `THIRD_PARTY_NOTICES.md` `[VERIFIED]` | create the file before the first port; per-file headers with source and commit (ROADMAP, "Building on existing work") |
| Kernels written after reading MIT code | — | attribute any kernel that follows an identifiable design, such as MLX's pre-scaled activations or llama.cpp's `mul_mv_ext`, even when no line is copied (a conservative default). Check file-level headers in the Mojo repository, whose root license reads `NOASSERTION` (R7 §15) |
| Authorship record | 50 commits from 2026-09-29 to 2026-10-01; one human author; AI co-author trailers on most commits `[VERIFIED: git log]` | keep the trailers; record the AI tool and version; require contributor sign-off (DCO or CLA) before accepting outside code; counsel's answers below |
| Private or undocumented Apple APIs | none; Sky has no GPU code yet. The plan rejects tinygrad's private compiler service (Part VI §8.3) | a CI check: no private symbols, no `__asm("air.*")` in default builds, no reliance on undocumented environment variables such as llama.cpp's `AGX_RELAX_CDM_CTXSTORE_TIMEOUT` (R6) |
| Model and drafter weights | — | record the license of every model and drafter used in demos and benchmarks in `sky-evaluation`; redistribute converted drafters only if the source license allows (counsel) |
| Name | "Sky" | trademark search (counsel) |
| Repository | private, created 2026-09-29, 0 stars, 0 forks `[VERIFIED: gh repo view, 2026-10-02]` | per §5.7 |

Questions for counsel on authorship and IP:
1. Who owns, and can register, copyright in code written with AI assistance? What can the owner represent and warrant about it in a transaction?
2. What do the terms of the AI tools used say about ownership of their output?
3. Does any employment, contractor or other agreement give a third party a claim on the owner's work? This is fact-finding, not an assumption.
4. What contributor agreement (DCO or CLA) keeps future options open, including relicensing?
5. May converted drafter weights, for example EAGLE-3 converted to Sky's format, be redistributed?
6. Is "Sky" available as a software trademark?

### 5.7 Open-source strategy

Facts that bear on the decision:
- **Apache-2.0 gives every recipient of released code a copyright license (§2) and a patent license (§3).** `[VERIFIED: LICENSE]` Once the code is released, any acquirer can use it without buying anything. A purchase would then be for people, lead time, stewardship and anything not released.
- **The field is open.**
  - MLX, llama.cpp and tinygrad are MIT; BaseRT is Apache-2.0.
  - Mojo was open-sourced under Apache-2.0 three weeks after Qualcomm completed its acquisition of Modular (R7 §5.4).
  - The one proprietary runtime in R7's survey, MetalRT, is slower than both MLX and llama.cpp on one of its four models (R7 §8.1).
- **Apple's own framework is open, and so is the plan's main proof.** The G8 exit criterion, an outside reproduction, is far easier with public code.

| Option | For | Against |
|---|---|---|
| **A. Everything open (Apache-2.0)**, from G6 or earlier | adoption; credibility; outside contributors; makes the G8 reproduction easy; Sky becomes the "popular engine" that MLX's policy asks for (§2.3) | the code has no exclusivity value; every lead becomes public the day it ships |
| **B. Open core**: compiler and runtime open; tuning database, trained drafters and enterprise features closed | keeps something to sell or license | tuning data is cheap for Apple to regenerate (§3.1); only trained drafters (G9) have real replacement cost |
| **C. Private until G8, then open** | protects the lead time | no adoption and no outside reproduction before G8; lower credibility when the claims appear |
| **D. Source-available or non-commercial license** | visibility without free commercial use | deters adoption; needs sole copyright ownership to relicense, which depends on the authorship questions in §5.6 |

**Recommendation (the owner decides):**
- Publish `sky-evaluation` and the methodology early. They build credibility and give away no engine.
- Release the engine under Apache-2.0 no later than G8, so that the outside reproduction and adoption can happen.
- Keep any trained drafter (G9) as a separately licensed asset.
- Decide with counsel before the first public release, because the authorship questions affect every option.

`[LIKELY: in this market, a closed engine without users has little to sell; see §5.8]`

### 5.8 Team and traction: an honest note

**Today** (2026-10-02, all VERIFIED):
- one human author; AI agents write most of the code (commit trailers);
- a private repository with 0 stars and 0 forks, created 2026-09-29;
- no GPU code, no Metal and no tensor IR (R1 §0);
- one development Mac, an M2 Pro with 16 GB; no M5 hardware (R1 Appendix A; SPINE D14).

Acquisitions of this kind are usually of teams with adoption [LIKELY: an acquirer pays for people and a lead it cannot hire or build faster; we have no sourced data on deal terms]. On today's facts there is nothing yet to discuss. The plan has to create the evidence.

**Evidence that would make the conversation credible.** The owner's Milestones 5–7 (Part I) are:
- beat MLX by 10–20% "without cheating";
- then by 30–50% on several real workloads, with less memory;
- then show an advantage MLX cannot recover by copying one kernel;
- all of it published and reproducible by outsiders.

In measurable form:

| Milestone | Measurable evidence | Gate |
|---|---|---|
| 5. Beat MLX by 10–20% without cheating | On the M5 Max, under the Part XXIII protocol: correctness gates passed; ≥ 1.1× mlx-lm on at least two §3 metrics where physics allows it (long-context decode, small models, prefill on M1–M4, cold TTFT). Plain short-context decode cannot reach 1.1× (§5.3) and must not be the metric. | G5 |
| 6. 30–50% on several real workloads, with less memory | Exact speculation at ≥ 1.2× mlx-lm's own speculative mode with the same draft, on a declared suite; peak RAM ≤ 0.9× at 32K; each result per workload class, never averaged into one number | G6 |
| 7. An advantage MLX cannot recover by copying one kernel | The combination in §3.9, demonstrated: speculative output = plain output in CI on ≥ 3 chips, plus the verify-cost curve, long-context attention and the static plan, each with its `sky explain` decomposition | G6–G8 |
| Published and reproducible | `sky-evaluation` public; an outside party reproduces within 5% | G8 |
| Users | installs, active users, outside contributors and at least one third-party integration, above thresholds the owner sets before G8 | G8 and after |
| Peer validation | the paper draft (G8), its submission, and an MLX or llama.cpp maintainer reproducing or adopting a Sky result | G8 and after |
| A team | at least one additional engineer or named collaborator with GPU-compiler or Apple-platform credentials (staffing: [Part XXVI](part26-roadmap-4-years.md)) | G6–G8 |

**Alternative outcomes.** These are likelier than an Apple acquisition on today's facts, and they are not exclusive of one another.

| Outcome | What it needs | What Sky gains |
|---|---|---|
| Open-source adoption | public release; `sky run hf://` that just works; honest benchmarks | users, contributors and standing; the base for every other outcome |
| Upstream partnership with MLX or llama.cpp | the verify kernel or exactness proven in Sky first. MLX asks that a kernel be proven "in a popular inference engine" before it becomes a builtin (mlx#4259) | influence, credit, and working relationships with Apple engineers |
| Work with Apple's ecosystem engineers | a demonstrated result on Apple hardware; Apple's engineers already contribute to llama.cpp (#20962) | visibility; possibly hardware access |
| Other acquirers | products that run local LLMs on Macs, or hardware vendors that buy compiler teams. Qualcomm–Modular is the one verified precedent in our evidence; its terms are not. | Sky's Apple-only scope narrows this set; non-Apple backends are out of scope in year 1 (SPINE §1) |
| A hire rather than a purchase | the same evidence, attached to the owner | the owner's career outcome |
| A research outcome | the paper | reputation |
| Scientific Sky | the language's own users | Sky's original mission, which the inference work must not consume ([Part XXVIII](part28-risks.md), R-21) |

## 6. Implementation tasks

| ID | Task | Where | Deliverable | Acceptance test |
|---|---|---|---|---|
| XXVII-T1 | Third-party notices and an attribution check | repository root; CI | `THIRD_PARTY_NOTICES.md` | CI fails if a source file carries a foreign copyright header that the file does not list |
| XXVII-T2 | Provenance and contribution policy | `CONTRIBUTING.md`; CI | AI-tool trailers kept; DCO sign-off required | a commit without sign-off fails CI |
| XXVII-T3 | One-page generator | `sky-evaluation/tools/onepager` | renders §5.2 from results JSON | a number absent from the JSON cannot be rendered; the G8 build fails if any `TBD` remains |
| XXVII-T4 | Invention disclosures P1–P7 for counsel | private, not published | one note per mechanism with its prior art | the owner's decision recorded before any public release |
| XXVII-T5 | The acquirer demo script | `sky-evaluation/demo` | scenes 1–7 of §5.4 | runs end to end on the benchmark host and logs every file hash and version |
| XXVII-T6 | Private-API lint | `sky-metal` build script; CI | no private symbols, no `air.*` assembly in default builds | CI fails on a deliberately inserted private call |
| XXVII-T7 | Model and drafter license registry | `sky-evaluation` | a license ID on every model in the results JSON | schema validation fails when it is missing |
| XXVII-T8 | Competitor watch | `sky-evaluation` | monthly re-measurement of MLX/mlx-lm, llama.cpp, Monolith, Core AI and MTPLX on pinned versions | a monthly JSON with version pins; Sky's claims re-checked against it |

## 7. Measurements that decide

- **Sky's lead in exact speculation at G6.** Verify-cost curves (M = 1…32) for Sky and for the then-current MLX on the M5 Max; byte-identity tests; MLX's release notes. This decides whether §3.2 is a real lead or a gap that has already closed.
- **Core AI's LLM path on OS 27**: Qwen3-32B decode and prefill on the M5 Max, at G5. This decides whether Apple's internal alternative already covers the category ([Part XXVIII](part28-risks.md), R-6).
- **Autotuner gain over hand-picked defaults** on small-M prefill and verify shapes, at G5. This decides whether §3.1 is worth more than hand-tuning.
- **Adoption after release**, against thresholds the owner sets before G8. This decides between acquisition-led and open-source-led outcomes.
- **Counsel's answers on authorship**, before G8. These decide what the IP file can say.

## 8. Risks and negative results

- **Negative result: no durable technical moat against Apple.** Every candidate is months at most. The lead must come from shipping first, with proof, and from people and users.
- **The kernel-level speculative advantage over MLX on the M5 Max may be about 1.2×, not 2×** (§3.2).
- **AI-assisted authorship is an open diligence question** (§5.6).
- **Releasing under Apache-2.0 makes the code available to any acquirer without a purchase** (§5.7).
- The skeptic's full register, with kill criteria, is in [Part XXVIII](part28-risks.md).

## Disagreement with the spine

SPINE §3, in the "Real MLX today" cell of the speculative-decode row, says mlx-lm's "verify costs β ≈ 0.48 per extra token, at which EAGLE-3 gains nothing (R8)". That number comes from the M2 Pro: MLX `qmv` with 4 rows costs 2.43× one row, and `qmv_wide` is not used for affine 4-bit on that GPU generation (R4 §7.D, R3-issues §E2).

On the headline chip, SPINE §2 itself cites contributor data on an M5 Max (mlx#3791): +12%, +48% and +125% at 3, 5 and 8 tokens. That implies β ≈ 0.06–0.18, and MLX's speculative mode reached 24.5 tok/s on a 70B (§3.2).

So on the M5 Max, MLX is not at "β ≈ 0.48", and the kernel-level advantage is about 1.2× (§3.2). Resolved: the spine now qualifies β ≈ 0.48 as an M2-class number (M5 Max: β ≈ 0.06–0.18), and sets the bar at ≥ 1.2× over mlx-lm's own speculative mode with the same draft. [Part XXVIII](part28-risks.md), R-3, turns this into a kill criterion: pivot below 1.2×, stop below 1.1×.

Resolved: the spine now says β ≈ 0.5–1 on M1/M2-class GPUs (R4) and β ≈ 0.06–0.18 on the M5 Max (mlx#3791), puts Sky's verify-kernel advantage over mlx-lm's speculation on the M5 at ≈ 1.2×, and sets the bar at ≥ 1.2× over mlx-lm's own speculation with the same draft, with exactness as the differentiator (SPINE §3).
