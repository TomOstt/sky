# Part XXVI — The 4-year research roadmap and staffing

This part is Roadmap C of the mission. It covers:
- what Sky Inference becomes in years 2–4;
- which research bets it makes, and when each is decided;
- multi-device, new chips, Core AI and publications;
- who the project needs, and in what order;
- what one founder with AI agents can and cannot do;
- the hardware and other budget items;
- how the project chooses between an acquisition conversation, a partnership and an open-source community path.

Year 1 is [Part XXV](part25-roadmap-12-months.md). The moat and the acquisition package are in [Part XXVII](part27-moat.md); the kill criteria are in [Part XXVIII](part28-risks.md).

The further ahead a plan looks, the less it should promise. So years 2–4 are written as themes, bets and decision points, each tied to a measurement, not as a schedule. Labels follow the report's convention: `[VERIFIED: source]`, `[LIKELY: reason]`, `[EXPERIMENTAL: what decides]`, `[MOONSHOT]`.

## 1. Summary

- **Year 1 ends in one of three states,** decided by measurements at G5, G6 and G8 (§2). Years 2–4 differ by state, and each bet below runs only in the states that justify it.
- **The research bets, ranked** by expected value against risk [LIKELY: our judgment from the evidence in Parts XII, XX, XXI and XXVII]:
  1. a trained DFlash-class drafter for the headline model: the only route to a 60+ tok/s average on Qwen3-32B, and the one asset whose cost is not only engineering time (Part XXVII §3.9);
  2. compiler-chosen mixed bits under a KL budget (1.10–1.18× decode, modeled);
  3. `portable` numerics: the same bits on every Apple GPU generation;
  4. MoE expert streaming, for models larger than one machine's memory;
  5. multi-device: the M5 Ultra first, Thunderbolt clusters later;
  6. Sky-source model authoring as the main front door, decided by adoption, not by us;
  7. ASTC texture-compressed weights [MOONSHOT];
  8. drafting on the Neural Engine [MOONSHOT];
  9. e-graph block search, the weakest evidence;
  10. a persistent megakernel [MOONSHOT], only on Part XX's triggers.
- **New chips.** Each new chip, the M6 first, goes through a fixed procedure built from year-1 tasks (§5). The measure is days from hardware arrival to a verified tuning table. The tuning database is an asset through coverage and verification records, never through exclusivity.
- **Core AI** is both the largest threat (R-6) and a possible partner. Interop is decided after its LLM path has been measured (§6).
- **Hiring order:**
  1. a GPU / Metal kernel engineer;
  2. an ML inference engineer;
  3. a performance and benchmark engineer;
  4. developer relations;
  5. a compiler engineer;
  6. a numerical computing engineer.

  Each role is triggered by a gate's evidence (§10).
- **The pace.** A solo founder with AI agents can write the code. Hardware access, verification, review and judgment set the pace (§10.3).
- **The acquisition conversation opens only if four things are all in place** at G8: the evidence, counsel's answers, a team, and adoption. Otherwise partnership and open-source paths come first. On today's facts they are the likelier outcomes (§9; Part XXVII §5.8).

## 2. Where year 1 leaves the project

| State | Evidence at G8 | Emphasis in years 2–4 |
|---|---|---|
| **A. A lead** | all of: G6 passed with Sky's exact speculation ≥ 1.3× mlx-lm's own speculative mode on the M5 Max; byte-identity in CI on ≥ 3 chips; an outside reproduction within 5%; Core AI not ahead with exact speculation (R-6) | depth (drafters, compression); reach (chips, the Ultra); adoption and a team. The acquisition conversation becomes possible (§9) |
| **B. Parity plus properties** | plain decode at parity; and either a speculative lead in [1.1×, 1.3×), or R-7 fired (MLX shipped cheap verification and exactness first); other §3 metrics still hold ≥ 1.2× (long context, cold TTFT, energy) | the properties: exactness, contracts, `sky explain`, portability. Upstream partnership; an open-source community; publications |
| **C. Stop** | any of: G5 not met by week 30 (R-20); R-6's STOP; speculative lead < 1.1× (R-3's STOP) | an open-source or research project at reduced scope; or the scientific language as the main line (Part XXVII §5.8) |

The 1.3× in states A and B is deliberately above the G6 pass bar of ≥ 1.2× ([Part XXV](part25-roadmap-12-months.md); Part XXVIII R-3). Passing a gate and holding a lead worth an acquisition conversation are different claims. A result in [1.2×, 1.3×) passes G6 and counts here as "parity plus properties".

## 3. The bets

### 3.1 The register

Every bet starts with a spike whose kill criterion is written down before the spike runs, as Part XII §11 does for ASTC (XXVI-T3).

| # | Bet | Label | Upside | What decides | Kill | Earliest | Needs |
|---|---|---|---|---|---|---|---|
| B1 | A trained DFlash-class drafter for Qwen3-32B, then for each new headline model | EXPERIMENTAL | τ ≈ 4.5 gives 81–93 tok/s at β ≤ 0.05 on the M5 Max (modeled, Part IV §8). DFlash reaches τ = 5.88 on Qwen3-8B at T = 0; DSpark 4.42–5.74 on Qwen3.8-27B [VERIFIED: R8 §2.3] | a pilot that reproduces the published 8B recipe; then τ ≥ 4.0 at T = 0.6 on the workload mix, measured before any runtime work (XXI-T12); R-4 passed at G5, since a 16-token block needs the neural-accelerator verify tier | τ < 4.0 after the pilot; or R-4 failed | year 2, first half | the ML inference engineer; rented data-center GPUs, with a budget unknown until the pilot (Part XXI §12) |
| B2 | Mixed bits per tensor under a KL budget | EXPERIMENTAL up to 8B; MOONSHOT for 32B on one machine | 4.5 → 3.8–4.1 average bits per weight at Q4_K_M-like KL, i.e. 1.10–1.18× decode [EXPERIMENTAL: R8 §4.4] | XII-T12: at mlx-lm `dynamic_quant`'s bits per weight, Sky's KLD is no higher; or at equal KLD, fewer bytes; on Qwen3-1.7B and 8B | no win over `dynamic_quant` and DWQ at equal bits per weight | year 2 | BF16 weights (64 GB for 32B); evaluation time |
| B3 | `portable` numerics: the same bits on every Apple GPU generation | EXPERIMENTAL | reproducible evaluation and audited deployment, where exactness may become a requirement (Part XXVIII §6) | the cost of `portable` against `inference` for decode, verify and prefill (Part X §14); Sky-owned transcendental functions in MSL, compared bit for bit across two macOS versions and two GPU families (Part III §11) | a cost above what the owner accepts for a product mode; it then stays a test mode | study in year 2; product in year 3 | the compiler engineer; the numerical computing engineer, who also owns F10's single libm |
| B4 | MoE expert streaming, for models above the GPU working set | EXPERIMENTAL | models that do not fit: a 128 GB machine has a working set of about 95 GB [LIKELY: Part XXIII §4.4], while Qwen3-235B-A22B at 4.5 bits per weight is about 132 GB [arithmetic on the nominal 235 B parameters]; also 30B-class MoE models on 16–36 GB Macs | SSD read bandwidth and expert hit rates under a policy chosen from profiles (Part X §9): "Paging the Experts" measured 0.00% hits for a 512 MiB LRU cache against 18.80% for random eviction, on an iPhone [VERIFIED abstract: R8 §7.4]; tokens per second against the competitors' over-subscribed runs on the same machine | no gain over the best competitor configuration on the same machine | year 2, second half | the ML inference engineer; 16–36 GB test Macs |
| B5 | Multi-device | EXPERIMENTAL | §4 | §4 | §4 | year 2 (the Ultra); year 3 (clusters) | §4 |
| B6 | Sky-source model authoring as the main front door | EXPERIMENTAL | the language as a separate asset; models as checked, explained source (Part XXVII §3.5) | adoption: outside model files written in Sky; the D7 two-front-door test passing for every family. Against it: Apple's model path is PyTorch into Core AI, and competitors' model files are 159–234 lines (Part XXVII §3.5) | R-22 at G8: if evaluators care only about the runtime, the engine is packaged separately and the Rust builders stay primary | year 2 | the compiler engineer |
| B7 | ASTC texture-compressed weights | MOONSHOT | Qwen3-32B at 3.56 bits per weight streams 14.2 GB per token: ≤ 1.26× decode roofline (Part XII §11). Apple's server model uses it [VERIFIED: R8 §4.2] | XII-T14: ASTC texture reads ≥ 90% of buffer bandwidth on the M5 Max; KLD against MLX 4-bit at equal bytes, after a LoRA recovery step | either test fails | spike in weeks 44–52; year 2 | the kernel engineer; quality runs |
| B8 | Drafting on the Neural Engine | MOONSHOT | Mirror-SD reports 2.8–5.8× wall time and about 30% over EAGLE-3, but on 8 M2 Ultra GPUs in tensor parallel plus 8 Neural Engines, not one chip [VERIFIED: R8 §2.5–§2.6] | the draft step's time on the Neural Engine through Core AI or Core ML (§6, C3), and the GPU verify slowdown under concurrent Neural Engine traffic: on one M5 Max both share 614 GB/s [LIKELY contention: R8 §2.6] | effective throughput not above the GPU drafter's | year 3 | the ML inference engineer; Core AI interop |
| B9 | E-graph block search | EXPERIMENTAL; the weakest evidence | Tensat found graphs up to 16% faster on convolutional models; Cranelift's production e-graph gained about 0.1%; an LLM decode graph is small and its useful rewrites are already in Part VIII §3 [VERIFIED: R7 §6] | VIII-T11 reproduces the deterministic pipeline's decisions in under a second; VIII-M5: an e-graph plan measurably faster on ≥ 2 model families (mixed formats, Q4_K_M layers) | R-13 froze the general layers at G5; or no measured win | year 2 | the compiler engineer |
| B10 | A persistent megakernel | MOONSHOT | ≤ 3% on dense 32B decode; about 15% of a 0.6B step is boundary bubbles (Part XX §8.2) | Part XX §8.4's four triggers: acquire/release atomics working on macOS 27; a stable co-residency probe; bubbles > 10% of the draft step after fusion; a known watchdog limit | any trigger unmet: no spike (XX-T13) | only on its triggers | the kernel engineer |
| B11 | Entropy-coded lossless weights | EXPERIMENTAL | 5.6–9% fewer bytes with bit-identical output; allowed in same-file comparisons because it is lossless (Part XII §10.4) | variable-length decoding within the GEMV's ALU budget | decoding costs more than the bytes it saves | year 2–3 | the kernel engineer |

Ordering rules:
- **Bets that change outputs are labeled modes** with a KL budget, never part of a same-file comparison (D5; Part XII Q-D6): B2, B7, W4A8 prefill, 4-bit KV.
- **Bets that need the neural-accelerator verify tier wait for R-4's G5 decision:** B1 and wide trees.
- **Only B1, B5 and B7 can move the headline numbers by more than about 20%.** The rest are single-digit to low-double-digit gains, or capabilities (B3, B4, B6) [LIKELY: from the upsides above].

### 3.2 Smaller research items

These continue year-1 work, decided by the owner when capacity allows:
- **W4A8 prefill with rotation (QuaRot, SpinQuant).** Research only. W4A8 with SmoothQuant cost Qwen3-8B 11.5 MMLU points, and there are no Qwen3 numbers for rotation (Part XII §9).
- **4-bit KV (XIV-T12).** Only once Qwen3 quality data exist (Part XIV KV-D5).
- **AI-proposed kernel variants under verification (Part X §12).** These run in Sky's build farm, never on a user's machine. They are verified like hand-written candidates: reference, poisoned outputs, adversarial inputs, tolerance and invariance. A variant is kept only if it wins on ≥ 2 chips. Unverified agent kernels have produced fake speedups elsewhere: a claimed 150× kernel was 3× slower [VERIFIED: R7 §12.1].
- **A shared tuning-record exchange (Part X §10).** ROADMAP.md already says shared cross-user optimization databases raise privacy and trust questions. The owner decides hosting, signing, consent and legal review before any code is written.

## 4. Multi-device

Multi-device is out of scope in year 1 (SPINE §1). Buyers will ask about it because MLX "now supports … RDMA over Thunderbolt" (WWDC26 State of the Union) [VERIFIED: R6 §A3].

### 4.1 The M5 Ultra

The facts [VERIFIED: R6 §A1]:
- shipped 2026-09-22;
- two dual-die M5 Max chips joined by UltraFusion, with more than 4.4 TB/s between dies;
- 1.2 TB/s of memory bandwidth; 64 or 80 GPU cores;
- up to 512 GB, with the 512 GB configuration shipping "late October".

What it changes:
- **Plain decode of Qwen3-32B** has a planning ceiling of 59.6 tok/s on the Ultra (Part IV §4). That is the owner's "60 tok/s" without speculation, and effective speculative rates scale from there.
- **Models that fit only there.** DeepSeek-V3/R1 at 4.5 bits per weight is about 377 GB (Part XIII §5.2). Qwen3-235B-A22B (§3.1, B4) also fits.
- **Unknowns.** NUMA effects across four dies are an open question (R6). So is whether one Metal device spans the dies [LIKELY: earlier Ultra parts present one GPU to Metal], and what bandwidth a die gets when the weights sit in another die's memory.

The plan:
- In year 1 the Ultra is an optional demo host (Part XXVIII §6).
- In year 2 it is a first-class target: the day-one kit, a NUMA probe that measures bandwidth by placement, tuning tables, and Tier H.
- The decision: if placement matters, a placement-aware memory plan (Part XIX) becomes a year-2 task. Otherwise the Ultra is simply a faster chip.

### 4.2 Thunderbolt clusters

We have measured no Thunderbolt latency or bandwidth. Two designs are possible:

| Design | What it buys at batch 1 | Exactness | What decides |
|---|---|---|---|
| **Pipeline parallel** (layers split across Macs) | models that fit no single Mac. It does not speed up one stream: each token passes through every stage in order [LIKELY: standard behavior of batch-1 pipelines] | each stage keeps its kernels' reduction orders, so the output can stay byte-identical to a single device with the same plan [LIKELY: the arithmetic per layer is unchanged] | demand for a model no single Mac holds |
| **Tensor parallel** (each layer split) | aggregate bandwidth, so it can speed up decode. It needs two collective operations per layer. On one M5 Max a Qwen3-32B layer's weight read takes about 0.53 ms (33.7 ms over 64 layers; Part XV §2.1), so each collective must cost a small fraction of that | it splits reductions across devices, so byte-identity with a single device needs a canonical cross-device reduction order [LIKELY] | the measured collective latency over Thunderbolt RDMA, against the per-layer time |

Mirror-SD's results (B8) used a tensor-parallel rig of 8 M2 Ultras (R8 §2.5). The decision is in year 3:
- measure RDMA latency and bandwidth per collective;
- build pipeline parallelism only for a model the project needs that no single Mac holds;
- build tensor parallelism only if the measured collective cost leaves a net gain.

## 5. New chips and the tuning database

The facts:
- The M6, Apple's first 2 nm part, ships in the Mac mini with a 12-core GPU, neural accelerators and up to 170 GB/s [VERIFIED: R6 §A1].
- MLX shipped M5 Pro/Max tunings nine days after those chips were announced, and M5 Ultra tunings seven days after it shipped [VERIFIED: R3-issues §B3]. Apple has hardware before launch; Sky does not (Part XXVII §2.1).

**The new-chip procedure.** Every step is an existing task:
1. the day-one kit (IV-T2): bandwidth, dispatch costs, `matmul2d` rates by tile and dtype, sustained clocks, `PSTR`;
2. calibration into `gpumodel.v1` records (XXII-T2);
3. the compiler-regression self-check on the chip's OS and Metal compiler (XI-T9);
4. tuning seeded from the nearest chip family (X-T10), then `-O3` on the hot kernel instances;
5. the exactness suite, the invariance suite and the same-quantization proof, run on the chip;
6. a dated "Sky on <chip>" page in `sky-evaluation`, with the shipped per-family default table.

**The measure** is days from hardware arrival to step 6. MLX's window above benefits from pre-launch access, so the two are not directly comparable.

**The tuning database as an asset, honestly.** Part XXVII §1 estimates that Apple would need days to weeks to regenerate tuning data. Its value is therefore not exclusivity. It lies in three things:
- coverage across chips and OS builds;
- an entry is stored only after it has been checked against the reference (Part X §6.4);
- the re-verification history across OS updates, which records when an OS changed speed or bits.

## 6. Core AI interop

The facts [VERIFIED: R6 §A3]:
- OS 27; a Swift API; PyTorch conversion through `coreai-torch`; ahead-of-time compilation; per-device specialization with a model cache;
- in-place KV "states"; custom Metal 4 kernels through `TorchMetalKernel`; Instruments support and a debugger;
- "fast inference across the CPU, GPU, and Neural Engine".
- Nobody we found has published its LLM decode or prefill speed.
- One measurement shows about 65 TFLOPS fp16 on the M5 Max GPU, and no int8 × int8 path [EXPERIMENTAL: one source].

The options:

| # | Option | When | What decides |
|---|---|---|---|
| C1 | Core AI as a third baseline (R-6) | G0–G5 | R-6 |
| C2 | Sky's kernels inside Core AI: the verify GEMV family, `flash_decode` with fixed splits, the keyed-Gumbel sampler, shipped as `TorchMetalKernel` custom kernels to give Core AI users exactness | year 2, after C1 | whether the custom-kernel interface admits Sky's argument ABI and fixed reduction schedules, and whether Core AI's scheduling preserves batch invariance [EXPERIMENTAL] |
| C3 | Core AI as the Neural Engine path for a drafter (B8) | year 3 | B8's measurements |
| C4 | Sky's exactness suite and the `sky-evaluation` protocol, offered as conformance tests for Core AI and MLX | year 2 | interest from the maintainers |

The decision rule:
- **If R-6 STOPs the Apple thesis** (Core AI ≥ mlx-lm with exact speculation), C2 and C4 become the Apple-facing path: Sky contributes.
- **If Core AI's LLM path is slow, or closed to MLX and GGUF files** (Part XXVIII §6), C1 continues as a baseline, and C2 is optional.

## 7. Publications

The mission names MLSys, ASPLOS and PLDI. Deadlines, page limits and artifact-evaluation rules must be checked on each venue's call for papers; none are given here.

| Paper | Venue fit | Evidence it needs | Earliest |
|---|---|---|---|
| Exact speculative decoding on Apple GPUs: batch-invariant verify kernels, keyed sampling, and rounds without host synchronization | MLSys | G6 and G8 results; the exactness suite; ≥ 3 chips | after G8 (the G8 paper draft) |
| A tensor IR with numerics contracts and a verifier-enforced reduction-schedule invariant | PLDI | TIR (Part VII), verifier rules V17–V18, the `inference` and `portable` contracts, the D7 two-front-door test | year 2 |
| What a compiler can and cannot win on unified-memory GPUs: measured limits and negative results | ASPLOS | Parts IV, V, XX and XXII; the closure tool; the negative results (megakernels, replay, 2-bit codebooks, plain decode) | after G8 |
| A hostile-review protocol for local LLM inference | MLSys, or a workshop | `sky-evaluation` and its outside reproduction | after G8 |

Rules:
- The patent or defensive-publication decision comes before any submission (XXVII-T4; Part XXVII §4, question 2).
- Artifacts are public.
- Every number comes from `sky-evaluation`.
- An academic co-author adds the peer validation Part XXVII §5.8 asks for.

## 8. Years 2–4 by theme

| Theme | Year 2 | Year 3 | Year 4 |
|---|---|---|---|
| Speculation | B1: pilot, then the 32B drafter if τ ≥ 4.0; acceptance studies per model; MTP for new hybrid models | B8 if C3 works; drafters for new headline models | a drafter per new headline model; trained drafters as a separately licensed asset (Part XXVII §5.7) |
| Compression | B7 spike; B2 up to 8B; B11 study; W8A8 if G7 shipped it | B2 on 32B if the evaluation budget exists; 4-bit KV if Qwen3 data exist | by results |
| Compiler | B3 cost study; B9 prototype if R-13 did not freeze the general layers; AI-proposed variants under verification | `portable` as a product mode if its cost is accepted; B6 if adopted | by results |
| Scale | the M5 Ultra as a first-class target; B4 for over-memory models | Thunderbolt pipeline parallel, if needed (§4.2) | tensor parallel, if the measured collectives allow it |
| Chips | the M6 through §5 | the next generation through §5 | the same |
| Apple stack | C2 and C4 | C3 | by the partnership's state |
| Serving | paged KV (XIV-T9) and multiple sessions | larger-batch serving, if users ask; it is out of scope in year 1 (SPINE §1) | — |
| Scientific Sky | the scientific-track fixes F2, F4, F5, F7, F9, F10 (III-T8); F10's single libm shared with B3 | tensor types for scientific code through `tensor.call` | the language's own roadmap |
| Other backends | — | decide on a non-Apple backend (Part VI §8.2, option D: MLIR, Triton or Mojo); R-23 notes that an Apple-only scope narrows the set of buyers | if decided |

```text
              year 1 (Part XXV)           year 2                       year 3                     year 4
gates         G0 … G5 … G6 … G8 ───┐
decisions                          ├─ DP-8 (§9) ── DP-8+6 (R-23) ──── DP-Y3: clusters, backends
speculation   EAGLE-3 · drafts · MTP ─ B1 pilot → 32B drafter? ─────── B8 via Core AI? ─── drafters per model
compression   same-file only ───────── B7 spike · B2 ≤ 8B · B11 ─────── B2 32B? · 4-bit KV? ───────────────
compiler      skeletons + static plan ─ B3 cost · B9 prototype? ─────── portable mode? · B6? ───────────────
scale         one Mac ───────────────── M5 Ultra · B4 ──────────────── pipeline parallel? ─ tensor parallel?
chips         M2 Pro · M5 Max · M4 Max ─ M6 (§5) ───────────────────── next generation (§5) ──────────────
people        owner + agents · hire 1 ─ hires 2–4 ──────────────────── hires 5–6 ─────────────────────────
```

## 9. Decision points tied to evidence

These follow Part XXVII §5.8 and Part XXVIII's kill criteria.

| # | When | Evidence (measured) | If | Then |
|---|---|---|---|---|
| DP-5 | G5 (week 20) | Core AI on Qwen3-32B under OS 27 | Core AI ≥ mlx-lm on decode and prefill, with exact speculation | STOP the Apple-acquisition thesis (R-6); interop through C2 and C4; the other outcomes of Part XXVII §5.8 |
| DP-6 | G6 (week 28) | Sky's speculative mode ÷ mlx-lm's, same draft | < 1.1× | STOP the speculative differentiator (R-3): state B or C |
| | | | [1.1×, 1.3×), and every other lead within what BaseRT already reports | contribute the verify kernels and exactness upstream; drop the separate-engine acquisition plan (R-2): state B |
| DP-8 | G8 (week 44) | outside reproduction; leads against the competitors' then-current releases; chips; counsel; adoption (thresholds from XXVIII-T4); team | **all of:** reproduction within 5%; exact speculation ≥ 1.3× mlx-lm's speculative mode; byte-identity on ≥ 3 chips; counsel can support ownership representations (R-24); adoption at or above the thresholds; at least one additional engineer or named collaborator (Part XXVII §5.8) | open acquisition conversations, in parallel with partnership talks (state A) |
| | | | the evidence is complete, but adoption or the team is below the bar | partnership and open source first; check again at G8 + 6 months |
| | | | the reproduction fails | publish nothing; fix and re-run (R-17); no conversations |
| DP-8+6 | G8 + 6 months | adoption; inbound interest | below the thresholds, and no potential acquirer or partner has made contact | STOP optimizing for acquisition (R-23). Continue toward the open-source, research or hire outcomes |
| DP-Y2a | year 2, after B1's pilot | τ at T = 0.6 on the workload mix | ≥ 4.0 | train the 32B drafter and license it separately (Part XXVII §5.7) |
| DP-Y2b | year 2 | the cost of `portable` (B3) | within the owner's tolerance | `portable` becomes a product mode |
| DP-Y2c | year 2 | Sky-source adoption (B6) | below a threshold the owner sets at G8 | the Rust builders stay primary; the language is packaged separately (R-22) |
| DP-Y3a | year 3 | Thunderbolt collective cost; demand for over-memory models | pipeline parallel needed; tensor parallel with a net gain | build what the measurements justify (§4.2) |
| DP-Y3b | year 3 | demand outside Apple: users, buyers, partners | present | decide on a non-Apple backend (Part VI §8.2, option D) |

## 10. Staffing

### 10.1 Roles

A role can be filled by an employee, a contractor or a named collaborator. Part XXVII §5.8 counts any of them as "a team".

| Role | Owns | Why a person, not only agents |
|---|---|---|
| **GPU / Metal kernel engineer** | Parts XI, XV, XVI, XVII, XX; `sky-metal` and the `sky-kernel` skeletons | judgment about hardware behavior that no oracle encodes: register spills, the fragile neural-accelerator path (R-10), the opaque MPP lowering, OS drift. It takes a person to know why a kernel is slow |
| **ML inference engineer** | Parts XII, XIV, XXI; `sky-quant`, `sky-llm::spec`; drafter training (B1) | choices about drafters, data and quality budgets; training on rented data-center GPUs; the acceptance studies |
| **Performance and benchmark engineer** | Part XXIII; `sky-evaluation`, performance CI, the protocol, the outside reproduction, the dashboard | separation of duties: the person who measures is not the person who wrote the kernel. Credibility is rated Critical (R-17) |
| **Compiler engineer** (IR, scheduler) | Parts VII–X and XXII; `sky-tir`, the scheduler, the autotuner; B3, B6, B9 | the general layers pay only if someone owns them (R-13); the language front door |
| **Developer relations / community** | releases, documentation, issue triage, model-coverage requests, relations with the MLX and llama.cpp maintainers | adoption is a precondition of every outcome (Part XXVII §5.8). The maintainers' review capacity is already a stated limit (R4 §9) |
| **Numerical computing engineer** (scientific Sky) | Parts II–III; Roadmap A's scientific track; F10's single libm, shared with B3 | keeps the language's original mission alive without consuming the inference work (R-21) |

### 10.2 Order and triggers

| Order | Role | Trigger | Why at that point |
|---|---|---|---|
| 1 | GPU / Metal kernel engineer | G5 passed (week 20). This follows R-20's mitigation, "collaborators recruited once G5 evidence exists". Earlier if a G1 or G4 kill criterion trips on kernel performance | G5–G6's critical path is kernel work on the neural-accelerator path. Kernels need the most expert review, and the owner's review is the bottleneck |
| 2 | ML inference engineer | G6 passed with ≥ 1.2× (week 28), or B1's pilot is funded | the drafters (EAGLE-3 conversion, DFlash, B1), acceptance studies, quantization quality |
| 3 | Performance and benchmark engineer | G8 starts (week 30), before the outside reproduction. It moves to first if the G5 review finds disputed measurements or an overloaded M5 host | the protocol, the reproduction and the dashboard need an owner other than the kernel author |
| 4 | Developer relations / community | the first public release, no later than G8 (Part XXVII §5.7). Part-time at first [assumption] | adoption is measured from G8 (R-23) |
| 5 | Compiler engineer | year 2, if R-13 did not freeze the general layers at G5 (moves to third if the autotuner and partition search gained ≥ 5%), or if B6 shows adoption | B3, B6 and B9 need an owner; until then the skeletons and the static plan suffice |
| 6 | Numerical computing engineer | year 2, when the scientific track resumes (III-T8's fixes) | Roadmap A is secondary until then (D16) |

Hiring uses work samples drawn from the plan's own tasks (XXVI-T2):
- for the kernel role, a batch-invariant verify GEMV judged by its oracle and its bandwidth (XI-T11);
- for the ML role, an acceptance study (XXI-T15);
- for the benchmark role, reproducing one Part XXIII cell from its pre-registration.

### 10.3 What a solo founder with AI agents can and cannot do

| Activity | With agents | What limits it | Evidence |
|---|---|---|---|
| Writing code across crates | fast. Sky reached about 75k lines of production Rust in 50 commits over three days, with one human author | review | [VERIFIED: git log; Part XXVII §3.9] |
| Correctness | strong where an oracle exists: the reference interpreter, bit-equality tests, golden ids, poisoned outputs | writing each oracle, and choosing its tolerances | Parts VII, XII, XXIII |
| Kernel candidates | fast to propose; unreliable to judge. Unverified agent kernels have produced fake speedups | a verification harness (Part X §12) | [VERIFIED: R7 §12.1] |
| Performance verification | cannot be accelerated: it needs a quiet machine and interleaved protocol runs. Tier H takes about two days, Tier A about a week per chip | machine time on one M5 Max | [LIKELY: Part XXIII §4.6] |
| Hardware | agents cannot buy, rack or keep machines; Apple has chips before launch | the owner's decisions and budget | Part XXVII §2.1 |
| Judgment | agents can prepare the evidence for a kill or pivot call, or for a choice such as the ALU or the neural-accelerator family; they cannot own the decision | the owner | Part XXVIII |
| Upstream relations and credibility | limited: mlx-lm's maintainers wrote "At the moment we don't accept AI written architecture changes" | people | [VERIFIED: R4 §9] |
| Due diligence | AI-assisted authorship is an open question for any transaction | counsel | Part XXVII §5.6; R-24 |
| Training a drafter | needs data-center GPUs and ML judgment about data | compute and a person | Part XXI §12 |
| Continuity | a bus factor of one | a team | R-20 |

The consequence: the plan's throughput is set by the owner's review rate, the M5 host's hours, and the share of work packages whose acceptance is an oracle. Agents beyond the review limit add rework, not speed [LIKELY]. The first hire's main value is a second reviewer with hardware judgment, not more code.

## 11. Budget items

This report states no prices. Each price must be checked at purchase, from the source named here, and recorded in the budget register (XXVI-T1).

| Item | Purpose | When | Quantity | Where to check the price |
|---|---|---|---|---|
| M5 Max, 40-core GPU, 128 GB (a Mac Studio preferred for thermals) | the benchmark host (Part XXV I1) | ordered at G0; in service by week 5 | 1 | Apple's configurator on the purchase date |
| A second M5 Max, 40-core, 128 GB | the perf runner and protocol isolation; an OS 27 volume for Core AI (Part XXV I3, I5) | by G4–G5 | 1 (recommended) | the same |
| A macOS CI runner, preferably a non-M5 GPU family | Metal correctness on every PR (Part XXV I2) | by week 5 | 1 | Apple, or the used and refurbished market |
| M4 Max, 40-core, 128 GB | the third chip (Part XXV I7) | by week 30 | 1, bought or borrowed | Apple; resellers |
| Optional: an M3 Max; an M5 Pro | reproduce mlx-lm's 1.82× on the M3 Max; compare with BaseRT on the M5 Pro (Part XXIII §4.4) | G8 | 1 each | the same |
| M5 Ultra, 256 or 512 GB | demo host; over-memory models; NUMA study (§4.1) | late year 1 or year 2 | 1 | Apple; the 512 GB configuration ships "late October" (R6) |
| M6 Mac mini | the new-chip procedure (§5) | year 2 | 1 | Apple |
| Wall-power meter | whole-system energy (Part XXV I6) | by week 24 | 1 | the vendor's list price |
| Storage, more than 1 TB on the benchmark host | Tiers H and A (Part XXV I8) | by week 20 | — | the vendor's list price |
| Data-center GPU time | training the 32B drafter (B1) | year 2 | unknown until the 8B pilot (Part XXI §12) | cloud providers' prices at the time of the pilot |
| Counsel | patents P1–P7, authorship, trademark, the contributor agreement (Part XXVII §4, §5.6) | before the first public release; before G8 | — | engagement terms |
| Thunderbolt cables and a second Ultra-class machine | the multi-device study (§4.2) | year 3 | — | Apple; vendors |
| Hosting and signing | the public `sky-evaluation`; a tuning-record exchange, if the owner decides on one | G8; year 2 | — | the providers |
| Conference registration and travel | publications (§7) | from year 2 | — | each venue |
| Salaries and contracts | the roles of §10 | from the triggers of §10.2 | — | market data at the time of hiring |

## 12. Implementation tasks

| ID | Task | Where | Deliverable | Acceptance test |
|---|---|---|---|---|
| XXVI-T1 | Budget register | `docs/inference/plan/budget.md` | each item of §11, with purpose, gate, quantity, and the date and source of its price check | no purchase is recorded without a dated price check |
| XXVI-T2 | Staffing plan | `docs/inference/plan/staffing.md` | roles, triggers, and work samples from §10.2 | each trigger cites a gate review record (XXV-T4) |
| XXVI-T3 | Bet register | `docs/inference/plan/bets.toml` | B1–B11 with status, spike, kill criterion and decision date | a bet cannot be marked started without a written kill criterion |
| XXVI-T4 | Year-2 plan | a successor to this part, written at week 48 | the plan for the state reached at G8 (§2) | the owner reviews it before week 52 |
| XXVI-T5 | Publication plan | `docs/inference/plan/papers.md` | per paper: venue, evidence, artifact plan, and the patent decision | no submission before XXVII-T4's decision is recorded |

## 13. Measurements that decide

| Measurement | Decides |
|---|---|
| τ of a drafter trained on the target's own sampled outputs, at T = 0.6 (B1) | whether the 60+ tok/s average is reachable on Qwen3-32B |
| the cost of `portable` against `inference` (B3) | product mode or test mode |
| ASTC texture-read bandwidth and quality (B7) | whether texture compression continues |
| NUMA effects on the M5 Ultra; Thunderbolt RDMA latency and bandwidth per collective (§4) | the multi-device designs |
| Core AI's LLM decode, prefill and exactness (§6) | the Apple thesis (R-6) and interop |
| days from hardware arrival to a verified tuning table on a new chip (§5) | whether reach across chips is real |
| adoption against the owner's thresholds (DP-8, DP-8+6) | the acquisition, partnership or open-source path |

## 14. Risks and negative results

- **Most bets are incremental.** B2 is 1.10–1.18×, B10 at most 3% on dense 32B, and B9 probably about nothing. Only a strong drafter (B1), more bandwidth (the Ultra; B5) and, if both its tests pass, ASTC (B7, at most 1.26×) can move the headline numbers by more than about 20%.
- **Apple can copy each bet in months** (R-7; Part XXVII §3). A trained drafter is the exception, because it costs compute and data.
- **Drafters age with their models.** Each new headline model needs a new drafter, at a training cost we cannot yet estimate.
- **Hiring is slow and uncertain**, and AI-assisted authorship is a diligence question for any transaction (R-24).
- **The language may never be the front door** (R-22). The engine must stand on its own.
- **Multi-device at batch 1 may disappoint.** Pipeline parallelism does not speed up one stream, and tensor parallelism depends on collective costs nobody has measured for us.
- **Negative results carried forward:**
  - the megakernel's preconditions fail on Metal (Part XX §8);
  - 2-bit codebook formats are ALU-bound on Apple GPUs (Part XII §10.1);
  - weight sparsity is rejected (Part XII §10.3);
  - command replay gives no gain on dense decode (Part XX §14);
  - plain short-context decode cannot pass about 1.05–1.14× of MLX (Part IV §4).

## 15. Disagreement with the spine

1. **The megakernel is ranked last, not first.** SPINE §6 lists it first among the G9 moonshots. The evidence ranks it last: at most 3% on dense 32B, and four unmet preconditions (Part XX §8). It runs only on Part XX §8.4's triggers (B10).
2. **"Draft-on-ANE" depends on Core AI, not on Sky's code generator.** SPINE §6 lists it as a G9 bet. In our evidence, only Core ML and Core AI packages reach the Neural Engine: `MTL4MachineLearningCommandEncoder` runs Core ML packages, and "the system automatically chooses an inference engine" (R6). So B8 depends on interop option C3 (§6). On one chip it also shares the memory bus with the GPU it is meant to relieve.
3. **Multi-device stays out of year 1, as the spine says.** But MLX already supports RDMA over Thunderbolt (R6 §A3), so a multi-device comparison will be requested in years 2–3. §4 plans for it.
