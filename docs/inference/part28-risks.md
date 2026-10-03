# Part XXVIII — Risks and reasons Sky may fail

This part is the skeptic's chapter. Its job is to try to kill the project with evidence, so that the plan learns early, and cheaply, if it is wrong. Every risk has:
- its evidence;
- a likelihood and an impact;
- a mitigation;
- a **kill criterion**: a measurement, tied to a gate (G0–G8 in SPINE §6), whose result means stop, pivot or withdraw a claim.

Labels follow the report's convention: `[VERIFIED: source]`, `[LIKELY: reason]`, `[EXPERIMENTAL: what decides]`, `[MOONSHOT]`. Research notes are cited as (R3 §7) and live in [`research/`](research/).

## 1. Summary

The five risks most likely to end the project, or its acquisition thesis:

1. **No M5 Max early enough (R-19).** The development Mac is an M2 Pro with 16 GB. Every M5-specific claim depends on hardware the project does not have, yet the spine's first M5 Max measurement is at G5 (weeks 14–20). Decided at G0.
2. **The speculative lead is smaller than planned on the headline chip (R-3, R-4, R-12).**
   - MLX's own data on an M5 Max imply a per-draft verify cost of β ≈ 0.06–0.18, not the ≈ 0.5–1 of M2-class GPUs. Sky's kernel advantage there is about 1.2×.
   - The neural-accelerator path for 16–32-token verification is unmeasured.
   - Published Qwen3-32B drafter acceptance is modest (τ 2.2–3.1).

   Decided at G1, G5 and G6.
3. **Competitors close the gap first (R-5, R-6, R-7, R-8, R-9).**
   - Monolith, from the author of MPK, targets the same gap.
   - Core AI is Apple's own ahead-of-time model compiler.
   - MLX tunes for new chips within a week of launch, and llama.cpp iterates with AI help and crowd-sourced tuning.
   - On the model buyers will actually run (Qwen3.6/3.8-27B with MTP), an MLX-based tool already claims 65–88 tok/s.

   Decided at G5 and G6.
4. **Credibility (R-15, R-17).** One disputed benchmark can undo the project with the people an acquirer would ask. Decided at every gate; finally at G8.
5. **Team and adoption (R-20, R-23, R-24).** One person with AI agents, a private repository, no users, and an open question over AI-assisted authorship. Technical success alone does not produce an acquisition. Decided at G8 and after.

The verdict (§7) is that the project is worth doing under five conditions, with the acquisition treated as one possible outcome among several.

## 2. How to read this part

**Likelihood** is a judgment, stated with its reason:
- **Certain**: a present fact;
- **High**: more likely than not on the evidence;
- **Medium**: plausible; the evidence is split;
- **Low**: possible, with evidence against it;
- **Unknown**: a question no measurement or source here can answer.

**Impact**:
- **Critical**: ends the project or the acquisition thesis;
- **High**: removes a headline claim;
- **Medium**: narrows the claims or delays a gate by more than four weeks;
- **Low**: cosmetic.

**Kill-criterion types**:
- **STOP**: end or suspend the project or the named strategy;
- **PIVOT**: change the headline or the plan;
- **CLAIM**: withdraw a specific claim, or never publish it.

Thresholds come from the spine's targets, from published numbers, or from R8's speculative model. Where a threshold is a policy proposal rather than a fact, it says so.

## 3. The register

| ID | Risk | Likelihood | Impact | Decided at | Kill type |
|---|---|---|---|---|---|
| R-1 | MLX is near the roofline on dense decode | Certain | Low for the revised pitch; Critical for the owner's table | G4, G5 | PIVOT / CLAIM |
| R-2 | Hand-written runtimes only tie MLX (BaseRT) | High | Medium | G6 | PIVOT |
| R-3 | MLX's verify cost on the M5 Max is already modest | High | High | G6 | PIVOT / STOP |
| R-4 | The neural accelerators may not make 16–32-token verification cheap | Medium | High | G5 | PIVOT |
| R-5 | Monolith/mpk-apple, with the MPK author behind it | High | High | G5 | PIVOT |
| R-6 | Core AI | Certain (exists); Unknown (speed) | Critical for the Apple thesis | G5 | STOP / PIVOT |
| R-7 | Apple can copy any single kernel or trick in months | High | High | G6 | PIVOT |
| R-8 | MLX and llama.cpp iterate fast | Certain | High | every gate; G8 | CLAIM |
| R-9 | The model landscape moved (hybrid 27B with MTP; MTPLX) | High | High | G5, G7 | PIVOT / CLAIM |
| R-10 | The M5 neural-accelerator path is fragile | High | High (M5 prefill) | G5 | CLAIM |
| R-11 | Batch-invariant kernels cost more than expected | Medium | High | G1, G4 | PIVOT |
| R-12 | Speculative gains depend on workload and drafters | Certain | High | G6, G8 | CLAIM / STOP |
| R-13 | The compiler may not pay for itself | Medium | Medium | G5 | PIVOT |
| R-14 | OS and driver drift; wired-memory panics | High | Medium (High on demo day) | G8 | CLAIM |
| R-15 | Quality disputes over "same quantization" | High | Medium–High | G3 onward | CLAIM |
| R-16 | Thermal throttling on laptops | High (prefill) | Medium | G8 | CLAIM |
| R-17 | Accusations of benchmark gaming | High | Critical | G8 | CLAIM / STOP |
| R-18 | Targets that are infeasible as written | Medium–High | Medium | G5 | restate |
| R-19 | M5 Max hardware access; a 16 GB development Mac | Certain (need) | Critical | G0 | STOP / PIVOT |
| R-20 | One person plus AI agents against Apple's teams | High | High | G4, G5 | PIVOT / STOP |
| R-21 | Scope creep between scientific Sky and inference | High | High | G4 | PIVOT |
| R-22 | Sky's language may be irrelevant to Apple's interest | High | Medium | G8 | PIVOT |
| R-23 | Acquisition dynamics | Low (Apple acquisition) | Critical for the stated ambition | G8 + 6 months | STOP |
| R-24 | IP and authorship questions block a transaction | Unknown | High | before G8 | PIVOT |

## 4. The risks in detail

### A. Physics and the incumbent

**R-1. MLX is near the roofline on dense decode.**
- *Evidence.*
  - On the M2 Pro, MLX streams at 182–184 GB/s (89–90% of 204.8 GB/s). Its large decode GEMVs run at 91–97% of that, and a whole decode step at context 128 runs at 94–97% on both 32B and 8B shapes (R3 §8.1).
  - mlx-lm decodes Qwen3-8B at 91% of MLX's own ceiling and 84% of the raw-Metal read (R4 §1, R6).
  - On an M5 Max, MLX decodes Qwen3-32B at 28.4 tok/s. That is ≈ 511 GB/s: ≈ 83–84% of the 614 GB/s spec, depending on the bytes counted, and ≈ 96% of the only achievable-bandwidth measurement on that chip, ≈ 530 GB/s (R5 §9.3, R3-issues §A4; LIKELY, one community run).
  - Command-buffer policy is worth ≈ 0% on dense models (R3-issues §D1).
- *Likelihood: Certain* that headroom is small. *High* that Sky's plain dense-decode gain is ≤ 1.05–1.10× (R3 §8.3: ≤ 3–6%; R4 §10.3: ≤ 1.1×).
- *Impact: Low* for the pitch the spine adopted (D18: parity is table stakes). *Critical* for the owner's original table, which is already withdrawn ([Part XXVII](part27-moat.md) §5.3).
- *Mitigation.* Treat plain decode as table stakes. Pitch speculation, long context, small models, TTFT, RAM, energy and exactness. Measure raw-Metal bandwidth on the M5 Max on day one ([Part IV](part04-hardware-limits.md) IV-T2).
- *Kill criterion.*
  - **G4, PIVOT.** If 8B plain decode on the M2 Pro stays below 1.0× mlx-lm after fusion, the static plan and replay, stop generating decode kernels and hand-write the decode path before going further.
  - **G5, CLAIM.** If 32B plain decode on the M5 Max is below 1.0× mlx-lm, no material may claim a decode advantage. The project continues only if R-3 and R-12 pass.

**R-2. Hand-written runtimes only tie MLX (BaseRT).**
- *Evidence.*
  - BaseRT (arXiv 2607.19438; hand-written Metal 4 kernels; an M5 Pro) on dense Qwen3.6-27B Q4: decode 18.1 tok/s against mlx-lm's 18.4 and llama.cpp's 15.4. Prefill 1.44× at a 128-token prompt but 1.02× at 2,048. Decode 1.33× on Qwen3-0.6B. Its own conclusion: "BaseRT and MLX both operate close to what the hardware's bandwidth allows" (R4 §8, R7 §8.1).
  - MetalRT, a proprietary runtime, is slower than both MLX and llama.cpp on Llama-3.2-3B (R7 §8.1).
  - Mojo's W4A16 matmuls run at 0.85–0.96× MLX on the M5 (R7 §5.4).
- *Likelihood: High* that Sky's plain dense numbers also tie.
- *Impact: Medium.* Expert teams have shipped native runtimes with this exact result, so "Sky is faster" must rest elsewhere. BaseRT also competes on short-prompt prefill and MoE.
- *Mitigation.* Compare against BaseRT where it runs. Lead with what it does not claim: exact speculation, long context, cold start, energy.
- *Kill criterion.* **G6, PIVOT.** The trigger is both of these:
  - every advantage Sky measures over mlx-lm falls inside what BaseRT already reports (≤ 1.05× plain decode, ≤ 1.44× short-prompt prefill, ≤ 1.33× small-model decode);
  - Sky's speculative mode is below 1.2× mlx-lm's own speculative mode.

  Then Sky is not differentiated as an engine. Contribute the verify kernels and exactness upstream, and drop the separate-engine acquisition plan.

**R-3. MLX's verify cost on the M5 Max is already modest.**
- *Evidence.*
  - Contributor data from an M5 Max, for a whole 70B 4-bit forward pass (mlx#3791, merged 2026-08-08): 75.9 ms at M = 1, 84.8 at M = 3, 112.7 at M = 5, 170.8 at M = 8. That is +12%, +48% and +125%: β ≈ 0.06, 0.12 and 0.18 per drafted token (derived).
  - In the same PR, MLX's speculative decoding reached 24.5 tok/s with a 0.6B draft and 3 drafted tokens, against about 13.2 tok/s plain. The ≈ 1.86× ratio is our derivation (R3-issues §E4).
  - On an M3 Max, mlx-lm reached 1.82× on Qwen2.5-32B with a 0.5B draft (R5 §9.4).
  - The "β ≈ 0.48" that SPINE §3 quoted from R8 §2.9 is an M2 Pro number: `qmv_wide` is not used for affine 4-bit on that GPU generation (R4 §7.D). SPINE §3 now gives β ≈ 0.5–1 for M1/M2-class GPUs and ≈ 0.06–0.18 for the M5 Max.
  - R8's model, S = τ / (1 + γc + γβ), with the published EAGLE-3 τ = 2.29 at 3 drafts and c = 0.03, gives ≈ 1.65× for an MLX-class verify (β ≈ 0.10) and ≈ 1.99× at Sky's target β = 0.02. The kernel is therefore worth ≈ 1.2× on the M5 Max ([Part XXVII](part27-moat.md) §3.2). `[EXPERIMENTAL arithmetic]`
- *Likelihood: High.* Derived from verified statements.
- *Impact: High.* The spine's bar of ≥ 1.2× over mlx-lm's own speculative mode with the same draft must come from more than the GEMV:
  - a faster draft step (c);
  - rejection-sampling acceptance at T > 0 (Σ min(p, q) instead of Σ p·q);
  - EAGLE-3 support, which mlx-lm lacks;
  - fused attention for 5–16 queries;
  - longer drafts, where MLX's cost rises;
  - no host synchronization per round.
- *Mitigation.*
  - Measure MLX's verify curve on the M5 Max before building ([Part IV](part04-hardware-limits.md) IV-T4).
  - Target M = 5–16, where MLX is weakest.
  - Implement rejection sampling and EAGLE-3 early.
- *Kill criterion.*
  - **G6, PIVOT.** Measure on the M5 Max, with the same draft model, the same prompts, at T = 0 and 0.6. If Sky's speculative mode is below 1.2× mlx-lm's speculative mode, follow R-2's pivot.
  - **G6, STOP.** If it is below 1.1×, stop presenting the speculative engine as a differentiator.

**R-4. The neural accelerators may not make 16–32-token verification cheap.**
- *Evidence.*
  - A one-row `matmul2d` is slower than a GEMV. MLX and llama.cpp route M below about 8–33 to ALU kernels (R6).
  - Apple's sample 64×32 tile reaches about half of peak; tiles of 64×64 or more are needed (R6 §A2).
  - For Qwen3-32B shapes at 13 ≤ M ≤ 32, MLX sends most projections to non-NAX split-K (R3-kernels §d).
  - The neural-accelerator free-verify width of 33–35 tokens assumes `matmul2d` is efficient at those widths ([Part IV](part04-hardware-limits.md) §8).
  - MTPLX's claimed 2.2–2.7× on an M5 Max, against a measured +24–28% on an M3 Max, is consistent with the neural accelerators helping. But the M5 claim is unverified (R8 §2.7).
- *Likelihood: Medium.*
- *Impact: High.* Two things depend on it: the spine's thesis that speculation turns the M5's idle matrix compute into tokens, and every plan for trees or block drafters that verify 16 tokens. Without the neural accelerators, the free width is about 8 on every chip.
- *Mitigation.* Measure `matmul2d` at M = 8, 16 and 32 on Qwen3-32B shapes first. Plan for 8 tokens as the base case.
- *Kill criterion.* **G5, or G1 if hardware arrives earlier, PIVOT.** If the best `matmul2d` verify kernel at M = 16 is not faster than the ALU verify kernel at M = 16:
  - plan verify widths of about 8 everywhere;
  - drop trees of 16–32 nodes and DFlash-class drafters from the plan;
  - remove "idle matrix compute" from the pitch.

### B. Competitors and Apple

**R-5. Monolith/mpk-apple is a direct competitor, with the MPK author behind it.**
- *Evidence.* `jiazhihao/mpk-apple` was opened on 2026-09-24 by the author of MPK and Mirage (R7 §7.1).
  - Its design: "the whole generation loop compiled into one GPU-resident static program of fused whole-GPU dispatches, replayed from an indirect command buffer with no CPU work on the critical path".
  - Its goals: "zero CPU↔GPU synchronizations per token"; ≥ 1.10× the better of MLX and llama.cpp; a stretch goal of ≥ 80% of the bandwidth bound.
  - Its status: milestone M0, with the 27B tests deferred for lack of RAM.
  - On NVIDIA, MPK took Qwen3-8B from 14.5 to 12.5 ms per token (1.16×) on an A100 (R7 §7.1).
- *Likelihood: High* that it ships a working Apple runtime within the year. `[LIKELY: an experienced author and a narrow design]`
- *Impact: High.* It occupies the same category, with academic credibility. Its stated goal is plain decode; it does not mention exact speculation.
- *Mitigation.* Track it monthly ([Part XXVII](part27-moat.md) XXVII-T8). Differentiate on exact speculation, long context, contracts and `sky explain`. Do not race on plain decode. Consider sharing the benchmark protocol.
- *Kill criterion.* **G5, PIVOT.** If Monolith, or any third party, publishes reproducible results that meet Sky's §3 targets on the same metrics on the M5 Max before Sky reaches G6, either collaborate or narrow Sky to what the other project does not cover.

**R-6. Core AI.**
- *Evidence.* Core AI (WWDC26; R6 §A3):
  - Apple describes it as "a brand new framework built right into the platform … designed to be the best way to bring and run models on device", and as "the inference framework powering on-device Apple Intelligence … fast inference across the CPU, GPU, and Neural Engine".
  - It converts PyTorch models (`coreai-torch`), compiles ahead of time, and specializes per device with a model cache.
  - It keeps "states" for in-place KV caches and accepts custom Metal 4 kernels (`TorchMetalKernel`).
  - It ships with Instruments support and a debugger, and targets OS 27.
  - One measurement shows it reaching about 65 TFLOPS fp16 on the M5 Max GPU, with no int8 × int8 path `[EXPERIMENTAL: one source]`.
  - Nobody we found has published its LLM decode or prefill speed.
- *Likelihood: Certain* that it exists. *Unknown* whether its LLM path is competitive.
- *Impact: Critical* for the Apple-acquisition thesis. Core AI is Apple's own model compiler and runtime with device specialization, the category Sky would sell. If it is competitive, Apple has no reason to buy. If it is not, Apple's Core AI team is the team that would fix it.
- *Mitigation.* Add Core AI as a third baseline as soon as an OS 27 benchmark host exists, at G0 if possible and at G5 at the latest. Position Sky where Core AI is not: exactness, contracts, `sky explain`, native MLX and GGUF files, a Rust library.
- *Kill criterion.* **G5.** Measure Qwen3-32B 4-bit, or the closest format Core AI supports, on the M5 Max:
  - **STOP the Apple-acquisition thesis** if Core AI is ≥ mlx-lm on decode and prefill **and** its speculative output matches its plain output. Pursue the other outcomes in [Part XXVII](part27-moat.md) §5.8.
  - **PIVOT** if it is ≥ mlx-lm without exact speculation: Sky's pitch to Apple narrows to exactness, contracts and explanation.

**R-7. Apple can copy any single kernel or trick in months.**
- *Evidence.*
  - MLX shipped M5 Pro/Max tunings nine days after the chips were announced, and "M5 ultra tunings" seven days after the M5 Ultra shipped (R3-issues §B3; R6 §A1).
  - Neural-accelerator support shipped with the base M5 (v0.30.0, 2025-11-19).
  - `qmv_wide`, a contributor kernel adapted from llama.cpp, merged on 2026-06-26.
  - GQA decode attention (1.08–1.31× on the M5 Max) merged in v0.32.2 (R3-issues §E2, §F3).
  - Apple's engineers wrote llama.cpp's M5 tensor kernel (#20962).
  - MLX releases about monthly (v0.32.0 to v0.32.3, July to September 2026).
- *Likelihood: High.*
- *Impact: High* for the moats ([Part XXVII](part27-moat.md) §3); *Medium* for the project.
- *Mitigation.* Compete on the combination, on proof and on adoption. Publish first. Make Sky's exactness tests the reference. Treat every lead as perishable.
- *Kill criterion.* **G6, PIVOT.** The trigger is MLX shipping both of these before Sky reaches G6:
  - verification within 1.3× of M = 1 up to M = 8 on the M5 Max;
  - speculative output identical to plain output.

  Then drop exact speculation as the headline and re-rank the differentiators. Continue only if another §3 metric holds a margin of at least 1.2×: long-context decode, cold TTFT or energy.

**R-8. MLX and llama.cpp iterate fast; llama.cpp with AI-assisted, crowd-sourced tuning.**
- *Evidence.*
  - ggerganov's 2026 Metal PRs disclose AI assistance, and #27370 came out of an AI-generated draft. The decode-attention tuning table covers 19 SKUs gathered in about two weeks (R5 §9.6).
  - On the same M5 Max, a newer llama.cpp build ran prefill 3.3× and decode 1.17× faster than an older one (R5 §14).
  - llama.cpp is attacking the same regime as Sky:
    - a 2-query verify attention kernel, 1.75× on the M5 Max (#23114);
    - speculative-decode peak 1.323× on the M4 Max (#26570);
    - an open PR claiming 1.84× → 2.27× (#27441, UNVERIFIED).
  - It already supports EAGLE-3, MTP, DFlash and DSpark drafting (R5 §7).
  - tinygrad credits an AI agent for its fastest Apple kernel (R2 §8.5).
- *Likelihood: Certain* that they iterate. *High* that some of Sky's leads close within weeks of publication.
- *Impact: High.* Benchmark leads age within a release.
- *Mitigation.* Re-measure against the latest releases at every gate. Pin versions. Keep the published results current. Lead with properties, not percentages.
- *Kill criterion.* **G8, CLAIM.** After re-baselining on the competitors' then-current releases, any headline metric that misses its §3 target is dropped from the acquisition material.

**R-9. The model landscape moved.**
- *Evidence.*
  - Qwen3.5/3.6/3.8-27B (February–August 2026) are hybrids: 48 Gated-DeltaNet linear-attention layers plus 16 full-attention layers, with an MTP head. They stream 14.4 GB per token, 20% less than Qwen3-32B, with a 42.6 tok/s roofline on the M5 Max and 4× less KV (R8 §1.7).
  - MTPLX, an MLX-based open-source tool, self-reports 65.2 tok/s at Qwen's sampling settings and 87.6 "optimized" for Qwen3.8-27B 4-bit on an M5 Max (UNVERIFIED). An independent M3 Max test measured +24–28% (R8 §2.7).
  - A public DSpark drafter for Qwen3.8-27B reports acceptance lengths of 4.4–5.7 on an H100 (R8 §2.3).
  - R8: a buyer evaluating "Qwen 32B-class on M5 Max" in late 2026 will run the 27B hybrid with MTP, not Qwen3-32B.
- *Likelihood: High.*
- *Impact: High.* Sky's Qwen3-32B "47–59 tok/s typical" could fall below an MLX-based tool's claim on the model buyers actually use.
- *Mitigation.*
  - SPINE D10 already makes the hybrids the second headline; pull them forward.
  - Reproduce MTPLX's claim on our M5 Max.
  - Build exact rollback of the DeltaNet state (R8 §3.5).
- *Kill criterion.*
  - **G5, PIVOT.** If MTPLX's ≥ 60 tok/s on Qwen3.6/3.8-27B reproduces under our protocol, Sky's speculative headline moves to that model and must beat that number, exactly.
  - **G7, CLAIM.** If Sky cannot run the hybrid models with exact rollback by G7, the speculative headline may not be presented as state of the art.

### C. Technical execution

**R-10. The M5 neural-accelerator path is fragile.**
- *Evidence.*
  - llama.cpp's tensor path was silently disabled on the M5 in some builds until #27461 (2026-09-01), because a probe failed and the Metal 4 language version was unset (R5 §9.5).
  - MLX's history:
    - Xcode 26.5 miscompiled the neural-accelerator GEMM (#3586);
    - split-K in bf16 produced garbage on the M5 Max (#3797);
    - `gather_qmm` corrupted results (#3887, #3856);
    - builds broke on the macOS 27 SDK (#4533, open; #4594);
    - split-K was routed to the wrong path for about five months after launch (#3422);
    - a downstream fork disabled the neural accelerators on the M5 entirely, citing wrong results (R3-kernels §d, R3-issues §B5).
  - A reverse-engineering study found eleven undocumented behaviors in the Metal 4.1 tensor path (Rigel, R7 §5.5).
  - MPP's lowering is opaque, and MSL 4.1 needs OS 27 (R6).
  - TF32 on by default changes fp32 results (#3534).
- *Likelihood: High* that Sky hits miscompiles or regressions.
- *Impact: High* for M5 prefill and for neural-accelerator verification. *Medium* overall, because ALU fallbacks exist.
- *Mitigation.*
  - Test every pipeline differentially against the reference interpreter, per OS and per Metal compiler (D6; R7 §12.2).
  - Key the tuning database by OS and compiler version.
  - Run probe kernels at startup, as llama.cpp does.
  - Keep ALU fallbacks, and file Feedback Assistant reports.
- *Kill criterion.* **G5, CLAIM.** If neural-accelerator kernels give wrong results on the pinned OS for Qwen3 shapes and no workaround appears within two weeks (a policy proposal), ship ALU-only on the M5 and withdraw the M5 prefill-parity claim. MLX would then lead prefill by the accelerators' factor ([Part IV](part04-hardware-limits.md) §5).

**R-11. Batch-invariant kernels may cost more than expected.**
- *Evidence.*
  - On NVIDIA serving stacks, batch invariance cost 1.62× in vLLM (Thinking Machines: Qwen3-8B, 26 s → 42 s) and an average of 34% in SGLang, ranging 24–55% (R8 §3.3).
  - The argument that batch 1 is cheaper (padding M wastes ALU work, not bandwidth) is untested on Apple GPUs.
  - At 4.5 bits per weight the M5 Max has about 7.5 ALU operations per streamed weight. Padding to 8 rows fits; padding to 32 may not (R8 §4.3).
  - R3's test kernel spilled registers from M = 4.
- *Likelihood: Medium.*
- *Impact: High.* Exactness is a pillar of the pitch. If invariance costs plain decode more than a few percent, Sky loses decode parity in exact mode.
- *Mitigation.*
  - Design the invariant family first ([Part XXIV](part24-roadmap-now.md) #3) and measure its cost per shape.
  - Keep a verify-and-replay fallback in the style of LLM-42 (R8 §3.5).
  - Allow `--numerics=relaxed` when exactness is not needed.
- *Kill criterion.*
  - **G1.** If the invariance cost at M = 1 on the large Qwen3-32B shapes exceeds 3% (R8 §11's target), redesign.
  - **G4, PIVOT.** If the end-to-end cost on 8B decode exceeds 5% (a policy proposal), make exactness an opt-in mode and remove "byte-identical at full speed" from the pitch.

**R-12. Speculative gains depend on the workload and on drafters per model.**
- *Evidence.*
  - The public Qwen3-32B EAGLE-3 drafter reaches acceptance lengths at T = 0.6 of 2.29 (HumanEval), 2.49 (GSM8K) and 2.15 (CNN/DM) with 3 drafts, and 3.08 on GSM8K with 7 (R8 §2.3).
  - AngelSlim's drafter reaches 1.86–2.01 with 2 drafts.
  - Acceptance in thinking mode is unmeasured.
  - Acceptance drops on high-entropy, creative text (R8 §2.7).
  - Prompt lookup gives about 1× on open-ended text (R8 §2.6), and llama.cpp's n-gram mode found no drafts in 256 tokens of fresh text (R5 §11).
  - No MLX-format EAGLE-3 or DFlash weights exist, and mlx-lm drops MTP tensors (R8 §2.3).
  - A 60+ tok/s average needs τ ≥ 3.5–4.5, a DFlash-class drafter that nobody has published for Qwen3-32B (R8 §2.9).
  - Modeled gains are 1.6–2.2× on code and math, 1.4–1.8× on chat, and about 1.2–1.4× on creative text (R8 §9).
- *Likelihood: Certain* that gains vary. *High* that a suite average is below the code-and-math number.
- *Impact: High.* "60 tok/s" is the pitch's most quoted number.
- *Mitigation.*
  - Use a declared workload suite, and always quote the workload, temperature and context.
  - Ship prompt lookup and suffix drafting for input-grounded work.
  - Convert the EAGLE-3 weights.
  - Train a drafter (G9) if compute becomes available.
- *Kill criterion.*
  - **G6, CLAIM.** On the M5 Max with EAGLE-3 at T = 0.6, if the suite average falls below 1.5× Sky's plain decode, restrict claims to the classes where ≥ 1.7× holds.
  - **G6, STOP.** If it falls below 1.2× on every class, stop the speculative headline.
  - **G8.** No "60 tok/s average" on the one-pager unless it was measured.

**R-13. The compiler may not pay for itself.**
- *Evidence.*
  - Every engine near the roofline on Apple GPUs uses hand-written kernels. tinygrad's generic code generator reaches 10–43% of bandwidth on 4-bit decode (R2 §0).
  - R7 expects autotuning to give 0–5% on bandwidth-bound GEMVs and 5–30% on small-M prefill GEMMs and attention (R7 §10.2).
  - Cranelift's production e-graph is worth about 0.1% (R7 §6).
  - Fusing small operations is worth 0.5–2.5% of MLX prefill (R3-issues §D8), and llama.cpp's fusion about 4% on 8B (R5).
  - The architecture already relies on hand-designed skeletons (D4). The risk is the general layers absorbing the schedule: the TIR, the scheduler, the partition search and the e-graph.
- *Likelihood: Medium.*
- *Impact: Medium.* Schedule, and the credibility of "a compiler" as opposed to "an engine".
- *Mitigation.* Build the skeletons first. Make `sky explain` attribute every gain to the layer that produced it. Keep the e-graph at G9.
- *Kill criterion.* **G5, PIVOT.** If the partition search and the autotuner improve the hot kernels by less than 5% over hand-picked defaults on small-M prefill and verify shapes (the low end of R7's range), freeze the general layers. Keep the skeletons and the static plan, and move the time to speculation and attention.

**R-14. OS and driver drift; wired-memory panics.**
- *Evidence.*
  - Compiling at run time ties results to the OS's Metal compiler ([Part VI](part06-architecture.md) §13).
  - macOS 27 changed residency decisions to per-`ResidencySet`. MLX's fix took one model from 0.46 to 13.19 tok/s (mlx#4211).
  - Kernel panics ("completeMemory() prepare count underflow") occur on M1–M5 when memory is wired; the issue has been open since 2026-03-01 and still reproduces on the macOS 27 beta (mlx#3186).
  - macOS unwires GPU memory after about 1 s idle: 26 ms per token became 429 ms after a pause (llama.cpp #10119).
  - A display-active watchdog kills long GPU work, and kernel panics under sustained load with the display on were reported on an M5 Max (R6).
- *Likelihood: High* that an OS update breaks or slows something during the plan.
- *Impact: Medium; High on demo day.*
- *Mitigation.* Pin the OS for benchmarks. Run CI on each OS beta. Keep a residency heartbeat, avoid one monolithic residency set, and never run long kernels.
- *Kill criterion.* **G8, CLAIM.** If the demo host shows a kernel panic or a wired-memory stall under the sustained protocol, no public demo until it runs cleanly three times in a row (a policy proposal).

### D. Measurement and credibility

**R-15. Quality disputes over "same quantization".**
- *Evidence.*
  - No lossless conversion exists between MLX's 4.5-bpw format and GGUF's 4.5-bpw types (R9 §0).
  - Files labeled "4-bit" differ: bartowski's Q4_0 is 4.67 bpw (a Q6_K output head, Q4_1 `ffn_down` tensors), Q4_K_M is 4.90, MLX is 4.50.
  - Q4_K_M changes the greedy token at 5.8% of positions (Mistral-7B; R8 §4.2), and bf16 against fp16 diverges on 49–100% of prompts (R8 §3.4).
  - llama.cpp uses f32 activations and an f16 KV cache, while mlx-lm uses bf16 (R9 §11).
  - Whether Sky's relayout cache counts as "the same file" is an open question (R9 §9).
- *Likelihood: High* that comparisons are challenged.
- *Impact: Medium–High.*
- *Mitigation.*
  - Follow D5: same-file pairs; layout changes only; dequantized values bit-equal by test.
  - Pass the correctness gates first, and publish KLD against BF16 and against the other engine.
  - Print bits per weight and bytes per token next to every speed.
  - Build every quantization ourselves from one pinned BF16 base, and publish the logit dumps.
- *Kill criterion.* **G3 and every later gate, CLAIM.** If, on identical weights, KLD(Sky ‖ other) > 0.1 × KLD(other ‖ BF16), or teacher-forced top-1 agreement is below 99%, no speed claim on that pair until it is fixed.

**R-16. Thermal throttling on laptops.**
- *Evidence.*
  - Laptop GPU clocks can drop to 0.84–1.52 GHz depending on power mode and fans. Sustained pp512 fell 19% in one llama.cpp measurement (R6; EXPERIMENTAL).
  - Secondary reports say a 14" M5 Max loses about 7.4% of sustained GPU performance in High Power Mode and about 25% in Automatic (R9 §6.4; LIKELY, unverified).
  - Decode is bandwidth-bound, and its sensitivity to clocks is unmeasured `[EXPERIMENTAL]`.
- *Likelihood: High* for prefill on a 14" laptop; *Unknown* for decode.
- *Impact: Medium.* Inconsistent numbers invite accusations of quoting bursts.
- *Mitigation.* Run 10-minute sustained, interleaved runs in High Power Mode on AC, stating the chassis ([Part XXIII](part23-benchmarks.md) §5.6). Use a Mac Studio host for headline numbers.
- *Kill criterion.* **G8, CLAIM** (a policy proposal):
  - if a metric's sustained lead (median of the last 5 of 10 minutes) is less than half its burst lead, publish only the sustained number;
  - if the sustained lead is below 1.05×, drop the claim.

**R-17. Accusations of benchmark gaming.**
- *Evidence.*
  - Sakana's claimed 150× speedup was a 3× slowdown, and KernelBench-style checks have been fooled into fake 50–120× results (R7 §12.1).
  - Gimlet's Metal speedups are measured against a weak baseline.
  - tinygrad's 195.7 tok/s claim does not state its configuration (R2 §8.5), and MetalRT reports the best of 5 runs (R7 §8.1).
  - Several standard measurements mislead (R9 §0, §6.4):
    - mmapped weights read only by the GPU are invisible to footprint and RSS;
    - llama-bench decodes from an empty KV cache;
    - mlx-lm counts the first token without its time;
    - llama-server caches prompts by default;
    - the idle gap changes TTFT by up to 8.8%.
- *Likelihood: High.* Any claim against MLX will be scrutinized by its maintainers and community.
- *Impact: Critical.* Credibility is the first thing an acquirer checks.
- *Mitigation.*
  - Follow the R9 §7 protocol: pre-registration, an external harness, correctness gates, confidence intervals ([Part XXIII](part23-benchmarks.md) §5).
  - Publish everything, and invite the maintainers to check our competitor settings.
  - Never show a projected number as measured (D13).
- *Kill criterion.*
  - **G8, CLAIM.** If an outside party cannot reproduce the headline within 5%, publish no claims.
  - **STOP the claim.** If a published claim is shown to be wrong, withdraw it publicly within days.

**R-18. Targets that are infeasible as written.**
- *Evidence.*
  - **(a) G5 decode.** An earlier spine set G5 at 32B decode ≥ 1.10× mlx-lm, the §3 *stretch* target; it now asks for ≥ 1.0× (≥ 1.05× only if the raw read is ≥ ≈ 575 GB/s). At 28.4 tok/s MLX moves ≈ 511 GB/s, so:
    - 1.10× needs ≈ 563 GB/s (92% of spec), above the only achievable-bandwidth measurement on the chip (≈ 530 GB/s);
    - 1.05× needs ≈ 537 GB/s at the bytes counted here; with Sky's kernel boundaries and streaming at 95% of the raw read, a raw read of ≈ 575 GB/s ([Part XVI](part16-decode.md) §2.1).

    `[arithmetic on R5 and R3-issues §A4]`
  - **(b) RAM.** "≤ 0.9× mlx-lm at 32K with fp16 KV" needs mlx-lm to use at least 3 GB more than Sky's floor of ≈ 27.5 GB ([Part IV](part04-hardware-limits.md) §6). mlx-lm's overhead is ≈ 1.4 GB on 8B at a 2K prompt (derived from R4 §1) and unmeasured at 32B and 32K `[EXPERIMENTAL]`.
  - **(c) Small models.** "≥ 1.5× mlx-lm on small models" exceeds BaseRT's 1.33× on 0.6B (R4, R7).
- *Likelihood: Medium–High* that at least one target is unreachable.
- *Impact: Medium.* Missing a mis-set target reads as failure.
- *Mitigation.* Re-baseline at G0 and G5, as the spine requires. Restate:
  - (a) as "≥ 95% of the raw-Metal read bandwidth measured at G0";
  - (b) as an absolute overhead above the physical floor;
  - (c) by keeping G4's ≥ 1.3×.
- *Kill criterion.* None for the project. At G5, any target that measurement shows to be infeasible is restated before G8, not "missed".

### E. Resources and focus

**R-19. M5 Max hardware access, and a 16 GB development Mac.**
- *Evidence.*
  - The development Mac is an M2 Pro with 16 GB and no neural accelerators (R1 Appendix A). It swapped 4.1 GB during this research (R9 §10), and benchmarks there shared the machine with agents at a load average near 4.8 (R1 Appendix B).
  - Qwen3-32B 4-bit needs ≥ 20 GB at 4K context ([Part IV](part04-hardware-limits.md) §6).
  - The 32-core M5 Max (36 GB, 460 GB/s) is not the headline chip (R6).
  - Monolith deferred its 27B tests for lack of RAM (R7): the same constraint.
- *Likelihood: Certain* that the hardware is needed. Whether it is obtained is the owner's decision (SPINE D14).
- *Impact: Critical.* Neither the headline workload nor the M5 thesis (R-3, R-4, R-10) can be measured without it.
- *Mitigation.* A 40-core M5 Max with 128 GB (SPINE D14) as an SSH benchmark host, preferably a Mac Studio for thermals. A self-hosted macOS CI runner.
- *Kill criterion.* **G0, STOP/PIVOT.** If access is not committed by the end of G0 (week 2):
  - suspend M5-specific and acquisition-oriented work;
  - re-plan as an M1–M4 project: prefill ≥ 1.25×, small models, long context, exactness;
  - revisit when access exists.

**R-20. One person plus AI agents against Apple's teams.**
- *Evidence.*
  - The Sky repository has one human author and 50 commits over three days, with AI co-author trailers `[VERIFIED: git log]`.
  - mlx-lm's maintainers: "At the moment we don't accept AI written architecture changes", and "the number of PRs is way beyond our capacity to review" (R4 §9).
  - MLX has at least four core maintainers (R3-issues). Apple also contributes through its ecosystem engineers (#20962) and the Core AI teams (R6).
  - llama.cpp pairs a maintainer with AI assistance and crowd-sourced tuning (R5 §9.6).
  - Apple's team sizes are not in our evidence.
- *Likelihood: High* that throughput is bounded by one person's review, decisions and hardware time, not by code generation.
- *Impact: High.* The schedule; a bus factor of one; and a discount that reviewers and acquirers apply to AI-written code `[LIKELY, from the mlx-lm statement]`.
- *Mitigation.* An oracle-gated agent workflow: the reference interpreter, differential tests and poisoned outputs (R7 §12.2). Strict scope. Collaborators recruited once G5 evidence exists. Open source.
- *Kill criterion* (a policy proposal):
  - **PIVOT** if G4 is not met by week 20 (six weeks late): cut G7 breadth to Qwen3 plus one other family.
  - **STOP** acquisition-oriented planning if G5 is not met by week 30, and continue as an open-source or research project.

**R-21. Scope creep between scientific Sky and inference.**
- *Evidence.*
  - Ten worktrees hold uncommitted scientific-Sky work, and three of them are needed for tensor authoring (D17, R1 §8).
  - The language lacks 12 features before a Qwen3 layer can be written in it (R1 §9.3b).
  - The CPU f64 programme (Roadmap A) does not move tokens per second (R10 §6).
- *Likelihood: High.*
- *Impact: High* (schedule).
- *Mitigation.* D17's freeze during G1–G4. The Rust builder first. Tensor types in the language only after G4.
- *Kill criterion.* **G4, PIVOT** (a policy proposal):
  - if more than 20% of non-bug-fix commits between G0 and G4 touched scientific-Sky crates, reset the freeze;
  - if language work sits on G4's critical path, defer it until after G6.

### F. Strategy and the acquisition

**R-22. Sky's language may be irrelevant to Apple's interest.**
- *Evidence.*
  - Apple's model path is PyTorch conversion into Core AI, with a Swift API, plus Python, C++ and Swift for MLX (R6 §A3).
  - Owning a language costs access to the PyTorch and Hugging Face ecosystem and to users who live in Python (R7 §4).
  - Sky's language cannot express a Qwen3 layer today (R1 §9.3b).
  - Model files are already short: 234 lines in mlx-lm and 159 in llama.cpp ([Part XXVII](part27-moat.md) §3.5).
- *Likelihood: High.* `[LIKELY]`
- *Impact: Medium.* The engine must stand alone, and the language may be left out of any transaction.
- *Mitigation.* The engine works without writing Sky (`sky run hf://`, the Rust builder). Keep the language as a separate asset. Add Python bindings only if adoption demands them.
- *Kill criterion.* **G8, PIVOT.** If evaluators or partners show interest only in the runtime, package the inference engine separately from the language.

**R-23. Acquisition dynamics.**
- *Evidence.*
  - Apple builds this layer in-house and in the open: MLX, Core AI, and contributions to llama.cpp (R3-issues, R6).
  - ExecuTorch now delegates Apple GPUs to MLX (R7 §8.3).
  - The only verified acquisition in our evidence is Qualcomm–Modular, completed 2026-07-29; its terms are not in our evidence (R7 §5.4).
  - Once released under Apache-2.0, Sky's code is available to anyone without a purchase ([Part XXVII](part27-moat.md) §5.7).
  - Apple-only scope narrows the set of buyers. Our research did not study Apple's acquisition history, and we make no claim about it.
- *Likelihood: Low* for an Apple acquisition at the owner's stated scale `[LIKELY: Apple's revealed preference in this layer is to build; Sky has no team or users today]`.
- *Impact: Critical* for the stated ambition; *Low* for the project's other outcomes.
- *Mitigation.* Pursue the outcomes in [Part XXVII](part27-moat.md) §5.8 in parallel: adoption, upstream partnership, other buyers. Build a team and users. Keep the evidence public.
- *Kill criterion.* **G8 plus six months, STOP optimizing for acquisition** if adoption is below the thresholds the owner sets before G8 and no potential acquirer or partner has made contact.

**R-24. IP and authorship questions block a transaction.**
- *Evidence.*
  - AI co-author trailers appear on most commits `[VERIFIED: git log]`.
  - mlx-lm declines AI-written architecture changes, a sign of reviewer concern (R4 §9).
  - Apache-2.0 §3 grants a patent license to recipients `[VERIFIED: LICENSE]`.
  - There is no `THIRD_PARTY_NOTICES.md` yet, and the plan learns from MIT-licensed kernels (D15).
- *Likelihood: Unknown.* It is a legal question.
- *Impact: High.* It could block or devalue a transaction.
- *Mitigation.* Bring in counsel before G8 publication ([Part XXVII](part27-moat.md) §5.6). Keep provenance records and notices, and require contributor sign-off.
- *Kill criterion.* **Before G8, PIVOT.** If counsel advises that the owner cannot give the ownership representations a transaction would need, drop the acquisition package and pursue open-source or partnership outcomes.

## 5. Kill criteria by gate

| Gate (week) | Measurement | Threshold | Action |
|---|---|---|---|
| G0 (2) | M5 Max access (40-core, 128 GB; SPINE D14) | not committed | **STOP/PIVOT** (R-19): suspend M5 and acquisition work; re-plan around M1–M4 |
| G0 (2) | Core AI baseline, if an OS 27 host exists | — | informs R-6 early |
| G1 (5) | Batch-invariant GEMV against a specialized GEMV at M = 1, 32B shapes, M2 Pro | cost > 3% | redesign (R-11) |
| G1 (5) | Verify cost on 32B shapes, M2 Pro | t(4)/t(1) > 1.45 (β > 0.15) after two kernel iterations | **PIVOT**: escalate to M5 Max measurement; at β = 0.15, EAGLE-3 gives ≤ 1.62× (R8 §2.9), below the ≥ 1.7× target |
| G1 (5) | Large-shape GEMV bandwidth (spine G1) | < 88% of measured raw read | gate fails; re-plan per [Part XXIV](part24-roadmap-now.md) #3 |
| G3 (9) | Correctness gates (R-15) | KLD or top-1 agreement fails | **CLAIM**: no speed claims |
| G4 (14) | 8B plain decode against mlx-lm, M2 Pro | < 1.0× | **PIVOT** (R-1): hand-write the decode path |
| G4 (14) | Batch-invariance cost, end-to-end 8B decode | > 5% | **PIVOT** (R-11): exactness becomes opt-in |
| G4 (week 20) | Schedule | G4 not met | **PIVOT** (R-20): cut G7 breadth |
| G4 (14) | Share of non-fix commits on scientific Sky, G0–G4 | > 20% | reset the freeze (R-21) |
| G5 (20) | 32B plain decode against mlx-lm, M5 Max | < 1.0× | **CLAIM** (R-1) |
| G5 (20) | `matmul2d` verify at M = 16 against the ALU verify at M = 16 | not faster | **PIVOT** (R-4): plan verify width ≈ 8 |
| G5 (20) | Neural-accelerator correctness on the pinned OS | wrong results, no workaround in 2 weeks | **CLAIM** (R-10): ALU-only; no M5 prefill-parity claim |
| G5 (20) | Autotuner and partition-search gain on small-M shapes | < 5% | **PIVOT** (R-13): freeze the general layers |
| G5 (20) | Core AI against mlx-lm on Qwen3-32B, OS 27 | ≥ on decode and prefill, with exact speculation | **STOP** the Apple thesis (R-6) |
| G5 (20) | Monolith, MTPLX or others, reproduced | meet Sky's §3 targets first | **PIVOT** (R-5, R-9) |
| G5 (week 30) | Schedule | G5 not met | **STOP** acquisition planning (R-20) |
| G6 (28) | Sky speculative ÷ mlx-lm speculative, same draft, M5 Max | < 1.2× (pivot); < 1.1× (stop) | **PIVOT/STOP** (R-2, R-3) |
| G6 (28) | Speculative ÷ Sky plain, declared suite, EAGLE-3, T = 0.6 | suite average < 1.5×; every class < 1.2× | **CLAIM / STOP** (R-12) |
| G6 (28) | MLX ships cheap verification and exactness first | both shipped | **PIVOT** (R-7) |
| G7 (40) | Hybrid models with exact rollback | not supported | **CLAIM** (R-9) |
| G8 (44) | Outside reproduction | outside 5% | **CLAIM**: no publication (R-17) |
| G8 (44) | Sustained against burst lead | sustained < 1.05× | **CLAIM** (R-16) |
| G8 (44) | Lead against competitors' then-current releases | below §3 targets | **CLAIM**: not in acquisition material (R-8) |
| Before G8 | Counsel on authorship | ownership cannot be warranted | **PIVOT** (R-24) |
| G8 + 6 months | Adoption and inbound interest | below the owner's thresholds | **STOP** optimizing for acquisition (R-23) |

## 6. What would make us wrong in the good direction

| If this turns out true | What we would observe | Where it shows |
|---|---|---|
| MLX's long-context decode is weak on the M5 Max | MLX's decode attention ran at 56% of peak with 8 KV heads at 32K (mlx#3837), and its GQA kernel starts only at 8K (mlx#4077). If our protocol confirms this, attention alone could give ≈ 1.2–1.5× at 32K–128K ([Part XV](part15-attention.md) §8.2). | G5 |
| The neural accelerators verify 16–32 tokens almost for free | `matmul2d` at M = 16–32 near the M = 1 time. Trees and block drafters become worthwhile, with 2.5–3× modeled for a DFlash-class drafter (R8 §9). | G5 |
| A strong Qwen3-32B drafter appears, or is cheap to train | Red Hat's `speculators` and SGLang's SpecForge are open (R8). With τ ≈ 4.5, the model gives 80–93 tok/s ([Part IV](part04-hardware-limits.md) §8). | G6–G9 |
| mlx-lm stays without EAGLE or MTP | community PRs stay closed for review capacity (R4 §9); Sky's exact EAGLE-3 path on Apple stays unique for longer | G6 |
| Core AI's LLM path is slow, or closed to MLX and GGUF formats | Apple's internal gap is larger than feared (R-6) | G5 |
| W8A8 int8 prefill is near-lossless and the int8 rate is ≈ 2× | Qwen3-8B lost only 0.7 MMLU points at W8A8 (R8 §4.2). 8-bit models gain 1.3–1.8× prefill `[EXPERIMENTAL]`. | G5 |
| The energy gains are large | mlx-lm's Python loop drew 7.1 W of CPU during 0.6B decode (R9 §6.3); a GPU-resident loop removes it, and speculation cuts DRAM traffic per token | G8 |
| Exactness becomes a requirement | evaluation reproducibility and audited deployments need it; the contracts become a selling point with customers ([Part XXVII](part27-moat.md) §3.3) | G8 and after |
| MLX keeps copying weights | cold TTFT ≥ 3× holds, because MLX declined mmap (mlx#4249) | G6 |
| Agents make one person as fast as a small team | G1–G6 met on schedule; Sky's history suggests code is not the bottleneck | G4–G6 |
| The M5 Ultra is the demo machine | plain 60 tok/s is within its planning ceiling of 59.6 ([Part IV](part04-hardware-limits.md) §4) | G8 |

## 7. Verdict

The project is worth doing, but as an engineering and research project whose most likely good outcome is an open, credible, Apple-specific inference engine with one property the incumbents lack. It is not worth doing as a bet on a single buyer.

It is worth doing under five conditions:
1. a 40-core M5 Max is in hand by G0;
2. the batch-invariant verify kernel meets β ≤ 0.15 on the M2 Pro by G1, and Sky's speculative mode beats mlx-lm's own by ≥ 1.2× on the M5 Max by G6, since everything distinctive rests on that kernel;
3. every claim follows the spine's honest targets and the R9 protocol, and none follows the owner's original table;
4. scientific Sky stays frozen through G4;
5. the engine is released openly early enough for outsiders to reproduce it by G8.

As an acquisition play it is a long shot. No technical moat survives more than months against Apple, which already owns MLX, Core AI, the compilers and the hardware. What an acquirer pays for is a demonstrated, reproducible lead plus a team and users, and none of those exists today. Proceed, measure early, honor the kill criteria above, and decide the acquisition question at G8 on evidence.

## 8. Implementation tasks

| ID | Task | Where | Deliverable | Acceptance test |
|---|---|---|---|---|
| XXVIII-T1 | A risk register reviewed at every gate | `docs/inference/risks.json` (proposed) | each R-n with its status, its measurement and the date checked | the gate review cannot close while any kill criterion due at that gate lacks a measurement |
| XXVIII-T2 | The M5 Max day-one kit at G1, not G5 | [Part IV](part04-hardware-limits.md) IV-T2 and IV-T4 | bandwidth, dispatch cost, `matmul2d` by tile and M, the MLX verify curve | results JSON committed by week 5 |
| XXVIII-T3 | Competitor watch | [Part XXVII](part27-moat.md) XXVII-T8 | monthly pinned re-measurement of MLX, llama.cpp, Monolith, Core AI and MTPLX | a monthly JSON; R-5 to R-9 updated |
| XXVIII-T4 | Adoption thresholds set by the owner | this part, R-23 | numbers fixed before G8 | committed before any public release |

## 9. Disagreement with the spine

1. **G5's decode exit criterion is a stretch target.** SPINE §6 sets G5 at "32B decode ≥ 1.10×", but §3 states ≥ 1.05× with 1.10× as a stretch. With MLX at 28.4 tok/s, about 96% of the only measured achievable bandwidth on the M5 Max, 1.10× needs ≈ 563 GB/s, above that measurement (R-18). We recommend:
   - G5 exits at ≥ 1.05×, or at "≥ 95% of the raw-Metal read bandwidth measured at G0";
   - a miss of 1.10× is never a kill criterion; only < 1.0× is (R-1).

   Resolved: the spine's G5 now asks for 32B decode ≥ 1.0× at short context, ≥ 1.05× only if the measured raw read is ≥ ≈ 575 GB/s, with ≥ 95% of the raw-Metal read as the bandwidth target (SPINE §3, §6).
2. **The first M5 Max measurement comes too late.** The spine places it at G5 (weeks 14–20). The thesis-critical numbers are:
   - the achievable bandwidth;
   - MLX's verify curve on the M5 Max;
   - `matmul2d` efficiency at M = 8–32.

   They decide R-3 and R-4 and can be measured in days ([Part IV](part04-hardware-limits.md) IV-T2, IV-T4). We recommend making them a G1 exit item; without an M5 Max by G1, R-19's kill criterion applies.

   Resolved: the spine now says M5 Max measurements begin as soon as the hardware exists, with G1 as the target: bandwidth, dispatch cost, the `matmul2d` rate and baselines (SPINE §6, G5).
3. **The spine's "MLX β ≈ 0.48" is an M2 Pro number.** SPINE §3's speculative row says MLX's verify costs "β ≈ 0.48 per extra token, at which EAGLE-3 gains nothing". On the M5 Max, the contributor data the spine cites in §2 (mlx#3791) imply β ≈ 0.06–0.18 (R-3). The bar "≥ 1.3× mlx-lm's own speculative mode" remains right, but the §3 cell should say "M2 Pro" so the pitch does not overstate MLX's weakness on the headline chip.

   Resolved: the spine now says β ≈ 0.5–1 on M1/M2-class GPUs and β ≈ 0.06–0.18 on the M5 Max (mlx#3791), puts Sky's verify-kernel advantage on the M5 at ≈ 1.2×, and sets the bar at ≥ 1.2× over mlx-lm's own speculation with the same draft, with exactness as the differentiator (SPINE §3).
