# Part XXV — The 12-month roadmap

This part is the plan for the first year. It is Roadmap B of the mission: run real small and medium LLMs end to end on Apple GPUs and compete with MLX. It continues [Part XXIV](part24-roadmap-now.md), which details the first ten implementations for weeks 0–14 (gates G0–G4), and runs to week 52.

For each gate, G0 to G8, it gives:
- the goal;
- the exit criteria, all measured;
- the deliverables;
- the implementation tasks the gate contains, by their chapter IDs;
- the dependencies;
- work packages sized for one AI agent each, with their interfaces;
- the hardware it needs;
- the kill or pivot criteria from [Part XXVIII](part28-risks.md).

Every task defined in Parts II–XXIII, XXVII and XXVIII is placed in a gate, in the CPU lane, or explicitly deferred (§10). The part also gives:
- the dependency graph and a Gantt chart;
- the infrastructure milestones;
- the CPU lane (Roadmap A);
- the performance gates;
- how progress is reported;
- when the ROADMAP.md amendments of [Part VI](part06-architecture.md) §9 land.

Years 2–4 are [Part XXVI](part26-roadmap-4-years.md).

Labels follow the report's convention: `[VERIFIED: source]`, `[LIKELY: reason]`, `[EXPERIMENTAL: what decides]`, `[MOONSHOT]`. A plan item is a commitment, not a claim, and carries no label. Every performance number below is a target from SPINE §3 or from a chapter, and it is `[EXPERIMENTAL]` until its gate measures it.

## 1. Summary

- **Four phases, nine gates.**
  - *Foundation, on the M2 Pro (G0–G4, weeks 0–14).* A measurement harness, the Metal backend, the tensor IR with its oracle, first tokens, then fused decode at parity with mlx-lm.
  - *The headline chip (G5, weeks 14–20).* Prefill on Metal 4 tensor ops, flash attention, the autotuner, and the decisive M5 Max measurements.
  - *The pitch and the breadth (G6–G7, weeks 20–40).* Exact speculative decoding and time to first token (TTFT); then more model families, both file formats, the hybrid models and a server.
  - *Proof (G8, weeks 30–44).* A public, reproducible benchmark on at least three chips, reproduced by an outside party.

  Weeks 44–52 hold the buffer, a re-baseline against the competitors' releases of that month, and the start of Part XXVI.
- **The critical path:**
  1. the harness;
  2. `sky-metal`;
  3. the quantized GEMV family;
  4. the unfused executor;
  5. fusion, the static plan and replay;
  6. M5 prefill and the neural-accelerator verify decision;
  7. verify attention, acceptance and the drafters;
  8. the G8 protocol runs.
- **What sets the pace is not code.** Three things gate the schedule more than code generation does [LIKELY: R-20; Part XXIII §4.6 puts one headline protocol run at about two days of machine time]:
  - a 40-core M5 Max with 128 GB in service by week 5;
  - the owner's review time;
  - machine time on that one host.
- **G5 (week 20) is the most consequential review.** Seven kill or pivot criteria are decided there: R-1, R-4, R-5, R-6, R-9, R-10 and R-13. G6 (week 28) decides whether speculation is the pitch: R-2, R-3, R-7 and R-12.
- **The owner's example performance gates are rewritten in §9:**
  - "fused elementwise ≤ 1.1× C" holds per element, not per whole process;
  - "matmul ≤ 1.2× Accelerate" needs fix F7 in the default mode;
  - "Sky runs a 1B LLM" and "7B ≥ MLX" stand, reworded to real models;
  - "1B ≥ 0.8× MLX" is the wrong bar;
  - for plain decode, "32B > MLX by 10%" is out of reach on the measured bandwidth, and "by 30%" is above the hardware ceiling. Both are replaced by gates on effective speculative throughput, long context, small models, TTFT, RAM and energy.
- **All 266 implementation tasks are placed (§10).** 253 fall in gates G0–G8 (G4 holds 52 and G5 48, the densest). Six are in the CPU lane, and seven are deferred to G9 and Part XXVI.

| Gate | Weeks | Months | Goal | Headline exit (measured) | Main kill or pivot criteria |
|---|---|---|---|---|---|
| G0 | 0–2 | M1 | ground truth | harness reproduces the published baselines within 5%; owner decisions recorded | R-19: no M5 Max committed → STOP/PIVOT |
| G1 | 1–5 | M1–M2 | Metal foundation | Q4 GEMV ≥ 88% of the raw read and ≥ mlx on every shape, bit-invariant for M = 1…16; M5 Max day-one kit committed | R-11: invariance cost > 3%; verify β > 0.15 |
| G2 | 1–6 | M1–M2 | TIR and oracle | Qwen3-0.6B logits equal HF fp32 within tolerance; dequantization bit-equal | none in Part XXVIII; a slip eats G4's slack |
| G3 | 5–9 | M2–M3 | first tokens | greedy output equals mlx-lm on 20 prompts × 128 tokens (near-ties explained); decode ≥ 0.7× | R-15: no speed claims until correctness gates pass |
| G4 | 9–14 | M3–M4 | parity on the M2 Pro | 8B decode ≥ 1.0×; 0.6B ≥ 1.3×; ≤ 6 kernels per layer; zero allocations per token | R-1, R-11, R-20 (latest week 20), R-21 |
| G5 | 14–20 | M4–M5 | lead on the M5 Max | 32B decode ≥ 1.0×; long context ≥ 1.05× / 1.10× / 1.2× at 8K / 16K / 32K; prefill ≥ 1.0× (M5), ≥ 1.25× (M1–M4); RAM ≤ mlx-lm | R-1, R-4, R-5, R-6, R-9, R-10, R-13, R-20 (latest week 30) |
| G6 | 20–28 | M5–M7 | exact speculation and TTFT | byte-identical speculation; ≥ 1.8× mlx-lm plain; ≥ 1.2× mlx-lm speculative; TTFT warm ≥ 1.15×, cold ≥ 3× | R-2, R-3, R-7, R-12 |
| G7 | 20–40 | M5–M10 | breadth | six families, hybrids, MoE, GGUF and a server, each with the correctness suite and ≥ parity | R-9: hybrid exact rollback |
| G8 | 30–44 | M7–M11 | proof | public `sky-evaluation`; ≥ 3 chips; energy ≤ 0.85×; outside reproduction within 5% | R-8, R-14, R-16, R-17, R-24 |

## 2. Calendar, assumptions and rules

### 2.1 Calendar

Week 0 is the week the owner accepts this report. The spine's gates are in relative weeks, re-planned after G0 (SPINE §6). Months have four or five weeks:

| Month | M1 | M2 | M3 | M4 | M5 | M6 | M7 | M8 | M9 | M10 | M11 | M12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Weeks | 0–4 | 5–8 | 9–13 | 14–17 | 18–21 | 22–26 | 27–30 | 31–34 | 35–39 | 40–43 | 44–47 | 48–52 |

Each gate has a target week, the end of its range in SPINE §6. G4 and G5 also have a latest acceptable week, from Part XXVIII R-20: G4 by week 20, G5 by week 30. Past those weeks the plan pivots instead of slipping.

### 2.2 Assumptions

| # | Assumption | If it fails |
|---|---|---|
| A1 | One owner directs AI agents, with no other engineer before G5 [VERIFIED: today's team, Part XXVII §5.8] | — (the plan is sized for it) |
| A2 | A 40-core M5 Max with 128 GB is reachable over SSH by week 5 (Part XXVIII §9 item 2; Part XXIII §14 item 3) | R-19: re-plan as an M1–M4 project (§7.1) |
| A3 | A self-hosted macOS runner, separate from the development laptop, runs Metal tests from week 5 (D14) | Metal tests run by hand on the development Mac, and G1's CI criterion fails |
| A4 | The owner can review about 8–12 work packages at a time [assumption; R-20 says review, not code generation, bounds throughput] | fewer packages in flight; the schedule stretches in proportion |
| A5 | Scientific Sky is frozen from G1 to G4, except for bug fixes and the two landings D17 allows (§2.3) | R-21's reset rule |
| A6 | The competitors do not close the openings before Sky measures them | the kill criteria of G5–G8 fire; that is their purpose (R-5, R-7, R-8) |

### 2.3 Scope rules

Part II §16 and Part XXVIII R-21 rely on this part to state these rules.

1. **The freeze.** From G1 to G4 (weeks 1–14) no new scientific-Sky feature lands, but bug fixes do. The exceptions are the D17 landings, in this order, each with full tests (Part II §13.5):
   - reductions (II-T3);
   - arrays you can build (II-T4).

   Sizes from variables lands only after it is re-based on `Poly` (II-T5), with the Sky-source front door in G7.
2. **The measured share.** At the G4 review, count the non-bug-fix commits between G0 and G4 that touched scientific-Sky crates. If they exceed 20%, the freeze is reset (R-21; a policy proposal).
3. **Language work.** The tensor language surface (Part VII §13) starts in G7's Sky-source lane at week 28, after G6's critical work. Language work never sits on G4's critical path (R-21).
4. **Dependency direction.** Tensor code never imports `sky_interp`, and no scientific crate depends on a tensor crate (Part II §16; D1). A CI test enforces this (VI-T1).
5. **Measured numbers only.** Nothing projected is shown as measured (D13). Every number on the dashboard (§11) has a run_id.
6. **Parked branches.** The seven worktrees that tensor authoring does not need stay parked, branches kept, until after G4 (D17).

### 2.4 How a gate closes

A gate closes at a review with four inputs:
1. **Exit metrics.** Every exit metric is measured under the protocol of [Part XXIII](part23-benchmarks.md) §5, with its run_id and confidence interval. The claim rules of Part XXIII §2 apply:
   - a target is met when the named CI bound clears it;
   - a median that clears the target while its CI straddles it is "not shown".
2. **Kill criteria.** Every kill criterion due at the gate is measured and entered in the risk register (XXVIII-T1). The review cannot close while one lacks a measurement.
3. **Tasks.** Every task of the gate is done, or explicitly moved with the reason.
4. **Re-plan.** The remaining gates are re-planned. A target that measurement shows to be infeasible is restated, not "missed" (R-18).

## 3. Dependency graph

```text
G0 ground truth (weeks 0–2)
  harness + baselines · roofline calibrated · M5 Max decision · ROADMAP PRs opened
     │
     ├────────────────────────────┬─────────────────────────────────┐
     ▼                            ▼                                 ▼
G1 Metal foundation (1–5)      G2 TIR + oracle (1–6)            M5 Max day-one kit (by week 5)
  sky-metal · qgemv            TIR · verifier · interpreter ·    bandwidth · dispatch · matmul2d by
  M = 1…16, bit-invariant ·    loader · .skyweights ·            tile · MLX verify curve ·
  macOS CI runner              Qwen3 builder + HF oracle         engineering baselines
     │                            │                                 │
     └─────────────┬──────────────┘                                 │
                   ▼                                                │
G3 first tokens (5–9)                                               │
  unfused executor · tokenizer · templates · correctness gates      │
                   ▼                                                │
G4 parity on the M2 Pro (9–14)                                      │
  optimizer + scheduler · fused decode · static memory plan ·       │
  GPU-resident loop · GQA attention · GPU sampler · exactness ·     │
  cost model · perf CI                                              │
                   ▼                                                ▼
G5 lead on the M5 Max (14–20) ◄── decides R-1, R-4, R-5, R-6, R-9, R-10, R-13
  MPP / neural-accelerator prefill · flash attention · autotuner v1 ·
  GGUF same-file pair · NA verify decision · M5 protocol baselines
       │                          │                              │
       ▼                          ▼                              ▼
G6 exact speculation + TTFT    G7 breadth (20–40)             hybrid lane (20–40)
  (20–28)                        Llama · Mistral · Gemma ·      Gated DeltaNet · MTP drafter ·
  verify attention · accept /    Phi · gpt-oss · Qwen3-MoE ·    exact state rollback
  commit / rewind · drafters ·   GGUF breadth · KV modes ·      (R-9: may become the
  EAGLE-3 · .skybin · prefix     server · Sky-source front      speculative headline)
  cache                          door
       │                          │                              │
       └──────────────────────────┴──────────────┬───────────────┘
                                                 ▼
G8 proof (30–44)
  public sky-evaluation · ≥ 3 chips · energy · "why faster" · paper draft · outside reproduction
                                                 ▼
weeks 44–52: re-baseline (R-8) · G8 + 6 months clock (R-23) · G9 spikes · year-2 plan (Part XXVI)

critical path:  harness → sky-metal → qgemv → unfused executor → fusion + plan + replay
                → M5 prefill and NA verify decision → verify attention + acceptance + EAGLE-3
                → G6 exit → G8 protocol runs
```

Two edges are easy to miss:
- **G5 needs GGUF.** G5's prefill target "≥ 1.0× llama.cpp" can only be measured on the GGUF file (D5: same-file pairs). So the GGUF reader and the Q4_0, Q8_0, Q4_K and Q6_K codecs move from G7 into G5 (§16 item 2).
- **G6's trees need G5's decision.** Trees wider than 8 tokens, and DFlash-class drafters, need the neural-accelerator verify tier. Its existence is decided at G5 (R-4; XXI-T2).

## 4. Gantt chart

```text
month                             M1   M2  M3   M4  M5  M6   M7  M8  M9   M10 M11 M12
week                              0    5    10   15   20   25   30   35   40   45   50
                                  |    |    |    |    |    |    |    |    |    |    |
G0 ground truth                   ███
G1 Metal foundation                █████
G2 TIR + oracle                    ██████
G3 first tokens                        █████
G4 parity on the M2 Pro                    ██████     ┊
G5 lead on the M5 Max                           ███████         ┊
G6 exact speculation + TTFT                           █████████
G7 breadth                                            █████████████████████
   hybrid lane (Qwen3.6/3.8 + MTP)                    █████████████████████
   GGUF and model families                              ███████████████████
   server                                                       ███████████
   Sky-source front door                                      █████████████
G8 proof                                                        ███████████████
re-baseline, decisions, G9 starts                                             ░░░░░░░░░
Roadmap A (CPU), low priority     ██            ░░░░░░░░░░░░░░░░░░░░░░░░░░░

M5 Max host (40-core, 128 GB)       D  ▲
macOS CI runner (Metal tests)          ▲
perf-regression runner                          ▲
sky-evaluation repository           p                 R                       P
OS 27 host (Core AI baseline)                         ▲
wall-power meter                                          ▲
M4 Max (third chip, 128 GB)                                     ▲
gate reviews (kill criteria)        ◆  ◆◆  ◆    ◆     ◆       ◆ ◇         ◆   ◆
```

Legend:
- `█` planned work; `░` low priority or buffer.
- `┊` latest acceptable exit (R-20): week 20 for G4, week 30 for G5.
- `D` decision; `▲` in service.
- `p` private repository; `R` protocol runs begin; `P` public.
- `◆` gate review with its kill criteria; `◇` the week-30 schedule check for G5.

Part XXIV's chart gives the week-by-week order of items #1–#10 inside G0–G4.

## 5. Month by month

Months M1–M3 summarize [Part XXIV](part24-roadmap-now.md). From M4 on, this part is the plan of record.

| Month | Weeks | Work in progress | Exits and decisions | Measured that month |
|---|---|---|---|---|
| M1 | 0–4 | G0; Part XXIV items #1, #2, #4, #5, #7 start in parallel; the M5 host and the CI runner ordered and set up; ROADMAP PRs opened (§12) | G0 exit (week 2): baselines; owner decisions (R-19) | M2 Pro baselines (engineering grade); R6's microbenchmarks reproduced |
| M2 | 5–8 | #3 finishes; #6; #8 starts; the M5 Max day-one kit runs | G1 exit (week 5); G2 exit (week 6) | GEMV bandwidth, invariance cost and verify β on the M2 Pro; on the M5 Max: raw read, dispatch costs, `matmul2d` by tile and M, MLX's verify curve |
| M3 | 9–13 | #9 and #10; the cost model; perf CI | G3 exit (week 9) | greedy agreement; correctness gates; decode ≥ 0.7× |
| M4 | 14–17 | G5: MPP GEMM, tile loaders, flash prefill, the GGUF pair, the autotuner | G4 exit (week 14) | 8B parity; 0.6B ≥ 1.3×; end-to-end invariance cost |
| M5 | 18–21 | G5 finishes; G6 and G7 start at week 20 | G5 exit (week 20): R-1, R-4, R-5, R-6, R-9, R-10, R-13 | M5 Max protocol baselines; the Core AI baseline; the MTPLX reproduction |
| M6 | 22–26 | G6: verify attention, accept and commit, drafters, EAGLE-3, TTFT. G7: the hybrid lane; GGUF breadth; the Llama and Mistral builders | — | acceptance τ; draft cost c; TTFT components |
| M7 | 27–30 | G6 finishes; G7: Gemma 3, Phi-4, gpt-oss; the Sky-source lane starts (week 28); G8 starts (week 30) | G6 exit (week 28): R-2, R-3, R-7, R-12; week-30 check (G5's latest week) | S9 (speculation), S5 (cold start), S10 (repeated prefix) |
| M8 | 31–34 | G7: MoE, KV modes, server; G8: M4 Max runs, energy | — | energy (E6); Tier A on the M4 Max |
| M9 | 35–39 | G7 finishes; G8: Tier H on the M5 Max; the "why faster" report; the paper draft | — | Tier H; sustained thermal runs (S4) |
| M10 | 40–43 | G8: the reproduction kit goes to an outside party; demo rehearsal; counsel (R-24) | G7 exit (week 40): R-9 | outside reproduction |
| M11 | 44–47 | publication; adoption tracking begins | G8 exit (week 44): R-8, R-14, R-16, R-17 | re-baseline against the competitors' current releases |
| M12 | 48–52 | buffer; the first G9 spikes (Part XXVI §3); the year-2 plan | the G8 + 6 months clock runs (R-23) | competitor watch |

## 6. Infrastructure milestones

| # | Milestone | Needed by | Why | Done when | Tasks | Owner's decision |
|---|---|---|---|---|---|---|
| I1 | **M5 Max, 40-core GPU, 128 GB, reachable over SSH** | decision by week 2 (G0); in service by week 5 (G1) | the headline chip. The thesis-critical numbers can be measured in days (Part XXVIII §9 item 2): achievable bandwidth, MLX's verify curve, `matmul2d` at M = 8–32. 128 GB is needed for 32B at 128K, 70B at 32K and the BF16 reference logits (Part XXIII §14 item 3) | day-one kit committed (IV-T2, IV-T4); `doctor` passes on the host | IV-T2, IV-T3, IV-T4, XXVIII-T2 | purchase; a Mac Studio is preferred for thermals (R-16, R-19) |
| I2 | **Self-hosted macOS CI runner** for Metal tests | week 5 (G1) | Metal correctness on every PR. The current runner is Linux (Part II) | a deliberately broken kernel fails CI; `cargo test -p sky-metal` runs on the runner | VI-T2, II-T7 | a machine separate from the development laptop. We recommend a GPU family other than M5 (Apple7–9): CI then covers the ALU path on every PR, while the M5 host covers the neural accelerators nightly (XXI-T9; XX-T7 tests both Apple8 and Apple10) |
| I3 | **Performance-regression runner** | week 14 (G4) | exact counters gate PRs; timings gate nightly runs. Correctness jobs on the same machine make timings noisier (Part XXIII §12.1) | a PR adding one dispatch per layer fails `perf-pr`; an injected 5% slowdown is flagged within 3 nights; ≤ 1 false alarm in two weeks of A/A runs | XXIII-T14 | a second, dedicated Mac (recommended), or a nightly window on the CI runner with correctness jobs paused |
| I4 | **`sky-evaluation` repository** | private at week 2 (G0); protocol runs from week 20 (G5); public at week 44 (G8) | the harness must not depend on trusting Sky's code (D13) | G0: the `doctor`, `fetch` and `build` phases with lockfiles. G8: an outside party reproduces the headline within 5% | XXIII-T5, XXIII-T18 | when to publish the methodology; Part XXVII §5.7 recommends early |
| I5 | **OS 27 host** for the Core AI baseline | G0 if possible; week 20 at the latest | R-6 is decided at G5 | Qwen3-32B, or the closest format Core AI supports, measured on the M5 Max under OS 27 | XXVII-T8, XXVIII-T3 | a second boot volume on the M5 host, or the second machine of I3 |
| I6 | **Wall-power meter** | week 24 (before G6's energy run, E6) | whole-system energy. `PSTR` is unverified on the M5 Max (Part XXIII §16 item 6) | `PSTR` and the meter agree, or the meter becomes mandatory | XX-T12, V-T9 | purchase |
| I7 | **M4 Max, 40-core GPU, 128 GB** | week 30 (G8 starts) | the third chip: no neural accelerators, GPU generation ≥ 15 (Part XXIII §4.4) | a Tier A run committed | XXIII-T18 | purchase or access |
| I8 | **More than 1 TB of storage** on the benchmark host | week 20 | Tier H and Tier A disk use [LIKELY: Part XXIII §17] | `benchmark.sh --dry-run` fits | XXIII-T15 | — |

Operating rules for the M5 host (XXV-T3):
- a reservation calendar for protocol runs;
- a pinned macOS build for benchmarks;
- `doctor` before every protocol run, and nothing else running during one;
- nightly performance jobs only in unreserved time.

When the repository becomes public, self-hosted runners must not run workflows from forks without approval [LIKELY: GitHub's documented guidance for self-hosted runners on public repositories].

## 7. The gates

### 7.0 Work packages

A **work package (WP)** is the unit one AI agent takes. The rules:
1. **One owner module.** A WP has one crate or module, a frozen input interface and a frozen output interface, named in its table: a Rust type, a file format or a JSON schema.
2. **An acceptance test that decides "done".** It is the chapter task's own acceptance test, run in CI or by a script on the benchmark host, with no human judgment needed. Correctness acceptance always uses an oracle, the mitigation of R-20:
   - the TIR reference interpreter;
   - bit-equality against `mx.dequantize` or stored HF logits;
   - golden files;
   - poisoned outputs.
3. **Bounded size.** About two weeks of agent work at most [assumption]. Larger work is split.
4. **Hardware time is booked.** A WP whose acceptance needs the M5 Max books it through the reservation calendar. It is not done until its result JSON, with a run_id, is committed.
5. **The owner reviews interfaces and tests,** before work starts and before merge. Not every line is reviewed [policy proposal].
6. **Concurrency.** The plan assumes 8–12 WPs in flight (assumption A4). Past that, more agents add rework, not speed [LIKELY: R-20].

**Interface freeze points.** Agents work in parallel safely only against frozen interfaces:

| When | Frozen | Defined in |
|---|---|---|
| week 2 (G0) | results JSON schema `sky-bench/1.1`; `skycount` counters; competitor pins | Part XXIII §7, §11.4; V-T1 |
| week 1 (G1 start) | the kernel ABI (one `constant` argument struct per kernel); `QFormat` and `QPacking`; the `Graph`, `Type` and `Dim` API | Part XI §3.6; Part XII §2.2; Part VII §3–§4 |
| week 6 (G2 exit) | TIR text format; `.skyweights` v1; `ModelSpec` | Part VII §9; Part XII §5.2; Part XIII §4 |
| week 9 (G4 start) | `OptimizedTir` and `FusionLegality`; `ScheduledPlan`; `StepState`; `DecodeProgram` | Part VIII §6.3; Part IX §1; Part XIII §11.1; Part XX §3 |
| week 14 (G5 start) | `sky-tune` GPU record keys; `gpumodel.v1` | Part X §6.1; Part XXII §4 |
| week 20 (G6 start) | `Drafter`, `VerifyProgram`, the acceptance-kernel interface | Part XIII §12; Part XXI §6–§7 |

### 7.1 G0 — Ground truth (weeks 0–2)

**Goal.** Nothing in this report is a result until it is measured with a protocol a hostile reviewer accepts (Part XXIV #1). G0 builds that protocol, reproduces the published baselines, calibrates the roofline, and records the owner's decisions.

**Exit criteria.**

| Criterion | Threshold | Source |
|---|---|---|
| The harness reproduces the published M2 Pro baselines | within 5%: mlx-lm Qwen3-8B 4-bit at 37–39 tok/s and Qwen3-0.6B at about 267 tok/s; llama.cpp Qwen3-8B Q4_0 at 34.8 tok/s (tg128) and 312 tok/s (pp512) [VERIFIED: R4, R5] | Part XXIV #1 |
| Results | schema-valid JSON (`sky-bench/1.1`); a second run on another day agrees within the reported CI | Part XXIV #1 |
| Metric arithmetic | a fake engine with a known rate checks every metric; the memory sampler sees a 1 GiB mmapped file read by the GPU as wired memory, not as footprint | XXIII-T6, XXIII-T7 |
| Microbenchmarks M1 and M2 | best read within 3% of 196.3 GB/s; dependent dispatch 1.1–1.25 µs [VERIFIED: R6]. At G0 they run from `research/microbench/`; they move into `sky-metal` at G1 | XXIII-T1, XXIII-T2 |
| Census | `skycount` reproduces mlx-lm's 693 dispatches, 457 barriers and 23 command buffers per 8B token [VERIFIED: R4] | XXIII-T11, V-T2 |
| Physics | `roofline.py` reproduces Part IV; per-file bytes per decode token equal the files' tensor sizes | IV-T1, XXIII-T20 |
| Owner decisions | recorded: the M5 Max (40-core, 128 GB); the CI runner; the worktrees committed; the `inference` numerics contract accepted; the openness intent (Part I) | R-19; D14; D17 |

G0 numbers from the development Mac are engineering numbers unless the machine is freshly rebooted, quiet and passes `doctor` (Part XXIII §14 item 5). It ran with 4.1 GB of swap during the research [VERIFIED: R9 §10].

**Deliverables.**
- The private `sky-evaluation` repository: drivers, samplers, gates, statistics, lockfiles.
- Baseline JSON; `skycount`; `roofline.py` with per-kernel functions.
- The risk register.
- The ROADMAP PRs, opened (§12).
- The ten worktrees, committed to their branches.

**Tasks (19).** IV-T1; V-T1, V-T2; XXIII-T1, XXIII-T3, XXIII-T5 (private; public at G8), XXIII-T6, XXIII-T7, XXIII-T9, XXIII-T11, XXIII-T20; XIX-T11; II-T1, II-T2, II-T6; III-T1, III-T8 (filing the issues only); VI-T4 (PRs opened); XXVIII-T1.

**Dependencies.** None. This is Part XXIV #1.

**Work packages.**

| WP | Tasks | Consumes → produces | Done when |
|---|---|---|---|
| G0-a harness core | XXIII-T5, XXIII-T6, XXIII-T9, V-T1 | pinned engines, prompt token ids → per-token timestamps, `sky-bench/1.1` JSON, bootstrap CIs | the fake-engine test passes; baselines within 5% |
| G0-b samplers and gates | XXIII-T7, XIX-T11 | system counters, SMC `PSTR` → memory, power and thermal series; the quiet-machine gate | R9's memprobe table reproduced; the null-engine floor recorded |
| G0-c microbenchmarks | XXIII-T1, XXIII-T3 | R6's sources, MLX and llama.cpp ops → `sky-microbench/1` JSON | Part XXIII §3.6's anchors reproduced |
| G0-d census and physics | XXIII-T11, V-T2, XXIII-T20, IV-T1 | engine processes; safetensors and GGUF headers → dispatch counts; bytes per token | 693 / 457 / 23 reproduced; parameter totals match the files |
| G0-e repository hygiene | II-T1, II-T2, II-T6, III-T1, III-T8, XXVIII-T1 | worktrees, `sky-bench compare` → commits, per-unit `RESULTS.md`, issues, the risk register | each task's acceptance test |
| G0-f owner decisions | VI-T4 (PRs), hardware orders | this report → a decisions log | recorded before week 2 ends |

**Hardware.** The M2 Pro, rebooted and quiet. Orders placed for I1 and I2.

**Kill and pivot criteria.**

| Measurement | Threshold | Action |
|---|---|---|
| M5 Max access (40-core; ≥ 48 GB in R-19; 128 GB planned) | not committed by week 2 | **STOP/PIVOT** (R-19). Suspend M5 and acquisition-oriented work. Re-plan as an M1–M4 project: prefill ≥ 1.25×, small models, long context, exactness. Revisit when access exists |
| Core AI baseline | if an OS 27 host already exists | informs R-6 early |

### 7.2 G1 — Metal foundation (weeks 1–5)

**Goal.** Rust drives Metal. The quantized GEMV family streams close to the raw read and is batch-invariant by construction. The M5 Max's thesis-critical numbers exist.

**Exit criteria.**

| Criterion | Threshold | Source |
|---|---|---|
| `sky-metal` microbenchmarks | reproduce R6 within 10%: read ≥ 190 GB/s, copy ≈ 183 GB/s, dependent dispatch ≈ 1.2 µs, independent ≈ 0.14 µs | Part XXIV #2 |
| CI | `cargo test -p sky-metal` runs on the macOS runner (I2) | VI-T2 |
| Q4 GEMV at M = 1, M2 Pro | ≥ mlx `quantized_matmul` on every Qwen3-8B and 32B shape; ≥ 88% of the 196 GB/s raw read on the large shapes (target ≥ 95%, 186 GB/s; XVI-T1); k/v_proj ≥ 150 GB/s (mlx: 114 GB/s [VERIFIED: R3]) | spine G1; Part XXIV #3 |
| Batch invariance | rows of M = 2…16 bit-identical to the M = 1 result; invariance cost at M = 1 on 32B shapes ≤ 3% | Part XXIV #3; R-11 |
| Verify cost, M2 Pro, 32B shapes | t(4)/t(1) ≤ 1.45 (β ≤ 0.15); t(8)/t(1) ≤ 1.5 | Part XXVIII §5; Part XXIV #3 |
| Dequantization | bit-exact against `mx.dequantize` on real Qwen3-8B tensors | Part XXIV #3 |
| M5 Max day-one kit, committed by week 5 | raw read, copy and write bandwidth; dependent and independent dispatch costs; `matmul2d` TFLOPS by tile and dtype, int8 included; sustained clocks over 10 minutes; `PSTR`; MLX's verify curve for M = 1…32 on 32B shapes; first engineering baselines | IV-T2, IV-T4, IV-T3; XXVIII-T2 |

**Deliverables.**
- `sky-metal`: device, buffers, libraries, pipelines, encoders, timing, a residency stub.
- The `qgemv` and verify-GEMV skeletons for MLX affine 4-bit, group 64.
- Calibration records.
- The M5 Max results JSON, with `roofline.py`'s M5 Max entry moved from LIKELY to VERIFIED (IV-T2).
- The macOS CI job; notices, sign-off and the private-API lint in CI.

**Tasks (29).** VI-T1, VI-T2, II-T7; XI-T1, XI-T2, XI-T3, XI-T6 (Metal 3 path; the Metal 4 path is G5), XI-T10, XI-T11, XI-T16; XVI-T1; XII-T7 (MLX affine codecs; the GGUF codecs are G5); XXI-T1; X-T1; XXII-T2; XXIII-T2 (M1–M3; M4–M10 follow in G4–G6); IV-T2, IV-T3 (engineering run; protocol re-run at G5), IV-T4; V-T4, V-T10, V-T11; XXVII-T1, XXVII-T2, XXVII-T6, XXVII-T8; XXVIII-T2, XXVIII-T3; II-T3 (a D17 landing, low priority).

**Dependencies.** G0's harness and the microbenchmark sources; I1; I2.

**Work packages.**

| WP | Tasks | Consumes → produces | Done when |
|---|---|---|---|
| G1-a `sky-metal` core (#2) | XI-T1, XI-T2, XI-T3, XI-T6 (Metal 3), XI-T16, VI-T1 | `objc2-metal` → `Device`, `Buffer::{shared, no_copy}`, `Library`, `Pipeline`, the encoder API, the one-struct kernel ABI (Part XI §3.6) | R6 reproduced within 10%; buffer, alignment and pipeline-cache tests pass |
| G1-b calibration | XXIII-T2 (M1–M3), X-T1, XXII-T2 | `sky-metal` → `GpuFacts`, the GPU fingerprint, `gpumodel.v1` records | t_dep within 10% of R6; BW_stream within 3% of R6's GEMV |
| G1-c decode GEMV (#3) | XVI-T1, XI-T10, XII-T7 | the kernel ABI, `QFormat` → the `qgemv` MSL template; `sky bench kernel qgemv` | ≥ 88% (target 95%) of the raw read; ≥ mlx on every shape |
| G1-d verify family (#3) | XI-T11, XXI-T1 | G1-c's template → M = 2…16 instances that share one reduction schedule | bitwise M-invariance on 1,000 fuzzed shapes and on real weights; the β criterion |
| G1-e M5 Max day-one kit | IV-T2, IV-T3, IV-T4, V-T4, V-T11, XXVIII-T2 | `run_m5.sh`, the harness → results JSON with run_ids | committed by week 5 |
| G1-f CI and provenance | VI-T2, II-T7, XXVII-T1, XXVII-T2, XXVII-T6 | workflows → Metal CI, the notices check, the DCO check, the private-API lint | each check fails on a deliberate violation |
| G1-g competitor watch | V-T10, XXVII-T8, XXVIII-T3 | pins → weekly and monthly JSON series | four consecutive weekly runs archived |
| G1-h reductions landing | II-T3 | the reductions worktree → a PR | Linux and macOS CI green; `engines_agree` holds |

**Hardware.** The M2 Pro; I1 and I2 in service.

**Kill and pivot criteria.**

| Measurement | Threshold | Action |
|---|---|---|
| Batch-invariant GEMV against a specialized GEMV at M = 1, 32B shapes, M2 Pro | cost > 3% | redesign (R-11) |
| Verify cost on 32B shapes, M2 Pro | t(4)/t(1) > 1.45 after two kernel iterations | **PIVOT**: escalate to the M5 Max measurement. At β = 0.15, EAGLE-3 gives ≤ 1.62×, below the 1.7× target (R-3; R8 §2.9) |
| Large-shape GEMV bandwidth | < 88% of the measured raw read | the gate fails; re-plan per Part XXIV #3 (keep the original layout; split the verify templates) |
| M5 Max in service | not by week 5 | R-19 applies (Part XXVIII §9 item 2) |
| `matmul2d` verify at M = 16 against the ALU verify at M = 16, if measured early on the M5 host | not faster | informs R-4 early; decided at G5 |

### 7.3 G2 — TIR and the oracle (weeks 1–6)

**Goal.** Tensors exist in Sky: the tensor IR (TIR), its verifier, its text format and its reference interpreter. Quantized weights load bit-exactly. Qwen3 is built as TIR and matches Hugging Face.

**Exit criteria.**

| Criterion | Threshold | Source |
|---|---|---|
| Op set | 100% verified and interpreted; text round-trip on every test graph; a differential property test against NumPy (dev-only) | Part XXIV #4 |
| Interpreter determinism | bit-identical results for 1, 2 and 8 threads | VII-T7 |
| Qwen3-0.6B oracle | logits for 8 fixed prompts match HF transformers (fp32, CPU) within the tolerance set by bf16 weight storage; greedy agreement for 32 steps, except documented near-ties | Part XXIV #6 |
| Same quantization (D5) | for every tensor of `mlx-community/Qwen3-8B-4bit` and `Qwen3-0.6B-4bit`, Sky's dequantized values are bit-equal to `mx.dequantize`: 0 mismatches | XII-T6 |
| `.skyweights` | every plane 16 KiB-aligned; atomic rename; a killed build leaves no file | XII-T5 |

**Deliverables.**
- `sky-tir`: graph, types, ops, verifier, text format, interpreter, composites, the counter-based RNG.
- `sky-quant` formats.
- `sky-hub`'s URI grammar and cache.
- Config normalization.
- The safetensors loader with MLX quantization inference.
- The relayout cache.
- The Qwen3 builder, with reference logits stored by sha256.

**Tasks (22).** II-T8; VII-T1, VII-T2, VII-T3, VII-T4, VII-T5, VII-T6, VII-T7, VII-T8, VII-T10, VII-T11; XII-T1, XII-T2, XII-T3, XII-T5, XII-T6; XIII-T1, XIII-T4, XIII-T5, XIII-T6, XIII-T7; II-T4 (a D17 landing, low priority).

**Dependencies.** II-T8 (extract `Poly`, share `Explanation`) comes before VII-T1 and VII-T2. The interpreter needs no GPU.

**Work packages.**

| WP | Tasks | Consumes → produces | Done when |
|---|---|---|---|
| G2-a TIR data model (#4) | II-T8, VII-T1, VII-T2, VII-T3 | `Poly`, `Explanation` → `Module`, `Graph`, `Type`, `Dim`, `SymTable`; the op table | interning and `prove_le` property tests; one passing and one failing case per type rule |
| G2-b views, verifier, text | VII-T4, VII-T5, VII-T6 | G2-a → index functions, verifier rules V1–V24, canonical text | a fixture per rule; round trip; 60 s of fuzzing per PR |
| G2-c interpreter and composites | VII-T7, VII-T8, VII-T10 | G2-a → `interp::run` in `Mode::F32` and `Mode::F64`; `rms_norm`, `rope`, `softmax`, `attention`, GLUs, Philox | known-answer tests; composites equal their decompositions bit for bit |
| G2-d formats | XII-T1, XII-T2 | → `QFormat`, `QPacking`, `dequant_ref`, type rules QT1–QT8 | golden vectors from `mx.dequantize` and ggml, bit-exact |
| G2-e names, configs, weights (#5) | XIII-T1, XIII-T4, XIII-T6, XIII-T7, XII-T3, XII-T5, XII-T6 | `hf://` → a snapshot path; `config.json` → `ModelSpec`; safetensors → `.skyweights` v1 | the same-quantization proof; the 31-config normalization test |
| G2-f Qwen3 builder (#6) | VII-T11, XIII-T5 | `ModelSpec` → `ModelGraph`: prefill with a symbolic T, decode with T = 1 | the 0.6B oracle criterion |
| G2-g arrays landing | II-T4 | the arrays worktree → a PR | Part II's acceptance test |

**Hardware.** The M2 Pro's CPU. HF transformers reference logits are generated once and stored by sha256.

**Kill and pivot criteria.** Part XXVIII sets none at G2. A slip moves G3 and eats the six weeks of slack before R-20's week-20 limit for G4. The review checks two things:
- the interpreter's speed on 0.6B and 8B, which decides which oracle tests run per PR and which nightly (VII-M4);
- the count of reused lines against Part II's 10–20% estimate (Part II §16).

### 7.4 G3 — First tokens (weeks 5–9)

**Goal.** `sky run hf://mlx-community/Qwen3-8B-4bit "…"` produces correct text on Metal, unfused. Every correctness gate passes before any speed number is shown.

**Exit criteria.**

| Criterion | Threshold | Source |
|---|---|---|
| Tokenizer and templates | the R9 golden-id corpus (168 of 168) and the 30 template cases are byte-identical [VERIFIED baseline: R9 §4.2, §9] | Part XXIV #7 |
| Greedy agreement | equals mlx-lm for 128 tokens on 20 prompts; every divergence explained by a measured near-tie | spine G3 |
| Correctness gates, MLX file pair | KLD(Sky ‖ mlx-lm) ≤ 0.1 × KLD(mlx-lm ‖ BF16); teacher-forced top-1 agreement ≥ 99% | D13; R-15 |
| Decode | Qwen3-8B ≥ 0.7× mlx-lm on the M2 Pro, unfused, with a host wait per token | spine G3 |
| `sky bench llm` | its Sky medians agree with `sky-evaluation`'s within the CI | XXIII-T10 |

**Deliverables.** The tokenizer, templates, stop set and streaming detokenizer; downloads with provenance; no-copy weight views with residency; the optimizer pipeline v0; the unfused executor and `sky run`; the quality harness.

**Tasks (17).** XIII-T2, XIII-T3, XIII-T9, XIII-T10, XIII-T11, XIII-T13, XIII-T17; XIV-T1; XI-T4, XI-T5, XI-T7; VIII-T1, VIII-T2, VIII-T3; XII-T10; XXIII-T8, XXIII-T10.

**Dependencies.** G1-c (the GEMV), G2-e (the weights), G2-f (the builder). The critical path runs through G3-e.

**Work packages.**

| WP | Tasks | Consumes → produces | Done when |
|---|---|---|---|
| G3-a tokenizer and templates (#7) | XIII-T9, XIII-T10, XIII-T11, XIII-T13 | `tokenizer.json`, chat templates → `Tokenizer`, `StopSet`, `BosPolicy`, `StreamDetokenizer` | 168 of 168 ids; 30 of 30 templates; the detokenized stream equals `decode` |
| G3-b downloads and provenance | XIII-T2, XIII-T3 | `hf://` → files with sha256; `Provenance`, `sky-models.lock` | a pinned snapshot makes zero network calls (mock server) |
| G3-c weights on the GPU | XI-T4, XI-T5 | `.skyweights` → no-copy views; the residency `Keeper` | RSS does not grow by the file size; the first token after a 2 s idle is within 10% of steady state |
| G3-d optimizer pipeline v0 | VIII-T1, VIII-T2, VIII-T3 | TIR → canonical TIR (`canon`, `fold`, `cse`, `views`); rules compiled from `.tirr` files | goldens per pass; interpreter results unchanged |
| G3-e unfused executor and CLI (#8) | XI-T7, XIV-T1, XIII-T17 | optimized TIR → one kernel per op (the MSL emitter), memory plan v0, KV slabs, loop v0 | greedy agreement; ≥ 0.7× |
| G3-f quality gates | XII-T10, XXIII-T8, XXIII-T10 | logit dumps from every engine → KLD, top-1 and Δp reports | two BF16 references agree to a KLD of about 1e-4 or less, or the pre-registered fallback is reported |

**Hardware.** The M2 Pro and the CI runner. First Qwen3-32B tokens on the M5 host are a stretch goal, not an exit: the M2 Pro cannot hold that model.

**Kill and pivot criteria.**

| Measurement | Threshold | Action |
|---|---|---|
| Correctness gates (R-15) | KLD or top-1 agreement fails | **CLAIM**: no speed claim on that file pair until fixed. This rule holds at every later gate |

### 7.5 G4 — Parity on the M2 Pro (weeks 9–14; latest week 20)

**Goal.** Fused decode with a static memory plan, a GPU-resident loop and GPU sampling: Qwen3-8B at or above mlx-lm, the 0.6B well above it, and the exactness suite green.

**Exit criteria.**

| Criterion | Threshold | Source |
|---|---|---|
| 8B decode, ≤ 1K context | ≥ 1.0× mlx-lm (aim 1.05–1.10×; modeled 1.09–1.11×, Part XVI §2.2) | spine G4 |
| 0.6B decode | ≥ 1.3× mlx-lm | spine G4 |
| 4K-context decode, 8B | ≥ 1.05× mlx-lm (stretch 1.10×); Part XV's per-context bounds met within 3% | Part XXIV #10 |
| Dispatches | ≤ 6 × L + 3 per token (mlx-lm: 693 on 8B) | XVI-T6 |
| Allocation | zero allocations per token (counting allocator); `currentAllocatedSize` constant over 10,000 tokens | XIX-T5 |
| Replay | 1,000 replayed tokens equal 1,000 single-step submissions bit for bit; 256 greedy tokens with K = 4 in flight equal K = 1 | Part XXIV #9; XX-T2 |
| Invariance | the logits of position p are identical for M = 1…8 and for chunk sizes 16…2048 (0.6B and 8B); end-to-end invariance cost on 8B decode ≤ 5% | IX-T7; R-11 |
| Seeded sampling | the same seed gives the same tokens across runs; χ² test against the CPU reference | XIII-T12 |
| Explain | `sky explain` lists every kernel with its bytes, FLOPs, and predicted and measured time | Part XXIV #9; XXII-T5 |
| Model residual | Part XXII's prediction of the full 8B step within 5%, or a term is added before any M5 projection is quoted | Part XXII §12 |
| Performance CI | the exact-counter gate runs on PRs (I3) | XXIII-T14 |

SPINE §6 also lists "prefill ≥ 1.0× mlx-lm" at G4. This part measures prefill at G4 but gates it at G5, where the GEMM skeletons land (§16 item 1).

**Deliverables.**
- The optimizer passes for decode and the scheduler.
- Generated prologues and epilogues; the memory planner.
- The GPU-resident decode loop; GQA decode attention; the GPU sampler.
- The invariance checker.
- The `inference` numerics text for the spec.
- The cost model with `sky explain` and `sky profile`.
- Performance CI.

**Tasks (52).** VI-T5, VI-T6; VII-T12; VIII-T4, VIII-T5 (its D7 leg runs at G7), VIII-T6, VIII-T7, VIII-T8, VIII-T9, VIII-T10; IX-T1, IX-T2, IX-T3, IX-T4, IX-T5, IX-T7, IX-T9, IX-T10; XI-T14, XI-T15; XIII-T12, XIII-T14; XIV-T2; XV-T1, XV-T2; XVI-T2, XVI-T3, XVI-T4, XVI-T5, XVI-T6, XVI-T7, XVI-T8; XIX-T1, XIX-T2, XIX-T3, XIX-T4, XIX-T5; XX-T1, XX-T2, XX-T3, XX-T4, XX-T5, XX-T6, XX-T11; XXII-T1, XXII-T3, XXII-T4, XXII-T5; XXIII-T4, XXIII-T12, XXIII-T13 (the static report; the "why faster" report is G8), XXIII-T14. V-T3 starts here with the mlx-lm #1470 reproduction and completes at G6.

G4 is the densest gate. Four tasks may trail into the first two weeks of G5 without blocking G4's exit, because no exit criterion depends on them: XXIII-T12 (`sky profile`), XI-T15, XXII-T4 (validation harness) and VIII-T10 (the optimizer-stage explain golden).

**Dependencies.** G3's exit; G2-c's composites; G1-d's verify family, for the invariance tests.

**Work packages.**

| WP | Tasks | Consumes → produces | Done when |
|---|---|---|---|
| G4-a optimizer for decode | VIII-T4, VIII-T5, VIII-T6, VIII-T7, VIII-T8, VIII-T9, VIII-T10 | canonical TIR → `OptimizedTir` and `FusionLegality` (Part VIII §6.3); the relayout plan; the contract log | the Qwen3 decode plan gives groups G1–G5 of Part VIII §6.2; plans bit-identical across specializations |
| G4-b scheduler (#9) | IX-T1, IX-T2, IX-T3, IX-T4, IX-T5, IX-T9, IX-T10, VII-T12 | `OptimizedTir` → `ScheduledPlan` (K1–K5 plus K2c), barriers, plan cache | a Qwen3 block yields exactly K1–K5 and K2c; greedy partition equals exhaustive |
| G4-c generated prologues and epilogues (#9) | XVI-T2, XVI-T3, XVI-T4, XVI-T5, XIV-T2 | skeleton hooks → norm prologue; residual, SwiGLU and Σx² epilogues; QK-norm, RoPE and KV append | layer outputs within tolerance; 4 fewer dispatches per layer |
| G4-d memory planner (#9) | XIX-T1, XIX-T2, XIX-T3, XIX-T4, XIX-T5 | lifetime intervals → arena offsets; `verify_plan`; poison mode | efficiency ≥ 0.95; property test against brute force |
| G4-e GPU-resident loop (#9) | XX-T1, XX-T2, XX-T3, XX-T4, XX-T5, XX-T6, XIII-T14, XVI-T6 | `ScheduledPlan` → `DecodeProgram`, `StepState`, K steps in flight, shared events, the token ring | median GPU gap between steps < 2 µs; ≤ 6 × L + 3 dispatches |
| G4-f decode attention (#10) | XV-T1, XV-T2, XI-T14 | KV slabs → `flash_decode` with a fixed split size, `attn_combine` | ≥ 85% of read bandwidth at 4K–32K on 8B shapes; invariance |
| G4-g sampling and exactness (#10) | XIII-T12, XVI-T7, IX-T7, VI-T5 | logits → the keyed-Gumbel sampler; the invariance checker; the `inference` contract's spec text | the invariance and sampling criteria |
| G4-h small and draft models | XVI-T8, XX-T11 | → a plan for 0.6B and 1.7B shapes | 0.6B ≥ 1.3× |
| G4-i cost model and explain | XXII-T1, XXII-T3, XXII-T4, XXII-T5, VI-T6, XXIII-T4, XXIII-T12, XXIII-T13, XI-T15 | plans → `KernelCost`, `predict_plan`, `sky explain`, `sky profile`, the closure check | closure error e ≤ 5% for Sky on 8B |
| G4-j performance CI | XXIII-T14, with XXIII-T2's M6–M8 | explain JSON → golden counter files; nightly timings | I3's acceptance test |

**Hardware.** The M2 Pro; the CI runner; the perf runner (I3); the M5 host for early fused 32B runs (not an exit).

**Kill and pivot criteria.**

| Measurement | Threshold | Action |
|---|---|---|
| 8B plain decode against mlx-lm, M2 Pro | < 1.0× | **PIVOT** (R-1): stop generating decode kernels and hand-write the decode path before going further |
| End-to-end invariance cost, 8B decode | > 5% | **PIVOT** (R-11): exactness becomes opt-in; remove "byte-identical at full speed" from the pitch |
| Schedule | G4 not met by week 20 | **PIVOT** (R-20): cut G7's breadth to Qwen3 plus one other family |
| Non-fix commits on scientific crates, G0–G4 | > 20% | reset the freeze (R-21) |

### 7.6 G5 — Lead on the M5 Max (weeks 14–20; latest week 30)

**Goal.** Move to the headline chip. Build prefill on MPP `matmul2d` (the neural accelerators on the M5), flash attention, the autotuner and the GGUF same-file pair. Take the decisive measurements, and run the largest set of kill criteria in the plan.

**Exit criteria** (on the M5 Max unless stated).

| Criterion | Threshold | Source |
|---|---|---|
| Decode GEMV streaming | ≥ 90% of achievable bandwidth; the commitment is ≥ 95% of the raw read measured at G1 | spine G5; Part XVI §1 |
| Qwen3-32B plain decode, ≤ 1K context | ≥ 1.0× mlx-lm; ≥ 1.05× only if the measured raw read is ≥ about 575 GB/s | spine §3 |
| Qwen3-32B long-context decode, 16-bit KV in the baseline's dtype | ≥ 1.05× at 8K, ≥ 1.10× at 16K, ≥ 1.2× at 32K; the bounds are about 1.08×, 1.14× and 1.24× at MLX's measured KV-read efficiency | spine §3; Part XV §13 |
| Prefill, M5 | ≥ 1.0× mlx-lm; headline at prompts ≥ 2,048 tokens | spine §3 |
| Prefill, M1–M4 (M2 Pro, Qwen3-8B) | ≥ 1.25× mlx-lm on the MLX file **and** ≥ 1.0× llama.cpp on the GGUF file. Below 1.1× the M1–M4 prefill claim is withdrawn (Part XVII §11) | spine §3 |
| Peak RAM | ≤ mlx-lm, by the system-wide metric | spine G5 |
| Small models | Qwen3-0.6B ≥ 1.3× mlx-lm (target 1.5×) | XVI-T8; R-18(c) |
| Verify cost, weight GEMVs | M ≤ 8 at ≤ 1.3× the M = 1 time on 32B shapes | spine §3 |
| Neural-accelerator verify decision | measured and recorded: `matmul2d` verify at M = 16 against the ALU verify at M = 16; whether the NA family streams at ≥ 95% of the raw read at M = 1 | XXI-T2, IX-T8; R-4 |
| Autotuner | three tunings of one instance agree within 3%; gain over defaults reported per kernel class | Part X §14 |
| Model residual | Part XXII's model against the measured full 32B step, reported; above 5%, a term is added before projections are quoted | Part XXII §12 |
| Competitors | Core AI baseline, if an OS 27 host exists (I5); MTPLX's and Monolith's claims reproduced or recorded as not reproducible | R-5, R-6, R-9; V-T11 |

**Deliverables.**
- `mpp_qgemm` with dequantized group-scaled tiles, and the `half × int4b` variant; the `simdgroup_matrix` fallback.
- The five-kernel prefill layer, with chunk pipelining and the first token sampled inside the last chunk.
- `flash_prefill` for M1–M4 and for the neural accelerators.
- The neural-accelerator decode and verify decision record.
- GGUF loading and the GGUF codecs.
- Autotuner v1, with `-O1`, `-O2` and `-O3`.
- The residency policy; memory contracts with byte units.
- The Metal 4 and ICB command paths, measured.
- The M5 Max protocol baselines (a Tier H subset).

**Tasks (48).** XVII-T1, XVII-T2, XVII-T3, XVII-T4, XVII-T5, XVII-T6, XVII-T7, XVII-T8, XVII-T9; XVIII-T6; XI-T9, XI-T12, XI-T13, XI-T17 (and XI-T6's Metal 4 path); XII-T4, XII-T11 (and XII-T7's GGUF codecs); XIII-T8; XV-T5, XV-T6, XV-T7, XV-T10, XV-T12; XVI-T9, XVI-T10, XVI-T11; X-T2, X-T3, X-T4, X-T5, X-T6, X-T7, X-T8; IX-T6, IX-T8; XIX-T7, XIX-T8, XIX-T9, XIX-T10; XX-T7, XX-T9; XXI-T2; XXII-T6; V-T5, V-T6, V-T7; XXIII-T15, XXIII-T16; XXVII-T7. IV-T3 is re-run here under the full protocol.

Tasks that may trail into the first two weeks of G6 without blocking G5's exit: V-T5 (the routing-cliff sweep), XXIII-T16 (rewound depth), XV-T10 (attention tuning entries), XVI-T11 (the decode cost contract) and IX-T6 (the split-combine variants).

**Dependencies.**
- G4's exit.
- I1, with protocol storage (I8); I5 for R-6.
- XXI-T2 needs XI-T12's `verify_mpp` probe.
- The GGUF pair is needed for the llama.cpp criterion (D5).

**Work packages.**

| WP | Tasks | Consumes → produces | Done when |
|---|---|---|---|
| G5-a MPP GEMM | XVII-T1, XI-T12 | the kernel ABI → `mpp_qgemm` fp16 with its tile family; the `verify_mpp` probe | M2 Pro ≥ 5.2 TFLOPS at 4096³; M5 Max ≥ 80% of the neural-accelerator peak measured at G1, on Qwen3-32B MLP shapes |
| G5-b quantized tiles | XVII-T2, XVII-T3, XII-T11 | codecs → dequantized fp16 tiles (MLX affine, Q4_0, Q8_0, Q4_K, Q6_K); `half × int4b` with a per-group post-scale | tiles bit-equal to `sky-quant`'s reference; ≥ 0.9× the fp16 rate |
| G5-c fallback GEMM | XVII-T4, XI-T13 | → `sg_qgemm` | ≥ 85% of peak on the M2 Pro at 4096³; never selected on GPU generation ≥ 17 |
| G5-d prefill program | XVII-T5, XVII-T6, XVII-T7, XVIII-T6, XI-T17 | prefill buckets of the `ScheduledPlan` → five kernels per layer; back-to-back chunks; the LM head and sampling in the last chunk; the watchdog budget | KV and last-row logits bit-identical for chunk sizes 64…4,096; < 5 µs gaps between chunks; TTFT at 128 tokens on 8B ≥ 1.2× mlx-lm |
| G5-e attention on the M5, and prefill attention | XV-T5, XV-T6, XV-T7, IX-T8, XXI-T2 | KV slabs → `flash_prefill` (M1–M4 and NA); NA decode and verify variants; the decision record | XV-T5's (a) and (b); XXI-T2's two records |
| G5-f GGUF same-file pair | XII-T4, XIII-T8, XVI-T9, XII-T7's GGUF codecs | GGUF → mapped views; Q4_0, Q8_0, Q4_K, Q6_K codecs | on the M2 Pro, Q4_0 ≥ 177 GB/s and Q4_K ≥ 165 GB/s on R5's shape (llama.cpp: 177 and 147 [VERIFIED: R5]) |
| G5-g autotuner v1 | X-T2, X-T3, X-T4, X-T5, X-T6, X-T7, X-T8, XV-T10, IX-T6 | the plan's open parameters → `sky-tune` GPU records | three tunings agree within 3%; pruning keeps the true winner in ≥ 95% of instances |
| G5-h memory, residency, contracts | XIX-T7, XIX-T8, XIX-T9, XIX-T10, XXII-T6, XVI-T11 | plans → residency sets; `sky explain --memory`; `bytes_per_token` contracts with byte units | `sky check` proves `bytes_per_token <= 18.2 GB` for Qwen3-32B and refutes 17.9 GB with a witness |
| G5-i command paths on the M5 | XX-T7, XX-T9, XI-T6's Metal 4 path, XI-T9 | `DecodeProgram` → the CPU-encoded ICB path; the Metal 4 submitter; the compiler self-check | identical tokens on every path; experiments E1–E2 of Part XX §10 recorded |
| G5-j M5 protocol and comparisons | IV-T3, XV-T12, XVI-T10, XVII-T8, XVII-T9, V-T5, V-T6, V-T7, XXIII-T15, XXIII-T16, XXVII-T7 | the harness → the Tier H subset; long-context and prefill matrices; routing cliffs; the int8 rate; MLX's attention efficiency | committed JSON with run_ids, including the Core AI and MTPLX rows |

**Hardware.** In G5 the M5 Max is the bottleneck: protocol runs, autotuning and decision measurements compete for it [LIKELY: Part XXIII §17]. A second M5 Max (I3 or I5) removes the conflict.

**Kill and pivot criteria** (Part XXVIII §5).

| Measurement | Threshold | Action |
|---|---|---|
| 32B plain decode against mlx-lm, M5 Max | < 1.0× | **CLAIM** (R-1): no material may claim a decode advantage; the project continues only if R-3 and R-12 pass |
| `matmul2d` verify at M = 16 against the ALU verify at M = 16 | not faster | **PIVOT** (R-4): plan verify widths of about 8; drop 16–32-node trees and DFlash-class drafters; remove "idle matrix compute" from the pitch |
| Neural-accelerator correctness on the pinned OS | wrong results, with no workaround in two weeks | **CLAIM** (R-10): ALU-only on the M5; no M5 prefill-parity claim |
| Autotuner and partition-search gain on small-M prefill and verify shapes | < 5% over hand-picked defaults | **PIVOT** (R-13): freeze the general layers; keep the skeletons and the static plan |
| Core AI against mlx-lm on Qwen3-32B, OS 27 | ≥ mlx-lm on decode and prefill, with exact speculation | **STOP** the Apple-acquisition thesis (R-6). Without exact speculation: **PIVOT**, the pitch to Apple narrows to exactness, contracts and explanation |
| Monolith, MTPLX or others, reproduced under our protocol | meet Sky's §3 targets first | **PIVOT** (R-5): collaborate, or narrow Sky. If MTPLX's ≥ 60 tok/s reproduces, the speculative headline moves to Qwen3.6/3.8-27B and must beat that number exactly (R-9) |
| Schedule | G5 not met by week 30 | **STOP** acquisition-oriented planning; continue as an open-source or research project (R-20) |
| A §3 target shown infeasible | — | restated before G8, not "missed" (R-18) |

### 7.7 G6 — Exact speculation and TTFT (weeks 20–28)

**Goal.** Ship the pitch: exact speculative decoding with draft models, prompt lookup and EAGLE-3, byte-identical to plain decoding, in rounds that never wait for the CPU. Also time to first token, with an ahead-of-time `.skybin`, a prefix cache and residency.

**Exit criteria** (M5 Max, Qwen3-32B 4-bit, unless stated).

| Criterion | Threshold | Source |
|---|---|---|
| Exactness | speculative output byte-identical to Sky's plain output at T = 0 and at T = 0.7 with a seed, on 100% of prompts, for every drafter; on the CI runner per PR and on the M5 Max nightly | D10; XXI-T9; XXIII-T17 |
| Against mlx-lm plain | effective throughput ≥ 1.8× | spine §3 |
| Against mlx-lm's own speculative mode, same 0.6B draft, same prompts, T = 0 and 0.6 | ≥ 1.2× (spine). Below 1.2× is a PIVOT, below 1.1× a STOP (Part XXVIII R-2, R-3) | spine §3; R-2, R-3 |
| EAGLE-3 against Sky plain, declared suite, T = 0.6 | ≥ 1.7× on each workload class claimed; suite average ≥ 1.5× | Part XXI; R-12 |
| Prompt lookup | ≥ 2× on summarization and code editing (greedy); never below 0.97× on fresh text | XXI-T7 |
| TTFT, warm | ≥ 1.15× better, prompt length stated | spine §3 |
| TTFT, cold (process start to first token, page cache warm, AOT `.skybin`) | ≥ 3× better | spine §3 |
| TTFT, repeated prefix, both prefix caches on | ≥ 1.3× better (about 2× at ≤ 8 new tokens) | spine §3 |
| Acceptance study | τ per drafter, workload, temperature and thinking mode, published internally | XXI-T15 |

**Deliverables.**
- Verify attention for chains and trees; GPU accept, commit and rewind kernels; the speculation runtime with pipelined rounds.
- Three drafters: a draft model; prompt lookup and suffix drafting; EAGLE-3, converted from the RedHat and AngelSlim checkpoints. Trees too, if R-4 passed.
- The exactness CI suite.
- For TTFT: the start-up trace, parallel start-up, progressive residency, pipelines without compiling, `.skybin`, the plan cache and the prefix cache.
- The acceptance study; the first energy numbers.

**Tasks (35).** XXI-T3, XXI-T4, XXI-T5, XXI-T6, XXI-T7, XXI-T8, XXI-T9, XXI-T10 (only if R-4 passed), XXI-T11, XXI-T15; XV-T3; XIV-T3, XIV-T7 (with trees), XIV-T8; XVIII-T1, XVIII-T2, XVIII-T3, XVIII-T4, XVIII-T5, XVIII-T7, XVIII-T8; XIII-T15, XIII-T16; XI-T8; X-T9; XX-T8, XX-T10, XX-T12; XXII-T7; XIX-T12; V-T3, V-T8; XXIII-T17; XXVII-T4, XXVIII-T4.

**Dependencies.**
- G4's invariance suite and decode loop.
- G5's verify decision (XXI-T2, IX-T8) and its neural-accelerator attention (XV-T5).
- The public EAGLE-3 weights (D10).
- I6, for energy.

**Work packages.**

| WP | Tasks | Consumes → produces | Done when |
|---|---|---|---|
| G6-a verify attention | XXI-T3, XV-T3 | `flash_decode` → up to 32 query rows; chain and tree masks; per-node window gather; NA tiles at long context | per-row output bit-equal to decode attention; ≤ 1.1× of q = 1 at 32K for q ≤ 8 on the NA path |
| G6-b accept, commit, rewind | XXI-T4, XXI-T5, XIV-T3, XIV-T7 | logits and drafts → GPU acceptance (keyed Gumbel; rejection mode), KV length counters, tree-path compaction | self-draft acceptance is 100%; KV bytes equal plain decode's after n rounds |
| G6-c speculation runtime | XIII-T15, XXI-T11, XX-T10, XX-T8 | `Drafter`, `VerifyProgram` → rounds with no host sync; ICB ranges for variable draft steps | GPU gap between rounds < 5 µs; output with a random drafter byte-identical to plain |
| G6-d draft models and lookup | XXI-T6, XXI-T7, XIX-T12 | 0.6B and 1.7B static plans; CPU n-gram and suffix-tree drafters | draft cost c ≤ 0.035 for 0.6B against a 32B target; prompt-lookup criterion |
| G6-e EAGLE-3 | XXI-T8 | RedHat and AngelSlim checkpoints → Sky's drafter format and runtime | τ within 5% of the RedHat card (bf16 drafter); ≥ 1.7× plain at β ≤ 0.05 |
| G6-f trees and sizing | XXI-T10, X-T9, XXII-T7 | measured β(M, context) → tree shapes and γ | the tree plan beats the best chain by ≥ 10% on the M5 Max, or trees are disabled for that chip |
| G6-g exactness CI | XXI-T9, V-T3, XXIII-T17 | → `spec_exactness.rs`; the mlx-lm #1470 script; scenarios S9 and S9v | every row green on the CI runner per PR and on the M5 Max nightly |
| G6-h start-up and AOT | XVIII-T1, XVIII-T2, XVIII-T3, XVIII-T4, XVIII-T5, XVIII-T7, XIII-T16, XI-T8 | plan cache and `.skybin` → a phase trace, overlapped start-up, residency by layer group | a second process start creates every pipeline in ≤ 50 ms on the M2 Pro; the plan cache holds no MSL and no binary |
| G6-i prefix cache | XIV-T8, XVIII-T8 | KV blocks → a radix tree, a RAM pool, `.skykv` restore | a hit equals a miss bitwise for prefill-written prefixes; restore-and-continue equals the uninterrupted session |
| G6-j measurements | XXI-T15, V-T8, XX-T12 | the harness → the acceptance table; cold start and peak RAM for every engine; CPU time and joules per token | committed with run_ids |
| G6-k owner items | XXVII-T4, XXVIII-T4 | → invention disclosures for counsel; adoption thresholds | both recorded before any public release |

**Hardware.** The M5 Max (scenarios S9, S5 and S10); the CI runner; the wall-power meter (I6).

**Kill and pivot criteria.**

| Measurement | Threshold | Action |
|---|---|---|
| Sky speculative ÷ mlx-lm speculative, same draft and prompts, T = 0 and 0.6, M5 Max | < 1.2× (pivot); < 1.1× (stop) | **PIVOT** (R-3, applying R-2's action): if every other measured advantage also lies within what BaseRT already reports (≤ 1.05× plain decode, ≤ 1.44× short-prompt prefill, ≤ 1.33× small models), contribute the verify kernels and exactness upstream and drop the separate-engine acquisition plan |
| the same | < 1.1× | **STOP** presenting the speculative engine as a differentiator (R-3) |
| Speculative ÷ Sky plain, EAGLE-3, T = 0.6, declared suite | suite average < 1.5×; or every class < 1.2× | **CLAIM**: restrict claims to the classes with ≥ 1.7×. **STOP** the speculative headline if every class is below 1.2× (R-12) |
| MLX ships both before Sky's G6: verify within 1.3× of M = 1 up to M = 8 on the M5 Max, and speculative output equal to plain output | both shipped | **PIVOT** (R-7): drop exact speculation as the headline; continue only if long-context decode, cold TTFT or energy keeps a margin of at least 1.2× |

### 7.8 G7 — Breadth (weeks 20–40)

**Goal.** Sky runs the models people use, from both file formats, with the same correctness suite and at least parity:
- Llama 3.x, Gemma 3, Mistral, Phi-4, gpt-oss (MXFP4) and Qwen3-MoE;
- the Qwen3.6/3.8 hybrids with MTP and exact state rollback;
- a server;
- the first model written in Sky source.

**Exit criteria.**

| Criterion | Threshold | Source |
|---|---|---|
| Families | Llama 3.x, Mistral, Gemma 3, Phi-4, gpt-oss and Qwen3-MoE each pass Part XIII §5.3 on their smallest checkpoint, and decode at ≥ 1.0× the baseline on their same-file pairs | spine G7 |
| GGUF formats | Q4_0, Q4_K_M and Q8_0 files, plus Q5_K, IQ4_NL and MXFP4 through XII-T8, stream at ≥ 88% of the raw read, or are reported as format-limited | Part XII Q-D4, §6.6 |
| Hybrids | Qwen3.6/3.8-27B: plain decode, plus MTP speculation byte-identical to plain on 32 prompts × T ∈ {0, 0.6}; commit cost ≤ 3% of a round; effective throughput measured against MTPLX on the same machine | XIV-T6; XXI-T13; R-9 |
| MoE speculation | speculative ≥ plain at every tested γ, or speculation is off by default for that model | XXI-T14 |
| Server | OpenAI-client conformance on streamed and non-streamed chat; idle CPU < 1% of a core over 60 s | XIII-T18 |
| Two front doors (D7) | `models/qwen3.sky` produces normalized L1 identical to the Rust builder's | VII-T14 |

**Tasks (25).** XIII-T18, XIII-T19, XIII-T20; XII-T8, XII-T9, XII-T13 (only if XVII-T8 measured an int8 rate ≥ 1.5× fp16); XIV-T4, XIV-T5, XIV-T6, XIV-T11; XV-T4, XV-T8, XV-T9, XV-T11 (only if a fixed split size costs > 3% somewhere at G5); XVIII-T9, XVIII-T10; XIX-T6; XXI-T12 (the Qwen3-8B DFlash runtime, only if R-4 passed; the Qwen3-32B drafter is G9), XXI-T13, XXI-T14; VII-T9; VII-T13, VII-T14, VI-T3, II-T5.

G7 runs as five lanes:

| Lane | Weeks | Tasks |
|---|---|---|
| hybrid models, pulled forward by R-9 | 20–40 | VII-T9, XIII-T20, XIV-T6, XXI-T13, XV-T9 (head dim 256) |
| families and formats | 22–40 | XIII-T19, XII-T8, XII-T9, XIV-T5, XV-T8, XXI-T14, XXI-T12, XII-T13 |
| KV modes | 22–40 | XIV-T4, XV-T4, XIV-T11, XIX-T6, XV-T11 |
| server | 30–40 | XIII-T18, XVIII-T9, XVIII-T10 |
| Sky-source front door | 28–40 | VII-T13, VII-T14, VI-T3, II-T5, and Lane A's language prerequisites (§8) |

**Dependencies.**
- G5's GGUF pair and codecs; G4's plan.
- G6's speculation runtime, for MTP and for MoE speculation.
- XVII-T8's int8 measurement, for XII-T13.
- R-4's decision, for XXI-T12.

**Work packages.**

| WP | Tasks | Consumes → produces | Done when |
|---|---|---|---|
| G7-a Llama 3.x and Mistral | XIII-T19 (`llama.rs`) | normalized config → TIR (the llama3 RoPE remap; θ up to 1e9 with fp32 angles) | Part XIII §5.3 on Llama 3.2-1B and Mistral 7B |
| G7-b Gemma 3 | XIII-T19 (`gemma3.rs`), XV-T8, XIV-T5 | sandwich norms, GeGLU-tanh, 5:1 local and global layers, ring caches | §5.3 on Gemma 3 1B; exact rewind across a ring wrap |
| G7-c Phi-4 | XIII-T19 (`phi3.rs`) | fused `qkv_proj` split; LongRoPE | §5.3 on Phi-4-mini |
| G7-d gpt-oss | XIII-T19 (`gpt_oss.rs`), XII-T8 (MXFP4), XIV-T5 (sinks) | MXFP4 experts, learned sinks, the harmony format | G-same against llama.cpp on gpt-oss-20b |
| G7-e Qwen3-MoE | XIII-T19 (`qwen3_moe.rs`), XXI-T14 | experts stacked by the relayout; expert-deduplicated verify | §5.3 on Qwen3-30B-A3B; speculation never loses, or is off |
| G7-f hybrids | VII-T9, XIII-T20, XIV-T6, XXI-T13, XV-T9 | Gated DeltaNet scan; gated attention; M-RoPE; MTP; double-buffered state | the hybrid criteria above; state memory 2 × 151 MB |
| G7-g GGUF breadth | XII-T8, XII-T9 | P1 codecs; the plane-split experiment | Part XII §6.6 targets, or "format-limited"; the RL2 decision record |
| G7-h KV modes | XIV-T4, XV-T4, XIV-T11, XIX-T6 | `--kv=int8`, fused into attention; bucket growth; `--kv grow` | int8 KV: 32K decode ≥ 1.12× fp16 KV on 32B and ≥ 0.98× at 1K, within the KL budget |
| G7-i server | XIII-T18, XVIII-T9, XVIII-T10 | `sky-llm` → `sky serve` | the server criterion |
| G7-j Sky-source front door | VII-T13, VII-T14, VI-T3, II-T5 | edition 2027 surface → TIR through `tensor.call` | `front_doors.rs`: byte-identical canonical text |
| G7-k DFlash for 8B (conditional) | XXI-T12 (runtime) | a DFlash checkpoint → a block drafter | τ within 5% of the card; ≥ 2.5× plain on HumanEval-style prompts on the M5 Max |
| G7-l W8A8 prefill (conditional) | XII-T13 | int8 GEMM with a dequantizing epilogue | Part XII §7.4's budget on Qwen3-8B 8-bit |
| G7-m binary-tree combine (conditional) | XV-T11 | an `attn_combine` variant | built only on G5's evidence |

**Hardware.** The M5 Max: gpt-oss-120b, Llama 70B at 32K and the 27B hybrid need the 128 GB machine (Part XXIII §4.4). The M2 Pro for small models.

**Kill and pivot criteria.**

| Measurement | Threshold | Action |
|---|---|---|
| Hybrid models with exact rollback | not supported by week 40 | **CLAIM** (R-9): the speculative headline may not be presented as state of the art |
| G4's schedule | G4 met after week 20 | G7 was already cut to Qwen3 plus one other family (R-20) |

### 7.9 G8 — Proof (weeks 30–44)

**Goal.** Make every claim public, reproducible by an outsider, sustained, explained, and measured on at least three chips.

**Exit criteria.**

| Criterion | Threshold | Source |
|---|---|---|
| `sky-evaluation` public | pre-registration, locks, raw data, reports and `reproduce.sh` verdicts | XXIII-T5, XXIII-T18 |
| Chips | at least three: the M5 Max (Tiers H and A), the M4 Max (Tier A), the M2 Pro (Tier B) | spine G8; Part XXIII §4.6 |
| Energy | ≤ 0.85× mlx-lm joules per token, plain decode; whole-system energy (`PSTR` or the meter); sustained runs of ≥ 10 minutes | spine §3 |
| Peak RAM | ≤ 0.9× mlx-lm at 32K with 16-bit KV; or, if mlx-lm's 32K overhead is under about 3 GB, restated as overhead above the 27.5 GB physical floor | spine §3; R-18 |
| Why faster | for each headline, an attribution whose unexplained share is ≤ 10% of the time difference | XXIII-T13 |
| Outside reproduction | the headline cells within 5% | spine G8 |
| Sustained | each headline's sustained lead ≥ 1.05× and ≥ half its burst lead; otherwise only the sustained number is published | R-16 |
| Paper draft | written. Submission waits for the patent or defensive-publication decision (XXVII-T4) | spine G8 |
| Demo | dress rehearsal on the M5 Max; every on-screen number traces to a run_id | XXIII-T19 |
| One-page summary | generated from results JSON, with no `TBD` left | XXVII-T3 |

**Tasks (6).** XXIII-T18, XXIII-T19; V-T9; X-T10 (needs a second M5-family chip; otherwise it moves to year 2); XXVII-T3, XXVII-T5. G8 also completes two earlier tasks: XXIII-T5 (the public repository) and XXIII-T13 (the "why faster" report).

**Dependencies.** G6's exit; G7's families for the Tier A matrix; I7 for the third chip; counsel's answers before publication (R-24).

**Work packages.**

| WP | Tasks | Consumes → produces | Done when |
|---|---|---|---|
| G8-a public repository | XXIII-T5 (public), XXIII-T18 | the private repository → the public one: results, `SHA256SUMS`, `reproduce.sh` | an outside party reproduces the headline cells within 5% |
| G8-b third chip | XXIII-T18 (M4 Max, Tier A) | I7 → Tier A JSON | committed |
| G8-c energy | V-T9 | S6 loops of ≥ 60 s; the meter | error ≤ 7%; joules per token for every engine |
| G8-d why faster | XXIII-T13 (`--compare`) | Sky's and the competitors' censuses → attribution | unexplained share ≤ 10% per headline |
| G8-e demo and one-pager | XXIII-T19, XXVII-T3, XXVII-T5 | results JSON → the demo; the one-page summary | dress rehearsal passed; no `TBD` |
| G8-f tuning transfer | X-T10 | M5 Max entries → seeded entries for another M5 chip; per-family default tables | within 3% of a full search at ≤ 20% of its time |
| G8-g paper draft | — | results → a draft | written |

**Hardware.** The M5 Max, the M4 Max (I7), the M2 Pro and the meter. Optional chips: an M3 Max, to reproduce mlx-lm's published 1.82× speculative result, and an M5 Pro, to compare with BaseRT (Part XXIII §4.4).

**Kill and pivot criteria.**

| Measurement | Threshold | Action |
|---|---|---|
| Outside reproduction | outside 5% | **CLAIM**: publish no claims (R-17) |
| Sustained against burst lead | sustained < 1.05×; or < half the burst lead | **CLAIM**: drop the claim; or publish only the sustained number (R-16) |
| Lead against the competitors' then-current releases | below the §3 targets | **CLAIM**: not in the acquisition material (R-8) |
| Demo host stability under the sustained protocol | a kernel panic or a wired-memory stall | no public demo until three clean runs in a row (R-14) |
| Counsel on authorship, before G8 | ownership cannot be warranted | **PIVOT** (R-24): drop the acquisition package; pursue open-source or partnership outcomes |
| Adoption and inbound interest, G8 + 6 months | below the owner's thresholds (XXVIII-T4), and no contact | **STOP** optimizing for acquisition (R-23; Part XXVI §9) |

### 7.10 After G8: weeks 44–52

- **Buffer.** G7 and G8 absorb slips from G4–G6 first. A family that misses week 40 moves to year 2; the reproduction does not slip.
- **Re-baseline.** Every headline is re-measured against the competitors' releases of that month (R-8), and `sky-evaluation` is updated.
- **The adoption clock.** The G8 + 6 months window of R-23 starts.
- **G9 spikes,** in the order of Part XXVI §3: ASTC feasibility; the drafter-training pilot on the 8B recipe; mixed bits on models up to 8B. The megakernel spike runs only on Part XX §8.4's triggers.
- **Year 2.** The year-2 plan and the first hires (Part XXVI §10).

**Deferred to G9 and Part XXVI (7 tasks):**

| Task | Why deferred |
|---|---|
| VIII-T11 | the e-graph prototype; EXPERIMENTAL, off by default; R-13 keeps it at G9 |
| XII-T12 | mixed-bit search; EXPERIMENTAL up to 8B, MOONSHOT for 32B; it is re-quantization, so never part of a same-file claim |
| XII-T14 | the ASTC spike [MOONSHOT] |
| XIV-T9 | paged KV for a multi-session server; the year-1 server serves one generation at a time (Part XIII §14), and paging buys nothing at batch 1 (Part XIV §16) |
| XIV-T10 | commit-on-demand KV; bucket growth (XIV-T11) covers year 1 |
| XIV-T12 | 4-bit KV research; no Qwen3 KV-quantization study exists (Part XIV KV-D5) |
| XX-T13 | the megakernel spike; only on Part XX §8.4's triggers |

Two partial deferrals: the Qwen3-32B drafter training of XXI-T12 (G9; Part XXVI §3), and the execution of III-T8's scientific fixes (year 2; §8).

## 8. Roadmap A: the CPU lane

Roadmap A is Part III's list of CPU fixes, F1–F11. Decision D16 makes it secondary, because no CPU fix moves a year-1 inference metric (Part III §7.4). It runs as a low-priority lane, in the order Part III §5.2 recommends, and it never takes a work-package slot from a gate's critical path.

| When | Task | Fix | Gate, per unit of work by slope (Part III §5.1, §6.3) | Priority |
|---|---|---|---|---|
| G0 (weeks 0–2) | III-T1 | benchmark hygiene: a slope mode, the fixed-cost split, corrected explanations, new cases including a sampler over 151,936 entries | the regenerated `RESULTS.md` reproduces Part III's per-unit ratios within ±10% | do now |
| G0 | II-T6 | `sky hw` reports the thread count the kernels use | `sky hw` and `sky check --explain` name the same count | do now |
| G0 | III-T8 | one issue per scientific-track fix (F2, F4, F5, F7, F9, F10), carrying R10's design, gate and test plan | the issues exist; none is scheduled inside G1–G4 | do now (filing only) |
| G1–G4 (weeks 1–14) | — | frozen: bug fixes only (§2.3) | — | — |
| G5 window (weeks 14–20) | III-T6 | F6: in-place reuse in native code | elementwise 10^7, one thread, ≤ 1.15× C per pass; peak RSS ≤ the SIR engine's | low; when capacity allows |
| G5 window | III-T7 | F8: a compile cache, starting with a written decision under non-negotiable 3 | the decision; code only if approved; then nbody's fixed cost ≤ 10 ms on a cache hit | low |
| G6–G7 (weeks 20–40) | III-T4 | an integer type: the design note | the owner approves it before any code | with the Sky-source lane |
| G7 (weeks 28–40) | III-T2 | F1: inline element access for big arrays | the indexing loop ≤ 1.5× C (today 57×) | with the Sky-source lane |
| G7 | III-T3 | element writes on the buffer representation | a write loop over 10^6 entries ≤ 2× C, with no allocation per iteration | with the Sky-source lane |
| G7 | III-T5 | F3: an SIR inliner and a leaner call protocol | rk4 ≤ 1.3× C per step; fib ≤ 2.0× C per call | with the Sky-source lane |
| Year 2 (Part XXVI) | III-T8's fixes | F2, F4, F5, F7, F9, F10 | Part III §5.1's gates | the scientific track; the numerical computing engineer |
| Not now | — | F11, an LLVM backend | revisit after F1–F3 | — |

Why this order:
- **Hygiene first,** because every later CPU gate is per unit of work, by slope.
- **F6 and F8 next.** Both are cheap and help scientific users, but F8 needs a decision under non-negotiable 3 first (Part III §10).
- **F1, element writes, the integer type and F3 come with the language work.** They are justified by D7's second front door and by user-written logits processors, so they belong with that work (Part III §7.4).
- **F7 is scientific only.** Its Accelerate route must be re-verified on M1, M3 and M4 (Part III §6.2).

## 9. Performance gates

### 9.1 The owner's example gates, judged

The mission (§73) asks for performance gates in the style of the owner's examples. Each example, against the evidence:

| Owner's gate | Verdict | Honest form | Gate | Evidence |
|---|---|---|---|---|
| "Fused elementwise ≤ 1.1× C" | **Realistic if reworded.** Already met per element for axpy (1.00–1.01×) and hypot (1.00×). Not met for chain6 (1.14–1.17×), whose divisor needs its own check. Not realistic on whole-process medians (1.5–2.4×), which time the front end and input construction [VERIFIED: Part III §6.1] | "Fused elementwise kernels (axpy, hypot, chain6) at ≤ 1.1× the best of C and Rust per element, one thread, output preallocated (`cargo bench -p sky-jit --bench kernels`); and ≤ 1.15× C per pass inside a Sky program, by slope" (Part III §6.3) | Lane A, after F6 (G5 window) | Part III §6.1 |
| "Matmul ≤ 1.2× Accelerate" | **Not realistic in the default mode without F7.** Sky's reproducible kernel is 14–15× Accelerate on one thread and 4.6–5.2× on all threads. It is already at 89% of its own no-FMA ceiling; the rest is the FMA ban and Apple's AMX unit [VERIFIED: Part III §6.2] | "Default mode: GEMM ≥ 45 GFLOP/s per P-core, and ≤ 3× Accelerate on all threads at n ≥ 256. Relaxed mode: ≤ 1.2× Accelerate at n ≥ 256. If F7's self-test passes on M1–M4: ≤ 1.3× Accelerate in the default mode" (Part III §6.3) | the scientific track (year 2) | Part III §6.2 |
| "Sky runs a 1B LLM" | **Realistic; reword it to exact models and their correctness gates.** Qwen3 has 0.6B and 1.7B models; Llama 3.2-1B and Gemma 3 1B arrive with G7 (Part XIII §5.2) | "Qwen3-0.6B logits match HF transformers in fp32 on the reference interpreter (G2). Qwen3-0.6B and 8B generate on Metal, with greedy output equal to mlx-lm on 20 prompts × 128 tokens and every divergence explained (G3)" | G2, G3 | Part XXIV #6, #8 |
| "Sky 1B ≥ 0.8× MLX" | **The wrong bar in both directions.** Unfused, with a host wait per token, a small model is overhead-bound and may fall below 0.8× at G3 (Part XXIV #8). Once fused, 0.8× is far too low: mlx-lm runs the 0.6B at 54.5% of bandwidth and is CPU-bound [VERIFIED: R4] | Replaced: Qwen3-8B ≥ 0.7× mlx-lm unfused (G3); Qwen3-0.6B ≥ 1.3× mlx-lm on the M2 Pro (G4); ≥ 1.3× on the M5 Max, target 1.5× (G5) | G3, G4, G5 | R4; XVI-T8; R-18(c) |
| "Sky 7B ≥ MLX" | **Realistic.** Qwen3 has no 7B; the 8B is the model on the development Mac. Modeled at 1.09–1.11× (Part XVI §2.2) `[EXPERIMENTAL: G4]` | "Qwen3-8B 4-bit decode ≥ 1.0× mlx-lm on the M2 Pro at short context, with GEMV streaming ≥ 95% of the raw read" | G4 | spine G4 |
| "Sky 32B > MLX by 10%" | **Not realistic for plain decode on the M5 Max.** MLX's 28.4 tok/s moves about 511 GB/s; 1.10× needs about 563 GB/s, above the only achievable-bandwidth measurement on that chip, about 530 GB/s [arithmetic in R-18 on R5 and R3-issues] | Replaced by the measurable form of the owner's Milestone 5 (Part XXVII §5.8): "Plain decode ≥ 1.0× and ≥ 95% of the raw read; and ≥ 1.1× on at least two §3 metrics where physics allows it: long-context decode (≥ 1.2× at 32K), small models (≥ 1.3×), prefill on M1–M4 (≥ 1.25×), cold TTFT (≥ 3×)" | G5 | R-18; Part XXVII §5.8 |
| "Sky 32B > MLX by 30%" | **Impossible for plain decode.** The datasheet ceiling, 33.9 tok/s, is 1.19× MLX's 28.4 [arithmetic on Part IV §4 and R5]. Realistic only as effective throughput from exact speculation | Replaced by Milestone 6: "Exact speculation ≥ 1.8× mlx-lm plain and ≥ 1.2× mlx-lm's own speculative mode with the same draft (R-3's pivot line is 1.3×), on a declared workload suite, with output byte-identical to Sky's plain output and each workload class reported separately" | G6 | Part IV §8; Part XXI §10 |

### 9.2 The gate ladder

These gates replace the owner's examples. Each is measured under Part XXIII's protocol. A ratio is Sky ÷ baseline on a same-file pair (baseline ÷ Sky for TTFT), and the CI bound named in Part XXIII §2 must clear the threshold.

| # | Metric | Chip and model | Threshold | Gate | Source |
|---|---|---|---|---|---|
| PG1 | Q4 GEMV streaming | M2 Pro; Qwen3-8B and 32B shapes | ≥ 88% of the raw read (target 95%); ≥ mlx on every shape | G1 | spine G1 |
| PG2 | Batch-invariant verify | M2 Pro; 32B shapes | β ≤ 0.15 (t(4)/t(1) ≤ 1.45); invariance cost ≤ 3% | G1 | R-3, R-11 |
| PG3 | First tokens | M2 Pro; Qwen3-8B | greedy agreement; correctness gates; ≥ 0.7× | G3 | spine G3 |
| PG4 | Dense decode parity | M2 Pro; Qwen3-8B | ≥ 1.0× (aim 1.05–1.10×); end-to-end invariance cost ≤ 5% | G4 | spine G4 |
| PG5 | Small-model decode | M2 Pro, then M5 Max; Qwen3-0.6B | ≥ 1.3× (target 1.5×) | G4, G5 | spine §3; R-18 |
| PG6 | Plain decode on the headline | M5 Max; Qwen3-32B, ≤ 1K context | ≥ 1.0× and ≥ 95% of the raw read; ≥ 1.05× only if the raw read is ≥ about 575 GB/s | G5 | spine §3 |
| PG7 | Long-context decode | M5 Max; Qwen3-32B, 16-bit KV | ≥ 1.05× / 1.10× / 1.2× at 8K / 16K / 32K (bounds about 1.08× / 1.14× / 1.24×) | G5 | Part XV §13 |
| PG8 | Prefill | M5 Max, 32B; M2 Pro, 8B | M5: ≥ 1.0× at ≥ 2,048-token prompts. M1–M4: ≥ 1.25× mlx-lm and ≥ 1.0× llama.cpp | G5 | spine §3 |
| PG9 | Verify cost on the headline chip | M5 Max; 32B shapes | weight GEMVs ≤ 1.3× of M = 1 up to M = 8; the neural-accelerator tier decided | G5 | spine §3; R-4 |
| PG10 | Exact speculation | M5 Max; Qwen3-32B | ≥ 1.8× mlx-lm plain; ≥ 1.2× mlx-lm speculative (pivot below 1.2×, stop below 1.1×); byte-identical output | G6 | spine §3 |
| PG11 | Speculation on a declared suite | M5 Max; EAGLE-3, T = 0.6 | ≥ 1.7× Sky plain on each claimed class; suite average ≥ 1.5× | G6 | Part XXI; R-12 |
| PG12 | TTFT | M5 Max; Qwen3-32B | warm ≥ 1.15×; cold ≥ 3×; repeated prefix ≥ 1.3× (about 2× at ≤ 8 new tokens) | G6 | spine §3 |
| PG13 | Hybrid with MTP | M5 Max; Qwen3.6/3.8-27B | exact; effective throughput against MTPLX on the same machine | G7 | R-9; XXI-T13 |
| PG14 | Breadth | each G7 family | correctness suite; ≥ 1.0× on its same-file pairs | G7 | spine G7 |
| PG15 | Peak RAM | M5 Max; Qwen3-32B | ≤ mlx-lm (G5); ≤ 0.9× at 32K (G8), or restated as overhead above the 27.5 GB floor | G5, G8 | spine §3; R-18 |
| PG16 | Energy | M5 Max | ≤ 0.85× mlx-lm joules per token, plain decode, whole system, sustained | G8 | spine §3 |
| PG17 | Reproduction | ≥ 3 chips | an outside party within 5% | G8 | spine G8 |

**"60 tok/s" appears in no gate.** It has two honest forms:
- effective throughput with exact speculation on Qwen3-32B: 47–59 tok/s typical with the public EAGLE-3 drafter (SPINE §3; Part XXI §10 models 47–61), ≥ 60 on input-grounded or greedy work, and a 60+ average only with a DFlash-class drafter that nobody has published (G9; Part XXI §15) `[EXPERIMENTAL: modeled]`;
- plain decode on an M5 Ultra, whose planning ceiling is 59.6 tok/s (Part IV §4).

### 9.3 Rules for every performance gate

- A target is met only when the CI bound named in Part XXIII §2 clears it. A straddling CI is "not shown".
- Bits per weight, weight bytes per token and KLD are printed next to every speed (D5, D13).
- Plain short-context decode is reported as parity when it is parity (Part XXIII §17).
- Every ratio is re-measured against the competitors' then-current releases at G5 and at G8 (R-8).
- Part XVII §11 treats 1.1× as the M1–M4 prefill gate, and Part XXIII §2 still lists "repeated prefix ≥ 5×". Both predate the current SPINE §3, which this part follows: ≥ 1.25× and ≥ 1.3×. A ≥ 5× TTFT ratio is reported only against an engine without prefix reuse, and labeled so.

## 10. Where every chapter task lands

Every implementation task defined in a design chapter, with the gate that owns it. A note in parentheses marks work that starts earlier or finishes later than its owning gate. "Conditional" tasks run only if the measurement named in their gate section allows it. XXV-T1 keeps this table as data and fails CI if a task is unplaced or placed twice.

| Part | Tasks | Placement |
|---|---:|---|
| [II](part02-sky-today.md) | 8 | **G0:** II-T1, II-T2, II-T6 · **G1:** II-T3, II-T7 · **G2:** II-T4, II-T8 · **G7:** II-T5 |
| [III](part03-current-performance.md) | 8 | **G0:** III-T1, III-T8 (filing; the fixes are year 2) · **Lane A:** III-T2, III-T3, III-T4, III-T5, III-T6, III-T7 |
| [IV](part04-hardware-limits.md) | 4 | **G0:** IV-T1 · **G1:** IV-T2, IV-T3 (re-run under the protocol at G5), IV-T4 |
| [V](part05-competitors.md) | 11 | **G0:** V-T1, V-T2 · **G1:** V-T4, V-T10, V-T11 · **G5:** V-T5, V-T6, V-T7 · **G6:** V-T3 (started at G4), V-T8 · **G8:** V-T9 |
| [VI](part06-architecture.md) | 6 | **G0:** VI-T4 (PRs opened) · **G1:** VI-T1, VI-T2 · **G4:** VI-T5, VI-T6 · **G7:** VI-T3 |
| [VII](part07-tensor-ir.md) | 14 | **G2:** VII-T1, VII-T2, VII-T3, VII-T4, VII-T5, VII-T6, VII-T7, VII-T8, VII-T10, VII-T11 · **G4:** VII-T12 · **G7:** VII-T9, VII-T13, VII-T14 |
| [VIII](part08-graph-optimizer.md) | 11 | **G3:** VIII-T1, VIII-T2, VIII-T3 · **G4:** VIII-T4, VIII-T5 (D7 leg at G7), VIII-T6, VIII-T7, VIII-T8, VIII-T9, VIII-T10 · **deferred:** VIII-T11 |
| [IX](part09-scheduler.md) | 10 | **G4:** IX-T1, IX-T2, IX-T3, IX-T4, IX-T5, IX-T7, IX-T9, IX-T10 · **G5:** IX-T6, IX-T8 |
| [X](part10-autotuner.md) | 10 | **G1:** X-T1 · **G5:** X-T2, X-T3, X-T4, X-T5, X-T6, X-T7, X-T8 · **G6:** X-T9 · **G8:** X-T10 (needs a second M5-family chip) |
| [XI](part11-metal-backend.md) | 17 | **G1:** XI-T1, XI-T2, XI-T3, XI-T6 (Metal 4 path at G5), XI-T10, XI-T11, XI-T16 · **G3:** XI-T4, XI-T5, XI-T7 · **G4:** XI-T14, XI-T15 · **G5:** XI-T9, XI-T12, XI-T13, XI-T17 · **G6:** XI-T8 |
| [XII](part12-quantization.md) | 14 | **G1:** XII-T7 (GGUF codecs at G5) · **G2:** XII-T1, XII-T2, XII-T3, XII-T5, XII-T6 · **G3:** XII-T10 · **G5:** XII-T4, XII-T11 · **G7:** XII-T8, XII-T9, XII-T13 (conditional) · **deferred:** XII-T12, XII-T14 |
| [XIII](part13-llm-runtime.md) | 20 | **G2:** XIII-T1, XIII-T4, XIII-T5, XIII-T6, XIII-T7 · **G3:** XIII-T2, XIII-T3, XIII-T9, XIII-T10, XIII-T11, XIII-T13, XIII-T17 · **G4:** XIII-T12, XIII-T14 · **G5:** XIII-T8 · **G6:** XIII-T15, XIII-T16 · **G7:** XIII-T18, XIII-T19, XIII-T20 |
| [XIV](part14-kv-cache.md) | 12 | **G3:** XIV-T1 · **G4:** XIV-T2 · **G6:** XIV-T3, XIV-T7, XIV-T8 · **G7:** XIV-T4, XIV-T5, XIV-T6, XIV-T11 · **deferred:** XIV-T9, XIV-T10, XIV-T12 |
| [XV](part15-attention.md) | 12 | **G4:** XV-T1, XV-T2 · **G5:** XV-T5, XV-T6, XV-T7, XV-T10, XV-T12 · **G6:** XV-T3 · **G7:** XV-T4, XV-T8, XV-T9, XV-T11 (conditional) |
| [XVI](part16-decode.md) | 11 | **G1:** XVI-T1 · **G4:** XVI-T2, XVI-T3, XVI-T4, XVI-T5, XVI-T6, XVI-T7, XVI-T8 · **G5:** XVI-T9, XVI-T10, XVI-T11 |
| [XVII](part17-prefill.md) | 9 | **G5:** XVII-T1, XVII-T2, XVII-T3, XVII-T4, XVII-T5, XVII-T6, XVII-T7, XVII-T8, XVII-T9 |
| [XVIII](part18-ttft.md) | 10 | **G5:** XVIII-T6 · **G6:** XVIII-T1, XVIII-T2, XVIII-T3, XVIII-T4, XVIII-T5, XVIII-T7, XVIII-T8 · **G7:** XVIII-T9, XVIII-T10 |
| [XIX](part19-memory.md) | 12 | **G0:** XIX-T11 · **G4:** XIX-T1, XIX-T2, XIX-T3, XIX-T4, XIX-T5 · **G5:** XIX-T7, XIX-T8, XIX-T9, XIX-T10 · **G6:** XIX-T12 · **G7:** XIX-T6 |
| [XX](part20-persistent-execution.md) | 13 | **G4:** XX-T1, XX-T2, XX-T3, XX-T4, XX-T5, XX-T6, XX-T11 · **G5:** XX-T7, XX-T9 · **G6:** XX-T8, XX-T10, XX-T12 · **deferred:** XX-T13 |
| [XXI](part21-speculative.md) | 15 | **G1:** XXI-T1 · **G5:** XXI-T2 · **G6:** XXI-T3, XXI-T4, XXI-T5, XXI-T6, XXI-T7, XXI-T8, XXI-T9, XXI-T10 (conditional), XXI-T11, XXI-T15 · **G7:** XXI-T12 (8B runtime, conditional; 32B drafter training deferred), XXI-T13, XXI-T14 |
| [XXII](part22-performance-model.md) | 7 | **G1:** XXII-T2 · **G4:** XXII-T1, XXII-T3, XXII-T4, XXII-T5 · **G5:** XXII-T6 · **G6:** XXII-T7 |
| [XXIII](part23-benchmarks.md) | 20 | **G0:** XXIII-T1, XXIII-T3, XXIII-T5 (public at G8), XXIII-T6, XXIII-T7, XXIII-T9, XXIII-T11, XXIII-T20 · **G1:** XXIII-T2 (M4–M10 at G4–G6) · **G3:** XXIII-T8, XXIII-T10 · **G4:** XXIII-T4, XXIII-T12, XXIII-T13 ("why faster" at G8), XXIII-T14 · **G5:** XXIII-T15, XXIII-T16 · **G6:** XXIII-T17 · **G8:** XXIII-T18, XXIII-T19 |
| [XXVII](part27-moat.md) | 8 | **G1:** XXVII-T1, XXVII-T2, XXVII-T6, XXVII-T8 · **G5:** XXVII-T7 · **G6:** XXVII-T4 · **G8:** XXVII-T3, XXVII-T5 |
| [XXVIII](part28-risks.md) | 4 | **G0:** XXVIII-T1 · **G1:** XXVIII-T2, XXVIII-T3 · **G6:** XXVIII-T4 |
| **Total** | **266** | G0 19 · G1 29 · G2 22 · G3 17 · G4 52 · G5 48 · G6 35 · G7 25 · G8 6 · Lane A 6 · deferred 7 |

## 11. Reporting progress

### 11.1 The weekly measured dashboard

Generated every week (XXV-T2) from three sources:
- `perf-history`, the nightly results (XXIII-T14);
- `sky-evaluation`, from the weekly quick tier on the M5 host, and protocol runs when they exist;
- the task index (XXV-T1).

| Panel | Content | Source |
|---|---|---|
| Gate exits | for the current and the next gate: each exit metric with its latest measured value, CI, target, status (met, not shown, failed, not yet measured) and run_id | `sky-evaluation`, `perf-history` |
| Exact counters | per token: dispatches, dependent dispatches, barriers, command buffers, allocations, weight and KV bytes; pipelines per plan; all against the golden files | `sky explain --format json`, `skycount` |
| Kill criteria | every R-n due at the current or next gate: measurement, threshold, value, date checked | `risks.json` (XXVIII-T1) |
| Correctness | G-tok, G-same, G-greedy, the invariance suite, the exactness suite: pass or fail per chip | CI |
| Competitors | pinned versions of MLX, mlx-lm, llama.cpp, Monolith, Core AI and MTPLX; their last numbers; any tracked gap that moved by more than 5% | V-T10, XXVII-T8 |
| Tasks | per gate: done, in progress or moved, by task ID; work packages in flight | XXV-T1 |
| Machine time | M5 host hours booked and used, by activity | the reservation calendar (XXV-T3) |

Rules:
- **Measured numbers only.** A projected number appears only in a column labeled "predicted (Part XXII)", never in a status cell.
- **Traceable.** Every number links to a run_id and a commit.
- **No rounding up.** A metric whose CI straddles its target is "not shown", never "met" (Part XXIII §2).
- **Immutable history.** Each weekly snapshot is committed (`status/<year>-W<nn>.md` in `sky-evaluation`), so it cannot be edited after the fact.

### 11.2 Explain reports

At every gate exit, each metric the gate claims gets a "why" report from `sky explain` (XXIII-T13):
- per kernel class: bytes, FLOPs, predicted and measured time, and the residual against Part XXII's model;
- the closure error e of the token budget (XXIII-T4);
- for comparisons: the attribution of the time difference to rule IDs and kernel classes, with the competitor's census from `skycount`;
- an unexplained share of at most 10% of the difference. Otherwise the report prints "attribution incomplete", and the claim is not made.

The static report runs in CI from G4. From G8, the `--compare` attribution ships with each published headline.

### 11.3 Gate review records

One record per gate (XXV-T4), with the four inputs of §2.4:
- exit metrics with run_ids;
- kill criteria with their measurements;
- tasks done or moved, with reasons;
- the re-plan, and the owner's decision.

The records are kept with the report, so that slips stay visible.

### 11.4 Competitor watch

- **Weekly** (V-T10): MLX, mlx-lm and llama.cpp at their main branches, pinned.
- **Monthly** (XXVII-T8): the full set, including Monolith, Core AI and MTPLX.

An issue opens when a tracked gap moves by more than 5%. The results feed the dashboard and the re-baselines at G5 and G8.

## 12. ROADMAP.md and spec amendments, and when they land

[Part VI](part06-architecture.md) §9 proposes the amendments as separate PRs once this report is accepted. They land in the order the work needs them:

| # | Amendment (Part VI §9) | Lands | Why then | Task |
|---|---|---|---|---|
| 1 | Principle 2: tensor programs use a sibling IR (TIR) that shares SIR's infrastructure; capabilities remain passes over an IR | G0, by week 2 | it governs `sky-tir` from its first commit (G2) | VI-T4 |
| 2 | Phase 5: "GPUs through MLIR" becomes "Apple GPUs through a native Metal backend (TIR → kernel IR → MSL)" | G0 | it governs `sky-metal` (G1) | VI-T4 |
| 3 | "Building on existing work": drop the MLIR/`melior` GPU row; add MLX, llama.cpp, tinygrad, Metal FlashAttention and ThunderMittens, with the policy "learn from them, write our own kernels, port only with notices" | G0 | before any kernel is written after reading MIT-licensed code (XXVII-T1) | VI-T4 |
| 4 | A new phase, "Inference on Apple silicon", with gates G0–G9 from this part | G0; updated at every gate review | it is the plan of record: re-planned after G0, re-baselined at G5 (SPINE §6) | VI-T4, XXV-T4 |
| 5a | Spec: "Numerics contracts", with `inference` and `portable`, including the exception to "tuning changes speed, never results" for tensor programs (Part II §17 item 2) | G4, by week 14 | G4's invariance and exactness tests test against this text; no exactness claim is made before it | VI-T5 |
| 5b | Spec: the tuning-database rule: `sky compile -O3` and `--autotune` may write the database, as an explicit opt-in, in the same data-only format | G5 | the autotuner lands in G5 | X-T8 |
| 5c | Spec: a byte unit for memory contracts (`cost bytes_per_token <= 18.2 GB`) | G5 | the memory contracts land in G5; the change touches `sky-types`, so it waits until the freeze ends | XIX-T10 |
| 5d | Spec: tensor types and storage formats (`in f16`, `in q4(group: 64)`), edition-gated | G7 (weeks 28–40) | with the Sky-source front door, after G6 (R-21) | VII-T13 |

ROADMAP.md's "Not language features" already says that shared cross-user optimization databases raise privacy and trust questions. Part X §10's opt-in exchange of tuning records therefore stays out of year 1 (Part XXVI §5).

## 13. Implementation tasks

| ID | Task | Where | Deliverable | Acceptance test |
|---|---|---|---|---|
| XXV-T1 | The plan as data | `docs/inference/plan/tasks.toml` | every chapter task ID with its gate, work package, status and dependencies | a CI check fails if a task ID defined in a chapter is unplaced or placed twice, or if a gate closes with one of its tasks neither done nor moved |
| XXV-T2 | The weekly dashboard | `sky-evaluation/tools/dashboard/` | the panels of §11.1, built from `perf-history`, results JSON, `risks.json` and `tasks.toml` | a status cell cannot render a number without a run_id; a predicted number in a status cell fails the build |
| XXV-T3 | Benchmark-host operations | `sky-evaluation/ops/` | SSH access, the reservation calendar, the pinned OS build, `doctor` before every protocol run, the nightly reboot of the perf runner | a protocol run refuses to start on an unpinned OS build or after a failed `doctor` |
| XXV-T4 | Gate review records | `docs/inference/plan/reviews/G<n>.md` | one §2.4 record per gate | with XXVIII-T1: a review cannot be marked closed while a kill criterion due at that gate lacks a measurement |

## 14. Measurements that decide the plan

| When | Measurement | Decides |
|---|---|---|
| G0 | the harness's reproduction of the M2 Pro baselines; run-to-run variation | whether any later ratio can be trusted; the number of blocks per cell (Part XXIII §16 item 2) |
| G1 | the M5 Max raw read bandwidth | whether the plain-decode target is ≥ 1.0× or ≥ 1.05× (the line is about 575 GB/s) |
| G1 | MLX's verify curve on the M5 Max; `matmul2d` by tile and M | R-3 and R-4, early; whether G5's neural-accelerator work is worth its weeks |
| G4 | the end-to-end invariance cost; Part XXII's residual on 8B | R-11; whether projections may be quoted |
| G5 | the autotuner's gain; MLX's attention efficiency at 8K–32K; Core AI; the MTPLX reproduction | R-13; the long-context target; R-6; whether the headline moves to the hybrid |
| G6 | β(M) on the M5 Max; τ per drafter and workload; mlx-lm's speculative mode with the same draft | the pitch (R-2, R-3, R-12) |
| G8 | the outside reproduction; sustained against burst; leads against the competitors' then-current releases | what may be published (R-8, R-16, R-17) |
| Every week | work packages in flight against work packages reviewed; M5 host hours | whether assumption A4 holds |

## 15. Risks and negative results

- **The calendar is optimistic for one person.** Part XXIV's fourteen weeks for G0–G4 assume five parallel agents from week 0 and no review backlog. R-20's week-20 and week-30 limits are the explicit slack; beyond them the plan pivots rather than slips.
- **G4 and G5 are dense.** G4 holds 52 tasks in five weeks and G5 48 in six (§10). Both depend on parallel work packages against frozen interfaces (§7.0), and both mark the tasks that may trail by two weeks without blocking their exit.
- **One M5 Max serializes G5–G8.** Protocol runs, autotuning, nightly performance jobs and development all compete for it. Tier H takes about two days and Tier A about a week per chip [LIKELY: Part XXIII §4.6]. A second M5 Max is the fix (Part XXIII §17).
- **G5 decides too much at once.** Seven kill criteria fall due at week 20. Their measurements are moved earlier on purpose: the day-one kit at G1, the reproduction of MTPLX and Monolith from G1. The G5 review should confirm, not discover.
- **The thesis-critical neural-accelerator path is the most fragile code in the plan** (R-10): miscompiles, an opaque lowering, OS drift. Its decision record (XXI-T2, IX-T8, XV-T5) lands before trees, DFlash and neural-accelerator-family plans are built.
- **Breadth can eat the proof.** G7 and G8 overlap for ten weeks. When they compete, G8 wins: a family that misses week 40 moves to year 2; the reproduction does not slip.
- **Negative results already priced in:**
  - plain short-context decode is at about parity on the M5 Max (Part XVI §12);
  - replay is worth about 0–1% on dense 32B decode (Part XX §9);
  - fusion is worth 1.03–1.08× (Part IX §15);
  - MoE verification is not free (Part XXI §13);
  - int8 activations do not give the same quality (Part XVII §14).

## 16. Reconciliations and disagreement with the spine

### 16.1 Disagreement with the spine

1. **G4's prefill criterion moves to G5.** SPINE §6 lists "prefill ≥ 1.0× mlx-lm on the M2 Pro" at G4. But:
   - the GEMM skeletons that make prefill competitive (MPP `matmul2d` with dequantized tiles, Part XVII) are G5 work in the spine itself;
   - Part XXIV's G4 success criteria omit prefill;
   - a G4 bar of 1.0× mlx-lm would sit below llama.cpp, which already prefills 1.35–1.42× faster than mlx-lm on the same machine (Part V §13 item 2).

   This part measures prefill at G4 and gates it at G5, with the spine's own G5 targets: ≥ 1.25× mlx-lm and ≥ 1.0× llama.cpp on M1–M4.
2. **GGUF loading moves from G7 to G5 for the Qwen3 files.** G5's "≥ 1.0× llama.cpp" can only be measured on the GGUF file (D5: same-file pairs). So the GGUF reader, direct mapping and the Q4_0, Q8_0, Q4_K and Q6_K codecs (XII-T4, XIII-T8, XVI-T9) are G5 tasks. GGUF breadth (the P1 codecs, plane split) stays in G7.
3. **The G6 bar.** SPINE §3 sets ≥ 1.2× over mlx-lm's own speculative mode, and Part XXVIII now matches it: pivot below 1.2×, stop below 1.1×.
   - This part uses the same thresholds.
   - Resolved: an earlier draft of Part XXVIII used 1.3×; the consistency review aligned it to the spine.
   - Part XXI §10 models 1.2–1.35× at T = 0, so either bar is at the edge `[EXPERIMENTAL]`.
4. **Small models.** SPINE §3 targets ≥ 1.5× mlx-lm. R-18(c) notes that this exceeds BaseRT's measured 1.33×, and recommends keeping G4's 1.3×. This part gates at ≥ 1.3× (G4 and G5) and reports 1.5× as the target (PG5).
5. **The hybrid models start at week 20, not late in G7.** R-9 makes Qwen3.6/3.8-27B with MTP the model a buyer will test. Its lane runs through G6 and G7, so that G5's MTPLX reproduction can move the headline in time. G7's exit criterion is unchanged.

### 16.2 Reconciliations with other parts

- **The harness location.** Part XXIV #1 keeps the harness in `benches/llm/` until G8. Part XXIII makes `sky-evaluation` a separate repository from G0. This part follows Part XXIII (private from G0, public at G8), because a harness separate from Sky's code is the point of D13. `sky bench llm` stays in the Sky repository (XXIII-T10). Resolved: Part XXIV #1 now places it in `sky-evaluation` from G0.
- **The first M5 Max measurements.** Part XXVIII §9 item 2 asks for them at G1, and the spine's G5 row now says "target G1". This part makes the day-one kit a G1 exit item (§7.2).
- **Prefill and TTFT numbers.** Part XVII §11 (a 1.1× M1–M4 gate) and Part XXIII §2 (a repeated-prefix target of ≥ 5×) predate the current SPINE §3. This part follows the spine: ≥ 1.25× and ≥ 1.3× (§9.3).
