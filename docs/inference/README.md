# Sky on Apple silicon: research report and implementation plan

This directory is the research report and implementation plan for turning Sky into a compiler and runtime for open LLMs on Apple GPUs. The goal is to beat MLX, mlx-lm and llama.cpp on real local workloads (same weights, same quantization, same prompt, same output quality) and to become technology a company such as Apple would rather acquire than rebuild.

It was written in October 2026 from three kinds of evidence:
- traced source code of tinygrad, MLX, mlx-lm and llama.cpp;
- Apple's documentation and the 2025–2026 literature;
- measurements on an M2 Pro (16-core GPU, 16 GB), with published third-party numbers for the M5 family.

## Start here

1. [Part I — Executive conclusion](part01-executive.md): the answer, the honest targets, and what the owner must decide.
2. [Part IV — Hardware limits](part04-hardware-limits.md): the physics, and why "MLX 35 → Sky 60 tok/s" on an M5 Max cannot be right as written.
3. [Part VI — Architecture](part06-architecture.md): what Sky becomes, with the one-page diagram.
4. [Part XXIV — The first ten implementations](part24-roadmap-now.md): what to build first, and how to know it works.
5. [Part XXVIII — Risks](part28-risks.md): the skeptic's case, and a kill criterion for every gate.

## The verdict in five lines

- Dense, short-context decode is bandwidth-bound, and MLX is already close to the limit. Plain decode on an M5 Max is about parity.
- The honest form of "60 tok/s" on Qwen3-32B is *effective* speed from exact speculative decoding: modeled at 47–59 tok/s typical with public drafters, ≥ 60 on input-grounded work. Plain 60 tok/s needs an M5 Ultra.
- Measured openings:
  - cheap and exact verification of drafted tokens;
  - long-context attention;
  - small and draft models;
  - prefill on M1–M4;
  - cold start;
  - memory and energy.
- Sky's lasting advantages would be the combination of exactness (speculative output byte-identical to plain output), per-chip measured plans, and a compiler that explains every byte. No single kernel is a lasting advantage.
- Worth building, through gates G0–G8 with measured exit criteria. As an acquisition play it is a long shot; see Part XXVII.

## All parts

| Part | Title | File |
|---|---|---|
| I | Executive conclusion | [part01-executive.md](part01-executive.md) |
| II | Sky as it exists today | [part02-sky-today.md](part02-sky-today.md) |
| III | Current performance: where Sky wins, loses, and why | [part03-current-performance.md](part03-current-performance.md) |
| IV | Hardware limits | [part04-hardware-limits.md](part04-hardware-limits.md) |
| V | Competitor architecture: MLX, mlx-lm, tinygrad, llama.cpp and the rest | [part05-competitors.md](part05-competitors.md) |
| VI | Sky's proposed architecture | [part06-architecture.md](part06-architecture.md) |
| VII | Tensor IR (TIR) | [part07-tensor-ir.md](part07-tensor-ir.md) |
| VIII | Graph optimizer | [part08-graph-optimizer.md](part08-graph-optimizer.md) |
| IX | Global scheduler | [part09-scheduler.md](part09-scheduler.md) |
| X | Autotuner and search | [part10-autotuner.md](part10-autotuner.md) |
| XI | The Metal backend | [part11-metal-backend.md](part11-metal-backend.md) |
| XII | Quantization engine | [part12-quantization.md](part12-quantization.md) |
| XIII | LLM runtime | [part13-llm-runtime.md](part13-llm-runtime.md) |
| XIV | KV-cache architecture | [part14-kv-cache.md](part14-kv-cache.md) |
| XV | Attention architecture | [part15-attention.md](part15-attention.md) |
| XVI | Decode optimization | [part16-decode.md](part16-decode.md) |
| XVII | Prefill optimization | [part17-prefill.md](part17-prefill.md) |
| XVIII | TTFT optimization | [part18-ttft.md](part18-ttft.md) |
| XIX | Memory architecture | [part19-memory.md](part19-memory.md) |
| XX | Persistent execution and the compiled decode loop | [part20-persistent-execution.md](part20-persistent-execution.md) |
| XXI | Speculative decoding | [part21-speculative.md](part21-speculative.md) |
| XXII | Performance model | [part22-performance-model.md](part22-performance-model.md) |
| XXIII | Benchmark suite, methodology, demo, profiling and explain | [part23-benchmarks.md](part23-benchmarks.md) |
| XXIV | The first ten implementations | [part24-roadmap-now.md](part24-roadmap-now.md) |
| XXV | The 12-month roadmap | [part25-roadmap-12-months.md](part25-roadmap-12-months.md) |
| XXVI | The 4-year research roadmap and staffing | [part26-roadmap-4-years.md](part26-roadmap-4-years.md) |
| XXVII | Moat, IP and the acquisition package | [part27-moat.md](part27-moat.md) |
| XXVIII | Risks and reasons Sky may fail | [part28-risks.md](part28-risks.md) |
| — | Top 50 ideas and top 10 breakthrough bets | [top50.md](top50.md) |
| — | The first 30 GitHub issues | [issues.md](issues.md) |
| — | Decision record (the spine: D1–D21, targets, gates) | [SPINE.md](SPINE.md) |
| — | Roofline model (every physics number) | [roofline.py](roofline.py) |
| — | Research notes R1–R10, with raw measurements and microbenchmarks | [research/](research/README.md) |

## How to read the evidence labels

Every non-obvious claim carries one of four labels:

| Label | Meaning |
|---|---|
| `[VERIFIED: source]` | Supported by source code (`path:line` at a recorded commit), an official document, a paper, a maintainer statement, or our own measurement with its command and raw output. |
| `[LIKELY: reason]` | A strong engineering inference; the reason is given. |
| `[EXPERIMENTAL: what decides]` | Needs a test, and the test is named. |
| `[MOONSHOT]` | High risk, high upside. |

Two further rules:
- No number in this report is invented. Each one is measured (with its source), computed by [`roofline.py`](roofline.py), or stated as an assumption.
- Numbers that need hardware we do not have yet are labeled as such. The plan measures them first (Part IV §12).

## How this report was produced

1. Ten research agents each took one area and wrote the notes in [`research/`](research/README.md):
   - R1: Sky today;
   - R2: tinygrad;
   - R3: MLX core;
   - R4: mlx-lm;
   - R5: llama.cpp;
   - R6: Apple silicon and Metal 4;
   - R7: compiler systems;
   - R8: inference algorithms;
   - R9: formats and benchmark methodology;
   - R10: Sky's CPU performance.
2. They ran every timing measurement on the shared M2 Pro behind a machine-wide lock, so no two benchmarks overlapped.
3. An architect fixed the decisions in a shared design document (the spine).
4. Chapter writers expanded the design from the notes. Where a writer's evidence contradicted the spine, the chapter says so in a "Disagreement with the spine" section, and the spine was corrected where the evidence was right.

The measurement scripts, raw outputs and Metal microbenchmark sources are kept under [`research/`](research/README.md), so every measured number can be reproduced.
