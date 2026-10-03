# Research notes and raw measurements

This directory is the evidence base of the report in [`../`](../README.md). It contains:
- ten research notes;
- the scripts that produced every measurement;
- the raw outputs;
- the Metal microbenchmarks.

The chapters cite the notes as (R3 §7) and so on.

## The notes

| Note | Topic | Main measurements |
|---|---|---|
| [R1-sky-today.md](R1-sky-today.md) | Sky as it exists at commit `a99e7e3`: pipeline, crates, types, SIR, engines, numerics, in-flight branches | Probes of the type system; GEMV and runtime-call costs |
| [R2-tinygrad.md](R2-tinygrad.md) | tinygrad at `512e9b2`, traced through the code | Qwen3-0.6B decode; GEMV by weight format; JIT and BEAM costs |
| [R3-mlx-core.md](R3-mlx-core.md), [R3-kernels.md](R3-kernels.md), [R3-issues.md](R3-issues.md) | MLX core at `0840c42`: evaluation, Metal backend, kernels, the neural-accelerator (NAX) path, maintainers' statements | Bandwidth through MLX; `quantized_matmul` at M = 1…32; prefill GEMM; dispatch cost |
| [R4-mlx-lm.md](R4-mlx-lm.md) | mlx-lm at `53b9af3`: generation loop, Qwen3 model code, KV caches, speculation | Qwen3-0.6B / 8B decode, prefill, TTFT; dispatch census; per-token time split; speculative decoding |
| [R5-llamacpp.md](R5-llamacpp.md) | llama.cpp at `81e39ad`: Metal backend, kernels, GGUF, speculation; published Apple numbers | `llama-bench` on Qwen3-0.6B / 8B; GEMV by format; feature ablations; verify cost |
| [R6-apple-metal.md](R6-apple-metal.md) | Apple silicon M1–M5 and Metal 4: neural accelerators, MPP / `matmul2d`, the command model, the memory model, residency | Raw-Metal bandwidth; dispatch and barrier cost; ICB replay; MPP GEMM; persistent-kernel experiments |
| [R7-compilers.md](R7-compilers.md) | Triton, TVM / Ansor, Halide, XLA, Inductor, MLIR / IREE, e-graphs, megakernels, prior art on Apple GPUs, CPU codegen | Literature only |
| [R8-algorithms.md](R8-algorithms.md) | Qwen3 arithmetic, speculative decoding, exactness and batch invariance, quantization quality, KV, sparsity, MoE | Model arithmetic scripts; speculation model |
| [R9-formats-methodology.md](R9-formats-methodology.md) | safetensors / GGUF layouts and alignment, the HF ecosystem, tokenizers and templates, architectures, benchmark methodology, memory and energy metrics | Alignment checks on real files; tokenizer and template corpora; memory-metric probe; system-power measurement |
| [R10-sky-cpu-perf.md](R10-sky-cpu-perf.md) | Sky's CPU performance today: per-unit ratios, root causes, Roadmap A | Slope measurements against C, Rust and Accelerate; instruction census of native loops |

## Where the data are

| Folder | From | Contents |
|---|---|---|
| [`microbench/`](microbench/) | R6 | Swift and C++ Metal microbenchmarks (bandwidth, dispatch, ICB, MTL4, GEMM / MPP, persistent-kernel probes), a `run_all.sh`, and `results/` |
| `data/R1` | R1 | Sky probe programs and outputs |
| `data/R2` | R2 | tinygrad benchmark scripts and logs. Four raw debug logs over 1.5 MB are omitted. |
| `data/R3` | R3 | `r3_bench_*.py` scripts and `r3_out_*.txt` raw outputs; the decode-step graph census (`r3_census.dot`) |
| `data/R4` | R4 | `r4_*.py` scripts, `*.out` summaries, `raw_*.json` per-run data, the Metal dispatch-counting interposer (`r4count.m`) |
| `data/R5` | R5 | `llama-bench` scripts and raw outputs, byte-count scripts |
| `data/R7` | R7 | The Qwen3-32B budget script |
| `data/R8` | R8 | Model-arithmetic and speculation scripts, model configs, Hugging Face API listings of public speculator weights |
| `data/R9` | R9 | Alignment and layout checks, tokenizer / template corpora and results, architecture configs, memory and power probes |
| `data/R10` | R10 | Sky CPU benchmark scripts, logs, disassembly of the n-body loop |
| [`tools/`](tools/) | — | `benchlock.sh` (the machine-wide benchmark lock); `setup.sh` (environment setup); `llama-cpp-build.sh` (the llama.cpp build, with the libc++ workaround below) |

Left out on purpose:
- third-party text: Apple documentation and WWDC transcripts, GitHub issue dumps, paper text, copies of other projects' source code;
- compiled binaries;
- logs over 1.5 MB.

The notes quote what they rely on and link to the primary sources.

## How the measurements were taken

| Item | Value |
|---|---|
| Machine | Apple M2 Pro: 10 CPU cores, 16-core GPU (Apple GPU family 8, `applegpu_g14s`), 16 GB unified memory, 200 GB/s spec bandwidth |
| OS and tooling | macOS 26.6.1, Command Line Tools only (no Xcode); MSL compiled at run time with `makeLibrary(source:)` |
| Versions | MLX 0.32.3, mlx-lm 0.32.0 (Python 3.12 venv); llama.cpp `81e39ad` (Metal, shaders embedded); tinygrad `512e9b2`; Sky `a99e7e3` built with the pinned Rust 1.85.1 |
| Models | `mlx-community/Qwen3-0.6B-4bit` and `Qwen3-8B-4bit` (MLX affine 4-bit, group 64, bf16 scales); `Qwen3-0.6B-Q4_0.gguf` (unsloth) and `Qwen3-8B-Q4_0.gguf` (bartowski); Qwen's own `Qwen3-8B-Q4_K_M.gguf` |
| Benchmark lock | Every timing run went through [`tools/benchlock.sh`](tools/benchlock.sh), a machine-wide lock, so no two benchmarks overlapped. |
| Caveats | Other non-benchmark work (compiles, downloads) sometimes ran at the same time. Three displays were attached. The GPU clock was not pinned. µs-scale numbers vary 10–20% between runs. The notes give medians of ≥ 5 runs after warmup and state these caveats where they matter. |

**Paths inside the notes.** The notes were written in a scratch workspace and refer to it as `$S`:

| Path in a note | Where it is now |
|---|---|
| `$S/research/…` | this directory (raw data under `data/`) |
| `$S/src/<project>/…` | the upstream repository at the commit recorded in the table above |

Code citations such as `mlx/backend/metal/kernels/quantized.h:756` are relative to those repositories.

**A toolchain note.** On this machine the Command Line Tools ship a stale, partial libc++ header directory (`/Library/Developer/CommandLineTools/usr/include/c++/v1`) that hides the SDK's headers. C++ builds fail with `'cstdio' file not found` or `'array' file not found`. The workaround used for llama.cpp, and needed by any C++ in Sky's build on such machines, is:

```text
-nostdinc++ -isystem $(xcrun --show-sdk-path)/usr/include/c++/v1
```
