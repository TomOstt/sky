# R6 Metal microbenchmarks (Apple M2 Pro, macOS 26.6.1)

Every benchmark compiles its MSL at runtime with `MTLDevice.makeLibrary(source:options:)`, because no offline
`metal` compiler is installed (Command Line Tools with SDK 15.4 only). Raw outputs are in `results/`.

## Machine and conditions

- **Hardware:** Apple M2 Pro with a 16-core GPU (`applegpu_g14s`, Apple8) and 16 GB of LPDDR5. Apple's spec is 200 GB/s.
- **OS:** macOS 26.6.1 (25G76). The GPU supports `MTLGPUFamily.metal4`.
- **Displays:** three were attached during all runs: 2× external 4K at 100 Hz plus the built-in panel. WindowServer composition therefore competes for a few percent of the GPU and DRAM.
- **Clocks:** the GPU clock was not pinned. Each program runs a warm-up spin of about 300–500 ms first.
- **Locking:** every timed run went through `bash $S/benchlock.sh …`, the machine-wide lock shared with the other agents.

## Build

```bash
S=/private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad
cd $S/research/microbench && mkdir -p bin results
for b in probe bw dispatch icb persist; do
  swiftc -O -parse-as-library common.swift $b.swift -o bin/$b -framework Metal
done
swiftc -O -parse-as-library common.swift gemm.swift -o bin/gemm -framework Metal -framework MetalPerformanceShaders
```

### Metal 4 (MTL4*) benchmark

SDK 15.4 has no `MTL4*` headers, so this benchmark uses Apple's metal-cpp from github.com/apple/metal-cpp, tag `release/metal-cpp_macOS26_iOS26`, cloned into `$S/research/R6-work/metal-cpp`. The CLT libc++ path is broken, so the build passes it explicitly:

```bash
MC=$S/research/R6-work/metal-cpp; (cd $MC && git checkout -q release/metal-cpp_macOS26_iOS26)
SDK=$(xcrun --show-sdk-path)
clang++ -std=c++17 -O2 -nostdinc++ -isystem $SDK/usr/include/c++/v1 -I$MC mtl4.cpp \
  -framework Metal -framework Foundation -framework QuartzCore -Wl,-undefined,dynamic_lookup -o bin/mtl4
```

## Run (each under the bench lock, each < 30 s)

```bash
./run_all.sh mtl4 dispatch icb bw            # writes results/<name>.txt
bash $S/benchlock.sh ./bin/probe   > results/probe.txt
bash $S/benchlock.sh ./bin/gemm    > results/gemm.txt
bash $S/benchlock.sh ./bin/gemm check        # correctness of MPP GEMMs at 4096^3 -> results/gemm_check.txt
bash $S/benchlock.sh ./bin/bw confirm        # interleaved scalar-vs-vector read recheck -> results/bw_confirm.txt
bash $S/benchlock.sh ./bin/persist > results/persist.txt
```

## Files

| File | What it measures |
|---|---|
| `common.swift` | Shared helpers: device, runtime MSL compile, pipelines, GPU-time, median, warm-up. |
| `probe.swift` | Feature probe. Covers GPU families, limits and counters. Checks which MTL4/MTLTensor ObjC classes exist and creates an MTL4 queue, allocator and command buffer. Compiles MSL 3.2/4.0/4.1, `<metal_tensor>` and MPP `matmul2d` (half, bfloat, int4, fp8), then runs and verifies the half/bfloat/int4 GEMMs. |
| `bw.swift` | Read, copy and write bandwidth over 1 GiB, sweeping load width, unroll, threadgroup size, grid and storage mode. Also re-reads at working-set sizes from 1 MiB to 1 GiB (cache effects), times an MLX-style 4-bit GEMV (group 64, fp16 scale and bias), and has a `confirm` mode. |
| `dispatch.swift` | Metal 3 dispatch overhead: 1/10/100/1000 tiny dependent dispatches across 7 encoding modes, plus untracked buffers and a "small kernel" variant. Also measures submission latency (empty CB, trivial CB, shared event, completion handler) and the GPU gap between command buffers. |
| `mtl4.cpp` | The same tiny-dispatch chain through `MTL4CommandQueue` / `MTL4CommandBuffer` / `MTL4ComputeCommandEncoder`, with an argument table, residency set and intra-pass barriers. GPU time comes from `MTL4CommitFeedback`. A Metal 3 baseline is built from the same C++ code. |
| `icb.swift` | Indirect command buffer: 1000 commands encoded once on the CPU, then replayed with `executeCommandsInBuffer`, with and without `setBarrier`. Compares against direct encoding. Also GPU-side ICB encoding (`compute_command` in MSL). |
| `gemm.swift` | `simdgroup_matrix` register-only peak, MPP `matmul2d` GEMM 4096³ (half×half→float, half×int4→float) across tile shapes, `MPSMatrixMultiplication` f16, and a full-matrix correctness check (`check`). |
| `persist.swift` | Device-wide barrier inside one dispatch, built from atomics and fences. Tests several spin-load variants, co-residency limits and cross-threadgroup visibility. Every spin is bounded, so the GPU never hangs. |
| `run_all.sh` | Runs a list of benchmarks sequentially, each under the bench lock. |
