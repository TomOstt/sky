
---------------------------------------------------------------------------------------------------

## 9. Published numbers and PR history (sub-searches R5a/R5b, primary sources)

Raw material: `$S/research/tmp-R5a/` (discussion #4167 body + edit history + all comments, Apple
newsroom/spec pages, Apple ML Research page and its JS chunk) and `$S/research/tmp-R5b/` (≈150 PR/issue/
discussion JSONs; `metal-sorted.tsv` = the 402 merged PRs with "metal" in the title). `#N` =
https://github.com/ggml-org/llama.cpp/pull/N. Labels as stated by the sub-searches; numbers copied
verbatim there.

### 9.1 Discussion #4167 — https://github.com/ggml-org/llama.cpp/discussions/4167 (VERIFIED)

"Performance of llama.cpp on Apple Silicon M-series", created 2023-11-22 by ggerganov, body last edited
2026-10-01T08:43:31Z (71 edits, all ggerganov except one by slaren). Model LLaMA-2 7B; PP = pp512,
TG = tg128 (t/s). **M1–M4 rows use build 8e672ef (2023-11-21); M5/M6 rows use c1d0e7a00
(2026-08-25, "bump version to 0.3.0 (#27696)") — not comparable across that boundary** (same M2 Ultra,
Q4_0: TG 94.27 → 125.21, PP 1238 → 1489 from software alone). Current instructions:
`git checkout c1d0e7a00 && cmake -B build && cmake --build build --target llama-bench -j &&
./build/bin/llama-bench -hf ggml-org/Llama-2-7B-GGUF:F16 -hf ggml-org/Llama-2-7B-GGUF:Q8_0
-hf ggml-org/Llama-2-7B-GGUF:Q4_0 -p 512 -n 128 -ngl 99 --delay 30`.

| Chip | BW GB/s | GPU cores | F16 PP | F16 TG | Q8_0 PP | Q8_0 TG | Q4_0 PP | Q4_0 TG | build | source comment (date) |
|---|---|---|---|---|---|---|---|---|---|---|
| M1 | 68 | 7 | | | 108.21 | 7.92 | 107.81 | 14.19 | 8e672ef | 7861106 (2023-12-15) |
| M1 | 68 | 8 | | | 117.25 | 7.91 | 117.96 | 14.15 | 8e672ef | 7737353 (2023-12-02) |
| M1 Pro | 200 | 14 | 262.65 | 12.75 | 235.16 | 21.95 | 232.55 | 35.52 | 8e672ef | 7669345 / 7997936 |
| M1 Pro | 200 | 16 | 302.14 | 12.75 | 270.37 | 22.34 | 266.25 | 36.41 | 8e672ef | body |
| M1 Max | 400 | 24 | 453.03 | 22.55 | 405.87 | 37.81 | 400.26 | 54.61 | 8e672ef | 7737442 |
| M1 Max | 400 | 32 | 599.53 | 23.03 | 537.37 | 40.2 | 530.06 | 61.19 | 8e672ef | 7664196 |
| M1 Ultra | 800 | 48 | 875.81 | 33.92 | 783.45 | 55.69 | 772.24 | 74.93 | 8e672ef | 7730376 |
| M1 Ultra | 800 | 64 | 1168.89 | 37.01 | 1042.95 | 59.87 | 1030.04 | 83.73 | 8e672ef | 8089048 |
| M2 | 100 | 8 | | | 147.27 | 12.18 | 145.91 | 21.7 | 8e672ef | 8034638 |
| M2 | 100 | 10 | 201.34 | 6.72 | 181.4 | 12.21 | 179.57 | 21.91 | 8e672ef | 7644774 |
| **M2 Pro (our chip)** | 200 | 16 | 312.65 | 12.47 | 288.46 | 22.7 | **294.24** | **37.87** | 8e672ef | 7663473 |
| M2 Pro | 200 | 19 | 384.38 | 13.06 | 344.5 | 23.01 | 341.19 | 38.86 | 8e672ef | 7824473 |
| M2 Max | 400 | 30 | 600.46 | 24.16 | 540.15 | 39.97 | 537.6 | 60.99 | 8e672ef | 7711079 |
| M2 Max | 400 | 38 | 755.67 | 24.65 | 677.91 | 41.83 | 671.31 | 65.95 | 8e672ef | 7653249 |
| M2 Ultra | 800 | 60 | 1128.59 | 39.86 | 1003.16 | 62.14 | 1013.81 | 88.64 | 8e672ef | 7654488 |
| M2 Ultra | 800 | 76 | 1401.85 | 41.02 | 1248.59 | 66.64 | 1238.48 | 94.27 | 8e672ef | body |
| M3 | 100 | 10 | | | 187.52 | 12.27 | 186.75 | 21.34 | 8e672ef | 7677748 |
| M3 Pro | 150 | 14 | | | 272.11 | 17.44 | 269.49 | 30.65 | 8e672ef | 7664636 |
| M3 Pro | 150 | 18 | 357.45 | 9.89 | 344.66 | 17.53 | 341.67 | 30.74 | 8e672ef | 7804807 |
| M3 Max | 300 | 30 | 589.41 | 19.54 | 566.4 | 34.3 | 567.59 | 56.58* | 8e672ef | 7931329 (*comment says 56.85) |
| M3 Max | 400 | 40 | 779.17 | 25.09 | 757.64 | 42.75 | 759.7 | 66.31 | 8e672ef | 7656623 |
| M3 Ultra | 800 | 60 | 1121.80 | 42.24 | 1085.76 | 63.55 | 1073.09 | 88.40 | 8e672ef | 12495617 (2025-03-14) |
| M3 Ultra | 800 | 80 | 1538.34 | 39.78 | 1487.51 | 63.93 | 1471.24 | 92.14 | 8e672ef | 12594707 (2025-03-23) |
| M4 | 120 | 10 | 230.18 | 7.43 | 223.64 | 13.54 | 221.29 | 24.11 | 8e672ef | 11202046 |
| M4 Pro | 273 | 16 | 381.14 | 17.19 | 367.13 | 30.54 | 364.06 | 49.64 | 8e672ef | 11493483 |
| M4 Pro | 273 | 20 | 464.48 | 17.18 | 449.62 | 30.69 | 439.78 | 50.74 | 8e672ef | 11185614 |
| M4 Max | 410 | 32 | 736.25 | 24.29 | 718.56 | 43.87 | 713.93 | 69.95 | 8e672ef | 15514109 (2026-01-16) |
| M4 Max | 546 | 40 | 922.83 | 31.64 | 891.94 | 54.05 | 885.68 | 83.06 | 8e672ef | 11233679 (2024-11-13) |
| M5 | 154 | 10 | | | 715.42 | 18.42 | 722.79 | 31.88 | c1d0e7a | 18210351 (2026-08-30) |
| M5 Pro | 307 | 16 | 1310.78 | 21.39 | 1302.59 | 38.94 | 1340.44 | 67.59 | 60081bb | 18583406 (2026-09-24) |
| M5 Pro | 307 | 20 | 1588.78 | 21.55 | 1553.86 | 38.92 | 1620.64 | 66.33 | c1d0e7a | 18213344 (2026-08-31; build line not shown) |
| **M5 Max** | **614** | **40** | 3158.49 | 37.11 | 3143.81 | 72.42 | **3219.99** | **119.92** | c1d0e7a | 18152504 (2026-08-25, ggerganov) |
| M5 Ultra | 1228 | 80 | 5207.14 | 73.60 | 4910.55 | 119.01 | 4944.66 | 179.10 | c1d0e7a | 18683605 (2026-09-30) |
| M6 | 170 | 8 (submitter: 12) | 952.53 | 11.59 | 970.46 | 21.01 | 988.93 | 35.74 | c1d0e7a (submitter: 0c1e57098) | 18694932 (2026-10-01) |

Empty rows in the body (no data): M3 8c, M4 8c, M4 "Ultra", M5 8c, **M5 Max 32c (460 GB/s)**, M5 Ultra 64c.
Table defects: M5 BW 154 vs Apple's 153; M5 Ultra 1228 = 2×614 (Apple says "1.2TB/s"); M6 core count.

**Other M5-family comments (VERIFIED as posted):** M5 Max 40c on the *old* build 8e672ef (no Neural
Accelerators): Q4_0 987.10 / 102.93 (2026-03-28) and 990.53 / 104.03 (2026-03-31) ⇒ **same chip,
software alone: PP ×3.3, TG ×1.17**. M5 Max on 52b3df002 (2026-06-22): Q4_0 3347.44 / 119.08. M5 base
with PR #16634 (2025-10-30): Q4_0 636.36 / 31.02 vs 247.68 / 29.62 without. ggerganov (2026-03-28):
M5 numbers "should not use the old build 8e672ef… because it does not make use of the neural
accelerators"; the tensor API "is enabled by default on master… we did the implementation on M2/M4
and haven't had the chance to optimize it for M5".
**Our cross-check:** M2 Pro 16c, LLaMA-2 7B Q4_0 TG 37.87 on the 2023 build ≈ 145 GB/s of weights;
our Qwen3-8B Q4_0 on the 2026 build: 154–159 GB/s (§8 B) — consistent with the software gains above.

### 9.2 Official Apple specs (VERIFIED, apple.com)

| chip | GPU cores | memory bandwidth | AI compute claim | dates | URL |
|---|---|---|---|---|---|
| M4 Max | 32 / 40 | 410 / **546 GB/s** | – | 2024-10-30 | apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/ |
| M5 | 10 | 153 GB/s | Neural Accelerator in each GPU core; ">4x peak GPU compute vs M4" | 2025-10-15 | apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/ |
| M5 Pro | 16 / 20 | 307 GB/s, ≤64 GB | NA per GPU core | 2026-03-03 (avail. 03-11) | apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/ |
| **M5 Max** | 32 / **40** | 460 / **614 GB/s**, ≤128 GB | NA per GPU core; "over 4x the peak GPU compute for AI compared to the previous generation" | same | same |
| M5 Ultra | 64 / 80 | 1.2 TB/s, ≤512 GB | "4.5x peak GPU compute for AI vs M3 Ultra" | 2026-08-25 (avail. 09-22) | apple.com/newsroom/2026/08/apple-introduces-m6-and-m5-ultra-for-a-big-leap-in-performance-and-ai-compute/ |

Apple ML Research, "Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU" (2025-11-19,
machinelearning.apple.com/research/exploring-llms-mlx-m5): MLX "leverages the Tensor Operations
(TensorOps) and Metal Performance Primitives framework introduced with Metal 4"; M5 vs M4 MacBook Pro,
4096-token prompt: TTFT speed-up 3.3–4.1× (Qwen3-8B 4-bit 18.6 s → 4.69 s; Qwen3-14B 4-bit 35.15 →
8.66 s), generation speed-up 1.19–1.27× ("120GB/s for the M4, 153GB/s for the M5").

### 9.3 32B-class 4-bit numbers (the headline workload)

| chip | runtime | model | prefill | decode | source / date | label |
|---|---|---|---|---|---|---|
| **M5 Max 40c (48 GB)** | llama.cpp b11146 (7fe450e19, 2026-09-23), `has tensor = true`, FA on | **Qwen3-32B Q4_K_M** | pp512 **667.0** (pp128 461, pp256 631) | tg128 **26.38** | mac-llm-bench PR #12 (github.com/enescingoz/mac-llm-bench/pull/12), 2026-09-27 | LIKELY (community, method documented) |
| same machine | MLX 0.32.2 / mlx-lm 0.31.3 | **Qwen3-32B-4bit** | pp512 **730.9** | tg128 **28.41** | same | LIKELY |
| M5 Max | llama.cpp (#28948 body, ggerganov) | qwen35 27B Q4_K_M | pp2048 362.4 | tg128 26.60 | 2026-09-15 | VERIFIED |
| M4 Max 128 GB | MLX | Qwen3-32B-4bit | ≈196 (derived from TTFT 2.60 s @512) | 24.5 @0.5K … 9.7 @32K | BTankut repo, 2026-03 | LIKELY |
| M3 Ultra 256 GB | llama.cpp fork (TurboQuant), FA, f16 KV | Qwen3-32B Q4_K_M | pp512 341.9 | tg128 30.42 | fork PR #22, 2026-03-29 | LIKELY |
| M3 Max | MLX (Awni Hannun) | Qwen2.5-32B-4bit | – | 19.68 (35.74 with 0.5B draft) | mlx-examples #1155, 2024-12-14 | VERIFIED |
| M2 Ultra | llama.cpp (ggerganov, #10581) | qwen2 32B Q4_0 / Q4_K / Q8_0 | – | pp1 29.25 / 25.89 / 17.84 | 2024-11-29 | VERIFIED |

Reading (EXPERIMENTAL arithmetic on LIKELY data): on M5 Max, Qwen3-32B Q4_K_M reads ≈19.2 GB/token ⇒
26.38 t/s ≈ **506 GB/s ≈ 82 % of spec**; MLX 4-bit (≈18.4 GB/token) 28.41 t/s ≈ 523 GB/s ≈ 85 %.
**MLX leads llama.cpp by ≈8 % decode and ≈10 % pp512 on the headline workload**, partly because Q4_K_M
reads ≈5 % more bytes and uses the slower Q4_K kernel (§8 E/G). No upstream dense-32B 4-bit
`llama-bench` run on M4 Max or M5 Max exists in ggml-org repos (negative result).

### 9.4 Speculative decoding reported on Apple

* llama.cpp draft model (server #10455, 2024-11-25; ggerganov: "designed specifically with
  limited-compute devices in mind"): community results mostly **no gain** — M4 Pro Qwen2.5-Coder-32B
  + 3B draft 10.78 vs ≈11 t/s (34 % acceptance); M4 Max Llama-3.3-70B + 1B–8B drafts 7.99 vs 8.7.
  ggerganov's own M2 Ultra data (#10581) shows why: 32B Q4_K verify of 8 tokens ≈ 4.3× one token.
* llama.cpp MTP (#22673, 2026-05-16; default `--spec-draft-n-max` 16 → 3 in #23269): mixed —
  M1 Ultra Qwen3.6-27B Q8_0 17 → 25; M3 Ultra 35B-A3B 1.35×; M4 Pro 27B 0.84× and 0.69×; M3 Max 0.67×;
  M1 Max 9B 0.87×. DFlash (#27342, M5 Pro, author's run): Qwen3.8-27B Q4_K_M 10.42 → 19.31 (1.85×);
  M2 Max user: 0.97× despite 76 % acceptance. No Apple numbers in the EAGLE3/DFlash/MTP PR bodies.
* Open PRs targeting the verify-batch weakness (UNVERIFIED claims): #27441 skinny mul_mm tiles,
  M4 Max spec decoding 1.84× → 2.27×; #27350 tensor-API mat-mat at BS 2–8 "41–72 % slower than
  mul_mv_ext"; #29110 M3 Ultra "raising n_max from 1 to 2 makes decode slower"; #27776 K-quant
  mul_mv_ext slower than mul_mv on M3 Pro. **All agree with our §8 E/G finding.**
* MLX: M3 Max Qwen2.5-32B-4bit + 0.5B-8bit draft, 4 drafts: **19.68 → 35.74 t/s (1.82×)** (VERIFIED,
  mlx-examples #1155); MTP on M4 Pro 1.57× (mlx-lm #990, open); DFlash claims ≈1.95–4× (UNVERIFIED).

### 9.5 PR history: how llama.cpp got fast on Apple (VERIFIED unless noted; chip "n/s" = not stated, probably M2 Ultra)

| theme | PR (merged) | what | Apple evidence (verbatim) |
|---|---|---|---|
| Metal backend born | #1642 (2023-06-04) | first Metal backend, GEMV only; mat-mat on CPU | – |
| Q4_0 GEMV | #1775 (2023-06-09, ikawrakow) | faster Q4_0 | "Memory throughput is now ~270 GB/s, so ~90% of theoretical limit" (30c M2 Max) |
| Q4_0 GEMV layout | #2188 (2023-07-12, lshzh-ww) | `N_DST 4` rows/simdgroup, `N_SIMDGROUP 2` — **unchanged to HEAD** | M1 Max 33B "GPU Read BW 167 → 275 GB/s", "117 → 79 ms/tok" |
| Q4_0/K-quant | #2212, #2290/#2294, #2375 (Jul 2023, ikawrakow) | full block / half block per thread; faster K-quants | M2 Max 65B Q4_K 194.1 → 141.0 ms/t |
| mat-mat on GPU | #2615 (2023-08-16) | drop MPS, custom simdgroup_matrix GEMM for all quant types | "approximately 88% of the hardware ALU" |
| small batch | #10491 (2024-11-25), **#10581** (2024-12-03) | GEMV for batch ≤4, then `mul_mv_ext` for ≤8 | M2 Ultra qwen2 7B Q4_0 pp8 144.31 → 260.05 |
| per-type constants | #12569 (2025-03-26), #15857, #16074, #20399 | N_R0/N_SG per type; function constants | qwen2 32B Q6_K tg32 21.25 → 22.97 |
| FA | #5021 (2024-04-30), #13493/#13496 (2025-05-13), **#15566** (2025-08-26), #16372, #27390 | FA, mask-block skip, **nwg=32 split-K + reduce**, mask pre-pass, KV dequant for batch ≥32 | M2 Ultra gemma3 1B tg128@d32768 98.65 → 172.59 |
| FA default | **#15434** (2025-08-30) | `-fa auto` + max GPU layers by default | – |
| encode overlap | #9698 (2024-10-01), #19369 (2026-02-05) | first nodes submitted by main thread; `n_main = max(64, 10 %)` | M2 Ultra 1B tg128 255 → 270; qwen3moe tg32 100.3 → 106.7 |
| residency | #11427 → #11648 → #17766 → #24074 (2025-01 → 2026-06) | residency sets; heartbeat (500 ms → 5 ms) after macOS began unwiring GPU memory after ~1 s idle | issue #10119: M2 Ultra 8B 26.34 ms/token → **429.08 ms after a 1.4 s pause** |
| graph reuse | **#14482** (2025-07-17) | `can_reuse` graph reuse | M2 Ultra qwen2 1.5B tg32 206.6 → 215.6 |
| fusion | #14596 (2025-07-18) → #16102/#16220 → **#28164** (2026-09-11, fusion table) → #28948 (2026-09-19, MoE/SSM) | ADD chains, RMS_NORM+MUL(+ADD), table-driven, MoE routing/reduce | qwen3moe 30B tg128 70.1 → 76.7; M5 Max qwen35moe tg128 104.7 → 117.8 |
| concurrency | #2358 (2023), removed #4919 (2024), **#15929** (2025-09-13) | range-tracked concurrency + graph reorder | gemma3 270M tg128 334 → 392; qwen3moe 30B 91.5 → 101.7 |
| async + shared | #15906 (2025-09-10) | async backend, one queue; private buffers measured 0.76× ⇒ shared | – |
| code split | **#15995** (2025-09-17), #26561 (2026-08-24), #29329 (2026-09-25) | device/context/ops split; per-op libraries compiled in parallel; per-dtype FA libs | cold compile M4 Max 6.05 → 4.52 s; M2 Ultra 16.9 → 2.6 s |
| M5 tensor API | **#16634** (2025-11-06), #17087, #21048, **#20962** (2026-04-25, Apple's Developer-Ecosystem-Engineering account), #27450, **#27461** (2026-09-01) | MPP `matmul2d`; standalone 64×128 tensor kernel by Apple; fixes for silent disable on M5 (probe failing, Metal 4 language version unset) | M5 base pp512 257 → 609 (#16634); M5 Max geomean +26.4 %, F16 pp512 +95 % (#20962); M4 Max 32c → M5 Max 40c pp2048 ×4.27 (mistral3 8B Q8_0) |
| backend sampling | **#17004** (2026-01-04), #23287, #28073 | GPU samplers; Metal top-k was slow until Aug 2026 | – |
| spec decoding | #10455 (2024-11-25), #18471/#19164 (n-gram), **#22673** MTP (2026-05-16), #18039 EAGLE3, #22105 DFlash, #26570 (FA-vec tuning for verify) | see 9.4 | M4 Max FA-vec tuning: spec decode peak 1.323× |
| FA-vec tuning | #23114 (closed), #26570 (2026-08-24), #27824, **#29075** (2026-09-23) | offline per-family table; "3884 rows across 19 SKUs → 1033 rows across 4 families" | #23114 Q=2 verify kernel: M2 Ultra 0.84×, M4 Max 1.47×, M5 Max 1.75× |

**Negative history (VERIFIED):** #14570 "metal : reuse graphs" (draft, never merged): it "does not lead
to any measurable improvement… we already do the CPU/GPU overlap by submitting the first 128 graph
nodes… enough to completely mask the CPU overhead", and "`MTLIndirectCommandBuffer` won't cut it".
No MTLBinaryArchive; no ANE/CoreML backend (issue #10453 open; CoreML PR #15262 closed unmerged).

### 9.6 Maintainer statements on design (VERIFIED, links in the sub-search report)

1. ggerganov (2023-06-17, issue #1867): "llama.cpp does not use MPS matrix multiplication as it is too
   inefficient… Third-party dependencies are highly undesired… a GEMM implementation should make use of
   the existing dot product kernels… Even if this means sacrificing a bit of performance."
2. slaren (2023-12-06, discussion #4345) on MLX: "too high level for what we need in ggml. However, the
   source code has a Metal backend, and we may be able to use it to learn how to better optimize our
   Metal kernels."
3. ggerganov (2024-11-17): "We don't support group size of 64 atm (which is what I think MLX uses), so
   can't make an apples-to-apples comparison with MLX."
4. ggerganov (#4167, 2024-02-09): "I haven't found a universal solution yet to achieve optimal
   performance for all batches for all Apple devices and models."
5. #15857: runtime compilation "unlocks… significantly more optimized kernels using the specific shapes
   of the current computation… through the MTLFunctionConstant mechanism… unroll loops significantly
   better."
6. #23114 / #27668 (2026): against per-device special cases ("would quickly become unmaintainable") ⇒
   offline tuning grid; "A table per family sounds good".
7. #10218: "BF16 does not seem to provide any benefits compared to F16".
8. Process: ggerganov's 2026 Metal PRs disclose AI assistance (e.g. "pi:llama.cpp/Qwen3.8-27B");
   #27370 came out of an AI-generated draft (#25556). One maintainer + AI + crowd-sourced per-chip
   tuning (19 SKUs in ≈2 weeks) ⇒ **very high iteration velocity**; a competitor must assume
   llama.cpp closes obvious gaps within weeks once they are pointed out.
