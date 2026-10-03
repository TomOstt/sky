# Thu Oct  1 20:31:12 IDT 2026 load: load averages: 6.55 6.53 5.45
## /private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad/src/llama.cpp/build/bin/llama-bench -m /private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad/models/Qwen3-8B-Q4_0.gguf -p 1,2,3,4,5,6,7,8,16 -n 0 -fa on -r 10 -o md
| model                          |       size |     params | backend    | threads |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --: | --------------: | -------------------: |
| qwen3 8B Q4_0                  |   4.45 GiB |     8.19 B | MTL,BLAS   |       6 |   1 |             pp1 |         36.32 ± 0.67 |
| qwen3 8B Q4_0                  |   4.45 GiB |     8.19 B | MTL,BLAS   |       6 |   1 |             pp2 |         47.65 ± 0.10 |
| qwen3 8B Q4_0                  |   4.45 GiB |     8.19 B | MTL,BLAS   |       6 |   1 |             pp3 |         65.47 ± 0.44 |
| qwen3 8B Q4_0                  |   4.45 GiB |     8.19 B | MTL,BLAS   |       6 |   1 |             pp4 |         69.71 ± 0.16 |
| qwen3 8B Q4_0                  |   4.45 GiB |     8.19 B | MTL,BLAS   |       6 |   1 |             pp5 |         74.67 ± 0.20 |
| qwen3 8B Q4_0                  |   4.45 GiB |     8.19 B | MTL,BLAS   |       6 |   1 |             pp6 |         73.66 ± 2.58 |
| qwen3 8B Q4_0                  |   4.45 GiB |     8.19 B | MTL,BLAS   |       6 |   1 |             pp7 |         65.53 ± 3.12 |
| qwen3 8B Q4_0                  |   4.45 GiB |     8.19 B | MTL,BLAS   |       6 |   1 |             pp8 |         77.34 ± 0.38 |
| qwen3 8B Q4_0                  |   4.45 GiB |     8.19 B | MTL,BLAS   |       6 |   1 |            pp16 |        135.54 ± 0.40 |

build: 81e39ad (1)
## /private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad/src/llama.cpp/build/bin/llama-bench -m /private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad/models/Qwen3-8B-Q4_K_M.gguf -p 1,2,3,4,5,6,7,8,16 -n 0 -fa on -r 10 -o md
| model                          |       size |     params | backend    | threads |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --: | --------------: | -------------------: |
| qwen3 8B Q4_K - Medium         |   4.68 GiB |     8.19 B | MTL,BLAS   |       6 |   1 |             pp1 |         31.63 ± 0.25 |
| qwen3 8B Q4_K - Medium         |   4.68 GiB |     8.19 B | MTL,BLAS   |       6 |   1 |             pp2 |         37.63 ± 0.34 |
| qwen3 8B Q4_K - Medium         |   4.68 GiB |     8.19 B | MTL,BLAS   |       6 |   1 |             pp3 |         39.08 ± 0.03 |
| qwen3 8B Q4_K - Medium         |   4.68 GiB |     8.19 B | MTL,BLAS   |       6 |   1 |             pp4 |         50.95 ± 0.42 |
| qwen3 8B Q4_K - Medium         |   4.68 GiB |     8.19 B | MTL,BLAS   |       6 |   1 |             pp5 |         52.85 ± 0.20 |
| qwen3 8B Q4_K - Medium         |   4.68 GiB |     8.19 B | MTL,BLAS   |       6 |   1 |             pp6 |         52.90 ± 0.21 |
| qwen3 8B Q4_K - Medium         |   4.68 GiB |     8.19 B | MTL,BLAS   |       6 |   1 |             pp7 |         51.89 ± 0.12 |
| qwen3 8B Q4_K - Medium         |   4.68 GiB |     8.19 B | MTL,BLAS   |       6 |   1 |             pp8 |         56.99 ± 0.21 |
| qwen3 8B Q4_K - Medium         |   4.68 GiB |     8.19 B | MTL,BLAS   |       6 |   1 |            pp16 |        117.36 ± 0.31 |

build: 81e39ad (1)
## /private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad/src/llama.cpp/build/bin/llama-bench -m /private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad/models/Qwen3-0.6B-Q4_0.gguf -p 1,2,3,4,5,6,7,8,16 -n 0 -fa on -r 10 -o md
| model                          |       size |     params | backend    | threads |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --: | --------------: | -------------------: |
| qwen3 0.6B Q4_0                | 358.78 MiB |   596.05 M | MTL,BLAS   |       6 |   1 |             pp1 |       233.30 ± 23.48 |
| qwen3 0.6B Q4_0                | 358.78 MiB |   596.05 M | MTL,BLAS   |       6 |   1 |             pp2 |        383.95 ± 3.69 |
| qwen3 0.6B Q4_0                | 358.78 MiB |   596.05 M | MTL,BLAS   |       6 |   1 |             pp3 |        535.73 ± 2.93 |
| qwen3 0.6B Q4_0                | 358.78 MiB |   596.05 M | MTL,BLAS   |       6 |   1 |             pp4 |        608.80 ± 2.79 |
| qwen3 0.6B Q4_0                | 358.78 MiB |   596.05 M | MTL,BLAS   |       6 |   1 |             pp5 |        685.84 ± 2.75 |
| qwen3 0.6B Q4_0                | 358.78 MiB |   596.05 M | MTL,BLAS   |       6 |   1 |             pp6 |        694.79 ± 2.10 |
| qwen3 0.6B Q4_0                | 358.78 MiB |   596.05 M | MTL,BLAS   |       6 |   1 |             pp7 |        674.95 ± 7.53 |
| qwen3 0.6B Q4_0                | 358.78 MiB |   596.05 M | MTL,BLAS   |       6 |   1 |             pp8 |       749.15 ± 16.06 |
| qwen3 0.6B Q4_0                | 358.78 MiB |   596.05 M | MTL,BLAS   |       6 |   1 |            pp16 |       1112.58 ± 3.26 |

build: 81e39ad (1)
# end Thu Oct  1 20:31:37 IDT 2026
