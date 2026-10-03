# Thu Oct  1 20:19:29 IDT 2026 load: load averages: 6.48 5.01 4.06
## env: BASE
| model                          |       size |     params | backend    | threads |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --: | --------------: | -------------------: |
| qwen3 8B Q4_0                  |   4.45 GiB |     8.19 B | MTL,BLAS   |       6 |   1 |           tg128 |         35.95 ± 0.34 |

build: 81e39ad (1)
## env: GGML_METAL_FUSION_DISABLE=1
| model                          |       size |     params | backend    | threads |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --: | --------------: | -------------------: |
| qwen3 8B Q4_0                  |   4.45 GiB |     8.19 B | MTL,BLAS   |       6 |   1 |           tg128 |         34.32 ± 0.82 |

build: 81e39ad (1)
## env: GGML_METAL_CONCURRENCY_DISABLE=1
| model                          |       size |     params | backend    | threads |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --: | --------------: | -------------------: |
| qwen3 8B Q4_0                  |   4.45 GiB |     8.19 B | MTL,BLAS   |       6 |   1 |           tg128 |         35.13 ± 0.09 |

build: 81e39ad (1)
## env: GGML_METAL_GRAPH_OPTIMIZE_DISABLE=1
| model                          |       size |     params | backend    | threads |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --: | --------------: | -------------------: |
| qwen3 8B Q4_0                  |   4.45 GiB |     8.19 B | MTL,BLAS   |       6 |   1 |           tg128 |         35.79 ± 0.14 |

build: 81e39ad (1)
## env: LLAMA_GRAPH_REUSE_DISABLE=1
| model                          |       size |     params | backend    | threads |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --: | --------------: | -------------------: |
| qwen3 8B Q4_0                  |   4.45 GiB |     8.19 B | MTL,BLAS   |       6 |   1 |           tg128 |         34.18 ± 0.15 |

build: 81e39ad (1)
# end Thu Oct  1 20:20:27 IDT 2026
