# Thu Oct  1 20:19:13 IDT 2026 load: load averages: 6.10 4.85 3.98
## env: BASE
| model                          |       size |     params | backend    | threads |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --: | --------------: | -------------------: |
| qwen3 0.6B Q4_0                | 358.78 MiB |   596.05 M | MTL,BLAS   |       6 |   1 |           tg128 |        250.22 ± 2.38 |

build: 81e39ad (1)
## env: GGML_METAL_FUSION_DISABLE=1
| model                          |       size |     params | backend    | threads |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --: | --------------: | -------------------: |
| qwen3 0.6B Q4_0                | 358.78 MiB |   596.05 M | MTL,BLAS   |       6 |   1 |           tg128 |        248.85 ± 1.37 |

build: 81e39ad (1)
## env: GGML_METAL_CONCURRENCY_DISABLE=1
| model                          |       size |     params | backend    | threads |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --: | --------------: | -------------------: |
| qwen3 0.6B Q4_0                | 358.78 MiB |   596.05 M | MTL,BLAS   |       6 |   1 |           tg128 |        239.60 ± 6.40 |

build: 81e39ad (1)
## env: GGML_METAL_GRAPH_OPTIMIZE_DISABLE=1
| model                          |       size |     params | backend    | threads |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --: | --------------: | -------------------: |
| qwen3 0.6B Q4_0                | 358.78 MiB |   596.05 M | MTL,BLAS   |       6 |   1 |           tg128 |        252.78 ± 2.25 |

build: 81e39ad (1)
## env: LLAMA_GRAPH_REUSE_DISABLE=1
| model                          |       size |     params | backend    | threads |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --: | --------------: | -------------------: |
| qwen3 0.6B Q4_0                | 358.78 MiB |   596.05 M | MTL,BLAS   |       6 |   1 |           tg128 |       144.09 ± 17.78 |

build: 81e39ad (1)
# end Thu Oct  1 20:19:29 IDT 2026
