# TODO 7.2 Module Sensitivity Candidate Report

Generated: 2026-10-14T15:42:45.095022+00:00

## Claim boundary

Only the frozen promoted subset is ranked. Weight experiments use fake-quant and do not prove deployment latency or memory gains.

## Attention/MLP recovery ranking

| Rank | Plan | Full dev | Net | High-risk net | Long-context net |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `layer_10_mlp_int8` | 338/883 | 10 | 0 | 0 |
| 2 | `layer_23_mlp_int8` | 332/883 | 4 | 0 | 0 |
| 3 | `layer_26_mlp_int8` | 332/883 | 4 | 0 | 0 |
| 4 | `layer_06_attn_int8` | 332/883 | 4 | 0 | -1 |
| 5 | `layer_23_attn_int8` | 329/883 | 1 | 0 | 1 |
| 6 | `layer_25_mlp_int8` | 328/883 | 0 | 1 | 1 |
| 7 | `layer_17_mlp_int8` | 325/883 | -3 | 0 | -1 |

## Special module / KV Cache loss-benefit ranking

| Rank | Plan | Full dev net | High-risk net | Benefit bytes |
| ---: | --- | ---: | ---: | ---: |
| 1 | `lm_head_int8` | 3 | 1 | 544693248 |
| 2 | `embedding_int8` | 3 | 0 | 544693248 |
| 3 | `embedding_int4` | 0 | 1 | 783433728 |
| 4 | `lm_head_int4` | -16 | 1 | 783433728 |

## Candidate protection policy

Protected recovery plans: `['layer_10_mlp_int8', 'layer_23_mlp_int8', 'layer_26_mlp_int8', 'layer_23_attn_int8']`

Special module precisions: `{'lm_head_quantized': 'int8', 'embedding_quantized': 'int8'}`

KV Cache precision: `fp16`
