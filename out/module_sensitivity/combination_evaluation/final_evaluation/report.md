# TODO 7.2 Module Sensitivity Candidate Report

Generated: 2026-10-14T17:31:56.219778+00:00

## Claim boundary

Only the frozen promoted subset is ranked. Weight experiments use fake-quant and do not prove deployment latency or memory gains.

## Attention/MLP recovery ranking

| Rank | Plan | Full dev | Net | High-risk net | Long-context net |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `candidate_l10_mlp_lm_head_embedding_int8` | 339/883 | 11 | 0 | 0 |
| 2 | `candidate_l10_mlp_lm_head_int8` | 337/883 | 9 | 0 | 0 |

## Special module / KV Cache loss-benefit ranking

| Rank | Plan | Full dev net | High-risk net | Benefit bytes |
| ---: | --- | ---: | ---: | ---: |

## Candidate protection policy

Protected recovery plans: `['candidate_l10_mlp_lm_head_embedding_int8', 'candidate_l10_mlp_lm_head_int8']`

Special module precisions: `{}`

KV Cache precision: `fp16`
