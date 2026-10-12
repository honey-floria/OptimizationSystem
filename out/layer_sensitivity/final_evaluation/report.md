# TODO 7.1 Layer Sensitivity Candidate Validation Report

Generated: 2026-10-12T09:53:08.379181+00:00

## Analysis scope

- Analysis scope: `candidate_subset`
- Evaluated candidate layers: `[10, 22, 23]`
- Candidate coverage: 3/28 layers
- Claim boundary: candidate validation only; this is not a complete 28-layer ranking

## Validation

- Full dev: 883 rows
- Contextual high risk: 91 rows
- Long context: 89 rows
- Determinism reruns: exact match
- Ranking conclusion strength: low

## Candidate ranking

| Rank | Layer | Full dev | Recovered | Regressed | High-risk recovered | Long-context recovered |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 23 | 344/883 | 22 | 6 | 0 | 1 |
| 2 | 10 | 330/883 | 12 | 10 | 1 | 1 |
| 3 | 22 | 328/883 | 15 | 15 | 1 | 1 |

## Recommended protection set

`[23, 10]`

The ranking is quality-first. Weight and activation errors remain diagnostic tie-break evidence.
