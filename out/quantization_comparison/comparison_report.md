# Quantization Comparison (TODO 6.5)

| Plan | Precision | Accuracy | Parse rate | GPU | Model MiB | P50 ms | tok/s | GPU h / 1M tok | L2 candidate |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| fp16-baseline | FP16 | 0.009060 | 1.0000 | NVIDIA A100-SXM4-80GB | 1676.7 | 777.1 | 220.9 | 1.258 | baseline |
| awq-int4-w4a16-g32 | INT4 W4A16 | 0.011325 | 1.0000 | NVIDIA A100-SXM4-40GB | 1680.0 | 1311.6 | 131.7 | 2.110 | True |
| awq-int4-w4a16-g64 | INT4 W4A16 | 0.005663 | 0.9977 | NVIDIA A100-SXM4-40GB | 1667.1 | 1312.5 | 131.6 | 2.110 | True |
| awq-int4-w4a16-g128 | INT4 W4A16 | 0.006795 | 1.0000 | NVIDIA A100-SXM4-40GB | 1660.5 | 1318.0 | 131.0 | 2.120 | True |
| gptq-int4-w4a16-g32 | INT4 W4A16 | 0.004530 | 1.0000 | NVIDIA A100-SXM4-40GB | 1679.5 | 1069.1 | 160.9 | 1.726 | True |
| gptq-int4-w4a16-g64 | INT4 W4A16 | 0.007928 | 0.9989 | NVIDIA A100-SXM4-40GB | 1668.8 | 1054.1 | 162.6 | 1.709 | True |
| gptq-int4-w4a16-g128 | INT4 W4A16 | 0.011325 | 0.9887 | NVIDIA A100-SXM4-40GB | 1663.5 | 1057.1 | 162.9 | 1.705 | False |
| smoothquant-int8-w8a8 | INT8 W8A8 | 0.006795 | 0.9604 | NVIDIA A100-SXM4-80GB | 1768.9 | 3080.9 | 56.4 | 4.922 | False |
| fp8-not-run | FP8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | False |

## Conclusion

The FP16 baseline is not production-quality, INT8 performance is missing, FP8 is unavailable, and FP16/INT4 benchmarks use different hardware or matrices. No deployment winner can be selected.

Performance deltas against FP16 are intentionally blank unless hardware and benchmark matrices match.
Monetary cost remains `not_run` until approved hourly GPU prices are provided.
