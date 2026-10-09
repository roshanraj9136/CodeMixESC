**Single-call baselines on all test turns and on the sampled subset (representativeness check).**

| Version | System | Turns | D-1 | D-2 | B-2 | F1 | R-L | chrF | BERTScore | CMI gap |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| EN | Zero-shot (all turns) | 100 | 22.53 | 60.36 | 5.31 | 15.24 | 11.82 | 19.50 | 66.68 | 0.022 |
|  | Zero-shot (subset) | 50 | 28.78 | 68.38 | 3.83 | 13.04 | 10.22 | 18.99 | 65.94 | 0.025 |
|  | Few-shot CoT (all turns) | 100 | 23.21 | 60.86 | 5.29 | 15.20 | 12.40 | 18.74 | 66.78 | 0.023 |
|  | Few-shot CoT (subset) | 50 | 30.36 | 68.92 | 3.86 | 13.41 | 10.90 | 19.02 | 66.15 | 0.024 |
| Light | Zero-shot (all turns) | 50 | 26.08 | 61.21 | 7.70 | 14.28 | 11.58 | 19.86 | 66.41 | 0.118 |
|  | Zero-shot (subset) | 50 | 26.08 | 61.21 | 7.70 | 14.28 | 11.58 | 19.86 | 66.41 | 0.118 |
|  | Few-shot CoT (all turns) | 50 | 27.23 | 66.41 | 6.81 | 14.61 | 11.98 | 19.28 | 65.74 | 0.158 |
|  | Few-shot CoT (subset) | 50 | 27.23 | 66.41 | 6.81 | 14.61 | 11.98 | 19.28 | 65.74 | 0.158 |
| Heavy | Zero-shot (all turns) | 50 | 34.15 | 71.48 | 4.52 | 8.93 | 7.50 | 18.06 | 63.23 | 0.314 |
|  | Zero-shot (subset) | 50 | 34.15 | 71.48 | 4.52 | 8.93 | 7.50 | 18.06 | 63.23 | 0.314 |
|  | Few-shot CoT (all turns) | 50 | 30.56 | 69.49 | 2.63 | 6.05 | 5.50 | 16.34 | 59.53 | 0.402 |
|  | Few-shot CoT (subset) | 50 | 30.56 | 69.49 | 2.63 | 6.05 | 5.50 | 16.34 | 59.53 | 0.402 |
