**Response quality on Light (50 common turns of the 50-turn subset).**

| System | D-1 | D-2 | B-1 | B-2 | B-3 | F1 | R-L | chrF | BERTScore |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Zero-shot | 26.08 | 61.21 | 19.08 | **7.70** | **3.60** | 14.28 | 11.58 | **19.86** | 66.41 |
| Few-shot CoT | 27.23 | 66.41 | **19.44** | 6.81 | 2.96 | **14.61** | **11.98** | 19.28 | 65.74 |
| MultiAgentESC | 29.83 | 67.14 | 18.03 | 6.12 | 2.05 | 13.10 | 10.45 | 18.91 | 65.96 |
| Translate-Pivot | **32.38** | **69.21** | 11.07 | 3.23 | 1.33 | 6.04 | 5.75 | 16.42 | 61.56 |
| CodeMixESC | 30.76 | 68.91 | 19.28 | 7.01 | 3.13 | 13.95 | 10.60 | 18.92 | **66.86** |

Distinct-n (D), corpus BLEU-n (B), unigram F1, ROUGE-L (R-L), corpus chrF and multilingual BERTScore F1, all ×100; best per column in bold.
