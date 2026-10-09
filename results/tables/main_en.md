**Response quality on EN (50 common turns of the 50-turn subset).**

| System | D-1 | D-2 | B-1 | B-2 | B-3 | F1 | R-L | chrF | BERTScore |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Zero-shot | 28.78 | 68.38 | 15.61 | 3.83 | 1.49 | 13.04 | 10.22 | 18.99 | 65.94 |
| Few-shot CoT | 30.36 | 68.92 | 15.82 | 3.86 | 1.35 | 13.41 | 10.90 | 19.02 | **66.15** |
| MultiAgentESC | **31.58** | 70.08 | 16.17 | 3.55 | 1.27 | 12.85 | 10.44 | 19.01 | 65.79 |
| CodeMixESC | 30.70 | **70.77** | **16.46** | **4.57** | **2.01** | **13.48** | **11.04** | **19.65** | 65.99 |

Distinct-n (D), corpus BLEU-n (B), unigram F1, ROUGE-L (R-L), corpus chrF and multilingual BERTScore F1, all ×100; best per column in bold.
