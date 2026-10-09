**Response quality on Heavy (50 common turns of the 50-turn subset).**

| System | D-1 | D-2 | B-1 | B-2 | B-3 | F1 | R-L | chrF | BERTScore |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Zero-shot | 34.15 | 71.48 | 13.20 | 4.52 | **2.19** | 8.93 | 7.50 | 18.06 | 63.23 |
| Few-shot CoT | 30.56 | 69.49 | 9.94 | 2.63 | 1.18 | 6.05 | 5.50 | 16.34 | 59.53 |
| MultiAgentESC | **35.66** | 73.35 | 10.59 | 3.05 | 1.28 | 6.65 | 5.66 | 16.97 | 60.77 |
| Translate-Pivot | 31.79 | 71.08 | 15.09 | **5.00** | 1.99 | 11.19 | 8.62 | **19.13** | 66.79 |
| CodeMixESC | 32.86 | **75.00** | **15.28** | 4.77 | 1.92 | **12.13** | **9.24** | 18.97 | **66.98** |

Distinct-n (D), corpus BLEU-n (B), unigram F1, ROUGE-L (R-L), corpus chrF and multilingual BERTScore F1, all ×100; best per column in bold.
