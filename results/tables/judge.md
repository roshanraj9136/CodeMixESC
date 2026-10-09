**Pairwise LLM judge (gemma-4-31b-it, both response orders) on a seeded sample of turns.**

| Version | Comparison | n | Fluency | Identification | Comforting | Suggestion | Overall | Language Naturalness |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| Light | CodeMixESC vs MultiAgentESC | 50 | 0 / 100 / 0 | 16 / 70 / 14 | 60 / 18 / 22‡ | 6 / 86 / 8 | 74 / 6 / 20‡ | 76 / 18 / 6‡ |
|  | CodeMixESC vs Translate-Pivot | 50 | 0 / 100 / 0 | 8 / 78 / 14 | 44 / 24 / 32 | 4 / 88 / 8 | 44 / 24 / 32 | 46 / 40 / 14‡ |
| Heavy | CodeMixESC vs MultiAgentESC | 50 | 2 / 98 / 0 | 6 / 90 / 4 | 64 / 28 / 8‡ | 8 / 90 / 2 | 86 / 10 / 4‡ | 82 / 16 / 2‡ |
|  | CodeMixESC vs Translate-Pivot | 50 | 4 / 96 / 0 | 14 / 74 / 12 | 42 / 30 / 28 | 8 / 80 / 12 | 56 / 20 / 24† | 50 / 44 / 6‡ |

Cells: win / tie / lose in % of the turns; † p < 0.05, ‡ p < 0.01 (two-sided sign test of wins against losses).
