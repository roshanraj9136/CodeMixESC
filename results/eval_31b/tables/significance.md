**Paired bootstrap (2000 resamples, seed 42) of the per-turn score differences Δ = CodeMixESC − system on the common turns (sentence chrF; CMI gap: lower is better).**

| Version | CodeMixESC | n | F1 | ROUGE-L | chrF | CMI gap |
|:--|:--|--:|--:|--:|--:|--:|
| Light | vs MultiAgentESC | 97 | +0.25 [-1.06, +1.57] | +1.27 [+0.15, +2.45]† | +0.51 [-0.30, +1.41] | -0.006 [-0.022, +0.010] |
|  | vs Translate-Pivot | 97 | +7.19 [+5.55, +8.96]‡ | +5.95 [+4.60, +7.48]‡ | +2.42 [+1.55, +3.43]‡ | -0.093 [-0.117, -0.071]‡ |
|  | vs w/o Register Gate | 97 | +0.36 [-0.04, +0.87] | +0.36 [-0.02, +0.83] | +0.09 [-0.07, +0.31] | -0.024 [-0.040, -0.012]‡ |

† p < 0.05, ‡ p < 0.01 (two-sided); the Markdown and CSV versions give the 95% CIs.
