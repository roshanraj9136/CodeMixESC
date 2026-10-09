**Strategy stability under code-mixing on the common turns: Jensen–Shannon divergence (bits) between the predicted-strategy distributions of EN and of the Hinglish version, over the 8 strategies + None (JSD) and over the 8 strategies on turns where both versions predicted one (JSD-S), and per-turn agreement in %.**

| System | EN→Light JSD ↓ | EN→Light JSD-S ↓ | EN→Light Agree % ↑ | EN→Heavy JSD ↓ | EN→Heavy JSD-S ↓ | EN→Heavy Agree % ↑ |
|:--|--:|--:|--:|--:|--:|--:|
| Few-shot CoT | **0.004** | **0.004** | **86.0** | **0.005** | **0.005** | **86.0** |
| MultiAgentESC | 0.014 | 0.023 | 60.0 | 0.081 | 0.122 | 54.0 |
| Translate-Pivot † | 0.015 | 0.038 | 66.0 | 0.016 | 0.022 | 66.0 |
| CodeMixESC | 0.022 | 0.029 | 70.0 | 0.016 | 0.039 | 62.0 |

† compared with MultiAgentESC on EN (the pivot runs MultiAgentESC on translations).
