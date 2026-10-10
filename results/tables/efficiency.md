**Efficiency on the common turns: logical LLM calls and summed API latency per turn, extra calls over MultiAgentESC on the same turns, share of turns on the full multi-agent path, and response length.**

| Version | System | Calls/turn | Δ calls vs MAESC | Latency (s) | Multi-path % | Words | >30 words % |
|:--|:--|--:|--:|--:|--:|--:|--:|
| EN | Zero-shot | **1.00** | -8.80 | **2.1** | 0.0 | 23.8 | **0.0** |
|  | Few-shot CoT | **1.00** | -8.80 | 3.1 | 0.0 | 22.2 | **0.0** |
|  | MultiAgentESC | 9.80 | +0.00 | 49.6 | 62.0 | 22.3 | **0.0** |
|  | CodeMixESC | 9.74 | -0.06 | 49.5 | 62.0 | 23.1 | 4.0 |
| Light | Zero-shot | **1.00** | -8.66 | **2.0** | 0.0 | 23.1 | 0.0 |
|  | Few-shot CoT | **1.00** | -8.66 | 3.0 | 0.0 | 22.3 | 0.0 |
|  | MultiAgentESC | 9.66 | +0.00 | 60.3 | 62.0 | 22.5 | 0.0 |
|  | Translate-Pivot | 11.72 | +2.06 | 67.1 | 62.0 | 24.4 | 0.0 |
|  | CodeMixESC | 10.20 | +0.54 | 71.9 | 62.0 | 22.2 | 0.0 |
|  | w/o Register Gate | 9.72 | +0.06 | 70.9 | 62.0 | 22.3 | 0.0 |
| Heavy | Zero-shot | **1.00** | -9.22 | **2.1** | 0.0 | 23.7 | **0.0** |
|  | Few-shot CoT | **1.00** | -9.22 | 2.9 | 0.0 | 22.7 | **0.0** |
|  | MultiAgentESC | 10.22 | +0.00 | 68.0 | 66.0 | 23.2 | **0.0** |
|  | Translate-Pivot | 11.22 | +1.00 | 62.2 | 58.0 | 24.3 | **0.0** |
|  | CodeMixESC | 10.70 | +0.48 | 61.0 | 66.0 | 23.8 | **0.0** |
|  | w/o Register Gate | 10.14 | -0.08 | 59.8 | 66.0 | 23.8 | **0.0** |
|  | w/o retriever fine-tuning | 10.38 | +0.16 | 56.4 | 64.0 | 23.9 | 2.0 |
|  | w/o cross-lingual retriever | 10.60 | +0.38 | 58.3 | 66.0 | 23.1 | **0.0** |

Calls count cached calls too; latency is the API time of the original calls.
