**Efficiency on the common turns: logical LLM calls and summed API latency per turn, extra calls over MultiAgentESC on the same turns, share of turns on the full multi-agent path, and response length.**

| Version | System | Calls/turn | Δ calls vs MAESC | Latency (s) | Multi-path % | Words | >30 words % |
|:--|:--|--:|--:|--:|--:|--:|--:|
| Light | MultiAgentESC | **6.31** | +0.00 | **240.7** | 32.0 | 22.6 | 0.0 |
|  | Translate-Pivot | 8.03 | +1.72 | 302.0 | 32.0 | 24.1 | 0.0 |
|  | CodeMixESC | 6.51 | +0.20 | 250.5 | 32.0 | 22.2 | 0.0 |
|  | w/o Register Gate | 6.35 | +0.04 | 245.3 | 32.0 | 22.5 | 0.0 |

Calls count cached calls too; latency is the API time of the original calls.
