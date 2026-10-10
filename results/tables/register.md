**Register match of the responses to the seeker on the common turns: mean CMI gap |CMI_r − CMI_s| and Hindi-fraction gap (lower is better), and script consistency in % (higher is better).**

| System | EN CMI gap ↓ | EN Hi-frac gap ↓ | EN Script % ↑ | Light CMI gap ↓ | Light Hi-frac gap ↓ | Light Script % ↑ | Heavy CMI gap ↓ | Heavy Hi-frac gap ↓ | Heavy Script % ↑ |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Zero-shot | 0.025 | 0.025 | 100.0 | **0.118** | **0.130** | 100.0 | 0.314 | 0.378 | 100.0 |
| Few-shot CoT | **0.024** | **0.024** | 100.0 | 0.158 | 0.158 | 100.0 | 0.402 | 0.515 | 100.0 |
| MultiAgentESC | 0.028 | 0.028 | 100.0 | 0.141 | 0.153 | 100.0 | 0.357 | 0.452 | 100.0 |
| Translate-Pivot | – | – | – | 0.136 | 0.521 | 100.0 | 0.208 | 0.223 | 100.0 |
| CodeMixESC | 0.025 | 0.025 | 100.0 | 0.137 | 0.144 | 100.0 | 0.120 | 0.135 | 100.0 |
| w/o Register Gate | – | – | – | 0.148 | 0.167 | 100.0 | 0.136 | 0.175 | 100.0 |
| w/o retriever fine-tuning | – | – | – | – | – | – | 0.116 | 0.129 | 100.0 |
| w/o cross-lingual retriever | – | – | – | – | – | – | **0.099** | **0.115** | 100.0 |
| Gold reference | 0.026 | 0.026 | 100.0 | 0.063 | 0.070 | 100.0 | 0.076 | 0.110 | 100.0 |

Mean seeker CMI_s: EN 0.019, Light 0.181, Heavy 0.420. Gold reference: the dataset's own supporter turns. Profiler: HingBERT-LID.
