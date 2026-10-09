**Ablations of CodeMixESC on the common turns.**

| Version | System | D-2 | B-2 | F1 | R-L | chrF | BERTScore | CMI gap ↓ | Script % ↑ |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| Light | CodeMixESC | 68.91 | **7.01** | **13.95** | **10.60** | **18.92** | **66.86** | **0.137** | 100.0 |
|  | w/o Register Gate | **69.49** | 6.87 | 13.73 | 10.58 | 18.88 | 66.54 | 0.148 | 100.0 |
| Heavy | CodeMixESC | 75.00 | **4.77** | **12.13** | **9.24** | **18.97** | **66.98** | **0.120** | 100.0 |
|  | w/o Register Gate | **75.30** | 4.73 | 11.86 | 9.10 | 18.92 | 66.86 | 0.136 | 100.0 |

Quality scores ×100; best per version in bold. w/o Register Gate is the same run before the gate.
