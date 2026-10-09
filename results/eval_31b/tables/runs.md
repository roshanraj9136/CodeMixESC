**Runs: records, coverage of the sampled subset, missing turns (expected but absent), failed turns (error field), 'None' responses, LLM calls that returned an empty or blocked answer and calls cut at the token limit (totals), duplicate uids, unreadable lines, and records whose reference or post differs from the current test data (stale).**

| System | Version | Records | Subset % | Missing | Failed | None | Empty calls | Truncated calls | Dup. | Unread. | Stale | In tables | Model | δ |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|:--|--:|
| maesc | Light | 98 | 98.0 | 2 | 0 | 0 | 0 | 71 | 0 | 0 | 0 | yes | gemma-4-31b-it | – |
| pivot | Light | 98 | 98.0 | 2 | 0 | 0 | 0 | 70 | 0 | 0 | 0 | yes | gemma-4-31b-it | – |
| codemixesc | Light | 97 | 97.0 | 3 | 0 | 0 | 0 | 73 | 0 | 0 | 0 | yes | gemma-4-31b-it | 0.20 |
| cmx_noft | Light | 48 | 48.0 | 52 | 0 | 0 | 0 | 15 | 0 | 0 | 0 | no | – | – |
| cmx_nogate | Light | 97 | 97.0 | 3 | 0 | 0 | 0 | 73 | 0 | 0 | 0 | yes | gemma-4-31b-it | 0.20 |
