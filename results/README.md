# Results

| Folder | What it holds |
|---|---|
| `runs/<system>/<version>.jsonl` | one record per evaluated turn (the main experiments): every agent's output, the retrieved cases, strategies, candidates, vote, Register Gate decision, LLM calls and latency; `*.meta.json` = run configuration and completeness. Agents: `gemma-4-26b-a4b-it`, temperature 0, multi-agent systems on the fixed 100-turn sample (`CODEMIX_SAMPLE_N=100`). Format: [docs/RUN_FORMAT.md](../docs/RUN_FORMAT.md) |
| `runs_31b/` | the same format for a first study with the larger `gemma-4-31b-it` agents (Light: MultiAgentESC, CodeMixESC with δ = 0.2, Translate-Pivot). The 31B model took ~35 s per call on the free tier, so the full grid runs on 26B |
| `eval_31b/` | evaluation tables of `runs_31b/` (`tables/*.md`) |
| `tables/` | evaluation tables of `runs/` (Markdown, CSV and LaTeX for the report), retrieval tables |
| `retrieval/` | retriever training log and retrieval-robustness results |
| `tuning/` | Register Gate threshold tuning on the development set (δ = 0.15) |
| `judge/` | pairwise LLM-judge verdicts (`gemma-4-31b-it`, both presentation orders) |

Versions: `en` = ESConv original, `light` / `heavy` = ESConv-HiEn (Light 98 of 100 conversations while the
last two rewrites wait for the next free-tier quota window).
