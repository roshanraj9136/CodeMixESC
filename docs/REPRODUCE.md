# Reproducing CodeMixESC

Every step is resumable and every LLM call is cached in `cache/llm_cache.sqlite`, so a step
interrupted by a quota limit simply continues when re-run (the client also sleeps through the
daily-quota reset). Commands are shown for Windows PowerShell; on Linux/macOS use
`.venv/bin/python` instead of `.venv\Scripts\python.exe`.

## 0. Environment
```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
.venv\Scripts\python.exe -m pip install -r requirements.txt
"GEMINI_API_KEY=<your free AI Studio key>" | Out-File -Encoding utf8 .env    # never commit .env
.venv\Scripts\python.exe -m pytest -q                                        # sanity check, no key needed
```
`CODEMIX_DEVICE=cpu|cuda` selects the device of the profiler, the retrievers and LaBSE. On a
4 GB GPU, let only the retriever training use the GPU and set `$env:CODEMIX_DEVICE="cpu"` for
everything else. Long steps (more than ~10 minutes) should run detached, e.g.
```powershell
Start-Process -WindowStyle Hidden -FilePath .venv\Scripts\python.exe `
  -ArgumentList "-X utf8 scripts\run_all.py" `
  -RedirectStandardOutput logs\run_all.out -RedirectStandardError logs\run_all.err
```

## 1. Data
```powershell
.venv\Scripts\python.exe scripts\setup_data.py --models      # base code, ESConv (1,300 conv.), all HF models
```

## 2. ESConv-HiEn (test + dev)
```powershell
.venv\Scripts\python.exe -X utf8 scripts\build_hien.py --split test     # gemini-3.5-flash-lite, ~1-2 days of free quota
.venv\Scripts\python.exe -X utf8 scripts\build_hien.py --split dev
.venv\Scripts\python.exe -X utf8 scripts\dataset_stats.py               # results/tables/hien_stats_test.*
```

## 3. Quality check (20% manual sample + LLM rater)
```powershell
.venv\Scripts\python.exe scripts\quality_check.py sample     # writes data/esconv_hien/qc/sheet_*.csv and view_*.html
# open view_light.html / view_heavy.html, fill sheet_light.csv / sheet_heavy.csv (1-5);
# extra bilingual raters: copy the sheets (e.g. sheet_light_rater2.csv) and fill them too
.venv\Scripts\python.exe scripts\quality_check.py llm        # gemma-4-31b-it rates all 200 conversations
.venv\Scripts\python.exe scripts\quality_check.py report     # agreement + flagged.json (< 3)
.venv\Scripts\python.exe scripts\quality_check.py fix        # rewrite flagged conversations with feedback
.venv\Scripts\python.exe scripts\quality_check.py llm --only_fixed
# re-check the rewritten conversations by hand, update their rows in the sheets, run `report` again
.venv\Scripts\python.exe -X utf8 scripts\dataset_stats.py    # final statistics after the fixes
```

## 4. Cross-lingual retriever
```powershell
.venv\Scripts\python.exe -X utf8 scripts\build_pairs.py          # PHINC (Zenodo) + ~5,000 ESConv Hinglish pairs
.venv\Scripts\python.exe -X utf8 scripts\train_retriever.py      # GPU; best checkpoint by dev retrieval -> models/codemix-retriever
.venv\Scripts\python.exe -X utf8 scripts\eval_retrieval.py       # results/tables/retrieval_test.*
.venv\Scripts\python.exe -X utf8 scripts\eval_retrieval.py --split dev
```
See [RETRIEVER.md](RETRIEVER.md) for options (frozen embeddings on 4 GB GPUs, cached MNRL for
larger batches; on a Kaggle/Colab T4 full fine-tuning fits).

## 5. Register Gate threshold
```powershell
.venv\Scripts\python.exe -X utf8 scripts\tune_delta.py           # results/tuning/delta.json (used by all later runs)
```

## 6. Calibration check and pilot (before the long runs)
`dataset_stats.py` (step 2) reports, per version, the share of turns whose seeker is treated as
writing plain English and HingBERT-LID's false-Hindi rate on the English originals. Expect
close to 100% plain English on `en` and a small share on Light/Heavy (only the first, short
turns). If many Light turns are treated as English, or English seekers as code-mixing, adjust
`MIN_STRONG_FRAC` / `AMBIGUOUS` in `codemixesc/profiler.py` before running the systems.

Pilot (10 minutes):
```powershell
.venv\Scripts\python.exe -X utf8 scripts\run_system.py --system codemixesc --version heavy --limit 10 --out_dir scratch\pilot
.venv\Scripts\python.exe scripts\llm_usage.py --since 1
```
Every step should finish with `STOP`. A step flagged "check token budget" (answers cut at the
token limit, e.g. because thinking tokens count against it) needs a larger budget
(`GEN_MAX_TOKENS` / `ANALYSIS_MAX_TOKENS` in `codemixesc/agents.py`) before the real runs; the pilot's
calls are cached, so nothing is wasted. Also read a few records in `scratch\pilot\codemixesc\heavy.jsonl`.

## 7. Experiments
```powershell
.venv\Scripts\python.exe -X utf8 scripts\run_all.py              # all 18 runs in priority order, then evaluate.py
```
or individually, e.g. `scripts\run_system.py --system codemixesc --version heavy`. Systems:
`zero_shot`, `fewshot_cot` (all 1,210 turns), `maesc`, `codemixesc` (en/light/heavy), `pivot`,
`cmx_noft`, `cmx_noxl` (light/heavy); multi-agent systems use the fixed 200-turn sample.
Progress is in `logs\{system}_{version}.log`.

## 8. Evaluation and judgement
```powershell
.venv\Scripts\python.exe -X utf8 scripts\evaluate.py             # results/tables/*.{md,csv,tex}, results/figures/*.png
.venv\Scripts\python.exe -X utf8 scripts\judge.py                # pairwise LLM judge (both orders)
.venv\Scripts\python.exe -X utf8 scripts\judge.py --export_human # sheets for bilingual volunteers
.venv\Scripts\python.exe -X utf8 scripts\judge.py --import_human <filled sheets>
```
See [EVALUATION.md](EVALUATION.md) for metric definitions.

## Budget (free tier, approximate)
| Step | Model | Calls | Time |
|---|---|---|---|
| ESConv-HiEn test + dev | gemini-3.5-flash-lite | ~500-700 (400/day cap) | 1-2 days |
| Quality check | gemma-4-31b-it | ~250 | hours |
| Retriever pairs | gemini-3.1-flash-lite | ~220 (400/day cap) | < 1 day |
| Retriever training | — (GPU) | — | ~1-2 h on a 4 GB GPU, less on a T4 |
| δ tuning | gemma-4-26b-a4b-it | ~1,800 | hours |
| 12 multi-agent runs | gemma-4-26b-a4b-it | ~26,000 (≈11 per turn) | ~3 days (input-token limit) |
| 6 single-call runs | gemma-4-26b-a4b-it | ~7,300 | ~5 h |
| LLM judge | gemma-4-31b-it | ~800 | hours |

Ablations reuse the cached Stage-1 calls of `codemixesc` (identical prompts), so they cost less
than a full run. `n_calls` and `latency` per turn are logged in every record either way.
