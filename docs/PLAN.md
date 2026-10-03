# CodeMixESC — build plan and shared conventions

Project: implement the proposal `d:\Downloads\CodeMixESC_Proposal_RoshanRaj_12341830 (1).pdf` in full.
Owner: Roshan Raj (IIT Bhilai, roll 12341830). The repo will be published **publicly** at github.com/roshanraj9136.

**CodeMixESC** extends MultiAgentESC (Xu et al., EMNLP 2025; code in `external/MultiAgentESC`) to
code-mixed Hinglish help-seekers. It has four parts:
1. A Code-Mix Profiler: HingBERT-LID plus the CMI, giving a register profile R.
2. Cross-lingual experience retrieval: paraphrase-multilingual-mpnet-base-v2, fine-tuned with MNRL on PHINC and ESConv Hinglish–English pairs.
3. Register-aware generation and selection: a fourth "language & cultural fit" criterion in debate, reflection, voting, judge and refiner.
4. A Register Gate: if |CMI_r − CMI_s| > δ or the script differs, refine once more.

Evaluation uses ESConv-HiEn, a new parallel test set built from the 100 ESConv test conversations, rewritten at two levels:
- Light: CMI 0.1–0.3.
- Heavy: CMI 0.3–0.5.

## Environment (Windows 11, Smart App Control ON)
- Python: always `C:/Users/hp/codemixesc/.venv/Scripts/python.exe -X utf8 ...`.
  - It is a python.org 3.12 venv with torch 2.5.1+cu121, sentence-transformers 3.3.1, transformers 4.46.3, scikit-learn 1.5.2, nltk, rouge-score, sacrebleu and bert-score.
  - If a native import fails with "Application Control policy", pin an older wheel. Do not use uv-managed Pythons.
- Use `C:/...` paths inside Python, never `/c/...`.
- GPU: one RTX 3050 Ti with **4 GB**. It throttles under load.
  - The retriever trainer (Agent B) has priority on the GPU.
  - Every other process sets `CODEMIX_DEVICE=cpu` unless the controller says otherwise. `profiler.py` and `retriever.py` honour this variable.
- Long jobs (more than 10 minutes) must run detached, e.g. PowerShell `Start-Process`, with output redirected to a log under `logs/`. Never tie them to a session-owned shell.

## Shared modules (already written by the controller; extend, don't fork)
- `codemixesc/llm.py` — `LLM(model, thinking="minimal")`.
  - Call `llm(prompt, system=None, max_tokens=..., temperature=0.0, tag="...", json_mode=False)`, or `llm.chat(messages, ...)`, which returns `(text, meta)`.
  - Every call is cached in SQLite at `cache/llm_cache.sqlite`, keyed on model, prompt and params. Identical calls are free, so ablations reuse shared stages automatically.
  - The **cross-process** rate limiter (`cache/ratelimit.sqlite`) enforces RPM/TPM and our own daily caps.
  - Real calls are logged to `cache/llm_calls.jsonl`.
  - The client retries 429/5xx and sleeps through daily-quota resets (midnight Pacific ≈ 12:30 IST).
  - Gemma models get the system prompt prepended to the first user turn.
- `codemixesc/profiler.py` — `Profiler()`:
  - `.tag_many(texts)` returns word-level EN/HI/X tags.
  - `.cmi(text)` and `.stats(text)` give the CMI, word counts and Hindi fraction.
  - `.profile(seeker_utterances)` returns R = {cmi, cmi_last, hi_frac, dominant, script}, pooled over all of the seeker's utterances so far.
  - `describe_register(R)` returns prompt-ready text.
  - CMI follows Gambäck & Das (2016). u counts punctuation, numbers, emoji, URLs, mentions and hashtags.
- `codemixesc/esconv.py`:
  - `load_esconv()` reads `data/esconv/ESConv.json`, which is the base repo's copy, 1,300 conversations.
  - `split()`: test = `dataset[:100]`, case bank = `dataset[100:]`, exactly as in the base code.
  - `turn_samples(conv, conv_id)` is an exact replica of the base `main.py` turn loop. Two consecutive supporter turns are merged, and `early = count <= 5` marks turns that go to the single zero-shot agent.
  - `all_samples(version)` takes version `en | light | heavy`. There are 1,210 test turns and 163 of them are early.
  - `sampled_uids(200, seed=42)` is the fixed 200-turn subset for the multi-agent systems; 29 of those turns are early.
  - `case_bank(exclude_conv=...)` returns (post, response, strategy, conv, problem_type) entries.
  - `load_version('light'|'heavy')` reads `data/esconv_hien/test_{level}.json`.
- `codemixesc/retriever.py`:
  - `ENCODERS = {roberta (base paper), labse, mpnet, mpnet-ft (models/codemix-retriever)}`.
  - `Retriever(encoder)` caches the case-bank embeddings in `cache/emb/*.npy`.
  - `.pairs(query, k=10)` returns `(post, "[strategy] response")` pairs exactly as `get_strategy()` builds them.
  - `.topk_many(queries)` batches many queries.

## LLM choices and daily budget (free tier only — never anything paid)

| Use | Model | Limit (ours) |
|---|---|---|
| All agents and baselines (open-weight, temp 0, thinking "minimal") | `gemma-4-26b-a4b-it` | 28 RPM, 15K input TPM, 14.3K requests/day |
| ESConv-HiEn test + dev rewriting | `gemini-3.5-flash-lite` | 400/day (the owner's Jarvis app shares this key) |
| Retriever training-pair rewriting (a different generator than the test set, on purpose) | `gemini-3.1-flash-lite` | 400/day |
| Dataset quality check and pairwise LLM judge | `gemma-4-31b-it` | slow (20–60 s per call), 14.3K/day |

- `gemma-4-31b-it` sometimes returns 500/503; the client retries these.
- Never print or commit the API key. It lives in `.env`, which is gitignored.

## Fixed experimental settings
- Dev conversations: `data/esconv_hien/dev_conv_ids.json` = [198, 218, 248, 292, 408, 539, 763, 848, 908, 1139, 1197, 1293].
  - They are excluded from retriever training pairs.
  - When evaluating dev retrieval, exclude them from the case bank, so a query cannot retrieve its own conversation.
- Test conversations (`dataset[:100]`) are never used for training.
- Multi-agent systems run on `sampled_uids(200)` for each version (en, light, heavy). Single-call baselines (zero-shot, few-shot CoT) run on all 1,210 turns.
- Systems:
  - `zero_shot`, `fewshot_cot`.
  - `maesc`: the original pipeline with the English roberta retriever.
  - `pivot`: Hinglish→English translation, then maesc, then translation back into the user's register.
  - `codemixesc`: the full system.
  - Ablations: `cmx_noft` (multilingual mpnet without fine-tuning), `cmx_noxl` (English roberta retriever) and `cmx_nogate` (codemixesc output before the gate; derived from the same run).
- Outputs: `results/runs/{system}/{version}.jsonl`, one JSON record per turn. The fields are listed in `docs/RUN_FORMAT.md`, written by Agent C.

## Ownership (parallel subagents)
- **A — data**: `scripts/build_hien.py`, `scripts/quality_check.py`, `data/esconv_hien/*`, `tests/test_profiler.py`.
- **B — retriever**: `scripts/build_pairs.py`, `scripts/train_retriever.py`, `scripts/eval_retrieval.py`, `data/pairs/*`, `models/codemix-retriever`, `results/retrieval/*`.
- **C — agents**: `codemixesc/prompts.py`, `codemixesc/agents.py`, `codemixesc/systems.py`, `scripts/run_system.py`, `scripts/tune_delta.py`, `docs/RUN_FORMAT.md`, `tests/test_agents.py`.
- **D — evaluation**: `codemixesc/metrics.py`, `scripts/evaluate.py`, `scripts/judge.py`, `tests/test_metrics.py`.
- **Controller**: `llm.py`, `profiler.py`, `esconv.py`, `retriever.py`, the long experiment launches, the report, and git/GitHub.
  - Changes to shared modules go through the controller. If you need a change, make it minimal and say so in your final report.

## Git rules
- The repo is `C:/Users/hp/codemixesc`. Commit only your own files, with explicit paths: `git commit -m "..." -- <paths>`.
- **No** `Co-Authored-By`, no Claude or AI attribution lines, no "Generated with" footers.
- Never commit `.env`, `cache/`, `external/` or large model files.
