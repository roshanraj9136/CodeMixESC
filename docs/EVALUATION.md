# Evaluation

| File | Role |
|---|---|
| `codemixesc/metrics.py` | every metric, as pure functions (the definitions below) |
| `scripts/evaluate.py` | scores all runs, writes `results/eval/`, `results/tables/`, `results/figures/` |
| `scripts/judge.py` | pairwise LLM judge, human-evaluation sheets and their analysis |
| `tests/test_metrics.py` | hand-computed values for every metric, end-to-end runs on synthetic records |

## 1. Which turns are compared

**Join.** Every run record (`docs/RUN_FORMAT.md`) is joined with its test sample by `uid`. The
reference, the dialogue context and the gold strategy come from the test data, not from the run
file. A record whose `reference` or `post` differs from the current data is counted as *stale*
(e.g. ESConv-HiEn was regenerated after the run) and reported.

**Scopes.** Each run is scored on three sets of turns:

| Scope | Turns |
|---|---|
| `all` | every turn of the run: 1,210 for the single-call baselines, 200 for the multi-agent systems |
| `subset` | the run's turns among the fixed subset `sampled_uids(200, seed=42)` |
| `common` | the subset turns that **every** system of the version answered |

All comparison tables use `common`, so every row of a table is computed on identical turns.
The single-call baselines on all 1,210 turns appear in `baselines_all` (a check that the
200-turn subset is representative) and in `metrics.json`. A run covering less than
`--min_coverage` (default 0.9) of the subset, e.g. one that is still running, is scored but
left out of the common turns and the tables, so it cannot shrink everyone's turns.

**Failures.** `run_system.py` never writes a failed turn (it goes to `{version}.errors.jsonl`
and is retried on the next run), so a failed turn is *missing*: it drops out of the common
turns of all systems alike. A record that does carry an `error` field, or a response that is
empty or `"None"`, is scored as the empty string by the quality metrics (no credit; failures
can never raise a score) and skipped by the register and length statistics (they describe
produced text). The runs table reports, per run: missing turns, failed records, `"None"`
responses, LLM calls that returned an empty/blocked answer (`n_failed_calls`) and calls cut at
the token limit (`n_truncated_calls`), duplicate uids, unreadable lines and stale records.

**Derived and paired systems.** `cmx_nogate` is the `codemixesc` run before the Register Gate
(`response := pre_gate_response`, `n_calls -= gate.calls`, `latency -= gate.latency`; its
`n_real_calls` is not derivable). It is derived for Light and Heavy, and for EN only when an
EN ablation run exists (ablations are run on the Hinglish versions). The Translate-Pivot runs
MultiAgentESC on translations, so its EN counterpart for strategy stability is MultiAgentESC
on the original English turns.

## 2. Tokenizer

All token-based metrics share `metrics.tokenize()`: Unicode NFC normalisation, lowercasing,
then the regex `\w+|[^\w\s]` (a token is a run of word characters, or one punctuation mark,
symbol or emoji). The regex runs on the `regex` package (installed with nltk), whose `\w`
follows Unicode and includes combining marks: Python's `re` would cut a Devanagari word such
as हिंदी at every vowel sign. *Word tokens* are the tokens containing a letter or digit
(punctuation, symbols and emojis removed).

## 3. Response quality

All scores are ×100. Per-turn scores are averaged over turns; corpus scores pool statistics.

| Metric | Definition and choices |
|---|---|
| Distinct-1/2 | corpus level (Li et al., 2016): distinct n-grams / all n-grams over all responses of the run; n-grams never cross a response boundary; punctuation tokens included |
| BLEU-1/2/3 | cumulative corpus BLEU, NLTK `corpus_bleu`, uniform weights 1/n over the 1..n-gram precisions, one reference per turn, all tokens. Smoothing: NLTK **method 3** (NIST geometric smoothing): an order without a single match gets precision 1/(2^k · #hypothesis n-grams), k = 1, 2, … for successive empty orders. It changes nothing when every order has a match (a full test set gives exactly the unsmoothed BLEU) and keeps BLEU-3 of a small corpus above 0. BLEU is 0 only if no unigram matches. NLTK is used as is (each hypothesis adds at least 1 to every order's precision denominator) |
| F1 | ParlAI-style unigram F1 (Miller et al., 2017), per turn: word tokens without the articles *a, an, the*, bag-of-words overlap with clipped counts, harmonic mean of precision and recall; equal to ParlAI's normalisation and formula (re-implemented in the tests) on ASCII text |
| ROUGE-L | LCS F-measure with β = 1 (`rouge_score`'s `fmeasure`), per turn, on word tokens, no stemming. `rouge_score` keeps only `[a-z0-9]` and silently deletes Devanagari or accented words, so the LCS is computed on our tokens; on ASCII text (without `_`) both give the same score (tested against `rouge_score`) |
| chrF | `sacrebleu` `CHRF()` defaults (character 6-grams, β = 2, case-sensitive, whitespace ignored; signature stored in `metrics.json`). Tables: corpus chrF; bootstrap: sentence chrF. Character n-grams give partial credit to Romanised Hindi spelling variants (*nahi / nahin / nhi*) |
| BERTScore | `bert_score` F1 with `bert-base-multilingual-cased`, its default layer (9), no idf, **no baseline rescaling** (no baseline exists for Hinglish): absolute values sit in a narrow high band, so compare systems rather than read values. An empty response scores 0 without running the model. Optional (`--no_bertscore`; skipped with a warning when the model cannot be loaded). Hash stored in `metrics.json` |

References are the dataset's supporter turns of the same version (the Hinglish rewrites for
Light and Heavy).

## 4. Register match

For every produced response, with the Code-Mix Profiler (HingBERT-LID):

- **CMI gap** `|CMI_r − CMI_s|` (the proposal's definition): `CMI_s = Profiler.profile(seeker
  utterances so far)["cmi"]`, the seeker's CMI pooled over all their utterances, i.e. the profile R
  the system computes; `CMI_r = Profiler.stats(response)["cmi"]`. No plain-English special case.
- **Hindi-fraction gap** `|hi_frac_r − hi_frac_s|`. The CMI is symmetric in the two languages
  (80 % and 20 % Hindi words give the same CMI), so only this gap notices a reply that flips the
  dominant language. It is also the distance the Register Gate uses by default.
- **Script consistency**: % of turns whose response script (`script_of`: Roman, Devanagari,
  Mixed) equals the seeker's. Turns where the seeker has written no letters yet (script `None`)
  are left out (no target); a response without letters (emoji or punctuation only) counts as a
  mismatch.

Means are over the turns with a response (the count is `n_register`). The register table adds
a **Gold reference** row (the dataset's supporter turns against the same seekers) as a point of
comparison, and states the mean seeker CMI_s per version. On EN, HingBERT-LID tags a few English
homographs (*to, me, do*) as Hindi, so English text has a small non-zero CMI; the EN columns are
a sanity check (CodeMixESC must keep answering English users in English), comparable across
systems. Reported results use `--profiler hingbert`; `--profiler lexicon` (a word list) exists
only for tests and dry runs, and `--profiler none` skips the register metrics.

## 5. Strategy stability

For each system with an EN run (the pivot: MultiAgentESC on EN) and each Hinglish version, on the
same turns (common EN turns ∩ common Hinglish turns):

- **JSD** (with None): Jensen–Shannon divergence between the distributions of `pred_strategy`
  over the 8 ESConv strategies + `"None"` (no strategy predicted: early or single path,
  fallback), with base-2 logarithms, so 0 = identical and 1 = disjoint distributions.
  `JSD = ½ KL(P‖M) + ½ KL(Q‖M)`, `M = ½(P + Q)`. (SciPy's `jensenshannon` returns √JSD.)
- **JSD-S** (strategies only): the same over the 8 strategies, on the turns where *both*
  versions predicted a strategy, so the two distributions cover the same turns.
- **Agreement**: % of turns with the same prediction in both versions (with `"None"` as a label),
  and the same on the JSD-S turns (`metrics.json`).

Zero-shot never predicts a strategy and is omitted; `cmx_nogate` shares codemixesc's strategies.
Unknown strategy names count as `"None"` and are reported (`n_invalid_strategy`).
Supplementary (`metrics.json`): **strategy match**, the % of turns whose predicted strategy is
one of the gold strategies (a merged turn `"A and B"` has both; names are matched whole, so
*Affirmation and Reassurance* is not split), over all turns (`"None"` = miss) and over the turns
with a prediction.

## 6. Efficiency

Per turn, on the common turns: mean logical LLM calls `n_calls` (cached calls included, the
efficiency measure), `n_real_calls` (`metrics.json` only: it depends on the cache state), summed
API latency (cache hits report the latency of the original call), extra calls and latency over
MultiAgentESC on the same turns, share of turns on the full multi-agent path, mean response length
in words (whitespace-separated chunks with a letter or digit, as the prompts' 30-word limit
counts them) and % of responses over 30 words. For gated systems `metrics.json` also gives the %
of turns where the gate fired and the % of those where its rewrite was kept.

## 7. Significance

Paired bootstrap over turns (Efron & Tibshirani, 1993; Koehn, 2004), CodeMixESC against every
other system of a version, on the common turns, for the per-turn F1, ROUGE-L, sentence chrF,
BERTScore F1 and CMI gap. Δ = mean(CodeMixESC) − mean(other) (for the CMI gap, negative is better).
2,000 resamples with `numpy.random.default_rng(42)` (`--n_boot`, `--seed`); the 95 % CI is the
percentile interval of the resampled Δ*; the two-sided p-value is
`p = min(1, 2 (1 + min(#{Δ* ≤ 0}, #{Δ* ≥ 0})) / (B + 1))`, Koehn's "better in x % of the
resamples" made two-sided, which agrees with the CI (p < 0.05 exactly when the 95 % CI excludes 0,
up to the +1 correction). p-values are not corrected for multiple comparisons. Tables mark
† p < 0.05 and ‡ p < 0.01; the Markdown and CSV versions also give the CIs.

## 8. LLM judge (`scripts/judge.py`)

- **Pairs and turns**: CodeMixESC vs MultiAgentESC and vs Translate-Pivot (`--pairs`), on Light
  and Heavy (`--versions`, EN optional), 100 turns each (`--n`) among the turns where both
  systems answered. Turns are ranked by `sha256("42:uid")` and the first n kept: seeded, and
  stable when a few candidate turns come or go, so different pairs and versions share most turns.
- **Prompt**: one prompt per comparison shows the whole dialogue context and responses A and B
  and asks for a one-sentence comparison plus a verdict `"A"`, `"B"` or `"tie"` per dimension, as
  JSON. Dimensions: ESConv's human-evaluation questions (Liu et al., 2021) asked of single
  responses, plus Language Naturalness:

  | Dimension | Question |
  |---|---|
  | Fluency | Which response is more fluent and understandable? |
  | Identification | Which response explores the user's situation more in depth and is more helpful in identifying the user's problem? |
  | Comforting | Which response is more skillful in comforting the user? |
  | Suggestion | Which response gives more helpful suggestions for the user's problem? |
  | Overall | Generally, which response is the better emotional support, the one the user would prefer? |
  | Language Naturalness | Whose language (script, Hindi–English mix, wording) sounds more natural for this particular user, given how the user writes? |

  The judge is told to judge each dimension on its own, ignore order and length, and answer tie
  when equal or not applicable.
- **Position bias**: every turn is judged in both orders. Per dimension: *win* if both orders
  prefer CodeMixESC, *lose* if both prefer the other system, *tie* otherwise. Identical responses
  are a tie without a call. `summary.json` reports how often the two orders agree and how often a
  decided verdict picks position A (≈ 50 % for an unbiased judge).
- **Model**: `gemma-4-31b-it`, temperature 0, thinking `minimal`, 600 output tokens, via the
  cached, rate-limited `LLM` client. Parsing accepts fenced or slightly broken JSON and key/value
  spelling variants; an unusable answer is retried twice with an "(Attempt k …)" note appended,
  because the client caches by prompt. A comparison still unusable after 3 attempts (or an API
  failure) leaves the turn out and is counted as failed.
- **Output**: `results/judge/{a}_vs_{b}_{version}.jsonl` (both raw answers, verdicts and final
  labels per turn), `results/judge/summary.json`, `results/tables/judge.{md,csv,tex}` (win / tie /
  lose % per dimension; † / ‡: two-sided exact sign test of wins against losses, ties excluded).
- **Cost**: at most 100 × 2 orders × 2 pairs × 2 versions = 800 calls. The 15K input tokens per
  minute of the free tier bound it to roughly 5–10 calls per minute: plan 2–3 hours, detached.
  Re-running is free (cache).

## 9. Human evaluation

`--export_human` writes `results/human/sheet.csv` (UTF-8 with BOM, opens in Excel), `key.json`
and `instructions.txt`. The sheet holds the first `--human_n` turns (default 50) of the judge's
sample per pair and version, skipping identical responses; A/B is randomised per item and the
items are shuffled (seed 42); the sheet never names a system or a version, the key does (keep it
away from the volunteers). Each volunteer fills a copy (A, B or tie per dimension; blank =
not judged) and saves it under their own name.

`--import_human sheet_asha.csv sheet_ravi.csv ...` maps answers to win / tie / lose for
CodeMixESC, takes the **majority label per turn** (tie when no label has a strict majority),
and writes `results/human/summary.json` and `results/tables/human.{md,csv,tex}` with win / tie /
lose % and the sign test per dimension, **Cohen's κ against the LLM judge** (majority label vs the
judge's final label on the same turns, per dimension and pooled) and the **inter-annotator
agreement** (mean pairwise Cohen's κ between volunteers, pooled over dimensions).

## 10. Outputs

| File | Content |
|---|---|
| `results/eval/metrics.json` | every number for every run and scope, run summaries, stability, bootstrap results, settings (tokenizer, BLEU smoothing, chrF signature, BERTScore hash, profiler, seeds, git commit) |
| `results/eval/per_turn.jsonl` | per-turn scores (F1, ROUGE-L, chrF, BERTScore, CMI_s, CMI_r, gaps, scripts, strategy, calls) |
| `tables/main_{en,light,heavy}` | response quality of the main systems (common turns) |
| `tables/register` | CMI gap, Hindi-fraction gap, script consistency per version, all systems + gold reference |
| `tables/stability` | JSD, JSD-S and agreement, EN → Light and EN → Heavy |
| `tables/ablation` | CodeMixESC vs w/o Register Gate, w/o retriever fine-tuning, w/o cross-lingual retriever (Light, Heavy) |
| `tables/efficiency` | calls, extra calls over MultiAgentESC, latency, multi-path %, length |
| `tables/significance` | paired bootstrap Δ of CodeMixESC against every system |
| `tables/baselines_all` | single-call baselines on all 1,210 turns vs the subset |
| `tables/runs` | coverage and failure counts of every run |
| `tables/strategy_dist` | predicted-strategy distributions (table view of the figures) |
| `tables/judge`, `tables/human` | LLM judge and human evaluation (written by `judge.py`) |
| `figures/strategy_dist_{system}.png` | predicted strategies on EN, Light, Heavy (same turns) |
| `figures/robustness.png` | F1, ROUGE-L, chrF, BERTScore and CMI gap of each main system from EN to Light to Heavy |

Every table comes as `.md` (bold best), `.csv` (raw numbers; CIs and p-values in separate
columns) and `.tex`. The LaTeX tables use booktabs, mark the best value per column (per version
block) in bold (nothing is bold when all rows tie), carry `\label{tab:<name>}` and a caption,
and shrink to the line width only when wider. Tables with 10 or more columns use `table*` (both
columns of an IEEE two-column page). In the paper preamble: `\usepackage{booktabs,graphicx}`;
then `\input{results/tables/main_heavy}`. Every figure also has a vector `.pdf` twin for LaTeX.
Re-running `evaluate.py` replaces all of its own tables and figures (stale ones are removed),
never `judge.*` or `human.*`.

## 11. Commands

Windows (PowerShell, from the repository root; Linux/macOS: `.venv/bin/python`):
```powershell
$env:CODEMIX_DEVICE="cpu"                                             # or cuda when the GPU is free
.venv\Scripts\python.exe -X utf8 scripts\evaluate.py                  # HingBERT-LID + BERTScore, all runs
.venv\Scripts\python.exe -X utf8 scripts\evaluate.py --versions light,heavy --no_bertscore
.venv\Scripts\python.exe -X utf8 scripts\judge.py                     # LLM judge, Light + Heavy, 100 turns
.venv\Scripts\python.exe -X utf8 scripts\judge.py --versions en,light,heavy
.venv\Scripts\python.exe -X utf8 scripts\judge.py --export_human --human_n 25
.venv\Scripts\python.exe -X utf8 scripts\judge.py --import_human results\human\sheet_asha.csv results\human\sheet_ravi.csv
.venv\Scripts\python.exe -m pytest -q tests\test_metrics.py
```
Dry runs (no models, no API calls): with the fake test data of a dry run,
```powershell
$env:CODEMIX_HIEN_DIR="scratch\fake_hien"
.venv\Scripts\python.exe -X utf8 scripts\evaluate.py --runs_dir scratch\dry_runs --out_dir scratch\dry_eval --profiler lexicon --no_bertscore
.venv\Scripts\python.exe -X utf8 scripts\judge.py --dry_run --runs_dir scratch\dry_runs      # writes scratch\judge_dry_run
```
(`--hien_dir scratch\fake_hien` does the same as the environment variable.) Other options:
`--systems`, `--min_coverage`, `--n_boot`, `--seed`, `--bertscore_model`, `--no_figures`
(`evaluate.py`); `--pairs`, `--n`, `--model`, `--thinking`, `--json_mode`, `--workers`,
`--out_dir` (`judge.py`). Everything is deterministic: rerunning on the same runs gives the same
numbers (bootstrap and sampling are seeded; LLM verdicts come from the cache).
