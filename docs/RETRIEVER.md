# Cross-lingual experience retriever

MultiAgentESC retrieves the 10 most similar ESConv cases for the seeker's last utterance with the
English-only `all-roberta-large-v1`. CodeMixESC replaces it with
`paraphrase-multilingual-mpnet-base-v2`, fine-tuned so that a Roman-script Hinglish post lands next
to its English meaning. The English case bank (13,484 seeker posts of `dataset[100:]`) is encoded
once and never translated. This is the only trained model of the project.

| Step | Script | Output |
|---|---|---|
| Training pairs | `scripts/build_pairs.py` | `data/pairs/{train,val}.jsonl`, `data/pairs/stats.json` |
| Fine-tuning | `scripts/train_retriever.py` | `models/codemix-retriever/`, `results/retrieval/train_log.json` |
| Evaluation | `scripts/eval_retrieval.py` | `results/retrieval/retrieval_{split}.json`, `results/tables/retrieval_{split}*.{md,tex}` |
| Metrics (shared) | `codemixesc/retrieval_eval.py` | used by both of the above and by the tests |

## 1. Training pairs
Each pair is `{"hi", "en", "src": "phinc"|"esconv", "level": "light"|"heavy"|null, "conv": int|null}`.

**PHINC** (Srivastava & Singh, 2020; Zenodo record 3605597): 13,738 Hinglish tweets with manual
English translations. The script reads `--phinc <file>` (.csv/.tsv/.zip, also .xlsx if `openpyxl`
is installed), else the largest table in `data/raw/phinc/`, else downloads the record there through
the Zenodo REST API (checksums verified). It decodes as utf-8-sig, utf-8, cp1252 or latin-1, takes
the columns by name (`Sentence`, `English_Translation`, case-insensitive) and otherwise the first two
non-index text columns. The file name, member, SHA-256, encoding and chosen columns go into `stats.json`.

**ESConv rewrites**: seeker utterances of the training conversations only. That is `dataset[100:]`
without the 12 dev conversations of `data/esconv_hien/dev_conv_ids.json`; the test conversations
`dataset[:100]` are never used. Utterances need at least 4 words and are deduplicated (15,787
candidates); 5,000 are sampled with seed 42. `gemini-3.1-flash-lite` rewrites them into Roman-script
Hinglish, 25 per call (JSON in, JSON out). That is a different generator from the one that built the
test set, so the retriever cannot learn one model's Hinglish style. Light and Heavy mixing alternate
per batch, and the level descriptions are imported from `scripts/build_hien.py`.

A rewrite is accepted only if it:
- is non-empty and in Roman script;
- is not identical to the English;
- keeps 0.4 to 3 times the source length (catches merged or truncated items);
- contains at least one Hindi word, with `--check_cmi` (HingBERT-LID).

Failed items get one retry: a new prompt that says what was wrong, with an "Attempt 2" suffix so
the prompt cache cannot return the old answer. The first pass takes about 200 calls, within the
model's 400/day cap, which `codemixesc/llm.py` enforces.

The script can be stopped and restarted:
- Every finished call is appended to `data/pairs/esconv_rewrites.jsonl`, and the LLM cache makes
  repeated calls free.
- `--max_calls N` stops after N real calls (exit code 3). Rerun the same command to continue.
- An API failure leaves its batch pending. It is never silently dropped.

**Cleaning** (same for both sources; `stats.json` records the count after every step):
1. Decode HTML entities. Remove URLs, `RT @user:` headers, `@mentions` and bare `RT` markers.
   Strip `#` from hashtags but keep the word. Collapse whitespace.
2. Drop pairs with fewer than 3 words on either side.
3. Roman script only: every letter on both sides must be Latin.
4. Drop pairs whose Hinglish equals the English (not code-mixed). Comparison is on lowercase word
   tokens.
5. Drop duplicates of the normalised Hinglish side (lowercase word tokens). The same key is used
   across sources.

**Validation split** (seed 42):
- 500 random PHINC pairs.
- Whole ESConv conversations until about 200 pairs are held out. Utterances of one conversation
  are related, so they never straddle the split.
- `stats.json` checks for leakage: no pair from test or dev conversations, no Hinglish text in
  both train and val.

## 2. Fine-tuning
Objective: in-batch Multiple Negatives Ranking loss. The anchor is the Hinglish sentence, the
positive is its English translation, and the other English sentences of the batch are negatives.

    L = -1/B Σ_i log( exp(cos(h_i,e_i)/τ) / Σ_j exp(cos(h_i,e_j)/τ) ),  τ = 0.05 (scale 20)

τ = 0.05 (scale 20) is the default of the sentence-transformers MNRL implementation and a common
choice for contrastive sentence-embedding training. It changes nothing at inference.

| Setting | Default | Why |
|---|---|---|
| batch / lr / warmup / epochs | 32 / 2e-5 / 10% / 3 | proposal (1–3 epochs); linear decay, AdamW, weight decay 0.01 |
| `max_seq_length` | 128 | tweets and seeker posts are short; bounds activation memory |
| batch sampler | `NO_DUPLICATES`, fixed | a duplicate text inside a batch would be a false negative (see below) |
| `--freeze_embeddings auto` | frozen if GPU < 10 GB (or CPU) | see the memory budget below |
| fp16 | on CUDA | halves activations; the Adam state stays fp32 |
| `--gradient_checkpointing` | off | recomputes layer activations; use it if a 4 GB card runs out of memory |
| `--cached --mini_batch_size 16` | off | GradCache MNRL: the same loss and in-batch negatives, computed in mini-batches, so `--batch_size 64` fits 4 GB |
| seed | 42 | model init, data order (sampler seeded with seed + epoch), dropout |

**Batch sampler fix.** In sentence-transformers 3.3.1, `NoDuplicatesBatchSampler` keeps its
shuffled indices in a Python `set`. A set of small integers iterates in ascending order, so every
epoch would get the same batches in file order. `train_retriever.py` overrides the sampler: it
keeps the permutation (re-seeded per epoch, so resumed runs see identical batches) and compares
normalised texts.

**Memory on a 4 GB RTX 3050 Ti.** The model has about 278M parameters; 192M of them are the XLM-R
word-embedding matrix (250k × 768). Full fine-tuning keeps fp32 weights, gradients and two AdamW
moments, 16 bytes per parameter, which is about 4.4 GB before any activations. Freezing only the
word embeddings leaves 86M trainable parameters. The static memory is then about 2.1 GB: weights
1.1, gradients 0.35, Adam 0.7.

Freezing is also a sensible inductive bias. Most of the 250k vocabulary rows (other languages and
scripts) never occur in the pairs. Updating only the rows that do would move them away from the
untouched rows of every other word and language, which can hurt cross-lingual alignment (cf. Wu &
Dredze, 2019, on freezing the lower layers of multilingual BERT). On a 16 GB Kaggle/Colab T4,
`auto` trains everything. Pass `--freeze_embeddings yes` there if the result must be comparable to
a run on the 3050 Ti. `train_log.json` records which setting was used and the peak GPU memory.

**Checkpoint selection (never on test).** About 4 times per epoch, plus at the last step, an
evaluator runs on ESConv-HiEn dev:
- Hinglish posts of `dev_light.json` and `dev_heavy.json` retrieve from the case bank *without* the
  dev conversations.
- `primary = mean(Overlap@10 with the same model's top-10 for the parallel English post,
  problem-type Precision@10)`.

The best checkpoint is restored at the end (`load_best_model_at_end`; `save_total_limit 2`, so
budget about 4 GB of disk for `models/codemix-retriever-ckpt/`, which is deleted after success).
Hinglish→English accuracy@1 and MRR@10 on `data/pairs/val.jsonl` are logged at every evaluation.
Without the dev files they become the selection metric, with a warning.

A collapsed model, one that embeds everything alike, would inflate Overlap@10. `dev_unique@10`
(the share of distinct cases among all retrieved ones) and the val MRR, where ties count as
errors, expose that.

`train_log.json` contains:
- the configuration;
- the device and peak memory;
- the frozen and trainable parameter counts;
- the evaluation before training (the untuned mpnet) and every evaluation;
- the best step and the final evaluation of the saved model, reloaded from disk exactly as the
  pipeline loads it;
- the training time, library versions and git commit.

## 3. Evaluation protocol (`eval_retrieval.py`)
**Queries.** Each query is the `post` of a supporter turn (`history[-1]['content']`), which is
exactly what `get_strategy()` sends to the retriever. That covers all 1,210 test turns and the
`sampled_uids(200)` subset used by the multi-agent systems, in every version present (EN, Light,
Heavy). The English counterpart of a Hinglish post is `content_en` of the same dialog turn.

**Encoders.** `roberta` (base paper), `labse`, `mpnet` (before fine-tuning) and `mpnet-ft`
(`models/codemix-retriever`). Unavailable encoders are skipped with a warning.

**Metrics** (k = 10):

| Metric | Measures |
|---|---|
| P@10 | problem-type precision per version |
| Overlap@10 self | Hinglish post vs its English original, same encoder |
| Overlap@10 vs RoBERTa-EN | the post vs the base paper's retriever on the English original: how much of MultiAgentESC's English experience the system still sees |
| Strategy JSD | base 2; between the strategy histograms of the cases retrieved for the Hinglish and the English posts, pooled over the queries (a systematic shift in the strategy evidence the agents deliberate on). The per-query mean is in the JSON |
| Random baseline | the bank share of the query's problem type (chance level) |

**Confidence intervals.** 95% percentile intervals from 2,000 cluster-bootstrap resamples. Whole
conversations are resampled, because turns of a conversation are correlated. Differences to
RoBERTa use paired resamples. In the tables, † means the CI of the difference excludes 0.

**Dev split.** `--split dev` evaluates the dev files against the bank without the dev
conversations. The bank embeddings always come from the full-bank cache file, the same one the
pipeline uses, and are subset by row.

**Tables.** All are booktabs `tabular`s with no float, so they can go straight into a paper's
table environment; the suggested caption is the second comment line.
- `retrieval_test.tex`: P@10 and Overlap@10 (8 columns, fits `\resizebox{\columnwidth}`).
- `retrieval_test_full.tex`: adds strategy JSD (10 columns, for a `table*`).
- `retrieval_test_s200.tex`: the 200-turn subset.
- The `.md` file has every number with its CI, plus the paired differences.

## 4. Commands (Windows, from the repository root)
The first run downloads the encoders from the Hugging Face Hub. `CODEMIX_DEVICE=cpu|cuda` selects
the device, as everywhere in the project.
```powershell
# PHINC: put the Zenodo download (csv or zip) into data\raw\phinc\, or let the script fetch it
.venv\Scripts\python.exe -X utf8 scripts\build_pairs.py --dry_run          # fake LLM, scratch\pairs_dry_run
.venv\Scripts\python.exe -X utf8 scripts\build_pairs.py                    # ~200 Flash-Lite calls, 15-30 min
# optional stricter filter (downloads HingBERT-LID):  --check_cmi

# training (> 10 min: run detached, log under logs\)
Start-Process -WindowStyle Hidden -FilePath .venv\Scripts\python.exe `
  -ArgumentList "-X","utf8","scripts\train_retriever.py" `
  -RedirectStandardOutput logs\train_retriever.log -RedirectStandardError logs\train_retriever.err
#   out of memory on 4 GB:  add "--gradient_checkpointing"  (or "--cached","--mini_batch_size","8")
#   interrupted:            same command with "--resume"
#   more negatives:         "--cached","--batch_size","64","--mini_batch_size","16"

.venv\Scripts\python.exe -X utf8 scripts\eval_retrieval.py --split dev
.venv\Scripts\python.exe -X utf8 scripts\eval_retrieval.py                 # test: results\tables\retrieval_test.*
```
Kaggle/Colab T4: `python scripts/train_retriever.py` (full fine-tuning by default; add
`--freeze_embeddings yes` for comparability with a 4 GB run). Copy `models/codemix-retriever/` back
to the same path on the owner's machine.

Smoke test without data or GPU: `python -m pytest tests/test_retrieval.py`. It builds a 2-layer
random BERT, runs `build_pairs --dry_run`, trains for a few steps on CPU and evaluates with stub and
tiny encoders, in under two minutes.
