# CodeMixESC

[![tests](https://github.com/roshanraj9136/CodeMixESC/actions/workflows/tests.yml/badge.svg)](https://github.com/roshanraj9136/CodeMixESC/actions/workflows/tests.yml)
[![Live demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://codemixesc.streamlit.app)
[![Retriever on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20model-codemix--retriever-orange)](https://huggingface.co/roshan9136/codemix-retriever)

**Multi-agent emotional support for code-mixed Hinglish help-seekers, with cross-lingual experience
retrieval and register-aware response selection.**
Roshan Raj (12341830), IIT Bhilai — NLP course project (DSL504).

Many people in India describe distress in Roman-script Hinglish (*"yaar result aaya, bahut tension ho rahi
hai, ghar pe kaise bataun"*). [MultiAgentESC](https://github.com/MindIntLab-HFUT/MultiAgentESC) (Xu et al.,
EMNLP 2025), a strong LLM multi-agent emotional-support system, retrieves past cases with an English-only
encoder, picks replies without looking at the user's language mix, and was evaluated on English only.
CodeMixESC keeps its three-stage pipeline and adds four components, **without fine-tuning the LLM agents**:

1. **Code-Mix Profiler**: HingBERT-LID word tags + the Code-Mixing Index (CMI) give the user's register R.
2. **Cross-lingual retriever**: multilingual mpnet fine-tuned contrastively (MNRL) on 16,951 Hinglish–English
   pairs, so Hinglish queries find the right cases in the English case bank.
3. **Register-aware generation and selection**: replies in the user's mix; debate, voting and the refiner add a
   *language and cultural fit* criterion.
4. **Register Gate**: if the reply's Hindi share is more than δ away from the user's, one re-write (≤ 1 extra
   LLM call; kept only if closer).

We also build **ESConv-HiEn**, a parallel Hinglish version of the ESConv test set at two mixing levels with
the strategy labels kept, for a controlled robustness study.

![CodeMixESC architecture](report/figures/architecture_slide.png)

## Results

**ESConv-HiEn** (new dataset; 100 conversations, 3,140 utterances per version,
[details](results/tables/hien_stats_test.md))

| Version | Hindi words | Mean CMI | Utterances in target band | Similarity to English (LaBSE) |
|---|---|---|---|---|
| English original | 1.5% | 0.01 | – | – |
| Light (target CMI 0.1–0.3) | 16% | 0.16 | 94.5% | 0.88 |
| Heavy (target CMI 0.3–0.5) | 50% | 0.35 | 87.5% | 0.71 |

**Retrieval robustness** (all 1,210 test turns, k = 10, 95% conversation-level bootstrap CI,
[full table](results/tables/retrieval_test.md)). Overlap@10 = share of the cases retrieved for a Hinglish post
that are also retrieved for its English original; P@10 = share with the query's problem type.

| Encoder | Overlap@10 Light | Overlap@10 Heavy | P@10 Light | P@10 Heavy |
|---|---|---|---|---|
| all-roberta-large-v1 (MultiAgentESC) | 56.9 ± 1.9 | 27.0 ± 2.3 | **35.8** ± 2.3 | **33.8** ± 2.4 |
| LaBSE | 61.4 ± 2.0 | 33.2 ± 2.2 | 28.9 ± 1.5 | 28.5 ± 1.3 |
| multilingual mpnet | 69.9 ± 1.6 | 31.3 ± 2.3 | 31.6 ± 2.0 | 28.4 ± 1.6 |
| **multilingual mpnet, fine-tuned (ours)** | **73.2** ± 1.3 | **53.3** ± 1.8 | 32.0 ± 2.0 | 31.5 ± 1.9 |

Heavy code-mixing breaks the English retriever (27% of its cases survive); ours keeps 53% (+26.3 points,
CI [24.6, 28.2]) and keeps the retrieved strategies stable (JSD 0.15 → 0.01), at a small cost in problem-type
precision (−2 to −4 points).

**Reply quality, first run** (Light, 97 common turns, agents `gemma-4-31b-it`;
[tables](results/eval_31b/tables)). Paired bootstrap: CodeMixESC vs MultiAgentESC ROUGE-L +1.27 (p < 0.05);
vs Translate-Pivot better on every metric (p < 0.01); Register Gate CMI gap 0.125 → 0.101 (p < 0.01).

| System | BLEU-2 | ROUGE-L | chrF | CMI gap ↓ | LLM calls / turn |
|---|---|---|---|---|---|
| MultiAgentESC | 7.43 | 11.42 | 19.03 | 0.107 | 6.31 |
| Translate-Pivot | 3.37 | 6.74 | 17.25 | 0.194 | 8.03 |
| CodeMixESC w/o Register Gate | 7.95 | 12.33 | 19.28 | 0.125 | 6.35 |
| **CodeMixESC** | **8.14** | **12.69** | **19.32** | **0.101** | 6.51 |

*The full grid (English, Light, Heavy; zero-shot, few-shot CoT, MultiAgentESC, Translate-Pivot, CodeMixESC and
ablations; pairwise LLM judge) runs on `gemma-4-26b-a4b-it` agents and a fixed 50-turn sample per version;
tables are written to [`results/tables/`](results/tables) as the runs finish.*

## Live demo

**[codemixesc.streamlit.app](https://codemixesc.streamlit.app)**: chat as a help-seeker in English or Hinglish
and watch the eight agents work step by step. The side panel shows your measured Hindi–English mix, the
emotion, cause and intention the agents inferred, the support strategy they voted for and the Register Gate's
language check. Run it locally with `pip install -r demo/requirements.txt` and
`streamlit run demo/streamlit_app.py` (needs a free `GEMINI_API_KEY` in `.env`).

## Quick start

```bash
pip install -r requirements.txt
python scripts/setup_data.py          # base code and ESConv
python -m pytest -q tests             # 110 tests, no API key needed
python scripts/run_all.py --dry_run   # the whole experiment plan with stand-ins
```

Full reproduction (dataset, retriever training, δ tuning, runs, evaluation, judge):
[docs/REPRODUCE.md](docs/REPRODUCE.md).

## Repository

| Path | Contents |
|---|---|
| `codemixesc/` | profiler, retriever, prompts (base prompts verbatim + register additions), agents, systems, cached rate-limited LLM client, metrics |
| `scripts/` | dataset build and quality check, retriever pairs and training, δ tuning, runs, evaluation, LLM judge, case studies, CLI chat |
| `demo/` | Streamlit live demo |
| `data/esconv_hien/` | ESConv-HiEn test and development sets, quality-check sample |
| `results/` | per-turn run records, tables, retrieval and tuning results ([guide](results/README.md)) |
| `report/` | paper (IEEE) and figures; [`docs/presentation/`](docs/presentation) holds the slides |
| `docs/` | [implementation notes and every deviation from the base code](docs/IMPLEMENTATION.md), [evaluation protocol](docs/EVALUATION.md), [retriever](docs/RETRIEVER.md), [run format](docs/RUN_FORMAT.md) |
| `tests/` | pytest suite, run by CI on every push |

## Data and licences

ESConv (Liu et al., 2021) is for academic research only and is not redistributed here
(`scripts/setup_data.py` downloads it); ESConv-HiEn is derived from it and shared under the same terms. PHINC
(Srivastava & Singh, 2020, CC BY 4.0) is downloaded by `scripts/build_pairs.py`. All LLMs are used through the
Gemini API free tier. CodeMixESC is a research prototype, not a crisis or counselling service.
