# CodeMixESC

**Multi-agent emotional support conversation for code-mixed Hinglish help-seekers, with
cross-lingual experience retrieval and register-aware response selection.**

Roshan Raj, IIT Bhilai — course project. Base paper: *MultiAgentESC: A LLM-based Multi-Agent
Collaboration Framework for Emotional Support Conversation* (Xu et al., EMNLP 2025),
[code](https://github.com/MindIntLab-HFUT/MultiAgentESC).

Many help-seekers in India describe distress in Roman-script Hindi–English
("yaar result aaya, bahut tension ho rahi hai, ghar pe kaise bataun"). MultiAgentESC retrieves
its "experience" with an English-only encoder, selects and refines responses without regard to
the user's language register, and was evaluated on English only. CodeMixESC keeps its
three-stage multi-agent pipeline and adds four components, without fine-tuning the LLM agents:

1. **Code-Mix Profiler** — HingBERT-LID word tags and the Code-Mixing Index give the seeker's
   register profile R = (CMI, dominant language, script), shown to every later agent.
2. **Cross-lingual experience retrieval** — `paraphrase-multilingual-mpnet-base-v2` fine-tuned
   with a contrastive (Multiple Negatives Ranking) loss on Hinglish–English pairs (PHINC +
   Hinglish rewrites of ESConv training utterances), so Hinglish queries retrieve the English
   ESConv case bank without translating it.
3. **Register-aware generation and selection** — the generator answers in the seeker's script
   and Hindi–English mix; debate, voting and the refiner add a *language and cultural fit*
   criterion.
4. **Register Gate** — if |CMI_response − CMI_seeker| > δ or the script differs, one more
   refinement with an explicit register instruction (at most one extra LLM call per turn).

```mermaid
flowchart LR
    S[Seeker message<br/>Roman Hinglish] --> P[Code-Mix Profiler<br/>HingBERT-LID + CMI → R]:::new
    P --> D{Decision maker}
    D -- early / not complex --> Z[Zero-shot agent<br/>+ R]:::mod
    D -- complex --> A[Emotion → Cause → Intention<br/>agents + R]:::mod
    A --> X[Cross-lingual retriever<br/>fine-tuned mSBERT, top-10]:::new
    X --> G[Strategy deliberation<br/>3-agent group chat]
    G --> GEN[Response per strategy<br/>conditioned on R]:::mod
    GEN --> V[Debate → Reflect → Vote<br/>+ language fit]:::mod
    V --> J[Judge on ties]
    J --> RF[Refiner<br/>+ register criterion]:::mod
    V --> RF
    RF --> RG{Register Gate<br/>CMI gap ≤ δ, same script?}:::new
    Z --> RG
    RG -- yes --> OUT[Final response]
    RG -- no: re-refine once --> OUT
    classDef new fill:#fde3c8,stroke:#e8710a,stroke-dasharray: 4 3
    classDef mod fill:#dbe8fb,stroke:#e8710a,stroke-dasharray: 4 3
```

## Evaluation
- **ESConv-HiEn** (new): the 100 ESConv test conversations of the base paper rewritten turn by
  turn into Roman-script Hinglish at two mixing levels, *Light* (CMI 0.1–0.3) and *Heavy*
  (CMI 0.3–0.5), with strategy labels copied, CMI-verified per utterance and quality-checked.
  English and Hinglish versions are parallel, so any change in behaviour is due to code-mixing.
- **Systems**: zero-shot and few-shot CoT prompting, MultiAgentESC, a translate-pivot baseline,
  CodeMixESC, and ablations without retriever fine-tuning, without the cross-lingual retriever
  and without the Register Gate.
- **Metrics**: retrieval Overlap@10 and problem-type Precision@10; Distinct-1/2, BLEU-1/2/3,
  F1, ROUGE-L, chrF, multilingual BERTScore; CMI gap and script consistency; Jensen–Shannon
  divergence of strategy distributions (English vs Hinglish); pairwise LLM/human judgements
  (Fluency, Identification, Comforting, Suggestion, Overall, Language Naturalness); LLM calls
  and latency per turn.

All LLM agents are open-weight models used through prompting (Gemma via the Gemini API free
tier, or any OpenAI-compatible server such as Ollama); every call is cached, so re-runs and
ablations that share stages cost nothing. Result tables are generated into `results/tables/`.

## Repository
```
codemixesc/            library
  profiler.py          Code-Mix Profiler (HingBERT-LID, CMI, register profile)
  retriever.py         experience retrieval over the ESConv case bank (any encoder)
  prompts.py           base prompts (verbatim) + register-aware additions + baselines
  agents.py            MultiAgentESC agents, AutoGen group-chat replica, Register Gate
  systems.py           all evaluated systems
  llm.py               cached, rate-limited LLM client (Gemini API / OpenAI-compatible)
  esconv.py            ESConv loading, base-paper split and turn samples
  metrics.py           evaluation metrics
  retrieval_eval.py    retrieval metrics
  testing.py           stand-ins for tests and dry runs
scripts/               setup_data, build_hien, quality_check, dataset_stats, build_pairs,
                       train_retriever, eval_retrieval, tune_delta, run_system, run_all,
                       evaluate, judge, chat
docs/                  SPEC (from the proposal), IMPLEMENTATION (fidelity and deviations),
                       RUN_FORMAT, RETRIEVER, EVALUATION, REPRODUCE, PLAN
tests/                 pytest suite (runs without models or API keys)
data/esconv_hien/      ESConv-HiEn (built by scripts/build_hien.py)
```

## Quick start
```bash
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/setup_data.py                        # base code + ESConv
python -m pytest -q                                 # all tests, no API key needed
python scripts/run_all.py --dry_run                 # whole experiment plan with stand-ins
python scripts/chat.py --dry_run --debug            # talk to the pipeline (stand-ins)
```
With a free Gemini API key in `.env` (`GEMINI_API_KEY=...`), `python scripts/chat.py --debug`
talks to the real system. The full reproduction (dataset, retriever training, δ tuning,
experiments, evaluation, judge) is in [docs/REPRODUCE.md](docs/REPRODUCE.md).

## Data and licences
ESConv (Liu et al., 2021) is for academic research only; it is not redistributed here
(`scripts/setup_data.py` fetches it). ESConv-HiEn is derived from it and shared under the same
terms. PHINC (Srivastava & Singh, 2020) is fetched from Zenodo by `scripts/build_pairs.py`.
The system is a research prototype for emotional support, not a crisis service and not a
replacement for professional help.
