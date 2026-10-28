# CodeMixESC

An emotional-support chatbot for people who write in Hinglish (Roman-script Hindi–English). It extends
[MultiAgentESC](https://github.com/MindIntLab-HFUT/MultiAgentESC) (EMNLP 2025) with four parts: a code-mix
profiler (HingBERT-LID + CMI), a cross-lingual retriever fine-tuned on Hinglish–English pairs, register-aware
agents, and a Register Gate that keeps the reply in the user's own language mix. The LLM agents are not
fine-tuned.

Roshan Raj (12341830), IIT Bhilai, DSL504 NLP course project.

## Live demo

**https://codemixesc.streamlit.app**

## Code

| Folder | Contents |
|---|---|
| `codemixesc/` | profiler, retriever, prompts, agents, systems, LLM client, metrics |
| `scripts/` | dataset build, retriever training, δ tuning, runs, evaluation, LLM judge |
| `demo/` | Streamlit app |
| `tests/` | pytest suite, run on every push |

## Run

```bash
pip install -r requirements.txt
python scripts/setup_data.py          # downloads the base code and ESConv
python -m pytest -q tests             # no API key needed
pip install -r demo/requirements.txt
streamlit run demo/streamlit_app.py   # needs a free GEMINI_API_KEY
```
