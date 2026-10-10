# CodeMixESC — specification (condensed from the project proposal)

Project: *Enhancing Multi-Agent Emotional Support Conversations for Code-Mixed Hinglish
Help-Seekers with Cross-Lingual Experience Retrieval and Register-Aware Response Selection*
(Roshan Raj, IIT Bhilai). Base paper: MultiAgentESC (Xu et al., EMNLP 2025),
code at https://github.com/MindIntLab-HFUT/MultiAgentESC (checked out in `external/MultiAgentESC`).

## Problem
MultiAgentESC is prompted and evaluated in English only. Hinglish (Roman-script Hindi–English)
help-seekers get (1) English-only experience retrieval (`all-roberta-large-v1`), (2) register-blind
response selection (the refiner checks only consistency, strategy alignment and stress relief),
and (3) no code-mixed evaluation. Goal: make the multi-agent pipeline robust for Hinglish without
fine-tuning the LLM agents.

## Method: CodeMixESC = MultiAgentESC + four components
The three stages of MultiAgentESC are kept: (i) dialogue analysis (decision maker `is_complex()`,
then emotion, cause and intention agents), (ii) strategy deliberation (a 3-agent round-robin group
chat over the top-10 retrieved ESConv cases), (iii) response generation (one response per strategy,
debate, reflection, vote, tie-breaking judge, refiner `self_reflection()`). The decision maker and
the judge are unchanged.

1. **Code-Mix Profiler (new).** HingBERT-LID (`l3cube-pune/hing-bert-lid`) tags every word; the
   Code-Mixing Index (Gambäck & Das 2016) is `CMI = 1 - max_i(w_i)/(n-u)` if `n>u` else 0
   (`u` = language-independent tokens: names, numbers, emojis, punctuation, URLs, mentions).
   Output: register profile `R = (CMI_s, dominant language, script)`, passed to all later agents.
   The analysis prompts also receive `R` and an instruction to interpret Hinglish distress
   expressions (tension, ghabrahat, log kya kahenge) in context.
2. **Cross-lingual experience retrieval (new).** Replace the English encoder by
   `paraphrase-multilingual-mpnet-base-v2` (278M), fine-tuned with the in-batch contrastive
   Multiple Negatives Ranking loss on Hinglish–English pairs `(h_i, e_i)`:
   `L = -1/B Σ_i log( exp(cos(h_i,e_i)/τ) / Σ_j exp(cos(h_i,e_j)/τ) )`.
   Training pairs: PHINC (13,738 Hinglish tweets with manual English translations; duplicates,
   URLs, user mentions and very short sentences removed; a small part held out for validation)
   plus ~5,000 ESConv *training* seeker utterances rewritten into Hinglish by an LLM.
   The English ESConv case bank is re-encoded once; it is never translated.
   This is the only trained model. Retrieval baselines: `all-roberta-large-v1` (base paper),
   LaBSE, multilingual mpnet without fine-tuning.
3. **Register-aware generation and selection (modified).** The response generator receives `R`
   and answers in the same script with a similar Hindi–English mix, keeping the strategy and the
   30-word limit. Debate, reflection (voting) and the refiner add a fourth criterion, *language
   and cultural fit*: "Does the response match the seeker's language, script and level of
   Hindi–English mixing, and does it sound natural?"
4. **Register Gate (new).** After refinement compute `CMI_r` of the response. If
   `|CMI_r − CMI_s| > δ` (initially δ = 0.2, tuned on a dev set) or the script differs, refine
   once more with an explicit register instruction. At most one extra LLM call per turn.

## Data
- **ESConv** (`data/esconv/ESConv.json`, 1,300 conversations). Turn-level samples exactly as the
  base `main.py`: test = `dataset[:100]` (1,210 supporter turns, 163 "early" turns with
  `count <= 5` answered by the single zero-shot agent), case bank = `dataset[100:]`
  (13,484 seeker-post → supporter-response pairs with strategy and problem_type).
- **ESConv-HiEn** (new, `data/esconv_hien/test_{light,heavy}.json`): the 100 test conversations
  rewritten turn by turn into Roman-script Hinglish by an LLM (one call per conversation),
  strategy labels copied. Light: CMI 0.1–0.3; Heavy: CMI 0.3–0.5; utterances out of band are
  regenerated. Planned quality check: a random ~20% sample rated manually (naturalness, meaning
  preservation, 1–5) and conversations below 3 rewritten and checked again. **Status of the released
  data:** only the LLM rater (`gemma-4-31b-it`) scored all conversations; the manual sample and the
  rewrite step were not run, and the report states the LLM-only result. Each dialog turn keeps `content`
  (Hinglish), `content_en`, `cmi`, `labse_sim`, `in_band`, ...
- **Dev set**: 12 training conversations (`data/esconv_hien/dev_conv_ids.json`), rewritten the
  same way (`dev_{light,heavy}.json`), for tuning δ and selecting the retriever checkpoint.
  Excluded from retriever training pairs; excluded from the case bank when evaluating on dev.
- Test conversations are never used for training (no leakage).

## Systems
`zero_shot`, `fewshot_cot` (single LLM call; all 1,210 turns), `maesc` (original pipeline,
English roberta retriever), `pivot` (Hinglish→English translation, MultiAgentESC, English→Hinglish
translation into the user's register; Hinglish versions only), `codemixesc` (full), ablations on
ESConv-HiEn: `cmx_noft` (multilingual mpnet, no fine-tuning), `cmx_noxl` (English roberta
retriever), `cmx_nogate` (codemixesc output before the gate, derived from the same run).
Multi-agent systems run on the fixed 200-turn subset `sampled_uids(200, seed=42)` of each version
(en, light, heavy). Efficiency: LLM calls and latency per turn for every system
(CodeMixESC ≤ MultiAgentESC + 1 call; pivot = MultiAgentESC + 2 calls).

## Metrics
- **Retrieval robustness**: for each Hinglish query, Overlap@10 with the cases retrieved for its
  parallel English original, and Problem-type Precision@10 (share of retrieved cases with the
  query conversation's `problem_type`). Target: higher than the English encoder, esp. on Heavy.
- **Response quality**: Distinct-1/2, BLEU-1/2/3, F1, ROUGE-L (base paper), plus chrF and
  multilingual BERTScore (robust to Hinglish spelling variation).
- **Register match**: mean CMI gap `|CMI_r − CMI_s|` and script consistency of responses.
- **Strategy stability**: Jensen–Shannon divergence between the strategy distributions obtained on
  the English and Hinglish versions of the same turns (lower = less disturbed by code-mixing).
- **Human / LLM evaluation**: pairwise win/tie/lose vs MultiAgentESC and Translate-Pivot on a
  sample, dimensions Fluency, Identification, Comforting, Suggestion, Overall + Language
  Naturalness; bilingual volunteers where available, complemented by an LLM judge.
- **Efficiency**: LLM calls and latency per turn.

## LLMs (free tier only; see docs/PLAN.md)
Agents and baselines: `gemma-4-26b-a4b-it` (temperature 0). Test/dev rewriting:
`gemini-3.5-flash-lite`. Retriever-pair rewriting: `gemini-3.1-flash-lite` (a different generator
than the test set, on purpose). Dataset quality check and LLM judge: `gemma-4-31b-it`.
