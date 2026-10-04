# Implementation notes: fidelity to MultiAgentESC and every deviation

The base system is re-implemented on our cached, rate-limited LLM client instead of AutoGen
0.2 + an OpenAI-compatible server, so that every call is cached (ablations reuse shared
stages for free), counted and timed. What is identical, and what differs and why, is listed
here so that the comparison stays controlled: **every deviation applies to all systems alike**
(`maesc`, `pivot`, `codemixesc` and the ablations share `codemixesc/agents.py`).

## Identical to the released code (tested)
- **Prompts**: all prompts and agent system messages are byte-for-byte those of `prompt.py`
  and `multiagent.py` (typos included); `tests/test_agents.py::test_base_prompts_verbatim`
  checks them against the AST of the base code.
- **Turn extraction**: test = `dataset[:100]`, consecutive supporter turns merged into one
  sample (`"A and B"` strategy), `post = history[-1]['content']`, `early = count <= 5`;
  `test_turn_samples_match_base_main_loop` compares all 1,210 turns with a direct port of
  `main.py`'s loop.
- **Control flow** (`systems.System._multiagent_turn`): early or not complex → single zero-shot
  agent; else emotion → cause → intention agents (each sees the previous analyses), top-10
  retrieval over the case bank, 3-agent deliberation, one response per valid strategy (examples
  filtered by strategy), debate and reflection with one agent per candidate, majority vote,
  judge on ties, refiner; no valid strategy → zero-shot agent.
- **AutoGen GroupChat** (`agents.group_chat`): round robin with `max_round = n + 1`; agent i sees
  the admin prompt and the replies of agents 0..i-1 and speaks once; the replies are
  `chat_history[1:]`. The manager's LLM is never called in round-robin mode.
- **Retrieval** (`retriever.Retriever`): case bank = every (seeker post, next supporter response,
  strategy) of `dataset[100:]` (13,484 entries), cosine similarity, top 10, stable tie order,
  pairs formatted `"(post, '[strategy] response')"`.
- Temperature 0; `max_tokens` 400 for analysis, deliberation, debate, reflection, judge and
  refiner; 100 for the decision maker.

## Deviations (all systems alike)
| # | Base behaviour | Ours | Why |
|---|---|---|---|
| 1 | Generation steps use `max_tokens=100` | 160 for zero-shot, per-strategy generation, gate, back-translation | the 30-word limit is in the prompt; Roman-script Hindi needs more sub-word tokens per word, so 100 tokens could cut a 30-word Hinglish reply |
| 2 | Regexes parse the raw output | markdown emphasis/headings are stripped first | `**Response:** [...]` (common in Gemma output) defeats every base regex and silently turns the answer into `"None"` |
| 3 | Strategy names `[a-zA-Z ]+` | letters, spaces, `-`, `&`, `/`, canonicalised to the 8 names (case and aliases) | the base regex can never parse `Self-disclosure` (hyphen), and `[Reflection of Feelings]` vs `[Reflection of feelings]` were counted as different votes |
| 4 | Strategy set = `list(set(...))` | first-mention order | set order depends on `PYTHONHASHSEED`, making the debate order (and the result) non-reproducible |
| 5 | Decision maker: `"yes" in response.lower()` | first `YES`/`NO` token (after `1.` if present), substring check as fallback | the substring check fires on "eyes", "yesterday" inside a NO explanation |
| 6 | Unparseable zero-shot answer → `"None"` | first content line of the answer | a missing `Response:` label is a formatting slip, not an empty answer |
| 7 | Candidates whose generation failed (`"None"`) enter the debate | dropped; if none is left, zero-shot fallback | a `"[X] None"` candidate can win the vote |
| 8 | Vote fails entirely → judge over `["None"]` | judge over all candidates | the base judge then has nothing to choose from |
| 9 | Judge / refiner unparseable → final response `"None"` | first tied candidate / unrefined response | keeps a real response instead of the string `"None"` |
| 10 | A vote with a non-standard tag counts as its own strategy | mapped to the candidate it names (strategy, else ≥50% word overlap) | votes for the same candidate must be counted together |
| 11 | AutoGen passes the speaker in a `name` field; the base setup (LiteLLM → Ollama) most likely dropped it, so each earlier reply reached the agent as a separate, unlabelled user turn | earlier replies are prefixed `agent_j: ` | our client merges consecutive user turns into one message (the Gemini API wants alternating roles), so the prefix keeps the turn boundaries that separate messages gave the base agents; it applies to every system |
| 12 | Retrieved posts/responses with line breaks show a literal `\n` (the case-bank file is written with `.replace("\n", "\\n")`) | identical: `Retriever.pairs()` applies the same escape for display; embeddings use the raw text, as in the base | fidelity of the deliberation and generation prompts |
| 13 | Final text used as produced | `clean_response`: strips a leading `[strategy]` tag, `Assistant:`/`Response:` labels and one pair of wrapping quotes/brackets | the evaluation must compare responses, not formatting |

## CodeMixESC additions (`codemixesc` and ablations only)
- Profiler: HingBERT-LID per word (any label other than EN/HI counts as language independent),
  CMI of Gambäck & Das; R pooled over all seeker utterances so far (each utterance tagged
  separately, long texts chunked so no word is truncated). Words are fed to the model lowercased
  and without apostrophes, the form of the L3Cube-HingLID training data. Names are labelled by
  the model like any other word (the HingLID scheme has no name class; ESConv is anonymised, so
  names are rare).
- Plain-English seekers: common English words that HingLID's training data mostly labels Hindi
  (`to` 53%, `do` 43%, `me` 96%, `he`/`hi` ~100%, `us` 79%, ...; about 6% of the words English
  ESConv users type) can be tagged Hindi in English text. The CMI keeps them, but the decision to
  treat a seeker as code-mixing requires at least 2 Hindi words that are not such homographs,
  making up at least 5% of the seeker's words (`profiler.is_plain_english`). Answering an English
  speaker in Hinglish is a worse error than answering a light code-mixer in English. For such
  seekers every register instruction says "reply in plain English" and the gate's target is 0.
- Register block (`### Language register`) in the emotion, cause and intention prompts (plus
  the Hinglish-expression hint for code-mixed seekers), in the generator (with the reply
  instruction), debate, reflection and refiner; the fourth criterion "language and cultural fit"
  in debate, reflection (voting round) and refiner. Strategy deliberation, the decision maker and
  the judge are unchanged, as in the proposal.
- The single-agent path (early / not complex turns) uses the zero-shot prompt with the register
  instruction, so Hinglish seekers are not answered in English on those turns.
- Register Gate on the final response of every path: one extra call if the register distance
  exceeds δ or the script differs; the rewrite is kept only if its register is closer to the
  seeker's (so the gate cannot make the match worse), otherwise the refined response stays.
  The distance is the Hindi-share gap `|h_r − h_s|` (h = Hindi words / language words). The CMI
  is symmetric (`CMI = min(h, 1 − h)`), so the proposal's `|CMI_r − CMI_s|` cannot tell a
  mostly-Hindi reply from a mostly-English one (an 82%-Hindi reply to an English seeker has a
  CMI gap of 0.18 and would pass δ = 0.2). The Hindi-share gap equals the CMI gap whenever both
  texts lean towards the same language and is larger exactly when the dominant language flips,
  so it adds the dominant-language part of R to the check. `--gate_metric cmi` runs the literal
  CMI criterion. The proposal's CMI gap remains the reported register metric.
- δ is tuned on the dev conversations by `scripts/tune_delta.py` and read from
  `results/tuning/delta.json` (0.2 until then), together with the distance it was tuned for.
- Strategy deliberation, the decision maker and the tie-breaking judge use the original prompts
  (proposal, Fig. 1). docs/PLAN.md listed the judge among the modified prompts; the proposal is
  followed.

## Robustness of the LLM client
- Empty or blocked answers are retried and never cached, so a later run asks again instead of
  replaying a failure; per-turn counts are kept in `n_failed_calls`.
- Concurrent identical prompts make one request and receive the same answer (first writer wins
  across processes), so ablations that share stages see exactly the same upstream outputs.
- Requests reserve an estimated token count and are settled with the count the API reports, so
  the tokens-per-minute window is not throttled by the estimate's safety margin.
- Daily quotas follow Pacific time with daylight saving (Gemini resets at Pacific midnight).

## Translate-pivot baseline
One call translates the whole dialogue context into English (JSON, up to 3 attempts; turns that
cannot be translated are kept as they are), MultiAgentESC runs unchanged on the English context
(including retrieval with the English encoder on the translated post), and one call translates
the final response into the seeker's register (profile R). Evaluated on light and heavy only.
