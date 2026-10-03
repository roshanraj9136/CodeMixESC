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
| 11 | AutoGen passes the speaker in a `name` field | earlier replies are prefixed `agent_j: ` | our backends have no `name` field; otherwise the agents cannot tell who said what |
| 12 | Final text used as produced | `clean_response`: strips a leading `[strategy]` tag, `Assistant:`/`Response:` labels and one pair of wrapping quotes/brackets | the evaluation must compare responses, not formatting |

## CodeMixESC additions (`codemixesc` and ablations only)
- Profiler: HingBERT-LID per word (any label other than EN/HI counts as language independent),
  CMI of Gambäck & Das; R pooled over all seeker utterances so far (each utterance tagged
  separately, long texts chunked so no word is truncated).
- Register block (`### Language register`) in the emotion, cause and intention prompts (plus
  the Hinglish-expression hint for code-mixed seekers), in the generator (with the reply
  instruction), debate, reflection and refiner; the fourth criterion "language and cultural fit"
  in debate, reflection (voting round) and refiner. Strategy deliberation, the decision maker and
  the judge are unchanged, as in the proposal.
- The single-agent path (early / not complex turns) uses the zero-shot prompt with the register
  instruction, so Hinglish seekers are not answered in English on those turns.
- Register Gate on the final response of every path: one extra call if `|CMI_r − CMI_s| > δ`
  or the script differs; the rewrite is kept only if its register is closer to the seeker's
  (so the gate cannot make the match worse), otherwise the refined response stays.
- δ is tuned on the dev conversations by `scripts/tune_delta.py` and read from
  `results/tuning/delta.json` (0.2 until then).

## Translate-pivot baseline
One call translates the whole dialogue context into English (JSON, up to 3 attempts; turns that
cannot be translated are kept as they are), MultiAgentESC runs unchanged on the English context
(including retrieval with the English encoder on the translated post), and one call translates
the final response into the seeker's register (profile R). Evaluated on light and heavy only.
