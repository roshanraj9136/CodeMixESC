# Run record format

`scripts/run_system.py` writes one JSON object per supporter turn to
`results/runs/{system}/{version}.jsonl` (dev runs: `{system}/dev_{version}.jsonl`), sorted by
conversation and turn, plus `{version}.meta.json` (configuration, model, δ, git commit, counts).
Turns that failed after all API retries go to `{version}.errors.jsonl` and are retried on the
next run; they are never written as records.

## Fields used by the evaluation

| Field | Type | Meaning |
|---|---|---|
| `uid` | str | `"{conv}-{turn}"`; identical across `en`, `light`, `heavy` (parallel versions) |
| `conv_id`, `turn` | int | test conversation index (0–99) or ESConv index for dev; dialog index of the turn |
| `version` | str | `en`, `light` or `heavy` |
| `system` | str | `zero_shot`, `fewshot_cot`, `maesc`, `pivot`, `codemixesc`, `cmx_noft`, `cmx_noxl` (or a custom `--name`) |
| `early` | bool | `count <= 5` in the base loop: answered by the single zero-shot agent |
| `gold_strategy` | str | ESConv strategy of the reference; merged turns read `"A and B"` |
| `reference` | str | gold supporter response in this version (Hinglish rewrite for `light`/`heavy`) |
| `post` | str | last utterance of the context (the retrieval query, as in the base code) |
| `path` | str | `single` (early turn, or the decision maker said the dialogue is not complex), `fallback` (no valid strategy after deliberation, zero-shot answer), `multi` (full pipeline) |
| `complex` | bool/null | decision maker's verdict; `null` for early turns (not asked) |
| `R` | object/null | seeker register profile `{cmi, cmi_last, hi_frac, dominant, script}` (systems with a profiler) |
| `pred_strategy` | str | one of the 8 ESConv strategies, or `"None"` (single/fallback paths and zero-shot) |
| `response` | str | **final response — the text that is evaluated** |
| `pre_gate_response` | str | response before the Register Gate (= `response` for systems without a gate); `cmx_nogate` is evaluated on this field of the `codemixesc` run |
| `gate` | object/null | `{triggered, accepted, delta, cmi_s, cmi_before, script_before, cmi_after, script_after, calls, latency[, candidate]}` |
| `n_calls` | int | logical LLM calls of the turn (cached or not) — the efficiency metric |
| `n_real_calls` | int | API requests made in this run (0 for cache hits; retries of blocked answers count) |
| `latency` | float | sum of the API latencies of the turn's calls, in seconds (cache hits report the latency measured when the call was made) |
| `calls_by_tag` | object | calls per step: `decide, single, fallback, emotion, cause, intention, deliberate, generate, debate, reflect, judge, refine, gate, translate_in, translate_out, fewshot` |

`cmx_nogate` = `codemixesc` with `response := pre_gate_response`, `n_calls -= gate.calls`,
`latency -= gate.latency`.

## Diagnostic fields (case studies)

`decide_raw` (decision maker output), `analysis` (`emotion`, `cause`, `intention` labels),
`retrieved` (indices of the 10 retrieved case-bank entries), `deliberation` (3 agent replies),
`strategies` (valid strategies after deliberation), `candidates` (`[strategy, response]` per
strategy), `debate` and `reflection` (agent replies), `vote` (`{strategies, responses}`),
`judge` (whether the tie-breaking judge was used), `ori_response` (winner before the refiner),
`refined_response` (after the refiner), `raw` (single-call baselines' raw output), and for the
pivot `pivot` = `{context_en, post_en, response_en, translation_complete}`.
