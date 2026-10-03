"""The agents of MultiAgentESC on the cached LLM client, plus the CodeMixESC Register Gate.

Every step mirrors the function of the same name in external/MultiAgentESC/multiagent.py
(same prompts, token limits and control flow). AutoGen's round-robin GroupChat with
max_round = n_agents + 1 is replicated by group_chat(): the admin posts the prompt, then
agent i speaks once and sees the prompt and the replies of agents 0..i-1 (AutoGen broadcasts
every message to every agent); chat_history[1:] is the list of the n replies. AutoGen passes
the speaker's name in a separate field; our backends have none, so earlier replies are
prefixed with "agent_j: ".

Output parsing follows the base regexes, hardened in the same way for every system, so a
different LLM's formatting habits cannot silently break the pipeline (docs/IMPLEMENTATION.md):
- markdown emphasis and headings are removed first ("**Response:** [...]" defeats the base regex);
- strategy names may contain '-' (the base [a-zA-Z ]+ can never parse "Self-disclosure");
- strategy names are canonicalised (case, aliases) so votes for one strategy are counted together;
- the strategy set keeps first-mention order (the base list(set(...)) depends on PYTHONHASHSEED);
- when an output cannot be parsed, the last valid response is kept instead of the string "None".
"""
import json
import re
from collections import Counter

from . import prompts as P

GEN_MAX_TOKENS = 160  # base: 100. The 30-word limit is in the prompt; Roman Hindi needs more sub-word tokens.
ANALYSIS_MAX_TOKENS = 400
DECIDE_MAX_TOKENS = 100

_STRAT = r"[A-Za-z][A-Za-z &/\-]*"
_ALIASES = {
    "question": "Question", "questions": "Question", "open-ended question": "Question",
    "open ended question": "Question", "asking questions": "Question",
    "restatement or paraphrasing": "Restatement or Paraphrasing", "restatement and paraphrasing":
        "Restatement or Paraphrasing", "restatement/paraphrasing": "Restatement or Paraphrasing",
    "restatement": "Restatement or Paraphrasing", "paraphrasing": "Restatement or Paraphrasing",
    "reflection of feelings": "Reflection of feelings", "reflection of feeling": "Reflection of feelings",
    "reflecting feelings": "Reflection of feelings", "reflection": "Reflection of feelings",
    "self-disclosure": "Self-disclosure", "self disclosure": "Self-disclosure", "selfdisclosure": "Self-disclosure",
    "affirmation and reassurance": "Affirmation and Reassurance", "affirmation": "Affirmation and Reassurance",
    "reassurance": "Affirmation and Reassurance",
    "providing suggestions": "Providing Suggestions", "providing suggestion": "Providing Suggestions",
    "provide suggestions": "Providing Suggestions", "suggestions": "Providing Suggestions",
    "suggestion": "Providing Suggestions",
    "information": "Information", "providing information": "Information", "provide information": "Information",
    "others": "Others", "other": "Others",
}
_ALIAS_KEYS = sorted(_ALIASES, key=len, reverse=True)


# ------------------------------------------------------------------ call accounting
class CallLog:
    """Wraps an LLM for one turn and records every logical call (cached or not), so that LLM
    calls and latency per turn can be reported. Latency is the API time of the original call."""

    def __init__(self, llm):
        self.llm = llm
        self.calls = []

    def chat(self, messages, system=P.SYSTEM, max_tokens=ANALYSIS_MAX_TOKENS, tag="", temperature=0.0):
        text, meta = self.llm.chat(messages, system=system, temperature=temperature, max_tokens=max_tokens, tag=tag)
        self.calls.append({"tag": tag, "cached": bool(meta.get("cached")), "real": int(meta.get("calls", 0)),
                           "latency": float(meta.get("latency") or 0.0)})
        return text or ""

    def ask(self, prompt, system=P.SYSTEM, max_tokens=ANALYSIS_MAX_TOKENS, tag=""):
        return self.chat([{"role": "user", "content": prompt}], system=system, max_tokens=max_tokens, tag=tag)

    def summary(self):
        return {"n_calls": len(self.calls), "n_real_calls": sum(c["real"] for c in self.calls),
                "latency": round(sum(c["latency"] for c in self.calls), 3),
                "calls_by_tag": dict(Counter(re.sub(r"\d+$", "", c["tag"]) for c in self.calls))}


# ------------------------------------------------------------------ parsing helpers
def strip_think(text):
    text = text or ""
    return text.split("</think>", 1)[1].strip() if "</think>" in text else text


def plain(text):
    """Removes reasoning blocks and markdown decoration before regex parsing."""
    text = strip_think(text)
    text = text.replace("**", "").replace("__", "")
    text = re.sub(r"^[ \t]*#+[ \t]*", "", text, flags=re.M)
    return text.strip()


def canonical_strategy(name):
    """Maps an LLM's spelling of a strategy to one of the 8 ESConv names, or None."""
    if not name:
        return None
    s = re.sub(r"\s+", " ", str(name).strip().strip("[]()*:.\"'`").replace("&", "and").replace("_", " ")).lower()
    for key in _ALIAS_KEYS:
        if s == key or (s.startswith(key) and not s[len(key)].isalnum()):
            return _ALIASES[key]
    return None


def parse_strategies(text):
    """Strategies named on 'Strategy:' lines, canonical, in first-mention order
    (base: clean_strategy(re.findall(r'Strategy:\\s*\\[?([a-zA-Z ]+)\\]?', ...)))."""
    found = []
    for m in re.finditer(r"Strategy\s*:\s*\[?\s*(" + _STRAT + ")", plain(text)):
        c = canonical_strategy(m.group(1))
        if c and c not in found:
            found.append(c)
    return found


def _unwrap(text):
    """Removes one pair of quotes/brackets around the whole text (not around a part of it)."""
    t = (text or "").strip()
    for a, b in (('"', '"'), ("'", "'"), ("“", "”"), ("[", "]"), ("(", ")")):
        if len(t) > 1 and t.startswith(a) and t.endswith(b):
            inner = t[1:-1]
            if (a == b and a not in inner) or (a != b and a not in inner and b not in inner):
                t = inner.strip()
    return t


def _cut_reasoning(text):
    return re.split(r"\s+Reasoning\s*:", text, maxsplit=1)[0].strip()


def _split_tag(text):
    """'[Question] How ...' / 'Question: How ...' / '(Question) How ...' -> ('Question', 'How ...')."""
    t = (text or "").strip()
    m = re.match(r"^[\[(]\s*(" + _STRAT + r")\s*[\])]\s*[:\-–]?\s*(.*)$", t, re.S)
    if m and (canonical_strategy(m.group(1)) or (m.group(2).strip() and len(m.group(1).split()) <= 4)):
        return canonical_strategy(m.group(1)), m.group(2).strip()  # a non-standard tag is dropped too
    m = re.match(r"^(" + _STRAT + r")\s*[:\-–]\s+(.*)$", t, re.S)
    if m and canonical_strategy(m.group(1)) and len(m.group(1).split()) <= 4:
        return canonical_strategy(m.group(1)), m.group(2).strip()
    return None, t


def parse_tagged(text):
    """'Response: [strategy] response' -> (strategy or None, response or None).
    Base regex: Response:\\s*\\[([a-zA-Z ]+)\\](.*) on one line."""
    t = plain(text)
    m = re.search(r"Response\s*:\s*(.*)", t)
    if not m:
        return None, None
    strategy, rest = _split_tag(_cut_reasoning(m.group(1)))
    rest = _unwrap(rest)
    return strategy, (rest or None)


def parse_single(text):
    """Zero-shot answer (base: re.findall(r'Response:\\s*(.*)', response)[0]); falls back to the
    first content line when the label is missing, instead of returning "None"."""
    t = plain(text)
    m = re.search(r"Response\s*:\s*(.*)", t)
    if m:
        return _unwrap(_split_tag(_cut_reasoning(m.group(1)))[1]) or "None"
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    lines = [ln for ln in lines if not re.match(r"^(Reasoning|Strategy|Emotion|Event|Intention)\s*:", ln)]
    return _unwrap(lines[0]) if lines else "None"


def parse_label(text, label, default):
    """First 'Label: value' (base: only the first line, default 'Negative' / 'Not mention')."""
    m = re.search(label + r"\s*:\s*(.+)", plain(text))
    return m.group(1).strip() if m else default


def parse_yes_no(text):
    """Decision maker. Base: '"yes" in response.lower()', which also fires on 'eyes' or
    'yesterday' inside the explanation; we read the first YES/NO token and fall back to it."""
    t = plain(text)
    m = re.search(r"1\.\s*\[?\s*(YES|NO)\b", t, re.I) or re.search(r"\b(YES|NO)\b", t, re.I)
    return m.group(1).upper() == "YES" if m else "yes" in t.lower()


_PLACEHOLDERS = {"", "none", "response", "[response]", "[strategy] [response]", "strategy", "[strategy]",
                 "origianl/refined response", "[origianl/refined response]", "rewritten response",
                 "[rewritten response]", "translated reply", "[translated reply]"}


def usable(response):
    """False for empty answers and for echoes of the answer template."""
    return bool(response) and response.strip().lower() not in _PLACEHOLDERS


def clean_response(text):
    """Final clean-up applied to the output of every system alike."""
    t = re.sub(r"\s+", " ", (text or "")).strip()
    t = re.sub(r"^(Assistant|Response)\s*:\s*", "", t, flags=re.I)
    t = _split_tag(t)[1]
    t = _unwrap(t)
    return t or "None"


def _overlap(a, b):
    ta, tb = set(re.findall(r"\w+", a.lower())), set(re.findall(r"\w+", b.lower()))
    return len(ta & tb) / max(1, min(len(ta), len(tb)))


def match_candidate(strategy, response, candidates):
    """The candidate (strategy, response) an agent's answer refers to, or None."""
    names = [s for s, _ in candidates]
    if strategy in names:
        return candidates[names.index(strategy)]
    if response:
        best = max(candidates, key=lambda c: _overlap(response, c[1]))
        if _overlap(response, best[1]) >= 0.5:
            return best
    return None


# ------------------------------------------------------------------ AutoGen GroupChat replica
def group_chat(log, prompt, personas, max_tokens=ANALYSIS_MAX_TOKENS, tag="group"):
    replies = []
    for i, persona in enumerate(personas):
        msgs = [{"role": "user", "content": prompt}]
        msgs += [{"role": "user", "content": f"agent_{j}: {r}"} for j, r in enumerate(replies)]
        replies.append(log.chat(msgs, system=persona, max_tokens=max_tokens, tag=f"{tag}{i}"))
    return replies


# ------------------------------------------------------------------ Stage 1: dialogue analysis
def is_complex(log, context):
    raw = log.ask(P.BEHAVIOR_CONTROL.format(context=context), max_tokens=DECIDE_MAX_TOKENS, tag="decide")
    return parse_yes_no(raw), raw


def single_agent_response(log, context, R=None, tag="single"):
    tpl = P.ZERO_SHOT if R is None else P.zero_shot_register(R)
    raw = log.ask(tpl.format(context=context), max_tokens=GEN_MAX_TOKENS, tag=tag)
    return parse_single(raw), raw


def analyse(log, context, R=None):
    """Emotion -> cause -> intention agents (each sees the previous analyses), as in main.py."""
    tpl = (lambda t: t) if R is None else (lambda t: P.analysis_prompt(t, R))
    emo = strip_think(log.ask(tpl(P.GET_EMOTION).format(context=context), tag="emotion"))
    cau = strip_think(log.ask(tpl(P.GET_CAUSE).format(emo_and_reason=emo, context=context), tag="cause"))
    inten = strip_think(log.ask(tpl(P.GET_INTENTION).format(emo_and_reason=emo, cau_and_reason=cau, context=context),
                                tag="intention"))
    return {"emotion": parse_label(emo, "Emotion", "Negative"), "cause": parse_label(cau, "Event", "Not mention"),
            "intention": parse_label(inten, "Intention", "Not mention"),
            "emo_and_reason": emo, "cau_and_reason": cau, "int_and_reason": inten}


# ------------------------------------------------------------------ Stage 2: strategy deliberation
def get_strategy(log, retriever, context, an, post, n=10, agent_num=3):
    pairs, idx = retriever.pairs(post, k=n)
    examples = "\n\n".join(f"{p}\n{r}" for p, r in pairs)
    prompt = P.SELECT_STRATEGY.format(context=context, emo_and_reason=an["emo_and_reason"],
                                      cau_and_reason=an["cau_and_reason"], int_and_reason=an["int_and_reason"],
                                      examples=examples)
    replies = group_chat(log, prompt, [P.SYSTEM] * agent_num, tag="deliberate")
    return parse_strategies(" ".join(replies)), pairs, [int(i) for i in idx], replies


def examples_for(pairs, strategy):
    """Retrieved cases of one strategy, '<post\\n[strategy] response>' blocks (main.py)."""
    out = ""
    for post, tagged in pairs:
        if tagged.split("]", 1)[0].strip("[").strip() == strategy:
            out += f"{post}\n{tagged}\n\n"
    return out.strip()


# ------------------------------------------------------------------ Stage 3: response generation
def response_with_strategy(log, context, an, strategy, examples, R=None):
    tpl = P.RESPONSE_WITH_STRATEGY if R is None else P.response_with_strategy_register(R)
    raw = log.ask(tpl.format(context=context, emo_and_reason=an["emo_and_reason"], cau_and_reason=an["cau_and_reason"],
                             int_and_reason=an["int_and_reason"], strategy=strategy, examples=examples),
                  max_tokens=GEN_MAX_TOKENS, tag="generate")
    _, resp = parse_tagged(raw)
    if not resp:  # base fallback: text after the strategy name, else "None"
        t = plain(raw)
        resp = _unwrap(t.split(strategy, 1)[1].lstrip("]):- ").strip()) if strategy in t else None
    return resp or "None", raw


def debate(log, context, an, responses, R=None):
    tpl = P.DEBATE if R is None else P.debate_register(R)
    prompt = tpl.format(context=context, emo_and_reason=an["emo_and_reason"], cau_and_reason=an["cau_and_reason"],
                        int_and_reason=an["int_and_reason"], responses_template="\n\n".join(responses))
    return group_chat(log, prompt, [P.DEBATE_PERSONA.format(response=r) for r in responses], tag="debate")


def reflect(log, context, an, debate_history, responses, R=None):
    tpl = P.REFLECT if R is None else P.reflect_register(R)
    prompt = tpl.format(context=context, emo_and_reason=an["emo_and_reason"], cau_and_reason=an["cau_and_reason"],
                        int_and_reason=an["int_and_reason"], discussion_content="\n\n".join(debate_history))
    return group_chat(log, prompt, [P.REFLECT_PERSONA.format(response=r) for r in responses], tag="reflect")


def vote(results, candidates):
    """Majority vote over the reflection round (base vote()). Each answer counts for the
    candidate it names; the text the first voter gave for a strategy is kept, as in the base."""
    count, s2r = {}, {}
    for result in results:
        head = re.split(r"\n\s*Reasoning", plain(result), maxsplit=1)[0]
        strategy, response = parse_tagged(head)
        cand = match_candidate(strategy, response, candidates)
        if cand is None:
            continue
        count[cand[0]] = count.get(cand[0], 0) + 1
        s2r.setdefault(cand[0], response if usable(response) else cand[1])
    if not count:
        return ["None"], ["None"]
    top = max(count.values())
    winners = [s for s, c in count.items() if c == top]
    return winners, [s2r[s] for s in winners]


def judge(log, context, pool):
    """Tie-breaking judge (unchanged prompt). pool: list of (strategy, response)."""
    template = "\n\n".join(f"[{s}] {r}" for s, r in pool)
    raw = log.ask(P.JUDGE.format(context=context, template_responses=template), tag="judge")
    strategy, response = parse_tagged(raw)
    cand = match_candidate(strategy, response, pool)
    if cand is None:  # unparseable verdict: first tied candidate instead of the base's "None"
        return pool[0][0], pool[0][1], raw
    return cand[0], response if usable(response) else cand[1], raw


def self_reflection(log, context, pred_strategy, response, R=None):
    """Refiner. Its strategy tag is discarded as in main.py; an unparseable answer keeps the input."""
    tpl = P.SELF_REFLECTION if R is None else P.self_reflection_register(R)
    raw = log.ask(tpl.format(context=context, pred_strategy=pred_strategy, response=response), tag="refine")
    _, refined = parse_tagged(raw)
    return (refined if usable(refined) else response), raw


# ------------------------------------------------------------------ CodeMixESC Register Gate
def script_mismatch(R, reg):
    return int(reg["script"] not in ("None", R["script"]) and R["script"] != "None")


def register_gate(log, profiler, context, R, strategy, response, delta):
    """If |CMI_r - CMI_s| > delta or the script differs, refine once more with an explicit
    register instruction (at most one extra call). The rewrite is kept only if its register is
    closer to the seeker's, so the gate can never make the register match worse."""
    before = profiler.register_of(response)
    gap = abs(before["cmi"] - R["cmi"])
    bad_script = script_mismatch(R, before)
    info = {"triggered": False, "accepted": False, "delta": delta, "cmi_s": R["cmi"], "cmi_before": before["cmi"],
            "script_before": before["script"], "cmi_after": before["cmi"], "script_after": before["script"],
            "calls": 0, "latency": 0.0}
    if gap <= delta and not bad_script:
        return response, info
    strategy = strategy if strategy in P.STRATEGIES else None
    raw = log.ask(P.gate_prompt(context, R, response, strategy, P.describe_mismatch(R, before)),
                  max_tokens=GEN_MAX_TOKENS, tag="gate")
    info.update(triggered=True, calls=1, latency=log.calls[-1]["latency"])
    new = parse_tagged(raw)[1] if strategy else parse_single(raw)
    new = clean_response(new) if usable(new) else None
    if not new:
        return response, info
    after = profiler.register_of(new)
    info.update(candidate=new, cmi_after=after["cmi"], script_after=after["script"])
    bad_after = script_mismatch(R, after)
    if bad_after < bad_script or (bad_after == bad_script and abs(after["cmi"] - R["cmi"]) < gap - 1e-9):
        info["accepted"] = True
        return new, info
    info.update(cmi_after=before["cmi"], script_after=before["script"])
    return response, info


# ------------------------------------------------------------------ translate-pivot baseline
def parse_turns(text, n):
    """{"turns": [{"id": i, "text": ...}]} (or a bare list) -> {id: text}."""
    t = strip_think(text).strip()
    m = re.search(r"\{.*\}|\[.*\]", t, re.S)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    items = obj.get("turns", []) if isinstance(obj, dict) else obj
    out = {}
    for it in items if isinstance(items, list) else []:
        if isinstance(it, dict) and "id" in it and isinstance(it.get("text"), str):
            try:
                i = int(it["id"])
            except (TypeError, ValueError):
                continue
            if 0 <= i < n and it["text"].strip():
                out[i] = it["text"].strip()
    return out


def translate_context(log, msgs, attempts=3):
    """Hinglish -> English translation of the whole dialogue context in one call (first extra
    call of the pivot). Untranslatable turns are kept as they are."""
    got = {}
    for attempt in range(attempts):
        raw = log.ask(P.translate_in_prompt(msgs, attempt), max_tokens=4096, tag="translate_in")
        got = parse_turns(raw, len(msgs))
        if len(got) == len(msgs):
            break
    return [{"role": m["role"], "content": got.get(i, m["content"])} for i, m in enumerate(msgs)], len(got) == len(msgs)


def translate_response(log, response, R):
    """English -> the seeker's register (second extra call of the pivot)."""
    raw = log.ask(P.TRANSLATE_OUT.format(register_instruction=P.register_instruction(R), response=response),
                  max_tokens=GEN_MAX_TOKENS, tag="translate_out")
    out = parse_single(raw)
    return out if usable(out) else response
