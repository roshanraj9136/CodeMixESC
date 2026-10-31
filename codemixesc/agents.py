import json
import re
from collections import Counter

from . import prompts as P
from .profiler import register_distance, target_cmi, target_hi_frac

GEN_MAX_TOKENS = 160
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


class CallLog:

    def __init__(self, llm):
        self.llm = llm
        self.calls = []

    def chat(self, messages, system=P.SYSTEM, max_tokens=ANALYSIS_MAX_TOKENS, tag="", temperature=0.0):
        text, meta = self.llm.chat(messages, system=system, temperature=temperature, max_tokens=max_tokens, tag=tag)
        self.calls.append({"tag": tag, "cached": bool(meta.get("cached")), "real": int(meta.get("calls", 0)),
                           "latency": float(meta.get("latency") or 0.0), "failed": bool(meta.get("failed")),
                           "truncated": meta.get("finish") in ("MAX_TOKENS", "length")})
        return text or ""

    def ask(self, prompt, system=P.SYSTEM, max_tokens=ANALYSIS_MAX_TOKENS, tag=""):
        return self.chat([{"role": "user", "content": prompt}], system=system, max_tokens=max_tokens, tag=tag)

    def summary(self):
        return {"n_calls": len(self.calls), "n_real_calls": sum(c["real"] for c in self.calls),
                "latency": round(sum(c["latency"] for c in self.calls), 3),
                "n_failed_calls": sum(c["failed"] for c in self.calls),
                "n_truncated_calls": sum(c["truncated"] for c in self.calls),
                "calls_by_tag": dict(Counter(re.sub(r"\d+$", "", c["tag"]) for c in self.calls))}


def strip_think(text):
    text = text or ""
    return text.split("</think>", 1)[1].strip() if "</think>" in text else text


def plain(text):
    text = strip_think(text)
    text = text.replace("**", "").replace("__", "").replace("`", "")
    text = re.sub(r"(?<![\w*])\*(?=\S)([^*\n]+?)(?<=\S)\*(?![\w*])", r"\1", text)
    text = re.sub(r"^[ \t]*#+[ \t]*", "", text, flags=re.M)
    return text.strip()


def canonical_strategy(name):
    if not name:
        return None
    s = re.sub(r"\s+", " ", str(name).strip().strip("[]()*:.\"'`").replace("&", "and").replace("_", " ")).lower()
    for key in _ALIAS_KEYS:
        if s == key or (s.startswith(key) and not s[len(key)].isalnum()):
            return _ALIASES[key]
    return None


def parse_strategies(text):
    found = []
    for m in re.finditer(r"Strategy\s*:\s*[\[(\"'“]?\s*(" + _STRAT + ")", plain(text)):
        c = canonical_strategy(m.group(1))
        if c and c not in found:
            found.append(c)
    return found


def _unwrap(text):
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
    t = (text or "").strip()
    m = re.match(r"^[\[(]\s*(" + _STRAT + r")\s*[\])]\s*[:\-–]?\s*(.*)$", t, re.S)
    if m and (canonical_strategy(m.group(1)) or (m.group(2).strip() and len(m.group(1).split()) <= 4)):
        return canonical_strategy(m.group(1)), m.group(2).strip()
    m = re.match(r"^(" + _STRAT + r")\s*[:\-–]\s+(.*)$", t, re.S)
    if m and canonical_strategy(m.group(1)) and len(m.group(1).split()) <= 4:
        return canonical_strategy(m.group(1)), m.group(2).strip()
    return None, t


def parse_tagged(text):
    t = plain(text)
    m = re.search(r"Response\s*:\s*(.*)", t)
    if not m:
        return None, None
    strategy, rest = _split_tag(_cut_reasoning(m.group(1)))
    rest = _unwrap(rest)
    return strategy, (rest or None)


def parse_single(text):
    t = plain(text)
    m = re.search(r"Response\s*:\s*(.*)", t)
    if m:
        return _unwrap(_split_tag(_cut_reasoning(m.group(1)))[1]) or "None"
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    lines = [ln for ln in lines if not re.match(r"^(Reasoning|Strategy|Emotion|Event|Intention)\s*:", ln)]
    return _unwrap(lines[0]) if lines else "None"


def parse_label(text, label, default):
    m = re.search(label + r"\s*:\s*(.+)", plain(text))
    return m.group(1).strip() if m else default


def parse_yes_no(text):
    t = plain(text)
    m = re.search(r"1\.\s*\[?\s*(YES|NO)\b", t, re.I) or re.search(r"\b(YES|NO)\b", t, re.I)
    return m.group(1).upper() == "YES" if m else "yes" in t.lower()


_PLACEHOLDERS = {"", "none", "response", "[response]", "[strategy] [response]", "strategy", "[strategy]",
                 "origianl/refined response", "[origianl/refined response]", "rewritten response",
                 "[rewritten response]", "translated reply", "[translated reply]"}


_META_RE = re.compile(r"^\W*(the\s+)?(original|origianl|refined)(\s*(/|or)\s*(original|origianl|refined))?\s+"
                      r"(response|version)\b", re.I)


def usable(response):
    t = (response or "").strip()
    return bool(t) and t.lower() not in _PLACEHOLDERS and not _META_RE.match(t)


_TRAILING_NOTE = re.compile(r"\s*[(\[](?:translation|translated|english|in english|meaning|\d+\s+words?)\b"
                            r"[^)\]]*[)\]]\s*$", re.I)


def clean_response(text):
    t = re.sub(r"\s+", " ", (text or "")).strip()
    while True:
        prev = t
        t = re.sub(r"^(Assistant|Response)\s*:\s*", "", t, flags=re.I)
        t = _split_tag(t)[1]
        t = _TRAILING_NOTE.sub("", t)
        t = _unwrap(t)
        if t == prev:
            return t or "None"


def _overlap(a, b):
    ta, tb = set(re.findall(r"\w+", a.lower())), set(re.findall(r"\w+", b.lower()))
    return len(ta & tb) / max(1, min(len(ta), len(tb)))


def match_candidate(strategy, response, candidates):
    names = [s for s, _ in candidates]
    if strategy in names:
        return candidates[names.index(strategy)]
    if response:
        best = max(candidates, key=lambda c: _overlap(response, c[1]))
        if _overlap(response, best[1]) >= 0.5:
            return best
    return None


def group_chat(log, prompt, personas, max_tokens=ANALYSIS_MAX_TOKENS, tag="group"):
    replies = []
    for i, persona in enumerate(personas):
        msgs = [{"role": "user", "content": prompt}]
        msgs += [{"role": "user", "content": f"agent_{j}: {r}"} for j, r in enumerate(replies)]
        replies.append(log.chat(msgs, system=persona, max_tokens=max_tokens, tag=f"{tag}{i}"))
    return replies


def is_complex(log, context):
    raw = log.ask(P.BEHAVIOR_CONTROL.format(context=context), max_tokens=DECIDE_MAX_TOKENS, tag="decide")
    return parse_yes_no(raw), raw


def single_agent_response(log, context, R=None, tag="single"):
    tpl = P.ZERO_SHOT if R is None else P.zero_shot_register(R)
    raw = log.ask(tpl.format(context=context), max_tokens=GEN_MAX_TOKENS, tag=tag)
    return parse_single(raw), raw


def analyse(log, context, R=None):
    tpl = (lambda t: t) if R is None else (lambda t: P.analysis_prompt(t, R))
    emo = strip_think(log.ask(tpl(P.GET_EMOTION).format(context=context), tag="emotion"))
    cau = strip_think(log.ask(tpl(P.GET_CAUSE).format(emo_and_reason=emo, context=context), tag="cause"))
    inten = strip_think(log.ask(tpl(P.GET_INTENTION).format(emo_and_reason=emo, cau_and_reason=cau, context=context),
                                tag="intention"))
    return {"emotion": parse_label(emo, "Emotion", "Negative"), "cause": parse_label(cau, "Event", "Not mention"),
            "intention": parse_label(inten, "Intention", "Not mention"),
            "emo_and_reason": emo, "cau_and_reason": cau, "int_and_reason": inten}


def get_strategy(log, retriever, context, an, post, n=10, agent_num=3):
    pairs, idx = retriever.pairs(post, k=n)
    examples = "\n\n".join(f"{p}\n{r}" for p, r in pairs)
    prompt = P.SELECT_STRATEGY.format(context=context, emo_and_reason=an["emo_and_reason"],
                                      cau_and_reason=an["cau_and_reason"], int_and_reason=an["int_and_reason"],
                                      examples=examples)
    replies = group_chat(log, prompt, [P.SYSTEM] * agent_num, tag="deliberate")
    return parse_strategies(" ".join(replies)), pairs, [int(i) for i in idx], replies


def examples_for(pairs, strategy):
    out = ""
    for post, tagged in pairs:
        if tagged.split("]", 1)[0].strip("[").strip() == strategy:
            out += f"{post}\n{tagged}\n\n"
    return out.strip()


def response_with_strategy(log, context, an, strategy, examples, R=None):
    tpl = P.RESPONSE_WITH_STRATEGY if R is None else P.response_with_strategy_register(R)
    raw = log.ask(tpl.format(context=context, emo_and_reason=an["emo_and_reason"], cau_and_reason=an["cau_and_reason"],
                             int_and_reason=an["int_and_reason"], strategy=strategy, examples=examples),
                  max_tokens=GEN_MAX_TOKENS, tag="generate")
    _, resp = parse_tagged(raw)
    if not resp:
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
    template = "\n\n".join(f"[{s}] {r}" for s, r in pool)
    raw = log.ask(P.JUDGE.format(context=context, template_responses=template), tag="judge")
    strategy, response = parse_tagged(raw)
    cand = match_candidate(strategy, response, pool)
    if cand is None:
        return pool[0][0], pool[0][1], raw
    return cand[0], response if usable(response) else cand[1], raw


def self_reflection(log, context, pred_strategy, response, R=None):
    tpl = P.SELF_REFLECTION if R is None else P.self_reflection_register(R)
    raw = log.ask(tpl.format(context=context, pred_strategy=pred_strategy, response=response), tag="refine")
    _, refined = parse_tagged(raw)
    return (refined if usable(refined) else response), raw


def script_mismatch(R, reg):
    return int(reg["script"] not in ("None", R["script"]) and R["script"] != "None")


def register_gate(log, profiler, context, R, strategy, response, delta, metric="hi_frac"):
    before = profiler.register_of(response)
    dist = register_distance(R, before, metric)
    bad_script = script_mismatch(R, before)
    unusable = not usable(response)
    info = {"triggered": False, "accepted": False, "delta": delta, "metric": metric, "cmi_s": R["cmi"],
            "cmi_target": target_cmi(R), "hi_frac_target": target_hi_frac(R), "cmi_before": before["cmi"],
            "hi_frac_before": before["hi_frac"], "script_before": before["script"], "distance_before": dist,
            "cmi_after": before["cmi"], "hi_frac_after": before["hi_frac"], "script_after": before["script"],
            "distance_after": dist, "calls": 0, "latency": 0.0}
    if (dist <= delta and not bad_script) or unusable:
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
    dist_after = register_distance(R, after, metric)
    info["candidate"] = new
    bad_after = script_mismatch(R, after)
    if bad_after < bad_script or (bad_after == bad_script and dist_after < dist - 1e-9):
        info.update(accepted=True, cmi_after=after["cmi"], hi_frac_after=after["hi_frac"], script_after=after["script"],
                    distance_after=dist_after)
        return new, info
    info["candidate_distance"] = dist_after
    return response, info


def parse_turns(text, n):
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
    got = {}
    for attempt in range(attempts):
        raw = log.ask(P.translate_in_prompt(msgs, attempt), max_tokens=8192, tag="translate_in")
        got = parse_turns(raw, len(msgs))
        if len(got) == len(msgs):
            break
    return [{"role": m["role"], "content": got.get(i, m["content"])} for i, m in enumerate(msgs)], len(got) == len(msgs)


def translate_response(log, response, R):
    raw = log.ask(P.TRANSLATE_OUT.format(register_instruction=P.register_instruction(R), response=response),
                  max_tokens=GEN_MAX_TOKENS, tag="translate_out")
    out = parse_single(raw)
    return out if usable(out) else response
