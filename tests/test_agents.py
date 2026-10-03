"""Multi-agent pipeline: fidelity to the base code (prompts, turn extraction, AutoGen group
chat), output parsers, vote, Register Gate, and every system end to end with stand-ins."""
import ast
import json
import os
import re

import pytest

from codemixesc import agents as A
from codemixesc import prompts as P
from codemixesc.esconv import ROOT, all_samples, load_esconv, sampled_uids, turn_samples
from codemixesc.profiler import is_plain_english
from codemixesc.systems import SYSTEMS, System, fewshot_examples
from codemixesc.testing import FakeLLM, HashRetriever, LexiconProfiler, write_fake_hien

BASE = os.path.join(ROOT, "external", "MultiAgentESC")
needs_base = pytest.mark.skipif(not os.path.exists(os.path.join(BASE, "multiagent.py")),
                                reason="external/MultiAgentESC not checked out")
needs_data = pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "data", "esconv", "ESConv.json")),
                                reason="data/esconv/ESConv.json missing (scripts/setup_data.py)")
R_EN = {"cmi": 0.0, "cmi_last": 0.0, "hi_frac": 0.0, "dominant": "English", "script": "Roman"}
R_LIGHT = {"cmi": 0.22, "cmi_last": 0.2, "hi_frac": 0.22, "dominant": "English", "script": "Roman"}
R_HEAVY = {"cmi": 0.45, "cmi_last": 0.4, "hi_frac": 0.55, "dominant": "Hindi", "script": "Roman"}
R_DEV = {"cmi": 0.1, "cmi_last": 0.1, "hi_frac": 0.9, "dominant": "Hindi", "script": "Devanagari"}


# ------------------------------------------------------------------ fidelity to the base code
@needs_base
def test_base_prompts_verbatim():
    import importlib.util
    spec = importlib.util.spec_from_file_location("base_prompt", os.path.join(BASE, "prompt.py"))
    bp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bp)
    assert P.BEHAVIOR_CONTROL == bp.prompts["behavior_control"]
    assert P.ZERO_SHOT == bp.prompts["zero_shot"]
    assert P.GET_EMOTION == bp.prompts["get_emotion"]
    assert P.GET_CAUSE == bp.prompts["get_cause"]
    assert P.GET_INTENTION == bp.prompts["get_intention"]

    def template(node, rename):
        return "".join(v.value if isinstance(v, ast.Constant) else
                       "{" + rename.get(ast.unparse(v.value), ast.unparse(v.value)) + "}" for v in node.values)
    found = {}
    for fn in ast.parse(open(os.path.join(BASE, "multiagent.py"), encoding="utf-8").read()).body:
        if isinstance(fn, ast.FunctionDef):
            for node in ast.walk(fn):
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.JoinedStr) \
                        and getattr(node.targets[0], "id", "") == "prompt":
                    found[fn.name] = template(node.value, {})
                if isinstance(node, ast.keyword) and node.arg == "system_message" and isinstance(node.value, ast.JoinedStr):
                    found[fn.name + ".persona"] = template(node.value, {"responses[i]": "response"})
    assert P.SELECT_STRATEGY == found["select_strategy_by_group"]
    assert P.RESPONSE_WITH_STRATEGY == found["response_with_strategy"]
    assert P.DEBATE == found["debate"] and P.DEBATE_PERSONA == found["debate.persona"]
    assert P.REFLECT == found["reflect"] and P.REFLECT_PERSONA == found["reflect.persona"]
    assert P.JUDGE == found["judge"] and P.SELF_REFLECTION == found["self_reflection"]


def _base_loop(sample):
    """Direct port of the sample loop of external/MultiAgentESC/main.py (LLM calls removed)."""
    def json2natural(history):
        out = ""
        for u in history:
            role = u["role"].capitalize()
            out += f"{role}: {u['content'].strip()} "
        return out.strip().replace("Assistant:", "Assistant:").replace("User:", "User:")
    dialog, count, history, ret = sample["dialog"], 0, [], []
    while True:
        save = {}
        if count == len(dialog):
            break
        if count != 0 and dialog[count]["speaker"] == "supporter":
            if (count < len(dialog) - 1 and dialog[count + 1]["speaker"] != "supporter") or (count == len(dialog) - 1):
                save["strategy"] = dialog[count]["annotation"]["strategy"]
                save["reference"] = dialog[count]["content"].strip()
                save["context"] = json2natural(history)
                save["post"] = history[-1]["content"]
                history.append({"content": dialog[count]["content"].strip(),
                                "role": "user" if dialog[count]["speaker"] == "seeker" else "assistant"})
                count += 1
            elif count < len(dialog) - 1 and dialog[count + 1]["speaker"] == "supporter":
                save["strategy"] = f"{dialog[count]['annotation']['strategy']} and {dialog[count + 1]['annotation']['strategy']}"
                save["reference"] = dialog[count]["content"].strip() + " " + dialog[count + 1]["content"].strip()
                save["context"] = json2natural(history)
                save["post"] = history[-1]["content"]
                for c in (count, count + 1):
                    history.append({"content": dialog[c]["content"].strip(),
                                    "role": "user" if dialog[c]["speaker"] == "seeker" else "assistant"})
                count += 2
            save["early"] = count <= 5
            ret.append(save)
        else:
            history.append({"content": dialog[count]["content"].strip(),
                            "role": "user" if dialog[count]["speaker"] == "seeker" else "assistant"})
            count += 1
    return ret


@needs_data
def test_turn_samples_match_base_main_loop():
    data = load_esconv()
    n = 0
    for ci, conv in enumerate(data[:100]):
        ours = turn_samples(conv, ci)
        base = _base_loop(conv)
        assert len(ours) == len(base)
        for o, b in zip(ours, base):
            assert (o["strategy"], o["reference"], o["context"], o["post"], o["early"]) == \
                   (b["strategy"], b["reference"], b["context"], b["post"], b["early"])
            n += 1
    assert n == 1210
    early = {s["uid"] for s in all_samples("en") if s["early"]}
    assert len(early) == 163 and len(set(sampled_uids(200)) & early) == 29


class Recorder:
    """LLM stand-in that records every call and answers from a script."""

    def __init__(self, answers=None, default="Response: ok"):
        self.calls, self.answers, self.default = [], list(answers or []), default

    def chat(self, messages, system=None, temperature=0.0, max_tokens=400, tag="", json_mode=False):
        self.calls.append({"messages": messages, "system": system, "tag": tag, "max_tokens": max_tokens})
        return (self.answers.pop(0) if self.answers else self.default), {"cached": False, "latency": 0.5, "calls": 1}


def test_group_chat_replicates_autogen_round_robin():
    rec = Recorder(answers=["Strategy: [Question]\nReasoning: a", "Strategy: [Others]\nReasoning: b", "c"])
    log = A.CallLog(rec)
    replies = A.group_chat(log, "PROMPT", ["sys0", "sys1", "sys2"], tag="deliberate")
    assert replies == ["Strategy: [Question]\nReasoning: a", "Strategy: [Others]\nReasoning: b", "c"]
    for i, call in enumerate(rec.calls):  # agent i sees the prompt and the i earlier replies
        assert call["system"] == f"sys{i}" and call["tag"] == f"deliberate{i}"
        assert [m["content"] for m in call["messages"]] == ["PROMPT"] + [f"agent_{j}: {replies[j]}" for j in range(i)]
        assert all(m["role"] == "user" for m in call["messages"])
    s = log.summary()
    assert s["n_calls"] == 3 and s["calls_by_tag"] == {"deliberate": 3} and s["latency"] == pytest.approx(1.5)


# ------------------------------------------------------------------ prompts
@pytest.mark.parametrize("R", [R_EN, R_LIGHT, R_HEAVY, R_DEV])
def test_register_prompts_format(R):
    kw = dict(context="C", emo_and_reason="E{x}", cau_and_reason="Ca", int_and_reason="I", strategy="Question",
              examples="X", responses_template="RT", discussion_content="D", pred_strategy="Q", response="r")
    for t in (P.analysis_prompt(P.GET_EMOTION, R), P.analysis_prompt(P.GET_CAUSE, R), P.analysis_prompt(P.GET_INTENTION, R),
              P.zero_shot_register(R), P.response_with_strategy_register(R), P.debate_register(R),
              P.reflect_register(R), P.self_reflection_register(R)):
        text = t.format(**kw)
        assert "### Language register" in text and "{" not in text.replace("E{x}", "")
    gen = P.response_with_strategy_register(R).format(**kw)
    if P.is_plain_english(R):
        assert "Reply in plain English" in gen and "Hindi-English mix" not in gen
    else:
        assert "Reply in the same script with a similar Hindi-English mix" in gen
    assert "language and cultural fit" in P.debate_register(R) and "language and cultural fit" in P.reflect_register(R)
    assert "fits the user's language and culture" in P.self_reflection_register(R)
    g = P.gate_prompt("ctx", R, "resp", "Question", "x")
    assert "[Question] resp" in g and "{" not in g
    assert "[strategy]" not in P.gate_prompt("ctx", R, "resp", None, "x")


# ------------------------------------------------------------------ parsers
@pytest.mark.parametrize("text,expected", [
    ("Response: [Question] How are you feeling now?\nReasoning: open", ("Question", "How are you feeling now?")),
    ("**Response:** [Self-disclosure] I once felt the same.\n**Reasoning:** x", ("Self-disclosure", "I once felt the same.")),
    ("Response:\n[Reflection of Feelings] You sound overwhelmed.", ("Reflection of feelings", "You sound overwhelmed.")),
    ("Response: Providing Suggestions - Maybe talk to your manager?", ("Providing Suggestions", "Maybe talk to your manager?")),
    ("Response: [Question] [What happened next?]", ("Question", "What happened next?")),
    ('Response: [Affirmation & Reassurance] "You are strong." Reasoning: x', ("Affirmation and Reassurance", "You are strong.")),
    ("Response: [Empathy] You seem sad.", (None, "You seem sad.")),
    ("No format here", (None, None)),
])
def test_parse_tagged(text, expected):
    assert A.parse_tagged(text) == expected


def test_parsers_misc():
    assert A.parse_strategies("Strategy: [Question]\nagent_1: **Strategy:** Self-disclosure\nStrategy: question\n"
                              "Strategy: [strategy]\nStrategy: Providing Suggestions and Affirmation") == \
        ["Question", "Self-disclosure", "Providing Suggestions"]
    assert [A.parse_yes_no(t) for t in ("1. YES\n2. ok", "1. NO\n2. the user's eyes, yesterday", "1. [NO]",
                                        "Yes, it does.", "**1. YES**")] == [True, False, False, True, True]
    assert A.parse_single("**Response:** \"Main samajh sakta hoon.\"") == "Main samajh sakta hoon."
    assert A.parse_single("I understand.\nReasoning: x") == "I understand."
    assert A.parse_label("**Emotion:** Anxiety\nReasoning: x", "Emotion", "Negative") == "Anxiety"
    assert A.parse_label("nothing", "Event", "Not mention") == "Not mention"
    assert [A.clean_response(t) for t in ('"I hear you."', "[Question] How are you?", "Assistant: Hi", "None")] == \
        ["I hear you.", "How are you?", "Hi", "None"]
    assert not A.usable(A.parse_tagged("Response: [strategy] [response]")[1])
    assert A.canonical_strategy("otherwise") is None and A.canonical_strategy("Others") == "Others"


def test_vote():
    cands = [("Question", "How are you?"), ("Reflection of feelings", "You seem sad.")]
    assert A.vote(["Response: [Question] How are you?\nReasoning: a",
                   "Response: [Reflection of Feelings] You seem sad.\nReasoning: b",
                   "**Response:** [question] How are you doing?\nReasoning c"], cands) == (["Question"], ["How are you?"])
    assert A.vote(["Response: [Question] How are you?\nReasoning", "Response: [Reflection of feelings] x\nReasoning"],
                  cands) == (["Question", "Reflection of feelings"], ["How are you?", "x"])
    assert A.vote(["Response: [Empathy] You seem sad.\nReasoning"], cands) == (["Reflection of feelings"], ["You seem sad."])
    assert A.vote(["garbage", "Response: [strategy] [response]"], cands) == (["None"], ["None"])


# ------------------------------------------------------------------ Register Gate
def test_register_gate():
    prof = LexiconProfiler()
    english = "I understand how hard this must be for you right now."
    hinglish = "Main samajh sakta hoon yaar, yeh sach mein bahut mushkil hai abhi."
    # within delta: no call
    log = A.CallLog(Recorder())
    out, info = A.register_gate(log, prof, "ctx", R_EN, "Question", english, 0.2)
    assert out == english and not info["triggered"] and info["calls"] == 0 and len(log.calls) == 0
    # English answer to a heavy Hinglish seeker: one call, closer rewrite accepted
    log = A.CallLog(Recorder(answers=[f"Response: [Question] {hinglish}"]))
    out, info = A.register_gate(log, prof, "ctx", R_HEAVY, "Question", english, 0.2)
    assert info["triggered"] and info["accepted"] and out == hinglish and info["calls"] == 1 and len(log.calls) == 1
    assert "too little Hindi" in log.llm.calls[0]["messages"][0]["content"]
    # a rewrite that is not closer is rejected; the original is kept
    log = A.CallLog(Recorder(answers=["Response: [Question] Still plain English here, sorry."]))
    out, info = A.register_gate(log, prof, "ctx", R_HEAVY, "Question", english, 0.2)
    assert info["triggered"] and not info["accepted"] and out == english
    # unparseable answer: original kept
    log = A.CallLog(Recorder(answers=["Response: [strategy] [rewritten response]"]))
    out, info = A.register_gate(log, prof, "ctx", R_HEAVY, "None", english, 0.2)
    assert info["triggered"] and not info["accepted"] and out == english
    # script mismatch triggers even when the CMI gap is small
    log = A.CallLog(Recorder(answers=["Response: [Question] Main samajh sakta hoon."]))
    out, info = A.register_gate(log, prof, "ctx", R_LIGHT, "Question", "मुझे पता है, यह मुश्किल है", 0.9)
    assert info["triggered"] and info["accepted"] and info["script_before"] == "Devanagari"


# ------------------------------------------------------------------ systems end to end
@pytest.fixture(scope="module")
def stand_ins(tmp_path_factory):
    os.environ["CODEMIX_HIEN_DIR"] = write_fake_hien(str(tmp_path_factory.mktemp("hien")), n_test=100)
    yield FakeLLM(), HashRetriever(), LexiconProfiler()
    os.environ.pop("CODEMIX_HIEN_DIR", None)


REQUIRED = {"uid", "conv_id", "turn", "version", "system", "early", "gold_strategy", "reference", "post", "path",
            "complex", "R", "pred_strategy", "response", "pre_gate_response", "gate", "n_calls", "n_real_calls",
            "latency", "calls_by_tag"}


class Spy:
    def __init__(self, inner):
        self.inner, self.prompts = inner, []

    def chat(self, messages, system=None, **kw):
        self.prompts.append((kw.get("tag", ""), messages[0]["content"]))
        return self.inner.chat(messages, system=system, **kw)


@needs_data
@pytest.mark.parametrize("name", list(SYSTEMS))
@pytest.mark.parametrize("version", ["en", "heavy"])
def test_system_end_to_end(stand_ins, name, version):
    if name == "pivot" and version == "en":
        pytest.skip("pivot is defined for Hinglish only")
    llm, retriever, profiler = stand_ins
    keep = set(sampled_uids(200))
    samples = [s for s in all_samples(version) if s["uid"] in keep][:40]
    spy = Spy(llm)
    system = System(name, spy, retriever=retriever, profiler=profiler, delta=0.2)
    recs = [system.respond(s, version) for s in samples]
    for s, r in zip(samples, recs):
        assert REQUIRED <= set(r), REQUIRED - set(r)
        assert r["response"] not in ("", "None") and r["pred_strategy"] in P.STRATEGIES + ["None"]
        assert r["n_calls"] == sum(r["calls_by_tag"].values())
        if r["early"] and SYSTEMS[name]["kind"] in ("multi", "pivot"):
            assert "decide" not in r["calls_by_tag"] and r["path"] == "single"
        if r["path"] in ("single", "fallback") and name != "fewshot_cot":  # few-shot CoT names a strategy
            assert r["pred_strategy"] == "None"
        if name == "pivot":  # no back-translation for a seeker treated as writing English
            assert r["calls_by_tag"]["translate_in"] >= 1
            assert r["calls_by_tag"].get("translate_out", 0) == (0 if is_plain_english(r["R"]) else 1)
        if r["gate"]:
            assert r["calls_by_tag"].get("gate", 0) == r["gate"]["calls"] <= 1
        else:
            assert r["response"] == r["pre_gate_response"]
    register_blind = name in ("zero_shot", "fewshot_cot", "maesc", "pivot")
    assert all(("### Language register" in p) != register_blind for t, p in spy.prompts
               if re.sub(r"\d+$", "", t) in ("emotion", "generate", "debate", "reflect", "refine", "single"))
    if name in ("codemixesc", "cmx_noft", "cmx_noxl") and version == "heavy":
        assert any(r["gate"]["triggered"] for r in recs)


@needs_data
def test_fewshot_examples_never_from_test_or_dev():
    text = fewshot_examples()
    assert text.count("Dialogue context:") == 4
    data = load_esconv()
    dev = set(json.load(open(os.path.join(ROOT, "data", "esconv_hien", "dev_conv_ids.json"))))
    for line in re.findall(r"Response: (.*)", text):
        hits = [i for i, c in enumerate(data) if any(t["content"].strip() == line.strip() for t in c["dialog"])]
        assert hits and all(i >= 100 and i not in dev for i in hits)


def test_gate_never_pushes_hindi_onto_english_seekers():
    prof = LexiconProfiler()
    R = prof.profile(["Hi, I lost my job and I feel terrible."])
    assert P.is_plain_english(R)
    hinglish = "Main samajh sakta hoon yaar, yeh sach mein bahut mushkil hai."
    log = A.CallLog(Recorder(answers=["Response: [Question] I understand, that sounds really hard."]))
    out, info = A.register_gate(log, prof, "ctx", R, "Question", hinglish, 0.2)
    assert info["cmi_target"] == 0.0 and info["triggered"] and info["accepted"]
    assert out == "I understand, that sounds really hard."
    assert "Reply in plain English" in log.llm.calls[0]["messages"][0]["content"]
