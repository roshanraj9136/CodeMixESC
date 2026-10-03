"""ESConv-HiEn builder: band checks, meaning-drift threshold and the repair loop, with a scripted
LLM, the lexicon profiler and a hashing stand-in for LaBSE."""
import json
import os
import re
import sys
import threading

import pytest

from codemixesc.testing import HashEncoder, LexiconProfiler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import build_hien as B  # noqa: E402


def st(cmi, hf, n=10):
    return {"cmi": cmi, "hi_frac": hf, "n_lang": n}


def test_bands_and_diagnosis():
    assert B.band_distance(st(0.1 - 2e-17, 0.1 - 2e-17), "light") == 0.0  # float noise at the edge
    assert B.band_distance(st(0.2, 0.2, n=3), "light") == 0.0  # too short to check
    assert B.band_distance(st(0.23, 0.77), "light") > 0  # CMI in band but Hindi-dominant: not Light
    assert B.band_distance(st(0.45, 0.55), "heavy") == 0.0 and B.band_distance(st(0.2, 0.2), "heavy") > 0
    assert "main language" in B.diagnose(st(0.23, 0.77), "light")[0]
    assert "too little" in B.diagnose(st(0.0, 0.0), "light")[0]
    assert B.diagnose(st(0.2, 0.2), "light", sim=0.3, sim_threshold=0.5)[-1].startswith("the meaning drifted")
    assert B.drift_threshold([0.8, 0.82, 0.79, 0.4]) == pytest.approx(0.55)
    assert B.drift_threshold([0.5, 0.52, 0.48]) == pytest.approx(0.35)  # self-calibrates to low LaBSE scores


class ScriptedLLM:
    """Rewrites with turn 1 left in English; the first fix round answers with the same text
    (rejected), the second with a Light Hinglish version."""

    def __init__(self):
        self.prompts = []

    def __call__(self, prompt, **kw):
        self.prompts.append(prompt)
        if prompt.startswith("Rewrite this emotional support conversation"):
            items = json.loads(re.search(r"Conversation:\n(\[.*\])\n\nReturn", prompt, re.S).group(1))
            turns = [{"id": it["id"], "text": it["text"] if it["id"] == 1 else "yaar " + it["text"] + " bahut"}
                     for it in items]
            return json.dumps({"turns": turns})
        ids = [int(x) for x in re.findall(r"^id (\d+) \(", prompt, re.M)]
        if "(Revision round" not in prompt:
            current = dict(re.findall(r"^id (\d+) .*?\nCurrent: (.*)$", prompt, re.M | re.S))
            return json.dumps({"turns": [{"id": i, "text": current.get(str(i), "")} for i in ids]})
        return json.dumps({"turns": [{"id": i, "text": "That sounds exhausting yaar, what happened at work recently?"}
                                     for i in ids]})


@pytest.fixture
def builder():
    b = B.Builder.__new__(B.Builder)
    b.llm, b.prof, b.labse, b.lock = ScriptedLLM(), LexiconProfiler(), HashEncoder(), threading.Lock()
    return b


def test_repair_loop(builder):
    conv = {"dialog": [{"speaker": "seeker", "content": "I feel so tired of everything at work lately", "annotation": {}},
                       {"speaker": "supporter", "content": "That sounds exhausting, what happened at work recently?",
                        "annotation": {"strategy": "Question"}},
                       {"speaker": "seeker", "content": "My manager keeps giving me more work every single day",
                        "annotation": {}}]}
    out = builder.build_conv(conv, "light", max_rounds=3)
    fix_prompts = [p for p in builder.llm.prompts if p.startswith("You are revising")]
    assert len(fix_prompts) >= 2 and "(Revision round" not in fix_prompts[0] and "(Revision round 2.)" in fix_prompts[1]
    t1 = out["dialog"][1]
    assert t1["content"] == "That sounds exhausting yaar, what happened at work recently?" and t1["regenerated"] == 1  # identical answer not counted
    assert t1["content_en"].startswith("That sounds exhausting") and t1["script"] == "Roman"
    assert all(k in t1 for k in ("cmi", "hi_frac", "labse_sim", "in_band", "meaning_ok", "band_checked"))
    assert out["mix_level"] == "light" and "drift_threshold" in out


def test_meaning_losing_fix_is_rejected(builder):
    builder.llm = ScriptedLLM()
    conv = {"dialog": [{"speaker": "seeker", "content": "I feel so tired of everything at work lately", "annotation": {}},
                       {"speaker": "supporter", "content": "That sounds exhausting, what happened at work recently?",
                        "annotation": {"strategy": "Question"}}]}
    real_call = builder.llm.__call__

    def unrelated_fix(prompt, **kw):
        out = real_call(prompt, **kw)
        return out.replace("That sounds exhausting yaar, what happened at work recently?",
                           "Mujhe nahi pata yaar, kal milte hain bas")
    builder.llm = type("L", (), {"__call__": staticmethod(unrelated_fix), "prompts": builder.llm.prompts})()
    out = builder.build_conv(conv, "light", max_rounds=2)
    assert out["dialog"][1]["content"] == "That sounds exhausting, what happened at work recently?"
    assert out["dialog"][1]["regenerated"] == 0 and not out["dialog"][1]["in_band"]
