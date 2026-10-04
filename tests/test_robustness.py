"""Regression tests for the failure modes found in the pipeline review: embedding caches per
bank, resuming runs (failed calls, cut-off lines, changed configuration, smoke runs), the
Register Gate on failed answers and English homographs, template echoes, incomplete
translations, and the exactness of delta tuning."""
import json
import os
import random
import subprocess
import sys
import threading

import numpy as np
import pytest

from codemixesc import agents as A
from codemixesc import profiler as P
from codemixesc import retriever as RT
from codemixesc.esconv import ROOT
from codemixesc.testing import FakeLLM, HashEncoder, HashRetriever, LexiconProfiler, write_fake_hien

needs_data = pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "data", "esconv", "ESConv.json")),
                                reason="data/esconv/ESConv.json missing (scripts/setup_data.py)")
PY = sys.executable


def _retriever(name, bank, emb_dir, monkeypatch):
    monkeypatch.setattr(RT, "EMB_DIR", str(emb_dir))
    r = RT.Retriever.__new__(RT.Retriever)
    r.name, r.model, r.lock, r.bank = name, HashEncoder(), threading.Lock(), bank
    r.emb = r._bank_embeddings()
    return r


def test_bank_embeddings_cached_per_bank(tmp_path, monkeypatch):
    full = [{"post": f"post number {i}", "response": "r", "strategy": "Question", "conv": i} for i in range(6)]
    dev = [b for b in full if b["conv"] not in (2, 3)]
    a = _retriever("enc", full, tmp_path, monkeypatch)
    b = _retriever("enc", dev, tmp_path, monkeypatch)  # e.g. the dev bank without the dev conversations
    assert a.emb.shape[0] == 6 and b.emb.shape[0] == 4 and len(os.listdir(tmp_path)) == 2
    assert np.allclose(b.emb, a.emb[[0, 1, 4, 5]])
    for f in os.listdir(tmp_path):  # stale files of the wrong size are never used
        np.save(os.path.join(tmp_path, f), np.zeros((99, 256), dtype=np.float32))
    assert np.allclose(_retriever("enc", full, tmp_path, monkeypatch).emb, a.emb)
    assert np.allclose(_retriever("enc", dev, tmp_path, monkeypatch).emb, b.emb)


def test_gate_leaves_failed_answers_alone():
    R = {"cmi": 0.4, "cmi_last": 0.4, "hi_frac": 0.6, "dominant": "Hindi", "script": "Roman"}
    log = A.CallLog(FakeLLM())
    out, info = A.register_gate(log, LexiconProfiler(), "ctx", R, "Question", "None", 0.2)
    assert out == "None" and not info["triggered"] and len(log.calls) == 0


class _Homographs(LexiconProfiler):
    def _label_chunks(self, chunks):
        return [["HI" if w.lower() in {"to", "do", "me", "hi", "yaar", "bahut", "hai", "kya"} else "EN" for w in ws]
                for ws in chunks]


@pytest.mark.parametrize("metric", ["hi_frac", "cmi"])
def test_gate_ignores_homographs_for_english_seekers(metric):
    prof = _Homographs()
    R = prof.profile(["Hi, I lost my job and I want to talk to someone."])
    assert P.is_plain_english(R)
    reply = "I'm sorry to hear that. Do you want to talk to me about it?"
    assert prof.register_of(reply)["hi_frac"] > 0.3  # raw share counts the homographs ...
    log = A.CallLog(FakeLLM())
    out, info = A.register_gate(log, prof, "ctx", R, "Question", reply, 0.2, metric)
    assert info["distance_before"] == 0.0 and not info["triggered"] and out == reply  # ... the gate does not


def test_template_and_instruction_echoes_are_not_responses():
    for t in ("original response", "[Original/refined response]", "The original response is appropriate.",
              "refined version", "Origianl response", "[strategy] [response]", "None", ""):
        assert not A.usable(A.clean_response(t) if t else t), t
    assert A.usable("Original thoughts can help you.")
    assert A.clean_response('["How are you?"]') == A.clean_response(A.clean_response('["How are you?"]')) == "How are you?"
    assert A.clean_response("Kya hua yaar? (Translation: What happened?)") == "Kya hua yaar?"


def test_translate_context_incomplete_keeps_originals():
    class Half:
        def chat(self, messages, system=None, **kw):
            return '{"turns": [{"id": 0, "text": "hello"}]}', {"cached": False, "latency": 0.1, "calls": 1}
    msgs = [{"role": "user", "content": "namaste"}, {"role": "assistant", "content": "kaise ho"}]
    log = A.CallLog(Half())
    out, complete = A.translate_context(log, msgs)
    assert not complete and [m["content"] for m in out] == ["hello", "kaise ho"] and len(log.calls) == 3


@needs_data
def test_delta_tuning_scores_every_delta_exactly(tmp_path, monkeypatch):
    """score(rows, d) from one gate call per turn equals running the real gate at each d."""
    monkeypatch.setenv("CODEMIX_HIEN_DIR", write_fake_hien(str(tmp_path / "hien"), n_test=1))
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import tune_delta as T
    from codemixesc.esconv import dev_ids, dev_samples
    from codemixesc.systems import System
    llm, ret, prof = FakeLLM(), HashRetriever(exclude_conv=set(dev_ids())), LexiconProfiler()
    lo = min(T.GRID)
    samples = sorted(random.Random(0).sample(dev_samples("heavy"), 25), key=lambda s: (s["conv_id"], s["turn"]))
    nogate = System("codemixesc", llm, retriever=ret, profiler=prof, gate=False)
    rows, recs = [], []
    for s in samples:
        rec = nogate.respond(s, "heavy")
        _, info = A.register_gate(A.CallLog(llm), prof, s["context"], rec["R"], rec["pred_strategy"],
                                  rec["pre_gate_response"], lo - 1e-9)
        cand = info.get("candidate") if info["accepted"] else None
        rows.append({"pre": rec["pre_gate_response"], "gated": cand, "dist_pre": info["distance_before"],
                     "bad_script": bool(A.script_mismatch(rec["R"], {"script": info["script_before"]})),
                     "dist_gated": info["distance_after"] if cand else None, "cmi_pre": 0, "cmi_gated": 0})
        recs.append((s, rec))
    for d in T.GRID:
        finals, _, _, calls = T.score(rows, d)
        real = [A.register_gate(A.CallLog(llm), prof, s["context"], rec["R"], rec["pred_strategy"],
                                rec["pre_gate_response"], d) for s, rec in recs]
        assert [out for out, _ in real] == finals and sum(i["calls"] for _, i in real) == calls


def _run(args, env_extra=None, out_dir=None):
    env = dict(os.environ, **(env_extra or {}))
    return subprocess.run([PY, os.path.join(ROOT, "scripts", "run_system.py"), *args, "--dry_run",
                           "--out_dir", str(out_dir)], cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)


@needs_data
def test_run_system_resume_retries_failures_and_survives_cut_lines(tmp_path):
    out = tmp_path / "runs"
    common = ["--system", "maesc", "--version", "en", "--subset", "all", "--limit", "8", "--workers", "2"]
    r1 = _run(common, {"CODEMIX_FAKE_FAIL": "single,decide"}, out)
    assert r1.returncode == 2, r1.stdout + r1.stderr  # every turn has a failed call
    recs = [json.loads(x) for x in open(out / "maesc" / "en.jsonl")]
    assert len(recs) == 8 and all(r["n_failed_calls"] for r in recs)
    with open(out / "maesc" / "en.jsonl", "a") as f:
        f.write('{"uid": "0-99", "cut off')  # a crash in the middle of a write
    r2 = _run(common, {"CODEMIX_FAKE_FAIL": ""}, out)
    assert r2.returncode == 0, r2.stdout + r2.stderr
    recs = [json.loads(x) for x in open(out / "maesc" / "en.jsonl")]
    assert len(recs) == 8 and not any(r["n_failed_calls"] for r in recs) and "8 with failed calls, retried" in r2.stdout
    meta = json.load(open(out / "maesc" / "en.meta.json"))
    assert meta["limit"] == 8 and meta["n_with_failed_calls"] == 0
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import run_all
    assert not run_all.finished(str(out), "maesc", "en")  # a --limit smoke run is never "finished"
    cmx = ["--system", "codemixesc", "--version", "en", "--limit", "4", "--workers", "2"]
    assert _run(cmx, None, out).returncode == 0
    r3 = _run(cmx + ["--delta", "0.33", "--name", "codemixesc"], None, out)  # same folder, other delta
    assert r3.returncode != 0 and "different configuration" in (r3.stdout + r3.stderr)


@needs_data
def test_fewshot_without_response_line_has_no_answer():
    from codemixesc.esconv import all_samples
    from codemixesc.systems import System

    class Rambling:
        def chat(self, messages, system=None, **kw):
            return "Let's think step by step. The user is sad because", {"cached": False, "latency": 0.1, "calls": 1}
    rec = System("fewshot_cot", Rambling()).respond(all_samples("en")[10], "en")
    assert rec["response"] == "None"
