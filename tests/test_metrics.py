"""Evaluation suite: every metric of codemixesc/metrics.py against hand-computed values (and
against the reference implementations where they apply), then scripts/evaluate.py and
scripts/judge.py end to end on synthetic run files. No model downloads, no API calls."""
import csv
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import Counter

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from codemixesc import metrics as M  # noqa: E402
from codemixesc.testing import LexiconProfiler  # noqa: E402


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"test_script_{name}", os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ================================================================== tokenizer and helpers
def test_tokenizer():
    assert M.tokenize("I'm FINE, yaar!!") == ["i", "'", "m", "fine", ",", "yaar", "!", "!"]
    assert M.tokenize("हिंदी में बात") == ["हिंदी", "में", "बात"]  # vowel signs stay inside the word
    assert M.tokenize("ok 😊") == ["ok", "😊"]
    assert M.tokenize("cafe\u0301") == M.tokenize("café") == ["café"]  # NFC
    assert M.tokenize(None) == []
    assert M.word_tokens("I'm fine, yaar! 😊") == ["i", "m", "fine", "yaar"]
    assert M.word_count("I'm fine - really 😊 ok") == 4  # a lone '-' or emoji is not a word


def test_missing_responses():
    assert M.is_missing(None) and M.is_missing("") and M.is_missing("  None ") and M.is_missing("   ")
    assert not M.is_missing("None of this helps")
    assert M.hypothesis("None") == "" and M.hypothesis(" ok ") == "ok"


# ================================================================== response quality
def test_distinct():
    toks = [["a", "b", "a"], ["a", "b", "c"]]
    assert M.distinct_n(toks, 1) == pytest.approx(100 * 3 / 6)
    assert M.distinct_n(toks, 2) == pytest.approx(100 * 3 / 4)  # ab ba | ab bc: no n-gram across responses
    assert M.distinct_n([["a"]], 2) == 0.0 and M.distinct_n([], 1) == 0.0


def test_bleu_hand_computed():
    hyp, ref = [M.tokenize("the cat sat on the mat")], [M.tokenize("the cat is on the mat")]
    # p1 = 5/6, p2 = 3/5 (the cat, on the, the mat), p3 = 1/4 (on the mat), brevity penalty 1
    assert M.corpus_bleu(hyp, ref, 1) == pytest.approx(100 * 5 / 6)
    assert M.corpus_bleu(hyp, ref, 2) == pytest.approx(100 * math.sqrt(5 / 6 * 3 / 5))
    assert M.corpus_bleu(hyp, ref, 3) == pytest.approx(50.0)  # (5/6 * 3/5 * 1/4) ** (1/3)
    # corpus statistics are pooled over sentences (not averaged): p1 = (5 + 2) / (6 + 2)
    hyp2, ref2 = hyp + [["a", "b"]], ref + [["a", "b", "c"]]
    bp = math.exp(1 - 9 / 8)
    assert M.corpus_bleu(hyp2, ref2, 1) == pytest.approx(100 * bp * 7 / 8)


def test_bleu_smoothing_and_edge_cases():
    # no 3-gram match: method 3 gives the empty order 1 / (2 * #3-grams) = 1/4 instead of 0
    hyp, ref = [["a", "b", "c", "d"]], [["a", "b", "x", "d"]]
    assert M.corpus_bleu(hyp, ref, 3) == pytest.approx(100 * (3 / 4 * 1 / 3 * 1 / 4) ** (1 / 3))
    assert M.corpus_bleu(hyp, ref, 2) == pytest.approx(100 * math.sqrt(3 / 4 * 1 / 3))  # unchanged when all match
    # brevity penalty exp(1 - r/c)
    short = M.corpus_bleu([["the", "cat"]], [M.tokenize("the cat sat on the mat")], 1)
    assert short == pytest.approx(100 * math.exp(1 - 6 / 2))
    assert M.corpus_bleu([["x", "y"]], [["a", "b"]], 1) == 0.0  # no unigram match at all
    assert M.corpus_bleu([], [], 2) == 0.0
    assert M.corpus_bleu([[]], [["a", "b"]], 1) == 0.0  # empty hypothesis


def parlai_f1(guess, answer):
    """ParlAI's F1Metric (normalize_answer + _prec_recall_f1_score), for comparison."""
    re_art = re.compile(r"\b(a|an|the)\b")
    re_punc = re.compile(r"[!\"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']")

    def norm(s):
        return re_art.sub(" ", re_punc.sub(" ", s.lower())).split()

    g, a = norm(guess), norm(answer)
    same = sum((Counter(g) & Counter(a)).values())
    if same == 0:
        return 0.0
    p, r = same / len(g), same / len(a)
    return 2 * p * r / (p + r)


def test_unigram_f1():
    assert M.unigram_f1("The cat sat.", "a cat sat down") == pytest.approx(80.0)  # P = 1, R = 2/3
    assert M.unigram_f1("yes yes yes", "yes no") == pytest.approx(40.0)  # clipped: P = 1/3, R = 1/2
    assert M.unigram_f1("hello", "bye") == 0.0 and M.unigram_f1("", "bye") == 0.0
    assert M.unigram_f1("the a an", "the") == 0.0  # only articles
    for h, r in [("I hear you, that sounds really hard.", "That sounds hard; I hear you!"),
                 ("Don't worry, it's going to be OK.", "do not worry - it will be ok"),
                 ("Have you talked to an expert about it?", "Did you talk to the experts? It helps.")]:
        assert M.unigram_f1(h, r) == pytest.approx(100 * parlai_f1(h, r))
    assert M.unigram_f1("मुझे पता है", "मुझे नहीं पता") == pytest.approx(100 * 2 / 3)  # Devanagari kept


def test_lcs_and_rouge_l():
    assert M.lcs_length(list("ABCBDAB"), list("BDCABA")) == 4  # CLRS example
    assert M.lcs_length([], ["a"]) == 0
    assert M.rouge_l("a b c d", "a c d e") == pytest.approx(75.0)  # LCS 3: P = R = 3/4
    assert M.rouge_l("", "a") == 0.0 and M.rouge_l("x", "y") == 0.0
    # Devanagari is scored (rouge_score drops every non-[a-z0-9] character and would give 0)
    assert M.rouge_l("मैं ठीक हूँ", "मैं ठीक नहीं हूँ") == pytest.approx(100 * 2 * 0.75 / 1.75)


def test_rouge_l_equals_rouge_score_on_ascii():
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    pairs = [("I hear you, that sounds really hard.", "That sounds hard; I hear you!"),
             ("Don't worry! It's 5 o'clock, take a break.", "do not worry, it is five. Take a short break"),
             ("Have you talked to your parents about the exam results?",
              "Maybe you could talk with your parents about your results."),
             ("It makes sense that you feel worried about this.", "You feel worried, and that makes sense."),
             ("Yaar, tension mat lo. Sab theek ho jayega!", "tension mat lo yaar, sab theek ho jayega")]
    for h, r in pairs:
        assert M.rouge_l(h, r) == pytest.approx(100 * scorer.score(r, h)["rougeL"].fmeasure)
    assert scorer.score("मैं ठीक नहीं हूँ", "मैं ठीक हूँ")["rougeL"].fmeasure == 0.0


def test_chrf():
    assert M.chrf_sentence("tension mat lo", "tension mat lo") == pytest.approx(100.0)
    assert M.chrf_sentence("", "tension mat lo") == 0.0
    s = M.chrf_sentence("mujhe nahi pata", "mujhe nahin pata")  # spelling variant gets partial credit
    assert 50 < s < 100
    assert M.chrf_corpus(["mujhe nahi pata"], ["mujhe nahin pata"]) == pytest.approx(s)  # one sentence: equal
    assert M.chrf_corpus(["abc", "xyz"], ["abc", "pqr"]) < 100
    assert "nc:6" in M.chrf_signature() and "nw:0" in M.chrf_signature()


def test_bertscore_if_model_is_cached():
    """Runs only where bert-base-multilingual-cased is already in the Hugging Face cache."""
    pytest.importorskip("bert_score")
    from huggingface_hub import try_to_load_from_cache
    if not isinstance(try_to_load_from_cache(M.BERTSCORE_MODEL, "config.json"), str):
        pytest.skip("multilingual BERT is not cached (no download in tests)")
    bs = M.BERTScore(device="cpu")
    same, other, empty = bs(["main theek hoon", "main theek hoon", ""], ["main theek hoon", "it is raining", "x"])
    assert same == pytest.approx(100.0, abs=0.01) and other < same and empty == 0.0


def test_bertscore_path_with_stub_model(monkeypatch, synthetic, tmp_path):
    """The BERTScore wrapper and its use in evaluate.py, with bert_score's scorer stubbed out
    (the multilingual model cannot be downloaded in the test environment)."""
    bert_score = pytest.importorskip("bert_score")
    import torch

    class StubScorer:
        def __init__(self, model_type=None, device=None, batch_size=64, **kw):
            self.hash = f"{model_type}_stub"

        def score(self, cands, refs, batch_size=64):
            assert all(c.strip() for c in cands)  # empty hypotheses never reach the model
            f = torch.tensor([1.0 if c == r else 0.5 for c, r in zip(cands, refs)])
            return f, f, f

    monkeypatch.setattr(bert_score, "BERTScorer", StubScorer)
    assert M.BERTScore(device="cpu")(["a", "", "b"], ["a", "x", "c"]) == [100.0, 0.0, 50.0]
    ev = load_script("evaluate")
    assert ev.main(["--runs_dir", str(synthetic / "runs"), "--out_dir", str(tmp_path), "--profiler", "none",
                    "--samples_file", str(synthetic / "samples.json"), "--subset_file", str(synthetic / "subset.json"),
                    "--versions", "light", "--no_figures", "--n_boot", "200"]) == 0
    m = json.loads((tmp_path / "eval" / "metrics.json").read_text(encoding="utf-8"))
    assert m["meta"]["bertscore"]["hash"] == "bert-base-multilingual-cased_stub"
    r = m["results"]["light"]["codemixesc"]["common"]
    assert r["bertscore_f1"] == pytest.approx(100 * 11 * 0.5 / 12)  # 11 answered turns at 0.5, the failed one 0
    assert m["significance"]["light"]["maesc"]["bertscore"]["n"] == 12
    assert "BERTScore" in (tmp_path / "tables" / "main_light.md").read_text(encoding="utf-8")


# ================================================================== register match
def test_register_match_with_lexicon_profiler():
    prof = LexiconProfiler()
    seeker = ["mujhe bahut tension hai"]  # HI HI EN HI: CMI_s = 1 - 3/4, hi_frac 3/4, Hindi-dominant
    r = M.register_match(prof, seeker, "I understand your tension")  # all EN
    assert r["cmi_s"] == pytest.approx(0.25) and r["cmi_r"] == 0.0 and r["cmi_gap"] == pytest.approx(0.25)
    assert r["hi_frac_gap"] == pytest.approx(0.75) and r["script_s"] == "Roman" and r["script_match"] is True
    # same CMI, opposite dominant language: only the Hindi fraction notices the flip
    r = M.register_match(prof, seeker, "tension mat lo yaar")  # EN EN EN HI: CMI 0.25, hi_frac 0.25
    assert r["cmi_gap"] == pytest.approx(0.0) and r["hi_frac_gap"] == pytest.approx(0.5)
    assert r["dominant_s"] == "Hindi" and r["dominant_r"] == "English"
    # pooled over all seeker utterances, as the system's profile R
    r = M.register_match(prof, ["mujhe bahut", "tension hai"], "ok")
    assert r["cmi_s"] == pytest.approx(0.25)
    assert M.register_match(prof, seeker, "मुझे पता है")["script_match"] is False  # Devanagari reply
    assert M.register_match(prof, seeker, "🙂🙂")["script_match"] is False  # a reply without letters
    assert M.register_match(prof, ["..."], "ok")["script_match"] is None  # no seeker script yet


def test_register_summary():
    prof = LexiconProfiler()
    rows = [M.register_match(prof, ["mujhe bahut tension hai"], "I understand your tension"),
            M.register_match(prof, ["mujhe bahut tension hai"], "mujhe bahut tension hai"),
            M.register_match(prof, ["..."], "ok"), None]  # None = a turn without a response
    s = M.register_summary(rows)
    assert s["n_register"] == 3 and s["n_script"] == 2
    assert s["cmi_gap"] == pytest.approx((0.25 + 0.0 + 0.0) / 3)
    assert s["script_consistency"] == pytest.approx(100.0)
    assert s["dominant_match"] == pytest.approx(100 * 2 / 3)


# ================================================================== strategies
def test_gold_and_predicted_strategies():
    assert M.gold_strategies("Affirmation and Reassurance and Question") == {"Affirmation and Reassurance", "Question"}
    assert M.gold_strategies("Providing Suggestions and Affirmation and Reassurance") == \
        {"Providing Suggestions", "Affirmation and Reassurance"}
    assert M.gold_strategies("Restatement or Paraphrasing") == {"Restatement or Paraphrasing"}
    assert M.normalize_strategy("question") == "Question" and M.normalize_strategy(" Others ") == "Others"
    assert M.normalize_strategy("None") == M.normalize_strategy(None) == M.normalize_strategy("Advice") == "None"
    match, match_pred = M.strategy_match(["Question", "None", "Others"],
                                         [{"Question"}, {"Question"}, {"Information"}])
    assert match == pytest.approx(100 / 3) and match_pred == pytest.approx(50.0)


def test_jsd():
    assert M.jsd([1, 2, 3], [2, 4, 6]) == pytest.approx(0.0)  # counts are normalised
    assert M.jsd([1, 0], [0, 1]) == pytest.approx(1.0)  # disjoint supports: 1 bit
    # P = (1, 0), Q = (1/2, 1/2): M = (3/4, 1/4); (log2(4/3) + 1/2 log2(2/3) + 1/2) / 2
    expected = 0.5 * (math.log2(4 / 3) + 0.5 * math.log2(2 / 3) + 0.5)
    assert M.jsd([1, 0], [0.5, 0.5]) == pytest.approx(expected) == pytest.approx(0.3112781)
    from scipy.spatial.distance import jensenshannon
    p, q = [5, 1, 0, 3, 1], [2, 2, 2, 0, 4]
    assert M.jsd(p, q) == pytest.approx(jensenshannon(p, q, base=2) ** 2)  # scipy returns the square root
    assert M.jsd([0, 0], [1, 0]) is None


def test_strategy_stability():
    a = {"1": "Question", "2": "None", "3": "Others", "4": "Question"}
    b = {"1": "Question", "2": "Question", "3": "Information", "5": "Others"}
    s = M.strategy_stability(a, b)  # shared turns 1, 2, 3
    assert s["n"] == 3 and s["agreement"] == pytest.approx(100 / 3)
    # with None: P = (Q 1/3, Others 1/3, None 1/3), Q = (Q 2/3, Information 1/3), M = (1/2, 1/6, 1/6, 1/6)
    kl_p = (math.log2((1 / 3) / 0.5) + 2 * math.log2(2)) / 3
    kl_q = 2 / 3 * math.log2((2 / 3) / 0.5) + 1 / 3 * math.log2(2)
    assert s["jsd"] == pytest.approx((kl_p + kl_q) / 2)
    # strategies only: turns 1 and 3 -> (Q, Others) vs (Q, Information)
    assert s["n_strategies"] == 2 and s["jsd_strategies"] == pytest.approx(0.5)
    assert s["agreement_strategies"] == pytest.approx(50.0)
    same = M.strategy_stability(a, a)
    assert same["jsd"] == 0.0 and same["agreement"] == 100.0


# ================================================================== statistics
def test_paired_bootstrap():
    rng = np.random.default_rng(0)
    b = rng.normal(size=200)
    a = b + 0.3 + rng.normal(scale=0.5, size=200)
    r1, r2 = M.paired_bootstrap(a, b), M.paired_bootstrap(a, b)
    assert r1 == r2  # seeded: reproducible
    assert r1["diff"] == pytest.approx(float(np.mean(a - b)))
    assert r1["ci_low"] < r1["diff"] < r1["ci_high"] and r1["ci_low"] > 0 and r1["p"] < 0.01
    flipped = M.paired_bootstrap(b, a)  # same resamples, mirrored differences
    assert flipped["diff"] == pytest.approx(-r1["diff"]) and flipped["p"] == pytest.approx(r1["p"])
    assert M.paired_bootstrap(a, b, seed=7)["ci_low"] != r1["ci_low"]
    tie = M.paired_bootstrap([1, 2, 3], [1, 2, 3])
    assert tie["diff"] == 0 and tie["p"] == 1.0 and tie["ci_low"] == tie["ci_high"] == 0
    sure = M.paired_bootstrap([1.0] * 50, [0.0] * 50, n_boot=999)
    assert sure["p"] == pytest.approx(2 / 1000) and sure["ci_low"] == sure["ci_high"] == 1.0
    noise = M.paired_bootstrap(rng.normal(size=100), rng.normal(size=100))
    assert (noise["p"] < 0.05) == (noise["ci_low"] > 0 or noise["ci_high"] < 0)  # p and CI agree
    assert M.paired_bootstrap([], []) is None
    with pytest.raises(ValueError):
        M.paired_bootstrap([1, 2], [1])


def test_cohen_kappa_and_sign_test():
    a = ["win", "win", "tie", "lose", "lose", "win"]
    b = ["win", "tie", "tie", "lose", "win", "win"]
    po, pe = 4 / 6, (3 * 3 + 1 * 2 + 2 * 1) / 36
    assert M.cohen_kappa(a, b) == pytest.approx((po - pe) / (1 - pe))
    from sklearn.metrics import cohen_kappa_score
    assert M.cohen_kappa(a, b) == pytest.approx(cohen_kappa_score(a, b))
    assert M.cohen_kappa(a, a) == pytest.approx(1.0)
    assert M.cohen_kappa(["win"] * 3, ["win"] * 3) is None and M.cohen_kappa([], []) is None
    assert M.sign_test(9, 1) == pytest.approx(22 / 1024)  # 2 * (C(10,0) + C(10,1)) / 2^10
    from scipy.stats import binomtest
    assert M.sign_test(30, 18) == pytest.approx(binomtest(30, 48).pvalue)
    assert M.sign_test(5, 5) == 1.0 and M.sign_test(0, 0) == 1.0


# ================================================================== synthetic runs
SEEKER = {"en": ["I am so worried about my exams", "My parents expect a lot from me", "I cannot sleep at night"],
          "light": ["I am so worried yaar, exams ka bahut tension hai", "My parents expect bahut kuch from me",
                    "I cannot sleep at night, pata nahi kyun"],
          "heavy": ["Mujhe bahut tension hai yaar, exams ki", "Mere ghar pe sab bahut expect karte hain",
                    "Raat ko neend nahi aati, pata nahi kya karun"]}
SUPPORT = {"en": ["That sounds really hard. What worries you most?", "I hear you, that is a lot of pressure.",
                  "Have you tried a short walk before bed?"],
           "light": ["That sounds really hard yaar. What worries you most?", "I hear you, bahut pressure hai.",
                     "Have you tried a short walk before bed, yaar?"],
           "heavy": ["Yeh sach mein bahut mushkil hai, aapko kya pareshan kar raha hai?",
                     "Main samajh sakta hoon, bahut pressure hai na", "Kya aapne sone se pehle walk try kiya hai?"]}
EN_RESP = ["I hear you, that sounds really hard.", "What do you think would help you right now?",
           "It makes sense that you feel worried about this exam."]
HI_RESP = ["Main samajh sakta hoon yaar, yeh sach mein bahut mushkil hai.",
           "Aapko abhi kya help karega, kuch socha hai?", "It makes sense ki aap itna pareshan ho, yaar."]
SUBSET = ["0-2", "0-6", "0-8", "0-10", "1-2", "1-4", "1-8", "1-12", "2-4", "2-6", "2-10", "2-12"]


def make_samples():
    out = {}
    for v in ("en", "light", "heavy"):
        out[v] = []
        for c in range(3):
            msgs = []
            for t in range(1, 7):
                msgs.append({"role": "user", "content": SEEKER[v][(c + t) % 3]})
                k = (c + t) % 3
                strategy = M.STRATEGIES[(c + t) % 8] if t % 3 else "Question and Affirmation and Reassurance"
                out[v].append({"uid": f"{c}-{2 * t}", "conv_id": c, "turn": 2 * t, "strategy": strategy,
                               "reference": SUPPORT[v][k], "context_msgs": list(msgs),
                               "context": " ".join(m["content"] for m in msgs), "post": msgs[-1]["content"],
                               "early": t <= 2, "problem_type": "ongoing depression"})
                msgs.append({"role": "assistant", "content": SUPPORT[v][k]})
    return out


def make_record(system, version, sample, i):
    hinglish = version != "en" and system in ("codemixesc", "cmx_noft", "cmx_noxl", "pivot")
    resp = (HI_RESP if hinglish else EN_RESP)[(i + (system == "pivot")) % 3]
    multi = not sample["early"] and system not in ("zero_shot", "fewshot_cot")
    pred = M.STRATEGIES[(i + len(system)) % 8] if (multi or system == "fewshot_cot") else "None"
    rec = {"uid": sample["uid"], "conv_id": sample["conv_id"], "turn": sample["turn"], "version": version,
           "system": system, "early": sample["early"], "gold_strategy": sample["strategy"],
           "reference": sample["reference"], "post": sample["post"], "R": None,
           "path": "multi" if multi else "single", "complex": True if multi else None, "pred_strategy": pred,
           "response": resp, "pre_gate_response": resp, "gate": None,
           "n_calls": 12 if multi else 1, "n_real_calls": 0, "latency": 3.0 if multi else 0.4,
           "calls_by_tag": {"generate": 3} if multi else {"single": 1}}
    if system in ("codemixesc", "cmx_noft", "cmx_noxl"):
        fired = version != "en" and i % 3 == 0
        rec["gate"] = {"triggered": fired, "accepted": fired, "delta": 0.2, "cmi_s": 0.3, "cmi_before": 0.0,
                       "script_before": "Roman", "cmi_after": 0.3, "script_after": "Roman",
                       "calls": 1 if fired else 0, "latency": 0.7 if fired else 0.0}
        if fired:
            rec["pre_gate_response"] = EN_RESP[i % 3]
            rec["n_calls"] += 1
            rec["latency"] += 0.7
            rec["calls_by_tag"] = dict(rec["calls_by_tag"], gate=1)
    return rec


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    """samples.json, subset.json and run files for all systems; with a duplicate line, an
    unreadable line, a failed turn, a 'None' response, a missing turn and an incomplete run."""
    d = tmp_path_factory.mktemp("synthetic")
    samples = make_samples()
    plan = {"zero_shot": ("en", "light", "heavy"), "fewshot_cot": ("en", "light", "heavy"),
            "maesc": ("en", "light", "heavy"), "codemixesc": ("en", "light", "heavy"),
            "pivot": ("light", "heavy"), "cmx_noft": ("light", "heavy"), "cmx_noxl": ("light", "heavy")}
    for system, versions in plan.items():
        os.makedirs(d / "runs" / system)
        for v in versions:
            lines = []
            for i, s in enumerate(samples[v]):
                if system not in ("zero_shot", "fewshot_cot") and s["uid"] not in SUBSET:
                    continue
                if (system, v) == ("maesc", "heavy") and s["uid"] == "2-12":
                    continue  # one missing turn: the common heavy turns shrink to 11
                if (system, v) == ("cmx_noxl", "heavy") and s["uid"] not in SUBSET[:3]:
                    continue  # an incomplete run: below --min_coverage
                rec = make_record(system, v, s, i)
                if (system, v) == ("codemixesc", "light") and s["uid"] == "1-8":
                    rec.update(response="None", pre_gate_response="None", error="RuntimeError('quota')")
                if (system, v) == ("fewshot_cot", "en") and s["uid"] == "0-2":
                    rec["response"] = "None"
                lines.append(json.dumps(rec, ensure_ascii=False))
            if (system, v) == ("zero_shot", "en"):
                lines.append(lines[0])  # a duplicate uid
            if (system, v) == ("maesc", "en"):
                lines.append('{"uid": "0-2", "respo')  # a line cut off by a killed run
            (d / "runs" / system / f"{v}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (d / "samples.json").write_text(json.dumps(samples, ensure_ascii=False), encoding="utf-8")
    (d / "subset.json").write_text(json.dumps(SUBSET), encoding="utf-8")
    return d


def read_runs(d, system, version):
    return {json.loads(line)["uid"]: json.loads(line)
            for line in (d / "runs" / system / f"{version}.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip() and line.endswith("}")}


@pytest.fixture(scope="module")
def evaluated(synthetic):
    ev = load_script("evaluate")
    out = synthetic / "out"
    code = ev.main(["--runs_dir", str(synthetic / "runs"), "--out_dir", str(out), "--profiler", "lexicon",
                    "--no_bertscore", "--samples_file", str(synthetic / "samples.json"),
                    "--subset_file", str(synthetic / "subset.json"), "--n_boot", "300"])
    assert code == 0
    return out, json.loads((out / "eval" / "metrics.json").read_text(encoding="utf-8"))


TABLES = ["main_en", "main_light", "main_heavy", "register", "stability", "ablation", "efficiency", "significance",
          "baselines_all", "runs", "strategy_dist"]


def test_evaluate_outputs(evaluated):
    out, m = evaluated
    for name in TABLES:
        for ext in ("md", "csv", "tex"):
            assert (out / "tables" / f"{name}.{ext}").exists(), f"{name}.{ext}"
    tex = (out / "tables" / "main_light.tex").read_text(encoding="utf-8")
    assert "\\toprule" in tex and "\\bottomrule" in tex and "\\textbf{" in tex and "\\label{tab:main_light}" in tex
    for fig in ("robustness", "strategy_dist_codemixesc", "strategy_dist_pivot", "strategy_dist_fewshot_cot"):
        assert (out / "figures" / f"{fig}.png").stat().st_size > 1000 and (out / "figures" / f"{fig}.pdf").exists()
    assert not (out / "figures" / "strategy_dist_zero_shot.png").exists()  # never predicts a strategy
    lines = (out / "eval" / "per_turn.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == sum(x["all"]["n"] for v in m["results"].values() for x in v.values())  # every scored turn


def test_evaluate_turn_sets(evaluated):
    _, m = evaluated
    assert m["meta"]["common_turns"] == {"en": 12, "light": 12, "heavy": 11}
    assert "cmx_noxl" not in m["meta"]["in_tables"]["heavy"] and "cmx_noxl" in m["meta"]["in_tables"]["light"]
    assert not m["runs"]["cmx_noxl/heavy"]["in_tables"] and m["results"]["heavy"]["cmx_noxl"]["subset"]["n"] == 3
    zs = m["results"]["light"]["zero_shot"]
    assert zs["all"]["n"] == 18 and zs["subset"]["n"] == 12 and zs["common"]["n"] == 12
    assert m["results"]["heavy"]["codemixesc"]["common"]["n"] == 11
    runs = m["runs"]
    assert runs["zero_shot/en"]["duplicates"] == 1 and runs["maesc/en"]["unreadable"] == 1
    assert runs["codemixesc/light"]["failed"] == 1 and runs["fewshot_cot/en"]["none"] == 1
    assert runs["maesc/heavy"]["missing"] == 1 and runs["zero_shot/en"]["missing"] == 0
    assert m["results"]["light"]["codemixesc"]["common"]["n_failed"] == 1


def test_evaluate_scores_match_metrics(synthetic, evaluated):
    out, m = evaluated
    samples = json.loads((synthetic / "samples.json").read_text(encoding="utf-8"))
    recs = read_runs(synthetic, "maesc", "light")
    ref = {s["uid"]: s for s in samples["light"]}
    uids = sorted(recs)
    r = m["results"]["light"]["maesc"]["common"]
    assert r["f1"] == pytest.approx(np.mean([M.unigram_f1(recs[u]["response"], ref[u]["reference"]) for u in uids]),
                                    abs=1e-5)
    hyp = [M.tokenize(recs[u]["response"]) for u in uids]
    refs = [M.tokenize(ref[u]["reference"]) for u in uids]
    assert r["bleu_2"] == pytest.approx(M.corpus_bleu(hyp, refs, 2), abs=1e-5)
    assert r["chrf"] == pytest.approx(M.chrf_corpus([recs[u]["response"] for u in uids],
                                                    [ref[u]["reference"] for u in uids]), abs=1e-5)
    assert r["distinct_1"] == pytest.approx(M.distinct_n(hyp, 1), abs=1e-5)
    assert r["bertscore_f1"] is None  # --no_bertscore
    prof = LexiconProfiler()
    gaps = [M.register_match(prof, [x["content"] for x in ref[u]["context_msgs"] if x["role"] == "user"],
                             recs[u]["response"])["cmi_gap"] for u in uids]
    assert r["cmi_gap"] == pytest.approx(np.mean(gaps), abs=1e-5)
    # a failed turn is scored as an empty response (F1 0) and skipped by the register statistics
    c = m["results"]["light"]["codemixesc"]["common"]
    assert c["n_register"] == 11 and c["n"] == 12


def test_evaluate_derives_nogate(synthetic, evaluated):
    out, m = evaluated
    assert "cmx_nogate" not in m["results"]["en"]  # no EN ablation run: not derived for EN
    cmx = read_runs(synthetic, "codemixesc", "heavy")
    per_turn = [json.loads(x) for x in (out / "eval" / "per_turn.jsonl").read_text(encoding="utf-8").splitlines()]
    nogate = {x["uid"]: x for x in per_turn if x["system"] == "cmx_nogate" and x["version"] == "heavy"}
    samples = {s["uid"]: s for s in json.loads((synthetic / "samples.json").read_text(encoding="utf-8"))["heavy"]}
    assert set(nogate) == set(cmx)
    for uid, rec in cmx.items():
        assert nogate[uid]["f1"] == pytest.approx(M.unigram_f1(rec["pre_gate_response"], samples[uid]["reference"]),
                                                  abs=1e-5)
        assert nogate[uid]["n_calls"] == rec["n_calls"] - rec["gate"]["calls"]
        assert nogate[uid]["latency"] == pytest.approx(rec["latency"] - rec["gate"]["latency"])
    eff_cmx, eff_ng = m["results"]["heavy"]["codemixesc"]["all"], m["results"]["heavy"]["cmx_nogate"]["all"]
    gate_calls = np.mean([r["gate"]["calls"] for r in cmx.values()])
    assert eff_cmx["n_calls"] - eff_ng["n_calls"] == pytest.approx(gate_calls, abs=1e-5)  # metrics.json: 6 decimals
    assert eff_ng["n_real_calls"] is None and eff_cmx["gate_triggered"] == pytest.approx(100 * gate_calls, abs=1e-4)
    # extra calls over MultiAgentESC on shared turns: the gate's calls only (same pipeline otherwise)
    assert eff_cmx["extra_calls_vs_maesc"] == pytest.approx(
        np.mean([cmx[u]["gate"]["calls"] for u in cmx if u != "2-12"]), abs=1e-5)


def test_evaluate_stability_and_significance(synthetic, evaluated):
    _, m = evaluated
    st = m["stability"]["light"]
    assert "zero_shot" not in st and "cmx_nogate" not in st
    assert st["pivot"]["common"]["en_system"] == "maesc" and st["maesc"]["common"]["en_system"] == "maesc"
    en, li = read_runs(synthetic, "maesc", "en"), read_runs(synthetic, "maesc", "light")
    expected = M.strategy_stability({u: en[u]["pred_strategy"] for u in SUBSET},
                                    {u: li[u]["pred_strategy"] for u in SUBSET})
    assert st["maesc"]["common"]["jsd"] == pytest.approx(expected["jsd"], abs=1e-6)
    sig = m["significance"]["light"]
    assert set(sig) == {"zero_shot", "fewshot_cot", "maesc", "pivot", "cmx_nogate", "cmx_noft", "cmx_noxl"}
    assert sig["maesc"]["f1"]["n"] == 12 and sig["maesc"]["cmi_gap"]["n"] == 11  # the failed turn has no register
    # same responses except codemixesc's failed turn 1-8, which scores F1 0 as an empty response
    noft = read_runs(synthetic, "cmx_noft", "light")["1-8"]
    ref = {s["uid"]: s for s in json.loads((synthetic / "samples.json").read_text(encoding="utf-8"))["light"]}
    assert sig["cmx_noft"]["f1"]["diff"] == pytest.approx(-M.unigram_f1(noft["response"], ref["1-8"]["reference"]) / 12,
                                                          abs=1e-5)
    heavy = m["significance"]["heavy"]
    assert heavy["cmx_noft"]["f1"]["diff"] == 0.0 and heavy["cmx_noft"]["f1"]["p"] == 1.0  # identical outputs
    assert "cmx_noxl" not in heavy  # left out of the heavy tables


def test_evaluate_tables_content(evaluated):
    out, _ = evaluated
    abl = (out / "tables" / "ablation.md").read_text(encoding="utf-8")
    assert "| Light |" in abl and "| Heavy |" in abl and "| EN |" not in abl  # ablations: Hinglish only
    assert "w/o cross-lingual retriever" in abl.split("| Heavy |")[0]
    reg = (out / "tables" / "register.md").read_text(encoding="utf-8")
    assert "Gold reference" in reg and "Translate-Pivot | – | – | – |" in reg
    sig = list(csv.DictReader(open(out / "tables" / "significance.csv", encoding="utf-8")))
    assert {"f1_diff", "f1_ci_low", "f1_ci_high", "f1_p"} <= set(sig[0])
    main = (out / "tables" / "main_heavy.md").read_text(encoding="utf-8")
    assert "11 common turns" in main and "BERTScore" not in main  # no empty column without BERTScore


@pytest.mark.skipif(not shutil.which("pdflatex"), reason="pdflatex not installed")
def test_latex_tables_compile(evaluated, tmp_path):
    out, _ = evaluated
    ieee = shutil.which("kpsewhich") and subprocess.run(["kpsewhich", "IEEEtran.cls"], capture_output=True,
                                                         text=True).stdout.strip()
    cls = "IEEEtran" if ieee else "article"
    body = "\n".join(f"\\input{{{(out / 'tables' / f'{n}.tex').as_posix()}}}\n\\clearpage" for n in TABLES)
    (tmp_path / "doc.tex").write_text(f"\\documentclass{{{cls}}}\n\\usepackage{{booktabs,graphicx}}\n"
                                      f"\\begin{{document}}\n{body}\n\\end{{document}}\n", encoding="utf-8")
    res = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "doc.tex"], cwd=tmp_path,
                         capture_output=True, text=True, timeout=180)
    assert res.returncode == 0, res.stdout[-2000:]
    assert "Overfull" not in res.stdout


def test_evaluate_handles_missing_systems(synthetic, tmp_path):
    ev = load_script("evaluate")
    os.makedirs(tmp_path / "tables")
    for name in ("main_light.md", "judge.md"):  # left over from earlier runs
        (tmp_path / "tables" / name).write_text("old", encoding="utf-8")
    code = ev.main(["--runs_dir", str(synthetic / "runs"), "--out_dir", str(tmp_path), "--profiler", "none",
                    "--no_bertscore", "--samples_file", str(synthetic / "samples.json"), "--subset_file",
                    str(synthetic / "subset.json"), "--systems", "zero_shot,pivot", "--versions", "en,heavy",
                    "--no_figures"])
    assert code == 0
    m = json.loads((tmp_path / "eval" / "metrics.json").read_text(encoding="utf-8"))
    assert set(m["results"]["en"]) == {"zero_shot"} and set(m["results"]["heavy"]) == {"zero_shot", "pivot"}
    assert m["significance"] == {} and "cmi_gap" not in m["results"]["heavy"]["pivot"]["common"]
    assert (tmp_path / "tables" / "main_en.md").exists()
    assert not (tmp_path / "tables" / "ablation.md").exists() and not (tmp_path / "tables" / "register.md").exists()
    assert not (tmp_path / "tables" / "main_light.md").exists()  # stale: light was not evaluated this time
    assert (tmp_path / "tables" / "judge.md").read_text(encoding="utf-8") == "old"  # not evaluate.py's file


# ================================================================== LLM judge
def test_parse_verdicts():
    jd = load_script("judge")
    text = ('Sure!\n```json\n{"analysis": "B is warmer.", "Fluency": "a", "Identification": "Tie", "comforting": "B",'
            ' "Suggestions": "tie", "Overall": "Response B", "Language Naturalness": "A"}\n```')
    assert jd.parse_verdicts(text) == {"fluency": "A", "identification": "tie", "comforting": "B",
                                       "suggestion": "tie", "overall": "B", "naturalness": "A"}
    broken = "fluency: A, identification: B, comforting: tie, suggestion: tie, overall: 'B', naturalness: \"A\""
    assert jd.parse_verdicts(broken)["overall"] == "B"
    assert jd.parse_verdicts('{"fluency": "A", "overall": "B"}') is None  # dimensions missing
    assert jd.parse_verdicts('{"fluency": "maybe", "identification": "A", "comforting": "A", "suggestion": "A", '
                             '"overall": "A", "naturalness": "A"}') is None
    assert jd.parse_verdicts("") is None


def test_combine_orders_and_sampling():
    jd = load_script("judge")
    ab = dict(fluency="A", identification="A", comforting="tie", suggestion="B", overall="A", naturalness="B")
    ba = dict(fluency="B", identification="A", comforting="tie", suggestion="A", overall="tie", naturalness="B")
    assert jd.combine(ab, ba) == dict(fluency="win", identification="tie", comforting="tie", suggestion="lose",
                                      overall="tie", naturalness="tie")
    cands = [f"{c}-{t}" for c in range(20) for t in range(1, 30, 2)]
    s = jd.sample_uids(cands, 10, seed=42)
    assert len(s) == 10 and len(set(s)) == 10 and set(s) <= set(cands) and s == jd.sample_uids(cands, 10, seed=42)
    reduced = [u for u in cands if u not in s[5:]][:-1] + s[5:]  # drop one turn outside the sample
    assert jd.sample_uids(reduced, 10, seed=42) == s
    assert jd.sample_uids(cands, 10, seed=1) != s


@pytest.fixture(scope="module")
def judged(synthetic):
    jd = load_script("judge")
    out = synthetic / "judge_out"
    args = ["--dry_run", "--runs_dir", str(synthetic / "runs"), "--samples_file", str(synthetic / "samples.json"),
            "--out_dir", str(out), "--n", "8"]
    assert jd.main(args) == 0
    return jd, out, args


def test_judge_dry_run(synthetic, judged):
    jd, out, _ = judged
    for pair in ("codemixesc_vs_maesc", "codemixesc_vs_pivot"):
        for v in ("light", "heavy"):
            assert (out / "judge" / f"{pair}_{v}.jsonl").exists()
    items = [json.loads(x) for x in (out / "judge" / "codemixesc_vs_maesc_light.jsonl").read_text(encoding="utf-8")
             .splitlines()]
    assert len(items) == 8 and "1-8" not in {it["uid"] for it in items}  # the failed turn is never sampled
    retried = 0
    for it in items:
        assert it["final"] is not None
        if it["identical"]:
            assert it["order_ab"] is None and set(it["final"].values()) == {"tie"}
            continue
        retried += it["order_ab"]["attempts"] > 1 or it["order_ba"]["attempts"] > 1
        # the fake judge prefers the longer response in both orders: the swap must map it back to a
        la, lb = len(it["response_a"]), len(it["response_b"])
        assert it["final"]["fluency"] == ("win" if la > lb else "lose" if lb > la else "tie")
        assert it["final"]["overall"] == it["final"]["fluency"]
    assert retried  # some first answers were prose: the attempt suffix produced a usable second answer
    summary = json.loads((out / "judge" / "summary.json").read_text(encoding="utf-8"))
    s = summary["results"]["codemixesc_vs_maesc_light"]
    assert s["n"] == 8 and s["fluency"]["win"] + s["fluency"]["tie"] + s["fluency"]["lose"] == pytest.approx(100)
    tex = (out / "tables" / "judge.tex").read_text(encoding="utf-8")
    assert "\\toprule" in tex and "Language Naturalness" in tex


def test_human_sheets_round_trip(synthetic, judged):
    jd, out, args = judged
    assert jd.main(args + ["--export_human", "--human_n", "12"]) == 0
    key = json.loads((out / "human" / "key.json").read_text(encoding="utf-8"))
    assert {k["A"] for k in key.values()} == {"codemixesc", "maesc", "pivot"}  # A/B randomised per item
    assert all(k["pair"][0] == "codemixesc" and {k["A"], k["B"]} == set(k["pair"]) for k in key.values())
    assert jd.main(args + ["--export_human", "--human_n", "3"]) == 0
    key = json.loads((out / "human" / "key.json").read_text(encoding="utf-8"))
    with open(out / "human" / "sheet.csv", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(key) == 12  # 3 turns x 2 pairs x 2 versions
    assert not re.search(r"codemixesc|maesc|pivot", (out / "human" / "sheet.csv").read_text(encoding="utf-8"))
    # two volunteers who agree: codemixesc better on fluency, the other system on overall, ties elsewhere
    for name in ("sheet_asha", "sheet_ravi"):
        with open(out / "human" / f"{name}.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            for r in rows:
                k = key[r["item"]]
                cmx = "A" if k["A"] == "codemixesc" else "B"
                other = "B" if cmx == "A" else "A"
                w.writerow(dict(r, fluency=cmx, overall=other.lower(), identification="tie", comforting="Tie",
                                suggestion="", naturalness="tie"))
    sheets = [str(out / "human" / f"{n}.csv") for n in ("sheet_asha", "sheet_ravi")]
    assert jd.main(["--out_dir", str(out), "--import_human"] + sheets) == 0
    summary = json.loads((out / "human" / "summary.json").read_text(encoding="utf-8"))["results"]
    for s in summary.values():
        assert s["fluency"]["win"] == 100.0 and s["overall"]["lose"] == 100.0 and s["comforting"]["tie"] == 100.0
        assert s["suggestion"]["n"] == 0 and s["suggestion"]["win"] is None  # left blank: not counted as ties
        assert s["kappa_annotators"] == pytest.approx(1.0) and s["n_llm_overlap"] > 0
    assert (out / "tables" / "human.md").exists() and (out / "tables" / "human.tex").exists()
