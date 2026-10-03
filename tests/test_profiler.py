"""Code-Mix Profiler: CMI definition, tokenisation, script detection, label mapping, and the
real tagging code (word alignment, chunking) on a tiny local BERT whose predictions are forced
through its classifier bias (huggingface.co is not needed)."""
import os

import pytest
import torch

from codemixesc import profiler as P
from codemixesc.testing import LexiconProfiler


def test_cmi_definition():
    assert P.cmi_from_tags([]) == 0.0
    assert P.cmi_from_tags(["X", "X"]) == 0.0  # n == u
    assert P.cmi_from_tags(["EN", "EN", "X"]) == 0.0  # monolingual
    assert P.cmi_from_tags(["EN", "HI"]) == pytest.approx(0.5)  # balanced mix is the maximum
    assert P.cmi_from_tags(["EN", "HI", "HI", "X"]) == pytest.approx(1 - 2 / 3)
    assert P.cmi_from_tags(["HI"] * 3 + ["EN"]) == pytest.approx(0.25)


def test_tokenize_and_language_independent_tokens():
    toks = P.tokenize("yaar result aaya :( I'm scared 100% https://x.y @bob #sad मुझे")
    assert toks == ["yaar", "result", "aaya", ":", "(", "I'm", "scared", "100", "%", "https://x.y", "@bob", "#sad", "मुझे"]
    assert [P._is_language_independent(t) for t in [":", "100", "https://x.y", "@bob", "#sad", "yaar", "I'm"]] == \
        [True, True, True, True, True, False, False]
    assert not P._needs_model("मुझे") and P._needs_model("yaar") and not P._needs_model("42")


def test_script_and_labels():
    assert P.script_of("bahut tension hai") == "Roman"
    assert P.script_of("मुझे डर लग रहा है") == "Devanagari"
    assert P.script_of("मुझे डर lag raha") == "Mixed"
    assert P.script_of("123 :)") == "None"
    assert [P.normalize_label(x) for x in ("EN", "hi", "OTHER", "LABEL_2", "Hindi")] == ["EN", "HI", "X", "X", "HI"]


def test_lexicon_profiler_profile_and_register():
    prof = LexiconProfiler()
    R = prof.profile(["yaar result aaya, bahut tension ho rahi hai", "ghar pe kaise bataun", "I am scared"])
    assert R["dominant"] == "Hindi" and R["script"] == "Roman" and 0.3 < R["cmi"] <= 0.5
    assert R["cmi_last"] == 0.0
    E = prof.profile(["I think the main issue is my job.", "Hi, it is hard."])
    assert E["cmi"] == 0.0 and P.is_plain_english(E)
    assert P.describe_register(E)[0].startswith("plain English")
    assert P.describe_register(R)[0].startswith("Hindi-dominant Hinglish")
    assert prof.profile([])["cmi"] == 0.0
    # long texts are chunked, never truncated: every word keeps its own tag
    long = " ".join(["mujhe bahut dukh hai and I feel lost"] * 60)
    st = prof.stats(long)
    assert st["n_lang"] == 480 and st["hi"] == 240


def _tiny_lid(path, bias):
    from transformers import BertConfig, BertForTokenClassification, BertTokenizerFast
    letters = [chr(c) for c in range(ord("a"), ord("z") + 1)]
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + letters + ["##" + c for c in letters] + ["'", "##'"]
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "vocab.txt"), "w") as f:
        f.write("\n".join(vocab))
    tok = BertTokenizerFast(vocab_file=os.path.join(path, "vocab.txt"), do_lower_case=True)
    cfg = BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                     intermediate_size=32, max_position_embeddings=512, num_labels=3,
                     id2label={0: "EN", 1: "HI", 2: "OTHER"}, label2id={"EN": 0, "HI": 1, "OTHER": 2})
    torch.manual_seed(0)
    model = BertForTokenClassification(cfg)
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.copy_(torch.tensor(bias, dtype=torch.float))
    model.save_pretrained(path)
    tok.save_pretrained(path)
    return path


def test_real_profiler_alignment_and_other_label(tmp_path):
    hi = P.Profiler(device="cpu", model_name=_tiny_lid(str(tmp_path / "hi"), [0.0, 10.0, 0.0]))
    words = "I'm so tired , 100 % done :) मुझे"
    tags = hi.tags(words)
    assert [t for _, t in hi.tag_many([words])[0]] == tags
    assert tags == ["HI", "HI", "HI", "X", "X", "X", "HI", "X", "X", "HI"]
    long = " ".join(["word"] * 350)  # 350 words -> 4 chunks, all labelled by the model
    assert hi.tags(long) == ["HI"] * 350
    other = P.Profiler(device="cpu", model_name=_tiny_lid(str(tmp_path / "other"), [0.0, 0.0, 10.0]))
    assert other.stats("I am fine")["n_lang"] == 0 and other.cmi("I am fine") == 0.0  # OTHER -> X
    en = P.Profiler(device="cpu", model_name=_tiny_lid(str(tmp_path / "en"), [10.0, 0.0, 0.0]))
    R = en.profile(["I am fine", "मुझे डर"])
    assert R["dominant"] == "English" and R["script"] == "Mixed" and R["cmi"] == pytest.approx(1 - 3 / 5)


class _FalsePositives(LexiconProfiler):
    """Simulates HingBERT-LID tagging English homographs (hi, me, to, he, do) as Hindi."""

    def _label_chunks(self, chunks):
        return [["HI" if w.lower() in {"hi", "me", "to", "he", "do", "us"} or w.lower() in
                 {"yaar", "bahut", "kya", "nahi", "hai", "mujhe", "karun"} else "EN" for w in ws] for ws in chunks]


def test_plain_english_robust_to_homograph_false_positives():
    prof = _FalsePositives()
    R = prof.profile(["Hi, how are you?", "I want to talk to someone, my boss told me to quit"])
    assert R["cmi"] > P.PLAIN_ENGLISH_CMI and R["n_hi_strong"] == 0
    assert P.is_plain_english(R) and P.target_cmi(R) == 0.0
    H = prof.profile(["yaar bahut tension hai", "mujhe kya karun pata nahi"])
    assert not P.is_plain_english(H) and P.target_cmi(H) == H["cmi"]
    one = prof.profile(["I am so tired yaar"])  # a single Hindi word is not enough evidence
    assert P.is_plain_english(one)
    assert P.tokenize("don’t you’re café") == ["don’t", "you’re", "café"]
