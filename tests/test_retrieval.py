"""Cross-lingual retriever pipeline: pair cleaning and PHINC loading, the rewrite batching of
build_pairs (dry run with the fake LLM), retrieval metrics on toy data, query sets, the batch
sampler, a training smoke test on a tiny random BERT, and eval_retrieval on stub and tiny models.
Nothing is downloaded: huggingface.co and zenodo.org are never contacted."""
import csv
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import urllib.error
import zipfile

import numpy as np
import pytest

from codemixesc import retrieval_eval as RE
from codemixesc.esconv import ESCONV_PATH, ROOT, all_samples, case_bank, dev_samples, load_esconv

needs_esconv = pytest.mark.skipif(not os.path.exists(ESCONV_PATH),
                                  reason="data/esconv/ESConv.json missing (run scripts/setup_data.py)")
OFFLINE = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "CODEMIX_DEVICE": "cpu"}


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"cmx_script_{name}", os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_script(name, args, env_extra, timeout=600):
    env = dict(os.environ, **OFFLINE, **env_extra)
    return subprocess.run([sys.executable, os.path.join(ROOT, "scripts", f"{name}.py"), *args], cwd=ROOT, env=env,
                          capture_output=True, text=True, encoding="utf-8", timeout=timeout)


@pytest.fixture(scope="module")
def bp():
    return load_script("build_pairs")


@pytest.fixture(scope="module")
def fake_hien(tmp_path_factory):
    from codemixesc.testing import write_fake_hien
    if not os.path.exists(ESCONV_PATH):
        pytest.skip("ESConv.json missing")
    return write_fake_hien(str(tmp_path_factory.mktemp("hien")))


@pytest.fixture(scope="module")
def pairs_dir(tmp_path_factory, bp):
    """A dry-run build of the pairs (fake LLM, synthetic PHINC), shared by several tests."""
    if not os.path.exists(ESCONV_PATH):
        pytest.skip("ESConv.json missing")
    out = tmp_path_factory.mktemp("pairs")
    phinc = bp.write_synthetic_phinc(str(out / "phinc.csv"), load_esconv())
    r = run_script("build_pairs", ["--dry_run", "--phinc", phinc, "--n_esconv", "300", "--out_dir", str(out),
                                   "--workers", "2"], {})
    assert r.returncode == 0, r.stdout + r.stderr
    return out


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    """A 2-layer randomly initialised BERT with a WordPiece vocabulary trained on case-bank text,
    wrapped as a SentenceTransformer (stands in for paraphrase-multilingual-mpnet-base-v2)."""
    if not os.path.exists(ESCONV_PATH):
        pytest.skip("ESConv.json missing")
    import torch
    from sentence_transformers import SentenceTransformer, models
    from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers, processors, trainers
    from tokenizers import models as tok_models
    from transformers import BertConfig, BertModel, BertTokenizerFast
    from codemixesc.testing import fake_hinglish
    base = tmp_path_factory.mktemp("tiny")
    texts = [b["post"] for b in case_bank()[:4000]]
    texts += [fake_hinglish(t, "heavy") for t in texts[:1000]]
    tok = Tokenizer(tok_models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.train_from_iterator(texts, trainers.WordPieceTrainer(vocab_size=2000, special_tokens=[
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]))
    cls, sep = tok.token_to_id("[CLS]"), tok.token_to_id("[SEP]")
    tok.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
                                                       special_tokens=[("[CLS]", cls), ("[SEP]", sep)])
    tok.decoder = decoders.WordPiece()
    hf = str(base / "hf")
    BertTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
                      sep_token="[SEP]", mask_token="[MASK]").save_pretrained(hf)
    torch.manual_seed(0)
    BertModel(BertConfig(vocab_size=tok.get_vocab_size(), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                         intermediate_size=64, max_position_embeddings=256)).save_pretrained(hf)
    path = str(base / "st")
    SentenceTransformer(modules=[models.Transformer(hf, max_seq_length=128), models.Pooling(32, "mean")],
                        device="cpu").save(path)
    return path


# ============================================================================ cleaning
def test_clean_text(bp):
    assert bp.clean_text("RT @user_1: yaar &amp; exam tension https://t.co/xyz  #pareshan  @bob") == \
        "yaar & exam tension pareshan"
    assert bp.clean_text("&amp;amp; kya &quot;baat&quot; hai\n\n www.example.com/x RT") == '& kya "baat" hai'
    assert bp.clean_text("mail me at a@b.com pls") == "mail me at a@b.com pls"  # an e-mail is not a mention
    assert bp.clean_text("PARTY at my place") == "PARTY at my place"  # RT only as a standalone token
    assert bp.word_count("main theek hoon :) 100%") == 4 and bp.word_count("I'm fine") == 2


def test_is_roman_and_text_key(bp):
    assert bp.is_roman("bahut tension hai 😟") and bp.is_roman("café yaar")
    assert not bp.is_roman("मुझे tension hai") and not bp.is_roman("123 :)") and not bp.is_roman("بہت اچھا")
    assert RE.text_key("Happy  birthday!!") == RE.text_key("happy birthday") == "happy birthday"


def test_clean_pairs_steps(bp):
    pairs = [{"hi": "yaar bahut tension hai #exam", "en": "friend, so much stress #exam"},
             {"hi": "Yaar bahut   tension hai exam!", "en": "duplicate of the first (normalised)"},
             {"hi": "ok yaar", "en": "ok friend"},                              # < 3 words
             {"hi": "मुझे बहुत टेंशन है", "en": "I am very stressed"},          # Devanagari
             {"hi": "I am very stressed", "en": "I am very stressed."},        # not code-mixed
             {"hi": "kal milte hain @amit", "en": "see you tomorrow", "src": "phinc"}]
    out, steps = bp.clean_pairs(pairs, 3)
    assert list(steps.values()) == [5, 4, 3, 2]  # >= 3 words, Roman, code-mixed, deduplicated
    assert [p["hi"] for p in out] == ["yaar bahut tension hai exam", "kal milte hain"]
    assert out[1]["src"] == "phinc"  # extra keys survive


# ============================================================================ PHINC loading
def _csv_bytes(rows, delimiter=",", encoding="utf-8"):
    buf = io.StringIO(newline="")
    csv.writer(buf, delimiter=delimiter).writerows(rows)
    return buf.getvalue().encode(encoding)


def test_phinc_csv_by_name_cp1252(bp, tmp_path):
    p = tmp_path / "English-Hindi code-mixed parallel corpus.csv"
    p.write_bytes(_csv_bytes([["SENTENCE", "english translation"], ["café mein milte hain yaar", "see you at the café"],
                              ["multi\nline tweet hai yeh", "this is a multi line tweet"]], encoding="cp1252"))
    pairs, meta = bp.load_phinc(str(p))
    assert meta["encoding"] == "cp1252" and meta["column_choice"] == "by name" and meta["rows"] == 2
    assert pairs[0] == {"hi": "café mein milte hain yaar", "en": "see you at the café"}
    assert pairs[1]["hi"] == "multi\nline tweet hai yeh"  # quoted newline kept inside the field


def test_phinc_tsv_fallback_and_zip(bp, tmp_path):
    rows = [["Unnamed: 0", "col_a", "col_b"], ["0", "kya haal hai bhai", "how are you brother"],
            ["1", "sab theek hai yaar", "all is well friend"]]
    tsv = tmp_path / "phinc.tsv"
    tsv.write_bytes(b"\xef\xbb\xbf" + _csv_bytes(rows, delimiter="\t"))
    pairs, meta = bp.load_phinc(str(tsv))
    assert meta["delimiter"] == "\t" and meta["encoding"] == "utf-8-sig"
    assert meta["columns"] == ["col_a", "col_b"] and meta["column_choice"] == "first two text columns"
    assert pairs[1] == {"hi": "sab theek hai yaar", "en": "all is well friend"}
    zp = tmp_path / "PHINC.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("PHINC/README.txt", "x" * 5000)  # larger, but a .txt never beats a .csv
        z.writestr("PHINC/data.csv", _csv_bytes([["Sentence", "English_Translation"], ["yeh sahi hai yaar", "this is right"]]))
        z.writestr("__MACOSX/PHINC/._data.csv", "junk")
    pairs, meta = bp.load_phinc(str(zp))
    assert meta["member"] == "PHINC/data.csv" and pairs == [{"hi": "yeh sahi hai yaar", "en": "this is right"}]


def test_phinc_download(bp, tmp_path, monkeypatch):
    payload = _csv_bytes([["Sentence", "English_Translation"], ["aaj mausam accha hai", "the weather is nice today"]])
    record = {"files": [{"key": "phinc.csv", "checksum": "md5:" + hashlib.md5(payload).hexdigest(),
                         "links": {"self": "https://zenodo.org/api/records/3605597/files/phinc.csv/content"}}]}

    def fake_urlopen(url, timeout):
        return io.BytesIO(json.dumps(record).encode() if url.endswith("/records/3605597") else payload)

    monkeypatch.setattr(bp, "_urlopen", fake_urlopen)
    paths = bp.download_phinc(str(tmp_path))
    assert [os.path.basename(p) for p in paths] == ["phinc.csv"] and bp.load_phinc(paths[0])[0][0]["hi"] == "aaj mausam accha hai"
    record["files"][0]["checksum"] = "md5:" + "0" * 32
    with pytest.raises(bp.PhincUnavailable, match="checksum"):
        bp.download_phinc(str(tmp_path / "again"))

    def offline(url, timeout):
        raise urllib.error.URLError("Tunnel connection failed: 403 Forbidden")

    monkeypatch.setattr(bp, "_urlopen", offline)
    with pytest.raises(bp.PhincUnavailable, match="Zenodo"):
        bp.download_phinc(str(tmp_path / "offline"))
    monkeypatch.setattr(bp, "RAW_DIR", str(tmp_path / "empty"))
    with pytest.raises(SystemExit, match="zenodo.org/records/3605597"):  # the clear manual-download message
        bp.main(["--n_esconv", "0", "--out_dir", str(tmp_path / "out")])


# ============================================================================ rewriting helpers
def test_parse_and_validate(bp):
    assert bp.parse_items('```json\n{"items": [{"id": 0, "text": "a"}, {"id": "1", "text": "b"}]}\n```') == {0: "a", 1: "b"}
    assert bp.parse_items('Sure: [{"id": 2, "hinglish": "c"}, {"x": 1}]') == {2: "c"}
    with pytest.raises(ValueError):
        bp.parse_items("Sorry, I cannot help with that.")
    src = "I am worried about my exams and my parents"
    assert bp.validate_rewrite(src, "Yaar I am worried about my exams aur parents ka pressure hai") == (True, "")
    assert bp.validate_rewrite(src, "") == (False, "empty")
    assert bp.validate_rewrite(src, "मुझे exams ki tension hai") == (False, "not Roman script")
    assert bp.validate_rewrite(src, "I am worried about my exams and my parents!") == (False, "identical")
    assert bp.validate_rewrite(src, "exams yaar") == (False, "length")
    from codemixesc.testing import LexiconProfiler
    assert bp.validate_rewrite(src, "I am really worried about my exams and my mother", LexiconProfiler()) == (False, "no Hindi")


def test_prompt_and_progress(bp, tmp_path):
    desc = {"light": "LIGHT desc", "heavy": "HEAVY desc"}
    p1 = bp.build_prompt("light", desc["light"], ["a b c d", "e f g h"])
    p2 = bp.build_prompt("light", desc["light"], ["a b c d"], ["it was returned unchanged"], attempt=2)
    assert "Use a light Hindi-English mix" in p1 and "LIGHT desc" in p1 and "ids 0..1" in p1 and "problem" not in p1
    assert '"problem": "it was returned unchanged"' in p2 and p2.endswith("following all the rules.)") and "Attempt 2" in p2
    prog = bp.Progress(str(tmp_path / "p.jsonl"))
    key = bp.Progress.key("m", 0.7, p1)
    prog.put(key, '{"items": []}', {"tag": "t"})
    with open(tmp_path / "p.jsonl", "a", encoding="utf-8") as f:
        f.write('{"key": "torn')  # an interrupted write
    assert bp.Progress(str(tmp_path / "p.jsonl")).get(key) == '{"items": []}'
    assert key != bp.Progress.key("m", 0.7, p2) != bp.Progress.key("other", 0.7, p1)


# ============================================================================ build_pairs dry run
@needs_esconv
def test_build_pairs_dry_run(pairs_dir):
    train = [json.loads(line) for line in open(pairs_dir / "train.jsonl", encoding="utf-8")]
    val = [json.loads(line) for line in open(pairs_dir / "val.jsonl", encoding="utf-8")]
    stats = json.load(open(pairs_dir / "stats.json", encoding="utf-8"))
    for p in train + val:
        assert set(p) == {"hi", "en", "src", "level", "conv"} and p["src"] in ("phinc", "esconv")
        if p["src"] == "phinc":
            assert p["level"] is None and p["conv"] is None
        else:
            assert p["level"] in ("light", "heavy") and p["conv"] >= 100  # never a test conversation
    dev = set(json.load(open(os.path.join(ROOT, "data", "esconv_hien", "dev_conv_ids.json"))))
    assert not any(p["conv"] in dev for p in train + val if p["src"] == "esconv")
    assert not {RE.text_key(p["hi"]) for p in train} & {RE.text_key(p["hi"]) for p in val}
    es_val_convs = {p["conv"] for p in val if p["src"] == "esconv"}
    assert es_val_convs and not es_val_convs & {p["conv"] for p in train if p["src"] == "esconv"}  # split by conversation
    ph, es = stats["phinc"]["steps"], stats["esconv"]["steps"]
    assert ph["rows read"] > ph["deduplicated (normalised Hinglish side)"] > 0
    assert list(ph.values()) == sorted(ph.values(), reverse=True)
    rw = stats["esconv"]["rewriting"]
    assert es["sampled"] == 300 and rw["failed_first_attempt"] > 0 and rw["valid_after_retry"] > 0  # retry path ran
    assert es["valid rewrite (after one retry)"] == rw["valid_first_attempt"] + rw["valid_after_retry"]
    assert stats["leakage_checks"] == {"pairs_from_test_conversations": 0, "pairs_from_dev_conversations": 0,
                                       "hinglish_in_train_and_val": 0}
    assert stats["train"]["total"] == len(train) and stats["val"]["total"] == len(val)
    levels = stats["esconv"]["levels"]
    assert abs(levels["light"] - levels["heavy"]) < 40  # batches alternate Light / Heavy


@needs_esconv
def test_build_pairs_resumes_from_progress(pairs_dir):
    before = open(pairs_dir / "train.jsonl", encoding="utf-8").read()
    r = run_script("build_pairs", ["--dry_run", "--phinc", str(pairs_dir / "phinc.csv"), "--n_esconv", "300",
                                   "--out_dir", str(pairs_dir), "--workers", "2"], {})
    assert r.returncode == 0, r.stderr
    calls = json.load(open(pairs_dir / "stats.json", encoding="utf-8"))["esconv"]["rewriting"]["calls"]
    assert calls["real"] == 0 and calls["from_progress"] > 0  # every batch came from esconv_rewrites.jsonl
    assert open(pairs_dir / "train.jsonl", encoding="utf-8").read() == before  # deterministic output


# ============================================================================ metrics
BANK = [{"problem_type": "job", "strategy": "Question"}, {"problem_type": "job", "strategy": "Others"},
        {"problem_type": "breakup", "strategy": "Question"}, {"problem_type": "exam", "strategy": "Information"}]


def test_overlap_and_precision():
    assert RE.overlap_at_k([0, 1, 2], [2, 1, 3], k=3) == pytest.approx(2 / 3)
    assert RE.overlap_at_k([0, 1, 2, 3], [9, 8, 1, 0], k=2) == 0.0  # only the top-k count
    assert np.allclose(RE.overlap_at_k([[0, 1], [2, 3]], [[1, 0], [2, 0]], k=2), [1.0, 0.5])
    with pytest.raises(ValueError):
        RE.overlap_at_k([0, 1], [0, 1], k=3)
    assert np.allclose(RE.problem_type_precision_at_k([[0, 1, 2], [2, 3, 0]], ["job", "breakup"], BANK, k=3), [2 / 3, 1 / 3])
    assert RE.problem_type_precision_at_k([2, 0, 1], "job", BANK, k=1) == 0.0
    assert np.allclose(RE.chance_precision(["job", "exam", "other"], BANK), [0.5, 0.25, 0.0])


def test_jsd_and_strategy_histograms():
    assert RE.jsd([1, 0], [0, 1]) == pytest.approx(1.0) and RE.jsd([2, 2], [1, 1]) == pytest.approx(0.0)
    assert RE.jsd([3, 1], [1, 3]) == pytest.approx(RE.jsd([1, 3], [3, 1]))
    assert 0 < RE.jsd([3, 1], [1, 3]) < 1
    H = RE.strategy_histograms([[0, 1], [2, 3]], BANK, k=2)
    labels = RE.strategy_labels(BANK)
    assert H.shape == (2, len(labels)) and H.sum() == 4 and H[0, labels.index("Question")] == 1
    pooled, per = RE.strategy_jsd([[0, 1], [2, 3]], [[0, 1], [0, 3]], BANK, k=2)
    assert per[0] == 0.0 and per[1] == 0.0  # same strategies (Question / Information) even though case 2 != 0
    pooled, per = RE.strategy_jsd([[0, 2]], [[1, 3]], BANK, k=2)
    assert pooled == pytest.approx(1.0) and per[0] == pytest.approx(1.0)


def test_pair_retrieval_and_topk():
    e = np.eye(4, dtype=np.float32)
    assert RE.pair_retrieval_metrics(e, e, ["a", "b", "c", "d"]) == (1.0, 1.0)
    acc, mrr = RE.pair_retrieval_metrics(e, e[[1, 0, 2, 3]], ["a", "b", "c", "d"])
    assert acc == 0.5 and mrr < 1
    assert RE.pair_retrieval_metrics(e[[0, 0, 2, 3]], e[[0, 0, 2, 3]], ["x y", "X y!", "c", "d"]) == (1.0, 1.0)  # shared translation
    assert RE.pair_retrieval_metrics(np.ones((3, 2)), np.ones((3, 2)), ["a", "b", "c"])[0] == 0.0  # collapse is no success
    q = np.array([[1.0, 0.0]], dtype=np.float32)
    bank = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.6, 0.8]], dtype=np.float32)
    assert RE.topk_from_embeddings(q, bank, k=3).tolist() == [[1, 2, 3]]  # stable: ties keep bank order


def test_bootstrap_ci():
    rng = np.random.default_rng(0)
    x = rng.normal(1.0, 1.0, 400)
    ci = RE.bootstrap_ci(x, n_boot=1000, seed=1)
    assert ci["value"] == pytest.approx(x.mean()) and ci["lo"] < ci["value"] < ci["hi"] and ci["n"] == 400
    assert RE.bootstrap_ci(x, n_boot=1000, seed=1) == ci  # deterministic
    groups = np.repeat(np.arange(40), 10)
    clustered = np.repeat(rng.normal(0, 1, 40), 10) + rng.normal(0, 0.1, 400)
    naive, cluster = RE.bootstrap_ci(clustered), RE.bootstrap_ci(clustered, groups=groups)
    assert cluster["hi"] - cluster["lo"] > 2 * (naive["hi"] - naive["lo"]) and cluster["n_groups"] == 40
    H = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 0, 1.0]])
    pooled = RE.bootstrap_ci(H, stat=lambda s, c: RE.jsd(s[:2], s[2:]), n_boot=200)
    assert pooled["value"] == pytest.approx(RE.jsd([2, 1], [1, 2])) and pooled["lo"] <= pooled["value"] <= pooled["hi"]
    assert RE.bootstrap_ci([])["value"] is None


# ============================================================================ query sets
@needs_esconv
def test_query_sets(fake_hien, monkeypatch):
    monkeypatch.setenv("CODEMIX_HIEN_DIR", fake_hien)
    en = {s["uid"]: s for s in all_samples("en")}
    heavy = RE.build_queries("test", "heavy")
    assert [q["uid"] for q in heavy] == list(en) and len(heavy) == 1210
    assert all(q["post_en"] == en[q["uid"]]["post"] and q["problem_type"] == en[q["uid"]]["problem_type"] for q in heavy)
    assert sum(q["post"] != q["post_en"] for q in heavy) > 1000  # the Hinglish post is what gets queried
    assert all(q["post"] == s["post"] for q, s in zip(heavy, all_samples("heavy")))
    dev = RE.build_queries("dev", "light")
    assert [q["uid"] for q in dev] == [s["uid"] for s in dev_samples("en")]
    ids = set(json.load(open(os.path.join(fake_hien, "dev_conv_ids.json"))))
    assert {q["conv_id"] for q in dev} == ids  # dev conv_id = the ESConv index
    assert RE.available_versions("dev") == ["en", "light", "heavy"]
    bank = RE.split_bank("dev")
    assert not {b["conv"] for b in bank} & ids and len(bank) < len(case_bank())


# ============================================================================ training
def test_no_duplicates_sampler_shuffles():
    from datasets import Dataset
    tr = load_script("train_retriever")
    hi = [f"hinglish sentence {i}" for i in range(60)]
    en = [f"english sentence {i % 50}" for i in range(60)]  # ten translations occur twice
    hi[7] = "Hinglish   Sentence 3!"  # a near-duplicate of row 3 after normalisation
    sampler = tr.ShuffledNoDuplicatesBatchSampler(Dataset.from_dict({"anchor": hi, "positive": en}), 8, False, seed=42)
    epoch0 = list(sampler)
    assert sorted(i for b in epoch0 for i in b) == list(range(60))  # every pair exactly once
    for batch in epoch0:
        keys = [RE.text_key(t) for i in batch for t in (hi[i], en[i])]
        assert len(keys) == len(set(keys))  # no in-batch duplicate text
    assert epoch0[0] != list(range(8))  # shuffled (3.3.1's sampler yields index order)
    assert list(sampler) == epoch0  # deterministic within an epoch (resume-safe)
    sampler.set_epoch(1)
    assert list(sampler) != epoch0 and len(sampler) == 8


@needs_esconv
def test_train_retriever_smoke(tmp_path, tiny_model, pairs_dir, fake_hien):
    out, log = tmp_path / "model", tmp_path / "train_log.json"
    r = run_script("train_retriever", [
        "--base_model", tiny_model, "--pairs_dir", str(pairs_dir), "--out_dir", str(out), "--ckpt_dir",
        str(tmp_path / "ckpt"), "--log", str(log), "--max_steps", "4", "--batch_size", "16", "--limit", "160",
        "--evals_per_epoch", "5", "--bank_limit", "1500", "--lr", "1e-3"], {"CODEMIX_HIEN_DIR": fake_hien})
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    L = json.load(open(log, encoding="utf-8"))
    assert L["device"]["device"] == "cpu" and L["params"]["word_embeddings_frozen"]  # auto: no GPU -> frozen
    assert 0 < L["params"]["frozen"] < L["params"]["total"]
    assert L["config"]["tau"] == pytest.approx(0.05) and L["config"]["loss"] == "MultipleNegativesRankingLoss"
    assert L["data"]["selection_metric"] == "dev" and L["data"]["dev_levels"] == ["heavy", "light"]
    assert [e["step"] for e in L["evals"]] == [2, 4] and L["best_step"] in (2, 4)
    for key in ("dev_overlap@10", "dev_p@10", "dev_primary", "val_acc@1", "val_mrr@10", "primary"):
        assert 0.0 <= L["final_eval"][key] <= 1.0 and key in L["eval_before_training"]
    best = next(e for e in L["evals"] if e["step"] == L["best_step"])
    assert L["final_eval"]["primary"] == pytest.approx(best["primary"], abs=1e-6)  # the best checkpoint was saved
    assert not (tmp_path / "ckpt").exists() and (out / "codemix_training.json").exists()
    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer(str(out), device="cpu").encode(["yaar bahut tension hai"])
    assert emb.shape == (1, 32)


@needs_esconv
def test_train_retriever_without_dev_files(tmp_path, tiny_model, pairs_dir):
    r = run_script("train_retriever", [
        "--base_model", tiny_model, "--pairs_dir", str(pairs_dir), "--out_dir", str(tmp_path / "m"), "--ckpt_dir",
        str(tmp_path / "c"), "--log", str(tmp_path / "log.json"), "--max_steps", "2", "--batch_size", "16",
        "--limit", "64", "--bank_limit", "500", "--freeze_embeddings", "no", "--cached", "--mini_batch_size", "4"],
        {"CODEMIX_HIEN_DIR": str(tmp_path / "no_hien")})
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "val MRR@10" in r.stderr  # the fallback is announced
    L = json.load(open(tmp_path / "log.json", encoding="utf-8"))
    assert L["data"]["selection_metric"] == "val_mrr@10" and L["params"]["frozen"] == 0
    assert L["final_eval"]["primary"] == L["final_eval"]["val_mrr@10"]
    assert L["config"]["loss"] == "CachedMultipleNegativesRankingLoss"


# ============================================================================ evaluation
@needs_esconv
def test_eval_retrieval_stub(tmp_path, fake_hien):
    r = run_script("eval_retrieval", ["--stub", "--n_boot", "200", "--results_dir", str(tmp_path)],
                   {"CODEMIX_HIEN_DIR": fake_hien})
    assert r.returncode == 0, r.stderr[-3000:]
    rep = json.load(open(tmp_path / "retrieval" / "retrieval_test.json", encoding="utf-8"))
    assert rep["subsets"] == {"all": {"en": 1210, "light": 1210, "heavy": 1210}, "s200": {"en": 200, "light": 200, "heavy": 200}}
    assert set(rep["results"]) == {"roberta", "labse", "mpnet", "mpnet-ft"} and not rep["skipped_encoders"]
    res = rep["results"]["mpnet-ft"]
    assert "overlap_self@10" not in res["en"]["all"] and "overlap_self@10" in res["heavy"]["all"]
    for v in ("light", "heavy"):
        for m in ("p@10", "overlap_self@10", "overlap_roberta_en@10", "jsd_strategy", "jsd_strategy_per_query"):
            c = res[v]["all"][m]
            assert 0.0 <= c["lo"] <= c["value"] <= c["hi"] <= 1.0 and c["n_groups"] == 100
    assert rep["results"]["roberta"]["en"]["all"]["overlap_roberta_en@10"]["value"] == 1.0
    d = rep["delta_vs_roberta"]["mpnet-ft"]["heavy"]["all"]["p@10"]
    assert d["value"] == pytest.approx(res["heavy"]["all"]["p@10"]["value"]
                                       - rep["results"]["roberta"]["heavy"]["all"]["p@10"]["value"])
    tex = open(tmp_path / "tables" / "retrieval_test.tex", encoding="utf-8").read()
    assert tex.count("\\begin{tabular}") == 1 and "\\toprule" in tex and "table" not in tex.replace("tabular", "")
    assert (tmp_path / "tables" / "retrieval_test_full.tex").exists() and (tmp_path / "tables" / "retrieval_test_s200.tex").exists()
    assert "| Encoder |" in open(tmp_path / "tables" / "retrieval_test.md", encoding="utf-8").read()


@needs_esconv
def test_eval_retrieval_real_models(tmp_path, tiny_model, fake_hien, monkeypatch):
    """The non-stub path (Retriever, SentenceTransformer loading, bank-embedding cache) with the tiny
    model registered as two encoders; an unavailable encoder is skipped; the dev split reuses the
    full-bank cache file instead of writing one for the smaller bank."""
    from codemixesc import retriever as R
    ev = load_script("eval_retrieval")
    monkeypatch.setenv("CODEMIX_HIEN_DIR", fake_hien)
    monkeypatch.setattr(R, "EMB_DIR", str(tmp_path / "emb"))
    monkeypatch.setitem(R.ENCODERS, "roberta", tiny_model)
    monkeypatch.setitem(R.ENCODERS, "mpnet-ft", tiny_model)
    monkeypatch.setitem(R.ENCODERS, "labse", str(tmp_path / "missing-model"))
    out = tmp_path / "results"
    with pytest.warns(UserWarning, match="skipping encoder labse"):
        ev.main(["--split", "dev", "--encoders", "roberta,labse,mpnet-ft", "--n_boot", "100", "--results_dir", str(out)])
    files = sorted(os.listdir(tmp_path / "emb"))
    assert len(files) == 2  # one file per encoder name, both for the FULL bank (no dev-bank file)
    assert all(np.load(tmp_path / "emb" / f).shape[0] == len(case_bank()) for f in files)
    rep = json.load(open(out / "retrieval" / "retrieval_dev.json", encoding="utf-8"))
    assert list(rep["skipped_encoders"]) == ["labse"] and rep["bank_size"] == len(RE.split_bank("dev"))
    same = rep["results"]["mpnet-ft"]["heavy"]["all"], rep["results"]["roberta"]["heavy"]["all"]
    assert same[0]["p@10"] == same[1]["p@10"]  # identical models give identical results
    assert rep["delta_vs_roberta"]["mpnet-ft"]["heavy"]["all"]["overlap_self@10"]["value"] == 0.0
    ev.main(["--split", "test", "--encoders", "roberta", "--versions", "en,heavy", "--n_boot", "100",
             "--results_dir", str(out)])
    assert sorted(os.listdir(tmp_path / "emb")) == files  # the test split reuses the same cache files
    rep = json.load(open(out / "retrieval" / "retrieval_test.json", encoding="utf-8"))
    assert rep["versions"] == ["en", "heavy"] and rep["results"]["roberta"]["heavy"]["all"]["n"] == 1210
