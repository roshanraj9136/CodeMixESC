import math
import os
import unicodedata
from collections import Counter

import numpy as np
import regex

from .esconv import STRATEGIES

TOKEN_PATTERN = r"\w+|[^\w\s]"
TOKEN_RE = regex.compile(TOKEN_PATTERN)
_ALNUM_RE = regex.compile(r"[^\W_]")
ARTICLES = frozenset({"a", "an", "the"})
LABELS = STRATEGIES + ["None"]
MISSING = "None"
BERTSCORE_MODEL = "bert-base-multilingual-cased"


def tokenize(text):
    return TOKEN_RE.findall(unicodedata.normalize("NFC", text or "").lower())


def is_word(token):
    return bool(_ALNUM_RE.search(token))


def word_tokens(text):
    return [t for t in tokenize(text) if is_word(t)]


def is_missing(response):
    return response is None or not str(response).strip() or str(response).strip() == MISSING


def hypothesis(response):
    return "" if is_missing(response) else str(response).strip()


def word_count(text):
    return sum(1 for w in (text or "").split() if _ALNUM_RE.search(w))


def mean(values):
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def pct(flags):
    vals = [bool(f) for f in flags if f is not None]
    return 100.0 * sum(vals) / len(vals) if vals else None


def ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(token_lists, n):
    grams = [g for toks in token_lists for g in ngrams(toks, n)]
    return 100.0 * len(set(grams)) / len(grams) if grams else 0.0


def corpus_bleu(hyp_tokens, ref_tokens, n):
    from nltk.translate.bleu_score import SmoothingFunction
    from nltk.translate.bleu_score import corpus_bleu as nltk_corpus_bleu
    if not hyp_tokens:
        return 0.0
    return 100.0 * nltk_corpus_bleu([[list(r)] for r in ref_tokens], [list(h) for h in hyp_tokens],
                                    weights=(1.0 / n,) * n, smoothing_function=SmoothingFunction().method3)


def f1_tokens(text):
    return [t for t in word_tokens(text) if t not in ARTICLES]


def unigram_f1(hyp, ref):
    h, r = f1_tokens(hyp), f1_tokens(ref)
    same = sum((Counter(h) & Counter(r)).values())
    if same == 0:
        return 0.0
    p, rec = same / len(h), same / len(r)
    return 100.0 * 2 * p * rec / (p + rec)


def lcs_length(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l(hyp, ref):
    h, r = word_tokens(hyp), word_tokens(ref)
    if not h or not r:
        return 0.0
    lcs = lcs_length(h, r)
    if lcs == 0:
        return 0.0
    p, rec = lcs / len(h), lcs / len(r)
    return 100.0 * 2 * p * rec / (p + rec)


_CHRF = None


def _chrf():
    global _CHRF
    if _CHRF is None:
        from sacrebleu.metrics import CHRF
        _CHRF = CHRF()
    return _CHRF


def chrf_corpus(hyps, refs):
    return _chrf().corpus_score(list(hyps), [list(refs)]).score


def chrf_sentence(hyp, ref):
    return _chrf().sentence_score(hyp, [ref]).score


def chrf_signature():
    _chrf().sentence_score("a", ["a"])
    return str(_chrf().get_signature())


class BERTScore:

    def __init__(self, model_type=BERTSCORE_MODEL, device=None, batch_size=64):
        import torch
        from bert_score import BERTScorer
        self.device = device or os.environ.get("CODEMIX_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.model_type = model_type
        self.scorer = BERTScorer(model_type=model_type, device=self.device, batch_size=batch_size)
        self.hash = self.scorer.hash

    def __call__(self, hyps, refs):
        out = [0.0] * len(hyps)
        idx = [i for i, h in enumerate(hyps) if h and h.strip()]
        if idx:
            _, _, f = self.scorer.score([hyps[i] for i in idx], [refs[i] for i in idx], batch_size=self.batch_size)
            for i, v in zip(idx, f.tolist()):
                out[i] = 100.0 * v
        return out


def register_match(profiler, seeker_utterances, response):
    from .profiler import script_of
    R = profiler.profile(seeker_utterances)
    st = profiler.stats(response)
    script_r = script_of(response or "")
    return {
        "cmi_s": R["cmi"], "cmi_r": st["cmi"], "cmi_gap": abs(st["cmi"] - R["cmi"]),
        "hi_frac_s": R["hi_frac"], "hi_frac_r": st["hi_frac"], "hi_frac_gap": abs(st["hi_frac"] - R["hi_frac"]),
        "dominant_s": R["dominant"], "dominant_r": "Hindi" if st["hi"] > st["en"] else "English",
        "script_s": R["script"], "script_r": script_r,
        "script_match": None if R["script"] == "None" else script_r == R["script"],
    }


def register_summary(rows):
    rows = [r for r in rows if r is not None]
    return {
        "n_register": len(rows),
        "cmi_gap": mean(r["cmi_gap"] for r in rows),
        "hi_frac_gap": mean(r["hi_frac_gap"] for r in rows),
        "cmi_r": mean(r["cmi_r"] for r in rows),
        "cmi_s": mean(r["cmi_s"] for r in rows),
        "hi_frac_r": mean(r["hi_frac_r"] for r in rows),
        "hi_frac_s": mean(r["hi_frac_s"] for r in rows),
        "n_script": sum(r["script_match"] is not None for r in rows),
        "script_consistency": pct(r["script_match"] for r in rows),
        "dominant_match": pct(r["dominant_r"] == r["dominant_s"] for r in rows),
    }


_CANON = {s.lower(): s for s in STRATEGIES}
_STRATEGY_RE = regex.compile("|".join(regex.escape(s) for s in sorted(STRATEGIES, key=len, reverse=True)),
                             regex.IGNORECASE)


def normalize_strategy(pred):
    return _CANON.get(str(pred if pred is not None else "").strip().lower(), "None")


def gold_strategies(gold):
    return {_CANON[m.lower()] for m in _STRATEGY_RE.findall(gold or "")}


def strategy_counts(preds, labels=LABELS):
    c = Counter(preds)
    return [c.get(label, 0) for label in labels]


def jsd(p, q):
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    if p.sum() <= 0 or q.sum() <= 0:
        return None
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2

    def kl(a):
        nz = a > 0
        return float(np.sum(a[nz] * np.log2(a[nz] / m[nz])))

    return min(1.0, max(0.0, 0.5 * kl(p) + 0.5 * kl(q)))


def strategy_stability(pred_a, pred_b):
    uids = sorted(set(pred_a) & set(pred_b))
    a = [normalize_strategy(pred_a[u]) for u in uids]
    b = [normalize_strategy(pred_b[u]) for u in uids]
    both = [(x, y) for x, y in zip(a, b) if x != "None" and y != "None"]
    return {
        "n": len(uids),
        "jsd": jsd(strategy_counts(a), strategy_counts(b)),
        "agreement": pct(x == y for x, y in zip(a, b)),
        "n_strategies": len(both),
        "jsd_strategies": jsd(strategy_counts([x for x, _ in both], STRATEGIES),
                              strategy_counts([y for _, y in both], STRATEGIES)) if both else None,
        "agreement_strategies": pct(x == y for x, y in both),
        "counts_a": dict(zip(LABELS, strategy_counts(a))),
        "counts_b": dict(zip(LABELS, strategy_counts(b))),
    }


def strategy_match(preds, golds):
    pairs = list(zip(preds, golds))
    return pct(p in g for p, g in pairs), pct(p in g for p, g in pairs if p != "None")


def paired_bootstrap(a, b, n_boot=2000, seed=42, ci=0.95):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired scores must have the same length")
    n = len(a)
    if n == 0:
        return None
    diff = a - b
    d = float(diff.mean())
    rng = np.random.default_rng(seed)
    boots = diff[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
    tail = min(int(np.sum(boots <= 0)), int(np.sum(boots >= 0)))
    return {"n": n, "mean_a": float(a.mean()), "mean_b": float(b.mean()), "diff": d,
            "ci_low": float(lo), "ci_high": float(hi), "p": min(1.0, 2 * (1 + tail) / (n_boot + 1)),
            "n_boot": n_boot, "seed": seed}


def cohen_kappa(a, b):
    if len(a) != len(b):
        raise ValueError("label sequences must have the same length")
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    if pe >= 1.0:
        return None
    return (po - pe) / (1 - pe)


def sign_test(wins, losses):
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2 ** n
    return min(1.0, 2 * tail)
