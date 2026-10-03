"""Evaluation metrics of CodeMixESC: pure functions, defined here once for every system.
docs/EVALUATION.md gives the definitions with the reasons behind each choice.

Tokenizer. Every token-based metric uses tokenize(): NFC normalisation, lowercasing, then the
regex  \\w+|[^\\w\\s]  (a token is a run of word characters, or one other non-space character:
punctuation, symbol, emoji). It runs on the `regex` package (an nltk dependency), whose \\w
follows Unicode (letters, combining marks, digits); Python's `re` does not count combining
marks as word characters and would cut a Devanagari word such as हिंदी at every vowel sign.
Word tokens are the tokens that contain a letter or a digit.

Scales. Distinct-n, BLEU-n, F1, ROUGE-L, chrF and BERTScore are x100. CMI gaps keep the CMI
scale [0, 0.5] and Hindi-fraction gaps the [0, 1] scale. Script consistency, strategy
agreement and strategy match are percentages. JSD is in bits, [0, 1].

A turn without a response (failed turn, or the placeholder "None" of run records) is scored
as the empty string by the quality metrics, so a failure can never raise a system's score;
register and length statistics describe produced text and skip such turns.
"""
import math
import os
import unicodedata
from collections import Counter

import numpy as np
import regex

from .esconv import STRATEGIES

TOKEN_PATTERN = r"\w+|[^\w\s]"
TOKEN_RE = regex.compile(TOKEN_PATTERN)
_ALNUM_RE = regex.compile(r"[^\W_]")  # a letter or digit; Unicode-aware like TOKEN_RE
ARTICLES = frozenset({"a", "an", "the"})
LABELS = STRATEGIES + ["None"]  # strategy categories of the stability analysis
MISSING = "None"  # what run records hold when a system produced no response
BERTSCORE_MODEL = "bert-base-multilingual-cased"


# ------------------------------------------------------------------ text
def tokenize(text):
    return TOKEN_RE.findall(unicodedata.normalize("NFC", text or "").lower())


def is_word(token):
    return bool(_ALNUM_RE.search(token))


def word_tokens(text):
    """Tokens without punctuation, symbols and emojis."""
    return [t for t in tokenize(text) if is_word(t)]


def is_missing(response):
    """True for a turn without a response: null, empty, or the placeholder 'None'."""
    return response is None or not str(response).strip() or str(response).strip() == MISSING


def hypothesis(response):
    """The text that is scored; a missing response is the empty string."""
    return "" if is_missing(response) else str(response).strip()


def word_count(text):
    """Words as the prompts' 30-word limit counts them: whitespace-separated chunks that
    contain at least one letter or digit (a lone '-' or emoji is not a word)."""
    return sum(1 for w in (text or "").split() if _ALNUM_RE.search(w))


def mean(values):
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def pct(flags):
    """Percentage of true values (None entries are skipped); None without values."""
    vals = [bool(f) for f in flags if f is not None]
    return 100.0 * sum(vals) / len(vals) if vals else None


# ------------------------------------------------------------------ response quality
def ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(token_lists, n):
    """Corpus-level Distinct-n (Li et al., 2016), x100: distinct n-grams / all n-grams, both
    counted over all responses together (n-grams never cross a response boundary)."""
    grams = [g for toks in token_lists for g in ngrams(toks, n)]
    return 100.0 * len(set(grams)) / len(grams) if grams else 0.0


def corpus_bleu(hyp_tokens, ref_tokens, n):
    """Cumulative corpus BLEU-n (Papineni et al., 2002), x100: NLTK's corpus_bleu with uniform
    weights 1/n over the 1..n-gram precisions and one reference per hypothesis.

    Smoothing: NLTK method 3 (NIST geometric sequence smoothing). It changes only an n-gram
    order without a single match, whose precision becomes 1 / (2^k * #hypothesis n-grams) with
    k = 1, 2, ... for successive empty orders. A full test set therefore gets exactly the
    unsmoothed BLEU, while a small corpus without any 3-gram match keeps a non-zero BLEU-3.
    The score is 0 only when not a single unigram matches."""
    from nltk.translate.bleu_score import SmoothingFunction
    from nltk.translate.bleu_score import corpus_bleu as nltk_corpus_bleu
    if not hyp_tokens:
        return 0.0
    return 100.0 * nltk_corpus_bleu([[list(r)] for r in ref_tokens], [list(h) for h in hyp_tokens],
                                    weights=(1.0 / n,) * n, smoothing_function=SmoothingFunction().method3)


def f1_tokens(text):
    """ParlAI's answer normalisation on our tokenizer: lowercase, no punctuation, no articles."""
    return [t for t in word_tokens(text) if t not in ARTICLES]


def unigram_f1(hyp, ref):
    """ParlAI-style unigram F1 (Miller et al., 2017), x100: harmonic mean of the precision and
    recall of the hypothesis words against the reference words, as bags (clipped counts)."""
    h, r = f1_tokens(hyp), f1_tokens(ref)
    same = sum((Counter(h) & Counter(r)).values())
    if same == 0:
        return 0.0
    p, rec = same / len(h), same / len(r)
    return 100.0 * 2 * p * rec / (p + rec)


def lcs_length(a, b):
    """Length of the longest common subsequence of two token lists (two-row DP)."""
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
    """ROUGE-L (Lin, 2004), x100: F-measure with beta = 1 (rouge_score's fmeasure) of the
    longest common subsequence of the word tokens, without stemming. rouge_score keeps only
    [a-z0-9] and silently deletes Devanagari and accented words; on ASCII text both give the
    same score (tests/test_metrics.py)."""
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
        _CHRF = CHRF()  # chrF: character 6-grams, beta = 2, case-sensitive, whitespace ignored
    return _CHRF


def chrf_corpus(hyps, refs):
    """Corpus chrF (Popovic, 2015) with sacrebleu's default settings, 0-100: character n-gram
    statistics are summed over the corpus before the F-score is taken. Character n-grams give
    partial credit to the many spellings of Romanised Hindi (nahi / nahin / nhi)."""
    return _chrf().corpus_score(list(hyps), [list(refs)]).score


def chrf_sentence(hyp, ref):
    """Sentence-level chrF, 0-100 (the per-turn score used by the paired bootstrap)."""
    return _chrf().sentence_score(hyp, [ref]).score


def chrf_signature():
    _chrf().sentence_score("a", ["a"])  # sacrebleu needs one evaluation before it can sign
    return str(_chrf().get_signature())


class BERTScore:
    """Multilingual BERTScore F1 (Zhang et al., 2020), x100.

    Encoder bert-base-multilingual-cased at bert_score's default layer for it (9), no idf
    weighting and no baseline rescaling (bert_score ships no baseline for Hinglish), so
    absolute values sit in a narrow high band: compare systems, not absolute numbers. An empty
    hypothesis scores 0 without running the model. Imports and downloads happen only here,
    so every other metric works without torch models."""

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


# ------------------------------------------------------------------ register match
def register_match(profiler, seeker_utterances, response):
    """Register match of one response to its seeker: the quantities the Register Gate checks.

    CMI_s = profiler.profile(seeker utterances so far)['cmi'] is the seeker's pooled CMI, the
    profile R the system's gate uses; CMI_r = profiler.stats(response)['cmi']. The CMI is
    symmetric in the two languages (80 % and 20 % Hindi words give the same CMI), so the
    Hindi-fraction gap |hi_frac_r - hi_frac_s| is reported as well: only it notices a reply
    that flips the dominant language. script_match is None when the seeker has written no
    letters yet (no target script); a response without letters does not match."""
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
    """Means over the turns with a response (rows from register_match). script_consistency is
    the % of turns whose response script equals the seeker's, over the turns where the seeker's
    script is defined."""
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


# ------------------------------------------------------------------ strategies
_CANON = {s.lower(): s for s in STRATEGIES}
_STRATEGY_RE = regex.compile("|".join(regex.escape(s) for s in sorted(STRATEGIES, key=len, reverse=True)),
                             regex.IGNORECASE)


def normalize_strategy(pred):
    """One of the 8 ESConv strategy names (case-insensitive) or 'None' (no strategy predicted,
    or a name outside the label set)."""
    return _CANON.get(str(pred if pred is not None else "").strip().lower(), "None")


def gold_strategies(gold):
    """Gold strategies of a turn. Two merged supporter utterances are labelled 'A and B'; names
    are matched whole, so 'Affirmation and Reassurance' is never split at its 'and'."""
    return {_CANON[m.lower()] for m in _STRATEGY_RE.findall(gold or "")}


def strategy_counts(preds, labels=LABELS):
    c = Counter(preds)
    return [c.get(label, 0) for label in labels]


def jsd(p, q):
    """Jensen-Shannon divergence in bits, (KL(P||M) + KL(Q||M)) / 2 with M = (P + Q) / 2 and
    base-2 logarithms: 0 for identical and 1 for disjoint distributions. p, q are counts or
    probabilities over the same categories; None if either is empty. (scipy's jensenshannon
    returns the square root of this, the JS distance.)"""
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
    """Strategy stability of a system between two versions of the same turns (EN vs Hinglish).

    pred_a, pred_b: {uid: predicted strategy}; only their shared uids are compared.
    jsd: JSD between the two distributions over the 8 strategies + 'None' ("with None").
    jsd_strategies: JSD over the 8 strategies on the turns where both versions predicted a
    strategy ("strategies only"; same turns on both sides).
    agreement, agreement_strategies: % of those turns with the same prediction."""
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
    """(% of turns whose predicted strategy is one of the gold strategies, 'None' counting as a
    miss; the same % over the turns with a predicted strategy). preds are normalised names,
    golds sets from gold_strategies()."""
    pairs = list(zip(preds, golds))
    return pct(p in g for p, g in pairs), pct(p in g for p, g in pairs if p != "None")


# ------------------------------------------------------------------ statistics
def paired_bootstrap(a, b, n_boot=2000, seed=42, ci=0.95):
    """Paired bootstrap over turns (Efron & Tibshirani, 1993; Koehn, 2004) for the difference
    d = mean(a) - mean(b), where a[i] and b[i] are two systems' scores on the same turn i.

    Turns are resampled with replacement n_boot times (numpy default_rng(seed): reproducible).
    The CI is the percentile interval of the resampled differences d*. The two-sided p-value is
    p = min(1, 2 * (1 + min(#{d* <= 0}, #{d* >= 0})) / (n_boot + 1)), Koehn's "one system is
    better in x % of the resamples" made two-sided; it agrees with the CI (p < 1 - ci exactly
    when the CI excludes 0, up to the +1 correction that keeps p > 0)."""
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
    """Cohen's kappa of two equally long label sequences; None when undefined (no items, or a
    chance agreement of 1 because both raters used one and the same label throughout)."""
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
    """Two-sided exact sign test of wins against losses, ties left out: the probability under
    Binomial(n, 1/2) of a split at least as uneven as the observed one."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2 ** n
    return min(1.0, 2 * tail)
