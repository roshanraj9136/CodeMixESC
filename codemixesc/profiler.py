"""Code-Mix Profiler: word-level language identification with HingBERT-LID and the
Code-Mixing Index (CMI) of Gamback and Das (2016).

    CMI = 1 - max_i(w_i) / (n - u)   if n > u,  else 0

w_i = number of words of language i, n = all words, u = language-independent words
(punctuation, numbers, emojis, URLs, @mentions, #hashtags). With two languages the CMI
lies in [0, 0.5]: 0 = monolingual, 0.5 = perfectly balanced mix.
"""
import os
import re
import threading

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

LID_MODEL = "l3cube-pune/hing-bert-lid"
WORD_RE = re.compile(r"https?://\S+|www\.\S+|[@#]\w+|[ऀ-ॿ]+|[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,]\d+)*|[^\sA-Za-z\dऀ-ॿ]")
DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
CHUNK_WORDS = 100  # words per model input; keeps every word well inside BERT's 512-token window


def tokenize(text):
    return WORD_RE.findall(text or "")


def _is_language_independent(tok):
    if tok.startswith(("http", "www.", "@", "#")):
        return True
    if tok[0].isdigit():
        return True
    if not any(ch.isalpha() for ch in tok):
        return True  # punctuation, symbols, emojis
    return False


def _needs_model(tok):
    """Roman-script, language-dependent words are labelled by HingBERT-LID; Devanagari words
    are Hindi by script and language-independent tokens are X, so neither needs the model."""
    return not _is_language_independent(tok) and not DEVANAGARI_RE.match(tok)


def script_of(text):
    dev = sum(1 for ch in text if "ऀ" <= ch <= "ॿ")
    lat = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    if dev + lat == 0:
        return "None"
    r = dev / (dev + lat)
    return "Roman" if r < 0.1 else ("Devanagari" if r > 0.9 else "Mixed")


def normalize_label(label):
    """Maps a HingBERT-LID label to EN / HI / X. The L3Cube-HingLID data uses EN and HI;
    anything else the model may emit (e.g. an OTHER class) is language independent, so it
    must count in u of the CMI and not in the denominator n - u."""
    lab = str(label).upper()
    if lab in ("EN", "ENG", "ENGLISH"):
        return "EN"
    if lab in ("HI", "HIN", "HINDI"):
        return "HI"
    return "X"


def cmi_from_tags(tags):
    """tags: list of 'EN' / 'HI' / 'X' (language independent)."""
    n = len(tags)
    u = sum(1 for t in tags if t == "X")
    if n - u <= 0:
        return 0.0
    w = max(sum(1 for t in tags if t == "EN"), sum(1 for t in tags if t == "HI"))
    return 1.0 - w / (n - u)


def stats_from_tags(tags):
    en, hi = tags.count("EN"), tags.count("HI")
    return {"cmi": cmi_from_tags(tags), "n_lang": en + hi, "en": en, "hi": hi,
            "hi_frac": hi / (en + hi) if en + hi else 0.0}


class Profiler:
    def __init__(self, device=None, batch_size=64, model_name=LID_MODEL):
        self.device = device or os.environ.get("CODEMIX_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name).to(self.device).eval()
        self.id2label = {int(k): normalize_label(v) for k, v in self.model.config.id2label.items()}
        if not {"EN", "HI"} <= set(self.id2label.values()):
            raise ValueError(f"{model_name} has unexpected labels {self.model.config.id2label}")
        self.batch_size = batch_size
        self._memo = {}
        self._lock = threading.Lock()

    @torch.no_grad()
    def _label_chunks(self, chunks):
        """chunks: list of word lists (each at most CHUNK_WORDS long). Returns one label per word."""
        out = []
        for s in range(0, len(chunks), self.batch_size):
            batch = chunks[s:s + self.batch_size]
            with self._lock:
                enc = self.tok(batch, is_split_into_words=True, truncation=True, max_length=512,
                               padding=True, return_tensors="pt")
                logits = self.model(**{k: v.to(self.device) for k, v in enc.items()}).logits
                lab = logits.argmax(-1).cpu().tolist()
            for b, words in enumerate(batch):
                first = {}  # label of the first sub-token of every word
                for pos, wid in enumerate(enc.word_ids(b)):
                    if wid is not None and wid not in first:
                        first[wid] = self.id2label[lab[b][pos]]
                vals = list(first.values())
                # only reachable if a single chunk overflows 512 sub-tokens (pathological input)
                fallback = "HI" if vals.count("HI") > vals.count("EN") else "EN"
                out.append([first.get(k, fallback) for k in range(len(words))])
        return out

    def tag_many(self, texts):
        """Returns, for each text, a list of (word, tag) with tag in {EN, HI, X}."""
        todo, seen = [], set()
        for i, t in enumerate(texts):
            if t not in self._memo and t not in seen:
                seen.add(t)
                todo.append(i)
        words = {i: tokenize(texts[i]) for i in todo}
        chunks, owners = [], []
        for i in todo:
            lw = [w for w in words[i] if _needs_model(w)]
            for s in range(0, len(lw), CHUNK_WORDS):
                chunks.append(lw[s:s + CHUNK_WORDS])
                owners.append(i)
        labels = {i: [] for i in todo}
        for i, lab in zip(owners, self._label_chunks(chunks)):
            labels[i].extend(lab)
        for i in todo:
            it = iter(labels[i])
            tagged = []
            for w in words[i]:
                if _is_language_independent(w):
                    tagged.append((w, "X"))
                elif DEVANAGARI_RE.match(w):
                    tagged.append((w, "HI"))
                else:
                    tagged.append((w, next(it)))
            self._memo[texts[i]] = tagged
        return [self._memo[t] for t in texts]

    def tags(self, text):
        return [t for _, t in self.tag_many([text])[0]]

    def cmi(self, text):
        return cmi_from_tags(self.tags(text))

    def cmi_many(self, texts):
        return [cmi_from_tags([t for _, t in tg]) for tg in self.tag_many(texts)]

    def stats(self, text):
        return stats_from_tags(self.tags(text))

    def stats_many(self, texts):
        return [stats_from_tags([t for _, t in tg]) for tg in self.tag_many(texts)]

    def profile(self, seeker_utterances):
        """Register profile R of a help-seeker, pooled over all their utterances so far
        (a single short reply such as 'yes' would otherwise give an unstable CMI). Every
        utterance is tagged on its own, so long conversations are never truncated."""
        utts = [u for u in seeker_utterances if u and u.strip()]
        tagged = self.tag_many(utts)
        tags = [t for tg in tagged for _, t in tg]
        st = stats_from_tags(tags)
        last = stats_from_tags([t for _, t in tagged[-1]]) if tagged else {"cmi": 0.0}
        return {
            "cmi": round(st["cmi"], 3),
            "cmi_last": round(last["cmi"], 3),
            "hi_frac": round(st["hi_frac"], 3),
            "dominant": "Hindi" if st["hi"] > st["en"] else "English",
            "script": script_of(" ".join(utts)),
        }

    def register_of(self, text):
        """Register of a single text (e.g. a generated response), in the same shape as R."""
        st = self.stats(text)
        return {"cmi": round(st["cmi"], 3), "hi_frac": round(st["hi_frac"], 3), "n_lang": st["n_lang"],
                "dominant": "Hindi" if st["hi"] > st["en"] else "English", "script": script_of(text or "")}


PLAIN_ENGLISH_CMI = 0.05  # below this an English-dominant seeker is treated as monolingual


def is_plain_english(R):
    return R["cmi"] < PLAIN_ENGLISH_CMI and R["dominant"] == "English" and R.get("script") in ("Roman", "None", None)


def describe_register(R):
    """Human-readable register description used inside prompts."""
    if is_plain_english(R):
        mix = "plain English (no Hindi mixing)"
    elif R["dominant"] == "English":
        mix = f"English-dominant Hinglish (about {round(100 * R['hi_frac'])}% Hindi words)"
    else:
        mix = f"Hindi-dominant Hinglish (about {round(100 * R['hi_frac'])}% Hindi words)"
    script = {"Roman": "Roman (Latin) script", "Devanagari": "Devanagari script",
              "Mixed": "a mix of Roman and Devanagari script"}.get(R["script"], "Roman (Latin) script")
    return mix, script
