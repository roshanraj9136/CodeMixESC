import os
import re
import threading

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

LID_MODEL = "l3cube-pune/hing-bert-lid"
_L = "A-Za-zÀ-ÖØ-öø-ÿ"
WORD_RE = re.compile(rf"https?://\S+|www\.\S+|[@#]\w+|[ऀ-ॿ]+|[{_L}]+(?:['’][{_L}]+)*|\d+(?:[.,]\d+)*|[^\s{_L}\dऀ-ॿ]")
DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
CHUNK_WORDS = 100


def tokenize(text):
    return WORD_RE.findall(text or "")


def _is_language_independent(tok):
    if tok.startswith(("http", "www.", "@", "#")):
        return True
    if tok[0].isdigit():
        return True
    if not any(ch.isalpha() for ch in tok):
        return True
    return False


def _needs_model(tok):
    return not _is_language_independent(tok) and not DEVANAGARI_RE.match(tok)


def script_of(text):
    dev = sum(1 for ch in text if "ऀ" <= ch <= "ॿ")
    lat = sum(1 for ch in text if ch.isalpha() and (ch.isascii() or "À" <= ch <= "ÿ"))
    if dev + lat == 0:
        return "None"
    r = dev / (dev + lat)
    return "Roman" if r < 0.1 else ("Devanagari" if r > 0.9 else "Mixed")


def normalize_label(label):
    lab = str(label).upper()
    if lab in ("EN", "ENG", "ENGLISH"):
        return "EN"
    if lab in ("HI", "HIN", "HINDI"):
        return "HI"
    return "X"


def cmi_from_tags(tags):
    n = len(tags)
    u = sum(1 for t in tags if t == "X")
    if n - u <= 0:
        return 0.0
    w = max(sum(1 for t in tags if t == "EN"), sum(1 for t in tags if t == "HI"))
    return 1.0 - w / (n - u)


def stats_from_tags(tags, words=None):
    en, hi = tags.count("EN"), tags.count("HI")
    out = {"cmi": cmi_from_tags(tags), "n_lang": en + hi, "en": en, "hi": hi,
           "hi_frac": hi / (en + hi) if en + hi else 0.0}
    if words is not None:
        out["hi_strong"] = sum(1 for w, t in zip(words, tags) if t == "HI" and w.lower() not in AMBIGUOUS)
    return out


AMBIGUOUS = frozenset("""
to do me he see ve re hi us use lol yea ah man haha hahaha ya hmm main key ha lay per th aa tho k sun g sake
pre bag covid i a
""".split())


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
        out = []
        for s in range(0, len(chunks), self.batch_size):
            batch_words = chunks[s:s + self.batch_size]

            batch = [[w.lower().replace("'", "").replace("’", "") for w in ws] for ws in batch_words]
            with self._lock:
                enc = self.tok(batch, is_split_into_words=True, truncation=True, max_length=512,
                               padding=True, return_tensors="pt")
                logits = self.model(**{k: v.to(self.device) for k, v in enc.items()}).logits
                lab = logits.argmax(-1).cpu().tolist()
            for b, words in enumerate(batch_words):
                first = {}
                for pos, wid in enumerate(enc.word_ids(b)):
                    if wid is not None and wid not in first:
                        first[wid] = self.id2label[lab[b][pos]]
                vals = list(first.values())

                fallback = "HI" if vals.count("HI") > vals.count("EN") else "EN"
                out.append([first.get(k, fallback) for k in range(len(words))])
        return out

    def tag_many(self, texts):
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
        tagged = self.tag_many([text])[0]
        return stats_from_tags([t for _, t in tagged], [w for w, _ in tagged])

    def stats_many(self, texts):
        return [stats_from_tags([t for _, t in tg], [w for w, _ in tg]) for tg in self.tag_many(texts)]

    def profile(self, seeker_utterances):
        utts = [u for u in seeker_utterances if u and u.strip()]
        tagged = self.tag_many(utts)
        st = stats_from_tags([t for tg in tagged for _, t in tg], [w for tg in tagged for w, _ in tg])
        last = stats_from_tags([t for _, t in tagged[-1]]) if tagged else {"cmi": 0.0}
        return {
            "cmi": round(st["cmi"], 3),
            "cmi_last": round(last["cmi"], 3),
            "hi_frac": round(st["hi_frac"], 3),
            "dominant": "Hindi" if st["hi"] > st["en"] else "English",
            "script": script_of(" ".join(utts)),
            "n_lang": st["n_lang"],
            "n_hi_strong": st["hi_strong"],
        }

    def register_of(self, text):
        st = self.stats(text)
        return {"cmi": round(st["cmi"], 3), "hi_frac": round(st["hi_frac"], 3), "n_lang": st["n_lang"],
                "n_hi_strong": st["hi_strong"], "dominant": "Hindi" if st["hi"] > st["en"] else "English",
                "script": script_of(text or "")}


PLAIN_ENGLISH_CMI = 0.05
MIN_STRONG_HINDI = 2
MIN_STRONG_FRAC = 0.05


def is_plain_english(R):
    if R["dominant"] != "English" or R.get("script") not in ("Roman", "None", None):
        return False
    if R["cmi"] < PLAIN_ENGLISH_CMI:
        return True
    if "n_hi_strong" in R:
        return R["n_hi_strong"] < MIN_STRONG_HINDI or R["n_hi_strong"] < MIN_STRONG_FRAC * max(1, R["n_lang"])
    return False


def target_cmi(R):
    return 0.0 if is_plain_english(R) else R["cmi"]


def target_hi_frac(R):
    return 0.0 if is_plain_english(R) else R["hi_frac"]


def hindi_share(R, reg):
    if is_plain_english(R) and reg.get("n_lang") and "n_hi_strong" in reg:
        return reg["n_hi_strong"] / reg["n_lang"]
    return reg["hi_frac"]


def register_distance(R, reg, metric="hi_frac"):
    h = hindi_share(R, reg)
    if metric == "cmi":
        return min(h, 1.0 - h) if is_plain_english(R) else abs(reg["cmi"] - R["cmi"])
    return abs(h - target_hi_frac(R))


def describe_register(R):
    if is_plain_english(R):
        mix = "plain English (no Hindi mixing)"
    elif R["dominant"] == "English":
        mix = f"English-dominant Hinglish (about {round(100 * R['hi_frac'])}% Hindi words)"
    else:
        mix = f"Hindi-dominant Hinglish (about {round(100 * R['hi_frac'])}% Hindi words)"
    script = {"Roman": "Roman (Latin) script", "Devanagari": "Devanagari script",
              "Mixed": "a mix of Roman and Devanagari script"}.get(R["script"], "Roman (Latin) script")
    return mix, script
