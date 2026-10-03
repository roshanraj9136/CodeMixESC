"""Code-Mix Profiler: word-level language identification with HingBERT-LID and the
Code-Mixing Index (CMI) of Gamback and Das (2016).

    CMI = 1 - max_i(w_i) / (n - u)   if n > u,  else 0

w_i = number of words of language i, n = all words, u = language-independent words
(punctuation, numbers, emojis, URLs, @mentions, #hashtags). With two languages the CMI
lies in [0, 0.5]: 0 = monolingual, 0.5 = perfectly balanced mix.
"""
import re
import threading
import unicodedata

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

LID_MODEL = "l3cube-pune/hing-bert-lid"
WORD_RE = re.compile(r"https?://\S+|www\.\S+|[@#]\w+|[ऀ-ॿ]+|[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,]\d+)*|[^\sA-Za-z\dऀ-ॿ]")


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


def script_of(text):
    dev = sum(1 for ch in text if "ऀ" <= ch <= "ॿ")
    lat = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    if dev + lat == 0:
        return "None"
    r = dev / (dev + lat)
    return "Roman" if r < 0.1 else ("Devanagari" if r > 0.9 else "Mixed")


def cmi_from_tags(tags):
    """tags: list of 'EN' / 'HI' / 'X' (language independent)."""
    n = len(tags)
    u = sum(1 for t in tags if t == "X")
    if n - u <= 0:
        return 0.0
    w = max(sum(1 for t in tags if t == "EN"), sum(1 for t in tags if t == "HI"))
    return 1.0 - w / (n - u)


class Profiler:
    def __init__(self, device=None, batch_size=64):
        import os
        self.device = device or os.environ.get("CODEMIX_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(LID_MODEL)
        self.model = AutoModelForTokenClassification.from_pretrained(LID_MODEL).to(self.device).eval()
        self.id2label = self.model.config.id2label
        self.batch_size = batch_size
        self._memo = {}
        self._lock = threading.Lock()

    @torch.no_grad()
    def tag_many(self, texts):
        """Returns, for each text, a list of (word, tag) with tag in {EN, HI, X}."""
        out = [None] * len(texts)
        todo = []
        for i, t in enumerate(texts):
            if t in self._memo:
                out[i] = self._memo[t]
            else:
                todo.append(i)
        for s in range(0, len(todo), self.batch_size):
            idx = todo[s:s + self.batch_size]
            words_list = [tokenize(texts[i]) for i in idx]
            # words the model must label (language-dependent, Roman script)
            lang_words = [[w for w in ws if not _is_language_independent(w) and not re.match(r"[ऀ-ॿ]", w)]
                          for ws in words_list]
            preds = [[] for _ in idx]
            nonempty = [j for j, lw in enumerate(lang_words) if lw]
            if nonempty:
                with self._lock:
                    enc = self.tok([lang_words[j] for j in nonempty], is_split_into_words=True,
                                   truncation=True, max_length=256, padding=True, return_tensors="pt")
                    logits = self.model(**{k: v.to(self.device) for k, v in enc.items()}).logits
                    lab = logits.argmax(-1).cpu().tolist()
                for b, j in enumerate(nonempty):
                    word_ids = enc.word_ids(b)
                    seen, p = set(), {}
                    for pos, wid in enumerate(word_ids):
                        if wid is not None and wid not in seen:
                            seen.add(wid)
                            p[wid] = self.id2label[lab[b][pos]]
                    # words cut by truncation default to the majority label
                    maj = max(set(p.values()), key=list(p.values()).count) if p else "EN"
                    preds[j] = [p.get(k, maj) for k in range(len(lang_words[j]))]
            for j, i in enumerate(idx):
                it = iter(preds[j])
                tagged = []
                for w in words_list[j]:
                    if _is_language_independent(w):
                        tagged.append((w, "X"))
                    elif re.match(r"[ऀ-ॿ]", w):
                        tagged.append((w, "HI"))
                    else:
                        tagged.append((w, next(it)))
                self._memo[texts[i]] = tagged
                out[i] = tagged
        return out

    def cmi(self, text):
        return cmi_from_tags([t for _, t in self.tag_many([text])[0]])

    def cmi_many(self, texts):
        return [cmi_from_tags([t for _, t in tg]) for tg in self.tag_many(texts)]

    def stats(self, text):
        tags = [t for _, t in self.tag_many([text])[0]]
        en, hi = tags.count("EN"), tags.count("HI")
        return {"cmi": cmi_from_tags(tags), "n_lang": en + hi, "en": en, "hi": hi,
                "hi_frac": hi / (en + hi) if en + hi else 0.0}

    def profile(self, seeker_utterances):
        """Register profile R of a help-seeker, pooled over all their utterances so far
        (a single short reply such as 'yes' would otherwise give an unstable CMI)."""
        text = " ".join(u for u in seeker_utterances if u)
        tags = [t for _, t in self.tag_many([text])[0]]
        en, hi = tags.count("EN"), tags.count("HI")
        last = self.stats(seeker_utterances[-1]) if seeker_utterances else {"cmi": 0.0}
        return {
            "cmi": round(cmi_from_tags(tags), 3),
            "cmi_last": round(last["cmi"], 3),
            "hi_frac": round(hi / (en + hi), 3) if en + hi else 0.0,
            "dominant": "Hindi" if hi > en else "English",
            "script": script_of(text),
        }


def describe_register(R):
    """Human-readable register description used inside prompts."""
    if R["cmi"] < 0.05 and R["dominant"] == "English":
        mix = "plain English (no Hindi mixing)"
    elif R["dominant"] == "English":
        mix = f"English-dominant Hinglish (about {round(100 * R['hi_frac'])}% Hindi words)"
    else:
        mix = f"Hindi-dominant Hinglish (about {round(100 * R['hi_frac'])}% Hindi words)"
    script = {"Roman": "Roman (Latin) script", "Devanagari": "Devanagari script",
              "Mixed": "a mix of Roman and Devanagari script"}.get(R["script"], "Roman (Latin) script")
    return mix, script
