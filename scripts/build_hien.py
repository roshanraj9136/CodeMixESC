"""Build ESConv-HiEn: the parallel Roman-script Hinglish version of the ESConv test set
(Light and Heavy mixing), plus the small development set.

Step 1  one LLM call per conversation rewrites every turn (names/spellings stay consistent,
        strategy labels are copied from the source turns).
Step 2  the CMI of every utterance is measured with HingBERT-LID; utterances outside the
        target band (and rewrites whose meaning drifted, by LaBSE similarity) are regenerated.
Step 3  (separate script quality_check.py) a 20% sample is rated for naturalness and meaning.

Usage: python scripts/build_hien.py --split test   (or --split dev)
"""
import argparse
import concurrent.futures as cf
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from codemixesc.esconv import hien_dir, load_esconv, split  # noqa: E402
from codemixesc.llm import LLM  # noqa: E402
from codemixesc.profiler import Profiler, script_of  # noqa: E402

BANDS = {"light": (0.10, 0.30), "heavy": (0.30, 0.50)}
MIN_WORDS_FOR_BAND = 5  # a 1-4 word utterance ("ok thanks") cannot hit a CMI band reliably
MIN_SIM = 0.55          # LaBSE similarity below this (and clearly below the conversation's median) = meaning drift
DRIFT_MARGIN = 0.15     # ... "clearly below": by this much, so the check calibrates itself per conversation
EPS = 1e-9              # 1/10 minority words gives CMI 0.09999999999999998, which must count as 0.10
GEN_MODEL = "gemini-3.5-flash-lite"

LEVEL_DESC = {
    "light": ("LIGHT mixing (target Code-Mixing Index 0.1-0.3): English stays the main language (70-90% of the "
              "words). Insert Hindi words and short Hindi phrases into the English sentences - fillers, emotion words "
              "and short phrases such as 'yaar', 'bahut', 'sach mein', 'kya karun', 'tension ho rahi hai', 'theek hai', "
              "'pata nahi'. Do not turn whole sentences into Hindi. Example: 'I am so stressed yaar, pata nahi how I will "
              "tell my parents about the result.'"),
    "heavy": ("HEAVY mixing (target Code-Mixing Index 0.3-0.5): about half of the words in English and half in "
              "Hindi. Switch language inside sentences: Hindi sentence frames carrying many English nouns, verbs, "
              "adjectives and short English clauses (feel, stress, job, exams, family, confident, honestly, I don't "
              "know, it's okay). Aim for 35-50% English words - more English than in everyday Hindi. Example: "
              "'Honestly mujhe feel ho raha hai ki I am not good enough, exams ka itna pressure hai and family "
              "bhi expect karti hai.'"),
}

REWRITE_PROMPT = """Rewrite this emotional support conversation into Roman-script Hinglish as it is naturally typed in chat by young Indians. Keep the meaning, emotions, names, turn order and the strategy of every supporter turn unchanged, and use a {level} Hindi-English mix.

Mixing level: {level_desc}

Rules:
- Roman (Latin) script only, never Devanagari.
- Rewrite every turn and return exactly {n} turns with the same ids. Do not merge, split, add or drop turns.
- Keep each turn's meaning and emotional tone; do not add or remove information.
- Supporter turns must keep their support strategy (shown as "strategy").
- Keep names, numbers and places unchanged. Use natural, consistent chat spellings (bahut, kya, hai, nahi, yaar, accha).
- "seeker" is the person looking for support, "supporter" is the helper.

Conversation:
{conv_json}

Return JSON of the form {{"turns": [{{"id": 0, "text": "..."}}, ...]}}."""

FIX_PROMPT = """You are revising a Roman-script Hinglish rewrite of an emotional support conversation.
Target: {level_desc}

Some rewritten turns do not meet the target. For each one you get the English original, the current Hinglish version and what is wrong. Rewrite each listed turn again so that it meets the target, keeps the original meaning, emotion and support strategy, stays in Roman script, and sounds like natural chat. Keep the same names and spellings as the rest of the conversation.

Conversation so far (for consistency of names and spellings):
{conv_text}

Turns to revise:
{items}

Return JSON of the form {{"turns": [{{"id": <id>, "text": "..."}}, ...]}} containing only the listed turns."""


def parse_turns(text):
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object")
    obj = json.loads(m.group(0))
    return {int(t["id"]): t["text"].strip() for t in obj["turns"]}


def band_distance(stat, level):
    """How far an utterance is from its level. Light is measured on the Hindi share (CMI 0.1-0.3
    with English as the main language, as LEVEL_DESC asks; CMI alone is symmetric and would accept
    a 77%-Hindi turn), Heavy on the CMI (0.3-0.5, either language may lead)."""
    if stat["n_lang"] < MIN_WORDS_FOR_BAND:
        return 0.0
    lo, hi = BANDS[level]
    x = stat["hi_frac"] if level == "light" else stat["cmi"]
    return max(0.0, lo - x - EPS, x - hi - EPS)


def drift_threshold(sims):
    """Similarity below which a turn of this conversation counts as meaning drift."""
    srt = sorted(sims)
    median = srt[len(srt) // 2] if srt else 1.0
    return min(MIN_SIM, median - DRIFT_MARGIN)


def diagnose(stat, level, sim=None, sim_threshold=None):
    probs = []
    if stat["n_lang"] >= MIN_WORDS_FOR_BAND:
        hf = round(100 * stat["hi_frac"])
        if band_distance(stat, level) > 0:
            if level == "light" and stat["hi_frac"] > 0.5:
                probs.append(f"Hindi became the main language ({hf}% Hindi words); keep English as the main "
                             "language with 10-30% Hindi words")
            elif level == "light" and stat["hi_frac"] > BANDS["light"][1]:
                probs.append(f"too much Hindi ({hf}% Hindi words); keep English as the clear main language "
                             "with 10-30% Hindi")
            elif level == "light":
                probs.append(f"too little mixing: only {hf}% Hindi words; use 10-30% Hindi words")
            else:
                probs.append(f"not mixed enough: {hf}% Hindi words; mix Hindi and English roughly evenly "
                             "(English 35-50%)")
        if sim is not None and sim_threshold is not None and sim < sim_threshold:
            probs.append("the meaning drifted from the English original; keep its exact meaning")
    return probs


def badness(stat, script, level, sim, sim_threshold):
    """Lexicographic: wrong script, then distance from the band, then meaning drift."""
    drift = max(0.0, sim_threshold - sim) if stat["n_lang"] >= MIN_WORDS_FOR_BAND else 0.0
    return (int(script not in ("Roman", "None")), round(band_distance(stat, level), 6), round(drift, 6))


class Builder:
    def __init__(self, model=GEN_MODEL):
        from sentence_transformers import SentenceTransformer
        from codemixesc.retriever import device
        self.llm = LLM(model, thinking=None)
        self.prof = Profiler()
        self.labse = SentenceTransformer("sentence-transformers/LaBSE", device=device())  # honours CODEMIX_DEVICE
        import threading
        self.lock = threading.Lock()

    def sims(self, en, hi):
        with self.lock:
            return self._sims(en, hi)

    def _sims(self, en, hi):
        a = self.labse.encode(en, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
        b = self.labse.encode(hi, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
        return (a * b).sum(1).astype(float).tolist() if len(en) else []

    def rewrite(self, conv, level, extra=""):
        """extra: reviewer feedback appended to the prompt (quality_check.py); empty = original prompt."""
        dialog = conv["dialog"]
        items = []
        for i, t in enumerate(dialog):
            d = {"id": i, "speaker": t["speaker"]}
            if t["speaker"] == "supporter":
                d["strategy"] = t["annotation"].get("strategy")
            d["text"] = t["content"].strip()
            items.append(d)
        prompt = REWRITE_PROMPT.format(level=level, level_desc=LEVEL_DESC[level], n=len(items),
                                       conv_json=json.dumps(items, ensure_ascii=False, indent=0))
        for attempt in range(4):
            suffix = "" if attempt == 0 else f"\n\n(Attempt {attempt + 1}: return exactly {len(items)} turns, ids 0..{len(items) - 1}.)"
            out = self.llm(prompt + extra + suffix, max_tokens=16000, json_mode=True, temperature=0.4, tag=f"hien-rewrite-{level}")
            try:
                turns = parse_turns(out)
                if sorted(turns) == list(range(len(items))) and all(turns.values()):
                    return [turns[i] for i in range(len(items))]
            except Exception:
                pass
        raise RuntimeError("rewrite failed")

    def build_conv(self, conv, level, max_rounds=3, extra=""):
        en = [t["content"].strip() for t in conv["dialog"]]
        hi = self.rewrite(conv, level, extra)
        regen = [0] * len(hi)
        for rnd in range(max_rounds + 1):
            stats = [self.prof.stats(h) for h in hi]
            sims = self.sims(en, hi)
            thr = drift_threshold([sm for st, sm in zip(stats, sims) if st["n_lang"] >= MIN_WORDS_FOR_BAND])
            bad = {}
            for i, (st, sm) in enumerate(zip(stats, sims)):
                p = diagnose(st, level, sm, thr)
                if script_of(hi[i]) not in ("Roman", "None"):
                    p.append("use Roman script only")
                if p:
                    bad[i] = p
            if not bad or rnd == max_rounds:
                break
            conv_text = "\n".join(f"[{i}] {conv['dialog'][i]['speaker']}: {h}" for i, h in enumerate(hi))
            items = "\n\n".join(
                f"id {i} ({conv['dialog'][i]['speaker']}"
                + (f", strategy {conv['dialog'][i]['annotation'].get('strategy')}" if conv['dialog'][i]['speaker'] == 'supporter' else "")
                + f")\nEnglish: {en[i]}\nCurrent: {hi[i]}\nProblem: {'; '.join(p)}"
                for i, p in bad.items())
            # from round 2 on the prompt must differ, or the cache would return round 1's answer again
            revision = f"\n\n(Revision round {rnd + 1}.)" if rnd else ""
            out = self.llm(FIX_PROMPT.format(level_desc=LEVEL_DESC[level], conv_text=conv_text, items=items) + revision,
                           max_tokens=8000, json_mode=True, temperature=0.4, tag=f"hien-fix-{level}")
            try:
                fixed = parse_turns(out)
            except Exception:
                continue
            cand = {i: txt for i, txt in fixed.items() if i in bad and txt and txt != hi[i]}
            new_sims = dict(zip(cand, self.sims([en[i] for i in cand], list(cand.values())))) if cand else {}
            for i, txt in cand.items():
                new = self.prof.stats(txt)
                # keep the new version only if it is better and does not lose the meaning
                if new_sims[i] >= min(sims[i], MIN_SIM) and \
                        badness(new, script_of(txt), level, new_sims[i], thr) < badness(stats[i], script_of(hi[i]), level, sims[i], thr):
                    hi[i] = txt
                    regen[i] += 1
        stats = [self.prof.stats(h) for h in hi]
        sims = self.sims(en, hi)
        thr = drift_threshold([sm for st, sm in zip(stats, sims) if st["n_lang"] >= MIN_WORDS_FOR_BAND])
        out = json.loads(json.dumps(conv))
        for i, t in enumerate(out["dialog"]):
            t["content_en"] = t["content"]
            t["content"] = hi[i]
            t["cmi"] = round(stats[i]["cmi"], 4)
            t["hi_frac"] = round(stats[i]["hi_frac"], 4)
            t["n_lang_words"] = stats[i]["n_lang"]
            t["script"] = script_of(hi[i])
            t["labse_sim"] = round(sims[i], 4)
            t["band_checked"] = stats[i]["n_lang"] >= MIN_WORDS_FOR_BAND
            t["in_band"] = ((not t["band_checked"]) or band_distance(stats[i], level) == 0) and \
                t["script"] in ("Roman", "None")
            t["meaning_ok"] = (not t["band_checked"]) or sims[i] >= thr
            t["regenerated"] = regen[i]
        out["mix_level"] = level
        out["drift_threshold"] = round(thr, 4)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["test", "dev"], default="test")
    ap.add_argument("--levels", default="light,heavy")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out_suffix", default="")
    args = ap.parse_args()
    out_dir = hien_dir()
    os.makedirs(out_dir, exist_ok=True)
    data = load_esconv()
    test, bank = split(data)
    if args.split == "test":
        convs = list(enumerate(test))
    else:
        rnd = random.Random(7)
        dev_ids = sorted(rnd.sample(range(100, len(data)), 12))
        json.dump(dev_ids, open(os.path.join(out_dir, "dev_conv_ids.json"), "w"))
        convs = [(i, data[i]) for i in dev_ids]
    if args.limit:
        convs = convs[:args.limit]
    b = Builder()
    for level in args.levels.split(","):
        results = [None] * len(convs)

        def job(k):
            ci, conv = convs[k]
            try:
                r = b.build_conv(conv, level)
            except Exception as e:  # one bad conversation must not discard the others
                return k, {"esconv_index": ci, "error": repr(e)[:500]}
            r["esconv_index"] = ci
            return k, r

        with cf.ThreadPoolExecutor(args.workers) as ex:
            futs = [ex.submit(job, k) for k in range(len(convs))]
            for n, f in enumerate(cf.as_completed(futs), 1):
                k, r = f.result()
                results[k] = r
                if n % 10 == 0:
                    print(f"[{level}] {n}/{len(convs)} conversations done", flush=True)
        failed = [r["esconv_index"] for r in results if "error" in r]
        path = os.path.join(out_dir, f"{args.split}_{level}{args.out_suffix}.json")
        if failed:  # never write a test file with holes; re-run to retry (finished calls are cached)
            path = path.replace(".json", ".partial.json")
            print(f"[{level}] {len(failed)} conversations failed: {failed}; re-run to retry", flush=True)
        json.dump(results, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        done = [c for c in results if "error" not in c]
        checked = [t for c in done for t in c["dialog"] if t["band_checked"]]
        inb = sum(t["in_band"] for t in checked) / max(1, len(checked))
        mok = sum(t["meaning_ok"] for t in checked) / max(1, len(checked))
        mean_cmi = sum(t["cmi"] for t in checked) / max(1, len(checked))
        sims = sorted(t["labse_sim"] for t in checked)
        q = (lambda f: sims[int(f * (len(sims) - 1))]) if sims else (lambda f: float("nan"))
        print(f"[{level}] saved {path}: {len(checked)} band-checked utterances, {100 * inb:.1f}% in band, "
              f"{100 * mok:.1f}% meaning ok, mean CMI {mean_cmi:.3f}, LaBSE sim p10/p50/p90 "
              f"{q(0.1):.2f}/{q(0.5):.2f}/{q(0.9):.2f}", flush=True)


if __name__ == "__main__":
    main()
