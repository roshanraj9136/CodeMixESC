"""Build the Hinglish -> English training pairs of the cross-lingual retriever (SPEC: Method 2).

Sources
  PHINC   Srivastava & Singh (2020), 13,738 Hinglish tweets with manual English translations
          (zenodo record 3605597). Read from --phinc (.csv/.tsv/.zip), from data/raw/phinc/, or
          downloaded there through the Zenodo REST API.
  ESConv  ~5,000 seeker utterances of the TRAINING conversations (dataset[100:] minus the dev
          conversations; never the test conversations dataset[:100]) rewritten into Roman-script
          Hinglish by gemini-3.1-flash-lite - a different generator than the one that built the
          ESConv-HiEn test set, so the retriever cannot fit one model's Hinglish style. 25
          utterances per call, Light and Heavy mixing alternating per batch.

Cleaning (proposal): URLs, @mentions, RT markers and HTML entities removed, '#' stripped from
hashtags (the word stays), whitespace collapsed; then pairs with < 3 words on either side,
non-Roman script, Hinglish identical to English (not code-mixed) and duplicates (normalised
Hinglish side) are dropped. stats.json records the count after every step.

Outputs: data/pairs/{train,val}.jsonl with {"hi", "en", "src": "phinc"|"esconv",
"level": "light"|"heavy"|null, "conv": int|null}, and data/pairs/stats.json.
Validation: 500 random PHINC pairs + whole ESConv conversations up to ~200 pairs (seed 42).

Resumable: every finished LLM batch is appended to <out_dir>/esconv_rewrites.jsonl (and the LLM
client caches every call), so an interrupted run continues where it stopped.
--dry_run replaces the LLM by a deterministic fake and writes to scratch/pairs_dry_run.

Usage: python scripts/build_pairs.py --phinc "data/raw/phinc/PHINC.zip"
"""
import argparse
import concurrent.futures as cf
import csv
import hashlib
import html
import importlib.util
import io
import json
import os
import random
import re
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.request
import zipfile
from functools import lru_cache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from codemixesc.esconv import ROOT, dev_ids, load_esconv  # noqa: E402
from codemixesc.retrieval_eval import text_key  # noqa: E402

GEN_MODEL = "gemini-3.1-flash-lite"
PHINC_RECORD = "3605597"
RAW_DIR = os.path.join(ROOT, "data", "raw", "phinc")
OUT_DIR = os.path.join(ROOT, "data", "pairs")
DRY_OUT_DIR = os.path.join(ROOT, "scratch", "pairs_dry_run")
TABLE_EXT = (".csv", ".tsv", ".txt", ".xlsx")
USER_AGENT = "CodeMixESC-build-pairs/1.0 (research; python urllib)"

# ============================================================================ text cleaning
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.I)
RT_MENTION_RE = re.compile(r"(?<!\w)RT\s*@\w+\s*:?")  # "RT @user:" retweet header
MENTION_RE = re.compile(r"(?<![\w@])@\w+:?")          # @user (not e-mail addresses)
RT_RE = re.compile(r"(?<!\w)RT(?!\w)\s*:?")            # a bare retweet marker (uppercase only)
HASHTAG_RE = re.compile(r"#(\w+)")
SPACE_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"\w+(?:'\w+)?")


def clean_text(text):
    """Proposal cleaning: HTML entities decoded, URLs, RT markers and @mentions removed, '#'
    stripped from hashtags (the word stays), whitespace collapsed."""
    t = text or ""
    for _ in range(3):  # tweets are sometimes escaped twice (&amp;amp;)
        u = html.unescape(t)
        if u == t:
            break
        t = u
    t = URL_RE.sub(" ", t)
    t = RT_MENTION_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    t = RT_RE.sub(" ", t)
    t = HASHTAG_RE.sub(r"\1", t)
    return SPACE_RE.sub(" ", t).strip()


def word_count(text):
    return len(WORD_RE.findall(text or ""))


@lru_cache(maxsize=None)
def _is_latin_letter(ch):
    return unicodedata.name(ch, "").startswith("LATIN")


def is_roman(text):
    """Roman script only: at least one letter, and every letter is Latin (no Devanagari, Urdu,
    Gurmukhi, ...). Digits, punctuation and emoji are not letters and are allowed."""
    letters = [ch for ch in (text or "") if ch.isalpha()]
    return bool(letters) and all(_is_latin_letter(ch) for ch in letters)


def clean_pairs(pairs, min_words=3, steps=None, prefix=""):
    """Cleans both sides and applies the filters in order, recording the count after each step.
    pairs: dicts with 'hi' and 'en' (other keys are kept)."""
    steps = {} if steps is None else steps
    out = []
    for p in pairs:
        q = dict(p, hi=clean_text(p["hi"]), en=clean_text(p["en"]))
        if word_count(q["hi"]) >= min_words and word_count(q["en"]) >= min_words:
            out.append(q)
    steps[f"{prefix}cleaned, >= {min_words} words on both sides"] = len(out)
    out = [p for p in out if is_roman(p["hi"]) and is_roman(p["en"])]
    steps[f"{prefix}Roman script on both sides"] = len(out)
    out = [p for p in out if text_key(p["hi"]) != text_key(p["en"])]
    steps[f"{prefix}Hinglish differs from English (code-mixed)"] = len(out)
    seen, uniq = set(), []
    for p in out:
        k = text_key(p["hi"])
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    steps[f"{prefix}deduplicated (normalised Hinglish side)"] = len(uniq)
    return uniq, steps


# ============================================================================ PHINC
class PhincUnavailable(Exception):
    pass


def _sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def _read_source_bytes(path):
    """Raw bytes of a table file, or of the largest table inside a .zip; plus the member name."""
    if path.lower().endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            members = [m for m in z.infolist() if not m.is_dir() and m.filename.lower().endswith(TABLE_EXT)
                       and "__macosx" not in m.filename.lower()]
            if not members:
                raise ValueError(f"{path} contains no .csv/.tsv/.txt/.xlsx file")
            tables = [m for m in members if not m.filename.lower().endswith(".txt")] or members  # not a README
            m = max(tables, key=lambda x: x.file_size)
            return z.read(m), m.filename
    with open(path, "rb") as f:
        return f.read(), os.path.basename(path)


def _decode(raw):
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    raise ValueError("undecodable file")  # unreachable: latin-1 decodes any byte string


def _pick_delimiter(text, name):
    first = text.split("\n", 1)[0]
    counts = {d: first.count(d) for d in (",", "\t", ";", "|")}
    default = "\t" if name.lower().endswith(".tsv") else ","
    best = max(counts, key=lambda d: (counts[d], d == default))
    return best if counts[best] > 0 else default


def read_table(path):
    """Rows (header first) of a .csv/.tsv/.txt/.xlsx file or zip member, and provenance."""
    raw, member = _read_source_bytes(path)
    meta = {"file": os.path.basename(path), "member": member, "sha256": _sha256(raw)}
    if member.lower().endswith(".xlsx"):
        try:
            import openpyxl
        except ImportError as e:
            raise ValueError(f"{member} is an Excel file; install openpyxl or save it as CSV") from e
        ws = openpyxl.load_workbook(io.BytesIO(raw), read_only=True).worksheets[0]
        rows = [["" if c is None else str(c) for c in r] for r in ws.iter_rows(values_only=True)]
        return rows, dict(meta, encoding="xlsx", delimiter=None)
    text, enc = _decode(raw)
    delim = _pick_delimiter(text, member)
    rows = list(csv.reader(io.StringIO(text, newline=""), delimiter=delim))
    return rows, dict(meta, encoding=enc, delimiter=delim)


def _norm_col(name):
    return re.sub(r"[^a-z0-9]+", "_", (name or "").strip().lower()).strip("_")


HI_COLS = ("sentence", "hinglish", "code_mixed", "codemixed", "code_mixed_sentence", "hinglish_sentence",
           "cm_sentence", "source", "src", "tweet", "text")
EN_COLS = ("english_translation", "english", "translation", "english_sentence", "en", "target", "tgt")
INDEX_COLS = ("", "id", "index", "no", "s_no", "sno", "sr_no", "serial", "serial_no")


def _is_index_col(name, values):
    if name in INDEX_COLS or name.startswith("unnamed"):
        return True
    vals = [v.strip() for v in values if v.strip()]
    return bool(vals) and all(v.isdigit() for v in vals)


def pick_columns(header, rows):
    """(hinglish column, english column, how). By name, case-insensitively (PHINC: 'Sentence',
    'English_Translation'); otherwise the first two non-index text columns."""
    names = [_norm_col(h) for h in header]
    en = next((names.index(c) for c in EN_COLS if c in names), None)
    if en is None:
        en = next((i for i, n in enumerate(names) if "english" in n or "transl" in n), None)
    hi = next((names.index(c) for c in HI_COLS if c in names and names.index(c) != en), None)
    if hi is None:
        hi = next((i for i, n in enumerate(names) if i != en and any(
            s in n for s in ("sentence", "hinglish", "mix", "tweet"))), None)
    if hi is not None and en is not None:
        return hi, en, "by name"
    text_cols = [i for i, n in enumerate(names)
                 if not _is_index_col(n, [r[i] for r in rows[:200] if i < len(r)])]
    if hi is None and en is not None:
        hi = next((i for i in text_cols if i != en), None)
        how = "english by name, hinglish = first other text column"
    elif en is None and hi is not None:
        en = next((i for i in text_cols if i != hi), None)
        how = "hinglish by name, english = first other text column"
    else:
        hi, en = (text_cols + [None, None])[:2]
        how = "first two text columns"
    if hi is None or en is None:
        raise ValueError(f"cannot find a Hinglish and an English column in {header}")
    return hi, en, how


def load_phinc(path):
    """[{'hi', 'en'}] of all PHINC rows (uncleaned) and the provenance of the file."""
    rows, meta = read_table(path)
    if len(rows) < 2:
        raise ValueError(f"{path}: no data rows")
    header, body = rows[0], rows[1:]
    hi, en, how = pick_columns(header, body)
    meta.update(columns=[header[hi], header[en]], column_choice=how)
    pairs = [{"hi": r[hi], "en": r[en]} for r in body if len(r) > max(hi, en)]
    meta["rows"] = len(body)
    return pairs, meta


def find_local_phinc(raw_dir=None):
    raw_dir = raw_dir or RAW_DIR
    if not os.path.isdir(raw_dir):
        return None
    files = [os.path.join(raw_dir, f) for f in sorted(os.listdir(raw_dir))
             if f.lower().endswith(TABLE_EXT + (".zip",))]
    return max(files, key=os.path.getsize) if files else None


def _urlopen(url, timeout):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(req, timeout=timeout)


def download_phinc(dest=None, record=PHINC_RECORD, timeout=60):
    """Downloads every file of the Zenodo record into dest (checksums verified when given)."""
    dest = dest or RAW_DIR
    api = f"https://zenodo.org/api/records/{record}"
    try:
        with _urlopen(api, timeout) as r:
            meta = json.load(r)
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise PhincUnavailable(f"cannot reach the Zenodo API ({api}): {e}") from e
    files = meta.get("files") or []
    if isinstance(files, dict):  # InvenioRDM native serialisation
        files = list((files.get("entries") or {}).values())
    if not files:
        raise PhincUnavailable(f"Zenodo record {record} lists no files")
    os.makedirs(dest, exist_ok=True)
    paths = []
    for f in files:
        key = f.get("key") or f.get("filename")
        links = f.get("links") or {}
        url = links.get("content") or links.get("download") or links.get("self") \
            or f"https://zenodo.org/records/{record}/files/{urllib.request.quote(key)}?download=1"
        out = os.path.join(dest, os.path.basename(key))
        print(f"[phinc] downloading {key} ...", flush=True)
        try:
            with _urlopen(url, timeout) as r:
                raw = r.read()
        except (urllib.error.URLError, OSError) as e:
            raise PhincUnavailable(f"download of {url} failed: {e}") from e
        algo, _, digest = (f.get("checksum") or "").partition(":")
        if digest and algo in hashlib.algorithms_available and hashlib.new(algo, raw).hexdigest() != digest:
            raise PhincUnavailable(f"checksum mismatch for {key}")
        with open(out + ".part", "wb") as fh:
            fh.write(raw)
        os.replace(out + ".part", out)
        paths.append(out)
    return paths


def resolve_phinc(args):
    """Path of the PHINC table: --phinc, else data/raw/phinc/, else a Zenodo download."""
    if args.phinc:
        if not os.path.exists(args.phinc):
            raise PhincUnavailable(f"--phinc {args.phinc} does not exist")
        return args.phinc
    local = find_local_phinc()
    if local:
        return local
    if args.no_download:
        raise PhincUnavailable(f"no PHINC file in {RAW_DIR} and --no_download is set")
    download_phinc()
    local = find_local_phinc()
    if not local:
        raise PhincUnavailable(f"the Zenodo record has no .csv/.tsv/.zip table (see {RAW_DIR})")
    return local


# ============================================================================ ESConv rewriting
PAIR_PROMPT = """Rewrite each message below into Roman-script Hinglish, the way young Indians naturally type in chat. The messages were written by people seeking emotional support. Use a {level} Hindi-English mix.

Mixing level: {level_desc}

Rules:
- Roman (Latin) script only, never Devanagari.
- Rewrite every message on its own and keep its id. Return exactly {n} items with ids {ids}.
- Keep the meaning, emotion, tone and every detail. Do not add, drop, answer or comment on anything.
- Keep names, numbers and places unchanged. Use natural chat spellings (bahut, kya, hai, nahi, yaar, accha).
{retry_rule}
Messages:
{items}

Return JSON of the form {{"items": [{{"id": 0, "text": "..."}}, ...]}}."""

RETRY_RULE = ("- A previous attempt at these messages broke the rules; the \"problem\" field says what went wrong. "
              "Fix it in the new version.\n")
ATTEMPT_SUFFIX = "\n\n(Attempt {attempt}: rewrite every listed message following all the rules.)"
PROBLEM_HINTS = {
    "missing": "it was missing from the answer",
    "empty": "the rewrite was empty",
    "not Roman script": "it used a non-Latin script; write Hindi words in Roman letters only",
    "identical": "it was returned unchanged in English; mix in Hindi as described",
    "length": "content was lost or added; keep the full meaning and nothing more",
    "no Hindi": "it contained no Hindi words; mix in Hindi as described",
}


def _load_level_desc():
    """LEVEL_DESC of scripts/build_hien.py, so pair and test-set rewriting share the definitions."""
    path = os.path.join(ROOT, "scripts", "build_hien.py")
    spec = importlib.util.spec_from_file_location("build_hien", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.LEVEL_DESC


def esconv_candidates(data, exclude, min_words=4):
    """Seeker utterances of the training conversations (dataset[100:] minus `exclude`),
    with >= min_words words, deduplicated; plus the count after each step."""
    steps, items = {}, []
    for ci in range(100, len(data)):  # dataset[:100] is the test set and is never used for training
        if ci in exclude:
            continue
        for t in data[ci]["dialog"]:
            if t["speaker"] == "seeker":
                items.append({"conv": ci, "text": clean_text(t["content"])})
    steps["seeker utterances of training conversations (test and dev excluded)"] = len(items)
    items = [it for it in items if word_count(it["text"]) >= min_words]
    steps[f">= {min_words} words"] = len(items)
    seen, uniq = set(), []
    for it in items:
        k = text_key(it["text"])
        if k not in seen:
            seen.add(k)
            uniq.append(it)
    steps["deduplicated"] = len(uniq)
    return uniq, steps


def build_prompt(level, level_desc, texts, problems=None, attempt=1):
    items = []
    for i, t in enumerate(texts):
        d = {"id": i, "text": t}
        if problems:
            d["problem"] = problems[i]
        items.append(d)
    prompt = PAIR_PROMPT.format(level=level, level_desc=level_desc, n=len(texts),
                                ids=f"0..{len(texts) - 1}", retry_rule=RETRY_RULE if problems else "",
                                items=json.dumps(items, ensure_ascii=False, indent=0))
    return prompt + (ATTEMPT_SUFFIX.format(attempt=attempt) if attempt > 1 else "")


def parse_items(text):
    """{id: text} from the model's JSON answer ({"items": [...]}, a bare list, or fenced JSON)."""
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        obj = json.loads(t)
    except ValueError:
        m = re.search(r"[\[{].*[\]}]", t, re.S)
        if not m:
            raise ValueError("no JSON in the answer")
        obj = json.loads(m.group(0))
    if isinstance(obj, dict):
        obj = obj.get("items") or obj.get("turns") or obj.get("messages") or \
            next((v for v in obj.values() if isinstance(v, list)), None)
    if not isinstance(obj, list):
        raise ValueError("no list of items in the answer")
    out = {}
    for it in obj:
        if isinstance(it, dict) and "id" in it:
            txt = it.get("text") or it.get("hinglish") or it.get("rewrite") or ""
            try:
                out[int(it["id"])] = str(txt)
            except (TypeError, ValueError):
                continue
    return out


def validate_rewrite(src, hi, profiler=None, ratio=(0.4, 3.0)):
    """(ok, problem). Roman script, non-empty, not identical to the English source, a sane
    length (a merged or truncated item is not a translation) and, with a profiler, some Hindi."""
    hi = clean_text(hi)
    if not hi:
        return False, "empty"
    if not is_roman(hi):
        return False, "not Roman script"
    if text_key(hi) == text_key(src):
        return False, "identical"
    r = word_count(hi) / max(1, word_count(src))
    if not ratio[0] <= r <= ratio[1]:
        return False, "length"
    if profiler is not None and profiler.stats(hi)["hi"] == 0:
        return False, "no Hindi"
    return True, ""


class Progress:
    """Append-only JSONL of finished LLM batches keyed by a hash of (model, temperature, prompt),
    so a changed sample, prompt or model never picks up a stale answer. The raw answer is
    stored, so changes to the validation apply on the next run."""

    def __init__(self, path):
        self.path, self.lock, self.done = path, threading.Lock(), {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        self.done[rec["key"]] = rec["answer"]
                    except (ValueError, KeyError):
                        continue  # a torn last line of an interrupted run

    @staticmethod
    def key(model, temperature, prompt):
        return hashlib.sha256(json.dumps([model, temperature, prompt], ensure_ascii=False).encode("utf-8")).hexdigest()

    def get(self, key):
        return self.done.get(key)

    def put(self, key, answer, meta):
        with self.lock:
            self.done[key] = answer
            with open(self.path, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps({"key": key, "answer": answer, **meta}, ensure_ascii=False) + "\n")


class Rewriter:
    """Runs the batched rewriting with one retry round for the failed items."""

    def __init__(self, llm, level_desc, progress, model_name, temperature=0.7, max_tokens=8192,
                 workers=4, max_calls=0, profiler=None):
        self.llm, self.level_desc, self.progress = llm, level_desc, progress
        self.model_name, self.temperature, self.max_tokens = model_name, temperature, max_tokens
        self.workers, self.max_calls, self.profiler = workers, max_calls, profiler
        self.lock = threading.Lock()
        self.calls = {"real": 0, "cached": 0, "from_progress": 0, "errors": 0, "skipped_budget": 0}

    def _ask(self, prompt, tag):
        """The answer to one prompt, or None when the call budget is used up or the call failed
        (the batch then stays pending, so a network outage never silently shrinks the data)."""
        key = Progress.key(self.model_name, self.temperature, prompt)
        hit = self.progress.get(key)
        if hit is not None:
            with self.lock:
                self.calls["from_progress"] += 1
            return hit
        with self.lock:
            if self.max_calls and self.calls["real"] >= self.max_calls:
                self.calls["skipped_budget"] += 1
                return None
        try:
            text, meta = self.llm.chat([{"role": "user", "content": prompt}], temperature=self.temperature,
                                       max_tokens=self.max_tokens, json_mode=True, tag=tag)
        except Exception as e:  # noqa: BLE001 - the client already retried; leave the batch for a rerun
            print(f"[pairs] {tag}: LLM error {e.__class__.__name__}: {str(e)[:200]}", flush=True)
            with self.lock:
                self.calls["errors"] += 1
            return None
        with self.lock:
            self.calls["cached" if meta.get("cached") else "real"] += 1
        self.progress.put(key, text, {"tag": tag, "t": round(time.time(), 1)})
        return text

    def _batch(self, level, items, attempt, problems=None):
        """items: [(sample index, source text)] -> ({index: hinglish}, {index: problem}, done?)."""
        prompt = build_prompt(level, self.level_desc[level], [t for _, t in items],
                              [PROBLEM_HINTS[p] for p in problems] if problems else None, attempt)
        answer = self._ask(prompt, tag=f"pairs-{level}-a{attempt}")
        if answer is None:
            return {}, {}, False
        try:
            got = parse_items(answer)
        except ValueError:
            got = {}
        ok, bad = {}, {}
        for j, (idx, src) in enumerate(items):
            if j not in got:
                bad[idx] = "missing"
                continue
            good, problem = validate_rewrite(src, got[j], self.profiler)
            if good:
                ok[idx] = clean_text(got[j])
            else:
                bad[idx] = problem
        return ok, bad, True

    def _run(self, jobs, attempt):
        results, failures, pending = {}, {}, 0
        with cf.ThreadPoolExecutor(max(1, self.workers)) as ex:
            futs = [ex.submit(self._batch, level, items, attempt, probs) for level, items, probs in jobs]
            for n, f in enumerate(cf.as_completed(futs), 1):
                ok, bad, done = f.result()
                results.update(ok)
                failures.update(bad)
                pending += not done
                if n % 20 == 0 or n == len(futs):
                    print(f"[pairs] attempt {attempt}: {n}/{len(futs)} batches", flush=True)
        return results, failures, pending

    def run(self, sample, batch_size):
        """sample: [{'conv', 'text'}]. Returns ({index: (hinglish, level)}, report)."""
        levels = {}
        jobs = []
        for b, s in enumerate(range(0, len(sample), batch_size)):
            level = ("light", "heavy")[b % 2]
            items = [(i, sample[i]["text"]) for i in range(s, min(s + batch_size, len(sample)))]
            for i, _ in items:
                levels[i] = level
            jobs.append((level, items, None))
        ok1, bad1, pend1 = self._run(jobs, attempt=1)
        retry_jobs = []
        for level in ("light", "heavy"):
            failed = [i for i in sorted(bad1) if levels[i] == level]
            for s in range(0, len(failed), batch_size):
                chunk = failed[s:s + batch_size]
                retry_jobs.append((level, [(i, sample[i]["text"]) for i in chunk], [bad1[i] for i in chunk]))
        ok2, bad2, pend2 = self._run(retry_jobs, attempt=2) if (retry_jobs and not pend1) else ({}, {}, 0)
        ok = {**ok1, **ok2}
        reasons1, reasons2 = {}, {}
        for p in bad1.values():
            reasons1[p] = reasons1.get(p, 0) + 1
        for p in bad2.values():
            reasons2[p] = reasons2.get(p, 0) + 1
        report = {"batches": len(jobs), "retry_batches": len(retry_jobs) if not pend1 else 0,
                  "valid_first_attempt": len(ok1), "failed_first_attempt": len(bad1),
                  "failure_reasons_first_attempt": reasons1, "valid_after_retry": len(ok2),
                  "failed_after_retry": len(bad2), "failure_reasons_after_retry": reasons2,
                  "pending_batches": pend1 + pend2, "calls": dict(self.calls)}
        return {i: (h, levels[i]) for i, h in ok.items()}, report


# ============================================================================ dry run stand-ins
class FakePairLLM:
    """Deterministic stand-in for LLM in --dry_run: answers the pair prompt in its JSON format with
    pseudo-Hinglish (codemixesc.testing.fake_hinglish) and fails on some items on purpose
    (unchanged English, Devanagari, missing ids, a non-JSON answer) so the validation and the
    retry path are exercised. Never used for real data."""

    def __init__(self):
        self.n_calls = 0

    def __call__(self, prompt, system=None, **kw):
        return self.chat([{"role": "user", "content": prompt}], system=system, **kw)[0]

    def chat(self, messages, system=None, temperature=0.0, max_tokens=400, tag="", json_mode=False):
        from codemixesc.testing import fake_hinglish
        self.n_calls += 1
        prompt = messages[0]["content"]
        level = "heavy" if "Use a heavy" in prompt else "light"
        retry = "(Attempt " in prompt
        items = json.loads(re.search(r"Messages:\n(\[.*\])\n\nReturn JSON", prompt, re.S).group(1))
        h = lambda t, m: int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16) % m  # noqa: E731
        if not retry and h(items[0]["text"], 23) == 0:
            return "Sorry, I cannot help with that.", {"cached": False}
        out = []
        for it in items:
            t = it["text"]
            hi = fake_hinglish(t, level, seed=int(retry))
            if text_key(hi) == text_key(t):
                hi = t + " yaar"
            if h(t, 97) == 0:  # fails at both attempts
                hi = t
            elif not retry:
                if h(t, 11) == 0:
                    hi = t
                elif h(t, 13) == 0:
                    hi = "मुझे " + hi
                elif h(t, 17) == 0:
                    continue
            out.append({"id": it["id"], "text": hi})
        return json.dumps({"items": out}, ensure_ascii=False), {"cached": False}


def write_synthetic_phinc(path, data, n=400, seed=42):
    """A PHINC-shaped CSV (Sentence, English_Translation) built from case-bank supporter turns,
    with tweet noise (URLs, mentions, RT, entities, hashtags) and rows that every cleaning step
    must drop. Only for --dry_run."""
    from codemixesc.testing import fake_hinglish
    rnd = random.Random(seed)
    sup = [t["content"].strip() for c in data[100:] for t in c["dialog"] if t["speaker"] == "supporter"]
    sup = [s for s in sup if 6 <= len(s.split()) <= 30]
    rows = []
    for i, en in enumerate(rnd.sample(sup, n)):
        hi = fake_hinglish(en, "heavy", seed=i)
        hi = hi if text_key(hi) != text_key(en) else hi + " yaar"
        noise = i % 6
        if noise == 1:
            hi = f"RT @user{i}: {hi} https://t.co/x{i}"
        elif noise == 2:
            hi = f"{hi} #bahut &amp; @friend{i}"
        elif noise == 3:
            hi = f"{hi}   &quot;sach&quot;  www.example.com/{i}"
        rows.append((hi, en))
    rows += [("ok", "ok"), ("haan yaar", "yes friend"), ("मैं ठीक हूँ और तुम", "I am fine and you"),
             ("I am fine and you", "I am fine and you"), (rows[0][0], rows[0][1]), ("", "empty Hinglish side")]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Sentence", "English_Translation"])
        w.writerows(rows)
    return path


# ============================================================================ splitting / output
def _val_size(n, total, name, max_share=0.2):
    """The requested validation size, capped at 20% of the source (only matters for tiny inputs)."""
    cap = int(max_share * total)
    if n > cap:
        print(f"[pairs] warning: {name} has only {total} pairs; validation holdout reduced from {n} to {cap}", flush=True)
        return cap
    return n


def holdout_random(pairs, n, seed):
    idx = set(random.Random(seed).sample(range(len(pairs)), min(n, len(pairs))))
    return [p for i, p in enumerate(pairs) if i not in idx], [p for i, p in enumerate(pairs) if i in idx]


def holdout_conversations(pairs, n, seed):
    """Whole conversations go to validation until it holds >= n pairs (utterances of one
    conversation are related, so splitting inside a conversation would leak)."""
    convs = sorted({p["conv"] for p in pairs})
    random.Random(seed).shuffle(convs)
    val_convs, count = set(), 0
    by_conv = {}
    for p in pairs:
        by_conv[p["conv"]] = by_conv.get(p["conv"], 0) + 1
    for c in convs:
        if count >= n:
            break
        val_convs.add(c)
        count += by_conv[c]
    return [p for p in pairs if p["conv"] not in val_convs], [p for p in pairs if p["conv"] in val_convs], sorted(val_convs)


def _portable(path):
    """Repo-relative path for the logs (no local user directories in published stats)."""
    path = os.path.abspath(path)
    return os.path.relpath(path, ROOT).replace(os.sep, "/") if path.startswith(ROOT) else os.path.basename(path)


def write_jsonl(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps({k: r.get(k) for k in ("hi", "en", "src", "level", "conv")}, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--phinc", help="PHINC .csv/.tsv/.zip (default: data/raw/phinc/, else Zenodo download)")
    ap.add_argument("--no_download", action="store_true", help="never contact Zenodo")
    ap.add_argument("--no_phinc", action="store_true", help="ESConv pairs only (deviates from the proposal)")
    ap.add_argument("--out_dir", default=None, help=f"default {OUT_DIR} ({DRY_OUT_DIR} with --dry_run)")
    ap.add_argument("--n_esconv", type=int, default=5000, help="seeker utterances to rewrite (0 = none)")
    ap.add_argument("--batch_size", type=int, default=25, help="utterances per LLM call")
    ap.add_argument("--n_val_phinc", type=int, default=500)
    ap.add_argument("--n_val_esconv", type=int, default=200)
    ap.add_argument("--min_words", type=int, default=3, help="minimum words on both sides of a pair")
    ap.add_argument("--min_seeker_words", type=int, default=4, help="minimum words of an ESConv source utterance")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default=GEN_MODEL)
    ap.add_argument("--temperature", type=float, default=0.7, help="some diversity of spellings and mixing")
    ap.add_argument("--workers", type=int, default=4, help="parallel LLM calls (the client enforces RPM)")
    ap.add_argument("--max_calls", type=int, default=0, help="stop after this many real LLM calls (0 = no cap)")
    ap.add_argument("--check_cmi", action="store_true", help="require >= 1 Hindi word (HingBERT-LID) per rewrite")
    ap.add_argument("--dry_run", action="store_true", help="fake LLM, synthetic PHINC if none given, scratch output")
    args = ap.parse_args(argv)

    out_dir = args.out_dir or (DRY_OUT_DIR if args.dry_run else OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    data = load_esconv()
    try:
        dev = set(dev_ids())
    except FileNotFoundError:
        raise SystemExit("dev_conv_ids.json not found: it is needed to keep the dev conversations out of training")
    stats = {"config": dict(vars(args)), "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    stats["config"]["out_dir"] = _portable(out_dir)

    # ---------------------------------------------------------------- PHINC
    phinc = []
    if not args.no_phinc:
        if args.dry_run and not args.phinc and not find_local_phinc():
            args.phinc = write_synthetic_phinc(os.path.join(out_dir, "phinc_synthetic.csv"), data)
        try:
            path = resolve_phinc(args)
        except PhincUnavailable as e:
            raise SystemExit(f"PHINC is not available: {e}\nDownload the corpus from https://zenodo.org/records/"
                             f"{PHINC_RECORD} (the .csv or .zip), put it into {RAW_DIR} or pass --phinc <file>.")
        raw, meta = load_phinc(path)
        stats["config"]["phinc"] = _portable(path)
        steps = {"rows read": meta["rows"]}
        raw = [p for p in raw if p["hi"].strip() and p["en"].strip()]
        steps["both sides non-empty"] = len(raw)
        phinc, steps = clean_pairs([dict(p, src="phinc", level=None, conv=None) for p in raw], args.min_words, steps)
        stats["phinc"] = {"source": meta, "steps": steps}
        print(f"[pairs] PHINC {meta['member']} ({meta['encoding']}, columns {meta['columns']}): "
              + ", ".join(f"{k}: {v}" for k, v in steps.items()), flush=True)

    # ---------------------------------------------------------------- ESConv
    esconv = []
    if args.n_esconv > 0:
        cands, steps = esconv_candidates(data, dev, args.min_seeker_words)
        sample = random.Random(args.seed).sample(cands, min(args.n_esconv, len(cands)))
        steps["sampled"] = len(sample)
        if args.dry_run:
            llm = FakePairLLM()
        else:
            from codemixesc.llm import LLM
            llm = LLM(args.model, thinking=None)
        profiler = None
        if args.check_cmi:
            if args.dry_run:
                from codemixesc.testing import LexiconProfiler
                profiler = LexiconProfiler()
            else:
                from codemixesc.profiler import Profiler
                profiler = Profiler()
        rw = Rewriter(llm, _load_level_desc(), Progress(os.path.join(out_dir, "esconv_rewrites.jsonl")),
                      args.model if not args.dry_run else "fake", args.temperature,
                      workers=args.workers, max_calls=args.max_calls, profiler=profiler)
        rewrites, report = rw.run(sample, args.batch_size)
        if report["pending_batches"]:
            print(f"[pairs] stopped after {report['calls']['real']} real LLM calls (--max_calls); "
                  f"{report['pending_batches']} batches still to do. Re-run the same command to continue.", flush=True)
            return 3
        steps["valid rewrite (first attempt)"] = report["valid_first_attempt"]
        steps["valid rewrite (after one retry)"] = len(rewrites)
        pairs = [{"hi": h, "en": sample[i]["text"], "src": "esconv", "level": lvl, "conv": sample[i]["conv"]}
                 for i, (h, lvl) in sorted(rewrites.items())]
        esconv, steps = clean_pairs(pairs, args.min_words, steps)
        bad = [p for p in esconv if p["conv"] < 100 or p["conv"] in dev]
        if bad:  # cannot happen by construction; a hard stop if the sampling code is ever changed
            raise AssertionError(f"{len(bad)} ESConv pairs come from test or dev conversations")
        stats["esconv"] = {"steps": steps, "rewriting": report, "model": args.model if not args.dry_run else "fake",
                           "levels": {lv: sum(p["level"] == lv for p in esconv) for lv in ("light", "heavy")}}
        print("[pairs] ESConv: " + ", ".join(f"{k}: {v}" for k, v in steps.items()), flush=True)

    # ---------------------------------------------------------------- union, split, write
    seen = {text_key(p["hi"]) for p in phinc}
    n_before = len(esconv)
    esconv = [p for p in esconv if text_key(p["hi"]) not in seen]
    stats["cross_source_duplicates_removed"] = n_before - len(esconv)
    n_val_ph, n_val_es = _val_size(args.n_val_phinc, len(phinc), "PHINC"), _val_size(args.n_val_esconv, len(esconv), "ESConv")
    stats["val_sizes_requested"] = {"phinc": args.n_val_phinc, "esconv": args.n_val_esconv}
    ph_train, ph_val = holdout_random(phinc, n_val_ph, args.seed)
    es_train, es_val, val_convs = holdout_conversations(esconv, n_val_es, args.seed) if esconv else ([], [], [])
    train, val = ph_train + es_train, ph_val + es_val
    if not train:
        raise SystemExit("no training pairs")
    leak = {text_key(p["hi"]) for p in train} & {text_key(p["hi"]) for p in val}
    stats["leakage_checks"] = {"pairs_from_test_conversations": sum(1 for p in esconv if p["conv"] < 100),
                               "pairs_from_dev_conversations": sum(1 for p in esconv if p["conv"] in dev),
                               "hinglish_in_train_and_val": len(leak)}
    stats["train"] = {"total": len(train), "phinc": len(ph_train), "esconv": len(es_train)}
    stats["val"] = {"total": len(val), "phinc": len(ph_val), "esconv": len(es_val), "esconv_val_conversations": val_convs}
    write_jsonl(os.path.join(out_dir, "train.jsonl"), train)
    write_jsonl(os.path.join(out_dir, "val.jsonl"), val)
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    print(f"[pairs] wrote {len(train)} train / {len(val)} val pairs to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
