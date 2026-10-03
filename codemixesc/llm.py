"""Thin, cached, rate-limited client for the Gemini API (hosted Gemma / Gemini models).

Every call is cached on disk (SQLite), so re-running an experiment or an ablation that
shares a prefix of the pipeline costs nothing. Every real call is also logged to a JSONL
file so the number of LLM calls and the latency per turn can be reported.
"""
import hashlib
import json
import os
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PATH = os.path.join(ROOT, "cache", "llm_cache.sqlite")
LOG_PATH = os.path.join(ROOT, "cache", "llm_calls.jsonl")
RL_PATH = os.path.join(ROOT, "cache", "ratelimit.sqlite")

# Free-tier limits read from AI Studio (2026-10-04): Gemma 4 = 30 RPM / 16K input TPM /
# 14.4K RPD; Flash-Lite = 15 RPM / 250K TPM / 500 RPD. We stay slightly below them, and the
# Flash-Lite caps leave ~100 requests/day for the owner's other app on the same key.
LIMITS = {
    "gemma-4-26b-a4b-it": dict(rpm=28, tpm=15000, rpd=14300),
    "gemma-4-31b-it": dict(rpm=28, tpm=15000, rpd=14300),
    "gemini-3.5-flash-lite": dict(rpm=14, tpm=240000, rpd=400),
    "gemini-3.1-flash-lite": dict(rpm=14, tpm=240000, rpd=400),
}

SAFETY_OFF = [
    {"category": c, "threshold": "BLOCK_NONE"}
    for c in ("HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
              "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT")
]


def load_key():
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    for path in (os.path.join(ROOT, ".env"),):
        if os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                if line.startswith("GEMINI_API_KEY="):
                    return line.split("=", 1)[1].strip().strip("\"'")
    raise RuntimeError("GEMINI_API_KEY not found (set it in the environment or in .env)")


class DailyQuotaExceeded(Exception):
    pass


class _Cache:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.lock = threading.Lock()
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=60)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")
        self.db.commit()

    def get(self, k):
        with self.lock:
            row = self.db.execute("SELECT v FROM c WHERE k=?", (k,)).fetchone()
        return None if row is None else json.loads(row[0])

    def put(self, k, v):
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO c VALUES (?,?)", (k, json.dumps(v, ensure_ascii=False)))
            self.db.commit()


class _Limiter:
    """Sliding-window limiter for requests and input tokens per minute, plus a daily cap.

    State lives in a small SQLite file so that several processes (parallel experiment
    runners) share one quota instead of each assuming it owns all of it."""

    def __init__(self, model, rpm, tpm, rpd):
        self.model, self.rpm, self.tpm, self.rpd = model, rpm, tpm, rpd
        os.makedirs(os.path.dirname(RL_PATH), exist_ok=True)
        self.db = sqlite3.connect(RL_PATH, check_same_thread=False, timeout=120, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS win (model TEXT, t REAL, tok INTEGER)")
        self.db.execute("CREATE TABLE IF NOT EXISTS daily (model TEXT, day TEXT, n INTEGER, PRIMARY KEY(model, day))")
        self.lock = threading.Lock()

    def acquire(self, est_tokens):
        while True:
            wait = 0.5
            with self.lock:
                self.db.execute("BEGIN IMMEDIATE")
                try:
                    now = time.time()
                    day = _pacific_day()
                    self.db.execute("DELETE FROM win WHERE model=? AND t<?", (self.model, now - 60))
                    n, used, first = self.db.execute(
                        "SELECT COUNT(*), COALESCE(SUM(tok),0), MIN(t) FROM win WHERE model=?", (self.model,)).fetchone()
                    row = self.db.execute("SELECT n FROM daily WHERE model=? AND day=?", (self.model, day)).fetchone()
                    today = row[0] if row else 0
                    if today >= self.rpd:
                        self.db.execute("COMMIT")
                        secs = _seconds_until_pacific_midnight()
                        print(f"[llm] own daily cap ({self.rpd}) reached for {self.model}; sleeping {secs/3600:.1f} h", flush=True)
                        time.sleep(secs)
                        continue
                    if n < self.rpm and used + est_tokens <= self.tpm:
                        self.db.execute("INSERT INTO win VALUES (?,?,?)", (self.model, now, est_tokens))
                        self.db.execute("INSERT INTO daily VALUES (?,?,1) ON CONFLICT(model, day) DO UPDATE SET n=n+1",
                                        (self.model, day))
                        self.db.execute("COMMIT")
                        return
                    self.db.execute("COMMIT")
                    if n >= self.rpm and first:
                        wait = max(wait, 60 - (now - first) + 0.05)
                except Exception:
                    self.db.execute("ROLLBACK")
                    raise
            time.sleep(min(wait, 5))

    def used_today(self):
        row = self.db.execute("SELECT n FROM daily WHERE model=? AND day=?", (self.model, _pacific_day())).fetchone()
        return row[0] if row else 0


_CACHE = None
_LIMITERS = {}
_GLOBAL_LOCK = threading.Lock()
_LOG_LOCK = threading.Lock()


def _pacific_day():
    return (datetime.now(timezone.utc) - timedelta(hours=7)).strftime("%Y-%m-%d")


def _seconds_until_pacific_midnight():
    # Gemini daily quotas reset at midnight Pacific time (PDT = UTC-7 until early November).
    now = datetime.now(timezone.utc)
    pac = now - timedelta(hours=7)
    nxt = (pac + timedelta(days=1)).replace(hour=0, minute=5, second=0, microsecond=0)
    return (nxt - pac).total_seconds()


class LLM:
    def __init__(self, model="gemma-4-26b-a4b-it", thinking="minimal", wait_on_daily_quota=True):
        global _CACHE
        self.model = model
        self.thinking = thinking
        self.key = load_key()
        self.wait_on_daily_quota = wait_on_daily_quota
        with _GLOBAL_LOCK:
            if _CACHE is None:
                _CACHE = _Cache(CACHE_PATH)
            if model not in _LIMITERS:
                lim = LIMITS.get(model, dict(rpm=10, tpm=100000, rpd=300))
                _LIMITERS[model] = _Limiter(model, lim["rpm"], lim["tpm"], lim["rpd"])
        self.cache = _CACHE
        self.limiter = _LIMITERS[model]

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _merge(messages):
        """Gemini wants alternating user/model turns; merge consecutive same-role messages."""
        out = []
        for m in messages:
            role = "model" if m["role"] == "assistant" else "user"
            if out and out[-1]["role"] == role:
                out[-1]["parts"][0]["text"] += "\n\n" + m["content"]
            else:
                out.append({"role": role, "parts": [{"text": m["content"]}]})
        return out

    def _key(self, system, messages, temperature, max_tokens):
        blob = json.dumps([self.model, self.thinking, system, messages, temperature, max_tokens],
                          ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _log(self, rec):
        with _LOG_LOCK:
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------ main API
    def chat(self, messages, system=None, temperature=0.0, max_tokens=400, tag="", json_mode=False):
        """messages: list of {"role": "user"|"assistant", "content": str}. Returns (text, meta)."""
        k = self._key(system, messages, temperature, max_tokens if not json_mode else ("json", max_tokens))
        hit = self.cache.get(k)
        if hit is not None:
            return hit["text"], {"cached": True, "latency": hit.get("latency", 0.0), "calls": 0}

        contents = self._merge(messages)
        if system:
            # Gemma models on the Gemini API take no system instruction, so the system
            # message is prepended to the first user turn (what chat templates do anyway).
            contents[0]["parts"][0]["text"] = system + "\n\n" + contents[0]["parts"][0]["text"]
        gen = {"temperature": temperature, "maxOutputTokens": max_tokens}
        if self.thinking:
            gen["thinkingConfig"] = {"thinkingLevel": self.thinking}
        if json_mode:
            gen["responseMimeType"] = "application/json"
        body = {"contents": contents, "generationConfig": gen, "safetySettings": SAFETY_OFF}
        est = int(sum(len(c["parts"][0]["text"]) for c in contents) / 3.0) + 50

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.key}"
        attempt = 0
        while True:
            self.limiter.acquire(est)
            t0 = time.time()
            try:
                req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                             headers={"Content-Type": "application/json"})
                resp = json.load(urllib.request.urlopen(req, timeout=300))
                latency = time.time() - t0
                break
            except urllib.error.HTTPError as e:
                msg = e.read().decode("utf-8", "replace")
                attempt += 1
                if e.code == 429 and ("PerDay" in msg or "per day" in msg.lower()):
                    if not self.wait_on_daily_quota:
                        raise DailyQuotaExceeded(msg[:300])
                    secs = _seconds_until_pacific_midnight()
                    print(f"[llm] daily quota hit for {self.model}; sleeping {secs/3600:.1f} h", flush=True)
                    time.sleep(secs)
                    attempt = 0
                    continue
                if e.code in (429, 500, 502, 503, 504) and attempt < 12:
                    time.sleep(min(90, 5 * 2 ** min(attempt, 4)))
                    continue
                if e.code == 400 and attempt < 3 and "INTERNAL" in msg:
                    time.sleep(5)
                    continue
                raise RuntimeError(f"Gemini API error {e.code}: {msg[:500]}")
            except Exception as e:  # network hiccups, timeouts
                attempt += 1
                if attempt < 12:
                    time.sleep(min(90, 5 * 2 ** min(attempt, 4)))
                    continue
                raise

        text = ""
        cands = resp.get("candidates") or []
        finish = cands[0].get("finishReason") if cands else "NO_CANDIDATE"
        if cands:
            parts = (cands[0].get("content") or {}).get("parts") or []
            text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        usage = resp.get("usageMetadata", {})
        rec = {"t": time.time(), "model": self.model, "tag": tag, "latency": round(latency, 3),
               "in": usage.get("promptTokenCount"), "out": usage.get("candidatesTokenCount"),
               "think": usage.get("thoughtsTokenCount"), "finish": finish}
        self._log(rec)
        self.cache.put(k, {"text": text, "latency": latency, "finish": finish})
        return text, {"cached": False, "latency": latency, "calls": 1}

    def __call__(self, prompt, system=None, **kw):
        return self.chat([{"role": "user", "content": prompt}], system=system, **kw)[0]
