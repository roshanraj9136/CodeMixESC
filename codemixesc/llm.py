"""Thin, cached, rate-limited LLM client.

The default backend is the Gemini API (hosted open-weight Gemma models and Gemini
Flash-Lite). Models written as "ollama:<name>", "groq:<name>" or "openai:<name>" go to an
OpenAI-compatible endpoint instead (local Ollama, Groq, or any server set in OPENAI_BASE_URL).

Every call is cached on disk (SQLite), so re-running an experiment or an ablation that
shares a prefix of the pipeline costs nothing. Every real call is also logged to a JSONL
file so the number of LLM calls and the latency per turn can be reported.
"""
import hashlib
import json
import os
import re
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
    "gemini-3.5-flash-lite": dict(rpm=14, tpm=240000, rpd=490),  # the other app on the key falls back to 3.1
    "gemini-3.1-flash-lite": dict(rpm=14, tpm=240000, rpd=400),
}
UNLIMITED = dict(rpm=100000, tpm=10 ** 9, rpd=10 ** 9)

# OpenAI-compatible backends: prefix -> (base URL, environment variable holding the key)
BACKENDS = {
    "ollama": (os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"), None),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "openai": (os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"), "OPENAI_API_KEY"),
}
# finish reasons after which an empty answer is worth one more try (blocked / no candidate)
_RETRY_EMPTY = {"SAFETY", "OTHER", "NO_CANDIDATE", "RECITATION", "PROHIBITED_CONTENT", "BLOCKLIST",
                "SPII", "MALFORMED_FUNCTION_CALL", "content_filter", None}

SAFETY_OFF = [
    {"category": c, "threshold": "BLOCK_NONE"}
    for c in ("HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
              "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT")
]


def load_key(name="GEMINI_API_KEY"):
    key = os.environ.get(name)
    if key:
        return key
    path = os.path.join(ROOT, ".env")
    if os.path.exists(path):
        raw = open(path, "rb").read()  # PowerShell writes UTF-8 with a BOM, or UTF-16
        text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8-sig")
        for line in text.splitlines():
            if line.strip().startswith(name + "="):
                return line.split("=", 1)[1].strip().strip("\"'")
    raise RuntimeError(f"{name} not found (set it in the environment or in .env)")


def split_model(model):
    """'ollama:qwen2.5:7b' -> ('ollama', 'qwen2.5:7b'); a bare name -> ('gemini', name)."""
    for prefix in BACKENDS:
        if model.startswith(prefix + ":"):
            return prefix, model[len(prefix) + 1:]
    return "gemini", model


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

    def put_if_absent(self, k, v):
        """First writer wins, across threads and processes; returns the stored value, so callers
        that raced on the same prompt all continue with the same answer."""
        with self.lock:
            self.db.execute("INSERT OR IGNORE INTO c VALUES (?,?)", (k, json.dumps(v, ensure_ascii=False)))
            self.db.commit()
            row = self.db.execute("SELECT v FROM c WHERE k=?", (k,)).fetchone()
        return json.loads(row[0])


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

    def acquire(self, est_tokens, wait_daily=True):
        """Blocks until a request with ~est_tokens input tokens may go out; returns a ticket for settle()."""
        est_tokens = min(est_tokens, self.tpm)  # a prompt larger than the whole budget still has to go out
        while True:
            wait, over = 0.5, False
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
                    ticket = None
                    if today >= self.rpd:
                        over = True
                    elif n < self.rpm and used + est_tokens <= self.tpm:
                        ticket = self.db.execute("INSERT INTO win VALUES (?,?,?)", (self.model, now, est_tokens)).lastrowid
                        self.db.execute("INSERT INTO daily VALUES (?,?,1) ON CONFLICT(model, day) DO UPDATE SET n=n+1",
                                        (self.model, day))
                    elif first:
                        wait = max(wait, 60 - (now - first) + 0.05)
                    self.db.execute("COMMIT")
                except Exception:
                    self.db.execute("ROLLBACK")
                    raise
            if ticket is not None:
                return ticket
            if over:  # outside the lock and the transaction
                if not wait_daily:
                    raise DailyQuotaExceeded(f"own daily cap ({self.rpd}) reached for {self.model}")
                secs = _seconds_until_pacific_midnight()
                print(f"[llm] own daily cap ({self.rpd}) reached for {self.model}; sleeping {secs/3600:.1f} h", flush=True)
                time.sleep(secs)
                continue
            time.sleep(min(wait, 5))

    def settle(self, ticket, tokens):
        """Replaces a request's estimated input tokens by the count the API reported, so the
        TPM window is not throttled by the estimate's safety margin."""
        if ticket is not None and tokens:
            with self.lock:
                self.db.execute("UPDATE win SET tok=? WHERE rowid=?", (int(tokens), ticket))

    def used_today(self):
        row = self.db.execute("SELECT n FROM daily WHERE model=? AND day=?", (self.model, _pacific_day())).fetchone()
        return row[0] if row else 0


_CACHE = None
_LIMITERS = {}
_GLOBAL_LOCK = threading.Lock()
_KEY_LOCKS = [threading.Lock() for _ in range(1024)]  # one in-flight request per prompt (striped by key)
_LOG_LOCK = threading.Lock()


def _pacific_now(now=None):
    """US Pacific time without tzdata (absent on many Windows installs): PDT (UTC-7) from the
    second Sunday of March 10:00 UTC to the first Sunday of November 09:00 UTC, else PST (UTC-8).
    Gemini daily quotas reset at Pacific midnight."""
    now = now or datetime.now(timezone.utc)

    def sunday(month, k):
        d = datetime(now.year, month, 1, tzinfo=timezone.utc)
        return d + timedelta(days=(6 - d.weekday()) % 7 + 7 * (k - 1))
    dst = sunday(3, 2) + timedelta(hours=10) <= now < sunday(11, 1) + timedelta(hours=9)
    return now - timedelta(hours=7 if dst else 8)


def _pacific_day():
    return _pacific_now().strftime("%Y-%m-%d")


def _seconds_until_pacific_midnight():
    pac = _pacific_now()
    nxt = pac.replace(hour=0, minute=5, second=0, microsecond=0)
    if nxt <= pac:
        nxt += timedelta(days=1)
    return (nxt - pac).total_seconds()


def _backoff(attempt):
    return min(90, 5 * 2 ** min(attempt, 4))


class LLM:
    def __init__(self, model="gemma-4-26b-a4b-it", thinking="minimal", wait_on_daily_quota=True):
        global _CACHE
        self.model = model
        self.backend, self.api_model = split_model(model)
        self.thinking = thinking if self.backend == "gemini" else None
        self.wait_on_daily_quota = wait_on_daily_quota
        if self.backend == "gemini":
            self.key = load_key("GEMINI_API_KEY")
        else:
            self.base_url, key_env = BACKENDS[self.backend]
            self.key = load_key(key_env) if key_env else "none"
        with _GLOBAL_LOCK:
            if _CACHE is None:
                _CACHE = _Cache(CACHE_PATH)
            if model not in _LIMITERS:
                default = UNLIMITED if self.backend == "ollama" else dict(rpm=10, tpm=100000, rpd=300)
                lim = LIMITS.get(model, default)
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
        blob = json.dumps([self.model, self.thinking, system or None, messages, float(temperature), max_tokens],
                          ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _log(self, rec):
        with _LOG_LOCK:
            os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _post(self, url, body, headers, label, est):
        """POST with retries (est = estimated input tokens for the limiter).
        Returns (JSON, latency, limiter ticket of the successful request)."""
        attempt = 0
        while True:
            ticket = self.limiter.acquire(est, self.wait_on_daily_quota)
            t0 = time.time()
            try:
                req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers)
                resp = json.load(urllib.request.urlopen(req, timeout=600))
                return resp, time.time() - t0, ticket
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
                    time.sleep(_backoff(attempt))
                    continue
                if e.code == 400 and attempt < 3 and "INTERNAL" in msg:
                    time.sleep(5)
                    continue
                raise RuntimeError(f"{label} API error {e.code}: {msg[:500]}")
            except Exception:  # network hiccups, timeouts
                attempt += 1
                if attempt < 12:
                    time.sleep(_backoff(attempt))
                    continue
                raise

    def _gemini(self, messages, system, temperature, max_tokens, json_mode):
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
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.api_model}:generateContent?key={self.key}"
        est = int(sum(len(c["parts"][0]["text"]) for c in contents) / 3.0) + 50
        resp, latency, ticket = self._post(url, body, {"Content-Type": "application/json"}, "Gemini", est)
        text = ""
        cands = resp.get("candidates") or []
        finish = cands[0].get("finishReason") if cands else "NO_CANDIDATE"
        if cands:
            parts = (cands[0].get("content") or {}).get("parts") or []
            text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        usage = resp.get("usageMetadata", {})
        self.limiter.settle(ticket, usage.get("promptTokenCount"))
        return text, finish, latency, {"in": usage.get("promptTokenCount"), "out": usage.get("candidatesTokenCount"),
                                       "think": usage.get("thoughtsTokenCount")}

    def _openai(self, messages, system, temperature, max_tokens, json_mode):
        msgs = [{"role": "system", "content": system}] if system else []
        for m in messages:  # same merging as for Gemini, so prompts are identical across backends
            if msgs and msgs[-1]["role"] == m["role"]:
                msgs[-1] = {"role": m["role"], "content": msgs[-1]["content"] + "\n\n" + m["content"]}
            else:
                msgs.append({"role": m["role"], "content": m["content"]})
        body = {"model": self.api_model, "messages": msgs, "temperature": temperature, "max_tokens": max_tokens}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"}
        est = int(sum(len(m["content"]) for m in msgs) / 3.0) + 50
        resp, latency, ticket = self._post(self.base_url.rstrip("/") + "/chat/completions", body, headers, self.backend, est)
        choice = (resp.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()  # reasoning models
        usage = resp.get("usage", {})
        self.limiter.settle(ticket, usage.get("prompt_tokens"))
        return text, choice.get("finish_reason"), latency, {"in": usage.get("prompt_tokens"),
                                                            "out": usage.get("completion_tokens"), "think": None}

    # ------------------------------------------------------------------ main API
    def chat(self, messages, system=None, temperature=0.0, max_tokens=400, tag="", json_mode=False):
        """messages: list of {"role": "user"|"assistant", "content": str}. Returns (text, meta)."""
        k = self._key(system, messages, temperature, max_tokens if not json_mode else ("json", max_tokens))
        hit = self.cache.get(k)
        if hit is None:
            with _KEY_LOCKS[int(k[:6], 16) % len(_KEY_LOCKS)]:  # concurrent identical prompts: one request
                hit = self.cache.get(k)
                if hit is None:
                    return self._fresh(k, messages, system, temperature, max_tokens, tag, json_mode)
        return hit["text"], {"cached": True, "latency": hit.get("latency", 0.0), "calls": 0,
                             "in": hit.get("in"), "out": hit.get("out"), "finish": hit.get("finish")}

    def _fresh(self, k, messages, system, temperature, max_tokens, tag, json_mode):
        call = self._gemini if self.backend == "gemini" else self._openai
        calls = 0
        for attempt in range(3):
            text, finish, latency, usage = call(messages, system, temperature, max_tokens, json_mode)
            calls += 1
            self._log({"t": time.time(), "model": self.model, "tag": tag, "latency": round(latency, 3),
                       **usage, "finish": finish})
            if text.strip() or finish not in _RETRY_EMPTY:  # a blocked/empty candidate is often transient
                break
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
        meta = {"cached": False, "latency": latency, "calls": calls, "in": usage.get("in"), "out": usage.get("out"),
                "finish": finish}
        if not text.strip():  # never cache a failure: a later run asks again instead of reusing ""
            meta["failed"] = True
            return text, meta
        stored = self.cache.put_if_absent(k, {"text": text, "latency": latency, "finish": finish,
                                              "in": usage.get("in"), "out": usage.get("out")})
        return stored["text"], meta

    def __call__(self, prompt, system=None, **kw):
        return self.chat([{"role": "user", "content": prompt}], system=system, **kw)[0]
