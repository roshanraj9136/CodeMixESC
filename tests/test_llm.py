"""LLM client: caching, retries, empty-answer handling, backends and the rate limiter, with the
HTTP layer mocked (no network, no key)."""
import io
import json
import time
import urllib.error

import pytest

from codemixesc import llm as L


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "CACHE_PATH", str(tmp_path / "cache.sqlite"))
    monkeypatch.setattr(L, "LOG_PATH", str(tmp_path / "calls.jsonl"))
    monkeypatch.setattr(L, "RL_PATH", str(tmp_path / "rl.sqlite"))
    monkeypatch.setattr(L, "_CACHE", None)
    monkeypatch.setattr(L, "_LIMITERS", {})
    monkeypatch.setattr(L.time, "sleep", lambda s: None)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    calls = []

    def install(responses):
        def fake_urlopen(req, timeout=None):
            calls.append({"url": req.full_url, "body": json.loads(req.data.decode("utf-8")),
                          "headers": dict(req.header_items())})
            r = responses.pop(0)
            if isinstance(r, Exception):
                raise r
            return io.BytesIO(json.dumps(r).encode("utf-8"))
        monkeypatch.setattr(L.urllib.request, "urlopen", fake_urlopen)
        return calls
    return install


def gemini(text, finish="STOP", thought=None):
    parts = ([{"text": thought, "thought": True}] if thought else []) + [{"text": text}]
    return {"candidates": [{"content": {"parts": parts}, "finishReason": finish}],
            "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 5}}


def http_error(code, msg="err"):
    return urllib.error.HTTPError("u", code, "x", {}, io.BytesIO(msg.encode("utf-8")))


def test_gemini_call_cache_and_request_shape(client):
    calls = client([gemini("Response: hi", thought="thinking...")])
    llm = L.LLM("gemma-4-26b-a4b-it")
    msgs = [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}, {"role": "assistant", "content": "c"}]
    text, meta = llm.chat(msgs, system="SYS", max_tokens=50, tag="t")
    assert text == "Response: hi" and not meta["cached"] and meta["calls"] == 1 and meta["in"] == 11
    body = calls[0]["body"]
    # system prepended to the first user turn; consecutive user messages merged; roles user/model
    assert body["contents"] == [{"role": "user", "parts": [{"text": "SYS\n\na\n\nb"}]},
                                {"role": "model", "parts": [{"text": "c"}]}]
    assert body["generationConfig"] == {"temperature": 0.0, "maxOutputTokens": 50, "thinkingConfig": {"thinkingLevel": "minimal"}}
    assert "gemma-4-26b-a4b-it:generateContent?key=test-key" in calls[0]["url"]
    text2, meta2 = llm.chat(msgs, system="SYS", max_tokens=50, tag="t")  # identical -> cache, no HTTP
    assert text2 == text and meta2["cached"] and meta2["calls"] == 0 and len(calls) == 1
    calls = client([gemini("other"), gemini('{"a": 1}')])
    assert llm.chat(msgs, system="SYS", max_tokens=51)[0] == "other"  # any parameter change is a new key
    assert llm.chat(msgs, system="SYS", max_tokens=50, json_mode=True)[0] == '{"a": 1}'
    assert calls[-1]["body"]["generationConfig"]["responseMimeType"] == "application/json"


def test_retries_transient_errors_but_not_bad_requests(client):
    client([http_error(503), http_error(429, "rate"), gemini("ok")])
    assert L.LLM("gemma-4-26b-a4b-it")("p") == "ok"
    client([http_error(400, "bad request")])
    with pytest.raises(RuntimeError):
        L.LLM("gemma-4-26b-a4b-it")("p2")


def test_daily_quota(client):
    client([http_error(429, "Quota exceeded for metric GenerateRequestsPerDay")])
    with pytest.raises(L.DailyQuotaExceeded):
        L.LLM("gemma-4-26b-a4b-it", wait_on_daily_quota=False)("p")


def test_blocked_answer_retried_before_caching(client):
    calls = client([gemini("", finish="SAFETY"), gemini("", finish="OTHER"), gemini("finally")])
    llm = L.LLM("gemma-4-26b-a4b-it")
    text, meta = llm.chat([{"role": "user", "content": "q"}])
    assert text == "finally" and meta["calls"] == 3 and len(calls) == 3
    calls = client([gemini("", finish="SAFETY")] * 3 + [gemini("later ok")])
    text, meta = llm.chat([{"role": "user", "content": "q2"}])
    assert text == "" and meta["calls"] == 3 and meta["failed"]  # gives up after 3 tries ...
    text, meta = llm.chat([{"role": "user", "content": "q2"}])  # ... but never caches the failure
    assert text == "later ok" and not meta["cached"]
    assert llm.chat([{"role": "user", "content": "q2"}])[1]["cached"]


def test_concurrent_identical_prompts_make_one_request(client):
    import threading
    calls = client([gemini("first"), gemini("second")])
    llm = L.LLM("gemma-4-26b-a4b-it")
    out = []
    threads = [threading.Thread(target=lambda: out.append(llm("same prompt"))) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert out == ["first", "first"] and len(calls) == 1


def test_key_normalisation_and_env_encodings(client, tmp_path, monkeypatch):
    llm = L.LLM("gemma-4-26b-a4b-it")
    msgs = [{"role": "user", "content": "x"}]
    assert llm._key(None, msgs, 0, 5) == llm._key("", msgs, 0.0, 5)
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.setattr(L, "ROOT", str(tmp_path))
    for enc, data in (("utf-8-sig", "GEMINI_API_KEY=abc\r\n"), ("utf-16", "GEMINI_API_KEY=abc\r\n")):
        (tmp_path / ".env").write_bytes(data.encode(enc))
        assert L.load_key() == "abc"


def test_pacific_time_dst():
    from datetime import datetime, timezone
    pdt = L._pacific_now(datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc))
    pst = L._pacific_now(datetime(2026, 11, 15, 12, 0, tzinfo=timezone.utc))
    assert pdt.hour == 5 and pst.hour == 4
    assert L._pacific_now(datetime(2026, 11, 1, 8, 59, tzinfo=timezone.utc)).hour == 1  # still PDT
    assert L._pacific_now(datetime(2026, 11, 1, 9, 0, tzinfo=timezone.utc)).hour == 1   # 01:00 PST again


def test_openai_compatible_backend(client):
    calls = client([{"choices": [{"message": {"content": "<think>hmm</think>Response: hey"}, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 7, "completion_tokens": 2}}])
    llm = L.LLM("groq:llama-3.3-70b-versatile")
    text, meta = llm.chat([{"role": "user", "content": "a"}, {"role": "user", "content": "b"}], system="S")
    assert text == "Response: hey" and meta["in"] == 7
    body = calls[0]["body"]
    assert body["model"] == "llama-3.3-70b-versatile" and calls[0]["url"].endswith("/chat/completions")
    assert body["messages"] == [{"role": "system", "content": "S"}, {"role": "user", "content": "a\n\nb"}]
    assert calls[0]["headers"]["Authorization"] == "Bearer groq-key"
    assert L.split_model("ollama:qwen2.5:7b") == ("ollama", "qwen2.5:7b")


def test_limiter_enforces_rpm(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "RL_PATH", str(tmp_path / "rl.sqlite"))
    now = [1000.0]
    monkeypatch.setattr(L.time, "time", lambda: now[0])
    waits = []

    def fake_sleep(s):
        waits.append(s)
        now[0] += s
    monkeypatch.setattr(L.time, "sleep", fake_sleep)
    lim = L._Limiter("m", rpm=3, tpm=10 ** 9, rpd=100)
    for _ in range(3):
        lim.acquire(10)
    assert waits == []
    lim.acquire(10)  # the 4th request in the same minute has to wait for the window to slide
    assert sum(waits) >= 59 and lim.used_today() == 4
    big = L._Limiter("big", rpm=100, tpm=1000, rpd=100)
    t = big.acquire(5000)  # a prompt larger than the whole TPM budget still goes out (no deadlock)
    big.settle(t, 300)     # the reported token count replaces the estimate
    assert big.db.execute("SELECT SUM(tok) FROM win WHERE model='big'").fetchone()[0] == 300
    capped = L._Limiter("capped", rpm=100, tpm=10 ** 9, rpd=1)
    capped.acquire(1)
    with pytest.raises(L.DailyQuotaExceeded):
        capped.acquire(1, wait_daily=False)
