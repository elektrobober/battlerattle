# AI Analysis via API — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the manual copy-paste AI step with an automated stage that sends chunk prompts to an LLM provider (Anthropic Batch/direct API or any OpenAI-compatible endpoint incl. local Ollama/LM Studio) and writes results to `manual_ai_results/`.

**Architecture:** New module `ai_providers.py` holds pure provider classes (no imports from `dnd_pipeline`). Orchestration (`run_ai_analysis`, job building, state cache, CLI) lives in `dnd_pipeline.py` and lazily imports the providers. Results land in the exact files `build_reports()` already reads, so reporting code is untouched. Manual mode stays the fallback.

**Tech Stack:** Python 3.10+ stdlib (`urllib`, `concurrent.futures`), `anthropic` SDK (lazy import, only for provider=anthropic), pytest with mocks — no network in tests.

**Spec:** `docs/superpowers/specs/2026-08-01-ai-api-analysis-design.md`

## Global Constraints

- API keys come only from environment variables (`api_key_env`, default `ANTHROPIC_API_KEY` for anthropic). Never write keys to config, git, or logs.
- Default model `claude-sonnet-5`; default provider `anthropic`; default mode `batch`.
- `ai.enabled` defaults to **false** — old configs keep working in manual mode.
- Results file format/location unchanged: `manual_ai_results/{chunk_name}_events.json`.
- Hand-placed result files (no state entry) are never overwritten unless `--force`.
- `ai_providers.py` must not import `dnd_pipeline` (avoids circular import).
- User-facing log messages in Russian, matching existing style (`logger.info(f"...")`).
- Tests make zero network calls; `anthropic` package must NOT be required to run the test suite (inject stub clients).
- Follow existing code idioms: `load_json`/`write_json`, `stable_hash`, `deep_merge`, f-string logging, type hints `dict[str, Any]`.

---

### Task 1: AI config resolution

**Files:**
- Modify: `dnd_pipeline.py` (add after `apply_quality_profile`, ~line 224)
- Test: `tests/test_ai_analysis.py` (new)

**Interfaces:**
- Consumes: existing `deep_merge(base, override)` from `dnd_pipeline.py`.
- Produces: `AI_DEFAULTS: dict[str, Any]`, `resolve_ai_config(cfg: dict[str, Any]) -> dict[str, Any]` (raises `ValueError` on bad provider / missing base_url), `resolve_ai_api_key(ai: dict[str, Any]) -> str | None`. Used by Tasks 5–7.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_ai_analysis.py
"""Tests for the API-driven AI analysis stage in dnd_pipeline."""
import pytest

import dnd_pipeline as dp


class TestResolveAiConfig:
    def test_defaults_when_section_missing(self):
        ai = dp.resolve_ai_config({})
        assert ai["enabled"] is False
        assert ai["provider"] == "anthropic"
        assert ai["model"] == "claude-sonnet-5"
        assert ai["mode"] == "batch"
        assert ai["max_output_tokens"] == 8000
        assert ai["concurrency"] == 2

    def test_overrides_merge(self):
        ai = dp.resolve_ai_config({"ai": {"enabled": True, "model": "llama3.1"}})
        assert ai["enabled"] is True
        assert ai["model"] == "llama3.1"
        assert ai["provider"] == "anthropic"  # untouched default

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="ai.provider"):
            dp.resolve_ai_config({"ai": {"provider": "gemini"}})

    def test_openai_compatible_requires_base_url(self):
        with pytest.raises(ValueError, match="base_url"):
            dp.resolve_ai_config({"ai": {"provider": "openai_compatible"}})

    def test_openai_compatible_with_base_url_ok(self):
        ai = dp.resolve_ai_config(
            {"ai": {"provider": "openai_compatible", "base_url": "http://localhost:11434/v1"}}
        )
        assert ai["base_url"] == "http://localhost:11434/v1"


class TestResolveAiApiKey:
    def test_anthropic_default_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        ai = dp.resolve_ai_config({"ai": {"provider": "anthropic"}})
        assert dp.resolve_ai_api_key(ai) == "sk-test"

    def test_custom_env_name(self, monkeypatch):
        monkeypatch.setenv("MY_KEY", "xyz")
        ai = dp.resolve_ai_config({"ai": {"api_key_env": "MY_KEY"}})
        assert dp.resolve_ai_api_key(ai) == "xyz"

    def test_missing_env_returns_none(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        ai = dp.resolve_ai_config({})
        assert dp.resolve_ai_api_key(ai) is None

    def test_openai_compatible_no_default_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "should-not-be-used")
        ai = dp.resolve_ai_config(
            {"ai": {"provider": "openai_compatible", "base_url": "http://x/v1"}}
        )
        assert dp.resolve_ai_api_key(ai) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_ai_analysis.py -v`
Expected: FAIL with `AttributeError: module 'dnd_pipeline' has no attribute 'resolve_ai_config'`

- [ ] **Step 3: Implement in `dnd_pipeline.py`** (place after `apply_quality_profile`)

```python
# ──────────────────────────────────────────────────────────────
# AI analysis config
# ──────────────────────────────────────────────────────────────

AI_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "provider": "anthropic",          # "anthropic" | "openai_compatible"
    "model": "claude-sonnet-5",
    "mode": "batch",                  # anthropic only: "batch" | "direct"
    "base_url": None,                 # openai_compatible: e.g. http://localhost:11434/v1
    "api_key_env": None,              # env var name; default ANTHROPIC_API_KEY for anthropic
    "max_output_tokens": 8000,
    "concurrency": 2,
}


def resolve_ai_config(cfg: dict[str, Any]) -> dict[str, Any]:
    ai = deep_merge(AI_DEFAULTS, cfg.get("ai") or {})
    if ai["provider"] not in ("anthropic", "openai_compatible"):
        raise ValueError(f"Неизвестный ai.provider: {ai['provider']} (жду anthropic или openai_compatible)")
    if ai["provider"] == "openai_compatible" and not ai["base_url"]:
        raise ValueError("Для ai.provider=openai_compatible нужен ai.base_url (например http://localhost:11434/v1)")
    return ai


def resolve_ai_api_key(ai: dict[str, Any]) -> str | None:
    env_name = ai.get("api_key_env") or ("ANTHROPIC_API_KEY" if ai["provider"] == "anthropic" else None)
    return os.environ.get(env_name) if env_name else None
```

Check top of file: `import os` must be present (add if missing).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_ai_analysis.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add dnd_pipeline.py tests/test_ai_analysis.py
git commit -m "feat: AI config resolution with provider validation and env-only keys"
```

---

### Task 2: ai_providers module core — dataclasses, schema, JSON parsing

**Files:**
- Create: `ai_providers.py` (repo root, next to `dnd_pipeline.py`)
- Test: `tests/test_ai_providers.py` (new)

**Interfaces:**
- Consumes: nothing from `dnd_pipeline` (hard rule).
- Produces: `ChunkJob(name: str, prompt: str, chunk_hash: str)`, `AIResult(name: str, data: dict | None = None, error: str | None = None)`, `EVENTS_SCHEMA: dict`, `parse_model_json(name: str, text: str, normalize: Callable[[str], str]) -> AIResult`. Used by Tasks 3–6.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_ai_providers.py
"""Tests for ai_providers: pure provider classes, no network, no anthropic package needed."""
import json

import ai_providers as ap


class TestParseModelJson:
    def _ok_payload(self):
        return {
            "session": "s", "chunk_index": 0, "scene_type": "gameplay",
            "actions": [], "dice_rolls": [], "mvp_signals": [], "summary": "x",
        }

    def test_valid_json(self):
        res = ap.parse_model_json("chunk_000", json.dumps(self._ok_payload()), lambda t: t)
        assert res.error is None
        assert res.data["summary"] == "x"

    def test_normalize_applied(self):
        fenced = "```json\n" + json.dumps(self._ok_payload()) + "\n```"
        strip = lambda t: t.strip().removeprefix("```json").removesuffix("```").strip()
        res = ap.parse_model_json("chunk_000", fenced, strip)
        assert res.error is None

    def test_broken_json(self):
        res = ap.parse_model_json("chunk_000", "{not json", lambda t: t)
        assert res.data is None
        assert "JSON" in res.error

    def test_missing_required_fields(self):
        res = ap.parse_model_json("chunk_000", '{"summary": "x"}', lambda t: t)
        assert res.data is None
        assert "actions" in res.error


class TestEventsSchema:
    def test_all_objects_forbid_additional_properties(self):
        def walk(node):
            if isinstance(node, dict):
                if node.get("type") == "object":
                    assert node.get("additionalProperties") is False
                    assert "required" in node
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)
        walk(ap.EVENTS_SCHEMA)

    def test_top_level_fields(self):
        props = ap.EVENTS_SCHEMA["properties"]
        for key in ("session", "chunk_index", "scene_type", "actions", "dice_rolls", "mvp_signals", "summary"):
            assert key in props
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_ai_providers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ai_providers'`

- [ ] **Step 3: Create `ai_providers.py`**

```python
"""LLM providers for the AI analysis stage.

Pure provider classes: no imports from dnd_pipeline (the orchestrator
imports us, not the other way around). Network code lives here; all
callers mock it in tests.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger("dnd_pipeline")

REQUIRED_RESULT_FIELDS = ("actions", "dice_rolls", "mvp_signals", "summary")


@dataclass
class ChunkJob:
    name: str          # chunk file stem, e.g. "chunk_000"
    prompt: str
    chunk_hash: str


@dataclass
class AIResult:
    name: str
    data: dict[str, Any] | None = None
    error: str | None = None


# JSON schema mirrors the format demanded by prompt_for_chunk().
# Structured outputs require additionalProperties: false on every object.
EVENTS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["session", "chunk_index", "scene_type", "actions", "dice_rolls", "mvp_signals", "summary"],
    "properties": {
        "session": {"type": "string"},
        "chunk_index": {"type": "integer"},
        "scene_type": {"type": "string"},
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["time", "character", "action", "outcome", "importance"],
                "properties": {
                    "time": {"type": "string"},
                    "character": {"type": "string"},
                    "action": {"type": "string"},
                    "outcome": {"type": "string"},
                    "importance": {"type": "string", "enum": ["low", "medium", "high"]},
                },
            },
        },
        "dice_rolls": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["time", "character", "roll_type", "die", "natural",
                             "modifier", "total", "context", "confidence", "raw_text"],
                "properties": {
                    "time": {"type": "string"},
                    "character": {"type": "string"},
                    "roll_type": {"type": "string"},
                    "die": {"type": ["string", "null"]},
                    "natural": {"type": ["integer", "null"]},
                    "modifier": {"type": ["integer", "null"]},
                    "total": {"type": ["integer", "null"]},
                    "context": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                    "raw_text": {"type": "string"},
                },
            },
        },
        "mvp_signals": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["time", "character", "category", "reason", "weight"],
                "properties": {
                    "time": {"type": "string"},
                    "character": {"type": "string"},
                    "category": {"type": "string"},
                    "reason": {"type": "string"},
                    "weight": {"type": "integer"},
                },
            },
        },
        "summary": {"type": "string"},
    },
}


def parse_model_json(name: str, text: str, normalize: Callable[[str], str]) -> AIResult:
    """Parse a model's text answer into a result dict; soft-fail on garbage."""
    try:
        data = json.loads(normalize(text))
    except ValueError as e:
        return AIResult(name=name, error=f"битый JSON от модели: {e}")
    if not isinstance(data, dict):
        return AIResult(name=name, error="ответ модели — не JSON-объект")
    missing = [k for k in REQUIRED_RESULT_FIELDS if k not in data]
    if missing:
        return AIResult(name=name, error=f"в ответе нет полей: {', '.join(missing)}")
    return AIResult(name=name, data=data)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_ai_providers.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add ai_providers.py tests/test_ai_providers.py
git commit -m "feat: ai_providers core - job/result types, events schema, tolerant JSON parsing"
```

---

### Task 3: OpenAICompatProvider (Ollama / LM Studio / OpenRouter / vLLM)

**Files:**
- Modify: `ai_providers.py` (append)
- Test: `tests/test_ai_providers.py` (append)

**Interfaces:**
- Consumes: `ChunkJob`, `AIResult`, `parse_model_json` from Task 2.
- Produces: `OpenAICompatProvider(model, base_url, api_key=None, max_output_tokens=8000, concurrency=2, normalize=None, timeout=600, retries=3)` with `analyze(jobs: list[ChunkJob], on_result: Callable[[AIResult], None] | None = None) -> list[AIResult]`. Used by Task 6.

- [ ] **Step 1: Write failing tests** (append to `tests/test_ai_providers.py`)

```python
import io
import urllib.error


def _fake_response(content: str):
    body = json.dumps({"choices": [{"message": {"content": content}}]}).encode("utf-8")

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    return Resp(body)


def _ok_content():
    return json.dumps({
        "session": "s", "chunk_index": 0, "scene_type": "gameplay",
        "actions": [], "dice_rolls": [], "mvp_signals": [], "summary": "ok",
    })


class TestOpenAICompatProvider:
    def _provider(self, **kw):
        defaults = dict(model="llama3.1", base_url="http://localhost:11434/v1", concurrency=1, retries=2)
        defaults.update(kw)
        return ap.OpenAICompatProvider(**defaults)

    def test_success_and_payload(self, monkeypatch):
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            captured["payload"] = json.loads(req.data.decode("utf-8"))
            captured["auth"] = req.headers.get("Authorization")
            return _fake_response(_ok_content())

        monkeypatch.setattr(ap.urllib.request, "urlopen", fake_urlopen)
        jobs = [ap.ChunkJob(name="chunk_000", prompt="PROMPT", chunk_hash="h")]
        results = self._provider().analyze(jobs)
        assert results[0].data["summary"] == "ok"
        assert captured["url"] == "http://localhost:11434/v1/chat/completions"
        assert captured["payload"]["model"] == "llama3.1"
        assert captured["payload"]["messages"][0]["content"] == "PROMPT"
        assert captured["auth"] is None  # no key for local runtimes

    def test_bearer_header_when_key_given(self, monkeypatch):
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["auth"] = req.headers.get("Authorization")
            return _fake_response(_ok_content())

        monkeypatch.setattr(ap.urllib.request, "urlopen", fake_urlopen)
        self._provider(api_key="sk-x").analyze([ap.ChunkJob("chunk_000", "p", "h")])
        assert captured["auth"] == "Bearer sk-x"

    def test_retry_on_500_then_success(self, monkeypatch):
        calls = {"n": 0}

        def fake_urlopen(req, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise urllib.error.HTTPError(req.full_url, 500, "boom", None, io.BytesIO(b""))
            return _fake_response(_ok_content())

        monkeypatch.setattr(ap.urllib.request, "urlopen", fake_urlopen)
        monkeypatch.setattr(ap.time, "sleep", lambda s: None)
        results = self._provider().analyze([ap.ChunkJob("chunk_000", "p", "h")])
        assert calls["n"] == 2
        assert results[0].error is None

    def test_400_fails_without_retry(self, monkeypatch):
        calls = {"n": 0}

        def fake_urlopen(req, timeout=None):
            calls["n"] += 1
            raise urllib.error.HTTPError(req.full_url, 400, "bad request", None, io.BytesIO(b""))

        monkeypatch.setattr(ap.urllib.request, "urlopen", fake_urlopen)
        results = self._provider().analyze([ap.ChunkJob("chunk_000", "p", "h")])
        assert calls["n"] == 1
        assert results[0].data is None
        assert "400" in results[0].error

    def test_broken_json_soft_fails(self, monkeypatch):
        monkeypatch.setattr(ap.urllib.request, "urlopen",
                            lambda req, timeout=None: _fake_response("{oops"))
        results = self._provider().analyze([ap.ChunkJob("chunk_000", "p", "h")])
        assert results[0].data is None
        assert "JSON" in results[0].error

    def test_on_result_callback_called(self, monkeypatch):
        monkeypatch.setattr(ap.urllib.request, "urlopen",
                            lambda req, timeout=None: _fake_response(_ok_content()))
        seen = []
        self._provider().analyze([ap.ChunkJob("chunk_000", "p", "h")], on_result=seen.append)
        assert [r.name for r in seen] == ["chunk_000"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_ai_providers.py -v -k OpenAICompat`
Expected: FAIL with `AttributeError: module 'ai_providers' has no attribute 'OpenAICompatProvider'`

- [ ] **Step 3: Implement** (append to `ai_providers.py`)

```python
class OpenAICompatProvider:
    """Any OpenAI-compatible /v1/chat/completions endpoint: Ollama, LM Studio, vLLM, OpenRouter, OpenAI."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None = None,
        max_output_tokens: int = 8000,
        concurrency: int = 2,
        normalize: Callable[[str], str] | None = None,
        timeout: int = 600,
        retries: int = 3,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_output_tokens = max_output_tokens
        self.concurrency = max(1, concurrency)
        self.normalize = normalize or (lambda t: t)
        self.timeout = timeout
        self.retries = retries

    def analyze(
        self,
        jobs: list[ChunkJob],
        on_result: Callable[[AIResult], None] | None = None,
    ) -> list[AIResult]:
        results: list[AIResult] = []
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            for res in pool.map(self._analyze_one, jobs):
                if on_result:
                    on_result(res)
                results.append(res)
        return results

    def _analyze_one(self, job: ChunkJob) -> AIResult:
        payload = {
            "model": self.model,
            "max_tokens": self.max_output_tokens,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": job.prompt}],
        }
        try:
            raw = self._post_with_retries(payload)
        except Exception as e:  # noqa: BLE001 — per-chunk soft fail, summary logged by caller
            return AIResult(name=job.name, error=str(e))
        return parse_model_json(job.name, raw, self.normalize)

    def _post_with_retries(self, payload: dict[str, Any]) -> str:
        url = f"{self.base_url}/chat/completions"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last_err: Exception | None = None
        for attempt in range(self.retries):
            req = urllib.request.Request(url, data=body, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                if e.code == 429 or e.code >= 500:
                    last_err = e
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"HTTP {e.code} от {url}: {e.reason}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"API не ответил после {self.retries} попыток: {last_err}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_ai_providers.py -v`
Expected: PASS (12 tests)

- [ ] **Step 5: Commit**

```bash
git add ai_providers.py tests/test_ai_providers.py
git commit -m "feat: OpenAI-compatible provider for local and cloud endpoints"
```

---

### Task 4: AnthropicProvider — direct, batch, resume

**Files:**
- Modify: `ai_providers.py` (append)
- Test: `tests/test_ai_providers.py` (append)

**Interfaces:**
- Consumes: `ChunkJob`, `AIResult`, `EVENTS_SCHEMA` from Task 2.
- Produces: `AnthropicProvider(model, api_key, mode="batch", max_output_tokens=8000, concurrency=2, poll_interval=30, client=None)` with `analyze(jobs, on_result=None, resume_batch_id=None, on_batch_created=None) -> list[AIResult]`. `client=None` → lazy `import anthropic`; tests inject a stub. Used by Task 6.

- [ ] **Step 1: Write failing tests** (append to `tests/test_ai_providers.py`)

```python
from types import SimpleNamespace


def _fake_message(text=None, stop_reason="end_turn"):
    content = [SimpleNamespace(type="text", text=text if text is not None else _ok_content())]
    return SimpleNamespace(stop_reason=stop_reason, content=content)


class FakeAnthropicClient:
    """Stub matching the slice of the anthropic SDK we use."""

    def __init__(self, batch_results=None, direct_message=None):
        self.created_requests = None
        self.retrieve_calls = 0

        outer = self

        class Batches:
            def create(self, requests):
                outer.created_requests = requests
                return SimpleNamespace(id="batch_123")

            def retrieve(self, batch_id):
                outer.retrieve_calls += 1
                status = "in_progress" if outer.retrieve_calls == 1 else "ended"
                return SimpleNamespace(
                    processing_status=status,
                    request_counts=SimpleNamespace(processing=1),
                )

            def results(self, batch_id):
                return iter(batch_results or [])

        class Messages:
            def __init__(self):
                self.batches = Batches()

            def create(self, **params):
                outer.direct_params = params
                return direct_message or _fake_message()

        self.messages = Messages()


def _batch_item(custom_id, result_type="succeeded", message=None):
    if result_type == "succeeded":
        result = SimpleNamespace(type="succeeded", message=message or _fake_message())
    else:
        result = SimpleNamespace(type=result_type)
    return SimpleNamespace(custom_id=custom_id, result=result)


class TestAnthropicProviderDirect:
    def test_direct_success_and_params(self):
        client = FakeAnthropicClient()
        p = ap.AnthropicProvider(model="claude-sonnet-5", api_key="k", mode="direct",
                                 concurrency=1, client=client)
        results = p.analyze([ap.ChunkJob("chunk_000", "PROMPT", "h")])
        assert results[0].data["summary"] == "ok"
        params = client.direct_params
        assert params["model"] == "claude-sonnet-5"
        assert params["output_config"]["format"]["type"] == "json_schema"
        assert params["messages"][0]["content"] == "PROMPT"

    def test_refusal_is_error(self):
        client = FakeAnthropicClient(direct_message=_fake_message(stop_reason="refusal"))
        p = ap.AnthropicProvider(model="m", api_key="k", mode="direct", concurrency=1, client=client)
        results = p.analyze([ap.ChunkJob("chunk_000", "p", "h")])
        assert results[0].data is None
        assert "refusal" in results[0].error

    def test_max_tokens_truncation_is_error(self):
        client = FakeAnthropicClient(direct_message=_fake_message(stop_reason="max_tokens"))
        p = ap.AnthropicProvider(model="m", api_key="k", mode="direct", concurrency=1, client=client)
        results = p.analyze([ap.ChunkJob("chunk_000", "p", "h")])
        assert results[0].data is None
        assert "max_tokens" in results[0].error


class TestAnthropicProviderBatch:
    def test_batch_flow(self, monkeypatch):
        monkeypatch.setattr(ap.time, "sleep", lambda s: None)
        items = [_batch_item("chunk_000"), _batch_item("chunk_001", result_type="errored")]
        client = FakeAnthropicClient(batch_results=items)
        p = ap.AnthropicProvider(model="m", api_key="k", mode="batch", client=client)
        created = []
        results = p.analyze(
            [ap.ChunkJob("chunk_000", "p0", "h0"), ap.ChunkJob("chunk_001", "p1", "h1")],
            on_batch_created=created.append,
        )
        assert created == ["batch_123"]
        assert [r["custom_id"] for r in client.created_requests] == ["chunk_000", "chunk_001"]
        by_name = {r.name: r for r in results}
        assert by_name["chunk_000"].data is not None
        assert by_name["chunk_001"].error == "batch result: errored"

    def test_resume_skips_create(self, monkeypatch):
        monkeypatch.setattr(ap.time, "sleep", lambda s: None)
        client = FakeAnthropicClient(batch_results=[_batch_item("chunk_000")])
        p = ap.AnthropicProvider(model="m", api_key="k", mode="batch", client=client)
        results = p.analyze([], resume_batch_id="batch_old")
        assert client.created_requests is None  # no new batch
        assert results[0].name == "chunk_000"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_ai_providers.py -v -k Anthropic`
Expected: FAIL with `AttributeError: module 'ai_providers' has no attribute 'AnthropicProvider'`

- [ ] **Step 3: Implement** (append to `ai_providers.py`)

```python
class AnthropicProvider:
    """Anthropic Messages API: Batch mode (default, -50% cost) or direct requests."""

    def __init__(
        self,
        model: str,
        api_key: str,
        mode: str = "batch",
        max_output_tokens: int = 8000,
        concurrency: int = 2,
        poll_interval: int = 30,
        client: Any = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.mode = mode
        self.max_output_tokens = max_output_tokens
        self.concurrency = max(1, concurrency)
        self.poll_interval = poll_interval
        self.client = client  # tests inject a stub; real client is created lazily

    def _get_client(self) -> Any:
        if self.client is None:
            import anthropic  # lazy: openai_compatible works without the package

            self.client = anthropic.Anthropic(api_key=self.api_key)
        return self.client

    def analyze(
        self,
        jobs: list[ChunkJob],
        on_result: Callable[[AIResult], None] | None = None,
        resume_batch_id: str | None = None,
        on_batch_created: Callable[[str], None] | None = None,
    ) -> list[AIResult]:
        if self.mode == "batch":
            return self._analyze_batch(jobs, on_result, resume_batch_id, on_batch_created)
        return self._analyze_direct(jobs, on_result)

    def _request_params(self, job: ChunkJob) -> dict[str, Any]:
        return {
            "model": self.model,
            "max_tokens": self.max_output_tokens,
            "output_config": {"format": {"type": "json_schema", "schema": EVENTS_SCHEMA}},
            "messages": [{"role": "user", "content": job.prompt}],
        }

    def _result_from_message(self, name: str, msg: Any) -> AIResult:
        if msg.stop_reason == "refusal":
            return AIResult(name=name, error="модель отказалась отвечать (refusal)")
        if msg.stop_reason == "max_tokens":
            return AIResult(name=name, error="ответ обрезан (max_tokens); подними ai.max_output_tokens")
        text = next((b.text for b in msg.content if b.type == "text"), "")
        try:
            return AIResult(name=name, data=json.loads(text))
        except ValueError as e:
            return AIResult(name=name, error=f"битый JSON: {e}")

    # ── direct ──
    def _analyze_direct(
        self, jobs: list[ChunkJob], on_result: Callable[[AIResult], None] | None
    ) -> list[AIResult]:
        results: list[AIResult] = []
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            for res in pool.map(self._analyze_one_direct, jobs):
                if on_result:
                    on_result(res)
                results.append(res)
        return results

    def _analyze_one_direct(self, job: ChunkJob) -> AIResult:
        try:
            msg = self._get_client().messages.create(**self._request_params(job))
        except Exception as e:  # noqa: BLE001 — per-chunk soft fail, summary logged by caller
            return AIResult(name=job.name, error=str(e))
        return self._result_from_message(job.name, msg)

    # ── batch ──
    def _analyze_batch(
        self,
        jobs: list[ChunkJob],
        on_result: Callable[[AIResult], None] | None,
        resume_batch_id: str | None,
        on_batch_created: Callable[[str], None] | None,
    ) -> list[AIResult]:
        client = self._get_client()
        batch_id = resume_batch_id
        if batch_id is None:
            batch = client.messages.batches.create(
                requests=[{"custom_id": j.name, "params": self._request_params(j)} for j in jobs]
            )
            batch_id = batch.id
            if on_batch_created:
                on_batch_created(batch_id)
            logger.info(f"Batch создан: {batch_id} ({len(jobs)} чанков). Обычно готов в течение часа.")
        while True:
            status = client.messages.batches.retrieve(batch_id)
            if status.processing_status == "ended":
                break
            logger.info(f"Batch {batch_id}: ждём, в обработке {status.request_counts.processing}…")
            time.sleep(self.poll_interval)
        results: list[AIResult] = []
        for item in client.messages.batches.results(batch_id):
            if item.result.type == "succeeded":
                res = self._result_from_message(item.custom_id, item.result.message)
            else:
                res = AIResult(name=item.custom_id, error=f"batch result: {item.result.type}")
            if on_result:
                on_result(res)
            results.append(res)
        return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_ai_providers.py -v`
Expected: PASS (17 tests)

- [ ] **Step 5: Commit**

```bash
git add ai_providers.py tests/test_ai_providers.py
git commit -m "feat: Anthropic provider with batch mode, structured outputs, and resume"
```

---

### Task 5: State cache and job building

**Files:**
- Modify: `dnd_pipeline.py` (add near reports section, after `read_manual_results`)
- Test: `tests/test_ai_analysis.py` (append)

**Interfaces:**
- Consumes: `load_json`, `write_json`, `stable_hash`, `prompt_for_chunk`, `Paths` from `dnd_pipeline.py`; `ChunkJob` from `ai_providers`.
- Produces: `ai_state_path(paths: Paths) -> Path`, `load_ai_state(paths) -> dict`, `save_ai_state(paths, state) -> None`, `build_ai_jobs(chunk_paths: list[Path], paths: Paths, state: dict, force: bool) -> tuple[list[ChunkJob], int]` (jobs, skipped_count). Used by Task 6.

State file `out/cache/ai_state.json` shape:

```json
{
  "chunks": {"chunk_000": {"chunk_hash": "abc", "model": "m", "provider": "p", "status": "done"}},
  "pending_batch": {"batch_id": "batch_1", "provider": "anthropic", "model": "m", "jobs": {"chunk_000": "abc"}}
}
```

- [ ] **Step 1: Write failing tests** (append to `tests/test_ai_analysis.py`)

```python
import json


def _make_session(tmp_path, chunks):
    """Create minimal Paths with chunk files; returns (paths, chunk_paths)."""
    paths = dp.build_paths(tmp_path, "test")
    dp.ensure_dirs(paths)
    chunk_paths = []
    for i, payload in enumerate(chunks):
        p = paths.chunks_dir / f"chunk_{i:03d}.json"
        dp.write_json(p, payload)
        chunk_paths.append(p)
    return paths, chunk_paths


class TestAiState:
    def test_load_missing_returns_empty(self, tmp_path):
        paths, _ = _make_session(tmp_path, [])
        state = dp.load_ai_state(paths)
        assert state == {"chunks": {}, "pending_batch": None}

    def test_roundtrip(self, tmp_path):
        paths, _ = _make_session(tmp_path, [])
        state = {"chunks": {"chunk_000": {"chunk_hash": "x"}}, "pending_batch": None}
        dp.save_ai_state(paths, state)
        assert dp.load_ai_state(paths) == state


class TestBuildAiJobs:
    def test_all_new_chunks_become_jobs(self, tmp_path):
        paths, chunk_paths = _make_session(tmp_path, [{"a": 1}, {"a": 2}])
        jobs, skipped = dp.build_ai_jobs(chunk_paths, paths, {"chunks": {}}, force=False)
        assert [j.name for j in jobs] == ["chunk_000", "chunk_001"]
        assert skipped == 0
        assert jobs[0].prompt  # prompt_for_chunk produced text

    def test_skip_done_chunk_with_matching_hash(self, tmp_path):
        paths, chunk_paths = _make_session(tmp_path, [{"a": 1}])
        h = dp.stable_hash({"a": 1})
        dp.write_json(paths.manual_ai_dir / "chunk_000_events.json", {"summary": "done"})
        state = {"chunks": {"chunk_000": {"chunk_hash": h}}}
        jobs, skipped = dp.build_ai_jobs(chunk_paths, paths, state, force=False)
        assert jobs == []
        assert skipped == 1

    def test_manual_file_without_state_is_skipped(self, tmp_path):
        paths, chunk_paths = _make_session(tmp_path, [{"a": 1}])
        dp.write_json(paths.manual_ai_dir / "chunk_000_events.json", {"summary": "manual"})
        jobs, skipped = dp.build_ai_jobs(chunk_paths, paths, {"chunks": {}}, force=False)
        assert jobs == []
        assert skipped == 1

    def test_changed_chunk_hash_reruns(self, tmp_path):
        paths, chunk_paths = _make_session(tmp_path, [{"a": 2}])
        dp.write_json(paths.manual_ai_dir / "chunk_000_events.json", {"summary": "stale"})
        state = {"chunks": {"chunk_000": {"chunk_hash": "old-hash"}}}
        jobs, skipped = dp.build_ai_jobs(chunk_paths, paths, state, force=False)
        assert [j.name for j in jobs] == ["chunk_000"]

    def test_force_reruns_everything(self, tmp_path):
        paths, chunk_paths = _make_session(tmp_path, [{"a": 1}])
        h = dp.stable_hash({"a": 1})
        dp.write_json(paths.manual_ai_dir / "chunk_000_events.json", {"summary": "done"})
        state = {"chunks": {"chunk_000": {"chunk_hash": h}}}
        jobs, skipped = dp.build_ai_jobs(chunk_paths, paths, state, force=True)
        assert [j.name for j in jobs] == ["chunk_000"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_ai_analysis.py -v -k "AiState or BuildAiJobs"`
Expected: FAIL with `AttributeError: ... 'load_ai_state'`

- [ ] **Step 3: Implement** (add to `dnd_pipeline.py` after `read_manual_results`)

```python
# ──────────────────────────────────────────────────────────────
# AI analysis stage (API providers; manual mode stays the fallback)
# ──────────────────────────────────────────────────────────────


def ai_state_path(paths: Paths) -> Path:
    return paths.cache_dir / "ai_state.json"


def load_ai_state(paths: Paths) -> dict[str, Any]:
    p = ai_state_path(paths)
    if p.exists():
        state = load_json(p)
        state.setdefault("chunks", {})
        state.setdefault("pending_batch", None)
        return state
    return {"chunks": {}, "pending_batch": None}


def save_ai_state(paths: Paths, state: dict[str, Any]) -> None:
    write_json(ai_state_path(paths), state)


def build_ai_jobs(
    chunk_paths: list[Path], paths: Paths, state: dict[str, Any], force: bool
) -> tuple[list[Any], int]:
    from ai_providers import ChunkJob

    jobs: list[Any] = []
    skipped = 0
    for p in sorted(chunk_paths):
        name = p.stem
        chunk = load_json(p)
        h = stable_hash(chunk)
        out = paths.manual_ai_dir / f"{name}_events.json"
        if out.exists() and not force:
            entry = state.get("chunks", {}).get(name)
            # entry is None → файл положен руками, не трогаем.
            if entry is None or entry.get("chunk_hash") == h:
                skipped += 1
                continue
        jobs.append(ChunkJob(name=name, prompt=prompt_for_chunk(chunk), chunk_hash=h))
    return jobs, skipped
```

Note: `build_paths` requires `session_name` — tests use `"test"`; verify `_make_session` matches actual `build_paths(session_dir, session_name)` signature.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_ai_analysis.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dnd_pipeline.py tests/test_ai_analysis.py
git commit -m "feat: AI state cache and idempotent job building"
```

---

### Task 6: run_ai_analysis orchestration

**Files:**
- Modify: `dnd_pipeline.py` (after `build_ai_jobs`)
- Test: `tests/test_ai_analysis.py` (append)

**Interfaces:**
- Consumes: Tasks 1, 5; `normalize_json_text`; providers from `ai_providers`.
- Produces: `make_ai_provider(ai: dict, api_key: str | None) -> Any`, `run_ai_analysis(chunk_paths: list[Path], cfg: dict, paths: Paths, force: bool = False) -> bool` (True → AI stage ran, reports can be built; False → manual mode). Used by Task 7.

- [ ] **Step 1: Write failing tests** (append to `tests/test_ai_analysis.py`)

```python
from types import SimpleNamespace

from ai_providers import AIResult


class FakeProvider:
    def __init__(self, results, expect_resume=None):
        self.results = results
        self.expect_resume = expect_resume
        self.seen_jobs = None
        self.got_resume = "NOT_CALLED"

    def analyze(self, jobs, on_result=None, resume_batch_id=None, on_batch_created=None):
        self.seen_jobs = jobs
        self.got_resume = resume_batch_id
        if on_batch_created:
            on_batch_created("batch_new")
        for r in self.results:
            if on_result:
                on_result(r)
        return self.results


def _ai_cfg(**kw):
    ai = {"enabled": True}
    ai.update(kw)
    return {"session_name": "test", "ai": ai}


class TestRunAiAnalysis:
    def test_disabled_returns_false(self, tmp_path):
        paths, chunk_paths = _make_session(tmp_path, [{"a": 1}])
        assert dp.run_ai_analysis(chunk_paths, {"session_name": "test"}, paths) is False

    def test_no_key_returns_false(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        paths, chunk_paths = _make_session(tmp_path, [{"a": 1}])
        assert dp.run_ai_analysis(chunk_paths, _ai_cfg(), paths) is False

    def test_writes_results_and_state(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        paths, chunk_paths = _make_session(tmp_path, [{"a": 1}])
        fake = FakeProvider([AIResult(name="chunk_000", data={"summary": "ok"})])
        monkeypatch.setattr(dp, "make_ai_provider", lambda ai, key: fake)
        assert dp.run_ai_analysis(chunk_paths, _ai_cfg(mode="direct"), paths) is True
        written = dp.load_json(paths.manual_ai_dir / "chunk_000_events.json")
        assert written == {"summary": "ok"}
        state = dp.load_ai_state(paths)
        assert state["chunks"]["chunk_000"]["chunk_hash"] == dp.stable_hash({"a": 1})
        assert state["pending_batch"] is None

    def test_batch_mode_saves_and_clears_pending(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        paths, chunk_paths = _make_session(tmp_path, [{"a": 1}])
        pending_snapshots = []
        fake = FakeProvider([AIResult(name="chunk_000", data={"summary": "ok"})])

        orig_save = dp.save_ai_state

        def spy_save(p, state):
            pending_snapshots.append(state.get("pending_batch"))
            orig_save(p, state)

        monkeypatch.setattr(dp, "save_ai_state", spy_save)
        monkeypatch.setattr(dp, "make_ai_provider", lambda ai, key: fake)
        dp.run_ai_analysis(chunk_paths, _ai_cfg(mode="batch"), paths)
        # first save: pending batch recorded; final save: cleared
        assert any(p and p["batch_id"] == "batch_new" for p in pending_snapshots)
        assert dp.load_ai_state(paths)["pending_batch"] is None

    def test_resume_pending_batch(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        paths, chunk_paths = _make_session(tmp_path, [{"a": 1}])
        # chunk_000 already has a result; pending batch exists for it
        dp.save_ai_state(paths, {
            "chunks": {},
            "pending_batch": {"batch_id": "batch_old", "provider": "anthropic",
                              "model": "claude-sonnet-5", "jobs": {"chunk_000": "h"}},
        })
        fake = FakeProvider([AIResult(name="chunk_000", data={"summary": "ok"})])
        monkeypatch.setattr(dp, "make_ai_provider", lambda ai, key: fake)
        dp.run_ai_analysis(chunk_paths, _ai_cfg(mode="batch"), paths)
        assert fake.got_resume == "batch_old"

    def test_failed_chunks_do_not_write_files(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        paths, chunk_paths = _make_session(tmp_path, [{"a": 1}, {"a": 2}])
        fake = FakeProvider([
            AIResult(name="chunk_000", data={"summary": "ok"}),
            AIResult(name="chunk_001", error="boom"),
        ])
        monkeypatch.setattr(dp, "make_ai_provider", lambda ai, key: fake)
        dp.run_ai_analysis(chunk_paths, _ai_cfg(mode="direct"), paths)
        assert (paths.manual_ai_dir / "chunk_000_events.json").exists()
        assert not (paths.manual_ai_dir / "chunk_001_events.json").exists()


class TestMakeAiProvider:
    def test_openai_compatible(self):
        ai = dp.resolve_ai_config(
            {"ai": {"provider": "openai_compatible", "base_url": "http://x/v1", "model": "llama3.1"}}
        )
        p = dp.make_ai_provider(ai, None)
        assert type(p).__name__ == "OpenAICompatProvider"
        assert p.model == "llama3.1"
        assert p.normalize is dp.normalize_json_text

    def test_anthropic(self):
        ai = dp.resolve_ai_config({"ai": {}})
        p = dp.make_ai_provider(ai, "sk-test")
        assert type(p).__name__ == "AnthropicProvider"
        assert p.mode == "batch"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_ai_analysis.py -v -k "RunAiAnalysis or MakeAiProvider"`
Expected: FAIL with `AttributeError: ... 'run_ai_analysis'`

- [ ] **Step 3: Implement** (add to `dnd_pipeline.py` after `build_ai_jobs`)

```python
def make_ai_provider(ai: dict[str, Any], api_key: str | None) -> Any:
    if ai["provider"] == "anthropic":
        from ai_providers import AnthropicProvider

        return AnthropicProvider(
            model=ai["model"],
            api_key=api_key,
            mode=ai["mode"],
            max_output_tokens=ai["max_output_tokens"],
            concurrency=ai["concurrency"],
        )
    from ai_providers import OpenAICompatProvider

    return OpenAICompatProvider(
        model=ai["model"],
        base_url=ai["base_url"],
        api_key=api_key,
        max_output_tokens=ai["max_output_tokens"],
        concurrency=ai["concurrency"],
        normalize=normalize_json_text,
    )


def run_ai_analysis(
    chunk_paths: list[Path], cfg: dict[str, Any], paths: Paths, force: bool = False
) -> bool:
    """Run chunks through the configured LLM provider. Returns False → manual mode."""
    ai = resolve_ai_config(cfg)
    if not ai["enabled"]:
        logger.info("AI-этап выключен (ai.enabled=false) — ручной режим.")
        return False
    api_key = resolve_ai_api_key(ai)
    if ai["provider"] == "anthropic" and not api_key:
        env_name = ai.get("api_key_env") or "ANTHROPIC_API_KEY"
        logger.warning(f"Нет API-ключа: переменная {env_name} пуста. Падаю в ручной режим.")
        return False

    state = load_ai_state(paths)
    pending = state.get("pending_batch")
    resume_batch_id = None
    if pending and pending.get("provider") == ai["provider"] and pending.get("model") == ai["model"]:
        resume_batch_id = pending["batch_id"]
        logger.info(f"Продолжаю незавершённый batch: {resume_batch_id}")

    jobs, skipped = build_ai_jobs(chunk_paths, paths, state, force)
    if skipped:
        logger.info(f"AI: пропущено {skipped} чанков — результаты уже есть")
    if not jobs and not resume_batch_id:
        logger.info("AI: все чанки уже посчитаны.")
        return True

    hash_by_name = {j.name: j.chunk_hash for j in jobs}
    if resume_batch_id and pending:
        hash_by_name.update(pending.get("jobs", {}))

    total = len(jobs) if jobs else len(hash_by_name)
    progress = {"n": 0}

    def on_result(res: Any) -> None:
        progress["n"] += 1
        if res.data is not None:
            write_json(paths.manual_ai_dir / f"{res.name}_events.json", res.data)
            state["chunks"][res.name] = {
                "chunk_hash": hash_by_name.get(res.name, ""),
                "model": ai["model"],
                "provider": ai["provider"],
                "status": "done",
            }
            save_ai_state(paths, state)
            logger.info(f"AI [{progress['n']}/{total}]: {res.name} готов")
        else:
            logger.warning(f"AI [{progress['n']}/{total}]: {res.name} ошибка: {res.error}")

    provider = make_ai_provider(ai, api_key)
    logger.info(f"AI-анализ: provider={ai['provider']}, model={ai['model']}, чанков в работе: {len(jobs)}")

    if ai["provider"] == "anthropic" and ai["mode"] == "batch":
        def on_batch_created(batch_id: str) -> None:
            state["pending_batch"] = {
                "batch_id": batch_id,
                "provider": ai["provider"],
                "model": ai["model"],
                "jobs": hash_by_name,
            }
            save_ai_state(paths, state)

        results = provider.analyze(
            jobs,
            on_result=on_result,
            resume_batch_id=resume_batch_id,
            on_batch_created=on_batch_created,
        )
        state["pending_batch"] = None
        save_ai_state(paths, state)
    else:
        results = provider.analyze(jobs, on_result=on_result)

    failed = [r for r in results if r.data is None]
    if failed:
        names = ", ".join(r.name for r in failed)
        logger.warning(
            f"AI: {len(failed)} чанков без результата: {names}. "
            f"Добить: python dnd_pipeline.py ai-analyze <session_dir>"
        )
    return True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_ai_analysis.py tests/test_ai_providers.py -v`
Expected: PASS (all)

- [ ] **Step 5: Run full suite (regression check)**

Run: `python -m pytest -q`
Expected: all green

- [ ] **Step 6: Commit**

```bash
git add dnd_pipeline.py tests/test_ai_analysis.py
git commit -m "feat: run_ai_analysis orchestration with progress, state, and batch resume"
```

---

### Task 7: CLI wiring, deps, docs

**Files:**
- Modify: `dnd_pipeline.py` — `cmd_run` (~line 1461), new `cmd_ai_analyze`, `main()` parser (~line 1527)
- Modify: `requirements.txt`, `config.example.json`, `README.md`
- Test: `tests/test_ai_analysis.py` (append)

**Interfaces:**
- Consumes: `run_ai_analysis`, `build_reports`, `load_cfg` from earlier tasks/existing code.
- Produces: subcommand `ai-analyze <session_dir> [--force]`; `cmd_run` auto-runs AI stage + reports when enabled.

- [ ] **Step 1: Write failing tests** (append to `tests/test_ai_analysis.py`)

```python
class TestCli:
    def _write_cfg(self, tmp_path):
        dp.write_json(tmp_path / "config.json", {"session_name": "test", "ai": {"enabled": True}})

    def test_ai_analyze_requires_chunks(self, tmp_path, capsys):
        self._write_cfg(tmp_path)
        rc = dp.main(["ai-analyze", str(tmp_path)])
        assert rc == 1  # понятная ошибка: чанков нет, сначала prepare-ai

    def test_ai_analyze_runs_analysis_and_reports(self, tmp_path, monkeypatch):
        self._write_cfg(tmp_path)
        paths, _ = _make_session(tmp_path, [{"a": 1}])
        calls = {}
        monkeypatch.setattr(dp, "run_ai_analysis",
                            lambda chunk_paths, cfg, paths, force=False: calls.update(
                                {"chunks": [p.name for p in chunk_paths], "force": force}) or True)
        monkeypatch.setattr(dp, "build_reports", lambda paths, cfg: calls.update({"reports": True}))
        rc = dp.main(["ai-analyze", str(tmp_path), "--force"])
        assert rc == 0
        assert calls["chunks"] == ["chunk_000.json"]
        assert calls["force"] is True
        assert calls["reports"] is True

    def test_ai_analyze_manual_mode_skips_reports(self, tmp_path, monkeypatch):
        self._write_cfg(tmp_path)
        paths, _ = _make_session(tmp_path, [{"a": 1}])
        monkeypatch.setattr(dp, "run_ai_analysis", lambda *a, **k: False)
        called = {"reports": False}
        monkeypatch.setattr(dp, "build_reports", lambda paths, cfg: called.update({"reports": True}))
        rc = dp.main(["ai-analyze", str(tmp_path)])
        assert rc == 0
        assert called["reports"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_ai_analysis.py -v -k Cli`
Expected: FAIL — `ai-analyze` is not a known subcommand (argparse SystemExit → check: argparse errors raise SystemExit(2); `dp.main` catches only inside `args.func`. Test expectation: `pytest.raises(SystemExit)` — adjust first test accordingly if needed.)

- [ ] **Step 3: Implement CLI changes in `dnd_pipeline.py`**

Replace the tail of `cmd_run`:

```python
    chunk_paths = make_chunks(clean, cfg, paths)
    make_prompts(chunk_paths, paths)
    if run_ai_analysis(chunk_paths, cfg, paths):
        build_reports(paths, cfg)
        logger.info("\nГотово: AI-анализ прошёл, отчёты в reports/.")
    else:
        logger.info("\nГотово. Дальше: открывай prompts/*.md, вручную прогоняй через AI и складывай JSON-ответы в manual_ai_results/.")
```

Add after `cmd_prepare_ai`:

```python
def cmd_ai_analyze(args: argparse.Namespace) -> None:
    session_dir, cfg, paths = load_cfg(args)
    chunk_paths = sorted(paths.chunks_dir.glob("chunk_*.json"))
    if not chunk_paths:
        raise RuntimeError(
            f"Нет чанков в {paths.chunks_dir}. Сначала: python dnd_pipeline.py prepare-ai {session_dir}"
        )
    if run_ai_analysis(chunk_paths, cfg, paths, force=getattr(args, "force", False)):
        build_reports(paths, cfg)
```

Add to `main()` after the `prepare-ai` parser block:

```python
    p_aa = sub.add_parser("ai-analyze", parents=[verbosity], help="run AI analysis over chunks via API, then build reports")
    p_aa.add_argument("session_dir", help="folder with session files")
    p_aa.add_argument("--config", help="path to config.json")
    p_aa.add_argument("--force", action="store_true", help="recompute all chunks even if results exist")
    p_aa.add_argument("--quality-profile", choices=["gentle", "balanced", "aggressive"], help="override quality_profile")
    p_aa.set_defaults(func=cmd_ai_analyze)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_ai_analysis.py -v`
Expected: PASS

- [ ] **Step 5: Update deps and docs**

`requirements.txt` — append:

```
anthropic>=0.75  # AI-этап (provider=anthropic); openai_compatible работает без него
```

`config.example.json` — add section (after `"quality_profile"`):

```json
  "ai": {
    "enabled": true,
    "provider": "anthropic",
    "model": "claude-sonnet-5",
    "mode": "batch",
    "base_url": null,
    "api_key_env": null,
    "max_output_tokens": 8000,
    "concurrency": 2
  },
```

`README.md` — in the pipeline-flow section replace the manual-AI description with: AI-этап автоматический через API (`ai.enabled`), провайдеры `anthropic` (Batch API по умолчанию, ключ в `ANTHROPIC_API_KEY`) и `openai_compatible` (локальные Ollama/LM Studio через `base_url`); ручной режим остаётся fallback'ом; повторный запуск/докачка — `python dnd_pipeline.py ai-analyze <session_dir>`; пересчёт другой моделью — `--force`. Add config field table rows for every `ai.*` key.

- [ ] **Step 6: Full suite + smoke**

Run: `python -m pytest -q`
Expected: all green

Run: `python dnd_pipeline.py --help && python dnd_pipeline.py ai-analyze --help`
Expected: `ai-analyze` listed with `--force`

- [ ] **Step 7: Commit**

```bash
git add dnd_pipeline.py requirements.txt config.example.json README.md tests/test_ai_analysis.py
git commit -m "feat: wire AI analysis into run and new ai-analyze subcommand"
```

---

## Self-Review Notes

- Spec coverage: config (T1), providers incl. local (T2–T4), structured outputs (T4), idempotency + batch resume (T5–T6), fallback/manual mode (T6), CLI + full-run integration (T7), deps/docs (T7), tests everywhere. No gaps.
- Type consistency: `ChunkJob`/`AIResult`/`analyze(...)` signatures identical across Tasks 2–6; `run_ai_analysis(chunk_paths, cfg, paths, force=False) -> bool` matches T7 call sites.
- Manual-file protection: `build_ai_jobs` skips result files without state entries (T5 test `test_manual_file_without_state_is_skipped`).
