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

    def test_no_property_uses_a_list_as_type(self):
        # Anthropic structured outputs only support a subset of JSON Schema
        # (basic types, enum, const, anyOf, allOf, $ref) — a "type" that is
        # itself a list (e.g. ["string", "null"]) isn't in that list.
        def walk(node):
            if isinstance(node, dict):
                assert not isinstance(node.get("type"), list)
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)
        walk(ap.EVENTS_SCHEMA)

    def test_nullable_dice_fields_use_anyof(self):
        props = ap.EVENTS_SCHEMA["properties"]["dice_rolls"]["items"]["properties"]
        for key, expected_type in (
            ("die", "string"), ("natural", "integer"),
            ("modifier", "integer"), ("total", "integer"),
        ):
            assert props[key] == {"anyOf": [{"type": expected_type}, {"type": "null"}]}


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


class TestSchemaParameterization:
    def test_parse_custom_required_fields(self):
        text = json.dumps({"recap": "r", "quest_hooks": [], "scenes": []})
        res = ap.parse_model_json("synthesis", text, lambda t: t,
                                  required_fields=("recap", "quest_hooks", "scenes"))
        assert res.error is None

    def test_parse_custom_required_fields_missing(self):
        res = ap.parse_model_json("synthesis", '{"recap": "r"}', lambda t: t,
                                  required_fields=("recap", "quest_hooks", "scenes"))
        assert "quest_hooks" in res.error

    def test_anthropic_custom_schema_in_params(self):
        client = FakeAnthropicClient()
        p = ap.AnthropicProvider(model="m", api_key="k", mode="direct", concurrency=1,
                                 client=client, schema=ap.SYNTHESIS_SCHEMA)
        p.analyze([ap.ChunkJob("synthesis", "p", "h")])
        assert client.direct_params["output_config"]["format"]["schema"] is ap.SYNTHESIS_SCHEMA

    def test_openai_compat_custom_required_fields(self, monkeypatch):
        payload = json.dumps({"recap": "r", "quest_hooks": [], "scenes": []})
        monkeypatch.setattr(ap.urllib.request, "urlopen",
                            lambda req, timeout=None: _fake_response(payload))
        p = ap.OpenAICompatProvider(model="m", base_url="http://x/v1", concurrency=1,
                                    required_fields=("recap", "quest_hooks", "scenes"))
        results = p.analyze([ap.ChunkJob("synthesis", "p", "h")])
        assert results[0].error is None


class TestSynthesisSchema:
    def test_structure(self):
        props = ap.SYNTHESIS_SCHEMA["properties"]
        assert set(ap.SYNTHESIS_SCHEMA["required"]) == {"recap", "quest_hooks", "scenes"}
        assert props["scenes"]["items"]["required"] == ["title", "chunk_index", "time", "image_prompt"]

    def test_all_objects_forbid_additional_properties(self):
        def walk(node):
            if isinstance(node, dict):
                if node.get("type") == "object":
                    assert node.get("additionalProperties") is False
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)
        walk(ap.SYNTHESIS_SCHEMA)


class TestRateLimitBackoff:
    def test_429_honors_retry_after_header(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr(ap.time, "sleep", sleeps.append)
        calls = {"n": 0}

        def fake_urlopen(req, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise urllib.error.HTTPError(req.full_url, 429, "rate limited",
                                             {"Retry-After": "17"}, io.BytesIO(b""))
            return _fake_response(_ok_content())

        monkeypatch.setattr(ap.urllib.request, "urlopen", fake_urlopen)
        p = ap.OpenAICompatProvider(model="m", base_url="http://x/v1", concurrency=1)
        results = p.analyze([ap.ChunkJob("chunk_000", "p", "h")])
        assert results[0].error is None
        assert sleeps[0] == 17

    def test_429_without_header_waits_long_not_seconds(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr(ap.time, "sleep", sleeps.append)
        calls = {"n": 0}

        def fake_urlopen(req, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise urllib.error.HTTPError(req.full_url, 429, "rate limited",
                                             {}, io.BytesIO(b""))
            return _fake_response(_ok_content())

        monkeypatch.setattr(ap.urllib.request, "urlopen", fake_urlopen)
        p = ap.OpenAICompatProvider(model="m", base_url="http://x/v1", concurrency=1)
        p.analyze([ap.ChunkJob("chunk_000", "p", "h")])
        assert sleeps[0] >= 20  # минутное TPM-окно, секундные ретраи бесполезны

    def test_500_keeps_fast_exponential_backoff(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr(ap.time, "sleep", sleeps.append)
        calls = {"n": 0}

        def fake_urlopen(req, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise urllib.error.HTTPError(req.full_url, 500, "boom", {}, io.BytesIO(b""))
            return _fake_response(_ok_content())

        monkeypatch.setattr(ap.urllib.request, "urlopen", fake_urlopen)
        p = ap.OpenAICompatProvider(model="m", base_url="http://x/v1", concurrency=1)
        p.analyze([ap.ChunkJob("chunk_000", "p", "h")])
        assert sleeps[0] == 1


class TestRateLimitBudget:
    """429 не должен съедать бюджет ретраев, отведённый под сетевые сбои и 5xx."""

    def _always_429(self, calls, ok_after, headers=None):
        def fake_urlopen(req, timeout=None):
            calls["n"] += 1
            if ok_after is None or calls["n"] <= ok_after:
                raise urllib.error.HTTPError(req.full_url, 429, "rate limited",
                                             headers or {}, io.BytesIO(b""))
            return _fake_response(_ok_content())
        return fake_urlopen

    def test_429_does_not_consume_5xx_retry_budget(self, monkeypatch):
        calls = {"n": 0}
        monkeypatch.setattr(ap.time, "sleep", lambda s: None)
        monkeypatch.setattr(ap.urllib.request, "urlopen", self._always_429(calls, ok_after=6))
        p = ap.OpenAICompatProvider(model="m", base_url="http://x/v1", concurrency=1, retries=2)
        results = p.analyze([ap.ChunkJob("chunk_000", "p", "h")])
        assert results[0].error is None
        assert calls["n"] == 7

    def test_429_reads_ratelimit_reset_tokens_header(self, monkeypatch):
        sleeps = []
        calls = {"n": 0}
        monkeypatch.setattr(ap.time, "sleep", sleeps.append)
        monkeypatch.setattr(ap.urllib.request, "urlopen",
                            self._always_429(calls, ok_after=1,
                                             headers={"x-ratelimit-reset-tokens": "6m0s"}))
        p = ap.OpenAICompatProvider(model="m", base_url="http://x/v1", concurrency=1)
        p.analyze([ap.ChunkJob("chunk_000", "p", "h")])
        assert sleeps[0] == 180  # окно 6 минут, ждём не дольше потолка

    def test_429_backoff_escalates_without_headers(self, monkeypatch):
        sleeps = []
        calls = {"n": 0}
        monkeypatch.setattr(ap.time, "sleep", sleeps.append)
        monkeypatch.setattr(ap.urllib.request, "urlopen", self._always_429(calls, ok_after=3))
        p = ap.OpenAICompatProvider(model="m", base_url="http://x/v1", concurrency=1)
        p.analyze([ap.ChunkJob("chunk_000", "p", "h")])
        assert sleeps == sorted(sleeps) and sleeps[0] < sleeps[-1]

    def test_429_gives_up_after_rate_limit_budget(self, monkeypatch):
        calls = {"n": 0}
        monkeypatch.setattr(ap.time, "sleep", lambda s: None)
        monkeypatch.setattr(ap.urllib.request, "urlopen", self._always_429(calls, ok_after=None))
        p = ap.OpenAICompatProvider(model="m", base_url="http://x/v1", concurrency=1,
                                    rate_limit_retries=4)
        results = p.analyze([ap.ChunkJob("chunk_000", "p", "h")])
        assert calls["n"] == 4
        assert "429" in results[0].error or "лимит" in results[0].error.lower()


class TestParseResetDuration:
    def test_formats(self):
        assert ap.parse_reset_duration("6m0s") == 360
        assert ap.parse_reset_duration("1.5s") == 1.5
        assert ap.parse_reset_duration("20ms") == 0.02
        assert ap.parse_reset_duration("1h2m3s") == 3723
        assert ap.parse_reset_duration("") is None
        assert ap.parse_reset_duration("garbage") is None


class TestRateLimitTinyHints:
    """OpenAI шлёт '6ms' в reset-заголовке — верить этому нельзя, 429 уже прилетел."""

    def _fake_429_then_ok(self, headers):
        calls = {"n": 0}

        def fake_urlopen(req, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise urllib.error.HTTPError(req.full_url, 429, "rate limited",
                                             headers, io.BytesIO(b""))
            return _fake_response(_ok_content())
        return fake_urlopen

    def test_sub_second_reset_header_ignored(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr(ap.time, "sleep", sleeps.append)
        monkeypatch.setattr(ap.urllib.request, "urlopen",
                            self._fake_429_then_ok({"x-ratelimit-reset-tokens": "6ms"}))
        p = ap.OpenAICompatProvider(model="m", base_url="http://x/v1", concurrency=1)
        p.analyze([ap.ChunkJob("chunk_000", "p", "h")])
        assert sleeps[0] >= 20

    def test_zero_retry_after_ignored(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr(ap.time, "sleep", sleeps.append)
        monkeypatch.setattr(ap.urllib.request, "urlopen",
                            self._fake_429_then_ok({"Retry-After": "0"}))
        p = ap.OpenAICompatProvider(model="m", base_url="http://x/v1", concurrency=1)
        p.analyze([ap.ChunkJob("chunk_000", "p", "h")])
        assert sleeps[0] >= 20


def _raw_response(payload: bytes):
    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    return Resp(payload)


class _FakeOpenAIBatchServer:
    """Мини-сервер Batch API: files → batches → poll → content."""

    def __init__(self, statuses=None, output_lines=None, error_lines=None):
        self.statuses = list(statuses or ["completed"])
        self.output_lines = output_lines
        self.error_lines = error_lines
        self.uploaded: list[bytes] = []
        self.created: list[dict] = []
        self.polls = 0

    def _ok_line(self, custom_id):
        return {
            "custom_id": custom_id,
            "response": {"status_code": 200,
                         "body": {"choices": [{"message": {"content": _ok_content()}}]}},
            "error": None,
        }

    def urlopen(self, req, timeout=None):
        url, method = req.full_url, req.get_method()
        if url.endswith("/files") and method == "POST":
            self.uploaded.append(req.data)
            return _raw_response(json.dumps({"id": "file-in"}).encode())
        if url.endswith("/batches") and method == "POST":
            self.created.append(json.loads(req.data.decode()))
            return _raw_response(json.dumps({"id": f"batch_{len(self.created)}",
                                             "status": "validating"}).encode())
        if "/batches/" in url and method == "GET":
            self.polls += 1
            status = self.statuses[min(self.polls - 1, len(self.statuses) - 1)]
            body = {"id": url.rsplit("/", 1)[-1], "status": status,
                    "request_counts": {"total": 2, "completed": 2, "failed": 0}}
            if status == "completed":
                body["output_file_id"] = "file-out"
                if self.error_lines:
                    body["error_file_id"] = "file-err"
            return _raw_response(json.dumps(body).encode())
        if "/files/file-out/content" in url:
            lines = self.output_lines
            if lines is None:
                names = [json.loads(l)["custom_id"]
                         for l in self.uploaded[-1].decode().splitlines() if l.strip().startswith("{")]
                lines = [self._ok_line(n) for n in names]
            return _raw_response("\n".join(json.dumps(l) for l in lines).encode())
        if "/files/file-err/content" in url:
            return _raw_response("\n".join(json.dumps(l) for l in self.error_lines).encode())
        raise AssertionError(f"неожиданный запрос: {method} {url}")


def _batch_provider(**kw):
    kw.setdefault("model", "gpt-4.1")
    kw.setdefault("base_url", "https://api.openai.com/v1")
    kw.setdefault("api_key", "sk-test")
    kw.setdefault("poll_interval", 0)
    return ap.OpenAIBatchProvider(**kw)


class TestOpenAIBatchProvider:
    def _jobs(self, n=2, prompt="p"):
        return [ap.ChunkJob(f"chunk_{i:03d}", prompt, f"h{i}") for i in range(n)]

    def test_full_flow_uploads_creates_and_parses(self, monkeypatch):
        server = _FakeOpenAIBatchServer(statuses=["in_progress", "completed"])
        monkeypatch.setattr(ap.time, "sleep", lambda s: None)
        monkeypatch.setattr(ap.urllib.request, "urlopen", server.urlopen)
        seen = []
        results = _batch_provider().analyze(self._jobs(), on_result=seen.append)

        assert [r.name for r in results] == ["chunk_000", "chunk_001"]
        assert all(r.error is None for r in results)
        assert len(seen) == 2
        # входной JSONL — по строке на чанк, в формате Batch API
        lines = [json.loads(l) for l in server.uploaded[0].decode().splitlines()
                 if l.strip().startswith("{")]
        assert [l["custom_id"] for l in lines] == ["chunk_000", "chunk_001"]
        assert lines[0]["method"] == "POST"
        assert lines[0]["url"] == "/v1/chat/completions"
        assert lines[0]["body"]["model"] == "gpt-4.1"
        assert lines[0]["body"]["messages"][0]["content"] == "p"
        assert server.created[0] == {"input_file_id": "file-in",
                                     "endpoint": "/v1/chat/completions",
                                     "completion_window": "24h"}

    def test_reports_created_batch_id_for_resume(self, monkeypatch):
        server = _FakeOpenAIBatchServer()
        monkeypatch.setattr(ap.time, "sleep", lambda s: None)
        monkeypatch.setattr(ap.urllib.request, "urlopen", server.urlopen)
        created = []
        _batch_provider().analyze(self._jobs(), on_batch_created=created.append)
        assert created == ["batch_1"]

    def test_resume_does_not_create_new_batch(self, monkeypatch):
        server = _FakeOpenAIBatchServer(
            output_lines=[{"custom_id": "chunk_000",
                           "response": {"status_code": 200,
                                        "body": {"choices": [{"message": {"content": _ok_content()}}]}},
                           "error": None}])
        monkeypatch.setattr(ap.time, "sleep", lambda s: None)
        monkeypatch.setattr(ap.urllib.request, "urlopen", server.urlopen)
        results = _batch_provider().analyze(self._jobs(), resume_batch_id="batch_9")
        assert server.uploaded == [] and server.created == []
        assert [r.name for r in results] == ["chunk_000"]

    def test_failed_lines_become_chunk_errors(self, monkeypatch):
        server = _FakeOpenAIBatchServer(output_lines=[
            {"custom_id": "chunk_000",
             "response": {"status_code": 200,
                          "body": {"choices": [{"message": {"content": _ok_content()}}]}},
             "error": None},
            {"custom_id": "chunk_001", "response": None,
             "error": {"code": "server_error", "message": "boom"}},
        ])
        monkeypatch.setattr(ap.time, "sleep", lambda s: None)
        monkeypatch.setattr(ap.urllib.request, "urlopen", server.urlopen)
        results = _batch_provider().analyze(self._jobs())
        by_name = {r.name: r for r in results}
        assert by_name["chunk_000"].error is None
        assert "boom" in by_name["chunk_001"].error

    def test_terminal_failed_status_raises_clear_error(self, monkeypatch):
        server = _FakeOpenAIBatchServer(statuses=["failed"])
        monkeypatch.setattr(ap.time, "sleep", lambda s: None)
        monkeypatch.setattr(ap.urllib.request, "urlopen", server.urlopen)
        results = _batch_provider().analyze(self._jobs())
        assert all(r.data is None for r in results)
        assert "failed" in results[0].error

    def test_splits_jobs_by_token_budget(self, monkeypatch):
        server = _FakeOpenAIBatchServer()
        monkeypatch.setattr(ap.time, "sleep", lambda s: None)
        monkeypatch.setattr(ap.urllib.request, "urlopen", server.urlopen)
        jobs = self._jobs(3, prompt="x" * 30_000)  # ~10k токенов на чанк
        results = _batch_provider(token_budget=25_000, max_output_tokens=1000).analyze(jobs)
        assert len(server.created) == 2  # 2 чанка + 1 чанк
        assert len(results) == 3

    def test_single_job_over_budget_still_sent(self, monkeypatch):
        server = _FakeOpenAIBatchServer()
        monkeypatch.setattr(ap.time, "sleep", lambda s: None)
        monkeypatch.setattr(ap.urllib.request, "urlopen", server.urlopen)
        jobs = self._jobs(1, prompt="x" * 300_000)
        results = _batch_provider(token_budget=1000).analyze(jobs)
        assert len(server.created) == 1 and len(results) == 1


class TestStructuredOutput:
    """У OpenAI есть strict json_schema — с ним recap не может приехать массивом."""

    def test_schema_goes_into_response_format(self, monkeypatch):
        seen = {}

        def fake_urlopen(req, timeout=None):
            seen["body"] = json.loads(req.data.decode())
            return _fake_response(_ok_content())

        monkeypatch.setattr(ap.urllib.request, "urlopen", fake_urlopen)
        p = ap.OpenAICompatProvider(model="gpt-4.1", base_url="https://api.openai.com/v1",
                                    concurrency=1, schema=ap.SYNTHESIS_SCHEMA,
                                    schema_name="synthesis")
        p.analyze([ap.ChunkJob("synthesis", "p", "h")])
        fmt = seen["body"]["response_format"]
        assert fmt["type"] == "json_schema"
        assert fmt["json_schema"]["strict"] is True
        assert fmt["json_schema"]["name"] == "synthesis"
        assert fmt["json_schema"]["schema"] == ap.SYNTHESIS_SCHEMA

    def test_without_schema_falls_back_to_json_object(self, monkeypatch):
        seen = {}

        def fake_urlopen(req, timeout=None):
            seen["body"] = json.loads(req.data.decode())
            return _fake_response(_ok_content())

        monkeypatch.setattr(ap.urllib.request, "urlopen", fake_urlopen)
        ap.OpenAICompatProvider(model="llama3.1", base_url="http://localhost:11434/v1",
                                concurrency=1).analyze([ap.ChunkJob("c", "p", "h")])
        assert seen["body"]["response_format"] == {"type": "json_object"}

    def test_batch_body_carries_schema(self, monkeypatch):
        server = _FakeOpenAIBatchServer()
        monkeypatch.setattr(ap.time, "sleep", lambda s: None)
        monkeypatch.setattr(ap.urllib.request, "urlopen", server.urlopen)
        _batch_provider(schema=ap.EVENTS_SCHEMA, schema_name="events").analyze(
            [ap.ChunkJob("chunk_000", "p", "h")])
        line = json.loads([l for l in server.uploaded[0].decode().splitlines()
                           if l.strip().startswith("{")][0])
        assert line["body"]["response_format"]["json_schema"]["schema"] == ap.EVENTS_SCHEMA

    def test_schemas_are_strict_compatible(self):
        """strict требует additionalProperties:false и все свойства в required."""
        def walk(node):
            if isinstance(node, dict):
                if node.get("type") == "object":
                    assert node.get("additionalProperties") is False
                    assert set(node.get("properties", {})) == set(node.get("required", []))
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)
        walk(ap.EVENTS_SCHEMA)
        walk(ap.SYNTHESIS_SCHEMA)
