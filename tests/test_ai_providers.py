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
