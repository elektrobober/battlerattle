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
