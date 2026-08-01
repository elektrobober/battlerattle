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
