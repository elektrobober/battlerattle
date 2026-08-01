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
