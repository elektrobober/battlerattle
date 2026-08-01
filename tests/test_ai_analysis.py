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
