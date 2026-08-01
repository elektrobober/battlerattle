"""Tests for the API-driven AI analysis stage in dnd_pipeline."""
import logging
from types import SimpleNamespace

import pytest

import dnd_pipeline as dp
from ai_providers import AIResult


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

    def test_resume_warns_about_chunks_not_in_pending_batch(self, tmp_path, monkeypatch, caplog):
        # Resuming a batch that only covered chunk_000 while a fresh
        # chunk_001 has shown up: the resumed batch can't include chunk_001,
        # so it must never be silently dropped — it should be flagged.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        paths, chunk_paths = _make_session(tmp_path, [{"a": 1}, {"a": 2}])
        dp.save_ai_state(paths, {
            "chunks": {},
            "pending_batch": {"batch_id": "batch_old", "provider": "anthropic",
                              "model": "claude-sonnet-5", "jobs": {"chunk_000": "h"}},
        })
        fake = FakeProvider([AIResult(name="chunk_000", data={"summary": "ok"})])
        monkeypatch.setattr(dp, "make_ai_provider", lambda ai, key: fake)

        with caplog.at_level(logging.INFO, logger="dnd_pipeline"):
            dp.run_ai_analysis(chunk_paths, _ai_cfg(mode="batch"), paths)

        messages = [r.message for r in caplog.records]
        assert (paths.manual_ai_dir / "chunk_000_events.json").exists()
        assert not (paths.manual_ai_dir / "chunk_001_events.json").exists()
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("chunk_001" in r.message for r in warnings)
        # chunk_000 got its result from the resumed batch — it must NOT be
        # listed as "left without result".
        assert not any("chunk_000" in r.message for r in warnings)
        assert any("1 чанков" in r.message for r in warnings)
        # progress counter should reflect the pending batch's job count (1),
        # not the freshly-built jobs list (2) — no "[N/2]" style logs.
        assert any("чанков в работе: 1" in m for m in messages)
        assert any("AI [1/1]: chunk_000" in m for m in messages)

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
