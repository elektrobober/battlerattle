# tests/test_pdf_chronicle.py
"""Tests for the PDF chronicle stage: report data, synthesis, assets, staging."""
import json

import pytest

import dnd_pipeline as dp
from ai_providers import AIResult


def _results_fixture():
    return [
        {
            "chunk_index": 0, "summary": "Драка в таверне",
            "actions": [
                {"time": "00:10:00.000", "character": "Гай", "action": "напал", "outcome": "успех", "importance": "high"},
                {"time": "00:05:00.000", "character": "Ангрон", "action": "вошёл", "outcome": "эффектно", "importance": "low"},
            ],
            "dice_rolls": [
                {"time": "00:11:00.000", "character": "Гай", "roll_type": "attack", "die": "d20",
                 "natural": 20, "modifier": 5, "total": 25, "context": "атака", "confidence": "high", "raw_text": "нат 20"},
                {"time": "00:12:00.000", "character": "Гай", "roll_type": "save", "die": "d20",
                 "natural": 1, "modifier": 2, "total": 3, "context": "спас", "confidence": "high", "raw_text": "единица"},
                {"time": "00:13:00.000", "character": "Ангрон", "roll_type": "attack", "die": "d20",
                 "natural": None, "modifier": None, "total": 18, "context": "без ната", "confidence": "low", "raw_text": "18"},
            ],
            "mvp_signals": [
                {"time": "00:10:30.000", "character": "Гай", "category": "combat", "reason": "яркая атака", "weight": 2},
                {"time": "00:14:00.000", "character": "Ангрон", "category": "fun", "reason": "шутка", "weight": "1"},
            ],
        },
        {
            "chunk_index": 1, "summary": "Допрос пленника",
            "actions": [], "dice_rolls": [],
            "mvp_signals": [
                {"time": "00:40:00.000", "character": "Гай", "category": "social", "reason": "жёсткий ход", "weight": 1},
            ],
        },
    ]


class TestComputeReportData:
    def test_actions_sorted_by_time(self):
        data = dp.compute_report_data(_results_fixture(), {"session_name": "t"})
        assert [a["character"] for a in data["actions"]] == ["Ангрон", "Гай"]

    def test_dice_stats_avg_and_crits(self):
        data = dp.compute_report_data(_results_fixture(), {"session_name": "t"})
        gai = data["dice_stats"]["Гай"]
        assert gai["avg"] == pytest.approx(10.5)
        assert gai["count"] == 2
        assert gai["nat20"] == 1
        assert gai["nat1"] == 1
        # Ангрон: natural=None не считается
        assert "Ангрон" not in data["dice_stats"]

    def test_mvp_scores_and_categories(self):
        data = dp.compute_report_data(_results_fixture(), {"session_name": "t"})
        assert data["mvp_scores"] == {"Гай": 3, "Ангрон": 1}
        assert data["mvp_categories"]["Гай"] == {"combat": 2, "social": 1}

    def test_string_weight_normalized(self):
        data = dp.compute_report_data(_results_fixture(), {"session_name": "t"})
        angron = [e for e in data["mvp_events"] if e["character"] == "Ангрон"]
        assert angron[0]["weight"] == 1

    def test_summaries_in_chunk_order(self):
        data = dp.compute_report_data(_results_fixture(), {"session_name": "t"})
        assert data["summaries"] == [
            {"chunk_index": 0, "summary": "Драка в таверне"},
            {"chunk_index": 1, "summary": "Допрос пленника"},
        ]


class FakeSynthProvider:
    def __init__(self, result):
        self.result = result
        self.seen_prompt = None

    def analyze(self, jobs, on_result=None):
        self.seen_prompt = jobs[0].prompt
        return [self.result]


def _synthesis_ok():
    return {
        "recap": "Партия ворвалась в таверну.",
        "quest_hooks": [{"title": "Щит Морбрина", "description": "Разведать поселение."}],
        "scenes": [{"title": "Драка в таверне", "chunk_index": 0, "time": "00:10:00.000",
                    "image_prompt": "dark fantasy tavern brawl, hulking warrior"}],
    }


def _session_with_results(tmp_path, monkeypatch, provider):
    paths = dp.build_paths(tmp_path, "test")
    dp.ensure_dirs(paths)
    for i, res in enumerate(_results_fixture()):
        dp.write_json(paths.manual_ai_dir / f"chunk_{i:03d}_events.json", res)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(dp, "make_synthesis_provider", lambda ai, key: provider)
    return paths


class TestBuildSynthesisInput:
    def test_compact_payload(self):
        payload = dp.build_synthesis_input(_results_fixture(), {"session_name": "t"})
        assert payload["session"] == "t"
        assert len(payload["summaries"]) == 2
        assert all("reason" in e for e in payload["mvp_top"])
        assert all(a["importance"] == "high" for a in payload["key_actions"])


class TestSynthesisPrompt:
    def test_mentions_party_appearance(self):
        party = [{"name": "Ангрон", "appearance_en": "hulking scarred warrior"}]
        prompt = dp.synthesis_prompt({"session": "t", "summaries": [], "mvp_top": [], "key_actions": []}, party)
        assert "hulking scarred warrior" in prompt
        assert "recap" in prompt


class TestRunSessionSynthesis:
    def test_disabled_returns_none(self, tmp_path):
        paths = dp.build_paths(tmp_path, "test")
        dp.ensure_dirs(paths)
        assert dp.run_session_synthesis({"session_name": "test"}, paths, []) is None

    def test_writes_result_and_prompts(self, tmp_path, monkeypatch):
        provider = FakeSynthProvider(AIResult(name="synthesis", data=_synthesis_ok()))
        paths = _session_with_results(tmp_path, monkeypatch, provider)
        cfg = {"session_name": "test", "ai": {"enabled": True}}
        result = dp.run_session_synthesis(cfg, paths, [])
        assert result["recap"].startswith("Партия")
        saved = dp.load_json(paths.out_dir / "session_synthesis.json")
        assert saved == result
        md = (paths.out_dir / "image_prompts.md").read_text(encoding="utf-8")
        assert "dark fantasy tavern brawl" in md
        assert "scene1" in md  # инструкция об именах файлов

    def test_cache_skips_provider(self, tmp_path, monkeypatch):
        provider = FakeSynthProvider(AIResult(name="synthesis", data=_synthesis_ok()))
        paths = _session_with_results(tmp_path, monkeypatch, provider)
        cfg = {"session_name": "test", "ai": {"enabled": True}}
        dp.run_session_synthesis(cfg, paths, [])
        provider.seen_prompt = None
        result2 = dp.run_session_synthesis(cfg, paths, [])
        assert result2 is not None
        assert provider.seen_prompt is None  # второй раз в провайдера не ходили

    def test_force_recomputes(self, tmp_path, monkeypatch):
        provider = FakeSynthProvider(AIResult(name="synthesis", data=_synthesis_ok()))
        paths = _session_with_results(tmp_path, monkeypatch, provider)
        cfg = {"session_name": "test", "ai": {"enabled": True}}
        dp.run_session_synthesis(cfg, paths, [])
        provider.seen_prompt = None
        dp.run_session_synthesis(cfg, paths, [], force=True)
        assert provider.seen_prompt is not None

    def test_provider_error_returns_none(self, tmp_path, monkeypatch):
        provider = FakeSynthProvider(AIResult(name="synthesis", error="boom"))
        paths = _session_with_results(tmp_path, monkeypatch, provider)
        cfg = {"session_name": "test", "ai": {"enabled": True}}
        assert dp.run_session_synthesis(cfg, paths, []) is None
