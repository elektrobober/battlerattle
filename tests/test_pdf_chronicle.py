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


class TestPdfAssets:
    def test_resolve_pdf_config_defaults(self):
        pdf = dp.resolve_pdf_config({})
        assert pdf["enabled"] is True
        assert pdf["campaign_title"] == "Хроники кампании"

    def test_assets_dir_default_and_override(self, tmp_path):
        assert dp.pdf_assets_dir(tmp_path, dp.resolve_pdf_config({})) == tmp_path / "report_assets"
        custom = tmp_path / "campaign_assets"
        pdf = dp.resolve_pdf_config({"pdf": {"assets_dir": str(custom)}})
        assert dp.pdf_assets_dir(tmp_path, pdf) == custom

    def test_load_party_missing_returns_empty(self, tmp_path):
        assert dp.load_party(tmp_path) == []

    def test_load_party_reads_and_filters(self, tmp_path):
        dp.write_json(tmp_path / "party.json", [
            {"name": "Ангрон", "class_ru": "Воин", "player": "Дима", "ref": "ref.jpg"},
            {"class_ru": "без имени — отбрасываем"},
        ])
        party = dp.load_party(tmp_path)
        assert len(party) == 1
        assert party[0]["name"] == "Ангрон"

    def test_find_scene_images(self, tmp_path):
        (tmp_path / "scene1_tavern.png").write_bytes(b"x")
        (tmp_path / "scene3.jpg").write_bytes(b"x")
        (tmp_path / "ref.1 Гай.jpg").write_bytes(b"x")
        (tmp_path / "scene_bad.png").write_bytes(b"x")
        scenes = dp.find_scene_images(tmp_path)
        assert sorted(scenes) == [1, 3]
        assert scenes[1].name == "scene1_tavern.png"


class TestBuildPdfData:
    def _base_args(self, tmp_path):
        report_data = dp.compute_report_data(_results_fixture(), {"session_name": "t"})
        synthesis = _synthesis_ok()
        party = [{"name": "Ангрон", "class_ru": "Воин", "player": "Дима", "ref": "ref.jpg",
                  "appearance_en": "warrior"}]
        (tmp_path / "scene1.png").write_bytes(b"png")
        scene_images = {1: tmp_path / "scene1.png"}
        return report_data, synthesis, party, scene_images

    def test_full_data(self, tmp_path):
        report_data, synthesis, party, scenes = self._base_args(tmp_path)
        data = dp.build_pdf_data({"session_name": "t"}, dp.resolve_pdf_config({}),
                                 report_data, synthesis, party, scenes)
        assert data["session"] == "t"
        assert data["campaign_title"] == "Хроники кампании"
        assert data["recap"].startswith("Партия")
        assert data["scenes"][0]["file"] == "images/scene1.png"
        assert data["party"][0]["ref_file"] is None  # ref.jpg не существует на диске
        assert data["mvp_scores"][0] == {"character": "Гай", "score": 3}

    def test_no_synthesis_degrades(self, tmp_path):
        report_data, _, party, scenes = self._base_args(tmp_path)
        data = dp.build_pdf_data({"session_name": "t"}, dp.resolve_pdf_config({}),
                                 report_data, None, party, scenes)
        assert data["recap"] == ""
        assert data["quest_hooks"] == []
        assert data["scenes"] == []


class TestStagePdfBuild:
    def test_stages_everything(self, tmp_path):
        paths = dp.build_paths(tmp_path / "session", "t")
        dp.ensure_dirs(paths)
        assets = tmp_path / "assets"
        assets.mkdir()
        (assets / "scene1.png").write_bytes(b"png")
        (assets / "ref.jpg").write_bytes(b"jpg")
        template_dir = tmp_path / "tpl"
        template_dir.mkdir()
        (template_dir / "report.typ").write_text("#let data = json(\"data.json\")")

        party = [{"name": "Ангрон", "ref": "ref.jpg"}]
        scene_images = {1: assets / "scene1.png"}
        report_data = dp.compute_report_data(_results_fixture(), {"session_name": "t"})
        data = dp.build_pdf_data({"session_name": "t"}, dp.resolve_pdf_config({}),
                                 report_data, _synthesis_ok(), party, scene_images)

        build_dir = dp.stage_pdf_build(paths, data, party, scene_images, assets, template_dir)
        assert (build_dir / "report.typ").exists()
        assert (build_dir / "data.json").exists()
        assert (build_dir / "images" / "scene1.png").exists()
        assert (build_dir / "images" / "ref.jpg").exists()
        saved = dp.load_json(build_dir / "data.json")
        assert saved["party"][0]["ref_file"] == "images/ref.jpg"
