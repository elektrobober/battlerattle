# tests/test_pdf_chronicle.py
"""Tests for the PDF chronicle stage: report data, synthesis, assets, staging."""
import pytest

import dnd_pipeline as dp


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
