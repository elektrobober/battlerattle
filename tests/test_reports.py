"""Characterization tests for build_reports (manual-AI → markdown)."""

import json

import dnd_pipeline as dp


def make_paths(tmp_path):
    paths = dp.build_paths(tmp_path, "sess")
    dp.ensure_dirs(paths)
    return paths


def test_build_reports_writes_all_four_files(tmp_path):
    paths = make_paths(tmp_path)
    result = {
        "session": "sess",
        "chunk_index": 0,
        "scene_type": "combat_or_rolls",
        "actions": [
            {"time": "0:00:01.000", "character": "Маг", "action": "кастует фаербол",
             "outcome": "попал", "importance": "high"},
        ],
        "dice_rolls": [
            {"time": "0:00:02.000", "character": "Маг", "roll_type": "attack",
             "natural": 18, "modifier": 5, "total": 23, "context": "атака", "confidence": "high"},
            {"time": "0:00:03.000", "character": "Маг", "roll_type": "save",
             "natural": 4, "modifier": 1, "total": 5, "context": "спас", "confidence": "high"},
        ],
        "mvp_signals": [
            {"time": "0:00:01.000", "character": "Маг", "category": "combat",
             "reason": "решил бой", "weight": 3},
        ],
        "summary": "маг затащил бой",
    }
    (paths.manual_ai_dir / "chunk_000.json").write_text(
        json.dumps(result, ensure_ascii=False), encoding="utf-8")

    dp.build_reports(paths, {"session_name": "sess"})

    timeline = (paths.reports_dir / "actions_timeline.md").read_text(encoding="utf-8")
    dice = (paths.reports_dir / "dice_stats.md").read_text(encoding="utf-8")
    mvp = (paths.reports_dir / "mvp_candidates.md").read_text(encoding="utf-8")
    session = (paths.reports_dir / "session_report.md").read_text(encoding="utf-8")

    assert "кастует фаербол" in timeline
    assert "Маг" in dice
    assert "11.00" in dice          # mean of natural 18 and 4
    assert "+3" in mvp              # weight applied
    assert "**Маг**: 3" in mvp      # score total
    assert "маг затащил бой" in session


def test_build_reports_handles_markdown_fenced_json(tmp_path):
    # Manual AI answers are often pasted wrapped in ```json fences + smart quotes.
    paths = make_paths(tmp_path)
    raw = '```json\n{“session”: “sess”, “chunk_index”: 1, ' \
          '“summary”: “тест фенсов”, “actions”: [], ' \
          '“dice_rolls”: [], “mvp_signals”: []}\n```'
    (paths.manual_ai_dir / "chunk_001.json").write_text(raw, encoding="utf-8")

    dp.build_reports(paths, {"session_name": "sess"})

    session = (paths.reports_dir / "session_report.md").read_text(encoding="utf-8")
    assert "тест фенсов" in session
