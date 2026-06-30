"""Characterization tests for timing repair and merge."""

import dnd_pipeline as dp


def seg(source, text, start, end, speaker="A", character=None):
    return {
        "source_file": source,
        "speaker": speaker,
        "character": character or speaker,
        "start": start,
        "end": end,
        "start_hms": dp.fmt_hms_ms(start),
        "end_hms": dp.fmt_hms_ms(end),
        "text": text,
    }


# ── estimated_max_duration_for_text ─────────────────────────────

def test_short_text_capped_at_short_max():
    assert dp.estimated_max_duration_for_text("Привет", {}) == 4.0


def test_medium_text_capped_at_medium_max():
    text = "x" * 60
    assert dp.estimated_max_duration_for_text(text, {}) == 7.0


def test_long_text_uses_chars_per_sec_formula():
    text = "слово " * 40  # ~240 chars
    d = dp.estimated_max_duration_for_text(text, {})
    assert 7.0 < d <= 18.0  # between medium cap and long hard cap


# ── repair_segment_timings ──────────────────────────────────────

def test_caps_end_to_next_same_source():
    # Long text so the text-length cap (>5s) does not bind; next-source cap does.
    long_text = "слово " * 40
    rows = [
        seg("t1.wav", long_text, 0.0, 100.0),  # absurd overlong end
        seg("t1.wav", "вторая реплика", 5.0, 8.0),
    ]
    out = dp.repair_segment_timings(rows, {})
    first = [r for r in out if r["text"] == long_text][0]
    assert first["end"] == 5.0
    assert "cap_to_next_same_source" in first.get("timing_repair", [])


def test_caps_short_text_by_length():
    rows = [seg("t1.wav", "Привет", 0.0, 50.0)]  # 6 chars but 50s long
    out = dp.repair_segment_timings(rows, {})
    assert out[0]["end"] == 4.0  # short-text max
    assert "cap_by_text_length" in out[0].get("timing_repair", [])


def test_ensures_positive_duration():
    rows = [seg("t1.wav", "реплика", 5.0, 5.0)]
    out = dp.repair_segment_timings(rows, {})
    assert out[0]["end"] > 5.0


def test_disabled_repair_passthrough():
    rows = [seg("t1.wav", "Привет", 0.0, 50.0)]
    out = dp.repair_segment_timings(rows, {"postprocess": {"repair_timings": False}})
    assert out[0]["end"] == 50.0


# ── merge_adjacent_segments ─────────────────────────────────────

def test_merges_same_speaker_within_gap():
    rows = [
        seg("t1.wav", "первая часть", 0.0, 2.0, character="Маг"),
        seg("t1.wav", "вторая часть", 2.5, 4.0, character="Маг"),
    ]
    out = dp.merge_adjacent_segments(rows, {})
    assert len(out) == 1
    assert out[0]["text"] == "первая часть вторая часть"
    assert out[0]["end"] == 4.0


def test_does_not_merge_different_speakers():
    rows = [
        seg("t1.wav", "реплика мага", 0.0, 2.0, character="Маг"),
        seg("t2.wav", "реплика вора", 2.2, 4.0, character="Вор"),
    ]
    out = dp.merge_adjacent_segments(rows, {})
    assert len(out) == 2


def test_does_not_merge_across_large_gap():
    rows = [
        seg("t1.wav", "первая", 0.0, 2.0, character="Маг"),
        seg("t1.wav", "вторая", 30.0, 32.0, character="Маг"),
    ]
    out = dp.merge_adjacent_segments(rows, {})
    assert len(out) == 2
