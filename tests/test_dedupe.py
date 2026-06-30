"""Characterization tests for dedupe primitives and the dedupe pass."""

import dnd_pipeline as dp


def row(source, text, start, end, speaker=None, rms_db=-20, priority=50, character=None):
    speaker = speaker or source
    return {
        "source_file": source,
        "speaker": speaker,
        "character": character or speaker,
        "speaker_priority": priority,
        "start": start,
        "end": end,
        "start_hms": dp.fmt_hms_ms(start),
        "end_hms": dp.fmt_hms_ms(end),
        "text": text,
        "rms_db": rms_db,
        "rms_db_original": rms_db,
    }


DEDUPE_CFG = {"dedupe": {"enabled": True}}


# ── primitives ──────────────────────────────────────────────────

def test_text_similarity_identical_is_one():
    assert dp.text_similarity("привет мир", "Привет Мир") == 1.0


def test_overlap_ratio_full_containment():
    a = {"start": 0.0, "end": 2.0}
    b = {"start": 0.0, "end": 4.0}
    assert dp.overlap_ratio(a, b) == 1.0  # shorter (a) fully inside b


def test_overlap_ratio_no_overlap():
    a = {"start": 0.0, "end": 1.0}
    b = {"start": 5.0, "end": 6.0}
    assert dp.overlap_ratio(a, b) == 0.0


# ── is_probable_duplicate ───────────────────────────────────────

def test_same_source_never_duplicate():
    a = row("t1.wav", "одинаковый длинный текст реплики", 0, 3)
    b = row("t1.wav", "одинаковый длинный текст реплики", 0, 3)
    assert dp.is_probable_duplicate(a, b, DEDUPE_CFG) is False


def test_cross_source_overlapping_similar_is_duplicate():
    a = row("t1.wav", "одинаковый длинный текст реплики игрока", 0.0, 3.0)
    b = row("t2.wav", "одинаковый длинный текст реплики игрока", 0.2, 3.1)
    assert dp.is_probable_duplicate(a, b, DEDUPE_CFG) is True


def test_short_text_not_duplicate():
    a = row("t1.wav", "да", 0.0, 1.0)
    b = row("t2.wav", "да", 0.0, 1.0)
    assert dp.is_probable_duplicate(a, b, DEDUPE_CFG) is False


def test_non_overlapping_not_duplicate():
    a = row("t1.wav", "одинаковый длинный текст реплики игрока", 0.0, 3.0)
    b = row("t2.wav", "одинаковый длинный текст реплики игрока", 50.0, 53.0)
    assert dp.is_probable_duplicate(a, b, DEDUPE_CFG) is False


# ── choose_better ───────────────────────────────────────────────

def test_choose_louder_wins():
    a = row("t1.wav", "текст", 0, 3, rms_db=-30)
    b = row("t2.wav", "текст", 0, 3, rms_db=-20)  # +10 dB louder
    assert dp.choose_better(a, b, DEDUPE_CFG) is b


def test_choose_higher_priority_when_rms_close():
    a = row("t1.wav", "текст", 0, 3, rms_db=-20, priority=90)
    b = row("t2.wav", "текст", 0, 3, rms_db=-20, priority=50)
    assert dp.choose_better(a, b, DEDUPE_CFG) is a


# ── deduplicate ─────────────────────────────────────────────────

def test_deduplicate_collapses_cross_source_pair():
    rows = [
        row("t1.wav", "одинаковый длинный текст реплики игрока", 0.0, 3.0, rms_db=-30),
        row("t2.wav", "одинаковый длинный текст реплики игрока", 0.2, 3.1, rms_db=-20),
    ]
    clean = dp.deduplicate(rows, DEDUPE_CFG)
    assert len(clean) == 1
    winner = clean[0]
    assert winner["source_file"] == "t2.wav"          # louder kept
    assert len(winner["deduped_from"]) == 1            # quieter recorded as dup
    payload = winner["deduped_from"][0]
    assert set(payload) == {
        "source_file", "speaker", "character", "start", "end",
        "start_hms", "end_hms", "text", "rms_db", "rms_db_original",
    }
    assert payload["source_file"] == "t1.wav"          # the loser is recorded


def test_deduplicate_keeps_distinct_lines():
    rows = [
        row("t1.wav", "первая осмысленная реплика игрока", 0.0, 3.0),
        row("t2.wav", "совсем другая вторая реплика мастера", 10.0, 13.0),
    ]
    clean = dp.deduplicate(rows, DEDUPE_CFG)
    assert len(clean) == 2


def test_deduplicate_disabled_passthrough():
    rows = [row("t1.wav", "текст один", 0, 3), row("t2.wav", "текст один", 0, 3)]
    clean = dp.deduplicate(rows, {"dedupe": {"enabled": False}})
    assert len(clean) == 2
    assert all(r["duplicate_status"] == "original" for r in clean)
