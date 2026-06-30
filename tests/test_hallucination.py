"""Characterization tests for the hallucination / garbage filter."""

import dnd_pipeline as dp


def hf_cfg(**overrides):
    hf = {"enabled": True}
    hf.update(overrides)
    return {"session_name": "s", "hallucination_filter": hf}


# ── is_repeated_word_garbage ────────────────────────────────────

def test_repeated_single_word_is_garbage():
    assert dp.is_repeated_word_garbage("кросс кросс кросс кросс") is True


def test_two_word_loop_is_garbage():
    assert dp.is_repeated_word_garbage("да нет да нет да нет") is True


def test_normal_sentence_is_not_garbage():
    assert dp.is_repeated_word_garbage("персонаж идёт по тёмному коридору осторожно") is False


def test_too_few_words_is_not_garbage():
    assert dp.is_repeated_word_garbage("кросс кросс") is False


# ── hallucination_reasons ───────────────────────────────────────

def test_blacklist_phrase_flagged():
    cfg = hf_cfg(blacklist=["продолжение следует"])
    row = {"text": "Продолжение следует...", "compression_ratio": 1.0, "rms_db": -10}
    assert "blacklist" in dp.hallucination_reasons(row, cfg)


def test_repeated_words_flagged():
    row = {"text": "кросс кросс кросс кросс", "compression_ratio": 1.0, "rms_db": -10}
    assert "repeated_words" in dp.hallucination_reasons(row, hf_cfg())


def test_very_high_compression_flagged():
    row = {"text": "длинная нормальная фраза тут", "compression_ratio": 9.0, "rms_db": -10}
    assert "very_high_compression" in dp.hallucination_reasons(row, hf_cfg())


def test_high_compression_flagged_for_long_text():
    row = {"text": "длинная нормальная фраза для проверки сжатия", "compression_ratio": 5.0, "rms_db": -10}
    assert "high_compression" in dp.hallucination_reasons(row, hf_cfg())


def test_low_rms_long_text_flagged():
    row = {"text": "это достаточно длинная фраза которую модель якобы услышала", "compression_ratio": 1.0, "rms_db": -90}
    assert "low_rms" in dp.hallucination_reasons(row, hf_cfg())


def test_short_reaction_with_low_rms_kept():
    # "да" at silence-level RMS must NOT be dropped (keep_short_reactions).
    row = {"text": "да", "compression_ratio": 1.0, "rms_db": -90}
    assert dp.hallucination_reasons(row, hf_cfg()) == []


def test_clean_row_has_no_reasons():
    row = {"text": "обычная осмысленная реплика игрока", "compression_ratio": 1.5, "rms_db": -20}
    assert dp.hallucination_reasons(row, hf_cfg()) == []


def test_disabled_filter_returns_no_reasons():
    row = {"text": "кросс кросс кросс кросс", "compression_ratio": 9.0, "rms_db": -90}
    assert dp.hallucination_reasons(row, hf_cfg(enabled=False)) == []


# ── filter_hallucinations ───────────────────────────────────────

def test_filter_splits_kept_and_rejected():
    rows = [
        {"text": "обычная реплика игрока про сюжет", "compression_ratio": 1.5, "rms_db": -20},
        {"text": "кросс кросс кросс кросс", "compression_ratio": 1.0, "rms_db": -10},
    ]
    kept = dp.filter_hallucinations(rows, hf_cfg())
    assert len(kept) == 1
    assert kept[0]["text"].startswith("обычная")
