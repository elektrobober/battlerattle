"""Tests for normalize_json_text Unicode handling."""

import unicodedata

import dnd_pipeline as dp

# Build the decomposed form explicitly so the source file's own encoding
# (which editors store as NFC) can't blur the distinction we are testing.
NFC_GAI = "Гай Гексан"                       # precomposed й (U+0439)
NFD_GAI = unicodedata.normalize("NFD", NFC_GAI)  # и (U+0438) + breve (U+0306)


def test_decomposed_input_differs_before_normalizing():
    # Sanity: the two byte sequences really are different until normalized.
    assert NFD_GAI != NFC_GAI
    assert "̆" in NFD_GAI


def test_nfc_collapses_decomposed_short_i():
    # AI copy-paste sometimes emits "й" as "и" + combining breve (NFD).
    # Without NFC, "Гай Гексан" splits into two distinct grouping keys.
    out = dp.normalize_json_text(NFD_GAI)
    assert "̆" not in out
    assert out == NFC_GAI


def test_nfc_makes_decomposed_and_precomposed_equal():
    assert dp.normalize_json_text(NFD_GAI) == dp.normalize_json_text(NFC_GAI)


def test_normalize_still_strips_fences_and_smart_quotes():
    raw = '```json\n{“character”: “Гай Гексан”}\n```'
    out = dp.normalize_json_text(raw)
    assert out == '{"character": "Гай Гексан"}'
