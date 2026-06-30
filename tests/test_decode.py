"""Tests for decode-option resolution and profile decode defaults."""

import dnd_pipeline as dp


def test_threshold_from_decode_block_is_emitted():
    cfg = {"decode": {"compression_ratio_threshold": 2.2}}
    opts = dp.resolve_decode_options(cfg, "mlx")
    assert opts["compression_ratio_threshold"] == 2.2


def test_mlx_uses_logprob_threshold_name():
    cfg = {"decode": {"logprob_threshold": -0.8}}
    opts = dp.resolve_decode_options(cfg, "mlx")
    assert opts["logprob_threshold"] == -0.8
    assert "log_prob_threshold" not in opts


def test_faster_uses_log_prob_threshold_name():
    cfg = {"decode": {"logprob_threshold": -0.8}}
    opts = dp.resolve_decode_options(cfg, "faster_whisper")
    assert opts["log_prob_threshold"] == -0.8
    assert "logprob_threshold" not in opts


def test_none_valued_option_is_dropped():
    cfg = {"decode": {"initial_prompt": None}}
    opts = dp.resolve_decode_options(cfg, "mlx")
    assert "initial_prompt" not in opts


def test_initial_prompt_passed_when_set():
    cfg = {"decode": {"initial_prompt": "Играют: Ангрон, Шиян."}}
    opts = dp.resolve_decode_options(cfg, "mlx")
    assert opts["initial_prompt"] == "Играют: Ангрон, Шиян."


def test_legacy_top_level_condition_fallback():
    cfg = {"condition_on_previous_text": False}
    opts = dp.resolve_decode_options(cfg, "mlx")
    assert opts["condition_on_previous_text"] is False


def test_decode_block_overrides_legacy_condition():
    cfg = {"condition_on_previous_text": True, "decode": {"condition_on_previous_text": False}}
    opts = dp.resolve_decode_options(cfg, "mlx")
    assert opts["condition_on_previous_text"] is False


def test_false_condition_is_not_dropped():
    cfg = {"decode": {"condition_on_previous_text": False}}
    opts = dp.resolve_decode_options(cfg, "mlx")
    assert "condition_on_previous_text" in opts
    assert opts["condition_on_previous_text"] is False


def test_mlx_alias_normalization():
    cfg = {"decode": {"logprob_threshold": -1.0}}
    assert "logprob_threshold" in dp.resolve_decode_options(cfg, "mlx-whisper")
    assert "logprob_threshold" in dp.resolve_decode_options(cfg, "mlx_whisper")
