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


def test_balanced_profile_injects_decode_defaults():
    cfg = dp.apply_quality_profile({"quality_profile": "balanced"})
    assert cfg["decode"]["no_speech_threshold"] == 0.6
    assert cfg["decode"]["condition_on_previous_text"] is False


def test_aggressive_profile_is_stricter():
    cfg = dp.apply_quality_profile({"quality_profile": "aggressive"})
    assert cfg["decode"]["compression_ratio_threshold"] == 2.2
    assert cfg["decode"]["hallucination_silence_threshold"] == 1.0


def test_gentle_profile_is_more_lenient():
    cfg = dp.apply_quality_profile({"quality_profile": "gentle"})
    assert cfg["decode"]["hallucination_silence_threshold"] == 5.0
    assert cfg["decode"]["logprob_threshold"] == -1.2


def test_user_decode_override_beats_profile_default():
    cfg = dp.apply_quality_profile({
        "quality_profile": "balanced",
        "decode": {"no_speech_threshold": 0.9},
    })
    assert cfg["decode"]["no_speech_threshold"] == 0.9
    # untouched keys still come from the profile
    assert cfg["decode"]["compression_ratio_threshold"] == 2.4


def test_user_initial_prompt_survives_profile_merge():
    cfg = dp.apply_quality_profile({
        "quality_profile": "balanced",
        "decode": {"initial_prompt": "Партия: Ангрон, Шиян."},
    })
    assert cfg["decode"]["initial_prompt"] == "Партия: Ангрон, Шиян."


class _CaptureMLX:
    """Stub standing in for the mlx_whisper module."""
    def __init__(self):
        self.captured = None

    def transcribe(self, audio, **kwargs):
        self.captured = kwargs
        return {"segments": []}


def test_mlx_transcribe_forwards_decode_options(monkeypatch):
    stub = _CaptureMLX()
    monkeypatch.setattr(dp, "mlx_whisper", stub)
    cfg = dp.apply_quality_profile({
        "quality_profile": "balanced",
        "transcription_backend": "mlx",
        "language": "ru",
        "decode": {"initial_prompt": "Партия: Ангрон."},
    })
    dp.run_mlx_transcribe(__import__("pathlib").Path("x.wav"), cfg)
    assert stub.captured["condition_on_previous_text"] is False
    assert stub.captured["logprob_threshold"] == -1.0
    assert stub.captured["no_speech_threshold"] == 0.6
    assert stub.captured["initial_prompt"] == "Партия: Ангрон."
    assert "log_prob_threshold" not in stub.captured


class _FakeFWModel:
    def __init__(self):
        self.captured = None

    def transcribe(self, audio, **kwargs):
        self.captured = kwargs
        return [], object()


def test_faster_transcribe_forwards_decode_options():
    model = _FakeFWModel()
    cfg = dp.apply_quality_profile({
        "quality_profile": "aggressive",
        "transcription_backend": "faster_whisper",
        "language": "ru",
    })
    dp.run_faster_whisper_transcribe(model, __import__("pathlib").Path("x.wav"), cfg)
    assert model.captured["condition_on_previous_text"] is False
    assert model.captured["log_prob_threshold"] == -0.8
    assert "logprob_threshold" not in model.captured
    assert model.captured["word_timestamps"] is False


def test_cache_signature_includes_decode_options():
    cfg = dp.apply_quality_profile({"quality_profile": "balanced", "transcription_backend": "mlx"})
    sig = dp.transcription_cache_signature(cfg, "mlx", "a.wav", "fp123")
    assert "decode_options" in sig
    assert sig["decode_options"]["no_speech_threshold"] == 0.6


def test_cache_hash_changes_when_threshold_changes():
    base = dp.apply_quality_profile({"quality_profile": "balanced", "transcription_backend": "mlx"})
    tuned = dp.apply_quality_profile({
        "quality_profile": "balanced",
        "transcription_backend": "mlx",
        "decode": {"no_speech_threshold": 0.9},
    })
    h1 = dp.stable_hash(dp.transcription_cache_signature(base, "mlx", "a.wav", "fp"))
    h2 = dp.stable_hash(dp.transcription_cache_signature(tuned, "mlx", "a.wav", "fp"))
    assert h1 != h2
