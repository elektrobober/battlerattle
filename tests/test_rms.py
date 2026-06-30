"""Characterization tests for segment_rms_db.

Pin the current RMS-dBFS behavior across all PCM sample widths (8/16/24/32-bit)
so the decode can be sped up without changing results.
"""

import math
import wave

import pytest

import dnd_pipeline as dp


def write_const_wav(path, value, width, n_frames, frame_rate=16000):
    """Write a mono PCM WAV where every sample equals `value`.

    `value` is the signed logical sample (for 8-bit it is offset by +128 on disk,
    matching how `wave` stores unsigned 8-bit PCM).
    """
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(width)
        wf.setframerate(frame_rate)
        if width == 1:
            frame = bytes([(value + 128) & 0xFF])
        else:
            frame = (value & ((1 << (8 * width)) - 1)).to_bytes(width, "little", signed=False) \
                if value >= 0 else value.to_bytes(width, "little", signed=True)
        wf.writeframes(frame * n_frames)


def write_two_part_wav(path, v1, v2, width, n_each, frame_rate=16000):
    """First `n_each` frames = v1, next `n_each` frames = v2 (16-bit only helper)."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(width)
        wf.setframerate(frame_rate)
        f1 = v1.to_bytes(width, "little", signed=True)
        f2 = v2.to_bytes(width, "little", signed=True)
        wf.writeframes(f1 * n_each + f2 * n_each)


# Half-scale constant → RMS == |value| == max_abs/2 → 20*log10(0.5) ≈ -6.0206 dB
HALF_SCALE_DB = 20.0 * math.log10(0.5)


@pytest.mark.parametrize("width,value,max_abs", [
    (1, 64, 128),               # 8-bit unsigned: logical 64, max_abs 128
    (2, 16384, 32768),          # 16-bit
    (3, 1 << 22, 1 << 23),      # 24-bit
    (4, 1 << 30, 1 << 31),      # 32-bit
])
def test_half_scale_constant_all_widths(tmp_path, width, value, max_abs):
    p = tmp_path / f"const_{width}.wav"
    write_const_wav(p, value, width, n_frames=16000)
    db = dp.segment_rms_db(p, 0.0, 1.0)
    assert db == pytest.approx(HALF_SCALE_DB, abs=0.05)


def test_silence_returns_floor(tmp_path):
    p = tmp_path / "silent.wav"
    write_const_wav(p, 0, 2, n_frames=16000)
    assert dp.segment_rms_db(p, 0.0, 1.0) == -120.0


def test_slice_selects_subrange(tmp_path):
    # 0..1s loud, 1..2s silent. Querying only the loud half must read half-scale.
    p = tmp_path / "two.wav"
    write_two_part_wav(p, 16384, 0, width=2, n_each=16000)
    loud = dp.segment_rms_db(p, 0.0, 1.0)
    assert loud == pytest.approx(HALF_SCALE_DB, abs=0.05)


def test_slice_averages_across_range(tmp_path):
    # Querying both halves: mean_square = value^2 / 2 → -6.02 - 3.01 ≈ -9.03 dB
    p = tmp_path / "two.wav"
    write_two_part_wav(p, 16384, 0, width=2, n_each=16000)
    full = dp.segment_rms_db(p, 0.0, 2.0)
    assert full == pytest.approx(HALF_SCALE_DB - 20.0 * math.log10(math.sqrt(2)), abs=0.05)


def test_zero_length_range_returns_none(tmp_path):
    p = tmp_path / "const.wav"
    write_const_wav(p, 16384, 2, n_frames=16000)
    assert dp.segment_rms_db(p, 1.0, 1.0) is None


def test_unreadable_file_returns_none(tmp_path):
    p = tmp_path / "not_a.wav"
    p.write_text("this is not a wav file", encoding="utf-8")
    assert dp.segment_rms_db(p, 0.0, 1.0) is None


def test_missing_file_returns_none(tmp_path):
    assert dp.segment_rms_db(tmp_path / "nope.wav", 0.0, 1.0) is None
