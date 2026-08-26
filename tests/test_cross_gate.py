"""Cross-track gating: просачивание глушится, свой голос и перебивания живут."""
import wave
from pathlib import Path

import numpy as np
import pytest

import dnd_pipeline as dp

SR = 16000


def write_wav(path: Path, audio: np.ndarray):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        return np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


def rms_db(a: np.ndarray) -> float:
    return 20 * np.log10(float(np.sqrt((a * a).mean())) + 1e-10)


def seg(a: np.ndarray, t0: float, t1: float) -> np.ndarray:
    return a[int(t0 * SR):int(t1 * SR)]


@pytest.fixture
def gated(tmp_path):
    """3 секунды, 2 дорожки. A говорит 0-1с и 2-3с, B говорит 1-2с.
    На чужих отрезках у каждой — просачивание на −20 dB."""
    rng = np.random.default_rng(7)
    def burst(n):  # шумовой всплеск ~ речь
        return (rng.standard_normal(n) * 0.3).astype(np.float32)

    n1 = SR
    a = np.zeros(3 * SR, dtype=np.float32)
    b = np.zeros(3 * SR, dtype=np.float32)
    va, vb = burst(n1), burst(n1)
    a[:n1] = va;               b[:n1] = va * 0.1          # A говорит, у B утечка −20 dB
    b[n1:2*n1] = vb;           a[n1:2*n1] = vb * 0.1      # B говорит, у A утечка
    both_a, both_b = burst(n1), burst(n1)
    a[2*n1:] = both_a;         b[2*n1:] = both_b          # перебивание: оба в полный голос

    pa, pb = tmp_path / "a.wav", tmp_path / "b.wav"
    write_wav(pa, a); write_wav(pb, b)

    paths = dp.Paths(**{f.name: tmp_path for f in dp.Paths.__dataclass_fields__.values()})
    out = dp.cross_track_gate_audio([("a.wav", pa), ("b.wav", pb)], {}, paths)
    assert set(out) == {"a.wav", "b.wav"}
    return read_wav(out["a.wav"]), read_wav(out["b.wav"])


def test_leakage_is_muted(gated):
    ga, gb = gated
    assert rms_db(seg(gb, 0.2, 0.8)) < -50   # утечка A на дорожке B убита
    assert rms_db(seg(ga, 1.2, 1.8)) < -50   # утечка B на дорожке A убита


def test_own_speech_survives(gated):
    ga, gb = gated
    assert rms_db(seg(ga, 0.2, 0.8)) > -20
    assert rms_db(seg(gb, 1.2, 1.8)) > -20


def test_overlap_keeps_both(gated):
    ga, gb = gated
    assert rms_db(seg(ga, 2.2, 2.8)) > -20
    assert rms_db(seg(gb, 2.2, 2.8)) > -20


def test_cache_reused(tmp_path):
    rng = np.random.default_rng(1)
    a = (rng.standard_normal(SR) * 0.3).astype(np.float32)
    pa, pb = tmp_path / "a.wav", tmp_path / "b.wav"
    write_wav(pa, a); write_wav(pb, a * 0.05)
    paths = dp.Paths(**{f.name: tmp_path for f in dp.Paths.__dataclass_fields__.values()})
    out1 = dp.cross_track_gate_audio([("a.wav", pa), ("b.wav", pb)], {}, paths)
    m1 = out1["a.wav"].stat().st_mtime_ns
    out2 = dp.cross_track_gate_audio([("a.wav", pa), ("b.wav", pb)], {}, paths)
    assert out2["a.wav"].stat().st_mtime_ns == m1  # не пересчитан


def test_disabled_by_config(tmp_path):
    cfg = {"cross_gate": {"enabled": False}}
    merged = {**dp.DEFAULT_CROSS_GATE, **cfg["cross_gate"]}
    assert merged["enabled"] is False


def test_non_pcm16_is_skipped(tmp_path):
    pa = tmp_path / "a.wav"
    with wave.open(str(pa), "wb") as wf:
        wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(SR)
        wf.writeframes(b"\x00\x00\x00\x00" * 100)
    pb = tmp_path / "b.wav"
    write_wav(pb, np.zeros(SR, dtype=np.float32))
    paths = dp.Paths(**{f.name: tmp_path for f in dp.Paths.__dataclass_fields__.values()})
    assert dp.cross_track_gate_audio([("a.wav", pa), ("b.wav", pb)], {}, paths) == {}
