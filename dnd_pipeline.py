#!/usr/bin/env python3
"""
D&D PodTrak Pipeline.

Локальная обработка многодорожечной записи D&D-сессии:
- transcription через faster-whisper;
- raw JSONL;
- дедупликация;
- clean JSONL/TXT/SRT;
- chunks;
- markdown-промпты для ручного AI-анализа без API;
- диагностика качества raw/clean;
- сбор простых отчётов из ручных AI-ответов.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import unicodedata
import wave
from dataclasses import dataclass
from difflib import SequenceMatcher
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from faster_whisper import WhisperModel
except Exception:  # pragma: no cover
    WhisperModel = None

try:
    import mlx_whisper
except Exception:  # pragma: no cover
    mlx_whisper = None


logger = logging.getLogger("dnd_pipeline")


def configure_logging(verbose: bool, quiet: bool) -> int:
    """Install a single stdout handler on the module logger and set its level.

    verbose → DEBUG, quiet → WARNING, neither → INFO. verbose wins if both set.
    Idempotent: clears prior handlers so repeated calls (e.g. in tests) do not
    stack. Leaves propagation enabled so pytest caplog can capture records.
    """
    if verbose:
        level = logging.DEBUG
    elif quiet:
        level = logging.WARNING
    else:
        level = logging.INFO

    for h in list(logger.handlers):
        logger.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = True
    return level


# ──────────────────────────────────────────────────────────────
# Utils
# ──────────────────────────────────────────────────────────────


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_md(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_name(name: str) -> str:
    return re.sub(r"[^\w.а-яА-ЯёЁ-]+", "_", name, flags=re.UNICODE)


def file_fingerprint(path: Path) -> str:
    st = path.stat()
    raw = f"{path.name}:{st.st_size}:{int(st.st_mtime)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]




def stable_hash(data: Any) -> str:
    raw = json.dumps(data, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def apply_quality_profile(cfg: dict[str, Any]) -> dict[str, Any]:
    """Apply opinionated defaults for transcription quality.

    profile=gentle: бережём тихую речь, меньше режем bleed.
    profile=balanced: дефолт.
    profile=aggressive: сильнее давим чужие голоса, выше риск съесть тихие реплики.
    """
    profile = str(cfg.get("quality_profile", "balanced")).strip().lower()
    profiles: dict[str, dict[str, Any]] = {
        "gentle": {
            "preprocess": {
                "highpass_enabled": True, "highpass_hz": 70,
                "denoise_enabled": True, "denoise_noise_floor_db": -28,
                "noise_gate_enabled": True, "noise_gate_threshold_db": -52,
                "noise_gate_ratio": 3, "noise_gate_attack_ms": 8, "noise_gate_release_ms": 350,
                "loudnorm_enabled": False,
            },
            "dedupe": {
                "similarity_threshold": 0.88, "min_overlap_ratio": 0.55,
                "min_text_len": 24, "prefer_louder_by_db": 4.5,
            },
            "postprocess": {"repair_timings": True, "merge_adjacent_same_speaker": True, "merge_max_gap_sec": 0.9},
            "decode": {
                "condition_on_previous_text": False,
                "compression_ratio_threshold": 2.6,
                "logprob_threshold": -1.2,
                "no_speech_threshold": 0.7,
                "hallucination_silence_threshold": 5.0,
                "initial_prompt": None,
            },
        },
        "balanced": {
            "preprocess": {
                "highpass_enabled": True, "highpass_hz": 80,
                "denoise_enabled": True, "denoise_noise_floor_db": -25,
                "noise_gate_enabled": True, "noise_gate_threshold_db": -46,
                "noise_gate_ratio": 5, "noise_gate_attack_ms": 10, "noise_gate_release_ms": 280,
                "loudnorm_enabled": False,
            },
            "dedupe": {
                "similarity_threshold": 0.84, "min_overlap_ratio": 0.45,
                "min_text_len": 20, "prefer_louder_by_db": 3.0,
            },
            "postprocess": {"repair_timings": True, "merge_adjacent_same_speaker": True, "merge_max_gap_sec": 1.1},
            "decode": {
                "condition_on_previous_text": False,
                "compression_ratio_threshold": 2.4,
                "logprob_threshold": -1.0,
                "no_speech_threshold": 0.6,
                "hallucination_silence_threshold": 2.0,
                "initial_prompt": None,
            },
        },
        "aggressive": {
            "preprocess": {
                "highpass_enabled": True, "highpass_hz": 90,
                "denoise_enabled": True, "denoise_noise_floor_db": -22,
                "noise_gate_enabled": True, "noise_gate_threshold_db": -40,
                "noise_gate_ratio": 8, "noise_gate_attack_ms": 12, "noise_gate_release_ms": 220,
                "loudnorm_enabled": False,
            },
            "dedupe": {
                "similarity_threshold": 0.80, "min_overlap_ratio": 0.35,
                "min_text_len": 18, "prefer_louder_by_db": 2.0,
            },
            "postprocess": {"repair_timings": True, "merge_adjacent_same_speaker": True, "merge_max_gap_sec": 1.4},
            "decode": {
                "condition_on_previous_text": False,
                "compression_ratio_threshold": 2.2,
                "logprob_threshold": -0.8,
                "no_speech_threshold": 0.5,
                "hallucination_silence_threshold": 1.0,
                "initial_prompt": None,
            },
        },
    }
    if profile not in profiles:
        logger.warning(f"Предупреждение: неизвестный quality_profile={profile!r}; использую balanced")
        profile = "balanced"
    # User config overrides profile defaults.
    return deep_merge(profiles[profile], cfg)


# ──────────────────────────────────────────────────────────────
# AI analysis config
# ──────────────────────────────────────────────────────────────

AI_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "provider": "anthropic",          # "anthropic" | "openai_compatible"
    "model": "claude-sonnet-5",
    "mode": "batch",                  # anthropic only: "batch" | "direct"
    "base_url": None,                 # openai_compatible: e.g. http://localhost:11434/v1
    "api_key_env": None,              # env var name; default ANTHROPIC_API_KEY for anthropic
    "max_output_tokens": 8000,
    "concurrency": 2,
}


def resolve_ai_config(cfg: dict[str, Any]) -> dict[str, Any]:
    ai = deep_merge(AI_DEFAULTS, cfg.get("ai") or {})
    if ai["provider"] not in ("anthropic", "openai_compatible"):
        raise ValueError(f"Неизвестный ai.provider: {ai['provider']} (жду anthropic или openai_compatible)")
    if ai["provider"] == "openai_compatible" and not ai["base_url"]:
        raise ValueError("Для ai.provider=openai_compatible нужен ai.base_url (например http://localhost:11434/v1)")
    return ai


def resolve_ai_api_key(ai: dict[str, Any]) -> str | None:
    env_name = ai.get("api_key_env") or ("ANTHROPIC_API_KEY" if ai["provider"] == "anthropic" else None)
    return os.environ.get(env_name) if env_name else None


def normalize_json_text(text: str) -> str:
    """Fix common manual-AI JSON copy issues: smart quotes, fences, Unicode form.

    NFC normalization collapses decomposed sequences (e.g. "й" pasted as
    "и" + combining breve) so character names group under one key in reports.
    """
    text = unicodedata.normalize("NFC", text.strip())
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return (
        text.replace("“", '"').replace("”", '"')
            .replace("‘", "'").replace("’", "'")
            .replace("﻿", "")
            .strip()
    )


# ──────────────────────────────────────────────────────────────
# Audio preprocessing
# ──────────────────────────────────────────────────────────────


def build_ffmpeg_filter(pre_cfg: dict[str, Any]) -> str:
    """Build an ffmpeg audio-filter chain that does not change duration.

    Важно: здесь нет silenceremove/atrim. Мы не вырезаем куски аудио,
    а только подавляем/чистим сигнал, чтобы таймкоды остались исходными.
    """
    filters: list[str] = []

    if pre_cfg.get("highpass_enabled", True):
        filters.append(f"highpass=f={int(pre_cfg.get('highpass_hz', 80))}")

    if pre_cfg.get("lowpass_enabled", False):
        filters.append(f"lowpass=f={int(pre_cfg.get('lowpass_hz', 12000))}")

    # FFT denoise. Мягкое шумоподавление; слишком агрессивное может портить речь.
    if pre_cfg.get("denoise_enabled", True):
        filters.append(f"afftdn=nf={int(pre_cfg.get('denoise_noise_floor_db', -25))}")

    # Noise gate. НЕ режет время, а приглушает тихий сигнал.
    if pre_cfg.get("noise_gate_enabled", True):
        threshold = pre_cfg.get("noise_gate_threshold_db", -45)
        ratio = pre_cfg.get("noise_gate_ratio", 6)
        attack = pre_cfg.get("noise_gate_attack_ms", 10)
        release = pre_cfg.get("noise_gate_release_ms", 250)
        filters.append(
            "agate="
            f"threshold={threshold}dB:"
            f"ratio={ratio}:"
            f"attack={attack}:"
            f"release={release}"
        )

    # Loudnorm может помочь Whisper, но иногда меняет perceived bleed. По умолчанию выключено.
    if pre_cfg.get("loudnorm_enabled", False):
        i = pre_cfg.get("loudnorm_i", -18)
        tp = pre_cfg.get("loudnorm_tp", -2)
        lra = pre_cfg.get("loudnorm_lra", 11)
        filters.append(f"loudnorm=I={i}:TP={tp}:LRA={lra}")

    # aresample не режет длительность, просто приводит частоту.
    filters.append(f"aresample={int(pre_cfg.get('sample_rate', 16000))}")

    return ",".join(filters)


def preprocess_track_audio(session_dir: Path, track: dict[str, Any], config: dict[str, Any], paths: Paths) -> Path:
    """Return path to audio that should be sent to Whisper.

    If preprocessing is disabled, returns original path.
    If enabled, creates/reuses a preprocessed WAV with the same duration timeline.
    """
    input_path = session_dir / track["file"]
    pre_cfg = dict(config.get("preprocess", {}) or {})
    # Можно переопределить preprocess для конкретной дорожки:
    # {"file": "...wav", "preprocess": {"noise_gate_threshold_db": -42}}
    pre_cfg.update(track.get("preprocess", {}) or {})

    if not pre_cfg.get("enabled", False):
        return input_path

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "Включён preprocess.enabled, но ffmpeg не найден. Установи: brew install ffmpeg "
            "или выключи preprocess.enabled в config.json"
        )

    fingerprint = file_fingerprint(input_path)
    cfg_hash = stable_hash(pre_cfg)
    out_name = f"{safe_name(input_path.stem)}.{fingerprint}.{cfg_hash}.pre.wav"
    output_path = paths.preprocess_dir / out_name

    if output_path.exists() and output_path.stat().st_size > 0:
        logger.info(f"Preprocess кэш: {track.get('speaker') or track['file']} ← {output_path.name}")
        return output_path

    filters = build_ffmpeg_filter(pre_cfg)
    logger.info(f"Preprocess: {input_path.name}")
    logger.info(f"  ffmpeg filter: {filters}")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-map",
        "0:a:0",
        "-af",
        filters,
        "-ar",
        str(int(pre_cfg.get("sample_rate", 16000))),
        "-ac",
        str(int(pre_cfg.get("channels", 1))),
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg preprocess failed for {input_path.name}: {e}") from e

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg не создал файл: {output_path}")

    return output_path




def maybe_limit_audio(input_path: Path, track: dict[str, Any], config: dict[str, Any], paths: Paths) -> Path:
    """Create/reuse a short test audio file if limit_minutes is configured.

    This is for fast quality checks. It keeps timestamps relative to the beginning
    of the original recording because it cuts from 00:00, not from the middle.
    """
    limit_minutes = config.get("limit_minutes") or config.get("__limit_minutes")
    if not limit_minutes:
        return input_path

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("Для --limit-minutes нужен ffmpeg. Установи: brew install ffmpeg")

    limit_seconds = float(limit_minutes) * 60.0
    cfg_hash = stable_hash({"limit_seconds": limit_seconds, "input": input_path.name, "fp": file_fingerprint(input_path)})
    out_name = f"{safe_name(input_path.stem)}.{cfg_hash}.limit_{int(limit_seconds)}s.wav"
    output_path = paths.work_dir / out_name
    if output_path.exists() and output_path.stat().st_size > 0:
        logger.info(f"Limit кэш: {track.get('speaker') or track['file']} ← {output_path.name}")
        return output_path

    logger.info(f"Limit: беру первые {limit_minutes} мин из {input_path.name}")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(input_path),
        "-t", str(limit_seconds),
        "-c:a", "pcm_s16le",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    return output_path


def get_backend(config: dict[str, Any]) -> str:
    return str(config.get("transcription_backend") or config.get("backend") or "faster_whisper").strip().lower()


_MLX_BACKENDS = {"mlx", "mlx_whisper", "mlx-whisper"}


def resolve_decode_options(cfg: dict[str, Any], backend: str) -> dict[str, Any]:
    """Build backend-correct decode kwargs from cfg['decode'].

    Drops None values so the library default applies. Maps the logprob
    threshold to the backend's parameter name (faster-whisper uses
    `log_prob_threshold`, mlx uses `logprob_threshold`). Falls back to a
    legacy top-level `condition_on_previous_text` when the decode block omits it.
    """
    decode = dict(cfg.get("decode", {}) or {})
    is_mlx = str(backend).strip().lower() in _MLX_BACKENDS

    condition = decode.get("condition_on_previous_text", cfg.get("condition_on_previous_text"))
    logprob = decode.get("logprob_threshold")

    opts: dict[str, Any] = {
        "condition_on_previous_text": condition,
        "initial_prompt": decode.get("initial_prompt"),
        "compression_ratio_threshold": decode.get("compression_ratio_threshold"),
        "no_speech_threshold": decode.get("no_speech_threshold"),
        "hallucination_silence_threshold": decode.get("hallucination_silence_threshold"),
        ("logprob_threshold" if is_mlx else "log_prob_threshold"): logprob,
    }
    return {k: v for k, v in opts.items() if v is not None}


def normalize_segment_from_mlx(seg: dict[str, Any]) -> dict[str, Any]:
    return {
        "start": float(seg.get("start", 0.0) or 0.0),
        "end": float(seg.get("end", seg.get("start", 0.0) or 0.0) or 0.0),
        "text": (seg.get("text") or "").strip(),
        "avg_logprob": seg.get("avg_logprob"),
        "no_speech_prob": seg.get("no_speech_prob"),
        "compression_ratio": seg.get("compression_ratio"),
    }


def run_mlx_transcribe(audio_path: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    if mlx_whisper is None:
        raise RuntimeError("Не установлен mlx-whisper. Выполни: pip install mlx-whisper")

    model = config.get("mlx_model") or config.get("model_size") or "mlx-community/whisper-large-v3-turbo"
    language = config.get("language", "ru")
    logger.info(f"  backend: mlx-whisper, model: {model}")

    # mlx-whisper не использует faster-whisper VAD. Очистку делаем через preprocess/noise gate.
    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=model,
        language=language,
        verbose=False,
        **resolve_decode_options(config, "mlx"),
    )
    segments = result.get("segments") or []
    return [normalize_segment_from_mlx(s) for s in segments]


def run_faster_whisper_transcribe(model: Any, audio_path: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    kwargs = dict(
        language=config.get("language", "ru"),
        vad_filter=bool(config.get("use_vad", True)),
        vad_parameters={"min_silence_duration_ms": int(config.get("vad_min_silence_ms", 800))},
        beam_size=int(config.get("beam_size", 5)),
        word_timestamps=False,
    )
    kwargs.update(resolve_decode_options(config, "faster_whisper"))
    segments, info = model.transcribe(str(audio_path), **kwargs)
    rows = []
    for seg in segments:
        rows.append({
            "start": float(seg.start),
            "end": float(seg.end),
            "text": (seg.text or "").strip(),
            "avg_logprob": getattr(seg, "avg_logprob", None),
            "no_speech_prob": getattr(seg, "no_speech_prob", None),
            "compression_ratio": getattr(seg, "compression_ratio", None),
        })
    return rows

def fmt_hms(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    s = int(seconds)
    return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def fmt_hms_ms(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    millis = int(round((seconds - int(seconds)) * 1000))
    s = int(seconds)
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}.{millis:03d}"


def fmt_srt(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    millis = int(round((seconds - int(seconds)) * 1000))
    s = int(seconds)
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d},{millis:03d}"


@dataclass
class Paths:
    session_dir: Path
    out_dir: Path
    cache_dir: Path
    preprocess_dir: Path
    raw_dir: Path
    clean_dir: Path
    chunks_dir: Path
    prompts_dir: Path
    manual_ai_dir: Path
    reports_dir: Path
    work_dir: Path


def build_paths(session_dir: Path, session_name: str) -> Paths:
    out_dir = session_dir / "_dnd_pipeline_out" / session_name
    return Paths(
        session_dir=session_dir,
        out_dir=out_dir,
        cache_dir=out_dir / "cache",
        preprocess_dir=out_dir / "preprocessed",
        raw_dir=out_dir / "raw",
        clean_dir=out_dir / "clean",
        chunks_dir=out_dir / "chunks",
        prompts_dir=out_dir / "prompts",
        manual_ai_dir=out_dir / "manual_ai_results",
        reports_dir=out_dir / "reports",
        work_dir=out_dir / "work",
    )


def ensure_dirs(paths: Paths) -> None:
    for p in [paths.out_dir, paths.cache_dir, paths.preprocess_dir, paths.raw_dir, paths.clean_dir, paths.chunks_dir, paths.prompts_dir, paths.manual_ai_dir, paths.reports_dir, paths.work_dir]:
        p.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────
# Audio RMS for duplicate winner selection
# ──────────────────────────────────────────────────────────────


def _decode_pcm_le(raw: bytes, sample_width: int) -> tuple[Any, float]:
    """Decode little-endian PCM bytes to a float64 sample array + max_abs.

    Vectorized via numpy. Trailing bytes that don't fill a full sample are dropped.
    """
    if sample_width == 1:
        # 8-bit PCM is unsigned with a 128 offset.
        usable = raw
        samples = np.frombuffer(usable, dtype=np.uint8).astype(np.float64) - 128.0
        return samples, 128.0
    if sample_width == 2:
        usable = raw[: len(raw) - (len(raw) % 2)]
        samples = np.frombuffer(usable, dtype="<i2").astype(np.float64)
        return samples, float(1 << 15)
    if sample_width == 4:
        usable = raw[: len(raw) - (len(raw) % 4)]
        samples = np.frombuffer(usable, dtype="<i4").astype(np.float64)
        return samples, float(1 << 31)
    # 24-bit: numpy has no native int24, so build it from bytes.
    usable = raw[: len(raw) - (len(raw) % 3)]
    b = np.frombuffer(usable, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
    vals = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
    vals = np.where(vals >= (1 << 23), vals - (1 << 24), vals)
    return vals.astype(np.float64), float(1 << 23)


def segment_rms_db(path: Path, start: float, end: float) -> float | None:
    """Return approximate RMS dBFS for a WAV segment using stdlib wave.

    Works for PCM WAV. If file is not readable as WAV, returns None.
    """
    try:
        with wave.open(str(path), "rb") as wf:
            sample_width = wf.getsampwidth()
            frame_rate = wf.getframerate()
            nframes = wf.getnframes()

            if sample_width not in (1, 2, 3, 4):
                return None

            start_frame = max(0, min(nframes, int(start * frame_rate)))
            end_frame = max(start_frame, min(nframes, int(end * frame_rate)))
            frame_count = end_frame - start_frame
            if frame_count <= 0:
                return None

            wf.setpos(start_frame)
            raw = wf.readframes(frame_count)
    except (wave.Error, OSError, EOFError, ValueError):
        return None

    if not raw:
        return None

    samples, max_abs = _decode_pcm_le(raw, sample_width)
    if samples.size == 0:
        return None

    rms = float(np.sqrt(np.mean(samples * samples)))
    if rms <= 0:
        return -120.0
    return 20.0 * math.log10(rms / max_abs)


# ──────────────────────────────────────────────────────────────
# Transcription
# ──────────────────────────────────────────────────────────────


def transcription_cache_signature(
    config: dict[str, Any],
    backend: str,
    transcription_path_name: str,
    transcription_fingerprint: str,
    track_preprocess: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the dict whose stable_hash forms the per-track cache key."""
    return {
        "backend": backend,
        "model_size": config.get("model_size", "medium"),
        "mlx_model": config.get("mlx_model"),
        "language": config.get("language", "ru"),
        "use_vad": config.get("use_vad", True),
        "vad_min_silence_ms": config.get("vad_min_silence_ms", 800),
        "beam_size": config.get("beam_size", 5),
        "condition_on_previous_text": config.get("condition_on_previous_text", False),
        "decode_options": resolve_decode_options(config, backend),
        "preprocess": {**(config.get("preprocess", {}) or {}), **(track_preprocess or {})},
        "limit_minutes": config.get("limit_minutes") or config.get("__limit_minutes"),
        "transcription_file": transcription_path_name,
        "transcription_fingerprint": transcription_fingerprint,
    }


def transcribe_track(model: Any, session_dir: Path, track: dict[str, Any], config: dict[str, Any], paths: Paths) -> list[dict[str, Any]]:
    file_name = track["file"]
    speaker = track.get("speaker") or track.get("character") or file_name
    character = track.get("character") or speaker
    priority = int(track.get("priority", 50))
    audio_path = session_dir / file_name

    if not audio_path.exists():
        logger.warning(f"ПРОПУСК: файл не найден — {audio_path}")
        return []

    backend = get_backend(config)
    base_transcription_path = preprocess_track_audio(session_dir, track, config, paths)
    transcription_path = maybe_limit_audio(base_transcription_path, track, config, paths)

    fingerprint = file_fingerprint(audio_path)
    transcription_fingerprint = file_fingerprint(transcription_path)
    transcription_cfg_hash = stable_hash(transcription_cache_signature(
        config,
        backend,
        transcription_path.name,
        transcription_fingerprint,
        track.get("preprocess", {}) or {},
    ))
    cache_path = paths.cache_dir / f"{safe_name(file_name)}.{fingerprint}.{transcription_cfg_hash}.jsonl"
    if cache_path.exists():
        logger.info(f"Кэш: {speaker} ← {cache_path.name}")
        return read_jsonl(cache_path)

    logger.info(f"\n=== Транскрибирую: {speaker} ({file_name}) ===")
    if base_transcription_path != audio_path:
        logger.info(f"  preprocessed: {base_transcription_path.name}")
    if transcription_path != base_transcription_path:
        logger.info(f"  test/limit source: {transcription_path.name}")

    if backend in ("mlx", "mlx_whisper", "mlx-whisper"):
        raw_segments = run_mlx_transcribe(transcription_path, config)
    elif backend in ("faster_whisper", "faster-whisper", "faster"):
        raw_segments = run_faster_whisper_transcribe(model, transcription_path, config)
    else:
        raise RuntimeError(f"Неизвестный transcription_backend: {backend}")

    rows: list[dict[str, Any]] = []
    for idx, seg in enumerate(raw_segments):
        text = (seg.get("text") or "").strip()
        if not text:
            continue

        start = float(seg.get("start", 0.0) or 0.0)
        end = float(seg.get("end", start) or start)
        if end <= start:
            end = start + 0.01

        row = {
            "session": config["session_name"],
            "track_index": track.get("index"),
            "source_file": file_name,
            "transcription_backend": backend,
            "transcription_file": transcription_path.name,
            "audio_preprocessed": base_transcription_path != audio_path,
            "audio_limited": transcription_path != base_transcription_path,
            "speaker": speaker,
            "character": character,
            "speaker_priority": priority,
            "start": start,
            "end": end,
            "start_hms": fmt_hms_ms(start),
            "end_hms": fmt_hms_ms(end),
            "text": text,
            "avg_logprob": seg.get("avg_logprob"),
            "no_speech_prob": seg.get("no_speech_prob"),
            "compression_ratio": seg.get("compression_ratio"),
            "rms_db": segment_rms_db(base_transcription_path, start, end),
            "rms_db_original": segment_rms_db(audio_path, start, end),
            "duplicate_status": "raw",
            "deduped_from": [],
        }
        rows.append(row)
        logger.debug(f"  [{fmt_hms(start)}] {text}")

    write_jsonl(cache_path, rows)
    logger.info(f"  → {len(rows)} реплик. Кэш сохранён: {cache_path.name}")
    return rows


def discover_tracks(session_dir: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the track list, auto-discovering from session files if needed.

    If config has a non-empty `tracks`, it wins (returned unchanged). Otherwise
    glob `session_dir` for audio files and derive each track's speaker/character
    from the filename: strip the `{session_name}-` prefix, then split on the first
    ". " into speaker/character (a single-part label sets speaker == character).
    The `dm_speaker` track gets priority 100; everyone else 50. The `master_mix`
    file and any `exclude` entries are skipped.
    """
    explicit = config.get("tracks")
    if explicit:
        return explicit

    session_name = config.get("session_name", "")
    extensions = {e.lower() for e in config.get("audio_extensions", [".wav"])}
    excluded = set(config.get("exclude", []) or [])
    if config.get("master_mix"):
        excluded.add(config["master_mix"])
    dm_speaker = config.get("dm_speaker")
    prefix = f"{session_name}-"

    tracks: list[dict[str, Any]] = []
    for path in sorted(session_dir.iterdir(), key=lambda p: p.name):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        if path.name in excluded:
            continue
        label = path.stem
        if label.startswith(prefix):
            label = label[len(prefix):]
        if ". " in label:
            speaker, character = label.split(". ", 1)
        else:
            speaker = character = label
        tracks.append({
            "file": path.name,
            "speaker": speaker,
            "character": character,
            "priority": 100 if speaker == dm_speaker else 50,
        })
    if not tracks:
        logger.warning(
            f"Не найдено аудиодорожек в {session_dir} "
            f"(расширения: {sorted(extensions)}). Проверь папку, audio_extensions и exclude."
        )
    return tracks


def transcribe_all(session_dir: Path, config: dict[str, Any], paths: Paths) -> list[dict[str, Any]]:
    backend = get_backend(config)
    model = None

    if backend in ("faster_whisper", "faster-whisper", "faster"):
        if WhisperModel is None:
            raise RuntimeError("Не установлен faster-whisper. Выполни: pip install -r requirements.txt")
        logger.info(f"Загружаю faster-whisper модель {config.get('model_size', 'medium')} ({config.get('device', 'cpu')}, {config.get('compute_type', 'int8')})")
        model = WhisperModel(
            config.get("model_size", "medium"),
            device=config.get("device", "cpu"),
            compute_type=config.get("compute_type", "int8"),
        )
    elif backend in ("mlx", "mlx_whisper", "mlx-whisper"):
        if mlx_whisper is None:
            raise RuntimeError("Не установлен mlx-whisper. Выполни: pip install mlx-whisper")
        logger.info(f"Использую MLX backend для Apple Silicon: {config.get('mlx_model') or config.get('model_size')}")
        if config.get("use_vad", True):
            logger.info("  note: VAD faster-whisper в MLX не применяется; чистку делает preprocess/noise gate.")
    else:
        raise RuntimeError(f"Неизвестный transcription_backend: {backend}")

    rows: list[dict[str, Any]] = []
    tracks = discover_tracks(session_dir, config)
    for i, track in enumerate(tracks):
        track = dict(track)
        track["index"] = i
        rows.extend(transcribe_track(model, session_dir, track, config, paths))

    rows.sort(key=lambda r: (r["start"], r["end"], r.get("speaker", "")))
    suffix = "_test" if (config.get("limit_minutes") or config.get("__limit_minutes")) else ""
    raw_path = paths.raw_dir / f"{config['session_name']}{suffix}_raw.jsonl"
    write_jsonl(raw_path, rows)
    logger.info(f"\nRaw сохранён: {raw_path}")
    logger.info(f"Raw реплик: {len(rows)}")
    return rows




# ──────────────────────────────────────────────────────────────
# Hallucination / silence garbage filter
# ──────────────────────────────────────────────────────────────


def normalize_text_for_filter(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower().replace("ё", "е"))


def is_repeated_word_garbage(text: str) -> bool:
    """Detect chunks like 'кросс кросс кросс...' or repeated subtitle garbage."""
    norm = normalize_text_for_filter(text)
    words = re.findall(r"[a-zа-я0-9]+", norm, flags=re.IGNORECASE)
    if len(words) < 4:
        return False
    counts = Counter(words)
    top_word, top_count = counts.most_common(1)[0]
    if top_count / max(1, len(words)) >= 0.70:
        return True
    # Also catch alternating two-word loops.
    if len(counts) <= 2 and len(words) >= 6:
        return True
    return False


def is_blacklisted_hallucination(text: str, cfg: dict[str, Any]) -> bool:
    norm = normalize_text_for_filter(text)
    hf = cfg.get("hallucination_filter", {}) or {}
    for phrase in hf.get("blacklist", []) or []:
        if normalize_text_for_filter(str(phrase)) in norm:
            return True
    return False


def hallucination_reasons(row: dict[str, Any], cfg: dict[str, Any]) -> list[str]:
    hf = cfg.get("hallucination_filter", {}) or {}
    if not hf.get("enabled", True):
        return []

    text = (row.get("text") or "").strip()
    norm = normalize_text_for_filter(text)
    reasons: list[str] = []

    # Do not over-filter real short reactions such as 'Да', 'Нет', 'Угу'.
    keep_short = bool(hf.get("keep_short_reactions", True))
    short_max = int(hf.get("short_reaction_max_chars", 8))
    is_short = len(norm) <= short_max

    if hf.get("drop_blacklisted_phrases", True) and is_blacklisted_hallucination(text, cfg):
        reasons.append("blacklist")

    if hf.get("drop_repeated_words", True) and is_repeated_word_garbage(text):
        reasons.append("repeated_words")

    comp = row.get("compression_ratio")
    if hf.get("drop_high_compression", True) and isinstance(comp, (int, float)):
        max_comp = float(hf.get("max_compression_ratio", 4.0))
        very_high = float(hf.get("very_high_compression_ratio", 8.0))
        if comp >= very_high:
            reasons.append("very_high_compression")
        elif comp >= max_comp and not is_short:
            reasons.append("high_compression")

    rms = row.get("rms_db")
    if hf.get("drop_low_rms", True) and isinstance(rms, (int, float)):
        low_rms = float(hf.get("low_rms_db", -78.0))
        min_chars = int(hf.get("min_chars_for_low_rms_keep", 18))
        # If the model heard a whole phrase while RMS is basically silence,
        # it is usually a hallucination. Short reactions are preserved unless
        # they are blacklisted.
        if rms <= low_rms and (len(norm) >= min_chars or not keep_short):
            reasons.append("low_rms")

    return reasons


def filter_hallucinations(rows: list[dict[str, Any]], cfg: dict[str, Any], paths: Paths | None = None) -> list[dict[str, Any]]:
    hf = cfg.get("hallucination_filter", {}) or {}
    if not hf.get("enabled", True):
        return rows

    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in rows:
        reasons = hallucination_reasons(row, cfg)
        if reasons:
            bad = dict(row)
            bad["reject_reasons"] = reasons
            rejected.append(bad)
        else:
            kept.append(row)

    if paths is not None:
        suffix = "_test" if (cfg.get("limit_minutes") or cfg.get("__limit_minutes")) else ""
        rej_path = paths.raw_dir / f"{cfg['session_name']}{suffix}_rejected_hallucinations.jsonl"
        write_jsonl(rej_path, rejected)
        logger.info(f"Hallucination filter: удалено {len(rejected)} из {len(rows)} сегментов → {rej_path}")

    return kept



# ──────────────────────────────────────────────────────────────
# Timing repair
# ──────────────────────────────────────────────────────────────


def estimated_max_duration_for_text(text: str, cfg: dict[str, Any]) -> float:
    """Reasonable max duration for a transcript segment.

    MLX/Whisper sometimes returns an end timestamp far after a short phrase.
    We do not change start time or audio; we only cap JSON/SRT segment end.
    """
    post = cfg.get("postprocess", {}) or {}
    text = re.sub(r"\s+", " ", (text or "").strip())
    n = len(text)
    min_dur = float(post.get("timing_min_duration_sec", 0.8))
    padding = float(post.get("timing_padding_sec", 1.2))
    chars_per_sec = float(post.get("timing_chars_per_sec", 13.0))

    if n <= int(post.get("timing_short_text_chars", 40)):
        return float(post.get("timing_short_max_duration_sec", 4.0))
    if n <= int(post.get("timing_medium_text_chars", 90)):
        return float(post.get("timing_medium_max_duration_sec", 7.0))

    return max(min_dur, min(float(post.get("timing_long_max_duration_sec", 18.0)), n / max(1.0, chars_per_sec) + padding))


def repair_segment_timings(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Fix obviously overlong end timestamps without re-transcribing.

    Rules:
    - if a later segment from the same source starts before current end, cap current end;
    - if a very short text has an absurdly long duration, cap by text length;
    - never move start and never create end <= start.
    """
    post = cfg.get("postprocess", {}) or {}
    if not post.get("repair_timings", True):
        return rows

    by_source: dict[str, list[int]] = {}
    out = [dict(r) for r in rows]
    for i, r in enumerate(out):
        by_source.setdefault(str(r.get("source_file") or r.get("speaker") or "?"), []).append(i)

    min_duration = float(post.get("timing_min_duration_sec", 0.8))
    max_overlap_slack = float(post.get("timing_same_track_overlap_slack_sec", 0.05))
    changed = 0

    # First cap segments by next segment in the same source track.
    for _src, indexes in by_source.items():
        indexes.sort(key=lambda i: (float(out[i].get("start", 0.0)), float(out[i].get("end", 0.0))))
        for pos, i in enumerate(indexes[:-1]):
            cur = out[i]
            nxt = out[indexes[pos + 1]]
            start = float(cur.get("start", 0.0))
            end = float(cur.get("end", start))
            next_start = float(nxt.get("start", end))
            if next_start > start + min_duration and end > next_start + max_overlap_slack:
                cur["end"] = next_start
                cur["end_hms"] = fmt_hms_ms(next_start)
                cur.setdefault("timing_repair", []).append("cap_to_next_same_source")
                changed += 1

    # Then cap by text-length heuristic.
    for r in out:
        start = float(r.get("start", 0.0))
        end = float(r.get("end", start))
        duration = max(0.0, end - start)
        max_duration = estimated_max_duration_for_text(str(r.get("text") or ""), cfg)
        if duration > max_duration:
            new_end = start + max(min_duration, max_duration)
            r["end"] = new_end
            r["end_hms"] = fmt_hms_ms(new_end)
            r.setdefault("timing_repair", []).append("cap_by_text_length")
            changed += 1
        if float(r.get("end", start)) <= start:
            r["end"] = start + min_duration
            r["end_hms"] = fmt_hms_ms(r["end"])
            r.setdefault("timing_repair", []).append("ensure_positive_duration")
            changed += 1

    if changed:
        logger.info(f"Timing repair: исправлено end-таймкодов: {changed}")
    return out

# ──────────────────────────────────────────────────────────────
# Dedupe
# ──────────────────────────────────────────────────────────────


def text_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def overlap_ratio(a: dict[str, Any], b: dict[str, Any]) -> float:
    overlap = max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    shorter = min(a["end"] - a["start"], b["end"] - b["start"])
    if shorter <= 0:
        return 0.0
    return overlap / shorter


def is_probable_duplicate(a: dict[str, Any], b: dict[str, Any], cfg: dict[str, Any]) -> bool:
    if a.get("source_file") == b.get("source_file"):
        return False

    dedupe = cfg.get("dedupe", {})
    min_text_len = int(dedupe.get("min_text_len", 20))
    if len(a.get("text", "")) < min_text_len or len(b.get("text", "")) < min_text_len:
        return False

    if overlap_ratio(a, b) < float(dedupe.get("min_overlap_ratio", 0.45)):
        return False

    return text_similarity(a.get("text", ""), b.get("text", "")) >= float(dedupe.get("similarity_threshold", 0.84))


def choose_better(a: dict[str, Any], b: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Choose which duplicate to keep."""
    prefer_louder_by_db = float(cfg.get("dedupe", {}).get("prefer_louder_by_db", 3.0))
    a_db = a.get("rms_db")
    b_db = b.get("rms_db")
    if isinstance(a_db, (int, float)) and isinstance(b_db, (int, float)):
        if b_db - a_db >= prefer_louder_by_db:
            return b
        if a_db - b_db >= prefer_louder_by_db:
            return a

    # Higher priority can be used for DM track or known clean tracks.
    a_pr = int(a.get("speaker_priority", 50))
    b_pr = int(b.get("speaker_priority", 50))
    if abs(a_pr - b_pr) >= 20:
        return a if a_pr > b_pr else b

    # Longer recognized text often means less truncated segment.
    if len(b.get("text", "")) > len(a.get("text", "")) + 5:
        return b
    return a


def _dedup_payload(r: dict[str, Any]) -> dict[str, Any]:
    """Compact record of a segment kept in another winner's `deduped_from`."""
    return {
        "source_file": r["source_file"],
        "speaker": r["speaker"],
        "character": r.get("character"),
        "start": r["start"],
        "end": r["end"],
        "start_hms": r["start_hms"],
        "end_hms": r["end_hms"],
        "text": r["text"],
        "rms_db": r.get("rms_db"),
        "rms_db_original": r.get("rms_db_original"),
    }


def deduplicate(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    if not cfg.get("dedupe", {}).get("enabled", True):
        return [{**r, "duplicate_status": "original", "deduped_from": []} for r in rows]

    clean: list[dict[str, Any]] = []
    search_back = int(cfg.get("dedupe", {}).get("search_back_segments", 40))

    for row in rows:
        duplicate_index: int | None = None
        start = max(0, len(clean) - search_back)
        for i in range(start, len(clean)):
            if is_probable_duplicate(row, clean[i], cfg):
                duplicate_index = i
                break

        if duplicate_index is None:
            item = dict(row)
            item["duplicate_status"] = "original"
            item["deduped_from"] = []
            clean.append(item)
            continue

        existing = clean[duplicate_index]
        better = choose_better(existing, row, cfg)

        if better is row:
            new_item = dict(row)
            new_item["duplicate_status"] = "original"
            new_item["deduped_from"] = existing.get("deduped_from", []) + [_dedup_payload(existing)]
            clean[duplicate_index] = new_item
        else:
            existing.setdefault("deduped_from", []).append(_dedup_payload(row))

    clean.sort(key=lambda r: (r["start"], r["end"], r.get("speaker", "")))
    return clean


def merge_adjacent_segments(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    post = cfg.get("postprocess", {}) or {}
    if not post.get("merge_adjacent_same_speaker", True):
        return rows
    max_gap = float(post.get("merge_max_gap_sec", 1.1))
    max_merged_chars = int(post.get("merge_max_merged_chars", 420))
    merged: list[dict[str, Any]] = []
    for row in rows:
        if not merged:
            merged.append(dict(row))
            continue
        prev = merged[-1]
        same_character = (prev.get("character") or prev.get("speaker")) == (row.get("character") or row.get("speaker"))
        gap = float(row["start"]) - float(prev["end"])
        can_merge = (
            same_character
            and 0 <= gap <= max_gap
            and len(prev.get("text", "")) + len(row.get("text", "")) <= max_merged_chars
            and not row.get("deduped_from")
        )
        if can_merge:
            prev["end"] = row["end"]
            prev["end_hms"] = row["end_hms"]
            prev["text"] = (prev.get("text", "").rstrip() + " " + row.get("text", "").lstrip()).strip()
            prev.setdefault("merged_from", []).append({
                "start": row["start"], "end": row["end"], "text": row.get("text", ""),
                "source_file": row.get("source_file"), "speaker": row.get("speaker"),
            })
            # Preserve average-ish RMS conservatively: louder of the pieces.
            if isinstance(row.get("rms_db"), (int, float)) and isinstance(prev.get("rms_db"), (int, float)):
                prev["rms_db"] = max(prev["rms_db"], row["rms_db"])
        else:
            merged.append(dict(row))
    return merged


def speaker_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        name = r.get("character") or r.get("speaker") or "?"
        out[name] = out.get(name, 0) + 1
    return dict(sorted(out.items(), key=lambda x: x[0]))


def write_quality_report(raw: list[dict[str, Any]], clean: list[dict[str, Any]], cfg: dict[str, Any], paths: Paths) -> None:
    suffix = "_test" if (cfg.get("limit_minutes") or cfg.get("__limit_minutes")) else ""
    dup_count = sum(len(r.get("deduped_from", []) or []) for r in clean)
    low_conf = []
    for r in clean:
        avg = r.get("avg_logprob")
        nsp = r.get("no_speech_prob")
        comp = r.get("compression_ratio")
        suspicious = False
        if isinstance(avg, (int, float)) and avg < -1.2:
            suspicious = True
        if isinstance(nsp, (int, float)) and nsp > 0.65:
            suspicious = True
        if isinstance(comp, (int, float)) and comp > 2.6:
            suspicious = True
        if suspicious:
            low_conf.append(r)
    md = [
        f"# Quality report — {cfg['session_name']}{suffix}",
        "",
        f"- quality_profile: `{cfg.get('quality_profile', 'balanced')}`",
        f"- backend: `{get_backend(cfg)}`",
        f"- model: `{cfg.get('model_size')}`",
        f"- raw реплик: **{len(raw)}**",
        f"- clean реплик: **{len(clean)}**",
        f"- склеено/удалено дублей: **{max(0, len(raw) - len(clean))}**",
        f"- записей в deduped_from: **{dup_count}**",
        f"- подозрительных сегментов по метрикам Whisper/MLX: **{len(low_conf)}**",
        "",
        "## Raw по персонажам",
    ]
    for k, v in speaker_counts(raw).items():
        md.append(f"- **{k}**: {v}")
    md += ["", "## Clean по персонажам"]
    for k, v in speaker_counts(clean).items():
        md.append(f"- **{k}**: {v}")
    md += ["", "## Примеры подозрительных сегментов"]
    for r in low_conf[:30]:
        md.append(f"- `{r.get('start_hms')}` **{r.get('character') or r.get('speaker')}**: {r.get('text')}  ")
        md.append(f"  avg_logprob={r.get('avg_logprob')} no_speech_prob={r.get('no_speech_prob')} compression_ratio={r.get('compression_ratio')}")
    out = paths.reports_dir / f"quality_report{suffix}.md"
    write_md(out, md)
    logger.info(f"Quality report: {out}")


# ──────────────────────────────────────────────────────────────
# Outputs
# ──────────────────────────────────────────────────────────────


def write_txt(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(f"[{fmt_hms(r['start'])}] {r.get('character') or r['speaker']}: {r['text']}\n")


def write_srt(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i, r in enumerate(rows, start=1):
            f.write(f"{i}\n")
            f.write(f"{fmt_srt(r['start'])} --> {fmt_srt(r['end'])}\n")
            f.write(f"{r.get('character') or r['speaker']}: {r['text']}\n\n")


def write_clean_outputs(rows: list[dict[str, Any]], cfg: dict[str, Any], paths: Paths) -> None:
    suffix = "_test" if (cfg.get("limit_minutes") or cfg.get("__limit_minutes")) else ""
    clean_jsonl = paths.clean_dir / f"{cfg['session_name']}{suffix}_clean.jsonl"
    clean_txt = paths.clean_dir / f"{cfg['session_name']}{suffix}_clean.txt"
    clean_srt = paths.clean_dir / f"{cfg['session_name']}{suffix}_clean.srt"
    write_jsonl(clean_jsonl, rows)
    write_txt(clean_txt, rows)
    write_srt(clean_srt, rows)
    logger.info(f"Clean JSONL: {clean_jsonl}")
    logger.info(f"Clean TXT:   {clean_txt}")
    logger.info(f"Clean SRT:   {clean_srt}")



def infer_scene_type(chunk_rows: list[dict[str, Any]]) -> str:
    """Rough scene classifier for downstream manual AI prompts."""
    text = " ".join(str(r.get("text") or "").lower() for r in chunk_rows)
    if re.search(r"\b(инициатив|атака|урон|спасброс|хитов|двадцат|натуралк|куб|брос|d20|д20)\b", text):
        return "combat_or_rolls"
    recap_hits = len(re.findall(r"прошл(ой|ая|ую)|рекап|в прошлой игре|что случилось", text))
    setup_hits = len(re.findall(r"микрофон|слышно|запис|наушник|погромче|потише|прибавь|включ", text))
    if recap_hits >= 2:
        return "recap"
    if setup_hits >= 4:
        return "setup"
    if re.search(r"договор|убежд|спрос|охранник|настоятельниц|колледж|таверн|библиотек", text):
        return "social_or_exploration"
    return "gameplay"

def make_chunks(rows: list[dict[str, Any]], cfg: dict[str, Any], paths: Paths) -> list[Path]:
    chunk_seconds = int(float(cfg.get("chunk_minutes", 10)) * 60)
    if not rows:
        return []

    prefix = "test_chunk" if (cfg.get("limit_minutes") or cfg.get("__limit_minutes")) else "chunk"
    for old in paths.chunks_dir.glob(f"{prefix}_*.json"):
        old.unlink()

    start0 = math.floor(rows[0]["start"] / chunk_seconds) * chunk_seconds
    chunks: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        idx = int((r["start"] - start0) // chunk_seconds)
        chunks.setdefault(idx, []).append(r)

    out_paths: list[Path] = []
    for idx in sorted(chunks):
        chunk_rows = chunks[idx]
        start = idx * chunk_seconds + start0
        end = start + chunk_seconds
        payload = {
            "session": cfg["session_name"],
            "chunk_index": idx,
            "scene_type": infer_scene_type(chunk_rows),
            "start": start,
            "end": end,
            "start_hms": fmt_hms_ms(start),
            "end_hms": fmt_hms_ms(end),
            "events": [
                {
                    "time": r["start_hms"],
                    "start": r["start"],
                    "end": r["end"],
                    "speaker": r["speaker"],
                    "character": r.get("character"),
                    "text": r["text"],
                }
                for r in chunk_rows
            ],
        }
        p = paths.chunks_dir / f"chunk_{idx:03d}.json"
        write_json(p, payload)
        out_paths.append(p)

    logger.info(f"Чанков создано: {len(out_paths)} → {paths.chunks_dir}")
    return out_paths


def prompt_for_chunk(chunk_json: dict[str, Any]) -> str:
    data = json.dumps(chunk_json, ensure_ascii=False, indent=2)
    return f"""# Ручной AI-анализ D&D-сессии

Ты анализируешь фрагмент D&D-сессии. Верни СТРОГО JSON без markdown-блока.

Задачи:
1. Вытащи действия персонажей.
2. Вытащи броски кубов: чистый бросок, модификатор, итог, тип броска, контекст.
3. Найди важные решения, помощь союзникам, урон, спасения, социальные победы, расследование, смешные/яркие моменты.
4. Дай кандидатов на MVP именно для этого чанка, с доказательствами по таймкодам.

Формат ответа:
{{
  "session": "{chunk_json.get('session')}",
  "chunk_index": {chunk_json.get('chunk_index')},
  "scene_type": "{chunk_json.get('scene_type', 'gameplay')}",
  "actions": [
    {{"time": "HH:MM:SS.mmm", "character": "...", "action": "...", "outcome": "...", "importance": "low|medium|high"}}
  ],
  "dice_rolls": [
    {{"time": "HH:MM:SS.mmm", "character": "...", "roll_type": "attack|damage|save|skill|initiative|other", "die": "d20|...", "natural": null, "modifier": null, "total": null, "context": "...", "confidence": "low|medium|high", "raw_text": "..."}}
  ],
  "mvp_signals": [
    {{"time": "HH:MM:SS.mmm", "character": "...", "category": "combat|social|support|idea|roleplay|fun|story", "reason": "...", "weight": 1}}
  ],
  "summary": "краткий пересказ чанка"
}}

Правила:
- Не выдумывай числа бросков. Если непонятно — null и confidence low.
- Не исправляй имена персонажей без необходимости.
- Если говорящий похож на ошибку разметки, всё равно укажи наиболее вероятного персонажа и confidence low.
- MVP оценивай по влиянию на сессию, не только по урону.
- Если scene_type = setup, техническую настройку микрофонов, громкости, наушников и флуд не считай важными действиями персонажей. Максимум importance=low, кроме явного запуска игры/рекапа.
- Если scene_type = recap, отличай факты прошлой сессии от действий текущей сессии. В actions пиши как recap/remembered action, а MVP-сигналы давай только за полезное структурирование или важные уточнения.
- Смешные шутки включай в MVP только если они реально яркие для чанка; вес 1, не больше.

Вот фрагмент:

{data}
"""


def make_prompts(chunk_paths: list[Path], paths: Paths) -> None:
    for old in paths.prompts_dir.glob("chunk_*_prompt.md"):
        old.unlink()

    for p in chunk_paths:
        chunk = load_json(p)
        prompt = prompt_for_chunk(chunk)
        out = paths.prompts_dir / f"{p.stem}_prompt.md"
        out.write_text(prompt, encoding="utf-8")
    logger.info(f"Промпты созданы: {paths.prompts_dir}")
    logger.info(f"AI-ответы вручную клади сюда: {paths.manual_ai_dir}")


# ──────────────────────────────────────────────────────────────
# Reports from manual AI results
# ──────────────────────────────────────────────────────────────


def read_manual_results(paths: Paths) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in sorted(paths.manual_ai_dir.glob("*.json")):
        try:
            raw = p.read_text(encoding="utf-8")
            rows.append(json.loads(normalize_json_text(raw)))
        except (OSError, ValueError) as e:
            logger.warning(f"Не смог прочитать {p}: {e}")
    return rows


# ──────────────────────────────────────────────────────────────
# AI analysis stage (API providers; manual mode stays the fallback)
# ──────────────────────────────────────────────────────────────


def ai_state_path(paths: Paths) -> Path:
    return paths.cache_dir / "ai_state.json"


def load_ai_state(paths: Paths) -> dict[str, Any]:
    p = ai_state_path(paths)
    if p.exists():
        state = load_json(p)
        state.setdefault("chunks", {})
        state.setdefault("pending_batch", None)
        return state
    return {"chunks": {}, "pending_batch": None}


def save_ai_state(paths: Paths, state: dict[str, Any]) -> None:
    write_json(ai_state_path(paths), state)


def build_ai_jobs(
    chunk_paths: list[Path], paths: Paths, state: dict[str, Any], force: bool
) -> tuple[list[Any], int]:
    from ai_providers import ChunkJob

    jobs: list[Any] = []
    skipped = 0
    for p in sorted(chunk_paths):
        name = p.stem
        chunk = load_json(p)
        h = stable_hash(chunk)
        out = paths.manual_ai_dir / f"{name}_events.json"
        if out.exists() and not force:
            entry = state.get("chunks", {}).get(name)
            # entry is None → файл положен руками, не трогаем.
            if entry is None or entry.get("chunk_hash") == h:
                skipped += 1
                continue
        jobs.append(ChunkJob(name=name, prompt=prompt_for_chunk(chunk), chunk_hash=h))
    return jobs, skipped


def make_ai_provider(ai: dict[str, Any], api_key: str | None) -> Any:
    if ai["provider"] == "anthropic":
        from ai_providers import AnthropicProvider

        return AnthropicProvider(
            model=ai["model"],
            api_key=api_key,
            mode=ai["mode"],
            max_output_tokens=ai["max_output_tokens"],
            concurrency=ai["concurrency"],
        )
    from ai_providers import OpenAICompatProvider

    return OpenAICompatProvider(
        model=ai["model"],
        base_url=ai["base_url"],
        api_key=api_key,
        max_output_tokens=ai["max_output_tokens"],
        concurrency=ai["concurrency"],
        normalize=normalize_json_text,
    )


def run_ai_analysis(
    chunk_paths: list[Path], cfg: dict[str, Any], paths: Paths, force: bool = False
) -> bool:
    """Run chunks through the configured LLM provider. Returns False → manual mode."""
    ai = resolve_ai_config(cfg)
    if not ai["enabled"]:
        logger.info("AI-этап выключен (ai.enabled=false) — ручной режим.")
        return False
    api_key = resolve_ai_api_key(ai)
    if ai["provider"] == "anthropic" and not api_key:
        env_name = ai.get("api_key_env") or "ANTHROPIC_API_KEY"
        logger.warning(f"Нет API-ключа: переменная {env_name} пуста. Падаю в ручной режим.")
        return False

    state = load_ai_state(paths)
    pending = state.get("pending_batch")
    resume_batch_id = None
    if pending and pending.get("provider") == ai["provider"] and pending.get("model") == ai["model"]:
        resume_batch_id = pending["batch_id"]
        logger.info(f"Продолжаю незавершённый batch: {resume_batch_id}")

    jobs, skipped = build_ai_jobs(chunk_paths, paths, state, force)
    if skipped:
        logger.info(f"AI: пропущено {skipped} чанков — результаты уже есть")
    if not jobs and not resume_batch_id:
        logger.info("AI: все чанки уже посчитаны.")
        return True

    hash_by_name = {j.name: j.chunk_hash for j in jobs}
    if resume_batch_id and pending:
        hash_by_name.update(pending.get("jobs", {}))

    if resume_batch_id and pending:
        total = len(pending.get("jobs", {}))
    else:
        total = len(jobs)
    progress = {"n": 0}

    def on_result(res: Any) -> None:
        progress["n"] += 1
        if res.data is not None:
            write_json(paths.manual_ai_dir / f"{res.name}_events.json", res.data)
            state["chunks"][res.name] = {
                "chunk_hash": hash_by_name.get(res.name, ""),
                "model": ai["model"],
                "provider": ai["provider"],
                "status": "done",
            }
            save_ai_state(paths, state)
            logger.info(f"AI [{progress['n']}/{total}]: {res.name} готов")
        else:
            logger.warning(f"AI [{progress['n']}/{total}]: {res.name} ошибка: {res.error}")

    provider = make_ai_provider(ai, api_key)
    logger.info(f"AI-анализ: provider={ai['provider']}, model={ai['model']}, чанков в работе: {total}")

    if ai["provider"] == "anthropic" and ai["mode"] == "batch":
        def on_batch_created(batch_id: str) -> None:
            state["pending_batch"] = {
                "batch_id": batch_id,
                "provider": ai["provider"],
                "model": ai["model"],
                "jobs": hash_by_name,
            }
            save_ai_state(paths, state)

        results = provider.analyze(
            jobs,
            on_result=on_result,
            resume_batch_id=resume_batch_id,
            on_batch_created=on_batch_created,
        )
        state["pending_batch"] = None
        save_ai_state(paths, state)
        if resume_batch_id and jobs:
            # Возобновлённый batch отдаёт результаты только своих чанков; из
            # свежесобранных jobs предупреждаем лишь о тех, кто остался ни с чем.
            done_names = {r.name for r in results if r.data is not None}
            leftover = [j.name for j in jobs if j.name not in done_names]
            if leftover:
                logger.warning(
                    f"AI: {len(leftover)} чанков не входили в возобновлённый batch и остались без результата: "
                    f"{', '.join(leftover)}. Запусти ai-analyze ещё раз."
                )
    else:
        results = provider.analyze(jobs, on_result=on_result)

    failed = [r for r in results if r.data is None]
    if failed:
        names = ", ".join(r.name for r in failed)
        logger.warning(
            f"AI: {len(failed)} чанков без результата: {names}. "
            f"Добить: python dnd_pipeline.py ai-analyze <session_dir>"
        )
    return True


def compute_report_data(results: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    """Общие расчёты для markdown-отчётов и PDF-хроники."""
    actions: list[dict[str, Any]] = []
    dice: list[dict[str, Any]] = []
    mvp_events: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for res in results:
        actions.extend(res.get("actions", []) or [])
        dice.extend(res.get("dice_rolls", []) or [])
        for item in res.get("mvp_signals", []) or []:
            item = dict(item)
            try:
                item["weight"] = int(item.get("weight", 1))
            except (ValueError, TypeError):
                item["weight"] = 1
            mvp_events.append(item)
        if res.get("summary"):
            summaries.append({"chunk_index": res.get("chunk_index"), "summary": res["summary"]})

    dice_stats: dict[str, dict[str, Any]] = {}
    for d in dice:
        natural = d.get("natural")
        if not isinstance(natural, int):
            continue
        char = d.get("character") or "?"
        st = dice_stats.setdefault(char, {"values": [], "nat20": 0, "nat1": 0})
        st["values"].append(natural)
        if natural == 20:
            st["nat20"] += 1
        if natural == 1:
            st["nat1"] += 1
    for char, st in dice_stats.items():
        values = st.pop("values")
        st["avg"] = sum(values) / len(values)
        st["count"] = len(values)

    mvp_scores: dict[str, int] = {}
    mvp_categories: dict[str, dict[str, int]] = {}
    for item in mvp_events:
        char = item.get("character") or "?"
        weight = item["weight"]
        mvp_scores[char] = mvp_scores.get(char, 0) + weight
        cat = item.get("category") or "?"
        mvp_categories.setdefault(char, {})
        mvp_categories[char][cat] = mvp_categories[char].get(cat, 0) + weight

    return {
        "actions": sorted(actions, key=lambda x: x.get("time", "")),
        "dice": dice,
        "dice_stats": dice_stats,
        "mvp_events": mvp_events,
        "mvp_scores": mvp_scores,
        "mvp_categories": mvp_categories,
        "summaries": summaries,
    }


def build_reports(paths: Paths, cfg: dict[str, Any]) -> None:
    results = read_manual_results(paths)
    paths.reports_dir.mkdir(parents=True, exist_ok=True)

    data = compute_report_data(results, cfg)

    # Actions timeline
    actions_md = [f"# Actions timeline — {cfg['session_name']}", ""]
    for a in data["actions"]:
        actions_md.append(f"- `{a.get('time', '')}` **{a.get('character', '?')}** — {a.get('action', '')} → {a.get('outcome', '')} _{a.get('importance', '')}_")
    write_md(paths.reports_dir / "actions_timeline.md", actions_md)

    # Dice stats
    dice_lines = [f"# Dice stats — {cfg['session_name']}", ""]
    for d in data["dice"]:
        char = d.get("character") or "?"
        dice_lines.append(f"- `{d.get('time', '')}` **{char}** {d.get('roll_type', '')}: natural={d.get('natural')} modifier={d.get('modifier')} total={d.get('total')} — {d.get('context', '')} _{d.get('confidence', '')}_")

    dice_lines.append("\n## Средние чистые d20, только где natural распознан")
    for char, st in sorted(data["dice_stats"].items()):
        dice_lines.append(f"- **{char}**: {st['avg']:.2f} по {st['count']} броскам")
    write_md(paths.reports_dir / "dice_stats.md", dice_lines)

    # MVP candidates
    mvp_lines = [f"# MVP candidates — {cfg['session_name']}", ""]
    for item in data["mvp_events"]:
        char = item.get("character") or "?"
        weight_int = item["weight"]
        mvp_lines.append(f"- `{item.get('time', '')}` **{char}** +{weight_int} [{item.get('category', '')}] — {item.get('reason', '')}")

    mvp_lines.append("\n## Итоговый счёт по MVP-сигналам")
    for char, value in sorted(data["mvp_scores"].items(), key=lambda x: x[1], reverse=True):
        mvp_lines.append(f"- **{char}**: {value}")
    write_md(paths.reports_dir / "mvp_candidates.md", mvp_lines)

    summaries_md = [f"- chunk {s['chunk_index']}: {s['summary']}" for s in data["summaries"]]
    session_report = [f"# Session report — {cfg['session_name']}", "", "## Summaries"] + summaries_md + ["", "## Files", "- actions_timeline.md", "- dice_stats.md", "- mvp_candidates.md"]
    write_md(paths.reports_dir / "session_report.md", session_report)

    logger.info(f"Отчёты собраны: {paths.reports_dir}")
    logger.info(f"Прочитано AI-ответов: {len(results)}")


# ──────────────────────────────────────────────────────────────
# Commands
# ──────────────────────────────────────────────────────────────


def load_cfg(args: argparse.Namespace) -> tuple[Path, dict[str, Any], Paths]:
    session_dir = Path(args.session_dir).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve() if args.config else session_dir / "config.json"
    cfg = load_json(config_path)
    if "session_name" not in cfg:
        raise ValueError("В config.json нужен session_name")

    # CLI overrides for fast experiments without editing config.json
    if getattr(args, "backend", None):
        cfg["transcription_backend"] = args.backend
    if getattr(args, "model", None):
        cfg["model_size"] = args.model
    if getattr(args, "limit_minutes", None):
        cfg["limit_minutes"] = args.limit_minutes
    if getattr(args, "quality_profile", None):
        cfg["quality_profile"] = args.quality_profile

    cfg = apply_quality_profile(cfg)

    paths = build_paths(session_dir, cfg["session_name"])
    ensure_dirs(paths)
    return session_dir, cfg, paths


def cmd_run(args: argparse.Namespace) -> None:
    session_dir, cfg, paths = load_cfg(args)
    raw = transcribe_all(session_dir, cfg, paths)
    filtered = filter_hallucinations(raw, cfg, paths)
    filtered = repair_segment_timings(filtered, cfg)
    clean = deduplicate(filtered, cfg)
    clean = repair_segment_timings(clean, cfg)
    clean = merge_adjacent_segments(clean, cfg)
    clean = repair_segment_timings(clean, cfg)
    write_clean_outputs(clean, cfg, paths)
    write_quality_report(raw, clean, cfg, paths)
    chunk_paths = make_chunks(clean, cfg, paths)
    make_prompts(chunk_paths, paths)
    if run_ai_analysis(chunk_paths, cfg, paths):
        build_reports(paths, cfg)
        logger.info("\nГотово: AI-анализ прошёл, отчёты в reports/.")
    else:
        logger.info("\nГотово. Дальше: открывай prompts/*.md, вручную прогоняй через AI и складывай JSON-ответы в manual_ai_results/.")



def raw_path_for_cfg(cfg: dict[str, Any], paths: Paths) -> Path:
    suffix = "_test" if (cfg.get("limit_minutes") or cfg.get("__limit_minutes")) else ""
    return paths.raw_dir / f"{cfg['session_name']}{suffix}_raw.jsonl"


def cmd_rebuild(args: argparse.Namespace) -> None:
    """Rebuild clean/chunks/prompts from existing raw JSONL without transcription."""
    session_dir, cfg, paths = load_cfg(args)
    raw_path = raw_path_for_cfg(cfg, paths)
    if not raw_path.exists():
        # Helpful fallback for people using --limit-minutes only during the old run.
        fallback = paths.raw_dir / f"{cfg['session_name']}_raw.jsonl"
        test_fallback = paths.raw_dir / f"{cfg['session_name']}_test_raw.jsonl"
        if test_fallback.exists() and (cfg.get("limit_minutes") or cfg.get("__limit_minutes")):
            raw_path = test_fallback
        elif fallback.exists():
            raw_path = fallback
        else:
            raise RuntimeError(f"Не найден raw JSONL для rebuild: {raw_path}")

    raw = read_jsonl(raw_path)
    logger.info(f"Rebuild из raw: {raw_path} ({len(raw)} сегментов)")
    filtered = filter_hallucinations(raw, cfg, paths)
    filtered = repair_segment_timings(filtered, cfg)
    clean = deduplicate(filtered, cfg)
    clean = repair_segment_timings(clean, cfg)
    clean = merge_adjacent_segments(clean, cfg)
    clean = repair_segment_timings(clean, cfg)
    write_clean_outputs(clean, cfg, paths)
    write_quality_report(raw, clean, cfg, paths)
    chunk_paths = make_chunks(clean, cfg, paths)
    make_prompts(chunk_paths, paths)
    logger.info("\nRebuild готов: транскрибация не запускалась.")

def cmd_prepare_ai(args: argparse.Namespace) -> None:
    session_dir, cfg, paths = load_cfg(args)
    clean_path = paths.clean_dir / f"{cfg['session_name']}_clean.jsonl"
    rows = read_jsonl(clean_path)
    if not rows:
        raise RuntimeError(f"Не найден clean JSONL: {clean_path}")
    chunk_paths = make_chunks(rows, cfg, paths)
    make_prompts(chunk_paths, paths)


def cmd_ai_analyze(args: argparse.Namespace) -> None:
    session_dir, cfg, paths = load_cfg(args)
    chunk_paths = sorted(paths.chunks_dir.glob("chunk_*.json"))
    if not chunk_paths:
        raise RuntimeError(
            f"Нет чанков в {paths.chunks_dir}. Сначала: python dnd_pipeline.py prepare-ai {session_dir}"
        )
    if run_ai_analysis(chunk_paths, cfg, paths, force=getattr(args, "force", False)):
        build_reports(paths, cfg)


def cmd_build_report(args: argparse.Namespace) -> None:
    session_dir, cfg, paths = load_cfg(args)
    build_reports(paths, cfg)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="D&D PodTrak local pipeline")

    verbosity = argparse.ArgumentParser(add_help=False)
    vgroup = verbosity.add_mutually_exclusive_group()
    vgroup.add_argument("--verbose", "-v", action="store_true", help="show per-line transcript and debug detail")
    vgroup.add_argument("--quiet", "-q", action="store_true", help="only warnings and errors")

    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", parents=[verbosity], help="full pipeline: transcribe → clean → chunks → prompts")
    p_run.add_argument("session_dir", help="folder with PodTrak wav files")
    p_run.add_argument("--config", help="path to config.json")
    p_run.add_argument("--backend", choices=["faster_whisper", "mlx"], help="override transcription_backend")
    p_run.add_argument("--model", help="override model_size, e.g. mlx-community/whisper-large-v3-turbo")
    p_run.add_argument("--limit-minutes", type=float, help="transcribe only first N minutes of every track for quick tests")
    p_run.add_argument("--quality-profile", choices=["gentle", "balanced", "aggressive"], help="override quality_profile")
    p_run.set_defaults(func=cmd_run)

    p_rebuild = sub.add_parser("rebuild", parents=[verbosity], help="rebuild clean/chunks/prompts from existing raw JSONL; no transcription")
    p_rebuild.add_argument("session_dir", help="folder with session files")
    p_rebuild.add_argument("--config", help="path to config.json")
    p_rebuild.add_argument("--limit-minutes", type=float, help="use *_test_raw.jsonl if rebuilding a test run")
    p_rebuild.add_argument("--quality-profile", choices=["gentle", "balanced", "aggressive"], help="override quality_profile")
    p_rebuild.set_defaults(func=cmd_rebuild)

    p_ai = sub.add_parser("prepare-ai", parents=[verbosity], help="create chunks and manual AI prompts from clean.jsonl")
    p_ai.add_argument("session_dir", help="folder with session files")
    p_ai.add_argument("--config", help="path to config.json")
    p_ai.add_argument("--quality-profile", choices=["gentle", "balanced", "aggressive"], help="override quality_profile")
    p_ai.set_defaults(func=cmd_prepare_ai)

    p_aa = sub.add_parser("ai-analyze", parents=[verbosity], help="run AI analysis over chunks via API, then build reports")
    p_aa.add_argument("session_dir", help="folder with session files")
    p_aa.add_argument("--config", help="path to config.json")
    p_aa.add_argument("--force", action="store_true", help="recompute all chunks even if results exist")
    p_aa.add_argument("--quality-profile", choices=["gentle", "balanced", "aggressive"], help="override quality_profile")
    p_aa.set_defaults(func=cmd_ai_analyze)

    p_report = sub.add_parser("build-report", parents=[verbosity], help="build reports from manual_ai_results/*.json")
    p_report.add_argument("session_dir", help="folder with session files")
    p_report.add_argument("--config", help="path to config.json")
    p_report.add_argument("--quality-profile", choices=["gentle", "balanced", "aggressive"], help="override quality_profile")
    p_report.set_defaults(func=cmd_build_report)

    args = parser.parse_args(argv)
    configure_logging(getattr(args, "verbose", False), getattr(args, "quiet", False))
    try:
        args.func(args)
        return 0
    except KeyboardInterrupt:
        logger.info("\nОстановлено пользователем.")
        return 130
    except Exception as e:  # noqa: BLE001 — top-level CLI handler: catch all to exit cleanly
        logger.error(f"Ошибка: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
