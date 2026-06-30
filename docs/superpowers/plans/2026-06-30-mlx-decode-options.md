# MLX Decode Options Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `config.json` decode settings actually reach both ASR backends, add per-profile anti-hallucination defaults, and support a user-authored `initial_prompt`.

**Architecture:** A pure resolver function maps config `decode` settings to backend-correct kwargs. Quality profiles inject sensible decode defaults via the existing `deep_merge`. Both transcribe functions splat the resolved kwargs. The per-track cache signature includes the resolved options so tuning invalidates stale caches.

**Tech Stack:** Python 3.12, pytest, faster-whisper, mlx-whisper, numpy.

## Global Constraints

- No new runtime dependencies beyond those already in `requirements.txt`.
- All transcription decode logic must be unit-testable without a model or audio (the resolver and cache-signature are pure functions).
- Drop only `None`-valued options (so the library default applies); never drop `False`/`0`.
- faster-whisper parameter name is `log_prob_threshold`; mlx-whisper is `logprob_threshold`.
- mlx backend aliases: `mlx`, `mlx_whisper`, `mlx-whisper`. Anything else uses faster-whisper naming.
- Legacy top-level `condition_on_previous_text` must keep working when no `decode` block sets it.
- `temperature` is intentionally left unset (library fallback schedule is wanted).

---

### Task 1: `resolve_decode_options` resolver

**Files:**
- Modify: `dnd_pipeline.py` (add function just after `get_backend`, ~line 337)
- Test: `tests/test_decode.py` (create)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `resolve_decode_options(cfg: dict[str, Any], backend: str) -> dict[str, Any]` — returns a kwargs dict ready to splat into a `transcribe(...)` call. Keys (when not None): `condition_on_previous_text`, `initial_prompt`, `compression_ratio_threshold`, `no_speech_threshold`, `hallucination_silence_threshold`, and `logprob_threshold` (mlx) or `log_prob_threshold` (faster-whisper).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_decode.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_decode.py -q`
Expected: FAIL with `AttributeError: module 'dnd_pipeline' has no attribute 'resolve_decode_options'`

- [ ] **Step 3: Write minimal implementation**

Insert after `get_backend` (after line 338 in `dnd_pipeline.py`):

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_decode.py -q`
Expected: PASS (9 passed)

- [ ] **Step 5: Commit**

```bash
git add dnd_pipeline.py tests/test_decode.py
git commit -m "feat: add resolve_decode_options resolver"
```

---

### Task 2: Per-profile decode defaults

**Files:**
- Modify: `dnd_pipeline.py` — `apply_quality_profile`, the `profiles` dict (lines 114-157)
- Test: `tests/test_decode.py` (append)

**Interfaces:**
- Consumes: `apply_quality_profile(cfg)` existing behavior, `deep_merge`.
- Produces: each profile in `apply_quality_profile` now contains a `"decode"` dict with keys `condition_on_previous_text`, `compression_ratio_threshold`, `logprob_threshold`, `no_speech_threshold`, `hallucination_silence_threshold`, `initial_prompt`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_decode.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_decode.py -q`
Expected: FAIL — `KeyError: 'decode'`

- [ ] **Step 3: Add the decode block to each profile**

In `dnd_pipeline.py`, inside `apply_quality_profile`'s `profiles` dict, add a `"decode"` key to each profile. For `gentle` (after its `postprocess` entry):

```python
            "decode": {
                "condition_on_previous_text": False,
                "compression_ratio_threshold": 2.6,
                "logprob_threshold": -1.2,
                "no_speech_threshold": 0.7,
                "hallucination_silence_threshold": 5.0,
                "initial_prompt": None,
            },
```

For `balanced`:

```python
            "decode": {
                "condition_on_previous_text": False,
                "compression_ratio_threshold": 2.4,
                "logprob_threshold": -1.0,
                "no_speech_threshold": 0.6,
                "hallucination_silence_threshold": 2.0,
                "initial_prompt": None,
            },
```

For `aggressive`:

```python
            "decode": {
                "condition_on_previous_text": False,
                "compression_ratio_threshold": 2.2,
                "logprob_threshold": -0.8,
                "no_speech_threshold": 0.5,
                "hallucination_silence_threshold": 1.0,
                "initial_prompt": None,
            },
```

(Place each `"decode"` entry as a sibling of `"preprocess"`/`"dedupe"`/`"postprocess"` inside its profile.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_decode.py -q`
Expected: PASS (14 passed)

- [ ] **Step 5: Commit**

```bash
git add dnd_pipeline.py tests/test_decode.py
git commit -m "feat: per-profile decode anti-hallucination defaults"
```

---

### Task 3: Wire resolver into both transcribe functions

**Files:**
- Modify: `dnd_pipeline.py` — `run_mlx_transcribe` (lines 361-366), `run_faster_whisper_transcribe` (lines 372-380)
- Test: `tests/test_decode.py` (append)

**Interfaces:**
- Consumes: `resolve_decode_options(cfg, backend)` from Task 1.
- Produces: both backends call `transcribe(..., **resolve_decode_options(config, <backend>))`. The hardcoded `condition_on_previous_text` line in `run_faster_whisper_transcribe` is removed (now supplied by the resolver).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_decode.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_decode.py -k forwards -q`
Expected: FAIL — captured kwargs lack `logprob_threshold` / `log_prob_threshold`.

- [ ] **Step 3: Wire the resolver into both functions**

In `run_mlx_transcribe`, replace the `mlx_whisper.transcribe(...)` call (lines 361-366) with:

```python
    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=model,
        language=language,
        verbose=False,
        **resolve_decode_options(config, "mlx"),
    )
```

In `run_faster_whisper_transcribe`, change the `kwargs` dict (lines 372-379) to remove the hardcoded `condition_on_previous_text` line and merge the resolver:

```python
    kwargs = dict(
        language=config.get("language", "ru"),
        vad_filter=bool(config.get("use_vad", True)),
        vad_parameters={"min_silence_duration_ms": int(config.get("vad_min_silence_ms", 800))},
        beam_size=int(config.get("beam_size", 5)),
        word_timestamps=False,
    )
    kwargs.update(resolve_decode_options(config, "faster_whisper"))
    segments, info = model.transcribe(str(audio_path), **kwargs)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_decode.py -q`
Expected: PASS (16 passed)

- [ ] **Step 5: Commit**

```bash
git add dnd_pipeline.py tests/test_decode.py
git commit -m "feat: forward decode options to mlx and faster-whisper"
```

---

### Task 4: Include decode options in the per-track cache signature

**Files:**
- Modify: `dnd_pipeline.py` — extract the inline cache-signature dict in `transcribe_track` (lines 547-560) into a helper, add decode options
- Test: `tests/test_decode.py` (append)

**Interfaces:**
- Consumes: `resolve_decode_options(cfg, backend)` from Task 1.
- Produces: `transcription_cache_signature(config, backend, transcription_path_name, transcription_fingerprint) -> dict[str, Any]` — the dict that gets `stable_hash`ed for the per-track cache key. Includes a `"decode_options"` entry.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_decode.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_decode.py -k cache -q`
Expected: FAIL — `module 'dnd_pipeline' has no attribute 'transcription_cache_signature'`

- [ ] **Step 3: Extract the helper and add decode options**

Add this function immediately above `transcribe_track` (before line 530):

```python
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
```

Then in `transcribe_track`, replace the inline `transcription_cfg_hash = stable_hash({ ... })` block (lines 547-560) with:

```python
    transcription_cfg_hash = stable_hash(transcription_cache_signature(
        config,
        backend,
        transcription_path.name,
        transcription_fingerprint,
        track.get("preprocess", {}) or {},
    ))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_decode.py -q`
Expected: PASS (18 passed)

- [ ] **Step 5: Commit**

```bash
git add dnd_pipeline.py tests/test_decode.py
git commit -m "feat: invalidate track cache when decode options change"
```

---

### Task 5: Update config.example.json and docs

**Files:**
- Modify: `config.example.json` (add `decode` block + `initial_prompt`)
- Modify: `README.md` (note device/compute_type are faster-whisper-only)
- Test: `tests/test_decode.py` (append)

**Interfaces:**
- Consumes: `apply_quality_profile`, `resolve_decode_options`.
- Produces: a `config.example.json` that documents the new `decode` block.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_decode.py`:

```python
import json
from pathlib import Path


def test_config_example_has_valid_decode_block():
    cfg = json.loads(Path("config.example.json").read_text(encoding="utf-8"))
    assert "decode" in cfg
    assert "initial_prompt" in cfg["decode"]
    merged = dp.apply_quality_profile(cfg)
    opts = dp.resolve_decode_options(merged, "mlx")
    assert "condition_on_previous_text" in opts
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_decode.py -k config_example -q`
Expected: FAIL — `assert 'decode' in cfg`

- [ ] **Step 3: Add the decode block to config.example.json**

In `config.example.json`, add this block after the `"condition_on_previous_text": false,` line (keep that line; the decode block is the new source of truth and the legacy line stays for backward-compat illustration):

```json
  "decode": {
    "initial_prompt": "Запись настольной игры Dungeons & Dragons на русском. Партия: Ангрон, Антернер, Шиян, Гай Гексан, Алкюр Септик. Ведёт Данжен Мастер. Термины: инициатива, спасбросок, проверка, крит, преимущество, помеха, заклинание. Реплики с правильной пунктуацией: «Я атакую гоблина. Бросаю d20 — выпало 17.»",
    "condition_on_previous_text": false,
    "compression_ratio_threshold": 2.4,
    "logprob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    "hallucination_silence_threshold": 2.0
  },
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_decode.py -k config_example -q`
Expected: PASS

- [ ] **Step 5: Add the README note**

In `README.md`, find the section describing config fields and add (under or near the backend/model description):

```markdown
> **MLX (Apple Silicon):** `device` and `compute_type` apply to the
> faster-whisper backend only. MLX ignores them and runs fp16 on the
> Metal GPU / Neural Engine. Decode quality on MLX is controlled by the
> `decode` block (`initial_prompt`, `condition_on_previous_text`,
> `compression_ratio_threshold`, `logprob_threshold`, `no_speech_threshold`,
> `hallucination_silence_threshold`), which the quality profiles set by default.
```

(If no matching config section exists, add it under the existing backend/quality-profile heading.)

- [ ] **Step 6: Run the full suite**

Run: `python3 -m pytest -q`
Expected: PASS (all tests; 19 added in test_decode.py plus the prior 47)

- [ ] **Step 7: Commit**

```bash
git add config.example.json README.md tests/test_decode.py
git commit -m "docs: document decode block and MLX device/compute_type caveat"
```

---

## Notes for the implementer

- Run the whole suite (`python3 -m pytest -q`) after Task 5; nothing in the prior 47 tests should regress.
- Do not change the post-hoc `hallucination_filter` — it is complementary and out of scope.
- Do not implement layered config here — it is a deferred follow-up (see the spec's non-goals).
