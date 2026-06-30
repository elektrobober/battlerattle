# Track Auto-Discovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Auto-discover audio tracks from the session directory (any file count) when `config["tracks"]` is absent, deriving speaker/character from the filename.

**Architecture:** A pure `discover_tracks(session_dir, config)` function returns the same track-dict shape the explicit `tracks` array produces. `transcribe_all` calls it instead of reading `config["tracks"]` directly. Explicit `tracks` still wins for backward compatibility.

**Tech Stack:** Python 3.12 stdlib (`pathlib`), pytest.

## Global Constraints

- No new runtime dependencies.
- Explicit non-empty `config["tracks"]` must be returned unchanged (backward compatibility).
- Discovery is filename-only — no audio decoding, non-recursive glob of `session_dir`.
- Returned track dicts have keys `file`, `speaker`, `character`, `priority` (same shape the rest of the pipeline reads).
- Filename parsing: strip a leading `f"{session_name}-"` prefix, then split the label on the first `". "` (speaker, character); a single-part label sets speaker == character.
- Priority is `100` when `speaker == config.get("dm_speaker")`, else `50`.
- Default audio extensions `[".wav"]`; extension match is case-insensitive.
- Exclude `config.get("master_mix")` and any name in `config.get("exclude", [])`.
- Results sorted by filename.

---

### Task 1: `discover_tracks` function

**Files:**
- Modify: `dnd_pipeline.py` — add `discover_tracks` (place it just before `transcribe_all`, ~line 721)
- Test: `tests/test_discovery.py` (create)

**Interfaces:**
- Produces: `discover_tracks(session_dir: Path, config: dict[str, Any]) -> list[dict[str, Any]]` — returns a list of `{"file": str, "speaker": str, "character": str, "priority": int}` dicts.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_discovery.py`:

```python
"""Tests for filename-based track auto-discovery."""

import dnd_pipeline as dp


def touch(d, name):
    (d / name).write_bytes(b"")


def test_explicit_tracks_returned_unchanged():
    explicit = [{"file": "a.wav", "speaker": "S", "character": "C", "priority": 50}]
    out = dp.discover_tracks(None, {"session_name": "s", "tracks": explicit})
    assert out is explicit


def test_discovers_all_files_regardless_of_count(tmp_path):
    for n in ["dnd_2-Антернер.wav", "dnd_2-Шиян.wav", "dnd_2-Гай Гексан.wav"]:
        touch(tmp_path, n)
    out = dp.discover_tracks(tmp_path, {"session_name": "dnd_2"})
    assert len(out) == 3


def test_five_vs_six_files(tmp_path):
    for n in ["dnd_2-A.wav", "dnd_2-B.wav", "dnd_2-C.wav", "dnd_2-D.wav", "dnd_2-E.wav"]:
        touch(tmp_path, n)
    out = dp.discover_tracks(tmp_path, {"session_name": "dnd_2"})
    assert len(out) == 5


def test_speaker_character_parsed_from_dotted_name(tmp_path):
    touch(tmp_path, "dnd_2-Дима. Ангрон.wav")
    out = dp.discover_tracks(tmp_path, {"session_name": "dnd_2"})
    assert out[0]["speaker"] == "Дима"
    assert out[0]["character"] == "Ангрон"


def test_single_name_sets_speaker_equals_character(tmp_path):
    touch(tmp_path, "dnd_2-Антернер.wav")
    out = dp.discover_tracks(tmp_path, {"session_name": "dnd_2"})
    assert out[0]["speaker"] == "Антернер"
    assert out[0]["character"] == "Антернер"


def test_prefix_stripped(tmp_path):
    touch(tmp_path, "dnd_2-Шиян.wav")
    out = dp.discover_tracks(tmp_path, {"session_name": "dnd_2"})
    assert out[0]["speaker"] == "Шиян"  # not "dnd_2-Шиян"


def test_master_mix_excluded(tmp_path):
    touch(tmp_path, "dnd_2-001.wav")
    touch(tmp_path, "dnd_2-Шиян.wav")
    out = dp.discover_tracks(tmp_path, {"session_name": "dnd_2", "master_mix": "dnd_2-001.wav"})
    files = [t["file"] for t in out]
    assert "dnd_2-001.wav" not in files
    assert "dnd_2-Шиян.wav" in files


def test_exclude_list_honored(tmp_path):
    touch(tmp_path, "dnd_2-skip.wav")
    touch(tmp_path, "dnd_2-keep.wav")
    out = dp.discover_tracks(tmp_path, {"session_name": "dnd_2", "exclude": ["dnd_2-skip.wav"]})
    files = [t["file"] for t in out]
    assert files == ["dnd_2-keep.wav"]


def test_dm_speaker_gets_priority_100(tmp_path):
    touch(tmp_path, "dnd_2-Данжен Мастер.wav")
    touch(tmp_path, "dnd_2-Дима. Ангрон.wav")
    out = dp.discover_tracks(tmp_path, {"session_name": "dnd_2", "dm_speaker": "Данжен Мастер"})
    by_speaker = {t["speaker"]: t["priority"] for t in out}
    assert by_speaker["Данжен Мастер"] == 100
    assert by_speaker["Дима"] == 50


def test_non_audio_files_ignored(tmp_path):
    touch(tmp_path, "dnd_2-Шиян.wav")
    touch(tmp_path, "notes.txt")
    touch(tmp_path, "config.json")
    out = dp.discover_tracks(tmp_path, {"session_name": "dnd_2"})
    assert [t["file"] for t in out] == ["dnd_2-Шиян.wav"]


def test_extension_match_is_case_insensitive(tmp_path):
    touch(tmp_path, "dnd_2-Шиян.WAV")
    out = dp.discover_tracks(tmp_path, {"session_name": "dnd_2"})
    assert len(out) == 1


def test_results_sorted_by_filename(tmp_path):
    for n in ["dnd_2-C.wav", "dnd_2-A.wav", "dnd_2-B.wav"]:
        touch(tmp_path, n)
    out = dp.discover_tracks(tmp_path, {"session_name": "dnd_2"})
    assert [t["file"] for t in out] == ["dnd_2-A.wav", "dnd_2-B.wav", "dnd_2-C.wav"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_discovery.py -q`
Expected: FAIL — `module 'dnd_pipeline' has no attribute 'discover_tracks'`

- [ ] **Step 3: Write minimal implementation**

Add this function in `dnd_pipeline.py` immediately before `def transcribe_all(`:

```python
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
    return tracks
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_discovery.py -q`
Expected: PASS (12 passed)

- [ ] **Step 5: Commit**

```bash
git add dnd_pipeline.py tests/test_discovery.py
git commit -m "feat: add discover_tracks filename-based auto-discovery"
```

---

### Task 2: Wire into transcribe_all + document config

**Files:**
- Modify: `dnd_pipeline.py:744` — `transcribe_all` track source
- Modify: `config.example.json` — make `tracks` optional, add `dm_speaker`/`audio_extensions`/`exclude` examples
- Modify: `README.md` — note auto-discovery
- Test: `tests/test_discovery.py` (append)

**Interfaces:**
- Consumes: `discover_tracks(session_dir, config)` from Task 1.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_discovery.py`:

```python
def test_transcribe_all_uses_discovery_when_no_tracks(tmp_path, monkeypatch):
    # No `tracks` in config → transcribe_all should discover the two files and
    # call transcribe_track once per discovered track.
    (tmp_path / "dnd_2-Шиян.wav").write_bytes(b"")
    (tmp_path / "dnd_2-Дима. Ангрон.wav").write_bytes(b"")

    seen = []

    def fake_transcribe_track(model, session_dir, track, config, paths):
        seen.append(track["speaker"])
        return []

    monkeypatch.setattr(dp, "transcribe_track", fake_transcribe_track)
    cfg = {"session_name": "dnd_2", "transcription_backend": "mlx"}
    # Avoid loading a real MLX model: stub the model-load path.
    monkeypatch.setattr(dp, "mlx_whisper", object())
    paths = dp.build_paths(tmp_path, "dnd_2")
    dp.ensure_dirs(paths)

    dp.transcribe_all(tmp_path, cfg, paths)
    assert sorted(seen) == ["Дима", "Шиян"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_discovery.py -k transcribe_all_uses_discovery -q`
Expected: FAIL — `transcribe_all` reads `config.get("tracks", [])` (empty) so `transcribe_track` is never called; `seen` stays empty.

- [ ] **Step 3: Change the track source in transcribe_all**

In `dnd_pipeline.py`, line 744, replace:

```python
    tracks = config.get("tracks", [])
```

with:

```python
    tracks = discover_tracks(session_dir, config)
```

- [ ] **Step 4: Run test to verify it passes + full suite**

Run: `python3 -m pytest -q`
Expected: PASS (all prior tests plus the new one; nothing regresses).

- [ ] **Step 5: Update config.example.json**

In `config.example.json`, add the new optional keys near the top (after `"master_mix"`), and add a comment-style note. Since JSON has no comments, add the keys and rely on the README for explanation. Insert after the `"master_mix": "dnd_2-001.wav",` line:

```json
  "dm_speaker": "Данжен Мастер",
  "audio_extensions": [".wav"],
  "exclude": [],
```

Leave the existing `"tracks": [...]` array in place as the explicit-override example (it still works and demonstrates the override).

- [ ] **Step 6: Add the README note**

In `README.md`, near the `tracks`/config documentation, add:

```markdown
> **Авто-обнаружение треков:** если `tracks` в конфиге не задан, пайплайн сам
> находит все аудиофайлы в папке сессии (расширения из `audio_extensions`,
> по умолчанию `.wav`) — сколько бы их ни было. Имя файла парсится как
> `Спикер. Персонаж` (например `dnd_2-Дима. Ангрон.wav` → спикер «Дима»,
> персонаж «Ангрон»); имя без `. ` даёт `спикер == персонаж`. Трек, чей спикер
> равен `dm_speaker`, получает приоритет 100. Файл `master_mix` и записи из
> `exclude` пропускаются. Явный `tracks` (если задан) всегда побеждает.
```

- [ ] **Step 7: Run the full suite + commit**

Run: `python3 -m pytest -q`
Expected: PASS (all tests).

```bash
git add dnd_pipeline.py config.example.json README.md tests/test_discovery.py
git commit -m "feat: wire track auto-discovery into transcribe_all; document config"
```

---

## Notes for the implementer

- Run the whole suite (`python3 -m pytest -q`) after Task 2; the prior 80 tests must not regress.
- `config.example.json` must remain valid JSON after the edit (watch trailing commas).
- Do not implement layered config — out of scope (separate follow-up).
