# Design: Track auto-discovery from session files

Date: 2026-06-30
Status: Approved

## Problem

`config.json` requires an explicit `tracks` array — one object per audio file
(`file`, `speaker`, `character`, `priority`). The pipeline is run after every
session, and the number of recorded channels varies (6 one week, 5 the next).
Today that means hand-editing the `tracks` array every session: deleting/adding
objects, fixing filenames, risking typos. The user wants to "just drop the files
in, however many there are" and have the pipeline pick them all up.

## Goals

1. Auto-discover audio tracks from the session directory when `tracks` is not
   given, regardless of how many files there are.
2. Derive `speaker`/`character` from the filename by convention (PodTrak already
   names files like `dnd_2-Дима. Ангрон.wav`).
3. Mark the DM track with priority 100 (needed by dedupe's `choose_better`)
   without re-listing every file.

## Non-goals

- Replacing the explicit `tracks` array — it stays as a supported override.
- Per-character priorities beyond the single DM marker (YAGNI; all players are 50).
- Recursive directory scanning or audio-content inspection (filename-only).
- Layered config (campaign base + per-session) — a separate deferred follow-up;
  this feature already removes the main per-session editing pain.

## Design

### Unit — `discover_tracks(session_dir: Path, config: dict) -> list[dict]`

A pure function (filename-only; no audio decoding) returning the list of track
dicts that `transcribe_all` iterates. Each dict has `file`, `speaker`,
`character`, `priority` — the same shape the explicit `tracks` array produces.

Algorithm:
1. **Explicit wins.** If `config.get("tracks")` is a non-empty list, return it
   unchanged (backward compatibility — existing configs like `may/` keep working).
2. **Glob.** List `session_dir` non-recursively for files whose suffix is in
   `config.get("audio_extensions", [".wav"])` (case-insensitive). Non-recursive so
   the `_dnd_pipeline_out/` output tree is never scanned.
3. **Exclude.** Drop any filename equal to `config.get("master_mix")` (the
   combined mix, not a per-speaker track) or listed in `config.get("exclude", [])`.
4. **Parse each filename** (stem = name without suffix):
   - Strip a leading `f"{session_name}-"` prefix if present → `label`.
   - Split `label` on the first `". "`: two parts → `speaker`, `character`;
     otherwise `speaker = character = label`.
   - Examples: `dnd_2-Дима. Ангрон.wav` → speaker `Дима`, character `Ангрон`;
     `dnd_2-Данжен Мастер.wav` → both `Данжен Мастер`; `dnd_2-Антернер.wav` →
     both `Антернер`.
5. **Priority.** `100` if `speaker == config.get("dm_speaker")`, else `50`.
6. **Sort** the results by filename for a stable track order, then return.

### Consumption change

`transcribe_all` currently does `tracks = config.get("tracks", [])`. It becomes
`tracks = discover_tracks(session_dir, config)`. No other call site changes; the
returned dicts have the same keys the rest of the pipeline already reads
(`preprocess_track_audio` reads `track["file"]`; `transcribe_track` reads
`speaker`/`character`/`priority` with the existing defaults).

### Config additions (all optional)

```jsonc
"dm_speaker": "Данжен Мастер",   // that speaker's track gets priority 100
"audio_extensions": [".wav"],     // default if omitted
"exclude": [],                    // extra filenames to skip; master_mix auto-excluded
// "tracks": [...]                // now OPTIONAL: present → used; absent/empty → auto-discover
```

## Testing (TDD)

Filename-only, so tests use `tmp_path` with empty touched files — no audio needed:

- Explicit non-empty `tracks` is returned unchanged (backward compat).
- A directory with N audio files yields N tracks (and 5 files → 5, 6 → 6 — the
  core requirement).
- `"Speaker. Character"` parses into distinct speaker/character; a single-name
  file sets speaker == character.
- The `master_mix` filename is excluded.
- `dm_speaker` match → priority 100; everyone else → 50.
- The `f"{session_name}-"` prefix is stripped from the label.
- Non-audio files (e.g. `.txt`, `.json`) are ignored; extension match is
  case-insensitive.

## Risks

- The `". "` parsing convention is fragile to off-pattern names (a filename with
  an unrelated `". "` would split oddly). Mitigation: the explicit `tracks` array
  remains a full override for edge cases, documented in `config.example.json`.
- If `session_dir` contains stray audio files (old takes), they would be picked
  up. Mitigation: `exclude` list, and the explicit-tracks override.
