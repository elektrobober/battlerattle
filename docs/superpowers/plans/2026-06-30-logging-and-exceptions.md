# Logging + Narrowed Exceptions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace all 41 `print()` calls with a level-aware module logger, add `--verbose`/`--quiet` flags, and narrow two over-broad `except` blocks.

**Architecture:** A module logger `logging.getLogger("dnd_pipeline")` plus a `configure_logging(verbose, quiet)` helper that installs a single stdout handler at the chosen level. Each print maps to DEBUG/INFO/WARNING/ERROR. Two over-broad excepts are narrowed; all three excepts route through the logger.

**Tech Stack:** Python 3.12 stdlib `logging` + `argparse`, pytest (`caplog`).

## Global Constraints

- No new runtime dependencies (stdlib `logging` only).
- Console output format is `%(message)s` — no `INFO:`/`WARNING:` prefixes (the existing Russian text conveys severity).
- The per-line live transcript (`dnd_pipeline.py:686`) goes to DEBUG (hidden unless `--verbose`).
- Do NOT touch the import-guard `except Exception` blocks at lines 37/42.
- `main`'s top-level `except Exception` stays broad (legit CLI handler) but logs via the logger; it still returns 1. `KeyboardInterrupt` still returns 130.
- `configure_logging` must be idempotent (clear prior handlers on the `dnd_pipeline` logger) and must NOT disable propagation (pytest `caplog` relies on it).

---

### Task 1: Module logger + `configure_logging`

**Files:**
- Modify: `dnd_pipeline.py` — add `import logging`, module logger, and `configure_logging` near the top (after the existing imports / before `# Utils`)
- Test: `tests/test_logging.py` (create)

**Interfaces:**
- Produces: module-level `logger = logging.getLogger("dnd_pipeline")` and `configure_logging(verbose: bool, quiet: bool) -> int` (returns the numeric level it applied; also sets `logger.level` and installs one stdout `StreamHandler` with format `%(message)s`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_logging.py`:

```python
"""Tests for logging configuration and level mapping."""

import logging

import dnd_pipeline as dp


def test_default_level_is_info():
    assert dp.configure_logging(verbose=False, quiet=False) == logging.INFO
    assert dp.logger.level == logging.INFO


def test_verbose_sets_debug():
    assert dp.configure_logging(verbose=True, quiet=False) == logging.DEBUG
    assert dp.logger.level == logging.DEBUG


def test_quiet_sets_warning():
    assert dp.configure_logging(verbose=False, quiet=True) == logging.WARNING
    assert dp.logger.level == logging.WARNING


def test_verbose_wins_when_both_set():
    assert dp.configure_logging(verbose=True, quiet=True) == logging.DEBUG


def test_configure_is_idempotent_no_handler_stacking():
    dp.configure_logging(verbose=False, quiet=False)
    dp.configure_logging(verbose=False, quiet=False)
    assert len(dp.logger.handlers) == 1


def test_propagation_stays_enabled():
    dp.configure_logging(verbose=False, quiet=False)
    assert dp.logger.propagate is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_logging.py -q`
Expected: FAIL — `module 'dnd_pipeline' has no attribute 'configure_logging'`

- [ ] **Step 3: Write minimal implementation**

In `dnd_pipeline.py`, add `import logging` to the import block (alphabetical, near `import json`). Then after the imports and before the `# Utils` separator, add:

```python
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
```

(`sys` is already imported.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_logging.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add dnd_pipeline.py tests/test_logging.py
git commit -m "feat: add module logger and configure_logging"
```

---

### Task 2: CLI flags + wire into main

**Files:**
- Modify: `dnd_pipeline.py` — `main()` (the `argparse` setup ~line 1440 and the dispatch ~line 1477)
- Test: `tests/test_logging.py` (append)

**Interfaces:**
- Consumes: `configure_logging(verbose, quiet)` from Task 1.
- Produces: every subcommand accepts `--verbose`/`-v` and `--quiet`/`-q` (mutually exclusive); `main` calls `configure_logging(args.verbose, args.quiet)` immediately after `parse_args`, before dispatching.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_logging.py`:

```python
def test_main_verbose_flag_sets_debug_level():
    # The "run" command will fail (no such dir), but logging is configured first.
    rc = dp.main(["run", "/nonexistent/session/dir", "--verbose"])
    assert rc == 1
    assert dp.logger.level == logging.DEBUG


def test_main_quiet_flag_sets_warning_level():
    rc = dp.main(["run", "/nonexistent/session/dir", "--quiet"])
    assert rc == 1
    assert dp.logger.level == logging.WARNING


def test_main_default_is_info_level():
    rc = dp.main(["run", "/nonexistent/session/dir"])
    assert rc == 1
    assert dp.logger.level == logging.INFO
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_logging.py -k main_ -q`
Expected: FAIL — `argument --verbose` unrecognized (SystemExit) or level not set.

- [ ] **Step 3: Add the shared flag parser and wire main**

In `main()`, before the subparsers are created, add a shared parent parser and apply it to every subparser via `parents=[...]`. Locate the line `sub = parser.add_subparsers(dest="command", required=True)` and insert just above it:

```python
    verbosity = argparse.ArgumentParser(add_help=False)
    vgroup = verbosity.add_mutually_exclusive_group()
    vgroup.add_argument("--verbose", "-v", action="store_true", help="show per-line transcript and debug detail")
    vgroup.add_argument("--quiet", "-q", action="store_true", help="only warnings and errors")
```

Then add `parents=[verbosity]` to each `sub.add_parser(...)` call. The four become:

```python
    p_run = sub.add_parser("run", parents=[verbosity], help="full pipeline: transcribe → clean → chunks → prompts")
    ...
    p_rebuild = sub.add_parser("rebuild", parents=[verbosity], help="rebuild clean/chunks/prompts from existing raw JSONL; no transcription")
    ...
    p_ai = sub.add_parser("prepare-ai", parents=[verbosity], help="create chunks and manual AI prompts from clean.jsonl")
    ...
    p_report = sub.add_parser("build-report", parents=[verbosity], help="build reports from manual_ai_results/*.json")
```

(Keep each subparser's existing arguments unchanged — only add the `parents=` kwarg.)

Then, right after `args = parser.parse_args(argv)`, add:

```python
    configure_logging(getattr(args, "verbose", False), getattr(args, "quiet", False))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_logging.py -q`
Expected: PASS (9 passed)

- [ ] **Step 5: Commit**

```bash
git add dnd_pipeline.py tests/test_logging.py
git commit -m "feat: add --verbose/--quiet flags wired to configure_logging"
```

---

### Task 3: Convert progress prints to logger calls

**Files:**
- Modify: `dnd_pipeline.py` — 33 `print(...)` calls (all except the three exception-block prints handled in Task 4)
- Test: `tests/test_logging.py` (append)

**Interfaces:**
- Consumes: `logger` from Task 1.
- Produces: progress/warning output now goes through `logger`. No signature changes.

The conversion is mechanical: replace `print(` with `logger.<level>(` keeping the same f-string argument. Apply this exact line→level mapping:

| Lines | Level |
|-------|-------|
| 190, 616 | `logger.warning` |
| 686 | `logger.debug` |
| 287, 291, 292, 346, 349, 410, 634, 637, 639, 641, 689, 700, 709, 711, 726, 727, 829, 916, 1113, 1145, 1146, 1147, 1210, 1267, 1268, 1346, 1347, 1392, 1417, 1428 | `logger.info` |

Special case for line 686: it currently is `print(f"  [{fmt_hms(start)}] {text}", flush=True)`. Drop the `flush=True` argument (invalid for `logger`): it becomes `logger.debug(f"  [{fmt_hms(start)}] {text}")`.

Do NOT touch lines 1283, 1482, 1485 (Task 4 owns those).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_logging.py`:

```python
import json
from pathlib import Path


def test_unknown_profile_logs_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="dnd_pipeline"):
        dp.apply_quality_profile({"quality_profile": "definitely_not_a_profile"})
    assert any("неизвестный" in r.message and r.levelno == logging.WARNING
               for r in caplog.records)


def test_missing_track_file_logs_warning(tmp_path, caplog):
    track = {"file": "nope.wav", "speaker": "X", "index": 0}
    cfg = {"session_name": "s"}
    paths = dp.build_paths(tmp_path, "s")
    dp.ensure_dirs(paths)
    with caplog.at_level(logging.WARNING, logger="dnd_pipeline"):
        rows = dp.transcribe_track(None, tmp_path, track, cfg, paths)
    assert rows == []
    assert any("ПРОПУСК" in r.message and r.levelno == logging.WARNING
               for r in caplog.records)


def test_make_chunks_logs_info(tmp_path, caplog):
    paths = dp.build_paths(tmp_path, "s")
    dp.ensure_dirs(paths)
    rows = [{
        "start": 0.0, "end": 2.0, "start_hms": dp.fmt_hms_ms(0.0),
        "speaker": "A", "character": "A", "text": "привет мир",
    }]
    cfg = {"session_name": "s", "chunk_minutes": 10}
    with caplog.at_level(logging.INFO, logger="dnd_pipeline"):
        dp.make_chunks(rows, cfg, paths)
    assert any("Чанков создано" in r.message and r.levelno == logging.INFO
               for r in caplog.records)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_logging.py -k "warning or info" -q`
Expected: FAIL — the messages are still going to `print`, so no matching log records.

- [ ] **Step 3: Convert the 33 prints**

Apply the mapping table above. Representative before/after:

```python
# 190 (before)  print(f"Предупреждение: неизвестный quality_profile={profile!r}; использую balanced")
#      (after)  logger.warning(f"Предупреждение: неизвестный quality_profile={profile!r}; использую balanced")

# 616 (before)  print(f"ПРОПУСК: файл не найден — {audio_path}")
#      (after)  logger.warning(f"ПРОПУСК: файл не найден — {audio_path}")

# 686 (before)  print(f"  [{fmt_hms(start)}] {text}", flush=True)
#      (after)  logger.debug(f"  [{fmt_hms(start)}] {text}")

# 1210 (before) print(f"Чанков создано: {len(out_paths)} → {paths.chunks_dir}")
#       (after) logger.info(f"Чанков создано: {len(out_paths)} → {paths.chunks_dir}")
```

Convert every line in the table the same way (`print(` → `logger.<level>(`, preserving the f-string). The 30 INFO lines all follow the `print(...)` → `logger.info(...)` pattern.

- [ ] **Step 4: Run tests to verify they pass + no regressions**

Run: `python3 -m pytest -q`
Expected: PASS (all prior tests plus the 3 new ones; nothing regresses).

- [ ] **Step 5: Commit**

```bash
git add dnd_pipeline.py tests/test_logging.py
git commit -m "refactor: route progress output through logger at mapped levels"
```

---

### Task 4: Narrow exception blocks + log them

**Files:**
- Modify: `dnd_pipeline.py` — `read_manual_results` (~1279-1283), `build_reports` weight parse (~1331-1334), `main` handler (~1478-1486)
- Test: `tests/test_logging.py` (append)

**Interfaces:**
- Consumes: `logger` from Task 1.
- Produces: narrowed excepts; `read_manual_results` logs WARNING per bad file and continues; `main` logs ERROR and returns 1.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_logging.py`:

```python
def test_read_manual_results_skips_bad_json_and_logs(tmp_path, caplog):
    paths = dp.build_paths(tmp_path, "s")
    dp.ensure_dirs(paths)
    (paths.manual_ai_dir / "good.json").write_text(
        json.dumps({"summary": "ok", "chunk_index": 0}), encoding="utf-8")
    (paths.manual_ai_dir / "bad.json").write_text("{ not valid json", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="dnd_pipeline"):
        rows = dp.read_manual_results(paths)
    assert len(rows) == 1
    assert rows[0]["summary"] == "ok"
    assert any("Не смог прочитать" in r.message and r.levelno == logging.WARNING
               for r in caplog.records)


def test_main_logs_error_and_returns_1_on_failure(caplog):
    with caplog.at_level(logging.ERROR, logger="dnd_pipeline"):
        rc = dp.main(["run", "/nonexistent/session/dir"])
    assert rc == 1
    assert any(r.levelno == logging.ERROR for r in caplog.records)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_logging.py -k "bad_json or returns_1" -q`
Expected: FAIL — current code prints (no log records captured) for these paths.

- [ ] **Step 3: Narrow and log**

`read_manual_results` — change the `except` block:

```python
        except (OSError, ValueError) as e:
            logger.warning(f"Не смог прочитать {p}: {e}")
```

`build_reports` weight parse — narrow the except (keep the silent fallback):

```python
        try:
            weight_int = int(weight)
        except (ValueError, TypeError):
            weight_int = 1
```

`main` handler — keep broad but log via logger; the `KeyboardInterrupt` branch also moves to the logger:

```python
    try:
        args.func(args)
        return 0
    except KeyboardInterrupt:
        logger.info("\nОстановлено пользователем.")
        return 130
    except Exception as e:  # noqa: BLE001 — top-level CLI handler: catch all to exit cleanly
        logger.error(f"Ошибка: {e}")
        return 1
```

- [ ] **Step 4: Run tests to verify they pass + full suite**

Run: `python3 -m pytest -q`
Expected: PASS (all tests; the two new ones included).

- [ ] **Step 5: Commit**

```bash
git add dnd_pipeline.py tests/test_logging.py
git commit -m "refactor: narrow read_manual_results/weight excepts, log via logger"
```

---

## Notes for the implementer

- Run the whole suite (`python3 -m pytest -q`) after Tasks 3 and 4; the prior 66 tests must not regress.
- Line numbers drift as you edit; rely on the surrounding code/function names, not exact lines.
- Do not add file handlers, colors, or rotation — stdout + levels only (YAGNI).
- `caplog` captures records via propagation; tests use `caplog.at_level(..., logger="dnd_pipeline")`.
