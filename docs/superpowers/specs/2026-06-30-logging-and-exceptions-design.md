# Design: Full logging + narrowed exception handling

Date: 2026-06-30
Status: Approved

## Problem

`dnd_pipeline.py` uses 41 bare `print()` calls for all output, with no levels
and no way to control verbosity. On a full multi-track session the per-line live
transcript (`dnd_pipeline.py:686`) emits hundreds-to-thousands of lines, flooding
the terminal. There is also no quiet mode for the recurring per-session workflow.

Separately, three `except Exception` blocks swallow errors too broadly:
- `read_manual_results` (1282) catches everything while only file-read / JSON-parse
  failures are expected, and prints rather than logs.
- `build_reports` weight parse (1333) catches everything for an `int()` fallback.
- `main` (1484) is a top-level CLI handler (legitimately broad) but prints instead
  of logging.

(The two import-guard `except Exception` blocks at lines 37/42 are intentional
optional-dependency handling and are out of scope.)

## Goals

1. Replace all 41 `print()` calls with a module logger at appropriate levels.
2. Add `--verbose`/`-v` and `--quiet`/`-q` flags to control verbosity.
3. Narrow the two over-broad `except` blocks; route all three through logging.

## Non-goals

- File handlers, log rotation, colored output, structured/JSON logs (YAGNI).
- Touching the import-guard excepts (lines 37/42).
- Changing what data is written to output files (only console output changes).

## Design

### Unit 1 — module logger + `configure_logging`

- Module-level `logger = logging.getLogger("dnd_pipeline")`.
- `configure_logging(verbose: bool, quiet: bool) -> int` resolves and applies the
  console logging level, returning the chosen level (for testability):
  - `verbose` → `logging.DEBUG`
  - `quiet` → `logging.WARNING`
  - neither → `logging.INFO`
  - `verbose` takes precedence if both are somehow set (argparse will make them
    mutually exclusive, so this is a safety fallback).
- Installs a single `StreamHandler` to `stdout` with format `%(message)s` (clean
  output; the existing Russian prefixes like `Предупреждение:` / `ПРОПУСК:`
  already convey severity to the reader). Idempotent: clears prior handlers on
  the `dnd_pipeline` logger before adding, so repeated calls in tests don't stack.

### Unit 2 — level mapping for the 41 prints

| Level | Messages |
|-------|----------|
| DEBUG | Per-line live transcript (`dnd_pipeline.py:686`) — the high-volume flood, now hidden unless `--verbose`. |
| INFO  | Stage announcements, cache-hit notes, output paths, counts, repair/filter summaries, "Готово", "Rebuild готов", "Чанков создано", "Промпты созданы", "Отчёты собраны", "Quality report", model-load notes, the MLX VAD note. |
| WARNING | "Предупреждение: неизвестный quality_profile" (190), "ПРОПУСК: файл не найден" (616), "Не смог прочитать {p}" (read_manual_results). |
| ERROR | The exception caught in `main` (1484). |

`KeyboardInterrupt` in `main` stays a normal user-facing message (INFO/print-like)
and keeps returning 130.

### Unit 3 — narrowed exceptions

- `read_manual_results` (1282): `except (OSError, ValueError) as e:` →
  `logger.warning("Не смог прочитать %s: %s", p, e)`. `json.JSONDecodeError`
  subclasses `ValueError`, so malformed JSON is covered; one bad file does not
  abort the others.
- `build_reports` weight (1333): `except (ValueError, TypeError):` → keep the
  silent `weight_int = 1` fallback (expected non-numeric weights like "high").
- `main` (1484): keep the broad `except Exception` (a top-level CLI handler must
  catch all to return a clean exit code) but switch to
  `logger.error("Ошибка: %s", e)` and add a comment explaining why broad is
  intentional here. Returns 1 unchanged.

### Unit 4 — CLI flags

Add a mutually-exclusive `--verbose`/`-v` and `--quiet`/`-q` group. To avoid
repeating it on every subparser, add it to a shared parent parser passed via
`parents=[...]` to each subparser (or to the top-level parser). `main` reads the
parsed flags and calls `configure_logging` before dispatching `args.func`.

## Testing (TDD)

- `configure_logging`: verbose→DEBUG, quiet→WARNING, default→INFO, verbose-wins
  precedence. Assert the returned level and the effective `logger.level`.
- `caplog`-based:
  - unknown `quality_profile` logs a WARNING (via `apply_quality_profile`).
  - `read_manual_results` on a directory containing one malformed `.json` and one
    valid `.json` logs a WARNING and still returns the valid row.
  - `main(["run", ...])` on a config that triggers an error returns 1 and logs an
    ERROR record. (Use a minimal failing invocation, e.g. a nonexistent session
    dir / missing config, so no model is needed.)

## Risks

- Default console view is nearly identical (INFO) except the per-line transcript,
  now `--verbose`-only. This is the intended fix for terminal flooding; documented
  in README.
- Tests must not stack handlers — `configure_logging` clears prior handlers.
  `caplog` works on the logger regardless of handlers.
