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
