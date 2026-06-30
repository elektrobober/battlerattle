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
