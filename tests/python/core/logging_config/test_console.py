# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The one console handler: its filter, its threshold, and its colours."""

from __future__ import annotations

import io
import logging
import os
import re

import pytest

from isaaccapture import logging_config
from isaaccapture.logging_config import _console, _core

_RESET = "\033[0m"
_YELLOW = "\033[33m"
_RED = "\033[31m"


@pytest.fixture(autouse=True)
def restore_console():
    """This process shares the one handler under test with every other test."""
    handler = _console.ensure_handler()
    level = handler.level
    formatter = handler.formatter
    console_format = dict(_console._console_format)
    colors = dict(_console._logger_colors)
    color_lookup = _console._logger_color
    published = os.environ.get("ISAACCAPTURE_LOG_LEVEL")
    yield
    logging_config.set_console_logger_name_filter(None)
    logging_config.set_console_content_filter(None)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    _console._console_format.clear()
    _console._console_format.update(console_format)
    _console._logger_colors.clear()
    _console._logger_colors.update(colors)
    _console._logger_color = color_lookup
    if published is None:
        os.environ.pop("ISAACCAPTURE_LOG_LEVEL", None)
    else:
        os.environ["ISAACCAPTURE_LOG_LEVEL"] = published


def record(name="isaaccapture.plugins.manus", message="gloves %s", args=("ready",)):
    return logging.LogRecord(name, logging.INFO, __file__, 1, message, args, None)


class TestLoggerNameFilter:
    def test_keeps_the_names_and_their_descendants(self):
        handler = _console.ensure_handler()
        logging_config.set_console_logger_name_filter({"isaaccapture.plugins.manus"})
        assert handler.filter(record())
        assert handler.filter(record(name="isaaccapture.plugins.manus.ManusTracker"))
        # Neither a name that only shares the leading text nor an ancestor passes.
        assert not handler.filter(record(name="isaaccapture.plugins.manusx"))
        assert not handler.filter(record(name="isaaccapture.plugins"))

    def test_a_name_left_out_of_the_next_set_no_longer_passes(self):
        handler = _console.ensure_handler()
        logging_config.set_console_logger_name_filter({"isaaccapture.plugins.manus"})
        assert handler.filter(record())

        logging_config.set_console_logger_name_filter({"isaaccapture.core"})
        assert not handler.filter(record())
        assert handler.filter(record(name="isaaccapture.core.Session"))

    def test_an_empty_set_passes_no_logger_and_none_passes_all(self):
        handler = _console.ensure_handler()
        logging_config.set_console_logger_name_filter(set())
        assert not handler.filter(record())

        logging_config.set_console_logger_name_filter(None)
        assert handler.filter(record())

    def test_editing_the_passed_set_afterwards_changes_nothing(self):
        handler = _console.ensure_handler()
        names = {"isaaccapture.core"}
        logging_config.set_console_logger_name_filter(names)
        names.add("isaaccapture.plugins.manus")
        assert not handler.filter(record())


class TestContentFilter:
    def test_matches_the_formatted_message(self):
        handler = _console.ensure_handler()
        assert logging_config.set_console_content_filter("ready") is True
        assert handler.filter(record())  # "ready" only exists after args are applied
        assert not handler.filter(record(args=("absent",)))

        assert logging_config.set_console_content_filter(None) is True
        assert handler.filter(record(args=("absent",)))

    def test_a_new_pattern_decides_messages_seen_before(self):
        handler = _console.ensure_handler()
        logging_config.set_console_content_filter("ready")
        assert handler.filter(record())

        logging_config.set_console_content_filter("absent")
        assert not handler.filter(record())

    def test_an_invalid_pattern_is_rejected_and_keeps_the_current_filter(self, caplog):
        handler = _console.ensure_handler()
        logging_config.set_console_content_filter("ready")

        assert logging_config.set_console_content_filter("(unclosed") is False

        assert [entry.levelno for entry in caplog.records] == [logging.ERROR]
        # An unfiltered console is not the right answer to a bad argument.
        assert not handler.filter(record(args=("absent",)))


class TestBothFilters:
    def test_a_record_below_warning_must_pass_both(self):
        handler = _console.ensure_handler()
        logging_config.set_console_logger_name_filter({"isaaccapture.plugins.manus"})
        logging_config.set_console_content_filter("ready")
        assert handler.filter(record())
        assert not handler.filter(record(args=("absent",)))
        assert not handler.filter(record(name="isaaccapture.core.Session"))


class TestConsoleLevel:
    def test_publishes_the_level_a_plugin_executable_will_read(self):
        # A plugin that falls back to local sinks reads this variable.
        logging_config.set_console_level("debug")
        assert _console.ensure_handler().level == logging.DEBUG
        assert os.environ["ISAACCAPTURE_LOG_LEVEL"] == "debug"

        logging_config.set_console_level("TRACE")
        assert _console.ensure_handler().level == _core.TRACE
        assert os.environ["ISAACCAPTURE_LOG_LEVEL"] == "trace"

    # spdlog's "warn" spelling is deliberately not a level name here, and a
    # stdlib level number is not a name either.
    @pytest.mark.parametrize("level", ["loud", "warn", logging.DEBUG, _core.TRACE])
    def test_an_invalid_level_is_logged_and_keeps_the_threshold(self, level, caplog):
        handler = _console.ensure_handler()
        before = (handler.level, os.environ.get("ISAACCAPTURE_LOG_LEVEL"))
        logging_config.set_console_level(level)
        assert (handler.level, os.environ.get("ISAACCAPTURE_LOG_LEVEL")) == before
        assert [entry.levelno for entry in caplog.records] == [logging.ERROR]


class _Tty(io.StringIO):
    def isatty(self) -> bool:
        return True


def render(
    stream, level=logging.INFO, name="isaaccapture.core.Probe", **options
) -> str:
    handler = logging.StreamHandler(stream)
    formatter = _console._LoggerNameColorFormatter(
        _core.LINE_FORMAT,
        datefmt=_core.DATE_FORMAT,
        handler=handler,
        **options,
    )
    return formatter.format(
        logging.LogRecord(name, level, __file__, 1, "msg", None, None)
    )


class TestColourFormatter:
    def test_warnings_and_errors_stand_out_on_a_terminal(self):
        assert render(_Tty(), logging.WARNING).startswith(_YELLOW)
        assert render(_Tty(), logging.WARNING).endswith(_RESET)
        assert render(_Tty(), logging.ERROR).startswith(_RED)
        assert render(_Tty(), logging.CRITICAL).startswith(_RED)

    def test_ordinary_records_are_left_plain(self):
        assert "\033[" not in render(_Tty(), logging.INFO)

    def test_nothing_is_coloured_off_a_terminal(self):
        # Escapes anywhere else are noise, and corrupt whatever parses the output.
        assert "\033[" not in render(io.StringIO(), logging.ERROR)

    def test_a_logger_colour_applies_inside_the_line_and_the_level_resumes(self):
        logging_config.set_logger_colors({"isaaccapture.core.Probe": "\033[36m"})
        line = render(_Tty(), logging.WARNING)
        assert f"[\033[36misaaccapture.core.Probe{_YELLOW}]" in line
        assert line.startswith(_YELLOW) and line.endswith(_RESET)

    def test_a_logger_colour_on_an_uncoloured_level_resets_after_the_name(self):
        logging_config.set_logger_colors({"isaaccapture.core.Probe": "\033[36m"})
        assert f"[\033[36misaaccapture.core.Probe{_RESET}]" in render(_Tty())

    def test_a_logger_colour_also_applies_to_its_descendants(self):
        logging_config.set_logger_colors({"isaaccapture.core": "\033[36m"})
        assert f"[\033[36misaaccapture.core.Probe{_RESET}]" in render(_Tty())
        # A name that only shares the leading text is not a descendant.
        assert "\033[36m" not in render(_Tty(), name="isaaccapture.corex.Probe")

    def test_the_most_specific_colour_wins(self):
        logging_config.set_logger_colors(
            {"isaaccapture.core": "\033[36m", "isaaccapture.core.Probe": "\033[35m"}
        )
        assert f"[\033[35misaaccapture.core.Probe{_RESET}]" in render(_Tty())

    def test_a_change_reaches_names_already_looked_up(self):
        logging_config.set_logger_colors({"isaaccapture.core": "\033[36m"})
        assert f"[\033[36misaaccapture.core.Probe{_RESET}]" in render(_Tty())
        logging_config.set_logger_colors({"isaaccapture.core": "\033[35m"})
        assert f"[\033[35misaaccapture.core.Probe{_RESET}]" in render(_Tty())
        logging_config.set_logger_colors({"isaaccapture.core": None})
        assert "\033[" not in render(_Tty())
        # Lookups never write what they resolve into the configured colours.
        assert "isaaccapture.core.Probe" not in _console._logger_colors

    def test_oob_progress_logger_is_cyan_by_default(self):
        name = "isaaccapture.cloudxr.oob_teleop_env"
        assert f"[\033[36m{name}{_RESET}]" in render(_Tty(), name=name)

    def test_the_last_segment_keeps_the_full_names_colour(self):
        logging_config.set_logger_colors({"isaaccapture.core.Probe": "\033[36m"})
        assert f"[\033[36mProbe{_RESET}]" in render(_Tty(), use_short_name=True)

    def test_the_record_is_handed_back_unmodified(self):
        # The file handler shares the record and must stay escape-free.
        logging_config.set_logger_colors({"isaaccapture.core.Probe": "\033[36m"})
        handler = logging.StreamHandler(_Tty())
        formatter = _console._LoggerNameColorFormatter(
            _core.LINE_FORMAT,
            datefmt=_core.DATE_FORMAT,
            handler=handler,
        )
        entry = logging.LogRecord(
            "isaaccapture.core.Probe", logging.ERROR, __file__, 1, "msg", None, None
        )
        formatter.format(entry)
        assert entry.name == "isaaccapture.core.Probe"


class TestLoggerColours:
    @pytest.mark.parametrize(
        "colour", ["\033[36m", "\033[38;2;255;136;0m", "\033[1m\033[36m"]
    )
    def test_accepts_sgr(self, colour):
        logging_config.set_logger_colors({"isaaccapture.core.Probe": colour})
        assert _console._logger_colors["isaaccapture.core.Probe"] == colour

    @pytest.mark.parametrize(
        "colour", ["red", "\033]0;title\007", "\033[36m\n", "", "\033[2J"]
    )
    def test_anything_that_is_not_purely_sgr_is_logged_and_skipped(
        self, colour, caplog
    ):
        # A registered value is written to the terminal verbatim.
        logging_config.set_logger_colors({"isaaccapture.core.Probe": colour})
        assert "isaaccapture.core.Probe" not in _console._logger_colors
        assert [entry.levelno for entry in caplog.records] == [logging.ERROR]

    def test_none_drops_a_colour(self):
        logging_config.set_logger_colors({"isaaccapture.core.Probe": "\033[36m"})
        logging_config.set_logger_colors({"isaaccapture.core.Probe": None})
        assert "isaaccapture.core.Probe" not in _console._logger_colors


def probe() -> logging.LogRecord:
    return logging.LogRecord(
        "isaaccapture.core.Probe", logging.INFO, __file__, 1, "msg", None, None
    )


class TestConsoleFormat:
    def test_defaults_are_time_level_and_the_last_name_segment(self):
        assert re.fullmatch(
            r"\[\d{2}:\d{2}:\d{2}\.\d{3}\] \[INFO \] \[Probe\] msg",
            _console.ensure_handler().format(probe()),
        )

    def test_only_the_given_keys_change(self):
        logging_config.set_console_format({"show_pid": True, "use_short_name": False})
        logging_config.set_console_format({"show_time": False})
        assert re.fullmatch(
            r"\[INFO \] \[isaaccapture\.core\.Probe\] \[pid:\d+\] msg",
            _console.ensure_handler().format(probe()),
        )

    def test_every_optional_column_can_be_dropped(self):
        logging_config.set_console_format(
            {
                "show_time": False,
                "show_level": False,
                "show_logger_name": False,
                "show_pid": False,
            }
        )
        assert _console.ensure_handler().format(probe()) == "msg"

    def test_last_segment_shortens_only_the_console_line(self):
        logging_config.set_console_format({"show_time": False})
        entry = probe()
        assert _console.ensure_handler().format(entry) == "[INFO ] [Probe] msg"
        # The file handler formats the same record next.
        assert entry.name == "isaaccapture.core.Probe"

    def test_an_unknown_key_is_logged_and_skipped(self, caplog):
        logging_config.set_console_format({"colour": True, "show_time": False})
        assert "colour" not in _console._console_format
        assert _console._console_format["show_time"] is False
        [record] = caplog.records
        assert record.levelno == logging.ERROR
        assert "'colour'" in record.getMessage()

    def test_a_wrong_type_is_logged_and_skipped(self, caplog):
        logging_config.set_console_format({"show_pid": 1, "show_time": False})
        assert _console._console_format["show_pid"] is False
        assert _console._console_format["show_time"] is False
        [record] = caplog.records
        assert record.levelno == logging.ERROR
        assert "'show_pid' must be a bool, got 1" in record.getMessage()
