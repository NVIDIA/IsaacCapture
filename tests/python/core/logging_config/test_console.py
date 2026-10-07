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
    colors = dict(_console._logger_colors)
    published = os.environ.get("ISAACCAPTURE_LOG_LEVEL")
    yield
    logging_config.set_console_filter(None)
    handler.setLevel(level)
    _console._logger_colors.clear()
    _console._logger_colors.update(colors)
    if published is None:
        os.environ.pop("ISAACCAPTURE_LOG_LEVEL", None)
    else:
        os.environ["ISAACCAPTURE_LOG_LEVEL"] = published


def record(name="isaaccapture.plugins.manus", message="gloves %s", args=("ready",)):
    return logging.LogRecord(name, logging.INFO, __file__, 1, message, args, None)


class TestKeywordFilter:
    def test_rejects_an_unknown_target(self):
        with pytest.raises(ValueError):
            _console.KeywordFilter("x", target="somewhere")

    def test_logger_name_target_ignores_the_message(self):
        keep = _console.KeywordFilter("manus", target="logger_name")
        assert keep.filter(record())
        assert not keep.filter(record(name="isaaccapture.core.Session"))
        assert not keep.filter(
            record(name="isaaccapture.core.Session", message="manus", args=None)
        )

    def test_content_target_matches_the_formatted_message(self):
        keep = _console.KeywordFilter("ready", target="content")
        assert keep.filter(record())  # "ready" only exists after args are applied
        assert not keep.filter(record(message="gloves %s", args=("absent",)))

    def test_both_is_the_union(self):
        keep = _console.KeywordFilter("manus", target="both")
        assert keep.filter(
            record(name="isaaccapture.core.Session", message="manus", args=None)
        )
        assert keep.filter(record(message="gloves %s", args=("ready",)))
        assert not keep.filter(
            record(name="isaaccapture.core.Session", message="x", args=None)
        )


class TestConsoleFilterApi:
    def test_set_and_clear(self):
        handler = _console.ensure_handler()
        logging_config.set_console_filter("manus", target="logger_name")
        assert handler.filter(record())
        assert not handler.filter(record(name="isaaccapture.core.Session"))

        logging_config.set_console_filter(None)
        assert handler.filter(record(name="isaaccapture.core.Session"))

    def test_a_rejected_argument_leaves_the_previous_filter_in_place(self):
        handler = _console.ensure_handler()
        logging_config.set_console_filter("manus", target="logger_name")

        with pytest.raises(ValueError):
            logging_config.set_console_filter("manus", target="somewhere")
        with pytest.raises(re.error):
            logging_config.set_console_filter("(unclosed")

        # An unfiltered console is not the right answer to a bad argument.
        assert not handler.filter(record(name="isaaccapture.core.Session"))


class TestConsoleLevel:
    def test_publishes_the_level_a_plugin_executable_will_read(self):
        # A plugin that falls back to local sinks reads this variable.
        logging_config.set_console_level("debug")
        assert _console.ensure_handler().level == logging.DEBUG
        assert os.environ["ISAACCAPTURE_LOG_LEVEL"] == "debug"

        logging_config.set_console_level(_core.TRACE)
        assert os.environ["ISAACCAPTURE_LOG_LEVEL"] == "trace"

    def test_publishes_a_number_for_a_level_with_no_name(self):
        logging_config.set_console_level(23)
        assert os.environ["ISAACCAPTURE_LOG_LEVEL"] == "23"


class _Tty(io.StringIO):
    def isatty(self) -> bool:
        return True


def render(stream, level=logging.INFO, name="isaaccapture.core.Probe") -> str:
    handler = logging.StreamHandler(stream)
    formatter = _console._LoggerNameColorFormatter(
        _core.LINE_FORMAT,
        datefmt=_core.DATE_FORMAT,
        handler=handler,
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

    def test_oob_progress_logger_is_cyan_by_default(self):
        name = "isaaccapture.cloudxr.oob_teleop_env"
        assert f"[\033[36m{name}{_RESET}]" in render(_Tty(), name=name)

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
    def test_rejects_anything_that_is_not_purely_sgr(self, colour):
        # A registered value is written to the terminal verbatim.
        with pytest.raises(ValueError):
            logging_config.set_logger_colors({"isaaccapture.core.Probe": colour})

    def test_none_drops_a_colour(self):
        logging_config.set_logger_colors({"isaaccapture.core.Probe": "\033[36m"})
        logging_config.set_logger_colors({"isaaccapture.core.Probe": None})
        assert "isaaccapture.core.Probe" not in _console._logger_colors
