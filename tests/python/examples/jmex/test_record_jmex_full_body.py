# SPDX-FileCopyrightText: Copyright (c) 2026 j-mex. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The recording script's command line, where there is no Isaac Lab."""

from __future__ import annotations

import sys

import pytest

from isaaccapture_examples.jmex import record_jmex_full_body


def test_help_exits_cleanly_without_isaac_lab(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        record_jmex_full_body.main(["record_jmex_full_body", "--help"])

    assert exit_info.value.code == 0
    assert "--no-plugin" in capsys.readouterr().out
    assert not [name for name in sys.modules if name.split(".")[0] == "isaaclab"]
