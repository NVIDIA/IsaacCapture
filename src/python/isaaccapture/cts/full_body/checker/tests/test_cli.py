# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``--list-checks``, which answers what the checker asks before a take exists."""

from __future__ import annotations

import pytest

from full_body_cts.checks import build_all
from full_body_cts.cli import main


def test_list_checks_needs_no_recording(capsys):
    assert main(["--list-checks"]) == 0
    assert len(capsys.readouterr().out.splitlines()) == len(build_all())


def test_list_checks_prints_the_group(capsys):
    main(["--list-checks"])
    lines = capsys.readouterr().out.splitlines()
    groups = {line.split()[1] for line in lines}
    assert groups == {"signal", "geometry", "posture"}


def test_a_run_still_requires_a_recording():
    with pytest.raises(SystemExit):
        main([])
