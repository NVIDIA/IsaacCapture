# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from egocap.core.config_utils import sanitize_task_name


def test_sanitize_task_name_examples():
    assert sanitize_task_name("Lego Disassembly Toss v2") == "Lego_Disassembly_Toss_v2"
    assert sanitize_task_name("pick & place!") == "pick_place"


def test_sanitize_task_name_empty_returns_unnamed():
    assert sanitize_task_name("") == "unnamed"
    assert sanitize_task_name(None) == "unnamed"


def test_sanitize_task_name_all_special_chars_returns_unnamed():
    assert sanitize_task_name("!!!") == "unnamed"


def test_sanitize_task_name_collapses_runs_and_strips_edges():
    assert sanitize_task_name("--pick---place--") == "pick_place"
    assert sanitize_task_name("  spaced   out  ") == "spaced_out"


def test_sanitize_task_name_keeps_hyphens_and_underscores():
    assert sanitize_task_name("already-snake_case") == "already-snake_case"
