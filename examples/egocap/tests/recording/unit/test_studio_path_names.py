# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from egocap.studio.studio_path_names import (
    canonicalize_studio_dir_name,
    sanitize_studio_dir_name,
)


def test_sanitize_empty_returns_empty():
    assert sanitize_studio_dir_name("") == ""
    assert sanitize_studio_dir_name(None) == ""


def test_sanitize_unmatched_passes_through():
    assert sanitize_studio_dir_name("_unmatched") == "_unmatched"


def test_sanitize_special_characters():
    assert sanitize_studio_dir_name("Kitchen & Bath?") == "Kitchen_Bath"
    assert sanitize_studio_dir_name("50% Off #1") == "50_Off_1"
    assert sanitize_studio_dir_name("a/b\\c") == "a_b_c"


def test_sanitize_collapses_whitespace_and_hyphens():
    assert sanitize_studio_dir_name("Dining   Room") == "Dining_Room"
    assert sanitize_studio_dir_name("Dining-Room") == "Dining_Room"
    assert sanitize_studio_dir_name("--Dining--") == "Dining"


def test_sanitize_all_special_chars_becomes_unknown():
    assert sanitize_studio_dir_name("###") == "unknown"


def test_canonicalize_empty_or_unmatched_unchanged():
    assert canonicalize_studio_dir_name("", {"Kitchen"}) == ""
    assert canonicalize_studio_dir_name("_unmatched", {"Kitchen"}) == "_unmatched"


def test_canonicalize_no_known_studios_returns_sanitized():
    assert canonicalize_studio_dir_name("Dining Room", None) == "Dining_Room"
    assert canonicalize_studio_dir_name("Dining Room", frozenset()) == "Dining_Room"


def test_canonicalize_matches_case_insensitively():
    known = {"Dining_Room"}
    assert canonicalize_studio_dir_name("dining_room", known) == "Dining_Room"
    assert canonicalize_studio_dir_name("Dining_Room", known) == "Dining_Room"


def test_canonicalize_unknown_studio_returns_sanitized():
    assert canonicalize_studio_dir_name("Garage", {"Dining_Room"}) == "Garage"
