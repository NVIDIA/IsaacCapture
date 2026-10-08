#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate isaacteleop module version reporting against CI-provided expected version."""

import os

import isaacteleop


if not hasattr(isaacteleop, "__version__"):
    raise SystemExit("isaacteleop module does not define __version__")

expected = os.environ.get("EXPECTED_ISAACTELEOP_VERSION", "").strip()
if not expected:
    raise SystemExit(
        "EXPECTED_ISAACTELEOP_VERSION is not set. This is a CI check: "
        "scripts/run_tests_with_cloudxr.sh sets it from the wheel name."
    )

reported = isaacteleop.__version__

print(f"isaacteleop.__version__: {reported}")
print(f"Expected version from CI: {expected}")

if reported != expected:
    raise SystemExit(
        f"Version mismatch: isaacteleop.__version__ ({reported}) != expected CI version ({expected})"
    )

print("Version check passed")
