<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CI helper tests

These standard-library unittest checks run through pre-commit, without CMake or
Isaac Capture runtime dependencies. Keep GitHub API fixtures hermetic.

Pre-commit rejects an unstaged `.pre-commit-config.yaml`; commit hook registration
with its tests and keep that config clean for a separate implementation commit.
