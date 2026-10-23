<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-OpenMDW-1.1
-->

# EgoPost agent notes

- Run repository-wide pre-commit with write access to the repository; some hooks
  open every checked file for writing even when its contents need no changes.
- Keep dependency security updates consistent with explicit installs in
  `inference/docker/`; quote version ranges in shell commands.
