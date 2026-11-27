<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Secure directory policy

- Validate each opened parent before creating children; reject symlinks and `..`
  before normalization. Create missing directories privately. Sticky ancestors
  protect entries but do not make a writable final directory safe.
- Traversal requires search permission, not read permission; use traversal-only
  descriptors where supported and test as a non-root caller.
- Preserve the same path policy in Python `logging_config/_core.py` and C++.
  Validation-only callers must never create a missing component.
