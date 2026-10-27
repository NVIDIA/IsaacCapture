<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# WebXR client dependencies

Keep `react` and `react-dom` on the same minor release supported by
`@react-three/fiber`. Validate dependency-range changes with a clean npm install;
do not bypass peer checks with `--force` or `--legacy-peer-deps`.

Complete client formatting before staging and running pre-commit; do not run
formatters concurrently with hooks, which detect any file changes during a run.

Session errors restored after reload must retain banner priority over asynchronous
startup notices and status cleanup until a stream connects successfully.
