<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# WebXR client dependencies

Validate IWER/DevUI upgrades with stationary controller-gizmo taps and drags in
an active, visible XR session. Successful connection and loader tests do not
verify controller manipulation.

Keep `react` and `react-dom` on the same minor release supported by
`@react-three/fiber`. Validate dependency-range changes with a clean npm install;
do not bypass peer checks with `--force` or `--legacy-peer-deps`.

Complete client formatting before staging and running pre-commit; do not run
formatters concurrently with hooks, which detect any file changes during a run.

Session errors restored after reload must retain banner priority over asynchronous
startup notices and status cleanup until a stream connects successfully.

Selecting Custom preserves current profile-driven settings; named profiles apply
their defaults.

Connect readiness must remain blocked during emulator loading and capability checks,
even when form validation or XR-store notifications update the button. Cover startup
with a delayed emulator bundle and click as soon as Connect becomes available.
