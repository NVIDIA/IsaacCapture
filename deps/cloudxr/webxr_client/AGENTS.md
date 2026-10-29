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
