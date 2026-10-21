<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# WebXR client dependencies

Keep `react` and `react-dom` on the same minor release supported by
`@react-three/fiber`. Validate dependency-range changes with a clean npm install;
do not bypass peer checks with `--force` or `--legacy-peer-deps`.

Synthetic XR browser checks use the dev server's hot reload. Do not edit or format client files while those checks run: remounting the app invalidates the session and input ownership observations.

Synthetic examples must start teleoperation through the existing client handler. Do not hard-code a controller ray to an assumed 3D panel position.

Long-running synthetic-input tests against MockCloudXR must supply decoded mock frames. Preserve the production decoder startup watchdog and input release on disconnect.
