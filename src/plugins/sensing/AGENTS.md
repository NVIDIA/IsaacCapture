<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SENSING scripts

Keep launchers that primarily compose Docker commands in Bash.
Keep machine-readable output on stdout and diagnostics on stderr.

Maintain one root README for the plugin overview, source map, and config provenance.
Link to the published GitHub Pages user guide, not its RST source.
Setup procedures and troubleshooting belong in that guide, not in both files.

Keep host and container quick starts to setup plus the same direct camera command.
Container setup runs on the host and owns image creation and runtime mounts;
camera runtime checks must not require build SDKs or cached vendor packages.

The quick-start camera command must open a live visual preview, not only print FPS.
Container preview must carry the desktop display and authorization into the image.

Support SSH launch with a local DISPLAY: when XAUTHORITY is unset, check the
current user's GDM authority file as well as ~/.Xauthority.
