# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A pure-Python drop-in shim for ``isaacteleop``: no CloudXR runtime, no OpenXR, no native
plugins, no headset, no browser tab. Covers exactly the import/usage surface documented in
``GumanIsaacTeleopUsage.md`` (see that file for the full spec this package implements against).

Install this package ahead of the real ``isaacteleop``/``isaaccapture`` on ``PYTHONPATH`` (or in
an environment that never installs the real one) to run a consumer like Guman's
``IsaacTeleopBackend`` against synthetic head/controller/hand/joint-state data.
"""
