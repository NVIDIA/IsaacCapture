# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Supervision of the Manus glove calibration worker, and storage of the MCAL files it exports.

The worker executable is ``manus_glove_calibrator`` (see ``src/plugins/manus/calibration``). This package
has no dependencies, so a UI or a server can embed it without a vendor SDK being installed.
"""
