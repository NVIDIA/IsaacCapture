# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Ego Capture UI web backend: a small FastAPI application that reports the status of
one backend-owned CaptureServer attempt to a read-only browser status page.

This package intentionally has no recording, calibration or control surface
yet -- it reports status only.
"""
