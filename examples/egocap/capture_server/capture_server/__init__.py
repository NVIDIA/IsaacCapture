# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""CaptureServer: standalone WebSocket tracking/camera source.

No external monorepo or Ego Capture UI import anywhere in this package. Importing this
package never opens a socket, a device, or an SDK.
"""
