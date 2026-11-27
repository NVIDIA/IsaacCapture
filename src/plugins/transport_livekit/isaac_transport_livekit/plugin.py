# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Registration is SDK-free; construct only the selected adapter."""

from isaac_media_transport.transport_plugins import TransportPlugin


def create():
    from .livekit_transport import LiveKitTransport

    return LiveKitTransport()


PLUGIN = TransportPlugin("livekit", 1, create)
