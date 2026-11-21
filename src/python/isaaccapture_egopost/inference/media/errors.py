# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Errors raised by the video backends, importable without loading a backend."""


class UnsupportedVideoCodecError(ValueError):
    """Raised when a video does not use an approved input codec."""


class VideoDecoderUnavailableError(RuntimeError):
    """Raised when no approved decoder can decode a supported stream."""


class VideoDecodeError(RuntimeError):
    """Raised when probing or streaming a supported video fails."""
