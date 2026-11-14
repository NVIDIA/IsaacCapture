// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <camera_plugin/camera_source.hpp>

#include <cstddef>
#include <cstdint>
#include <span>

namespace plugins::camera
{

/**
 * @brief A camera that needs no hardware. Options: `width`, `height`, `fps`, `streams` and
 * `format` (bgra8, rgba8, gray8 or nv12).
 *
 * Frame n of stream s starts with n as eight little-endian bytes; every other byte of the
 * image is synthetic_fill(n, s). A reader can therefore check that it got the whole of the
 * frame it was told about.
 */
Backend synthetic_backend();

// The value of every byte of a synthetic image after its first eight.
constexpr std::byte synthetic_fill(std::uint64_t frame, std::size_t stream)
{
    return static_cast<std::byte>((frame + 17 * stream) & 0xff);
}

// The frame number a synthetic image starts with.
std::uint64_t synthetic_frame_number(std::span<const std::byte> image);

} // namespace plugins::camera
