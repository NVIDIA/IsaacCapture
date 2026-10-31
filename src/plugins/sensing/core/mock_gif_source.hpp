// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Mock backend only. An animated GIF decoded once and held in device memory, so the
// mock camera can show real moving images.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace plugins
{
namespace sensing
{

/// Time in seconds at which each GIF frame stops showing: the running total of the frame delays,
/// each clamped to at least 20 ms, as browsers do for the tiny delays some encoders write.
std::vector<double> gif_frame_ends(const std::vector<int>& delays_ms);

/// Index of the frame showing `seconds` into the animation, which loops over its whole length.
size_t gif_frame_index(const std::vector<double>& frame_ends, double seconds);

class MockGifSource
{
public:
    /// Transparent pixels are composited onto this gray, so an emoji or sticker without a
    /// background reads as an image rather than as whatever colour sits under its transparency.
    static constexpr uint8_t kBackground = 48;

    /// Decodes every frame to opaque RGBA8 and uploads them to the current CUDA device.
    /// @throws std::runtime_error if the file cannot be read or is not a GIF.
    explicit MockGifSource(const std::string& path);
    ~MockGifSource();

    MockGifSource(const MockGifSource&) = delete;
    MockGifSource& operator=(const MockGifSource&) = delete;

    uint32_t width() const
    {
        return m_width;
    }
    uint32_t height() const
    {
        return m_height;
    }
    size_t frame_count() const
    {
        return m_frame_ends.size();
    }
    /// Row stride of every frame, in bytes. Rows are tightly packed.
    size_t pitch() const
    {
        return static_cast<size_t>(m_width) * 4;
    }
    /// Device pointer to one RGBA8 frame.
    const uint8_t* frame(size_t index) const;
    size_t frame_index_at(double seconds) const
    {
        return gif_frame_index(m_frame_ends, seconds);
    }

private:
    uint32_t m_width = 0;
    uint32_t m_height = 0;
    std::vector<double> m_frame_ends;
    uint8_t* m_device_frames = nullptr;
};

} // namespace sensing
} // namespace plugins
