// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// NVENC H.264 encoder for low-latency streaming. P4 preset +
// ULTRA_LOW_LATENCY tuning + CBR + bf=0 + extraOutputDelay=0.

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace camera_viz::codec
{

enum class PixelFormat
{
    kRGBA8,
    kBGRA8,
};

struct EncoderConfig
{
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t bitrate_bps = 15'000'000;
    uint32_t fps = 30;
    // GOP / IDR period. 0 → fps*5.
    uint32_t gop = 0;
    int gpu_id = 0;
    PixelFormat pixel_format = PixelFormat::kRGBA8;
};

class H264Encoder
{
public:
    explicit H264Encoder(const EncoderConfig& cfg);
    ~H264Encoder();

    H264Encoder(const H264Encoder&) = delete;
    H264Encoder& operator=(const H264Encoder&) = delete;
    H264Encoder(H264Encoder&&) = delete;
    H264Encoder& operator=(H264Encoder&&) = delete;

    // Encode one GPU-resident RGBA8 frame. ``rgba_device_ptr`` points to
    // the top-left of a HxWx4 buffer; ``row_pitch_bytes`` is the byte
    // stride between rows (== W*4 for tightly packed).
    // Returns an Annex-B byte vector; empty during NVENC warmup.
    // Throws on failure; caller should drop and reconstruct.
    std::vector<uint8_t> encode(uintptr_t rgba_device_ptr, std::size_t row_pitch_bytes);

    // Flush remaining packets from NVENC's internal queue.
    std::vector<uint8_t> end_of_stream();

    uint32_t width() const noexcept;
    uint32_t height() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace camera_viz::codec
