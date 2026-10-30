// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// NVDEC H.264 decoder + NV12 → RGBA8 colour conversion. Zero-latency
// (no reorder buffer, decode-order output). Supports BT.709 limited
// and BT.601 full range.

#pragma once

#include <cstdint>
#include <memory>

namespace camera_viz::codec
{

struct DecoderConfig
{
    uint32_t width = 0;
    uint32_t height = 0;
    // BT.601 full-range (ITU-T T.871) when true; BT.709 limited (16-235) when false.
    bool full_range = false;
    int gpu_id = 0;
};

class H264Decoder
{
public:
    explicit H264Decoder(const DecoderConfig& cfg);
    ~H264Decoder();

    H264Decoder(const H264Decoder&) = delete;
    H264Decoder& operator=(const H264Decoder&) = delete;
    H264Decoder(H264Decoder&&) = delete;
    H264Decoder& operator=(H264Decoder&&) = delete;

    // Feed one Annex-B access unit. Returns true iff a frame was
    // produced and converted into the caller's RGBA8 buffer.
    // ``rgba_out_device_ptr`` points to the top-left of a HxWx4 GPU
    // buffer; ``row_pitch_bytes`` is the byte stride between rows.
    // Synchronous on an internal stream — buffer is safe to read from
    // any stream once decode() returns.
    bool decode(const uint8_t* packet, std::size_t packet_size, uintptr_t rgba_out_device_ptr, std::size_t row_pitch_bytes);

    // Tear down NVDEC state. Use on stream-timeout / disconnect so the
    // next packet sees a fresh DPB.
    void reset();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace camera_viz::codec
