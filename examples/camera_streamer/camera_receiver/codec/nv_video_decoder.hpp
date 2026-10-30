// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// NVDEC H.264/HEVC video decoder. It can either copy decoded NV12 into a caller-owned
// tight NV12 buffer or convert into caller-owned RGBA8. Zero-latency
// (no reorder buffer, decode-order output). RGBA conversion supports BT.709
// limited and BT.601 full range.

#pragma once

#include <cstdint>
#include <memory>

namespace camera_receiver::codec
{

enum class Codec
{
    H264 = 0,
    HEVC = 1,
};

struct DecoderConfig
{
    uint32_t width = 0;
    uint32_t height = 0;
    // BT.601 full-range (ITU-T T.871) when true; BT.709 limited (16-235) when false.
    bool full_range = false;
    int gpu_id = 0;
    // Elementary-stream codec. Must match the sender.
    Codec codec = Codec::H264;
};

class NvVideoDecoder
{
public:
    explicit NvVideoDecoder(const DecoderConfig& cfg);
    ~NvVideoDecoder();

    NvVideoDecoder(const NvVideoDecoder&) = delete;
    NvVideoDecoder& operator=(const NvVideoDecoder&) = delete;
    NvVideoDecoder(NvVideoDecoder&&) = delete;
    NvVideoDecoder& operator=(NvVideoDecoder&&) = delete;

    uint32_t width() const;
    uint32_t height() const;

    // Feed one Annex-B access unit. Returns true iff a frame was
    // produced and converted into the caller's RGBA8 buffer.
    // ``rgba_out_device_ptr`` points to the top-left of a HxWx4 GPU
    // buffer; ``row_pitch_bytes`` is the byte stride between rows.
    // Synchronous on an internal stream — buffer is safe to read from
    // any stream once decode() returns.
    bool decode(const uint8_t* packet,
                std::size_t packet_size,
                uintptr_t rgba_out_device_ptr,
                std::size_t row_pitch_bytes,
                int rotation = 0);

    // Feed one Annex-B access unit. Returns true iff a frame was produced and
    // copied into a caller-owned tight-ish NV12 buffer shaped
    // ((height * 3) / 2, width). ``row_pitch_bytes`` is the byte stride between
    // output rows; the Y plane starts at ``nv12_out_device_ptr`` and the UV
    // plane starts at row ``height``.
    bool decode_nv12(const uint8_t* packet,
                     std::size_t packet_size,
                     uintptr_t nv12_out_device_ptr,
                     std::size_t row_pitch_bytes,
                     int rotation = 0);

    // Tear down NVDEC state. Use on stream-timeout / disconnect so the
    // next packet sees a fresh DPB.
    void reset();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace camera_receiver::codec
