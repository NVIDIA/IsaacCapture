// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace plugins
{
namespace sensing
{

struct EncoderConfig
{
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t bitrate_bps = 40'000'000;
    /// VBR ceiling; 0 keeps CBR. The H.264 output is a recording, not a
    /// fixed-rate transport, so CBR would pad static scenes and starve motion.
    uint32_t peak_bitrate_bps = 60'000'000;
    uint32_t fps = 30;
    /// IDR period in frames; 0 defers to fps*5.
    uint32_t gop = 0;
    bool full_range = false;
};

/// One encoded Annex-B unit and the capture stamp of the frame it came from.
struct EncodedUnit
{
    std::vector<uint8_t> data;
    int64_t timestamp_ns = 0;

    bool empty() const
    {
        return data.empty();
    }
};

/**
 * @brief H.264 encoder on the Jetson V4L2 M2M engine (/dev/v4l2-nvenc).
 *
 * Jetson has no libnvidia-encode — the NVIDIA Video Codec SDK is dGPU-only —
 * so this wraps NvVideoEncoder from the Jetson Multimedia API instead.
 *
 * Submission is asynchronous: the encoder needs several input frames before it
 * emits the first bitstream unit, so submit() never blocks waiting for output
 * and poll() drains whatever a capture-plane thread has completed.
 */
class JetsonEncoder
{
public:
    explicit JetsonEncoder(const EncoderConfig& config);
    ~JetsonEncoder();

    JetsonEncoder(const JetsonEncoder&) = delete;
    JetsonEncoder& operator=(const JetsonEncoder&) = delete;
    JetsonEncoder(JetsonEncoder&&) = delete;
    JetsonEncoder& operator=(JetsonEncoder&&) = delete;

    /**
     * @brief Convert one GPU-resident RGBA8 frame to NV12 and queue it.
     * @param rgba_device_ptr Device pointer to a HxWx4 RGBA8 buffer.
     * @param row_pitch_bytes Byte stride between rows.
     * @param timestamp_ns Capture stamp, carried on the V4L2 output buffer and
     *        returned by poll() on the unit this frame produces.
     * @return false when no input buffer is free (frame dropped).
     */
    bool submit(uintptr_t rgba_device_ptr, std::size_t row_pitch_bytes, int64_t timestamp_ns);

    /** @brief Take any completed unit; empty during encoder warmup. */
    EncodedUnit poll();

    /** @brief Signal EOS and drain the remaining bitstream. */
    EncodedUnit end_of_stream();

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;

    /// Output buffers queued so far; below the buffer count they are free by
    /// construction and need no dequeue first.
    uint32_t m_queued = 0;
};

} // namespace sensing
} // namespace plugins
