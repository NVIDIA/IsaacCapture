// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "camera_capture.hpp"

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
    /// What submit()/submit_dmabuf() will be given; fixed for the encoder's lifetime.
    PixelKind input = PixelKind::CudaRgba;
};

/// One encoded Annex-B unit.
struct EncodedUnit
{
    std::vector<uint8_t> data;
    /// The encoder's own frame-rate numbering, not the stamp given to submit().
    int64_t timestamp_ns = 0;
    /// The unit is an IDR (with its parameter sets), decodable on its own. Not set on the
    /// concatenated tail from end_of_stream().
    bool keyframe = false;

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
 * One thread must drive an encoder: submit(), poll(), request_keyframe() and set_bitrate() are not
 * safe to call concurrently.
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
     * @param timestamp_ns Stamp set on the V4L2 output buffer. On the verified Orin driver the unit
     *        poll() returns does not carry it back (units are numbered from the encoder's own frame rate), so
     *        callers pair units with submissions by order and must not match on this value.
     * @return false when no input buffer is free (frame dropped).
     */
    bool submit(uintptr_t rgba_device_ptr, std::size_t row_pitch_bytes, int64_t timestamp_ns);

    /**
     * @brief Queue one NV12 DMA-BUF frame for encoding, with no copy. Only for `input == DmaBufNv12`.
     * @param dmabuf_fd The NvBufSurface fd the encoder reads.
     * @param lease Keeps the capture buffer alive: held until the encoder's input plane returns this buffer, and
     *        released at once if the frame is dropped.
     * @return false when no input buffer is free (frame dropped).
     */
    bool submit_dmabuf(int dmabuf_fd, int64_t timestamp_ns, FrameLease lease);

    /** @brief Take any completed unit; empty during encoder warmup. */
    EncodedUnit poll();

    /** @brief Signal EOS and drain the remaining bitstream, concatenated into one unit. */
    EncodedUnit end_of_stream();

    /** @brief Signal EOS and drain the remaining bitstream, one unit per frame. */
    std::vector<EncodedUnit> drain();

    /** @brief Make the next submitted frame an IDR, preceded by SPS/PPS. */
    void request_keyframe();

    /**
     * @brief Change the target bitrate of the running stream.
     * @return false when this backend cannot change it while encoding.
     */
    bool set_bitrate(uint32_t bitrate_bps);

    /** @brief Names the encoder behind this class, for diagnostics. */
    static const char* backend_name();

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;

    /// Output buffers queued so far; below the buffer count they are free by
    /// construction and need no dequeue first.
    uint32_t m_queued = 0;
};

} // namespace sensing
} // namespace plugins
