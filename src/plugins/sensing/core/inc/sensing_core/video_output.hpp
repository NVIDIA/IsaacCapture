// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "camera_capture.hpp"
#include "sensing_types.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace plugins
{
namespace sensing
{

struct VideoOutputConfig
{
    uint32_t source_width = 0;
    uint32_t source_height = 0;
    /// Uniform output-only reduction; 1 publishes the sensor's native size.
    uint32_t scale_div = 1;
    /// The frames submit() will be given; DMA-BUF input needs scale_div 1 (no CPU or CUDA copy exists to reduce).
    PixelKind input = PixelKind::CudaRgba;
    /// Publication rate cap, independent of the capture rate.
    uint32_t fps = 30;
    int gpu_id = 0;
    /// Hardware encoder settings: the target bitrate and the VBR ceiling.
    uint32_t bitrate_bps = 8'000'000;
    uint32_t peak_bitrate_bps = 0;
    /// IDR period in frames; 0 defers to the encoder's default.
    uint32_t gop = 0;
    bool full_range = false;
    /// Also send every encoded unit to the PC video stage at this address (host:port); empty disables it.
    std::string direct_host;
    uint16_t direct_port = 0;
    /// What each frame's application timestamp is. SensorCaptureMonotonic fails the output on a frame whose sensor
    /// time cannot be placed on CLOCK_MONOTONIC.
    TimestampBasis timestamp_basis = TimestampBasis::FrameReadyMonotonic;
    FrameCorrelation correlation = FrameCorrelation::Stamp;
};

/// Where this stream's encoded video also goes (the PC video stage), and whether that link is up.
struct DirectLinkStatus
{
    std::string host;
    uint16_t port = 0;
    bool connected = false;
    std::string error;
};

struct VideoOutputStats
{
    uint64_t offered = 0; ///< Captures handed to submit().
    uint64_t paced_out = 0; ///< Skipped to hold the publication rate.
    uint64_t replaced = 0; ///< Snapshots overwritten by a newer capture before the worker took them.
    uint64_t snapshots = 0; ///< GPU snapshots taken.
    uint64_t published = 0; ///< Encoded units handed to the direct link (not proof the PC received them).
    uint64_t encoder_busy = 0; ///< Snapshots the encoder had no input buffer for, dropped before encoding.
    uint64_t keyframes_requested = 0; ///< Keyframes the PC asked for.
    uint64_t matched = 0; ///< Units matched to their capture by the stamp the encoder returned.
    /// Units whose returned stamp matched no submitted frame (sent without a capture time), and units with no frame
    /// in flight (skipped).
    uint64_t unmatched = 0;
    uint64_t encoder_dropped = 0; ///< Submitted frames the encoder never returned a unit for.
};

/**
 * @brief Encodes one sensor's frames on hardware and hands them to the direct link to the PC.
 *
 * submit() runs on the caller's polling loop. A pixel-buffer frame (SIPL) is copied into a snapshot in GPU memory
 * before returning, because the camera may overwrite its slot at the second following poll. A DMA-BUF frame (Argus)
 * is not copied: it goes to the encoder as it is, and its lease returns the buffer to the camera when the encoder is
 * done with it. Everything else -- the optional reduction, the hardware encode and the send queue -- happens on this
 * output's own worker thread, so a slow network cannot hold up capture or the other sensors.
 *
 * Two snapshot slots bound the memory: one the worker is encoding, one waiting. A newer capture replaces the waiting
 * one; the worker never waits for it. The encoder is the JetsonEncoder class: the Jetson V4L2 hardware engine. There
 * is no software encoder: if the hardware encoder cannot start, construction fails.
 *
 * The PC's keyframe requests (and a dropped or reconnected link) are applied before the next frame is encoded.
 */
class VideoOutput
{
public:
    /// @throws std::runtime_error if the geometry is unusable (see validate_geometry), the hardware
    ///         encoder cannot start, or CUDA fails.
    explicit VideoOutput(const VideoOutputConfig& config);

    /// Stops the worker; a snapshot not yet published is dropped.
    ~VideoOutput();

    VideoOutput(const VideoOutput&) = delete;
    VideoOutput& operator=(const VideoOutput&) = delete;

    /// Never blocks on the network. A frame sooner than the publication rate allows is skipped. Borrowed CUDA
    /// RGBA pixels are copied out before this returns; a leased DMA-BUF frame's lease is moved into the output and
    /// released once the encoder is finished with it (or at once if the frame is dropped).
    void submit(CaptureFrame& frame);

    VideoOutputStats stats() const;

    DirectLinkStatus direct_status() const;

    bool failed() const;
    std::string failure() const;

    /// Names the hardware encoder and the transport's view of the stream, for diagnostics.
    std::string describe();

    /**
     * Encodes what is waiting, drains the encoder and publishes every remaining unit with its own
     * stamp. Call once, at the end of the stream: the encoder cannot take frames afterwards.
     */
    void flush();

    /**
     * Output size after dividing: both source dimensions must divide exactly and leave positive,
     * even results. Nothing is rounded or stretched.
     * @throws std::runtime_error naming the problem.
     */
    static void validate_geometry(uint32_t width, uint32_t height, uint32_t divisor);

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace sensing
} // namespace plugins
