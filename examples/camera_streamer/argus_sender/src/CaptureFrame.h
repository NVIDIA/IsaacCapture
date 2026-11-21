/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ARGUS_SENDER_CAPTURE_FRAME_H
#define ARGUS_SENDER_CAPTURE_FRAME_H

// The captured-frame boundary between a camera backend and the encoder. No vendor SDK type appears here, so code
// above this seam builds without any camera SDK.

#include <cstdint>
#include <functional>
#include <string>
#include <utility>

namespace argus_sender
{

// Which clock a frame's sensorTime is in, and in what units. The value is never converted between domains here.
enum SensorClockDomain
{
    // The backend reports no sensor time; sensorTime is 0.
    SENSOR_CLOCK_NONE,
    // The platform's TSC, in nanoseconds.
    SENSOR_CLOCK_TSC_NS,
    // A backend counter in its own ticks. Not nanoseconds: map it before comparing with another clock.
    SENSOR_CLOCK_RAW_COUNTER
};

const char* sensorClockDomainName(SensorClockDomain domain);

// One camera as a backend reports it. `id` is the backend's own index; indices of different backends are
// unrelated, so configuration names cameras by logical stream, not by id.
struct CameraDescriptor
{
    uint32_t id = 0;
    std::string name;
    uint32_t width = 0;
    uint32_t height = 0;
    double fps = 0.0;
};

// Returns a frame's buffer to its backend when destroyed or reset. Move-only; exactly one holder owns the buffer and
// the backend must outlive every lease it issued. The callback runs on whichever thread drops the lease, must be
// thread-safe, and must not throw.
class FrameLease
{
public:
    FrameLease()
    {
    }
    explicit FrameLease(std::function<void()> release) : m_release(std::move(release))
    {
    }
    FrameLease(FrameLease&& other) noexcept : m_release(std::move(other.m_release))
    {
        other.m_release = nullptr;
    }
    FrameLease& operator=(FrameLease&& other) noexcept
    {
        if (this != &other)
        {
            reset();
            m_release = std::move(other.m_release);
            other.m_release = nullptr;
        }
        return *this;
    }
    ~FrameLease()
    {
        reset();
    }

    bool held() const
    {
        return static_cast<bool>(m_release);
    }

    void reset()
    {
        if (!m_release)
            return;
        std::function<void()> release;
        release.swap(m_release);
        release();
    }

private:
    FrameLease(const FrameLease&);
    FrameLease& operator=(const FrameLease&);

    std::function<void()> m_release;
};

enum SurfaceColorRange
{
    COLOR_RANGE_UNKNOWN,
    COLOR_RANGE_LIMITED,
    COLOR_RANGE_FULL
};

// Native NV12 in a DMA-BUF. Contract: the producer has finished writing when the frame is returned. A backend that
// cannot deliver that must wait or copy before returning, not describe a still-written surface here.
struct Nv12Surface
{
    int fd = -1;
    uint32_t pitchBytes[2] = { 0, 0 };
    uint32_t offsetBytes[2] = { 0, 0 };
    // True for NvBuffer block-linear layout, false for pitch-linear.
    bool blockLinear = false;
    // What the producer declares; UNKNOWN means it declares nothing and a consumer must not assume.
    SurfaceColorRange colorRange = COLOR_RANGE_UNKNOWN;
};

// One captured frame. Pixels stay valid until `lease` is destroyed or reset, so an encoder that holds the buffer keeps
// the lease and drops it at its actual completion boundary.
struct CaptureFrame
{
    uint32_t cameraId = 0;
    // Identifies one capture run; it is the AVS1 bundle session id, so a frame is named by (sessionEpoch, cameraId,
    // sequence) and a clock mapping made in an earlier run is never applied to a later one.
    uint64_t sessionEpoch = 0;
    // Per-camera, strictly increasing while the camera runs; a gap means frames were missed.
    uint64_t sequence = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    // The sensor's capture time, unscaled, in `sensorClock`. Only SENSOR_CLOCK_TSC_NS may go into an AVS1 *_ns TSC
    // field (with its valid flag); a raw counter must not until its units are declared. SENSOR_CLOCK_NONE (and 0) when
    // the backend reports none; never invented.
    uint64_t sensorTime = 0;
    SensorClockDomain sensorClock = SENSOR_CLOCK_NONE;
    // CLOCK_MONOTONIC when the backend had the frame ready: processing time, not capture time.
    uint64_t readyMonotonicNs = 0;
    Nv12Surface surface;
    FrameLease lease;
};

} // namespace argus_sender

#endif // ARGUS_SENDER_CAPTURE_FRAME_H
