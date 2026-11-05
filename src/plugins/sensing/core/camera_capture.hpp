// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// The capture boundary of the sender: what a camera backend (SIPL, Argus) hands to the encoder. No vendor SDK
// type appears here, so the sender can select a backend at startup without including either SDK.

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace plugins
{
namespace sensing
{

enum class CameraBackend
{
    Auto,
    Argus,
    Sipl,
};

/// One camera as a backend reports it. `id` is the backend's own index (a SIPL pipeline index, an Argus device
/// index); the two are unrelated, so configuration names cameras by logical stream, not by id.
struct CameraInfo
{
    uint32_t id = 0;
    std::string name;
    uint32_t width = 0;
    uint32_t height = 0;
    double fps = 0.0;
};

/// RGBA8 in GPU memory, as the SIPL backend converts it. Borrowed: the backend recycles the buffer, so a consumer
/// copies it out before its next acquire_latest() for the same camera, and the frame's lease is empty.
struct CudaRgbaPixels
{
    uintptr_t device_ptr = 0;
    size_t pitch_bytes = 0;
};

/// NV12 in a DMA-BUF, as an Argus buffer stream produces it. Leased: valid until the frame's lease is released.
/// Contract: the producer has finished writing when the frame is returned, and the surface is pitch-linear NV12 with
/// the pitches and offsets below. A backend that cannot deliver that (block-linear, or still being written) must
/// convert or wait before returning, not describe it here for the encoder to guess.
struct DmaBufNv12Pixels
{
    int fd = -1;
    uint32_t pitch_bytes[2] = { 0, 0 };
    uint32_t offset_bytes[2] = { 0, 0 };
};

/// Returns a frame's buffer to its backend when destroyed or reset. Move-only: exactly one holder owns the buffer,
/// and the backend must outlive every lease it issued. The release callback must not throw (it runs from a
/// destructor), and it runs on whichever thread drops the lease, so it must be thread-safe.
class FrameLease
{
public:
    FrameLease() = default;
    explicit FrameLease(std::function<void()> release) : m_release(std::move(release))
    {
    }
    FrameLease(FrameLease&& other) noexcept : m_release(std::exchange(other.m_release, nullptr))
    {
    }
    FrameLease& operator=(FrameLease&& other) noexcept
    {
        if (this != &other)
        {
            reset();
            m_release = std::exchange(other.m_release, nullptr);
        }
        return *this;
    }
    FrameLease(const FrameLease&) = delete;
    FrameLease& operator=(const FrameLease&) = delete;
    ~FrameLease()
    {
        reset();
    }

    void reset()
    {
        if (m_release)
        {
            std::exchange(m_release, nullptr)();
        }
    }

private:
    std::function<void()> m_release;
};

/// One captured frame. Pixels that are leased stay valid until `lease` is destroyed or reset, so a consumer that
/// needs them longer (an encoder holding an input buffer) keeps the lease; borrowed pixels (see CudaRgbaPixels)
/// are valid only until the next acquire_latest() for the same camera.
struct CaptureFrame
{
    uint32_t camera_id = 0;
    uint64_t sequence = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    /// CLOCK_MONOTONIC and CLOCK_REALTIME read together when the backend had the frame ready: processing time,
    /// not the sensor's capture time.
    uint64_t monotonic_ns = 0;
    uint64_t unix_ns = 0;
    /// The backend's own sensor timestamp, unscaled and in its own clock domain; never convert it to another
    /// domain here. Zero when the backend reports none.
    uint64_t sensor_time = 0;
    std::variant<CudaRgbaPixels, DmaBufNv12Pixels> pixels;
    FrameLease lease;
};

/// A backend instance owns its whole rig: SIPL hands out one process-wide object, so there is one instance per
/// backend, never one per camera.
class CameraCapture
{
public:
    virtual ~CameraCapture() = default;

    virtual const std::vector<CameraInfo>& cameras() const = 0;
    virtual void start() = 0;
    /// Stops capture. Leased frames already handed out stay valid until their leases are released; borrowed frames
    /// (see CudaRgbaPixels) are invalid once capture stops.
    virtual void stop() = 0;
    /// The newest frame of `camera_id` not yet taken, or nullopt when none has arrived. Never blocks.
    virtual std::optional<CaptureFrame> acquire_latest(uint32_t camera_id) = 0;
};

} // namespace sensing
} // namespace plugins
