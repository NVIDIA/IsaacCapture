// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// SIPL capture for the SENSING SG8A carrier: every sensor in one platform
// config, ISP0 YUV out, converted to RGBA8 on the GPU.
//
// One instance owns every sensor. This is not a style choice --
// INvSIPLCamera::GetInstance() hands out a single process-wide object, so a
// per-sensor camera class is not expressible. Do not reintroduce one.
//
// The SIPL and NvSci headers are pimpl'd away because they drag in NvMedia and
// X11-style macros that collide with mcap's StatusCode; keeping them out of
// this header is what lets frame_sink.cpp include both.

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace plugins
{
namespace sensing
{

struct SiplConfig
{
    /// Vendor platform config, e.g. <pkg>/query/sg8a_agth_g2a/shw5g.json.
    std::string platform_config_json;
    /// Named config inside it, e.g. "SHW5G_2".
    std::string platform_config_name;
    /// One mask per deserializer, in transport order. Required for GMSL.
    std::vector<uint32_t> link_masks;
    /// Directory holding <module>.nito; the ISP has no tuning without it.
    std::string nito_dir = "/var/nvidia/nvcam/settings/sipl";
    int gpu_id = 0;
    /// Overrides the range implied by the reconciled colour standard.
    bool full_range = false;
    /// Swap the chroma planes. Defaults off: the ISP0 surface order is
    /// requested explicitly, so it does not need discovering from the frame.
    bool swap_uv = false;
    /// ISP0 buffers per sensor. Six is the SIPL default; more absorbs a slow
    /// consumer at the cost of 6 MB each at 2560x1984 NV12.
    uint32_t isp0_buffers = 6;
    uint32_t frame_timeout_ms = 1000;
};

/// One sensor as the platform config describes it. `id` is the SIPL pipeline
/// index, which is NOT the GMSL link index, the CSI virtual channel, or the
/// JSON sensorInfo.id -- for SHW5G_2 those are 2 and 3 while this is 0 and 1.
struct SensorInfo
{
    uint32_t id = 0;
    std::string name;
    uint32_t width = 0;
    uint32_t height = 0;
    double fps = 0.0;
};

/**
 * @brief Borrowed view of the most recently converted frame.
 *
 * `ptr` addresses one of three producer-owned slots and stays valid until the
 * second subsequent latest() call for the same sensor, at which point the
 * producer may overwrite it. Copy out before then; do not cache a FrameView
 * across update() ticks.
 */
struct FrameView
{
    uintptr_t ptr = 0;
    size_t pitch = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    /// CLOCK_MONOTONIC, stamped at YUV->RGBA conversion. Round-trips through
    /// the encoder, so it is what identifies a unit on the way back out.
    uint64_t timestamp_ns = 0;
    /// Capture time on the TSC timebase, shared by every sensor on the rig.
    /// This is what makes the two eyes pairable; a host clock is not.
    uint64_t capture_tsc_ns = 0;
    uint64_t sequence = 0;
};

class SiplCamera
{
public:
    explicit SiplCamera(const SiplConfig& config);
    ~SiplCamera();

    SiplCamera(const SiplCamera&) = delete;
    SiplCamera& operator=(const SiplCamera&) = delete;

    /// Enumerate a platform config without touching the hardware. Cheap: the
    /// query API only parses the driver database and the JSON.
    static std::vector<SensorInfo> query(const std::string& platform_config_json,
                                         const std::string& platform_config_name,
                                         const std::vector<uint32_t>& link_masks);

    /// Sensors this instance configured, in pipeline-index order.
    const std::vector<SensorInfo>& sensors() const;

    void start();
    void stop();

    /// Non-blocking mailbox read. nullopt when no new frame has arrived.
    std::optional<FrameView> latest(uint32_t sensor_id);

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace sensing
} // namespace plugins
