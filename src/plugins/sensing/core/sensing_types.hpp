// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Plain configuration types, deliberately free of the SIPL and NvSci headers, which define X11-style
// macros (Success, Status, None) that collide with other libraries'.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace plugins
{
namespace sensing
{

/// One captured sensor and the LiveKit track that carries it.
struct StreamConfig
{
    /// SIPL pipeline index, validated against the platform config at startup.
    uint32_t sensor_id = 0;
    /// Name of the LiveKit camera track carrying this sensor.
    std::string livekit_track;
};

/// Where and how the tracks are published.
struct LiveKitConfig
{
    std::string url;
    /// The join token itself, not a path. It names the room and the participant. Never logged.
    std::string token;
    /// Publication rate cap, independent of the capture rate.
    uint32_t fps = 30;
    /// Output-only uniform reduction of every track; 1 publishes the sensor's native size.
    uint32_t scale_div = 1;
    /// Target bitrate of each track's hardware encoder; the VBR ceiling is half as much again.
    uint32_t bitrate_bps = 8'000'000;
};

struct SensingConfig
{
    /// Vendor platform config JSON. Required -- SIPL cannot enumerate without one.
    std::string platform_config_json;
    /// Named config inside it.
    std::string platform_config_name = "SHW5G_2";
    /// One mask per deserializer, in transport order.
    std::vector<uint32_t> link_masks{ 0x0000, 0x1100 };
    std::string nito_dir = "/var/nvidia/nvcam/settings/sipl";
    int gpu_id = 0;

    /// IDR period in frames; 0 defers to the encoder default of fps*5.
    uint32_t gop = 0;

    bool full_range = false;
    /// Defaults off: set_sipl_buf_attributes() requests the ISP0 surface order explicitly.
    bool swap_uv = false;
    /// ISP0 buffers per sensor, ~6 MB each at 2560x1984 NV12.
    uint32_t isp0_buffers = 6;

    LiveKitConfig livekit;
};

} // namespace sensing
} // namespace plugins
