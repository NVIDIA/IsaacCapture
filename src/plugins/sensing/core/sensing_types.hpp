// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Plain configuration and frame types, deliberately free of the SIPL, NvSci and
// V4L2 headers. Those define X11-style macros (Success, Status, None) that
// collide with mcap's enum class StatusCode, so the sink layer must be able to
// describe a frame without pulling them in.

#pragma once

#include <schema/sensing_generated.h>

#include <cstdint>
#include <string>
#include <vector>

namespace plugins
{
namespace sensing
{

/// One captured sensor and where its frames go. Any destination may be empty: an
/// ipc-only or livekit-only stream skips the encoder entirely, which is the point
/// of those paths.
struct StreamConfig
{
    /// SIPL pipeline index, validated against the platform config at startup.
    uint32_t sensor_id = 0;
    /// H.264 elementary stream path; empty disables encoding for this sensor.
    std::string output_path;
    /// Unix socket serving RGBA8 frames as CUDA memory; empty disables it.
    std::string ipc_socket_path;
    /// Name of the LiveKit camera track carrying this sensor; empty disables it.
    std::string livekit_track;
};

/// Where and how the livekit= streams are published.
struct LiveKitConfig
{
    std::string url;
    /// The join token itself, not a path. It names the room and the participant. Never logged.
    std::string token;
    /// Publication rate cap, independent of the capture rate.
    uint32_t fps = 30;
    /// Output-only uniform reduction of every track; 1 publishes the sensor's native size.
    uint32_t scale_div = 1;
};

struct SensingConfig
{
    /// Vendor platform config JSON. Required -- SIPL cannot enumerate without
    /// one, and there is no sensible default path.
    std::string platform_config_json;
    /// Named config inside it.
    std::string platform_config_name = "SHW5G_2";
    /// One mask per deserializer, in transport order.
    std::vector<uint32_t> link_masks{ 0x0000, 0x1100 };
    std::string nito_dir = "/var/nvidia/nvcam/settings/sipl";
    int gpu_id = 0;

    /// 0.13 bits/pixel at 2560x1984@60. The old 20 Mbps default was chosen for
    /// 1080p30 and carries 4.9x the pixels at the same rate.
    uint32_t bitrate_bps = 40'000'000;
    /// VBR ceiling. Ignored under CBR.
    uint32_t peak_bitrate_bps = 60'000'000;
    /// IDR period in frames; 0 defers to the encoder default of fps*5.
    uint32_t gop = 0;

    bool full_range = false;
    /// Defaults off: set_sipl_buf_attributes() requests the ISP0 surface
    /// order explicitly, so it does not need discovering from the frame.
    bool swap_uv = false;
    /// ISP0 buffers per sensor, ~6 MB each at 2560x1984 NV12.
    uint32_t isp0_buffers = 6;

    /// Used when any stream names a livekit= track.
    LiveKitConfig livekit;
};

struct SensingFrame
{
    uint32_t sensor_id = 0;

    /// H.264 Annex-B data for one frame.
    std::vector<uint8_t> h264_data;

    core::FrameMetadataSensingT metadata;

    int64_t sample_time_local_common_clock_ns = 0;
    /// SIPL frameCaptureTSC. Shared across sensors, unlike the host stamp.
    int64_t sample_time_raw_device_clock_ns = 0;
};

} // namespace sensing
} // namespace plugins
