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

/// One captured sensor and the stream that carries it to the PC.
struct StreamConfig
{
    /// Send every encoded unit of this stream to the PC video stage at host:port (AVS1 over TCP); empty means no
    /// destination yet (one can be set over the management channel). Each stream needs its own port on the PC.
    std::string direct_host;
    uint16_t direct_port = 0;
    /// SIPL pipeline index, validated against the platform config at startup.
    uint32_t sensor_id = 0;
    /// The stream's name: what the PC, the management channel and the logs call this sensor.
    std::string name;
};

/// The clock and event a frame's application timestamp is read from.
enum class TimestampBasis
{
    /// CLOCK_MONOTONIC when the frame was ready to encode (`monotonic_us`); recordable.
    FrameReadyMonotonic,
    /// The sensor's capture event converted to CLOCK_MONOTONIC (`capture_monotonic_us`); unverified on hardware.
    SensorCaptureMonotonic,
    /// CLOCK_REALTIME when the frame finished conversion (`processing_unix_us`); not recordable.
    ProcessingUnix,
};

struct EncodeConfig
{
    /// Rate cap of the encoded stream, independent of the capture rate.
    uint32_t fps = 30;
    /// Output-only uniform reduction of every stream; 1 sends the sensor's native size.
    uint32_t scale_div = 1;
    /// Target bitrate of each stream's hardware encoder; the VBR ceiling is half as much again.
    uint32_t bitrate_bps = 8'000'000;
    /// What each frame's application timestamp is; the receiving side's streams file must declare the same basis.
    TimestampBasis timestamp_basis = TimestampBasis::FrameReadyMonotonic;
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

    /// Argus only: the sensor mode index applied to every camera.
    uint32_t argus_sensor_mode = 0;

    EncodeConfig encode;
};

} // namespace sensing
} // namespace plugins
