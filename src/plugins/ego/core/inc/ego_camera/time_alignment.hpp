// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <schema/orbbec_ego_audio_generated.h>

#include <cstdint>
#include <memory>
#include <string>

namespace plugins::ego
{

// Syntax/trust checks require no SDK context and precede output creation.
void validate_time_alignment_profile(const std::string& path);

// Hash the SDK library that supplies the version symbol, not a configured path.
std::string loaded_sdk_binary_sha256();

int64_t sdk_timestamp_ns(uint64_t timestamp_us);

struct TimingIdentity
{
    std::string serial;
    std::string firmware;
    std::string sdk_version;
    std::string video_format;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t fps = 0;
    uint32_t imu_rate_hz = 0;
    uint32_t audio_rate_hz = 0;
    uint16_t audio_channels = 0;
    uint16_t audio_bits = 0;
    bool right_enabled = true;
    std::string sdk_binary_sha256;
};

// The SDK audio path has no physical sample-loss counter. Its retained PCM
// index is usable for slicing, but never certifies hardware continuity.
class TimeAlignment
{
public:
    explicit TimeAlignment(const std::string& capture_profile_path = {});
    ~TimeAlignment();
    TimeAlignment(const TimeAlignment&) = delete;
    TimeAlignment& operator=(const TimeAlignment&) = delete;
    void configure(const TimingIdentity& identity);
    void begin_epoch(uint32_t epoch);
    bool observe_left(int64_t raw_ns, int64_t arrival_ns);
    int64_t sample_time(const std::string& stream, int64_t raw_ns, int64_t arrival_ns);
    core::OrbbecEgoAudioTimingT audio_timing(int64_t raw_ns, uint32_t sample_count, uint32_t sample_rate);
    bool has_update() const;
    std::string snapshot() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace plugins::ego
