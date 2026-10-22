// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <schema/orbbec_ego_audio_generated.h>
#include <schema/orbbec_ego_calibration_generated.h>
#include <schema/orbbec_ego_camera_generated.h>
#include <schema/orbbec_ego_device_state_generated.h>
#include <schema/orbbec_ego_imu_generated.h>

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace plugins::ego
{

struct StreamConfig
{
    core::OrbbecEgoCameraStream camera;
    std::string output_path;
    core::OrbbecEgoPixelFormat pixel_format = core::OrbbecEgoPixelFormat_Mjpg;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t fps = 0;
};

enum class McapMediaMode
{
    Sidecar,
    Embedded,
};

struct PropertySetting
{
    std::string name;
    double value = 0.0;
};

class CaptureCancelled : public std::runtime_error
{
public:
    CaptureCancelled() : std::runtime_error("Ego capture cancelled")
    {
    }
};

struct CaptureConfig
{
    std::function<bool()> stop_requested;
    std::string device_uid;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t fps = 0;
    uint32_t bitrate = 0;
    bool dynamic_bitrate = false;
    bool dynamic_bitrate_set = false;
    bool preview = false;
    bool enable_imu = false;
    uint32_t imu_rate = 400;
    // These are requests, not a PID contract. start_imu validates them against
    // the profiles enumerated from the selected device.
    float accel_full_scale_g = 24.0f;
    float gyro_full_scale_dps = 2000.0f;
    bool enable_audio = false;
    std::string audio_output;
    std::string collection_prefix;
    std::string plugin_root_id;
    bool record_device_diagnostics = true;
    std::string mcap_filename;
    std::string mcap_media_spool;
    McapMediaMode mcap_media_mode = McapMediaMode::Sidecar;
    bool keep_media_sidecars = false;
    std::string calibration_output;
    std::string time_alignment;
    std::vector<PropertySetting> properties;
    bool persist_controls = false;
    uint32_t reconnect_timeout_seconds = 30;
    uint32_t reconnect_interval_milliseconds = 1000;
};

// Validates device-independent stream fields. Profile availability and
// simultaneous-stream compatibility are resolved from the selected device.
void validate_stream_config(const StreamConfig& stream, const CaptureConfig& config);
// Applies release certification policy to the SDK-resolved profile, including
// requests where zero selected the SDK default.
void validate_resolved_stream_config(const StreamConfig& stream, uint32_t resolved_fps);

struct CapturedFrame
{
    core::OrbbecEgoFrameMetadataT metadata;
    std::vector<uint8_t> encoded_data;
    int64_t sample_time_local_common_clock_ns = 0;
    int64_t sample_time_raw_device_clock_ns = 0;
};

// Facts are refreshed by the capture owner; a publishing thread cannot refresh their age.
enum class CaptureAvailability
{
    Unknown,
    Absent,
    Starting,
    Streaming,
    Recovering,
    Failed
};
struct CaptureStatus
{
    CaptureAvailability availability = CaptureAvailability::Unknown;
    int64_t observed_at_ns = 0;
    std::string error;
};

class IMetadataSink
{
public:
    virtual ~IMetadataSink() = default;
    // Payload references are borrowed; copy any data retained after the call returns.
    // Timestamp pairs are (local common-clock ns, raw device-clock ns).
    virtual void on_frame_metadata(const CapturedFrame& frame) = 0;
    virtual void on_imu_batch(const core::OrbbecEgoImuBatchT&, int64_t, int64_t)
    {
    }
    virtual void on_audio_chunk(const core::OrbbecEgoAudioChunkT&, int64_t, int64_t)
    {
    }
    virtual void on_encoded_video_frame(const CapturedFrame&)
    {
    }
    virtual void on_pcm_audio_chunk(const core::OrbbecEgoPcmAudioChunkT&, int64_t, int64_t)
    {
    }
    virtual void on_calibration(const core::OrbbecEgoCalibrationT&, int64_t, int64_t)
    {
    }
    virtual void on_device_state(const core::OrbbecEgoDeviceStateT&, int64_t, int64_t)
    {
    }
    virtual void on_capture_status(const CaptureStatus&)
    {
    }
    virtual void close()
    {
    }
    // Sealing and promotion are separate so producer cleanup can veto delivery.
    virtual void commit()
    {
    }
    virtual void abort()
    {
        close();
    }
    virtual std::string error() const
    {
        return {};
    }
};

std::unique_ptr<IMetadataSink> compose_metadata_sinks(std::vector<std::unique_ptr<IMetadataSink>> sinks);

class FrameSink
{
public:
    FrameSink(const std::vector<StreamConfig>& streams,
              std::unique_ptr<IMetadataSink> metadata_sink = nullptr,
              bool write_media_sidecars = true);
    ~FrameSink();

    FrameSink(const FrameSink&) = delete;
    FrameSink& operator=(const FrameSink&) = delete;

    // The observer runs synchronously after the recording gate, with canonical bytes.
    void on_frame(const CapturedFrame& frame, const std::function<void(const CapturedFrame&)>& accepted_frame = {});
    void begin_capture_epoch();
    IMetadataSink* metadata_sink();
    void close_media();
    void close_metadata();
    void commit_metadata();
    void abort_metadata();
    std::string metadata_error() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

std::unique_ptr<FrameSink> create_frame_sink(const std::vector<StreamConfig>& streams,
                                             const std::string& collection_prefix);
std::unique_ptr<FrameSink> create_frame_sink(const std::vector<StreamConfig>& streams, const CaptureConfig& config);

struct StreamStats
{
    uint64_t frame_count = 0;
    uint64_t byte_count = 0;
    uint64_t sequence_gaps = 0;
    uint64_t last_sequence = 0;
    int64_t last_device_timestamp_ns = 0;
    uint64_t epoch_frame_count = 0;
};

struct AuxiliaryStats
{
    uint64_t accel_samples = 0;
    uint64_t gyro_samples = 0;
    uint64_t audio_samples = 0;
    uint64_t publish_queue_peak = 0;
    uint64_t dropped_events = 0;
    uint64_t dropped_video_frame_sets = 0;
    uint32_t capture_epoch = 0;
    uint32_t reconnect_attempts = 0;
    uint32_t successful_reconnects = 0;
};

class EgoCamera
{
public:
    EgoCamera(const CaptureConfig& config, const std::vector<StreamConfig>& streams, std::unique_ptr<FrameSink> sink);
    ~EgoCamera();

    EgoCamera(const EgoCamera&) = delete;
    EgoCamera& operator=(const EgoCamera&) = delete;

    void update();
    void close();
    void print_stats() const;
    const std::map<core::OrbbecEgoCameraStream, StreamStats>& stats() const;
    const AuxiliaryStats& auxiliary_stats() const;
    bool preview_closed() const;

    static void list_capabilities(const CaptureConfig& config);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace plugins::ego
