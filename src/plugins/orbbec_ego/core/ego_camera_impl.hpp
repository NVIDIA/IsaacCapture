// SPDX-FileCopyrightText: Copyright (c) 2026 ORBBEC INC. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device_diagnostics.hpp"
#include "frame_liveness.hpp"
#include "inc/ego_camera/ego_camera.hpp"

#include <libobsensor/ObSensor.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace plugins::ego
{

class Preview;

namespace detail
{
// SDK 2.9 builds place EGO callbacks on Device or DataAcquisitionDevice.
// Keep handle adoption and destruction matched to the selected SDK headers.
#if __has_include(<libobsensor/hpp/DataAcquisitionDevice.hpp>)
using EgoSdkDevice = ob::DataAcquisitionDevice;
#else
using EgoSdkDevice = ob::Device;
#endif

std::shared_ptr<EgoSdkDevice> adopt_ego_device(std::shared_ptr<ob::Device> device);

class RecoverableCaptureError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

class FatalReconnectError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

using ActiveProfiles = std::map<core::OrbbecEgoCameraStream, std::shared_ptr<ob::VideoStreamProfile>>;

OBFrameType to_ob_frame(core::OrbbecEgoCameraStream stream);
OBSensorType to_ob_sensor(core::OrbbecEgoCameraStream stream);
bool has_sensor(const std::shared_ptr<ob::Device>& device, OBSensorType sensor);
std::shared_ptr<ob::VideoStreamProfile> select_profile(const std::shared_ptr<ob::Device>& device,
                                                       const StreamConfig& stream,
                                                       const CaptureConfig& config);
std::shared_ptr<ob::Config> make_video_config(const ActiveProfiles& profiles, const std::vector<StreamConfig>& streams);
std::string describe_profiles(const ActiveProfiles& profiles);
} // namespace detail

class EgoCamera::Impl
{
public:
    Impl(const CaptureConfig& config, const std::vector<StreamConfig>& streams, std::unique_ptr<FrameSink> sink);
    ~Impl();
    void close();
    void update();
    void print_stats() const;
    const std::map<core::OrbbecEgoCameraStream, StreamStats>& stats() const;
    const AuxiliaryStats& auxiliary_stats() const;
    bool preview_closed() const;

private:
    struct ImuEvent
    {
        core::OrbbecEgoImuBatchT batch;
        int64_t local_ns = 0;
        int64_t device_ns = 0;
    };
    struct CalibrationEvent
    {
        core::OrbbecEgoCalibrationT calibration;
        int64_t local_ns = 0;
    };
    struct DeviceStateEvent
    {
        core::OrbbecEgoDeviceStateT state;
        int64_t local_ns = 0;
    };
    using PublishEvent = std::variant<ImuEvent, CalibrationEvent, DeviceStateEvent>;

    struct PendingImu
    {
        std::vector<core::OrbbecEgoImuSample> samples;
        uint64_t sequence = 0;
        int64_t first_local_ns = 0;
        float latest_temperature_c = 0.0f;
    };

    static constexpr size_t kMaxQueuedEvents = 4096;

    // Capture lifecycle and shutdown.
    static bool is_removed_device_cleanup_error(const ob::Error& error, bool device_removal_confirmed) noexcept;
    static std::string stop_pipeline_noexcept(std::unique_ptr<ob::Pipeline>& pipeline,
                                              bool& started,
                                              const char* name,
                                              bool device_removal_confirmed = false) noexcept;
    void check_cancelled() const;
    void update_capture();
    void shutdown(bool promote_metadata);
    void shutdown_noexcept(bool promote_metadata) noexcept;

    // Video capture.
    struct VideoStartupReadiness;
    std::shared_ptr<VideoStartupReadiness> prepare_video_startup_readiness() const;
    void start_initial_video(const std::shared_ptr<ob::Config>& pipeline_config,
                             const std::shared_ptr<VideoStartupReadiness>& readiness);
    void wait_for_video_readiness(const std::shared_ptr<VideoStartupReadiness>& readiness);
    void start_recovered_video(const std::shared_ptr<ob::Config>& pipeline_config);
    void wait_for_recovered_video_readiness();
    void validate_video_liveness() const;
    void receive_video_frame_set(std::shared_ptr<ob::FrameSet> frame_set);
    void drain_video_frames(bool wait_for_frame = true);
    void report_dropped_video_frame_sets(uint64_t dropped);
    void process_frame_set(const std::shared_ptr<ob::FrameSet>& frame_set, int64_t arrival_time_local_common_clock_ns);

    // Device identity and recovery.
    static bool device_list_contains_uid(const std::shared_ptr<ob::DeviceList>& devices, const std::string& uid);
    bool is_selected_physical_device(const std::shared_ptr<ob::DeviceInfo>& info) const;
    std::shared_ptr<ob::Device> find_selected_device();
    void reacquire_device();
    void reset_stream_readiness();
    void stop_capture_for_reconnect(bool device_removal_confirmed = false);
    void begin_recovery(const std::string& reason);
    void wait_for_recovery_stability() const;
    void attempt_reconnect();

    // Device controls.
    void prepare_controls(const std::vector<PropertySetting>& settings);
    void reapply_controls(std::string_view operation = "reconnect");
    void verify_active_controls();
    void restore_properties();

    // Auxiliary event publication.
    template <typename Event>
    void enqueue(Event&& event)
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (events_.size() >= kMaxQueuedEvents)
        {
            ++auxiliary_stats_.dropped_events;
            if (async_error_.empty())
                async_error_ = "Ego metadata queue is full; capture stopped to avoid silent loss";
            return;
        }
        events_.emplace_back(std::forward<Event>(event));
        auxiliary_stats_.publish_queue_peak = std::max<uint64_t>(auxiliary_stats_.publish_queue_peak, events_.size());
    }

    void set_async_error(const std::string& message);
    void drain_events();
    void publish_capture_status(CaptureAvailability availability, const std::string& error = {}) noexcept;
    DiagnosticOrigin diagnostic_origin(int64_t observed_at_ns) const;
    void publish_recovery_event(core::OrbbecEgoRecoveryStage stage, const std::string& reason);
    void capture_device_snapshot();
    void publish_calibration(const std::shared_ptr<ob::Config>& pipeline_config);
    void start_device_state();
    void publish_periodic_device_state(const std::string& final_error = {}, bool direct = false);

    // IMU capture.
    void add_imu_sample(core::OrbbecEgoImuSensor sensor, const OBFloat3D& value, float temperature, uint64_t timestamp_us);
    void flush_imu(core::OrbbecEgoImuSensor sensor);
    void start_imu();
    void wait_for_auxiliary_readiness();
    void wait_for_active_auxiliary_readiness();
    void validate_auxiliary_liveness() const;

    CaptureConfig config_;
    std::unique_ptr<ob::Context> context_;
    std::shared_ptr<detail::EgoSdkDevice> device_;
    std::unique_ptr<ob::Pipeline> pipeline_;
    std::unique_ptr<ob::Pipeline> imu_pipeline_;
    std::vector<StreamConfig> streams_;
    detail::ActiveProfiles active_profiles_;
    std::unique_ptr<FrameSink> sink_;
#if defined(ORBBEC_EGO_ENABLE_PREVIEW)
    std::unique_ptr<Preview> preview_;
#endif
    std::map<core::OrbbecEgoCameraStream, StreamStats> stats_;
    VideoLiveness video_liveness_;
    std::vector<PropertySetting> original_properties_;
    std::vector<PropertySetting> requested_controls_;
    AuxiliaryStats auxiliary_stats_;
    uint64_t reported_dropped_video_frame_sets_ = 0;
    int64_t last_drop_warning_ns_ = 0;
    PendingImu accel_pending_;
    PendingImu gyro_pending_;
    mutable std::mutex queue_mutex_;
    mutable std::mutex video_queue_mutex_;
    std::condition_variable video_queue_cv_;
    std::deque<std::pair<std::shared_ptr<ob::FrameSet>, int64_t>> video_frame_sets_;
    std::deque<PublishEvent> events_;
    std::string async_error_;
    std::string selected_device_uid_;
    std::string selected_device_serial_;
    uint16_t selected_device_vid_ = 0;
    uint16_t selected_device_pid_ = 0;
    std::string selected_firmware_;
    std::string initial_profile_description_;
    std::string recovery_reason_;
    std::string recovery_last_error_;
    std::chrono::steady_clock::time_point last_device_poll_{};
    std::chrono::steady_clock::time_point recovery_deadline_{};
    std::chrono::steady_clock::time_point next_reconnect_attempt_{};
    int64_t capture_epoch_started_ns_ = 0;
    uint32_t capture_epoch_ = 0;
    OBCallbackId device_changed_callback_id_ = 0;
    std::atomic<bool> accepting_callbacks_{ true };
    std::atomic<bool> capture_active_{ false };
    std::atomic<bool> device_removed_{ false };
    std::atomic<bool> accel_ready_{ false };
    std::atomic<bool> gyro_ready_{ false };
    std::atomic<bool> active_accel_ready_{ false };
    std::atomic<bool> active_gyro_ready_{ false };
    std::atomic<int64_t> last_accel_arrival_ns_{ 0 };
    std::atomic<int64_t> last_gyro_arrival_ns_{ 0 };
    std::mutex auxiliary_readiness_mutex_;
    std::condition_variable auxiliary_readiness_wake_;
    bool video_pipeline_started_ = false;
    bool imu_pipeline_started_ = false;
    bool controls_applied_ = false;
    bool device_changed_callback_registered_ = false;
    bool recovering_ = false;
    bool recovery_device_present_ = false;
    bool shutdown_complete_ = false;
    std::string cleanup_error_;
};

} // namespace plugins::ego
