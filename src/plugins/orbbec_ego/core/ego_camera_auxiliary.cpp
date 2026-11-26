// SPDX-FileCopyrightText: Copyright (c) 2026 ORBBEC INC. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "calibration.hpp"
#include "device_controls.hpp"
#include "diagnostics.hpp"
#include "ego_camera_impl.hpp"
#include "inc/ego_camera/cancellation.hpp"
#include "inc/ego_camera/sdk_timestamp.hpp"

#include <oxr_utils/os_time.hpp>

#include <iostream>
#include <iterator>
#include <optional>
#include <type_traits>
#include <utility>

namespace plugins::ego
{

using detail::describe_profiles;
using detail::RecoverableCaptureError;
using detail::to_ob_sensor;

namespace
{

OBIMUSampleRate imu_rate(uint32_t rate)
{
    if (rate == 400)
        return OB_SAMPLE_RATE_400_HZ;
    if (rate == 1000)
        return OB_SAMPLE_RATE_1_KHZ;
    throw std::invalid_argument("EGO IMU rate must be 400 or 1000 Hz");
}

OBAccelFullScaleRange accel_scale(float scale)
{
    if (scale == 2)
        return OB_ACCEL_FS_2g;
    if (scale == 3)
        return OB_ACCEL_FS_3g;
    if (scale == 4)
        return OB_ACCEL_FS_4g;
    if (scale == 6)
        return OB_ACCEL_FS_6g;
    if (scale == 8)
        return OB_ACCEL_FS_8g;
    if (scale == 12)
        return OB_ACCEL_FS_12g;
    if (scale == 16)
        return OB_ACCEL_FS_16g;
    if (scale == 24)
        return OB_ACCEL_FS_24g;
    throw std::invalid_argument("Unsupported Ego accelerometer full scale");
}

OBGyroFullScaleRange gyro_scale(float scale)
{
    if (scale == 16)
        return OB_GYRO_FS_16dps;
    if (scale == 31)
        return OB_GYRO_FS_31dps;
    if (scale == 62)
        return OB_GYRO_FS_62dps;
    if (scale == 125)
        return OB_GYRO_FS_125dps;
    if (scale == 250)
        return OB_GYRO_FS_250dps;
    if (scale == 400)
        return OB_GYRO_FS_400dps;
    if (scale == 500)
        return OB_GYRO_FS_500dps;
    if (scale == 800)
        return OB_GYRO_FS_800dps;
    if (scale == 1000)
        return OB_GYRO_FS_1000dps;
    if (scale == 2000)
        return OB_GYRO_FS_2000dps;
    throw std::invalid_argument("Unsupported Ego gyroscope full scale");
}

std::shared_ptr<core::OrbbecEgoCameraIntrinsicsT> camera_intrinsics(const OBCalibrationParam& param, OBSensorType sensor)
{
    const auto& source = param.intrinsics[sensor];
    const auto& distortion = param.distortion[sensor];
    auto result = std::make_shared<core::OrbbecEgoCameraIntrinsicsT>();
    result->width = source.width;
    result->height = source.height;
    result->fx = source.fx;
    result->fy = source.fy;
    result->cx = source.cx;
    result->cy = source.cy;
    result->distortion_model = distortion.model;
    result->distortion = { distortion.k1, distortion.k2, distortion.k3, distortion.k4,
                           distortion.k5, distortion.k6, distortion.p1, distortion.p2 };
    return result;
}

std::shared_ptr<core::OrbbecEgoExtrinsicsT> extrinsics(const OBCalibrationParam& param, OBSensorType from, OBSensorType to)
{
    const auto& source = param.extrinsics[from][to];
    auto result = std::make_shared<core::OrbbecEgoExtrinsicsT>();
    result->rotation.assign(std::begin(source.rot), std::end(source.rot));
    result->translation_mm.assign(std::begin(source.trans), std::end(source.trans));
    return result;
}

} // namespace

void EgoCamera::Impl::publish_capture_status(CaptureAvailability availability, const std::string& error) noexcept
{
    try
    {
        if (auto* metadata = sink_->metadata_sink())
            metadata->on_capture_status({ availability, core::os_monotonic_now_ns(), error });
    }
    catch (const std::exception& failure)
    {
        detail::log_warning() << "Ego device status notification failed: " << failure.what();
    }
}

DiagnosticOrigin EgoCamera::Impl::diagnostic_origin(int64_t observed_at_ns) const
{
    return { selected_device_uid_, capture_epoch_, observed_at_ns };
}

void EgoCamera::Impl::publish_recovery_event(core::OrbbecEgoRecoveryStage stage, const std::string& reason)
{
    const auto now = core::os_monotonic_now_ns();
    auto state = recovery_event(diagnostic_origin(now), stage, reason, auxiliary_stats_.reconnect_attempts);
    enqueue(DeviceStateEvent{ std::move(state), now });
    drain_events();
}

void EgoCamera::Impl::capture_device_snapshot()
{
    std::vector<core::OrbbecEgoDevicePropertyValue> properties;
    for (int index = 0; index < device_->getSupportedPropertyCount(); ++index)
    {
        const auto item = device_->getSupportedProperty(static_cast<uint32_t>(index));
        if ((item.permission & OB_PERMISSION_READ) == 0 ||
            (item.type != OB_BOOL_PROPERTY && item.type != OB_INT_PROPERTY && item.type != OB_FLOAT_PROPERTY))
        {
            continue;
        }
        try
        {
            properties.emplace_back(static_cast<int32_t>(item.id), read_property(device_, item));
        }
        catch (const ob::Error& error)
        {
            detail::log_warning() << "Ego property snapshot read failed for " << item.id << ": " << error.what();
        }
    }
    std::optional<float> temperature_c;
    try
    {
        OBDeviceTemperature temperature{};
        uint32_t size = sizeof(temperature);
        device_->getStructuredData(OB_STRUCT_DEVICE_TEMPERATURE, reinterpret_cast<uint8_t*>(&temperature), &size);
        if (size >= sizeof(temperature))
            temperature_c = temperature.imuTemp;
    }
    catch (const ob::Error& error)
    {
        detail::log_warning() << "Ego IMU temperature snapshot unavailable: " << error.what();
    }
    // Publish at observation time; recovery never relabels these startup-only reads.
    const auto now = core::os_monotonic_now_ns();
    auto state = device_snapshot(diagnostic_origin(now), temperature_c, properties);
    enqueue(DeviceStateEvent{ std::move(state), now });
}

void EgoCamera::Impl::validate_auxiliary_liveness() const
{
    if (!capture_active_.load(std::memory_order_acquire))
        return;
    const int64_t now = core::os_monotonic_now_ns();
    const auto validate = [this, now](bool expected, const std::atomic<int64_t>& last_arrival, const char* name)
    {
        if (!expected)
            return;
        const int64_t observed = last_arrival.load(std::memory_order_acquire);
        const int64_t reference = observed == 0 ? capture_epoch_started_ns_ : observed;
        if (now - reference > 5'000'000'000LL)
            throw RecoverableCaptureError(std::string("Ego ") + name + " stopped delivering samples for five seconds");
    };
    validate(config_.enable_imu, last_accel_arrival_ns_, "Accel");
    validate(config_.enable_imu, last_gyro_arrival_ns_, "Gyro");
}

void EgoCamera::Impl::set_async_error(const std::string& message)
{
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (async_error_.empty())
        async_error_ = message;
}

void EgoCamera::Impl::drain_events()
{
    std::deque<PublishEvent> pending;
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        pending.swap(events_);
    }
    auto* metadata = sink_->metadata_sink();
    for (auto& event : pending)
    {
        std::visit(
            [this, metadata](auto& value)
            {
                using Event = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<Event, ImuEvent>)
                {
                    if (value.batch.sensor == core::OrbbecEgoImuSensor_Accel)
                        auxiliary_stats_.accel_samples += value.batch.samples.size();
                    else
                        auxiliary_stats_.gyro_samples += value.batch.samples.size();
                    if (metadata)
                        metadata->on_imu_batch(value.batch, value.local_ns, value.device_ns);
                }
                else if constexpr (std::is_same_v<Event, CalibrationEvent>)
                {
                    if (metadata)
                        metadata->on_calibration(value.calibration, value.local_ns, 0);
                }
                else if constexpr (std::is_same_v<Event, DeviceStateEvent>)
                {
                    if (metadata)
                        metadata->on_device_state(value.state, value.local_ns, 0);
                }
            },
            event);
    }
}

void EgoCamera::Impl::add_imu_sample(core::OrbbecEgoImuSensor sensor,
                                     const OBFloat3D& value,
                                     float temperature,
                                     uint64_t timestamp_us)
{
    auto& pending = sensor == core::OrbbecEgoImuSensor_Accel ? accel_pending_ : gyro_pending_;
    const int64_t arrival_ns = core::os_monotonic_now_ns();
    const int64_t device_ns = sdk_timestamp_ns(timestamp_us);
    if (pending.samples.empty())
        pending.first_local_ns = arrival_ns;
    pending.samples.emplace_back(value.x, value.y, value.z, arrival_ns, device_ns);
    pending.latest_temperature_c = temperature;
    if (pending.samples.size() < 32 && arrival_ns - pending.first_local_ns < 20'000'000)
        return;
    flush_imu(sensor);
}

void EgoCamera::Impl::flush_imu(core::OrbbecEgoImuSensor sensor)
{
    auto& pending = sensor == core::OrbbecEgoImuSensor_Accel ? accel_pending_ : gyro_pending_;
    if (pending.samples.empty())
        return;
    const int64_t local_ns = pending.samples.back().sample_time_local_common_clock_ns();
    const int64_t device_ns = pending.samples.back().sample_time_raw_device_clock_ns();
    core::OrbbecEgoImuBatchT batch;
    batch.sensor = sensor;
    batch.sequence_number = pending.sequence++;
    batch.sample_rate_hz = config_.imu_rate;
    batch.full_scale =
        sensor == core::OrbbecEgoImuSensor_Accel ? config_.accel_full_scale_g : config_.gyro_full_scale_dps;
    batch.capture_epoch = capture_epoch_;
    batch.temperature_c = pending.latest_temperature_c;
    batch.samples.swap(pending.samples);
    enqueue(ImuEvent{ std::move(batch), local_ns, device_ns });
}

void EgoCamera::Impl::start_imu()
{
    check_cancelled();
    auto imu_config = std::make_shared<ob::Config>();
    imu_config->enableAccelStream(accel_scale(config_.accel_full_scale_g), imu_rate(config_.imu_rate));
    imu_config->enableGyroStream(gyro_scale(config_.gyro_full_scale_dps), imu_rate(config_.imu_rate));
    imu_pipeline_ = std::make_unique<ob::Pipeline>(device_);
    imu_pipeline_->start(imu_config,
                         [this](std::shared_ptr<ob::FrameSet> frame_set)
                         {
                             if (!accepting_callbacks_.load(std::memory_order_acquire))
                                 return;
                             try
                             {
                                 if (const auto raw = frame_set ? frame_set->getFrame(OB_FRAME_ACCEL) : nullptr)
                                 {
                                     last_accel_arrival_ns_.store(core::os_monotonic_now_ns(), std::memory_order_release);
                                     if (!accel_ready_.exchange(true, std::memory_order_acq_rel))
                                         auxiliary_readiness_wake_.notify_all();
                                     if (capture_active_.load(std::memory_order_acquire))
                                     {
                                         const auto frame = raw->as<ob::AccelFrame>();
                                         add_imu_sample(core::OrbbecEgoImuSensor_Accel, frame->getValue(),
                                                        frame->getTemperature(), frame->getTimeStampUs());
                                         if (!active_accel_ready_.exchange(true, std::memory_order_acq_rel))
                                             auxiliary_readiness_wake_.notify_all();
                                     }
                                 }
                                 if (const auto raw = frame_set ? frame_set->getFrame(OB_FRAME_GYRO) : nullptr)
                                 {
                                     last_gyro_arrival_ns_.store(core::os_monotonic_now_ns(), std::memory_order_release);
                                     if (!gyro_ready_.exchange(true, std::memory_order_acq_rel))
                                         auxiliary_readiness_wake_.notify_all();
                                     if (capture_active_.load(std::memory_order_acquire))
                                     {
                                         const auto frame = raw->as<ob::GyroFrame>();
                                         add_imu_sample(core::OrbbecEgoImuSensor_Gyro, frame->getValue(),
                                                        frame->getTemperature(), frame->getTimeStampUs());
                                         if (!active_gyro_ready_.exchange(true, std::memory_order_acq_rel))
                                             auxiliary_readiness_wake_.notify_all();
                                     }
                                 }
                             }
                             catch (const std::exception& error)
                             {
                                 set_async_error(std::string("IMU callback failed: ") + error.what());
                             }
                         });
    imu_pipeline_started_ = true;
    detail::log_info() << "Ego IMU started at " << config_.imu_rate << " Hz" << std::endl;
}

void EgoCamera::Impl::wait_for_auxiliary_readiness()
{
    const auto ready = [this]
    {
        return !config_.enable_imu ||
               (accel_ready_.load(std::memory_order_acquire) && gyro_ready_.load(std::memory_order_acquire));
    };
    std::unique_lock<std::mutex> lock(auxiliary_readiness_mutex_);
    if (wait_capture_ready(auxiliary_readiness_wake_, lock, std::chrono::seconds(10), ready, config_.stop_requested))
        return;
    std::string missing;
    if (config_.enable_imu && !accel_ready_.load(std::memory_order_relaxed))
        missing += " Accel";
    if (config_.enable_imu && !gyro_ready_.load(std::memory_order_relaxed))
        missing += " Gyro";
    throw std::runtime_error("Ego uid=" + selected_device_uid_ + " firmware=" + selected_firmware_ +
                             " active profiles [" + describe_profiles(active_profiles_) +
                             "] timed out waiting for auxiliary streams:" + missing);
}

void EgoCamera::Impl::wait_for_active_auxiliary_readiness()
{
    const auto ready = [this]
    {
        return !config_.enable_imu || (active_accel_ready_.load(std::memory_order_acquire) &&
                                       active_gyro_ready_.load(std::memory_order_acquire));
    };
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!ready() && std::chrono::steady_clock::now() < deadline)
    {
        check_cancelled();
        drain_video_frames();
        drain_events();
        if (const auto failure = sink_->metadata_error(); !failure.empty())
            throw std::runtime_error("Ego metadata publication failed: " + failure);
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (!async_error_.empty())
                throw std::runtime_error(async_error_);
        }
        std::unique_lock<std::mutex> lock(auxiliary_readiness_mutex_);
        auxiliary_readiness_wake_.wait_for(lock, std::chrono::milliseconds(20), ready);
    }
    if (ready())
        return;
    std::string missing;
    if (config_.enable_imu && !active_accel_ready_.load(std::memory_order_relaxed))
        missing += " Accel";
    if (config_.enable_imu && !active_gyro_ready_.load(std::memory_order_relaxed))
        missing += " Gyro";
    throw std::runtime_error("Ego uid=" + selected_device_uid_ + " firmware=" + selected_firmware_ +
                             " active profiles [" + describe_profiles(active_profiles_) +
                             "] timed out waiting for auxiliary streams after video became active:" + missing);
}

void EgoCamera::Impl::publish_calibration(const std::shared_ptr<ob::Config>& pipeline_config)
{
    core::OrbbecEgoCalibrationT value;
    value.device_uid = device_->getDeviceInfo()->getUid();
    value.capture_epoch = capture_epoch_ + (recovering_ ? 1U : 0U);
    std::string sdk_calibration_error;
    try
    {
        const auto fill_structured_calibration = [this, &value](const std::shared_ptr<ob::Config>& config)
        {
            const auto param = pipeline_->getCalibrationParam(config);
            value.color_left = camera_intrinsics(param, OB_SENSOR_COLOR_LEFT);
            value.color_right = camera_intrinsics(param, OB_SENSOR_COLOR_RIGHT);
            value.left_to_right = extrinsics(param, OB_SENSOR_COLOR_LEFT, OB_SENSOR_COLOR_RIGHT);
            value.accel_to_left = extrinsics(param, OB_SENSOR_ACCEL, OB_SENSOR_COLOR_LEFT);
            value.gyro_to_left = extrinsics(param, OB_SENSOR_GYRO, OB_SENSOR_COLOR_LEFT);
        };
        try
        {
            fill_structured_calibration(pipeline_config);
        }
        catch (const ob::Error&)
        {
            // Firmware calibration is indexed by MJPEG profiles even when the active stream is H.264/H.265.
            auto calibration_config = std::make_shared<ob::Config>();
            for (const auto& [stream, profile] : active_profiles_)
            {
                const auto fallback =
                    device_->getSensor(to_ob_sensor(stream))
                        ->getStreamProfileList()
                        ->getVideoStreamProfile(profile->getWidth(), profile->getHeight(), OB_FORMAT_MJPG, 0);
                calibration_config->enableStream(fallback);
            }
            fill_structured_calibration(calibration_config);
        }
    }
    catch (const ob::Error& error)
    {
        sdk_calibration_error = error.what();
    }
    try
    {
        value.raw_alignment_yaml = raw_data(device_, OB_RAW_DATA_ALIGN_CALIB_YAML);
    }
    catch (const std::exception& error)
    {
        detail::log_error() << "Raw alignment calibration unavailable: " << error.what() << std::endl;
    }
    try
    {
        value.raw_imu_yaml = raw_data(device_, OB_RAW_DATA_IMU_CALIB_YAML);
    }
    catch (const std::exception& error)
    {
        detail::log_error() << "Raw IMU calibration unavailable: " << error.what() << std::endl;
    }
    if (!value.color_left || !value.color_right || !value.left_to_right)
    {
        core::OrbbecEgoCalibrationT fallback;
        if (populate_stereo_calibration_from_yaml(value.raw_alignment_yaml, fallback))
        {
            if (!value.color_left)
                value.color_left = std::move(fallback.color_left);
            if (!value.color_right)
                value.color_right = std::move(fallback.color_right);
            if (!value.left_to_right)
                value.left_to_right = std::move(fallback.left_to_right);
        }
        else
            detail::log_error() << "Raw alignment calibration does not contain a usable stereo pair." << std::endl;
    }
    if (!value.accel_to_left || !value.gyro_to_left)
    {
        if (populate_imu_calibration_from_yaml(value.raw_imu_yaml, value))
            detail::log_info() << "Using raw IMU calibration for missing IMU-to-left-camera extrinsics." << std::endl;
        else
            detail::log_error()
                << "Raw IMU calibration does not contain a usable IMU-to-left-camera transform." << std::endl;
    }
    if (!sdk_calibration_error.empty())
    {
        if (value.color_left && value.color_right)
            detail::log_info() << "Using raw alignment calibration because SDK profile calibration is unavailable: "
                               << sdk_calibration_error << std::endl;
        else
            detail::log_error() << "Structured calibration unavailable: " << sdk_calibration_error << std::endl;
    }
    const auto now = core::os_monotonic_now_ns();
    enqueue(CalibrationEvent{ std::move(value), now });
}

void EgoCamera::Impl::start_device_state()
{
    check_cancelled();
    device_->setEgoStateCallback(
        [this](const OBEgoStateReport& report)
        {
            if (!accepting_callbacks_.load(std::memory_order_acquire))
                return;
            if (!capture_active_.load(std::memory_order_acquire))
                return;
            const auto now = core::os_monotonic_now_ns();
            auto state = sdk_report(diagnostic_origin(now), report.sequence, report.work_state, report.state_flags,
                                    report.error_flags, report.storage_free_bytes);
            enqueue(DeviceStateEvent{ std::move(state), now });
        });
}

void EgoCamera::Impl::publish_periodic_device_state(const std::string& final_error, bool direct)
{
    if (selected_device_uid_.empty())
        return;
    // Disconnect detection belongs to removal callbacks and existing stream liveness checks.
    core::OrbbecEgoCaptureSummaryT summary;
    uint64_t dropped_video_frame_sets = 0;
    {
        std::lock_guard<std::mutex> lock(video_queue_mutex_);
        dropped_video_frame_sets = auxiliary_stats_.dropped_video_frame_sets;
    }
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        summary.queue_capacity = static_cast<uint32_t>(kMaxQueuedEvents);
        summary.queue_peak = static_cast<uint32_t>(auxiliary_stats_.publish_queue_peak);
        summary.dropped_events = auxiliary_stats_.dropped_events;
        if (!async_error_.empty() || summary.dropped_events != 0)
        {
            summary.capture_health = core::OrbbecEgoCaptureHealth_Incomplete;
            summary.reason = !async_error_.empty() ? async_error_ : "dropped metadata event";
        }
        else if (dropped_video_frame_sets != 0)
        {
            summary.capture_health = core::OrbbecEgoCaptureHealth_Warning;
            summary.reason = "dropped video frame set";
        }
        else if (events_.size() >= kMaxQueuedEvents * 85 / 100)
        {
            summary.capture_health = core::OrbbecEgoCaptureHealth_Warning;
            summary.reason = "metadata queue reached 85 percent capacity";
        }
        else
            summary.capture_health = core::OrbbecEgoCaptureHealth_Healthy;
    }
    if (!final_error.empty())
    {
        summary.capture_health = core::OrbbecEgoCaptureHealth_Incomplete;
        summary.reason = final_error;
    }
    if (summary.capture_health == core::OrbbecEgoCaptureHealth_Healthy)
    {
        for (const auto& [stream, stats] : stats_)
        {
            if (stats.sequence_gaps == 0)
                continue;
            if (!summary.reason.empty())
                summary.reason += "; ";
            summary.reason += std::string(core::EnumNameOrbbecEgoCameraStream(stream)) +
                              " sequence gaps: " + std::to_string(stats.sequence_gaps);
            summary.capture_health = core::OrbbecEgoCaptureHealth_Warning;
        }
    }
    const auto now = core::os_monotonic_now_ns();
    summary.dropped_video_frame_sets = dropped_video_frame_sets;
    summary.is_final = direct;
    auto state = capture_summary(diagnostic_origin(now), std::move(summary));
    // Producers are frozen at shutdown; bypass the stopped producer queue for the final summary.
    if (direct)
    {
        if (auto* metadata = sink_->metadata_sink())
            metadata->on_device_state(state, now, 0);
    }
    else
        enqueue(DeviceStateEvent{ std::move(state), now });
    last_device_poll_ = std::chrono::steady_clock::now();
}

} // namespace plugins::ego
