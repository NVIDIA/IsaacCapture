// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#if defined(ORBBEC_EGO_ENABLE_PREVIEW)
#    include "preview.hpp"
#endif

#include "calibration.hpp"
#include "device_controls.hpp"
#include "device_diagnostics.hpp"
#include "diagnostics.hpp"
#include "inc/ego_camera/cancellation.hpp"
#include "inc/ego_camera/ego_camera.hpp"
#include "inc/ego_camera/sdk_timestamp.hpp"

#include <libobsensor/ObSensor.hpp>
#include <oxr_utils/os_time.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>
#include <variant>

namespace plugins::ego
{

namespace
{

// SDK 2.9 builds place EGO callbacks on Device or DataAcquisitionDevice.
// Keep handle adoption and destruction matched to the selected SDK headers.
#if __has_include(<libobsensor/hpp/DataAcquisitionDevice.hpp>)
using EgoSdkDevice = ob::DataAcquisitionDevice;
#else
using EgoSdkDevice = ob::Device;
#endif

std::shared_ptr<EgoSdkDevice> adopt_ego_device(std::shared_ptr<ob::Device> device)
{
#if __has_include(<libobsensor/hpp/DataAcquisitionDevice.hpp>)
    return EgoSdkDevice::fromDevice(std::move(device));
#else
    return device;
#endif
}

const char* prepare_sdk_logging()
{
    static std::once_flag once;
    std::call_once(once,
                   []
                   {
                       const auto logger = isaaccapture::Logger::get("isaaccapture.plugins.orbbec_ego.sdk");
                       // Global severity resets every SDK sink; disable native sinks after it.
                       ob::Context::setLoggerSeverity(OB_LOG_SEVERITY_DEBUG);
                       ob::Context::setLoggerToConsole(OB_LOG_SEVERITY_NONE);
                       ob::Context::setLoggerToFile(OB_LOG_SEVERITY_NONE, "");
                       ob::Context::setLoggerToCallback(
                           OB_LOG_SEVERITY_DEBUG,
                           [logger](OBLogSeverity severity, const char* message)
                           {
                               const auto level = severity == OB_LOG_SEVERITY_DEBUG ? spdlog::level::debug :
                                                  severity == OB_LOG_SEVERITY_INFO  ? spdlog::level::info :
                                                  severity == OB_LOG_SEVERITY_WARN  ? spdlog::level::warn :
                                                  severity == OB_LOG_SEVERITY_FATAL ? spdlog::level::critical :
                                                                                      spdlog::level::err;
                               // Vendor callbacks must not unwind into the SDK.
                               try
                               {
                                   std::string line = message ? message : "";
                                   while (!line.empty() && (line.back() == '\n' || line.back() == '\r'))
                                       line.pop_back();
                                   logger->log(level, "{}", line);
                               }
                               catch (...)
                               {
                               }
                           });
                   });
    return "";
}

constexpr size_t kMaxQueuedEvents = 4096;
constexpr size_t kMaxQueuedVideoFrameSets = 256;

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

OBFormat to_ob_format(core::OrbbecEgoPixelFormat format)
{
    switch (format)
    {
    case core::OrbbecEgoPixelFormat_Mjpg:
        return OB_FORMAT_MJPG;
    case core::OrbbecEgoPixelFormat_H264:
        return OB_FORMAT_H264;
    case core::OrbbecEgoPixelFormat_H265:
        return OB_FORMAT_H265;
    default:
        throw std::invalid_argument("Unsupported Ego pixel format");
    }
}

OBFrameType to_ob_frame(core::OrbbecEgoCameraStream stream)
{
    switch (stream)
    {
    case core::OrbbecEgoCameraStream_ColorLeft:
        return OB_FRAME_COLOR_LEFT;
    case core::OrbbecEgoCameraStream_ColorRight:
        return OB_FRAME_COLOR_RIGHT;
    default:
        throw std::invalid_argument("Unsupported Ego camera stream");
    }
}

OBSensorType to_ob_sensor(core::OrbbecEgoCameraStream stream)
{
    switch (stream)
    {
    case core::OrbbecEgoCameraStream_ColorLeft:
        return OB_SENSOR_COLOR_LEFT;
    case core::OrbbecEgoCameraStream_ColorRight:
        return OB_SENSOR_COLOR_RIGHT;
    default:
        throw std::invalid_argument("Unsupported Ego camera stream");
    }
}

bool has_sensor(const std::shared_ptr<ob::Device>& device, OBSensorType sensor)
{
    const auto sensors = device->getSensorList();
    for (uint32_t index = 0; index < sensors->getCount(); ++index)
    {
        if (sensors->getSensorType(index) == sensor)
            return true;
    }
    return false;
}

void print_profiles(const std::shared_ptr<ob::Device>& device, core::OrbbecEgoCameraStream stream)
{
    const auto profiles = device->getSensor(to_ob_sensor(stream))->getStreamProfileList();
    std::ostringstream profiles_message;
    profiles_message << "Available " << core::EnumNameOrbbecEgoCameraStream(stream) << " profiles:";
    for (uint32_t index = 0; index < profiles->getCount(); ++index)
    {
        const auto profile = profiles->getProfile(index);
        if (!profile->is<ob::VideoStreamProfile>())
            continue;
        const auto video = profile->as<ob::VideoStreamProfile>();
        profiles_message << " " << video->getWidth() << "x" << video->getHeight() << "@" << video->getFps() << " "
                         << ob::TypeHelper::convertOBFormatTypeToString(video->getFormat());
    }
    detail::log_warning() << profiles_message.str();
}

std::shared_ptr<ob::VideoStreamProfile> select_profile(const std::shared_ptr<ob::Device>& device,
                                                       const StreamConfig& stream,
                                                       const CaptureConfig& config)
{
    validate_stream_config(stream, config);
    const auto width = stream.width != 0 ? stream.width : config.width;
    const auto height = stream.height != 0 ? stream.height : config.height;
    const auto fps = stream.fps != 0 ? stream.fps : config.fps;
    const auto format = to_ob_format(stream.pixel_format);
    try
    {
        const auto profile = device->getSensor(to_ob_sensor(stream.camera))
                                 ->getStreamProfileList()
                                 ->getVideoStreamProfile(
                                     static_cast<int>(width), static_cast<int>(height), format, static_cast<int>(fps));
        detail::log_info()
            << "Selected " << core::EnumNameOrbbecEgoCameraStream(stream.camera) << " profile: " << profile->getWidth()
            << "x" << profile->getHeight() << "@" << profile->getFps() << " "
            << core::EnumNameOrbbecEgoPixelFormat(stream.pixel_format) << std::endl;
        return profile;
    }
    catch (const ob::Error& error)
    {
        print_profiles(device, stream.camera);
        throw std::runtime_error("No matching Ego profile for " +
                                 std::string(core::EnumNameOrbbecEgoCameraStream(stream.camera)) + ": " + error.what());
    }
}

using ActiveProfiles = std::map<core::OrbbecEgoCameraStream, std::shared_ptr<ob::VideoStreamProfile>>;

std::shared_ptr<ob::Config> make_video_config(const ActiveProfiles& profiles, const std::vector<StreamConfig>& streams)
{
    auto config = std::make_shared<ob::Config>();
    for (const auto& [_, profile] : profiles)
        config->enableStream(profile);
    const bool has_interframe_video = std::any_of(streams.begin(), streams.end(),
                                                  [](const StreamConfig& stream)
                                                  {
                                                      return stream.pixel_format == core::OrbbecEgoPixelFormat_H264 ||
                                                             stream.pixel_format == core::OrbbecEgoPixelFormat_H265;
                                                  });
    // Single-eye capture must not wait for an unrequested color frame.
    // MJPG capture must retain an available eye even when its stereo partner is absent.
    config->setFrameAggregateOutputMode(profiles.size() == 1 ? OB_FRAME_AGGREGATE_OUTPUT_ANY_SITUATION :
                                        has_interframe_video ? OB_FRAME_AGGREGATE_OUTPUT_COLOR_FRAME_REQUIRE :
                                                               OB_FRAME_AGGREGATE_OUTPUT_ANY_SITUATION);
    return config;
}

std::string describe_profiles(const ActiveProfiles& profiles)
{
    std::ostringstream output;
    bool first = true;
    for (const auto& [stream, profile] : profiles)
    {
        if (!first)
            output << ", ";
        first = false;
        output << core::EnumNameOrbbecEgoCameraStream(stream) << "=" << profile->getWidth() << "x"
               << profile->getHeight() << "@" << profile->getFps() << " "
               << ob::TypeHelper::convertOBFormatTypeToString(profile->getFormat());
    }
    return output.str();
}

} // namespace

class EgoCamera::Impl
{
public:
    Impl(const CaptureConfig& config, const std::vector<StreamConfig>& streams, std::unique_ptr<FrameSink> sink)
        : config_(config),
          context_(std::make_unique<ob::Context>(prepare_sdk_logging())),
          streams_(streams),
          sink_(std::move(sink))
    {
#if !defined(ORBBEC_EGO_ENABLE_PREVIEW)
        if (config_.preview)
        {
            throw std::runtime_error(
                "SDL preview is not available in this build. Reconfigure with -DORBBEC_EGO_ENABLE_PREVIEW=ON.");
        }
#endif
#if defined(__linux__) || defined(__ANDROID__)
        context_->setUvcBackendType(OB_UVC_BACKEND_TYPE_LIBUVC);
#endif
        publish_capture_status(CaptureAvailability::Unknown);
        check_cancelled();
        const auto devices = context_->queryDeviceList();
        if (devices->getCount() == 0)
        {
            publish_capture_status(CaptureAvailability::Absent);
            throw std::runtime_error("No Ego devices found. Check USB connection and udev permissions.");
        }

        for (uint32_t index = 0; index < devices->getCount(); ++index)
        {
            check_cancelled();
            const auto candidate = devices->getDevice(index);
            if (!config.device_uid.empty() && candidate->getDeviceInfo()->getUid() != config.device_uid)
                continue;
            bool supports_requested_streams = true;
            for (const auto& stream : streams_)
                supports_requested_streams =
                    supports_requested_streams && has_sensor(candidate, to_ob_sensor(stream.camera));
            if (supports_requested_streams)
            {
                // Adopt the EGO callback owner before pipelines; newer SDK
                // wrappers transfer the base device handle.
                device_ = adopt_ego_device(candidate);
                break;
            }
        }
        if (!device_)
            throw std::runtime_error("No Ego device matches the requested UID and ColorLeft/ColorRight sensors.");
        const auto selected_info = device_->getDeviceInfo();
        selected_device_uid_ = selected_info->getUid();
        selected_device_serial_ = selected_info->getSerialNumber();
        selected_device_vid_ = selected_info->getVid();
        selected_device_pid_ = selected_info->getPid();
        selected_firmware_ = selected_info->getFirmwareVersion();
        publish_capture_status(CaptureAvailability::Starting);

        for (const auto& stream : streams_)
        {
            const auto profile = select_profile(device_, stream, config);
            active_profiles_.emplace(stream.camera, profile);
        }
        initial_profile_description_ = describe_profiles(active_profiles_);
        try
        {
            for (const auto& stream : streams_)
                validate_resolved_stream_config(stream, active_profiles_.at(stream.camera)->getFps());
        }
        catch (const std::exception& error)
        {
            throw std::runtime_error("Ego uid=" + selected_device_uid_ + " firmware=" + selected_firmware_ +
                                     " resolved profiles [" + describe_profiles(active_profiles_) +
                                     "] are not certified: " + error.what());
        }
        const auto pipeline_config = make_video_config(active_profiles_, streams_);

        std::vector<PropertySetting> settings = config.properties;
        if (config.bitrate != 0)
            settings.push_back({ "OB_PROP_COLOR_BITRATE_INT", static_cast<double>(config.bitrate) });
        if (config.dynamic_bitrate_set)
            settings.push_back({ "OB_PROP_COLOR_DYNAMIC_BITRATE_ENABLE_BOOL", config.dynamic_bitrate ? 1.0 : 0.0 });
        struct CaptureReadiness
        {
            std::mutex mutex;
            std::condition_variable wake;
            std::vector<bool> seen;
            std::string error;
            std::atomic<bool> complete{ false };
        };
        struct ExpectedFrame
        {
            OBFrameType frame_type;
            uint32_t width;
            uint32_t height;
            uint32_t fps;
            OBFormat format;
            std::string name;
        };
        std::vector<ExpectedFrame> expected_frames;
        expected_frames.reserve(active_profiles_.size());
        for (const auto& [stream, profile] : active_profiles_)
        {
            expected_frames.push_back({ to_ob_frame(stream), profile->getWidth(), profile->getHeight(), profile->getFps(),
                                        profile->getFormat(), core::EnumNameOrbbecEgoCameraStream(stream) });
        }
        auto readiness = std::make_shared<CaptureReadiness>();
        readiness->seen.resize(expected_frames.size());
        const auto wait_for_video_readiness = [this, &readiness, &expected_frames]
        {
            std::string readiness_error;
            {
                std::unique_lock<std::mutex> lock(readiness->mutex);
                const bool ready = wait_capture_ready(
                    readiness->wake, lock, std::chrono::seconds(10),
                    [&readiness]
                    { return readiness->complete.load(std::memory_order_acquire) || !readiness->error.empty(); },
                    config_.stop_requested);
                if (!ready)
                {
                    readiness_error = "timed out waiting for";
                    for (size_t index = 0; index < expected_frames.size(); ++index)
                    {
                        if (!readiness->seen[index])
                            readiness_error += " " + expected_frames[index].name;
                    }
                }
                else
                    readiness_error = readiness->error;
            }
            if (!readiness_error.empty())
            {
                throw std::runtime_error("Ego uid=" + selected_device_uid_ + " firmware=" + selected_firmware_ +
                                         " active profiles [" + describe_profiles(active_profiles_) +
                                         "] failed readiness: " + readiness_error);
            }
        };
        try
        {
            if (!settings.empty())
            {
                try
                {
                    prepare_controls(settings);
                }
                catch (const CaptureCancelled&)
                {
                    throw;
                }
                catch (const std::exception& error)
                {
                    throw std::runtime_error("Ego uid=" + selected_device_uid_ + " firmware=" + selected_firmware_ +
                                             " controls for profiles [" + describe_profiles(active_profiles_) +
                                             "] failed: " + error.what());
                }
            }
            capture_device_snapshot();
            pipeline_ = std::make_unique<ob::Pipeline>(device_);
#if defined(ORBBEC_EGO_ENABLE_PREVIEW)
            if (config_.preview)
                preview_ = std::make_unique<Preview>();
#endif
            publish_calibration(pipeline_config);
            start_device_state();
            if (config_.enable_imu)
                start_imu();
            wait_for_auxiliary_readiness();
            try
            {
                pipeline_->start(
                    pipeline_config,
                    [this, readiness, expected_frames](std::shared_ptr<ob::FrameSet> frame_set)
                    {
                        if (!frame_set || !accepting_callbacks_.load(std::memory_order_acquire))
                            return;
                        const bool active_capture = capture_active_.load(std::memory_order_acquire);
                        std::vector<bool> observed(expected_frames.size());
                        std::string observation_error;
                        if (!readiness->complete.load(std::memory_order_acquire))
                        {
                            try
                            {
                                for (size_t index = 0; index < expected_frames.size(); ++index)
                                {
                                    const auto& expected = expected_frames[index];
                                    const auto raw_frame = frame_set->getFrame(expected.frame_type);
                                    if (!raw_frame)
                                        continue;
                                    const auto frame = raw_frame->as<ob::VideoFrame>();
                                    const auto profile =
                                        frame ? frame->getStreamProfile()->as<ob::VideoStreamProfile>() : nullptr;
                                    if (!frame || !profile || frame->getWidth() != expected.width ||
                                        frame->getHeight() != expected.height ||
                                        frame->getFormat() != expected.format || profile->getFps() != expected.fps)
                                    {
                                        observation_error =
                                            "SDK returned a different active profile for " + expected.name;
                                        if (frame && profile)
                                        {
                                            observation_error +=
                                                ": actual=" + std::to_string(frame->getWidth()) + "x" +
                                                std::to_string(frame->getHeight()) + "@" +
                                                std::to_string(profile->getFps()) + " " +
                                                ob::TypeHelper::convertOBFormatTypeToString(frame->getFormat());
                                        }
                                        break;
                                    }
                                    observed[index] = true;
                                }
                            }
                            catch (const std::exception& error)
                            {
                                observation_error = error.what();
                            }
                            catch (...)
                            {
                                observation_error = "unknown error while inspecting a frameset";
                            }
                        }
                        if (!observation_error.empty())
                        {
                            std::lock_guard<std::mutex> lock(readiness->mutex);
                            readiness->error = std::move(observation_error);
                            readiness->wake.notify_one();
                            return;
                        }
                        if (!active_capture)
                        {
                            if (readiness->complete.load(std::memory_order_relaxed))
                                return;
                            std::lock_guard<std::mutex> lock(readiness->mutex);
                            auto& seen = readiness->seen;
                            for (size_t index = 0; index < observed.size(); ++index)
                                seen[index] = seen[index] || observed[index];
                            if (std::all_of(seen.begin(), seen.end(), [](bool value) { return value; }))
                            {
                                readiness->complete.store(true, std::memory_order_release);
                                readiness->wake.notify_one();
                            }
                            return;
                        }
                        const int64_t arrival_time_local_common_clock_ns = core::os_monotonic_now_ns();
                        {
                            std::lock_guard<std::mutex> lock(video_queue_mutex_);
                            if (!accepting_callbacks_.load(std::memory_order_relaxed))
                                return;
                            if (video_frame_sets_.size() >= kMaxQueuedVideoFrameSets)
                            {
                                ++auxiliary_stats_.dropped_video_frame_sets;
                                return;
                            }
                            video_frame_sets_.emplace_back(std::move(frame_set), arrival_time_local_common_clock_ns);
                        }
                        video_queue_cv_.notify_one();
                    });
                video_pipeline_started_ = true;
            }
            catch (const ob::Error& error)
            {
                throw std::runtime_error("Ego uid=" + selected_device_uid_ + " firmware=" + selected_firmware_ +
                                         " rejected simultaneous profiles [" + describe_profiles(active_profiles_) +
                                         "]: " + error.what());
            }
            wait_for_video_readiness();
            // SDK stream startup can reset image controls; reapply before capture begins.
            reapply_controls("startup");
            verify_active_controls();
            capture_active_.store(true, std::memory_order_release);
            capture_epoch_started_ns_ = core::os_monotonic_now_ns();
            publish_periodic_device_state();
            wait_for_active_auxiliary_readiness();
            device_changed_callback_id_ = context_->registerDeviceChangedCallback(
                [this](const std::shared_ptr<ob::DeviceList>& removed, const std::shared_ptr<ob::DeviceList>&)
                {
                    if (device_list_contains_uid(removed, selected_device_uid_))
                        device_removed_.store(true, std::memory_order_release);
                });
            device_changed_callback_registered_ = true;
            publish_capture_status(CaptureAvailability::Streaming);
            detail::log_info() << "Ego pipeline started for " << selected_device_uid_ << std::endl;
        }
        catch (...)
        {
            try
            {
                shutdown(false);
            }
            catch (const std::exception& error)
            {
                detail::log_error() << "Ego cleanup after startup error failed: " << error.what() << std::endl;
            }
            throw;
        }
    }

    ~Impl()
    {
        shutdown_noexcept(false);
    }

    static bool is_removed_device_cleanup_error(const ob::Error& error, bool device_removal_confirmed) noexcept
    {
        // SDK 2.9 reports deactivated components as call-sequence errors. Only
        // the removed capture instance may tolerate this during reconnect teardown.
        return device_removal_confirmed && error.getStatus() == OB_ERROR_WRONG_API_CALL_SEQUENCE &&
               error.getExceptionType() == OB_EXCEPTION_TYPE_WRONG_API_CALL_SEQUENCE && error.what() &&
               std::string_view(error.what()).find("Device is deactivated/disconnected! DeviceComponentId:") == 0;
    }

    static std::string stop_pipeline_noexcept(std::unique_ptr<ob::Pipeline>& pipeline,
                                              bool& started,
                                              const char* name,
                                              bool device_removal_confirmed = false) noexcept
    {
        if (!pipeline)
            return {};
        const auto stop_started = std::chrono::steady_clock::now();
        std::string failure;
        if (started)
        {
            try
            {
                pipeline->stop();
            }
            catch (const ob::Error& error)
            {
                if (is_removed_device_cleanup_error(error, device_removal_confirmed))
                    detail::log_info()
                        << "Ego " << name << " pipeline already inactive after confirmed removal: " << error.what();
                else
                    failure = std::string("Ego ") + name + " pipeline stop failed: " + error.what();
            }
            catch (const std::exception& error)
            {
                failure = std::string("Ego ") + name + " pipeline stop failed: " + error.what();
            }
            catch (...)
            {
                failure = std::string("Ego ") + name + " pipeline stop failed: unknown error";
            }
        }
        started = false;
        pipeline.reset();
        detail::log_info()
            << "Ego " << name << " pipeline stop completed in "
            << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - stop_started).count()
            << " ms";
        return failure;
    }

    static bool device_list_contains_uid(const std::shared_ptr<ob::DeviceList>& devices, const std::string& uid)
    {
        if (!devices)
            return false;
        for (uint32_t index = 0; index < devices->getCount(); ++index)
        {
            try
            {
                // Read the removal snapshot without trying to open a disconnected device.
                const auto* candidate_uid = devices->getUid(index);
                if (candidate_uid && uid == candidate_uid)
                    return true;
            }
            catch (const std::exception&)
            {
            }
        }
        return false;
    }

    bool is_selected_physical_device(const std::shared_ptr<ob::DeviceInfo>& info) const
    {
        if (!info)
            return false;
        // SDK 2.9.0 EGO UIDs include the USB enumeration address and can change after a physical reconnect.
        if (!selected_device_serial_.empty())
        {
            return info->getSerialNumber() == selected_device_serial_ && info->getVid() == selected_device_vid_ &&
                   info->getPid() == selected_device_pid_;
        }
        return info->getUid() == selected_device_uid_;
    }

    std::shared_ptr<ob::Device> find_selected_device()
    {
        const auto devices = context_->queryDeviceList();
        for (uint32_t index = 0; index < devices->getCount(); ++index)
        {
            const auto candidate = devices->getDevice(index);
            if (!is_selected_physical_device(candidate->getDeviceInfo()))
                continue;
            if (std::all_of(streams_.begin(), streams_.end(),
                            [&candidate](const StreamConfig& stream)
                            { return has_sensor(candidate, to_ob_sensor(stream.camera)); }))
            {
                return candidate;
            }
        }
        return nullptr;
    }

    void publish_capture_status(CaptureAvailability availability, const std::string& error = {}) noexcept
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

    DiagnosticOrigin diagnostic_origin(int64_t observed_at_ns) const
    {
        return { selected_device_uid_, capture_epoch_, observed_at_ns };
    }

    void publish_recovery_event(core::OrbbecEgoRecoveryStage stage, const std::string& reason)
    {
        const auto now = core::os_monotonic_now_ns();
        auto state = recovery_event(diagnostic_origin(now), stage, reason, auxiliary_stats_.reconnect_attempts);
        enqueue(DeviceStateEvent{ std::move(state), now });
        drain_events();
    }

    void reset_stream_readiness()
    {
        accel_ready_.store(false, std::memory_order_release);
        gyro_ready_.store(false, std::memory_order_release);
        active_accel_ready_.store(false, std::memory_order_release);
        active_gyro_ready_.store(false, std::memory_order_release);
        last_accel_arrival_ns_.store(0, std::memory_order_release);
        last_gyro_arrival_ns_.store(0, std::memory_order_release);
        last_video_arrival_ns_.clear();
    }

    void stop_capture_for_reconnect(bool device_removal_confirmed = false)
    {
        accepting_callbacks_.store(false, std::memory_order_release);
        capture_active_.store(false, std::memory_order_release);
        try
        {
            if (device_)
                device_->clearEgoStateCallback();
        }
        catch (const ob::Error& error)
        {
            if (is_removed_device_cleanup_error(error, device_removal_confirmed))
            {
                detail::log_info()
                    << "Ego device-state callback already inactive after confirmed removal: " << error.what();
            }
            else
            {
                cleanup_error_ +=
                    std::string("Ego device-state callback clear during reconnect failed: ") + error.what() + "; ";
                detail::log_warning() << cleanup_error_;
            }
        }
        catch (const std::exception& error)
        {
            cleanup_error_ +=
                std::string("Ego device-state callback clear during reconnect failed: ") + error.what() + "; ";
            detail::log_warning() << cleanup_error_;
        }
        catch (...)
        {
            cleanup_error_ += "Ego device-state callback clear during reconnect failed: unknown error; ";
            detail::log_warning() << cleanup_error_;
        }
        const auto report_stop_error = [this](const std::string& error)
        {
            if (!error.empty())
            {
                cleanup_error_ += error + "; ";
                detail::log_warning() << "Warning: " << error << std::endl;
            }
        };
        report_stop_error(stop_pipeline_noexcept(pipeline_, video_pipeline_started_, "video", device_removal_confirmed));
        report_stop_error(stop_pipeline_noexcept(imu_pipeline_, imu_pipeline_started_, "IMU", device_removal_confirmed));
        drain_video_frames(false);
        flush_imu(core::OrbbecEgoImuSensor_Accel);
        flush_imu(core::OrbbecEgoImuSensor_Gyro);
        drain_events();
        active_profiles_.clear();
        device_.reset();
        reset_stream_readiness();
    }

    void begin_recovery(const std::string& reason)
    {
        publish_capture_status(device_removed_.load(std::memory_order_acquire) ? CaptureAvailability::Absent :
                                                                                 CaptureAvailability::Recovering,
                               reason);
        if (config_.reconnect_timeout_seconds == 0)
            throw RecoverableCaptureError(reason + "; automatic reconnect is disabled");
        if (recovering_)
            return;
        recovering_ = true;
        recovery_device_present_ = !device_removed_.load(std::memory_order_acquire);
        recovery_reason_ = reason;
        recovery_last_error_.clear();
        const auto now = std::chrono::steady_clock::now();
        recovery_deadline_ = now + std::chrono::seconds(config_.reconnect_timeout_seconds);
        next_reconnect_attempt_ = now;
        publish_recovery_event(core::OrbbecEgoRecoveryStage_Started, reason);
        detail::log_error() << "Ego device serial=" << selected_device_serial_ << " uid=" << selected_device_uid_
                            << " disconnected; waiting up to " << config_.reconnect_timeout_seconds
                            << " seconds for the same physical device" << std::endl;
        stop_capture_for_reconnect(device_removed_.load(std::memory_order_acquire));
    }

    void start_recovered_video(const std::shared_ptr<ob::Config>& pipeline_config)
    {
        pipeline_ = std::make_unique<ob::Pipeline>(device_);
        pipeline_->start(pipeline_config,
                         [this](std::shared_ptr<ob::FrameSet> frame_set)
                         {
                             if (!frame_set || !accepting_callbacks_.load(std::memory_order_acquire) ||
                                 !capture_active_.load(std::memory_order_acquire))
                             {
                                 return;
                             }
                             const int64_t arrival_time_local_common_clock_ns = core::os_monotonic_now_ns();
                             {
                                 std::lock_guard<std::mutex> lock(video_queue_mutex_);
                                 if (video_frame_sets_.size() >= kMaxQueuedVideoFrameSets)
                                 {
                                     ++auxiliary_stats_.dropped_video_frame_sets;
                                     return;
                                 }
                                 video_frame_sets_.emplace_back(std::move(frame_set), arrival_time_local_common_clock_ns);
                             }
                             video_queue_cv_.notify_one();
                         });
        video_pipeline_started_ = true;
    }

    void reapply_controls(std::string_view operation = "reconnect")
    {
        check_cancelled();
        try
        {
            for (const auto& setting : requested_controls_)
            {
                check_cancelled();
                const auto item = find_property(device_, setting.name);
                validate_property_value(device_, item, setting.value);
                if ((item.permission & OB_PERMISSION_READ) != 0)
                {
                    // Bitrate cannot be rewritten while streaming; retain controls already at the requested value.
                    const double tolerance =
                        item.type == OB_FLOAT_PROPERTY ? std::max(1.0, std::abs(setting.value)) * 1e-6 : 0.0;
                    if (std::abs(read_property(device_, item) - setting.value) <= tolerance)
                        continue;
                }
                write_property(device_, item, setting.value);
                if ((item.permission & OB_PERMISSION_READ) != 0)
                    verify_property_readback(device_, item, setting.value, operation);
            }
        }
        catch (const CaptureCancelled&)
        {
            throw;
        }
        catch (const std::exception& error)
        {
            throw FatalReconnectError("failed to restore requested controls: " + std::string(error.what()));
        }
    }

    void wait_for_recovered_video_readiness()
    {
        const auto ready = [this]
        {
            return std::all_of(streams_.begin(), streams_.end(),
                               [this](const StreamConfig& stream)
                               { return stats_.at(stream.camera).epoch_frame_count > 0; });
        };
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (!ready() && std::chrono::steady_clock::now() < deadline)
        {
            check_cancelled();
            drain_video_frames();
            drain_events();
            if (const auto failure = sink_->metadata_error(); !failure.empty())
                throw std::runtime_error("Ego metadata publication failed: " + failure);
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (!async_error_.empty())
                throw std::runtime_error(async_error_);
        }
        if (ready())
            return;
        std::string missing;
        for (const auto& stream : streams_)
        {
            if (stats_.at(stream.camera).epoch_frame_count == 0)
                missing += " " + std::string(core::EnumNameOrbbecEgoCameraStream(stream.camera));
        }
        throw std::runtime_error("Ego uid=" + selected_device_uid_ + " firmware=" + selected_firmware_ +
                                 " active profiles [" + describe_profiles(active_profiles_) +
                                 "] timed out waiting for video streams after reconnect:" + missing);
    }

    void wait_for_recovery_stability() const
    {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (std::chrono::steady_clock::now() < deadline)
        {
            check_cancelled();
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }

        const int64_t now = core::os_monotonic_now_ns();
        const auto is_recent = [now](const std::atomic<int64_t>& arrival)
        { return now - arrival.load(std::memory_order_acquire) <= 500'000'000LL; };
        std::string stalled;
        if (config_.enable_imu && !is_recent(last_accel_arrival_ns_))
            stalled += " Accel";
        if (config_.enable_imu && !is_recent(last_gyro_arrival_ns_))
            stalled += " Gyro";
        if (!stalled.empty())
        {
            throw std::runtime_error("Ego uid=" + selected_device_uid_ +
                                     " auxiliary streams did not remain active during reconnect readiness:" + stalled);
        }
    }

    void check_cancelled() const
    {
        check_capture_cancelled(config_.stop_requested);
    }

    void attempt_reconnect()
    {
        check_cancelled();
        const auto now = std::chrono::steady_clock::now();
        if (now >= recovery_deadline_)
        {
            std::string reason = "Ego device serial=" + selected_device_serial_ + " last_uid=" + selected_device_uid_ +
                                 " did not recover within " + std::to_string(config_.reconnect_timeout_seconds) +
                                 " seconds";
            if (!recovery_last_error_.empty())
                reason += "; last restart error: " + recovery_last_error_;
            publish_recovery_event(core::OrbbecEgoRecoveryStage_Failed, reason);
            throw std::runtime_error(reason);
        }
        if (now < next_reconnect_attempt_)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            return;
        }
        next_reconnect_attempt_ = now + std::chrono::milliseconds(config_.reconnect_interval_milliseconds);
        ++auxiliary_stats_.reconnect_attempts;
        try
        {
            auto recovered_device = find_selected_device();
            if (!recovered_device)
            {
                recovery_device_present_ = false;
                publish_capture_status(CaptureAvailability::Absent, recovery_reason_);
                return;
            }
            recovery_device_present_ = true;
            publish_capture_status(CaptureAvailability::Recovering, recovery_reason_);
            const auto recovered_info = recovered_device->getDeviceInfo();
            const auto recovered_uid = recovered_info->getUid();
            const auto firmware = recovered_info->getFirmwareVersion();
            if (firmware != selected_firmware_)
                throw FatalReconnectError("firmware changed from " + selected_firmware_ + " to " + firmware);
            if (recovered_uid != selected_device_uid_)
            {
                detail::log_info()
                    << "Ego device serial=" << selected_device_serial_
                    << " re-enumerated from uid=" << selected_device_uid_ << " to uid=" << recovered_uid << std::endl;
            }
            device_ = adopt_ego_device(std::move(recovered_device));
            selected_device_uid_ = recovered_uid;
            for (const auto& stream : streams_)
            {
                const auto profile = select_profile(device_, stream, config_);
                validate_resolved_stream_config(stream, profile->getFps());
                active_profiles_.emplace(stream.camera, profile);
            }
            if (describe_profiles(active_profiles_) != initial_profile_description_)
            {
                throw FatalReconnectError("resolved profiles changed to [" + describe_profiles(active_profiles_) + "]");
            }

            accepting_callbacks_.store(true, std::memory_order_release);
            reset_stream_readiness();
            const auto pipeline_config = make_video_config(active_profiles_, streams_);
            start_device_state();
            if (config_.enable_imu)
                start_imu();
            wait_for_auxiliary_readiness();
            start_recovered_video(pipeline_config);
            wait_for_recovery_stability();
            reapply_controls();
            verify_active_controls();
            publish_calibration(pipeline_config);

            // A restart may reset device counters; reset continuity checks and encoded-video gates.
            ++capture_epoch_;
            auxiliary_stats_.capture_epoch = capture_epoch_;
            for (auto& [_, stats] : stats_)
                stats.epoch_frame_count = 0;
            sink_->begin_capture_epoch();
            capture_epoch_started_ns_ = core::os_monotonic_now_ns();
            capture_active_.store(true, std::memory_order_release);
            wait_for_recovered_video_readiness();
            wait_for_active_auxiliary_readiness();
            publish_periodic_device_state();
            ++auxiliary_stats_.successful_reconnects;
            recovering_ = false;
            device_removed_.store(false, std::memory_order_release);
            publish_recovery_event(core::OrbbecEgoRecoveryStage_Recovered, recovery_reason_);
            publish_capture_status(CaptureAvailability::Streaming);
            detail::log_info() << "Ego device " << selected_device_uid_ << " recovered as capture epoch "
                               << capture_epoch_ << std::endl;
        }
        catch (const CaptureCancelled&)
        {
            throw;
        }
        catch (const FatalReconnectError& error)
        {
            recovery_last_error_ = error.what();
            try
            {
                stop_capture_for_reconnect();
                publish_recovery_event(core::OrbbecEgoRecoveryStage_Failed, recovery_last_error_);
            }
            catch (const std::exception& cleanup_error)
            {
                recovery_last_error_ += "; cleanup failed: " + std::string(cleanup_error.what());
            }
            throw std::runtime_error("Ego reconnect cannot safely continue: " + recovery_last_error_);
        }
        catch (const std::exception& error)
        {
            recovery_last_error_ = error.what();
            const auto sink_failure = sink_->metadata_error();
            detail::log_error() << "Ego reconnect attempt " << auxiliary_stats_.reconnect_attempts
                                << " failed: " << recovery_last_error_ << std::endl;
            try
            {
                stop_capture_for_reconnect();
            }
            catch (const std::exception& cleanup_error)
            {
                recovery_last_error_ += "; cleanup failed: " + std::string(cleanup_error.what());
            }
            if (!sink_failure.empty())
                throw std::runtime_error("Ego metadata publication failed during reconnect: " + sink_failure);
        }
    }

    void prepare_controls(const std::vector<PropertySetting>& settings)
    {
        requested_controls_.clear();
        for (const auto& setting : settings)
        {
            const auto expected = std::find_if(requested_controls_.begin(), requested_controls_.end(),
                                               [&setting](const auto& current) { return current.name == setting.name; });
            if (expected == requested_controls_.end())
                requested_controls_.push_back(setting);
            else
                *expected = setting;
            const auto item = find_property(device_, setting.name);
            if ((item.permission & OB_PERMISSION_WRITE) == 0)
                throw std::invalid_argument(std::string(item.name) + " is read-only");
            if (!config_.persist_controls && (item.permission & OB_PERMISSION_READ) == 0)
            {
                throw std::invalid_argument(std::string(item.name) +
                                            " is write-only; use --persist-controls to acknowledge it cannot be restored");
            }
            if (!config_.persist_controls &&
                std::none_of(original_properties_.begin(), original_properties_.end(),
                             [&setting](const PropertySetting& original) { return original.name == setting.name; }))
            {
                original_properties_.push_back({ setting.name, read_property(device_, item) });
            }
        }

        struct PreflightProfile
        {
            OBFrameType frame_type;
            uint32_t width;
            uint32_t height;
            uint32_t fps;
            OBFormat format;
            std::string name;
        };
        std::vector<PreflightProfile> expected_profiles;
        expected_profiles.reserve(active_profiles_.size());
        for (const auto& [stream, profile] : active_profiles_)
        {
            expected_profiles.push_back({ to_ob_frame(stream), profile->getWidth(), profile->getHeight(),
                                          profile->getFps(), profile->getFormat(),
                                          core::EnumNameOrbbecEgoCameraStream(stream) });
        }
        struct PreflightState
        {
            std::mutex mutex;
            std::condition_variable wake;
            std::vector<bool> seen;
            std::string error;
            bool accepting = true;
        };
        auto preflight_state = std::make_shared<PreflightState>();
        preflight_state->seen.resize(expected_profiles.size());
        auto preflight = std::make_unique<ob::Pipeline>(device_);
        bool preflight_started = false;
        try
        {
            // EGO encoded streams can time out in SDK pull mode while callback delivery works,
            // so control preflight mirrors the capture callback path.
            preflight->start(
                make_video_config(active_profiles_, streams_),
                [preflight_state, expected_profiles](std::shared_ptr<ob::FrameSet> frame_set)
                {
                    if (!frame_set)
                        return;
                    std::lock_guard<std::mutex> lock(preflight_state->mutex);
                    if (!preflight_state->accepting)
                        return;
                    try
                    {
                        for (size_t index = 0; index < expected_profiles.size(); ++index)
                        {
                            const auto& expected = expected_profiles[index];
                            const auto raw_frame = frame_set->getFrame(expected.frame_type);
                            if (!raw_frame)
                                continue;
                            const auto frame = raw_frame->as<ob::VideoFrame>();
                            if (!frame)
                                continue;
                            const auto profile = frame->getStreamProfile()->as<ob::VideoStreamProfile>();
                            if (!profile || frame->getFormat() != expected.format || frame->getWidth() != expected.width ||
                                frame->getHeight() != expected.height || profile->getFps() != expected.fps)
                            {
                                preflight_state->error = "SDK returned a different profile for " + expected.name;
                                break;
                            }
                            preflight_state->seen[index] = true;
                        }
                    }
                    catch (const std::exception& error)
                    {
                        preflight_state->error = error.what();
                    }
                    const bool complete = std::all_of(
                        preflight_state->seen.begin(), preflight_state->seen.end(), [](bool seen) { return seen; });
                    if (complete || !preflight_state->error.empty())
                        preflight_state->wake.notify_one();
                });
            preflight_started = true;
            std::string preflight_error;
            {
                std::unique_lock<std::mutex> lock(preflight_state->mutex);
                const bool finished = wait_capture_ready(
                    preflight_state->wake, lock, std::chrono::seconds(10),
                    [&preflight_state]
                    {
                        return !preflight_state->error.empty() ||
                               std::all_of(preflight_state->seen.begin(), preflight_state->seen.end(),
                                           [](bool seen) { return seen; });
                    },
                    config_.stop_requested);
                if (!finished)
                {
                    preflight_error = "timed out waiting for";
                    for (size_t index = 0; index < expected_profiles.size(); ++index)
                    {
                        if (!preflight_state->seen[index])
                            preflight_error += " " + expected_profiles[index].name;
                    }
                }
                else
                    preflight_error = preflight_state->error;
                preflight_state->accepting = false;
            }
            if (!preflight_error.empty())
                throw std::runtime_error(preflight_error);
        }
        catch (const std::exception& error)
        {
            {
                std::lock_guard<std::mutex> lock(preflight_state->mutex);
                preflight_state->accepting = false;
            }
            const auto stop_error = stop_pipeline_noexcept(preflight, preflight_started, "control preflight");
            original_properties_.clear();
            // Ctrl-C during the preflight wait is a cancellation, not a rejected profile.
            if (dynamic_cast<const CaptureCancelled*>(&error) != nullptr)
            {
                if (!stop_error.empty())
                    detail::log_warning() << stop_error;
                throw;
            }
            std::string message = "Ego uid=" + selected_device_uid_ + " firmware=" + selected_firmware_ +
                                  " control preflight rejected simultaneous profiles [" +
                                  describe_profiles(active_profiles_) + "]: " + error.what();
            if (!stop_error.empty())
                message += "; " + stop_error;
            throw std::runtime_error(message);
        }
        {
            std::lock_guard<std::mutex> lock(preflight_state->mutex);
            preflight_state->accepting = false;
        }
        if (const auto error = stop_pipeline_noexcept(preflight, preflight_started, "control preflight"); !error.empty())
        {
            original_properties_.clear();
            throw std::runtime_error(error);
        }

        try
        {
            for (const auto& setting : settings)
                validate_property_value(device_, find_property(device_, setting.name), setting.value);
            for (const auto& setting : settings)
            {
                const auto item = find_property(device_, setting.name);
                controls_applied_ = true;
                write_property(device_, item, setting.value);
                if ((item.permission & OB_PERMISSION_READ) != 0)
                    verify_property_readback(device_, item, setting.value, "set");
                detail::log_info() << "Set " << item.name << "=" << setting.value << std::endl;
            }
        }
        catch (...)
        {
            if (controls_applied_)
            {
                try
                {
                    restore_properties();
                }
                catch (const std::exception& error)
                {
                    detail::log_error()
                        << "Failed to restore Ego controls after configuration error: " << error.what() << std::endl;
                }
            }
            else
                original_properties_.clear();
            throw;
        }
    }

    void verify_active_controls()
    {
        for (const auto& setting : requested_controls_)
        {
            const auto item = find_property(device_, setting.name);
            if ((item.permission & OB_PERMISSION_READ) == 0)
                continue;
            try
            {
                verify_property_readback(device_, item, setting.value, "active-profile");
            }
            catch (const std::exception& error)
            {
                throw std::runtime_error("Ego uid=" + selected_device_uid_ + " firmware=" + selected_firmware_ +
                                         " active profiles [" + describe_profiles(active_profiles_) +
                                         "] changed a requested control: " + error.what());
            }
        }
    }

    void capture_device_snapshot()
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

    void reacquire_device()
    {
        const auto devices = context_->queryDeviceList();
        for (uint32_t index = 0; index < devices->getCount(); ++index)
        {
            const auto candidate = devices->getDevice(index);
            if (is_selected_physical_device(candidate->getDeviceInfo()))
            {
                selected_device_uid_ = candidate->getDeviceInfo()->getUid();
                device_ = adopt_ego_device(candidate);
                return;
            }
        }
        throw std::runtime_error("Ego device " + selected_device_uid_ + " disconnected before controls restored");
    }

    void close()
    {
        shutdown(true);
    }

    void shutdown(bool promote_metadata)
    {
        const auto shutdown_started = std::chrono::steady_clock::now();
        if (shutdown_complete_)
            return;
        accepting_callbacks_.store(false, std::memory_order_release);
        capture_active_.store(false, std::memory_order_release);
        std::string shutdown_error = cleanup_error_;
        if (recovering_)
            shutdown_error += "Ego capture stopped before device recovery completed";
        const auto remember_error = [&shutdown_error](const std::string& error)
        {
            if (error.empty())
                return;
            if (!shutdown_error.empty())
                shutdown_error += "; ";
            shutdown_error += error;
        };
        if (device_changed_callback_registered_)
        {
            try
            {
                context_->unregisterDeviceChangedCallback(device_changed_callback_id_);
                device_changed_callback_registered_ = false;
            }
            catch (const std::exception& error)
            {
                remember_error(std::string("Ego device-change callback unregister failed: ") + error.what());
            }
        }
        try
        {
            if (device_)
                device_->clearEgoStateCallback();
        }
        catch (const ob::Error& error)
        {
            remember_error(std::string("Ego device-state callback clear failed: ") + error.what());
        }
        catch (...)
        {
            remember_error("Ego device-state callback clear failed: unknown error");
        }
        // Freeze callbacks before stopping producers so SDK stop latency cannot
        // turn post-stop frames into queue overflow or a longer recording tail.
        remember_error(stop_pipeline_noexcept(pipeline_, video_pipeline_started_, "video"));
        remember_error(stop_pipeline_noexcept(imu_pipeline_, imu_pipeline_started_, "IMU"));
        const auto finalize = [&remember_error](const char* operation, auto&& function)
        {
            try
            {
                function();
            }
            catch (const std::exception& error)
            {
                remember_error(std::string("Ego ") + operation + " failed: " + error.what());
            }
            catch (...)
            {
                remember_error(std::string("Ego ") + operation + " failed: unknown error");
            }
        };
        finalize("video drain", [this] { drain_video_frames(false); });
        finalize("media sidecar close", [this] { sink_->close_media(); });
        // Video loss is reported, not fatal.
        for (const auto& [stream, stats] : stats_)
        {
            if (stats.sequence_gaps != 0)
            {
                detail::log_warning() << "Ego " << core::EnumNameOrbbecEgoCameraStream(stream) << " recorded "
                                      << stats.sequence_gaps << " sequence gaps";
            }
        }
        finalize("IMU flush",
                 [this]
                 {
                     flush_imu(core::OrbbecEgoImuSensor_Accel);
                     flush_imu(core::OrbbecEgoImuSensor_Gyro);
                 });
        finalize("event drain", [this] { drain_events(); });
        uint64_t dropped_video_frame_sets = 0;
        {
            std::lock_guard<std::mutex> lock(video_queue_mutex_);
            dropped_video_frame_sets = auxiliary_stats_.dropped_video_frame_sets;
        }
        if (dropped_video_frame_sets != 0)
            detail::log_warning() << "Ego capture dropped " << dropped_video_frame_sets << " video frame sets";
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (!async_error_.empty())
                remember_error(async_error_);
            if (auxiliary_stats_.dropped_events != 0)
            {
                remember_error("Ego capture dropped " + std::to_string(auxiliary_stats_.dropped_events) +
                               " metadata events");
            }
        }
        if (const auto failure = sink_->metadata_error(); !failure.empty())
            remember_error("Ego metadata publication failed: " + failure);
        try
        {
            restore_properties();
        }
        catch (const std::exception& error)
        {
            remember_error(error.what());
        }
        finalize("SDK resource release",
                 [this]
                 {
                     device_.reset();
                     context_.reset();
                 });
        // Aborted capture must not produce a healthy final summary, even if cleanup succeeds.
        const auto final_diagnostic_error = !shutdown_error.empty() ? shutdown_error :
                                            promote_metadata        ? std::string{} :
                                                                      "Capture aborted before completion";
        finalize("final diagnostic summary",
                 [this, &final_diagnostic_error] { publish_periodic_device_state(final_diagnostic_error, true); });
        if (promote_metadata && shutdown_error.empty())
        {
            finalize("metadata close", [this] { sink_->close_metadata(); });
            if (shutdown_error.empty())
                finalize("metadata commit", [this] { sink_->commit_metadata(); });
        }
        else
            finalize("metadata abort", [this] { sink_->abort_metadata(); });
        shutdown_complete_ = true;
        detail::log_info()
            << "Ego shutdown completed in "
            << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - shutdown_started).count()
            << " ms; archive_committed=" << (promote_metadata && shutdown_error.empty());
        if (!shutdown_error.empty())
            throw std::runtime_error(shutdown_error);
    }

    void shutdown_noexcept(bool promote_metadata) noexcept
    {
        try
        {
            shutdown(promote_metadata);
        }
        catch (const std::exception& error)
        {
            detail::log_error() << "Ego shutdown failed: " << error.what() << std::endl;
        }
        catch (...)
        {
            detail::log_error() << "Ego shutdown failed: unknown error" << std::endl;
        }
    }

    void restore_properties()
    {
        if (!controls_applied_ || original_properties_.empty())
            return;
        reacquire_device();
        std::string restore_error;
        for (auto it = original_properties_.rbegin(); it != original_properties_.rend(); ++it)
        {
            try
            {
                const auto item = find_property(device_, it->name);
                write_property(device_, item, it->value, false);
                verify_property_readback(device_, item, it->value, "restore");
                detail::log_info() << "Restored " << item.name << "=" << it->value << std::endl;
            }
            catch (const std::exception& error)
            {
                if (!restore_error.empty())
                    restore_error += "; ";
                restore_error += "failed to restore " + it->name + ": " + error.what();
            }
        }
        original_properties_.clear();
        controls_applied_ = false;
        if (!restore_error.empty())
            throw std::runtime_error(restore_error);
    }

    void update()
    {
        try
        {
            update_capture();
        }
        catch (const CaptureCancelled&)
        {
            throw;
        }
        catch (const std::exception& error)
        {
            cleanup_error_ = error.what();
            publish_capture_status(CaptureAvailability::Failed, error.what());
            throw;
        }
    }

    void update_capture()
    {
        if (recovering_)
        {
            publish_capture_status(
                recovery_device_present_ ? CaptureAvailability::Recovering : CaptureAvailability::Absent,
                recovery_reason_);
            attempt_reconnect();
            return;
        }
        try
        {
            if (device_removed_.load(std::memory_order_acquire))
                throw RecoverableCaptureError("SDK reported removal of UID " + selected_device_uid_);
            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                if (!async_error_.empty())
                    throw std::runtime_error(async_error_);
            }
            if (const auto error = sink_->metadata_error(); !error.empty())
                throw std::runtime_error("Ego metadata publication failed: " + error);
            drain_events();
            drain_video_frames();
            validate_video_liveness();
            validate_auxiliary_liveness();
            publish_capture_status(CaptureAvailability::Streaming);
            if (std::chrono::steady_clock::now() - last_device_poll_ >= std::chrono::seconds(5))
                publish_periodic_device_state();
            drain_events();
            if (const auto error = sink_->metadata_error(); !error.empty())
                throw std::runtime_error("Ego metadata publication failed: " + error);
        }
        catch (const RecoverableCaptureError& error)
        {
            begin_recovery(error.what());
        }
    }

    void validate_video_liveness() const
    {
        if (!capture_active_.load(std::memory_order_acquire))
            return;
        const int64_t now = core::os_monotonic_now_ns();
        for (const auto& stream : streams_)
        {
            const auto observed = last_video_arrival_ns_.find(stream.camera);
            const int64_t reference =
                observed == last_video_arrival_ns_.end() ? capture_epoch_started_ns_ : observed->second;
            if (now - reference > 5'000'000'000LL)
            {
                throw RecoverableCaptureError("Ego " + std::string(core::EnumNameOrbbecEgoCameraStream(stream.camera)) +
                                              " stopped delivering frames for five seconds");
            }
        }
    }

    void validate_auxiliary_liveness() const
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

    void drain_video_frames(bool wait_for_frame = true)
    {
        std::deque<std::pair<std::shared_ptr<ob::FrameSet>, int64_t>> pending;
        uint64_t dropped = 0;
        {
            std::unique_lock<std::mutex> lock(video_queue_mutex_);
            if (wait_for_frame && video_frame_sets_.empty())
                video_queue_cv_.wait_for(lock, std::chrono::milliseconds(20));
            pending.swap(video_frame_sets_);
            dropped = auxiliary_stats_.dropped_video_frame_sets;
        }
        report_dropped_video_frame_sets(dropped);
        for (const auto& [frame_set, arrival_time_local_common_clock_ns] : pending)
            process_frame_set(frame_set, arrival_time_local_common_clock_ns);
    }

    // Overflow drops whole frame sets for as long as the host lags, so warn at most every five seconds.
    void report_dropped_video_frame_sets(uint64_t dropped)
    {
        if (dropped == reported_dropped_video_frame_sets_)
            return;
        const int64_t now = core::os_monotonic_now_ns();
        if (reported_dropped_video_frame_sets_ != 0 && now - last_drop_warning_ns_ < 5'000'000'000)
            return;
        detail::log_warning() << "Ego video callback queue was full; " << dropped - reported_dropped_video_frame_sets_
                              << " more video frame sets lost (" << dropped << " total); recording continues";
        reported_dropped_video_frame_sets_ = dropped;
        last_drop_warning_ns_ = now;
    }

    void process_frame_set(const std::shared_ptr<ob::FrameSet>& frame_set, int64_t arrival_time_local_common_clock_ns)
    {
        if (!frame_set)
            return;
        for (const auto& stream : streams_)
        {
            const auto raw_frame = frame_set->getFrame(to_ob_frame(stream.camera));
            if (!raw_frame)
                continue;
            const auto frame = raw_frame->as<ob::VideoFrame>();
            if (!frame)
                throw std::runtime_error("Ego video frameset contained a non-video frame");

            const auto profile = frame->getStreamProfile()->as<ob::VideoStreamProfile>();
            const auto expected = active_profiles_.at(stream.camera);
            if (!profile || frame->getWidth() != expected->getWidth() || frame->getHeight() != expected->getHeight() ||
                frame->getFormat() != expected->getFormat() || profile->getFps() != expected->getFps())
            {
                throw std::runtime_error(
                    "Ego " + std::string(core::EnumNameOrbbecEgoCameraStream(stream.camera)) +
                    " delivered a frame outside the selected active profile " + std::to_string(expected->getWidth()) +
                    "x" + std::to_string(expected->getHeight()) + "@" + std::to_string(expected->getFps()) + " " +
                    ob::TypeHelper::convertOBFormatTypeToString(expected->getFormat()));
            }
            CapturedFrame captured;
            captured.metadata.stream = stream.camera;
            captured.metadata.sequence_number = frame->getIndex();
            captured.metadata.width = frame->getWidth();
            captured.metadata.height = frame->getHeight();
            captured.metadata.fps = profile->getFps();
            captured.metadata.pixel_format = stream.pixel_format;
            captured.metadata.encoded_bytes = frame->getDataSize();
            captured.metadata.capture_epoch = capture_epoch_;
            captured.encoded_data.assign(frame->getData(), frame->getData() + frame->getDataSize());
            captured.sample_time_raw_device_clock_ns = sdk_timestamp_ns(frame->getTimeStampUs());
            captured.sample_time_local_common_clock_ns = arrival_time_local_common_clock_ns;
            auto& stats = stats_[stream.camera];
            if (stats.epoch_frame_count > 0 && captured.metadata.sequence_number <= stats.last_sequence)
            {
                throw std::runtime_error(
                    "Ego " + std::string(core::EnumNameOrbbecEgoCameraStream(stream.camera)) +
                    " delivered a non-increasing sequence: previous=" + std::to_string(stats.last_sequence) +
                    " current=" + std::to_string(captured.metadata.sequence_number));
            }
            if (stats.epoch_frame_count > 0 && captured.sample_time_raw_device_clock_ns <= stats.last_device_timestamp_ns)
            {
                throw std::runtime_error("Ego " + std::string(core::EnumNameOrbbecEgoCameraStream(stream.camera)) +
                                         " delivered a non-increasing raw device timestamp: previous=" +
                                         std::to_string(stats.last_device_timestamp_ns) +
                                         " current=" + std::to_string(captured.sample_time_raw_device_clock_ns));
            }
            std::function<void(const CapturedFrame&)> accepted_frame;
#if defined(ORBBEC_EGO_ENABLE_PREVIEW)
            if (preview_)
                accepted_frame = [this](const CapturedFrame& accepted) { preview_->submit(accepted); };
#endif
            sink_->on_frame(captured, accepted_frame);

            if (stats.epoch_frame_count > 0 && captured.metadata.sequence_number > stats.last_sequence + 1)
            {
                const uint64_t missing = captured.metadata.sequence_number - stats.last_sequence - 1;
                stats.sequence_gaps += missing;
                detail::log_warning() << "Ego " << core::EnumNameOrbbecEgoCameraStream(stream.camera)
                                      << " sequence gap: previous=" << stats.last_sequence
                                      << " current=" << captured.metadata.sequence_number << " missing=" << missing;
            }
            stats.frame_count++;
            stats.epoch_frame_count++;
            stats.byte_count += captured.encoded_data.size();
            stats.last_sequence = captured.metadata.sequence_number;
            stats.last_device_timestamp_ns = captured.sample_time_raw_device_clock_ns;
            last_video_arrival_ns_[stream.camera] = arrival_time_local_common_clock_ns;
        }
    }

    void print_stats() const
    {
        uint64_t dropped_video_frame_sets = 0;
        AuxiliaryStats auxiliary_stats;
        auxiliary_stats.accel_samples = auxiliary_stats_.accel_samples;
        auxiliary_stats.gyro_samples = auxiliary_stats_.gyro_samples;
        {
            std::lock_guard<std::mutex> lock(video_queue_mutex_);
            dropped_video_frame_sets = auxiliary_stats_.dropped_video_frame_sets;
        }
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            auxiliary_stats.publish_queue_peak = auxiliary_stats_.publish_queue_peak;
            auxiliary_stats.dropped_events = auxiliary_stats_.dropped_events;
        }
        for (const auto& [stream, stats] : stats_)
        {
            detail::log_info()
                << "  " << core::EnumNameOrbbecEgoCameraStream(stream) << ": " << stats.frame_count << " frames, "
                << stats.byte_count << " bytes, " << stats.sequence_gaps << " sequence gaps" << std::endl;
        }
        detail::log_info()
            << "  IMU: accel=" << auxiliary_stats.accel_samples << " gyro=" << auxiliary_stats.gyro_samples
            << " samples; queue_peak=" << auxiliary_stats.publish_queue_peak
            << " dropped=" << auxiliary_stats.dropped_events << " video_frame_sets_dropped=" << dropped_video_frame_sets
            << "; capture_epoch=" << auxiliary_stats_.capture_epoch
            << " reconnect_attempts=" << auxiliary_stats_.reconnect_attempts
            << " successful_reconnects=" << auxiliary_stats_.successful_reconnects << std::endl;
    }

    const std::map<core::OrbbecEgoCameraStream, StreamStats>& stats() const
    {
        return stats_;
    }

    const AuxiliaryStats& auxiliary_stats() const
    {
        return auxiliary_stats_;
    }
    bool preview_closed() const
    {
#if defined(ORBBEC_EGO_ENABLE_PREVIEW)
        return preview_ && preview_->closed();
#else
        return false;
#endif
    }

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

    void set_async_error(const std::string& message)
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (async_error_.empty())
            async_error_ = message;
    }

    void drain_events()
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

    struct PendingImu
    {
        std::vector<core::OrbbecEgoImuSample> samples;
        uint64_t sequence = 0;
        int64_t first_local_ns = 0;
        float latest_temperature_c = 0.0f;
    };

    void add_imu_sample(core::OrbbecEgoImuSensor sensor, const OBFloat3D& value, float temperature, uint64_t timestamp_us)
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

    void flush_imu(core::OrbbecEgoImuSensor sensor)
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

    void start_imu()
    {
        check_cancelled();
        auto imu_config = std::make_shared<ob::Config>();
        imu_config->enableAccelStream(accel_scale(config_.accel_full_scale_g), imu_rate(config_.imu_rate));
        imu_config->enableGyroStream(gyro_scale(config_.gyro_full_scale_dps), imu_rate(config_.imu_rate));
        imu_pipeline_ = std::make_unique<ob::Pipeline>(device_);
        imu_pipeline_->start(
            imu_config,
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
                            add_imu_sample(core::OrbbecEgoImuSensor_Accel, frame->getValue(), frame->getTemperature(),
                                           frame->getTimeStampUs());
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
                            add_imu_sample(core::OrbbecEgoImuSensor_Gyro, frame->getValue(), frame->getTemperature(),
                                           frame->getTimeStampUs());
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

    void wait_for_auxiliary_readiness()
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

    void wait_for_active_auxiliary_readiness()
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

    static std::shared_ptr<core::OrbbecEgoCameraIntrinsicsT> camera_intrinsics(const OBCalibrationParam& param,
                                                                               OBSensorType sensor)
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

    static std::shared_ptr<core::OrbbecEgoExtrinsicsT> extrinsics(const OBCalibrationParam& param,
                                                                  OBSensorType from,
                                                                  OBSensorType to)
    {
        const auto& source = param.extrinsics[from][to];
        auto result = std::make_shared<core::OrbbecEgoExtrinsicsT>();
        result->rotation.assign(std::begin(source.rot), std::end(source.rot));
        result->translation_mm.assign(std::begin(source.trans), std::end(source.trans));
        return result;
    }

    void publish_calibration(const std::shared_ptr<ob::Config>& pipeline_config)
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

    void start_device_state()
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

    void publish_periodic_device_state(const std::string& final_error = {}, bool direct = false)
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

private:
    CaptureConfig config_;
    std::unique_ptr<ob::Context> context_;
    std::shared_ptr<EgoSdkDevice> device_;
    std::unique_ptr<ob::Pipeline> pipeline_;
    std::unique_ptr<ob::Pipeline> imu_pipeline_;
    std::vector<StreamConfig> streams_;
    ActiveProfiles active_profiles_;
    std::unique_ptr<FrameSink> sink_;
#if defined(ORBBEC_EGO_ENABLE_PREVIEW)
    std::unique_ptr<Preview> preview_;
#endif
    std::map<core::OrbbecEgoCameraStream, StreamStats> stats_;
    std::map<core::OrbbecEgoCameraStream, int64_t> last_video_arrival_ns_;
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

EgoCamera::EgoCamera(const CaptureConfig& config, const std::vector<StreamConfig>& streams, std::unique_ptr<FrameSink> sink)
    : impl_(std::make_unique<Impl>(config, streams, std::move(sink)))
{
}

EgoCamera::~EgoCamera() = default;

void EgoCamera::update()
{
    impl_->update();
}

void EgoCamera::close()
{
    impl_->close();
}

void EgoCamera::print_stats() const
{
    impl_->print_stats();
}

const std::map<core::OrbbecEgoCameraStream, StreamStats>& EgoCamera::stats() const
{
    return impl_->stats();
}

const AuxiliaryStats& EgoCamera::auxiliary_stats() const
{
    return impl_->auxiliary_stats();
}

bool EgoCamera::preview_closed() const
{
    return impl_->preview_closed();
}

void EgoCamera::list_capabilities(const CaptureConfig& config)
{
    ob::Context context(prepare_sdk_logging());
#if defined(__linux__) || defined(__ANDROID__)
    context.setUvcBackendType(OB_UVC_BACKEND_TYPE_LIBUVC);
#endif
    const auto devices = context.queryDeviceList();
    for (uint32_t index = 0; index < devices->getCount(); ++index)
    {
        const auto device = devices->getDevice(index);
        if (config.device_uid.empty() || device->getDeviceInfo()->getUid() == config.device_uid)
        {
            print_capabilities(device);
            return;
        }
    }
    throw std::runtime_error("No matching Ego device found");
}

} // namespace plugins::ego
