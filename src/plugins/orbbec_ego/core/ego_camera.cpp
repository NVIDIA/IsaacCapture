// SPDX-FileCopyrightText: Copyright (c) 2026 ORBBEC INC. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#if defined(ORBBEC_EGO_ENABLE_PREVIEW)
#    include "preview.hpp"
#endif

#include "device_controls.hpp"
#include "diagnostics.hpp"
#include "ego_camera_impl.hpp"
#include "inc/ego_camera/cancellation.hpp"

#include <oxr_utils/os_time.hpp>

#include <iostream>
#include <utility>

namespace plugins::ego
{

using detail::adopt_ego_device;
using detail::describe_profiles;
using detail::has_sensor;
using detail::make_video_config;
using detail::RecoverableCaptureError;
using detail::select_profile;
using detail::to_ob_sensor;

namespace
{

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

} // namespace

namespace detail
{

std::shared_ptr<EgoSdkDevice> adopt_ego_device(std::shared_ptr<ob::Device> device)
{
#if __has_include(<libobsensor/hpp/DataAcquisitionDevice.hpp>)
    return EgoSdkDevice::fromDevice(std::move(device));
#else
    return device;
#endif
}

} // namespace detail

EgoCamera::Impl::Impl(const CaptureConfig& config, const std::vector<StreamConfig>& streams, std::unique_ptr<FrameSink> sink)
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
            supports_requested_streams = supports_requested_streams && has_sensor(candidate, to_ob_sensor(stream.camera));
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
    const auto readiness = prepare_video_startup_readiness();
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
        start_initial_video(pipeline_config, readiness);
        wait_for_video_readiness(readiness);
        // SDK stream startup can reset image controls; reapply before capture begins.
        reapply_controls("startup");
        verify_active_controls();
        capture_epoch_started_ns_ = core::os_monotonic_now_ns();
        capture_active_.store(true, std::memory_order_release);
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

EgoCamera::Impl::~Impl()
{
    shutdown_noexcept(false);
}

bool EgoCamera::Impl::is_removed_device_cleanup_error(const ob::Error& error, bool device_removal_confirmed) noexcept
{
    // SDK 2.9 reports deactivated components as call-sequence errors. Only
    // the removed capture instance may tolerate this during reconnect teardown.
    return device_removal_confirmed && error.getStatus() == OB_ERROR_WRONG_API_CALL_SEQUENCE &&
           error.getExceptionType() == OB_EXCEPTION_TYPE_WRONG_API_CALL_SEQUENCE && error.what() &&
           std::string_view(error.what()).find("Device is deactivated/disconnected! DeviceComponentId:") == 0;
}

std::string EgoCamera::Impl::stop_pipeline_noexcept(std::unique_ptr<ob::Pipeline>& pipeline,
                                                    bool& started,
                                                    const char* name,
                                                    bool device_removal_confirmed) noexcept
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
        << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - stop_started).count() << " ms";
    return failure;
}

void EgoCamera::Impl::check_cancelled() const
{
    check_capture_cancelled(config_.stop_requested);
}

void EgoCamera::Impl::close()
{
    shutdown(true);
}

void EgoCamera::Impl::shutdown(bool promote_metadata)
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
            remember_error("Ego capture dropped " + std::to_string(auxiliary_stats_.dropped_events) + " metadata events");
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

void EgoCamera::Impl::shutdown_noexcept(bool promote_metadata) noexcept
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

void EgoCamera::Impl::update()
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

void EgoCamera::Impl::update_capture()
{
    if (recovering_)
    {
        publish_capture_status(
            recovery_device_present_ ? CaptureAvailability::Recovering : CaptureAvailability::Absent, recovery_reason_);
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

// orbbec_ego.sh verify and test_ego_script.sh parse these lines; change their patterns and fixtures together.
void EgoCamera::Impl::print_stats() const
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

const std::map<core::OrbbecEgoCameraStream, StreamStats>& EgoCamera::Impl::stats() const
{
    return stats_;
}

const AuxiliaryStats& EgoCamera::Impl::auxiliary_stats() const
{
    return auxiliary_stats_;
}

bool EgoCamera::Impl::preview_closed() const
{
#if defined(ORBBEC_EGO_ENABLE_PREVIEW)
    return preview_ && preview_->closed();
#else
    return false;
#endif
}

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
