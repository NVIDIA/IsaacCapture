// SPDX-FileCopyrightText: Copyright (c) 2026 ORBBEC INC. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "diagnostics.hpp"
#include "ego_camera_impl.hpp"

#include <oxr_utils/os_time.hpp>

#include <algorithm>
#include <iostream>
#include <thread>
#include <utility>

namespace plugins::ego
{

using detail::adopt_ego_device;
using detail::describe_profiles;
using detail::FatalReconnectError;
using detail::has_sensor;
using detail::make_video_config;
using detail::RecoverableCaptureError;
using detail::select_profile;
using detail::to_ob_sensor;

bool EgoCamera::Impl::device_list_contains_uid(const std::shared_ptr<ob::DeviceList>& devices, const std::string& uid)
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

bool EgoCamera::Impl::is_selected_physical_device(const std::shared_ptr<ob::DeviceInfo>& info) const
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

std::shared_ptr<ob::Device> EgoCamera::Impl::find_selected_device()
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

void EgoCamera::Impl::reset_stream_readiness()
{
    accel_ready_.store(false, std::memory_order_release);
    gyro_ready_.store(false, std::memory_order_release);
    active_accel_ready_.store(false, std::memory_order_release);
    active_gyro_ready_.store(false, std::memory_order_release);
    last_accel_arrival_ns_.store(0, std::memory_order_release);
    last_gyro_arrival_ns_.store(0, std::memory_order_release);
    video_liveness_.reset();
}

void EgoCamera::Impl::stop_capture_for_reconnect(bool device_removal_confirmed)
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
            detail::log_info() << "Ego device-state callback already inactive after confirmed removal: " << error.what();
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
        cleanup_error_ += std::string("Ego device-state callback clear during reconnect failed: ") + error.what() + "; ";
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

void EgoCamera::Impl::begin_recovery(const std::string& reason)
{
    publish_capture_status(
        device_removed_.load(std::memory_order_acquire) ? CaptureAvailability::Absent : CaptureAvailability::Recovering,
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

void EgoCamera::Impl::wait_for_recovery_stability() const
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

void EgoCamera::Impl::attempt_reconnect()
{
    check_cancelled();
    const auto now = std::chrono::steady_clock::now();
    if (now >= recovery_deadline_)
    {
        std::string reason = "Ego device serial=" + selected_device_serial_ + " last_uid=" + selected_device_uid_ +
                             " did not recover within " + std::to_string(config_.reconnect_timeout_seconds) + " seconds";
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
                << "Ego device serial=" << selected_device_serial_ << " re-enumerated from uid=" << selected_device_uid_
                << " to uid=" << recovered_uid << std::endl;
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

        reset_stream_readiness();
        accepting_callbacks_.store(true, std::memory_order_release);
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
        detail::log_info()
            << "Ego device " << selected_device_uid_ << " recovered as capture epoch " << capture_epoch_ << std::endl;
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

void EgoCamera::Impl::reacquire_device()
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

} // namespace plugins::ego
