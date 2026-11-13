// SPDX-FileCopyrightText: Copyright (c) 2026 ORBBEC INC. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "device_controls.hpp"
#include "diagnostics.hpp"
#include "ego_camera_impl.hpp"
#include "inc/ego_camera/cancellation.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <utility>

namespace plugins::ego
{

using detail::describe_profiles;
using detail::FatalReconnectError;
using detail::make_video_config;
using detail::to_ob_frame;

void EgoCamera::Impl::reapply_controls(std::string_view operation)
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

void EgoCamera::Impl::prepare_controls(const std::vector<PropertySetting>& settings)
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
        expected_profiles.push_back({ to_ob_frame(stream), profile->getWidth(), profile->getHeight(), profile->getFps(),
                                      profile->getFormat(), core::EnumNameOrbbecEgoCameraStream(stream) });
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

void EgoCamera::Impl::verify_active_controls()
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

void EgoCamera::Impl::restore_properties()
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

} // namespace plugins::ego
