// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Hand tracking data injection via push devices
#pragma once

#include <log_bridge/logger.hpp>
#include <openxr/openxr.h>
#include <oxr_utils/oxr_session_handles.hpp>
#include <oxr_utils/oxr_time.hpp>

#include <XR_NVX1_device_interface.h>
#include <string>
#include <vector>

namespace plugin_utils
{

/**
 * @brief Manages a single-hand push device for injecting hand tracking data.
 *
 * Create one instance per hand. Destroying the injector automatically signals
 * inactivity to the runtime before releasing the push device, so readers see
 * `isActive = false` rather than stale data.
 */
class HandInjector
{
public:
    HandInjector(XrInstance instance, XrSession session, XrHandEXT hand, XrSpace base_space);

    ~HandInjector();

    HandInjector(const HandInjector&) = delete;
    HandInjector& operator=(const HandInjector&) = delete;

    HandInjector(HandInjector&&) = delete;
    HandInjector& operator=(HandInjector&&) = delete;

    /** @brief OpenXR extensions the session must enable for push devices. */
    static std::vector<std::string> get_required_extensions();

    /** @brief Push a full set of joint locations. */
    void push(const XrHandJointLocationEXT* joints, XrTime timestamp);

private:
    void load_functions(XrInstance instance);
    void create_device(XrSession session, XrHandEXT hand, XrSpace base_space);
    void cleanup();

    XrPushDeviceNV device_ = XR_NULL_HANDLE;

    PFN_xrCreatePushDeviceNV pfn_create_ = nullptr;
    PFN_xrDestroyPushDeviceNV pfn_destroy_ = nullptr;
    PFN_xrPushDevicePushHandTrackingNV pfn_push_ = nullptr;

    core::XrTimeConverter time_converter_;

    std::shared_ptr<spdlog::logger> logger_ = isaaccapture::Logger::get("isaaccapture.plugins.plugin_utils.HandInjector");
};

} // namespace plugin_utils
