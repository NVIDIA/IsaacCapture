// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Hand tracking data injection via push devices

#include "inc/plugin_utils/hand_injector.hpp"

#include <oxr_utils/oxr_funcs.hpp>

#include <cstring>
#include <stdexcept>
#include <string>

namespace plugin_utils
{

namespace
{
void CheckXrResult(XrResult result, const char* message)
{
    if (XR_FAILED(result))
    {
        throw std::runtime_error(std::string(message) + " failed with XrResult: " + std::to_string(result));
    }
}
} // namespace

std::vector<std::string> HandInjector::get_required_extensions()
{
    std::vector<std::string> extensions = { XR_NVX1_DEVICE_INTERFACE_BASE_EXTENSION_NAME };
    for (const auto& ext : core::XrTimeConverter::get_required_extensions())
    {
        extensions.push_back(ext);
    }
    return extensions;
}

HandInjector::HandInjector(XrInstance instance, XrSession session, XrHandEXT hand, XrSpace base_space)
    : time_converter_(core::OpenXRSessionHandles{ instance, session, base_space, ::xrGetInstanceProcAddr })
{
    try
    {
        load_functions(instance);
        create_device(session, hand, base_space);
    }
    catch (...)
    {
        cleanup();
        throw;
    }
}

HandInjector::~HandInjector()
{
    cleanup();
}

void HandInjector::load_functions(XrInstance instance)
{
    core::loadExtensionFunction(
        instance, xrGetInstanceProcAddr, "xrCreatePushDeviceNV", reinterpret_cast<PFN_xrVoidFunction*>(&pfn_create_));
    core::loadExtensionFunction(
        instance, xrGetInstanceProcAddr, "xrDestroyPushDeviceNV", reinterpret_cast<PFN_xrVoidFunction*>(&pfn_destroy_));
    core::loadExtensionFunction(instance, xrGetInstanceProcAddr, "xrPushDevicePushHandTrackingNV",
                                reinterpret_cast<PFN_xrVoidFunction*>(&pfn_push_));
}

void HandInjector::create_device(XrSession session, XrHandEXT hand, XrSpace base_space)
{
    XrPushDeviceHandTrackingInfoNV hand_info{ XR_TYPE_PUSH_DEVICE_HAND_TRACKING_INFO_NV };
    hand_info.hand = hand;
    hand_info.jointSet = XR_HAND_JOINT_SET_DEFAULT_EXT;

    XrPushDeviceCreateInfoNV create_info{ XR_TYPE_PUSH_DEVICE_CREATE_INFO_NV };
    create_info.next = &hand_info;
    create_info.baseSpace = base_space;
    create_info.deviceTypeUuidValid = XR_FALSE;
    create_info.deviceUuidValid = XR_FALSE;
    strncpy(create_info.localizedName, hand == XR_HAND_LEFT_EXT ? "Left Hand" : "Right Hand",
            sizeof(create_info.localizedName) - 1);
    create_info.localizedName[sizeof(create_info.localizedName) - 1] = '\0';
    strncpy(create_info.serial, hand == XR_HAND_LEFT_EXT ? "LEFT" : "RIGHT", sizeof(create_info.serial) - 1);
    create_info.serial[sizeof(create_info.serial) - 1] = '\0';

    CheckXrResult(pfn_create_(session, &create_info, nullptr, &device_),
                  (std::string("xrCreatePushDeviceNV(") + (hand == XR_HAND_LEFT_EXT ? "left" : "right") + ")").c_str());
}

void HandInjector::push(const XrHandJointLocationEXT* joints, XrTime timestamp)
{
    if (device_ == XR_NULL_HANDLE)
    {
        throw std::runtime_error("HandInjector: push device not initialized");
    }

    XrPushDeviceHandTrackingDataNV data{ XR_TYPE_PUSH_DEVICE_HAND_TRACKING_DATA_NV };
    data.timestamp = timestamp;
    data.jointCount = XR_HAND_JOINT_COUNT_EXT;
    data.jointLocations = joints;

    CheckXrResult(pfn_push_(device_, &data), "xrPushDevicePushHandTrackingNV");
}

void HandInjector::cleanup()
{
    if (pfn_destroy_ && device_ != XR_NULL_HANDLE)
    {
        // Best-effort: signal inactive before destroying so readers see is_active=false.
        // If the clock fails we skip the push rather than sending a bogus timestamp;
        // the subsequent device destruction still tears down the push device cleanly.
        if (pfn_push_)
        {
            try
            {
                XrTime time = time_converter_.os_monotonic_now();
                XrHandJointLocationEXT dummy{};
                XrPushDeviceHandTrackingDataNV data{ XR_TYPE_PUSH_DEVICE_HAND_TRACKING_DATA_NV };
                data.timestamp = time;
                data.jointCount = 0;
                data.jointLocations = &dummy;
                pfn_push_(device_, &data);
            }
            catch (const std::exception& e)
            {
                logger_->warn("cleanup: skipping inactive push, clock failed: {}", e.what());
            }
        }
        pfn_destroy_(device_);
        device_ = XR_NULL_HANDLE;
    }
}

} // namespace plugin_utils
