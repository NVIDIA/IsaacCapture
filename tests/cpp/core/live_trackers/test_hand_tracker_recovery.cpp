// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <deviceio_session/deviceio_session.hpp>
#include <deviceio_trackers/hand_tracker.hpp>
#include <oxr_utils/oxr_session_handles.hpp>

#include <XR_MNDX_xdev_space.h>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <time.h>

namespace
{

class FakeHandRuntime
{
public:
    FakeHandRuntime()
    {
        if (current_ != nullptr)
        {
            throw std::logic_error("FakeHandRuntime: one instance at a time");
        }
        current_ = this;
    }

    ~FakeHandRuntime()
    {
        current_ = nullptr;
    }

    FakeHandRuntime(const FakeHandRuntime&) = delete;
    FakeHandRuntime& operator=(const FakeHandRuntime&) = delete;

    core::OpenXRSessionHandles handles() const
    {
        core::OpenXRSessionHandles handles;
        handles.instance = reinterpret_cast<XrInstance>(1);
        handles.session = reinterpret_cast<XrSession>(1);
        handles.space = reinterpret_cast<XrSpace>(1);
        handles.xrGetInstanceProcAddr = &get_instance_proc_addr;
        return handles;
    }

    int left_xdev_create_attempts() const
    {
        return left_xdev_create_attempts_;
    }

private:
    static XrResult XRAPI_CALL get_system(XrInstance, const XrSystemGetInfo*, XrSystemId* system_id)
    {
        *system_id = 1;
        return XR_SUCCESS;
    }

    static XrResult XRAPI_CALL get_system_properties(XrInstance, XrSystemId, XrSystemProperties* properties)
    {
        auto* hand_tracking = static_cast<XrSystemHandTrackingPropertiesEXT*>(properties->next);
        hand_tracking->supportsHandTracking = XR_TRUE;
        return XR_SUCCESS;
    }

    static XrResult XRAPI_CALL create_reference_space(XrSession, const XrReferenceSpaceCreateInfo*, XrSpace* space)
    {
        *space = reinterpret_cast<XrSpace>(2);
        return XR_SUCCESS;
    }

    static XrResult XRAPI_CALL destroy_space(XrSpace)
    {
        return XR_SUCCESS;
    }

    static XrResult XRAPI_CALL locate_space(XrSpace, XrSpace, XrTime, XrSpaceLocation* location)
    {
        location->locationFlags = 0;
        return XR_SUCCESS;
    }

    static XrResult XRAPI_CALL create_xdev_list(XrSession, const XrCreateXDevListInfoMNDX*, XrXDevListMNDX* list)
    {
        *list = reinterpret_cast<XrXDevListMNDX>(1);
        return XR_SUCCESS;
    }

    static XrResult XRAPI_CALL destroy_xdev_list(XrXDevListMNDX)
    {
        return XR_SUCCESS;
    }

    static XrResult XRAPI_CALL enumerate_xdevs(XrXDevListMNDX, uint32_t capacity, uint32_t* count, XrXDevIdMNDX* xdevs)
    {
        *count = 1;
        if (capacity != 0)
        {
            xdevs[0] = 42;
        }
        return XR_SUCCESS;
    }

    static XrResult XRAPI_CALL get_xdev_properties(XrXDevListMNDX,
                                                   const XrGetXDevInfoMNDX*,
                                                   XrXDevPropertiesMNDX* properties)
    {
        std::strncpy(properties->name, "Avatar glove", sizeof(properties->name) - 1);
        properties->canCreateSpace = XR_TRUE;
        return XR_SUCCESS;
    }

    static XrResult XRAPI_CALL create_hand_tracker(XrSession,
                                                   const XrHandTrackerCreateInfoEXT* create_info,
                                                   XrHandTrackerEXT* tracker)
    {
        const auto* xdev_info = static_cast<const XrCreateHandTrackerXDevMNDX*>(create_info->next);
        if (xdev_info != nullptr && create_info->hand == XR_HAND_LEFT_EXT)
        {
            ++current_->left_xdev_create_attempts_;
            if (current_->fail_first_left_xdev_create_)
            {
                current_->fail_first_left_xdev_create_ = false;
                return XR_ERROR_RUNTIME_FAILURE;
            }
        }

        *tracker = reinterpret_cast<XrHandTrackerEXT>(current_->next_tracker_handle_++);
        return XR_SUCCESS;
    }

    static XrResult XRAPI_CALL destroy_hand_tracker(XrHandTrackerEXT)
    {
        return XR_SUCCESS;
    }

    static XrResult XRAPI_CALL locate_hand_joints(XrHandTrackerEXT,
                                                  const XrHandJointsLocateInfoEXT*,
                                                  XrHandJointLocationsEXT* locations)
    {
        locations->isActive = XR_FALSE;
        return XR_SUCCESS;
    }

    static XrResult XRAPI_CALL timespec_to_time(XrInstance, const struct timespec* timespec, XrTime* time)
    {
        *time = static_cast<XrTime>(timespec->tv_sec) * 1'000'000'000 + timespec->tv_nsec;
        return XR_SUCCESS;
    }

    static XrResult XRAPI_CALL time_to_timespec(XrInstance, XrTime time, struct timespec* timespec)
    {
        timespec->tv_sec = static_cast<time_t>(time / 1'000'000'000);
        timespec->tv_nsec = static_cast<long>(time % 1'000'000'000);
        return XR_SUCCESS;
    }

    static XrResult XRAPI_CALL get_instance_proc_addr(XrInstance, const char* name, PFN_xrVoidFunction* function)
    {
        static const std::map<std::string_view, PFN_xrVoidFunction> functions = {
            { "xrGetSystem", reinterpret_cast<PFN_xrVoidFunction>(&get_system) },
            { "xrGetSystemProperties", reinterpret_cast<PFN_xrVoidFunction>(&get_system_properties) },
            { "xrCreateReferenceSpace", reinterpret_cast<PFN_xrVoidFunction>(&create_reference_space) },
            { "xrDestroySpace", reinterpret_cast<PFN_xrVoidFunction>(&destroy_space) },
            { "xrLocateSpace", reinterpret_cast<PFN_xrVoidFunction>(&locate_space) },
            { "xrCreateXDevListMNDX", reinterpret_cast<PFN_xrVoidFunction>(&create_xdev_list) },
            { "xrDestroyXDevListMNDX", reinterpret_cast<PFN_xrVoidFunction>(&destroy_xdev_list) },
            { "xrEnumerateXDevsMNDX", reinterpret_cast<PFN_xrVoidFunction>(&enumerate_xdevs) },
            { "xrGetXDevPropertiesMNDX", reinterpret_cast<PFN_xrVoidFunction>(&get_xdev_properties) },
            { "xrCreateHandTrackerEXT", reinterpret_cast<PFN_xrVoidFunction>(&create_hand_tracker) },
            { "xrDestroyHandTrackerEXT", reinterpret_cast<PFN_xrVoidFunction>(&destroy_hand_tracker) },
            { "xrLocateHandJointsEXT", reinterpret_cast<PFN_xrVoidFunction>(&locate_hand_joints) },
            { "xrConvertTimespecTimeToTimeKHR", reinterpret_cast<PFN_xrVoidFunction>(&timespec_to_time) },
            { "xrConvertTimeToTimespecTimeKHR", reinterpret_cast<PFN_xrVoidFunction>(&time_to_timespec) },
        };
        const auto it = functions.find(name);
        *function = it == functions.end() ? nullptr : it->second;
        return it == functions.end() ? XR_ERROR_FUNCTION_UNSUPPORTED : XR_SUCCESS;
    }

    inline static FakeHandRuntime* current_ = nullptr;

    uintptr_t next_tracker_handle_ = 1;
    int left_xdev_create_attempts_ = 0;
    bool fail_first_left_xdev_create_ = true;
};

} // anonymous namespace

TEST_CASE("HandTracker retries an incomplete XDev binding without a topology change", "[unit][hand][xdev]")
{
    FakeHandRuntime runtime;
    auto hand = std::make_shared<core::HandTracker>();
    auto session = core::DeviceIOSession::run({ hand }, runtime.handles());

    REQUIRE(runtime.left_xdev_create_attempts() == 1);

    session->update();

    CHECK(runtime.left_xdev_create_attempts() == 2);
}
