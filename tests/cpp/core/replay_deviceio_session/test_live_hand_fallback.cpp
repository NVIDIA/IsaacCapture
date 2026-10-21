// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "live_hand_tracker_impl.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstring>
#include <map>
#include <utility>
#include <vector>

namespace
{
struct Tracker
{
    XrXDevIdMNDX id;
    XrHandEXT hand;
};

struct Runtime
{
    std::map<XrXDevIdMNDX, const char*> devices;
    std::vector<std::pair<XrXDevIdMNDX, XrHandEXT>> created;
    std::map<std::pair<XrXDevIdMNDX, XrHandEXT>, bool> active;
    bool default_creation_fails = false;
    size_t live_trackers = 0;
    bool list_alive = false;
};
Runtime* runtime;

XRAPI_ATTR XrResult XRAPI_CALL get_system(XrInstance, const XrSystemGetInfo*, XrSystemId* id)
{
    *id = 1;
    return XR_SUCCESS;
}
XRAPI_ATTR XrResult XRAPI_CALL get_properties(XrInstance, XrSystemId, XrSystemProperties* properties)
{
    static_cast<XrSystemHandTrackingPropertiesEXT*>(properties->next)->supportsHandTracking = XR_TRUE;
    return XR_SUCCESS;
}
XRAPI_ATTR XrResult XRAPI_CALL create_list(XrSession, const XrCreateXDevListInfoMNDX*, XrXDevListMNDX* list)
{
    runtime->list_alive = true;
    *list = reinterpret_cast<XrXDevListMNDX>(runtime);
    return XR_SUCCESS;
}
XRAPI_ATTR XrResult XRAPI_CALL destroy_list(XrXDevListMNDX)
{
    CHECK(runtime->live_trackers == 0);
    runtime->list_alive = false;
    return XR_SUCCESS;
}
XRAPI_ATTR XrResult XRAPI_CALL enumerate_devices(XrXDevListMNDX, uint32_t capacity, uint32_t* count, XrXDevIdMNDX* ids)
{
    *count = static_cast<uint32_t>(runtime->devices.size());
    if (capacity)
    {
        size_t i = 0;
        for (const auto& [id, name] : runtime->devices)
        {
            ids[i++] = id;
        }
    }
    return XR_SUCCESS;
}
XRAPI_ATTR XrResult XRAPI_CALL device_properties(XrXDevListMNDX,
                                                 const XrGetXDevInfoMNDX* info,
                                                 XrXDevPropertiesMNDX* properties)
{
    std::strncpy(properties->name, runtime->devices.at(info->id), sizeof(properties->name) - 1);
    return XR_SUCCESS;
}
XRAPI_ATTR XrResult XRAPI_CALL create_tracker(XrSession, const XrHandTrackerCreateInfoEXT* info, XrHandTrackerEXT* handle)
{
    const auto* device = static_cast<const XrCreateHandTrackerXDevMNDX*>(info->next);
    if (!device && runtime->default_creation_fails)
    {
        return XR_ERROR_FEATURE_UNSUPPORTED;
    }
    if (device)
    {
        REQUIRE(runtime->list_alive);
    }
    auto* tracker = new Tracker{ device ? device->id : 0, info->hand };
    runtime->created.emplace_back(tracker->id, tracker->hand);
    ++runtime->live_trackers;
    *handle = reinterpret_cast<XrHandTrackerEXT>(tracker);
    return XR_SUCCESS;
}
XRAPI_ATTR XrResult XRAPI_CALL destroy_tracker(XrHandTrackerEXT handle)
{
    --runtime->live_trackers;
    delete reinterpret_cast<Tracker*>(handle);
    return XR_SUCCESS;
}
XRAPI_ATTR XrResult XRAPI_CALL locate_joints(XrHandTrackerEXT handle,
                                             const XrHandJointsLocateInfoEXT*,
                                             XrHandJointLocationsEXT* locations)
{
    const auto& tracker = *reinterpret_cast<Tracker*>(handle);
    locations->isActive = runtime->active[{ tracker.id, tracker.hand }];
    for (uint32_t i = 0; i < locations->jointCount; ++i)
    {
        locations->jointLocations[i] = {};
        locations->jointLocations[i].pose.orientation.w = 1;
        locations->jointLocations[i].pose.position.x = static_cast<float>(tracker.id);
        locations->jointLocations[i].locationFlags =
            XR_SPACE_LOCATION_POSITION_VALID_BIT | XR_SPACE_LOCATION_ORIENTATION_VALID_BIT;
    }
    return XR_SUCCESS;
}
#if defined(XR_USE_TIMESPEC)
XRAPI_ATTR XrResult XRAPI_CALL convert_time(XrInstance, const timespec* ts, XrTime* time)
{
    *time = ts->tv_sec * 1000000000LL + ts->tv_nsec;
    return XR_SUCCESS;
}
XRAPI_ATTR XrResult XRAPI_CALL convert_from_time(XrInstance, XrTime time, timespec* ts)
{
    ts->tv_sec = time / 1000000000LL;
    ts->tv_nsec = time % 1000000000LL;
    return XR_SUCCESS;
}
#elif defined(XR_USE_PLATFORM_WIN32)
XRAPI_ATTR XrResult XRAPI_CALL convert_time(XrInstance, const LARGE_INTEGER* counter, XrTime* time)
{
    *time = counter->QuadPart;
    return XR_SUCCESS;
}
XRAPI_ATTR XrResult XRAPI_CALL convert_from_time(XrInstance, XrTime time, LARGE_INTEGER* counter)
{
    counter->QuadPart = time;
    return XR_SUCCESS;
}
#endif
XRAPI_ATTR XrResult XRAPI_CALL unused_core_function()
{
    return XR_SUCCESS;
}
XRAPI_ATTR XrResult XRAPI_CALL get_proc(XrInstance, const char* name, PFN_xrVoidFunction* result)
{
#define PROVIDE(NAME, FUNCTION)                                                                                        \
    if (std::strcmp(name, NAME) == 0)                                                                                  \
    {                                                                                                                  \
        *result = reinterpret_cast<PFN_xrVoidFunction>(&FUNCTION);                                                     \
        return XR_SUCCESS;                                                                                             \
    }
    PROVIDE("xrGetSystem", get_system)
    PROVIDE("xrGetSystemProperties", get_properties)
    PROVIDE("xrCreateReferenceSpace", unused_core_function)
    PROVIDE("xrDestroySpace", unused_core_function)
    PROVIDE("xrLocateSpace", unused_core_function)
    PROVIDE("xrCreateXDevListMNDX", create_list)
    PROVIDE("xrDestroyXDevListMNDX", destroy_list)
    PROVIDE("xrEnumerateXDevsMNDX", enumerate_devices)
    PROVIDE("xrGetXDevPropertiesMNDX", device_properties)
    PROVIDE("xrCreateHandTrackerEXT", create_tracker)
    PROVIDE("xrDestroyHandTrackerEXT", destroy_tracker)
    PROVIDE("xrLocateHandJointsEXT", locate_joints)
#if defined(XR_USE_TIMESPEC)
    PROVIDE("xrConvertTimespecTimeToTimeKHR", convert_time)
    PROVIDE("xrConvertTimeToTimespecTimeKHR", convert_from_time)
#elif defined(XR_USE_PLATFORM_WIN32)
    PROVIDE("xrConvertWin32PerformanceCounterToTimeKHR", convert_time)
    PROVIDE("xrConvertTimeToWin32PerformanceCounterKHR", convert_from_time)
#endif
#undef PROVIDE
    *result = nullptr;
    return XR_ERROR_FUNCTION_UNSUPPORTED;
}
} // namespace

TEST_CASE("Live hands defer unknown XDevs until active data is absent", "[deviceio][xdev][hand]")
{
    Runtime state;
    runtime = &state;
    state.devices = { { 1, "Push Hand Tracker Left" },
                      { 2, "Push Hand Tracker Right" },
                      { 3, "Head Device" },
                      { 4, "Hand Tracking Left" },
                      { 5, "Vendor Right" },
                      { 6, "Other Left" } };
    state.active[{ 1, XR_HAND_LEFT_EXT }] = true;
    state.active[{ 2, XR_HAND_RIGHT_EXT }] = true;
    {
        core::LiveHandTrackerImpl hands({ {}, {}, {}, get_proc }, nullptr);
        CHECK(state.created.size() == 6); // two explicit, two display, two defaults
        hands.update(1000000000);
        CHECK(state.created.size() == 6);
        REQUIRE(hands.get_left_hand().get());

        state.active[{ 1, XR_HAND_LEFT_EXT }] = false;
        // A created, inactive display/default must not mask the unknown real source.
        state.active[{ 6, XR_HAND_LEFT_EXT }] = true;
        hands.update(2000000000);
        CHECK(state.created.size() == 8); // both unknown left candidates, not just first
        REQUIRE(hands.get_left_hand().get());
        CHECK(hands.get_left_hand().get()->joints()->poses()->Get(0)->pose().position().x() == 6);
        REQUIRE(hands.get_right_hand().get());
        hands.update(3000000000);
        CHECK(state.created.size() == 8); // probing is bounded, not per-frame

        state.active[{ 6, XR_HAND_LEFT_EXT }] = false;
        state.active[{ 2, XR_HAND_RIGHT_EXT }] = false;
        state.active[{ 5, XR_HAND_RIGHT_EXT }] = true;
        hands.update(4000000000);
        CHECK_FALSE(hands.get_left_hand().get());
        REQUIRE(hands.get_right_hand().get());
        CHECK(state.created.size() == 9); // the other hand has its own deferred fallback
        hands.update(5000000000);
        CHECK(state.created.size() == 9);
        state.active[{ 6, XR_HAND_LEFT_EXT }] = true;
        hands.update(6000000000);
        REQUIRE(hands.get_left_hand().get()); // retained fallback recovers without recreation
    }
    CHECK(state.live_trackers == 0);
    CHECK_FALSE(state.list_alive);
}

TEST_CASE("Live hands retain unknown-only devices when default creation fails", "[deviceio][xdev][hand]")
{
    Runtime state;
    runtime = &state;
    state.devices = { { 4, "Vendor Left" }, { 5, "Vendor Right" } };
    state.default_creation_fails = true;
    state.active[{ 4, XR_HAND_LEFT_EXT }] = true;
    state.active[{ 5, XR_HAND_RIGHT_EXT }] = true;
    {
        core::LiveHandTrackerImpl hands({ {}, {}, {}, get_proc }, nullptr);
        CHECK(state.created.empty());
        hands.update(1000000000);
        REQUIRE(hands.get_left_hand().get());
        REQUIRE(hands.get_right_hand().get());
        CHECK(state.created.size() == 2);
    }
    CHECK(state.live_trackers == 0);
    CHECK_FALSE(state.list_alive);
}
