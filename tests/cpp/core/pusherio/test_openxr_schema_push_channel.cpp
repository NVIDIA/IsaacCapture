// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <oxr_utils/oxr_session_handles.hpp>
#include <pusherio/openxr_schema_push_channel.hpp>

#include <XR_NVX1_push_tensor.h>
#include <cstdint>
#include <ctime>
#include <map>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace
{

// The XR_NVX1_push_tensor and timespec functions OpenXRSchemaPushChannel loads, served from
// memory so push_buffer runs without an OpenXR runtime.
int g_push_count = 0;

XrResult XRAPI_CALL create_collection(XrSession,
                                      const XrPushTensorCollectionCreateInfoNV*,
                                      XrPushTensorCollectionCreateResultNV*,
                                      XrPushTensorCollectionNV* collection)
{
    *collection = reinterpret_cast<XrPushTensorCollectionNV>(1);
    return XR_SUCCESS;
}

XrResult XRAPI_CALL destroy_collection(XrPushTensorCollectionNV)
{
    return XR_SUCCESS;
}

XrResult XRAPI_CALL push_data(XrPushTensorCollectionNV, const XrPushTensorCollectionDataNV*)
{
    ++g_push_count;
    return XR_SUCCESS;
}

XrResult XRAPI_CALL timespec_to_time(XrInstance, const struct timespec* ts, XrTime* time)
{
    *time = static_cast<XrTime>(ts->tv_sec) * 1000000000 + ts->tv_nsec;
    return XR_SUCCESS;
}

XrResult XRAPI_CALL time_to_timespec(XrInstance, XrTime time, struct timespec* ts)
{
    ts->tv_sec = static_cast<time_t>(time / 1000000000);
    ts->tv_nsec = static_cast<long>(time % 1000000000);
    return XR_SUCCESS;
}

XrResult XRAPI_CALL get_instance_proc_addr(XrInstance, const char* name, PFN_xrVoidFunction* function)
{
    static const std::map<std::string_view, PFN_xrVoidFunction> functions = {
        { "xrCreatePushTensorCollectionNV", reinterpret_cast<PFN_xrVoidFunction>(&create_collection) },
        { "xrDestroyPushTensorCollectionNV", reinterpret_cast<PFN_xrVoidFunction>(&destroy_collection) },
        { "xrPushTensorCollectionDataNV", reinterpret_cast<PFN_xrVoidFunction>(&push_data) },
        { "xrConvertTimespecTimeToTimeKHR", reinterpret_cast<PFN_xrVoidFunction>(&timespec_to_time) },
        { "xrConvertTimeToTimespecTimeKHR", reinterpret_cast<PFN_xrVoidFunction>(&time_to_timespec) },
    };
    const auto it = functions.find(name);
    *function = it != functions.end() ? it->second : nullptr;
    return it != functions.end() ? XR_SUCCESS : XR_ERROR_FUNCTION_UNSUPPORTED;
}

core::OpenXRSessionHandles make_handles()
{
    core::OpenXRSessionHandles handles;
    handles.instance = reinterpret_cast<XrInstance>(1);
    handles.session = reinterpret_cast<XrSession>(1);
    handles.space = reinterpret_cast<XrSpace>(1);
    handles.xrGetInstanceProcAddr = &get_instance_proc_addr;
    return handles;
}

core::SchemaPusherConfig make_config()
{
    return core::SchemaPusherConfig{ .collection_id = "pedals",
                                     .max_flatbuffer_size = 16,
                                     .tensor_identifier = "pedal_state",
                                     .localized_name = "Pedal state",
                                     .app_name = "PusherTest" };
}

} // namespace

TEST_CASE("OpenXRSchemaPushChannel validates buffer metadata before pushing", "[pusherio][unit]")
{
    g_push_count = 0;
    core::OpenXRSchemaPushChannel channel(make_handles(), make_config());
    const std::vector<uint8_t> oversized(17, 0);

    REQUIRE_THROWS_AS(channel.push_buffer(oversized.data(), oversized.size(), 100, 80), std::runtime_error);
    REQUIRE_THROWS_AS(channel.push_buffer(nullptr, 1, 100, 80), std::invalid_argument);
    REQUIRE_THROWS_AS(channel.push_buffer(nullptr, 0, 100, -1), std::runtime_error);
    REQUIRE(g_push_count == 0);

    channel.push_buffer(oversized.data(), 16, 100, 80);
    REQUIRE(g_push_count == 1);
}
