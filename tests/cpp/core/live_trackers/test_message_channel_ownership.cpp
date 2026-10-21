// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <deviceio_trackers/message_channel_tracker.hpp>
#include <live_trackers/live_deviceio_factory.hpp>
#include <oxr_utils/oxr_funcs.hpp>
#include <oxr_utils/oxr_session_handles.hpp>

#include <XR_NV_opaque_data_channel.h>
#include <cstring>
#include <vector>

namespace
{

struct OpaqueRuntime
{
    XrResult query_result = XR_SUCCESS;
    XrOpaqueDataChannelNV channel = reinterpret_cast<XrOpaqueDataChannelNV>(0x33);
    std::vector<XrOpaqueDataChannelNV> shutdowns;
    std::vector<XrOpaqueDataChannelNV> destroys;
};

OpaqueRuntime* runtime = nullptr;

XRAPI_ATTR XrResult XRAPI_CALL create_channel(XrInstance, const XrOpaqueDataChannelCreateInfoNV*, XrOpaqueDataChannelNV* out)
{
    *out = runtime->channel;
    return XR_SUCCESS;
}

XRAPI_ATTR XrResult XRAPI_CALL destroy_channel(XrOpaqueDataChannelNV channel)
{
    runtime->destroys.push_back(channel);
    return XR_SUCCESS;
}

XRAPI_ATTR XrResult XRAPI_CALL shutdown_channel(XrOpaqueDataChannelNV channel)
{
    runtime->shutdowns.push_back(channel);
    return XR_SUCCESS;
}

XRAPI_ATTR XrResult XRAPI_CALL get_state(XrOpaqueDataChannelNV, XrOpaqueDataChannelStateNV* state)
{
    state->state = XR_OPAQUE_DATA_CHANNEL_STATUS_CONNECTED_NV;
    return runtime->query_result;
}

XRAPI_ATTR XrResult XRAPI_CALL get_system(XrInstance, const XrSystemGetInfo*, XrSystemId* out)
{
    *out = 1;
    return XR_SUCCESS;
}

// The status/retirement boundary never sends, receives or converts a timestamp.
// Required entries are loaded, but invoking one indicates a test scope violation.
XRAPI_ATTR void XRAPI_CALL unused_entry()
{
    FAIL("Unexpected OpenXR operation at the status/retirement boundary");
}

XRAPI_ATTR XrResult XRAPI_CALL get_proc(XrInstance, const char* name, PFN_xrVoidFunction* out)
{
    if (std::strcmp(name, "xrCreateOpaqueDataChannelNV") == 0)
        *out = reinterpret_cast<PFN_xrVoidFunction>(create_channel);
    else if (std::strcmp(name, "xrDestroyOpaqueDataChannelNV") == 0)
        *out = reinterpret_cast<PFN_xrVoidFunction>(destroy_channel);
    else if (std::strcmp(name, "xrShutdownOpaqueDataChannelNV") == 0)
        *out = reinterpret_cast<PFN_xrVoidFunction>(shutdown_channel);
    else if (std::strcmp(name, "xrGetOpaqueDataChannelStateNV") == 0)
        *out = reinterpret_cast<PFN_xrVoidFunction>(get_state);
    else if (std::strcmp(name, "xrGetSystem") == 0)
        *out = reinterpret_cast<PFN_xrVoidFunction>(get_system);
    else
        *out = unused_entry;
    return XR_SUCCESS;
}

struct RuntimeScope
{
    OpaqueRuntime state;
    RuntimeScope()
    {
        runtime = &state;
    }
    ~RuntimeScope()
    {
        runtime = nullptr;
    }
};

} // namespace

TEST_CASE("message channel status failures preserve retirement ownership", "[live_trackers][channel]")
{
    RuntimeScope scope;
    core::OpenXRSessionHandles handles{ reinterpret_cast<XrInstance>(0x11), reinterpret_cast<XrSession>(0x22),
                                        XR_NULL_HANDLE, get_proc };
    core::MessageChannelTracker tracker({});
    core::LiveDeviceIOFactory factory(handles, nullptr, {});
    auto impl = factory.create_message_channel_tracker_impl(&tracker);

    SECTION("transient query error retains the original channel until shutdown and destroy")
    {
        scope.state.query_result = XR_ERROR_VALIDATION_FAILURE;
        REQUIRE(impl->get_status() == core::MessageChannelStatus::DISCONNECTED);
        impl.reset();
        REQUIRE(scope.state.shutdowns == std::vector<XrOpaqueDataChannelNV>{ scope.state.channel });
        REQUIRE(scope.state.destroys == std::vector<XrOpaqueDataChannelNV>{ scope.state.channel });
    }
    SECTION("healthy query leaves the same retirement obligation")
    {
        REQUIRE(impl->get_status() == core::MessageChannelStatus::CONNECTED);
        impl.reset();
        REQUIRE(scope.state.shutdowns == std::vector<XrOpaqueDataChannelNV>{ scope.state.channel });
        REQUIRE(scope.state.destroys == std::vector<XrOpaqueDataChannelNV>{ scope.state.channel });
    }
    SECTION("fatal instance loss suppresses further native calls")
    {
        scope.state.query_result = XR_ERROR_INSTANCE_LOST;
        REQUIRE(impl->get_status() == core::MessageChannelStatus::DISCONNECTED);
        impl.reset();
        REQUIRE(scope.state.shutdowns.empty());
        REQUIRE(scope.state.destroys.empty());
    }
}
