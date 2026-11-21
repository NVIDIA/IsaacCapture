// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <deviceio_base/controller_tracker_base.hpp>
#include <deviceio_base/head_tracker_base.hpp>
#include <deviceio_trackers/controller_tracker.hpp>
#include <deviceio_trackers/head_tracker.hpp>
#include <live_trackers/live_deviceio_factory.hpp>
#include <openxr/openxr_platform.h>
#include <oxr/oxr_session.hpp>
#include <schema/head_generated.h>

#include <XR_NVX1_action_context.h>
#include <cstdlib>
#include <cstring>
#include <deque>
static int begins = 0, ends = 0;
static bool running = false;
static bool fail_begin = false, fail_end = false;
static XrSessionState runtime_state = XR_SESSION_STATE_READY;
static std::deque<XrSessionState> events;
static XrSession event_session = reinterpret_cast<XrSession>(2);
static int locates = 0;
static int syncs = 0, reads = 0, haptics = 0;
static XrSession session = reinterpret_cast<XrSession>(2);

template <class Handle, class Info, class Result>
XrResult create_handle(Handle, const Info*, Result* out)
{
    *out = reinterpret_cast<Result>(10);
    return XR_SUCCESS;
}
template <class Handle>
XrResult destroy_handle(Handle)
{
    return XR_SUCCESS;
}
XrResult string_path(XrInstance, const char*, XrPath* p)
{
    *p = 1;
    return XR_SUCCESS;
}
XrResult suggest(XrInstance, const XrInteractionProfileSuggestedBinding*)
{
    return XR_SUCCESS;
}
XrResult attach(XrSession, const XrSessionActionSetsAttachInfo*)
{
    return XR_SUCCESS;
}
XrResult sync_actions(XrSession, const XrActionsSyncInfo2NV*, XrActionsSyncState2NV*)
{
    ++syncs;
    return running ? XR_SUCCESS : XR_ERROR_SESSION_NOT_RUNNING;
}
template <class State>
XrResult read_action(XrSession, const XrActionStateGetInfo*, State* p)
{
    ++reads;
    p->isActive = XR_TRUE;
    return running ? XR_SUCCESS : XR_ERROR_SESSION_NOT_RUNNING;
}
XrResult apply_haptic(XrSession, const XrHapticActionInfo*, const XrHapticBaseHeader*)
{
    ++haptics;
    return running ? XR_SUCCESS : XR_ERROR_SESSION_NOT_RUNNING;
}
XrResult stop_haptic(XrSession, const XrHapticActionInfo*)
{
    ++haptics;
    return running ? XR_SUCCESS : XR_ERROR_SESSION_NOT_RUNNING;
}
extern "C"
{
    XrResult __wrap_xrCreateInstance(const XrInstanceCreateInfo*, XrInstance* p)
    {
        *p = reinterpret_cast<XrInstance>(1);
        return XR_SUCCESS;
    }
    XrResult __wrap_xrDestroyInstance(XrInstance)
    {
        return XR_SUCCESS;
    }
    XrResult __wrap_xrGetSystem(XrInstance, const XrSystemGetInfo*, XrSystemId* p)
    {
        *p = 1;
        return XR_SUCCESS;
    }
    XrResult __wrap_xrCreateSession(XrInstance, const XrSessionCreateInfo*, XrSession* p)
    {
        *p = session;
        return XR_SUCCESS;
    }
    XrResult __wrap_xrDestroySession(XrSession)
    {
        return XR_SUCCESS;
    }
    XrResult __wrap_xrCreateReferenceSpace(XrSession, const XrReferenceSpaceCreateInfo*, XrSpace* p)
    {
        *p = reinterpret_cast<XrSpace>(3);
        return XR_SUCCESS;
    }
    XrResult __wrap_xrDestroySpace(XrSpace)
    {
        return XR_SUCCESS;
    }
    XrResult __wrap_xrEnumerateViewConfigurations(
        XrInstance, XrSystemId, uint32_t cap, uint32_t* n, XrViewConfigurationType* p)
    {
        *n = 1;
        if (cap)
            *p = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
        return XR_SUCCESS;
    }
    XrResult __wrap_xrBeginSession(XrSession, const XrSessionBeginInfo*)
    {
        ++begins;
        if (fail_begin)
            return XR_ERROR_RUNTIME_FAILURE;
        if (runtime_state != XR_SESSION_STATE_READY)
            return XR_ERROR_SESSION_NOT_READY;
        if (running)
            return XR_ERROR_SESSION_RUNNING;
        running = true;
        return XR_SUCCESS;
    }
    XrResult __wrap_xrEndSession(XrSession)
    {
        ++ends;
        if (fail_end)
            return XR_ERROR_RUNTIME_FAILURE;
        if (runtime_state != XR_SESSION_STATE_STOPPING)
            return XR_ERROR_SESSION_NOT_STOPPING;
        if (!running)
            return XR_ERROR_SESSION_NOT_RUNNING;
        running = false;
        return XR_SUCCESS;
    }
    XrResult __wrap_xrPollEvent(XrInstance, XrEventDataBuffer* p)
    {
        if (events.empty())
            return XR_EVENT_UNAVAILABLE;
        auto* e = reinterpret_cast<XrEventDataSessionStateChanged*>(p);
        e->type = XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED;
        e->session = event_session;
        e->state = events.front();
        if (event_session == session)
            runtime_state = e->state;
        events.pop_front();
        return XR_SUCCESS;
    }
    XrResult __wrap_xrLocateSpace(XrSpace, XrSpace, XrTime, XrSpaceLocation* p)
    {
        ++locates;
        if (!running || runtime_state == XR_SESSION_STATE_STOPPING || runtime_state == XR_SESSION_STATE_IDLE)
        {
            p->locationFlags = 0;
            return XR_SUCCESS;
        }
        p->locationFlags = XR_SPACE_LOCATION_POSITION_VALID_BIT | XR_SPACE_LOCATION_ORIENTATION_VALID_BIT |
                           XR_SPACE_LOCATION_POSITION_TRACKED_BIT | XR_SPACE_LOCATION_ORIENTATION_TRACKED_BIT;
        return XR_SUCCESS;
    }
    XrResult to_time(XrInstance, const timespec*, XrTime* p)
    {
        *p = 1;
        return XR_SUCCESS;
    }
    XrResult to_ts(XrInstance, XrTime, timespec*)
    {
        return XR_SUCCESS;
    }
    XrResult __wrap_xrGetInstanceProcAddr(XrInstance, const char* name, PFN_xrVoidFunction* p)
    {
        if (std::strcmp(name, "xrConvertTimespecTimeToTimeKHR") == 0)
            *p = reinterpret_cast<PFN_xrVoidFunction>(to_time);
        else if (std::strcmp(name, "xrCreateReferenceSpace") == 0)
            *p = reinterpret_cast<PFN_xrVoidFunction>(__wrap_xrCreateReferenceSpace);
        else if (std::strcmp(name, "xrDestroySpace") == 0)
            *p = reinterpret_cast<PFN_xrVoidFunction>(__wrap_xrDestroySpace);
        else if (std::strcmp(name, "xrLocateSpace") == 0)
            *p = reinterpret_cast<PFN_xrVoidFunction>(__wrap_xrLocateSpace);
        else if (std::strcmp(name, "xrStringToPath") == 0)
            *p = reinterpret_cast<PFN_xrVoidFunction>(string_path);
        else if (std::strcmp(name, "xrCreateActionSet") == 0)
            *p = reinterpret_cast<PFN_xrVoidFunction>(create_handle<XrInstance, XrActionSetCreateInfo, XrActionSet>);
        else if (std::strcmp(name, "xrDestroyActionSet") == 0)
            *p = reinterpret_cast<PFN_xrVoidFunction>(destroy_handle<XrActionSet>);
        else if (std::strcmp(name, "xrCreateAction") == 0)
            *p = reinterpret_cast<PFN_xrVoidFunction>(create_handle<XrActionSet, XrActionCreateInfo, XrAction>);
        else if (std::strcmp(name, "xrCreateActionSpace") == 0)
            *p = reinterpret_cast<PFN_xrVoidFunction>(create_handle<XrSession, XrActionSpaceCreateInfo, XrSpace>);
        else if (std::strcmp(name, "xrSuggestInteractionProfileBindings") == 0)
            *p = reinterpret_cast<PFN_xrVoidFunction>(suggest);
        else if (std::strcmp(name, "xrAttachSessionActionSets") == 0)
            *p = reinterpret_cast<PFN_xrVoidFunction>(attach);
        else if (std::strcmp(name, "xrCreateInstanceActionContextNV") == 0)
            *p = reinterpret_cast<PFN_xrVoidFunction>(
                create_handle<XrInstance, XrInstanceActionContextCreateInfoNV, XrInstanceActionContextNV>);
        else if (std::strcmp(name, "xrDestroyInstanceActionContextNV") == 0)
            *p = reinterpret_cast<PFN_xrVoidFunction>(destroy_handle<XrInstanceActionContextNV>);
        else if (std::strcmp(name, "xrCreateSessionActionContextNV") == 0)
            *p = reinterpret_cast<PFN_xrVoidFunction>(
                create_handle<XrSession, XrSessionActionContextCreateInfoNV, XrSessionActionContextNV>);
        else if (std::strcmp(name, "xrDestroySessionActionContextNV") == 0)
            *p = reinterpret_cast<PFN_xrVoidFunction>(destroy_handle<XrSessionActionContextNV>);
        else if (std::strcmp(name, "xrSyncActions2NV") == 0)
            *p = reinterpret_cast<PFN_xrVoidFunction>(sync_actions);
        else if (std::strcmp(name, "xrGetActionStatePose") == 0)
            *p = reinterpret_cast<PFN_xrVoidFunction>(read_action<XrActionStatePose>);
        else if (std::strcmp(name, "xrGetActionStateBoolean") == 0)
            *p = reinterpret_cast<PFN_xrVoidFunction>(read_action<XrActionStateBoolean>);
        else if (std::strcmp(name, "xrGetActionStateFloat") == 0)
            *p = reinterpret_cast<PFN_xrVoidFunction>(read_action<XrActionStateFloat>);
        else if (std::strcmp(name, "xrGetActionStateVector2f") == 0)
            *p = reinterpret_cast<PFN_xrVoidFunction>(read_action<XrActionStateVector2f>);
        else if (std::strcmp(name, "xrApplyHapticFeedback") == 0)
            *p = reinterpret_cast<PFN_xrVoidFunction>(apply_haptic);
        else if (std::strcmp(name, "xrStopHapticFeedback") == 0)
            *p = reinterpret_cast<PFN_xrVoidFunction>(stop_haptic);
        else
            *p = reinterpret_cast<PFN_xrVoidFunction>(to_ts);
        return XR_SUCCESS;
    }
}
namespace
{
struct Runtime
{
    Runtime()
    {
        setenv("ISAAC_TELEOP_DISABLE_CXR_ENV_CHECKS", "1", 1);
        begins = 0;
        ends = 0;
        running = false;
        fail_begin = false;
        fail_end = false;
        runtime_state = XR_SESSION_STATE_READY;
        locates = 0;
        syncs = reads = haptics = 0;
        events.clear();
        event_session = session;
    }
};
}
TEST_CASE_METHOD(Runtime, "owned session ends and begins across reconnect", "[unit][oxr]")
{
    core::OpenXRSession owned("test", {});
    REQUIRE(begins == 1);
    const auto handles = owned.get_handles();
    REQUIRE(handles.session_running);
    CHECK(handles.is_running());
    events = { XR_SESSION_STATE_READY, XR_SESSION_STATE_READY };
    owned.get_provider_snapshot();
    CHECK(begins == 1);
    events = { XR_SESSION_STATE_STOPPING, XR_SESSION_STATE_STOPPING, XR_SESSION_STATE_IDLE };
    const int active_locates = locates;
    const auto stopped = owned.get_provider_snapshot();
    CHECK(locates == active_locates);
    CHECK(ends == 1);
    CHECK_FALSE(handles.is_running());
    CHECK(stopped.state == core::OpenXRProviderState::AVAILABLE);
    CHECK(stopped.headset_state == core::OpenXRHeadsetState::DISCONNECTED);
    const int stopped_locates = locates;
    owned.get_provider_snapshot();
    CHECK(locates == stopped_locates);
    events = { XR_SESSION_STATE_READY, XR_SESSION_STATE_READY };
    const auto resumed = owned.get_provider_snapshot();
    CHECK(begins == 2);
    CHECK(handles.is_running());
    CHECK(resumed.headset_state == core::OpenXRHeadsetState::CONNECTED);
}
TEST_CASE_METHOD(Runtime, "foreign session events do not stop owned session", "[unit][oxr]")
{
    core::OpenXRSession owned("test", {});
    event_session = reinterpret_cast<XrSession>(99);
    events = { XR_SESSION_STATE_STOPPING, XR_SESSION_STATE_LOSS_PENDING };
    const auto status = owned.get_provider_snapshot();
    CHECK(ends == 0);
    CHECK(status.state == core::OpenXRProviderState::AVAILABLE);
    CHECK(status.headset_state == core::OpenXRHeadsetState::CONNECTED);
}
TEST_CASE_METHOD(Runtime, "terminal session loss stays failed", "[unit][oxr]")
{
    core::OpenXRSession owned("test", {});
    events = { XR_SESSION_STATE_LOSS_PENDING, XR_SESSION_STATE_READY };
    const auto failed = owned.get_provider_snapshot();
    CHECK(failed.state == core::OpenXRProviderState::FAILED);
    CHECK_FALSE(owned.get_handles().is_running());
    CHECK(failed.reason == core::OpenXRProviderReason::SESSION_LOST);
    events = { XR_SESSION_STATE_READY };
    CHECK(owned.get_provider_snapshot().state == core::OpenXRProviderState::FAILED);
    CHECK(begins == 1);
}

TEST_CASE_METHOD(Runtime, "session lifecycle API failures suppress later input", "[unit][oxr]")
{
    core::OpenXRSession owned("test", {});
    SECTION("end fails")
    {
        fail_end = true;
        events = { XR_SESSION_STATE_STOPPING };
    }
    SECTION("begin fails after stop")
    {
        events = { XR_SESSION_STATE_STOPPING };
        owned.get_provider_snapshot();
        fail_begin = true;
        events = { XR_SESSION_STATE_READY };
    }
    const int before_locates = locates;
    const auto failed = owned.get_provider_snapshot();
    CHECK(failed.state == core::OpenXRProviderState::FAILED);
    CHECK(failed.reason == core::OpenXRProviderReason::POLL_ERROR);
    CHECK_FALSE(owned.get_handles().is_running());
    CHECK(locates == before_locates);
}
TEST_CASE("external OpenXR handles retain caller-managed lifecycle", "[unit][oxr]")
{
    CHECK(core::OpenXRSessionHandles{}.is_running());
}

TEST_CASE_METHOD(Runtime, "live head clears stopped samples and resumes with copied handles", "[unit][oxr]")
{
    core::OpenXRSession owned("test", {});
    auto handles = owned.get_handles();
    SECTION("owned lifecycle")
    {
    }
    SECTION("external lifecycle")
    {
        handles.session_running.reset();
    }
    core::HeadTracker head;
    core::LiveDeviceIOFactory factory(handles, nullptr, {});
    auto tracker = factory.create_head_tracker_impl(&head);
    tracker->update(1);
    REQUIRE(tracker->get_head());
    CHECK(tracker->get_head()->is_valid());
    events = { XR_SESSION_STATE_STOPPING };
    owned.get_provider_snapshot();
    const int before_locates = locates;
    tracker->update(2);
    if (handles.session_running)
    {
        CHECK_FALSE(tracker->get_head());
        CHECK(locates == before_locates);
    }
    else
    {
        CHECK(tracker->get_head());
        CHECK(locates == before_locates + 1);
    }
    events = { XR_SESSION_STATE_READY };
    owned.get_provider_snapshot();
    tracker->update(3);
    REQUIRE(tracker->get_head());
    CHECK(tracker->get_head()->is_valid());
}

TEST_CASE_METHOD(Runtime, "live controllers suppress stopped input and haptics then recover", "[unit][oxr]")
{
    core::OpenXRSession owned("test", {});
    const auto handles = owned.get_handles();
    core::ControllerTracker controller;
    core::LiveDeviceIOFactory factory(handles, nullptr, {});
    auto tracker = factory.create_controller_tracker_impl(&controller);
    tracker->update(1);
    REQUIRE(tracker->get_left_controller());
    REQUIRE(tracker->get_right_controller());
    tracker->apply_left_haptic_feedback(0.5f, 0.f, 0.f);
    const int before_sync = syncs, before_reads = reads, before_haptics = haptics;
    CHECK(before_sync > 0);
    CHECK(before_reads > 0);
    CHECK(before_haptics > 0);
    events = { XR_SESSION_STATE_STOPPING };
    owned.get_provider_snapshot();
    tracker->update(2);
    tracker->apply_left_haptic_feedback(0.5f, 0.f, 0.f);
    tracker->apply_right_haptic_feedback(0.f, 0.f, 0.f);
    CHECK_FALSE(tracker->get_left_controller());
    CHECK_FALSE(tracker->get_right_controller());
    CHECK(syncs == before_sync);
    CHECK(reads == before_reads);
    CHECK(haptics == before_haptics);
    events = { XR_SESSION_STATE_READY };
    owned.get_provider_snapshot();
    tracker->update(3);
    CHECK(tracker->get_left_controller());
    CHECK(tracker->get_right_controller());
    CHECK(syncs == before_sync + 1);
    tracker->apply_left_haptic_feedback(0.5f, 0.f, 0.f);
    CHECK(haptics == before_haptics + 1);
}

TEST_CASE_METHOD(Runtime, "initial session must already be READY before eager begin", "[unit][oxr]")
{
    runtime_state = XR_SESSION_STATE_IDLE;
    CHECK_THROWS(core::OpenXRSession("test", {}));
    CHECK(begins == 1);
    CHECK_FALSE(running);
}

TEST_CASE_METHOD(Runtime, "focus changes retain a running session", "[unit][oxr]")
{
    core::OpenXRSession owned("test", {});
    events = { XR_SESSION_STATE_SYNCHRONIZED, XR_SESSION_STATE_VISIBLE, XR_SESSION_STATE_FOCUSED };
    const auto status = owned.get_provider_snapshot();
    CHECK(status.state == core::OpenXRProviderState::AVAILABLE);
    CHECK(owned.get_handles().is_running());
    CHECK(begins == 1);
    CHECK(ends == 0);
}
