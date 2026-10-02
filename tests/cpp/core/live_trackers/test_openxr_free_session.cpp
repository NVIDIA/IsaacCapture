// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// A session whose trackers all work without OpenXR runs with null OpenXR handles; any tracker
// that needs OpenXR still requires them.

#include <catch2/catch_test_macros.hpp>
#include <deviceio_session/deviceio_session.hpp>
#include <deviceio_trackers/hand_tracker.hpp>
#include <deviceio_trackers/head_tracker.hpp>
#include <live_trackers/live_deviceio_factory.hpp>
#include <oxr_utils/oxr_session_handles.hpp>

#include <memory>
#include <stdexcept>

TEST_CASE("requires_openxr: OpenXR trackers need it, no trackers need none", "[unit][openxr_free]")
{
    auto head = std::make_shared<core::HeadTracker>();
    auto hands = std::make_shared<core::HandTracker>();

    CHECK(core::DeviceIOSession::requires_openxr({ head }));
    CHECK(core::DeviceIOSession::requires_openxr({ head, hands }));
    CHECK(core::LiveDeviceIOFactory::requires_openxr({ hands }));
    CHECK_FALSE(core::DeviceIOSession::requires_openxr({}));
}

TEST_CASE("DeviceIOSession: runs with null handles when no tracker needs OpenXR", "[unit][openxr_free]")
{
    auto session = core::DeviceIOSession::run({}, core::OpenXRSessionHandles{});

    REQUIRE(session);
    session->update();
}

TEST_CASE("DeviceIOSession: null handles are rejected when a tracker needs OpenXR", "[unit][openxr_free]")
{
    auto head = std::make_shared<core::HeadTracker>();

    CHECK_THROWS_AS(core::DeviceIOSession::run({ head }, core::OpenXRSessionHandles{}), std::invalid_argument);
}
