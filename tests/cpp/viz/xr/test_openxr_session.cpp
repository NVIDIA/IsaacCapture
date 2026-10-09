// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <viz/core/vk_context.hpp>
#include <viz/xr/openxr_session.hpp>

#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

// [xr]: needs a reachable OpenXR runtime + HMD. Filtered out by default.

// ── Stage 1: instance + system (no graphics) ──────────────────────────

TEST_CASE("OpenXrSession stage 1 creates an instance and finds an HMD system", "[xr][viz_xr]")
{
    try
    {
        viz::OpenXrSession sess("viz_xr_test", {});
        REQUIRE(sess.instance() != XR_NULL_HANDLE);
        REQUIRE(sess.system_id() != XR_NULL_SYSTEM_ID);
        REQUIRE_FALSE(sess.is_graphics_attached());
    }
    catch (const std::runtime_error& e)
    {
        SKIP(std::string("no OpenXR runtime / HMD: ") + e.what());
    }
}

// XR_KHR_convert_timespec_time: XrTime ↔ steady_clock must round-trip
// within microsecond resolution.
TEST_CASE("OpenXrSession time conversion round-trips when extension is available", "[xr][viz_xr]")
{
    std::unique_ptr<viz::OpenXrSession> sess;
    try
    {
        sess = std::make_unique<viz::OpenXrSession>("viz_xr_test_time_conv", std::vector<std::string>{});
    }
    catch (const std::runtime_error& e)
    {
        SKIP(std::string("no OpenXR runtime / HMD: ") + e.what());
    }

    if (!sess->has_time_conversion())
    {
        SKIP("Runtime does not advertise XR_KHR_convert_timespec_time");
    }

    const auto t0 = std::chrono::steady_clock::now();
    const XrTime xr = sess->steady_clock_to_xr_time(t0);
    REQUIRE(xr != 0);
    const auto t1 = sess->xr_time_to_steady_clock(xr);

    // Both directions just (de)compose timespec; 1 µs slack is generous.
    const auto drift = std::chrono::abs(t1 - t0);
    INFO("round-trip drift: " << std::chrono::duration_cast<std::chrono::nanoseconds>(drift).count() << " ns");
    CHECK(drift <= std::chrono::microseconds(1));

    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    const auto t2 = std::chrono::steady_clock::now();
    const XrTime xr2 = sess->steady_clock_to_xr_time(t2);
    CHECK(xr2 > xr);
}

// Loud failure when the extension isn't enabled.
TEST_CASE("OpenXrSession time conversion throws when extension is unavailable", "[xr][viz_xr]")
{
    std::unique_ptr<viz::OpenXrSession> sess;
    try
    {
        sess = std::make_unique<viz::OpenXrSession>("viz_xr_test_time_conv_disabled", std::vector<std::string>{});
    }
    catch (const std::runtime_error& e)
    {
        SKIP(std::string("no OpenXR runtime / HMD: ") + e.what());
    }

    if (sess->has_time_conversion())
    {
        SUCCEED("Runtime supports the extension; no negative test possible here");
        return;
    }

    CHECK_THROWS_AS(sess->xr_time_to_steady_clock(0), std::runtime_error);
    CHECK_THROWS_AS(sess->steady_clock_to_xr_time(std::chrono::steady_clock::now()), std::runtime_error);
}

// ── Stage 2: attach_graphics + session lifetime ───────────────────────

namespace
{

// Build stage-1 OpenXrSession + XR-bound VkContext, or SKIP if either
// piece isn't available on the host. The pair destroys its second member first,
// so the session goes before the VkContext it renders with; the other order
// crashes in xrDestroySession on a connected runtime.
std::pair<std::unique_ptr<viz::VkContext>, std::unique_ptr<viz::OpenXrSession>> make_stage1(const char* app_name)
{
    std::unique_ptr<viz::OpenXrSession> sess;
    try
    {
        sess = std::make_unique<viz::OpenXrSession>(app_name, std::vector<std::string>{});
    }
    catch (const std::runtime_error& e)
    {
        SKIP(std::string("no OpenXR runtime / HMD: ") + e.what());
    }

    auto vk = std::make_unique<viz::VkContext>();
    viz::VkContext::Config cfg{};
    cfg.xr_instance = sess->instance();
    cfg.xr_system_id = sess->system_id();
    try
    {
        vk->init(cfg);
    }
    catch (const std::runtime_error& e)
    {
        SKIP(std::string("XR-bound VkContext init failed: ") + e.what());
    }
    return { std::move(vk), std::move(sess) };
}

} // namespace

TEST_CASE("OpenXrSession attach_graphics constructs session + spaces + view config", "[xr][viz_xr]")
{
    auto [vk, sess] = make_stage1("viz_xr_test_attach");
    sess->attach_graphics(*vk);

    REQUIRE(sess->is_graphics_attached());
    REQUIRE(sess->session() != XR_NULL_HANDLE);
    REQUIRE(sess->reference_space() != XR_NULL_HANDLE);
    REQUIRE(sess->view_count() == 2u); // stereo HMD
    const auto& views = sess->view_configuration_views();
    CHECK(views[0].recommendedImageRectWidth > 0);
    CHECK(views[0].recommendedImageRectHeight > 0);
    CHECK(views[1].recommendedImageRectWidth > 0);
    CHECK(views[1].recommendedImageRectHeight > 0);
    // Not running yet — runtime needs STATE_READY before xrBeginSession.
    CHECK_FALSE(sess->session_running());
    CHECK_FALSE(sess->exit_requested());

    // Some runtimes only advance state once a frame is requested, so
    // we don't assert session_running here — construction is the test.
    for (int i = 0; i < 5; ++i)
    {
        sess->poll_events();
    }
}

// end_session(): xrRequestExitSession -> STOPPING -> xrEndSession, no
// XR_ERROR_SESSION_NOT_STOPPING. Needs a connected headset to reach a running session.
TEST_CASE("OpenXrSession::end_session stops a running session and is idempotent", "[xr][viz_xr]")
{
    auto [vk, sess] = make_stage1("viz_xr_test_end_session");
    sess->attach_graphics(*vk);

    // Never ran: nothing to end.
    CHECK(sess->end_session());

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!sess->session_running() && std::chrono::steady_clock::now() < deadline)
    {
        sess->poll_events();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (!sess->session_running())
    {
        SKIP("runtime did not start the session (no headset connected)");
    }

    CHECK(sess->end_session());
    CHECK_FALSE(sess->session_running());
    CHECK(sess->end_session()); // idempotent
}

// VIEW space + near/far Z plumbing.
TEST_CASE("OpenXrSession exposes VIEW space and propagates near/far Z config", "[xr][viz_xr]")
{
    auto [vk, sess] = make_stage1("viz_xr_test_view_space");
    viz::OpenXrSession::Config sess_cfg{};
    sess_cfg.near_z = 0.1f;
    sess_cfg.far_z = 250.0f;
    sess->attach_graphics(*vk, sess_cfg);

    REQUIRE(sess->view_space() != XR_NULL_HANDLE);
    CHECK(sess->near_z() == 0.1f);
    CHECK(sess->far_z() == 250.0f);

    // Must not throw regardless of session state. Validity flag depends
    // on whether the runtime is tracking yet.
    XrSpaceLocation loc{ XR_TYPE_SPACE_LOCATION };
    const bool valid = sess->locate_view_space(/*time=*/0, &loc);
    INFO("locate_view_space returned " << (valid ? "valid" : "invalid") << ", flags=0x" << std::hex << loc.locationFlags);
    SUCCEED();
}

// Double-attach is a programming error.
TEST_CASE("OpenXrSession::attach_graphics is single-shot", "[xr][viz_xr]")
{
    auto [vk, sess] = make_stage1("viz_xr_test_double_attach");
    sess->attach_graphics(*vk);
    CHECK_THROWS_AS(sess->attach_graphics(*vk), std::logic_error);
}
