// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// PoseProvider tests on a scripted camera input: the stub adapter for a live run, and a scripted engine for call
// order, setup resolution and failures.

#include "fake_camera_input.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <pose_camera_input/camera_input.hpp>
#include <pose_engine_api/pose_engine.hpp>
#include <pose_provider/pose_provider.hpp>
#include <pose_undistort/undistort.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace pose_provider;
using namespace pose_provider::testing;
using namespace std::chrono_literals;
using Catch::Matchers::WithinAbs;

namespace
{

constexpr auto kShortTimeout = 50ms;

// Held by shared_ptr so that a test can read it after PoseProvider has destroyed the engine.
struct Script
{
    EngineCapabilities capabilities{
        { InputMode::Stereo }, { MemoryKind::Host }, { PixelFormat::Gray8 }, { DistortionModel::None }
    };
    std::string validation_error;
    std::set<std::string> throwing;
    std::string fatal_reason;
    bool drains = true;
    int estimates = 0;
    bool estimate_when_drained = false;
    int estimates_taken = 0;
    std::vector<std::string> calls; // Lifecycle calls, in order.
    std::optional<ResolvedPoseSetup> initialized; // The setup initialize() received.
};

class FakeEngine final : public IPoseEngine
{
public:
    explicit FakeEngine(std::shared_ptr<Script> script) : script_(std::move(script))
    {
    }

    EngineCapabilities capabilities() const override
    {
        throw_if_scripted("capabilities");
        return script_->capabilities;
    }

    EngineIdentity identity() const override
    {
        return EngineIdentity{ "fake", "1", "none", "none" };
    }

    std::string validate_configuration(const ResolvedPoseSetup&) const override
    {
        record("validate_configuration");
        return script_->validation_error;
    }

    void initialize(const ResolvedPoseSetup& setup) override
    {
        record("initialize");
        script_->initialized = setup;
    }

    void start() override
    {
        record("start");
    }

    EnqueueResult on_image_group(EngineImageGroup) override
    {
        return EnqueueResult::RejectedNotRunning;
    }

    EnqueueResult on_imu(const EngineImuSample&) override
    {
        return EnqueueResult::RejectedNotRunning;
    }

    void finish_input() override
    {
        record("finish_input");
        finish_called_ = true;
    }

    bool input_drained() const override
    {
        throw_if_scripted("input_drained");
        const bool drained = script_->drains && finish_called_;
        if (drained && script_->estimate_when_drained)
        {
            script_->estimate_when_drained = false;
            ++script_->estimates;
        }
        return drained;
    }

    bool try_get_next_pose(EnginePose& out) override
    {
        throw_if_scripted("try_get_next_pose");
        if (script_->estimates <= 0)
        {
            return false;
        }
        --script_->estimates;
        ++script_->estimates_taken;
        out = EnginePose{};
        out.group_id = 7;
        return true;
    }

    EngineHealth health() const override
    {
        throw_if_scripted("health");
        EngineHealth health;
        health.fatal_error = !script_->fatal_reason.empty();
        health.fatal_reason = script_->fatal_reason;
        return health;
    }

    void stop() override
    {
        record("stop");
    }

private:
    void record(const std::string& call) const
    {
        script_->calls.push_back(call);
        throw_if_scripted(call);
    }

    void throw_if_scripted(const std::string& call) const
    {
        if (script_->throwing.contains(call))
        {
            throw std::runtime_error(call + " failed");
        }
    }

    std::shared_ptr<Script> script_;
    bool finish_called_ = false;
};

std::unique_ptr<IPoseEngine> fake_engine(const std::shared_ptr<Script>& script)
{
    return std::make_unique<FakeEngine>(script);
}

// An input whose source description is already waiting.
std::unique_ptr<ICameraInput> ready_input(const std::shared_ptr<FakeCameraSource>& camera,
                                          SourceDescription description = fake_stereo_description())
{
    camera->push(std::move(description));
    return std::make_unique<FakeCameraInput>(camera);
}

long stop_calls(const Script& script)
{
    return std::count(script.calls.begin(), script.calls.end(), std::string("stop"));
}

bool update_until_finished(PoseProvider& provider)
{
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    provider.update();
    while (!provider.finished() && std::chrono::steady_clock::now() < deadline)
    {
        std::this_thread::sleep_for(1ms);
        provider.update();
    }
    return provider.finished();
}

void check_transform(const Transform& actual, const Transform& expected)
{
    for (std::size_t i = 0; i < 3; ++i)
    {
        CHECK_THAT(actual.translation_m[i], WithinAbs(expected.translation_m[i], 1e-12));
    }
    for (std::size_t i = 0; i < 4; ++i)
    {
        CHECK_THAT(actual.rotation_xyzw[i], WithinAbs(expected.rotation_xyzw[i], 1e-12));
    }
}

using Calls = std::vector<std::string>;

// A change to a description the profile accepts, and the Setup failure reason it must produce.
struct SetupRejection
{
    std::function<void(SourceDescription&, Script&)> change;
    std::string reason;
};

} // namespace

TEST_CASE("PoseProvider runs the stub engine through a live run", "[unit][pose_provider]")
{
    auto camera = std::make_shared<FakeCameraSource>();
    PoseProvider provider(ProviderConfig{}, make_pose_engine(), std::make_unique<FakeCameraInput>(camera));

    SECTION("start requested after Ready")
    {
        camera->push(fake_stereo_description());
        CHECK(provider.status().state == ProviderState::Preparing);
        provider.update();
        CHECK(provider.status().state == ProviderState::Ready);
        provider.update();
        CHECK(provider.status().state == ProviderState::Ready);
        provider.request_start();
        provider.update();
        CHECK(provider.status().state == ProviderState::Running);
    }
    SECTION("start requested before the camera input is ready")
    {
        provider.request_start();
        provider.update();
        CHECK(provider.status().state == ProviderState::Preparing);
        CHECK_FALSE(provider.status().failed);
        camera->push(fake_stereo_description());
        provider.update();
        CHECK(provider.status().state == ProviderState::Running);
    }

    provider.request_finish();
    provider.request_finish();
    provider.update();
    const ProviderStatus status = provider.status();
    CHECK(status.state == ProviderState::Finished);
    CHECK_FALSE(status.failed);
    CHECK(status.failure_reason.empty());
    CHECK_FALSE(status.failed_component.has_value());
    CHECK_FALSE(status.engine_health.fatal_error);
    CHECK(status.engine_health.not_running_rejections == 0);
    CHECK(status.engine_health.invalid_input_rejections == 0);
    CHECK(status.startup_timeout == ProviderConfig{}.startup_timeout);
    CHECK(status.drain_timeout == ProviderConfig{}.drain_timeout);
    CHECK(camera->input_destroyed());
    CHECK(provider.wait_fd() == -1);

    provider.update();
    provider.request_start();
    provider.request_finish();
    provider.abort();
    CHECK(provider.status().state == ProviderState::Finished);
    CHECK_FALSE(provider.status().failed);
}

TEST_CASE("Every path calls the engine in contract order and stops it exactly once", "[unit][pose_provider]")
{
    auto script = std::make_shared<Script>();
    auto camera = std::make_shared<FakeCameraSource>();

    SECTION("normal finish")
    {
        PoseProvider provider(ProviderConfig{}, fake_engine(script), ready_input(camera));
        provider.request_start();
        provider.update();
        CHECK(provider.status().state == ProviderState::Running);
        provider.request_finish();
        provider.update();
        CHECK(provider.finished());
        CHECK_FALSE(provider.status().failed);
        CHECK(script->calls == Calls{ "validate_configuration", "initialize", "start", "finish_input", "stop" });
        CHECK(camera->input_destroyed());
    }
    SECTION("finish while Preparing")
    {
        PoseProvider provider(ProviderConfig{}, fake_engine(script), std::make_unique<FakeCameraInput>(camera));
        provider.request_finish();
        provider.update();
        CHECK(provider.finished());
        CHECK_FALSE(provider.status().failed);
        CHECK(script->calls == Calls{ "stop" });
        CHECK(camera->input_destroyed());
    }
    SECTION("health() is not read from an engine that was never initialized")
    {
        script->throwing = { "health" };
        PoseProvider provider(ProviderConfig{}, fake_engine(script), std::make_unique<FakeCameraInput>(camera));
        provider.request_finish();
        provider.update();
        CHECK(provider.finished());
        CHECK_FALSE(provider.status().failed);
        CHECK(script->calls == Calls{ "stop" });
        CHECK(camera->input_destroyed());
    }
    SECTION("finish while Ready, with a start pending")
    {
        PoseProvider provider(ProviderConfig{}, fake_engine(script), ready_input(camera));
        provider.update();
        CHECK(provider.status().state == ProviderState::Ready);
        provider.request_start();
        provider.request_finish();
        provider.update();
        CHECK(provider.finished());
        CHECK_FALSE(provider.status().failed);
        CHECK(script->calls == Calls{ "validate_configuration", "initialize", "stop" });
        CHECK(camera->input_destroyed());
    }
    SECTION("abort() while Running")
    {
        PoseProvider provider(ProviderConfig{}, fake_engine(script), ready_input(camera));
        provider.request_start();
        provider.update();
        CHECK_FALSE(camera->input_destroyed());
        provider.abort();
        const ProviderStatus status = provider.status();
        CHECK(status.state == ProviderState::Finished);
        CHECK(status.failed);
        CHECK(status.failed_component == ProviderComponent::Provider);
        CHECK(status.failure_reason == "aborted");
        CHECK(script->calls == Calls{ "validate_configuration", "initialize", "start", "stop" });
        CHECK(camera->input_destroyed());
        provider.abort();
        CHECK(provider.status().failure_reason == "aborted");
        CHECK(script->calls == Calls{ "validate_configuration", "initialize", "start", "stop" });
    }
    SECTION("abort() after a clean finish")
    {
        PoseProvider provider(ProviderConfig{}, fake_engine(script), ready_input(camera));
        provider.request_start();
        provider.update();
        provider.request_finish();
        provider.update();
        const Calls before = script->calls;
        provider.abort();
        CHECK_FALSE(provider.status().failed);
        CHECK(script->calls == before);
        CHECK(camera->input_destroyed());
    }
    SECTION("stop() throws on a clean finish")
    {
        script->throwing = { "stop" };
        PoseProvider provider(ProviderConfig{}, fake_engine(script), ready_input(camera));
        provider.request_start();
        provider.update();
        provider.request_finish();
        provider.update();
        const ProviderStatus status = provider.status();
        CHECK(status.state == ProviderState::Finished);
        CHECK(status.failed);
        CHECK(status.failed_component == ProviderComponent::Engine);
        CHECK(status.failure_reason.starts_with("stop:"));
        CHECK(stop_calls(*script) == 1);
        CHECK(camera->input_destroyed());
    }
    SECTION("destroyed before any update()")
    {
        {
            PoseProvider provider(ProviderConfig{}, fake_engine(script), ready_input(camera));
        }
        CHECK(script->calls == Calls{ "stop" });
        CHECK(camera->input_destroyed());
    }
    SECTION("destroyed while Running")
    {
        {
            PoseProvider provider(ProviderConfig{}, fake_engine(script), ready_input(camera));
            provider.request_start();
            provider.update();
        }
        CHECK(script->calls == Calls{ "validate_configuration", "initialize", "start", "stop" });
        CHECK(camera->input_destroyed());
    }
}

TEST_CASE("Startup failures end the run with the failing component", "[unit][pose_provider]")
{
    auto script = std::make_shared<Script>();
    auto camera = std::make_shared<FakeCameraSource>();
    ProviderConfig short_startup;
    short_startup.startup_timeout = kShortTimeout;
    std::optional<PoseProvider> provider;
    std::optional<ProviderComponent> component;
    std::string reason;
    std::optional<Calls> calls;

    SECTION("no camera input within the startup timeout")
    {
        provider.emplace(short_startup, fake_engine(script));
        CHECK(provider->wait_fd() == -1);
        CHECK(update_until_finished(*provider));
        component = ProviderComponent::CameraInput;
        reason = "startup timeout";
        calls = Calls{ "stop" };
    }
    SECTION("the camera input fails before it is ready")
    {
        camera->push(InputFailed{ "no such device" });
        provider.emplace(ProviderConfig{}, fake_engine(script), std::make_unique<FakeCameraInput>(camera));
        provider->update();
        component = ProviderComponent::CameraInput;
        reason = "no such device";
        calls = Calls{ "stop" };
    }
    SECTION("capabilities() throws")
    {
        script->throwing = { "capabilities" };
        provider.emplace(ProviderConfig{}, fake_engine(script), ready_input(camera));
        provider->update();
        component = ProviderComponent::Engine;
        reason = "capabilities:";
        calls = Calls{ "stop" };
    }
    SECTION("the engine rejects the setup")
    {
        script->validation_error = "bad setup";
        provider.emplace(ProviderConfig{}, fake_engine(script), ready_input(camera));
        provider->update();
        component = ProviderComponent::Setup;
        reason = "bad setup";
        calls = Calls{ "validate_configuration", "stop" };
    }
    SECTION("initialize() throws")
    {
        script->throwing = { "initialize" };
        provider.emplace(ProviderConfig{}, fake_engine(script), ready_input(camera));
        provider->update();
        component = ProviderComponent::Engine;
        reason = "initialize:";
        calls = Calls{ "validate_configuration", "initialize", "stop" };
    }
    SECTION("start() throws")
    {
        script->throwing = { "start" };
        provider.emplace(ProviderConfig{}, fake_engine(script), ready_input(camera));
        provider->request_start();
        provider->update();
        component = ProviderComponent::Engine;
        reason = "start:";
        calls = Calls{ "validate_configuration", "initialize", "start", "stop" };
    }
    SECTION("not started within the startup timeout")
    {
        provider.emplace(short_startup, fake_engine(script), ready_input(camera));
        CHECK(update_until_finished(*provider));
        component = ProviderComponent::Provider;
        reason = "startup timeout";
        calls = Calls{ "validate_configuration", "initialize", "stop" };
    }

    const ProviderStatus status = provider->status();
    CHECK(status.state == ProviderState::Finished);
    CHECK(status.failed);
    CHECK(status.failed_component == component);
    CHECK(status.failure_reason.find(reason) != std::string::npos);
    CHECK(script->calls == *calls);
    CHECK(provider->wait_fd() == -1);
    CHECK(provider->wait_timeout() == 0ms);
}

TEST_CASE("The setup is resolved from the camera's description", "[unit][pose_provider]")
{
    auto script = std::make_shared<Script>();
    auto camera = std::make_shared<FakeCameraSource>();
    SourceDescription description = fake_stereo_description();
    // The rig's reference is not the left eye. The eyes turn about different axes, and neither rotation is quite unit
    // length.
    const double scale = 1 + 1e-4;
    const double half = std::sqrt(0.5) * scale;
    description.left.T_rig_camera =
        Transform{ { 0.01, 0.02, 0.03 }, { 0.5 * scale, 0.5 * scale, 0.5 * scale, 0.5 * scale } };
    description.right.T_rig_camera = Transform{ { 0.012, 0.095, 0.031 }, { half, half, 0.0, 0.0 } };
    description.right.intrinsics =
        Intrinsics{ 101.0, 102.0, 3.25, 1.75, DistortionModel::RadTan5, { -0.2, 0.05, 1e-3, -1e-3, 0.01 } };
    description.right.rate_hz = 60.0;
    description.right.timed_event = TimedEvent::FrameEnd;
    PoseProvider provider(ProviderConfig{}, fake_engine(script), ready_input(camera, description));
    provider.update();
    REQUIRE(provider.status().state == ProviderState::Ready);
    REQUIRE(script->initialized.has_value());
    const ResolvedPoseSetup& setup = *script->initialized;

    CHECK(setup.input_mode == InputMode::Stereo);
    CHECK(setup.memory_kind == MemoryKind::Host);
    CHECK_FALSE(setup.imu.has_value());
    CHECK(setup.body_frame == "left_optical");
    CHECK(setup.engine_identity == EngineIdentity{ "fake", "1", "none", "none" });
    REQUIRE(setup.streams.size() == 2);
    const StreamSetup& left = setup.streams[0];
    const StreamSetup& right = setup.streams[1];
    CHECK(left.id == StreamId{ 0 });
    CHECK(right.id == StreamId{ 1 });
    CHECK(left.rate_hz == 30.0);
    CHECK(right.rate_hz == 60.0);
    for (const StreamSetup* stream : { &left, &right })
    {
        CHECK(stream->layout == ImageLayout{ 8, 4, PixelFormat::Gray8, 8 });
        CHECK(stream->calibration.distortion == DistortionModel::None);
        CHECK(stream->calibration.coefficients.empty());
    }

    // Each eye's own lens, undistorted at the delivered size, with the principal point used as given.
    const auto widest = [](const Intrinsics& lens)
    {
        return widest_undistorted_calibration(
            CameraCalibration{ lens.fx, lens.fy, lens.cx, lens.cy, lens.distortion, lens.coefficients, {} }, 8, 4);
    };
    const CameraCalibration left_expected = widest(*description.left.intrinsics);
    const CameraCalibration right_expected = widest(*description.right.intrinsics);
    CHECK(left.calibration.fx == left_expected.fx);
    CHECK(left.calibration.fy == left_expected.fy);
    CHECK(right.calibration.fx == right_expected.fx);
    CHECK(right.calibration.fy == right_expected.fy);
    CHECK(right.calibration.fx != description.right.intrinsics->fx);
    CHECK(left.calibration.cx == 3.5);
    CHECK(left.calibration.cy == 1.5);
    CHECK(right.calibration.cx == 3.25);
    CHECK(right.calibration.cy == 1.75);

    // Body is the left eye's optical frame, so the right eye sits about one baseline along its x axis.
    check_transform(left.calibration.T_body_camera, Transform{});
    check_transform(right.calibration.T_body_camera,
                    Transform{ { 0.075, 0.001, 0.002 }, { std::sqrt(0.5), 0.0, 0.0, std::sqrt(0.5) } });
}

TEST_CASE("A description the camera profile cannot use fails the setup, naming the source and the eye",
          "[unit][pose_provider]")
{
    const SetupRejection rows[] = {
        { [](SourceDescription& d, Script&) { d.backend = "other"; },
          "other device 'fake-device': no camera profile for this backend" },
        { [](SourceDescription& d, Script&)
          {
              d.backend.clear();
              d.device_id.clear();
          },
          "streams 'fake/left' and 'fake/right': no camera profile for this backend" },
        { [](SourceDescription& d, Script& s)
          {
              d.left.format = PixelFormat::RGBA8;
              s.capabilities.pixel_formats.clear();
          },
          "oak device 'fake-device': the engine does not take Stereo host Gray8 images without distortion" },
        { [](SourceDescription& d, Script&) { d.left.format = PixelFormat::RGBA8; },
          "oak device 'fake-device': left eye 'fake/left' is not Gray8" },
        { [](SourceDescription& d, Script&) { d.right.shutter = Shutter::Rolling; },
          "oak device 'fake-device': right eye 'fake/right' has no global shutter" },
        { [](SourceDescription& d, Script&) { d.left.shutter = Shutter::Unknown; },
          "oak device 'fake-device': left eye 'fake/left' has no global shutter" },
        { [](SourceDescription& d, Script&) { d.left.timed_event = TimedEvent::Unknown; },
          "oak device 'fake-device': left eye 'fake/left' does not state which event its sample times mark" },
        { [](SourceDescription& d, Script&) { d.left.rate_hz.reset(); },
          "oak device 'fake-device': left eye 'fake/left' has no positive nominal rate" },
        { [](SourceDescription& d, Script&) { d.right.rate_hz = 0.0; },
          "oak device 'fake-device': right eye 'fake/right' has no positive nominal rate" },
        { [](SourceDescription& d, Script&) { d.right.intrinsics.reset(); },
          "oak device 'fake-device': right eye 'fake/right' has no intrinsics" },
        { [](SourceDescription& d, Script&) { d.left.T_rig_camera.reset(); },
          "oak device 'fake-device': left eye 'fake/left' has no pose in the rig" },
        { [](SourceDescription& d, Script&) {
             d.right.T_rig_camera->rotation_xyzw = { 0.0, 0.0, 0.0, 0.99 };
         },
          "oak device 'fake-device': right eye 'fake/right' has a pose in the rig that is not a rigid transform" },
        { [](SourceDescription& d, Script&) { d.right.intrinsics->distortion = DistortionModel::Tilted14; },
          "oak device 'fake-device': right eye 'fake/right': source calibration needs one finite coefficient per term "
          "of its distortion model" },
    };
    for (const SetupRejection& row : rows)
    {
        INFO(row.reason);
        auto script = std::make_shared<Script>();
        auto camera = std::make_shared<FakeCameraSource>();
        SourceDescription description = fake_stereo_description();
        row.change(description, *script);
        PoseProvider provider(ProviderConfig{}, fake_engine(script), ready_input(camera, description));
        provider.update();
        const ProviderStatus status = provider.status();
        CHECK(status.state == ProviderState::Finished);
        CHECK(status.failed_component == ProviderComponent::Setup);
        CHECK(status.failure_reason == row.reason);
        CHECK(script->calls == Calls{ "stop" });
    }
}

TEST_CASE("A last camera event fails the run only before a finish request", "[unit][pose_provider]")
{
    auto script = std::make_shared<Script>();
    script->drains = false;
    auto camera = std::make_shared<FakeCameraSource>();
    PoseProvider provider(ProviderConfig{}, fake_engine(script), ready_input(camera));
    provider.request_start();
    provider.update();
    REQUIRE(provider.status().state == ProviderState::Running);

    SECTION("the source ends while Running")
    {
        camera->push(fake_frame(1));
        camera->push_end();
        provider.update();
        const ProviderStatus status = provider.status();
        CHECK(status.state == ProviderState::Finished);
        CHECK(status.failed_component == ProviderComponent::CameraInput);
        CHECK(status.failure_reason == "the camera input ended");
        CHECK(script->calls == Calls{ "validate_configuration", "initialize", "start", "stop" });
    }
    SECTION("the input fails after the finish request")
    {
        provider.request_finish();
        provider.update();
        REQUIRE(provider.status().state == ProviderState::Draining);
        camera->push(fake_frame(1));
        camera->push(InputFailed{ "transport lost" });
        provider.update();
        CHECK(provider.status().state == ProviderState::Draining);
        CHECK_FALSE(provider.status().failed);
        script->drains = true;
        provider.update();
        CHECK(provider.finished());
        CHECK_FALSE(provider.status().failed);
    }
    CHECK(camera->input_destroyed());
    CHECK(camera->frames_released() == camera->frames_delivered());
}

TEST_CASE("update() takes every waiting camera event, and the wait timeout follows the state", "[unit][pose_provider]")
{
    auto script = std::make_shared<Script>();
    script->drains = false;
    auto camera = std::make_shared<FakeCameraSource>();
    ProviderConfig config;
    config.startup_timeout = kMaxProviderTimeout;
    PoseProvider provider(config, fake_engine(script), std::make_unique<FakeCameraInput>(camera));
    const int fd = provider.wait_fd();
    REQUIRE(fd >= 0);

    provider.update();
    CHECK(provider.status().state == ProviderState::Preparing);
    // Nothing needs polling until the camera input is ready.
    CHECK(provider.wait_timeout() > 1min);

    camera->push(fake_stereo_description());
    camera->push(fake_frame(1));
    provider.update();
    CHECK(provider.status().state == ProviderState::Ready);
    CHECK_FALSE(readable(fd));
    // The engine has no wake handle, so it is polled long before the startup deadline.
    CHECK(provider.wait_timeout() < 1s);

    camera->push(fake_frame(2));
    provider.update();
    CHECK(provider.status().state == ProviderState::Ready);
    CHECK_FALSE(readable(fd));

    provider.request_start();
    provider.update();
    camera->push(fake_frame(3));
    camera->push(FrameLoss{ 4, 1 });
    camera->push(fake_frame(5));
    provider.update();
    CHECK(provider.status().state == ProviderState::Running);
    CHECK_FALSE(readable(fd));
    CHECK(provider.wait_timeout() < 1s);
    CHECK(camera->frames_delivered() == 4);
    CHECK(camera->frames_released() == 4);

    provider.request_finish();
    provider.update();
    camera->push(fake_frame(6));
    camera->push_end();
    provider.update();
    CHECK(provider.status().state == ProviderState::Draining);
    CHECK_FALSE(readable(fd));
    CHECK(camera->frames_released() == 5);
    CHECK(provider.wait_timeout() > 0ms);
    CHECK(provider.wait_timeout() < 1s);
}

TEST_CASE("Failures after startup keep the first reason and stop the engine once", "[unit][pose_provider]")
{
    auto script = std::make_shared<Script>();
    script->throwing = { "stop" };
    auto camera = std::make_shared<FakeCameraSource>();
    ProviderConfig config;
    std::optional<PoseProvider> provider;
    const auto start_provider = [&] { provider.emplace(config, fake_engine(script), ready_input(camera)); };
    std::string reason;

    SECTION("fatal error while Ready")
    {
        start_provider();
        provider->update();
        REQUIRE(provider->status().state == ProviderState::Ready);
        script->fatal_reason = "tracker diverged";
        provider->update();
        CHECK(provider->status().engine_health.fatal_error);
        CHECK(script->calls == Calls{ "validate_configuration", "initialize", "stop" });
        reason = "tracker diverged";
    }
    SECTION("fatal error while Running")
    {
        start_provider();
        provider->request_start();
        provider->update();
        REQUIRE(provider->status().state == ProviderState::Running);
        script->fatal_reason = "tracker diverged";
        provider->update();
        reason = "tracker diverged";
    }
    SECTION("fatal error while Draining")
    {
        start_provider();
        script->drains = false;
        provider->request_start();
        provider->update();
        provider->request_finish();
        provider->update();
        REQUIRE(provider->status().state == ProviderState::Draining);
        script->fatal_reason = "tracker diverged";
        provider->update();
        reason = "tracker diverged";
    }
    SECTION("finish_input() throws")
    {
        start_provider();
        script->throwing.insert("finish_input");
        provider->request_start();
        provider->update();
        provider->request_finish();
        provider->update();
        CHECK(provider->status().failure_reason.starts_with("finish_input:"));
        reason = "finish_input:";
    }
    SECTION("try_get_next_pose() throws")
    {
        start_provider();
        script->throwing.insert("try_get_next_pose");
        provider->request_start();
        provider->update();
        CHECK(provider->status().failure_reason.starts_with("try_get_next_pose:"));
        CHECK(script->calls == Calls{ "validate_configuration", "initialize", "start", "stop" });
        reason = "try_get_next_pose:";
    }
    SECTION("an estimate for an image group never accepted")
    {
        start_provider();
        provider->request_start();
        provider->update();
        script->estimates = 1;
        provider->update();
        reason = "never accepted";
    }
    SECTION("not drained within the drain timeout")
    {
        config.drain_timeout = kShortTimeout;
        start_provider();
        script->drains = false;
        provider->request_start();
        provider->update();
        provider->request_finish();
        provider->update();
        CHECK(provider->status().state == ProviderState::Draining);
        CHECK_FALSE(provider->status().failed);
        provider->request_finish();
        CHECK(update_until_finished(*provider));
        CHECK(std::count(script->calls.begin(), script->calls.end(), std::string("finish_input")) == 1);
        reason = "drain timeout";
    }
    SECTION("health() keeps throwing")
    {
        start_provider();
        provider->request_start();
        provider->update();
        script->throwing.insert("health");
        provider->update();
        CHECK(provider->status().failure_reason.starts_with("health:"));
        reason = "health:";
    }

    const auto check_first_failure = [&]
    {
        const ProviderStatus status = provider->status();
        CHECK(status.state == ProviderState::Finished);
        CHECK(status.failed);
        CHECK(status.failed_component == ProviderComponent::Engine);
        CHECK(status.failure_reason.find(reason) != std::string::npos);
        CHECK(status.failure_reason.find("stop:") == std::string::npos);
        CHECK(stop_calls(*script) == 1);
        CHECK(camera->input_destroyed());
    };
    check_first_failure();
    const std::string first_reason = provider->status().failure_reason;
    provider->request_finish();
    provider->update();
    provider->abort();
    check_first_failure();
    CHECK(provider->status().failure_reason == first_reason);
}

TEST_CASE("Draining takes an estimate queued as the engine reports drained", "[unit][pose_provider]")
{
    auto script = std::make_shared<Script>();
    script->estimate_when_drained = true;
    auto camera = std::make_shared<FakeCameraSource>();
    PoseProvider provider(ProviderConfig{}, fake_engine(script), ready_input(camera));
    provider.request_start();
    provider.update();
    REQUIRE(provider.status().state == ProviderState::Running);
    provider.request_finish();
    provider.update();

    CHECK(provider.finished());
    // A provider that polled before reading input_drained() would finish cleanly with no estimate taken.
    CHECK(script->estimates_taken == 1);
    CHECK(provider.status().failed);
    CHECK(provider.status().failed_component == ProviderComponent::Engine);
    CHECK(provider.status().failure_reason.find("never accepted") != std::string::npos);
}

TEST_CASE("The constructor rejects a null engine and timeouts out of range", "[unit][pose_provider]")
{
    auto script = std::make_shared<Script>();
    CHECK_THROWS_AS(PoseProvider(ProviderConfig{}, nullptr), std::invalid_argument);

    ProviderConfig config;
    config.startup_timeout = 0ms;
    CHECK_THROWS_AS(PoseProvider(config, fake_engine(script)), std::invalid_argument);

    config = ProviderConfig{};
    config.drain_timeout = -1ms;
    CHECK_THROWS_AS(PoseProvider(config, fake_engine(script)), std::invalid_argument);

    config = ProviderConfig{};
    config.startup_timeout = kMaxProviderTimeout + 1ms;
    CHECK_THROWS_AS(PoseProvider(config, fake_engine(script)), std::invalid_argument);
}
