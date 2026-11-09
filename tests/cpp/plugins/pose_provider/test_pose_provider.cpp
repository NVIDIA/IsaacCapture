// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// PoseProvider lifecycle tests: the stub adapter for a live run, and a scripted engine for call order and failures.

#include <catch2/catch_test_macros.hpp>
#include <pose_engine_api/pose_engine.hpp>
#include <pose_provider/pose_provider.hpp>

#include <algorithm>
#include <chrono>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace pose_provider;
using namespace std::chrono_literals;

namespace
{

constexpr StreamId kLeft{ 10 };
constexpr StreamId kRight{ 11 };
constexpr auto kShortTimeout = 50ms;

// A Stereo setup the stub accepts.
ResolvedPoseSetup stub_setup(const IPoseEngine& engine)
{
    ResolvedPoseSetup setup;
    setup.input_mode = InputMode::Stereo;
    setup.memory_kind = MemoryKind::Host;
    setup.body_frame = "left_optical";
    for (StreamId id : { kLeft, kRight })
    {
        StreamSetup stream;
        stream.id = id;
        stream.layout = ImageLayout{ 8, 4, PixelFormat::Gray8, 8 };
        stream.rate_hz = 30.0;
        stream.calibration.fx = 100.0;
        stream.calibration.fy = 100.0;
        stream.calibration.cx = 3.5;
        stream.calibration.cy = 1.5;
        if (id == kRight)
        {
            stream.calibration.T_body_camera.translation_m = { 0.075, 0.0, 0.0 };
        }
        setup.streams.push_back(stream);
    }
    setup.engine_identity = engine.identity();
    return setup;
}

// Held by shared_ptr so that a test can read it after PoseProvider has destroyed the engine.
struct Script
{
    std::string validation_error;
    std::set<std::string> throwing;
    std::string fatal_reason;
    bool drains = true;
    int estimates = 0;
    bool estimate_when_drained = false;
    int estimates_taken = 0;
    std::vector<std::string> calls; // Lifecycle calls, in order.
};

class FakeEngine final : public IPoseEngine
{
public:
    explicit FakeEngine(std::shared_ptr<Script> script) : script_(std::move(script))
    {
    }

    EngineCapabilities capabilities() const override
    {
        return {};
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

    void initialize(const ResolvedPoseSetup&) override
    {
        record("initialize");
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

SetupSupplier fake_setup()
{
    return [] { return std::optional<ResolvedPoseSetup>(ResolvedPoseSetup{}); };
}

SetupSupplier no_setup()
{
    return [] { return std::optional<ResolvedPoseSetup>(); };
}

std::unique_ptr<IPoseEngine> fake_engine(const std::shared_ptr<Script>& script)
{
    return std::make_unique<FakeEngine>(script);
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

using Calls = std::vector<std::string>;

} // namespace

TEST_CASE("PoseProvider runs the stub engine through a live run", "[unit][pose_provider]")
{
    auto engine = make_pose_engine();
    const ResolvedPoseSetup setup = stub_setup(*engine);
    int supplier_calls = 0;
    int calls_without_setup = 0;
    PoseProvider provider(ProviderConfig{}, std::move(engine),
                          [&]() -> std::optional<ResolvedPoseSetup>
                          {
                              if (supplier_calls++ < calls_without_setup)
                              {
                                  return std::nullopt;
                              }
                              return setup;
                          });

    SECTION("start requested after Ready")
    {
        CHECK(provider.status().state == ProviderState::Preparing);
        provider.update();
        CHECK(provider.status().state == ProviderState::Ready);
        provider.update();
        CHECK(provider.status().state == ProviderState::Ready);
        provider.request_start();
        provider.update();
        CHECK(provider.status().state == ProviderState::Running);
    }
    SECTION("start requested before the setup arrives")
    {
        calls_without_setup = 2;
        provider.request_start();
        provider.update();
        provider.update();
        CHECK(provider.status().state == ProviderState::Preparing);
        CHECK_FALSE(provider.status().failed);
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

    SECTION("normal finish")
    {
        PoseProvider provider(ProviderConfig{}, fake_engine(script), fake_setup());
        provider.request_start();
        provider.update();
        CHECK(provider.status().state == ProviderState::Running);
        provider.request_finish();
        provider.update();
        CHECK(provider.finished());
        CHECK_FALSE(provider.status().failed);
        CHECK(script->calls == Calls{ "validate_configuration", "initialize", "start", "finish_input", "stop" });
    }
    SECTION("finish while Preparing")
    {
        PoseProvider provider(ProviderConfig{}, fake_engine(script), no_setup());
        provider.request_finish();
        provider.update();
        CHECK(provider.finished());
        CHECK_FALSE(provider.status().failed);
        CHECK(script->calls == Calls{ "stop" });
    }
    SECTION("health() is not read from an engine that was never initialized")
    {
        script->throwing = { "health" };
        PoseProvider provider(ProviderConfig{}, fake_engine(script), no_setup());
        provider.request_finish();
        provider.update();
        CHECK(provider.finished());
        CHECK_FALSE(provider.status().failed);
        CHECK(script->calls == Calls{ "stop" });
    }
    SECTION("finish while Ready, with a start pending")
    {
        PoseProvider provider(ProviderConfig{}, fake_engine(script), fake_setup());
        provider.update();
        CHECK(provider.status().state == ProviderState::Ready);
        provider.request_start();
        provider.request_finish();
        provider.update();
        CHECK(provider.finished());
        CHECK_FALSE(provider.status().failed);
        CHECK(script->calls == Calls{ "validate_configuration", "initialize", "stop" });
    }
    SECTION("abort() while Running")
    {
        PoseProvider provider(ProviderConfig{}, fake_engine(script), fake_setup());
        provider.request_start();
        provider.update();
        provider.abort();
        const ProviderStatus status = provider.status();
        CHECK(status.state == ProviderState::Finished);
        CHECK(status.failed);
        CHECK(status.failed_component == ProviderComponent::Provider);
        CHECK(status.failure_reason == "aborted");
        CHECK(script->calls == Calls{ "validate_configuration", "initialize", "start", "stop" });
        provider.abort();
        CHECK(provider.status().failure_reason == "aborted");
        CHECK(script->calls == Calls{ "validate_configuration", "initialize", "start", "stop" });
    }
    SECTION("abort() after a clean finish")
    {
        PoseProvider provider(ProviderConfig{}, fake_engine(script), fake_setup());
        provider.request_start();
        provider.update();
        provider.request_finish();
        provider.update();
        const Calls before = script->calls;
        provider.abort();
        CHECK_FALSE(provider.status().failed);
        CHECK(script->calls == before);
    }
    SECTION("stop() throws on a clean finish")
    {
        script->throwing = { "stop" };
        PoseProvider provider(ProviderConfig{}, fake_engine(script), fake_setup());
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
    }
    SECTION("destroyed before any update()")
    {
        {
            PoseProvider provider(ProviderConfig{}, fake_engine(script), fake_setup());
        }
        CHECK(script->calls == Calls{ "stop" });
    }
    SECTION("destroyed while Running")
    {
        {
            PoseProvider provider(ProviderConfig{}, fake_engine(script), fake_setup());
            provider.request_start();
            provider.update();
        }
        CHECK(script->calls == Calls{ "validate_configuration", "initialize", "start", "stop" });
    }
}

TEST_CASE("Startup failures end the run with the failing component", "[unit][pose_provider]")
{
    auto script = std::make_shared<Script>();
    ProviderConfig short_startup;
    short_startup.startup_timeout = kShortTimeout;
    std::optional<PoseProvider> provider;
    std::optional<ProviderComponent> component;
    std::string reason;
    std::optional<Calls> calls;

    SECTION("no setup within the startup timeout")
    {
        provider.emplace(short_startup, fake_engine(script));
        CHECK(update_until_finished(*provider));
        component = ProviderComponent::Setup;
        reason = "startup timeout";
        calls = Calls{ "stop" };
    }
    SECTION("the setup supplier throws")
    {
        provider.emplace(ProviderConfig{}, fake_engine(script),
                         []() -> std::optional<ResolvedPoseSetup> { throw std::runtime_error("no camera metadata"); });
        provider->update();
        component = ProviderComponent::Setup;
        reason = "no camera metadata";
        calls = Calls{ "stop" };
    }
    SECTION("the engine rejects the setup")
    {
        script->validation_error = "bad setup";
        provider.emplace(ProviderConfig{}, fake_engine(script), fake_setup());
        provider->update();
        component = ProviderComponent::Setup;
        reason = "bad setup";
        calls = Calls{ "validate_configuration", "stop" };
    }
    SECTION("the stub rejects a one-stream setup")
    {
        auto engine = make_pose_engine();
        ResolvedPoseSetup setup = stub_setup(*engine);
        setup.streams.pop_back();
        provider.emplace(ProviderConfig{}, std::move(engine), [setup] { return std::optional(setup); });
        provider->update();
        component = ProviderComponent::Setup;
        reason = "stereo input needs exactly two streams";
    }
    SECTION("initialize() throws")
    {
        script->throwing = { "initialize" };
        provider.emplace(ProviderConfig{}, fake_engine(script), fake_setup());
        provider->update();
        component = ProviderComponent::Engine;
        reason = "initialize:";
        calls = Calls{ "validate_configuration", "initialize", "stop" };
    }
    SECTION("start() throws")
    {
        script->throwing = { "start" };
        provider.emplace(ProviderConfig{}, fake_engine(script), fake_setup());
        provider->request_start();
        provider->update();
        component = ProviderComponent::Engine;
        reason = "start:";
        calls = Calls{ "validate_configuration", "initialize", "start", "stop" };
    }
    SECTION("not started within the startup timeout")
    {
        provider.emplace(short_startup, fake_engine(script), fake_setup());
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
    if (calls)
    {
        CHECK(script->calls == *calls);
    }
}

TEST_CASE("Failures after startup keep the first reason and stop the engine once", "[unit][pose_provider]")
{
    auto script = std::make_shared<Script>();
    script->throwing = { "stop" };
    ProviderConfig config;
    std::optional<PoseProvider> provider;
    const auto start_provider = [&] { provider.emplace(config, fake_engine(script), fake_setup()); };
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
    PoseProvider provider(ProviderConfig{}, fake_engine(script), fake_setup());
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
