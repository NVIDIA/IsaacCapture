// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// PoseProvider tests on a scripted camera input: the stub adapter for a live run, and a scripted engine for call
// order, setup resolution, failures and the output, with a recording plugin session in place of OpenXR.

#include "fake_camera_input.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <flatbuffers/flatbuffers.h>
#include <pose_camera_input/camera_input.hpp>
#include <pose_engine_api/pose_engine.hpp>
#include <pose_provider/pose_provider.hpp>
#include <pose_undistort/undistort.hpp>
#include <pusherio/plugin_session.hpp>
#include <pusherio/schema_push_channel.hpp>
#include <schema/plugin_device_status_generated.h>
#include <schema/se3_tracker_generated.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <fstream>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <variant>
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
    std::deque<uint64_t> estimates; // Group IDs of the estimates try_get_next_pose() returns, in order.
    // Each estimate carries this state and pose, and its group's anchor.
    TrackingState tracking_state = TrackingState::Initializing;
    Transform pose;
    std::optional<uint64_t> estimate_when_drained; // Queued as input_drained() first returns true.
    int estimates_taken = 0;
    EnqueueResult admission = EnqueueResult::Accepted; // What on_image_group() returns.
    std::vector<EngineImageGroup> offered; // An accepted group keeps its holds until stop() or release().
    std::vector<std::string> calls; // Lifecycle calls, in order.
    std::optional<ResolvedPoseSetup> initialized; // The setup initialize() received.
};

// Drops the group's holds, as an engine does after its last read.
void release(EngineImageGroup& group)
{
    for (EngineImage& image : group.images)
    {
        std::get<HostImage>(image.storage).hold.reset();
    }
}

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

    EnqueueResult on_image_group(EngineImageGroup group) override
    {
        throw_if_scripted("on_image_group");
        if (script_->admission != EnqueueResult::Accepted)
        {
            release(group);
        }
        script_->offered.push_back(std::move(group));
        return script_->admission;
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
            script_->estimates.push_back(*script_->estimate_when_drained);
            script_->estimate_when_drained.reset();
        }
        return drained;
    }

    bool try_get_next_pose(EnginePose& out) override
    {
        throw_if_scripted("try_get_next_pose");
        if (script_->estimates.empty())
        {
            return false;
        }
        out = EnginePose{ script_->estimates.front(), {}, script_->pose, script_->tracking_state };
        for (const EngineImageGroup& group : script_->offered)
        {
            if (group.group_id == out.group_id)
            {
                out.anchor = group.anchor;
            }
        }
        script_->estimates.pop_front();
        ++script_->estimates_taken;
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
        for (EngineImageGroup& group : script_->offered)
        {
            release(group);
        }
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

constexpr const char* kPoseCollection = "pose/run";
constexpr const char* kStatusCollection = "pose_root/device_status";

ProviderConfig output_config()
{
    ProviderConfig config;
    config.output_collection = kPoseCollection;
    config.plugin_root_id = "pose_root";
    return config;
}

struct PublishedPose
{
    std::array<float, 3> position{};
    std::array<float, 4> orientation{}; // x, y, z, w
    bool valid = false;
    AnchorTimes times;
};

struct PublishedStatus
{
    std::string path;
    core::PluginDeviceState state = core::PluginDeviceState_UNKNOWN;
    core::PluginDeviceReason reason = core::PluginDeviceReason_NONE;
    std::string error;
};

// Stands in for OpenXR. Records each collection's opening and closing in the script's calls, marking a closing before
// the camera input's release or after its session's, and decodes exactly the bytes pushed. With failing_session,
// opening the session throws, and with null_session the factory returns none; opening a collection in failing_open, or
// pushing to one in failing_push, throws.
struct FakeOutput
{
    std::shared_ptr<Script> script;
    std::shared_ptr<FakeCameraSource> camera;
    std::vector<PublishedPose> poses;
    std::vector<PublishedStatus> statuses;
    bool failing_session = false;
    bool null_session = false;
    bool session_open = false;
    std::set<std::string> failing_open;
    std::set<std::string> failing_push;

    std::unique_ptr<core::ISchemaPushChannel> open(const core::SchemaPusherConfig& config);

    void push(const std::string& collection_id, const uint8_t* pushed, std::size_t size, AnchorTimes times)
    {
        if (failing_push.contains(collection_id))
        {
            throw std::runtime_error("cannot push to " + collection_id);
        }
        const std::vector<uint8_t> buffer(pushed, pushed + size);
        const uint8_t* data = buffer.data();
        flatbuffers::Verifier verifier(data, size);
        if (collection_id == kPoseCollection)
        {
            REQUIRE(verifier.VerifyBuffer<core::Se3TrackerPose>(nullptr));
            const auto* sample = flatbuffers::GetRoot<core::Se3TrackerPose>(data);
            const core::Point& p = sample->pose()->position();
            const core::Quaternion& q = sample->pose()->orientation();
            poses.push_back({ { p.x(), p.y(), p.z() }, { q.x(), q.y(), q.z(), q.w() }, sample->is_valid(), times });
            return;
        }
        if (collection_id != kStatusCollection)
        {
            FAIL_CHECK("push to an unexpected collection '" << collection_id << "'");
            return;
        }
        REQUIRE(verifier.VerifyBuffer<core::PluginDeviceStatusSnapshot>(nullptr));
        const auto* snapshot = flatbuffers::GetRoot<core::PluginDeviceStatusSnapshot>(data);
        CHECK(snapshot->devices()->size() == 1);
        const core::PluginDeviceStatus* device = snapshot->devices()->Get(0);
        statuses.push_back({ device->path()->str(), device->state(), device->reason(),
                             device->error() != nullptr ? device->error()->str() : std::string() });
    }

    void close(const std::string& collection_id)
    {
        script->calls.push_back("close " + collection_id +
                                (camera->input_destroyed() ? "" : " before the camera input's release") +
                                (session_open ? "" : " after its session's close"));
    }
};

// One collection of a FakeOutput, closed when the provider destroys it.
class FakeChannel final : public core::ISchemaPushChannel
{
public:
    FakeChannel(FakeOutput& output, std::string collection_id)
        : output_(output), collection_id_(std::move(collection_id))
    {
    }
    ~FakeChannel() override
    {
        output_.close(collection_id_);
    }

    void push_buffer(const uint8_t* data, std::size_t size, int64_t common, int64_t device) override
    {
        output_.push(collection_id_, data, size, AnchorTimes{ common, device });
    }

private:
    FakeOutput& output_;
    std::string collection_id_;
};

// A FakeOutput as the provider's plugin session.
class FakeSession final : public core::IPluginSession
{
public:
    explicit FakeSession(std::shared_ptr<FakeOutput> output) : output_(std::move(output))
    {
        output_->session_open = true;
    }
    ~FakeSession() override
    {
        output_->session_open = false;
    }

    core::IPluginPullChannel& pull_channel() override
    {
        throw std::logic_error("FakeSession has no pull channel");
    }

    std::unique_ptr<core::ISchemaPushChannel> create_schema_push_channel(const core::SchemaPusherConfig& config) override
    {
        return output_->open(config);
    }

    std::unique_ptr<core::IHandTrackingPushChannel> create_hand_tracking_push_channel(XrHandEXT) override
    {
        throw std::logic_error("FakeSession has no hand tracking push");
    }

private:
    std::shared_ptr<FakeOutput> output_;
};

std::unique_ptr<core::ISchemaPushChannel> FakeOutput::open(const core::SchemaPusherConfig& config)
{
    if (failing_open.contains(config.collection_id))
    {
        throw std::runtime_error("cannot open " + config.collection_id);
    }
    if (config.collection_id == kPoseCollection)
    {
        // What an Se3Tracker reader on the collection expects.
        CHECK(config.tensor_identifier == "se3_tracker");
        CHECK(config.max_flatbuffer_size == 256);
    }
    script->calls.push_back("open " + config.collection_id);
    return std::make_unique<FakeChannel>(*this, config.collection_id);
}

std::shared_ptr<FakeOutput> fake_output(const std::shared_ptr<Script>& script,
                                        const std::shared_ptr<FakeCameraSource>& camera)
{
    auto output = std::make_shared<FakeOutput>();
    output->script = script;
    output->camera = camera;
    return output;
}

std::unique_ptr<core::IPluginSession> open_fake_session(const std::shared_ptr<FakeOutput>& output,
                                                        const core::PluginSessionRequirements& requirements,
                                                        const std::vector<std::shared_ptr<core::ITracker>>& trackers)
{
    CHECK(requirements.schema_push);
    CHECK_FALSE(requirements.hand_tracking_push);
    CHECK_FALSE(requirements.wrist_tracking_pull.has_value());
    CHECK(trackers.empty());
    if (output->failing_session)
    {
        throw std::runtime_error("cannot open the plugin session");
    }
    if (output->null_session)
    {
        return nullptr;
    }
    return std::make_unique<FakeSession>(output);
}

core::PluginSessionFactory open_on(const std::shared_ptr<FakeOutput>& output)
{
    return [output](const core::PluginSessionRequirements& requirements,
                    std::vector<std::shared_ptr<core::ITracker>> trackers)
    { return open_fake_session(output, requirements, trackers); };
}

// The common clock, CLOCK_MONOTONIC, which steady_clock reads on Linux.
int64_t now_ns()
{
    return std::chrono::nanoseconds(std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::string read_file(const char* path)
{
    std::ifstream file(path);
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

void check_status(const PublishedStatus& status,
                  core::PluginDeviceState state,
                  core::PluginDeviceReason reason,
                  const std::string& error = "")
{
    CHECK(status.path == "/se3_tracker/pose_provider");
    CHECK(status.state == state);
    CHECK(status.reason == reason);
    CHECK(status.error == error);
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

// A change to the frame after fake_frame(1), or events queued before it, and the InputPreparation failure reason.
struct FrameRejection
{
    std::function<void(FakeCameraSource&, FakeFrame&)> change;
    std::string reason;
    std::size_t offered = 1; // Groups offered before the failure.
};

// After the finish request, with an accepted group still in the engine: Draining ends the engine's input once the
// camera input's last event is taken, then finishes cleanly when the engine drains.
void check_clean_finish(PoseProvider& provider, Script& script)
{
    provider.update();
    CHECK(provider.status().state == ProviderState::Draining);
    CHECK_FALSE(provider.status().failed);
    CHECK(script.calls.back() == "finish_input");
    CHECK(provider.wait_fd() == -1);
    script.drains = true;
    provider.update();
    CHECK(provider.finished());
    CHECK_FALSE(provider.status().failed);
}

constexpr uint32_t kPaddedStride = 11;

// 8x4 Gray8 rows kPaddedStride bytes apart at level base + 9 x + 20 y, with every padding byte at 255.
std::vector<std::byte> padded_ramp(int base)
{
    std::vector<std::byte> bytes(kPaddedStride * 4, std::byte{ 255 });
    for (uint32_t y = 0; y < 4; ++y)
    {
        for (uint32_t x = 0; x < 8; ++x)
        {
            bytes[y * kPaddedStride + x] = static_cast<std::byte>(base + 9 * x + 20 * y);
        }
    }
    return bytes;
}

// fake_frame(sequence) with padded ramps that differ per frame, and right-eye times 5 ns after the left eye's.
FakeFrame padded_frame(uint64_t sequence)
{
    FakeFrame frame = fake_frame(sequence);
    frame.left.layout.stride_bytes = kPaddedStride;
    frame.right.layout.stride_bytes = kPaddedStride;
    frame.left_bytes = padded_ramp(10 + static_cast<int>(sequence));
    frame.right_bytes = padded_ramp(100 + static_cast<int>(sequence));
    *frame.right.local_common_clock_ns += 5;
    *frame.right.raw_device_clock_ns += 5;
    return frame;
}

std::vector<int> levels(const std::byte* data, std::size_t size)
{
    std::vector<int> out;
    for (std::size_t i = 0; i < size; ++i)
    {
        out.push_back(std::to_integer<int>(data[i]));
    }
    return out;
}

// The prepared image of @p stream in @p group.
const HostImage& image_of(const EngineImageGroup& group, StreamId stream)
{
    const auto image = std::find_if(group.images.begin(), group.images.end(),
                                    [&](const EngineImage& candidate) { return candidate.stream == stream; });
    REQUIRE(image != group.images.end());
    return std::get<HostImage>(image->storage);
}

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

    // The stub checks each group against the setup and returns one estimate per group, in order.
    for (uint64_t sequence = 1; sequence <= 3; ++sequence)
    {
        camera->push(fake_frame(sequence));
    }
    provider.update();
    provider.request_finish();
    provider.request_finish();
    CHECK(update_until_finished(provider));
    const ProviderStatus status = provider.status();
    CHECK(status.state == ProviderState::Finished);
    CHECK_FALSE(status.failed);
    CHECK(status.engine_health.dropped_image_groups == 0);
    CHECK(status.frames_dropped == 0);
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
    // The engine holds an accepted group and has not drained.
    camera->push(fake_frame(1));
    provider.update();
    REQUIRE(script->offered.size() == 1);
    camera->push(fake_frame(2));

    SECTION("the source ends while Running")
    {
        camera->push_end();
        provider.update();
        const ProviderStatus status = provider.status();
        CHECK(status.state == ProviderState::Finished);
        CHECK(status.failed_component == ProviderComponent::CameraInput);
        CHECK(status.failure_reason == "the camera input ended");
        CHECK(script->calls == Calls{ "validate_configuration", "initialize", "start", "stop" });
        CHECK(script->offered.size() == 2);
    }
    // Frame 2 follows the finish request in these sections, so it is released without being offered.
    SECTION("the input fails after the finish request")
    {
        camera->push(InputFailed{ "transport lost" });
        provider.request_finish();
        check_clean_finish(provider, *script);
        CHECK(script->offered.size() == 1);
    }
    SECTION("the source ends after the finish request")
    {
        camera->push_end();
        provider.request_finish();
        check_clean_finish(provider, *script);
        CHECK(script->offered.size() == 1);
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

    camera->push(fake_frame(6));
    provider.request_finish();
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
    SECTION("fatal error while the camera input drains")
    {
        start_provider();
        provider->request_start();
        provider->update();
        camera->withhold_end(true);
        provider->request_finish();
        provider->update();
        REQUIRE(provider->status().state == ProviderState::Draining);
        script->fatal_reason = "tracker diverged";
        provider->update();
        CHECK(script->calls == Calls{ "validate_configuration", "initialize", "start", "stop" });
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
        script->estimates = { 7 };
        provider->update();
        reason = "image group 7, which is not an accepted group";
    }
    SECTION("the engine rejects an image group as invalid input")
    {
        start_provider();
        script->admission = EnqueueResult::RejectedInvalidInput;
        provider->request_start();
        provider->update();
        camera->push(fake_frame(1));
        provider->update();
        reason = "rejected image group 1 as invalid input";
    }
    SECTION("the engine rejects an image group as not running")
    {
        start_provider();
        script->admission = EnqueueResult::RejectedNotRunning;
        provider->request_start();
        provider->update();
        camera->push(fake_frame(1));
        provider->update();
        reason = "rejected image group 1 as not running";
    }
    SECTION("the engine returns an unknown admission result")
    {
        start_provider();
        script->admission = static_cast<EnqueueResult>(99);
        provider->request_start();
        provider->update();
        camera->push(fake_frame(1));
        provider->update();
        reason = "rejected image group 1 with an unknown result";
    }
    SECTION("on_image_group() throws")
    {
        start_provider();
        script->throwing.insert("on_image_group");
        provider->request_start();
        provider->update();
        camera->push(fake_frame(1));
        provider->update();
        reason = "on_image_group: on_image_group failed";
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
    script->estimate_when_drained = 1;
    auto camera = std::make_shared<FakeCameraSource>();
    auto output = fake_output(script, camera);
    PoseProvider provider(output_config(), fake_engine(script), ready_input(camera), open_on(output));
    provider.request_start();
    provider.update();
    camera->push(fake_frame(1));
    provider.update();
    REQUIRE(provider.status().state == ProviderState::Running);
    provider.request_finish();
    provider.update();

    CHECK(provider.finished());
    // A provider that polled before reading input_drained() would finish with the estimate still queued.
    CHECK(script->estimates_taken == 1);
    CHECK(output->poses.size() == 1);
    CHECK_FALSE(provider.status().failed);
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
    config.output_collection = std::string(kMaxOutputCollectionBytes + 1, 'c');
    CHECK_THROWS_AS(PoseProvider(config, fake_engine(script)), std::invalid_argument);
    config.output_collection.pop_back();
    CHECK_NOTHROW(PoseProvider(config, fake_engine(script)));

    config = ProviderConfig{};
    config.startup_timeout = kMaxProviderTimeout + 1ms;
    CHECK_THROWS_AS(PoseProvider(config, fake_engine(script)), std::invalid_argument);
}

TEST_CASE("Running undistorts each frame into packed storage and offers it with the anchor's times",
          "[unit][pose_provider]")
{
    auto script = std::make_shared<Script>();
    auto camera = std::make_shared<FakeCameraSource>();
    SourceDescription description = fake_stereo_description();
    // Wide enough that its map moves samples by a visible fraction of a pixel at 8x4.
    description.right.intrinsics =
        Intrinsics{ 6.0, 6.06, 3.25, 1.75, DistortionModel::RadTan5, { -0.3, 0.05, 1e-3, -1e-3, 0.01 } };
    PoseProvider provider(ProviderConfig{}, fake_engine(script), ready_input(camera, description));
    provider.request_start();
    provider.update();
    REQUIRE(provider.status().state == ProviderState::Running);
    const std::array<FakeFrame, 2> frames{ padded_frame(1), padded_frame(2) };
    for (const FakeFrame& frame : frames)
    {
        camera->push(frame);
    }
    provider.update();

    // Both leases end in the update that took them, while the engine still holds the prepared storage.
    CHECK(camera->frames_released() == 2);
    REQUIRE(script->offered.size() == 2);
    CHECK(script->offered[0].group_id != 0);
    CHECK(script->offered[1].group_id > script->offered[0].group_id);

    // Each group keeps its own frame's pixels while the engine holds both.
    const std::array<Intrinsics, 2> lenses{ *description.left.intrinsics, *description.right.intrinsics };
    for (std::size_t g = 0; g < frames.size(); ++g)
    {
        CAPTURE(g);
        const EngineImageGroup& group = script->offered[g];
        const FakeFrame& frame = frames[g];
        CHECK(group.anchor.local_common_clock_ns == *frame.left.local_common_clock_ns);
        CHECK(group.anchor.raw_device_clock_ns == *frame.left.raw_device_clock_ns);
        CHECK(group.fusion_time_ns == *frame.left.local_common_clock_ns);
        CHECK(group.images.size() == 2);
        const std::array<const std::vector<std::byte>*, 2> delivered{ &frame.left_bytes, &frame.right_bytes };
        for (uint32_t i = 0; i < 2; ++i)
        {
            CAPTURE(i);
            const HostImage& image = image_of(group, StreamId{ i });
            CHECK(image.layout == ImageLayout{ 8, 4, PixelFormat::Gray8, 8 });
            CHECK(image.size_bytes == 32);
            const Intrinsics& lens = lenses[i];
            const UndistortMap map = make_undistort_map(
                CameraCalibration{ lens.fx, lens.fy, lens.cx, lens.cy, lens.distortion, lens.coefficients, {} },
                script->initialized->streams[i].calibration, 8, 4);
            std::vector<std::byte> expected(32);
            remap_gray8(map, delivered[i]->data(), kPaddedStride, expected.data());
            CHECK(levels(image.data, 32) == levels(expected.data(), 32));
        }
    }
    // The right lens has distortion, so its prepared pixels are not a copy of the delivered ones.
    CHECK(levels(image_of(script->offered[0], StreamId{ 1 }).data, 8) != levels(frames[0].right_bytes.data(), 8));
}

TEST_CASE("Only the left stream's times are checked", "[unit][pose_provider]")
{
    auto script = std::make_shared<Script>();
    auto camera = std::make_shared<FakeCameraSource>();
    PoseProvider provider(ProviderConfig{}, fake_engine(script), ready_input(camera));
    provider.request_start();
    provider.update();
    // The first frame's device time is 0. The right stream's times go back, then are absent and negative.
    std::array<FakeFrame, 3> frames{ fake_frame(0), fake_frame(1), fake_frame(2) };
    frames[1].right.local_common_clock_ns = 5;
    frames[1].right.raw_device_clock_ns = 5;
    frames[2].right.local_common_clock_ns.reset();
    frames[2].right.raw_device_clock_ns = -1;
    for (const FakeFrame& frame : frames)
    {
        camera->push(frame);
    }
    provider.update();

    CHECK_FALSE(provider.status().failed);
    REQUIRE(script->offered.size() == 3);
    for (std::size_t i = 0; i < frames.size(); ++i)
    {
        CHECK(script->offered[i].anchor.local_common_clock_ns == *frames[i].left.local_common_clock_ns);
        CHECK(script->offered[i].anchor.raw_device_clock_ns == *frames[i].left.raw_device_clock_ns);
    }
}

TEST_CASE("A hold keeps its prepared storage after the provider is destroyed", "[unit][pose_provider]")
{
    auto script = std::make_shared<Script>();
    auto camera = std::make_shared<FakeCameraSource>();
    std::optional<HostImage> kept;
    std::vector<int> prepared;
    {
        PoseProvider provider(ProviderConfig{}, fake_engine(script), ready_input(camera));
        provider.request_start();
        provider.update();
        camera->push(fake_frame(1));
        provider.update();
        REQUIRE(script->offered.size() == 1);
        kept = std::get<HostImage>(script->offered[0].images[0].storage);
        prepared = levels(kept->data, kept->size_bytes);
    }
    // A hold that does not own its storage reads freed memory here.
    CHECK(levels(kept->data, kept->size_bytes) == prepared);
    kept.reset();
}

TEST_CASE("A frame that cannot be prepared fails the run with InputPreparation, naming the stream",
          "[unit][pose_provider]")
{
    const FrameRejection rows[] = {
        { [](FakeCameraSource&, FakeFrame& f) { f.left.layout.format = PixelFormat::RGBA8; },
          "left eye 'fake/left': frame is not Gray8" },
        { [](FakeCameraSource&, FakeFrame& f) { f.left.layout.width = 4; },
          "left eye 'fake/left': frame is 4x4 with stride 8, not 8x4 with stride at least 8" },
        { [](FakeCameraSource&, FakeFrame& f) { f.right.layout.height = 3; },
          "right eye 'fake/right': frame is 8x3 with stride 8, not 8x4 with stride at least 8" },
        { [](FakeCameraSource&, FakeFrame& f) { f.left.layout.stride_bytes = 7; },
          "left eye 'fake/left': frame is 8x4 with stride 7, not 8x4 with stride at least 8" },
        { [](FakeCameraSource&, FakeFrame& f) { f.right_bytes.resize(31); },
          "right eye 'fake/right': frame has 31 readable bytes; its layout needs 32" },
        { [](FakeCameraSource&, FakeFrame& f) { f.left.local_common_clock_ns.reset(); },
          "left eye 'fake/left': frame lacks a common-clock or device-clock time" },
        { [](FakeCameraSource&, FakeFrame& f) { f.left.raw_device_clock_ns.reset(); },
          "left eye 'fake/left': frame lacks a common-clock or device-clock time" },
        { [](FakeCameraSource&, FakeFrame& f) { f.left.raw_device_clock_ns = -1; },
          "left eye 'fake/left': device-clock time -1 ns is negative" },
        { [](FakeCameraSource&, FakeFrame& f) { f.left.local_common_clock_ns = 1'033'333'333; },
          "left eye 'fake/left': frame times (common 1033333333 ns, device 66666666 ns) do not both follow the "
          "previous frame's (1033333333 ns, 33333333 ns)" },
        { [](FakeCameraSource&, FakeFrame& f) { f.left.raw_device_clock_ns = 33'333'333; },
          "left eye 'fake/left': frame times (common 1066666666 ns, device 33333333 ns) do not both follow the "
          "previous frame's (1033333333 ns, 33333333 ns)" },
        { [](FakeCameraSource& c, FakeFrame& f)
          {
              c.push(fake_frame(2));
              f.sequence = 3;
          },
          "left eye 'fake/left': frame times (common 1066666666 ns, device 66666666 ns) do not both follow the "
          "previous frame's (1066666666 ns, 66666666 ns)",
          2 },
        { [](FakeCameraSource&, FakeFrame& f) { f.sequence = 1; },
          "frame at capture number 1 does not follow the previous capture: expected 2 or later" },
        { [](FakeCameraSource&, FakeFrame& f) { f.sequence = 0; },
          "frame at capture number 0 does not follow the previous capture: expected 2 or later" },
        { [](FakeCameraSource& c, FakeFrame&) {
             c.push(FrameLoss{ 1, 2 });
         },
          "loss report at capture number 1 does not follow the previous capture: expected 2 or later" },
    };
    for (const FrameRejection& row : rows)
    {
        INFO(row.reason);
        auto script = std::make_shared<Script>();
        auto camera = std::make_shared<FakeCameraSource>();
        PoseProvider provider(ProviderConfig{}, fake_engine(script), ready_input(camera));
        provider.request_start();
        provider.update();
        camera->push(fake_frame(1));
        FakeFrame second = fake_frame(2);
        row.change(*camera, second);
        camera->push(std::move(second));
        provider.update();
        const ProviderStatus status = provider.status();
        CHECK(status.state == ProviderState::Finished);
        CHECK(status.failed_component == ProviderComponent::InputPreparation);
        CHECK(status.failure_reason == row.reason);
        CHECK(script->offered.size() == row.offered);
        CHECK(script->calls == Calls{ "validate_configuration", "initialize", "start", "stop" });
        CHECK(camera->frames_released() == camera->frames_delivered());
    }
}

TEST_CASE("Lost captures count from the first frame taken while Running, reported or not", "[unit][pose_provider]")
{
    auto script = std::make_shared<Script>();
    auto camera = std::make_shared<FakeCameraSource>();
    PoseProvider provider(ProviderConfig{}, fake_engine(script), ready_input(camera));
    // Before admission opens.
    camera->push(FrameLoss{ 1, 2 });
    camera->push(fake_frame(3));
    provider.update();
    REQUIRE(provider.status().state == ProviderState::Ready);
    provider.request_start();
    provider.update();
    REQUIRE(provider.status().state == ProviderState::Running);

    // Skipped while the engine was initializing, so reported before the first frame taken while Running.
    camera->push(FrameLoss{ 4, 5 });
    camera->push(fake_frame(9));
    camera->push(FrameLoss{ 11, 1 }); // 10 is lost without a report.
    camera->push(fake_frame(12));
    camera->push(fake_frame(14)); // 13 is lost without a report.
    camera->push(FrameLoss{ 15, 1 });
    provider.update();
    CHECK(provider.status().state == ProviderState::Running);
    CHECK(provider.status().frames_lost == 4);
    CHECK(script->offered.size() == 3);

    // After the finish request, neither a loss nor a gap counts.
    camera->push(FrameLoss{ 16, 3 });
    camera->push(fake_frame(20));
    provider.request_finish();
    provider.update();
    const ProviderStatus status = provider.status();
    CHECK(status.state == ProviderState::Finished);
    CHECK_FALSE(status.failed);
    CHECK(status.frames_lost == 4);
    CHECK(script->offered.size() == 3);
}

TEST_CASE("Full prepared storage drops whole frames until the engine releases a group", "[unit][pose_provider]")
{
    auto script = std::make_shared<Script>();
    auto camera = std::make_shared<FakeCameraSource>();
    PoseProvider provider(ProviderConfig{}, fake_engine(script), ready_input(camera));
    const std::size_t capacity = provider.status().prepared_image_groups;
    REQUIRE(capacity > 0);
    provider.request_start();
    provider.update();
    uint64_t sequence = 0;

    // A group the engine drops returns its storage when the call returns.
    script->admission = EnqueueResult::DroppedBackpressure;
    for (std::size_t i = 0; i <= capacity; ++i)
    {
        camera->push(fake_frame(++sequence));
    }
    provider.update();
    CHECK(script->offered.size() == capacity + 1);
    CHECK(provider.status().frames_dropped == 0);

    // The scripted engine holds accepted storage until it is released.
    script->admission = EnqueueResult::Accepted;
    for (std::size_t i = 0; i <= capacity; ++i)
    {
        camera->push(fake_frame(++sequence));
    }
    provider.update();
    CHECK(script->offered.size() == 2 * capacity + 1);
    CHECK(provider.status().frames_dropped == 1);
    std::set<const std::byte*> held;
    for (std::size_t i = capacity + 1; i < script->offered.size(); ++i)
    {
        for (const EngineImage& image : script->offered[i].images)
        {
            held.insert(std::get<HostImage>(image.storage).data);
        }
    }
    CHECK(held.size() == 2 * capacity);

    release(script->offered[capacity + 1]);
    camera->push(fake_frame(++sequence));
    provider.update();
    CHECK(script->offered.size() == 2 * capacity + 2);
    CHECK(provider.status().frames_dropped == 1);
    CHECK(provider.status().frames_lost == 0);
    CHECK_FALSE(provider.status().failed);
    CHECK(camera->frames_released() == camera->frames_delivered());
}

TEST_CASE("Estimates follow accepted-group order, at most one per accepted group", "[unit][pose_provider]")
{
    auto script = std::make_shared<Script>();
    auto camera = std::make_shared<FakeCameraSource>();
    auto output = fake_output(script, camera);
    PoseProvider provider(output_config(), fake_engine(script), ready_input(camera), open_on(output));
    provider.request_start();
    provider.update();
    // The first and last groups are accepted; the engine drops the middle one.
    for (const EnqueueResult admission :
         { EnqueueResult::Accepted, EnqueueResult::DroppedBackpressure, EnqueueResult::Accepted })
    {
        script->admission = admission;
        camera->push(fake_frame(script->offered.size() + 1));
        provider.update();
    }
    REQUIRE(script->offered.size() == 3);
    const uint64_t first = script->offered[0].group_id;
    const uint64_t dropped = script->offered[1].group_id;
    const uint64_t last = script->offered[2].group_id;
    std::optional<uint64_t> breaks; // The group whose estimate breaks the order.
    std::size_t published = 1; // Estimates published before any break.

    SECTION("an estimate for each accepted group")
    {
        script->estimates = { first, last };
        published = 2;
    }
    SECTION("an accepted group without an estimate")
    {
        script->estimates = { last };
    }
    SECTION("the dropped group")
    {
        script->estimates = { dropped };
        breaks = dropped;
        published = 0;
    }
    SECTION("a repeated group")
    {
        script->estimates = { first, first };
        breaks = first;
    }
    SECTION("an earlier group after a later one")
    {
        script->estimates = { last, first };
        breaks = first;
    }
    provider.update();

    const ProviderStatus status = provider.status();
    CHECK(script->estimates.empty());
    CHECK(output->poses.size() == published);
    CHECK(status.failed == breaks.has_value());
    if (breaks)
    {
        CHECK(status.failed_component == ProviderComponent::Engine);
        CHECK(status.failure_reason == "estimate for image group " + std::to_string(*breaks) +
                                           ", which is not an accepted group awaiting its estimate");
    }
}

TEST_CASE("Draining admits nothing after the finish request and ends engine input after the camera input's end",
          "[unit][pose_provider]")
{
    auto script = std::make_shared<Script>();
    auto camera = std::make_shared<FakeCameraSource>();
    PoseProvider provider(ProviderConfig{}, fake_engine(script), ready_input(camera));
    provider.request_start();
    provider.update();
    camera->push(fake_frame(1));
    provider.update();
    REQUIRE(script->offered.size() == 1);
    camera->withhold_end(true);
    camera->push(fake_frame(2));
    provider.request_finish();
    provider.update();

    CHECK(provider.status().state == ProviderState::Draining);
    CHECK(script->calls == Calls{ "validate_configuration", "initialize", "start" });
    CHECK(script->offered.size() == 1);
    CHECK(camera->frames_released() == 2);
    // Estimates are still taken while the camera input drains.
    script->estimates = { script->offered[0].group_id };
    provider.update();
    CHECK(script->estimates_taken == 1);
    CHECK(provider.status().state == ProviderState::Draining);
    CHECK_FALSE(provider.status().failed);

    camera->withhold_end(false);
    provider.update();
    CHECK(provider.finished());
    CHECK_FALSE(provider.status().failed);
    CHECK(script->calls == Calls{ "validate_configuration", "initialize", "start", "finish_input", "stop" });
    CHECK(camera->detach_calls() == 1);
}

TEST_CASE("A camera input that does not end within the drain timeout fails the run", "[unit][pose_provider]")
{
    auto script = std::make_shared<Script>();
    auto camera = std::make_shared<FakeCameraSource>();
    ProviderConfig config;
    config.drain_timeout = kShortTimeout;
    PoseProvider provider(config, fake_engine(script), ready_input(camera));
    provider.request_start();
    provider.update();
    camera->withhold_end(true);
    provider.request_finish();
    CHECK(update_until_finished(provider));

    const ProviderStatus status = provider.status();
    CHECK(status.failed_component == ProviderComponent::CameraInput);
    CHECK(status.failure_reason == "the camera input did not end within the drain timeout (50 ms)");
    CHECK(script->calls == Calls{ "validate_configuration", "initialize", "start", "stop" });
}

TEST_CASE("Each estimate is published with its anchor's times, valid only while tracking", "[unit][pose_provider]")
{
    auto script = std::make_shared<Script>();
    auto camera = std::make_shared<FakeCameraSource>();
    auto output = fake_output(script, camera);
    PoseProvider provider(output_config(), fake_engine(script), ready_input(camera), open_on(output));
    provider.request_start();
    provider.update();
    for (uint64_t sequence = 1; sequence <= 3; ++sequence)
    {
        camera->push(fake_frame(sequence));
        provider.update();
    }
    REQUIRE(script->offered.size() == 3);
    CHECK_FALSE(provider.status().latest_tracking_state.has_value());

    // A rotation whose components all differ, so that any reordering of x, y, z, w shows.
    script->pose = Transform{ { 1.5, -2.0, 0.25 }, { 0.1, 0.3, 0.5, std::sqrt(1.0 - 0.35) } };
    const std::array<TrackingState, 3> states{ TrackingState::Initializing, TrackingState::Tracking, TrackingState::Lost };
    int64_t before_last = 0;
    for (std::size_t i = 0; i < states.size(); ++i)
    {
        script->tracking_state = states[i];
        script->estimates = { script->offered[i].group_id };
        before_last = now_ns();
        provider.update();
    }
    const int64_t after_last = now_ns();

    REQUIRE(output->poses.size() == 3);
    for (std::size_t i = 0; i < states.size(); ++i)
    {
        const PublishedPose& sample = output->poses[i];
        const bool tracking = states[i] == TrackingState::Tracking;
        CHECK(sample.valid == tracking);
        CHECK(sample.times.local_common_clock_ns == script->offered[i].anchor.local_common_clock_ns);
        CHECK(sample.times.raw_device_clock_ns == script->offered[i].anchor.raw_device_clock_ns);
        const Transform expected = tracking ? script->pose : Transform{};
        for (std::size_t k = 0; k < 3; ++k)
        {
            CHECK(sample.position[k] == static_cast<float>(expected.translation_m[k]));
        }
        for (std::size_t k = 0; k < 4; ++k)
        {
            CHECK(sample.orientation[k] == static_cast<float>(expected.rotation_xyzw[k]));
        }
    }
    const ProviderStatus status = provider.status();
    CHECK_FALSE(status.failed);
    CHECK(status.latest_tracking_state == TrackingState::Lost);
    CHECK(status.last_valid_measurement_time_ns == script->offered[1].anchor.local_common_clock_ns);
    REQUIRE(status.last_publication_time_ns.has_value());
    CHECK(*status.last_publication_time_ns >= before_last);
    CHECK(*status.last_publication_time_ns <= after_last);
}

TEST_CASE("The output opens around the engine's initialization and closes after the camera input's release",
          "[unit][pose_provider]")
{
    auto script = std::make_shared<Script>();
    auto camera = std::make_shared<FakeCameraSource>();
    auto output = fake_output(script, camera);
    ProviderConfig config = output_config();
    Calls calls{ "open pose_root/device_status",
                 "validate_configuration",
                 "initialize",
                 "open pose/run",
                 "start",
                 "finish_input",
                 "stop",
                 "close pose/run",
                 "close pose_root/device_status" };
    bool aborts = false;

    SECTION("with a plugin root ID, the device status opens first and closes last")
    {
    }
    SECTION("an abort closes both")
    {
        aborts = true;
        calls = { "open pose_root/device_status",
                  "validate_configuration",
                  "initialize",
                  "open pose/run",
                  "start",
                  "stop",
                  "close pose/run",
                  "close pose_root/device_status" };
    }
    SECTION("without a plugin root ID, no device status")
    {
        config.plugin_root_id.clear();
        calls = { "validate_configuration", "initialize", "open pose/run", "start",
                  "finish_input",           "stop",       "close pose/run" };
    }
    SECTION("without an output collection, nothing opens")
    {
        config.output_collection.clear();
        calls = { "validate_configuration", "initialize", "start", "finish_input", "stop" };
    }
    PoseProvider provider(config, fake_engine(script), ready_input(camera), open_on(output));
    provider.request_start();
    provider.update();
    camera->push(fake_frame(1));
    provider.update();
    script->tracking_state = TrackingState::Tracking;
    script->estimates = { script->offered.at(0).group_id };
    provider.update();
    if (aborts)
    {
        provider.abort();
    }
    else
    {
        provider.request_finish();
        CHECK(update_until_finished(provider));
    }

    const bool publishes = !config.output_collection.empty();
    CHECK(provider.finished());
    CHECK(provider.status().failed == aborts);
    CHECK(script->calls == calls);
    CHECK(output->poses.size() == (publishes ? 1u : 0u));
    CHECK(provider.status().latest_tracking_state.has_value() == publishes);
}

TEST_CASE("An output that cannot open or publish fails the run with Output; the device status never does",
          "[unit][pose_provider]")
{
    auto script = std::make_shared<Script>();
    auto camera = std::make_shared<FakeCameraSource>();
    auto output = fake_output(script, camera);
    std::string reason;
    Calls calls{ "open pose_root/device_status",
                 "validate_configuration",
                 "initialize",
                 "open pose/run",
                 "start",
                 "finish_input",
                 "stop",
                 "close pose/run",
                 "close pose_root/device_status" };
    int taken = 2; // Estimates taken from the engine.
    bool reports_failure = true; // The final device status names the failure.

    SECTION("the plugin session cannot open")
    {
        output->failing_session = true;
        reason = "the plugin session or the device status could not open: cannot open the plugin session";
        calls = { "stop" };
        taken = 0;
        reports_failure = false;
    }
    SECTION("the plugin session factory returns no session")
    {
        output->null_session = true;
        reason = "the plugin session or the device status could not open: the plugin session factory returned no session";
        calls = { "stop" };
        taken = 0;
        reports_failure = false;
    }
    SECTION("the device status cannot open")
    {
        output->failing_open = { kStatusCollection };
        reason = "the plugin session or the device status could not open: cannot open pose_root/device_status";
        calls = { "stop" };
        taken = 0;
        reports_failure = false;
    }
    SECTION("the pose collection cannot open")
    {
        output->failing_open = { kPoseCollection };
        reason = "output collection 'pose/run' could not open: cannot open pose/run";
        calls = { "open pose_root/device_status", "validate_configuration", "initialize", "stop",
                  "close pose_root/device_status" };
        taken = 0;
    }
    SECTION("an estimate cannot be published")
    {
        output->failing_push = { kPoseCollection };
        reason = "estimate for image group 1 not published: cannot push to pose/run";
        calls = { "open pose_root/device_status",
                  "validate_configuration",
                  "initialize",
                  "open pose/run",
                  "start",
                  "stop",
                  "close pose/run",
                  "close pose_root/device_status" };
        taken = 1;
    }
    SECTION("the device status cannot be published")
    {
        output->failing_push = { kStatusCollection };
        // Dropping the status publisher closes its collection at once.
        calls = { "open pose_root/device_status",
                  "validate_configuration",
                  "initialize",
                  "open pose/run",
                  "start",
                  "close pose_root/device_status before the camera input's release",
                  "finish_input",
                  "stop",
                  "close pose/run" };
    }
    PoseProvider provider(output_config(), fake_engine(script), ready_input(camera), open_on(output));
    provider.request_start();
    provider.update();
    camera->push(fake_frame(1));
    camera->push(fake_frame(2));
    provider.update();
    script->tracking_state = TrackingState::Tracking;
    script->estimates = { 1, 2 };
    provider.update();
    provider.request_finish();
    CHECK(update_until_finished(provider));

    const ProviderStatus status = provider.status();
    CHECK(status.failed == !reason.empty());
    CHECK(script->calls == calls);
    CHECK(script->estimates_taken == taken);
    CHECK(camera->input_destroyed());
    if (status.failed)
    {
        CHECK(status.failed_component == ProviderComponent::Output);
        CHECK(status.failure_reason == reason);
        // Nothing is recorded for an estimate that was not published.
        CHECK(output->poses.empty());
        CHECK_FALSE(status.latest_tracking_state.has_value());
        CHECK_FALSE(status.last_publication_time_ns.has_value());
        if (reports_failure)
        {
            REQUIRE_FALSE(output->statuses.empty());
            check_status(output->statuses.back(), core::PluginDeviceState_FAILED, core::PluginDeviceReason_DEVICE_ERROR,
                         "Output: " + reason);
        }
        else
        {
            CHECK(output->statuses.empty());
        }
    }
    else
    {
        CHECK(output->poses.size() == 2);
        CHECK(output->statuses.empty());
    }
}

TEST_CASE("The device status follows the run under the manifest's device path", "[unit][pose_provider]")
{
    auto script = std::make_shared<Script>();
    auto camera = std::make_shared<FakeCameraSource>();
    auto output = fake_output(script, camera);
    PoseProvider provider(
        output_config(), fake_engine(script), std::make_unique<FakeCameraInput>(camera), open_on(output));
    provider.update();
    REQUIRE(provider.status().state == ProviderState::Preparing);
    REQUIRE(output->statuses.size() == 1);
    check_status(output->statuses[0], core::PluginDeviceState_UNKNOWN, core::PluginDeviceReason_NO_CURRENT_DATA);
    // The status reader rejects a report whose path the manifest does not declare.
    CHECK(read_file(POSE_PROVIDER_MANIFEST).find("path: \"/se3_tracker/pose_provider\"") != std::string::npos);
    // While the run waits for the camera input, an update() at least every heartbeat period keeps the report current.
    CHECK(provider.wait_timeout() > 0ms);
    CHECK(provider.wait_timeout() <= 1s);

    camera->push(fake_stereo_description());
    provider.update();
    REQUIRE(provider.status().state == ProviderState::Ready);
    check_status(output->statuses.back(), core::PluginDeviceState_CONNECTED, core::PluginDeviceReason_NONE);
    provider.request_start();
    provider.update();
    REQUIRE(provider.status().state == ProviderState::Running);
    check_status(output->statuses.back(), core::PluginDeviceState_CONNECTED, core::PluginDeviceReason_NONE);

    SECTION("a finished run")
    {
        script->drains = false;
        provider.request_finish();
        provider.update();
        REQUIRE(provider.status().state == ProviderState::Draining);
        check_status(output->statuses.back(), core::PluginDeviceState_CONNECTED, core::PluginDeviceReason_NONE);
        script->drains = true;
        CHECK(update_until_finished(provider));
        check_status(output->statuses.back(), core::PluginDeviceState_UNKNOWN, core::PluginDeviceReason_NO_CURRENT_DATA);
    }
    SECTION("a failed run")
    {
        script->fatal_reason = "tracker diverged";
        provider.update();
        REQUIRE(provider.finished());
        check_status(output->statuses.back(), core::PluginDeviceState_FAILED, core::PluginDeviceReason_DEVICE_ERROR,
                     "Engine: engine fatal error: tracker diverged");
    }
}
