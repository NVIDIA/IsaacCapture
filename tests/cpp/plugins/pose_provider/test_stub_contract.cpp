// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// IPoseEngine contract tests, run against the stub adapter through make_pose_engine().

#include <catch2/catch_test_macros.hpp>
#include <pose_engine_api/pose_engine.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <variant>
#include <vector>

using namespace pose_provider;
using namespace std::chrono_literals;

namespace
{

constexpr uint32_t kWidth = 8;
constexpr uint32_t kHeight = 4;
constexpr StreamId kLeft{ 10 };
constexpr StreamId kRight{ 11 };
constexpr auto kTimeout = 5s;
// Generous so that a loaded CI machine does not flake.
constexpr auto kNonBlocking = 200ms;

ImageLayout gray_layout()
{
    return ImageLayout{ kWidth, kHeight, PixelFormat::Gray8, kWidth };
}

ImuSetup imu_setup()
{
    ImuSetup imu;
    imu.rate_hz = 200.0;
    imu.accel_noise_density = 2e-3;
    imu.gyro_noise_density = 2e-4;
    imu.accel_random_walk = 3e-3;
    imu.gyro_random_walk = 2e-5;
    return imu;
}

ResolvedPoseSetup make_setup(const IPoseEngine& engine, InputMode mode, const std::string& settings = {})
{
    ResolvedPoseSetup setup;
    setup.input_mode = mode;
    setup.memory_kind = MemoryKind::Host;
    setup.body_frame = "left_optical";
    for (StreamId id : { kLeft, kRight })
    {
        StreamSetup stream;
        stream.id = id;
        stream.layout = gray_layout();
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
    if (mode == InputMode::StereoImu)
    {
        setup.imu = imu_setup();
    }
    setup.engine_settings = settings;
    setup.engine_identity = engine.identity();
    return setup;
}

std::unique_ptr<IPoseEngine> started_engine(InputMode mode, const std::string& settings = {})
{
    auto engine = make_pose_engine();
    engine->initialize(make_setup(*engine, mode, settings));
    engine->start();
    return engine;
}

// Counts image storage still held.
struct StorageLedger
{
    std::atomic<int> held{ 0 };
};

EngineImage make_image(StreamId stream, const std::shared_ptr<StorageLedger>& ledger)
{
    auto pixels = std::make_shared<std::vector<std::byte>>(kWidth * kHeight, std::byte{ 0x5a });
    ++ledger->held;
    HostImage host;
    host.layout = gray_layout();
    host.data = pixels->data();
    host.size_bytes = pixels->size();
    host.hold = std::shared_ptr<const void>(pixels->data(), [pixels, ledger](const void*) { --ledger->held; });
    return EngineImage{ stream, host };
}

AnchorTimes anchor_for(uint64_t group_id)
{
    const auto id = static_cast<int64_t>(group_id);
    return AnchorTimes{ id * 33'000'000 + 7, id * 1000 + 3 };
}

EngineImageGroup make_group(uint64_t id, const std::shared_ptr<StorageLedger>& ledger)
{
    EngineImageGroup group;
    group.group_id = id;
    group.fusion_time_ns = static_cast<int64_t>(id) * 33'000'000;
    group.anchor = anchor_for(id);
    group.images.push_back(make_image(kLeft, ledger));
    group.images.push_back(make_image(kRight, ledger));
    return group;
}

EngineImuSample make_imu(uint64_t id)
{
    EngineImuSample sample;
    sample.input_id = id;
    sample.fusion_time_ns = static_cast<int64_t>(id) * 5'000'000;
    sample.acceleration_mps2 = { 0.0, 0.0, 9.81 };
    return sample;
}

bool wait_until(const std::function<bool()>& condition, std::chrono::milliseconds timeout = kTimeout)
{
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!condition())
    {
        if (std::chrono::steady_clock::now() > deadline)
        {
            return false;
        }
        std::this_thread::sleep_for(1ms);
    }
    return true;
}

template <typename Call>
std::chrono::steady_clock::duration elapsed(Call&& call)
{
    const auto start = std::chrono::steady_clock::now();
    call();
    return std::chrono::steady_clock::now() - start;
}

// Polls estimates until input is drained, then takes whatever is still queued.
std::vector<EnginePose> collect_until_drained(IPoseEngine& engine)
{
    std::vector<EnginePose> poses;
    EnginePose pose;
    const bool drained = wait_until(
        [&]
        {
            while (engine.try_get_next_pose(pose))
            {
                poses.push_back(pose);
            }
            return engine.input_drained();
        });
    REQUIRE(drained);
    while (engine.try_get_next_pose(pose))
    {
        poses.push_back(pose);
    }
    return poses;
}

void require_not_decreased(const EngineHealth& before, const EngineHealth& after)
{
    CHECK(after.dropped_image_groups >= before.dropped_image_groups);
    CHECK(after.dropped_imu_samples >= before.dropped_imu_samples);
    CHECK(after.imu_gaps >= before.imu_gaps);
    CHECK(after.invalid_input_rejections >= before.invalid_input_rejections);
    CHECK(after.not_running_rejections >= before.not_running_rejections);
    CHECK(after.output_stalls >= before.output_stalls);
    CHECK((after.fatal_error || !before.fatal_error));
}

// Fills the one-slot pose queue and leaves the worker waiting to publish group 1.
void stall_output(IPoseEngine& engine, const std::shared_ptr<StorageLedger>& ledger)
{
    REQUIRE(engine.on_image_group(make_group(0, ledger)) == EnqueueResult::Accepted);
    REQUIRE(engine.on_image_group(make_group(1, ledger)) == EnqueueResult::Accepted);
    REQUIRE(wait_until([&] { return engine.health().output_stalls == 1; }));
}

} // namespace

TEST_CASE("Capabilities and identity are available before initialize()", "[unit][pose_provider]")
{
    auto engine = make_pose_engine();
    const EngineCapabilities capabilities = engine->capabilities();
    CHECK(capabilities.input_modes.contains(InputMode::Stereo));
    CHECK(capabilities.memory_kinds.contains(MemoryKind::Host));
    CHECK(capabilities.pixel_formats.contains(PixelFormat::Gray8));
    CHECK(capabilities.distortion_models.contains(DistortionModel::None));
    const EngineIdentity identity = engine->identity();
    CHECK_FALSE(identity.adapter.empty());
    CHECK_FALSE(identity.adapter_version.empty());
    CHECK_FALSE(identity.engine_library.empty());
    CHECK_FALSE(identity.engine_library_version.empty());
}

TEST_CASE("validate_configuration() accepts a valid setup and explains a rejected one", "[unit][pose_provider]")
{
    auto engine = make_pose_engine();
    CHECK(engine->validate_configuration(make_setup(*engine, InputMode::Stereo)).empty());
    CHECK(engine->validate_configuration(make_setup(*engine, InputMode::StereoImu)).empty());

    ResolvedPoseSetup setup = make_setup(*engine, InputMode::Stereo);
    SECTION("one stream")
    {
        setup.streams.pop_back();
    }
    SECTION("duplicate stream IDs")
    {
        setup.streams[1].id = setup.streams[0].id;
    }
    SECTION("stride below the row size")
    {
        setup.streams[0].layout.stride_bytes = kWidth - 1;
    }
    SECTION("coefficients missing for the distortion model")
    {
        setup.streams[0].calibration.distortion = DistortionModel::RadTan5;
    }
    SECTION("non-rigid camera extrinsic")
    {
        setup.streams[1].calibration.T_body_camera.rotation_xyzw = { 0.0, 0.0, 0.0, 2.0 };
    }
    SECTION("IMU calibration in Stereo mode")
    {
        setup.imu = imu_setup();
    }
    SECTION("StereoImu without IMU calibration")
    {
        setup.input_mode = InputMode::StereoImu;
    }
    SECTION("resolved for another engine")
    {
        setup.engine_identity.adapter = "another_adapter";
    }
    SECTION("unknown engine setting")
    {
        setup.engine_settings = "queue_depth=3";
    }
    CHECK_FALSE(engine->validate_configuration(setup).empty());
    CHECK_THROWS_AS(engine->initialize(setup), std::invalid_argument);
}

TEST_CASE("Input is admitted only while running", "[unit][pose_provider]")
{
    auto ledger = std::make_shared<StorageLedger>();
    auto engine = make_pose_engine();
    CHECK(engine->on_image_group(make_group(0, ledger)) == EnqueueResult::RejectedNotRunning);
    CHECK_THROWS_AS(engine->start(), std::logic_error);

    engine->initialize(make_setup(*engine, InputMode::StereoImu));
    CHECK_THROWS_AS(engine->initialize(make_setup(*engine, InputMode::StereoImu)), std::logic_error);
    CHECK_THROWS_AS(engine->finish_input(), std::logic_error);
    CHECK(engine->on_imu(make_imu(0)) == EnqueueResult::RejectedNotRunning);

    engine->start();
    CHECK(engine->on_imu(make_imu(1)) == EnqueueResult::Accepted);
    CHECK(engine->on_image_group(make_group(1, ledger)) == EnqueueResult::Accepted);
    EnginePose pose;
    REQUIRE(wait_until([&] { return engine->try_get_next_pose(pose); }));
    // All accepted input is processed, but more can arrive until finish_input().
    CHECK_FALSE(engine->input_drained());

    engine->finish_input();
    engine->finish_input();
    CHECK(engine->on_image_group(make_group(2, ledger)) == EnqueueResult::RejectedNotRunning);
    CHECK(engine->on_imu(make_imu(2)) == EnqueueResult::RejectedNotRunning);
    CHECK(collect_until_drained(*engine).empty());

    engine->stop();
    engine->stop();
    CHECK(engine->on_image_group(make_group(3, ledger)) == EnqueueResult::RejectedNotRunning);
    CHECK(engine->health().not_running_rejections == 5);
    CHECK(engine->input_drained());
    CHECK(ledger->held == 0);
}

TEST_CASE("stop() ends a partial startup, and repeated calls have no further effect", "[unit][pose_provider]")
{
    auto ledger = std::make_shared<StorageLedger>();
    auto engine = make_pose_engine();
    SECTION("never initialized")
    {
        engine->stop();
        engine->stop();
    }
    SECTION("initialized, never started")
    {
        engine->initialize(make_setup(*engine, InputMode::Stereo));
        engine->stop();
        engine->stop();
        CHECK_THROWS_AS(engine->start(), std::logic_error);
    }
    CHECK(engine->on_image_group(make_group(0, ledger)) == EnqueueResult::RejectedNotRunning);
    CHECK(ledger->held == 0);
}

TEST_CASE("Destruction without stop() ends the run and releases storage", "[unit][pose_provider]")
{
    auto ledger = std::make_shared<StorageLedger>();
    auto engine = started_engine(InputMode::Stereo);
    CHECK(engine->on_image_group(make_group(0, ledger)) == EnqueueResult::Accepted);
    engine.reset();
    CHECK(ledger->held == 0);
}

TEST_CASE("Input that does not match the resolved setup is rejected and counted", "[unit][pose_provider]")
{
    auto ledger = std::make_shared<StorageLedger>();
    auto engine = started_engine(InputMode::Stereo);
    CHECK(engine->on_imu(make_imu(0)) == EnqueueResult::RejectedInvalidInput);

    std::vector<EngineImageGroup> invalid;
    invalid.push_back(make_group(1, ledger));
    invalid.back().images.pop_back();
    invalid.push_back(make_group(2, ledger));
    invalid.back().images[1].stream = StreamId{ 99 };
    invalid.push_back(make_group(3, ledger));
    invalid.back().images[1].stream = kLeft;
    invalid.push_back(make_group(4, ledger));
    std::get<HostImage>(invalid.back().images[0].storage).layout.stride_bytes = kWidth + 1;
    invalid.push_back(make_group(5, ledger));
    std::get<HostImage>(invalid.back().images[0].storage).size_bytes = kWidth * kHeight - 1;
    invalid.push_back(make_group(6, ledger));
    std::get<HostImage>(invalid.back().images[1].storage).hold.reset();
    for (EngineImageGroup& group : invalid)
    {
        CHECK(engine->on_image_group(std::move(group)) == EnqueueResult::RejectedInvalidInput);
    }
    // Rejected storage is released once each call returns, without a read.
    CHECK(ledger->held == 0);
    CHECK(engine->health().invalid_input_rejections == 7);

    CHECK(engine->on_image_group(make_group(7, ledger)) == EnqueueResult::Accepted);
    engine->finish_input();
    const std::vector<EnginePose> poses = collect_until_drained(*engine);
    REQUIRE(poses.size() == 1);
    CHECK(poses[0].group_id == 7);
    CHECK_FALSE(engine->health().fatal_error);
}

TEST_CASE("Image and IMU input is processed in admission order, one estimate per group", "[unit][pose_provider]")
{
    auto ledger = std::make_shared<StorageLedger>();
    auto engine = started_engine(InputMode::StereoImu);
    // The stub's placeholder translation is (image groups processed before it, IMU samples processed before it, 0).
    EngineImuSample sample = make_imu(0);
    CHECK(engine->on_image_group(make_group(0, ledger)) == EnqueueResult::Accepted);
    CHECK(engine->on_imu(sample) == EnqueueResult::Accepted);
    sample = make_imu(1);
    CHECK(engine->on_imu(sample) == EnqueueResult::Accepted);
    CHECK(engine->on_image_group(make_group(1, ledger)) == EnqueueResult::Accepted);
    sample = make_imu(2);
    CHECK(engine->on_imu(sample) == EnqueueResult::Accepted);
    // The engine copied the sample, so the caller may reuse it at once.
    sample.acceleration_mps2.fill(std::numeric_limits<double>::quiet_NaN());
    CHECK(engine->on_image_group(make_group(2, ledger)) == EnqueueResult::Accepted);
    engine->finish_input();

    const std::vector<EnginePose> poses = collect_until_drained(*engine);
    REQUIRE(poses.size() == 3);
    const std::array<double, 3> imu_before{ 0.0, 2.0, 3.0 };
    for (uint64_t i = 0; i < poses.size(); ++i)
    {
        CHECK(poses[i].group_id == i);
        CHECK(poses[i].anchor.local_common_clock_ns == anchor_for(i).local_common_clock_ns);
        CHECK(poses[i].anchor.raw_device_clock_ns == anchor_for(i).raw_device_clock_ns);
        CHECK(poses[i].T_slamworld_body.translation_m[0] == static_cast<double>(i));
        CHECK(poses[i].T_slamworld_body.translation_m[1] == imu_before[i]);
    }
    const EngineHealth health = engine->health();
    CHECK_FALSE(health.fatal_error);
    CHECK(health.invalid_input_rejections == 0);
    CHECK(ledger->held == 0);
}

TEST_CASE("Leading groups report Initializing until SlamWorld is fixed", "[unit][pose_provider]")
{
    auto ledger = std::make_shared<StorageLedger>();
    auto engine = started_engine(InputMode::Stereo, "initializing_groups=2");
    for (uint64_t id = 0; id < 4; ++id)
    {
        REQUIRE(engine->on_image_group(make_group(id, ledger)) == EnqueueResult::Accepted);
    }
    engine->finish_input();
    const std::vector<EnginePose> poses = collect_until_drained(*engine);
    REQUIRE(poses.size() == 4);
    CHECK(poses[0].tracking_state == TrackingState::Initializing);
    CHECK(poses[1].tracking_state == TrackingState::Initializing);
    CHECK(poses[2].tracking_state == TrackingState::Tracking);
    CHECK(poses[3].tracking_state == TrackingState::Tracking);
}

TEST_CASE("try_get_next_pose() and health() never block", "[unit][pose_provider]")
{
    auto engine = started_engine(InputMode::Stereo);
    EnginePose pose;
    bool got = true;
    CHECK(elapsed([&] { got = engine->try_get_next_pose(pose); }) < kNonBlocking);
    CHECK_FALSE(got);
    CHECK(elapsed([&] { engine->health(); }) < kNonBlocking);
}

TEST_CASE("A stalled consumer stalls output without dropping an estimate", "[unit][pose_provider]")
{
    auto ledger = std::make_shared<StorageLedger>();
    auto engine = started_engine(InputMode::Stereo, "image_capacity=2;pose_capacity=1");
    stall_output(*engine, ledger);

    // Group 1 has been read but still counts against image_capacity until it is published.
    CHECK(engine->on_image_group(make_group(2, ledger)) == EnqueueResult::Accepted);
    CHECK(engine->on_image_group(make_group(3, ledger)) == EnqueueResult::DroppedBackpressure);
    CHECK(ledger->held == 2);

    std::this_thread::sleep_for(20ms);
    const EngineHealth stalled = engine->health();
    CHECK(stalled.output_stalls == 1);
    CHECK(stalled.dropped_image_groups == 1);
    CHECK(elapsed([&] { engine->health(); }) < kNonBlocking);
    EnqueueResult admission = EnqueueResult::Accepted;
    CHECK(elapsed([&] { admission = engine->on_image_group(make_group(4, ledger)); }) < kNonBlocking);
    CHECK(admission == EnqueueResult::DroppedBackpressure);

    engine->finish_input();
    const std::vector<EnginePose> poses = collect_until_drained(*engine);
    REQUIRE(poses.size() == 3);
    for (uint64_t i = 0; i < poses.size(); ++i)
    {
        CHECK(poses[i].group_id == i);
    }
    const EngineHealth after = engine->health();
    require_not_decreased(stalled, after);
    CHECK(after.dropped_image_groups == 2);
    CHECK(ledger->held == 0);
}

TEST_CASE("Consecutive IMU drops form one gap, which the stub reports as a fatal error", "[unit][pose_provider]")
{
    auto ledger = std::make_shared<StorageLedger>();
    auto engine = started_engine(InputMode::StereoImu, "imu_capacity=1;pose_capacity=1");
    stall_output(*engine, ledger);
    REQUIRE(engine->on_imu(make_imu(0)) == EnqueueResult::Accepted);
    CHECK(engine->on_imu(make_imu(1)) == EnqueueResult::DroppedBackpressure);
    CHECK(engine->on_imu(make_imu(2)) == EnqueueResult::DroppedBackpressure);
    const EngineHealth dropped = engine->health();
    CHECK(dropped.dropped_imu_samples == 2);
    CHECK(dropped.imu_gaps == 1);
    CHECK_FALSE(dropped.fatal_error);

    EnginePose pose;
    SECTION("delivered ahead of the next accepted sample")
    {
        // Polling frees the stub's worker thread to take sample 0; drops until then extend the same gap.
        uint64_t next_id = 3;
        REQUIRE(wait_until(
            [&]
            {
                engine->try_get_next_pose(pose);
                return engine->on_imu(make_imu(next_id++)) == EnqueueResult::Accepted;
            }));
        CHECK(engine->health().imu_gaps == 1);
        REQUIRE(wait_until([&] { return engine->health().fatal_error; }));

        // The failed run processes nothing more, but still releases storage it accepts.
        while (engine->try_get_next_pose(pose))
        {
        }
        CHECK(engine->on_image_group(make_group(9, ledger)) == EnqueueResult::Accepted);
        REQUIRE(wait_until([&] { return ledger->held == 0; }));
        CHECK_FALSE(wait_until([&] { return engine->try_get_next_pose(pose); }, 50ms));
        engine->finish_input();
        CHECK_FALSE(engine->input_drained());
    }
    SECTION("delivered when input finishes")
    {
        engine->finish_input();
        REQUIRE(wait_until(
            [&]
            {
                while (engine->try_get_next_pose(pose))
                {
                }
                return engine->health().fatal_error;
            }));
    }

    const EngineHealth failed = engine->health();
    CHECK_FALSE(failed.fatal_reason.empty());
    require_not_decreased(dropped, failed);
    engine->stop();
    const EngineHealth stopped = engine->health();
    CHECK(stopped.fatal_error);
    require_not_decreased(failed, stopped);
    CHECK(ledger->held == 0);
}

TEST_CASE("stop() without a consumer releases storage that was never read", "[unit][pose_provider]")
{
    auto ledger = std::make_shared<StorageLedger>();
    auto engine = started_engine(InputMode::Stereo, "image_capacity=4;pose_capacity=1");
    stall_output(*engine, ledger);
    REQUIRE(engine->on_image_group(make_group(2, ledger)) == EnqueueResult::Accepted);
    REQUIRE(engine->on_image_group(make_group(3, ledger)) == EnqueueResult::Accepted);
    CHECK(ledger->held == 4);

    engine->stop();
    CHECK(ledger->held == 0);
    CHECK_FALSE(engine->input_drained());
}

TEST_CASE("stop() during an output stall is not reported as drained", "[unit][pose_provider]")
{
    auto ledger = std::make_shared<StorageLedger>();
    auto engine = started_engine(InputMode::Stereo, "pose_capacity=1");
    stall_output(*engine, ledger);
    engine->finish_input();
    engine->stop();
    CHECK_FALSE(engine->input_drained());
}

TEST_CASE("input_drained() becomes true only after every storage read; queued estimates drain afterwards",
          "[unit][pose_provider]")
{
    auto ledger = std::make_shared<StorageLedger>();
    auto engine = started_engine(InputMode::Stereo, "pose_capacity=1");
    stall_output(*engine, ledger);
    REQUIRE(engine->on_image_group(make_group(2, ledger)) == EnqueueResult::Accepted);
    engine->finish_input();
    CHECK_FALSE(engine->input_drained());

    std::vector<EnginePose> poses;
    EnginePose pose;
    REQUIRE(wait_until(
        [&]
        {
            if (engine->input_drained())
            {
                return true;
            }
            if (engine->try_get_next_pose(pose))
            {
                poses.push_back(pose);
            }
            return false;
        }));
    CHECK(ledger->held == 0);
    while (engine->try_get_next_pose(pose))
    {
        poses.push_back(pose);
    }
    CHECK(poses.size() == 3);
}
