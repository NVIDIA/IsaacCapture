// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The camera input contract: StereoFrame's lease, and the event stream as the scripted fake delivers it.

#include "fake_camera_input.hpp"

#include <catch2/catch_test_macros.hpp>
#include <pose_camera_input/camera_input.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

using namespace pose_provider;
using namespace pose_provider::testing;

namespace
{

constexpr ImageLayout kGray{ 8, 4, PixelFormat::Gray8, 8 };

struct Fixture
{
    std::shared_ptr<FakeCameraSource> source = std::make_shared<FakeCameraSource>();
    std::unique_ptr<ICameraInput> input = std::make_unique<FakeCameraInput>(source);
};

template <typename T>
T take(ICameraInput& input)
{
    CameraEvent event;
    REQUIRE(input.poll(event));
    REQUIRE(std::holds_alternative<T>(event));
    return std::get<T>(std::move(event));
}

bool exhausted(ICameraInput& input)
{
    CameraEvent event;
    return !input.poll(event);
}

// A lease whose end is counted.
std::shared_ptr<const void> counted_lease(int& releases)
{
    return std::shared_ptr<const void>(nullptr, [&releases](std::nullptr_t) { ++releases; });
}

StreamImage eye_over(const std::vector<std::byte>& bytes, int64_t raw_ns)
{
    StreamImage eye;
    eye.layout = kGray;
    eye.data = bytes.data();
    eye.size_bytes = bytes.size();
    eye.raw_device_clock_ns = raw_ns;
    return eye;
}

} // namespace

TEST_CASE("A stereo frame ends its lease once and keeps its metadata", "[unit][pose_provider]")
{
    const std::vector<std::byte> bytes(32, std::byte{ 1 });
    int releases = 0;
    auto frame = std::make_unique<StereoFrame>(7, eye_over(bytes, 100), eye_over(bytes, 101), counted_lease(releases));
    CHECK(frame->left().data == bytes.data());

    SECTION("release()")
    {
        frame->release();
        CHECK(releases == 1);
        CHECK(frame->left().data == nullptr);
        CHECK(frame->right().size_bytes == 0);
        CHECK(frame->sequence() == 7);
        CHECK(frame->left().layout == kGray);
        CHECK(frame->right().raw_device_clock_ns == 101);
        frame->release();
        frame.reset();
        CHECK(releases == 1);
    }
    SECTION("destruction")
    {
        frame.reset();
        CHECK(releases == 1);
    }
    SECTION("move construction")
    {
        StereoFrame moved(std::move(*frame));
        CHECK(moved.left().data == bytes.data());
        CHECK(frame->left().data == nullptr);
        frame.reset();
        CHECK(releases == 0);
        moved.release();
        CHECK(releases == 1);
    }
    SECTION("move assignment ends the target's own lease first")
    {
        int other_releases = 0;
        StereoFrame other(8, eye_over(bytes, 200), eye_over(bytes, 201), counted_lease(other_releases));
        other = std::move(*frame);
        CHECK(other_releases == 1);
        CHECK(other.sequence() == 7);
        CHECK(other.left().data == bytes.data());
        CHECK(frame->left().data == nullptr);
        CHECK(releases == 0);
        frame.reset();
        other.release();
        CHECK(releases == 1);
    }
}

TEST_CASE("A new input has no event, and the description arrives first with its values", "[unit][pose_provider]")
{
    Fixture f;
    CHECK(exhausted(*f.input));
    CHECK_FALSE(readable(f.input->wait_fd()));

    f.source->push(fake_stereo_description());
    f.source->push(fake_frame(1));

    const auto description = take<SourceDescription>(*f.input);
    CHECK(description.session_id == "fake-session");
    CHECK(description.left.stream_id == "fake/left");
    CHECK(description.right.stream_id == "fake/right");
    CHECK(description.left.format == PixelFormat::Gray8);
    CHECK(description.left.shutter == Shutter::Global);
    REQUIRE(description.right.intrinsics.has_value());
    CHECK(description.right.intrinsics->cx == 3.5);
    REQUIRE(description.right.T_rig_camera.has_value());
    CHECK(description.right.T_rig_camera->translation_m[0] == 0.075);
    CHECK(take<StereoFrame>(*f.input).sequence() == 1);
}

TEST_CASE("The fake rejects a script that no input could deliver", "[unit][pose_provider]")
{
    Fixture f;
    CHECK_THROWS_AS(f.source->push(fake_frame(1)), std::logic_error);
    CHECK_THROWS_AS(f.source->push(FrameLoss{ 1, 1 }), std::logic_error);
    f.source->push(fake_stereo_description());
    CHECK_THROWS_AS(f.source->push(fake_stereo_description()), std::logic_error);

    SECTION("after a failure")
    {
        f.source->push(InputFailed{ "source lost" });
        CHECK_THROWS_AS(f.source->push(fake_frame(1)), std::logic_error);
        CHECK_THROWS_AS(f.source->push(InputFailed{ "again" }), std::logic_error);
    }
    SECTION("after an unrequested end")
    {
        f.source->push_end();
        CHECK_THROWS_AS(f.source->push(FrameLoss{ 1, 1 }), std::logic_error);
        CHECK_THROWS_AS(f.source->push_end(), std::logic_error);
    }
    SECTION("data after detach() is dropped, and its order still checked")
    {
        f.input->detach();
        f.source->push(fake_frame(1));
        CHECK_THROWS_AS(f.source->push(fake_stereo_description()), std::logic_error);
        take<SourceDescription>(*f.input);
        CHECK(take<InputEnded>(*f.input).requested);
        CHECK_NOTHROW(f.source->push(fake_frame(2)));
        CHECK(exhausted(*f.input));
    }
}

TEST_CASE("Each event is delivered once, in the order queued", "[unit][pose_provider]")
{
    Fixture f;
    f.source->push(fake_stereo_description());
    f.source->push(fake_frame(1));
    f.source->push(FrameLoss{ 2, 3 });
    f.source->push(fake_frame(5));

    take<SourceDescription>(*f.input);
    CHECK(take<StereoFrame>(*f.input).sequence() == 1);
    const auto loss = take<FrameLoss>(*f.input);
    CHECK(loss.first_sequence == 2);
    CHECK(loss.count == 3);
    CHECK(take<StereoFrame>(*f.input).sequence() == 5);
    CHECK(exhausted(*f.input));
    CHECK(f.source->frames_delivered() == 2);
    CHECK(f.source->frames_released() == 2);
}

TEST_CASE("A frame stays readable until released, even after its input is destroyed", "[unit][pose_provider]")
{
    Fixture f;
    f.source->push(fake_stereo_description());
    FakeFrame scripted = fake_frame(1);
    scripted.right.raw_device_clock_ns.reset();
    scripted.left_bytes.assign(32, std::byte{ 7 });
    scripted.right_bytes.assign(32, std::byte{ 9 });
    f.source->push(std::move(scripted));
    take<SourceDescription>(*f.input);

    auto frame = take<StereoFrame>(*f.input);
    CHECK(frame.left().layout == kGray);
    REQUIRE(frame.left().size_bytes == 32);
    CHECK(frame.left().data[31] == std::byte{ 7 });
    CHECK(frame.right().data[0] == std::byte{ 9 });
    CHECK(frame.left().raw_device_clock_ns == 33'333'333);
    CHECK_FALSE(frame.right().raw_device_clock_ns.has_value());

    f.input.reset();
    CHECK(f.source->input_destroyed());
    CHECK(frame.right().data[31] == std::byte{ 9 });
    CHECK(f.source->frames_released() == 0);
    frame.release();
    CHECK(f.source->frames_released() == 1);
}

TEST_CASE("detach() delivers what was accepted, then InputEnded", "[unit][pose_provider]")
{
    Fixture f;
    SECTION("after the description")
    {
        f.source->push(fake_stereo_description());
        f.source->push(fake_frame(1));
        f.input->detach();
        f.source->push(fake_frame(2));

        take<SourceDescription>(*f.input);
        CHECK(take<StereoFrame>(*f.input).sequence() == 1);
        CHECK(take<InputEnded>(*f.input).requested);
    }
    SECTION("before the description")
    {
        f.input->detach();
        f.source->push(fake_stereo_description());
        CHECK(take<InputEnded>(*f.input).requested);
    }
    SECTION("the source ends without detach()")
    {
        f.source->push(fake_stereo_description());
        f.source->push_end();
        f.input->detach();
        take<SourceDescription>(*f.input);
        CHECK_FALSE(take<InputEnded>(*f.input).requested);
    }
    CHECK(exhausted(*f.input));
    f.input->detach();
    CHECK(exhausted(*f.input));
    CHECK(f.source->detach_calls() == 2);
}

TEST_CASE("InputFailed is the last event", "[unit][pose_provider]")
{
    Fixture f;
    SECTION("after frames")
    {
        f.source->push(fake_stereo_description());
        f.source->push(fake_frame(1));
        f.source->push(InputFailed{ "source lost" });
        take<SourceDescription>(*f.input);
        take<StereoFrame>(*f.input);
    }
    SECTION("instead of the description")
    {
        f.source->push(InputFailed{ "source lost" });
    }
    SECTION("during the drain after detach()")
    {
        f.source->push(fake_stereo_description());
        f.input->detach();
        f.source->push(InputFailed{ "source lost" });
        take<SourceDescription>(*f.input);
    }
    CHECK(take<InputFailed>(*f.input).reason == "source lost");
    CHECK(exhausted(*f.input));
    f.input->detach();
    CHECK(exhausted(*f.input));
    CHECK_FALSE(readable(f.input->wait_fd()));
}

TEST_CASE("The wait descriptor is readable exactly while an event is waiting", "[unit][pose_provider]")
{
    Fixture f;
    const int fd = f.input->wait_fd();
    CHECK_FALSE(readable(fd));

    f.source->push(fake_stereo_description());
    f.source->push(fake_frame(1));
    CHECK(readable(fd));
    take<SourceDescription>(*f.input);
    CHECK(readable(fd));
    take<StereoFrame>(*f.input);
    CHECK_FALSE(readable(fd));
    CHECK(exhausted(*f.input));
    CHECK_FALSE(readable(fd));

    f.input->detach();
    CHECK(readable(fd));
    take<InputEnded>(*f.input);
    CHECK_FALSE(readable(fd));
}
