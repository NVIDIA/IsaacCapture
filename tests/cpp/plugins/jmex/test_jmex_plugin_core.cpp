// SPDX-FileCopyrightText: Copyright (c) 2026 j-mex. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Unit tests for the jmex plugin's SDK-free parts. The name table is checked for the invariants
// the mapping relies on, not row by row.

#include "openxr_convention.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <flatbuffers/flatbuffers.h>
#include <jmex/body_joint_mapping.hpp>
#include <jmex/device_clock.hpp>
#include <jmex/full_body_publisher.hpp>
#include <jmex/receiver_session.hpp>
#include <oxr_utils/os_time.hpp>
#include <schema/full_body_generated.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

using Catch::Approx;
using Catch::Matchers::ContainsSubstring;
using namespace plugins::jmex;

namespace
{

constexpr size_t k_slot_count = core::BodyJoint_NUM_JOINTS;

//! The joint table of a channel carrying a joint for every slot, in slot order.
std::vector<std::string> complete_skeleton()
{
    std::vector<std::string> names;
    for (const std::string_view name : k_standard_human_skeleton_joint_names)
    {
        names.emplace_back(name);
    }
    return names;
}

//! A frame in which every joint carries a distinct, valid pose: position (i, 2i, 3i), identity
//! rotation. The values only have to be distinguishable.
std::vector<JointPose> frame_for(size_t joint_count)
{
    std::vector<JointPose> poses;
    for (size_t i = 0; i < joint_count; ++i)
    {
        const float f = static_cast<float>(i);
        poses.push_back(JointPose{ { f, 2.0f * f, 3.0f * f }, { 0.0f, 0.0f, 0.0f, 1.0f } });
    }
    return poses;
}

const core::BodyJointPose& slot_of(const core::BodyJoints& joints, core::BodyJoint slot)
{
    return *joints.joints()->Get(slot);
}

size_t valid_count(const core::BodyJoints& joints)
{
    size_t valid = 0;
    for (flatbuffers::uoffset_t slot = 0; slot < joints.joints()->size(); ++slot)
    {
        valid += joints.joints()->Get(slot)->is_valid() ? 1 : 0;
    }
    return valid;
}

//! A frame as ReceiverSession hands it out, filled from frame_for(): slot i carries joint i. Frames
//! are told apart by their Player time, which the device clock carries through.
FullBodyFrame frame_at(uint64_t player_time_ns)
{
    FullBodyFrame frame;
    BodyJointMapping::resolve(complete_skeleton()).apply(frame_for(k_slot_count), frame.joints);
    frame.all_joint_poses_tracked = true;
    frame.player_time_ns = player_time_ns;
    return frame;
}

//! One sample as the transport received it.
struct PushedSample
{
    std::vector<uint8_t> buffer;
    int64_t sample_time_ns = 0;
    int64_t raw_device_time_ns = 0;
};

/*!
 * @brief Stands in for the CloudXR runtime behind the publisher's transport.
 *
 * Records each transport the publisher asks for, in order. After restart() the transports already
 * up refuse every push, as a restarted runtime's do; after go_down() no new one comes up either.
 */
class FakeRuntime
{
public:
    struct Transport
    {
        bool wait_for_system = false;
        //! How many of the publisher's transports were still up when it asked for this one.
        int others_up = 0;
        bool came_up = false;
        bool refuses_pushes = false;
        std::vector<PushedSample> samples;
    };

    FullBodyPublisher::TransportFactory factory()
    {
        return [this](bool wait_for_system) -> FullBodyPublisher::PushBufferFunction
        {
            transports_.push_back(Transport{ .wait_for_system = wait_for_system, .others_up = up_ });
            if (down_)
            {
                throw std::runtime_error("the runtime is down");
            }
            transports_.back().came_up = true;

            const size_t index = transports_.size() - 1;
            // Counted as up until the publisher lets go of its push function.
            const auto up = std::make_shared<UpWhileHeld>(up_);
            return
                [this, index, up](const uint8_t* buffer, size_t size, int64_t sample_time_ns, int64_t raw_device_time_ns)
            {
                Transport& transport = transports_[index];
                if (transport.refuses_pushes)
                {
                    throw std::runtime_error("the runtime restarted");
                }
                transport.samples.push_back(PushedSample{ { buffer, buffer + size }, sample_time_ns, raw_device_time_ns });
            };
        };
    }

    void restart()
    {
        for (Transport& transport : transports_)
        {
            transport.refuses_pushes = true;
        }
    }

    void go_down()
    {
        restart();
        down_ = true;
    }

    const std::vector<Transport>& transports() const
    {
        return transports_;
    }

private:
    struct UpWhileHeld
    {
        explicit UpWhileHeld(int& up) : up_(up)
        {
            ++up_;
        }
        ~UpWhileHeld()
        {
            --up_;
        }
        UpWhileHeld(const UpWhileHeld&) = delete;
        UpWhileHeld& operator=(const UpWhileHeld&) = delete;

        int& up_;
    };

    std::vector<Transport> transports_;
    int up_ = 0;
    bool down_ = false;
};

} // namespace

TEST_CASE("openxr convention: MOXI axes become OpenXR axes", "[unit][jmex][convention]")
{
    const float identity[4] = { 1.0f, 0.0f, 0.0f, 0.0f };

    SECTION("each MOXI axis lands on the OpenXR axis that means the same thing")
    {
        const float left[3] = { 1.0f, 0.0f, 0.0f };
        const float behind[3] = { 0.0f, 1.0f, 0.0f };
        const float up[3] = { 0.0f, 0.0f, 1.0f };

        const JointPose to_left = moxi_pose_to_openxr(left, identity);
        REQUIRE(to_left.position[0] == Approx(-1.0f)); // the operator's left is OpenXR -X
        REQUIRE(to_left.position[1] == Approx(0.0f));
        REQUIRE(to_left.position[2] == Approx(0.0f));

        const JointPose to_behind = moxi_pose_to_openxr(behind, identity);
        REQUIRE(to_behind.position[0] == Approx(0.0f));
        REQUIRE(to_behind.position[1] == Approx(0.0f));
        REQUIRE(to_behind.position[2] == Approx(1.0f)); // behind is OpenXR +Z, forward being -Z

        const JointPose to_up = moxi_pose_to_openxr(up, identity);
        REQUIRE(to_up.position[0] == Approx(0.0f));
        REQUIRE(to_up.position[1] == Approx(1.0f)); // up is OpenXR +Y
        REQUIRE(to_up.position[2] == Approx(0.0f));
    }

    SECTION("the map is a rotation, not a reflection")
    {
        // Both frames are right-handed, so the image of a right-handed triad must stay one:
        // x_image cross y_image == z_image. A single flipped sign would mirror every pose --
        // left and right limbs swapped -- and nothing downstream would report an error.
        const float x[3] = { 1.0f, 0.0f, 0.0f };
        const float y[3] = { 0.0f, 1.0f, 0.0f };
        const float z[3] = { 0.0f, 0.0f, 1.0f };

        const JointPose ix = moxi_pose_to_openxr(x, identity);
        const JointPose iy = moxi_pose_to_openxr(y, identity);
        const JointPose iz = moxi_pose_to_openxr(z, identity);

        const float cross[3] = {
            ix.position[1] * iy.position[2] - ix.position[2] * iy.position[1],
            ix.position[2] * iy.position[0] - ix.position[0] * iy.position[2],
            ix.position[0] * iy.position[1] - ix.position[1] * iy.position[0],
        };
        REQUIRE(cross[0] == Approx(iz.position[0]));
        REQUIRE(cross[1] == Approx(iz.position[1]));
        REQUIRE(cross[2] == Approx(iz.position[2]));
    }

    SECTION("the quaternion is reordered scalar-last and its axis carried by the same map")
    {
        // A quarter turn about MOXI's up axis, scalar-first as the SDK answers.
        const float half = 0.7071068f;
        const float quarter_turn_about_up[4] = { half, 0.0f, 0.0f, half };
        const float origin[3] = { 0.0f, 0.0f, 0.0f };

        const JointPose pose = moxi_pose_to_openxr(origin, quarter_turn_about_up);

        REQUIRE(pose.orientation_xyzw[0] == Approx(0.0f));
        REQUIRE(pose.orientation_xyzw[1] == Approx(half)); // the axis moved to OpenXR +Y, as up does
        REQUIRE(pose.orientation_xyzw[2] == Approx(0.0f));
        REQUIRE(pose.orientation_xyzw[3] == Approx(half)); // the scalar, unchanged and now last
    }

    SECTION("a rotation about MOXI's lateral axis keeps its angle and flips its axis")
    {
        // The lateral axis is the only one whose sign flips.
        const float half = 0.7071068f;
        const float quarter_turn_about_left[4] = { half, half, 0.0f, 0.0f };
        const float origin[3] = { 0.0f, 0.0f, 0.0f };

        const JointPose pose = moxi_pose_to_openxr(origin, quarter_turn_about_left);

        REQUIRE(pose.orientation_xyzw[0] == Approx(-half));
        REQUIRE(pose.orientation_xyzw[1] == Approx(0.0f));
        REQUIRE(pose.orientation_xyzw[2] == Approx(0.0f));
        REQUIRE(pose.orientation_xyzw[3] == Approx(half));
    }
}

TEST_CASE("body joint mapping: the name table covers all 24 slots exactly once", "[unit][jmex][mapping]")
{
    std::set<std::string_view> seen;
    for (const std::string_view name : k_standard_human_skeleton_joint_names)
    {
        INFO("joint " << name);
        REQUIRE_FALSE(name.empty());
        // A repeated name would leave one slot unfilled while another was contested.
        REQUIRE(seen.insert(name).second);
    }

    // Resolving is the table read backwards; the two must agree for every slot.
    const BodyJointMapping mapping = BodyJointMapping::resolve(complete_skeleton());
    for (size_t slot = 0; slot < k_slot_count; ++slot)
    {
        INFO("slot " << slot);
        REQUIRE(mapping.joint_index(static_cast<core::BodyJoint>(slot)) == static_cast<int>(slot));
    }
}

TEST_CASE("body joint mapping: a name the table does not carry fills no slot", "[unit][jmex][mapping]")
{
    const BodyJointMapping mapping =
        BodyJointMapping::resolve({ "left_shoulder_pitch_joint", "left_thumb_proximal_joint", "" });

    REQUIRE(mapping.filled_slot_count() == 0);
    REQUIRE(mapping.contested_joint_names().empty());
}

TEST_CASE("body joint mapping: a model carrying every joint fills every slot", "[unit][jmex][mapping]")
{
    const std::vector<std::string> names = complete_skeleton();
    const BodyJointMapping mapping = BodyJointMapping::resolve(names);

    REQUIRE(mapping.filled_slot_count() == k_slot_count);
    REQUIRE(mapping.contested_joint_names().empty());

    core::BodyJoints joints;
    REQUIRE(mapping.apply(frame_for(names.size()), joints) == k_slot_count);

    REQUIRE(valid_count(joints) == k_slot_count);
    // complete_skeleton() is in slot order, so slot i carries joint i's position (i, 2i, 3i).
    const core::Point& head = slot_of(joints, core::BodyJoint_HEAD).pose().position();
    REQUIRE(head.x() == Approx(static_cast<float>(core::BodyJoint_HEAD)));
    REQUIRE(head.y() == Approx(2.0f * static_cast<float>(core::BodyJoint_HEAD)));
}

TEST_CASE("body joint mapping: joints the model does not carry leave their slots invalid", "[unit][jmex][mapping]")
{
    std::vector<std::string> names = complete_skeleton();
    names[core::BodyJoint_LEFT_HAND] = "something_else";

    const BodyJointMapping mapping = BodyJointMapping::resolve(names);

    REQUIRE(mapping.filled_slot_count() == k_slot_count - 1);
    REQUIRE(mapping.joint_index(core::BodyJoint_LEFT_HAND) == -1);

    core::BodyJoints joints;
    REQUIRE(mapping.apply(frame_for(names.size()), joints) == k_slot_count - 1);

    REQUIRE_FALSE(slot_of(joints, core::BodyJoint_LEFT_HAND).is_valid());
    REQUIRE(slot_of(joints, core::BodyJoint_RIGHT_HAND).is_valid());
    REQUIRE(valid_count(joints) == k_slot_count - 1);
}

TEST_CASE("body joint mapping: a second joint claiming a filled slot is reported, not swallowed", "[unit][jmex][mapping]")
{
    std::vector<std::string> names = complete_skeleton();
    const std::string contested(k_standard_human_skeleton_joint_names[core::BodyJoint_NECK]);
    names.push_back(contested);

    const BodyJointMapping mapping = BodyJointMapping::resolve(names);

    // The earlier joint keeps the slot.
    REQUIRE(mapping.joint_index(core::BodyJoint_NECK) == static_cast<int>(core::BodyJoint_NECK));
    REQUIRE(mapping.filled_slot_count() == k_slot_count);
    REQUIRE(mapping.contested_joint_names() == std::vector<std::string>{ contested });
}

TEST_CASE("body joint mapping: a bad frame costs its own slots and nothing else", "[unit][jmex][mapping]")
{
    const std::vector<std::string> names = complete_skeleton();
    const BodyJointMapping mapping = BodyJointMapping::resolve(names);
    core::BodyJoints joints;

    SECTION("a joint whose position is not a number")
    {
        std::vector<JointPose> frame = frame_for(names.size());
        frame[core::BodyJoint_LEFT_ELBOW].position[1] = std::numeric_limits<float>::quiet_NaN();

        mapping.apply(frame, joints);

        REQUIRE_FALSE(slot_of(joints, core::BodyJoint_LEFT_ELBOW).is_valid());
        REQUIRE(valid_count(joints) == k_slot_count - 1);
    }

    SECTION("a joint whose quaternion is not a rotation")
    {
        std::vector<JointPose> frame = frame_for(names.size());
        frame[core::BodyJoint_RIGHT_KNEE].orientation_xyzw[3] = 0.0f; // all four components now zero

        mapping.apply(frame, joints);

        REQUIRE_FALSE(slot_of(joints, core::BodyJoint_RIGHT_KNEE).is_valid());
        REQUIRE(valid_count(joints) == k_slot_count - 1);
    }

    SECTION("float round-off in the quaternion is not a bad frame")
    {
        std::vector<JointPose> frame = frame_for(names.size());
        frame[core::BodyJoint_PELVIS].orientation_xyzw[3] = 1.0f + 1e-5f;

        mapping.apply(frame, joints);

        REQUIRE(slot_of(joints, core::BodyJoint_PELVIS).is_valid());
    }

    SECTION("a frame that carried nothing invalidates every slot and ends no pairing")
    {
        REQUIRE(mapping.apply(std::vector<JointPose>{}, joints) == 0);

        REQUIRE(valid_count(joints) == 0);

        // The mapping holds no per-frame state, so the next frame fills every slot again.
        mapping.apply(frame_for(names.size()), joints);
        REQUIRE(valid_count(joints) == k_slot_count);
    }
}

TEST_CASE("device clock: device carries the Player time unchanged", "[unit][jmex][device_clock]")
{
    // Unix epoch nanoseconds on Player's machine: 2026-10-07 04:20:00.123456789 UTC.
    const uint64_t player_time_ns = 1791346800123456789;

    // Whatever the local time the frame was picked up at.
    REQUIRE(raw_device_time_ns(DeviceClock::DEVICE, 123456789012345, player_time_ns) == 1791346800123456789);
    REQUIRE(raw_device_time_ns(DeviceClock::DEVICE, 987654321098765, player_time_ns) == 1791346800123456789);
}

TEST_CASE("device clock: local carries the sample time unchanged", "[unit][jmex][device_clock]")
{
    // Both clocks carry the same value, whatever Player stamped on the frame.
    REQUIRE(raw_device_time_ns(DeviceClock::LOCAL, 123456789012345, 1791346800123456789) == 123456789012345);
    REQUIRE(raw_device_time_ns(DeviceClock::LOCAL, 123456789012345, 0) == 123456789012345);
}

TEST_CASE("device clock: the flag wins over the environment, which wins over the default", "[unit][jmex][device_clock]")
{
    REQUIRE(choose_device_clock(nullptr, nullptr) == DeviceClock::DEVICE);
    REQUIRE(choose_device_clock(nullptr, "local") == DeviceClock::LOCAL);
    REQUIRE(choose_device_clock("local", nullptr) == DeviceClock::LOCAL);
    REQUIRE(choose_device_clock("device", "local") == DeviceClock::DEVICE);
    REQUIRE(choose_device_clock("local", "device") == DeviceClock::LOCAL);

    // An environment variable set to nothing is how a shell unsets one for a single command.
    REQUIRE(choose_device_clock(nullptr, "") == DeviceClock::DEVICE);
}

TEST_CASE("device clock: a value it cannot read stops the plugin and says where it came from",
          "[unit][jmex][device_clock]")
{
    REQUIRE_THROWS_WITH(choose_device_clock("Device", nullptr), ContainsSubstring("--device-clock="));
    REQUIRE_THROWS_WITH(choose_device_clock("", nullptr), ContainsSubstring("--device-clock="));
    REQUIRE_THROWS_WITH(choose_device_clock(nullptr, "loc"), ContainsSubstring("JMEX_DEVICE_CLOCK"));
    REQUIRE_THROWS_WITH(choose_device_clock("sequence", nullptr), ContainsSubstring("--device-clock="));
    REQUIRE_THROWS_WITH(choose_device_clock(nullptr, "sequence"), ContainsSubstring("JMEX_DEVICE_CLOCK"));

    // Even when the flag would have won: a misspelt environment variable is left in place for the
    // next launch, which has no flag to cover for it.
    REQUIRE_THROWS_WITH(choose_device_clock("local", "dev"), ContainsSubstring("JMEX_DEVICE_CLOCK"));
}

TEST_CASE("device clock: the startup line names it the way the flag spells it", "[unit][jmex][device_clock]")
{
    REQUIRE(std::string(device_clock_name(DeviceClock::DEVICE)) == "device");
    REQUIRE(std::string(device_clock_name(DeviceClock::LOCAL)) == "local");

    // What the log says can be pasted back in.
    for (DeviceClock clock : { DeviceClock::DEVICE, DeviceClock::LOCAL })
    {
        REQUIRE(choose_device_clock(device_clock_name(clock), nullptr) == clock);
    }
}

TEST_CASE("full body publisher: a frame goes out as one FullBodyPose", "[unit][jmex][publisher]")
{
    FakeRuntime runtime;
    const std::atomic<bool> stop{ false };
    // Unix epoch nanoseconds on Player's machine, nothing like what the local clock reads.
    const uint64_t player_time_ns = 1791346800123456789;

    SECTION("with the device clock, the sample carries the frame's Player time")
    {
        FullBodyPublisher publisher(runtime.factory(), DeviceClock::DEVICE);

        // Up before the first frame, and brought up the one way that waits for the OpenXR system.
        REQUIRE(runtime.transports().size() == 1);
        CHECK(runtime.transports()[0].wait_for_system);

        const int64_t before_ns = core::os_monotonic_now_ns();
        CHECK(publisher.publish(frame_at(player_time_ns), stop));
        const int64_t after_ns = core::os_monotonic_now_ns();

        REQUIRE(runtime.transports()[0].samples.size() == 1);
        const PushedSample& sample = runtime.transports()[0].samples[0];

        flatbuffers::Verifier verifier(sample.buffer.data(), sample.buffer.size());
        REQUIRE(verifier.VerifyBuffer<core::FullBodyPose>(nullptr));
        const core::FullBodyPose& pose = *flatbuffers::GetRoot<core::FullBodyPose>(sample.buffer.data());

        CHECK(pose.all_joint_poses_tracked());
        REQUIRE(pose.joints() != nullptr);
        for (size_t i = 0; i < k_slot_count; ++i)
        {
            INFO("slot " << i);
            const core::BodyJointPose& slot = slot_of(*pose.joints(), static_cast<core::BodyJoint>(i));
            const float f = static_cast<float>(i);
            CHECK(slot.is_valid());
            CHECK(slot.pose().position().x() == f);
            CHECK(slot.pose().position().y() == 2.0f * f);
            CHECK(slot.pose().position().z() == 3.0f * f);
            CHECK(slot.pose().orientation().x() == 0.0f);
            CHECK(slot.pose().orientation().y() == 0.0f);
            CHECK(slot.pose().orientation().z() == 0.0f);
            CHECK(slot.pose().orientation().w() == 1.0f);
        }

        // Picked up while publish() ran, on the clock every source on this machine shares.
        CHECK(sample.sample_time_ns >= before_ns);
        CHECK(sample.sample_time_ns <= after_ns);
        CHECK(sample.raw_device_time_ns == 1791346800123456789);
    }

    SECTION("with the local clock, the raw device clock is the sample time")
    {
        FullBodyPublisher publisher(runtime.factory(), DeviceClock::LOCAL);
        CHECK(publisher.publish(frame_at(player_time_ns), stop));

        REQUIRE(runtime.transports()[0].samples.size() == 1);
        const PushedSample& sample = runtime.transports()[0].samples[0];
        CHECK(sample.raw_device_time_ns == sample.sample_time_ns);
    }
}

TEST_CASE("full body publisher: after a push throws, the transport comes back or the run ends", "[unit][jmex][recovery]")
{
    FakeRuntime runtime;
    std::atomic<bool> stop{ false };
    // Every backoff the publisher asked for, in milliseconds. None is slept.
    std::vector<int> waits;
    const auto no_sleep = [&waits](int wait_ms, const std::atomic<bool>& stop_flag)
    {
        waits.push_back(wait_ms);
        return !stop_flag.load(std::memory_order_relaxed);
    };

    SECTION("a push that throws once costs that frame, and the next goes out on a new transport")
    {
        FullBodyPublisher publisher(runtime.factory(), DeviceClock::DEVICE, no_sleep);
        CHECK(publisher.publish(frame_at(1000), stop));

        runtime.restart();
        CHECK_FALSE(publisher.publish(frame_at(2000), stop)); // recovered, but this frame is gone

        REQUIRE(runtime.transports().size() == 2);
        CHECK(runtime.transports()[1].came_up);
        CHECK_FALSE(runtime.transports()[1].wait_for_system);
        CHECK(runtime.transports()[1].others_up == 0); // the dead one was let go first
        CHECK(runtime.transports()[1].samples.empty()); // the frame that failed is not pushed again

        CHECK(publisher.publish(frame_at(3000), stop));

        REQUIRE(runtime.transports().size() == 2);
        REQUIRE(runtime.transports()[1].samples.size() == 1);
        CHECK(runtime.transports()[1].samples[0].raw_device_time_ns == 3000);
        CHECK(runtime.transports()[0].samples.size() == 1);

        CHECK(publisher.push_failures() == 1);
        CHECK(publisher.session_recoveries() == 1);
        CHECK(publisher.recoveries_abandoned() == 0);
    }

    SECTION("a runtime that never comes back ends the run once the backoff is spent")
    {
        FullBodyPublisher publisher(runtime.factory(), DeviceClock::DEVICE, no_sleep);
        runtime.go_down();

        REQUIRE_THROWS_WITH(publisher.publish(frame_at(1000), stop), ContainsSubstring("restart the CloudXR runtime"));

        CHECK(runtime.transports().size() == 1 + 8); // startup, then eight attempts
        CHECK(publisher.push_failures() == 1);
        CHECK(publisher.session_recoveries() == 0);
        CHECK(publisher.recoveries_abandoned() == 1);

        // One backoff before each attempt.
        REQUIRE(waits.size() == 8);
        CHECK(waits.front() <= 500); // so a blip costs a blip
        for (size_t attempt = 1; attempt < waits.size(); ++attempt)
        {
            INFO("attempt " << attempt);
            // A long outage is not retried at the rate of a short one.
            CHECK(waits[attempt] >= waits[attempt - 1]);
        }
        int total_ms = 0;
        for (const int wait_ms : waits)
        {
            total_ms += wait_ms;
        }
        CHECK(total_ms >= 20000); // outlasts a CloudXR runtime restart
    }

    SECTION("a stop during a backoff ends the recovery, and the run, without an error")
    {
        // Ctrl-C lands during the third backoff, after two attempts have failed.
        const auto stop_during_third_wait = [&waits, &stop](int wait_ms, const std::atomic<bool>&)
        {
            waits.push_back(wait_ms);
            if (waits.size() == 3)
            {
                stop.store(true, std::memory_order_relaxed);
            }
            return !stop.load(std::memory_order_relaxed);
        };
        FullBodyPublisher publisher(runtime.factory(), DeviceClock::DEVICE, stop_during_third_wait);
        runtime.go_down();

        bool pushed = true;
        REQUIRE_NOTHROW(pushed = publisher.publish(frame_at(1000), stop));
        CHECK_FALSE(pushed);

        CHECK(publisher.push_failures() == 1);
        CHECK(publisher.session_recoveries() == 0);
        CHECK(publisher.recoveries_abandoned() == 0);
    }

    SECTION("the real backoff is slept in slices, so a stop lands inside the first one")
    {
        FullBodyPublisher publisher(runtime.factory(), DeviceClock::DEVICE);
        runtime.go_down();

        std::jthread signaller(
            [&stop]
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                stop.store(true, std::memory_order_relaxed);
            });

        const auto started = std::chrono::steady_clock::now();
        REQUIRE_NOTHROW(publisher.publish(frame_at(1000), stop));
        const auto took =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started);

        CHECK(took.count() < 500); // the first backoff
    }
}
