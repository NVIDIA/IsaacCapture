// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// A schema tracker verifies each sample as its FlatBuffer table: one that fails ends the session update with a
// message naming the collection and what its publisher called the data; valid data passes under any tensor name.
// A fake tensor runtime stands in for OpenXR on timespec platforms.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <deviceio_base/tracker_vendor.hpp>
#include <deviceio_session/deviceio_session.hpp>
#include <deviceio_trackers/full_body_tracker.hpp>
#include <deviceio_trackers/generic_3axis_pedal_tracker.hpp>
#include <deviceio_trackers/joint_se3_pose_tracker.hpp>
#include <schema/full_body_generated.h>
#include <schema/pedals_generated.h>
#include <schema/serialized.hpp>
#if defined(XR_USE_TIMESPEC)
#    include <tensor_test_support/fake_tensor_runtime.hpp>
#endif

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(XR_USE_TIMESPEC)

namespace
{

using tensor_test_support::FakeTensorRuntime;

std::vector<uint8_t> pedal_bytes(float left_pedal)
{
    core::Generic3AxisPedalOutputT native;
    native.left_pedal = left_pedal;
    const auto packed = core::pack<core::Generic3AxisPedalOutput>(native);
    return std::vector<uint8_t>(packed.buffer().begin(), packed.buffer().end());
}

} // namespace

TEST_CASE("A schema tracker refuses a sample that is not its data, naming what was published", "[unit][schema_tracker]")
{
    FakeTensorRuntime runtime;
    runtime.create("pedals", 64, "generic_3axis_pedal");
    runtime.push("pedals", std::vector<uint8_t>(64, 0xFF), 1);

    auto tracker = std::make_shared<core::JointSe3PoseTracker>("pedals");
    auto session = core::DeviceIOSession::run({ tracker }, runtime.handles());
    CHECK_THROWS_MATCHES(session->update(), std::runtime_error,
                         Catch::Matchers::Message("SchemaTracker: a sample in collection 'pedals' (published as "
                                                  "'generic_3axis_pedal') is not valid 'joint_se3_pose' data for "
                                                  "JointSe3PoseTracker; check its collection_id and its publisher"));
}

TEST_CASE("A schema tracker refuses an unnamed sample that is not its data", "[unit][schema_tracker]")
{
    FakeTensorRuntime runtime;
    runtime.create("pedals", 64);
    runtime.push("pedals", std::vector<uint8_t>(64, 0xFF), 1);

    auto tracker = std::make_shared<core::JointSe3PoseTracker>("pedals");
    auto session = core::DeviceIOSession::run({ tracker }, runtime.handles());
    CHECK_THROWS_WITH(session->update(),
                      "SchemaTracker: a sample in collection 'pedals' is not valid 'joint_se3_pose' "
                      "data for JointSe3PoseTracker; check its collection_id and its publisher");
}

TEST_CASE("A schema tracker reads valid data under any tensor name", "[unit][schema_tracker]")
{
    FakeTensorRuntime runtime;
    // A sender may name its tensor freely; the name only causes a warning.
    runtime.create("pedals", 256, "my_pedals");
    runtime.push("pedals", pedal_bytes(0.25f), 1);

    auto tracker = std::make_shared<core::Generic3AxisPedalTracker>("pedals");
    auto session = core::DeviceIOSession::run({ tracker }, runtime.handles());
    session->update();
    const auto& data = tracker->get_data(*session);
    REQUIRE(data);
    CHECK(data->left_pedal() == 0.25f);
}

TEST_CASE("A full-body vendor reads another vendor's collection of the same table", "[unit][schema_tracker]")
{
    FakeTensorRuntime runtime;
    // Xsens names its tensor full_body_pose and the Noitom backend expects full_body: the same FullBodyPose layout.
    runtime.create("noitom_mocap", 4096, "full_body_pose");
    core::FullBodyPoseT native;
    native.joints = std::make_unique<core::BodyJoints>();
    native.all_joint_poses_tracked = true;
    const auto packed = core::pack<core::FullBodyPose>(native);
    runtime.push("noitom_mocap", std::vector<uint8_t>(packed.buffer().begin(), packed.buffer().end()), 1);

    auto tracker = std::make_shared<core::FullBodyTracker>();
    core::VendorConfig vendors;
    vendors.tracker_vendors.push_back({ tracker.get(), core::TrackerVendor{ "body.noitom", {} } });
    auto session = core::DeviceIOSession::run({ tracker }, runtime.handles(), std::nullopt, vendors);
    session->update();
    const auto& pose = tracker->get_body_pose(*session);
    REQUIRE(pose);
    CHECK(pose->all_joint_poses_tracked());
}

#endif
