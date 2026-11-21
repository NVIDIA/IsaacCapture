// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Unit tests for the generated RobotDescription FlatBuffer types.

#include <catch2/catch_test_macros.hpp>
#include <flatbuffers/flatbuffers.h>
#include <schema/robot_description_generated.h>
#include <schema/serialized.hpp>
#include <schema/timestamp_generated.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#define VT(field) (field + 2) * 2

static_assert(core::RobotJointDescription::VT_NAME == VT(0));
static_assert(core::RobotJointDescription::VT_KIND == VT(1));
static_assert(core::RobotJointDescription::VT_HAS_LIMITS == VT(2));
static_assert(core::RobotJointDescription::VT_MIN_POSITION == VT(3));
static_assert(core::RobotJointDescription::VT_MAX_POSITION == VT(4));
static_assert(core::RobotJointDescription::VT_MAX_VELOCITY == VT(5));
static_assert(core::RobotJointDescription::VT_MAX_EFFORT == VT(6));
static_assert(core::RobotJointDescription::VT_UNIT == VT(7));

static_assert(core::RobotGripperDescription::VT_NAME == VT(0));
static_assert(core::RobotGripperDescription::VT_SERIES == VT(1));
static_assert(core::RobotGripperDescription::VT_JOINTS == VT(2));
static_assert(core::RobotGripperDescription::VT_OPEN_POSITION == VT(3));
static_assert(core::RobotGripperDescription::VT_CLOSED_POSITION == VT(4));

static_assert(core::RobotValueDescription::VT_NAME == VT(0));
static_assert(core::RobotValueDescription::VT_KIND == VT(1));
static_assert(core::RobotValueDescription::VT_UNIT == VT(2));

static_assert(core::RobotTextPair::VT_KEY == VT(0));
static_assert(core::RobotTextPair::VT_VALUE == VT(1));

static_assert(core::RobotSeriesDescription::VT_NAME == VT(0));
static_assert(core::RobotSeriesDescription::VT_SHAPE == VT(1));
static_assert(core::RobotSeriesDescription::VT_SHAPE_NAME == VT(2));
static_assert(core::RobotSeriesDescription::VT_RATE_HZ == VT(3));
static_assert(core::RobotSeriesDescription::VT_STALE_AFTER_S == VT(4));
static_assert(core::RobotSeriesDescription::VT_JOINTS == VT(5));
static_assert(core::RobotSeriesDescription::VT_HAS_VELOCITY == VT(6));
static_assert(core::RobotSeriesDescription::VT_HAS_EFFORT == VT(7));
static_assert(core::RobotSeriesDescription::VT_VALUES == VT(8));
static_assert(core::RobotSeriesDescription::VT_ATTRIBUTES == VT(9));

static_assert(core::RobotDescription::VT_ROBOT == VT(0));
static_assert(core::RobotDescription::VT_SERIES == VT(1));
static_assert(core::RobotDescription::VT_GRIPPERS == VT(2));
static_assert(core::RobotDescription::VT_ATTRIBUTES == VT(3));

static_assert(core::RobotDescriptionRecord::VT_DATA == VT(0));
static_assert(core::RobotDescriptionRecord::VT_TIMESTAMP == VT(1));

static_assert(core::RobotJointKind_REVOLUTE == 0);
static_assert(core::RobotJointKind_PRISMATIC == 1);
static_assert(core::RobotJointKind_CONTINUOUS == 2);
static_assert(core::RobotJointKind_GRIPPER == 3);

static_assert(core::RobotValueKind_ANY == 0);
static_assert(core::RobotValueKind_NUMBER == 1);
static_assert(core::RobotValueKind_FLAG == 2);
static_assert(core::RobotValueKind_TEXT == 3);
static_assert(core::RobotValueKind_NUMBERS == 4);

static_assert(core::RobotSeriesShape_JOINTS == 0);
static_assert(core::RobotSeriesShape_STATUS == 1);
static_assert(core::RobotSeriesShape_CUSTOM == 2);

namespace
{

std::shared_ptr<core::RobotTextPairT> make_pair(const std::string& key, const std::string& value)
{
    auto pair = std::make_shared<core::RobotTextPairT>();
    pair->key = key;
    pair->value = value;
    return pair;
}

std::shared_ptr<core::RobotValueDescriptionT> make_value(const std::string& name,
                                                         core::RobotValueKind kind,
                                                         const std::string& unit)
{
    auto value = std::make_shared<core::RobotValueDescriptionT>();
    value->name = name;
    value->kind = kind;
    value->unit = unit;
    return value;
}

//! A robot with one series of each shape and one gripper.
core::RobotDescriptionT make_description()
{
    core::RobotDescriptionT description;
    description.robot = "test_arm";
    description.attributes.push_back(make_pair("maker", "nobody"));

    auto arm = std::make_shared<core::RobotSeriesDescriptionT>();
    arm->name = "arm";
    arm->shape = core::RobotSeriesShape_JOINTS;
    arm->shape_name = "joints";
    arm->rate_hz = 20.0;
    arm->stale_after_s = 0.5;
    arm->has_velocity = true;
    arm->has_effort = true;

    auto shoulder = std::make_shared<core::RobotJointDescriptionT>();
    shoulder->name = "shoulder";
    shoulder->kind = core::RobotJointKind_REVOLUTE;
    shoulder->has_limits = true;
    shoulder->min_position = -2.8;
    shoulder->max_position = 2.8;
    shoulder->max_velocity = 2.0;
    shoulder->max_effort = 40.0;
    arm->joints.push_back(std::move(shoulder));

    auto roll = std::make_shared<core::RobotJointDescriptionT>();
    roll->name = "roll";
    roll->kind = core::RobotJointKind_CONTINUOUS;
    arm->joints.push_back(std::move(roll));

    auto jaw = std::make_shared<core::RobotJointDescriptionT>();
    jaw->name = "jaw";
    jaw->kind = core::RobotJointKind_GRIPPER;
    jaw->has_limits = true;
    jaw->min_position = 0.0;
    jaw->max_position = 0.08;
    jaw->unit = "m";
    arm->joints.push_back(std::move(jaw));
    description.series.push_back(std::move(arm));

    auto status = std::make_shared<core::RobotSeriesDescriptionT>();
    status->name = "status";
    status->shape = core::RobotSeriesShape_STATUS;
    status->shape_name = "status";
    status->stale_after_s = 2.0;
    status->values.push_back(make_value("motors_on", core::RobotValueKind_ANY, ""));
    status->values.push_back(make_value("battery", core::RobotValueKind_NUMBER, "V"));
    description.series.push_back(std::move(status));

    auto force = std::make_shared<core::RobotSeriesDescriptionT>();
    force->name = "wrist_force";
    force->shape = core::RobotSeriesShape_CUSTOM;
    force->shape_name = "force";
    force->values.push_back(make_value("fz", core::RobotValueKind_NUMBER, "N"));
    force->values.push_back(make_value("contact", core::RobotValueKind_FLAG, ""));
    force->values.push_back(make_value("note", core::RobotValueKind_TEXT, ""));
    force->values.push_back(make_value("raw", core::RobotValueKind_NUMBERS, ""));
    force->attributes.push_back(make_pair("at", "wrist"));
    description.series.push_back(std::move(force));

    auto gripper = std::make_shared<core::RobotGripperDescriptionT>();
    gripper->name = "jaw";
    gripper->series = "arm";
    gripper->joints = { "jaw", "roll" };
    gripper->open_position = 0.08;
    gripper->closed_position = 0.0;
    description.grippers.push_back(std::move(gripper));

    return description;
}

} // namespace

TEST_CASE("RobotDescription: every field is optional", "[schema][robot_description]")
{
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(core::CreateRobotDescriptionRecord(builder));

    const auto* record = flatbuffers::GetRoot<core::RobotDescriptionRecord>(builder.GetBufferPointer());
    REQUIRE(record != nullptr);
    CHECK(record->data() == nullptr);
    CHECK(record->timestamp() == nullptr);
}

TEST_CASE("RobotDescription: defaults of a joint, a value and a series", "[schema][robot_description]")
{
    const core::RobotJointDescriptionT joint;
    CHECK(joint.name.empty());
    CHECK(joint.kind == core::RobotJointKind_REVOLUTE);
    CHECK_FALSE(joint.has_limits);
    CHECK(joint.min_position == 0.0);
    CHECK(joint.max_position == 0.0);
    CHECK(joint.max_velocity == 0.0);
    CHECK(joint.max_effort == 0.0);
    CHECK(joint.unit.empty());

    const core::RobotValueDescriptionT value;
    CHECK(value.kind == core::RobotValueKind_ANY);
    CHECK(value.unit.empty());

    const core::RobotSeriesDescriptionT series;
    CHECK(series.shape == core::RobotSeriesShape_JOINTS);
    CHECK(series.rate_hz == 0.0);
    CHECK(series.stale_after_s == 0.0);
    CHECK_FALSE(series.has_velocity);
    CHECK_FALSE(series.has_effort);
    CHECK(series.joints.empty());
    CHECK(series.values.empty());
    CHECK(series.attributes.empty());
}

TEST_CASE("RobotDescription: a joints series packs and reads back, joints in order", "[schema][robot_description]")
{
    const core::RobotDescriptionT native = make_description();
    const auto description = core::pack<core::RobotDescription>(native);

    CHECK(description->robot()->str() == "test_arm");
    REQUIRE(description->series() != nullptr);
    REQUIRE(description->series()->size() == 3);

    const auto* arm = description->series()->Get(0);
    CHECK(arm->name()->str() == "arm");
    CHECK(arm->shape() == core::RobotSeriesShape_JOINTS);
    CHECK(arm->shape_name()->str() == "joints");
    CHECK(arm->rate_hz() == 20.0);
    CHECK(arm->stale_after_s() == 0.5);
    CHECK(arm->has_velocity());
    CHECK(arm->has_effort());
    REQUIRE(arm->joints() != nullptr);
    REQUIRE(arm->joints()->size() == 3);

    const auto* shoulder = arm->joints()->Get(0);
    CHECK(shoulder->name()->str() == "shoulder");
    CHECK(shoulder->kind() == core::RobotJointKind_REVOLUTE);
    CHECK(shoulder->has_limits());
    CHECK(shoulder->min_position() == -2.8);
    CHECK(shoulder->max_position() == 2.8);
    CHECK(shoulder->max_velocity() == 2.0);
    CHECK(shoulder->max_effort() == 40.0);

    const auto* roll = arm->joints()->Get(1);
    CHECK(roll->name()->str() == "roll");
    CHECK(roll->kind() == core::RobotJointKind_CONTINUOUS);
    CHECK_FALSE(roll->has_limits());

    const auto* jaw = arm->joints()->Get(2);
    CHECK(jaw->name()->str() == "jaw");
    CHECK(jaw->kind() == core::RobotJointKind_GRIPPER);
    CHECK(jaw->max_position() == 0.08);
    CHECK(jaw->unit()->str() == "m");
}

TEST_CASE("RobotDescription: status and custom series keep their values in order", "[schema][robot_description]")
{
    const core::RobotDescriptionT native = make_description();
    const auto description = core::pack<core::RobotDescription>(native);

    const auto* status = description->series()->Get(1);
    CHECK(status->name()->str() == "status");
    CHECK(status->shape() == core::RobotSeriesShape_STATUS);
    CHECK(status->shape_name()->str() == "status");
    CHECK(status->stale_after_s() == 2.0);
    REQUIRE(status->values()->size() == 2);
    CHECK(status->values()->Get(0)->name()->str() == "motors_on");
    CHECK(status->values()->Get(0)->kind() == core::RobotValueKind_ANY);
    CHECK(status->values()->Get(1)->name()->str() == "battery");
    CHECK(status->values()->Get(1)->kind() == core::RobotValueKind_NUMBER);
    CHECK(status->values()->Get(1)->unit()->str() == "V");

    const auto* force = description->series()->Get(2);
    CHECK(force->name()->str() == "wrist_force");
    CHECK(force->shape() == core::RobotSeriesShape_CUSTOM);
    CHECK(force->shape_name()->str() == "force");
    REQUIRE(force->values()->size() == 4);
    CHECK(force->values()->Get(0)->name()->str() == "fz");
    CHECK(force->values()->Get(0)->unit()->str() == "N");
    CHECK(force->values()->Get(1)->kind() == core::RobotValueKind_FLAG);
    CHECK(force->values()->Get(2)->kind() == core::RobotValueKind_TEXT);
    CHECK(force->values()->Get(3)->kind() == core::RobotValueKind_NUMBERS);
    REQUIRE(force->attributes()->size() == 1);
    CHECK(force->attributes()->Get(0)->key()->str() == "at");
    CHECK(force->attributes()->Get(0)->value()->str() == "wrist");
}

TEST_CASE("RobotDescription: grippers and attributes read back", "[schema][robot_description]")
{
    const core::RobotDescriptionT native = make_description();
    const auto description = core::pack<core::RobotDescription>(native);

    REQUIRE(description->grippers() != nullptr);
    REQUIRE(description->grippers()->size() == 1);
    const auto* gripper = description->grippers()->Get(0);
    CHECK(gripper->name()->str() == "jaw");
    CHECK(gripper->series()->str() == "arm");
    REQUIRE(gripper->joints()->size() == 2);
    CHECK(gripper->joints()->Get(0)->str() == "jaw");
    CHECK(gripper->joints()->Get(1)->str() == "roll");
    CHECK(gripper->open_position() == 0.08);
    CHECK(gripper->closed_position() == 0.0);

    REQUIRE(description->attributes()->size() == 1);
    CHECK(description->attributes()->Get(0)->key()->str() == "maker");
    CHECK(description->attributes()->Get(0)->value()->str() == "nobody");
}

TEST_CASE("RobotDescription: a record carries the description and its timestamp", "[schema][robot_description]")
{
    core::RobotDescriptionRecordT native;
    native.data = std::make_shared<core::RobotDescriptionT>(make_description());
    native.timestamp = std::make_shared<core::DeviceDataTimestamp>(11, 10, 0);
    const auto record = core::pack<core::RobotDescriptionRecord>(native);

    const auto bytes = record.buffer();
    flatbuffers::Verifier verifier(bytes.data(), bytes.size());
    REQUIRE(verifier.VerifyBuffer<core::RobotDescriptionRecord>(nullptr));

    REQUIRE(record->data() != nullptr);
    CHECK(record->data()->robot()->str() == "test_arm");
    REQUIRE(record->timestamp() != nullptr);
    CHECK(record->timestamp()->available_time_local_common_clock() == 11);
    CHECK(record->timestamp()->sample_time_local_common_clock() == 10);

    core::RobotDescriptionRecordT unpacked;
    record->UnPackTo(&unpacked);
    REQUIRE(unpacked.data != nullptr);
    CHECK(unpacked.data->series.size() == 3);
    CHECK(unpacked.data->series[0]->joints[2]->name == "jaw");
    CHECK(unpacked.data->grippers[0]->joints == std::vector<std::string>{ "jaw", "roll" });
}

TEST_CASE("RobotDescription: the record names itself and embeds its binary schema", "[schema][robot_description]")
{
    CHECK(std::string(core::RobotDescriptionRecord::GetFullyQualifiedName()) == "core.RobotDescriptionRecord");
    CHECK(core::RobotDescriptionRecord::BinarySchema::size() > 0);
    CHECK(flatbuffers::BufferHasIdentifier(core::RobotDescriptionRecord::BinarySchema::data(), "BFBS"));
}

#undef VT
