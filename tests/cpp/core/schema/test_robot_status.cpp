// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Unit tests for the generated RobotStatus FlatBuffer types.

#include <catch2/catch_test_macros.hpp>
#include <flatbuffers/flatbuffers.h>
#include <schema/robot_status_generated.h>
#include <schema/serialized.hpp>
#include <schema/timestamp_generated.h>

#include <memory>
#include <string>
#include <utility>

#define VT(field) (field + 2) * 2

static_assert(core::RobotStatusReading::VT_NAME == VT(0));
static_assert(core::RobotStatusReading::VT_KIND == VT(1));
static_assert(core::RobotStatusReading::VT_NUMBER == VT(2));
static_assert(core::RobotStatusReading::VT_FLAG == VT(3));
static_assert(core::RobotStatusReading::VT_TEXT == VT(4));
static_assert(core::RobotStatusReading::VT_VALID == VT(5));

static_assert(core::RobotFault::VT_CODE == VT(0));
static_assert(core::RobotFault::VT_TEXT == VT(1));
static_assert(core::RobotFault::VT_SOURCE == VT(2));
static_assert(core::RobotFault::VT_SEVERITY == VT(3));

static_assert(core::RobotStatus::VT_READINGS == VT(0));
static_assert(core::RobotStatus::VT_FAULTS == VT(1));

static_assert(core::RobotStatusRecord::VT_DATA == VT(0));
static_assert(core::RobotStatusRecord::VT_TIMESTAMP == VT(1));

static_assert(core::RobotReadingKind_NUMBER == 0);
static_assert(core::RobotReadingKind_FLAG == 1);
static_assert(core::RobotReadingKind_TEXT == 2);

static_assert(core::RobotFaultSeverity_WARNING == 0);
static_assert(core::RobotFaultSeverity_ERROR == 1);
static_assert(core::RobotFaultSeverity_FATAL == 2);

namespace
{

std::shared_ptr<core::RobotStatusReadingT> make_reading(const std::string& name, core::RobotReadingKind kind)
{
    auto reading = std::make_shared<core::RobotStatusReadingT>();
    reading->name = name;
    reading->kind = kind;
    return reading;
}

core::RobotStatusT make_status()
{
    core::RobotStatusT status;

    auto motors_on = make_reading("motors_on", core::RobotReadingKind_FLAG);
    motors_on->flag = true;
    status.readings.push_back(std::move(motors_on));

    auto battery = make_reading("battery", core::RobotReadingKind_NUMBER);
    battery->number = 24.5;
    status.readings.push_back(std::move(battery));

    auto mode = make_reading("mode", core::RobotReadingKind_TEXT);
    mode->text = "position";
    status.readings.push_back(std::move(mode));

    auto unread = make_reading("temperature", core::RobotReadingKind_NUMBER);
    unread->valid = false;
    status.readings.push_back(std::move(unread));

    auto fault = std::make_shared<core::RobotFaultT>();
    fault->code = "overtemperature";
    fault->text = "joint3 is too hot";
    fault->source = "joint3";
    fault->severity = core::RobotFaultSeverity_FATAL;
    status.faults.push_back(std::move(fault));

    return status;
}

} // namespace

TEST_CASE("RobotStatus: every field is optional", "[schema][robot_status]")
{
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(core::CreateRobotStatusRecord(builder));

    const auto* record = flatbuffers::GetRoot<core::RobotStatusRecord>(builder.GetBufferPointer());
    REQUIRE(record != nullptr);
    CHECK(record->data() == nullptr);
    CHECK(record->timestamp() == nullptr);
}

TEST_CASE("RobotStatus: defaults of a reading and of a fault", "[schema][robot_status]")
{
    const core::RobotStatusReadingT reading;
    CHECK(reading.name.empty());
    CHECK(reading.kind == core::RobotReadingKind_NUMBER);
    CHECK(reading.number == 0.0);
    CHECK_FALSE(reading.flag);
    CHECK(reading.text.empty());
    CHECK(reading.valid);

    const core::RobotFaultT fault;
    CHECK(fault.code.empty());
    CHECK(fault.text.empty());
    CHECK(fault.source.empty());
    CHECK(fault.severity == core::RobotFaultSeverity_ERROR);
}

TEST_CASE("RobotStatus: a status packs and reads back, readings in order", "[schema][robot_status]")
{
    const core::RobotStatusT native = make_status();
    const auto status = core::pack<core::RobotStatus>(native);

    REQUIRE(status->readings() != nullptr);
    REQUIRE(status->readings()->size() == 4);

    const auto* motors_on = status->readings()->Get(0);
    CHECK(motors_on->name()->str() == "motors_on");
    CHECK(motors_on->kind() == core::RobotReadingKind_FLAG);
    CHECK(motors_on->flag());
    CHECK(motors_on->valid());

    const auto* battery = status->readings()->Get(1);
    CHECK(battery->name()->str() == "battery");
    CHECK(battery->kind() == core::RobotReadingKind_NUMBER);
    CHECK(battery->number() == 24.5);

    const auto* mode = status->readings()->Get(2);
    CHECK(mode->name()->str() == "mode");
    CHECK(mode->kind() == core::RobotReadingKind_TEXT);
    CHECK(mode->text()->str() == "position");

    const auto* unread = status->readings()->Get(3);
    CHECK(unread->name()->str() == "temperature");
    CHECK_FALSE(unread->valid());

    REQUIRE(status->faults() != nullptr);
    REQUIRE(status->faults()->size() == 1);
    const auto* fault = status->faults()->Get(0);
    CHECK(fault->code()->str() == "overtemperature");
    CHECK(fault->text()->str() == "joint3 is too hot");
    CHECK(fault->source()->str() == "joint3");
    CHECK(fault->severity() == core::RobotFaultSeverity_FATAL);
}

TEST_CASE("RobotStatus: a record carries the status and its timestamp", "[schema][robot_status]")
{
    core::RobotStatusRecordT native;
    native.data = std::make_shared<core::RobotStatusT>(make_status());
    native.timestamp = std::make_shared<core::DeviceDataTimestamp>(30, 20, 0);
    const auto record = core::pack<core::RobotStatusRecord>(native);

    const auto bytes = record.buffer();
    flatbuffers::Verifier verifier(bytes.data(), bytes.size());
    REQUIRE(verifier.VerifyBuffer<core::RobotStatusRecord>(nullptr));

    REQUIRE(record->data() != nullptr);
    CHECK(record->data()->readings()->size() == 4);
    CHECK(record->data()->faults()->size() == 1);
    REQUIRE(record->timestamp() != nullptr);
    CHECK(record->timestamp()->available_time_local_common_clock() == 30);
    CHECK(record->timestamp()->sample_time_local_common_clock() == 20);
    CHECK(record->timestamp()->sample_time_raw_device_clock() == 0);

    core::RobotStatusRecordT unpacked;
    record->UnPackTo(&unpacked);
    REQUIRE(unpacked.data != nullptr);
    CHECK(unpacked.data->readings[1]->number == 24.5);
    CHECK(unpacked.data->faults[0]->code == "overtemperature");
}

TEST_CASE("RobotStatus: a status without faults reads as no faults", "[schema][robot_status]")
{
    core::RobotStatusT native = make_status();
    native.faults.clear();
    const auto status = core::pack<core::RobotStatus>(native);

    CHECK(status->readings()->size() == 4);
    CHECK((status->faults() == nullptr || status->faults()->size() == 0));
}

TEST_CASE("RobotStatus: the record names itself and embeds its binary schema", "[schema][robot_status]")
{
    CHECK(std::string(core::RobotStatusRecord::GetFullyQualifiedName()) == "core.RobotStatusRecord");
    CHECK(core::RobotStatusRecord::BinarySchema::size() > 0);
    CHECK(flatbuffers::BufferHasIdentifier(core::RobotStatusRecord::BinarySchema::data(), "BFBS"));
}

#undef VT
