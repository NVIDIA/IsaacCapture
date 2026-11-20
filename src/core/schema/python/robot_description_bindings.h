// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Python bindings for the RobotDescription FlatBuffer schema.
// Types: RobotJointKind, RobotValueKind, RobotSeriesShape (enums), RobotJointDescription,
// RobotGripperDescription, RobotValueDescription, RobotTextPair, RobotSeriesDescription,
// RobotDescription (tables) and RobotDescriptionRecord, all tables exposed as encoded views.
// Also robot_state_record_schema(): the record schema of each standard robot-state shape.

#pragma once

#include "schema_serialized.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <schema/joint_state_generated.h>
#include <schema/robot_description_generated.h>
#include <schema/robot_status_generated.h>

#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace core
{

//! Fully qualified name and embedded binary schema of a record table.
template <typename RecordT>
std::pair<std::string, py::bytes> robot_state_record_schema_of()
{
    return { RecordT::GetFullyQualifiedName(),
             py::bytes(reinterpret_cast<const char*>(RecordT::BinarySchema::data()), RecordT::BinarySchema::size()) };
}

inline void bind_robot_description(py::module& m)
{
    py::enum_<RobotJointKind>(m, "RobotJointKind")
        .value("REVOLUTE", RobotJointKind_REVOLUTE)
        .value("PRISMATIC", RobotJointKind_PRISMATIC)
        .value("CONTINUOUS", RobotJointKind_CONTINUOUS)
        .value("GRIPPER", RobotJointKind_GRIPPER);

    py::enum_<RobotValueKind>(m, "RobotValueKind")
        .value("ANY", RobotValueKind_ANY)
        .value("NUMBER", RobotValueKind_NUMBER)
        .value("FLAG", RobotValueKind_FLAG)
        .value("TEXT", RobotValueKind_TEXT)
        .value("NUMBERS", RobotValueKind_NUMBERS);

    py::enum_<RobotSeriesShape>(m, "RobotSeriesShape")
        .value("JOINTS", RobotSeriesShape_JOINTS)
        .value("STATUS", RobotSeriesShape_STATUS)
        .value("CUSTOM", RobotSeriesShape_CUSTOM);

    serialized_class<RobotJointDescription>(m, "RobotJointDescription", "Encoded description of one joint.")
        .def(py::init(
                 [](const std::string& name, RobotJointKind kind, bool has_limits, double min_position,
                    double max_position, double max_velocity, double max_effort, const std::string& unit)
                 {
                     RobotJointDescriptionT native;
                     native.name = name;
                     native.kind = kind;
                     native.has_limits = has_limits;
                     native.min_position = min_position;
                     native.max_position = max_position;
                     native.max_velocity = max_velocity;
                     native.max_effort = max_effort;
                     native.unit = unit;
                     return pack<RobotJointDescription>(native);
                 }),
             py::arg("name") = std::string{}, py::arg("kind") = RobotJointKind_REVOLUTE, py::arg("has_limits") = false,
             py::arg("min_position") = 0.0, py::arg("max_position") = 0.0, py::arg("max_velocity") = 0.0,
             py::arg("max_effort") = 0.0, py::arg("unit") = std::string{}, "Encode one joint's description.")
        .def_property_readonly("name", string_field(&RobotJointDescription::name))
        .def_property_readonly("kind", field(&RobotJointDescription::kind))
        .def_property_readonly("has_limits", field(&RobotJointDescription::has_limits))
        .def_property_readonly("min_position", field(&RobotJointDescription::min_position))
        .def_property_readonly("max_position", field(&RobotJointDescription::max_position))
        .def_property_readonly("max_velocity", field(&RobotJointDescription::max_velocity))
        .def_property_readonly("max_effort", field(&RobotJointDescription::max_effort))
        .def_property_readonly("unit", string_field(&RobotJointDescription::unit))
        .def("__repr__",
             [](const Serialized<RobotJointDescription>& self)
             {
                 const auto* name = self->name();
                 return "RobotJointDescription(name=" + (name != nullptr ? name->str() : std::string{}) + ")";
             });

    serialized_class<RobotGripperDescription>(
        m, "RobotGripperDescription", "Encoded description of a gripper: named joints of one joints series.")
        .def(py::init(
                 [](const std::string& name, const std::string& series, const std::vector<std::string>& joints,
                    double open_position, double closed_position)
                 {
                     RobotGripperDescriptionT native;
                     native.name = name;
                     native.series = series;
                     native.joints = joints;
                     native.open_position = open_position;
                     native.closed_position = closed_position;
                     return pack<RobotGripperDescription>(native);
                 }),
             py::arg("name") = std::string{}, py::arg("series") = std::string{},
             py::arg("joints") = std::vector<std::string>{}, py::arg("open_position") = 0.0,
             py::arg("closed_position") = 0.0, "Encode one gripper's description.")
        .def_property_readonly("name", string_field(&RobotGripperDescription::name))
        .def_property_readonly("series", string_field(&RobotGripperDescription::series))
        .def_property_readonly("joints",
                               [](const Serialized<RobotGripperDescription>& self)
                               {
                                   std::vector<std::string> names;
                                   if (const auto* joints = self->joints())
                                   {
                                       names.reserve(joints->size());
                                       for (const auto* joint : *joints)
                                       {
                                           names.push_back(joint->str());
                                       }
                                   }
                                   return names;
                               })
        .def_property_readonly("open_position", field(&RobotGripperDescription::open_position))
        .def_property_readonly("closed_position", field(&RobotGripperDescription::closed_position))
        .def("__repr__",
             [](const Serialized<RobotGripperDescription>& self)
             {
                 const auto* name = self->name();
                 return "RobotGripperDescription(name=" + (name != nullptr ? name->str() : std::string{}) + ")";
             });

    serialized_class<RobotValueDescription>(
        m, "RobotValueDescription", "Encoded description of one status reading or one field of a robot's own shape.")
        .def(py::init(
                 [](const std::string& name, RobotValueKind kind, const std::string& unit)
                 {
                     RobotValueDescriptionT native;
                     native.name = name;
                     native.kind = kind;
                     native.unit = unit;
                     return pack<RobotValueDescription>(native);
                 }),
             py::arg("name") = std::string{}, py::arg("kind") = RobotValueKind_ANY, py::arg("unit") = std::string{},
             "Encode one value's description.")
        .def_property_readonly("name", string_field(&RobotValueDescription::name))
        .def_property_readonly("kind", field(&RobotValueDescription::kind))
        .def_property_readonly("unit", string_field(&RobotValueDescription::unit))
        .def("__repr__",
             [](const Serialized<RobotValueDescription>& self)
             {
                 const auto* name = self->name();
                 return "RobotValueDescription(name=" + (name != nullptr ? name->str() : std::string{}) + ")";
             });

    serialized_class<RobotTextPair>(m, "RobotTextPair", "Encoded key and value, both text.")
        .def(py::init(
                 [](const std::string& key, const std::string& value)
                 {
                     RobotTextPairT native;
                     native.key = key;
                     native.value = value;
                     return pack<RobotTextPair>(native);
                 }),
             py::arg("key") = std::string{}, py::arg("value") = std::string{}, "Encode one key and value.")
        .def_property_readonly("key", string_field(&RobotTextPair::key))
        .def_property_readonly("value", string_field(&RobotTextPair::value))
        .def("__repr__",
             [](const Serialized<RobotTextPair>& self)
             {
                 const auto* key = self->key();
                 return "RobotTextPair(key=" + (key != nullptr ? key->str() : std::string{}) + ")";
             });

    // One data series: `joints` for JOINTS, `values` for STATUS and CUSTOM, in the order of a record.
    serialized_class<RobotSeriesDescription>(
        m, "RobotSeriesDescription", "Encoded description of one data series a robot sends.")
        .def(py::init(
                 [](const std::string& name, RobotSeriesShape shape, const std::string& shape_name, double rate_hz,
                    double stale_after_s, const std::vector<Serialized<RobotJointDescription>>& joints,
                    bool has_velocity, bool has_effort, const std::vector<Serialized<RobotValueDescription>>& values,
                    const std::vector<Serialized<RobotTextPair>>& attributes)
                 {
                     RobotSeriesDescriptionT native;
                     native.name = name;
                     native.shape = shape;
                     native.shape_name = shape_name;
                     native.rate_hz = rate_hz;
                     native.stale_after_s = stale_after_s;
                     native.joints = to_native_vector(joints, "joints");
                     native.has_velocity = has_velocity;
                     native.has_effort = has_effort;
                     native.values = to_native_vector(values, "values");
                     native.attributes = to_native_vector(attributes, "attributes");
                     return pack<RobotSeriesDescription>(native);
                 }),
             py::arg("name") = std::string{}, py::arg("shape") = RobotSeriesShape_JOINTS,
             py::arg("shape_name") = std::string{}, py::arg("rate_hz") = 0.0, py::arg("stale_after_s") = 0.0,
             py::arg("joints") = std::vector<Serialized<RobotJointDescription>>{}, py::arg("has_velocity") = false,
             py::arg("has_effort") = false, py::arg("values") = std::vector<Serialized<RobotValueDescription>>{},
             py::arg("attributes") = std::vector<Serialized<RobotTextPair>>{}, "Encode one data series' description.")
        .def_property_readonly("name", string_field(&RobotSeriesDescription::name))
        .def_property_readonly("shape", field(&RobotSeriesDescription::shape))
        .def_property_readonly("shape_name", string_field(&RobotSeriesDescription::shape_name))
        .def_property_readonly("rate_hz", field(&RobotSeriesDescription::rate_hz))
        .def_property_readonly("stale_after_s", field(&RobotSeriesDescription::stale_after_s))
        .def_property_readonly("joints", [](const Serialized<RobotSeriesDescription>& self)
                               { return narrow_vector(self, self->joints()); })
        .def_property_readonly("has_velocity", field(&RobotSeriesDescription::has_velocity))
        .def_property_readonly("has_effort", field(&RobotSeriesDescription::has_effort))
        .def_property_readonly("values", [](const Serialized<RobotSeriesDescription>& self)
                               { return narrow_vector(self, self->values()); })
        .def_property_readonly("attributes", [](const Serialized<RobotSeriesDescription>& self)
                               { return narrow_vector(self, self->attributes()); })
        .def("__repr__",
             [](const Serialized<RobotSeriesDescription>& self)
             {
                 const auto* name = self->name();
                 return "RobotSeriesDescription(name=" + (name != nullptr ? name->str() : std::string{}) + ")";
             });

    serialized_class<RobotDescription>(
        m, "RobotDescription", "Encoded description of a robot: its data series and grippers.")
        .def(py::init(
                 [](const std::string& robot, const std::vector<Serialized<RobotSeriesDescription>>& series,
                    const std::vector<Serialized<RobotGripperDescription>>& grippers,
                    const std::vector<Serialized<RobotTextPair>>& attributes)
                 {
                     RobotDescriptionT native;
                     native.robot = robot;
                     native.series = to_native_vector(series, "series");
                     native.grippers = to_native_vector(grippers, "grippers");
                     native.attributes = to_native_vector(attributes, "attributes");
                     return pack<RobotDescription>(native);
                 }),
             py::arg("robot") = std::string{}, py::arg("series") = std::vector<Serialized<RobotSeriesDescription>>{},
             py::arg("grippers") = std::vector<Serialized<RobotGripperDescription>>{},
             py::arg("attributes") = std::vector<Serialized<RobotTextPair>>{}, "Encode a robot's description.")
        .def_property_readonly("robot", string_field(&RobotDescription::robot))
        .def_property_readonly(
            "series", [](const Serialized<RobotDescription>& self) { return narrow_vector(self, self->series()); })
        .def_property_readonly(
            "grippers", [](const Serialized<RobotDescription>& self) { return narrow_vector(self, self->grippers()); })
        .def_property_readonly("attributes", [](const Serialized<RobotDescription>& self)
                               { return narrow_vector(self, self->attributes()); })
        .def("__repr__",
             [](const Serialized<RobotDescription>& self)
             {
                 const auto* robot = self->robot();
                 const auto* series = self->series();
                 return "RobotDescription(robot=" + (robot != nullptr ? robot->str() : std::string{}) +
                        ", series=" + std::to_string(series != nullptr ? series->size() : 0) + ")";
             });

    bind_record<RobotDescriptionRecord, RobotDescription>(m, "RobotDescriptionRecord", "RobotDescription");

    m.def(
        "robot_state_record_schema",
        [](const std::string& shape)
        {
            if (shape == "joints")
            {
                return robot_state_record_schema_of<JointStateOutputRecord>();
            }
            if (shape == "status")
            {
                return robot_state_record_schema_of<RobotStatusRecord>();
            }
            if (shape == "description")
            {
                return robot_state_record_schema_of<RobotDescriptionRecord>();
            }
            throw py::value_error("robot_state_record_schema: shape must be joints, status or description, got '" +
                                  shape + "'");
        },
        py::arg("shape"),
        "Fully qualified record name and binary schema of the shape \"joints\", \"status\" or \"description\".");
}

} // namespace core
