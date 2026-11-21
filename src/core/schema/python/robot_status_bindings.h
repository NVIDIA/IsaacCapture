// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Python bindings for the RobotStatus FlatBuffer schema.
// Types: RobotReadingKind, RobotFaultSeverity (enums), RobotStatusReading, RobotFault,
// RobotStatus (tables) and RobotStatusRecord, all tables exposed as encoded views.

#pragma once

#include "schema_serialized.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <schema/robot_status_generated.h>

#include <string>
#include <vector>

namespace py = pybind11;

namespace core
{

inline void bind_robot_status(py::module& m)
{
    py::enum_<RobotReadingKind>(m, "RobotReadingKind")
        .value("NUMBER", RobotReadingKind_NUMBER)
        .value("FLAG", RobotReadingKind_FLAG)
        .value("TEXT", RobotReadingKind_TEXT);

    py::enum_<RobotFaultSeverity>(m, "RobotFaultSeverity")
        .value("WARNING", RobotFaultSeverity_WARNING)
        .value("ERROR", RobotFaultSeverity_ERROR)
        .value("FATAL", RobotFaultSeverity_FATAL);

    // One named reading; `kind` says which of number, flag, text holds the value.
    serialized_class<RobotStatusReading>(m, "RobotStatusReading", "Encoded named reading of a robot.")
        .def(py::init(
                 [](const std::string& name, RobotReadingKind kind, double number, bool flag, const std::string& text,
                    bool valid)
                 {
                     RobotStatusReadingT native;
                     native.name = name;
                     native.kind = kind;
                     native.number = number;
                     native.flag = flag;
                     native.text = text;
                     native.valid = valid;
                     return pack<RobotStatusReading>(native);
                 }),
             py::arg("name") = std::string{}, py::arg("kind") = RobotReadingKind_NUMBER, py::arg("number") = 0.0,
             py::arg("flag") = false, py::arg("text") = std::string{}, py::arg("valid") = true,
             "Encode one named reading.")
        .def_property_readonly("name", string_field(&RobotStatusReading::name))
        .def_property_readonly("kind", field(&RobotStatusReading::kind))
        .def_property_readonly("number", field(&RobotStatusReading::number))
        .def_property_readonly("flag", field(&RobotStatusReading::flag))
        .def_property_readonly("text", string_field(&RobotStatusReading::text))
        .def_property_readonly("valid", field(&RobotStatusReading::valid))
        .def("__repr__",
             [](const Serialized<RobotStatusReading>& self)
             {
                 const auto* name = self->name();
                 return "RobotStatusReading(name=" + (name != nullptr ? name->str() : std::string{}) + ")";
             });

    serialized_class<RobotFault>(m, "RobotFault", "Encoded fault a robot reports.")
        .def(py::init(
                 [](const std::string& code, const std::string& text, const std::string& source, RobotFaultSeverity severity)
                 {
                     RobotFaultT native;
                     native.code = code;
                     native.text = text;
                     native.source = source;
                     native.severity = severity;
                     return pack<RobotFault>(native);
                 }),
             py::arg("code") = std::string{}, py::arg("text") = std::string{}, py::arg("source") = std::string{},
             py::arg("severity") = RobotFaultSeverity_ERROR, "Encode one fault.")
        .def_property_readonly("code", string_field(&RobotFault::code))
        .def_property_readonly("text", string_field(&RobotFault::text))
        .def_property_readonly("source", string_field(&RobotFault::source))
        .def_property_readonly("severity", field(&RobotFault::severity))
        .def("__repr__",
             [](const Serialized<RobotFault>& self)
             {
                 const auto* code = self->code();
                 return "RobotFault(code=" + (code != nullptr ? code->str() : std::string{}) + ")";
             });

    // Status at one moment: readings in the description's order, plus the faults.
    serialized_class<RobotStatus>(m, "RobotStatus", "Encoded status of a robot: named readings and faults.")
        .def(py::init(
                 [](const std::vector<Serialized<RobotStatusReading>>& readings,
                    const std::vector<Serialized<RobotFault>>& faults)
                 {
                     RobotStatusT native;
                     native.readings = to_native_vector(readings, "readings");
                     native.faults = to_native_vector(faults, "faults");
                     return pack<RobotStatus>(native);
                 }),
             py::arg("readings") = std::vector<Serialized<RobotStatusReading>>{},
             py::arg("faults") = std::vector<Serialized<RobotFault>>{}, "Encode a robot status.")
        .def_property_readonly(
            "readings", [](const Serialized<RobotStatus>& self) { return narrow_vector(self, self->readings()); })
        .def_property_readonly(
            "faults", [](const Serialized<RobotStatus>& self) { return narrow_vector(self, self->faults()); })
        .def("__repr__",
             [](const Serialized<RobotStatus>& self)
             {
                 const auto* readings = self->readings();
                 const auto* faults = self->faults();
                 return "RobotStatus(readings=" + std::to_string(readings != nullptr ? readings->size() : 0) +
                        ", faults=" + std::to_string(faults != nullptr ? faults->size() : 0) + ")";
             });

    bind_record<RobotStatusRecord, RobotStatus>(m, "RobotStatusRecord", "RobotStatus");
}

} // namespace core
