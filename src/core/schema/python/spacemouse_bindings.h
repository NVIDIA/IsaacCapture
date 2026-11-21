// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Python bindings for the SpaceMouse FlatBuffer schema.
// Types: SpaceMouseOutput (table), exposed as an encoded view.

#pragma once

#include "pose_bindings.h"
#include "schema_serialized.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <schema/spacemouse_generated.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace core
{

inline void bind_spacemouse(py::module& m)
{
    serialized_class<SpaceMouseOutput>(m, "SpaceMouseOutput", "Encoded SpaceMouse axis and button state.")
        .def(py::init(
                 [](const Point& translation, const Point& rotation, std::vector<uint8_t> buttons, bool connected)
                 {
                     SpaceMouseOutputT native;
                     native.translation = std::make_shared<Point>(translation);
                     native.rotation = std::make_shared<Point>(rotation);
                     native.buttons = std::move(buttons);
                     native.connected = connected;
                     return pack<SpaceMouseOutput>(native);
                 }),
             py::arg("translation") = Point(), py::arg("rotation") = Point(),
             py::arg("buttons") = std::vector<uint8_t>{}, py::arg("connected") = false,
             "Encode a SpaceMouse state. Omitted axes are zero and the device reads as disconnected.")
        .def_property_readonly(
            "translation", field(&SpaceMouseOutput::translation), py::return_value_policy::reference_internal)
        .def_property_readonly("rotation", field(&SpaceMouseOutput::rotation), py::return_value_policy::reference_internal)
        .def_property_readonly("buttons", vector_field(&SpaceMouseOutput::buttons))
        .def_property_readonly("connected", field(&SpaceMouseOutput::connected))
        .def("__repr__",
             [](const Serialized<SpaceMouseOutput>& self)
             {
                 const auto point_str = [](const Point* point)
                 {
                     return point != nullptr ? "(" + std::to_string(point->x()) + ", " + std::to_string(point->y()) +
                                                   ", " + std::to_string(point->z()) + ")" :
                                               std::string("None");
                 };
                 const auto* buttons = self->buttons();
                 return "SpaceMouseOutput(translation=" + point_str(self->translation()) +
                        ", rotation=" + point_str(self->rotation()) +
                        ", buttons=" + std::to_string(buttons != nullptr ? buttons->size() : 0) +
                        ", connected=" + (self->connected() ? "True" : "False") + ")";
             });

    bind_record<SpaceMouseOutputRecord, SpaceMouseOutput>(m, "SpaceMouseOutputRecord", "SpaceMouseOutput");
}

} // namespace core
