// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Python bindings for the SpaceMouse FlatBuffer schema.
// Types: SpaceMouseOutput (table), exposed as an encoded view.

#pragma once

#include "schema_serialized.h"

#include <pybind11/pybind11.h>
#include <schema/spacemouse_generated.h>

#include <cstdint>
#include <string>
#include <vector>

namespace py = pybind11;

namespace core
{

inline void bind_spacemouse(py::module& m)
{
    serialized_class<SpaceMouseOutput>(m, "SpaceMouseOutput", "Encoded raw SpaceMouse axis/button state.")
        .def(py::init(
                 [](std::vector<float> translation, std::vector<float> rotation, std::vector<uint16_t> pressed_buttons,
                    bool is_valid)
                 {
                     SpaceMouseOutputT native;
                     native.translation = std::move(translation);
                     native.rotation = std::move(rotation);
                     native.pressed_buttons = std::move(pressed_buttons);
                     native.is_valid = is_valid;
                     return pack<SpaceMouseOutput>(native);
                 }),
             py::arg("translation"), py::arg("rotation"), py::arg("pressed_buttons"), py::arg("is_valid"),
             "Encode a SpaceMouse axis/button snapshot.")
        .def_property_readonly("translation", vector_field(&SpaceMouseOutput::translation))
        .def_property_readonly("rotation", vector_field(&SpaceMouseOutput::rotation))
        .def_property_readonly("pressed_buttons", vector_field(&SpaceMouseOutput::pressed_buttons))
        .def_property_readonly("is_valid", field(&SpaceMouseOutput::is_valid))
        .def("__repr__",
             [](const Serialized<SpaceMouseOutput>& self)
             {
                 std::string result = "SpaceMouseOutput(pressed_buttons=[";
                 const auto* buttons = self->pressed_buttons();
                 if (buttons != nullptr)
                 {
                     for (size_t i = 0; i < buttons->size(); ++i)
                     {
                         if (i > 0)
                             result += ", ";
                         result += std::to_string((*buttons)[i]);
                     }
                 }
                 result += "], is_valid=" + std::to_string(self->is_valid()) + ")";
                 return result;
             });

    bind_record<SpaceMouseOutputRecord, SpaceMouseOutput>(m, "SpaceMouseOutputRecord", "SpaceMouseOutput");
}

} // namespace core
