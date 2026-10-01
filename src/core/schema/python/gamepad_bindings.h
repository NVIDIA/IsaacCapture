// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Python bindings for the Gamepad FlatBuffer schema.
// Types: GamepadOutput (table), exposed as an encoded view.

#pragma once

#include "schema_serialized.h"

#include <pybind11/pybind11.h>
#include <schema/gamepad_generated.h>

#include <cstdint>
#include <string>
#include <vector>

namespace py = pybind11;

namespace core
{

inline void bind_gamepad(py::module& m)
{
    serialized_class<GamepadOutput>(m, "GamepadOutput", "Encoded raw joystick-API button/axis state.")
        .def(py::init(
                 [](std::vector<uint16_t> pressed_buttons, std::vector<float> axes, bool is_valid)
                 {
                     GamepadOutputT native;
                     native.pressed_buttons = std::move(pressed_buttons);
                     native.axes = std::move(axes);
                     native.is_valid = is_valid;
                     return pack<GamepadOutput>(native);
                 }),
             py::arg("pressed_buttons"), py::arg("axes"), py::arg("is_valid"), "Encode a gamepad button/axis snapshot.")
        .def_property_readonly("pressed_buttons", vector_field(&GamepadOutput::pressed_buttons))
        .def_property_readonly("axes", vector_field(&GamepadOutput::axes))
        .def_property_readonly("is_valid", field(&GamepadOutput::is_valid))
        .def("__repr__",
             [](const Serialized<GamepadOutput>& self)
             {
                 std::string result = "GamepadOutput(pressed_buttons=[";
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
                 result += "], axes=[";
                 const auto* axes = self->axes();
                 if (axes != nullptr)
                 {
                     for (size_t i = 0; i < axes->size(); ++i)
                     {
                         if (i > 0)
                             result += ", ";
                         result += std::to_string((*axes)[i]);
                     }
                 }
                 result += "], is_valid=" + std::to_string(self->is_valid()) + ")";
                 return result;
             });

    bind_record<GamepadOutputRecord, GamepadOutput>(m, "GamepadOutputRecord", "GamepadOutput");
}

} // namespace core
