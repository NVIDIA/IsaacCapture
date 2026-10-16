// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Python bindings for the Gamepad FlatBuffer schema.
// GamepadInputState is a struct; GamepadSnapshot is a table, exposed as an encoded view.

#pragma once

#include "schema_serialized.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <schema/gamepad_generated.h>

#include <memory>
#include <optional>
#include <string>

namespace py = pybind11;

namespace core
{

inline void bind_gamepad(py::module& m)
{
    py::class_<GamepadInputState>(m, "GamepadInputState")
        .def(py::init<bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, float, float,
                      float, float, float, float>(),
             py::arg("south_click") = false, py::arg("east_click") = false, py::arg("west_click") = false,
             py::arg("north_click") = false, py::arg("view_click") = false, py::arg("menu_click") = false,
             py::arg("shoulder_left_click") = false, py::arg("shoulder_right_click") = false,
             py::arg("thumbstick_left_click") = false, py::arg("thumbstick_right_click") = false,
             py::arg("dpad_up_click") = false, py::arg("dpad_down_click") = false, py::arg("dpad_left_click") = false,
             py::arg("dpad_right_click") = false, py::arg("thumbstick_left_x") = 0.0f,
             py::arg("thumbstick_left_y") = 0.0f, py::arg("thumbstick_right_x") = 0.0f,
             py::arg("thumbstick_right_y") = 0.0f, py::arg("trigger_left_value") = 0.0f,
             py::arg("trigger_right_value") = 0.0f,
             "Gamepad inputs; omitted buttons are released and omitted axes centered.")
        .def_property_readonly("south_click", &GamepadInputState::south_click)
        .def_property_readonly("east_click", &GamepadInputState::east_click)
        .def_property_readonly("west_click", &GamepadInputState::west_click)
        .def_property_readonly("north_click", &GamepadInputState::north_click)
        .def_property_readonly("view_click", &GamepadInputState::view_click)
        .def_property_readonly("menu_click", &GamepadInputState::menu_click)
        .def_property_readonly("shoulder_left_click", &GamepadInputState::shoulder_left_click)
        .def_property_readonly("shoulder_right_click", &GamepadInputState::shoulder_right_click)
        .def_property_readonly("thumbstick_left_click", &GamepadInputState::thumbstick_left_click)
        .def_property_readonly("thumbstick_right_click", &GamepadInputState::thumbstick_right_click)
        .def_property_readonly("dpad_up_click", &GamepadInputState::dpad_up_click)
        .def_property_readonly("dpad_down_click", &GamepadInputState::dpad_down_click)
        .def_property_readonly("dpad_left_click", &GamepadInputState::dpad_left_click)
        .def_property_readonly("dpad_right_click", &GamepadInputState::dpad_right_click)
        .def_property_readonly("thumbstick_left_x", &GamepadInputState::thumbstick_left_x)
        .def_property_readonly("thumbstick_left_y", &GamepadInputState::thumbstick_left_y)
        .def_property_readonly("thumbstick_right_x", &GamepadInputState::thumbstick_right_x)
        .def_property_readonly("thumbstick_right_y", &GamepadInputState::thumbstick_right_y)
        .def_property_readonly("trigger_left_value", &GamepadInputState::trigger_left_value)
        .def_property_readonly("trigger_right_value", &GamepadInputState::trigger_right_value)
        .def("__repr__",
             [](const GamepadInputState& self)
             {
                 std::string held;
                 const auto add_held = [&held](const char* name, bool pressed)
                 {
                     if (pressed)
                         held += (held.empty() ? "" : ", ") + std::string(name);
                 };
                 add_held("south", self.south_click());
                 add_held("east", self.east_click());
                 add_held("west", self.west_click());
                 add_held("north", self.north_click());
                 add_held("view", self.view_click());
                 add_held("menu", self.menu_click());
                 add_held("shoulder_left", self.shoulder_left_click());
                 add_held("shoulder_right", self.shoulder_right_click());
                 add_held("thumbstick_left", self.thumbstick_left_click());
                 add_held("thumbstick_right", self.thumbstick_right_click());
                 add_held("dpad_up", self.dpad_up_click());
                 add_held("dpad_down", self.dpad_down_click());
                 add_held("dpad_left", self.dpad_left_click());
                 add_held("dpad_right", self.dpad_right_click());
                 return "GamepadInputState(held=[" + held + "], left=(" + std::to_string(self.thumbstick_left_x()) +
                        ", " + std::to_string(self.thumbstick_left_y()) + "), right=(" +
                        std::to_string(self.thumbstick_right_x()) + ", " + std::to_string(self.thumbstick_right_y()) +
                        "), triggers=(" + std::to_string(self.trigger_left_value()) + ", " +
                        std::to_string(self.trigger_right_value()) + "))";
             });

    serialized_class<GamepadSnapshot>(
        m, "GamepadSnapshot", "Encoded gamepad snapshot: the input state, absent while no pad is open.")
        .def(py::init(
                 [](std::optional<GamepadInputState> inputs)
                 {
                     GamepadSnapshotT native;
                     if (inputs)
                         native.inputs = std::make_shared<GamepadInputState>(*inputs);
                     return pack<GamepadSnapshot>(native);
                 }),
             py::arg("inputs") = std::nullopt, "Encode a gamepad snapshot. Without inputs it reads as no pad open.")
        .def_property_readonly("inputs", field(&GamepadSnapshot::inputs), py::return_value_policy::reference_internal)
        .def("__repr__", [](const Serialized<GamepadSnapshot>& self)
             { return "GamepadSnapshot(inputs=" + std::string(self->inputs() != nullptr ? "..." : "None") + ")"; });

    bind_record<GamepadSnapshotRecord, GamepadSnapshot>(m, "GamepadSnapshotRecord", "GamepadSnapshot");
}

} // namespace core
