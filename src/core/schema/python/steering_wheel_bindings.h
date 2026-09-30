// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "schema_serialized.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <schema/steering_wheel_generated.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace core
{

inline void bind_steering_wheel(py::module& m)
{
    serialized_class<SteeringWheelOutput>(m, "SteeringWheelOutput", "Encoded steering wheel payload.")
        .def(py::init(
                 [](float steering, float throttle, float brake, float clutch, std::vector<uint8_t> buttons, int hat_x,
                    int hat_y, bool connected, int64_t sample_time_monotonic_ns)
                 {
                     SteeringWheelOutputT native;
                     native.steering = steering;
                     native.throttle = throttle;
                     native.brake = brake;
                     native.clutch = clutch;
                     native.buttons = std::move(buttons);
                     native.hat_x = hat_x;
                     native.hat_y = hat_y;
                     native.connected = connected;
                     native.sample_time_monotonic_ns = sample_time_monotonic_ns;
                     return pack<SteeringWheelOutput>(native);
                 }),
             py::arg("steering") = 0.0f, py::arg("throttle") = 0.0f, py::arg("brake") = 0.0f, py::arg("clutch") = 0.0f,
             py::arg("buttons") = std::vector<uint8_t>{}, py::arg("hat_x") = 0, py::arg("hat_y") = 0,
             py::arg("connected") = false, py::arg("sample_time_monotonic_ns") = 0)
        .def_property_readonly("steering", field(&SteeringWheelOutput::steering))
        .def_property_readonly("throttle", field(&SteeringWheelOutput::throttle))
        .def_property_readonly("brake", field(&SteeringWheelOutput::brake))
        .def_property_readonly("clutch", field(&SteeringWheelOutput::clutch))
        .def_property_readonly("buttons", vector_field(&SteeringWheelOutput::buttons))
        .def_property_readonly("hat_x", field(&SteeringWheelOutput::hat_x))
        .def_property_readonly("hat_y", field(&SteeringWheelOutput::hat_y))
        .def_property_readonly("connected", field(&SteeringWheelOutput::connected))
        .def_property_readonly("sample_time_monotonic_ns", field(&SteeringWheelOutput::sample_time_monotonic_ns))
        .def("__repr__",
             [](const Serialized<SteeringWheelOutput>& output)
             {
                 return "SteeringWheelOutput(steering=" + std::to_string(output->steering()) +
                        ", throttle=" + std::to_string(output->throttle()) +
                        ", brake=" + std::to_string(output->brake()) + ", clutch=" + std::to_string(output->clutch()) +
                        ", buttons=" + std::to_string((output->buttons() ? output->buttons()->size() : 0)) +
                        ", hat_x=" + std::to_string(output->hat_x()) + ", hat_y=" + std::to_string(output->hat_y()) + ")";
             });

    bind_record<SteeringWheelOutputRecord, SteeringWheelOutput>(m, "SteeringWheelOutputRecord", "SteeringWheelOutput");
}

} // namespace core
