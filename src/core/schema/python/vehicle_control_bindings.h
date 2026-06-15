// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "schema_serialized.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <schema/vehicle_control_generated.h>

#include <cstdint>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace core
{

inline void bind_vehicle_control(py::module& m)
{
    serialized_class<VehicleControlCommand>(m, "VehicleControlCommand", "Encoded vehicle control payload.")
        .def(py::init([](uint64_t sequence, float steer, float accel, float throttle, float brake)
        {
            VehicleControlCommandT native;
            native.sequence = sequence;
            native.steer = steer;
            native.accel = accel;
            native.throttle = throttle;
            native.brake = brake;
            return pack<VehicleControlCommand>(native);
        }),
             py::arg("sequence") = 0, py::arg("steer") = 0.0f, py::arg("accel") = 0.0f, py::arg("throttle") = 0.0f, py::arg("brake") = 0.0f)
        .def_property_readonly("sequence", field(&VehicleControlCommand::sequence))
        .def_property_readonly("steer", field(&VehicleControlCommand::steer))
        .def_property_readonly("accel", field(&VehicleControlCommand::accel))
        .def_property_readonly("throttle", field(&VehicleControlCommand::throttle))
        .def_property_readonly("brake", field(&VehicleControlCommand::brake));

    bind_record<VehicleControlCommandRecord, VehicleControlCommand>(m, "VehicleControlCommandRecord", "VehicleControlCommand");
}

} // namespace core
