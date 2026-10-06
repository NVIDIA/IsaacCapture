// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <schema/soma_common_v0_generated.h>

namespace py = pybind11;

namespace core
{

inline void bind_soma_common_v0(py::module& m)
{
    py::enum_<SomaHandednessV0>(m, "SomaHandednessV0")
        .value("UNSPECIFIED", SomaHandednessV0_UNSPECIFIED)
        .value("LEFT", SomaHandednessV0_LEFT)
        .value("RIGHT", SomaHandednessV0_RIGHT);

    py::class_<SomaJointRotationV0>(m, "SomaJointRotationV0")
        .def(py::init<>())
        .def(py::init<const Quaternion&, bool>(), py::arg("rotation"), py::arg("is_valid") = false)
        .def_property_readonly("rotation", &SomaJointRotationV0::rotation, py::return_value_policy::reference_internal)
        .def_property_readonly("is_valid", &SomaJointRotationV0::is_valid);
}

} // namespace core
