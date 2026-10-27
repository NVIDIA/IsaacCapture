// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Python bindings for the TactileDataArray FlatBuffer schema (see tactile_array.fbs).
// Types: TactileDataArray (table) and TactileDataArrayRecord, exposed as encoded views.

#pragma once

#include "schema_array_views.h"
#include "schema_serialized.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <schema/tactile_array_generated.h>

#include <string>

namespace py = pybind11;

namespace core
{

inline void bind_tactile_array(py::module& m)
{
    using FloatRows = py::array_t<float, py::array::c_style | py::array::forcecast>;

    serialized_class<TactileDataArray>(m, "TactileDataArray", "Encoded per-frame tactile reading.")
        .def(py::init(
                 [](const FloatRows& taxel_forces)
                 {
                     TactileDataArrayT native;
                     native.taxel_forces = structs_from_rows<Point, float, 3>(taxel_forces, "taxel_forces");
                     return pack<TactileDataArray>(native);
                 }),
             py::arg("taxel_forces").none(false), "Encode a tactile frame. taxel_forces is an (N, 3) array.")
        .def_property_readonly(
            "taxel_forces",
            [](py::object self)
            {
                const auto& view = self.cast<const Serialized<TactileDataArray>&>();
                return struct_vector_view<float, 3>(self, view->taxel_forces());
            },
            "Per-taxel force as an (N, 3) float32 view aliasing this object's buffer; call .copy() to own it.")
        .def("__repr__",
             [](const Serialized<TactileDataArray>& self)
             {
                 const auto* forces = self->taxel_forces();
                 return "TactileDataArray(taxels=" + std::to_string(forces != nullptr ? forces->size() : 0) + ")";
             });

    bind_record<TactileDataArrayRecord, TactileDataArray>(m, "TactileDataArrayRecord", "TactileDataArray");
}

} // namespace core
