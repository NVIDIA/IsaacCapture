// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Python bindings for the TactileData FlatBuffer schema (internal only; see tactile.fbs).
// Types: TactileData (table) and TactileDataRecord, exposed as encoded views.

#pragma once

#include "schema_array_views.h"
#include "schema_serialized.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <schema/tactile_generated.h>

#include <cstdint>
#include <optional>
#include <string>

namespace py = pybind11;

namespace core
{

inline void bind_tactile(py::module& m)
{
    using FloatRows = py::array_t<float, py::array::c_style | py::array::forcecast>;

    serialized_class<TactileData>(m, "TactileData", "Encoded per-frame tactile reading (internal only).")
        .def(py::init(
                 [](const std::optional<FloatRows>& taxel_force, const std::string& device_id, uint32_t shape_version)
                 {
                     TactileDataT native;
                     if (taxel_force.has_value())
                     {
                         native.taxel_force = structs_from_rows<Point, float, 3>(*taxel_force, "taxel_force");
                     }
                     native.taxel_count = static_cast<uint32_t>(native.taxel_force.size());
                     native.device_id = device_id;
                     native.shape_version = shape_version;
                     return pack<TactileData>(native);
                 }),
             py::arg("taxel_force") = py::none(), py::arg("device_id") = std::string{}, py::arg("shape_version") = 0u,
             "Encode a tactile frame. taxel_force is an (N, 3) array; taxel_count is set to N.")
        .def_property_readonly("taxel_count", field(&TactileData::taxel_count))
        .def_property_readonly(
            "taxel_force",
            [](py::object self)
            {
                const auto& view = self.cast<const Serialized<TactileData>&>();
                return struct_vector_view<float, 3>(self, view->taxel_force());
            },
            "Per-taxel force as an (N, 3) float32 view aliasing this object's buffer; call .copy() to own it.")
        .def_property_readonly("device_id", string_field(&TactileData::device_id))
        .def_property_readonly("shape_version", field(&TactileData::shape_version))
        .def("__repr__",
             [](const Serialized<TactileData>& self)
             {
                 const auto* device_id = self->device_id();
                 return "TactileData(taxel_count=" + std::to_string(self->taxel_count()) +
                        ", device_id=" + (device_id != nullptr ? device_id->str() : std::string{}) +
                        ", shape_version=" + std::to_string(self->shape_version()) + ")";
             });

    bind_record<TactileDataRecord, TactileData>(m, "TactileDataRecord", "TactileData");
}

} // namespace core
