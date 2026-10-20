// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "schema_serialized.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <schema/tactile_generated.h>
#include <schema/tactile_metadata_generated.h>
#include <schema/tactile_validation.hpp>

#include <string>
#include <vector>

namespace py = pybind11;

namespace core
{

inline void bind_tactile(py::module& m)
{
    py::enum_<TaxelStatus>(m, "TaxelStatus")
        .value("VALID", TaxelStatus_VALID)
        .value("INVALID", TaxelStatus_INVALID)
        .value("SATURATED", TaxelStatus_SATURATED)
        .value("UNAVAILABLE", TaxelStatus_UNAVAILABLE);

    serialized_class<TactileData>(m, "TactileData", "One taxel-major tactile measurement frame.")
        .def(py::init(
                 [](const std::vector<float>& values, const std::vector<TaxelStatus>& status)
                 {
                     TactileDataT native;
                     native.values = values;
                     native.status = status;
                     return pack<TactileData>(native);
                 }),
             py::arg("values") = std::vector<float>{}, py::arg("status") = std::vector<TaxelStatus>{})
        .def_property_readonly("values", vector_field(&TactileData::values))
        .def_property_readonly("status",
                               [](const Serialized<TactileData>& self)
                               {
                                   std::vector<TaxelStatus> status;
                                   const auto* encoded = self->status();
                                   if (encoded != nullptr)
                                   {
                                       status.reserve(encoded->size());
                                       for (uint8_t value : *encoded)
                                       {
                                           status.push_back(static_cast<TaxelStatus>(value));
                                       }
                                   }
                                   return status;
                               })
        .def("__repr__",
             [](const Serialized<TactileData>& self)
             {
                 return "TactileData(values=" + std::to_string(self->values() != nullptr ? self->values()->size() : 0) +
                        ")";
             });

    m.def(
        "validate_tactile_data", [](const Serialized<TactileMetadata>& metadata, const Serialized<TactileData>& data)
        { validate_tactile_data(*metadata, *data); }, py::arg("metadata"), py::arg("data"),
        "Validate a tactile frame against its immutable metadata.");

    bind_record<TactileDataRecord, TactileData>(m, "TactileDataRecord", "TactileData");
}

} // namespace core
