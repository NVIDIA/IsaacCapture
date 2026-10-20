// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "schema_serialized.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <schema/tactile_metadata_generated.h>
#include <schema/tactile_validation.hpp>

#include <memory>
#include <string>
#include <vector>

namespace py = pybind11;

namespace core
{

inline void bind_tactile_metadata(py::module& m)
{
    py::enum_<TactileQuantity>(m, "TactileQuantity")
        .value("UNKNOWN", TactileQuantity_UNKNOWN)
        .value("FORCE", TactileQuantity_FORCE)
        .value("PRESSURE", TactileQuantity_PRESSURE)
        .value("RAW", TactileQuantity_RAW);

    py::enum_<TactileUnit>(m, "TactileUnit")
        .value("UNKNOWN", TactileUnit_UNKNOWN)
        .value("NEWTON", TactileUnit_NEWTON)
        .value("PASCAL", TactileUnit_PASCAL)
        .value("COUNT", TactileUnit_COUNT);

    py::enum_<TactileValueFrame>(m, "TactileValueFrame")
        .value("NONE", TactileValueFrame_NONE)
        .value("TAXEL", TactileValueFrame_TAXEL);

    py::class_<TactileSkinWeight>(m, "TactileSkinWeight", "One tactile-node skinning influence.")
        .def(py::init<uint32_t, float>(), py::arg("node_index"), py::arg("weight"))
        .def_property_readonly("node_index", &TactileSkinWeight::node_index)
        .def_property_readonly("weight", &TactileSkinWeight::weight);

    serialized_class<TactileMeasurement>(m, "TactileMeasurement", "Semantics shared by each taxel value.")
        .def(py::init(
                 [](TactileQuantity quantity, TactileUnit unit, uint8_t component_count, TactileValueFrame frame)
                 {
                     TactileMeasurementT native;
                     native.quantity = quantity;
                     native.unit = unit;
                     native.component_count = component_count;
                     native.frame = frame;
                     auto encoded = pack<TactileMeasurement>(native);
                     validate_tactile_measurement(*encoded);
                     return encoded;
                 }),
             py::arg("quantity"), py::arg("unit"), py::arg("component_count"), py::arg("frame"))
        .def_property_readonly("quantity", field(&TactileMeasurement::quantity))
        .def_property_readonly("unit", field(&TactileMeasurement::unit))
        .def_property_readonly("component_count", field(&TactileMeasurement::component_count))
        .def_property_readonly("frame", field(&TactileMeasurement::frame));

    serialized_class<TactileNode>(m, "TactileNode", "One deformation node in shape-frame coordinates.")
        .def(py::init(
                 [](const std::string& node_id, const Pose& bind_pose)
                 {
                     TactileNodeT native;
                     native.node_id = node_id;
                     native.bind_pose = std::make_shared<Pose>(bind_pose);
                     return pack<TactileNode>(native);
                 }),
             py::arg("node_id"), py::arg("bind_pose"))
        .def_property_readonly("node_id", string_field(&TactileNode::node_id))
        .def_property_readonly("bind_pose", field(&TactileNode::bind_pose), py::return_value_policy::reference_internal);

    serialized_class<TactileTaxel>(m, "TactileTaxel", "Taxel geometry and optional deformation binding.")
        .def(py::init(
                 [](const std::string& taxel_id, const Pose& pose, const std::vector<TactileSkinWeight>& skin)
                 {
                     TactileTaxelT native;
                     native.taxel_id = taxel_id;
                     native.pose = std::make_shared<Pose>(pose);
                     native.skin = skin;
                     return pack<TactileTaxel>(native);
                 }),
             py::arg("taxel_id"), py::arg("pose"), py::arg("skin") = std::vector<TactileSkinWeight>{})
        .def_property_readonly("taxel_id", string_field(&TactileTaxel::taxel_id))
        .def_property_readonly("pose", field(&TactileTaxel::pose), py::return_value_policy::reference_internal)
        .def_property_readonly("skin",
                               [](const Serialized<TactileTaxel>& self)
                               {
                                   std::vector<TactileSkinWeight> values;
                                   const auto* skin = self->skin();
                                   if (skin != nullptr)
                                   {
                                       values.reserve(skin->size());
                                       for (const auto* influence : *skin)
                                       {
                                           values.push_back(*influence);
                                       }
                                   }
                                   return values;
                               });

    serialized_class<TactileMetadata>(m, "TactileMetadata", "Immutable tactile geometry and semantics.")
        .def(py::init(
                 [](const std::string& shape_frame_id, const Serialized<TactileMeasurement>& measurement,
                    const std::vector<Serialized<TactileTaxel>>& taxels, const std::vector<Serialized<TactileNode>>& nodes,
                    const std::vector<Serialized<SemanticProfile>>& profiles, uint16_t schema_version)
                 {
                     TactileMetadataT native;
                     native.schema_version = schema_version;
                     native.shape_frame_id = shape_frame_id;
                     native.measurement = std::make_shared<TactileMeasurementT>();
                     measurement->UnPackTo(native.measurement.get());
                     native.nodes = to_native_vector(nodes, "nodes");
                     native.taxels = to_native_vector(taxels, "taxels");
                     native.profiles = to_native_vector(profiles, "profiles");
                     auto encoded = pack<TactileMetadata>(native);
                     validate_tactile_metadata(*encoded);
                     return encoded;
                 }),
             py::arg("shape_frame_id"), py::arg("measurement"), py::arg("taxels"),
             py::arg("nodes") = std::vector<Serialized<TactileNode>>{},
             py::arg("profiles") = std::vector<Serialized<SemanticProfile>>{}, py::arg("schema_version") = 1)
        .def_property_readonly("schema_version", field(&TactileMetadata::schema_version))
        .def_property_readonly("shape_frame_id", string_field(&TactileMetadata::shape_frame_id))
        .def_property_readonly(
            "measurement", [](const Serialized<TactileMetadata>& self) { return self.narrow(self->measurement()); })
        .def_property_readonly(
            "nodes", [](const Serialized<TactileMetadata>& self) { return narrow_vector(self, self->nodes()); })
        .def_property_readonly(
            "taxels", [](const Serialized<TactileMetadata>& self) { return narrow_vector(self, self->taxels()); })
        .def_property_readonly(
            "profiles", [](const Serialized<TactileMetadata>& self) { return narrow_vector(self, self->profiles()); })
        .def("__repr__",
             [](const Serialized<TactileMetadata>& self)
             {
                 return "TactileMetadata(shape_frame_id='" +
                        (self->shape_frame_id() != nullptr ? self->shape_frame_id()->str() : std::string{}) +
                        "', nodes=" + std::to_string(self->nodes() != nullptr ? self->nodes()->size() : 0) +
                        ", taxels=" + std::to_string(self->taxels() != nullptr ? self->taxels()->size() : 0) + ")";
             });
}

} // namespace core
