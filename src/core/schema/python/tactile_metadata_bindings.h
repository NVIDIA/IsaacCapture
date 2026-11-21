// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Encoded tactile metadata views. Pose arrays use rows (x, y, z, qx, qy, qz, qw).
// Complete-layout semantics are validated by TactileMetadata using the shared schema validator.

#pragma once

#include "schema_array_views.h"
#include "schema_serialized.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <schema/tactile_metadata_generated.h>
#include <schema/tactile_metadata_validation.hpp>

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace core
{

inline void bind_tactile_metadata(py::module& m)
{
    using FloatRows = py::array_t<float, py::array::c_style | py::array::forcecast>;
    using IndexRows = py::array_t<uint32_t, py::array::c_style | py::array::forcecast>;

    py::enum_<SegmentSource>(m, "SegmentSource")
        .value("NONE", SegmentSource_NONE)
        .value("HandSkeletonRef", SegmentSource_HandSkeletonRef);
    py::enum_<AlgorithmSource>(m, "AlgorithmSource").value("NONE", AlgorithmSource_NONE).value("LBS", AlgorithmSource_LBS);

    serialized_class<HandSkeletonRef>(
        m, "HandSkeletonRef", "Ordered hand joint identifiers referenced by tactile segments.")
        .def(py::init(
                 [](const std::vector<HandJoint>& joint_ids)
                 {
                     HandSkeletonRefT native;
                     native.joint_ids = joint_ids;
                     return pack<HandSkeletonRef>(native);
                 }),
             py::arg("joint_ids"))
        .def_property_readonly("joint_ids",
                               [](const Serialized<HandSkeletonRef>& self)
                               {
                                   // FlatBuffers exposes enum-vector storage as uint8_t.
                                   std::vector<HandJoint> joint_ids;
                                   joint_ids.reserve(self->joint_ids()->size());
                                   for (const auto joint : *self->joint_ids())
                                   {
                                       joint_ids.push_back(static_cast<HandJoint>(joint));
                                   }
                                   return joint_ids;
                               });

    serialized_class<LBS>(m, "LBS", "Linear blend skinning bind poses and four-slot influences.")
        .def(py::init(
                 [](const FloatRows& segment_bind_poses, const FloatRows& taxel_bind_poses,
                    const IndexRows& taxel_segment_indices, const FloatRows& taxel_segment_weights)
                 {
                     LBST native;
                     native.segment_bind_poses =
                         structs_from_rows<Pose, float, 7>(segment_bind_poses, "segment_bind_poses");
                     native.taxel_bind_poses = structs_from_rows<Pose, float, 7>(taxel_bind_poses, "taxel_bind_poses");
                     native.taxel_segment_indices =
                         structs_from_rows<UInt4, uint32_t, 4>(taxel_segment_indices, "taxel_segment_indices");
                     native.taxel_segment_weights =
                         structs_from_rows<Float4, float, 4>(taxel_segment_weights, "taxel_segment_weights");
                     return pack<LBS>(native);
                 }),
             py::arg("segment_bind_poses"), py::arg("taxel_bind_poses"), py::arg("taxel_segment_indices"),
             py::arg("taxel_segment_weights"),
             "Encode bind poses as (J, 7)/(N, 7) XYZ/XYZW rows and influences as (N, 4) arrays.")
        .def_property_readonly(
            "segment_bind_poses", [](py::object self)
            { return struct_vector_view<float, 7>(self, self.cast<const Serialized<LBS>&>()->segment_bind_poses()); })
        .def_property_readonly(
            "taxel_bind_poses", [](py::object self)
            { return struct_vector_view<float, 7>(self, self.cast<const Serialized<LBS>&>()->taxel_bind_poses()); })
        .def_property_readonly("taxel_segment_indices",
                               [](py::object self) {
                                   return struct_vector_view<uint32_t, 4>(
                                       self, self.cast<const Serialized<LBS>&>()->taxel_segment_indices());
                               })
        .def_property_readonly(
            "taxel_segment_weights", [](py::object self)
            { return struct_vector_view<float, 4>(self, self.cast<const Serialized<LBS>&>()->taxel_segment_weights()); });

    serialized_class<TactileMetadata>(m, "TactileMetadata", "Static tactile layout and its segment source.")
        .def(py::init(
                 [](const Serialized<LBS>& geometry_mapping, const Serialized<HandSkeletonRef>& segment_source)
                 {
                     TactileMetadataT native;
                     LBST lbs;
                     geometry_mapping->UnPackTo(&lbs);
                     native.geometry_mapping.Set(std::move(lbs));
                     HandSkeletonRefT hand;
                     segment_source->UnPackTo(&hand);
                     native.segment_source.Set(std::move(hand));
                     auto metadata = pack<TactileMetadata>(native);
                     try
                     {
                         verify_tactile_metadata(*metadata);
                     }
                     catch (const std::runtime_error& error)
                     {
                         throw py::value_error(error.what());
                     }
                     return metadata;
                 }),
             py::arg("geometry_mapping"), py::arg("segment_source"))
        .def_property_readonly("geometry_mapping_type", field(&TactileMetadata::geometry_mapping_type))
        .def_property_readonly("segment_source_type", field(&TactileMetadata::segment_source_type))
        .def_property_readonly("geometry_mapping",
                               [](const Serialized<TactileMetadata>& self) -> std::optional<Serialized<LBS>>
                               {
                                   if (self->geometry_mapping_type() == AlgorithmSource_NONE)
                                   {
                                       return std::nullopt;
                                   }
                                   if (const auto* geometry_mapping = self->geometry_mapping_as_LBS())
                                   {
                                       return self.narrow(geometry_mapping);
                                   }
                                   throw py::value_error("unsupported tactile geometry mapping");
                               })
        .def_property_readonly("segment_source",
                               [](const Serialized<TactileMetadata>& self) -> std::optional<Serialized<HandSkeletonRef>>
                               {
                                   if (self->segment_source_type() == SegmentSource_NONE)
                                   {
                                       return std::nullopt;
                                   }
                                   if (const auto* segment_source = self->segment_source_as_HandSkeletonRef())
                                   {
                                       return self.narrow(segment_source);
                                   }
                                   throw py::value_error("unsupported tactile segment source");
                               });
}

} // namespace core
