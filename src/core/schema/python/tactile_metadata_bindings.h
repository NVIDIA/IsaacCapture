// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Encoded tactile metadata views. Pose arrays use rows (x, y, z, qx, qy, qz, qw).

#pragma once

#include "schema_array_views.h"
#include "schema_serialized.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <schema/tactile_metadata_generated.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
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
                     std::array<bool, HandJoint_NUM_JOINTS> seen{};
                     for (const auto joint : joint_ids)
                     {
                         if (joint >= HandJoint_NUM_JOINTS || seen[joint])
                         {
                             throw py::value_error(
                                 "joint_ids: expected unique HandJoint identifiers, excluding NUM_JOINTS");
                         }
                         seen[joint] = true;
                     }
                     HandSkeletonRefT native;
                     native.joint_ids = joint_ids;
                     return pack<HandSkeletonRef>(native);
                 }),
             py::arg("joint_ids"))
        .def_property_readonly("joint_ids", vector_field(&HandSkeletonRef::joint_ids));

    auto poses_from_rows = [](const FloatRows& rows, const char* name)
    {
        auto poses = structs_from_rows<Pose, float, 7>(rows, name);
        for (py::ssize_t i = 0; i < rows.shape(0); ++i)
        {
            const float* row = rows.data(i);
            for (int c = 0; c < 7; ++c)
            {
                if (!std::isfinite(row[c]))
                {
                    throw py::value_error(std::string(name) + ": poses must be finite");
                }
            }
            const float norm2 = row[3] * row[3] + row[4] * row[4] + row[5] * row[5] + row[6] * row[6];
            if (std::abs(norm2 - 1.0f) > 1e-4f)
            {
                throw py::value_error(std::string(name) + ": orientations must be unit quaternions (x, y, z, w)");
            }
        }
        return poses;
    };

    serialized_class<LBS>(m, "LBS", "Linear blend skinning bind poses and four-slot influences.")
        .def(py::init(
                 [poses_from_rows](const FloatRows& segment_bind_poses, const FloatRows& taxel_bind_poses,
                                   const IndexRows& taxel_segment_indices, const FloatRows& taxel_segment_weights)
                 {
                     LBST native;
                     native.segment_bind_poses = poses_from_rows(segment_bind_poses, "segment_bind_poses");
                     native.taxel_bind_poses = poses_from_rows(taxel_bind_poses, "taxel_bind_poses");
                     native.taxel_segment_indices =
                         structs_from_rows<UInt4, uint32_t, 4>(taxel_segment_indices, "taxel_segment_indices");
                     native.taxel_segment_weights =
                         structs_from_rows<Float4, float, 4>(taxel_segment_weights, "taxel_segment_weights");
                     const size_t taxels = native.taxel_bind_poses.size();
                     if (native.taxel_segment_indices.size() != taxels || native.taxel_segment_weights.size() != taxels)
                     {
                         throw py::value_error("all taxel_* arrays must have the same length");
                     }
                     for (size_t i = 0; i < taxels; ++i)
                     {
                         float total = 0;
                         for (int k = 0; k < 4; ++k)
                         {
                             if (native.taxel_segment_indices[i].values()->Get(k) >= native.segment_bind_poses.size())
                             {
                                 throw py::value_error("taxel_segment_indices: slot is not a segment index");
                             }
                             const float weight = native.taxel_segment_weights[i].values()->Get(k);
                             if (!std::isfinite(weight) || weight < 0)
                             {
                                 throw py::value_error("taxel_segment_weights: weights must be finite and nonnegative");
                             }
                             total += weight;
                         }
                         if (std::abs(total - 1.0f) > 1e-5f)
                         {
                             throw py::value_error("taxel_segment_weights: each row must sum to one");
                         }
                     }
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
                     const auto* poses = geometry_mapping->segment_bind_poses();
                     const auto* joint_ids = segment_source->joint_ids();
                     if ((poses != nullptr ? poses->size() : 0) != (joint_ids != nullptr ? joint_ids->size() : 0))
                     {
                         throw py::value_error(
                             "segment_bind_poses and segment_source.joint_ids must have the same length");
                     }
                     TactileMetadataT native;
                     LBST lbs;
                     geometry_mapping->UnPackTo(&lbs);
                     native.geometry_mapping.Set(std::move(lbs));
                     HandSkeletonRefT hand;
                     segment_source->UnPackTo(&hand);
                     native.segment_source.Set(std::move(hand));
                     return pack<TactileMetadata>(native);
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
