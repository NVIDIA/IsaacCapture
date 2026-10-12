// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Python bindings for the TactileShape FlatBuffer schema (internal only; see tactile_shape.fbs).
// Types: TactileShape (table) and TactileShapeRecord, exposed as encoded views. Every vector
// reads as a zero-copy NumPy view; see schema_array_views.h for the aliasing caveat.

#pragma once

#include "schema_array_views.h"
#include "schema_serialized.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <schema/tactile_shape_generated.h>

#include <cstdint>
#include <string>
#include <vector>

namespace py = pybind11;

namespace core
{

// joint_parents value of a root joint: XR_HAND_JOINT_MAX_ENUM_EXT.
constexpr uint32_t TACTILE_ROOT_PARENT = 0x7FFFFFFF;

inline void bind_tactile_shape(py::module& m)
{
    using FloatRows = py::array_t<float, py::array::c_style | py::array::forcecast>;
    using IndexRows = py::array_t<uint32_t, py::array::c_style | py::array::forcecast>;

    // Joint rows are interleaved Pose structs, so position and orientation are strided views.
    constexpr auto pose_stride = static_cast<py::ssize_t>(sizeof(Pose));

    auto shape =
        serialized_class<TactileShape>(m, "TactileShape", "Encoded tactile skeleton and taxel layout (internal only).");
    shape
        .def(py::init(
                 [](const FloatRows& joint_bind_positions, const FloatRows& joint_bind_orientations,
                    const IndexRows& joint_parents, const FloatRows& taxel_positions, const FloatRows& taxel_normals,
                    const FloatRows& taxel_tangents, const IndexRows& taxel_joint_indices,
                    const FloatRows& taxel_joint_weights, const std::string& device_id, uint32_t shape_version)
                 {
                     const auto positions =
                         structs_from_rows<Point, float, 3>(joint_bind_positions, "joint_bind_positions");
                     const auto orientations =
                         structs_from_rows<Quaternion, float, 4>(joint_bind_orientations, "joint_bind_orientations");
                     if (joint_parents.ndim() != 1)
                     {
                         throw py::value_error("joint_parents: expected an (J,) array");
                     }
                     const size_t joints = positions.size();
                     if (orientations.size() != joints || static_cast<size_t>(joint_parents.shape(0)) != joints)
                     {
                         throw py::value_error(
                             "joint_bind_positions, joint_bind_orientations and joint_parents must have the same length");
                     }

                     TactileShapeT native;
                     native.joint_count = static_cast<uint32_t>(joints);
                     native.joint_bind_poses.reserve(joints);
                     for (size_t j = 0; j < joints; ++j)
                     {
                         native.joint_bind_poses.emplace_back(positions[j], orientations[j]);
                     }
                     native.joint_parents.assign(joint_parents.data(), joint_parents.data() + joints);
                     for (uint32_t parent : native.joint_parents)
                     {
                         if (parent != TACTILE_ROOT_PARENT && parent >= joints)
                         {
                             throw py::value_error("joint_parents: " + std::to_string(parent) +
                                                   " is neither a joint index nor ROOT_PARENT");
                         }
                     }

                     native.taxel_positions = structs_from_rows<Point, float, 3>(taxel_positions, "taxel_positions");
                     native.taxel_normals = structs_from_rows<Point, float, 3>(taxel_normals, "taxel_normals");
                     native.taxel_tangents = structs_from_rows<Point, float, 3>(taxel_tangents, "taxel_tangents");
                     native.taxel_joint_indices =
                         structs_from_rows<TaxelJointIndices, uint32_t, 4>(taxel_joint_indices, "taxel_joint_indices");
                     native.taxel_joint_weights =
                         structs_from_rows<TaxelJointWeights, float, 4>(taxel_joint_weights, "taxel_joint_weights");
                     const size_t taxels = native.taxel_positions.size();
                     if (native.taxel_normals.size() != taxels || native.taxel_tangents.size() != taxels ||
                         native.taxel_joint_indices.size() != taxels || native.taxel_joint_weights.size() != taxels)
                     {
                         throw py::value_error("every taxel_* array must have the same length");
                     }
                     for (const auto& slots : native.taxel_joint_indices)
                     {
                         for (uint32_t joint : *slots.joints())
                         {
                             if (joint >= joints)
                             {
                                 throw py::value_error("taxel_joint_indices: " + std::to_string(joint) +
                                                       " is not a joint index");
                             }
                         }
                     }
                     native.taxel_count = static_cast<uint32_t>(taxels);
                     native.device_id = device_id;
                     native.shape_version = shape_version;
                     return pack<TactileShape>(native);
                 }),
             py::arg("joint_bind_positions"), py::arg("joint_bind_orientations"), py::arg("joint_parents"),
             py::arg("taxel_positions"), py::arg("taxel_normals"), py::arg("taxel_tangents"),
             py::arg("taxel_joint_indices"), py::arg("taxel_joint_weights"), py::arg("device_id") = std::string{},
             py::arg("shape_version") = 0u,
             "Encode a tactile shape. Joint arrays are (J, 3), (J, 4) and (J,); taxel arrays are (N, 3) or "
             "(N, 4). joint_count and taxel_count are set to J and N. Mismatched lengths and out-of-range "
             "joint indices raise ValueError; weights are not checked.")
        .def_property_readonly("joint_count", field(&TactileShape::joint_count))
        .def_property_readonly("taxel_count", field(&TactileShape::taxel_count))
        .def_property_readonly(
            "joint_bind_positions",
            [pose_stride](py::object self)
            {
                const auto* poses = self.cast<const Serialized<TactileShape>&>()->joint_bind_poses();
                if (poses == nullptr || poses->size() == 0)
                {
                    return py::array_t<float>(std::vector<py::ssize_t>{ 0, 3 });
                }
                const auto* first = reinterpret_cast<const float*>(&poses->Get(0)->position());
                return strided_field_view<float>(self, first, pose_stride, poses->size(), 3);
            },
            "Joint bind positions as a (J, 3) float32 strided view.")
        .def_property_readonly(
            "joint_bind_orientations",
            [pose_stride](py::object self)
            {
                const auto* poses = self.cast<const Serialized<TactileShape>&>()->joint_bind_poses();
                if (poses == nullptr || poses->size() == 0)
                {
                    return py::array_t<float>(std::vector<py::ssize_t>{ 0, 4 });
                }
                const auto* first = reinterpret_cast<const float*>(&poses->Get(0)->orientation());
                return strided_field_view<float>(self, first, pose_stride, poses->size(), 4);
            },
            "Joint bind orientations (XYZW) as a (J, 4) float32 strided view.")
        .def_property_readonly(
            "joint_parents",
            [](py::object self)
            { return scalar_vector_view<uint32_t>(self, self.cast<const Serialized<TactileShape>&>()->joint_parents()); },
            "Parent joint per joint as a (J,) uint32 view; ROOT_PARENT marks a root.")
        .def_property_readonly(
            "taxel_positions",
            [](py::object self) {
                return struct_vector_view<float, 3>(
                    self, self.cast<const Serialized<TactileShape>&>()->taxel_positions());
            },
            "Taxel bind positions as an (N, 3) float32 view.")
        .def_property_readonly(
            "taxel_normals",
            [](py::object self)
            { return struct_vector_view<float, 3>(self, self.cast<const Serialized<TactileShape>&>()->taxel_normals()); },
            "Taxel unit normals as an (N, 3) float32 view.")
        .def_property_readonly(
            "taxel_tangents",
            [](py::object self) {
                return struct_vector_view<float, 3>(self, self.cast<const Serialized<TactileShape>&>()->taxel_tangents());
            },
            "Taxel unit tangents as an (N, 3) float32 view.")
        .def_property_readonly(
            "taxel_joint_indices",
            [](py::object self) {
                return struct_vector_view<uint32_t, 4>(
                    self, self.cast<const Serialized<TactileShape>&>()->taxel_joint_indices());
            },
            "Skinning joint per slot as an (N, 4) uint32 view.")
        .def_property_readonly(
            "taxel_joint_weights",
            [](py::object self) {
                return struct_vector_view<float, 4>(
                    self, self.cast<const Serialized<TactileShape>&>()->taxel_joint_weights());
            },
            "Skinning weight per slot as an (N, 4) float32 view.")
        .def_property_readonly("device_id", string_field(&TactileShape::device_id))
        .def_property_readonly("shape_version", field(&TactileShape::shape_version))
        .def("__repr__",
             [](const Serialized<TactileShape>& self)
             {
                 const auto* device_id = self->device_id();
                 return "TactileShape(joint_count=" + std::to_string(self->joint_count()) +
                        ", taxel_count=" + std::to_string(self->taxel_count()) +
                        ", device_id=" + (device_id != nullptr ? device_id->str() : std::string{}) +
                        ", shape_version=" + std::to_string(self->shape_version()) + ")";
             });
    shape.attr("ROOT_PARENT") = TACTILE_ROOT_PARENT;

    bind_record<TactileShapeRecord, TactileShape>(m, "TactileShapeRecord", "TactileShape");
}

} // namespace core
