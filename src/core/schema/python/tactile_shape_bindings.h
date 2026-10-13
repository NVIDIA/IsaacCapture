// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Python bindings for the TactileShape FlatBuffer schema (see tactile_shape.fbs), exposed as an
// encoded view. Each joint and taxel attribute reads as a zero-copy NumPy view strided over the
// TactileJoint / TactileTaxel structs; see schema_array_views.h for the aliasing caveat.

#pragma once

#include "schema_array_views.h"
#include "schema_serialized.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <schema/tactile_shape_generated.h>

#include <array>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

namespace py = pybind11;

namespace core
{

// TactileJoint.parent of a root joint: XR_HAND_JOINT_MAX_ENUM_EXT.
constexpr uint32_t TACTILE_ROOT_PARENT = 0x7FFFFFFF;

inline void bind_tactile_shape(py::module& m)
{
    using FloatRows = py::array_t<float, py::array::c_style | py::array::forcecast>;
    using IndexRows = py::array_t<uint32_t, py::array::c_style | py::array::forcecast>;

    constexpr auto joint_stride = static_cast<py::ssize_t>(sizeof(TactileJoint));
    constexpr auto taxel_stride = static_cast<py::ssize_t>(sizeof(TactileTaxel));

    // (count, width) view of one field across a vector of structs; absent reads as (0, width).
    auto struct_field_view = [](py::object self, const auto* first, py::ssize_t stride, size_t count, py::ssize_t width)
    {
        using T = std::remove_cv_t<std::remove_pointer_t<decltype(first)>>;
        if (count == 0)
        {
            return width > 0 ? py::array_t<T>(std::vector<py::ssize_t>{ 0, width }) : py::array_t<T>(py::ssize_t{ 0 });
        }
        return strided_field_view<T>(self, first, stride, static_cast<py::ssize_t>(count), width);
    };
    auto joints_of = [](py::object self) { return self.cast<const Serialized<TactileShape>&>()->joints(); };
    auto taxels_of = [](py::object self) { return self.cast<const Serialized<TactileShape>&>()->taxels(); };

    auto shape = serialized_class<TactileShape>(m, "TactileShape", "Encoded tactile skeleton and taxel layout.");
    shape
        .def(py::init(
                 [](const FloatRows& joint_bind_positions, const FloatRows& joint_bind_orientations,
                    const IndexRows& joint_parents, const FloatRows& taxel_positions, const FloatRows& taxel_normals,
                    const FloatRows& taxel_tangents, const IndexRows& taxel_joint_indices,
                    const FloatRows& taxel_joint_weights)
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
                     native.joints.reserve(joints);
                     for (size_t j = 0; j < joints; ++j)
                     {
                         const uint32_t parent = joint_parents.data()[j];
                         if (parent != TACTILE_ROOT_PARENT && parent >= joints)
                         {
                             throw py::value_error("joint_parents: " + std::to_string(parent) +
                                                   " is neither a joint index nor ROOT_PARENT");
                         }
                         native.joints.emplace_back(Pose(positions[j], orientations[j]), parent);
                     }

                     const auto points = structs_from_rows<Point, float, 3>(taxel_positions, "taxel_positions");
                     const auto normals = structs_from_rows<Point, float, 3>(taxel_normals, "taxel_normals");
                     const auto tangents = structs_from_rows<Point, float, 3>(taxel_tangents, "taxel_tangents");
                     const auto indices = structs_from_rows<std::array<uint32_t, 4>, uint32_t, 4>(
                         taxel_joint_indices, "taxel_joint_indices");
                     const auto weights =
                         structs_from_rows<std::array<float, 4>, float, 4>(taxel_joint_weights, "taxel_joint_weights");
                     const size_t taxels = points.size();
                     if (normals.size() != taxels || tangents.size() != taxels || indices.size() != taxels ||
                         weights.size() != taxels)
                     {
                         throw py::value_error("every taxel_* array must have the same length");
                     }
                     native.taxels.reserve(taxels);
                     for (size_t t = 0; t < taxels; ++t)
                     {
                         for (uint32_t joint : indices[t])
                         {
                             if (joint >= joints)
                             {
                                 throw py::value_error("taxel_joint_indices: " + std::to_string(joint) +
                                                       " is not a joint index");
                             }
                         }
                         native.taxels.emplace_back(points[t], normals[t], tangents[t],
                                                    ::flatbuffers::span<const uint32_t, 4>(indices[t].data(), 4),
                                                    ::flatbuffers::span<const float, 4>(weights[t].data(), 4));
                     }
                     return pack<TactileShape>(native);
                 }),
             py::arg("joint_bind_positions"), py::arg("joint_bind_orientations"), py::arg("joint_parents"),
             py::arg("taxel_positions"), py::arg("taxel_normals"), py::arg("taxel_tangents"),
             py::arg("taxel_joint_indices"), py::arg("taxel_joint_weights"),
             "Encode a tactile shape. Joint arrays are (J, 3), (J, 4) and (J,); taxel arrays are (N, 3) or "
             "(N, 4). Mismatched lengths and out-of-range joint indices raise ValueError; weights are not "
             "checked.")
        .def_property_readonly(
            "joint_bind_positions",
            [=](py::object self)
            {
                const auto* joints = joints_of(self);
                const size_t count = joints != nullptr ? joints->size() : 0;
                const float* first =
                    count > 0 ? reinterpret_cast<const float*>(&joints->Get(0)->bind_pose().position()) : nullptr;
                return struct_field_view(self, first, joint_stride, count, 3);
            },
            "Joint bind positions as a (J, 3) float32 strided view.")
        .def_property_readonly(
            "joint_bind_orientations",
            [=](py::object self)
            {
                const auto* joints = joints_of(self);
                const size_t count = joints != nullptr ? joints->size() : 0;
                const float* first =
                    count > 0 ? reinterpret_cast<const float*>(&joints->Get(0)->bind_pose().orientation()) : nullptr;
                return struct_field_view(self, first, joint_stride, count, 4);
            },
            "Joint bind orientations (XYZW) as a (J, 4) float32 strided view.")
        .def_property_readonly(
            "joint_parents",
            [=, offset = FBS_FIELD_OFFSET(TactileJoint, parent)](py::object self)
            {
                const auto* joints = joints_of(self);
                const size_t count = joints != nullptr ? joints->size() : 0;
                const uint32_t* first = count > 0 ? fbs_field_address<uint32_t>(*joints->Get(0), offset) : nullptr;
                return struct_field_view(self, first, joint_stride, count, 0);
            },
            "Parent joint per joint as a (J,) uint32 strided view; ROOT_PARENT marks a root.")
        .def_property_readonly(
            "taxel_positions",
            [=](py::object self)
            {
                const auto* taxels = taxels_of(self);
                const size_t count = taxels != nullptr ? taxels->size() : 0;
                const float* first = count > 0 ? reinterpret_cast<const float*>(&taxels->Get(0)->position()) : nullptr;
                return struct_field_view(self, first, taxel_stride, count, 3);
            },
            "Taxel bind positions as an (N, 3) float32 strided view.")
        .def_property_readonly(
            "taxel_normals",
            [=](py::object self)
            {
                const auto* taxels = taxels_of(self);
                const size_t count = taxels != nullptr ? taxels->size() : 0;
                const float* first = count > 0 ? reinterpret_cast<const float*>(&taxels->Get(0)->normal()) : nullptr;
                return struct_field_view(self, first, taxel_stride, count, 3);
            },
            "Taxel unit normals as an (N, 3) float32 strided view.")
        .def_property_readonly(
            "taxel_tangents",
            [=](py::object self)
            {
                const auto* taxels = taxels_of(self);
                const size_t count = taxels != nullptr ? taxels->size() : 0;
                const float* first = count > 0 ? reinterpret_cast<const float*>(&taxels->Get(0)->tangent()) : nullptr;
                return struct_field_view(self, first, taxel_stride, count, 3);
            },
            "Taxel unit tangents as an (N, 3) float32 strided view.")
        .def_property_readonly(
            "taxel_joint_indices",
            [=](py::object self)
            {
                const auto* taxels = taxels_of(self);
                const size_t count = taxels != nullptr ? taxels->size() : 0;
                const uint32_t* first = count > 0 ? taxels->Get(0)->joint_indices()->data() : nullptr;
                return struct_field_view(self, first, taxel_stride, count, 4);
            },
            "Skinning joint per slot as an (N, 4) uint32 strided view.")
        .def_property_readonly(
            "taxel_joint_weights",
            [=](py::object self)
            {
                const auto* taxels = taxels_of(self);
                const size_t count = taxels != nullptr ? taxels->size() : 0;
                const float* first = count > 0 ? taxels->Get(0)->joint_weights()->data() : nullptr;
                return struct_field_view(self, first, taxel_stride, count, 4);
            },
            "Skinning weight per slot as an (N, 4) float32 strided view.")
        .def("__repr__",
             [](const Serialized<TactileShape>& self)
             {
                 const auto* joints = self->joints();
                 const auto* taxels = self->taxels();
                 return "TactileShape(joints=" + std::to_string(joints != nullptr ? joints->size() : 0) +
                        ", taxels=" + std::to_string(taxels != nullptr ? taxels->size() : 0) + ")";
             });
    shape.attr("ROOT_PARENT") = TACTILE_ROOT_PARENT;
}

} // namespace core
