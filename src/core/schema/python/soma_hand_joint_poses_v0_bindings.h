// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "schema_array_views.h"
#include "schema_serialized.h"

#include <pybind11/pybind11.h>
#include <schema/soma_hand_joint_poses_v0_generated.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace py = pybind11;

namespace core
{

inline const SomaHandJointPoseV0& first_soma_hand_joint_pose_v0(const py::object& self)
{
    return *(*self.cast<const SomaHandJointPoseArrayV0&>().values())[0];
}

constexpr py::ssize_t SOMA_HAND_JOINT_POSE_STRIDE = static_cast<py::ssize_t>(sizeof(SomaHandJointPoseV0));
constexpr py::ssize_t SOMA_HAND_JOINT_POSE_COUNT = static_cast<py::ssize_t>(SomaHandJointV0_NUM_JOINTS);

static_assert(sizeof(SomaHandJointPoseArrayV0) ==
                  sizeof(SomaHandJointPoseV0) * static_cast<size_t>(SomaHandJointV0_NUM_JOINTS),
              "SomaHandJointPoseArrayV0.values length must equal SomaHandJointV0::NUM_JOINTS");

inline void bind_soma_hand_joint_poses_v0(py::module& m)
{
    py::class_<SomaHandJointPoseV0>(m, "SomaHandJointPoseV0")
        .def(py::init<>())
        .def(py::init<const Pose&, bool>(), py::arg("pose"), py::arg("is_valid") = false)
        .def_property_readonly("pose", &SomaHandJointPoseV0::pose, py::return_value_policy::reference_internal)
        .def_property_readonly("is_valid", &SomaHandJointPoseV0::is_valid);

    py::class_<SomaHandJointPoseArrayV0>(m, "SomaHandJointPoseArrayV0")
        .def(py::init<>())
        .def(
            "values",
            [](const SomaHandJointPoseArrayV0& self, size_t index) -> const SomaHandJointPoseV0*
            {
                if (index >= static_cast<size_t>(SomaHandJointV0_NUM_JOINTS))
                {
                    throw py::index_error("SomaHandJointPoseArrayV0 index out of range");
                }
                return (*self.values())[index];
            },
            py::arg("index"), py::return_value_policy::reference_internal)
        .def_property_readonly(
            "positions",
            [](py::object self)
            {
                const auto* first =
                    reinterpret_cast<const float*>(&first_soma_hand_joint_pose_v0(self).pose().position());
                return strided_field_view<float>(self, first, SOMA_HAND_JOINT_POSE_STRIDE, SOMA_HAND_JOINT_POSE_COUNT, 3);
            },
            "Joint positions as a writable (25, 3) float32 view in SomaHandJointV0 order.")
        .def_property_readonly(
            "orientations",
            [](py::object self)
            {
                const auto* first =
                    reinterpret_cast<const float*>(&first_soma_hand_joint_pose_v0(self).pose().orientation());
                return strided_field_view<float>(self, first, SOMA_HAND_JOINT_POSE_STRIDE, SOMA_HAND_JOINT_POSE_COUNT, 4);
            },
            "Unit XYZW quaternions as a writable (25, 4) float32 view.")
        .def_property_readonly(
            "is_valid",
            [offset = FBS_FIELD_OFFSET(SomaHandJointPoseV0, is_valid)](py::object self)
            {
                const auto* first = fbs_field_address<uint8_t>(first_soma_hand_joint_pose_v0(self), offset);
                return strided_field_view<uint8_t>(
                    self, first, SOMA_HAND_JOINT_POSE_STRIDE, SOMA_HAND_JOINT_POSE_COUNT, 0);
            },
            "Per-joint validity as a writable (25,) uint8 view.");

    serialized_class<SomaHandJointPosesV0>(m, "SomaHandJointPosesV0", "Encoded evaluated SOMA hand poses v0.")
        .def(py::init(
                 [](const SomaHandJointPoseArrayV0& joint_poses, SomaHandednessV0 handedness)
                 {
                     SomaHandJointPosesV0T native;
                     native.joint_poses = std::make_shared<SomaHandJointPoseArrayV0>(joint_poses);
                     native.handedness = handedness;
                     return pack<SomaHandJointPosesV0>(native);
                 }),
             py::arg("joint_poses") = SomaHandJointPoseArrayV0(), py::arg("handedness") = SomaHandednessV0_UNSPECIFIED)
        .def_property_readonly(
            "joint_poses", field(&SomaHandJointPosesV0::joint_poses), py::return_value_policy::reference_internal)
        .def_property_readonly("handedness", field(&SomaHandJointPosesV0::handedness));

    bind_record<SomaHandJointPosesV0Record, SomaHandJointPosesV0>(
        m, "SomaHandJointPosesV0Record", "SomaHandJointPosesV0");
}

} // namespace core
