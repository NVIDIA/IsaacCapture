// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "schema_array_views.h"
#include "schema_serialized.h"

#include <pybind11/pybind11.h>
#include <schema/soma_hand_joint_poses_generated.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace py = pybind11;

namespace core
{

inline const SomaHandJointPose& first_soma_hand_joint_pose(const py::object& self)
{
    return *(*self.cast<const SomaHandJointPoseArray&>().values())[0];
}

constexpr py::ssize_t SOMA_HAND_JOINT_POSE_STRIDE = static_cast<py::ssize_t>(sizeof(SomaHandJointPose));
constexpr py::ssize_t SOMA_HAND_JOINT_POSE_COUNT = static_cast<py::ssize_t>(SomaHandJoint_NUM_JOINTS);

static_assert(sizeof(SomaHandJointPoseArray) ==
                  sizeof(SomaHandJointPose) * static_cast<size_t>(SomaHandJoint_NUM_JOINTS),
              "SomaHandJointPoseArray.values length must equal SomaHandJoint::NUM_JOINTS");

inline void bind_soma_hand_joint_poses(py::module& m)
{
    py::class_<SomaHandJointPose>(m, "SomaHandJointPose")
        .def(py::init<>())
        .def(py::init<const Pose&, bool>(), py::arg("pose"), py::arg("is_valid") = false)
        .def_property_readonly("pose", &SomaHandJointPose::pose, py::return_value_policy::reference_internal)
        .def_property_readonly("is_valid", &SomaHandJointPose::is_valid);

    py::class_<SomaHandJointPoseArray>(m, "SomaHandJointPoseArray")
        .def(py::init<>())
        .def(
            "values",
            [](const SomaHandJointPoseArray& self, size_t index) -> const SomaHandJointPose*
            {
                if (index >= static_cast<size_t>(SomaHandJoint_NUM_JOINTS))
                {
                    throw py::index_error("SomaHandJointPoseArray index out of range");
                }
                return (*self.values())[index];
            },
            py::arg("index"), py::return_value_policy::reference_internal)
        .def_property_readonly(
            "positions",
            [](py::object self)
            {
                const auto* first =
                    reinterpret_cast<const float*>(&first_soma_hand_joint_pose(self).pose().position());
                return strided_field_view<float>(self, first, SOMA_HAND_JOINT_POSE_STRIDE, SOMA_HAND_JOINT_POSE_COUNT, 3);
            },
            "Joint positions as a writable (25, 3) float32 view in SomaHandJoint order.")
        .def_property_readonly(
            "orientations",
            [](py::object self)
            {
                const auto* first =
                    reinterpret_cast<const float*>(&first_soma_hand_joint_pose(self).pose().orientation());
                return strided_field_view<float>(self, first, SOMA_HAND_JOINT_POSE_STRIDE, SOMA_HAND_JOINT_POSE_COUNT, 4);
            },
            "Unit XYZW quaternions as a writable (25, 4) float32 view.")
        .def_property_readonly(
            "is_valid",
            [offset = FBS_FIELD_OFFSET(SomaHandJointPose, is_valid)](py::object self)
            {
                const auto* first = fbs_field_address<uint8_t>(first_soma_hand_joint_pose(self), offset);
                return strided_field_view<uint8_t>(
                    self, first, SOMA_HAND_JOINT_POSE_STRIDE, SOMA_HAND_JOINT_POSE_COUNT, 0);
            },
            "Per-joint validity as a writable (25,) uint8 view.");

    serialized_class<SomaHandJointPoses>(m, "SomaHandJointPoses", "Encoded evaluated SOMA hand poses.")
        .def(py::init(
                 [](const SomaHandJointPoseArray& joint_poses, SomaHandedness handedness)
                 {
                     SomaHandJointPosesT native;
                     native.joint_poses = std::make_shared<SomaHandJointPoseArray>(joint_poses);
                     native.handedness = handedness;
                     return pack<SomaHandJointPoses>(native);
                 }),
             py::arg("joint_poses") = SomaHandJointPoseArray(), py::arg("handedness") = SomaHandedness_UNSPECIFIED)
        .def_property_readonly(
            "joint_poses", field(&SomaHandJointPoses::joint_poses), py::return_value_policy::reference_internal)
        .def_property_readonly("handedness", field(&SomaHandJointPoses::handedness));

    bind_record<SomaHandJointPosesRecord, SomaHandJointPoses>(
        m, "SomaHandJointPosesRecord", "SomaHandJointPoses");
}

} // namespace core
