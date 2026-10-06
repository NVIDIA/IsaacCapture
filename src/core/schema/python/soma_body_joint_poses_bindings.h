// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "schema_array_views.h"
#include "schema_serialized.h"

#include <pybind11/pybind11.h>
#include <schema/soma_body_joint_poses_generated.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace py = pybind11;

namespace core
{

inline const SomaBodyJointPose& first_soma_body_joint_pose(const py::object& self)
{
    return *(*self.cast<const SomaBodyJointPoseArray&>().values())[0];
}

constexpr py::ssize_t SOMA_BODY_JOINT_POSE_STRIDE = static_cast<py::ssize_t>(sizeof(SomaBodyJointPose));
constexpr py::ssize_t SOMA_BODY_JOINT_POSE_COUNT = static_cast<py::ssize_t>(SomaBodyJoint_NUM_JOINTS);

static_assert(sizeof(SomaBodyJointPoseArray) ==
                  sizeof(SomaBodyJointPose) * static_cast<size_t>(SomaBodyJoint_NUM_JOINTS),
              "SomaBodyJointPoseArray.values length must equal SomaBodyJoint::NUM_JOINTS");

inline void bind_soma_body_joint_poses(py::module& m)
{
    py::class_<SomaBodyJointPose>(m, "SomaBodyJointPose")
        .def(py::init<>())
        .def(py::init<const Pose&, bool>(), py::arg("pose"), py::arg("is_valid") = false)
        .def_property_readonly("pose", &SomaBodyJointPose::pose, py::return_value_policy::reference_internal)
        .def_property_readonly("is_valid", &SomaBodyJointPose::is_valid);

    py::class_<SomaBodyJointPoseArray>(m, "SomaBodyJointPoseArray")
        .def(py::init<>())
        .def(
            "values",
            [](const SomaBodyJointPoseArray& self, size_t index) -> const SomaBodyJointPose*
            {
                if (index >= static_cast<size_t>(SomaBodyJoint_NUM_JOINTS))
                {
                    throw py::index_error("SomaBodyJointPoseArray index out of range");
                }
                return (*self.values())[index];
            },
            py::arg("index"), py::return_value_policy::reference_internal)
        .def_property_readonly(
            "positions",
            [](py::object self)
            {
                const auto* first =
                    reinterpret_cast<const float*>(&first_soma_body_joint_pose(self).pose().position());
                return strided_field_view<float>(self, first, SOMA_BODY_JOINT_POSE_STRIDE, SOMA_BODY_JOINT_POSE_COUNT, 3);
            },
            "Joint positions as a writable (77, 3) float32 view in SomaBodyJoint order.")
        .def_property_readonly(
            "orientations",
            [](py::object self)
            {
                const auto* first =
                    reinterpret_cast<const float*>(&first_soma_body_joint_pose(self).pose().orientation());
                return strided_field_view<float>(self, first, SOMA_BODY_JOINT_POSE_STRIDE, SOMA_BODY_JOINT_POSE_COUNT, 4);
            },
            "Unit XYZW quaternions as a writable (77, 4) float32 view.")
        .def_property_readonly(
            "is_valid",
            [offset = FBS_FIELD_OFFSET(SomaBodyJointPose, is_valid)](py::object self)
            {
                const auto* first = fbs_field_address<uint8_t>(first_soma_body_joint_pose(self), offset);
                return strided_field_view<uint8_t>(
                    self, first, SOMA_BODY_JOINT_POSE_STRIDE, SOMA_BODY_JOINT_POSE_COUNT, 0);
            },
            "Per-joint validity as a writable (77,) uint8 view.");

    serialized_class<SomaBodyJointPoses>(m, "SomaBodyJointPoses", "Encoded evaluated SOMA body poses.")
        .def(py::init(
                 [](const SomaBodyJointPoseArray& joint_poses)
                 {
                     SomaBodyJointPosesT native;
                     native.joint_poses = std::make_shared<SomaBodyJointPoseArray>(joint_poses);
                     return pack<SomaBodyJointPoses>(native);
                 }),
             py::arg("joint_poses") = SomaBodyJointPoseArray())
        .def_property_readonly(
            "joint_poses", field(&SomaBodyJointPoses::joint_poses), py::return_value_policy::reference_internal);

    bind_record<SomaBodyJointPosesRecord, SomaBodyJointPoses>(
        m, "SomaBodyJointPosesRecord", "SomaBodyJointPoses");
}

} // namespace core
