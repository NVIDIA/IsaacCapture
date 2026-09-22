// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "schema_array_views.h"
#include "schema_serialized.h"

#include <pybind11/pybind11.h>
#include <schema/soma_body_joint_poses_v0_generated.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace py = pybind11;

namespace core
{

inline const SomaBodyJointPoseV0& first_soma_body_joint_pose_v0(const py::object& self)
{
    return *(*self.cast<const SomaBodyJointPoseArrayV0&>().values())[0];
}

constexpr py::ssize_t SOMA_BODY_JOINT_POSE_STRIDE = static_cast<py::ssize_t>(sizeof(SomaBodyJointPoseV0));
constexpr py::ssize_t SOMA_BODY_JOINT_POSE_COUNT = static_cast<py::ssize_t>(SomaBodyJointV0_NUM_JOINTS);

static_assert(sizeof(SomaBodyJointPoseArrayV0) ==
                  sizeof(SomaBodyJointPoseV0) * static_cast<size_t>(SomaBodyJointV0_NUM_JOINTS),
              "SomaBodyJointPoseArrayV0.values length must equal SomaBodyJointV0::NUM_JOINTS");

inline void bind_soma_body_joint_poses_v0(py::module& m)
{
    py::class_<SomaBodyJointPoseV0>(m, "SomaBodyJointPoseV0")
        .def(py::init<>())
        .def(py::init<const Pose&, bool>(), py::arg("pose"), py::arg("is_valid") = false)
        .def_property_readonly("pose", &SomaBodyJointPoseV0::pose, py::return_value_policy::reference_internal)
        .def_property_readonly("is_valid", &SomaBodyJointPoseV0::is_valid);

    py::class_<SomaBodyJointPoseArrayV0>(m, "SomaBodyJointPoseArrayV0")
        .def(py::init<>())
        .def(
            "values",
            [](const SomaBodyJointPoseArrayV0& self, size_t index) -> const SomaBodyJointPoseV0*
            {
                if (index >= static_cast<size_t>(SomaBodyJointV0_NUM_JOINTS))
                {
                    throw py::index_error("SomaBodyJointPoseArrayV0 index out of range");
                }
                return (*self.values())[index];
            },
            py::arg("index"), py::return_value_policy::reference_internal)
        .def_property_readonly(
            "positions",
            [](py::object self)
            {
                const auto* first =
                    reinterpret_cast<const float*>(&first_soma_body_joint_pose_v0(self).pose().position());
                return strided_field_view<float>(self, first, SOMA_BODY_JOINT_POSE_STRIDE, SOMA_BODY_JOINT_POSE_COUNT, 3);
            },
            "Joint positions as a writable (77, 3) float32 view in SomaBodyJointV0 order.")
        .def_property_readonly(
            "orientations",
            [](py::object self)
            {
                const auto* first =
                    reinterpret_cast<const float*>(&first_soma_body_joint_pose_v0(self).pose().orientation());
                return strided_field_view<float>(self, first, SOMA_BODY_JOINT_POSE_STRIDE, SOMA_BODY_JOINT_POSE_COUNT, 4);
            },
            "Unit XYZW quaternions as a writable (77, 4) float32 view.")
        .def_property_readonly(
            "is_valid",
            [offset = FBS_FIELD_OFFSET(SomaBodyJointPoseV0, is_valid)](py::object self)
            {
                const auto* first = fbs_field_address<uint8_t>(first_soma_body_joint_pose_v0(self), offset);
                return strided_field_view<uint8_t>(
                    self, first, SOMA_BODY_JOINT_POSE_STRIDE, SOMA_BODY_JOINT_POSE_COUNT, 0);
            },
            "Per-joint validity as a writable (77,) uint8 view.");

    serialized_class<SomaBodyJointPosesV0>(m, "SomaBodyJointPosesV0", "Encoded evaluated SOMA body poses v0.")
        .def(py::init(
                 [](const SomaBodyJointPoseArrayV0& joint_poses)
                 {
                     SomaBodyJointPosesV0T native;
                     native.joint_poses = std::make_shared<SomaBodyJointPoseArrayV0>(joint_poses);
                     return pack<SomaBodyJointPosesV0>(native);
                 }),
             py::arg("joint_poses") = SomaBodyJointPoseArrayV0())
        .def_property_readonly(
            "joint_poses", field(&SomaBodyJointPosesV0::joint_poses), py::return_value_policy::reference_internal);

    bind_record<SomaBodyJointPosesV0Record, SomaBodyJointPosesV0>(
        m, "SomaBodyJointPosesV0Record", "SomaBodyJointPosesV0");
}

} // namespace core
