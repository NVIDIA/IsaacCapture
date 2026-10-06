// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "schema_serialized.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <schema/soma_hand_joint_poses_generated.h>

#include <algorithm>
#include <string>
#include <vector>

namespace py = pybind11;

namespace core
{

inline void bind_soma_hand_joint_poses(py::module& m)
{
    py::class_<SomaHandJointPose>(m, "SomaHandJointPose", "One keyed evaluated SOMA hand pose.")
        .def(py::init<SomaHandJoint, const Pose&>(), py::arg("joint"), py::arg("pose"))
        .def_property_readonly("joint", &SomaHandJointPose::joint)
        .def_property_readonly("pose", &SomaHandJointPose::pose, py::return_value_policy::reference_internal)
        .def("__repr__", [](const SomaHandJointPose& self)
             { return "SomaHandJointPose(joint=" + std::string(EnumNameSomaHandJoint(self.joint())) + ")"; });

    serialized_class<SomaHandJointPoses>(m, "SomaHandJointPoses", "Encoded evaluated SOMA hand poses.")
        .def(py::init(
                 [](std::vector<SomaHandJointPose> joint_poses, SomaHandedness handedness)
                 {
                     std::sort(joint_poses.begin(), joint_poses.end(),
                               [](const auto& a, const auto& b) { return a.joint() < b.joint(); });
                     const auto invalid =
                         std::find_if(joint_poses.begin(), joint_poses.end(),
                                      [](const auto& entry) { return entry.joint() >= SomaHandJoint_NUM_JOINTS; });
                     if (invalid != joint_poses.end())
                     {
                         throw py::value_error("joint_poses: NUM_JOINTS is not a joint");
                     }
                     const auto duplicate =
                         std::adjacent_find(joint_poses.begin(), joint_poses.end(),
                                            [](const auto& a, const auto& b) { return a.joint() == b.joint(); });
                     if (duplicate != joint_poses.end())
                     {
                         throw py::value_error("joint_poses: duplicate SomaHandJoint " +
                                               std::string(EnumNameSomaHandJoint(duplicate->joint())));
                     }
                     SomaHandJointPosesT native;
                     native.joint_poses = std::move(joint_poses);
                     native.handedness = handedness;
                     return pack<SomaHandJointPoses>(native);
                 }),
             py::arg("joint_poses") = std::vector<SomaHandJointPose>{},
             py::arg("handedness") = SomaHandedness_UNSPECIFIED)
        .def_property_readonly("joint_poses",
                               [](const Serialized<SomaHandJointPoses>& self)
                               {
                                   std::vector<SomaHandJointPose> out;
                                   const auto* entries = self->joint_poses();
                                   if (entries != nullptr)
                                   {
                                       out.reserve(entries->size());
                                       for (const auto* entry : *entries)
                                       {
                                           out.push_back(*entry);
                                       }
                                   }
                                   return out;
                               })
        .def(
            "lookup",
            [](const Serialized<SomaHandJointPoses>& self, SomaHandJoint joint) -> py::object
            {
                const auto* entries = self->joint_poses();
                const auto* found = entries != nullptr ? entries->LookupByKey(joint) : nullptr;
                return found != nullptr ? py::cast(*found) : py::none();
            },
            py::arg("joint"), "Return one provided joint pose, or None when absent.")
        .def_property_readonly("handedness", field(&SomaHandJointPoses::handedness));

    bind_record<SomaHandJointPosesRecord, SomaHandJointPoses>(m, "SomaHandJointPosesRecord", "SomaHandJointPoses");
}

} // namespace core
