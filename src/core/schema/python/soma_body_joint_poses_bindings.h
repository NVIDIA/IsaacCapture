// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "schema_serialized.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <schema/soma_body_joint_poses_generated.h>

#include <algorithm>
#include <string>
#include <vector>

namespace py = pybind11;

namespace core
{

inline void bind_soma_body_joint_poses(py::module& m)
{
    py::class_<SomaBodyJointPose>(m, "SomaBodyJointPose", "One keyed evaluated SOMA body pose.")
        .def(py::init<SomaBodyJoint, const Pose&>(), py::arg("joint"), py::arg("pose"))
        .def_property_readonly("joint", &SomaBodyJointPose::joint)
        .def_property_readonly("pose", &SomaBodyJointPose::pose, py::return_value_policy::reference_internal)
        .def("__repr__", [](const SomaBodyJointPose& self)
             { return "SomaBodyJointPose(joint=" + std::string(EnumNameSomaBodyJoint(self.joint())) + ")"; });

    serialized_class<SomaBodyJointPoses>(m, "SomaBodyJointPoses", "Encoded evaluated SOMA body poses.")
        .def(py::init(
                 [](std::vector<SomaBodyJointPose> joint_poses)
                 {
                     std::sort(joint_poses.begin(), joint_poses.end(),
                               [](const auto& a, const auto& b) { return a.joint() < b.joint(); });
                     const auto invalid =
                         std::find_if(joint_poses.begin(), joint_poses.end(),
                                      [](const auto& entry) { return entry.joint() >= SomaBodyJoint_NUM_JOINTS; });
                     if (invalid != joint_poses.end())
                     {
                         throw py::value_error("joint_poses: NUM_JOINTS is not a joint");
                     }
                     const auto duplicate =
                         std::adjacent_find(joint_poses.begin(), joint_poses.end(),
                                            [](const auto& a, const auto& b) { return a.joint() == b.joint(); });
                     if (duplicate != joint_poses.end())
                     {
                         throw py::value_error("joint_poses: duplicate SomaBodyJoint " +
                                               std::string(EnumNameSomaBodyJoint(duplicate->joint())));
                     }
                     SomaBodyJointPosesT native;
                     native.joint_poses = std::move(joint_poses);
                     return pack<SomaBodyJointPoses>(native);
                 }),
             py::arg("joint_poses") = std::vector<SomaBodyJointPose>{})
        .def_property_readonly("joint_poses",
                               [](const Serialized<SomaBodyJointPoses>& self)
                               {
                                   std::vector<SomaBodyJointPose> out;
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
            [](const Serialized<SomaBodyJointPoses>& self, SomaBodyJoint joint) -> py::object
            {
                const auto* entries = self->joint_poses();
                const auto* found = entries != nullptr ? entries->LookupByKey(joint) : nullptr;
                return found != nullptr ? py::cast(*found) : py::none();
            },
            py::arg("joint"), "Return one provided joint pose, or None when absent.");

    bind_record<SomaBodyJointPosesRecord, SomaBodyJointPoses>(m, "SomaBodyJointPosesRecord", "SomaBodyJointPoses");
}

} // namespace core
