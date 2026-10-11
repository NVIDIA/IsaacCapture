// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "schema_serialized.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <schema/soma_hand_joint_rotations_generated.h>

#include <algorithm>
#include <string>
#include <vector>

namespace py = pybind11;

namespace core
{

inline void bind_soma_hand_joint_rotations(py::module& m)
{
    py::enum_<SomaHandJoint>(m, "SomaHandJoint")
        .value("WRIST", SomaHandJoint_WRIST)
        .value("THUMB1", SomaHandJoint_THUMB1)
        .value("THUMB2", SomaHandJoint_THUMB2)
        .value("THUMB3", SomaHandJoint_THUMB3)
        .value("THUMB_END", SomaHandJoint_THUMB_END)
        .value("INDEX1", SomaHandJoint_INDEX1)
        .value("INDEX2", SomaHandJoint_INDEX2)
        .value("INDEX3", SomaHandJoint_INDEX3)
        .value("INDEX4", SomaHandJoint_INDEX4)
        .value("INDEX_END", SomaHandJoint_INDEX_END)
        .value("MIDDLE1", SomaHandJoint_MIDDLE1)
        .value("MIDDLE2", SomaHandJoint_MIDDLE2)
        .value("MIDDLE3", SomaHandJoint_MIDDLE3)
        .value("MIDDLE4", SomaHandJoint_MIDDLE4)
        .value("MIDDLE_END", SomaHandJoint_MIDDLE_END)
        .value("RING1", SomaHandJoint_RING1)
        .value("RING2", SomaHandJoint_RING2)
        .value("RING3", SomaHandJoint_RING3)
        .value("RING4", SomaHandJoint_RING4)
        .value("RING_END", SomaHandJoint_RING_END)
        .value("PINKY1", SomaHandJoint_PINKY1)
        .value("PINKY2", SomaHandJoint_PINKY2)
        .value("PINKY3", SomaHandJoint_PINKY3)
        .value("PINKY4", SomaHandJoint_PINKY4)
        .value("PINKY_END", SomaHandJoint_PINKY_END)
        .value("NUM_JOINTS", SomaHandJoint_NUM_JOINTS);

    py::class_<SomaHandJointRotation>(m, "SomaHandJointRotation", "One keyed SOMA hand rotation.")
        .def(py::init<SomaHandJoint, const Quaternion&>(), py::arg("joint"), py::arg("rotation"))
        .def_property_readonly("joint", &SomaHandJointRotation::joint)
        .def_property_readonly("rotation", &SomaHandJointRotation::rotation, py::return_value_policy::reference_internal)
        .def("__repr__", [](const SomaHandJointRotation& self)
             { return "SomaHandJointRotation(joint=" + std::string(EnumNameSomaHandJoint(self.joint())) + ")"; });

    serialized_class<SomaHandJointRotations>(m, "SomaHandJointRotations", "Encoded SOMA hand joint rotations.")
        .def(py::init(
                 [](std::vector<SomaHandJointRotation> joint_rotations, const Point& global_translation,
                    bool global_translation_is_valid, SomaHandedness handedness)
                 {
                     std::sort(joint_rotations.begin(), joint_rotations.end(),
                               [](const auto& a, const auto& b) { return a.joint() < b.joint(); });
                     const auto invalid =
                         std::find_if(joint_rotations.begin(), joint_rotations.end(),
                                      [](const auto& entry) { return entry.joint() >= SomaHandJoint_NUM_JOINTS; });
                     if (invalid != joint_rotations.end())
                     {
                         throw py::value_error("joint_rotations: NUM_JOINTS is not a joint");
                     }
                     const auto duplicate =
                         std::adjacent_find(joint_rotations.begin(), joint_rotations.end(),
                                            [](const auto& a, const auto& b) { return a.joint() == b.joint(); });
                     if (duplicate != joint_rotations.end())
                     {
                         throw py::value_error("joint_rotations: duplicate SomaHandJoint " +
                                               std::string(EnumNameSomaHandJoint(duplicate->joint())));
                     }
                     SomaHandJointRotationsT native;
                     native.joint_rotations = std::move(joint_rotations);
                     native.global_translation = std::make_shared<Point>(global_translation);
                     native.global_translation_is_valid = global_translation_is_valid;
                     native.handedness = handedness;
                     return pack<SomaHandJointRotations>(native);
                 }),
             py::arg("joint_rotations") = std::vector<SomaHandJointRotation>{}, py::arg("global_translation") = Point(),
             py::arg("global_translation_is_valid") = false, py::arg("handedness") = SomaHandedness_UNSPECIFIED)
        .def_property_readonly("joint_rotations",
                               [](const Serialized<SomaHandJointRotations>& self)
                               {
                                   std::vector<SomaHandJointRotation> out;
                                   const auto* entries = self->joint_rotations();
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
            [](const Serialized<SomaHandJointRotations>& self, SomaHandJoint joint) -> py::object
            {
                const auto* entries = self->joint_rotations();
                const auto* found = entries != nullptr ? entries->LookupByKey(joint) : nullptr;
                return found != nullptr ? py::cast(*found) : py::none();
            },
            py::arg("joint"), "Return one provided joint rotation, or None when absent.")
        .def_property_readonly("global_translation", field(&SomaHandJointRotations::global_translation),
                               py::return_value_policy::reference_internal)
        .def_property_readonly("global_translation_is_valid", field(&SomaHandJointRotations::global_translation_is_valid))
        .def_property_readonly("handedness", field(&SomaHandJointRotations::handedness));

    bind_record<SomaHandJointRotationsRecord, SomaHandJointRotations>(
        m, "SomaHandJointRotationsRecord", "SomaHandJointRotations");
}

} // namespace core
