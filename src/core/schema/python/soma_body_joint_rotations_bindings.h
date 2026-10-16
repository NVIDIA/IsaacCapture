// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "schema_serialized.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <schema/soma_body_joint_rotations_generated.h>

#include <algorithm>
#include <string>
#include <vector>

namespace py = pybind11;

namespace core
{

inline void bind_soma_body_joint_rotations(py::module& m)
{
    py::enum_<SomaBodyJoint>(m, "SomaBodyJoint")
        .value("HIPS", SomaBodyJoint_HIPS)
        .value("SPINE1", SomaBodyJoint_SPINE1)
        .value("SPINE2", SomaBodyJoint_SPINE2)
        .value("CHEST", SomaBodyJoint_CHEST)
        .value("NECK1", SomaBodyJoint_NECK1)
        .value("NECK2", SomaBodyJoint_NECK2)
        .value("HEAD", SomaBodyJoint_HEAD)
        .value("HEAD_END", SomaBodyJoint_HEAD_END)
        .value("JAW", SomaBodyJoint_JAW)
        .value("LEFT_EYE", SomaBodyJoint_LEFT_EYE)
        .value("RIGHT_EYE", SomaBodyJoint_RIGHT_EYE)
        .value("LEFT_SHOULDER", SomaBodyJoint_LEFT_SHOULDER)
        .value("LEFT_ARM", SomaBodyJoint_LEFT_ARM)
        .value("LEFT_FORE_ARM", SomaBodyJoint_LEFT_FORE_ARM)
        .value("LEFT_HAND", SomaBodyJoint_LEFT_HAND)
        .value("LEFT_HAND_THUMB1", SomaBodyJoint_LEFT_HAND_THUMB1)
        .value("LEFT_HAND_THUMB2", SomaBodyJoint_LEFT_HAND_THUMB2)
        .value("LEFT_HAND_THUMB3", SomaBodyJoint_LEFT_HAND_THUMB3)
        .value("LEFT_HAND_THUMB_END", SomaBodyJoint_LEFT_HAND_THUMB_END)
        .value("LEFT_HAND_INDEX1", SomaBodyJoint_LEFT_HAND_INDEX1)
        .value("LEFT_HAND_INDEX2", SomaBodyJoint_LEFT_HAND_INDEX2)
        .value("LEFT_HAND_INDEX3", SomaBodyJoint_LEFT_HAND_INDEX3)
        .value("LEFT_HAND_INDEX4", SomaBodyJoint_LEFT_HAND_INDEX4)
        .value("LEFT_HAND_INDEX_END", SomaBodyJoint_LEFT_HAND_INDEX_END)
        .value("LEFT_HAND_MIDDLE1", SomaBodyJoint_LEFT_HAND_MIDDLE1)
        .value("LEFT_HAND_MIDDLE2", SomaBodyJoint_LEFT_HAND_MIDDLE2)
        .value("LEFT_HAND_MIDDLE3", SomaBodyJoint_LEFT_HAND_MIDDLE3)
        .value("LEFT_HAND_MIDDLE4", SomaBodyJoint_LEFT_HAND_MIDDLE4)
        .value("LEFT_HAND_MIDDLE_END", SomaBodyJoint_LEFT_HAND_MIDDLE_END)
        .value("LEFT_HAND_RING1", SomaBodyJoint_LEFT_HAND_RING1)
        .value("LEFT_HAND_RING2", SomaBodyJoint_LEFT_HAND_RING2)
        .value("LEFT_HAND_RING3", SomaBodyJoint_LEFT_HAND_RING3)
        .value("LEFT_HAND_RING4", SomaBodyJoint_LEFT_HAND_RING4)
        .value("LEFT_HAND_RING_END", SomaBodyJoint_LEFT_HAND_RING_END)
        .value("LEFT_HAND_PINKY1", SomaBodyJoint_LEFT_HAND_PINKY1)
        .value("LEFT_HAND_PINKY2", SomaBodyJoint_LEFT_HAND_PINKY2)
        .value("LEFT_HAND_PINKY3", SomaBodyJoint_LEFT_HAND_PINKY3)
        .value("LEFT_HAND_PINKY4", SomaBodyJoint_LEFT_HAND_PINKY4)
        .value("LEFT_HAND_PINKY_END", SomaBodyJoint_LEFT_HAND_PINKY_END)
        .value("RIGHT_SHOULDER", SomaBodyJoint_RIGHT_SHOULDER)
        .value("RIGHT_ARM", SomaBodyJoint_RIGHT_ARM)
        .value("RIGHT_FORE_ARM", SomaBodyJoint_RIGHT_FORE_ARM)
        .value("RIGHT_HAND", SomaBodyJoint_RIGHT_HAND)
        .value("RIGHT_HAND_THUMB1", SomaBodyJoint_RIGHT_HAND_THUMB1)
        .value("RIGHT_HAND_THUMB2", SomaBodyJoint_RIGHT_HAND_THUMB2)
        .value("RIGHT_HAND_THUMB3", SomaBodyJoint_RIGHT_HAND_THUMB3)
        .value("RIGHT_HAND_THUMB_END", SomaBodyJoint_RIGHT_HAND_THUMB_END)
        .value("RIGHT_HAND_INDEX1", SomaBodyJoint_RIGHT_HAND_INDEX1)
        .value("RIGHT_HAND_INDEX2", SomaBodyJoint_RIGHT_HAND_INDEX2)
        .value("RIGHT_HAND_INDEX3", SomaBodyJoint_RIGHT_HAND_INDEX3)
        .value("RIGHT_HAND_INDEX4", SomaBodyJoint_RIGHT_HAND_INDEX4)
        .value("RIGHT_HAND_INDEX_END", SomaBodyJoint_RIGHT_HAND_INDEX_END)
        .value("RIGHT_HAND_MIDDLE1", SomaBodyJoint_RIGHT_HAND_MIDDLE1)
        .value("RIGHT_HAND_MIDDLE2", SomaBodyJoint_RIGHT_HAND_MIDDLE2)
        .value("RIGHT_HAND_MIDDLE3", SomaBodyJoint_RIGHT_HAND_MIDDLE3)
        .value("RIGHT_HAND_MIDDLE4", SomaBodyJoint_RIGHT_HAND_MIDDLE4)
        .value("RIGHT_HAND_MIDDLE_END", SomaBodyJoint_RIGHT_HAND_MIDDLE_END)
        .value("RIGHT_HAND_RING1", SomaBodyJoint_RIGHT_HAND_RING1)
        .value("RIGHT_HAND_RING2", SomaBodyJoint_RIGHT_HAND_RING2)
        .value("RIGHT_HAND_RING3", SomaBodyJoint_RIGHT_HAND_RING3)
        .value("RIGHT_HAND_RING4", SomaBodyJoint_RIGHT_HAND_RING4)
        .value("RIGHT_HAND_RING_END", SomaBodyJoint_RIGHT_HAND_RING_END)
        .value("RIGHT_HAND_PINKY1", SomaBodyJoint_RIGHT_HAND_PINKY1)
        .value("RIGHT_HAND_PINKY2", SomaBodyJoint_RIGHT_HAND_PINKY2)
        .value("RIGHT_HAND_PINKY3", SomaBodyJoint_RIGHT_HAND_PINKY3)
        .value("RIGHT_HAND_PINKY4", SomaBodyJoint_RIGHT_HAND_PINKY4)
        .value("RIGHT_HAND_PINKY_END", SomaBodyJoint_RIGHT_HAND_PINKY_END)
        .value("LEFT_LEG", SomaBodyJoint_LEFT_LEG)
        .value("LEFT_SHIN", SomaBodyJoint_LEFT_SHIN)
        .value("LEFT_FOOT", SomaBodyJoint_LEFT_FOOT)
        .value("LEFT_TOE_BASE", SomaBodyJoint_LEFT_TOE_BASE)
        .value("LEFT_TOE_END", SomaBodyJoint_LEFT_TOE_END)
        .value("RIGHT_LEG", SomaBodyJoint_RIGHT_LEG)
        .value("RIGHT_SHIN", SomaBodyJoint_RIGHT_SHIN)
        .value("RIGHT_FOOT", SomaBodyJoint_RIGHT_FOOT)
        .value("RIGHT_TOE_BASE", SomaBodyJoint_RIGHT_TOE_BASE)
        .value("RIGHT_TOE_END", SomaBodyJoint_RIGHT_TOE_END)
        .value("NUM_JOINTS", SomaBodyJoint_NUM_JOINTS);

    py::class_<SomaBodyJointRotation>(m, "SomaBodyJointRotation", "One keyed SOMA body rotation.")
        .def(py::init<SomaBodyJoint, const Quaternion&>(), py::arg("joint"), py::arg("rotation"))
        .def_property_readonly("joint", &SomaBodyJointRotation::joint)
        .def_property_readonly("rotation", &SomaBodyJointRotation::rotation, py::return_value_policy::reference_internal)
        .def("__repr__", [](const SomaBodyJointRotation& self)
             { return "SomaBodyJointRotation(joint=" + std::string(EnumNameSomaBodyJoint(self.joint())) + ")"; });

    serialized_class<SomaBodyJointRotations>(m, "SomaBodyJointRotations", "Encoded SOMA body joint rotations.")
        .def(py::init(
                 [](std::vector<SomaBodyJointRotation> joint_rotations, const Point& global_translation,
                    bool global_translation_is_valid)
                 {
                     std::sort(joint_rotations.begin(), joint_rotations.end(),
                               [](const auto& a, const auto& b) { return a.joint() < b.joint(); });
                     const auto invalid =
                         std::find_if(joint_rotations.begin(), joint_rotations.end(),
                                      [](const auto& entry) { return entry.joint() >= SomaBodyJoint_NUM_JOINTS; });
                     if (invalid != joint_rotations.end())
                     {
                         throw py::value_error("joint_rotations: NUM_JOINTS is not a joint");
                     }
                     const auto duplicate =
                         std::adjacent_find(joint_rotations.begin(), joint_rotations.end(),
                                            [](const auto& a, const auto& b) { return a.joint() == b.joint(); });
                     if (duplicate != joint_rotations.end())
                     {
                         throw py::value_error("joint_rotations: duplicate SomaBodyJoint " +
                                               std::string(EnumNameSomaBodyJoint(duplicate->joint())));
                     }
                     SomaBodyJointRotationsT native;
                     native.joint_rotations = std::move(joint_rotations);
                     native.global_translation = std::make_shared<Point>(global_translation);
                     native.global_translation_is_valid = global_translation_is_valid;
                     return pack<SomaBodyJointRotations>(native);
                 }),
             py::arg("joint_rotations") = std::vector<SomaBodyJointRotation>{}, py::arg("global_translation") = Point(),
             py::arg("global_translation_is_valid") = false)
        .def_property_readonly("joint_rotations",
                               [](const Serialized<SomaBodyJointRotations>& self)
                               {
                                   std::vector<SomaBodyJointRotation> out;
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
            [](const Serialized<SomaBodyJointRotations>& self, SomaBodyJoint joint) -> py::object
            {
                const auto* entries = self->joint_rotations();
                const auto* found = entries != nullptr ? entries->LookupByKey(joint) : nullptr;
                return found != nullptr ? py::cast(*found) : py::none();
            },
            py::arg("joint"), "Return one provided joint rotation, or None when absent.")
        .def_property_readonly("global_translation", field(&SomaBodyJointRotations::global_translation),
                               py::return_value_policy::reference_internal)
        .def_property_readonly(
            "global_translation_is_valid", field(&SomaBodyJointRotations::global_translation_is_valid));

    bind_record<SomaBodyJointRotationsRecord, SomaBodyJointRotations>(
        m, "SomaBodyJointRotationsRecord", "SomaBodyJointRotations");
}

} // namespace core
