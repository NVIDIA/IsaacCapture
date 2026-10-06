// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "schema_array_views.h"
#include "schema_serialized.h"

#include <pybind11/pybind11.h>
#include <schema/soma_hand_joint_rotations_generated.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace py = pybind11;

namespace core
{

inline const SomaJointRotation& first_soma_hand_rotation(const py::object& self)
{
    return *(*self.cast<const SomaHandJointRotationArray&>().values())[0];
}

constexpr py::ssize_t SOMA_HAND_JOINT_STRIDE = static_cast<py::ssize_t>(sizeof(SomaJointRotation));
constexpr py::ssize_t SOMA_HAND_JOINT_COUNT = static_cast<py::ssize_t>(SomaHandJoint_NUM_JOINTS);

static_assert(sizeof(SomaHandJointRotationArray) ==
                  sizeof(SomaJointRotation) * static_cast<size_t>(SomaHandJoint_NUM_JOINTS),
              "SomaHandJointRotationArray.values length must equal SomaHandJoint::NUM_JOINTS");

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

    py::class_<SomaHandJointRotationArray>(m, "SomaHandJointRotationArray")
        .def(py::init<>())
        .def(
            "values",
            [](const SomaHandJointRotationArray& self, size_t index) -> const SomaJointRotation*
            {
                if (index >= static_cast<size_t>(SomaHandJoint_NUM_JOINTS))
                {
                    throw py::index_error("SomaHandJointRotationArray index out of range");
                }
                return (*self.values())[index];
            },
            py::arg("index"), py::return_value_policy::reference_internal)
        .def_property_readonly(
            "rotations",
            [](py::object self)
            {
                const auto* first = reinterpret_cast<const float*>(&first_soma_hand_rotation(self).rotation());
                return strided_field_view<float>(self, first, SOMA_HAND_JOINT_STRIDE, SOMA_HAND_JOINT_COUNT, 4);
            },
            "Unit XYZW quaternions as a writable (25, 4) float32 view.")
        .def_property_readonly(
            "is_valid",
            [offset = FBS_FIELD_OFFSET(SomaJointRotation, is_valid)](py::object self)
            {
                const auto* first = fbs_field_address<uint8_t>(first_soma_hand_rotation(self), offset);
                return strided_field_view<uint8_t>(self, first, SOMA_HAND_JOINT_STRIDE, SOMA_HAND_JOINT_COUNT, 0);
            },
            "Per-joint validity as a writable (25,) uint8 view.");

    serialized_class<SomaHandJointRotations>(m, "SomaHandJointRotations", "Encoded SOMA hand joint rotations.")
        .def(py::init(
                 [](const SomaHandJointRotationArray& joint_rotations, const Point& global_translation,
                    bool global_translation_is_valid, SomaHandedness handedness)
                 {
                     SomaHandJointRotationsT native;
                     native.joint_rotations = std::make_shared<SomaHandJointRotationArray>(joint_rotations);
                     native.global_translation = std::make_shared<Point>(global_translation);
                     native.global_translation_is_valid = global_translation_is_valid;
                     native.handedness = handedness;
                     return pack<SomaHandJointRotations>(native);
                 }),
             py::arg("joint_rotations") = SomaHandJointRotationArray(), py::arg("global_translation") = Point(),
             py::arg("global_translation_is_valid") = false, py::arg("handedness") = SomaHandedness_UNSPECIFIED)
        .def_property_readonly("joint_rotations", field(&SomaHandJointRotations::joint_rotations),
                               py::return_value_policy::reference_internal)
        .def_property_readonly("global_translation", field(&SomaHandJointRotations::global_translation),
                               py::return_value_policy::reference_internal)
        .def_property_readonly(
            "global_translation_is_valid", field(&SomaHandJointRotations::global_translation_is_valid))
        .def_property_readonly("handedness", field(&SomaHandJointRotations::handedness));

    bind_record<SomaHandJointRotationsRecord, SomaHandJointRotations>(
        m, "SomaHandJointRotationsRecord", "SomaHandJointRotations");
}

} // namespace core
