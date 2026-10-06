// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "schema_array_views.h"
#include "schema_serialized.h"

#include <pybind11/pybind11.h>
#include <schema/soma_body_joint_rotations_generated.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace py = pybind11;

namespace core
{

inline const SomaJointRotation& first_soma_body_joint_rotations_rotation(const py::object& self)
{
    return *(*self.cast<const SomaBodyJointRotationArray&>().values())[0];
}

constexpr py::ssize_t SOMA_BODY_JOINT_STRIDE = static_cast<py::ssize_t>(sizeof(SomaJointRotation));
constexpr py::ssize_t SOMA_BODY_JOINT_COUNT = static_cast<py::ssize_t>(SomaBodyJoint_NUM_JOINTS);

static_assert(sizeof(SomaBodyJointRotationArray) ==
                  sizeof(SomaJointRotation) * static_cast<size_t>(SomaBodyJoint_NUM_JOINTS),
              "SomaBodyJointRotationArray.values length must equal SomaBodyJoint::NUM_JOINTS");

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

    py::class_<SomaBodyJointRotationArray>(m, "SomaBodyJointRotationArray")
        .def(py::init<>())
        .def(
            "values",
            [](const SomaBodyJointRotationArray& self, size_t index) -> const SomaJointRotation*
            {
                if (index >= static_cast<size_t>(SomaBodyJoint_NUM_JOINTS))
                {
                    throw py::index_error("SomaBodyJointRotationArray index out of range");
                }
                return (*self.values())[index];
            },
            py::arg("index"), py::return_value_policy::reference_internal)
        .def_property_readonly(
            "rotations",
            [](py::object self)
            {
                const auto* first =
                    reinterpret_cast<const float*>(&first_soma_body_joint_rotations_rotation(self).rotation());
                return strided_field_view<float>(self, first, SOMA_BODY_JOINT_STRIDE, SOMA_BODY_JOINT_COUNT, 4);
            },
            "Unit XYZW quaternions as a writable (77, 4) float32 view.")
        .def_property_readonly(
            "is_valid",
            [offset = FBS_FIELD_OFFSET(SomaJointRotation, is_valid)](py::object self)
            {
                const auto* first = fbs_field_address<uint8_t>(first_soma_body_joint_rotations_rotation(self), offset);
                return strided_field_view<uint8_t>(self, first, SOMA_BODY_JOINT_STRIDE, SOMA_BODY_JOINT_COUNT, 0);
            },
            "Per-joint validity as a writable (77,) uint8 view.");

    serialized_class<SomaBodyJointRotations>(m, "SomaBodyJointRotations", "Encoded SOMA body joint rotations.")
        .def(py::init(
                 [](const SomaBodyJointRotationArray& joint_rotations, const Point& global_translation,
                    bool global_translation_is_valid)
                 {
                     SomaBodyJointRotationsT native;
                     native.joint_rotations = std::make_shared<SomaBodyJointRotationArray>(joint_rotations);
                     native.global_translation = std::make_shared<Point>(global_translation);
                     native.global_translation_is_valid = global_translation_is_valid;
                     return pack<SomaBodyJointRotations>(native);
                 }),
             py::arg("joint_rotations") = SomaBodyJointRotationArray(), py::arg("global_translation") = Point(),
             py::arg("global_translation_is_valid") = false)
        .def_property_readonly("joint_rotations", field(&SomaBodyJointRotations::joint_rotations),
                               py::return_value_policy::reference_internal)
        .def_property_readonly("global_translation", field(&SomaBodyJointRotations::global_translation),
                               py::return_value_policy::reference_internal)
        .def_property_readonly(
            "global_translation_is_valid", field(&SomaBodyJointRotations::global_translation_is_valid));

    bind_record<SomaBodyJointRotationsRecord, SomaBodyJointRotations>(
        m, "SomaBodyJointRotationsRecord", "SomaBodyJointRotations");
}

} // namespace core
