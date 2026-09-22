// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "schema_array_views.h"
#include "schema_serialized.h"

#include <pybind11/pybind11.h>
#include <schema/soma_body_v0_generated.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace py = pybind11;

namespace core
{

inline const SomaJointRotationV0& first_soma_body_v0_rotation(const py::object& self)
{
    return *(*self.cast<const SomaBodyJointRotationArrayV0&>().values())[0];
}

constexpr py::ssize_t SOMA_BODY_JOINT_STRIDE = static_cast<py::ssize_t>(sizeof(SomaJointRotationV0));
constexpr py::ssize_t SOMA_BODY_JOINT_COUNT = static_cast<py::ssize_t>(SomaBodyJointV0_NUM_JOINTS);

static_assert(sizeof(SomaBodyJointRotationArrayV0) ==
                  sizeof(SomaJointRotationV0) * static_cast<size_t>(SomaBodyJointV0_NUM_JOINTS),
              "SomaBodyJointRotationArrayV0.values length must equal SomaBodyJointV0::NUM_JOINTS");

inline void bind_soma_body_v0(py::module& m)
{
    py::enum_<SomaBodyJointV0>(m, "SomaBodyJointV0")
        .value("HIPS", SomaBodyJointV0_HIPS)
        .value("SPINE1", SomaBodyJointV0_SPINE1)
        .value("SPINE2", SomaBodyJointV0_SPINE2)
        .value("CHEST", SomaBodyJointV0_CHEST)
        .value("NECK1", SomaBodyJointV0_NECK1)
        .value("NECK2", SomaBodyJointV0_NECK2)
        .value("HEAD", SomaBodyJointV0_HEAD)
        .value("HEAD_END", SomaBodyJointV0_HEAD_END)
        .value("JAW", SomaBodyJointV0_JAW)
        .value("LEFT_EYE", SomaBodyJointV0_LEFT_EYE)
        .value("RIGHT_EYE", SomaBodyJointV0_RIGHT_EYE)
        .value("LEFT_SHOULDER", SomaBodyJointV0_LEFT_SHOULDER)
        .value("LEFT_ARM", SomaBodyJointV0_LEFT_ARM)
        .value("LEFT_FORE_ARM", SomaBodyJointV0_LEFT_FORE_ARM)
        .value("LEFT_HAND", SomaBodyJointV0_LEFT_HAND)
        .value("LEFT_HAND_THUMB1", SomaBodyJointV0_LEFT_HAND_THUMB1)
        .value("LEFT_HAND_THUMB2", SomaBodyJointV0_LEFT_HAND_THUMB2)
        .value("LEFT_HAND_THUMB3", SomaBodyJointV0_LEFT_HAND_THUMB3)
        .value("LEFT_HAND_THUMB_END", SomaBodyJointV0_LEFT_HAND_THUMB_END)
        .value("LEFT_HAND_INDEX1", SomaBodyJointV0_LEFT_HAND_INDEX1)
        .value("LEFT_HAND_INDEX2", SomaBodyJointV0_LEFT_HAND_INDEX2)
        .value("LEFT_HAND_INDEX3", SomaBodyJointV0_LEFT_HAND_INDEX3)
        .value("LEFT_HAND_INDEX4", SomaBodyJointV0_LEFT_HAND_INDEX4)
        .value("LEFT_HAND_INDEX_END", SomaBodyJointV0_LEFT_HAND_INDEX_END)
        .value("LEFT_HAND_MIDDLE1", SomaBodyJointV0_LEFT_HAND_MIDDLE1)
        .value("LEFT_HAND_MIDDLE2", SomaBodyJointV0_LEFT_HAND_MIDDLE2)
        .value("LEFT_HAND_MIDDLE3", SomaBodyJointV0_LEFT_HAND_MIDDLE3)
        .value("LEFT_HAND_MIDDLE4", SomaBodyJointV0_LEFT_HAND_MIDDLE4)
        .value("LEFT_HAND_MIDDLE_END", SomaBodyJointV0_LEFT_HAND_MIDDLE_END)
        .value("LEFT_HAND_RING1", SomaBodyJointV0_LEFT_HAND_RING1)
        .value("LEFT_HAND_RING2", SomaBodyJointV0_LEFT_HAND_RING2)
        .value("LEFT_HAND_RING3", SomaBodyJointV0_LEFT_HAND_RING3)
        .value("LEFT_HAND_RING4", SomaBodyJointV0_LEFT_HAND_RING4)
        .value("LEFT_HAND_RING_END", SomaBodyJointV0_LEFT_HAND_RING_END)
        .value("LEFT_HAND_PINKY1", SomaBodyJointV0_LEFT_HAND_PINKY1)
        .value("LEFT_HAND_PINKY2", SomaBodyJointV0_LEFT_HAND_PINKY2)
        .value("LEFT_HAND_PINKY3", SomaBodyJointV0_LEFT_HAND_PINKY3)
        .value("LEFT_HAND_PINKY4", SomaBodyJointV0_LEFT_HAND_PINKY4)
        .value("LEFT_HAND_PINKY_END", SomaBodyJointV0_LEFT_HAND_PINKY_END)
        .value("RIGHT_SHOULDER", SomaBodyJointV0_RIGHT_SHOULDER)
        .value("RIGHT_ARM", SomaBodyJointV0_RIGHT_ARM)
        .value("RIGHT_FORE_ARM", SomaBodyJointV0_RIGHT_FORE_ARM)
        .value("RIGHT_HAND", SomaBodyJointV0_RIGHT_HAND)
        .value("RIGHT_HAND_THUMB1", SomaBodyJointV0_RIGHT_HAND_THUMB1)
        .value("RIGHT_HAND_THUMB2", SomaBodyJointV0_RIGHT_HAND_THUMB2)
        .value("RIGHT_HAND_THUMB3", SomaBodyJointV0_RIGHT_HAND_THUMB3)
        .value("RIGHT_HAND_THUMB_END", SomaBodyJointV0_RIGHT_HAND_THUMB_END)
        .value("RIGHT_HAND_INDEX1", SomaBodyJointV0_RIGHT_HAND_INDEX1)
        .value("RIGHT_HAND_INDEX2", SomaBodyJointV0_RIGHT_HAND_INDEX2)
        .value("RIGHT_HAND_INDEX3", SomaBodyJointV0_RIGHT_HAND_INDEX3)
        .value("RIGHT_HAND_INDEX4", SomaBodyJointV0_RIGHT_HAND_INDEX4)
        .value("RIGHT_HAND_INDEX_END", SomaBodyJointV0_RIGHT_HAND_INDEX_END)
        .value("RIGHT_HAND_MIDDLE1", SomaBodyJointV0_RIGHT_HAND_MIDDLE1)
        .value("RIGHT_HAND_MIDDLE2", SomaBodyJointV0_RIGHT_HAND_MIDDLE2)
        .value("RIGHT_HAND_MIDDLE3", SomaBodyJointV0_RIGHT_HAND_MIDDLE3)
        .value("RIGHT_HAND_MIDDLE4", SomaBodyJointV0_RIGHT_HAND_MIDDLE4)
        .value("RIGHT_HAND_MIDDLE_END", SomaBodyJointV0_RIGHT_HAND_MIDDLE_END)
        .value("RIGHT_HAND_RING1", SomaBodyJointV0_RIGHT_HAND_RING1)
        .value("RIGHT_HAND_RING2", SomaBodyJointV0_RIGHT_HAND_RING2)
        .value("RIGHT_HAND_RING3", SomaBodyJointV0_RIGHT_HAND_RING3)
        .value("RIGHT_HAND_RING4", SomaBodyJointV0_RIGHT_HAND_RING4)
        .value("RIGHT_HAND_RING_END", SomaBodyJointV0_RIGHT_HAND_RING_END)
        .value("RIGHT_HAND_PINKY1", SomaBodyJointV0_RIGHT_HAND_PINKY1)
        .value("RIGHT_HAND_PINKY2", SomaBodyJointV0_RIGHT_HAND_PINKY2)
        .value("RIGHT_HAND_PINKY3", SomaBodyJointV0_RIGHT_HAND_PINKY3)
        .value("RIGHT_HAND_PINKY4", SomaBodyJointV0_RIGHT_HAND_PINKY4)
        .value("RIGHT_HAND_PINKY_END", SomaBodyJointV0_RIGHT_HAND_PINKY_END)
        .value("LEFT_LEG", SomaBodyJointV0_LEFT_LEG)
        .value("LEFT_SHIN", SomaBodyJointV0_LEFT_SHIN)
        .value("LEFT_FOOT", SomaBodyJointV0_LEFT_FOOT)
        .value("LEFT_TOE_BASE", SomaBodyJointV0_LEFT_TOE_BASE)
        .value("LEFT_TOE_END", SomaBodyJointV0_LEFT_TOE_END)
        .value("RIGHT_LEG", SomaBodyJointV0_RIGHT_LEG)
        .value("RIGHT_SHIN", SomaBodyJointV0_RIGHT_SHIN)
        .value("RIGHT_FOOT", SomaBodyJointV0_RIGHT_FOOT)
        .value("RIGHT_TOE_BASE", SomaBodyJointV0_RIGHT_TOE_BASE)
        .value("RIGHT_TOE_END", SomaBodyJointV0_RIGHT_TOE_END)
        .value("NUM_JOINTS", SomaBodyJointV0_NUM_JOINTS);

    py::class_<SomaBodyJointRotationArrayV0>(m, "SomaBodyJointRotationArrayV0")
        .def(py::init<>())
        .def(
            "values",
            [](const SomaBodyJointRotationArrayV0& self, size_t index) -> const SomaJointRotationV0*
            {
                if (index >= static_cast<size_t>(SomaBodyJointV0_NUM_JOINTS))
                {
                    throw py::index_error("SomaBodyJointRotationArrayV0 index out of range");
                }
                return (*self.values())[index];
            },
            py::arg("index"), py::return_value_policy::reference_internal)
        .def_property_readonly(
            "rotations",
            [](py::object self)
            {
                const auto* first = reinterpret_cast<const float*>(&first_soma_body_v0_rotation(self).rotation());
                return strided_field_view<float>(self, first, SOMA_BODY_JOINT_STRIDE, SOMA_BODY_JOINT_COUNT, 4);
            },
            "Unit XYZW quaternions as a writable (77, 4) float32 view.")
        .def_property_readonly(
            "is_valid",
            [offset = FBS_FIELD_OFFSET(SomaJointRotationV0, is_valid)](py::object self)
            {
                const auto* first = fbs_field_address<uint8_t>(first_soma_body_v0_rotation(self), offset);
                return strided_field_view<uint8_t>(self, first, SOMA_BODY_JOINT_STRIDE, SOMA_BODY_JOINT_COUNT, 0);
            },
            "Per-joint validity as a writable (77,) uint8 view.");

    serialized_class<SomaBodyJointRotationsV0>(m, "SomaBodyJointRotationsV0", "Encoded SOMA body joint rotations v0.")
        .def(py::init(
                 [](const SomaBodyJointRotationArrayV0& joint_rotations, const Point& global_translation,
                    bool global_translation_is_valid)
                 {
                     SomaBodyJointRotationsV0T native;
                     native.joint_rotations = std::make_shared<SomaBodyJointRotationArrayV0>(joint_rotations);
                     native.global_translation = std::make_shared<Point>(global_translation);
                     native.global_translation_is_valid = global_translation_is_valid;
                     return pack<SomaBodyJointRotationsV0>(native);
                 }),
             py::arg("joint_rotations") = SomaBodyJointRotationArrayV0(), py::arg("global_translation") = Point(),
             py::arg("global_translation_is_valid") = false)
        .def_property_readonly("joint_rotations", field(&SomaBodyJointRotationsV0::joint_rotations),
                               py::return_value_policy::reference_internal)
        .def_property_readonly("global_translation", field(&SomaBodyJointRotationsV0::global_translation),
                               py::return_value_policy::reference_internal)
        .def_property_readonly(
            "global_translation_is_valid", field(&SomaBodyJointRotationsV0::global_translation_is_valid));

    bind_record<SomaBodyJointRotationsV0Record, SomaBodyJointRotationsV0>(
        m, "SomaBodyJointRotationsV0Record", "SomaBodyJointRotationsV0");
}

} // namespace core
