// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "schema_array_views.h"
#include "schema_serialized.h"

#include <pybind11/pybind11.h>
#include <schema/soma_hand_v0_generated.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace py = pybind11;

namespace core
{

inline const SomaJointRotationV0& first_soma_hand_rotation(const py::object& self)
{
    return *(*self.cast<const SomaHandJointRotationArrayV0&>().values())[0];
}

constexpr py::ssize_t SOMA_HAND_JOINT_STRIDE = static_cast<py::ssize_t>(sizeof(SomaJointRotationV0));
constexpr py::ssize_t SOMA_HAND_JOINT_COUNT = static_cast<py::ssize_t>(SomaHandJointV0_NUM_JOINTS);

static_assert(sizeof(SomaHandJointRotationArrayV0) ==
                  sizeof(SomaJointRotationV0) * static_cast<size_t>(SomaHandJointV0_NUM_JOINTS),
              "SomaHandJointRotationArrayV0.values length must equal SomaHandJointV0::NUM_JOINTS");

inline void bind_soma_hand_v0(py::module& m)
{
    py::enum_<SomaHandJointV0>(m, "SomaHandJointV0")
        .value("WRIST", SomaHandJointV0_WRIST)
        .value("THUMB1", SomaHandJointV0_THUMB1)
        .value("THUMB2", SomaHandJointV0_THUMB2)
        .value("THUMB3", SomaHandJointV0_THUMB3)
        .value("THUMB_END", SomaHandJointV0_THUMB_END)
        .value("INDEX1", SomaHandJointV0_INDEX1)
        .value("INDEX2", SomaHandJointV0_INDEX2)
        .value("INDEX3", SomaHandJointV0_INDEX3)
        .value("INDEX4", SomaHandJointV0_INDEX4)
        .value("INDEX_END", SomaHandJointV0_INDEX_END)
        .value("MIDDLE1", SomaHandJointV0_MIDDLE1)
        .value("MIDDLE2", SomaHandJointV0_MIDDLE2)
        .value("MIDDLE3", SomaHandJointV0_MIDDLE3)
        .value("MIDDLE4", SomaHandJointV0_MIDDLE4)
        .value("MIDDLE_END", SomaHandJointV0_MIDDLE_END)
        .value("RING1", SomaHandJointV0_RING1)
        .value("RING2", SomaHandJointV0_RING2)
        .value("RING3", SomaHandJointV0_RING3)
        .value("RING4", SomaHandJointV0_RING4)
        .value("RING_END", SomaHandJointV0_RING_END)
        .value("PINKY1", SomaHandJointV0_PINKY1)
        .value("PINKY2", SomaHandJointV0_PINKY2)
        .value("PINKY3", SomaHandJointV0_PINKY3)
        .value("PINKY4", SomaHandJointV0_PINKY4)
        .value("PINKY_END", SomaHandJointV0_PINKY_END)
        .value("NUM_JOINTS", SomaHandJointV0_NUM_JOINTS);

    py::class_<SomaHandJointRotationArrayV0>(m, "SomaHandJointRotationArrayV0")
        .def(py::init<>())
        .def(
            "values",
            [](const SomaHandJointRotationArrayV0& self, size_t index) -> const SomaJointRotationV0*
            {
                if (index >= static_cast<size_t>(SomaHandJointV0_NUM_JOINTS))
                {
                    throw py::index_error("SomaHandJointRotationArrayV0 index out of range");
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
            [offset = FBS_FIELD_OFFSET(SomaJointRotationV0, is_valid)](py::object self)
            {
                const auto* first = fbs_field_address<uint8_t>(first_soma_hand_rotation(self), offset);
                return strided_field_view<uint8_t>(self, first, SOMA_HAND_JOINT_STRIDE, SOMA_HAND_JOINT_COUNT, 0);
            },
            "Per-joint validity as a writable (25,) uint8 view.");

    serialized_class<SomaHandJointRotationsV0>(m, "SomaHandJointRotationsV0", "Encoded SOMA hand joint rotations v0.")
        .def(py::init(
                 [](const SomaHandJointRotationArrayV0& joint_rotations, const Point& global_translation,
                    bool global_translation_is_valid, SomaHandednessV0 handedness)
                 {
                     SomaHandJointRotationsV0T native;
                     native.joint_rotations = std::make_shared<SomaHandJointRotationArrayV0>(joint_rotations);
                     native.global_translation = std::make_shared<Point>(global_translation);
                     native.global_translation_is_valid = global_translation_is_valid;
                     native.handedness = handedness;
                     return pack<SomaHandJointRotationsV0>(native);
                 }),
             py::arg("joint_rotations") = SomaHandJointRotationArrayV0(), py::arg("global_translation") = Point(),
             py::arg("global_translation_is_valid") = false, py::arg("handedness") = SomaHandednessV0_UNSPECIFIED)
        .def_property_readonly("joint_rotations", field(&SomaHandJointRotationsV0::joint_rotations),
                               py::return_value_policy::reference_internal)
        .def_property_readonly("global_translation", field(&SomaHandJointRotationsV0::global_translation),
                               py::return_value_policy::reference_internal)
        .def_property_readonly(
            "global_translation_is_valid", field(&SomaHandJointRotationsV0::global_translation_is_valid))
        .def_property_readonly("handedness", field(&SomaHandJointRotationsV0::handedness));

    bind_record<SomaHandJointRotationsV0Record, SomaHandJointRotationsV0>(
        m, "SomaHandJointRotationsV0Record", "SomaHandJointRotationsV0");
}

} // namespace core
