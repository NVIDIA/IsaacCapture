// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Python bindings for the GumanManusGloveFrame FlatBuffer schema.
// GumanManusGloveFrame is a table, exposed as an encoded view.

#pragma once

#include "schema_serialized.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <schema/guman_manus_glove_generated.h>

#include <vector>

namespace py = pybind11;

namespace core
{

inline void bind_guman_manus_glove(py::module& m)
{
    serialized_class<GumanManusGloveFrame>(m, "GumanManusGloveFrame", "Guman native Manus-glove hand-skeleton frame.")
        .def(py::init(
                 [](const std::vector<float>& left_hand_pose, const std::vector<float>& right_hand_pose)
                 {
                     GumanManusGloveFrameT native;
                     native.left_hand_pose = left_hand_pose;
                     native.right_hand_pose = right_hand_pose;
                     return pack<GumanManusGloveFrame>(native);
                 }),
             py::arg("left_hand_pose"), py::arg("right_hand_pose"), "Encode a Guman Manus-glove frame.")
        .def_property_readonly("left_hand_pose", vector_field(&GumanManusGloveFrame::left_hand_pose))
        .def_property_readonly("right_hand_pose", vector_field(&GumanManusGloveFrame::right_hand_pose));

    bind_record<GumanManusGloveFrameRecord, GumanManusGloveFrame>(
        m, "GumanManusGloveFrameRecord", "GumanManusGloveFrame");
}

} // namespace core
