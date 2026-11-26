// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <schema/soma_common_generated.h>

namespace py = pybind11;

namespace core
{

inline void bind_soma_common(py::module& m)
{
    py::enum_<SomaHandSide>(m, "SomaHandSide")
        .value("UNSPECIFIED", SomaHandSide_UNSPECIFIED)
        .value("LEFT", SomaHandSide_LEFT)
        .value("RIGHT", SomaHandSide_RIGHT);
}

} // namespace core
