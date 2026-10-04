// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Python bindings for the Keyboard FlatBuffer schema.
// Types: KeyboardOutput (table), exposed as an encoded view.

#pragma once

#include "schema_serialized.h"

#include <pybind11/pybind11.h>
#include <schema/keyboard_generated.h>

#include <cstdint>
#include <string>
#include <vector>

namespace py = pybind11;

namespace core
{

inline void bind_keyboard(py::module& m)
{
    serialized_class<KeyboardOutput>(m, "KeyboardOutput", "Encoded raw keyboard press state.")
        .def(py::init(
                 [](std::vector<uint16_t> pressed_keys, bool is_valid)
                 {
                     KeyboardOutputT native;
                     native.pressed_keys = std::move(pressed_keys);
                     native.is_valid = is_valid;
                     return pack<KeyboardOutput>(native);
                 }),
             py::arg("pressed_keys"), py::arg("is_valid"), "Encode a keyboard press-state snapshot.")
        .def_property_readonly("pressed_keys", vector_field(&KeyboardOutput::pressed_keys))
        .def_property_readonly("is_valid", field(&KeyboardOutput::is_valid))
        .def("__repr__",
             [](const Serialized<KeyboardOutput>& self)
             {
                 std::string result = "KeyboardOutput(pressed_keys=[";
                 const auto* keys = self->pressed_keys();
                 if (keys != nullptr)
                 {
                     for (size_t i = 0; i < keys->size(); ++i)
                     {
                         if (i > 0)
                             result += ", ";
                         result += std::to_string((*keys)[i]);
                     }
                 }
                 result += "], is_valid=" + std::to_string(self->is_valid()) + ")";
                 return result;
             });

    bind_record<KeyboardOutputRecord, KeyboardOutput>(m, "KeyboardOutputRecord", "KeyboardOutput");
}

} // namespace core
