// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Python bindings for the Keyboard FlatBuffer schema.
// Types: KeyAction (enum), KeyEvent (struct), KeyboardOutput (table), exposed as an encoded view.

#pragma once

#include "schema_serialized.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <schema/keyboard_generated.h>

#include <cstdint>
#include <string>
#include <vector>

namespace py = pybind11;

namespace core
{

inline void bind_keyboard(py::module& m)
{
    // KeyEvent's repr, shared with KeyboardOutput's so its events print the same way.
    const auto key_event_repr = [](const KeyEvent& event)
    {
        return "KeyEvent(code=" + std::to_string(event.code()) +
               ", action=" + (event.action() == KeyAction_Press ? "PRESS" : "RELEASE") +
               ", timestamp_ns=" + std::to_string(event.timestamp_ns()) + ")";
    };

    py::enum_<KeyAction>(m, "KeyAction").value("RELEASE", KeyAction_Release).value("PRESS", KeyAction_Press);

    py::class_<KeyEvent>(m, "KeyEvent")
        .def(py::init<int64_t, uint16_t, KeyAction>(), py::arg("timestamp_ns"), py::arg("code"), py::arg("action"))
        .def_property_readonly("timestamp_ns", &KeyEvent::timestamp_ns)
        .def_property_readonly("code", &KeyEvent::code)
        .def_property_readonly("action", &KeyEvent::action)
        .def("__repr__", key_event_repr);

    serialized_class<KeyboardOutput>(m, "KeyboardOutput", "Encoded keyboard state: held keys plus ordered events.")
        .def(py::init(
                 [](std::vector<uint16_t> held_keys, std::vector<KeyEvent> events)
                 {
                     KeyboardOutputT native;
                     native.held_keys = std::move(held_keys);
                     native.events = std::move(events);
                     return pack<KeyboardOutput>(native);
                 }),
             py::arg("held_keys"), py::arg("events") = std::vector<KeyEvent>{}, "Encode a keyboard snapshot.")
        .def_property_readonly("held_keys", vector_field(&KeyboardOutput::held_keys))
        .def_property_readonly("events",
                               // Structs are copied out by value: they carry no pointers into the buffer.
                               [](const Serialized<KeyboardOutput>& self)
                               {
                                   const auto* events = self->events();
                                   std::vector<KeyEvent> out;
                                   if (events != nullptr)
                                   {
                                       out.reserve(events->size());
                                       for (const auto* event : *events)
                                       {
                                           out.push_back(*event);
                                       }
                                   }
                                   return out;
                               })
        .def("__repr__",
             [key_event_repr](const Serialized<KeyboardOutput>& self)
             {
                 std::string result = "KeyboardOutput(held_keys=[";
                 const auto* keys = self->held_keys();
                 if (keys != nullptr)
                 {
                     for (size_t i = 0; i < keys->size(); ++i)
                     {
                         if (i > 0)
                             result += ", ";
                         result += std::to_string((*keys)[i]);
                     }
                 }
                 result += "], events=[";
                 if (const auto* events = self->events())
                 {
                     for (size_t i = 0; i < events->size(); ++i)
                     {
                         if (i > 0)
                             result += ", ";
                         result += key_event_repr(*events->Get(i));
                     }
                 }
                 return result + "])";
             });

    bind_record<KeyboardOutputRecord, KeyboardOutput>(m, "KeyboardOutputRecord", "KeyboardOutput");
}

} // namespace core
