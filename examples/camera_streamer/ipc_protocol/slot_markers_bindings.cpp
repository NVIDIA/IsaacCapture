// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Thin Python binding over the slot markers. The operations live in the header, shared with the C++ readers.

#include "ipc_protocol/slot_markers.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using camera_ipc::SlotMarkers;

PYBIND11_MODULE(_slot_markers, m)
{
    m.doc() = "Write-attempt markers for camera_receiver's CUDA IPC pixel slots";

    m.attr("LAYOUT_VERSION") = camera_ipc::kMarkerLayoutVersion;
    m.attr("REGION_SIZE") = SlotMarkers::region_size();
    m.attr("MAX_STREAMS") = camera_ipc::kMaxStreams;
    m.attr("MAX_SLOTS") = camera_ipc::kMaxSlots;

    py::class_<SlotMarkers>(m, "SlotMarkers")
        .def_static("create", &SlotMarkers::create, py::arg("name_prefix"), py::arg("slot_counts"))
        .def_static("attach", &SlotMarkers::attach, py::arg("name"), py::arg("size"), py::arg("layout_version"),
                    py::arg("epoch"))
        .def("close", &SlotMarkers::close)
        .def_property_readonly("name", &SlotMarkers::name)
        .def_property_readonly("size", [](const SlotMarkers&) { return SlotMarkers::region_size(); })
        .def_property_readonly("epoch", &SlotMarkers::epoch)
        .def_property_readonly("layout_version", &SlotMarkers::layout_version)
        .def_property_readonly("stream_count", &SlotMarkers::stream_count)
        .def("slot_count", &SlotMarkers::slot_count, py::arg("stream"))
        .def("begin_write", &SlotMarkers::begin_write, py::arg("stream"), py::arg("slot"))
        .def("load", &SlotMarkers::load, py::arg("stream"), py::arg("slot"))
        .def("unchanged", &SlotMarkers::unchanged, py::arg("stream"), py::arg("slot"), py::arg("generation"));
}
