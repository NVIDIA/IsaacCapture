// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "argus_camera.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using camera_viz::argus::ArgusCamera;
using camera_viz::argus::ArgusConfig;
using camera_viz::argus::FrameView;

PYBIND11_MODULE(_camera_viz_argus, m)
{
    py::class_<ArgusConfig>(m, "ArgusConfig")
        .def(py::init<>())
        .def_readwrite("name", &ArgusConfig::name)
        .def_readwrite("sensor_ids", &ArgusConfig::sensor_ids)
        .def_readwrite("sensor_mode", &ArgusConfig::sensor_mode)
        .def_readwrite("width", &ArgusConfig::width)
        .def_readwrite("height", &ArgusConfig::height)
        .def_readwrite("fps", &ArgusConfig::fps)
        .def_readwrite("gpu_id", &ArgusConfig::gpu_id)
        .def_readwrite("full_range", &ArgusConfig::full_range)
        .def_readwrite("swap_uv", &ArgusConfig::swap_uv)
        .def_readwrite("acquire_timeout_ms", &ArgusConfig::acquire_timeout_ms)
        .def_readwrite("repeat_capture", &ArgusConfig::repeat_capture);

    py::class_<FrameView>(m, "FrameView")
        .def_readonly("left_ptr", &FrameView::left_ptr)
        .def_readonly("left_pitch", &FrameView::left_pitch)
        .def_readonly("right_ptr", &FrameView::right_ptr)
        .def_readonly("right_pitch", &FrameView::right_pitch)
        .def_readonly("width", &FrameView::width)
        .def_readonly("height", &FrameView::height)
        .def_readonly("timestamp_ns", &FrameView::timestamp_ns)
        .def_readonly("sequence", &FrameView::sequence)
        .def_readonly("stereo", &FrameView::stereo);

    py::class_<ArgusCamera>(m, "ArgusCamera")
        .def(py::init<const ArgusConfig&>())
        .def("start", &ArgusCamera::start, py::call_guard<py::gil_scoped_release>())
        .def("stop", &ArgusCamera::stop, py::call_guard<py::gil_scoped_release>())
        .def("latest",
             [](ArgusCamera& camera) -> py::object
             {
                 auto frame = camera.latest();
                 if (!frame)
                 {
                     return py::none();
                 }
                 return py::cast(*frame);
             })
        .def_property_readonly("is_stereo", &ArgusCamera::is_stereo)
        .def_property_readonly("width", &ArgusCamera::width)
        .def_property_readonly("height", &ArgusCamera::height);
}
