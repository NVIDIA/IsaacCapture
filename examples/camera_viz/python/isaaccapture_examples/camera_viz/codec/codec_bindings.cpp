// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// pybind11 bindings — public surface is ``camera_viz.codec``.

#include "h264_decoder.hpp"
#include "h264_encoder.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace py = pybind11;
using camera_viz::codec::DecoderConfig;
using camera_viz::codec::EncoderConfig;
using camera_viz::codec::H264Decoder;
using camera_viz::codec::H264Encoder;
using camera_viz::codec::PixelFormat;

namespace
{

// Extracts device pointer + row pitch from an object exposing
// __cuda_array_interface__. Expects HxWx4 RGBA8 matching the configured
// (width, height).
struct CudaBuf
{
    std::uintptr_t ptr;
    std::size_t row_pitch_bytes;
};

CudaBuf extract_cuda_buffer(const py::object& obj, std::uint32_t expected_width, std::uint32_t expected_height)
{
    if (!py::hasattr(obj, "__cuda_array_interface__"))
    {
        throw std::runtime_error("expected object exposing __cuda_array_interface__");
    }
    py::dict iface = obj.attr("__cuda_array_interface__").cast<py::dict>();
    if (!iface.contains("shape") || !iface.contains("data"))
    {
        throw std::runtime_error("__cuda_array_interface__ missing required keys (shape/data)");
    }

    py::tuple shape = iface["shape"].cast<py::tuple>();
    if (shape.size() != 3)
    {
        throw std::runtime_error("expected 3D array (H, W, 4); got rank " + std::to_string(shape.size()));
    }
    const auto h = shape[0].cast<std::uint32_t>();
    const auto w = shape[1].cast<std::uint32_t>();
    const auto c = shape[2].cast<std::size_t>();
    if (c != 4)
    {
        throw std::runtime_error("expected 4-channel RGBA; got " + std::to_string(c));
    }
    if (h != expected_height || w != expected_width)
    {
        throw std::runtime_error("frame size " + std::to_string(w) + "x" + std::to_string(h) +
                                 " does not match configured " + std::to_string(expected_width) + "x" +
                                 std::to_string(expected_height));
    }

    py::tuple data = iface["data"].cast<py::tuple>();
    const auto ptr = data[0].cast<std::uintptr_t>();

    // Default to tightly packed RGBA8.
    std::size_t row_pitch = static_cast<std::size_t>(w) * 4u;
    if (iface.contains("strides") && !iface["strides"].is_none())
    {
        py::tuple strides = iface["strides"].cast<py::tuple>();
        if (strides.size() != 3)
        {
            throw std::runtime_error("strides rank does not match shape rank");
        }
        const auto row_stride = strides[0].cast<std::ptrdiff_t>();
        const auto pixel_stride = strides[1].cast<std::ptrdiff_t>();
        const auto chan_stride = strides[2].cast<std::ptrdiff_t>();
        if (pixel_stride != 4 || chan_stride != 1)
        {
            throw std::runtime_error("non-contiguous-per-pixel RGBA layout not supported");
        }
        if (row_stride < static_cast<std::ptrdiff_t>(w * 4u))
        {
            throw std::runtime_error("row stride smaller than W*4");
        }
        row_pitch = static_cast<std::size_t>(row_stride);
    }

    return CudaBuf{ ptr, row_pitch };
}

} // namespace

PYBIND11_MODULE(_camera_viz_codec, m)
{
    m.doc() = "Native H.264 NVENC encoder + NVDEC decoder for camera_viz.";

    py::enum_<PixelFormat>(m, "PixelFormat").value("RGBA8", PixelFormat::kRGBA8).value("BGRA8", PixelFormat::kBGRA8);

    py::class_<EncoderConfig>(m, "EncoderConfig")
        .def(py::init<>())
        .def_readwrite("width", &EncoderConfig::width)
        .def_readwrite("height", &EncoderConfig::height)
        .def_readwrite("bitrate_bps", &EncoderConfig::bitrate_bps)
        .def_readwrite("fps", &EncoderConfig::fps)
        .def_readwrite("gop", &EncoderConfig::gop)
        .def_readwrite("gpu_id", &EncoderConfig::gpu_id)
        .def_readwrite("pixel_format", &EncoderConfig::pixel_format);

    py::class_<H264Encoder>(m, "H264Encoder")
        .def(py::init<const EncoderConfig&>(), py::arg("config"))
        .def_property_readonly("width", &H264Encoder::width)
        .def_property_readonly("height", &H264Encoder::height)
        .def(
            "encode",
            [](H264Encoder& self, const py::object& rgba) -> py::bytes
            {
                const CudaBuf buf = extract_cuda_buffer(rgba, self.width(), self.height());
                std::vector<std::uint8_t> packet;
                {
                    py::gil_scoped_release release;
                    packet = self.encode(buf.ptr, buf.row_pitch_bytes);
                }
                return py::bytes(reinterpret_cast<const char*>(packet.data()), packet.size());
            },
            py::arg("rgba"), "Encode one GPU-resident RGBA8 frame (shape (H, W, 4) via __cuda_array_interface__).")
        .def(
            "end_of_stream",
            [](H264Encoder& self) -> py::bytes
            {
                std::vector<std::uint8_t> packet;
                {
                    py::gil_scoped_release release;
                    packet = self.end_of_stream();
                }
                return py::bytes(reinterpret_cast<const char*>(packet.data()), packet.size());
            },
            "Flush remaining packets from NVENC's internal queue.");

    py::class_<DecoderConfig>(m, "DecoderConfig")
        .def(py::init<>())
        .def_readwrite("width", &DecoderConfig::width)
        .def_readwrite("height", &DecoderConfig::height)
        .def_readwrite("full_range", &DecoderConfig::full_range)
        .def_readwrite("gpu_id", &DecoderConfig::gpu_id);

    py::class_<H264Decoder>(m, "H264Decoder")
        .def(py::init<const DecoderConfig&>(), py::arg("config"))
        .def(
            "decode",
            [](H264Decoder& self, const py::buffer& packet, const py::object& rgba_out, std::uint32_t width,
               std::uint32_t height) -> bool
            {
                py::buffer_info pkt = packet.request();
                if (pkt.ndim != 1)
                {
                    throw std::runtime_error("packet must be a 1D byte buffer");
                }
                if (pkt.itemsize != 1)
                {
                    throw std::runtime_error("packet must be a uint8 buffer");
                }
                const CudaBuf buf = extract_cuda_buffer(rgba_out, width, height);
                bool produced = false;
                {
                    py::gil_scoped_release release;
                    produced = self.decode(static_cast<const std::uint8_t*>(pkt.ptr),
                                           static_cast<std::size_t>(pkt.size), buf.ptr, buf.row_pitch_bytes);
                }
                return produced;
            },
            py::arg("packet"), py::arg("rgba_out"), py::arg("width"), py::arg("height"),
            "Feed one Annex-B AU. Returns True iff a frame was written to rgba_out.")
        .def("reset", &H264Decoder::reset, "Tear down NVDEC state. Use after long stream silence.");
}
