// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// pybind11 bindings — public surface is ``camera_receiver.codec``.

#include "cuda_buffers.hpp"
#include "nv_video_decoder.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace py = pybind11;
using camera_receiver::codec::Codec;
using camera_receiver::codec::DecoderConfig;
using camera_receiver::codec::NvVideoDecoder;

PYBIND11_MODULE(_camera_receiver_codec, m)
{
    m.doc() = "Native NVDEC H.264/HEVC decoder for camera_receiver.";

    py::enum_<Codec>(m, "Codec").value("H264", Codec::H264).value("HEVC", Codec::HEVC);

    py::class_<DecoderConfig>(m, "DecoderConfig")
        .def(py::init<>())
        .def_readwrite("width", &DecoderConfig::width)
        .def_readwrite("height", &DecoderConfig::height)
        .def_readwrite("full_range", &DecoderConfig::full_range)
        .def_readwrite("gpu_id", &DecoderConfig::gpu_id)
        .def_readwrite("codec", &DecoderConfig::codec);

    py::class_<NvVideoDecoder>(m, "NvVideoDecoder")
        .def(py::init<const DecoderConfig&>(), py::arg("config"))
        .def(
            "decode",
            [](NvVideoDecoder& self, const py::buffer& packet, const py::object& rgba_out, std::uint32_t width,
               std::uint32_t height, int rotation) -> bool
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
                // Destination capacity must match the decoder, not caller-supplied dimensions.
                if (width != self.width() || height != self.height())
                    throw std::runtime_error("width/height do not match the configured decoder size");
                const CudaBuf buf = extract_cuda_buffer(rgba_out, width, height);
                bool produced = false;
                {
                    py::gil_scoped_release release;
                    produced = self.decode(static_cast<const std::uint8_t*>(pkt.ptr),
                                           static_cast<std::size_t>(pkt.size), buf.ptr, buf.row_pitch_bytes, rotation);
                }
                return produced;
            },
            py::arg("packet"), py::arg("rgba_out"), py::arg("width"), py::arg("height"), py::arg("rotation") = 0,
            "Feed one Annex-B AU. Returns True iff a frame was written to rgba_out.")
        .def(
            "decode_nv12",
            [](NvVideoDecoder& self, const py::buffer& packet, const py::object& nv12_out, std::uint32_t width,
               std::uint32_t height, int rotation) -> bool
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
                // Destination capacity must match the decoder, not caller-supplied dimensions.
                if (width != self.width() || height != self.height())
                    throw std::runtime_error("width/height do not match the configured decoder size");
                const CudaBuf buf = extract_nv12_buffer(nv12_out, width, height);
                bool produced = false;
                {
                    py::gil_scoped_release release;
                    produced =
                        self.decode_nv12(static_cast<const std::uint8_t*>(pkt.ptr), static_cast<std::size_t>(pkt.size),
                                         buf.ptr, buf.row_pitch_bytes, rotation);
                }
                return produced;
            },
            py::arg("packet"), py::arg("nv12_out"), py::arg("width"), py::arg("height"), py::arg("rotation") = 0,
            "Feed one Annex-B AU. Returns True iff a frame was copied to a tight NV12 GPU buffer.")
        .def("reset", &NvVideoDecoder::reset, "Tear down NVDEC state. Use after long stream silence.");
}
