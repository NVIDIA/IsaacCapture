// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "schema_serialized.h"

#include <schema/orbbec_ego_camera_generated.h>

#include <cstdint>
#include <string>

namespace core
{

inline void bind_orbbec_ego_camera(py::module& m)
{

    py::enum_<OrbbecEgoCameraStream>(m, "OrbbecEgoCameraStream")
        .value("ColorLeft", OrbbecEgoCameraStream_ColorLeft)
        .value("ColorRight", OrbbecEgoCameraStream_ColorRight);

    py::enum_<OrbbecEgoPixelFormat>(m, "OrbbecEgoPixelFormat")
        .value("Mjpg", OrbbecEgoPixelFormat_Mjpg)
        .value("H264", OrbbecEgoPixelFormat_H264)
        .value("H265", OrbbecEgoPixelFormat_H265);

    serialized_class<OrbbecEgoFrameMetadata>(m, "OrbbecEgoFrameMetadata", "Encoded EGO payload.")
        .def(py::init(
                 [](OrbbecEgoCameraStream stream, uint64_t sequence_number, uint32_t width, uint32_t height,
                    uint32_t fps, OrbbecEgoPixelFormat pixel_format, uint64_t encoded_bytes, uint32_t capture_epoch)
                 {
                     OrbbecEgoFrameMetadataT native;
                     native.stream = stream;
                     native.sequence_number = sequence_number;
                     native.width = width;
                     native.height = height;
                     native.fps = fps;
                     native.pixel_format = pixel_format;
                     native.encoded_bytes = encoded_bytes;
                     native.capture_epoch = capture_epoch;
                     return pack<OrbbecEgoFrameMetadata>(native);
                 }),
             py::arg("stream") = OrbbecEgoCameraStream_ColorLeft, py::arg("sequence_number") = 0, py::arg("width") = 0,
             py::arg("height") = 0, py::arg("fps") = 0, py::arg("pixel_format") = OrbbecEgoPixelFormat_Mjpg,
             py::arg("encoded_bytes") = 0, py::arg("capture_epoch") = 0)
        .def_property_readonly("stream", field(&OrbbecEgoFrameMetadata::stream))
        .def_property_readonly("sequence_number", field(&OrbbecEgoFrameMetadata::sequence_number))
        .def_property_readonly("width", field(&OrbbecEgoFrameMetadata::width))
        .def_property_readonly("height", field(&OrbbecEgoFrameMetadata::height))
        .def_property_readonly("fps", field(&OrbbecEgoFrameMetadata::fps))
        .def_property_readonly("pixel_format", field(&OrbbecEgoFrameMetadata::pixel_format))
        .def_property_readonly("encoded_bytes", field(&OrbbecEgoFrameMetadata::encoded_bytes))
        .def_property_readonly("capture_epoch", field(&OrbbecEgoFrameMetadata::capture_epoch))
        .def("__repr__",
             [](const Serialized<OrbbecEgoFrameMetadata>& self)
             {
                 return "OrbbecEgoFrameMetadata(stream=" + std::string(EnumNameOrbbecEgoCameraStream(self->stream())) +
                        ", sequence_number=" + std::to_string(self->sequence_number()) + ")";
             });

    bind_record<OrbbecEgoFrameMetadataRecord, OrbbecEgoFrameMetadata>(
        m, "OrbbecEgoFrameMetadataRecord", "OrbbecEgoFrameMetadata");
}

} // namespace core
