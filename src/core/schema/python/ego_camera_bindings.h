// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "schema_serialized.h"

#include <pybind11/stl.h>
#include <schema/ego_camera_generated.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace core
{

inline void bind_ego_camera(py::module& m)
{

    py::enum_<EgoCameraStream>(m, "EgoCameraStream")
        .value("ColorLeft", EgoCameraStream_ColorLeft)
        .value("ColorRight", EgoCameraStream_ColorRight);

    py::enum_<EgoPixelFormat>(m, "EgoPixelFormat")
        .value("Mjpg", EgoPixelFormat_Mjpg)
        .value("H264", EgoPixelFormat_H264)
        .value("H265", EgoPixelFormat_H265);

    // Struct vectors contain values; Serialized handles describe tables only.
    py::class_<EgoFrameMetadataEntry>(m, "EgoFrameMetadataEntry")
        .def(py::init<int32_t, int64_t>(), py::arg("key") = 0, py::arg("value") = 0)
        .def_property("key", &EgoFrameMetadataEntry::key, &EgoFrameMetadataEntry::mutate_key)
        .def_property("value", &EgoFrameMetadataEntry::value, &EgoFrameMetadataEntry::mutate_value);

    serialized_class<EgoFrameMetadata>(m, "EgoFrameMetadata", "Encoded EGO payload.")
        .def(py::init(
                 [](EgoCameraStream stream, uint64_t sequence_number, uint32_t width, uint32_t height, uint32_t fps,
                    EgoPixelFormat pixel_format, uint64_t encoded_bytes,
                    const std::vector<EgoFrameMetadataEntry>& sdk_metadata, uint32_t capture_epoch)
                 {
                     EgoFrameMetadataT native;
                     native.stream = stream;
                     native.sequence_number = sequence_number;
                     native.width = width;
                     native.height = height;
                     native.fps = fps;
                     native.pixel_format = pixel_format;
                     native.encoded_bytes = encoded_bytes;
                     native.sdk_metadata = sdk_metadata;
                     native.capture_epoch = capture_epoch;
                     return pack<EgoFrameMetadata>(native);
                 }),
             py::arg("stream") = EgoCameraStream_ColorLeft, py::arg("sequence_number") = 0, py::arg("width") = 0,
             py::arg("height") = 0, py::arg("fps") = 0, py::arg("pixel_format") = EgoPixelFormat_Mjpg,
             py::arg("encoded_bytes") = 0, py::arg("sdk_metadata") = std::vector<EgoFrameMetadataEntry>{},
             py::arg("capture_epoch") = 0)
        .def_property_readonly("stream", field(&EgoFrameMetadata::stream))
        .def_property_readonly("sequence_number", field(&EgoFrameMetadata::sequence_number))
        .def_property_readonly("width", field(&EgoFrameMetadata::width))
        .def_property_readonly("height", field(&EgoFrameMetadata::height))
        .def_property_readonly("fps", field(&EgoFrameMetadata::fps))
        .def_property_readonly("pixel_format", field(&EgoFrameMetadata::pixel_format))
        .def_property_readonly("encoded_bytes", field(&EgoFrameMetadata::encoded_bytes))
        .def_property_readonly("sdk_metadata",
                               [](const Serialized<EgoFrameMetadata>& self)
                               {
                                   std::vector<EgoFrameMetadataEntry> values;
                                   const auto* encoded = self->sdk_metadata();
                                   if (encoded != nullptr)
                                   {
                                       values.reserve(encoded->size());
                                       for (const auto* value : *encoded)
                                       {
                                           values.push_back(*value);
                                       }
                                   }
                                   return values;
                               })
        .def_property_readonly("capture_epoch", field(&EgoFrameMetadata::capture_epoch))
        .def("__repr__",
             [](const Serialized<EgoFrameMetadata>& self)
             {
                 return "EgoFrameMetadata(stream=" + std::string(EnumNameEgoCameraStream(self->stream())) +
                        ", sequence_number=" + std::to_string(self->sequence_number()) + ")";
             });

    bind_record<EgoFrameMetadataRecord, EgoFrameMetadata>(m, "EgoFrameMetadataRecord", "EgoFrameMetadata");

    serialized_class<EgoEncodedVideoFrame>(m, "EgoEncodedVideoFrame", "Encoded EGO payload.")
        .def(py::init(
                 [](EgoCameraStream stream, uint64_t sequence_number, uint32_t width, uint32_t height, uint32_t fps,
                    EgoPixelFormat pixel_format, const std::vector<uint8_t>& encoded_data, uint32_t capture_epoch)
                 {
                     EgoEncodedVideoFrameT native;
                     native.stream = stream;
                     native.sequence_number = sequence_number;
                     native.width = width;
                     native.height = height;
                     native.fps = fps;
                     native.pixel_format = pixel_format;
                     native.encoded_data = encoded_data;
                     native.capture_epoch = capture_epoch;
                     return pack<EgoEncodedVideoFrame>(native);
                 }),
             py::arg("stream") = EgoCameraStream_ColorLeft, py::arg("sequence_number") = 0, py::arg("width") = 0,
             py::arg("height") = 0, py::arg("fps") = 0, py::arg("pixel_format") = EgoPixelFormat_Mjpg,
             py::arg("encoded_data") = std::vector<uint8_t>{}, py::arg("capture_epoch") = 0)
        .def_property_readonly("stream", field(&EgoEncodedVideoFrame::stream))
        .def_property_readonly("sequence_number", field(&EgoEncodedVideoFrame::sequence_number))
        .def_property_readonly("width", field(&EgoEncodedVideoFrame::width))
        .def_property_readonly("height", field(&EgoEncodedVideoFrame::height))
        .def_property_readonly("fps", field(&EgoEncodedVideoFrame::fps))
        .def_property_readonly("pixel_format", field(&EgoEncodedVideoFrame::pixel_format))
        .def_property_readonly("encoded_data", vector_field(&EgoEncodedVideoFrame::encoded_data))
        .def_property_readonly("capture_epoch", field(&EgoEncodedVideoFrame::capture_epoch));

    bind_record<EgoEncodedVideoFrameRecord, EgoEncodedVideoFrame>(
        m, "EgoEncodedVideoFrameRecord", "EgoEncodedVideoFrame");
}

} // namespace core
