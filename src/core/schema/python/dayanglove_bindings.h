// SPDX-FileCopyrightText: Copyright (c) 2026 mwang259. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "schema_serialized.h"

#include <schema/dayanglove_generated.h>

namespace core
{
inline void bind_dayanglove(py::module& m)
{
    serialized_class<DAYANGloveFrame>(m, "DAYANGloveFrame", "DAYANGlove network fields and receiver-local arrival time.")
        .def_property_readonly("schema_version", field(&DAYANGloveFrame::schema_version))
        .def_property_readonly("glove_id", string_field(&DAYANGloveFrame::glove_id))
        .def_property_readonly("hand", field(&DAYANGloveFrame::hand))
        .def_property_readonly("frame_number", field(&DAYANGloveFrame::frame_number))
        .def_property_readonly("device_sequence", field(&DAYANGloveFrame::device_sequence))
        .def_property_readonly("received_unix_ns", field(&DAYANGloveFrame::received_unix_ns))
        .def_property_readonly("monotonic_ns", field(&DAYANGloveFrame::monotonic_ns))
        .def_property_readonly("mano_pose45", vector_field(&DAYANGloveFrame::mano_pose45))
        .def_property_readonly("joints21_xyz_m", vector_field(&DAYANGloveFrame::joints21_xyz_m))
        .def_property_readonly("palm_adc", vector_field(&DAYANGloveFrame::palm_adc))
        .def_property_readonly("mirror_x", field(&DAYANGloveFrame::mirror_x))
        .def_property_readonly("simulation", field(&DAYANGloveFrame::simulation))
        .def_property_readonly("pose_valid", field(&DAYANGloveFrame::pose_valid))
        .def_property_readonly("identity_source", string_field(&DAYANGloveFrame::identity_source))
        .def_property_readonly("status", field(&DAYANGloveFrame::status))
        .def_property_readonly("output_sequence", field(&DAYANGloveFrame::output_sequence))
        .def_property_readonly("session_id", string_field(&DAYANGloveFrame::session_id))
        .def_property_readonly("response_version", string_field(&DAYANGloveFrame::response_version))
        .def_property_readonly("pose_source", string_field(&DAYANGloveFrame::pose_source))
        .def_property_readonly("source_timestamp", field(&DAYANGloveFrame::source_timestamp))
        .def_property_readonly("replay_loop", field(&DAYANGloveFrame::replay_loop))
        .def_property_readonly("recording_wear", field(&DAYANGloveFrame::recording_wear))
        .def_property_readonly("pose_id", string_field(&DAYANGloveFrame::pose_id))
        .def_property_readonly("device_sequence_valid", field(&DAYANGloveFrame::device_sequence_valid))
        .def_property_readonly("received_monotonic_ns", field(&DAYANGloveFrame::received_monotonic_ns))
        .def_property_readonly("available", field(&DAYANGloveFrame::available));
    bind_record<DAYANGloveFrameRecord, DAYANGloveFrame>(m, "DAYANGloveFrameRecord", "DAYANGloveFrame");
}
} // namespace core
