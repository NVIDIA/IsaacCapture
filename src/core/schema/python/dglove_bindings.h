// SPDX-FileCopyrightText: Copyright (c) 2026 mwang259. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "schema_serialized.h"

#include <schema/dglove_generated.h>

namespace core
{
inline void bind_dglove(py::module& m)
{
    serialized_class<DGloveFrame>(m, "DGloveFrame", "DGlove network fields and receiver-local arrival time.")
        .def_property_readonly("schema_version", field(&DGloveFrame::schema_version))
        .def_property_readonly("glove_id", string_field(&DGloveFrame::glove_id))
        .def_property_readonly("hand", field(&DGloveFrame::hand))
        .def_property_readonly("frame_number", field(&DGloveFrame::frame_number))
        .def_property_readonly("device_sequence", field(&DGloveFrame::device_sequence))
        .def_property_readonly("received_unix_ns", field(&DGloveFrame::received_unix_ns))
        .def_property_readonly("monotonic_ns", field(&DGloveFrame::monotonic_ns))
        .def_property_readonly("mano_pose45", vector_field(&DGloveFrame::mano_pose45))
        .def_property_readonly("joints21_xyz_m", vector_field(&DGloveFrame::joints21_xyz_m))
        .def_property_readonly("palm_adc", vector_field(&DGloveFrame::palm_adc))
        .def_property_readonly("mirror_x", field(&DGloveFrame::mirror_x))
        .def_property_readonly("simulation", field(&DGloveFrame::simulation))
        .def_property_readonly("pose_valid", field(&DGloveFrame::pose_valid))
        .def_property_readonly("identity_source", string_field(&DGloveFrame::identity_source))
        .def_property_readonly("status", field(&DGloveFrame::status))
        .def_property_readonly("output_sequence", field(&DGloveFrame::output_sequence))
        .def_property_readonly("session_id", string_field(&DGloveFrame::session_id))
        .def_property_readonly("response_version", string_field(&DGloveFrame::response_version))
        .def_property_readonly("pose_source", string_field(&DGloveFrame::pose_source))
        .def_property_readonly("source_timestamp", field(&DGloveFrame::source_timestamp))
        .def_property_readonly("replay_loop", field(&DGloveFrame::replay_loop))
        .def_property_readonly("recording_wear", field(&DGloveFrame::recording_wear))
        .def_property_readonly("pose_id", string_field(&DGloveFrame::pose_id))
        .def_property_readonly("device_sequence_valid", field(&DGloveFrame::device_sequence_valid))
        .def_property_readonly("received_monotonic_ns", field(&DGloveFrame::received_monotonic_ns))
        .def_property_readonly("available", field(&DGloveFrame::available));
    bind_record<DGloveFrameRecord, DGloveFrame>(m, "DGloveFrameRecord", "DGloveFrame");
}
} // namespace core
