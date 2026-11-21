// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <oxr_utils/os_time.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <robot_state_store/robot_state_store.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace
{

py::bytes to_bytes(const std::vector<uint8_t>& bytes)
{
    return py::bytes(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

//! The bytes of `data`, valid while it lives.
std::span<const uint8_t> as_span(const py::bytes& data)
{
    const std::string_view view = data;
    return { reinterpret_cast<const uint8_t*>(view.data()), view.size() };
}

int64_t now_or(const std::optional<int64_t>& time_ns)
{
    return time_ns ? *time_ns : core::os_monotonic_now_ns();
}

} // namespace

PYBIND11_MODULE(_robot_state, m)
{
    m.doc() = "Isaac Capture robot state - the store of a robot's data series on the server machine";

    m.attr("DESCRIPTION_TOPIC") = py::str(std::string(core::kRobotDescriptionTopic));

    m.def(
        "monotonic_now_ns", []() { return core::os_monotonic_now_ns(); },
        "This machine's monotonic clock in nanoseconds: the clock of receive times and of ages.");

    py::class_<core::RobotSeriesInfo>(m, "RobotSeriesInfo", "One data series as its robot declared it.")
        .def_readonly("name", &core::RobotSeriesInfo::name, "The topic its records travel on.")
        .def_readonly(
            "shape", &core::RobotSeriesInfo::shape, "'description', 'joints', 'status' or the robot's own shape name.")
        .def_readonly("schema_name", &core::RobotSeriesInfo::schema_name)
        .def_readonly("schema_encoding", &core::RobotSeriesInfo::schema_encoding)
        .def_property_readonly(
            "schema_data", [](const core::RobotSeriesInfo& self) { return to_bytes(self.schema_data); })
        .def_readonly("message_encoding", &core::RobotSeriesInfo::message_encoding)
        .def_readonly("stale_after_ns", &core::RobotSeriesInfo::stale_after_ns,
                      "A record older than this is not fresh; 0 means it never goes stale.")
        .def_readonly("in_order", &core::RobotSeriesInfo::in_order, "True when every record is delivered.")
        .def_readonly("rate_hz", &core::RobotSeriesInfo::rate_hz, "Nominal records per second; 0 when not stated.")
        .def("__repr__", [](const core::RobotSeriesInfo& self)
             { return "RobotSeriesInfo(name='" + self.name + "', shape='" + self.shape + "')"; });

    py::class_<core::RobotRecord>(m, "RobotRecord", "The newest record of a series and how old it is.")
        .def_property_readonly(
            "data",
            [](const core::RobotRecord& self) -> std::optional<py::bytes>
            {
                if (!self.data)
                {
                    return std::nullopt;
                }
                return to_bytes(*self.data);
            },
            "The record's bytes; None when the series has no record.")
        .def_readonly("sequence", &core::RobotRecord::sequence, "Records taken on this series in this robot session.")
        .def_readonly("sample_time_ns", &core::RobotRecord::sample_time_ns, "Time of the reading, robot clock.")
        .def_readonly("publish_time_ns", &core::RobotRecord::publish_time_ns,
                      "Time the robot app handed the reading over, robot clock.")
        .def_readonly("receive_time_ns", &core::RobotRecord::receive_time_ns, "Arrival time, this machine's clock.")
        .def_readonly("age_ns", &core::RobotRecord::age_ns,
                      "Time since the reading, on this machine's clock, at least receive_age_ns; None without a clock "
                      "estimate.")
        .def_readonly("receive_age_ns", &core::RobotRecord::receive_age_ns, "Time since the record arrived.")
        .def_readonly("clock_two_way", &core::RobotRecord::clock_two_way,
                      "False while the clock estimate is one-way: age_ns is then too large, never too small.")
        .def_readonly("fresh", &core::RobotRecord::fresh,
                      "True for a record within its series' stale_after_ns; False without a record.")
        .def("__repr__",
             [](const core::RobotRecord& self)
             {
                 return "RobotRecord(sequence=" + std::to_string(self.sequence) +
                        ", bytes=" + (self.data ? std::to_string(self.data->size()) : std::string("None")) +
                        ", fresh=" + (self.fresh ? "True" : "False") + ")";
             });

    py::class_<core::RobotStateStore, std::shared_ptr<core::RobotStateStore>>(
        m, "RobotStateStore",
        "Each data series a robot declared and its newest record. Thread-safe.\n\n"
        "The Teleop Session Server fills one from the robot-state link messages; a reader in another program fills "
        "its own from snapshots.")
        .def(py::init<>())
        .def("begin_session", &core::RobotStateStore::begin_session,
             "A robot app that sends robot state connected: drops every series and record.")
        .def("end_session", &core::RobotStateStore::end_session, "It left; series and records stay.")
        .def(
            "add_setup",
            [](core::RobotStateStore& self, const py::bytes& payload) { self.add_setup(as_span(payload)); },
            py::arg("payload"),
            "Take the payload of a ROBOT_STATE_SETUP link message. RuntimeError, changing nothing, when malformed.")
        .def(
            "add_samples",
            [](core::RobotStateStore& self, const py::bytes& payload, const std::optional<int64_t>& receive_time_ns)
            { self.add_samples(as_span(payload), now_or(receive_time_ns)); },
            py::arg("payload"), py::arg("receive_time_ns") = py::none(),
            "Take the payload of a ROBOT_STATE_SAMPLES or ROBOT_STATE_RECORDS link message, prefix included. "
            "receive_time_ns None is the time of the call. RuntimeError, changing nothing, when malformed.")
        .def(
            "declare_series",
            [](core::RobotStateStore& self, std::string name, std::string shape, std::string schema_name,
               std::string schema_encoding, const py::bytes& schema_data, std::string message_encoding,
               int64_t stale_after_ns, bool in_order, double rate_hz)
            {
                const auto bytes = as_span(schema_data);
                self.declare_series(
                    core::RobotSeriesInfo{ std::move(name), std::move(shape), std::move(schema_name),
                                           std::move(schema_encoding), std::vector<uint8_t>(bytes.begin(), bytes.end()),
                                           std::move(message_encoding), stale_after_ns, in_order, rate_hz });
            },
            py::arg("name"), py::arg("shape"), py::arg("schema_name"), py::arg("schema_encoding"),
            py::arg("schema_data"), py::arg("message_encoding"), py::arg("stale_after_ns") = 0,
            py::arg("in_order") = false, py::arg("rate_hz") = 0.0,
            "Declare or replace a series without a link message. ValueError for an empty name.")
        .def(
            "put_record",
            [](core::RobotStateStore& self, const std::string& name, const py::bytes& data, int64_t sample_time_ns,
               const std::optional<int64_t>& publish_time_ns, const std::optional<int64_t>& receive_time_ns,
               const std::optional<int64_t>& robot_minus_server_ns, bool clock_two_way)
            {
                core::RobotClockSample clock;
                if (robot_minus_server_ns)
                {
                    clock.robot_minus_server_ns = *robot_minus_server_ns;
                    clock.valid = true;
                    clock.two_way = clock_two_way;
                }
                self.put_record(name, as_span(data), sample_time_ns,
                                publish_time_ns ? *publish_time_ns : sample_time_ns, now_or(receive_time_ns), clock);
            },
            py::arg("name"), py::arg("data"), py::arg("sample_time_ns"), py::kw_only(),
            py::arg("publish_time_ns") = py::none(), py::arg("receive_time_ns") = py::none(),
            py::arg("robot_minus_server_ns") = py::none(), py::arg("clock_two_way") = false,
            "Put one record without a link message. publish_time_ns None is sample_time_ns, receive_time_ns None is "
            "the time of the call, robot_minus_server_ns None is no clock estimate. ValueError for an undeclared "
            "series.")
        .def_property_readonly("connected", &core::RobotStateStore::connected,
                               "True while a robot app that sends robot state is connected.")
        .def_property_readonly("session_count", &core::RobotStateStore::session_count, "begin_session() calls.")
        .def_property_readonly("generation", &core::RobotStateStore::generation, "Rises with every change.")
        .def_property_readonly("setup_generation", &core::RobotStateStore::setup_generation,
                               "Rises with begin_session() and with each change of a series' declaration.")
        .def_property_readonly("dropped_records", &core::RobotStateStore::dropped_records,
                               "Records not taken: undeclared channel, or from an older payload.")
        .def("series", &core::RobotStateStore::series, "The declared series, in declaration order.")
        .def(
            "series_info",
            [](const core::RobotStateStore& self, const std::string& name) { return self.series_info(name); },
            py::arg("name"), "The declaration of one series, or None.")
        .def(
            "latest",
            [](const core::RobotStateStore& self, const std::string& name, const std::optional<int64_t>& now_ns)
            { return self.latest(name, now_or(now_ns)); },
            py::arg("name"), py::arg("now_ns") = py::none(),
            "The newest record of a series with its ages at now_ns, on this machine's monotonic clock (None: the time "
            "of the call).")
        .def(
            "export_snapshot",
            [](const core::RobotStateStore& self, uint64_t known_setup_generation)
            { return to_bytes(self.export_snapshot(known_setup_generation)); },
            py::arg("known_setup_generation") = 0,
            "The store's content in a private format. The declarations are left out when known_setup_generation "
            "equals setup_generation.")
        .def(
            "import_snapshot",
            [](core::RobotStateStore& self, const py::bytes& snapshot) { self.import_snapshot(as_span(snapshot)); },
            py::arg("snapshot"),
            "Replace this store's content with a snapshot's. RuntimeError, changing nothing, on bytes "
            "export_snapshot() of this build did not make.");
}
