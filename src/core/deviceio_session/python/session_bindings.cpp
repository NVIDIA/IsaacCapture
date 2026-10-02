// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <deviceio_base/tracker_vendor.hpp>
#include <deviceio_py_utils/session.hpp>
#include <deviceio_session/deviceio_session.hpp>
#include <deviceio_session/replay_session.hpp>
#include <openxr/openxr.h>
#include <pybind11/stl.h>

#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

PYBIND11_MODULE(_deviceio_session, m)
{
    m.doc() = "Isaac Teleop DeviceIO - Session management";

    py::module_::import("isaaccapture.deviceio_trackers._deviceio_trackers");

    // ---- McapRecordingConfig (live recording) ----
    py::class_<core::McapRecordingConfig>(m, "McapRecordingConfig",
                                          "Configuration for MCAP recording. "
                                          "Pass to DeviceIOSession.run() to enable recording, "
                                          "or omit / pass None to disable.")
        .def(py::init(
                 [](const std::string& filename,
                    const std::vector<std::pair<std::shared_ptr<core::ITracker>, std::string>>& tracker_names)
                 {
                     core::McapRecordingConfig config;
                     config.filename = filename;
                     for (const auto& [tracker, name] : tracker_names)
                     {
                         config.tracker_names.emplace_back(tracker.get(), name);
                     }
                     return config;
                 }),
             py::arg("filename"),
             py::arg("tracker_names") = std::vector<std::pair<std::shared_ptr<core::ITracker>, std::string>>{})
        .def_readwrite("filename", &core::McapRecordingConfig::filename)
        .def(
            "get_tracker_names",
            [](const core::McapRecordingConfig& c)
            {
                py::list result;
                for (const auto& [tracker, name] : c.tracker_names)
                {
                    result.append(py::make_tuple(py::cast(tracker), name));
                }
                return result;
            },
            "Return the list of (tracker, channel_name) pairs.");

    // ---- McapReplayConfig (replay) ----
    py::class_<core::McapReplayConfig>(m, "McapReplayConfig",
                                       "Configuration for MCAP replay sessions. "
                                       "Pass to ReplaySession.run() to create a replay session.")
        .def(py::init(
                 [](const std::string& filename,
                    const std::vector<std::pair<std::shared_ptr<core::ITracker>, std::string>>& tracker_names)
                 {
                     core::McapReplayConfig config;
                     config.filename = filename;
                     for (const auto& [tracker, name] : tracker_names)
                     {
                         config.tracker_names.emplace_back(tracker.get(), name);
                     }
                     return config;
                 }),
             py::arg("filename"),
             py::arg("tracker_names") = std::vector<std::pair<std::shared_ptr<core::ITracker>, std::string>>{})
        .def_readwrite("filename", &core::McapReplayConfig::filename)
        .def(
            "get_tracker_names",
            [](const core::McapReplayConfig& c)
            {
                py::list result;
                for (const auto& [tracker, name] : c.tracker_names)
                {
                    result.append(py::make_tuple(py::cast(tracker), name));
                }
                return result;
            },
            "Return the list of (tracker, channel_name) pairs.");

    // ---- TrackerVendor / VendorConfig (live vendor selection) ----
    py::class_<core::TrackerVendor>(m, "TrackerVendor",
                                    "Per-tracker vendor selection: a string id (e.g. \"body.pico-xr\") plus "
                                    "free-form vendor params. Pass inside VendorConfig to DeviceIOSession.run().")
        .def(py::init(
                 [](std::string id, std::map<std::string, std::string> params)
                 {
                     core::TrackerVendor vendor;
                     vendor.id = std::move(id);
                     vendor.params = std::move(params);
                     return vendor;
                 }),
             py::arg("id"), py::arg("params") = std::map<std::string, std::string>{})
        .def_readwrite("id", &core::TrackerVendor::id)
        .def_readwrite("params", &core::TrackerVendor::params);

    py::class_<core::VendorConfig>(m, "VendorConfig",
                                   "Per-session vendor selection for vendored trackers (live sessions only). "
                                   "Trackers not listed use their default vendor id.")
        .def(py::init(
                 [](const std::vector<std::pair<std::shared_ptr<core::ITracker>, core::TrackerVendor>>& tracker_vendors)
                 {
                     core::VendorConfig config;
                     for (const auto& [tracker, vendor] : tracker_vendors)
                     {
                         config.tracker_vendors.emplace_back(tracker.get(), vendor);
                     }
                     return config;
                 }),
             py::arg("tracker_vendors") = std::vector<std::pair<std::shared_ptr<core::ITracker>, core::TrackerVendor>>{})
        .def(
            "get_tracker_vendors",
            [](const core::VendorConfig& c)
            {
                py::list result;
                for (const auto& [tracker, vendor] : c.tracker_vendors)
                {
                    result.append(py::make_tuple(py::cast(tracker), vendor));
                }
                return result;
            },
            "Return the list of (tracker, TrackerVendor) pairs.");

    // ---- DeviceIOSession (live) ----
    py::class_<core::PyDeviceIOSession, core::ITrackerSession, std::unique_ptr<core::PyDeviceIOSession>>(
        m, "DeviceIOSession")
        .def("update", &core::PyDeviceIOSession::update, "Update session and all trackers")
        .def("close", &core::PyDeviceIOSession::close,
             "Release the native session immediately (usually automatic via context manager)")
        .def("__enter__", &core::PyDeviceIOSession::enter)
        .def("__exit__", &core::PyDeviceIOSession::exit)
        .def_static("get_required_extensions", &core::DeviceIOSession::get_required_extensions, py::arg("trackers"),
                    py::arg("vendor_config") = core::VendorConfig{},
                    "Aggregate OpenXR extensions required for a live session with these tracker types "
                    "(not a per-tracker instance method). Pass a VendorConfig to resolve vendored trackers.")
        .def_static("requires_openxr", &core::DeviceIOSession::requires_openxr, py::arg("trackers"),
                    py::arg("vendor_config") = core::VendorConfig{},
                    "Whether any of these trackers needs an OpenXR session. When False, run() accepts "
                    "handles=None and no OpenXR runtime is needed.")
        .def_static(
            "run",
            [](const std::vector<std::shared_ptr<core::ITracker>>& trackers,
               std::optional<core::OpenXRSessionHandles> handles,
               std::optional<core::McapRecordingConfig> recording_config, core::VendorConfig vendor_config)
            {
                const core::OpenXRSessionHandles resolved = handles.value_or(core::OpenXRSessionHandles{});
                const bool has_handles = resolved.instance != XR_NULL_HANDLE && resolved.session != XR_NULL_HANDLE &&
                                         resolved.space != XR_NULL_HANDLE && resolved.xrGetInstanceProcAddr != nullptr;
                if (!has_handles && core::DeviceIOSession::requires_openxr(trackers, vendor_config))
                {
                    throw std::runtime_error(
                        "DeviceIOSession.run: invalid OpenXRSessionHandles (instance, session, space must be non-null "
                        "handles and xrGetInstanceProcAddr must be set); a tracker in this session needs OpenXR");
                }
                auto session = core::DeviceIOSession::run(
                    trackers, resolved, std::move(recording_config), std::move(vendor_config));
                return std::make_unique<core::PyDeviceIOSession>(std::move(session));
            },
            py::arg("trackers"), py::arg("handles") = py::none(), py::arg("recording_config") = py::none(),
            py::arg("vendor_config") = core::VendorConfig{},
            "Create and initialize a session with trackers. "
            "Pass a McapRecordingConfig to enable MCAP recording, and a VendorConfig to select "
            "vendors for any vendored trackers. handles may be None only when requires_openxr() is False.");

    // ---- ReplaySession ----
    py::class_<core::PyReplaySession, core::ITrackerSession, std::unique_ptr<core::PyReplaySession>>(m, "ReplaySession")
        .def("update", &core::PyReplaySession::update, "Advance replay by one frame")
        .def("close", &core::PyReplaySession::close,
             "Release the native session immediately (usually automatic via context manager)")
        .def("__enter__", &core::PyReplaySession::enter)
        .def("__exit__", &core::PyReplaySession::exit)
        .def_static(
            "run",
            [](const core::McapReplayConfig& config)
            {
                auto session = core::ReplaySession::run(config);
                return std::make_unique<core::PyReplaySession>(std::move(session));
            },
            py::arg("config"),
            "Create a replay session that reads recorded data from an MCAP file. "
            "The McapReplayConfig.tracker_names maps tracker objects to their MCAP channel base names.");
}
