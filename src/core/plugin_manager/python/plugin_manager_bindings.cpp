// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "plugin_manager/plugin_manager.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace core;

PYBIND11_MODULE(_plugin_manager, m)
{
    m.doc() = "Isaac Teleop Plugin Manager bindings";

    // Register custom exception
    py::register_exception<PluginCrashException>(m, "PluginCrashException");

    py::enum_<ProcessState>(m, "ProcessState")
        .value("RUNNING", ProcessState::RUNNING)
        .value("EXITED", ProcessState::EXITED)
        .value("SIGNALED", ProcessState::SIGNALED)
        .value("STOPPED", ProcessState::STOPPED)
        .value("ERROR", ProcessState::ERROR);

    py::enum_<ProcessReason>(m, "ProcessReason")
        .value("NONE", ProcessReason::NONE)
        .value("CLEAN_EXIT", ProcessReason::CLEAN_EXIT)
        .value("NONZERO_EXIT", ProcessReason::NONZERO_EXIT)
        .value("SIGNAL", ProcessReason::SIGNAL)
        .value("EXPLICIT_STOP", ProcessReason::EXPLICIT_STOP)
        .value("WAIT_ERROR", ProcessReason::WAIT_ERROR)
        .value("SIGNAL_ERROR", ProcessReason::SIGNAL_ERROR);

    py::class_<ProcessSnapshot>(m, "ProcessSnapshot")
        .def_readonly("state", &ProcessSnapshot::state)
        .def_readonly("reason", &ProcessSnapshot::reason)
        .def_readonly("pid", &ProcessSnapshot::pid)
        .def_readonly("exit_code", &ProcessSnapshot::exit_code)
        .def_readonly("term_signal", &ProcessSnapshot::term_signal)
        .def_readonly("error_code", &ProcessSnapshot::error_code)
        .def_readonly("error", &ProcessSnapshot::error);

    py::class_<DeviceInfo>(m, "DeviceInfo")
        .def_readonly("path", &DeviceInfo::path)
        .def_readonly("type", &DeviceInfo::type)
        .def_readonly("description", &DeviceInfo::description);

    py::class_<PluginInfo>(m, "PluginInfo")
        .def_readonly("name", &PluginInfo::name)
        .def_readonly("description", &PluginInfo::description)
        .def_readonly("command", &PluginInfo::command)
        .def_readonly("version", &PluginInfo::version)
        .def_readonly("working_dir", &PluginInfo::working_dir)
        .def_property_readonly("args",
                               [](const PluginInfo& info)
                               {
                                   py::tuple args(info.args.size());
                                   for (std::size_t index = 0; index < info.args.size(); ++index)
                                   {
                                       args[index] = info.args[index];
                                   }
                                   return args;
                               })
        .def_property_readonly("devices",
                               [](const PluginInfo& info)
                               {
                                   py::tuple devices(info.devices.size());
                                   for (std::size_t index = 0; index < info.devices.size(); ++index)
                                   {
                                       devices[index] = py::cast(info.devices[index]);
                                   }
                                   return devices;
                               });

    py::class_<Plugin, std::unique_ptr<Plugin>>(m, "Plugin")
        .def("stop", &Plugin::stop, "Explicitly stop the plugin (throws PluginCrashException if crashed)")
        .def("check_health", &Plugin::check_health, "Check if plugin has crashed (throws PluginCrashException if crashed)")
        .def("get_process_snapshot", &Plugin::get_process_snapshot,
             "Poll and return the cached process state without throwing process-health exceptions")
        .def("__enter__", [](Plugin* self) { return self; })
        .def("__exit__", [](Plugin* self, py::object, py::object, py::object) { self->stop(); });

    py::class_<PluginManager>(m, "PluginManager")
        .def(py::init<const std::vector<std::string>&>(), py::arg("search_paths"),
             "Create a PluginManager and discover plugins in the given search paths")
        .def("get_plugin_names", &PluginManager::get_plugin_names, "Get list of discovered plugin names")
        .def("get_plugin_info", &PluginManager::get_plugin_info, py::arg("plugin_name"),
             "Get the complete immutable descriptor for a discovered plugin")
        .def("query_devices", &PluginManager::query_devices, py::arg("plugin_name"),
             "Query available devices from a plugin")
        .def("start", &PluginManager::start, py::arg("plugin_name"), py::arg("plugin_root_id"),
             py::arg("plugin_args") = std::vector<std::string>{},
             "Start a plugin and return a RAII handle. plugin_args are appended after plugin.yaml args.");
}
