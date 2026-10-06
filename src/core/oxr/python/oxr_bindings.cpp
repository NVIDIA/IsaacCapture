// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <openxr/openxr.h>
#include <oxr/oxr_session.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace core
{

/**
 * @brief Python-facing session wrapper: destroys the underlying OpenXRSession in __exit__.
 *
 * Binding OpenXRSession directly with a no-op __exit__ leaves destruction to the pybind
 * holder, which can run at garbage collection -- after the CloudXR runtime/IPC socket has
 * been torn down -- producing "Broken pipe"/invalid-handle errors. Holding the session in a
 * unique_ptr and resetting it in close()/exit() tears it down deterministically at
 * context-manager exit, while the runtime is still alive. Mirrors PyDeviceIOSession.
 */
class PyOpenXRSession
{
public:
    PyOpenXRSession(const std::string& app_name, const std::vector<std::string>& extensions)
        : impl_(std::make_unique<OpenXRSession>(app_name, extensions))
    {
    }

    OpenXRSessionHandles get_handles() const
    {
        if (!impl_)
        {
            throw std::runtime_error("OpenXRSession has been closed/destroyed");
        }
        return impl_->get_handles();
    }

    OpenXRProviderSnapshot get_provider_snapshot()
    {
        if (!impl_)
        {
            throw std::runtime_error("OpenXRSession has been closed/destroyed");
        }
        return impl_->get_provider_snapshot();
    }

    void close()
    {
        impl_.reset();
    }

    PyOpenXRSession& enter()
    {
        if (!impl_)
        {
            throw std::runtime_error("OpenXRSession has been closed/destroyed");
        }
        return *this;
    }

    void exit(py::object, py::object, py::object)
    {
        close();
    }

private:
    std::unique_ptr<OpenXRSession> impl_;
};

} // namespace core

PYBIND11_MODULE(_oxr, m)
{
    m.doc() = "Isaac Teleop OXR - OpenXR Session Module";

    py::enum_<core::OpenXRProviderState>(m, "OpenXRProviderState")
        .value("AVAILABLE", core::OpenXRProviderState::AVAILABLE)
        .value("FAILED", core::OpenXRProviderState::FAILED);

    py::enum_<core::OpenXRHeadsetState>(m, "OpenXRHeadsetState")
        .value("CONNECTED", core::OpenXRHeadsetState::CONNECTED)
        .value("DISCONNECTED", core::OpenXRHeadsetState::DISCONNECTED);

    py::enum_<core::OpenXRProviderReason>(m, "OpenXRProviderReason")
        .value("NONE", core::OpenXRProviderReason::NONE)
        .value("TRACKING_UNAVAILABLE", core::OpenXRProviderReason::TRACKING_UNAVAILABLE)
        .value("SESSION_LOST", core::OpenXRProviderReason::SESSION_LOST)
        .value("INSTANCE_LOST", core::OpenXRProviderReason::INSTANCE_LOST)
        .value("POLL_ERROR", core::OpenXRProviderReason::POLL_ERROR);

    py::class_<core::OpenXRProviderSnapshot>(m, "OpenXRProviderSnapshot")
        .def_readonly("state", &core::OpenXRProviderSnapshot::state)
        .def_readonly("headset_state", &core::OpenXRProviderSnapshot::headset_state)
        .def_readonly("reason", &core::OpenXRProviderSnapshot::reason)
        .def_readonly("result_code", &core::OpenXRProviderSnapshot::result_code)
        .def_readonly("error", &core::OpenXRProviderSnapshot::error);

    // OpenXRSessionHandles structure (for sharing)
    py::class_<core::OpenXRSessionHandles>(m, "OpenXRSessionHandles")
        .def(py::init<>())
        // Constructor from raw handle values (enables non-Kit usage and external runtime integration)
        .def(py::init(
                 [](uint64_t instance, uint64_t session, uint64_t space, uint64_t xr_get_instance_proc_addr)
                 {
                     return core::OpenXRSessionHandles(
                         reinterpret_cast<XrInstance>(instance), reinterpret_cast<XrSession>(session),
                         reinterpret_cast<XrSpace>(space),
                         reinterpret_cast<PFN_xrGetInstanceProcAddr>(xr_get_instance_proc_addr));
                 }),
             py::arg("instance"), py::arg("session"), py::arg("space"), py::arg("xr_get_instance_proc_addr"),
             "Create OpenXRSessionHandles from raw handle values (as integers)")
        .def_property_readonly(
            "instance", [](const core::OpenXRSessionHandles& self) { return reinterpret_cast<size_t>(self.instance); },
            "Get OpenXR instance handle as integer")
        .def_property_readonly(
            "session", [](const core::OpenXRSessionHandles& self) { return reinterpret_cast<size_t>(self.session); },
            "Get OpenXR session handle as integer")
        .def_property_readonly(
            "space", [](const core::OpenXRSessionHandles& self) { return reinterpret_cast<size_t>(self.space); },
            "Get OpenXR space handle as integer")
        .def_property_readonly(
            "proc_addr",
            [](const core::OpenXRSessionHandles& self) { return reinterpret_cast<size_t>(self.xrGetInstanceProcAddr); },
            "Get xrGetInstanceProcAddr function pointer as integer");

    // OpenXRSession (Python-facing wrapper; see core::PyOpenXRSession for the lifetime contract)
    py::class_<core::PyOpenXRSession, std::unique_ptr<core::PyOpenXRSession>>(m, "OpenXRSession")
        .def(py::init<const std::string&, const std::vector<std::string>&>(), py::arg("app_name"),
             py::arg("extensions") = std::vector<std::string>())
        .def("get_handles", &core::PyOpenXRSession::get_handles, "Get session handles for sharing")
        .def("get_provider_snapshot", &core::PyOpenXRSession::get_provider_snapshot,
             "Poll and return the cached owned-provider health snapshot")
        .def("close", &core::PyOpenXRSession::close,
             "Release the native session immediately (usually automatic via context manager)")
        .def("__enter__", &core::PyOpenXRSession::enter)
        .def("__exit__", &core::PyOpenXRSession::exit);
}
