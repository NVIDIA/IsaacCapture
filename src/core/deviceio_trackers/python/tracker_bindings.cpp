// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "generated_tracker_binding_includes.inc"

#include <deviceio_trackers/controller_tracker.hpp>
#include <deviceio_trackers/full_body_tracker.hpp>
#include <deviceio_trackers/gamepad_tracker.hpp>
#include <deviceio_trackers/hand_tracker.hpp>
#include <deviceio_trackers/haptic_command_reader_tracker.hpp>
#include <deviceio_trackers/head_tracker.hpp>
#include <deviceio_trackers/message_channel_tracker.hpp>
#include <deviceio_trackers/tensor_push_tracker.hpp>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <schema/hand_generated.h>
#include <schema/message_channel_generated.h>

#include <array>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string_view>

namespace py = pybind11;

namespace
{

// Hand an encoded snapshot to Python, mapping "no payload" onto None.
//
// C++ spells absence as an empty handle, but exposing that to Python would make an
// inactive device answer field reads with defaults -- a disconnected pedal would read
// 0.0, indistinguishable from a pedal at rest. None makes the same mistake an
// AttributeError instead, and is what the caller already tests for.
template <typename T>
py::object to_python(const core::Serialized<T>& handle)
{
    return handle ? py::cast(handle) : py::none();
}

} // namespace

// Handing a snapshot to Python copies a shared_ptr to an immutable buffer, so there is no
// payload clone on the read path and no aliasing of live tracker storage: what a caller
// reads this frame keeps its values after the next session.update(), which publishes a new
// buffer rather than refilling this one.

PYBIND11_MODULE(_deviceio_trackers, m)
{
    // Load schema pybind converters (the encoded table views) before exposing tracker accessors.
    py::module_::import("isaaccapture.schema._schema");

    m.doc() = "Isaac Teleop DeviceIO - Tracker classes";

    py::class_<core::ITrackerSession>(m, "ITrackerSession");

    py::class_<core::ITracker, std::shared_ptr<core::ITracker>>(m, "ITracker").def("get_name", &core::ITracker::get_name);

    py::class_<core::HandTracker, core::ITracker, std::shared_ptr<core::HandTracker>>(m, "HandTracker")
        .def(py::init<>())
        .def(
            "get_left_hand",
            [](const core::HandTracker& self, const core::ITrackerSession& session)
            { return to_python(self.get_left_hand(session)); },
            py::arg("session"), "Get the left hand tracked state (None if inactive)")
        .def(
            "get_right_hand",
            [](const core::HandTracker& self, const core::ITrackerSession& session)
            { return to_python(self.get_right_hand(session)); },
            py::arg("session"), "Get the right hand tracked state (None if inactive)");

    py::class_<core::HeadTracker, core::ITracker, std::shared_ptr<core::HeadTracker>>(m, "HeadTracker")
        .def(py::init<>())
        .def(
            "get_head",
            [](const core::HeadTracker& self, const core::ITrackerSession& session)
            { return to_python(self.get_head(session)); },
            py::arg("session"), "Get the head tracked state (None if inactive)");

    m.def("discover_gamepad_device", &core::discover_gamepad_device,
          "The first Linux joystick-API gamepad (/dev/input/by-path/*-joystick, else the lowest "
          "/dev/input/jsN, e.g. a Bluetooth pad), or None when none is connected.");

    py::class_<core::GamepadTracker, core::ITracker, std::shared_ptr<core::GamepadTracker>>(
        m, "GamepadTracker",
        "In-process gamepad read from a Linux joystick-API device (/dev/input/jsN). Needs no plugin "
        "process, no OpenXR extension and no input group membership.")
        .def(py::init<std::string>(), py::arg("device_path") = "",
             "Read device_path (e.g. '/dev/input/js0'); empty picks the first connected joystick.")
        .def_property_readonly("device_path", &core::GamepadTracker::device_path)
        .def(
            "get_gamepad_data",
            [](const core::GamepadTracker& self, const core::ITrackerSession& session)
            { return to_python(self.get_data(session)); },
            py::arg("session"), "Get this frame's GamepadOutput (None while no gamepad is connected)");

    py::class_<core::ControllerTracker, core::ITracker, std::shared_ptr<core::ControllerTracker>>(m, "ControllerTracker")
        .def(py::init<>())
        .def(
            "get_left_controller",
            [](const core::ControllerTracker& self, const core::ITrackerSession& session)
            { return to_python(self.get_left_controller(session)); },
            py::arg("session"), "Get the left controller tracked state (None if inactive)")
        .def(
            "get_right_controller",
            [](const core::ControllerTracker& self, const core::ITrackerSession& session)
            { return to_python(self.get_right_controller(session)); },
            py::arg("session"), "Get the right controller tracked state (None if inactive)")
        .def(
            "apply_left_haptic_feedback",
            [](const core::ControllerTracker& self, const core::ITrackerSession& session, float amplitude,
               float frequency_hz, float duration_s)
            { self.apply_left_haptic_feedback(session, amplitude, frequency_hz, duration_s); },
            py::arg("session"), py::arg("amplitude"), py::arg("frequency_hz") = 0.0f, py::arg("duration_s") = 0.0f,
            "Apply one frame of haptic vibration to the left controller.\n"
            "amplitude: [0, 1]; 0 stops any active pulse instead of issuing a zero-amplitude pulse.\n"
            "frequency_hz: 0 selects the runtime's default frequency.\n"
            "duration_s: 0 selects the shortest pulse the runtime supports.")
        .def(
            "apply_right_haptic_feedback",
            [](const core::ControllerTracker& self, const core::ITrackerSession& session, float amplitude,
               float frequency_hz, float duration_s)
            { self.apply_right_haptic_feedback(session, amplitude, frequency_hz, duration_s); },
            py::arg("session"), py::arg("amplitude"), py::arg("frequency_hz") = 0.0f, py::arg("duration_s") = 0.0f,
            "Apply one frame of haptic vibration to the right controller. See apply_left_haptic_feedback.");

    py::enum_<core::MessageChannelStatus>(m, "MessageChannelStatus")
        .value("CONNECTING", core::MessageChannelStatus::CONNECTING)
        .value("CONNECTED", core::MessageChannelStatus::CONNECTED)
        .value("SHUTTING", core::MessageChannelStatus::SHUTTING)
        .value("DISCONNECTED", core::MessageChannelStatus::DISCONNECTED)
        .value("UNKNOWN", core::MessageChannelStatus::UNKNOWN);

    py::class_<core::MessageChannelTracker, core::ITracker, std::shared_ptr<core::MessageChannelTracker>>(
        m, "MessageChannelTracker")
        .def(py::init(
                 [](py::bytes channel_uuid, const std::string& channel_name, size_t max_message_size)
                 {
                     std::string uuid_str = channel_uuid;
                     if (uuid_str.size() != core::MessageChannelTracker::CHANNEL_UUID_SIZE)
                     {
                         throw std::invalid_argument("MessageChannelTracker: channel_uuid must be exactly 16 bytes");
                     }
                     std::array<uint8_t, core::MessageChannelTracker::CHANNEL_UUID_SIZE> uuid{};
                     std::memcpy(uuid.data(), uuid_str.data(), uuid.size());
                     return std::make_shared<core::MessageChannelTracker>(uuid, channel_name, max_message_size);
                 }),
             py::arg("channel_uuid"), py::arg("channel_name") = "",
             py::arg("max_message_size") = core::MessageChannelTracker::DEFAULT_MAX_MESSAGE_SIZE,
             "Construct a MessageChannelTracker for XR_NV_opaque_data_channel")
        .def(
            "get_messages",
            [](const core::MessageChannelTracker& self, const core::ITrackerSession& session)
            { return to_python(self.get_messages(session)); },
            py::arg("session"), "Get all messages drained during the last update (possibly empty)")
        .def(
            "get_status",
            [](const core::MessageChannelTracker& self, const core::ITrackerSession& session) -> core::MessageChannelStatus
            { return self.get_status(session); },
            py::arg("session"), "Get current channel connection state")
        .def(
            "send_message",
            [](const core::MessageChannelTracker& self, const core::ITrackerSession& session,
               const core::Serialized<core::MessageChannelMessages>& message)
            {
                const auto* payload = message ? message->payload() : nullptr;
                self.send_message(session, payload != nullptr ? std::vector<uint8_t>(payload->begin(), payload->end()) :
                                                                std::vector<uint8_t>{});
            },
            py::arg("session"), py::arg("message"), "Send a MessageChannelMessages payload over the message channel");

    py::class_<core::HapticCommandReaderTracker, core::ITracker, std::shared_ptr<core::HapticCommandReaderTracker>>(
        m, "HapticCommandReaderTracker")
        .def(py::init<const std::string&, std::size_t>(), py::arg("collection_id"),
             py::arg("max_payload_size") = core::HapticCommandReaderTracker::DEFAULT_MAX_PAYLOAD_SIZE)
        .def(
            "get_data",
            [](const core::HapticCommandReaderTracker& self, const core::ITrackerSession& session)
            { return to_python(self.get_data(session)); },
            py::arg("session"), "Get the latest haptic command (None when no data available)")
        .def(
            "get_data",
            [](const core::HapticCommandReaderTracker& self, const core::ITrackerSession& session,
               std::string_view endpoint) { return to_python(self.get_data(session, endpoint)); },
            py::arg("session"), py::arg("endpoint"),
            "Get the latest haptic command for one endpoint (None when no data available)");

    // py::class_ blocks for every manifest tracker; the accessor name comes from the
    // manifest's python_accessor key.
#include "generated_tracker_bindings.inc"

    py::class_<core::TensorPushTracker, core::ITracker, std::shared_ptr<core::TensorPushTracker>> tensor_push_tracker(
        m, "TensorPushTracker");
    tensor_push_tracker.attr("DEFAULT_MAX_PAYLOAD_SIZE") =
        static_cast<size_t>(core::TensorPushTracker::DEFAULT_MAX_PAYLOAD_SIZE);
    tensor_push_tracker
        .def(py::init<std::string, std::string, size_t>(), py::arg("collection_id"), py::arg("tensor_identifier"),
             py::arg("max_payload_size") = core::TensorPushTracker::DEFAULT_MAX_PAYLOAD_SIZE,
             "Generic producer ITracker: pushes opaque serialized payloads as tensor samples over "
             "XR_NVX1_push_tensor. Pairs with a consumer on the same collection_id + tensor_identifier.")
        .def(
            "push",
            [](const core::TensorPushTracker& self, const core::ITrackerSession& session, py::bytes payload)
            {
                const std::string bytes = payload;
                const std::vector<uint8_t> buffer(bytes.begin(), bytes.end());
                self.push(session, buffer);
            },
            py::arg("session"), py::arg("payload"),
            "Push one serialized payload (bytes, length <= max_payload_size) to the paired consumer.");

    py::class_<core::FullBodyTracker, core::ITracker, std::shared_ptr<core::FullBodyTracker>>(m, "FullBodyTracker")
        .def(py::init<>(),
             "Construct a vendor-agnostic full body tracker marker. The live session selects the "
             "vendor via VendorConfig (default: native PICO XR_BD_body_tracking); replay is vendor-neutral.")
        .def(
            "get_body_pose",
            [](const core::FullBodyTracker& self, const core::ITrackerSession& session)
            { return to_python(self.get_body_pose(session)); },
            py::arg("session"), "Get full body pose tracked state (None if inactive)");

    m.attr("NUM_JOINTS") = static_cast<int>(core::HandJoint_NUM_JOINTS);
    m.attr("JOINT_PALM") = static_cast<int>(core::HandJoint_PALM);
    m.attr("JOINT_WRIST") = static_cast<int>(core::HandJoint_WRIST);
    m.attr("JOINT_THUMB_TIP") = static_cast<int>(core::HandJoint_THUMB_TIP);
    m.attr("JOINT_INDEX_TIP") = static_cast<int>(core::HandJoint_INDEX_TIP);
}
