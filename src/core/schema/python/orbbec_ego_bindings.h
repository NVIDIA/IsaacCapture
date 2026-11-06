// SPDX-FileCopyrightText: Copyright (c) 2026 ORBBEC INC. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "schema_serialized.h"

#include <pybind11/stl.h>
#include <schema/orbbec_ego_calibration_generated.h>
#include <schema/orbbec_ego_device_state_utils.hpp>
#include <schema/orbbec_ego_imu_generated.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace core
{

inline void bind_orbbec_ego(py::module& m)
{

    py::enum_<OrbbecEgoImuSensor>(m, "OrbbecEgoImuSensor")
        .value("Accel", OrbbecEgoImuSensor_Accel)
        .value("Gyro", OrbbecEgoImuSensor_Gyro);

    py::class_<OrbbecEgoImuSample>(m, "OrbbecEgoImuSample")
        .def(py::init<double, double, double, int64_t, int64_t>(), py::arg("x_si") = 0.0, py::arg("y_si") = 0.0,
             py::arg("z_si") = 0.0, py::arg("sample_time_local_common_clock_ns") = 0,
             py::arg("sample_time_raw_device_clock_ns") = 0)
        .def_property("x_si", &OrbbecEgoImuSample::x_si, &OrbbecEgoImuSample::mutate_x_si)
        .def_property("y_si", &OrbbecEgoImuSample::y_si, &OrbbecEgoImuSample::mutate_y_si)
        .def_property("z_si", &OrbbecEgoImuSample::z_si, &OrbbecEgoImuSample::mutate_z_si)
        .def_property("sample_time_local_common_clock_ns", &OrbbecEgoImuSample::sample_time_local_common_clock_ns,
                      &OrbbecEgoImuSample::mutate_sample_time_local_common_clock_ns)
        .def_property("sample_time_raw_device_clock_ns", &OrbbecEgoImuSample::sample_time_raw_device_clock_ns,
                      &OrbbecEgoImuSample::mutate_sample_time_raw_device_clock_ns);

    serialized_class<OrbbecEgoImuBatch>(m, "OrbbecEgoImuBatch", "Encoded EGO payload.")
        .def(py::init(
                 [](OrbbecEgoImuSensor sensor, uint64_t sequence_number, uint32_t sample_rate_hz, float full_scale,
                    const std::vector<OrbbecEgoImuSample>& samples, uint32_t capture_epoch, float temperature_c)
                 {
                     OrbbecEgoImuBatchT native;
                     native.sensor = sensor;
                     native.sequence_number = sequence_number;
                     native.sample_rate_hz = sample_rate_hz;
                     native.full_scale = full_scale;
                     native.samples = samples;
                     native.capture_epoch = capture_epoch;
                     native.temperature_c = temperature_c;
                     return pack<OrbbecEgoImuBatch>(native);
                 }),
             py::arg("sensor") = OrbbecEgoImuSensor_Accel, py::arg("sequence_number") = 0, py::arg("sample_rate_hz") = 0,
             py::arg("full_scale") = 0.0f, py::arg("samples") = std::vector<OrbbecEgoImuSample>{},
             py::arg("capture_epoch") = 0, py::arg("temperature_c") = 0.0f)
        .def_property_readonly("sensor", field(&OrbbecEgoImuBatch::sensor))
        .def_property_readonly("sequence_number", field(&OrbbecEgoImuBatch::sequence_number))
        .def_property_readonly("sample_rate_hz", field(&OrbbecEgoImuBatch::sample_rate_hz))
        .def_property_readonly("full_scale", field(&OrbbecEgoImuBatch::full_scale))
        .def_property_readonly("samples",
                               [](const Serialized<OrbbecEgoImuBatch>& self)
                               {
                                   std::vector<OrbbecEgoImuSample> values;
                                   const auto* encoded = self->samples();
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
        .def_property_readonly("capture_epoch", field(&OrbbecEgoImuBatch::capture_epoch))
        .def_property_readonly("temperature_c", field(&OrbbecEgoImuBatch::temperature_c));

    bind_record<OrbbecEgoImuBatchRecord, OrbbecEgoImuBatch>(m, "OrbbecEgoImuBatchRecord", "OrbbecEgoImuBatch");

    serialized_class<OrbbecEgoCameraIntrinsics>(m, "OrbbecEgoCameraIntrinsics", "Encoded EGO payload.")
        .def(py::init(
                 [](uint32_t width, uint32_t height, float fx, float fy, float cx, float cy, int32_t distortion_model,
                    const std::vector<float>& distortion)
                 {
                     OrbbecEgoCameraIntrinsicsT native;
                     native.width = width;
                     native.height = height;
                     native.fx = fx;
                     native.fy = fy;
                     native.cx = cx;
                     native.cy = cy;
                     native.distortion_model = distortion_model;
                     native.distortion = distortion;
                     return pack<OrbbecEgoCameraIntrinsics>(native);
                 }),
             py::arg("width") = 0, py::arg("height") = 0, py::arg("fx") = 0.0f, py::arg("fy") = 0.0f,
             py::arg("cx") = 0.0f, py::arg("cy") = 0.0f, py::arg("distortion_model") = 0,
             py::arg("distortion") = std::vector<float>{})
        .def_property_readonly("width", field(&OrbbecEgoCameraIntrinsics::width))
        .def_property_readonly("height", field(&OrbbecEgoCameraIntrinsics::height))
        .def_property_readonly("fx", field(&OrbbecEgoCameraIntrinsics::fx))
        .def_property_readonly("fy", field(&OrbbecEgoCameraIntrinsics::fy))
        .def_property_readonly("cx", field(&OrbbecEgoCameraIntrinsics::cx))
        .def_property_readonly("cy", field(&OrbbecEgoCameraIntrinsics::cy))
        .def_property_readonly("distortion_model", field(&OrbbecEgoCameraIntrinsics::distortion_model))
        .def_property_readonly("distortion", vector_field(&OrbbecEgoCameraIntrinsics::distortion));

    serialized_class<OrbbecEgoExtrinsics>(m, "OrbbecEgoExtrinsics", "Encoded EGO payload.")
        .def(py::init(
                 [](const std::vector<float>& rotation, const std::vector<float>& translation_mm)
                 {
                     OrbbecEgoExtrinsicsT native;
                     native.rotation = rotation;
                     native.translation_mm = translation_mm;
                     return pack<OrbbecEgoExtrinsics>(native);
                 }),
             py::arg("rotation") = std::vector<float>{}, py::arg("translation_mm") = std::vector<float>{})
        .def_property_readonly("rotation", vector_field(&OrbbecEgoExtrinsics::rotation))
        .def_property_readonly("translation_mm", vector_field(&OrbbecEgoExtrinsics::translation_mm));

    serialized_class<OrbbecEgoCalibration>(m, "OrbbecEgoCalibration", "Encoded EGO payload.")
        .def(py::init(
                 [](const std::string& device_uid, const Serialized<OrbbecEgoCameraIntrinsics>* color_left,
                    const Serialized<OrbbecEgoCameraIntrinsics>* color_right,
                    const Serialized<OrbbecEgoExtrinsics>* left_to_right,
                    const Serialized<OrbbecEgoExtrinsics>* accel_to_left,
                    const Serialized<OrbbecEgoExtrinsics>* gyro_to_left, const std::string& raw_alignment_yaml,
                    const std::string& raw_imu_yaml, uint32_t capture_epoch)
                 {
                     OrbbecEgoCalibrationT native;
                     native.device_uid = device_uid;
                     if (color_left != nullptr && *color_left)
                     {
                         native.color_left = std::make_shared<OrbbecEgoCameraIntrinsicsT>();
                         (*color_left)->UnPackTo(native.color_left.get());
                     }
                     if (color_right != nullptr && *color_right)
                     {
                         native.color_right = std::make_shared<OrbbecEgoCameraIntrinsicsT>();
                         (*color_right)->UnPackTo(native.color_right.get());
                     }
                     if (left_to_right != nullptr && *left_to_right)
                     {
                         native.left_to_right = std::make_shared<OrbbecEgoExtrinsicsT>();
                         (*left_to_right)->UnPackTo(native.left_to_right.get());
                     }
                     if (accel_to_left != nullptr && *accel_to_left)
                     {
                         native.accel_to_left = std::make_shared<OrbbecEgoExtrinsicsT>();
                         (*accel_to_left)->UnPackTo(native.accel_to_left.get());
                     }
                     if (gyro_to_left != nullptr && *gyro_to_left)
                     {
                         native.gyro_to_left = std::make_shared<OrbbecEgoExtrinsicsT>();
                         (*gyro_to_left)->UnPackTo(native.gyro_to_left.get());
                     }
                     native.raw_alignment_yaml = raw_alignment_yaml;
                     native.raw_imu_yaml = raw_imu_yaml;
                     native.capture_epoch = capture_epoch;
                     return pack<OrbbecEgoCalibration>(native);
                 }),
             py::arg("device_uid") = std::string{}, py::arg("color_left") = nullptr, py::arg("color_right") = nullptr,
             py::arg("left_to_right") = nullptr, py::arg("accel_to_left") = nullptr, py::arg("gyro_to_left") = nullptr,
             py::arg("raw_alignment_yaml") = std::string{}, py::arg("raw_imu_yaml") = std::string{},
             py::arg("capture_epoch") = 0)
        .def_property_readonly("device_uid", string_field(&OrbbecEgoCalibration::device_uid))
        .def_property_readonly("color_left",
                               [](const Serialized<OrbbecEgoCalibration>& self) -> py::object
                               {
                                   const auto* value = self->color_left();
                                   return value != nullptr ? py::cast(self.narrow(value)) : py::none();
                               })
        .def_property_readonly("color_right",
                               [](const Serialized<OrbbecEgoCalibration>& self) -> py::object
                               {
                                   const auto* value = self->color_right();
                                   return value != nullptr ? py::cast(self.narrow(value)) : py::none();
                               })
        .def_property_readonly("left_to_right",
                               [](const Serialized<OrbbecEgoCalibration>& self) -> py::object
                               {
                                   const auto* value = self->left_to_right();
                                   return value != nullptr ? py::cast(self.narrow(value)) : py::none();
                               })
        .def_property_readonly("accel_to_left",
                               [](const Serialized<OrbbecEgoCalibration>& self) -> py::object
                               {
                                   const auto* value = self->accel_to_left();
                                   return value != nullptr ? py::cast(self.narrow(value)) : py::none();
                               })
        .def_property_readonly("gyro_to_left",
                               [](const Serialized<OrbbecEgoCalibration>& self) -> py::object
                               {
                                   const auto* value = self->gyro_to_left();
                                   return value != nullptr ? py::cast(self.narrow(value)) : py::none();
                               })
        .def_property_readonly("raw_alignment_yaml", string_field(&OrbbecEgoCalibration::raw_alignment_yaml))
        .def_property_readonly("raw_imu_yaml", string_field(&OrbbecEgoCalibration::raw_imu_yaml))
        .def_property_readonly("capture_epoch", field(&OrbbecEgoCalibration::capture_epoch));

    bind_record<OrbbecEgoCalibrationRecord, OrbbecEgoCalibration>(
        m, "OrbbecEgoCalibrationRecord", "OrbbecEgoCalibration");

    py::enum_<OrbbecEgoCaptureHealth>(m, "OrbbecEgoCaptureHealth")
        .value("Healthy", OrbbecEgoCaptureHealth_Healthy)
        .value("Warning", OrbbecEgoCaptureHealth_Warning)
        .value("Incomplete", OrbbecEgoCaptureHealth_Incomplete);

    py::enum_<OrbbecEgoRecoveryStage>(m, "OrbbecEgoRecoveryStage")
        .value("Started", OrbbecEgoRecoveryStage_Started)
        .value("Recovered", OrbbecEgoRecoveryStage_Recovered)
        .value("Failed", OrbbecEgoRecoveryStage_Failed);

    py::enum_<OrbbecEgoDiagnosticEvent>(m, "OrbbecEgoDiagnosticEvent")
        .value("SdkReport", OrbbecEgoDiagnosticEvent_OrbbecEgoSdkReport)
        .value("CaptureSummary", OrbbecEgoDiagnosticEvent_OrbbecEgoCaptureSummary)
        .value("RecoveryEvent", OrbbecEgoDiagnosticEvent_OrbbecEgoRecoveryEvent)
        .value("DeviceSnapshot", OrbbecEgoDiagnosticEvent_OrbbecEgoDeviceSnapshot);

    py::class_<OrbbecEgoDevicePropertyValue>(m, "OrbbecEgoDevicePropertyValue")
        .def(py::init<int32_t, double>(), py::arg("property_id") = 0, py::arg("value") = 0.0)
        .def_property("property_id", &OrbbecEgoDevicePropertyValue::property_id,
                      &OrbbecEgoDevicePropertyValue::mutate_property_id)
        .def_property("value", &OrbbecEgoDevicePropertyValue::value, &OrbbecEgoDevicePropertyValue::mutate_value);

    serialized_class<OrbbecEgoSdkReport>(m, "OrbbecEgoSdkReport", "Typed EGO diagnostic event.")
        .def(py::init(
                 [](uint64_t sequence_number, int32_t work_mode, uint64_t status_flags, uint64_t error_flags,
                    uint64_t storage_free_bytes)
                 {
                     OrbbecEgoSdkReportT native;
                     native.sequence_number = sequence_number;
                     native.work_mode = work_mode;
                     native.status_flags = status_flags;
                     native.error_flags = error_flags;
                     native.storage_free_bytes = storage_free_bytes;
                     return pack<OrbbecEgoSdkReport>(native);
                 }),
             py::arg("sequence_number"), py::arg("work_mode"), py::arg("status_flags"), py::arg("error_flags"),
             py::arg("storage_free_bytes"))
        .def_property_readonly("sequence_number", field(&OrbbecEgoSdkReport::sequence_number))
        .def_property_readonly("work_mode", field(&OrbbecEgoSdkReport::work_mode))
        .def_property_readonly("status_flags", field(&OrbbecEgoSdkReport::status_flags))
        .def_property_readonly("error_flags", field(&OrbbecEgoSdkReport::error_flags))
        .def_property_readonly("storage_free_bytes", field(&OrbbecEgoSdkReport::storage_free_bytes));

    serialized_class<OrbbecEgoCaptureSummary>(m, "OrbbecEgoCaptureSummary", "Typed EGO diagnostic event.")
        .def(py::init(
                 [](OrbbecEgoCaptureHealth capture_health, const std::string& reason, uint32_t queue_capacity,
                    uint32_t queue_peak, uint64_t dropped_events, uint64_t dropped_video_frame_sets, bool is_final)
                 {
                     if (capture_health < OrbbecEgoCaptureHealth_MIN || capture_health > OrbbecEgoCaptureHealth_MAX)
                         throw py::value_error("Invalid EGO capture health");
                     OrbbecEgoCaptureSummaryT native;
                     native.capture_health = capture_health;
                     native.reason = reason;
                     native.queue_capacity = queue_capacity;
                     native.queue_peak = queue_peak;
                     native.dropped_events = dropped_events;
                     native.dropped_video_frame_sets = dropped_video_frame_sets;
                     native.is_final = is_final;
                     return pack<OrbbecEgoCaptureSummary>(native);
                 }),
             py::arg("capture_health"), py::arg("reason"), py::arg("queue_capacity"), py::arg("queue_peak"),
             py::arg("dropped_events"), py::arg("dropped_video_frame_sets"), py::arg("is_final"))
        .def_property_readonly("capture_health", field(&OrbbecEgoCaptureSummary::capture_health))
        .def_property_readonly("reason", string_field(&OrbbecEgoCaptureSummary::reason))
        .def_property_readonly("queue_capacity", field(&OrbbecEgoCaptureSummary::queue_capacity))
        .def_property_readonly("queue_peak", field(&OrbbecEgoCaptureSummary::queue_peak))
        .def_property_readonly("dropped_events", field(&OrbbecEgoCaptureSummary::dropped_events))
        .def_property_readonly("dropped_video_frame_sets", field(&OrbbecEgoCaptureSummary::dropped_video_frame_sets))
        .def_property_readonly("is_final", field(&OrbbecEgoCaptureSummary::is_final));

    serialized_class<OrbbecEgoRecoveryEvent>(m, "OrbbecEgoRecoveryEvent", "Typed EGO diagnostic event.")
        .def(py::init(
                 [](OrbbecEgoRecoveryStage stage, const std::string& reason, uint32_t reconnect_attempt)
                 {
                     if (stage < OrbbecEgoRecoveryStage_MIN || stage > OrbbecEgoRecoveryStage_MAX)
                         throw py::value_error("Invalid EGO recovery stage");
                     OrbbecEgoRecoveryEventT native;
                     native.stage = stage;
                     native.reason = reason;
                     native.reconnect_attempt = reconnect_attempt;
                     return pack<OrbbecEgoRecoveryEvent>(native);
                 }),
             py::arg("stage"), py::arg("reason"), py::arg("reconnect_attempt"))
        .def_property_readonly("stage", field(&OrbbecEgoRecoveryEvent::stage))
        .def_property_readonly("reason", string_field(&OrbbecEgoRecoveryEvent::reason))
        .def_property_readonly("reconnect_attempt", field(&OrbbecEgoRecoveryEvent::reconnect_attempt));

    serialized_class<OrbbecEgoDeviceSnapshot>(
        m, "OrbbecEgoDeviceSnapshot", "Successful startup reads; unavailable temperature is None.")
        .def(py::init(
                 [](std::optional<float> imu_temperature_c, const std::vector<OrbbecEgoDevicePropertyValue>& properties)
                 {
                     if (imu_temperature_c && !std::isfinite(*imu_temperature_c))
                         throw py::value_error("EGO IMU temperature must be finite or None");
                     OrbbecEgoDeviceSnapshotT native;
                     if (imu_temperature_c)
                         native.imu_temperature_c = *imu_temperature_c;
                     native.properties = properties;
                     return pack<OrbbecEgoDeviceSnapshot>(native);
                 }),
             py::arg("imu_temperature_c"), py::arg("properties"))
        .def_property_readonly("imu_temperature_c",
                               [](const Serialized<OrbbecEgoDeviceSnapshot>& self) -> py::object
                               {
                                   const auto temperature = self->imu_temperature_c();
                                   return temperature ? py::cast(*temperature) : py::none();
                               })
        .def_property_readonly("properties",
                               [](const Serialized<OrbbecEgoDeviceSnapshot>& self)
                               {
                                   std::vector<OrbbecEgoDevicePropertyValue> values;
                                   if (self->properties())
                                       for (const auto* value : *self->properties())
                                           values.push_back(*value);
                                   return values;
                               });

    using DiagnosticPayload = std::variant<Serialized<OrbbecEgoSdkReport>, Serialized<OrbbecEgoCaptureSummary>,
                                           Serialized<OrbbecEgoRecoveryEvent>, Serialized<OrbbecEgoDeviceSnapshot>>;
    serialized_class<OrbbecEgoDeviceState>(
        m, "OrbbecEgoDeviceState", "One diagnostic event, separate from NV current availability.")
        .def(py::init(
                 [](const std::string& device_uid, uint32_t capture_epoch, int64_t observed_at_ns,
                    const DiagnosticPayload& event)
                 {
                     OrbbecEgoDeviceStateT native;
                     native.device_uid = device_uid;
                     native.capture_epoch = capture_epoch;
                     native.observed_at_ns = observed_at_ns;
                     std::visit(
                         [&](const auto& value)
                         {
                             using Table = std::remove_cvref_t<decltype(*value)>;
                             typename Table::NativeTableType payload;
                             value->UnPackTo(&payload);
                             native.event.Set(std::move(payload));
                         },
                         event);
                     auto encoded = pack<OrbbecEgoDeviceState>(native);
                     validate_orbbec_ego_diagnostic(*encoded);
                     return encoded;
                 }),
             py::arg("device_uid"), py::arg("capture_epoch"), py::arg("observed_at_ns"), py::arg("event"))
        .def_property_readonly("device_uid", string_field(&OrbbecEgoDeviceState::device_uid))
        .def_property_readonly("capture_epoch", field(&OrbbecEgoDeviceState::capture_epoch))
        .def_property_readonly("observed_at_ns", field(&OrbbecEgoDeviceState::observed_at_ns))
        .def_property_readonly("event_type", field(&OrbbecEgoDeviceState::event_type))
        .def_property_readonly("event",
                               [](const Serialized<OrbbecEgoDeviceState>& self) -> py::object
                               {
                                   validate_orbbec_ego_diagnostic(*self);
                                   // Nested event views retain the record buffer after the parent is released.
                                   switch (self->event_type())
                                   {
                                   case OrbbecEgoDiagnosticEvent_OrbbecEgoSdkReport:
                                       return py::cast(self.narrow(self->event_as_OrbbecEgoSdkReport()));
                                   case OrbbecEgoDiagnosticEvent_OrbbecEgoCaptureSummary:
                                       return py::cast(self.narrow(self->event_as_OrbbecEgoCaptureSummary()));
                                   case OrbbecEgoDiagnosticEvent_OrbbecEgoRecoveryEvent:
                                       return py::cast(self.narrow(self->event_as_OrbbecEgoRecoveryEvent()));
                                   case OrbbecEgoDiagnosticEvent_OrbbecEgoDeviceSnapshot:
                                       return py::cast(self.narrow(self->event_as_OrbbecEgoDeviceSnapshot()));
                                   default:
                                       throw py::value_error("Unknown EGO diagnostic event type");
                                   }
                               });

    bind_record<OrbbecEgoDeviceStateRecord, OrbbecEgoDeviceState>(
        m, "OrbbecEgoDeviceStateRecord", "OrbbecEgoDeviceState");
}

} // namespace core
