// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Python bindings for the TactileMetadata and TactileData FlatBuffer schemas.
// Tables are exposed as encoded views; per-taxel fields read as NumPy views into the buffer
// (see schema_array_views.h) and are written from NumPy arrays.

#pragma once

#include "schema_array_views.h"
#include "schema_serialized.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <schema/tactile_validation.hpp>

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace py = pybind11;

namespace core
{

namespace tactile_detail
{

template <typename T>
using CArray = py::array_t<T, py::array::c_style | py::array::forcecast>;

//! Rows of `array`, which must be (rows, width), or (rows,) when width is 0.
template <typename T>
size_t checked_rows(const CArray<T>& array, const char* name, py::ssize_t width)
{
    const bool ok = width > 0 ? array.ndim() == 2 && array.shape(1) == width : array.ndim() == 1;
    if (!ok)
    {
        throw py::value_error(std::string(name) + " must have shape (N" +
                              (width > 0 ? ", " + std::to_string(width) : std::string(",")) + ")");
    }
    return static_cast<size_t>(array.shape(0));
}

//! Integers of `array` narrowed to Out, refusing values Out cannot hold rather than wrapping them.
template <typename Out>
std::vector<Out> checked_integers(const CArray<int64_t>& array, const char* name, int64_t min, int64_t max)
{
    std::vector<Out> values;
    values.reserve(static_cast<size_t>(array.size()));
    for (py::ssize_t i = 0; i < array.size(); ++i)
    {
        const int64_t value = array.data()[i];
        if (value < min || value > max)
        {
            throw py::value_error(std::string(name) + " holds " + std::to_string(value) + ", outside [" +
                                  std::to_string(min) + ", " + std::to_string(max) + "]");
        }
        values.push_back(static_cast<Out>(value));
    }
    return values;
}

//! (count, width) view of a vector's elements, `stride` bytes apart, or None when it is absent.
template <typename T>
py::object vector_view(py::object owner, const uint8_t* first, size_t count, py::ssize_t stride, py::ssize_t width)
{
    if (first == nullptr || count == 0)
    {
        return py::none();
    }
    return strided_field_view<T>(
        owner, reinterpret_cast<const T*>(first), stride, static_cast<py::ssize_t>(count), width);
}

template <typename V>
const uint8_t* vector_data(const V* vector)
{
    return vector != nullptr ? vector->Data() : nullptr;
}

template <typename V>
size_t vector_size(const V* vector)
{
    return vector != nullptr ? vector->size() : 0;
}

template <typename T>
std::shared_ptr<typename T::NativeTableType> unpack(const Serialized<T>* handle)
{
    if (handle == nullptr || !*handle)
    {
        return nullptr;
    }
    auto native = std::make_shared<typename T::NativeTableType>();
    (*handle)->UnPackTo(native.get());
    return native;
}

constexpr py::ssize_t SEGMENT_STRIDE = sizeof(TactileSegment);
constexpr py::ssize_t TAXEL_STRIDE = sizeof(TactileTaxel);

} // namespace tactile_detail

inline void bind_tactile(py::module& m)
{
    using namespace tactile_detail;

    py::enum_<TaxelState>(m, "TaxelState")
        .value("VALID", TaxelState_VALID)
        .value("SATURATED", TaxelState_SATURATED)
        .value("INVALID", TaxelState_INVALID)
        .value("UNAVAILABLE", TaxelState_UNAVAILABLE);

    serialized_class<TactileDevice>(m, "TactileDevice", "Encoded identity of a tactile device.")
        .def(py::init(
                 [](const std::string& vendor, const std::string& model, const std::string& serial_number,
                    const std::string& firmware_version, BodySide side, std::optional<float> nominal_rate_hz,
                    const std::string& calibration_id)
                 {
                     TactileDeviceT native;
                     native.vendor = vendor;
                     native.model = model;
                     native.serial_number = serial_number;
                     native.firmware_version = firmware_version;
                     native.side = side;
                     if (nominal_rate_hz)
                     {
                         native.nominal_rate_hz = *nominal_rate_hz;
                     }
                     native.calibration_id = calibration_id;
                     return pack<TactileDevice>(native);
                 }),
             py::arg("vendor") = std::string{}, py::arg("model") = std::string{},
             py::arg("serial_number") = std::string{}, py::arg("firmware_version") = std::string{},
             py::arg("side") = BodySide_UNSPECIFIED, py::arg("nominal_rate_hz") = py::none(),
             py::arg("calibration_id") = std::string{})
        .def_property_readonly("vendor", string_field(&TactileDevice::vendor))
        .def_property_readonly("model", string_field(&TactileDevice::model))
        .def_property_readonly("serial_number", string_field(&TactileDevice::serial_number))
        .def_property_readonly("firmware_version", string_field(&TactileDevice::firmware_version))
        .def_property_readonly("side", field(&TactileDevice::side))
        .def_property_readonly("nominal_rate_hz",
                               [](const Serialized<TactileDevice>& self) -> std::optional<float>
                               {
                                   const auto rate = self->nominal_rate_hz();
                                   return rate ? std::optional<float>(*rate) : std::nullopt;
                               })
        .def_property_readonly("calibration_id", string_field(&TactileDevice::calibration_id))
        .def("__repr__",
             [](const Serialized<TactileDevice>& self)
             {
                 const auto* vendor = self->vendor();
                 const auto* model = self->model();
                 return "TactileDevice(vendor=" + (vendor != nullptr ? vendor->str() : std::string{}) +
                        ", model=" + (model != nullptr ? model->str() : std::string{}) +
                        ", side=" + EnumNameBodySide(self->side()) + ")";
             });

    serialized_class<TactileChannels>(m, "TactileChannels", "Encoded set of TactileData vectors a device fills.")
        .def(py::init(
                 [](bool normal_force, bool shear_force, uint8_t raw_values_per_taxel, bool taxel_state)
                 {
                     TactileChannelsT native;
                     native.normal_force = normal_force;
                     native.shear_force = shear_force;
                     native.raw_values_per_taxel = raw_values_per_taxel;
                     native.taxel_state = taxel_state;
                     return pack<TactileChannels>(native);
                 }),
             py::arg("normal_force") = false, py::arg("shear_force") = false, py::arg("raw_values_per_taxel") = 0,
             py::arg("taxel_state") = false)
        .def_property_readonly("normal_force", field(&TactileChannels::normal_force))
        .def_property_readonly("shear_force", field(&TactileChannels::shear_force))
        .def_property_readonly("raw_values_per_taxel", field(&TactileChannels::raw_values_per_taxel))
        .def_property_readonly("taxel_state", field(&TactileChannels::taxel_state))
        .def("__repr__",
             [](const Serialized<TactileChannels>& self)
             {
                 return std::string("TactileChannels(normal_force=") + (self->normal_force() ? "True" : "False") +
                        ", shear_force=" + (self->shear_force() ? "True" : "False") +
                        ", raw_values_per_taxel=" + std::to_string(self->raw_values_per_taxel()) +
                        ", taxel_state=" + (self->taxel_state() ? "True" : "False") + ")";
             });

    serialized_class<TactileExtension>(m, "TactileExtension", "Encoded vendor data, named by a registry id.")
        .def(py::init(
                 [](const std::string& id, uint32_t version, const py::bytes& data)
                 {
                     TactileExtensionT native;
                     native.id = id;
                     native.version = version;
                     const std::string bytes = data;
                     native.data.assign(bytes.begin(), bytes.end());
                     return pack<TactileExtension>(native);
                 }),
             py::arg("id"), py::arg("version"), py::arg("data") = py::bytes())
        .def_property_readonly("id", string_field(&TactileExtension::id))
        .def_property_readonly("version", field(&TactileExtension::version))
        .def_property_readonly("data",
                               [](const Serialized<TactileExtension>& self)
                               {
                                   const auto* data = self->data();
                                   return data != nullptr ?
                                              py::bytes(reinterpret_cast<const char*>(data->data()), data->size()) :
                                              py::bytes();
                               })
        .def("__repr__",
             [](const Serialized<TactileExtension>& self)
             {
                 const auto* id = self->id();
                 return "TactileExtension(id=" + (id != nullptr ? id->str() : std::string{}) +
                        ", version=" + std::to_string(self->version()) + ")";
             });

    // Per-row views alias the buffer; see schema_array_views.h for why they are writable.
    serialized_class<TactileShape>(m, "TactileShape", "Encoded segments and taxels of a tactile device.")
        .def(py::init(
                 [](const CArray<float>& segment_bind_positions, const CArray<float>& segment_bind_orientations,
                    const CArray<float>& taxel_positions, const CArray<float>& taxel_orientations,
                    const CArray<int64_t>& taxel_segments, const CArray<float>& taxel_weights,
                    std::optional<CArray<float>> taxel_areas)
                 {
                     const size_t segment_count = checked_rows(segment_bind_positions, "segment_bind_positions", 3);
                     const size_t taxel_count = checked_rows(taxel_positions, "taxel_positions", 3);
                     if (checked_rows(segment_bind_orientations, "segment_bind_orientations", 4) != segment_count)
                     {
                         throw py::value_error("segment_bind_orientations must have one row per segment");
                     }
                     if (checked_rows(taxel_orientations, "taxel_orientations", 4) != taxel_count ||
                         checked_rows(taxel_segments, "taxel_segments", 4) != taxel_count ||
                         checked_rows(taxel_weights, "taxel_weights", 4) != taxel_count)
                     {
                         throw py::value_error(
                             "taxel_orientations, taxel_segments and taxel_weights must have one row per taxel");
                     }
                     const auto segments = checked_integers<uint16_t>(
                         taxel_segments, "taxel_segments", 0, std::numeric_limits<uint16_t>::max());

                     const auto pose = [](const float* p, const float* q)
                     { return Pose(Point(p[0], p[1], p[2]), Quaternion(q[0], q[1], q[2], q[3])); };
                     TactileShapeT native;
                     for (size_t s = 0; s < segment_count; ++s)
                     {
                         native.segments.emplace_back(
                             pose(segment_bind_positions.data() + 3 * s, segment_bind_orientations.data() + 4 * s));
                     }
                     for (size_t i = 0; i < taxel_count; ++i)
                     {
                         native.taxels.emplace_back(
                             pose(taxel_positions.data() + 3 * i, taxel_orientations.data() + 4 * i),
                             flatbuffers::span<const uint16_t, 4>(segments.data() + 4 * i, 4),
                             flatbuffers::span<const float, 4>(taxel_weights.data() + 4 * i, 4));
                     }
                     if (taxel_areas)
                     {
                         if (checked_rows(*taxel_areas, "taxel_areas", 0) != taxel_count)
                         {
                             throw py::value_error("taxel_areas must have one value per taxel");
                         }
                         native.taxel_areas.assign(taxel_areas->data(), taxel_areas->data() + taxel_count);
                     }
                     return pack<TactileShape>(native);
                 }),
             py::arg("segment_bind_positions"), py::arg("segment_bind_orientations"), py::arg("taxel_positions"),
             py::arg("taxel_orientations"), py::arg("taxel_segments"), py::arg("taxel_weights"),
             py::arg("taxel_areas") = py::none(),
             "Encode a shape from per-segment (S, 3) positions and (S, 4) XYZW orientations, and per-taxel "
             "(N, 3) positions, (N, 4) orientations, (N, 4) segment indices, (N, 4) weights and (N,) areas.")
        .def_property_readonly(
            "segment_count", [](const Serialized<TactileShape>& self) { return vector_size(self->segments()); })
        .def_property_readonly(
            "taxel_count", [](const Serialized<TactileShape>& self) { return vector_size(self->taxels()); })
        .def_property_readonly("segment_bind_positions",
                               [](py::object self)
                               {
                                   const auto* segments = self.cast<const Serialized<TactileShape>&>()->segments();
                                   return vector_view<float>(
                                       self, vector_data(segments), vector_size(segments), SEGMENT_STRIDE, 3);
                               })
        .def_property_readonly(
            "segment_bind_orientations",
            [](py::object self)
            {
                const auto* segments = self.cast<const Serialized<TactileShape>&>()->segments();
                const auto* first = segments != nullptr && segments->size() > 0 ?
                                        reinterpret_cast<const uint8_t*>(&segments->Get(0)->bind_pose().orientation()) :
                                        nullptr;
                return vector_view<float>(self, first, vector_size(segments), SEGMENT_STRIDE, 4);
            })
        .def_property_readonly("taxel_positions",
                               [](py::object self)
                               {
                                   const auto* taxels = self.cast<const Serialized<TactileShape>&>()->taxels();
                                   return vector_view<float>(
                                       self, vector_data(taxels), vector_size(taxels), TAXEL_STRIDE, 3);
                               })
        .def_property_readonly("taxel_orientations",
                               [](py::object self)
                               {
                                   const auto* taxels = self.cast<const Serialized<TactileShape>&>()->taxels();
                                   const auto* first =
                                       taxels != nullptr && taxels->size() > 0 ?
                                           reinterpret_cast<const uint8_t*>(&taxels->Get(0)->pose().orientation()) :
                                           nullptr;
                                   return vector_view<float>(self, first, vector_size(taxels), TAXEL_STRIDE, 4);
                               })
        .def_property_readonly("taxel_segments",
                               [](py::object self)
                               {
                                   const auto* taxels = self.cast<const Serialized<TactileShape>&>()->taxels();
                                   const auto* first = taxels != nullptr && taxels->size() > 0 ?
                                                           taxels->Get(0)->segments()->Data() :
                                                           nullptr;
                                   return vector_view<uint16_t>(self, first, vector_size(taxels), TAXEL_STRIDE, 4);
                               })
        .def_property_readonly("taxel_weights",
                               [](py::object self)
                               {
                                   const auto* taxels = self.cast<const Serialized<TactileShape>&>()->taxels();
                                   const auto* first = taxels != nullptr && taxels->size() > 0 ?
                                                           taxels->Get(0)->weights()->Data() :
                                                           nullptr;
                                   return vector_view<float>(self, first, vector_size(taxels), TAXEL_STRIDE, 4);
                               })
        .def_property_readonly("taxel_areas",
                               [](py::object self)
                               {
                                   const auto* areas = self.cast<const Serialized<TactileShape>&>()->taxel_areas();
                                   return vector_view<float>(
                                       self, vector_data(areas), vector_size(areas), sizeof(float), 0);
                               })
        .def("__repr__",
             [](const Serialized<TactileShape>& self)
             {
                 return "TactileShape(segments=" + std::to_string(vector_size(self->segments())) +
                        ", taxels=" + std::to_string(vector_size(self->taxels())) + ")";
             });

    serialized_class<TactileMetadata>(m, "TactileMetadata", "Encoded description of a tactile device.")
        .def(py::init(
                 [](const Serialized<TactileShape>& shape, const Serialized<TactileChannels>& channels,
                    const Serialized<TactileDevice>* device, const Serialized<SkeletonBinding>* binding,
                    const std::vector<Serialized<TactileExtension>>& extensions)
                 {
                     TactileMetadataT native;
                     native.shape = unpack(&shape);
                     native.channels = unpack(&channels);
                     native.device = unpack(device);
                     native.binding = unpack(binding);
                     native.extensions = to_native_vector(extensions, "extensions");
                     auto metadata = pack<TactileMetadata>(native);
                     const std::string error = validate_tactile_metadata(*metadata);
                     if (!error.empty())
                     {
                         throw py::value_error(error);
                     }
                     return metadata;
                 }),
             py::arg("shape"), py::arg("channels"), py::arg("device") = nullptr, py::arg("binding") = nullptr,
             py::arg("extensions") = std::vector<Serialized<TactileExtension>>{},
             "Encode and validate a device description; raises ValueError naming the first rule it breaks.")
        .def_property_readonly("device",
                               [](const Serialized<TactileMetadata>& self) -> py::object
                               { return self->device() != nullptr ? py::cast(self.narrow(self->device())) : py::none(); })
        .def_property_readonly("shape",
                               [](const Serialized<TactileMetadata>& self) -> py::object
                               { return self->shape() != nullptr ? py::cast(self.narrow(self->shape())) : py::none(); })
        .def_property_readonly(
            "binding",
            [](const Serialized<TactileMetadata>& self) -> py::object
            { return self->binding() != nullptr ? py::cast(self.narrow(self->binding())) : py::none(); })
        .def_property_readonly(
            "channels",
            [](const Serialized<TactileMetadata>& self) -> py::object
            { return self->channels() != nullptr ? py::cast(self.narrow(self->channels())) : py::none(); })
        .def_property_readonly("extensions", [](const Serialized<TactileMetadata>& self)
                               { return narrow_vector(self, self->extensions()); })
        .def("__repr__",
             [](const Serialized<TactileMetadata>& self)
             {
                 const auto* shape = self->shape();
                 return "TactileMetadata(taxels=" + std::to_string(shape != nullptr ? vector_size(shape->taxels()) : 0) +
                        ")";
             });

    serialized_class<TactileData>(m, "TactileData", "Encoded frame of per-taxel tactile readings.")
        .def(py::init(
                 [](std::optional<CArray<float>> normal_force, std::optional<CArray<float>> shear_force,
                    std::optional<CArray<int64_t>> raw, std::optional<CArray<int64_t>> taxel_state,
                    std::optional<uint32_t> sequence)
                 {
                     TactileDataT native;
                     if (normal_force)
                     {
                         const size_t n = checked_rows(*normal_force, "normal_force", 0);
                         native.normal_force.assign(normal_force->data(), normal_force->data() + n);
                     }
                     if (shear_force)
                     {
                         const size_t n = checked_rows(*shear_force, "shear_force", 2);
                         for (size_t i = 0; i < n; ++i)
                         {
                             native.shear_force.emplace_back(shear_force->data()[2 * i], shear_force->data()[2 * i + 1]);
                         }
                     }
                     if (raw)
                     {
                         checked_rows(*raw, "raw", 0);
                         native.raw = checked_integers<int32_t>(
                             *raw, "raw", std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max());
                     }
                     if (taxel_state)
                     {
                         checked_rows(*taxel_state, "taxel_state", 0);
                         native.taxel_state =
                             checked_integers<TaxelState>(*taxel_state, "taxel_state", TaxelState_MIN, TaxelState_MAX);
                     }
                     if (sequence)
                     {
                         native.sequence = *sequence;
                     }
                     return pack<TactileData>(native);
                 }),
             py::arg("normal_force") = py::none(), py::arg("shear_force") = py::none(), py::arg("raw") = py::none(),
             py::arg("taxel_state") = py::none(), py::arg("sequence") = py::none(),
             "Encode one frame from (N,) normal forces, (N, 2) shear forces, (N * k,) raw readings and (N,) "
             "TaxelState values; leave out the channels the device does not declare.")
        .def_property_readonly("normal_force",
                               [](py::object self)
                               {
                                   const auto* values = self.cast<const Serialized<TactileData>&>()->normal_force();
                                   return vector_view<float>(
                                       self, vector_data(values), vector_size(values), sizeof(float), 0);
                               })
        .def_property_readonly("shear_force",
                               [](py::object self)
                               {
                                   const auto* values = self.cast<const Serialized<TactileData>&>()->shear_force();
                                   return vector_view<float>(
                                       self, vector_data(values), vector_size(values), sizeof(TactileShear), 2);
                               })
        .def_property_readonly("raw",
                               [](py::object self)
                               {
                                   const auto* values = self.cast<const Serialized<TactileData>&>()->raw();
                                   return vector_view<int32_t>(
                                       self, vector_data(values), vector_size(values), sizeof(int32_t), 0);
                               })
        .def_property_readonly("taxel_state",
                               [](py::object self)
                               {
                                   const auto* values = self.cast<const Serialized<TactileData>&>()->taxel_state();
                                   return vector_view<uint8_t>(
                                       self, vector_data(values), vector_size(values), sizeof(uint8_t), 0);
                               })
        .def_property_readonly("sequence",
                               [](const Serialized<TactileData>& self) -> std::optional<uint32_t>
                               {
                                   const auto sequence = self->sequence();
                                   return sequence ? std::optional<uint32_t>(*sequence) : std::nullopt;
                               })
        .def("__repr__",
             [](const Serialized<TactileData>& self)
             {
                 return "TactileData(normal_force=" + std::to_string(vector_size(self->normal_force())) +
                        ", shear_force=" + std::to_string(vector_size(self->shear_force())) +
                        ", raw=" + std::to_string(vector_size(self->raw())) +
                        ", taxel_state=" + std::to_string(vector_size(self->taxel_state())) + ")";
             });

    bind_record<TactileDataRecord, TactileData>(m, "TactileDataRecord", "TactileData");

    m.def(
        "validate_tactile_metadata",
        [](const Serialized<TactileMetadata>& metadata) { return validate_tactile_metadata(*metadata); },
        py::arg("metadata"), "Check a device description. Returns \"\" when valid, otherwise the first rule it breaks.");
    m.def(
        "validate_tactile_data",
        [](const Serialized<TactileData>& data, const Serialized<TactileMetadata>& metadata)
        {
            const std::string error = validate_tactile_metadata(*metadata);
            return error.empty() ? validate_tactile_data(*data, *metadata) : "metadata: " + error;
        },
        py::arg("data"), py::arg("metadata"),
        "Check that a frame fills exactly the channels `metadata` declares, one entry per taxel. Returns \"\" "
        "when valid, otherwise what is wrong.");
}

} // namespace core
