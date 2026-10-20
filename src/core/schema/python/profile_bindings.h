// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "schema_serialized.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <schema/profile_generated.h>

#include <memory>
#include <string>
#include <vector>

namespace py = pybind11;

namespace core
{

inline void bind_profile(py::module& m)
{
    py::enum_<ProfileHandling>(m, "ProfileHandling")
        .value("OPTIONAL", ProfileHandling_OPTIONAL)
        .value("REQUIRED", ProfileHandling_REQUIRED);

    serialized_class<ContentDescriptor>(m, "ContentDescriptor", "Versioned content referenced by a profile.")
        .def(py::init(
                 [](const std::string& logical_id, const std::string& media_type, const std::string& version,
                    const std::vector<std::string>& uris, py::bytes sha256, uint64_t byte_size, py::bytes inline_data)
                 {
                     const std::string digest = sha256;
                     const std::string embedded = inline_data;
                     if (!digest.empty() && digest.size() != 32)
                     {
                         throw py::value_error("sha256 must be empty or exactly 32 bytes");
                     }
                     ContentDescriptorT native;
                     native.logical_id = logical_id;
                     native.media_type = media_type;
                     native.version = version;
                     native.uris = uris;
                     native.sha256.assign(digest.begin(), digest.end());
                     native.byte_size = byte_size;
                     native.inline_data.assign(embedded.begin(), embedded.end());
                     return pack<ContentDescriptor>(native);
                 }),
             py::arg("logical_id"), py::arg("media_type") = std::string{}, py::arg("version") = std::string{},
             py::arg("uris") = std::vector<std::string>{}, py::arg("sha256") = py::bytes(), py::arg("byte_size") = 0,
             py::arg("inline_data") = py::bytes())
        .def_property_readonly("logical_id", string_field(&ContentDescriptor::logical_id))
        .def_property_readonly("media_type", string_field(&ContentDescriptor::media_type))
        .def_property_readonly("version", string_field(&ContentDescriptor::version))
        .def_property_readonly("uris",
                               [](const Serialized<ContentDescriptor>& self)
                               {
                                   std::vector<std::string> values;
                                   const auto* uris = self->uris();
                                   if (uris != nullptr)
                                   {
                                       values.reserve(uris->size());
                                       for (const auto* uri : *uris)
                                       {
                                           values.push_back(uri != nullptr ? uri->str() : std::string{});
                                       }
                                   }
                                   return values;
                               })
        .def_property_readonly("sha256", bytes_field(&ContentDescriptor::sha256))
        .def_property_readonly("byte_size", field(&ContentDescriptor::byte_size))
        .def_property_readonly("inline_data", bytes_field(&ContentDescriptor::inline_data));

    serialized_class<SemanticProfile>(m, "SemanticProfile", "Versioned typed profile with an opaque payload.")
        .def(py::init(
                 [](const std::string& profile_id, uint16_t version_major, uint16_t version_minor,
                    ProfileHandling handling, const std::string& payload_schema, py::bytes payload,
                    const std::vector<Serialized<ContentDescriptor>>& resources)
                 {
                     if (profile_id.empty())
                     {
                         throw py::value_error("profile_id must not be empty");
                     }
                     const std::string data = payload;
                     if (!data.empty() && payload_schema.empty())
                     {
                         throw py::value_error("payload_schema must name non-empty payload bytes");
                     }
                     SemanticProfileT native;
                     native.profile_id = profile_id;
                     native.version_major = version_major;
                     native.version_minor = version_minor;
                     native.handling = handling;
                     native.payload_schema = payload_schema;
                     native.payload.assign(data.begin(), data.end());
                     native.resources = to_native_vector(resources, "resources");
                     return pack<SemanticProfile>(native);
                 }),
             py::arg("profile_id"), py::arg("version_major") = 1, py::arg("version_minor") = 0,
             py::arg("handling") = ProfileHandling_OPTIONAL, py::arg("payload_schema") = std::string{},
             py::arg("payload") = py::bytes(), py::arg("resources") = std::vector<Serialized<ContentDescriptor>>{})
        .def_property_readonly("profile_id", string_field(&SemanticProfile::profile_id))
        .def_property_readonly("version_major", field(&SemanticProfile::version_major))
        .def_property_readonly("version_minor", field(&SemanticProfile::version_minor))
        .def_property_readonly("handling", field(&SemanticProfile::handling))
        .def_property_readonly("payload_schema", string_field(&SemanticProfile::payload_schema))
        .def_property_readonly("payload", bytes_field(&SemanticProfile::payload))
        .def_property_readonly(
            "resources", [](const Serialized<SemanticProfile>& self) { return narrow_vector(self, self->resources()); })
        .def("__repr__",
             [](const Serialized<SemanticProfile>& self)
             {
                 return "SemanticProfile(profile_id='" +
                        (self->profile_id() != nullptr ? self->profile_id()->str() : std::string{}) +
                        "', version=" + std::to_string(self->version_major()) + "." +
                        std::to_string(self->version_minor()) + ")";
             });
}

} // namespace core
