// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Python bindings for the SkeletonBinding FlatBuffer schema and the skeleton profile registry.
// Types: BodySide (enum), SkeletonProfileRef, AssetRef and SkeletonBinding (tables, exposed as
// encoded views), and SkeletonProfile (a registry entry).

#pragma once

#include "schema_serialized.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <schema/skeleton_binding.hpp>

#include <memory>
#include <string>
#include <vector>

namespace py = pybind11;

namespace core
{

inline void bind_skeleton_binding(py::module& m)
{
    py::enum_<BodySide>(m, "BodySide")
        .value("UNSPECIFIED", BodySide_UNSPECIFIED)
        .value("LEFT", BodySide_LEFT)
        .value("RIGHT", BodySide_RIGHT);

    m.attr("SKELETON_PROFILE_OPENXR_HAND") = std::string(SKELETON_PROFILE_OPENXR_HAND);
    m.attr("SKELETON_PROFILE_OPENXR_BODY_BD") = std::string(SKELETON_PROFILE_OPENXR_BODY_BD);
    m.attr("SKELETON_PROFILE_URDF_LINK") = std::string(SKELETON_PROFILE_URDF_LINK);

    serialized_class<SkeletonProfileRef>(m, "SkeletonProfileRef", "Encoded reference to a skeleton profile.")
        .def(py::init(
                 [](const std::string& id, uint32_t version)
                 {
                     SkeletonProfileRefT native;
                     native.id = id;
                     native.version = version;
                     return pack<SkeletonProfileRef>(native);
                 }),
             py::arg("id"), py::arg("version"), "Encode a reference to profile `id` at major `version`.")
        .def_property_readonly("id", string_field(&SkeletonProfileRef::id))
        .def_property_readonly("version", field(&SkeletonProfileRef::version))
        .def("__repr__",
             [](const Serialized<SkeletonProfileRef>& self)
             {
                 const auto* id = self->id();
                 return "SkeletonProfileRef(id=" + (id != nullptr ? id->str() : std::string{}) +
                        ", version=" + std::to_string(self->version()) + ")";
             });

    serialized_class<AssetRef>(m, "AssetRef", "Encoded reference to an asset, identified by its digest.")
        .def(py::init(
                 [](const std::string& digest, const std::string& uri, const std::string& media_type)
                 {
                     AssetRefT native;
                     native.digest = digest;
                     native.uri = uri;
                     native.media_type = media_type;
                     return pack<AssetRef>(native);
                 }),
             py::arg("digest"), py::arg("uri") = std::string{}, py::arg("media_type") = std::string{},
             "Encode an asset reference. `asset_digest()` computes `digest` from the asset's bytes.")
        .def_property_readonly("digest", string_field(&AssetRef::digest))
        .def_property_readonly("uri", string_field(&AssetRef::uri))
        .def_property_readonly("media_type", string_field(&AssetRef::media_type))
        .def("__repr__",
             [](const Serialized<AssetRef>& self)
             {
                 const auto* digest = self->digest();
                 return "AssetRef(digest=" + (digest != nullptr ? digest->str() : std::string{}) + ")";
             });

    serialized_class<SkeletonBinding>(m, "SkeletonBinding", "Encoded names of a device's segments in a skeleton.")
        .def(py::init(
                 [](const std::vector<std::string>& segment_names, const Serialized<SkeletonProfileRef>* profile,
                    const Serialized<AssetRef>* asset)
                 {
                     SkeletonBindingT native;
                     native.segment_names = segment_names;
                     if (profile != nullptr && *profile)
                     {
                         native.profile = std::make_shared<SkeletonProfileRefT>();
                         (*profile)->UnPackTo(native.profile.get());
                     }
                     if (asset != nullptr && *asset)
                     {
                         native.asset = std::make_shared<AssetRefT>();
                         (*asset)->UnPackTo(native.asset.get());
                     }
                     return pack<SkeletonBinding>(native);
                 }),
             py::arg("segment_names"), py::arg("profile") = nullptr, py::arg("asset") = nullptr,
             "Encode a binding. Without `profile` it is generic: the names are labels only.")
        .def_property_readonly("segment_names",
                               [](const Serialized<SkeletonBinding>& self)
                               {
                                   std::vector<std::string> names;
                                   if (const auto* encoded = self->segment_names())
                                   {
                                       for (const auto* name : *encoded)
                                       {
                                           names.push_back(name->str());
                                       }
                                   }
                                   return names;
                               })
        .def_property_readonly(
            "profile",
            [](const Serialized<SkeletonBinding>& self) -> py::object
            {
                const auto* profile = self->profile();
                return profile != nullptr ? py::cast(self.narrow(profile)) : py::none();
            },
            "The profile the names refer to, or None for a generic binding.")
        .def_property_readonly(
            "asset",
            [](const Serialized<SkeletonBinding>& self) -> py::object
            {
                const auto* asset = self->asset();
                return asset != nullptr ? py::cast(self.narrow(asset)) : py::none();
            },
            "The asset the profile names segments in, or None.")
        .def("__repr__",
             [](const Serialized<SkeletonBinding>& self)
             {
                 const auto* profile = self->profile();
                 const auto* names = self->segment_names();
                 return "SkeletonBinding(profile=" +
                        (profile != nullptr && profile->id() != nullptr ? profile->id()->str() : std::string("None")) +
                        ", segments=" + std::to_string(names != nullptr ? names->size() : 0) + ")";
             });

    py::class_<SkeletonProfile>(m, "SkeletonProfile", "A registered skeleton profile.")
        .def_property_readonly("id", [](const SkeletonProfile& self) { return std::string(self.id); })
        .def_readonly("version", &SkeletonProfile::version)
        .def_readonly("segment_names", &SkeletonProfile::segment_names,
                      "The names the profile defines; empty when they come from the asset.")
        .def_readonly("requires_side", &SkeletonProfile::requires_side)
        .def_readonly("requires_asset", &SkeletonProfile::requires_asset)
        .def("__repr__", [](const SkeletonProfile& self)
             { return "SkeletonProfile(id=" + std::string(self.id) + ", version=" + std::to_string(self.version) + ")"; });

    m.def("skeleton_profiles", &skeleton_profiles, py::return_value_policy::reference, "Every registered profile.");
    m.def("find_skeleton_profile", &find_skeleton_profile, py::arg("id"), py::arg("version"),
          py::return_value_policy::reference, "The registered profile `id` at `version`, or None.");
    m.def(
        "validate_skeleton_binding",
        [](const Serialized<SkeletonBinding>& binding, size_t segment_count, BodySide side)
        { return validate_skeleton_binding(*binding, segment_count, side); },
        py::arg("binding"), py::arg("segment_count"), py::arg("side") = BodySide_UNSPECIFIED,
        "Check `binding` against a device of `segment_count` segments on `side`. Returns \"\" when valid, "
        "otherwise what is wrong. A profile that is not registered reads as a generic binding.");
}

} // namespace core
