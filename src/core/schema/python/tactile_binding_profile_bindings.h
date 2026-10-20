// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "schema_serialized.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <schema/profile_generated.h>
#include <schema/tactile_binding_profile_generated.h>
#include <schema/tactile_profiles.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace py = pybind11;

namespace core
{

inline Serialized<SemanticProfile> make_tactile_profile(const Serialized<TactileBindingProfile>& binding,
                                                        ProfileHandling handling,
                                                        const std::vector<Serialized<ContentDescriptor>>& resources)
{
    std::string_view profile_id;
    switch (binding->kind())
    {
    case TactileBindingKind_OPENXR_HAND:
        profile_id = TACTILE_OPENXR_HAND_PROFILE_ID;
        break;
    case TactileBindingKind_URDF:
        profile_id = TACTILE_URDF_PROFILE_ID;
        break;
    case TactileBindingKind_ISAACCAPTURE_POSE:
        profile_id = TACTILE_ISAACCAPTURE_POSE_PROFILE_ID;
        break;
    default:
        throw py::value_error("binding profile kind must not be NONE");
    }

    SemanticProfileT native;
    native.profile_id = profile_id;
    native.version_major = 1;
    native.handling = handling;
    native.payload_schema = TACTILE_BINDING_PROFILE_SCHEMA;
    const auto payload = binding.buffer();
    native.payload.assign(payload.begin(), payload.end());
    native.resources = to_native_vector(resources, "resources");
    return pack<SemanticProfile>(native);
}

inline void bind_tactile_binding_profile(py::module& m)
{
    m.attr("TACTILE_OPENXR_HAND_PROFILE_ID") = std::string(TACTILE_OPENXR_HAND_PROFILE_ID);
    m.attr("TACTILE_URDF_PROFILE_ID") = std::string(TACTILE_URDF_PROFILE_ID);
    m.attr("TACTILE_ISAACCAPTURE_POSE_PROFILE_ID") = std::string(TACTILE_ISAACCAPTURE_POSE_PROFILE_ID);
    m.attr("TACTILE_BINDING_PROFILE_SCHEMA") = std::string(TACTILE_BINDING_PROFILE_SCHEMA);

    py::enum_<TactileHandedness>(m, "TactileHandedness")
        .value("UNSPECIFIED", TactileHandedness_UNSPECIFIED)
        .value("LEFT", TactileHandedness_LEFT)
        .value("RIGHT", TactileHandedness_RIGHT);

    py::enum_<UrdfEntityKind>(m, "UrdfEntityKind").value("LINK", UrdfEntityKind_LINK).value("JOINT", UrdfEntityKind_JOINT);

    py::enum_<TactileBindingKind>(m, "TactileBindingKind")
        .value("NONE", TactileBindingKind_NONE)
        .value("OPENXR_HAND", TactileBindingKind_OPENXR_HAND)
        .value("URDF", TactileBindingKind_URDF)
        .value("ISAACCAPTURE_POSE", TactileBindingKind_ISAACCAPTURE_POSE);

    serialized_class<OpenXRHandTactileBinding>(
        m, "OpenXRHandTactileBinding", "Map tactile nodes to XrHandJointEXT ordinals.")
        .def(py::init(
                 [](TactileHandedness handedness, const std::vector<uint8_t>& node_joint_indices)
                 {
                     if (handedness == TactileHandedness_UNSPECIFIED)
                     {
                         throw py::value_error("handedness must be LEFT or RIGHT");
                     }
                     for (uint8_t index : node_joint_indices)
                     {
                         if (index >= 26)
                         {
                             throw py::value_error("OpenXR hand joint indices must be less than 26");
                         }
                     }
                     OpenXRHandTactileBindingT native;
                     native.handedness = handedness;
                     native.node_joint_indices = node_joint_indices;
                     return pack<OpenXRHandTactileBinding>(native);
                 }),
             py::arg("handedness"), py::arg("node_joint_indices"))
        .def_property_readonly("handedness", field(&OpenXRHandTactileBinding::handedness))
        .def_property_readonly("node_joint_indices", vector_field(&OpenXRHandTactileBinding::node_joint_indices));

    serialized_class<UrdfNodeBinding>(m, "UrdfNodeBinding", "Map one tactile node to a URDF link or joint.")
        .def(py::init(
                 [](uint32_t node_index, UrdfEntityKind entity_kind, const std::string& entity_name)
                 {
                     if (entity_name.empty())
                     {
                         throw py::value_error("entity_name must not be empty");
                     }
                     UrdfNodeBindingT native;
                     native.node_index = node_index;
                     native.entity_kind = entity_kind;
                     native.entity_name = entity_name;
                     return pack<UrdfNodeBinding>(native);
                 }),
             py::arg("node_index"), py::arg("entity_kind"), py::arg("entity_name"))
        .def_property_readonly("node_index", field(&UrdfNodeBinding::node_index))
        .def_property_readonly("entity_kind", field(&UrdfNodeBinding::entity_kind))
        .def_property_readonly("entity_name", string_field(&UrdfNodeBinding::entity_name));

    serialized_class<UrdfTactileBinding>(m, "UrdfTactileBinding", "Bind tactile nodes to a referenced URDF.")
        .def(py::init(
                 [](const std::string& resource_id, const std::vector<Serialized<UrdfNodeBinding>>& nodes)
                 {
                     if (resource_id.empty())
                     {
                         throw py::value_error("resource_id must not be empty");
                     }
                     UrdfTactileBindingT native;
                     native.resource_id = resource_id;
                     native.nodes = to_native_vector(nodes, "nodes");
                     return pack<UrdfTactileBinding>(native);
                 }),
             py::arg("resource_id"), py::arg("nodes") = std::vector<Serialized<UrdfNodeBinding>>{})
        .def_property_readonly("resource_id", string_field(&UrdfTactileBinding::resource_id))
        .def_property_readonly(
            "nodes", [](const Serialized<UrdfTactileBinding>& self) { return narrow_vector(self, self->nodes()); });

    serialized_class<IsaacCapturePoseTactileBinding>(
        m, "IsaacCapturePoseTactileBinding", "Associate tactile nodes with an IsaacCapture pose stream.")
        .def(py::init(
                 [](const std::string& pose_schema, const std::string& tracker_name, const std::string& collection_id,
                    const std::string& channel, const std::string& subject_id, const std::string& device_instance_id,
                    const std::string& skeleton_instance_id, const std::vector<uint32_t>& node_pose_indices)
                 {
                     if (pose_schema.empty())
                     {
                         throw py::value_error("pose_schema must not be empty");
                     }
                     if (tracker_name.empty() && collection_id.empty())
                     {
                         throw py::value_error("tracker_name or collection_id must identify the pose stream");
                     }
                     IsaacCapturePoseTactileBindingT native;
                     native.pose_schema = pose_schema;
                     native.tracker_name = tracker_name;
                     native.collection_id = collection_id;
                     native.channel = channel;
                     native.subject_id = subject_id;
                     native.device_instance_id = device_instance_id;
                     native.skeleton_instance_id = skeleton_instance_id;
                     native.node_pose_indices = node_pose_indices;
                     return pack<IsaacCapturePoseTactileBinding>(native);
                 }),
             py::arg("pose_schema"), py::arg("tracker_name") = std::string{}, py::arg("collection_id") = std::string{},
             py::arg("channel") = std::string{}, py::arg("subject_id") = std::string{},
             py::arg("device_instance_id") = std::string{}, py::arg("skeleton_instance_id") = std::string{},
             py::arg("node_pose_indices") = std::vector<uint32_t>{})
        .def_property_readonly("pose_schema", string_field(&IsaacCapturePoseTactileBinding::pose_schema))
        .def_property_readonly("tracker_name", string_field(&IsaacCapturePoseTactileBinding::tracker_name))
        .def_property_readonly("collection_id", string_field(&IsaacCapturePoseTactileBinding::collection_id))
        .def_property_readonly("channel", string_field(&IsaacCapturePoseTactileBinding::channel))
        .def_property_readonly("subject_id", string_field(&IsaacCapturePoseTactileBinding::subject_id))
        .def_property_readonly("device_instance_id", string_field(&IsaacCapturePoseTactileBinding::device_instance_id))
        .def_property_readonly(
            "skeleton_instance_id", string_field(&IsaacCapturePoseTactileBinding::skeleton_instance_id))
        .def_property_readonly("node_pose_indices", vector_field(&IsaacCapturePoseTactileBinding::node_pose_indices));

    serialized_class<TactileBindingProfile>(m, "TactileBindingProfile", "Typed tactile binding profile payload.")
        .def(py::init(
                 [](const Serialized<OpenXRHandTactileBinding>* openxr_hand, const Serialized<UrdfTactileBinding>* urdf,
                    const Serialized<IsaacCapturePoseTactileBinding>* isaaccapture_pose)
                 {
                     const int selected = (openxr_hand != nullptr) + (urdf != nullptr) + (isaaccapture_pose != nullptr);
                     if (selected != 1)
                     {
                         throw py::value_error("exactly one tactile binding must be supplied");
                     }
                     TactileBindingProfileT native;
                     if (openxr_hand != nullptr)
                     {
                         native.kind = TactileBindingKind_OPENXR_HAND;
                         native.openxr_hand = std::make_shared<OpenXRHandTactileBindingT>();
                         (*openxr_hand)->UnPackTo(native.openxr_hand.get());
                     }
                     else if (urdf != nullptr)
                     {
                         native.kind = TactileBindingKind_URDF;
                         native.urdf = std::make_shared<UrdfTactileBindingT>();
                         (*urdf)->UnPackTo(native.urdf.get());
                     }
                     else
                     {
                         native.kind = TactileBindingKind_ISAACCAPTURE_POSE;
                         native.isaaccapture_pose = std::make_shared<IsaacCapturePoseTactileBindingT>();
                         (*isaaccapture_pose)->UnPackTo(native.isaaccapture_pose.get());
                     }
                     return pack<TactileBindingProfile>(native);
                 }),
             py::arg("openxr_hand").none(true) = nullptr, py::arg("urdf").none(true) = nullptr,
             py::arg("isaaccapture_pose").none(true) = nullptr)
        .def_property_readonly("kind", field(&TactileBindingProfile::kind))
        .def_property_readonly(
            "openxr_hand", [](const Serialized<TactileBindingProfile>& self) -> py::object
            { return self->openxr_hand() != nullptr ? py::cast(self.narrow(self->openxr_hand())) : py::none(); })
        .def_property_readonly("urdf", [](const Serialized<TactileBindingProfile>& self) -> py::object
                               { return self->urdf() != nullptr ? py::cast(self.narrow(self->urdf())) : py::none(); })
        .def_property_readonly("isaaccapture_pose",
                               [](const Serialized<TactileBindingProfile>& self) -> py::object
                               {
                                   return self->isaaccapture_pose() != nullptr ?
                                              py::cast(self.narrow(self->isaaccapture_pose())) :
                                              py::none();
                               });

    m.def("make_tactile_binding_profile", &make_tactile_profile, py::arg("binding"),
          py::arg("handling") = ProfileHandling_REQUIRED,
          py::arg("resources") = std::vector<Serialized<ContentDescriptor>>{},
          "Wrap a typed tactile binding in its versioned SemanticProfile envelope.");
}

} // namespace core
