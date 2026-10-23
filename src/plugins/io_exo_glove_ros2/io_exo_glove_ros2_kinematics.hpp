// SPDX-FileCopyrightText: Copyright (c) 2026 IO. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "io_exo_glove_ros2_hand.hpp"

#include <openxr/openxr.h>
#include <tf2_msgs/msg/tf_message.hpp>

#include <array>
#include <cstddef>

namespace plugins
{
namespace io_exo_glove_ros2
{

/*!
 * @brief The exoskeleton's articulation, re-expressed as an anatomical OpenXR hand.
 *
 * The exoskeleton publishes the poses of its *linkage mechanism*: in the vendor's own URDF each
 * finger is a chain of four revolute links whose segments are 7.6 cm and 7.5 cm long, roughly twice a
 * human phalanx, and whose arm leaves the finger plane by 7 cm between two joints. Injecting those
 * frames as hand joints therefore yields a hand with a palm of a few millimetres and fingers twice a
 * palm long. What carries over is the *articulation*: the bend at each linkage joint, and the spread
 * at the knuckle. This module reads those joint angles from the published frame quaternions and drives
 * a hand skeleton whose landmarks come from the vendor's human-hand model instead.
 *
 * Everything below is a constant derived from the two URDFs that ship with the vendor SDK
 * (`io_exotrans2hand_project_zenoh_22.04_x86_v2.1.4/src/exo_tf_pkg/description/`), so nothing here
 * depends on when a frame happens to arrive:
 *
 *  - ``blender_human_skeleton_v5.urdf`` gives the mechanism's chain: ``joint_<side>SkeletonBase`` is
 *    the fixed root, and each finger is ``joint_<F>1`` (the knuckle spread) followed by ``joint_<F>2``,
 *    ``3``, ``4`` (the bends). Every joint carries an ``<origin rpy>`` rest rotation and a revolute
 *    ``<axis>``, so the published child frame is parent * R_rest(rpy) * R(axis, theta); the joint's
 *    absolute angle follows from the two frames' quaternions (see the table in the .cpp).
 *  - ``blender_human.urdf`` gives the anatomy: the wrist frame is ``link_<side>Hand_Y``, each finger
 *    starts at ``joint_<side>InHand<F>`` (the metacarpal base) and continues through
 *    ``joint_<side>Hand<F>1`` (MCP), ``2`` (PIP) and ``3`` (DIP).
 *
 * The anatomy is placed in the OpenXR wrist frame directly -- ``-Z`` runs along each bone, ``+Y`` is
 * the back of the hand, ``+X`` points from the little finger towards the index -- which is the
 * convention ``XR_EXT_hand_tracking`` states, so no frame alignment between the glove and the headset
 * is needed at run time: only angles cross over, and the wrist pose comes from the wrist source.
 */

//! One anatomical landmark: the OpenXR joint and its offset from the parent landmark, in metres.
struct HumanLandmark
{
    XrHandJointEXT joint;
    XrVector3f offset;
};

//! One finger's anatomical chain: the landmarks in order and how many it has. The thumb has four
//! joints (CMC, MCP, IP, tip); the other fingers five (CMC, MCP, PIP, DIP, tip).
struct HumanFinger
{
    const char* name;
    std::size_t count;
    const HumanLandmark* landmarks;
};

//! Distal phalanx length, metres: the human hand model stops at the DIP joint, so the tip is placed
//! one distal phalanx further along. Anatomical range 1.8-2.5 cm.
inline constexpr float kDistalPhalanxLength = 0.022f;

//! Palm length used for the PALM joint, metres: halfway from the wrist to the middle MCP.
inline constexpr float kPalmCentreFromWrist = 0.045f;

//! Landmarks from ``blender_human.urdf``, mapped into the OpenXR wrist frame by a -90 deg rotation
//! about Z (that model runs its bones along X and its palm normal along Y) and rounded to 0.1 mm.
inline constexpr HumanLandmark kThumbLandmarks[] = {
    { XR_HAND_JOINT_THUMB_METACARPAL_EXT, { 0.0208f, -0.0023f, -0.0243f } },
    { XR_HAND_JOINT_THUMB_PROXIMAL_EXT, { 0.0000f, 0.0000f, -0.0387f } },
    { XR_HAND_JOINT_THUMB_DISTAL_EXT, { 0.0000f, 0.0000f, -0.0269f } },
    { XR_HAND_JOINT_THUMB_TIP_EXT, { 0.0000f, 0.0000f, -0.0220f } },
};
inline constexpr HumanLandmark kIndexLandmarks[] = {
    { XR_HAND_JOINT_INDEX_METACARPAL_EXT, { 0.0111f, -0.0043f, -0.0383f } },
    { XR_HAND_JOINT_INDEX_PROXIMAL_EXT, { 0.0000f, 0.0000f, -0.0559f } },
    { XR_HAND_JOINT_INDEX_INTERMEDIATE_EXT, { 0.0000f, 0.0000f, -0.0381f } },
    { XR_HAND_JOINT_INDEX_DISTAL_EXT, { 0.0000f, 0.0000f, -0.0216f } },
    { XR_HAND_JOINT_INDEX_TIP_EXT, { 0.0000f, 0.0000f, -0.0220f } },
};
inline constexpr HumanLandmark kMiddleLandmarks[] = {
    { XR_HAND_JOINT_MIDDLE_METACARPAL_EXT, { -0.0014f, -0.0014f, -0.0368f } },
    { XR_HAND_JOINT_MIDDLE_PROXIMAL_EXT, { 0.0000f, 0.0000f, -0.0545f } },
    { XR_HAND_JOINT_MIDDLE_INTERMEDIATE_EXT, { 0.0000f, 0.0000f, -0.0415f } },
    { XR_HAND_JOINT_MIDDLE_DISTAL_EXT, { 0.0000f, 0.0000f, -0.0260f } },
    { XR_HAND_JOINT_MIDDLE_TIP_EXT, { 0.0000f, 0.0000f, -0.0220f } },
};
inline constexpr HumanLandmark kRingLandmarks[] = {
    { XR_HAND_JOINT_RING_METACARPAL_EXT, { -0.0103f, 0.0003f, -0.0344f } },
    { XR_HAND_JOINT_RING_PROXIMAL_EXT, { 0.0000f, 0.0000f, -0.0490f } },
    { XR_HAND_JOINT_RING_INTERMEDIATE_EXT, { 0.0000f, 0.0000f, -0.0362f } },
    { XR_HAND_JOINT_RING_DISTAL_EXT, { 0.0000f, 0.0000f, -0.0251f } },
    { XR_HAND_JOINT_RING_TIP_EXT, { 0.0000f, 0.0000f, -0.0220f } },
};
inline constexpr HumanLandmark kLittleLandmarks[] = {
    { XR_HAND_JOINT_LITTLE_METACARPAL_EXT, { -0.0203f, 0.0031f, -0.0295f } },
    { XR_HAND_JOINT_LITTLE_PROXIMAL_EXT, { 0.0000f, 0.0000f, -0.0450f } },
    { XR_HAND_JOINT_LITTLE_INTERMEDIATE_EXT, { 0.0000f, 0.0000f, -0.0290f } },
    { XR_HAND_JOINT_LITTLE_DISTAL_EXT, { 0.0000f, 0.0000f, -0.0183f } },
    { XR_HAND_JOINT_LITTLE_TIP_EXT, { 0.0000f, 0.0000f, -0.0220f } },
};

//! How many fingers the anatomy and the mechanism both carry.
inline constexpr std::size_t kHumanFingerCount = 5;
inline constexpr std::size_t kRigFingerCount = 5;

//! The five fingers, in OpenXR joint order (thumb first).
inline constexpr HumanFinger kHumanFingers[] = {
    { "thumb", 4, kThumbLandmarks }, { "index", 5, kIndexLandmarks },   { "middle", 5, kMiddleLandmarks },
    { "ring", 5, kRingLandmarks },   { "little", 5, kLittleLandmarks },
};

//! The mechanism's frame suffixes per finger, in chain order. ``mechanism`` names are
//! ``link_<Side>Skeleton<...>``. The first four entries are the revolute joints whose angles drive the
//! anatomy (for the thumb: ThumbBase, Thumb1..3). The fifth names the thumb's last revolute joint,
//! Thumb4, which the anatomy does not use; it is empty for the other fingers, whose fingertip frame is
//! named separately.
struct RigFinger
{
    const char* name;
    std::array<const char*, 5> links; //!< F1, F2, F3, F4, Thumb4 (thumb only)
};

//! Mechanism chain per finger (left; the right hand's names differ only in ``Left``/``Right``).
inline constexpr RigFinger kRigFingers[] = {
    { "thumb", { "ThumbBase", "Thumb1", "Thumb2", "Thumb3", "Thumb4" } },
    { "index", { "Index1", "Index2", "Index3", "Index4", "" } },
    { "middle", { "Middle1", "Middle2", "Middle3", "Middle4", "" } },
    { "ring", { "Ring1", "Ring2", "Ring3", "Ring4", "" } },
    { "little", { "Pinky1", "Pinky2", "Pinky3", "Pinky4", "" } },
};

//! Articulation read from one frame of the mechanism: per finger, the knuckle spread and the bends.
struct FingerAngles
{
    //! Knuckle spread, radians, positive towards the index finger.
    float spread = 0.0f;
    //! Bends along the chain, radians, positive towards the palm. Fingers: MCP, PIP, DIP. Thumb:
    //! CMC, MCP, IP.
    std::array<float, 3> bend{ 0.0f, 0.0f, 0.0f };
};

//! Angles of all five fingers plus whether the mechanism was present at all.
struct HandAngles
{
    std::array<FingerAngles, 5> fingers{};
    bool valid = false;
    //! Names the mechanism did resolve, for the one-time startup log.
    std::array<std::string, 5> metacarpal_frame;
};

//! Reads the mechanism's articulation out of one link message. False when the mechanism's root is
//! absent, i.e. that glove is not streaming.
HandAngles read_hand_angles(const tf2_msgs::msg::TFMessage& msg, Hand hand);

//! Builds the anatomical 26-joint skeleton for @p hand from an articulation reading, expressed in the
//! OpenXR wrist frame (the wrist is the identity pose). Every joint is filled and carries a unit
//! orientation, because OpenXR requires all 26 to be valid while a hand is active.
HandSkeleton build_anatomical_skeleton(const HandAngles& angles, Hand hand);

//! Root frame of the mechanism for @p hand, e.g. ``link_LeftSkeletonBase``.
const char* mechanism_root_frame(Hand hand);

} // namespace io_exo_glove_ros2
} // namespace plugins
