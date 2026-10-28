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
 * a hand skeleton whose landmarks come from the vendor's human-hand model instead. A linkage bend is
 * not a joint angle -- the mechanism moves roughly half as far as the finger -- so each one is
 * calibrated against the recording's open and fist plateaus and mapped onto its human range; see the
 * table and ``read_hand_angles`` in the .cpp.
 *
 * Everything below is a constant derived from the two URDFs that ship with the vendor SDK
 * (`io_exotrans2hand_project_zenoh_22.04_x86_v2.1.4/src/exo_tf_pkg/description/`), so nothing here
 * depends on when a frame happens to arrive:
 *
 *  - ``blender_human_skeleton_v5.urdf`` gives the mechanism's chain: ``joint_<side>SkeletonBase`` is
 *    the fixed root, and each finger is ``joint_<F>1`` (the knuckle spread) followed by ``joint_<F>2``,
 *    ``3``, ``4`` (the bends). The thumb has one more link: ``ThumbBase`` is a *revolute* mount,
 *    ``Thumb1`` lifts the thumb out of the palm, and ``Thumb2``..``Thumb4`` are its three bends.
 *    Every joint carries an ``<origin rpy>`` rest rotation and a revolute ``<axis>``, so the published
 *    child frame is parent * R_rest(rpy) * R(axis, theta); the joint's absolute angle follows from the
 *    two frames' quaternions (see the table in the .cpp).
 *  - ``blender_human.urdf`` gives the anatomy: the wrist frame is ``link_<side>Hand_Y``, each finger
 *    starts at ``joint_<side>InHand<F>`` (the metacarpal base) and continues through
 *    ``joint_<side>Hand<F>1`` (MCP), ``2`` (PIP) and ``3`` (DIP). The exported model leaves every
 *    hand joint's ``rpy`` at zero, so its metacarpals all run straight down -Z and its thumb hangs
 *    parallel to the fingers; the A-pose fan and the thumb's own frame are therefore applied here
 *    (``kFingerFanRpy`` and ``kThumbRestRpy``), keeping the model's CMC positions and bone lengths.
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
//! Each offset is parent-relative, in the frame after the previous joint; the metacarpal offset is
//! therefore the one the fan rest rotation at the CMC turns outwards.
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

//! Metacarpal A-pose rest rotation per non-thumb finger, in URDF rpy form, applied at the CMC before its
//! offset. The human export's hand-joint rpy are all zero, which leaves the knuckles 31 mm apart and
//! rolls the finger plane 13 deg about -Z; this fan puts them on the 56 mm line an adult hand has,
//! level with the wrist. Derived and checked by ``io_exo_glove_ros2_kinematics_tables.py``.
inline constexpr std::array<std::array<float, 3>, kRigFingerCount - 1> kFingerFanRpy = { {
    { 0.08180f, -0.31673f, -0.01307f }, // index
    { 0.02648f, -0.20165f, -0.00268f }, // middle
    { -0.00700f, -0.01531f, 0.00005f }, // ring
    { -0.07024f, 0.16102f, -0.00567f }, // little
} };

//! The thumb's own rest frame: the mechanism's ThumbBase rest yaw about the palm normal, then the
//! Thumb1 lift. Without it the thumb is a parallel fifth finger. In URDF rpy form.
inline constexpr std::array<float, 3> kThumbRestRpy = { 0.79435f, -0.90676f, -0.35566f };
//! Thumb revolute axes in the OpenXR wrist frame, mapped from the v5 base frame. ``ThumbBase`` and
//! ``Thumb1`` lie in the palm plane (the lift); ``Thumb2``..``Thumb4`` are the palm normal (flexion
//! in the palm plane). The flex axis sign is data-derived: with +Y the model thumb tip moved away
//! from the index on both recordings, while -Y moved it with the vendor ``<side>_thumb_tip`` frame
//! (the harness's recorded excursion reaches cosine 0.98-0.99 after the sign flip). The calibrated
//! bend stays positive; only the model's sense of flexion is flipped.
inline constexpr std::array<float, 3> kThumbBaseAxis = { 0.54f, 0.00f, -0.84f };
inline constexpr std::array<float, 3> kThumbLiftAxis = { 0.84f, 0.00f, 0.54f };
inline constexpr std::array<float, 3> kThumbFlexAxis = { 0.00f, -1.00f, 0.00f };

//! The mechanism's frame suffixes per finger, in chain order. ``mechanism`` names are
//! ``link_<Side>Skeleton<...>``. The four entries are the revolute joints whose angles drive the
//! anatomy: the knuckle spread (F1, or Thumb1 for the thumb, which *lifts* it out of the palm) then
//! the three bends (F2..F4, or Thumb2..Thumb4). @c parent is the frame the first joint hangs off;
//! the thumb's is the revolute ThumbBase, which is read too.
struct RigFinger
{
    const char* name;
    std::array<const char*, 4> links; //!< F1, F2, F3, F4
    const char* parent;
};

//! Mechanism chain per finger (left; the right hand's names differ only in ``Left``/``Right``).
//! ``ThumbBase`` is the revolute mount whose child is ``Thumb1``, so the thumb's lift is ``Thumb1``
//! and the three bends are ``Thumb2``, ``Thumb3`` and ``Thumb4`` -- the same F1..F4 pattern as the
//! fingers, shifted one link out, plus the mount's own angle.
inline constexpr RigFinger kRigFingers[] = {
    { "thumb", { "Thumb1", "Thumb2", "Thumb3", "Thumb4" }, "ThumbBase" },
    { "index", { "Index1", "Index2", "Index3", "Index4" }, "" },
    { "middle", { "Middle1", "Middle2", "Middle3", "Middle4" }, "" },
    { "ring", { "Ring1", "Ring2", "Ring3", "Ring4" }, "" },
    { "little", { "Pinky1", "Pinky2", "Pinky3", "Pinky4" }, "" },
};

//! Articulation read from one frame of the mechanism: per finger, the knuckle spread and the bends.
struct FingerAngles
{
    //! Fingers: knuckle spread, radians, positive towards the index finger. Thumb: the Thumb1 lift,
    //! positive out of the palm.
    float spread = 0.0f;
    //! Bends along the chain, radians, positive towards the palm. Fingers: MCP, PIP, DIP. Thumb:
    //! CMC, MCP, IP (mechanism Thumb2, Thumb3, Thumb4).
    std::array<float, 3> bend{ 0.0f, 0.0f, 0.0f };
    //! Thumb only: the revolute ThumbBase angle, radians, positive as the mechanism's own axis.
    float base = 0.0f;
};

//! Angles of all five fingers plus whether the whole hand was present.
struct HandAngles
{
    std::array<FingerAngles, 5> fingers{};
    bool valid = false;
};

/*!
 * @brief Previous angle of every mechanism joint per hand, so a continuously turning revolute joint
 * is not wrapped to (-pi, pi].
 *
 * ``signed_angle_about`` returns the shortest signed angle, which is right for one message but jumps
 * by 2*pi when the joint crosses the boundary. The left ThumbBase motivated this: the v5 left mount
 * lacks the pi roll the right's has, so its physical rest is a half turn from the URDF rest and the
 * open reading sits near +pi; flexion then crosses the boundary and the raw series jumps a full turn
 * even though the mount turned smoothly. The derived rest constant below moves the open reference to
 * zero, and this keeps any remaining crossing continuous: the last value per hand and joint is kept
 * and +-2*pi is added until the new reading is within pi of it; the first sample initialises. Call
 * @c reset when a glove drops out, so the first frame after a gap re-initialises rather than
 * unwrapping against a stale pose.
 */
struct JointAngleUnwrapper
{
    //! Unwraps @p value for one mechanism joint against its previous value for @p hand.
    float unwrap(Hand hand, std::size_t finger, std::size_t slot, float value);
    //! Same for ThumbBase, which is a revolute joint outside the four-slot finger table.
    float unwrap_thumb_base(Hand hand, float value);
    //! Forgets both hands' previous readings.
    void reset();

private:
    std::array<std::array<std::array<float, 4>, kRigFingerCount>, 2> m_joint{};
    std::array<std::array<std::array<bool, 4>, kRigFingerCount>, 2> m_joint_initialized{};
    std::array<float, 2> m_thumb_base{};
    std::array<bool, 2> m_thumb_base_initialized{};
};

//! Reads the mechanism's articulation out of one link message. Valid only when the root and every
//! finger's four link frames are present: a hand missing any finger is invalid, because that finger
//! would otherwise sit at its open zero and be injected as an open finger. False when the mechanism
//! is absent, i.e. that glove is not streaming. @p unwrapper, when given, carries the previous
//! angles across messages so a joint that crosses +-pi stays continuous; the plugin keeps one per
//! hand.
//!
//! @p flip_bends, when true, injects every bend with the opposite sense -- the runtime correction for
//! a glove whose bend direction disagrees with the model. The signed raw delta (the calibrated
//! ``sign * (raw - zero)``) is multiplied by -1 before the gain mapping and the dead-band clamp, so
//! the open hand stays open and its slight extension does not eat the flipped fist's travel. The
//! spread and the thumb's mount are not bends and are left alone.
HandAngles read_hand_angles(const tf2_msgs::msg::TFMessage& msg,
                            Hand hand,
                            JointAngleUnwrapper* unwrapper = nullptr,
                            bool flip_bends = false);

//! Builds the anatomical 26-joint skeleton for @p hand from an articulation reading, expressed in the
//! OpenXR wrist frame (the wrist is the identity pose). Every joint is filled and carries a unit
//! orientation, because OpenXR requires all 26 to be valid while a hand is active. The anatomy is
//! built in the left hand's frame and mirrored in X for the right hand.
HandSkeleton build_anatomical_skeleton(const HandAngles& angles, Hand hand);

//! Distance between the MIDDLE_PROXIMAL (12) and MIDDLE_TIP (15) joints of @p skeleton, in metres.
//! The skeleton is wrist-relative, so this is a frame-free scalar for the injected middle finger.
float middle_joint_span(const HandSkeleton& skeleton);

//! Distance between the mechanism's ``link_<Side>SkeletonMiddle1`` and ``<side>_middle_tip``
//! translations, in metres, both published in the same frame by the driver. False when either frame
//! is absent or they are expressed in different frames, in which case no span can be measured.
bool mechanism_middle_span(const tf2_msgs::msg::TFMessage& msg, Hand hand, float& span_m);

//! Size of the rolling window the bend-direction check keeps per hand, samples at the hand tick.
inline constexpr std::size_t kBendDirectionWindow = 64;
//! Samples required before a correlation is computed.
inline constexpr std::size_t kBendDirectionMinSamples = 40;
//! Swing (max - min) one of the two series must exceed before it is correlated, in metres. A flat
//! hand carries no direction information and must not flip anything.
inline constexpr float kBendDirectionSwingM = 0.010f;
//! A correlation below -kBendDirectionCorrelation flips the bends; above + it flips them back. In
//! between is inconclusive and keeps the current decision.
inline constexpr float kBendDirectionCorrelation = 0.5f;
//! Minimum time between two decisions, in seconds, so a flipping hypothesis cannot chatter.
inline constexpr double kBendDirectionPeriodS = 2.0;

/*!
 * @brief Decides, from data per hand, whether the glove's bend direction matches the injected model.
 *
 * The glove's sign is not reliable across sessions: the same finger code and driver session can
 * publish the mechanism's quaternions with the opposite sense, which injects an inverted hand (a
 * flat hand looks curled and a fist looks extended). This keeps a rolling window of two frame-free
 * scalars sampled on every hand tick -- the nominal model's MIDDLE_PROXIMAL -> MIDDLE_TIP distance in
 * the wrist frame, and the mechanism's own link_<Side>SkeletonMiddle1 -> <side>_middle_tip distance --
 * and correlates them. Both shrink together when the directions agree and move apart when they do
 * not, without comparing positions across frames.
 *
 * The model series is built with the nominal (unflipped) bends, so a decision never feeds back into
 * its own input: negative correlation means "glove opposite to model" -> flip; positive correlation
 * means "glove matches model" -> no flip, so a later positive correlation flips a hand back. Windows
 * with fewer than kBendDirectionMinSamples samples, too little swing, or a correlation inside
 * +-kBendDirectionCorrelation are inconclusive and leave the decision as it was (no flip at start).
 */
class BendDirectionChecker
{
public:
    //! What a sample changed, for the caller to log.
    enum class Change
    {
        None,
        Flipped,
        Unflipped,
    };

    //! Feeds one hand tick's pair of spans and, when the window is ready and the period has elapsed,
    //! re-decides. @p now_s is a monotonic clock in seconds.
    Change sample(float model_span_m, float glove_span_m, double now_s);

    //! True while every bend of this hand is injected with the opposite sense.
    bool flipped() const
    {
        return m_flipped;
    }

    //! Pearson correlation of the current window, or 0 until one has been computed.
    float correlation() const
    {
        return m_correlation;
    }

    //! Samples currently in the window.
    std::size_t samples() const
    {
        return m_count;
    }

    //! Drops the window and its timing, e.g. when the glove drops out, but keeps the decision.
    void clear_window();

private:
    std::array<float, kBendDirectionWindow> m_model{};
    std::array<float, kBendDirectionWindow> m_glove{};
    std::size_t m_count = 0;
    std::size_t m_next = 0;
    bool m_flipped = false;
    bool m_evaluated = false;
    double m_last_evaluation_s = 0.0;
    float m_correlation = 0.0f;
};

//! Root frame of the mechanism for @p hand, e.g. ``link_LeftSkeletonBase``.
const char* mechanism_root_frame(Hand hand);

} // namespace io_exo_glove_ros2
} // namespace plugins
