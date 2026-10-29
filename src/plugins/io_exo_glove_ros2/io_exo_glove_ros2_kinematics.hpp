// SPDX-FileCopyrightText: Copyright (c) 2026 IO. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "io_exo_glove_ros2_hand.hpp"

#include <openxr/openxr.h>
#include <tf2_msgs/msg/tf_message.hpp>

#include <array>
#include <cstddef>
#include <deque>
#include <string>

namespace plugins
{
namespace io_exo_glove_ros2
{

/*!
 * @brief The exoskeleton's fingertip motion, re-expressed as an anatomical OpenXR hand.
 *
 * The exoskeleton publishes the poses of its *linkage mechanism*: in the vendor's own URDF each
 * finger is a chain of four revolute links whose segments are 7.6 cm and 7.5 cm long, roughly twice a
 * human phalanx, and whose arm leaves the finger plane by 7 cm between two joints. Injecting those
 * frames as hand joints therefore yields a hand with a palm of a few millimetres and fingers twice a
 * palm long. What carries over is the *articulation*: the bend of the finger and the spread at the
 * knuckle. This module reads that articulation from the glove's own frames and drives a hand skeleton
 * whose landmarks come from the vendor's human-hand model instead.
 *
 * Flexion is derived from the **fingertip**, not from the linkage's joint angles. For each finger the
 * chord is the distance between the knuckle frame (``link_<Side>Skeleton<Finger>1``, or
 * ``link_<Side>SkeletonThumbBase`` for the thumb) and the vendor's own tip frame
 * (``<side>_<finger>_tip``); that chord can only shrink when the finger curls, so no session-specific
 * sign or gain can invert the hand. The glove's chord travel is smaller than the anatomical chain's
 * (the operator's half-closed hand used to read as open), so each finger first normalises its own
 * chord against a rolling window of the travel it has actually seen: see @c FingerChordGain. The
 * normalised chord is converted to one total bend of the anatomical chain (MCP/PIP/DIP with fixed
 * segment lengths) and split over the three joints with fixed weights; see @c kFingerBendWeights and
 * @c solve_total_bend. The bend's sign is the sign of the tip's palmward angle in the palm frame, so
 * a tip that is on the palm side always curls the hand towards the palm. The knuckle spread and the
 * thumb's mount still come from the linkage's own joint angles (they are not flexion), and the palm
 * frame is built from the knuckle row.
 *
 * The thumb also needs **opposition**: a chord shortening alone bends it in the palm plane, but
 * bringing the tip to the index is a swing across and out of that plane. The chord's lateral angle
 * around the palm normal, normalised over the same window, drives an extra CMC rotation about an axis
 * in the palm plane (@c kThumbOppositionAxis); see @c FingerAngles::opposition.
 *
 * Everything below is a constant derived from the two URDFs that ship with the vendor SDK
 * (`io_exotrans2hand_project_zenoh_22.04_x86_v2.1.4/src/exo_tf_pkg/description/`), so nothing here
 * depends on when a frame happens to arrive:
 *
 *  - ``blender_human_skeleton_v5.urdf`` gives the mechanism's chain: ``joint_<side>SkeletonBase`` is
 *    the fixed root, and each finger is ``joint_<F>1`` (the knuckle spread) followed by ``<F>2``,
 *    ``3``, ``4`` (the bends). The thumb has one more link: ``ThumbBase`` is a *revolute* mount,
 *    ``Thumb1`` lifts the thumb out of the palm, and ``Thumb2``..``Thumb4`` are its three bends.
 *    The first frame of a finger is the knuckle; the second is the first bend joint's origin only
 *    1.1 cm away along the knuckle row, not the next phalanx. The knuckle's spread axis is local
 *    ``+Z``, i.e. the palm normal: the four first-frame ``+Z`` axes agree to within 2 degrees and
 *    match the mechanism root's ``+Z``. The palm frame below therefore takes @e dorsal from those
 *    axes and @e across from the knuckle row, which is exactly the pair that stays live and rigid.
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
 * is needed at run time: only the tip chords and the spread cross over, and the wrist pose comes from
 * the wrist source.
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
//! knuckle spread (F1, or Thumb1 for the thumb, which *lifts* it out of the palm); the bends come
//! from the tip chord, so only F1 is read as an angle. @c parent is the frame the first joint hangs
//! off; the thumb's is the revolute ThumbBase, which is read too. @c tip is the vendor fingertip
//! frame suffix: ``<side>_<tip>_tip`` (the little finger's link is ``Pinky`` but its tip frame is
//! ``pinky``).
struct RigFinger
{
    const char* name;
    const char* tip;
    std::array<const char*, 4> links; //!< F1, F2, F3, F4 (only F1 is read as an angle)
    const char* parent;
};

//! Mechanism chain per finger (left; the right hand's names differ only in ``Left``/``Right``).
//! ``ThumbBase`` is the revolute mount whose child is ``Thumb1``, so the thumb's lift is ``Thumb1``
//! and its chord runs from the mount to the tip.
inline constexpr RigFinger kRigFingers[] = {
    { "thumb", "thumb", { "Thumb1", "Thumb2", "Thumb3", "Thumb4" }, "ThumbBase" },
    { "index", "index", { "Index1", "Index2", "Index3", "Index4" }, "" },
    { "middle", "middle", { "Middle1", "Middle2", "Middle3", "Middle4" }, "" },
    { "ring", "ring", { "Ring1", "Ring2", "Ring3", "Ring4" }, "" },
    { "little", "pinky", { "Pinky1", "Pinky2", "Pinky3", "Pinky4" }, "" },
};

//! Articulation read from one frame of the mechanism: per finger, the knuckle spread and the bends.
struct FingerAngles
{
    //! Fingers: knuckle spread, radians, positive towards the index finger. Thumb: the Thumb1 lift,
    //! positive out of the palm.
    float spread = 0.0f;
    //! Bends along the chain, radians, positive towards the palm. Fingers: MCP, PIP, DIP. Thumb:
    //! CMC, MCP, IP. Derived from the tip chord, not from the linkage's joint angles.
    std::array<float, 3> bend{ 0.0f, 0.0f, 0.0f };
    //! Thumb only: the revolute ThumbBase angle, radians, positive as the mechanism's own axis.
    float base = 0.0f;
    //! Thumb only: opposition of the CMC out of the palm plane, radians, 0 when the thumb is at
    //! rest and positive towards the palm/index. Driven by the lateral chord angle; the flexion
    //! solve above supplies the in-plane half of the same motion.
    float opposition = 0.0f;
};

//! Angles of all five fingers plus whether the whole hand was present.
struct HandAngles
{
    std::array<FingerAngles, 5> fingers{};
    //! Knuckle-to-tip chord per finger, metres, as read from the mechanism (before the model's
    //! length clamps). Diagnostics and the stale-wrist check use it.
    std::array<float, 5> chord_m{};
    //! Palmward angle of the chord per finger, radians, positive towards the palm. This is the sign
    //! the bend takes, so an inverted linkage cannot invert the injected hand.
    std::array<float, 5> palmward_rad{};
    bool valid = false;
};

//! One finger's tip chord in the palm frame.
struct TipChord
{
    //! Distance from the knuckle to the vendor tip frame, metres.
    float chord_m = 0.0f;
    //! Angle of the chord from the palm's distal axis towards the palm (negative = dorsal).
    float palmward_rad = 0.0f;
    //! Angle of the chord around the palm normal, from the distal axis towards the palm's across
    //! axis, radians, sign-normalised so both hands read the same: it falls as the thumb swings
    //! towards the index (that is the opposition the injected thumb needs). Fingers carry it too,
    //! but only the thumb uses it.
    float lateral_rad = 0.0f;
    bool valid = false;
};

//! All five fingers' chords from one link message, and whether the whole hand was present.
struct HandChords
{
    std::array<TipChord, kHumanFingerCount> fingers{};
    bool valid = false;
};

//! Fixed weights that split one total bend over a finger's three joints, in chain order
//! (MCP, PIP, DIP; the thumb's CMC, MCP, IP). The MCP leads, as it does in a real fist. The weights
//! are the same for the thumb's chain; see @c kThumbBendMaxRad for its smaller ranges.
inline constexpr std::array<float, 3> kFingerBendWeights = { 0.45f, 0.35f, 0.20f };

//! Human maxima the bend is clamped to, radians, in chain order: finger MCP 85 deg, PIP 95, DIP 70.
inline constexpr std::array<float, 3> kFingerBendMaxRad = {
    1.48353f,
    1.65806f,
    1.22173f,
};

//! The thumb's three joints are the CMC (35 deg), MCP (55) and IP (75); the same weights split its
//! total bend, and each joint is clamped to its own maximum.
inline constexpr std::array<float, 3> kThumbBendMaxRad = {
    0.61087f,
    0.95993f,
    1.30900f,
};

//! The anatomical chain one finger's chord is solved against: the three bone lengths from the MCP to
//! the tip (the thumb's from its mount), the bend weights and the joint maxima.
struct BendModel
{
    std::array<float, 3> segment_m;
    std::array<float, 3> weights;
    std::array<float, 3> max_rad;
};

//! Chain model for @p finger (0 = thumb), built from the landmark table's bone lengths. The thumb's
//! chord starts at the mechanism's ThumbBase and spans the CMC metacarpal too, so its first segment
//! is the CMC->MCP bone; a finger's chord starts at its MCP.
BendModel bend_model(std::size_t finger);

/*!
 * @brief Total bend of @p model's chain whose MCP-to-tip chord is @p chord_m, radians.
 *
 * The chain is planar: each joint bends the bones after it towards the palm by its share of the
 * total, and the chord is the distance from the first joint to the tip. A longer chord is a
 * straighter finger, so the chord falls monotonically as the total bend grows; the total is found by
 * bisection between 0 and the value at which every joint has reached its maximum. A chord longer
 * than the straight chain returns 0; one shorter than the fully curled chain returns that maximum
 * bend, so the result is always in range and bone lengths never change.
 */
float solve_total_bend(const BendModel& model, float chord_m);

//! @p total_bend split by @p model's weights, each joint clamped to its maximum, radians.
std::array<float, 3> distribute_bend(const BendModel& model, float total_bend);

//! Thumb opposition axis in the left anatomy's wrist frame, unit length. It lies in the palm plane,
//! across the hand; the thumb's flexion axis is the palm normal, so this is the second degree of
//! freedom the pinch needs. The angle is positive towards the palm/index (the chain is mirrored for
//! the right hand, so one axis serves both).
inline constexpr std::array<float, 3> kThumbOppositionAxis = { -1.0f, 0.0f, 0.0f };

//! CMC opposition at a full traverse of the learned lateral range, radians (about 69 deg).
inline constexpr float kThumbOppositionMaxRad = 1.20f;

//! Share of the opposition repeated at the thumb's MCP, radians (about 20 deg). The IP is left to
//! the flexion solve: opposition is a CMC/MCP motion in the human thumb.
inline constexpr float kThumbOppositionMcpMaxRad = 0.35f;

/*!
 * @brief Per-session, per-finger travel gain learned from the operator's own chord.
 *
 * The mechanism's chord travel is smaller than the anatomical chain's, so an absolute chord-to-bend
 * map leaves a half-closed glove looking open -- only a full fist reaches the maxima. Each finger
 * keeps a rolling window (@c kWindowS) of its chord and takes robust extents (the @c kPercentile and
 * 1 - it quantiles) from it. The extents are low-passed with @c kExtentTauS, so a window update
 * cannot step the injected pose. A window whose 5th..95th span is below @c kMinChordSpanM has not
 * seen a gesture; it is ignored and the previous extents kept, so a hand held still cannot make the
 * gain chase noise.
 *
 * The same window tracks the thumb chord's lateral angle for opposition, with @c
 * kMinLateralSpanRad as the guard. Until a lateral range has been learned, opposition is zero: the
 * absolute open lateral angle is not a rig constant and cannot be assumed.
 *
 * The normalised travel is mapped onto the fixed anatomical chain by @c read_hand_angles: the
 * longest chord seen is 0 deg and the shortest is every joint's human maximum.
 */
class FingerChordGain
{
public:
    //! Rolling-window length, seconds.
    static constexpr double kWindowS = 6.0;
    //! Lower quantile of the window taken as the short extent; the upper is 1 - this.
    static constexpr float kPercentile = 0.05f;
    //! A chord window narrower than this has not seen a gesture; metres.
    static constexpr float kMinChordSpanM = 0.005f;
    //! Time constant of the extent low-pass, seconds.
    static constexpr float kExtentTauS = 1.0f;
    //! Time constant that keeps the opposition's learned-gate transition continuous, seconds. The
    //! extents themselves still move with kExtentTauS.
    static constexpr float kOppositionTauS = 0.25f;
    //! A lateral window narrower than this has not seen an opposition; radians (about 5.7 deg).
    //! The lateral angle is a direction, not a length, so its guard is smaller than the chord's.
    static constexpr float kMinLateralSpanRad = 0.10f;
    //! Readings needed before a quantile means anything.
    static constexpr std::size_t kMinSamples = 8;

    /*!
     * @param model The anatomical chain this finger solves against; its straight and fully curled
     *        chords seed the extents, so a hand held still from the start still maps sensibly.
     * @param thumb Whether to track the lateral angle for opposition (the thumb only).
     */
    FingerChordGain(const BendModel& model, bool thumb);

    //! Adds one reading at monotonic @p now_s. A rewind clears the window.
    void update(float chord_m, float lateral_rad, double now_s);

    //! @p chord_m's normalised travel against the smoothed extents, in [0, 1]: 0 at the longest
    //! chord seen (or longer), 1 at the shortest (or shorter).
    float fraction(float chord_m) const;

    //! The thumb's smoothed opposition in [0, 1]: 0 at the widest lateral angle seen, 1 at the
    //! narrowest. A faster second filter on top of the extents keeps the 0-to-learned transition
    //! continuous; it stays 0 until a lateral range has been learned.
    float opposition() const
    {
        return m_opposition;
    }

    float chord_max() const
    {
        return m_chord_max;
    }

    float chord_min() const
    {
        return m_chord_min;
    }

    float lateral_max() const
    {
        return m_lateral_max;
    }

    float lateral_min() const
    {
        return m_lateral_min;
    }

    bool lateral_learned() const
    {
        return m_lateral_learned;
    }

private:
    struct Sample
    {
        double t;
        float chord;
        float lateral;
    };

    float percentile(float p, bool lateral) const;

    BendModel m_model;
    bool m_thumb = false;
    std::deque<Sample> m_samples;
    bool m_has_last = false;
    double m_last_s = 0.0;

    //! Smoothed extents actually used, and the window extents they are filtered towards.
    float m_chord_max;
    float m_chord_min;
    float m_target_chord_max;
    float m_target_chord_min;

    float m_lateral_max = 0.0f;
    float m_lateral_min = 0.0f;
    float m_target_lateral_max = 0.0f;
    float m_target_lateral_min = 0.0f;
    bool m_lateral_learned = false;
    bool m_has_lateral = false;
    float m_opposition = 0.0f;
};

//! One hand's five gains, one per finger in OpenXR order (thumb first).
struct HandChordGain
{
    HandChordGain();

    std::array<FingerChordGain, kHumanFingerCount> fingers;
};

/*!
 * @brief Reads the five fingertips' chords in the glove's palm frame.
 *
 * The palm frame is built from the knuckle row: @e across is index knuckle - little knuckle (mirrored
 * for the right hand so it always points anatomically from the little side towards the index), and
 * @e dorsal is the mean of the four first-frame local ``+Z`` axes -- the URDF's knuckle-spread axis,
 * i.e. the palm normal. @e distal is ``dorsal x across``. (The linkage's second frame is the first
 * bend joint's origin only 1.1 cm from the first along the knuckle row, so "knuckle -> next frame" is
 * not a distal direction on this rig; the spread axis is the reliable live palm normal.)
 *
 * Each finger's chord is ``|<side>_<finger>_tip - knuckle|`` and its palmward angle is measured from
 * @e distal towards the palm, so a flexed tip is positive on both hands. Every frame used must carry
 * the same ``header.frame_id``: translations in different parents cannot be subtracted. A missing
 * frame, a frame mismatch or a degenerate palm frame makes the whole hand invalid, exactly like a
 * finger missing its linkage: half a hand must not be injected.
 */
HandChords read_tip_chords(const tf2_msgs::msg::TFMessage& msg, Hand hand);

//! Reads the whole hand's articulation: the spread and thumb mount from the linkage angles, the
//! three bends of every finger from its tip chord (see @c read_tip_chords), and the thumb's
//! opposition from the chord's lateral angle. @p gain carries the per-session travel learned so far
//! (one instance per hand, kept across messages) and @p now_s is the monotonic arrival time; an
//! invalid frame leaves the gain untouched. False @c valid when any frame or finger is missing, or
//! the tip frames are not in one frame.
HandAngles read_hand_angles(const tf2_msgs::msg::TFMessage& msg, Hand hand, HandChordGain& gain, double now_s);

//! Builds the anatomical 26-joint skeleton for @p hand from an articulation reading, expressed in the
//! OpenXR wrist frame (the wrist is the identity pose). Every joint is filled and carries a unit
//! orientation, because OpenXR requires all 26 to be valid while a hand is active. The anatomy is
//! built in the left hand's frame and mirrored in X for the right hand.
HandSkeleton build_anatomical_skeleton(const HandAngles& angles, Hand hand);

//! Distance between the MIDDLE_PROXIMAL (12) and MIDDLE_TIP (15) joints of @p skeleton, in metres.
//! The skeleton is wrist-relative, so this is a frame-free scalar for the injected middle finger.
float middle_joint_span(const HandSkeleton& skeleton);

/*!
 * @brief Detects a wrist source that has stopped updating while the fingers keep moving.
 *
 * A live optical wrist changes every frame; a stalled one repeats bit-identically while the glove's
 * tip chords keep changing. The guard counts frames whose wrist pose equals the previous one (all
 * seven floats) while any chord moved more than @c kChordMotionM; after @c kFrozenHoldS of that it
 * reports the wrist frozen, once, and asks the caller to switch to a controller (when one is
 * available) or to withhold the hand for those frames. Any change in the wrist pose clears the
 * state, so a genuinely live source can never latch. Before the hold elapses the frame is still
 * pushed: a source that updates more slowly than the glove (a valid-but-untracked optical pose, for
 * example) repeats a pose for a frame or two without being stale, and the caller's smoothing already
 * gives those frames distinct wrist poses.
 */
class FrozenWristGuard
{
public:
    //! Chord motion that counts as "the fingers moved", metres.
    static constexpr float kChordMotionM = 0.002f;
    //! Continuous duplicate-wrist time before the source is called frozen, seconds.
    static constexpr double kFrozenHoldS = 0.2;

    //! What the caller should do with this frame.
    enum class Action
    {
        Push, //!< Inject the hand normally.
        Suppress, //!< Do not push the hand this frame: the wrist is stale.
        SwitchToController, //!< Use the controller wrist source for this frame.
    };

    struct Result
    {
        Action action = Action::Push;
        bool newly_frozen = false;
        bool thawed = false;
    };

    //! Feeds one frame's raw (pre-smoothing) wrist pose and tip chords. @p now_s is monotonic
    //! seconds. @p controller_available says whether the caller has a controller wrist source that
    //! can take over; when it is false a frozen wrist is suppressed instead.
    Result update(const XrPosef& pose,
                  const std::array<float, kHumanFingerCount>& chords_m,
                  bool controller_available,
                  double now_s);

    bool frozen() const
    {
        return m_frozen;
    }

    //! Forgets the previous pose and the freeze state, e.g. when the glove or the source drops out.
    void reset();

private:
    Action frozen_action(bool controller_available) const;

    bool m_has_pose = false;
    bool m_frozen = false;
    XrPosef m_pose{ { 0.0f, 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, 0.0f } };
    std::array<float, kHumanFingerCount> m_chords{};
    bool m_has_duplicate = false;
    double m_first_duplicate_s = 0.0;
};

} // namespace io_exo_glove_ros2
} // namespace plugins
