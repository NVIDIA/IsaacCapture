// SPDX-FileCopyrightText: Copyright (c) 2026 IO. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "io_exo_glove_ros2_hand.hpp"

#include <oxr_utils/math.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>
#include <vector>

namespace plugins
{
namespace io_exo_glove_ros2
{

/*
 * The injected hand is an anatomical chain of fixed human bone lengths driven by the exoskeleton's
 * own mechanism joint angles (r31), with a per-session calibration captured once at startup.
 *
 * Structure, after the reference hand-identification project (a human hand URDF with fixed bone
 * lengths, a weighted solve to the device's measured targets, joint limits and a straightness prior,
 * plus an installation calibration fitted once per rig):
 *
 *  - Fixed bone lengths: the human model's phalanx lengths (MCP -> PIP -> DIP -> tip, and
 *    CMC -> MCP -> IP -> tip for the thumb). The chain cannot stretch.
 *  - Measured target: each finger's bend comes from its mechanism MCP and DIP absolute angles
 *    (`theta = signed_angle(conj(R_rest) * conj(R_parent) * R_child, axis)` with the v5 URDF rest
 *    rotations and axes), normalised against the session's open-pose angle and the baked fist angle.
 *    The tip direction still comes from the glove's own tip frame (for the residual and the thumb's
 *    aim). The chord is the fallback driver for a reference written by r30 or earlier.
 *  - Joint limits: MCP/PIP/DIP 85/95/70 deg, thumb CMC/MCP/IP 35/55/75 deg.
 *  - Straightness prior: the total bend is split 0.45/0.35/0.20 (proximal first) and each joint is
 *    clamped to its own limit, so the chain's own chord is exactly the mapped radius and the shape is
 *    a human fist split rather than a solver artefact. All five digits use it, so the four fingers
 *    read 0/0/0 open and 85/95/70 at the fist by construction, and the thumb (inverted travel, chord
 *    grows as it closes) reads 0 open and 35/55/75 closed. There is no iterative solve and no warm
 *    start: the output is a function of this frame's chord and direction only.
 *  - One-time calibration: the tip-frame orientation is NOT a rig constant - the same finger was
 *    measured 147 deg from the human axis in one driver session and 54 deg in another, with the base
 *    orientation identical. So the first frames after the plugin starts record, per hand and digit,
 *    the open chord and the rotation that takes the device's open tip direction onto the digit's
 *    human rest axis. That rotation is then applied to every live device direction. Nothing is
 *    learned after the window.
 *  - Persistent reference: those measurements (including the open-pose angle of all 20 mechanism
 *    joints, the opening/tightest thumb-to-finger gaps and the r35 open deadband) can be written to
 *    the config instead, so the operator calibrates once. A complete block skips the window
 *    entirely and drives the hand from its first frame; an r31 four-section block still loads with
 *    deadband 0 (the old map); a three-section block runs the chord driver; a missing or invalid
 *    block falls back to the window above. A loaded reference is authoritative and does not widen;
 *    a captured one still widens. The block format lives in README.md and in this file's parser.
 *
 * The landmarks, the knuckle spread and the thumb mount are unchanged. The shipped mechanism URDF
 * gives every one of its 42 revolute joints the same full-turn limit (-3.14..3.14), so those limits
 * are not a hand's travel and are not used; the open end is the session's measured angle and the
 * closed end is the baked kAngleClosed fist table.
 *
 * The chain itself is unchanged: human phalanx lengths, PALM and the four metacarpals on the human
 * row, `-Z` along every bone and `+Y` the back of the hand, the right hand built as the left's X
 * mirror. Fingers are anchored at the mechanism's knuckles; the thumb at its CMC. The skeleton stays
 * wrist-relative in the glove's `<side>_hand` frame, so BASE's `wrist_source`/`wrist_alignment` and
 * placement are unchanged.
 *
 * The hot path uses fixed arrays only and allocates nothing. The session window is 40 frames of
 * fixed storage per hand; closing it is a bounded sort over the last half of those frames, once.
 * Until the window closes every digit is held open (the straight chain): the open reference is what
 * makes a chord meaningful, and guessing from a stale table during the window is exactly the
 * failure this replaces. A window that cannot be satisfied by its deadline falls back to the baked
 * table with a warning - it never blocks the hand from being injected.
 */

namespace
{

//! Mechanism link a landmark rides or a chain frame is named after.
enum class Link : std::uint8_t
{
    Base,
    ThumbBase,
    Thumb2,
    Thumb3,
    Index1,
    Index2,
    Index3,
    Middle1,
    Middle2,
    Middle3,
    Ring1,
    Ring2,
    Ring3,
    Pinky1,
    Pinky2,
    Pinky3,
    ThumbTip,
    IndexTip,
    MiddleTip,
    RingTip,
    PinkyTip,
    Count,
};

constexpr const char* kLeftLinkNames[] = {
    "link_LeftSkeletonBase",    "link_LeftSkeletonThumbBase",
    "link_LeftSkeletonThumb2",  "link_LeftSkeletonThumb3",
    "link_LeftSkeletonIndex1",  "link_LeftSkeletonIndex2",
    "link_LeftSkeletonIndex3",  "link_LeftSkeletonMiddle1",
    "link_LeftSkeletonMiddle2", "link_LeftSkeletonMiddle3",
    "link_LeftSkeletonRing1",   "link_LeftSkeletonRing2",
    "link_LeftSkeletonRing3",   "link_LeftSkeletonPinky1",
    "link_LeftSkeletonPinky2",  "link_LeftSkeletonPinky3",
    "left_thumb_tip",           "left_index_tip",
    "left_middle_tip",          "left_ring_tip",
    "left_pinky_tip",
};

constexpr const char* kRightLinkNames[] = {
    "link_RightSkeletonBase",    "link_RightSkeletonThumbBase",
    "link_RightSkeletonThumb2",  "link_RightSkeletonThumb3",
    "link_RightSkeletonIndex1",  "link_RightSkeletonIndex2",
    "link_RightSkeletonIndex3",  "link_RightSkeletonMiddle1",
    "link_RightSkeletonMiddle2", "link_RightSkeletonMiddle3",
    "link_RightSkeletonRing1",   "link_RightSkeletonRing2",
    "link_RightSkeletonRing3",   "link_RightSkeletonPinky1",
    "link_RightSkeletonPinky2",  "link_RightSkeletonPinky3",
    "right_thumb_tip",           "right_index_tip",
    "right_middle_tip",          "right_ring_tip",
    "right_pinky_tip",
};

static_assert(sizeof(kLeftLinkNames) / sizeof(kLeftLinkNames[0]) == static_cast<std::size_t>(Link::Count),
              "one name per mechanism link");
static_assert(sizeof(kRightLinkNames) / sizeof(kRightLinkNames[0]) == static_cast<std::size_t>(Link::Count),
              "one name per mechanism link");

constexpr std::size_t kSlotCount = 4;

//! The revolute chain per finger, in the URDF's parent-to-child order. Slot 0 is the knuckle
//! spread (the thumb's mount); slots 1..3 are MCP/PIP/DIP (thumb CMC/MCP/IP).
constexpr const char* kMechanismFrames[2][kFingerCount][kSlotCount] = {
    { // left
      { "link_LeftSkeletonThumbBase", "link_LeftSkeletonThumb1", "link_LeftSkeletonThumb2", "link_LeftSkeletonThumb3" },
      { "link_LeftSkeletonIndex1", "link_LeftSkeletonIndex2", "link_LeftSkeletonIndex3", "link_LeftSkeletonIndex4" },
      { "link_LeftSkeletonMiddle1", "link_LeftSkeletonMiddle2", "link_LeftSkeletonMiddle3", "link_LeftSkeletonMiddle4" },
      { "link_LeftSkeletonRing1", "link_LeftSkeletonRing2", "link_LeftSkeletonRing3", "link_LeftSkeletonRing4" },
      { "link_LeftSkeletonPinky1", "link_LeftSkeletonPinky2", "link_LeftSkeletonPinky3", "link_LeftSkeletonPinky4" } },
    { // right
      { "link_RightSkeletonThumbBase", "link_RightSkeletonThumb1", "link_RightSkeletonThumb2", "link_RightSkeletonThumb3" },
      { "link_RightSkeletonIndex1", "link_RightSkeletonIndex2", "link_RightSkeletonIndex3", "link_RightSkeletonIndex4" },
      { "link_RightSkeletonMiddle1", "link_RightSkeletonMiddle2", "link_RightSkeletonMiddle3",
        "link_RightSkeletonMiddle4" },
      { "link_RightSkeletonRing1", "link_RightSkeletonRing2", "link_RightSkeletonRing3", "link_RightSkeletonRing4" },
      { "link_RightSkeletonPinky1", "link_RightSkeletonPinky2", "link_RightSkeletonPinky3", "link_RightSkeletonPinky4" } },
};

//! One mechanism joint: the URDF rest rotation and revolute axis, the baked travel endpoints, and
//! the human maximum it maps onto. All angles are radians; `open` and `closed` are constants, not
//! session state. The spread uses these; the finger/thumb bends are driven by the chord extents.
struct JointSpec
{
    std::array<float, 3> rest_rpy;
    std::array<float, 3> axis;
    float open;
    float closed;
    float human_max;
};

constexpr JointSpec kJointSpecs[2][kFingerCount][kSlotCount] = {
    // left
    {
        {
            // thumb
            { { 0.00000000f, 0.00000000f, 0.99442000f },
              { 1.00000000f, 0.00000000f, 0.00000000f },
              0.02827713f,
              -0.59990102f,
              0.43633231f },
            { { 1.57080000f, 1.57080000f, 0.00000000f },
              { 0.00000000f, 0.00000000f, 1.00000000f },
              -0.93498785f,
              -0.29863674f,
              0.61086524f },
            { { -1.30900000f, 0.00000000f, 0.00000000f },
              { 1.00000000f, 0.00000000f, 0.00000000f },
              0.36756274f,
              0.69573218f,
              0.95993109f },
            { { -0.81433000f, 0.00000000f, 0.00000000f },
              { 1.00000000f, 0.00000000f, 0.00000000f },
              -1.17372114f,
              -2.22547084f,
              1.30899694f },
        },
        {
            // index
            { { 0.00000000f, 0.26180000f, 0.11990000f },
              { 0.00000000f, 0.00000000f, 1.00000000f },
              -0.41471726f,
              -0.11477739f,
              0.43633231f },
            { { -0.75647000f, 0.00000000f, 0.00000000f },
              { 1.00000000f, 0.00000000f, 0.00000000f },
              0.02501087f,
              -1.06962330f,
              1.48352986f },
            { { 0.46849000f, 0.00000000f, 0.00000000f },
              { 1.00000000f, 0.00000000f, 0.00000000f },
              -1.84938771f,
              -2.43058998f,
              1.65806279f },
            { { 1.22170000f, 0.00000000f, 0.00000000f },
              { 1.00000000f, 0.00000000f, 0.00000000f },
              0.24336880f,
              -1.36814577f,
              1.22173048f },
        },
        {
            // middle
            { { 0.00000000f, 0.00000000f, 0.11990000f },
              { 0.00000000f, 0.00000000f, 1.00000000f },
              -0.10080153f,
              0.03065192f,
              0.43633231f },
            { { -0.75647000f, 0.00000000f, 0.00000000f },
              { 1.00000000f, 0.00000000f, 0.00000000f },
              -0.01796378f,
              -1.18984253f,
              1.48352986f },
            { { 0.46849000f, 0.00000000f, 0.00000000f },
              { 1.00000000f, 0.00000000f, 0.00000000f },
              -1.70903832f,
              -2.33036834f,
              1.65806279f },
            { { 1.22170000f, 0.00000000f, 0.00000000f },
              { 1.00000000f, 0.00000000f, 0.00000000f },
              0.23020176f,
              -1.74089128f,
              1.22173048f },
        },
        {
            // ring
            { { 0.00000000f, 0.00000000f, 0.11990000f },
              { 0.00000000f, 0.00000000f, 1.00000000f },
              -0.11112601f,
              0.13872233f,
              0.43633231f },
            { { -0.75647000f, 0.00000000f, 0.00000000f },
              { 1.00000000f, 0.00000000f, 0.00000000f },
              0.27333722f,
              -1.19147712f,
              1.48352986f },
            { { 0.46849000f, 0.00000000f, 0.00000000f },
              { 1.00000000f, 0.00000000f, 0.00000000f },
              -1.84640223f,
              -2.24301035f,
              1.65806279f },
            { { 1.22170000f, 0.00000000f, 0.00000000f },
              { 1.00000000f, 0.00000000f, 0.00000000f },
              0.29145634f,
              -1.51670248f,
              1.22173048f },
        },
        {
            // little
            { { 0.00000000f, -0.17453000f, 0.11990000f },
              { 0.00000000f, 0.00000000f, 1.00000000f },
              -0.00692416f,
              0.16527922f,
              0.43633231f },
            { { -0.75647000f, 0.00000000f, 0.00000000f },
              { 1.00000000f, 0.00000000f, 0.00000000f },
              0.49950856f,
              -1.13933212f,
              1.48352986f },
            { { 0.46849000f, 0.00000000f, 0.00000000f },
              { 1.00000000f, 0.00000000f, 0.00000000f },
              -2.24201388f,
              -2.56804067f,
              1.65806279f },
            { { 1.22170000f, 0.00000000f, 0.00000000f },
              { 1.00000000f, 0.00000000f, 0.00000000f },
              0.58794305f,
              -1.27511150f,
              1.22173048f },
        },
    },
    // right
    {
        {
            // thumb
            { { 3.14160000f, 0.00000000f, 2.14720000f },
              { 1.00000000f, 0.00000000f, 0.00000000f },
              -0.06417305f,
              0.35971835f,
              0.43633231f },
            { { 1.57080000f, 1.57080000f, 0.00000000f },
              { 0.00000000f, 0.00000000f, 1.00000000f },
              0.85516760f,
              0.16627850f,
              0.61086524f },
            { { -1.30900000f, 0.00000000f, 0.00000000f },
              { 1.00000000f, 0.00000000f, 0.00000000f },
              0.46691422f,
              0.99619085f,
              0.95993109f },
            { { -0.81433000f, 0.00000000f, 0.00000000f },
              { 1.00000000f, 0.00000000f, 0.00000000f },
              -1.14881260f,
              -2.22100574f,
              1.30899694f },
        },
        {
            // index
            { { 0.00000000f, -0.26180000f, -0.11990000f },
              { 0.00000000f, 0.00000000f, 1.00000000f },
              0.38964669f,
              0.21224971f,
              0.43633231f },
            { { -0.75647000f, 0.00000000f, 0.00000000f },
              { 1.00000000f, 0.00000000f, 0.00000000f },
              -0.17666394f,
              1.04379427f,
              1.48352986f },
            { { 0.46849000f, 0.00000000f, 0.00000000f },
              { 1.00000000f, 0.00000000f, 0.00000000f },
              -1.60939557f,
              -2.09337169f,
              1.65806279f },
            { { 1.22170000f, 0.00000000f, 0.00000000f },
              { 1.00000000f, 0.00000000f, 0.00000000f },
              -0.03537232f,
              -1.03480989f,
              1.22173048f },
        },
        {
            // middle
            { { 0.00000000f, 0.00000000f, -0.11990000f },
              { 0.00000000f, 0.00000000f, 1.00000000f },
              0.04023734f,
              0.22473346f,
              0.43633231f },
            { { -0.75647000f, 0.00000000f, 0.00000000f },
              { 1.00000000f, 0.00000000f, 0.00000000f },
              -0.15596873f,
              0.93924665f,
              1.48352986f },
            { { 0.46849000f, 0.00000000f, 0.00000000f },
              { 1.00000000f, 0.00000000f, 0.00000000f },
              -1.44490797f,
              -1.98703462f,
              1.65806279f },
            { { 1.22170000f, 0.00000000f, 0.00000000f },
              { 1.00000000f, 0.00000000f, 0.00000000f },
              -0.20485956f,
              -1.65269882f,
              1.22173048f },
        },
        {
            // ring
            { { 0.00000000f, 0.00000000f, -0.11990000f },
              { 0.00000000f, 0.00000000f, 1.00000000f },
              0.03805731f,
              -0.16319400f,
              0.43633231f },
            { { -0.75647000f, 0.00000000f, 0.00000000f },
              { 1.00000000f, 0.00000000f, 0.00000000f },
              0.25393106f,
              -1.20300895f,
              1.48352986f },
            { { 0.46849000f, 0.00000000f, 0.00000000f },
              { 1.00000000f, 0.00000000f, 0.00000000f },
              -1.76032693f,
              -2.08000213f,
              1.65806279f },
            { { 1.22170000f, 0.00000000f, 0.00000000f },
              { 1.00000000f, 0.00000000f, 0.00000000f },
              0.27639554f,
              -1.44187290f,
              1.22173048f },
        },
        {
            // little
            { { 0.00000000f, 0.17453000f, -0.11990000f },
              { 0.00000000f, 0.00000000f, 1.00000000f },
              -0.14372266f,
              0.02608615f,
              0.43633231f },
            { { -0.75647000f, 0.00000000f, 0.00000000f },
              { 1.00000000f, 0.00000000f, 0.00000000f },
              0.23522014f,
              -1.17004186f,
              1.48352986f },
            { { 0.46849000f, 0.00000000f, 0.00000000f },
              { 1.00000000f, 0.00000000f, 0.00000000f },
              -2.10659963f,
              -2.46172649f,
              1.65806279f },
            { { 1.22170000f, 0.00000000f, 0.00000000f },
              { 1.00000000f, 0.00000000f, 0.00000000f },
              0.51215049f,
              -1.33456117f,
              1.22173048f },
        },
    },
};
// clang-format on

static_assert(std::size(kMechanismFrames) == 2 && std::size(kMechanismFrames[0]) == kFingerCount,
              "one mechanism chain per hand and finger");

//! r31: the absolute mechanism joint angle at the operator's FIST, radians, per hand/finger/slot.
//! This is the closed end of the angle travel the digits are driven from, with the same URDF
//! rest/axis formula (see read_mechanism_angles). The open end is the session's own open-pose angle,
//! stored in the reference block, so the travel is `closed - open_session` and a joint at this angle
//! maps to exactly its human limit. Slots 1..3 are MCP/PIP/DIP (thumb CMC/MCP/IP); slot 0 is the
//! knuckle spread and is not driven from the angle. The PIP's own hinge barely moves in this
//! mechanism's fist (it is a coupled linkage), so the finger's fraction uses the MCP and the DIP only
//! (see finger_angle_fraction); its entry (and slot 0) is kept for the reference round-trip and
//! diagnostics.
//!
//! r35: the MCP and DIP entries are re-measured from the **current mounting's fist**
//! (`sim/out/live_20261112_095821.jsonl`, frame 6968, the less-flexed of 6968/6980; helper:
//! `build/handtrack/r35/measure/derive_r35_closed.py`). They are the angles at which the requested
//! fist frames read exactly 85/95/70; the r31 table was the 09:18 mounting's deeper fist and left the
//! 09:58 fist at 80-84 deg. A frame at or past these values still clamps to 1, so the 09:18 take
//! (whose fist is deeper on every one of these joints) still reaches its maxima. The r35 open
//! deadband does **not** touch this end: f = 1 at `theta_closed` by construction.
constexpr float kAngleClosed[2][kFingerCount][kSlotCount] = {
    {
        // left
        { -0.36617008f, -0.53005649f, 0.82257368f, 3.95805768f }, // thumb
        { -0.32899456f, -0.86666233f, -1.75632483f, -1.10565120f }, // index
        { 0.02042035f, -1.08988125f, -1.82666160f, -1.75453693f }, // middle
        { 0.08255407f, -1.01645261f, -1.83660997f, -1.53658516f }, // ring
        { 0.14067354f, -0.90146000f, -1.86610604f, -1.17318231f }, // little
    },
    {
        // right
        { 0.29304078f, 0.45518187f, 0.87598075f, 4.04898933f }, // thumb
        { 0.35552357f, -0.61291105f, -1.71530959f, -0.67898685f }, // index
        { -0.11815879f, -0.96155149f, -1.81898215f, -1.46437771f }, // middle
        { 0.03159046f, -0.87609893f, -2.02493100f, -1.34612686f }, // ring
        { -0.29426251f, -0.64858610f, -1.93661734f, -0.89143116f }, // little
    },
};

//! A joint's angle travel below this cannot define a fraction (the mechanism barely articulates it
//! over the take); that digit falls back to the chord map and the plugin warns once.
constexpr float kMinAngleTravelRad = 0.0872665f;

//! A hinge within this of its open angle counts as open for the thumb's neutral hold. A real flexed
//! hinge is far outside it, and sensor noise is far inside.
constexpr float kAtOpenPoseToleranceRad = 0.01f;

//! The glove's own open and closed chord per hand and finger, metres. ``open > closed``.
struct ChordExtents
{
    float open;
    float closed;
};

// clang-format off
// r21b, derived by derive_r21b_table.py from raw_tf_ref2.jsonl (the only ground truth). Both hands
// are derived independently: chord_open is the median over the open windows, chord_closed the median
// over the fist window, and the direction offset is the minimal rotation from the mean open device
// tip direction to the finger's rest axis (metacarpal -> MCP).
//
// r22: `open` is only the fallback (and the open end of the baked travel the capture looks for);
// a captured session replaces it with the chord_open it measured, while `closed` stays the closed
// end of the rig's travel - the open end is the part the driver restart moves, the fist the part
// that is a property of the mechanism.
constexpr ChordExtents kChordExtents[2][kFingerCount] = {
    {
        { 0.09011911f, 0.10886922f }, // thumb (ref3: open < closed, inverted travel)
        { 0.09890552f, 0.06305978f }, // index
        { 0.10543076f, 0.05899305f }, // middle
        { 0.10424736f, 0.05286557f }, // ring
        { 0.07870803f, 0.04936622f }, // little
    },
    {
        { 0.08928633f, 0.10510936f }, // thumb (ref3: open < closed, inverted travel)
        { 0.10983548f, 0.06954958f }, // index
        { 0.11762919f, 0.05968725f }, // middle
        { 0.11074475f, 0.05203476f }, // ring
        { 0.08275736f, 0.04921697f }, // little
    },
};

//! r21b: minimal rotation from the device's open tip direction to the finger's rest axis, per hand
//! and finger (thumb, index, middle, ring, little). Both hands derived; no mirroring. The thumb is
//! identity: it is driven by its own tip direction unchanged. r22 keeps these only as the fallback
//! used when a session's open reference cannot be captured; a captured session replaces them with
//! the same rotation measured on that session's own first frames.
constexpr std::array<std::array<XrQuaternionf, kFingerCount>, 2> kDirectionOffset = { {
    {
        XrQuaternionf{ 0.00000000f, 0.00000000f, 0.00000000f, 1.00000000f }, // thumb
        XrQuaternionf{ 0.42154996f, 0.02818449f, 0.06026819f, 0.90436111f }, // index
        XrQuaternionf{ 0.34403690f, 0.00866007f, 0.14819825f, 0.92714664f }, // middle
        XrQuaternionf{ 0.29247291f, 0.01526291f, 0.15790447f, 0.94302323f }, // ring
        XrQuaternionf{ 0.29150117f, 0.02707342f, 0.18340426f, 0.93843326f }, // little
    },
    {
        XrQuaternionf{ 0.00000000f, 0.00000000f, 0.00000000f, 1.00000000f }, // thumb
        XrQuaternionf{ 0.45675761f, 0.04709433f, -0.00400393f, 0.88833472f }, // index
        XrQuaternionf{ 0.40686125f, 0.02566110f, 0.10362145f, 0.90723096f }, // middle
        XrQuaternionf{ 0.35572017f, 0.03345634f, 0.09739670f, 0.92890135f }, // ring
        XrQuaternionf{ 0.32757172f, 0.03518467f, 0.05485354f, 0.94257620f }, // little
    },
} };

// clang-format on

//! The thumb's open device tip direction, per hand, in the solve frame (unit, metres). There is no
//! OpenXR landmark for the thumb metacarpal head, so unlike the four fingers the thumb has no human
//! rest axis to measure at runtime; its reference is the direction the device pointed at the open
//! pose of the baked take (raw_tf_ref3, the same take the thumb chords come from, derived by
//! derive_r22_table.py). A session's capture therefore re-establishes the baked mount instead of a
//! model axis, and is the identity when the tip frame has not changed.
inline constexpr std::array<XrVector3f, 2> kThumbOpenDirection = {
    XrVector3f{ 0.20501270f, 0.82140181f, -0.53223008f }, // left
    XrVector3f{ -0.31660446f, 0.54554962f, -0.77597502f }, // right
};

//! The session window: 40 valid frames, or 2 s once at least 20 are in, whichever comes first, and
//! never later than 4 s. That is the plugin-start premise (the operator holds the hand open); a
//! stream slower than that falls back to the baked table.
constexpr std::size_t kSessionWindowFrames = 40;
constexpr std::size_t kSessionMinimumFrames = 20;
constexpr double kSessionSoftWindowSeconds = 2.0;
constexpr double kSessionHardDeadlineSeconds = 4.0;

//! The chord noise floor, as an offset subtracted from the chord's travel (not a step): 1 mm once the
//! session reference is in use, 2.5 mm while it is still being captured. The old 8 mm was a *step*
//! worth ~40 deg of real bend at the four fingers - the operator's "my finger has to bend before the
//! model moves", and the cliff it created is what made a finger snap from a full curl to zero when the
//! chord drifted back across it. Subtracting the floor instead of gating on it keeps the map C0 (and
//! linear) from the first degree.
constexpr float kOpenDeadbandM = 0.001f;
constexpr float kCaptureDeadbandM = 0.0025f;

//! A travel below this cannot map rotation onto a range without amplifying noise; that joint reads
//! extension and the plugin warns once.
constexpr float kMinTravelRad = 0.05f;
//! A chord window narrower than this cannot drive the radius map; that finger reads its straight
//! length and the plugin warns once.
constexpr float kMinChordSpanM = 0.005f;
constexpr float kTwoPi = 6.28318530717958647692f;
constexpr float kPi = 3.14159265358979323846f;

//! One static OpenXR joint and the fixed offset that attaches it to a mechanism link, in that
//! link's frame, metres. The wrist is not in the table: it is the origin of the skeleton.
struct LandmarkAttachment
{
    XrHandJointEXT joint;
    Link link;
    std::array<float, 3> offset;
};

//! PALM, the four metacarpals, the four MCPs and the thumb CMC, from the r15 table. The thumb's
//! MCP/IP/tip are generated from the CMC by the angle-driven chain, not attached.
inline constexpr LandmarkAttachment kLandmarkAttachments[] = {
    { XR_HAND_JOINT_PALM_EXT, Link::Base, { 0.013624824f, -0.090857048f, -0.016251572f } },
    { XR_HAND_JOINT_THUMB_METACARPAL_EXT, Link::ThumbBase, { -0.049634069f, -0.055869903f, -0.015002549f } },
    { XR_HAND_JOINT_INDEX_METACARPAL_EXT, Link::Base, { 0.037056791f, -0.099468885f, -0.027881093f } },
    { XR_HAND_JOINT_INDEX_PROXIMAL_EXT, Link::Index1, { 0.003533930f, 0.006868323f, -0.001038579f } },
    { XR_HAND_JOINT_MIDDLE_METACARPAL_EXT, Link::Base, { 0.012576311f, -0.106678988f, -0.022461589f } },
    { XR_HAND_JOINT_MIDDLE_PROXIMAL_EXT, Link::Middle1, { 0.0f, 0.0f, 0.0f } },
    { XR_HAND_JOINT_RING_METACARPAL_EXT, Link::Base, { -0.004399851f, -0.114274426f, -0.019806898f } },
    { XR_HAND_JOINT_RING_PROXIMAL_EXT, Link::Ring1, { 0.003209198f, -0.016398319f, 0.000327239f } },
    { XR_HAND_JOINT_LITTLE_METACARPAL_EXT, Link::Base, { -0.023127895f, -0.127444973f, -0.016162299f } },
    { XR_HAND_JOINT_LITTLE_PROXIMAL_EXT, Link::Pinky1, { 0.005070448f, -0.035046682f, 0.003644863f } },
};

//! Right-hand attachment overrides, metres, in the mirrored (left) frame the right hand is built in.
//! A right-hand frame is mirrored into the left frame and the left table is then used unchanged, which
//! is exact whenever the right URDF's link rest frame is the mirror of the left's. It is not for the
//! ThumbBase (rest rpy 3.1416,0,2.1472 vs 0,0,0.99442), and the right take shows the consequence: the
//! CMC landed 75 mm from the wrist instead of ~44 and the thumb could not pinch, and the little's
//! metacarpal->MCP distance drifted 6.75 mm.
//!
//! r27 fitted these four for the right take with the left hand of *another* session as the reference;
//! that fit put the CMC 45 mm from the wrist while the same session's left sits at 33 mm, so the right
//! hand read as bigger and its row was skewed. r30 refits them with an explicit symmetry term: the
//! per-frame offset that puts each right landmark on the left hand's *same-frame* landmark is
//! computed from the operator's two-hand take, and the median is baked here (deterministic, no
//! runtime fitting). The resulting landmarks match the left at the open pose within 0.6 mm (row
//! width, MCP wrist distances, tip spans) and the CMC within 0.5 mm; the right ring/little pinch
//! follows the left's because the thumb now starts from the same place. Residuals are in the r30
//! report. The raw ThumbBase value differs from the left's (x sign and z) because the right ThumbBase
//! rest frame is not a mirror; the *landmark* it produces is the symmetric one.
struct RightAttachmentOverride
{
    XrHandJointEXT joint;
    std::array<float, 3> offset;
};
inline constexpr RightAttachmentOverride kRightAttachmentOverrides[] = {
    { XR_HAND_JOINT_THUMB_METACARPAL_EXT, { 0.049633000f, -0.052335000f, 0.024651000f } },
    { XR_HAND_JOINT_INDEX_PROXIMAL_EXT, { 0.004048000f, 0.006579000f, -0.001039000f } },
    { XR_HAND_JOINT_RING_PROXIMAL_EXT, { -0.000345000f, -0.016698000f, 0.000327000f } },
    { XR_HAND_JOINT_LITTLE_PROXIMAL_EXT, { -0.010946000f, -0.033677000f, 0.003645000f } },
};

//! Human middle MCP->tip length over the mechanism's, so the injected hand is human-sized.
inline constexpr float kHandScale = 0.4963108905f;

//! Translation from the mechanism's wrist-relative frame to the OpenXR wrist frame, metres: it puts
//! the anatomical wrist at the skeleton origin, so WRIST -> MIDDLE_PROXIMAL is the human 91.3 mm.
inline constexpr std::array<float, 3> kWristOffset = { -0.011914288f, 0.088796544f, 0.017472511f };

//! The back of the hand (+Y of the OpenXR anatomy) expressed in the mechanism's wrist frame.
inline constexpr std::array<float, 3> kDorsal = { 0.036613141f, -0.206152855f, 0.977834586f };

//! The OpenXR wrist frame's hand frame candidates, best first.
constexpr const char* kLeftWristFrames[] = { "joint_LeftSkeletonBase", "left_hand", "joint_left_hand" };
constexpr const char* kRightWristFrames[] = { "joint_RightSkeletonBase", "right_hand", "joint_right_hand" };

//! One non-thumb finger: the five OpenXR joints in chain order, the mechanism links that drive each
//! phalanx, and the human segment lengths (metres). The lengths are constants; the chain cannot
//! stretch. Root -> tip they must be strictly non-increasing (the sanity check the operator reads),
//! which is why the index and little have their middle segment longer than the distal one: the
//! earlier table carried 21.58/22.00 and 18.31/22.00, i.e. a distal longer than the middle, and the
//! two were transposed here with the chain totals unchanged.
struct FingerChain
{
    std::array<XrHandJointEXT, 5> joints;
    Link knuckle;
    Link intermediate;
    Link distal;
    Link tip;
    std::array<float, 3> segment_m;
};

inline constexpr FingerChain kFingerChains[] = {
    { { XR_HAND_JOINT_INDEX_METACARPAL_EXT, XR_HAND_JOINT_INDEX_PROXIMAL_EXT, XR_HAND_JOINT_INDEX_INTERMEDIATE_EXT,
        XR_HAND_JOINT_INDEX_DISTAL_EXT, XR_HAND_JOINT_INDEX_TIP_EXT },
      Link::Index1,
      Link::Index2,
      Link::Index3,
      Link::IndexTip,
      { 0.03806f, 0.02200f, 0.02158f } },
    { { XR_HAND_JOINT_MIDDLE_METACARPAL_EXT, XR_HAND_JOINT_MIDDLE_PROXIMAL_EXT, XR_HAND_JOINT_MIDDLE_INTERMEDIATE_EXT,
        XR_HAND_JOINT_MIDDLE_DISTAL_EXT, XR_HAND_JOINT_MIDDLE_TIP_EXT },
      Link::Middle1,
      Link::Middle2,
      Link::Middle3,
      Link::MiddleTip,
      { 0.04153f, 0.02603f, 0.02200f } },
    { { XR_HAND_JOINT_RING_METACARPAL_EXT, XR_HAND_JOINT_RING_PROXIMAL_EXT, XR_HAND_JOINT_RING_INTERMEDIATE_EXT,
        XR_HAND_JOINT_RING_DISTAL_EXT, XR_HAND_JOINT_RING_TIP_EXT },
      Link::Ring1,
      Link::Ring2,
      Link::Ring3,
      Link::RingTip,
      { 0.03628f, 0.02511f, 0.02200f } },
    { { XR_HAND_JOINT_LITTLE_METACARPAL_EXT, XR_HAND_JOINT_LITTLE_PROXIMAL_EXT, XR_HAND_JOINT_LITTLE_INTERMEDIATE_EXT,
        XR_HAND_JOINT_LITTLE_DISTAL_EXT, XR_HAND_JOINT_LITTLE_TIP_EXT },
      Link::Pinky1,
      Link::Pinky2,
      Link::Pinky3,
      Link::PinkyTip,
      { 0.02901f, 0.02200f, 0.01831f } },
};

//! Human thumb segment lengths, metres: CMC->MCP, MCP->IP, IP->tip (strictly non-increasing).
inline constexpr std::array<float, 3> kThumbSegments = { 0.03872f, 0.02690f, 0.02200f };

constexpr std::array<XrHandJointEXT, 4> kThumbChain = {
    XR_HAND_JOINT_THUMB_METACARPAL_EXT,
    XR_HAND_JOINT_THUMB_PROXIMAL_EXT,
    XR_HAND_JOINT_THUMB_DISTAL_EXT,
    XR_HAND_JOINT_THUMB_TIP_EXT,
};

const char* link_name(Link link, Hand hand)
{
    const std::size_t index = static_cast<std::size_t>(link);
    return hand == Hand::Left ? kLeftLinkNames[index] : kRightLinkNames[index];
}

//! The URDF's fixed-axis rpy convention: R = Rz(yaw) * Ry(pitch) * Rx(roll).
XrQuaternionf quaternion_from_rpy(const std::array<float, 3>& rpy)
{
    const float cr = std::cos(0.5f * rpy[0]);
    const float sr = std::sin(0.5f * rpy[0]);
    const float cp = std::cos(0.5f * rpy[1]);
    const float sp = std::sin(0.5f * rpy[1]);
    const float cy = std::cos(0.5f * rpy[2]);
    const float sy = std::sin(0.5f * rpy[2]);
    return XrQuaternionf{ sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy,
                          cr * cp * cy + sr * sp * sy };
}

//! Signed rotation of @p q about @p axis, radians, in (-pi, pi].
float signed_angle_about(const XrQuaternionf& q, const std::array<float, 3>& axis)
{
    XrVector3f v{ q.x, q.y, q.z };
    float w = q.w;
    if (w < 0.0f)
    {
        v = XrVector3f{ -v.x, -v.y, -v.z };
        w = -w;
    }
    return 2.0f * std::atan2(v.x * axis[0] + v.y * axis[1] + v.z * axis[2], w);
}

//! Re-expresses the shortest angle as the representative within half a turn of @p reference. The
//! mapping is stateless: this fixes the quaternion's wrap around the baked open angle, it does not
//! remember a previous frame.
float unwrap_near(float angle, float reference)
{
    return angle + kTwoPi * std::round((reference - angle) / kTwoPi);
}

//! Rotates @p v by the unit quaternion @p q (R v = q v q*).
XrVector3f rotate_vector(const XrQuaternionf& q, const XrVector3f& v)
{
    const XrVector3f u{ q.x, q.y, q.z };
    const float w = q.w;
    const XrVector3f t{ 2.0f * (u.y * v.z - u.z * v.y), 2.0f * (u.z * v.x - u.x * v.z), 2.0f * (u.x * v.y - u.y * v.x) };
    return XrVector3f{ v.x + w * t.x + (u.y * t.z - u.z * t.y), v.y + w * t.y + (u.z * t.x - u.x * t.z),
                       v.z + w * t.z + (u.x * t.y - u.y * t.x) };
}

const geometry_msgs::msg::TransformStamped* find_frame(const tf2_msgs::msg::TFMessage& msg,
                                                       const std::array<const char*, 3>& frames)
{
    for (const char* wanted : frames)
    {
        if (wanted == nullptr || *wanted == '\0')
        {
            break;
        }
        for (const auto& transform : msg.transforms)
        {
            if (transform.child_frame_id == wanted)
            {
                return &transform;
            }
        }
    }
    return nullptr;
}

const geometry_msgs::msg::TransformStamped* find_named(const tf2_msgs::msg::TFMessage& msg, const char* name)
{
    for (const auto& transform : msg.transforms)
    {
        if (transform.child_frame_id == name)
        {
            return &transform;
        }
    }
    return nullptr;
}

const geometry_msgs::msg::TransformStamped* find_link(const tf2_msgs::msg::TFMessage& msg, Link link, Hand hand)
{
    return find_named(msg, link_name(link, hand));
}

XrVector3f vector_of(const std::array<float, 3>& values)
{
    return XrVector3f{ values[0], values[1], values[2] };
}

XrVector3f subtract(const XrVector3f& a, const XrVector3f& b)
{
    return XrVector3f{ a.x - b.x, a.y - b.y, a.z - b.z };
}

float dot(const XrVector3f& a, const XrVector3f& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

float length_of(const XrVector3f& v)
{
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

XrVector3f cross(const XrVector3f& a, const XrVector3f& b)
{
    return XrVector3f{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}

//! Distance between two published frames' origins, metres; negative when either is missing.
float frame_distance(const tf2_msgs::msg::TFMessage& msg, const char* first, const char* second)
{
    const auto* a = find_named(msg, first);
    const auto* b = find_named(msg, second);
    if (a == nullptr || b == nullptr)
    {
        return -1.0f;
    }
    const XrVector3f delta{ static_cast<float>(a->transform.translation.x - b->transform.translation.x),
                            static_cast<float>(a->transform.translation.y - b->transform.translation.y),
                            static_cast<float>(a->transform.translation.z - b->transform.translation.z) };
    return length_of(delta);
}

XrVector3f normalized_or(const XrVector3f& v, const XrVector3f& fallback)
{
    const float length = length_of(v);
    if (!(length > 1e-6f))
    {
        return fallback;
    }
    return XrVector3f{ v.x / length, v.y / length, v.z / length };
}

//! Rodrigues rotation with the angle's cosine and sine already evaluated, so a chain's forward
//! kinematics keeps the trigonometry to one pair per bone.
XrVector3f rotate_about_cached(const XrVector3f& axis, float cosine, float sine, const XrVector3f& v)
{
    const XrVector3f a = cross(axis, v);
    const float d = dot(axis, v) * (1.0f - cosine);
    return XrVector3f{ v.x * cosine + a.x * sine + axis.x * d, v.y * cosine + a.y * sine + axis.y * d,
                       v.z * cosine + a.z * sine + axis.z * d };
}

//! Rotates @p v about the unit @p axis by @p angle (Rodrigues).
XrVector3f rotate_about(const XrVector3f& axis, float angle, const XrVector3f& v)
{
    return rotate_about_cached(axis, std::cos(angle), std::sin(angle), v);
}

//! Re-expresses a point from the mechanism's wrist-relative frame into the skeleton frame: one
//! fixed scale and one fixed translation, both rig constants.
XrVector3f to_skeleton(const XrVector3f& in_wrist)
{
    return XrVector3f{ kHandScale * in_wrist.x + kWristOffset[0], kHandScale * in_wrist.y + kWristOffset[1],
                       kHandScale * in_wrist.z + kWristOffset[2] };
}

//! A published frame as a rigid pose. @p mirror puts a right-hand frame into the left hand's
//! coordinates, the way the two vendor URDFs relate.
struct Pose
{
    XrVector3f position;
    XrQuaternionf orientation;
};

XrQuaternionf normalized_quaternion(const geometry_msgs::msg::TransformStamped& frame)
{
    XrQuaternionf q{ static_cast<float>(frame.transform.rotation.x), static_cast<float>(frame.transform.rotation.y),
                     static_cast<float>(frame.transform.rotation.z), static_cast<float>(frame.transform.rotation.w) };
    const float length = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    if (!(length > 1e-6f))
    {
        return XrQuaternionf{ 0.0f, 0.0f, 0.0f, 1.0f };
    }
    q.x /= length;
    q.y /= length;
    q.z /= length;
    q.w /= length;
    return q;
}

Pose frame_pose(const geometry_msgs::msg::TransformStamped& frame, bool mirror)
{
    Pose pose{};
    pose.position = XrVector3f{ static_cast<float>(frame.transform.translation.x),
                                static_cast<float>(frame.transform.translation.y),
                                static_cast<float>(frame.transform.translation.z) };
    pose.orientation = normalized_quaternion(frame);
    if (mirror)
    {
        pose.position.x = -pose.position.x;
        pose.orientation =
            XrQuaternionf{ pose.orientation.x, -pose.orientation.y, -pose.orientation.z, pose.orientation.w };
    }
    return pose;
}

//! A published transform as a pose, without normalising or mirroring: the wrist pose is the glove's
//! own, which is what `wrist_alignment` is captured against.
XrPosef to_pose(const geometry_msgs::msg::Transform& transform)
{
    XrPosef pose{};
    pose.orientation.x = static_cast<float>(transform.rotation.x);
    pose.orientation.y = static_cast<float>(transform.rotation.y);
    pose.orientation.z = static_cast<float>(transform.rotation.z);
    pose.orientation.w = static_cast<float>(transform.rotation.w);
    pose.position.x = static_cast<float>(transform.translation.x);
    pose.position.y = static_cast<float>(transform.translation.y);
    pose.position.z = static_cast<float>(transform.translation.z);
    return pose;
}

XrVector3f attach(const Pose& link, const std::array<float, 3>& offset)
{
    const XrVector3f rotated = rotate_vector(link.orientation, vector_of(offset));
    return XrVector3f{ link.position.x + rotated.x, link.position.y + rotated.y, link.position.z + rotated.z };
}

XrHandJointLocationEXT make_valid_joint(const XrPosef& pose)
{
    XrHandJointLocationEXT joint{};
    joint.pose = pose;
    joint.radius = kHandJointRadius;
    joint.locationFlags = kPoseValidFlags;
    return joint;
}

XrVector3f joint_position(const HandSkeleton& skeleton, XrHandJointEXT joint)
{
    return skeleton.joints[static_cast<std::size_t>(joint)].pose.position;
}

//! Rotation whose -Z axis is @p forward (OpenXR's joint convention: -Z runs along the bone) and
//! whose roll is pinned by @p up_hint.
XrQuaternionf orientation_along(const XrVector3f& forward, const XrVector3f& up_hint)
{
    const float length = length_of(forward);
    if (!(length > 1e-6f))
    {
        return XrQuaternionf{ 0.0f, 0.0f, 0.0f, 1.0f };
    }
    const XrVector3f z_axis{ -forward.x / length, -forward.y / length, -forward.z / length };

    // up_hint may be parallel to the bone on a degenerate frame; fall back to any perpendicular.
    XrVector3f up = up_hint;
    const float projection = up.x * z_axis.x + up.y * z_axis.y + up.z * z_axis.z;
    up.x -= projection * z_axis.x;
    up.y -= projection * z_axis.y;
    up.z -= projection * z_axis.z;
    float up_length = length_of(up);
    if (!(up_length > 1e-6f))
    {
        up = std::fabs(z_axis.x) < 0.9f ? XrVector3f{ 1.0f, 0.0f, 0.0f } : XrVector3f{ 0.0f, 1.0f, 0.0f };
        const float fallback_dot = up.x * z_axis.x + up.y * z_axis.y + up.z * z_axis.z;
        up.x -= fallback_dot * z_axis.x;
        up.y -= fallback_dot * z_axis.y;
        up.z -= fallback_dot * z_axis.z;
        up_length = length_of(up);
    }
    up.x /= up_length;
    up.y /= up_length;
    up.z /= up_length;

    // x = up x z, y = z x x
    const XrVector3f x_axis{ up.y * z_axis.z - up.z * z_axis.y, up.z * z_axis.x - up.x * z_axis.z,
                             up.x * z_axis.y - up.y * z_axis.x };
    const XrVector3f y_axis{ z_axis.y * x_axis.z - z_axis.z * x_axis.y, z_axis.z * x_axis.x - z_axis.x * x_axis.z,
                             z_axis.x * x_axis.y - z_axis.y * x_axis.x };

    // Rotation matrix (columns x, y, z) -> quaternion.
    const float m00 = x_axis.x, m01 = y_axis.x, m02 = z_axis.x;
    const float m10 = x_axis.y, m11 = y_axis.y, m12 = z_axis.y;
    const float m20 = x_axis.z, m21 = y_axis.z, m22 = z_axis.z;
    XrQuaternionf q{};
    const float trace = m00 + m11 + m22;
    if (trace > 0.0f)
    {
        const float scale = std::sqrt(trace + 1.0f) * 2.0f;
        q.w = 0.25f * scale;
        q.x = (m21 - m12) / scale;
        q.y = (m02 - m20) / scale;
        q.z = (m10 - m01) / scale;
    }
    else if (m00 > m11 && m00 > m22)
    {
        const float scale = std::sqrt(1.0f + m00 - m11 - m22) * 2.0f;
        q.w = (m21 - m12) / scale;
        q.x = 0.25f * scale;
        q.y = (m01 + m10) / scale;
        q.z = (m02 + m20) / scale;
    }
    else if (m11 > m22)
    {
        const float scale = std::sqrt(1.0f + m11 - m00 - m22) * 2.0f;
        q.w = (m02 - m20) / scale;
        q.x = (m01 + m10) / scale;
        q.y = 0.25f * scale;
        q.z = (m12 + m21) / scale;
    }
    else
    {
        const float scale = std::sqrt(1.0f + m22 - m00 - m11) * 2.0f;
        q.w = (m10 - m01) / scale;
        q.x = (m02 + m20) / scale;
        q.y = (m12 + m21) / scale;
        q.z = 0.25f * scale;
    }
    const float norm = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    if (norm > 1e-6f)
    {
        q.x /= norm;
        q.y /= norm;
        q.z /= norm;
        q.w /= norm;
    }
    return q;
}

//! Orients one chain from its attached positions: -Z along the bone, +Y the back of the hand.
template <std::size_t N>
void orient_chain(HandSkeleton& skeleton, const std::array<XrHandJointEXT, N>& chain)
{
    for (std::size_t i = 0; i < N; ++i)
    {
        const XrVector3f current = joint_position(skeleton, chain[i]);
        XrVector3f forward;
        if (i + 1 < N)
        {
            forward = subtract(joint_position(skeleton, chain[i + 1]), current);
        }
        else
        {
            forward = subtract(current, joint_position(skeleton, chain[i - 1]));
        }
        if (!(length_of(forward) > 1e-6f))
        {
            forward = XrVector3f{ 0.0f, 0.0f, -1.0f };
        }
        skeleton.joints[static_cast<std::size_t>(chain[i])].pose.orientation =
            orientation_along(forward, vector_of(kDorsal));
    }
}

//! The hinge axis for a chain running along @p direction with @p dorsal the back of the hand: the
//! axis in the palm plane that makes a positive angle curl towards the palm.
XrVector3f bend_axis_for(const XrVector3f& direction, const XrVector3f& dorsal)
{
    XrVector3f palm_perp = XrVector3f{ -dorsal.x, -dorsal.y, -dorsal.z };
    const float along = dot(palm_perp, direction);
    palm_perp = subtract(palm_perp, XrVector3f{ along * direction.x, along * direction.y, along * direction.z });
    if (!(length_of(palm_perp) > 1e-4f))
    {
        // The bone is along the palm normal; any perpendicular serves as the bend plane.
        const XrVector3f fallback =
            std::fabs(direction.x) < 0.9f ? XrVector3f{ 1.0f, 0.0f, 0.0f } : XrVector3f{ 0.0f, 1.0f, 0.0f };
        const float fallback_along = dot(fallback, direction);
        palm_perp = subtract(fallback, XrVector3f{ fallback_along * direction.x, fallback_along * direction.y,
                                                   fallback_along * direction.z });
    }
    palm_perp = normalized_or(palm_perp, XrVector3f{ 0.0f, 1.0f, 0.0f });
    return normalized_or(cross(direction, palm_perp), palm_perp);
}

[[maybe_unused]] void warn_degenerate_once()
{
    static std::atomic<bool> warned{ false };
    if (!warned.exchange(true, std::memory_order_relaxed))
    {
        std::fprintf(stderr, "io_exo_glove_ros2: a joint's baked travel is below %.2f rad; its flexion is zero\n",
                     static_cast<double>(kMinTravelRad));
    }
}

//! One hand's mechanism angles, [finger][slot], radians.
struct MechanismAngles
{
    std::array<std::array<float, kSlotCount>, kFingerCount> joint{};
};

//! Recovers every mechanism joint's absolute angle from the published link quaternions. The parent
//! of slot 0 is the mechanism root; the others are the previous link's frame. False when a frame is
//! missing or the message mixes reference frames, because then the hand is not complete.
bool read_mechanism_angles(const tf2_msgs::msg::TFMessage& msg,
                           Hand hand,
                           const std::string& reference_frame,
                           MechanismAngles& angles)
{
    // The right hand is the left's X mirror; mirroring the published quaternions into the left frame
    // lets both hands use the one mechanism table (the vendor URDFs differ only by that mirror).
    const std::size_t side_names = hand == Hand::Right ? 1 : 0;
    const bool mirror = false;
    const auto* root = find_link(msg, Link::Base, hand);
    if (root == nullptr || root->header.frame_id != reference_frame)
    {
        return false;
    }
    auto rotation_of = [mirror](const geometry_msgs::msg::TransformStamped& frame) -> XrQuaternionf
    {
        const XrQuaternionf q = normalized_quaternion(frame);
        return mirror ? XrQuaternionf{ q.x, -q.y, -q.z, q.w } : q;
    };
    const XrQuaternionf root_rotation = rotation_of(*root);
    for (std::size_t finger = 0; finger < kFingerCount; ++finger)
    {
        XrQuaternionf parent = root_rotation;
        for (std::size_t slot = 0; slot < kSlotCount; ++slot)
        {
            const auto* frame = find_named(msg, kMechanismFrames[side_names][finger][slot]);
            if (frame == nullptr || frame->header.frame_id != reference_frame)
            {
                return false;
            }
            const XrQuaternionf child = rotation_of(*frame);
            const JointSpec& spec = kJointSpecs[side_names][finger][slot];
            const XrQuaternionf rest = quaternion_from_rpy(spec.rest_rpy);
            const XrQuaternionf relative = multiply_quaternions(conjugate_quaternion(parent), child);
            const XrQuaternionf residual = multiply_quaternions(conjugate_quaternion(rest), relative);
            angles.joint[finger][slot] = unwrap_near(signed_angle_about(residual, spec.axis), spec.open);
            parent = child;
        }
    }
    return true;
}

void warn_narrow_open_once()
{
    static std::atomic<bool> warned{ false };
    if (!warned.exchange(true, std::memory_order_relaxed))
    {
        std::fprintf(stderr,
                     "io_exo_glove_ros2: the session's open chord is more than 15%% below the baked open; the window "
                     "may have caught a bent hand (the reference widens if the hand opens further)\n");
    }
}

void warn_degenerate_chord_once()
{
    static std::atomic<bool> warned{ false };
    if (!warned.exchange(true, std::memory_order_relaxed))
    {
        std::fprintf(stderr, "io_exo_glove_ros2: a finger's baked chord span is below %.1f mm; it reads extension\n",
                     static_cast<double>(kMinChordSpanM * 1000.0f));
    }
}

void warn_degenerate_angle_once()
{
    static std::atomic<bool> warned{ false };
    if (!warned.exchange(true, std::memory_order_relaxed))
    {
        std::fprintf(stderr,
                     "io_exo_glove_ros2: a mechanism joint's angle travel is below %.1f deg; that digit reads 0\n",
                     static_cast<double>(kMinAngleTravelRad) * (180.0 / 3.14159265358979323846));
    }
}

//! The finger and thumb hinge limits, radians, in chain order (MCP/PIP/DIP, CMC/MCP/IP). These are
//! the human maxima the prior split is clamped to, so the chain cannot leave the human range.
inline constexpr std::array<float, 3> kFingerBendMaxRad = { 1.48352986f, 1.65806279f, 1.22173048f };
inline constexpr std::array<float, 3> kThumbBendMaxRad = { 0.61086524f, 0.95993109f, 1.30899694f };
//! The thumb's rest abduction from the index metacarpal, radians (38 deg): the open human hand's
//! thumb, in the palm plane and sideways, not lifted towards the viewer.
inline constexpr float kThumbRestAbductionRad = 0.66322512f;

//! Angle of a rotation, radians, in [0, pi].
float rotation_angle(const XrQuaternionf& q)
{
    return 2.0f * std::acos(std::min(1.0f, std::fabs(q.w)));
}

//! Rounds a length to the block's 3-decimal mm spelling and returns the float the parser will
//! produce for it. The captured reference is quantized through this at the close of the window, so
//! the values a run uses are exactly the values the exported config carries: a later reload is the
//! same mapping, not a 0.5 um approximation of it.
float quantized_reference_m(float metres)
{
    char text[32];
    std::snprintf(text, sizeof(text), "%.3f", static_cast<double>(metres) * 1000.0);
    return std::strtof(text, nullptr) * 0.001f;
}

//! The same for an angle: 3 decimals of a degree, which is what the block's angle section spells,
//! so the captured reference and the reloaded block are the same mapping.
float quantized_reference_rad(float radians)
{
    // Round to 0.001 deg without snprintf: the angle reference is re-quantized whenever the session
    // sees a more open frame, and a formatting call there would show up on the hot path.
    constexpr float kRadToDeg = 180.0f / 3.14159265358979323846f;
    constexpr float kDegToRad = 3.14159265358979323846f / 180.0f;
    return std::round(radians * kRadToDeg * 1000.0f) / 1000.0f * kDegToRad;
}

//! The same for an offset rotation: 6 decimals per component, then normalized exactly as the
//! parser normalizes a block.
XrQuaternionf quantized_reference_quaternion(float x, float y, float z, float w)
{
    char text[32];
    auto rounded = [&text](float value) -> float
    {
        std::snprintf(text, sizeof(text), "%.6f", static_cast<double>(value));
        return std::strtof(text, nullptr);
    };
    const float qx = rounded(x);
    const float qy = rounded(y);
    const float qz = rounded(z);
    const float qw = rounded(w);
    const float norm = std::sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
    if (!(norm > 0.0f))
    {
        return XrQuaternionf{ 0.0f, 0.0f, 0.0f, 1.0f };
    }
    return XrQuaternionf{ qx / norm, qy / norm, qz / norm, qw / norm };
}

//! The shortest rotation taking unit @p from onto unit @p to. This is the "installation calibration"
//! of the reference project at our scale: one rotation per digit, fitted once from the open pose.
XrQuaternionf minimal_rotation(const XrVector3f& from, const XrVector3f& to)
{
    const XrVector3f a = normalized_or(from, XrVector3f{ 0.0f, 0.0f, -1.0f });
    const XrVector3f b = normalized_or(to, a);
    const float cosine = std::clamp(dot(a, b), -1.0f, 1.0f);
    XrVector3f axis = cross(a, b);
    const float sine = length_of(axis);
    if (sine < 1e-6f)
    {
        if (cosine > 0.0f)
        {
            return XrQuaternionf{ 0.0f, 0.0f, 0.0f, 1.0f };
        }
        // Half a turn: any perpendicular is an axis; the sign of w = 0 is not observable.
        axis =
            std::fabs(a.x) < 0.9f ? cross(a, XrVector3f{ 1.0f, 0.0f, 0.0f }) : cross(a, XrVector3f{ 0.0f, 1.0f, 0.0f });
        axis = normalized_or(axis, XrVector3f{ 0.0f, 1.0f, 0.0f });
        return XrQuaternionf{ axis.x, axis.y, axis.z, 0.0f };
    }
    const float half = 0.5f * std::atan2(sine, cosine);
    const float scale = std::sin(half) / sine;
    return XrQuaternionf{ axis.x * scale, axis.y * scale, axis.z * scale, std::cos(half) };
}

//! One digit's session open reference: the chord it read at the open end, and the rotation that puts
//! the live device tip direction back onto that digit's rest axis.
struct SessionDigitReference
{
    float chord_open_m = 0.0f;
    XrQuaternionf direction_offset{ 0.0f, 0.0f, 0.0f, 1.0f };
    float offset_angle_rad = 0.0f;
};

//! A travel below this over the whole window means the hand's frames did not change: a driver that
//! repeats one frame (an idle or frozen glove) must not be read as a hand that flexes. Sensor noise
//! on a live hand is tens of micrometres at the least, so 10 um separates the two cleanly.
constexpr float kStaleTravelM = 1e-5f;
//! The same for the joint angles, radians: a live mechanism's angles move far more than 0.006 deg
//! when the hand moves, so a window below this did not see a hand at all.
constexpr float kStaleAngleRad = 1e-4f;

//! The thumb-to-finger gap needs this much range before it can define an opposition at all.
constexpr float kMinOppositionSpanM = 0.005f;
//! Time constant of the thumb-target blend: a pinch that switches finger is spread over ~0.2 s.
constexpr float kOppositionBlendSeconds = 0.06f;
//! One blend step is at most this long: the plugin sees the glove at ~125 Hz, and a burst of frames
//! arriving faster than that must not freeze the blend.
constexpr float kOppositionFrameSeconds = 0.008f;
//! The opposition reaches 1 when the thumb-index distance has closed to (1 - depth) of its opening
//! value. The depth is the learned range clamped to this band so the map is smooth while the session
//! is still finding its minimum.
constexpr float kMinOppositionDepth = 0.45f;
constexpr float kMaxOppositionDepth = 0.75f;

//! One hand's capture window. Fixed storage: the last frames' per-digit device tip direction and
//! rest axis are kept so the capture can take the median of the last half of the window.
struct SessionCapture
{
    std::array<SessionDigitReference, kFingerCount> digit{};
    std::array<std::array<XrVector3f, kSessionWindowFrames>, kFingerCount> direction{};
    std::array<std::array<XrVector3f, kSessionWindowFrames>, kFingerCount> rest{};
    //! r31: the absolute mechanism joint angle of every window frame, [finger][slot], so the open
    //! pose can be taken as the median of the window's last half exactly like the chords. The raw
    //! angle is unwrapped around the baked open by read_mechanism_angles, so the stored values are
    //! continuous and a reloaded block reproduces them.
    std::array<std::array<std::array<float, kSessionWindowFrames>, kSlotCount>, kFingerCount> angle{};
    //! The open-pose angle this session drives from, [finger][slot], radians.
    std::array<std::array<float, kSlotCount>, kFingerCount> angle_open{};
    //! r35: the open deadband in radians of model bend, hand-level; see SessionReferenceBlock.
    float angle_deadband_rad = 0.0f;
    //! The same deadband as the affine map f = raw * scale - offset, where d = deadband/85 deg:
    //! scale = 1/(1-d), offset = d/(1-d). Precomputed so the hot path keeps one divide per joint.
    float angle_deadband_scale = 1.0f;
    float angle_deadband_offset = 0.0f;
    //! Largest angle excursion seen in the window, radians, for the stale guard.
    std::array<std::array<float, kSlotCount>, kFingerCount> angle_low{};
    std::array<std::array<float, kSlotCount>, kFingerCount> angle_high{};
    std::array<float, kFingerCount> chord_high{};
    std::array<float, kFingerCount> chord_low{};
    std::array<XrVector3f, kFingerCount> tip_first{};
    std::array<float, kFingerCount> tip_travel{};
    //! |glove thumb tip - glove finger tip| for the four fingers (index, middle, ring, little): each
    //! finger's opening value from the window and the running minimum that the opposition map is
    //! normalised against. The finger with the smallest gap is the one being pinched.
    std::array<float, kFingerCount - 1> gap_high{};
    std::array<float, kFingerCount - 1> gap_low{};
    bool gap_valid = false;
    //! The thumb's opposition target after blending, and the blended opposition fraction. A target
    //! switch is spread over kOppositionBlendSeconds instead of stepping.
    XrVector3f aim{};
    float opposition = 0.0f;
    bool aim_valid = false;
    std::int64_t aim_ns = 0;
    std::size_t frames = 0;
    std::int64_t first_ns = 0;
    //! Duration of the window, seconds, kept for the export header's capture statistics.
    float capture_seconds = 0.0f;
    bool closed = false;
    bool fallback = false;
    bool frozen = false;
    //! True when this reference came from the config instead of this run's window (export header).
    bool loaded = false;
    //! True when the hand is driven from the mechanism joint angles: a new-format block was loaded,
    //! or the window measured the open angles. False on a legacy block and on a fallback, which keep
    //! the r30 chord driver.
    bool angle_driver = false;
    //! True when the loaded block carries the r35 deadband section: the block is the operator's
    //! calibration, so the angle open reference is authoritative and does not widen. A four-section
    //! r31 block leaves this false and keeps the old widen-on-a-more-open-pose behaviour.
    bool angle_authoritative = false;
    //! True while this frame is the session's most-open pose (the angle reference's own frame). The
    //! thumb, which the gap drives, is held exactly neutral there so "open reads 0.000" includes it.
    bool at_open_pose = false;
    //! open_metric() of angle_open, cached so the hot path only measures this frame.
    float angle_open_metric = 0.0f;
};

//! Per-hand capture state. Static storage makes the capture stateful with no allocation; the plugin
//! builds both hands from one serialized ROS subscription callback, so no lock is needed here.
std::array<SessionCapture, 2> g_session{};

//! The per-hand log line, written once when the window closes (fixed buffer: the hot path allocates
//! nothing). ``take_session_log_line`` hands it to the caller and clears the flag.
std::array<std::array<char, 512>, 2> g_session_log{};
std::array<bool, 2> g_session_log_pending{};
//! Whether the pending line is a warning (a fallback, a stale stream or a rejected block).
std::array<bool, 2> g_session_log_warning{};

constexpr const char* kDigitNames[kFingerCount] = { "thumb", "index", "middle", "ring", "little" };

void enqueue_session_log(std::size_t side, bool warning, const char* format, ...)
{
    va_list arguments;
    va_start(arguments, format);
    std::vsnprintf(g_session_log[side].data(), g_session_log[side].size(), format, arguments);
    va_end(arguments);
    g_session_log_pending[side] = true;
    g_session_log_warning[side] = warning;
}

std::int64_t monotonic_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

//! True when this frame's chords and directions should move the digits: the window is closed, it did
//! not fall back, and the stream is not stale. Otherwise the hand is held open/neutral.
bool session_drives_digits(std::size_t side)
{
    const SessionCapture& capture = g_session[side];
    return capture.closed && !capture.fallback && !capture.frozen;
}

//! True when the digits are driven from the mechanism joint angles (r31): the window captured the
//! open angles, or a new-format block supplied them. A legacy block and a fallback keep the chord.
bool session_angle_driver(std::size_t side)
{
    const SessionCapture& capture = g_session[side];
    return capture.closed && !capture.fallback && !capture.frozen && capture.angle_driver;
}

//! True while this frame is the session's most-open pose; the thumb is held neutral there.
bool session_at_open_pose(std::size_t side)
{
    // r35: a loaded reference computes at_open_pose per hinge (all hinges open), a captured one from
    // the hand-level metric as in r31; either way the thumb is neutral only at a genuinely open hand.
    const SessionCapture& capture = g_session[side];
    return capture.closed && !capture.fallback && !capture.frozen && capture.at_open_pose;
}

//! The session's open-pose angle for one joint, radians; 0 when there is no angle reference.
float session_angle_open(std::size_t side, std::size_t finger, std::size_t slot)
{
    const SessionCapture& capture = g_session[side];
    return session_angle_driver(side) ? capture.angle_open[finger][slot] : 0.0f;
}

//! The hand's open deadband in radians of model bend; 0 when there is no angle reference.
float session_angle_deadband(std::size_t side)
{
    const SessionCapture& capture = g_session[side];
    return session_angle_driver(side) ? capture.angle_deadband_rad : 0.0f;
}

//! The open chord the radius map normalises against: the captured one once the window is satisfied,
//! the baked table before the window closes and when it fell back.
float session_chord_open(std::size_t side, std::size_t finger)
{
    const SessionCapture& capture = g_session[side];
    const bool captured = capture.closed && !capture.fallback;
    return captured ? capture.digit[finger].chord_open_m : kChordExtents[side][finger].open;
}

//! The opening thumb-to-index gap learned in the window, metres; 0 when there is no usable one.
float session_thumb_index_open(std::size_t side)
{
    const SessionCapture& capture = g_session[side];
    return capture.closed && !capture.fallback && !capture.frozen ? capture.gap_high[0] : 0.0f;
}

//! The smallest thumb-to-index gap seen so far this session, metres.
float session_thumb_index_low(std::size_t side)
{
    const SessionCapture& capture = g_session[side];
    return capture.closed && !capture.fallback && !capture.frozen ? capture.gap_low[0] : 0.0f;
}

/*!
 * @brief The thumb's opposition target and fraction for this frame, blended across target changes.
 *
 * The target is the finger whose glove tip is nearest to the glove thumb tip - that is the finger
 * the operator is pinching, whatever it is - and the aim is that finger's *injected* tip. The
 * fraction is the selected finger's own gap normalised against its own opening value and the range
 * seen so far (a running minimum, clamped to a sane depth band so the map cannot jump while the
 * minimum is still being discovered). Both the aim point and the fraction are blended with
 * kOppositionBlendSeconds, so switching from one finger to another is a ~0.2 s move instead of a
 * step. Returns false when there is no usable reference (window open, fallback, stale).
 */
struct ThumbOpposition
{
    XrVector3f aim{};
    float fraction = 0.0f;
    int target = -1; //!< index, middle, ring or little (0..3), -1 when there is no reference
    bool valid = false;
};

ThumbOpposition thumb_opposition(std::size_t side,
                                 const std::array<float, kFingerCount - 1>& gap_m,
                                 const std::array<XrVector3f, kFingerCount - 1>& tip_m)
{
    SessionCapture& capture = g_session[side];
    ThumbOpposition out;
    if (!capture.closed || capture.fallback || capture.frozen || !capture.gap_valid)
    {
        return out;
    }
    // Which finger is the thumb touching? The glove's own gap decides, every frame.
    std::size_t target = 0;
    float best = gap_m[0];
    for (std::size_t finger = 1; finger < gap_m.size(); ++finger)
    {
        if (gap_m[finger] < best)
        {
            best = gap_m[finger];
            target = finger;
        }
    }
    capture.gap_low[target] = std::min(capture.gap_low[target], gap_m[target]);
    const float open_m = capture.gap_high[target];
    float fraction = 0.0f;
    if (open_m >= kMinOppositionSpanM)
    {
        // The depth is the range seen so far, clamped to a sane band: while the session is still
        // discovering its minimum the raw range starts at zero, and dividing by it would make the
        // opposition jump by tens of degrees between two frames.
        const float depth =
            std::clamp(open_m - capture.gap_low[target], kMinOppositionDepth * open_m, kMaxOppositionDepth * open_m);
        fraction = std::clamp((open_m - gap_m[target]) / depth, 0.0f, 1.0f);
        // r31: the thumb is the one digit the angle reference does not drive, so a session whose
        // thumb-to-finger gap sits a millimetre below the stored open would render a slightly
        // opposed thumb at the open pose. Zero the first 10 % of the gap closure and ramp over the
        // next 10 %: the open pose is exactly 0, and a pinch (which closes >45 % of the gap) is
        // untouched.
        const float open_gate = std::clamp((open_m - gap_m[target]) / (0.10f * open_m), 0.0f, 1.0f);
        fraction *= open_gate;
    }

    // Blend towards this frame's target so a pinch that changes finger is a move, not a step.
    const std::int64_t now = monotonic_ns();
    float alpha = 1.0f;
    if (capture.aim_valid && capture.aim_ns != 0)
    {
        const float dt = static_cast<float>(static_cast<double>(now - capture.aim_ns) * 1e-9);
        alpha = 1.0f - std::exp(-std::max(dt, kOppositionFrameSeconds) / kOppositionBlendSeconds);
        alpha = std::clamp(alpha, 0.0f, 1.0f);
        out.aim = XrVector3f{ capture.aim.x + alpha * (tip_m[target].x - capture.aim.x),
                              capture.aim.y + alpha * (tip_m[target].y - capture.aim.y),
                              capture.aim.z + alpha * (tip_m[target].z - capture.aim.z) };
        out.fraction = capture.opposition + alpha * (fraction - capture.opposition);
    }
    else
    {
        out.aim = tip_m[target];
        out.fraction = fraction;
    }
    capture.aim = out.aim;
    capture.opposition = out.fraction;
    capture.aim_ns = now;
    capture.aim_valid = true;
    out.target = static_cast<int>(target);
    out.valid = true;
    return out;
}

//! How open a mechanism pose is, as one number: the sum of the four fingers' MCP/DIP angles
//! measured from their baked closed ends, in units of their own travel. More open is *smaller*, so
//! the most-open frame of a session is the metric's minimum. Only the eight hinges that carry the
//! flexion are used (the PIP is a coupling link, the spread a separate DOF).
float open_metric(std::size_t side, const std::array<std::array<float, kSlotCount>, kFingerCount>& angles)
{
    float metric = 0.0f;
    for (std::size_t finger = 1; finger < kFingerCount; ++finger)
    {
        // Slots 1 (MCP) and 3 (DIP) are the hinges that carry the curl; unrolled because this runs
        // on every frame.
        const float travel_mcp = kAngleClosed[side][finger][1] - kJointSpecs[side][finger][1].open;
        if (std::fabs(travel_mcp) >= kMinAngleTravelRad)
        {
            metric += (angles[finger][1] - kAngleClosed[side][finger][1]) / travel_mcp;
        }
        const float travel_dip = kAngleClosed[side][finger][3] - kJointSpecs[side][finger][3].open;
        if (std::fabs(travel_dip) >= kMinAngleTravelRad)
        {
            metric += (angles[finger][3] - kAngleClosed[side][finger][3]) / travel_dip;
        }
    }
    return metric;
}

float open_metric(std::size_t side, const MechanismAngles& angles)
{
    return open_metric(side, angles.joint);
}

//! Widens the session's open reference with this frame's chords and angles: it may only grow, so a
//! straighter pose seen after the window (or a window that caught the hand part-way bent) still
//! widens it, and it can never collapse. The chord side keeps its r30 per-digit widening. The angle
//! side works on the hand as a whole: when this frame is more open than the stored reference (by
//! open_metric) every joint's open angle is replaced by this frame's, so the session's most-open
//! pose reads exactly 0 on every joint at once instead of each joint getting its own extreme.
void widen_open_reference(std::size_t side,
                          const std::array<float, kFingerCount>& chord_m,
                          const std::array<float, kFingerCount - 1>& gap_m,
                          const MechanismAngles& angles)
{
    SessionCapture& capture = g_session[side];
    if (!capture.closed || capture.fallback || capture.frozen)
    {
        return;
    }
    for (std::size_t finger = 1; finger < kFingerCount; ++finger)
    {
        capture.digit[finger].chord_open_m = std::max(capture.digit[finger].chord_open_m, chord_m[finger]);
    }
    for (std::size_t finger = 0; finger < gap_m.size(); ++finger)
    {
        capture.gap_high[finger] = std::max(capture.gap_high[finger], gap_m[finger]);
    }
    if (!capture.angle_driver)
    {
        return;
    }
    const float current = open_metric(side, angles);
    // r35: for a loaded reference the thumb's neutral hold is keyed to *every* hinge being at or
    // past its open angle, not to the hand-level sum. The sum can look open when one finger is very
    // extended while another is curled (the 09:58 ring pinch), which used to park the thumb at the
    // neutral pose through the pinch. The per-hinge test keeps the open hand neutral and lets the
    // gap driver oppose during any pinch.
    bool all_open = true;
    for (std::size_t finger = 1; finger < kFingerCount && all_open; ++finger)
    {
        for (std::size_t slot = 1; slot < kSlotCount && all_open; slot += 2)
        {
            const float open = capture.angle_open[finger][slot];
            const float travel = kAngleClosed[side][finger][slot] - open;
            if (std::fabs(travel) < kMinAngleTravelRad)
            {
                continue;
            }
            const float moved = (angles.joint[finger][slot] - open) * (travel >= 0.0f ? 1.0f : -1.0f);
            all_open = !(moved > kAtOpenPoseToleranceRad);
        }
    }
    // angle_open_metric is maintained wherever angle_open is written (capture close, config load,
    // and the update below), so the hot path computes one metric, not two.
    //
    // r35: a *loaded* angle reference is authoritative and does not widen. r31 widened even a loaded
    // block to the most-extended frame of the session, which is exactly what turned a relaxed or
    // stiff straight hand into curl (the anchor drifted to the hyperextended pose and the deadband
    // was left to cover the whole straight range). The reference now keeps the calibrated open pose;
    // the open deadband absorbs the session-to-session variation instead. A *captured* reference
    // still widens, so the start-up window keeps its r31 behaviour.
    if (!capture.angle_authoritative && current < capture.angle_open_metric - 1e-4f)
    {
        for (std::size_t finger = 0; finger < kFingerCount; ++finger)
        {
            for (std::size_t slot = 0; slot < kSlotCount; ++slot)
            {
                capture.angle_open[finger][slot] = quantized_reference_rad(angles.joint[finger][slot]);
            }
        }
        capture.angle_open_metric = open_metric(side, capture.angle_open);
    }
    capture.at_open_pose = capture.angle_authoritative ? all_open : (current <= capture.angle_open_metric + 1e-3f);
}

//! The rotation applied to the live device direction: the captured session offset, or the baked one
//! before the window closes and when it fell back.
const XrQuaternionf& session_direction_offset(std::size_t side, std::size_t finger)
{
    const SessionCapture& capture = g_session[side];
    const bool captured = capture.closed && !capture.fallback;
    return captured ? capture.digit[finger].direction_offset : kDirectionOffset[side][finger];
}

//! The median of the first @p count values, in place (insertion sort; at most 40 entries).
float median_in_place(float* values, std::size_t count)
{
    for (std::size_t i = 1; i < count; ++i)
    {
        const float value = values[i];
        std::size_t j = i;
        while (j > 0 && values[j - 1] > value)
        {
            values[j] = values[j - 1];
            --j;
        }
        values[j] = value;
    }
    return values[count / 2];
}

float component_of(const XrVector3f& value, int component)
{
    return component == 0 ? value.x : (component == 1 ? value.y : value.z);
}

void set_component(XrVector3f& value, int component, float setting)
{
    if (component == 0)
    {
        value.x = setting;
    }
    else if (component == 1)
    {
        value.y = setting;
    }
    else
    {
        value.z = setting;
    }
}

//! Closes the window and fits the per-digit reference: the open-end chord seen in the window and the
//! rotation from the median device tip direction of the window's last half to the digit's rest axis.
void close_session_window(std::size_t side, bool fallback)
{
    SessionCapture& capture = g_session[side];
    const std::size_t count = capture.frames;
    const std::size_t first = count / 2; // the last half of the window, for stability
    const std::size_t used = count - first;
    if (!fallback)
    {
        std::array<float, kSessionWindowFrames> scratch{};
        for (std::size_t finger = 0; finger < kFingerCount; ++finger)
        {
            SessionDigitReference& digit = capture.digit[finger];
            // The open end of this digit's travel: maximum chord for a normal finger, minimum for the
            // inverted thumb (its chord grows as it closes).
            digit.chord_open_m = kChordExtents[side][finger].open > kChordExtents[side][finger].closed ?
                                     capture.chord_high[finger] :
                                     capture.chord_low[finger];
            XrVector3f direction{};
            XrVector3f rest{};
            for (int component = 0; component < 3; ++component)
            {
                for (std::size_t i = 0; i < used; ++i)
                {
                    scratch[i] = component_of(capture.direction[finger][first + i], component);
                }
                set_component(direction, component, median_in_place(scratch.data(), used));
                for (std::size_t i = 0; i < used; ++i)
                {
                    scratch[i] = component_of(capture.rest[finger][first + i], component);
                }
                set_component(rest, component, median_in_place(scratch.data(), used));
            }
            digit.direction_offset = minimal_rotation(direction, rest);
            // r31: the same median over the window's last half gives the open-pose mechanism joint
            // angles. Quantized to the block's spelling so a captured run and a reloaded block use
            // exactly the same reference.
            for (std::size_t slot = 0; slot < kSlotCount; ++slot)
            {
                for (std::size_t i = 0; i < used; ++i)
                {
                    scratch[i] = capture.angle[finger][slot][first + i];
                }
                capture.angle_open[finger][slot] = quantized_reference_rad(median_in_place(scratch.data(), used));
            }
            capture.angle_open_metric = open_metric(side, capture.angle_open);
            // Quantize to the exported spelling now: the run then maps exactly what the config will
            // carry, and a reload reproduces this reference bit for bit.
            digit.chord_open_m = quantized_reference_m(digit.chord_open_m);
            digit.direction_offset = quantized_reference_quaternion(
                digit.direction_offset.x, digit.direction_offset.y, digit.direction_offset.z, digit.direction_offset.w);
            digit.offset_angle_rad = rotation_angle(digit.direction_offset);
        }
        for (std::size_t finger = 0; finger < capture.gap_high.size(); ++finger)
        {
            capture.gap_high[finger] = quantized_reference_m(capture.gap_high[finger]);
            capture.gap_low[finger] = quantized_reference_m(capture.gap_low[finger]);
        }
    }
    capture.closed = true;
    capture.fallback = fallback;
    capture.capture_seconds = static_cast<float>(static_cast<double>(monotonic_ns() - capture.first_ns) * 1e-9);

    if (!fallback)
    {
        // A hand whose frames never moved in the window is a stale stream (an idle or frozen glove):
        // it has no reference to capture, and it must not be read as flexion. The hand is held
        // open/neutral instead and the operator is told.
        float chord_travel = 0.0f;
        float tip_travel = 0.0f;
        float angle_travel = 0.0f;
        for (std::size_t finger = 0; finger < kFingerCount; ++finger)
        {
            chord_travel = std::max(chord_travel, capture.chord_high[finger] - capture.chord_low[finger]);
            tip_travel = std::max(tip_travel, capture.tip_travel[finger]);
            for (std::size_t slot = 0; slot < kSlotCount; ++slot)
            {
                angle_travel = std::max(angle_travel, capture.angle_high[finger][slot] - capture.angle_low[finger][slot]);
            }
        }
        // r31: the joint angles are the flexion signal now, so "never changed" must also mean the
        // angles never changed; the chord/tip terms stay for a hand whose angles are unavailable.
        capture.frozen = chord_travel < kStaleTravelM && tip_travel < kStaleTravelM && angle_travel < kStaleAngleRad;
        capture.angle_driver = !capture.frozen;
        // A window that caught the hand part-way bent learns a bent pose as "open"; say so, and let the
        // monotone widening repair it if the operator opens the hand later.
        for (std::size_t finger = 1; finger < kFingerCount; ++finger)
        {
            const float baked = kChordExtents[side][finger].open;
            if (baked > 0.0f && capture.digit[finger].chord_open_m < 0.85f * baked)
            {
                warn_narrow_open_once();
            }
        }
    }
    else
    {
        capture.angle_driver = false;
    }

    // One line per hand, with the per-finger chord_open and offset angle, or the fallback warning.
    char* const buffer = g_session_log[side].data();
    const std::size_t capacity = g_session_log[side].size();
    int written = 0;
    if (fallback)
    {
        written = std::snprintf(buffer, capacity,
                                "session open reference not captured: %s: only %zu valid frame(s) in %.2f s; using the "
                                "baked table",
                                side == 0 ? "left" : "right", count, kSessionHardDeadlineSeconds);
    }
    else if (capture.frozen)
    {
        written = std::snprintf(buffer, capacity,
                                "session open reference stale: %s: the hand's frames did not change in the %zu-frame "
                                "window; holding the hand open/neutral",
                                side == 0 ? "left" : "right", count);
    }
    else
    {
        const double elapsed = static_cast<double>(capture.capture_seconds);
        written = std::snprintf(buffer, capacity, "session open reference captured: %s:", side == 0 ? "left" : "right");
        for (std::size_t finger = 0;
             finger < kFingerCount && written > 0 && static_cast<std::size_t>(written) < capacity; ++finger)
        {
            const SessionDigitReference& digit = capture.digit[finger];
            const int appended =
                std::snprintf(buffer + written, capacity - static_cast<std::size_t>(written), " %s %.1f mm (%+.1f deg)",
                              kDigitNames[finger], static_cast<double>(digit.chord_open_m) * 1000.0,
                              static_cast<double>(digit.offset_angle_rad) * (180.0 / 3.14159265358979323846));
            if (appended < 0)
            {
                break;
            }
            written += appended;
        }
        if (written > 0 && static_cast<std::size_t>(written) < capacity)
        {
            std::snprintf(buffer + written, capacity - static_cast<std::size_t>(written), "; %zu frames in %.2f s",
                          count, elapsed);
        }
    }
    g_session_log_pending[side] = true;
    g_session_log_warning[side] = fallback || capture.frozen;
}

//! Feeds one complete frame into the hand's session window. A frame counts only when the whole hand
//! was built, so a sparse or partial stream cannot skew the reference. Closing is bounded work; a
//! window that cannot be satisfied by its deadline falls back to the baked table and never blocks
//! the hand from being injected.
void capture_session_reference(std::size_t side,
                               const HandSkeleton& skeleton,
                               const std::array<XrVector3f, kFingerCount>& tip_m,
                               const std::array<float, kFingerCount>& chord_m,
                               const std::array<float, kFingerCount - 1>& gap_m,
                               const MechanismAngles& angles)
{
    SessionCapture& capture = g_session[side];
    if (capture.closed)
    {
        return;
    }
    const std::int64_t now = monotonic_ns();
    if (capture.frames == 0)
    {
        capture.first_ns = now;
    }
    const double elapsed = static_cast<double>(now - capture.first_ns) * 1e-9;

    for (std::size_t finger = 0; finger < kFingerCount; ++finger)
    {
        // r31: the mechanism's own absolute joint angles for this frame. They are the session's
        // open reference when the window closes, exactly like the chord is for the chord driver.
        if (capture.frames < kSessionWindowFrames)
        {
            for (std::size_t slot = 0; slot < kSlotCount; ++slot)
            {
                capture.angle[finger][slot][capture.frames] = angles.joint[finger][slot];
            }
        }
        if (capture.frames == 0)
        {
            for (std::size_t slot = 0; slot < kSlotCount; ++slot)
            {
                capture.angle_low[finger][slot] = angles.joint[finger][slot];
                capture.angle_high[finger][slot] = angles.joint[finger][slot];
            }
        }
        else
        {
            for (std::size_t slot = 0; slot < kSlotCount; ++slot)
            {
                capture.angle_low[finger][slot] = std::min(capture.angle_low[finger][slot], angles.joint[finger][slot]);
                capture.angle_high[finger][slot] = std::max(capture.angle_high[finger][slot], angles.joint[finger][slot]);
            }
        }
        // The digit's rest axis: the four fingers use the human metacarpal -> MCP direction, which is
        // rig-constant geometry the session cannot move. The thumb has no such landmark and uses the
        // baked open device direction instead (see kThumbOpenDirection).
        XrVector3f root{};
        XrVector3f rest{};
        if (finger == 0)
        {
            root = joint_position(skeleton, XR_HAND_JOINT_THUMB_METACARPAL_EXT);
            rest = kThumbOpenDirection[side];
        }
        else
        {
            const FingerChain& chain = kFingerChains[finger - 1];
            root = joint_position(skeleton, chain.joints[1]);
            rest = normalized_or(subtract(root, joint_position(skeleton, chain.joints[0])), kThumbOpenDirection[side]);
        }
        const XrVector3f direction = normalized_or(subtract(tip_m[finger], root), rest);
        if (capture.frames < kSessionWindowFrames)
        {
            capture.direction[finger][capture.frames] = direction;
            capture.rest[finger][capture.frames] = rest;
        }
        if (capture.frames == 0)
        {
            capture.chord_high[finger] = chord_m[finger];
            capture.chord_low[finger] = chord_m[finger];
            capture.tip_first[finger] = tip_m[finger];
            capture.tip_travel[finger] = 0.0f;
        }
        else
        {
            capture.chord_high[finger] = std::max(capture.chord_high[finger], chord_m[finger]);
            capture.chord_low[finger] = std::min(capture.chord_low[finger], chord_m[finger]);
            capture.tip_travel[finger] =
                std::max(capture.tip_travel[finger], length_of(subtract(tip_m[finger], capture.tip_first[finger])));
        }
    }
    bool gaps_valid = true;
    for (std::size_t finger = 0; finger < gap_m.size(); ++finger)
    {
        gaps_valid = gaps_valid && gap_m[finger] >= 0.0f;
    }
    if (gaps_valid)
    {
        if (!capture.gap_valid)
        {
            capture.gap_high = gap_m;
            capture.gap_low = gap_m;
            capture.gap_valid = true;
        }
        else
        {
            for (std::size_t finger = 0; finger < gap_m.size(); ++finger)
            {
                capture.gap_high[finger] = std::max(capture.gap_high[finger], gap_m[finger]);
                capture.gap_low[finger] = std::min(capture.gap_low[finger], gap_m[finger]);
            }
        }
    }
    ++capture.frames;

    const bool satisfied = capture.frames >= kSessionWindowFrames ||
                           (capture.frames >= kSessionMinimumFrames && elapsed >= kSessionSoftWindowSeconds);
    if (satisfied)
    {
        close_session_window(side, false);
    }
    else if (elapsed >= kSessionHardDeadlineSeconds)
    {
        close_session_window(side, true);
    }
}


// ------------------------------------------------------------------------------------------------
// Persistent session reference: block text helpers
//
// The on-disk spelling is one quoted scalar per hand:
//   "chord thumb=98.913 index=105.432 middle=104.251 ring=78.713 little=82.101;
//    offset thumb=0.000000,0.000000,0.000000,1.000000 index=...;
//    gap index=63.814/32.260 middle=82.139/43.477 ring=93.178/65.431 little=91.636/80.139;
//    angle thumb0=... index1=... ...; deadband 25.000"
// mm for chords and gaps, a unit quaternion (x,y,z,w) for each offset, the thumb-to-finger gap as
// open/tightest, degrees for the 20 angle entries and the r35 deadband. The names make a block
// auditable; section order is not significant. Three sections = chord driver, four = r31 angle
// driver (deadband 0), five = r35 angle driver + open deadband.
// ------------------------------------------------------------------------------------------------

constexpr const char* kReferenceDigitKeys[kFingerCount] = { "thumb", "index", "middle", "ring", "little" };

//! Chords a block may carry, mm: outside this range it cannot be this rig's open hand.
constexpr float kReferenceChordMinMm = 20.0f;
constexpr float kReferenceChordMaxMm = 160.0f;
//! A unit rotation rounded to six decimals is within 2e-6 of 1; anything beyond this is not one.
constexpr float kReferenceQuaternionTolerance = 1e-3f;
//! Angles a block may carry, degrees. The thumb's last hinge sits near 4 rad (~230 deg) at the
//! operator's fist (the v5 URDF rest is a half turn from the physical link), so the range is wide;
//! it still rejects a typo that dropped a digit or wrote radians.
constexpr float kReferenceAngleMinDeg = -720.0f;
constexpr float kReferenceAngleMaxDeg = 720.0f;
//! r35 open deadband, degrees of model bend: the measured relaxed-vs-stiff straight offset plus the
//! margin, capped at this value. A block outside the range is rejected, so the cap is enforced in the
//! file as well as at measurement time.
constexpr float kAngleDeadbandMinDeg = 0.0f;
constexpr float kAngleDeadbandMaxDeg = 25.0f;

//! Splits an angle-section key (`<digit><slot>`, e.g. `index1`) into its finger and URDF slot.
bool reference_joint_for_key(const std::string& key, std::size_t& finger, std::size_t& slot)
{
    if (key.size() < 2)
    {
        return false;
    }
    const char digit = key.back();
    if (digit < '0' || digit >= static_cast<char>('0' + kSlotCount))
    {
        return false;
    }
    const std::string name = key.substr(0, key.size() - 1);
    for (std::size_t i = 0; i < kFingerCount; ++i)
    {
        if (name == kReferenceDigitKeys[i])
        {
            finger = i;
            slot = static_cast<std::size_t>(digit - '0');
            return true;
        }
    }
    return false;
}

std::string trim_reference_text(const std::string& text)
{
    const std::size_t first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
    {
        return {};
    }
    const std::size_t last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

std::string reference_number(float value, int places)
{
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.*f", places, static_cast<double>(value));
    return buffer;
}

std::vector<std::string> split_reference_text(const std::string& text, char delimiter)
{
    std::vector<std::string> parts;
    std::size_t begin = 0;
    while (true)
    {
        const std::size_t end = text.find(delimiter, begin);
        parts.push_back(text.substr(begin, end == std::string::npos ? std::string::npos : end - begin));
        if (end == std::string::npos)
        {
            break;
        }
        begin = end + 1;
    }
    return parts;
}

//! Whitespace-separated ``name=value`` fields of one section (empty fields are skipped).
std::vector<std::string> split_reference_fields(const std::string& text)
{
    std::vector<std::string> fields;
    std::size_t cursor = 0;
    while (cursor < text.size())
    {
        const std::size_t begin = text.find_first_not_of(" \t", cursor);
        if (begin == std::string::npos)
        {
            break;
        }
        const std::size_t end = text.find_first_of(" \t", begin);
        fields.push_back(text.substr(begin, end == std::string::npos ? std::string::npos : end - begin));
        if (end == std::string::npos)
        {
            break;
        }
        cursor = end;
    }
    return fields;
}

bool parse_reference_float(const std::string& text, float& value)
{
    if (text.empty())
    {
        return false;
    }
    const char* begin = text.c_str();
    char* end = nullptr;
    const float parsed = std::strtof(begin, &end);
    if (end == begin || *end != '\0' || !std::isfinite(parsed))
    {
        return false;
    }
    value = parsed;
    return true;
}

//! Index into the digit tables for a block key, searching @p first..kFingerCount; -1 when unknown.
int reference_digit_for_key(const std::string& key, std::size_t first)
{
    for (std::size_t finger = first; finger < kFingerCount; ++finger)
    {
        if (key == kReferenceDigitKeys[finger])
        {
            return static_cast<int>(finger);
        }
    }
    return -1;
}

//! One chain: where it starts, its straight direction, its flexion axis, the constant segment lengths
//! and the per-joint limits.
struct ChainSolve
{
    XrVector3f root;
    XrVector3f direction;
    XrVector3f axis;
    const std::array<float, 3>* segment_m;
    const std::array<float, 3>* limit_rad;
};

//! The chain's joint positions (MCP/CMC, PIP/MCP, DIP/IP) and its tip for one angle triple.
struct ChainPoints
{
    std::array<XrVector3f, 3> joint;
    XrVector3f tip;
};

//! Forward kinematics of the constant-length chain: each bone is the previous one rotated about the
//! flexion axis by its own (relative) angle, so the joint angles add up along the chain.
void chain_forward(const ChainSolve& chain, const std::array<float, 3>& angle, ChainPoints& points)
{
    points.joint[0] = chain.root;
    XrVector3f bone = chain.direction;
    XrVector3f position = chain.root;
    for (std::size_t i = 0; i < angle.size(); ++i)
    {
        bone = rotate_about_cached(chain.axis, std::cos(angle[i]), std::sin(angle[i]), bone);
        const float segment = (*chain.segment_m)[i];
        position =
            XrVector3f{ position.x + segment * bone.x, position.y + segment * bone.y, position.z + segment * bone.z };
        if (i + 1 < angle.size())
        {
            points.joint[i + 1] = position;
        }
    }
    points.tip = position;
}

//! The thumb mount opposition that puts @p target in the chain's bend plane. The plane always
//! contains the back-of-hand normal; the mount direction is rotated about it. The branch closest to
//! the mapped opposition is chosen (that keeps the palm-side bend direction) and the result is
//! clamped to the CMC's human range.
[[maybe_unused]] float aim_mount(
    const XrVector3f& dorsal, float seed, float limit, const XrVector3f& root, const XrVector3f& target)
{
    const XrVector3f rest{ 0.0f, 0.0f, -1.0f };
    const float rest_along = dot(rest, dorsal);
    XrVector3f rest_perp =
        subtract(rest, XrVector3f{ rest_along * dorsal.x, rest_along * dorsal.y, rest_along * dorsal.z });
    const XrVector3f relative = subtract(target, root);
    const float target_along = dot(relative, dorsal);
    XrVector3f target_perp =
        subtract(relative, XrVector3f{ target_along * dorsal.x, target_along * dorsal.y, target_along * dorsal.z });
    if (!(length_of(rest_perp) > 1e-4f) || !(length_of(target_perp) > 1e-4f))
    {
        return seed;
    }
    rest_perp = normalized_or(rest_perp, dorsal);
    target_perp = normalized_or(target_perp, rest_perp);
    const float phi = std::atan2(dot(dorsal, cross(rest_perp, target_perp)), dot(rest_perp, target_perp));
    float opposition = -phi;
    while (opposition > kPi)
    {
        opposition -= kTwoPi;
    }
    while (opposition <= -kPi)
    {
        opposition += kTwoPi;
    }
    float best = opposition;
    for (const float candidate : { opposition + kPi, opposition - kPi })
    {
        float wrapped = candidate;
        while (wrapped > kPi)
        {
            wrapped -= kTwoPi;
        }
        while (wrapped <= -kPi)
        {
            wrapped += kTwoPi;
        }
        if (std::fabs(wrapped - seed) < std::fabs(best - seed))
        {
            best = wrapped;
        }
    }
    return std::clamp(best, -limit, limit);
}

//! Signed knuckle spread: `theta - open` clamped to the human +-25 deg.
float map_spread(float angle, const JointSpec& spec)
{
    return std::clamp(angle - spec.open, -spec.human_max, spec.human_max);
}

//! The thumb mount's opposition by magnitude, so a reversed take still opposes towards the index.
[[maybe_unused]] float map_opposition(float angle, const JointSpec& spec)
{
    const float travel = std::fabs(spec.closed - spec.open);
    if (!(travel >= kMinTravelRad))
    {
        warn_degenerate_once();
        return 0.0f;
    }
    return std::min(spec.human_max, spec.human_max * std::fabs(angle - spec.open) / travel);
}


//! The human fist profile: MCP/PIP/DIP 85/95/70 and thumb CMC/MCP/IP 35/55/75. Each joint moves to
//! its own limit as the fraction goes to 1 (see finger_bend / the thumb build). The split used to be the fixed
//! 0.45/0.35/0.20 prior, which saturates the MCP at 85/0.45 = 189 deg of total bend, the PIP at 271 and the DIP at 350
//! - i.e. for more than half of the travel one joint is pinned while the rest keep moving, and the tip's speed steps at
//! every saturation. Scaling each joint by the *same* fraction of its own limit instead reaches 85/95/70 exactly at
//! fraction 1 with no joint ever pinned before it:
//!     angle_i = fraction * limit_i,  total = fraction * sum(limit_i) = fraction * 250 deg.
//! That is the "straightness/human-shape prior" of the reference project with the weights expressed
//! per joint, and it makes the transfer from the chord fraction to the bend a straight line.

//! The chord's fraction of the session's travel: 0 at the session's open chord, 1 at the baked closed
//! chord. The noise floor is *subtracted* from the travel, so the map is continuous and linear from
//! the first degree instead of gated behind a step.
float chord_fraction(std::size_t side, std::size_t finger, float chord_m)
{
    const float open_m = session_chord_open(side, finger);
    const float closed_m = kChordExtents[side][finger].closed;
    const float span = open_m - closed_m;
    if (!(span >= kMinChordSpanM))
    {
        warn_degenerate_chord_once();
        return 0.0f;
    }
    const float floor_m = std::min(session_drives_digits(side) ? kOpenDeadbandM : kCaptureDeadbandM, 0.5f * span);
    const float fraction = (open_m - chord_m - floor_m) / (span - floor_m);
    return std::clamp(fraction, 0.0f, 1.0f);
}

//! The four fingers' joint angles for a bend fraction: each joint moves linearly to its own limit, so
//! a fraction of 0.08 is a 20 deg total bend and a fraction of 1 is exactly 85/95/70.
std::array<float, 3> finger_bend(float fraction)
{
    return { fraction * kFingerBendMaxRad[0], fraction * kFingerBendMaxRad[1], fraction * kFingerBendMaxRad[2] };
}

//! One mechanism joint's fraction of its own travel, 0 at the session's open-pose angle and 1 at the
//! baked fist angle. Signed, so a joint whose travel is inverted (parts of the thumb) works
//! unchanged; a frame that extends past the open pose clamps to 0 instead of curling.
float joint_angle_fraction(std::size_t side, std::size_t finger, std::size_t slot, float angle)
{
    const float open = session_angle_open(side, finger, slot);
    const float travel = kAngleClosed[side][finger][slot] - open;
    if (!(std::fabs(travel) >= kMinAngleTravelRad))
    {
        warn_degenerate_angle_once();
        return 0.0f;
    }
    const float raw = (angle - open) / travel;
    // r35 open deadband. The reference stores one hand-level value in model degrees (the angle the
    // operator actually sees); converting it to a fraction of *this* hinge's travel is the same as a
    // per-joint angle deadband `D = fraction * |travel|`:
    //     f = clamp((raw - d) / (1 - d), 0, 1) == clamp((moved - D) / (|travel| - D), 0, 1)
    // Both ends stay exact: raw = 0 reads 0 for the whole deadband, raw = 1 still reads exactly 1,
    // so the fist is not compressed. A zero deadband (an r31 four-section block) is the old map.
    const SessionCapture& capture = g_session[side];
    // scale/offset are 1/(1-d) and d/(1-d), precomputed at load/capture close. With no deadband
    // they are exactly 1 and 0, so a captured or legacy block keeps the r31 result bit for bit; the
    // branchless form also keeps the hot path at one divide per hinge.
    return std::clamp(raw * capture.angle_deadband_scale - capture.angle_deadband_offset, 0.0f, 1.0f);
}

//! One finger's fraction from its own mechanism hinges. The PIP's hinge is a coupling link that moves
//! very little in this mechanism's fist, so the MCP and DIP carry the curl; the fraction is the
//! larger of the two, so each model joint reaches its own limit when its own hinge reaches its baked
//! closed angle and the finger as a whole reaches 85/95/70 at that pose.
float finger_angle_fraction(std::size_t side, std::size_t finger, const std::array<float, kSlotCount>& joint)
{
    const float mcp = joint_angle_fraction(side, finger, 1, joint[1]);
    const float dip = joint_angle_fraction(side, finger, 3, joint[3]);
    return std::max(mcp, dip);
}

//! The thumb chain's rest (zero-opposition) direction: the index metacarpal projected into the palm
//! plane and rotated by kThumbRestAbductionRad about the palm normal, on the thumb's side. That is an
//! open human hand's thumb - in the palm plane, abducted sideways - so a fully open glove renders a
//! thumb that lies in the palm instead of pointing in front of it (the old rest was the hand's
//! forward axis, which put the thumb ~78 deg out of the plane).
XrVector3f thumb_rest_direction(const XrVector3f& cmc,
                                const XrVector3f& wrist,
                                const XrVector3f& index_metacarpal,
                                const XrVector3f& dorsal)
{
    const float forward_along = dot(index_metacarpal, dorsal);
    const XrVector3f forward = normalized_or(
        subtract(index_metacarpal,
                 XrVector3f{ forward_along * dorsal.x, forward_along * dorsal.y, forward_along * dorsal.z }),
        XrVector3f{ 0.0f, 1.0f, 0.0f });
    const XrVector3f lateral = normalized_or(cross(dorsal, forward), XrVector3f{ 1.0f, 0.0f, 0.0f });
    // Which side of the index metacarpal the thumb is on, in the palm plane.
    const XrVector3f to_thumb = subtract(cmc, wrist);
    const float side = dot(lateral, to_thumb) >= 0.0f ? 1.0f : -1.0f;
    return normalized_or(rotate_about(dorsal, side * kThumbRestAbductionRad, forward), forward);
}

//! Lays one finger's human chain out from its attached MCP and solves it onto @p target. The chain
//! starts along the metacarpal, its bend plane is rolled by @p spread, @p fraction is the finger's
//! bend fraction (angle-driven in r31, chord-driven on a legacy block), and @p mirror_spread flips
//! the signed spread when a right hand is built in the left frame.
void build_finger(HandSkeleton& skeleton,
                  const FingerChain& finger,
                  const XrVector3f& glove_tip,
                  float fraction,
                  float spread,
                  bool mirror_spread,
                  std::size_t side,
                  std::size_t state_index)
{
    const XrVector3f mcp = joint_position(skeleton, finger.joints[1]);
    const XrVector3f metacarpal = joint_position(skeleton, finger.joints[0]);
    const XrVector3f direction = normalized_or(subtract(mcp, metacarpal), XrVector3f{ 0.0f, 0.0f, -1.0f });
    const XrVector3f dorsal = normalized_or(vector_of(kDorsal), XrVector3f{ 0.0f, 1.0f, 0.0f });
    const float signed_spread = mirror_spread ? -spread : spread;

    // The target is the glove tip's own direction from the knuckle at the radius the chord
    // normalisation gives; the device supplies the direction, the human chain the reach.
    const XrVector3f glove_direction = normalized_or(subtract(glove_tip, mcp), direction);
    // r22: the tip-frame orientation is not a rig constant, so the direction is corrected with the
    // rotation this session measured on its own first frames (baked one before the window closes).
    const XrVector3f corrected =
        normalized_or(rotate_vector(session_direction_offset(side, state_index), glove_direction), glove_direction);
    // The fraction is already 0 until the session window closes or a reference is loaded: the open
    // reference is what makes a chord or an angle meaningful, so a guess from the baked table is
    // exactly what this change removes.
    const std::array<float, 3> angle = finger_bend(fraction);
    // The target is where the fist profile would put the tip; it is only used to roll the bend plane.
    const float reach = finger.segment_m[0] + finger.segment_m[1] + finger.segment_m[2];
    const XrVector3f target{ mcp.x + reach * corrected.x, mcp.y + reach * corrected.y, mcp.z + reach * corrected.z };

    // The bend plane is the anatomical one - the palm-side axis rolled by the measured knuckle
    // spread - so every finger always curls towards the palm. Deriving the plane from the device's
    // tip direction instead made the four fingers curl *dorsally* on 60 % of the live frames (the
    // device's target has no reliable side near a straight finger), which the operator saw as the
    // fingers bending backwards. The direction still sets the target the residual is reported against.
    const XrVector3f bend_axis = rotate_about(direction, signed_spread, bend_axis_for(direction, dorsal));

    const ChainSolve chain{ mcp, direction, bend_axis, &finger.segment_m, &kFingerBendMaxRad };
    ChainPoints points{};
    chain_forward(chain, angle, points);
    const float residual = length_of(subtract(points.tip, target));
    skeleton.joints[static_cast<std::size_t>(finger.joints[2])] =
        make_valid_joint(XrPosef{ XrQuaternionf{ 0.0f, 0.0f, 0.0f, 1.0f }, points.joint[1] });
    skeleton.joints[static_cast<std::size_t>(finger.joints[3])] =
        make_valid_joint(XrPosef{ XrQuaternionf{ 0.0f, 0.0f, 0.0f, 1.0f }, points.joint[2] });
    skeleton.joints[static_cast<std::size_t>(finger.joints[4])] =
        make_valid_joint(XrPosef{ XrQuaternionf{ 0.0f, 0.0f, 0.0f, 1.0f }, points.tip });
    skeleton.tip_residual_m[state_index] = residual;
    skeleton.tip_target_m[state_index] = target;
}

//! The total thumb bend, radians, at which the chain's tip gets closest to @p target in its bend
//! plane, at most the sum of the three hinge limits. A bounded scan plus a golden-section refine: no
//! allocation, deterministic, and the result is smooth in the target.
float aim_thumb_bend(const ChainSolve& chain, const XrVector3f& target, float total_limit)
{
    auto error_at = [&](float total) -> float
    {
        const std::array<float, 3> angle = { total * kThumbBendMaxRad[0] / total_limit,
                                             total * kThumbBendMaxRad[1] / total_limit,
                                             total * kThumbBendMaxRad[2] / total_limit };
        ChainPoints points{};
        chain_forward(chain, angle, points);
        return length_of(subtract(points.tip, target));
    };
    constexpr int kSamples = 24;
    const float step = total_limit / static_cast<float>(kSamples);
    float best = 0.0f;
    float best_error = error_at(0.0f);
    for (int sample = 1; sample <= kSamples; ++sample)
    {
        const float total = step * static_cast<float>(sample);
        const float error = error_at(total);
        if (error < best_error)
        {
            best_error = error;
            best = total;
        }
    }
    float low = std::max(0.0f, best - step);
    float high = std::min(total_limit, best + step);
    for (int iteration = 0; iteration < 12; ++iteration)
    {
        const float third = (high - low) / 3.0f;
        if (error_at(low + third) < error_at(high - third))
        {
            high -= third;
        }
        else
        {
            low += third;
        }
    }
    return 0.5f * (low + high);
}

//! Lays the thumb's human chain out from its attached CMC. Its *opposition* - how far it is rotated
//! about the CMC towards the injected index tip - is driven by the glove's own thumb-to-index
//! distance, the only sign-stable signal the device gives (its tip frame's orientation is not a rig
//! constant, and its thumb chord is inverted with an open value that moved 17 mm between takes). At
//! the session's opening distance the thumb is neutral/abducted; at the smallest distance seen this
//! session it is fully opposed, with the chain aimed at the injected index tip. The three hinges are
//! the same fraction of their own 35/55/75 deg limits, so the CMC can never leave its range (the old
//! device-direction "aim" reached 67 deg).
void build_thumb(HandSkeleton& skeleton,
                 const XrVector3f& glove_tip,
                 const std::array<float, kFingerCount - 1>& gap_m,
                 const std::array<XrVector3f, kFingerCount - 1>& tip_m,
                 std::size_t side)
{
    const XrVector3f cmc = joint_position(skeleton, XR_HAND_JOINT_THUMB_METACARPAL_EXT);
    const XrVector3f dorsal = normalized_or(vector_of(kDorsal), XrVector3f{ 0.0f, 1.0f, 0.0f });
    // The rest (zero-opposition) pose is the open human hand's thumb: in the palm plane, abducted
    // sideways from the index metacarpal by kThumbRestAbductionRad. The old rest was the hand's
    // forward axis, which lifted the thumb ~78 deg out of the palm plane - the operator's "thumb sits
    // in front of the palm".
    const XrVector3f index_metacarpal =
        normalized_or(subtract(joint_position(skeleton, XR_HAND_JOINT_INDEX_PROXIMAL_EXT),
                               joint_position(skeleton, XR_HAND_JOINT_INDEX_METACARPAL_EXT)),
                      XrVector3f{ 0.0f, 0.0f, -1.0f });
    const XrVector3f rest =
        thumb_rest_direction(cmc, joint_position(skeleton, XR_HAND_JOINT_WRIST_EXT), index_metacarpal, dorsal);
    const float total_limit = kThumbBendMaxRad[0] + kThumbBendMaxRad[1] + kThumbBendMaxRad[2];

    // The opposition target is the finger the operator is pinching - the one whose glove tip is
    // nearest the glove thumb tip - and the chain swings in the plane spanned by its rest axis and
    // the direction to *that finger's* injected tip. The target and the amount are blended across
    // switches, so moving the pinch from one finger to another is a ~0.2 s move.
    const bool driven = session_drives_digits(side);
    const ThumbOpposition opposition = driven ? thumb_opposition(side, gap_m, tip_m) : ThumbOpposition{};
    const XrVector3f aim = opposition.valid ? opposition.aim : tip_m[0];
    const XrVector3f to_target = normalized_or(subtract(aim, cmc), rest);
    XrVector3f axis = cross(rest, to_target);
    if (!(length_of(axis) > 1e-4f))
    {
        // The target lies along the thumb's rest axis: any bend plane is as good as another.
        axis = bend_axis_for(rest, dorsal);
    }
    axis = normalized_or(axis, XrVector3f{ 0.0f, 1.0f, 0.0f });

    const ChainSolve chain{ cmc, rest, axis, &kThumbSegments, &kThumbBendMaxRad };
    // At the session's open pose the thumb is exactly neutral: the gap that drives it is a few
    // tenths of a millimetre away from the stored opening there, and "every joint reads 0.000"
    // includes the thumb.
    const bool at_open = session_angle_driver(side) && session_at_open_pose(side);
    const float total =
        opposition.valid && !at_open ? opposition.fraction * aim_thumb_bend(chain, aim, total_limit) : 0.0f;
    const std::array<float, 3> angle = { total * kThumbBendMaxRad[0] / total_limit,
                                         total * kThumbBendMaxRad[1] / total_limit,
                                         total * kThumbBendMaxRad[2] / total_limit };
    ChainPoints points{};
    chain_forward(chain, angle, points);
    // How far the thumb ended from the device's own thumb tip, metres. The thumb follows the chord's
    // sign-stable opposition rather than the device's thumb direction, so this is a distance to the
    // device tip, not a solver error.
    const float residual = length_of(subtract(points.tip, glove_tip));
    skeleton.joints[static_cast<std::size_t>(XR_HAND_JOINT_THUMB_PROXIMAL_EXT)] =
        make_valid_joint(XrPosef{ XrQuaternionf{ 0.0f, 0.0f, 0.0f, 1.0f }, points.joint[1] });
    skeleton.joints[static_cast<std::size_t>(XR_HAND_JOINT_THUMB_DISTAL_EXT)] =
        make_valid_joint(XrPosef{ XrQuaternionf{ 0.0f, 0.0f, 0.0f, 1.0f }, points.joint[2] });
    skeleton.joints[static_cast<std::size_t>(XR_HAND_JOINT_THUMB_TIP_EXT)] =
        make_valid_joint(XrPosef{ XrQuaternionf{ 0.0f, 0.0f, 0.0f, 1.0f }, points.tip });
    skeleton.tip_residual_m[0] = residual;
    skeleton.tip_target_m[0] = glove_tip;
    skeleton.thumb_target = opposition.target;
}

} // namespace

void reset_hand_build_state()
{
    // Forget the session window: the next build starts from a fresh capture (the solve itself is
    // stateless, so there is no warm start to forget). An installed reference is part of that state,
    // so it is forgotten too: a caller that wants it after a reset installs it again.
    g_session = {};
    g_session_log = {};
    g_session_log_pending = {};
    g_session_log_warning = {};
}

bool session_reference_captured(Hand hand)
{
    return g_session[hand == Hand::Right ? 1 : 0].closed && !g_session[hand == Hand::Right ? 1 : 0].fallback;
}

bool session_reference_fallback(Hand hand)
{
    return g_session[hand == Hand::Right ? 1 : 0].fallback;
}

bool session_angle_driver(Hand hand)
{
    return session_angle_driver(hand == Hand::Right ? 1 : 0);
}

float session_angle_open(Hand hand, std::size_t finger, std::size_t slot)
{
    if (finger >= kFingerCount || slot >= kSlotCount)
    {
        return 0.0f;
    }
    return session_angle_open(hand == Hand::Right ? 1 : 0, finger, slot);
}

float session_angle_deadband(Hand hand)
{
    return session_angle_deadband(hand == Hand::Right ? 1 : 0);
}

XrVector3f thumb_rest_direction(const HandSkeleton& skeleton)
{
    const XrVector3f cmc = joint_position(skeleton, XR_HAND_JOINT_THUMB_METACARPAL_EXT);
    const XrVector3f index_metacarpal =
        normalized_or(subtract(joint_position(skeleton, XR_HAND_JOINT_INDEX_PROXIMAL_EXT),
                               joint_position(skeleton, XR_HAND_JOINT_INDEX_METACARPAL_EXT)),
                      XrVector3f{ 0.0f, 0.0f, -1.0f });
    return thumb_rest_direction(cmc, joint_position(skeleton, XR_HAND_JOINT_WRIST_EXT), index_metacarpal,
                                normalized_or(vector_of(kDorsal), XrVector3f{ 0.0f, 1.0f, 0.0f }));
}

bool session_reference_stale(Hand hand)
{
    return g_session[hand == Hand::Right ? 1 : 0].frozen;
}

SessionOppositionReference session_opposition_reference(Hand hand)
{
    const std::size_t side = hand == Hand::Right ? 1 : 0;
    SessionOppositionReference reference;
    reference.frozen = g_session[side].frozen;
    reference.open_m = session_thumb_index_open(side);
    reference.low_m = session_thumb_index_low(side);
    reference.valid = reference.open_m > 0.0f;
    return reference;
}

SessionOpenReference session_open_reference(Hand hand, std::size_t finger)
{
    const std::size_t side = hand == Hand::Right ? 1 : 0;
    SessionOpenReference reference;
    if (finger >= kFingerCount)
    {
        return reference;
    }
    const SessionCapture& capture = g_session[side];
    const bool captured = capture.closed && !capture.fallback;
    reference.captured = session_reference_captured(hand);
    reference.fallback = capture.fallback;
    reference.frames = capture.frames;
    reference.angle_driver = session_angle_driver(side);
    if (captured)
    {
        reference.chord_open_m = capture.digit[finger].chord_open_m;
        reference.offset_angle_rad = capture.digit[finger].offset_angle_rad;
    }
    else
    {
        reference.chord_open_m = kChordExtents[side][finger].open;
        reference.offset_angle_rad = rotation_angle(kDirectionOffset[side][finger]);
    }
    return reference;
}

const char* take_session_log_line(Hand hand)
{
    const std::size_t side = hand == Hand::Right ? 1 : 0;
    if (!g_session_log_pending[side])
    {
        return "";
    }
    g_session_log_pending[side] = false;
    return g_session_log[side].data();
}

bool session_log_line_is_warning(Hand hand)
{
    return g_session_log_warning[hand == Hand::Right ? 1 : 0];
}

bool parse_session_reference_block(const char* text, SessionReferenceBlock& block, std::string& error)
{
    error.clear();
    block = SessionReferenceBlock{};
    if (text == nullptr || *text == '\0')
    {
        error = "empty block";
        return false;
    }
    const std::vector<std::string> sections = split_reference_text(text, ';');
    // r31: the angle section is optional (three sections = the chord driver). r35: an optional fifth
    // `deadband` section follows it; a four-section r31 block still loads with deadband 0.
    if (sections.size() != 3 && sections.size() != 4 && sections.size() != 5)
    {
        error = "expected 3, 4 or 5 sections separated by ';' (chord, offset, gap[, angle[, deadband]]), found " +
                std::to_string(sections.size());
        return false;
    }
    block.has_angles = sections.size() >= 4;

    bool saw_chord = false;
    bool saw_offset = false;
    bool saw_gap = false;
    bool saw_angle = false;
    bool saw_deadband = false;
    std::array<bool, kFingerCount> chord_seen{};
    std::array<bool, kFingerCount> offset_seen{};
    std::array<bool, kFingerCount - 1> gap_seen{};
    std::array<std::array<bool, kSlotCount>, kFingerCount> angle_seen{};

    for (const std::string& raw_section : sections)
    {
        const std::string section = trim_reference_text(raw_section);
        const std::size_t space = section.find_first_of(" \t");
        const std::string name = section.substr(0, space);
        const std::string body = space == std::string::npos ? std::string{} : trim_reference_text(section.substr(space));
        const bool chord = name == "chord";
        const bool offset = name == "offset";
        const bool gap = name == "gap";
        const bool angle = name == "angle";
        const bool deadband = name == "deadband";
        if (!chord && !offset && !gap && !angle && !deadband)
        {
            error = "unknown section '" + name + "' (expected chord, offset, gap, angle or deadband)";
            return false;
        }
        if ((chord && saw_chord) || (offset && saw_offset) || (gap && saw_gap) || (angle && saw_angle) ||
            (deadband && saw_deadband))
        {
            error = "duplicate '" + name + "' section";
            return false;
        }
        saw_chord = saw_chord || chord;
        saw_offset = saw_offset || offset;
        saw_gap = saw_gap || gap;
        saw_angle = saw_angle || angle;
        saw_deadband = saw_deadband || deadband;
        if (body.empty())
        {
            error = "'" + name + "' has no values";
            return false;
        }
        if (deadband)
        {
            float degrees = 0.0f;
            if (!parse_reference_float(body, degrees))
            {
                error = "deadband: '" + body + "' is not a single number";
                return false;
            }
            if (!(degrees >= kAngleDeadbandMinDeg && degrees <= kAngleDeadbandMaxDeg))
            {
                error = "deadband: " + reference_number(degrees, 3) + " deg is outside " +
                        reference_number(kAngleDeadbandMinDeg, 3) + " .. " + reference_number(kAngleDeadbandMaxDeg, 3) +
                        " deg";
                return false;
            }
            block.angle_deadband_rad = degrees * (kPi / 180.0f);
            block.has_deadband = true;
            continue;
        }

        for (const std::string& field : split_reference_fields(body))
        {
            const std::size_t equals = field.find('=');
            if (equals == std::string::npos || equals == 0 || equals + 1 >= field.size())
            {
                error = "'" + name + "': expected <digit>=<value>, got '" + field + "'";
                return false;
            }
            const std::string key = field.substr(0, equals);
            const std::string value = field.substr(equals + 1);
            if (angle)
            {
                std::size_t index = 0;
                std::size_t slot = 0;
                if (!reference_joint_for_key(key, index, slot))
                {
                    error = "angle: unknown joint '" + key + "' (expected <digit>0..3)";
                    return false;
                }
                if (angle_seen[index][slot])
                {
                    error = "angle: duplicate '" + key + "'";
                    return false;
                }
                angle_seen[index][slot] = true;
                float degrees = 0.0f;
                if (!parse_reference_float(value, degrees))
                {
                    error = "angle " + key + ": '" + value + "' is not a number";
                    return false;
                }
                if (degrees < kReferenceAngleMinDeg || degrees > kReferenceAngleMaxDeg)
                {
                    error = "angle " + key + ": " + reference_number(degrees, 3) + " deg is outside " +
                            reference_number(kReferenceAngleMinDeg, 0) + " .. " +
                            reference_number(kReferenceAngleMaxDeg, 0) + " deg";
                    return false;
                }
                block.angle_open_rad[index][slot] = quantized_reference_rad(degrees * (kPi / 180.0f));
                continue;
            }
            const int digit = reference_digit_for_key(key, gap ? 1 : 0);
            if (digit < 0)
            {
                error = "'" + name + "': unknown digit '" + key + "'";
                return false;
            }
            const std::size_t index = static_cast<std::size_t>(digit);

            if (chord)
            {
                if (chord_seen[index])
                {
                    error = "chord: duplicate '" + key + "'";
                    return false;
                }
                chord_seen[index] = true;
                float millimetres = 0.0f;
                if (!parse_reference_float(value, millimetres))
                {
                    error = "chord " + key + ": '" + value + "' is not a number";
                    return false;
                }
                if (millimetres < kReferenceChordMinMm || millimetres > kReferenceChordMaxMm)
                {
                    error = "chord " + key + ": " + reference_number(millimetres, 3) + " mm is outside " +
                            reference_number(kReferenceChordMinMm, 0) + "-" +
                            reference_number(kReferenceChordMaxMm, 0) + " mm";
                    return false;
                }
                block.chord_open_m[index] = millimetres * 0.001f;
                continue;
            }

            if (offset)
            {
                if (offset_seen[index])
                {
                    error = "offset: duplicate '" + key + "'";
                    return false;
                }
                offset_seen[index] = true;
                const std::vector<std::string> components = split_reference_text(value, ',');
                if (components.size() != 4)
                {
                    error = "offset " + key + ": expected 4 comma-separated numbers (x,y,z,w), found " +
                            std::to_string(components.size());
                    return false;
                }
                std::array<float, 4> quaternion{};
                for (std::size_t component = 0; component < 4; ++component)
                {
                    if (!parse_reference_float(trim_reference_text(components[component]), quaternion[component]))
                    {
                        error = "offset " + key + ": component " + std::to_string(component + 1) + " is not a number";
                        return false;
                    }
                }
                const float norm = std::sqrt(quaternion[0] * quaternion[0] + quaternion[1] * quaternion[1] +
                                             quaternion[2] * quaternion[2] + quaternion[3] * quaternion[3]);
                if (!(std::fabs(norm - 1.0f) <= kReferenceQuaternionTolerance))
                {
                    error = "offset " + key + ": quaternion norm " + reference_number(norm, 6) +
                            " is not 1 (a unit rotation is required)";
                    return false;
                }
                block.direction_offset[index] =
                    quantized_reference_quaternion(quaternion[0], quaternion[1], quaternion[2], quaternion[3]);
                continue;
            }

            // gap
            if (gap_seen[index - 1])
            {
                error = "gap: duplicate '" + key + "'";
                return false;
            }
            gap_seen[index - 1] = true;
            const std::vector<std::string> ends = split_reference_text(value, '/');
            if (ends.size() != 2)
            {
                error = "gap " + key + ": expected <open>/<tightest>, found " + std::to_string(ends.size()) + " value(s)";
                return false;
            }
            float open_mm = 0.0f;
            float tight_mm = 0.0f;
            if (!parse_reference_float(trim_reference_text(ends[0]), open_mm) ||
                !parse_reference_float(trim_reference_text(ends[1]), tight_mm))
            {
                error = "gap " + key + ": '" + value + "' is not two numbers";
                return false;
            }
            if (open_mm < 0.0f || tight_mm < 0.0f)
            {
                error = "gap " + key + ": negative gap (" + reference_number(open_mm, 3) + " mm open, " +
                        reference_number(tight_mm, 3) + " mm tightest)";
                return false;
            }
            if (tight_mm > open_mm)
            {
                error = "gap " + key + ": tightest " + reference_number(tight_mm, 3) + " mm exceeds open " +
                        reference_number(open_mm, 3) + " mm";
                return false;
            }
            block.gap_open_m[index - 1] = open_mm * 0.001f;
            block.gap_tight_m[index - 1] = tight_mm * 0.001f;
        }
    }

    for (std::size_t finger = 0; finger < kFingerCount; ++finger)
    {
        if (!chord_seen[finger])
        {
            error = std::string("missing chord for ") + kReferenceDigitKeys[finger];
            return false;
        }
        if (!offset_seen[finger])
        {
            error = std::string("missing offset for ") + kReferenceDigitKeys[finger];
            return false;
        }
    }
    for (std::size_t finger = 0; finger < kFingerCount - 1; ++finger)
    {
        if (!gap_seen[finger])
        {
            error = std::string("missing gap for ") + kReferenceDigitKeys[finger + 1];
            return false;
        }
    }
    if (sections.size() >= 4)
    {
        if (!saw_angle)
        {
            error =
                "expected 4 or 5 sections separated by ';' (chord, offset, gap, angle[, deadband]), "
                "found no angle section";
            return false;
        }
        for (std::size_t finger = 0; finger < kFingerCount; ++finger)
        {
            for (std::size_t slot = 0; slot < kSlotCount; ++slot)
            {
                if (!angle_seen[finger][slot])
                {
                    error = std::string("missing angle for ") + kReferenceDigitKeys[finger] +
                            static_cast<char>('0' + static_cast<int>(slot));
                    return false;
                }
            }
        }
    }
    if (sections.size() == 5 && !saw_deadband)
    {
        error =
            "expected 5 sections separated by ';' (chord, offset, gap, angle, deadband), "
            "found no deadband section";
        return false;
    }
    return true;
}

SessionReferenceLoad install_session_reference(Hand hand, const char* text)
{
    const std::size_t side = hand == Hand::Right ? 1 : 0;
    const char* const side_name = side == 0 ? "left" : "right";
    SessionCapture& capture = g_session[side];
    if (capture.closed)
    {
        // Already captured (or installed) this run: never let a second source overwrite it.
        return capture.loaded ? SessionReferenceLoad::Loaded : SessionReferenceLoad::NotProvided;
    }
    if (text == nullptr || *text == '\0')
    {
        enqueue_session_log(
            side, false, "session reference: %s: not in config; capturing from the first frames", side_name);
        return SessionReferenceLoad::NotProvided;
    }

    SessionReferenceBlock block;
    std::string error;
    if (!parse_session_reference_block(text, block, error))
    {
        enqueue_session_log(side, true, "session reference: %s: rejected (%s); capturing from the first frames",
                            side_name, error.c_str());
        return SessionReferenceLoad::Rejected;
    }
    // The four fingers' open chord must still be above the baked closed end, or the chord map would
    // have no travel to work with. The thumb is excluded: its chord is inverted and not mapped.
    for (std::size_t finger = 1; finger < kFingerCount; ++finger)
    {
        if (!(block.chord_open_m[finger] - kChordExtents[side][finger].closed >= kMinChordSpanM))
        {
            enqueue_session_log(side, true,
                                "session reference: %s: rejected (chord %s %s mm is not above the baked closed "
                                "chord %s mm); capturing from the first frames",
                                side_name, kReferenceDigitKeys[finger],
                                reference_number(block.chord_open_m[finger] * 1000.0f, 3).c_str(),
                                reference_number(kChordExtents[side][finger].closed * 1000.0f, 3).c_str());
            return SessionReferenceLoad::Rejected;
        }
    }

    for (std::size_t finger = 0; finger < kFingerCount; ++finger)
    {
        SessionDigitReference& digit = capture.digit[finger];
        digit.chord_open_m = block.chord_open_m[finger];
        digit.direction_offset = block.direction_offset[finger];
        digit.offset_angle_rad = rotation_angle(digit.direction_offset);
    }
    capture.gap_high = block.gap_open_m;
    capture.gap_low = block.gap_tight_m;
    capture.gap_valid = true;
    capture.closed = true;
    capture.fallback = false;
    capture.frozen = false;
    capture.loaded = true;
    capture.frames = 0;
    if (block.has_angles)
    {
        for (std::size_t finger = 0; finger < kFingerCount; ++finger)
        {
            for (std::size_t slot = 0; slot < kSlotCount; ++slot)
            {
                capture.angle_open[finger][slot] = block.angle_open_rad[finger][slot];
            }
        }
        capture.angle_open_metric = open_metric(side, capture.angle_open);
        capture.angle_driver = true;
        capture.angle_authoritative = block.has_deadband;
        capture.angle_deadband_rad = block.angle_deadband_rad;
        const float deadband_fraction = std::clamp(block.angle_deadband_rad / kFingerBendMaxRad[0], 0.0f, 0.9f);
        capture.angle_deadband_scale = 1.0f / (1.0f - deadband_fraction);
        capture.angle_deadband_offset = deadband_fraction / (1.0f - deadband_fraction);
        if (block.has_deadband)
        {
            constexpr float kRadToDegLocal = 180.0f / kPi;
            enqueue_session_log(side, false,
                                "session reference: %s: loaded from config (joint-angle driver, open deadband %s deg)",
                                side_name, reference_number(capture.angle_deadband_rad * kRadToDegLocal, 3).c_str());
        }
        else
        {
            // A four-section r31 block has no deadband. It still loads and runs the r31 map exactly
            // (deadband 0), so an operator's existing block does not change behaviour silently.
            enqueue_session_log(side, true,
                                "session reference: %s: loaded from config (joint-angle driver, no open "
                                "deadband: r31 map; re-export the reference to add it)",
                                side_name);
        }
    }
    else
    {
        // A block from an earlier round has no open-pose angles. It still loads (its chords, offsets
        // and gaps are complete), but the digits must run the r30 chord driver; the warning tells the
        // operator how to get the angle driver back.
        capture.angle_driver = false;
        enqueue_session_log(side, true,
                            "session reference: %s: loaded from config (no angle section: using the CHORD driver; "
                            "re-export the reference to drive from the mechanism's joint angles)",
                            side_name);
    }
    return SessionReferenceLoad::Loaded;
}

bool session_reference_available(Hand hand)
{
    const SessionCapture& capture = g_session[hand == Hand::Right ? 1 : 0];
    return capture.closed && !capture.fallback && !capture.frozen;
}

bool session_reference_loaded(Hand hand)
{
    return g_session[hand == Hand::Right ? 1 : 0].loaded;
}

bool format_session_reference_block(Hand hand, std::string& text)
{
    const SessionCapture& capture = g_session[hand == Hand::Right ? 1 : 0];
    if (!capture.closed || capture.fallback || capture.frozen)
    {
        text.clear();
        return false;
    }
    text.clear();
    text += "chord";
    for (std::size_t finger = 0; finger < kFingerCount; ++finger)
    {
        text += " ";
        text += kReferenceDigitKeys[finger];
        text += "=";
        text += reference_number(capture.digit[finger].chord_open_m * 1000.0f, 3);
    }
    text += "; offset";
    for (std::size_t finger = 0; finger < kFingerCount; ++finger)
    {
        const XrQuaternionf& rotation = capture.digit[finger].direction_offset;
        text += " ";
        text += kReferenceDigitKeys[finger];
        text += "=";
        text += reference_number(rotation.x, 6);
        text += ",";
        text += reference_number(rotation.y, 6);
        text += ",";
        text += reference_number(rotation.z, 6);
        text += ",";
        text += reference_number(rotation.w, 6);
    }
    text += "; gap";
    for (std::size_t finger = 1; finger < kFingerCount; ++finger)
    {
        text += " ";
        text += kReferenceDigitKeys[finger];
        text += "=";
        text += reference_number(capture.gap_high[finger - 1] * 1000.0f, 3);
        text += "/";
        text += reference_number(capture.gap_low[finger - 1] * 1000.0f, 3);
    }
    // The r31 open-pose mechanism joint angles: `<digit><slot>=<degrees>` for the 20 joints. They are
    // what makes the angle driver survive a restart; a block without them still loads and uses the
    // chord driver.
    if (capture.angle_driver)
    {
        constexpr float kRadToDeg = 180.0f / 3.14159265358979323846f;
        text += "; angle";
        for (std::size_t finger = 0; finger < kFingerCount; ++finger)
        {
            for (std::size_t slot = 0; slot < kSlotCount; ++slot)
            {
                text += " ";
                text += kReferenceDigitKeys[finger];
                text += static_cast<char>('0' + static_cast<int>(slot));
                text += "=";
                text += reference_number(capture.angle_open[finger][slot] * kRadToDeg, 3);
            }
        }
        // r35: the hand-level open deadband in model degrees. Only a block that *came with* the
        // deadband section writes it back (authoritative, so export -> reload is byte-identical).
        // A captured window cannot measure the relaxed-vs-stiff offset, so its export stays a
        // four-section r31 block and keeps the r31 widening on reload; the same holds for a loaded
        // four-section block, so an old block is never silently upgraded.
        if (capture.angle_authoritative)
        {
            text += "; deadband ";
            text += reference_number(capture.angle_deadband_rad * kRadToDeg, 3);
        }
    }
    return true;
}

bool export_session_reference_file(const std::string& path, const char* plugin_version, std::string& error)
{
    error.clear();
    if (path.empty())
    {
        error = "no output path";
        return false;
    }
    constexpr const char* kSides[2] = { "left", "right" };
    std::array<std::string, 2> blocks;
    bool any = false;
    for (std::size_t side = 0; side < 2; ++side)
    {
        any = format_session_reference_block(side == 0 ? Hand::Left : Hand::Right, blocks[side]) || any;
    }
    if (!any)
    {
        error = "no session reference is available yet (no capture and none loaded)";
        return false;
    }

    char stamp[64] = "unknown time";
    const std::time_t now = std::time(nullptr);
    std::tm local{};
    if (localtime_r(&now, &local) != nullptr)
    {
        std::strftime(stamp, sizeof(stamp), "%Y-%m-%dT%H:%M:%S%z", &local);
    }

    std::string snippet;
    snippet += "# SPDX-FileCopyrightText: Copyright (c) 2026 IO. All rights reserved.\n";
    snippet += "# SPDX-License-Identifier: Apache-2.0\n";
    snippet += "#\n";
    snippet += "# io_exo_glove_ros2 session reference, generated ";
    snippet += stamp;
    snippet += " by io_exo_glove_ros2 ";
    snippet += plugin_version == nullptr ? "unknown" : plugin_version;
    snippet += ".\n";
    snippet += "# Paste these keys into config/io_exo_glove_ros2.yaml, or pass this file as\n";
    snippet += "# --session-reference=<this file>. Format and operator workflow: README.md, \"Session reference\".\n";
    for (std::size_t side = 0; side < 2; ++side)
    {
        const SessionCapture& capture = g_session[side];
        snippet += "# ";
        snippet += kSides[side];
        snippet += ": ";
        if (blocks[side].empty())
        {
            snippet += "no usable reference (the capture window will learn one)\n";
        }
        else if (capture.loaded)
        {
            snippet += "loaded from the config (kept unchanged)\n";
        }
        else
        {
            snippet += "captured this run, " + std::to_string(capture.frames) + " frames in " +
                       reference_number(capture.capture_seconds, 3) + " s\n";
        }
    }
    for (std::size_t side = 0; side < 2; ++side)
    {
        if (blocks[side].empty())
        {
            continue;
        }
        snippet += "reference_";
        snippet += kSides[side];
        snippet += ": \"";
        snippet += blocks[side];
        snippet += "\"\n";
    }

    std::FILE* file = std::fopen(path.c_str(), "wb");
    if (file == nullptr)
    {
        error = "cannot open '" + path + "' for writing";
        return false;
    }
    const std::size_t written = std::fwrite(snippet.data(), 1, snippet.size(), file);
    const int close_result = std::fclose(file);
    if (written != snippet.size() || close_result != 0)
    {
        error = "cannot write '" + path + "'";
        return false;
    }
    return true;
}

std::array<HandJointSlot, kHandJointSlotCount> make_hand_joint_slots(Hand hand)
{
    std::array<HandJointSlot, kHandJointSlotCount> slots{};
    for (std::size_t i = 0; i < kHandJointSlotCount; ++i)
    {
        slots[i].joint = static_cast<XrHandJointEXT>(i);
        slots[i].frames = { "", "", "" };
    }

    const char* const* wrist_frames = hand == Hand::Left ? kLeftWristFrames : kRightWristFrames;
    slots[static_cast<std::size_t>(XR_HAND_JOINT_WRIST_EXT)].frames = { wrist_frames[0], wrist_frames[1],
                                                                        wrist_frames[2] };
    for (const LandmarkAttachment& attachment : kLandmarkAttachments)
    {
        slots[static_cast<std::size_t>(attachment.joint)].frames = { link_name(attachment.link, hand), "", "" };
    }

    // The thumb's phalanges are generated from its CMC; log the mechanism links that drive them.
    slots[static_cast<std::size_t>(XR_HAND_JOINT_THUMB_PROXIMAL_EXT)].frames = { link_name(Link::Thumb2, hand), "", "" };
    slots[static_cast<std::size_t>(XR_HAND_JOINT_THUMB_DISTAL_EXT)].frames = { link_name(Link::Thumb3, hand), "", "" };
    slots[static_cast<std::size_t>(XR_HAND_JOINT_THUMB_TIP_EXT)].frames = { link_name(Link::ThumbTip, hand), "", "" };

    for (const FingerChain& finger : kFingerChains)
    {
        slots[static_cast<std::size_t>(finger.joints[2])].frames = { link_name(finger.intermediate, hand), "", "" };
        slots[static_cast<std::size_t>(finger.joints[3])].frames = { link_name(finger.distal, hand), "", "" };
        slots[static_cast<std::size_t>(finger.joints[4])].frames = { link_name(finger.tip, hand), "", "" };
    }
    return slots;
}

XrQuaternionf multiply_quaternions(const XrQuaternionf& a, const XrQuaternionf& b)
{
    // Same convention as oxr_utils::multiply_poses, which is defined for poses.
    XrQuaternionf result{};
    result.w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z;
    result.x = a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y;
    result.y = a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x;
    result.z = a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w;
    return result;
}

XrQuaternionf conjugate_quaternion(const XrQuaternionf& q)
{
    return XrQuaternionf{ -q.x, -q.y, -q.z, q.w };
}

XrPosef rotate_pose(const XrQuaternionf& rotation, const XrPosef& pose)
{
    // A pose is a rigid transform; rotating it is composing a pure rotation in front of it.
    const XrPosef pure_rotation{ rotation, XrVector3f{ 0.0f, 0.0f, 0.0f } };
    return oxr_utils::multiply_poses(pure_rotation, pose);
}

XrPosef invert_pose(const XrPosef& pose)
{
    XrPosef inverse{};
    inverse.orientation = conjugate_quaternion(pose.orientation);
    const XrPosef pure_rotation{ inverse.orientation, XrVector3f{ 0.0f, 0.0f, 0.0f } };
    const XrPosef negated_position{ XrQuaternionf{ 0.0f, 0.0f, 0.0f, 1.0f },
                                    XrVector3f{ -pose.position.x, -pose.position.y, -pose.position.z } };
    return oxr_utils::multiply_poses(pure_rotation, negated_position);
}

HandSkeleton build_hand_skeleton(const tf2_msgs::msg::TFMessage& msg, Hand hand)
{
    HandSkeleton skeleton;
    const bool right = hand == Hand::Right;
    const auto slots = make_hand_joint_slots(hand);

    // The wrist first: it is what makes this a hand the plugin can place in the room, and its pose
    // is what `wrist_alignment` is captured against. A message without it carries no hand.
    const auto* wrist = find_frame(msg, slots[static_cast<std::size_t>(XR_HAND_JOINT_WRIST_EXT)].frames);
    if (wrist == nullptr)
    {
        return skeleton;
    }
    const auto* base_frame = find_link(msg, Link::Base, hand);
    if (base_frame == nullptr || base_frame->header.frame_id != wrist->header.frame_id)
    {
        return HandSkeleton{};
    }
    const std::string& reference_frame = wrist->header.frame_id;

    MechanismAngles angles;
    if (!read_mechanism_angles(msg, hand, reference_frame, angles))
    {
        return HandSkeleton{};
    }

    // The geometry is expressed relative to the mechanism's base link, exactly as r15's table was
    // derived: the anatomical wrist is put at the skeleton origin by the fixed translation, so
    // WRIST -> MIDDLE_PROXIMAL is the human 91.3 mm whatever the rig's base link sits at. The wrist
    // itself is the glove's `<side>_hand` frame, which is what `wrist_alignment` captures.
    const Pose base = frame_pose(*base_frame, right);
    auto relative_pose = [&](const geometry_msgs::msg::TransformStamped& frame) -> Pose
    {
        const Pose pose = frame_pose(frame, right);
        const XrQuaternionf inverse = conjugate_quaternion(base.orientation);
        return Pose{ rotate_vector(inverse, subtract(pose.position, base.position)),
                     multiply_quaternions(inverse, pose.orientation) };
    };

    // The static row: PALM, the metacarpals, the four MCPs and the thumb CMC.
    for (const LandmarkAttachment& attachment : kLandmarkAttachments)
    {
        const auto* frame = find_link(msg, attachment.link, hand);
        if (frame == nullptr || frame->header.frame_id != reference_frame)
        {
            return HandSkeleton{};
        }
        const Pose link = relative_pose(*frame);
        XrPosef pose{};
        // The right hand uses the left table mirrored, except the thumb base, whose right URDF rest
        // frame is not a mirror (kRightThumbBaseOffset above).
        std::array<float, 3> offset = attachment.offset;
        if (right)
        {
            for (const RightAttachmentOverride& override_entry : kRightAttachmentOverrides)
            {
                if (override_entry.joint == attachment.joint)
                {
                    offset = override_entry.offset;
                }
            }
        }
        pose.position = to_skeleton(attach(link, offset));
        pose.orientation = XrQuaternionf{ 0.0f, 0.0f, 0.0f, 1.0f };
        skeleton.joints[static_cast<std::size_t>(attachment.joint)] = make_valid_joint(pose);
        skeleton.resolved[static_cast<std::size_t>(attachment.joint)] = link_name(attachment.link, hand);
    }

    const std::size_t side = right ? 1 : 0;

    // Every finger's target is the glove's own tip frame, expressed in this skeleton's frame exactly
    // like the landmarks: base-relative (mirrored into the left frame for the right hand) and then
    // through the fixed wrist scale/translation. All five targets are resolved before any chain is
    // solved, so a message missing a tip leaves the hand untracked instead of advancing the state.
    std::array<XrVector3f, kFingerCount> targets{};
    std::array<float, kFingerCount> chords{};
    for (std::size_t f = 0; f < std::size(kFingerChains); ++f)
    {
        const Link tip_link = static_cast<Link>(static_cast<int>(Link::IndexTip) + static_cast<int>(f));
        const auto* tip_frame = find_named(msg, link_name(tip_link, hand));
        if (tip_frame == nullptr || tip_frame->header.frame_id != reference_frame)
        {
            return HandSkeleton{};
        }
        targets[f + 1] = to_skeleton(relative_pose(*tip_frame).position);
        chords[f + 1] = frame_distance(msg, kMechanismFrames[side][f + 1][0], link_name(tip_link, hand));
        if (!(chords[f + 1] >= 0.0f))
        {
            return HandSkeleton{};
        }
    }
    const auto* thumb_tip_frame = find_named(msg, link_name(Link::ThumbTip, hand));
    if (thumb_tip_frame == nullptr || thumb_tip_frame->header.frame_id != reference_frame)
    {
        return HandSkeleton{};
    }
    targets[0] = to_skeleton(relative_pose(*thumb_tip_frame).position);
    chords[0] = frame_distance(msg, kMechanismFrames[side][0][0], link_name(Link::ThumbTip, hand));
    if (!(chords[0] >= 0.0f))
    {
        return HandSkeleton{};
    }
    // The glove's own thumb-to-index distance, in the glove frame: the only sign-stable measure of
    // which way the thumb is going, and what the inserted thumb's opposition is driven from.
    // The glove's own thumb-to-finger gaps, in the glove frame: which finger is being pinched is
    // decided by which of these is smallest, and each is what the thumb's opposition is driven from.
    std::array<float, kFingerCount - 1> gap_m{};
    for (std::size_t finger = 0; finger < gap_m.size(); ++finger)
    {
        const Link tip_link = static_cast<Link>(static_cast<int>(Link::IndexTip) + static_cast<int>(finger));
        gap_m[finger] = frame_distance(msg, link_name(Link::ThumbTip, hand), link_name(tip_link, hand));
        if (!(gap_m[finger] >= 0.0f))
        {
            return HandSkeleton{};
        }
    }

    // The session reference is fitted here, before any chain is laid out, so the frame that closes
    // the window already uses the capture. Only a complete hand reaches this point.
    capture_session_reference(side, skeleton, targets, chords, gap_m, angles);
    widen_open_reference(side, chords, gap_m, angles);

    // The four fingers, each anchored at its attached MCP, rolling the bend plane with the knuckle
    // spread and laying the constant human chain out at the session's bend fraction. r31: the
    // fraction comes from the mechanism's own joint angles (MCP/DIP), not the chord; a legacy
    // reference block or a capture fallback keeps the r30 chord map. The table's finger order is
    // thumb, index, middle, ring, little; the chains start at index, the thumb is solved below.
    for (std::size_t f = 0; f < std::size(kFingerChains); ++f)
    {
        const FingerChain& finger = kFingerChains[f];
        const float spread = map_spread(angles.joint[f + 1][0], kJointSpecs[side][f + 1][0]);
        float fraction = 0.0f;
        if (session_drives_digits(side))
        {
            fraction = session_angle_driver(side) ? finger_angle_fraction(side, f + 1, angles.joint[f + 1]) :
                                                    chord_fraction(side, f + 1, chords[f + 1]);
        }
        build_finger(skeleton, finger, targets[f + 1], fraction, spread, right, side, f + 1);
        skeleton.resolved[static_cast<std::size_t>(finger.joints[2])] = link_name(finger.intermediate, hand);
        skeleton.resolved[static_cast<std::size_t>(finger.joints[3])] = link_name(finger.distal, hand);
        skeleton.resolved[static_cast<std::size_t>(finger.joints[4])] = link_name(finger.tip, hand);
    }

    // The thumb is built last: its opposition target is one of the injected fingertips.
    std::array<XrVector3f, kFingerCount - 1> injected_tips{};
    for (std::size_t finger = 0; finger < injected_tips.size(); ++finger)
    {
        injected_tips[finger] = joint_position(skeleton, kFingerChains[finger].joints[4]);
    }
    build_thumb(skeleton, targets[0], gap_m, injected_tips, side);
    skeleton.resolved[static_cast<std::size_t>(XR_HAND_JOINT_THUMB_PROXIMAL_EXT)] = link_name(Link::Thumb2, hand);
    skeleton.resolved[static_cast<std::size_t>(XR_HAND_JOINT_THUMB_DISTAL_EXT)] = link_name(Link::Thumb3, hand);
    skeleton.resolved[static_cast<std::size_t>(XR_HAND_JOINT_THUMB_TIP_EXT)] = link_name(Link::ThumbTip, hand);

    skeleton.wrist_frame = wrist->child_frame_id;
    skeleton.wrist_pose = to_pose(wrist->transform);
    skeleton.joints[static_cast<std::size_t>(XR_HAND_JOINT_WRIST_EXT)] =
        make_valid_joint(XrPosef{ XrQuaternionf{ 0.0f, 0.0f, 0.0f, 1.0f }, XrVector3f{ 0.0f, 0.0f, 0.0f } });
    skeleton.resolved[static_cast<std::size_t>(XR_HAND_JOINT_WRIST_EXT)] = skeleton.wrist_frame;

    // PALM and the WRIST are anchors, not bones; their orientation stays identity.
    skeleton.joints[static_cast<std::size_t>(XR_HAND_JOINT_PALM_EXT)].pose.orientation =
        XrQuaternionf{ 0.0f, 0.0f, 0.0f, 1.0f };

    orient_chain(skeleton, kThumbChain);
    for (const FingerChain& finger : kFingerChains)
    {
        orient_chain(skeleton, finger.joints);
    }

    skeleton.filled = kOpenXrHandJointCount;

    if (right)
    {
        // The right hand is the X mirror of the left. The input frames were mirrored into the left
        // frame above; mirror the finished skeleton back so the table and the axes stay one copy.
        for (XrHandJointLocationEXT& joint : skeleton.joints)
        {
            joint.pose.position.x = -joint.pose.position.x;
            joint.pose.orientation = XrQuaternionf{ joint.pose.orientation.x, -joint.pose.orientation.y,
                                                    -joint.pose.orientation.z, joint.pose.orientation.w };
        }
    }

    return skeleton;
}

} // namespace io_exo_glove_ros2
} // namespace plugins
