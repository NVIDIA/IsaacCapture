// SPDX-FileCopyrightText: Copyright (c) 2026 IO. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "io_exo_glove_ros2_pack.hpp"

#include <openxr/openxr.h>
#include <tf2_msgs/msg/tf_message.hpp>

#include <array>
#include <cstddef>
#include <cstring>
#include <string>
#include <utility>

namespace plugins
{
namespace io_exo_glove_ros2
{

//! OpenXR hand joints; all of them are filled while a hand is active.
inline constexpr std::size_t kOpenXrHandJointCount = XR_HAND_JOINT_COUNT_EXT;

//! Radius the injected joints carry, in metres. The exoskeleton reports poses, not joint sizes.
inline constexpr float kHandJointRadius = 0.01f;

//! Flags of a joint whose pose is filled; the TRACKED bits are added once the wrist is tracked.
inline constexpr XrSpaceLocationFlags kPoseValidFlags =
    XR_SPACE_LOCATION_POSITION_VALID_BIT | XR_SPACE_LOCATION_ORIENTATION_VALID_BIT;
inline constexpr XrSpaceLocationFlags kPoseTrackedFlags =
    kPoseValidFlags | XR_SPACE_LOCATION_POSITION_TRACKED_BIT | XR_SPACE_LOCATION_ORIENTATION_TRACKED_BIT;

//! Joints the injected hand carries: all 26 OpenXR joints.
inline constexpr std::size_t kHandJointSlotCount = kOpenXrHandJointCount;

//! Fingers the injected hand drives, in the order of ``HandSkeleton::tip_residual_m``: thumb, index,
//! middle, ring, little.
inline constexpr std::size_t kFingerCount = 5;

/*!
 * @brief A fixed-capacity string for the per-rig frame names the plugin logs.
 *
 * ``HandSkeleton`` is filled on every link message, and its frame names are constant for a rig. A
 * ``std::string`` would allocate on every frame for names longer than the small-string buffer, which
 * the hot path must not do; this type offers the subset of the ``std::string`` interface the plugin
 * uses (``c_str()``, ``empty()``, assignment from a ``std::string``/``const char*`` and conversion
 * back for concatenation) and stores the characters inline.
 */
template <std::size_t Capacity>
class FixedString
{
public:
    FixedString() = default;
    FixedString& operator=(const std::string& value)
    {
        assign(value.c_str());
        return *this;
    }
    FixedString& operator=(const char* value)
    {
        assign(value);
        return *this;
    }
    bool empty() const
    {
        return m_size == 0;
    }
    const char* c_str() const
    {
        return m_data.data();
    }
    const char* data() const
    {
        return m_data.data();
    }
    std::size_t size() const
    {
        return m_size;
    }
    operator const char*() const
    {
        return m_data.data();
    }
    bool operator==(const char* other) const
    {
        return std::strcmp(m_data.data(), other) == 0;
    }
    bool operator!=(const char* other) const
    {
        return !(*this == other);
    }

private:
    void assign(const char* value)
    {
        const std::size_t length = value == nullptr ? 0 : std::strlen(value);
        m_size = length < Capacity ? length : Capacity - 1;
        std::memcpy(m_data.data(), value == nullptr ? "" : value, m_size);
        m_data[m_size] = '\0';
    }

    std::array<char, Capacity> m_data{};
    std::size_t m_size = 0;
};

//! Concatenation support so the plugin's startup log can append an inline name to a std::string
//! without allocating a temporary. Found by ADL: the right operand is in this namespace.
template <std::size_t N>
std::string operator+(std::string&& lhs, const FixedString<N>& rhs)
{
    lhs += rhs.c_str();
    return std::move(lhs);
}
template <std::size_t N>
std::string operator+(const std::string& lhs, const FixedString<N>& rhs)
{
    return lhs + rhs.c_str();
}

//! One OpenXR joint and the exoskeleton frames that can carry it, best candidate first. The wrist
//! resolves against the exoskeleton's hand/base frames; every other joint rides a linkage frame.
struct HandJointSlot
{
    XrHandJointEXT joint;
    std::array<const char*, 3> frames;
};

//! Slots for one hand, in OpenXR joint order. Static storage: the returned array only borrows names.
std::array<HandJointSlot, kHandJointSlotCount> make_hand_joint_slots(Hand hand);

//! One hand's skeleton, wrist-relative: the wrist is the identity pose and every other joint is an
//! anatomical landmark driven by the exoskeleton's own frames.
struct HandSkeleton
{
    //! Wrist-relative joints, indexed by ``XrHandJointEXT``. Every entry starts invalid; a hand the
    //! message does not carry stays that way: absence is not a pose at the origin.
    std::array<XrHandJointLocationEXT, kOpenXrHandJointCount> joints{};

    //! Joints resolved from this message. A complete hand fills all 26; a partial hand is not
    //! injected, so the only other value is 0.
    std::size_t filled = 0;

    //! Distance from each solved fingertip to its CCD target, metres, indexed [thumb, index, middle,
    //! ring, little]. The target is the glove tip's direction in this skeleton's frame at the radius
    //! the session-normalised chord gives, so a residual above the solver tolerance means the
    //! constant human chain could not reach that direction.
    std::array<float, kFingerCount> tip_residual_m{};

    //! Which finger the thumb is opposing towards, as an index into the four fingers (0 = index,
    //! 3 = little), or -1 when there is no thumb reference. It is the finger whose glove tip is
    //! nearest the glove thumb tip, i.e. the one being pinched.
    int thumb_target = -1;

    //! The CCD target each finger was solved onto, metres, in this skeleton's frame, same order. The
    //! target is `knuckle + map_chord_radius(chord) * normalize(glove tip - knuckle)` after the
    //! session's direction offset.
    std::array<XrVector3f, kFingerCount> tip_target_m{};

    //! Glove frame the wrist came from; empty when this hand is absent from the message.
    FixedString<48> wrist_frame;

    //! Wrist pose in the glove's own frame, for the one-time alignment onto the tracked wrist. It is
    //! the exoskeleton's published hand frame, exactly as in BASE.
    XrPosef wrist_pose{ { 0.0f, 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, 0.0f } };

    //! Frame each joint's geometry was read from, for the one-time startup log; empty when unresolved.
    std::array<FixedString<48>, kHandJointSlotCount> resolved;

    bool tracked() const
    {
        return !wrist_frame.empty();
    }
};

/*!
 * @brief Reads one hand's anatomical skeleton out of @p msg.
 *
 * The landmarks, the knuckle spread and the thumb mount come from the published link poses exactly
 * as before. Each finger's flexion is the fixed human chain (MCP -> PIP -> DIP -> tip) driven by the
 * exoskeleton's own mechanism joint angles: the MCP and DIP absolute angles (v5 URDF rest + axis)
 * are normalised between this session's open-pose angle and the baked fist angle, and the chain's
 * joints are that fraction of their human limits (85/95/70). The thumb keeps the gap-driven
 * opposition and its own 35/55/75 hinges. A legacy reference block or a capture fallback runs the
 * r30 chord driver instead.
 *
 * The tip-frame orientation is not a rig constant, so the first frames after the plugin starts are
 * captured once per hand: the open chord and the rotation from the device's open tip direction to
 * the digit's human rest axis. That rotation is applied to every live device direction afterwards.
 * The same measurements can be installed from a config block (see install_session_reference()), in
 * which case no window runs and the first frame already maps. Until the window closes every digit
 * is held open; a window that cannot be satisfied before its deadline falls back to the baked table
 * and the hand is injected regardless. A message missing any frame the build needs yields an
 * untracked skeleton, because half a hand is never injected. ``tip_residual_m`` reports how far each
 * solved tip ended from its target.
 */
HandSkeleton build_hand_skeleton(const tf2_msgs::msg::TFMessage& msg, Hand hand);

/*!
 * @brief One digit's session open reference, as captured (or as baked when the capture fell back).
 *
 * ``finger`` is in the order of ``HandSkeleton::tip_residual_m``: thumb, index, middle, ring,
 * little. ``offset_angle_rad`` is the rotation that takes the live device tip direction onto the
 * digit's rest axis, i.e. how far this session's tip frame sits from the human axis.
 */
struct SessionOpenReference
{
    float chord_open_m = 0.0f;
    float offset_angle_rad = 0.0f;
    std::size_t frames = 0;
    bool captured = false;
    bool fallback = false;
    //! True when the hand is driven from its own mechanism joint angles (r31), i.e. the open
    //! angles are available. False when the chord driver is in use (a legacy block or a fallback).
    bool angle_driver = false;
};
SessionOpenReference session_open_reference(Hand hand, std::size_t finger);

//! The session open pose's absolute mechanism joint angle, radians, for one hand/finger/slot.
//! Zero when the hand has no angle reference (legacy block or fallback): the chord driver is then
//! in use. Slot order is the URDF chain (spread/mount, then MCP/CMC, PIP/MCP, DIP/IP).
float session_angle_open(Hand hand, std::size_t finger, std::size_t slot);

//! The hand's open deadband in radians of model bend (0 when the block carries no deadband, or the
//! chord driver is in use). This is the hand-level value from the reference; the driver converts it
//! to a fraction of each hinge's own travel (see joint_angle_fraction).
float session_angle_deadband(Hand hand);

//! True when this hand's digits are driven from the mechanism joint angles rather than the chord:
//! a new-format reference was loaded, or the capture window measured the open angles.
bool session_angle_driver(Hand hand);

//! True once the hand's open reference was captured from its own first frames.
bool session_reference_captured(Hand hand);

//! True when the window missed its deadline and the baked table is in use. The hand is still injected.
bool session_reference_fallback(Hand hand);

//! The thumb chain's rest (zero-opposition) direction for this skeleton, in its frame: in the palm
//! plane, abducted from the index metacarpal by kThumbRestAbductionRad. The CMC hinge is measured
//! against it, so diagnostics do not mistake the abduction for flexion.
XrVector3f thumb_rest_direction(const HandSkeleton& skeleton);

//! True when the hand's frames did not change in the window (a stale/frozen stream). Such a hand is
//! held open/neutral instead of being read as flexion, and the capture line says so.
bool session_reference_stale(Hand hand);

/*!
 * @brief The glove's own thumb-to-index distance, which drives the inserted thumb's opposition.
 *
 * ``open_m`` is the opening distance measured in the window and ``low_m`` the smallest distance seen
 * this session; the opposition runs from 0 at ``open_m`` to 1 at ``low_m``. Both are 0 (and
 * ``valid`` false) before the window closes, after a fallback, and on a stale stream.
 */
struct SessionOppositionReference
{
    float open_m = 0.0f;
    float low_m = 0.0f;
    bool valid = false;
    bool frozen = false;
};
SessionOppositionReference session_opposition_reference(Hand hand);

/*!
 * @brief One hand's persistent session reference: everything the start-up window used to learn.
 *
 * With a complete block in the config (or from ``--session-reference=``) the window is skipped and
 * the hand is mapped from the first frame. The on-disk spelling is documented in README.md; it
 * carries, per hand, the open chord and the direction-offset rotation for all five digits and the
 * opening and tightest thumb-to-finger gap for the four fingers.
 */
struct SessionReferenceBlock
{
    //! Open chord per digit, metres, in the order of ``HandSkeleton::tip_residual_m`` (thumb, index,
    //! middle, ring, little). The thumb's travel is inverted, so its open chord is its smallest.
    std::array<float, kFingerCount> chord_open_m{};
    //! Rotation that takes the device's open tip direction onto the digit's human rest axis.
    std::array<XrQuaternionf, kFingerCount> direction_offset{};
    //! Glove thumb-to-finger gap at the open pose and at its tightest, metres (index..little).
    std::array<float, kFingerCount - 1> gap_open_m{};
    std::array<float, kFingerCount - 1> gap_tight_m{};
    //! Absolute mechanism joint angle at the session's open pose, radians, [finger][slot]. Present
    //! only in the r31 format; a block written by an earlier round leaves @c has_angles false and is
    //! still accepted (the hand then runs the chord driver, so an old block cannot leave it unmapped).
    std::array<std::array<float, 4>, kFingerCount> angle_open_rad{};
    //! r35 open deadband, radians of *model* bend (a hand-level scalar). It is stored only by a
    //! five-section block; a four-section r31 block leaves @c has_deadband false and uses 0, which is
    //! the r31 map bit for bit. The value is the measured relaxed-vs-stiff straight offset plus the
    //! margin, capped at 25 deg. It is converted to a fraction of each joint's own travel, so the
    //! closed end is never compressed: a joint at @c kAngleClosed still reads exactly 1.
    float angle_deadband_rad = 0.0f;
    bool has_deadband = false;
    bool has_angles = false;
};

//! Outcome of installing one hand's block at start-up.
enum class SessionReferenceLoad
{
    NotProvided, //!< No block for this hand: the window captures the reference as before.
    Loaded, //!< A complete, valid block: the window is skipped.
    Rejected, //!< A block was provided but is invalid: warned about, the window captures instead.
};

//! Parses one hand's block (``"chord thumb=...; offset thumb=...; gap index=.../..."``).
//!
//! Rejects a truncated or unknown block, a non-unit quaternion, a chord outside 20-160 mm, a
//! negative gap and a tightest gap larger than the open one, writing a one-line reason into @p error.
//! Not on the hot path; it runs once per hand at start-up.
bool parse_session_reference_block(const char* text, SessionReferenceBlock& block, std::string& error);

/*!
 * @brief Installs one hand's block before the first frame and queues its one-line startup log.
 *
 * A valid block makes the hand drive from the first frame and queues
 * ``session reference: <side>: loaded from config``. A missing block queues why the window will
 * capture instead; an invalid one queues the reason as a warning. Either way the window captures a
 * reference exactly as in r27, so a broken block can never leave the hand unmapped.
 */
SessionReferenceLoad install_session_reference(Hand hand, const char* text);

//! Formats the hand's current (captured or loaded) reference as the block text; false when the hand
//! has no usable reference. Not on the hot path.
bool format_session_reference_block(Hand hand, std::string& text);

//! True when the hand has a usable reference: captured or loaded, and neither fallen back nor stale.
bool session_reference_available(Hand hand);

//! True when that reference came from the config rather than this run's capture window.
bool session_reference_loaded(Hand hand);

/*!
 * @brief Writes the YAML snippet (comment header plus ``reference_left``/``reference_right``).
 *
 * The header carries the date, @p plugin_version and, per hand, whether the block was captured this
 * run (with its frame count and duration) or loaded from the config. Hands without a usable
 * reference are omitted, so a one-handed rig exports the one block it has. The snippet is exactly
 * what the plugin's config (or ``--session-reference=``) accepts.
 *
 * Returns false with a reason in @p error when nothing is available or the file cannot be written.
 */
bool export_session_reference_file(const std::string& path, const char* plugin_version, std::string& error);

/*!
 * @brief The "session open reference captured: ..." (or fallback) log line for @p hand.
 *
 * Returns the line exactly once, when the window closes (or when a reference block is installed
 * or rejected), and an empty string otherwise. The text is static storage per hand; nothing here
 * allocates, so it may be called from the frame callback.
 */
const char* take_session_log_line(Hand hand);

//! True when the pending log line is a warning (a fallback, a stale stream or a rejected block).
bool session_log_line_is_warning(Hand hand);

//! Forgets any per-session state: the warm-started angles and the open-reference window.
void reset_hand_build_state();

//! Composes @p rotation onto @p pose: the rotation is applied to the pose's orientation and to its
//! position, i.e. the pose is re-expressed one frame further out.
XrPosef rotate_pose(const XrQuaternionf& rotation, const XrPosef& pose);

//! Inverse of a rigid pose, for the wrist-relative step.
XrPosef invert_pose(const XrPosef& pose);

//! Hamilton product, and conjugate (the inverse for a unit quaternion).
XrQuaternionf multiply_quaternions(const XrQuaternionf& a, const XrQuaternionf& b);
XrQuaternionf conjugate_quaternion(const XrQuaternionf& q);

} // namespace io_exo_glove_ros2
} // namespace plugins
