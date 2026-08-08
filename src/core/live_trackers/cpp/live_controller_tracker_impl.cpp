// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "live_controller_tracker_impl.hpp"

#include <mcap/recording_traits.hpp>
#include <oxr_utils/oxr_funcs.hpp>
#include <schema/controller_bfbs_generated.h>
#include <schema/serialized.hpp>
#include <schema/timestamp_generated.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <optional>
#include <stdexcept>

namespace core
{

namespace
{

XrActionSetPtr createActionSetInContext(const OpenXRCoreFunctions& funcs, XrInstance instance, XrInstanceActionContextNV ctx)
{
    XrInstanceActionContextInfoNV ctx_info{ XR_TYPE_INSTANCE_ACTION_CONTEXT_INFO_NV };
    ctx_info.instanceActionContext = ctx;

    XrActionSetCreateInfo set_info{ XR_TYPE_ACTION_SET_CREATE_INFO };
    set_info.next = &ctx_info;
    strcpy(set_info.actionSetName, "controller_tracking");
    strcpy(set_info.localizedActionSetName, "Controller Tracking");
    set_info.priority = 0;

    return createActionSet(funcs, instance, set_info);
}

// ---- OpenXR action helpers ----

XrPath xr_path_from_string(const OpenXRCoreFunctions& funcs, XrInstance instance, const char* s)
{
    XrPath path = XR_NULL_PATH;
    XrResult res = funcs.xrStringToPath(instance, s, &path);
    if (XR_FAILED(res))
    {
        throw std::runtime_error(std::string("xrStringToPath failed for '") + s + "': " + std::to_string(res));
    }
    return path;
}

bool get_boolean_action_state(XrSession session, const OpenXRCoreFunctions& core_funcs, XrAction action, XrPath subaction_path)
{
    XrActionStateGetInfo get_info{ XR_TYPE_ACTION_STATE_GET_INFO };
    get_info.action = action;
    get_info.subactionPath = subaction_path;

    XrActionStateBoolean state{ XR_TYPE_ACTION_STATE_BOOLEAN };
    XrResult result = core_funcs.xrGetActionStateBoolean(session, &get_info, &state);
    if (XR_FAILED(result))
    {
        throw std::runtime_error("xrGetActionStateBoolean failed: " + std::to_string(result));
    }
    if (state.isActive)
    {
        return state.currentState;
    }
    return false;
}

float get_float_action_state(XrSession session, const OpenXRCoreFunctions& core_funcs, XrAction action, XrPath subaction_path)
{
    XrActionStateGetInfo get_info{ XR_TYPE_ACTION_STATE_GET_INFO };
    get_info.action = action;
    get_info.subactionPath = subaction_path;

    XrActionStateFloat state{ XR_TYPE_ACTION_STATE_FLOAT };
    XrResult result = core_funcs.xrGetActionStateFloat(session, &get_info, &state);
    if (XR_FAILED(result))
    {
        throw std::runtime_error("xrGetActionStateFloat failed: " + std::to_string(result));
    }
    if (state.isActive)
    {
        return state.currentState;
    }
    return 0.0f;
}

bool get_vector2_action_state(XrSession session,
                              const OpenXRCoreFunctions& core_funcs,
                              XrAction action,
                              XrPath subaction_path,
                              float& out_x,
                              float& out_y)
{
    XrActionStateGetInfo get_info{ XR_TYPE_ACTION_STATE_GET_INFO };
    get_info.action = action;
    get_info.subactionPath = subaction_path;

    XrActionStateVector2f state{ XR_TYPE_ACTION_STATE_VECTOR2F };
    XrResult result = core_funcs.xrGetActionStateVector2f(session, &get_info, &state);
    if (XR_FAILED(result))
    {
        throw std::runtime_error("xrGetActionStateVector2f failed: " + std::to_string(result));
    }
    if (state.isActive)
    {
        out_x = state.currentState.x;
        out_y = state.currentState.y;
        return true;
    }
    out_x = out_y = 0.0f;
    return false;
}

bool get_pose_action_active(XrSession session, const OpenXRCoreFunctions& core_funcs, XrAction action, XrPath subaction_path)
{
    XrActionStateGetInfo get_info{ XR_TYPE_ACTION_STATE_GET_INFO };
    get_info.action = action;
    get_info.subactionPath = subaction_path;

    XrActionStatePose state{ XR_TYPE_ACTION_STATE_POSE };
    XrResult result = core_funcs.xrGetActionStatePose(session, &get_info, &state);
    if (XR_FAILED(result))
    {
        throw std::runtime_error("xrGetActionStatePose failed: " + std::to_string(result));
    }
    return state.isActive;
}

XrSpacePtr create_space(const OpenXRCoreFunctions& funcs, XrSession session, XrAction action, XrPath subaction_path)
{
    assert(action != XR_NULL_HANDLE);
    assert(subaction_path != XR_NULL_PATH);

    XrActionSpaceCreateInfo space_info{ XR_TYPE_ACTION_SPACE_CREATE_INFO };
    space_info.action = action;
    space_info.subactionPath = subaction_path;
    space_info.poseInActionSpace.orientation = { 0.0f, 0.0f, 0.0f, 1.0f };
    space_info.poseInActionSpace.position = { 0.0f, 0.0f, 0.0f };

    return createActionSpace(funcs, session, &space_info);
}

XrAction create_action(const OpenXRCoreFunctions& funcs,
                       XrActionSet action_set,
                       XrPath left_hand_path,
                       XrPath right_hand_path,
                       const char* name,
                       const char* localized_name,
                       XrActionType type)
{
    XrAction out_action;

    XrPath hand_paths[2] = { left_hand_path, right_hand_path };

    XrActionCreateInfo action_info{ XR_TYPE_ACTION_CREATE_INFO };
    action_info.actionType = type;
    strcpy(action_info.actionName, name);
    strcpy(action_info.localizedActionName, localized_name);
    action_info.countSubactionPaths = 2;
    action_info.subactionPaths = hand_paths;

    XrResult res = funcs.xrCreateAction(action_set, &action_info, &out_action);
    if (XR_FAILED(res))
    {
        throw std::runtime_error(std::string("Failed to create action ") + name + ": " + std::to_string(res));
    }

    return out_action;
}

} // anonymous namespace

// ============================================================================
// LiveControllerTrackerImpl
// ============================================================================

std::unique_ptr<ControllerMcapChannels> LiveControllerTrackerImpl::create_mcap_channels(mcap::McapWriter& writer,
                                                                                        std::string_view base_name)
{
    return std::make_unique<ControllerMcapChannels>(
        writer, base_name,
        std::vector<std::string>(ControllerRecordingTraits::recording_channels.begin(),
                                 ControllerRecordingTraits::recording_channels.end()));
}

LiveControllerTrackerImpl::LiveControllerTrackerImpl(const OpenXRSessionHandles& handles,
                                                     std::unique_ptr<ControllerMcapChannels> mcap_channels)
    : core_funcs_(OpenXRCoreFunctions::load(handles.instance, handles.xrGetInstanceProcAddr)),
      time_converter_(handles),
      instance_(handles.instance),
      session_(handles.session),
      base_space_(handles.space),
      left_hand_path_(xr_path_from_string(core_funcs_, handles.instance, "/user/hand/left")),
      right_hand_path_(xr_path_from_string(core_funcs_, handles.instance, "/user/hand/right")),
      action_ctx_funcs_(ActionContextFunctions::load(handles.instance, handles.xrGetInstanceProcAddr)),
      instance_action_context_(createInstanceActionContext(action_ctx_funcs_, handles.instance)),
      session_action_context_(nullptr, nullptr),
      action_set_(createActionSetInContext(core_funcs_, handles.instance, instance_action_context_.get())),
      grip_pose_action_(create_action(core_funcs_,
                                      action_set_.get(),
                                      left_hand_path_,
                                      right_hand_path_,
                                      "grip_pose",
                                      "Grip Pose",
                                      XR_ACTION_TYPE_POSE_INPUT)),
      aim_pose_action_(create_action(
          core_funcs_, action_set_.get(), left_hand_path_, right_hand_path_, "aim_pose", "Aim Pose", XR_ACTION_TYPE_POSE_INPUT)),
      primary_click_action_(create_action(core_funcs_,
                                          action_set_.get(),
                                          left_hand_path_,
                                          right_hand_path_,
                                          "primary_click",
                                          "Primary Click",
                                          XR_ACTION_TYPE_BOOLEAN_INPUT)),
      secondary_click_action_(create_action(core_funcs_,
                                            action_set_.get(),
                                            left_hand_path_,
                                            right_hand_path_,
                                            "secondary_click",
                                            "Secondary Click",
                                            XR_ACTION_TYPE_BOOLEAN_INPUT)),
      thumbstick_action_(create_action(core_funcs_,
                                       action_set_.get(),
                                       left_hand_path_,
                                       right_hand_path_,
                                       "thumbstick",
                                       "Thumbstick",
                                       XR_ACTION_TYPE_VECTOR2F_INPUT)),
      thumbstick_click_action_(create_action(core_funcs_,
                                             action_set_.get(),
                                             left_hand_path_,
                                             right_hand_path_,
                                             "thumbstick_click",
                                             "Thumbstick Click",
                                             XR_ACTION_TYPE_BOOLEAN_INPUT)),
      menu_click_action_(create_action(core_funcs_,
                                       action_set_.get(),
                                       left_hand_path_,
                                       right_hand_path_,
                                       "menu_click",
                                       "Menu Click",
                                       XR_ACTION_TYPE_BOOLEAN_INPUT)),
      squeeze_value_action_(create_action(core_funcs_,
                                          action_set_.get(),
                                          left_hand_path_,
                                          right_hand_path_,
                                          "squeeze_value",
                                          "Squeeze Value",
                                          XR_ACTION_TYPE_FLOAT_INPUT)),
      trigger_value_action_(create_action(core_funcs_,
                                          action_set_.get(),
                                          left_hand_path_,
                                          right_hand_path_,
                                          "trigger_value",
                                          "Trigger Value",
                                          XR_ACTION_TYPE_FLOAT_INPUT)),
      haptic_action_(create_action(core_funcs_,
                                   action_set_.get(),
                                   left_hand_path_,
                                   right_hand_path_,
                                   "haptic_output",
                                   "Haptic Output",
                                   XR_ACTION_TYPE_VIBRATION_OUTPUT)),
      left_grip_space_(create_space(core_funcs_, session_, grip_pose_action_, left_hand_path_)),
      right_grip_space_(create_space(core_funcs_, session_, grip_pose_action_, right_hand_path_)),
      left_aim_space_(create_space(core_funcs_, session_, aim_pose_action_, left_hand_path_)),
      right_aim_space_(create_space(core_funcs_, session_, aim_pose_action_, right_hand_path_)),
      mcap_channels_(std::move(mcap_channels))
{
    // Suggest interaction profile bindings (chained to this action context)
    std::vector<XrActionSuggestedBinding> bindings;
    auto add_binding = [&](XrAction action, const char* path) {
        bindings.push_back({ action, xr_path_from_string(core_funcs_, handles.instance, path) });
    };

    add_binding(grip_pose_action_, "/user/hand/left/input/grip/pose");
    add_binding(grip_pose_action_, "/user/hand/right/input/grip/pose");
    add_binding(aim_pose_action_, "/user/hand/left/input/aim/pose");
    add_binding(aim_pose_action_, "/user/hand/right/input/aim/pose");
    add_binding(thumbstick_action_, "/user/hand/left/input/thumbstick");
    add_binding(thumbstick_action_, "/user/hand/right/input/thumbstick");
    add_binding(thumbstick_click_action_, "/user/hand/left/input/thumbstick/click");
    add_binding(thumbstick_click_action_, "/user/hand/right/input/thumbstick/click");
    add_binding(squeeze_value_action_, "/user/hand/left/input/squeeze/value");
    add_binding(squeeze_value_action_, "/user/hand/right/input/squeeze/value");
    add_binding(trigger_value_action_, "/user/hand/left/input/trigger/value");
    add_binding(trigger_value_action_, "/user/hand/right/input/trigger/value");
    add_binding(primary_click_action_, "/user/hand/left/input/x/click");
    add_binding(secondary_click_action_, "/user/hand/left/input/y/click");
    add_binding(primary_click_action_, "/user/hand/right/input/a/click");
    add_binding(secondary_click_action_, "/user/hand/right/input/b/click");
    // Oculus Touch exposes menu only on the left controller; right hand has no such path,
    // so we bind left only. Right-hand menu_click will report false (action inactive).
    add_binding(menu_click_action_, "/user/hand/left/input/menu/click");
    // Haptic output bindings — one per side onto the standard haptic component path
    // every conformant motion-controller interaction profile exposes.
    add_binding(haptic_action_, "/user/hand/left/output/haptic");
    add_binding(haptic_action_, "/user/hand/right/output/haptic");

    XrInstanceActionContextInfoNV binding_ctx_info{ XR_TYPE_INSTANCE_ACTION_CONTEXT_INFO_NV };
    binding_ctx_info.instanceActionContext = instance_action_context_.get();

    XrInteractionProfileSuggestedBinding suggested_bindings{ XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING };
    suggested_bindings.next = &binding_ctx_info;
    suggested_bindings.interactionProfile =
        xr_path_from_string(core_funcs_, handles.instance, "/interaction_profiles/oculus/touch_controller");
    suggested_bindings.countSuggestedBindings = static_cast<uint32_t>(bindings.size());
    suggested_bindings.suggestedBindings = bindings.data();

    XrResult result = core_funcs_.xrSuggestInteractionProfileBindings(handles.instance, &suggested_bindings);
    if (XR_FAILED(result))
    {
        throw std::runtime_error("Failed to suggest interaction profile bindings: " + std::to_string(result));
    }

    // Also offer the PICO profile. The runtime binds whichever matches the
    // connected hardware, which is what makes the headset identifiable through
    // a streaming runtime that presents every device identically.
    //
    // Suggested separately and tolerantly on purpose: the call rejects a whole
    // profile if any binding path is unsupported, and controller input must keep
    // working even when a profile is refused. Measured accepted on both a PICO 4
    // Ultra and a Quest 3.
    XrInteractionProfileSuggestedBinding pico_bindings{ XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING };
    pico_bindings.next = &binding_ctx_info;
    pico_bindings.interactionProfile =
        xr_path_from_string(core_funcs_, handles.instance, "/interaction_profiles/bytedance/pico4_controller");
    pico_bindings.countSuggestedBindings = static_cast<uint32_t>(bindings.size());
    pico_bindings.suggestedBindings = bindings.data();
    if (XR_FAILED(core_funcs_.xrSuggestInteractionProfileBindings(handles.instance, &pico_bindings)))
    {
        std::cout << "[ControllerTracker] PICO interaction profile not accepted; headset identification "
                     "will report no profile"
                  << std::endl;
    }

    // Create session action context (makes the instance context immutable)
    session_action_context_ =
        createSessionActionContext(action_ctx_funcs_, handles.session, instance_action_context_.get());

    // Attach action sets to the session action context
    XrSessionActionContextInfoNV sess_ctx_info{ XR_TYPE_SESSION_ACTION_CONTEXT_INFO_NV };
    sess_ctx_info.sessionActionContext = session_action_context_.get();

    XrActionSet action_set_handle = action_set_.get();
    XrSessionActionSetsAttachInfo attach_info{ XR_TYPE_SESSION_ACTION_SETS_ATTACH_INFO };
    attach_info.next = &sess_ctx_info;
    attach_info.countActionSets = 1;
    attach_info.actionSets = &action_set_handle;

    result = core_funcs_.xrAttachSessionActionSets(handles.session, &attach_info);
    if (XR_FAILED(result))
    {
        throw std::runtime_error("Failed to attach action sets: " + std::to_string(result));
    }

    logger_->info("ControllerTracker initialized (left + right) with action context");
}

std::string LiveControllerTrackerImpl::get_interaction_profile() const
{
    return interaction_profile_;
}

std::string LiveControllerTrackerImpl::query_interaction_profile() const
{
    if (action_ctx_funcs_.get_current_interaction_profile == nullptr)
    {
        return {};
    }

    // Both hands normally report the same profile; take the left and fall back to
    // the right so one inactive controller does not hide the headset's identity.
    for (const XrPath hand_path : { left_hand_path_, right_hand_path_ })
    {
        XrInteractionProfileGetInfo2NV get_info{ XR_TYPE_INTERACTION_PROFILE_GET_INFO_2_NV };
        get_info.topLevelUserPath = hand_path;
        get_info.sessionActionContext = session_action_context_.get();

        XrInteractionProfileState profile_state{ XR_TYPE_INTERACTION_PROFILE_STATE };
        if (XR_FAILED(action_ctx_funcs_.get_current_interaction_profile(session_, &get_info, &profile_state)) ||
            profile_state.interactionProfile == XR_NULL_PATH)
        {
            continue;
        }

        char buf[XR_MAX_PATH_LENGTH]{};
        uint32_t written = 0;
        if (XR_SUCCEEDED(
                core_funcs_.xrPathToString(instance_, profile_state.interactionProfile, sizeof(buf), &written, buf)))
        {
            return std::string(buf);
        }
    }
    return {};
}

void LiveControllerTrackerImpl::update(int64_t monotonic_time_ns)
{
    // Invalidate first, publish last: the encodes below are the only writers, so no exit
    // path can leave a caller reading last frame's poses. That matters for both hands at
    // once here -- a locate failure on the left aborts before the right is even queried.
    left_tracked_.reset();
    right_tracked_.reset();
    last_update_time_ = monotonic_time_ns;
    const XrTime xr_time = time_converter_.convert_monotonic_ns_to_xrtime(monotonic_time_ns);

    // Sync actions via xrSyncActions2NV with our session action context
    XrActiveActionSet active_action_set{ action_set_.get(), XR_NULL_PATH };

    XrActionsSyncInfo2NV sync_info{ XR_TYPE_ACTIONS_SYNC_INFO_2_NV };
    sync_info.countActiveActionSets = 1;
    sync_info.activeActionSets = &active_action_set;
    sync_info.sessionActionContext = session_action_context_.get();

    XrActionsSyncState2NV sync_state{ XR_TYPE_ACTIONS_SYNC_STATE_2_NV };

    XrResult result = action_ctx_funcs_.sync_actions_2(session_, &sync_info, &sync_state);
    if (XR_FAILED(result))
    {
        // Policy: action sync failure is a critical tracker/runtime error.
        throw std::runtime_error("[ControllerTracker] xrSyncActions2NV failed: " + std::to_string(result));
    }

    // Nothing is bound until actions have synced, so this cannot be resolved at
    // construction. Re-read only when the runtime reports a change (a headset
    // swap), which keeps this off the per-frame IPC path.
    if (interaction_profile_.empty() || sync_state.interactionProfileChanged)
    {
        interaction_profile_ = query_interaction_profile();
    }

    std::optional<ControllerSnapshotT> left_native;
    std::optional<ControllerSnapshotT> right_native;

    auto update_controller = [&](XrPath hand_path, const XrSpacePtr& grip_space, const XrSpacePtr& aim_space,
                                 std::optional<ControllerSnapshotT>& tracked)
    {
        if (!get_pose_action_active(session_, core_funcs_, grip_pose_action_, hand_path))
        {
            // Policy: controller not active is a common runtime condition.
            tracked.reset();
            return;
        }

        ControllerPose grip_pose{};
        ControllerPose aim_pose{};

        XrSpaceLocation grip_location{ XR_TYPE_SPACE_LOCATION };
        result = core_funcs_.xrLocateSpace(grip_space.get(), base_space_, xr_time, &grip_location);
        if (XR_FAILED(result))
        {
            throw std::runtime_error("[ControllerTracker] xrLocateSpace(grip) failed: " + std::to_string(result));
        }
        if (XR_SUCCEEDED(result))
        {
            bool is_valid = (grip_location.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT) &&
                            (grip_location.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT);
            if (is_valid)
            {
                Point position(
                    grip_location.pose.position.x, grip_location.pose.position.y, grip_location.pose.position.z);
                Quaternion orientation(grip_location.pose.orientation.x, grip_location.pose.orientation.y,
                                       grip_location.pose.orientation.z, grip_location.pose.orientation.w);
                grip_pose = ControllerPose(Pose(position, orientation), true);
            }
        }

        XrSpaceLocation aim_location{ XR_TYPE_SPACE_LOCATION };
        result = core_funcs_.xrLocateSpace(aim_space.get(), base_space_, xr_time, &aim_location);
        if (XR_FAILED(result))
        {
            throw std::runtime_error("[ControllerTracker] xrLocateSpace(aim) failed: " + std::to_string(result));
        }
        if (XR_SUCCEEDED(result))
        {
            bool is_valid = (aim_location.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT) &&
                            (aim_location.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT);
            if (is_valid)
            {
                Point position(aim_location.pose.position.x, aim_location.pose.position.y, aim_location.pose.position.z);
                Quaternion orientation(aim_location.pose.orientation.x, aim_location.pose.orientation.y,
                                       aim_location.pose.orientation.z, aim_location.pose.orientation.w);
                aim_pose = ControllerPose(Pose(position, orientation), true);
            }
        }

        bool primary_click = get_boolean_action_state(session_, core_funcs_, primary_click_action_, hand_path);
        bool secondary_click = get_boolean_action_state(session_, core_funcs_, secondary_click_action_, hand_path);

        float thumbstick_x = 0.0f, thumbstick_y = 0.0f;
        get_vector2_action_state(session_, core_funcs_, thumbstick_action_, hand_path, thumbstick_x, thumbstick_y);

        bool thumbstick_click = get_boolean_action_state(session_, core_funcs_, thumbstick_click_action_, hand_path);
        bool menu_click = get_boolean_action_state(session_, core_funcs_, menu_click_action_, hand_path);
        float squeeze_value = get_float_action_state(session_, core_funcs_, squeeze_value_action_, hand_path);
        float trigger_value = get_float_action_state(session_, core_funcs_, trigger_value_action_, hand_path);

        ControllerInputState inputs(primary_click, secondary_click, thumbstick_click, menu_click, thumbstick_x,
                                    thumbstick_y, squeeze_value, trigger_value);

        tracked.emplace();
        tracked->grip_pose = std::make_shared<ControllerPose>(grip_pose);
        tracked->aim_pose = std::make_shared<ControllerPose>(aim_pose);
        tracked->inputs = std::make_shared<ControllerInputState>(inputs);
    };

    update_controller(left_hand_path_, left_grip_space_, left_aim_space_, left_native);
    update_controller(right_hand_path_, right_grip_space_, right_aim_space_, right_native);

    const DeviceDataTimestamp timestamp(last_update_time_, last_update_time_, xr_time);
    left_tracked_ = publish_and_record(mcap_channels_.get(), 0, timestamp, value_ptr(left_native));
    right_tracked_ = publish_and_record(mcap_channels_.get(), 1, timestamp, value_ptr(right_native));
}

const Serialized<ControllerSnapshot>& LiveControllerTrackerImpl::get_left_controller() const
{
    return left_tracked_;
}

const Serialized<ControllerSnapshot>& LiveControllerTrackerImpl::get_right_controller() const
{
    return right_tracked_;
}

void LiveControllerTrackerImpl::apply_left_haptic_feedback(float amplitude, float frequency_hz, float duration_s) const
{
    apply_haptic_feedback(Side::Left, amplitude, frequency_hz, duration_s);
}

void LiveControllerTrackerImpl::apply_right_haptic_feedback(float amplitude, float frequency_hz, float duration_s) const
{
    apply_haptic_feedback(Side::Right, amplitude, frequency_hz, duration_s);
}

void LiveControllerTrackerImpl::apply_haptic_feedback(Side side, float amplitude, float frequency_hz, float duration_s) const
{
    const XrPath subaction_path = (side == Side::Left) ? left_hand_path_ : right_hand_path_;
    const size_t slot = (side == Side::Left) ? 0 : 1;
    const char* const side_name = (side == Side::Left) ? "left" : "right";

    // Map non-finite inputs (NaN / +-Inf) to zero so they hit the explicit-stop
    // / runtime-default branches below. Without this, std::clamp leaves NaN
    // unchanged (unordered comparisons) and static_cast<XrDuration>(NaN/Inf)
    // is UB per [conv.fpint] (XrDuration is int64).
    const float safe_amplitude = std::isfinite(amplitude) ? amplitude : 0.0f;
    const float safe_duration_s = std::isfinite(duration_s) ? duration_s : 0.0f;
    const float safe_frequency_hz = std::isfinite(frequency_hz) ? frequency_hz : 0.0f;

    XrHapticActionInfo info{ XR_TYPE_HAPTIC_ACTION_INFO };
    info.action = haptic_action_;
    info.subactionPath = subaction_path;

    // amplitude==0 issues an explicit stop so an in-flight rumble aborts when
    // the upstream deadband closes.
    if (safe_amplitude <= 0.0f)
    {
        if (core_funcs_.xrStopHapticFeedback == nullptr)
        {
            // Runtime does not advertise the entry point — silently no-op.
            return;
        }
        const XrResult stop_result = core_funcs_.xrStopHapticFeedback(session_, &info);
        if (XR_FAILED(stop_result))
        {
            bool expected = false;
            if (stop_haptic_error_logged_[slot].compare_exchange_strong(expected, true))
            {
                logger_->warn("xrStopHapticFeedback({}) failed: {}; further errors for this side will be silenced.",
                              side_name, static_cast<int>(stop_result));
            }
        }
        return;
    }

    if (core_funcs_.xrApplyHapticFeedback == nullptr)
    {
        // Runtime does not advertise the entry point — silently no-op.
        return;
    }

    XrHapticVibration vibration{ XR_TYPE_HAPTIC_VIBRATION };
    vibration.amplitude = std::clamp(safe_amplitude, 0.0f, 1.0f);
    // 1e18 ns (~31 years) caps the converted duration well below INT64_MAX so
    // the cast cannot overflow on absurdly large finite inputs.
    constexpr double k_max_duration_ns = 1.0e18;
    vibration.duration =
        (safe_duration_s <= 0.0f) ?
            XR_MIN_HAPTIC_DURATION :
            static_cast<XrDuration>(std::clamp(static_cast<double>(safe_duration_s) * 1.0e9, 0.0, k_max_duration_ns));
    vibration.frequency = (safe_frequency_hz <= 0.0f) ? XR_FREQUENCY_UNSPECIFIED : safe_frequency_hz;

    const XrResult apply_result =
        core_funcs_.xrApplyHapticFeedback(session_, &info, reinterpret_cast<const XrHapticBaseHeader*>(&vibration));
    if (XR_FAILED(apply_result))
    {
        bool expected = false;
        if (apply_haptic_error_logged_[slot].compare_exchange_strong(expected, true))
        {
            logger_->warn("xrApplyHapticFeedback({}) failed: {}; further errors for this side will be silenced.",
                          side_name, static_cast<int>(apply_result));
        }
    }
}

} // namespace core
