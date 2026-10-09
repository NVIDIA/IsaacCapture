// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <openxr/openxr.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace viz
{

class VkContext;

// OpenXR runtime + (eventually) graphics-bound session.
//
// Two stages because session creation needs a VkContext, but
// VkContext's XR-bound init needs the XrInstance + systemId from
// stage 1. Stage-1-only state is legal — instance / system_id /
// time-conversion plumbing can be exercised without a VkContext.
//
//   stage 1 (ctor):           xrCreateInstance + xrGetSystem + extension probing.
//   stage 2 (attach_graphics): xrCreateSession + reference / VIEW spaces + view config.
//
// Threading: single-threaded. Frame-loop methods (poll_events,
// wait_frame, begin_frame, locate_views, locate_view_space, end_frame)
// must run on the same thread (OpenXR is not thread-safe per session).
class OpenXrSession
{
public:
    struct Config
    {
        // LOCAL = seated/head-centered (default, fits teleop dashboards).
        // STAGE = room-scale; requires recenter / guardian setup.
        XrReferenceSpaceType reference_space_type = XR_REFERENCE_SPACE_TYPE_LOCAL;

        // Reverse-Z near/far in meters. Drives per-eye projection AND
        // XrCompositionLayerDepthInfoKHR (must match — runtime uses
        // both for reprojection).
        float near_z = 0.05f;
        float far_z = 100.0f;
    };

    // Stage 1. system_wait_seconds: how long to poll xrGetSystem when
    // runtime returns XR_ERROR_FORM_FACTOR_UNAVAILABLE (HMD not yet
    // connected — typical for streaming runtimes).
    //   0  — fail fast.
    //   >0 — poll for that many seconds, then throw.
    //   <0 — poll forever (Ctrl-C to break).
    explicit OpenXrSession(const std::string& app_name,
                           const std::vector<std::string>& extra_extensions = {},
                           int system_wait_seconds = 0);

    ~OpenXrSession();

    OpenXrSession(const OpenXrSession&) = delete;
    OpenXrSession& operator=(const OpenXrSession&) = delete;
    OpenXrSession(OpenXrSession&&) = delete;
    OpenXrSession& operator=(OpenXrSession&&) = delete;

    // Stage 2. Call once, after VkContext was XR-bound using instance()
    // + system_id(). Throws std::logic_error on double-attach.
    void attach_graphics(const VkContext& vk, const Config& config);
    void attach_graphics(const VkContext& vk); // default Config
    bool is_graphics_attached() const noexcept
    {
        return session_ != nullptr;
    }

    // ── Stage 1 accessors (always valid post-ctor) ──────────────────────

    XrInstance instance() const noexcept
    {
        return instance_.get();
    }
    XrSystemId system_id() const noexcept
    {
        return system_id_;
    }

    // True iff XR_KHR_composition_layer_depth is enabled. Drives whether
    // XrBackend allocates depth swapchains + chains depth_info per view.
    bool has_depth_composition_layer() const noexcept
    {
        return has_depth_composition_layer_;
    }

    // True iff the shaped composition-layer extension is enabled. Drives
    // whether XrBackend accepts CylinderLayer / EquirectLayer.
    bool has_cylinder_composition_layer() const noexcept
    {
        return has_cylinder_composition_layer_;
    }
    bool has_equirect2_composition_layer() const noexcept
    {
        return has_equirect2_composition_layer_;
    }

    // True iff XR_KHR_convert_timespec_time is enabled. When false the
    // conversion methods throw. CLOCK_MONOTONIC == steady_clock on Linux.
    bool has_time_conversion() const noexcept
    {
        return has_time_conversion_;
    }
    std::chrono::steady_clock::time_point xr_time_to_steady_clock(XrTime time) const;
    XrTime steady_clock_to_xr_time(std::chrono::steady_clock::time_point t) const;

    // ── Stage 2 accessors (XR_NULL_HANDLE / empty before attach) ────────

    XrSession session() const noexcept
    {
        return session_.get();
    }
    XrSpace reference_space() const noexcept
    {
        return reference_space_.get();
    }
    // Head reference space. Locate against reference_space at any
    // XrTime to get head pose; locate_view_space() wraps it.
    XrSpace view_space() const noexcept
    {
        return view_space_.get();
    }
    XrViewConfigurationType view_configuration_type() const noexcept
    {
        return view_configuration_type_;
    }
    // Picked from the runtime's first-advertised mode at attach_graphics:
    // ALPHA_BLEND on passthrough, OPAQUE on pure-VR, ADDITIVE on
    // optical see-through. Same binary works across all three.
    XrEnvironmentBlendMode environment_blend_mode() const noexcept
    {
        return environment_blend_mode_;
    }
    float near_z() const noexcept
    {
        return config_.near_z;
    }
    float far_z() const noexcept
    {
        return config_.far_z;
    }

    const std::vector<XrViewConfigurationView>& view_configuration_views() const noexcept
    {
        return view_configuration_views_;
    }
    uint32_t view_count() const noexcept
    {
        return static_cast<uint32_t>(view_configuration_views_.size());
    }

    // ── Frame-loop primitives (require attach_graphics) ─────────────────

    // Drains the event queue, updates running/exit flags, drives the
    // auto begin/end on READY/STOPPING. Idempotent; call every frame.
    void poll_events();

    // True in SYNCHRONIZED/VISIBLE/FOCUSED — the only states where
    // xrWaitFrame/xrBeginFrame are valid.
    bool session_running() const noexcept
    {
        return session_running_;
    }
    bool exit_requested() const noexcept
    {
        return exit_requested_;
    }

    // True once since the last call if the runtime moved the reference space's
    // origin (recenter, guardian rebound). Consume-once: a pose LATCHED in the
    // old space is silently wrong afterwards and its owner must re-anchor, so
    // this must not be missed by a caller that polls it every frame.
    bool take_reference_space_changed() noexcept
    {
        const bool changed = reference_space_changed_;
        reference_space_changed_ = false;
        return changed;
    }

    // Ends a running session the way the spec requires: xrRequestExitSession, then poll until the
    // runtime reports STOPPING (poll_events() issues xrEndSession). Returns true once the session
    // is no longer running, including when it never ran; false on a failed request or timeout.
    // Idempotent; the destructor calls it. Same-thread rule as poll_events.
    bool end_session(std::chrono::milliseconds timeout = std::chrono::milliseconds(2000));

    // Throws on hard XR failures. XR_FRAME_DISCARDED on begin_frame is
    // non-fatal — pair with end_frame to keep the protocol balanced.
    bool wait_frame(XrFrameState* out_state);
    void begin_frame();

    // Locate views in reference_space. Returns false on tracking loss
    // (out_views resized to zero poses); throws on hard failures.
    bool locate_views(XrTime predicted_display_time, XrViewState* out_view_state, std::vector<XrView>* out_views);

    // Head pose at predicted_display_time. Never throws — XrBackend
    // calls this between xrBeginFrame and xrEndFrame, where a throw
    // would unbalance the protocol. Returns false on any failure.
    bool locate_view_space(XrTime predicted_display_time, XrSpaceLocation* out_location) const;

    // layers may be empty (blank frame; valid per spec).
    void end_frame(XrTime predicted_display_time, const std::vector<const XrCompositionLayerBaseHeader*>& layers);

private:
    // RAII via unique_ptr + PFN deleter — same pattern as core::OpenXRSession.
    using InstanceHandle = std::unique_ptr<std::remove_pointer_t<XrInstance>, PFN_xrDestroyInstance>;
    using SessionHandle = std::unique_ptr<std::remove_pointer_t<XrSession>, PFN_xrDestroySession>;
    using SpaceHandle = std::unique_ptr<std::remove_pointer_t<XrSpace>, PFN_xrDestroySpace>;

    void create_instance(const std::string& app_name, const std::vector<std::string>& extra_extensions);
    void wait_for_system(int system_wait_seconds);
    void enumerate_view_configuration();
    void enumerate_environment_blend_mode();
    void create_session(const VkContext& vk);
    void create_reference_space(XrReferenceSpaceType type);
    void handle_session_state_change(XrSessionState new_state);

    Config config_;

    // Member declaration order matters: destruction is reverse, and
    // OpenXR requires Spaces → Session → Instance teardown order.
    InstanceHandle instance_{ nullptr, nullptr };
    XrSystemId system_id_ = XR_NULL_SYSTEM_ID;
    bool has_depth_composition_layer_ = false;
    bool has_cylinder_composition_layer_ = false;
    bool has_equirect2_composition_layer_ = false;
    bool has_time_conversion_ = false;
    // Type-erased so this header doesn't need XR_USE_TIMESPEC; .cpp casts back.
    PFN_xrVoidFunction xr_convert_timespec_time_to_time_ = nullptr;
    PFN_xrVoidFunction xr_convert_time_to_timespec_time_ = nullptr;

    SessionHandle session_{ nullptr, nullptr };
    SpaceHandle reference_space_{ nullptr, nullptr };
    SpaceHandle view_space_{ nullptr, nullptr };

    XrViewConfigurationType view_configuration_type_ = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
    std::vector<XrViewConfigurationView> view_configuration_views_;
    XrEnvironmentBlendMode environment_blend_mode_ = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;

    XrSessionState state_ = XR_SESSION_STATE_UNKNOWN;
    bool session_running_ = false;
    bool exit_requested_ = false;
    bool reference_space_changed_ = false;
};

} // namespace viz
