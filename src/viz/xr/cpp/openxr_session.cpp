// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/viz/xr/openxr_session.hpp"

#include <viz/core/vk_context.hpp>

#define XR_USE_TIMESPEC
#include <viz/core/openxr_platform_compat.hpp>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <unordered_set>

namespace viz
{

namespace
{

void check_xr(XrResult r, const char* what)
{
    if (XR_FAILED(r))
    {
        throw std::runtime_error(std::string("OpenXrSession: ") + what + " failed: XrResult=" + std::to_string(r));
    }
}

} // namespace

OpenXrSession::OpenXrSession(const std::string& app_name,
                             const std::vector<std::string>& extra_extensions,
                             int system_wait_seconds)
{
    create_instance(app_name, extra_extensions);
    wait_for_system(system_wait_seconds);
}

OpenXrSession::~OpenXrSession()
{
    // End the session before unique_ptr's xrDestroySession. Best-effort — dtor can't throw.
    try
    {
        end_session();
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "OpenXrSession: end_session failed during teardown: %s\n", e.what());
    }
    // Member destruction order is reverse declaration: view_space →
    // reference_space → session → instance. Each unique_ptr's deleter
    // runs xrDestroy* in the right order.
}

bool OpenXrSession::end_session(std::chrono::milliseconds timeout)
{
    if (!session_ || !session_running_)
    {
        return true;
    }
    // xrEndSession is only valid in STOPPING, which the runtime enters after an exit request;
    // poll_events() then ends the session from its STOPPING handler.
    const XrResult r = xrRequestExitSession(session_.get());
    if (XR_FAILED(r))
    {
        std::fprintf(stderr, "OpenXrSession: xrRequestExitSession failed: XrResult=%d\n", static_cast<int>(r));
        return false;
    }
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (true)
    {
        poll_events();
        if (!session_running_)
        {
            return true;
        }
        if (std::chrono::steady_clock::now() >= deadline)
        {
            std::fprintf(
                stderr, "OpenXrSession: runtime did not report STOPPING within %lld ms; destroying the session anyway\n",
                static_cast<long long>(timeout.count()));
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

void OpenXrSession::create_instance(const std::string& app_name, const std::vector<std::string>& extra_extensions)
{
    // Enumerate runtime-advertised extensions once. Used to opt-in to
    // optional extensions (depth / time conversion) and to validate
    // caller-requested extras before xrCreateInstance.
    std::unordered_set<std::string> available_exts;
    {
        uint32_t count = 0;
        if (xrEnumerateInstanceExtensionProperties(nullptr, 0, &count, nullptr) == XR_SUCCESS && count > 0)
        {
            std::vector<XrExtensionProperties> available(count, XrExtensionProperties{ XR_TYPE_EXTENSION_PROPERTIES });
            if (xrEnumerateInstanceExtensionProperties(nullptr, count, &count, available.data()) == XR_SUCCESS)
            {
                available_exts.reserve(available.size());
                for (const auto& ext : available)
                {
                    available_exts.emplace(ext.extensionName);
                }
            }
        }
    }
    const bool runtime_has_depth_layer = available_exts.count(XR_KHR_COMPOSITION_LAYER_DEPTH_EXTENSION_NAME) > 0;
    const bool runtime_has_time_conversion = available_exts.count(XR_KHR_CONVERT_TIMESPEC_TIME_EXTENSION_NAME) > 0;
    const bool runtime_has_cylinder_layer = available_exts.count(XR_KHR_COMPOSITION_LAYER_CYLINDER_EXTENSION_NAME) > 0;
    const bool runtime_has_equirect2_layer = available_exts.count(XR_KHR_COMPOSITION_LAYER_EQUIRECT2_EXTENSION_NAME) > 0;

    // Build the request list deduped: required → opt-in → caller extras.
    // Caller extras are validated; passing an unsupported one is fatal.
    std::vector<const char*> exts;
    std::unordered_set<std::string> requested;
    auto add_unique = [&](const char* name)
    {
        if (requested.emplace(name).second)
        {
            exts.push_back(name);
        }
    };
    add_unique(XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME);
    if (runtime_has_depth_layer)
    {
        add_unique(XR_KHR_COMPOSITION_LAYER_DEPTH_EXTENSION_NAME);
    }
    if (runtime_has_time_conversion)
    {
        add_unique(XR_KHR_CONVERT_TIMESPEC_TIME_EXTENSION_NAME);
    }
    // Shaped composition layers (CylinderLayer / EquirectLayer). Opt-in
    // when advertised; layers requiring them are rejected at add_layer
    // when the runtime lacks the extension.
    if (runtime_has_cylinder_layer)
    {
        add_unique(XR_KHR_COMPOSITION_LAYER_CYLINDER_EXTENSION_NAME);
    }
    if (runtime_has_equirect2_layer)
    {
        add_unique(XR_KHR_COMPOSITION_LAYER_EQUIRECT2_EXTENSION_NAME);
    }
    for (const auto& e : extra_extensions)
    {
        if (available_exts.count(e) == 0)
        {
            throw std::runtime_error(std::string("OpenXrSession: requested extension '") + e +
                                     "' is not advertised by the runtime");
        }
        add_unique(e.c_str());
    }

    XrInstanceCreateInfo info{ XR_TYPE_INSTANCE_CREATE_INFO };
    info.applicationInfo.apiVersion = XR_CURRENT_API_VERSION;
    std::strncpy(info.applicationInfo.applicationName, app_name.c_str(), XR_MAX_APPLICATION_NAME_SIZE - 1);
    std::strncpy(info.applicationInfo.engineName, "Televiz", XR_MAX_ENGINE_NAME_SIZE - 1);
    info.enabledExtensionCount = static_cast<uint32_t>(exts.size());
    info.enabledExtensionNames = exts.data();

    XrInstance raw_instance = XR_NULL_HANDLE;
    check_xr(xrCreateInstance(&info, &raw_instance), "xrCreateInstance");
    instance_ = InstanceHandle(raw_instance, &xrDestroyInstance);
    has_depth_composition_layer_ = runtime_has_depth_layer;
    has_cylinder_composition_layer_ = runtime_has_cylinder_layer;
    has_equirect2_composition_layer_ = runtime_has_equirect2_layer;

    // Resolve PFNs only if both succeed — leave the feature off rather
    // than half-working.
    if (runtime_has_time_conversion)
    {
        PFN_xrVoidFunction to_time_fn = nullptr;
        PFN_xrVoidFunction from_time_fn = nullptr;
        if (xrGetInstanceProcAddr(instance_.get(), "xrConvertTimespecTimeToTimeKHR", &to_time_fn) == XR_SUCCESS &&
            xrGetInstanceProcAddr(instance_.get(), "xrConvertTimeToTimespecTimeKHR", &from_time_fn) == XR_SUCCESS &&
            to_time_fn != nullptr && from_time_fn != nullptr)
        {
            xr_convert_timespec_time_to_time_ = to_time_fn;
            xr_convert_time_to_timespec_time_ = from_time_fn;
            has_time_conversion_ = true;
        }
    }
}

void OpenXrSession::wait_for_system(int system_wait_seconds)
{
    XrSystemGetInfo sys_info{ XR_TYPE_SYSTEM_GET_INFO };
    sys_info.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;

    // FORM_FACTOR_UNAVAILABLE = runtime up but HMD not connected yet
    // (typical for streaming runtimes). Poll within the wait window;
    // any other XrResult fails immediately.
    constexpr auto kPollInterval = std::chrono::milliseconds(200);
    constexpr auto kLogEvery = std::chrono::seconds(3);
    const bool wait_forever = system_wait_seconds < 0;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(system_wait_seconds);
    auto last_log = std::chrono::steady_clock::now();
    bool announced = false;
    while (true)
    {
        const XrResult r = xrGetSystem(instance_.get(), &sys_info, &system_id_);
        if (XR_SUCCEEDED(r))
        {
            if (announced)
            {
                std::fprintf(stderr, "OpenXrSession: HMD connected.\n");
            }
            return;
        }
        if (r != XR_ERROR_FORM_FACTOR_UNAVAILABLE)
        {
            throw std::runtime_error(std::string("OpenXrSession: xrGetSystem failed: XrResult=") + std::to_string(r));
        }
        const auto now = std::chrono::steady_clock::now();
        if (!wait_forever && now >= deadline)
        {
            throw std::runtime_error(
                "OpenXrSession: xrGetSystem timed out waiting for HMD "
                "(XR_ERROR_FORM_FACTOR_UNAVAILABLE, -35) after " +
                std::to_string(system_wait_seconds) +
                "s. The OpenXR runtime is up but no headset is connected to it. "
                "Connect the headset, check NV_DEVICE_PROFILE matches it, or raise "
                "xr_system_wait_seconds to block until it connects. "
                "See docs/source/references/cloudxr.rst, Troubleshooting.");
        }
        if (!announced || (now - last_log) >= kLogEvery)
        {
            if (wait_forever)
            {
                std::fprintf(stderr, "OpenXrSession: waiting for HMD to connect...\n");
            }
            else
            {
                const auto remaining = std::chrono::duration_cast<std::chrono::seconds>(deadline - now).count();
                std::fprintf(stderr, "OpenXrSession: waiting for HMD to connect (%llds remaining)...\n",
                             static_cast<long long>(remaining));
            }
            std::fflush(stderr);
            announced = true;
            last_log = now;
        }
        std::this_thread::sleep_for(kPollInterval);
    }
}

void OpenXrSession::attach_graphics(const VkContext& vk)
{
    attach_graphics(vk, Config{});
}

void OpenXrSession::attach_graphics(const VkContext& vk, const Config& config)
{
    if (session_)
    {
        throw std::logic_error("OpenXrSession::attach_graphics called twice");
    }
    if (!vk.is_initialized())
    {
        throw std::invalid_argument("OpenXrSession::attach_graphics: VkContext is not initialized");
    }
    config_ = config;

    // Stage 2 is exception-safe via RAII: any throw here unwinds the
    // unique_ptrs allocated so far in declaration order (instance_ stays
    // alive since it was created in stage 1).
    enumerate_view_configuration();
    enumerate_environment_blend_mode();
    create_session(vk);
    create_reference_space(config_.reference_space_type);
}

void OpenXrSession::enumerate_view_configuration()
{
    uint32_t count = 0;
    check_xr(xrEnumerateViewConfigurationViews(instance_.get(), system_id_, view_configuration_type_, 0, &count, nullptr),
             "xrEnumerateViewConfigurationViews(count)");
    if (count == 0)
    {
        throw std::runtime_error("OpenXrSession: runtime reports zero views for PRIMARY_STEREO");
    }
    view_configuration_views_.assign(count, XrViewConfigurationView{ XR_TYPE_VIEW_CONFIGURATION_VIEW });
    check_xr(xrEnumerateViewConfigurationViews(instance_.get(), system_id_, view_configuration_type_, count, &count,
                                               view_configuration_views_.data()),
             "xrEnumerateViewConfigurationViews(data)");
}

void OpenXrSession::enumerate_environment_blend_mode()
{
    // OpenXR returns modes in the runtime's preference order; pick the
    // first. ALPHA_BLEND on passthrough HMDs, OPAQUE on pure-VR,
    // ADDITIVE on optical see-through.
    uint32_t count = 0;
    check_xr(xrEnumerateEnvironmentBlendModes(instance_.get(), system_id_, view_configuration_type_, 0, &count, nullptr),
             "xrEnumerateEnvironmentBlendModes(count)");
    if (count == 0)
    {
        throw std::runtime_error("OpenXrSession: runtime advertises zero environment blend modes");
    }
    std::vector<XrEnvironmentBlendMode> modes(count);
    check_xr(xrEnumerateEnvironmentBlendModes(
                 instance_.get(), system_id_, view_configuration_type_, count, &count, modes.data()),
             "xrEnumerateEnvironmentBlendModes(data)");
    environment_blend_mode_ = modes.front();
    // Log so "why is there no passthrough?" is one grep away.
    const char* mode_str = "UNKNOWN";
    switch (environment_blend_mode_)
    {
    case XR_ENVIRONMENT_BLEND_MODE_OPAQUE:
        mode_str = "OPAQUE (VR)";
        break;
    case XR_ENVIRONMENT_BLEND_MODE_ADDITIVE:
        mode_str = "ADDITIVE (optical see-through)";
        break;
    case XR_ENVIRONMENT_BLEND_MODE_ALPHA_BLEND:
        mode_str = "ALPHA_BLEND (camera passthrough)";
        break;
    default:
        break;
    }
    std::fprintf(stderr, "OpenXrSession: env blend mode = %s\n", mode_str);
}

void OpenXrSession::create_session(const VkContext& vk)
{
    XrGraphicsBindingVulkan2KHR binding{ XR_TYPE_GRAPHICS_BINDING_VULKAN2_KHR };
    binding.instance = vk.instance();
    binding.physicalDevice = vk.physical_device();
    binding.device = vk.device();
    binding.queueFamilyIndex = vk.queue_family_index();
    binding.queueIndex = 0;

    XrSessionCreateInfo info{ XR_TYPE_SESSION_CREATE_INFO };
    info.next = &binding;
    info.systemId = system_id_;

    XrSession raw_session = XR_NULL_HANDLE;
    check_xr(xrCreateSession(instance_.get(), &info, &raw_session), "xrCreateSession");
    session_ = SessionHandle(raw_session, &xrDestroySession);
}

void OpenXrSession::create_reference_space(XrReferenceSpaceType type)
{
    XrReferenceSpaceCreateInfo info{ XR_TYPE_REFERENCE_SPACE_CREATE_INFO };
    info.referenceSpaceType = type;
    info.poseInReferenceSpace.orientation = XrQuaternionf{ 0.0f, 0.0f, 0.0f, 1.0f };
    info.poseInReferenceSpace.position = XrVector3f{ 0.0f, 0.0f, 0.0f };

    XrSpace raw_ref = XR_NULL_HANDLE;
    check_xr(xrCreateReferenceSpace(session_.get(), &info, &raw_ref), "xrCreateReferenceSpace");
    reference_space_ = SpaceHandle(raw_ref, &xrDestroySpace);

    // VIEW space — head pose queries locate against reference_space.
    XrReferenceSpaceCreateInfo view_info{ XR_TYPE_REFERENCE_SPACE_CREATE_INFO };
    view_info.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_VIEW;
    view_info.poseInReferenceSpace.orientation = XrQuaternionf{ 0.0f, 0.0f, 0.0f, 1.0f };
    view_info.poseInReferenceSpace.position = XrVector3f{ 0.0f, 0.0f, 0.0f };

    XrSpace raw_view = XR_NULL_HANDLE;
    check_xr(xrCreateReferenceSpace(session_.get(), &view_info, &raw_view), "xrCreateReferenceSpace(view)");
    view_space_ = SpaceHandle(raw_view, &xrDestroySpace);
}

std::chrono::steady_clock::time_point OpenXrSession::xr_time_to_steady_clock(XrTime time) const
{
    if (!has_time_conversion_)
    {
        throw std::runtime_error(
            "OpenXrSession::xr_time_to_steady_clock: XR_KHR_convert_timespec_time not available on this runtime");
    }
    timespec ts{};
    auto fn = reinterpret_cast<PFN_xrConvertTimeToTimespecTimeKHR>(xr_convert_time_to_timespec_time_);
    const XrResult r = fn(instance_.get(), time, &ts);
    if (XR_FAILED(r))
    {
        throw std::runtime_error("OpenXrSession: xrConvertTimeToTimespecTimeKHR failed: XrResult=" + std::to_string(r));
    }
    // CLOCK_MONOTONIC = std::chrono::steady_clock on Linux per spec.
    return std::chrono::steady_clock::time_point{ std::chrono::seconds{ ts.tv_sec } +
                                                  std::chrono::nanoseconds{ ts.tv_nsec } };
}

XrTime OpenXrSession::steady_clock_to_xr_time(std::chrono::steady_clock::time_point t) const
{
    if (!has_time_conversion_)
    {
        throw std::runtime_error(
            "OpenXrSession::steady_clock_to_xr_time: XR_KHR_convert_timespec_time not available on this runtime");
    }
    const auto duration = t.time_since_epoch();
    const auto secs = std::chrono::duration_cast<std::chrono::seconds>(duration);
    const auto nsecs = std::chrono::duration_cast<std::chrono::nanoseconds>(duration - secs);
    timespec ts{ static_cast<time_t>(secs.count()), static_cast<long>(nsecs.count()) };
    XrTime out = 0;
    auto fn = reinterpret_cast<PFN_xrConvertTimespecTimeToTimeKHR>(xr_convert_timespec_time_to_time_);
    const XrResult r = fn(instance_.get(), &ts, &out);
    if (XR_FAILED(r))
    {
        throw std::runtime_error("OpenXrSession: xrConvertTimespecTimeToTimeKHR failed: XrResult=" + std::to_string(r));
    }
    return out;
}

void OpenXrSession::poll_events()
{
    while (true)
    {
        XrEventDataBuffer event{ XR_TYPE_EVENT_DATA_BUFFER };
        const XrResult r = xrPollEvent(instance_.get(), &event);
        if (r == XR_EVENT_UNAVAILABLE)
        {
            return;
        }
        if (XR_FAILED(r))
        {
            throw std::runtime_error("OpenXrSession: xrPollEvent failed: XrResult=" + std::to_string(r));
        }
        switch (event.type)
        {
        case XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED:
        {
            const auto* state_change = reinterpret_cast<const XrEventDataSessionStateChanged*>(&event);
            if (state_change->session == session_.get())
            {
                handle_session_state_change(state_change->state);
            }
            break;
        }
        case XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING:
            // Runtime is going away — quit cleanly.
            exit_requested_ = true;
            break;
        case XR_TYPE_EVENT_DATA_REFERENCE_SPACE_CHANGE_PENDING:
            // Pose origin shifted (recenter, guardian rebound). locate_views
            // absorbs it, but a pose the app LATCHED in the old space does not:
            // it is now wrong by the whole recenter transform. Surfaced through
            // FrameInfo::reference_space_changed so the owner can re-anchor.
            reference_space_changed_ = true;
            break;
        default:
            // Ignore unknown / extension event types we didn't ask for.
            break;
        }
    }
}

void OpenXrSession::handle_session_state_change(XrSessionState new_state)
{
    state_ = new_state;
    switch (new_state)
    {
    case XR_SESSION_STATE_READY:
    {
        XrSessionBeginInfo info{ XR_TYPE_SESSION_BEGIN_INFO };
        info.primaryViewConfigurationType = view_configuration_type_;
        const XrResult r = xrBeginSession(session_.get(), &info);
        if (XR_SUCCEEDED(r))
        {
            session_running_ = true;
        }
        else
        {
            // wait_frame returns false when !running, so this would
            // silently spin nullopt frames forever. Surface it as
            // exit_requested_; throwing from poll_events would unbalance
            // begin_frame's protocol guard.
            std::fprintf(
                stderr, "OpenXrSession: xrBeginSession failed: XrResult=%d (requesting exit)\n", static_cast<int>(r));
            exit_requested_ = true;
        }
        break;
    }
    case XR_SESSION_STATE_SYNCHRONIZED:
    case XR_SESSION_STATE_VISIBLE:
    case XR_SESSION_STATE_FOCUSED:
        // Focus/visibility shift after READY — still running.
        session_running_ = true;
        break;
    case XR_SESSION_STATE_STOPPING:
        if (session_running_)
        {
            const XrResult r = xrEndSession(session_.get());
            if (XR_FAILED(r))
            {
                std::fprintf(stderr, "OpenXrSession: xrEndSession failed: XrResult=%d\n", static_cast<int>(r));
            }
            session_running_ = false;
        }
        break;
    case XR_SESSION_STATE_EXITING:
    case XR_SESSION_STATE_LOSS_PENDING:
        exit_requested_ = true;
        session_running_ = false;
        break;
    default:
        break;
    }
}

bool OpenXrSession::wait_frame(XrFrameState* out_state)
{
    if (!session_running_)
    {
        return false;
    }
    XrFrameWaitInfo wait_info{ XR_TYPE_FRAME_WAIT_INFO };
    *out_state = XrFrameState{ XR_TYPE_FRAME_STATE };
    const XrResult r = xrWaitFrame(session_.get(), &wait_info, out_state);
    if (XR_FAILED(r))
    {
        throw std::runtime_error("OpenXrSession: xrWaitFrame failed: XrResult=" + std::to_string(r));
    }
    return true;
}

void OpenXrSession::begin_frame()
{
    XrFrameBeginInfo info{ XR_TYPE_FRAME_BEGIN_INFO };
    const XrResult r = xrBeginFrame(session_.get(), &info);
    // XR_FRAME_DISCARDED is non-fatal — per spec the app must still
    // call xrEndFrame to balance. Treat as success here.
    if (r != XR_SUCCESS && r != XR_FRAME_DISCARDED)
    {
        throw std::runtime_error("OpenXrSession: xrBeginFrame failed: XrResult=" + std::to_string(r));
    }
}

bool OpenXrSession::locate_views(XrTime predicted_display_time, XrViewState* out_view_state, std::vector<XrView>* out_views)
{
    if (!session_running_)
    {
        return false;
    }
    XrViewLocateInfo locate_info{ XR_TYPE_VIEW_LOCATE_INFO };
    locate_info.viewConfigurationType = view_configuration_type_;
    locate_info.displayTime = predicted_display_time;
    locate_info.space = reference_space_.get();

    *out_view_state = XrViewState{ XR_TYPE_VIEW_STATE };
    out_views->assign(view_count(), XrView{ XR_TYPE_VIEW });

    uint32_t got = 0;
    const XrResult r = xrLocateViews(session_.get(), &locate_info, out_view_state, view_count(), &got, out_views->data());
    if (XR_FAILED(r))
    {
        return false;
    }
    // Validity flags must be set, else returned poses are zero/identity.
    constexpr XrViewStateFlags kRequired = XR_VIEW_STATE_POSITION_VALID_BIT | XR_VIEW_STATE_ORIENTATION_VALID_BIT;
    return (out_view_state->viewStateFlags & kRequired) == kRequired;
}

bool OpenXrSession::locate_view_space(XrTime predicted_display_time, XrSpaceLocation* out_location) const
{
    // Non-throwing: this runs between xrBeginFrame/xrEndFrame and a
    // throw would unbalance the protocol. Tracking-loss + hard failures
    // both surface as `false`.
    *out_location = XrSpaceLocation{ XR_TYPE_SPACE_LOCATION };
    if (!session_running_ || view_space_ == nullptr)
    {
        return false;
    }
    const XrResult r = xrLocateSpace(view_space_.get(), reference_space_.get(), predicted_display_time, out_location);
    if (XR_FAILED(r))
    {
        return false;
    }
    constexpr XrSpaceLocationFlags kRequired =
        XR_SPACE_LOCATION_POSITION_VALID_BIT | XR_SPACE_LOCATION_ORIENTATION_VALID_BIT;
    return (out_location->locationFlags & kRequired) == kRequired;
}

void OpenXrSession::end_frame(XrTime predicted_display_time,
                              const std::vector<const XrCompositionLayerBaseHeader*>& layers)
{
    XrFrameEndInfo info{ XR_TYPE_FRAME_END_INFO };
    info.displayTime = predicted_display_time;
    info.environmentBlendMode = environment_blend_mode_;
    info.layerCount = static_cast<uint32_t>(layers.size());
    info.layers = layers.empty() ? nullptr : layers.data();
    const XrResult r = xrEndFrame(session_.get(), &info);
    if (XR_FAILED(r))
    {
        throw std::runtime_error("OpenXrSession: xrEndFrame failed: XrResult=" + std::to_string(r));
    }
}

} // namespace viz
