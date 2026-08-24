// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/oxr/oxr_session.hpp"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <utility>

namespace core
{

namespace
{

constexpr std::chrono::seconds kSystemRetryDelay{ 1 };

// Helper to get user home directory (cross-platform: HOME on Unix, USERPROFILE on Windows)
std::string get_home_dir()
{
#ifdef _WIN32
    const char* home = std::getenv("USERPROFILE");
#else
    const char* home = std::getenv("HOME");
#endif
    if (home == nullptr)
    {
        throw std::runtime_error("Failed to get user home directory");
    }
    return std::string(home);
}

// Ensure an environment variable is set. If not set, warn and set to default_value.
// Returns the final value (either existing or newly set).
void ensure_env_set(const char* env_name, const std::string& default_value)
{
    const char* current_value = std::getenv(env_name);

    if (current_value != nullptr && current_value[0] != '\0')
    {
        return;
    }

    // Not set - warn and set default
    isaaccapture::Logger::get("isaaccapture.core.OpenXRSession")
        ->error(
            "{} environment variable is not set. Please set it before running, e.g.: export {}={}  (or set "
            "ISAAC_TELEOP_DISABLE_CXR_ENV_CHECKS to disable this check)",
            env_name, env_name, default_value);

    throw std::runtime_error("Environment variable " + std::string(env_name) + " is not set");
}

// Opt in to CloudXR's XR_META_body_tracking_full_body.
//
// CloudXR withholds that extension unless this is set, because its support is not
// conformant: the joints are relayed from the connected client rather than produced
// by a conformant implementation. isaacteleop wants it -- the Meta full-body tracker
// (body.quest-cloudxr) is built on it.
//
// Set here, immediately before xrCreateInstance, rather than in the shared
// ~/.cloudxr/run/cloudxr.env: that file is sourced by every CloudXR application, and
// opting all of them in is not ours to decide. This scopes it to processes that
// create an isaacteleop OpenXR session.
//
// overwrite=0, so an operator who exports it explicitly -- including to "0" -- wins.
void opt_in_to_meta_body_tracking()
{
    setenv("NV_CXR_ENABLE_NON_CONFORMANT_META_BODY_TRACKING", "1", 0 /* do not overwrite */);
}

// Ensure required environment variables are set before xrCreateInstance.
void ensure_cloudxr_runtime_configured()
{
    if (std::getenv("ISAAC_TELEOP_DISABLE_CXR_ENV_CHECKS") != nullptr)
    {
        return;
    }

    const std::string home = get_home_dir();

    // NV_CXR_RUNTIME_DIR - required by some OpenXR runtimes for IPC
    ensure_env_set("NV_CXR_RUNTIME_DIR", home + "/.cloudxr/run");

    // XR_RUNTIME_JSON - tells OpenXR loader which runtime to use
    ensure_env_set("XR_RUNTIME_JSON", home + "/.cloudxr/share/openxr/1/openxr_cloudxr.json");
}

} // anonymous namespace

OpenXRSession::OpenXRSession(const std::string& app_name, const std::vector<std::string>& extensions, bool wait_for_system)
    : instance_(XR_NULL_HANDLE, &xrDestroyInstance),
      system_id_(XR_NULL_SYSTEM_ID),
      session_(XR_NULL_HANDLE, &xrDestroySession),
      space_(XR_NULL_HANDLE, &xrDestroySpace),
      wait_for_system_(wait_for_system)
{
    create_instance(app_name, extensions);
    create_system();
    create_session();
    create_reference_space();
    begin();
}

OpenXRSessionHandles OpenXRSession::get_handles() const
{
    // Pass the global xrGetInstanceProcAddr - oxr_session links against OpenXR loader
    return OpenXRSessionHandles(instance_.get(), session_.get(), space_.get(), ::xrGetInstanceProcAddr);
}

OpenXRProviderSnapshot OpenXRSession::get_provider_snapshot()
{
    if (provider_snapshot_.state == OpenXRProviderState::FAILED)
    {
        return provider_snapshot_;
    }

    const auto fail = [this](OpenXRProviderReason reason, std::optional<std::int32_t> result_code, std::string error)
    {
        provider_snapshot_.state = OpenXRProviderState::FAILED;
        provider_snapshot_.reason = reason;
        provider_snapshot_.result_code = result_code;
        provider_snapshot_.error = std::move(error);
        return provider_snapshot_;
    };

    std::optional<OpenXRProviderReason> terminal_reason;
    std::string terminal_error;
    while (true)
    {
        XrEventDataBuffer event{ XR_TYPE_EVENT_DATA_BUFFER };
        const XrResult result = xrPollEvent(instance_.get(), &event);
        if (result == XR_EVENT_UNAVAILABLE)
        {
            break;
        }
        if (result == XR_ERROR_INSTANCE_LOST)
        {
            return fail(OpenXRProviderReason::INSTANCE_LOST, static_cast<std::int32_t>(result),
                        "xrPollEvent failed because the OpenXR instance was lost");
        }
        if (XR_FAILED(result))
        {
            if (terminal_reason)
            {
                return fail(*terminal_reason, std::nullopt, std::move(terminal_error));
            }
            return fail(OpenXRProviderReason::POLL_ERROR, static_cast<std::int32_t>(result),
                        "xrPollEvent failed with XrResult " + std::to_string(result));
        }

        if (event.type == XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING && !terminal_reason)
        {
            const auto* loss = reinterpret_cast<const XrEventDataInstanceLossPending*>(&event);
            terminal_reason = OpenXRProviderReason::INSTANCE_LOST;
            terminal_error = "OpenXR instance loss is pending at time " + std::to_string(loss->lossTime);
        }
        if (event.type != XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED)
        {
            continue;
        }

        const auto* state_change = reinterpret_cast<const XrEventDataSessionStateChanged*>(&event);
        if (state_change->session != session_.get())
        {
            continue;
        }
        if (state_change->state == XR_SESSION_STATE_LOSS_PENDING && !terminal_reason)
        {
            terminal_reason = OpenXRProviderReason::SESSION_LOST;
            terminal_error = "OpenXR session entered XR_SESSION_STATE_LOSS_PENDING";
        }
        if (state_change->state == XR_SESSION_STATE_EXITING && !terminal_reason)
        {
            terminal_reason = OpenXRProviderReason::SESSION_LOST;
            terminal_error = "OpenXR session entered XR_SESSION_STATE_EXITING";
        }
    }
    if (terminal_reason)
    {
        return fail(*terminal_reason, std::nullopt, std::move(terminal_error));
    }

    XrSystemGetInfo system_info{ XR_TYPE_SYSTEM_GET_INFO };
    system_info.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
    XrSystemId system_id = XR_NULL_SYSTEM_ID;
    const XrResult result = xrGetSystem(instance_.get(), &system_info, &system_id);
    if (result == XR_ERROR_FORM_FACTOR_UNAVAILABLE)
    {
        provider_snapshot_.headset_state = OpenXRHeadsetState::DISCONNECTED;
        provider_snapshot_.reason = OpenXRProviderReason::FORM_FACTOR_UNAVAILABLE;
        provider_snapshot_.result_code = static_cast<std::int32_t>(result);
        provider_snapshot_.error = "OpenXR HMD form factor is unavailable (XrResult " + std::to_string(result) + ")";
        return provider_snapshot_;
    }
    if (result == XR_ERROR_INSTANCE_LOST)
    {
        return fail(OpenXRProviderReason::INSTANCE_LOST, static_cast<std::int32_t>(result),
                    "xrGetSystem failed because the OpenXR instance was lost");
    }
    if (XR_FAILED(result))
    {
        return fail(OpenXRProviderReason::POLL_ERROR, static_cast<std::int32_t>(result),
                    "xrGetSystem failed with XrResult " + std::to_string(result));
    }

    provider_snapshot_.headset_state = OpenXRHeadsetState::CONNECTED;
    provider_snapshot_.reason = OpenXRProviderReason::NONE;
    provider_snapshot_.result_code.reset();
    provider_snapshot_.error.clear();
    return provider_snapshot_;
}

void OpenXRSession::create_instance(const std::string& app_name, const std::vector<std::string>& extensions)
{
    // Ensure XR_RUNTIME_JSON is configured before calling xrCreateInstance
    ensure_cloudxr_runtime_configured();

    // Independent of the checks above: those can be disabled, opting in should not be.
    opt_in_to_meta_body_tracking();

    XrInstanceCreateInfo create_info{ XR_TYPE_INSTANCE_CREATE_INFO };
    create_info.applicationInfo.apiVersion = XR_CURRENT_API_VERSION;
    strncpy(create_info.applicationInfo.applicationName, app_name.c_str(), XR_MAX_APPLICATION_NAME_SIZE - 1);
    strncpy(create_info.applicationInfo.engineName, "OXR_Tracking", XR_MAX_ENGINE_NAME_SIZE - 1);

    // Create a combined list with required extensions for headless/overlay mode
    std::vector<std::string> all_extensions = extensions;

    // Add headless and overlay extensions automatically
    all_extensions.push_back("XR_MND_headless");
    all_extensions.push_back("XR_EXTX_overlay");

    // Convert vector<string> to array of const char* for OpenXR API
    std::vector<const char*> extension_ptrs;
    for (const auto& ext : all_extensions)
    {
        extension_ptrs.push_back(ext.c_str());
    }

    create_info.enabledExtensionCount = static_cast<uint32_t>(extension_ptrs.size());
    create_info.enabledExtensionNames = extension_ptrs.empty() ? nullptr : extension_ptrs.data();

    XrInstance instance = XR_NULL_HANDLE;
    XrResult result = xrCreateInstance(&create_info, &instance);
    if (XR_FAILED(result))
    {
        throw std::runtime_error("Failed to create OpenXR instance: " + std::to_string(result));
    }

    instance_.reset(instance);

    logger_->info("Created OpenXR instance");
}

void OpenXRSession::create_system()
{
    XrSystemGetInfo system_info{ XR_TYPE_SYSTEM_GET_INFO };
    system_info.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;

    bool logged_waiting = false;
    while (true)
    {
        XrResult result = xrGetSystem(instance_.get(), &system_info, &system_id_);
        if (XR_SUCCEEDED(result))
        {
            break;
        }

        if (result != XR_ERROR_FORM_FACTOR_UNAVAILABLE)
        {
            throw std::runtime_error("Failed to get OpenXR system: " + std::to_string(result));
        }

        if (!wait_for_system_)
        {
            // xrCreateInstance already succeeded, so the runtime was found
            // (a missing one gives -51). No headset is attached to it.
            throw std::runtime_error(
                "Failed to get OpenXR system: XR_ERROR_FORM_FACTOR_UNAVAILABLE (-35). "
                "The OpenXR runtime is up but no headset is connected to it, and this "
                "session asked for wait_for_system=false rather than waiting for one. "
                "Connect the headset, or check NV_DEVICE_PROFILE matches it. "
                "See docs/source/references/cloudxr.rst, Troubleshooting.");
        }

        if (!logged_waiting)
        {
            logger_->info("OpenXR HMD form factor is unavailable; waiting for a system...");
            logged_waiting = true;
        }

        std::this_thread::sleep_for(kSystemRetryDelay);
    }

    logger_->info("Created OpenXR system");
}

void OpenXRSession::create_session()
{
    // XrSessionCreateInfoOverlayEXTX structure for overlay/headless mode
    struct XrSessionCreateInfoOverlayEXTX
    {
        XrStructureType type;
        const void* next;
        uint32_t createFlags;
        uint32_t sessionLayersPlacement;
    };

    XrSessionCreateInfoOverlayEXTX overlay_info{};
    overlay_info.type = (XrStructureType)1000033000; // XR_TYPE_SESSION_CREATE_INFO_OVERLAY_EXTX
    overlay_info.next = nullptr;
    overlay_info.createFlags = 0;
    overlay_info.sessionLayersPlacement = 0;

    XrSessionCreateInfo create_info{ XR_TYPE_SESSION_CREATE_INFO };
    create_info.next = &overlay_info;
    create_info.systemId = system_id_;

    XrSession session = XR_NULL_HANDLE;
    XrResult result = xrCreateSession(instance_.get(), &create_info, &session);
    if (XR_FAILED(result))
    {
        throw std::runtime_error("Failed to create OpenXR session: " + std::to_string(result));
    }

    session_.reset(session);

    logger_->info("Created OpenXR session (headless mode), handle: {}", fmt::ptr(session_.get()));
}

void OpenXRSession::create_reference_space()
{
    XrReferenceSpaceCreateInfo create_info{ XR_TYPE_REFERENCE_SPACE_CREATE_INFO };
    create_info.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_STAGE;
    create_info.poseInReferenceSpace.orientation.w = 1.0f;

    XrSpace space = XR_NULL_HANDLE;
    XrResult result = xrCreateReferenceSpace(session_.get(), &create_info, &space);
    if (XR_FAILED(result))
    {
        throw std::runtime_error("Failed to create reference space: " + std::to_string(result));
    }

    space_.reset(space);

    logger_->info("Created reference space, handle: {}", fmt::ptr(space_.get()));
}

void OpenXRSession::begin()
{
    // Enumerate view configurations to find a valid one
    uint32_t view_config_count = 0;
    XrResult result = xrEnumerateViewConfigurations(instance_.get(), system_id_, 0, &view_config_count, nullptr);
    if (XR_FAILED(result) || view_config_count == 0)
    {
        throw std::runtime_error("Failed to enumerate view configurations: " + std::to_string(result));
    }

    std::vector<XrViewConfigurationType> view_configs(view_config_count);
    result = xrEnumerateViewConfigurations(
        instance_.get(), system_id_, view_config_count, &view_config_count, view_configs.data());
    if (XR_FAILED(result))
    {
        throw std::runtime_error("Failed to get view configurations: " + std::to_string(result));
    }

    // Find the primary stereo view configuration (preferred), or use the first available
    XrViewConfigurationType selected_view_config = view_configs[0];
    for (const auto& config : view_configs)
    {
        if (config == XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO)
        {
            selected_view_config = config;
            break;
        }
    }

    XrSessionBeginInfo begin_info{ XR_TYPE_SESSION_BEGIN_INFO };
    begin_info.primaryViewConfigurationType = selected_view_config;

    result = xrBeginSession(session_.get(), &begin_info);
    if (XR_FAILED(result))
    {
        throw std::runtime_error("Failed to begin OpenXR session: " + std::to_string(result));
    }
}

} // namespace core
