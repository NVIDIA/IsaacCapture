// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Define XR_NO_PROTOTYPES to prevent OpenXR headers from declaring function prototypes
// This forces us to use xrGetInstanceProcAddr for all OpenXR functions
#define XR_NO_PROTOTYPES

#include <openxr/openxr.h>

#include <XR_NVX1_action_context.h>
#include <cassert>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

// When XR_NO_PROTOTYPES is defined, even xrGetInstanceProcAddr is not declared
// We need to manually declare it here so we can bootstrap the dynamic loading
// This will use whatever OpenXR loader is already loaded in the process
extern "C"
{
    XRAPI_ATTR XrResult XRAPI_CALL xrGetInstanceProcAddr(XrInstance instance,
                                                         const char* name,
                                                         PFN_xrVoidFunction* function);
    XRAPI_ATTR XrResult XRAPI_CALL xrEnumerateInstanceExtensionProperties(const char* layerName,
                                                                          uint32_t propertyCapacityInput,
                                                                          uint32_t* propertyCountOutput,
                                                                          XrExtensionProperties* properties);
}

namespace core
{

// Helper structure to hold dynamically loaded core OpenXR function pointers
// These are the core functions used by the trackers (not extensions)
struct OpenXRCoreFunctions
{
    // Core functions needed by trackers
    PFN_xrGetSystem xrGetSystem;
    PFN_xrGetSystemProperties xrGetSystemProperties;
    PFN_xrCreateReferenceSpace xrCreateReferenceSpace;
    PFN_xrDestroySpace xrDestroySpace;
    PFN_xrLocateSpace xrLocateSpace;

    // Action system functions (for controller tracking)
    PFN_xrStringToPath xrStringToPath;
    PFN_xrPathToString xrPathToString;
    PFN_xrCreateActionSet xrCreateActionSet;
    PFN_xrDestroyActionSet xrDestroyActionSet;
    PFN_xrCreateAction xrCreateAction;
    PFN_xrSuggestInteractionProfileBindings xrSuggestInteractionProfileBindings;
    PFN_xrAttachSessionActionSets xrAttachSessionActionSets;
    PFN_xrCreateActionSpace xrCreateActionSpace;
    PFN_xrSyncActions xrSyncActions;
    PFN_xrGetActionStateBoolean xrGetActionStateBoolean;
    PFN_xrGetActionStateFloat xrGetActionStateFloat;
    PFN_xrGetActionStateVector2f xrGetActionStateVector2f;
    PFN_xrGetActionStatePose xrGetActionStatePose;

    // Haptic output (optional, used by controller-tracker haptic feedback).
    PFN_xrApplyHapticFeedback xrApplyHapticFeedback;
    PFN_xrStopHapticFeedback xrStopHapticFeedback;

    // Load all core functions from an instance using the provided xrGetInstanceProcAddr
    static OpenXRCoreFunctions load(XrInstance instance, PFN_xrGetInstanceProcAddr getProcAddr)
    {
        assert(getProcAddr);

        OpenXRCoreFunctions results{};
        bool success = true;

        success &= XR_SUCCEEDED(
            getProcAddr(instance, "xrGetSystem", reinterpret_cast<PFN_xrVoidFunction*>(&results.xrGetSystem)));
        success &= XR_SUCCEEDED(getProcAddr(
            instance, "xrGetSystemProperties", reinterpret_cast<PFN_xrVoidFunction*>(&results.xrGetSystemProperties)));
        success &= XR_SUCCEEDED(getProcAddr(instance, "xrCreateReferenceSpace",
                                            reinterpret_cast<PFN_xrVoidFunction*>(&results.xrCreateReferenceSpace)));
        success &= XR_SUCCEEDED(
            getProcAddr(instance, "xrDestroySpace", reinterpret_cast<PFN_xrVoidFunction*>(&results.xrDestroySpace)));
        success &= XR_SUCCEEDED(
            getProcAddr(instance, "xrLocateSpace", reinterpret_cast<PFN_xrVoidFunction*>(&results.xrLocateSpace)));

        if (!success)
        {
            throw std::runtime_error("Failed to load core OpenXR functions");
        }

        // Action system functions (optional, for controller tracking)
        // Note: These don't fail the load if not available, as they're only needed by controller tracker
        getProcAddr(instance, "xrStringToPath", reinterpret_cast<PFN_xrVoidFunction*>(&results.xrStringToPath));
        getProcAddr(instance, "xrPathToString", reinterpret_cast<PFN_xrVoidFunction*>(&results.xrPathToString));
        getProcAddr(instance, "xrCreateActionSet", reinterpret_cast<PFN_xrVoidFunction*>(&results.xrCreateActionSet));
        getProcAddr(instance, "xrDestroyActionSet", reinterpret_cast<PFN_xrVoidFunction*>(&results.xrDestroyActionSet));
        getProcAddr(instance, "xrCreateAction", reinterpret_cast<PFN_xrVoidFunction*>(&results.xrCreateAction));
        getProcAddr(instance, "xrSuggestInteractionProfileBindings",
                    reinterpret_cast<PFN_xrVoidFunction*>(&results.xrSuggestInteractionProfileBindings));
        getProcAddr(instance, "xrAttachSessionActionSets",
                    reinterpret_cast<PFN_xrVoidFunction*>(&results.xrAttachSessionActionSets));
        getProcAddr(instance, "xrCreateActionSpace", reinterpret_cast<PFN_xrVoidFunction*>(&results.xrCreateActionSpace));
        getProcAddr(instance, "xrSyncActions", reinterpret_cast<PFN_xrVoidFunction*>(&results.xrSyncActions));
        getProcAddr(instance, "xrGetActionStateBoolean",
                    reinterpret_cast<PFN_xrVoidFunction*>(&results.xrGetActionStateBoolean));
        getProcAddr(
            instance, "xrGetActionStateFloat", reinterpret_cast<PFN_xrVoidFunction*>(&results.xrGetActionStateFloat));
        getProcAddr(instance, "xrGetActionStateVector2f",
                    reinterpret_cast<PFN_xrVoidFunction*>(&results.xrGetActionStateVector2f));
        getProcAddr(
            instance, "xrGetActionStatePose", reinterpret_cast<PFN_xrVoidFunction*>(&results.xrGetActionStatePose));

        // Haptic output (optional — tracker haptic methods check for null pointers
        // and silently no-op when the runtime does not advertise them).
        getProcAddr(
            instance, "xrApplyHapticFeedback", reinterpret_cast<PFN_xrVoidFunction*>(&results.xrApplyHapticFeedback));
        getProcAddr(
            instance, "xrStopHapticFeedback", reinterpret_cast<PFN_xrVoidFunction*>(&results.xrStopHapticFeedback));

        return results;
    }
};

// Load a single extension function; throws std::runtime_error on failure.
// Use from tracker/session code that already has instance and xrGetInstanceProcAddr.
inline void loadExtensionFunction(XrInstance instance,
                                  PFN_xrGetInstanceProcAddr getProcAddr,
                                  const char* name,
                                  PFN_xrVoidFunction* fn_ptr)
{
    assert(getProcAddr && fn_ptr);
    XrResult r = getProcAddr(instance, name, fn_ptr);
    if (XR_FAILED(r))
    {
        throw std::runtime_error(std::string("Failed to get ") + name + " function pointer: " + std::to_string(r));
    }
}

// Smart pointer type aliases for OpenXR resources
using XrActionSetPtr = std::unique_ptr<std::remove_pointer_t<XrActionSet>, PFN_xrDestroyActionSet>;
using XrSpacePtr = std::unique_ptr<std::remove_pointer_t<XrSpace>, PFN_xrDestroySpace>;
using XrInstanceActionContextPtr =
    std::unique_ptr<std::remove_pointer_t<XrInstanceActionContextNV>, PFN_xrDestroyInstanceActionContextNV>;
using XrSessionActionContextPtr =
    std::unique_ptr<std::remove_pointer_t<XrSessionActionContextNV>, PFN_xrDestroySessionActionContextNV>;

// Dynamically loaded XR_NVX1_action_context extension function pointers.
struct ActionContextFunctions
{
    PFN_xrCreateInstanceActionContextNV create_instance_ctx;
    PFN_xrDestroyInstanceActionContextNV destroy_instance_ctx;
    PFN_xrCreateSessionActionContextNV create_session_ctx;
    PFN_xrDestroySessionActionContextNV destroy_session_ctx;
    PFN_xrSyncActions2NV sync_actions_2;
    // Optional: absent on runtimes predating the query. Null-check before use.
    PFN_xrGetCurrentInteractionProfile2NV get_current_interaction_profile;

    static ActionContextFunctions load(XrInstance instance, PFN_xrGetInstanceProcAddr getProcAddr)
    {
        ActionContextFunctions f{};
        loadExtensionFunction(instance, getProcAddr, "xrCreateInstanceActionContextNV",
                              reinterpret_cast<PFN_xrVoidFunction*>(&f.create_instance_ctx));
        loadExtensionFunction(instance, getProcAddr, "xrDestroyInstanceActionContextNV",
                              reinterpret_cast<PFN_xrVoidFunction*>(&f.destroy_instance_ctx));
        loadExtensionFunction(instance, getProcAddr, "xrCreateSessionActionContextNV",
                              reinterpret_cast<PFN_xrVoidFunction*>(&f.create_session_ctx));
        loadExtensionFunction(instance, getProcAddr, "xrDestroySessionActionContextNV",
                              reinterpret_cast<PFN_xrVoidFunction*>(&f.destroy_session_ctx));
        loadExtensionFunction(
            instance, getProcAddr, "xrSyncActions2NV", reinterpret_cast<PFN_xrVoidFunction*>(&f.sync_actions_2));
        loadExtensionFunction(instance, getProcAddr, "xrGetCurrentInteractionProfile2NV",
                              reinterpret_cast<PFN_xrVoidFunction*>(&f.get_current_interaction_profile));

        if (!f.create_instance_ctx || !f.destroy_instance_ctx || !f.create_session_ctx || !f.destroy_session_ctx ||
            !f.sync_actions_2)
        {
            throw std::runtime_error("Required XR_NVX1_action_context extension functions are missing");
        }
        return f;
    }
};

// Create an action set with automatic cleanup - throws on failure
inline XrActionSetPtr createActionSet(const OpenXRCoreFunctions& funcs,
                                      XrInstance instance,
                                      const XrActionSetCreateInfo& createInfo)
{
    assert(funcs.xrDestroyActionSet);

    XrActionSet actionSet = XR_NULL_HANDLE;
    XrResult result = funcs.xrCreateActionSet(instance, &createInfo, &actionSet);

    if (XR_FAILED(result))
    {
        throw std::runtime_error("Failed to create action set: " + std::to_string(result));
    }

    return XrActionSetPtr(actionSet, funcs.xrDestroyActionSet);
}

// Create a reference space with automatic cleanup - throws on failure
inline XrSpacePtr createReferenceSpace(const OpenXRCoreFunctions& funcs,
                                       XrSession session,
                                       const XrReferenceSpaceCreateInfo& createInfo)
{
    assert(funcs.xrDestroySpace);

    XrSpace space = XR_NULL_HANDLE;
    XrResult result = funcs.xrCreateReferenceSpace(session, &createInfo, &space);

    if (XR_FAILED(result))
    {
        throw std::runtime_error("Failed to create reference space: " + std::to_string(result));
    }

    return XrSpacePtr(space, funcs.xrDestroySpace);
}

// Create an action space with automatic cleanup - throws on failure
inline XrSpacePtr createActionSpace(const OpenXRCoreFunctions& funcs,
                                    XrSession session,
                                    const XrActionSpaceCreateInfo* createInfo)
{
    assert(funcs.xrDestroySpace);

    XrSpace space = XR_NULL_HANDLE;
    XrResult result = funcs.xrCreateActionSpace(session, createInfo, &space);

    if (XR_FAILED(result))
    {
        throw std::runtime_error("Failed to create action space: " + std::to_string(result));
    }

    return XrSpacePtr(space, funcs.xrDestroySpace);
}

// Create an instance action context with automatic cleanup - throws on failure
inline XrInstanceActionContextPtr createInstanceActionContext(const ActionContextFunctions& funcs, XrInstance instance)
{
    XrInstanceActionContextCreateInfoNV create_info{ XR_TYPE_INSTANCE_ACTION_CONTEXT_CREATE_INFO_NV };
    XrInstanceActionContextNV ctx = XR_NULL_HANDLE;
    XrResult result = funcs.create_instance_ctx(instance, &create_info, &ctx);
    if (XR_FAILED(result))
    {
        throw std::runtime_error("Failed to create instance action context: " + std::to_string(result));
    }
    return XrInstanceActionContextPtr(ctx, funcs.destroy_instance_ctx);
}

// Create a session action context with automatic cleanup - throws on failure
inline XrSessionActionContextPtr createSessionActionContext(const ActionContextFunctions& funcs,
                                                            XrSession session,
                                                            XrInstanceActionContextNV instance_ctx)
{
    XrSessionActionContextCreateInfoNV create_info{ XR_TYPE_SESSION_ACTION_CONTEXT_CREATE_INFO_NV };
    create_info.instanceActionContext = instance_ctx;
    XrSessionActionContextNV ctx = XR_NULL_HANDLE;
    XrResult result = funcs.create_session_ctx(session, &create_info, &ctx);
    if (XR_FAILED(result))
    {
        throw std::runtime_error("Failed to create session action context: " + std::to_string(result));
    }
    return XrSessionActionContextPtr(ctx, funcs.destroy_session_ctx);
}

} // namespace core
