// Copyright 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0 OR BSL-1.0
/*!
 * @file
 * @brief Header for OpenXR XR_NVX1_action_context extension.
 * @ingroup external_openxr
 */

#pragma once

#include "openxr_extension_helpers.h"


#ifdef __cplusplus
extern "C" {
#endif


/*
 * Default action contexts:
 * OpenXR has a default instance action context and a default session action
 * context. Action sets created without XrInstanceActionContextInfoNV belong to
 * the default instance context; xrAttachSessionActionSets attaches them to the
 * default session context. This extension adds named instance and session
 * action contexts (XrInstanceActionContextNV, XrSessionActionContextNV) so
 * that applications can create multiple action configurations and bind them
 * to sessions in a single "attach all" step.
 *
 * XR_NVX1_action_context: xrCreateSessionActionContextNV creates an empty
 * session action context from an instance action context and makes that
 * instance context immutable. Action sets are attached to the session context
 * via xrAttachSessionActionSets by chaining XrSessionActionContextInfoNV; all
 * action sets in that call must belong to the same instance action context
 * (the one the session context was created from).
 *
 * xrAttachSessionActionSets: when not chaining XrSessionActionContextInfoNV
 * (or sessionActionContext is XR_NULL_HANDLE), only action sets from the
 * default instance action context may be attached, to the default session
 * context. The runtime must return
 * XR_ERROR_ACTION_SETS_DIFFERENT_INSTANCE_ACTION_CONTEXT_NV if any action set
 * belongs to an XrInstanceActionContextNV. When chaining
 * XrSessionActionContextInfoNV with a valid sessionActionContext, the action
 * sets are attached to that session context and must all belong to the same
 * instance action context.
 *
 * xrSyncActions may only be used with action sets attached via
 * xrAttachSessionActionSets (default context). Action sets that belong to an
 * XrInstanceActionContextNV must be synced with xrSyncActions2NV. The runtime
 * must return XR_ERROR_ACTION_SETS_DIFFERENT_INSTANCE_ACTION_CONTEXT_NV if
 * xrSyncActions is called with action sets that belong to an instance action
 * context.
 *
 * The core session-level action APIs (xrGetActionStateBoolean,
 * xrGetActionStateFloat, xrGetActionStateVector2f, xrGetActionStatePose,
 * xrEnumerateBoundSourcesForAction, xrGetInputSourceLocalizedName,
 * xrApplyHapticFeedback, xrStopHapticFeedback, xrCreateActionSpace) apply to
 * any action attached to the session, whether from the default context or
 * from an XrSessionActionContextNV; they do not take a session action context
 * parameter.
 */

// Extension number 848
#define XR_NVX1_action_context 1
#define XR_NVX1_action_context_SPEC_VERSION 1
#define XR_NVX1_ACTION_CONTEXT_EXTENSION_NAME "XR_NVX1_action_context"

/*
 * Handle definitions.
 */
XR_DEFINE_HANDLE(XrInstanceActionContextNV)
XR_DEFINE_HANDLE(XrSessionActionContextNV)

/*
 * Structure types.
 */
XR_STRUCT_ENUM(XR_TYPE_INSTANCE_ACTION_CONTEXT_CREATE_INFO_NV, 1000848001);
XR_STRUCT_ENUM(XR_TYPE_SESSION_ACTION_CONTEXT_CREATE_INFO_NV, 1000848002);
XR_STRUCT_ENUM(XR_TYPE_INSTANCE_ACTION_CONTEXT_INFO_NV, 1000848003);
XR_STRUCT_ENUM(XR_TYPE_SESSION_ACTION_CONTEXT_INFO_NV, 1000848007);
XR_STRUCT_ENUM(XR_TYPE_ACTIONS_SYNC_INFO_2_NV, 1000848004);
XR_STRUCT_ENUM(XR_TYPE_ACTIONS_SYNC_STATE_2_NV, 1000848005);
XR_STRUCT_ENUM(XR_TYPE_INTERACTION_PROFILE_GET_INFO_2_NV, 1000848006);

typedef struct XrInstanceActionContextCreateInfoNV {
    XrStructureType type;
    const void *XR_MAY_ALIAS next;
} XrInstanceActionContextCreateInfoNV;

typedef struct XrSessionActionContextCreateInfoNV {
    XrStructureType type;
    const void *XR_MAY_ALIAS next;
    XrInstanceActionContextNV instanceActionContext;
} XrSessionActionContextCreateInfoNV;

/*
 * Returned when not all action sets belong to the expected instance action
 * context. An action set belongs to an instance action context if and only if
 * it was created with that context (XrInstanceActionContextInfoNV in
 * xrCreateActionSet createInfo->next). Returned by xrSyncActions when any
 * active action set belongs to an instance action context (use
 * xrSyncActions2NV instead). Returned by xrAttachSessionActionSets when
 * attaching to a session context and the action sets do not all belong to that
 * session context's instance action context.
 */
XR_RESULT_ENUM(XR_ERROR_ACTION_SETS_DIFFERENT_INSTANCE_ACTION_CONTEXT_NV, -1000848000);

/*
 * Returned when an action set is already attached to another
 * XrSessionActionContextNV for this session (e.g. when attaching via
 * xrAttachSessionActionSets with XrSessionActionContextInfoNV). An action set
 * from an instance action context may only be attached to one session action
 * context at a time; destroying that session action context allows attaching
 * it again.
 */
XR_RESULT_ENUM(XR_ERROR_ACTION_SET_ALREADY_ATTACHED_TO_SESSION_CONTEXT_NV, -1000848001);

/*
 * Chain struct for instance-level action APIs. Valid only when chained from
 * XrActionSetCreateInfo::next or XrInteractionProfileSuggestedBinding::next.
 * When present, the given instanceActionContext is used instead of the
 * default. If chained from any other structure, the runtime must return
 * XR_ERROR_VALIDATION_FAILURE.
 */
typedef struct XrInstanceActionContextInfoNV {
    XrStructureType type;
    const void *XR_MAY_ALIAS next;
    XrInstanceActionContextNV instanceActionContext;
} XrInstanceActionContextInfoNV;

/*
 * Chain struct for xrAttachSessionActionSets. Valid when chained from
 * XrSessionActionSetsAttachInfo::next. When present and sessionActionContext
 * is not XR_NULL_HANDLE, the action sets are attached to that session action
 * context; they must all belong to the same instance action context (the one
 * that session context was created from). When absent or sessionActionContext
 * is XR_NULL_HANDLE, action sets are attached to the default session context
 * and must belong to the default instance action context.
 */
typedef struct XrSessionActionContextInfoNV {
    XrStructureType type;
    const void *XR_MAY_ALIAS next;
    XrSessionActionContextNV sessionActionContext;
} XrSessionActionContextInfoNV;

typedef struct XrActionsSyncInfo2NV {
    XrStructureType type;
    const void *XR_MAY_ALIAS next;
    uint32_t countActiveActionSets;
    const XrActiveActionSet *activeActionSets;
    XrSessionActionContextNV sessionActionContext;
} XrActionsSyncInfo2NV;

typedef struct XrActionsSyncState2NV {
    XrStructureType type;
    void *XR_MAY_ALIAS next;
    XrBool32 interactionProfileChanged;
} XrActionsSyncState2NV;

typedef struct XrInteractionProfileGetInfo2NV {
    XrStructureType type;
    const void *XR_MAY_ALIAS next;
    XrPath topLevelUserPath;
    XrSessionActionContextNV sessionActionContext;
} XrInteractionProfileGetInfo2NV;

/*
 * Function pointer types.
 */

/*!
 * Creates an instance action context. createInfo specifies no additional
 * parameters; the context is populated by subsequent xrCreateActionSet and
 * xrSuggestInteractionProfileBindings calls that chain XrInstanceActionContextInfoNV.
 */
typedef XrResult(XRAPI_PTR *PFN_xrCreateInstanceActionContextNV)(
    XrInstance instance,
    const XrInstanceActionContextCreateInfoNV *createInfo,
    XrInstanceActionContextNV *instanceActionContext);

typedef void(XRAPI_PTR *PFN_xrDestroyInstanceActionContextNV)(
    XrInstanceActionContextNV instanceActionContext);

/*!
 * Creates an empty session action context from the given instance action
 * context. The instance context becomes immutable. Attach action sets via
 * xrAttachSessionActionSets with XrSessionActionContextInfoNV chained from
 * XrSessionActionSetsAttachInfo::next; all action sets in that call must
 * belong to the same instance action context.
 */
typedef XrResult(XRAPI_PTR *PFN_xrCreateSessionActionContextNV)(
    XrSession session,
    const XrSessionActionContextCreateInfoNV *createInfo,
    XrSessionActionContextNV *sessionActionContext);

typedef void(XRAPI_PTR *PFN_xrDestroySessionActionContextNV)(
    XrSessionActionContextNV sessionActionContext);

/*!
 * Syncs action state for the given session action context. syncInfo is the
 * same as XrActionsSyncInfo plus sessionActionContext. When sessionActionContext
 * is a valid handle, activeActionSets must be action sets attached to that
 * session action context. When sessionActionContext is XR_NULL_HANDLE, the
 * default session action context (from xrAttachSessionActionSets) is used and
 * activeActionSets must belong to the default instance action context.
 *
 * xrSyncActions2NV never queues XrEventDataInteractionProfileChanged; it only
 * reports profile changes via syncState->interactionProfileChanged. This
 * applies whether sessionActionContext is a valid handle or XR_NULL_HANDLE.
 * syncState is required (non-NULL); the runtime must return
 * XR_ERROR_VALIDATION_FAILURE if syncState is NULL. syncState->interactionProfileChanged
 * is XR_TRUE if the current interaction profile for any active top-level user
 * path has changed since the last sync for this session action context.
 */
typedef XrResult(XRAPI_PTR *PFN_xrSyncActions2NV)(XrSession session,
                                                  const XrActionsSyncInfo2NV *syncInfo,
                                                  XrActionsSyncState2NV *syncState);

/*!
 * Returns the current interaction profile for the given top-level user path
 * in the specified session action context. getInfo->sessionActionContext may be
 * XR_NULL_HANDLE to use the default session action context (from
 * xrAttachSessionActionSets); the same errors apply (e.g.
 * XR_ERROR_ACTIONSET_NOT_ATTACHED if no action sets have been attached).
 */
typedef XrResult(XRAPI_PTR *PFN_xrGetCurrentInteractionProfile2NV)(
    XrSession session,
    const XrInteractionProfileGetInfo2NV *getInfo,
    XrInteractionProfileState *state);

#ifdef __cplusplus
}
#endif
