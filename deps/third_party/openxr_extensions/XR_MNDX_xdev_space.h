// Copyright 2023-2024, Collabora, Ltd.
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  Preview header for XR_MNDX_xdev_space extension.
 * @author Jakob Bornecrantz <jakob@collabora.com>
 * @ingroup external_openxr
 */
#ifndef XR_MNDX_XDEV_SPACE_H
#define XR_MNDX_XDEV_SPACE_H 1

#include "openxr_extension_helpers.h"

#ifdef __cplusplus
extern "C" {
#endif

// Extension number 445 (444 prefix)
#define XR_MNDX_xdev_space 1
#define XR_MNDX_xdev_space_SPEC_VERSION 2
#define XR_MNDX_XDEV_SPACE_EXTENSION_NAME "XR_MNDX_xdev_space"


XR_DEFINE_ATOM(XrXDevIdMNDX)
XR_DEFINE_HANDLE(XrXDevListMNDX)


XR_STRUCT_ENUM(XR_TYPE_SYSTEM_XDEV_SPACE_PROPERTIES_MNDX, 1000444001);
// XrSystemXDevSpacePropertiesMNDX extends XrSystemProperties
typedef struct XrSystemXDevSpacePropertiesMNDX {
    XrStructureType       type;
    void* XR_MAY_ALIAS    next;
    XrBool32              supportsXDevSpace;
} XrSystemXDevSpacePropertiesMNDX;

XR_STRUCT_ENUM(XR_TYPE_CREATE_XDEV_LIST_INFO_MNDX, 1000444002);
typedef struct XrCreateXDevListInfoMNDX {
    XrStructureType             type;
    const void* XR_MAY_ALIAS    next;
} XrCreateXDevListInfoMNDX;

XR_STRUCT_ENUM(XR_TYPE_GET_XDEV_INFO_MNDX, 1000444003);
typedef struct XrGetXDevInfoMNDX {
    XrStructureType             type;
    const void* XR_MAY_ALIAS    next;
    XrXDevIdMNDX                id;
} XrGetXDevInfoMNDX;

XR_STRUCT_ENUM(XR_TYPE_XDEV_PROPERTIES_MNDX, 1000444004);
typedef struct XrXDevPropertiesMNDX {
    XrStructureType       type;
    void* XR_MAY_ALIAS    next;
    char                  name[256];
    char                  serial[256];
    XrBool32              canCreateSpace;
} XrXDevPropertiesMNDX;

XR_STRUCT_ENUM(XR_TYPE_CREATE_XDEV_SPACE_INFO_MNDX, 1000444005);
typedef struct XrCreateXDevSpaceInfoMNDX {
    XrStructureType             type;
    const void* XR_MAY_ALIAS    next;
    XrXDevListMNDX              xdevList;
    XrXDevIdMNDX                id;
    XrPosef                     offset;
} XrCreateXDevSpaceInfoMNDX;

XR_STRUCT_ENUM(XR_TYPE_CREATE_HAND_TRACKER_XDEV_MNDX, 1000444006);
typedef struct XrCreateHandTrackerXDevMNDX {
    XrStructureType             type;
    const void* XR_MAY_ALIAS    next;
    XrXDevListMNDX              xdevList;
    XrXDevIdMNDX                id;
} XrCreateHandTrackerXDevMNDX;

typedef XrResult (XRAPI_PTR *PFN_xrCreateXDevListMNDX)(XrSession session, const XrCreateXDevListInfoMNDX *info, XrXDevListMNDX *xdevList);
typedef XrResult (XRAPI_PTR *PFN_xrGetXDevListGenerationNumberMNDX)(XrXDevListMNDX xdevList, uint64_t *outGeneration);
typedef XrResult (XRAPI_PTR *PFN_xrEnumerateXDevsMNDX)(XrXDevListMNDX xdevList, uint32_t xdevCapacityInput, uint32_t* xdevCountOutput, XrXDevIdMNDX* xdevs);
typedef XrResult (XRAPI_PTR *PFN_xrGetXDevPropertiesMNDX)(XrXDevListMNDX xdevList, const XrGetXDevInfoMNDX *info, XrXDevPropertiesMNDX *properties);
typedef XrResult (XRAPI_PTR *PFN_xrDestroyXDevListMNDX)(XrXDevListMNDX xdevList);
typedef XrResult (XRAPI_PTR *PFN_xrCreateXDevSpaceMNDX)(XrSession session, const XrCreateXDevSpaceInfoMNDX *createInfo, XrSpace *space);


#ifdef __cplusplus
}
#endif

#endif
