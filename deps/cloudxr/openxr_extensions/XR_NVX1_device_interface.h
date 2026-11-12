// Copyright 2025, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0 OR MIT
/*!
 * @file
 * @brief Header for OpenXR device interface.
 * @ingroup external_openxr
 */

#pragma once

#include <openxr/openxr.h>

#ifdef __cplusplus
extern "C"
{
#endif

// Extension number 528 (527 prefix)
#define XR_NVX1_device_interface_base 1
#define XR_NVX1_device_interface_base_SPEC_VERSION 1
#define XR_NVX1_DEVICE_INTERFACE_BASE_EXTENSION_NAME "XR_NVX1_device_interface_base"


#define XR_MAX_PUSH_DEVICE_NAME_SIZE 256
#define XR_MAX_PUSH_DEVICE_SERIAL_SIZE 256


    XR_DEFINE_HANDLE(XrPushDeviceNV)

// Structure type enums
#define XR_TYPE_SYSTEM_PUSH_DEVICE_PROPERTIES_NV ((XrStructureType)1000527000)
#define XR_TYPE_PUSH_DEVICE_CREATE_INFO_NV ((XrStructureType)1000527001)
#define XR_TYPE_PUSH_DEVICE_CREATE_RESULT_NV ((XrStructureType)1000527002)
#define XR_TYPE_PUSH_DEVICE_HAND_TRACKING_INFO_NV ((XrStructureType)1000527003)
#define XR_TYPE_PUSH_DEVICE_HAND_TRACKING_DATA_NV ((XrStructureType)1000527004)

// Error codes
#define XR_ERROR_INCOMPATIBLE_DEVICE_TYPES_NV -1000527000
#define XR_ERROR_ACTION_SPACE_NOT_ACTIVE_NV -1000527001
#define XR_ERROR_MISSING_DEVICE_CAPABILITY_NV -1000527002
#define XR_ERROR_INCOMPATIBLE_SPACE_TYPE_NV -1000527003

    // Note: XrUuidEXT is already defined in openxr.h as typedef XrUuid XrUuidEXT

    /*!
     * Chained into XrSystemProperties to query push device support.
     */
    typedef struct XrSystemPushDevicePropertiesNV
    {
        XrStructureType type; // XR_TYPE_SYSTEM_PUSH_DEVICE_PROPERTIES_NV
        void* next;
        XrBool32 supportsPushDevices;
    } XrSystemPushDevicePropertiesNV;


    typedef struct XrPushDeviceCreateInfoNV
    {
        XrStructureType type; // XR_TYPE_PUSH_DEVICE_CREATE_INFO_NV
        const void* next;
        XrSpace baseSpace;

        XrUuidEXT deviceTypeUuid;
        XrBool32 deviceTypeUuidValid;

        XrUuidEXT deviceUuid;
        XrBool32 deviceUuidValid;

        char localizedName[XR_MAX_PUSH_DEVICE_NAME_SIZE];
        char serial[XR_MAX_PUSH_DEVICE_SERIAL_SIZE];
    } XrPushDeviceCreateInfoNV;

    typedef struct XrPushDeviceCreateResultNV
    {
        XrStructureType type; // XR_TYPE_PUSH_DEVICE_CREATE_RESULT_NV
        void* next;
        XrUuidEXT uuid;
        XrBool32 uuidValid;
    } XrPushDeviceCreateResultNV;


    typedef XrResult(XRAPI_PTR* PFN_xrCreatePushDeviceNV)(XrSession session,
                                                          const XrPushDeviceCreateInfoNV* createInfo,
                                                          XrPushDeviceCreateResultNV* createResult, // Can be null.
                                                          XrPushDeviceNV* pushDevice);
    typedef XrResult(XRAPI_PTR* PFN_xrDestroyPushDeviceNV)(XrPushDeviceNV PushDevice);


    /*
     *
     * Hand tracking
     *
     */

    /*!
     * Chained into xrCreatePushDeviceNV to add hand tracking capabilities to the push
     * device.
     */
    typedef struct XrPushDeviceHandTrackingInfoNV
    {
        XrStructureType type; // XR_TYPE_PUSH_DEVICE_HAND_TRACKING_INFO_NV
        const void* next;
        XrHandEXT hand;
        XrHandJointSetEXT jointSet;
    } XrPushDeviceHandTrackingInfoNV;

    /*!
     * Data being pushed into the device.
     */
    typedef struct XrPushDeviceHandTrackingDataNV
    {
        XrStructureType type; // XR_TYPE_PUSH_DEVICE_HAND_TRACKING_DATA_NV
        const void* next;
        XrTime timestamp;
        uint32_t jointCount;
        const XrHandJointLocationEXT* jointLocations;
    } XrPushDeviceHandTrackingDataNV;

    typedef XrResult(XRAPI_PTR* PFN_xrPushDevicePushHandTrackingNV)(XrPushDeviceNV PushDevice,
                                                                    const XrPushDeviceHandTrackingDataNV* data);


#ifdef __cplusplus
}
#endif
