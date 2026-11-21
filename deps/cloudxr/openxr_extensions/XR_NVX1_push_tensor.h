// Copyright 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0 OR MIT
/*!
 * @file
 * @brief Header for OpenXR push tensor interface.
 * @ingroup external_openxr
 */

#pragma once

#include "openxr_extension_helpers.h"
#include "XR_NVX1_tensor_data.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * General documentation for the XR_NVX1_push_tensor extension:
 *
 * This extension provides an interface for applications to push tensor data
 * into the OpenXR runtime. It enables applications to provide various types
 * of data (sensor readings, ML inference results, custom data streams, etc.)
 * that can be consumed by the runtime or other applications.
 *
 * Key concepts:
 * - Push Tensor: A tensor object that applications can write data into.
 *   Each push tensor has a unique tensor collection ID that can be used to
 *   identify the collection in the tensor data list.
 *
 * - Tensor Properties: Each push tensor is created with specific properties
 *   including data type, dimensions, element types, identifiers, and UUIDs.
 *
 * - Data Pushing: Applications can push data samples with timing metadata
 *   to the runtime, which then becomes available through the tensor data
 *   interface.
 *
 * Typical usage flow:
 * 1. Query system for push tensor support
 * 2. Create a push tensor from the session with desired properties
 * 3. Receive the tensor collection ID for the created tensor
 * 4. Push data samples with timing information
 * 5. Destroy the push tensor when done
 *
 * The extension is designed to work seamlessly with the XR_NVX1_tensor_data
 * extension, allowing data pushed through this interface to be consumed through
 * the tensor data retrieval API.
 */

/*
 *
 * Constant definitions.
 *
 */

// Extension number 851 (850 prefix)
#define XR_NVX1_push_tensor 1
#define XR_NVX1_push_tensor_SPEC_VERSION 1
#define XR_NVX1_PUSH_TENSOR_EXTENSION_NAME "XR_NVX1_push_tensor"

/*
 *
 * Handle definitions.
 *
 */

XR_DEFINE_HANDLE(XrPushTensorCollectionNV)

/*
 *
 * Result enums and struct enums.
 *
 */

XR_STRUCT_ENUM(XR_TYPE_SYSTEM_PUSH_TENSOR_PROPERTIES_NV, 1000850001);
XR_STRUCT_ENUM(XR_TYPE_PUSH_TENSOR_CREATE_INFO_NV, 1000850002);
XR_STRUCT_ENUM(XR_TYPE_PUSH_TENSOR_COLLECTION_CREATE_INFO_NV, 1000850003);
XR_STRUCT_ENUM(XR_TYPE_PUSH_TENSOR_COLLECTION_CREATE_RESULT_NV, 1000850004);
XR_STRUCT_ENUM(XR_TYPE_PUSH_TENSOR_COLLECTION_DATA_NV, 1000850005);
XR_STRUCT_ENUM(XR_TYPE_PUSH_TENSOR_DLPACK_CREATE_INFO_NV, 1000850006);
XR_STRUCT_ENUM(XR_TYPE_PUSH_TENSOR_ELEMENT_CREATE_INFO_NV, 1000850007);


/*
 *
 * Typed structs for function arguments.
 *
 */

// XrSystemPushTensorPropertiesNV extends XrSystemProperties
typedef struct XrSystemPushTensorPropertiesNV {
    XrStructureType                      type; // XR_TYPE_SYSTEM_PUSH_TENSOR_PROPERTIES_NV
    void* XR_MAY_ALIAS                   next;
    XrBool32                             supportsPushTensors;
} XrSystemPushTensorPropertiesNV;

/*!
 * Information for one tensor to be created in the push tensor collection.
 *
 * For example:
 * - If dataType is XR_TENSOR_DATA_TYPE_OPAQUE_FIXED_SIZE_DATA_NV, no additional
 *   extension structure is required
 * - If dataType is XR_TENSOR_DATA_TYPE_DLPACK_NV, chain XrPushTensorDlpackCreateInfoNV
 * - If dataType is XR_TENSOR_DATA_TYPE_ELEMENT_NV, chain XrPushTensorElementCreateInfoNV
 *
 * Note: XR_TENSOR_DATA_TYPE_UNKNOWN_NV is not allowed and must not be used.
 */
typedef struct XrPushTensorCreateInfoNV {
    XrStructureType                      type; // XR_TYPE_PUSH_TENSOR_CREATE_INFO_NV
    const void* XR_MAY_ALIAS             next;

    XrTensorPropertiesDataNV             properties;
} XrPushTensorCreateInfoNV;

/*!
 * Information for creating a push tensor.
 *
 * The XrTensorCollectionPropertiesData structure (and its chained extension structures)
 * define the properties of the tensor to be created. Applications must provide
 * appropriate extension structures based on the dataType specified.
 */
typedef struct XrPushTensorCollectionCreateInfoNV {
    XrStructureType                      type; // XR_TYPE_PUSH_TENSOR_COLLECTION_CREATE_INFO_NV
    const void* XR_MAY_ALIAS             next;

    /*!
     * Array of tensor create infos, the length of the array is specified by the
     * tensorCount member of the XrTensorCollectionPropertiesData structure.
     */
    const XrPushTensorCreateInfoNV*      tensors;

    /*!
     * Tensor collection properties data. The tensor collection ID will be
     * assigned by the runtime when the tensor collection is created.
     */
    XrTensorCollectionPropertiesData     data;
} XrPushTensorCollectionCreateInfoNV;

/*!
 * Result information from creating a push tensor.
 */
typedef struct XrPushTensorCollectionCreateResultNV {
    XrStructureType                      type; // XR_TYPE_PUSH_TENSOR_COLLECTION_CREATE_RESULT_NV
    void* XR_MAY_ALIAS                   next;

    /*!
     * The tensor collection ID assigned to this push tensor.
     * This collection ID can be used to retrieve the tensor data through the
     * XR_NVX1_tensor_data extension.
     */
    XrTensorCollectionIDNV               tensorCollectionId;
} XrPushTensorCollectionCreateResultNV;

/*!
 * Data to push into a tensor.
 */
typedef struct XrPushTensorCollectionDataNV {
    XrStructureType                      type; // XR_TYPE_PUSH_TENSOR_COLLECTION_DATA_NV
    const void* XR_MAY_ALIAS             next;

    /*!
     * The time for this sample, in OpenXR's time domain. See the documentation
     * for XrTensorSampleMetadataNV for more information.
     *
     * If not available, this will be 0.
     */
    XrTime                               timestamp;

    /*!
     * The raw device timestamp, in nanoseconds. See the documentation for
     * XrTensorSampleMetadataNV for more information.
     *
     * If not available, this will be 0.
     */
    uint64_t                             rawDeviceTimestamp;

    /*!
     * Pointer to the tensor collection sample data buffer.
     * The data must match the format specified when creating the tensor.
     */
    const uint8_t*                       buffer;

    /*!
     * Number of bytes in the buffer.
     * Must be >= totalSampleSize specified at creation.
     */
    uint32_t                             bufferSize;
} XrPushTensorCollectionDataNV;

/*!
 * This holds the data decoding description for tensors that encode data via the
 * DLPack format.
 */
// XrPushTensorDlpackCreateInfoNV extends XrPushTensorCollectionCreateInfoNV
typedef struct XrPushTensorDlpackCreateInfoNV {
    XrStructureType                      type; // XR_TYPE_PUSH_TENSOR_DLPACK_CREATE_INFO_NV
    const void* XR_MAY_ALIAS             next;

    /*!
     * DLPack properties data for the tensor.
     */
    XrTensorDlpackPropertiesDataNV       data;
} XrPushTensorDlpackCreateInfoNV;

// XrPushTensorElementCreateInfoNV extends XrPushTensorCollectionCreateInfoNV
typedef struct XrPushTensorElementCreateInfoNV {
    XrStructureType                      type; // XR_TYPE_PUSH_TENSOR_ELEMENT_CREATE_INFO_NV
    const void* XR_MAY_ALIAS             next;

    XrTensorElementTypeNV                elementType;
    uint32_t                             elementDimensionCount;
    uint32_t                             elementDimensions[XR_MAX_TENSOR_ELEMENT_DIMENSIONS];
} XrPushTensorElementCreateInfoNV;


/*
 *
 * Function declarations.
 *
 */

/*!
 * Function pointer type for @ep{xrCreatePushTensorCollectionNV}
 *
 * Creates a push tensor that applications can use to push data into the runtime.
 *
 * @param session The session handle.
 * @param createInfo Information about the tensor to create.
 * @param createResult Output parameter for creation result (can be NULL).
 * @param pushTensorCollection Output parameter for the created push tensor handle.
 * @return XR_SUCCESS if the tensor was created successfully, otherwise an error code.
 */
typedef XrResult (XRAPI_PTR *PFN_xrCreatePushTensorCollectionNV)(
    XrSession session,
    const XrPushTensorCollectionCreateInfoNV *createInfo,
    XrPushTensorCollectionCreateResultNV *createResult,
    XrPushTensorCollectionNV *pushTensorCollection);

/*!
 * Function pointer type for @ep{xrDestroyPushTensorCollectionNV}
 *
 * Destroys a push tensor and releases its resources.
 * After destruction, the tensor collection ID is no longer valid and the tensor
 * will be removed from the tensor list.
 *
 * @param pushTensorCollection The push tensor handle to destroy.
 * @return XR_SUCCESS if the tensor was destroyed successfully, otherwise an error code.
 */
typedef XrResult (XRAPI_PTR *PFN_xrDestroyPushTensorCollectionNV)(XrPushTensorCollectionNV pushTensorCollection);

/*!
 * Function pointer type for @ep{xrPushTensorCollectionDataNV}
 *
 * Pushes a data sample into the tensor.
 * The data becomes available through the tensor data retrieval API.
 *
 * @param pushTensorCollection The push tensor handle.
 * @param tensorCollectionData The data to push, including timing metadata.
 * @return XR_SUCCESS if the data was pushed successfully, otherwise an error code.
 */
typedef XrResult (XRAPI_PTR *PFN_xrPushTensorCollectionDataNV)(
    XrPushTensorCollectionNV pushTensorCollection,
    const XrPushTensorCollectionDataNV *tensorCollectionData);

#ifdef __cplusplus
}
#endif
