// Copyright 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0 OR MIT
/*!
 * @file
 * @brief Header for OpenXR tensor data interface.
 * @ingroup external_openxr
 */

#pragma once

#include "openxr_extension_helpers.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * General documentation for the XR_NVX1_tensor_data extension:
 *
 * This extension provides a generic interface for streaming tensor data from
 * OpenXR runtimes to applications. It enables runtimes to expose various types
 * of data (sensor readings, ML inference results, tracking data, etc.) in a
 * standardized tensor format that applications can discover and consume.
 *
 * Key concepts:
 * - Tensor List: A versioned list of tensor collections available from the runtime.
 *   The list has a generation number that increments when collections are added or
 *   removed, allowing applications to detect when their cached view is stale.
 *
 * - Tensor Collections: Each entry in the tensor list is a collection that can
 *   contain one or more individual tensors. Collections have metadata like
 *   identifiers, UUIDs, and total byte size.
 *
 * - Individual Tensors: Each tensor within a collection has its own properties
 *   including data type, size, offset within the collection, and identifier.
 *   Applications query these properties to understand the data format.
 *
 * - Sample Retrieval: Applications retrieve data samples per collection (containing
 *   all tensors). Samples can be retrieved individually or in batches. Each sample
 *   includes timing metadata (device timestamp, arrival timestamp, sample index).
 *
 * - Data Types: Supports element tensors (numeric arrays), DLPack tensors
 *   (for ML framework interoperability), and extensible future types.
 *
 * Typical usage flow:
 * 1. Query system for tensor data support
 * 2. Create a tensor list from the session
 * 3. Query list properties to discover available tensor collections
 * 4. Query collection properties for each entry in the list
 * 5. Query individual tensor properties within each collection
 * 6. Retrieve tensor collection data samples (single or batch)
 * 7. Call xrGetTensorListLatestGenerationNV to get the latest generation number
 *    and compare it to the generation number of the cached list.
 * 8. If the generation number is different, update with xrUpdateTensorListNV.
 * 9. If the generation number is the same, the cached list is still valid.
 *
 * The extension is designed for real-time streaming with timing metadata that
 * acknowledges clock drift and network latency, while also providing raw device
 * timestamps for offline post-processing scenarios.
 */

/*
 *
 * Constant definitions.
 *
 */

#define XR_MAX_TENSOR_IDENTIFIER_SIZE 256
#define XR_MAX_TENSOR_LOCALIZED_NAME_SIZE 256
#define XR_MAX_TENSOR_DLPACK_DIMENSIONS 32
#define XR_MAX_TENSOR_ELEMENT_DIMENSIONS 32

// Extension number 850 (849 prefix)
#define XR_NVX1_tensor_data 1
#define XR_NVX1_tensor_data_SPEC_VERSION 1
#define XR_NVX1_TENSOR_DATA_EXTENSION_NAME "XR_NVX1_tensor_data"

/*
 *
 * Handle definitions.
 *
 */

XR_DEFINE_HANDLE(XrTensorListNV)

XR_DEFINE_ATOM(XrTensorCollectionIDNV)

/*
 *
 * Result enums and struct enums.
 *
 */

// If a tensor element of a list is no longer valid, the runtime will return this error.
XR_RESULT_ENUM(XR_ERROR_TENSOR_LOST_NV, -1000849000);
XR_RESULT_ENUM(XR_ERROR_INVALID_TENSOR_INDEX_NV, -1000849001);

XR_STRUCT_ENUM(XR_TYPE_SYSTEM_TENSOR_DATA_PROPERTIES_NV, 1000849001);
XR_STRUCT_ENUM(XR_TYPE_CREATE_TENSOR_LIST_INFO_NV, 1000849002);
XR_STRUCT_ENUM(XR_TYPE_SYSTEM_TENSOR_LIST_PROPERTIES_NV, 1000849003);
XR_STRUCT_ENUM(XR_TYPE_TENSOR_PROPERTIES_NV, 1000849004);
XR_STRUCT_ENUM(XR_TYPE_TENSOR_COLLECTION_PROPERTIES_NV, 1000849005);
XR_STRUCT_ENUM(XR_TYPE_TENSOR_DATA_RETRIEVAL_INFO_NV, 1000849006);
XR_STRUCT_ENUM(XR_TYPE_TENSOR_DATA_NV, 1000849007);
XR_STRUCT_ENUM(XR_TYPE_TENSOR_ELEMENT_PROPERTIES_NV, 1000849008);
XR_STRUCT_ENUM(XR_TYPE_TENSOR_DLPACK_PROPERTIES_NV, 1000849009);


/*
 *
 * Typed enum definitions.
 *
 */

typedef enum XrTensorDataTypeNV {
    /*!
     * The tensor is of a type that is not known to the application, it has not
     * enabled the required extension for the needed data type.
     */
    XR_TENSOR_DATA_TYPE_UNKNOWN_NV = 1000849000,

    /*!
     * The tensor source has not provided any data decoding description to the
     * runtime, the minimum size and stride is always required and is part of
     * the base tensor properties.
     */
    XR_TENSOR_DATA_TYPE_OPAQUE_FIXED_SIZE_DATA_NV = 1000849001,


    /*!
     * The data decoding description follows the DLPack interface, see the
     * associated structures.
     */
    XR_TENSOR_DATA_TYPE_DLPACK_NV = 1000849002,

    /*!
     * Element tensor data type.
     */
    XR_TENSOR_DATA_TYPE_ELEMENT_NV = 1000849003,

    XR_TENSOR_DATA_TYPE_MAX_ENUM = 0x7FFFFFFF,
} XrTensorDataTypeNV;


/*
 *
 * Non-typed structs.
 *
 */

 /*!
  * This struct is used to store the properties of a tensor collection.
  *
  * It is broken out into a separate struct to allow for easy re-use for
  * future extensions, such as the push tensor extension.
  */
 typedef struct XrTensorCollectionPropertiesData {
    /*!
     * Collection identifier string. If the first byte is zero, the identifier is
     * not valid (the runtime must zero all trailing bytes).
     */
    char                                 identifier[XR_MAX_TENSOR_IDENTIFIER_SIZE];

    /*!
     * Localized display name for the collection. If the first byte is zero, the
     * name is not valid (the runtime must zero all trailing bytes).
     */
    char                                 localizedName[XR_MAX_TENSOR_LOCALIZED_NAME_SIZE];

    /*!
     * Collection UUID. If all bytes are zero, the UUID is not valid.
     */
    XrUuidEXT                            uuid;

    /*!
     * Number of tensors in this collection.
     */
    uint32_t                             tensorCount;

    /*!
     * Total byte size of all tensors in the collection combined, this is not
     * the same as the stride of the sample. The stride might be larger than the
     * total sample size, if there is padding between samples.
     */
    uint32_t                             totalSampleSize;
} XrTensorCollectionPropertiesData;

/*!
 * This struct is used to store the properties of an individual tensor within a tensor collection.
 *
 * It is broken out into a separate struct to allow for easy re-use for
 * future extensions, such as the push tensor extension.
 */
 typedef struct XrTensorPropertiesDataNV {
    /*!
     * The data type of the tensor. Additional type-specific properties for the
     * tensor (such as DLPack properties) are chained in by the application via
     * the next pointer in the containing structure like XrTensorPropertiesNV.
     */
    XrTensorDataTypeNV                   dataType;

    /*!
     * The raw size of individual tensor data in bytes. This field is always
     * provided by the runtime, even when no data decoding descriptor is
     * available (such as with XR_TENSOR_DATA_TYPE_OPAQUE_FIXED_SIZE_DATA_NV) or
     * when the runtime cannot interpret the tensor format. This ensures
     * applications can always allocate appropriately sized buffers for data
     * retrieval, regardless of whether type-specific properties (like
     * XrTensorDlpackPropertiesNV) are present or understood by the application.
     */
    uint32_t                             dataTypeSize;

    /*!
     * The byte offset of the individual tensor data within the tensor
     * collection sample.
     */
    uint32_t                             offset;

    /*!
     * Tensor identifier string. If the first byte is zero, the identifier is
     * not valid (the runtime must zero all trailing bytes).
     */
    char                                 identifier[XR_MAX_TENSOR_IDENTIFIER_SIZE];
} XrTensorPropertiesDataNV;

typedef struct XrTensorSampleMetadataNV {
    /*!
     * Monotonically increasing index of the sample, starting from 0.
     */
    int64_t                              sampleIndex;

    /*!
     * The raw device timestamp, in nanoseconds. Because the offset between the
     * device's clock and the runtime's clock is an estimate, they may drift
     * over time. If not available, this will be 0.
     */
    uint64_t                             rawDeviceTimestamp;

    /*!
     * When was this sample generated.
     *
     * In general this timestamp is less accurate than if all of the data was
     * processed in an offline fashion, since this is generated in realtime.
     * - The data might be captured on a different device that could be on a
     *   different clock.
     * - The data is processed in realtime and as such the estimate of the
     *   offset to the device's clock and the runtime's clock is not perfect.
     * - In some cases the data is streamed over the internet.
     * - Most XR devices don't ship with atomic clocks (or servers for that
     *   matter).
     *
     * It is good enough to run an XR application, but might not be good enough
     * for scientific purposes, as such the rawDeviceTimestamp is also provided,
     * to allow the application to save the samples and process them offline.
     */
    XrTime                               timestamp;

    /*!
     * When did this sample arrive at the runtime, in OpenXR's time domain.
     * Note that this is not the same as when the sample was generated.
     */
    XrTime                               arrivalTimestamp;
} XrTensorSampleMetadataNV;


/*
 *
 * Typed structs for function arguments.
 *
 */

// XrSystemTensorDataPropertiesNV extends XrSystemProperties
typedef struct XrSystemTensorDataPropertiesNV {
    XrStructureType                      type; // XR_TYPE_SYSTEM_TENSOR_DATA_PROPERTIES_NV
    void* XR_MAY_ALIAS                   next;
    XrBool32                             supportsTensorData;
} XrSystemTensorDataPropertiesNV;

typedef struct XrCreateTensorListInfoNV {
    XrStructureType                      type; // XR_TYPE_CREATE_TENSOR_LIST_INFO_NV
    const void* XR_MAY_ALIAS             next;
} XrCreateTensorListInfoNV;

typedef struct XrSystemTensorListPropertiesNV {
    XrStructureType                      type; // XR_TYPE_SYSTEM_TENSOR_LIST_PROPERTIES_NV
    void* XR_MAY_ALIAS                   next;

    /*!
     * The generation number of the tensor list. This is incremented by the
     * runtime whenever the tensor list is updated (and a snapshot of the tensor
     * list is taken). This is used to determine if the cached tensor list is
     * still valid.
     */
    uint64_t                             generationNumber;

    /*!
     * The number of tensor collections in the list.
     */
    uint32_t                             tensorCollectionCount;
} XrSystemTensorListPropertiesNV;

typedef struct XrTensorCollectionPropertiesNV {
    XrStructureType                      type; // XR_TYPE_TENSOR_COLLECTION_PROPERTIES_NV
    void* XR_MAY_ALIAS                   next;

    /*!
     * This is unique to this instance of the runtime, and the instance of the
     * internal tensor object. If a provider of a tensor object destroys its
     * tensor the collection ID will not be reused.
     *
     * This lets the application track the tensor object across generations,
     * and even if an instance of the tensor object is destroyed and recreated,
     * with exactly the same identifier and UUID, the application will still be
     * able to detect that it is a new tensor object.
     */
    XrTensorCollectionIDNV               tensorCollectionId;

    /*!
     * The stride in bytes between consecutive samples in a buffer for batch
     * retrieval. The app must allocate enough memory for X * sampleBatchStride
     * bytes to receive X samples. This is only used for the batch read call,
     * and allows the runtime to optimize memory copies depending on the hardware
     * doing the copies (CPU / GPU / DMA-Engine etc), while also letting the
     * application allocate the receiving buffer ahead of time.
     */
    uint32_t                             sampleBatchStride;

    XrTensorCollectionPropertiesData     data;
} XrTensorCollectionPropertiesNV;

/*!
 * This struct is used to store the properties of a tensor.
 */
typedef struct XrTensorPropertiesNV {
    XrStructureType                      type; // XR_TYPE_TENSOR_PROPERTIES_NV
    void* XR_MAY_ALIAS                   next;

    XrTensorPropertiesDataNV             data;
} XrTensorPropertiesNV;

typedef struct XrTensorDataRetrievalInfoNV {
    XrStructureType                      type; // XR_TYPE_TENSOR_DATA_RETRIEVAL_INFO_NV
    const void* XR_MAY_ALIAS             next;

    /*!
     * The index of the tensor collection to retrieve data from. This indexes
     * into the tensor list, not the tensor collection ID itself.
     */
    uint32_t                             tensorCollectionIndex;

    /*!
     * The sample index to retrieve from. This can be:
     * - A non-negative value: Retrieve samples starting from this index or later.
     *   The runtime will return samples with
     *   @code{cpp} index >= startSampleIndex @endcode. For example, if
     *   @p startSampleIndex is 0, the runtime will return the oldest available
     *   sample.
     * - A negative value: Index relative to the latest sample, where -1 is
     *   the newest sample, -2 is the second newest, and so on. The runtime
     *   calculates the absolute index as
     *   @code{cpp} (latest_sample_index + startSampleIndex + 1) @endcode,
     *   clamped to be @code{cpp} >= 0 @endcode. For example, if the latest
     *   sample is 100 and @p startSampleIndex is -120, this resolves to
     *   sample index 0. After conversion, the same rules as positive indices
     *   apply.
     *
     * Note that if the sample at this exact index is not available, the runtime
     * will return the nearest available sample that satisfies the constraint.
     *
     * If the resolved absolute index is larger than the latest sample index,
     * the runtime will not return any samples.
     */
    int64_t                              startSampleIndex;
} XrTensorDataRetrievalInfoNV;

typedef struct XrTensorDataNV {
    XrStructureType                      type; // XR_TYPE_TENSOR_DATA_NV
    void* XR_MAY_ALIAS                   next;

    /*!
     * Array of metadata structures to be filled by the runtime.
     * Must have at least metadataCapacity elements.
     */
    XrTensorSampleMetadataNV*            metadataArray;

    /*!
     * The capacity of the metadata array, the maximum number of samples to
     * retrieve.
     */
    uint32_t                             metadataCapacity;

    /*!
     * Memory allocated by the application to store all tensor data samples.
     * Samples are stored contiguously using the stride from tensor properties.
     * The runtime will write the samples to the buffer in the order they are
     * retrieved.
     */
    void*                                buffer;

    /*!
     * The capacity of the buffer in bytes, the maximum number of bytes that the
     * runtime can write to the buffer.
     *
     * Must be >= metadataCapacity * tensorCollectionProperties->sampleBatchStride.
     */
    uint32_t                             bufferCapacity;

    /*!
     * Filled in by the runtime: actual number of samples written to the
     * metadata array and buffer.
     */
    uint32_t                             writtenSampleCount;
} XrTensorDataNV;


/*
 *
 * DLPack type definitions.
 *
 */

/*!
 * This struct is used to store the DLPack properties of a tensor.
 *
 * It is broken out into a separate struct to allow for easy re-use for
 * future extensions, such as the push tensor extension.
 */
 typedef struct XrTensorDlpackPropertiesDataNV {
    uint32_t                             versionMajor;
    uint32_t                             versionMinor;
    uint32_t                             dtype;
    int32_t                              ndim;
    int64_t                              shape[XR_MAX_TENSOR_DLPACK_DIMENSIONS];
    int64_t                              strides[XR_MAX_TENSOR_DLPACK_DIMENSIONS];
    int64_t                              byte_offset;
} XrTensorDlpackPropertiesDataNV;

// XrTensorDlpackPropertiesNV extends XrTensorCollectionPropertiesNV
typedef struct XrTensorDlpackPropertiesNV {
    XrStructureType                      type; // XR_TYPE_TENSOR_DLPACK_PROPERTIES_NV
    void* XR_MAY_ALIAS                   next;

    /*!
     * DLPack properties data. Only supports CPU tensors.
     */
    XrTensorDlpackPropertiesDataNV       data;
} XrTensorDlpackPropertiesNV;


/*
 *
 * Element type definitions.
 *
 */

typedef enum XrTensorElementTypeNV {
    XR_TENSOR_ELEMENT_TYPE_FLOAT32_NV = 1000849000,
    XR_TENSOR_ELEMENT_TYPE_FLOAT64_NV = 1000849001,
    XR_TENSOR_ELEMENT_TYPE_INT32_NV = 1000849002,
    XR_TENSOR_ELEMENT_TYPE_INT64_NV = 1000849003,
    XR_TENSOR_ELEMENT_TYPE_UINT32_NV = 1000849004,
    XR_TENSOR_ELEMENT_TYPE_UINT64_NV = 1000849005,

    // A XrTime timestamp, in OpenXR's time domain.
    XR_TENSOR_ELEMENT_TYPE_XR_TIME_NV = 1000849006,

    // A XrDuration time duration, in OpenXR's time domain.
    XR_TENSOR_ELEMENT_TYPE_XR_TIME_DURATION_NV = 1000849007,

    // Timestamp in a non-OpenXR time domain, in nanoseconds.
    XR_TENSOR_ELEMENT_TYPE_TIMESTAMP_NS_NV = 1000849008,

    // Time duration in a non-OpenXR time domain, in nanoseconds.
    XR_TENSOR_ELEMENT_TYPE_TIME_DURATION_NS_NV = 1000849009,

    /*!
     * Fixed size opaque data, the application will need to get the schema of
     * the tensor data outside of the runtime to interpret the data, such as
     * documentation or an external schema file.
     */
    XR_TENSOR_ELEMENT_TYPE_OPAQUE_DATA_NV = 1000849010,

    XR_TENSOR_ELEMENT_TYPE_MAX_ENUM = 0x7FFFFFFF,
} XrTensorElementTypeNV;

// XrTensorElementPropertiesNV extends XrTensorCollectionPropertiesNV
typedef struct XrTensorElementPropertiesNV {
    XrStructureType                      type; // XR_TYPE_TENSOR_ELEMENT_PROPERTIES_NV
    void* XR_MAY_ALIAS                   next;
    XrTensorElementTypeNV                elementType;
    uint32_t                             elementDimensionCount;
    uint32_t                             elementDimensions[XR_MAX_TENSOR_ELEMENT_DIMENSIONS];
} XrTensorElementPropertiesNV;


/*
 *
 * Function declarations.
 *
 */

/*!
 * Function pointer type for @ep{xrGetTensorListLatestGenerationNV}
 *
 * Gets the latest generation number from the session/system. The tensor list
 * is a cached view of the system state, and this function queries the session
 * directly for the current generation to determine if the cached list needs
 * to be updated.
 *
 * Thread Safety: This function is NOT externally synchronized and is safe to
 * call from multiple threads on the same session object.
 *
 * @param session The session handle.
 * @param outGenerationNumber Output parameter for the latest generation number.
 * @return XR_SUCCESS if the latest generation number was retrieved
 *         successfully, otherwise an error code.
 */
typedef XrResult (XRAPI_PTR *PFN_xrGetTensorListLatestGenerationNV)(XrSession session, uint64_t *outGenerationNumber);

/*!
 * Creates a tensor list from the session.
 *
 * Thread Safety: This function is NOT externally synchronized and is safe to
 * call from multiple threads on the same session object.
 *
 * @param session The session handle.
 * @param info Creation info for the tensor list.
 * @param tensorList Output parameter for the created tensor list handle.
 * @return XR_SUCCESS if the tensor list was created successfully, otherwise an error code.
 */
typedef XrResult (XRAPI_PTR *PFN_xrCreateTensorListNV)(XrSession session, const XrCreateTensorListInfoNV *info, XrTensorListNV *tensorList);

/*!
 * Get properties of the tensor list.
 *
 * Thread Safety: This function is NOT externally synchronized and is safe to
 * call from multiple threads on the same tensor list object.
 *
 * @param tensorList The tensor list handle.
 * @param properties Output parameter for the tensor list properties.
 * @return XR_SUCCESS if properties were retrieved successfully, otherwise an error code.
 */
typedef XrResult (XRAPI_PTR *PFN_xrGetTensorListPropertiesNV)(XrTensorListNV tensorList, XrSystemTensorListPropertiesNV *properties);

/*!
 * Destroys a tensor list.
 *
 * Thread Safety: This function is externally synchronized, like all destroy functions.
 *
 * @param tensorList The tensor list handle to destroy.
 * @return XR_SUCCESS if the tensor list was destroyed successfully, otherwise an error code.
 */
typedef XrResult (XRAPI_PTR *PFN_xrDestroyTensorListNV)(XrTensorListNV tensorList);

/*!
 * Updates the tensor list to reflect the current state of available tensors.
 *
 * IMPORTANT: This function is externally synchronized with all other tensor list
 * functions. Applications should avoid calling this function unless necessary,
 * as it requires exclusive access and may block other operations on the list.
 *
 * To minimize calls to this function, applications should cache the generation
 * number and only update when the list has changed:
 *
 * Example usage pattern:
 * @code{cpp}
 * // Cache the generation number from the current list
 * XrSystemTensorListPropertiesNV listProps = {XR_TYPE_SYSTEM_TENSOR_LIST_PROPERTIES_NV};
 * xrGetTensorListPropertiesNV(tensorList, &listProps);
 * uint64_t cachedGeneration = listProps.generationNumber;
 *
 * // Later, check if an update is needed
 * uint64_t latestGeneration;
 * xrGetTensorListLatestGenerationNV(session, &latestGeneration);
 *
 * if (latestGeneration != cachedGeneration) {
 *     // List has changed, update is needed
 *     xrUpdateTensorListNV(tensorList);
 *     cachedGeneration = latestGeneration;
 * }
 * // else: cached list is still valid, no update needed
 * @endcode
 *
 * @note xrCreateTensorListNV implicitly performs an initial update, so calling
 *       this function immediately after creation is unnecessary.
 *
 * @param tensorList The tensor list handle to update.
 * @return XR_SUCCESS if the list was updated successfully, otherwise an error code.
 */
typedef XrResult (XRAPI_PTR *PFN_xrUpdateTensorListNV)(XrTensorListNV tensorList);

/*!
 * Get properties for a specific tensor collection from the tensor list.
 *
 * This function retrieves both the base tensor properties and any type-specific
 * properties. Type-specific properties (such as XrTensorDlpackPropertiesNV) can
 * be chained to the properties structure via the next pointer.
 *
 * A simple way to handle different tensor data types is to chain all possible
 * type-specific property structures together. The runtime will fill in the
 * appropriate structure based on the tensor's data type and leave the others
 * unmodified. This allows applications to handle multiple tensor types without
 * needing to query the data type first.
 *
 * Thread Safety: This function is NOT externally synchronized and is safe to
 * call from multiple threads on the same tensor list object.
 *
 * @param tensorList The tensor list handle.
 * @param index The index of the tensor collection in the list (0-based).
 * @param properties Pointer to XrTensorCollectionPropertiesNV structure to receive the
 *                   tensor collection properties. Type-specific properties can be chained
 *                   via the next pointer.
 * @return XR_SUCCESS if properties were retrieved successfully,
 *         XR_ERROR_INVALID_TENSOR_INDEX_NV if the index is out of range,
 *         otherwise an appropriate error code.
 */
typedef XrResult (XRAPI_PTR *PFN_xrGetTensorCollectionPropertiesNV)(XrTensorListNV tensorList, uint32_t index, XrTensorCollectionPropertiesNV *properties);

/*!
 * Get properties for a specific tensor from a tensor collection.
 *
 * @param tensorList The tensor list handle.
 * @param collection_index The index of the collection in the list (0-based).
 * @param tensor_index The index of the tensor in the collection (0-based).
 * @param properties Pointer to XrTensorPropertiesNV structure to receive the
 *                   tensor properties.
 * @return XR_SUCCESS if properties were retrieved successfully,
 *         XR_ERROR_INVALID_TENSOR_INDEX_NV if the index is out of range,
 *         otherwise an appropriate error code.
 */
typedef XrResult (XRAPI_PTR *PFN_xrGetTensorPropertiesNV)(XrTensorListNV tensorList, uint32_t collection_index, uint32_t tensor_index, XrTensorPropertiesNV *tensorProperties);

/*!
 * Get multiple tensor data samples in a batch, this function does not follow
 * the two call pattern common in OpenXR. The app decides how many samples to
 * retrieve and how much memory to allocate for the buffer.
 *
 * Thread Safety: This function is NOT externally synchronized and is safe to
 * call from multiple threads on the same tensor list object.
 *
 * @param tensorList The tensor list handle.
 * @param retrievalInfo Information about which tensor and which samples to retrieve.
 * @param tensorDataBatch Structure to receive multiple tensor data samples and metadata.
 * @return XR_SUCCESS if data was retrieved, otherwise an error code.
 */
typedef XrResult (XRAPI_PTR *PFN_xrGetTensorDataNV)(XrTensorListNV tensorList, const XrTensorDataRetrievalInfoNV *retrievalInfo, XrTensorDataNV *tensorData);

#ifdef __cplusplus
}
#endif
