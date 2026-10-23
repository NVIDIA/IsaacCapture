// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/live_trackers/schema_tracker_base.hpp"

#include <openxr/openxr.h>
#include <oxr_utils/oxr_funcs.hpp>

#include <XR_NVX1_tensor_data.h>
#include <cassert>
#include <cstring>
#include <stdexcept>

namespace core
{

namespace
{

bool is_known_non_fatal_tensor_unavailable(XrResult result)
{
    return result == XR_ERROR_TENSOR_LOST_NV;
}

} // namespace

// =============================================================================
// SchemaTracker Implementation
// =============================================================================

SchemaTrackerBase::SchemaTrackerBase(const OpenXRSessionHandles& handles, SchemaTrackerConfig config)
    : m_handles(handles), m_config(std::move(config)), m_time_converter(handles)
{
    // Validate handles
    assert(handles.instance != XR_NULL_HANDLE && "OpenXR instance handle cannot be null");
    assert(handles.session != XR_NULL_HANDLE && "OpenXR session handle cannot be null");
    assert(handles.xrGetInstanceProcAddr && "xrGetInstanceProcAddr cannot be null");

    // Initialize extension functions using the provided xrGetInstanceProcAddr
    initialize_tensor_data_functions();

    // Create tensor list
    create_tensor_list();

    m_logger->info("SchemaTracker initialized, looking for collection: {}", m_config.collection_id);
}

SchemaTrackerBase::~SchemaTrackerBase()
{
    // m_tensor_list is guaranteed to be non-null by create_tensor_list(), or the constructor would have thrown.
    assert(m_tensor_list != XR_NULL_HANDLE && m_destroy_list_fn != nullptr);

    XrResult result = m_destroy_list_fn(m_tensor_list);
    if (result != XR_SUCCESS)
    {
        m_logger->warn("Failed to destroy tensor list, result={}", static_cast<int>(result));
    }
}

std::vector<std::string> SchemaTrackerBase::get_required_extensions()
{
    std::vector<std::string> exts = { "XR_NVX1_tensor_data" };
    for (const auto& ext : XrTimeConverter::get_required_extensions())
    {
        exts.push_back(ext);
    }
    return exts;
}

bool SchemaTrackerBase::ensure_collection()
{
    uint64_t latest_generation = 0;
    XrResult result = m_get_latest_gen_fn(m_handles.session, &latest_generation);
    if (result != XR_SUCCESS)
    {
        if (is_known_non_fatal_tensor_unavailable(result))
        {
            m_target_collection.reset();
            return false;
        }
        throw std::runtime_error("Failed to get latest generation, result=" + std::to_string(result));
    }

    if (latest_generation == m_cached_generation && m_target_collection.has_value())
    {
        return true;
    }

    if (latest_generation != m_cached_generation)
    {
        result = m_update_list_fn(m_tensor_list);
        if (result != XR_SUCCESS)
        {
            if (is_known_non_fatal_tensor_unavailable(result))
            {
                m_target_collection.reset();
                return false;
            }
            throw std::runtime_error("Failed to update tensor list, result=" + std::to_string(result));
        }
        m_cached_generation = latest_generation;
    }

    XrSystemTensorListPropertiesNV list_props{ XR_TYPE_SYSTEM_TENSOR_LIST_PROPERTIES_NV };
    result = m_get_list_props_fn(m_tensor_list, &list_props);
    if (result != XR_SUCCESS)
    {
        if (is_known_non_fatal_tensor_unavailable(result))
        {
            m_target_collection.reset();
            return false;
        }
        throw std::runtime_error("Failed to get list properties, result=" + std::to_string(result));
    }

    for (uint32_t i = 0; i < list_props.tensorCollectionCount; ++i)
    {
        XrTensorCollectionPropertiesNV coll_props{ XR_TYPE_TENSOR_COLLECTION_PROPERTIES_NV };
        result = m_get_coll_props_fn(m_tensor_list, i, &coll_props);
        if (result != XR_SUCCESS)
        {
            if (is_known_non_fatal_tensor_unavailable(result))
            {
                m_target_collection.reset();
                return false;
            }
            throw std::runtime_error("Failed to get collection properties, result=" + std::to_string(result));
        }

        if (std::strncmp(coll_props.data.identifier, m_config.collection_id.c_str(), XR_MAX_TENSOR_IDENTIFIER_SIZE) != 0)
        {
            continue;
        }

        if (coll_props.data.totalSampleSize > m_config.max_flatbuffer_size)
        {
            throw std::runtime_error(
                "SchemaTracker: collection '" + m_config.collection_id +
                "' reports totalSampleSize=" + std::to_string(coll_props.data.totalSampleSize) +
                " which exceeds configured max_flatbuffer_size=" + std::to_string(m_config.max_flatbuffer_size));
        }

        m_sample_batch_stride = coll_props.sampleBatchStride;
        m_sample_size = coll_props.data.totalSampleSize;

        // A recreated collection can reuse its list index, but never its runtime ID.
        if (!m_target_collection || i != m_target_collection->index ||
            coll_props.tensorCollectionId != m_target_collection->id)
        {
            m_last_sample_index.reset();
            m_logger->info("Found target collection at index {}", i);
        }
        m_target_collection = TargetCollection{ i, coll_props.tensorCollectionId };
        return true;
    }

    m_target_collection.reset();
    return false;
}

bool SchemaTrackerBase::read_all_samples(std::vector<SampleResult>& samples)
{
    if (!ensure_collection())
    {
        return false;
    }

    SampleResult result;
    while (read_next_sample(result))
    {
        samples.push_back(std::move(result));
    }

    // read_next_sample clears m_target_collection on API failure.
    // If it's now empty we hit an error mid-drain, not a normal end-of-data.
    if (!m_target_collection.has_value())
    {
        return false;
    }

    return true;
}

const SchemaTrackerConfig& SchemaTrackerBase::config() const
{
    return m_config;
}

void SchemaTrackerBase::initialize_tensor_data_functions()
{
    loadExtensionFunction(m_handles.instance, m_handles.xrGetInstanceProcAddr, "xrGetTensorListLatestGenerationNV",
                          reinterpret_cast<PFN_xrVoidFunction*>(&m_get_latest_gen_fn));
    loadExtensionFunction(m_handles.instance, m_handles.xrGetInstanceProcAddr, "xrCreateTensorListNV",
                          reinterpret_cast<PFN_xrVoidFunction*>(&m_create_list_fn));
    loadExtensionFunction(m_handles.instance, m_handles.xrGetInstanceProcAddr, "xrGetTensorListPropertiesNV",
                          reinterpret_cast<PFN_xrVoidFunction*>(&m_get_list_props_fn));
    loadExtensionFunction(m_handles.instance, m_handles.xrGetInstanceProcAddr, "xrGetTensorCollectionPropertiesNV",
                          reinterpret_cast<PFN_xrVoidFunction*>(&m_get_coll_props_fn));
    loadExtensionFunction(m_handles.instance, m_handles.xrGetInstanceProcAddr, "xrGetTensorDataNV",
                          reinterpret_cast<PFN_xrVoidFunction*>(&m_get_data_fn));
    loadExtensionFunction(m_handles.instance, m_handles.xrGetInstanceProcAddr, "xrUpdateTensorListNV",
                          reinterpret_cast<PFN_xrVoidFunction*>(&m_update_list_fn));
    loadExtensionFunction(m_handles.instance, m_handles.xrGetInstanceProcAddr, "xrDestroyTensorListNV",
                          reinterpret_cast<PFN_xrVoidFunction*>(&m_destroy_list_fn));
}

void SchemaTrackerBase::create_tensor_list()
{
    XrCreateTensorListInfoNV createInfo{ XR_TYPE_CREATE_TENSOR_LIST_INFO_NV };
    createInfo.next = nullptr;

    XrResult result = m_create_list_fn(m_handles.session, &createInfo, &m_tensor_list);
    if (result != XR_SUCCESS)
    {
        throw std::runtime_error("Failed to create tensor list, result=" + std::to_string(result));
    }
}

bool SchemaTrackerBase::read_next_sample(SampleResult& out)
{
    if (!m_target_collection.has_value())
    {
        return false;
    }

    // Prepare retrieval info
    XrTensorDataRetrievalInfoNV retrievalInfo{ XR_TYPE_TENSOR_DATA_RETRIEVAL_INFO_NV };
    retrievalInfo.next = nullptr;
    retrievalInfo.tensorCollectionIndex = m_target_collection->index;
    retrievalInfo.startSampleIndex = m_last_sample_index.has_value() ? m_last_sample_index.value() + 1 : 0;

    // Prepare output buffers (read one sample at a time for simplicity)
    XrTensorSampleMetadataNV metadata{};
    std::vector<uint8_t> dataBuffer(m_sample_batch_stride);

    XrTensorDataNV tensorData{ XR_TYPE_TENSOR_DATA_NV };
    tensorData.next = nullptr;
    tensorData.metadataArray = &metadata;
    tensorData.metadataCapacity = 1;
    tensorData.buffer = dataBuffer.data();
    tensorData.bufferCapacity = static_cast<uint32_t>(dataBuffer.size());
    tensorData.writtenSampleCount = 0;

    // Retrieve samples
    XrResult result = m_get_data_fn(m_tensor_list, &retrievalInfo, &tensorData);
    if (result != XR_SUCCESS)
    {
        if (result == XR_ERROR_TENSOR_LOST_NV)
        {
            // Policy: temporary collection loss is treated as non-fatal.
            m_logger->warn(
                "xrGetTensorDataNV: XR_ERROR_TENSOR_LOST_NV -- tensor/collection data lost; "
                "clearing target collection.");
        }
        else
        {
            m_target_collection.reset();
            throw std::runtime_error("[SchemaTracker] xrGetTensorDataNV failed, result=" + std::to_string(result));
        }
        m_target_collection.reset();
        return false;
    }

    if (tensorData.writtenSampleCount == 0)
    {
        return false; // No new samples
    }

    // Guard against invalid runtime state: batch stride must cover at least one full sample.
    if (dataBuffer.size() < m_sample_size)
    {
        m_logger->warn("read_next_sample: dataBuffer ({} B) is smaller than m_sample_size ({} B); skipping copy.",
                       dataBuffer.size(), m_sample_size);
        return false;
    }

    // Copy data to output buffer (trim to sample size, not batch stride)
    out.buffer.resize(m_sample_size);
    std::memcpy(out.buffer.data(), dataBuffer.data(), m_sample_size);

    // Advance past this sample only after successful copy
    if (!m_last_sample_index.has_value() || metadata.sampleIndex > m_last_sample_index.value())
    {
        m_last_sample_index = metadata.sampleIndex;
    }

    // Convert XrTime values from tensor metadata back to monotonic nanoseconds.
    int64_t available_ns =
        m_time_converter.convert_xrtime_to_monotonic_ns(static_cast<XrTime>(metadata.arrivalTimestamp));
    int64_t sample_ns = m_time_converter.convert_xrtime_to_monotonic_ns(static_cast<XrTime>(metadata.timestamp));

    out.timestamp = DeviceDataTimestamp(available_ns, // available_time_local_common_clock
                                        sample_ns, // sample_time_local_common_clock
                                        static_cast<int64_t>(metadata.rawDeviceTimestamp) // sample_time_raw_device_clock
    );

    return true;
}

} // namespace core
