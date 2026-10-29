// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The XR_NVX1_tensor_data reader functions and timespec conversions SchemaTrackerBase loads,
// served from memory, so a live tracker impl runs without an OpenXR runtime. XrTime is
// nanoseconds on the monotonic clock. One instance at a time.

#include <oxr_utils/oxr_session_handles.hpp>
#include <oxr_utils/oxr_time.hpp>

#include <XR_NVX1_tensor_data.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace tensor_test_support
{

class FakeTensorRuntime
{
public:
    FakeTensorRuntime()
    {
        if (s_current != nullptr)
        {
            throw std::logic_error("FakeTensorRuntime: one instance at a time");
        }
        s_current = this;
    }

    ~FakeTensorRuntime()
    {
        s_current = nullptr;
    }

    FakeTensorRuntime(const FakeTensorRuntime&) = delete;
    FakeTensorRuntime& operator=(const FakeTensorRuntime&) = delete;

    core::OpenXRSessionHandles handles() const
    {
        core::OpenXRSessionHandles handles;
        handles.instance = reinterpret_cast<XrInstance>(1);
        handles.session = reinterpret_cast<XrSession>(1);
        handles.space = reinterpret_cast<XrSpace>(1);
        handles.xrGetInstanceProcAddr = &get_instance_proc_addr;
        return handles;
    }

    //! Creates `collection_id` with samples of `sample_size` bytes, as a producer starting up does.
    void create(const std::string& collection_id, uint32_t sample_size)
    {
        m_collections.push_back(
            std::make_shared<Collection>(Collection{ collection_id, sample_size, m_next_collection_id++ }));
        ++m_generation;
    }

    //! Removes `collection_id`, as a producer going away does.
    void destroy(const std::string& collection_id)
    {
        auto it = find(collection_id);
        (*it)->destroyed = true;
        m_collections.erase(it);
        ++m_generation;
    }

    //! Pushes `payload`, zero-padded to the collection's sample size, as received at `time_ns`.
    void push(const std::string& collection_id, std::span<const uint8_t> payload, int64_t time_ns)
    {
        Collection& collection = **find(collection_id);
        if (payload.size() > collection.sample_size)
        {
            throw std::invalid_argument("FakeTensorRuntime: payload larger than the sample size");
        }
        std::vector<uint8_t> bytes(collection.sample_size, 0);
        std::copy(payload.begin(), payload.end(), bytes.begin());
        collection.samples.push_back({ collection.next_index++, std::move(bytes), time_ns });
    }

private:
    struct Sample
    {
        int64_t index;
        std::vector<uint8_t> bytes;
        int64_t time_ns;
    };

    struct Collection
    {
        std::string id;
        uint32_t sample_size;
        XrTensorCollectionIDNV instance_id;
        std::vector<Sample> samples{};
        int64_t next_index = 0;
        bool destroyed = false;
    };

    using Listed = std::vector<std::shared_ptr<Collection>>;

    std::vector<std::shared_ptr<Collection>>::iterator find(const std::string& collection_id)
    {
        auto it = std::find_if(m_collections.begin(), m_collections.end(),
                               [&](const auto& collection) { return collection->id == collection_id; });
        if (it == m_collections.end())
        {
            throw std::invalid_argument("FakeTensorRuntime: no collection " + collection_id);
        }
        return it;
    }

    //! The collections `list` saw at its last update; a reader holds indices into this.
    static Listed& listed(XrTensorListNV list)
    {
        return s_current->m_lists.at(reinterpret_cast<uintptr_t>(list));
    }

    static XrResult XRAPI_CALL latest_generation(XrSession, uint64_t* generation)
    {
        *generation = s_current->m_generation;
        return XR_SUCCESS;
    }

    static XrResult XRAPI_CALL create_list(XrSession, const XrCreateTensorListInfoNV*, XrTensorListNV* list)
    {
        const uintptr_t handle = s_current->m_next_list++;
        s_current->m_lists[handle] = {};
        *list = reinterpret_cast<XrTensorListNV>(handle);
        return XR_SUCCESS;
    }

    static XrResult XRAPI_CALL update_list(XrTensorListNV list)
    {
        listed(list) = s_current->m_collections;
        return XR_SUCCESS;
    }

    static XrResult XRAPI_CALL list_properties(XrTensorListNV list, XrSystemTensorListPropertiesNV* properties)
    {
        properties->generationNumber = s_current->m_generation;
        properties->tensorCollectionCount = static_cast<uint32_t>(listed(list).size());
        return XR_SUCCESS;
    }

    static XrResult XRAPI_CALL collection_properties(XrTensorListNV list,
                                                     uint32_t index,
                                                     XrTensorCollectionPropertiesNV* properties)
    {
        const Collection& collection = *listed(list).at(index);
        if (collection.destroyed)
        {
            return XR_ERROR_TENSOR_LOST_NV;
        }
        std::strncpy(properties->data.identifier, collection.id.c_str(), XR_MAX_TENSOR_IDENTIFIER_SIZE - 1);
        properties->data.tensorCount = 1;
        properties->data.totalSampleSize = collection.sample_size;
        properties->sampleBatchStride = collection.sample_size;
        properties->tensorCollectionId = collection.instance_id;
        return XR_SUCCESS;
    }

    //! Writes the first sample at or after the requested index, one per call.
    static XrResult XRAPI_CALL get_data(XrTensorListNV list, const XrTensorDataRetrievalInfoNV* info, XrTensorDataNV* data)
    {
        const Collection& collection = *listed(list).at(info->tensorCollectionIndex);
        if (collection.destroyed)
        {
            return XR_ERROR_TENSOR_LOST_NV;
        }
        data->writtenSampleCount = 0;
        for (const Sample& sample : collection.samples)
        {
            if (sample.index < info->startSampleIndex)
            {
                continue;
            }
            std::memcpy(data->buffer, sample.bytes.data(), sample.bytes.size());
            data->metadataArray[0].sampleIndex = sample.index;
            data->metadataArray[0].timestamp = sample.time_ns;
            data->metadataArray[0].arrivalTimestamp = sample.time_ns;
            data->metadataArray[0].rawDeviceTimestamp = static_cast<uint64_t>(sample.time_ns);
            data->writtenSampleCount = 1;
            break;
        }
        return XR_SUCCESS;
    }

    static XrResult XRAPI_CALL destroy_list(XrTensorListNV list)
    {
        s_current->m_lists.erase(reinterpret_cast<uintptr_t>(list));
        return XR_SUCCESS;
    }

    static XrResult XRAPI_CALL timespec_to_time(XrInstance, const struct timespec* ts, XrTime* time)
    {
        *time = static_cast<XrTime>(ts->tv_sec) * 1000000000 + ts->tv_nsec;
        return XR_SUCCESS;
    }

    static XrResult XRAPI_CALL time_to_timespec(XrInstance, XrTime time, struct timespec* ts)
    {
        ts->tv_sec = static_cast<time_t>(time / 1000000000);
        ts->tv_nsec = static_cast<long>(time % 1000000000);
        return XR_SUCCESS;
    }

    static XrResult XRAPI_CALL get_instance_proc_addr(XrInstance, const char* name, PFN_xrVoidFunction* function)
    {
        static const std::map<std::string_view, PFN_xrVoidFunction> functions = {
            { "xrGetTensorListLatestGenerationNV", reinterpret_cast<PFN_xrVoidFunction>(&latest_generation) },
            { "xrCreateTensorListNV", reinterpret_cast<PFN_xrVoidFunction>(&create_list) },
            { "xrUpdateTensorListNV", reinterpret_cast<PFN_xrVoidFunction>(&update_list) },
            { "xrGetTensorListPropertiesNV", reinterpret_cast<PFN_xrVoidFunction>(&list_properties) },
            { "xrGetTensorCollectionPropertiesNV", reinterpret_cast<PFN_xrVoidFunction>(&collection_properties) },
            { "xrGetTensorDataNV", reinterpret_cast<PFN_xrVoidFunction>(&get_data) },
            { "xrDestroyTensorListNV", reinterpret_cast<PFN_xrVoidFunction>(&destroy_list) },
            { "xrConvertTimespecTimeToTimeKHR", reinterpret_cast<PFN_xrVoidFunction>(&timespec_to_time) },
            { "xrConvertTimeToTimespecTimeKHR", reinterpret_cast<PFN_xrVoidFunction>(&time_to_timespec) },
        };
        const auto it = functions.find(name);
        *function = it != functions.end() ? it->second : nullptr;
        return it != functions.end() ? XR_SUCCESS : XR_ERROR_FUNCTION_UNSUPPORTED;
    }

    inline static FakeTensorRuntime* s_current = nullptr;

    std::vector<std::shared_ptr<Collection>> m_collections;
    std::map<uintptr_t, Listed> m_lists;
    uintptr_t m_next_list = 1;
    XrTensorCollectionIDNV m_next_collection_id = 1;
    uint64_t m_generation = 1;
};

} // namespace tensor_test_support
