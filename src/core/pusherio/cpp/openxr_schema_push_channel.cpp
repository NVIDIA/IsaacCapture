// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/pusherio/openxr_schema_push_channel.hpp"

#include <log_bridge/logger.hpp>
#include <oxr_utils/oxr_funcs.hpp>
#include <oxr_utils/oxr_time.hpp>

#include <XR_NVX1_push_tensor.h>
#include <XR_NVX1_tensor_data.h>
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace core
{

namespace
{

// DLPack dtype code for uint8: code=1 (unsigned int), bits=8
// Formula: (code << 8) | bits
constexpr uint32_t DLPACK_DTYPE_UINT8 = (1 << 8) | 8;

} // namespace

class OpenXRSchemaPushChannel::Impl
{
public:
    Impl(const OpenXRSessionHandles& handles, SchemaPusherConfig config)
        : config_(std::move(config)), time_converter_(handles)
    {
        assert(handles.instance != XR_NULL_HANDLE && "OpenXR instance handle cannot be null");
        assert(handles.session != XR_NULL_HANDLE && "OpenXR session handle cannot be null");
        assert(handles.xrGetInstanceProcAddr && "xrGetInstanceProcAddr cannot be null");

        initialize_push_tensor_functions(handles);
        create_tensor_collection(handles);

        logger_->info("SchemaPusher initialized for collection: {}", config_.collection_id);
    }

    ~Impl()
    {
        // push_tensor_ is guaranteed non-null after construction succeeds.
        assert(push_tensor_ != XR_NULL_HANDLE && destroy_fn_ != nullptr);

        const XrResult result = destroy_fn_(push_tensor_);
        if (result != XR_SUCCESS)
        {
            logger_->warn("Failed to destroy push tensor collection, result={}", static_cast<int>(result));
        }
    }

    const SchemaPusherConfig& config() const
    {
        return config_;
    }

    void push_buffer(const uint8_t* buffer,
                     size_t size,
                     int64_t sample_time_local_common_clock_ns,
                     int64_t sample_time_raw_device_clock_ns)
    {
        std::vector<uint8_t> padded_buffer(config_.max_flatbuffer_size, 0);
        if (size != 0)
        {
            std::memcpy(padded_buffer.data(), buffer, size);
        }

        const XrTime xr_time = time_converter_.convert_monotonic_ns_to_xrtime(sample_time_local_common_clock_ns);

        XrPushTensorCollectionDataNV tensor_data{};
        tensor_data.type = XR_TYPE_PUSH_TENSOR_COLLECTION_DATA_NV;
        tensor_data.next = nullptr;
        tensor_data.timestamp = xr_time;
        tensor_data.rawDeviceTimestamp = static_cast<uint64_t>(sample_time_raw_device_clock_ns);
        tensor_data.buffer = padded_buffer.data();
        tensor_data.bufferSize = static_cast<uint32_t>(config_.max_flatbuffer_size);

        const XrResult result = push_fn_(push_tensor_, &tensor_data);
        if (result != XR_SUCCESS)
        {
            throw std::runtime_error("Failed to push tensor data, result=" + std::to_string(result));
        }
    }

private:
    void initialize_push_tensor_functions(const OpenXRSessionHandles& handles)
    {
        loadExtensionFunction(handles.instance, handles.xrGetInstanceProcAddr, "xrCreatePushTensorCollectionNV",
                              reinterpret_cast<PFN_xrVoidFunction*>(&create_fn_));
        loadExtensionFunction(handles.instance, handles.xrGetInstanceProcAddr, "xrPushTensorCollectionDataNV",
                              reinterpret_cast<PFN_xrVoidFunction*>(&push_fn_));
        loadExtensionFunction(handles.instance, handles.xrGetInstanceProcAddr, "xrDestroyPushTensorCollectionNV",
                              reinterpret_cast<PFN_xrVoidFunction*>(&destroy_fn_));
    }

    void create_tensor_collection(const OpenXRSessionHandles& handles)
    {
        XrPushTensorDlpackCreateInfoNV dlpack_info{};
        dlpack_info.type = XR_TYPE_PUSH_TENSOR_DLPACK_CREATE_INFO_NV;
        dlpack_info.next = nullptr;
        dlpack_info.data.versionMajor = 1;
        dlpack_info.data.versionMinor = 0;
        dlpack_info.data.dtype = DLPACK_DTYPE_UINT8;
        dlpack_info.data.ndim = 1;
        dlpack_info.data.shape[0] = static_cast<int64_t>(config_.max_flatbuffer_size);
        dlpack_info.data.strides[0] = sizeof(uint8_t);
        dlpack_info.data.byte_offset = 0;

        XrPushTensorCreateInfoNV tensor_info{};
        tensor_info.type = XR_TYPE_PUSH_TENSOR_CREATE_INFO_NV;
        tensor_info.next = &dlpack_info;
        tensor_info.properties.dataType = XR_TENSOR_DATA_TYPE_DLPACK_NV;
        tensor_info.properties.dataTypeSize = config_.max_flatbuffer_size;
        tensor_info.properties.offset = 0;
        std::strncpy(
            tensor_info.properties.identifier, config_.tensor_identifier.c_str(), XR_MAX_TENSOR_IDENTIFIER_SIZE - 1);
        tensor_info.properties.identifier[XR_MAX_TENSOR_IDENTIFIER_SIZE - 1] = '\0';

        XrPushTensorCollectionCreateInfoNV create_info{};
        create_info.type = XR_TYPE_PUSH_TENSOR_COLLECTION_CREATE_INFO_NV;
        create_info.next = nullptr;
        create_info.tensors = &tensor_info;
        create_info.data.tensorCount = 1;
        create_info.data.totalSampleSize = config_.max_flatbuffer_size;
        std::strncpy(create_info.data.identifier, config_.collection_id.c_str(), XR_MAX_TENSOR_IDENTIFIER_SIZE - 1);
        create_info.data.identifier[XR_MAX_TENSOR_IDENTIFIER_SIZE - 1] = '\0';
        std::strncpy(
            create_info.data.localizedName, config_.localized_name.c_str(), XR_MAX_TENSOR_LOCALIZED_NAME_SIZE - 1);
        create_info.data.localizedName[XR_MAX_TENSOR_LOCALIZED_NAME_SIZE - 1] = '\0';
        std::memset(&create_info.data.uuid, 0, sizeof(create_info.data.uuid));

        XrPushTensorCollectionCreateResultNV create_result{};
        create_result.type = XR_TYPE_PUSH_TENSOR_COLLECTION_CREATE_RESULT_NV;
        create_result.next = nullptr;

        const XrResult result = create_fn_(handles.session, &create_info, &create_result, &push_tensor_);
        if (result != XR_SUCCESS)
        {
            throw std::runtime_error("Failed to create push tensor collection, result=" + std::to_string(result));
        }
    }

    SchemaPusherConfig config_;
    XrTimeConverter time_converter_;
    std::shared_ptr<spdlog::logger> logger_ = isaaccapture::Logger::get("isaaccapture.core.SchemaPusher");
    XrPushTensorCollectionNV push_tensor_{ XR_NULL_HANDLE };
    PFN_xrCreatePushTensorCollectionNV create_fn_{ nullptr };
    PFN_xrPushTensorCollectionDataNV push_fn_{ nullptr };
    PFN_xrDestroyPushTensorCollectionNV destroy_fn_{ nullptr };
};

OpenXRSchemaPushChannel::OpenXRSchemaPushChannel(const OpenXRSessionHandles& handles, SchemaPusherConfig config)
    : impl_(std::make_unique<Impl>(handles, std::move(config)))
{
}

OpenXRSchemaPushChannel::~OpenXRSchemaPushChannel() = default;

std::vector<std::string> OpenXRSchemaPushChannel::get_required_extensions()
{
    std::vector<std::string> required_extensions = { "XR_NVX1_push_tensor", "XR_NVX1_tensor_data" };
    for (const auto& ext : XrTimeConverter::get_required_extensions())
    {
        required_extensions.push_back(ext);
    }
    return required_extensions;
}

const SchemaPusherConfig& OpenXRSchemaPushChannel::config() const
{
    return impl_->config();
}

void OpenXRSchemaPushChannel::push_buffer(const uint8_t* buffer,
                                          size_t size,
                                          int64_t sample_time_local_common_clock_ns,
                                          int64_t sample_time_raw_device_clock_ns)
{
    impl_->push_buffer(buffer, size, sample_time_local_common_clock_ns, sample_time_raw_device_clock_ns);
}

std::unique_ptr<ISchemaPushChannel> make_openxr_schema_push_channel(const OpenXRSessionHandles& handles,
                                                                    SchemaPusherConfig config)
{
    return std::make_unique<OpenXRSchemaPushChannel>(handles, std::move(config));
}

} // namespace core
