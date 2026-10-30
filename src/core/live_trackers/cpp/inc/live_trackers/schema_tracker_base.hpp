// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <log_bridge/logger.hpp>
#include <oxr_utils/oxr_session_handles.hpp>
#include <oxr_utils/oxr_time.hpp>
#include <schema/timestamp_generated.h>

#include <XR_NVX1_tensor_data.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace core
{

namespace detail
{

inline bool is_valid_schema_sample_layout(uint32_t sample_size,
                                          uint32_t sample_stride,
                                          size_t maximum_sample_size,
                                          size_t maximum_sample_stride) noexcept
{
    return sample_size > 0 && sample_size <= maximum_sample_size && sample_stride >= sample_size &&
           sample_stride <= maximum_sample_stride;
}

} // namespace detail

/*!
 * @brief Configuration for OpenXR tensor-backed FlatBuffer readers (used by SchemaTracker).
 */
struct SchemaTrackerConfig
{
    static constexpr size_t DEFAULT_MAX_SAMPLE_BATCH_STRIDE = 1024 * 1024;

    //! Tensor collection identifier for discovery (e.g., "head_data").
    std::string collection_id;

    //! Maximum serialized FlatBuffer message size in bytes.
    size_t max_flatbuffer_size;

    //! Tensor name within the collection (e.g., "head_pose").
    std::string tensor_identifier;

    //! Human-readable description for debugging and runtime display.
    std::string localized_name;

    //! Maximum allocation for one runtime-padded sample retrieval.
    size_t max_sample_batch_stride = DEFAULT_MAX_SAMPLE_BATCH_STRIDE;
};

/*!
 * @brief Utility class for reading FlatBuffer schema data via OpenXR tensor extensions.
 *
 * This class handles all the OpenXR tensor extension calls for reading data.
 * Use it via composition in live ITrackerImpl implementations.
 *
 * The caller is responsible for creating the OpenXR session with the required extensions
 * (XR_NVX1_TENSOR_DATA_EXTENSION_NAME).
 *
 * See LiveGeneric3AxisPedalTrackerImpl for a concrete usage example.
 */
class SchemaTrackerBase
{
public:
    /*!
     * @brief Constructs the tracker and initializes the OpenXR tensor list.
     * @param handles OpenXR session handles.
     * @param config Configuration for the tensor collection.
     * @throws std::runtime_error if initialization fails.
     */
    SchemaTrackerBase(const OpenXRSessionHandles& handles, SchemaTrackerConfig config);

    ~SchemaTrackerBase();

    SchemaTrackerBase(const SchemaTrackerBase&) = delete;
    SchemaTrackerBase& operator=(const SchemaTrackerBase&) = delete;
    SchemaTrackerBase(SchemaTrackerBase&&) = delete;
    SchemaTrackerBase& operator=(SchemaTrackerBase&&) = delete;

    /*!
     * @brief Get required OpenXR extensions for tensor data reading and time conversion.
     * @return Vector of required extension name strings.
     */
    static std::vector<std::string> get_required_extensions();

    /*!
     * @brief A single tensor sample with its data buffer and timestamps.
     *
     * The DeviceDataTimestamp fields are populated as follows:
     *   - available_time_local_common_clock: system monotonic nanoseconds when the runtime
     *     received the sample (converted from XrTime via xrConvertTimeToTimespecTimeKHR).
     *   - sample_time_local_common_clock: system monotonic nanoseconds when the sample was
     *     captured on the push side (converted from XrTime symmetrically with push_buffer).
     *   - sample_time_raw_device_clock: raw device clock nanoseconds, unchanged from what
     *     the pusher provided.
     */
    struct SampleResult
    {
        std::vector<uint8_t> buffer;
        DeviceDataTimestamp timestamp;
    };

    /*!
     * @brief Access the configuration.
     */
    const SchemaTrackerConfig& config() const;

protected:
    // Raw tensor bytes stay behind the typed SchemaTracker boundary so every
    // hand-written reader must pass through schema verification before field access.
    bool read_all_samples(std::vector<SampleResult>& samples);
    void note_rejected_flatbuffer_samples(size_t count);

private:
    void initialize_tensor_data_functions();
    void create_tensor_list();
    bool ensure_collection();
    bool read_next_sample(SampleResult& out);

    OpenXRSessionHandles m_handles;
    SchemaTrackerConfig m_config;
    XrTimeConverter m_time_converter;
    std::shared_ptr<spdlog::logger> m_logger = isaaccapture::Logger::get("isaaccapture.core.SchemaTrackerBase");

    XrTensorListNV m_tensor_list{ XR_NULL_HANDLE };

    PFN_xrGetTensorListLatestGenerationNV m_get_latest_gen_fn{ nullptr };
    PFN_xrCreateTensorListNV m_create_list_fn{ nullptr };
    PFN_xrGetTensorListPropertiesNV m_get_list_props_fn{ nullptr };
    PFN_xrGetTensorCollectionPropertiesNV m_get_coll_props_fn{ nullptr };
    PFN_xrGetTensorDataNV m_get_data_fn{ nullptr };
    PFN_xrUpdateTensorListNV m_update_list_fn{ nullptr };
    PFN_xrDestroyTensorListNV m_destroy_list_fn{ nullptr };

    std::optional<uint32_t> m_target_collection_index;
    uint32_t m_sample_batch_stride{ 0 };
    uint32_t m_sample_size{ 0 };

    uint64_t m_cached_generation{ 0 };
    uint64_t m_rejected_flatbuffer_samples{ 0 };
    uint64_t m_next_rejection_warning{ 1 };

    std::optional<int64_t> m_last_sample_index;
};

} // namespace core
