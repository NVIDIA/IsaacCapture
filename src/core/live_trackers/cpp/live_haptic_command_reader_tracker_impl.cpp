// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "live_haptic_command_reader_tracker_impl.hpp"

#include <flatbuffers/flatbuffers.h>
#include <schema/serialized.hpp>

#include <memory>
#include <string>
#include <string_view>

namespace core
{

namespace
{

// Canonical identifiers for the vendor-neutral HapticCommand payload. The
// producer (TensorPushTracker created by PushTensorHapticDevice) pushes under
// the same "haptic_command" tensor identifier; SchemaTrackerBase filters by
// collection_id, these strings just keep the runtime's per-tensor diagnostics
// aligned.
constexpr const char* kHapticCommandTensorIdentifier = "haptic_command";
constexpr const char* kHapticCommandLocalizedName = "HapticCommand";

SchemaTrackerConfig make_haptic_command_reader_config(const HapticCommandReaderTracker* tracker)
{
    SchemaTrackerConfig cfg;
    cfg.collection_id = tracker->collection_id();
    cfg.max_flatbuffer_size = tracker->max_payload_size();
    cfg.tensor_identifier = kHapticCommandTensorIdentifier;
    cfg.localized_name = kHapticCommandLocalizedName;
    return cfg;
}

} // namespace

LiveHapticCommandReaderTrackerImpl::LiveHapticCommandReaderTrackerImpl(const OpenXRSessionHandles& handles,
                                                                       const HapticCommandReaderTracker* tracker)
    : schema_reader_(handles, make_haptic_command_reader_config(tracker), /*mcap_channels=*/nullptr)
{
}

void LiveHapticCommandReaderTrackerImpl::update(int64_t /*monotonic_time_ns*/)
{
    samples_.clear();
    const bool present = schema_reader_.read_all_samples(samples_);
    if (samples_.empty())
    {
        // Producer collection gone: drop stale commands so a disconnected device
        // stops rendering. A present-but-sample-less tick holds the last values.
        if (!present)
        {
            tracked_by_endpoint_.clear();
            latest_endpoint_.clear();
        }
        return;
    }

    // Latest-wins per endpoint across every sample drained this tick. The single
    // collection interleaves all endpoints, so bucket by HapticCommand.endpoint
    // instead of collapsing to one latest sample (which would drop every endpoint
    // but the last one pushed each frame).
    for (auto& sample : samples_)
    {
        const auto* fb = detail::verified_flatbuffer_root<HapticCommand>(sample.buffer);
        if (fb == nullptr)
        {
            continue;
        }
        std::string endpoint = fb->endpoint() != nullptr ? fb->endpoint()->str() : std::string{};
        // The wire already carries HapticCommand, so adopt the sample's bytes instead of
        // unpacking and re-encoding them. `fb` dangles past this point; read the endpoint
        // out first.
        tracked_by_endpoint_[endpoint] = Serialized<HapticCommand>::adopt(std::move(sample.buffer));
        latest_endpoint_ = std::move(endpoint);
    }
}

const Serialized<HapticCommand>& LiveHapticCommandReaderTrackerImpl::get_data() const
{
    // Backward-compatible latest-across-all-endpoints view: the endpoint of the
    // most recently drained sample.
    return get_data(latest_endpoint_);
}

const Serialized<HapticCommand>& LiveHapticCommandReaderTrackerImpl::get_data(std::string_view endpoint) const
{
    static const Serialized<HapticCommand> kEmpty{};
    const auto it = tracked_by_endpoint_.find(endpoint);
    return it != tracked_by_endpoint_.end() ? it->second : kEmpty;
}

} // namespace core
