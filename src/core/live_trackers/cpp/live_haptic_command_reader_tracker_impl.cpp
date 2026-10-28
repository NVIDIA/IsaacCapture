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
    auto batch = schema_reader_.read_all_verified_samples();
    if (!batch.collection_present || batch.rejected_sample_count != 0)
    {
        // Haptic commands drive actuators. A malformed sample cannot identify which
        // endpoint it targeted, so reject the whole batch and stop every stale command.
        tracked_by_endpoint_.clear();
        latest_endpoint_.clear();
        return;
    }

    if (batch.samples.empty())
    {
        // A present-but-sample-less tick holds the last command; device plugins own
        // their command timeout. Collection loss above clears the reader immediately.
        return;
    }

    // Latest-wins per endpoint across every sample drained this tick. The single
    // collection interleaves all endpoints, so bucket by HapticCommand.endpoint
    // instead of collapsing to one latest sample (which would drop every endpoint
    // but the last one pushed each frame).
    for (auto& verified : batch.samples)
    {
        std::string endpoint =
            verified.payload->endpoint() != nullptr ? verified.payload->endpoint()->str() : std::string{};
        // The verified handle already owns the wire bytes; move it into endpoint state
        // without unpacking, re-encoding, or exposing unverified storage.
        tracked_by_endpoint_[endpoint] = std::move(verified.payload);
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
