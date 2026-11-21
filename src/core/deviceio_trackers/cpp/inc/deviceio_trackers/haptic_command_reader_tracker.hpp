// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <deviceio_base/haptic_command_reader_tracker_base.hpp>
#include <schema/haptic_command_generated.h>

#include <chrono>
#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

namespace core
{

// Consumer ITracker (plugin side): reads the most-recent HapticCommand
// FlatBuffer per endpoint pushed by a TensorPushTracker on the same
// `collection_id` (the producer encodes one HapticCommand per endpoint and
// pushes them under the canonical "haptic_command" tensor identifier). Commands
// are bucketed by their `endpoint` field so multiple endpoints (e.g. left/right)
// sharing one collection are read independently. A vendor plugin reuses this
// directly instead of writing its own SchemaTracker boilerplate.
class HapticCommandReaderTracker : public ITracker
{
public:
    static constexpr std::size_t DEFAULT_MAX_PAYLOAD_SIZE = 256;

    // max_sample_age rejects commands that were already stale when the live
    // reader drained them. An unset limit preserves the latest-value behavior.
    explicit HapticCommandReaderTracker(const std::string& collection_id,
                                        std::size_t max_payload_size = DEFAULT_MAX_PAYLOAD_SIZE,
                                        std::optional<std::chrono::nanoseconds> max_sample_age = std::nullopt);

    std::string_view get_name() const override
    {
        return TRACKER_NAME;
    }

    // Latest command across all endpoints (backward compatible). Correct for a
    // single-endpoint device; a multi-endpoint device should use the endpoint
    // overload, as this returns whichever endpoint was pushed last.
    const Serialized<HapticCommand>& get_data(const ITrackerSession& session) const;

    // Latest command for `endpoint`; the handle is empty until a sample for that
    // endpoint arrives, or after the producer collection disappears.
    const Serialized<HapticCommand>& get_data(const ITrackerSession& session, std::string_view endpoint) const;

    const std::string& collection_id() const
    {
        return collection_id_;
    }

    std::size_t max_payload_size() const
    {
        return max_payload_size_;
    }

    std::optional<std::chrono::nanoseconds> max_sample_age() const
    {
        return max_sample_age_;
    }

private:
    static constexpr const char* TRACKER_NAME = "HapticCommandReaderTracker";

    std::string collection_id_;
    std::size_t max_payload_size_{ DEFAULT_MAX_PAYLOAD_SIZE };
    std::optional<std::chrono::nanoseconds> max_sample_age_;
};

} // namespace core
