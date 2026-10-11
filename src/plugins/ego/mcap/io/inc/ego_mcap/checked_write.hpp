// SPDX-FileCopyrightText: Copyright (c) 2026 ORBBEC INC. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <mcap/tracker_channels.hpp>

#include <stdexcept>

namespace ego_mcap
{
template <typename RecordT>
void write_checked(mcap::McapWriter& writer,
                   core::McapTrackerChannels<RecordT>& channels,
                   size_t index,
                   const core::Serialized<RecordT>& record)
{
    const auto count = writer.statistics().messageCount;
    channels.write(index, record);
    // The official helper logs status errors; capture must stop rather than lose a record.
    if (writer.statistics().messageCount != count + 1)
        throw std::runtime_error("Ego MCAP record was not accepted by the writer");
}
} // namespace ego_mcap
