// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <deviceio_base/keyboard_tracker_base.hpp>
#include <mcap/tracker_channels.hpp>
#include <schema/keyboard_generated.h>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace core
{

class KeyboardInputState;

using KeyboardMcapChannels = McapTrackerChannels<KeyboardOutputRecord>;

// In-process keyboard: drains the provider state shared with KeyboardTracker once per frame.
// Unlike the OpenXR-backed impls it uses no extensions and never touches the session handles.
class LiveKeyboardTrackerImpl : public IKeyboardTrackerImpl
{
public:
    static std::vector<std::string> required_extensions()
    {
        return {};
    }
    static std::unique_ptr<KeyboardMcapChannels> create_mcap_channels(mcap::McapWriter& writer,
                                                                      std::string_view base_name);

    LiveKeyboardTrackerImpl(std::shared_ptr<KeyboardInputState> state,
                            std::unique_ptr<KeyboardMcapChannels> mcap_channels);

    LiveKeyboardTrackerImpl(const LiveKeyboardTrackerImpl&) = delete;
    LiveKeyboardTrackerImpl& operator=(const LiveKeyboardTrackerImpl&) = delete;
    LiveKeyboardTrackerImpl(LiveKeyboardTrackerImpl&&) = delete;
    LiveKeyboardTrackerImpl& operator=(LiveKeyboardTrackerImpl&&) = delete;

    void update(int64_t monotonic_time_ns) override;
    const Serialized<KeyboardOutput>& get_data() const override;

private:
    std::shared_ptr<KeyboardInputState> state_;
    // The snapshot published each frame, encoded from a local in update().
    Serialized<KeyboardOutput> tracked_;
    std::unique_ptr<KeyboardMcapChannels> mcap_channels_;
};

} // namespace core
