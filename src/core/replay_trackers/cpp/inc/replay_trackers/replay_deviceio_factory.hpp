// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <mcap/recorded_schemas.hpp>

#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace core
{

class ITracker;
class ITrackerImpl;
class TrackerSampleStreamReader;
class TrackerSampleSource;
class ControllerTracker;
class IControllerTrackerImpl;
class FullBodyTracker;
class IFullBodyTrackerImpl;
class TensorPushTracker;
class ITensorPushTrackerImpl;
class HandTracker;
class IHandTrackerImpl;
class HeadTracker;
class IHeadTrackerImpl;
class KeyboardTracker;
class IKeyboardTrackerImpl;
class HapticCommandReaderTracker;
class IHapticCommandReaderTrackerImpl;
class MessageChannelTracker;
class IMessageChannelTrackerImpl;

// Same fragment the live factory header uses: forward decls are direction-agnostic.
#include "generated_tracker_forward_decls.inc"

/**
 * @brief Factory for replay (MCAP-backed) tracker implementations.
 *
 * Opens a fresh McapReader per tracker impl so each tracker has its own
 * FileReader buffer; crossing an MCAP chunk boundary in one tracker cannot
 * overwrite another tracker's pre-fetched message data pointer.
 *
 * What the recording was written under is read once, when the factory is built, because it
 * belongs to the file and not to any one reader. A recording with no summary section has to
 * be scanned end to end to produce it, and a session builds a tracker impl per entry in its
 * config. Reading it here also means a recording this build cannot decode is turned down
 * before a single tracker impl exists.
 *
 * An empty filename is a session with no recording -- every tracker in it is push-fed -- and
 * yields schemas that declare nothing rather than an error.
 *
 * Built on a sample stream instead, every impl reads the newest records the stream has
 * received, through the same replay impls.
 */
class ReplayDeviceIOFactory
{
public:
    ReplayDeviceIOFactory(std::string filename,
                          const std::vector<std::pair<const ITracker*, std::string>>& tracker_names);

    //! `stream` must outlive every impl this factory creates, each of which subscribes to its topics on it.
    //! Throws std::invalid_argument for a tracker with metadata (e.g. TactileTracker), whose impl reads it before
    //! any sample arrives.
    ReplayDeviceIOFactory(TrackerSampleStreamReader& stream,
                          const std::vector<std::pair<const ITracker*, std::string>>& tracker_names);

    /** Create tracker impl from a tracker instance using dynamic dispatch. */
    std::unique_ptr<ITrackerImpl> create_tracker_impl(const ITracker& tracker);

    std::unique_ptr<IHeadTrackerImpl> create_head_tracker_impl(const HeadTracker* tracker);
    std::unique_ptr<IHandTrackerImpl> create_hand_tracker_impl(const HandTracker* tracker);
    std::unique_ptr<IControllerTrackerImpl> create_controller_tracker_impl(const ControllerTracker* tracker);
    std::unique_ptr<IFullBodyTrackerImpl> create_full_body_tracker_impl(const FullBodyTracker* tracker);
    std::unique_ptr<ITensorPushTrackerImpl> create_tensor_push_tracker_impl(const TensorPushTracker* tracker);
    std::unique_ptr<IMessageChannelTrackerImpl> create_message_channel_tracker_impl(const MessageChannelTracker* tracker);
    std::unique_ptr<IHapticCommandReaderTrackerImpl> create_haptic_command_reader_tracker_impl(
        const HapticCommandReaderTracker* tracker);
    std::unique_ptr<IKeyboardTrackerImpl> create_keyboard_tracker_impl(const KeyboardTracker* tracker);
    // create_<name>_tracker_impl for every manifest tracker.
#include "generated_replay_factory_declarations.inc"

private:
    void map_names(const std::vector<std::pair<const ITracker*, std::string>>& tracker_names);
    std::string_view get_name(const ITracker* tracker) const;
    //! Where the next impl reads its records: a reader of its own, or the shared stream.
    TrackerSampleSource open_source() const;

    std::string filename_;
    RecordedSchemas recorded_schemas_;
    TrackerSampleStreamReader* stream_ = nullptr;
    std::unordered_map<const ITracker*, std::string> name_map_;
};

} // namespace core
