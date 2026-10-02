// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/replay_trackers/replay_deviceio_factory.hpp"

#include "generated_replay_includes.inc"
#include "replay_controller_tracker_impl.hpp"
#include "replay_full_body_tracker_impl.hpp"
#include "replay_hand_tracker_impl.hpp"
#include "replay_haptic_command_reader_tracker_impl.hpp"
#include "replay_head_tracker_impl.hpp"
#include "replay_keyboard_tracker_impl.hpp"
#include "replay_message_channel_tracker_impl.hpp"
#include "replay_tensor_push_tracker_impl.hpp"

#include <deviceio_trackers/controller_tracker.hpp>
#include <deviceio_trackers/full_body_tracker.hpp>
#include <deviceio_trackers/hand_tracker.hpp>
#include <deviceio_trackers/haptic_command_reader_tracker.hpp>
#include <deviceio_trackers/head_tracker.hpp>
#include <deviceio_trackers/keyboard_tracker.hpp>
#include <deviceio_trackers/message_channel_tracker.hpp>
#include <deviceio_trackers/tensor_push_tracker.hpp>
#include <mcap/reader.hpp>

#include <cassert>
#include <memory>
#include <stdexcept>
#include <string>

namespace core
{

namespace
{

std::unique_ptr<mcap::McapReader> open_reader(const std::string& filename)
{
    auto reader = std::make_unique<mcap::McapReader>();
    auto status = reader->open(filename);
    if (!status.ok())
    {
        throw std::runtime_error("ReplayDeviceIOFactory: failed to open MCAP file '" + filename + "': " + status.message);
    }
    return reader;
}

/*!
 * @brief What the recording declares, read from a reader of its own.
 *
 * Its own, so that every reader a tracker impl is handed is in the same state -- none of them
 * carrying a parsed summary the others do not have.
 *
 * No filename is a session with nothing to replay, which declares nothing. A filename that
 * was given and cannot be opened is still an error.
 */
RecordedSchemas read_recorded_schemas(const std::string& filename)
{
    if (filename.empty())
    {
        return RecordedSchemas();
    }

    const std::unique_ptr<mcap::McapReader> reader = open_reader(filename);
    return RecordedSchemas(*reader);
}

std::unique_ptr<ITrackerImpl> try_create_head_impl(ReplayDeviceIOFactory& factory, const ITracker& tracker)
{
    auto* typed = dynamic_cast<const HeadTracker*>(&tracker);
    return typed ? factory.create_head_tracker_impl(typed) : nullptr;
}

std::unique_ptr<ITrackerImpl> try_create_hand_impl(ReplayDeviceIOFactory& factory, const ITracker& tracker)
{
    auto* typed = dynamic_cast<const HandTracker*>(&tracker);
    return typed ? factory.create_hand_tracker_impl(typed) : nullptr;
}

std::unique_ptr<ITrackerImpl> try_create_controller_impl(ReplayDeviceIOFactory& factory, const ITracker& tracker)
{
    auto* typed = dynamic_cast<const ControllerTracker*>(&tracker);
    return typed ? factory.create_controller_tracker_impl(typed) : nullptr;
}

std::unique_ptr<ITrackerImpl> try_create_full_body_impl(ReplayDeviceIOFactory& factory, const ITracker& tracker)
{
    auto* typed = dynamic_cast<const FullBodyTracker*>(&tracker);
    return typed ? factory.create_full_body_tracker_impl(typed) : nullptr;
}

std::unique_ptr<ITrackerImpl> try_create_tensor_push_impl(ReplayDeviceIOFactory& factory, const ITracker& tracker)
{
    auto* typed = dynamic_cast<const TensorPushTracker*>(&tracker);
    return typed ? factory.create_tensor_push_tracker_impl(typed) : nullptr;
}

std::unique_ptr<ITrackerImpl> try_create_message_channel_impl(ReplayDeviceIOFactory& factory, const ITracker& tracker)
{
    auto* typed = dynamic_cast<const MessageChannelTracker*>(&tracker);
    return typed ? factory.create_message_channel_tracker_impl(typed) : nullptr;
}

std::unique_ptr<ITrackerImpl> try_create_haptic_command_reader_impl(ReplayDeviceIOFactory& factory,
                                                                    const ITracker& tracker)
{
    auto* typed = dynamic_cast<const HapticCommandReaderTracker*>(&tracker);
    return typed ? factory.create_haptic_command_reader_tracker_impl(typed) : nullptr;
}

std::unique_ptr<ITrackerImpl> try_create_keyboard_impl(ReplayDeviceIOFactory& factory, const ITracker& tracker)
{
    auto* typed = dynamic_cast<const KeyboardTracker*>(&tracker);
    return typed ? factory.create_keyboard_tracker_impl(typed) : nullptr;
}

#include "generated_replay_try_create.inc"

using TryCreateFn = std::unique_ptr<ITrackerImpl> (*)(ReplayDeviceIOFactory&, const ITracker&);

inline const TryCreateFn k_tracker_dispatch[] = {
    &try_create_head_impl,
    &try_create_hand_impl,
    &try_create_controller_impl,
    &try_create_full_body_impl,
#include "generated_replay_dispatch_rows.inc"
    &try_create_haptic_command_reader_impl,
    &try_create_tensor_push_impl,
    &try_create_message_channel_impl,
    &try_create_keyboard_impl,
};

} // namespace

ReplayDeviceIOFactory::ReplayDeviceIOFactory(std::string filename,
                                             const std::vector<std::pair<const ITracker*, std::string>>& tracker_names)
    : filename_(std::move(filename)), recorded_schemas_(read_recorded_schemas(filename_))
{
    for (const auto& [tracker, name] : tracker_names)
    {
        auto [it, inserted] = name_map_.emplace(tracker, name);
        if (!inserted)
        {
            throw std::invalid_argument("ReplayDeviceIOFactory: duplicate tracker pointer for channel name '" + name +
                                        "' (already mapped as '" + it->second + "')");
        }
    }
}

std::unique_ptr<ITrackerImpl> ReplayDeviceIOFactory::create_tracker_impl(const ITracker& tracker)
{
    for (const auto& try_create : k_tracker_dispatch)
    {
        if (std::unique_ptr<ITrackerImpl> impl = try_create(*this, tracker))
        {
            return impl;
        }
    }
    throw std::invalid_argument("ReplayDeviceIOFactory::create_tracker_impl: unsupported tracker type '" +
                                std::string(tracker.get_name()) + "'");
}

std::string_view ReplayDeviceIOFactory::get_name(const ITracker* tracker) const
{
    auto it = name_map_.find(tracker);
    assert(it != name_map_.end() && "get_name called for tracker not in name_map_");
    return it->second;
}

std::unique_ptr<IHeadTrackerImpl> ReplayDeviceIOFactory::create_head_tracker_impl(const HeadTracker* tracker)
{
    return std::make_unique<ReplayHeadTrackerImpl>(open_reader(filename_), get_name(tracker), recorded_schemas_);
}

std::unique_ptr<IHandTrackerImpl> ReplayDeviceIOFactory::create_hand_tracker_impl(const HandTracker* tracker)
{
    return std::make_unique<ReplayHandTrackerImpl>(open_reader(filename_), get_name(tracker), recorded_schemas_);
}

std::unique_ptr<IControllerTrackerImpl> ReplayDeviceIOFactory::create_controller_tracker_impl(const ControllerTracker* tracker)
{
    return std::make_unique<ReplayControllerTrackerImpl>(open_reader(filename_), get_name(tracker), recorded_schemas_);
}

std::unique_ptr<IFullBodyTrackerImpl> ReplayDeviceIOFactory::create_full_body_tracker_impl(const FullBodyTracker* tracker)
{
    return std::make_unique<ReplayFullBodyTrackerImpl>(open_reader(filename_), get_name(tracker), recorded_schemas_);
}

std::unique_ptr<ITensorPushTrackerImpl> ReplayDeviceIOFactory::create_tensor_push_tracker_impl(
    const TensorPushTracker* /*tracker*/)
{
    return std::make_unique<ReplayTensorPushTrackerImpl>();
}

std::unique_ptr<IMessageChannelTrackerImpl> ReplayDeviceIOFactory::create_message_channel_tracker_impl(
    const MessageChannelTracker* tracker)
{
    return std::make_unique<ReplayMessageChannelTrackerImpl>(
        open_reader(filename_), get_name(tracker), recorded_schemas_);
}

std::unique_ptr<IHapticCommandReaderTrackerImpl> ReplayDeviceIOFactory::create_haptic_command_reader_tracker_impl(
    const HapticCommandReaderTracker* /*tracker*/)
{
    return std::make_unique<ReplayHapticCommandReaderTrackerImpl>();
}

std::unique_ptr<IKeyboardTrackerImpl> ReplayDeviceIOFactory::create_keyboard_tracker_impl(const KeyboardTracker* tracker)
{
    return std::make_unique<ReplayKeyboardTrackerImpl>(
        open_reader(filename_), get_name(tracker), recorded_schemas_, tracker->input_state());
}

#include "generated_replay_factory_methods.inc"

} // namespace core
