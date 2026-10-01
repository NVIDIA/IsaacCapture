// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/deviceio_trackers/gamepad_tracker.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace core
{

std::optional<std::string> discover_gamepad_device()
{
    std::error_code ec;
    std::vector<std::string> candidates;
    for (const auto& entry : std::filesystem::directory_iterator("/dev/input/by-path", ec))
    {
        const std::string name = entry.path().filename().string();
        // "*-event-joystick" is the evdev node (/dev/input/eventN) of the same device; the
        // joystick-API reader needs the "*-joystick" one (/dev/input/jsN).
        if (name.ends_with("-joystick") && !name.ends_with("-event-joystick"))
            candidates.push_back(entry.path().string());
    }
    if (!candidates.empty())
    {
        std::sort(candidates.begin(), candidates.end());
        return candidates.front();
    }

    // Bluetooth pads (BlueZ uhid devices) get no by-path link: fall back to the lowest
    // /dev/input/jsN.
    std::optional<std::pair<int, std::string>> lowest;
    for (const auto& entry : std::filesystem::directory_iterator("/dev/input", ec))
    {
        const std::string name = entry.path().filename().string();
        if (name.size() <= 2 || !name.starts_with("js") ||
            !std::all_of(name.begin() + 2, name.end(), [](unsigned char c) { return std::isdigit(c) != 0; }))
            continue;
        const int number = std::stoi(name.substr(2));
        if (!lowest || number < lowest->first)
            lowest.emplace(number, entry.path().string());
    }
    if (!lowest)
        return std::nullopt;
    return lowest->second;
}

GamepadTracker::GamepadTracker(std::string device_path) : device_path_(std::move(device_path))
{
}

const Serialized<GamepadOutput>& GamepadTracker::get_data(const ITrackerSession& session) const
{
    return static_cast<const IGamepadTrackerImpl&>(session.get_tracker_impl(*this)).get_data();
}

} // namespace core
