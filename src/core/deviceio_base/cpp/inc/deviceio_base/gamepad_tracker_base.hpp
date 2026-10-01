// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tracker.hpp"

#include <schema/serialized.hpp>

namespace core
{

struct GamepadOutput;

// Abstract base interface for GamepadTracker implementations.
class IGamepadTrackerImpl : public ITrackerImpl
{
public:
    // Empty while no gamepad is connected (or, in replay, when the recording has no sample).
    virtual const Serialized<GamepadOutput>& get_data() const = 0;
};

} // namespace core
