// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tracker.hpp"

#include <schema/serialized.hpp>

namespace core
{

struct SpaceMouseOutput;

// Abstract base interface for SpaceMouseTracker implementations.
class ISpaceMouseTrackerImpl : public ITrackerImpl
{
public:
    // Empty while no SpaceMouse is connected (or, in replay, when the recording has no sample).
    virtual const Serialized<SpaceMouseOutput>& get_data() const = 0;
};

} // namespace core
