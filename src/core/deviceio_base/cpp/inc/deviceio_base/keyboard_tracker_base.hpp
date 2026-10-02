// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tracker.hpp"

#include <schema/serialized.hpp>

namespace core
{

struct KeyboardOutput;

// Abstract base interface for KeyboardTracker implementations.
class IKeyboardTrackerImpl : public ITrackerImpl
{
public:
    // Empty when no provider is attached (or, in replay, when the recording has no sample).
    virtual const Serialized<KeyboardOutput>& get_data() const = 0;
};

} // namespace core
