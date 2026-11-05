// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/revohuman_plugin/types.hpp"

namespace plugins
{
namespace revohuman
{

const char* hand_name(Hand hand)
{
    return hand == Hand::Left ? "left" : "right";
}

} // namespace revohuman
} // namespace plugins
