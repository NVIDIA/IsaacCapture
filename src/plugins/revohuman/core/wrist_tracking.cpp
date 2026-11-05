// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/revohuman_plugin/wrist_tracking.hpp"

namespace plugins
{
namespace revohuman
{

InjectorLifecycle::InjectorLifecycle(std::int64_t stall_ns) : stall_ns_(stall_ns)
{
}

InjectorLifecycle::Step InjectorLifecycle::on_frame(std::int64_t now_ns)
{
    Step step;
    if (!active_)
    {
        active_ = true;
        step.create = true;
        step.reason = "first_frame";
    }
    last_push_ns_ = now_ns;
    return step;
}

InjectorLifecycle::Step InjectorLifecycle::on_tick(std::int64_t now_ns)
{
    Step step;
    if (active_ && now_ns - last_push_ns_ > stall_ns_)
    {
        active_ = false;
        step.destroy = true;
        step.reason = "stall";
    }
    return step;
}

} // namespace revohuman
} // namespace plugins
