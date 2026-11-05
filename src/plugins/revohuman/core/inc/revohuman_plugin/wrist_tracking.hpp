// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>

namespace plugins
{
namespace revohuman
{

//! When the OpenXR hand injector of one hand must exist: from the first hand frame until no frame
//! was pushed for stall_ns. The caller owns the injector and does what each Step says, in order:
//! destroy, then create.
class InjectorLifecycle
{
public:
    struct Step
    {
        bool destroy = false;
        bool create = false;
        std::string reason; //!< Why the injector was destroyed or created.
    };

    explicit InjectorLifecycle(std::int64_t stall_ns);

    //! A hand frame is about to be pushed; now_ns is the current monotonic time.
    Step on_frame(std::int64_t now_ns);
    //! Periodic check: destroys the injector when nothing was pushed for stall_ns.
    Step on_tick(std::int64_t now_ns);

    bool active() const
    {
        return active_;
    }

private:
    std::int64_t stall_ns_;
    bool active_ = false;
    std::int64_t last_push_ns_ = 0;
};

} // namespace revohuman
} // namespace plugins
