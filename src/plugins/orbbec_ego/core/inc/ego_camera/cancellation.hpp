// SPDX-FileCopyrightText: Copyright (c) 2026 ORBBEC INC. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ego_camera.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <mutex>

namespace plugins::ego
{
inline void check_capture_cancelled(const std::function<bool()>& cancelled)
{
    if (cancelled && cancelled())
        throw CaptureCancelled();
}

template <typename Predicate, typename Rep, typename Period>
bool wait_capture_ready(std::condition_variable& wake,
                        std::unique_lock<std::mutex>& lock,
                        const std::chrono::duration<Rep, Period>& timeout,
                        Predicate ready,
                        const std::function<bool()>& cancelled)
{
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!ready())
    {
        check_capture_cancelled(cancelled);
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline)
            return false;
        wake.wait_until(lock, std::min(deadline, now + std::chrono::milliseconds(20)));
    }
    check_capture_cancelled(cancelled);
    return true;
}
} // namespace plugins::ego
