// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace plugins
{
namespace avatar
{

inline constexpr auto kAvatarDataTimeout = std::chrono::seconds(3);

class DatasetFreshness
{
public:
    using Clock = std::chrono::steady_clock;

    void observe(int64_t seconds, int64_t nanoseconds, Clock::time_point now)
    {
        const auto stamp = std::make_pair(seconds, nanoseconds);
        if (!m_last_stamp || stamp != *m_last_stamp)
        {
            m_last_stamp = stamp;
            m_last_fresh_data = now;
        }
    }

    bool fresh(Clock::time_point now) const noexcept
    {
        return m_last_fresh_data && now - *m_last_fresh_data < kAvatarDataTimeout;
    }

private:
    std::optional<std::pair<int64_t, int64_t>> m_last_stamp;
    std::optional<Clock::time_point> m_last_fresh_data;
};

class StreamOutputRecovery
{
public:
    using Clock = std::chrono::steady_clock;

    template <typename Output, typename Factory>
    Output* update(std::unique_ptr<Output>& output,
                   bool usable,
                   Clock::time_point now,
                   Clock::duration retry_interval,
                   Factory&& create)
    {
        if (!usable)
        {
            // Consumers retain cached samples until the output resource is withdrawn.
            output.reset();
            return nullptr;
        }
        if (!output && (!m_retry_after || now >= *m_retry_after))
        {
            try
            {
                output = std::forward<Factory>(create)();
                m_retry_after.reset();
                m_error.reset();
            }
            catch (const std::exception& error)
            {
                m_error = std::string("output setup failed: ") + error.what();
                m_retry_after = now + retry_interval;
            }
            catch (...)
            {
                m_error = "output setup failed with an unknown error";
                m_retry_after = now + retry_interval;
            }
        }
        return output.get();
    }

    const std::optional<std::string>& error() const noexcept
    {
        return m_error;
    }

private:
    std::optional<Clock::time_point> m_retry_after;
    std::optional<std::string> m_error;
};

class GloveWaitReminder
{
public:
    using Clock = std::chrono::steady_clock;

    bool observe(bool ready, Clock::time_point now, Clock::duration interval)
    {
        if (ready)
        {
            m_last_log.reset();
            return false;
        }
        if (!m_last_log || now - *m_last_log >= interval)
        {
            m_last_log = now;
            return true;
        }
        return false;
    }

private:
    std::optional<Clock::time_point> m_last_log;
};

} // namespace avatar
} // namespace plugins
