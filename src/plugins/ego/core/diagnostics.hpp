// SPDX-FileCopyrightText: Copyright (c) 2026 ORBBEC INC. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <log_bridge/logger.hpp>

#include <sstream>
#include <utility>

namespace plugins::ego::detail
{
class DiagnosticLine
{
public:
    explicit DiagnosticLine(spdlog::level::level_enum level) : level_(level)
    {
    }
    ~DiagnosticLine() noexcept
    {
        try
        {
            auto message = stream_.str();
            while (!message.empty() && (message.back() == '\n' || message.back() == '\r'))
                message.pop_back();
            isaaccapture::Logger::get("isaaccapture.plugins.ego.capture")->log(level_, "{}", message);
        }
        catch (...)
        {
        }
    }
    template <typename T>
    DiagnosticLine& operator<<(T&& value)
    {
        stream_ << std::forward<T>(value);
        return *this;
    }
    DiagnosticLine& operator<<(std::ostream& (*manipulator)(std::ostream&))
    {
        manipulator(stream_);
        return *this;
    }

private:
    spdlog::level::level_enum level_;
    std::ostringstream stream_;
};
inline DiagnosticLine log_info()
{
    return DiagnosticLine(spdlog::level::info);
}
inline DiagnosticLine log_error()
{
    return DiagnosticLine(spdlog::level::err);
}
inline DiagnosticLine log_warning()
{
    return DiagnosticLine(spdlog::level::warn);
}
} // namespace plugins::ego::detail
