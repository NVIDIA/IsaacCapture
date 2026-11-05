// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/revohuman_plugin/event_log.hpp"

#include <cmath>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <type_traits>

namespace plugins
{
namespace revohuman
{

EventLog::EventLog(const std::string& path)
{
    if (path.empty())
    {
        return;
    }
    out_.open(path, std::ios::out | std::ios::app);
    if (!out_)
    {
        throw std::runtime_error("event log: cannot open " + path);
    }
}

std::string EventLog::quote(const std::string& s)
{
    std::string out = "\"";
    for (const unsigned char c : s)
    {
        switch (c)
        {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if (c < 0x20)
            {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                out += buf;
            }
            else
            {
                out += static_cast<char>(c);
            }
        }
    }
    return out + "\"";
}

void EventLog::write(const char* event, std::initializer_list<Field> fields)
{
    if (!enabled())
    {
        return;
    }
    std::ostringstream line;
    line.precision(17);
    line << "{\"event\":" << quote(event);
    for (const auto& [key, value] : fields)
    {
        line << "," << quote(key) << ":";
        std::visit(
            [&line](const auto& v)
            {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<T, std::string>)
                {
                    line << quote(v);
                }
                else if constexpr (std::is_same_v<T, bool>)
                {
                    line << (v ? "true" : "false");
                }
                else if constexpr (std::is_same_v<T, double>)
                {
                    if (std::isfinite(v))
                    {
                        line << v;
                    }
                    else
                    {
                        line << "null";
                    }
                }
                else
                {
                    line << v;
                }
            },
            value);
    }
    line << "}\n";
    out_ << line.str();
    out_.flush();
}

} // namespace revohuman
} // namespace plugins
