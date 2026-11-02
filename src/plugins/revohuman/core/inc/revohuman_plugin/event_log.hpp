// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <string>
#include <utility>
#include <variant>

namespace plugins
{
namespace revohuman
{

//! One JSON object per line, flushed per event, so a run can be audited offline
//! (pushed timestamps, injector transitions, shape hashes). Disabled when the path is empty.
class EventLog
{
public:
    using Value = std::variant<std::int64_t, std::uint64_t, double, bool, std::string>;
    using Field = std::pair<const char*, Value>;

    explicit EventLog(const std::string& path);

    bool enabled() const
    {
        return out_.is_open();
    }
    void write(const char* event, std::initializer_list<Field> fields);

    //! JSON string literal with escapes, quotes included.
    static std::string quote(const std::string& s);

private:
    std::ofstream out_;
};

} // namespace revohuman
} // namespace plugins
