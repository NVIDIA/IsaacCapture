// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/camera_plugin/options.hpp"

#include <charconv>
#include <stdexcept>
#include <system_error>

namespace plugins::camera
{

namespace
{

constexpr std::string_view CAMERA_ARGUMENT_FORM = "ID:BACKEND[:KEY=VALUE[,KEY=VALUE]...]";

}

OptionReader::OptionReader(const CameraConfig& config) : m_config(config)
{
}

std::uint32_t OptionReader::number(std::string_view key, std::uint32_t fallback, std::uint32_t minimum, std::uint32_t maximum)
{
    m_read.emplace(key);
    const auto found = m_config.options.find(std::string(key));
    if (found == m_config.options.end())
    {
        return fallback;
    }
    const std::string& value = found->second;
    std::uint32_t result = 0;
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
    if (error != std::errc() || end != value.data() + value.size() || result < minimum || result > maximum)
    {
        fail("option '" + std::string(key) + "' is '" + value + "'; it takes a whole number from " +
             std::to_string(minimum) + " to " + std::to_string(maximum));
    }
    return result;
}

std::string OptionReader::text(std::string_view key, std::string_view fallback)
{
    m_read.emplace(key);
    const auto found = m_config.options.find(std::string(key));
    return found == m_config.options.end() ? std::string(fallback) : found->second;
}

void OptionReader::finish() const
{
    for (const auto& [key, value] : m_config.options)
    {
        if (!m_read.contains(key))
        {
            fail("there is no option '" + key + "'");
        }
    }
}

void OptionReader::fail(const std::string& what) const
{
    throw std::invalid_argument("camera '" + m_config.id + "' (" + m_config.backend + " backend): " + what);
}

CameraConfig parse_camera_argument(std::string_view argument)
{
    const auto fail = [&](const std::string& what) -> void
    {
        throw std::invalid_argument("--camera=" + std::string(argument) + ": " + what + "; the form is " +
                                    std::string(CAMERA_ARGUMENT_FORM));
    };

    CameraConfig config;
    const std::size_t first = argument.find(':');
    if (first == std::string_view::npos)
    {
        fail("a camera needs an id and a backend");
    }
    const std::size_t second = argument.find(':', first + 1);
    config.id = argument.substr(0, first);
    config.backend = argument.substr(first + 1, second == std::string_view::npos ? second : second - first - 1);
    if (config.id.empty() || config.backend.empty())
    {
        fail("a camera needs an id and a backend");
    }

    std::string_view rest = second == std::string_view::npos ? std::string_view() : argument.substr(second + 1);
    while (!rest.empty())
    {
        const std::size_t comma = rest.find(',');
        const std::string_view pair = rest.substr(0, comma);
        const std::size_t equals = pair.find('=');
        if (equals == std::string_view::npos || equals == 0)
        {
            fail("'" + std::string(pair) + "' is not KEY=VALUE");
        }
        if (!config.options.emplace(pair.substr(0, equals), pair.substr(equals + 1)).second)
        {
            fail("option '" + std::string(pair.substr(0, equals)) + "' is given twice");
        }
        rest = comma == std::string_view::npos ? std::string_view() : rest.substr(comma + 1);
    }
    return config;
}

} // namespace plugins::camera
