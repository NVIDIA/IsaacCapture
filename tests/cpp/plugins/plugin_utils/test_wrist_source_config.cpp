// SPDX-FileCopyrightText: Copyright (c) 2026 Agibot Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <plugin_utils/wrist_pose_source.hpp>

#include <cmath>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>

namespace
{

class Environment
{
public:
    explicit Environment(const char* name) : m_name(name)
    {
        if (const char* value = std::getenv(name))
        {
            m_previous = value;
        }
        unsetenv(name);
    }
    ~Environment()
    {
        if (m_previous)
        {
            setenv(m_name, m_previous->c_str(), 1);
        }
        else
        {
            unsetenv(m_name);
        }
    }
    void set(const char* value)
    {
        setenv(m_name, value, 1);
    }

private:
    const char* m_name;
    std::optional<std::string> m_previous;
};

constexpr const char* kVariable = "ISAAC_TEST_WRIST_CONFIG";

} // namespace

TEST_CASE("Wrist source mode accepts only named modes", "[plugin_utils][wrist_config]")
{
    Environment env(kVariable);
    using plugin_utils::WristSourceMode;
    CHECK(plugin_utils::wrist_source_mode_from_env(kVariable, WristSourceMode::Auto) == WristSourceMode::Auto);
    env.set("hand_tracking");
    CHECK(plugin_utils::wrist_source_mode_from_env(kVariable, WristSourceMode::Auto) == WristSourceMode::HandTracking);
    env.set("controller");
    CHECK(plugin_utils::wrist_source_mode_from_env(kVariable, WristSourceMode::Auto) == WristSourceMode::Controller);
    env.set("auto");
    CHECK(plugin_utils::wrist_source_mode_from_env(kVariable, WristSourceMode::HandTracking) == WristSourceMode::Auto);
    env.set("controler");
    CHECK_THROWS_AS(plugin_utils::wrist_source_mode_from_env(kVariable, WristSourceMode::Auto), std::runtime_error);
}

TEST_CASE("Required wrist offsets reject missing and malformed input", "[plugin_utils][wrist_config]")
{
    Environment env(kVariable);
    CHECK_THROWS_AS(plugin_utils::pose_from_env(kVariable), std::runtime_error);
    for (const char* value :
         { "", "0,0,0,0,0,0", "0,0,0,0,0,0,0", "nan,0,0,0,0,0,1", "0,0,0,0,0,inf,1", "0,0,0,0,0,0,1,junk" })
    {
        env.set(value);
        CHECK_THROWS_AS(plugin_utils::pose_from_env(kVariable), std::runtime_error);
    }
}

TEST_CASE("Wrist offsets normalize finite quaternions and preserve metres", "[plugin_utils][wrist_config]")
{
    Environment env(kVariable);
    for (const char* value : { "0.1,-0.2,0.3,0,0,0,2", "0.1,-0.2,0.3,0,0,0,3e38" })
    {
        env.set(value);
        const auto pose = plugin_utils::pose_from_env(kVariable);
        CHECK(std::abs(pose.position.x - 0.1F) < 1.0e-6F);
        CHECK(std::abs(pose.position.y + 0.2F) < 1.0e-6F);
        CHECK(std::abs(pose.position.z - 0.3F) < 1.0e-6F);
        CHECK(std::abs(pose.orientation.w - 1.0F) < 1.0e-6F);
    }
}

TEST_CASE("Optional wrist offsets fall back only when unset", "[plugin_utils][wrist_config]")
{
    Environment env(kVariable);
    const XrPosef fallback{ { 0, 0, 0, 1 }, { 1, 2, 3 } };
    CHECK(plugin_utils::pose_from_env(kVariable, fallback).position.y == 2);
    env.set("invalid");
    CHECK_THROWS_AS(plugin_utils::pose_from_env(kVariable, fallback), std::runtime_error);
}
