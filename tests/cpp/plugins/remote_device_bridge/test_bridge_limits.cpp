// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <remote_plugin_session/bridge_limits.hpp>

using remote_plugin_session::kDefaultMaxSchemaBytes;
using remote_plugin_session::schema_cap_error;

TEST_CASE("schema_cap_error accepts a schema sample at the cap", "[remote_device_bridge][unit]")
{
    REQUIRE_FALSE(schema_cap_error("tactile", kDefaultMaxSchemaBytes, kDefaultMaxSchemaBytes).has_value());
    REQUIRE_FALSE(schema_cap_error("tactile", 4096, kDefaultMaxSchemaBytes).has_value());
}

TEST_CASE("schema_cap_error refuses a schema sample above the cap", "[remote_device_bridge][unit]")
{
    REQUIRE(schema_cap_error("tactile", kDefaultMaxSchemaBytes + 1, kDefaultMaxSchemaBytes) ==
            "schema 'tactile' sample size 32769 bytes exceeds bridge cap 32768 bytes (runtime limit); channel refused");
    REQUIRE(schema_cap_error("tactile", 40960, kDefaultMaxSchemaBytes).has_value());
    REQUIRE_FALSE(schema_cap_error("tactile", 40960, 65536).has_value()); // --max-schema-bytes raises the cap
}
