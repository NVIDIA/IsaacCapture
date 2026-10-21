// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <string>

namespace remote_plugin_session
{

//! Largest schema sample the runtime takes: 32 KiB pushed fine up to 720 Hz; >= 40 KiB fails with -2 after a few
//! pushes and the runtime then drops the client's IPC.
constexpr size_t kDefaultMaxSchemaBytes = 32 * 1024;

//! Empty if a schema channel with this sample size fits the bridge cap, else the refusal text for the client.
inline std::string schema_cap_error(const std::string& collection_id, size_t sample_bytes, size_t cap_bytes)
{
    if (sample_bytes <= cap_bytes)
    {
        return {};
    }
    return "schema '" + collection_id + "' sample size " + std::to_string(sample_bytes) + " bytes exceeds bridge cap " +
           std::to_string(cap_bytes) + " bytes (runtime limit); channel refused";
}

} // namespace remote_plugin_session
