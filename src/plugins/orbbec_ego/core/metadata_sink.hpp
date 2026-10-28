// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "inc/ego_camera/ego_camera.hpp"

namespace plugins::ego
{
std::unique_ptr<IMetadataSink> make_schema_metadata_sink(const std::vector<StreamConfig>& streams,
                                                         const std::string& prefix,
                                                         const std::string& plugin_root_id = {});
std::unique_ptr<IMetadataSink> make_mcap_metadata_sink(const std::vector<StreamConfig>& streams,
                                                       const std::string& filename,
                                                       bool include_media,
                                                       bool include_structured = true,
                                                       bool record_device_diagnostics = true);
} // namespace plugins::ego
