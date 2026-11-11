// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <iostream>
#include <string>

namespace camera_bridge
{

inline void log_info(const std::string& message)
{
    std::cerr << "[info] " << message << "\n";
}

inline void log_warn(const std::string& message)
{
    std::cerr << "[warn] " << message << "\n";
}

inline void log_error(const std::string& message)
{
    std::cerr << "[error] " << message << "\n";
}

} // namespace camera_bridge
