// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Mock backend only, for tests: not an operator interface.

#pragma once

#include <string>

namespace plugins
{
namespace sensing
{

/// Every sensor constructed afterwards shows this GIF as it is -- no mirror, no eye offset -- instead of
/// its module's fixed picture. An empty path restores the fixed pictures.
void mock_set_test_picture(const std::string& gif_path);

} // namespace sensing
} // namespace plugins
