// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

namespace isaaccapture::detail
{

// std::filesystem does not expose POSIX ownership, sticky bits, or no-follow
// traversal. Keep this policy in one testable helper instead of open-coding it
// at each sink setup site.
bool ensure_secure_directory(const std::filesystem::path& directory);

} // namespace isaaccapture::detail
