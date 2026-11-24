// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <string>

namespace isaaccapture
{
// POSIX paths must have no symlinks or replaceable ancestors; the target must
// belong to the effective user and forbid other writers. Windows uses the
// filesystem API without POSIX ownership checks. Reasons name the failed component.
bool ensure_secure_directory(const std::filesystem::path& directory, std::string* reason = nullptr);
// Apply the same policy without creating any missing path components.
bool is_secure_directory(const std::filesystem::path& directory, std::string* reason = nullptr);
} // namespace isaaccapture
