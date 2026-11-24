// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

namespace isaaccapture::detail
{

// POSIX paths must have no symlink components or replaceable ancestors; the
// target must belong to the effective user and forbid other writers. Windows
// uses filesystem directory creation without POSIX ownership checks.
bool ensure_secure_directory(const std::filesystem::path& directory);

} // namespace isaaccapture::detail
