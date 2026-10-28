// SPDX-FileCopyrightText: Copyright (c) 2026 ORBBEC INC. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

namespace plugins::ego
{
// Inputs remain available after success or failure; existing output files are rejected.
void merge_recording(const std::filesystem::path& recording,
                     const std::filesystem::path& media,
                     const std::filesystem::path& output);
}
