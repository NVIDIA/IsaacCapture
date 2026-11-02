// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <openxr/openxr.h>

#include <atomic>
#include <memory>

namespace core
{

// Wrapper for OpenXR session handles
// This struct is used to pass OpenXR session information between modules
// without requiring them to link against the OpenXR loader library
struct OpenXRSessionHandles
{
    XrInstance instance{};
    XrSession session{};
    XrSpace space{};
    PFN_xrGetInstanceProcAddr xrGetInstanceProcAddr{};
    // Owned sessions share lifecycle state; externally supplied handles retain their
    // caller-managed lifecycle when this optional field is absent.
    std::shared_ptr<const std::atomic_bool> session_running{};

    bool is_running() const
    {
        return !session_running || session_running->load();
    }
};

} // namespace core
