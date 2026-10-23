// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <oxr_utils/oxr_session_handles.hpp>
#include <pusherio/hand_tracking_push_channel.hpp>

#include <memory>

namespace plugin_utils
{

std::unique_ptr<core::IHandTrackingPushChannel> make_openxr_hand_tracking_push_channel(
    const core::OpenXRSessionHandles& handles, XrHandEXT hand);

} // namespace plugin_utils
