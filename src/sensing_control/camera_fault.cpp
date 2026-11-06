// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/sensing_control/camera_fault.hpp"

#include <array>
#include <string>

namespace plugins
{
namespace sensing
{

namespace
{

struct Signature
{
    /// A fragment of the capture stack's error text.
    std::string_view needle;
    std::string_view code;
    std::string_view message;
};

// Most specific first. Each entry is a fragment of the text the capture stack was seen to produce for that
// situation; extend the table, not the logic, when a new one is observed.
constexpr std::array<Signature, 4> kSignatures = { {
    { "link lock", "camera_link_not_locked",
      "A camera's GMSL link did not lock. Check that each camera is powered (its power cable is connected) and that "
      "its GMSL cable is seated at both ends." },
    { "Failed to detect", "camera_not_detected",
      "A camera was not detected. Check that each camera is powered (its power cable is connected) and that its GMSL "
      "cable is seated at both ends." },
    { "SIPLStatus 10", "camera_start_failed",
      "The camera system could not start. The usual cause is a camera with no power or no GMSL link: check each "
      "camera's power cable and GMSL cable." },
    { "INvSIPLCamera::Init failed", "camera_start_failed",
      "The camera system could not start. The usual cause is a camera with no power or no GMSL link: check each "
      "camera's power cable and GMSL cable." },
} };

} // namespace

Fault diagnose_camera_failure(std::string_view what)
{
    for (const auto& signature : kSignatures)
    {
        if (what.find(signature.needle) != std::string_view::npos)
        {
            return { std::string(signature.code),
                     std::string(signature.message) + " The producer retries automatically. (" + std::string(what) + ")" };
        }
    }
    return { "camera_error",
             "The cameras could not be started. The producer retries automatically. (" + std::string(what) + ")" };
}

} // namespace sensing
} // namespace plugins
