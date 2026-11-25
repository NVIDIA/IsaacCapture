/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ARGUS_SENDER_CAPTURE_BACKEND_H
#define ARGUS_SENDER_CAPTURE_BACKEND_H

#include <string>

namespace argus_sender
{

enum CaptureBackendKind
{
    CAPTURE_BACKEND_AUTO,
    CAPTURE_BACKEND_ARGUS,
    CAPTURE_BACKEND_SIPL
};

const char* captureBackendName(CaptureBackendKind kind);

// "auto", "argus" or "sipl". Returns false for anything else.
bool parseCaptureBackend(const std::string& text, CaptureBackendKind* kind);

// Which backends this binary was built with.
struct BackendAvailability
{
    bool argus = false;
    bool sipl = false;
};

// An explicit request must be built in or fails with its own reason. Auto picks the only built-in backend; with
// several it fails and names the override instead of guessing.
bool selectCaptureBackend(CaptureBackendKind requested,
                          const BackendAvailability& available,
                          CaptureBackendKind* selected,
                          std::string* error);

} // namespace argus_sender

#endif // ARGUS_SENDER_CAPTURE_BACKEND_H
