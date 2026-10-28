// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "inc/ego_camera/ego_camera.hpp"

#include <memory>

namespace plugins::ego
{

class Preview
{
public:
    Preview();
    ~Preview();
    Preview(const Preview&) = delete;
    Preview& operator=(const Preview&) = delete;

    void submit(const CapturedFrame& frame) noexcept;
    bool closed() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace plugins::ego
