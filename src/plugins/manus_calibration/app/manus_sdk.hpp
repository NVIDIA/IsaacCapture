// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <manus_calibration/engine.hpp>

namespace plugins::manus::calibration
{

// GloveCalibrationSdk backed by the Manus Integrated SDK. The SDK delivers its landscape through
// C callbacks, so this class owns process-wide state and at most one instance may exist.
class ManusGloveSdk final : public GloveCalibrationSdk
{
public:
    ManusGloveSdk() = default;
    ~ManusGloveSdk() override;

    ManusGloveSdk(const ManusGloveSdk&) = delete;
    ManusGloveSdk& operator=(const ManusGloveSdk&) = delete;

    std::optional<std::string> version() override;
    SdkStatus initialize() override;
    ConnectResult connect() override;
    std::vector<SdkGlove> gloves() override;
    SdkStatus step_count(uint32_t glove_id, uint32_t& count) override;
    SdkStatus step_data(uint32_t glove_id, uint32_t index, SdkStep& step) override;
    SdkStatus start(uint32_t glove_id) override;
    SdkStatus start_step(uint32_t glove_id, uint32_t index) override;
    SdkStatus finish(uint32_t glove_id) override;
    SdkStatus stop(uint32_t glove_id) override;
    SdkStatus export_calibration(uint32_t glove_id, std::vector<uint8_t>& bytes) override;
    SdkStatus shutdown() override;

private:
    bool m_initialized = false;
    bool m_connected = false;
};

} // namespace plugins::manus::calibration
