// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "manus_sdk.hpp"

#include <log_bridge/logger.hpp>

#include <ManusSDK.h>
#include <ManusSDKTypeInitializers.h>
#include <cstring>
#include <mutex>

namespace plugins::manus::calibration
{

namespace
{

// The landscape callback has no user-data argument, so the latest glove list lives here.
std::mutex g_landscape_mutex;
std::vector<SdkGlove> g_landscape_gloves;

std::string code_text(SDKReturnCode code)
{
    return "SDK return code " + std::to_string(static_cast<int>(code));
}

// Folds the SDK return code and the operation's own result flag into one status.
SdkStatus checked(SDKReturnCode code, bool operation_result, const char* operation)
{
    if (code != SDKReturnCode::SDKReturnCode_Success)
    {
        return SdkStatus::failure(std::string(operation) + " failed, " + code_text(code));
    }
    if (!operation_result)
    {
        return SdkStatus::failure(std::string(operation) + " was refused by the SDK");
    }
    return SdkStatus::success();
}

void on_log(LogSeverity severity, const char* text, uint32_t length) noexcept
{
    // Called from inside SDK frames: nothing may escape, and a dropped vendor line is acceptable.
    try
    {
        static const auto logger = isaaccapture::Logger::get("isaaccapture.plugins.manus.ManusSdk");
        const std::string message(text, length);
        switch (severity)
        {
        case LogSeverity_Debug:
            logger->debug("{}", message);
            break;
        case LogSeverity_Warn:
            logger->warn("{}", message);
            break;
        case LogSeverity_Error:
            logger->error("{}", message);
            break;
        default:
            logger->info("{}", message);
            break;
        }
    }
    catch (...)
    {
    }
}

void on_landscape(const Landscape* landscape) noexcept
{
    if (landscape == nullptr)
    {
        return;
    }
    try
    {
        std::vector<SdkGlove> gloves;
        const auto& list = landscape->gloveDevices;
        for (uint32_t i = 0; i < list.gloveCount; ++i)
        {
            const GloveLandscapeData& glove = list.gloves[i];
            if (glove.side == ::Side::Side_Left || glove.side == ::Side::Side_Right)
            {
                gloves.push_back({ glove.side == ::Side::Side_Left ? plugins::manus::calibration::Side::Left :
                                                                     plugins::manus::calibration::Side::Right,
                                   glove.id });
            }
        }
        std::lock_guard<std::mutex> lock(g_landscape_mutex);
        g_landscape_gloves = std::move(gloves);
    }
    catch (...)
    {
    }
}

// Registered with empty handlers so the SDK session has the same streams as a full Manus client. Whether the
// calibration calls depend on them is not established.
void on_skeleton(const SkeletonStreamInfo*) noexcept
{
}

void on_raw_device_data(const RawDeviceDataInfo*) noexcept
{
}

} // namespace

ManusGloveSdk::~ManusGloveSdk()
{
    // Best effort: a destructor cannot report, and the engine has already checked the result.
    (void)shutdown();
}

std::optional<std::string> ManusGloveSdk::version()
{
    // The version query needs an initialized SDK, so the ready event reports null before a start.
    if (!m_initialized)
    {
        return std::nullopt;
    }
    ManusVersion sdk_version{};
    ManusVersion core_version{};
    bool compatible = false;
    if (CoreSdk_GetVersionsAndCheckCompatibility(&sdk_version, &core_version, &compatible) !=
        SDKReturnCode::SDKReturnCode_Success)
    {
        return std::nullopt;
    }
    return std::string(sdk_version.versionInfo, strnlen(sdk_version.versionInfo, sizeof(sdk_version.versionInfo)));
}

SdkStatus ManusGloveSdk::initialize()
{
    // Registered first so messages emitted during initialization are captured too.
    CoreSdk_RegisterCallbackForOnLog(on_log);
    m_initialized = true;
    const SDKReturnCode init = CoreSdk_InitializeIntegrated();
    if (init != SDKReturnCode::SDKReturnCode_Success)
    {
        return SdkStatus::failure("CoreSdk_InitializeIntegrated failed, " + code_text(init));
    }

    CoreSdk_RegisterCallbackForRawSkeletonStream(on_skeleton);
    CoreSdk_RegisterCallbackForLandscapeStream(on_landscape);
    CoreSdk_RegisterCallbackForRawDeviceDataStream(on_raw_device_data);

    // Same coordinate setup as the tracking plugin; the SDK requires one before connecting.
    CoordinateSystemVUH vuh;
    CoordinateSystemVUH_Init(&vuh);
    vuh.handedness = ::Side::Side_Right;
    vuh.up = AxisPolarity::AxisPolarity_PositiveY;
    vuh.view = AxisView::AxisView_ZToViewer;
    vuh.unitScale = 1.0f;
    const SDKReturnCode coordinates = CoreSdk_InitializeCoordinateSystemWithVUH(vuh, true);
    if (coordinates != SDKReturnCode::SDKReturnCode_Success)
    {
        return SdkStatus::failure("CoreSdk_InitializeCoordinateSystemWithVUH failed, " + code_text(coordinates));
    }
    return SdkStatus::success();
}

ConnectResult ManusGloveSdk::connect()
{
    // Blocks up to one second while it listens for hosts.
    if (CoreSdk_LookForHosts(1, false) != SDKReturnCode::SDKReturnCode_Success)
    {
        return { ConnectState::Connecting, "host lookup failed" };
    }
    uint32_t host_count = 0;
    if (CoreSdk_GetNumberOfAvailableHostsFound(&host_count) != SDKReturnCode::SDKReturnCode_Success || host_count == 0)
    {
        return { ConnectState::Connecting, "no host found yet" };
    }
    std::vector<ManusHost> hosts(host_count);
    if (CoreSdk_GetAvailableHostsFound(hosts.data(), host_count) != SDKReturnCode::SDKReturnCode_Success)
    {
        return { ConnectState::Connecting, "cannot read the host list" };
    }
    const SDKReturnCode connected = CoreSdk_ConnectToHost(hosts[0]);
    if (connected == SDKReturnCode::SDKReturnCode_NotConnected)
    {
        return { ConnectState::Connecting, "host not connected yet" };
    }
    if (connected != SDKReturnCode::SDKReturnCode_Success)
    {
        return { ConnectState::Failed, code_text(connected) };
    }
    m_connected = true;
    // Hand motion is not used by calibration; set to none as other Manus clients do once connected.
    CoreSdk_SetRawSkeletonHandMotion(HandMotion_None);
    return { ConnectState::Connected, {} };
}

std::vector<SdkGlove> ManusGloveSdk::gloves()
{
    std::lock_guard<std::mutex> lock(g_landscape_mutex);
    return g_landscape_gloves;
}

SdkStatus ManusGloveSdk::step_count(uint32_t glove_id, uint32_t& count)
{
    GloveCalibrationArgs args{};
    args.gloveId = glove_id;
    count = 0;
    return checked(CoreSdk_GloveCalibrationGetNumberOfSteps(args, &count), true, "GloveCalibrationGetNumberOfSteps");
}

SdkStatus ManusGloveSdk::step_data(uint32_t glove_id, uint32_t index, SdkStep& step)
{
    GloveCalibrationStepArgs args{};
    args.gloveId = glove_id;
    args.stepIndex = index;
    GloveCalibrationStepData data{};
    const SdkStatus status =
        checked(CoreSdk_GloveCalibrationGetStepData(args, &data), true, "GloveCalibrationGetStepData");
    if (!status.ok)
    {
        return status;
    }
    step.index = data.index;
    // The SDK fills fixed arrays; bound the read in case a string is not terminated.
    step.title.assign(data.title, strnlen(data.title, sizeof(data.title)));
    step.instruction.assign(data.description, strnlen(data.description, sizeof(data.description)));
    step.time_s = data.time;
    return SdkStatus::success();
}

SdkStatus ManusGloveSdk::start(uint32_t glove_id)
{
    GloveCalibrationArgs args{};
    args.gloveId = glove_id;
    bool result = false;
    const SDKReturnCode code = CoreSdk_GloveCalibrationStart(args, &result);
    return checked(code, result, "GloveCalibrationStart");
}

SdkStatus ManusGloveSdk::start_step(uint32_t glove_id, uint32_t index)
{
    GloveCalibrationStepArgs args{};
    args.gloveId = glove_id;
    args.stepIndex = index;
    bool result = false;
    const SDKReturnCode code = CoreSdk_GloveCalibrationStartStep(args, &result);
    return checked(code, result, "GloveCalibrationStartStep");
}

SdkStatus ManusGloveSdk::finish(uint32_t glove_id)
{
    GloveCalibrationArgs args{};
    args.gloveId = glove_id;
    bool result = false;
    const SDKReturnCode code = CoreSdk_GloveCalibrationFinish(args, &result);
    return checked(code, result, "GloveCalibrationFinish");
}

SdkStatus ManusGloveSdk::stop(uint32_t glove_id)
{
    GloveCalibrationArgs args{};
    args.gloveId = glove_id;
    bool result = false;
    const SDKReturnCode code = CoreSdk_GloveCalibrationStop(args, &result);
    return checked(code, result, "GloveCalibrationStop");
}

SdkStatus ManusGloveSdk::export_calibration(uint32_t glove_id, std::vector<uint8_t>& bytes)
{
    bytes.clear();
    // The byte read below takes no glove id: it returns the glove most recently sized.
    uint32_t size = 0;
    const SdkStatus sized = checked(CoreSdk_GetGloveCalibrationSize(glove_id, &size), true, "GetGloveCalibrationSize");
    if (!sized.ok)
    {
        return sized;
    }
    if (size == 0)
    {
        return SdkStatus::failure("the SDK reports a zero-size calibration");
    }
    std::vector<uint8_t> buffer(size);
    const SdkStatus read = checked(CoreSdk_GetGloveCalibration(buffer.data(), size), true, "GetGloveCalibration");
    if (!read.ok)
    {
        return read;
    }
    bytes = std::move(buffer);
    return SdkStatus::success();
}

SdkStatus ManusGloveSdk::shutdown()
{
    if (!m_initialized)
    {
        return SdkStatus::success();
    }
    m_initialized = false;
    CoreSdk_RegisterCallbackForRawSkeletonStream(nullptr);
    CoreSdk_RegisterCallbackForLandscapeStream(nullptr);
    CoreSdk_RegisterCallbackForRawDeviceDataStream(nullptr);
    SdkStatus status = SdkStatus::success();
    if (m_connected)
    {
        const SDKReturnCode disconnected = CoreSdk_Disconnect();
        m_connected = false;
        if (disconnected != SDKReturnCode::SDKReturnCode_Success)
        {
            status = SdkStatus::failure("CoreSdk_Disconnect failed, " + code_text(disconnected));
        }
    }
    const SDKReturnCode down = CoreSdk_ShutDown();
    if (down != SDKReturnCode::SDKReturnCode_Success && status.ok)
    {
        status = SdkStatus::failure("CoreSdk_ShutDown failed, " + code_text(down));
    }
    CoreSdk_RegisterCallbackForOnLog(nullptr);
    std::lock_guard<std::mutex> lock(g_landscape_mutex);
    g_landscape_gloves.clear();
    return status;
}

} // namespace plugins::manus::calibration
