// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "export_writer.hpp"
#include "protocol.hpp"

#include <atomic>
#include <cstdint>
#include <functional>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace plugins::manus::calibration
{

struct SdkStatus
{
    bool ok = true;
    std::string message;

    static SdkStatus success()
    {
        return {};
    }

    static SdkStatus failure(std::string message)
    {
        return { false, std::move(message) };
    }
};

struct SdkGlove
{
    Side side;
    uint32_t id;
};

struct SdkStep
{
    uint32_t index = 0;
    std::string title;
    std::string instruction;
    // The SDK's step time: zero is an instant hold, positive a timed motion. A negative value marks a
    // continuous step whose duration is only an estimate; the engine paces it by the operator.
    float time_s = 0.0f;
};

enum class ConnectState
{
    Connecting,
    Connected,
    Failed,
};

struct ConnectResult
{
    ConnectState state;
    std::string message;
};

// The Manus calibration calls the engine needs. Every method reports failure through its return value
// (SDK return code and the operation's own result already folded together by the implementation).
class GloveCalibrationSdk
{
public:
    virtual ~GloveCalibrationSdk() = default;

    virtual std::optional<std::string> version() = 0;
    virtual SdkStatus initialize() = 0;
    // One connection attempt; the engine calls it repeatedly until Connected or Failed.
    virtual ConnectResult connect() = 0;
    // Gloves currently reported by the SDK landscape.
    virtual std::vector<SdkGlove> gloves() = 0;
    virtual SdkStatus step_count(uint32_t glove_id, uint32_t& count) = 0;
    virtual SdkStatus step_data(uint32_t glove_id, uint32_t index, SdkStep& step) = 0;
    virtual SdkStatus start(uint32_t glove_id) = 0;
    virtual SdkStatus start_step(uint32_t glove_id, uint32_t index) = 0;
    virtual SdkStatus finish(uint32_t glove_id) = 0;
    virtual SdkStatus stop(uint32_t glove_id) = 0;
    // The size query and the byte read must happen back to back: the byte read has no glove argument.
    virtual SdkStatus export_calibration(uint32_t glove_id, std::vector<uint8_t>& bytes) = 0;
    // Releases the SDK. A failure means the gloves may still be held, so it can never end as a completed result.
    virtual SdkStatus shutdown() = 0;
};

class Clock
{
public:
    virtual ~Clock() = default;
    // Monotonic seconds.
    virtual double now_s() = 0;
};

struct EngineConfig
{
    double discovery_timeout_s = 15.0;
    double progress_interval_s = 0.25;
    double connect_retry_s = 1.0;
};

// Single-threaded step machine for one calibration attempt. The owner feeds it commands and calls
// tick() regularly; the only thread-safe entry point is request_cancel().
class CalibrationEngine
{
public:
    using EventSink = std::function<void(const Event&)>;

    CalibrationEngine(
        GloveCalibrationSdk& sdk, ExportWriter& writer, Clock& clock, EventSink sink, EngineConfig config = {});

    void handle(const Command& command);
    // Time-driven work: discovery polling, sampling progress, glove presence checks, pending cancellation.
    void tick();
    // Safe from a signal handler or another thread; takes effect on the next tick() or handle().
    void request_cancel() noexcept;

    bool finished() const
    {
        return m_finished;
    }

    Outcome outcome() const
    {
        return m_outcome;
    }

private:
    enum class State
    {
        Idle,
        Discovering,
        AwaitingStep,
        CapturingStep,
        Saving,
        Finished,
    };

    void begin_attempt(const StartCommand& command);
    void handle_confirm(const ConfirmCommand& command);
    void handle_cancel(const CancelCommand& command);
    void poll_discovery();
    void begin_glove();
    void offer_prompt();
    void step_done();
    void finish_glove();
    void advance_capture();
    bool check_gloves();
    bool terminate_if_cancelled();
    void fail(ErrorCode code,
              std::string message,
              std::optional<Side> side = std::nullopt,
              std::optional<uint32_t> step = std::nullopt);
    void terminate(Outcome outcome, std::optional<ErrorInfo> error);
    void emit_status(Phase phase);
    void reject(const std::string& command,
                std::optional<std::string> attempt,
                RejectReason reason,
                std::string message,
                std::optional<std::string> token = std::nullopt);

    GloveCalibrationSdk& m_sdk;
    ExportWriter& m_writer;
    Clock& m_clock;
    EventSink m_sink;
    EngineConfig m_config;

    State m_state = State::Idle;
    std::atomic<bool> m_cancel_requested{ false };
    bool m_finished = false;
    Outcome m_outcome = Outcome::Cancelled;

    std::string m_attempt;
    std::string m_staging_dir;
    std::vector<Side> m_sides;
    std::vector<GloveRef> m_pinned;
    bool m_sdk_initialized = false;
    bool m_connected = false;
    double m_discovery_started_s = 0.0;
    double m_last_connect_s = 0.0;
    std::vector<GloveRef> m_last_reported_gloves;

    size_t m_glove_index = 0;
    bool m_calibrating = false;
    uint32_t m_step_index = 0;
    uint32_t m_step_count = 0;
    uint64_t m_prompt_seq = 0;
    std::string m_current_token;
    StepMode m_current_mode = StepMode::Hold;
    double m_current_duration_s = 0.0;
    double m_step_started_s = 0.0;
    double m_last_progress_s = 0.0;
    std::set<std::string> m_accepted_tokens;
    std::vector<ResultFile> m_exported;
};

} // namespace plugins::manus::calibration
