// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/manus_calibration/engine.hpp"

#include "inc/manus_calibration/sha256.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace plugins::manus::calibration
{

namespace
{

std::string glove_text(Side side, uint32_t id)
{
    return std::string(side_name(side)) + " glove (id " + std::to_string(id) + ")";
}

} // namespace

CalibrationEngine::CalibrationEngine(
    GloveCalibrationSdk& sdk, ExportWriter& writer, Clock& clock, EventSink sink, EngineConfig config)
    : m_sdk(sdk), m_writer(writer), m_clock(clock), m_sink(std::move(sink)), m_config(config)
{
}

void CalibrationEngine::request_cancel() noexcept
{
    m_cancel_requested.store(true, std::memory_order_relaxed);
}

void CalibrationEngine::handle(const Command& command)
{
    if (m_finished)
    {
        return;
    }
    if (terminate_if_cancelled())
    {
        return;
    }
    if (const auto* start = std::get_if<StartCommand>(&command))
    {
        begin_attempt(*start);
    }
    else if (const auto* confirm = std::get_if<ConfirmCommand>(&command))
    {
        handle_confirm(*confirm);
    }
    else if (const auto* cancel = std::get_if<CancelCommand>(&command))
    {
        handle_cancel(*cancel);
    }
}

void CalibrationEngine::tick()
{
    if (m_finished || terminate_if_cancelled())
    {
        return;
    }
    switch (m_state)
    {
    case State::Discovering:
        poll_discovery();
        break;
    case State::AwaitingStep:
        check_gloves();
        break;
    case State::CapturingStep:
        if (check_gloves())
        {
            advance_capture();
        }
        break;
    case State::Idle:
    case State::Saving:
    case State::Finished:
        break;
    }
}

void CalibrationEngine::reject(const std::string& command,
                               std::optional<std::string> attempt,
                               RejectReason reason,
                               std::string message,
                               std::optional<std::string> token)
{
    m_sink(RejectedEvent{ std::move(attempt), command, reason, std::move(message), std::move(token) });
}

void CalibrationEngine::emit_status(Phase phase)
{
    m_sink(StatusEvent{ m_attempt, phase, m_pinned.empty() ? m_last_reported_gloves : m_pinned });
}

void CalibrationEngine::begin_attempt(const StartCommand& command)
{
    if (m_state != State::Idle)
    {
        reject("start", command.attempt, RejectReason::Busy, "an attempt is already active");
        return;
    }
    if (!m_writer.directory_usable(command.staging_dir))
    {
        reject("start", command.attempt, RejectReason::Invalid, "staging_dir is not a writable directory");
        return;
    }
    m_attempt = command.attempt;
    m_sides = command.sides;
    m_staging_dir = command.staging_dir;
    m_state = State::Discovering;
    m_discovery_started_s = m_clock.now_s();
    m_last_connect_s = m_discovery_started_s - m_config.connect_retry_s;
    emit_status(Phase::Discovering);

    // Set before the call so a partially initialized SDK is still shut down on failure.
    m_sdk_initialized = true;
    const SdkStatus init = m_sdk.initialize();
    if (!init.ok)
    {
        fail(ErrorCode::SdkInitFailed, "Manus SDK initialization failed: " + init.message);
    }
}

void CalibrationEngine::poll_discovery()
{
    const double now = m_clock.now_s();
    if (!m_connected && now - m_last_connect_s >= m_config.connect_retry_s)
    {
        m_last_connect_s = now;
        const ConnectResult connect = m_sdk.connect();
        if (connect.state == ConnectState::Failed)
        {
            fail(ErrorCode::SdkInitFailed, "cannot connect to the Manus host: " + connect.message);
            return;
        }
        m_connected = (connect.state == ConnectState::Connected);
    }

    std::vector<GloveRef> found;
    std::vector<Side> missing;
    if (m_connected)
    {
        const std::vector<SdkGlove> landscape = m_sdk.gloves();
        for (const Side side : m_sides)
        {
            const auto glove =
                std::find_if(landscape.begin(), landscape.end(), [side](const SdkGlove& g) { return g.side == side; });
            if (glove == landscape.end())
            {
                missing.push_back(side);
            }
            else
            {
                found.push_back({ side, glove->id });
            }
        }
    }

    // Report discovery progress only when the set of visible gloves changes.
    const bool changed =
        found.size() != m_last_reported_gloves.size() ||
        !std::equal(found.begin(), found.end(), m_last_reported_gloves.begin(),
                    [](const GloveRef& a, const GloveRef& b) { return a.side == b.side && a.glove_id == b.glove_id; });
    if (changed)
    {
        m_last_reported_gloves = found;
        emit_status(Phase::Discovering);
    }

    if (m_connected && missing.empty())
    {
        // The ids are pinned for the whole attempt: a different id later means a replaced glove.
        m_pinned = found;
        begin_glove();
        return;
    }
    if (now - m_discovery_started_s >= m_config.discovery_timeout_s)
    {
        std::string names;
        for (const Side side : m_sides)
        {
            const bool seen =
                std::any_of(found.begin(), found.end(), [side](const GloveRef& g) { return g.side == side; });
            if (!seen)
            {
                names += (names.empty() ? "" : ", ");
                names += side_name(side);
            }
        }
        fail(ErrorCode::NoGloves, "no " + names + " glove discovered within " +
                                      std::to_string(static_cast<int>(m_config.discovery_timeout_s)) + " s");
    }
}

namespace
{

// Empty when the SDK's step time can be acted on; otherwise why it cannot.
std::string unusable_step_time(float time_s)
{
    if (!std::isfinite(time_s))
    {
        return "the SDK reported a non-finite step time";
    }
    return {};
}

} // namespace

void CalibrationEngine::begin_glove()
{
    const GloveRef glove = m_pinned[m_glove_index];
    uint32_t count = 0;
    const SdkStatus counted = m_sdk.step_count(glove.glove_id, count);
    if (!counted.ok)
    {
        fail(ErrorCode::SdkError, "cannot read the calibration step count: " + counted.message, glove.side);
        return;
    }
    if (count == 0)
    {
        fail(ErrorCode::SdkError, "the SDK reports no calibration steps", glove.side);
        return;
    }
    const SdkStatus started = m_sdk.start(glove.glove_id);
    if (!started.ok)
    {
        fail(ErrorCode::SdkError, "cannot start calibration: " + started.message, glove.side);
        return;
    }
    m_calibrating = true;
    m_step_count = count;
    m_step_index = 0;

    // Check every step before the operator performs any, so an unsupported routine fails up front.
    for (uint32_t index = 0; index < count; ++index)
    {
        SdkStep step;
        const SdkStatus read = m_sdk.step_data(glove.glove_id, index, step);
        if (!read.ok)
        {
            fail(ErrorCode::SdkError, "cannot read calibration step data: " + read.message, glove.side, index);
            return;
        }
        if (const std::string problem = unusable_step_time(step.time_s); !problem.empty())
        {
            fail(ErrorCode::StepFailed, problem, glove.side, index);
            return;
        }
    }
    offer_prompt();
}

void CalibrationEngine::offer_prompt()
{
    const GloveRef glove = m_pinned[m_glove_index];
    SdkStep step;
    const SdkStatus read = m_sdk.step_data(glove.glove_id, m_step_index, step);
    if (!read.ok)
    {
        fail(ErrorCode::SdkError, "cannot read calibration step data: " + read.message, glove.side, m_step_index);
        return;
    }
    if (const std::string problem = unusable_step_time(step.time_s); !problem.empty())
    {
        fail(ErrorCode::StepFailed, problem, glove.side, m_step_index);
        return;
    }
    m_current_token = m_attempt + "-" + std::to_string(++m_prompt_seq);
    // A negative time marks a continuous step with only an estimated duration. The SDK offers no way to
    // detect its completion, so it is paced by the operator like a hold: confirm starts the SDK step and
    // the next prompt follows. The estimate is guidance only.
    const bool continuous = step.time_s < 0.0f;
    m_current_mode = step.time_s > 0.0f ? StepMode::Motion : StepMode::Hold;
    m_current_duration_s = m_current_mode == StepMode::Motion ? static_cast<double>(step.time_s) : 0.0;
    const double estimated_s = continuous ? -static_cast<double>(step.time_s) : 0.0;
    m_state = State::AwaitingStep;
    emit_status(Phase::AwaitingStep);
    m_sink(PromptEvent{ m_attempt, m_current_token, glove.side, glove.glove_id, m_step_index, m_step_count, step.title,
                        step.instruction, m_current_mode, m_current_duration_s, continuous, estimated_s });
}

void CalibrationEngine::handle_confirm(const ConfirmCommand& command)
{
    if (m_state == State::Idle || command.attempt != m_attempt)
    {
        reject("confirm", command.attempt, RejectReason::UnknownAttempt, "no such attempt", command.token);
        return;
    }
    // A repeat of an already accepted token is acknowledged again without starting another SDK step.
    if (m_accepted_tokens.count(command.token) != 0)
    {
        m_sink(StepAcceptedEvent{ m_attempt, command.token });
        return;
    }
    if (m_state != State::AwaitingStep)
    {
        reject("confirm", command.attempt, RejectReason::Unexpected, "no step is awaiting confirmation", command.token);
        return;
    }
    if (command.token != m_current_token)
    {
        reject("confirm", command.attempt, RejectReason::StaleToken, "the token does not match the offered step",
               command.token);
        return;
    }
    if (!check_gloves())
    {
        return;
    }

    const GloveRef glove = m_pinned[m_glove_index];
    const SdkStatus started = m_sdk.start_step(glove.glove_id, m_step_index);
    if (!started.ok)
    {
        fail(ErrorCode::StepFailed, "the SDK refused to start the step: " + started.message, glove.side, m_step_index);
        return;
    }
    m_accepted_tokens.insert(command.token);
    m_sink(StepAcceptedEvent{ m_attempt, command.token });

    if (m_current_mode == StepMode::Motion)
    {
        m_state = State::CapturingStep;
        m_step_started_s = m_clock.now_s();
        m_last_progress_s = m_step_started_s;
        emit_status(Phase::CapturingStep);
        m_sink(StepProgressEvent{ m_attempt, m_current_token, 0.0 });
        return;
    }
    step_done();
}

void CalibrationEngine::handle_cancel(const CancelCommand& command)
{
    if (m_state == State::Idle || command.attempt != m_attempt)
    {
        reject("cancel", command.attempt, RejectReason::UnknownAttempt, "no such attempt");
        return;
    }
    request_cancel();
    terminate_if_cancelled();
}

void CalibrationEngine::advance_capture()
{
    const double now = m_clock.now_s();
    const double elapsed = std::min(std::max(now - m_step_started_s, 0.0), m_current_duration_s);
    const bool window_over = elapsed >= m_current_duration_s;
    if (window_over || now - m_last_progress_s >= m_config.progress_interval_s)
    {
        m_last_progress_s = now;
        m_sink(StepProgressEvent{ m_attempt, m_current_token, elapsed });
    }
    if (window_over)
    {
        step_done();
    }
}

void CalibrationEngine::step_done()
{
    ++m_step_index;
    if (m_step_index < m_step_count)
    {
        offer_prompt();
        return;
    }
    finish_glove();
}

void CalibrationEngine::finish_glove()
{
    const GloveRef glove = m_pinned[m_glove_index];
    if (!check_gloves())
    {
        return;
    }
    const SdkStatus finished = m_sdk.finish(glove.glove_id);
    if (!finished.ok)
    {
        fail(ErrorCode::SdkError, "cannot finish calibration: " + finished.message, glove.side);
        return;
    }
    m_calibrating = false;

    m_state = State::Saving;
    emit_status(Phase::Saving);
    if (terminate_if_cancelled())
    {
        return;
    }
    std::vector<uint8_t> bytes;
    const SdkStatus exported = m_sdk.export_calibration(glove.glove_id, bytes);
    if (!exported.ok)
    {
        fail(ErrorCode::ExportFailed, "cannot export the calibration: " + exported.message, glove.side);
        return;
    }
    if (bytes.empty())
    {
        fail(ErrorCode::ExportFailed, "the SDK exported an empty calibration", glove.side);
        return;
    }
    if (terminate_if_cancelled())
    {
        return;
    }
    const std::string basename = m_attempt + "_" + std::string(side_name(glove.side)) + ".mcal";
    if (const auto error = m_writer.write(m_staging_dir, basename, bytes))
    {
        fail(ErrorCode::ExportFailed, "cannot save the calibration: " + *error, glove.side);
        return;
    }
    const ResultFile file{ glove.side, basename, bytes.size(), sha256_hex(bytes) };
    m_exported.push_back(file);
    m_sink(ExportedEvent{ m_attempt, glove.side, glove.glove_id, file.file, file.bytes, file.sha256 });

    ++m_glove_index;
    if (m_glove_index < m_pinned.size())
    {
        if (!terminate_if_cancelled())
        {
            begin_glove();
        }
        return;
    }
    terminate(Outcome::Completed, std::nullopt);
}

bool CalibrationEngine::check_gloves()
{
    const std::vector<SdkGlove> landscape = m_sdk.gloves();
    for (const GloveRef& pinned : m_pinned)
    {
        const auto same_side =
            std::find_if(landscape.begin(), landscape.end(), [&](const SdkGlove& g) { return g.side == pinned.side; });
        if (same_side == landscape.end())
        {
            fail(ErrorCode::GloveMissing, glove_text(pinned.side, pinned.glove_id) + " is no longer connected",
                 pinned.side, m_calibrating ? std::optional<uint32_t>(m_step_index) : std::nullopt);
            return false;
        }
        if (same_side->id != pinned.glove_id)
        {
            fail(ErrorCode::GloveReplaced,
                 std::string(side_name(pinned.side)) + " glove changed from id " + std::to_string(pinned.glove_id) +
                     " to id " + std::to_string(same_side->id),
                 pinned.side, m_calibrating ? std::optional<uint32_t>(m_step_index) : std::nullopt);
            return false;
        }
    }
    return true;
}

bool CalibrationEngine::terminate_if_cancelled()
{
    if (m_finished || !m_cancel_requested.load(std::memory_order_relaxed))
    {
        return m_finished;
    }
    terminate(Outcome::Cancelled, std::nullopt);
    return true;
}

void CalibrationEngine::fail(ErrorCode code, std::string message, std::optional<Side> side, std::optional<uint32_t> step)
{
    terminate(Outcome::Failed, ErrorInfo{ code, std::move(message), side, step });
}

void CalibrationEngine::terminate(Outcome outcome, std::optional<ErrorInfo> error)
{
    if (m_finished)
    {
        return;
    }
    if (!m_attempt.empty())
    {
        emit_status(Phase::Stopping);
    }
    // An SDK calibration left running would keep the glove in calibration mode, so stop it first. If it
    // cannot be stopped the cleanup is uncertain, which must not be reported as a clean cancellation.
    if (m_calibrating)
    {
        const GloveRef glove = m_pinned[m_glove_index];
        const SdkStatus stopped = m_sdk.stop(glove.glove_id);
        m_calibrating = false;
        if (!stopped.ok && outcome == Outcome::Cancelled)
        {
            outcome = Outcome::Failed;
            error = ErrorInfo{ ErrorCode::SdkError, "cannot stop calibration: " + stopped.message, glove.side,
                               m_step_index };
        }
    }
    if (m_sdk_initialized)
    {
        const SdkStatus released = m_sdk.shutdown();
        m_sdk_initialized = false;
        // A completed or cancelled outcome promises the gloves are free, so a failed release downgrades it.
        if (!released.ok && outcome != Outcome::Failed)
        {
            outcome = Outcome::Failed;
            error = ErrorInfo{ ErrorCode::SdkError, "cannot shut down the SDK: " + released.message, std::nullopt,
                               std::nullopt };
        }
    }

    m_state = State::Finished;
    m_finished = true;
    m_outcome = outcome;

    // Sent last, after the SDK is down: a supervisor that sees `result` can rely on the gloves being free.
    ResultEvent result;
    if (!m_attempt.empty())
    {
        result.attempt = m_attempt;
    }
    result.outcome = outcome;
    if (outcome == Outcome::Completed)
    {
        result.files = m_exported;
    }
    result.error = std::move(error);
    m_sink(result);
}

} // namespace plugins::manus::calibration
