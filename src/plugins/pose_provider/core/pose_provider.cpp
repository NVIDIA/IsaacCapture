// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/pose_provider/pose_provider.hpp"

#include <log_bridge/logger.hpp>

#include <exception>
#include <stdexcept>
#include <string>
#include <utility>

namespace pose_provider
{

namespace
{

constexpr const char* kLoggerName = "isaaccapture.plugins.pose_provider.PoseProvider";

const char* component_name(ProviderComponent component)
{
    switch (component)
    {
    case ProviderComponent::Provider:
        return "Provider";
    case ProviderComponent::Setup:
        return "Setup";
    case ProviderComponent::Engine:
        return "Engine";
    }
    return "unknown";
}

std::string ms_text(std::chrono::milliseconds duration)
{
    return std::to_string(duration.count()) + " ms";
}

void check_timeout(const char* name, std::chrono::milliseconds timeout)
{
    if (timeout <= std::chrono::milliseconds::zero() || timeout > kMaxProviderTimeout)
    {
        throw std::invalid_argument(std::string(name) + " must be in (0, " + ms_text(kMaxProviderTimeout) + "], not " +
                                    ms_text(timeout));
    }
}

std::string health_text(const EngineHealth& health)
{
    return "dropped image groups " + std::to_string(health.dropped_image_groups) + ", dropped IMU samples " +
           std::to_string(health.dropped_imu_samples) + ", IMU gaps " + std::to_string(health.imu_gaps) +
           ", invalid-input rejections " + std::to_string(health.invalid_input_rejections) + ", not-running rejections " +
           std::to_string(health.not_running_rejections) + ", output stalls " + std::to_string(health.output_stalls);
}

} // namespace

PoseProvider::PoseProvider(ProviderConfig config, std::unique_ptr<IPoseEngine> engine, SetupSupplier setup_supplier)
    : config_(config), setup_supplier_(std::move(setup_supplier)), engine_(std::move(engine))
{
    if (!engine_)
    {
        throw std::invalid_argument("PoseProvider needs an engine");
    }
    check_timeout("startup_timeout", config_.startup_timeout);
    check_timeout("drain_timeout", config_.drain_timeout);
    startup_deadline_ = Clock::now() + config_.startup_timeout;
    status_.startup_timeout = config_.startup_timeout;
    status_.drain_timeout = config_.drain_timeout;
}

PoseProvider::~PoseProvider()
{
    try
    {
        abort();
    }
    catch (...)
    {
    }
}

void PoseProvider::update()
{
    try
    {
        ProviderState before = status_.state;
        while (status_.state != ProviderState::Finished)
        {
            step();
            if (status_.state == before)
            {
                break;
            }
            before = status_.state;
        }
    }
    catch (const std::exception& e)
    {
        fail(ProviderComponent::Provider, std::string("internal error: ") + e.what());
    }
    catch (...)
    {
        fail(ProviderComponent::Provider, "internal error");
    }
}

void PoseProvider::request_start()
{
    start_requested_ = true;
}

void PoseProvider::request_finish()
{
    finish_requested_ = true;
}

void PoseProvider::abort()
{
    if (status_.state != ProviderState::Finished)
    {
        fail(ProviderComponent::Provider, "aborted");
    }
}

bool PoseProvider::finished() const
{
    return status_.state == ProviderState::Finished;
}

ProviderStatus PoseProvider::status() const
{
    return status_;
}

void PoseProvider::step()
{
    switch (status_.state)
    {
    case ProviderState::Preparing:
        prepare();
        break;
    case ProviderState::Ready:
        await_start();
        break;
    case ProviderState::Running:
        run();
        break;
    case ProviderState::Draining:
        drain();
        break;
    case ProviderState::Finished:
        break;
    }
}

void PoseProvider::prepare()
{
    if (finish_requested_)
    {
        finish_run();
        return;
    }

    std::optional<ResolvedPoseSetup> setup;
    if (setup_supplier_)
    {
        try
        {
            setup = setup_supplier_();
        }
        catch (const std::exception& e)
        {
            fail(ProviderComponent::Setup, std::string("setup supplier: ") + e.what());
            return;
        }
        catch (...)
        {
            fail(ProviderComponent::Setup, "setup supplier: unknown exception");
            return;
        }
    }

    if (!setup)
    {
        if (Clock::now() >= startup_deadline_)
        {
            fail(ProviderComponent::Setup,
                 "no setup within the startup timeout (" + ms_text(config_.startup_timeout) + ")");
        }
        return;
    }

    std::string rejection;
    if (!call_engine("validate_configuration", [&] { rejection = engine_->validate_configuration(*setup); }))
    {
        return;
    }
    if (!rejection.empty())
    {
        fail(ProviderComponent::Setup, "the engine rejected the setup: " + rejection);
        return;
    }
    if (!call_engine("initialize", [&] { engine_->initialize(*setup); }))
    {
        return;
    }
    setup_ = std::move(setup);
    enter(ProviderState::Ready);
}

void PoseProvider::await_start()
{
    if (!check_health())
    {
        return;
    }
    // A finish takes priority over a pending start, and never needs finish_input() before start().
    if (finish_requested_)
    {
        finish_run();
        return;
    }
    if (start_requested_)
    {
        if (call_engine("start", [&] { engine_->start(); }))
        {
            enter(ProviderState::Running);
        }
        return;
    }
    if (Clock::now() >= startup_deadline_)
    {
        fail(ProviderComponent::Provider,
             "not started within the startup timeout (" + ms_text(config_.startup_timeout) + ")");
    }
}

void PoseProvider::run()
{
    // Checked first, so that no new input is admitted once a finish is requested.
    if (finish_requested_)
    {
        drain_deadline_ = Clock::now() + config_.drain_timeout;
        enter(ProviderState::Draining);
        return;
    }
    if (!take_estimates())
    {
        return;
    }
    check_health();
}

void PoseProvider::drain()
{
    if (!input_finished_)
    {
        if (!call_engine("finish_input", [&] { engine_->finish_input(); }))
        {
            return;
        }
        input_finished_ = true;
    }

    // Read before polling, so that no estimate can appear after a drained result.
    bool drained = false;
    if (!call_engine("input_drained", [&] { drained = engine_->input_drained(); }))
    {
        return;
    }
    if (!take_estimates() || !check_health())
    {
        return;
    }
    if (drained)
    {
        finish_run();
        return;
    }
    if (Clock::now() >= drain_deadline_)
    {
        fail(ProviderComponent::Engine,
             "the engine did not drain within the drain timeout (" + ms_text(config_.drain_timeout) + ")");
    }
}

void PoseProvider::enter(ProviderState state)
{
    status_.state = state;
    auto logger = isaaccapture::Logger::get(kLoggerName);
    switch (state)
    {
    case ProviderState::Ready:
        logger->info("Ready: the engine is initialized");
        break;
    case ProviderState::Running:
        logger->info("Running");
        break;
    case ProviderState::Draining:
        logger->info("Draining: finishing input, then waiting up to {} for the engine", ms_text(config_.drain_timeout));
        break;
    case ProviderState::Preparing:
    case ProviderState::Finished:
        break;
    }
}

bool PoseProvider::take_estimates()
{
    for (;;)
    {
        EnginePose pose;
        bool taken = false;
        if (!call_engine("try_get_next_pose", [&] { taken = engine_->try_get_next_pose(pose); }))
        {
            return false;
        }
        if (!taken)
        {
            return true;
        }
        // No image group is admitted yet, so every estimate breaks the accepted-group order.
        fail(ProviderComponent::Engine,
             "estimate for image group " + std::to_string(pose.group_id) + ", which was never accepted");
        return false;
    }
}

bool PoseProvider::check_health()
{
    EngineHealth health;
    if (!call_engine("health", [&] { health = engine_->health(); }))
    {
        return false;
    }
    status_.engine_health = health;
    if (health.fatal_error)
    {
        fail(ProviderComponent::Engine, "engine fatal error: " + health.fatal_reason);
        return false;
    }
    return true;
}

bool PoseProvider::call_engine(const char* call, const std::function<void()>& work)
{
    try
    {
        work();
        return true;
    }
    catch (const std::exception& e)
    {
        fail(ProviderComponent::Engine, std::string(call) + ": " + e.what());
    }
    catch (...)
    {
        fail(ProviderComponent::Engine, std::string(call) + ": unknown exception");
    }
    return false;
}

void PoseProvider::fail(ProviderComponent component, std::string reason)
{
    if (status_.state == ProviderState::Finished)
    {
        return;
    }
    record_failure(component, std::move(reason));
    finish_run();
}

void PoseProvider::record_failure(ProviderComponent component, std::string reason)
{
    auto logger = isaaccapture::Logger::get(kLoggerName);
    if (status_.failed)
    {
        logger->warn("Later failure ({}), not recorded: {}", component_name(component), reason);
        return;
    }
    status_.failed = true;
    status_.failed_component = component;
    status_.failure_reason = std::move(reason);
    logger->error("Run failed ({}): {}", component_name(component), status_.failure_reason);
}

void PoseProvider::stop_engine()
{
    if (engine_stopped_)
    {
        return;
    }
    engine_stopped_ = true;
    try
    {
        engine_->stop();
    }
    catch (const std::exception& e)
    {
        record_failure(ProviderComponent::Engine, std::string("stop: ") + e.what());
    }
    catch (...)
    {
        record_failure(ProviderComponent::Engine, "stop: unknown exception");
    }
}

void PoseProvider::finish_run()
{
    // Engine calls here go straight to record_failure(): fail() would re-enter finish_run() before Finished is set.
    stop_engine();
    // The contract does not promise health() before initialize(), and an engine that was never initialized has no
    // run to report on.
    if (setup_)
    {
        try
        {
            status_.engine_health = engine_->health();
            if (status_.engine_health.fatal_error && !status_.failed)
            {
                record_failure(ProviderComponent::Engine, "engine fatal error: " + status_.engine_health.fatal_reason);
            }
        }
        catch (const std::exception& e)
        {
            record_failure(ProviderComponent::Engine, std::string("health: ") + e.what());
        }
        catch (...)
        {
            record_failure(ProviderComponent::Engine, "health: unknown exception");
        }
    }

    status_.state = ProviderState::Finished;
    auto logger = isaaccapture::Logger::get(kLoggerName);
    if (status_.failed)
    {
        logger->error("Finished: failed ({}): {}; {}", component_name(*status_.failed_component),
                      status_.failure_reason, health_text(status_.engine_health));
    }
    else
    {
        logger->info("Finished; {}", health_text(status_.engine_health));
    }
}

} // namespace pose_provider
