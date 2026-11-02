// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <manus_calibration/engine.hpp>

#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace plugins::manus::calibration::testing
{

// One ordered record of SDK calls and emitted events, so tests can assert cross-cutting ordering.
using CallLog = std::vector<std::string>;

class FakeClock final : public Clock
{
public:
    double now_s() override
    {
        return now;
    }

    void advance(double seconds)
    {
        now += seconds;
    }

    double now = 100.0;
};

class MemoryWriter final : public ExportWriter
{
public:
    bool directory_usable(const std::string&) override
    {
        return usable;
    }

    std::optional<std::string> write(const std::string& dir,
                                     const std::string& basename,
                                     const std::vector<uint8_t>& bytes) override
    {
        if (fail_with)
        {
            return fail_with;
        }
        files[dir + "/" + basename] = bytes;
        return std::nullopt;
    }

    bool usable = true;
    std::optional<std::string> fail_with;
    std::map<std::string, std::vector<uint8_t>> files;
};

class FakeSdk final : public GloveCalibrationSdk
{
public:
    explicit FakeSdk(CallLog& log) : m_log(log)
    {
    }

    // Names the call that fails: initialize, connect, step_count, step_data, start, start_step, finish, stop,
    // export. `fail_nth` counts occurrences of that call, starting at 1.
    std::string fail_call;
    int fail_nth = 1;

    std::vector<SdkGlove> landscape = { { Side::Left, 10 }, { Side::Right, 20 } };
    // Number of connect() calls that report Connecting before Connected.
    int connecting_attempts = 0;
    uint32_t steps = 3;
    // Time of each step, indexed by step; missing entries are untimed.
    std::vector<float> step_times = { 0.0f, 2.0f, 0.0f };
    // Per-glove override of step_times, keyed by glove id.
    std::map<uint32_t, std::vector<float>> glove_step_times;
    std::vector<uint8_t> export_bytes = { 1, 2, 3, 4, 5 };
    // What shutdown() reports; a failure models an SDK that could not be released.
    SdkStatus shutdown_result = SdkStatus::success();
    std::function<void()> on_export;
    std::function<void()> on_start_step;

    std::optional<std::string> version() override
    {
        return std::string("test-sdk");
    }

    SdkStatus initialize() override
    {
        return step("initialize");
    }

    ConnectResult connect() override
    {
        record("connect");
        if (should_fail("connect"))
        {
            return { ConnectState::Failed, "injected" };
        }
        if (connect_calls++ < connecting_attempts)
        {
            return { ConnectState::Connecting, "waiting" };
        }
        return { ConnectState::Connected, {} };
    }

    std::vector<SdkGlove> gloves() override
    {
        return landscape;
    }

    SdkStatus step_count(uint32_t, uint32_t& count) override
    {
        count = steps;
        return step("step_count");
    }

    SdkStatus step_data(uint32_t glove, uint32_t index, SdkStep& out) override
    {
        out.index = index;
        out.title = "Step " + std::to_string(index);
        out.instruction = "Do pose " + std::to_string(index);
        const auto override_times = glove_step_times.find(glove);
        const std::vector<float>& times = override_times != glove_step_times.end() ? override_times->second : step_times;
        out.time_s = index < times.size() ? times[index] : 0.0f;
        return step("step_data");
    }

    SdkStatus start(uint32_t glove) override
    {
        return step("start", glove);
    }

    SdkStatus start_step(uint32_t glove, uint32_t index) override
    {
        const SdkStatus status = step("start_step", glove, index);
        if (status.ok && on_start_step)
        {
            on_start_step();
        }
        return status;
    }

    SdkStatus finish(uint32_t glove) override
    {
        return step("finish", glove);
    }

    SdkStatus stop(uint32_t glove) override
    {
        return step("stop", glove);
    }

    SdkStatus export_calibration(uint32_t glove, std::vector<uint8_t>& bytes) override
    {
        if (on_export)
        {
            on_export();
        }
        const SdkStatus status = step("export", glove);
        bytes = export_bytes;
        return status;
    }

    SdkStatus shutdown() override
    {
        record("shutdown");
        return shutdown_result;
    }

    int count(const std::string& call) const
    {
        int n = 0;
        for (const auto& entry : m_log)
        {
            n += entry.rfind("sdk:" + call, 0) == 0 ? 1 : 0;
        }
        return n;
    }

private:
    void record(const std::string& call)
    {
        m_log.push_back("sdk:" + call);
    }

    bool should_fail(const std::string& call)
    {
        if (call != fail_call)
        {
            return false;
        }
        return ++seen_[call] == fail_nth;
    }

    SdkStatus step(const std::string& call, uint32_t glove = 0, uint32_t index = 0)
    {
        record(call + (glove ? ":" + std::to_string(glove) : std::string()) +
               (call == "start_step" ? ":" + std::to_string(index) : std::string()));
        return should_fail(call) ? SdkStatus::failure("injected") : SdkStatus::success();
    }

    CallLog& m_log;
    std::map<std::string, int> seen_;
    int connect_calls = 0;
};

inline std::string event_name(const Event& event)
{
    struct Names
    {
        std::string operator()(const ReadyEvent&) const
        {
            return "ready";
        }
        std::string operator()(const StatusEvent&) const
        {
            return "status";
        }
        std::string operator()(const PromptEvent&) const
        {
            return "prompt";
        }
        std::string operator()(const StepAcceptedEvent&) const
        {
            return "step_accepted";
        }
        std::string operator()(const StepProgressEvent&) const
        {
            return "step_progress";
        }
        std::string operator()(const RejectedEvent&) const
        {
            return "rejected";
        }
        std::string operator()(const ExportedEvent&) const
        {
            return "exported";
        }
        std::string operator()(const ResultEvent&) const
        {
            return "result";
        }
    };
    return std::visit(Names{}, event);
}

// Engine plus fakes, with every emitted event captured in order.
struct Rig
{
    Rig() : sdk(log)
    {
        engine = std::make_unique<CalibrationEngine>(
            sdk, writer, clock,
            [this](const Event& event)
            {
                events.push_back(event);
                log.push_back("event:" + event_name(event));
            },
            config);
    }

    void send(const Command& command)
    {
        engine->handle(command);
    }

    void start(std::vector<Side> sides = { Side::Left, Side::Right })
    {
        send(StartCommand{ "att1", std::move(sides), "/stage" });
    }

    // Ticks until a prompt is offered or the engine finishes; discovery needs a few ticks.
    void run_to_prompt()
    {
        for (int i = 0; i < 40 && !engine->finished(); ++i)
        {
            if (last_prompt() && !prompt_consumed())
            {
                return;
            }
            clock.advance(0.5);
            engine->tick();
        }
    }

    const PromptEvent* last_prompt() const
    {
        for (auto it = events.rbegin(); it != events.rend(); ++it)
        {
            if (const auto* prompt = std::get_if<PromptEvent>(&*it))
            {
                return prompt;
            }
        }
        return nullptr;
    }

    // True when the newest prompt has already been confirmed.
    bool prompt_consumed() const
    {
        const PromptEvent* prompt = last_prompt();
        for (const Event& event : events)
        {
            if (const auto* accepted = std::get_if<StepAcceptedEvent>(&event))
            {
                if (prompt && accepted->token == prompt->token)
                {
                    return true;
                }
            }
        }
        return false;
    }

    void confirm_current()
    {
        if (last_prompt() == nullptr)
        {
            throw std::runtime_error("no prompt was offered");
        }
        send(ConfirmCommand{ "att1", last_prompt()->token });
    }

    // Confirms each freshly offered prompt and advances the clock until the engine finishes.
    void run_to_end()
    {
        for (int i = 0; i < 500 && !engine->finished(); ++i)
        {
            if (last_prompt() && !prompt_consumed())
            {
                confirm_current();
            }
            clock.advance(0.3);
            engine->tick();
        }
    }

    const ResultEvent* result() const
    {
        for (const Event& event : events)
        {
            if (const auto* r = std::get_if<ResultEvent>(&event))
            {
                return r;
            }
        }
        return nullptr;
    }

    std::vector<const RejectedEvent*> rejections() const
    {
        std::vector<const RejectedEvent*> out;
        for (const Event& event : events)
        {
            if (const auto* r = std::get_if<RejectedEvent>(&event))
            {
                out.push_back(r);
            }
        }
        return out;
    }

    template <typename T>
    std::vector<const T*> all() const
    {
        std::vector<const T*> out;
        for (const Event& event : events)
        {
            if (const auto* e = std::get_if<T>(&event))
            {
                out.push_back(e);
            }
        }
        return out;
    }

    size_t index_of(const std::string& entry) const
    {
        for (size_t i = 0; i < log.size(); ++i)
        {
            if (log[i] == entry)
            {
                return i;
            }
        }
        return log.size();
    }

    CallLog log;
    FakeSdk sdk;
    MemoryWriter writer;
    FakeClock clock;
    EngineConfig config;
    std::vector<Event> events;
    std::unique_ptr<CalibrationEngine> engine;
};

} // namespace plugins::manus::calibration::testing
