// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "fake_sdk.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <manus_calibration/protocol.hpp>
#include <manus_calibration/sha256.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <string>
#include <vector>

namespace
{

using namespace plugins::manus::calibration;
using namespace plugins::manus::calibration::testing;
using Catch::Matchers::ContainsSubstring;

ConfirmCommand confirm(const std::string& token, const std::string& attempt = "att1")
{
    return ConfirmCommand{ attempt, token };
}

// One result, last in the stream, after exactly one SDK shutdown: the terminal contract every path must keep.
void require_clean_terminal(const Rig& rig)
{
    REQUIRE(rig.engine->finished());
    REQUIRE(rig.all<ResultEvent>().size() == 1);
    REQUIRE(event_name(rig.events.back()) == "result");
    REQUIRE(rig.sdk.count("shutdown") == 1);
    REQUIRE(rig.index_of("sdk:shutdown") < rig.index_of("event:result"));
}

const ResultEvent& the_result(const Rig& rig)
{
    REQUIRE(rig.result() != nullptr);
    return *rig.result();
}

// Confirms the current prompt, then advances time until a new prompt is offered or the engine finishes.
void confirm_and_advance(Rig& rig)
{
    const size_t prompts = rig.all<PromptEvent>().size();
    rig.confirm_current();
    for (int i = 0; i < 100 && !rig.engine->finished() && rig.all<PromptEvent>().size() == prompts; ++i)
    {
        rig.clock.advance(0.5);
        rig.engine->tick();
    }
}

void tick_for(Rig& rig, int ticks, double step_s = 0.3)
{
    for (int i = 0; i < ticks && !rig.engine->finished(); ++i)
    {
        rig.clock.advance(step_s);
        rig.engine->tick();
    }
}

// Runs until the engine finishes without any confirmation, e.g. a discovery timeout.
void tick_until_finished(Rig& rig, double step_s = 0.5)
{
    for (int i = 0; i < 200 && !rig.engine->finished(); ++i)
    {
        rig.clock.advance(step_s);
        rig.engine->tick();
    }
}

std::vector<std::string> sdk_calls(const Rig& rig)
{
    std::vector<std::string> calls;
    for (const std::string& entry : rig.log)
    {
        if (entry.rfind("sdk:", 0) == 0)
        {
            calls.push_back(entry);
        }
    }
    return calls;
}

bool has_call(const Rig& rig, const std::string& call)
{
    const auto calls = sdk_calls(rig);
    return std::find(calls.begin(), calls.end(), call) != calls.end();
}

} // namespace

// --- happy paths -------------------------------------------------------------------------------------------

TEST_CASE("a left-only calibration completes with one exported file", "[unit][engine][flow]")
{
    Rig rig;
    rig.start({ Side::Left });
    rig.run_to_end();

    require_clean_terminal(rig);
    const ResultEvent& result = the_result(rig);
    CHECK(result.outcome == Outcome::Completed);
    CHECK(result.attempt == std::optional<std::string>("att1"));
    CHECK_FALSE(result.error.has_value());
    REQUIRE(result.files.size() == 1);
    CHECK(result.files[0].side == Side::Left);
    CHECK(result.files[0].file == "att1_left.mcal");
    CHECK(result.files[0].bytes == rig.sdk.export_bytes.size());
    CHECK(result.files[0].sha256 == sha256_hex(rig.sdk.export_bytes));
    CHECK(rig.writer.files.at("/stage/att1_left.mcal") == rig.sdk.export_bytes);
    CHECK(rig.writer.files.size() == 1);

    CHECK(rig.all<PromptEvent>().size() == 3);
    CHECK(rig.all<ExportedEvent>().size() == 1);
    CHECK(rig.sdk.count("start:") == 1);
    CHECK(rig.sdk.count("start_step") == 3);
    CHECK(rig.sdk.count("finish") == 1);
    CHECK(rig.sdk.count("stop") == 0);
    CHECK(rig.sdk.count("export") == 1);
    CHECK(rig.engine->outcome() == Outcome::Completed);
}

TEST_CASE("a right-only calibration uses the right glove id", "[unit][engine][flow]")
{
    Rig rig;
    rig.start({ Side::Right });
    rig.run_to_end();

    require_clean_terminal(rig);
    const ResultEvent& result = the_result(rig);
    CHECK(result.outcome == Outcome::Completed);
    REQUIRE(result.files.size() == 1);
    CHECK(result.files[0].side == Side::Right);
    CHECK(result.files[0].file == "att1_right.mcal");
    for (const PromptEvent* prompt : rig.all<PromptEvent>())
    {
        CHECK(prompt->side == Side::Right);
        CHECK(prompt->glove_id == 20);
    }
    CHECK(has_call(rig, "sdk:start:20"));
    CHECK_FALSE(has_call(rig, "sdk:start:10"));
}

TEST_CASE("both gloves are calibrated left then right with one file each", "[unit][engine][flow]")
{
    Rig rig;
    rig.start({ Side::Left, Side::Right });
    rig.run_to_end();

    require_clean_terminal(rig);
    const ResultEvent& result = the_result(rig);
    CHECK(result.outcome == Outcome::Completed);
    REQUIRE(result.files.size() == 2);
    CHECK(result.files[0].side == Side::Left);
    CHECK(result.files[1].side == Side::Right);
    CHECK(rig.writer.files.count("/stage/att1_left.mcal") == 1);
    CHECK(rig.writer.files.count("/stage/att1_right.mcal") == 1);

    const auto prompts = rig.all<PromptEvent>();
    REQUIRE(prompts.size() == 6);
    for (size_t i = 0; i < prompts.size(); ++i)
    {
        CHECK(prompts[i]->side == (i < 3 ? Side::Left : Side::Right));
        CHECK(prompts[i]->glove_id == (i < 3 ? 10U : 20U));
        CHECK(prompts[i]->index == i % 3);
        CHECK(prompts[i]->count == 3);
    }

    // Export of the first glove finishes before the second glove starts calibrating.
    CHECK(rig.index_of("sdk:export:10") < rig.index_of("sdk:start:20"));
    CHECK(rig.index_of("sdk:finish:10") < rig.index_of("sdk:export:10"));
    CHECK(rig.index_of("sdk:finish:20") < rig.index_of("sdk:export:20"));
    CHECK(rig.index_of("sdk:start:10") < rig.index_of("sdk:start:20"));
}

TEST_CASE("prompts carry the SDK step data and distinguish hold from motion", "[unit][engine][flow]")
{
    Rig rig;
    rig.sdk.step_times = { 0.0f, 2.0f, 0.0f };
    rig.start({ Side::Left });
    rig.run_to_prompt();

    const PromptEvent* first = rig.last_prompt();
    REQUIRE(first != nullptr);
    CHECK(first->attempt == "att1");
    CHECK(first->title == "Step 0");
    CHECK(first->instruction == "Do pose 0");
    CHECK(first->mode == StepMode::Hold);
    CHECK(first->duration_s == 0.0);
    CHECK(first->index == 0);
    CHECK(first->count == 3);

    confirm_and_advance(rig);
    const PromptEvent* second = rig.last_prompt();
    CHECK(second->index == 1);
    CHECK(second->mode == StepMode::Motion);
    CHECK(second->duration_s == 2.0);

    confirm_and_advance(rig);
    const PromptEvent* third = rig.last_prompt();
    CHECK(third->index == 2);
    CHECK(third->mode == StepMode::Hold);
    CHECK(third->duration_s == 0.0);
}

TEST_CASE("a zero step time is a hold and a positive one is a motion with that duration", "[unit][engine][flow]")
{
    Rig rig;
    rig.sdk.steps = 4;
    rig.sdk.step_times = { 0.0f, 0.5f, 0.001f, 0.0f };
    rig.start({ Side::Left });
    rig.run_to_end();

    require_clean_terminal(rig);
    const auto prompts = rig.all<PromptEvent>();
    REQUIRE(prompts.size() == 4);
    CHECK(prompts[0]->mode == StepMode::Hold);
    CHECK(prompts[0]->duration_s == 0.0);
    CHECK(prompts[1]->mode == StepMode::Motion);
    CHECK(prompts[1]->duration_s == 0.5);
    CHECK(prompts[2]->mode == StepMode::Motion);
    CHECK(prompts[2]->duration_s == 0.001);
    CHECK(prompts[3]->mode == StepMode::Hold);
    CHECK(prompts[3]->duration_s == 0.0);
    for (const PromptEvent* prompt : prompts)
    {
        CHECK_FALSE(prompt->continuous);
        CHECK(prompt->estimated_s == 0.0);
    }
}

TEST_CASE("a step time below one millisecond is reported as one millisecond", "[unit][engine][flow]")
{
    Rig rig;
    rig.sdk.steps = 2;
    rig.sdk.step_times = { 0.0004f, -0.0004f };
    rig.start({ Side::Left });
    rig.run_to_end();

    require_clean_terminal(rig);
    const auto prompts = rig.all<PromptEvent>();
    REQUIRE(prompts.size() == 2);
    // The mode, the duration and the estimate must agree after the wire rounds to milliseconds.
    CHECK(prompts[0]->mode == StepMode::Motion);
    CHECK(prompts[0]->duration_s == 0.001);
    CHECK(prompts[1]->continuous);
    CHECK(prompts[1]->estimated_s == 0.001);
}

TEST_CASE("a negative step time is a continuous hold carrying the SDK estimate", "[unit][engine][flow][continuous]")
{
    Rig rig;
    rig.sdk.step_times = { -1.0f, -0.001f, 0.0f };
    rig.start({ Side::Left });
    rig.run_to_prompt();

    const PromptEvent* first = rig.last_prompt();
    REQUIRE(first != nullptr);
    CHECK(first->mode == StepMode::Hold);
    CHECK(first->duration_s == 0.0);
    CHECK(first->continuous);
    CHECK(first->estimated_s == 1.0);
    CHECK(rig.sdk.count("start_step") == 0);

    // Confirming starts the SDK step once, and the next prompt follows with no sampling window in between.
    rig.confirm_current();
    rig.engine->tick();
    CHECK(rig.sdk.count("start_step") == 1);
    CHECK(rig.all<StepProgressEvent>().empty());
    const auto prompts = rig.all<PromptEvent>();
    REQUIRE(prompts.size() == 2);
    CHECK(prompts[1]->index == 1);
    CHECK(prompts[1]->mode == StepMode::Hold);
    CHECK(prompts[1]->continuous);
    CHECK(prompts[1]->estimated_s == 0.001);

    // A zero-time step after continuous ones is a plain hold again.
    confirm_and_advance(rig);
    const PromptEvent* third = rig.last_prompt();
    CHECK(third->index == 2);
    CHECK_FALSE(third->continuous);
    CHECK(third->estimated_s == 0.0);
}

TEST_CASE("a routine of all continuous steps, like the real gloves report, completes with both files",
          "[unit][engine][flow][continuous]")
{
    const std::vector<float> real_times = { -2.0f, -2.0f, -10.0f, -5.0f, -3.0f, -3.0f, -3.0f, -3.0f };
    Rig rig;
    rig.sdk.steps = 8;
    rig.sdk.step_times = real_times;
    rig.start({ Side::Left, Side::Right });
    rig.run_to_end();

    require_clean_terminal(rig);
    const ResultEvent& result = the_result(rig);
    CHECK(result.outcome == Outcome::Completed);
    REQUIRE(result.files.size() == 2);
    CHECK(rig.writer.files.count("/stage/att1_left.mcal") == 1);
    CHECK(rig.writer.files.count("/stage/att1_right.mcal") == 1);

    const auto prompts = rig.all<PromptEvent>();
    REQUIRE(prompts.size() == 16);
    for (size_t i = 0; i < prompts.size(); ++i)
    {
        CHECK(prompts[i]->mode == StepMode::Hold);
        CHECK(prompts[i]->duration_s == 0.0);
        CHECK(prompts[i]->continuous);
        CHECK(prompts[i]->estimated_s == -static_cast<double>(real_times[i % 8]));
    }
    // Every step is started exactly once and no sampling window is reported.
    CHECK(rig.sdk.count("start_step") == 16);
    CHECK(rig.all<StepProgressEvent>().empty());
    CHECK(rig.all<StepAcceptedEvent>().size() == 16);
}

// --- continuous steps under window pacing -----------------------------------------------------------------

namespace
{

EngineConfig window_pacing()
{
    EngineConfig config;
    config.continuous_pacing = ContinuousPacing::Window;
    return config;
}

} // namespace

TEST_CASE("window pacing holds a continuous step for its estimate after the SDK starts it",
          "[unit][engine][flow][continuous][window]")
{
    Rig rig(window_pacing());
    rig.sdk.steps = 2;
    rig.sdk.step_times = { -2.0f, 0.0f };
    rig.start({ Side::Left });
    rig.run_to_prompt();

    const PromptEvent* first = rig.last_prompt();
    REQUIRE(first != nullptr);
    CHECK(first->continuous);
    CHECK(first->mode == StepMode::Hold);
    CHECK(first->duration_s == 0.0);
    CHECK(first->estimated_s == 2.0);

    rig.confirm_current();
    CHECK(rig.sdk.count("start_step") == 1);
    CHECK(rig.all<StepAcceptedEvent>().size() == 1);
    REQUIRE(rig.all<StepProgressEvent>().size() == 1);
    CHECK(rig.all<StepProgressEvent>()[0]->elapsed_s == 0.0);

    // No next prompt before the estimate has elapsed, and exactly at it the next prompt appears.
    rig.clock.advance(1.9);
    rig.engine->tick();
    CHECK(rig.all<PromptEvent>().size() == 1);
    rig.clock.advance(0.1);
    rig.engine->tick();
    REQUIRE(rig.all<PromptEvent>().size() == 2);
    CHECK_FALSE(rig.last_prompt()->continuous);
    CHECK(rig.all<StepProgressEvent>().back()->elapsed_s == 2.0);

    // A duplicate confirm during or after the window is acknowledged and never starts another SDK step.
    rig.send(ConfirmCommand{ "att1", first->token });
    CHECK(rig.sdk.count("start_step") == 1);
}

TEST_CASE("window pacing holds the last continuous step before finishing and exporting",
          "[unit][engine][flow][continuous][window]")
{
    Rig rig(window_pacing());
    rig.sdk.steps = 1;
    rig.sdk.step_times = { -3.0f };
    rig.start({ Side::Left });
    rig.run_to_prompt();
    rig.confirm_current();

    rig.clock.advance(2.9);
    rig.engine->tick();
    CHECK_FALSE(rig.engine->finished());
    CHECK(rig.sdk.count("finish") == 0);
    CHECK(rig.writer.files.empty());

    rig.clock.advance(0.1);
    rig.engine->tick();
    require_clean_terminal(rig);
    CHECK(the_result(rig).outcome == Outcome::Completed);
    CHECK(rig.writer.files.count("/stage/att1_left.mcal") == 1);
}

TEST_CASE("a routine of all continuous steps completes under window pacing with bounded progress",
          "[unit][engine][flow][continuous][window]")
{
    const std::vector<float> real_times = { -2.0f, -2.0f, -10.0f, -5.0f, -3.0f, -3.0f, -3.0f, -3.0f };
    Rig rig(window_pacing());
    rig.sdk.steps = 8;
    rig.sdk.step_times = real_times;
    rig.start({ Side::Left, Side::Right });
    rig.run_to_end();

    require_clean_terminal(rig);
    CHECK(the_result(rig).outcome == Outcome::Completed);
    CHECK(rig.sdk.count("start_step") == 16);
    CHECK(rig.all<StepAcceptedEvent>().size() == 16);
    const auto prompts = rig.all<PromptEvent>();
    REQUIRE(prompts.size() == 16);
    for (const StepProgressEvent* progress : rig.all<StepProgressEvent>())
    {
        // Progress never exceeds the estimate of the prompt it belongs to.
        for (const PromptEvent* prompt : prompts)
        {
            if (prompt->token == progress->token)
            {
                CHECK(progress->elapsed_s <= prompt->estimated_s);
            }
        }
    }
}

TEST_CASE("cancelling inside a window-paced continuous step ends cancelled and publishes no file",
          "[unit][engine][flow][continuous][window]")
{
    Rig rig(window_pacing());
    rig.sdk.steps = 2;
    rig.sdk.step_times = { -5.0f, -5.0f };
    rig.start({ Side::Left });
    rig.run_to_prompt();
    rig.confirm_current();
    rig.clock.advance(1.0);
    rig.engine->tick();

    rig.engine->request_cancel();
    rig.engine->tick();

    require_clean_terminal(rig);
    CHECK(the_result(rig).outcome == Outcome::Cancelled);
    CHECK(rig.writer.files.empty());
}

TEST_CASE("a glove disappearing during a window-paced continuous step fails the attempt",
          "[unit][engine][flow][continuous][window]")
{
    Rig rig(window_pacing());
    rig.sdk.steps = 1;
    rig.sdk.step_times = { -5.0f };
    rig.start({ Side::Left });
    rig.run_to_prompt();
    rig.confirm_current();

    rig.sdk.landscape = {};
    rig.clock.advance(1.0);
    rig.engine->tick();

    require_clean_terminal(rig);
    CHECK(the_result(rig).outcome == Outcome::Failed);
    CHECK(rig.writer.files.empty());
}

namespace
{

struct BadTime
{
    const char* name;
    float value;
};

const BadTime kBadTimes[] = {
    { "NaN", std::nanf("") },
    { "+inf", std::numeric_limits<float>::infinity() },
    { "-inf", -std::numeric_limits<float>::infinity() },
};

// The attempt failed with step_failed for this glove and step, with nothing offered, exported or published.
void require_step_failed(const Rig& rig, Side side, uint32_t step)
{
    require_clean_terminal(rig);
    const ResultEvent& result = the_result(rig);
    CHECK(result.outcome == Outcome::Failed);
    CHECK(result.files.empty());
    REQUIRE(result.error.has_value());
    CHECK(result.error->code == ErrorCode::StepFailed);
    REQUIRE(result.error->side.has_value());
    CHECK(*result.error->side == side);
    REQUIRE(result.error->step.has_value());
    CHECK(*result.error->step == step);
    CHECK(rig.all<StepAcceptedEvent>().empty());
}

} // namespace

TEST_CASE("an unusable step time at index 0 fails before any prompt", "[unit][engine][flow][step_time]")
{
    for (const BadTime& bad : kBadTimes)
    {
        CAPTURE(bad.name);
        Rig rig;
        rig.sdk.step_times = { bad.value, 0.0f, 0.0f };
        rig.start({ Side::Left });
        rig.run_to_end();

        require_step_failed(rig, Side::Left, 0);
        CHECK(rig.all<PromptEvent>().empty());
        CHECK(rig.sdk.count("start_step") == 0);
        CHECK(rig.sdk.count("stop") == 1);
        CHECK(rig.index_of("sdk:stop:10") < rig.index_of("sdk:shutdown"));
        CHECK(rig.all<ExportedEvent>().empty());
        CHECK(rig.writer.files.empty());
    }
}

TEST_CASE("an unusable step time at a later index fails before the first prompt", "[unit][engine][flow][step_time]")
{
    for (const BadTime& bad : kBadTimes)
    {
        CAPTURE(bad.name);
        Rig rig;
        rig.sdk.steps = 5;
        rig.sdk.step_times = { 0.0f, 2.0f, 0.0f, bad.value, 0.0f };
        rig.start({ Side::Left });
        rig.run_to_end();

        require_step_failed(rig, Side::Left, 3);
        CHECK(rig.all<PromptEvent>().empty());
        CHECK(rig.sdk.count("start_step") == 0);
        CHECK(rig.sdk.count("stop") == 1);
        CHECK(rig.index_of("sdk:stop:10") < rig.index_of("sdk:shutdown"));
        CHECK(rig.all<ExportedEvent>().empty());
        CHECK(rig.writer.files.empty());
    }
}

TEST_CASE("an unusable step time on the second glove fails only once the first glove has finished",
          "[unit][engine][flow][step_time]")
{
    for (const BadTime& bad : kBadTimes)
    {
        CAPTURE(bad.name);
        Rig rig;
        rig.sdk.glove_step_times[20] = { 0.0f, bad.value, 0.0f };
        rig.start();
        rig.run_to_end();

        // The right glove is only inspected when it is reached, after the left glove ran and exported.
        require_clean_terminal(rig);
        const ResultEvent& result = the_result(rig);
        CHECK(result.outcome == Outcome::Failed);
        CHECK(result.files.empty());
        REQUIRE(result.error.has_value());
        CHECK(result.error->code == ErrorCode::StepFailed);
        REQUIRE(result.error->side.has_value());
        CHECK(*result.error->side == Side::Right);
        REQUIRE(result.error->step.has_value());
        CHECK(*result.error->step == 1);

        const auto prompts = rig.all<PromptEvent>();
        REQUIRE(prompts.size() == 3);
        for (const PromptEvent* prompt : prompts)
        {
            CHECK(prompt->side == Side::Left);
        }
        CHECK(rig.all<StepAcceptedEvent>().size() == 3);
        CHECK(rig.sdk.count("start_step:20") == 0);
        CHECK(rig.index_of("sdk:start:20") < rig.index_of("sdk:stop:20"));
        CHECK(rig.index_of("sdk:stop:20") < rig.index_of("sdk:shutdown"));
        CHECK(rig.sdk.count("stop:20") == 1);
    }
}

TEST_CASE("every prompt has a unique token", "[unit][engine][flow]")
{
    Rig rig;
    rig.start();
    rig.run_to_end();

    std::set<std::string> tokens;
    for (const PromptEvent* prompt : rig.all<PromptEvent>())
    {
        tokens.insert(prompt->token);
    }
    CHECK(tokens.size() == rig.all<PromptEvent>().size());
}

TEST_CASE("each prompt needs its own confirmation, including the second glove's first", "[unit][engine][confirm]")
{
    Rig rig;
    rig.start();
    rig.run_to_prompt();

    for (int i = 0; i < 3; ++i)
    {
        confirm_and_advance(rig);
    }

    // The left glove is done and the right glove's first prompt is waiting.
    REQUIRE(rig.last_prompt() != nullptr);
    CHECK(rig.last_prompt()->side == Side::Right);
    CHECK(rig.last_prompt()->index == 0);
    CHECK(rig.sdk.count("start_step") == 3);
    CHECK(rig.all<StepAcceptedEvent>().size() == 3);

    // Time passing never starts a step by itself.
    tick_for(rig, 40);
    CHECK(rig.sdk.count("start_step") == 3);
    CHECK(rig.all<StepAcceptedEvent>().size() == 3);
    CHECK_FALSE(rig.engine->finished());

    rig.confirm_current();
    CHECK(rig.sdk.count("start_step") == 4);
    CHECK(has_call(rig, "sdk:start_step:20:0"));
}

// --- confirmation handling ---------------------------------------------------------------------------------

TEST_CASE("a repeated confirmation is acknowledged again without a second SDK step", "[unit][engine][confirm]")
{
    Rig rig;
    rig.start({ Side::Left });
    rig.run_to_prompt();
    const std::string token = rig.last_prompt()->token;

    rig.send(confirm(token));
    REQUIRE(rig.sdk.count("start_step") == 1);
    const size_t prompts_after_first = rig.all<PromptEvent>().size();

    // The prompt has already advanced; the duplicate still gets its acknowledgment and nothing else.
    rig.send(confirm(token));
    rig.send(confirm(token));

    size_t acks = 0;
    for (const StepAcceptedEvent* accepted : rig.all<StepAcceptedEvent>())
    {
        acks += accepted->token == token ? 1 : 0;
    }
    CHECK(acks == 3);
    CHECK(rig.sdk.count("start_step") == 1);
    CHECK(rig.all<PromptEvent>().size() == prompts_after_first);
    CHECK(rig.rejections().empty());
}

TEST_CASE("a repeated confirmation during sampling does not restart the step", "[unit][engine][confirm]")
{
    Rig rig;
    rig.start({ Side::Left });
    rig.run_to_prompt();
    confirm_and_advance(rig);
    REQUIRE(rig.last_prompt()->mode == StepMode::Motion);
    const std::string token = rig.last_prompt()->token;

    rig.confirm_current();
    REQUIRE(rig.sdk.count("start_step") == 2);
    tick_for(rig, 2);
    rig.send(confirm(token));

    CHECK(rig.sdk.count("start_step") == 2);
    CHECK_FALSE(rig.engine->finished());
    CHECK(rig.rejections().empty());
}

TEST_CASE("a token that was never offered is rejected as stale and changes nothing", "[unit][engine][confirm]")
{
    Rig rig;
    rig.start({ Side::Left });
    rig.run_to_prompt();
    const std::string offered = rig.last_prompt()->token;
    const size_t events_before = rig.events.size();

    rig.send(confirm("att1-999"));
    // The next real token is also not valid until its prompt has been offered.
    rig.send(confirm("att1-2"));

    const auto rejections = rig.rejections();
    REQUIRE(rejections.size() == 2);
    for (const RejectedEvent* rejection : rejections)
    {
        CHECK(rejection->reason == RejectReason::StaleToken);
        CHECK(rejection->command == "confirm");
        CHECK(rejection->attempt == std::optional<std::string>("att1"));
        REQUIRE(rejection->token.has_value());
    }
    CHECK(rejections[0]->token == std::optional<std::string>("att1-999"));
    CHECK(rejections[1]->token == std::optional<std::string>("att1-2"));
    CHECK(rig.events.size() == events_before + 2);
    CHECK(rig.sdk.count("start_step") == 0);
    CHECK_FALSE(rig.engine->finished());

    // The real prompt is still waiting and still works.
    CHECK(rig.last_prompt()->token == offered);
    rig.send(confirm(offered));
    CHECK(rig.sdk.count("start_step") == 1);
}

TEST_CASE("a confirmation for another attempt is rejected as an unknown attempt", "[unit][engine][confirm]")
{
    Rig rig;
    rig.start({ Side::Left });
    rig.run_to_prompt();

    rig.send(confirm(rig.last_prompt()->token, "other"));

    const auto rejections = rig.rejections();
    REQUIRE(rejections.size() == 1);
    CHECK(rejections[0]->reason == RejectReason::UnknownAttempt);
    CHECK(rejections[0]->attempt == std::optional<std::string>("other"));
    CHECK(rig.sdk.count("start_step") == 0);
}

TEST_CASE("confirmations before any prompt are rejected", "[unit][engine][confirm]")
{
    SECTION("before start")
    {
        Rig rig;
        rig.send(confirm("att1-1"));
        const auto rejections = rig.rejections();
        REQUIRE(rejections.size() == 1);
        CHECK(rejections[0]->reason == RejectReason::UnknownAttempt);
        CHECK(sdk_calls(rig).empty());
        CHECK_FALSE(rig.engine->finished());
    }
    SECTION("during discovery")
    {
        Rig rig;
        rig.sdk.landscape.clear();
        rig.start();
        rig.send(confirm("att1-1"));
        const auto rejections = rig.rejections();
        REQUIRE(rejections.size() == 1);
        CHECK(rejections[0]->reason == RejectReason::Unexpected);
        CHECK(rig.sdk.count("start_step") == 0);
        CHECK_FALSE(rig.engine->finished());
    }
}

TEST_CASE("a different token while a step is sampling is rejected and sampling continues", "[unit][engine][confirm]")
{
    Rig rig;
    rig.start({ Side::Left });
    rig.run_to_prompt();
    confirm_and_advance(rig);
    rig.confirm_current();
    REQUIRE(rig.sdk.count("start_step") == 2);

    rig.send(confirm("att1-3"));

    const auto rejections = rig.rejections();
    REQUIRE(rejections.size() == 1);
    CHECK(rejections[0]->reason == RejectReason::Unexpected);
    CHECK(rig.sdk.count("start_step") == 2);

    rig.run_to_end();
    require_clean_terminal(rig);
    CHECK(the_result(rig).outcome == Outcome::Completed);
}

TEST_CASE("a confirmation while the calibration is being saved is rejected", "[unit][engine][confirm]")
{
    Rig rig;
    rig.sdk.on_export = [&rig]() { rig.send(confirm("late-token")); };
    rig.start({ Side::Left });
    rig.run_to_end();

    require_clean_terminal(rig);
    CHECK(the_result(rig).outcome == Outcome::Completed);
    const auto rejections = rig.rejections();
    REQUIRE(rejections.size() == 1);
    CHECK(rejections[0]->reason == RejectReason::Unexpected);
    CHECK(rejections[0]->token == std::optional<std::string>("late-token"));
}

TEST_CASE("commands after the terminal result are ignored", "[unit][engine][terminal]")
{
    Rig rig;
    rig.start({ Side::Left });
    rig.run_to_end();
    require_clean_terminal(rig);

    const size_t events_before = rig.events.size();
    const size_t calls_before = sdk_calls(rig).size();
    rig.send(confirm("att1-1"));
    rig.send(CancelCommand{ "att1" });
    rig.send(StartCommand{ "att2", { Side::Left }, "/stage" });
    rig.engine->request_cancel();
    tick_for(rig, 5);

    CHECK(rig.events.size() == events_before);
    CHECK(sdk_calls(rig).size() == calls_before);
    CHECK(rig.engine->outcome() == Outcome::Completed);
}

// --- start handling ----------------------------------------------------------------------------------------

TEST_CASE("a second start while an attempt is active is rejected as busy", "[unit][engine][start]")
{
    Rig rig;
    rig.start({ Side::Left });
    rig.run_to_prompt();

    rig.send(StartCommand{ "att2", { Side::Right }, "/other" });

    const auto rejections = rig.rejections();
    REQUIRE(rejections.size() == 1);
    CHECK(rejections[0]->reason == RejectReason::Busy);
    CHECK(rejections[0]->command == "start");
    CHECK(rejections[0]->attempt == std::optional<std::string>("att2"));
    CHECK(rig.sdk.count("start:") == 1);

    // The running attempt is unaffected.
    rig.run_to_end();
    require_clean_terminal(rig);
    CHECK(the_result(rig).attempt == std::optional<std::string>("att1"));
    CHECK(the_result(rig).outcome == Outcome::Completed);
}

TEST_CASE("a start with an unusable output directory is rejected before any SDK call", "[unit][engine][start]")
{
    Rig rig;
    rig.writer.usable = false;
    rig.start();

    const auto rejections = rig.rejections();
    REQUIRE(rejections.size() == 1);
    CHECK(rejections[0]->reason == RejectReason::Invalid);
    CHECK(rejections[0]->attempt == std::optional<std::string>("att1"));
    CHECK(sdk_calls(rig).empty());
    CHECK_FALSE(rig.engine->finished());
    CHECK(rig.all<StatusEvent>().empty());

    // The engine is still idle, so a corrected start proceeds.
    rig.writer.usable = true;
    rig.start();
    rig.run_to_end();
    require_clean_terminal(rig);
    CHECK(the_result(rig).outcome == Outcome::Completed);
}

TEST_CASE("ticking and cancelling before a start does nothing harmful", "[unit][engine][start]")
{
    Rig rig;
    rig.engine->tick();
    rig.send(CancelCommand{ "att1" });

    const auto rejections = rig.rejections();
    REQUIRE(rejections.size() == 1);
    CHECK(rejections[0]->reason == RejectReason::UnknownAttempt);
    CHECK(rejections[0]->command == "cancel");
    CHECK(sdk_calls(rig).empty());
    CHECK_FALSE(rig.engine->finished());
}

TEST_CASE("a cancel requested before the start ends as cancelled without touching the SDK", "[unit][engine][start]")
{
    Rig rig;
    rig.engine->request_cancel();
    rig.start();

    REQUIRE(rig.engine->finished());
    const ResultEvent& result = the_result(rig);
    CHECK(result.outcome == Outcome::Cancelled);
    CHECK_FALSE(result.attempt.has_value());
    CHECK(result.files.empty());
    CHECK(sdk_calls(rig).empty());
    CHECK(rig.events.size() == 1);
}

// --- discovery ---------------------------------------------------------------------------------------------

TEST_CASE("discovery waits through connecting attempts", "[unit][engine][discovery]")
{
    Rig rig;
    rig.sdk.connecting_attempts = 2;
    rig.start({ Side::Left });
    rig.run_to_prompt();

    REQUIRE(rig.last_prompt() != nullptr);
    CHECK(rig.sdk.count("connect") == 3);
    rig.run_to_end();
    require_clean_terminal(rig);
    CHECK(the_result(rig).outcome == Outcome::Completed);
}

TEST_CASE("gloves that appear late are picked up and reported", "[unit][engine][discovery]")
{
    Rig rig;
    rig.sdk.landscape.clear();
    rig.start();
    tick_for(rig, 6, 0.5);
    CHECK(rig.last_prompt() == nullptr);
    CHECK_FALSE(rig.engine->finished());

    rig.sdk.landscape = { { Side::Right, 20 } };
    tick_for(rig, 4, 0.5);
    CHECK(rig.last_prompt() == nullptr);

    rig.sdk.landscape = { { Side::Left, 10 }, { Side::Right, 20 } };
    rig.run_to_prompt();
    REQUIRE(rig.last_prompt() != nullptr);
    CHECK(rig.last_prompt()->side == Side::Left);

    // Discovery progress is reported as the visible set changes, never as a repeated identical status.
    std::vector<size_t> glove_counts;
    for (const StatusEvent* status : rig.all<StatusEvent>())
    {
        if (status->phase == Phase::Discovering)
        {
            glove_counts.push_back(status->gloves.size());
        }
    }
    REQUIRE(glove_counts.size() >= 2);
    for (size_t i = 1; i < glove_counts.size(); ++i)
    {
        CHECK(glove_counts[i] != glove_counts[i - 1]);
    }
    CHECK(glove_counts.back() == 2);
}

TEST_CASE("discovery times out with no_gloves and shuts the SDK down", "[unit][engine][discovery]")
{
    Rig rig;
    rig.sdk.landscape.clear();
    rig.start();
    tick_until_finished(rig);

    require_clean_terminal(rig);
    const ResultEvent& result = the_result(rig);
    CHECK(result.outcome == Outcome::Failed);
    REQUIRE(result.error.has_value());
    CHECK(result.error->code == ErrorCode::NoGloves);
    CHECK_THAT(result.error->message, ContainsSubstring("left, right"));
    CHECK(result.files.empty());
    CHECK(rig.clock.now - 100.0 >= rig.config.discovery_timeout_s);
    CHECK(rig.sdk.count("start:") == 0);
    CHECK(rig.sdk.count("stop") == 0);
}

TEST_CASE("the timeout names only the gloves that were not found", "[unit][engine][discovery]")
{
    Rig rig;
    rig.sdk.landscape = { { Side::Left, 10 } };
    rig.start({ Side::Left, Side::Right });
    tick_until_finished(rig);

    require_clean_terminal(rig);
    const ResultEvent& result = the_result(rig);
    REQUIRE(result.error.has_value());
    CHECK(result.error->code == ErrorCode::NoGloves);
    CHECK_THAT(result.error->message, ContainsSubstring("right"));
    CHECK_THAT(result.error->message, !ContainsSubstring("left"));
    // Nothing is calibrated when only some of the requested gloves are present.
    CHECK(rig.sdk.count("start:") == 0);
}

TEST_CASE("a failed connection ends the attempt with the host's message", "[unit][engine][discovery]")
{
    Rig rig;
    rig.sdk.fail_call = "connect";
    rig.start();
    tick_until_finished(rig);

    require_clean_terminal(rig);
    const ResultEvent& result = the_result(rig);
    CHECK(result.outcome == Outcome::Failed);
    REQUIRE(result.error.has_value());
    CHECK(result.error->code == ErrorCode::SdkInitFailed);
    CHECK_THAT(result.error->message, ContainsSubstring("injected"));
}

// --- progress ----------------------------------------------------------------------------------------------

TEST_CASE("sampling progress is monotonic and never exceeds the step duration", "[unit][engine][progress]")
{
    Rig rig;
    rig.start({ Side::Left });
    rig.run_to_prompt();
    confirm_and_advance(rig);

    const PromptEvent* motion = rig.last_prompt();
    REQUIRE(motion->mode == StepMode::Motion);
    const std::string token = motion->token;
    const double duration = motion->duration_s;
    const size_t prompts_before = rig.all<PromptEvent>().size();
    rig.confirm_current();

    // Mid-window: progress is flowing and no further prompt has been offered.
    tick_for(rig, 5);
    CHECK(rig.all<PromptEvent>().size() == prompts_before);

    tick_for(rig, 10);

    std::vector<double> elapsed;
    for (const StepProgressEvent* progress : rig.all<StepProgressEvent>())
    {
        if (progress->token == token)
        {
            elapsed.push_back(progress->elapsed_s);
        }
    }
    REQUIRE(elapsed.size() >= 3);
    CHECK(elapsed.front() == 0.0);
    CHECK(elapsed.back() == duration);
    for (size_t i = 0; i < elapsed.size(); ++i)
    {
        CHECK(elapsed[i] >= 0.0);
        CHECK(elapsed[i] <= duration);
        if (i > 0)
        {
            CHECK(elapsed[i] >= elapsed[i - 1]);
        }
    }
    // The next prompt appears only once the window is over.
    CHECK(rig.all<PromptEvent>().size() == prompts_before + 1);
    CHECK(rig.last_prompt()->index == 2);
}

TEST_CASE("a long gap between ticks still reports progress capped at the duration", "[unit][engine][progress]")
{
    Rig rig;
    rig.start({ Side::Left });
    rig.run_to_prompt();
    confirm_and_advance(rig);
    const std::string token = rig.last_prompt()->token;
    rig.confirm_current();

    rig.clock.advance(50.0);
    rig.engine->tick();

    const auto progress = rig.all<StepProgressEvent>();
    REQUIRE_FALSE(progress.empty());
    CHECK(progress.back()->token == token);
    CHECK(progress.back()->elapsed_s == 2.0);
    CHECK(rig.last_prompt()->index == 2);
}

TEST_CASE("progress events appear only for motion steps", "[unit][engine][progress]")
{
    Rig rig;
    rig.sdk.step_times = { 0.0f, 0.0f, 0.0f };
    rig.start({ Side::Left });
    rig.run_to_end();

    require_clean_terminal(rig);
    CHECK(rig.all<StepProgressEvent>().empty());
}

// --- glove presence ----------------------------------------------------------------------------------------

TEST_CASE("the other glove disappearing while a prompt is waiting fails the attempt", "[unit][engine][gloves]")
{
    Rig rig;
    rig.start();
    rig.run_to_prompt();
    REQUIRE(rig.last_prompt()->side == Side::Left);

    rig.sdk.landscape = { { Side::Left, 10 } };
    tick_for(rig, 1);

    require_clean_terminal(rig);
    const ResultEvent& result = the_result(rig);
    CHECK(result.outcome == Outcome::Failed);
    REQUIRE(result.error.has_value());
    CHECK(result.error->code == ErrorCode::GloveMissing);
    CHECK(result.error->side == std::optional<Side>(Side::Right));
    CHECK(result.files.empty());
    // The glove that was being calibrated is told to stop before the SDK goes down.
    CHECK(has_call(rig, "sdk:stop:10"));
    CHECK(rig.index_of("sdk:stop:10") < rig.index_of("sdk:shutdown"));
}

TEST_CASE("the calibrated glove disappearing during sampling fails with the step index", "[unit][engine][gloves]")
{
    Rig rig;
    rig.start({ Side::Left });
    rig.run_to_prompt();
    confirm_and_advance(rig);
    rig.confirm_current();

    rig.sdk.landscape = { { Side::Right, 20 } };
    tick_for(rig, 1);

    require_clean_terminal(rig);
    const ResultEvent& result = the_result(rig);
    REQUIRE(result.error.has_value());
    CHECK(result.error->code == ErrorCode::GloveMissing);
    CHECK(result.error->side == std::optional<Side>(Side::Left));
    CHECK(result.error->step == std::optional<uint32_t>(1));
    CHECK(has_call(rig, "sdk:stop:10"));
}

TEST_CASE("a changed glove id is reported as a replaced glove", "[unit][engine][gloves]")
{
    Rig rig;
    rig.start({ Side::Left });
    rig.run_to_prompt();

    rig.sdk.landscape = { { Side::Left, 11 }, { Side::Right, 20 } };
    tick_for(rig, 1);

    require_clean_terminal(rig);
    const ResultEvent& result = the_result(rig);
    CHECK(result.outcome == Outcome::Failed);
    REQUIRE(result.error.has_value());
    CHECK(result.error->code == ErrorCode::GloveReplaced);
    CHECK(result.error->side == std::optional<Side>(Side::Left));
    CHECK_THAT(result.error->message, ContainsSubstring("10"));
    CHECK_THAT(result.error->message, ContainsSubstring("11"));
}

TEST_CASE("a glove replaced during sampling fails the attempt", "[unit][engine][gloves]")
{
    Rig rig;
    rig.start({ Side::Left });
    rig.run_to_prompt();
    confirm_and_advance(rig);
    rig.confirm_current();

    rig.sdk.landscape = { { Side::Left, 99 }, { Side::Right, 20 } };
    tick_for(rig, 1);

    require_clean_terminal(rig);
    REQUIRE(the_result(rig).error.has_value());
    CHECK(the_result(rig).error->code == ErrorCode::GloveReplaced);
    CHECK(the_result(rig).error->step == std::optional<uint32_t>(1));
}

TEST_CASE("a glove that vanishes before its confirmation is processed never starts the step", "[unit][engine][gloves]")
{
    Rig rig;
    rig.start({ Side::Left });
    rig.run_to_prompt();
    const std::string token = rig.last_prompt()->token;

    rig.sdk.landscape.clear();
    rig.send(confirm(token));

    require_clean_terminal(rig);
    CHECK(rig.sdk.count("start_step") == 0);
    REQUIRE(the_result(rig).error.has_value());
    CHECK(the_result(rig).error->code == ErrorCode::GloveMissing);
}

TEST_CASE("the second glove vanishing mid-attempt keeps the first export out of the result", "[unit][engine][gloves]")
{
    Rig rig;
    rig.start();
    rig.run_to_prompt();
    for (int i = 0; i < 3; ++i)
    {
        confirm_and_advance(rig);
    }
    REQUIRE(rig.all<ExportedEvent>().size() == 1);

    rig.sdk.landscape = { { Side::Left, 10 } };
    tick_for(rig, 1);

    require_clean_terminal(rig);
    const ResultEvent& result = the_result(rig);
    CHECK(result.outcome == Outcome::Failed);
    CHECK(result.files.empty());
    CHECK(serialize_event(result).find("files") == std::string::npos);
}

// --- SDK failure injection ---------------------------------------------------------------------------------

namespace
{

struct Injection
{
    const char* call;
    ErrorCode code;
    bool names_glove;
    int stops;
};

} // namespace

TEST_CASE("an SDK failure at each call ends the attempt as failed", "[unit][engine][failure]")
{
    // The engine receives the SDK return code and operation result already folded into one status, so a
    // failing return code and a false result boolean look identical here.
    const std::vector<Injection> injections = {
        { "initialize", ErrorCode::SdkInitFailed, false, 0 },
        { "connect", ErrorCode::SdkInitFailed, false, 0 },
        { "step_count", ErrorCode::SdkError, true, 0 },
        { "step_data", ErrorCode::SdkError, true, 1 },
        { "start", ErrorCode::SdkError, true, 0 },
        { "start_step", ErrorCode::StepFailed, true, 1 },
        { "finish", ErrorCode::SdkError, true, 1 },
        { "export", ErrorCode::ExportFailed, true, 0 },
    };
    for (const Injection& injection : injections)
    {
        DYNAMIC_SECTION("failure at " << injection.call)
        {
            Rig rig;
            rig.sdk.fail_call = injection.call;
            rig.start();
            rig.run_to_end();
            tick_until_finished(rig);

            require_clean_terminal(rig);
            const ResultEvent& result = the_result(rig);
            CHECK(result.outcome == Outcome::Failed);
            CHECK(result.files.empty());
            REQUIRE(result.error.has_value());
            CHECK(result.error->code == injection.code);
            CHECK(result.error->side.has_value() == injection.names_glove);
            CHECK_THAT(result.error->message, ContainsSubstring("injected"));
            // A calibration that was running is stopped; one that was never started is not.
            CHECK(rig.sdk.count("stop") == injection.stops);
            CHECK(rig.engine->outcome() == Outcome::Failed);
            // Nothing is published for a failed attempt, so no exported file is ever advertised.
            CHECK(serialize_event(result).find("\"files\"") == std::string::npos);
        }
    }
}

TEST_CASE("a failure on the second step names that step", "[unit][engine][failure]")
{
    Rig rig;
    rig.sdk.fail_call = "step_data";
    rig.sdk.fail_nth = 2;
    rig.start({ Side::Left });
    rig.run_to_end();

    require_clean_terminal(rig);
    REQUIRE(the_result(rig).error.has_value());
    CHECK(the_result(rig).error->code == ErrorCode::SdkError);
    CHECK(the_result(rig).error->step == std::optional<uint32_t>(1));
    CHECK(rig.sdk.count("stop") == 1);
}

TEST_CASE("a start_step failure never reports the step as accepted", "[unit][engine][failure]")
{
    Rig rig;
    rig.sdk.fail_call = "start_step";
    rig.start({ Side::Left });
    rig.run_to_prompt();
    const std::string token = rig.last_prompt()->token;
    rig.send(confirm(token));

    require_clean_terminal(rig);
    CHECK(rig.all<StepAcceptedEvent>().empty());
    CHECK(the_result(rig).error->code == ErrorCode::StepFailed);

    // A retry of the same token after the failure gets nothing.
    const size_t events = rig.events.size();
    rig.send(confirm(token));
    CHECK(rig.events.size() == events);
}

TEST_CASE("a failed export of the second glove fails the whole attempt", "[unit][engine][failure]")
{
    Rig rig;
    rig.sdk.fail_call = "export";
    rig.sdk.fail_nth = 2;
    rig.start();
    rig.run_to_end();

    require_clean_terminal(rig);
    const ResultEvent& result = the_result(rig);
    CHECK(result.outcome == Outcome::Failed);
    REQUIRE(result.error.has_value());
    CHECK(result.error->code == ErrorCode::ExportFailed);
    CHECK(result.error->side == std::optional<Side>(Side::Right));
    // The first glove's file was written and announced, but a failed attempt lists no files.
    CHECK(rig.all<ExportedEvent>().size() == 1);
    CHECK(result.files.empty());
    CHECK(rig.sdk.count("stop") == 0);
}

TEST_CASE("a calibration with no steps fails before starting the SDK routine", "[unit][engine][failure]")
{
    Rig rig;
    rig.sdk.steps = 0;
    rig.start();
    rig.run_to_end();

    require_clean_terminal(rig);
    const ResultEvent& result = the_result(rig);
    CHECK(result.outcome == Outcome::Failed);
    REQUIRE(result.error.has_value());
    CHECK(result.error->code == ErrorCode::SdkError);
    CHECK(result.error->side == std::optional<Side>(Side::Left));
    CHECK_THAT(result.error->message, ContainsSubstring("no calibration steps"));
    CHECK(rig.sdk.count("start:") == 0);
    CHECK(rig.sdk.count("stop") == 0);
    CHECK(rig.all<PromptEvent>().empty());
}

TEST_CASE("an empty export is rejected and nothing is written", "[unit][engine][failure]")
{
    Rig rig;
    rig.sdk.export_bytes.clear();
    rig.start();
    rig.run_to_end();

    require_clean_terminal(rig);
    const ResultEvent& result = the_result(rig);
    CHECK(result.outcome == Outcome::Failed);
    REQUIRE(result.error.has_value());
    CHECK(result.error->code == ErrorCode::ExportFailed);
    CHECK_THAT(result.error->message, ContainsSubstring("empty"));
    CHECK(rig.writer.files.empty());
    CHECK(rig.all<ExportedEvent>().empty());
    CHECK(result.files.empty());
}

TEST_CASE("an export whose size and bytes disagree is rejected and nothing is written", "[unit][engine][failure]")
{
    // The size/bytes comparison happens in the SDK binding, which reports a mismatch as a failed export even
    // though the byte buffer was partly filled. The engine must not write that buffer.
    Rig rig;
    rig.sdk.fail_call = "export";
    rig.sdk.export_bytes = { 1, 2, 3 };
    rig.start({ Side::Left });
    rig.run_to_end();

    require_clean_terminal(rig);
    CHECK(the_result(rig).error->code == ErrorCode::ExportFailed);
    CHECK(rig.writer.files.empty());
    CHECK(rig.all<ExportedEvent>().empty());
}

TEST_CASE("a write failure fails the attempt and nothing is announced", "[unit][engine][failure]")
{
    Rig rig;
    rig.writer.fail_with = "disk full";
    rig.start({ Side::Left });
    rig.run_to_end();

    require_clean_terminal(rig);
    const ResultEvent& result = the_result(rig);
    CHECK(result.outcome == Outcome::Failed);
    REQUIRE(result.error.has_value());
    CHECK(result.error->code == ErrorCode::ExportFailed);
    CHECK(result.error->side == std::optional<Side>(Side::Left));
    CHECK_THAT(result.error->message, ContainsSubstring("disk full"));
    CHECK(rig.all<ExportedEvent>().empty());
    CHECK(result.files.empty());
    // The glove's calibration had already finished, so there is nothing to stop.
    CHECK(rig.sdk.count("stop") == 0);
}

TEST_CASE("a write failure on the second glove leaves no files in the result", "[unit][engine][failure]")
{
    Rig rig;
    rig.sdk.on_export = [&rig]()
    {
        if (rig.sdk.count("export") >= 1)
        {
            rig.writer.fail_with = "read-only filesystem";
        }
    };
    rig.start();
    rig.run_to_end();

    require_clean_terminal(rig);
    const ResultEvent& result = the_result(rig);
    CHECK(result.outcome == Outcome::Failed);
    REQUIRE(result.error.has_value());
    CHECK(result.error->side == std::optional<Side>(Side::Right));
    CHECK(rig.all<ExportedEvent>().size() == 1);
    CHECK(result.files.empty());
}

// --- cancellation ------------------------------------------------------------------------------------------

TEST_CASE("cancelling during discovery ends cancelled with the SDK shut down", "[unit][engine][cancel]")
{
    SECTION("through the thread-safe request")
    {
        Rig rig;
        rig.sdk.landscape.clear();
        rig.start();
        tick_for(rig, 2, 0.5);
        REQUIRE_FALSE(rig.engine->finished());

        rig.engine->request_cancel();
        tick_for(rig, 1, 0.5);

        require_clean_terminal(rig);
        CHECK(the_result(rig).outcome == Outcome::Cancelled);
        CHECK(the_result(rig).files.empty());
        CHECK_FALSE(the_result(rig).error.has_value());
        CHECK(rig.sdk.count("stop") == 0);
        CHECK(rig.sdk.count("start:") == 0);
    }
    SECTION("through a cancel command")
    {
        Rig rig;
        rig.sdk.landscape.clear();
        rig.start();
        rig.send(CancelCommand{ "att1" });

        require_clean_terminal(rig);
        CHECK(the_result(rig).outcome == Outcome::Cancelled);
        CHECK(rig.sdk.count("stop") == 0);
    }
}

TEST_CASE("cancelling while a prompt awaits confirmation stops the glove calibration", "[unit][engine][cancel]")
{
    Rig rig;
    rig.start();
    rig.run_to_prompt();
    const std::string token = rig.last_prompt()->token;

    rig.send(CancelCommand{ "att1" });

    require_clean_terminal(rig);
    const ResultEvent& result = the_result(rig);
    CHECK(result.outcome == Outcome::Cancelled);
    CHECK(result.files.empty());
    CHECK_FALSE(result.error.has_value());
    CHECK(rig.sdk.count("stop") == 1);
    CHECK(has_call(rig, "sdk:stop:10"));
    CHECK(rig.index_of("sdk:stop:10") < rig.index_of("sdk:shutdown"));
    CHECK(rig.engine->outcome() == Outcome::Cancelled);

    // The prompt that was waiting can no longer start a step.
    const size_t events = rig.events.size();
    rig.send(confirm(token));
    CHECK(rig.sdk.count("start_step") == 0);
    CHECK(rig.events.size() == events);
}

TEST_CASE("cancelling during sampling stops the step and offers nothing more", "[unit][engine][cancel]")
{
    Rig rig;
    rig.start({ Side::Left });
    rig.run_to_prompt();
    confirm_and_advance(rig);
    rig.confirm_current();
    tick_for(rig, 2);
    const size_t prompts = rig.all<PromptEvent>().size();
    const size_t progress = rig.all<StepProgressEvent>().size();

    rig.engine->request_cancel();
    tick_for(rig, 1);

    require_clean_terminal(rig);
    CHECK(the_result(rig).outcome == Outcome::Cancelled);
    CHECK(rig.sdk.count("stop") == 1);
    CHECK(rig.all<PromptEvent>().size() == prompts);
    CHECK(rig.all<StepProgressEvent>().size() == progress);
    CHECK(rig.sdk.count("export") == 0);

    // Further time has no effect once finished.
    tick_for(rig, 10);
    CHECK(rig.all<StepProgressEvent>().size() == progress);
}

TEST_CASE("cancelling while saving writes nothing and reports cancelled", "[unit][engine][cancel]")
{
    Rig rig;
    rig.sdk.on_export = [&rig]() { rig.engine->request_cancel(); };
    rig.start();
    rig.run_to_end();

    require_clean_terminal(rig);
    const ResultEvent& result = the_result(rig);
    CHECK(result.outcome == Outcome::Cancelled);
    CHECK(result.files.empty());
    CHECK(rig.writer.files.empty());
    CHECK(rig.all<ExportedEvent>().empty());
    // The glove's calibration had already finished, so only shutdown is needed.
    CHECK(rig.sdk.count("stop") == 0);
    CHECK(rig.sdk.count("export") == 1);
}

TEST_CASE("cancelling after the first glove was exported leaves no files in the result", "[unit][engine][cancel]")
{
    Rig rig;
    rig.start();
    rig.run_to_prompt();
    for (int i = 0; i < 3; ++i)
    {
        confirm_and_advance(rig);
    }
    REQUIRE(rig.all<ExportedEvent>().size() == 1);
    REQUIRE(rig.last_prompt()->side == Side::Right);

    rig.send(CancelCommand{ "att1" });

    require_clean_terminal(rig);
    const ResultEvent& result = the_result(rig);
    CHECK(result.outcome == Outcome::Cancelled);
    CHECK(result.files.empty());
    CHECK(has_call(rig, "sdk:stop:20"));
    CHECK_FALSE(has_call(rig, "sdk:stop:10"));
    CHECK(serialize_event(result).find("files") == std::string::npos);
}

TEST_CASE("a cancel for another attempt is rejected and the attempt continues", "[unit][engine][cancel]")
{
    Rig rig;
    rig.start({ Side::Left });
    rig.run_to_prompt();

    rig.send(CancelCommand{ "other" });

    const auto rejections = rig.rejections();
    REQUIRE(rejections.size() == 1);
    CHECK(rejections[0]->reason == RejectReason::UnknownAttempt);
    CHECK(rejections[0]->command == "cancel");
    CHECK_FALSE(rig.engine->finished());

    rig.run_to_end();
    require_clean_terminal(rig);
    CHECK(the_result(rig).outcome == Outcome::Completed);
}

TEST_CASE("cancelling twice yields one terminal result", "[unit][engine][cancel]")
{
    Rig rig;
    rig.start();
    rig.run_to_prompt();

    rig.send(CancelCommand{ "att1" });
    rig.send(CancelCommand{ "att1" });
    rig.engine->request_cancel();
    tick_for(rig, 3);

    require_clean_terminal(rig);
    CHECK(rig.sdk.count("stop") == 1);
    CHECK(rig.sdk.count("shutdown") == 1);
}

TEST_CASE("a cancel after completion does not change the outcome", "[unit][engine][cancel]")
{
    Rig rig;
    rig.start({ Side::Left });
    rig.run_to_end();
    require_clean_terminal(rig);

    rig.send(CancelCommand{ "att1" });
    rig.engine->request_cancel();
    rig.engine->tick();

    CHECK(rig.engine->outcome() == Outcome::Completed);
    CHECK(the_result(rig).outcome == Outcome::Completed);
    CHECK(rig.all<ResultEvent>().size() == 1);
}

TEST_CASE("a calibration that cannot be stopped is not reported as cleanly cancelled", "[unit][engine][cancel]")
{
    Rig rig;
    rig.sdk.fail_call = "stop";
    rig.start({ Side::Left });
    rig.run_to_prompt();

    rig.send(CancelCommand{ "att1" });

    require_clean_terminal(rig);
    const ResultEvent& result = the_result(rig);
    CHECK(result.outcome == Outcome::Failed);
    REQUIRE(result.error.has_value());
    CHECK(result.error->code == ErrorCode::SdkError);
    CHECK_THAT(result.error->message, ContainsSubstring("cannot stop"));
    CHECK(result.files.empty());
    // The SDK is still shut down even though the stop failed.
    CHECK(rig.sdk.count("shutdown") == 1);
}

TEST_CASE("a failed stop during a failure keeps the original error", "[unit][engine][cancel]")
{
    Rig rig;
    rig.sdk.fail_call = "stop";
    rig.start({ Side::Left });
    rig.run_to_prompt();

    rig.sdk.landscape.clear();
    tick_for(rig, 1);

    require_clean_terminal(rig);
    REQUIRE(the_result(rig).error.has_value());
    CHECK(the_result(rig).error->code == ErrorCode::GloveMissing);
    CHECK(the_result(rig).outcome == Outcome::Failed);
}

// --- terminal ordering -------------------------------------------------------------------------------------

TEST_CASE("every terminal path announces stopping, shuts down, then sends the result", "[unit][engine][terminal]")
{
    SECTION("completed")
    {
        Rig rig;
        rig.start({ Side::Left });
        rig.run_to_end();
        require_clean_terminal(rig);
        CHECK(rig.index_of("event:exported") < rig.index_of("sdk:shutdown"));
    }
    SECTION("failed")
    {
        Rig rig;
        rig.sdk.fail_call = "start";
        rig.start({ Side::Left });
        rig.run_to_end();
        require_clean_terminal(rig);
    }
    SECTION("cancelled")
    {
        Rig rig;
        rig.start({ Side::Left });
        rig.run_to_prompt();
        rig.send(CancelCommand{ "att1" });
        require_clean_terminal(rig);
    }
}

TEST_CASE("the last status before the result is stopping", "[unit][engine][terminal]")
{
    Rig rig;
    rig.start({ Side::Left });
    rig.run_to_prompt();
    rig.send(CancelCommand{ "att1" });

    require_clean_terminal(rig);
    const auto statuses = rig.all<StatusEvent>();
    REQUIRE_FALSE(statuses.empty());
    CHECK(statuses.back()->phase == Phase::Stopping);
    CHECK(statuses.back()->attempt == "att1");
}

TEST_CASE("status reports each phase of a completed attempt in order", "[unit][engine][terminal]")
{
    Rig rig;
    rig.start({ Side::Left });
    rig.run_to_end();
    require_clean_terminal(rig);

    std::vector<Phase> phases;
    for (const StatusEvent* status : rig.all<StatusEvent>())
    {
        if (phases.empty() || phases.back() != status->phase)
        {
            phases.push_back(status->phase);
        }
    }
    CHECK(phases.front() == Phase::Discovering);
    CHECK(phases.back() == Phase::Stopping);
    CHECK(std::find(phases.begin(), phases.end(), Phase::AwaitingStep) != phases.end());
    CHECK(std::find(phases.begin(), phases.end(), Phase::CapturingStep) != phases.end());
    CHECK(std::find(phases.begin(), phases.end(), Phase::Saving) != phases.end());
}

TEST_CASE("a completed result lists exactly the exported files with matching hashes", "[unit][engine][terminal]")
{
    Rig rig;
    rig.start();
    rig.run_to_end();
    require_clean_terminal(rig);

    const ResultEvent& result = the_result(rig);
    const auto exported = rig.all<ExportedEvent>();
    REQUIRE(result.files.size() == exported.size());
    for (size_t i = 0; i < exported.size(); ++i)
    {
        CHECK(result.files[i].side == exported[i]->side);
        CHECK(result.files[i].file == exported[i]->file);
        CHECK(result.files[i].bytes == exported[i]->bytes);
        CHECK(result.files[i].sha256 == exported[i]->sha256);
        CHECK(rig.writer.files.count(std::string("/stage/") + result.files[i].file) == 1);
    }
}

// --- cleanup evidence --------------------------------------------------------------------------------------

TEST_CASE("clean runs report the gloves released", "[unit][engine][released]")
{
    SECTION("a completed calibration")
    {
        Rig rig;
        rig.start({ Side::Left });
        rig.run_to_end();
        require_clean_terminal(rig);
        CHECK(the_result(rig).outcome == Outcome::Completed);
        CHECK(the_result(rig).released);
    }
    SECTION("a cancel at the first prompt")
    {
        Rig rig;
        rig.start({ Side::Left });
        rig.run_to_prompt();
        rig.send(CancelCommand{ "att1" });
        require_clean_terminal(rig);
        CHECK(the_result(rig).outcome == Outcome::Cancelled);
        CHECK(the_result(rig).released);
    }
    SECTION("a failure whose cleanup succeeded")
    {
        Rig rig;
        rig.start({ Side::Left });
        rig.run_to_prompt();
        rig.sdk.landscape.clear();
        tick_for(rig, 1);
        require_clean_terminal(rig);
        CHECK(the_result(rig).outcome == Outcome::Failed);
        CHECK(the_result(rig).released);
    }
    SECTION("a discovery timeout")
    {
        Rig rig;
        rig.sdk.landscape.clear();
        rig.start({ Side::Left });
        tick_until_finished(rig);
        require_clean_terminal(rig);
        CHECK(the_result(rig).outcome == Outcome::Failed);
        CHECK(the_result(rig).released);
    }
}

TEST_CASE("a cancel before the start never touched the SDK and so reports it released", "[unit][engine][released]")
{
    Rig rig;
    rig.engine->request_cancel();
    rig.start();

    REQUIRE(rig.engine->finished());
    CHECK(sdk_calls(rig).empty());
    CHECK(the_result(rig).released);
}

TEST_CASE("a failed SDK shutdown after a would-be completed run fails the attempt and reports not released",
          "[unit][engine][released]")
{
    Rig rig;
    rig.sdk.shutdown_result = SdkStatus::failure("injected shutdown");
    rig.start({ Side::Left });
    rig.run_to_end();

    require_clean_terminal(rig);
    const ResultEvent& result = the_result(rig);
    CHECK(result.outcome == Outcome::Failed);
    CHECK(result.files.empty());
    CHECK_FALSE(result.released);
    REQUIRE(result.error.has_value());
    CHECK(result.error->code == ErrorCode::SdkError);
    CHECK_THAT(result.error->message, ContainsSubstring("cannot shut down the SDK"));
    CHECK_THAT(result.error->message, ContainsSubstring("injected shutdown"));
    CHECK_THAT(serialize_event(result), ContainsSubstring("\"released\":false"));
    CHECK(rig.engine->outcome() == Outcome::Failed);
}

TEST_CASE("a failed SDK shutdown after a cancel downgrades it to failed and reports not released",
          "[unit][engine][released]")
{
    Rig rig;
    rig.sdk.shutdown_result = SdkStatus::failure("injected shutdown");
    rig.start({ Side::Left });
    rig.run_to_prompt();

    rig.send(CancelCommand{ "att1" });

    require_clean_terminal(rig);
    const ResultEvent& result = the_result(rig);
    CHECK(result.outcome == Outcome::Failed);
    CHECK_FALSE(result.released);
    REQUIRE(result.error.has_value());
    CHECK_THAT(result.error->message, ContainsSubstring("cannot shut down the SDK"));
}

TEST_CASE("a failed SDK shutdown during an earlier failure keeps the original error and reports not released",
          "[unit][engine][released]")
{
    Rig rig;
    rig.sdk.shutdown_result = SdkStatus::failure("injected shutdown");
    rig.start({ Side::Left });
    rig.run_to_prompt();

    rig.sdk.landscape.clear();
    tick_for(rig, 1);

    require_clean_terminal(rig);
    const ResultEvent& result = the_result(rig);
    CHECK(result.outcome == Outcome::Failed);
    CHECK_FALSE(result.released);
    REQUIRE(result.error.has_value());
    // The attempt's own failure stays the reported cause; the release failure is carried by the flag.
    CHECK(result.error->code == ErrorCode::GloveMissing);
}

TEST_CASE("a failed calibration stop during a cancel reports not released", "[unit][engine][released]")
{
    Rig rig;
    rig.sdk.fail_call = "stop";
    rig.start({ Side::Left });
    rig.run_to_prompt();

    rig.send(CancelCommand{ "att1" });

    require_clean_terminal(rig);
    const ResultEvent& result = the_result(rig);
    CHECK(result.outcome == Outcome::Failed);
    CHECK_FALSE(result.released);
    REQUIRE(result.error.has_value());
    CHECK_THAT(result.error->message, ContainsSubstring("cannot stop"));
}

TEST_CASE("a failed calibration stop during an earlier failure keeps the original error and reports not released",
          "[unit][engine][released]")
{
    Rig rig;
    rig.sdk.fail_call = "stop";
    rig.start({ Side::Left });
    rig.run_to_prompt();

    rig.sdk.landscape.clear();
    tick_for(rig, 1);

    require_clean_terminal(rig);
    const ResultEvent& result = the_result(rig);
    CHECK(result.outcome == Outcome::Failed);
    CHECK_FALSE(result.released);
    REQUIRE(result.error.has_value());
    CHECK(result.error->code == ErrorCode::GloveMissing);
}
