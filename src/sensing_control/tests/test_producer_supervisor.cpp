// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>
#include <sensing_control/producer_supervisor.hpp>

#include <atomic>
#include <chrono>
#include <functional>
#include <future>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <vector>

using namespace plugins::sensing;
using namespace std::chrono_literals;
using Json = nlohmann::json;

namespace
{

/// Everything the fake factory and its sessions record, shared with the test thread.
struct World
{
    std::mutex mutex;
    int factory_calls = 0;
    int fail_factory_until_call = 0; ///< calls up to and including this number throw
    std::string factory_failure = "INvSIPLCamera::Init failed with SIPLStatus 10";
    std::function<void(World&)> on_factory; ///< may throw; runs under no lock
    std::vector<DirectTargets> targets_seen;
    int sessions_created = 0;
    int update_calls = 0;
    int throw_update_after = -1; ///< a session's update() throws on its N+1th call; -1 never
    int flushes = 0;
    std::vector<std::tuple<std::string, std::string, std::uint16_t>> retargets;
    std::vector<std::thread::id> retarget_threads; ///< the thread each retarget was called on
    std::vector<std::string> sequence; ///< "update" and "retarget:<track>:<host>:<port>" in call order
    std::vector<std::string> published;
    std::string update_failure = "room lost"; ///< the text a throwing update() carries
    std::uint64_t frames = 5; ///< what every stream reports
    bool advance_frames = false; ///< every streams() call adds a frame
    std::map<std::string, std::string> direct_errors; ///< per track
};

class FakeSession : public CameraSession
{
public:
    FakeSession(World& world, DirectTargets targets) : m_world(world), m_targets(std::move(targets))
    {
    }
    void update() override
    {
        std::lock_guard<std::mutex> lock(m_world.mutex);
        ++m_world.update_calls;
        m_world.sequence.emplace_back("update");
        if (m_world.throw_update_after >= 0 && ++m_updates > m_world.throw_update_after)
            throw std::runtime_error(m_world.update_failure);
    }
    void flush() override
    {
        std::lock_guard<std::mutex> lock(m_world.mutex);
        ++m_world.flushes;
    }
    std::vector<StreamStatus> streams() const override
    {
        std::lock_guard<std::mutex> lock(m_world.mutex);
        std::vector<StreamStatus> out;
        for (const auto& target : m_targets)
        {
            StreamStatus s;
            s.track = target.first;
            if (m_world.advance_frames)
                ++m_world.frames;
            s.frames = m_world.frames;
            const auto error = m_world.direct_errors.find(target.first);
            if (error != m_world.direct_errors.end())
                s.direct_error = error->second;
            // The session's own view of where it sends, as the real producer reports it.
            s.direct_host = target.second.host;
            s.direct_port = target.second.port;
            out.push_back(s);
        }
        return out;
    }
    void set_direct_target(const std::string& track, const std::string& host, std::uint16_t port) override
    {
        std::lock_guard<std::mutex> lock(m_world.mutex);
        m_world.retargets.emplace_back(track, host, port);
        m_world.retarget_threads.push_back(std::this_thread::get_id());
        m_world.sequence.push_back("retarget:" + track + ":" + host + ":" + std::to_string(port));
        m_targets[track] = { host, port };
    }

private:
    World& m_world;
    DirectTargets m_targets;
    int m_updates = 0;
};

struct Harness
{
    explicit Harness(std::vector<StreamStatus> configured = default_streams(),
                     bool with_publish = true,
                     std::chrono::milliseconds stall_timeout = 0ms)
        : status("rel")
    {
        SupervisorConfig config;
        config.stall_timeout = stall_timeout;
        config.retry_interval = 100ms;
        config.poll_interval = 2ms;
        config.status_interval = 20ms;
        supervisor = std::make_unique<ProducerSupervisor>(
            status, std::move(configured),
            [this](const DirectTargets& targets) -> std::unique_ptr<CameraSession>
            {
                std::function<void(World&)> hook;
                {
                    std::lock_guard<std::mutex> lock(world.mutex);
                    ++world.factory_calls;
                    world.targets_seen.push_back(targets);
                    hook = world.on_factory;
                    if (world.factory_calls <= world.fail_factory_until_call)
                        throw std::runtime_error(world.factory_failure);
                    ++world.sessions_created;
                }
                if (hook)
                    hook(world);
                return std::make_unique<FakeSession>(world, targets);
            },
            with_publish ? std::function<void(const std::string&)>(
                               [this](const std::string& text)
                               {
                                   std::lock_guard<std::mutex> lock(world.mutex);
                                   world.published.push_back(text);
                               }) :
                           std::function<void(const std::string&)>(),
            config);
    }

    ~Harness()
    {
        stop_and_join();
    }

    static std::vector<StreamStatus> default_streams()
    {
        StreamStatus a;
        a.track = "a";
        StreamStatus b;
        b.track = "b";
        b.direct_host = "cfg-host";
        b.direct_port = 4000;
        return { a, b };
    }

    void start()
    {
        thread = std::thread(
            [this]
            {
                run_thread_id = std::this_thread::get_id();
                supervisor->run([this] { return stop.load(); });
            });
    }

    void stop_and_join()
    {
        stop = true;
        if (thread.joinable())
            thread.join();
    }

    template <typename Pred>
    bool wait_for(Pred pred, std::chrono::milliseconds limit = 5s)
    {
        const auto until = std::chrono::steady_clock::now() + limit;
        while (std::chrono::steady_clock::now() < until)
        {
            if (pred())
                return true;
            std::this_thread::sleep_for(5ms);
        }
        return pred();
    }

    int sessions()
    {
        std::lock_guard<std::mutex> lock(world.mutex);
        return world.sessions_created;
    }

    int updates()
    {
        std::lock_guard<std::mutex> lock(world.mutex);
        return world.update_calls;
    }

    bool state_is(ProducerState state)
    {
        return status.snapshot().state == state;
    }

    int count_events(const std::string& event, const std::string& code = "")
    {
        std::lock_guard<std::mutex> lock(world.mutex);
        int n = 0;
        for (const auto& text : world.published)
        {
            const auto j = Json::parse(text);
            if (j["event"] == event && (code.empty() || j["error"]["code"] == code))
                ++n;
        }
        return n;
    }

    World world;
    ProducerStatus status;
    std::unique_ptr<ProducerSupervisor> supervisor;
    std::atomic<bool> stop{ false };
    std::thread thread;
    std::atomic<std::thread::id> run_thread_id{};
};

} // namespace

TEST_CASE("a camera that will not start is a reported state, announced once, then recovers", "[supervisor]")
{
    Harness h;
    h.world.fail_factory_until_call = 4;
    h.start();

    REQUIRE(h.wait_for([&] { return h.state_is(ProducerState::CameraFault); }));
    {
        const auto s = h.status.snapshot();
        REQUIRE(s.fault.has_value());
        CHECK(s.fault->code == "camera_start_failed");
        CHECK(s.fault->message.find("power cable") != std::string::npos);
        CHECK(s.fault->message.find("SIPLStatus 10") != std::string::npos);
        CHECK(s.streams.size() == 2); // the configured streams stay listed while the cameras are down
    }

    REQUIRE(h.wait_for([&] { return h.state_is(ProducerState::Streaming); }));
    CHECK_FALSE(h.status.snapshot().fault.has_value());
    CHECK(h.world.factory_calls == 5); // four failures, then one success

    // Four identical failures produced exactly one announcement.
    CHECK(h.count_events("error", "camera_start_failed") == 1);
    CHECK(h.count_events("error") == 1);
    h.stop_and_join();
}

TEST_CASE("a different failure message is announced again", "[supervisor]")
{
    Harness h;
    h.world.fail_factory_until_call = 1000;
    h.start();
    REQUIRE(h.wait_for([&] { return h.count_events("error") == 1; }));
    {
        std::lock_guard<std::mutex> lock(h.world.mutex);
        h.world.factory_failure = "brand new problem";
    }
    REQUIRE(h.wait_for([&] { return h.count_events("error") == 2; }));
    CHECK(h.status.snapshot().fault->code == "startup_failed");
    h.stop_and_join();
}

TEST_CASE("a StartupFailure supplies its own fault and state", "[supervisor]")
{
    // A factory that has already explained its failure.
    SupervisorConfig config;
    config.retry_interval = 100ms;
    config.poll_interval = 2ms;
    std::atomic<int> calls{ 0 };
    ProducerStatus status("r");
    ProducerSupervisor supervisor(
        status, Harness::default_streams(),
        [&](const DirectTargets&) -> std::unique_ptr<CameraSession>
        {
            ++calls;
            throw StartupFailure(Fault{ "room_unreachable", "Cannot reach the PC video stage." }, ProducerState::Error);
        },
        nullptr, config);
    std::atomic<bool> stop{ false };
    std::thread thread([&] { supervisor.run([&] { return stop.load(); }); });
    const auto until = std::chrono::steady_clock::now() + 5s;
    while (status.snapshot().state != ProducerState::Error && std::chrono::steady_clock::now() < until)
        std::this_thread::sleep_for(5ms);
    const auto s = status.snapshot();
    stop = true;
    thread.join();
    CHECK(s.state == ProducerState::Error);
    REQUIRE(s.fault.has_value());
    CHECK(s.fault->code == "room_unreachable");
    CHECK(s.fault->message == "Cannot reach the PC video stage.");
    CHECK(calls.load() >= 1);
}

TEST_CASE("update() throwing reports Error and the session is restarted", "[supervisor]")
{
    Harness h;
    h.world.throw_update_after = 3;
    h.world.update_failure = "room lost";
    h.start();

    REQUIRE(h.wait_for([&] { return h.count_events("error", "stream_failed") >= 1; }));
    REQUIRE(h.wait_for([&] { return h.state_is(ProducerState::Error); }));
    const auto s = h.status.snapshot();
    REQUIRE(s.fault.has_value());
    CHECK(s.fault->code == "stream_failed");
    CHECK(s.fault->message.find("room lost") != std::string::npos);

    // After the retry interval a fresh session starts and streams again.
    REQUIRE(h.wait_for([&] { return h.sessions() >= 2; }));
    REQUIRE(h.wait_for([&] { return h.state_is(ProducerState::Streaming) || h.state_is(ProducerState::Error); }));
    h.stop_and_join();
    CHECK(h.world.flushes >= 1);
}

TEST_CASE("an unrecognised update() failure is reported as stream_failed with its text", "[supervisor]")
{
    Harness h;
    h.world.throw_update_after = 3;
    h.world.update_failure = "something odd";
    h.start();

    REQUIRE(h.wait_for([&] { return h.count_events("error", "stream_failed") >= 1; }));
    REQUIRE(h.wait_for([&] { return h.state_is(ProducerState::Error); }));
    const auto s = h.status.snapshot();
    REQUIRE(s.fault.has_value());
    CHECK(s.fault->code == "stream_failed");
    CHECK(s.fault->message.find("something odd") != std::string::npos);
    CHECK(h.count_events("error", "startup_failed") == 0);
    h.stop_and_join();
}

TEST_CASE("a restarted session keeps targets set at runtime", "[supervisor]")
{
    Harness h;
    h.start();
    REQUIRE(h.wait_for([&] { return h.state_is(ProducerState::Streaming); }));

    SECTION("set while streaming reaches the live session and the next factory call")
    {
        const auto result = h.supervisor->set_direct_target("a", "10.9.8.7", 5730);
        CHECK(result.ok);
        // The running thread applies it between updates, so it arrives shortly after the call returns.
        REQUIRE(h.wait_for(
            [&]
            {
                std::lock_guard<std::mutex> lock(h.world.mutex);
                return !h.world.retargets.empty();
            }));
        {
            std::lock_guard<std::mutex> lock(h.world.mutex);
            REQUIRE(h.world.retargets.size() == 1);
            CHECK(std::get<0>(h.world.retargets[0]) == "a");
            CHECK(std::get<1>(h.world.retargets[0]) == "10.9.8.7");
            CHECK(std::get<2>(h.world.retargets[0]) == 5730);
        }
        // The status shows what the session reports after it applied the change.
        REQUIRE(h.wait_for(
            [&]
            {
                const auto s = h.status.snapshot();
                return !s.streams.empty() && s.streams[0].direct_host == "10.9.8.7" &&
                       s.streams[0].direct_port == 5730 && s.streams[0].frames == 5;
            }));
        // A broadcast notified the change.
        CHECK(h.count_events("state_changed") >= 1);

        // Make the session fail; the next factory call must see the remembered target.
        {
            std::lock_guard<std::mutex> lock(h.world.mutex);
            h.world.throw_update_after = 0;
        }
        REQUIRE(h.wait_for([&] { return h.sessions() >= 2; }, 10s));
        std::lock_guard<std::mutex> lock(h.world.mutex);
        const auto& last = h.world.targets_seen.back();
        REQUIRE(last.count("a") == 1);
        CHECK(last.at("a").host == "10.9.8.7");
        CHECK(last.at("a").port == 5730);
        CHECK(last.at("b").host == "cfg-host"); // untouched streams keep their configured target
        CHECK(last.at("b").port == 4000);
    }
    SECTION("an empty host disables the link and is remembered as such")
    {
        CHECK(h.supervisor->set_direct_target("b", "", 1234).ok);
        REQUIRE(h.wait_for(
            [&]
            {
                std::lock_guard<std::mutex> lock(h.world.mutex);
                return !h.world.retargets.empty();
            }));
        {
            std::lock_guard<std::mutex> lock(h.world.mutex);
            h.world.throw_update_after = 0;
        }
        REQUIRE(h.wait_for([&] { return h.sessions() >= 2; }, 10s));
        std::lock_guard<std::mutex> lock(h.world.mutex);
        CHECK(h.world.targets_seen.back().at("b").host.empty());
        CHECK(h.world.targets_seen.back().at("b").port == 0);
    }
    SECTION("an unknown stream is rejected and changes nothing")
    {
        const auto result = h.supervisor->set_direct_target("nope", "h", 1);
        CHECK_FALSE(result.ok);
        CHECK(result.code == "unknown_stream");
        std::lock_guard<std::mutex> lock(h.world.mutex);
        CHECK(h.world.retargets.empty());
    }
    h.stop_and_join();
}

TEST_CASE("a target set while the cameras are down is used when they come up", "[supervisor]")
{
    Harness h;
    h.world.fail_factory_until_call = 1000;
    h.start();
    REQUIRE(h.wait_for([&] { return h.state_is(ProducerState::CameraFault); }));

    CHECK(h.supervisor->set_direct_target("a", "10.0.0.5", 5731).ok);
    {
        std::lock_guard<std::mutex> lock(h.world.mutex);
        CHECK(h.world.retargets.empty()); // there was no session to retarget
        h.world.fail_factory_until_call = 0;
    }
    REQUIRE(h.wait_for([&] { return h.state_is(ProducerState::Streaming); }));
    std::lock_guard<std::mutex> lock(h.world.mutex);
    CHECK(h.world.targets_seen.back().at("a").host == "10.0.0.5");
    CHECK(h.world.targets_seen.back().at("a").port == 5731);
}

TEST_CASE("the stop flag ends run() with Stopping, even during a retry wait", "[supervisor]")
{
    SECTION("while streaming")
    {
        Harness h;
        h.start();
        REQUIRE(h.wait_for([&] { return h.state_is(ProducerState::Streaming); }));
        h.stop_and_join();
        CHECK(h.state_is(ProducerState::Stopping));
        CHECK(h.world.flushes == 1);
    }
    SECTION("while the cameras are down")
    {
        Harness h;
        h.world.fail_factory_until_call = 1000;
        h.start();
        REQUIRE(h.wait_for([&] { return h.state_is(ProducerState::CameraFault); }));
        const auto begin = std::chrono::steady_clock::now();
        h.stop_and_join();
        CHECK(std::chrono::steady_clock::now() - begin < 2s);
        CHECK(h.state_is(ProducerState::Stopping));
        CHECK_FALSE(h.status.snapshot().fault.has_value());
    }
}

TEST_CASE("an empty publish callback is allowed", "[supervisor]")
{
    Harness h(Harness::default_streams(), /*with_publish=*/false);
    h.start();
    REQUIRE(h.wait_for([&] { return h.state_is(ProducerState::Streaming); }));
    CHECK(h.supervisor->set_direct_target("a", "h", 1).ok);
    h.stop_and_join();
}

TEST_CASE("a factory failure is diagnosed and sets the state from the diagnosis", "[supervisor]")
{
    SECTION("an unrecognised startup failure is a generic CameraFault")
    {
        Harness h;
        h.world.fail_factory_until_call = 1000;
        h.world.factory_failure = "LiveKit: cannot join the room";
        h.start();
        REQUIRE(h.wait_for([&] { return h.state_is(ProducerState::CameraFault); }));
        const auto s = h.status.snapshot();
        REQUIRE(s.fault.has_value());
        CHECK(s.fault->code == "startup_failed");
        CHECK(s.fault->message.find("LiveKit: cannot join the room") != std::string::npos);
        CHECK(h.count_events("error", "startup_failed") == 1);
        h.stop_and_join();
    }
    SECTION("a driver line about an open circuit is a CameraFault")
    {
        Harness h;
        h.world.fail_factory_until_call = 1000;
        h.world.factory_failure = "INvSIPLCamera::Init failed with SIPLStatus 10 [Severity 2 : OPEN_CIRCUIT]";
        h.start();
        REQUIRE(h.wait_for([&] { return h.state_is(ProducerState::CameraFault); }));
        const auto s = h.status.snapshot();
        REQUIRE(s.fault.has_value());
        CHECK(s.fault->code == "camera_power_missing");
        h.stop_and_join();
    }
}

TEST_CASE("a stream whose frames stop is failed and the session restarted", "[supervisor][stall]")
{
    Harness h(Harness::default_streams(), true, 300ms);
    h.start();
    REQUIRE(h.wait_for([&] { return h.state_is(ProducerState::Streaming); }));
    // The fake session's frame count never moves, so the watchdog fires.
    REQUIRE(h.wait_for([&] { return h.count_events("error", "camera_no_frames") >= 1; }));
    {
        const auto s = h.status.snapshot();
        // The text names the stall; camera_fault.cpp files "no frame for" under CameraFault.
        if (s.fault.has_value() && s.fault->code == "camera_no_frames")
        {
            CHECK(s.fault->message.find("no frame for") != std::string::npos);
            CHECK(s.fault->message.find("(stream ") != std::string::npos);
            CHECK(s.fault->message.find("from sensor") != std::string::npos);
        }
    }
    REQUIRE(h.wait_for([&] { return h.sessions() >= 2; }));
    h.stop_and_join();
    CHECK(h.world.flushes >= 1);
}

TEST_CASE("the stall state and text are those of the diagnosis", "[supervisor][stall]")
{
    Harness h(Harness::default_streams(), true, 300ms);
    h.start();
    REQUIRE(h.wait_for([&] { return h.count_events("error", "camera_no_frames") >= 1; }));
    const auto event = [&]
    {
        std::lock_guard<std::mutex> lock(h.world.mutex);
        for (const auto& text : h.world.published)
        {
            const auto j = Json::parse(text);
            if (j["event"] == "error" && j["error"]["code"] == "camera_no_frames")
                return j;
        }
        return Json();
    }();
    CHECK(event["error"]["message"].get<std::string>().find("no frame for") != std::string::npos);
    h.stop_and_join();
}

TEST_CASE("frames that keep advancing are not a stall", "[supervisor][stall]")
{
    Harness h(Harness::default_streams(), true, 300ms);
    h.world.advance_frames = true;
    h.start();
    REQUIRE(h.wait_for([&] { return h.state_is(ProducerState::Streaming); }));
    std::this_thread::sleep_for(1200ms);
    CHECK(h.state_is(ProducerState::Streaming));
    CHECK(h.count_events("error") == 0);
    h.stop_and_join();
    CHECK(h.world.sessions_created == 1);
}

TEST_CASE("a stall timeout of zero disables the watchdog", "[supervisor][stall]")
{
    Harness h(Harness::default_streams(), true, 0ms);
    h.start();
    REQUIRE(h.wait_for([&] { return h.state_is(ProducerState::Streaming); }));
    std::this_thread::sleep_for(1200ms); // frames never move
    CHECK(h.state_is(ProducerState::Streaming));
    CHECK(h.count_events("error") == 0);
    h.stop_and_join();
    CHECK(h.world.sessions_created == 1);
}

TEST_CASE("a direct-link error is published once per distinct reason", "[supervisor][direct_error]")
{
    Harness h;
    h.world.advance_frames = true;
    h.start();
    REQUIRE(h.wait_for([&] { return h.state_is(ProducerState::Streaming); }));
    CHECK(h.count_events("error", "direct_link_failed") == 0);

    {
        std::lock_guard<std::mutex> lock(h.world.mutex);
        h.world.direct_errors["a"] = "cannot connect to 10.0.0.5:5600: Connection refused";
    }
    REQUIRE(h.wait_for([&] { return h.count_events("error", "direct_link_failed") == 1; }));
    // Many status refreshes later, still one.
    std::this_thread::sleep_for(300ms);
    CHECK(h.count_events("error", "direct_link_failed") == 1);

    {
        std::lock_guard<std::mutex> lock(h.world.mutex);
        h.world.direct_errors["a"] = "10.0.0.5:5600 closed the control connection";
    }
    REQUIRE(h.wait_for([&] { return h.count_events("error", "direct_link_failed") == 2; }));

    // Clearing publishes nothing.
    {
        std::lock_guard<std::mutex> lock(h.world.mutex);
        h.world.direct_errors["a"] = "";
    }
    std::this_thread::sleep_for(300ms);
    CHECK(h.count_events("error", "direct_link_failed") == 2);

    // The same reason coming back after it cleared is a new occurrence.
    {
        std::lock_guard<std::mutex> lock(h.world.mutex);
        h.world.direct_errors["a"] = "10.0.0.5:5600 closed the control connection";
    }
    REQUIRE(h.wait_for([&] { return h.count_events("error", "direct_link_failed") == 3; }));

    // It names the stream and the reason, and does not change the producer state.
    {
        std::lock_guard<std::mutex> lock(h.world.mutex);
        const auto j = Json::parse(h.world.published.back());
        const auto message = j["error"]["message"].get<std::string>();
        CHECK(message.find("Stream a: ") != std::string::npos);
        CHECK(message.find("closed the control connection") != std::string::npos);
    }
    CHECK(h.state_is(ProducerState::Streaming));
    h.stop_and_join();
}

TEST_CASE("direct-link errors of different streams are independent", "[supervisor][direct_error]")
{
    Harness h;
    h.world.advance_frames = true;
    h.start();
    REQUIRE(h.wait_for([&] { return h.state_is(ProducerState::Streaming); }));
    {
        std::lock_guard<std::mutex> lock(h.world.mutex);
        h.world.direct_errors["a"] = "same reason";
        h.world.direct_errors["b"] = "same reason";
    }
    REQUIRE(h.wait_for([&] { return h.count_events("error", "direct_link_failed") == 2; }));
    h.stop_and_join();
}

namespace
{

/// A factory hook that parks the first call until released, so the test can act while a session is being built.
struct FactoryGate
{
    explicit FactoryGate(World& world) : released(release.get_future().share())
    {
        world.on_factory = [this](World&)
        {
            if (!entered_once.exchange(true))
                entered.set_value();
            released.wait();
        };
    }
    std::promise<void> entered_promise;
    std::promise<void> release;
    std::shared_future<void> released;
    std::atomic<bool> entered_once{ false };
    std::promise<void>& entered = entered_promise;
};

} // namespace

TEST_CASE("a change made while the factory runs is applied before the session's first update", "[supervisor][threading]")
{
    Harness h;
    FactoryGate gate(h.world);
    auto entered = gate.entered_promise.get_future();
    h.start();
    REQUIRE(entered.wait_for(5s) == std::future_status::ready);

    // The factory is blocked: the session does not exist yet. The call must return at once.
    const auto begin = std::chrono::steady_clock::now();
    CHECK(h.supervisor->set_direct_target("a", "10.1.1.1", 5741).ok);
    CHECK(std::chrono::steady_clock::now() - begin < 1s);
    {
        std::lock_guard<std::mutex> lock(h.world.mutex);
        CHECK(h.world.retargets.empty());
        REQUIRE(h.world.targets_seen.size() == 1);
        CHECK(h.world.targets_seen[0].at("a").host.empty()); // chosen after the factory took its snapshot
    }
    gate.release.set_value();

    REQUIRE(h.wait_for([&] { return h.updates() >= 3; }));
    std::lock_guard<std::mutex> lock(h.world.mutex);
    REQUIRE(h.world.retargets.size() == 1);
    CHECK(std::get<0>(h.world.retargets[0]) == "a");
    CHECK(std::get<1>(h.world.retargets[0]) == "10.1.1.1");
    CHECK(std::get<2>(h.world.retargets[0]) == 5741);
    REQUIRE(h.world.sequence.size() >= 2);
    CHECK(h.world.sequence[0] == "retarget:a:10.1.1.1:5741"); // before the first update
    CHECK(h.world.sequence[1] == "update");
}

TEST_CASE("a target chosen before the factory runs is passed to it and not applied again", "[supervisor][threading]")
{
    Harness h;
    CHECK(h.supervisor->set_direct_target("a", "10.2.2.2", 5742).ok);
    h.start();
    REQUIRE(h.wait_for([&] { return h.updates() >= 5; }));
    h.stop_and_join();
    std::lock_guard<std::mutex> lock(h.world.mutex);
    REQUIRE(h.world.targets_seen.size() >= 1);
    CHECK(h.world.targets_seen[0].at("a").host == "10.2.2.2");
    CHECK(h.world.targets_seen[0].at("a").port == 5742);
    CHECK(h.world.retargets.empty());
}

TEST_CASE("set_direct_target never calls into the session from the caller's thread", "[supervisor][threading]")
{
    Harness h;
    h.start();
    REQUIRE(h.wait_for([&] { return h.state_is(ProducerState::Streaming); }));
    for (int i = 0; i < 5; ++i)
        CHECK(h.supervisor->set_direct_target("a", "10.3.3." + std::to_string(i), 5743).ok);
    REQUIRE(h.wait_for(
        [&]
        {
            std::lock_guard<std::mutex> lock(h.world.mutex);
            return !h.world.retargets.empty() && std::get<1>(h.world.retargets.back()) == "10.3.3.4";
        }));
    h.stop_and_join();
    std::lock_guard<std::mutex> lock(h.world.mutex);
    REQUIRE_FALSE(h.world.retarget_threads.empty());
    for (const auto& id : h.world.retarget_threads)
    {
        CHECK(id == h.run_thread_id.load());
        CHECK(id != std::this_thread::get_id());
    }
}

TEST_CASE("changes to the same track coalesce to the last", "[supervisor][threading]")
{
    Harness h;
    FactoryGate gate(h.world);
    auto entered = gate.entered_promise.get_future();
    h.start();
    REQUIRE(entered.wait_for(5s) == std::future_status::ready);

    CHECK(h.supervisor->set_direct_target("a", "10.4.4.1", 1).ok);
    CHECK(h.supervisor->set_direct_target("a", "10.4.4.2", 2).ok);
    CHECK(h.supervisor->set_direct_target("a", "10.4.4.3", 3).ok);
    CHECK(h.supervisor->set_direct_target("b", "10.4.4.9", 9).ok);
    gate.release.set_value();

    REQUIRE(h.wait_for([&] { return h.updates() >= 3; }));
    h.stop_and_join();
    std::lock_guard<std::mutex> lock(h.world.mutex);
    REQUIRE(h.world.retargets.size() == 2); // one per track, not one per call
    int a_calls = 0;
    for (const auto& r : h.world.retargets)
    {
        if (std::get<0>(r) == "a")
        {
            ++a_calls;
            CHECK(std::get<1>(r) == "10.4.4.3");
            CHECK(std::get<2>(r) == 3);
        }
    }
    CHECK(a_calls == 1);
}

TEST_CASE("the status shows the session's reported host and port", "[supervisor][threading]")
{
    Harness h;
    h.start();
    REQUIRE(h.wait_for([&] { return h.state_is(ProducerState::Streaming); }));
    CHECK(h.supervisor->set_direct_target("a", "10.5.5.5", 5745).ok);
    REQUIRE(h.wait_for(
        [&]
        {
            const auto s = h.status.snapshot();
            for (const auto& stream : s.streams)
                if (stream.track == "a")
                    return stream.direct_host == "10.5.5.5" && stream.direct_port == 5745;
            return false;
        }));
    const auto s = h.status.snapshot();
    for (const auto& stream : s.streams)
    {
        if (stream.track == "b")
        {
            CHECK(stream.direct_host == "cfg-host");
            CHECK(stream.direct_port == 4000);
        }
    }
}
