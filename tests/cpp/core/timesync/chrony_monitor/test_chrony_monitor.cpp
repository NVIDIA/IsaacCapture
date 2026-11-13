// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <timesync/chrony_monitor.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <future>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

using namespace isaaccapture::timesync;
using std::chrono::milliseconds;

static int g_failures = 0;

#define CHECK(cond)                                                                                                    \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(cond))                                                                                                   \
        {                                                                                                              \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond);                              \
            ++g_failures;                                                                                              \
        }                                                                                                              \
    } while (0)

namespace
{

const char* const kGoodLine =
    "A9FEA97B,169.254.169.123,4,1707763422.384912731,0.000000011,-0.000001098,0.000008212,1.382,0.001,0.019,"
    "0.000289,0.000357,2.0,Normal";

using Clock = std::chrono::steady_clock;

// Fake runner state shared with the test; the runner may be called from the monitor thread.
struct Fake
{
    std::mutex mutex;
    CommandResult result;
    std::vector<std::string> last_argv;
    milliseconds last_timeout{ 0 };
    std::atomic<int> calls{ 0 };
    std::set<std::thread::id> threads;
    std::atomic<int64_t> clock{ 1000 };
    std::atomic<int64_t> realtime{ 7'000'000'000'000 };
    int64_t advance_by = 0; // clock advance while the "command" runs
    milliseconds block{ 0 }; // real sleep inside the runner

    CommandRunner runner()
    {
        return [this](const std::vector<std::string>& argv, milliseconds timeout)
        {
            CommandResult r;
            milliseconds b;
            {
                std::lock_guard<std::mutex> lock(mutex);
                last_argv = argv;
                last_timeout = timeout;
                threads.insert(std::this_thread::get_id());
                r = result;
                b = block;
            }
            clock += advance_by;
            realtime += advance_by;
            ++calls;
            if (b.count() > 0)
            {
                std::this_thread::sleep_for(b);
            }
            return r;
        };
    }
    ChronyMonitor::NowFn now()
    {
        return [this] { return clock.load(); };
    }
    ChronyMonitor::NowFn now_realtime()
    {
        return [this] { return realtime.load(); };
    }
};

CommandResult good()
{
    CommandResult r;
    r.started = true;
    r.exit_code = 0;
    r.output = std::string(kGoodLine) + "\n";
    return r;
}

void test_argv_local()
{
    Fake f;
    f.result = good();
    ChronyMonitorConfig cfg;
    cfg.chronyc_path = "/usr/bin/chronyc";
    ChronyMonitor m(cfg, f.runner(), f.now());
    m.poll_once();
    CHECK((f.last_argv == std::vector<std::string>{ "/usr/bin/chronyc", "-c", "tracking" }));
}

void test_argv_remote()
{
    Fake f;
    f.result = good();
    ChronyMonitorConfig cfg;
    cfg.chronyc_path = "/usr/bin/chronyc";
    cfg.host = "10.0.0.5";
    cfg.port = 323;
    ChronyMonitor m(cfg, f.runner(), f.now());
    m.poll_once();
    CHECK((f.last_argv ==
           std::vector<std::string>{ "/usr/bin/chronyc", "-c", "-h", "10.0.0.5", "-p", "323", "tracking" }));
}

void test_good_poll()
{
    Fake f;
    f.result = good();
    f.advance_by = 500;
    ChronyMonitorConfig cfg;
    cfg.command_timeout = milliseconds(1234);
    ChronyMonitor m(cfg, f.runner(), f.now());
    CHECK(m.snapshot().last_status == MonitorStatus::NotPolled);
    CHECK(m.poll_once() == MonitorStatus::Ok);
    const auto s = m.snapshot();
    CHECK(s.last_status == MonitorStatus::Ok);
    CHECK(s.polls == 1);
    CHECK(s.failures == 0);
    CHECK(s.evidence.has_value());
    if (s.evidence)
    {
        CHECK(s.evidence->tracking.reference_id == "A9FEA97B");
        CHECK(s.evidence->tracking.stratum == 4);
        CHECK(s.evidence->observed_monotonic_ns == 1000); // read before the runner advanced the clock
    }
    CHECK(f.clock.load() == 1500);
    CHECK(f.last_timeout == milliseconds(1234));
}

void test_realtime_stamp_read_before_runner()
{
    Fake f;
    f.result = good();
    f.advance_by = 500; // the runner advances both clocks
    ChronyMonitor m({}, f.runner(), f.now(), f.now_realtime());
    CHECK(m.poll_once() == MonitorStatus::Ok);
    const auto s = m.snapshot();
    CHECK(s.evidence.has_value());
    if (s.evidence)
    {
        CHECK(s.evidence->observed_monotonic_ns == 1000);
        CHECK(s.evidence->observed_realtime_ns == 7'000'000'000'000);
    }
    CHECK(f.realtime.load() == 7'000'000'000'500);
}

void test_default_realtime_clock_is_system_realtime()
{
    Fake f;
    f.result = good();
    ChronyMonitor m({}, f.runner(), f.now());
    CHECK(m.poll_once() == MonitorStatus::Ok);
    const auto s = m.snapshot();
    CHECK(s.evidence.has_value());
    if (s.evidence)
    {
        CHECK(s.evidence->observed_realtime_ns > 0);
        CHECK(s.evidence->observed_monotonic_ns == 1000);
    }
}

void test_failure_statuses()
{
    Fake f;
    ChronyMonitor m({}, f.runner(), f.now());

    f.result = CommandResult{};
    f.result.started = true;
    f.result.timed_out = true;
    CHECK(m.poll_once() == MonitorStatus::TimedOut);

    f.result = CommandResult{}; // started=false
    CHECK(m.poll_once() == MonitorStatus::CommandFailed);

    f.result = good();
    f.result.exit_code = 2;
    CHECK(m.poll_once() == MonitorStatus::CommandFailed);

    f.result = good();
    f.result.truncated = true;
    CHECK(m.poll_once() == MonitorStatus::CommandFailed);

    f.result = good();
    f.result.output = "garbage\n";
    CHECK(m.poll_once() == MonitorStatus::ParseFailed);

    const auto s = m.snapshot();
    CHECK(s.polls == 5);
    CHECK(s.failures == 5);
    CHECK(!s.evidence.has_value());
}

void test_evidence_kept_across_failure()
{
    Fake f;
    f.result = good();
    ChronyMonitor m({}, f.runner(), f.now());
    CHECK(m.poll_once() == MonitorStatus::Ok);
    f.clock = 9000;
    f.result = CommandResult{};
    f.result.started = true;
    f.result.timed_out = true;
    CHECK(m.poll_once() == MonitorStatus::TimedOut);
    const auto s = m.snapshot();
    CHECK(s.last_status == MonitorStatus::TimedOut);
    CHECK(s.polls == 2);
    CHECK(s.failures == 1);
    CHECK(s.evidence.has_value());
    if (s.evidence)
    {
        CHECK(s.evidence->observed_monotonic_ns == 1000);
    }
}

void test_to_string_distinct()
{
    const MonitorStatus all[] = { MonitorStatus::NotPolled, MonitorStatus::Ok, MonitorStatus::CommandFailed,
                                  MonitorStatus::TimedOut, MonitorStatus::ParseFailed };
    std::set<std::string> names;
    for (auto s : all)
    {
        names.insert(to_string(s));
    }
    CHECK(names.size() == 5);
    CHECK(names.count("unknown") == 0);
}

bool wait_for_calls(const Fake& f, int n, milliseconds limit)
{
    const auto end = Clock::now() + limit;
    while (Clock::now() < end)
    {
        if (f.calls.load() >= n)
        {
            return true;
        }
        std::this_thread::sleep_for(milliseconds(2));
    }
    return f.calls.load() >= n;
}

void test_thread_polls_and_single_start()
{
    Fake f;
    f.result = good();
    ChronyMonitorConfig cfg;
    cfg.poll_interval = milliseconds(10);
    ChronyMonitor m(cfg, f.runner(), f.now());
    m.start();
    m.start();
    CHECK(wait_for_calls(f, 3, milliseconds(2000)));
    m.stop();
    CHECK(m.snapshot().polls >= 3);
    std::lock_guard<std::mutex> lock(f.mutex);
    CHECK(f.threads.size() == 1);
    CHECK(f.threads.count(std::this_thread::get_id()) == 0);
}

void test_stop_prompt_and_idempotent()
{
    Fake f;
    f.result = good();
    ChronyMonitorConfig cfg;
    cfg.poll_interval = std::chrono::seconds(60);
    ChronyMonitor m(cfg, f.runner(), f.now());
    m.start();
    CHECK(wait_for_calls(f, 1, milliseconds(2000)));
    const auto t0 = Clock::now();
    m.stop();
    m.stop();
    CHECK(Clock::now() - t0 < std::chrono::seconds(1));
}

void test_destructor_stops()
{
    Fake f;
    f.result = good();
    const auto t0 = Clock::now();
    {
        ChronyMonitorConfig cfg;
        cfg.poll_interval = std::chrono::seconds(60);
        ChronyMonitor m(cfg, f.runner(), f.now());
        m.start();
        CHECK(wait_for_calls(f, 1, milliseconds(2000)));
    }
    CHECK(Clock::now() - t0 < std::chrono::seconds(2));
}

void test_stop_during_blocking_runner()
{
    Fake f;
    f.result = good();
    f.block = milliseconds(300);
    ChronyMonitorConfig cfg;
    cfg.poll_interval = std::chrono::seconds(60);
    ChronyMonitor m(cfg, f.runner(), f.now());
    m.start();
    // Wait until the runner has been entered (calls increments just before it sleeps).
    CHECK(wait_for_calls(f, 1, milliseconds(2000)));
    const auto t0 = Clock::now();
    m.stop();
    CHECK(Clock::now() - t0 < milliseconds(300 + 700));
}

void test_restart_after_stop()
{
    Fake f;
    f.result = good();
    ChronyMonitorConfig cfg;
    cfg.poll_interval = milliseconds(5);
    ChronyMonitor m(cfg, f.runner(), f.now());
    for (int i = 0; i < 3; ++i)
    {
        const int before = f.calls.load();
        m.start();
        CHECK(wait_for_calls(f, before + 2, milliseconds(2000)));
        m.stop();
        const int after_stop = f.calls.load();
        std::this_thread::sleep_for(milliseconds(30));
        CHECK(f.calls.load() == after_stop); // no worker left polling
    }
}

void test_concurrent_start_stop_never_hangs_or_leaks_a_worker()
{
    // Runs on a helper thread so a deadlock fails the test instead of hanging the run; a hung helper is detached.
    auto body = []
    {
        Fake f;
        f.result = good();
        ChronyMonitorConfig cfg;
        cfg.poll_interval = milliseconds(1);
        ChronyMonitor m(cfg, f.runner(), f.now());
        for (int i = 0; i < 200; ++i)
        {
            std::thread starter([&m] { m.start(); });
            std::thread stopper([&m] { m.stop(); });
            starter.join();
            stopper.join();
        }
        m.stop();
        const int after_stop = f.calls.load();
        std::this_thread::sleep_for(milliseconds(30));
        return f.calls.load() == after_stop;
    };
    std::packaged_task<bool()> task(body);
    auto fut = task.get_future();
    std::thread t(std::move(task));
    if (fut.wait_for(std::chrono::seconds(30)) != std::future_status::ready)
    {
        std::fprintf(stderr, "concurrent start/stop hung\n");
        ++g_failures;
        t.detach();
        return;
    }
    t.join();
    CHECK(fut.get()); // no worker still polling after the final stop
}

} // namespace

int main()
{
    test_argv_local();
    test_argv_remote();
    test_good_poll();
    test_realtime_stamp_read_before_runner();
    test_default_realtime_clock_is_system_realtime();
    test_failure_statuses();
    test_evidence_kept_across_failure();
    test_to_string_distinct();
    test_thread_polls_and_single_start();
    test_stop_prompt_and_idempotent();
    test_destructor_stops();
    test_stop_during_blocking_runner();
    test_restart_after_stop();
    test_concurrent_start_stop_never_hangs_or_leaks_a_worker();
    if (g_failures)
    {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("timesync_chrony_monitor: all checks passed\n");
    return 0;
}
