// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <chrony_clock/chrony_capture_clock.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using camera_bridge::ChronyCaptureClock;
using camera_bridge::ChronyCaptureClockConfig;
using isaaccapture::timesync::ClockReader;
using isaaccapture::timesync::CommandResult;
using isaaccapture::timesync::CommandRunner;
using isaaccapture::timesync::Qualification;
using isaaccapture::timesync::Verdict;

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

constexpr int64_t kMs = 1'000'000;
constexpr int64_t kSec = 1'000'000'000;
const char* const kExpectedRefId = "C0A80001";

// chronyc -c tracking: refid,name,stratum,ref time,system time,last offset,rms offset,frequency,residual freq,skew,
// root delay,root dispersion,update interval,leap status.
std::string tracking_line(const std::string& refid,
                          int stratum,
                          double ref_time_s,
                          double root_dispersion_s,
                          const std::string& leap = "Normal")
{
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "%s,host,%d,%.9f,0.000000011,-0.000001098,0.000008212,1.382,0.001,0.019,"
                  "0.000100,%.9f,2.0,%s",
                  refid.c_str(), stratum, ref_time_s, root_dispersion_s, leap.c_str());
    return buf;
}

// Answers by argv: no "-h", or "-h 127.0.0.1", is the PC's chronyd; any other "-h HOST" is the camera host's.
struct FakeChrony
{
    std::mutex mutex;
    CommandResult pc;
    CommandResult sender;
    std::atomic<int> pc_calls{ 0 };
    std::atomic<int> sender_calls{ 0 };

    CommandRunner runner()
    {
        return [this](const std::vector<std::string>& argv, std::chrono::milliseconds)
        {
            bool remote = false;
            for (size_t i = 0; i + 1 < argv.size(); ++i)
                remote = remote || (argv[i] == "-h" && argv[i + 1] != "127.0.0.1");
            (remote ? sender_calls : pc_calls)++;
            std::lock_guard<std::mutex> lock(mutex);
            return remote ? sender : pc;
        };
    }
};

struct FakeClocks
{
    std::atomic<int64_t> monotonic{ 1'000 * kSec };
    std::atomic<int64_t> realtime{ 1'800'000'000 * kSec + 123'456'789 };

    ClockReader reader()
    {
        return ClockReader{ [this] { return monotonic.load(); }, [this] { return realtime.load(); } };
    }
    void advance(int64_t ns)
    {
        monotonic += ns;
        realtime += ns;
    }
};

CommandResult ok(const std::string& line)
{
    CommandResult r;
    r.started = true;
    r.exit_code = 0;
    r.output = line + "\n";
    return r;
}

ChronyCaptureClockConfig make_config()
{
    ChronyCaptureClockConfig c;
    c.sender_host = "camera-host";
    c.expected_sender_reference_id = kExpectedRefId;
    c.pc_port = 3323;
    c.poll_interval = std::chrono::milliseconds(50);
    return c;
}

struct Rig
{
    FakeChrony chrony;
    FakeClocks clocks;
    ChronyCaptureClockConfig config = make_config();
    std::unique_ptr<ChronyCaptureClock> clock;

    double now_s() const
    {
        return static_cast<double>(clocks.realtime.load()) * 1e-9;
    }
    void set_sender(const std::string& refid, int stratum, double disp_s, const std::string& leap = "Normal")
    {
        std::lock_guard<std::mutex> lock(chrony.mutex);
        chrony.sender = ok(tracking_line(refid, stratum, now_s() - 1.0, disp_s, leap));
    }
    void set_pc(const std::string& refid, int stratum, double disp_s)
    {
        std::lock_guard<std::mutex> lock(chrony.mutex);
        chrony.pc = ok(tracking_line(refid, stratum, now_s() - 1.0, disp_s));
    }
    void healthy()
    {
        set_sender("A9FEA97B", 3, 0.0005); // the camera host, following its own upstream
        set_pc(kExpectedRefId, 4, 0.0005); // this machine's chronyd, following the camera host
    }
    void build()
    {
        clock = std::make_unique<ChronyCaptureClock>(config, chrony.runner(), clocks.reader());
    }
    Qualification qualify(int64_t age_ns = 10 * kMs, bool valid = true, int64_t upstream_ns = 100'000)
    {
        return clock->qualify(valid, clocks.realtime.load() - age_ns, upstream_ns);
    }
};

void test_happy_path()
{
    Rig r;
    r.healthy();
    r.build();
    r.clock->poll_once();
    CHECK(r.chrony.pc_calls == 1 && r.chrony.sender_calls == 1);
    const int64_t realtime = r.clocks.realtime.load();
    const Qualification q = r.clock->qualify(true, realtime - 10 * kMs, 100'000);
    CHECK(q.verdict == Verdict::Qualified);
    CHECK(q.capture_monotonic_ns == (realtime - 10 * kMs) - (realtime - r.clocks.monotonic.load()));
    CHECK(q.uncertainty_ns > 0 && q.uncertainty_ns <= r.config.budget_ns);
    CHECK(q.generation >= 1);
}

void test_camera_host_local_reference()
{
    // The camera host may serve its own clock (chrony `local`); only this machine has to follow it.
    Rig r;
    r.healthy();
    r.set_sender("7F7F0101", 10, 0.0005);
    r.build();
    r.clock->poll_once();
    CHECK(r.qualify().verdict == Verdict::Qualified);
}

void test_pc_local_reference_refused()
{
    Rig r;
    r.healthy();
    r.set_pc("7F7F0101", 10, 0.0005);
    r.build();
    r.clock->poll_once();
    CHECK(r.qualify().verdict == Verdict::Unsynchronised);
}

void test_sender_invalid()
{
    Rig r;
    r.healthy();
    r.build();
    r.clock->poll_once();
    CHECK(r.qualify(10 * kMs, false).verdict == Verdict::SenderInvalid);
}

void test_no_poll_yet()
{
    Rig r;
    r.healthy();
    r.build();
    CHECK(r.qualify().verdict == Verdict::NoSenderEvidence);
}

void test_sender_unreachable()
{
    Rig r;
    r.healthy();
    r.chrony.sender = CommandResult{}; // started == false
    r.build();
    r.clock->poll_once();
    CHECK(r.chrony.pc_calls == 1 && r.chrony.sender_calls == 1);
    CHECK(r.qualify().verdict == Verdict::NoSenderEvidence);
}

void test_reference_mismatch()
{
    Rig r;
    r.healthy();
    r.set_pc("C0A80002", 4, 0.0005); // this machine follows something other than the camera host
    r.build();
    r.clock->poll_once();
    CHECK(r.qualify().verdict == Verdict::ReferenceMismatch);
}

void test_unsynchronised()
{
    Rig r;
    r.healthy();
    r.set_sender("A9FEA97B", 3, 0.0005, "Not synchronised");
    r.build();
    r.clock->poll_once();
    CHECK(r.qualify().verdict == Verdict::Unsynchronised);
}

void test_stale_evidence()
{
    Rig r;
    r.healthy();
    r.build();
    r.clock->poll_once();
    r.clocks.advance(11 * kSec); // max_evidence_age is 10 s
    CHECK(r.qualify().verdict == Verdict::StaleEvidence);
}

void test_realtime_step()
{
    Rig r;
    r.healthy();
    r.build();
    r.clock->poll_once();
    const Qualification first = r.qualify();
    CHECK(first.verdict == Verdict::Qualified);

    r.clocks.realtime += 50 * kMs;
    const Qualification stepped = r.qualify();
    CHECK(stepped.verdict == Verdict::NoLocalClock);
    CHECK(stepped.generation > first.generation);

    const int64_t realtime = r.clocks.realtime.load();
    const Qualification after = r.clock->qualify(true, realtime - 10 * kMs, 100'000);
    CHECK(after.verdict == Verdict::Qualified);
    CHECK(after.generation == stepped.generation);
    CHECK(after.capture_monotonic_ns == (realtime - 10 * kMs) - (realtime - r.clocks.monotonic.load()));
}

void test_over_budget()
{
    Rig r;
    r.healthy();
    r.set_pc(kExpectedRefId, 4, 0.010); // the link dispersion is what this machine adds beyond the camera host
    r.build();
    r.clock->poll_once();
    const Qualification q = r.qualify();
    CHECK(q.verdict == Verdict::OverBudget);
    CHECK(q.uncertainty_ns > r.config.budget_ns);
}

void test_start_stop()
{
    Rig r;
    r.healthy();
    r.build();
    r.clock->start();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while ((r.chrony.pc_calls < 2 || r.chrony.sender_calls < 2) && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    CHECK(r.chrony.pc_calls >= 2);
    CHECK(r.chrony.sender_calls >= 2);
    CHECK(r.qualify().verdict == Verdict::Qualified);

    const auto t0 = std::chrono::steady_clock::now();
    r.clock->stop();
    CHECK(std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(500));
    const int polls = r.chrony.pc_calls;
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    CHECK(r.chrony.pc_calls == polls);
}

void test_concurrent_qualify()
{
    Rig r;
    r.healthy();
    r.build();
    r.clock->poll_once();
    std::atomic<int> bad{ 0 };
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t)
    {
        threads.emplace_back(
            [&]
            {
                for (int i = 0; i < 200; ++i)
                {
                    const Qualification q = r.qualify();
                    if (q.verdict == Verdict::Qualified)
                    {
                        if (q.uncertainty_ns <= 0)
                            ++bad;
                    }
                    else if (std::string(isaaccapture::timesync::to_string(q.verdict)) == "unknown")
                    {
                        ++bad;
                    }
                }
            });
    }
    for (std::thread& t : threads)
        t.join();
    CHECK(bad == 0);
}

} // namespace

int main()
{
    test_happy_path();
    test_camera_host_local_reference();
    test_pc_local_reference_refused();
    test_sender_invalid();
    test_no_poll_yet();
    test_sender_unreachable();
    test_reference_mismatch();
    test_unsynchronised();
    test_stale_evidence();
    test_realtime_step();
    test_over_budget();
    test_start_stop();
    test_concurrent_qualify();
    if (g_failures != 0)
    {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("chrony_clock tests passed\n");
    return 0;
}
