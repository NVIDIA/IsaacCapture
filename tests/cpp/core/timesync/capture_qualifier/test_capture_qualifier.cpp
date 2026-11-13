// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <timesync/capture_qualifier.hpp>

#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <set>
#include <string>
#include <utility>

using namespace isaaccapture::timesync;

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
constexpr int64_t kSecond = 1'000'000'000;
constexpr int64_t kPcMono = 50 * kSecond;
constexpr int64_t kWallMinusMono = 1'700'000'000'000'000'000;
constexpr int64_t kNow = kPcMono + kSecond;
constexpr int64_t kCaptureAge = 10 * kMs;
constexpr int64_t kSensorNs = 30'000;
constexpr int64_t kPcLocalNs = 20'000;
const char* const kSenderId = "A9FEA97B"; // the camera host, as the PC's chronyd reports it
const char* const kUpstreamId = "5BBD5B9D"; // whatever the camera host itself follows

// Double-to-ns conversion: the header does not state the rounding, so allow 1 ns per converted bound.
bool within(int64_t actual, int64_t expected, int64_t tol)
{
    return std::llabs(actual - expected) <= tol;
}

ChronyTracking synced(const std::string& id = kSenderId)
{
    ChronyTracking t;
    t.reference_id = id;
    t.stratum = 3;
    t.leap_status = "Normal";
    return t;
}

struct Fixture
{
    SenderCapture capture;
    std::optional<ChronyEvidence> sender;
    std::optional<ChronyEvidence> pc;
    LocalClockObservation clock;
    int64_t now = kNow;
    QualifierConfig config;

    Fixture()
    {
        config.expected_sender_reference_id = kSenderId;
        capture.valid = true;
        capture.mapping_uncertainty_ns = kSensorNs;
        capture.realtime_ns = kWallMinusMono + (kNow - kCaptureAge);
        // The PC's chronyd follows the camera host: pc is the follower and carries the expected reference id; sender
        // is the server and has an arbitrary upstream id.
        sender = ChronyEvidence{ synced(kUpstreamId), kNow - kSecond };
        pc = ChronyEvidence{ synced(), kNow - kSecond };
        clock.sample.monotonic_ns = kPcMono;
        clock.sample.realtime_ns = kPcMono + kWallMinusMono;
        clock.sample.read_uncertainty_ns = kPcLocalNs;
        clock.generation = 1;
        clock.usable = true;
    }

    Qualification run() const
    {
        return qualify_capture(capture, sender, pc, clock, now, config);
    }

    void set_capture_monotonic(int64_t mono)
    {
        capture.realtime_ns = kWallMinusMono + mono;
    }
};

void test_to_string()
{
    const Verdict all[] = { Verdict::Qualified,         Verdict::SenderInvalid, Verdict::NoSenderEvidence,
                            Verdict::NoPcEvidence,      Verdict::StaleEvidence, Verdict::Unsynchronised,
                            Verdict::ReferenceMismatch, Verdict::NoLocalClock,  Verdict::ImplausibleAge,
                            Verdict::OverBudget };
    std::set<std::string> names;
    for (Verdict v : all)
    {
        const char* s = to_string(v);
        CHECK(s != nullptr && std::strlen(s) > 0);
        if (s)
            names.insert(s);
    }
    CHECK(names.size() == sizeof(all) / sizeof(all[0]));
}

void test_chrony_error_bound()
{
    ChronyTracking t;
    CHECK(chrony_error_bound_ns(t, 0) == 0);
    CHECK(chrony_error_bound_ns(t, 5 * kSecond) == 0);

    t.system_time_s = 1e-4;
    t.root_dispersion_s = 2e-4;
    t.root_delay_s = 4e-4;
    // 100 us + 200 us + 400 us / 2
    CHECK(within(chrony_error_bound_ns(t, 0), 500'000, 1));

    t.skew_ppm = 10.0;
    // skew term: 10e-6 * 2 s = 20 us
    CHECK(within(chrony_error_bound_ns(t, 2 * kSecond), 520'000, 1));

    ChronyTracking pos;
    pos.system_time_s = 3e-4;
    ChronyTracking neg;
    neg.system_time_s = -3e-4;
    CHECK(within(chrony_error_bound_ns(pos, 0), 300'000, 1));
    CHECK(chrony_error_bound_ns(neg, 0) == chrony_error_bound_ns(pos, 0));

    ChronyTracking skew_only;
    skew_only.skew_ppm = 10.0;
    CHECK(chrony_error_bound_ns(skew_only, -kSecond) >= 0);
}

void test_qualified_happy_path()
{
    Fixture f;
    const Qualification q = f.run();
    CHECK(q.verdict == Verdict::Qualified);
    CHECK(q.capture_monotonic_ns == f.capture.realtime_ns - kWallMinusMono);
    CHECK(q.capture_monotonic_ns == kNow - kCaptureAge);
    CHECK(q.generation == 1);
    CHECK(q.terms.sensor_ns == kSensorNs);
    CHECK(q.terms.pc_local_ns == kPcLocalNs);
    CHECK(q.terms.inter_host_ns == 0);
    CHECK(q.uncertainty_ns == q.terms.sensor_ns + q.terms.inter_host_ns + q.terms.pc_local_ns);
}

void test_conversion_uses_pc_offset_not_sample_time()
{
    Fixture f;
    f.now = kNow + 5 * kMs; // later than the clock sample; the offset still applies
    f.set_capture_monotonic(f.now - kMs);
    f.sender->observed_monotonic_ns = f.now - kSecond;
    f.pc->observed_monotonic_ns = f.now - kSecond;
    const Qualification q = f.run();
    CHECK(q.verdict == Verdict::Qualified);
    CHECK(q.capture_monotonic_ns == f.now - kMs);
}

// Evidence with every chrony term zero, so each test sets only the term under test.
ChronyEvidence zero_evidence()
{
    return ChronyEvidence{ synced(), 0 };
}

// The camera host's last source update is not newer than the PC's copy of it, so the shared upstream part cancels.
void share_upstream_sample(ChronyEvidence& follower, ChronyEvidence& server)
{
    server.tracking.ref_time_unix_s = 100.0;
    follower.tracking.ref_time_unix_s = 100.0;
}

void test_inter_host_terms()
{
    Fixture f;
    f.pc->tracking.system_time_s = 1e-4;
    f.pc->tracking.root_dispersion_s = 2e-4;
    f.pc->tracking.root_delay_s = 4e-4;
    f.pc->tracking.skew_ppm = 10.0;
    f.sender->tracking.system_time_s = -2e-4;
    f.sender->tracking.root_dispersion_s = 1e-4;
    f.sender->tracking.root_delay_s = 2e-4;
    f.sender->tracking.skew_ppm = 5.0;
    f.config.budget_ns = 10 * kMs;
    share_upstream_sample(*f.pc, *f.sender);
    const Qualification q = f.run();
    // Both reports are one second old, follower (PC) update interval 0. |system time|: 100 us + 200 us; root dispersion
    // difference 200 - 100 = 100 us; root delay difference (400 - 200) / 2 = 100 us; skew * age: 10 us + 5 us.
    CHECK(q.verdict == Verdict::Qualified);
    CHECK(within(q.terms.inter_host_ns, 515'000, 2));
    CHECK(q.uncertainty_ns == q.terms.sensor_ns + q.terms.inter_host_ns + q.terms.pc_local_ns);
}

void test_upstream_cancels()
{
    // The camera host is internet-disciplined: large shared root fields, small link added by the PC.
    Fixture f;
    f.sender->tracking.root_delay_s = 0.1;
    f.sender->tracking.root_dispersion_s = 0.01;
    f.pc->tracking.root_delay_s = 0.1 + 200e-6;
    f.pc->tracking.root_dispersion_s = 0.01 + 50e-6;
    share_upstream_sample(*f.pc, *f.sender);
    const Qualification q = f.run();
    // Link only: 50 us + 200 us / 2 = 150 us. The summed model would charge about 0.11 s + 0.11 s.
    CHECK(within(q.terms.inter_host_ns, 150'000, 50)); // double error on the 0.1 s operands
    CHECK(q.terms.inter_host_ns < kMs);
    CHECK(q.verdict == Verdict::Qualified);
    CHECK(f.config.budget_ns == 5 * kMs);
    CHECK(q.uncertainty_ns == q.terms.sensor_ns + q.terms.inter_host_ns + q.terms.pc_local_ns);
}

void test_pc_below_camera_host_clamps_to_zero()
{
    Fixture f;
    share_upstream_sample(*f.pc, *f.sender);
    f.sender->tracking.root_delay_s = 0.1;
    f.sender->tracking.root_dispersion_s = 0.01;
    f.pc->tracking.root_delay_s = 0.05;
    f.pc->tracking.root_dispersion_s = 0.005;
    CHECK(f.run().terms.inter_host_ns == 0);
    CHECK(f.run().verdict == Verdict::Qualified);
    // Only the dispersion is below: the delay difference still counts, the dispersion one does not.
    f.pc->tracking.root_delay_s = 0.1 + 400e-6;
    CHECK(within(f.run().terms.inter_host_ns, 200'000, 50));
    // Only the delay is below.
    f.pc->tracking.root_delay_s = 0.05;
    f.pc->tracking.root_dispersion_s = 0.01 + 80e-6;
    CHECK(within(f.run().terms.inter_host_ns, 80'000, 50));
}

void test_copy_slack()
{
    // In the direct-call tests below, s is the follower (the PC) and p the server (the camera host).
    ChronyEvidence s = zero_evidence();
    ChronyEvidence p = zero_evidence();
    p.tracking.skew_ppm = 1.0;
    s.tracking.update_interval_s = 16.0;
    // 1e-6 * 16 s
    CHECK(inter_host_bound_ns(s, p, 0, 0) == 16'000);
    // The follower's skew and the server's update interval do not enter the slack.
    ChronyEvidence s2 = zero_evidence();
    s2.tracking.skew_ppm = 1.0;
    ChronyEvidence p2 = zero_evidence();
    p2.tracking.update_interval_s = 16.0;
    CHECK(inter_host_bound_ns(s2, p2, 0, 0) == 0);
}

void test_aging_charged_per_host()
{
    ChronyEvidence s = zero_evidence();
    ChronyEvidence p = zero_evidence();
    s.tracking.skew_ppm = 2.0;
    p.tracking.skew_ppm = 1.0;
    // follower 2 ppm * 3 s = 6 us; server 1 ppm * 4 s = 4 us.
    CHECK(inter_host_bound_ns(s, p, 3 * kSecond, 4 * kSecond) == 10'000);
    CHECK(inter_host_bound_ns(s, p, 3 * kSecond, 0) == 6'000);
    CHECK(inter_host_bound_ns(s, p, 0, 4 * kSecond) == 4'000);
    // Negative ages are treated as 0.
    CHECK(inter_host_bound_ns(s, p, -kSecond, -kSecond) == 0);

    // Through qualify_capture, with the fixture's reports one second old: 2 us + 1 us.
    Fixture f;
    f.pc->tracking.skew_ppm = 2.0;
    f.sender->tracking.skew_ppm = 1.0;
    CHECK(f.run().terms.inter_host_ns == 3'000);
    f.pc->observed_monotonic_ns = f.now - 3 * kSecond;
    f.sender->observed_monotonic_ns = f.now - 4 * kSecond;
    CHECK(f.run().terms.inter_host_ns == 10'000);
}

void test_both_system_times_charged()
{
    ChronyEvidence s = zero_evidence();
    ChronyEvidence p = zero_evidence();
    s.tracking.system_time_s = 100e-6;
    CHECK(within(inter_host_bound_ns(s, p, 0, 0), 100'000, 1));
    s.tracking.system_time_s = 0.0;
    p.tracking.system_time_s = 250e-6;
    CHECK(within(inter_host_bound_ns(s, p, 0, 0), 250'000, 1));
    s.tracking.system_time_s = 100e-6;
    CHECK(within(inter_host_bound_ns(s, p, 0, 0), 350'000, 1));
    // The sign does not matter: only the magnitude is charged.
    s.tracking.system_time_s = -100e-6;
    p.tracking.system_time_s = -250e-6;
    CHECK(within(inter_host_bound_ns(s, p, 0, 0), 350'000, 1));
    s.tracking.system_time_s = 100e-6;
    p.tracking.system_time_s = -250e-6;
    CHECK(within(inter_host_bound_ns(s, p, 0, 0), 350'000, 1));
}

void test_inter_host_bound_direct()
{
    ChronyEvidence s = zero_evidence();
    ChronyEvidence p = zero_evidence();
    CHECK(inter_host_bound_ns(s, p, 0, 0) == 0);
    CHECK(inter_host_bound_ns(s, p, 100 * kSecond, 100 * kSecond) == 0);

    // Root fields equal on both sides cancel completely, however large.
    s.tracking.root_delay_s = p.tracking.root_delay_s = 0.1;
    s.tracking.root_dispersion_s = p.tracking.root_dispersion_s = 0.01;
    share_upstream_sample(s, p);
    CHECK(inter_host_bound_ns(s, p, 0, 0) == 0);

    // Dispersion difference counts in full, delay difference at half; exact in binary doubles.
    ChronyEvidence s2 = zero_evidence();
    ChronyEvidence p2 = zero_evidence();
    s2.tracking.root_dispersion_s = 0.5;
    s2.tracking.root_delay_s = 0.25;
    CHECK(inter_host_bound_ns(s2, p2, 0, 0) == 500'000'000 + 125'000'000);

    // A follower's root fields below the server's contribute nothing, never a negative term.
    share_upstream_sample(p2, s2);
    CHECK(inter_host_bound_ns(p2, s2, 0, 0) == 0);

    // Combined: 1 us slack + 2 us follower aging + 3 us server aging = 6 us.
    ChronyEvidence s3 = zero_evidence();
    ChronyEvidence p3 = zero_evidence();
    p3.tracking.skew_ppm = 1.0;
    s3.tracking.update_interval_s = 1.0;
    s3.tracking.skew_ppm = 2.0;
    CHECK(inter_host_bound_ns(s3, p3, kSecond, 3 * kSecond) == 1'000 + 2'000 + 3'000);

    // Rounds up and saturates.
    ChronyEvidence s4 = zero_evidence();
    s4.tracking.system_time_s = 1.5e-9;
    CHECK(inter_host_bound_ns(s4, zero_evidence(), 0, 0) == 2);
    s4.tracking.system_time_s = 1e12;
    CHECK(inter_host_bound_ns(s4, zero_evidence(), 0, 0) == INT64_MAX / 4); // the saturation value
}

// Follower (PC) root fields 0.5 s dispersion and 0.25 s delay, server's 0.25 s and 0.125 s; all exact in binary
// doubles.
struct RootFieldPair
{
    ChronyEvidence follower = zero_evidence();
    ChronyEvidence server = zero_evidence();

    RootFieldPair()
    {
        follower.tracking.root_dispersion_s = 0.5;
        follower.tracking.root_delay_s = 0.25;
        server.tracking.root_dispersion_s = 0.25;
        server.tracking.root_delay_s = 0.125;
        follower.tracking.ref_time_unix_s = 100.0;
    }

    int64_t bound() const
    {
        return inter_host_bound_ns(follower, server, 0, 0);
    }
};

constexpr int64_t kCancelledBoundNs = 250'000'000 + 62'500'000; // (0.5 - 0.25) + (0.25 - 0.125) / 2
constexpr int64_t kUncancelledBoundNs = 500'000'000 + 125'000'000; // 0.5 + 0.25 / 2

void test_cancellation_requires_a_shared_upstream_sample()
{
    // Server last updated no later than the follower copied it: cancellation as before.
    RootFieldPair same;
    same.server.tracking.ref_time_unix_s = 100.0;
    CHECK(same.bound() == kCancelledBoundNs);
    same.server.tracking.ref_time_unix_s = 40.0;
    CHECK(same.bound() == kCancelledBoundNs);

    // Server updated after the follower: its root fields may differ from the copy, so nothing cancels.
    RootFieldPair newer;
    newer.server.tracking.ref_time_unix_s = 101.0;
    CHECK(newer.bound() == kUncancelledBoundNs);
    CHECK(newer.bound() > same.bound());
    // Growing the newer server's root fields changes nothing: the bound is the follower's full values.
    newer.server.tracking.root_dispersion_s = 0.4;
    newer.server.tracking.root_delay_s = 0.3;
    CHECK(newer.bound() == kUncancelledBoundNs);

    // Unknown server update time: nothing cancels.
    RootFieldPair unknown;
    unknown.server.tracking.ref_time_unix_s = 0.0;
    CHECK(unknown.bound() == kUncancelledBoundNs);

    // Unknown follower update time with a known server one: the server is newer, nothing cancels.
    RootFieldPair no_follower_time;
    no_follower_time.follower.tracking.ref_time_unix_s = 0.0;
    no_follower_time.server.tracking.ref_time_unix_s = 100.0;
    CHECK(no_follower_time.bound() == kUncancelledBoundNs);

    // The other terms are charged on top in both cases.
    Fixture f;
    f.pc->tracking = same.follower.tracking;
    f.pc->tracking.system_time_s = 1e-4;
    f.sender->tracking = same.server.tracking;
    f.sender->tracking.system_time_s = -2e-4;
    f.config.budget_ns = 10 * kSecond;
    CHECK(within(f.run().terms.inter_host_ns, kCancelledBoundNs + 300'000, 2));
    f.sender->tracking.ref_time_unix_s = 101.0;
    CHECK(within(f.run().terms.inter_host_ns, kUncancelledBoundNs + 300'000, 2));
}

void test_newer_server_root_growth_never_shrinks_the_bound()
{
    // Follower evidence fixed; only the server's root fields grow, with a newer update time.
    RootFieldPair original;
    original.server.tracking.ref_time_unix_s = 100.0;
    const int64_t before = original.bound();

    for (double growth : { 0.01, 0.1, 0.25, 0.4, 1.0 })
    {
        RootFieldPair grown;
        grown.server.tracking.ref_time_unix_s = 100.5;
        grown.server.tracking.root_dispersion_s += growth;
        grown.server.tracking.root_delay_s += growth;
        CHECK(grown.bound() >= before);
    }

    // The same growth through qualify_capture.
    Fixture f;
    f.pc->tracking = original.follower.tracking;
    f.sender->tracking = original.server.tracking;
    f.config.budget_ns = 10 * kSecond;
    const int64_t before_run = f.run().terms.inter_host_ns;
    f.sender->tracking.ref_time_unix_s = 100.5;
    f.sender->tracking.root_dispersion_s = 0.6;
    f.sender->tracking.root_delay_s = 0.6;
    CHECK(f.run().terms.inter_host_ns >= before_run);
}

void test_simple_rejections()
{
    {
        Fixture f;
        f.capture.valid = false;
        CHECK(f.run().verdict == Verdict::SenderInvalid);
    }
    {
        Fixture f;
        f.sender.reset();
        CHECK(f.run().verdict == Verdict::NoSenderEvidence);
    }
    {
        Fixture f;
        f.pc.reset();
        CHECK(f.run().verdict == Verdict::NoPcEvidence);
    }
    {
        Fixture f;
        f.clock.usable = false;
        CHECK(f.run().verdict == Verdict::NoLocalClock);
    }
}

void test_evidence_age_boundary()
{
    const int64_t max_age = QualifierConfig{}.max_evidence_age_ns;
    for (bool sender_side : { true, false })
    {
        Fixture f;
        auto& e = sender_side ? f.sender : f.pc;
        e->observed_monotonic_ns = f.now - max_age;
        CHECK(f.run().verdict == Verdict::Qualified);
        e->observed_monotonic_ns = f.now - max_age - 1;
        CHECK(f.run().verdict == Verdict::StaleEvidence);
    }
}

void test_unsynchronised()
{
    {
        Fixture f;
        f.sender->tracking.leap_status = "Not synchronised";
        CHECK(f.run().verdict == Verdict::Unsynchronised);
    }
    {
        Fixture f;
        f.pc->tracking.leap_status = "Not synchronised";
        CHECK(f.run().verdict == Verdict::Unsynchronised);
    }
    {
        Fixture f;
        f.sender->tracking.leap_status = "Insert second";
        f.pc->tracking.leap_status = "Delete second";
        CHECK(f.run().verdict == Verdict::Qualified);
    }
}

void test_reference_matching()
{
    {
        Fixture f;
        f.pc->tracking.reference_id = "DEADBEEF";
        CHECK(f.run().verdict == Verdict::ReferenceMismatch);
    }
    {
        // Ids compare case-insensitively (the header documents upper-case expected ids only).
        Fixture f;
        f.pc->tracking.reference_id = "a9fea97b";
        CHECK(f.run().verdict == Verdict::Qualified);
    }
    {
        Fixture f;
        f.config.expected_sender_reference_id = "a9fea97b";
        CHECK(f.run().verdict == Verdict::Qualified);
    }
}

void test_reference_is_the_pcs_not_the_camera_hosts()
{
    // The camera host's own upstream is not what is checked.
    Fixture g;
    g.sender->tracking.reference_id = "0A0B0C0D";
    CHECK(g.run().verdict == Verdict::Qualified);

    Fixture h;
    h.config.expected_sender_reference_id.clear();
    CHECK(h.run().verdict == Verdict::ReferenceMismatch);

    // Swapped evidence: the PC reports the camera host's upstream, not the camera host.
    Fixture f;
    std::swap(f.sender, f.pc);
    CHECK(f.run().verdict == Verdict::ReferenceMismatch);
}

void test_unsynchronised_pc()
{
    Fixture f;
    f.pc->tracking.leap_status = "Not synchronised";
    CHECK(f.run().verdict == Verdict::Unsynchronised);

    Fixture g;
    g.pc->tracking.stratum = 0;
    CHECK(g.run().verdict == Verdict::Unsynchronised);
}

void test_local_reference_pc_refused()
{
    Fixture f;
    f.pc->tracking.reference_id = std::string(kChronyLocalReferenceId);
    CHECK(f.run().verdict == Verdict::Unsynchronised);
}

void test_local_reference_camera_host_accepted()
{
    // The camera host may serve its own clock; only the PC has to follow it.
    Fixture f;
    f.sender->tracking.reference_id = std::string(kChronyLocalReferenceId);
    CHECK(f.run().verdict == Verdict::Qualified);
}

void test_unsynchronised_camera_host()
{
    Fixture f;
    f.sender->tracking.leap_status = "Not synchronised";
    CHECK(f.run().verdict == Verdict::Unsynchronised);
}

void test_implausible_age()
{
    const QualifierConfig def;
    {
        Fixture f;
        f.set_capture_monotonic(f.now - def.max_capture_age_ns - 1);
        CHECK(f.run().verdict == Verdict::ImplausibleAge);
    }
    {
        Fixture f;
        f.set_capture_monotonic(f.now + def.future_tolerance_ns + 1);
        CHECK(f.run().verdict == Verdict::ImplausibleAge);
    }
    {
        Fixture f;
        f.set_capture_monotonic(f.now + def.future_tolerance_ns / 2);
        const Qualification q = f.run();
        CHECK(q.verdict == Verdict::Qualified);
        CHECK(q.capture_monotonic_ns == f.now + def.future_tolerance_ns / 2);
    }
}

void test_budget_boundary()
{
    Fixture f;
    const int64_t unc = kSensorNs + kPcLocalNs; // chrony terms are zero, so the total is exact integers

    f.config.budget_ns = unc;
    Qualification q = f.run();
    CHECK(q.verdict == Verdict::Qualified);
    CHECK(q.uncertainty_ns == unc);

    f.config.budget_ns = unc - 1;
    q = f.run();
    CHECK(q.verdict == Verdict::OverBudget);
    CHECK(q.uncertainty_ns == unc);
    CHECK(q.terms.sensor_ns == kSensorNs);
    CHECK(q.terms.pc_local_ns == kPcLocalNs);
    CHECK(q.generation == f.clock.generation);
}

void test_usable_verdicts()
{
    const Verdict usable[] = { Verdict::Qualified, Verdict::OverBudget };
    const Verdict unusable[] = { Verdict::SenderInvalid, Verdict::NoSenderEvidence, Verdict::NoPcEvidence,
                                 Verdict::StaleEvidence, Verdict::Unsynchronised,   Verdict::ReferenceMismatch,
                                 Verdict::NoLocalClock,  Verdict::ImplausibleAge };
    for (Verdict v : usable)
    {
        Qualification q;
        q.verdict = v;
        CHECK(q.usable());
    }
    for (Verdict v : unusable)
    {
        Qualification q;
        q.verdict = v;
        CHECK(!q.usable());
    }
}

void test_over_budget_is_usable_with_the_same_instant()
{
    Fixture wide;
    wide.config.budget_ns = 100 * kMs;
    const Qualification ok = wide.run();
    CHECK(ok.verdict == Verdict::Qualified);

    Fixture tight;
    tight.config.budget_ns = 1;
    const Qualification over = tight.run();
    CHECK(over.verdict == Verdict::OverBudget);
    CHECK(over.usable());
    CHECK(over.capture_monotonic_ns == ok.capture_monotonic_ns);
    CHECK(over.capture_monotonic_ns != 0);
    CHECK(over.uncertainty_ns > tight.config.budget_ns);
    CHECK(over.uncertainty_ns == ok.uncertainty_ns);
}

void test_generation_carried_through()
{
    Fixture f;
    f.clock.generation = 3;
    CHECK(f.run().generation == 3);
    // A newer observation is simply reported; the qualifier does not compare generations.
    f.clock.generation = 4;
    f.clock.sample.realtime_ns += 7 * kSecond; // step already folded into the observation's own offset
    f.set_capture_monotonic(f.now - kCaptureAge);
    f.capture.realtime_ns += 7 * kSecond;
    const Qualification q = f.run();
    CHECK(q.verdict == Verdict::Qualified);
    CHECK(q.generation == 4);
    CHECK(q.capture_monotonic_ns == f.now - kCaptureAge);
}

void test_error_bound_rounds_up()
{
    ChronyTracking t;
    t.root_dispersion_s = 1.5e-9;
    CHECK(chrony_error_bound_ns(t, 0) == 2);
    t.root_dispersion_s = 1e-9;
    CHECK(chrony_error_bound_ns(t, 0) == 1);
    t.root_dispersion_s = 0.0;
    CHECK(chrony_error_bound_ns(t, 0) == 0);
}

void test_overflowing_inputs_are_refused()
{
    {
        Fixture f;
        f.sender->observed_monotonic_ns = INT64_MIN; // now - observed overflows
        CHECK(f.run().verdict == Verdict::StaleEvidence);
        f.sender->observed_monotonic_ns = f.now - kSecond;
        f.pc->observed_monotonic_ns = INT64_MIN;
        CHECK(f.run().verdict == Verdict::StaleEvidence);
    }
    {
        Fixture f;
        f.capture.realtime_ns = INT64_MIN; // realtime - offset overflows
        CHECK(f.run().verdict == Verdict::ImplausibleAge);
    }
    {
        Fixture f;
        f.clock.sample.realtime_ns = INT64_MAX; // realtime - monotonic overflows
        f.clock.sample.monotonic_ns = -5;
        CHECK(f.run().verdict == Verdict::ImplausibleAge);
    }
}

void test_newer_generation_is_reported_for_the_caller_to_pin()
{
    // The qualifier does not compare generations: a take must pin the generation of its first frame and end
    // when a later Qualified result carries a different one.
    Fixture f;
    f.clock.generation = 1;
    const Qualification first = f.run();
    f.clock.generation = 2;
    const Qualification later = f.run();
    CHECK(first.verdict == Verdict::Qualified);
    CHECK(later.verdict == Verdict::Qualified);
    CHECK(first.generation == 1);
    CHECK(later.generation == 2);
    CHECK(later.generation != first.generation);
}

void test_pc_local_term_includes_step_tolerance()
{
    Fixture f;
    f.clock.step_tolerance_ns = 70'000;
    const Qualification q = f.run();
    CHECK(q.verdict == Verdict::Qualified);
    CHECK(q.terms.pc_local_ns == kPcLocalNs + 70'000);
    CHECK(q.uncertainty_ns == q.terms.sensor_ns + q.terms.inter_host_ns + q.terms.pc_local_ns);

    f.config.budget_ns = kSensorNs + kPcLocalNs + 70'000 - 1;
    CHECK(f.run().verdict == Verdict::OverBudget);
}

// ref_time is a small whole number of seconds so a 1 ns difference survives the double arithmetic.
constexpr double kRefTimeS = 1000.0;
constexpr int64_t kObservedAtLimit = 1000 * kSecond + 2'048 * kSecond;

void test_source_update_age_gate()
{
    CHECK(QualifierConfig{}.max_source_update_age_ns == 2'048 * kSecond);
    for (bool sender_side : { true, false })
    {
        Fixture f;
        f.sender->tracking.ref_time_unix_s = kRefTimeS;
        f.pc->tracking.ref_time_unix_s = kRefTimeS;
        f.sender->observed_realtime_ns = kObservedAtLimit;
        f.pc->observed_realtime_ns = kObservedAtLimit;
        CHECK(f.run().verdict == Verdict::Qualified);
        auto& e = sender_side ? f.sender : f.pc;
        e->observed_realtime_ns = kObservedAtLimit + 1;
        CHECK(f.run().verdict == Verdict::StaleEvidence);
    }
}

void test_source_update_gate_exemptions()
{
    {
        // A local reference has no source updates.
        Fixture f;
        f.sender->tracking.reference_id = std::string(kChronyLocalReferenceId);
        f.sender->tracking.ref_time_unix_s = 1.0;
        f.sender->observed_realtime_ns = kWallMinusMono;
        CHECK(f.run().verdict == Verdict::Qualified);
    }
    {
        Fixture f;
        f.sender->tracking.ref_time_unix_s = kRefTimeS;
        f.sender->observed_realtime_ns = kObservedAtLimit + 1;
        CHECK(f.run().verdict == Verdict::StaleEvidence);
        f.config.max_source_update_age_ns = 0;
        CHECK(f.run().verdict == Verdict::Qualified);
    }
    {
        Fixture f;
        f.pc->tracking.ref_time_unix_s = kRefTimeS;
        f.pc->observed_realtime_ns = kObservedAtLimit + 1;
        CHECK(f.run().verdict == Verdict::StaleEvidence);
        f.pc->observed_realtime_ns = 0;
        CHECK(f.run().verdict == Verdict::Qualified);
    }
}

void test_future_tolerance_min_behaves_like_zero()
{
    Fixture f;
    f.config.future_tolerance_ns = INT64_MIN;
    f.set_capture_monotonic(f.now + 1);
    CHECK(f.run().verdict == Verdict::ImplausibleAge);
    f.set_capture_monotonic(f.now);
    CHECK(f.run().verdict == Verdict::Qualified);
}

void test_negative_future_tolerance_behaves_like_zero()
{
    Fixture f;
    f.config.future_tolerance_ns = -5;
    f.set_capture_monotonic(f.now + 1);
    CHECK(f.run().verdict == Verdict::ImplausibleAge);
    f.set_capture_monotonic(f.now);
    CHECK(f.run().verdict == Verdict::Qualified);
}

void test_max_future_tolerance_accepts_far_future_capture()
{
    Fixture f;
    f.config.future_tolerance_ns = INT64_MAX;
    f.set_capture_monotonic(f.now + 100 * kSecond);
    CHECK(f.run().verdict == Verdict::Qualified);
}

// Observation exactly 1 s before ref_time, so ref_time is exactly at the default future tolerance.
constexpr int64_t kObservedOneSecBeforeRef = 999 * kSecond;

void test_source_update_future_gate()
{
    CHECK(QualifierConfig{}.source_update_future_tolerance_ns == kSecond);
    for (bool sender_side : { true, false })
    {
        Fixture f;
        f.sender->tracking.ref_time_unix_s = kRefTimeS;
        f.pc->tracking.ref_time_unix_s = kRefTimeS;
        f.sender->observed_realtime_ns = kRefTimeS * kSecond;
        f.pc->observed_realtime_ns = kRefTimeS * kSecond;
        auto& e = sender_side ? f.sender : f.pc;
        e->observed_realtime_ns = kObservedOneSecBeforeRef;
        CHECK(f.run().verdict == Verdict::Qualified);
        e->observed_realtime_ns = kObservedOneSecBeforeRef - 1'000;
        CHECK(f.run().verdict == Verdict::StaleEvidence);
    }
}

void test_source_update_future_gate_exemptions_and_zero_tolerance()
{
    {
        // A local reference is exempt.
        Fixture f;
        f.sender->tracking.reference_id = std::string(kChronyLocalReferenceId);
        f.sender->tracking.ref_time_unix_s = kRefTimeS;
        f.sender->observed_realtime_ns = kSecond;
        CHECK(f.run().verdict == Verdict::Qualified);
    }
    {
        // An unknown observation time skips the gate.
        Fixture f;
        f.sender->tracking.ref_time_unix_s = kRefTimeS;
        f.sender->observed_realtime_ns = 0;
        CHECK(f.run().verdict == Verdict::Qualified);
        f.sender->observed_realtime_ns = -1;
        CHECK(f.run().verdict == Verdict::Qualified);
    }
    for (int64_t tol : { int64_t{ 0 }, int64_t{ -5 }, INT64_MIN })
    {
        Fixture f;
        f.config.source_update_future_tolerance_ns = tol;
        f.sender->tracking.ref_time_unix_s = kRefTimeS;
        f.pc->tracking.ref_time_unix_s = kRefTimeS;
        f.sender->observed_realtime_ns = kRefTimeS * kSecond;
        f.pc->observed_realtime_ns = kRefTimeS * kSecond;
        CHECK(f.run().verdict == Verdict::Qualified);
        f.pc->observed_realtime_ns = kRefTimeS * kSecond - 1'000;
        CHECK(f.run().verdict == Verdict::StaleEvidence);
        f.pc->observed_realtime_ns = kRefTimeS * kSecond;
        f.sender->observed_realtime_ns = kRefTimeS * kSecond - 1'000;
        CHECK(f.run().verdict == Verdict::StaleEvidence);
    }
}

} // namespace

int main()
{
    test_to_string();
    test_chrony_error_bound();
    test_qualified_happy_path();
    test_conversion_uses_pc_offset_not_sample_time();
    test_inter_host_terms();
    test_upstream_cancels();
    test_pc_below_camera_host_clamps_to_zero();
    test_copy_slack();
    test_aging_charged_per_host();
    test_both_system_times_charged();
    test_inter_host_bound_direct();
    test_cancellation_requires_a_shared_upstream_sample();
    test_newer_server_root_growth_never_shrinks_the_bound();
    test_simple_rejections();
    test_evidence_age_boundary();
    test_unsynchronised();
    test_reference_matching();
    test_reference_is_the_pcs_not_the_camera_hosts();
    test_unsynchronised_pc();
    test_local_reference_pc_refused();
    test_local_reference_camera_host_accepted();
    test_unsynchronised_camera_host();
    test_implausible_age();
    test_budget_boundary();
    test_usable_verdicts();
    test_over_budget_is_usable_with_the_same_instant();
    test_generation_carried_through();
    test_error_bound_rounds_up();
    test_overflowing_inputs_are_refused();
    test_newer_generation_is_reported_for_the_caller_to_pin();
    test_pc_local_term_includes_step_tolerance();
    test_source_update_age_gate();
    test_source_update_gate_exemptions();
    test_future_tolerance_min_behaves_like_zero();
    test_negative_future_tolerance_behaves_like_zero();
    test_max_future_tolerance_accepts_far_future_capture();
    test_source_update_future_gate();
    test_source_update_future_gate_exemptions_and_zero_tolerance();
    if (g_failures)
    {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("timesync_capture_qualifier: all checks passed\n");
    return 0;
}
