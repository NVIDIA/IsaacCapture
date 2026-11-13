// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/timesync/capture_qualifier.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>

namespace isaaccapture::timesync
{

namespace
{

constexpr int64_t kSaturated = std::numeric_limits<int64_t>::max() / 4;

int64_t to_ns_saturating(double seconds)
{
    const double ns = seconds * 1e9;
    if (!(ns < static_cast<double>(kSaturated)))
    {
        return kSaturated;
    }
    return static_cast<int64_t>(std::ceil(ns));
}

int64_t add_saturating(int64_t a, int64_t b)
{
    return (a > kSaturated - b) ? kSaturated : a + b;
}

std::string upper(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::toupper(c); });
    return s;
}

// Age of a report; an unrepresentable difference reads as maximal, so it is refused as stale.
int64_t evidence_age_ns(const ChronyEvidence& e, int64_t now_monotonic_ns)
{
    int64_t age = 0;
    return __builtin_sub_overflow(now_monotonic_ns, e.observed_monotonic_ns, &age) ? INT64_MAX : age;
}

bool stale(const ChronyEvidence& e, int64_t now_monotonic_ns, int64_t max_age_ns)
{
    const int64_t age = evidence_age_ns(e, now_monotonic_ns);
    return age < 0 || age > max_age_ns;
}

// True when chronyd's last source update is too old at the moment it was queried.
bool source_quiet(const ChronyEvidence& e, int64_t max_age_ns, int64_t future_tolerance_ns)
{
    if (max_age_ns <= 0 || e.observed_realtime_ns <= 0 || e.tracking.reference_id == kChronyLocalReferenceId)
    {
        return false;
    }
    const double age_s = static_cast<double>(e.observed_realtime_ns) * 1e-9 - e.tracking.ref_time_unix_s;
    // A source update later than the query by more than the tolerance is impossible evidence, not fresh evidence.
    const double lead_s = static_cast<double>(std::max<int64_t>(future_tolerance_ns, 0)) * 1e-9;
    return !(age_s <= static_cast<double>(max_age_ns) * 1e-9 && age_s >= -lead_s); // also true for a NaN
}

} // namespace

const char* to_string(Verdict v)
{
    switch (v)
    {
    case Verdict::Qualified:
        return "qualified";
    case Verdict::SenderInvalid:
        return "sender_invalid";
    case Verdict::NoSenderEvidence:
        return "no_sender_evidence";
    case Verdict::NoPcEvidence:
        return "no_pc_evidence";
    case Verdict::StaleEvidence:
        return "stale_evidence";
    case Verdict::Unsynchronised:
        return "unsynchronised";
    case Verdict::ReferenceMismatch:
        return "reference_mismatch";
    case Verdict::NoLocalClock:
        return "no_local_clock";
    case Verdict::ImplausibleAge:
        return "implausible_age";
    case Verdict::OverBudget:
        return "over_budget";
    }
    return "unknown";
}

int64_t chrony_error_bound_ns(const ChronyTracking& t, int64_t age_ns)
{
    const double age_s = static_cast<double>(std::max<int64_t>(age_ns, 0)) * 1e-9;
    const double bound_s =
        std::fabs(t.system_time_s) + t.root_dispersion_s + 0.5 * t.root_delay_s + t.skew_ppm * 1e-6 * age_s;
    return to_ns_saturating(bound_s); // rounded up, so the bound never understates
}

int64_t inter_host_bound_ns(const ChronyEvidence& follower,
                            const ChronyEvidence& server,
                            int64_t follower_age_ns,
                            int64_t server_age_ns)
{
    const ChronyTracking& s = follower.tracking;
    const ChronyTracking& p = server.tracking;
    // Root delay and dispersion are cumulative along the chain to the reference, so what the follower adds beyond
    // the server's own values is the link between them, and the shared upstream part cancels. That holds only when
    // the follower's copy of the server's values and the server's report describe the same upstream sample: the
    // server has not updated from its own source since the follower last updated from it. A server whose last update
    // is newer than the follower's may have different root values than the ones the follower copied, so subtracting
    // them would shrink the bound without any new measurement of the link. Then nothing cancels and the follower's
    // full root values are charged, which is larger and never smaller than the truth.
    const bool same_upstream_sample = p.ref_time_unix_s > 0.0 && p.ref_time_unix_s <= s.ref_time_unix_s;
    const double link_dispersion_s =
        same_upstream_sample ? std::max(s.root_dispersion_s - p.root_dispersion_s, 0.0) : s.root_dispersion_s;
    const double link_delay_s = same_upstream_sample ? std::max(s.root_delay_s - p.root_delay_s, 0.0) : s.root_delay_s;
    // The follower copied the server's values when it last polled; the server's dispersion has grown since.
    const double copy_slack_s = p.skew_ppm * 1e-6 * s.update_interval_s;
    const double follower_age_s = static_cast<double>(std::max<int64_t>(follower_age_ns, 0)) * 1e-9;
    const double server_age_s = static_cast<double>(std::max<int64_t>(server_age_ns, 0)) * 1e-9;
    const double bound_s = std::fabs(s.system_time_s) + std::fabs(p.system_time_s) + link_dispersion_s +
                           0.5 * link_delay_s + copy_slack_s + s.skew_ppm * 1e-6 * follower_age_s +
                           p.skew_ppm * 1e-6 * server_age_s;
    return to_ns_saturating(bound_s);
}

Qualification qualify_capture(const SenderCapture& capture,
                              const std::optional<ChronyEvidence>& sender_host,
                              const std::optional<ChronyEvidence>& pc_host,
                              const LocalClockObservation& pc_clock,
                              int64_t now_monotonic_ns,
                              const QualifierConfig& config)
{
    Qualification q;
    auto refuse = [&q](Verdict v)
    {
        q.verdict = v;
        return q;
    };

    if (!capture.valid || capture.mapping_uncertainty_ns < 0)
    {
        return refuse(Verdict::SenderInvalid);
    }
    if (!sender_host)
    {
        return refuse(Verdict::NoSenderEvidence);
    }
    if (!pc_host)
    {
        return refuse(Verdict::NoPcEvidence);
    }
    if (stale(*sender_host, now_monotonic_ns, config.max_evidence_age_ns) ||
        stale(*pc_host, now_monotonic_ns, config.max_evidence_age_ns) ||
        source_quiet(*sender_host, config.max_source_update_age_ns, config.source_update_future_tolerance_ns) ||
        source_quiet(*pc_host, config.max_source_update_age_ns, config.source_update_future_tolerance_ns))
    {
        return refuse(Verdict::StaleEvidence);
    }
    // The PC follows the camera host. A local reference on the PC says nothing about the camera host; the camera
    // host may be one, because its own clock is what the PC measures.
    const ChronyEvidence& follower = *pc_host;
    const ChronyEvidence& server = *sender_host;
    if (!is_synchronised(follower.tracking, false) || !is_synchronised(server.tracking, true))
    {
        return refuse(Verdict::Unsynchronised);
    }
    if (config.expected_sender_reference_id.empty() ||
        upper(follower.tracking.reference_id) != upper(config.expected_sender_reference_id))
    {
        return refuse(Verdict::ReferenceMismatch);
    }
    if (!pc_clock.usable)
    {
        return refuse(Verdict::NoLocalClock);
    }

    q.generation = pc_clock.generation;
    int64_t offset = 0;
    int64_t capture_age = 0;
    if (__builtin_sub_overflow(pc_clock.sample.realtime_ns, pc_clock.sample.monotonic_ns, &offset) ||
        __builtin_sub_overflow(capture.realtime_ns, offset, &q.capture_monotonic_ns) ||
        __builtin_sub_overflow(now_monotonic_ns, q.capture_monotonic_ns, &capture_age) ||
        capture_age < -std::max<int64_t>(config.future_tolerance_ns, 0) || capture_age > config.max_capture_age_ns)
    {
        return refuse(Verdict::ImplausibleAge);
    }

    q.terms.sensor_ns = capture.mapping_uncertainty_ns;
    q.terms.inter_host_ns = inter_host_bound_ns(
        follower, server, evidence_age_ns(follower, now_monotonic_ns), evidence_age_ns(server, now_monotonic_ns));
    q.terms.pc_local_ns =
        add_saturating(pc_clock.sample.read_uncertainty_ns, std::max<int64_t>(pc_clock.step_tolerance_ns, 0));
    q.uncertainty_ns = add_saturating(add_saturating(q.terms.sensor_ns, q.terms.inter_host_ns), q.terms.pc_local_ns);
    q.verdict = q.uncertainty_ns > config.budget_ns ? Verdict::OverBudget : Verdict::Qualified;
    return q;
}

} // namespace isaaccapture::timesync
