// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "chrony_tracking.hpp"
#include "local_clock.hpp"

#include <cstdint>
#include <optional>
#include <string>

namespace isaaccapture::timesync
{

/// A chrony report plus when it was read, on the PC's monotonic clock. Evidence for the camera host is stamped
/// on receipt at the PC, so its age includes the delay in getting it here.
struct ChronyEvidence
{
    ChronyTracking tracking;
    int64_t observed_monotonic_ns = 0;
    int64_t observed_realtime_ns = 0; ///< The PC's CLOCK_REALTIME at the same instant; 0 when unknown.
};

/// What the sender claims about one frame: the capture instant in its CLOCK_REALTIME, and how far off its own
/// sensor-to-realtime mapping can be.
struct SenderCapture
{
    int64_t realtime_ns = 0;
    int64_t mapping_uncertainty_ns = 0;
    bool valid = false;
};

struct QualifierConfig
{
    int64_t budget_ns = 5'000'000;
    /// A report older than this is refused, never extrapolated. This bounds how long ago chronyd was asked, not how
    /// long ago it last heard from its source; see max_source_update_age_ns.
    int64_t max_evidence_age_ns = 10'000'000'000;
    /// Refuse when chronyd's last source update (Ref time) is older than this at the time of the query, so frequent
    /// polling cannot hide a reference that went quiet. 0 disables it. A local reference has no source updates and
    /// is exempt. The default is twice chrony's longest default poll interval.
    int64_t max_source_update_age_ns = 2'048'000'000'000;
    /// A source update stamped later than the query by more than this is impossible evidence and is refused. It
    /// allows for the two hosts' clocks disagreeing by a little.
    int64_t source_update_future_tolerance_ns = 1'000'000'000;
    /// A capture older than this, or later than the future tolerance, is implausible; both bounds are inclusive.
    int64_t max_capture_age_ns = 1'000'000'000;
    /// Negative values are treated as 0.
    int64_t future_tolerance_ns = 1'000'000;
    /// Upper-case hex reference id the PC's chronyd must have selected: the camera host. The PC follows the camera
    /// host, which may serve its own clock (chrony `local`) or follow UTC. This is a topology check, not
    /// authentication or a quality measure (an IPv6 source is reported as a 32-bit hash).
    std::string expected_sender_reference_id;
};

enum class Verdict
{
    Qualified,
    SenderInvalid,
    NoSenderEvidence,
    NoPcEvidence,
    StaleEvidence,
    Unsynchronised,
    ReferenceMismatch,
    NoLocalClock, ///< Wide bracket or no sample.
    ImplausibleAge,
    OverBudget,
};

const char* to_string(Verdict v);

struct UncertaintyTerms
{
    int64_t sensor_ns = 0; ///< The sender's own sensor-to-realtime mapping.
    int64_t inter_host_ns = 0; ///< Camera host versus PC wall clock, from both chrony reports and their age.
    int64_t pc_local_ns = 0; ///< The PC's wall/monotonic bracket.
};

struct Qualification
{
    Verdict verdict = Verdict::SenderInvalid;
    /// PC CLOCK_MONOTONIC; meaningful when `usable()`.
    int64_t capture_monotonic_ns = 0;
    /// Sum of the terms, filled whenever `usable()`. It is the honest bound on `capture_monotonic_ns`, whether or not
    /// it met the budget.
    int64_t uncertainty_ns = 0;
    uint64_t generation = 0; ///< The PC local-clock generation the conversion used.
    UncertaintyTerms terms;

    /// Whether `capture_monotonic_ns` is a converted capture instant with a stated bound. `OverBudget` is usable: every
    /// validity check passed and only the precision target was missed, so the instant is still known to within
    /// `uncertainty_ns`. Every other verdict means the conversion could not be vouched for at all. A caller that
    /// records frames publishes the instant and the bound when this holds and `Qualified` only says the bound met
    /// the budget.
    bool usable() const
    {
        return verdict == Verdict::Qualified || verdict == Verdict::OverBudget;
    }
};

/// chrony's bound on a host clock against its reference, |System time| + Root dispersion + Root delay / 2, plus
/// skew * age for the time since the report. Always non-negative; rounded up to the next nanosecond so it never
/// understates, and saturating.
int64_t chrony_error_bound_ns(const ChronyTracking& t, int64_t age_ns);

/// The bound between a following host's wall clock and the host it follows, as `qualify_capture` charges it (see
/// below). The ages are of each report, on the PC's monotonic clock. Rounded up and saturating.
int64_t inter_host_bound_ns(const ChronyEvidence& follower,
                            const ChronyEvidence& server,
                            int64_t follower_age_ns,
                            int64_t server_age_ns);

/// Converts a sender realtime capture stamp to PC monotonic and decides whether it may be recorded.
///
/// The inter-host term bounds sender realtime against PC realtime along camera host system clock -> camera host
/// NTP clock -> PC chronyd's measurement -> PC system clock: both hosts' |System time|, the PC's Root dispersion and
/// Root delay / 2 minus the camera host's (those fields are cumulative to the reference, so the difference is the
/// link between the two and the camera host's own upstream error cancels), a slack for the camera host's dispersion
/// growing since the PC copied it, and each report's skew times its age. The assumptions are that the PC's source is
/// the camera host and that its root fields are cumulative through the camera host's; a differing reference id is
/// refused, and the subtraction is clamped at zero. The bound is derived from chrony's own reports, not measured
/// independently of them.
///
/// `Qualified` carries the PC clock generation. The caller must pin the generation when a take starts and end the
/// take, not continue it, when a later frame reports a different one.
Qualification qualify_capture(const SenderCapture& capture,
                              const std::optional<ChronyEvidence>& sender_host,
                              const std::optional<ChronyEvidence>& pc_host,
                              const LocalClockObservation& pc_clock,
                              int64_t now_monotonic_ns,
                              const QualifierConfig& config);

} // namespace isaaccapture::timesync
