// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <timesync/capture_qualifier.hpp>
#include <timesync/chrony_tracking.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

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

const char* const kGoodLine =
    "A9FEA97B,169.254.169.123,4,1707763422.384912731,0.000000011,-0.000001098,0.000008212,1.382,0.001,0.019,"
    "0.000289,0.000357,2.0,Normal";

bool near(double a, double b)
{
    return std::fabs(a - b) <= 1e-12 * std::fmax(1.0, std::fabs(b));
}

// Builds a 14-field line with one field replaced (index 0-based).
std::string with_field(int index, const std::string& value)
{
    const char* names[14] = { "A9FEA97B",    "169.254.169.123", "4",           "1707763422.384912731",
                              "0.000000011", "-0.000001098",    "0.000008212", "1.382",
                              "0.001",       "0.019",           "0.000289",    "0.000357",
                              "2.0",         "Normal" };
    std::string out;
    for (int i = 0; i < 14; ++i)
    {
        if (i)
            out += ',';
        out += (i == index) ? value : names[i];
    }
    return out;
}

ChronyTracking make(int stratum, const std::string& id, const std::string& leap)
{
    ChronyTracking t;
    t.reference_id = id;
    t.stratum = stratum;
    t.leap_status = leap;
    return t;
}

void test_valid_line()
{
    auto r = parse_tracking_csv(kGoodLine);
    CHECK(r.error == ChronyParseError::None);
    CHECK(r.tracking.has_value());
    if (!r.tracking)
        return;
    const auto& t = *r.tracking;
    CHECK(t.reference_id == "A9FEA97B");
    CHECK(t.reference_name == "169.254.169.123");
    CHECK(t.stratum == 4);
    CHECK(near(t.ref_time_unix_s, 1707763422.384912731));
    CHECK(near(t.system_time_s, 0.000000011));
    CHECK(near(t.last_offset_s, -0.000001098));
    CHECK(near(t.rms_offset_s, 0.000008212));
    CHECK(near(t.frequency_ppm, 1.382));
    CHECK(near(t.residual_freq_ppm, 0.001));
    CHECK(near(t.skew_ppm, 0.019));
    CHECK(near(t.root_delay_s, 0.000289));
    CHECK(near(t.root_dispersion_s, 0.000357));
    CHECK(near(t.update_interval_s, 2.0));
    CHECK(t.leap_status == "Normal");
    CHECK(is_synchronised(t, false));
}

void test_local_reference_line()
{
    auto r = parse_tracking_csv(
        "7F7F0101,,10,1707763422.000000000,0.000000000,0.000000000,0.000000000,0.000,0.000,0.000,0.000000,"
        "0.000000,64.0,Normal");
    CHECK(r.error == ChronyParseError::None);
    CHECK(r.tracking.has_value());
    if (!r.tracking)
        return;
    CHECK(r.tracking->reference_id == "7F7F0101");
    CHECK(r.tracking->reference_id == kChronyLocalReferenceId);
    CHECK(r.tracking->reference_name.empty());
    CHECK(r.tracking->stratum == 10);
    CHECK(!is_synchronised(*r.tracking, false));
    CHECK(is_synchronised(*r.tracking, true));
}

void test_not_synchronised_line()
{
    auto r = parse_tracking_csv(
        "00000000,,0,0.000000000,0.000000000,0.000000000,0.000000000,0.000,0.000,1000000.000,0.000000,"
        "0.000000,0.0,Not synchronised");
    CHECK(r.error == ChronyParseError::None);
    CHECK(r.tracking.has_value());
    if (!r.tracking)
        return;
    CHECK(r.tracking->leap_status == "Not synchronised");
    CHECK(!is_synchronised(*r.tracking, false));
    CHECK(!is_synchronised(*r.tracking, true));
}

void test_empty_and_field_count()
{
    auto e = parse_tracking_csv("");
    CHECK(e.error == ChronyParseError::Empty);
    CHECK(!e.tracking.has_value());

    const std::string good = kGoodLine;
    auto few = parse_tracking_csv(good.substr(0, good.rfind(',')));
    CHECK(few.error == ChronyParseError::FieldCount);
    CHECK(!few.tracking.has_value());

    auto many = parse_tracking_csv(good + ",extra");
    CHECK(many.error == ChronyParseError::FieldCount);
    CHECK(!many.tracking.has_value());

    auto one = parse_tracking_csv("A9FEA97B");
    CHECK(one.error == ChronyParseError::FieldCount);
}

void test_bad_numbers()
{
    // Every numeric field: non-numeric, NaN and inf are rejected.
    const int numeric[] = { 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    for (int idx : numeric)
    {
        for (const char* bad : { "abc", "nan", "inf", "-inf", "" })
        {
            auto r = parse_tracking_csv(with_field(idx, bad));
            CHECK(r.error == ChronyParseError::BadNumber);
            CHECK(!r.tracking.has_value());
        }
    }
    // Fields that cannot be negative: skew, root delay, root dispersion.
    for (int idx : { 9, 10, 11 })
    {
        auto r = parse_tracking_csv(with_field(idx, "-0.000001"));
        CHECK(r.error == ChronyParseError::BadNumber);
        CHECK(!r.tracking.has_value());
    }
    // Signed fields stay signed.
    CHECK(parse_tracking_csv(with_field(4, "-0.5")).error == ChronyParseError::None);
    CHECK(parse_tracking_csv(with_field(5, "-0.5")).error == ChronyParseError::None);
}

void test_stratum_bounds()
{
    // Header: parse may reject (BadStratum) or accept, but a stratum outside 1..15 is never synchronised.
    for (const char* s : { "0", "16", "17", "255" })
    {
        auto r = parse_tracking_csv(with_field(2, s));
        CHECK(r.error == ChronyParseError::None || r.error == ChronyParseError::BadStratum);
        CHECK(r.tracking.has_value() == (r.error == ChronyParseError::None));
        if (r.tracking)
        {
            CHECK(!is_synchronised(*r.tracking, false));
            CHECK(!is_synchronised(*r.tracking, true));
        }
    }
    for (const char* s : { "-1", "x", "4.5", "" })
    {
        auto r = parse_tracking_csv(with_field(2, s));
        CHECK(r.error != ChronyParseError::None);
        CHECK(!r.tracking.has_value());
    }
    auto ok = parse_tracking_csv(with_field(2, "15"));
    CHECK(ok.error == ChronyParseError::None);
    CHECK(ok.tracking.has_value() && is_synchronised(*ok.tracking, false));
}

void test_line_endings()
{
    for (const char* suffix : { "\n", "\r\n" })
    {
        auto r = parse_tracking_csv(std::string(kGoodLine) + suffix);
        CHECK(r.error == ChronyParseError::None);
        CHECK(r.tracking.has_value());
        if (r.tracking)
            CHECK(r.tracking->leap_status == "Normal");
    }
}

void test_reference_id_case()
{
    auto r = parse_tracking_csv(with_field(0, "a9fea97b"));
    CHECK(r.error == ChronyParseError::None);
    CHECK(r.tracking.has_value() && r.tracking->reference_id == "A9FEA97B");
    auto l = parse_tracking_csv(with_field(0, "7f7f0101"));
    CHECK(l.tracking.has_value() && l.tracking->reference_id == "7F7F0101");
}

void test_is_synchronised()
{
    CHECK(is_synchronised(make(4, "A9FEA97B", "Normal"), false));
    CHECK(is_synchronised(make(4, "A9FEA97B", "Normal"), true));
    CHECK(is_synchronised(make(1, "A9FEA97B", "Normal"), false));
    CHECK(is_synchronised(make(15, "A9FEA97B", "Normal"), false));
    CHECK(!is_synchronised(make(0, "A9FEA97B", "Normal"), false));
    CHECK(!is_synchronised(make(16, "A9FEA97B", "Normal"), false));
    CHECK(is_synchronised(make(4, "A9FEA97B", "Insert second"), false));
    CHECK(is_synchronised(make(4, "A9FEA97B", "Delete second"), false));
    CHECK(!is_synchronised(make(4, "A9FEA97B", "Not synchronised"), false));
    CHECK(!is_synchronised(make(4, "A9FEA97B", "Not synchronised"), true));

    const auto local = make(10, std::string(kChronyLocalReferenceId), "Normal");
    CHECK(!is_synchronised(local, false));
    CHECK(is_synchronised(local, true));
    CHECK(is_synchronised(make(10, std::string(kChronyLocalReferenceId), "Insert second"), true));
    CHECK(!is_synchronised(make(10, std::string(kChronyLocalReferenceId), "Not synchronised"), true));
}

// Captured from `chronyc -c tracking` on chrony 4.5-1ubuntu4.2 while synchronised to an internet NTP server.
void test_real_chrony_4_5_capture()
{
    const char* line =
        "5BBD5B9D,91.189.91.157,3,1794516440.890180266,-0.000037218,-0.000042871,0.001530292,-12.973,-0.003,0.278,"
        "0.055693440,0.001789562,258.4,Normal";
    const ChronyParseResult r = parse_tracking_csv(line);
    CHECK(r.error == ChronyParseError::None);
    CHECK(r.tracking.has_value());
    if (!r.tracking)
        return;
    const ChronyTracking& t = *r.tracking;
    CHECK(t.reference_id == "5BBD5B9D");
    CHECK(t.reference_name == "91.189.91.157");
    CHECK(t.stratum == 3);
    CHECK(near(t.ref_time_unix_s, 1794516440.890180266));
    CHECK(near(t.system_time_s, -0.000037218));
    CHECK(near(t.last_offset_s, -0.000042871));
    CHECK(near(t.rms_offset_s, 0.001530292));
    CHECK(near(t.frequency_ppm, -12.973));
    CHECK(near(t.residual_freq_ppm, -0.003));
    CHECK(near(t.skew_ppm, 0.278));
    CHECK(near(t.root_delay_s, 0.055693440));
    CHECK(near(t.root_dispersion_s, 0.001789562));
    CHECK(near(t.update_interval_s, 258.4));
    CHECK(t.leap_status == "Normal");
    // The id is the upper-case hex of the IPv4 address 91.189.91.157.
    char hex[9];
    std::snprintf(hex, sizeof(hex), "%02X%02X%02X%02X", 91, 189, 91, 157);
    CHECK(t.reference_id == hex);
    CHECK(is_synchronised(t, false));
    // An internet-disciplined host's absolute bound is far above a 5 ms budget, hence the relative composition.
    // 27.84672 ms (root delay / 2) + 1.789562 ms + 0.037218 ms
    const int64_t bound = chrony_error_bound_ns(t, 0);
    CHECK(std::llabs(bound - 29'673'500) <= 1'000);
}

} // namespace

int main()
{
    test_valid_line();
    test_local_reference_line();
    test_not_synchronised_line();
    test_empty_and_field_count();
    test_bad_numbers();
    test_stratum_bounds();
    test_line_endings();
    test_reference_id_case();
    test_is_synchronised();
    test_real_chrony_4_5_capture();
    if (g_failures)
    {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("timesync_chrony_tracking: all checks passed\n");
    return 0;
}
