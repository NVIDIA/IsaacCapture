// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/ego_camera/time_alignment.hpp"

#include <libobsensor/h/Version.h>
#include <nlohmann/json.hpp>
#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <dlfcn.h>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace plugins::ego
{
namespace
{
using Json = nlohmann::json;
constexpr size_t kMaxModelBytes = 8192;
constexpr std::array<const char*, 4> kStreams{ "left_rgb", "right_rgb", "accel", "gyro" };

int64_t integer(const Json& value, const char* name, bool nonnegative = false)
{
    if (!value.is_number_integer() ||
        (value.is_number_unsigned() && value.get<uint64_t>() > static_cast<uint64_t>(INT64_MAX)))
        throw std::invalid_argument(std::string("EGO time alignment requires int64 ") + name);
    const auto result = value.get<int64_t>();
    if (nonnegative && result < 0)
        throw std::invalid_argument(std::string("EGO time alignment requires nonnegative ") + name);
    return result;
}

void text_field(const Json& value, const char* name)
{
    if (!value.is_string() || value.get_ref<const std::string&>().empty())
        throw std::invalid_argument(std::string("EGO time alignment requires nonempty ") + name);
}

Json read_profile(const std::string& path)
{
    if (path.empty())
        return Json();
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::invalid_argument("Unable to open EGO time alignment: " + path);
    std::string bytes(kMaxModelBytes + 1, '\0');
    input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    bytes.resize(static_cast<size_t>(input.gcount()));
    if (bytes.size() > kMaxModelBytes)
        throw std::invalid_argument("EGO capture timing profile exceeds 8192 bytes");
    // Reject duplicate keys so the C++ and offline validators interpret the
    // same timing provenance; JSON's permissive last-key-wins rule is unsafe here.
    std::vector<std::set<std::string>> object_keys;
    auto value = Json::parse(
        bytes,
        [&object_keys](int, Json::parse_event_t event, Json& parsed)
        {
            if (event == Json::parse_event_t::object_start)
                object_keys.emplace_back();
            else if (event == Json::parse_event_t::object_end)
                object_keys.pop_back();
            else if (event == Json::parse_event_t::key && !object_keys.back().insert(parsed.get<std::string>()).second)
                throw std::invalid_argument("Duplicate EGO time alignment JSON key");
            return true;
        });
    if (integer(value.at("version"), "version") != 1 || value.at("scope") != "capture_profile" ||
        value.at("status") != "validated" || value.at("reference_stream") != "left_rgb")
        throw std::invalid_argument(
            "--time-alignment requires a validated version-1 capture_profile, not recording anchors");
    if (value.contains("epochs") || value.contains("host_bridge") || value.contains("audio_segments"))
        throw std::invalid_argument("EGO capture profile cannot reuse recording-bound clock anchors");
    text_field(value.at("model_id"), "model_id");
    for (const auto* key : { "serial", "firmware", "sdk_version", "sdk_binary_sha256" })
        text_field(value.at("identity").at(key), key);
    const auto digest = value.at("identity").at("sdk_binary_sha256").get<std::string>();
    if (digest.size() != 64 || digest.find_first_not_of("0123456789abcdef") != std::string::npos)
        throw std::invalid_argument("EGO SDK binary SHA-256 must contain 64 lowercase hexadecimal characters");
    const auto& selected_profile = value.at("profile");
    const auto& video = selected_profile.at("video");
    const auto& format = video.at("format");
    if (format != "mjpg" && format != "h264" && format != "h265")
        throw std::invalid_argument("EGO timing video format must be mjpg, h264, or h265");
    for (const auto* key : { "width", "height", "fps" })
    {
        const auto number = integer(video.at(key), key, true);
        if (number == 0 || number > UINT32_MAX)
            throw std::invalid_argument(std::string("EGO timing profile requires positive uint32 ") + key);
    }
    integer(selected_profile.at("imu_rate_hz"), "imu_rate_hz", true);
    const auto& audio = selected_profile.at("audio");
    for (const auto* key : { "sample_rate_hz", "channels", "bits_per_sample" })
        integer(audio.at(key), key, true);
    const auto& validation = value.at("validation");
    const auto error = integer(validation.at("max_abs_error_ns"), "max_abs_error_ns", true);
    const auto uncertainty = integer(validation.at("uncertainty_ns"), "uncertainty_ns", true);
    if (static_cast<long double>(error) + uncertainty > 500'000'000.0L / integer(video.at("fps"), "fps"))
        throw std::invalid_argument("EGO timing validation bound exceeds half a video frame");
    const auto held_out_duration = integer(validation.at("held_out_duration_ns"), "held_out_duration_ns", true);
    if (held_out_duration < 900'000'000'000LL)
        throw std::invalid_argument("EGO capture profile requires at least 15 minutes of independent held-out validation");
    const auto max_duration = integer(value.at("max_capture_duration_ns"), "max_capture_duration_ns", true);
    if (max_duration == 0 || max_duration > held_out_duration)
        throw std::invalid_argument(
            "EGO capture duration must be positive and within the independently validated duration");
    text_field(validation.at("method"), "method");
    const auto& evidence = validation.at("evidence");
    if (!evidence.is_array() || evidence.empty())
        throw std::invalid_argument("EGO capture profile requires validation evidence");
    for (const auto& entry : evidence)
        text_field(entry, "evidence");
    const auto& streams = value.at("streams");
    if (!streams.is_object() || !streams.contains("left_rgb"))
        throw std::invalid_argument("EGO capture profile must include the left reference stream");
    for (const auto& [name, mapping] : streams.items())
    {
        if (std::find(kStreams.begin(), kStreams.end(), name) == kStreams.end())
            throw std::invalid_argument("EGO capture profile contains an unsupported stream: " + name);
        integer(mapping.at("offset_ns"), "offset_ns");
        const auto& scale = mapping.at("scale");
        // Drift anchors are recording-specific; rebasing them to an arbitrary
        // future left timestamp would change the calibrated clock relationship.
        if (!scale.is_number() || scale.get<double>() != 1)
            throw std::invalid_argument("Reusable EGO capture profiles require scale=1; fit drift per recording");
        if (mapping.size() != 2)
            throw std::invalid_argument("EGO capture mappings only accept reusable offset_ns and scale");
    }
    if (streams.at("left_rgb").at("offset_ns") != 0 || streams.at("left_rgb").at("scale") != 1)
        throw std::invalid_argument("EGO left reference mapping must be identity");
    return value;
}
} // namespace

void validate_time_alignment_profile(const std::string& path)
{
    read_profile(path);
}

int64_t sdk_timestamp_ns(uint64_t timestamp_us)
{
    // Reject before casting or multiplying; corrupt SDK values must retain
    // their original unsigned value in diagnostics instead of overflowing.
    constexpr auto maximum_us = static_cast<uint64_t>(std::numeric_limits<int64_t>::max() / 1000);
    if (timestamp_us > maximum_us)
        throw std::overflow_error("EGO SDK timestamp_us=" + std::to_string(timestamp_us) +
                                  " exceeds int64 nanosecond range");
    return static_cast<int64_t>(timestamp_us) * 1000;
}

std::string loaded_sdk_binary_sha256()
{
    static const std::string digest = []
    {
        Dl_info library{};
        if (dladdr(reinterpret_cast<const void*>(&ob_get_version), &library) == 0 || !library.dli_fname)
            throw std::runtime_error("Unable to resolve the loaded EGO SDK library for SHA-256 provenance");
        std::ifstream input(library.dli_fname, std::ios::binary);
        if (!input)
            throw std::runtime_error("Unable to read the loaded EGO SDK library for SHA-256 provenance");
        std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(EVP_MD_CTX_new(), EVP_MD_CTX_free);
        if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1)
            throw std::runtime_error("Unable to initialize the EGO SDK SHA-256 digest");
        std::array<char, 16384> bytes{};
        while (input.read(bytes.data(), static_cast<std::streamsize>(bytes.size())) || input.gcount() != 0)
            if (EVP_DigestUpdate(context.get(), bytes.data(), static_cast<size_t>(input.gcount())) != 1)
                throw std::runtime_error("Unable to update the EGO SDK SHA-256 digest");
        if (!input.eof())
            throw std::runtime_error("Unable to finish reading the loaded EGO SDK library");
        std::array<unsigned char, EVP_MAX_MD_SIZE> hash{};
        unsigned int hash_size = 0;
        if (EVP_DigestFinal_ex(context.get(), hash.data(), &hash_size) != 1 || hash_size != 32)
            throw std::runtime_error("Unable to finalize the EGO SDK SHA-256 digest");
        constexpr char hex[] = "0123456789abcdef";
        std::string result;
        result.reserve(hash_size * 2);
        for (unsigned int index = 0; index < hash_size; ++index)
        {
            result.push_back(hex[hash[index] >> 4]);
            result.push_back(hex[hash[index] & 15]);
        }
        return result;
    }();
    return digest;
}

class TimeAlignment::Impl
{
public:
    explicit Impl(const std::string& path) : profile(read_profile(path))
    {
    }
    mutable std::mutex mutex;
    Json profile;
    Json provenance;
    uint32_t epoch = 0;
    std::optional<int64_t> reference_anchor;
    int64_t local_anchor = 0;
    int64_t valid_to = 0;
    bool expired = false;
    std::optional<int64_t> expired_at_reference;
    std::map<std::string, int64_t> first_mapped_raw;
    std::map<std::string, int64_t> last_mapped_local;
    uint64_t audio_segment = 0;
    uint64_t audio_index = 0;
    std::optional<int64_t> audio_previous;
    long double audio_previous_duration = 0;
    bool audio_restart = false;
    bool updated = true;
};

TimeAlignment::TimeAlignment(const std::string& path) : impl_(std::make_unique<Impl>(path))
{
}
TimeAlignment::~TimeAlignment() = default;

void TimeAlignment::configure(const TimingIdentity& value)
{
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const Json identity{ { "serial", value.serial },
                         { "firmware", value.firmware },
                         { "sdk_version", value.sdk_version },
                         { "sdk_binary_sha256", value.sdk_binary_sha256 } };
    const Json profile{
        { "video",
          { { "format", value.video_format }, { "width", value.width }, { "height", value.height }, { "fps", value.fps } } },
        { "imu_rate_hz", value.imu_rate_hz },
        { "audio",
          { { "sample_rate_hz", value.audio_rate_hz },
            { "channels", value.audio_channels },
            { "bits_per_sample", value.audio_bits } } }
    };
    if (!impl_->profile.is_null())
    {
        const auto& mappings = impl_->profile.at("streams");
        if ((value.right_enabled && !mappings.contains("right_rgb")) ||
            (value.imu_rate_hz != 0 && (!mappings.contains("accel") || !mappings.contains("gyro"))))
            throw std::invalid_argument("EGO capture timing profile is missing an enabled stream");
        if (impl_->profile.at("identity") != identity || impl_->profile.at("profile") != profile)
            throw std::invalid_argument(
                "EGO time alignment identity/profile does not match the selected device and streams");
        const auto& validation = impl_->profile.at("validation");
        const long double bound =
            static_cast<long double>(integer(validation.at("max_abs_error_ns"), "max_abs_error_ns")) +
            integer(validation.at("uncertainty_ns"), "uncertainty_ns");
        if (value.fps == 0 || bound > 500'000'000.0L / value.fps)
            throw std::invalid_argument("EGO timing validation bound exceeds half a video frame");
    }
    // Audio getTimeStampUs() semantics depend on the SDK transport; the source
    // label identifies the API, without assuming a USB host-estimated block start.
    impl_->provenance = { { "version", 1 },
                          { "scope", "recording" },
                          { "status", "unverified" },
                          { "model_id", "capture-provenance" },
                          { "reference_stream", "left_rgb" },
                          { "identity", identity },
                          { "profile", profile },
                          { "epochs", Json::array() },
                          { "sources",
                            { { "left_rgb", "sdk_uvc_raw" },
                              { "right_rgb", "sdk_uvc_raw" },
                              { "accel", "sdk_imu_raw" },
                              { "gyro", "sdk_imu_raw" },
                              { "audio", "sdk_audio_raw" } } },
                          { "sample_time_policy", "host_arrival_fallback" } };
    if (!impl_->profile.is_null())
        impl_->provenance["capture_profile"] = impl_->profile;
    impl_->updated = true;
}

void TimeAlignment::begin_epoch(uint32_t epoch)
{
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->epoch = epoch;
    impl_->reference_anchor.reset();
    impl_->first_mapped_raw.clear();
    impl_->last_mapped_local.clear();
    impl_->expired = false;
    impl_->expired_at_reference.reset();
    if (impl_->audio_previous || impl_->audio_index != 0)
        ++impl_->audio_segment;
    impl_->audio_index = 0;
    impl_->audio_previous.reset();
    impl_->audio_restart = epoch != 0;
    impl_->updated = true;
}

bool TimeAlignment::observe_left(int64_t raw_ns, int64_t arrival_ns)
{
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->profile.is_null() || impl_->reference_anchor)
        return false;
    if (raw_ns < 0 || arrival_ns < 0)
        throw std::invalid_argument("EGO time alignment received a negative anchor");
    impl_->reference_anchor = raw_ns;
    impl_->local_anchor = arrival_ns;
    const auto duration = integer(impl_->profile.at("max_capture_duration_ns"), "max_capture_duration_ns");
    impl_->valid_to = raw_ns > INT64_MAX - duration ? INT64_MAX : raw_ns + duration;
    impl_->updated = true;
    return true;
}

int64_t TimeAlignment::sample_time(const std::string& stream, int64_t raw_ns, int64_t arrival_ns)
{
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (!impl_->reference_anchor || impl_->profile.is_null() || !impl_->profile.at("streams").contains(stream))
        return arrival_ns;
    const auto& mapping = impl_->profile.at("streams").at(stream);
    const long double reference_time = static_cast<long double>(raw_ns) + integer(mapping.at("offset_ns"), "offset_ns");
    if (stream == "left_rgb" && raw_ns >= impl_->valid_to && !impl_->expired)
    {
        impl_->expired = true;
        impl_->expired_at_reference = raw_ns;
        impl_->updated = true;
    }
    if (impl_->expired || reference_time < *impl_->reference_anchor || reference_time >= impl_->valid_to)
        return arrival_ns;
    const long double result = static_cast<long double>(impl_->local_anchor) + reference_time - *impl_->reference_anchor;
    const long double rounded = std::round(result);
    if (!std::isfinite(rounded) || rounded < 0 || rounded > std::numeric_limits<int64_t>::max())
        throw std::runtime_error("EGO mapped sample time is outside int64 range");
    const auto mapped = static_cast<int64_t>(rounded);
    // Arrival fallback is a different time source. Comparing it with the first
    // mapped sample can permanently suppress a valid negative-offset mapping.
    const auto previous = impl_->last_mapped_local.find(stream);
    if (previous != impl_->last_mapped_local.end() && mapped <= previous->second)
        throw std::runtime_error("EGO calibrated sample clock regressed within a capture epoch");
    if (impl_->first_mapped_raw.emplace(stream, raw_ns).second)
        impl_->updated = true;
    impl_->last_mapped_local[stream] = mapped;
    return mapped;
}

core::EgoAudioTimingT TimeAlignment::audio_timing(int64_t raw_ns, uint32_t sample_count, uint32_t sample_rate)
{
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (sample_count == 0 || sample_rate == 0)
        throw std::invalid_argument("EGO audio timing requires nonempty PCM and a sample rate");
    bool discontinuity = impl_->audio_restart;
    if (impl_->audio_previous)
    {
        const long double delta = static_cast<long double>(raw_ns) - *impl_->audio_previous;
        // SDK audio time can jitter; only observable large gaps/reversals split a segment.
        if (delta <= 0 || std::abs(delta - impl_->audio_previous_duration) >
                              std::max(5'000'000.0L, 4 * impl_->audio_previous_duration))
        {
            ++impl_->audio_segment;
            impl_->audio_index = 0;
            discontinuity = true;
            impl_->updated = true;
        }
    }
    core::EgoAudioTimingT result;
    result.segment_id = impl_->audio_segment;
    result.first_sample_index = impl_->audio_index;
    result.continuity_known = false;
    result.discontinuity = discontinuity;
    if (sample_count > UINT64_MAX - impl_->audio_index)
        throw std::overflow_error("EGO retained audio sample index overflow");
    impl_->audio_index += sample_count;
    impl_->audio_previous = raw_ns;
    impl_->audio_previous_duration = 1'000'000'000.0L * sample_count / sample_rate;
    impl_->audio_restart = false;
    return result;
}

bool TimeAlignment::has_update() const
{
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->updated;
}

std::string TimeAlignment::snapshot() const
{
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto result = impl_->provenance;
    result["capture_epoch"] = impl_->epoch;
    if (impl_->reference_anchor)
    {
        result["sample_time_policy"] = impl_->expired ? "expired_capture_profile_host_arrival_fallback" :
                                                        "validated_relative_video_imu_with_estimated_host_origin";
        result["host_bridge"] = { { "epoch", impl_->epoch },
                                  { "reference_anchor_ns", *impl_->reference_anchor },
                                  { "local_anchor_ns", impl_->local_anchor },
                                  { "host_anchor_method", "left_arrival_estimate" },
                                  { "host_absolute_accuracy", "unverified" },
                                  { "first_mapped_raw_ns", impl_->first_mapped_raw },
                                  { "valid_from_ns", *impl_->reference_anchor },
                                  { "valid_to_ns", impl_->valid_to },
                                  { "status", impl_->expired ? "expired" : "active" } };
        if (impl_->expired_at_reference)
            result["host_bridge"]["expired_at_reference_ns"] = *impl_->expired_at_reference;
    }
    result["audio_sample_time_policy"] = "host_arrival_fallback";
    result["audio_segment_id"] = impl_->audio_segment;
    const auto json = result.dump();
    if (json.size() > 16 * 1024)
        throw std::runtime_error("EGO timing provenance exceeds the calibration payload budget");
    impl_->updated = false;
    return json;
}

} // namespace plugins::ego
