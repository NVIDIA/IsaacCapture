/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Avs1Wire.h"

#include <arpa/inet.h>

#include <cstring>
#include <limits>

namespace argus_sender
{
namespace
{

const uint16_t kKnownBundleFlags =
    AVS1_BUNDLE_SENDER_CLOCK_VALID | AVS1_BUNDLE_SENDER_CLOCK_TAI | AVS1_BUNDLE_SENDER_TSC_VALID;
const uint16_t kKnownStreamFlags = AVS1_STREAM_PRESENT | AVS1_STREAM_STEREO;
const uint16_t kKnownSampleFlags = AVS1_SAMPLE_PRESENT | AVS1_SAMPLE_SOF_TSC_VALID | AVS1_SAMPLE_FRAME_TIMESTAMP_VALID |
                                   AVS1_SAMPLE_SENDER_TIMING_VALID;

bool fail(const char** error, const char* message)
{
    if (error)
        *error = message;
    return false;
}

bool validateSenderTsc(uint16_t bundleFlags, uint64_t senderCaptureReadyTscNs, const char** error)
{
    // Header offset 56 was a reserved zero before SENDER_TSC_VALID existed, so
    // the flag and the field must agree in both directions to stay
    // byte-compatible with older senders and receivers.
    if (bundleFlags & AVS1_BUNDLE_SENDER_TSC_VALID)
    {
        if (!(bundleFlags & AVS1_BUNDLE_SENDER_CLOCK_VALID))
            return fail(error, "AVS1 sender TSC flag requires a valid sender clock");
        if (senderCaptureReadyTscNs == 0)
            return fail(error, "AVS1 sender TSC flag requires a nonzero TSC");
        return true;
    }
    if (senderCaptureReadyTscNs != 0)
        return fail(error, "AVS1 sender TSC requires the sender TSC flag");
    return true;
}

void appendU16(std::vector<uint8_t>* out, uint16_t value)
{
    const uint16_t encoded = htons(value);
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&encoded);
    out->insert(out->end(), bytes, bytes + sizeof(encoded));
}

void appendU32(std::vector<uint8_t>* out, uint32_t value)
{
    const uint32_t encoded = htonl(value);
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&encoded);
    out->insert(out->end(), bytes, bytes + sizeof(encoded));
}

void appendU64(std::vector<uint8_t>* out, uint64_t value)
{
    appendU32(out, static_cast<uint32_t>(value >> 32));
    appendU32(out, static_cast<uint32_t>(value & 0xffffffffULL));
}

bool readU16(const uint8_t* data, size_t size, size_t* offset, uint16_t* value)
{
    if (*offset + 2 > size)
        return false;
    uint16_t encoded = 0;
    std::memcpy(&encoded, data + *offset, sizeof(encoded));
    *value = ntohs(encoded);
    *offset += 2;
    return true;
}

bool readU32(const uint8_t* data, size_t size, size_t* offset, uint32_t* value)
{
    if (*offset + 4 > size)
        return false;
    uint32_t encoded = 0;
    std::memcpy(&encoded, data + *offset, sizeof(encoded));
    *value = ntohl(encoded);
    *offset += 4;
    return true;
}

bool readU64(const uint8_t* data, size_t size, size_t* offset, uint64_t* value)
{
    uint32_t high = 0;
    uint32_t low = 0;
    if (!readU32(data, size, offset, &high) || !readU32(data, size, offset, &low))
    {
        return false;
    }
    *value = (static_cast<uint64_t>(high) << 32) | low;
    return true;
}

bool validateStream(
    const Avs1StreamRecord& stream, size_t streamIndex, size_t* payloadBytes, size_t* sampleCount, const char** error)
{
    if (stream.flags & ~kKnownStreamFlags)
        return fail(error, "unknown AVS1 stream flags");
    const bool stereo = (stream.flags & AVS1_STREAM_STEREO) != 0;
    const size_t expectedSamples = stereo ? 2 : 1;
    if (stream.samples.size() != expectedSamples)
        return fail(error, "AVS1 stream camera count does not match its type");
    if (stream.codec != AVS1_CODEC_HEVC && stream.codec != AVS1_CODEC_RAW)
        return fail(error, "unknown AVS1 codec");
    if (stream.pixelFormat != AVS1_PIXEL_FORMAT_NV12)
        return fail(error, "unknown AVS1 pixel format");
    if (stream.width == 0 || stream.height == 0 || stream.width % 2 != 0 || stream.height % 2 != 0)
    {
        return fail(error, "invalid AVS1 NV12 dimensions");
    }
    if (stream.fpsDenominator == 0)
        return fail(error, "AVS1 FPS denominator is zero");

    const bool streamPresent = (stream.flags & AVS1_STREAM_PRESENT) != 0;
    for (size_t i = 0; i < stream.samples.size(); ++i)
    {
        const Avs1SampleRecord& sample = stream.samples[i];
        if (sample.flags & ~kKnownSampleFlags)
            return fail(error, "unknown AVS1 sample flags");
        if (sample.cameraIndex != i)
            return fail(error, "AVS1 camera indices must follow config order");
        const uint8_t expectedRole = stereo ? (i == 0 ? AVS1_ROLE_LEFT : AVS1_ROLE_RIGHT) : AVS1_ROLE_MONO;
        if (sample.role != expectedRole)
            return fail(error, "AVS1 sample role does not match stream type");
        const bool samplePresent = (sample.flags & AVS1_SAMPLE_PRESENT) != 0;
        if (samplePresent != streamPresent)
            return fail(error, "AVS1 stereo/mono stream presence must be atomic");
        if (samplePresent && sample.payload.empty())
            return fail(error, "present AVS1 sample has an empty payload");
        if (!samplePresent && !sample.payload.empty())
            return fail(error, "absent AVS1 sample has a payload");
        if (stream.codec == AVS1_CODEC_RAW && samplePresent)
        {
            const uint64_t expected = static_cast<uint64_t>(stream.width) * stream.height * 3ULL / 2ULL;
            if (sample.payload.size() != expected)
                return fail(error, "raw AVS1 NV12 payload has the wrong size");
        }
        if (*payloadBytes > std::numeric_limits<uint32_t>::max() - sample.payload.size())
        {
            return fail(error, "AVS1 payload area exceeds 32-bit size");
        }
        *payloadBytes += sample.payload.size();
        ++*sampleCount;
    }
    (void)streamIndex;
    return true;
}

} // namespace

bool buildAvs1WireRecord(const Avs1Bundle& bundle, std::vector<uint8_t>* out, const char** error)
{
    if (error)
        *error = 0;
    if (!out)
        return fail(error, "AVS1 output is null");
    if (bundle.sequence == 0 || bundle.sessionId == 0)
        return fail(error, "AVS1 sequence and session ID must be nonzero");
    if (bundle.flags & ~kKnownBundleFlags)
        return fail(error, "unknown AVS1 bundle flags");
    if ((bundle.flags & AVS1_BUNDLE_SENDER_CLOCK_TAI) && !(bundle.flags & AVS1_BUNDLE_SENDER_CLOCK_VALID))
    {
        return fail(error, "AVS1 TAI flag requires a valid sender clock");
    }
    if (!validateSenderTsc(bundle.flags, bundle.senderCaptureReadyTscNs, error))
    {
        return false;
    }
    if (bundle.streams.empty() || bundle.streams.size() > std::numeric_limits<uint16_t>::max())
    {
        return fail(error, "invalid AVS1 stream count");
    }

    size_t payloadBytes = 0;
    size_t sampleCount = 0;
    for (size_t i = 0; i < bundle.streams.size(); ++i)
    {
        if (!validateStream(bundle.streams[i], i, &payloadBytes, &sampleCount, error))
        {
            return false;
        }
    }
    if (sampleCount > std::numeric_limits<uint16_t>::max())
        return fail(error, "invalid AVS1 sample count");

    const uint64_t descriptorBytes =
        bundle.streams.size() * kAvs1StreamDescriptorBytes + sampleCount * kAvs1SampleDescriptorBytes;
    const uint64_t recordBytes = kAvs1HeaderBytes + descriptorBytes + payloadBytes;
    if (descriptorBytes > std::numeric_limits<uint32_t>::max() || recordBytes > std::numeric_limits<uint32_t>::max())
    {
        return fail(error, "AVS1 record exceeds 32-bit TCP framing");
    }

    out->clear();
    out->reserve(kAvs1LengthPrefixBytes + static_cast<size_t>(recordBytes));
    appendU32(out, static_cast<uint32_t>(recordBytes));
    out->push_back('A');
    out->push_back('V');
    out->push_back('S');
    out->push_back('1');
    appendU16(out, kAvs1Version);
    appendU16(out, bundle.flags);
    appendU32(out, static_cast<uint32_t>(kAvs1HeaderBytes));
    appendU16(out, static_cast<uint16_t>(bundle.streams.size()));
    appendU16(out, static_cast<uint16_t>(sampleCount));
    appendU64(out, bundle.sequence);
    appendU64(out, bundle.sessionId);
    appendU64(out, bundle.senderCaptureReadyNs);
    appendU64(out, bundle.senderTcpEnqueueNs);
    appendU32(out, static_cast<uint32_t>(descriptorBytes));
    appendU32(out, static_cast<uint32_t>(payloadBytes));
    appendU64(out, bundle.senderCaptureReadyTscNs);

    uint16_t firstSample = 0;
    for (size_t i = 0; i < bundle.streams.size(); ++i)
    {
        const Avs1StreamRecord& stream = bundle.streams[i];
        appendU16(out, static_cast<uint16_t>(i));
        appendU16(out, stream.flags);
        out->push_back(stream.codec);
        out->push_back(stream.pixelFormat);
        appendU16(out, stream.rotation);
        appendU32(out, stream.width);
        appendU32(out, stream.height);
        appendU32(out, stream.fpsNumerator);
        appendU32(out, stream.fpsDenominator);
        appendU64(out, stream.sequence);
        appendU16(out, firstSample);
        appendU16(out, static_cast<uint16_t>(stream.samples.size()));
        appendU32(out, 0);
        firstSample += static_cast<uint16_t>(stream.samples.size());
    }

    uint32_t payloadOffset = 0;
    for (size_t streamIndex = 0; streamIndex < bundle.streams.size(); ++streamIndex)
    {
        const Avs1StreamRecord& stream = bundle.streams[streamIndex];
        for (size_t cameraIndex = 0; cameraIndex < stream.samples.size(); ++cameraIndex)
        {
            const Avs1SampleRecord& sample = stream.samples[cameraIndex];
            appendU16(out, static_cast<uint16_t>(streamIndex));
            out->push_back(sample.cameraIndex);
            out->push_back(sample.role);
            appendU16(out, sample.flags);
            appendU16(out, 0);
            appendU32(out, sample.deviceIndex);
            appendU32(out, sample.sensorModeIndex);
            appendU64(out, sample.frameNumber);
            appendU64(out, sample.sofTscNs);
            appendU64(out, sample.frameTimestampNs);
            appendU64(out, sample.senderCaptureReadyNs);
            appendU64(out, sample.senderProcessSubmitNs);
            appendU64(out, sample.senderProcessDoneNs);
            appendU32(out, payloadOffset);
            appendU32(out, static_cast<uint32_t>(sample.payload.size()));
            payloadOffset += static_cast<uint32_t>(sample.payload.size());
        }
    }
    for (size_t streamIndex = 0; streamIndex < bundle.streams.size(); ++streamIndex)
    {
        const Avs1StreamRecord& stream = bundle.streams[streamIndex];
        for (size_t cameraIndex = 0; cameraIndex < stream.samples.size(); ++cameraIndex)
        {
            const std::vector<uint8_t>& payload = stream.samples[cameraIndex].payload;
            out->insert(out->end(), payload.begin(), payload.end());
        }
    }
    return out->size() == kAvs1LengthPrefixBytes + recordBytes;
}

bool parseAvs1WireRecord(const uint8_t* data, size_t size, Avs1Bundle* bundle, const char** error)
{
    if (error)
        *error = 0;
    if (!data || !bundle || size < kAvs1LengthPrefixBytes + kAvs1HeaderBytes)
        return fail(error, "short AVS1 record");
    size_t offset = 0;
    uint32_t recordBytes = 0;
    if (!readU32(data, size, &offset, &recordBytes) || recordBytes != size - kAvs1LengthPrefixBytes)
    {
        return fail(error, "AVS1 length prefix mismatch");
    }
    if (std::memcmp(data + offset, "AVS1", 4) != 0)
        return fail(error, "bad AVS1 magic");
    offset += 4;
    uint16_t version = 0;
    uint16_t streamCount = 0;
    uint16_t sampleCount = 0;
    uint32_t headerBytes = 0;
    uint32_t descriptorBytes = 0;
    uint32_t payloadBytes = 0;
    Avs1Bundle parsed;
    if (!readU16(data, size, &offset, &version) || !readU16(data, size, &offset, &parsed.flags) ||
        !readU32(data, size, &offset, &headerBytes) || !readU16(data, size, &offset, &streamCount) ||
        !readU16(data, size, &offset, &sampleCount) || !readU64(data, size, &offset, &parsed.sequence) ||
        !readU64(data, size, &offset, &parsed.sessionId) || !readU64(data, size, &offset, &parsed.senderCaptureReadyNs) ||
        !readU64(data, size, &offset, &parsed.senderTcpEnqueueNs) || !readU32(data, size, &offset, &descriptorBytes) ||
        !readU32(data, size, &offset, &payloadBytes) || !readU64(data, size, &offset, &parsed.senderCaptureReadyTscNs))
    {
        return fail(error, "short AVS1 header");
    }
    if (version != kAvs1Version || headerBytes != kAvs1HeaderBytes || streamCount == 0)
    {
        return fail(error, "unsupported AVS1 header");
    }
    if (!validateSenderTsc(parsed.flags, parsed.senderCaptureReadyTscNs, error))
    {
        return false;
    }
    const uint64_t expectedDescriptors = static_cast<uint64_t>(streamCount) * kAvs1StreamDescriptorBytes +
                                         static_cast<uint64_t>(sampleCount) * kAvs1SampleDescriptorBytes;
    if (descriptorBytes != expectedDescriptors ||
        static_cast<uint64_t>(headerBytes) + descriptorBytes + payloadBytes != recordBytes)
    {
        return fail(error, "AVS1 descriptor or payload size mismatch");
    }

    struct ParsedSampleLocation
    {
        uint32_t offset = 0;
        uint32_t size = 0;
    };
    std::vector<uint16_t> firstSamples(streamCount);
    std::vector<uint16_t> streamSampleCounts(streamCount);
    parsed.streams.resize(streamCount);
    for (uint16_t i = 0; i < streamCount; ++i)
    {
        uint16_t streamIndex = 0;
        uint32_t reserved32 = 0;
        Avs1StreamRecord& stream = parsed.streams[i];
        if (!readU16(data, size, &offset, &streamIndex) || !readU16(data, size, &offset, &stream.flags) ||
            offset + 2 > size)
        {
            return fail(error, "short AVS1 stream descriptor");
        }
        stream.codec = data[offset++];
        stream.pixelFormat = data[offset++];
        if (!readU16(data, size, &offset, &stream.rotation) || !readU32(data, size, &offset, &stream.width) ||
            !readU32(data, size, &offset, &stream.height) || !readU32(data, size, &offset, &stream.fpsNumerator) ||
            !readU32(data, size, &offset, &stream.fpsDenominator) || !readU64(data, size, &offset, &stream.sequence) ||
            !readU16(data, size, &offset, &firstSamples[i]) || !readU16(data, size, &offset, &streamSampleCounts[i]) ||
            !readU32(data, size, &offset, &reserved32))
        {
            return fail(error, "short AVS1 stream descriptor");
        }
        if (streamIndex != i || reserved32 != 0 ||
            static_cast<uint32_t>(firstSamples[i]) + streamSampleCounts[i] > sampleCount)
        {
            return fail(error, "invalid AVS1 stream descriptor index");
        }
        stream.samples.resize(streamSampleCounts[i]);
    }

    std::vector<ParsedSampleLocation> locations(sampleCount);
    for (uint16_t flat = 0; flat < sampleCount; ++flat)
    {
        uint16_t streamIndex = 0;
        uint16_t flags = 0;
        uint16_t reserved16 = 0;
        uint32_t deviceIndex = 0;
        uint32_t modeIndex = 0;
        uint64_t frameNumber = 0;
        uint64_t sofTscNs = 0;
        uint64_t frameTimestampNs = 0;
        uint64_t captureReadyNs = 0;
        uint64_t processSubmitNs = 0;
        uint64_t processDoneNs = 0;
        if (!readU16(data, size, &offset, &streamIndex) || offset + 2 > size)
            return fail(error, "short AVS1 sample descriptor");
        const uint8_t cameraIndex = data[offset++];
        const uint8_t role = data[offset++];
        if (!readU16(data, size, &offset, &flags) || !readU16(data, size, &offset, &reserved16) ||
            !readU32(data, size, &offset, &deviceIndex) || !readU32(data, size, &offset, &modeIndex) ||
            !readU64(data, size, &offset, &frameNumber) || !readU64(data, size, &offset, &sofTscNs) ||
            !readU64(data, size, &offset, &frameTimestampNs) || !readU64(data, size, &offset, &captureReadyNs) ||
            !readU64(data, size, &offset, &processSubmitNs) || !readU64(data, size, &offset, &processDoneNs) ||
            !readU32(data, size, &offset, &locations[flat].offset) ||
            !readU32(data, size, &offset, &locations[flat].size))
        {
            return fail(error, "short AVS1 sample descriptor");
        }
        if (streamIndex >= streamCount || cameraIndex >= parsed.streams[streamIndex].samples.size() ||
            firstSamples[streamIndex] + cameraIndex != flat || reserved16 != 0)
        {
            return fail(error, "invalid AVS1 sample descriptor index");
        }
        Avs1SampleRecord& sample = parsed.streams[streamIndex].samples[cameraIndex];
        sample.flags = flags;
        sample.cameraIndex = cameraIndex;
        sample.role = role;
        sample.deviceIndex = deviceIndex;
        sample.sensorModeIndex = modeIndex;
        sample.frameNumber = frameNumber;
        sample.sofTscNs = sofTscNs;
        sample.frameTimestampNs = frameTimestampNs;
        sample.senderCaptureReadyNs = captureReadyNs;
        sample.senderProcessSubmitNs = processSubmitNs;
        sample.senderProcessDoneNs = processDoneNs;
    }
    const size_t payloadStart = kAvs1LengthPrefixBytes + headerBytes + descriptorBytes;
    for (uint16_t streamIndex = 0; streamIndex < streamCount; ++streamIndex)
    {
        Avs1StreamRecord& stream = parsed.streams[streamIndex];
        for (size_t cameraIndex = 0; cameraIndex < stream.samples.size(); ++cameraIndex)
        {
            const size_t flat = firstSamples[streamIndex] + cameraIndex;
            const uint64_t end = static_cast<uint64_t>(locations[flat].offset) + locations[flat].size;
            if (end > payloadBytes)
                return fail(error, "AVS1 sample payload exceeds record");
            const uint8_t* begin = data + payloadStart + locations[flat].offset;
            stream.samples[cameraIndex].payload.assign(begin, begin + locations[flat].size);
        }
    }

    size_t validatedPayload = 0;
    size_t validatedSamples = 0;
    for (size_t i = 0; i < parsed.streams.size(); ++i)
    {
        if (!validateStream(parsed.streams[i], i, &validatedPayload, &validatedSamples, error))
        {
            return false;
        }
    }
    if (validatedPayload != payloadBytes || validatedSamples != sampleCount)
        return fail(error, "AVS1 payload coverage mismatch");
    *bundle = std::move(parsed);
    return true;
}

} // namespace argus_sender
