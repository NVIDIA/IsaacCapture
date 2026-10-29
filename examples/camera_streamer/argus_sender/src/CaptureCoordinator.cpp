/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "CaptureCoordinator.h"

#include "Avs1Wire.h"
#include "CameraCatalog.h"
#include "CaptureManager.h"
#include "CapturePriorityPolicy.h"
#include "Config.h"
#include "EGLGlobal.h"
#include "Error.h"
#include "FrameAlignmentPolicy.h"
#include "FrameBundleAssembler.h"
#include "MixedRateScheduler.h"
#include "Runtime.h"
#include "StreamProcessor.h"
#include "TcpTransport.h"

#include <Argus/Argus.h>
#include <EGL/egl.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <memory>
#include <thread>
#include <unistd.h>
#include <vector>

namespace argus_sender
{
namespace
{

const uint32_t kTcpQueueDepth = 1;
const uint32_t kPrioritySyncAttempts = 120;
const uint32_t kPrioritySyncConsecutiveFrames = 3;

bool releaseSecondaryCaptures(CaptureManager* capture, size_t priorityDomainIndex, uint64_t delayNs)
{
    if (!capture)
        return false;
    if (capture->domainCount() <= 1)
        return true;
    if (delayNs > 0)
        std::this_thread::sleep_for(std::chrono::nanoseconds(delayNs));
    for (size_t domainIndex = 0; domainIndex < capture->domainCount(); ++domainIndex)
    {
        if (domainIndex != priorityDomainIndex && !capture->requestDomainCapture(domainIndex))
        {
            return false;
        }
    }
    return true;
}

class PendingCopyScope
{
public:
    explicit PendingCopyScope(std::vector<std::unique_ptr<StreamProcessor>>* processors) : m_processors(processors)
    {
    }

    ~PendingCopyScope()
    {
        if (!m_processors)
            return;
        for (size_t i = 0; i < m_processors->size(); ++i)
            (*m_processors)[i]->discardPendingCopies();
    }

private:
    std::vector<std::unique_ptr<StreamProcessor>>* m_processors;
};

uint64_t createSessionId()
{
    const uint64_t now = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
    return (now << 16) ^ static_cast<uint64_t>(getpid()) ^ 0x4156533100000000ULL;
}

std::vector<Avs1StreamRecord> buildStreamTemplates(const AppConfig& config,
                                                   const CaptureManager& capture,
                                                   const std::vector<uint32_t>& streamFps)
{
    std::vector<Avs1StreamRecord> templates(config.streams.size());
    for (size_t streamIndex = 0; streamIndex < config.streams.size(); ++streamIndex)
    {
        const StreamConfig& configStream = config.streams[streamIndex];
        const ResolvedOutputGeometry& geometry = capture.geometry(streamIndex);
        Avs1StreamRecord& stream = templates[streamIndex];
        if (configStream.type == STREAM_TYPE_STEREO)
            stream.flags |= AVS1_STREAM_STEREO;
        stream.codec = configStream.output.codec == CODEC_HEVC ? AVS1_CODEC_HEVC : AVS1_CODEC_RAW;
        stream.pixelFormat = AVS1_PIXEL_FORMAT_NV12;
        stream.rotation = static_cast<uint16_t>(configStream.output.rotation);
        stream.width = geometry.width;
        stream.height = geometry.height;
        stream.fpsNumerator = streamFps[streamIndex];
        stream.fpsDenominator = 1;
        stream.samples.resize(configStream.cameras.size());
        for (size_t camera = 0; camera < configStream.cameras.size(); ++camera)
        {
            Avs1SampleRecord& sample = stream.samples[camera];
            sample.cameraIndex = static_cast<uint8_t>(camera);
            sample.role = configStream.type == STREAM_TYPE_STEREO ? (camera == 0 ? AVS1_ROLE_LEFT : AVS1_ROLE_RIGHT) :
                                                                    AVS1_ROLE_MONO;
            sample.deviceIndex = configStream.cameras[camera].device;
            sample.sensorModeIndex = configStream.cameras[camera].mode;
        }
    }
    return templates;
}

struct AlignmentFrameLocation
{
    size_t domainIndex = 0;
};

bool alignCapturedFrames(CaptureManager* capture,
                         std::vector<CapturedFrameSet>* captured,
                         bool startup,
                         bool* unavailableWarned)
{
    if (!capture || !captured || captured->size() != capture->streamCount())
    {
        return false;
    }

    const std::chrono::steady_clock::time_point alignStart = std::chrono::steady_clock::now();
    uint32_t drops = 0;
    uint64_t finalSpanNs = 0;
    FrameAlignmentDecision finalDecision = FRAME_ALIGNMENT_KEEP;
    while (true)
    {
        std::vector<FrameAlignmentSample> samples;
        std::vector<AlignmentFrameLocation> locations;
        for (size_t streamIndex = 0; streamIndex < captured->size(); ++streamIndex)
        {
            const CapturedFrameSet& frames = (*captured)[streamIndex];
            if (frames.timings.size() != frames.frames.size())
                return false;
            for (size_t cameraIndex = 0; cameraIndex < frames.timings.size(); ++cameraIndex)
            {
                FrameAlignmentSample sample;
                sample.valid = frames.timings[cameraIndex].hasSensorSofTimestampTsc;
                sample.sofTscNs = frames.timings[cameraIndex].sensorSofTimestampTscNs;
                sample.framePeriodNs = capture->framePeriodNs(streamIndex);
                samples.push_back(sample);
                AlignmentFrameLocation location;
                location.domainIndex = capture->domainFor(streamIndex, cameraIndex);
                locations.push_back(location);
            }
        }

        size_t sampleIndex = 0;
        finalDecision = chooseFrameToReacquire(samples, &sampleIndex, &finalSpanNs);
        if (finalDecision == FRAME_ALIGNMENT_UNAVAILABLE)
        {
            if (!unavailableWarned || !*unavailableWarned)
            {
                std::fprintf(
                    stderr,
                    "Warning: global frame alignment skipped because a due frame lacks a valid SOF TSC timestamp\n");
                if (unavailableWarned)
                    *unavailableWarned = true;
            }
            return true;
        }
        if (finalDecision == FRAME_ALIGNMENT_KEEP || drops >= kMaxFrameAlignmentDrops)
        {
            break;
        }
        if (sampleIndex >= locations.size())
            return false;
        const AlignmentFrameLocation& location = locations[sampleIndex];
        if (!capture->reacquireDomain(location.domainIndex, captured))
        {
            return false;
        }
        ++drops;
    }

    if (verboseLoggingEnabled() && (startup || drops > 0 || finalDecision == FRAME_ALIGNMENT_REACQUIRE))
    {
        const uint64_t elapsedUs = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - alignStart).count());
        std::printf("argus_sender[alignment]: %s dropped=%u, tsc_span_us=%.3f, within_half_frame=%s, reacquire_us=%llu\n",
                    startup ? "startup" : "periodic", drops, static_cast<double>(finalSpanNs) / 1000.0,
                    finalDecision == FRAME_ALIGNMENT_KEEP ? "yes" : "no", static_cast<unsigned long long>(elapsedUs));
    }
    return true;
}

bool captureTscSpanNs(const std::vector<CapturedFrameSet>& captured, uint64_t* spanNs)
{
    if (!spanNs)
        return false;
    uint64_t earliest = 0;
    uint64_t latest = 0;
    size_t samples = 0;
    for (size_t stream = 0; stream < captured.size(); ++stream)
    {
        for (size_t camera = 0; camera < captured[stream].timings.size(); ++camera)
        {
            const FrameTiming& timing = captured[stream].timings[camera];
            if (!timing.hasSensorSofTimestampTsc || timing.sensorSofTimestampTscNs == 0)
            {
                return false;
            }
            if (earliest == 0 || timing.sensorSofTimestampTscNs < earliest)
            {
                earliest = timing.sensorSofTimestampTscNs;
            }
            latest = std::max(latest, timing.sensorSofTimestampTscNs);
            ++samples;
        }
    }
    if (samples == 0)
        return false;
    *spanNs = latest - earliest;
    return true;
}

// The sender TSC "now" is only trusted when it lands shortly after the due
// captures' VI SOF stamps, which proves it is in the same Tegra TSC domain.
// Returns 0 when the check fails so the bundle leaves
// AVS1_BUNDLE_SENDER_TSC_VALID clear. Each verdict is logged once per process.
uint64_t validatedCaptureReadyTscNs(uint64_t tscNowNs, const std::vector<BundleStreamInput>& inputs)
{
    static bool enabledLogged = false;
    static bool disabledLogged = false;
    const uint64_t kMaxSofLagNs = 500000000ULL;
    uint64_t latestSofTscNs = 0;
    for (size_t stream = 0; stream < inputs.size(); ++stream)
    {
        if (!inputs[stream].present)
            continue;
        for (size_t camera = 0; camera < inputs[stream].timings.size(); ++camera)
        {
            const FrameTiming& timing = inputs[stream].timings[camera];
            if (timing.hasSensorSofTimestampTsc)
            {
                latestSofTscNs = std::max(latestSofTscNs, timing.sensorSofTimestampTscNs);
            }
        }
    }
    const bool valid =
        tscNowNs != 0 && latestSofTscNs != 0 && tscNowNs > latestSofTscNs && tscNowNs - latestSofTscNs < kMaxSofLagNs;
    if (!valid)
    {
        if (!disabledLogged)
        {
            disabledLogged = true;
            std::fprintf(
                stderr,
                "argus_sender[coordinator]: sensor TSC domain check failed: now=%llu sof=%llu -> sensor SOF wall stamps disabled\n",
                static_cast<unsigned long long>(tscNowNs), static_cast<unsigned long long>(latestSofTscNs));
        }
        return 0;
    }
    if (!enabledLogged)
    {
        enabledLogged = true;
        std::fprintf(
            stderr,
            "argus_sender[coordinator]: sensor SOF -> capture-ready lag %.1f ms; sensor SOF wall stamps enabled\n",
            static_cast<double>(tscNowNs - latestSofTscNs) / 1e6);
    }
    return tscNowNs;
}

bool frameSetReferenceSofTscNs(const CapturedFrameSet& frames,
                               uint64_t toleranceNs,
                               uint64_t* referenceSofTscNs,
                               uint64_t* spanNs)
{
    if (!referenceSofTscNs || frames.timings.empty())
        return false;
    std::vector<uint64_t> timestamps;
    timestamps.reserve(frames.timings.size());
    for (size_t i = 0; i < frames.timings.size(); ++i)
    {
        const FrameTiming& timing = frames.timings[i];
        if (!timing.hasSensorSofTimestampTsc || timing.sensorSofTimestampTscNs == 0)
        {
            return false;
        }
        timestamps.push_back(timing.sensorSofTimestampTscNs);
    }
    return computeSynchronizedTscReference(timestamps, toleranceNs, referenceSofTscNs, spanNs);
}

} // namespace

bool printCameraModes()
{
    ArgusSamples::EGLDisplayHolder display(true);
    PROPAGATE_ERROR(display.initialize(EGL_DEFAULT_DISPLAY));
    Argus::UniqueObj<Argus::CameraProvider> provider(Argus::CameraProvider::create());
    Argus::ICameraProvider* iProvider = Argus::interface_cast<Argus::ICameraProvider>(provider);
    if (!iProvider)
        ORIGINATE_ERROR("Failed to create CameraProvider");
    if (verboseLoggingEnabled())
        std::printf("Argus Version: %s\n", iProvider->getVersion().c_str());
    const bool ok = listCameraModes(provider.get());
    provider.reset();
    display.cleanup();
    return ok;
}

bool runConfiguredCapture(const AppConfig& config, bool* receiverDisconnected, bool appendDump)
{
    if (receiverDisconnected)
        *receiverDisconnected = false;
    ArgusSamples::EGLDisplayHolder display(true);
    PROPAGATE_ERROR(display.initialize(EGL_DEFAULT_DISPLAY));
    Argus::UniqueObj<Argus::CameraProvider> provider(Argus::CameraProvider::create());
    Argus::ICameraProvider* iProvider = Argus::interface_cast<Argus::ICameraProvider>(provider);
    if (!iProvider)
        ORIGINATE_ERROR("Failed to create CameraProvider");
    std::printf("Argus Version: %s\n", iProvider->getVersion().c_str());

    CaptureManager capture(config);
    if (!capture.initialize(provider.get(), display.get()))
        return false;
    std::vector<uint32_t> streamFps;
    for (size_t i = 0; i < config.streams.size(); ++i)
    {
        const uint32_t fps = config.streams[i].output.fps ? config.streams[i].output.fps : capture.nominalFps(i);
        if (fps == 0)
            ORIGINATE_ERROR("Stream %s resolved to zero FPS", config.streams[i].id.c_str());
        streamFps.push_back(fps);
    }
    if (config.streams[0].output.fps == 0)
    {
        for (size_t i = 1; i < streamFps.size(); ++i)
        {
            if (streamFps[i] != streamFps[0])
            {
                ORIGINATE_ERROR(
                    "Native-rate streams resolved to incompatible FPS values %u and %u; configure explicit FPS",
                    streamFps[0], streamFps[i]);
            }
        }
    }
    MixedRateScheduler scheduler(streamFps);
    const size_t baseStreamIndex = selectPriorityStream(config, streamFps);
    if (config.streams[baseStreamIndex].capture.mode != CAPTURE_MODE_MAILBOX)
    {
        std::fprintf(
            stderr,
            "Configuration error: priority stream %s is configured for %s capture; EGL priority capture requires mailbox mode, while direct-buffer HEVC priority capture has no FIFO mode\n",
            config.streams[baseStreamIndex].id.c_str(), captureModeName(config.streams[baseStreamIndex].capture.mode));
        return false;
    }
    const size_t baseDomainIndex = capture.streamDomain(baseStreamIndex);
    const uint64_t baseFramePeriodNs = capture.framePeriodNs(baseStreamIndex);
    bool hasSecondaryMonoStream = false;
    for (size_t i = 0; i < config.streams.size(); ++i)
    {
        if (i != baseStreamIndex && config.streams[i].type == STREAM_TYPE_MONO)
        {
            hasSecondaryMonoStream = true;
            break;
        }
    }
    const uint64_t secondaryReleaseDelayNs =
        hasSecondaryMonoStream ? 0 : secondaryCaptureReleaseDelayNs(baseFramePeriodNs);
    const uint32_t baseRateLimit = config.streams[baseStreamIndex].output.fps == 0 ? 0 : scheduler.baseFps();
    StereoOutputRateLimiter baseRateLimiter(baseRateLimit);

    std::printf("Configured AVS1 sender: %zu logical stream(s), base cadence %u FPS, priority=%s/domain%zu, logging=%s\n",
                config.streams.size(), scheduler.baseFps(), config.streams[baseStreamIndex].id.c_str(), baseDomainIndex,
                verboseLoggingEnabled() ? "verbose" : "concise");
    double estimatedRawGbps = 0.0;
    for (size_t i = 0; i < config.streams.size(); ++i)
    {
        const StreamConfig& stream = config.streams[i];
        const ResolvedOutputGeometry& geometry = capture.geometry(i);
        if (verboseLoggingEnabled())
        {
            std::printf("  [%zu] %s: %s, %zu camera(s), capture=%s", i, stream.id.c_str(), streamTypeName(stream.type),
                        stream.cameras.size(),
                        stream.output.codec == CODEC_HEVC ? "direct-buffer" : captureModeName(stream.capture.mode));
            if (stream.output.codec != CODEC_HEVC && stream.capture.mode == CAPTURE_MODE_FIFO)
                std::printf(":%u", stream.capture.fifoLength);
            std::printf(", %s %ux%u @ %u FPS, resize=%s, receiver_rotation=%u", codecTypeName(stream.output.codec),
                        geometry.width, geometry.height, streamFps[i], resizeModeName(stream.output.resizeMode),
                        stream.output.rotation);
            if (stream.output.codec == CODEC_HEVC)
                std::printf(", bitrate=%u Mbps per camera", stream.output.bitrateMbps);
            if (stream.output.remap.enabled())
                std::printf(", remap=%s (capture %ux%u)", stream.output.remap.file.c_str(), geometry.captureWidth,
                            geometry.captureHeight);
            std::printf("\n");
        }
        if (stream.output.codec == CODEC_RAW_NV12)
        {
            estimatedRawGbps += static_cast<double>(geometry.width) * geometry.height * 1.5 * streamFps[i] *
                                stream.cameras.size() * 8.0 / 1.0e9;
        }
    }
    if (estimatedRawGbps > 0.9)
    {
        std::fprintf(stderr, "Warning: configured raw NV12 payload rate is approximately %.2f Gbps before TCP overhead\n",
                     estimatedRawGbps);
    }
    if (verboseLoggingEnabled())
    {
        std::printf("Listening for AVS1 receiver on %s:%u\n", config.network.bindHost.c_str(), config.network.port);
    }
    int fd = acceptTcpClient(config.network.bindHost, config.network.port, config.network.reconnectDelayMs);
    if (fd < 0)
        return false;

    TcpPacketSenderSettings transportSettings;
    transportSettings.port = config.network.port;
    transportSettings.dumpWirePath = config.network.dumpWirePath;
    for (size_t i = 0; i < config.streams.size(); ++i)
    {
        transportSettings.streamLabels.push_back(config.streams[i].id);
        for (size_t camera = 0; camera < config.streams[i].cameras.size(); ++camera)
        {
            if (config.streams[i].type == STREAM_TYPE_STEREO)
            {
                transportSettings.sampleLabels.push_back(config.streams[i].id + (camera == 0 ? ".left" : ".right"));
            }
            else
            {
                transportSettings.sampleLabels.push_back(config.streams[i].id);
            }
        }
    }
    transportSettings.kpi = config.runtime.kpi;
    transportSettings.logIntervalSeconds = config.runtime.logIntervalSeconds;
    TcpPacketSender tcpSender(&fd, transportSettings, kTcpQueueDepth, config.network.port, "avs1", appendDump);
    if (!tcpSender.start())
    {
        close(fd);
        return false;
    }

    const std::vector<Avs1StreamRecord> templates = buildStreamTemplates(config, capture, streamFps);
    FrameBundleAssembler assembler(&tcpSender, createSessionId(), templates);
    std::vector<std::unique_ptr<StreamProcessor>> processors;
    for (size_t i = 0; i < config.streams.size(); ++i)
    {
        std::unique_ptr<StreamProcessor> processor(
            new StreamProcessor(config.streams[i], capture.geometry(i), i, config.runtime.kpi,
                                config.runtime.logIntervalSeconds, config.runtime.benchEncode));
        if (!processor->start(&assembler, streamFps[i]))
        {
            tcpSender.shutdown(false);
            close(fd);
            return false;
        }
        processors.push_back(std::move(processor));
    }

    bool ok = capture.startPriorityDomain(baseDomainIndex);
    if (ok)
    {
        ok = capture.discardDomainWarmup(baseDomainIndex, config.alignment.warmupFrames);
    }
    if (ok)
    {
        ok = capture.waitForDomainSynchronization(baseDomainIndex, kPrioritySyncAttempts, kPrioritySyncConsecutiveFrames,
                                                  prioritySyncToleranceNs(baseFramePeriodNs));
    }
    if (ok)
        ok = capture.startSecondaryDomains(baseDomainIndex);
    if (ok && capture.domainCount() > 1)
        ok = capture.discardWarmup(config.alignment.warmupFrames);
    if (ok && capture.domainCount() > 1)
    {
        ok = capture.waitForDomainSynchronization(baseDomainIndex, kPrioritySyncAttempts, kPrioritySyncConsecutiveFrames,
                                                  prioritySyncToleranceNs(baseFramePeriodNs));
    }
    const bool continuousFifoSelection = ok && capture.hasContinuousFifoDomains();
    if (continuousFifoSelection && verboseLoggingEnabled())
    {
        std::printf(
            "argus_sender[alignment]: periodic reacquisition disabled while FIFO domains are active; alignment.interval_frames=%u is inactive and secondary FIFO domains select the closest candidate to every priority SOF TSC\n",
            config.alignment.intervalFrames);
    }

    uint64_t bundleSequence = 0;
    std::vector<uint64_t> streamSequences(config.streams.size(), 0);
    std::vector<uint64_t> intervalStreamFrames(config.streams.size(), 0);
    uint64_t intervalBundles = 0;
    uint64_t intervalBaseRateDrops = 0;
    uint64_t totalBaseRateDrops = 0;
    uint64_t intervalGlobalTscSamples = 0;
    double intervalGlobalTscSpanSumNs = 0.0;
    uint64_t intervalGlobalTscSpanMaxNs = 0;
    bool alignmentTimestampUnavailableWarned = false;
    std::chrono::steady_clock::time_point statsStart = std::chrono::steady_clock::now();
    while (ok && !stopRequested() && !tcpSender.hasError() && !assembler.hasError() && !capture.hasError() &&
           (config.runtime.frames == 0 || bundleSequence < config.runtime.frames))
    {
        std::vector<CapturedFrameSet> captured;
        PendingCopyScope pendingCopyScope(&processors);
        capture.prepareCaptured(&captured);
        std::vector<BundleStreamInput> bundleInputs(config.streams.size());
        if (!capture.acquireDomain(baseDomainIndex, &captured))
        {
            ok = false;
            break;
        }
        uint64_t priorityReferenceSofTscNs = 0;
        uint64_t priorityTscSpanNs = 0;
        if (!frameSetReferenceSofTscNs(captured[baseStreamIndex], prioritySyncToleranceNs(baseFramePeriodNs),
                                       &priorityReferenceSofTscNs, &priorityTscSpanNs))
        {
            std::fprintf(stderr,
                         "Priority stream %s is not synchronized at runtime: tsc_span_us=%.3f, tolerance_us=%.3f\n",
                         config.streams[baseStreamIndex].id.c_str(), static_cast<double>(priorityTscSpanNs) / 1000.0,
                         static_cast<double>(prioritySyncToleranceNs(baseFramePeriodNs)) / 1000.0);
            ok = false;
            break;
        }
        if (!baseRateLimiter.shouldOutput(captured[baseStreamIndex].timings[0]))
        {
            ++intervalBaseRateDrops;
            ++totalBaseRateDrops;
            if (!releaseSecondaryCaptures(&capture, baseDomainIndex, 0))
            {
                ok = false;
                break;
            }
            for (size_t domainIndex = 0; domainIndex < capture.domainCount(); ++domainIndex)
            {
                if (domainIndex != baseDomainIndex && !capture.domainUsesContinuousFifo(domainIndex) &&
                    !capture.acquireDomain(domainIndex, &captured))
                {
                    ok = false;
                    break;
                }
            }
            if (!ok)
                break;
            for (size_t i = 0; i < config.streams.size(); ++i)
                capture.recordCaptured(i, captured[i]);
            continue;
        }

        const std::vector<bool> due = scheduler.next();
        if (!due[baseStreamIndex])
        {
            std::fprintf(stderr, "Internal error: AVS1 base stream is not due\n");
            ok = false;
            break;
        }
        for (size_t i = 0; i < config.streams.size(); ++i)
        {
            if (!due[i])
                continue;
            if (!processors[i]->waitForInputBuffers())
            {
                ok = false;
                break;
            }
            ++streamSequences[i];
            bundleInputs[i].present = true;
            bundleInputs[i].sequence = streamSequences[i];
        }
        if (!ok)
            break;

        const uint64_t nextBundleSequence = bundleSequence + 1;
        const bool alignmentCycle = config.alignment.intervalFrames > 0 && !continuousFifoSelection &&
                                    nextBundleSequence % config.alignment.intervalFrames == 0;
        std::vector<bool> copiesSubmitted(config.streams.size(), false);
        bool priorityCopyStarted = false;

        if (!alignmentCycle)
        {
            for (size_t i = 0; i < config.streams.size(); ++i)
            {
                if (due[i] && config.streams[i].output.codec == CODEC_HEVC && capture.streamDomain(i) == baseDomainIndex)
                {
                    if (!processors[i]->submitCopies(&captured[i], nextBundleSequence))
                    {
                        ok = false;
                        break;
                    }
                    copiesSubmitted[i] = true;
                    priorityCopyStarted = true;
                }
            }
        }
        if (ok && !releaseSecondaryCaptures(&capture, baseDomainIndex, priorityCopyStarted ? secondaryReleaseDelayNs : 0))
        {
            ok = false;
        }

        for (size_t domainIndex = 0; ok && domainIndex < capture.domainCount(); ++domainIndex)
        {
            if (domainIndex == baseDomainIndex)
                continue;
            const bool acquired = capture.domainUsesContinuousFifo(domainIndex) ?
                                      capture.acquireDomainClosestTo(domainIndex, priorityReferenceSofTscNs, &captured) :
                                      capture.acquireDomain(domainIndex, &captured);
            if (!acquired)
            {
                ok = false;
                break;
            }
            if (!alignmentCycle)
            {
                for (size_t i = 0; i < config.streams.size(); ++i)
                {
                    if (due[i] && !copiesSubmitted[i] && config.streams[i].output.codec == CODEC_HEVC &&
                        capture.streamDomain(i) == domainIndex)
                    {
                        if (!processors[i]->submitCopies(&captured[i], nextBundleSequence))
                        {
                            ok = false;
                            break;
                        }
                        copiesSubmitted[i] = true;
                    }
                }
            }
        }
        if (!ok)
            break;

        if (alignmentCycle && !alignCapturedFrames(&capture, &captured, false, &alignmentTimestampUnavailableWarned))
        {
            ok = false;
            break;
        }
        for (size_t i = 0; i < config.streams.size(); ++i)
        {
            capture.recordCaptured(i, captured[i]);
            if (due[i])
                bundleInputs[i].timings = captured[i].timings;
        }
        uint64_t globalTscSpanNs = 0;
        if (captureTscSpanNs(captured, &globalTscSpanNs))
        {
            ++intervalGlobalTscSamples;
            intervalGlobalTscSpanSumNs += static_cast<double>(globalTscSpanNs);
            intervalGlobalTscSpanMaxNs = std::max(intervalGlobalTscSpanMaxNs, globalTscSpanNs);
        }

        uint32_t captureClockDomain = SENDER_CLOCK_UNKNOWN;
        const uint64_t senderCaptureReadyNs = readSenderClockNs(&captureClockDomain);
        // Same instant as senderCaptureReadyNs so receivers can place each
        // sample's VI SOF TSC on the sender wall clock.
        const uint64_t tscNowNs = readTscNowNs();
        const uint64_t senderCaptureReadyTscNs = validatedCaptureReadyTscNs(tscNowNs, bundleInputs);
        uint32_t submitClockDomain = SENDER_CLOCK_UNKNOWN;
        const uint64_t senderProcessSubmitNs = readSenderClockNs(&submitClockDomain);
        if (captureClockDomain != submitClockDomain)
            captureClockDomain = SENDER_CLOCK_UNKNOWN;
        if (!assembler.beginBundle(nextBundleSequence, bundleInputs, senderCaptureReadyNs, senderProcessSubmitNs,
                                   captureClockDomain, senderCaptureReadyTscNs))
        {
            ok = false;
            break;
        }
        // Submit every due stream's copies before waiting on any of them: the
        // raw streams' whole pipeline (VIC copy + remap/pack) runs on their
        // per-camera worker threads, so starting them here overlaps that work
        // with the head VIC copies and NVENC queueing instead of serializing
        // the two stages inside one bundle tick.
        for (size_t i = 0; i < config.streams.size(); ++i)
        {
            if (due[i] && config.streams[i].output.codec == CODEC_HEVC && !copiesSubmitted[i] &&
                !processors[i]->submitCopies(&captured[i], nextBundleSequence))
            {
                ok = false;
                break;
            }
        }
        for (size_t i = 0; ok && i < config.streams.size(); ++i)
        {
            if (due[i] && config.streams[i].output.codec == CODEC_RAW_NV12 &&
                !processors[i]->submitCopies(&captured[i], nextBundleSequence))
            {
                ok = false;
            }
        }
        for (size_t i = 0; ok && i < config.streams.size(); ++i)
        {
            if (due[i] && config.streams[i].output.codec == CODEC_HEVC && !processors[i]->finishSubmit())
                ok = false;
        }
        for (size_t i = 0; ok && i < config.streams.size(); ++i)
        {
            if (due[i] && config.streams[i].output.codec == CODEC_RAW_NV12 && !processors[i]->finishSubmit())
            {
                ok = false;
            }
        }
        if (!ok)
            break;
        bundleSequence = nextBundleSequence;
        ++intervalBundles;
        for (size_t i = 0; i < due.size(); ++i)
            intervalStreamFrames[i] += due[i] ? 1 : 0;

        if (config.runtime.kpi && config.runtime.logIntervalSeconds > 0)
        {
            const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
            const double elapsed = std::chrono::duration<double>(now - statsStart).count();
            if (elapsed >= static_cast<double>(config.runtime.logIntervalSeconds))
            {
                std::printf("argus_sender[coordinator]: %.2f bundle FPS, total=%llu, streams=",
                            static_cast<double>(intervalBundles) / elapsed,
                            static_cast<unsigned long long>(bundleSequence));
                for (size_t i = 0; i < intervalStreamFrames.size(); ++i)
                {
                    std::printf("%s%s:%.2f", i == 0 ? "" : ",", config.streams[i].id.c_str(),
                                static_cast<double>(intervalStreamFrames[i]) / elapsed);
                    intervalStreamFrames[i] = 0;
                }
                if (intervalGlobalTscSamples > 0)
                {
                    std::printf(", global_tsc_span_us(avg/max)=%.3f/%.3f",
                                intervalGlobalTscSpanSumNs / static_cast<double>(intervalGlobalTscSamples) / 1000.0,
                                static_cast<double>(intervalGlobalTscSpanMaxNs) / 1000.0);
                }
                else
                {
                    std::printf(", global_tsc_span_us=unavailable");
                }
                std::printf(", base_rate_drops(interval/total)=%llu/%llu\n",
                            static_cast<unsigned long long>(intervalBaseRateDrops),
                            static_cast<unsigned long long>(totalBaseRateDrops));
                capture.logFifoCandidateStats();
                intervalBundles = 0;
                intervalBaseRateDrops = 0;
                intervalGlobalTscSamples = 0;
                intervalGlobalTscSpanSumNs = 0.0;
                intervalGlobalTscSpanMaxNs = 0;
                statsStart = now;
            }
        }
    }

    if (!capture.stop())
        ok = false;
    const bool disconnected = tcpSender.disconnected() && !stopRequested();
    if (!disconnected)
    {
        for (size_t i = 0; i < processors.size(); ++i)
        {
            if (!processors[i]->drain())
                ok = false;
        }
        for (size_t i = 0; i < processors.size(); ++i)
            processors[i]->sendEos();
    }
    bool processorError = false;
    for (size_t i = 0; i < processors.size(); ++i)
    {
        processorError = processorError || processors[i]->hasError();
        processors[i]->shutdown();
    }
    tcpSender.shutdown(ok && !processorError && !stopRequested());
    if (fd >= 0)
        close(fd);
    processors.clear();
    capture.releaseResources();
    provider.reset();
    display.cleanup();
    if (receiverDisconnected)
        *receiverDisconnected = disconnected;
    return disconnected || (ok && !processorError && !assembler.hasError());
}

} // namespace argus_sender
