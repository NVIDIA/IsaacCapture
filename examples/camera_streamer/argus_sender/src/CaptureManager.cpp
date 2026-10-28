/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "CaptureManager.h"

#include "CameraCatalog.h"
#include "CaptureDomainPlan.h"
#include "CapturePriorityPolicy.h"
#include "Error.h"
#include "FrameAlignmentPolicy.h"
#include "HevcEncoder.h"
#include "PriorityFrameHandoff.h"
#include "Runtime.h"
#include "StereoAlignmentPolicy.h"

#include <Argus/BufferStream.h>
#include <Argus/EGLImage.h>
#include <Argus/Event.h>
#include <Argus/EventProvider.h>
#include <Argus/EventQueue.h>
#include <Argus/Settings.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <future>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace argus_sender
{

CapturedDmaBuffer::CapturedDmaBuffer() : m_buffer(NULL)
{
}

CapturedDmaBuffer::~CapturedDmaBuffer()
{
    reset();
}

CapturedDmaBuffer::CapturedDmaBuffer(CapturedDmaBuffer&& other) noexcept : m_buffer(other.release())
{
}

CapturedDmaBuffer& CapturedDmaBuffer::operator=(CapturedDmaBuffer&& other) noexcept
{
    if (this != &other)
    {
        reset();
        m_buffer = other.release();
    }
    return *this;
}

DmaBuffer* CapturedDmaBuffer::get() const
{
    return m_buffer;
}

DmaBuffer* CapturedDmaBuffer::release()
{
    DmaBuffer* buffer = m_buffer;
    m_buffer = NULL;
    return buffer;
}

void CapturedDmaBuffer::reset(DmaBuffer* buffer)
{
    if (m_buffer && m_buffer != buffer && !m_buffer->releaseToArgus())
    {
        std::fprintf(stderr, "Failed to release unused direct-capture DMABUF to Argus\n");
    }
    m_buffer = buffer;
}

namespace
{

const uint64_t kCaptureWaitTimeoutNs = 100000000ULL;
const uint64_t kIdleWaitTimeoutNs = 5000000000ULL;
const uint32_t kWaitPollMs = 100;
const size_t kFifoCandidateWindowLength = 2;
// Use the same direct-output pool size for head and wrist cameras. Continuous
// acquisition and latest-frame selection keep excess filled buffers drained.
const uint32_t kDirectCaptureBufferCount = 10;

uint64_t steadyNowNs()
{
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

int64_t deltaNs(uint64_t right, uint64_t left)
{
    if (right >= left)
    {
        const uint64_t value = right - left;
        return value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ? std::numeric_limits<int64_t>::max() :
                                                                                    static_cast<int64_t>(value);
    }
    const uint64_t value = left - right;
    return value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ? std::numeric_limits<int64_t>::min() :
                                                                                -static_cast<int64_t>(value);
}

uint64_t absoluteDeltaNs(int64_t value)
{
    if (value >= 0)
        return static_cast<uint64_t>(value);
    return static_cast<uint64_t>(-(value + 1)) + 1ULL;
}

void updateAtomicMaximum(std::atomic<uint64_t>* target, uint64_t value)
{
    uint64_t current = target->load();
    while (current < value && !target->compare_exchange_weak(current, value))
    {
    }
}

} // namespace

class CaptureManager::Impl
{
public:
    struct StreamState
    {
        StreamConfig config;
        ResolvedOutputGeometry geometry;
        std::vector<size_t> cameraIndices;
        uint32_t nominalFps = 0;
        uint64_t framePeriodNs = 0;
        std::chrono::steady_clock::time_point intervalStart;
        uint64_t intervalFrames = 0;
        uint64_t intervalTscSamples = 0;
        double intervalTscSumNs = 0.0;
        int64_t intervalTscMinNs = 0;
        int64_t intervalTscMaxNs = 0;
        uint64_t totalFrames = 0;
        std::vector<uint64_t> acquireSamples;
        std::vector<double> acquireSumNs;
        std::vector<uint64_t> acquireMaxNs;
        std::vector<uint64_t> lastSelectedFrameNumbers;
        std::vector<uint64_t> selectedFrameGapsInterval;
        std::vector<bool> hasLastSelectedFrameNumber;
    };

    struct CameraState
    {
        size_t streamIndex = 0;
        size_t cameraIndex = 0;
        size_t domainIndex = 0;
        CameraConfig config;
        std::string label;
        Argus::CameraDevice* device = NULL;
        Argus::SensorMode* sensorMode = NULL;
        Argus::ISensorMode* iSensorMode = NULL;
        Argus::UniqueObj<Argus::OutputStream> outputStream;
        std::vector<std::unique_ptr<DmaBuffer>> dmaBuffers;
        std::vector<Argus::UniqueObj<Argus::Buffer>> argusBuffers;
        Argus::IBufferOutputStream* iBufferStream = NULL;
        Argus::UniqueObj<EGLStream::FrameConsumer> consumer;
        EGLStream::IFrameConsumer* iConsumer = NULL;
        uint64_t lastFrameNumber = 0;
        bool hasLastFrameNumber = false;
    };

    struct PendingFrame
    {
        explicit PendingFrame(size_t index) : cameraFlatIndex(index)
        {
        }

        size_t cameraFlatIndex;
        Argus::UniqueObj<EGLStream::Frame> frame;
        CapturedDmaBuffer dmaBuffer;
        FrameTiming timing;
        bool duplicate = false;
    };

    struct CaptureSessionState
    {
        std::vector<size_t> cameraIndices;
        Argus::UniqueObj<Argus::CaptureSession> session;
        Argus::ICaptureSession* iSession = NULL;
        Argus::IEventProvider* iEventProvider = NULL;
        Argus::UniqueObj<Argus::EventQueue> eventQueue;
        Argus::IEventQueue* iEventQueue = NULL;
        Argus::UniqueObj<Argus::Request> request;
        uint64_t captureCompleteCredits = 0;
        bool repeating = false;
    };

    struct DomainState
    {
        size_t index = 0;
        std::vector<size_t> cameraIndices;
        std::vector<std::unique_ptr<CaptureSessionState>> sessions;
        std::thread worker;
        std::thread eventWorker;
        std::array<std::thread, 2> cameraWorkers;
        std::array<LatestFrameSlot<PendingFrame>, 2> headFrames;
        mutable std::mutex mutex;
        std::condition_variable slotAvailable;
        std::condition_variable framesAvailable;
        std::vector<std::unique_ptr<PendingFrame>> pending;
        std::deque<std::vector<std::unique_ptr<PendingFrame>>> candidates;
        bool hasPending = false;
        bool stop = false;
        bool error = false;
        bool demandDriven = false;
        bool fifoMode = false;
        uint32_t fifoLength = 0;
        bool continuousFifo = false;
        bool captureRequested = false;
        bool selectionTimestampUnavailableWarned = false;
        bool priorityStereo = false;
        size_t priorityStereoStreamIndex = std::numeric_limits<size_t>::max();
        size_t stereoLeftFlatIndex = std::numeric_limits<size_t>::max();
        size_t stereoRightFlatIndex = std::numeric_limits<size_t>::max();
        std::atomic<bool> stoppingRepeat{ false };
        std::atomic<uint64_t> candidateOverflowDiscardsInterval{ 0 };
        std::atomic<uint64_t> candidateOverflowDiscardsTotal{ 0 };
        std::atomic<uint64_t> candidateSelectionDiscardsInterval{ 0 };
        std::atomic<uint64_t> candidateSelectionDiscardsTotal{ 0 };
        std::atomic<uint64_t> candidateSelectionsInterval{ 0 };
        std::atomic<uint64_t> candidateSelectionsTotal{ 0 };
        std::atomic<uint64_t> candidateDeltaSumNsInterval{ 0 };
        std::atomic<uint64_t> candidateDeltaMaxNsInterval{ 0 };
        std::atomic<uint64_t> candidatePastSelectionsInterval{ 0 };
        std::atomic<uint64_t> candidateFutureSelectionsInterval{ 0 };
        std::atomic<uint64_t> candidateDepthPeakInterval{ 0 };
        std::atomic<uint64_t> fifoEventBacklog{ 0 };
        std::atomic<uint64_t> fifoEventBacklogPeakInterval{ 0 };
        std::atomic<uint64_t> stereoPairsInterval{ 0 };
        std::atomic<uint64_t> stereoPairsTotal{ 0 };
        std::atomic<uint64_t> stereoRepairPairsInterval{ 0 };
        std::atomic<uint64_t> stereoRepairPairsTotal{ 0 };
        std::atomic<uint64_t> stereoLeftDropsInterval{ 0 };
        std::atomic<uint64_t> stereoLeftDropsTotal{ 0 };
        std::atomic<uint64_t> stereoRightDropsInterval{ 0 };
        std::atomic<uint64_t> stereoRightDropsTotal{ 0 };
        std::atomic<uint64_t> stereoReacquiresInterval{ 0 };
        std::atomic<uint64_t> stereoReacquiresTotal{ 0 };
        std::atomic<uint64_t> stereoReacquireWaitSumNsInterval{ 0 };
        std::atomic<uint64_t> stereoReacquireWaitMaxNsInterval{ 0 };
        std::atomic<uint64_t> stereoSelectedAgeSumNsInterval{ 0 };
        std::atomic<uint64_t> stereoSelectedAgeMaxNsInterval{ 0 };
        std::atomic<uint64_t> stereoInitialDeltaSumNsInterval{ 0 };
        std::atomic<uint64_t> stereoInitialDeltaMaxNsInterval{ 0 };
        std::atomic<uint64_t> stereoFinalDeltaSumNsInterval{ 0 };
        std::atomic<uint64_t> stereoFinalDeltaMaxNsInterval{ 0 };
        std::atomic<uint64_t> headSlotOverwritesInterval{ 0 };
        std::atomic<uint64_t> headSlotOverwritesTotal{ 0 };
    };

    explicit Impl(const AppConfig& config) : m_config(config), m_started(false), m_error(false)
    {
    }

    ~Impl()
    {
        stop();
        cleanup();
    }

    bool initialize(Argus::CameraProvider* provider, EGLDisplay display)
    {
        if (!provider || display == EGL_NO_DISPLAY)
            ORIGINATE_ERROR("Capture manager got invalid Argus/EGL context");

        Argus::ICameraProvider* iProvider = Argus::interface_cast<Argus::ICameraProvider>(provider);
        if (!iProvider)
            ORIGINATE_ERROR("Failed to get CameraProvider interface");

        std::vector<CaptureEndpointDescriptor> endpoints;
        m_streams.resize(m_config.streams.size());
        m_cameraLookup.resize(m_config.streams.size());
        for (size_t streamIndex = 0; streamIndex < m_config.streams.size(); ++streamIndex)
        {
            m_streams[streamIndex].reset(new StreamState());
            StreamState& stream = *m_streams[streamIndex];
            stream.config = m_config.streams[streamIndex];
            stream.intervalStart = std::chrono::steady_clock::now();
            m_cameraLookup[streamIndex].resize(stream.config.cameras.size());
            std::vector<ImageSize> sourceSizes;
            for (size_t cameraIndex = 0; cameraIndex < stream.config.cameras.size(); ++cameraIndex)
            {
                std::unique_ptr<CameraState> camera(new CameraState());
                camera->streamIndex = streamIndex;
                camera->cameraIndex = cameraIndex;
                camera->config = stream.config.cameras[cameraIndex];
                std::ostringstream label;
                label << stream.config.id << ".";
                if (stream.config.type == STREAM_TYPE_STEREO)
                    label << (cameraIndex == 0 ? "left" : "right");
                else
                    label << "mono";
                camera->label = label.str();
                camera->device = selectCameraDevice(provider, camera->config.device);
                if (!camera->device)
                {
                    ORIGINATE_ERROR(
                        "Stream %s selected unavailable device %u", stream.config.id.c_str(), camera->config.device);
                }
                camera->sensorMode = selectSensorMode(camera->device, camera->config.mode);
                camera->iSensorMode = Argus::interface_cast<Argus::ISensorMode>(camera->sensorMode);
                if (!camera->iSensorMode)
                {
                    ORIGINATE_ERROR("Stream %s selected unavailable mode %u on d%u", stream.config.id.c_str(),
                                    camera->config.mode, camera->config.device);
                }
                const Argus::Size2D<uint32_t> resolution = camera->iSensorMode->getResolution();
                const uint64_t period = camera->iSensorMode->getFrameDurationRange().min();
                if (period == 0)
                {
                    ORIGINATE_ERROR("Stream %s sensor mode has zero frame duration", stream.config.id.c_str());
                }

                ImageSize sourceSize;
                sourceSize.width = resolution.width();
                sourceSize.height = resolution.height();
                sourceSizes.push_back(sourceSize);
                stream.framePeriodNs = std::max(stream.framePeriodNs, period);

                CaptureEndpointDescriptor endpoint;
                endpoint.streamIndex = streamIndex;
                endpoint.cameraIndex = cameraIndex;
                endpoint.streamType = stream.config.type;
                endpoint.modeIndex = camera->config.mode;
                endpoint.sourceWidth = resolution.width();
                endpoint.sourceHeight = resolution.height();
                endpoint.framePeriodNs = period;
                endpoints.push_back(endpoint);

                const size_t flatIndex = m_cameras.size();
                m_cameraLookup[streamIndex][cameraIndex] = flatIndex;
                stream.cameraIndices.push_back(flatIndex);
                m_cameras.push_back(std::move(camera));
            }

            std::string geometryError;
            if (!resolveOutputGeometry(stream.config.output, sourceSizes, &stream.geometry, &geometryError))
            {
                ORIGINATE_ERROR("Stream %s geometry error: %s", stream.config.id.c_str(), geometryError.c_str());
            }
            stream.nominalFps = roundedFrameRateFromDurationNs(stream.framePeriodNs);
            stream.acquireSamples.resize(stream.cameraIndices.size(), 0);
            stream.lastSelectedFrameNumbers.resize(stream.cameraIndices.size(), 0);
            stream.selectedFrameGapsInterval.resize(stream.cameraIndices.size(), 0);
            stream.hasLastSelectedFrameNumber.resize(stream.cameraIndices.size(), false);
            stream.acquireSumNs.resize(stream.cameraIndices.size(), 0.0);
            stream.acquireMaxNs.resize(stream.cameraIndices.size(), 0);
        }

        CaptureDomainPlan plan;
        std::string planError;
        if (!buildCaptureDomainPlan(endpoints, &plan, &planError))
            ORIGINATE_ERROR("Capture-domain plan failed: %s", planError.c_str());

        std::vector<uint32_t> streamFps;
        for (size_t streamIndex = 0; streamIndex < m_streams.size(); ++streamIndex)
        {
            streamFps.push_back(m_config.streams[streamIndex].output.fps ? m_config.streams[streamIndex].output.fps :
                                                                           m_streams[streamIndex]->nominalFps);
        }
        const size_t priorityStreamIndex = selectPriorityStream(m_config, streamFps);

        m_domains.clear();
        m_domains.reserve(plan.domains.size());
        for (size_t domainIndex = 0; domainIndex < plan.domains.size(); ++domainIndex)
        {
            m_domains.emplace_back(new DomainState());
            DomainState& domain = *m_domains.back();
            domain.index = domainIndex;
            for (size_t member = 0; member < plan.domains[domainIndex].size(); ++member)
            {
                const size_t cameraFlatIndex = plan.domains[domainIndex][member];
                CameraState& camera = *m_cameras[cameraFlatIndex];
                camera.domainIndex = domainIndex;
                domain.cameraIndices.push_back(cameraFlatIndex);
            }

            std::vector<std::vector<size_t>> sessionCameraGroups;
            const bool independentPriorityStereo =
                priorityStreamIndex < m_streams.size() &&
                m_streams[priorityStreamIndex]->config.type == STREAM_TYPE_STEREO && domain.cameraIndices.size() == 2 &&
                m_cameras[domain.cameraIndices[0]]->streamIndex == priorityStreamIndex &&
                m_cameras[domain.cameraIndices[1]]->streamIndex == priorityStreamIndex;
            if (independentPriorityStereo)
            {
                for (size_t member = 0; member < domain.cameraIndices.size(); ++member)
                {
                    sessionCameraGroups.push_back(std::vector<size_t>(1, domain.cameraIndices[member]));
                }
            }
            else
            {
                sessionCameraGroups.push_back(domain.cameraIndices);
            }

            for (size_t sessionIndex = 0; sessionIndex < sessionCameraGroups.size(); ++sessionIndex)
            {
                std::unique_ptr<CaptureSessionState> captureSession(new CaptureSessionState());
                captureSession->cameraIndices = sessionCameraGroups[sessionIndex];
                std::vector<Argus::CameraDevice*> devices;
                for (size_t member = 0; member < captureSession->cameraIndices.size(); ++member)
                {
                    devices.push_back(m_cameras[captureSession->cameraIndices[member]]->device);
                }

                Argus::Status sessionStatus = Argus::STATUS_OK;
                captureSession->session.reset(iProvider->createCaptureSession(devices, &sessionStatus));
                captureSession->iSession = Argus::interface_cast<Argus::ICaptureSession>(captureSession->session);
                captureSession->iEventProvider = Argus::interface_cast<Argus::IEventProvider>(captureSession->session);
                if (!captureSession->iSession || !captureSession->iEventProvider)
                {
                    ORIGINATE_ERROR("Failed to create capture domain %zu session %zu (status %d)", domainIndex,
                                    sessionIndex, sessionStatus);
                }

                std::vector<Argus::EventType> eventTypes;
                if (captureSession->iEventProvider->getAvailableEventTypes(&eventTypes) != Argus::STATUS_OK ||
                    eventTypes.empty())
                {
                    ORIGINATE_ERROR(
                        "Failed to enumerate events for capture domain %zu session %zu", domainIndex, sessionIndex);
                }
                captureSession->eventQueue.reset(captureSession->iEventProvider->createEventQueue(eventTypes));
                captureSession->iEventQueue = Argus::interface_cast<Argus::IEventQueue>(captureSession->eventQueue);
                if (!captureSession->eventQueue || !captureSession->iEventQueue)
                {
                    ORIGINATE_ERROR(
                        "Failed to create event queue for capture domain %zu session %zu", domainIndex, sessionIndex);
                }

                for (size_t member = 0; member < captureSession->cameraIndices.size(); ++member)
                {
                    CameraState& camera = *m_cameras[captureSession->cameraIndices[member]];
                    const StreamState& stream = *m_streams[camera.streamIndex];
                    const bool directHevc = stream.config.output.codec == CODEC_HEVC;
                    Argus::UniqueObj<Argus::OutputStreamSettings> settings(
                        captureSession->iSession->createOutputStreamSettings(directHevc ? Argus::STREAM_TYPE_BUFFER :
                                                                                          Argus::STREAM_TYPE_EGL));
                    Argus::IOutputStreamSettings* iSettings =
                        Argus::interface_cast<Argus::IOutputStreamSettings>(settings);
                    if (!iSettings)
                    {
                        ORIGINATE_ERROR("Failed to create output settings for %s", camera.label.c_str());
                    }
                    iSettings->setCameraDevice(camera.device);
                    const bool fifoMode = stream.config.capture.mode == CAPTURE_MODE_FIFO;
                    if (domain.sessions.empty() && member == 0)
                    {
                        domain.fifoMode = fifoMode;
                        domain.fifoLength = fifoMode ? stream.config.capture.fifoLength : 0;
                    }
                    else if (domain.fifoMode != fifoMode ||
                             (fifoMode && domain.fifoLength != stream.config.capture.fifoLength))
                    {
                        ORIGINATE_ERROR("Capture domain %zu mixes incompatible EGL stream modes", domainIndex);
                    }
                    if (directHevc)
                    {
                        Argus::IBufferOutputStreamSettings* iBufferSettings =
                            Argus::interface_cast<Argus::IBufferOutputStreamSettings>(settings);
                        if (!iBufferSettings ||
                            iBufferSettings->setBufferType(Argus::BUFFER_TYPE_EGL_IMAGE) != Argus::STATUS_OK)
                        {
                            ORIGINATE_ERROR("Failed to configure direct DMABUF capture for %s", camera.label.c_str());
                        }
                        iBufferSettings->setMetadataEnable(true);
                    }
                    else
                    {
                        Argus::IEGLOutputStreamSettings* iEglSettings =
                            Argus::interface_cast<Argus::IEGLOutputStreamSettings>(settings);
                        if (!iEglSettings)
                        {
                            ORIGINATE_ERROR("Failed to create EGL settings for %s", camera.label.c_str());
                        }
                        iEglSettings->setPixelFormat(Argus::PIXEL_FMT_YCbCr_420_888);
                        iEglSettings->setResolution(
                            Argus::Size2D<uint32_t>(stream.geometry.captureWidth, stream.geometry.captureHeight));
                        iEglSettings->setEGLDisplay(display);
                        if (iEglSettings->setMode(fifoMode ? Argus::EGL_STREAM_MODE_FIFO :
                                                             Argus::EGL_STREAM_MODE_MAILBOX) != Argus::STATUS_OK)
                        {
                            ORIGINATE_ERROR(
                                "Failed to set %s mode for %s", fifoMode ? "FIFO" : "mailbox", camera.label.c_str());
                        }
                        if (fifoMode && (domain.fifoLength == 0 ||
                                         iEglSettings->setFifoLength(domain.fifoLength) != Argus::STATUS_OK))
                        {
                            ORIGINATE_ERROR(
                                "Failed to set FIFO length %u for %s", domain.fifoLength, camera.label.c_str());
                        }
                        iEglSettings->setMetadataEnable(true);
                    }
                    camera.outputStream.reset(captureSession->iSession->createOutputStream(settings.get()));
                    if (!camera.outputStream)
                        ORIGINATE_ERROR("Failed to create output stream %s", camera.label.c_str());
                    if (directHevc)
                    {
                        camera.iBufferStream = Argus::interface_cast<Argus::IBufferOutputStream>(camera.outputStream);
                        if (!camera.iBufferStream)
                        {
                            ORIGINATE_ERROR("Failed to create direct buffer stream %s", camera.label.c_str());
                        }
                        Argus::UniqueObj<Argus::BufferSettings> bufferSettings(
                            camera.iBufferStream->createBufferSettings());
                        Argus::IEGLImageBufferSettings* iEglImageSettings =
                            Argus::interface_cast<Argus::IEGLImageBufferSettings>(bufferSettings);
                        if (!bufferSettings || !iEglImageSettings ||
                            iEglImageSettings->setEGLDisplay(display) != Argus::STATUS_OK)
                        {
                            ORIGINATE_ERROR("Failed to create direct buffer settings for %s", camera.label.c_str());
                        }
                        camera.dmaBuffers.reserve(kDirectCaptureBufferCount);
                        camera.argusBuffers.resize(kDirectCaptureBufferCount);
                        const Argus::Size2D<uint32_t> directSize(stream.geometry.width, stream.geometry.height);
                        for (uint32_t bufferIndex = 0; bufferIndex < kDirectCaptureBufferCount; ++bufferIndex)
                        {
                            std::unique_ptr<DmaBuffer> dmaBuffer(
                                DmaBuffer::create(directSize, NVBUF_COLOR_FORMAT_NV12, NVBUF_LAYOUT_BLOCK_LINEAR));
                            if (!dmaBuffer || !dmaBuffer->mapEglImage() ||
                                iEglImageSettings->setEGLImage(dmaBuffer->getEglImage()) != Argus::STATUS_OK)
                            {
                                ORIGINATE_ERROR("Failed to allocate direct capture DMABUF %u for %s", bufferIndex,
                                                camera.label.c_str());
                            }
                            camera.argusBuffers[bufferIndex].reset(
                                camera.iBufferStream->createBuffer(bufferSettings.get()));
                            Argus::IBuffer* iBuffer =
                                Argus::interface_cast<Argus::IBuffer>(camera.argusBuffers[bufferIndex]);
                            if (!iBuffer)
                            {
                                ORIGINATE_ERROR("Failed to create Argus direct buffer %u for %s", bufferIndex,
                                                camera.label.c_str());
                            }
                            iBuffer->setClientData(dmaBuffer.get());
                            if (!dmaBuffer->bindArgusBuffer(camera.iBufferStream, camera.argusBuffers[bufferIndex].get()) ||
                                !dmaBuffer->releaseToArgus())
                            {
                                ORIGINATE_ERROR("Failed to release direct buffer %u for initial %s capture",
                                                bufferIndex, camera.label.c_str());
                            }
                            camera.dmaBuffers.push_back(std::move(dmaBuffer));
                        }
                    }
                    else
                    {
                        camera.consumer.reset(EGLStream::FrameConsumer::create(camera.outputStream.get()));
                        camera.iConsumer = Argus::interface_cast<EGLStream::IFrameConsumer>(camera.consumer);
                        if (!camera.iConsumer)
                            ORIGINATE_ERROR("Failed to create consumer %s", camera.label.c_str());
                    }
                }

                captureSession->request.reset(captureSession->iSession->createRequest(Argus::CAPTURE_INTENT_PREVIEW));
                Argus::IRequest* iRequest = Argus::interface_cast<Argus::IRequest>(captureSession->request);
                if (!iRequest)
                    ORIGINATE_ERROR("Failed to create request for domain %zu session %zu", domainIndex, sessionIndex);
                Argus::IAutoControlSettings* iAutoControl =
                    Argus::interface_cast<Argus::IAutoControlSettings>(iRequest->getAutoControlSettings());
                if (!iAutoControl ||
                    iAutoControl->setAeAntibandingMode(Argus::AE_ANTIBANDING_MODE_OFF) != Argus::STATUS_OK)
                {
                    ORIGINATE_ERROR("Failed to disable AE antibanding for capture domain %zu session %zu", domainIndex,
                                    sessionIndex);
                }
                for (size_t member = 0; member < captureSession->cameraIndices.size(); ++member)
                {
                    CameraState& camera = *m_cameras[captureSession->cameraIndices[member]];
                    const StreamState& stream = *m_streams[camera.streamIndex];
                    if (iRequest->enableOutputStream(camera.outputStream.get()) != Argus::STATUS_OK)
                    {
                        ORIGINATE_ERROR("Failed to enable output %s", camera.label.c_str());
                    }
                    Argus::InterfaceProvider* streamSettings = iRequest->getStreamSettings(camera.outputStream.get());
                    Argus::IStreamSettings* iStreamSettings =
                        Argus::interface_cast<Argus::IStreamSettings>(streamSettings);
                    if (!iStreamSettings)
                    {
                        ORIGINATE_ERROR("Failed to get stream crop settings for %s", camera.label.c_str());
                    }
                    const NormalizedClip& clip = stream.geometry.clips[camera.cameraIndex];
                    if (iStreamSettings->setSourceClipRect(
                            Argus::Rectangle<float>(clip.left, clip.top, clip.right, clip.bottom)) != Argus::STATUS_OK)
                    {
                        ORIGINATE_ERROR("Failed to set source crop for %s", camera.label.c_str());
                    }
                }
                Argus::ISourceSettings* sourceSettings =
                    Argus::interface_cast<Argus::ISourceSettings>(iRequest->getSourceSettings());
                if (!sourceSettings)
                {
                    ORIGINATE_ERROR(
                        "Failed to get source settings for domain %zu session %zu", domainIndex, sessionIndex);
                }
                CameraState& firstCamera = *m_cameras[captureSession->cameraIndices.front()];
                if (sourceSettings->setSensorMode(firstCamera.sensorMode) != Argus::STATUS_OK)
                {
                    std::fprintf(stderr,
                                 "Warning: capture domain %zu session %zu rejected explicit mode; using Argus default\n",
                                 domainIndex, sessionIndex);
                }
                // Pin the frame duration to the sensor mode's fastest rate.
                // Without this, auto-exposure is free to stretch the exposure
                // (and with it the frame period) in dimmer light — the whole
                // rig then follows the trigger down (measured: 20 ms exposure
                // -> 43.7 fps on every camera under evening lab lighting). A
                // fixed-rate robot camera must hold cadence and let AE
                // compensate with gain instead.
                const uint64_t pinnedFrameDurationNs = firstCamera.iSensorMode->getFrameDurationRange().min();
                if (sourceSettings->setFrameDurationRange(
                        Argus::Range<uint64_t>(pinnedFrameDurationNs, pinnedFrameDurationNs)) != Argus::STATUS_OK)
                {
                    std::fprintf(stderr, "Warning: capture domain %zu session %zu rejected the pinned frame duration\n",
                                 domainIndex, sessionIndex);
                }
                domain.sessions.push_back(std::move(captureSession));
            }

            if (verboseLoggingEnabled())
            {
                std::printf("Capture domain %zu:", domainIndex);
                for (size_t member = 0; member < domain.cameraIndices.size(); ++member)
                {
                    const CameraState& camera = *m_cameras[domain.cameraIndices[member]];
                    const StreamState& stream = *m_streams[camera.streamIndex];
                    const bool fifoMode = stream.config.capture.mode == CAPTURE_MODE_FIFO;
                    std::printf("%s %s(d%u/m%u/%s,capture-mode=%s", member == 0 ? "" : ",", camera.label.c_str(),
                                camera.config.device, camera.config.mode,
                                stream.config.output.codec == CODEC_HEVC ? "direct-buffer" : "egl-stream",
                                fifoMode ? "fifo" : "mailbox");
                    if (fifoMode && stream.config.output.codec != CODEC_HEVC)
                        std::printf(",egl-fifo-length=%u", stream.config.capture.fifoLength);
                    if (independentPriorityStereo)
                        std::printf("/independent-session");
                    std::printf(")");
                }
                std::printf("\n");
            }
        }
        return true;
    }

    bool startPriorityDomain(size_t priorityDomainIndex)
    {
        if (m_started || priorityDomainIndex >= m_domains.size())
            return false;
        if (m_domains[priorityDomainIndex]->fifoMode)
        {
            std::fprintf(
                stderr,
                "Priority capture domain %zu is configured for FIFO consumption; priority capture requires mailbox policy for both direct-buffer and EGL streams\n",
                priorityDomainIndex);
            return false;
        }
        m_error = false;
        for (size_t i = 0; i < m_domains.size(); ++i)
        {
            DomainState& domain = *m_domains[i];
            std::lock_guard<std::mutex> lock(domain.mutex);
            domain.stop = false;
            domain.error = false;
            domain.hasPending = false;
            domain.pending.clear();
            domain.candidates.clear();
            for (size_t eye = 0; eye < domain.headFrames.size(); ++eye)
                domain.headFrames[eye].reset();
            domain.continuousFifo = i != priorityDomainIndex && domain.fifoMode;
            domain.demandDriven = i != priorityDomainIndex && !domain.continuousFifo;
            domain.captureRequested = false;
            domain.selectionTimestampUnavailableWarned = false;
            for (size_t sessionIndex = 0; sessionIndex < domain.sessions.size(); ++sessionIndex)
            {
                domain.sessions[sessionIndex]->captureCompleteCredits = 0;
                domain.sessions[sessionIndex]->repeating = false;
            }
            domain.priorityStereo = false;
            domain.priorityStereoStreamIndex = std::numeric_limits<size_t>::max();
            domain.stereoLeftFlatIndex = std::numeric_limits<size_t>::max();
            domain.stereoRightFlatIndex = std::numeric_limits<size_t>::max();
            if (i == priorityDomainIndex && domain.cameraIndices.size() == 2)
            {
                const size_t firstFlatIndex = domain.cameraIndices[0];
                const size_t secondFlatIndex = domain.cameraIndices[1];
                const CameraState& first = *m_cameras[firstFlatIndex];
                const CameraState& second = *m_cameras[secondFlatIndex];
                if (first.streamIndex == second.streamIndex &&
                    m_streams[first.streamIndex]->config.type == STREAM_TYPE_STEREO &&
                    first.cameraIndex != second.cameraIndex && first.cameraIndex < 2 && second.cameraIndex < 2)
                {
                    domain.priorityStereo = true;
                    domain.priorityStereoStreamIndex = first.streamIndex;
                    domain.stereoLeftFlatIndex = first.cameraIndex == 0 ? firstFlatIndex : secondFlatIndex;
                    domain.stereoRightFlatIndex = first.cameraIndex == 1 ? firstFlatIndex : secondFlatIndex;
                }
            }
            domain.stoppingRepeat = false;
            domain.candidateOverflowDiscardsInterval = 0;
            domain.candidateOverflowDiscardsTotal = 0;
            domain.candidateSelectionDiscardsInterval = 0;
            domain.candidateSelectionDiscardsTotal = 0;
            domain.candidateSelectionsInterval = 0;
            domain.candidateSelectionsTotal = 0;
            domain.candidateDeltaSumNsInterval = 0;
            domain.candidateDeltaMaxNsInterval = 0;
            domain.candidatePastSelectionsInterval = 0;
            domain.candidateFutureSelectionsInterval = 0;
            domain.candidateDepthPeakInterval = 0;
            domain.fifoEventBacklog = 0;
            domain.fifoEventBacklogPeakInterval = 0;
            domain.stereoPairsInterval = 0;
            domain.stereoPairsTotal = 0;
            domain.stereoRepairPairsInterval = 0;
            domain.stereoRepairPairsTotal = 0;
            domain.stereoLeftDropsInterval = 0;
            domain.stereoLeftDropsTotal = 0;
            domain.stereoRightDropsInterval = 0;
            domain.stereoRightDropsTotal = 0;
            domain.stereoReacquiresInterval = 0;
            domain.stereoReacquiresTotal = 0;
            domain.stereoReacquireWaitSumNsInterval = 0;
            domain.stereoReacquireWaitMaxNsInterval = 0;
            domain.stereoSelectedAgeSumNsInterval = 0;
            domain.stereoSelectedAgeMaxNsInterval = 0;
            domain.stereoInitialDeltaSumNsInterval = 0;
            domain.stereoInitialDeltaMaxNsInterval = 0;
            domain.stereoFinalDeltaSumNsInterval = 0;
            domain.stereoFinalDeltaMaxNsInterval = 0;
            domain.headSlotOverwritesInterval = 0;
            domain.headSlotOverwritesTotal = 0;
        }
        if (!startDomain(priorityDomainIndex))
        {
            stop();
            return false;
        }
        m_started = true;
        if (verboseLoggingEnabled())
        {
            std::printf(
                "argus_sender[capture]: priority domain %zu started; secondary domains are held until priority warm-up completes\n",
                priorityDomainIndex);
        }
        return true;
    }

    bool startSecondaryDomains(size_t priorityDomainIndex)
    {
        if (!m_started || priorityDomainIndex >= m_domains.size())
            return false;
        for (size_t i = 0; i < m_domains.size(); ++i)
        {
            if (i == priorityDomainIndex)
                continue;
            if (!startDomain(i))
            {
                stop();
                return false;
            }
        }
        if (verboseLoggingEnabled() && hasContinuousFifoDomains())
        {
            std::printf(
                "argus_sender[capture]: secondary domains started; FIFO domains consume continuously with a %zu-frame candidate window\n",
                kFifoCandidateWindowLength);
        }
        else if (verboseLoggingEnabled())
        {
            std::printf("argus_sender[capture]: secondary domains started in demand-driven capture mode\n");
        }
        return true;
    }

    bool discardDomainWarmup(size_t domainIndex, uint32_t frameCount)
    {
        if (domainIndex >= m_domains.size())
            return false;
        for (uint32_t frame = 0; frame < frameCount; ++frame)
        {
            if (!requestDomainCapture(domainIndex))
                return false;
            std::vector<std::unique_ptr<PendingFrame>> discarded;
            if (!takeDomain(domainIndex, &discarded))
                return false;
        }
        if (frameCount > 0 && verboseLoggingEnabled())
        {
            std::printf("argus_sender[capture-domain:%zu]: warm-up complete, discarded %u frame set(s)\n", domainIndex,
                        frameCount);
        }
        return true;
    }

    bool discardWarmup(uint32_t frameCount)
    {
        if (frameCount == 0)
            return true;
        std::vector<std::future<bool>> warmups;
        for (size_t domainIndex = 0; domainIndex < m_domains.size(); ++domainIndex)
        {
            warmups.push_back(std::async(std::launch::async, [this, domainIndex, frameCount]()
                                         { return discardDomainWarmup(domainIndex, frameCount); }));
        }
        bool ok = true;
        for (size_t i = 0; i < warmups.size(); ++i)
            ok = warmups[i].get() && ok;
        return ok;
    }

    bool waitForDomainSynchronization(size_t domainIndex,
                                      uint32_t maxFrameSets,
                                      uint32_t requiredConsecutive,
                                      uint64_t toleranceNs)
    {
        if (domainIndex >= m_domains.size() || maxFrameSets == 0 || requiredConsecutive == 0)
        {
            return false;
        }
        DomainState& domain = *m_domains[domainIndex];
        if (domain.cameraIndices.size() < 2)
            return true;

        uint32_t consecutive = 0;
        uint64_t lastSpanNs = 0;
        for (uint32_t attempt = 1; attempt <= maxFrameSets; ++attempt)
        {
            if (!requestDomainCapture(domainIndex))
                return false;
            std::vector<std::unique_ptr<PendingFrame>> pending;
            if (!takeDomain(domainIndex, &pending))
                return false;
            uint64_t earliest = std::numeric_limits<uint64_t>::max();
            uint64_t latest = 0;
            for (size_t i = 0; i < pending.size(); ++i)
            {
                if (!pending[i]->timing.hasSensorSofTimestampTsc || pending[i]->timing.sensorSofTimestampTscNs == 0)
                {
                    std::fprintf(stderr,
                                 "Priority capture domain %zu has no SOF TSC timestamp; refusing unlocked startup\n",
                                 domainIndex);
                    return false;
                }
                earliest = std::min(earliest, pending[i]->timing.sensorSofTimestampTscNs);
                latest = std::max(latest, pending[i]->timing.sensorSofTimestampTscNs);
            }
            lastSpanNs = latest - earliest;
            if (lastSpanNs <= toleranceNs)
                ++consecutive;
            else
                consecutive = 0;
            if (consecutive >= requiredConsecutive)
            {
                std::printf(
                    "argus_sender[capture-domain:%zu]: priority synchronization locked after %u frame set(s), tsc_span_us=%.3f\n",
                    domainIndex, attempt, static_cast<double>(lastSpanNs) / 1000.0);
                return true;
            }
        }
        std::fprintf(
            stderr,
            "Priority capture domain %zu did not reach %u consecutive frame sets within %.3f us after %u attempts; refusing unlocked startup with tsc_span_us=%.3f\n",
            domainIndex, requiredConsecutive, static_cast<double>(toleranceNs) / 1000.0, maxFrameSets,
            static_cast<double>(lastSpanNs) / 1000.0);
        return false;
    }

    bool requestDomainCapture(size_t domainIndex)
    {
        if (domainIndex >= m_domains.size())
            return false;
        DomainState& domain = *m_domains[domainIndex];
        {
            std::lock_guard<std::mutex> lock(domain.mutex);
            if (domain.error || domain.stop)
                return false;
            if (domain.continuousFifo)
                return true;
            if (!domain.demandDriven)
                return true;
            domain.captureRequested = true;
        }
        domain.slotAvailable.notify_one();
        return true;
    }

    void prepareCaptured(std::vector<CapturedFrameSet>* captured) const
    {
        if (!captured)
            return;
        captured->clear();
        captured->resize(m_streams.size());
        for (size_t streamIndex = 0; streamIndex < m_streams.size(); ++streamIndex)
        {
            (*captured)[streamIndex].frames.resize(m_streams[streamIndex]->cameraIndices.size());
            (*captured)[streamIndex].dmaBuffers.resize(m_streams[streamIndex]->cameraIndices.size());
            (*captured)[streamIndex].timings.resize(m_streams[streamIndex]->cameraIndices.size());
        }
    }

    bool acquireDomain(size_t domainIndex, std::vector<CapturedFrameSet>* captured)
    {
        if (!captured || captured->size() != m_streams.size())
            return false;
        std::vector<std::unique_ptr<PendingFrame>> pending;
        if (!takeDomain(domainIndex, &pending))
            return false;
        return movePendingToCaptured(&pending, captured);
    }

    bool acquireDomainClosestTo(size_t domainIndex, uint64_t referenceSofTscNs, std::vector<CapturedFrameSet>* captured)
    {
        if (!captured || captured->size() != m_streams.size() || domainIndex >= m_domains.size())
        {
            return false;
        }
        DomainState& domain = *m_domains[domainIndex];
        if (!domain.continuousFifo)
            return acquireDomain(domainIndex, captured);

        const CameraState& firstCamera = *m_cameras[domain.cameraIndices.front()];
        const uint64_t framePeriodNs = m_streams[firstCamera.streamIndex]->framePeriodNs;
        const uint64_t selectionWaitNs = framePeriodNs > 0 ? framePeriodNs * 2ULL : kCaptureWaitTimeoutNs;
        const std::chrono::steady_clock::time_point deadline =
            std::chrono::steady_clock::now() + std::chrono::nanoseconds(selectionWaitNs);

        std::vector<std::unique_ptr<PendingFrame>> selected;
        int64_t selectedDeltaNs = 0;
        size_t selectionDiscards = 0;
        {
            std::unique_lock<std::mutex> lock(domain.mutex);
            while (!candidateReadyLocked(domain, referenceSofTscNs, framePeriodNs) && !domain.error && !domain.stop &&
                   !stopRequested())
            {
                if (domain.framesAvailable.wait_until(lock, deadline) == std::cv_status::timeout)
                {
                    break;
                }
            }
            if (domain.candidates.empty() || domain.error || domain.stop || stopRequested())
            {
                return false;
            }

            std::vector<uint64_t> candidateTimestamps;
            candidateTimestamps.reserve(domain.candidates.size());
            for (size_t i = 0; i < domain.candidates.size(); ++i)
            {
                candidateTimestamps.push_back(candidateSofTscNs(domain.candidates[i]));
            }
            size_t selectedIndex = domain.candidates.size() - 1;
            if (!chooseClosestTscCandidate(candidateTimestamps, referenceSofTscNs, &selectedIndex, &selectedDeltaNs))
            {
                selectedDeltaNs = 0;
                if (!domain.selectionTimestampUnavailableWarned)
                {
                    std::fprintf(
                        stderr,
                        "Warning: FIFO candidate selection for domain %zu lacks valid SOF TSC; selecting newest candidate\n",
                        domain.index);
                    domain.selectionTimestampUnavailableWarned = true;
                }
            }

            selectionDiscards = selectedIndex;
            while (selectedIndex > 0)
            {
                domain.candidates.pop_front();
                --selectedIndex;
            }
            selected.swap(domain.candidates.front());
            domain.candidates.pop_front();
        }

        if (selectionDiscards > 0)
        {
            domain.candidateSelectionDiscardsInterval.fetch_add(selectionDiscards);
            domain.candidateSelectionDiscardsTotal.fetch_add(selectionDiscards);
        }
        domain.candidateSelectionsInterval.fetch_add(1);
        domain.candidateSelectionsTotal.fetch_add(1);
        const uint64_t absoluteDelta = absoluteDeltaNs(selectedDeltaNs);
        domain.candidateDeltaSumNsInterval.fetch_add(absoluteDelta);
        updateAtomicMaximum(&domain.candidateDeltaMaxNsInterval, absoluteDelta);
        if (selectedDeltaNs < 0)
            domain.candidatePastSelectionsInterval.fetch_add(1);
        else if (selectedDeltaNs > 0)
            domain.candidateFutureSelectionsInterval.fetch_add(1);

        return movePendingToCaptured(&selected, captured);
    }

    bool reacquireDomain(size_t domainIndex, std::vector<CapturedFrameSet>* captured)
    {
        if (domainIndex >= m_domains.size() || !captured)
            return false;
        const DomainState& domain = *m_domains[domainIndex];
        for (size_t i = 0; i < domain.cameraIndices.size(); ++i)
        {
            const CameraState& camera = *m_cameras[domain.cameraIndices[i]];
            (*captured)[camera.streamIndex].frames[camera.cameraIndex].reset();
            (*captured)[camera.streamIndex].dmaBuffers[camera.cameraIndex].reset();
            (*captured)[camera.streamIndex].timings[camera.cameraIndex] = FrameTiming();
        }
        if (!requestDomainCapture(domainIndex))
            return false;
        return acquireDomain(domainIndex, captured);
    }

    void recordCaptured(size_t streamIndex, const CapturedFrameSet& frames)
    {
        if (streamIndex >= m_streams.size() || frames.frames.size() != m_streams[streamIndex]->cameraIndices.size() ||
            frames.dmaBuffers.size() != frames.frames.size())
        {
            return;
        }
        for (size_t i = 0; i < frames.frames.size(); ++i)
        {
            // The HEVC processor can take ownership of a direct DMABUF
            // before this periodic stats hook. Successful acquisition leaves
            // a steady-clock marker even when SOF TSC is unavailable.
            if (!frames.frames[i].get() && !frames.dmaBuffers[i].get() &&
                (i >= frames.timings.size() || frames.timings[i].acquiredSteadyNs == 0))
                return;
        }
        recordStats(m_streams[streamIndex].get(), frames);
    }

    bool stop()
    {
        bool ok = true;
        std::vector<std::vector<bool>> waitForIdle(m_domains.size());
        for (size_t i = 0; i < m_domains.size(); ++i)
        {
            if (!m_domains[i])
                continue;
            DomainState& domain = *m_domains[i];
            waitForIdle[i].assign(domain.sessions.size(), false);
            bool repeating = false;
            for (size_t sessionIndex = 0; sessionIndex < domain.sessions.size(); ++sessionIndex)
            {
                repeating = repeating || domain.sessions[sessionIndex]->repeating;
            }
            {
                std::lock_guard<std::mutex> lock(domain.mutex);
                if (domain.continuousFifo && repeating)
                {
                    domain.stoppingRepeat = true;
                    domain.candidates.clear();
                }
                else
                {
                    domain.stop = true;
                }
                domain.captureRequested = false;
            }
            domain.slotAvailable.notify_all();
            domain.framesAvailable.notify_all();
        }
        for (size_t i = 0; i < m_domains.size(); ++i)
        {
            if (!m_domains[i])
                continue;
            DomainState& domain = *m_domains[i];
            for (size_t sessionIndex = 0; sessionIndex < domain.sessions.size(); ++sessionIndex)
            {
                CaptureSessionState& session = *domain.sessions[sessionIndex];
                if (session.iSession && session.repeating)
                {
                    waitForIdle[i][sessionIndex] = true;
                    session.iSession->stopRepeat();
                    if (session.iSession->isRepeating())
                    {
                        std::fprintf(stderr, "Capture domain %zu session %zu remained in repeat mode after stopRepeat\n",
                                     i, sessionIndex);
                        ok = false;
                    }
                }
                session.repeating = false;
            }
        }
        // Keep continuous FIFO consumers alive until all repeated requests
        // retire. During this phase they drain and immediately release frames
        // instead of placing them back into the candidate window.
        for (size_t i = 0; i < m_domains.size(); ++i)
        {
            if (!m_domains[i])
                continue;
            DomainState& domain = *m_domains[i];
            if (!domain.continuousFifo)
                continue;
            for (size_t sessionIndex = 0; sessionIndex < domain.sessions.size(); ++sessionIndex)
            {
                CaptureSessionState& session = *domain.sessions[sessionIndex];
                if (session.iSession && waitForIdle[i][sessionIndex])
                {
                    const Argus::Status status = session.iSession->waitForIdle(kIdleWaitTimeoutNs);
                    if (status != Argus::STATUS_OK)
                    {
                        std::fprintf(
                            stderr,
                            "Capture domain %zu session %zu failed to become idle while draining FIFO (status %d)\n", i,
                            sessionIndex, status);
                        ok = false;
                    }
                }
            }
        }
        for (size_t i = 0; i < m_domains.size(); ++i)
        {
            if (!m_domains[i])
                continue;
            DomainState& domain = *m_domains[i];
            {
                std::lock_guard<std::mutex> lock(domain.mutex);
                domain.stop = true;
                domain.pending.clear();
                domain.hasPending = false;
                domain.candidates.clear();
                for (size_t eye = 0; eye < domain.headFrames.size(); ++eye)
                    domain.headFrames[eye].reset();
            }
            domain.slotAvailable.notify_all();
            domain.framesAvailable.notify_all();
        }
        for (size_t i = 0; i < m_domains.size(); ++i)
        {
            if (!m_domains[i])
                continue;
            DomainState& domain = *m_domains[i];
            if (domain.worker.joinable())
                domain.worker.join();
            if (domain.eventWorker.joinable())
                domain.eventWorker.join();
            for (size_t worker = 0; worker < domain.cameraWorkers.size(); ++worker)
            {
                if (domain.cameraWorkers[worker].joinable())
                    domain.cameraWorkers[worker].join();
            }
        }
        for (size_t i = 0; i < m_domains.size(); ++i)
        {
            if (!m_domains[i])
                continue;
            DomainState& domain = *m_domains[i];
            {
                std::lock_guard<std::mutex> lock(domain.mutex);
                domain.pending.clear();
                domain.hasPending = false;
                domain.candidates.clear();
                for (size_t eye = 0; eye < domain.headFrames.size(); ++eye)
                    domain.headFrames[eye].reset();
            }
            if (domain.continuousFifo)
                continue;
            for (size_t sessionIndex = 0; sessionIndex < domain.sessions.size(); ++sessionIndex)
            {
                CaptureSessionState& session = *domain.sessions[sessionIndex];
                if (session.iSession && waitForIdle[i][sessionIndex])
                {
                    const Argus::Status status = session.iSession->waitForIdle(kIdleWaitTimeoutNs);
                    if (status != Argus::STATUS_OK)
                    {
                        std::fprintf(stderr, "Capture domain %zu session %zu failed to become idle (status %d)\n", i,
                                     sessionIndex, status);
                        ok = false;
                    }
                }
            }
        }
        m_started = false;
        if (!ok)
            m_error = true;
        return ok;
    }

    bool hasError() const
    {
        if (m_error.load())
            return true;
        for (size_t i = 0; i < m_domains.size(); ++i)
        {
            std::lock_guard<std::mutex> lock(m_domains[i]->mutex);
            if (m_domains[i]->error)
                return true;
        }
        return false;
    }

    bool domainUsesContinuousFifo(size_t domainIndex) const
    {
        if (domainIndex >= m_domains.size())
            return false;
        std::lock_guard<std::mutex> lock(m_domains[domainIndex]->mutex);
        return m_domains[domainIndex]->continuousFifo;
    }

    bool hasContinuousFifoDomains() const
    {
        for (size_t i = 0; i < m_domains.size(); ++i)
        {
            std::lock_guard<std::mutex> lock(m_domains[i]->mutex);
            if (m_domains[i]->continuousFifo)
                return true;
        }
        return false;
    }

    void logFifoCandidateStats()
    {
        for (size_t domainIndex = 0; domainIndex < m_domains.size(); ++domainIndex)
        {
            DomainState& domain = *m_domains[domainIndex];
            size_t candidateDepth = 0;
            bool continuousFifo = false;
            {
                std::lock_guard<std::mutex> lock(domain.mutex);
                continuousFifo = domain.continuousFifo;
                candidateDepth = domain.candidates.size();
            }
            if (domain.priorityStereo && domain.priorityStereoStreamIndex < m_streams.size())
            {
                const uint64_t pairs = domain.stereoPairsInterval.exchange(0);
                const uint64_t pairsTotal = domain.stereoPairsTotal.load();
                const uint64_t repairs = domain.stereoRepairPairsInterval.exchange(0);
                const uint64_t repairsTotal = domain.stereoRepairPairsTotal.load();
                const uint64_t leftDrops = domain.stereoLeftDropsInterval.exchange(0);
                const uint64_t leftDropsTotal = domain.stereoLeftDropsTotal.load();
                const uint64_t rightDrops = domain.stereoRightDropsInterval.exchange(0);
                const uint64_t rightDropsTotal = domain.stereoRightDropsTotal.load();
                const uint64_t reacquires = domain.stereoReacquiresInterval.exchange(0);
                const uint64_t reacquiresTotal = domain.stereoReacquiresTotal.load();
                const uint64_t waitSumNs = domain.stereoReacquireWaitSumNsInterval.exchange(0);
                const uint64_t waitMaxNs = domain.stereoReacquireWaitMaxNsInterval.exchange(0);
                const uint64_t selectedAgeSumNs = domain.stereoSelectedAgeSumNsInterval.exchange(0);
                const uint64_t selectedAgeMaxNs = domain.stereoSelectedAgeMaxNsInterval.exchange(0);
                const uint64_t initialDeltaSumNs = domain.stereoInitialDeltaSumNsInterval.exchange(0);
                const uint64_t initialDeltaMaxNs = domain.stereoInitialDeltaMaxNsInterval.exchange(0);
                const uint64_t finalDeltaSumNs = domain.stereoFinalDeltaSumNsInterval.exchange(0);
                const uint64_t finalDeltaMaxNs = domain.stereoFinalDeltaMaxNsInterval.exchange(0);
                const uint64_t slotOverwrites = domain.headSlotOverwritesInterval.exchange(0);
                const uint64_t slotOverwritesTotal = domain.headSlotOverwritesTotal.load();
                if (!verboseLoggingEnabled())
                {
                    std::printf(
                        "argus_sender[head-handoff:%s]: slot_overwrites(interval/total)=%llu/%llu, alignment_drops(left/right,interval)=%llu/%llu\n",
                        m_streams[domain.priorityStereoStreamIndex]->config.id.c_str(),
                        static_cast<unsigned long long>(slotOverwrites),
                        static_cast<unsigned long long>(slotOverwritesTotal),
                        static_cast<unsigned long long>(leftDrops), static_cast<unsigned long long>(rightDrops));
                }
                if (pairs > 0)
                {
                    if (verboseLoggingEnabled())
                        std::printf(
                            "argus_sender[head-align:%s]: pairs(interval/total)=%llu/%llu, repaired_pairs(interval/total)=%llu/%llu, intentional_drops(left/right,interval)=%llu/%llu, intentional_drops(left/right,total)=%llu/%llu, alignment_reacquires(interval/total)=%llu/%llu, slot_overwrites(interval/total)=%llu/%llu, reacquire_wait_us(avg/max)=%.3f/%.3f, selected_age_us(avg/max)=%.3f/%.3f, abs_tsc_delta_us(initial_avg/max)=%.3f/%.3f, abs_tsc_delta_us(final_avg/max)=%.3f/%.3f\n",
                            m_streams[domain.priorityStereoStreamIndex]->config.id.c_str(),
                            static_cast<unsigned long long>(pairs), static_cast<unsigned long long>(pairsTotal),
                            static_cast<unsigned long long>(repairs), static_cast<unsigned long long>(repairsTotal),
                            static_cast<unsigned long long>(leftDrops), static_cast<unsigned long long>(rightDrops),
                            static_cast<unsigned long long>(leftDropsTotal),
                            static_cast<unsigned long long>(rightDropsTotal),
                            static_cast<unsigned long long>(reacquires), static_cast<unsigned long long>(reacquiresTotal),
                            static_cast<unsigned long long>(slotOverwrites),
                            static_cast<unsigned long long>(slotOverwritesTotal),
                            reacquires > 0 ? static_cast<double>(waitSumNs) / static_cast<double>(reacquires) / 1000.0 :
                                             0.0,
                            static_cast<double>(waitMaxNs) / 1000.0,
                            static_cast<double>(selectedAgeSumNs) / static_cast<double>(pairs) / 1000.0,
                            static_cast<double>(selectedAgeMaxNs) / 1000.0,
                            static_cast<double>(initialDeltaSumNs) / static_cast<double>(pairs) / 1000.0,
                            static_cast<double>(initialDeltaMaxNs) / 1000.0,
                            static_cast<double>(finalDeltaSumNs) / static_cast<double>(pairs) / 1000.0,
                            static_cast<double>(finalDeltaMaxNs) / 1000.0);
                }
            }
            if (!verboseLoggingEnabled())
                continue;
            if (!continuousFifo || domain.cameraIndices.empty())
                continue;

            const CameraState& camera = *m_cameras[domain.cameraIndices.front()];
            const uint64_t overflowInterval = domain.candidateOverflowDiscardsInterval.exchange(0);
            const uint64_t selectionInterval = domain.candidateSelectionDiscardsInterval.exchange(0);
            const uint64_t overflowTotal = domain.candidateOverflowDiscardsTotal.load();
            const uint64_t selectionTotal = domain.candidateSelectionDiscardsTotal.load();
            const uint64_t selections = domain.candidateSelectionsInterval.exchange(0);
            const uint64_t selectionsTotal = domain.candidateSelectionsTotal.load();
            const uint64_t deltaSumNs = domain.candidateDeltaSumNsInterval.exchange(0);
            const uint64_t deltaMaxNs = domain.candidateDeltaMaxNsInterval.exchange(0);
            const uint64_t pastSelections = domain.candidatePastSelectionsInterval.exchange(0);
            const uint64_t futureSelections = domain.candidateFutureSelectionsInterval.exchange(0);
            const uint64_t depthPeak = domain.candidateDepthPeakInterval.exchange(candidateDepth);
            const uint64_t eventBacklogPeak =
                domain.fifoEventBacklogPeakInterval.exchange(domain.fifoEventBacklog.load());
            const double deltaAverageUs =
                selections > 0 ? static_cast<double>(deltaSumNs) / static_cast<double>(selections) / 1000.0 : 0.0;

            std::printf(
                "argus_sender[fifo:%s]: fifo_length=%u, candidate_depth(current/peak/capacity)=%zu/%llu/%zu, fifo_event_backlog(current/peak)=%llu/%llu, intentional_discards(interval/total)=%llu/%llu (overflow=%llu/%llu, selection=%llu/%llu), selected(interval/total)=%llu/%llu, selected_abs_tsc_delta_us(avg/max)=%.3f/%.3f, selected_side(past/future/exact)=%llu/%llu/%llu\n",
                camera.label.c_str(), domain.fifoLength, candidateDepth, static_cast<unsigned long long>(depthPeak),
                kFifoCandidateWindowLength, static_cast<unsigned long long>(domain.fifoEventBacklog.load()),
                static_cast<unsigned long long>(eventBacklogPeak),
                static_cast<unsigned long long>(overflowInterval + selectionInterval),
                static_cast<unsigned long long>(overflowTotal + selectionTotal),
                static_cast<unsigned long long>(overflowInterval), static_cast<unsigned long long>(overflowTotal),
                static_cast<unsigned long long>(selectionInterval), static_cast<unsigned long long>(selectionTotal),
                static_cast<unsigned long long>(selections), static_cast<unsigned long long>(selectionsTotal),
                deltaAverageUs, static_cast<double>(deltaMaxNs) / 1000.0,
                static_cast<unsigned long long>(pastSelections), static_cast<unsigned long long>(futureSelections),
                static_cast<unsigned long long>(selections - pastSelections - futureSelections));
        }
    }

    size_t streamCount() const
    {
        return m_streams.size();
    }
    size_t domainCount() const
    {
        return m_domains.size();
    }

    size_t streamDomain(size_t streamIndex) const
    {
        if (streamIndex >= m_streams.size() || m_streams[streamIndex]->cameraIndices.empty())
        {
            return std::numeric_limits<size_t>::max();
        }
        return m_cameras[m_streams[streamIndex]->cameraIndices.front()]->domainIndex;
    }

    size_t domainFor(size_t streamIndex, size_t cameraIndex) const
    {
        if (streamIndex >= m_cameraLookup.size() || cameraIndex >= m_cameraLookup[streamIndex].size())
        {
            return std::numeric_limits<size_t>::max();
        }
        return m_cameras[m_cameraLookup[streamIndex][cameraIndex]]->domainIndex;
    }

    const ResolvedOutputGeometry& geometry(size_t streamIndex) const
    {
        return m_streams[streamIndex]->geometry;
    }

    uint32_t nominalFps(size_t streamIndex) const
    {
        return m_streams[streamIndex]->nominalFps;
    }

    uint64_t framePeriodNs(size_t streamIndex) const
    {
        return m_streams[streamIndex]->framePeriodNs;
    }

    void releaseResources()
    {
        cleanup();
    }

private:
    bool startDomain(size_t domainIndex)
    {
        DomainState& domain = *m_domains[domainIndex];
        if (domain.priorityStereo && verboseLoggingEnabled())
        {
            std::printf(
                "argus_sender[head-align:%s]: independent left/right Argus sessions and continuous acquisition; acquired surfaces are retained without copying\n",
                m_streams[domain.priorityStereoStreamIndex]->config.id.c_str());
        }
        try
        {
            domain.worker = std::thread(&Impl::domainThreadMain, this, &domain);
            if (domain.priorityStereo)
            {
                domain.eventWorker = std::thread(&Impl::priorityEventThreadMain, this, &domain);
                domain.cameraWorkers[0] =
                    std::thread(&Impl::priorityCameraThreadMain, this, &domain, domain.stereoLeftFlatIndex, 0);
                domain.cameraWorkers[1] =
                    std::thread(&Impl::priorityCameraThreadMain, this, &domain, domain.stereoRightFlatIndex, 1);
            }
        }
        catch (...)
        {
            return failDomain(&domain, "failed to create capture worker");
        }
        for (size_t sessionIndex = 0; sessionIndex < domain.sessions.size(); ++sessionIndex)
        {
            CaptureSessionState& session = *domain.sessions[sessionIndex];
            if (session.iSession->repeat(session.request.get()) != Argus::STATUS_OK)
            {
                return failDomain(&domain, "failed to start repeating capture");
            }
            session.repeating = true;
        }
        return true;
    }

    bool stopping(DomainState* domain) const
    {
        std::lock_guard<std::mutex> lock(domain->mutex);
        return domain->stop || domain->error || stopRequested();
    }

    bool failDomain(DomainState* domain, const char* message, Argus::Status status = Argus::STATUS_OK)
    {
        if (status == Argus::STATUS_OK)
        {
            std::fprintf(stderr, "capture domain %zu: %s\n", domain->index, message);
        }
        else
        {
            std::fprintf(stderr, "capture domain %zu: %s (status %d)\n", domain->index, message, status);
        }
        {
            std::lock_guard<std::mutex> lock(domain->mutex);
            domain->error = true;
        }
        m_error = true;
        domain->framesAvailable.notify_all();
        domain->slotAvailable.notify_all();
        return false;
    }

    bool waitForSessionCaptureComplete(DomainState* domain,
                                       CaptureSessionState* session,
                                       const std::atomic<bool>* expectedStop)
    {
        if (session->captureCompleteCredits > 0)
        {
            --session->captureCompleteCredits;
            return true;
        }
        while (!stopping(domain))
        {
            const Argus::Status status =
                session->iEventProvider->waitForEvents(session->eventQueue.get(), kCaptureWaitTimeoutNs);
            if (status == Argus::STATUS_TIMEOUT)
                continue;
            if (status != Argus::STATUS_OK)
            {
                if (expectedStop && expectedStop->load())
                    return false;
                return failDomain(domain, "failed waiting for capture event", status);
            }
            uint64_t completeCount = 0;
            const Argus::Event* event = NULL;
            while ((event = session->iEventQueue->getNextEvent()) != NULL)
            {
                const Argus::IEvent* iEvent = Argus::interface_cast<const Argus::IEvent>(event);
                if (!iEvent)
                    continue;
                const Argus::EventType eventType = iEvent->getEventType();
                if (eventType == Argus::EVENT_TYPE_CAPTURE_COMPLETE)
                {
                    ++completeCount;
                    continue;
                }
                const Argus::IEventError* iError = Argus::interface_cast<const Argus::IEventError>(event);
                if (!iError)
                    continue;
                if (expectedStop && expectedStop->load())
                    return false;
                const char* eventName = eventType.getName();
                const bool correctable = eventName && std::strstr(eventName, "CORR") && !std::strstr(eventName, "UNCORR");
                if (correctable)
                {
                    std::fprintf(
                        stderr,
                        "Warning: capture domain %zu received correctable Argus event %s (status %d); continuing\n",
                        domain->index, eventName, iError->getStatus());
                    continue;
                }
                return failDomain(domain, "received Argus capture error", iError->getStatus());
            }
            if (completeCount > 0)
            {
                session->captureCompleteCredits += domain->continuousFifo ? completeCount : 1ULL;
                --session->captureCompleteCredits;
                return true;
            }
        }
        return false;
    }

    bool waitForCaptureComplete(DomainState* domain, const std::atomic<bool>* expectedStop = NULL)
    {
        for (size_t sessionIndex = 0; sessionIndex < domain->sessions.size(); ++sessionIndex)
        {
            if (!waitForSessionCaptureComplete(domain, domain->sessions[sessionIndex].get(), expectedStop))
            {
                return false;
            }
        }
        uint64_t backlog = 0;
        for (size_t sessionIndex = 0; sessionIndex < domain->sessions.size(); ++sessionIndex)
        {
            backlog += domain->sessions[sessionIndex]->captureCompleteCredits;
        }
        updateAtomicMaximum(&domain->fifoEventBacklogPeakInterval, backlog);
        domain->fifoEventBacklog = backlog;
        return true;
    }

    void priorityEventThreadMain(DomainState* domain)
    {
        // Error events still need a consumer, but capture-complete events must
        // not gate direct-buffer acquisition.
        while (!stopping(domain) && waitForCaptureComplete(domain))
        {
        }
    }

    std::unique_ptr<PendingFrame> acquireCameraFrame(DomainState* domain,
                                                     size_t flatIndex,
                                                     const std::atomic<bool>* expectedStop,
                                                     bool continuousAcquire = false)
    {
        CameraState& camera = *m_cameras[flatIndex];
        Argus::Status status = Argus::STATUS_OK;
        std::unique_ptr<PendingFrame> sample(new PendingFrame(flatIndex));
        const uint64_t droughtStartNs = steadyNowNs();
        uint64_t acquiredTscNs = 0;
        do
        {
            status = Argus::STATUS_OK;
            if (camera.iBufferStream)
            {
                Argus::Buffer* argusBuffer = camera.iBufferStream->acquireBuffer(kCaptureWaitTimeoutNs, &status);
                acquiredTscNs = readTscNowNs();
                if (argusBuffer)
                {
                    Argus::IBuffer* iBuffer = Argus::interface_cast<Argus::IBuffer>(argusBuffer);
                    DmaBuffer* dmaBuffer =
                        iBuffer ? const_cast<DmaBuffer*>(static_cast<const DmaBuffer*>(iBuffer->getClientData())) : NULL;
                    if (!dmaBuffer || !dmaBuffer->markArgusAcquired(argusBuffer))
                    {
                        camera.iBufferStream->releaseBuffer(argusBuffer);
                        failDomain(domain, "failed to resolve acquired direct DMABUF");
                        return std::unique_ptr<PendingFrame>();
                    }
                    sample->dmaBuffer.reset(dmaBuffer);
                }
            }
            else
            {
                EGLStream::Frame* frame = camera.iConsumer->acquireFrame(kCaptureWaitTimeoutNs, &status);
                acquiredTscNs = readTscNowNs();
                sample->frame.reset(frame);
            }
            if (status == Argus::STATUS_TIMEOUT && continuousAcquire && !stopping(domain))
            {
                const uint64_t limitNs = camera.hasLastFrameNumber ? kPrioritySteadyDroughtNs : kPriorityStartupDroughtNs;
                if (priorityFrameDroughtExpired(droughtStartNs, steadyNowNs(), camera.hasLastFrameNumber))
                {
                    std::fprintf(stderr, "Priority camera %s produced no frame for %.1f s (%s)\n", camera.label.c_str(),
                                 static_cast<double>(limitNs) / 1e9,
                                 camera.hasLastFrameNumber ? "steady-state drought" : "startup timeout");
                    failDomain(domain, "priority camera frame drought");
                    return std::unique_ptr<PendingFrame>();
                }
                continue;
            }
            break;
        } while (!stopping(domain));
        if ((!sample->frame && !sample->dmaBuffer.get()) || status != Argus::STATUS_OK)
        {
            if (stopping(domain) || (expectedStop && expectedStop->load()))
            {
                return std::unique_ptr<PendingFrame>();
            }
            failDomain(domain, camera.iBufferStream ? "failed to acquire direct buffer" : "failed to acquire EGL frame",
                       status);
            return std::unique_ptr<PendingFrame>();
        }
        const uint64_t acquiredNs = steadyNowNs();
        const bool timingRead =
            camera.iBufferStream ?
                readFrameTiming(sample->dmaBuffer.get()->getArgusBuffer(), camera.label.c_str(), &sample->timing) :
                readFrameTiming(sample->frame.get(), camera.label.c_str(), &sample->timing);
        if (!timingRead)
        {
            failDomain(domain, "failed to read frame metadata");
            return std::unique_ptr<PendingFrame>();
        }
        sample->timing.acquiredSteadyNs = acquiredNs;
        sample->timing.acquiredTscNs = acquiredTscNs;
        sample->duplicate = camera.hasLastFrameNumber && sample->timing.frameNumber == camera.lastFrameNumber;
        if (camera.hasLastFrameNumber && sample->timing.frameNumber > camera.lastFrameNumber + 1)
        {
            sample->timing.frameNumberGap = sample->timing.frameNumber - camera.lastFrameNumber - 1;
        }
        camera.lastFrameNumber = sample->timing.frameNumber;
        camera.hasLastFrameNumber = true;
        return sample;
    }

    void priorityCameraThreadMain(DomainState* domain, size_t flatIndex, size_t eye)
    {
        PriorityFrameProgress progress(steadyNowNs());
        while (!stopping(domain))
        {
            std::unique_ptr<PendingFrame> sample = acquireCameraFrame(domain, flatIndex, NULL, true);
            if (!sample)
                break;
            if (sample->duplicate)
            {
                const uint64_t frameNumber = sample->timing.frameNumber;
                // Return the acquired surface before rate-limiting retries
                // of the same frame number.
                sample.reset();
                if (progress.droughtExpired(steadyNowNs()))
                {
                    std::fprintf(
                        stderr, "Priority camera %s produced no new frame for %.1f s (repeated frame number %llu)\n",
                        m_cameras[flatIndex]->label.c_str(), static_cast<double>(progress.droughtLimitNs()) / 1e9,
                        static_cast<unsigned long long>(frameNumber));
                    failDomain(domain, "priority camera new-frame drought");
                    break;
                }
                std::this_thread::sleep_for(std::chrono::microseconds(100));
                continue;
            }
            progress.noteFreshFrame(steadyNowNs());
            std::unique_ptr<PendingFrame> stale;
            {
                std::lock_guard<std::mutex> lock(domain->mutex);
                if (domain->stop || domain->error || stopRequested())
                    break;
                LatestFrameSlot<PendingFrame>& slot = domain->headFrames[eye];
                if (slot.occupied())
                {
                    domain->headSlotOverwritesInterval.fetch_add(1);
                    domain->headSlotOverwritesTotal.fetch_add(1);
                }
                stale = slot.replace(std::move(sample));
            }
            // Recycle superseded Argus buffers outside the queue lock.
            stale.reset();
            domain->framesAvailable.notify_all();
        }
    }

    std::unique_ptr<PendingFrame> takePriorityCameraFrame(DomainState* domain, size_t eye)
    {
        std::unique_lock<std::mutex> lock(domain->mutex);
        while (!domain->headFrames[eye].occupied() && !domain->error && !domain->stop && !stopRequested())
        {
            domain->framesAvailable.wait_for(lock, std::chrono::milliseconds(kWaitPollMs));
        }
        if (domain->error || domain->stop || stopRequested())
            return std::unique_ptr<PendingFrame>();
        std::unique_ptr<PendingFrame> sample = domain->headFrames[eye].take();
        return sample;
    }

    bool acquireDomainSet(DomainState* domain,
                          std::vector<std::unique_ptr<PendingFrame>>* pending,
                          const std::atomic<bool>* expectedStop = NULL)
    {
        pending->clear();
        for (size_t member = 0; member < domain->cameraIndices.size(); ++member)
        {
            std::unique_ptr<PendingFrame> sample =
                acquireCameraFrame(domain, domain->cameraIndices[member], expectedStop);
            if (!sample)
                return false;
            pending->push_back(std::move(sample));
        }
        for (size_t member = 0; member < pending->size(); ++member)
        {
            if ((*pending)[member]->duplicate)
            {
                pending->clear();
                break;
            }
        }
        return true;
    }

    bool replacePriorityCameraFrame(DomainState* domain, PendingFrame* sample)
    {
        const size_t eye = sample->cameraFlatIndex == domain->stereoLeftFlatIndex ? 0 : 1;
        // Release the older surface before waiting for a replacement so it
        // cannot starve the producer's finite DMABUF pool.
        sample->frame.reset();
        sample->dmaBuffer.reset();
        std::unique_ptr<PendingFrame> replacement = takePriorityCameraFrame(domain, eye);
        if (!replacement)
            return false;
        sample->frame.reset(replacement->frame.release());
        sample->dmaBuffer.reset(replacement->dmaBuffer.release());
        sample->timing = replacement->timing;
        sample->duplicate = false;
        return true;
    }

    bool acquirePriorityStereoSet(DomainState* domain, std::vector<std::unique_ptr<PendingFrame>>* pending)
    {
        if (!domain || !pending || domain->priorityStereoStreamIndex >= m_streams.size())
        {
            return false;
        }
        pending->clear();
        std::unique_ptr<PendingFrame> leftSample = takePriorityCameraFrame(domain, 0);
        if (!leftSample)
            return false;
        std::unique_ptr<PendingFrame> rightSample = takePriorityCameraFrame(domain, 1);
        if (!rightSample)
            return false;
        pending->push_back(std::move(leftSample));
        pending->push_back(std::move(rightSample));

        PendingFrame* left = NULL;
        PendingFrame* right = NULL;
        for (size_t i = 0; i < pending->size(); ++i)
        {
            if ((*pending)[i]->cameraFlatIndex == domain->stereoLeftFlatIndex)
            {
                left = (*pending)[i].get();
            }
            else if ((*pending)[i]->cameraFlatIndex == domain->stereoRightFlatIndex)
            {
                right = (*pending)[i].get();
            }
        }
        if (!left || !right)
            return failDomain(domain, "priority stereo pair is incomplete");

        const uint64_t framePeriodNs = m_streams[domain->priorityStereoStreamIndex]->framePeriodNs;
        int64_t delta = 0;
        if (!stereoAlignmentDeltaNs(left->timing, right->timing, &delta))
        {
            return failDomain(domain, "head stereo alignment requires comparable SOF timestamps");
        }
        const int64_t initialDelta = delta;
        uint32_t leftDrops = 0;
        uint32_t rightDrops = 0;
        uint32_t reacquires = 0;
        uint64_t reacquireWaitNs = 0;
        while (reacquires < kStereoMaxAlignmentDrops)
        {
            const StereoAlignmentAction action =
                chooseStereoAlignmentAction(left->timing, right->timing, framePeriodNs, reacquires, &delta);
            PendingFrame* older = NULL;
            if (action == STEREO_ALIGNMENT_DROP_LEFT)
            {
                older = left;
                ++leftDrops;
            }
            else if (action == STEREO_ALIGNMENT_DROP_RIGHT)
            {
                older = right;
                ++rightDrops;
            }
            else
            {
                break;
            }
            const uint64_t reacquireStartNs = steadyNowNs();
            if (!replacePriorityCameraFrame(domain, older))
                return false;
            reacquireWaitNs += steadyNowNs() - reacquireStartNs;
            ++reacquires;
            if (!stereoAlignmentDeltaNs(left->timing, right->timing, &delta))
            {
                return failDomain(domain, "selectively reacquired head frame has no comparable SOF timestamp");
            }
        }
        const bool aligned = framePeriodNs != 0 && absoluteDeltaNs(delta) <= framePeriodNs / 2ULL;
        const uint64_t readyNs = steadyNowNs();
        const uint64_t oldestAcquiredNs = std::min(left->timing.acquiredSteadyNs, right->timing.acquiredSteadyNs);
        const uint64_t selectedAgeNs =
            oldestAcquiredNs != 0 && readyNs > oldestAcquiredNs ? readyNs - oldestAcquiredNs : 0;

        domain->stereoPairsInterval.fetch_add(1);
        domain->stereoPairsTotal.fetch_add(1);
        const uint64_t drops = leftDrops + rightDrops;
        if (drops > 0)
        {
            domain->stereoRepairPairsInterval.fetch_add(1);
            domain->stereoRepairPairsTotal.fetch_add(1);
        }
        domain->stereoLeftDropsInterval.fetch_add(leftDrops);
        domain->stereoLeftDropsTotal.fetch_add(leftDrops);
        domain->stereoRightDropsInterval.fetch_add(rightDrops);
        domain->stereoRightDropsTotal.fetch_add(rightDrops);
        domain->stereoReacquiresInterval.fetch_add(reacquires);
        domain->stereoReacquiresTotal.fetch_add(reacquires);
        domain->stereoReacquireWaitSumNsInterval.fetch_add(reacquireWaitNs);
        updateAtomicMaximum(&domain->stereoReacquireWaitMaxNsInterval, reacquireWaitNs);
        domain->stereoSelectedAgeSumNsInterval.fetch_add(selectedAgeNs);
        updateAtomicMaximum(&domain->stereoSelectedAgeMaxNsInterval, selectedAgeNs);
        const uint64_t initialDeltaNs = absoluteDeltaNs(initialDelta);
        const uint64_t finalDeltaNs = absoluteDeltaNs(delta);
        domain->stereoInitialDeltaSumNsInterval.fetch_add(initialDeltaNs);
        updateAtomicMaximum(&domain->stereoInitialDeltaMaxNsInterval, initialDeltaNs);
        domain->stereoFinalDeltaSumNsInterval.fetch_add(finalDeltaNs);
        updateAtomicMaximum(&domain->stereoFinalDeltaMaxNsInterval, finalDeltaNs);

        if (!aligned)
        {
            std::fprintf(
                stderr,
                "Head stereo pair could not be associated within half a %.3f us frame period after %u selective reacquisition(s): initial_delta_us=%.3f, final_delta_us=%.3f, drops(left/right)=%u/%u\n",
                static_cast<double>(framePeriodNs) / 1000.0, reacquires, static_cast<double>(initialDelta) / 1000.0,
                static_cast<double>(delta) / 1000.0, leftDrops, rightDrops);
            return failDomain(domain, "selective stereo frame association exhausted its retry budget");
        }

        return true;
    }

    void domainThreadMain(DomainState* domain)
    {
        while (!stopping(domain))
        {
            bool demandGranted = false;
            if (!domain->continuousFifo)
            {
                std::unique_lock<std::mutex> lock(domain->mutex);
                domain->slotAvailable.wait(
                    lock,
                    [domain]()
                    {
                        return domain->stop || stopRequested() ||
                               (!domain->hasPending && (!domain->demandDriven || domain->captureRequested));
                    });
                if (domain->stop || stopRequested())
                    break;
                demandGranted = domain->demandDriven;
                if (demandGranted)
                    domain->captureRequested = false;
            }
            if (!domain->priorityStereo &&
                !waitForCaptureComplete(domain, domain->continuousFifo ? &domain->stoppingRepeat : NULL))
                break;
            std::vector<std::unique_ptr<PendingFrame>> pending;
            const bool acquired =
                domain->priorityStereo ?
                    acquirePriorityStereoSet(domain, &pending) :
                    acquireDomainSet(domain, &pending, domain->continuousFifo ? &domain->stoppingRepeat : NULL);
            if (!acquired)
                break;
            if (domain->continuousFifo && domain->stoppingRepeat.load())
            {
                pending.clear();
                continue;
            }
            if (pending.empty())
            {
                if (demandGranted)
                {
                    std::lock_guard<std::mutex> lock(domain->mutex);
                    domain->captureRequested = true;
                    domain->slotAvailable.notify_one();
                }
                continue;
            }
            if (domain->continuousFifo)
            {
                size_t overflowDiscards = 0;
                size_t candidateDepth = 0;
                {
                    std::lock_guard<std::mutex> lock(domain->mutex);
                    if (domain->stop || stopRequested())
                        break;
                    while (domain->candidates.size() >= kFifoCandidateWindowLength)
                    {
                        domain->candidates.pop_front();
                        ++overflowDiscards;
                    }
                    domain->candidates.push_back(std::move(pending));
                    candidateDepth = domain->candidates.size();
                }
                if (overflowDiscards > 0)
                {
                    domain->candidateOverflowDiscardsInterval.fetch_add(overflowDiscards);
                    domain->candidateOverflowDiscardsTotal.fetch_add(overflowDiscards);
                }
                updateAtomicMaximum(&domain->candidateDepthPeakInterval, candidateDepth);
                domain->framesAvailable.notify_all();
                continue;
            }
            {
                std::lock_guard<std::mutex> lock(domain->mutex);
                if (domain->stop || stopRequested())
                    break;
                domain->pending.swap(pending);
                domain->hasPending = true;
            }
            domain->framesAvailable.notify_one();
        }
    }

    bool takeDomain(size_t domainIndex, std::vector<std::unique_ptr<PendingFrame>>* pending)
    {
        if (!pending || domainIndex >= m_domains.size())
            return false;
        DomainState& domain = *m_domains[domainIndex];
        std::unique_lock<std::mutex> lock(domain.mutex);
        while (!(domain.continuousFifo ? !domain.candidates.empty() : domain.hasPending) && !domain.error &&
               !domain.stop && !stopRequested())
        {
            domain.framesAvailable.wait_for(lock, std::chrono::milliseconds(kWaitPollMs));
        }
        if (domain.continuousFifo)
        {
            if (domain.candidates.empty())
                return false;
            pending->clear();
            pending->swap(domain.candidates.front());
            domain.candidates.pop_front();
            return true;
        }
        if (!domain.hasPending)
            return false;
        pending->clear();
        pending->swap(domain.pending);
        domain.hasPending = false;
        lock.unlock();
        domain.slotAvailable.notify_one();
        return true;
    }

    uint64_t candidateSofTscNs(const std::vector<std::unique_ptr<PendingFrame>>& pending) const
    {
        if (pending.empty())
            return 0;
        uint64_t earliest = std::numeric_limits<uint64_t>::max();
        uint64_t latest = 0;
        for (size_t i = 0; i < pending.size(); ++i)
        {
            if (!pending[i] || !pending[i]->timing.hasSensorSofTimestampTsc ||
                pending[i]->timing.sensorSofTimestampTscNs == 0)
            {
                return 0;
            }
            earliest = std::min(earliest, pending[i]->timing.sensorSofTimestampTscNs);
            latest = std::max(latest, pending[i]->timing.sensorSofTimestampTscNs);
        }
        return earliest + (latest - earliest) / 2ULL;
    }

    bool candidateReadyLocked(const DomainState& domain, uint64_t referenceSofTscNs, uint64_t framePeriodNs) const
    {
        if (domain.candidates.empty())
            return false;
        if (referenceSofTscNs == 0)
            return true;
        const uint64_t newestTscNs = candidateSofTscNs(domain.candidates.back());
        if (newestTscNs == 0 || newestTscNs >= referenceSofTscNs || framePeriodNs == 0)
        {
            return true;
        }
        return referenceSofTscNs - newestTscNs <= framePeriodNs / 2ULL;
    }

    bool movePendingToCaptured(std::vector<std::unique_ptr<PendingFrame>>* pending,
                               std::vector<CapturedFrameSet>* captured)
    {
        if (!pending || !captured || captured->size() != m_streams.size())
            return false;
        for (size_t i = 0; i < pending->size(); ++i)
        {
            PendingFrame& sample = *(*pending)[i];
            CameraState& camera = *m_cameras[sample.cameraFlatIndex];
            CapturedFrameSet& streamFrames = (*captured)[camera.streamIndex];
            if (camera.cameraIndex >= streamFrames.frames.size() ||
                camera.cameraIndex >= streamFrames.dmaBuffers.size() || camera.cameraIndex >= streamFrames.timings.size())
            {
                return false;
            }
            streamFrames.frames[camera.cameraIndex].reset(sample.frame.release());
            streamFrames.dmaBuffers[camera.cameraIndex].reset(sample.dmaBuffer.release());
            streamFrames.timings[camera.cameraIndex] = sample.timing;
        }
        return true;
    }

    void recordStats(StreamState* stream, const CapturedFrameSet& frames)
    {
        if (m_config.runtime.logIntervalSeconds == 0)
            return;
        const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
        if (stream->intervalFrames == 0)
            stream->intervalStart = now;
        ++stream->intervalFrames;
        ++stream->totalFrames;
        for (size_t cameraIndex = 0; cameraIndex < frames.timings.size() && cameraIndex < stream->acquireSamples.size();
             ++cameraIndex)
        {
            const FrameTiming& timing = frames.timings[cameraIndex];
            if (stream->hasLastSelectedFrameNumber[cameraIndex] &&
                timing.frameNumber > stream->lastSelectedFrameNumbers[cameraIndex] + 1)
            {
                stream->selectedFrameGapsInterval[cameraIndex] +=
                    timing.frameNumber - stream->lastSelectedFrameNumbers[cameraIndex] - 1;
            }
            stream->lastSelectedFrameNumbers[cameraIndex] = timing.frameNumber;
            stream->hasLastSelectedFrameNumber[cameraIndex] = true;
            if (!timing.hasSensorSofTimestampTsc || timing.acquiredTscNs < timing.sensorSofTimestampTscNs)
            {
                continue;
            }
            const uint64_t delayNs = timing.acquiredTscNs - timing.sensorSofTimestampTscNs;
            ++stream->acquireSamples[cameraIndex];
            stream->acquireSumNs[cameraIndex] += static_cast<double>(delayNs);
            stream->acquireMaxNs[cameraIndex] = std::max(stream->acquireMaxNs[cameraIndex], delayNs);
        }
        if (frames.timings.size() == 2 && frames.timings[0].hasSensorSofTimestampTsc &&
            frames.timings[1].hasSensorSofTimestampTsc)
        {
            const int64_t value =
                deltaNs(frames.timings[1].sensorSofTimestampTscNs, frames.timings[0].sensorSofTimestampTscNs);
            if (stream->intervalTscSamples == 0)
            {
                stream->intervalTscMinNs = value;
                stream->intervalTscMaxNs = value;
            }
            else
            {
                stream->intervalTscMinNs = std::min(stream->intervalTscMinNs, value);
                stream->intervalTscMaxNs = std::max(stream->intervalTscMaxNs, value);
            }
            stream->intervalTscSumNs += static_cast<double>(value);
            ++stream->intervalTscSamples;
        }
        const double elapsed = std::chrono::duration<double>(now - stream->intervalStart).count();
        if (elapsed < static_cast<double>(m_config.runtime.logIntervalSeconds))
            return;
        if (verboseLoggingEnabled() || stream->config.type == STREAM_TYPE_STEREO)
        {
            std::printf("argus_sender[capture:%s]: %.2f FPS, total=%llu", stream->config.id.c_str(),
                        static_cast<double>(stream->intervalFrames) / elapsed,
                        static_cast<unsigned long long>(stream->totalFrames));
            if (stream->intervalTscSamples > 0)
            {
                std::printf(", tsc_delta_us(avg/min/max,right-left)=%.3f/%.3f/%.3f",
                            stream->intervalTscSumNs / static_cast<double>(stream->intervalTscSamples) / 1000.0,
                            static_cast<double>(stream->intervalTscMinNs) / 1000.0,
                            static_cast<double>(stream->intervalTscMaxNs) / 1000.0);
            }
            if (verboseLoggingEnabled())
            {
                std::printf(", camera_settings(latest exposure_us/analog_gain/isp_digital_gain)=");
                for (size_t cameraIndex = 0; cameraIndex < frames.timings.size(); ++cameraIndex)
                {
                    const FrameTiming& timing = frames.timings[cameraIndex];
                    const CameraState& camera = *m_cameras[stream->cameraIndices[cameraIndex]];
                    std::printf("%s%s:", cameraIndex == 0 ? "" : ",", camera.label.c_str());
                    if (timing.hasSensorExposureTime)
                    {
                        std::printf("%.3f", static_cast<double>(timing.sensorExposureTimeNs) / 1000.0);
                    }
                    else
                        std::printf("n/a");
                    std::printf("/");
                    if (timing.hasSensorAnalogGain)
                        std::printf("%.3f", timing.sensorAnalogGain);
                    else
                        std::printf("n/a");
                    std::printf("/");
                    if (timing.hasIspDigitalGain)
                        std::printf("%.3f", timing.ispDigitalGain);
                    else
                        std::printf("n/a");
                }
                std::printf(", awb(latest mode/cct_K/gain_r:g_even:g_odd:b)=");
                for (size_t cameraIndex = 0; cameraIndex < frames.timings.size(); ++cameraIndex)
                {
                    const FrameTiming& timing = frames.timings[cameraIndex];
                    const CameraState& camera = *m_cameras[stream->cameraIndices[cameraIndex]];
                    std::printf("%s%s:", cameraIndex == 0 ? "" : ",", camera.label.c_str());
                    if (timing.hasAwbMode)
                        std::printf("%s", timing.awbModeName);
                    else
                        std::printf("n/a");
                    std::printf("/");
                    if (timing.hasAwbCct)
                        std::printf("%u", timing.awbCct);
                    else
                        std::printf("n/a");
                    std::printf("/");
                    if (timing.hasAwbGains)
                    {
                        std::printf("%.3f:%.3f:%.3f:%.3f", timing.awbGainR, timing.awbGainGEven, timing.awbGainGOdd,
                                    timing.awbGainB);
                    }
                    else
                        std::printf("n/a");
                }
            }
            std::printf("\n");
        }
        std::printf("argus_sender[sof-acquire:%s]: ", stream->config.id.c_str());
        for (size_t cameraIndex = 0; cameraIndex < stream->acquireSamples.size(); ++cameraIndex)
        {
            const CameraState& camera = *m_cameras[stream->cameraIndices[cameraIndex]];
            std::printf("%s%s=", cameraIndex == 0 ? "" : ",", camera.label.c_str());
            if (stream->acquireSamples[cameraIndex] > 0)
            {
                std::printf(
                    "%.3f/%.3fms(avg/max,n=%llu)",
                    stream->acquireSumNs[cameraIndex] / static_cast<double>(stream->acquireSamples[cameraIndex]) / 1.0e6,
                    static_cast<double>(stream->acquireMaxNs[cameraIndex]) / 1.0e6,
                    static_cast<unsigned long long>(stream->acquireSamples[cameraIndex]));
            }
            else
            {
                std::printf("n/a");
            }
            std::printf("[selected_frame_gaps=%llu]",
                        static_cast<unsigned long long>(stream->selectedFrameGapsInterval[cameraIndex]));
            stream->selectedFrameGapsInterval[cameraIndex] = 0;
            stream->acquireSamples[cameraIndex] = 0;
            stream->acquireSumNs[cameraIndex] = 0.0;
            stream->acquireMaxNs[cameraIndex] = 0;
        }
        std::printf("\n");
        stream->intervalStart = now;
        stream->intervalFrames = 0;
        stream->intervalTscSamples = 0;
        stream->intervalTscSumNs = 0.0;
    }

    void cleanup()
    {
        for (size_t i = 0; i < m_domains.size(); ++i)
        {
            if (!m_domains[i])
                continue;
            m_domains[i]->pending.clear();
            m_domains[i]->candidates.clear();
            for (size_t eye = 0; eye < m_domains[i]->headFrames.size(); ++eye)
            {
                m_domains[i]->headFrames[eye].reset();
            }
            for (size_t sessionIndex = 0; sessionIndex < m_domains[i]->sessions.size(); ++sessionIndex)
            {
                CaptureSessionState& session = *m_domains[i]->sessions[sessionIndex];
                session.request.reset();
                session.eventQueue.reset();
                session.iEventQueue = NULL;
                session.iEventProvider = NULL;
            }
        }
        for (size_t i = 0; i < m_cameras.size(); ++i)
        {
            m_cameras[i]->consumer.reset();
            m_cameras[i]->iConsumer = NULL;
            m_cameras[i]->argusBuffers.clear();
            m_cameras[i]->iBufferStream = NULL;
            m_cameras[i]->dmaBuffers.clear();
            m_cameras[i]->outputStream.reset();
        }
        for (size_t i = 0; i < m_domains.size(); ++i)
        {
            if (!m_domains[i])
                continue;
            for (size_t sessionIndex = 0; sessionIndex < m_domains[i]->sessions.size(); ++sessionIndex)
            {
                m_domains[i]->sessions[sessionIndex]->session.reset();
                m_domains[i]->sessions[sessionIndex]->iSession = NULL;
            }
            m_domains[i]->sessions.clear();
        }
    }

    AppConfig m_config;
    std::vector<std::unique_ptr<StreamState>> m_streams;
    std::vector<std::unique_ptr<CameraState>> m_cameras;
    std::vector<std::vector<size_t>> m_cameraLookup;
    std::vector<std::unique_ptr<DomainState>> m_domains;
    bool m_started;
    std::atomic<bool> m_error;
};

CaptureManager::CaptureManager(const AppConfig& config) : m_impl(new Impl(config))
{
}

CaptureManager::~CaptureManager()
{
}
bool CaptureManager::initialize(Argus::CameraProvider* provider, EGLDisplay display)
{
    return m_impl->initialize(provider, display);
}
bool CaptureManager::startPriorityDomain(size_t priorityDomainIndex)
{
    return m_impl->startPriorityDomain(priorityDomainIndex);
}
bool CaptureManager::startSecondaryDomains(size_t priorityDomainIndex)
{
    return m_impl->startSecondaryDomains(priorityDomainIndex);
}
bool CaptureManager::discardDomainWarmup(size_t domainIndex, uint32_t frameCount)
{
    return m_impl->discardDomainWarmup(domainIndex, frameCount);
}
bool CaptureManager::discardWarmup(uint32_t frameCount)
{
    return m_impl->discardWarmup(frameCount);
}
bool CaptureManager::waitForDomainSynchronization(size_t domainIndex,
                                                  uint32_t maxFrameSets,
                                                  uint32_t requiredConsecutive,
                                                  uint64_t toleranceNs)
{
    return m_impl->waitForDomainSynchronization(domainIndex, maxFrameSets, requiredConsecutive, toleranceNs);
}
bool CaptureManager::requestDomainCapture(size_t domainIndex)
{
    return m_impl->requestDomainCapture(domainIndex);
}
void CaptureManager::prepareCaptured(std::vector<CapturedFrameSet>* captured) const
{
    m_impl->prepareCaptured(captured);
}
bool CaptureManager::acquireDomain(size_t domainIndex, std::vector<CapturedFrameSet>* captured)
{
    return m_impl->acquireDomain(domainIndex, captured);
}
bool CaptureManager::acquireDomainClosestTo(size_t domainIndex,
                                            uint64_t referenceSofTscNs,
                                            std::vector<CapturedFrameSet>* captured)
{
    return m_impl->acquireDomainClosestTo(domainIndex, referenceSofTscNs, captured);
}
bool CaptureManager::reacquireDomain(size_t domainIndex, std::vector<CapturedFrameSet>* captured)
{
    return m_impl->reacquireDomain(domainIndex, captured);
}
void CaptureManager::recordCaptured(size_t streamIndex, const CapturedFrameSet& frames)
{
    m_impl->recordCaptured(streamIndex, frames);
}
bool CaptureManager::stop()
{
    return m_impl->stop();
}
void CaptureManager::releaseResources()
{
    m_impl->releaseResources();
}
bool CaptureManager::hasError() const
{
    return m_impl->hasError();
}
size_t CaptureManager::streamCount() const
{
    return m_impl->streamCount();
}
size_t CaptureManager::domainCount() const
{
    return m_impl->domainCount();
}
size_t CaptureManager::streamDomain(size_t streamIndex) const
{
    return m_impl->streamDomain(streamIndex);
}
size_t CaptureManager::domainFor(size_t streamIndex, size_t cameraIndex) const
{
    return m_impl->domainFor(streamIndex, cameraIndex);
}
bool CaptureManager::domainUsesContinuousFifo(size_t domainIndex) const
{
    return m_impl->domainUsesContinuousFifo(domainIndex);
}
bool CaptureManager::hasContinuousFifoDomains() const
{
    return m_impl->hasContinuousFifoDomains();
}
void CaptureManager::logFifoCandidateStats()
{
    m_impl->logFifoCandidateStats();
}
const ResolvedOutputGeometry& CaptureManager::geometry(size_t streamIndex) const
{
    return m_impl->geometry(streamIndex);
}
uint32_t CaptureManager::nominalFps(size_t streamIndex) const
{
    return m_impl->nominalFps(streamIndex);
}
uint64_t CaptureManager::framePeriodNs(size_t streamIndex) const
{
    return m_impl->framePeriodNs(streamIndex);
}

} // namespace argus_sender
