/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

// SIPL capture. The ISP0 surface is handed on as native NV12 and returned to SIPL when
// the consumer drops its lease; there is no RGBA conversion. Needs C++17 because the SIPL headers do.

#include "SiplCapture.h"

#include "ScopedRelease.h"
#include "SensorClock.h"

#include <NvSIPLCamera.hpp>
#include <NvSIPLCameraQuery.hpp>
#include <NvSIPLClient.hpp>
#include <NvSIPLCommon.hpp>
#include <NvSIPLPipelineMgr.hpp>
#include <NvSIPLTrace.hpp>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <nvbufsurface.h>
#include <nvbufsurface_nvscibuf.h>
#include <nvscibuf.h>
#include <nvscisync.h>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <variant>

namespace argus_sender
{
namespace
{

using nvsipl::INvSIPLCamera;
using nvsipl::INvSIPLCameraQuery;
using nvsipl::INvSIPLClient;
using nvsipl::NVSIPL_STATUS_OK;
using nvsipl::SIPLStatus;

constexpr auto kIsp0 = INvSIPLClient::ConsumerDesc::OutputType::ISP0;
constexpr auto kIcp = INvSIPLClient::ConsumerDesc::OutputType::ICP;

// SIPL allocates and registers ICP capture buffers even when only an ISP output is consumed; without them
// RegisterImages fails with a bare NVSIPL_STATUS_ERROR.
constexpr uint32_t kIcpBuffers = 6;

uint64_t monotonicNs()
{
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

// SIPL reports most failures as NVSIPL_STATUS_ERROR, which names nothing; its own log has the reason above a
// threshold that defaults to silent. CAMERA_SENDER_SIPL_TRACE=1..4 raises it (error, warn, info, debug).
void initSiplTrace()
{
    const char* level = std::getenv("CAMERA_SENDER_SIPL_TRACE");
    if (level == nullptr || std::atoi(level) <= 0)
        return;
    nvsipl::INvSIPLTrace* trace = nvsipl::INvSIPLTrace::GetInstance();
    if (trace != nullptr)
        trace->SetLevel(static_cast<nvsipl::INvSIPLTrace::TraceLevel>(std::atoi(level)));
}

void checkSipl(SIPLStatus status, const char* what)
{
    if (status == NVSIPL_STATUS_OK)
        return;
    std::ostringstream oss;
    oss << what << " failed with SIPLStatus " << static_cast<int>(status);
    throw std::runtime_error(oss.str());
}

void checkSci(NvSciError error, const char* what)
{
    if (error == NvSciError_Success)
        return;
    std::ostringstream oss;
    oss << what << " failed with NvSciError " << static_cast<int>(error);
    throw std::runtime_error(oss.str());
}

// ISP0 can only produce block-linear, semi-planar, BT.709 extended-range YUV 4:2:0: GetImageAttributes rejects an
// attribute list asking for pitch-linear or another colour standard.
NvSciError setSiplBufferAttributes(NvSciBufAttrList attrList)
{
    NvSciBufType bufType = NvSciBufType_Image;
    NvSciBufAttrValAccessPerm accessPerm = NvSciBufAccessPerm_ReadWrite;
    // Must be true: NvmmImageFillNvSciBufAttrList sets NeedCpuAccess on the NvBufSurface list, reconciliation ORs it
    // in, and a producer list saying false then fails RegisterImages with a bare NVSIPL_STATUS_ERROR.
    bool cpuAccess = true;
    bool cpuCache = true;
    NvSciBufSurfMemLayout memLayout = NvSciSurfMemLayout_SemiPlanar;
    NvSciBufAttrValImageLayoutType layoutType = NvSciBufImage_BlockLinearType;
    NvSciBufSurfType surfType = NvSciSurfType_YUV;
    NvSciBufSurfSampleType sampleType = NvSciSurfSampleType_420;
    NvSciBufSurfBPC surfBpc = NvSciSurfBPC_8;
    NvSciBufSurfComponentOrder compOrder = NvSciSurfComponentOrder_YUV;
    NvSciBufAttrValColorStd colorStd[] = { NvSciColorStd_REC709_ER };

    NvSciBufAttrKeyValuePair kvp[] = {
        { NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType) },
        { NvSciBufGeneralAttrKey_RequiredPerm, &accessPerm, sizeof(accessPerm) },
        { NvSciBufGeneralAttrKey_NeedCpuAccess, &cpuAccess, sizeof(cpuAccess) },
        { NvSciBufGeneralAttrKey_EnableCpuCache, &cpuCache, sizeof(cpuCache) },
        { NvSciBufImageAttrKey_Layout, &layoutType, sizeof(layoutType) },
        { NvSciBufImageAttrKey_SurfType, &surfType, sizeof(surfType) },
        { NvSciBufImageAttrKey_SurfMemLayout, &memLayout, sizeof(memLayout) },
        { NvSciBufImageAttrKey_SurfSampleType, &sampleType, sizeof(sampleType) },
        { NvSciBufImageAttrKey_SurfBPC, &surfBpc, sizeof(surfBpc) },
        { NvSciBufImageAttrKey_SurfComponentOrder, &compOrder, sizeof(compOrder) },
        { NvSciBufImageAttrKey_SurfColorStd, &colorStd, sizeof(colorStd) },
    };
    return NvSciBufAttrListSetAttrs(attrList, kvp, sizeof(kvp) / sizeof(kvp[0]));
}

// The NvBufSurface view of the same buffer must be described before allocation, or RegisterImages rejects the buffer
// with NVSIPL_STATUS_INVALID_STATE. Adapted from SetNvBufNvSciBufAttributes in NVIDIA's NvNvSciBufHelper.cpp
// (BSD-3-Clause); only the semi-planar BT.709 extended-range case is reachable on this ISP path.
NvSciError setNvBufAttributes(NvSciBufAttrList nvbufAttrs, NvSciBufAttrList siplAttrs, int gpuId)
{
    NvSciBufAttrKeyValuePair img[] = {
        { NvSciBufImageAttrKey_SurfWidthBase, nullptr, 0 },  { NvSciBufImageAttrKey_SurfHeightBase, nullptr, 0 },
        { NvSciBufImageAttrKey_SurfSampleType, nullptr, 0 }, { NvSciBufImageAttrKey_SurfColorStd, nullptr, 0 },
        { NvSciBufImageAttrKey_SurfMemLayout, nullptr, 0 },  { NvSciBufImageAttrKey_SurfBPC, nullptr, 0 },
        { NvSciBufImageAttrKey_Layout, nullptr, 0 },
    };
    const NvSciError error = NvSciBufAttrListGetAttrs(siplAttrs, img, sizeof(img) / sizeof(img[0]));
    if (error != NvSciError_Success)
        return error;

    const uint32_t width = *static_cast<const uint32_t*>(img[0].value);
    const uint32_t height = *static_cast<const uint32_t*>(img[1].value);
    const NvSciBufSurfSampleType sample = *static_cast<const NvSciBufSurfSampleType*>(img[2].value);
    const NvSciBufAttrValColorStd color = *static_cast<const NvSciBufAttrValColorStd*>(img[3].value);
    const NvSciBufSurfMemLayout mem = *static_cast<const NvSciBufSurfMemLayout*>(img[4].value);
    const NvSciBufSurfBPC bpc = *static_cast<const NvSciBufSurfBPC*>(img[5].value);
    const NvSciBufAttrValImageLayoutType layout = *static_cast<const NvSciBufAttrValImageLayoutType*>(img[6].value);

    if (color != NvSciColorStd_REC709_ER || sample != NvSciSurfSampleType_420 || bpc != NvSciSurfBPC_8 ||
        mem != NvSciSurfMemLayout_SemiPlanar)
    {
        std::fprintf(stderr, "[sipl] no NV12 NvBufSurface format for colourStd=%d sample=%d bpc=%d memLayout=%d\n",
                     static_cast<int>(color), static_cast<int>(sample), static_cast<int>(bpc), static_cast<int>(mem));
        return NvSciError_BadParameter;
    }

    NvmmImageParams params{};
    params.gpuId = static_cast<uint32_t>(gpuId);
    params.width = width;
    params.height = height;
    params.colorFormat = NVBUF_COLOR_FORMAT_NV12_709_ER;
    params.layout = (layout == NvSciBufImage_PitchLinearType) ? NVBUF_LAYOUT_PITCH : NVBUF_LAYOUT_BLOCK_LINEAR;
    params.displayscanformat = NVBUF_DISPLAYSCANFORMAT_PROGRESSIVE;
    params.isProtected = false;
    return (NvmmImageFillNvSciBufAttrList(&params, nvbufAttrs) == 0) ? NvSciError_Success : NvSciError_BadParameter;
}

std::vector<uint8_t> loadNito(const std::string& dir, const std::string& module)
{
    const std::string path = dir + "/" + module + ".nito";
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in)
        throw std::runtime_error("cannot open NITO file " + path + " (the ISP has no tuning without it)");
    const std::streamsize size = static_cast<std::streamsize>(in.tellg());
    in.seekg(0);
    std::vector<uint8_t> blob(static_cast<size_t>(size));
    if (!in.read(reinterpret_cast<char*>(blob.data()), size))
        throw std::runtime_error("short read on NITO file " + path);
    return blob;
}

// module -> variant -> sensorConfigs -> variant; `id` is the pipeline index SetPipelineCfg wants.
std::vector<CameraDescriptor> flatten(const nvsipl::sensorconfig::SensorSystemConfig& config)
{
    std::vector<CameraDescriptor> out;
    for (const auto& module : config.modules)
    {
        std::visit(
            [&](const auto& moduleType)
            {
                for (const auto& sensorVariant : moduleType.sensorConfigs)
                {
                    std::visit(
                        [&](const auto& sensor)
                        {
                            CameraDescriptor info;
                            info.id = sensor.id;
                            info.name = module.name;
                            if (!sensor.vcInfoList.empty())
                            {
                                const auto& vc = sensor.vcInfoList.front();
                                info.width = vc.resolution.width;
                                info.height = vc.resolution.height;
                                info.fps = static_cast<double>(vc.fps);
                            }
                            out.push_back(std::move(info));
                        },
                        sensorVariant);
                }
            },
            module.moduleType);
    }
    return out;
}

} // namespace

std::vector<CameraDescriptor> querySiplCameras(const std::string& platformConfigJson,
                                               const std::string& platformConfigName,
                                               const std::vector<uint32_t>& linkMasks)
{
    std::unique_ptr<INvSIPLCameraQuery> api = INvSIPLCameraQuery::GetInstance();
    if (!api)
        throw std::runtime_error("INvSIPLCameraQuery::GetInstance returned null");
    checkSipl(api->ParseDatabase(), "ParseDatabase");
    checkSipl(api->ParseJsonFile(platformConfigJson), "ParseJsonFile");
    nvsipl::sensorconfig::SensorSystemConfig config;
    checkSipl(api->GetSensorSystemConfig(platformConfigName, config), "GetSensorSystemConfig");
    if (!linkMasks.empty())
        checkSipl(api->ApplyMask(config, linkMasks), "ApplyMask");
    return flatten(config);
}

struct SiplCapture::Impl
{
    // One ISP0 buffer. SIPL owns it between registration and Stop; the NvBufSurface view is created once.
    struct Slot
    {
        NvSciBufObj buf = nullptr;
        NvBufSurface* surface = nullptr;
    };

    // The newest frame the consumer has not taken. A frame replaced before it is taken goes straight back to SIPL.
    struct Pending
    {
        INvSIPLClient::INvSIPLBuffer* buffer = nullptr;
        const Slot* slot = nullptr;
        uint64_t sequence = 0;
        // The capture timestamp as a platform TSC in nanoseconds; 0 unless the mapping check passed.
        uint64_t sensorTscNs = 0;
        uint64_t readyMonotonicNs = 0;
        bool captureMapped = false;
        int64_t captureMonotonicNs = 0;
        int64_t captureUncertaintyNs = 0;
        int64_t captureAnchorAgeNs = 0;
        int64_t captureRealtimeNs = 0;
    };

    struct Sensor
    {
        CameraDescriptor info;
        nvsipl::NvSIPLPipelineConfiguration pipelineConfig{};
        nvsipl::NvSIPLPipelineQueues queues{};
        NvSciBufAttrList bufAttrs = nullptr;
        std::vector<NvSciBufObj> bufObjects;
        // Never read; allocated because SIPL requires it.
        NvSciBufAttrList icpAttrs = nullptr;
        std::vector<NvSciBufObj> icpBufObjects;
        std::vector<Slot> slots;
        NvSciSyncObj syncObj = nullptr;
        // One wait context can wait on only one fence at a time, so each sensor's frame thread has its own.
        NvSciSyncCpuWaitContext cpuWaitContext = nullptr;

        // Places the sensor's capture timestamp on CLOCK_MONOTONIC; used on this sensor's frame thread only.
        std::unique_ptr<SensorClockMap> clockMap;
        std::string lastMappingReason;

        std::mutex mutex;
        bool hasPending = false;
        Pending pending;
        uint64_t received = 0;

        std::thread frameThread;
        std::thread eventThread;
    };

    explicit Impl(const SiplConfig& cfg) : config(cfg)
    {
    }

    SiplConfig config;
    NvSciBufModule bufModule = nullptr;
    NvSciSyncModule syncModule = nullptr;

    std::unique_ptr<INvSIPLCamera> camera;
    std::unique_ptr<INvSIPLCameraQuery> queryApi;
    std::unique_ptr<nvsipl::sensorconfig::SensorSystemConfig> systemConfig;
    std::vector<CameraDescriptor> descriptors;
    std::vector<std::unique_ptr<Sensor>> sensors;

    // Signalled whenever a frame becomes pending or capture fails, so waitForFrame needs no polling.
    std::mutex wakeMutex;
    std::condition_variable wake;

    // The platform counter's frequency, read once; 0 where it is not readable.
    int64_t counterHz = 0;

    std::atomic<bool> running{ false };
    std::atomic<bool> failed{ false };
    std::mutex errorMutex;
    std::string failureMessage;

    Sensor* find(uint32_t id)
    {
        for (auto& sensor : sensors)
        {
            if (sensor->info.id == id)
                return sensor.get();
        }
        return nullptr;
    }

    void setFailure(const std::string& message)
    {
        std::lock_guard<std::mutex> guard(errorMutex);
        if (failureMessage.empty())
            failureMessage = message;
        failed.store(true);
        std::lock_guard<std::mutex> wakeGuard(wakeMutex);
        wake.notify_all();
    }

    void throwIfFailed()
    {
        if (!failed.load())
            return;
        std::lock_guard<std::mutex> guard(errorMutex);
        throw std::runtime_error(failureMessage.empty() ? "SIPL capture failed" : failureMessage);
    }

    void configure();
    void allocateBuffers(Sensor& sensor);
    void allocateIcpBuffers(Sensor& sensor);
    void registerBuffers(Sensor& sensor);
    void allocateSync(Sensor& sensor);
    void registerNito(Sensor& sensor);
    void frameLoop(Sensor& sensor);
    void eventLoop(Sensor& sensor);
    void dropPending(Sensor& sensor);
    void joinThreads();
    void cleanup();
};

void SiplCapture::Impl::configure()
{
    counterHz = readCounterHz();
    checkSci(NvSciBufModuleOpen(&bufModule), "NvSciBufModuleOpen");
    checkSci(NvSciSyncModuleOpen(&syncModule), "NvSciSyncModuleOpen");

    initSiplTrace();
    queryApi = INvSIPLCameraQuery::GetInstance();
    if (!queryApi)
        throw std::runtime_error("INvSIPLCameraQuery::GetInstance returned null");
    checkSipl(queryApi->ParseDatabase(), "ParseDatabase");
    checkSipl(queryApi->ParseJsonFile(config.platformConfigJson), "ParseJsonFile");

    systemConfig = std::make_unique<nvsipl::sensorconfig::SensorSystemConfig>();
    checkSipl(queryApi->GetSensorSystemConfig(config.platformConfigName, *systemConfig), "GetSensorSystemConfig");
    if (config.linkMasks.empty())
        throw std::runtime_error("link masks are required for a GMSL platform configuration");
    checkSipl(queryApi->ApplyMask(*systemConfig, config.linkMasks), "ApplyMask");
    if (systemConfig->modules.empty())
        throw std::runtime_error("no modules left in '" + config.platformConfigName +
                                 "' after applying the link masks; check the masks against the configuration");

    descriptors = flatten(*systemConfig);
    if (descriptors.empty())
        throw std::runtime_error("platform configuration '" + config.platformConfigName + "' declares no sensors");

    camera = INvSIPLCamera::GetInstance();
    if (!camera)
        throw std::runtime_error("INvSIPLCamera::GetInstance returned null");
    checkSipl(camera->SetPlatformCfg(*systemConfig), "SetPlatformCfg");

    for (const auto& info : descriptors)
    {
        std::unique_ptr<Sensor> sensor(new Sensor());
        sensor->info = info;
        checkSci(NvSciSyncCpuWaitContextAlloc(syncModule, &sensor->cpuWaitContext), "NvSciSyncCpuWaitContextAlloc");
        SensorClockMap::Config clockConfig;
        clockConfig.counterHz = counterHz;
        sensor->clockMap.reset(new SensorClockMap(clockConfig, readCounterPair));
        // ISP0 only: ICP would double the capture bandwidth for a raw stream nothing here consumes.
        sensor->pipelineConfig.captureOutputRequested = false;
        sensor->pipelineConfig.isp0OutputRequested = true;
        sensor->pipelineConfig.isp1OutputRequested = false;
        sensor->pipelineConfig.isp2OutputRequested = false;
        // Subframe is enabled by default and then needs a slice count that is not set here.
        sensor->pipelineConfig.disableSubframe = true;
        sensor->pipelineConfig.bufferCfg.maxIsp0BufferCount = config.isp0Buffers;
        checkSipl(camera->SetPipelineCfg(info.id, sensor->pipelineConfig, sensor->queues), "SetPipelineCfg");
        sensors.push_back(std::move(sensor));
    }

    // Order is fixed by SIPL: what describes a pipeline (GetImageAttributes, FillNvSciSyncAttrList) comes before
    // Init(), what binds a resource to it (RegisterImages, RegisterNvSciSyncObj) after. Init() still returns OK
    // out of order, and the first Register call then fails with NVSIPL_STATUS_INVALID_STATE.
    for (auto& sensor : sensors)
    {
        allocateIcpBuffers(*sensor);
        allocateBuffers(*sensor);
        allocateSync(*sensor);
    }
    checkSipl(camera->Init(), "INvSIPLCamera::Init");
    for (auto& sensor : sensors)
        registerBuffers(*sensor);
}

void SiplCapture::Impl::allocateBuffers(Sensor& sensor)
{
    NvSciBufAttrList siplAttrs = nullptr;
    NvSciBufAttrList nvbufAttrs = nullptr;
    NvSciBufAttrList conflict = nullptr;
    checkSci(NvSciBufAttrListCreate(bufModule, &siplAttrs), "NvSciBufAttrListCreate sipl");
    checkSci(NvSciBufAttrListCreate(bufModule, &nvbufAttrs), "NvSciBufAttrListCreate nvbuf");
    checkSci(setSiplBufferAttributes(siplAttrs), "NvSciBufAttrListSetAttrs");
    checkSipl(camera->GetImageAttributes(sensor.info.id, kIsp0, siplAttrs), "GetImageAttributes");
    checkSci(setNvBufAttributes(nvbufAttrs, siplAttrs, config.gpuId), "setNvBufAttributes");

    NvSciBufAttrList unreconciled[] = { siplAttrs, nvbufAttrs };
    checkSci(NvSciBufAttrListReconcile(unreconciled, 2U, &sensor.bufAttrs, &conflict), "NvSciBufAttrListReconcile");
    NvSciBufAttrListFree(siplAttrs);
    NvSciBufAttrListFree(nvbufAttrs);
    if (conflict)
        NvSciBufAttrListFree(conflict);

    sensor.slots.resize(config.isp0Buffers);
    for (uint32_t i = 0; i < config.isp0Buffers; ++i)
    {
        Slot& slot = sensor.slots[i];
        checkSci(NvSciBufObjAlloc(sensor.bufAttrs, &slot.buf), "NvSciBufObjAlloc");
        sensor.bufObjects.push_back(slot.buf);
        if (NvmmNvSciBufToNvBufSurface(slot.buf, &slot.surface) != 0 || slot.surface == nullptr)
            throw std::runtime_error("NvmmNvSciBufToNvBufSurface failed");
        slot.surface->numFilled = 1;
    }
}

void SiplCapture::Impl::allocateIcpBuffers(Sensor& sensor)
{
    NvSciBufAttrList attrs = nullptr;
    NvSciBufAttrList conflict = nullptr;
    checkSci(NvSciBufAttrListCreate(bufModule, &attrs), "NvSciBufAttrListCreate icp");
    // Only the type and permissions; GetImageAttributes fills in the raw format.
    NvSciBufType bufType = NvSciBufType_Image;
    NvSciBufAttrValAccessPerm perm = NvSciBufAccessPerm_ReadWrite;
    NvSciBufAttrKeyValuePair kvp[] = {
        { NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType) },
        { NvSciBufGeneralAttrKey_RequiredPerm, &perm, sizeof(perm) },
    };
    checkSci(NvSciBufAttrListSetAttrs(attrs, kvp, 2), "NvSciBufAttrListSetAttrs icp");
    checkSipl(camera->GetImageAttributes(sensor.info.id, kIcp, attrs), "GetImageAttributes ICP");

    NvSciBufAttrList unreconciled[] = { attrs };
    checkSci(NvSciBufAttrListReconcile(unreconciled, 1U, &sensor.icpAttrs, &conflict), "NvSciBufAttrListReconcile icp");
    NvSciBufAttrListFree(attrs);
    if (conflict)
        NvSciBufAttrListFree(conflict);

    for (uint32_t i = 0; i < kIcpBuffers; ++i)
    {
        NvSciBufObj obj = nullptr;
        checkSci(NvSciBufObjAlloc(sensor.icpAttrs, &obj), "NvSciBufObjAlloc icp");
        sensor.icpBufObjects.push_back(obj);
    }
}

void SiplCapture::Impl::registerBuffers(Sensor& sensor)
{
    checkSipl(camera->RegisterImages(sensor.info.id, kIcp, sensor.icpBufObjects), "RegisterImages ICP");
    checkSipl(camera->RegisterImages(sensor.info.id, kIsp0, sensor.bufObjects), "RegisterImages ISP0");
    checkSipl(camera->RegisterNvSciSyncObj(sensor.info.id, kIsp0, nvsipl::NVSIPL_EOFSYNCOBJ, sensor.syncObj),
              "RegisterNvSciSyncObj");
    registerNito(sensor);
}

void SiplCapture::Impl::allocateSync(Sensor& sensor)
{
    NvSciSyncAttrList signaler = nullptr;
    NvSciSyncAttrList waiter = nullptr;
    NvSciSyncAttrList reconciled = nullptr;
    NvSciSyncAttrList conflict = nullptr;
    checkSci(NvSciSyncAttrListCreate(syncModule, &signaler), "NvSciSyncAttrListCreate signaler");
    checkSci(NvSciSyncAttrListCreate(syncModule, &waiter), "NvSciSyncAttrListCreate waiter");

    // The end-of-frame fence is waited on by the CPU before the surface is handed on.
    NvSciSyncAccessPerm perm = NvSciSyncAccessPerm_WaitOnly;
    bool cpuWaiter = true;
    NvSciSyncAttrKeyValuePair kvp[] = {
        { NvSciSyncAttrKey_NeedCpuAccess, &cpuWaiter, sizeof(cpuWaiter) },
        { NvSciSyncAttrKey_RequiredPerm, &perm, sizeof(perm) },
    };
    checkSci(NvSciSyncAttrListSetAttrs(waiter, kvp, 2), "NvSciSyncAttrListSetAttrs");
    checkSipl(
        camera->FillNvSciSyncAttrList(sensor.info.id, kIsp0, signaler, nvsipl::SIPL_SIGNALER), "FillNvSciSyncAttrList");

    NvSciSyncAttrList unreconciled[] = { signaler, waiter };
    checkSci(NvSciSyncAttrListReconcile(unreconciled, 2U, &reconciled, &conflict), "NvSciSyncAttrListReconcile");
    checkSci(NvSciSyncObjAlloc(reconciled, &sensor.syncObj), "NvSciSyncObjAlloc");
    NvSciSyncAttrListFree(signaler);
    NvSciSyncAttrListFree(waiter);
    if (reconciled)
        NvSciSyncAttrListFree(reconciled);
    if (conflict)
        NvSciSyncAttrListFree(conflict);
}

void SiplCapture::Impl::registerNito(Sensor& sensor)
{
    std::vector<uint8_t> blob = loadNito(config.nitoDir, sensor.info.name);
    nvsipl::ISiplControlAuto* autoControl = nullptr;
    checkSipl(camera->RegisterAutoControlPlugin(sensor.info.id, nvsipl::NV_PLUGIN, autoControl, blob),
              "RegisterAutoControlPlugin");
}

void SiplCapture::Impl::dropPending(Sensor& sensor)
{
    std::lock_guard<std::mutex> guard(sensor.mutex);
    if (sensor.hasPending)
    {
        sensor.pending.buffer->Release();
        sensor.hasPending = false;
    }
}

void SiplCapture::Impl::frameLoop(Sensor& sensor)
{
    try
    {
        while (running.load())
        {
            INvSIPLClient::INvSIPLBuffer* buffer = nullptr;
            const SIPLStatus status = sensor.queues.isp0CompletionQueue->Get(buffer, config.frameTimeoutMs * 1000U);
            if (status == nvsipl::NVSIPL_STATUS_TIMED_OUT)
                continue;
            if (status == nvsipl::NVSIPL_STATUS_EOF)
                break;
            checkSipl(status, "isp0CompletionQueue->Get");
            if (!buffer)
                continue;

            // The buffer belongs to SIPL until it is handed to `pending`; any failure before that returns it.
            ScopedRelease<INvSIPLClient::INvSIPLBuffer> owned(buffer);
            INvSIPLClient::INvSIPLNvMBuffer* nvm = dynamic_cast<INvSIPLClient::INvSIPLNvMBuffer*>(buffer);
            if (!nvm)
                throw std::runtime_error("ISP0 buffer is not an INvSIPLNvMBuffer");

            // The ISP is still writing until this fence clears.
            NvSciSyncFence fence = NvSciSyncFenceInitializer;
            checkSipl(nvm->GetEOFNvSciSyncFence(&fence), "GetEOFNvSciSyncFence");
            const NvSciError wait = NvSciSyncFenceWait(&fence, sensor.cpuWaitContext, config.frameTimeoutMs * 1000UL);
            NvSciSyncFenceClear(&fence);
            checkSci(wait, "NvSciSyncFenceWait");

            const NvSciBufObj arrived = nvm->GetNvSciBufImage();
            const Slot* slot = nullptr;
            for (const Slot& candidate : sensor.slots)
            {
                if (candidate.buf == arrived)
                {
                    slot = &candidate;
                    break;
                }
            }
            if (!slot)
                throw std::runtime_error("ISP0 delivered a buffer this process never registered");

            Pending next;
            next.buffer = buffer;
            next.slot = slot;
            // The capture counter in the SIPL clock domain, unscaled: the start-of-frame value, not end-of-frame.
            next.readyMonotonicNs = monotonicNs();
            // The start-of-frame value counts the platform counter's ticks (measured against CNTVCT_EL0 on the board);
            // it is only reported as a TSC in nanoseconds while the mapping check passes.
            if (sensor.clockMap)
            {
                const int64_t ready = static_cast<int64_t>(next.readyMonotonicNs);
                const SensorTimeMapping mapping =
                    sensor.clockMap->map(nvm->GetImageData().frameCaptureStartTSC, ready, ready);
                if (mapping.valid)
                {
                    next.sensorTscNs = static_cast<uint64_t>(
                        counterTicksToNs(static_cast<int64_t>(nvm->GetImageData().frameCaptureStartTSC), counterHz));
                    next.captureMapped = true;
                    next.captureMonotonicNs = mapping.captureMonotonicNs;
                    next.captureUncertaintyNs = mapping.uncertaintyNs;
                    next.captureAnchorAgeNs = mapping.anchorAgeNs;
                    next.captureRealtimeNs = mapping.hasRealtime ? mapping.captureRealtimeNs : 0;
                    sensor.lastMappingReason.clear();
                }
                else if (mapping.reason != sensor.lastMappingReason)
                {
                    // Reported when the reason changes, not per frame: the frames still flow without a sensor time.
                    std::fprintf(
                        stderr, "[sipl] sensor %u: no capture time: %s\n", sensor.info.id, mapping.reason.c_str());
                    sensor.lastMappingReason = mapping.reason;
                }
            }

            {
                std::lock_guard<std::mutex> guard(sensor.mutex);
                if (sensor.hasPending)
                    sensor.pending.buffer->Release();
                next.sequence = ++sensor.received;
                sensor.pending = next;
                sensor.hasPending = true;
                // `pending` owns the buffer now, and its release path (a replacement or stop) takes over.
                owned.dismiss();
            }
            std::lock_guard<std::mutex> wakeGuard(wakeMutex);
            wake.notify_all();
        }
    }
    catch (const std::exception& e)
    {
        std::ostringstream oss;
        oss << "SIPL sensor " << sensor.info.id << " capture error: " << e.what();
        std::fprintf(stderr, "[sipl] %s\n", oss.str().c_str());
        setFailure(oss.str());
        running.store(false);
    }
}

void SiplCapture::Impl::eventLoop(Sensor& sensor)
{
    // A GMSL link drop arrives here and nowhere else; without this drain the symptom is a silent stall.
    while (running.load())
    {
        nvsipl::NvSIPLPipelineNotifier::NotificationData data{};
        if (sensor.queues.notificationQueue->Get(data, config.frameTimeoutMs * 1000U) != NVSIPL_STATUS_OK)
            continue;
        using Notification = nvsipl::NvSIPLPipelineNotifier;
        switch (data.eNotifType)
        {
        case Notification::NOTIF_ERROR_DESERIALIZER_FAILURE:
        case Notification::NOTIF_ERROR_SERIALIZER_FAILURE:
        case Notification::NOTIF_ERROR_SENSOR_FAILURE:
        case Notification::NOTIF_ERROR_INTERNAL_FAILURE:
        case Notification::NOTIF_ERROR_ICP_CAPTURE_FAILURE:
        case Notification::NOTIF_ERROR_ISP_PROCESSING_FAILURE:
        {
            std::ostringstream oss;
            oss << "SIPL sensor " << sensor.info.id << " pipeline error, notification "
                << static_cast<int>(data.eNotifType);
            std::fprintf(stderr, "[sipl] %s\n", oss.str().c_str());
            setFailure(oss.str());
            running.store(false);
            break;
        }
        default:
            break;
        }
    }
}

void SiplCapture::Impl::joinThreads()
{
    running.store(false);
    for (auto& sensor : sensors)
    {
        if (sensor->frameThread.joinable())
            sensor->frameThread.join();
        if (sensor->eventThread.joinable())
            sensor->eventThread.join();
    }
}

void SiplCapture::Impl::cleanup()
{
    joinThreads();
    for (auto& sensor : sensors)
        dropPending(*sensor);
    if (camera)
    {
        camera->Stop();
        camera->Deinit();
    }
    for (auto& sensor : sensors)
    {
        for (Slot& slot : sensor->slots)
        {
            if (slot.surface)
                NvBufSurfaceDestroy(slot.surface);
            if (slot.buf)
                NvSciBufObjFree(slot.buf);
        }
        sensor->slots.clear();
        sensor->bufObjects.clear();
        for (NvSciBufObj obj : sensor->icpBufObjects)
            NvSciBufObjFree(obj);
        sensor->icpBufObjects.clear();
        if (sensor->icpAttrs)
            NvSciBufAttrListFree(sensor->icpAttrs);
        if (sensor->bufAttrs)
            NvSciBufAttrListFree(sensor->bufAttrs);
        if (sensor->syncObj)
            NvSciSyncObjFree(sensor->syncObj);
        if (sensor->cpuWaitContext)
            NvSciSyncCpuWaitContextFree(sensor->cpuWaitContext);
        sensor->cpuWaitContext = nullptr;
        sensor->icpAttrs = nullptr;
        sensor->bufAttrs = nullptr;
        sensor->syncObj = nullptr;
    }
    camera.reset();
    queryApi.reset();
    if (syncModule)
        NvSciSyncModuleClose(syncModule);
    if (bufModule)
        NvSciBufModuleClose(bufModule);
    syncModule = nullptr;
    bufModule = nullptr;
}

SiplCapture::SiplCapture(const SiplConfig& config) : m_impl(new Impl(config))
{
    if (config.platformConfigJson.empty() || config.platformConfigName.empty())
        throw std::invalid_argument("SiplConfig needs both platformConfigJson and platformConfigName");
    try
    {
        m_impl->configure();
    }
    catch (...)
    {
        m_impl->cleanup();
        throw;
    }
}

SiplCapture::~SiplCapture()
{
    m_impl->cleanup();
}

const std::vector<CameraDescriptor>& SiplCapture::cameras() const
{
    return m_impl->descriptors;
}

void SiplCapture::start()
{
    if (m_impl->running.load())
        return;
    m_impl->running.store(true);
    for (auto& sensor : m_impl->sensors)
    {
        if (sensor->clockMap)
            sensor->clockMap->reset();
        Impl::Sensor* raw = sensor.get();
        raw->frameThread = std::thread([this, raw] { m_impl->frameLoop(*raw); });
        raw->eventThread = std::thread([this, raw] { m_impl->eventLoop(*raw); });
    }
    try
    {
        checkSipl(m_impl->camera->Start(), "INvSIPLCamera::Start");
    }
    catch (...)
    {
        m_impl->joinThreads();
        throw;
    }
}

void SiplCapture::stop()
{
    m_impl->joinThreads();
    for (auto& sensor : m_impl->sensors)
        m_impl->dropPending(*sensor);
    if (m_impl->camera)
        m_impl->camera->Stop();
}

bool SiplCapture::waitForFrame(uint32_t timeoutMs)
{
    const auto anyPending = [this]()
    {
        if (m_impl->failed.load())
            return true;
        for (auto& sensor : m_impl->sensors)
        {
            std::lock_guard<std::mutex> guard(sensor->mutex);
            if (sensor->hasPending)
                return true;
        }
        return false;
    };
    std::unique_lock<std::mutex> lock(m_impl->wakeMutex);
    m_impl->wake.wait_for(lock, std::chrono::milliseconds(timeoutMs), anyPending);
    lock.unlock();
    m_impl->throwIfFailed();
    return anyPending();
}

bool SiplCapture::acquireLatest(uint32_t cameraId, CaptureFrame* frame)
{
    m_impl->throwIfFailed();
    Impl::Sensor* sensor = m_impl->find(cameraId);
    if (!sensor)
        throw std::runtime_error("no SIPL pipeline with index " + std::to_string(cameraId));

    Impl::Pending taken;
    {
        std::lock_guard<std::mutex> guard(sensor->mutex);
        if (!sensor->hasPending)
            return false;
        taken = sensor->pending;
        sensor->hasPending = false;
    }

    const NvBufSurfaceParams& params = taken.slot->surface->surfaceList[0];
    CaptureFrame out;
    out.cameraId = cameraId;
    out.sessionEpoch = m_impl->config.sessionEpoch;
    out.sequence = taken.sequence;
    out.width = params.width;
    out.height = params.height;
    out.sensorTime = taken.sensorTscNs;
    out.sensorClock = taken.captureMapped ? SENSOR_CLOCK_TSC_NS : SENSOR_CLOCK_NONE;
    out.captureMapped = taken.captureMapped;
    out.captureMonotonicNs = taken.captureMonotonicNs;
    out.captureUncertaintyNs = taken.captureUncertaintyNs;
    out.captureAnchorAgeNs = taken.captureAnchorAgeNs;
    out.captureRealtimeNs = taken.captureRealtimeNs;
    out.readyMonotonicNs = taken.readyMonotonicNs;
    // bufferDesc is the surface's DMA-BUF descriptor; whether the encoder accepts it for this NvSci-backed buffer is
    // not established off the board.
    out.surface.fd = static_cast<int>(params.bufferDesc);
    for (int plane = 0; plane < 2; ++plane)
    {
        out.surface.pitchBytes[plane] = params.planeParams.pitch[plane];
        out.surface.offsetBytes[plane] = params.planeParams.offset[plane];
    }
    out.surface.blockLinear = params.layout == NVBUF_LAYOUT_BLOCK_LINEAR;
    out.surface.colorRange = COLOR_RANGE_FULL;
    INvSIPLClient::INvSIPLBuffer* buffer = taken.buffer;
    out.lease = FrameLease([buffer]() { buffer->Release(); });
    *frame = std::move(out);
    return true;
}

} // namespace argus_sender
