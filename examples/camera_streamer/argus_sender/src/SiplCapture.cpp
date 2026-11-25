/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

// SIPL capture. A raw sensor's ISP0 surface is handed on as native NV12 and returned to SIPL when
// the consumer drops its lease; there is no RGBA conversion. Needs C++17 because the SIPL headers do.

#include "SiplCapture.h"

#include "ScopedRelease.h"
#include "SensorClock.h"

#include <NvSIPLCamera.hpp>
#include <NvSIPLCameraQuery.hpp>
#include <NvSIPLCapStructs.h>
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
#include <nvbufsurftransform.h>
#include <nvscibuf.h>
#include <nvscisync.h>
#include <set>
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

// Calls fn(sensor) for every sensor of the configuration; the nesting is the one `flatten` walks.
template <typename Config, typename Fn>
void forEachSensor(Config& config, Fn fn)
{
    for (auto& module : config.modules)
    {
        std::visit(
            [&](auto& moduleType)
            {
                for (auto& sensorVariant : moduleType.sensorConfigs)
                    std::visit(fn, sensorVariant);
            },
            module.moduleType);
    }
}

// Sensors that share a sensor group (a stereo pair) run as one SIPL pipeline group: the sensor with device index 0
// is the primary, and the others take its ISP program and auto control. Returns the ids of those others.
std::set<uint32_t> groupSecondaries(const nvsipl::sensorconfig::SensorSystemConfig& config)
{
    std::set<uint32_t> out;
    forEachSensor(config,
                  [&](const auto& sensor)
                  {
                      if (sensor.sensorGroup > 0 && sensor.deviceIndex > 0)
                          out.insert(sensor.id);
                  });
    return out;
}

// A virtual channel with no Bayer order (cfa 0, which no NVSIPL_PIXEL_ORDER_* value is) belongs to a YUV camera with
// its own ISP. The query library leaves such a channel at its RAW12 default, which SIPL rejects at
// GetImageAttributes, so it is declared YUV422/UYVY before SetPlatformCfg. Returns the ids of the sensors corrected.
std::set<uint32_t> declareYuv422Sensors(nvsipl::sensorconfig::SensorSystemConfig& config)
{
    std::set<uint32_t> out;
    forEachSensor(config,
                  [&](auto& sensor)
                  {
                      for (auto& vc : sensor.vcInfoList)
                      {
                          if (vc.cfa != 0)
                              continue;
                          vc.inputFormat = NVSIPL_CAP_INPUT_FORMAT_TYPE_YUV422;
                          vc.cfa = NVSIPL_PIXEL_ORDER_UYVY;
                          out.insert(sensor.id);
                      }
                  });
    return out;
}

// One value of a reconciled buffer attribute list.
template <typename T>
T attrValue(const NvSciBufAttrKeyValuePair& attr)
{
    if (attr.value == nullptr)
        throw std::runtime_error("the capture buffer attributes lack a plane geometry key");
    return *static_cast<const T*>(attr.value);
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

    // An NV12 picture converted from a YUV camera's capture output. Leased to the consumer like a SIPL buffer.
    struct Nv12Out
    {
        NvBufSurface* surface = nullptr;
        bool inUse = false;
    };

    // The newest frame the consumer has not taken. A frame replaced before it is taken goes straight back to its
    // owner: SIPL for an ISP0 buffer, the sensor's NV12 pool for a converted picture (`nv12`; `buffer` is then null).
    struct Pending
    {
        INvSIPLClient::INvSIPLBuffer* buffer = nullptr;
        Nv12Out* nv12 = nullptr;
        const NvBufSurface* surface = nullptr;
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
        // In a sensor group without being its primary; see groupSecondaries.
        bool groupSecondary = false;
        // A YUV camera: pictures come from the capture output (ICP) as UYVY and are converted to NV12 by the VIC;
        // the ISP is not used and there is no NITO. See declareYuv422Sensors.
        bool yuvPassthrough = false;
        nvsipl::NvSIPLPipelineConfiguration pipelineConfig{};
        nvsipl::NvSIPLPipelineQueues queues{};
        NvSciBufAttrList bufAttrs = nullptr;
        std::vector<NvSciBufObj> bufObjects;
        // Never read; allocated because SIPL requires it.
        NvSciBufAttrList icpAttrs = nullptr;
        std::vector<NvSciBufObj> icpBufObjects;
        std::vector<Slot> slots;
        // YUV cameras only: the NV12 pictures handed to the consumer.
        std::vector<Nv12Out> nv12Pool;
        bool nv12Starved = false;
        NvSciSyncObj syncObj = nullptr;
        // One wait context can wait on only one fence at a time, so each sensor's frame thread has its own.
        NvSciSyncCpuWaitContext cpuWaitContext = nullptr;

        // Places the sensor's capture timestamp on CLOCK_MONOTONIC; used on this sensor's frame thread only.
        std::unique_ptr<SensorTimeMapper> clockMap;
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

    // sensor.mutex held by the caller.
    static void releasePending(Pending& pending)
    {
        if (pending.buffer)
            pending.buffer->Release();
        if (pending.nv12)
            pending.nv12->inUse = false;
        pending.buffer = nullptr;
        pending.nv12 = nullptr;
    }

    void configure();
    void allocateBuffers(Sensor& sensor);
    void allocateIcpBuffers(Sensor& sensor);
    void allocateYuvBuffers(Sensor& sensor);
    void registerBuffers(Sensor& sensor);
    void allocateSync(Sensor& sensor, INvSIPLClient::ConsumerDesc::OutputType output);
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

    const std::set<uint32_t> yuvSensors = declareYuv422Sensors(*systemConfig);
    for (uint32_t id : yuvSensors)
        std::fprintf(stderr, "[sipl] sensor %u: YUV422/UYVY camera, capture output converted to NV12 by the VIC\n", id);

    descriptors = flatten(*systemConfig);
    if (descriptors.empty())
        throw std::runtime_error("platform configuration '" + config.platformConfigName + "' declares no sensors");

    camera = INvSIPLCamera::GetInstance();
    if (!camera)
        throw std::runtime_error("INvSIPLCamera::GetInstance returned null");
    checkSipl(camera->SetPlatformCfg(*systemConfig), "SetPlatformCfg");

    const std::set<uint32_t> secondaries = groupSecondaries(*systemConfig);
    for (const auto& info : descriptors)
    {
        std::unique_ptr<Sensor> sensor(new Sensor());
        sensor->info = info;
        sensor->groupSecondary = secondaries.count(info.id) != 0;
        sensor->yuvPassthrough = yuvSensors.count(info.id) != 0;
        checkSci(NvSciSyncCpuWaitContextAlloc(syncModule, &sensor->cpuWaitContext), "NvSciSyncCpuWaitContextAlloc");
        SensorClockMap::Config clockConfig;
        sensor->clockMap.reset(new SensorTimeMapper(counterHz, clockConfig, readCounterPair));
        // ISP0 only for a raw sensor: ICP would double the capture bandwidth for a raw stream nothing here consumes.
        // A YUV camera is the opposite: its capture output is the picture and the ISP cannot take it.
        sensor->pipelineConfig.captureOutputRequested = sensor->yuvPassthrough;
        sensor->pipelineConfig.isp0OutputRequested = !sensor->yuvPassthrough;
        sensor->pipelineConfig.isp1OutputRequested = false;
        sensor->pipelineConfig.isp2OutputRequested = false;
        // Subframe is enabled by default and then needs a slice count that is not set here.
        sensor->pipelineConfig.disableSubframe = true;
        sensor->pipelineConfig.bufferCfg.maxIsp0BufferCount = config.isp0Buffers;
        if (sensor->yuvPassthrough)
            sensor->pipelineConfig.bufferCfg.maxCaptureBufferCount = config.isp0Buffers;
        checkSipl(camera->SetPipelineCfg(info.id, sensor->pipelineConfig, sensor->queues), "SetPipelineCfg");
        sensors.push_back(std::move(sensor));
    }

    // Order is fixed by SIPL: what describes a pipeline (GetImageAttributes, FillNvSciSyncAttrList) comes before
    // Init(), what binds a resource to it (RegisterImages, RegisterNvSciSyncObj) after. Init() still returns OK
    // out of order, and the first Register call then fails with NVSIPL_STATUS_INVALID_STATE.
    for (auto& sensor : sensors)
    {
        if (sensor->yuvPassthrough)
        {
            allocateYuvBuffers(*sensor);
            allocateSync(*sensor, kIcp);
            continue;
        }
        allocateIcpBuffers(*sensor);
        allocateBuffers(*sensor);
        allocateSync(*sensor, kIsp0);
    }
    checkSipl(camera->Init(), "INvSIPLCamera::Init");
    for (auto& sensor : sensors)
        registerBuffers(*sensor);
    // Auto control comes after every sensor's images: a group primary is refused with NVSIPL_STATUS_INVALID_STATE
    // while its secondary has none registered.
    for (auto& sensor : sensors)
    {
        if (!sensor->groupSecondary && !sensor->yuvPassthrough)
            registerNito(*sensor);
    }
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

// A YUV camera's capture output: UYVY buffers SIPL fills, each with an NvBufSurface view the VIC reads, and the NV12
// pictures the VIC writes for the encoder. NvmmImageFillNvSciBufAttrList knows no UYVY, so the buffers are allocated
// from SIPL's attributes alone and each view is imported from the DMA-BUF and the reconciled geometry.
void SiplCapture::Impl::allocateYuvBuffers(Sensor& sensor)
{
    NvSciBufAttrList siplAttrs = nullptr;
    NvSciBufAttrList conflict = nullptr;
    checkSci(NvSciBufAttrListCreate(bufModule, &siplAttrs), "NvSciBufAttrListCreate icp");
    NvSciBufType bufType = NvSciBufType_Image;
    NvSciBufAttrValAccessPerm perm = NvSciBufAccessPerm_ReadWrite;
    NvSciBufAttrKeyValuePair kvp[] = {
        { NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType) },
        { NvSciBufGeneralAttrKey_RequiredPerm, &perm, sizeof(perm) },
    };
    checkSci(NvSciBufAttrListSetAttrs(siplAttrs, kvp, 2), "NvSciBufAttrListSetAttrs icp");
    checkSipl(camera->GetImageAttributes(sensor.info.id, kIcp, siplAttrs), "GetImageAttributes ICP");
    NvSciBufAttrList unreconciled[] = { siplAttrs };
    checkSci(NvSciBufAttrListReconcile(unreconciled, 1U, &sensor.bufAttrs, &conflict), "NvSciBufAttrListReconcile icp");
    NvSciBufAttrListFree(siplAttrs);
    if (conflict)
        NvSciBufAttrListFree(conflict);

    NvSciBufAttrKeyValuePair geometry[] = {
        { NvSciBufImageAttrKey_PlaneCount, nullptr, 0 },  { NvSciBufImageAttrKey_PlaneWidth, nullptr, 0 },
        { NvSciBufImageAttrKey_PlaneHeight, nullptr, 0 }, { NvSciBufImageAttrKey_PlanePitch, nullptr, 0 },
        { NvSciBufImageAttrKey_PlaneOffset, nullptr, 0 }, { NvSciBufImageAttrKey_PlaneAlignedSize, nullptr, 0 },
        { NvSciBufImageAttrKey_Size, nullptr, 0 },        { NvSciBufImageAttrKey_Layout, nullptr, 0 },
    };
    checkSci(NvSciBufAttrListGetAttrs(sensor.bufAttrs, geometry, sizeof(geometry) / sizeof(geometry[0])),
             "NvSciBufAttrListGetAttrs icp geometry");
    if (attrValue<uint32_t>(geometry[0]) != 1)
        throw std::runtime_error("expected one UYVY plane from the capture output");
    NvBufSurfaceMapParams map{};
    map.num_planes = 1;
    map.gpuId = static_cast<uint32_t>(config.gpuId);
    map.memType = NVBUF_MEM_SURFACE_ARRAY;
    map.scanformat = NVBUF_DISPLAYSCANFORMAT_PROGRESSIVE;
    map.colorFormat = NVBUF_COLOR_FORMAT_UYVY;
    map.planes[0].width = attrValue<uint32_t>(geometry[1]);
    map.planes[0].height = attrValue<uint32_t>(geometry[2]);
    map.planes[0].pitch = attrValue<uint32_t>(geometry[3]);
    map.planes[0].offset = static_cast<uint32_t>(attrValue<uint64_t>(geometry[4]));
    map.planes[0].psize = static_cast<uint32_t>(attrValue<uint64_t>(geometry[5]));
    map.totalSize = static_cast<uint32_t>(attrValue<uint64_t>(geometry[6]));
    map.layout = attrValue<NvSciBufAttrValImageLayoutType>(geometry[7]) == NvSciBufImage_PitchLinearType ?
                     NVBUF_LAYOUT_PITCH :
                     NVBUF_LAYOUT_BLOCK_LINEAR;

    sensor.slots.resize(config.isp0Buffers);
    for (uint32_t i = 0; i < config.isp0Buffers; ++i)
    {
        Slot& slot = sensor.slots[i];
        checkSci(NvSciBufObjAlloc(sensor.bufAttrs, &slot.buf), "NvSciBufObjAlloc icp");
        sensor.bufObjects.push_back(slot.buf);
        int fd = -1;
        checkSci(NvSciBufObjGetDmaBufFd(slot.buf, &fd), "NvSciBufObjGetDmaBufFd");
        map.fd = static_cast<uint64_t>(fd);
        if (NvBufSurfaceImport(&slot.surface, &map) != 0 || slot.surface == nullptr)
            throw std::runtime_error("NvBufSurfaceImport failed for a UYVY capture buffer");
        slot.surface->numFilled = 1;
    }

    // Block-linear NV12 like the ISP0 path. The _709_ER tag asks the VIC to expand the camera's limited range, so
    // every stream of a SIPL profile is full range as the consumers are told.
    sensor.nv12Pool.resize(config.isp0Buffers);
    for (Nv12Out& out : sensor.nv12Pool)
    {
        NvBufSurfaceAllocateParams alloc{};
        alloc.params.width = sensor.info.width;
        alloc.params.height = sensor.info.height;
        alloc.params.layout = NVBUF_LAYOUT_BLOCK_LINEAR;
        alloc.params.colorFormat = NVBUF_COLOR_FORMAT_NV12_709_ER;
        alloc.params.memType = NVBUF_MEM_SURFACE_ARRAY;
        alloc.memtag = NvBufSurfaceTag_CAMERA;
        if (NvBufSurfaceAllocate(&out.surface, 1, &alloc) != 0 || out.surface == nullptr)
            throw std::runtime_error("NvBufSurfaceAllocate NV12 failed");
        out.surface->numFilled = 1;
    }
}

void SiplCapture::Impl::registerBuffers(Sensor& sensor)
{
    if (sensor.yuvPassthrough)
    {
        checkSipl(camera->RegisterImages(sensor.info.id, kIcp, sensor.bufObjects), "RegisterImages ICP");
        checkSipl(camera->RegisterNvSciSyncObj(sensor.info.id, kIcp, nvsipl::NVSIPL_EOFSYNCOBJ, sensor.syncObj),
                  "RegisterNvSciSyncObj ICP");
        return;
    }
    checkSipl(camera->RegisterImages(sensor.info.id, kIcp, sensor.icpBufObjects), "RegisterImages ICP");
    checkSipl(camera->RegisterImages(sensor.info.id, kIsp0, sensor.bufObjects), "RegisterImages ISP0");
    checkSipl(camera->RegisterNvSciSyncObj(sensor.info.id, kIsp0, nvsipl::NVSIPL_EOFSYNCOBJ, sensor.syncObj),
              "RegisterNvSciSyncObj");
}

void SiplCapture::Impl::allocateSync(Sensor& sensor, INvSIPLClient::ConsumerDesc::OutputType output)
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
    checkSipl(camera->FillNvSciSyncAttrList(sensor.info.id, output, signaler, nvsipl::SIPL_SIGNALER),
              "FillNvSciSyncAttrList");

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
        releasePending(sensor.pending);
        sensor.hasPending = false;
    }
}

void SiplCapture::Impl::frameLoop(Sensor& sensor)
{
    try
    {
        nvsipl::INvSIPLFrameCompletionQueue* queue =
            sensor.yuvPassthrough ? sensor.queues.captureCompletionQueue : sensor.queues.isp0CompletionQueue;
        if (sensor.yuvPassthrough)
        {
            // Transform session parameters are per thread.
            NvBufSurfTransformConfigParams session{};
            session.compute_mode = NvBufSurfTransformCompute_VIC;
            if (NvBufSurfTransformSetSessionParams(&session) != NvBufSurfTransformError_Success)
                throw std::runtime_error("NvBufSurfTransformSetSessionParams failed");
        }
        while (running.load())
        {
            INvSIPLClient::INvSIPLBuffer* buffer = nullptr;
            const SIPLStatus status = queue->Get(buffer, config.frameTimeoutMs * 1000U);
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
                throw std::runtime_error("SIPL delivered a buffer this process never registered");

            Pending next;
            next.buffer = buffer;
            next.surface = slot->surface;
            if (sensor.yuvPassthrough)
            {
                Nv12Out* out = nullptr;
                {
                    std::lock_guard<std::mutex> guard(sensor.mutex);
                    for (Nv12Out& candidate : sensor.nv12Pool)
                    {
                        if (!candidate.inUse)
                        {
                            candidate.inUse = true;
                            out = &candidate;
                            break;
                        }
                    }
                }
                // Every NV12 picture pending or with the encoder: this capture goes back to SIPL unconverted.
                if (!out && !sensor.nv12Starved)
                    std::fprintf(stderr, "[sipl] sensor %u: no free NV12 picture, dropping\n", sensor.info.id);
                sensor.nv12Starved = !out;
                if (!out)
                    continue;
                // Same size, whole picture: a format conversion only. The call returns when the VIC has written.
                NvBufSurfTransformParams transform{};
                if (NvBufSurfTransform(slot->surface, out->surface, &transform) != NvBufSurfTransformError_Success)
                {
                    std::lock_guard<std::mutex> guard(sensor.mutex);
                    out->inUse = false;
                    throw std::runtime_error("NvBufSurfTransform UYVY to NV12 failed");
                }
                // The consumer gets the NV12 picture; `owned` returns the capture buffer to SIPL.
                next.buffer = nullptr;
                next.nv12 = out;
                next.surface = out->surface;
            }
            // The capture counter in the SIPL clock domain, unscaled: the start-of-frame value, not end-of-frame.
            next.readyMonotonicNs = monotonicNs();
            // The start-of-frame value counts the platform counter's ticks (measured against CNTVCT_EL0 on the board);
            // it is only reported as a TSC in nanoseconds while the mapping check passes.
            if (sensor.clockMap)
            {
                const int64_t ready = static_cast<int64_t>(next.readyMonotonicNs);
                const SensorTimeMapping mapping = sensor.clockMap->map(
                    nvm->GetImageData().frameCaptureStartTSC, SensorTimestampUnit::CounterTicks, ready, ready);
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
                    releasePending(sensor.pending);
                next.sequence = ++sensor.received;
                sensor.pending = next;
                sensor.hasPending = true;
                // `pending` owns the SIPL buffer now, and its release path (a replacement or stop) takes over.
                if (next.buffer)
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
        for (Nv12Out& out : sensor->nv12Pool)
        {
            if (out.surface)
                NvBufSurfaceDestroy(out.surface);
        }
        sensor->nv12Pool.clear();
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

    const NvBufSurfaceParams& params = taken.surface->surfaceList[0];
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
    // ISP0 writes BT.709 extended range; a YUV camera's picture is converted to the same by the VIC.
    out.surface.colorRange = COLOR_RANGE_FULL;
    Impl::Pending held = taken;
    out.lease = FrameLease(
        [sensor, held]() mutable
        {
            std::lock_guard<std::mutex> guard(sensor->mutex);
            Impl::releasePending(held);
        });
    *frame = std::move(out);
    return true;
}

} // namespace argus_sender
