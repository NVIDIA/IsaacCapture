// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Portions adapted from NVIDIA's sipl_coe_unit_sample
// (/usr/src/jetson_multimedia_api/samples/unittest_samples/), BSD-3-Clause,
// Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.

#include "sipl_camera.hpp"

#include "yuv_to_rgba.cuh"

#include <NvSIPLCamera.hpp>
#include <NvSIPLCameraQuery.hpp>
#include <NvSIPLClient.hpp>
#include <NvSIPLCommon.hpp>
#include <NvSIPLPipelineMgr.hpp>
#include <NvSIPLTrace.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <cudaEGL.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <mutex>
#include <nvbufsurface.h>
#include <nvbufsurface_nvscibuf.h>
#include <nvscibuf.h>
#include <nvscisync.h>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace plugins
{
namespace sensing
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

/// SIPL allocates and registers ICP capture buffers unconditionally, even when
/// only an ISP output is consumed -- CNvSIPLMaster::AllocateBuffers hardcodes
/// the ICP slot to true. Skip them and RegisterImages fails with a bare
/// NVSIPL_STATUS_ERROR that names nothing.
constexpr uint32_t kIcpBuffers = 6;

uint64_t monotonic_ns()
{
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

/// SIPL reports almost every failure as NVSIPL_STATUS_ERROR (10), which says
/// nothing on its own -- the reason only appears in its own log, above a
/// threshold that defaults to silent. SENSING_SIPL_TRACE=1..4 raises it
/// (1=error, 2=warn, 3=info, 4=debug).
void init_sipl_trace()
{
    const char* level = std::getenv("SENSING_SIPL_TRACE");
    if (level == nullptr)
    {
        return;
    }
    const int value = std::atoi(level);
    if (value <= 0)
    {
        return;
    }
    auto* trace = nvsipl::INvSIPLTrace::GetInstance();
    if (trace != nullptr)
    {
        trace->SetLevel(static_cast<nvsipl::INvSIPLTrace::TraceLevel>(value));
    }
}

void check_sipl(SIPLStatus status, const char* what)
{
    if (status != NVSIPL_STATUS_OK)
    {
        std::ostringstream oss;
        oss << what << " failed with SIPLStatus " << static_cast<int>(status);
        throw std::runtime_error(oss.str());
    }
}

void check_sci(NvSciError err, const char* what)
{
    if (err != NvSciError_Success)
    {
        std::ostringstream oss;
        oss << what << " failed with NvSciError " << static_cast<int>(err);
        throw std::runtime_error(oss.str());
    }
}

void check_cuda(CUresult result, const char* what)
{
    if (result == CUDA_SUCCESS)
    {
        return;
    }
    const char* name = nullptr;
    const char* text = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &text);
    std::ostringstream oss;
    oss << what << " failed";
    if (name)
    {
        oss << ": " << name;
    }
    if (text)
    {
        oss << " (" << text << ")";
    }
    throw std::runtime_error(oss.str());
}

cudaTextureObject_t texture_for_array(CUarray array);

void check_runtime(cudaError_t result, const char* what)
{
    if (result != cudaSuccess)
    {
        std::ostringstream oss;
        oss << what << " failed: " << cudaGetErrorString(result);
        throw std::runtime_error(oss.str());
    }
}

cudaTextureObject_t texture_for_array(CUarray array)
{
    cudaResourceDesc resource_desc{};
    resource_desc.resType = cudaResourceTypeArray;
    resource_desc.res.array.array = reinterpret_cast<cudaArray_t>(array);

    cudaTextureDesc texture_desc{};
    texture_desc.addressMode[0] = cudaAddressModeClamp;
    texture_desc.addressMode[1] = cudaAddressModeClamp;
    texture_desc.filterMode = cudaFilterModePoint;
    texture_desc.readMode = cudaReadModeElementType;
    texture_desc.normalizedCoords = 0;

    cudaTextureObject_t texture = 0;
    check_runtime(cudaCreateTextureObject(&texture, &resource_desc, &texture_desc, nullptr), "cudaCreateTextureObject");
    return texture;
}

/**
 * @brief Buffer attributes for the ISP0 output.
 *
 * Block-linear and REC709_ER are what the ISP produces, not what we would
 * prefer. Neither is negotiable: GetImageAttributes rejects an attribute list
 * asking for pitch-linear or BT.601 with NVSIPL_STATUS_BAD_ARGUMENT. So the
 * conversion adapts to the ISP rather than the other way round -- convert()
 * takes the CUDA-array path, and yuv_to_rgba.cu is called with bt709=true.
 * `nvsipl_camera -v 3` prints the same thing: "YUV 420 SEMI-PLANAR UINT8 BL
 * REC_709ER".
 */
NvSciError set_sipl_buf_attributes(NvSciBufAttrList attr_list)
{
    NvSciBufType buf_type = NvSciBufType_Image;
    NvSciBufAttrValAccessPerm access_perm = NvSciBufAccessPerm_ReadWrite;
    // Must be true, and it is not about us wanting CPU access. NvmmImageFill-
    // NvSciBufAttrList sets NeedCpuAccess=true on the NvBufSurface list, so
    // reconciliation ORs it in; if the producer list says false, SIPL's
    // GetVerifiedBufferAttributes sees a producer/reconciled mismatch and
    // RegisterImages fails with a bare NVSIPL_STATUS_ERROR.
    bool cpu_access = true;
    bool cpu_cache = true;
    NvSciBufSurfMemLayout mem_layout = NvSciSurfMemLayout_SemiPlanar;
    NvSciBufAttrValImageLayoutType layout_type = NvSciBufImage_BlockLinearType;
    NvSciBufSurfType surf_type = NvSciSurfType_YUV;
    NvSciBufSurfSampleType sample_type = NvSciSurfSampleType_420;
    NvSciBufSurfBPC surf_bpc = NvSciSurfBPC_8;
    NvSciBufSurfComponentOrder comp_order = NvSciSurfComponentOrder_YUV;
    NvSciBufAttrValColorStd color_std[] = { NvSciColorStd_REC709_ER };

    NvSciBufAttrKeyValuePair kvp[] = {
        { NvSciBufGeneralAttrKey_Types, &buf_type, sizeof(buf_type) },
        { NvSciBufGeneralAttrKey_RequiredPerm, &access_perm, sizeof(access_perm) },
        { NvSciBufGeneralAttrKey_NeedCpuAccess, &cpu_access, sizeof(cpu_access) },
        { NvSciBufGeneralAttrKey_EnableCpuCache, &cpu_cache, sizeof(cpu_cache) },
        { NvSciBufImageAttrKey_Layout, &layout_type, sizeof(layout_type) },
        { NvSciBufImageAttrKey_SurfType, &surf_type, sizeof(surf_type) },
        { NvSciBufImageAttrKey_SurfMemLayout, &mem_layout, sizeof(mem_layout) },
        { NvSciBufImageAttrKey_SurfSampleType, &sample_type, sizeof(sample_type) },
        { NvSciBufImageAttrKey_SurfBPC, &surf_bpc, sizeof(surf_bpc) },
        { NvSciBufImageAttrKey_SurfComponentOrder, &comp_order, sizeof(comp_order) },
        { NvSciBufImageAttrKey_SurfColorStd, &color_std, sizeof(color_std) },
    };

    return NvSciBufAttrListSetAttrs(attr_list, kvp, sizeof(kvp) / sizeof(kvp[0]));
}


/// Fill a second attribute list with the NvBufSurface attributes matching the
/// SIPL list, so the allocated buffer satisfies both consumers. Without it the
/// buffer is allocated for SIPL alone and RegisterImages rejects it with
/// NVSIPL_STATUS_INVALID_STATE -- a status whose only documented meaning is
/// "pipeline had seen an init error", which sends you looking in the wrong
/// place entirely.
///
/// Adapted from SetNvBufNvSciBufAttributes in NVIDIA's NvNvSciBufHelper.cpp
/// (BSD-3-Clause). Its colour table has exactly two entries, both REC709_ER;
/// that is the whole set this ISP path can produce.
NvSciError set_nvbuf_attributes(NvSciBufAttrList nvbuf_attrs, NvSciBufAttrList sipl_attrs, int gpu_id)
{
    NvSciBufAttrKeyValuePair img[] = {
        { NvSciBufImageAttrKey_SurfWidthBase, nullptr, 0 },  { NvSciBufImageAttrKey_SurfHeightBase, nullptr, 0 },
        { NvSciBufImageAttrKey_SurfSampleType, nullptr, 0 }, { NvSciBufImageAttrKey_SurfColorStd, nullptr, 0 },
        { NvSciBufImageAttrKey_SurfMemLayout, nullptr, 0 },  { NvSciBufImageAttrKey_SurfBPC, nullptr, 0 },
        { NvSciBufImageAttrKey_Layout, nullptr, 0 },
    };
    const NvSciError err = NvSciBufAttrListGetAttrs(sipl_attrs, img, sizeof(img) / sizeof(img[0]));
    if (err != NvSciError_Success)
    {
        return err;
    }

    const auto width = *static_cast<const uint32_t*>(img[0].value);
    const auto height = *static_cast<const uint32_t*>(img[1].value);
    const auto sample = *static_cast<const NvSciBufSurfSampleType*>(img[2].value);
    const auto color = *static_cast<const NvSciBufAttrValColorStd*>(img[3].value);
    const auto mem = *static_cast<const NvSciBufSurfMemLayout*>(img[4].value);
    const auto bpc = *static_cast<const NvSciBufSurfBPC*>(img[5].value);
    const auto layout = *static_cast<const NvSciBufAttrValImageLayoutType*>(img[6].value);

    NvBufSurfaceColorFormat pixfmt;
    if (color == NvSciColorStd_REC709_ER && sample == NvSciSurfSampleType_420 && bpc == NvSciSurfBPC_8 &&
        mem == NvSciSurfMemLayout_SemiPlanar)
    {
        pixfmt = NVBUF_COLOR_FORMAT_NV12_709_ER;
    }
    else if (color == NvSciColorStd_REC709_ER && sample == NvSciSurfSampleType_420 && bpc == NvSciSurfBPC_8 &&
             mem == NvSciSurfMemLayout_Planar)
    {
        pixfmt = NVBUF_COLOR_FORMAT_YUV420_709_ER;
    }
    else
    {
        std::cerr << "[sipl] no NvBufSurface colour format for colourStd=" << static_cast<int>(color)
                  << " sample=" << static_cast<int>(sample) << " bpc=" << static_cast<int>(bpc)
                  << " memLayout=" << static_cast<int>(mem) << std::endl;
        return NvSciError_BadParameter;
    }

    NvmmImageParams params{};
    params.gpuId = static_cast<uint32_t>(gpu_id);
    params.width = width;
    params.height = height;
    params.colorFormat = pixfmt;
    params.layout = (layout == NvSciBufImage_PitchLinearType) ? NVBUF_LAYOUT_PITCH : NVBUF_LAYOUT_BLOCK_LINEAR;
    params.displayscanformat = NVBUF_DISPLAYSCANFORMAT_PROGRESSIVE;
    params.isProtected = false;

    return (NvmmImageFillNvSciBufAttrList(&params, nvbuf_attrs) == 0) ? NvSciError_Success : NvSciError_BadParameter;
}

/// Read back what reconciliation settled on, and report whether it is extended
/// range. _ER is full range (0-255), _SR is studio range (16-235); decoding one
/// as the other subtracts a black level that is not there and crushes shadows,
/// which on a dim scene loses most of the image. Returns true for full range.
bool check_color_std(NvSciBufAttrList reconciled)
{
    NvSciBufAttrKeyValuePair query{ NvSciBufImageAttrKey_SurfColorStd, nullptr, 0 };
    if (NvSciBufAttrListGetAttrs(reconciled, &query, 1) != NvSciError_Success || query.value == nullptr)
    {
        std::cerr << "[sipl] warning: could not read back SurfColorStd; assuming BT.709 extended range" << std::endl;
        return true;
    }
    const auto* value = static_cast<const NvSciBufAttrValColorStd*>(query.value);
    if (*value != NvSciColorStd_REC709_ER && *value != NvSciColorStd_REC709_SR)
    {
        std::ostringstream oss;
        oss << "ISP0 reconciled to colour standard " << static_cast<int>(*value)
            << "; yuv_to_rgba.cu implements BT.601 and BT.709 only, and decoding anything else as"
               " 709 would be a silent hue error";
        throw std::runtime_error(oss.str());
    }
    return *value == NvSciColorStd_REC709_ER;
}

std::vector<uint8_t> load_nito(const std::string& dir, const std::string& module)
{
    const std::string path = dir + "/" + module + ".nito";
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in)
    {
        throw std::runtime_error("cannot open NITO file " + path +
                                 " (the ISP has no tuning; run the vendor install.sh on the host)");
    }
    const auto size = static_cast<std::streamsize>(in.tellg());
    in.seekg(0);
    std::vector<uint8_t> blob(static_cast<size_t>(size));
    if (!in.read(reinterpret_cast<char*>(blob.data()), size))
    {
        throw std::runtime_error("short read on NITO file " + path);
    }
    return blob;
}

/// Walk a SensorSystemConfig into flat SensorInfo records. The nesting is
/// module -> variant -> sensorConfigs -> variant, and `id` is the pipeline
/// index SetPipelineCfg wants.
std::vector<SensorInfo> flatten(const nvsipl::sensorconfig::SensorSystemConfig& cfg)
{
    std::vector<SensorInfo> out;
    for (const auto& module : cfg.modules)
    {
        std::visit(
            [&](const auto& mod)
            {
                for (const auto& sensor_variant : mod.sensorConfigs)
                {
                    std::visit(
                        [&](const auto& sensor)
                        {
                            SensorInfo info;
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
                        sensor_variant);
                }
            },
            module.moduleType);
    }
    return out;
}

} // namespace

// =============================================================================
// Impl
// =============================================================================

struct SiplCamera::Impl
{
    /// One ISP0 buffer, imported into CUDA once at registration. Nothing here
    /// is re-created per frame -- that is the whole point of registering.
    struct Slot
    {
        NvSciBufObj buf = nullptr;
        NvBufSurface* surface = nullptr;
        CUgraphicsResource resource = nullptr;
        CUeglFrame egl_frame{};
    };

    struct DeviceBuffer
    {
        uint8_t* ptr = nullptr;
        size_t pitch = 0;
    };

    struct Sensor
    {
        SensorInfo info;
        nvsipl::NvSIPLPipelineConfiguration pipeline_config{};
        nvsipl::NvSIPLPipelineQueues queues{};

        /// From the reconciled surface, not from config: the ISP decides.
        bool full_range = true;
        NvSciBufAttrList buf_attrs = nullptr;
        std::vector<NvSciBufObj> buf_objects;
        /// ICP pool. Never read -- allocated only because SIPL requires it.
        NvSciBufAttrList icp_attrs = nullptr;
        std::vector<NvSciBufObj> icp_buf_objects;
        std::vector<Slot> slots;
        NvSciSyncObj sync_obj = nullptr;

        /// Triple-buffered RGBA8 mailbox. A reader leases the published slot
        /// until its next latest() call, so the producer never overwrites it.
        std::array<DeviceBuffer, 3> buffers{};
        mutable std::mutex publish_mutex;
        int publish_idx = -1;
        int lease_idx = -1;
        uint64_t published_sequence = 0;
        uint64_t consumed_sequence = 0;
        uint64_t published_timestamp_ns = 0;
        uint64_t published_capture_tsc_ns = 0;

        std::thread frame_thread;
        std::thread event_thread;
    };

    explicit Impl(const SiplConfig& cfg) : config(cfg)
    {
    }

    SiplConfig config;

    NvSciBufModule sci_buf_module = nullptr;
    NvSciSyncModule sci_sync_module = nullptr;
    NvSciSyncCpuWaitContext cpu_wait_context = nullptr;

    CUdevice cu_device = 0;
    CUcontext cu_context = nullptr;
    bool cu_context_retained = false;
    CUstream convert_stream = nullptr;

    std::unique_ptr<INvSIPLCamera> camera;
    std::unique_ptr<INvSIPLCameraQuery> query_api;
    std::unique_ptr<nvsipl::sensorconfig::SensorSystemConfig> system_config;

    std::vector<SensorInfo> sensor_infos;
    std::vector<std::unique_ptr<Sensor>> sensors;

    std::atomic<bool> running{ false };
    std::atomic<bool> failed{ false };
    mutable std::mutex error_mutex;
    std::string failure_message;

    Sensor* find(uint32_t sensor_id)
    {
        for (auto& s : sensors)
        {
            if (s->info.id == sensor_id)
            {
                return s.get();
            }
        }
        return nullptr;
    }

    void set_failure(const std::string& message)
    {
        std::lock_guard<std::mutex> guard(error_mutex);
        if (failure_message.empty())
        {
            failure_message = message;
        }
        failed.store(true);
    }

    void throw_if_failed() const
    {
        if (!failed.load())
        {
            return;
        }
        std::lock_guard<std::mutex> guard(error_mutex);
        throw std::runtime_error(failure_message.empty() ? "SIPL capture failed" : failure_message);
    }

    void init_cuda();
    void init_nvsci();
    void configure();
    void allocate_buffers(Sensor& sensor);
    void allocate_icp_buffers(Sensor& sensor);
    void register_buffers(Sensor& sensor);
    void allocate_sync(Sensor& sensor);
    void register_sync(Sensor& sensor);
    void register_nito(Sensor& sensor);
    void frame_loop(Sensor& sensor);
    void event_loop(Sensor& sensor);
    void convert(Sensor& sensor, const Slot& slot, uint32_t write_idx);
    void publish(Sensor& sensor, uint32_t write_idx, uint64_t timestamp_ns, uint64_t capture_tsc_ns);
    uint32_t pick_write_index(const Sensor& sensor) const;
    void cleanup();
};

void SiplCamera::Impl::init_cuda()
{
    check_cuda(cuInit(0), "cuInit");
    check_cuda(cuDeviceGet(&cu_device, config.gpu_id), "cuDeviceGet");
    check_cuda(cuDevicePrimaryCtxRetain(&cu_context, cu_device), "cuDevicePrimaryCtxRetain");
    cu_context_retained = true;
    check_cuda(cuCtxSetCurrent(cu_context), "cuCtxSetCurrent");
    check_cuda(cuStreamCreate(&convert_stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate");
}

void SiplCamera::Impl::init_nvsci()
{
    check_sci(NvSciBufModuleOpen(&sci_buf_module), "NvSciBufModuleOpen");
    check_sci(NvSciSyncModuleOpen(&sci_sync_module), "NvSciSyncModuleOpen");
    check_sci(NvSciSyncCpuWaitContextAlloc(sci_sync_module, &cpu_wait_context), "NvSciSyncCpuWaitContextAlloc");
}

void SiplCamera::Impl::configure()
{
    init_sipl_trace();
    query_api = INvSIPLCameraQuery::GetInstance();
    if (!query_api)
    {
        throw std::runtime_error("INvSIPLCameraQuery::GetInstance returned null");
    }
    // Loads the UDDF driver .so libs out of /usr/lib/nvsipl_drv.
    check_sipl(query_api->ParseDatabase(), "ParseDatabase");
    check_sipl(query_api->ParseJsonFile(config.platform_config_json), "ParseJsonFile");

    system_config = std::make_unique<nvsipl::sensorconfig::SensorSystemConfig>();
    check_sipl(query_api->GetSensorSystemConfig(config.platform_config_name, *system_config), "GetSensorSystemConfig");
    if (config.link_masks.empty())
    {
        throw std::runtime_error("link masks are required for a GMSL platform config");
    }
    check_sipl(query_api->ApplyMask(*system_config, config.link_masks), "ApplyMask");
    if (system_config->modules.empty())
    {
        throw std::runtime_error("no modules left in '" + config.platform_config_name +
                                 "' after applying the link masks -- check the mask against the config");
    }

    sensor_infos = flatten(*system_config);
    if (sensor_infos.empty())
    {
        throw std::runtime_error("platform config '" + config.platform_config_name + "' declares no sensors");
    }

    camera = INvSIPLCamera::GetInstance();
    if (!camera)
    {
        throw std::runtime_error("INvSIPLCamera::GetInstance returned null");
    }
    check_sipl(camera->SetPlatformCfg(*system_config), "SetPlatformCfg");

    for (const auto& info : sensor_infos)
    {
        auto sensor = std::make_unique<Sensor>();
        sensor->info = info;
        // ISP0 only. ICP would double the capture bandwidth for a raw stream
        // nothing here consumes, and ISP1/ISP2 are a downscale we do not want.
        sensor->pipeline_config.captureOutputRequested = false;
        sensor->pipeline_config.isp0OutputRequested = true;
        sensor->pipeline_config.isp1OutputRequested = false;
        sensor->pipeline_config.isp2OutputRequested = false;
        // Defaults to false, i.e. subframe ENABLED, which needs a sliceCount we
        // do not set. NVIDIA's own samples disable it unless asked for.
        sensor->pipeline_config.disableSubframe = true;
        sensor->pipeline_config.bufferCfg.maxIsp0BufferCount = config.isp0_buffers;
        check_sipl(camera->SetPipelineCfg(info.id, sensor->pipeline_config, sensor->queues), "SetPipelineCfg");
        sensors.push_back(std::move(sensor));
    }

    // Order is not ours to choose. Everything that *describes* a pipeline --
    // GetImageAttributes and FillNvSciSyncAttrList -- must happen before Init();
    // everything that *binds* a resource to it -- RegisterImages and
    // RegisterNvSciSyncObj -- must happen after. Get it wrong and Init()
    // still returns OK, but the pipeline is left in an error state and the
    // first Register* call returns NVSIPL_STATUS_INVALID_STATE with no clue
    // why. Verified against a -v 4 trace of NVIDIA's own nvsipl_camera.
    for (auto& sensor : sensors)
    {
        allocate_icp_buffers(*sensor);
        allocate_buffers(*sensor);
        allocate_sync(*sensor);
    }

    check_sipl(camera->Init(), "INvSIPLCamera::Init");

    for (auto& sensor : sensors)
    {
        register_buffers(*sensor);
        register_sync(*sensor);
        register_nito(*sensor);
    }
}

void SiplCamera::Impl::allocate_buffers(Sensor& sensor)
{
    NvSciBufAttrList sipl_attrs = nullptr;
    NvSciBufAttrList nvbuf_attrs = nullptr;
    NvSciBufAttrList conflict = nullptr;
    check_sci(NvSciBufAttrListCreate(sci_buf_module, &sipl_attrs), "NvSciBufAttrListCreate sipl");
    check_sci(NvSciBufAttrListCreate(sci_buf_module, &nvbuf_attrs), "NvSciBufAttrListCreate nvbuf");
    check_sci(set_sipl_buf_attributes(sipl_attrs), "NvSciBufAttrListSetAttrs");
    check_sipl(camera->GetImageAttributes(sensor.info.id, kIsp0, sipl_attrs), "GetImageAttributes");

    // Both consumers must be described before allocation: SIPL writes the
    // surface, NvBufSurface reads it on the way to CUDA. Reconciling the SIPL
    // list alone yields a buffer RegisterImages will not accept.
    check_sci(set_nvbuf_attributes(nvbuf_attrs, sipl_attrs, config.gpu_id), "set_nvbuf_attributes");

    NvSciBufAttrList unreconciled[] = { sipl_attrs, nvbuf_attrs };
    check_sci(NvSciBufAttrListReconcile(unreconciled, 2U, &sensor.buf_attrs, &conflict), "NvSciBufAttrListReconcile");
    NvSciBufAttrListFree(sipl_attrs);
    NvSciBufAttrListFree(nvbuf_attrs);
    if (conflict)
    {
        NvSciBufAttrListFree(conflict);
    }
    sensor.full_range = check_color_std(sensor.buf_attrs);

    sensor.slots.resize(config.isp0_buffers);
    for (uint32_t i = 0; i < config.isp0_buffers; ++i)
    {
        Slot& slot = sensor.slots[i];
        check_sci(NvSciBufObjAlloc(sensor.buf_attrs, &slot.buf), "NvSciBufObjAlloc");
        sensor.buf_objects.push_back(slot.buf);

        if (NvmmNvSciBufToNvBufSurface(slot.buf, &slot.surface) != 0 || slot.surface == nullptr)
        {
            throw std::runtime_error("NvmmNvSciBufToNvBufSurface failed");
        }
        slot.surface->numFilled = 1;
    }

    // RGBA8 destination slots. Allocated from the reported geometry, never from
    // a CLI flag -- see "Geometry comes from the query" in the plan.
    for (auto& dest : sensor.buffers)
    {
        void* ptr = nullptr;
        size_t pitch = 0;
        check_runtime(cudaMallocPitch(&ptr, &pitch, static_cast<size_t>(sensor.info.width) * 4, sensor.info.height),
                      "cudaMallocPitch");
        dest.ptr = static_cast<uint8_t*>(ptr);
        dest.pitch = pitch;
    }
}

void SiplCamera::Impl::allocate_icp_buffers(Sensor& sensor)
{
    NvSciBufAttrList attrs = nullptr;
    NvSciBufAttrList conflict = nullptr;
    check_sci(NvSciBufAttrListCreate(sci_buf_module, &attrs), "NvSciBufAttrListCreate icp");

    // Only the type and permissions: GetImageAttributes fills the raw format in.
    NvSciBufType buf_type = NvSciBufType_Image;
    NvSciBufAttrValAccessPerm perm = NvSciBufAccessPerm_ReadWrite;
    NvSciBufAttrKeyValuePair kvp[] = {
        { NvSciBufGeneralAttrKey_Types, &buf_type, sizeof(buf_type) },
        { NvSciBufGeneralAttrKey_RequiredPerm, &perm, sizeof(perm) },
    };
    check_sci(NvSciBufAttrListSetAttrs(attrs, kvp, 2), "NvSciBufAttrListSetAttrs icp");
    check_sipl(camera->GetImageAttributes(sensor.info.id, kIcp, attrs), "GetImageAttributes ICP");

    NvSciBufAttrList unreconciled[] = { attrs };
    check_sci(NvSciBufAttrListReconcile(unreconciled, 1U, &sensor.icp_attrs, &conflict), "NvSciBufAttrListReconcile icp");
    NvSciBufAttrListFree(attrs);
    if (conflict)
    {
        NvSciBufAttrListFree(conflict);
    }

    for (uint32_t i = 0; i < kIcpBuffers; ++i)
    {
        NvSciBufObj obj = nullptr;
        check_sci(NvSciBufObjAlloc(sensor.icp_attrs, &obj), "NvSciBufObjAlloc icp");
        sensor.icp_buf_objects.push_back(obj);
    }
}

void SiplCamera::Impl::register_buffers(Sensor& sensor)
{
    check_sipl(camera->RegisterImages(sensor.info.id, kIcp, sensor.icp_buf_objects), "RegisterImages ICP");
    check_sipl(camera->RegisterImages(sensor.info.id, kIsp0, sensor.buf_objects), "RegisterImages ISP0");

    // Import into CUDA only once SIPL owns the buffers. Doing it at allocation
    // time hands SIPL a surface that already has an EGL image bound to it.
    // Registering per frame instead would put an EGL round trip on the 60 Hz
    // path for no benefit.
    for (auto& slot : sensor.slots)
    {
        if (NvBufSurfaceMapEglImage(slot.surface, 0) != 0)
        {
            throw std::runtime_error("NvBufSurfaceMapEglImage failed");
        }
        check_cuda(cuGraphicsEGLRegisterImage(&slot.resource, slot.surface->surfaceList[0].mappedAddr.eglImage,
                                              CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY),
                   "cuGraphicsEGLRegisterImage");
        check_cuda(cuGraphicsResourceGetMappedEglFrame(&slot.egl_frame, slot.resource, 0, 0),
                   "cuGraphicsResourceGetMappedEglFrame");
    }
}

void SiplCamera::Impl::allocate_sync(Sensor& sensor)
{
    NvSciSyncAttrList signaler = nullptr;
    NvSciSyncAttrList waiter = nullptr;
    NvSciSyncAttrList reconciled = nullptr;
    NvSciSyncAttrList conflict = nullptr;

    check_sci(NvSciSyncAttrListCreate(sci_sync_module, &signaler), "NvSciSyncAttrListCreate signaler");
    check_sci(NvSciSyncAttrListCreate(sci_sync_module, &waiter), "NvSciSyncAttrListCreate waiter");

    // We wait on the CPU before handing the surface to the conversion kernel.
    NvSciSyncAccessPerm perm = NvSciSyncAccessPerm_WaitOnly;
    bool cpu_waiter = true;
    NvSciSyncAttrKeyValuePair kvp[] = {
        { NvSciSyncAttrKey_NeedCpuAccess, &cpu_waiter, sizeof(cpu_waiter) },
        { NvSciSyncAttrKey_RequiredPerm, &perm, sizeof(perm) },
    };
    check_sci(NvSciSyncAttrListSetAttrs(waiter, kvp, 2), "NvSciSyncAttrListSetAttrs");
    check_sipl(
        camera->FillNvSciSyncAttrList(sensor.info.id, kIsp0, signaler, nvsipl::SIPL_SIGNALER), "FillNvSciSyncAttrList");

    NvSciSyncAttrList unreconciled[] = { signaler, waiter };
    check_sci(NvSciSyncAttrListReconcile(unreconciled, 2U, &reconciled, &conflict), "NvSciSyncAttrListReconcile");
    check_sci(NvSciSyncObjAlloc(reconciled, &sensor.sync_obj), "NvSciSyncObjAlloc");
    NvSciSyncAttrListFree(signaler);
    NvSciSyncAttrListFree(waiter);
    if (reconciled)
    {
        NvSciSyncAttrListFree(reconciled);
    }
    if (conflict)
    {
        NvSciSyncAttrListFree(conflict);
    }
}

void SiplCamera::Impl::register_sync(Sensor& sensor)
{
    check_sipl(camera->RegisterNvSciSyncObj(sensor.info.id, kIsp0, nvsipl::NVSIPL_EOFSYNCOBJ, sensor.sync_obj),
               "RegisterNvSciSyncObj");
}

void SiplCamera::Impl::register_nito(Sensor& sensor)
{
    auto blob = load_nito(config.nito_dir, sensor.info.name);
    nvsipl::ISiplControlAuto* auto_control = nullptr;
    check_sipl(camera->RegisterAutoControlPlugin(sensor.info.id, nvsipl::NV_PLUGIN, auto_control, blob),
               "RegisterAutoControlPlugin");
}

void SiplCamera::Impl::convert(Sensor& sensor, const Slot& slot, uint32_t write_idx)
{
    const CUeglFrame& frame = slot.egl_frame;
    DeviceBuffer& dest = sensor.buffers[write_idx];

    if (frame.planeCount < 2)
    {
        throw std::runtime_error("ISP0 surface has fewer than two planes");
    }

    // Requested SemiPlanar YUV, so this is NV12 unless swap_uv flips it, and
    // REC709_ER, so the kernel gets bt709=true.
    const YuvLayout layout = config.swap_uv ? YuvLayout::YVU420SemiPlanar : YuvLayout::YUV420SemiPlanar;
    constexpr bool kBt709 = true;

    if (frame.frameType == CU_EGL_FRAME_TYPE_PITCH)
    {
        const auto* y_plane = static_cast<const uint8_t*>(frame.frame.pPitch[0]);
        const auto* uv_plane = static_cast<const uint8_t*>(frame.frame.pPitch[1]);
        launch_yuv420_pitch_to_rgba(y_plane, uv_plane, nullptr, static_cast<int>(frame.pitch),
                                    static_cast<int>(frame.pitch), 0, sensor.info.width, sensor.info.height, dest.ptr,
                                    static_cast<int>(dest.pitch), layout, sensor.full_range, kBt709,
                                    reinterpret_cast<cudaStream_t>(convert_stream));
    }
    else if (frame.frameType == CU_EGL_FRAME_TYPE_ARRAY)
    {
        // Block-linear: CUDA exposes the planes as arrays, so the kernel reads
        // them through texture objects. Those are built per frame because the
        // CUarray handles come from the per-frame CUeglFrame; the expensive
        // part, cuGraphicsEGLRegisterImage, is done once per buffer.
        cudaTextureObject_t y_tex = texture_for_array(frame.frame.pArray[0]);
        cudaTextureObject_t uv_tex = texture_for_array(frame.frame.pArray[1]);
        launch_yuv420_array_to_rgba(y_tex, uv_tex, 0, sensor.info.width, sensor.info.height, dest.ptr,
                                    static_cast<int>(dest.pitch), layout, sensor.full_range, kBt709,
                                    reinterpret_cast<cudaStream_t>(convert_stream));
        check_cuda(cuStreamSynchronize(convert_stream), "cuStreamSynchronize");
        check_runtime(cudaDestroyTextureObject(y_tex), "cudaDestroyTextureObject");
        check_runtime(cudaDestroyTextureObject(uv_tex), "cudaDestroyTextureObject");
        check_runtime(cudaGetLastError(), "SIPL YUV to RGBA kernel");
        return;
    }
    else
    {
        throw std::runtime_error("unsupported CUDA EGL frame type from ISP0");
    }

    check_cuda(cuStreamSynchronize(convert_stream), "cuStreamSynchronize");
    check_runtime(cudaGetLastError(), "SIPL YUV to RGBA kernel");
}

void SiplCamera::Impl::frame_loop(Sensor& sensor)
{
    try
    {
        check_cuda(cuCtxSetCurrent(cu_context), "cuCtxSetCurrent frame_loop");
        while (running.load())
        {
            INvSIPLClient::INvSIPLBuffer* buffer = nullptr;
            const SIPLStatus status = sensor.queues.isp0CompletionQueue->Get(buffer, config.frame_timeout_ms * 1000U);
            if (status == nvsipl::NVSIPL_STATUS_TIMED_OUT)
            {
                continue;
            }
            if (status == nvsipl::NVSIPL_STATUS_EOF)
            {
                break;
            }
            check_sipl(status, "isp0CompletionQueue->Get");
            if (!buffer)
            {
                continue;
            }

            auto* nvm = dynamic_cast<INvSIPLClient::INvSIPLNvMBuffer*>(buffer);
            if (!nvm)
            {
                buffer->Release();
                throw std::runtime_error("ISP0 buffer is not an INvSIPLNvMBuffer");
            }

            // The ISP is still writing until this fence clears.
            NvSciSyncFence fence = NvSciSyncFenceInitializer;
            check_sipl(nvm->GetEOFNvSciSyncFence(&fence), "GetEOFNvSciSyncFence");
            const NvSciError wait = NvSciSyncFenceWait(&fence, cpu_wait_context, config.frame_timeout_ms * 1000UL);
            NvSciSyncFenceClear(&fence);
            check_sci(wait, "NvSciSyncFenceWait");

            const NvSciBufObj arrived = nvm->GetNvSciBufImage();
            const Slot* slot = nullptr;
            for (const auto& candidate : sensor.slots)
            {
                if (candidate.buf == arrived)
                {
                    slot = &candidate;
                    break;
                }
            }
            if (!slot)
            {
                buffer->Release();
                throw std::runtime_error("ISP0 delivered a buffer this process never registered");
            }

            // The TSC is what makes the two eyes pairable; do not substitute a
            // host clock, which carries this sensor's own queueing jitter.
            const auto& meta = nvm->GetImageData();
            const uint64_t capture_tsc_ns = meta.frameCaptureTSC;
            // SENSING_AE_PROBE=1 logs the sensor's exposure and gain once a
            // second. AE failures are invisible in the frame rate and hard to
            // judge by eye, and this is the only place the numbers surface.
            if (std::getenv("SENSING_AE_PROBE") != nullptr)
            {
                static thread_local uint64_t probe_n = 0;
                if ((probe_n++ % 60) == 0)
                {
                    std::cerr << "[ae] sensor " << sensor.info.id << " numExp=" << meta.numExposures
                              << " expValid=" << static_cast<int>(meta.sensorExpInfo.expTimeValid)
                              << " exp[0]=" << meta.sensorExpInfo.exposureTime[0] << "s"
                              << " gainValid=" << static_cast<int>(meta.sensorExpInfo.gainValid)
                              << " gain[0]=" << meta.sensorExpInfo.sensorGain[0] << std::endl;
                }
            }

            const uint32_t write_idx = pick_write_index(sensor);
            convert(sensor, *slot, write_idx);
            if (std::getenv("SENSING_LUMA_PROBE") != nullptr)
            {
                static thread_local uint64_t luma_n = 0;
                if ((luma_n++ % 60) == 0)
                {
                    // Mean over one row out of every 64, straight off the RGBA
                    // the consumer receives -- the only number that says what
                    // the operator actually sees.
                    const auto& dst = sensor.buffers[write_idx];
                    const uint32_t step = 64;
                    std::vector<uint8_t> row(dst.pitch);
                    double sum = 0.0;
                    uint64_t n = 0;
                    for (uint32_t y = 0; y < sensor.info.height; y += step)
                    {
                        if (cudaMemcpy(row.data(), dst.ptr + static_cast<size_t>(y) * dst.pitch, dst.pitch,
                                       cudaMemcpyDeviceToHost) != cudaSuccess)
                            break;
                        for (uint32_t x = 0; x < sensor.info.width; ++x)
                        {
                            sum += 0.2126 * row[x * 4 + 0] + 0.7152 * row[x * 4 + 1] + 0.0722 * row[x * 4 + 2];
                            ++n;
                        }
                    }
                    if (n)
                        std::cerr << "[luma] sensor " << sensor.info.id << " mean=" << (sum / n) << "/255" << std::endl;
                }
            }
            buffer->Release();
            publish(sensor, write_idx, monotonic_ns(), capture_tsc_ns);
        }
    }
    catch (const std::exception& e)
    {
        std::ostringstream oss;
        oss << "SIPL sensor " << sensor.info.id << " capture error: " << e.what();
        std::cerr << "[sipl] " << oss.str() << std::endl;
        set_failure(oss.str());
        running.store(false);
    }
}

void SiplCamera::Impl::event_loop(Sensor& sensor)
{
    // A GMSL link drop arrives here and nowhere else. Without this drain the
    // symptom is a silent stall on the frame queue.
    while (running.load())
    {
        nvsipl::NvSIPLPipelineNotifier::NotificationData data{};
        const SIPLStatus status = sensor.queues.notificationQueue->Get(data, config.frame_timeout_ms * 1000U);
        if (status != NVSIPL_STATUS_OK)
        {
            continue;
        }
        using Notif = nvsipl::NvSIPLPipelineNotifier;
        switch (data.eNotifType)
        {
        case Notif::NOTIF_ERROR_DESERIALIZER_FAILURE:
        case Notif::NOTIF_ERROR_SERIALIZER_FAILURE:
        case Notif::NOTIF_ERROR_SENSOR_FAILURE:
        case Notif::NOTIF_ERROR_INTERNAL_FAILURE:
        {
            std::ostringstream oss;
            oss << "SIPL sensor " << sensor.info.id << " pipeline error, notification "
                << static_cast<int>(data.eNotifType);
            std::cerr << "[sipl] " << oss.str() << std::endl;
            set_failure(oss.str());
            running.store(false);
            break;
        }
        default:
            break;
        }
    }
}

void SiplCamera::Impl::publish(Sensor& sensor, uint32_t write_idx, uint64_t timestamp_ns, uint64_t capture_tsc_ns)
{
    std::lock_guard<std::mutex> guard(sensor.publish_mutex);
    sensor.publish_idx = static_cast<int>(write_idx);
    sensor.published_timestamp_ns = timestamp_ns;
    sensor.published_capture_tsc_ns = capture_tsc_ns;
    ++sensor.published_sequence;
}

uint32_t SiplCamera::Impl::pick_write_index(const Sensor& sensor) const
{
    std::lock_guard<std::mutex> guard(sensor.publish_mutex);
    if (sensor.publish_idx < 0)
    {
        return 0;
    }
    // Skip the published slot and any slot still leased to a reader. With three
    // slots one is always free, so this cannot spin.
    uint32_t idx = static_cast<uint32_t>((sensor.publish_idx + 1) % 3);
    if (static_cast<int>(idx) == sensor.lease_idx)
    {
        idx = (idx + 1) % 3;
    }
    return idx;
}

void SiplCamera::Impl::cleanup()
{
    running.store(false);
    for (auto& sensor : sensors)
    {
        if (sensor->frame_thread.joinable())
        {
            sensor->frame_thread.join();
        }
        if (sensor->event_thread.joinable())
        {
            sensor->event_thread.join();
        }
    }
    if (camera)
    {
        camera->Stop();
        camera->Deinit();
    }

    for (auto& sensor : sensors)
    {
        for (auto& slot : sensor->slots)
        {
            if (slot.resource)
            {
                cuGraphicsUnregisterResource(slot.resource);
            }
            if (slot.surface)
            {
                NvBufSurfaceUnMapEglImage(slot.surface, 0);
                NvBufSurfaceDestroy(slot.surface);
            }
            if (slot.buf)
            {
                NvSciBufObjFree(slot.buf);
            }
        }
        sensor->slots.clear();
        sensor->buf_objects.clear();
        for (auto& obj : sensor->icp_buf_objects)
        {
            NvSciBufObjFree(obj);
        }
        sensor->icp_buf_objects.clear();
        if (sensor->icp_attrs)
        {
            NvSciBufAttrListFree(sensor->icp_attrs);
            sensor->icp_attrs = nullptr;
        }
        if (sensor->buf_attrs)
        {
            NvSciBufAttrListFree(sensor->buf_attrs);
            sensor->buf_attrs = nullptr;
        }
        if (sensor->sync_obj)
        {
            NvSciSyncObjFree(sensor->sync_obj);
            sensor->sync_obj = nullptr;
        }
        for (auto& dest : sensor->buffers)
        {
            if (dest.ptr)
            {
                cudaFree(dest.ptr);
                dest.ptr = nullptr;
            }
        }
    }

    camera.reset();
    query_api.reset();

    if (cpu_wait_context)
    {
        NvSciSyncCpuWaitContextFree(cpu_wait_context);
        cpu_wait_context = nullptr;
    }
    if (sci_sync_module)
    {
        NvSciSyncModuleClose(sci_sync_module);
        sci_sync_module = nullptr;
    }
    if (sci_buf_module)
    {
        NvSciBufModuleClose(sci_buf_module);
        sci_buf_module = nullptr;
    }
    if (convert_stream)
    {
        cuStreamDestroy(convert_stream);
        convert_stream = nullptr;
    }
    if (cu_context_retained)
    {
        cuDevicePrimaryCtxRelease(cu_device);
        cu_context_retained = false;
        cu_context = nullptr;
    }
}

// =============================================================================
// SiplCamera
// =============================================================================

SiplCamera::SiplCamera(const SiplConfig& config) : m_impl(std::make_unique<Impl>(config))
{
    if (config.platform_config_json.empty() || config.platform_config_name.empty())
    {
        throw std::invalid_argument("SiplConfig needs both platform_config_json and platform_config_name");
    }
    try
    {
        m_impl->init_cuda();
        m_impl->init_nvsci();
        m_impl->configure();
    }
    catch (...)
    {
        m_impl->cleanup();
        throw;
    }
}

SiplCamera::~SiplCamera()
{
    m_impl->cleanup();
}

std::vector<SensorInfo> SiplCamera::query(const std::string& platform_config_json,
                                          const std::string& platform_config_name,
                                          const std::vector<uint32_t>& link_masks)
{
    auto api = INvSIPLCameraQuery::GetInstance();
    if (!api)
    {
        throw std::runtime_error("INvSIPLCameraQuery::GetInstance returned null");
    }
    check_sipl(api->ParseDatabase(), "ParseDatabase");
    check_sipl(api->ParseJsonFile(platform_config_json), "ParseJsonFile");

    nvsipl::sensorconfig::SensorSystemConfig cfg;
    check_sipl(api->GetSensorSystemConfig(platform_config_name, cfg), "GetSensorSystemConfig");
    if (!link_masks.empty())
    {
        check_sipl(api->ApplyMask(cfg, link_masks), "ApplyMask");
    }
    return flatten(cfg);
}

const std::vector<SensorInfo>& SiplCamera::sensors() const
{
    return m_impl->sensor_infos;
}

void SiplCamera::start()
{
    if (m_impl->running.load())
    {
        return;
    }
    m_impl->running.store(true);
    for (auto& sensor : m_impl->sensors)
    {
        sensor->frame_thread = std::thread([this, s = sensor.get()] { m_impl->frame_loop(*s); });
        sensor->event_thread = std::thread([this, s = sensor.get()] { m_impl->event_loop(*s); });
    }
    try
    {
        check_sipl(m_impl->camera->Start(), "INvSIPLCamera::Start");
    }
    catch (...)
    {
        m_impl->running.store(false);
        for (auto& sensor : m_impl->sensors)
        {
            if (sensor->frame_thread.joinable())
            {
                sensor->frame_thread.join();
            }
            if (sensor->event_thread.joinable())
            {
                sensor->event_thread.join();
            }
        }
        throw;
    }
}

void SiplCamera::stop()
{
    m_impl->running.store(false);
    for (auto& sensor : m_impl->sensors)
    {
        if (sensor->frame_thread.joinable())
        {
            sensor->frame_thread.join();
        }
        if (sensor->event_thread.joinable())
        {
            sensor->event_thread.join();
        }
    }
    if (m_impl->camera)
    {
        m_impl->camera->Stop();
    }
}

std::optional<FrameView> SiplCamera::latest(uint32_t sensor_id)
{
    m_impl->throw_if_failed();
    Impl::Sensor* sensor = m_impl->find(sensor_id);
    if (!sensor)
    {
        throw std::runtime_error("no SIPL pipeline with index " + std::to_string(sensor_id));
    }

    std::lock_guard<std::mutex> guard(sensor->publish_mutex);
    if (sensor->publish_idx < 0 || sensor->consumed_sequence == sensor->published_sequence)
    {
        return std::nullopt;
    }
    sensor->consumed_sequence = sensor->published_sequence;

    // The caller now holds this slot; pick_write_index() skips it until the
    // next latest() moves the lease on.
    sensor->lease_idx = sensor->publish_idx;

    FrameView view;
    view.ptr = reinterpret_cast<uintptr_t>(sensor->buffers[sensor->publish_idx].ptr);
    view.pitch = sensor->buffers[sensor->publish_idx].pitch;
    view.width = sensor->info.width;
    view.height = sensor->info.height;
    view.timestamp_ns = sensor->published_timestamp_ns;
    view.capture_tsc_ns = sensor->published_capture_tsc_ns;
    view.sequence = sensor->published_sequence;
    return view;
}

} // namespace sensing
} // namespace plugins
