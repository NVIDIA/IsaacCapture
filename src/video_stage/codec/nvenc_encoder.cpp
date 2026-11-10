// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/video_stage/nvenc_encoder.hpp"

#include <ffnvcodec/dynlink_loader.h>

#include <cstring>
#include <stdexcept>
#include <string>

namespace video_stage
{

namespace
{

void check_nv(NVENCSTATUS status, const char* what)
{
    if (status != NV_ENC_SUCCESS)
        throw std::runtime_error(std::string("NvencEncoder: ") + what + " failed with status " + std::to_string(status));
}

} // namespace

struct NvencEncoder::Impl
{
    CudaContext& context;
    EncoderConfig config;

    NvencFunctions* nvenc = nullptr;
    NV_ENCODE_API_FUNCTION_LIST fn{};
    void* encoder = nullptr;
    NV_ENC_OUTPUT_PTR bitstream = nullptr;

    struct Slot
    {
        CUdeviceptr ptr = 0;
        size_t pitch = 0;
        NV_ENC_REGISTERED_PTR registered = nullptr;
        bool in_use = false;
    };
    std::vector<Slot> slots;

    Impl(CudaContext& ctx, const EncoderConfig& cfg) : context(ctx), config(cfg)
    {
        if (config.geometry.width == 0 || config.geometry.height == 0 || (config.geometry.width % 2) != 0 ||
            (config.geometry.height % 2) != 0)
            throw std::invalid_argument("NvencEncoder: NV12 needs non-zero even dimensions");
        try
        {
            context.make_current();
            if (nvenc_load_functions(&nvenc, nullptr) < 0 || nvenc == nullptr)
                throw std::runtime_error("NvencEncoder: libnvidia-encode is not available");
            fn.version = NV_ENCODE_API_FUNCTION_LIST_VER;
            check_nv(nvenc->NvEncodeAPICreateInstance(&fn), "NvEncodeAPICreateInstance");

            NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS open{};
            open.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
            open.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
            open.device = context.handle();
            open.apiVersion = NVENCAPI_VERSION;
            check_nv(fn.nvEncOpenEncodeSessionEx(&open, &encoder), "nvEncOpenEncodeSessionEx");

            configure();
            allocate_surfaces();

            NV_ENC_CREATE_BITSTREAM_BUFFER out{};
            out.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
            check_nv(fn.nvEncCreateBitstreamBuffer(encoder, &out), "nvEncCreateBitstreamBuffer");
            bitstream = out.bitstreamBuffer;
        }
        catch (...)
        {
            // A constructor that throws does not run the destructor, so a failure part way releases what exists.
            shutdown();
            throw;
        }
    }

    ~Impl()
    {
        shutdown();
    }

    // Releases the session, the registered surfaces and the loaded library; safe to call on a partly built object.
    void shutdown() noexcept
    {
        context.make_current_noexcept();
        if (encoder)
        {
            for (Slot& s : slots)
            {
                if (s.registered)
                    fn.nvEncUnregisterResource(encoder, s.registered);
                if (s.ptr)
                    cuMemFree(s.ptr);
                s = Slot{};
            }
            if (bitstream)
                fn.nvEncDestroyBitstreamBuffer(encoder, bitstream);
            fn.nvEncDestroyEncoder(encoder);
            encoder = nullptr;
            bitstream = nullptr;
        }
        if (nvenc)
            nvenc_free_functions(&nvenc);
    }

    void configure()
    {
        const GUID codec = NV_ENC_CODEC_H264_GUID;
        const GUID preset = NV_ENC_PRESET_P1_GUID;
        const NV_ENC_TUNING_INFO tuning = NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY;

        NV_ENC_PRESET_CONFIG preset_config{};
        preset_config.version = NV_ENC_PRESET_CONFIG_VER;
        preset_config.presetCfg.version = NV_ENC_CONFIG_VER;
        check_nv(fn.nvEncGetEncodePresetConfigEx(encoder, codec, preset, tuning, &preset_config),
                 "nvEncGetEncodePresetConfigEx");

        NV_ENC_CONFIG enc_config = preset_config.presetCfg;
        enc_config.version = NV_ENC_CONFIG_VER;
        enc_config.gopLength = config.idr_period;
        enc_config.frameIntervalP = 1; // no B-frames
        enc_config.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR;
        enc_config.rcParams.averageBitRate = config.bitrate_bps;
        enc_config.rcParams.maxBitRate = config.bitrate_bps;
        // A buffer of one frame's worth of bits keeps the output close to the instantaneous rate.
        const uint32_t frame_bits = static_cast<uint32_t>(static_cast<uint64_t>(config.bitrate_bps) *
                                                          config.fps_denominator / config.fps_numerator);
        enc_config.rcParams.vbvBufferSize = frame_bits;
        enc_config.rcParams.vbvInitialDelay = frame_bits;
        enc_config.rcParams.lookaheadDepth = 0;
        NV_ENC_CONFIG_H264& h264 = enc_config.encodeCodecConfig.h264Config;
        h264.idrPeriod = config.idr_period;
        h264.repeatSPSPPS = 1;
        h264.chromaFormatIDC = 1;
        h264.h264VUIParameters.videoSignalTypePresentFlag = 1;
        h264.h264VUIParameters.videoFormat = NV_ENC_VUI_VIDEO_FORMAT_UNSPECIFIED;
        h264.h264VUIParameters.videoFullRangeFlag = config.full_range ? 1 : 0;

        NV_ENC_INITIALIZE_PARAMS init{};
        init.version = NV_ENC_INITIALIZE_PARAMS_VER;
        init.encodeGUID = codec;
        init.presetGUID = preset;
        init.tuningInfo = tuning;
        init.encodeWidth = config.geometry.width;
        init.encodeHeight = config.geometry.height;
        init.darWidth = config.geometry.width;
        init.darHeight = config.geometry.height;
        init.maxEncodeWidth = config.geometry.width;
        init.maxEncodeHeight = config.geometry.height;
        init.frameRateNum = config.fps_numerator;
        init.frameRateDen = config.fps_denominator;
        init.enablePTD = 1;
        init.encodeConfig = &enc_config;
        check_nv(fn.nvEncInitializeEncoder(encoder, &init), "nvEncInitializeEncoder");
    }

    void allocate_surfaces()
    {
        slots.resize(config.input_surfaces);
        const size_t rows = static_cast<size_t>(config.geometry.height) * 3 / 2;
        for (Slot& s : slots)
        {
            // One allocation: luma rows then chroma rows, the layout NVENC reads for NV12.
            check_cu(cuMemAllocPitch(&s.ptr, &s.pitch, config.geometry.width, rows, 16), "cuMemAllocPitch");
            NV_ENC_REGISTER_RESOURCE reg{};
            reg.version = NV_ENC_REGISTER_RESOURCE_VER;
            reg.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR;
            reg.width = config.geometry.width;
            reg.height = config.geometry.height;
            reg.pitch = static_cast<uint32_t>(s.pitch);
            reg.resourceToRegister = reinterpret_cast<void*>(s.ptr);
            reg.bufferFormat = NV_ENC_BUFFER_FORMAT_NV12;
            reg.bufferUsage = NV_ENC_INPUT_IMAGE;
            check_nv(fn.nvEncRegisterResource(encoder, &reg), "nvEncRegisterResource");
            s.registered = reg.registeredResource;
        }
    }
};

NvencEncoder::NvencEncoder(CudaContext& context, const EncoderConfig& config)
    : m_impl(std::make_unique<Impl>(context, config))
{
}

NvencEncoder::~NvencEncoder() = default;

void NvencEncoder::release(const Surface& surface)
{
    m_impl->slots.at(surface.slot).in_use = false;
}

bool NvencEncoder::acquire(Surface& out)
{
    Impl& e = *m_impl;
    for (uint32_t i = 0; i < e.slots.size(); ++i)
    {
        Impl::Slot& s = e.slots[i];
        if (s.in_use)
            continue;
        s.in_use = true;
        out.slot = i;
        out.view.geometry = e.config.geometry;
        out.view.color.full_range = e.config.full_range;
        out.view.y = reinterpret_cast<uint8_t*>(s.ptr);
        out.view.y_pitch = s.pitch;
        out.view.uv = reinterpret_cast<uint8_t*>(s.ptr) + s.pitch * e.config.geometry.height;
        out.view.uv_pitch = s.pitch;
        return true;
    }
    return false;
}

EncodedPacket NvencEncoder::encode(const Surface& surface, cudaStream_t stream, uint64_t key, bool force_idr)
{
    Impl& e = *m_impl;
    Impl::Slot& slot = e.slots.at(surface.slot);
    EncodedPacket packet;
    try
    {
        e.context.make_current();
        // NVENC reads the surface directly; wait for this stream's work (the warp) to finish writing it.
        if (cudaStreamSynchronize(stream) != cudaSuccess)
            throw std::runtime_error("NvencEncoder: the GPU work feeding the encoder failed");

        NV_ENC_MAP_INPUT_RESOURCE map{};
        map.version = NV_ENC_MAP_INPUT_RESOURCE_VER;
        map.registeredResource = slot.registered;
        check_nv(e.fn.nvEncMapInputResource(e.encoder, &map), "nvEncMapInputResource");
        // Unmap on every exit path: a registered surface that stays mapped cannot be reused or freed.
        struct MapGuard
        {
            Impl& impl;
            NV_ENC_INPUT_PTR mapped;
            ~MapGuard()
            {
                impl.fn.nvEncUnmapInputResource(impl.encoder, mapped);
            }
        } map_guard{ e, map.mappedResource };

        NV_ENC_PIC_PARAMS pic{};
        pic.version = NV_ENC_PIC_PARAMS_VER;
        pic.inputBuffer = map.mappedResource;
        pic.bufferFmt = map.mappedBufferFmt;
        pic.inputWidth = e.config.geometry.width;
        pic.inputHeight = e.config.geometry.height;
        pic.inputPitch = static_cast<uint32_t>(slot.pitch);
        pic.outputBitstream = e.bitstream;
        pic.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
        pic.inputTimeStamp = key;
        if (force_idr)
            pic.encodePicFlags = NV_ENC_PIC_FLAG_FORCEIDR | NV_ENC_PIC_FLAG_OUTPUT_SPSPPS;
        check_nv(e.fn.nvEncEncodePicture(e.encoder, &pic), "nvEncEncodePicture");

        NV_ENC_LOCK_BITSTREAM lock{};
        lock.version = NV_ENC_LOCK_BITSTREAM_VER;
        lock.outputBitstream = e.bitstream;
        check_nv(e.fn.nvEncLockBitstream(e.encoder, &lock), "nvEncLockBitstream");
        struct LockGuard
        {
            Impl& impl;
            ~LockGuard()
            {
                impl.fn.nvEncUnlockBitstream(impl.encoder, impl.bitstream);
            }
        } lock_guard{ e };
        const auto* bytes = static_cast<const uint8_t*>(lock.bitstreamBufferPtr);
        packet.data.assign(bytes, bytes + lock.bitstreamSizeInBytes);
        packet.key = lock.outputTimeStamp;
        packet.keyframe = lock.pictureType == NV_ENC_PIC_TYPE_IDR;
    }
    catch (...)
    {
        slot.in_use = false;
        throw;
    }
    slot.in_use = false;
    return packet;
}

} // namespace video_stage
