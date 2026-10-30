// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "h264_decoder.hpp"

#include "NvDecoder/NvDecoder.h"
#include "nv12_to_rgba.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <sstream>
#include <stdexcept>

namespace camera_viz::codec
{

namespace
{

inline void check_cu(CUresult result, const char* what)
{
    if (result != CUDA_SUCCESS)
    {
        const char* err = nullptr;
        cuGetErrorString(result, &err);
        std::ostringstream os;
        os << "H264Decoder: " << what << " failed: " << (err ? err : "unknown");
        throw std::runtime_error(os.str());
    }
}

inline void check_rt(cudaError_t result, const char* what)
{
    if (result != cudaSuccess)
    {
        std::ostringstream os;
        os << "H264Decoder: " << what << " failed: " << cudaGetErrorString(result);
        throw std::runtime_error(os.str());
    }
}

} // namespace

struct H264Decoder::Impl
{
    DecoderConfig cfg;

    CUdevice cu_device = 0;
    CUcontext cu_context = nullptr;
    cudaStream_t stream = nullptr;
    std::unique_ptr<NvDecoder> decoder;
    bool warned_size_mismatch = false;

    Impl(const DecoderConfig& c) : cfg(c)
    {
        if (cfg.width == 0 || cfg.height == 0)
        {
            throw std::runtime_error("H264Decoder: width/height must be > 0");
        }
        check_cu(cuInit(0), "cuInit");
        check_cu(cuDeviceGet(&cu_device, cfg.gpu_id), "cuDeviceGet");
        check_cu(cuDevicePrimaryCtxRetain(&cu_context, cu_device), "cuDevicePrimaryCtxRetain");

        check_cu(cuCtxPushCurrent(cu_context), "cuCtxPushCurrent");
        try
        {
            check_rt(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
            // Zero-latency, decode-order output. Output stays on GPU.
            decoder = std::make_unique<NvDecoder>(cu_context,
                                                  true, // bUseDeviceFrame
                                                  cudaVideoCodec_H264,
                                                  true, // bLowLatency
                                                  false, // bDeviceFramePitched
                                                  nullptr, // pCropRect
                                                  nullptr, // pResizeDim
                                                  false, // bExtractSEIMessage
                                                  0, 0, // max width/height (auto)
                                                  1000, // nClockRate
                                                  true); // bForceZeroLatency
        }
        catch (...)
        {
            if (stream)
            {
                cudaStreamDestroy(stream);
                stream = nullptr;
            }
            cuCtxPopCurrent(nullptr);
            if (cu_context)
            {
                cuDevicePrimaryCtxRelease(cu_device);
                cu_context = nullptr;
            }
            throw;
        }
        check_cu(cuCtxPopCurrent(nullptr), "cuCtxPopCurrent");
    }

    ~Impl()
    {
        decoder.reset();
        if (stream)
        {
            cudaStreamDestroy(stream);
            stream = nullptr;
        }
        if (cu_context)
        {
            cuDevicePrimaryCtxRelease(cu_device);
            cu_context = nullptr;
        }
    }

    bool decode_one(const uint8_t* packet, std::size_t packet_size, uintptr_t rgba_out, std::size_t rgba_row_bytes)
    {
        if (!decoder)
        {
            throw std::runtime_error("H264Decoder: decoder unavailable");
        }

        check_cu(cuCtxPushCurrent(cu_context), "cuCtxPushCurrent");
        int n_frames = 0;
        try
        {
            n_frames = decoder->Decode(packet, static_cast<int>(packet_size));
        }
        catch (...)
        {
            cuCtxPopCurrent(nullptr);
            throw;
        }
        if (n_frames == 0)
        {
            check_cu(cuCtxPopCurrent(nullptr), "cuCtxPopCurrent");
            return false;
        }

        uint8_t* nv12 = decoder->GetLockedFrame();
        if (!nv12)
        {
            check_cu(cuCtxPopCurrent(nullptr), "cuCtxPopCurrent");
            return false;
        }

        const int w = decoder->GetWidth();
        const int h = decoder->GetHeight();
        const int pitch = decoder->GetDeviceFramePitch();
        const int luma_size = decoder->GetLumaPlaneSize();

        if (w != static_cast<int>(cfg.width) || h != static_cast<int>(cfg.height))
        {
            // Caller's RGBA buffer is fixed at construction; drop frames
            // that don't match. Log once.
            if (!warned_size_mismatch)
            {
                std::ostringstream os;
                os << "H264Decoder: stream is " << w << "x" << h << " but configured for " << cfg.width << "x"
                   << cfg.height << "; dropping frames";
                std::fputs(os.str().c_str(), stderr);
                std::fputc('\n', stderr);
                warned_size_mismatch = true;
            }
            decoder->UnlockFrame(&nv12);
            check_cu(cuCtxPopCurrent(nullptr), "cuCtxPopCurrent");
            return false;
        }

        launch_nv12_to_rgba(nv12, nv12 + luma_size, pitch, pitch, w, h, reinterpret_cast<uint8_t*>(rgba_out),
                            static_cast<int>(rgba_row_bytes), cfg.full_range, stream);
        // UnlockFrame returns the surface to NVDEC's pool, so the
        // kernel reading it must finish first.
        const cudaError_t sync_err = cudaStreamSynchronize(stream);
        decoder->UnlockFrame(&nv12);
        check_cu(cuCtxPopCurrent(nullptr), "cuCtxPopCurrent");
        if (sync_err != cudaSuccess)
        {
            std::ostringstream os;
            os << "H264Decoder: NV12→RGBA kernel sync failed: " << cudaGetErrorString(sync_err);
            throw std::runtime_error(os.str());
        }
        return true;
    }

    void reset_decoder()
    {
        if (!decoder)
            return;
        check_cu(cuCtxPushCurrent(cu_context), "cuCtxPushCurrent");
        decoder.reset();
        try
        {
            decoder = std::make_unique<NvDecoder>(
                cu_context, true, cudaVideoCodec_H264, true, false, nullptr, nullptr, false, 0, 0, 1000, true);
        }
        catch (...)
        {
            cuCtxPopCurrent(nullptr);
            throw;
        }
        check_cu(cuCtxPopCurrent(nullptr), "cuCtxPopCurrent");
        warned_size_mismatch = false;
    }
};

H264Decoder::H264Decoder(const DecoderConfig& cfg) : impl_(std::make_unique<Impl>(cfg))
{
}

H264Decoder::~H264Decoder() = default;

bool H264Decoder::decode(const uint8_t* packet,
                         std::size_t packet_size,
                         uintptr_t rgba_out_device_ptr,
                         std::size_t row_pitch_bytes)
{
    return impl_->decode_one(packet, packet_size, rgba_out_device_ptr, row_pitch_bytes);
}

void H264Decoder::reset()
{
    impl_->reset_decoder();
}

} // namespace camera_viz::codec
