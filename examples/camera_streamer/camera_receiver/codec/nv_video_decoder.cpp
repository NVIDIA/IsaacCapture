// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nv_video_decoder.hpp"

#include "NvDecoder/NvDecoder.h"
#include "nv12_to_rgba.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <mutex>
#include <sstream>
#include <stdexcept>

namespace camera_receiver::codec
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
        os << "NvVideoDecoder: " << what << " failed: " << (err ? err : "unknown");
        throw std::runtime_error(os.str());
    }
}

inline void check_rt(cudaError_t result, const char* what)
{
    if (result != cudaSuccess)
    {
        std::ostringstream os;
        os << "NvVideoDecoder: " << what << " failed: " << cudaGetErrorString(result);
        throw std::runtime_error(os.str());
    }
}

inline cudaVideoCodec to_cuvid_codec(Codec c)
{
    return (c == Codec::HEVC) ? cudaVideoCodec_HEVC : cudaVideoCodec_H264;
}

} // namespace

struct NvVideoDecoder::Impl
{
    DecoderConfig cfg;

    CUdevice cu_device = 0;
    CUcontext cu_context = nullptr;
    cudaStream_t stream = nullptr;
    std::unique_ptr<NvDecoder> decoder;
    std::mutex decoder_mutex;
    bool warned_size_mismatch = false;

    Impl(const DecoderConfig& c) : cfg(c)
    {
        if (cfg.width == 0 || cfg.height == 0)
        {
            throw std::runtime_error("NvVideoDecoder: width/height must be > 0");
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
                                                  to_cuvid_codec(cfg.codec),
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

    bool decode_one(
        const uint8_t* packet, std::size_t packet_size, uintptr_t rgba_out, std::size_t rgba_row_bytes, int rotation)
    {
        std::lock_guard<std::mutex> lock(decoder_mutex);
        if (!decoder)
        {
            throw std::runtime_error("NvVideoDecoder: decoder unavailable");
        }

        if (rotation != 0 && rotation != 180)
        {
            throw std::runtime_error("NvVideoDecoder: rotation must be 0 or 180");
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
                os << "NvVideoDecoder: stream is " << w << "x" << h << " but configured for " << cfg.width << "x"
                   << cfg.height << "; dropping frames";
                std::fputs(os.str().c_str(), stderr);
                std::fputc('\n', stderr);
                warned_size_mismatch = true;
            }
            decoder->UnlockFrame(&nv12);
            check_cu(cuCtxPopCurrent(nullptr), "cuCtxPopCurrent");
            return false;
        }

        cudaError_t sync_err = cudaSuccess;
        try
        {
            launch_nv12_to_rgba(nv12, nv12 + luma_size, pitch, pitch, w, h, reinterpret_cast<uint8_t*>(rgba_out),
                                static_cast<int>(rgba_row_bytes), cfg.full_range, rotation, stream);
            // UnlockFrame returns the surface to NVDEC's pool, so the
            // kernel reading it must finish first.
            sync_err = cudaStreamSynchronize(stream);
        }
        catch (...)
        {
            // Complete any submitted reads before returning the surface to NVDEC.
            cudaStreamSynchronize(stream);
            decoder->UnlockFrame(&nv12);
            cuCtxPopCurrent(nullptr);
            throw;
        }
        decoder->UnlockFrame(&nv12);
        check_cu(cuCtxPopCurrent(nullptr), "cuCtxPopCurrent");
        if (sync_err != cudaSuccess)
        {
            std::ostringstream os;
            os << "NvVideoDecoder: NV12→RGBA kernel sync failed: " << cudaGetErrorString(sync_err);
            throw std::runtime_error(os.str());
        }
        return true;
    }

    bool decode_one_nv12(
        const uint8_t* packet, std::size_t packet_size, uintptr_t nv12_out, std::size_t nv12_row_bytes, int rotation)
    {
        std::lock_guard<std::mutex> lock(decoder_mutex);
        if (!decoder)
        {
            throw std::runtime_error("NvVideoDecoder: decoder unavailable");
        }
        if (nv12_row_bytes < cfg.width)
        {
            throw std::runtime_error("NvVideoDecoder: NV12 row stride smaller than width");
        }
        if (rotation != 0 && rotation != 180)
        {
            throw std::runtime_error("NvVideoDecoder: rotation must be 0 or 180");
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
        const int src_pitch = decoder->GetDeviceFramePitch();
        const int luma_size = decoder->GetLumaPlaneSize();

        if (w != static_cast<int>(cfg.width) || h != static_cast<int>(cfg.height))
        {
            if (!warned_size_mismatch)
            {
                std::ostringstream os;
                os << "NvVideoDecoder: stream is " << w << "x" << h << " but configured for " << cfg.width << "x"
                   << cfg.height << "; dropping frames";
                std::fputs(os.str().c_str(), stderr);
                std::fputc('\n', stderr);
                warned_size_mismatch = true;
            }
            decoder->UnlockFrame(&nv12);
            check_cu(cuCtxPopCurrent(nullptr), "cuCtxPopCurrent");
            return false;
        }

        cudaError_t sync_err = cudaSuccess;
        try
        {
            auto* dst_y = reinterpret_cast<uint8_t*>(nv12_out);
            auto* dst_uv = dst_y + nv12_row_bytes * static_cast<std::size_t>(h);
            if (rotation == 180)
            {
                launch_nv12_rotate_180(nv12, nv12 + luma_size, src_pitch, src_pitch, w, h, dst_y,
                                       static_cast<int>(nv12_row_bytes), stream);
            }
            else
            {
                check_rt(
                    cudaMemcpy2DAsync(dst_y, nv12_row_bytes, nv12, src_pitch, w, h, cudaMemcpyDeviceToDevice, stream),
                    "cudaMemcpy2DAsync NV12 Y");
                check_rt(cudaMemcpy2DAsync(dst_uv, nv12_row_bytes, nv12 + luma_size, src_pitch, w, h / 2,
                                           cudaMemcpyDeviceToDevice, stream),
                         "cudaMemcpy2DAsync NV12 UV");
            }
            sync_err = cudaStreamSynchronize(stream);
        }
        catch (...)
        {
            // The Y copy may already be queued when the UV copy fails.
            cudaStreamSynchronize(stream);
            decoder->UnlockFrame(&nv12);
            cuCtxPopCurrent(nullptr);
            throw;
        }
        decoder->UnlockFrame(&nv12);
        check_cu(cuCtxPopCurrent(nullptr), "cuCtxPopCurrent");
        if (sync_err != cudaSuccess)
        {
            std::ostringstream os;
            os << "NvVideoDecoder: NV12 copy sync failed: " << cudaGetErrorString(sync_err);
            throw std::runtime_error(os.str());
        }
        return true;
    }

    void reset_decoder()
    {
        std::lock_guard<std::mutex> lock(decoder_mutex);
        check_cu(cuCtxPushCurrent(cu_context), "cuCtxPushCurrent");
        decoder.reset();
        try
        {
            decoder = std::make_unique<NvDecoder>(
                cu_context, true, to_cuvid_codec(cfg.codec), true, false, nullptr, nullptr, false, 0, 0, 1000, true);
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

NvVideoDecoder::NvVideoDecoder(const DecoderConfig& cfg) : impl_(std::make_unique<Impl>(cfg))
{
}

NvVideoDecoder::~NvVideoDecoder() = default;

bool NvVideoDecoder::decode(const uint8_t* packet,
                            std::size_t packet_size,
                            uintptr_t rgba_out_device_ptr,
                            std::size_t row_pitch_bytes,
                            int rotation)
{
    return impl_->decode_one(packet, packet_size, rgba_out_device_ptr, row_pitch_bytes, rotation);
}

bool NvVideoDecoder::decode_nv12(const uint8_t* packet,
                                 std::size_t packet_size,
                                 uintptr_t nv12_out_device_ptr,
                                 std::size_t row_pitch_bytes,
                                 int rotation)
{
    return impl_->decode_one_nv12(packet, packet_size, nv12_out_device_ptr, row_pitch_bytes, rotation);
}

void NvVideoDecoder::reset()
{
    impl_->reset_decoder();
}

} // namespace camera_receiver::codec
