// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/video_stage/nvdec_decoder.hpp"

#include <ffnvcodec/dynlink_loader.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace video_stage
{

namespace
{

void check_cu_v(CUresult r, const char* what)
{
    check_cu(r, what);
}

} // namespace

struct NvdecDecoder::Impl
{
    CudaContext& context;
    cudaStream_t stream;
    FrameCallback on_frame;
    FormatCallback on_format_change;

    CuvidFunctions* cvdl = nullptr;
    CUvideoparser parser = nullptr;
    CUvideodecoder decoder = nullptr;
    CUVIDEOFORMAT format{};
    Geometry geometry;
    uint64_t pictures = 0;
    uint64_t access_units = 0;
    // Set by the callbacks so decode() can surface an error thrown inside the parser's C callbacks.
    std::string callback_error;

    Impl(CudaContext& ctx, cudaStream_t s, FrameCallback f, FormatCallback fc)
        : context(ctx), stream(s), on_frame(std::move(f)), on_format_change(std::move(fc))
    {
        if (cuvid_load_functions(&cvdl, nullptr) < 0 || cvdl == nullptr)
            throw std::runtime_error("NvdecDecoder: libnvcuvid is not available");
        context.make_current();
        CUVIDPARSERPARAMS params{};
        params.CodecType = cudaVideoCodec_H264;
        // The parser only needs a minimum here; the sequence callback reports the real requirement.
        params.ulMaxNumDecodeSurfaces = 1;
        params.ulClockRate = 0;
        params.ulErrorThreshold = 100;
        params.ulMaxDisplayDelay = 0; // output as soon as decoded; the stream has no B-frames
        params.pUserData = this;
        params.pfnSequenceCallback = &Impl::sequence_cb;
        params.pfnDecodePicture = &Impl::decode_cb;
        params.pfnDisplayPicture = &Impl::display_cb;
        check_cu_v(cvdl->cuvidCreateVideoParser(&parser, &params), "cuvidCreateVideoParser");
    }

    ~Impl()
    {
        context.make_current();
        if (parser)
            cvdl->cuvidDestroyVideoParser(parser);
        destroy_decoder();
        cuvid_free_functions(&cvdl);
    }

    void destroy_decoder()
    {
        if (decoder)
        {
            cvdl->cuvidDestroyDecoder(decoder);
            decoder = nullptr;
        }
    }

    static int CUDAAPI sequence_cb(void* user, CUVIDEOFORMAT* fmt)
    {
        auto* self = static_cast<Impl*>(user);
        try
        {
            return self->on_sequence(fmt);
        }
        catch (const std::exception& e)
        {
            self->callback_error = e.what();
            return 0; // 0 = failure: the parser stops
        }
    }

    int on_sequence(CUVIDEOFORMAT* fmt)
    {
        if (fmt->codec != cudaVideoCodec_H264 || fmt->chroma_format != cudaVideoChromaFormat_420 ||
            fmt->bit_depth_luma_minus8 != 0)
            throw std::runtime_error("NvdecDecoder: only 8-bit 4:2:0 H.264 is supported");
        const unsigned width = static_cast<unsigned>(fmt->display_area.right - fmt->display_area.left);
        const unsigned height = static_cast<unsigned>(fmt->display_area.bottom - fmt->display_area.top);
        const Geometry geo{ width, height };
        const bool same = decoder != nullptr && format.coded_width == fmt->coded_width &&
                          format.coded_height == fmt->coded_height && geometry == geo &&
                          format.min_num_decode_surfaces == fmt->min_num_decode_surfaces;
        if (same)
            return static_cast<int>(format.min_num_decode_surfaces);

        destroy_decoder();
        CUVIDDECODECREATEINFO info{};
        info.ulWidth = fmt->coded_width;
        info.ulHeight = fmt->coded_height;
        info.ulMaxWidth = fmt->coded_width;
        info.ulMaxHeight = fmt->coded_height;
        // The codec's own minimum plus margin so a consumer holding a surface does not stall the parser.
        info.ulNumDecodeSurfaces = fmt->min_num_decode_surfaces + 2;
        info.CodecType = fmt->codec;
        info.ChromaFormat = fmt->chroma_format;
        info.bitDepthMinus8 = fmt->bit_depth_luma_minus8;
        info.ulCreationFlags = cudaVideoCreate_PreferCUVID;
        info.OutputFormat = cudaVideoSurfaceFormat_NV12;
        info.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;
        info.ulTargetWidth = width;
        info.ulTargetHeight = height;
        info.display_area.left = static_cast<short>(fmt->display_area.left);
        info.display_area.top = static_cast<short>(fmt->display_area.top);
        info.display_area.right = static_cast<short>(fmt->display_area.right);
        info.display_area.bottom = static_cast<short>(fmt->display_area.bottom);
        info.ulNumOutputSurfaces = 3;
        context.make_current();
        check_cu_v(cvdl->cuvidCreateDecoder(&decoder, &info), "cuvidCreateDecoder");
        format = *fmt;
        geometry = geo;
        if (on_format_change)
            on_format_change(geo);
        return static_cast<int>(info.ulNumDecodeSurfaces);
    }

    static int CUDAAPI decode_cb(void* user, CUVIDPICPARAMS* pic)
    {
        auto* self = static_cast<Impl*>(user);
        if (self->cvdl->cuvidDecodePicture(self->decoder, pic) != CUDA_SUCCESS)
        {
            self->callback_error = "cuvidDecodePicture failed";
            return 0;
        }
        return 1;
    }

    static int CUDAAPI display_cb(void* user, CUVIDPARSERDISPINFO* disp)
    {
        auto* self = static_cast<Impl*>(user);
        try
        {
            self->on_display(disp);
            return 1;
        }
        catch (const std::exception& e)
        {
            self->callback_error = e.what();
            return 0;
        }
    }

    void on_display(CUVIDPARSERDISPINFO* disp)
    {
        CUVIDPROCPARAMS proc{};
        proc.progressive_frame = disp->progressive_frame;
        proc.second_field = disp->repeat_first_field + 1;
        proc.top_field_first = disp->top_field_first;
        proc.unpaired_field = disp->repeat_first_field < 0;
        proc.output_stream = stream;
        CUdeviceptr dptr = 0;
        unsigned pitch = 0;
        check_cu_v(cvdl->cuvidMapVideoFrame(decoder, disp->picture_index, &dptr, &pitch, &proc), "cuvidMapVideoFrame");

        DecodedFrame frame;
        frame.key = static_cast<uint64_t>(disp->timestamp);
        frame.view.geometry = geometry;
        frame.view.y = reinterpret_cast<uint8_t*>(dptr);
        // The decoder's NV12 output surface holds the chroma plane right after the luma plane, `target height` rows on.
        frame.view.uv = frame.view.y + static_cast<size_t>(pitch) * geometry.height;
        frame.view.y_pitch = pitch;
        frame.view.uv_pitch = pitch;
        frame.view.color.full_range = format.video_signal_description.video_full_range_flag != 0;
        CuvidFunctions* cvdl_copy = cvdl;
        CUvideodecoder dec = decoder;
        frame.lease = std::shared_ptr<void>(reinterpret_cast<void*>(dptr), [cvdl_copy, dec](void* p)
                                            { cvdl_copy->cuvidUnmapVideoFrame(dec, reinterpret_cast<CUdeviceptr>(p)); });
        ++pictures;
        on_frame(std::move(frame));
    }
};

NvdecDecoder::NvdecDecoder(CudaContext& context, cudaStream_t stream, FrameCallback on_frame, FormatCallback on_format_change)
    : m_impl(std::make_unique<Impl>(context, stream, std::move(on_frame), std::move(on_format_change)))
{
}

NvdecDecoder::~NvdecDecoder() = default;

void NvdecDecoder::decode(const uint8_t* access_unit, size_t size, uint64_t key)
{
    Impl& d = *m_impl;
    d.context.make_current();
    d.callback_error.clear();
    CUVIDSOURCEDATAPACKET packet{};
    packet.payload = access_unit;
    packet.payload_size = size;
    packet.flags = CUVID_PKT_TIMESTAMP | CUVID_PKT_ENDOFPICTURE;
    packet.timestamp = static_cast<CUvideotimestamp>(key);
    ++d.access_units;
    const CUresult r = d.cvdl->cuvidParseVideoData(d.parser, &packet);
    if (!d.callback_error.empty())
        throw std::runtime_error("NvdecDecoder: " + d.callback_error);
    check_cu_v(r, "cuvidParseVideoData");
}

uint64_t NvdecDecoder::pictures() const
{
    return m_impl->pictures;
}

uint64_t NvdecDecoder::access_units() const
{
    return m_impl->access_units;
}

} // namespace video_stage
