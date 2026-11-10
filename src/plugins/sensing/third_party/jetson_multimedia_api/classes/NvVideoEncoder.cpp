/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "NvVideoEncoder.h"
#include "NvLogging.h"

#include <cstring>
#include <errno.h>
#include <libv4l2.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "nvbufsurface.h"

#define ENCODER_DEV "/dev/nvhost-msenc"
#define ENCODER_DEV_ALT "/dev/v4l2-nvenc"
#define ENCODER_GPU_DEV "/dev/nvidia0"

#define ENCODER_COMP_NAME "NVENC"

#define CHECK_V4L2_RETURN(ret, str)              \
    if (ret < 0) {                               \
        COMP_SYS_ERROR_MSG(str << ": failed");   \
        return -1;                               \
    } else {                                     \
        COMP_DEBUG_MSG(str << ": success");      \
        return 0;                                \
    }

#define RETURN_ERROR_IF_FORMATS_SET() \
    if (output_plane_pixfmt != 0 || capture_plane_pixfmt != 0) { \
        COMP_ERROR_MSG("Should be called before setting plane formats") \
        return -1; \
    }

#define RETURN_ERROR_IF_OUTPUT_PLANE_FORMAT_NOT_SET() \
if (output_plane_pixfmt != 0) { \
    COMP_ERROR_MSG("Should be called before setting output plane format") \
    return -1; \
}

#define RETURN_ERROR_IF_BUFFERS_REQUESTED() \
    if (output_plane.getNumBuffers() != 0 && capture_plane.getNumBuffers() != 0) { \
        COMP_ERROR_MSG("Should be called before requesting buffers on either plane") \
        return -1; \
    }

#define RETURN_ERROR_IF_BUFFERS_NOT_REQUESTED() \
    if (output_plane.getNumBuffers() == 0 || capture_plane.getNumBuffers() == 0) { \
        COMP_ERROR_MSG("Should be called before requesting buffers on either plane") \
        return -1; \
    }

#define RETURN_ERROR_IF_FORMATS_NOT_SET() \
    if (output_plane_pixfmt == 0 || capture_plane_pixfmt == 0) { \
        COMP_ERROR_MSG("Should be called after setting plane formats") \
        return -1; \
    }

using namespace std;

NvVideoEncoder::NvVideoEncoder(const char *name, const char *dev_node, int flags)
    : NvV4l2Element(name, dev_node, flags, valid_fields)
{
    gpu_enabled = false;
}

NvVideoEncoder *
NvVideoEncoder::createVideoEncoder(const char *name, int flags)
{
    NvVideoEncoder *enc;
    NvBufSurfaceDeviceInfo deviceInfo;
    memset(&deviceInfo, 0, sizeof(deviceInfo));
    NvBufSurfaceGetDeviceInfo(&deviceInfo);
    if (deviceInfo.driverType == NVBUF_DRIVER_TYPE_NVGPU && deviceInfo.isIntegratedGpu)
    {
        if (access (ENCODER_DEV, F_OK) == 0)
        {
            enc = new NvVideoEncoder(name, ENCODER_DEV, flags);
        }
        else if (access(ENCODER_DEV_ALT, F_OK) == 0)
        {
            enc = new NvVideoEncoder(name, ENCODER_DEV_ALT, flags);
        }
        else
        {
            return NULL;
        }
    }
    else
    {
        if (access(ENCODER_GPU_DEV, F_OK) == 0)
        {
            enc = new NvVideoEncoder(name, ENCODER_GPU_DEV, flags);
            enc->gpu_enabled = true;
        }
        else
            return NULL;
    }

    if (enc->isInError())
    {
        delete enc;
        return NULL;
    }
    return enc;
}

NvVideoEncoder::~NvVideoEncoder()
{
}

bool
NvVideoEncoder::isGPUEnabled()
{
    return this->gpu_enabled;
}

int NvVideoEncoder::setOutputPlaneFormat(uint32_t pixfmt, uint32_t width,
                                         uint32_t height, v4l2_colorspace cs)
{
    struct v4l2_format format;
    uint32_t num_bufferplanes;
    NvBuffer::NvBufferPlaneFormat planefmts[MAX_PLANES] = {{0}};

    if (pixfmt != V4L2_PIX_FMT_YUV420M &&
        pixfmt != V4L2_PIX_FMT_P010M && pixfmt != V4L2_PIX_FMT_NV12M &&
        pixfmt != V4L2_PIX_FMT_NV24M && pixfmt != V4L2_PIX_FMT_NV24_10LE &&
        pixfmt != V4L2_PIX_FMT_YUV444M && pixfmt != V4L2_PIX_FMT_YUV444_10LE)
    {
        COMP_ERROR_MSG("Only YUV420M, NV24, NV24_10LE, YUV444, P010M and NV12M are supported");
        return -1;
    }

    output_plane_pixfmt = pixfmt;
    NvBuffer::fill_buffer_plane_format(&num_bufferplanes, planefmts, width,
            height, pixfmt);
    output_plane.setBufferPlaneFormat(num_bufferplanes, planefmts);

    memset(&format, 0, sizeof(struct v4l2_format));
    format.type = output_plane.getBufType();
    format.fmt.pix_mp.width = width;
    format.fmt.pix_mp.height = height;
    format.fmt.pix_mp.pixelformat = pixfmt;
    format.fmt.pix_mp.num_planes = num_bufferplanes;
    format.fmt.pix_mp.colorspace = cs;

    return output_plane.setFormat(format);
}

int NvVideoEncoder::setOutputPlaneFormat(uint32_t pixfmt, uint32_t width,
                                         uint32_t height, v4l2_colorspace cs,
                                         bool extended)
{
    struct v4l2_format format;
    uint32_t num_bufferplanes;
    NvBuffer::NvBufferPlaneFormat planefmts[MAX_PLANES];

    if (pixfmt != V4L2_PIX_FMT_YUV420M &&
        pixfmt != V4L2_PIX_FMT_P010M && pixfmt != V4L2_PIX_FMT_NV12M &&
        pixfmt != V4L2_PIX_FMT_NV24M && pixfmt != V4L2_PIX_FMT_NV24_10LE &&
        pixfmt != V4L2_PIX_FMT_YUV444M && pixfmt != V4L2_PIX_FMT_YUV444_10LE)
    {
        COMP_ERROR_MSG("Only YUV420M, NV24, NV24_10LE, YUV444, P010M and NV12M are supported");
        return -1;
    }

    output_plane_pixfmt = pixfmt;
    NvBuffer::fill_buffer_plane_format(&num_bufferplanes, planefmts, width,
                                       height, pixfmt);
    output_plane.setBufferPlaneFormat(num_bufferplanes, planefmts);

    memset(&format, 0, sizeof(struct v4l2_format));
    format.type = output_plane.getBufType();
    format.fmt.pix_mp.width = width;
    format.fmt.pix_mp.height = height;
    format.fmt.pix_mp.pixelformat = pixfmt;
    format.fmt.pix_mp.num_planes = num_bufferplanes;

    /* Extended color format maps to full color range; otherwise signal the
     * limited (studio) range explicitly. */
    format.fmt.pix_mp.quantization =
        extended ? V4L2_QUANTIZATION_FULL_RANGE : V4L2_QUANTIZATION_LIM_RANGE;

    format.fmt.pix_mp.colorspace = cs;

    /* Program the matrix coefficients (ycbcr_enc) and transfer function
     * (xfer_func) to match the colorspace. V4L2 has no dedicated constant for
     * the BT.2020 10-/12-bit transfer function, and BT.601 shares the BT.709
     * OETF, so for those colorspaces the transfer characteristic is left as
     * DEFAULT and derived from the colorspace by the encoder backend (this
     * lets BT.601 be signalled as SMPTE 170M rather than BT.709).
     * https://www.kernel.org/doc/html/v6.8/userspace-api/media/v4l/colorspaces-defs.html */
    switch (format.fmt.pix_mp.colorspace)
    {
    case V4L2_COLORSPACE_REC709:
        format.fmt.pix_mp.ycbcr_enc = V4L2_YCBCR_ENC_709;
        format.fmt.pix_mp.xfer_func = V4L2_XFER_FUNC_709;
        break;
    case V4L2_COLORSPACE_BT2020:
        format.fmt.pix_mp.ycbcr_enc = V4L2_YCBCR_ENC_BT2020;
        format.fmt.pix_mp.xfer_func = V4L2_XFER_FUNC_DEFAULT;
        break;
    case V4L2_COLORSPACE_SMPTE170M:
        format.fmt.pix_mp.ycbcr_enc = V4L2_YCBCR_ENC_601;
        format.fmt.pix_mp.xfer_func = V4L2_XFER_FUNC_DEFAULT;
        break;
    default:
        format.fmt.pix_mp.ycbcr_enc = V4L2_YCBCR_ENC_DEFAULT;
        format.fmt.pix_mp.xfer_func = V4L2_XFER_FUNC_DEFAULT;
        break;
    }

    return output_plane.setFormat(format);
}

int
NvVideoEncoder::setCapturePlaneFormat(uint32_t pixfmt, uint32_t width,
        uint32_t height, uint32_t sizeimage)
{
    struct v4l2_format format;

    memset(&format, 0, sizeof(struct v4l2_format));
    format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    switch (pixfmt)
    {
        case V4L2_PIX_FMT_H264:
        case V4L2_PIX_FMT_H265:
        case V4L2_PIX_FMT_VP8:
        case V4L2_PIX_FMT_VP9:
        case V4L2_PIX_FMT_AV1:
            capture_plane_pixfmt = pixfmt;
            break;
        default:
            ERROR_MSG("Unknown supported pixel format for encoder " << pixfmt);
            return -1;
    }

    format.fmt.pix_mp.pixelformat = pixfmt;
    format.fmt.pix_mp.width = width;
    format.fmt.pix_mp.height = height;
    format.fmt.pix_mp.num_planes = 1;
    format.fmt.pix_mp.plane_fmt[0].sizeimage = sizeimage;

    return capture_plane.setFormat(format);
}

int
NvVideoEncoder::setFrameRate(uint32_t framerate_num, uint32_t framerate_den)
{
    struct v4l2_streamparm parms;

    RETURN_ERROR_IF_FORMATS_NOT_SET();

    memset(&parms, 0, sizeof(parms));
    parms.parm.output.timeperframe.numerator = framerate_den;
    parms.parm.output.timeperframe.denominator = framerate_num;
    parms.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;

    CHECK_V4L2_RETURN(output_plane.setStreamParms(parms),
            "Setting framerate to " << framerate_num << "/" << framerate_den);
}

int
NvVideoEncoder::setBitrate(uint32_t bitrate)
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    // Local change: the plane pixel-format debug prints are removed (see ../README.md).
    RETURN_ERROR_IF_FORMATS_NOT_SET();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEO_BITRATE;
    control.value = bitrate;

    CHECK_V4L2_RETURN(setExtControls(ctrls),
            "Setting encoder bitrate to " << bitrate);
}

int
NvVideoEncoder::setEncoderCommand(int cmd, int flags)
{
   int ret=0;
   struct v4l2_encoder_cmd v4l2_enc_cmd;
   v4l2_enc_cmd.cmd = cmd;
   v4l2_enc_cmd.flags = flags;

   ret = v4l2_ioctl(fd, VIDIOC_ENCODER_CMD, v4l2_enc_cmd);
   if (ret < 0)
     printf(" Error in encoder command \n");

  return ret;
}

int
NvVideoEncoder::setPeakBitrate(uint32_t peak_bitrate)
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    // Local change: the plane pixel-format debug prints are removed (see ../README.md).
    RETURN_ERROR_IF_FORMATS_NOT_SET();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEO_BITRATE_PEAK;
    control.value = peak_bitrate;

    CHECK_V4L2_RETURN(setExtControls(ctrls),
            "Setting encoder peak bitrate to " << peak_bitrate);
}

int
NvVideoEncoder::setProfile(uint32_t profile)
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    RETURN_ERROR_IF_FORMATS_NOT_SET();
    RETURN_ERROR_IF_BUFFERS_REQUESTED();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    switch (capture_plane_pixfmt)
    {
        case V4L2_PIX_FMT_H264:
            control.id = V4L2_CID_MPEG_VIDEO_H264_PROFILE;
            break;
        case V4L2_PIX_FMT_H265:
            control.id = V4L2_CID_MPEG_VIDEO_H265_PROFILE;
            break;
        default:
            COMP_ERROR_MSG("Unsupported encoder type");
            return -1;
    }
    control.value = profile;

    CHECK_V4L2_RETURN(setExtControls(ctrls),
            "Setting encoder profile to " << profile);
}

int
NvVideoEncoder::setLevel(uint32_t level)
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    RETURN_ERROR_IF_FORMATS_NOT_SET();
    RETURN_ERROR_IF_BUFFERS_REQUESTED();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    switch (capture_plane_pixfmt)
    {
        case V4L2_PIX_FMT_H264:
            control.id = V4L2_CID_MPEG_VIDEO_H264_LEVEL;
            control.value = (enum v4l2_mpeg_video_h264_level)level;
            break;
        case V4L2_PIX_FMT_H265:
            control.id = V4L2_CID_MPEG_VIDEOENC_H265_LEVEL;
            control.value = (enum v4l2_mpeg_video_h265_level)level;
            break;
        default:
            COMP_WARN_MSG("Currently only supported for H.264 and H.265");
            return 0;
    }

    CHECK_V4L2_RETURN(setExtControls(ctrls),
            "Setting encoder level to " << level);
}

int
NvVideoEncoder::setConstantQp(bool enabled_rc)
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    RETURN_ERROR_IF_FORMATS_NOT_SET();
    RETURN_ERROR_IF_BUFFERS_REQUESTED();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEO_FRAME_RC_ENABLE;
    control.value = enabled_rc; // if false: disable rate control

    CHECK_V4L2_RETURN(setExtControls(ctrls),
            "Setting encoder constant qp to " << enabled_rc);
}

int
NvVideoEncoder::setRateControlMode(enum v4l2_mpeg_video_bitrate_mode mode)
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    RETURN_ERROR_IF_FORMATS_NOT_SET();
    RETURN_ERROR_IF_BUFFERS_REQUESTED();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEO_BITRATE_MODE;
    control.value = mode;

    CHECK_V4L2_RETURN(setExtControls(ctrls),
            "Setting encoder rate control mode to " << mode);
}

int
NvVideoEncoder::setMaxPerfMode(int flag)
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    RETURN_ERROR_IF_FORMATS_NOT_SET();
    RETURN_ERROR_IF_BUFFERS_REQUESTED();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEO_MAX_PERFORMANCE;
    control.value = flag;

    CHECK_V4L2_RETURN(setExtControls(ctrls),
            "Enabling Maximum Performance ");
}

int
NvVideoEncoder::setIFrameInterval(uint32_t interval)
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    RETURN_ERROR_IF_FORMATS_NOT_SET();
    RETURN_ERROR_IF_BUFFERS_REQUESTED();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEO_GOP_SIZE;
    control.value = interval;

    CHECK_V4L2_RETURN(setExtControls(ctrls),
            "Setting encoder I-frame interval to " << interval);
}

int
NvVideoEncoder::setIDRInterval(uint32_t interval)
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    RETURN_ERROR_IF_FORMATS_NOT_SET();
    RETURN_ERROR_IF_BUFFERS_REQUESTED();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEO_IDR_INTERVAL;
    control.value = interval;

    CHECK_V4L2_RETURN(setExtControls(ctrls),
            "Setting encoder IDR interval to " << interval);
}

int
NvVideoEncoder::forceIDR()
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    RETURN_ERROR_IF_FORMATS_NOT_SET();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_MFC51_VIDEO_FORCE_FRAME_TYPE;

    CHECK_V4L2_RETURN(setExtControls(ctrls),
            "Forcing IDR");
}

int
NvVideoEncoder::setTemporalTradeoff(v4l2_enc_temporal_tradeoff_level_type level)
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    RETURN_ERROR_IF_FORMATS_NOT_SET();
    RETURN_ERROR_IF_BUFFERS_REQUESTED();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEOENC_TEMPORAL_TRADEOFF_LEVEL;
    control.value = level;

    CHECK_V4L2_RETURN(setExtControls(ctrls),
            "Setting encoder temporal tradeoff level to " << level);
}

int
NvVideoEncoder::setSliceLength(v4l2_enc_slice_length_type type, uint32_t length)
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;
    v4l2_enc_slice_length_param param = {type, length};

    RETURN_ERROR_IF_FORMATS_NOT_SET();
    RETURN_ERROR_IF_BUFFERS_REQUESTED();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEOENC_SLICE_LENGTH_PARAM;
    control.string = (char *) &param;

    CHECK_V4L2_RETURN(setExtControls(ctrls),
            "Setting encoder packet type to " << type << ", length to " <<
            length);
}

int
NvVideoEncoder::setHWPresetType(v4l2_enc_hw_preset_type type)
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    RETURN_ERROR_IF_FORMATS_NOT_SET();
    RETURN_ERROR_IF_BUFFERS_REQUESTED();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEOENC_HW_PRESET_TYPE_PARAM;
    control.value = type;

    CHECK_V4L2_RETURN(setExtControls(ctrls),
            "Setting encoder HW Preset type to " << type);
}

int
NvVideoEncoder::setROIParams(uint32_t buffer_index,
        v4l2_enc_frame_ROI_params &params)
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    RETURN_ERROR_IF_FORMATS_NOT_SET();
    RETURN_ERROR_IF_BUFFERS_NOT_REQUESTED();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEOENC_ROI_PARAMS;
    control.string = (char *) &params;
    params.config_store = buffer_index;

    CHECK_V4L2_RETURN(setExtControls(ctrls),
            "Setting encoder ROI params for buffer " << buffer_index);
}

int
NvVideoEncoder::setPPEInitParams(v4l2_enc_ppe_init_params &params)
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    RETURN_ERROR_IF_FORMATS_NOT_SET();
    RETURN_ERROR_IF_BUFFERS_NOT_REQUESTED();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEOENC_PPE_INIT_PARAMS;
    control.string = (char *) &params;

    CHECK_V4L2_RETURN(setExtControls(ctrls),
            "Setting encoder PPE initialization params");
}

int
NvVideoEncoder::SetInputMetaParams(uint32_t buffer_index,
        v4l2_ctrl_videoenc_input_metadata &params)
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    RETURN_ERROR_IF_FORMATS_NOT_SET();
    RETURN_ERROR_IF_BUFFERS_NOT_REQUESTED();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEOENC_INPUT_METADATA;
    control.string = (char *) &params;
    params.config_store = buffer_index;

    CHECK_V4L2_RETURN(setExtControls(ctrls),
            "Setting encoder input metadata Params");
}

int
NvVideoEncoder::enableROI(v4l2_enc_enable_roi_param &params)
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    RETURN_ERROR_IF_FORMATS_NOT_SET();
    RETURN_ERROR_IF_BUFFERS_REQUESTED();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEOENC_ENABLE_ROI_PARAM;
    control.string = (char *) &params;

    CHECK_V4L2_RETURN(setExtControls(ctrls),
            "Enabling encoder ROI");
}

int
NvVideoEncoder::enableReconCRC(v4l2_enc_enable_reconcrc_param &params)
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    RETURN_ERROR_IF_FORMATS_NOT_SET();
    RETURN_ERROR_IF_BUFFERS_REQUESTED();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEOENC_ENABLE_RECONCRC_PARAM;
    control.string = (char *) &params;

    CHECK_V4L2_RETURN(setExtControls(ctrls),
            "Enabling encoder ReconCRC");
}

int
NvVideoEncoder::enableExternalRPS(v4l2_enc_enable_ext_rps_ctr &params)
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    RETURN_ERROR_IF_FORMATS_NOT_SET();
    RETURN_ERROR_IF_BUFFERS_REQUESTED();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEOENC_ENABLE_EXTERNAL_RPS_CONTROL;
    control.string = (char *) &params;

    CHECK_V4L2_RETURN(setExtControls(ctrls),
            "Enabling encoder External RPS");
}

int
NvVideoEncoder::enableExternalRC(v4l2_enc_enable_ext_rate_ctr &params)
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    RETURN_ERROR_IF_FORMATS_NOT_SET();
    RETURN_ERROR_IF_BUFFERS_REQUESTED();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEOENC_ENABLE_EXTERNAL_RATE_CONTROL;
    control.string = (char *) &params;

    CHECK_V4L2_RETURN(setExtControls(ctrls),
            "Enabling encoder External RC");
}

int NvVideoEncoder::setH265DisableSAO(bool enabled)
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    RETURN_ERROR_IF_FORMATS_NOT_SET();
    RETURN_ERROR_IF_BUFFERS_REQUESTED();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEOENC_H265_DISABLE_SAO;
    control.value = enabled;

    CHECK_V4L2_RETURN(setExtControls(ctrls),
                      "Setting encoder SAO filter to " << enabled);
}

int
NvVideoEncoder::enableAV1Tile(v4l2_enc_av1_tile_config &params)
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    RETURN_ERROR_IF_FORMATS_NOT_SET();
    RETURN_ERROR_IF_BUFFERS_REQUESTED();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEOENC_AV1_TILE_CONFIGURATION;
    control.string = (char *) &params;

    CHECK_V4L2_RETURN(setExtControls(ctrls),
            "Enabling encoder AV1 tile");
}

int
NvVideoEncoder::setAV1SsimRdo(bool enabled)
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    RETURN_ERROR_IF_FORMATS_NOT_SET();
    RETURN_ERROR_IF_BUFFERS_REQUESTED();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEOENC_AV1_ENABLE_SSIMRDO;
    control.value = enabled;

    CHECK_V4L2_RETURN(setExtControls(ctrls),
            "Setting encoder Ssim Rdo to " << enabled);
}

int
NvVideoEncoder::setAV1DisableCDFUpdate(bool disabled)
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    RETURN_ERROR_IF_FORMATS_NOT_SET();
    RETURN_ERROR_IF_BUFFERS_REQUESTED();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEOENC_AV1_DISABLE_CDF_UPDATE;
    control.value = disabled;

    CHECK_V4L2_RETURN(setExtControls(ctrls),
            "Setting encoder CDF update to " << !disabled);
}

int NvVideoEncoder::enableAV1TileGroups(bool enabled)
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    RETURN_ERROR_IF_FORMATS_NOT_SET();
    RETURN_ERROR_IF_BUFFERS_REQUESTED();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEOENC_AV1_ENABLE_TILE_GROUPS;
    control.value = enabled;

    CHECK_V4L2_RETURN(setExtControls(ctrls),
                      "Setting encoder tile groups present flag to " << enabled);
}

int NvVideoEncoder::setAV1FrameIDPresentFlag(bool enabled)
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    RETURN_ERROR_IF_FORMATS_NOT_SET();
    RETURN_ERROR_IF_BUFFERS_REQUESTED();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEOENC_AV1_ENABLE_FRAMEID_NUMBERS;
    control.value = enabled;

    CHECK_V4L2_RETURN(setExtControls(ctrls),
                      "Setting encoder frameID present flag to " << enabled);
}

int
NvVideoEncoder::setAV1ErrResilienceMode(bool enabled)
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    RETURN_ERROR_IF_FORMATS_NOT_SET();
    RETURN_ERROR_IF_BUFFERS_REQUESTED();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEOENC_AV1_ERR_RESILIENT_MODE;
    control.value = enabled;

    CHECK_V4L2_RETURN(setExtControls(ctrls),
                      "Setting encoder Error Resilient Mode to " << enabled);
}

int
NvVideoEncoder::setVirtualBufferSize(uint32_t size)
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;
    v4l2_enc_virtual_buffer_size buffer_size = {size};

    RETURN_ERROR_IF_FORMATS_NOT_SET();
    RETURN_ERROR_IF_BUFFERS_REQUESTED();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEOENC_VIRTUALBUFFER_SIZE;
    control.string = (char *) &buffer_size;

    CHECK_V4L2_RETURN(setExtControls(ctrls),
            "Setting encoder virtual buffer size to " << size);
}

int
NvVideoEncoder::setNumReferenceFrames(uint32_t num_frames)
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;
    v4l2_enc_num_ref_frames frames = { num_frames };

    RETURN_ERROR_IF_FORMATS_NOT_SET();
    RETURN_ERROR_IF_BUFFERS_REQUESTED();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEOENC_NUM_REFERENCE_FRAMES;
    control.string = (char *) &frames;

    CHECK_V4L2_RETURN(setExtControls(ctrls),
            "Setting encoder number of reference frames to " << num_frames);
}

int
NvVideoEncoder::setSliceIntrarefresh(uint32_t interval)
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;
    v4l2_enc_slice_intrarefresh_param param = {interval};

    RETURN_ERROR_IF_FORMATS_NOT_SET();
    RETURN_ERROR_IF_BUFFERS_REQUESTED();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEOENC_SLICE_INTRAREFRESH_PARAM;
    control.string = (char *) &param;

    CHECK_V4L2_RETURN(setExtControls(ctrls),
            "Setting encoder slice intrarefresh interval to " << interval);
}

int NvVideoEncoder::setCUDASliceIntrarefresh(uint32_t interval, uint32_t count)
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;
    v4l2_ctrl_intra_refresh param = {0};

    RETURN_ERROR_IF_FORMATS_NOT_SET();
    RETURN_ERROR_IF_BUFFERS_REQUESTED();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    param.enableIntraRefresh = 1;
    param.intraRefreshPeriod = interval;
    param.intraRefreshCnt = count;

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEOENC_SLICE_INTRAREFRESH_PARAM;
    control.string = (char *)&param;

    CHECK_V4L2_RETURN(setExtControls(ctrls),
                      "Setting encoder slice intrarefresh interval to " << interval);
}

int
NvVideoEncoder::setNumBFrames(uint32_t num)
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    RETURN_ERROR_IF_FORMATS_NOT_SET();
    RETURN_ERROR_IF_BUFFERS_REQUESTED();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEOENC_NUM_BFRAMES;
    control.value = num;

    CHECK_V4L2_RETURN(setExtControls(ctrls),
            "Setting encoder number of B frames to " << num);
}

int
NvVideoEncoder::setInsertSpsPpsAtIdrEnabled(bool enabled)
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    RETURN_ERROR_IF_FORMATS_NOT_SET();
    RETURN_ERROR_IF_BUFFERS_REQUESTED();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEOENC_INSERT_SPS_PPS_AT_IDR;
    control.value = enabled;

    CHECK_V4L2_RETURN(setExtControls(ctrls),
            "Setting encoder SPSPPSatIDR to " << enabled);
}

int
NvVideoEncoder::setCABAC(bool enabled)
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    RETURN_ERROR_IF_FORMATS_NOT_SET();
    RETURN_ERROR_IF_BUFFERS_REQUESTED();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEO_H264_ENTROPY_MODE;
    control.value = enabled;

    CHECK_V4L2_RETURN(setExtControls(ctrls),
            "Setting encoder CABAC encoding to " << enabled);
}

int
NvVideoEncoder::setSliceLevelEncode(bool enabled)
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    RETURN_ERROR_IF_FORMATS_NOT_SET();
    RETURN_ERROR_IF_BUFFERS_REQUESTED();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEOENC_ENABLE_SLICE_LEVEL_ENCODE;
    control.value = enabled;

    CHECK_V4L2_RETURN(setExtControls(ctrls),
            "Setting encode slice level encoding to " << enabled);
}

int
NvVideoEncoder::enableMotionVectorReporting()
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    RETURN_ERROR_IF_FORMATS_NOT_SET();
    RETURN_ERROR_IF_BUFFERS_REQUESTED();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEOENC_ENABLE_METADATA_MV;
    control.value = 1;

    CHECK_V4L2_RETURN(setExtControls(ctrls),
            "Enabling encoder motion vector reporting");
}

int
NvVideoEncoder::getMetadata(uint32_t buffer_index,
        v4l2_ctrl_videoenc_outputbuf_metadata &enc_metadata)
{
    v4l2_ctrl_video_metadata metadata;
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    RETURN_ERROR_IF_BUFFERS_NOT_REQUESTED();

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    metadata.buffer_index = buffer_index;
    metadata.VideoEncMetadata = &enc_metadata;

    control.id = V4L2_CID_MPEG_VIDEOENC_METADATA;
    control.string = (char *)&metadata;

    CHECK_V4L2_RETURN(getExtControls(ctrls),
            "Getting encoder output metadata for buffer " << buffer_index);
}

int
NvVideoEncoder::getMotionVectors(uint32_t buffer_index,
        v4l2_ctrl_videoenc_outputbuf_metadata_MV &enc_mv_metadata)
{
    v4l2_ctrl_video_metadata metadata;
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    RETURN_ERROR_IF_BUFFERS_NOT_REQUESTED();

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    metadata.buffer_index = buffer_index;
    metadata.VideoEncMetadataMV = &enc_mv_metadata;

    control.id = V4L2_CID_MPEG_VIDEOENC_METADATA_MV;
    control.string = (char *)&metadata;

    CHECK_V4L2_RETURN(getExtControls(ctrls),
            "Getting encoder output MV metadata for buffer " << buffer_index);
}

int
NvVideoEncoder::setQpRange(uint32_t MinQpI, uint32_t MaxQpI, uint32_t MinQpP,
            uint32_t MaxQpP, uint32_t MinQpB, uint32_t MaxQpB)
{
    v4l2_ctrl_video_qp_range qprange;
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    RETURN_ERROR_IF_FORMATS_NOT_SET();
    RETURN_ERROR_IF_BUFFERS_REQUESTED();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    qprange.MinQpI = MinQpI;
    qprange.MaxQpI = MaxQpI;
    qprange.MinQpP = MinQpP;
    qprange.MaxQpP = MaxQpP;
    qprange.MinQpB = MinQpB;
    qprange.MaxQpB = MaxQpB;

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEOENC_QP_RANGE;
    control.string = (char *)&qprange;

    CHECK_V4L2_RETURN(setExtControls(ctrls),
            "Setting encoder Qp range " << ctrls.count);
}

int NvVideoEncoder::setCudaConstantQp(uint32_t qpI, uint32_t qpP,
                                      uint32_t qpB)
{
    v4l2_ctrl_video_constqp qp_param;
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    RETURN_ERROR_IF_FORMATS_NOT_SET();
    RETURN_ERROR_IF_BUFFERS_REQUESTED();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    qp_param.constQpI = qpI;
    qp_param.constQpP = qpP;
    qp_param.constQpB = qpB;

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEOENC_CUDA_CONSTQP;
    control.string = (char *)&qp_param;

    CHECK_V4L2_RETURN(setExtControls(ctrls),
                      "Setting encoder const Qp " << ctrls.count);
}

int
NvVideoEncoder::setSampleAspectRatioWidth(uint32_t sar_width)
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    RETURN_ERROR_IF_FORMATS_NOT_SET();
    RETURN_ERROR_IF_BUFFERS_REQUESTED();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    switch (capture_plane_pixfmt)
    {
        case V4L2_PIX_FMT_H264:
            control.id = V4L2_CID_MPEG_VIDEO_H264_VUI_EXT_SAR_WIDTH;
            break;
        case V4L2_PIX_FMT_H265:
            control.id = V4L2_CID_MPEG_VIDEOENC_H265_VUI_EXT_SAR_WIDTH;
            break;
        default:
            COMP_ERROR_MSG("Unsupported encoder type");
            return -1;
    }
    control.value = sar_width;

    CHECK_V4L2_RETURN(setExtControls(ctrls),
            "Setting encoder SAR width to " << sar_width);
}

int
NvVideoEncoder::setSampleAspectRatioHeight(uint32_t sar_height)
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    RETURN_ERROR_IF_FORMATS_NOT_SET();
    RETURN_ERROR_IF_BUFFERS_REQUESTED();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    switch (capture_plane_pixfmt)
    {
        case V4L2_PIX_FMT_H264:
            control.id = V4L2_CID_MPEG_VIDEO_H264_VUI_EXT_SAR_HEIGHT;
            break;
        case V4L2_PIX_FMT_H265:
            control.id = V4L2_CID_MPEG_VIDEOENC_H265_VUI_EXT_SAR_HEIGHT;
            break;
        default:
            COMP_ERROR_MSG("Unsupported encoder type");
            return -1;
    }
    control.value = sar_height;

    CHECK_V4L2_RETURN(setExtControls(ctrls),
            "Setting encoder SAR width to " << sar_height);
}

int
NvVideoEncoder::setInsertVuiEnabled(bool enabled)
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    RETURN_ERROR_IF_FORMATS_NOT_SET();
    RETURN_ERROR_IF_BUFFERS_REQUESTED();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEOENC_INSERT_VUI;
    control.value = enabled;

    CHECK_V4L2_RETURN(setExtControls(ctrls),
            "Setting encoder InsertVUI to " << enabled);
}

int
NvVideoEncoder::setExtendedColorFormat(bool enabled)
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    RETURN_ERROR_IF_FORMATS_NOT_SET();
    RETURN_ERROR_IF_BUFFERS_REQUESTED();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEOENC_EXTEDED_COLORFORMAT;
    control.value = enabled;

    CHECK_V4L2_RETURN(setExtControls(ctrls),
            "Setting encoder extended colorformat to " << enabled);
}

int
NvVideoEncoder::setInsertAudEnabled(bool enabled)
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    RETURN_ERROR_IF_FORMATS_NOT_SET();
    RETURN_ERROR_IF_BUFFERS_REQUESTED();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEOENC_INSERT_AUD;
    control.value = enabled;

    CHECK_V4L2_RETURN(setExtControls(ctrls),
            "Setting encoder InsertAUD to " << enabled);
}

int
NvVideoEncoder::setAlliFramesEncode(bool enabled)
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    RETURN_ERROR_IF_FORMATS_NOT_SET();
    RETURN_ERROR_IF_BUFFERS_REQUESTED();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEOENC_ENABLE_ALLIFRAME_ENCODE;
    control.value = enabled;

    CHECK_V4L2_RETURN(setExtControls(ctrls),
            "Setting encoder InsertAUD to " << enabled);
}

int
NvVideoEncoder::DevicePoll(v4l2_ctrl_video_device_poll *devicepoll)
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    RETURN_ERROR_IF_FORMATS_NOT_SET();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEO_DEVICE_POLL;
    control.string = (char *)devicepoll;

    CHECK_V4L2_RETURN(setExtControls(ctrls),
            "Done calling video device poll ");
}

int
NvVideoEncoder::SetPollInterrupt()
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    RETURN_ERROR_IF_FORMATS_NOT_SET();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_SET_POLL_INTERRUPT;
    control.value = 1;

    CHECK_V4L2_RETURN(setExtControls(ctrls),
            "Setting encoder poll interrupt to 1 ");
}

int
NvVideoEncoder::ClearPollInterrupt()
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    RETURN_ERROR_IF_FORMATS_NOT_SET();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_SET_POLL_INTERRUPT;
    control.value = 0;

    CHECK_V4L2_RETURN(setExtControls(ctrls),
            "Setting encoder poll interrupt to 0 ");
}

int
NvVideoEncoder::setPocType(uint32_t pocType)
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    RETURN_ERROR_IF_FORMATS_NOT_SET();
    RETURN_ERROR_IF_BUFFERS_REQUESTED();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEOENC_POC_TYPE;
    control.value = pocType;

    CHECK_V4L2_RETURN(setExtControls(ctrls),
            "Setting Pic_order_cnt_type to " << pocType);
}

int NvVideoEncoder::setInitQP(uint32_t IinitQP,
            uint32_t PinitQP, uint32_t BinitQP)
{
    v4l2_ctrl_video_init_qp initqp;
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    RETURN_ERROR_IF_FORMATS_NOT_SET();
    RETURN_ERROR_IF_BUFFERS_REQUESTED();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    initqp.IInitQP = IinitQP;
    initqp.PInitQP = PinitQP;
    initqp.BInitQP = BinitQP;

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEOENC_INIT_FRAME_QP;
    control.string = (char *)&initqp;

    CHECK_V4L2_RETURN(setExtControls(ctrls),
            "Setting encoder Init QP " << ctrls.count);
}

int
NvVideoEncoder::setFramesToEncode(uint32_t framesToEncode)
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEOENC_FRAMES_TO_ENCODE;
    control.value = framesToEncode;

    CHECK_V4L2_RETURN(setExtControls(ctrls),
            "Setting num_frames_to_encode to " << framesToEncode);
}

int
NvVideoEncoder::setChromaFactorIDC(uint8_t value)
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    RETURN_ERROR_IF_FORMATS_NOT_SET();
    RETURN_ERROR_IF_BUFFERS_REQUESTED();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEOENC_H265_CHROMA_FACTOR_IDC;
    control.value = value;

    CHECK_V4L2_RETURN(setExtControls(ctrls),
            "Setting chroma_format_idc to " << value);
}

int
NvVideoEncoder::setLossless(bool enabled)
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    RETURN_ERROR_IF_FORMATS_NOT_SET();
    RETURN_ERROR_IF_BUFFERS_REQUESTED();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEOENC_ENABLE_LOSSLESS;
    control.value = enabled;

    CHECK_V4L2_RETURN(setExtControls(ctrls),
            "Setting lossless encoding to " << enabled);
}

int NvVideoEncoder::setDisableAMP(bool enabled)
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    RETURN_ERROR_IF_FORMATS_NOT_SET();
    RETURN_ERROR_IF_BUFFERS_REQUESTED();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEOENC_H265_DISABLE_AMP;
    control.value = enabled;

    CHECK_V4L2_RETURN(setExtControls(ctrls),
                      "Setting disable AMP types to " << enabled);
}

int NvVideoEncoder::enableTwoPassCBR()
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    RETURN_ERROR_IF_FORMATS_NOT_SET();
    RETURN_ERROR_IF_BUFFERS_REQUESTED();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEOENC_TWO_PASS_CBR;
    control.value = 1;

    CHECK_V4L2_RETURN(setExtControls(ctrls),
                      "Setting two pass CBR to 1");
}

int NvVideoEncoder::setGPUId(uint32_t gpuId)
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    RETURN_ERROR_IF_FORMATS_NOT_SET();
    RETURN_ERROR_IF_BUFFERS_REQUESTED();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEO_CUDA_GPU_ID;
    control.value = gpuId;

    CHECK_V4L2_RETURN(setExtControls(ctrls),
                      "Setting GPU Id to " << gpuId);
}

int NvVideoEncoder::setCudaMemType(uint32_t cumemType)
{
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    RETURN_ERROR_IF_OUTPUT_PLANE_FORMAT_NOT_SET();
    RETURN_ERROR_IF_BUFFERS_REQUESTED();

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id = V4L2_CID_MPEG_VIDEO_CUDA_MEM_TYPE;
    control.value = cumemType;
    CHECK_V4L2_RETURN(setExtControls(ctrls),
                      "Setting cumemType to " << cumemType);
}
