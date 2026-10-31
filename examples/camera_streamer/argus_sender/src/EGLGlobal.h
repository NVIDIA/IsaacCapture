/*
 * SPDX-License-Identifier: BSD-3-Clause
 */

/*
 * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef EGLGLOBAL_H
#define EGLGLOBAL_H

#include <EGL/egl.h>
#include <EGL/eglext.h>

namespace ArgusSamples
{

#ifdef EGL_KHR_stream
#    define EGL_KHR_STREAM_FUNCS                                                                                       \
        EGL_EXTN_FUNC(PFNEGLCREATESTREAMKHRPROC, eglCreateStreamKHR)                                                   \
        EGL_EXTN_FUNC(PFNEGLDESTROYSTREAMKHRPROC, eglDestroyStreamKHR)                                                 \
        EGL_EXTN_FUNC(PFNEGLSTREAMATTRIBKHRPROC, eglStreamAttribKHR)                                                   \
        EGL_EXTN_FUNC(PFNEGLQUERYSTREAMKHRPROC, eglQueryStreamKHR)
#else
#    define EGL_KHR_STREAM_FUNCS
#endif

#ifdef EGL_KHR_stream_consumer_gltexture
#    define EGL_KHR_STREAM_CONSUMER_GLTEXTURE_FUNCS                                                                    \
        EGL_EXTN_FUNC(PFNEGLSTREAMCONSUMERGLTEXTUREEXTERNALKHRPROC, eglStreamConsumerGLTextureExternalKHR)             \
        EGL_EXTN_FUNC(PFNEGLSTREAMCONSUMERACQUIREKHRPROC, eglStreamConsumerAcquireKHR)                                 \
        EGL_EXTN_FUNC(PFNEGLSTREAMCONSUMERRELEASEKHRPROC, eglStreamConsumerReleaseKHR)
#else
#    define EGL_KHR_STREAM_CONSUMER_GLTEXTURE_FUNCS
#endif

#ifdef EGL_KHR_stream_producer_eglsurface
#    define EGL_KHR_STREAM_PRODUCER_EGLSURFACE_FUNCS                                                                   \
        EGL_EXTN_FUNC(PFNEGLCREATESTREAMPRODUCERSURFACEKHRPROC, eglCreateStreamProducerSurfaceKHR)
#else
#    define EGL_KHR_STREAM_PRODUCER_EGLSURFACE_FUNCS
#endif

#ifdef EGL_KHR_fence_sync
#    define EGL_KHR_FENCE_SYNC_FUNCS                                                                                   \
        EGL_EXTN_FUNC(PFNEGLCREATESYNCKHRPROC, eglCreateSyncKHR)                                                       \
        EGL_EXTN_FUNC(PFNEGLDESTROYSYNCKHRPROC, eglDestroySyncKHR)                                                     \
        EGL_EXTN_FUNC(PFNEGLCLIENTWAITSYNCKHRPROC, eglClientWaitSyncKHR)                                               \
        EGL_EXTN_FUNC(PFNEGLGETSYNCATTRIBKHRPROC, eglGetSyncAttribKHR)
#else
#    define EGL_KHR_FENCE_SYNC_FUNCS
#endif

#ifdef EGL_NV_stream_sync
#    define EGL_NV_STREAM_SYNC_FUNCS EGL_EXTN_FUNC(PFNEGLCREATESTREAMSYNCNVPROC, eglCreateStreamSyncNV)
#else
#    define EGL_NV_STREAM_SYNC_FUNCS
#endif

#ifdef EGL_KHR_reusable_sync
#    define EGL_KHR_REUSABLE_SYNC_FUNCS EGL_EXTN_FUNC(PFNEGLSIGNALSYNCKHRPROC, eglSignalSyncKHR)
#else
#    define EGL_KHR_REUSABLE_SYNC_FUNCS
#endif

#ifdef EGL_KHR_image
#    define EGL_KHR_IMAGE_FUNCS                                                                                        \
        EGL_EXTN_FUNC(PFNEGLCREATEIMAGEKHRPROC, eglCreateImageKHR)                                                     \
        EGL_EXTN_FUNC(PFNEGLDESTROYIMAGEKHRPROC, eglDestroyImageKHR)
#else
#    define EGL_KHR_IMAGE_FUNCS
#endif

#ifdef EGL_KHR_wait_sync
#    define EGL_KHR_WAIT_SYNC_FUNCS EGL_EXTN_FUNC(PFNEGLWAITSYNCKHRPROC, eglWaitSyncKHR)
#else
#    define EGL_KHR_WAIT_SYNC_FUNCS
#endif


#define EGL_EXTN_FUNC_LIST                                                                                             \
    EGL_KHR_STREAM_FUNCS                                                                                               \
    EGL_KHR_STREAM_CONSUMER_GLTEXTURE_FUNCS                                                                            \
    EGL_KHR_STREAM_PRODUCER_EGLSURFACE_FUNCS                                                                           \
    EGL_KHR_FENCE_SYNC_FUNCS                                                                                           \
    EGL_NV_STREAM_SYNC_FUNCS                                                                                           \
    EGL_KHR_REUSABLE_SYNC_FUNCS                                                                                        \
    EGL_KHR_IMAGE_FUNCS                                                                                                \
    EGL_KHR_WAIT_SYNC_FUNCS

#define EGL_EXTN_FUNC(_type, _name) extern _type _name;
EGL_EXTN_FUNC_LIST
#undef EGL_EXTN_FUNC

/*!
 * A class to initialize and cleanup an EGL display.
 */
class EGLDisplayHolder
{
public:
    EGLDisplayHolder(bool disableWindow = false);
    ~EGLDisplayHolder();

    bool initialize(EGLNativeDisplayType native = EGL_DEFAULT_DISPLAY);
    bool cleanup();

    EGLDisplay get() const;

private:
    EGLDisplay m_display;
    bool m_disableWindow;
};

/*!
 * A class to wrap and destroy an EGL image.
 */
class EGLImageHolder
{
public:
    EGLImageHolder(EGLDisplay eglDisplay, EGLImageKHR eglImage);
    ~EGLImageHolder();

    EGLImageKHR getImage() const;

private:
    EGLDisplay m_display;
    EGLImageKHR m_image;
};

/*!
 * A class to create and destroy an EGL stream.
 */
class EGLStreamHolder
{
public:
    EGLStreamHolder();
    ~EGLStreamHolder();

    bool create(EGLDisplay display);
    bool destroy();

    EGLStreamKHR get() const;

    EGLint getState() const;

private:
    EGLDisplay m_display;
    EGLStreamKHR m_stream;
};

// Helper macro to ensure the stream is in the given state.
#define CHECK_STREAM_STATE(stream, state)                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        EGLint state = (stream).getState();                                                                            \
        if (state != EGL_STREAM_STATE_##state##_KHR)                                                                   \
            ORIGINATE_ERROR("Stream not in " #state " state but in state 0x%x", state);                                \
    } while (0)

} // namespace ArgusSamples

#endif // EGLGLOBAL_H
