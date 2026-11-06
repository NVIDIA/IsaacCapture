/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Minimal NvBufSurf wrapper for argus_sender.
 *
 * NVIDIA's sample NvBufSurface.cpp also wraps transform helpers and therefore
 * requires libnvbufsurftransform.so. The sender only needs allocation, so this
 * file keeps the standalone binary off that optional dependency.
 */

#include <NvBufSurface.h>
#include <cstdio>
#include <cstring>

int NvBufSurf::NvDestroy(int fd)
{
    if (fd <= 0)
        return -1;

    NvBufSurface* nvbufSurf = NULL;
    NvBufSurfaceFromFd(fd, reinterpret_cast<void**>(&nvbufSurf));
    if (!nvbufSurf)
        return -1;

    return NvBufSurfaceDestroy(nvbufSurf);
}

int NvBufSurf::NvAllocate(NvCommonAllocateParams* allocateParams, uint32_t numBuffers, int* fd)
{
    if (!allocateParams || !fd || numBuffers < 1)
        return -1;

    NvBufSurfaceAllocateParams inputParams;
    std::memset(&inputParams, 0, sizeof(inputParams));
    inputParams.params.width = allocateParams->width;
    inputParams.params.height = allocateParams->height;
    inputParams.params.memType = allocateParams->memType;
    inputParams.params.layout = allocateParams->layout;
    inputParams.params.colorFormat = allocateParams->colorFormat;
    inputParams.memtag = allocateParams->memtag;
    inputParams.chromaSubsampling.chromaLocHoriz = allocateParams->chromaSubsampling.chromaLocHoriz;
    inputParams.chromaSubsampling.chromaLocVert = allocateParams->chromaSubsampling.chromaLocVert;

    for (uint32_t index = 0; index < numBuffers; ++index)
    {
        NvBufSurface* nvbufSurf = NULL;
        const int ret = NvBufSurfaceAllocate(&nvbufSurf, 1, &inputParams);
        if (ret < 0 || !nvbufSurf)
            return -1;
        nvbufSurf->numFilled = 1;
        fd[index] = nvbufSurf->surfaceList[0].bufferDesc;
    }

    return 0;
}

int NvBufSurf::NvTransform(NvCommonTransformParams*, int, int)
{
    std::fprintf(stderr, "NvBufSurf::NvTransform is not included in this standalone build\n");
    return -1;
}

int NvBufSurf::NvTransformAsync(NvCommonTransformParams*, NvBufSurfTransformSyncObj_t*, int, int)
{
    std::fprintf(stderr, "NvBufSurf::NvTransformAsync is not included in this standalone build\n");
    return -1;
}
