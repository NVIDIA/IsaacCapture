/*
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
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

#ifndef ERROR_H
#define ERROR_H

#include <stdio.h>

#define LOG_ERROR(_file, _func, _line, _str, ...)                                                                      \
    do                                                                                                                 \
    {                                                                                                                  \
        fprintf(stderr, "Error generated. %s, %s:%d ", _file, _func, _line);                                           \
        fprintf(stderr, _str, ##__VA_ARGS__);                                                                          \
        fprintf(stderr, "\n");                                                                                         \
    } while (0)

/**
 * Simply report an error.
 */
#define REPORT_ERROR(_str, ...)                                                                                        \
    do                                                                                                                 \
    {                                                                                                                  \
        LOG_ERROR(__FILE__, __FUNCTION__, __LINE__, (_str), ##__VA_ARGS__);                                            \
    } while (0)

/**
 * Report and return an error that was first detected in the current method.
 */
#define ORIGINATE_ERROR(_str, ...)                                                                                     \
    do                                                                                                                 \
    {                                                                                                                  \
        LOG_ERROR(__FILE__, __FUNCTION__, __LINE__, (_str), ##__VA_ARGS__);                                            \
        return false;                                                                                                  \
    } while (0)

/**
 * Report and return an error that was first detected in some method
 * called by the current method.
 */
#define PROPAGATE_ERROR(_err)                                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        bool peResult = (_err);                                                                                        \
        if (peResult != true)                                                                                          \
        {                                                                                                              \
            LOG_ERROR(__FILE__, __FUNCTION__, __LINE__, "(propagating)");                                              \
            return false;                                                                                              \
        }                                                                                                              \
    } while (0)

#endif // ERROR_H
