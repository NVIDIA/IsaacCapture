# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import shutil
import subprocess

import pytest


_STATE = r"""
#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

struct TestState {
    std::string fault;
    std::vector<std::string> events;
    int context_depth = 0;
    int copy_calls = 0;
    bool locked = false;
    uint8_t surface[12] = {};
};
inline TestState state;
"""

_CUDA_DRIVER = r"""
#pragma once
#include "test_state.h"
using CUdevice = int;
using CUcontext = void*;
using CUresult = int;
constexpr CUresult CUDA_SUCCESS = 0;
inline CUresult cuInit(unsigned) { return CUDA_SUCCESS; }
inline CUresult cuDeviceGet(CUdevice* device, int id) { *device = id; return CUDA_SUCCESS; }
inline CUresult cuDevicePrimaryCtxRetain(CUcontext* context, CUdevice) {
    *context = reinterpret_cast<CUcontext>(1);
    return CUDA_SUCCESS;
}
inline CUresult cuDevicePrimaryCtxRelease(CUdevice) { return CUDA_SUCCESS; }
inline CUresult cuGetErrorString(CUresult, const char** error) {
    *error = "injected driver error";
    return CUDA_SUCCESS;
}
inline CUresult cuCtxPushCurrent(CUcontext) {
    ++state.context_depth;
    state.events.push_back("push");
    return CUDA_SUCCESS;
}
inline CUresult cuCtxPopCurrent(CUcontext*) {
    --state.context_depth;
    state.events.push_back("pop");
    return CUDA_SUCCESS;
}
"""

_CUDA_RUNTIME = r"""
#pragma once
#include <cstddef>
#include "test_state.h"
using cudaStream_t = void*;
using cudaError_t = int;
constexpr cudaError_t cudaSuccess = 0;
constexpr unsigned cudaStreamNonBlocking = 1;
constexpr int cudaMemcpyDeviceToDevice = 0;
inline const char* cudaGetErrorString(cudaError_t) { return "injected CUDA error"; }
inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned) {
    *stream = reinterpret_cast<cudaStream_t>(2);
    return cudaSuccess;
}
inline cudaError_t cudaStreamDestroy(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaStreamSynchronize(cudaStream_t) {
    state.events.push_back("sync");
    return state.fault.find("sync") != std::string::npos ? 1 : cudaSuccess;
}
inline cudaError_t cudaMemcpy2DAsync(void*, std::size_t, const void*, std::size_t,
                                    std::size_t, std::size_t, int, cudaStream_t) {
    const std::string operation = ++state.copy_calls == 1 ? "copy_y" : "copy_uv";
    state.events.push_back(operation);
    return state.fault.find(operation) == 0 ? 1 : cudaSuccess;
}
"""

_NVDEC = r"""
#pragma once
#include "cuda.h"
enum cudaVideoCodec { cudaVideoCodec_H264, cudaVideoCodec_HEVC };
enum cudaVideoSurfaceFormat { cudaVideoSurfaceFormat_NV12, cudaVideoSurfaceFormat_P016 };
class NvDecoder {
public:
    template <typename... Args> explicit NvDecoder(Args...) {}
    int Decode(const uint8_t*, int) {
        state.events.push_back("decode");
        if (state.fault == "decode") throw std::runtime_error("injected decode error");
        return state.fault == "no_frame" ? 0 : 1;
    }
    uint8_t* GetLockedFrame() {
        if (state.fault == "null_frame") return nullptr;
        if (state.locked) throw std::runtime_error("frame still locked");
        state.locked = true;
        state.events.push_back("lock");
        return state.surface;
    }
    void UnlockFrame(uint8_t** frame) {
        state.events.push_back("unlock");
        state.locked = false;
        *frame = nullptr;
    }
    int GetWidth() { return state.fault == "size_mismatch" ? 8 : 4; }
    int GetHeight() { return 2; }
    int GetDeviceFramePitch() { return 4; }
    int GetLumaPlaneSize() { return 8; }
    cudaVideoSurfaceFormat GetOutputFormat() {
        return state.fault == "format" ? cudaVideoSurfaceFormat_P016 : cudaVideoSurfaceFormat_NV12;
    }
};
"""

_HARNESS = r"""
#include "nv_video_decoder.hpp"
#include "nv12_to_rgba.cuh"
#include "test_state.h"
#include <iostream>

namespace camera_receiver::codec {
void launch_nv12_to_rgba(const uint8_t*, const uint8_t*, int, int, int, int,
                         uint8_t*, int, bool, int, cudaStream_t) {
    state.events.push_back("rgba");
    if (state.fault.find("launch") == 0) throw std::runtime_error("injected launch error");
}
void launch_nv12_rotate_180(const uint8_t*, const uint8_t*, int, int, int, int,
                            uint8_t*, int, cudaStream_t) {
    state.events.push_back("rotate");
    if (state.fault.find("launch") == 0) throw std::runtime_error("injected launch error");
}
}

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    using namespace camera_receiver::codec;
    const std::string mode = argv[1];
    const std::string fault = argv[2];
    DecoderConfig config;
    config.width = 4;
    config.height = 2;
    NvVideoDecoder decoder(config);
    state.events.clear();
    state.fault = fault;
    uint8_t output[32] = {};
    const uint8_t packet[] = {0, 0, 1};
    auto decode = [&] {
        return mode == "rgba"
            ? decoder.decode(packet, sizeof(packet), reinterpret_cast<uintptr_t>(output), 16)
            : decoder.decode_nv12(packet, sizeof(packet), reinterpret_cast<uintptr_t>(output),
                                   4, mode == "nv12_rotate" ? 180 : 0);
    };
    std::string error;
    bool produced = false;
    try { produced = decode(); }
    catch (const std::exception& exception) { error = exception.what(); }

    bool failed = false;
    auto require = [&](bool condition, const char* message) {
        if (!condition) { std::cerr << message << '\n'; failed = true; }
    };
    const bool drops = fault == "no_frame" || fault == "null_frame" || fault == "size_mismatch";
    const bool throws = fault != "none" && !drops;
    require(!error.empty() == throws, "unexpected exception outcome");
    require(produced == (fault == "none"), "unexpected frame result");
    require(state.context_depth == 0, "CUDA context push/pop is unbalanced");
    require(!state.locked, "NVDEC frame remains locked");
    auto count = [&](const std::string& event) {
        return std::count(state.events.begin(), state.events.end(), event);
    };
    auto position = [&](const std::string& event) {
        return std::find(state.events.begin(), state.events.end(), event) - state.events.begin();
    };
    const bool frame_locked = fault != "decode" && fault != "no_frame" && fault != "null_frame";
    require(count("unlock") == (frame_locked ? 1 : 0), "frame must be unlocked exactly once");
    require(count("pop") == 1, "context must be popped exactly once");
    if (frame_locked) require(position("unlock") < position("pop"), "unlock must precede context pop");
    const bool submitted = count("rgba") || count("rotate") || count("copy_y");
    if (submitted) {
        require(count("sync") == 1, "submitted GPU work must synchronize exactly once");
        require(position("sync") < position("unlock"), "GPU reads must finish before frame unlock");
    }
    if (fault == "format") {
        require(!submitted, "unsupported format reached an 8-bit GPU operation");
        require(error.find("only 8-bit NV12") != std::string::npos, "missing format error");
    }
    if (fault.find("launch") == 0)
        require(error == "injected launch error", "cleanup replaced the launch exception");
    if (fault.find("copy_uv") == 0) {
        require(count("copy_y") == 1 && count("copy_uv") == 1, "UV failure did not follow Y copy");
        require(error.find("NV12 UV") != std::string::npos, "cleanup replaced the UV copy exception");
    }
    if (!failed) {
        state.fault = "none";
        state.copy_calls = 0;
        require(decode(), "decoder did not accept the next packet");
        require(state.context_depth == 0 && !state.locked, "next packet leaked decoder resources");
    }
    return failed ? 1 : 0;
}
"""


@pytest.fixture(scope="module")
def native_decoder_probe(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("native decoder cleanup tests require a C++17 compiler")
    build = tmp_path_factory.mktemp("decoder_cleanup")
    for name, content in {
        "test_state.h": _STATE,
        "cuda.h": _CUDA_DRIVER,
        "cuda_runtime.h": _CUDA_RUNTIME,
        "NvDecoder/NvDecoder.h": _NVDEC,
        "probe.cpp": _HARNESS,
    }.items():
        target = build / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    codec = Path(__file__).resolve().parents[1] / "codec"
    executable = build / "decoder_cleanup"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            f"-I{build}",
            f"-I{codec}",
            f"-I{codec / 'inc'}",
            str(codec / "nv_video_decoder.cpp"),
            str(build / "probe.cpp"),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return executable


@pytest.mark.parametrize("mode", ["rgba", "nv12", "nv12_rotate"])
@pytest.mark.parametrize(
    "fault",
    ["none", "format", "sync", "decode", "no_frame", "null_frame", "size_mismatch"],
)
def test_native_decoder_resources_and_format(native_decoder_probe, mode, fault):
    result = subprocess.run(
        [str(native_decoder_probe), mode, fault],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "mode,fault",
    [
        ("rgba", "launch"),
        ("rgba", "launch_cleanup_sync"),
        ("nv12_rotate", "launch"),
        ("nv12_rotate", "launch_cleanup_sync"),
        ("nv12", "copy_y"),
        ("nv12", "copy_uv"),
        ("nv12", "copy_uv_cleanup_sync"),
    ],
)
def test_native_decoder_cleans_up_gpu_submission_errors(
    native_decoder_probe, mode, fault
):
    result = subprocess.run(
        [str(native_decoder_probe), mode, fault],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stdout + result.stderr
