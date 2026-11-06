# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test the real Python bindings with a CPU-only decoder implementation."""

import importlib.util
import subprocess
import sysconfig
from pathlib import Path
from types import SimpleNamespace

import pybind11
import pytest


@pytest.fixture(scope="module")
def bindings(tmp_path_factory):
    path = tmp_path_factory.mktemp("decoder_dimensions")
    stub = path / "decoder_stub.cpp"
    stub.write_text("""#include <nv_video_decoder.hpp>
namespace camera_receiver::codec {
struct NvVideoDecoder::Impl { DecoderConfig cfg; };
NvVideoDecoder::NvVideoDecoder(const DecoderConfig& cfg) : impl_(new Impl{cfg}) {}
NvVideoDecoder::~NvVideoDecoder() = default;
uint32_t NvVideoDecoder::width() const { return impl_->cfg.width; }
uint32_t NvVideoDecoder::height() const { return impl_->cfg.height; }
bool NvVideoDecoder::decode(const uint8_t*, std::size_t, uintptr_t, std::size_t, int) { return true; }
bool NvVideoDecoder::decode_nv12(const uint8_t*, std::size_t, uintptr_t, std::size_t, int) { return true; }
void NvVideoDecoder::reset() {}
}
""")
    codec = Path(__file__).resolve().parents[2] / "camera_receiver" / "codec"
    extension = path / (
        "_camera_receiver_codec" + sysconfig.get_config_var("EXT_SUFFIX")
    )
    subprocess.run(
        [
            "c++",
            "-shared",
            "-fPIC",
            "-std=c++17",
            str(codec / "codec_bindings.cpp"),
            str(stub),
            "-o",
            str(extension),
            "-I" + str(codec),
            "-I" + pybind11.get_include(),
            "-I" + sysconfig.get_paths()["include"],
        ],
        check=True,
    )
    spec = importlib.util.spec_from_file_location("_camera_receiver_codec", extension)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("method", ["decode", "decode_nv12"])
@pytest.mark.parametrize(
    "width,height", [(1280, 720), (640, 720), (1280, 360), (640, 360), (1920, 1080)]
)
def test_binding_checks_dimensions_against_decoder(bindings, method, width, height):
    config = bindings.DecoderConfig()
    config.width, config.height = 1280, 720
    decoder = bindings.NvVideoDecoder(config)
    shape = (height, width, 4) if method == "decode" else (height * 3 // 2, width)
    output = SimpleNamespace(
        __cuda_array_interface__={
            "shape": shape,
            "data": (1234, False),
            "typestr": "|u1",
            "strides": None,
        }
    )
    call = getattr(decoder, method)
    if (width, height) == (config.width, config.height):
        assert call(b"au", output, width, height)
    else:
        with pytest.raises(RuntimeError, match="configured decoder size"):
            call(b"au", output, width, height)
