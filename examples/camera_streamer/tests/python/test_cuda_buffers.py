# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the native buffer checks without a CUDA device or toolkit."""

import importlib.util
import subprocess
import sysconfig
from pathlib import Path
from types import SimpleNamespace

import pybind11
import pytest


@pytest.fixture(scope="module")
def validators(tmp_path_factory):
    path = tmp_path_factory.mktemp("cuda_buffers")
    source = path / "bindings.cpp"
    source.write_text("""#include <cuda_buffers.hpp>
PYBIND11_MODULE(_test_cuda_buffers, m) {
    m.def("nv12", [](py::object o) { return extract_nv12_buffer(o, 4, 2).row_pitch_bytes; });
    m.def("rgba", [](py::object o) { return extract_cuda_buffer(o, 4, 2).row_pitch_bytes; });
}
""")
    extension = path / ("_test_cuda_buffers" + sysconfig.get_config_var("EXT_SUFFIX"))
    codec = Path(__file__).resolve().parents[2] / "camera_receiver" / "codec"
    subprocess.run(
        [
            "c++",
            "-shared",
            "-fPIC",
            "-std=c++17",
            str(source),
            "-o",
            str(extension),
            "-I" + str(codec),
            "-I" + pybind11.get_include(),
            "-I" + sysconfig.get_paths()["include"],
        ],
        check=True,
    )
    spec = importlib.util.spec_from_file_location("_test_cuda_buffers", extension)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "kind,shape,pitch", [("nv12", (3, 4), 4), ("rgba", (2, 4, 4), 16)]
)
@pytest.mark.parametrize(
    "typestr,readonly,error",
    [
        ("|u1", False, None),
        ("<f4", False, "uint8"),
        ("|u1", True, "writable"),
        ("|i1", False, "uint8"),
    ],
)
def test_output_buffer_contract(
    validators, kind, shape, pitch, typestr, readonly, error
):
    output = SimpleNamespace(
        __cuda_array_interface__={
            "shape": shape,
            "data": (1234, readonly),
            "typestr": typestr,
            "strides": None,
            "version": 3,
        }
    )
    validate = getattr(validators, kind)
    if error:
        with pytest.raises(RuntimeError, match=error):
            validate(output)
    else:
        assert validate(output) == pitch
