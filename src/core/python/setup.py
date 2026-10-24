# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Setup script that forces platform-specific wheel generation.

Since we include pre-compiled .so/.pyd files (pybind11 extensions built by CMake),
we need to override setuptools to recognize this as a platform wheel, not pure Python.
"""

from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    """Distribution that always forces a platform-specific wheel."""

    def has_ext_modules(self):
        return True


class BuildPy(build_py):
    """Retire optional camera data left by a previous build configuration."""

    def run(self):
        super().run()
        if not self.package_data.get("isaaccapture.ovxr_session_manager"):
            # Setuptools reuses build_lib; omitted package data is not deleted.
            # Remove only these generated optional outputs, never source files.
            root = (
                Path(self.build_lib) / "isaaccapture/ovxr_session_manager/openxr_layer"
            )
            for name in (
                "libXrApiLayer_isaacteleop_pip.so",
                "XrApiLayer_isaacteleop_pip.json",
            ):
                (root / name).unlink(missing_ok=True)


setup(distclass=BinaryDistribution, cmdclass={"build_py": BuildPy})
