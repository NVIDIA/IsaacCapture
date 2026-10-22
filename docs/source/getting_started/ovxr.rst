.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

OVXR installation
=================

OVXR provides optional XR presentation for applications that use Isaac Capture
for teleoperation input. Select the ``ovxr`` extra when installing Isaac Capture
to include OVXR and its rendering dependencies.

Install XR support
------------------

Follow :doc:`/development/build_from_source/index` to build Isaac Capture and
select a Python environment matching the generated wheel. Include ``ovxr`` in
the installation requirement:

.. code-block:: bash

   uv pip install "isaaccapture[ovxr]==X.Y+local" --find-links=./install/wheels/
   uv pip check

Replace ``X.Y`` with the version of your source build, as described in the build
guide. Base installations and imports do not require OVXR.

OVXR declares its compatible OVRTX and OVStage versions in its package metadata.
The consuming application's dependency constraints must accept those versions.
See the package pages below for supported Python versions and platforms.

Optional packages and licenses
------------------------------

The ``ovxr`` extra installs OVXR and its OVRTX/OVStage dependencies separately
from the Isaac Capture wheel. These packages retain their own licenses:

* `OVXR <https://pypi.org/project/ovxr/>`__,
  `OVRTX <https://pypi.org/project/ovrtx/>`__ and
  `OVStage <https://pypi.org/project/ovstage/>`__ are NVIDIA proprietary software.
  Review the license included with each distribution before use.
* `OVUI <https://pypi.org/project/ovui/>`__ is Apache-2.0 and is installed by
  the independent ``desktop-mirror`` extra. Review its package metadata for
  supported Python versions and platforms.

CloudXR native libraries are bundled in Isaac Capture wheels and remain subject
to their separate NVIDIA license and CloudXR EULA acceptance.

Optional camera panels
----------------------

Camera picture-in-picture panels require the native camera helper. Follow the
source-build guide to configure a build for your application's Python version,
then enable camera support and rebuild the wheel:

.. code-block:: bash

   cmake -S . -B build -DBUILD_OVXR_CAMERA_QUADS=ON
   cmake --build build --target python_package python_stubs python_wheel --parallel

Install the generated wheel with its ``ovxr`` extra using the instructions
above. The extra includes the CUDA Python bindings used for GPU camera output;
``BUILD_OVXR_CAMERA_QUADS`` controls whether the wheel contains the native helper.
The helper is disabled by default.

Camera support requires Vulkan and CUDA development files. On Linux x86-64,
CMake uses installed SDKs when available and downloads missing development files
into the build directory. The GPU driver and Vulkan runtime loader must be
installed on the host. Other platforms require installed development SDKs.
Use ``CUDAToolkit_ROOT`` to select a CUDA SDK and ``VULKAN_SDK`` to help locate a
Vulkan SDK.

Optional desktop mirror
-----------------------

Select both extras with ``isaaccapture[ovxr,desktop-mirror]`` in the installation
requirement or the application's optional dependencies to include the OVUI
mirror. The mirror requires a desktop display associated with a CUDA device.
The native camera helper is required only when camera panels are also enabled.

Initialize the renderer before starting the mirror. The mirror uses the OpenGL
backend and defaults ``OMNIUI_BACKEND`` to ``opengl`` when it is unset. Configure
this setting before the first OVUI initialization; existing settings and an
already initialized backend are preserved. The CUDA image-copy path does not
support the Vulkan UI backend. This setting does not change headset rendering.

Use from another application
----------------------------

For a uv-managed application, declare Isaac Capture with its ``ovxr`` extra and
select the source-built wheel in the application's project configuration:

.. code-block:: toml

   [project.optional-dependencies]
   ovxr = ["isaaccapture[ovxr]"]

   [tool.uv.sources]
   isaaccapture = { path = "path/to/your/local-wheel.whl", extra = "ovxr" }

Replace the wheel path with the file produced by your build. Refresh the
application's lock file after changing the wheel or dependency selection, then
run with ``uv run --extra ovxr``. Installing packages with ``uv pip install``
alone does not update the application's lock file.
