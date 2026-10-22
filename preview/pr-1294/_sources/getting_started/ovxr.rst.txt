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

CloudXR native libraries are bundled in Isaac Capture wheels and remain subject
to their separate NVIDIA license and CloudXR EULA acceptance.

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
