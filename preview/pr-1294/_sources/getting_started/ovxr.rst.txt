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

``isaaccapture.teleop_session_manager.ovxr_session`` provides ``OvxrSession``
and ``OvxrSessionConfig``. Implement ``OvxrSessionHost`` to supply the
application's renderer, Stage and ordinal ledger in an ``OvxrSessionBinding``.

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

Application-specific OVXR versions
----------------------------------

Isaac Capture pins the tested OVXR build. OVXR's pre-GA API can change between
builds, as described in its
`release notes <https://github.com/NVIDIA-Omniverse/ovxr/blob/main/CHANGELOG.md>`__.
uv respects that pin and reports a conflict if the application requires
incompatible OVRTX or OVStage versions.

To validate another OVXR build with the same Isaac Capture release, select an
OVXR release whose declared dependencies match the application's OVRTX/OVStage
requirements. Add an override in the application's root ``pyproject.toml``:

.. code-block:: toml

   [tool.uv]
   override-dependencies = ["ovxr==<compatible-version>"]

Replace ``<compatible-version>`` with the selected release. If the application
already has an ``override-dependencies`` list, append this entry to that list.
Then refresh the lock file and synchronize the environment using the ``ovxr``
extra from the example above:

.. code-block:: bash

   uv lock --upgrade-package ovxr
   uv sync --extra ovxr

An `override <https://docs.astral.sh/uv/concepts/resolution/#dependency-overrides>`__
changes dependency resolution; it does not establish compatibility with Isaac
Capture's OVXR integration. Validate the application's XR workflow before
relying on the selected build.

Applications using different provider versions need separate project
environments and lock files. Each can install the same Isaac Capture wheel
with its own OVXR/OVRTX/OVStage combination.
