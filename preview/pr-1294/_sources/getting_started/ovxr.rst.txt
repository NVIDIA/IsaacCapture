.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

OVXR installation
=================

OVXR is an optional presentation dependency. Build Isaac Capture from source and
select its ``ovxr`` extra only for applications that need XR presentation.
The optional packages are available publicly; no internal package index, GitHub
Actions artifact or local Kit build is needed.

.. warning::

   OVXR 0.1.0.5842 requires OVRTX 0.6.0.1791947691 and OVStage 0.3.0.1791986420.
   Applications pinning OVRTX 0.5.x or OVStage 0.2.x, including the current
   Isaac Lab main dependency set, need a coordinated provider upgrade first.
   This package selection prepares the Isaac Capture PRs; compatibility and
   runtime/headset validation with the public providers remain pending.

Optional packages and licenses
------------------------------

The ``isaaccapture[ovxr]`` extra retrieves OVXR and its declared OVRTX/OVStage
dependencies. Base Isaac Capture installation/import does not request or load
OVXR. These OV packages are downloaded separately, rather than bundled in the
Isaac Capture wheel, and retain their own licenses.

* **OVXR 0.1.0.5842, OVRTX and OVStage:** NVIDIA proprietary software; review the
  license included with each selected distribution. Package information is at
  `OVXR <https://pypi.org/project/ovxr/0.1.0.5842/>`__,
  `OVRTX <https://pypi.org/project/ovrtx/0.6.0.1791947691/>`__ and
  `OVStage <https://pypi.org/project/ovstage/0.3.0.1791986420/>`__.
  OVXR and OVStage come from PyPI; OVRTX's PyPI installer retrieves its native
  wheel from `NVIDIA's public Python index <https://pypi.nvidia.com>`__.

The README's **Separately retrieved optional materials** section contains the
external-materials notice. Existing CloudXR native libraries are bundled in
Isaac Capture wheels and remain subject to their separate NVIDIA license and
CloudXR EULA acceptance.

Install OVXR
------------

With the application's Python environment selected, run:

.. code-block:: bash

   uv pip install ovxr==0.1.0.5842 --index https://pypi.org/simple
   uv pip check

Use ``--python /path/to/environment/bin/python`` when that environment is not
selected. This installs OVXR and its exact OVRTX/OVStage dependencies; the
consuming application's other constraints must accept that provider set.
Successful installation is not runtime, streaming or headset qualification.

The Linux x86_64 OVXR wheel has SHA256
``146c90f1130456e836e120bff77e2f4f27bf387b67258e8b5c8641a0c0633854``.
A Windows wheel is also published; package availability is not platform runtime
verification. OVXR 0.1.0.5842 has no Linux aarch64 wheel.

Build Isaac Capture from source
--------------------------------

Follow :doc:`/getting_started/build_from_source/index` for the managed Python and CMake workflow.
Build and install your own wheel with its ``ovxr`` extra using the same Python
environment as the consuming application. The optional dependency must be
selected explicitly; ordinary installations and imports do not require OVXR.
Keep the source-built wheel selected in the consuming project's scoped uv source
configuration and use ordinary lock refresh/sync/run after a rebuild. Do not
copy an installed environment or bypass a wheel hash mismatch.

.. Section underline lengths include inline markup characters in the source title.

Keep ordinary ``uv run`` resolution consistent
----------------------------------------------

``uv pip install`` updates an environment, not an application's ``uv.lock``.
Declare the local source-built Isaac Capture wheel in the consuming application's
optional dependency set, then refresh its lock through the normal workflow.
Do not use permanent ``--no-sync`` to bypass incompatible application pins.

For example, a Linux x86_64 application can select:

.. code-block:: toml

   [project.optional-dependencies]
   ovxr = [
     "isaaccapture[ovxr]; platform_system == 'Linux' and platform_machine == 'x86_64'",
   ]

   [tool.uv.sources]
   isaaccapture = { path = "path/to/your/local-wheel.whl", extra = "ovxr" }

OVXR and its dependencies resolve from their public package metadata. No internal
index or credentials are required. Preserve any existing source selections for
other optional dependencies. OVXR's exact OVRTX/OVStage pins must agree with the
application; regenerate its lock and use ordinary ``uv run`` with its OVXR extra
selected. Base Isaac Capture and other teleoperation paths should not request
the ``ovxr`` extra.
