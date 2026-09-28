.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Build from Source
=================

This page describes how to build Isaac Teleop from source, including core libraries, plugins, and
examples. The instructions align with the project's CMake configuration and the CI workflow
(:code-file:`build-ubuntu.yml <.github/workflows/build-ubuntu.yml>` in the GitHub repository).

.. contents:: Steps
   :local:
   :depth: 1

.. admonition:: Next Steps

   - To build and serve the **WebXR client** locally, see :doc:`webxr`.

Prerequisites
-------------

- **CMake** 3.24 or higher (Ubuntu 22.04's apt ``cmake`` is 3.22 — install a newer one
  from `Kitware's APT repository <https://apt.kitware.com/>`_ or with ``pip install cmake``)
- **C++20** compatible compiler
- **Python** 3.11, 3.12, or 3.13 (default 3.11; see ``ISAAC_TELEOP_PYTHON_VERSION`` in root ``CMakeLists.txt``)
- **uv** for Python dependency management and managed Python
- **Internet connection** for downloading dependencies via CMake FetchContent

.. note::
   **Optional — only needed to build the Televiz visualization module,** ``BUILD_VIZ``.
   ``BUILD_VIZ`` is auto-detected from the GPU stack: it defaults to ``ON`` when both of the
   following are found at configure time and to ``OFF`` otherwise, so a core-only source
   build still configures on a machine without them. Watch the
   ``-- BUILD_VIZ: <ON|OFF> (Vulkan=... CUDAToolkit=...)`` configure line to see which one
   is missing.

   - **Vulkan headers + loader** — ``libvulkan-dev`` on Linux, the LunarG SDK on Windows.
   - **CUDA Toolkit** (cudart at link time) — ``nvidia-cuda-toolkit`` or the official NVIDIA
     installer.

   ``BUILD_VIZ=ON`` additionally requires **glslangValidator** to compile shaders to SPIR-V
   (``glslang-tools`` on Linux, ``brew install glslang`` on macOS; ships with the Vulkan SDK
   on Windows), and on Linux **wayland-scanner** (``libwayland-dev``) for GLFW's Wayland
   backend — pass ``-DGLFW_BUILD_WAYLAND=OFF`` to build Televiz against X11 only instead.
   Neither gates the auto-default: if one is missing, the configure fails naming it rather
   than quietly dropping the module. Most users do not need any of this:
   ``pip install isaacteleop`` already ships the compiled ``isaacteleop.viz`` module. See
   `Other Build options`_ for the full option table.

.. _one-time-setup:

One time setup
--------------

Install build tools and dependencies, such as CMake, clang-format. See :code-file:`build-ubuntu.yml <.github/workflows/build-ubuntu.yml>` in the GitHub repository for
the list of dependencies. On **Ubuntu**, install build tools and clang-format:

.. code-block:: bash

   sudo apt-get update
   sudo apt-get install -y build-essential cmake libegl-dev libx11-dev libwayland-dev clang-format-14 ccache patchelf pkg-config glslang-tools

Runtime-only dependencies (needed to actually run teleop, not to build):

.. code-block:: bash

   # adb — required for OOB teleop (``--setup-oob``) to talk to the headset over USB.
   # coturn — required for USB-local mode (``--usb-local``); runs a local TURN server
   #          so WebRTC ICE can relay traffic from the headset to the CloudXR backend
   #          over the USB cable.
   sudo apt-get install -y android-tools-adb coturn

Our build system uses `uv`_ for Python version and dependency management. Install `uv`_ if not already installed:

.. code-block:: bash

   curl -LsSf https://astral.sh/uv/install.sh | sh

The installer drops ``uv`` in ``~/.local/bin``, which is not on ``PATH`` in most shells. Add it for
the current shell — and to your ``~/.bashrc`` to make it stick — or every ``uv`` command below
fails with ``uv: command not found``:

.. code-block:: bash

   export PATH="$HOME/.local/bin:$PATH"

.. note::
   While the build system uses `uv`_, the final Python packages can be installed via any Python package manager
   such as `pip <https://pip.pypa.io/>`_ or `conda <https://conda.io/>`_.

1. Clone the repository
-----------------------

.. code-block:: bash

   git clone https://github.com/NVIDIA/IsaacTeleop.git
   cd IsaacTeleop

.. note::
   Dependencies (OpenXR SDK, pybind11, yaml-cpp) are automatically downloaded
   during CMake configuration using FetchContent. No manual dependency installation or
   git submodule initialization is required.

Pre-download CloudXR SDK (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

   If you are using the default flow, skip this step. The
   :code-file:`CMakeLists.txt <src/core/cloudxr/python/CMakeLists.txt>`
   will automatically download the CloudXR SDK by calling the
   :code-file:`download_cloudxr_runtime_sdk.sh <scripts/download_cloudxr_runtime_sdk.sh>`
   script.

Sometimes NVIDIA might share early access CloudXR SDKs with you. In that case, you may get
tarballs such as:

- ``CloudXR-<version-for-runtime-sdk>-Linux-<arch>-sdk.tar.gz`` (CloudXR Runtime SDK)
- ``CloudXR-exp-<version-for-runtime-sdk>-Linux-<arch>-sdk.tar.gz`` (experimental
  runtime). Packaged by default as ``isaacteleop.cloudxr_exp``; needed for Jetson Orin
  support (for example :doc:`/getting_started/televiz`) until the default runtime covers
  those platforms.
- ``nvidia-cloudxr-<version-for-web-sdk>.tgz`` (CloudXR Web SDK)

You can place them in the :code-file:`deps/cloudxr/` directory and update the ``deps/cloudxr/.env``
file to locally override the default version defined in :code-file:`deps/cloudxr/.env.default`,
like this:

.. code-block:: bash

   CXR_RUNTIME_SDK_VERSION=<version-for-runtime-sdk>
   CXR_WEB_SDK_VERSION=<version-for-web-sdk>

The experimental runtime is packaged into the wheel as ``isaacteleop.cloudxr_exp`` by default
(``ENABLE_CLOUDXR_EXP_BUNDLE=ON``). Pass ``-DENABLE_CLOUDXR_EXP_BUNDLE=OFF`` to skip it.
Select it at runtime with ``ISAAC_TELEOP_CLOUDXR_EXP``.
See :ref:`dedicated-cloudxr-runtime`.

2. CMake: Configure and build
-----------------------------

From the project root:

.. code-block:: bash

   cmake -B build                       # configure
   cmake --build build --parallel       # build
   cmake --install build                # install

Add any other options as ``-D`` flags on the configure line, for example
``cmake -B build -DCMAKE_BUILD_TYPE=Debug``.

.. important::

   The Python version is baked into a build directory's CMake cache and its build
   venv, so ``ISAAC_TELEOP_PYTHON_VERSION`` cannot be changed on an existing tree —
   configuring again with a different value fails with an explanatory error. Give
   each version its own directory:

   .. code-block:: bash

      cmake -B build-py3.12 -DISAAC_TELEOP_PYTHON_VERSION=3.12
      cmake --build build-py3.12 --parallel

This will:

1. Fetch dependencies (OpenXR SDK, yaml-cpp, pybind11, FlatBuffers, MCAP, and optionally Catch2 for tests) via FetchContent in ``deps/third_party/CMakeLists.txt``
2. Build core C++ libraries (schema, oxr_utils, plugin_manager, oxr, pusherio, deviceio, mcap, etc.) and Python bindings
3. Build the Python wheel
4. Build examples (if enabled)
5. Install to ``./install`` (default prefix set in root ``CMakeLists.txt``)


C++ Formatting Enforcement (Linux)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On Linux, **clang-format** is enforced by default; the build fails if formatting changes would be applied. The project
uses **clang-format-14** for consistent results across distributions (see ``cmake/ClangFormat.cmake``).

To disable enforcement, set ``ENABLE_CLANG_FORMAT_CHECK`` to ``OFF``:

.. code-block:: bash

   cmake -B build -DENABLE_CLANG_FORMAT_CHECK=OFF

Useful targets:

- ``clang_format_check`` — verifies formatting (part of ``ALL`` on Linux)
- ``clang_format_fix`` — applies formatting in place

.. code-block:: bash

   cmake --build build --target clang_format_check
   cmake --build build --target clang_format_fix

Other Build options
~~~~~~~~~~~~~~~~~~~

The CMake options (defined in root :code-file:`CMakeLists.txt` and :code-file:`cmake/SetupPython.cmake`):

.. list-table:: Common CMake Options
   :widths: 20 36 44
   :header-rows: 1

   * - Option
     - CMake flag
     - Default / Notes
   * - **Build type**
     - ``CMAKE_BUILD_TYPE``
     - ``Release`` or ``Debug``
   * - **Install prefix**
     - ``CMAKE_INSTALL_PREFIX``
     - ``./install``

.. list-table:: Common Isaac Teleop Options
   :widths: 24 36 40
   :header-rows: 1

   * - Option
     - CMake flag
     - Default / Notes
   * - **Examples**
     - ``BUILD_EXAMPLES``
     - ``ON``
   * - **Python bindings**
     - ``BUILD_PYTHON_BINDINGS``
     - ``ON``
   * - **Python version**
     - ``ISAAC_TELEOP_PYTHON_VERSION``
     - ``3.11`` (3.11, 3.12, or 3.13)
   * - **Testing**
     - ``BUILD_TESTING``
     - ``ON``; enables CTest and Catch2
   * - **Clang-format check**
     - ``ENABLE_CLANG_FORMAT_CHECK``
     - ``ON`` on Linux
   * - **Televiz visualization**
     - ``BUILD_VIZ``
     - Auto: ``ON`` when Vulkan and the CUDA Toolkit are detected, else ``OFF``. Force with ``-DBUILD_VIZ=ON`` / ``-DBUILD_VIZ=OFF``. (Most users don't need this — ``pip install isaacteleop`` already ships the compiled ``isaacteleop.viz`` module.)

.. list-table:: Plugin Specific Options
   :widths: 26 34 40
   :header-rows: 1

   * - Option
     - CMake flag
     - Default / Notes
   * - **Plugins master switch**
     - ``BUILD_PLUGINS``
     - ``ON``
   * - **OAK camera plugin**
     - ``BUILD_PLUGIN_OAK_CAMERA``
     - ``OFF``; when ``ON``, builds DepthAI v3.x and pulls its dependencies through
       vcpkg, so it also needs ``CMAKE_TOOLCHAIN_FILE`` on a fresh build directory.
       See :doc:`/device/oak`.
   * - **Teleop ROS2 example only**
     - ``BUILD_EXAMPLE_TELEOP_ROS2``
     - ``OFF``; when ``ON``, only ``examples/teleop_ros2`` (e.g. Docker)

Examples
~~~~~~~~

Build for a different Python version — each needs its own build directory
(``3.11``, ``3.12``, ``3.13`` are supported):

.. code-block:: bash

   cmake -B build-py3.12 -DISAAC_TELEOP_PYTHON_VERSION=3.12
   cmake --build build-py3.12 --parallel

Debug build:

.. code-block:: bash

   cmake -B build -DCMAKE_BUILD_TYPE=Debug
   cmake --build build

Build without examples:

.. code-block:: bash

   cmake -B build -DBUILD_EXAMPLES=OFF
   cmake --build build

Build without Python bindings:

.. code-block:: bash

   cmake -B build -DBUILD_PYTHON_BINDINGS=OFF
   cmake --build build

Build with the OAK camera plugin. It needs the vcpkg toolchain, and CMake only
reads ``CMAKE_TOOLCHAIN_FILE`` on a build tree's **first** configure, so delete
``build/`` first if it already exists (see :doc:`/device/oak`):

.. code-block:: bash

   rm -rf build
   cmake -B build -DBUILD_PLUGIN_OAK_CAMERA=ON \
       -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
   cmake --build build --target camera_plugin_oak --parallel

Build only the teleop_ros2 example (e.g. for Docker, as in :code-file:`build-ubuntu.yml <.github/workflows/build-ubuntu.yml>` teleop-ros2-docker job):

.. code-block:: bash

   cmake -B build -DBUILD_EXAMPLES=OFF -DBUILD_EXAMPLE_TELEOP_ROS2=ON
   cmake --build build

Clean rebuild (``--fresh`` wipes the CMake cache and reconfigures):

.. code-block:: bash

   cmake -B build --fresh
   cmake --build build

3. Running tests
----------------

When ``BUILD_TESTING`` is ``ON``, CTest is enabled at the top level. Run all tests either via the CMake ``test`` target or with ``ctest``:

.. code-block:: bash

   cmake --build build --target test

   # Or with ctest (e.g. parallel, output on failure)
   ctest --test-dir build --output-on-failure --parallel

The CI uses ``ctest`` (see :code-file:`build-ubuntu.yml <.github/workflows/build-ubuntu.yml>`).

4. Install the ``isaacteleop`` pip package
------------------------------------------

The wheels are built in the ``./install/wheels/`` directory. Install the package from the wheels.
Using ``pip``, you need to pass the ``--no-index`` option to automatically find the right wheel
based on the Python version.  Note that ``pip`` and ``uv pip`` has slightly different options.

.. code-block:: bash

   # Pass --no-index to use only wheels in ./install/wheels/;
   # Pass --force-reinstall to replace an existing install.
   pip install "isaacteleop[retargeters,cloudxr,ui]" --find-links=./install/wheels/ --no-index --force-reinstall

.. code-block:: bash

   # Pass --reinstall to replace an existing install.
   uv pip install "isaacteleop[retargeters,cloudxr,ui]" --find-links=./install/wheels/ --reinstall

Alternative: install directly from source with pip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The repository also ships a root :code-file:`pyproject.toml` using the
`scikit-build-core <https://scikit-build-core.readthedocs.io/>`_ backend, which
drives the same top-level ``CMakeLists.txt``. This lets you build and install
with standard Python tooling, without running the ``cmake -B build`` steps
yourself; the classic flow above (which CI uses to produce ``install/wheels/`` and
the released wheels) is unchanged, so the two coexist.

.. code-block:: bash

   pip install .            # build + install the compiled wheel from source
   pip install -e .         # editable / developer install

.. note::

   The CMake build tree is kept under ``build-wheel/<cache-tag>/`` (e.g.
   ``build-wheel/cpython-311/``) — one per Python version, so different
   interpreters never share a configured CMake cache — instead of a temporary
   directory. Re-installs are therefore incremental. It sits outside ``build/``, so
   it never collides with a classic ``cmake -B build`` tree. Both are gitignored.

What this path does and how it differs from the classic flow:

- **Full build, no overrides.** The backend passes no ``cmake.define`` overrides,
  so ``BUILD_EXAMPLES``, ``BUILD_TESTING``, and ``BUILD_PLUGINS`` stay at their
  CMake defaults: a ``pip install`` builds the full default target — extensions,
  examples, C++ tests, and plugins (a plugin without its SDK skips gracefully). It
  installs only the ``isaacteleop_wheel`` and ``isaacteleop_binaries`` components,
  so C++ library/header install rules do not leak into the wheel.
- **clang-format is enforced.** Because the gate is left on, the build **fails** if
  ``clang-format-14`` is not installed or any C++ file is unformatted — a
  ``pip install`` therefore requires ``clang-format-14`` on Linux.
- **All executables on PATH.** Every executable built from ``src/`` and
  ``examples/`` — example programs, C++ test binaries, plugin tools
  (``oxr_simple_api_demo``, ``se3_printer``, ``schema_tests``, …) — is installed
  into the venv's ``bin/`` (the wheel's scripts scheme; see the executable-install
  loop in the root ``CMakeLists.txt``), so they are on ``PATH`` once the venv is
  active. They statically link the teleop core libraries, so they are
  self-contained. Shared libraries (Python extension modules, plugin ``.so``) are
  not executables and are **not** placed in ``bin/``.
- **ABI.** Extensions are compiled against the interpreter that runs the build
  (so the wheel's ABI tag matches) and against NumPy 2.x, so a single wheel works
  with both NumPy 1.x and 2.x at runtime.
- **No type stubs.** The pip-built wheel omits the ``.pyi`` stubs that the classic
  and released wheels ship (stub generation shells out to ``uv``, which is not
  guaranteed inside pip's build isolation). Imports and runtime behavior are
  unaffected.
- **Version.** The pip-facing version is derived from the ``VERSION`` file as
  ``MAJOR.MINOR+local`` — the same value the classic flow produces for a
  non-CI/local build. The pip path is a from-source/dev build, so it is always
  tagged ``+local``; the git-aware release versioning (tags, ``a1`` / ``rc1`` /
  ``.devN`` labels) stays with the classic flow via
  :code-file:`cmake/IsaacTeleopVersion.cmake`.

.. admonition:: Editable installs and iterating on pure-Python subpackages

   An editable install (``pip install -e .``) never recompiles on import. Pure-Python
   subpackages resolve straight to ``src/python/isaacteleop/``, so edits **take effect
   live** — a fresh interpreter is enough.

   Compiled extensions are the exception: ``.so``/``.pyd`` still come from the CMake
   build tree, so C++ changes need a ``cmake --build`` (or a re-run of
   ``pip install -e .``) first.

See :doc:`/references/build` for the full build-system reference.

.. admonition:: ``retargeters`` extra on aarch64 (Jetson / DGX)

   The full ``retargeters`` extra does **not** resolve on ``aarch64`` (some of its
   pinned dependencies have no aarch64 wheels). On Jetson/DGX-class robotics
   targets, install the ``retargeters-lite`` extra instead:

   .. code-block:: bash

      pip install "isaacteleop[retargeters-lite]"

.. toctree::
   :hidden:

   webxr

..
   References
.. _`uv`: https://docs.astral.sh/uv/
