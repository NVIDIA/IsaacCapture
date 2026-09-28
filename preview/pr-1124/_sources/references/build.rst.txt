Build System
============

Project structure
-----------------

The layout below reflects the actual ``CMakeLists.txt`` hierarchy (root ``CMakeLists.txt`` and ``src/core/CMakeLists.txt``):

.. code-block:: text
   :class: code-100col

   IsaacTeleop/
   ├── CMakeLists.txt              # Top-level: deps, core, examples, plugins, etc.
   ├── cmake/
   │   ├── DepthAIVcpkgManifest.cmake  # OAK/DepthAI vcpkg manifest (pre-project)
   │   ├── SetupPython.cmake       # BUILD_PYTHON_BINDINGS, uv-managed Python
   │   ├── ClangFormat.cmake       # clang_format_check / clang_format_fix
   │   └── ...
   ├── deps/
   │   ├── CMakeLists.txt          # Adds third_party
   │   ├── third_party/CMakeLists.txt  # FetchContent: OpenXR, yaml-cpp, pybind11, etc.
   │   └── cloudxr/                # CloudXR (CI / optional)
   ├── src/core/
   │   ├── CMakeLists.txt          # Core options, subdirs in dependency order
   │   ├── schema/                 # FlatBuffer schemas and generated code
   │   ├── oxr_utils/              # Header-only OpenXR utilities
   │   ├── plugin_manager/         # Plugin manager (C++ and Python)
   │   ├── oxr/                    # OpenXR session management (C++ and Python)
   │   ├── pusherio/               # PusherIO (depends on oxr)
   │   ├── deviceio/               # Device I/O library (C++ and Python)
   │   ├── mcap/                   # MCAP recording (depends on deviceio)
   │   ├── retargeting_engine/     # Retargeting (Python)
   │   ├── retargeting_engine_ui/  # Retargeting UI (Python)
   │   ├── teleop_session_manager/ # Teleop session manager (Python)
   │   ├── cloudxr/                # CloudXR runtime helper (Python)
   │   └── python/                 # Python wheel packaging (when BUILD_PYTHON_BINDINGS=ON)
   ├── src/plugins/                # Built when BUILD_PLUGINS=ON
   │   ├── plugin_utils/
   │   ├── controller_synthetic_hands/
   │   ├── generic_3axis_pedal/
   │   ├── manus/
   │   ├── oak/                    # When BUILD_PLUGIN_OAK_CAMERA=ON
   |   └── haptikos/
   └── examples/
       ├── oxr/                    # OpenXR examples (C++ and Python)
       ├── retargeting/
       ├── teleop_session_manager/
       ├── teleop_ros2/
       ├── schemaio/
       └── native_openxr/

CMake integration
-----------------

The project uses a modern CMake target-based approach. Libraries export namespaced alias targets (e.g. ``oxr::oxr_core``, ``deviceio::deviceio_session``, ``isaacteleop_schema``); include directories are propagated. Package config files are generated for use after install. See the respective ``CMakeLists.txt`` in ``src/core`` and under ``examples/`` for target names and usage.

Using the Python wheel
----------------------

After building, install the wheel with ``uv`` or ``pip``:

.. code-block:: bash

   uv pip install isaacteleop --find-links=./install/wheels/ --reinstall

   # or
   pip install install/wheels/isaacteleop-*.whl

Installing from source with pip (scikit-build-core)
---------------------------------------------------

In addition to the classic CMake flow above (which stages the package
and produces a wheel under ``install/wheels/``), the repository ships a root
:code-file:`pyproject.toml` that exposes ``isaacteleop`` through the
`scikit-build-core <https://scikit-build-core.readthedocs.io/>`_ build backend,
driving the *same* top-level ``CMakeLists.txt`` (so the two paths coexist):

.. code-block:: bash

   pip install .            # build + install the compiled wheel from source
   pip install -e .         # editable / developer install

For the walkthrough — how this scoped build differs from the classic flow (ABI,
omitted type stubs, versioning) and the caveat on iterating on pure-Python
subpackages under an editable install — see
:doc:`/getting_started/build_from_source/index`.

Build directory layout and per-Python-version builds
----------------------------------------------------

A CMake binary directory holds exactly one configured Python version — the
interpreter and ABI are baked into ``CMakeCache.txt`` and into the build venv. The
two build paths therefore never share a directory:

- **classic CMake** → whatever you pass to ``-B``, ``build/`` by convention.
  ``ISAAC_TELEOP_PYTHON_VERSION`` selects the interpreter (default ``3.11``), and
  changing it on an existing tree is rejected with an error rather than silently
  reusing the old one — give each version its own directory.

  .. code-block:: bash

     cmake -B build
     cmake --build build --parallel
     cmake --install build

     cmake -B build-py3.12 -DISAAC_TELEOP_PYTHON_VERSION=3.12   # a second version

- **pip / scikit-build-core** → ``build-wheel/<cache-tag>/`` (e.g.
  ``build-wheel/cpython-312/``), set by ``build-dir`` in
  :code-file:`pyproject.toml` and chosen automatically per interpreter, keyed off
  ``sys.implementation.cache_tag``. It lives outside ``build/`` so ``pip install``
  and a classic ``cmake -B build`` never collide, and ``rm -rf build`` leaves the
  incremental wheel tree intact.

``build/`` and ``build-wheel/`` are gitignored.

Output locations
----------------

After a successful build and install, relative to your build directory:

- **C++ libraries:** ``src/core/`` (and under each module)
- **Python wheel:** ``wheels/isaacteleop-*.whl``
- **Examples (binaries):** under ``examples/`` (e.g. ``examples/oxr/cpp/``)
- **Installed files:** ``install/`` (or your ``CMAKE_INSTALL_PREFIX``)
  - Libraries: ``install/lib/``
  - Headers: ``install/include/``
  - Wheels: ``install/wheels/``
  - Examples: ``install/examples/``

Troubleshooting
---------------

Dependencies fail to download
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FetchContent in ``deps/third_party/CMakeLists.txt`` requires network access. If downloads fail, check connectivity. Key repositories:

- OpenXR SDK: https://github.com/KhronosGroup/OpenXR-SDK.git
- pybind11: https://github.com/pybind/pybind11.git
- yaml-cpp: https://github.com/jbeder/yaml-cpp.git
- FlatBuffers, Catch2, MCAP: see ``deps/third_party/CMakeLists.txt`` for URLs and tags.

You can inspect the ``_deps`` directory in your build tree for fetch logs.

CMake can't find OpenXR
~~~~~~~~~~~~~~~~~~~~~~~

OpenXR is fetched automatically; it is not a system package. If configuration fails, try a clean configure
(``--fresh`` wipes the CMake cache and reconfigures):

.. code-block:: bash

   cmake -B build --fresh

Examples or tests can't find the library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When building from the top-level, examples and tests use the build tree. Ensure you run
``cmake --build build`` from the project root and run executables from the build directory
(or use ``cmake --install build`` and run from ``install/``).

uv or Python version
~~~~~~~~~~~~~~~~~~~~

``cmake/SetupPython.cmake`` requires **uv** and uses ``ISAAC_TELEOP_PYTHON_VERSION``. Install uv as
in :ref:`One time setup <one-time-setup>` and pass ``-DISAAC_TELEOP_PYTHON_VERSION=3.12`` (or 3.11,
3.13) if you need a specific version.

Reference
---------

- Root build and options: ``CMakeLists.txt``
- Core modules and options: ``src/core/CMakeLists.txt``
- Dependencies: ``deps/third_party/CMakeLists.txt``
- Python and uv: ``cmake/SetupPython.cmake``
- OAK/DepthAI vcpkg manifest: ``cmake/DepthAIVcpkgManifest.cmake``
- CI (Ubuntu, matrix build_type/python_version/arch): ``.github/workflows/build-ubuntu.yml``
