.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Generated Tracker Code
======================

Trackers whose live impl only moves a FlatBuffer between a plugin and the host via
``SchemaTracker`` or ``SchemaPusher`` — *schema-based* trackers — are not hand-written. They are
declared in a TOML manifest and their C++ and Python glue is generated when you configure the build.
Adding one is a ``.fbs`` schema plus roughly ten lines of TOML; see
:ref:`Schema-based trackers (manifest) <schema-based-tracker-manifest>` for the authoring workflow.

This page records what is generated today, what is deliberately still hand-written, and what
would have to change to generate the rest.

How it works
------------

Three inputs drive the generator:

- :code-file:`src/core/deviceio_trackers/trackers.toml` — one ``[[tracker]]`` entry per tracker,
  carrying only the values that differ from the defaults.
- :code-file:`src/core/deviceio_trackers/defaults.toml` — the defaults, expressed with
  ``%placeholder%`` substitutions (``%name%``, ``%name_CamelCase%``, and so on).
- :code-dir:`src/core/codegen` — ``manifest.py`` resolves the placeholders, ``templates.py``
  derives type and file names, ``templates/`` holds the C++ bodies as ``*.hpp.template`` /
  ``*.cpp.template`` under ``pull/`` and ``push/`` (``@KEY@`` substitution via ``template_renderer.py``),
  and ``generate_trackers.py`` renders the output.

:code-file:`cmake/GenerateTrackers.cmake` runs the generator through ``execute_process`` at
**configure** time, not build time, because CMake needs the resulting source list before it can
define targets. Output lands in ``${CMAKE_BINARY_DIR}/generated/trackers/`` and is **not** checked
into git, exactly like the ``flatc`` output. Sources are wired through an explicit
``generated_sources.cmake`` list (not by globbing that directory). Two consequences worth knowing:

- Grepping ``src/`` for a generated class such as ``Se3Tracker`` finds the manifest entry and the
  generator, not a ``.cpp``. Searching, debugging, and clangd all need a configured build tree.
- The manifests, every ``.py`` under ``src/core/codegen``, and every template under
  ``templates/`` are registered as ``CMAKE_CONFIGURE_DEPENDS``, so editing any of them re-runs
  configure on the next build.

The generator rewrites a file only when its content changed. Configure also passes
``--prune-stale`` so that after renaming or removing a tracker, orphan headers cannot remain on
the generated include path and keep satisfying a stale ``#include``.

What each entry produces
------------------------

Per manifest entry, in ``${CMAKE_BINARY_DIR}/generated/trackers/``:

.. code-block:: text
   :class: code-100col

   deviceio_base/<header>_base.hpp                    # I<Class>Impl interface
   deviceio_trackers/inc/deviceio_trackers/<header>.hpp
   deviceio_trackers/<header>.cpp                     # the ITracker facade
   live_trackers/live_<header>_impl.{hpp,cpp}         # wraps SchemaTracker / SchemaPusher
   replay_trackers/replay_<header>_impl.{hpp,cpp}

``header`` defaults from ``name`` (``<name>_tracker``, or ``name`` when it already ends in
``_tracker``). Override it when the public ``#include`` stem must keep a historical path.

The shared registration points stay hand-written and ``#include`` a generated ``.inc`` fragment,
so hand-written trackers keep their own rows. The fragments under ``generated/trackers/inc/`` cover
the live and replay factories (includes, forward declarations, try-create thunks, dispatch rows,
factory declarations and definitions), the MCAP recording traits in
:code-file:`src/core/mcap/cpp/inc/mcap/recording_traits.hpp`, the pybind blocks in
:code-file:`src/core/deviceio_trackers/python/tracker_bindings.cpp`, and a
``_generated_tracker_exports.py`` that :code-file:`src/python/isaacteleop/deviceio_trackers/__init__.py`
star-imports. Because that last one is spliced into ``__all__``, a new manifest entry needs no
Python edit at all.

``.pyi`` stubs need no work either: :code-file:`src/core/python/generate_stubs.py` derives them from
the built module.

Generated today
---------------

.. list-table::
   :header-rows: 1
   :widths: 26 26 48

   * - Manifest entry
     - Class
     - Notes
   * - ``joint_state``
     - ``JointStateTracker``
     - One generic joint-space device (leader arm, exoskeleton, ...)
   * - ``se3_tracker``
     - ``Se3Tracker``
     - Overrides ``class``; the default would give ``Se3TrackerTracker``
   * - ``oglo_tactile``
     - ``OgloTactileTracker``
     - MCAP channels are ``oglo``/``oglo_tracked``, so ``channel`` is overridden
   * - ``generic_3axis_pedal``
     - ``Generic3AxisPedalTracker``
     - Schema lives in ``pedals.fbs``, so ``schema`` and ``channel`` are overridden
   * - ``haptic_command``
     - ``HapticCommandPushTracker``
     - ``direction = "push"``: a typed producer wrapping ``SchemaPusher``
   * - ``frame_metadata_oak``
     - ``FrameMetadataTrackerOak``
     - Overrides ``class`` and ``header`` (file stem ``frame_metadata_tracker_oak``)

Still hand-written
------------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Tracker
     - Why it is not generated
   * - ``HeadTracker``, ``HandTracker``, ``ControllerTracker``
     - Real ``xrLocate*`` / hand-tracking calls, not schema-based FlatBuffer readers
   * - ``FullBodyTracker`` (PICO vendor)
     - Native ``XR_BD_body_tracking``
   * - ``FullBodyTracker`` (Noitom vendor)
     - Genuinely schema-based by mechanism, but a **vendor** of an existing facade — see below
   * - ``MessageChannelTracker``
     - ``XR_NV_opaque_data_channel`` with its own connection state machine
   * - ``HapticCommandReaderTracker``
     - Cross-process consumer: ``read_all_samples`` on one collection, buckets by
       ``HapticCommand.endpoint`` (left/right). Paired with generated ``HapticCommandPushTracker``.
   * - ``TensorPushTracker``
     - Deliberately kept as the untyped ``bytes`` escape hatch

A quick way to tell the two groups apart: only schema-based impls mention ``SchemaTracker`` or
``SchemaPusher``. Of the hand-written live impls that use those helpers, each is listed above.

Future work
-----------

**Multi-endpoint reader (rejected generated shape).** We briefly generated
``HapticCommandReaderTracker`` with a ``multi_endpoint`` template (bucket samples by
``HapticCommand.endpoint`` on one push-tensor collection). That shape was removed: it made the
codegen tree branchy for a single device, and a generic multi-endpoint reader does not fit
``pull/`` or ``push``. The hand-written reader remains the supported approach; a future redesign
(split tensors/collections per endpoint, or a dedicated generator) is optional follow-up.

**Vendored shape (Noitom).** ``LiveFullBodyTrackerNoitomImpl`` reads
``SchemaTracker<FullBodyPoseRecord, FullBodyPose>``, so it is schema-based by mechanism, but it is a
second *vendor* of the hand-written ``FullBodyTracker`` rather than its own tracker type. Generating
it would need a shape that emits **only** a live impl and a vendor-keyed dispatch row — no facade,
base interface, recording traits, replay half, or pybind block, since it shares all of those with the
PICO vendor. Its ``collection_id`` and ``max_flatbuffer_size`` also arrive through
``TrackerVendor::params`` at runtime instead of being fixed at generation time.

**Schema pybind bindings.** The field-by-field ``src/core/schema/python/*_bindings.h`` files are
still hand-written. They are derivable from the ``.bfbs`` reflection data ``flatc`` already emits
(``--bfbs-gen-embed --reflect-names --reflect-types``), so this is a viable follow-up, but it is a
separate generator with its own risks.

**Manifest ``python_accessor`` (pybind method name).** Generated ``pull`` trackers use
``fragments/pybind_pull.template``, which binds whatever string ``python_accessor`` holds (default
``get_data``; ``oglo_tactile`` and ``generic_3axis_pedal`` override with device-specific names).
``direction = "push"`` uses ``push`` via defaults. Future work: remove the manifest key and
hard-code ``get_data`` on readers (or maybe ``pull``, paired with ``push`` on producers).
Maybe rename the C++ facade to match if we pick ``pull``, so Python and C++ stay aligned without
per-tracker overrides.

If a prospective shape makes the templates hard to follow, leaving that tracker hand-written is the
correct outcome — the generator exists to remove mechanical duplication, not to model every device.
