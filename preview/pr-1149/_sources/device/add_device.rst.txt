.. _device-interface-device-plugin:

Add a New Device
================

For new hardware that requires a custom driver or SDK, create an Isaac Teleop device plugin
(C++ level). Plugins push data via OpenXR tensor collections. Existing plugins include Manus
gloves, OAK-D camera, Haptikos exoskeletons, controller synthetic hands, and foot pedals. After
creating the plugin, update the retargeting pipeline config to consume data from the new plugin's
source node. See the `Plugins directory <https://github.com/NVIDIA/IsaacTeleop/tree/main/src/plugins/>`_
for examples.

To add a new device that streams typed data over the OpenXR runtime, follow these four steps.
The reference implementation is the **generic 3-axis foot pedal**:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Component
     - Code Location
   * - Data Schema
     - :code-file:`src/core/schema/fbs/pedals.fbs`
   * - Device Plugin
     - :code-dir:`src/plugins/generic_3axis_pedal`
   * - Tracker facade (``Generic3AxisPedalTracker``)
     - Generated at configure time from :code-file:`src/core/deviceio_trackers/trackers.toml`
       (see :ref:`Schema-based tracker manifest <schema-based-tracker-manifest>`)
   * - Live backend (``LiveGeneric3AxisPedalTrackerImpl``)
     - Generated alongside the facade under ``${CMAKE_BINARY_DIR}/generated/trackers/``
   * - ``SchemaTracker`` / ``SampleResult``
     - :code-file:`src/core/live_trackers/cpp/inc/live_trackers/schema_tracker.hpp`
   * - Debug printer
     - :code-file:`examples/schemaio/pedal_printer.cpp`


Step 1: Define the data schema
------------------------------

Define a FlatBuffer schema (`.fbs`) under :code-dir:`src/core/schema/fbs`. The schema drives both
serialization in the plugin and deserialization in the tracker; pusher and reader must agree
on the same schema ahead of time (the schema is not sent over the wire).

Reference schema: :code-file:`src/core/schema/fbs/pedals.fbs`

.. code-block:: idl
   :class: code-100col

   include "point.fbs";
   include "timestamp.fbs";

   namespace core;

   table Generic3AxisPedalOutput {
     left_pedal: float (id: 0);
     right_pedal: float (id: 1);
     rudder: float (id: 2);
   }

   table Generic3AxisPedalOutputRecord {
     data: Generic3AxisPedalOutput (id: 0);
     timestamp: DeviceDataTimestamp (id: 1);
   }

- **Output table** — The primary payload type (e.g. ``Generic3AxisPedalOutput``) with the
  device fields. This is what the plugin serializes and pushes.
- **Record wrapper** — A table that wraps the output plus ``DeviceDataTimestamp``
  (e.g. ``Generic3AxisPedalOutputRecord``). This is the root type written to MCAP channels
  by the recorder.
- **root_type** — Set to the Record type (e.g. ``root_type Generic3AxisPedalOutputRecord;``).

Include ``timestamp.fbs`` for ``DeviceDataTimestamp``; include other shared types (e.g.
``point.fbs``) as needed. After adding or changing a schema, rebuild so that the C++ and
Python generated code (e.g. ``pedals_generated.h``, ``pedals_bfbs_generated.h``) is updated.

Step 2: Implement a device plugin
---------------------------------

The plugin runs in a separate process (or as part of a host app), reads hardware, and pushes
serialized FlatBuffer data via OpenXR using the **SchemaPusher** from the ``pusherio`` library.
Reuse the same pattern as the example apps in ``examples/schemaio/`` (e.g. ``pedal_pusher``
which uses ``SchemaPusher``).

- **OpenXR session** — Create an ``OpenXRSession`` with extensions from
  ``SchemaPusher::get_required_extensions()`` (includes ``XR_NVX1_push_tensor`` and
  ``XR_NVX1_tensor_data``).
- **SchemaPusher** — Construct a ``SchemaPusher`` with ``OpenXRSessionHandles`` and a
  ``SchemaPusherConfig``: ``collection_id``, ``max_flatbuffer_size``, ``tensor_identifier``,
  ``localized_name``, and optionally ``app_name``. The ``collection_id`` and
  ``tensor_identifier`` must match what the tracker uses.
- **Push loop** — In your update loop, fill the schema's native type (e.g.
  ``Generic3AxisPedalOutputT``), serialize it with ``FlatBufferBuilder`` and the generated
  ``Pack()``, then call ``pusher_.push_buffer(ptr, size, sample_time_local_common_clock_ns,
  sample_time_raw_device_clock_ns)``. Use a monotonic clock (e.g. ``core::os_monotonic_now_ns()``)
  for the local common clock; use the device's own clock for the raw device clock if available.

Reference implementation: :code-dir:`src/plugins/generic_3axis_pedal`. The plugin holds a
``core::SchemaPusher`` member, opens the Linux joystick device, maps axes to
``Generic3AxisPedalOutputT``, and calls ``push_current_state()`` from ``update()``. See
:code-file:`generic_3axis_pedal_plugin.hpp <src/plugins/generic_3axis_pedal/generic_3axis_pedal_plugin.hpp>` and
:code-file:`generic_3axis_pedal_plugin.cpp <src/plugins/generic_3axis_pedal/generic_3axis_pedal_plugin.cpp>`.

Step 3: Implement a tracker
----------------------------

.. _schema-based-tracker-manifest:

Schema-based trackers (manifest)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the device is a **schema-based FlatBuffer tracker** over OpenXR tensor collections (plugin
serializes a table; the host reads or writes it via ``SchemaTracker`` / ``SchemaPusher`` with no
``xrLocate*`` calls and no custom connection state), you usually **do not** hand-write the
seven-layer tracker stack. Instead:

1. Add or extend the FlatBuffer schema (Step 1).
2. Add a ``[[tracker]]`` entry to :code-file:`src/core/deviceio_trackers/trackers.toml`. Defaults
   in :code-file:`src/core/deviceio_trackers/defaults.toml` expand ``%name%``, ``%name_CamelCase%``,
   MCAP channel names, and class names; override only genuine exceptions (``se3_tracker`` uses
   ``class = "Se3Tracker"``; pedals use ``schema = "pedals"`` and ``channel = "pedals"``).
3. Reconfigure/rebuild. ``cmake/GenerateTrackers.cmake`` runs
   :code-file:`src/core/codegen/generate_trackers.py` and wires generated sources into
   ``deviceio_trackers``, ``live_trackers``, and ``replay_trackers``.

Use ``direction = "push"`` when Teleop **pushes** a typed table to a plugin (e.g.
``HapticCommandPushTracker``). Cross-process **consumers** that bucket multiple
``HapticCommand.endpoint`` samples on one collection (e.g. ``HapticCommandReaderTracker`` for Manus
haptics) stay hand-written — see :doc:`../references/generated_trackers`.

For what the generator emits, which trackers it covers today, and which shapes are not supported
yet, see :doc:`../references/generated_trackers`.

Diagnose surprising defaults with::

   python src/core/codegen/generate_trackers.py \\
     --manifest src/core/deviceio_trackers/trackers.toml \\
     --defaults src/core/deviceio_trackers/defaults.toml \\
     --out-dir /tmp/isaac-teleop-trackers \\
     --emit-cmake /tmp/isaac-teleop-trackers.cmake \\
     --print-resolved

(``--out-dir`` and ``--emit-cmake`` are required by the CLI even when ``--print-resolved``
skips writing generated sources; temporary paths are fine.)

Hand-written trackers
~~~~~~~~~~~~~~~~~~~~~

Trackers that are not schema-based FlatBuffer readers/writers (OpenXR ``xrLocate*``, opaque message
channels, multi-endpoint haptic readers, …) still use the manual facade + live/replay impl pattern
below.

The tracker runs inside a consumer process (e.g. Teleop pipeline or a small reader app). It
implements the **ITracker** interface (tracker **facade** in ``deviceio_trackers``); the
**live backend** in ``live_trackers`` composes **SchemaTracker** to read raw
tensor samples from OpenXR. Implement a concrete tracker class (e.g.
``HandTracker``) that:

- **Extends ITracker** — Override ``get_name()``,
  ``get_schema_name()``, ``get_schema_text()``, and ``get_record_channels()``.
  For OpenXR, add a static ``required_extensions()`` on your live ``ITrackerImpl`` and register
  the tracker type in ``LiveDeviceIOFactory::get_required_extensions`` (tensor/schema readers
  usually forward ``SchemaTrackerBase::get_required_extensions()``). Callers use
  ``DeviceIOSession::get_required_extensions(trackers)``. Return the Record type name
  and binary schema for MCAP; return at least one channel name.
- **Holds user configuration** — Same logical inputs as the pusher (e.g. ``collection_id``,
  ``max_flatbuffer_size``). The live ``ITrackerImpl`` builds the internal tensor settings
  (``SchemaTrackerConfig``) so they match the plugin.
- **Factory registration** — Register your tracker in the live factory dispatch table
  (see ``LiveDeviceIOFactory``). The factory constructs an ``ITrackerImpl`` that holds
  a ``SchemaTracker``, builds a ``SchemaTrackerConfig`` from the tracker's stored
  configuration, and implements ``update(int64_t monotonic_time_ns)``.

In the **Impl**:

- **Construction** — Build the ``SchemaTrackerConfig`` from the tracker's configuration and
  hand it to the ``SchemaTracker``, along with the MCAP channels (or ``nullptr`` when
  recording is disabled) and the sub-channel indices to write.
- **update()** — Call ``m_schema_reader.update(m_tracked)``, where ``m_tracked`` is the
  published ``Serialized<Generic3AxisPedalOutput>`` handle. That one call reads the pending
  samples, writes each of them to MCAP when channels are attached, and publishes the final
  one. A tick with no new samples leaves the last-known handle in place; an absent
  collection empties it.
- **get_data()** — Return the published handle. See
  :ref:`Reading a payload <data-schema-convention>` for what a consumer may assume about it.

Recording needs no per-tracker serialization code: ``SchemaTracker`` writes through
``McapTrackerChannels``, which wraps the payload in the Record type with its
``DeviceDataTimestamp``. It writes two sub-channels, and replay reads only the
second -- see :ref:`tracked-sub-channel` before overriding ``mcap_channels`` in a
manifest entry.

Reference implementation — generated ``SchemaTracker`` reader and a hand-written OpenXR
locate tracker:

- **Generated schema-based reader** — after configure, under
  ``${CMAKE_BINARY_DIR}/generated/trackers/``: facade
  ``deviceio_trackers/generic_3axis_pedal_tracker.cpp`` (``Generic3AxisPedalTracker``), base
  ``deviceio_base/generic_3axis_pedal_tracker_base.hpp``, and live backend
  ``live_trackers/live_generic_3axis_pedal_tracker_impl.cpp``
  (``LiveGeneric3AxisPedalTrackerImpl``). The live impl composes ``SchemaTracker`` and uses
  ``read_all_samples()`` with ``std::vector<SchemaTrackerBase::SampleResult>``. See
  :code-file:`src/core/live_trackers/cpp/inc/live_trackers/schema_tracker.hpp` for
  ``SchemaTracker`` / ``SampleResult``.
- **Hand-written OpenXR locate tracker** — facade
  :code-file:`src/core/deviceio_trackers/cpp/hand_tracker.cpp` (``HandTracker``) and live
  backend :code-file:`src/core/live_trackers/cpp/live_hand_tracker_impl.cpp`
  (``LiveHandTrackerImpl``): same facade / ``ITrackerImpl`` split, but ``update()`` drives
  ``xrLocate*`` rather than ``SchemaTracker``.

Step 4: Implement a simple C++ printer (optional)
-------------------------------------------------

A minimal reader app verifies the full path: plugin (or pusher) pushes; printer discovers
the collection and prints samples. Pattern (see :code-file:`examples/schemaio/pedal_printer.cpp`):

1. Create the tracker (e.g. ``std::make_shared<Generic3AxisPedalTracker>(collection_id,
   max_flatbuffer_size)``).
2. Get required extensions with ``DeviceIOSession::get_required_extensions(trackers)`` and
   create an ``OpenXRSession``.
3. Create a ``DeviceIOSession`` with ``DeviceIOSession::run(trackers, oxr_session->get_handles())``.
4. Loop: call ``session->update()``, then read ``tracker->get_data(*session)``. If the
   returned handle is non-empty, use the latest sample; otherwise sleep briefly and repeat.

Use the same ``collection_id`` (and optionally ``tensor_identifier``) as the plugin. See
:ref:`Schema IO example: build and run <schema-io-example>` above for building and running
``pedal_pusher`` and ``pedal_printer``.

.. _schema-io-example:

Schema IO example: build and run
--------------------------------

The **Schema IO example** is a simpler example that demonstrates pushing and reading serialized
FlatBuffer data via the OpenXR runtime using the Generic Tensor Collection interface.
It provides two binaries: **pedal_pusher** (serializes and pushes ``Generic3AxisPedalOutput`` using
``SchemaPusher``) and **pedal_printer** (reads via ``Generic3AxisPedalTracker`` and
``DeviceIOSession``). Both use the ``XR_NVX1_push_tensor`` and ``XR_NVX1_tensor_data`` extensions.
Pusher and reader agree on the schema (``Generic3AxisPedalOutput`` from ``pedals.fbs``), so the
schema is not sent over the wire.

**Build** (from the project root, with examples enabled):

.. code-block:: bash

   cmake -B build -DBUILD_EXAMPLES=ON
   cmake --build build --parallel
   cmake --install build

**Run** pusher and printer in separate terminals:

.. code-block:: bash

   # Terminal 1: Start printer
   ./install/examples/schemaio/pedal_printer

   # Terminal 2: Start pusher
   ./install/examples/schemaio/pedal_pusher

The printer discovers the tensor collection created by the pusher and prints received samples.
Both exit after 100 samples, or press Ctrl+C to exit early.

**Components**

- **SchemaPusher** (``pusherio`` library) — Pushes serialized FlatBuffer data via OpenXR tensor
  extensions: takes externally-provided OpenXR session handles, creates a tensor collection with
  the configured identifier, provides ``push_buffer()`` for raw serialized data. Use composition
  to create typed wrappers (e.g. ``Generic3AxisPedalPusher`` in :code-file:`examples/schemaio/pedal_pusher.cpp`).
- **SchemaTracker** (``live_trackers``) — Helper for reading FlatBuffer schema data via
  OpenXR tensor collections: discovers collections by identifier, reads pending ``SampleResult``
  values, records them when MCAP channels are attached, and publishes the final one as a
  ``Serialized<...>``. Live tracker implementations (e.g. ``LiveGeneric3AxisPedalTrackerImpl``)
  compose a ``SchemaTracker`` and implement ``ITrackerImpl::update()`` on top of it.
- **Generic3AxisPedalTracker** (tracker facade in ``deviceio_trackers``) — Concrete ``ITracker`` for
  ``Generic3AxisPedalOutput``: holds configuration and
  ``get_data(session)`` returning ``Serialized<Generic3AxisPedalOutput>`` via the session’s
  ``IGeneric3AxisPedalTrackerImpl``.
- **DeviceIOSession** — Session manager: collects required OpenXR extensions from registered
  trackers, creates tracker implementations with session handles, and calls ``update()`` on all
  trackers during the update loop.
