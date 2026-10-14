Device Trackers
===============

Trackers (defined in :code-dir:`src/core/deviceio_trackers`) are the consumer-side API for reading device
data from an active :code-file:`DeviceIOSession <src/core/deviceio_session/cpp/inc/deviceio_session/deviceio_session.hpp>`.
Each tracker manages one logical device, queries the OpenXR runtime every frame,
and exposes the latest sample through typed ``get_*()`` accessors.

There are two categories of trackers:

**OpenXR-direct trackers** -- read pose and input data through standard OpenXR
APIs (``xrLocateSpace``, ``xrSyncActions``, etc.):

- :code-file:`HeadTracker <src/core/deviceio_trackers/cpp/inc/deviceio_trackers/head_tracker.hpp>` -- HMD head pose
- :code-file:`HandTracker <src/core/deviceio_trackers/cpp/inc/deviceio_trackers/hand_tracker.hpp>` -- articulated hand joints (left and right)
- :code-file:`ControllerTracker <src/core/deviceio_trackers/cpp/inc/deviceio_trackers/controller_tracker.hpp>` -- controller poses and button/axis inputs (left and right)
- :code-file:`FullBodyTracker <src/core/deviceio_trackers/cpp/inc/deviceio_trackers/full_body_tracker.hpp>` -- vendor-agnostic 24-joint full body pose; default vendor reads the PICO ``XR_BD_body_tracking`` extension (see `Vendor Selection`_)

**SchemaTracker-based trackers** -- create new device type by defining a FlatBuffer schema and
reading it from OpenXR tensor collections via the
:code-file:`SchemaTracker <src/core/live_trackers/cpp/inc/live_trackers/schema_tracker.hpp>` utility.

- :code-file:`FrameMetadataTrackerOak <src/core/deviceio_trackers/trackers.toml>` -- frame metadata for one OAK camera stream (generated)
- :code-file:`Generic3AxisPedalTracker <src/core/deviceio_trackers/trackers.toml>` -- foot pedal axis values (generated)
- :code-file:`JointStateTracker <src/core/deviceio_trackers/trackers.toml>` -- named joint-space device state (leader arms, exoskeletons, gloves, ...) (generated)
- :code-file:`Se3Tracker <src/core/deviceio_trackers/trackers.toml>` -- generic SE3 (6-DoF) pose sources (tracker pucks, mocap rigid bodies, logical trackers) (generated)

All trackers follow the same lifecycle:

1. Construct the tracker.
2. Pass it (along with any other trackers) to ``DeviceIOSession::run()``.
3. Call ``session.update()`` each frame.
4. Read data with the tracker's ``get_*()`` method.

.. note::

    The ``DeviceIOSession`` is considered a low-level API. In practice, it is recommended to
    use the :doc:`../getting_started/teleop_session` to manage a teleop session with multiple
    device trackers and retargeters to work together.

.. _data-schema-convention:

Data Schema Convention
----------------------

Every tracker's data is defined by a FlatBuffers schema under
:code-dir:`src/core/schema/fbs`. Each schema follows a two-tier convention:

.. code-block:: idl

   // 1. Payload table -- the actual data, and what trackers hand to consumers.
   table Xxx {
       field_a: SomeType (id: 0);
       field_b: AnotherType (id: 1);
   }

   // 2. Record wrapper -- used as the MCAP recording root type.
   //    Adds a DeviceDataTimestamp alongside the payload.
   table XxxRecord {
       data: Xxx (id: 0);
       timestamp: DeviceDataTimestamp (id: 1);
   }

.. _tracked-sub-channel:

The ``_tracked`` recording sub-channel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A tracker that reads from a tensor collection can see **several samples per
frame**, and it records both views of that: every sample goes to ``<channel>``,
while only the final sample of each ``update()`` -- the one the live consumer
actually observed -- goes to ``<channel>_tracked``. Both carry the same
``XxxRecord`` root type; they differ only in which samples reach them.

Replay reads ``<channel>_tracked`` **exclusively**, so that a replayed session
yields exactly the values the live session did rather than the intermediate
samples. The per-sample channel is there for offline analysis.

Both names come from :code-file:`src/core/deviceio_trackers/defaults.toml` and
apply to every generated pull tracker:

.. code-block:: toml

   mcap_channels = ["%channel%", "%channel%_tracked"]
   replay_channels = ["%channel%_tracked"]

A manifest entry that overrides ``mcap_channels`` must keep the ``_tracked``
entry and list it in ``replay_channels``. Recording still succeeds without it,
but the resulting file cannot be replayed.

   root_type XxxRecord;

- **Payload table** (e.g. ``HeadPose``, ``HandPose``, ``ControllerSnapshot``) --
  contains the device-specific fields. All fields are present whenever the table
  itself is present.

- **Record wrapper** (e.g. ``HeadPoseRecord``) -- wraps the payload plus a
  ``DeviceDataTimestamp``. This is the ``root_type`` written to MCAP channels by
  the recorder.

Reading a payload
~~~~~~~~~~~~~~~~~

The ``get_*()`` accessors hand out the payload table itself as an owning handle
over the encoded bytes -- ``Serialized<HeadPose>`` in C++
(:code-file:`src/core/schema/cpp/inc/schema/serialized.hpp`), a read-only view
class (``HeadPose``) in Python. Reads go straight into the buffer, so there is no
unpack step and joint arrays come back as zero-copy NumPy views.

An **empty handle is the absent payload**: the device is inactive, no sample has
arrived yet, or replay hit a gap. Test it with ``if (handle)`` in C++; in Python
the accessor returns ``None``.

Each ``session.update()`` publishes a *new* buffer rather than refilling the
previous one, so a handle read this frame keeps its values after the next update.

To build a payload from Python, pass every field to its constructor -- the view
classes expose no setters.

.. warning::

   Immutability is a **contract, not an enforcement**. The joint-array properties
   hand out *writable* NumPy views, because NumPy cannot export a read-only array
   over DLPack before 2.1. Writing through one changes what every holder of that
   buffer sees, including handles read on earlier frames. Copy first if you mean
   to modify.

.. note::

   ``MessageChannelMessagesTracked`` wraps its payload in a table, because that
   payload is a **list** and something has to hold the vector. Once a tracker has
   run one ``update()`` the handle is non-empty for the rest of the session, and
   an empty ``data`` vector -- not an empty handle -- means no messages arrived
   this frame. Before that first update the handle is empty like any other.

Shared Types
~~~~~~~~~~~~

**DeviceDataTimestamp** (:code-file:`src/core/schema/fbs/timestamp.fbs`)

All timestamp fields are ``int64`` nanoseconds.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Field
     - Description
   * - ``available_time_local_common_clock``
     - System monotonic time when the sample became available to the recording
       system. Useful for measuring pipeline latency.
   * - ``sample_time_local_common_clock``
     - System monotonic time when the sample was captured. Enables
       cross-device synchronization (values from different devices share the
       same clock domain).
   * - ``sample_time_raw_device_clock``
     - Timestamp from the device's own clock. Values from different devices
       are **not** directly comparable.

**Pose** (:code-file:`src/core/schema/fbs/pose.fbs`)

.. code-block:: idl

   struct Point      { x: float; y: float; z: float; }
   struct Quaternion  { x: float; y: float; z: float; w: float; }
   struct Pose {
     position: Point;       // meters
     orientation: Quaternion;
   }

.. _tracker-reference:

Tracker Reference
-----------------

HeadTracker
~~~~~~~~~~~

Tracks the HMD head pose via the OpenXR view space.

- Schema: :code-file:`src/core/schema/fbs/head.fbs`
- C++ header: ``#include <deviceio_trackers/head_tracker.hpp>``
- Python import: ``from isaacteleop.deviceio import HeadTracker``
- Record channels: ``head`` | MCAP schema: ``core.HeadPoseRecord``
- Tests:

  - :code-file:`tests/cpp/core/schema/test_head.cpp`
  - :code-file:`tests/python/core/schema/test_head.py`

- Examples:

  - :code-file:`examples/oxr/cpp/oxr_simple_api_demo.cpp`
  - :code-file:`examples/oxr/python/modular_example.py`

HandTracker
~~~~~~~~~~~

Tracks articulated hand joints (26 joints per hand, following the OpenXR
``XrHandJointEXT`` ordering) using the ``XR_EXT_hand_tracking`` extension.

- Schema: :code-file:`src/core/schema/fbs/hand.fbs`
- C++ header: ``#include <deviceio_trackers/hand_tracker.hpp>``
- Python import: ``from isaacteleop.deviceio import HandTracker``
- Record channels: ``left_hand``, ``right_hand`` | MCAP schema: ``core.HandPoseRecord``
- Tests:

  - :code-file:`tests/cpp/core/schema/test_hand.cpp`
  - :code-file:`tests/python/core/schema/test_hand.py`
  - :code-file:`examples/oxr/python/test_synthetic_hands.py`

- Examples:

  - :code-file:`examples/oxr/cpp/oxr_simple_api_demo.cpp`
  - :code-file:`examples/oxr/python/modular_example.py`
  - :code-file:`examples/retargeting/python/isaacteleop_examples/retargeting/sources_example.py`

ControllerTracker
~~~~~~~~~~~~~~~~~

Tracks both left and right controllers -- grip and aim poses, plus button and
axis inputs. Uses standard OpenXR action bindings.

- Schema: :code-file:`src/core/schema/fbs/controller.fbs`
- C++ header: ``#include <deviceio_trackers/controller_tracker.hpp>``
- Python import: ``from isaacteleop.deviceio import ControllerTracker``
- Record channels: ``left_controller``, ``right_controller`` | MCAP schema: ``core.ControllerSnapshotRecord``
- Tests:

  - :code-file:`tests/cpp/core/schema/test_controller.cpp`
  - :code-file:`tests/python/core/schema/test_controller.py`
  - :code-file:`examples/oxr/python/test_controller_tracker.py`

- Examples:

  - :code-file:`examples/retargeting/python/isaacteleop_examples/retargeting/sources_example.py`
  - :code-file:`examples/teleop/python/locomotion_retargeting_example.py`
  - :code-file:`examples/teleop/python/gripper_retargeting_example_simple.py`

FullBodyTracker
~~~~~~~~~~~~~~~

Tracks 24 body joints through a vendor-selected backend. The tracker itself is
a vendor-agnostic marker and carries no vendor or live/replay state: a live
session picks the backend via ``VendorConfig`` (see `Vendor Selection`_), and
replay reads the recorded ``full_body`` channel regardless of which vendor
produced it. When no vendor is selected, the default vendor ``body.pico-xr``
reads the PICO ``XR_BD_body_tracking`` extension directly.

- Schema: :code-file:`src/core/schema/fbs/full_body.fbs`
- C++ header: ``#include <deviceio_trackers/full_body_tracker.hpp>``
- Python import: ``from isaacteleop.deviceio import FullBodyTracker``
- Record channels: ``full_body`` | MCAP schema: ``core.FullBodyPoseRecord``
- Tests:

  - :code-file:`tests/cpp/core/schema/test_full_body.cpp`
  - :code-file:`tests/python/core/schema/test_full_body.py`
  - :code-file:`examples/oxr/python/test_full_body_tracker.py`

- Examples:

  - :code-file:`examples/schemaio/full_body_printer.cpp`
  - :code-file:`examples/mcap_record_replay/cpp/record_full_body.cpp`
  - :code-file:`examples/mcap_record_replay/python/live_full_body.py`
  - :code-file:`examples/mcap_record_replay/python/record_full_body.py`
  - :code-file:`examples/mcap_record_replay/python/replay_full_body.py`

.. note::

   ``FullBodyTrackerPico`` remains available as a deprecated alias for
   ``FullBodyTracker`` so existing scripts run unchanged.

FrameMetadataTrackerOak
~~~~~~~~~~~~~~~~~~~~~~~

Per-frame metadata for a **single** OAK camera stream. Create one tracker per
stream, passing the tensor collection the plugin publishes that stream under --
``{collection_prefix}/{StreamName}``, e.g. ``"oak_camera/Color"``. Uses the
:code-file:`SchemaTracker <src/core/live_trackers/cpp/inc/live_trackers/schema_tracker.hpp>`
utility internally.

- Schema: :code-file:`src/core/schema/fbs/oak.fbs`
- Manifest: :code-file:`src/core/deviceio_trackers/trackers.toml` (``frame_metadata_oak``)
- C++ header: ``#include <deviceio_trackers/frame_metadata_tracker_oak.hpp>``
- Python import: ``from isaacteleop.deviceio import FrameMetadataTrackerOak``
- Record channels: ``oak``, ``oak_tracked`` | MCAP schema: ``core.FrameMetadataOakRecord``
- Tests:

  - :code-file:`tests/cpp/core/schema/test_oak.cpp`
  - :code-file:`tests/python/core/schema/test_camera.py`
  - :code-file:`examples/oxr/python/test_oak_camera.py`

- Examples:

  - :code-file:`examples/schemaio/frame_metadata_printer.cpp`

Generic3AxisPedalTracker
~~~~~~~~~~~~~~~~~~~~~~~~

Reads foot pedal axis values pushed by a device plugin through OpenXR tensor
collections. Uses the :code-file:`SchemaTracker <src/core/live_trackers/cpp/inc/live_trackers/schema_tracker.hpp>`
utility internally.

- Schema: :code-file:`src/core/schema/fbs/pedals.fbs`
- C++ header: ``#include <deviceio_trackers/generic_3axis_pedal_tracker.hpp>``
- Python import: ``from isaacteleop.deviceio import Generic3AxisPedalTracker``
- Record channels: ``pedals``, ``pedals_tracked`` | MCAP schema: ``core.Generic3AxisPedalOutputRecord``
- Tests:

  - :code-file:`tests/cpp/core/schema/test_pedals.cpp`
  - :code-file:`tests/python/core/schema/test_pedals.py`

- Examples:

  - :code-file:`examples/schemaio/pedal_printer.cpp`
  - :code-file:`examples/teleop/python/foot_pedal_locomotion_example.py`

.. note::

   The Python method is named ``get_pedal_data()`` (instead of the C++
   ``get_data()``).

.. _vendor-selection:

Vendor Selection
----------------

Some trackers are **vendor-agnostic markers**: the tracker declares *what*
device data it represents, while a live session chooses *which* backend
("vendor") produces that data. This mirrors how live-vs-replay is chosen at the
session level -- the same tracker instance works across vendors and across live
and replay. ``FullBodyTracker`` is currently the only vendored tracker; its
default vendor ``body.pico-xr`` reads the PICO ``XR_BD_body_tracking``
extension.

Select a vendor by passing a ``VendorConfig`` to both
``DeviceIOSession.get_required_extensions()`` and ``DeviceIOSession.run()``. A
``VendorConfig`` maps tracker instances to a ``TrackerVendor(id, params)``,
where ``id`` selects the backend from the live factory's vendor registry and
``params`` carries free-form string key/value options for it. Trackers left out
of the config use their default vendor. Vendor selections on non-vendored
trackers, and unknown vendor ids, are rejected at session construction.

.. code-block:: python

   import isaacteleop.deviceio as deviceio

   body = deviceio.FullBodyTracker()

   # Select the backend for the vendored tracker (default shown explicitly).
   vendor_config = deviceio.VendorConfig([
       (body, deviceio.TrackerVendor("body.pico-xr")),
   ])

   required_extensions = deviceio.DeviceIOSession.get_required_extensions(
       [body], vendor_config
   )
   with deviceio.DeviceIOSession.run(
       [body], handles, None, vendor_config
   ) as session:
       ...

Replay is always vendor-neutral: the replay full-body impl reads the recorded
``full_body`` channel regardless of which live vendor produced it, so
``VendorConfig`` applies to live sessions only. The vendor registry is open for
additional pre-built plugin vendors without changing the tracker marker.

When driving devices through the higher-level teleop session manager, vendor
selection is carried on the DeviceIO source itself via its ``vendor`` argument
(e.g. ``FullBodySource(name="full_body", vendor=deviceio.TrackerVendor("body.pico-xr"))``),
so it travels with the pipeline into both extension discovery and session
construction; see :doc:`../getting_started/teleop_session`.

.. _tracker-usage-example:

Usage Examples
--------------

For end-to-end usage patterns combining trackers with a ``DeviceIOSession``, see:

- **C++**: :code-file:`examples/oxr/cpp/oxr_simple_api_demo.cpp`
- **Python**: :code-file:`examples/oxr/python/modular_example.py`

For higher-level usage with the teleop session manager and retargeting, see:

- :code-file:`examples/retargeting/python/isaacteleop_examples/retargeting/sources_example.py`
- :code-file:`examples/teleop/python/gripper_retargeting_example_simple.py`
- :code-file:`examples/teleop/python/locomotion_retargeting_example.py`
