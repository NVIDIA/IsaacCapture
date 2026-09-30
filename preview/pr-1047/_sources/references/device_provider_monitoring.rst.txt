.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Device provider monitoring
==========================

``TeleopSession`` exposes a read-only snapshot of the providers and devices in
the active session. A *provider* is the component that owns and reports a
device, such as the live OpenXR runtime or a configured plugin with device
monitoring support. Configured plugins without monitoring support continue to
launch as before but are not included in the monitoring inventory.

Providers and devices remain in the status inventory for the lifetime of the
session, including after provider failure or shutdown.

Reading status
--------------

The getters return cached immutable values and perform no device, process, or
OpenXR I/O:

.. code-block:: python

   from isaaccapture.teleop_session_manager import DeviceState

   with TeleopSession(config) as session:
       while True:
           session.step()
           snapshot = session.get_status()
           for device in snapshot.devices:
               if device.status not in (DeviceState.CONNECTED, DeviceState.DISABLED):
                   print(device.id, device.status.value, device.reason.value, device.error)

           provider = session.get_provider_status("plugin/my_plugin")
           device = session.get_device_status("my_plugin/my/device")

``get_provider_status()`` and ``get_device_status()`` return ``None`` for an
unknown ID. ``StatusSnapshot.updated_at_ns`` is the monotonic time when the
current snapshot was assembled; it is not an observation-history field.

Identity and inventory
----------------------

- A monitored plugin provider ID is ``plugin/<plugin_root_id>``.
- A monitored plugin device ID is
  ``<plugin_root_id><manifest device path>``.
- The live OpenXR provider is ``openxr/runtime`` and owns
  ``openxr/headset``.
- Replay sessions expose neither a live OpenXR provider nor configured plugin
  providers.

Plugin device descriptors come from ``PluginInfo.devices`` in ``plugin.yaml``.
Unknown paths in a status report never create devices. The identity rules form
the integration contract for plugins that support monitoring.

Provider states
---------------

``INITIALIZING``
   The plugin is being resolved or launched, or the owned OpenXR session is
   being created. Every owned device is ``INITIALIZING``.

``AVAILABLE``
   The plugin process is running, or the OpenXR runtime and session are
   operational. This does not imply that an owned physical device is connected.

``FAILED``
   Provider startup failed, a plugin exited or was signaled, process observation
   failed, DeviceIO update failed, or the OpenXR runtime/session was lost.
   Every owned device is ``UNAVAILABLE``.

``STOPPED``
   The provider was intentionally stopped during session teardown. Every owned
   device is ``UNAVAILABLE`` without treating teardown as a failure.

Device states
-------------

``INITIALIZING``
   The provider is still initializing.

``UNKNOWN``
   The provider is available, but no authoritative current conclusion exists.
   This includes absent, stale, malformed, incomplete, or unsupported reports.

``CONNECTED`` / ``DISCONNECTED``
   The available provider explicitly reports that the device is present or
   absent.

``DEGRADED``
   The device is connected but has a recoverable impairment, partial
   functionality, or active recovery.

``FAILED``
   The provider remains available but reports a device-specific nonrecoverable
   failure.

``UNAVAILABLE``
   Provider failure or teardown prevents current device observation. Provider
   precedence overrides any earlier device report.

``DISABLED``
   The manifest declares the capability, but startup arguments intentionally
   excluded it for this plugin instance.

Device state records the current condition; its reason records the cause and
must not restate the state. The error field carries optional human-readable detail
rather than another reason code.

Plugin status transport
-----------------------

A monitored plugin publishes complete ``PluginDeviceStatusSnapshot``
FlatBuffers through a
``PluginDeviceStatusPublisher`` and the existing OpenXR ``SchemaPusher``
transport. The collection ID is derived from launcher-managed metadata as
``<plugin_root_id>/device_status``. The schema version is 1, unchanged state is
republished every second, and a report is stale after three seconds.

Every snapshot must contain each manifest path exactly once. Duplicate,
unknown, missing, malformed, or unsupported entries make the provider's
declared devices ``UNKNOWN`` until a valid current snapshot arrives.

.. code-block:: cpp

   using plugin_utils::PluginDeviceStatusEntry;
   using plugin_utils::PluginDeviceStatusPublisher;

   auto extensions = PluginDeviceStatusPublisher::get_required_extensions();
   core::OpenXRSession session("MyPlugin", extensions);
   PluginDeviceStatusPublisher status(session.get_handles(), plugin_root_id);

   status.publish_if_changed(
       {PluginDeviceStatusEntry{
           .path = "/my/device",
           .state = core::PluginDeviceState_CONNECTED,
           .reason = core::PluginDeviceReason_NONE,
       }},
       monotonic_time_ns);

Plugin integrations should follow the same contract: publish from the
plugin's normal update thread rather than an SDK callback thread. Status
transport must not change ordinary data publication, reconnection, or tracker
behavior.

OpenXR ownership
----------------

For an owned OpenXR session, Isaac Teleop polls lifecycle events and HMD
availability. Runtime or session loss fails the provider; form-factor
unavailability disconnects the headset while leaving the runtime provider
available.

For externally supplied ``oxr_handles``, Isaac Teleop does not consume the
external owner's event queue. The provider is available after setup and the
headset remains ``UNKNOWN`` unless the owner supplies health through another
integration.
