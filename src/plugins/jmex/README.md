<!--
SPDX-FileCopyrightText: Copyright (c) 2026 j-mex. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# j-mex AgileMaster plugin

Streams a [j-mex AgileMaster](https://jmex.com.tw/agilemaster/) motion capture channel into Isaac
Capture as the operator's body pose: `FullBodyPose` under the `body.jmex` vendor, so anything that
consumes `FullBodyPose` can use it. Linux x86_64 only.

AgileMaster has three parts: the **j-mex Mocap Suit** the operator wears, **MOXI Player**, which
solves the suit onto a skeleton and streams it on a numbered channel, and the **MOXI Receiver SDK**
this plugin links against.

```text
Mocap Suit -> MOXI Player -- channel --> jmex_plugin -- "jmex_full_body" --> FullBodySource
              (standard human skeleton)                 (FullBodyPose, body.jmex)
```

## Prerequisites

1. **MOXI Receiver SDK**. It is not a public download: request it, with MOXI Player and its
   license key, through the [AgileMaster product page](https://jmex.com.tw/agilemaster/).
2. **MOXI Player for Robot**, with the **standard human skeleton** loaded, on this machine or
   another on the same subnet. Only MOXI Player for Robot works with this plugin.
3. **A CloudXR runtime.** It is required even though no headset is.

## Building

The plugin builds only with `BUILD_PLUGIN_JMEX` on. Unpack the SDK into this directory as
`MOXIReceiverSDK/`, so that `src/plugins/jmex/MOXIReceiverSDK/sdk/linux-x64/lib/cmake/MOXIReceiverSDK/`
exists, then:

```bash
cmake -B build -DBUILD_PLUGIN_JMEX=ON
cmake --build build --parallel
cmake --install build
```

To keep the SDK elsewhere, point `CMAKE_PREFIX_PATH` at the SDK's platform directory, the one
containing `lib/cmake/MOXIReceiverSDK`, or `MOXIReceiverSDK_DIR` at that `lib/cmake/MOXIReceiverSDK`
directory:

```bash
cmake -B build -DBUILD_PLUGIN_JMEX=ON -DCMAKE_PREFIX_PATH=/path/to/MOXIReceiverSDK/sdk/linux-x64
```

The configure output names the SDK it found: `jmex plugin: MOXIReceiverSDK <version> from <dir>`.
Without one, configure warns `BUILD_PLUGIN_JMEX=ON but no MOXIReceiverSDK <minimum version> or newer`,
skips the plugin, and the rest of the tree still builds. The project's own build prerequisites
still apply; see [Build from Source](https://nvidia.github.io/IsaacCapture/main/development/build_from_source/index.html).

## Running

1. **Start the CloudXR runtime** in a terminal of its own and leave it running. In a venv set up as
   in [Running an example](../../../examples/README.md#running-an-example), install the example
   once, then start the runtime:

   ```bash
   uv pip install -e ./examples/jmex
   python -m isaaccapture.cloudxr.service run
   ```

   The runtime needs a concrete device profile, and the default, `Quest3`, is one. To use another,
   hand the service an env file:

   ```bash
   echo 'NV_DEVICE_PROFILE=<profile>' > cloudxr.env
   python -m isaaccapture.cloudxr.service run --cloudxr-env-config cloudxr.env
   ```

2. **Load the runtime's environment** in every terminal you start the plugin from:

   ```bash
   source ~/.cloudxr/run/cloudxr.env
   ```

3. **Start the plugin:**

   ```bash
   # With the defaults
   ./install/plugins/jmex/jmex_plugin

   # Player's file replay is fixed on channel 255
   ./install/plugins/jmex/jmex_plugin --channel=255
   ```

   It logs `Waiting for MOXI Player to pair on channel <N>...` and waits. If Player disconnects, the
   plugin waits for it again, and Player pairs again on its own.

4. **Stream from MOXI Player** on the same channel, with the standard human skeleton loaded.
   Player's own README covers its setup and calibration. Once paired, the log reads
   `channel <N> full body filled 24 of 24 slots`.

5. **Check the stream** with no simulator in the way: with the plugin running, record a few seconds
   as under [Recording a body pose](../../../examples/jmex/README.md#recording-a-body-pose), adding
   `--no-plugin` so that the script does not start a second plugin. `joints=24/24` on the progress
   lines means frames from the standard human skeleton have arrived.

### Choosing where the receiver listens

| Flag | Default | When to set it |
| --- | --- | --- |
| `--channel=` | see `--help` | Player streams on another channel |
| `--collection-id=` | see `--help` | More than one of these plugins runs on one machine. Consumers must name the same id |
| `--bind-ip=` | every interface | Player should reach this machine on one address only; see below |
| `--broadcast-ip=` | the whole subnet | Another MOXI Player on the subnet would otherwise contend for the same channel |
| `--tcp-port=` | see `--help` | A second plugin on this machine; with the default, the second one to start finds the port taken |
| `--udp-start-port=` | see `--help` | A firewall rule needs predictable data ports. Each channel receives on its own port above this one, moved higher if that one is taken |

One plugin receives one channel. Every flag but `--help` takes its value after `=`. An argument the
plugin does not know, or a value it cannot use, stops it with exit code 1 and names the argument.
`--help` lists the flags and their defaults.

Launched by `TeleopSession`, the plugin also gets the `args` in `plugin.yaml`, which keep
`--bind-ip=` and `--broadcast-ip=` on this machine: only a MOXI Player on this machine can pair unless
`plugin_args` sets the two flags again. Started by hand, it takes the defaults, and any Player on the
subnet can.

Nothing authenticates a pairing. The channel's data port listens on every interface whatever
`--bind-ip=` says, and takes motion frames from any sender: only a firewall keeps another host from
sending poses into a paired channel. For a Player on another machine, set `--bind-ip=` to this
machine's address on the interface facing it and `--broadcast-ip=` to Player's address, and let only
Player's machine reach the TCP port and the UDP ports above `--udp-start-port=`:

```bash
./install/plugins/jmex/jmex_plugin --channel=255 --bind-ip=192.0.2.10 --broadcast-ip=192.0.2.20
```

### Choosing the device clock

Each sample's common clock is the moment the plugin picked the frame up. `--device-clock=` chooses
what goes in its raw device clock:

| `--device-clock=` | Raw device clock |
| --- | --- |
| `device` (default) | The frame's time from MOXI Player, in nanoseconds: since the Unix epoch on Player's machine when the stream carries an accuracy timestamp, otherwise derived from the frame's sequence id |
| `local` | The same pickup time as the common clock |

### Letting Isaac Teleop launch it

`TeleopSession` can start the plugin when the session starts and stop it when the session ends. Add
a `PluginConfig` that names the plugin and points at an installed tree; a plugin that has only been
built is not found:

```python
from pathlib import Path
from isaaccapture.teleop_session_manager import PluginConfig, TeleopSessionConfig

session_config = TeleopSessionConfig(
    app_name="AgileMasterTeleop",
    pipeline=pipeline,
    plugins=[
        PluginConfig(
            plugin_name="jmex",  # plugin.yaml's name
            plugin_root_id="jmex",
            search_paths=[Path("install/plugins")],
            plugin_args=["--channel=255"],  # optional; the last value of a flag wins
            required=True,  # refuse to start, rather than wait in silence, if it is not found
        )
    ],
)
```

While bringing a device up, start the plugin by hand, as under [Running](#running).

### Stopping

`Ctrl-C` or `SIGTERM` stops it. Once it is waiting for Player it stops cleanly, and the last line
is the run's counters, starting `stopping -- published`. Before it reaches that state it exits at
once, without the counters. If the CloudXR runtime restarts underneath it, the plugin retries with a
growing backoff, reconnects and carries on; frames sent during the outage are lost. A runtime that
is still gone when the retries run out ends the run with exit code 1.

## Consuming the data

Wire a `FullBodySource` into your retargeting graph with the `body.jmex` vendor:

```python
from isaaccapture.deviceio import TrackerVendor
from isaaccapture.retargeting_engine.deviceio_source_nodes import FullBodySource

source = FullBodySource(
    name="agilemaster",  # this node's name in your graph
    vendor=TrackerVendor("body.jmex"),
)
```

The vendor reads the plugin's default collection unless told otherwise. A plugin started with another
`--collection-id=` needs the consumer to name the same id. A mismatch raises no error anywhere; the
source simply never delivers a sample:

```python
TrackerVendor("body.jmex", {"collection_id": "mysuit_full_body"})
```

| Item | Value |
| --- | --- |
| Collection | The plugin's default (see `--help`), or whatever `--collection-id=` names |
| Schema | `FullBodyPose` |
| Vendor | `body.jmex` |
| Consumer | `FullBodySource`, or a `FullBodyTracker` with that vendor |
| Contents | Every `BodyJoint` slot: position in meters and orientation, in OpenXR axes |

- With the standard human skeleton loaded, every slot arrives on every frame. Any other model fills
  only the slots whose joint names match; on a humanoid URDF, typically the knees and elbows.
- A slot with no joint, or with no usable value in a frame, goes out with `is_valid` false.
  `all_joint_poses_tracked` is true only when every slot is valid.
- When Player disconnects, the plugin stops publishing but keeps the collection, so a consumer goes
  on reading the last frame until Player pairs again. The plugin logs
  `MOXI Player disconnected; waiting for it to come back` when that happens.
- The spine, foot, neck, collar and hand slots have no sensor of their own: read them for position,
  not orientation.
- The skeleton has fixed dimensions, so positions describe the operator's posture, not their size.

For a complete consumer, which drives a Unitree G1 in Isaac Lab from the body pose and records it to
MCAP, see the [example](../../../examples/jmex/README.md).

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| Stuck at `Waiting for MOXI Player to pair` | Player is streaming, and **on the same channel**: a mismatch looks exactly like Player not running. With two hosts, a plugin started by `TeleopSession` needs `--bind-ip=` and `--broadcast-ip=` in `plugin_args` ([Choosing where the receiver listens](#choosing-where-the-receiver-listens)); also check the hosts share a subnet and that the wireless access point does not isolate clients |
| Stuck at `OpenXR HMD form factor is unavailable; waiting for a system...` | The runtime hands out no system. Use a concrete device profile such as the default `Quest3`: an `auto-*` profile waits for a client to connect, and nothing times that wait out |
| `Environment variable NV_CXR_RUNTIME_DIR is not set` | `~/.cloudxr/run/cloudxr.env` has not been sourced in this terminal |
| `Failed to create OpenXR instance: -51` | The environment is loaded, but the runtime it names is not running |
| `OpenXR session unrecoverable after` … | The runtime went away and did not come back before the retries ran out. Start it again, then the plugin |
| No `install/plugins/jmex/` after `cmake --install build` | The build was configured without `-DBUILD_PLUGIN_JMEX=ON`, or found no SDK (next row) |
| `BUILD_PLUGIN_JMEX=ON but no MOXIReceiverSDK` at configure time | No SDK in `MOXIReceiverSDK/`, and neither `CMAKE_PREFIX_PATH` nor `MOXIReceiverSDK_DIR` points at one. An SDK older than the version the warning names is skipped the same way |
| `no joint on this channel is named for a body slot` | Player has a model loaded whose joint names match no slot. Load the standard human skeleton |
| Fewer than 24 of 24 slots filled at pairing | Player has another model loaded. The lines after the count name each empty slot |
| A consumer never receives a sample while the plugin is paired | The consumer's `collection_id` is not the one `--collection-id=` names |

Everything before the channel — the suit, the gloves, the dongle, Player's setup and calibration —
is covered by Player's own README and glove guide.

## License

The source files here are Apache-2.0, as is the MOXI Receiver SDK they link against. This project
does not redistribute the SDK; it is obtained from j-mex. MOXI Player is a separate proprietary
application under its own terms.
