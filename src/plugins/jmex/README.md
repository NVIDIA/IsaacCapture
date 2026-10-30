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

1. **MOXI Receiver SDK** 0.8.0 or newer. It is Apache-2.0 but not a public download: request it,
   with MOXI Player and its license key, through the
   [AgileMaster product page](https://jmex.com.tw/agilemaster/).
2. **MOXI Player for Robot**, with the **standard human skeleton** loaded, on this machine or
   another on the same subnet. Only MOXI Player for Robot works with this plugin.
3. **A CloudXR runtime.** It is required even though no headset is.

## Building

Unpack the SDK into this directory as `MOXIReceiverSDK/`, so that
`src/plugins/jmex/MOXIReceiverSDK/sdk/linux-x64/lib/cmake/MOXIReceiverSDK/` exists. No build flags
are then needed:

```bash
cmake -B build
cmake --build build --parallel
cmake --install build
```

To keep the SDK elsewhere, point `JMEX_SDK_ROOT` (a CMake or environment variable) or
`CMAKE_PREFIX_PATH` at the SDK's platform directory, the one containing `lib/cmake/MOXIReceiverSDK`:

```bash
cmake -B build -DJMEX_SDK_ROOT=/path/to/MOXIReceiverSDK-0.8.0/sdk/linux-x64
```

The configure output names the SDK it found: `jmex plugin: MOXIReceiverSDK <version> from <dir>`.
Without one, the plugin is skipped with `Skipping jmex plugin build:` and the rest of the tree still
builds. The project's own build prerequisites still apply; see
[Build from Source](../../../docs/source/getting_started/build_from_source/index.rst).

## Running

1. **Start the CloudXR runtime** in a terminal of its own and leave it running. Start it from a
   Python that has this repository's `isaaccapture` — `uv pip install -e ./examples/jmex` puts it
   there:

   ```bash
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
   # Defaults: channel 1, collection id "jmex_full_body"
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
| `--channel=` | `1` | Player streams on another channel. Its file replay is fixed on 255 |
| `--collection-id=` | `jmex_full_body` | More than one of these plugins runs on one machine. Consumers must name the same id |
| `--bind-ip=` | every interface | Rarely. It narrows only the TCP server, so a Player answering from an interface it leaves out cannot connect |
| `--broadcast-ip=` | `255.255.255.255` | Another MOXI Player on the subnet would otherwise contend for the same channel |
| `--tcp-port=` | `7000` | A second plugin on this machine; with the default, the second one to start finds the port taken |
| `--udp-start-port=` | `10100` | A firewall rule needs predictable data ports. Channel N receives on this + 1 + N |

One plugin receives one channel. A second plugin on this machine needs only its own `--tcp-port=`.
Every flag takes its value after `=`. An argument the plugin does not know, or a value it cannot
use, stops it with exit code 1 and names the argument.

```bash
./install/plugins/jmex/jmex_plugin --channel=255 --bind-ip=192.0.2.10 --broadcast-ip=192.0.2.10
```

### Choosing the device clock

Each sample's common clock is the moment the plugin picked the frame up. `--device-clock=` chooses
what goes in its raw device clock:

| `--device-clock=` | Raw device clock |
| --- | --- |
| `device` (default) | The time MOXI Player stamped on the frame, in nanoseconds since the Unix epoch on Player's machine |
| `local` | The same pickup time as the common clock |

`JMEX_DEVICE_CLOCK` sets the same from the environment, for a launcher that cannot add plugin
arguments; the flag wins when both are given. Any other value stops the plugin at startup.

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

While bringing a device up, starting the plugin by hand is easier: it can be restarted without
restarting the simulator, and its log stays separate.

### Logs and stopping

Started by hand, the plugin logs to stderr and to a file under `/tmp/isaaccapture-<uid>/logs/`
(`ISAACCAPTURE_LOG_DIR` moves it). Started by `TeleopSession`, its log lines go to that session's
console and log file, and anything else the process prints goes to the session's capture file in the
same directory, `<timestamp>.isaaccapture.<pid>.native.log`.

`Ctrl-C` or `SIGTERM` stops it. Once it is waiting for Player it stops cleanly, and the last line
is the run's counters, starting `stopping -- published`. Before it reaches that state it exits at
once, without the counters. If the CloudXR runtime restarts underneath it, the plugin reconnects
within about 23 seconds and carries on; frames sent during the outage are lost. A runtime that
does not come back in that time ends the run with exit code 1.

## Consuming the data

Wire a `FullBodySource` into your retargeting graph with the `body.jmex` vendor:

```python
from isaaccapture.deviceio_session import TrackerVendor
from isaaccapture.retargeting_engine.deviceio_source_nodes import FullBodySource

source = FullBodySource(
    name="agilemaster",  # this node's name in your graph
    vendor=TrackerVendor("body.jmex"),
)
```

The vendor reads `jmex_full_body` unless told otherwise. A plugin started with another
`--collection-id=` needs the consumer to name the same id. A mismatch raises no error anywhere; the
source simply never delivers a sample:

```python
TrackerVendor("body.jmex", {"collection_id": "mysuit_full_body"})
```

| Item | Value |
| --- | --- |
| Collection | `jmex_full_body`, or whatever `--collection-id=` names |
| Schema | `FullBodyPose` |
| Vendor | `body.jmex` |
| Consumer | `FullBodySource`, or a `FullBodyTracker` with that vendor |
| Contents | The 24 `BodyJoint` slots: position in meters and orientation, in OpenXR axes |

- With the standard human skeleton loaded, all 24 slots arrive on every frame. Any other model fills
  only the slots whose joint names match; on a humanoid URDF, typically the knees and elbows.
- A slot with no joint, or with no usable value in a frame, goes out with `is_valid` false.
  `all_joint_poses_tracked` is true only when all 24 are valid.
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
| Stuck at `Waiting for MOXI Player to pair` | Player is streaming, and **on the same channel**: a mismatch looks exactly like Player not running. With two hosts, check they share a subnet and that the wireless access point does not isolate clients |
| Stuck at `OpenXR HMD form factor is unavailable; waiting for a system...` | The runtime hands out no system. Use a concrete device profile such as the default `Quest3`: an `auto-*` profile waits for a client to connect, and nothing times that wait out |
| `Environment variable NV_CXR_RUNTIME_DIR is not set` | `~/.cloudxr/run/cloudxr.env` has not been sourced in this terminal |
| `Failed to create OpenXR instance: -51` | The environment is loaded, but the runtime it names is not running |
| `OpenXR session unrecoverable after` … | The runtime went away and did not come back within about 23 seconds. Start it again, then the plugin |
| `Skipping jmex plugin build:` at configure time | No SDK 0.8.0 or newer in `MOXIReceiverSDK/`, and neither `JMEX_SDK_ROOT` nor `CMAKE_PREFIX_PATH` points at one. An older SDK is skipped the same way |
| `no joint on this channel is named for a body slot` | Player has a model loaded whose joint names match no slot. Load the standard human skeleton |
| Fewer than 24 of 24 slots filled at pairing | Player has another model loaded. The lines after the count name each empty slot |
| A consumer never receives a sample while the plugin is paired | The consumer's `collection_id` is not the one `--collection-id=` names |

Everything before the channel — the suit, the gloves, the dongle, Player's setup and calibration —
is covered by Player's own README and glove guide.

## License

The source files here are Apache-2.0, as is the MOXI Receiver SDK they link against. This project
does not redistribute the SDK; it is obtained from j-mex. MOXI Player is a separate proprietary
application under its own terms.
