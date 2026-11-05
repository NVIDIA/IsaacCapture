<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# OGLO Tactile Glove Plugin

C++ plugin that streams **OGLO** tactile gloves over BLE and pushes them into the
IsaacTeleop pipeline over OpenXR. Each glove (one BLE device per hand) provides
**80 tactile taxels** (5 fingers × 4×4) plus a **6-axis IMU** at 100 Hz.

It follows the standard [Add a New Device](../../../docs/source/device/add_device.rst)
pattern and is modeled on the OAK camera plugin.

## Data path

```
OGLO glove (BLE notify, packed12 v5)
  → OgloBleClient (BlueZ over libdbus)        [BLE thread]
  → PacketParser (config-driven decode)       [BLE thread]
  → queue → IGloveSink                         [single consumer thread]
       → SchemaPusher → host OgloTactileTracker → shared TeleopSession MCAP
```

The parser reads the device **Config characteristic** first and branches on the
notify `flags` byte (`0x04` packed12 v5 — primary; `0x02`/`0x01` schema-4 fallback),
so packet sizes are never hardcoded (wire spec: the OGLO firmware packed12-v5 packet format).

## Build

Linux only (BlueZ). Complete the [source-build prerequisites](https://nvidia.github.io/IsaacCapture/main/development/build_from_source/index.html)
first, then:

```bash
sudo apt install libdbus-1-dev
```

```bash
cmake -B build -DBUILD_PLUGIN_OGLO=ON
cmake --build build --target oglo_tactile_plugin --parallel
```

`nlohmann/json` (MIT) is fetched automatically via FetchContent.

> **BLE backend.** The plugin talks to BlueZ directly over **libdbus** (AFL-2.1,
> permissive) — no copyleft dependency, works out of the box on any Linux host
> with the BlueZ daemon. The transport is isolated behind the `OgloBleClient`
> interface (`ble/oglo_ble_client.hpp`), so an alternative backend can be dropped
> in via `make_ble_client()` without touching the parser, schema, or tracker.

## Usage

```bash
# Push for a host OgloTactileTracker into a shared TeleopSession MCAP
./build/src/plugins/oglo_tactile/oglo_tactile_plugin --side right --collection-prefix=oglo
./build/src/plugins/oglo_tactile/oglo_tactile_plugin --side left  --collection-prefix=oglo
```

| Option | Description |
|--------|-------------|
| `--side left\|right` | **Required.** Selects the `OGLO LEFT` / `OGLO RIGHT` device. |
| `--collection-prefix=PREFIX` | **Required.** OpenXR collection prefix (pushes `PREFIX/left`, `PREFIX/right`). |
| `--device-name=NAME` | Pin an exact advertised BLE name (multiple gloves nearby). |
| `--scan-timeout-ms=N` | BLE scan timeout (default 15000). |

## Recorded output

The plugin only pushes over OpenXR; recording happens host-side, where the
`OgloTactileTracker` reads the pushed collection into the session MCAP. Per hand it
writes channels `oglo_<side>/oglo` and `oglo_<side>/oglo_tracked`, schema
`core.OgloGloveSampleRecord` (flatbuffer). Each message carries `seq`,
`device_time_us`, `taxels[80]` (raw 12-bit, `finger,row,col`), `accel_x/y/z`,
`gyro_x/y/z` (raw IMU LSB), plus a `DeviceDataTimestamp`
(`sample_time_local_common_clock` is on the shared host monotonic clock, so OGLO
aligns with Quest hand/head streams).

## Tests

`test_oglo_packet_parser` validates the wire decode against the firmware's own
12-bit packing reference (round-trip), plus the schema-4 fallback and
malformed-packet rejection:

```bash
ctest --test-dir build -R oglo_packet_parser --output-on-failure
# or standalone, no IsaacTeleop deps:
cd src/plugins/oglo_tactile/tests
g++ -std=c++20 -I.. test_oglo_packet_parser.cpp ../oglo_packet_parser.cpp -o t && ./t
```
