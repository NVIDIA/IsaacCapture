<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Vehicle Teleop Example

This example sends steering wheel and pedal input through Isaac Teleop, retargets it to a vehicle command, and publishes that command over ZMQ for a Kia Panda worker.

The remote side uses the native Linux steering wheel plugin and Isaac Teleop OpenXR session. The vehicle side subscribes to the command stream and writes to `PandaRunner`.

## Setup

Build and install Isaac Teleop with examples enabled. From the Isaac Teleop repository root:

```bash
cmake -B build -DBUILD_EXAMPLES=ON
cmake --build build --parallel 2
cmake --install build
```

Clone the vehicle-side dependencies into this example's `thirdparty` directory. These repositories are intentionally not added as git submodules.

```bash
cd examples/vehicle_teleop
mkdir -p thirdparty
git clone https://github.com/commaai/panda.git thirdparty/panda
git clone git@github.com:UCR-CISL/kia-opendbc.git thirdparty/kia-opendbc
```

You can use existing local checkouts instead, as long as the paths match `thirdparty/panda` and `thirdparty/kia-opendbc`.

## Remote Side

Start the CloudXR runtime:

```bash
python3 -m isaacteleop.cloudxr
```

In a separate terminal, activate the CloudXR environment printed by that command, then start the Isaac Teleop steering worker:

```bash
source ~/.cloudxr/run/cloudxr.env
./scripts/run_isaac_remote_steering_worker.sh --verbose
```

By default, the worker binds to `tcp://*:5555` on topic `kia_control`.

Useful options:

```bash
./scripts/run_isaac_remote_steering_worker.sh --device /dev/input/js0
./scripts/run_isaac_remote_steering_worker.sh --bind "tcp://*:5555"
./scripts/run_isaac_remote_steering_worker.sh --log-mcap logs/kia_control.mcap
./scripts/run_isaac_remote_steering_worker.sh --plugin-binary /path/to/steering_wheel_plugin
```

Steering wheel axis mapping is configured in `config/steering_wheel_config.yaml`. The default mapping is for a Logitech G920/G923-style setup where steering, throttle, brake, and clutch are raw Linux joystick axes.

## Vehicle Side

Run the Panda worker on the vehicle machine:

```bash
./scripts/run_kia_panda_worker.sh --connect "tcp://<remote-ip>:5555"
```

For local testing without opening PandaRunner, use dry-run mode:

```bash
./scripts/run_kia_panda_worker.sh --connect "tcp://<remote-ip>:5555" --dry-run
```

Do not run the live vehicle-side worker unless the vehicle-side hardware and safety process are ready.

## Replay Logs

If the remote worker was started with `--log-mcap`, replay the recorded commands with:

```bash
./scripts/replay_command_mcap.sh logs/kia_control.mcap
./scripts/replay_command_mcap.sh logs/kia_control.mcap --realtime
```

The replay command prints the retargeted vehicle command values. It does not write to PandaRunner.

## Troubleshooting

If the steering worker cannot find the native plugin, pass `--plugin-binary` explicitly. In a source build, the default location is usually `build/src/plugins/steering_wheel/steering_wheel_plugin`. In an install tree, it is usually `plugins/steering_wheel/steering_wheel_plugin`.

If the worker fails while creating the OpenXR session, confirm that the CloudXR runtime is running, the CloudXR environment has been sourced in the worker terminal, and the OpenXR client is connected.

If the Panda worker cannot import `opendbc` or `panda`, confirm the two dependency repositories were cloned under `examples/vehicle_teleop/thirdparty/`.
