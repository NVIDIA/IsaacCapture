<!--
SPDX-FileCopyrightText: Copyright (c) 2026 mwang259. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DAYANGlove setup and usage

The DAYANGlove plugin receives v1 Protobuf frames from the external DGlove
application over UDP and publishes `DAYANGloveFrame` data to Isaac Capture. The
DGlove application connects to the glove and estimates hand pose before sending
the data. The plugin requires Python 3.11 or newer, Protobuf, and a configured
CloudXR/OpenXR runtime with push-device and tensor-data support. Obtain the
DGlove application and any device-specific resources separately from its
provider.

## Build and install

From the Isaac Capture source root, with the project's build prerequisites installed:

```bash
cmake -S . -B build/dayanglove -DBUILD_PLUGIN_DAYANGLOVE=ON -DBUILD_TESTING=ON
cmake --build build/dayanglove --target dayanglove_plugin dayanglove_reader --parallel
cmake --install build/dayanglove --prefix "$PWD/install" --component dayanglove
uv venv --python 3.11 build/dayanglove/venv
uv pip install --python build/dayanglove/venv/bin/python -r install/plugins/dayanglove/requirements.txt
```

## Run

Start CloudXR and load its exported environment in both terminals. Start the
receiver and publisher in the first terminal:

```bash
build/dayanglove/venv/bin/python install/plugins/dayanglove/source.py
```

Enable UDP output in the DGlove application. If it runs on the same machine as
the plugin, use `127.0.0.1:50051` as the destination. In the second terminal, run
the standalone reader to inspect the published data.

```bash
install/plugins/dayanglove/dayanglove_reader --seconds=15
```

The default collections are `dayanglove/left` and `dayanglove/right`. To change the
prefix, pass the same `--collection-prefix` to both processes. For UDP diagnostics
without CloudXR, stop the source and run `receiver.py --duration 15` with the
same Python interpreter.

## Frame contract and tests

A valid pose contains 45 local MANO axis-angle components in radians and 21
wrist-relative joint positions in metres. The pose uses a canonical right-hand
representation. Consumers use `mirror_x` when displaying the left hand.
Each frame also carries all 210 original palm ADC values. Frames with
`pose_valid=false` can still carry palm data. A hand expires independently when
its frames stop arriving. Version 1 carries no raw back-of-hand channels or
world-space wrist pose.

The test suite uses generated inputs and exercises UDP reception, ordering,
expiry, exact bridge conversion and native FlatBuffers round trips. Run it with
`ctest --test-dir build/dayanglove --output-on-failure -R '^dayanglove_'` after building
the plugin. These tests need no glove, inference model or CloudXR session.
Live hardware validation is separate.

The bundled `protocol/glove_frame.proto` and `protocol/glove_frame_pb2.py`
are licensed under Apache-2.0 with attribution to the DGlove contributors. See
`LICENSES/Apache-2.0.txt` in the source tree or
`protocol/LICENSE-Apache-2.0.txt` in the installed plugin.
