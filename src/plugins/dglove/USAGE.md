<!--
SPDX-FileCopyrightText: Copyright (c) 2026 mwang259. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DGlove setup and usage

The Linux plugin receives DGlove v1 UDP/Protobuf frames and publishes
`DGloveFrame` data to Isaac Capture. A separate DGlove application connects to
the glove and computes pose before sending it. The plugin requires Python 3.11
or newer, Protobuf, and a configured CloudXR/OpenXR runtime with push-device and
tensor-data support. Obtain the DGlove application and any device-specific
resources separately from its provider.

## Build and install

From the Isaac Capture source root, with the project's build prerequisites installed:

```bash
cmake -S . -B build/dglove -DBUILD_PLUGIN_DGLOVE=ON -DBUILD_TESTING=ON
cmake --build build/dglove --target dglove_plugin dglove_reader --parallel
cmake --install build/dglove --prefix "$PWD/install" --component dglove
uv venv --python 3.11 build/dglove/venv
uv pip install --python build/dglove/venv/bin/python -r install/plugins/dglove/requirements.txt
```

## Run

Start CloudXR and load its exported environment in both terminals. Start the
receiver and publisher in the first terminal:

```bash
build/dglove/venv/bin/python install/plugins/dglove/source.py
```

Enable DGlove UDP output to `127.0.0.1:50051`. In the second terminal, read data
from the runtime with the independent reader:

```bash
install/plugins/dglove/dglove_reader --seconds=15
```

The default collections are `dglove/left` and `dglove/right`. To change the
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
`ctest --test-dir build/dglove --output-on-failure -R '^dglove_'` after building
the plugin. These tests need no glove, inference model or CloudXR session.
Live hardware validation is separate.

The bundled `protocol/glove_frame.proto` and `protocol/glove_frame_pb2.py`
are licensed under Apache-2.0 with attribution to the DGlove contributors. See
`LICENSES/Apache-2.0.txt` in the source tree or
`protocol/LICENSE-Apache-2.0.txt` in the installed plugin.
