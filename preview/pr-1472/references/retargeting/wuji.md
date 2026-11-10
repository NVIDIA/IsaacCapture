<!-- SPDX-FileCopyrightText: Copyright (c) 2026 Wuji Technology. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Retargeter: Wuji Glove to Wuji Hand

`WujiHandRetargeter` maps hand tracking from any OpenXR
`XR_EXT_hand_tracking` source to the 20 joint commands of a Wuji Hand or
Wuji Hand 2. It wraps the retargeting support in `wuji_sdk` and does not
depend on the Wuji glove plugin.

## At a glance

| Stage         | What happens                                                                                                                               |
|---------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| Input         | A 26-joint OpenXR `HandInput` for the configured side.                                                                                     |
| Mapping       | The OpenXR joints are reduced to the 21-joint MediaPipe hand layout<br/>(positions only; joint orientations are not used).                 |
| Retargeting   | `wuji_sdk.retargeting.RetargetSession` maps the hand pose to the<br/>selected Wuji hand model.                                             |
| Output        | `hand_joints` contains 20 joint angles in firmware order;<br/>`hand_valid` indicates whether they came from a valid input frame.           |
| Tracking loss | Invalid input resets the retargeting session. The hardware demo holds<br/>the last valid command instead of forwarding the invalid output. |

## Installation

Install the Wuji retargeting extra:

```console
$ pip install 'isaaccapture[wuji]'
```

## Use it from Python

```python
from isaaccapture.retargeters import (
    WujiHandRetargeter,
    WujiHandRetargeterConfig,
)

config = WujiHandRetargeterConfig(
    model="wuji_hand_2",
    hand_side="right",
)
retargeter = WujiHandRetargeter(config, name="wuji_right")
```

`model` accepts `"wuji_hand"` or `"wuji_hand_2"`. `hand_side` accepts
`"left"` or `"right"`.

## Run the example

The example supports a device-free synthetic pose, recorded keypoints, and a
real Wuji hand. Install it from the repository root, in a venv set up as in
[step 4 of Build from source](../../development/build_from_source/index.md#build-from-source-install-package), with
`wuji` added to the extras that step installs:

```console
$ uv pip install -e ./examples/retargeting

# Device-free smoke test
$ python -m isaaccapture_examples.retargeting.wuji_hand_retargeter_demo \
    --mode synthetic --model wuji_hand_2 --hand right

# Replay MediaPipe-format keypoints
$ python -m isaaccapture_examples.retargeting.wuji_hand_retargeter_demo \
    --mode replay --replay my_take.pkl --model wuji_hand_2 --hand right

# Drive real hardware and let TeleopSession launch the Wuji glove plugin
$ python -m isaaccapture_examples.retargeting.wuji_hand_retargeter_demo \
    --mode drive --hand right --plugin-path build/src/plugins
```

Drive mode needs a running CloudXR runtime, an OpenXR hand-tracking source, and
a reachable Wuji hand.

`--model` can be omitted: the model is detected from the connected hand, and
passing one that disagrees with it is an error. A hand whose reported side
differs from `--hand` produces a mirrored-motion warning. Without
`--plugin-path`, the OpenXR hand source must already be running.

## Validate

The tests import the Python package staged by a full build, so build the project
first — the glove plugin’s `install.sh` only builds the plugin target:

```console
$ cmake --build build
$ ctest --test-dir build \
    -R 'retargeting_test_wuji_(hand_retargeter|mapping_tables)' \
    --output-on-failure
```

Both tests run by default — CTest installs the `wuji` extra for the
end-to-end retargeter test. Configure with `-DTEST_WUJI_RETARGETER=OFF` to
keep the Wuji SDK out of the test environment; the retargeter test then skips,
while the mapping-table cross-check still runs because it needs no SDK.

#### SEE ALSO
[Wuji Glove](../../device/hand_tracking/wuji_glove.md) – install and run the Wuji glove input plugin.

[Build a Retargeting Pipeline](index.md#isaac-teleop-pipeline-builder) – connect the hand source and retargeter
into a complete pipeline.
