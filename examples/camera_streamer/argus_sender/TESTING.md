# Testing argus_sender

## Host unit and loopback tests

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

The default suite does not open a camera. It covers YAML capture-mode parsing and validation,
output geometry, mixed-rate scheduling, capture-priority selection and release
timing, stereo alignment policy, AVS1 serialization/parsing, asynchronous bundle
assembly and ordering, plus TCP send/dump/disconnect and bounded stalled-receiver
drain behavior.

The matching receiver parser tests are:

```bash
cd ~/IsaacCapture/examples/camera_streamer
.venv/bin/python -m pytest tests/python -q
```

## 120-frame hardware smoke test

The opt-in test opens the devices in
`configs/hardware_stereo_smoke.yaml`, creates two HEVC encoders, and runs a local
AVS1 receiver:

```bash
cmake -S . -B build \
  -DARGUS_SENDER_HARDWARE_TESTS=ON \
  -DARGUS_SENDER_HARDWARE_CONFIG="$PWD/configs/hardware_stereo_smoke.yaml"
cmake --build build -j$(nproc)
ctest --test-dir build -R '^avs1_hardware_smoke$' --output-on-failure -V
```

The test receives exactly 120 bundles and verifies:

- contiguous AVS1 bundle sequence numbers;
- a present stereo HEVC stream with nonempty left/right payloads;
- monotonically increasing per-eye SOF TSC timestamps when available;
- VPS, SPS, PPS, and an IRAP NAL in each first access unit;
- clean sender exit with no reconnecting process left behind.

Override the loopback port with `-DARGUS_SENDER_TEST_PORT=PORT`. To use other
devices, modes, dimensions, or rates, copy the hardware YAML and pass its path
through `ARGUS_SENDER_HARDWARE_CONFIG`.
