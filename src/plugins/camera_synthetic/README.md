<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Synthetic camera plugin

A camera plugin that needs no hardware. Its cameras write frames whose every byte is known
from the frame number, so a reader can check that it received the whole of the frame it was
told about. It is what the tests and the latency bench read, a way to try the camera
interface on any Linux machine, and the plugin a vendor's camera plugin is copied from
([Writing a camera plugin](../../core/camera/README.md#writing-a-camera-plugin)).

Like every camera plugin it is one process that owns its cameras and serves their frames to
any number of readers in other processes, which read its buffers in place through a
[`CameraReader`](../../core/camera/README.md#reading-cameras).

```yaml
# ~/.config/isaaccapture/cameras.yaml
cameras:
  head:
    backend: synthetic
```

```bash
camera_synthetic_plugin
```

```cpp
CameraReader reader = CameraReader::open("head");
while (auto frames = reader.next(std::chrono::seconds(1)))
{
    const ImageView left = frames->image(0);
}
```

Neither side is told where the other is: the plugin listens on `camera-synthetic.sock` in
this user's endpoint directory, and the reader looks through that directory for the plugin
that publishes `head`.

## Running it

**By hand**, from the build tree or from `plugins/camera_synthetic` of an install:

```bash
./build/src/plugins/camera_synthetic/app/camera_synthetic_plugin --camera=head:width=1920,height=1080
```

**Through the plugin manager**, as an application starts any Isaac Capture plugin. The
plugin's name is `camera_synthetic`; `plugin.yaml` sits next to the executable, in the build
tree and in the install. The handle stops the plugin when it goes away.

```cpp
core::PluginManager manager({ "<install>/plugins" });
std::unique_ptr<core::Plugin> plugin = manager.start("camera_synthetic", "camera", { "--camera=head" });
```

The `isaaccapture` Python package bundles the executable and its `plugin.yaml` under
`plugins/camera_synthetic`, so an application that installed the wheel starts the plugin the
same way, with `BUNDLED_PLUGINS_DIR` (from `isaaccapture.teleop_session_manager`) as the
search path.

The arguments are those of every camera plugin;
[Running a plugin](../../core/camera/README.md#running-a-plugin) has the whole of it.

| Argument | |
|---|---|
| `--camera=ID[:KEY=VALUE,...]` | One camera: the id readers ask for and its options. Give it once per camera. |
| `--config=FILE` | The cameras to serve, instead of `~/.config/isaaccapture/cameras.yaml`. The plugin serves the cameras with `backend: synthetic` and leaves the others to their plugins. |
| `--auto` | Serve what is found plugged in, whatever a file says. A synthetic camera is never found, so this plugin then has nothing to serve and stops. |
| `--instance=NAME` | Listen on `camera-NAME.sock` instead of `camera-synthetic.sock`, for a second synthetic plugin beside the first. Give the second its own cameras with `--camera` or `--config`; from the user's file it would serve the same cameras as the first, and a reader that looks for one of them is refused because two plugins publish it. |
| `--endpoint=PATH` | Listen on this socket file. A reader is then told the path. |
| `--slots=N` | Frames kept per stream, 8 by default. |
| `--help` | Print the arguments. |

Without `--camera`, `--config` and a file at the default path, the plugin has no camera and
stops with an error that says so.

## Options of a synthetic camera

| Option | Default | Takes |
|---|---|---|
| `width` | 1280 | 16 to 8192 |
| `height` | 720 | 16 to 8192 |
| `fps` | 60 | 1 to 1000 |
| `streams` | 2 | 1 to 8, captured together as one frame set |
| `format` | `bgra8` | `bgra8`, `rgba8`, `gray8`, `nv12` |

The streams of a camera `head` are `head/left` and `head/right` when there are two, and
`head/0`, `head/1`, ... otherwise. Frame n of stream s starts with n as eight little-endian
bytes; every other byte of the image is `(n + 17 * s) & 0xff`
(`core/inc/camera_synthetic/synthetic_backend.hpp`). An option the plugin does not know is an
error.

## Measuring

Every frame set carries the times it passed each step, on the host's monotonic clock.
`camera_bench` reads a group like any other reader, from whichever camera plugin publishes
it, and prints them:

```text
$ camera_bench --group=head --frames=600
head: 2 stream(s)
  head/left                1280x720 bgra8, 60.00 fps, 8 slots of 3686400 bytes
  head/right               1280x720 bgra8, 60.00 fps, 8 slots of 3686400 bytes
read 600 frame sets in 10.00 s (60.0 per second), skipped 0
                                p50        p90        p99        max   (microseconds)
  capture -> sdk ready        ...
  sdk ready -> published      ...
  published -> sent           ...
  sent -> reader              ...
  capture -> reader           ...
```

`published -> sent` and `sent -> reader` are what the plugin and the process boundary add; the
steps before them belong to the camera and its SDK.

Without `--group` the bench lists what this user's camera plugins publish, and the endpoint
of the plugin that publishes each group:

```text
$ camera_bench
head: 2 stream(s) on /run/user/1000/isaaccapture/camera-synthetic.sock
  head/left                1280x720 bgra8, 60.00 fps, 8 slots of 3686400 bytes
  head/right               1280x720 bgra8, 60.00 fps, 8 slots of 3686400 bytes
```

The bench finds the plugin as a reader does. `--instance=NAME` or `--endpoint=PATH` reads
from that one plugin instead. `--window` and `--order` read as a recorder would, and
`--check-synthetic` checks every byte of every image against the synthetic pattern.

## Layout

- `core/`: the synthetic capture code, the plugin's backend. A static library, so that the
  bench and the tests check frames against the same pattern.
- `app/`: the `camera_synthetic_plugin` executable: a `main` that hands the backend to
  `run_camera_plugin`.
- `tools/`: the `camera_bench` executable.
- `plugin.yaml`: the manifest the plugin manager reads.

Linux only, and built whenever plugins are (`BUILD_PLUGINS`, on by default): it needs no
SDK, so it has no build option of its own. It is installed to `plugins/camera_synthetic`
with `camera_bench`, `plugin.yaml` and this file.

## Tests

`tests/cpp/plugins/camera_synthetic`, run with `ctest -L camera_synthetic`.

- **Unit** (`ctest -L unit`): the synthetic backend's options and layouts; two synthetic
  cameras read through one host, every byte checked.
- **Integration** (`ctest -L integration`): `camera_synthetic_plugin` and `camera_bench` as
  processes of their own. A reader in the test process reads from the plugin and checks every
  byte; a second plugin cannot take a running one's endpoint; `SIGTERM` is a clean stop; a
  plugin that was killed is replaced by the next; a plugin without arguments serves its
  cameras of the user's configuration file on `camera-synthetic.sock`, where a reader that
  was told nothing finds it; a second instance runs beside it and both are found by group; a
  file whose cameras are all another backend's stops the plugin; the plugin manager starts
  and stops the plugin; the bench finds a camera, reports each step and lists what is
  published and where.
