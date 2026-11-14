<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Camera plugin

One process that owns the cameras and serves their frames to any number of readers in other
processes. Viewers, encoders and recorders read the plugin's buffers in place through
[`camera_reader`](../../core/camera_reader); none of them links a camera SDK.

```bash
camera_plugin --endpoint=/tmp/cameras.sock --camera=head:synthetic
```

```cpp
CameraReader reader = CameraReader::open("/tmp/cameras.sock", "head");
while (auto frames = reader.next(std::chrono::seconds(1)))
{
    const ImageView left = frames->image(0);
}
```

## Running it

| Argument | Meaning |
|---|---|
| `--endpoint=PATH` | The socket file readers connect to. Only one plugin serves a path at a time. |
| `--camera=ID:BACKEND[:KEY=VALUE,...]` | One camera: the id readers ask for, the backend that opens it, and that backend's options. Repeat it for each camera. |
| `--slots=N` | Frames kept per stream, 8 by default. A pool holds the newest frame set, one slot for the camera to write ahead, and the windows of all readers; a reader that does not fit is refused with the numbers. |
| `--list-backends` | Print the backends this build has. |

A misspelt argument or option is an error, not something ignored. `SIGINT` and `SIGTERM` stop
the plugin cleanly; readers see their connection close.

Every ten seconds the plugin logs, per camera, how many frame sets it published. It warns
when a camera had to drop frame sets because its pool was full, or when a reader did not take
its events; both mean the pool or a reader's window needs another look.

## Measuring

Every frame set carries the times it passed each step, on the host's monotonic clock.
`camera_bench` reads a group like any other reader and prints them:

```text
$ camera_bench --endpoint=/tmp/cameras.sock --group=head --frames=600
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
steps before them belong to the camera and its SDK. Without `--group` the bench lists what the
plugin publishes. `--window` and `--order` read as a recorder would, and `--check-synthetic`
checks every byte of every image against the synthetic backend's pattern.

## Backends

A backend opens cameras through one SDK. **Which backends a plugin has is decided when it is
built**, so nobody builds or installs an SDK they do not use.

| Backend | Needs | Options |
|---|---|---|
| `synthetic` | nothing; always built | `width` (1280), `height` (720), `fps` (60), `streams` (2), `format` (`bgra8`, `rgba8`, `gray8`, `nv12`) |

The synthetic camera writes frames whose every byte is known from the frame number, so a
reader can check that it received the whole of the frame it was told about. It is what the
tests and the latency bench read.

### Adding a backend

1. Put it in `backends/<name>/` as a library of its own that implements `CameraSource`
   (`core/inc/camera_plugin/camera_source.hpp`) and returns a `Backend` from one function.
2. Add an option `BUILD_CAMERA_BACKEND_<NAME>`, off by default, next to the other build
   options in the top-level `CMakeLists.txt`.
3. In `backends/CMakeLists.txt`, add the directory and link the library only when the option
   is on, and pass the option to `backends.cpp` as a compile definition; there, add the
   backend to `built_backends()` under that definition.

A backend that is off is not configured at all, so its SDK is neither looked for nor built. A
camera that names a backend the build does not have is refused with the list of those it has.

## Layout

- `core/`: what a backend implements, and `CameraHost`, which gives each camera its pools and
  puts all of them behind one endpoint.
- `backends/`: the backends of this build.
- `app/`: the `camera_plugin` executable.
- `bench/`: the `camera_bench` executable.

## Tests

`tests/cpp/plugins/camera`, run with `ctest -L camera_plugin`. One case starts the plugin as a
process of its own and reads from it, kills it, and starts another on the same path; another
runs the bench against a camera in the test process.
