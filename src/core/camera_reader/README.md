<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Camera interface: the reader

What an application links to get camera frames. It talks to the camera plugin's
[endpoint](../camera_endpoint) and hands out frame sets whose pixels are the plugin's own
buffers, mapped read-only; nothing is copied.

```cpp
#include <camera_reader/camera_reader.hpp>

CameraReader reader = CameraReader::open("head");
while (auto frames = reader.next(std::chrono::seconds(1)))
{
    const ImageView left = frames->image(0);
    process(left.plane(0), left.stream->width, left.stream->height, left.stream->planes[0].stride_bytes);
}
```

- **A `FrameSet` is a hold on the frames.** While it exists the plugin does not reuse those
  buffers, so the pixels stay as they are. It gives them back when it is destroyed, or
  earlier with `release()`. Hold at most as many as the reader's window.
- **Window and order** are chosen in `open` ([terms](../camera_core/README.md)). The default,
  a window of one, always gives the newest frame set and says in `skipped()` how many were
  passed over. A recorder asks for a larger window and gets every frame set in order.
- **`next` returns nothing on timeout** and throws `CameraDisconnected` when the plugin is
  gone. Opening fails with a `CameraReaderError` that says why: nothing listens on the
  endpoint, the group does not exist, or the subscription does not fit the pool.
- **Which plugin.** By default the reader looks for this user's plugin on its default
  endpoint. A `PluginLocation` names another instance of the plugin, or a socket path.
- **The reader does not start the plugin.** Like every Isaac Capture plugin, the camera plugin
  is started by the application through the plugin manager, or by hand or a service. A
  reader that finds none says where it looked and how to start one.
- **One reader reads one group.** For two groups, open two readers; `descriptor()` lets an
  application wait on several at once.
- A `FrameSet` may be given back from another thread than the one that calls `next`, and
  after its reader is gone.

## Tests

`tests/cpp/core/camera_reader`, run with `ctest -L camera_reader`.

- **Unit** (`ctest -L unit`): the reader against a real endpoint in the same process, and
  against a stand-in plugin for what a real one does not do on demand.
- **Integration** (`ctest -L integration`): the reader in the test process and
  `camera_test_peer` as the other process, over the real socket. A reader gets whole frame
  sets from a plugin in another process; a reader process that is killed while it holds
  frame sets gives them back; a reader is told when the plugin's process is killed and can
  still read what it holds.
