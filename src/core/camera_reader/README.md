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

CameraReader reader = CameraReader::open(endpoint, "head");
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
- **One reader reads one group.** For two groups, open two readers; `descriptor()` lets an
  application wait on several at once.
- A `FrameSet` may be given back from another thread than the one that calls `next`, and
  after its reader is gone.

## Tests

`tests/cpp/core/camera_reader`, run with `ctest -L camera_reader`. They are tagged `unit`
and read from a real endpoint in the same process.
