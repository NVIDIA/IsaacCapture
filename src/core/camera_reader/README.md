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
  buffers, so the pixels stay as they are, across a restart of the camera and after the
  reader object is gone. It gives them back when it is destroyed, or earlier with
  `release()`; give it back only when nothing reads its views any more.
- **Window and order** are chosen in `open` ([terms](../camera_core/README.md)). The window is
  how many frame sets the reader may hold at once. The default, a window of one in newest
  order, gives the newest frame set and says in `skipped()` how many were passed over. A
  recorder asks for a larger window and gets every frame set in order.
- **Newest means newest when you ask.** In newest order `next` asks the plugin for a frame
  set, and the plugin holds the newest one it has at that moment, or the next one published.
  Nothing is held for a reader between calls, so a reader that was busy for a second gets
  what the camera saw last, not what it saw a second ago. See "A frame set that waited".
- **`next` returns nothing on timeout** and throws `CameraDisconnected` when the plugin is
  gone. Called by a reader that already holds its window's worth, it throws
  `CameraWindowExhausted` at once, because no frame set could arrive.
- **Opening checks the plugin's answer.** It fails with a `CameraReaderError` that says why:
  nothing listens on the endpoint, the group does not exist, the subscription does not fit
  the pool, the plugin speaks a protocol version this reader does not, the answer is not the
  group that was asked for, the group uses a kind of memory or a format this build does not
  know, or its description would let a view run off the end of a pool.
- **`list` reports every group**, also one this build cannot read, with the reason in
  `unsupported`.
- **Which plugin.** By default the reader looks for this user's plugin on its default
  endpoint. A `PluginLocation` names another instance of the plugin, or a socket path.
- **The reader does not start the plugin.** Like every Isaac Capture plugin, the camera plugin
  is started by the application through the plugin manager, or by hand or a service. A
  reader that finds none says where it looked and how to start one.
- **One reader reads one group.** For two groups, open two readers; `descriptor()` lets an
  application wait on several at once.
- A `FrameSet` may be given back from another thread than the one that calls `next`, and
  after its reader is gone. The connection stays open until the last frame set of a reader
  that is gone has been given back, because the hold is the connection's.

## A frame set that waited

A newest-order reader that asked and stopped waiting (its `next` timed out) is still owed a
frame set, and is sent the next one published, whether or not it is in `next` at that
moment. By the time it comes back, that frame set may have been overtaken. So a frame set
that is already waiting when `next` is called is not handed out: `next` gives it back and
asks again. The plugin answers at once with the newest one, which is the same frame set
when nothing newer exists, so this costs a round trip and never a frame. A reader that
waits in `next` until its frame set comes never pays it.

Not covered: a process that is stopped while it waits inside `next` is handed, when it
continues, the frame set that arrived meanwhile. A reader that needs to know how old its
frame is has the times of every step in `timeline()`.

## Python

The same reader as `isaaccapture.camera`:

```python
from isaaccapture.camera import CameraReader

with CameraReader.open("head") as reader:
    for frames in reader:
        left = frames.image(0).array()   # (height, width, 4) uint8, read-only
```

- An array is a view of the plugin's buffer, not a copy, and cannot be written to. **It
  keeps its frame set held**: it shows that frame for as long as the array exists, after
  `release()`, after the loop has moved on and after the reader is closed. The price is the
  window: an array that is kept counts as a frame set the reader holds.
- `open` takes the window, the order (`"newest"`, `"in_order"`) and where the plugin is as
  keyword arguments. The window is two by default, not one as in C++: in the loop above
  `left` still names the array of the frame before while the next frame set is asked for.
- A reader that holds its window's worth raises `CameraWindowExhausted` from `next()`, with
  what to do about it: drop an array, or copy what is needed from it.
- `for frames in reader` lets go of the frame set before at each step. With `reader.next()`
  the caller does: `frames.release()`, or a `with` block.
- `with frames.borrow() as frames:` is for code that must not keep anything: the frame set
  goes back to the plugin when the block ends, and an array of it that still exists then
  raises `BufferError` instead of holding the frame set without a word.
- Arrays are NumPy arrays over host memory. A DLPack export is added with the first pool
  kind that is not host memory, where it is the way to hand a frame to a tensor library.
- `bgra8` and `rgba8` images are `(height, width, 4)`, `gray8` is `(height, width)`; `nv12`
  has two planes, `plane(0)` and `plane(1)`. `data` is the flat bytes, also for encoded video.
- The plugin ships in the package under `plugins/camera`, where the plugin manager finds it.

## Tests

`tests/cpp/core/camera_reader`, run with `ctest -L camera_reader`.

- **Unit** (`ctest -L unit`): the reader against a real endpoint in the same process, and
  against a stand-in plugin for what a real one does not do on demand.
- **Integration** (`ctest -L integration`): the reader in the test process and
  `camera_test_peer` as the other process, over the real socket. A reader gets whole frame
  sets from a plugin in another process; a reader that was away for sixty frame periods is
  handed a frame set published within the last few; a reader process that is killed while it
  holds frame sets gives them back; a reader is told when the plugin's process is killed and
  can still read what it holds.

`tests/python/core/camera` (`ctest -R camera_python`) reads from the plugin that is bundled
into the package, started as a process and, in one case, through the plugin manager.
