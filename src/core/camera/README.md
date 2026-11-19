<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Camera interface

The camera interface is the one place Isaac Capture applications get camera frames from. Each
camera is owned by one process, a camera plugin, which keeps its frames in fixed pools of shared
buffers; viewers, encoders and recorders in other processes on the same machine read those
buffers in place.

The library has two sides: publishing, which a camera plugin is built on, and reading, which an
application uses. It is Linux only: the part that crosses between processes, the shared memory
and the socket, is what ties it to Linux.

What is in it, included as `<camera/NAME.hpp>` from the target `camera::camera`:

- `types.hpp`: the description of a group and its streams, the frame event a reader is sent,
  and `problem_with`, the check of a description.
- `publisher.hpp`: `Publisher`, which decides which pool slot holds which frame set and who may
  still read it.
- `host_pool.hpp`: `HostPool` and `HostPoolView`, one stream's slots in shared host memory.
- `message_socket.hpp`: `MessageListener` and `MessageSocket`, the connection between the
  processes.
- `endpoint_path.hpp`: where a camera plugin listens.
- `unique_fd.hpp`: `UniqueFd`, which owns one file descriptor and closes it.
- `messages.hpp`: the messages a camera plugin and a reader send each other, and the codec that
  turns them into bytes and back.
- `endpoint.hpp`: `Endpoint`, a camera plugin's end of the connection, which serves a
  `Publisher`'s groups to readers.
- `camera_reader.hpp`: `CameraReader`, the reader's end and what an application uses.
- `camera_source.hpp`: `Backend`, `CameraSource` and `FrameOutput`, what a vendor's capture code
  implements and what it publishes through.
- `camera_host.hpp`: `CameraHost`, which gives each camera its pools and puts all of them behind
  one endpoint.
- `options.hpp`: `OptionReader`, how a backend reads a camera's options, so that one it does
  not know is an error.
- `config.hpp`: `load_config_file`, the cameras a configuration file names for one backend.
- `camera_plugin.hpp`: `run_camera_plugin`, everything a camera plugin's `main` does.

## Terms

- **Stream**: one sensor's images. It has a pool of **slots**; a slot holds one image.
- **Group**: one or more streams captured together, such as a stereo pair or a triggered rig.
  A group publishes one **frame set** at a time: one image per stream, under one sequence number.
- **Window**: how many frame sets are kept for a reader at a time, unread and held together.
- **Order**: `NEWEST` hands the reader the newest frame set there is each time it asks, and skips
  the rest (a viewer, a live encoder). `IN_ORDER` sends every frame set in sequence (a recorder).
  The default is `NEWEST` for a window of one and `IN_ORDER` otherwise.
- **Lease**: a reader's hold on a frame set. Its slots are not reused until the reader releases
  it or goes away.
- **Session**: one run of a group, from the plugin starting to publish it until that plugin
  ends, named by a random `session_id` in the group's description.
- **Sequence**: a frame set's place among those the group published in the session. It is the
  same for every reader, so a reader that skips sees a gap, and with the session id it names
  a capture for good. A capture that was dropped before publication takes no number; it
  shows as a gap in the images' `device_frame_id`.
- **Generation**: counts restarts of the camera within the session, so a reader can tell a
  restart from a skip. A restart does not reset the sequence.

## Reading cameras

`CameraReader` is what an application uses to get camera frames. It finds the camera plugin
that publishes a camera group, talks to that plugin's [endpoint](#the-endpoint) and hands out
frame sets whose pixels are the plugin's own buffers, mapped read-only; nothing is copied.

```cpp
#include <camera/camera_reader.hpp>

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
- **Window and order** are chosen in `open` ([terms](#terms)). The window is how many frame
  sets the reader may hold at once. The default, a window of one in newest order, gives the
  newest frame set and says in `skipped()` how many were passed over. A recorder asks for a
  larger window and gets every frame set in order.
- **Newest means newest when you ask.** In newest order `next` asks the plugin for a frame
  set, and the plugin holds the newest one it has at that moment, or the next one published.
  Nothing is held for a reader between calls, so a reader that was busy for a second gets
  what the camera saw last, not what it saw a second ago. See "A frame set that waited".
- **`next` returns nothing on timeout** and throws `CameraDisconnected` when the plugin is
  gone. A camera that stops delivering while its plugin runs is a timeout as well: the
  reader decides how long without a frame set is too long, and the plugin's log says why.
  Called by a reader that already holds its window's worth, it throws
  `CameraWindowExhausted` at once, because no frame set could arrive.
- **Opening checks the plugin's answer.** It fails with a `CameraReaderError` that says why:
  no plugin publishes the group or two do, nothing answers on the endpoint that was named,
  the group does not exist, the subscription does not fit the pool, the plugin speaks a
  protocol version this reader does not, the answer is not the group that was asked for, the
  group uses a kind of memory or a format this build does not know, or its description would
  let a view run off the end of a pool.
- **`list` reports every group**, also one this build cannot read, with the reason in
  `unsupported`, and with the `endpoint` of the plugin that publishes it.
- **Which plugin.** The reader finds it; see "How a reader finds its camera". A
  `PluginLocation` names one plugin instead, by its instance or by its socket path.
- **The reader does not start a plugin.** Like every Isaac Capture plugin, a camera plugin is
  started by the application through the plugin manager, or by hand or a service. A reader
  that finds none says where it looked and how to start one.
- **One reader reads one group.** For two groups, open two readers; `descriptor()` lets an
  application wait on several at once.
- A `FrameSet` may be given back from another thread than the one that calls `next`, and
  after its reader is gone. The connection stays open until the last frame set of a reader
  that is gone has been given back, because the hold is the connection's.

### How a reader finds its camera

There is one camera plugin per vendor or SDK. Each is a process of its own with an endpoint
of its own, a socket file in the user's endpoint directory
([Where a plugin listens](#where-a-plugin-listens)). An application knows the name of a
camera group, not which plugin publishes it.

- **An empty `PluginLocation` means: look.** `open` asks each endpoint in that directory what
  it publishes, one after another, and opens the group on the one that publishes it. `list`
  returns the groups of every one that says what it publishes, in the order of their socket
  paths. No plugin running is an empty list for `list`, and for `open` an error.
- **A plugin that does not serve yet is passed over at once.** An endpoint accepts readers
  only from its `start()`, so a plugin that is still opening its cameras refuses the
  connection, and so does a socket file left by a plugin that was killed. Neither is an
  error and neither costs a wait.
- **A plugin that takes the connection and does not say what it publishes is passed over.**
  One that stays silent is given half a second; one that refuses the reader, closes the
  connection or answers something the reader cannot use costs no wait. An endpoint that
  cannot be connected to for another reason than a refusal is passed over the same way. That
  includes a plugin that stopped accepting long ago and whose queue of connections is full:
  the connect is refused at once, so it costs no wait either.
  `list` leaves such a plugin out. If `open` does not find the group, the error names the
  endpoint and the reason.
- **Nobody publishes the group:** the error names the group and the directory that was
  searched, lists what is published there and by which endpoint, and says how to start a
  plugin.
- **Two plugins publish the group:** the error names both endpoints. The reader does not
  pick one; a `PluginLocation` says which.
- **A `PluginLocation` with an instance or an endpoint is that one plugin.** Nothing else is
  asked. The plugin gets the full two seconds to answer, and one that is not there is an
  error, for `list` too.
- **Nothing is remembered.** Every `open` and `list` looks again, so a plugin that was
  started or stopped a moment ago is seen as it is now.

### A frame set that waited

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

### Python

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
- `open` takes the window and the order (`"newest"`, `"in_order"`) as keyword arguments. The
  window is two by default, not one as in C++: in the loop above `left` still names the array
  of the frame before while the next frame set is asked for. The order is `"newest"` unless
  `"in_order"` is given; a larger window alone does not change it, as it does in C++.
- **Which plugin** is as in C++ ("How a reader finds its camera"). `open` and `list` look
  through this user's camera plugins; `instance=` or `endpoint=` names one plugin instead.
  Each `GroupListing` of `list` has the `endpoint` of the plugin that publishes the group.
- A reader that holds its window's worth raises `CameraWindowExhausted` from `next()`, with
  what to do about it: drop an array, or copy what is needed from it.
- `for frames in reader` lets go of the frame set before at each step. With `reader.next()`
  the caller does: `frames.release()`, or a `with` block.
- **From an event loop**, wait on `reader.fileno()` and read with `next(0)` until it returns
  `None`. In newest order the plugin sends a frame set only when it is asked, and `next(0)`
  is what asks, so call it once before waiting:

  ```python
  def on_readable():
      while (frames := reader.next(0)) is not None:   # takes what came, asks for the next
          use(frames)
          frames.release()

  loop.add_reader(reader.fileno(), on_readable)
  on_readable()                                       # the first request
  ```

  `next(0)` does not wait for a frame set, but it may take up to 20 ms when it exchanges one
  that waited for the newest (see "A frame set that waited"). A thread that blocks in
  `next()` is the other way to keep a loop free.
- `with frames.borrow() as frames:` is for code that must not keep anything: the frame set
  goes back to the plugin when the block ends, and an array of it that still exists then
  raises `BufferError` instead of holding the frame set without a word.
- Arrays are NumPy arrays over host memory. A DLPack export is added with the first pool
  kind that is not host memory, where it is the way to hand a frame to a tensor library.
- `bgra8` and `rgba8` images are `(height, width, 4)`, `gray8` is `(height, width)`; `nv12`
  has two planes, `plane(0)` and `plane(1)`. `data` is the flat bytes of the whole slot.
- The synthetic camera plugin ships in the package under `plugins/camera_synthetic`. The
  plugin manager, given `BUNDLED_PLUGINS_DIR` (the package's `plugins` directory, from
  `isaaccapture.teleop_session_manager`) as its search path, starts it by its name,
  `camera_synthetic`.

## Publishing cameras

A camera plugin is one vendor's or one SDK's capture code and this library. The capture code
opens the cameras and writes the images. The library gives each camera its pools, publishes the
frame sets, serves them on the plugin's [endpoint](#the-endpoint), and is the plugin's command
line and configuration.

```cpp
#include <camera/camera_plugin.hpp>

int main(int argc, char** argv)
{
    return core::camera::run_camera_plugin(argc, argv, my_backend());
}
```

There is one camera plugin per vendor or SDK. Each is a small executable with its own
`plugin.yaml` and install folder, and its own build option when it needs an SDK. Readers find
the plugin that publishes a camera
([How a reader finds its camera](#how-a-reader-finds-its-camera)); none of them links a camera
SDK. [`camera_synthetic`](../../plugins/camera_synthetic) is the plugin that needs no hardware
and the one to copy; see [Writing a camera plugin](#writing-a-camera-plugin).

`run_camera_plugin` is everything a plugin's `main` does. It reads the arguments, chooses the
cameras, takes the endpoint, opens the cameras through the plugin's backend, serves until
`SIGINT` or `SIGTERM`, and returns the exit code: 0 after a clean stop or `--help`, 2 for a
command line it cannot parse, 1 when the plugin cannot run. An argument that parses and cannot
be honoured, such as an option its backend refuses, returns 1.

`CameraHost` gives each camera a pool of shared memory per stream ([Host pool](#host-pool)) and
puts all of them behind one endpoint. `run_camera_plugin` uses it; a test can use it without a
process.

### Running a plugin

What follows is true of every camera plugin, because all of it is this library.

**Which cameras.** A camera has an id that readers ask for and the options of the plugin's
backend. The plugin takes its cameras from, in this order:

1. `--camera=ID[:KEY=VALUE,...]` on the command line, once per camera;
2. `--config=FILE`, or without it `$XDG_CONFIG_HOME/isaaccapture/cameras.yaml`
   (`~/.config/isaaccapture/cameras.yaml`) when that file exists;
3. otherwise, or with `--auto`, every camera its backend finds plugged in.

`--camera`, `--config` and `--auto` are alternatives: a command line that mixes them is
refused.

**The configuration file** is one for all the camera plugins of a user. A camera takes two
lines, its id and `backend:`, and one more for each option of that backend as `name: value`.
`slots:` at the top level sets the pool size.

```yaml
# ~/.config/isaaccapture/cameras.yaml
cameras:
  head:
    backend: synthetic
    width: 1920
slots: 8
```

A plugin serves the cameras whose `backend:` is its own and leaves the others to their
plugins. A file that names cameras, but none for this plugin's backend, stops the plugin with
an error that names the file and lists its cameras with their backends. That is how a
misspelt backend shows.

**Where it listens.** On `camera-<instance>.sock` in the user's endpoint directory
([Where a plugin listens](#where-a-plugin-listens)). A plugin's instance name is the name of
its backend, so the synthetic plugin listens on `camera-synthetic.sock`. A second plugin of
the same backend takes another name with `--instance=NAME`. Give it its own cameras with
`--camera` or `--config` as well: the instance name does not change which cameras a plugin
takes, so without them it serves the same cameras as the first, and a reader that looks for
one of them finds it on two plugins and is refused. `--endpoint=PATH` sets the socket file
outright; a reader is then told the path. Only one plugin serves an endpoint at a time: a
plugin takes its endpoint before it opens its cameras, and accepts readers once they are
open ([When readers can reach it](#when-readers-can-reach-it)).

**Pool size.** `--slots=N`, 8 by default, is the number of frames kept per stream. A pool
holds the newest frame set, one slot for the camera to write ahead, and the windows of all
readers; a reader that does not fit is refused with the numbers. When readers want every
frame in order, the frame sets waiting for them are kept in place of the newest one: as many
as the largest of their windows ([Pool budget](#pool-budget)).

**Who starts it.** The application, through the plugin manager, like every other Isaac
Capture plugin: `plugin.yaml` sits next to the executable, and the handle the plugin manager
returns stops the plugin when it goes away. The `--plugin-root-id` the plugin manager adds is
accepted and not used. A plugin can also be run by hand or from a service. A reader never
starts it ([Reading cameras](#reading-cameras)).

A misspelt argument, option or setting is an error, not something ignored. `SIGINT` and
`SIGTERM` stop the plugin cleanly; readers see their connection close. `--help` prints the
arguments under the name of the executable.

Every ten seconds the plugin logs, per camera, how many frame sets it published. It warns
when a camera had to drop frame sets because its pool was full, or when a reader did not take
its events; both mean the pool or a reader's window needs another look. The loggers are
`isaaccapture.camera.run_camera_plugin` and `isaaccapture.camera.CameraHost`. The usage, and
the refusal of an argument that comes with it, are for the terminal and are not logged.

### What a backend implements

A backend is one camera SDK as its plugin sees it, and a plugin has exactly one. It is one
function that returns a `Backend` (`camera_source.hpp`):

- a `name`: what a camera's configuration says in `backend:`, and the plugin's instance name;
- a `description`: a short phrase without a full stop, which the plugin's `--help` shows
  after the backend's name;
- `open`, which is called once with every camera the plugin is to serve and returns one
  `CameraSource` per camera. What the SDK has open is the backend's, not a camera's: an SDK
  that serves a whole rig from one object, or keeps state for the process, opens it once in
  that call and lets the sources share it;
- optionally `discover`, which returns the cameras that are plugged in, for a plugin that is
  not told which to serve.

Each source publishes one group. It reads its options through `OptionReader` and calls
`finish()`, so a misspelt option is refused.

A source gets the slot for an image through its `FrameOutput` in one of two ways:

- **The library picks the slot.** `acquire()` hands out a free slot, the source writes the
  image into it and calls `publish()`. When nothing is free the frame is dropped. This is the
  way to start with.
- **The SDK picks the buffer.** For an SDK that is given its buffers up front and says which
  one it filled: the source hands the SDK the memory of every slot (`slot_count()`,
  `slot_memory()`), asks to be told when a slot is free again (`on_slot_free()`), calls
  `claim()` for the buffer the SDK filled, then `publish()`. `backend_reserve()` says how
  many buffers the SDK needs in hand to keep capturing; readers are never given so many that
  it is left with fewer.

In both, `publish()` says the images are complete. A slot is not handed out or given back to
the SDK while a reader holds it, also across `restart()`, which is how a source says that
its camera restarted.

**A camera plugin publishes images as pixels and does not encode them.** The backend
interface and the protocol carry no encoded video and no encoder control. A consumer that
streams or records is a reader: it reads the frames and encodes them itself, with the
settings its own purpose needs, and what it does about a lost or late frame is its own.

### Writing a camera plugin

[`src/plugins/camera_synthetic`](../../plugins/camera_synthetic) is the template. For a
vendor or SDK called `<name>`:

1. **Directory.** Copy `src/plugins/camera_synthetic` to `src/plugins/camera_<name>`, without
   `tools/`: `camera_bench` reads any camera plugin and stays where it is. Leave out the block
   in `app/CMakeLists.txt` that bundles the plugin into the Python package as well: the wheel
   does not ship a vendor's SDK.
2. **`core/`: the backend.** A static library `camera_<name>_core` that links
   `camera::camera` and the SDK. It implements `CameraSource` for one camera and has one
   function that returns the `Backend`, named `<name>`. No type of the SDK leaves this
   directory.
3. **`app/main.cpp`.** A few lines: `return run_camera_plugin(argc, argv, <name>_backend());`.
   The executable is `camera_<name>_plugin`. There is nothing else to write for the command line,
   the configuration file or the endpoint.
4. **`plugin.yaml`.** `name: camera_<name>`, `command: "./camera_<name>_plugin"`, a version, a
   description and the `devices:` entry of type `camera`. `app/CMakeLists.txt` copies it next
   to the built executable, so the plugin manager can start the plugin from the build tree
   when it is given the directory above the executable's (`build/src/plugins/camera_<name>`),
   as it is given `<install>/plugins` for an install. The plugin is installed to
   `plugins/camera_<name>`.
5. **Build option.** A plugin that needs an SDK gets its own `BUILD_PLUGIN_<NAME>` option,
   off by default, next to the other plugin options in the top-level `CMakeLists.txt`, and
   `src/plugins/CMakeLists.txt` adds its directory only when the option is on. So nobody
   builds or installs an SDK they do not use. `camera_synthetic` needs none and is built
   whenever plugins are.
6. **Tests.** In `tests/cpp/plugins/camera_<name>`, wired under the same option, with a
   CTest label that starts with `camera_`. Tag what runs without the camera `unit` (the
   backend's options and what it describes) and what starts the plugin as a process
   `integration`; the cases of `tests/cpp/plugins/camera_synthetic` show both.
7. **Documents.** A `README.md` with the backend's options and what the SDK needs, and an
   `AGENTS.md` that points to "Publishing cameras" in [`AGENTS.md`](AGENTS.md) here for the
   rules every camera plugin follows.

## How it works

### Slots, windows and leases

`Publisher` decides which pool slot holds which frame set and who may still read it. It has no
slot memory and no sockets in it, and uses the standard library only.

A backend gets a slot in one of two ways, then publishes:

```cpp
// The backend writes the image itself.
auto slot = publisher.acquire_slot(group, stream);  // nothing: every slot is in use, drop the frame
// ... write the pixels into that slot ...
publisher.publish(group, images, timeline);

// The camera SDK owns the buffers and picked one.
publisher.claim_slot(group, stream, slot_the_sdk_filled);
publisher.publish(group, images, timeline);
```

For the second way, `set_slot_free_callback` tells the backend the moment a slot is held by
nobody, so the buffer can go back to the SDK. A group uses one way or the other: once it has
that callback, its slots are handed out with `claim_slot` only.

A reader subscribes with a window and an order, receives `FrameEvent`s through an `EventSink`,
and returns each one with `release`. Events carry slot indices, the frame set's timeline and the
number of frame sets the reader skipped. A slow reader skips; it never slows capture or another
reader, and no call blocks on a reader. Each group has its own lock, so one camera never waits
while another camera's events are being sent.

A subscription starts in two steps: `subscribe` reserves it, which counts it in the pool's
budget, and `activate` starts delivery. In between, whoever serves the reader tells it what it
subscribed to, so that answer always comes before the first frame set.

#### When a frame set is held for a reader

An in-order reader is sent each frame set as it is published, while its window has room. It
starts, the moment its delivery is activated, with what the ring kept: nothing published
between its reservation and its activation waits for the next publish.

A newest-order reader is sent one only when it asks, with `request_newest`: the newest frame set
there is at that moment, or the next one published if it already has the newest. Nothing is
held for a reader that is not asking. Holding a frame set ahead of time, the moment it is
published, would hand a reader that was busy for a second a frame that is a second old, and
keep every newer one from it.

The request names the last frame set the reader took. That lets a reader give back a frame
set it did not use (it had stopped waiting when the frame set arrived) and ask again: it gets
the same one back if nothing newer exists, and the newer one otherwise.

#### A camera SDK that owns its buffers

Some capture stacks are given their buffers up front and pick the one they fill, and stop
capturing when they run out. They map onto the second way like this:

- every buffer the SDK was given is a slot, and the group is added with a **backend reserve**
  equal to the number of buffers the SDK needs in hand to keep capturing;
- when the SDK hands a filled buffer over, the backend calls `claim_slot` for it, and
  `publish` once the images of the group are there. An image whose partner does not arrive
  in time is given up with `abandon_slot` and goes straight back to the SDK;
- the slot-free callback returns a buffer to the SDK the moment the ring and every reader
  have let go of it.

Because the budget below counts the reserve, readers can never take so many buffers that the
SDK is left short. A consumer that works on the buffers in place, such as an encoder, is a
reader like any other: it holds a buffer until it has finished consuming it.

#### What publishing, holding and releasing mean

Three different things are true of a slot at different times, and they are kept apart:

- **Ready**: `publish` says the image is complete. A backend publishes a slot only when
  everything that writes it has finished.
- **Held**: a lease says a reader may read the slot. It is not reused or rewritten while any
  lease or the ring refers to it, across a camera restart as well.
- **Done**: `release` says everything the reader started on the slot has finished. For host
  memory that is the moment it stops reading.

When a reader goes away its leases are taken back at once. That is right for host memory: a
reader that is gone reads nothing, and one that closed its connection while still reading
gets changing pixels, never memory that is gone. It is not evidence that work a reader handed
to a GPU or a hardware encoder has finished, so a pool kind where such work can outlive the
connection has to bring its own rule for these slots before it is added.

#### Pool budget

A slot is in use while the ring of recent frame sets or any lease refers to it, so a subscription
is accepted only when the pool can cover it:

```text
slots >= ring history + sum of reader windows + backend reserve
```

Ring history is one frame set, or the largest window among in-order readers. The backend reserve
is the number of slots the backend needs for itself: one to write ahead, more for an SDK that
keeps several captures in flight. A subscription that does not fit is refused with a message that
gives these numbers.

### Shared memory and the socket

The memory behind the slots, and the connection that tells a reader which slot to look at.

#### Host pool

`HostPool` is one stream's slots in host memory: a single anonymous file in memory
(`memfd_create`), with the slots laid out one page-aligned stride apart. The owner maps it
writable, then seals it. A reader is sent the descriptor and maps it with `HostPoolView`, read
only. Both sides derive the layout from the slot count and slot size, which the stream's
description carries.

```cpp
HostPool pool(stream.slot_count, stream.slot_bytes, "camera.left");
std::span<std::byte> pixels = pool.slot(index);          // the owner writes here

HostPoolView view(std::move(received_descriptor), stream.slot_count, stream.slot_bytes);
std::span<const std::byte> image = view.slot(index);     // a reader sees the same bytes
```

Both sides map the pool with its pages already in place (`MAP_POPULATE`), so neither the first
frames a camera writes nor the first frames a reader looks at pay for page faults.

Why an anonymous sealed file and not named shared memory (`shm_open`):

- **Nothing to clean up.** There is no name to collide with another instance and no file left in
  `/dev/shm` after a crash. The memory is freed when the last process that holds it goes away.
- **Readers cannot damage it.** The seals stop any process that receives the descriptor from
  mapping it writable, writing to it, or resizing it. One misbehaving reader cannot change what
  the others see, and cannot make the owner fault by shrinking the file under it.
- **Access follows the connection.** Only a process that was handed the descriptor can map the
  pool.

#### Message socket

`MessageListener` and `MessageSocket` are the connection: a Unix domain socket of the
sequenced-packet kind (`SOCK_SEQPACKET`). The owner listens on a socket file, each reader
connects, and either side sends messages with file descriptors attached.

```cpp
MessageListener listener(path);                          // the owner takes the path
listener.listen();                                       // and accepts once it can answer
auto reader = listener.accept(timeout);
const int descriptors[] = { pool.descriptor() };
reader->send(description_bytes, descriptors);

MessageSocket socket = MessageSocket::connect(path);     // a reader
ReceivedMessage message = socket.receive(buffer, timeout);
HostPoolView view(std::move(message.descriptors[0]), slot_count, slot_bytes);
```

- **Whole messages.** One `send` is one `receive`, in order, so there is no framing to get wrong.
- **Sending never blocks.** If a reader stops draining its queue the sender gets `WOULD_BLOCK`
  and moves on; capture never waits for a reader.
- **A dead peer is seen at once.** When a process exits or crashes the other end reads
  `CLOSED`, which is how the frames a crashed reader held are taken back.
- **Owner only.** The socket file is created with mode `0600`.
- **One listener per path.** A lock file next to the socket file (`<path>.lock`) decides who
  owns the path, so of two processes that start together exactly one gets it, and a socket
  file left by a crashed process is replaced safely. Anything else found at the path is left
  alone. The lock file stays behind, empty.
- **The path is taken first, connections come later.** Constructing a listener locks the path
  and puts the socket file there; `listen()` starts accepting. In between, a connect is refused
  at once (`ECONNREFUSED`) rather than queued. A camera plugin takes its endpoint before it
  opens its cameras, so a second copy of it fails early, and listens when it can answer, so no
  reader is left waiting on a plugin that is still starting.
- **A connect never waits.** It is made or refused at once. A listener whose owner has stopped
  accepting keeps every connection in its queue; when the queue is full a connect is refused
  (`EAGAIN`) instead of waiting for that owner, so one stuck plugin cannot hold up a reader
  that is looking through the endpoints.

#### Where a plugin listens

Each camera plugin is a process of its own with an endpoint of its own. A user's endpoints are
socket files in one directory: `$XDG_RUNTIME_DIR/isaaccapture`, or `/tmp/isaaccapture-<uid>`
when that variable is not set. A plugin makes the directory for its owner alone and refuses one
that others can enter.

`default_endpoint_path(instance)` is the socket path of the plugin with that instance name:
`camera-<instance>.sock` in that directory. A plugin's instance name is by default the name of
its backend, and a reader that is told the instance gets the same path.
`default_endpoint_directory()` is the directory itself.

`default_endpoint_paths()` lists the endpoints in that directory, sorted: the socket files with
those names, without the lock files beside them. A reader that was not told which plugin serves
a camera group looks through this list. The call only reads the directory. It does not connect,
so a socket file left by a plugin that was killed is listed too, and the reader finds out when
it connects. A directory that does not exist yet gives an empty list and is not created.

#### Why a socket and shared memory, and not a messaging library

Three things have to cross between the processes: the frame memory, a small event per frame
set, and the news that a peer has gone.

Frame memory is named by a file descriptor: a sealed memory file here, and the same holds for
the buffers of hardware capture and codec stacks (DMA-BUF) and for GPU memory that can be
exported. Attaching a descriptor to a message on a Unix socket is the standard way to hand one
to another process, so a socket is needed whatever else is used. Once it is there it also
carries the events and reports dead peers, which leaves a messaging library nothing to do.

| Considered | Why it is not used |
|---|---|
| iceoryx2 | It moves self-contained data through host memory that it allocates. Frame memory that exists only as a descriptor would still need a socket beside it, and its C++ binding adds a Rust toolchain to the build. |
| CUDA inter-process handles | A way to share one kind of GPU memory, not a way to connect processes: the handle still has to travel over some connection, and host memory and DMA-BUF are not covered. |
| ZeroMQ, gRPC, DDS | They copy the payload or share host memory only, and none of them pass descriptors. |
| PipeWire, GStreamer | The same mechanism underneath, wrapped in a media graph and, for PipeWire, a daemon. A reader here needs neither. |

### The protocol

What a camera plugin and a reader say to each other over their socket. Each message is one
socket message:

```text
reader                                  plugin
  |  handshake request                     |
  |  (version, group, window, order)       |
  |--------------------------------------->|
  |  handshake response (the group) with   |
  |  one pool descriptor per stream        |
  |<---------------------------------------|
  |  acquire (after which frame set)       |   newest order only: once per frame set
  |--------------------------------------->|
  |  frame set (lease, slot per stream,    |
  |  times)                                |
  |<---------------------------------------|
  |  release (lease)                       |
  |--------------------------------------->|
```

- **One connection, one purpose.** A connection carries one subscription, so the reader says
  what it came for in its first message and the plugin's answer settles it. A reader that
  wants two groups opens two connections. A request that names no group asks only what the
  plugin publishes, and the response lists every group.
- **The answer comes first.** The handshake response is the first message a plugin sends. The
  subscription is reserved, the response is sent, and only then does delivery start.
- **A refusal ends the connection.** The response then carries the reason: a protocol version
  the plugin no longer speaks, a group that does not exist, or a window that does not fit
  the pool.
- **A newest-order reader asks for each frame set.** `acquire` means: hold the newest frame
  set for me, now if there is one after the one I name, otherwise as soon as there is. The
  frame set is chosen when the request is served, so a reader that was busy gets what the
  camera saw last. An in-order reader is sent every frame set without asking.
- **Closing the connection is the unsubscribe.** Every frame set the reader still held goes
  back to the pool.

#### What this protocol is not

It is spoken between two processes on one machine. A lease id means something on one
connection; a slot index and the file descriptors that travel with the handshake response
name memory of that machine. None of that is a network packet's identity, and nothing here
is a recording format. A transport that sends frames to another machine carries what
identifies a capture there (the group's `session_id` and a frame set's `sequence`, its
times, its flags) in its own messages, and a recorder maps the times to its own timestamps.

The descriptors attached to a handshake response are one per stream, in stream order. That
is the rule for host memory, the one kind there is. A kind of memory that needs something
else describes its attachments when it is added.

#### Schema

The messages are FlatBuffers tables in
[`schema/fbs/camera_protocol.fbs`](../schema/fbs/camera_protocol.fbs), with the other
schemas of the repository. They are never recorded, but they are a contract in the same
way: a reader and a plugin can come from different releases. The golden check of
[`schema`](../schema/README.md) therefore guards them like any other schema, and the same
rules apply: fields are appended with fresh ids and left optional, nothing is renumbered.
The numbers of the messages in their union and the bits of the frame flags are written out
in the schema and pinned by the codec and its tests. The one struct, a plane's offset and
stride, is complete as it is and is never extended.

A matching golden says the bytes still line up. It does not say two releases still
understand each other, so the rules below are tested with messages built the way a newer
and an older peer would build them.

#### A reader and a plugin from different releases

- **Version.** Each side has a newest and an oldest protocol version it speaks. The reader
  sends its newest; the plugin answers with the older of that and its own newest, and
  refuses a reader that is older than its oldest. The reader checks the answer against its
  own range. A version is added when one side comes to depend on an operation the other
  must carry out; fields and messages that only inform need none.
- **A message type this build does not know** is ignored when its sender marked it
  `skippable`, which is for messages that only inform. Not marked, it is an operation the
  sender counts on, and the receiver does not carry on as if it had not been sent: a plugin
  drops the reader, which the reader notices, and a reader's `open` or `next` fails with
  the reason.
- **A group this build cannot read**, because it uses a pixel format or a kind of memory
  added later, is still listed, with the reason. The groups next to it are
  unaffected, and a reader that asks for it is told why it cannot have it.
- **A request for something added later**, such as a delivery order, is refused with the
  reason.

#### Codec

`MessageEncoder` and `decode_message` turn the types of `types.hpp` into bytes and back, so
nothing outside the codec sees FlatBuffers. The encoder reuses its buffer; a frame set is
encoded without allocating. It refuses a message that names more streams, images or planes
than a group can have, and writes nothing in its place.

The decoder treats its input as coming from another process: it verifies the buffer and bounds
the counts that have a limit: streams per group, planes per stream, images per frame set. That
establishes that the bytes are a message, not that what they describe makes sense. Whether a
group's planes fit its slots is checked by `problem_with`, by the plugin for what a backend
declares and by the reader for what it was sent.

### The endpoint

`Endpoint` is a camera plugin's side of the conversation with readers. It puts a `Publisher`
on a socket and speaks the protocol.

```cpp
Publisher publisher;
Endpoint endpoint(publisher, socket_path);             // takes the path

// ... the plugin opens its cameras, and so knows what each one publishes ...
const GroupId head = publisher.add_group(description);
HostPool left(slots, bytes, "head.left"), right(slots, bytes, "head.right");
endpoint.set_group_memory(head, { left.descriptor(), right.descriptor() });
endpoint.start();                                      // readers can connect from here

// ... the backend acquires slots, writes pixels and publishes; readers come and go ...
endpoint.stop();                                       // before the pools go away
```

#### When readers can reach it

An endpoint is visible to readers from `start()` to `stop()`, and at no other time.

- **Constructing it takes the path.** The socket file is there and the path is locked, so a
  second process that wants the same path fails at once with "another process has taken ...",
  before it has opened a camera.
- **Until `start()` a reader that connects is refused at once.** It is not accepted and then
  left waiting for an answer nobody sends. A reader that looks through the user's camera
  plugins ([How a reader finds its camera](#how-a-reader-finds-its-camera)) passes over such
  an endpoint without waiting. So a plugin constructs its endpoint first, opens its cameras,
  and calls `start()` when it can answer.
- **`stop()` gives the path up.** The socket file is removed and a reader is refused again.
  A later `start()` takes the path again.

#### What runs where

- **One thread of its own** accepts readers and handles what they send: the handshake, the
  requests for the newest frame set and the releases. It sleeps in `poll` until a reader
  sends something.
- **A frame set is sent from the thread that publishes** to a reader that is waiting for it:
  straight from `Publisher::publish`, with a send that does not block. There is no queue and
  no hand-over to another thread between a frame set being published and its event being on
  the socket. A frame set that already exists when a newest-order reader asks, and the one
  that follows a release, are sent from the endpoint's thread.
- **The answer to a handshake comes before any frame set.** The subscription is reserved,
  which counts it in the pool's budget, then the response is sent, then delivery starts.
  Capture does not wait for any of it. A reader that is gone before it could be answered
  leaves no reservation behind.

#### When a reader goes away

A closed connection is the signal, whether the reader exited, crashed or was killed. Its
subscription ends and every frame set it held goes back to the pool. A reader that sends
bytes that are not a message, a message only a plugin sends, or an operation this build
does not know, is dropped the same way; other readers are not affected.

Why taking the frame sets back at once is right for host memory, and what another kind of
memory needs before it is served from here, is in
[What publishing, holding and releasing mean](#what-publishing-holding-and-releasing-mean).

#### Logging

`isaaccapture.camera.Endpoint` logs a reader subscribing, being refused and leaving.
A reader that only asks what is published is logged at debug, because a reader that looks for
its camera asks that of every plugin of its user. Nothing is logged per frame.

## Tests

`tests/cpp/core/camera`, built as one executable, `camera_tests`. `ctest -L camera` runs them. A
label is a regular expression, so that also runs the tests of the camera plugins, whose labels
start with `camera_`, and in a build with the Python bindings the Python tests below
(`camera_python`); `ctest -L "^camera$"` is the form that selects this library's tests alone.

- **Unit** (`ctest -L unit`): everything that runs in one process. The endpoint's cases play the
  reader's part with the socket and the codec directly. The reader's cases run it against a
  real endpoint in the same process, and against a stand-in plugin for what a real one does not
  do on demand. The cases about finding a plugin point `XDG_RUNTIME_DIR` at a temporary
  directory and put there what the case is about: plugins with different groups or the same
  one, a plugin that has not started, a socket file nobody is behind, a plugin that does not
  answer, a plugin that refuses the reader or goes away while it is asked.

  The publishing side's cases use stand-in backends, and what they serve is read with a real
  reader in the same process. Camera arguments, options and configuration files: a camera of
  the command line is the plugin's backend's; a plugin takes its own cameras from a file it
  shares with other plugins; a file with cameras and none for the plugin's backend is refused
  with what it names. The host with a camera that publishes on demand, with one whose SDK owns
  the buffers, with one that cannot start, and across a camera restart. `run_camera_plugin` on
  a thread, stopped with the signal a plugin process is stopped with: the instance name is the
  backend's unless it is given; the backend is opened once with all its cameras; what is
  served without arguments; what ends a plugin with an error; what is answered with the usage.
- **Integration** (`ctest -L integration`): the cases that start a second process. One sends
  it a pool over the socket: it maps the pool, sees the owner's writes and cannot write itself.
  The others have the reader in the test process and `camera_test_peer`
  (`tests/cpp/core/camera/test_peer`) as the other process, over the real socket. A reader gets
  whole frame sets from a plugin in another process; a reader that was away for sixty frame
  periods is handed a frame set published while it was away; a reader process that is killed
  while it holds frame sets gives them back; a reader is told when the plugin's process is
  killed and can still read what it holds; a reader finds two groups on two plugin processes by
  name, and still finds the one after the other was killed.

`tests/python/core/camera` (`ctest -L camera_python`) reads from the synthetic camera plugin
that is bundled into the package, started as a process and, in one case, through the plugin
manager. Besides the reading, the cases cover the finding, each in a user's directories of
its own: a reader that is told nothing finds the plugin; a listing names the endpoint of its
plugin; two plugins side by side are both found by group; a group on two plugins, and a
group nobody publishes, are errors that name the endpoints and the directory.
