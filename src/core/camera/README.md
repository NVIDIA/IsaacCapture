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
- **Sequence**: a frame set's place among the group's captures in the session. It is the same
  for every reader, so a reader that skips sees a gap, and with the session id it names a
  capture for good.
- **Generation**: counts restarts of the camera within the session, so a reader can tell a
  restart from a skip. A restart does not reset the sequence.

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

## Tests

`tests/cpp/core/camera`, built as one executable, `camera_tests`. `ctest -L camera` runs them; a
label is a regular expression, so `ctest -L "^camera$"` is the form that selects this library's
tests alone. They are tagged `unit`, so `ctest -L unit` runs them too.

One case is tagged `integration` instead: a second process is sent a pool over the socket, maps
it, sees the owner's writes and cannot write itself.
