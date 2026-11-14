<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Camera interface: sharing frames between processes

The process that owns the cameras and the processes that read them are on one machine. This
module is what crosses between them. Linux only.

[`camera_core`](../camera_core) decides which slot holds which frame; this module provides the
memory behind the slots and the connection that tells a reader which slot to look at.

## Host pool

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

## Message socket

`MessageListener` and `MessageSocket` are the connection: a Unix domain socket of the
sequenced-packet kind (`SOCK_SEQPACKET`). The owner listens on a socket file, each reader
connects, and either side sends messages with file descriptors attached.

```cpp
MessageListener listener(path);                          // the owner
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
  owns the path, so of two processes that start together exactly one listens, and a socket
  file left by a crashed process is replaced safely. Anything else found at the path is left
  alone. The lock file stays behind, empty.

## Where the plugin listens

`default_endpoint_path()` is the socket path a camera plugin and its readers agree on without
being told: `$XDG_RUNTIME_DIR/isaaccapture/camera.sock`, or under `/tmp/isaaccapture-<uid>/`
when that variable is not set. A second plugin on the same machine is a named instance and
listens on `camera-<instance>.sock` beside it. The plugin makes the directory for its owner
alone and refuses one that others can enter.

## Why a socket and shared memory, and not a messaging library

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

`tests/cpp/core/camera_ipc`, run with `ctest -L camera_ipc`. They are tagged `unit`, so
`ctest -L unit` runs them too.

One case is tagged `integration` instead: a second process is sent a pool over the socket, maps
it, sees the owner's writes and cannot write itself.
