<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Camera interface: the endpoint

The camera plugin's side of the conversation with readers. It puts a
[`camera_core`](../camera_core) `Publisher` on a socket ([`camera_ipc`](../camera_ipc)) and
speaks the [`camera_protocol`](../camera_protocol).

```cpp
Publisher publisher;
const GroupId head = publisher.add_group(description);
HostPool left(slots, bytes, "head.left"), right(slots, bytes, "head.right");

Endpoint endpoint(publisher, socket_path);
endpoint.set_group_memory(head, { left.descriptor(), right.descriptor() });
endpoint.start();
// ... the backend acquires slots, writes pixels and publishes; readers come and go ...
```

## What runs where

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

## When a reader goes away

A closed connection is the signal, whether the reader exited, crashed or was killed. Its
subscription ends and every frame set it held goes back to the pool. A reader that sends
bytes that are not a message, a message only a plugin sends, or an operation this build
does not know, is dropped the same way; other readers are not affected.

Taking the frame sets back at once is right for host memory, which is all there is today: a
reader that is gone reads nothing. A closed connection is not evidence that work a reader
handed to a GPU or a hardware encoder has finished, so a kind of memory where that matters
needs its own rule for these slots before it is served from here (see
[`camera_core`](../camera_core), "What publishing, holding and releasing mean").

## Logging

`isaaccapture.camera_endpoint.Endpoint` logs readers arriving, subscribing, being refused and
leaving. Nothing is logged per frame.

## Tests

`tests/cpp/core/camera_endpoint`, run with `ctest -L camera_endpoint`. They are tagged `unit`
and play the reader's part with the socket and the codec directly.
