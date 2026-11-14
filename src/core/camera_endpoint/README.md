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

- **One thread of its own** accepts readers and handles what they send: the handshake
  and the releases. It sleeps in `poll` until a reader sends something.
- **Frame events are sent from the thread that publishes**, straight from
  `Publisher::publish`, with a send that does not block. There is no queue and no hand-over
  to another thread between a frame set being published and its event being on the socket.
  The event that follows a release is sent from the endpoint's thread for the same reason.
- A reader's subscription takes effect before the answer to it is sent, so a `Frame` can
  reach the reader ahead of the handshake response.

## When a reader goes away

A closed connection is the signal, whether the reader exited, crashed or was killed. Its
subscription ends and every frame set it held goes back to the pool. A reader that sends
bytes that are not a message, or a message only a plugin sends, is dropped the same way;
other readers are not affected.

## Logging

`isaaccapture.camera_endpoint.Endpoint` logs readers arriving, subscribing, being refused and
leaving. Nothing is logged per frame.

## Tests

`tests/cpp/core/camera_endpoint`, run with `ctest -L camera_endpoint`. They are tagged `unit`
and play the reader's part with the socket and the codec directly.
