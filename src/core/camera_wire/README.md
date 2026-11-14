<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Camera interface: messages between the plugin and a reader

What the camera plugin and a reader say to each other over their socket
([`camera_ipc`](../camera_ipc)). Each socket message is one of these:

```text
reader                               plugin
  |  Hello (protocol version)           |
  |------------------------------------>|
  |       Welcome (what is published)   |
  |<------------------------------------|
  |  Subscribe (group, window, order)   |
  |------------------------------------>|
  |  Subscribed + one pool descriptor   |
  |             per stream              |
  |<------------------------------------|
  |       Frame (lease, slot per image) |   once per frame set
  |<------------------------------------|
  |  Release (lease)                    |
  |------------------------------------>|
```

A connection carries one subscription; a reader that wants two groups opens two. Closing the
connection ends the subscription and returns every frame set the reader still held. A `Frame`
can overtake `Subscribed`, because capture does not wait for the answer to be sent; it belongs
to the subscription all the same.

## Schema

[`fbs/camera_wire.fbs`](fbs/camera_wire.fbs) defines the messages as FlatBuffers tables. It is
kept here and not in [`schema`](../schema): that directory holds what goes into recordings,
and these messages are never recorded.

A reader and a plugin may come from different builds, so the schema only grows:

- Append fields with fresh ids and leave them optional; append union members. Never reuse or
  renumber an id, and use tables, not structs.
- A message type this build does not know decodes to nothing and is ignored.
- A stream whose format or kind of memory this build does not know makes its group
  unreadable, and the reader is told so; other groups are unaffected.
- `WIRE_PROTOCOL_VERSION` changes only when the two sides would misread each other. The
  plugin refuses a reader with a different version in `Welcome`.

## Codec

`WireEncoder` and `decode_wire_message` turn [`camera_core`](../camera_core) types into bytes
and back, so nothing outside this module sees FlatBuffers. The encoder reuses its buffer; a
frame event is encoded without allocating. The decoder treats its input as coming from another
process: it verifies the buffer and refuses counts that would not fit a group.

## Tests

`tests/cpp/core/camera_wire`, run with `ctest -L camera_wire`. They are tagged `unit`.
