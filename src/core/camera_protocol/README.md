<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Camera interface: the protocol between the plugin and a reader

What the camera plugin and a reader say to each other over their socket
([`camera_ipc`](../camera_ipc)). There are four messages, each one socket message:

```text
reader                                  plugin
  |  handshake request                     |
  |  (version, group, window, order)       |
  |--------------------------------------->|
  |  handshake response (the group) with   |
  |  one pool descriptor per stream        |
  |<---------------------------------------|
  |  frame (lease, slot per image, times)  |   once per frame set
  |<---------------------------------------|
  |  release (lease)                       |
  |--------------------------------------->|
```

- **One connection, one purpose.** A connection carries one subscription, so the reader says
  what it came for in its first message and the plugin's answer settles it. A reader that
  wants two groups opens two connections. A request that names no group asks only what the
  plugin publishes, and the response lists every group.
- **A refusal ends the connection.** The response then carries the reason: another protocol
  version, a group that does not exist, or a window that does not fit the pool.
- **Closing the connection is the unsubscribe.** Every frame set the reader still held goes
  back to the pool.
- **A frame can overtake the handshake response,** because capture does not wait for the
  answer to be sent. It belongs to the reader all the same.

## Schema

The messages are FlatBuffers tables in
[`schema/fbs/camera_protocol.fbs`](../schema/fbs/camera_protocol.fbs), with the other
schemas of the repository. They are never recorded, but they are a contract in the same
way: a reader and a plugin can come from different releases. The golden check of
[`schema`](../schema/README.md) therefore guards them like any other schema, and the same
rules apply: fields are appended with fresh ids and left optional, nothing is renumbered,
and tables are used rather than structs.

On top of that:

- A message type this build does not know decodes to nothing and is ignored.
- A stream whose format or kind of memory this build does not know makes its group
  unreadable, and the reader is told so.
- `PROTOCOL_VERSION` changes only when the two sides would misread each other even though
  the schema is compatible. The plugin refuses a reader with another version.

## Codec

`MessageEncoder` and `decode_message` turn [`camera_core`](../camera_core) types into bytes
and back, so nothing outside this module sees FlatBuffers. The encoder reuses its buffer; a
frame is encoded without allocating. The decoder treats its input as coming from another
process: it verifies the buffer and refuses counts that would not fit a group.

## Tests

`tests/cpp/core/camera_protocol`, run with `ctest -L camera_protocol`. They are tagged `unit`.
