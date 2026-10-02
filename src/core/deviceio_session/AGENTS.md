<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent notes — `deviceio_session`

**CRITICAL (non-optional):** Before editing this package, complete the mandatory **`AGENTS.md` preflight** in [`../../../AGENTS.md`](../../../AGENTS.md) (read every applicable `AGENTS.md` on your paths, not just this file).

## Update loop

- **`DeviceIOSession::update`** reads the clock once with **`core::os_monotonic_now_ns()`** (via `#include <oxr_utils/os_time.hpp>`) and passes that value to **`ITrackerImpl::update(int64_t)`** for every registered impl.
- **No** session-owned **`XrTimeConverter`** is required solely to drive that loop (OpenXR conversion stays inside live impls).

## Sessions without OpenXR

- **`DeviceIOSession::run` accepts null `OpenXRSessionHandles` only when `requires_openxr()` is false** (every tracker's live impl opts out via `static constexpr bool requires_openxr = false`). Anything else throws `std::invalid_argument`. A new impl that never touches the handles should opt out; one that does must not.

## Implementation / includes

- **`deviceio_session.cpp`**: if the TU uses **`XR_NULL_HANDLE`** or other OpenXR macros, include **`<openxr/openxr.h>`** explicitly after the session header so **`XR_NO_PROTOTYPES`** is already established by **`oxr_utils/oxr_funcs.hpp`** pulled in through **`deviceio_session.hpp`**.
- **`mcap_impl.cpp`** is the sole TU that defines **`MCAP_IMPLEMENTATION`** and includes **`<mcap/writer.hpp>`** and **`<mcap/reader.hpp>`**. All other TUs (`deviceio_session.cpp`, `replay_session.cpp`) get declarations only.

## Replay session

- **`ReplaySession`** lives in `replay_session.hpp` / `replay_session.cpp` alongside `DeviceIOSession`. Uses **`McapReplayConfig`** (not `McapRecordingConfig`).
- Factory method is **`ReplaySession::run(const McapReplayConfig&)`**.

## Related docs

- Tracker interface contract: [`../deviceio_base/AGENTS.md`](../deviceio_base/AGENTS.md)
- Live factory + impls: [`../live_trackers/AGENTS.md`](../live_trackers/AGENTS.md)
- Replay factory + impls: [`../replay_trackers/AGENTS.md`](../replay_trackers/AGENTS.md)
