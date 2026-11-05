// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * The backend's measured stamp-to-receipt delay of each LiveKit stream, polled once per second while any LiveKit
 * tile is shown. The first request starts the backend's subscription, which stops by itself when requests stop.
 */

import { useEffect, useState } from "react";
import { fetchLiveKitLatency } from "./livekitPreview";
import type { LiveKitLatencyEntry } from "./livekitPreview";

const POLL_INTERVAL_MS = 1000;

export function useLiveKitLatency(active: boolean): Record<string, LiveKitLatencyEntry> {
  const [latency, setLatency] = useState<Record<string, LiveKitLatencyEntry>>({});

  useEffect(() => {
    if (!active) {
      setLatency({});
      return;
    }
    const controller = new AbortController();
    let cancelled = false;
    const poll = async (): Promise<void> => {
      try {
        const next = await fetchLiveKitLatency(controller.signal);
        if (!cancelled) setLatency(next);
      } catch {
        // The backend is unreachable or the request was aborted: keep the last value until the next poll.
      }
    };
    void poll();
    const timer = setInterval(() => void poll(), POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      controller.abort();
      clearInterval(timer);
    };
  }, [active]);

  return latency;
}
