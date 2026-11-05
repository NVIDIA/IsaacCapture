// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * The credential for the LiveKit video CaptureServer offers.
 *
 * Which streams exist (URL, room, publisher, tracks) comes from CaptureServer in the session snapshot. This
 * only fetches the viewer token, so a backend without signing keys can still list the streams.
 */

/** The backend answered, but did not issue a token (e.g. 503: signing credentials missing). */
export class LiveKitTokenHttpError extends Error {}

/** The backend's answer was not the expected `{token}` body. */
export class LiveKitTokenParseError extends Error {}

/** Fetches a fresh subscribe-only token for `room`; throws LiveKitTokenHttpError with the backend's reason. */
export async function fetchLiveKitToken(room: string, signal?: AbortSignal): Promise<string> {
  const response = await fetch(`/api/livekit-token?room=${encodeURIComponent(room)}`, { signal });
  if (!response.ok) {
    // The backend's own reason (e.g. missing signing credentials) is the useful part for the operator.
    let detail = `HTTP ${response.status}`;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      // Keep the status-only message.
    }
    throw new LiveKitTokenHttpError(detail);
  }
  const body = (await response.json()) as { token?: unknown };
  if (typeof body.token !== "string" || body.token === "") {
    throw new LiveKitTokenParseError("token must be a nonempty string");
  }
  return body.token;
}

/** The backend's measured delay from a stream's publisher stamp to its own receipt, per stream name. */
export interface LiveKitLatencyEntry {
  basis: string;
  summary: { median_ms: number; p95_ms: number; samples: number } | null;
  note: string;
}

/** Fetches the backend-side stamp-to-receipt delay of the offered LiveKit streams. */
export async function fetchLiveKitLatency(signal?: AbortSignal): Promise<Record<string, LiveKitLatencyEntry>> {
  const response = await fetch("/api/livekit-latency", { signal });
  if (!response.ok) return {};
  const body = (await response.json()) as { streams?: Record<string, LiveKitLatencyEntry> };
  return body.streams ?? {};
}
