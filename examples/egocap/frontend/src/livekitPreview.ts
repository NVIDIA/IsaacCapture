// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Wire shape of GET /api/livekit-preview: where the camera host publishes and a viewer token.
 *
 * These cameras are display-only. They come straight from LiveKit to the browser and are neither
 * requested from CaptureServer nor recorded.
 */

export interface LiveKitPreviewTrack {
  /** Operator-facing tile label. */
  name: string;
  /** The published video track this tile shows. */
  track: string;
}

export interface LiveKitPreviewConfig {
  url: string;
  room: string;
  /** Participant identity of the camera host; a track name is only unique within one participant. */
  publisher: string;
  token: string;
  tracks: LiveKitPreviewTrack[];
}

export class LiveKitPreviewParseError extends Error {}

/** The backend answered, but not with a usable preview configuration (e.g. 503: signing credentials missing). */
export class LiveKitPreviewHttpError extends Error {}

function requireString(value: unknown, field: string): string {
  if (typeof value !== "string" || value === "") {
    throw new LiveKitPreviewParseError(`${field} must be a nonempty string`);
  }
  return value;
}

/** Validates a parsed GET /api/livekit-preview body; anything else is a protocol error, never a partial config. */
export function parseLiveKitPreview(body: unknown): LiveKitPreviewConfig {
  if (typeof body !== "object" || body === null) throw new LiveKitPreviewParseError("body must be an object");
  const raw = body as Record<string, unknown>;
  if (!Array.isArray(raw.tracks) || raw.tracks.length === 0) {
    throw new LiveKitPreviewParseError("tracks must be a nonempty array");
  }
  const tracks = raw.tracks.map((item, i) => {
    if (typeof item !== "object" || item === null) throw new LiveKitPreviewParseError(`tracks[${i}] must be an object`);
    const entry = item as Record<string, unknown>;
    return { name: requireString(entry.name, `tracks[${i}].name`), track: requireString(entry.track, `tracks[${i}].track`) };
  });
  return {
    url: requireString(raw.url, "url"),
    room: requireString(raw.room, "room"),
    publisher: requireString(raw.publisher, "publisher"),
    token: requireString(raw.token, "token"),
    tracks,
  };
}

/** Fetches the preview config, or null when this station has no LiveKit preview configured (HTTP 404). */
export async function fetchLiveKitPreview(signal?: AbortSignal): Promise<LiveKitPreviewConfig | null> {
  const response = await fetch("/api/livekit-preview", { signal });
  if (response.status === 404) return null;
  if (!response.ok) {
    // The backend's own reason (e.g. missing signing credentials) is the useful part for the operator.
    let detail = `HTTP ${response.status}`;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      // Keep the status-only message.
    }
    throw new LiveKitPreviewHttpError(detail);
  }
  return parseLiveKitPreview(await response.json());
}
