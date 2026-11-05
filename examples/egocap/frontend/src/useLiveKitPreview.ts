// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * One LiveKit room connection for every preview tile.
 *
 * Connects once the backend says a preview is configured, picks each configured track out of the camera host's
 * participant (whether it was published before we joined or after), and lets the SDK handle ordinary network
 * reconnects. Only a full disconnect starts a new attempt, with a freshly fetched token.
 */

import { useEffect, useState } from "react";
import { Room, RoomEvent, Track } from "livekit-client";
import type { RemoteParticipant, RemoteTrack } from "livekit-client";
import { fetchLiveKitPreview, LiveKitPreviewHttpError, LiveKitPreviewParseError } from "./livekitPreview";
import type { LiveKitPreviewConfig } from "./livekitPreview";

/** Delay before a new attempt after a failed connect or a full disconnect. */
const RETRY_DELAY_MS = 3000;

export type LiveKitPhase = "loading" | "unconfigured" | "connecting" | "connected" | "reconnecting" | "error";

export interface LiveKitTile {
  name: string;
  /** The subscribed video track, or null while the camera host has not published it. */
  track: RemoteTrack | null;
}

export interface LiveKitPreviewState {
  phase: LiveKitPhase;
  /** Why the preview is unavailable, when ``phase`` is "error". */
  message: string | null;
  /** One entry per configured track, in configuration order; empty until the configuration is known. */
  tiles: LiveKitTile[];
}

const INITIAL: LiveKitPreviewState = { phase: "loading", message: null, tiles: [] };

/** The configured tiles with the given subscribed tracks filled in by track name. */
function tilesFor(config: LiveKitPreviewConfig, tracks: Map<string, RemoteTrack>): LiveKitTile[] {
  return config.tracks.map((t) => ({ name: t.name, track: tracks.get(t.track) ?? null }));
}

/** Connects to the configured LiveKit room for the lifetime of the calling component. */
export function useLiveKitPreview(): LiveKitPreviewState {
  const [state, setState] = useState<LiveKitPreviewState>(INITIAL);

  useEffect(() => {
    // `cancelled` ends every in-flight attempt and pending retry when the component unmounts.
    let cancelled = false;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;
    let room: Room | null = null;
    const controller = new AbortController();

    function scheduleRetry(): void {
      if (cancelled || retryTimer !== null) return;
      retryTimer = setTimeout(() => {
        retryTimer = null;
        void attempt();
      }, RETRY_DELAY_MS);
    }

    function fail(message: string, config: LiveKitPreviewConfig | null): void {
      if (cancelled) return;
      setState({ phase: "error", message, tiles: config === null ? [] : tilesFor(config, new Map()) });
      scheduleRetry();
    }

    async function attempt(): Promise<void> {
      let config: LiveKitPreviewConfig | null;
      try {
        // A fresh token for every attempt: the previous one may have expired while we were disconnected.
        config = await fetchLiveKitPreview(controller.signal);
      } catch (error) {
        if (cancelled) return;
        // Say why the preview is unavailable, including when the backend could not be reached at all: staying
        // silent would make a configured station look like one with no cameras. The retry continues either way.
        const reason =
          error instanceof LiveKitPreviewHttpError || error instanceof LiveKitPreviewParseError
            ? error.message
            : "Cannot reach the backend";
        fail(reason, null);
        return;
      }
      if (cancelled) return;
      if (config === null) {
        setState({ phase: "unconfigured", message: null, tiles: [] });
        return;
      }
      const active = config;
      const tracks = new Map<string, RemoteTrack>();
      const publish = (phase: LiveKitPhase): void => {
        if (!cancelled) setState({ phase, message: null, tiles: tilesFor(active, tracks) });
      };

      /** Records the camera host's configured video tracks that are already subscribed. */
      const adopt = (participant: RemoteParticipant): void => {
        if (participant.identity !== active.publisher) return;
        for (const publication of participant.trackPublications.values()) {
          if (publication.kind === Track.Kind.Video && publication.track) {
            tracks.set(publication.trackName, publication.track);
          }
        }
      };

      const next = new Room();
      room = next;
      next
        .on(RoomEvent.TrackSubscribed, (track, publication, participant) => {
          if (participant.identity !== active.publisher || track.kind !== Track.Kind.Video) return;
          tracks.set(publication.trackName, track);
          publish("connected");
        })
        .on(RoomEvent.TrackUnsubscribed, (track, publication) => {
          // Only drop the mapping if it is still this track: a republication may already have replaced it.
          if (tracks.get(publication.trackName) === track) tracks.delete(publication.trackName);
          publish("connected");
        })
        .on(RoomEvent.ParticipantDisconnected, (participant) => {
          if (participant.identity !== active.publisher) return;
          tracks.clear();
          publish("connected");
        })
        .on(RoomEvent.Reconnecting, () => publish("reconnecting"))
        .on(RoomEvent.Reconnected, () => {
          // The SDK resumed the session; pick up anything published while it was away.
          next.remoteParticipants.forEach(adopt);
          publish("connected");
        })
        .on(RoomEvent.Disconnected, () => {
          // Not a reconnect: the SDK gave up or the server closed the room. Start over with a new token.
          tracks.clear();
          fail("Disconnected from the LiveKit room", active);
        });

      publish("connecting");
      try {
        await next.connect(active.url, active.token);
      } catch (error) {
        if (!cancelled) fail(error instanceof Error ? error.message : String(error), active);
        return;
      }
      if (cancelled) {
        void next.disconnect();
        return;
      }
      // Tracks published before we joined arrive here rather than through TrackSubscribed.
      next.remoteParticipants.forEach(adopt);
      publish("connected");
    }

    void attempt();

    return () => {
      cancelled = true;
      controller.abort();
      if (retryTimer !== null) clearTimeout(retryTimer);
      // Detaches every track and closes the connection; the Disconnected handler above is a no-op once cancelled.
      void room?.disconnect();
      room = null;
    };
  }, []);

  return state;
}
