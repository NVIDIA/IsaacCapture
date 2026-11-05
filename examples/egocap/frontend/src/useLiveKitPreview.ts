// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * One LiveKit room connection for every offered video stream.
 *
 * CaptureServer says which streams exist; this connects to their room only to receive the video. It picks each
 * stream's track out of the camera host's participant (whether published before we joined or after), and lets the
 * SDK handle ordinary network reconnects. Only a full disconnect starts a new attempt, with a fresh token.
 */

import { useEffect, useState } from "react";
import { Room, RoomEvent, Track } from "livekit-client";
import type { RemoteParticipant, RemoteTrack } from "livekit-client";
import type { LiveKitStream } from "./captureProtocol";
import { fetchLiveKitToken } from "./livekitPreview";

/** Delay before a new attempt after a failed connect or a full disconnect. */
const RETRY_DELAY_MS = 3000;

export type LiveKitPhase = "idle" | "connecting" | "connected" | "reconnecting" | "error";

export interface LiveKitTile {
  name: string;
  /** The subscribed video track, or null while the camera host has not published it. */
  track: RemoteTrack | null;
}

export interface LiveKitPreviewState {
  phase: LiveKitPhase;
  /** Why video is unavailable, when ``phase`` is "error". */
  message: string | null;
  /** One entry per offered stream, in the order offered; empty when nothing is offered. */
  tiles: LiveKitTile[];
}

const IDLE: LiveKitPreviewState = { phase: "idle", message: null, tiles: [] };

/** The offered streams with the given subscribed tracks filled in by track name. */
function tilesFor(streams: LiveKitStream[], tracks: Map<string, RemoteTrack>): LiveKitTile[] {
  return streams.map((s) => ({ name: s.name, track: tracks.get(s.track) ?? null }));
}

/**
 * Receives the offered LiveKit streams for the lifetime of the calling component. All streams of one session
 * share a room and publisher, so the first stream names the connection.
 */
export function useLiveKitPreview(streams: LiveKitStream[]): LiveKitPreviewState {
  const [state, setState] = useState<LiveKitPreviewState>(IDLE);
  // A new session offering the same streams must not reconnect, so the effect depends on content, not identity.
  const key = JSON.stringify(streams);

  useEffect(() => {
    const offered: LiveKitStream[] = JSON.parse(key);
    if (offered.length === 0) {
      setState(IDLE);
      return;
    }
    const { url, room: roomName, publisher } = offered[0];
    // `cancelled` ends every in-flight attempt and pending retry when the streams change or the component unmounts.
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

    function fail(message: string): void {
      if (cancelled) return;
      setState({ phase: "error", message, tiles: tilesFor(offered, new Map()) });
      scheduleRetry();
    }

    async function attempt(): Promise<void> {
      const tracks = new Map<string, RemoteTrack>();
      const publish = (phase: LiveKitPhase): void => {
        if (!cancelled) setState({ phase, message: null, tiles: tilesFor(offered, tracks) });
      };
      publish("connecting");
      let token: string;
      try {
        // A fresh token for every attempt: the previous one may have expired while we were disconnected.
        token = await fetchLiveKitToken(roomName, controller.signal);
      } catch (error) {
        // Say why video is unavailable, including when the backend could not be reached at all.
        if (!cancelled) fail(error instanceof Error && error.name !== "TypeError" ? error.message : "Cannot reach the backend");
        return;
      }
      if (cancelled) return;

      /** Records the camera host's offered video tracks that are already subscribed. */
      const adopt = (participant: RemoteParticipant): void => {
        if (participant.identity !== publisher) return;
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
          if (participant.identity !== publisher || track.kind !== Track.Kind.Video) return;
          tracks.set(publication.trackName, track);
          publish("connected");
        })
        .on(RoomEvent.TrackUnsubscribed, (track, publication) => {
          // Only drop the mapping if it is still this track: a republication may already have replaced it.
          if (tracks.get(publication.trackName) === track) tracks.delete(publication.trackName);
          publish("connected");
        })
        .on(RoomEvent.ParticipantDisconnected, (participant) => {
          if (participant.identity !== publisher) return;
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
          fail("Disconnected from the LiveKit room");
        });

      try {
        await next.connect(url, token);
      } catch (error) {
        if (!cancelled) fail(error instanceof Error ? error.message : String(error));
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
  }, [key]);

  return state;
}
