// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Integration-level tests for useCaptureSession's effect: real timers
 * (faked), a controlled WebSocket stub, and the actual ordering/ownership
 * bugs this hook was built to avoid -- not just the pure reducer's state
 * table (see useCaptureSession.test.ts for that). A fatal-message ->
 * staleTimer-not-cleared -> unwanted-reconnect bug already slipped past
 * reducer-only tests once; these exercise the real effect.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, renderHook } from "@testing-library/react";
import { useCaptureSession, RecordingCommandUncertainError } from "./useCaptureSession";
import { PREVIEW_VERSION, SNAPSHOT_TYPE, BINARY_HEADER_SIZE, BINARY_KIND_TRACKING } from "./captureProtocol";

class FakeWebSocket {
  static instances: FakeWebSocket[] = [];
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  url: string;
  binaryType = "blob";
  readyState = FakeWebSocket.CONNECTING;
  closed = false;
  onopen: (() => void) | null = null;
  onmessage: ((ev: { data: unknown }) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  /** Every text frame actually sent, for the recording-command round-trip tests. */
  sent: string[] = [];

  constructor(url: string) {
    this.url = url;
    FakeWebSocket.instances.push(this);
  }

  close(): void {
    if (this.closed) return;
    this.closed = true;
    this.readyState = FakeWebSocket.CLOSED;
    this.onclose?.();
  }

  send(data: string): void {
    this.sent.push(data);
  }

  // --- test-only driver methods, not part of the real WebSocket API ---
  open(): void {
    this.readyState = FakeWebSocket.OPEN;
    this.onopen?.();
  }

  emitText(data: string): void {
    this.onmessage?.({ data });
  }

  emitBinary(data: ArrayBuffer): void {
    this.onmessage?.({ data });
  }
}

function validSnapshotBody(overrides: Record<string, unknown> = {}): string {
  return JSON.stringify({
    version: PREVIEW_VERSION,
    type: SNAPSHOT_TYPE,
    backend_instance_id: "b1",
    revision: 1,
    station_label: "Station",
    profile_label: "Profile",
    capture: {
      connecting: false,
      started: null,
      status: null,
      stopped: null,
      error_message: null,
      binary_generation: null,
    },
    samples: { tracking: null, cameras: [] },
    ...overrides,
  });
}

function trackingStreamJson(): Record<string, unknown> {
  return {
    kind: 1,
    stream_id: 0,
    schema: 1,
    pose_representation: 1,
    quaternion_order: 1,
    position_unit: 1,
    axes: 1,
    hand_joint_order: 1,
    layout: {
      payload_size: 13,
      fields: [
        { name: "head.present", dtype: 1, shape: [], offset: 0 },
        { name: "head.position", dtype: 2, shape: [3], offset: 1 },
      ],
    },
  };
}

function runningSnapshotBody(generation: number, revision = 2): string {
  return validSnapshotBody({
    revision,
    capture: {
      connecting: false,
      started: { session_id: "s1", running: false, mode: 0, streams: [trackingStreamJson()] },
      status: { session_id: "s1", type: 2, running: true, startup_stage: null, message: null, status: null },
      stopped: null,
      error_message: null,
      binary_generation: generation,
    },
  });
}

function trackingBinaryFrame(generation: number, sequence: bigint): ArrayBuffer {
  const buf = new ArrayBuffer(BINARY_HEADER_SIZE + 13);
  const view = new DataView(buf);
  view.setUint32(0, generation, true);
  view.setUint8(4, BINARY_KIND_TRACKING);
  view.setUint8(5, 0);
  view.setUint16(6, 0, true);
  view.setBigUint64(8, sequence, true);
  view.setBigUint64(16, 1000n, true);
  view.setBigUint64(24, 2000n, true);
  view.setUint8(32, 1); // head.present = true
  view.setFloat32(33, 1, true);
  view.setFloat32(37, 2, true);
  view.setFloat32(41, 3, true);
  return buf;
}

/** A tracking binary frame with an explicit, possibly-wrong payload byte
 * count -- header fields are otherwise valid for the given generation. */
function trackingBinaryFrameWithPayloadLength(generation: number, sequence: bigint, payloadLength: number): ArrayBuffer {
  const buf = new ArrayBuffer(BINARY_HEADER_SIZE + payloadLength);
  const view = new DataView(buf);
  view.setUint32(0, generation, true);
  view.setUint8(4, BINARY_KIND_TRACKING);
  view.setUint8(5, 0);
  view.setUint16(6, 0, true);
  view.setBigUint64(8, sequence, true);
  view.setBigUint64(16, 1000n, true);
  view.setBigUint64(24, 2000n, true);
  return buf;
}

/** Same as runningSnapshotBody, but also carries a samples.tracking fact
 * (sequence/age_ms) -- needed to exercise the backend's own reported
 * producer-side age, which runningSnapshotBody alone leaves null. */
function runningSnapshotWithTrackingAge(generation: number, revision: number, ageMs: number): string {
  return validSnapshotBody({
    revision,
    capture: {
      connecting: false,
      started: { session_id: "s1", running: false, mode: 0, streams: [trackingStreamJson()] },
      status: { session_id: "s1", type: 2, running: true, startup_stage: null, message: null, status: null },
      stopped: null,
      error_message: null,
      binary_generation: generation,
    },
    samples: { tracking: { sequence: "1", age_ms: ageMs, diagnostic: null }, cameras: [] },
  });
}

describe("useCaptureSession (effect integration)", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    FakeWebSocket.instances = [];
    vi.stubGlobal("WebSocket", FakeWebSocket);
    window.localStorage.clear();
    sessionStorage.clear();
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
    vi.useRealTimers();
    window.localStorage.clear();
  });

  it("operatorToken persists across remount (localStorage), New operator clears it, and every outgoing command carries the current name", async () => {
    const first = renderHook(() => useCaptureSession());
    expect(first.result.current.operatorToken).toBeNull();

    act(() => first.result.current.setOperatorToken("Alice"));
    expect(first.result.current.operatorToken).toBe("Alice");
    first.unmount();

    // A fresh mount (e.g. a reload) reads the same persisted name back.
    const second = renderHook(() => useCaptureSession());
    expect(second.result.current.operatorToken).toBe("Alice");

    const ws = FakeWebSocket.instances[1]; // unmounting `first` closed instances[0]; the remount opens a new one
    act(() => ws.open());
    act(() => ws.emitText(runningSnapshotBody(1)));
    act(() => second.result.current.setControlToken("test-control-token"));
    act(() => {
      // No reply is ever sent in this test -- only the outgoing envelope
      // matters here, so the eventual "connection closed" rejection on
      // cleanup is expected and intentionally swallowed.
      second.result.current.sendRecordingCommand("set_task", { batch_id: "b1", task: "x" }).catch(() => {});
    });
    expect(JSON.parse(ws.sent[0]).operator_token).toBe("Alice");

    act(() => second.result.current.clearOperatorToken());
    expect(second.result.current.operatorToken).toBeNull();
    expect(window.localStorage.getItem("egocap_operator_token")).toBeNull();
  });

  it.each([["   "], ["x".repeat(129)]])("a stored operator value that is now invalid (%j) is not trusted on mount", (stored) => {
    window.localStorage.setItem("egocap_operator_token", stored);
    const { result } = renderHook(() => useCaptureSession());
    expect(result.current.operatorToken).toBeNull();
  });

  it("controlToken survives a remount via sessionStorage, clearControlToken removes it, and every outgoing command carries it", async () => {
    const first = renderHook(() => useCaptureSession());
    expect(first.result.current.controlToken).toBeNull();

    act(() => first.result.current.setControlToken("  real-token  "));
    expect(first.result.current.controlToken).toBe("real-token"); // trimmed
    first.unmount();

    // A fresh mount (e.g. a reload) DOES see the previous token -- it's
    // kept in sessionStorage so a reload doesn't force re-entry; only
    // closing the tab/window (which clears sessionStorage itself) does.
    const second = renderHook(() => useCaptureSession());
    expect(second.result.current.controlToken).toBe("real-token");

    act(() => second.result.current.setControlToken("another-token"));
    const ws = FakeWebSocket.instances[1]; // unmounting `first` closed instances[0]; the remount opens a new one
    act(() => ws.open());
    act(() => ws.emitText(runningSnapshotBody(1)));
    act(() => {
      second.result.current.sendRecordingCommand("set_task", { batch_id: "b1", task: "x" }).catch(() => {});
    });
    expect(JSON.parse(ws.sent[0]).control_token).toBe("another-token");

    act(() => second.result.current.clearControlToken());
    expect(second.result.current.controlToken).toBeNull();
    expect(sessionStorage.getItem("egocap.controlToken")).toBeNull();

    // clearControlToken's sessionStorage removal is itself visible to a new mount too.
    const third = renderHook(() => useCaptureSession());
    expect(third.result.current.controlToken).toBeNull();
  });

  it("drops the stored token once a snapshot reports a different backend_instance_id (a restarted backend)", () => {
    const { result } = renderHook(() => useCaptureSession());
    const ws = FakeWebSocket.instances[0];
    act(() => ws.open());
    // First snapshot (backend_instance_id "b1", validSnapshotBody's default) binds the freshly-entered token to it.
    act(() => ws.emitText(runningSnapshotBody(1)));
    act(() => result.current.setControlToken("real-token"));
    expect(result.current.controlToken).toBe("real-token");

    // Same backend, a later revision -- the token survives.
    act(() => ws.emitText(runningSnapshotBody(1, 3)));
    expect(result.current.controlToken).toBe("real-token");

    // A different backend_instance_id (the backend restarted) -- the stale token is dropped automatically.
    act(() => ws.emitText(validSnapshotBody({ backend_instance_id: "b2", revision: 1 })));
    expect(result.current.controlToken).toBeNull();
    expect(sessionStorage.getItem("egocap.controlToken")).toBeNull();
    expect(sessionStorage.getItem("egocap.controlTokenBackendInstanceId")).toBeNull();
  });

  it("a token restored from a reload for a stale backend is dropped by the first snapshot from a different backend", () => {
    // Simulates a reload after the backend already restarted once: both the
    // token and its old binding survive in sessionStorage from before the
    // restart, so this mount never sees the matching "b1" snapshot itself.
    sessionStorage.setItem("egocap.controlToken", "real-token");
    sessionStorage.setItem("egocap.controlTokenBackendInstanceId", "b1");
    const { result } = renderHook(() => useCaptureSession());
    expect(result.current.controlToken).toBe("real-token");
    const ws = FakeWebSocket.instances[0];
    act(() => ws.open());
    act(() => ws.emitText(validSnapshotBody({ backend_instance_id: "b2", revision: 1 })));
    expect(result.current.controlToken).toBeNull();
    expect(sessionStorage.getItem("egocap.controlToken")).toBeNull();
    expect(sessionStorage.getItem("egocap.controlTokenBackendInstanceId")).toBeNull();
  });

  it("a token entered while reconnecting (stale snapshot still showing the old backend) is not bound to that stale id", async () => {
    // Regression test: state.snapshot is not cleared on reconnect (see the
    // reducer), so while "reconnecting" it can still report the OLD
    // backend's id. Binding against it here would pair a freshly-entered
    // token with the wrong id, and the real new backend's first snapshot
    // would then immediately (and wrongly) clear it out again.
    const { result } = renderHook(() => useCaptureSession());
    const ws1 = FakeWebSocket.instances[0];
    act(() => ws1.open());
    act(() => ws1.emitText(runningSnapshotBody(1))); // backend_instance_id "b1"
    act(() => result.current.setControlToken("token-a"));
    expect(result.current.controlToken).toBe("token-a");

    act(() => ws1.close());
    expect(result.current.phase).toBe("reconnecting");
    // state.snapshot still reports "b1" here -- entering a new token must not bind to it.
    act(() => result.current.clearControlToken());
    act(() => result.current.setControlToken("token-b"));
    expect(result.current.controlToken).toBe("token-b");

    // The real new backend finally reconnects (a fresh socket, after the backoff delay) with a different id.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1_000); // first backoff delay (~500ms + jitter)
    });
    const ws2 = FakeWebSocket.instances[1];
    act(() => ws2.open());
    act(() => ws2.emitText(validSnapshotBody({ backend_instance_id: "b2", revision: 1 })));
    expect(result.current.phase).toBe("connected");
    // Must have bound fresh to "b2" just now, not wrongly to the stale "b1" seen while reconnecting.
    expect(result.current.controlToken).toBe("token-b");
  });

  it("setControlToken with only whitespace clears it rather than storing an empty string", () => {
    const { result } = renderHook(() => useCaptureSession());
    act(() => result.current.setControlToken("Alice"));
    expect(result.current.controlToken).toBe("Alice");
    act(() => result.current.setControlToken("   "));
    expect(result.current.controlToken).toBeNull();
  });

  it("sendRecordingCommand rejects without sending anything when no control token has been entered", async () => {
    const { result } = renderHook(() => useCaptureSession());
    const ws1 = FakeWebSocket.instances[0];
    act(() => ws1.open());
    act(() => ws1.emitText(runningSnapshotBody(1)));
    expect(result.current.controlToken).toBeNull();

    let rejection: unknown;
    await act(async () => {
      await result.current.sendRecordingCommand("start_episode", { batch_id: "b1" }).catch((err: unknown) => {
        rejection = err;
      });
    });
    expect(rejection).toBeInstanceOf(Error);
    expect(ws1.sent).toHaveLength(0);
  });

  it("a fatal protocol error clears the staleTimer and never reconnects (the B regression)", async () => {
    const { result } = renderHook(() => useCaptureSession());

    const ws1 = FakeWebSocket.instances[0];
    act(() => ws1.open());
    act(() => ws1.emitText(validSnapshotBody())); // a valid snapshot arms staleTimer

    expect(result.current.phase).toBe("connected");

    // Now a fatal, well-formed-but-unsupported message arrives (a
    // syntactically-malformed string is a different, non-fatal path --
    // see the dedicated malformed-JSON test below).
    act(() => ws1.emitText(validSnapshotBody({ version: 2 })));

    expect(result.current.phase).toBe("unavailable");
    expect(ws1.closed).toBe(true);

    const socketCountAfterFatal = FakeWebSocket.instances.length;

    // Advance well past the stale deadline (3s) and every reconnect delay
    // (up to 8s + jitter) combined -- if the old staleTimer was never
    // cleared, its callback fires in here and calls scheduleReconnect(),
    // opening a new socket despite the fatal/unavailable phase.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(20_000);
    });

    expect(result.current.phase).toBe("unavailable");
    expect(FakeWebSocket.instances.length).toBe(socketCountAfterFatal);
  });

  it("normal close schedules exactly one replacement socket, and the old socket's late messages are ignored", async () => {
    const { result } = renderHook(() => useCaptureSession());
    const ws1 = FakeWebSocket.instances[0];
    act(() => ws1.open());
    act(() => ws1.emitText(validSnapshotBody()));
    expect(result.current.phase).toBe("connected");

    act(() => ws1.close());
    expect(result.current.phase).toBe("reconnecting");

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1_000); // first backoff delay (~500ms + jitter)
    });
    expect(FakeWebSocket.instances.length).toBe(2);
    const ws2 = FakeWebSocket.instances[1];
    expect(ws2.closed).toBe(false);

    // The old (already-closed) socket somehow fires a late message --
    // must not resurrect "connected" from stale data.
    act(() => ws1.emitText(validSnapshotBody({ revision: 99 })));
    expect(result.current.phase).not.toBe("connected");

    act(() => ws2.open());
    act(() => ws2.emitText(validSnapshotBody({ revision: 5 })));
    expect(result.current.phase).toBe("connected");
    expect(result.current.snapshot?.revision).toBe(5);
  });

  it("explicit retry and unmount both clear owned timers so no stray reconnect/bootstrap fires afterward", async () => {
    const { result, unmount } = renderHook(() => useCaptureSession());
    const ws1 = FakeWebSocket.instances[0];
    act(() => ws1.open());
    act(() => ws1.emitText(validSnapshotBody({ version: 2 }))); // fatal -> unavailable
    expect(result.current.phase).toBe("unavailable");

    act(() => result.current.retry());
    // retry() tears down the effect and restarts it synchronously via a
    // state update; a fresh socket should now exist.
    expect(FakeWebSocket.instances.length).toBe(2);
    expect(result.current.phase).toBe("connecting");

    const ws2 = FakeWebSocket.instances[1];
    unmount();
    expect(ws2.closed).toBe(true);

    const countAtUnmount = FakeWebSocket.instances.length;
    await act(async () => {
      await vi.advanceTimersByTimeAsync(20_000);
    });
    // No bootstrap timeout / stale / reconnect callback should fire and
    // open another socket after unmount.
    expect(FakeWebSocket.instances.length).toBe(countAtUnmount);
  });

  it("wrong generation, wrong sequence order, and a terminal snapshot cannot repopulate current tracking", async () => {
    const { result } = renderHook(() => useCaptureSession());
    const ws1 = FakeWebSocket.instances[0];
    act(() => ws1.open());
    act(() => ws1.emitText(runningSnapshotBody(1)));

    act(() => ws1.emitBinary(trackingBinaryFrame(1, 10n)));
    expect(result.current.trackingSample?.decoded.head.position).toEqual([1, 2, 3]);

    // Wrong generation: silently discarded, does not throw/teardown, does
    // not change the current display.
    act(() => ws1.emitBinary(trackingBinaryFrame(99, 11n)));
    expect(result.current.trackingSample?.decoded.head.position).toEqual([1, 2, 3]);
    expect(result.current.phase).toBe("connected");

    // Obsolete (non-increasing) sequence on the correct generation: also a
    // routine discard.
    const framedButStale = trackingBinaryFrame(1, 5n);
    act(() => ws1.emitBinary(framedButStale));
    expect(result.current.trackingSample?.decoded.head.position).toEqual([1, 2, 3]);

    // Capture goes terminal -- a late/obsolete binary sample for the old
    // generation must not repopulate tracking.
    act(() =>
      ws1.emitText(
        validSnapshotBody({
          revision: 3,
          capture: {
            connecting: false,
            started: { session_id: "s1", running: false, mode: 0, streams: [trackingStreamJson()] },
            status: null,
            stopped: { session_id: "s1", running: false, reason: 1, error_message: null, status: null },
            error_message: null,
            binary_generation: null,
          },
        }),
      ),
    );
    expect(result.current.trackingSample).toBeNull();
    act(() => ws1.emitBinary(trackingBinaryFrame(1, 12n)));
    expect(result.current.trackingSample).toBeNull();
  });

  it("syntactically malformed JSON (unparseable, not just unsupported) is a retryable TRANSPORT_FAILURE, not fatal", async () => {
    const { result } = renderHook(() => useCaptureSession());
    const ws1 = FakeWebSocket.instances[0];
    act(() => ws1.open());
    act(() => ws1.emitText(validSnapshotBody()));
    expect(result.current.phase).toBe("connected");

    act(() => ws1.emitText("{not json"));
    expect(result.current.phase).toBe("reconnecting");
    expect(ws1.closed).toBe(true);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1_000);
    });
    expect(FakeWebSocket.instances.length).toBe(2); // did reconnect, unlike the fatal case
  });

  it("a malformed binary header (genuine framing violation) is fatal, distinct from a routine wrong-generation discard", async () => {
    const { result } = renderHook(() => useCaptureSession());
    const ws1 = FakeWebSocket.instances[0];
    act(() => ws1.open());
    act(() => ws1.emitText(runningSnapshotBody(1)));
    expect(result.current.phase).toBe("connected");

    // Too short to even contain a header -- a framing violation, not a
    // native-invalid-pose sample.
    act(() => ws1.emitBinary(new ArrayBuffer(4)));
    expect(result.current.phase).toBe("unavailable");
    expect(ws1.closed).toBe(true);
  });

  it("binary before a RECONNECTED socket's own first snapshot is fatal, not silently dropped (per-socket bootstrap)", async () => {
    const { result } = renderHook(() => useCaptureSession());
    const ws1 = FakeWebSocket.instances[0];
    act(() => ws1.open());
    act(() => ws1.emitText(validSnapshotBody())); // first socket bootstraps
    expect(result.current.phase).toBe("connected");

    act(() => ws1.close()); // ordinary close -> reconnect scheduled
    expect(result.current.phase).toBe("reconnecting");
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1_000);
    });
    expect(FakeWebSocket.instances.length).toBe(2);
    const ws2 = FakeWebSocket.instances[1];
    act(() => ws2.open());

    // The replacement socket never sent its own snapshot yet -- binary
    // here must be an ordering violation on THIS socket, not silently
    // dropped because some prior socket already bootstrapped once.
    act(() => ws2.emitBinary(trackingBinaryFrame(1, 1n)));
    expect(result.current.phase).toBe("unavailable");
    expect(result.current.diagnostic).toMatch(/before any snapshot/);
    expect(ws2.closed).toBe(true);

    const socketCountAfterFatal = FakeWebSocket.instances.length;
    await act(async () => {
      await vi.advanceTimersByTimeAsync(20_000);
    });
    expect(FakeWebSocket.instances.length).toBe(socketCountAfterFatal); // no automatic retry
  });

  it("a wrong-length payload for the current active association is fatal, both short and long", async () => {
    const payloadSize = 13; // trackingStreamJson()'s layout.payload_size

    for (const badLength of [payloadSize - 1, payloadSize + 1]) {
      FakeWebSocket.instances = [];
      const { result, unmount } = renderHook(() => useCaptureSession());
      const ws1 = FakeWebSocket.instances[0];
      act(() => ws1.open());
      act(() => ws1.emitText(runningSnapshotBody(1)));
      act(() => ws1.emitBinary(trackingBinaryFrame(1, 10n)));
      expect(result.current.trackingSample?.decoded.head.position).toEqual([1, 2, 3]);

      act(() => ws1.emitBinary(trackingBinaryFrameWithPayloadLength(1, 11n, badLength)));
      expect(result.current.phase).toBe("unavailable");
      expect(result.current.diagnostic).toMatch(/payload size mismatch/);
      expect(result.current.trackingSample).toBeNull(); // not left showing the old sample
      expect(ws1.closed).toBe(true);

      unmount();
    }
  });

  it("tracking freshness anchors to the installed sample's own receipt, not the latest heartbeat", async () => {
    const { result } = renderHook(() => useCaptureSession());
    const ws1 = FakeWebSocket.instances[0];
    act(() => ws1.open());
    act(() => ws1.emitText(runningSnapshotWithTrackingAge(1, 2, 5)));
    act(() => ws1.emitBinary(trackingBinaryFrame(1, 10n)));
    expect(result.current.trackingSample?.decoded.head.position).toEqual([1, 2, 3]);

    const firstReceiptTime = result.current.trackingSample?.snapshotReceivedAtMonotonic;
    const firstAgeMs = result.current.trackingSample?.snapshotAgeMs;
    expect(typeof firstReceiptTime).toBe("number");
    expect(firstAgeMs).toBe(5);

    // Time passes with the SOCKET itself staying healthy -- regular
    // heartbeats keep arriving (each resetting the connection's own
    // staleTimer, well under STALE_AFTER_MS apart) and each still
    // advertises a low producer-side age_ms, as if fresh samples kept
    // being produced -- but no matching binary is ever actually delivered.
    for (let revision = 3; revision <= 7; revision++) {
      await act(async () => {
        await vi.advanceTimersByTimeAsync(1_000);
      });
      act(() => ws1.emitText(runningSnapshotWithTrackingAge(1, revision, 5)));
    }

    // The displayed sample's own receipt/age baseline must NOT have been
    // refreshed by that heartbeat -- it is still anchored to sample1's
    // original install moment, now five seconds stale, not "just arrived".
    expect(result.current.trackingSample?.snapshotReceivedAtMonotonic).toBe(firstReceiptTime);
    expect(result.current.trackingSample?.snapshotAgeMs).toBe(firstAgeMs);
    expect(result.current.trackingSample?.decoded.head.position).toEqual([1, 2, 3]); // still sample1

    // A genuinely new accepted binary DOES refresh both.
    act(() => ws1.emitBinary(trackingBinaryFrame(1, 11n)));
    expect(result.current.trackingSample?.snapshotReceivedAtMonotonic).not.toBe(firstReceiptTime);
  });

  it("the gap between a heartbeat's receipt and its matching binary's later arrival is not silently discarded", async () => {
    const { result } = renderHook(() => useCaptureSession());
    const ws1 = FakeWebSocket.instances[0];
    act(() => ws1.open());
    act(() => ws1.emitText(runningSnapshotWithTrackingAge(1, 2, 1000))); // producer already 1000ms stale when reported
    const snapshotReceiptTime = performance.now();

    // The matching binary for this association arrives meaningfully LATER
    // than the heartbeat that reported its producer-side age -- a real
    // gap, not an instantaneous pair.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(500);
    });
    act(() => ws1.emitBinary(trackingBinaryFrame(1, 10n)));

    const sample = result.current.trackingSample;
    expect(sample).not.toBeNull();
    expect(sample?.snapshotAgeMs).toBe(1000);
    // The baseline receipt reference is the SNAPSHOT's own receipt instant
    // -- not "now" at binary-install time, which would make it look like
    // it just arrived and silently discard the 500ms gap just advanced.
    expect(sample?.snapshotReceivedAtMonotonic).toBeLessThanOrEqual(snapshotReceiptTime + 5);
    expect(performance.now() - (sample?.snapshotReceivedAtMonotonic ?? 0)).toBeGreaterThanOrEqual(500);
  });

  it("sendRecordingCommand round-trips a real recording_command/recording_reply pair, applies the embedded snapshot, and snapshot/binary delivery keeps working alongside it", async () => {
    const { result } = renderHook(() => useCaptureSession());
    const ws1 = FakeWebSocket.instances[0];
    act(() => ws1.open());
    act(() => ws1.emitText(runningSnapshotBody(1)));
    act(() => result.current.setControlToken("test-control-token"));
    expect(result.current.phase).toBe("connected");

    let settled: unknown;
    await act(async () => {
      const pending = result.current.sendRecordingCommand("start_episode", { batch_id: "b1" });
      // The frame actually sent over the wire is the real contract the
      // backend's parse_recording_command expects.
      expect(ws1.sent).toHaveLength(1);
      const sentBody = JSON.parse(ws1.sent[0]) as { type: string; request_id: string; action: string; payload: unknown };
      expect(sentBody.type).toBe("recording_command");
      expect(sentBody.action).toBe("start_episode");
      expect(sentBody.payload).toEqual({ batch_id: "b1" });

      ws1.emitText(
        JSON.stringify({
          type: "recording_reply",
          request_id: sentBody.request_id,
          outcome: "ok",
          reason: null,
          snapshot: {
            batch_id: "b1",
            episode_phase: "RECORDING",
            current_episode_id: "ep-1",
            current_episode_elapsed_ms: 1234,
            task_text: "Pour water.",
            default_operator: null,
            auto_fill_operator_name: false,
            saved_count: 0,
            last_result: null,
            ready_to_record: false,
            revision: 0,
          },
        }),
      );
      settled = await pending;
    });
    expect(settled).toEqual({ outcome: "ok", reason: null });
    expect(result.current.recordingSnapshot).toEqual({
      batchId: "b1",
      episodePhase: "RECORDING",
      currentEpisodeId: "ep-1",
      currentEpisodeElapsedMs: 1234,
      taskText: "Pour water.",
      defaultOperator: null,
      autoFillOperatorName: false,
      savedCount: 0,
      lastResult: null,
      readyToRecord: false,
      revision: 0,
    });
    expect(result.current.recordingSnapshotReceivedAtMonotonic).not.toBeNull();

    // The reply round-trip must not have disturbed ordinary snapshot/binary
    // tracking delivery on the same socket.
    act(() => ws1.emitBinary(trackingBinaryFrame(1, 10n)));
    expect(result.current.trackingSample?.decoded.head.position).toEqual([1, 2, 3]);
    expect(result.current.phase).toBe("connected");
  });

  it("a Stop is given far longer than 5 s to settle before its outcome is called unknown", async () => {
    const { result } = renderHook(() => useCaptureSession());
    const ws1 = FakeWebSocket.instances[0];
    act(() => ws1.open());
    act(() => ws1.emitText(runningSnapshotBody(1)));
    act(() => result.current.setControlToken("test-control-token"));

    let rejection: unknown;
    await act(async () => {
      result.current.sendRecordingCommand("stop_episode", { batch_id: "b1", episode_id: "e1" }).catch((err: unknown) => {
        rejection = err;
      });
      for (let elapsed = 0; elapsed < 60_000; elapsed += 1_000) {
        await vi.advanceTimersByTimeAsync(1_000);
        ws1.emitText(runningSnapshotBody(1)); // healthy heartbeats, so only the command timeout is under test
      }
    });
    expect(rejection).toBeUndefined();
    expect(ws1.closed).toBe(false);

    await act(async () => {
      for (let elapsed = 0; elapsed < 61_000; elapsed += 1_000) {
        await vi.advanceTimersByTimeAsync(1_000);
        if (!ws1.closed) ws1.emitText(runningSnapshotBody(1));
      }
    });
    expect(rejection).toBeInstanceOf(RecordingCommandUncertainError);
  });

  it("a sent command's timeout actively resolves the resulting uncertainty by reusing the existing reconnect path, picking up a change that actually applied before the reply was lost", async () => {
    const { result } = renderHook(() => useCaptureSession());
    const ws1 = FakeWebSocket.instances[0];
    act(() => ws1.open());
    act(() => ws1.emitText(runningSnapshotBody(1)));
    act(() => result.current.setControlToken("test-control-token"));

    let rejection: unknown;
    await act(async () => {
      const pending = result.current.sendRecordingCommand("set_task", { batch_id: "b1", task: "Pour water." });
      pending.catch((err: unknown) => {
        rejection = err;
      });
      // Never reply to the recording command -- simulates the backend
      // actually applying it but its own reply getting lost. Keep
      // re-arming the connection's own staleTimer (STALE_AFTER_MS=3000)
      // with ordinary heartbeats along the way, same as a real healthy
      // connection would receive, so only the recording-command timeout
      // (5000ms) is actually under test here.
      await vi.advanceTimersByTimeAsync(2000);
      ws1.emitText(runningSnapshotBody(1));
      await vi.advanceTimersByTimeAsync(2000);
      ws1.emitText(runningSnapshotBody(1));
      await vi.advanceTimersByTimeAsync(1000);
    });
    expect(rejection).toBeInstanceOf(RecordingCommandUncertainError);

    // No new protocol action/second send -- the timeout actively resolves
    // the uncertainty by reusing the EXISTING transport teardown/reconnect
    // path, not a retry of the mutation or a new command.
    expect(ws1.sent).toHaveLength(1);
    expect(ws1.closed).toBe(true);

    // The reconnect's own backoff delay elapses, producing a replacement
    // connection -- same machinery as any ordinary dropped connection.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000);
    });
    expect(FakeWebSocket.instances).toHaveLength(2);
    const ws2 = FakeWebSocket.instances[1];

    // The replacement connection's own bootstrap carries the real,
    // already-applied state -- the original set_task DID succeed before
    // its reply was lost, and the fresh connection's initial
    // recording_snapshot (sent unconditionally on every new connection,
    // same as the real backend's sender loop) is what resolves it.
    act(() => ws2.open());
    act(() => ws2.emitText(runningSnapshotBody(1)));
    await act(async () => {
      ws2.emitText(
        JSON.stringify({
          type: "recording_snapshot",
          snapshot: {
            batch_id: "b1",
            episode_phase: "IDLE",
            current_episode_id: null,
            current_episode_elapsed_ms: null,
            task_text: "Pour water.",
            default_operator: null,
            auto_fill_operator_name: false,
            saved_count: 0,
            last_result: null,
            ready_to_record: false,
            revision: 0,
          },
        }),
      );
    });
    expect(result.current.recordingSnapshot?.taskText).toBe("Pour water.");
    expect(result.current.phase).toBe("connected");

    // A LATE reply for the original, already-timed-out request (arriving
    // on the now-dead first socket) must be ignored, not misapplied to
    // the new connection's own state.
    act(() => {
      ws1.emitText(
        JSON.stringify({
          type: "recording_reply",
          request_id: JSON.parse(ws1.sent[0]).request_id,
          outcome: "ok",
          reason: null,
          snapshot: {
            batch_id: "b1",
            episode_phase: "RECORDING",
            current_episode_id: "ep-1",
            current_episode_elapsed_ms: 9999,
            task_text: "a different, stale task",
            default_operator: null,
            auto_fill_operator_name: false,
            saved_count: 0,
            last_result: null,
            ready_to_record: false,
            revision: 0,
          },
        }),
      );
    });
    expect(result.current.recordingSnapshot?.taskText).toBe("Pour water.");
    expect(result.current.phase).toBe("connected");
  });

  it("drops a recording snapshot older than one already applied on the same socket", async () => {
    const { result } = renderHook(() => useCaptureSession());
    const ws1 = FakeWebSocket.instances[0];
    act(() => ws1.open());
    act(() => ws1.emitText(runningSnapshotBody(1)));
    const push = (revision: number, savedCount: number) =>
      act(() => {
        ws1.emitText(
          JSON.stringify({
            type: "recording_snapshot",
            snapshot: {
              batch_id: "b1",
              episode_phase: "IDLE",
              current_episode_id: null,
              current_episode_elapsed_ms: null,
              task_text: "Pour water.",
              default_operator: null,
              auto_fill_operator_name: false,
              saved_count: savedCount,
              last_result: null,
              ready_to_record: true,
              revision,
            },
          }),
        );
      });
    push(5, 2);
    push(4, 1);
    expect(result.current.recordingSnapshot?.savedCount).toBe(2);
    push(5, 3);
    expect(result.current.recordingSnapshot?.savedCount).toBe(3);
  });

  it("sendRecordingCommand rejects a second send while one is already in flight", async () => {
    const { result } = renderHook(() => useCaptureSession());
    const ws1 = FakeWebSocket.instances[0];
    act(() => ws1.open());
    act(() => ws1.emitText(runningSnapshotBody(1)));
    act(() => result.current.setControlToken("test-control-token"));

    let firstPending: Promise<unknown> = Promise.resolve();
    let secondRejection: unknown;
    await act(async () => {
      firstPending = result.current.sendRecordingCommand("start_episode", { batch_id: "b1" });
      await result.current.sendRecordingCommand("stop_episode", { batch_id: "b1", episode_id: "e1" }).catch((err: unknown) => {
        secondRejection = err;
      });
    });
    expect(secondRejection).toBeInstanceOf(Error);

    // Settle the first so the test doesn't leave a dangling unhandled rejection.
    const sentBody = JSON.parse(ws1.sent[0]) as { request_id: string };
    await act(async () => {
      ws1.emitText(
        JSON.stringify({
          type: "recording_reply",
          request_id: sentBody.request_id,
          outcome: "ok",
          reason: null,
          snapshot: {
            batch_id: "b1",
            episode_phase: "RECORDING",
            current_episode_id: "ep-1",
            current_episode_elapsed_ms: 1234,
            task_text: "",
            default_operator: null,
            auto_fill_operator_name: false,
            saved_count: 0,
            last_result: null,
            ready_to_record: false,
            revision: 0,
          },
        }),
      );
      await firstPending;
    });
  });
});
