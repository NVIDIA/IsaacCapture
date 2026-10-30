import { describe, expect, it } from "vitest";
import { INITIAL_STATE, reducer, type Action, type ConnectionState } from "./useCaptureSession";
import { PREVIEW_VERSION, SNAPSHOT_TYPE, type PreviewSnapshot } from "./captureProtocol";

function snapshot(overrides: Partial<PreviewSnapshot> = {}): PreviewSnapshot {
  return {
    version: PREVIEW_VERSION,
    type: SNAPSHOT_TYPE,
    backend_instance_id: "b1",
    revision: 1,
    station_label: "Station",
    profile_label: "Profile",
    connecting: false,
    started: null,
    status: null,
    stopped: null,
    error_message: null,
    binary_generation: null,
    tracking: null,
    cameras: [],
    visualization: null,
    ...overrides,
  };
}

describe("useCaptureSession reducer", () => {
  it("a fatal INCOMPATIBLE_PROTOCOL moves to unavailable from any phase", () => {
    for (const phase of ["connecting", "synchronizing", "connected", "reconnecting"] as const) {
      const state: ConnectionState = { phase, snapshot: null, diagnostic: null };
      const next = reducer(state, { type: "INCOMPATIBLE_PROTOCOL", diagnostic: "bad version" });
      expect(next.phase).toBe("unavailable");
      expect(next.diagnostic).toBe("bad version");
    }
  });

  it("TRANSPORT_FAILURE never downgrades an already-unavailable (fatal) phase -- the manual-retry-only rule", () => {
    const unavailable: ConnectionState = { phase: "unavailable", snapshot: null, diagnostic: "bad version" };
    const next = reducer(unavailable, { type: "TRANSPORT_FAILURE", diagnostic: "connection closed" });
    expect(next.phase).toBe("unavailable");
    expect(next.diagnostic).toBe("bad version");
  });

  it("STALE is a no-op once already unavailable, and does not revert to reconnecting", () => {
    const unavailable: ConnectionState = { phase: "unavailable", snapshot: snapshot(), diagnostic: "bad version" };
    const next = reducer(unavailable, { type: "STALE" });
    expect(next).toEqual(unavailable);
  });

  it("SOCKET_OPEN moves out of unavailable into synchronizing -- explicit Retry must actually recover", () => {
    const unavailable: ConnectionState = { phase: "unavailable", snapshot: null, diagnostic: "bad version" };
    const next = reducer(unavailable, { type: "SOCKET_OPEN" });
    expect(next.phase).toBe("synchronizing");
  });

  it("EXPLICIT_RETRY resets phase/snapshot/diagnostic regardless of prior phase", () => {
    const unavailable: ConnectionState = { phase: "unavailable", snapshot: snapshot(), diagnostic: "bad version" };
    const next = reducer(unavailable, { type: "EXPLICIT_RETRY" });
    expect(next).toEqual({ phase: "connecting", snapshot: null, diagnostic: null });
  });

  it("a valid SNAPSHOT always installs connected regardless of starting phase", () => {
    for (const phase of ["connecting", "synchronizing", "reconnecting", "unavailable"] as const) {
      const state: ConnectionState = { phase, snapshot: null, diagnostic: "stale" };
      const snap = snapshot({ revision: 7 });
      const next = reducer(state, { type: "SNAPSHOT", snapshot: snap });
      expect(next).toEqual({ phase: "connected", snapshot: snap, diagnostic: null });
    }
  });

  it("STALE/BOOTSTRAP_TIMEOUT from connected move to reconnecting, retaining the last snapshot", () => {
    const snap = snapshot();
    const connected: ConnectionState = { phase: "connected", snapshot: snap, diagnostic: null };
    const stale = reducer(connected, { type: "STALE" });
    expect(stale.phase).toBe("reconnecting");
    expect(stale.snapshot).toBe(snap);
  });

  it("the initial state is connecting with no snapshot", () => {
    expect(INITIAL_STATE).toEqual({ phase: "connecting", snapshot: null, diagnostic: null });
  });

  it("reducer is pure: identical input state/action produce a deep-equal result with no shared mutation", () => {
    const state: ConnectionState = { phase: "connecting", snapshot: null, diagnostic: null };
    const action: Action = { type: "SOCKET_OPEN" };
    const a = reducer(state, action);
    const b = reducer(state, action);
    expect(a).toEqual(b);
    expect(state).toEqual({ phase: "connecting", snapshot: null, diagnostic: null });
  });
});
