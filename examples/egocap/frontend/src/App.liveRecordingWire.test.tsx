/**
 * End-to-end evidence that a real command produced by RecordingPanel
 * reaches the wire in the exact shape backend/egocap_web/
 * recording_command.py's parser requires -- not a mocked transport, not a
 * manually-constructed message. Renders the real useCaptureSession hook
 * (no mock) wired through the real buildRecordingActions, drives a real
 * browser-shaped WebSocket stub through connect -> snapshot bootstrap,
 * exercises the real RecordingPanel's inline task editor as an operator
 * would, and inspects the literal frame sent. A matching backend-side test
 * (tests/backend/test_recording_command.py) proves the Python parser
 * accepts this exact shape; this file does not call the Python parser itself.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { buildRecordingActions } from "./recordingActions";
import { RecordingPanel } from "./RecordingPanel";
import { useCaptureSession } from "./useCaptureSession";
import { PREVIEW_VERSION, SNAPSHOT_TYPE } from "./captureProtocol";
import type { RecordingSnapshot } from "./recordingTypes";

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
  closeCode: number | undefined;
  closeReason: string | undefined;
  onopen: (() => void) | null = null;
  onmessage: ((ev: { data: unknown }) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  sent: string[] = [];

  constructor(url: string) {
    this.url = url;
    FakeWebSocket.instances.push(this);
  }

  close(code?: number, reason?: string): void {
    if (this.closed) return;
    this.closed = true;
    this.closeCode = code;
    this.closeReason = reason;
    this.readyState = FakeWebSocket.CLOSED;
    this.onclose?.();
  }

  send(data: string): void {
    this.sent.push(data);
  }

  open(): void {
    this.readyState = FakeWebSocket.OPEN;
    this.onopen?.();
  }

  emitText(data: string): void {
    this.onmessage?.({ data });
  }
}

function validSnapshotBody(): string {
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
  });
}

const activeSnapshot: RecordingSnapshot = {
  batchId: "b1",
  episodePhase: "IDLE",
  currentEpisodeId: null,
  currentEpisodeElapsedMs: null,
  taskText: "Pour water into the cup.",
  defaultOperator: null,
  autoFillOperatorName: false,
  savedCount: 0,
  lastResult: null,
  readyToRecord: true,
  revision: 0,
};

/** Wires the real capture transport to the real RecordingPanel through the
 * real buildRecordingActions, exactly as App.tsx does -- a supplied fixture
 * snapshot stands in for the backend's recording observation. */
function Harness() {
  const { sendRecordingCommand, operatorToken, setOperatorToken, clearOperatorToken, controlToken, setControlToken, clearControlToken } =
    useCaptureSession();
  const actions = buildRecordingActions(sendRecordingCommand);
  return (
    <RecordingPanel
      recordingSnapshot={activeSnapshot}
      actions={actions}
      operatorToken={operatorToken}
      setOperatorToken={setOperatorToken}
      clearOperatorToken={clearOperatorToken}
      controlToken={controlToken}
      setControlToken={setControlToken}
      clearControlToken={clearControlToken}
    />
  );
}

describe("RecordingPanel -> real transport -> real backend wire shape", () => {
  beforeEach(() => {
    FakeWebSocket.instances = [];
    vi.stubGlobal("WebSocket", FakeWebSocket);
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("a real submitted task-edit sends the flat snake_case payload the backend parser requires, and the client does not close its socket", async () => {
    render(<Harness />);
    const ws = FakeWebSocket.instances[0];
    act(() => ws.open());
    act(() => ws.emitText(validSnapshotBody()));

    // Exercise the real RecordingPanel exactly as an operator would: enter
    // an operator name and the control token first (recording controls are
    // gated until both are set), then t to edit, type new text, Enter to submit.
    fireEvent.change(screen.getByLabelText(/operator/i), { target: { value: "Alice" } });
    fireEvent.click(screen.getByRole("button", { name: /set operator/i }));
    fireEvent.change(screen.getByLabelText(/control token/i), { target: { value: "real-control-token" } });
    fireEvent.click(screen.getByRole("button", { name: /set control token/i }));
    fireEvent.click(screen.getByRole("button", { name: /edit task \(t\)/i }));
    const input = screen.getByDisplayValue(activeSnapshot.taskText!) as HTMLInputElement;
    fireEvent.change(input, { target: { value: "Fold the towel in half." } });
    await act(async () => {
      fireEvent.keyDown(input, { key: "Enter" });
    });

    const recordingFrames = ws.sent.map((raw) => JSON.parse(raw) as Record<string, unknown>).filter((m) => m.type === "recording_command");
    expect(recordingFrames).toHaveLength(1);
    const frame = recordingFrames[0];
    expect(frame.action).toBe("set_task");
    expect(frame.payload).toEqual({ batch_id: "b1", task: "Fold the towel in half." });
    expect(frame.operator_token).toBe("Alice");
    expect(frame.control_token).toBe("real-control-token");

    // Reply as the real backend handler would: a correlated real rejection
    // (no batch "b1" actually exists in this test's transport-only backend
    // double).
    await act(async () => {
      ws.emitText(
        JSON.stringify({
          type: "recording_reply",
          request_id: frame.request_id,
          outcome: "rejected",
          reason: "no matching batch",
          snapshot: {
            batch_id: "a-different-real-batch",
            episode_phase: "IDLE",
            current_episode_id: null,
            current_episode_elapsed_ms: null,
            task_text: null,
            default_operator: null,
            auto_fill_operator_name: false,
            saved_count: 0,
            last_result: null,
            ready_to_record: true,
            revision: 0,
          },
        }),
      );
    });

    // The backend parser accepts this exact payload (see
    // tests/backend/test_recording_command.py's matching assertion), so the
    // client has no reason to close. A fake socket cannot observe a server
    // close code; this only shows the client did not close its own socket.
    expect(ws.closed).toBe(false);
    expect(await screen.findByText(/no matching batch/i)).not.toBeNull();
  });
});
