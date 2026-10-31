/**
 * Component-level tests for RecordingPanel: the honest no-backend default
 * (never a fabricated success state), buttons/shortcuts sharing the same
 * eligibility guard, the inline task editor seeding from the supplied
 * current task text (not local form state, since there is none), Save
 * showing its real outcome instead of silently doing nothing, Discard
 * acting immediately with no confirmation dialog, and New batch's
 * stricter idle-only guard plus preserved task text on success.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { RecordingPanel } from "./RecordingPanel";
import { RecordingCommandUncertainError } from "./useCaptureSession";
import type { RecordingSnapshot } from "./recordingTypes";

afterEach(() => {
  cleanup();
});

// Required props every test needs but most don't care about -- spread
// first so a test exercising the operator or control-token gate itself can
// still override any of these with an explicit prop after the spread.
const defaultOperatorProps = {
  operatorToken: "<test-operator>",
  setOperatorToken: vi.fn(),
  clearOperatorToken: vi.fn(),
  controlToken: "<test-token>",
  setControlToken: vi.fn(),
  clearControlToken: vi.fn(),
};

function activeSnapshot(overrides: Partial<RecordingSnapshot> = {}): RecordingSnapshot {
  return {
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
    ...overrides,
  };
}

function statusText(): string {
  return document.querySelector(".recording-status")?.textContent ?? "";
}

describe("RecordingPanel", () => {
  it("with no recording backend connected, every control is disabled -- with no hardcoded banner describing WHY (that's never this component's call)", () => {
    render(<RecordingPanel {...defaultOperatorProps} recordingSnapshot={null} />);

    // No special "not connected" banner -- a backend-implementation detail
    // the frontend must never know or render; see recordingEligibility's
    // own per-control reasons instead (checked below via title).
    expect(screen.queryByText(/not connected/i)).toBeNull();
    expect(screen.queryByText(/no data is recorded/i)).toBeNull();
    for (const name of [/record \(s\)/i, /stop\/save \(w\)/i, /discard \(a\)/i, /edit task \(t\)/i, /new batch \(r\)/i]) {
      expect((screen.getByRole("button", { name }) as HTMLButtonElement).disabled).toBe(true);
    }
  });

  it("with every action wired but no backend snapshot, Edit task is disabled too and the status line invents no values", () => {
    const noop = vi.fn().mockResolvedValue({ outcome: "ok", reason: null });
    render(
      <RecordingPanel {...defaultOperatorProps}
        recordingSnapshot={null}
        actions={{ onSetTask: noop, onStartEpisode: noop, onStopEpisode: noop, onDiscardEpisode: noop, onNewBatch: noop }}
      />,
    );
    for (const name of [/record \(s\)/i, /stop\/save \(w\)/i, /discard \(a\)/i, /edit task \(t\)/i, /new batch \(r\)/i]) {
      expect((screen.getByRole("button", { name }) as HTMLButtonElement).disabled).toBe(true);
    }
    const status = document.querySelector(".recording-status")?.textContent ?? "";
    expect(status).toContain("Episode: —");
    expect(status).toContain("Task: —");
    expect(status).toContain("Saved: —");
    expect(status).not.toMatch(/unset|Saved: 0/);
  });

  it("while an episode is recording, Record explains that one is already active, not that sources are not ready", () => {
    const noop = vi.fn().mockResolvedValue({ outcome: "ok", reason: null });
    render(
      <RecordingPanel {...defaultOperatorProps}
        recordingSnapshot={activeSnapshot({ episodePhase: "RECORDING", currentEpisodeId: "e1", readyToRecord: false })}
        actions={{ onStartEpisode: noop }}
      />,
    );
    const record = screen.getByRole("button", { name: /record \(s\)/i }) as HTMLButtonElement;
    expect(record.disabled).toBe(true);
    expect(record.title).toBe("An episode is already active");
  });

  it("with no task set, Record says to set a task first", () => {
    const noop = vi.fn().mockResolvedValue({ outcome: "ok", reason: null });
    render(
      <RecordingPanel {...defaultOperatorProps}
        recordingSnapshot={activeSnapshot({ taskText: "", readyToRecord: false })}
        actions={{ onStartEpisode: noop }}
      />,
    );
    const record = screen.getByRole("button", { name: /record \(s\)/i }) as HTMLButtonElement;
    expect(record.disabled).toBe(true);
    expect(record.title).toBe("Set a task first");
  });

  it("with a whitespace-only task, Record still says to set a task first", () => {
    const noop = vi.fn().mockResolvedValue({ outcome: "ok", reason: null });
    render(
      <RecordingPanel {...defaultOperatorProps}
        recordingSnapshot={activeSnapshot({ taskText: "   ", readyToRecord: false })}
        actions={{ onStartEpisode: noop }}
      />,
    );
    const record = screen.getByRole("button", { name: /record \(s\)/i }) as HTMLButtonElement;
    expect(record.disabled).toBe(true);
    expect(record.title).toBe("Set a task first");
  });

  it("with a task set but sources not ready, Record explains that, not the task gate", () => {
    const noop = vi.fn().mockResolvedValue({ outcome: "ok", reason: null });
    render(
      <RecordingPanel {...defaultOperatorProps}
        recordingSnapshot={activeSnapshot({ readyToRecord: false })}
        actions={{ onStartEpisode: noop }}
      />,
    );
    const record = screen.getByRole("button", { name: /record \(s\)/i }) as HTMLButtonElement;
    expect(record.disabled).toBe(true);
    expect(record.title).toBe("Required sources not ready");
  });

  it("after an interrupted take, Record points to New batch as the way to clear it", () => {
    const noop = vi.fn().mockResolvedValue({ outcome: "ok", reason: null });
    render(
      <RecordingPanel {...defaultOperatorProps}
        recordingSnapshot={activeSnapshot({ episodePhase: "RECOVERY_REQUIRED", readyToRecord: false })}
        actions={{ onStartEpisode: noop, onNewBatch: noop }}
      />,
    );
    expect((screen.getByRole("button", { name: /record \(s\)/i }) as HTMLButtonElement).title).toBe(
      "A take was interrupted -- use New batch to clear it",
    );
    expect((screen.getByRole("button", { name: /new batch \(r\)/i }) as HTMLButtonElement).disabled).toBe(false);
    expect(document.querySelector(".recording-status")?.textContent ?? "").toContain("Recovery required");
  });

  it("there is no End session/Quit control at all", () => {
    render(<RecordingPanel {...defaultOperatorProps} recordingSnapshot={activeSnapshot()} />);
    expect(screen.queryByRole("button", { name: /end session/i })).toBeNull();
    expect(screen.queryByRole("button", { name: /quit/i })).toBeNull();
  });

  it("Record click and its 's' shortcut both go through the same eligibility guard to the same target", async () => {
    const onStartEpisode = vi.fn().mockResolvedValue({ outcome: "ok", reason: null });
    render(<RecordingPanel {...defaultOperatorProps} recordingSnapshot={activeSnapshot()} actions={{ onStartEpisode }} />);

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /record \(s\)/i }));
    });
    expect(onStartEpisode).toHaveBeenCalledWith({ batchId: "b1" });

    await act(async () => {
      fireEvent.keyDown(window, { key: "s" });
    });
    expect(onStartEpisode).toHaveBeenCalledTimes(2);
  });

  it("Discard acts immediately -- no confirmation dialog appears", async () => {
    const onDiscardEpisode = vi.fn().mockResolvedValue({ outcome: "ok", reason: null });
    render(
      <RecordingPanel {...defaultOperatorProps}
        recordingSnapshot={activeSnapshot({ episodePhase: "RECORDING", currentEpisodeId: "e1" })}
        actions={{ onDiscardEpisode }}
      />,
    );

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /discard \(a\)/i }));
    });
    expect(onDiscardEpisode).toHaveBeenCalledWith({ batchId: "b1", episodeId: "e1" });
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("New batch is disabled while recording, with the specific guidance reason, and its 'r' shortcut no-ops too", async () => {
    const onNewBatch = vi.fn().mockResolvedValue({ outcome: "ok", reason: null });
    render(
      <RecordingPanel {...defaultOperatorProps}
        recordingSnapshot={activeSnapshot({ episodePhase: "RECORDING", currentEpisodeId: "e1" })}
        actions={{ onNewBatch }}
      />,
    );

    const newBatchButton = screen.getByRole("button", { name: /new batch \(r\)/i }) as HTMLButtonElement;
    expect(newBatchButton.disabled).toBe(true);
    expect(newBatchButton.title).toMatch(/stop\/save or discard the current take first/i);

    fireEvent.keyDown(window, { key: "r" });
    expect(onNewBatch).not.toHaveBeenCalled();
  });

  it("a successful New batch preserves task text and resets totals via the next observed snapshot", async () => {
    const onNewBatch = vi.fn().mockResolvedValue({ outcome: "ok", reason: null });
    const { rerender } = render(
      <RecordingPanel {...defaultOperatorProps} recordingSnapshot={activeSnapshot({ savedCount: 3 })} actions={{ onNewBatch }} />,
    );

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /new batch \(r\)/i }));
      rerender(<RecordingPanel {...defaultOperatorProps} recordingSnapshot={activeSnapshot({ batchId: "b2" })} actions={{ onNewBatch }} />);
    });

    expect(onNewBatch).toHaveBeenCalledWith({ batchId: "b1" });
    expect(statusText()).toMatch(/Saved: 0/);
    expect(statusText()).toMatch(/Pour water into the cup\./);
  });

  it("the inline task editor seeds from the supplied current task text, and Escape cancels back to it", () => {
    render(<RecordingPanel {...defaultOperatorProps} recordingSnapshot={activeSnapshot({ taskText: "Fold the towel." })} actions={{ onSetTask: vi.fn() }} />);

    fireEvent.click(screen.getByRole("button", { name: /edit task \(t\)/i }));
    const input = screen.getByDisplayValue("Fold the towel.") as HTMLInputElement;

    fireEvent.change(input, { target: { value: "something unsaved" } });
    fireEvent.keyDown(input, { key: "Escape" });

    expect(screen.queryByDisplayValue("something unsaved")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: /edit task \(t\)/i }));
    expect(screen.getByDisplayValue("Fold the towel.")).not.toBeNull();
  });

  it("an unsaved task-edit draft survives a batch change, with a visible context-change indication", async () => {
    const { rerender } = render(<RecordingPanel {...defaultOperatorProps} recordingSnapshot={activeSnapshot()} actions={{ onSetTask: vi.fn() }} />);

    fireEvent.click(screen.getByRole("button", { name: /edit task \(t\)/i }));
    const input = screen.getByDisplayValue("Pour water into the cup.") as HTMLInputElement;
    fireEvent.change(input, { target: { value: "still typing" } });

    rerender(<RecordingPanel {...defaultOperatorProps} recordingSnapshot={activeSnapshot({ batchId: "b2" })} actions={{ onSetTask: vi.fn() }} />);

    expect(screen.getByDisplayValue("still typing")).not.toBeNull(); // not silently discarded/committed
    expect(screen.getByText(/batch changed since you started editing/i)).not.toBeNull();
  });

  it("without a backend snapshot Edit task cannot open, so there is no Save that could do nothing", () => {
    const onSetTask = vi.fn().mockResolvedValue({ outcome: "ok", reason: null });
    render(<RecordingPanel {...defaultOperatorProps} recordingSnapshot={null} actions={{ onSetTask }} />);

    const edit = screen.getByRole("button", { name: /edit task \(t\)/i }) as HTMLButtonElement;
    expect(edit.disabled).toBe(true);
    fireEvent.click(edit);
    expect(screen.queryByRole("button", { name: /^save$/i })).toBeNull();
    expect(onSetTask).not.toHaveBeenCalled();
  });

  it("a successful Stop/Save is not mislabeled as a stale/discarded target", async () => {
    const onStopEpisode = vi.fn().mockResolvedValue({ outcome: "ok", reason: null });
    const { rerender } = render(
      <RecordingPanel {...defaultOperatorProps} recordingSnapshot={activeSnapshot({ episodePhase: "RECORDING", currentEpisodeId: "e1" })} actions={{ onStopEpisode }} />,
    );

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /stop\/save \(w\)/i }));
      // The backend's own reply clears the episode -- this is the command's
      // OWN success changing the snapshot, not a different actor.
      rerender(<RecordingPanel {...defaultOperatorProps} recordingSnapshot={activeSnapshot({ savedCount: 1 })} actions={{ onStopEpisode }} />);
    });

    expect(screen.queryByText(/target changed/i)).toBeNull();
    expect(screen.queryByText(/discarded/i)).toBeNull();
  });

  it("another viewer's command is reflected via an observed snapshot change, without this viewer sending anything", () => {
    const { rerender } = render(<RecordingPanel {...defaultOperatorProps} recordingSnapshot={activeSnapshot()} />);
    expect(statusText()).toMatch(/^Idle/);

    rerender(<RecordingPanel {...defaultOperatorProps} recordingSnapshot={activeSnapshot({ episodePhase: "RECORDING", currentEpisodeId: "e9" })} />);
    expect(statusText()).toMatch(/^Recording/);
  });

  it("the elapsed-time display derives from the backend's own monotonic figure, ticks live, and freezes a final duration once stopped", async () => {
    // A real backend-supplied MONOTONIC timing fact (paired with this
    // browser's own receipt-time performance.now()), not a shared
    // wall-clock basis -- safe across machines/clock drift.
    vi.useFakeTimers();
    try {
      const onStartEpisode = vi.fn().mockResolvedValue({ outcome: "ok", reason: null });
      const onStopEpisode = vi.fn().mockResolvedValue({ outcome: "ok", reason: null });
      const { rerender } = render(<RecordingPanel {...defaultOperatorProps} recordingSnapshot={activeSnapshot()} actions={{ onStartEpisode, onStopEpisode }} />);

      await act(async () => {
        fireEvent.click(screen.getByRole("button", { name: /record \(s\)/i }));
        rerender(
          <RecordingPanel {...defaultOperatorProps}
            recordingSnapshot={activeSnapshot({ episodePhase: "RECORDING", currentEpisodeId: "e1", currentEpisodeElapsedMs: 0 })}
            recordingSnapshotReceivedAtMonotonic={performance.now()}
            actions={{ onStartEpisode, onStopEpisode }}
          />,
        );
      });
      expect(statusText()).toMatch(/Elapsed: 0:00/);

      await act(async () => {
        await vi.advanceTimersByTimeAsync(3000);
      });
      expect(statusText()).toMatch(/Elapsed: 0:03/);

      await act(async () => {
        fireEvent.click(screen.getByRole("button", { name: /stop\/save \(w\)/i }));
        rerender(
          <RecordingPanel {...defaultOperatorProps}
            recordingSnapshot={activeSnapshot({
              savedCount: 1,
              lastResult: { episodeId: "e1", disposition: "KEPT", issues: [], durationMs: 3000 },
            })}
            recordingSnapshotReceivedAtMonotonic={performance.now()}
            actions={{ onStartEpisode, onStopEpisode }}
          />,
        );
      });
      // The finished episode's own duration stays visible/accounted for,
      // instead of the elapsed display simply vanishing the instant recording stops.
      expect(statusText()).toMatch(/Elapsed: 0:03/);

      await act(async () => {
        await vi.advanceTimersByTimeAsync(3000);
      });
      // Frozen, not still ticking, once idle again.
      expect(statusText()).toMatch(/Elapsed: 0:03/);
    } finally {
      vi.useRealTimers();
    }
  });

  it("a fresh viewer (reconnect mid-episode) shows real elapsed time from the backend's monotonic figure, immune to wall-clock mismatch", () => {
    // Regression for: elapsed time must never be computed from when THIS
    // viewer happened to first observe "recording", and must never rely
    // on a shared wall clock between backend and browser -- simulate a
    // browser whose wall clock is wildly different from reality; the
    // backend-supplied monotonic figure must still produce the correct
    // elapsed time regardless.
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2030-01-01T00:00:00Z"));
    try {
      render(
        <RecordingPanel {...defaultOperatorProps}
          recordingSnapshot={activeSnapshot({ episodePhase: "RECORDING", currentEpisodeId: "e1", currentEpisodeElapsedMs: 45_000 })}
          recordingSnapshotReceivedAtMonotonic={performance.now()}
        />,
      );
      expect(statusText()).toMatch(/Elapsed: 0:45/);
    } finally {
      vi.useRealTimers();
    }
  });

  it("a snapshot change observed while editingTask is reconciled once Escape exits it, not stuck stale", () => {
    const { rerender } = render(<RecordingPanel {...defaultOperatorProps} recordingSnapshot={activeSnapshot()} actions={{ onSetTask: vi.fn() }} />);

    fireEvent.click(screen.getByRole("button", { name: /edit task \(t\)/i }));
    const input = screen.getByDisplayValue("Pour water into the cup.") as HTMLInputElement;

    // Another viewer starts recording while this tab is mid-edit -- the
    // snapshot prop change is ignored by the reducer right now (editingTask
    // is not a stable phase), but must still be picked up once this tab
    // leaves editingTask, not stay stuck on stale "Idle" forever.
    rerender(
      <RecordingPanel {...defaultOperatorProps}
        recordingSnapshot={activeSnapshot({ episodePhase: "RECORDING", currentEpisodeId: "e9" })}
        actions={{ onSetTask: vi.fn() }}
      />,
    );

    fireEvent.keyDown(input, { key: "Escape" });
    expect(statusText()).toMatch(/^Recording/);
  });

  it("a rejection reconciles to the latest known fact instead of blindly reapplying a now-stale previousPhase", async () => {
    let resolveStart: (result: { outcome: "ok" | "rejected"; reason: string | null }) => void = () => {};
    const onStartEpisode = vi.fn(
      () =>
        new Promise<{ outcome: "ok" | "rejected"; reason: string | null }>((resolve) => {
          resolveStart = resolve;
        }),
    );
    const { rerender } = render(<RecordingPanel {...defaultOperatorProps} recordingSnapshot={activeSnapshot()} actions={{ onStartEpisode }} />);

    fireEvent.click(screen.getByRole("button", { name: /record \(s\)/i }));
    expect(statusText()).toMatch(/^Starting/);

    // Another viewer wins the race and starts recording first -- this
    // tab's snapshot prop updates while its own start_episode is still
    // pending (a non-stable phase), so the reducer ignores it for now.
    rerender(
      <RecordingPanel {...defaultOperatorProps}
        recordingSnapshot={activeSnapshot({ episodePhase: "RECORDING", currentEpisodeId: "e9" })}
        actions={{ onStartEpisode }}
      />,
    );

    // This tab's own request is rejected (an episode is already active).
    await act(async () => {
      resolveStart({ outcome: "rejected", reason: "an episode is already active" });
    });

    // Must reconcile to the real current fact (Recording), not revert to
    // the stale "Idle" this command's own previousPhase would blindly apply.
    expect(statusText()).toMatch(/^Recording/);
  });

  it("a timed-out command lands on a genuine 'unknown' state that blocks every action until a real observed snapshot resolves it", async () => {
    const actions = {
      onStartEpisode: vi.fn().mockRejectedValue(new RecordingCommandUncertainError("timed out waiting for a reply")),
      onStopEpisode: vi.fn(),
      onDiscardEpisode: vi.fn(),
      onSetTask: vi.fn(),
      onNewBatch: vi.fn(),
    };
    const { rerender } = render(<RecordingPanel {...defaultOperatorProps} recordingSnapshot={activeSnapshot()} actions={actions} />);

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /record \(s\)/i }));
    });

    // Never silently reverts to idle/recording on an unresolved outcome.
    expect(statusText()).toMatch(/^Outcome unknown/);
    for (const name of [/record \(s\)/i, /stop\/save \(w\)/i, /discard \(a\)/i, /edit task \(t\)/i, /new batch \(r\)/i]) {
      const button = screen.getByRole("button", { name }) as HTMLButtonElement;
      expect(button.disabled).toBe(true);
      expect(button.title).toMatch(/outcome is unknown/i);
    }

    // The command actually succeeded on the backend -- a real observed
    // snapshot (e.g. from the active-resolution fresh-state read) is what
    // resolves the uncertainty, never a timer or a guess.
    rerender(
      <RecordingPanel {...defaultOperatorProps}
        recordingSnapshot={activeSnapshot({ episodePhase: "RECORDING", currentEpisodeId: "e1" })}
        actions={actions}
      />,
    );
    expect(statusText()).toMatch(/^Recording/);
  });

  it("a timed-out task save that did NOT actually take effect returns to editingTask with the draft intact, not silently losing the retry", async () => {
    const onSetTask = vi.fn().mockRejectedValue(new RecordingCommandUncertainError("timed out waiting for a reply"));
    const { rerender } = render(<RecordingPanel {...defaultOperatorProps} recordingSnapshot={activeSnapshot()} actions={{ onSetTask }} />);

    fireEvent.click(screen.getByRole("button", { name: /edit task \(t\)/i }));
    const input = screen.getByDisplayValue("Pour water into the cup.");
    fireEvent.change(input, { target: { value: "Fold the towel." } });
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Save" }));
    });
    expect(statusText()).toMatch(/^Outcome unknown/);

    // A real observed snapshot arrives whose task text is still the OLD
    // text -- the save genuinely did not take effect.
    rerender(<RecordingPanel {...defaultOperatorProps} recordingSnapshot={activeSnapshot()} actions={{ onSetTask }} />);

    // Back in editingTask with the user's draft intact, not idle with the
    // retry opportunity silently lost.
    expect(statusText()).toMatch(/^Editing task/);
    expect((screen.getByDisplayValue("Fold the towel.") as HTMLInputElement).value).toBe("Fold the towel.");
  });

  it("a timed-out task save that DID actually take effect resolves to idle, confirmed by the matching observed task text", async () => {
    const onSetTask = vi.fn().mockRejectedValue(new RecordingCommandUncertainError("timed out waiting for a reply"));
    const { rerender } = render(<RecordingPanel {...defaultOperatorProps} recordingSnapshot={activeSnapshot()} actions={{ onSetTask }} />);

    fireEvent.click(screen.getByRole("button", { name: /edit task \(t\)/i }));
    const input = screen.getByDisplayValue("Pour water into the cup.");
    fireEvent.change(input, { target: { value: "Fold the towel." } });
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Save" }));
    });
    expect(statusText()).toMatch(/^Outcome unknown/);

    // The observed snapshot's task text now matches the draft exactly --
    // the save DID take effect before its reply was lost.
    rerender(<RecordingPanel {...defaultOperatorProps} recordingSnapshot={activeSnapshot({ taskText: "Fold the towel." })} actions={{ onSetTask }} />);

    expect(statusText()).toMatch(/^Idle/);
    expect(statusText()).toMatch(/Fold the towel\./);
  });

  it("with no operator set, every control is disabled and shortcuts cannot fire, regardless of what the backend snapshot allows", () => {
    const onStartEpisode = vi.fn().mockResolvedValue({ outcome: "ok", reason: null });
    render(
      <RecordingPanel {...defaultOperatorProps} recordingSnapshot={activeSnapshot()} actions={{ onStartEpisode }} operatorToken={null} setOperatorToken={vi.fn()} />,
    );

    for (const name of [/record \(s\)/i, /stop\/save \(w\)/i, /discard \(a\)/i, /edit task \(t\)/i, /new batch \(r\)/i]) {
      expect((screen.getByRole("button", { name }) as HTMLButtonElement).disabled).toBe(true);
    }
    fireEvent.keyDown(window, { key: "s" });
    expect(onStartEpisode).not.toHaveBeenCalled();
    expect(screen.getByLabelText(/operator/i)).not.toBeNull();
  });

  it("entering an operator name and submitting calls setOperatorToken with the trimmed value", () => {
    const setOperatorToken = vi.fn();
    render(<RecordingPanel {...defaultOperatorProps} recordingSnapshot={activeSnapshot()} operatorToken={null} setOperatorToken={setOperatorToken} />);

    fireEvent.change(screen.getByLabelText(/operator/i), { target: { value: "  Alice  " } });
    fireEvent.click(screen.getByRole("button", { name: /set operator/i }));
    expect(setOperatorToken).toHaveBeenCalledWith("Alice");
  });

  it("with an operator set, New operator calls clearOperatorToken and returns the panel to requiring entry", () => {
    const clearOperatorToken = vi.fn();
    const operatorName = "Alice";
    render(<RecordingPanel {...defaultOperatorProps} recordingSnapshot={activeSnapshot()} operatorToken={operatorName} clearOperatorToken={clearOperatorToken} />);

    expect(screen.getByText(/operator: alice/i)).not.toBeNull();
    fireEvent.click(screen.getByRole("button", { name: /new operator/i }));
    expect(clearOperatorToken).toHaveBeenCalled();
  });

  it("the operator entry field caps input at 128 characters", () => {
    render(<RecordingPanel {...defaultOperatorProps} recordingSnapshot={activeSnapshot()} operatorToken={null} />);
    expect((screen.getByLabelText(/operator/i) as HTMLInputElement).maxLength).toBe(128);
  });

  it("clearing the operator mid task-edit stops Save from sending a command with no operator", () => {
    const onSetTask = vi.fn().mockResolvedValue({ outcome: "ok", reason: null });
    const { rerender } = render(
      <RecordingPanel {...defaultOperatorProps} recordingSnapshot={activeSnapshot()} actions={{ onSetTask }} operatorToken="<Alice>" />,
    );

    fireEvent.click(screen.getByRole("button", { name: /edit task \(t\)/i }));
    // "New operator" clears the token while the editor stays open -- the
    // local interaction phase is unaffected by that, only operatorToken is.
    rerender(<RecordingPanel {...defaultOperatorProps} recordingSnapshot={activeSnapshot()} actions={{ onSetTask }} operatorToken={null} />);

    // Save is disabled, but Enter in the input calls the same submit
    // handler directly -- exercise that path, not just the button.
    const input = screen.getByDisplayValue("Pour water into the cup.") as HTMLInputElement;
    fireEvent.keyDown(input, { key: "Enter" });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    expect(onSetTask).not.toHaveBeenCalled();
  });

  it("with no control token entered, every control is disabled and shortcuts cannot fire, regardless of what the backend snapshot allows", () => {
    // All five actions wired with real handlers -- if only onStartEpisode
    // were supplied, stop/discard/task-edit/new-batch would already be
    // disabled just because they have no handler, not because the token
    // gate actually blocks them. Wiring all five isolates the gate.
    const noop = vi.fn().mockResolvedValue({ outcome: "ok", reason: null });
    render(
      <RecordingPanel
        {...defaultOperatorProps}
        recordingSnapshot={activeSnapshot()}
        actions={{
          onStartEpisode: noop,
          onStopEpisode: noop,
          onDiscardEpisode: noop,
          onSetTask: noop,
          onNewBatch: noop,
        }}
        controlToken={null}
      />,
    );

    for (const name of [/record \(s\)/i, /stop\/save \(w\)/i, /discard \(a\)/i, /edit task \(t\)/i, /new batch \(r\)/i]) {
      expect((screen.getByRole("button", { name }) as HTMLButtonElement).disabled).toBe(true);
    }
    fireEvent.keyDown(window, { key: "s" });
    expect(noop).not.toHaveBeenCalled();
    expect(screen.getByLabelText(/control token/i)).not.toBeNull();
  });

  it("clicking Clear calls clearControlToken, and the entry form reappears empty", () => {
    const clearControlToken = vi.fn();
    const { rerender } = render(
      <RecordingPanel {...defaultOperatorProps} clearControlToken={clearControlToken} recordingSnapshot={activeSnapshot()} controlToken="<real-token>" />,
    );

    fireEvent.click(screen.getByRole("button", { name: /clear/i }));
    expect(clearControlToken).toHaveBeenCalledTimes(1);

    // A real parent would now pass controlToken={null} once state settles.
    rerender(<RecordingPanel {...defaultOperatorProps} clearControlToken={clearControlToken} recordingSnapshot={activeSnapshot()} controlToken={null} />);
    expect((screen.getByLabelText(/control token/i) as HTMLInputElement).value).toBe("");
  });

  it("a token that becomes null externally (not via the Clear button) still resets a stale draft", () => {
    // The Clear button's own onClick already empties the draft itself, so a
    // test that goes through it can pass even if the draft-reset effect is
    // broken/removed. This exercises the effect in isolation: an unsubmitted
    // draft is sitting in the entry form, a token arrives from elsewhere
    // (unmounting the form without ever touching the draft), and the token
    // is cleared externally again (e.g. the backend_instance_id auto-clear)
    // -- the reappearing form must not show the old draft text.
    const { rerender } = render(<RecordingPanel {...defaultOperatorProps} recordingSnapshot={activeSnapshot()} controlToken={null} />);
    fireEvent.change(screen.getByLabelText(/control token/i), { target: { value: "stale-draft" } });
    expect((screen.getByLabelText(/control token/i) as HTMLInputElement).value).toBe("stale-draft");

    rerender(<RecordingPanel {...defaultOperatorProps} recordingSnapshot={activeSnapshot()} controlToken="<some-token>" />);
    expect(screen.queryByLabelText(/control token/i)).toBeNull();

    rerender(<RecordingPanel {...defaultOperatorProps} recordingSnapshot={activeSnapshot()} controlToken={null} />);
    expect((screen.getByLabelText(/control token/i) as HTMLInputElement).value).toBe("");
  });

  it("clearing the control token mid task-edit stops Save from sending a command with no credential", async () => {
    const onSetTask = vi.fn().mockResolvedValue({ outcome: "ok", reason: null });
    const { rerender } = render(
      <RecordingPanel {...defaultOperatorProps} recordingSnapshot={activeSnapshot()} actions={{ onSetTask }} controlToken="<real-token>" />,
    );

    fireEvent.click(screen.getByRole("button", { name: /edit task \(t\)/i }));
    // Clearing the token happens while the editor stays open -- the local
    // interaction phase is unaffected by that, only controlToken is.
    rerender(
      <RecordingPanel {...defaultOperatorProps} recordingSnapshot={activeSnapshot()} actions={{ onSetTask }} controlToken={null} />,
    );

    // Save is disabled, but Enter in the input calls the same submit
    // handler directly -- exercise that path, not just the button.
    const input = screen.getByDisplayValue("Pour water into the cup.") as HTMLInputElement;
    fireEvent.keyDown(input, { key: "Enter" });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    expect(onSetTask).not.toHaveBeenCalled();
  });

  it("a station default operator arriving while the user is mid-entry never overwrites their draft", () => {
    const setOperatorToken = vi.fn();
    const { rerender } = render(
      <RecordingPanel
        {...defaultOperatorProps}
        operatorToken={null}
        setOperatorToken={setOperatorToken}
        recordingSnapshot={activeSnapshot({ defaultOperator: null, autoFillOperatorName: true })}
      />,
    );
    const input = screen.getByLabelText("Operator") as HTMLInputElement;
    fireEvent.change(input, { target: { value: "Alex" } });

    // The station's default arrives only now, after the user already started typing.
    rerender(
      <RecordingPanel
        {...defaultOperatorProps}
        operatorToken={null}
        setOperatorToken={setOperatorToken}
        recordingSnapshot={activeSnapshot({ defaultOperator: "Susan Calvin", autoFillOperatorName: true, revision: 1 })}
      />,
    );

    expect(setOperatorToken).not.toHaveBeenCalled();
    expect(input.value).toBe("Alex");
  });

  it("a deliberately cleared operator entry is never auto-refilled by a later-arriving default", () => {
    const setOperatorToken = vi.fn();
    const { rerender } = render(
      <RecordingPanel
        {...defaultOperatorProps}
        operatorToken={null}
        setOperatorToken={setOperatorToken}
        recordingSnapshot={activeSnapshot({ defaultOperator: null, autoFillOperatorName: true })}
      />,
    );
    const input = screen.getByLabelText("Operator") as HTMLInputElement;
    // Types then deletes it back to blank -- a deliberate clear, not an untouched field.
    fireEvent.change(input, { target: { value: "Alex" } });
    fireEvent.change(input, { target: { value: "" } });

    rerender(
      <RecordingPanel
        {...defaultOperatorProps}
        operatorToken={null}
        setOperatorToken={setOperatorToken}
        recordingSnapshot={activeSnapshot({ defaultOperator: "Susan Calvin", autoFillOperatorName: true, revision: 1 })}
      />,
    );

    expect(setOperatorToken).not.toHaveBeenCalled();
    expect(input.value).toBe("");
  });

  it("a station default operator only prefills the DRAFT, never commits, when the station opts in", () => {
    const setOperatorToken = vi.fn();
    const { rerender } = render(
      <RecordingPanel
        {...defaultOperatorProps}
        operatorToken={null}
        setOperatorToken={setOperatorToken}
        recordingSnapshot={activeSnapshot({ defaultOperator: null, autoFillOperatorName: true })}
      />,
    );

    rerender(
      <RecordingPanel
        {...defaultOperatorProps}
        operatorToken={null}
        setOperatorToken={setOperatorToken}
        recordingSnapshot={activeSnapshot({ defaultOperator: "Susan Calvin", autoFillOperatorName: true, revision: 1 })}
      />,
    );

    // Never auto-committed -- the operator still has to press "Set operator".
    expect(setOperatorToken).not.toHaveBeenCalled();
    expect((screen.getByLabelText("Operator") as HTMLInputElement).value).toBe("Susan Calvin");
  });

  it("a station default operator is never applied at all when the station has not opted in (off by default)", () => {
    const setOperatorToken = vi.fn();
    const { rerender } = render(
      <RecordingPanel
        {...defaultOperatorProps}
        operatorToken={null}
        setOperatorToken={setOperatorToken}
        recordingSnapshot={activeSnapshot({ defaultOperator: null, autoFillOperatorName: false })}
      />,
    );

    rerender(
      <RecordingPanel
        {...defaultOperatorProps}
        operatorToken={null}
        setOperatorToken={setOperatorToken}
        recordingSnapshot={activeSnapshot({ defaultOperator: "Susan Calvin", autoFillOperatorName: false, revision: 1 })}
      />,
    );

    expect(setOperatorToken).not.toHaveBeenCalled();
    expect((screen.getByLabelText("Operator") as HTMLInputElement).value).toBe("");
  });
});
