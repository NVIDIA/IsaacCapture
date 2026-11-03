/**
 * Voice-command behavior through the real RecordingPanel: Space toggling,
 * duplicate/invalid dispatch guards, the phase-conditional task-draft
 * dispatch (dispatching BEGIN_TASK_EDIT when idle vs UPDATE_TASK_DRAFT when
 * already editing, so a second "task <text>" updates the draft instead of
 * silently no-opping), and the discard confirmation step. Drives the real
 * useVoiceRecognition hook (not a
 * component-level mock of it) via a fake `window.SpeechRecognition`
 * constructor -- the same small-fake-event-source pattern as
 * useVoiceRecognition.test.ts, just wired through the real browser global
 * this component's default factory actually reads.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { RecordingPanel } from "./RecordingPanel";
import type { RecordingSnapshot } from "./recordingTypes";
import type { SpeechRecognitionLike } from "./useVoiceRecognition";

class FakeRecognition implements SpeechRecognitionLike {
  static instances: FakeRecognition[] = [];
  continuous = false;
  interimResults = false;
  lang = "";
  onstart: SpeechRecognitionLike["onstart"] = null;
  onspeechstart: SpeechRecognitionLike["onspeechstart"] = null;
  onresult: SpeechRecognitionLike["onresult"] = null;
  onerror: SpeechRecognitionLike["onerror"] = null;
  onend: SpeechRecognitionLike["onend"] = null;
  // Real SpeechRecognitionResultList is cumulative for the whole session;
  // mirrored here so the hook's dedup guard sees realistic behavior.
  private accumulated: { isFinal: boolean; 0: { transcript: string } }[] = [];

  constructor() {
    FakeRecognition.instances.push(this);
  }

  start(): void {}
  stop(): void {}

  emitStart(): void {
    this.onstart?.();
  }

  emitFinal(transcript: string): void {
    const resultIndex = this.accumulated.length;
    this.accumulated.push({ isFinal: true, 0: { transcript } });
    this.onresult?.({ resultIndex, results: this.accumulated });
  }

  /** Reports interim text for a NEW utterance -- this, not onspeechstart (which with
   * continuous=true fires once per recognizer SESSION, not once per utterance), is what the real
   * hook binds the utterance-start callback to. Use this instead of bare emitFinal whenever a
   * test cares about the per-utterance target binding. */
  emitInterim(transcript: string): void {
    const resultIndex = this.accumulated.length;
    this.accumulated.push({ isFinal: false, 0: { transcript } });
    this.onresult?.({ resultIndex, results: this.accumulated });
  }

  /** Finalizes the CURRENT (most recently pushed) utterance's entry IN PLACE -- a real browser
   * reports interim-then-final updates at the SAME resultIndex, never a new one. */
  finalizeLast(transcript: string): void {
    const resultIndex = this.accumulated.length - 1;
    this.accumulated[resultIndex] = { isFinal: true, 0: { transcript } };
    this.onresult?.({ resultIndex, results: this.accumulated });
  }

  /** Simulates a complete utterance: an interim observation, then its final result -- use this
   * instead of bare emitFinal whenever a test cares about the per-utterance target binding. */
  speak(transcript: string): void {
    this.emitInterim(transcript);
    this.finalizeLast(transcript);
  }
}

beforeEach(() => {
  FakeRecognition.instances = [];
  (window as unknown as { SpeechRecognition: unknown }).SpeechRecognition = FakeRecognition;
});

afterEach(() => {
  cleanup();
  delete (window as unknown as { SpeechRecognition?: unknown }).SpeechRecognition;
});

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

function enableVoice(): FakeRecognition {
  fireEvent.click(screen.getByRole("button", { name: /voice: off/i }));
  return FakeRecognition.instances[FakeRecognition.instances.length - 1];
}

function voiceFeedbackText(): string {
  return document.querySelector(".recording-voice-status")?.textContent ?? "";
}

describe("RecordingPanel voice commands", () => {
  it("renders the voice toggle and Space enables it", () => {
    render(<RecordingPanel {...defaultOperatorProps} recordingSnapshot={activeSnapshot()} />);
    expect(screen.getByRole("button", { name: /voice: off/i })).toBeTruthy();
    fireEvent.keyDown(window, { key: " " });
    expect(screen.getByRole("button", { name: /voice: starting/i })).toBeTruthy();
    act(() => FakeRecognition.instances[FakeRecognition.instances.length - 1].emitStart());
    expect(screen.getByRole("button", { name: /voice: listening/i })).toBeTruthy();
  });

  it("does not toggle on Space while focus is in a text input", () => {
    render(<RecordingPanel {...defaultOperatorProps} recordingSnapshot={activeSnapshot()} />);
    const operatorExtra = document.createElement("input");
    document.body.appendChild(operatorExtra);
    operatorExtra.focus();
    fireEvent.keyDown(window, { key: " " });
    expect(screen.getByRole("button", { name: /voice: off/i })).toBeTruthy();
    document.body.removeChild(operatorExtra);
  });

  it("does not toggle on Space while focus is in a <summary> element (native Space activation)", () => {
    render(<RecordingPanel {...defaultOperatorProps} recordingSnapshot={activeSnapshot()} />);
    const summary = document.createElement("summary");
    summary.tabIndex = 0;
    document.body.appendChild(summary);
    summary.focus();
    fireEvent.keyDown(window, { key: " " });
    expect(screen.getByRole("button", { name: /voice: off/i })).toBeTruthy();
    document.body.removeChild(summary);
  });

  it("does not swallow Space when voice is unsupported (no toggle rendered)", () => {
    delete (window as unknown as { SpeechRecognition?: unknown }).SpeechRecognition;
    render(<RecordingPanel {...defaultOperatorProps} recordingSnapshot={activeSnapshot()} />);
    expect(screen.queryByRole("button", { name: /voice:/i })).toBeNull();
    // fireEvent.keyDown returns false if preventDefault() was called -- assert the default
    // (e.g. a page scroll) was left alone, not just that the handler didn't throw.
    expect(fireEvent.keyDown(window, { key: " " })).toBe(true);
  });

  it('"record" dispatches the same onStartEpisode handler as the Record button, by the correct batch', async () => {
    const onStartEpisode = vi.fn().mockResolvedValue({ outcome: "ok", reason: null });
    render(<RecordingPanel {...defaultOperatorProps} recordingSnapshot={activeSnapshot()} actions={{ onStartEpisode }} />);
    const recognition = enableVoice();
    await act(async () => recognition.speak("start recording"));
    expect(onStartEpisode).toHaveBeenCalledWith({ batchId: "b1" });
  });

  it('a target-bound "record" with the binding resolved fine but the action itself ineligible shows the real reason, not "please repeat"', async () => {
    const onStartEpisode = vi.fn().mockResolvedValue({ outcome: "ok", reason: null });
    render(
      <RecordingPanel
        {...defaultOperatorProps}
        recordingSnapshot={activeSnapshot({ taskText: "" })} // no task set -- startEpisode is ineligible
        actions={{ onStartEpisode }}
      />,
    );
    const recognition = enableVoice();
    await act(async () => recognition.speak("record"));
    expect(onStartEpisode).not.toHaveBeenCalled();
    expect(voiceFeedbackText()).toMatch(/set a task first/i);
    expect(voiceFeedbackText()).not.toMatch(/please repeat/i); // repeating can't fix a missing task
  });

  it('a "record" with no recording snapshot at all (backend not connected) shows the real reason, not "please repeat"', async () => {
    // With no snapshot, resolveUtteranceTarget's own live.batchId is null, so it would ALSO
    // reject -- but eligibility.startEpisode's "Recording backend not connected yet" is checked
    // first and is the far more useful message; repeating can't fix a disconnected backend either.
    const onStartEpisode = vi.fn().mockResolvedValue({ outcome: "ok", reason: null });
    render(<RecordingPanel {...defaultOperatorProps} recordingSnapshot={null} actions={{ onStartEpisode }} />);
    const recognition = enableVoice();
    await act(async () => recognition.speak("record"));
    expect(onStartEpisode).not.toHaveBeenCalled();
    expect(voiceFeedbackText()).toMatch(/recording backend not connected yet/i);
    expect(voiceFeedbackText()).not.toMatch(/please repeat/i);
  });

  it('a target-bound "stop" with nothing currently recording shows the real reason, not "please repeat"', async () => {
    const onStopEpisode = vi.fn().mockResolvedValue({ outcome: "ok", reason: null });
    render(
      <RecordingPanel
        {...defaultOperatorProps}
        recordingSnapshot={activeSnapshot({ episodePhase: "IDLE" })} // nothing recording
        actions={{ onStopEpisode }}
      />,
    );
    const recognition = enableVoice();
    await act(async () => recognition.speak("stop"));
    expect(onStopEpisode).not.toHaveBeenCalled();
    expect(voiceFeedbackText()).toMatch(/no active recording to stop/i);
    expect(voiceFeedbackText()).not.toMatch(/please repeat/i);
  });

  it('a target-bound "discard" with nothing currently recording shows the real reason, not "please repeat"', async () => {
    const onDiscardEpisode = vi.fn().mockResolvedValue({ outcome: "ok", reason: null });
    render(
      <RecordingPanel
        {...defaultOperatorProps}
        recordingSnapshot={activeSnapshot({ episodePhase: "IDLE" })} // nothing recording
        actions={{ onDiscardEpisode }}
      />,
    );
    const recognition = enableVoice();
    await act(async () => recognition.speak("discard"));
    expect(onDiscardEpisode).not.toHaveBeenCalled();
    expect(voiceFeedbackText()).toMatch(/no active episode to discard/i);
    expect(voiceFeedbackText()).not.toMatch(/please repeat/i);
  });

  it('a full "record", 10s take, "stop" sequence through one recognizer session, each command bound to its own live snapshot', async () => {
    // The exact "record, 10s take, stop" scenario the original review described failing: two
    // separate commands in ONE recognizer session, with the recording snapshot genuinely
    // changing (IDLE -> RECORDING) between them, each getting its own fresh binding.
    const onStartEpisode = vi.fn().mockResolvedValue({ outcome: "ok", reason: null });
    const onStopEpisode = vi.fn().mockResolvedValue({ outcome: "ok", reason: null });
    const { rerender } = render(
      <RecordingPanel
        {...defaultOperatorProps}
        recordingSnapshot={activeSnapshot({ episodePhase: "IDLE" })}
        actions={{ onStartEpisode, onStopEpisode }}
      />,
    );
    const recognition = enableVoice();
    await act(async () => recognition.speak("record"));
    expect(onStartEpisode).toHaveBeenCalledWith({ batchId: "b1" });

    // The take is now live -- a real backend snapshot update, not a guess.
    rerender(
      <RecordingPanel
        {...defaultOperatorProps}
        recordingSnapshot={activeSnapshot({ episodePhase: "RECORDING", currentEpisodeId: "e1" })}
        actions={{ onStartEpisode, onStopEpisode }}
      />,
    );
    await act(async () => recognition.speak("stop"));
    expect(onStopEpisode).toHaveBeenCalledWith({ batchId: "b1", episodeId: "e1" });
  });

  it('"task <text>" begins the task editor when idle, and a second "task <text>" updates the draft instead of silently no-opping', async () => {
    const onSetTask = vi.fn().mockResolvedValue({ outcome: "ok", reason: null });
    render(<RecordingPanel {...defaultOperatorProps} recordingSnapshot={activeSnapshot()} actions={{ onSetTask }} />);
    const recognition = enableVoice();
    await act(async () => recognition.emitFinal("task fold the towel"));
    const input = document.querySelector(".recording-task-editor-input") as HTMLInputElement;
    expect(input.value).toBe("fold the towel");

    await act(async () => recognition.emitFinal("task wash the dishes"));
    const inputAfter = document.querySelector(".recording-task-editor-input") as HTMLInputElement;
    expect(inputAfter.value).toBe("wash the dishes");
  });

  it('"save task" commits the voice-dictated draft through the existing Save path', async () => {
    const onSetTask = vi.fn().mockResolvedValue({ outcome: "ok", reason: null });
    render(<RecordingPanel {...defaultOperatorProps} recordingSnapshot={activeSnapshot()} actions={{ onSetTask }} />);
    const recognition = enableVoice();
    await act(async () => recognition.emitFinal("task fold the towel"));
    await act(async () => recognition.emitFinal("save task"));
    expect(onSetTask).toHaveBeenCalledWith({ batchId: "b1" }, "fold the towel");
  });

  it('"discard" opens the browser\'s native confirm() rather than a second spoken command; declining sends no command', async () => {
    const onDiscardEpisode = vi.fn().mockResolvedValue({ outcome: "ok", reason: null });
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(false);
    render(
      <RecordingPanel
        {...defaultOperatorProps}
        recordingSnapshot={activeSnapshot({ episodePhase: "RECORDING", currentEpisodeId: "e1" })}
        actions={{ onDiscardEpisode }}
      />,
    );
    const recognition = enableVoice();
    await act(async () => recognition.speak("discard"));
    expect(confirmSpy).toHaveBeenCalledWith("Discard this take?");
    expect(onDiscardEpisode).not.toHaveBeenCalled();
    expect(voiceFeedbackText()).toMatch(/discard cancelled/i);
    confirmSpy.mockRestore();
  });

  it("accepting window.confirm dispatches onDiscardEpisode with the bound target", async () => {
    const onDiscardEpisode = vi.fn().mockResolvedValue({ outcome: "ok", reason: null });
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(true);
    render(
      <RecordingPanel
        {...defaultOperatorProps}
        recordingSnapshot={activeSnapshot({ episodePhase: "RECORDING", currentEpisodeId: "e1" })}
        actions={{ onDiscardEpisode }}
      />,
    );
    const recognition = enableVoice();
    await act(async () => recognition.speak("discard"));
    expect(onDiscardEpisode).toHaveBeenCalledWith({ batchId: "b1", episodeId: "e1" });
    confirmSpy.mockRestore();
  });

  it("while editing the task, a spoken record/stop command is ignored with feedback rather than dispatched", async () => {
    const onStartEpisode = vi.fn().mockResolvedValue({ outcome: "ok", reason: null });
    const onSetTask = vi.fn().mockResolvedValue({ outcome: "ok", reason: null });
    render(<RecordingPanel {...defaultOperatorProps} recordingSnapshot={activeSnapshot()} actions={{ onStartEpisode, onSetTask }} />);
    const recognition = enableVoice();
    await act(async () => recognition.emitFinal("task fold the towel")); // enters editingTask
    await act(async () => recognition.emitFinal("record"));
    expect(onStartEpisode).not.toHaveBeenCalled();
    expect(voiceFeedbackText()).toMatch(/finish or cancel the task edit/i);
  });

  it("rejects a delayed command whose bound-at-utterance-start batch is no longer current, rather than retargeting or using the stale one", async () => {
    const onStartEpisode = vi.fn().mockResolvedValue({ outcome: "ok", reason: null });
    const { rerender } = render(
      <RecordingPanel {...defaultOperatorProps} recordingSnapshot={activeSnapshot({ batchId: "old-batch" })} actions={{ onStartEpisode }} />,
    );
    const recognition = enableVoice();
    act(() => recognition.emitInterim("rec")); // interim observation binds the target to "old-batch", live right now
    // The batch rotates (e.g. another viewer's New batch) before the utterance's final result arrives.
    rerender(<RecordingPanel {...defaultOperatorProps} recordingSnapshot={activeSnapshot({ batchId: "new-batch" })} actions={{ onStartEpisode }} />);
    await act(async () => recognition.finalizeLast("record"));
    expect(onStartEpisode).not.toHaveBeenCalled();
    expect(voiceFeedbackText()).toMatch(/voice command target is unknown or changed/i);
  });

  it("rejects a final-only (never interim-observed) stop after the target has since changed, rather than silently binding to the new one at delivery time", async () => {
    // The critical safety case: a result that arrives already final with no interim first must
    // NEVER bind to whatever target happens to be live at delivery -- it falls through to the
    // same "no binding was ever captured" reject path as if onUtteranceStart never fired at all.
    const onStopEpisode = vi.fn().mockResolvedValue({ outcome: "ok", reason: null });
    const { rerender } = render(
      <RecordingPanel
        {...defaultOperatorProps}
        recordingSnapshot={activeSnapshot({ episodePhase: "RECORDING", currentEpisodeId: "old-episode" })}
        actions={{ onStopEpisode }}
      />,
    );
    const recognition = enableVoice();
    // The episode changes with NO interim ever observed for this utterance (e.g. a very short
    // "stop" recognized in one shot) before its final result arrives.
    rerender(
      <RecordingPanel
        {...defaultOperatorProps}
        recordingSnapshot={activeSnapshot({ episodePhase: "RECORDING", currentEpisodeId: "new-episode" })}
        actions={{ onStopEpisode }}
      />,
    );
    await act(async () => recognition.emitFinal("stop")); // bare final, no prior interim
    expect(onStopEpisode).not.toHaveBeenCalled();
    expect(voiceFeedbackText()).toMatch(/voice command target is unknown or changed/i);
  });

  it("dispatches normally when the bound-at-utterance-start target is still current at result time", async () => {
    const onStartEpisode = vi.fn().mockResolvedValue({ outcome: "ok", reason: null });
    render(<RecordingPanel {...defaultOperatorProps} recordingSnapshot={activeSnapshot({ batchId: "b1" })} actions={{ onStartEpisode }} />);
    const recognition = enableVoice();
    await act(async () => recognition.speak("record")); // interim + final, nothing changes in between
    expect(onStartEpisode).toHaveBeenCalledWith({ batchId: "b1" });
  });

  it("rejects (never falls back to the live target) when no utterance-start binding was ever captured, e.g. a bare final with no interim observation", async () => {
    const onStartEpisode = vi.fn().mockResolvedValue({ outcome: "ok", reason: null });
    render(<RecordingPanel {...defaultOperatorProps} recordingSnapshot={activeSnapshot({ batchId: "b1" })} actions={{ onStartEpisode }} />);
    const recognition = enableVoice();
    await act(async () => recognition.emitFinal("record")); // no emitInterim() call at all
    expect(onStartEpisode).not.toHaveBeenCalled();
    expect(voiceFeedbackText()).toMatch(/voice command target is unknown or changed/i);
  });

  it("binds once for one utterance reported as a stable-prefix entry then an unstable-tail entry, not two overlapping utterances", async () => {
    // The hook's own pending-interim tracking (see useVoiceRecognition.ts) treats a second interim
    // entry arriving before a final as the SAME in-progress utterance, not a new overlapping one --
    // onUtteranceStart fires once, so this binds "b1" once and dispatches normally.
    const onStartEpisode = vi.fn().mockResolvedValue({ outcome: "ok", reason: null });
    render(<RecordingPanel {...defaultOperatorProps} recordingSnapshot={activeSnapshot({ batchId: "b1" })} actions={{ onStartEpisode }} />);
    const recognition = enableVoice();
    act(() => recognition.emitInterim("rec")); // stable-prefix entry
    act(() => recognition.emitInterim("record")); // unstable-tail entry -- same utterance
    await act(async () => recognition.finalizeLast("record"));
    expect(onStartEpisode).toHaveBeenCalledTimes(1);
  });

  it("delayed Stop across an episode change is rejected, never applied to the newer episode", async () => {
    const onStopEpisode = vi.fn().mockResolvedValue({ outcome: "ok", reason: null });
    const { rerender } = render(
      <RecordingPanel
        {...defaultOperatorProps}
        recordingSnapshot={activeSnapshot({ episodePhase: "RECORDING", currentEpisodeId: "old-episode" })}
        actions={{ onStopEpisode }}
      />,
    );
    const recognition = enableVoice();
    act(() => recognition.emitInterim("st")); // interim observation binds "old-episode", live right now
    // The episode changes (e.g. the take was stopped/discarded and a new one started by another
    // viewer) before the utterance's final result arrives.
    rerender(
      <RecordingPanel
        {...defaultOperatorProps}
        recordingSnapshot={activeSnapshot({ episodePhase: "RECORDING", currentEpisodeId: "new-episode" })}
        actions={{ onStopEpisode }}
      />,
    );
    await act(async () => recognition.finalizeLast("stop"));
    expect(onStopEpisode).not.toHaveBeenCalled();
    expect(voiceFeedbackText()).toMatch(/voice command target is unknown or changed/i);
  });

  it("clears an unfinished utterance binding when the recognizer session resets (disable/re-enable before a final result)", async () => {
    const onStartEpisode = vi.fn().mockResolvedValue({ outcome: "ok", reason: null });
    render(<RecordingPanel {...defaultOperatorProps} recordingSnapshot={activeSnapshot({ batchId: "b1" })} actions={{ onStartEpisode }} />);
    const recognition = enableVoice();
    act(() => recognition.emitInterim("record")); // interim observation begins, binds "b1" -- but no final result ever arrives
    // Space off, then Space on again -- a fresh recognizer session, before the first utterance
    // ever resolved. Without the session-start reset, the next utterance's own interim
    // observation would see the old unconsumed binding and mark itself "invalid", rejecting the
    // first real command.
    fireEvent.click(screen.getByRole("button", { name: /voice: starting/i }));
    const recognition2 = enableVoice();
    await act(async () => recognition2.speak("record"));
    expect(onStartEpisode).toHaveBeenCalledWith({ batchId: "b1" });
  });

  it("a repeated interim update for the same utterance does not rebind onto a target that changed in between", async () => {
    // The real safety property under test: the FIRST interim observation is what binds the
    // target, once. A browser repeatedly reporting refined interim text for the SAME utterance
    // (e.g. "rec" -> "reco" -> "record") must never look like a second, fresh onUtteranceStart --
    // if it did, the binding would silently pick up whatever target is live at that LATER
    // interim update, not the one live when the utterance actually began.
    const onStartEpisode = vi.fn().mockResolvedValue({ outcome: "ok", reason: null });
    const { rerender } = render(
      <RecordingPanel {...defaultOperatorProps} recordingSnapshot={activeSnapshot({ batchId: "old-batch" })} actions={{ onStartEpisode }} />,
    );
    const recognition = enableVoice();
    act(() => recognition.emitInterim("rec")); // binds "old-batch", the first and only observation
    rerender(<RecordingPanel {...defaultOperatorProps} recordingSnapshot={activeSnapshot({ batchId: "new-batch" })} actions={{ onStartEpisode }} />);
    // A repeated interim update for the SAME utterance (same resultIndex) -- must not re-fire
    // onUtteranceStart or rebind onto "new-batch".
    act(() => recognition.onresult?.({ resultIndex: 0, results: [{ isFinal: false, 0: { transcript: "record" } }] }));
    await act(async () => recognition.finalizeLast("record"));
    expect(onStartEpisode).not.toHaveBeenCalled(); // rejected: bound target ("old-batch") no longer matches live ("new-batch")
    expect(voiceFeedbackText()).toMatch(/voice command target is unknown or changed/i);
  });
});
