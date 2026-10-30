// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Lifecycle tests for useVoiceRecognition against a small fake recognizer --
 * not a mock-browser framework, just enough of the real SpeechRecognition
 * surface (onresult/onerror/onend, start/stop) to drive the hook's own
 * restart/session-guard/error logic.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, renderHook } from "@testing-library/react";
import { useVoiceRecognition, type SpeechRecognitionLike } from "./useVoiceRecognition";

afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

class FakeRecognition implements SpeechRecognitionLike {
  static instances: FakeRecognition[] = [];
  continuous = false;
  interimResults = false;
  lang = "";
  started = false;
  stopped = false;
  onstart: SpeechRecognitionLike["onstart"] = null;
  onspeechstart: SpeechRecognitionLike["onspeechstart"] = null;
  onresult: SpeechRecognitionLike["onresult"] = null;
  onerror: SpeechRecognitionLike["onerror"] = null;
  onend: SpeechRecognitionLike["onend"] = null;
  // Real SpeechRecognitionResultList is cumulative for the whole session --
  // each onresult event reports the FULL list so far, with resultIndex
  // pointing at where the new entries start. Mirrored here so the dedup
  // guard under test sees realistic behavior.
  private accumulated: { isFinal: boolean; 0: { transcript: string } }[] = [];

  constructor() {
    FakeRecognition.instances.push(this);
  }

  start(): void {
    this.started = true;
  }

  stop(): void {
    this.stopped = true;
  }

  // --- test-only driver methods ---
  emitFinal(transcript: string): void {
    const resultIndex = this.accumulated.length;
    this.accumulated.push({ isFinal: true, 0: { transcript } });
    this.onresult?.({ resultIndex, results: this.accumulated });
  }

  /** Reports interim text for a NEW utterance -- pushes a new, not-yet-final entry, same as a
   * real browser's first report of a just-begun utterance. */
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

  /** Re-delivers the full accumulated list at a given resultIndex -- for
   * exercising the dedup guard against a genuinely overlapping/repeated range. */
  emitRaw(resultIndex: number, results: { isFinal: boolean; 0: { transcript: string } }[]): void {
    this.onresult?.({ resultIndex, results });
  }

  emitEnd(): void {
    this.onend?.();
  }

  emitError(error: string): void {
    this.onerror?.({ error });
  }

  emitStart(): void {
    this.onstart?.();
  }
}

function setup(onFinalResult = vi.fn()) {
  FakeRecognition.instances = [];
  const hook = renderHook(() =>
    useVoiceRecognition(onFinalResult, {
      recognitionFactory: () => new FakeRecognition(),
      isSupported: () => true,
    }),
  );
  return { ...hook, onFinalResult };
}

describe("useVoiceRecognition", () => {
  it("starts disabled and reports unsupported when isSupported() is false", () => {
    const { result } = renderHook(() =>
      useVoiceRecognition(vi.fn(), { recognitionFactory: () => new FakeRecognition(), isSupported: () => false }),
    );
    expect(result.current.status).toBe("unsupported");
    expect(result.current.enabled).toBe(false);
  });

  it('reports "starting" until the recognizer actually fires onstart, then "listening"', () => {
    const { result } = setup();
    act(() => result.current.toggle());
    expect(result.current.enabled).toBe(true);
    expect(result.current.status).toBe("starting"); // enabled=true is not yet proof the mic is listening
    act(() => FakeRecognition.instances[0].emitStart());
    expect(result.current.status).toBe("listening");
  });

  it("starts one recognizer on toggle and delivers a final result", () => {
    const { result, onFinalResult } = setup();
    act(() => result.current.toggle());
    expect(FakeRecognition.instances).toHaveLength(1);
    act(() => FakeRecognition.instances[0].emitFinal("record"));
    expect(onFinalResult).toHaveBeenCalledWith("record");
  });

  it("fires onUtteranceStart once per utterance, for one speechstart-like observation followed by two interim+final utterances", () => {
    // Does not rely on onspeechstart being per-utterance in the first place: with continuous=true,
    // Chrome is reported (not independently verified here) to fire it once per recognizer
    // SESSION, not once per utterance -- so without the interim-result-based rebind, only the
    // FIRST command after enabling voice mode would ever get a fresh onUtteranceStart; every
    // later command in the same session would bind onto stale state.
    const onUtteranceStart = vi.fn();
    FakeRecognition.instances = [];
    const { result } = renderHook(() =>
      useVoiceRecognition(vi.fn(), {
        recognitionFactory: () => new FakeRecognition(),
        isSupported: () => true,
        onUtteranceStart,
      }),
    );
    act(() => result.current.toggle());
    const first = FakeRecognition.instances[0];
    expect(onUtteranceStart).not.toHaveBeenCalled();
    act(() => first.emitInterim("rec")); // the browser's first observation of utterance #1
    expect(onUtteranceStart).toHaveBeenCalledTimes(1);
    act(() => first.finalizeLast("record")); // same utterance's eventual final -- no re-fire
    expect(onUtteranceStart).toHaveBeenCalledTimes(1);
    act(() => first.emitInterim("st")); // utterance #2's own first interim observation
    expect(onUtteranceStart).toHaveBeenCalledTimes(2);
    act(() => first.finalizeLast("stop"));
    expect(onUtteranceStart).toHaveBeenCalledTimes(2); // still just the two utterances' own starts
  });

  it("never fires onUtteranceStart for a result that arrives already final with no interim first -- it must fall through to the caller's reject/repeat path, not silently bind at delivery time", () => {
    // This is the critical safety property: a target-bound command whose final is delayed until
    // AFTER the live batch/episode changes must never bind to the NEW current target at delivery
    // time -- that would defeat the entire reason onUtteranceStart exists. A skipped-interim
    // final is treated exactly like "no onUtteranceStart ever fired for this utterance".
    const onUtteranceStart = vi.fn();
    const onFinalResult = vi.fn();
    FakeRecognition.instances = [];
    const { result } = renderHook(() =>
      useVoiceRecognition(onFinalResult, {
        recognitionFactory: () => new FakeRecognition(),
        isSupported: () => true,
        onUtteranceStart,
      }),
    );
    act(() => result.current.toggle());
    const first = FakeRecognition.instances[0];
    act(() => first.emitFinal("stop")); // arrives already final -- no prior interim observation
    expect(onUtteranceStart).not.toHaveBeenCalled();
    expect(onFinalResult).toHaveBeenCalledWith("stop"); // the transcript itself is still delivered
  });

  it("does not re-fire onUtteranceStart for repeated interim updates to the same utterance", () => {
    const onUtteranceStart = vi.fn();
    FakeRecognition.instances = [];
    const { result } = renderHook(() =>
      useVoiceRecognition(vi.fn(), {
        recognitionFactory: () => new FakeRecognition(),
        isSupported: () => true,
        onUtteranceStart,
      }),
    );
    act(() => result.current.toggle());
    const first = FakeRecognition.instances[0];
    act(() => first.emitInterim("r"));
    act(() => first.onresult?.({ resultIndex: 0, results: [{ isFinal: false, 0: { transcript: "re" } }] }));
    act(() => first.onresult?.({ resultIndex: 0, results: [{ isFinal: false, 0: { transcript: "rec" } }] }));
    act(() => first.finalizeLast("record"));
    expect(onUtteranceStart).toHaveBeenCalledTimes(1);
  });

  it("treats a second interim entry arriving before a final as the SAME pending utterance, not two", () => {
    // The spec allows more than one interim result index before an utterance's final; since the
    // event stream alone can't distinguish that from a genuinely new utterance, the second index
    // is treated as the SAME pending one here. onUtteranceStart must fire once, not once per index.
    const onUtteranceStart = vi.fn();
    const onFinalResult = vi.fn();
    FakeRecognition.instances = [];
    const { result } = renderHook(() =>
      useVoiceRecognition(onFinalResult, {
        recognitionFactory: () => new FakeRecognition(),
        isSupported: () => true,
        onUtteranceStart,
      }),
    );
    act(() => result.current.toggle());
    const first = FakeRecognition.instances[0];
    act(() => first.emitInterim("st")); // stable-prefix entry, index 0
    expect(onUtteranceStart).toHaveBeenCalledTimes(1);
    act(() => first.emitInterim("stop")); // unstable-tail entry, index 1 -- same utterance
    expect(onUtteranceStart).toHaveBeenCalledTimes(1); // no re-fire for the second index
    act(() => first.finalizeLast("stop"));
    expect(onUtteranceStart).toHaveBeenCalledTimes(1); // still just the one utterance
    expect(onFinalResult).toHaveBeenCalledTimes(1);
    expect(onFinalResult).toHaveBeenCalledWith("stop");
    // A fresh utterance arriving already final (no interim of its own) must still deliver its
    // transcript, but must NOT fire onUtteranceStart -- the skipped-interim safety rule above.
    act(() => first.emitFinal("record"));
    expect(onUtteranceStart).toHaveBeenCalledTimes(1);
    expect(onFinalResult).toHaveBeenCalledTimes(2);
    expect(onFinalResult).toHaveBeenLastCalledWith("record");
  });

  it("fires onSessionStart once per recognizer session -- the initial enable and every restart", () => {
    vi.useFakeTimers();
    const onSessionStart = vi.fn();
    FakeRecognition.instances = [];
    const { result } = renderHook(() =>
      useVoiceRecognition(vi.fn(), { recognitionFactory: () => new FakeRecognition(), isSupported: () => true, onSessionStart }),
    );
    act(() => result.current.toggle());
    expect(onSessionStart).toHaveBeenCalledTimes(1);
    const first = FakeRecognition.instances[0];
    act(() => first.emitEnd());
    act(() => {
      vi.runAllTimers();
    });
    expect(onSessionStart).toHaveBeenCalledTimes(2); // the restart is a new session too
  });

  it("treats a synchronous start() throw as a persistent failure, not an uncaught exception", () => {
    class ThrowingRecognition extends FakeRecognition {
      start(): void {
        throw new Error("boom");
      }
    }
    FakeRecognition.instances = [];
    const { result } = renderHook(() =>
      useVoiceRecognition(vi.fn(), { recognitionFactory: () => new ThrowingRecognition(), isSupported: () => true }),
    );
    expect(() => act(() => result.current.toggle())).not.toThrow();
    expect(result.current.status).toBe("error");
    expect(result.current.enabled).toBe(false);
  });

  it("a network error stops automatic restart and surfaces an actionable error, same as a permission error", () => {
    vi.useFakeTimers();
    const { result } = setup();
    act(() => result.current.toggle());
    const first = FakeRecognition.instances[0];
    act(() => first.emitError("network"));
    expect(result.current.status).toBe("error");
    expect(result.current.enabled).toBe(false);
    act(() => first.emitEnd());
    act(() => {
      vi.runAllTimers();
    });
    expect(FakeRecognition.instances).toHaveLength(1); // no restart after a persistent error
  });

  it("persistent listening survives many consecutive silence (no-speech) cycles -- it never disables on its own", () => {
    vi.useFakeTimers();
    const { result } = setup();
    act(() => result.current.toggle());
    for (let i = 0; i < 10; i += 1) {
      const current = FakeRecognition.instances[FakeRecognition.instances.length - 1];
      act(() => current.emitError("no-speech"));
      act(() => current.emitEnd());
      act(() => {
        vi.runAllTimers();
      });
    }
    expect(result.current.status).not.toBe("error");
    expect(result.current.enabled).toBe(true);
    expect(FakeRecognition.instances).toHaveLength(11);
  });

  it("restarts after a normal end while still enabled, using one scheduled restart", () => {
    vi.useFakeTimers();
    const { result } = setup();
    act(() => result.current.toggle());
    const first = FakeRecognition.instances[0];
    act(() => first.emitEnd());
    expect(FakeRecognition.instances).toHaveLength(1); // not yet -- restart is scheduled, not immediate
    act(() => {
      vi.runAllTimers();
    });
    expect(FakeRecognition.instances).toHaveLength(2);
  });

  it("does not restart after end once voice mode has been disabled", () => {
    vi.useFakeTimers();
    const { result } = setup();
    act(() => result.current.toggle()); // enable
    const first = FakeRecognition.instances[0];
    act(() => result.current.toggle()); // disable
    act(() => first.emitEnd());
    act(() => {
      vi.runAllTimers();
    });
    expect(FakeRecognition.instances).toHaveLength(1);
  });

  it("stops automatic restart and surfaces an actionable error on a permission error", () => {
    vi.useFakeTimers();
    const { result } = setup();
    act(() => result.current.toggle());
    const first = FakeRecognition.instances[0];
    act(() => first.emitError("not-allowed"));
    expect(result.current.status).toBe("error");
    expect(result.current.errorMessage).not.toBeNull();
    expect(result.current.enabled).toBe(false);
    act(() => first.emitEnd());
    act(() => {
      vi.runAllTimers();
    });
    expect(FakeRecognition.instances).toHaveLength(1); // no restart after a persistent error
  });

  it("allows restart after a transient error (e.g. no-speech)", () => {
    vi.useFakeTimers();
    const { result } = setup();
    act(() => result.current.toggle());
    const first = FakeRecognition.instances[0];
    act(() => first.emitError("no-speech"));
    expect(result.current.enabled).toBe(true); // not disabled by a transient error
    act(() => first.emitEnd());
    act(() => {
      vi.runAllTimers();
    });
    expect(FakeRecognition.instances).toHaveLength(2);
  });

  it("never overlaps recognizers: a stale instance's end event after a fresh start is ignored", () => {
    vi.useFakeTimers();
    const { result } = setup();
    act(() => result.current.toggle());
    const first = FakeRecognition.instances[0];
    act(() => result.current.toggle()); // disable (stops + supersedes session)
    act(() => result.current.toggle()); // re-enable -> a second, fresh instance
    expect(FakeRecognition.instances).toHaveLength(2);
    act(() => first.emitEnd()); // the stale first instance's end event
    act(() => {
      vi.runAllTimers();
    });
    expect(FakeRecognition.instances).toHaveLength(2); // no spurious third restart from the stale instance
  });

  it("stops recognition on unmount", () => {
    const { result, unmount } = setup();
    act(() => result.current.toggle());
    const first = FakeRecognition.instances[0];
    unmount();
    expect(first.stopped).toBe(true);
  });

  it("delivers only results at or after resultIndex, guarding against a repeated/overlapping range", () => {
    const { result, onFinalResult } = setup();
    act(() => result.current.toggle());
    const first = FakeRecognition.instances[0];
    act(() => {
      first.onresult?.({
        resultIndex: 0,
        results: [
          { isFinal: true, 0: { transcript: "record" } },
          { isFinal: true, 0: { transcript: "stop" } },
        ],
      });
    });
    expect(onFinalResult).toHaveBeenNthCalledWith(1, "record");
    expect(onFinalResult).toHaveBeenNthCalledWith(2, "stop");
    onFinalResult.mockClear();
    // A second event reporting an overlapping range (resultIndex back at 0) must not redeliver "record"/"stop".
    act(() => {
      first.onresult?.({
        resultIndex: 0,
        results: [
          { isFinal: true, 0: { transcript: "record" } },
          { isFinal: true, 0: { transcript: "stop" } },
          { isFinal: true, 0: { transcript: "discard" } },
        ],
      });
    });
    expect(onFinalResult).toHaveBeenCalledTimes(1);
    expect(onFinalResult).toHaveBeenCalledWith("discard");
  });
});
