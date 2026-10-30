/**
 * Owns the browser SpeechRecognition lifecycle only: starting/stopping one
 * recognizer instance, restarting it after a normal end while voice mode
 * stays enabled, and delivering each FINAL result's transcript to the
 * caller. It has no knowledge of the recording-command grammar or any
 * recording state -- `onFinalResult` is the one hand-off point; everything
 * else (parsing a phrase, deciding which existing handler to call) is the
 * caller's job (see RecordingPanel.tsx). `onUtteranceStart` (optional) fires
 * once per utterance, at the browser's first INTERIM observation of it --
 * not literal speech onset. With `continuous=true`, Chrome is reported (not
 * independently verified here) to fire `onspeechstart` only once per
 * recognizer SESSION, not once per utterance, so it can't drive this;
 * `onUtteranceStart` instead fires when an event reports an
 * interim result and no interim was already pending for the utterance in
 * progress, and stays pending (suppressing a re-fire) until that utterance's
 * final is delivered. KNOWN LIMITATION, not fixed here: if the browser
 * withdraws a pending interim without ever delivering a final for it,
 * "pending" is never reset until the recognizer session restarts -- real
 * browser acceptance of this case is unverified. The spec allows more than one interim result index to
 * arrive before an utterance's final (e.g. a stable prefix and an unstable
 * tail reported as separate entries is one way that can happen, though
 * nothing here depends on that specific shape); since there is no reliable
 * way to tell "another interim index for the SAME utterance" apart from "a
 * genuinely new utterance" from the event stream alone, every interim index
 * seen while one is already pending is treated as the former -- this errs
 * toward never leaving a real command's target binding "invalid" or
 * starving the next utterance's own first interim, at the cost of
 * potentially coalescing two truly back-to-back utterances if the browser
 * ever reports multiple in flight at once. A result that arrives
 * already final with no interim first never fires it, by design: the caller
 * uses this to bind a command to whatever recording state is live AT THAT
 * MOMENT, so a delayed final result can't silently retarget onto a
 * DIFFERENT batch/episode that became current while the result was still
 * in flight; binding at final-delivery time for a skipped-interim result
 * would defeat that guarantee outright, so it's deliberately left unbound
 * (the caller's existing "no binding was ever captured" path handles it).
 *
 * `recognitionFactory` defaults to the real browser constructor but is
 * overridable for tests -- "a small fake recognition event source," not a
 * mock-browser framework.
 */

import { useCallback, useEffect, useRef, useState } from "react";

/** The one slice of the real SpeechRecognition API this hook actually uses. */
export interface SpeechRecognitionLike {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  start(): void;
  stop(): void;
  onstart: (() => void) | null;
  onspeechstart: (() => void) | null;
  onresult: ((event: SpeechRecognitionResultEventLike) => void) | null;
  onerror: ((event: SpeechRecognitionErrorEventLike) => void) | null;
  onend: (() => void) | null;
}

export interface SpeechRecognitionResultEventLike {
  resultIndex: number;
  results: ArrayLike<{ isFinal: boolean; 0: { transcript: string } }>;
}

export interface SpeechRecognitionErrorEventLike {
  error: string;
}

declare global {
  interface Window {
    SpeechRecognition?: new () => SpeechRecognitionLike;
    webkitSpeechRecognition?: new () => SpeechRecognitionLike;
  }
}

/** The real browser constructor, or `undefined` if this browser has neither
 * the standard nor the vendor-prefixed one. */
function browserSpeechRecognitionCtor(): (new () => SpeechRecognitionLike) | undefined {
  if (typeof window === "undefined") return undefined;
  return window.SpeechRecognition ?? window.webkitSpeechRecognition;
}

/** Only called when `isSupported()` has already confirmed a constructor
 * exists -- see the `enabled && supported` guard around `startRecognition`'s
 * one call site below. */
function defaultRecognitionFactory(): SpeechRecognitionLike {
  const Ctor = browserSpeechRecognitionCtor();
  if (Ctor === undefined) throw new Error("SpeechRecognition is not supported in this browser");
  return new Ctor();
}

/** Errors the real API reports that mean "don't keep retrying automatically" --
 * a permission denial, an unreachable recognition service, or an unsupported
 * configuration won't fix itself by restarting. Everything else -- most
 * notably "no-speech", which fires on every ordinary silence timeout -- must
 * keep voice mode listening: persistent listening has to survive a quiet
 * station, not stop after a few seconds of nobody talking. */
const PERSISTENT_ERRORS = new Set(["not-allowed", "service-not-allowed", "audio-capture", "network", "language-not-supported"]);

/** A short, nonzero delay before each restart -- long enough that a tight
 * restart loop can't spin the CPU, short enough that it's imperceptible for
 * the normal (healthy) restart-after-silence case. Not a backoff schedule --
 * always the same delay. */
const RESTART_DELAY_MS = 300;

export type VoiceRecognitionStatus = "unsupported" | "disabled" | "starting" | "listening" | "error";

export interface UseVoiceRecognitionResult {
  status: VoiceRecognitionStatus;
  /** Set once voice mode is explicitly enabled; cleared on disable. Distinct
   * from `status === "unsupported"`, which is a browser capability fact. */
  enabled: boolean;
  toggle: () => void;
  errorMessage: string | null;
}

export function useVoiceRecognition(
  onFinalResult: (text: string) => void,
  options: {
    /** Builds one recognizer instance -- called once per start/restart. */
    recognitionFactory?: () => SpeechRecognitionLike;
    /** Whether this browser can be asked to recognize speech at all --
     * checked statically (constructor presence), never by constructing and
     * discarding an instance, which some browsers treat as a real
     * permission-triggering action. */
    isSupported?: () => boolean;
    /** Fires once per utterance, at its first interim observation -- see
     * the module docstring. */
    onUtteranceStart?: () => void;
    /** Fires once per recognizer SESSION, right before ``start()`` is called
     * (covers both the initial enable and every restart-after-end) -- the
     * caller uses this to clear any transient per-utterance state it was
     * keeping, so an utterance that began in a now-superseded session
     * (e.g. Space disabled it before a final result arrived) can never
     * poison the first command of the next session. */
    onSessionStart?: () => void;
  } = {},
): UseVoiceRecognitionResult {
  const recognitionFactory = options.recognitionFactory ?? defaultRecognitionFactory;
  const isSupported = options.isSupported ?? (() => browserSpeechRecognitionCtor() !== undefined);
  const [enabled, setEnabled] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  // Whether the CURRENT recognizer instance has actually fired its own
  // onstart -- `enabled` alone is not proof the microphone is listening
  // yet (there is a real gap between calling start() and the browser
  // actually beginning to listen, and another gap during each restart).
  const [recognizing, setRecognizing] = useState(false);
  const supported = isSupported();

  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  // Bumped every time a new recognizer instance is actually started -- an
  // `onend`/`onerror` from a PREVIOUS instance (already superseded by a
  // disable or a fresh start) must never schedule a restart or overwrite
  // state for the current one.
  const sessionIdRef = useRef(0);
  const restartTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const onFinalResultRef = useRef(onFinalResult);
  useEffect(() => {
    onFinalResultRef.current = onFinalResult;
  }, [onFinalResult]);
  const onUtteranceStartRef = useRef(options.onUtteranceStart);
  useEffect(() => {
    onUtteranceStartRef.current = options.onUtteranceStart;
  }, [options.onUtteranceStart]);
  const onSessionStartRef = useRef(options.onSessionStart);
  useEffect(() => {
    onSessionStartRef.current = options.onSessionStart;
  }, [options.onSessionStart]);

  const stopRecognition = useCallback(() => {
    sessionIdRef.current += 1; // supersedes any in-flight instance's callbacks
    if (restartTimeoutRef.current !== null) {
      clearTimeout(restartTimeoutRef.current);
      restartTimeoutRef.current = null;
    }
    setRecognizing(false);
    if (recognitionRef.current !== null) {
      const r = recognitionRef.current;
      recognitionRef.current = null;
      r.onstart = null;
      r.onspeechstart = null;
      r.onresult = null;
      r.onerror = null;
      r.onend = null;
      r.stop();
    }
  }, []);

  /** Treats a startup/session failure as a hard stop: shows the message,
   * disables voice mode, and never schedules a restart. Shared by a
   * synchronous construct/start() throw and a real `not-allowed`-class
   * error event. */
  const failPersistently = useCallback((message: string) => {
    setErrorMessage(message);
    setEnabled(false);
  }, []);

  const startRecognition = useCallback(() => {
    let recognition: SpeechRecognitionLike;
    try {
      recognition = recognitionFactory();
    } catch (err) {
      failPersistently(`Voice recognition could not start: ${err instanceof Error ? err.message : String(err)}`);
      return;
    }
    sessionIdRef.current += 1;
    const mySessionId = sessionIdRef.current;
    onSessionStartRef.current?.();
    recognition.continuous = true;
    // interimResults=true is what makes onresult fire once per result INDEX as soon as the
    // browser first reports it (interim), not just once per utterance's eventual final text --
    // does not rely on onspeechstart alone being per-utterance: with continuous=true, Chrome is
    // reported (not independently verified here) to fire it once per recognizer SESSION, not
    // once per utterance, which would bind every command after the first onto stale state. See
    // the pending-interim onUtteranceStart logic below instead.
    recognition.interimResults = true;
    recognition.onstart = () => {
      if (sessionIdRef.current !== mySessionId) return;
      setRecognizing(true);
    };
    // Scoped to THIS recognizer instance (reset on every start/restart, never shared across
    // sessions). processedUpTo is a high-water mark so each index's eventual final fires
    // onFinalResult exactly once. pendingInterim tracks the CURRENT in-progress utterance, not a
    // result index -- see the module docstring for why: an additional interim index seen while
    // one is pending is treated as the SAME utterance, not a new one, so onUtteranceStart fires
    // at most once per utterance -- see the loop below.
    let processedUpTo = -1;
    let pendingInterim = false;
    recognition.onresult = (event) => {
      if (sessionIdRef.current !== mySessionId) return;
      const start = Math.max(event.resultIndex, processedUpTo + 1);
      let sawInterimThisEvent = false;
      for (let i = start; i < event.results.length; i += 1) {
        const result = event.results[i];
        if (result.isFinal) {
          // A final always clears "pending", even if this exact index never reported an interim
          // first (see the reject/repeat note below): the utterance it closes out is over either
          // way, and the NEXT interim must be treated as a fresh utterance.
          if (i > processedUpTo) {
            processedUpTo = i;
            pendingInterim = false;
            onFinalResultRef.current(result[0].transcript);
          }
        } else {
          sawInterimThisEvent = true;
        }
      }
      // Bind ONLY on an interim observation, never on a result that arrives already final with
      // no interim first: a target-bound command (stop/discard) whose final is delayed until
      // AFTER the live batch/episode changes must not silently bind to the NEW current target at
      // delivery time -- it must fall through to the caller's existing "no binding was ever
      // captured" reject/repeat path instead, exactly like a genuinely missed onUtteranceStart.
      // Finals are processed above BEFORE this check so a final that closes out the pending
      // utterance and a fresh interim for the NEXT utterance in the same event both fire
      // correctly; a later interim for the SAME still-pending utterance must never re-fire.
      if (sawInterimThisEvent && !pendingInterim) {
        pendingInterim = true;
        onUtteranceStartRef.current?.();
      }
    };
    recognition.onerror = (event) => {
      if (sessionIdRef.current !== mySessionId) return;
      if (PERSISTENT_ERRORS.has(event.error)) {
        failPersistently(`Voice recognition error: ${event.error}. Enable voice again after fixing it.`);
        return;
      }
      // Transient (e.g. "no-speech", every ordinary silence timeout): the
      // browser's own `onend` still fires after this -- the restart below
      // handles it, unconditionally, so a quiet station keeps listening.
    };
    recognition.onend = () => {
      if (sessionIdRef.current !== mySessionId) return;
      recognitionRef.current = null;
      setRecognizing(false);
      if (!enabledRef.current) return;
      // One short scheduled restart -- never an immediate re-entrant
      // start() from inside the same end event, and never more than one
      // pending restart at a time (stopRecognition clears it above).
      restartTimeoutRef.current = setTimeout(() => {
        restartTimeoutRef.current = null;
        if (sessionIdRef.current === mySessionId && enabledRef.current) startRecognitionRef.current();
      }, RESTART_DELAY_MS);
    };
    recognitionRef.current = recognition;
    try {
      recognition.start();
    } catch (err) {
      recognitionRef.current = null;
      failPersistently(`Voice recognition could not start: ${err instanceof Error ? err.message : String(err)}`);
    }
  }, [recognitionFactory, failPersistently]);

  // Stable refs so the closures above (captured once per recognizer
  // instance) always see the LATEST enabled flag / restart function, not a
  // stale one from whenever that particular instance started.
  const enabledRef = useRef(enabled);
  useEffect(() => {
    enabledRef.current = enabled;
  }, [enabled]);
  const startRecognitionRef = useRef(startRecognition);
  useEffect(() => {
    startRecognitionRef.current = startRecognition;
  }, [startRecognition]);

  useEffect(() => {
    if (enabled && supported) {
      setErrorMessage(null);
      startRecognition();
    } else {
      stopRecognition();
    }
    return () => {
      stopRecognition();
    };
    // Intentionally only keyed on `enabled`/`supported`: starting a new recognizer for every
    // render of `startRecognition`/`stopRecognition` (which both change identity when
    // `recognitionFactory` is an inline default) would restart recognition on unrelated re-renders.
  }, [enabled, supported]);

  const toggle = useCallback(() => {
    setEnabled((prev) => !prev);
  }, []);

  // errorMessage is checked BEFORE `enabled`: a persistent-error disable
  // sets both errorMessage and enabled=false together, and the resulting
  // status must read "error", not plain "disabled" -- the two are
  // different facts for the UI to show. "starting" covers the real gap
  // between `enabled` becoming true and the recognizer's own onstart
  // actually firing (including during each restart), so "listening" is
  // never shown before the microphone is truly active.
  const status: VoiceRecognitionStatus = !supported
    ? "unsupported"
    : errorMessage !== null
      ? "error"
      : !enabled
        ? "disabled"
        : recognizing
          ? "listening"
          : "starting";

  return { status, enabled: enabled && supported, toggle, errorMessage };
}
