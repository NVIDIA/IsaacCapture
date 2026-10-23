/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * RecorderContext.tsx - React context for XRInputRecorder state.
 *
 * Owns the recorder instance and React-visible state. Heavy per-frame work runs in
 * RecorderComponent via useFrame; this context provides actions and state
 * to descendant components.
 */

import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';

import { type Recording, type ReplayPacing, XRInputRecorder } from './xrInputRecorder';

export interface RecorderContextValue {
  recorder: XRInputRecorder;
  mode: 'idle' | 'recording' | 'replaying';
  savedRecording: Recording | null;
  recordedFrameCount: number;
  replayPacing: ReplayPacing;
  startRecord: () => void;
  stopRecord: () => void;
  startReplay: () => void;
  stopReplay: () => void;
  recordingInterrupted: boolean;
  onFrameState: () => void;
  setReplayPacing: (pacing: ReplayPacing) => void;
  onSaveRecording: () => void;
  onLoadRecording: () => void;
  onFrameRecord: (count: number) => void;
}

const RecorderContext = createContext<RecorderContextValue | null>(null);

/**
 * Surface load feedback on the static #loadRecordingStatus element that sits
 * beside the "Load Recording…" button in the settings panel (both live in
 * index.html, not the React tree). No-op if the element is absent.
 */
function setLoadStatus(message: string, type: 'success' | 'error'): void {
  const el = document.getElementById('loadRecordingStatus');
  if (!el) return;
  el.textContent = message;
  el.className = `load-recording-status show ${type}`;
}

export function RecorderProvider({ children }: { children: React.ReactNode }) {
  const recorder = useMemo(() => new XRInputRecorder(), []);
  useEffect(() => () => recorder.dispose(), [recorder]);
  const [mode, setMode] = useState<'idle' | 'recording' | 'replaying'>('idle');
  const [savedRecording, setSavedRecordingState] = useState<Recording | null>(null);
  const [recordedFrameCount, setRecordedFrameCount] = useState(0);
  const [replayPacing, setReplayPacingState] = useState<ReplayPacing>('time');
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const startRecord = useCallback(() => {
    if (recorder.mode !== 'idle') return;
    recorder.startRecording();
    setMode('recording');
    setRecordedFrameCount(0);
  }, [recorder]);

  const stopRecord = useCallback(() => {
    if (recorder.mode !== 'recording') return;
    recorder.stopRecording();
    const rec = recorder.getRecording();
    setSavedRecordingState(rec);
    setRecordedFrameCount(rec.frames.length);
    setMode('idle');
  }, [recorder]);

  const startReplay = useCallback(() => {
    if (recorder.mode !== 'idle' || !savedRecording) return;
    recorder.startReplay(savedRecording, true, replayPacing);
    setMode('replaying');
  }, [recorder, replayPacing, savedRecording]);

  const stopReplay = useCallback(() => {
    if (recorder.mode !== 'replaying') return;
    recorder.stopReplay();
    setMode('idle');
  }, [recorder]);

  const onFrameState = useCallback(() => {
    if (mode !== recorder.mode) {
      if (mode === 'recording') {
        setSavedRecordingState(recorder.getRecording());
        setRecordedFrameCount(recorder.recordedFrameCount);
      }
      setMode(recorder.mode);
    }
  }, [mode, recorder]);

  const setReplayPacing = useCallback((pacing: ReplayPacing) => {
    setReplayPacingState(pacing);
  }, []);

  const onSaveRecording = useCallback(() => {
    if (!savedRecording) return;
    const blob = new Blob([JSON.stringify(savedRecording)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'isaacteleop-input-recording.json';
    a.click();
    URL.revokeObjectURL(url);
  }, [savedRecording]);

  const acceptRecording = useCallback((r: Recording) => {
    setSavedRecordingState(r);
    setRecordedFrameCount(r.frames.length);
  }, []);

  const onFileChange = useCallback(
    async (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (!file) return;
      try {
        const recording = XRInputRecorder.importJSON(await file.text());
        acceptRecording(recording);
        setLoadStatus(
          `Loaded ${recording.frames.length} frame${recording.frames.length === 1 ? '' : 's'} from "${file.name}".`,
          'success'
        );
      } catch (error) {
        const detail = error instanceof Error ? error.message : String(error);
        console.error('[Recorder] Failed to load recording:', error);
        setLoadStatus(`Could not load "${file.name}": ${detail}`, 'error');
      } finally {
        event.target.value = '';
      }
    },
    [acceptRecording]
  );

  const onLoadRecording = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const onFrameRecord = useCallback((count: number) => {
    setRecordedFrameCount(count);
  }, []);

  const value: RecorderContextValue = {
    recorder,
    mode,
    savedRecording,
    recordedFrameCount,
    replayPacing,
    startRecord,
    stopRecord,
    startReplay,
    stopReplay,
    recordingInterrupted: recorder.recordingInterrupted,
    onFrameState,
    setReplayPacing,
    onSaveRecording,
    onLoadRecording,
    onFrameRecord,
  };

  return (
    <RecorderContext.Provider value={value}>
      {children}
      <input
        ref={fileInputRef}
        type="file"
        accept=".json,application/json"
        hidden
        onChange={onFileChange}
      />
    </RecorderContext.Provider>
  );
}

export function useRecorder(): RecorderContextValue {
  const ctx = useContext(RecorderContext);
  if (!ctx) throw new Error('useRecorder must be used inside RecorderProvider');
  return ctx;
}

/**
 * True when a plain letter is being typed rather than used as a shortcut.
 *
 * Shared with CloudXRUI's panel-reset key, which runs off the same listener:
 * one definition means the two cannot disagree about what counts as typing.
 */
export function isEditableKeyboardTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  if (target instanceof HTMLInputElement) return target.type !== 'file';
  return target instanceof HTMLTextAreaElement || target instanceof HTMLSelectElement;
}

interface TeleopKeyboardShortcutOptions {
  recordingEnabled: boolean;
  startTeleop?: () => void;
  teleopStartDisabled?: boolean;
  /** Reset Teleop. Only reachable as a key because its button lives in the XR canvas. */
  resetTeleop?: () => void;
}

/**
 * Build the keydown handler for the Teleop Play and recording controls exposed by the XR UI.
 *
 * Returns a handler rather than registering a listener: CloudXRUI already owns a
 * window keydown listener for the panel-reset key, and a second one would mean two
 * places deciding what a letter means. The caller reports whether the key was
 * consumed, so its own keys can be skipped when this one handles the event.
 */
export function useTeleopShortcutHandler({
  recordingEnabled,
  startTeleop,
  teleopStartDisabled = false,
  resetTeleop,
}: TeleopKeyboardShortcutOptions): (event: KeyboardEvent) => boolean {
  const {
    recorder,
    savedRecording,
    startRecord,
    stopRecord,
    startReplay,
    stopReplay,
    onSaveRecording,
  } = useRecorder();

  return useCallback(
    (event: KeyboardEvent): boolean => {
      if (!recordingEnabled && !startTeleop && !resetTeleop) return false;

      const key =
        event.code === 'KeyT'
          ? 't'
          : event.code === 'KeyC'
            ? 'c'
            : event.code === 'KeyP'
              ? 'p'
              : event.code === 'KeyS'
                ? 's'
                : event.code === 'KeyX'
                  ? 'x'
                  : event.key.toLowerCase();

      if (key === 't') {
        if (!startTeleop || teleopStartDisabled) return false;
        startTeleop();
      } else if (key === 'c') {
        if (!recordingEnabled) return false;
        if (recorder.mode === 'recording') {
          stopRecord();
        } else if (recorder.mode === 'idle') {
          startRecord();
        } else {
          return false;
        }
      } else if (key === 'p') {
        if (!recordingEnabled) return false;
        if (recorder.mode === 'replaying') {
          stopReplay();
        } else if (recorder.mode === 'idle' && savedRecording) {
          startReplay();
        } else {
          return false;
        }
      } else if (key === 's') {
        if (!recordingEnabled || recorder.mode !== 'idle' || !savedRecording) return false;
        onSaveRecording();
      } else if (key === 'x') {
        if (!resetTeleop) return false;
        resetTeleop();
      } else {
        return false;
      }

      event.preventDefault();
      return true;
    },
    [
      recorder,
      onSaveRecording,
      recordingEnabled,
      savedRecording,
      startRecord,
      startReplay,
      startTeleop,
      stopRecord,
      stopReplay,
      teleopStartDisabled,
      resetTeleop,
    ]
  );
}
