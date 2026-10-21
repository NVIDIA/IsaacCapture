/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import { useFrame } from '@react-three/fiber';

import {
  controllerSweep,
  type Hand,
  type InputTrace,
  type Quaternion,
  SyntheticXRController,
  type SyntheticDevice,
  parseTrace,
} from './controller';
import './panel.css';

declare global {
  interface Window {
    syntheticXR?: SyntheticXRController;
  }
}

function saveTrace(trace: InputTrace): void {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(trace, null, 2)], { type: 'application/json' })
  );
  const a = document.createElement('a');
  a.href = url;
  a.download = 'synthetic-xr-input.json';
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function SyntheticXRPanel({
  ready,
  connected,
  replaying,
  onStartTeleop,
  startingOrRunning,
}: {
  ready: boolean;
  connected: boolean;
  replaying: boolean;
  onStartTeleop: () => Promise<void>;
  startingOrRunning: boolean;
}) {
  const [api, setApi] = useState<SyntheticXRController>();
  const [hand, setHand] = useState<Hand>('right');
  const [, refresh] = useState(0);
  const [error, setError] = useState('');
  const [trace, setTrace] = useState<InputTrace>();
  const enabled = new URLSearchParams(location.search).get('syntheticXR') === '1';

  useEffect(() => {
    if (!enabled || !ready || !window.xrDevice) return;
    const controller = new SyntheticXRController(window.xrDevice as unknown as SyntheticDevice);
    window.syntheticXR = controller;
    const unsubscribe = controller.subscribe(() => refresh(n => n + 1));
    setApi(controller);
    const stopOnHidden = () => {
      if (document.hidden) controller.stop();
    };
    document.addEventListener('visibilitychange', stopOnHidden);
    return () => {
      document.removeEventListener('visibilitychange', stopOnHidden);
      unsubscribe();
      controller.dispose();
      if (window.syntheticXR === controller) delete window.syntheticXR;
    };
  }, [enabled, ready]);

  useEffect(() => {
    if (!connected) api?.stop();
    api?.setSuspended(replaying);
  }, [api, connected, replaying]);

  if (!enabled)
    return ready && window.xrDevice
      ? createPortal(
          <aside className="synthetic-xr">
            <button
              onClick={() => {
                const url = new URL(location.href);
                url.searchParams.set('syntheticXR', '1');
                location.assign(url.href);
              }}
            >
              Open Synthetic XR controls
            </button>
          </aside>,
          document.body
        )
      : null;
  const act = (fn: () => unknown) => {
    try {
      fn();
      setError('');
    } catch (e) {
      setError(String(e));
    }
  };
  const state = api?.getState()[hand];
  const busy = !api || api.isPlaying || replaying;
  const rotate = (axis: number, delta: number) => {
    if (!api || !state) return;
    const r = [0, 0, 0, Math.cos(delta / 2)];
    r[axis] = Math.sin(delta / 2);
    const [x, y, z, w] = state.quaternion;
    const [a, b, c, d] = r;
    const q: Quaternion = [
      d * x + a * w + b * z - c * y,
      d * y - a * z + b * w + c * x,
      d * z + a * y - b * x + c * w,
      d * w - a * x - b * y - c * z,
    ];
    api.setController(hand, { quaternion: q });
  };
  return createPortal(
    <section className="synthetic-xr" aria-label="Synthetic XR Teleoperation">
      <details open>
        <summary>
          Synthetic XR Teleoperation <small>Preview</small>
        </summary>
        {!api ? (
          <p>
            {ready
              ? 'Native XR is active. Desktop controls require the IWER fallback.'
              : 'Waiting for the desktop XR runtime…'}
          </p>
        ) : (
          <>
            <p>
              {connected ? 'Connected' : 'Connect the WebXR client to send input'} · {api.status}
            </p>
            <button
              id="synthetic-xr-start-teleop"
              disabled={!connected || startingOrRunning || replaying}
              onClick={() => {
                void onStartTeleop().catch(e => setError(String(e)));
              }}
            >
              {startingOrRunning ? 'Teleoperation started' : 'Start teleoperation'}
            </button>
            <label>
              Controller{' '}
              <select value={hand} onChange={e => setHand(e.target.value as Hand)}>
                <option value="left">Left</option>
                <option value="right">Right</option>
              </select>
            </label>
            <fieldset disabled={busy}>
              <legend>Pose in metres · IWER local space</legend>
              {(['X', 'Y', 'Z'] as const).map((name, index) => (
                <label key={name}>
                  {name}
                  <input
                    aria-label={`${name} position`}
                    type="number"
                    step="0.02"
                    min="-10"
                    max="10"
                    value={state?.position[index] ?? 0}
                    onChange={e =>
                      act(() => {
                        const position = [...state!.position] as [number, number, number];
                        position[index] = Number(e.target.value);
                        api.setController(hand, { position });
                      })
                    }
                  />
                  <button onClick={() => act(() => rotate(index, -Math.PI / 18))}>−10°</button>
                  <button onClick={() => act(() => rotate(index, Math.PI / 18))}>+10°</button>
                </label>
              ))}
              <label>
                <input
                  type="checkbox"
                  checked={Boolean(state?.grip)}
                  onChange={e =>
                    act(() => api.setController(hand, { grip: e.target.checked ? 1 : 0 }))
                  }
                />
                Grip / clutch
              </label>
              <label>
                Trigger{' '}
                <input
                  aria-label="Trigger"
                  type="range"
                  min="0"
                  max="1"
                  step="0.01"
                  value={state?.trigger ?? 0}
                  onChange={e =>
                    act(() => api.setController(hand, { trigger: Number(e.target.value) }))
                  }
                />
              </label>
              <button onClick={() => act(() => api.reset())}>Reset virtual poses</button>
            </fieldset>
            <div className="synthetic-xr-actions">
              <button
                disabled={busy || api.isRecording || !connected}
                onClick={() =>
                  act(() => {
                    void api.play(controllerSweep()).catch(e => setError(String(e)));
                  })
                }
              >
                Run controller sweep
              </button>
              <button
                disabled={busy}
                onClick={() =>
                  act(() => {
                    if (api.isRecording) {
                      const saved = api.finishRecording();
                      setTrace(saved);
                      saveTrace(saved);
                    } else api.record();
                  })
                }
              >
                {api.isRecording ? 'Save input trace' : 'Record input'}
              </button>
              <label>
                Load trace{' '}
                <input
                  aria-label="Load synthetic trace"
                  type="file"
                  accept=".json"
                  disabled={busy || api.isRecording}
                  onChange={async e => {
                    const file = e.target.files?.[0];
                    if (!file) return;
                    if (file.size > 32 * 1024 * 1024) {
                      setError('Trace file exceeds 32 MB.');
                      return;
                    }
                    try {
                      setTrace(parseTrace(JSON.parse(await file.text())));
                      setError('');
                    } catch (err) {
                      setError(String(err));
                    }
                  }}
                />
              </label>
              <button
                disabled={busy || !trace || api.isRecording || !connected}
                onClick={() =>
                  act(() => {
                    void api.play(trace).catch(e => setError(String(e)));
                  })
                }
              >
                Replay once
              </button>
              <button className="synthetic-xr-stop" onClick={() => act(() => api.stop())}>
                Stop input
              </button>
            </div>
            {replaying && <p>Stop browser recording replay to use synthetic controllers.</p>}
            <p className="synthetic-xr-note">
              Virtual controller input through IWER. Stop releases grip and trigger; use the
              application’s stop control to stop the robot task.
            </p>
          </>
        )}
        {error && <p role="alert">{error}</p>}
      </details>
    </section>,
    document.body
  );
}

export function SyntheticXRFrameObserver() {
  useFrame((state, _delta, frame) => {
    const reference = state.gl.xr.getReferenceSpace();
    if (frame && reference) window.syntheticXR?.observeFrame(frame, reference);
  });
  return null;
}
