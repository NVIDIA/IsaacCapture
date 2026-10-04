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

/** CloudXR's input-source events must describe the same sources as its tracking frame. */
export class XRReplaySession {
  private _sources: XRInputSource[] | null = null;
  private readonly _events = new EventTarget();
  readonly session: XRSession;

  constructor(readonly nativeSession: XRSession) {
    this.session = new Proxy(nativeSession, {
      get: (target, property) => {
        if (property === 'inputSources') return this._sources ?? target.inputSources;
        if (property === 'addEventListener' || property === 'removeEventListener') {
          return (
            type: string,
            listener: EventListenerOrEventListenerObject,
            options?: boolean | AddEventListenerOptions
          ) => {
            const owner = type === 'inputsourceschange' ? this._events : target;
            owner[property](type, listener, options);
          };
        }
        const value = Reflect.get(target, property, target);
        return typeof value === 'function' ? value.bind(target) : value;
      },
    });
    nativeSession.addEventListener('inputsourceschange', this._onNativeSourcesChange);
  }

  setInputSources(sources: XRInputSource[] | null): void {
    const previous = Array.from(this._sources ?? this.nativeSession.inputSources);
    this._sources = sources;
    const next = Array.from(sources ?? this.nativeSession.inputSources);
    const added = next.filter(source => !previous.includes(source));
    const removed = previous.filter(source => !next.includes(source));
    if (added.length || removed.length) this._dispatch(added, removed);
  }

  dispose(): void {
    this.nativeSession.removeEventListener('inputsourceschange', this._onNativeSourcesChange);
  }

  private _onNativeSourcesChange = (event: XRInputSourcesChangeEvent): void => {
    // Recorded device presence is independent of live tracking and controller disconnects.
    if (this._sources === null) this._dispatch(Array.from(event.added), Array.from(event.removed));
  };

  private _dispatch(added: XRInputSource[], removed: XRInputSource[]): void {
    this._events.dispatchEvent(
      Object.assign(new Event('inputsourceschange'), { session: this.session, added, removed })
    );
  }
}
