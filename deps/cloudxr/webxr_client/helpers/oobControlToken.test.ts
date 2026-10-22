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
 *
 * @jest-environment jsdom
 */

type TokenModule = typeof import('./oobControlToken');

let captureOobControlToken: TokenModule['captureOobControlToken'];
let getOobControlToken: TokenModule['getOobControlToken'];

const HUB = 'oobEnable=1&serverIP=10.0.0.5&port=48322';

function visit(url: string): void {
  window.history.replaceState({ keep: 'me' }, '', url);
}

function hubQuery(): URLSearchParams {
  return new URLSearchParams(HUB);
}

// Fresh module per test: the in-memory fallback map is module state.
beforeEach(async () => {
  window.sessionStorage.clear();
  jest.resetModules();
  ({ captureOobControlToken, getOobControlToken } = await import('./oobControlToken'));
});

describe('captureOobControlToken', () => {
  it('lifts a fragment token, strips it, and keeps the route and history state', () => {
    visit(`/client/?${HUB}#/real/dexmate?controlToken=abc-123_XYZ`);
    captureOobControlToken();
    expect(window.location.hash).toBe('#/real/dexmate');
    expect(window.location.search).toBe(`?${HUB}`);
    expect(window.history.state).toEqual({ keep: 'me' });
    expect(getOobControlToken(hubQuery())).toBe('abc-123_XYZ');
  });

  it('handles a fragment with no route', () => {
    visit(`/client/?${HUB}#?controlToken=tok`);
    captureOobControlToken();
    expect(window.location.hash).toBe('');
    expect(getOobControlToken(hubQuery())).toBe('tok');
  });

  it('round-trips a URL-encoded token', () => {
    visit(`/client/?${HUB}#/real?controlToken=${encodeURIComponent('a&b=c?d#e')}`);
    captureOobControlToken();
    expect(getOobControlToken(hubQuery())).toBe('a&b=c?d#e');
  });

  it('reads a legacy query token and strips it, keeping other params', () => {
    visit(`/client/?${HUB}&controlToken=legacy#/real`);
    captureOobControlToken();
    expect(window.location.search).toBe(`?${HUB}`);
    expect(window.location.hash).toBe('#/real');
    expect(getOobControlToken(hubQuery())).toBe('legacy');
  });

  it('prefers the fragment token when both are present', () => {
    visit(`/client/?${HUB}&controlToken=old#/real?controlToken=new`);
    captureOobControlToken();
    expect(getOobControlToken(hubQuery())).toBe('new');
    expect(window.location.search).not.toContain('controlToken');
    expect(window.location.hash).toBe('#/real');
  });

  it('keeps the stored token across a reload that carries none', () => {
    visit(`/client/?${HUB}#/real?controlToken=tok`);
    captureOobControlToken();
    visit(`/client/?${HUB}#/real`);
    captureOobControlToken();
    expect(getOobControlToken(hubQuery())).toBe('tok');
  });

  it('replaces a stale stored token with a newly supplied one', () => {
    visit(`/client/?${HUB}#/real?controlToken=stale`);
    captureOobControlToken();
    visit(`/client/?${HUB}#/real?controlToken=fresh`);
    captureOobControlToken();
    expect(getOobControlToken(hubQuery())).toBe('fresh');
  });

  it('does not reuse one hub endpoint token for another', () => {
    visit(`/client/?${HUB}#/real?controlToken=tok`);
    captureOobControlToken();
    expect(getOobControlToken(new URLSearchParams('serverIP=10.0.0.6&port=48322'))).toBeUndefined();
    expect(getOobControlToken(new URLSearchParams('serverIP=10.0.0.5&port=49000'))).toBeUndefined();
  });

  it('leaves the URL alone when there is no token', () => {
    visit(`/client/?${HUB}#/real`);
    captureOobControlToken();
    expect(window.location.href).toContain(`?${HUB}#/real`);
    expect(getOobControlToken(hubQuery())).toBeUndefined();
  });

  it('prefers the fresh in-memory token over a stale stored one when the write is denied', () => {
    window.sessionStorage.setItem('oobControlToken:10.0.0.5:48322', 'stale');
    const spy = jest.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('quota');
    });
    const warn = jest.spyOn(console, 'warn').mockImplementation(() => undefined);
    visit(`/client/?${HUB}#/real?controlToken=fresh`);
    captureOobControlToken();
    expect(getOobControlToken(hubQuery())).toBe('fresh');
    spy.mockRestore();
    warn.mockRestore();
  });

  it('still serves the token for this page load when sessionStorage is denied', () => {
    const spy = jest.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('denied');
    });
    const warn = jest.spyOn(console, 'warn').mockImplementation(() => undefined);
    visit(`/client/?${HUB}#/real?controlToken=memonly`);
    captureOobControlToken();
    expect(getOobControlToken(hubQuery())).toBe('memonly');
    expect(warn).toHaveBeenCalled();
    spy.mockRestore();
    warn.mockRestore();
  });
});
