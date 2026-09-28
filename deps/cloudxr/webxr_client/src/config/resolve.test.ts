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

import { URL_PARAMS, type UrlParam } from './params';
import { readUrlParam, seedsFromParams } from './resolve';

const params = (query: string) => new URLSearchParams(query);
const seed = (query: string) => seedsFromParams(params(query));
const read = (query: string, key: string) => readUrlParam(params(query), key);

describe('URL_PARAMS', () => {
  it('has unique keys and unique element ids', () => {
    const keys = URL_PARAMS.map(p => p.key);
    const ids = URL_PARAMS.filter(p => p.elementId).map(p => p.elementId);
    expect(new Set(keys).size).toBe(keys.length);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it('still covers the params that were URL-settable before the registry', () => {
    const keys = new Set(URL_PARAMS.map(p => p.key));
    for (const k of [
      'serverIP',
      'port',
      'codec',
      'panelHiddenAtStart',
      'headless',
      'autoRefreshMode',
    ]) {
      expect(keys.has(k)).toBe(true);
    }
  });

  it('declares the direct transport/OOB params (no element binding)', () => {
    for (const k of [
      'turnServer',
      'turnUsername',
      'turnCredential',
      'iceRelayOnly',
      'oobEnable',
      'controlToken',
    ]) {
      const param = URL_PARAMS.find(p => p.key === k);
      expect(param).toBeDefined();
      expect(param!.elementId).toBeUndefined();
    }
  });
});

describe('seedsFromParams (form-backed params only)', () => {
  it('returns values for present form params keyed by key', () => {
    const seeds = seed('serverIP=10.0.0.2&port=49100&codec=av1');
    expect(seeds.get('serverIP')).toBe('10.0.0.2');
    expect(seeds.get('port')).toBe('49100');
    expect(seeds.get('codec')).toBe('av1');
  });

  it('omits params that are absent', () => {
    const seeds = seed('serverIP=10.0.0.2');
    expect(seeds.has('port')).toBe(false);
    expect(seeds.size).toBe(1);
  });

  it('never seeds direct (non-form) params even when present', () => {
    expect(seed('turnServer=turn:host&controlToken=secret&oobEnable=1').size).toBe(0);
  });

  it('rejects values that fail validation', () => {
    expect(seed('codec=mpeg2').has('codec')).toBe(false);
    expect(seed('headless=banana').has('headless')).toBe(false);
    expect(seed('autoRefreshMode=sometimes').has('autoRefreshMode')).toBe(false);
    expect(seed('port=notanumber').has('port')).toBe(false);
    expect(seed('immersiveMode=hologram').has('immersiveMode')).toBe(false);
  });

  it('accepts valid enum, boolean and numeric values', () => {
    expect(seed('headless=true').get('headless')).toBe('true');
    expect(seed('immersiveMode=vr').get('immersiveMode')).toBe('vr');
    expect(seed('referenceSpace=local-floor').get('referenceSpace')).toBe('local-floor');
    expect(seed('posePredictionFactor=0.5').get('posePredictionFactor')).toBe('0.5');
    expect(seed('xrOffsetX=-12').get('xrOffsetX')).toBe('-12');
  });

  it('exposes every form-backed param as URL-settable', () => {
    const formParams = URL_PARAMS.filter(p => p.elementId);
    const query = formParams
      .map(p => `${p.url ?? p.key}=${encodeURIComponent(sampleValid(p.key))}`)
      .join('&');
    expect(seed(query).size).toBe(formParams.length);
  });
});

describe('readUrlParam (direct params)', () => {
  it('reads a present param value', () => {
    expect(read('turnServer=turn:127.0.0.1:3478', 'turnServer')).toBe('turn:127.0.0.1:3478');
    expect(read('controlToken=abc123', 'controlToken')).toBe('abc123');
    expect(read('iceRelayOnly=1', 'iceRelayOnly')).toBe('1');
  });

  it('returns null for absent params and unknown keys', () => {
    expect(read('oobEnable=1', 'turnServer')).toBeNull();
    expect(read('totallyUnknown=1', 'totallyUnknown')).toBeNull();
  });

  it('applies validation to validated params', () => {
    expect(read('codec=av1', 'codec')).toBe('av1');
    expect(read('codec=mpeg2', 'codec')).toBeNull();
  });
});

describe('url alias (param.url overrides param.key)', () => {
  const aliased: UrlParam[] = [{ key: 'serverIP', url: 'ip', elementId: 'serverIpInput' }];

  it('resolves by the url alias, not the key', () => {
    expect(seedsFromParams(params('ip=10.0.0.9'), aliased).get('serverIP')).toBe('10.0.0.9');
    expect(readUrlParam(params('ip=10.0.0.9'), 'serverIP', aliased)).toBe('10.0.0.9');
  });

  it('does not match the key when an alias is set', () => {
    expect(seedsFromParams(params('serverIP=10.0.0.9'), aliased).has('serverIP')).toBe(false);
    expect(readUrlParam(params('serverIP=10.0.0.9'), 'serverIP', aliased)).toBeNull();
  });
});

/** A representative valid value per form param, for the coverage test above. */
function sampleValid(key: string): string {
  const numeric = new Set([
    'port',
    'deviceFrameRate',
    'maxStreamingBitrateMbps',
    'perEyeWidth',
    'perEyeHeight',
    'reprojectionGridCols',
    'reprojectionGridRows',
    'posePredictionFactor',
    'xrOffsetX',
    'xrOffsetY',
    'xrOffsetZ',
    'mediaPort',
    'streamTestDurationSeconds',
    'reconnectMaxAttempts',
    'reconnectDelayMs',
    'controlPanelDistance',
    'controlPanelHeight',
    'controlPanelAngleDegrees',
  ]);
  if (numeric.has(key)) return '1';
  const enums: Record<string, string> = {
    serverType: 'manual',
    codec: 'av1',
    immersiveMode: 'ar',
    referenceSpace: 'auto',
    controlPanelPosition: 'center',
    controllerModelVisibility: 'show',
    showTraceInXR: 'false',
    showRecordingControls: 'false',
    replayPacing: 'time',
    autoRefreshMode: 'clean',
    enablePoseSmoothing: 'true',
    enableTexSubImage2D: 'true',
    useQuestColorWorkaround: 'true',
    panelHiddenAtStart: 'true',
    headless: 'true',
    streamTestMode: 'warn',
    reconnectEnabled: 'true',
    controlPanelTrackHeadset: 'true',
  };
  return enums[key] ?? 'x';
}
