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

import {
  formatSystemNotice,
  formatSystemNoticeBody,
  isSystemNoticeMessage,
  SystemNotice,
} from './serverMessages';

/** A well-formed notice, matching what the Isaac Lab host sends. */
const validNotice = (): SystemNotice => ({
  level: 'warning',
  title: 'Workstation below recommended spec',
  summary: 'Teleoperation may run below the 45 FPS target.',
  items: [
    {
      name: 'CPU governor',
      actual: 'powersave',
      required: 'performance',
      detail: 'sudo cpupower frequency-set -g performance',
    },
  ],
  doc_url: 'https://example.invalid/docs',
});

const validMessage = () => ({ type: 'system_notice', message: validNotice() });

describe('isSystemNoticeMessage', () => {
  it('accepts a well-formed notice', () => {
    expect(isSystemNoticeMessage(validMessage())).toBe(true);
  });

  it('accepts a notice with no items and no optional fields', () => {
    expect(
      isSystemNoticeMessage({
        type: 'system_notice',
        message: { level: 'info', title: 't', summary: 's', items: [] },
      })
    ).toBe(true);
  });

  it.each([
    ['a non-object', 42],
    ['null', null],
    ['undefined', undefined],
    ['an array', []],
  ])('rejects %s', (_label, value) => {
    expect(isSystemNoticeMessage(value)).toBe(false);
  });

  it.each([
    ['another message type', { type: 'teleop_command', message: validNotice() }],
    ['a missing type', { message: validNotice() }],
    ['a missing message body', { type: 'system_notice' }],
    ['a null message body', { type: 'system_notice', message: null }],
    ['a non-object message body', { type: 'system_notice', message: 'nope' }],
  ])('rejects %s', (_label, value) => {
    expect(isSystemNoticeMessage(value)).toBe(false);
  });

  it.each([
    ['an unknown level', { level: 'catastrophe' }],
    ['a missing level', { level: undefined }],
    ['a non-string title', { title: 7 }],
    ['a missing title', { title: undefined }],
    ['a non-string summary', { summary: {} }],
    ['a non-array items', { items: 'none' }],
    ['a missing items', { items: undefined }],
    ['a non-string doc_url', { doc_url: 3 }],
  ])('rejects %s', (_label, override) => {
    expect(
      isSystemNoticeMessage({ type: 'system_notice', message: { ...validNotice(), ...override } })
    ).toBe(false);
  });

  // The XR panel dereferences every item field while rendering, so a bad item
  // must be rejected here rather than thrown from inside the render loop.
  it.each([
    ['a null item', [null]],
    ['an undefined item', [undefined]],
    ['a non-object item', ['CPU governor']],
    ['an item missing name', [{ actual: 'a', required: 'r' }]],
    ['an item missing actual', [{ name: 'n', required: 'r' }]],
    ['an item missing required', [{ name: 'n', actual: 'a' }]],
    ['an item with a non-string name', [{ name: 1, actual: 'a', required: 'r' }]],
    ['an item with a non-string detail', [{ name: 'n', actual: 'a', required: 'r', detail: 5 }]],
    ['one bad item among good ones', [{ name: 'n', actual: 'a', required: 'r' }, null]],
  ])('rejects %s', (_label, items) => {
    expect(
      isSystemNoticeMessage({ type: 'system_notice', message: { ...validNotice(), items } })
    ).toBe(false);
  });

  it('accepts an item without the optional detail', () => {
    const items = [{ name: 'n', actual: 'a', required: 'r' }];
    expect(
      isSystemNoticeMessage({ type: 'system_notice', message: { ...validNotice(), items } })
    ).toBe(true);
  });
});

describe('formatSystemNoticeBody', () => {
  // The XR panel styles the title separately and renders only this body, so
  // anything missing here is invisible to the operator wearing the headset --
  // the primary display surface for this feature.
  it('includes the remediation hint for each item', () => {
    expect(formatSystemNoticeBody(validNotice())).toContain(
      'sudo cpupower frequency-set -g performance'
    );
  });

  it('includes the documentation link', () => {
    expect(formatSystemNoticeBody(validNotice())).toContain('https://example.invalid/docs');
  });

  it('omits the title, which the XR panel renders separately', () => {
    expect(formatSystemNoticeBody(validNotice())).not.toContain(
      'Workstation below recommended spec'
    );
  });

  it('matches the body the 2D banner shows', () => {
    // Guards against the two surfaces drifting apart again.
    const notice = validNotice();
    expect(formatSystemNotice(notice)).toBe(`${notice.title}\n${formatSystemNoticeBody(notice)}`);
  });

  it('uses ASCII bullets so the MSDF font atlas can render them', () => {
    expect(formatSystemNoticeBody(validNotice())).not.toMatch(/[\u2022\u00b7\u25aa]/);
  });

  it('separates every line with a newline for pre-line rendering', () => {
    // uikit collapses whitespace unless whiteSpace="pre-line"; the body relies
    // on those newlines surviving to stay readable in the headset.
    const lines = formatSystemNoticeBody(validNotice()).split('\n');
    expect(lines.length).toBeGreaterThanOrEqual(4); // summary + item + detail + doc_url
  });
});

describe('formatSystemNotice', () => {
  it('renders every accepted notice without throwing', () => {
    const text = formatSystemNotice(validNotice());
    expect(text).toContain('Workstation below recommended spec');
    expect(text).toContain('- CPU governor: powersave (need performance)');
    expect(text).toContain('sudo cpupower frequency-set -g performance');
    expect(text).toContain('https://example.invalid/docs');
  });

  it('separates entries with newlines for the pre-line status box', () => {
    expect(formatSystemNotice(validNotice()).split('\n').length).toBeGreaterThan(1);
  });

  it('omits optional fields that are absent', () => {
    const notice: SystemNotice = {
      level: 'info',
      title: 't',
      summary: 's',
      items: [{ name: 'n', actual: 'a', required: 'r' }],
    };
    expect(formatSystemNotice(notice)).toBe('t\ns\n- n: a (need r)');
  });
});
