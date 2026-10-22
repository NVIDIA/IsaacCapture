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
 * OOB control token delivery.
 *
 * The host puts the token in the URL fragment (`#/route?controlToken=…`) so it is
 * never sent in an HTTP request target or Referer. The page lifts it into
 * sessionStorage (tab-scoped, keyed by hub endpoint so a reload keeps auth) and
 * strips it from the address bar. A `?controlToken=` query is still read for
 * hand-built bookmarks, but that URL has already been sent over the network.
 */

const PARAM = 'controlToken';
const STORAGE_PREFIX = 'oobControlToken:';

// Fallback when sessionStorage is denied: survives this page load only.
const memoryTokens = new Map<string, string>();

function endpointKey(search: URLSearchParams): string {
  const host = (search.get('serverIP') ?? '').trim();
  const port = (search.get('port') ?? '').trim();
  return `${STORAGE_PREFIX}${host}:${port}`;
}

function storeToken(key: string, token: string): void {
  memoryTokens.set(key, token);
  try {
    window.sessionStorage.setItem(key, token);
  } catch {
    console.warn(
      '[Teleop] sessionStorage unavailable; OOB control token will not survive a reload'
    );
  }
}

function loadToken(key: string): string | null {
  // This page load's token wins: a denied write must not fall back to an older stored one.
  const current = memoryTokens.get(key);
  if (current) return current;
  try {
    return window.sessionStorage.getItem(key) || null;
  } catch {
    return null;
  }
}

/**
 * Lifts a control token out of the URL (fragment first, then legacy query) into
 * session storage and removes it from the address bar. Call before anything
 * rewrites or reads the fragment. A supplied token replaces any stored one;
 * with none supplied the stored token for this endpoint is kept.
 */
export function captureOobControlToken(): void {
  const { pathname, search, hash } = window.location;
  const searchParams = new URLSearchParams(search);

  // Fragment layout: `#<route>?<params>`; split at the first '?'.
  const fragment = hash.replace(/^#/, '');
  const split = fragment.indexOf('?');
  const route = split === -1 ? fragment : fragment.slice(0, split);
  const fragmentParams = new URLSearchParams(split === -1 ? '' : fragment.slice(split + 1));

  const token = fragmentParams.get(PARAM) || searchParams.get(PARAM);
  if (!searchParams.has(PARAM) && !fragmentParams.has(PARAM)) return;

  if (token) storeToken(endpointKey(searchParams), token);

  searchParams.delete(PARAM);
  fragmentParams.delete(PARAM);
  const newSearch = searchParams.toString();
  const newFragmentParams = fragmentParams.toString();
  const newHash =
    route || newFragmentParams
      ? `#${route}${newFragmentParams ? `?${newFragmentParams}` : ''}`
      : '';
  window.history.replaceState(
    window.history.state,
    '',
    `${pathname}${newSearch ? `?${newSearch}` : ''}${newHash}`
  );
}

/** Control token for the hub named by the current `serverIP`/`port` query, if any. */
export function getOobControlToken(search: URLSearchParams): string | undefined {
  return loadToken(endpointKey(search)) ?? undefined;
}
