// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * The side panel's width: its limits, how a requested width is clamped to the window, and remembering the choice in
 * this browser. Pure and independent of React so the rules are testable on their own.
 */

export const SIDEBAR_MIN_PX = 400;
export const SIDEBAR_DEFAULT_PX = 460;
export const SIDEBAR_MAX_PX = 1000;
/** The main column (cameras, status bar, controls) never gets narrower than this, however wide the panel is dragged. */
export const MAIN_MIN_PX = 560;

const STORAGE_KEY = "egocap.sidebarWidth";

/** The widest the panel may be in a window this wide: what the main column leaves, within the absolute limits. */
export function maxSidebarWidth(viewportWidth: number): number {
  return Math.max(SIDEBAR_MIN_PX, Math.min(SIDEBAR_MAX_PX, viewportWidth - MAIN_MIN_PX));
}

/** A requested width brought into the allowed range for this window, as a whole number of pixels. A non-finite
 * request (a bad stored value) is the default rather than NaN reaching the layout. */
export function clampSidebarWidth(width: number, viewportWidth: number): number {
  if (!Number.isFinite(width)) return Math.min(SIDEBAR_DEFAULT_PX, maxSidebarWidth(viewportWidth));
  return Math.round(Math.min(Math.max(width, SIDEBAR_MIN_PX), maxSidebarWidth(viewportWidth)));
}

/** The width remembered in this browser, or the default. Storage can be blocked or hold junk; neither is an error. */
export function loadSidebarWidth(): number {
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (stored !== null) {
      const parsed = Number(stored);
      if (Number.isFinite(parsed)) return parsed;
    }
  } catch {
    // storage blocked: use the default
  }
  return SIDEBAR_DEFAULT_PX;
}

export function saveSidebarWidth(width: number): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, String(width));
  } catch {
    // storage blocked: the width just is not remembered
  }
}
