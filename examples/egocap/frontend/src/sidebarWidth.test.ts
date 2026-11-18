// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/** The side panel's width rules: clamped to the window so the cameras keep their room, and remembered safely. */
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  MAIN_MIN_PX,
  SIDEBAR_DEFAULT_PX,
  SIDEBAR_MAX_PX,
  SIDEBAR_MIN_PX,
  clampSidebarWidth,
  loadSidebarWidth,
  maxSidebarWidth,
  saveSidebarWidth,
} from "./sidebarWidth";

afterEach(() => {
  window.localStorage.clear();
  vi.restoreAllMocks();
});

describe("maxSidebarWidth", () => {
  it("leaves the main column its minimum, within the absolute limits", () => {
    expect(maxSidebarWidth(1600)).toBe(SIDEBAR_MAX_PX); // 1040 would be allowed, the absolute limit wins
    expect(maxSidebarWidth(1200)).toBe(1200 - MAIN_MIN_PX);
    expect(maxSidebarWidth(700)).toBe(SIDEBAR_MIN_PX); // a tiny window still gets the minimum, never less
  });
});

describe("clampSidebarWidth", () => {
  it("keeps a width that fits and rounds to whole pixels", () => {
    expect(clampSidebarWidth(500.4, 1600)).toBe(500);
  });

  it("holds the minimum and the window's maximum", () => {
    expect(clampSidebarWidth(10, 1600)).toBe(SIDEBAR_MIN_PX);
    expect(clampSidebarWidth(5000, 1600)).toBe(SIDEBAR_MAX_PX);
    expect(clampSidebarWidth(900, 1200)).toBe(1200 - MAIN_MIN_PX);
  });

  it("falls back to the default, within the window, for a junk request", () => {
    expect(clampSidebarWidth(Number.NaN, 1600)).toBe(SIDEBAR_DEFAULT_PX);
    expect(clampSidebarWidth(Number.POSITIVE_INFINITY, 1600)).toBe(SIDEBAR_DEFAULT_PX);
    expect(clampSidebarWidth(Number.NaN, 900)).toBe(Math.min(SIDEBAR_DEFAULT_PX, maxSidebarWidth(900)));
  });
});

describe("remembering the width", () => {
  it("round-trips a saved width", () => {
    saveSidebarWidth(612);
    expect(loadSidebarWidth()).toBe(612);
  });

  it("is the default with nothing stored, with junk stored, or when storage throws", () => {
    expect(loadSidebarWidth()).toBe(SIDEBAR_DEFAULT_PX);
    window.localStorage.setItem("egocap.sidebarWidth", "wide");
    expect(loadSidebarWidth()).toBe(SIDEBAR_DEFAULT_PX);
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("blocked");
    });
    expect(loadSidebarWidth()).toBe(SIDEBAR_DEFAULT_PX);
  });

  it("does not throw when storage refuses a write", () => {
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("blocked");
    });
    expect(() => saveSidebarWidth(500)).not.toThrow();
  });
});
