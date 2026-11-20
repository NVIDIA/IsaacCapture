// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/** Modal focus across disclosures: a summary can take focus, and what a collapsed one hides cannot. */
import { afterEach, describe, expect, it } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { useModalFocus } from "./settingsParts";

afterEach(cleanup);

function Modal() {
  const dialogRef = useModalFocus(true);
  return (
    <div ref={dialogRef} tabIndex={-1} role="dialog" aria-label="Modal">
      <button type="button">First</button>
      <details>
        <summary>Collapsed</summary>
        <button type="button">Inside collapsed</button>
        <details open>
          <summary>Nested</summary>
          <input aria-label="Nested field" />
        </details>
      </details>
      <details open>
        <summary>Open</summary>
        <input aria-label="Shown field" />
      </details>
    </div>
  );
}

/** Press Tab and say where focus went. */
function tab(shiftKey = false): string | null {
  fireEvent.keyDown(document.activeElement as Element, { key: "Tab", shiftKey });
  const focused = document.activeElement as HTMLElement;
  return focused.getAttribute("aria-label") ?? focused.textContent;
}

describe("useModalFocus", () => {
  it("stops at each summary and skips everything a collapsed disclosure hides", () => {
    render(<Modal />);
    expect(document.activeElement).toBe(screen.getByRole("dialog"));
    expect([tab(), tab(), tab(), tab(), tab()]).toEqual(["First", "Collapsed", "Open", "Shown field", "First"]);
    expect([tab(true), tab(true), tab(true)]).toEqual(["Shown field", "Open", "Collapsed"]);
  });

  it("reaches what a disclosure holds once it is opened, nested ones included", () => {
    render(<Modal />);
    (screen.getByText("Collapsed").parentElement as HTMLDetailsElement).open = true;
    expect([tab(), tab(), tab(), tab(), tab(), tab()]).toEqual(["First", "Collapsed", "Inside collapsed", "Nested", "Nested field", "Open"]);
  });

  it("moves on from a field whose disclosure was collapsed around it", () => {
    render(<Modal />);
    screen.getByLabelText("Shown field").focus();
    (screen.getByText("Open").parentElement as HTMLDetailsElement).open = false;
    expect([tab(), tab(), tab(), tab()]).toEqual(["First", "Collapsed", "Open", "First"]);
  });
});
