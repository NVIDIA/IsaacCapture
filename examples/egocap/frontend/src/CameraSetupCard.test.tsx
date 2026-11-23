// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/** The Camera setup card: a setup saved elsewhere while the draft is changed is taken once the draft is clean. */
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { CameraSetupCard } from "./CameraSetupCard";
import { OPERATOR_SETTINGS_FIXTURE } from "./settingsTestDoubles";

afterEach(cleanup);

describe("CameraSetupCard", () => {
  it("keeps a changed draft over a setup saved elsewhere, and takes that setup once the draft is clean again", () => {
    const props = { cameraSources: ["Ego", "Wrist A"], controlToken: "tok", save: vi.fn(async () => undefined) };
    const { rerender } = render(<CameraSetupCard operator={OPERATOR_SETTINGS_FIXTURE} {...props} />);
    const ego = () => screen.getByLabelText("Ego (head)") as HTMLSelectElement;
    const type = () => screen.getByLabelText("Camera type") as HTMLSelectElement;
    fireEvent.change(ego(), { target: { value: "Ego" } });
    // Another browser saves a camera type while this draft is changed: the draft is kept as typed.
    rerender(<CameraSetupCard operator={{ ...OPERATOR_SETTINGS_FIXTURE, camera_type: "oak_d" }} {...props} />);
    expect([ego().value, type().value]).toEqual(["Ego", ""]);
    // Putting the field back leaves nothing of the operator's own, so the saved setup is shown.
    fireEvent.change(ego(), { target: { value: "" } });
    expect([ego().value, type().value]).toEqual(["", "oak_d"]);
    expect((screen.getByRole("button", { name: "Save camera setup" }) as HTMLButtonElement).disabled).toBe(true);
  });
});
