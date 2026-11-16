// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/** The Stations card: adding, selecting, renaming and deleting stations, and what a take in progress freezes. */
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { useState } from "react";
import { SettingsHttpError } from "./settingsApi";
import { STATIONS_FIXTURE, fakeStationRegistry } from "./settingsTestDoubles";
import { StationsCard, type StationsBackend } from "./StationsCard";
import type { StationListing, StationRecord } from "./stationsApi";

afterEach(cleanup);

const BENCH: StationRecord = {
  id: "st_1",
  name: "Bench A",
  camera_type: "oak_d",
  camera_roles: { ego: "cam_head", wrist_left: "cam_left", wrist_right: null },
  problems: [],
  headset_address: null,
};
const CART: StationRecord = {
  id: "st_2",
  name: "Cart B",
  camera_type: null,
  camera_roles: { ego: null, wrist_left: null, wrist_right: "cam_grip" },
  problems: ["no camera is assigned to the ego (head) role", "camera 'cam_grip' (right wrist) is not offered by the capture session"],
  headset_address: null,
};
const TWO_STATIONS: StationListing = { ...STATIONS_FIXTURE, stations: [BENCH, CART], selected: "st_1", label: "Bench A" };

/** The card over an in-memory registry, re-rendered with each listing the backend answers with. */
function renderCard(initial: StationListing = STATIONS_FIXTURE, options: { controlToken?: string | null; backend?: Partial<StationsBackend> } = {}) {
  const registry = fakeStationRegistry(initial);
  const backend: StationsBackend = {
    addStation: vi.fn(registry.addStation),
    updateStation: vi.fn(registry.updateStation),
    deleteStation: vi.fn(registry.deleteStation),
    selectStation: vi.fn(registry.selectStation),
    ...options.backend,
  };
  const onListing = vi.fn();
  function Harness() {
    const [listing, setListing] = useState(initial);
    return (
      <StationsCard
        listing={listing}
        controlToken={options.controlToken === undefined ? "tok" : options.controlToken}
        backend={backend}
        inherited={{ camera_type: "oak_d", camera_roles: { ego: "Ego", wrist_left: null, wrist_right: null } }}
        onListing={(next) => {
          onListing(next);
          setListing(next);
        }}
      />
    );
  }
  render(<Harness />);
  return { backend, onListing, registry };
}

const rows = () => within(screen.getByRole("list", { name: "Stations" })).getAllByRole("listitem");
const button = (name: string) => screen.getByRole("button", { name }) as HTMLButtonElement;

describe("StationsCard with no stations", () => {
  it("says which label recordings carry and offers only to add one", () => {
    renderCard();
    expect(screen.getByText(/Recordings carry the deployment's label, “Station 01”/)).toBeTruthy();
    expect(screen.queryByRole("list", { name: "Stations" })).toBeNull();
    expect(button("Add station").disabled).toBe(true); // nothing typed yet
  });

  it("gives the first station the camera setup already saved and puts it in use", async () => {
    const { backend } = renderCard();
    fireEvent.change(screen.getByLabelText("New station name"), { target: { value: "  Bench A " } });
    fireEvent.click(button("Add station"));
    await waitFor(() => expect(rows()).toHaveLength(2)); // the station and the "No station" choice
    expect(backend.addStation).toHaveBeenCalledWith("tok", {
      camera_type: "oak_d",
      camera_roles: { ego: "Ego", wrist_left: null, wrist_right: null },
      name: "Bench A",
    });
    expect(backend.selectStation).toHaveBeenCalledWith("tok", "st_1");
    expect((screen.getByLabelText("Use Bench A") as HTMLInputElement).checked).toBe(true);
    expect(within(rows()[0]).getByText("in use")).toBeTruthy();
    expect((screen.getByLabelText("New station name") as HTMLInputElement).value).toBe("");
  });

  it("leaves the first station unselected when it is added during a take", async () => {
    const registry = fakeStationRegistry();
    const addStation = vi.fn(async (token: string, fields: { name?: string }) => ({
      ...(await registry.addStation(token, fields)),
      locked_reason: "finish or discard the current take before changing the station in use",
    }));
    const { backend } = renderCard(STATIONS_FIXTURE, { backend: { addStation } });
    fireEvent.change(screen.getByLabelText("New station name"), { target: { value: "Bench A" } });
    fireEvent.keyDown(screen.getByLabelText("New station name"), { key: "Enter" });
    await waitFor(() => expect(rows()).toHaveLength(2));
    expect(backend.selectStation).not.toHaveBeenCalled();
    expect(screen.getByText("Finish or discard the current take before changing the station in use.")).toBeTruthy();
  });
});

describe("StationsCard with stations", () => {
  it("lists each station with what fills its roles and why it cannot record", () => {
    renderCard(TWO_STATIONS);
    const [bench, cart, none] = rows();
    expect(bench.textContent).toContain("Ego (head): cam_head · Left wrist: cam_left · Right wrist: —");
    expect(within(bench).getByText("in use")).toBeTruthy();
    expect(cart.textContent).toContain("Ego (head): — · Left wrist: — · Right wrist: cam_grip");
    expect(cart.textContent).toContain("Cannot record yet: no camera is assigned to the ego (head) role; camera 'cam_grip' (right wrist) is not offered by the capture session.");
    expect(within(none).getByText("Recording is held until a station is in use.")).toBeTruthy();
  });

  it("adds a later station bare and does not change the station in use", async () => {
    const { backend } = renderCard(TWO_STATIONS);
    fireEvent.change(screen.getByLabelText("New station name"), { target: { value: "Desk C" } });
    fireEvent.click(button("Add station"));
    await waitFor(() => expect(rows()).toHaveLength(4));
    expect(backend.addStation).toHaveBeenCalledWith("tok", { name: "Desk C" });
    expect(backend.selectStation).not.toHaveBeenCalled();
  });

  it("puts another station in use, and none", async () => {
    const { backend, onListing } = renderCard(TWO_STATIONS);
    fireEvent.click(screen.getByLabelText("Use Cart B"));
    await waitFor(() => expect((screen.getByLabelText("Use Cart B") as HTMLInputElement).checked).toBe(true));
    expect(backend.selectStation).toHaveBeenLastCalledWith("tok", "st_2");
    fireEvent.click(screen.getByLabelText("Use no station"));
    await waitFor(() => expect((screen.getByLabelText("Use no station") as HTMLInputElement).checked).toBe(true));
    expect(backend.selectStation).toHaveBeenLastCalledWith("tok", null);
    expect(onListing).toHaveBeenLastCalledWith(expect.objectContaining({ selected: null, label: "Station 01" }));
  });

  it("renames a station with Enter or the button, and Escape or an unchanged name saves nothing", async () => {
    const { backend } = renderCard(TWO_STATIONS);
    fireEvent.click(within(rows()[1]).getByRole("button", { name: "Rename" }));
    const field = screen.getByLabelText("New name for Cart B") as HTMLInputElement;
    expect(button("Save name").disabled).toBe(true); // unchanged
    fireEvent.keyDown(field, { key: "Enter" });
    expect(backend.updateStation).not.toHaveBeenCalled();
    fireEvent.click(within(rows()[1]).getByRole("button", { name: "Rename" }));
    fireEvent.change(screen.getByLabelText("New name for Cart B"), { target: { value: "Cart Z" } });
    const escape = new KeyboardEvent("keydown", { key: "Escape", bubbles: true });
    const stop = vi.spyOn(escape, "stopPropagation");
    fireEvent(screen.getByLabelText("New name for Cart B"), escape);
    expect(stop).toHaveBeenCalled(); // the dialog must not close on it
    expect(backend.updateStation).not.toHaveBeenCalled();
    fireEvent.click(within(rows()[1]).getByRole("button", { name: "Rename" }));
    fireEvent.change(screen.getByLabelText("New name for Cart B"), { target: { value: " Cart Z " } });
    fireEvent.keyDown(screen.getByLabelText("New name for Cart B"), { key: "Enter" });
    await waitFor(() => expect(screen.getByLabelText("Use Cart Z")).toBeTruthy());
    expect(backend.updateStation).toHaveBeenCalledWith("tok", "st_2", { name: "Cart Z" });
    fireEvent.click(within(rows()[0]).getByRole("button", { name: "Rename" }));
    fireEvent.change(screen.getByLabelText("New name for Bench A"), { target: { value: "Bench Z" } });
    fireEvent.click(button("Save name"));
    await waitFor(() => expect(screen.getByLabelText("Use Bench Z")).toBeTruthy());
    fireEvent.click(within(rows()[0]).getByRole("button", { name: "Rename" }));
    fireEvent.click(button("Cancel"));
    expect(screen.queryByLabelText("New name for Bench Z")).toBeNull();
  });

  it("deletes a station only after the operator confirms", async () => {
    const { backend } = renderCard(TWO_STATIONS);
    fireEvent.click(within(rows()[1]).getByRole("button", { name: "Delete" }));
    fireEvent.click(button("Keep"));
    expect(backend.deleteStation).not.toHaveBeenCalled();
    fireEvent.click(within(rows()[1]).getByRole("button", { name: "Delete" }));
    fireEvent.click(button("Delete Cart B"));
    await waitFor(() => expect(rows()).toHaveLength(2));
    expect(backend.deleteStation).toHaveBeenCalledWith("tok", "st_2");
  });

  it("shows the backend's reason for a refused change and keeps the list", async () => {
    const addStation = vi.fn(async () => {
      throw new SettingsHttpError("a station named 'Bench A' already exists");
    });
    renderCard(TWO_STATIONS, { backend: { addStation } });
    fireEvent.change(screen.getByLabelText("New station name"), { target: { value: "bench a" } });
    fireEvent.click(button("Add station"));
    expect(await screen.findByText("Not saved: a station named 'Bench A' already exists")).toBeTruthy();
    expect(rows()).toHaveLength(3);
  });

  it("freezes the station in use during a take and leaves the others editable", () => {
    renderCard({ ...TWO_STATIONS, locked_reason: "finish or discard the current take before changing the station in use" });
    const [bench, cart] = rows();
    expect((screen.getByLabelText("Use Cart B") as HTMLInputElement).disabled).toBe(true);
    expect((screen.getByLabelText("Use no station") as HTMLInputElement).disabled).toBe(true);
    expect((within(bench).getByRole("button", { name: "Rename" }) as HTMLButtonElement).disabled).toBe(true);
    expect((within(bench).getByRole("button", { name: "Delete" }) as HTMLButtonElement).disabled).toBe(true);
    expect((within(cart).getByRole("button", { name: "Rename" }) as HTMLButtonElement).disabled).toBe(false);
    expect((within(cart).getByRole("button", { name: "Delete" }) as HTMLButtonElement).disabled).toBe(false);
    expect(screen.getByText("Finish or discard the current take before changing the station in use.")).toBeTruthy();
  });

  it("changes nothing without the control token and says where to enter it", () => {
    const { backend } = renderCard(TWO_STATIONS, { controlToken: null });
    expect((screen.getByLabelText("Use Cart B") as HTMLInputElement).disabled).toBe(true);
    expect((screen.getByLabelText("New station name") as HTMLInputElement).disabled).toBe(true);
    expect((within(rows()[1]).getByRole("button", { name: "Rename" }) as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByText(/Enter the control token in the recording panel/)).toBeTruthy();
    expect(backend.addStation).not.toHaveBeenCalled();
  });

  it("shows the warning of a registry that could not be read", () => {
    renderCard({ ...STATIONS_FIXTURE, warning: "Saved stations are unreadable. Adding a station replaces them." });
    expect(screen.getByText("Saved stations are unreadable. Adding a station replaces them.")).toBeTruthy();
  });
});
