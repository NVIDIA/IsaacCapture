// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/** The Stations card: adding, selecting, editing in place and deleting stations, and what a take in progress freezes. */
import { afterEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { useState } from "react";
import { SettingsHttpError } from "./settingsApi";
import { STATIONS_FIXTURE, fakeStationRegistry } from "./settingsTestDoubles";
import type { HeadsetLink } from "./setupApi";
import { StationsCard, type StationsBackend } from "./StationsCard";
import type { StationListing, StationRecord } from "./stationsApi";

afterEach(cleanup);

const BENCH: StationRecord = {
  id: "st_1",
  name: "Bench A",
  camera_type: "oak_d",
  camera_roles: { ego: "cam_head", wrist_left: "cam_left", wrist_right: null },
  problems: [],
};
const CART: StationRecord = {
  id: "st_2",
  name: "Cart B",
  camera_type: null,
  camera_roles: { ego: null, wrist_left: null, wrist_right: "cam_grip" },
  problems: ["no camera is assigned to the ego (head) role", "camera 'cam_grip' (right wrist) is not offered by the capture session"],
};
const TWO_STATIONS: StationListing = { ...STATIONS_FIXTURE, stations: [BENCH, CART], selected: "st_1", label: "Bench A" };
const HEADSET: HeadsetLink = { address: "10.0.0.42", saved: "10.0.0.42", changed: false };
const HEADSET_MOVED: HeadsetLink = { address: "10.0.0.42", saved: "10.0.0.7", changed: true };
const MOVED_WARNING = "This host's address changed from 10.0.0.7 to 10.0.0.42. Enter the new one in the headset.";

interface CardOptions {
  controlToken?: string | null;
  backend?: Partial<StationsBackend>;
  cameraSources?: string[];
  headset?: HeadsetLink | null;
}

/** The card over an in-memory registry, re-rendered with each listing the backend answers with. */
function renderCard(initial: StationListing = STATIONS_FIXTURE, options: CardOptions = {}) {
  const registry = fakeStationRegistry(initial);
  const backend: StationsBackend = {
    addStation: vi.fn(registry.addStation),
    updateStation: vi.fn(registry.updateStation),
    deleteStation: vi.fn(registry.deleteStation),
    selectStation: vi.fn(registry.selectStation),
    ...options.backend,
  };
  const onListing = vi.fn();
  // Lets a test deliver a listing the card did not ask for, as a reload after another browser's change does.
  const outside: { deliver: (listing: StationListing) => void } = { deliver: () => undefined };
  function Harness() {
    const [listing, setListing] = useState(initial);
    outside.deliver = setListing;
    return (
      <StationsCard
        listing={listing}
        controlToken={options.controlToken === undefined ? "tok" : options.controlToken}
        backend={backend}
        inherited={{ camera_type: "oak_d", camera_roles: { ego: "Ego", wrist_left: null, wrist_right: null } }}
        cameraSources={options.cameraSources ?? ["cam_head", "cam_left", "cam_right"]}
        headset={options.headset ?? null}
        onListing={(next) => {
          onListing(next);
          setListing(next);
        }}
      />
    );
  }
  render(<Harness />);
  return { backend, onListing, registry, deliver: (listing: StationListing) => act(() => outside.deliver(listing)) };
}

const rows = () => within(screen.getByRole("list", { name: "Stations" })).getAllByRole("listitem");
const button = (name: string) => screen.getByRole("button", { name }) as HTMLButtonElement;
const editor = (name: string) => screen.getByRole("group", { name: `Edit ${name}` });
const field = (station: string, label: string) => within(editor(station)).getByLabelText(label) as HTMLInputElement & HTMLSelectElement;
/** Open a station's editor from its row. */
function edit(row: number) {
  fireEvent.click(within(rows()[row]).getByRole("button", { name: "Edit" }));
}

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

  it("shows the first station's row and the refusal when it was added but could not be put in use", async () => {
    const selectStation = vi.fn(async () => {
      throw new SettingsHttpError("stations are not available yet");
    });
    const { onListing } = renderCard(STATIONS_FIXTURE, { backend: { selectStation } });
    fireEvent.change(screen.getByLabelText("New station name"), { target: { value: "Bench A" } });
    fireEvent.click(button("Add station"));
    expect(await screen.findByText("Not saved: stations are not available yet")).toBeTruthy();
    expect((screen.getByLabelText("Use Bench A") as HTMLInputElement).checked).toBe(false);
    // The station exists, so its name is not left in the field to be added twice.
    expect((screen.getByLabelText("New station name") as HTMLInputElement).value).toBe("");
    expect(onListing).toHaveBeenCalledTimes(1);
    expect(onListing).toHaveBeenCalledWith(expect.objectContaining({ selected: null, stations: [expect.objectContaining({ name: "Bench A" })] }));
  });
});

describe("StationsCard with stations", () => {
  it("lists each station with what fills its roles and why it cannot record", () => {
    renderCard(TWO_STATIONS);
    const [bench, cart, none] = rows();
    expect(bench.textContent).toContain("Ego (head): cam_head · Left wrist: cam_left · Right wrist: —");
    expect(within(bench).getByText("in use")).toBeTruthy();
    expect(cart.textContent).toContain("Ego (head): — · Left wrist: — · Right wrist: cam_grip");
    const problems = within(cart).getByText("Cannot record yet: no camera is assigned to the ego (head) role; camera 'cam_grip' (right wrist) is not offered by the capture session.");
    expect(problems.className).toContain("tone-warning");
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

  it("shows why the station in use cannot change while an interrupted take waits for recovery", async () => {
    const refused = async (): Promise<StationListing> => {
      throw new SettingsHttpError("recover or discard the interrupted take first");
    };
    const { backend } = renderCard(TWO_STATIONS, { backend: { selectStation: vi.fn(refused), deleteStation: vi.fn(refused), updateStation: vi.fn(refused) } });
    const refusal = "Not saved: recover or discard the interrupted take first";
    fireEvent.click(screen.getByLabelText("Use Cart B"));
    expect(await screen.findByText(refusal)).toBeTruthy();
    expect((screen.getByLabelText("Use Bench A") as HTMLInputElement).checked).toBe(true);
    // Deleting the station in use is refused the same way, and the row stays.
    fireEvent.click(within(rows()[0]).getByRole("button", { name: "Delete" }));
    fireEvent.click(button("Delete Bench A"));
    await waitFor(() => expect(backend.deleteStation).toHaveBeenCalledWith("tok", "st_1"));
    expect(await screen.findByText(refusal)).toBeTruthy();
    expect(rows()).toHaveLength(3);
    // A rename of it is refused in its own editor, beside what was typed.
    edit(0);
    fireEvent.change(field("Bench A", "Name"), { target: { value: "Bench Z" } });
    fireEvent.click(button("Save"));
    expect(await within(editor("Bench A")).findByText(refusal)).toBeTruthy();
    expect(field("Bench A", "Name").value).toBe("Bench Z");
  });

  it("keeps the typed name when the add is refused, so it can be corrected", async () => {
    const addStation = vi.fn(async () => {
      throw new SettingsHttpError("a station named 'Bench A' already exists");
    });
    renderCard(TWO_STATIONS, { backend: { addStation } });
    const name = screen.getByLabelText("New station name") as HTMLInputElement;
    fireEvent.change(name, { target: { value: "bench a" } });
    fireEvent.keyDown(name, { key: "Enter" });
    await screen.findByText("Not saved: a station named 'Bench A' already exists");
    expect(name.value).toBe("bench a");
    expect(name.disabled).toBe(false);
  });

  it("moves keyboard focus to the delete confirmation, and back to Edit when the station is kept", () => {
    renderCard(TWO_STATIONS);
    fireEvent.click(within(rows()[1]).getByRole("button", { name: "Delete" }));
    expect(document.activeElement).toBe(button("Delete Cart B"));
    fireEvent.click(button("Keep"));
    expect(document.activeElement).toBe(within(rows()[1]).getByRole("button", { name: "Edit" }));
  });

  it("freezes the station in use during a take and leaves the others editable", () => {
    renderCard({ ...TWO_STATIONS, locked_reason: "finish or discard the current take before changing the station in use" });
    const [bench, cart] = rows();
    expect((screen.getByLabelText("Use Cart B") as HTMLInputElement).disabled).toBe(true);
    expect((screen.getByLabelText("Use no station") as HTMLInputElement).disabled).toBe(true);
    expect((within(bench).getByRole("button", { name: "Edit" }) as HTMLButtonElement).disabled).toBe(true);
    expect((within(bench).getByRole("button", { name: "Delete" }) as HTMLButtonElement).disabled).toBe(true);
    expect((within(cart).getByRole("button", { name: "Edit" }) as HTMLButtonElement).disabled).toBe(false);
    expect((within(cart).getByRole("button", { name: "Delete" }) as HTMLButtonElement).disabled).toBe(false);
    expect(screen.getByText("Finish or discard the current take before changing the station in use.")).toBeTruthy();
  });

  it("changes nothing without the control token and says where to enter it", () => {
    const { backend } = renderCard(TWO_STATIONS, { controlToken: null });
    expect((screen.getByLabelText("Use Cart B") as HTMLInputElement).disabled).toBe(true);
    expect((screen.getByLabelText("New station name") as HTMLInputElement).disabled).toBe(true);
    expect((within(rows()[1]).getByRole("button", { name: "Edit" }) as HTMLButtonElement).disabled).toBe(true);
    expect((within(rows()[1]).getByRole("button", { name: "Delete" }) as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByText(/Enter the control token in the recording panel/)).toBeTruthy();
    expect(backend.addStation).not.toHaveBeenCalled();
  });

  it("shows the warning of a registry that could not be read", () => {
    renderCard({ ...STATIONS_FIXTURE, warning: "Saved stations are unreadable. Adding a station replaces them." });
    expect(screen.getByText("Saved stations are unreadable. Adding a station replaces them.")).toBeTruthy();
  });
});

describe("a station's inline editor", () => {
  it("opens in the station's own row, one row at a time, with the station's saved values", () => {
    renderCard(TWO_STATIONS);
    expect(screen.queryByRole("group")).toBeNull();
    edit(0);
    expect(rows()[0].contains(editor("Bench A"))).toBe(true);
    expect(field("Bench A", "Name").value).toBe("Bench A");
    expect(field("Bench A", "Name").maxLength).toBe(64);
    expect(document.activeElement).toBe(field("Bench A", "Name"));
    expect(field("Bench A", "Camera type").value).toBe("oak_d");
    expect(Array.from(field("Bench A", "Camera type").options).map((option) => option.textContent)).toEqual([
      "Not set",
      "SENSING GMSL cameras",
      "OAK-D cameras — USB, read by CaptureServer",
    ]);
    expect(field("Bench A", "Ego (head)").value).toBe("cam_head");
    expect(field("Bench A", "Left wrist").value).toBe("cam_left");
    expect(field("Bench A", "Right wrist").value).toBe("");
    // The row's own actions give way to the editor's.
    expect(within(rows()[0]).queryByRole("button", { name: "Edit" })).toBeNull();
    expect(button("Save").disabled).toBe(true); // nothing changed yet
    edit(1);
    expect(screen.queryByRole("group", { name: "Edit Bench A" })).toBeNull();
    expect(rows()[1].contains(editor("Cart B"))).toBe(true);
    expect(field("Cart B", "Camera type").value).toBe("");
  });

  it("offers the session's cameras and the ones already saved, and one camera to one role only", () => {
    renderCard(TWO_STATIONS);
    edit(1);
    const right = field("Cart B", "Right wrist");
    expect(Array.from(right.options).map((option) => option.textContent)).toEqual(["Not assigned", "cam_head", "cam_left", "cam_right", "cam_grip (not in this session)"]);
    expect(right.value).toBe("cam_grip");
    fireEvent.change(field("Cart B", "Ego (head)"), { target: { value: "cam_head" } });
    expect((within(right).getByRole("option", { name: "cam_head" }) as HTMLOptionElement).disabled).toBe(true);
    expect((within(field("Cart B", "Left wrist")).getByRole("option", { name: "cam_head" }) as HTMLOptionElement).disabled).toBe(true);
    expect((within(field("Cart B", "Ego (head)")).getByRole("option", { name: "cam_head" }) as HTMLOptionElement).disabled).toBe(false);
    expect((within(field("Cart B", "Ego (head)")).getByRole("option", { name: "cam_grip (not in this session)" }) as HTMLOptionElement).disabled).toBe(true);
  });

  it("says so when the session offers no cameras to assign", () => {
    renderCard({ ...TWO_STATIONS, stations: [{ ...CART, camera_roles: { ego: null, wrist_left: null, wrist_right: null } }] }, { cameraSources: [] });
    edit(0);
    expect(within(editor("Cart B")).getByText(/The session offers no cameras yet/)).toBeTruthy();
    expect(button("Swap wrists").disabled).toBe(true);
  });

  it("saves name, camera type and roles in one request of only what changed, then closes", async () => {
    const { backend, onListing } = renderCard(TWO_STATIONS);
    edit(0);
    fireEvent.change(field("Bench A", "Name"), { target: { value: " Bench Z " } });
    fireEvent.change(field("Bench A", "Camera type"), { target: { value: "sensing_gmsl" } });
    fireEvent.change(field("Bench A", "Right wrist"), { target: { value: "cam_right" } });
    fireEvent.click(button("Save"));
    await waitFor(() => expect(screen.queryByRole("group")).toBeNull());
    expect(backend.updateStation).toHaveBeenCalledTimes(1);
    expect(backend.updateStation).toHaveBeenCalledWith("tok", "st_1", { name: "Bench Z", camera_type: "sensing_gmsl", camera_roles: { wrist_right: "cam_right" } });
    expect(onListing).toHaveBeenCalledTimes(1);
    expect(rows()[0].textContent).toContain("Ego (head): cam_head · Left wrist: cam_left · Right wrist: cam_right");
    expect(screen.getByLabelText("Use Bench Z")).toBeTruthy();
  });

  it("does not send back a field another browser changed while the editor was open", async () => {
    const { backend, registry, deliver } = renderCard(TWO_STATIONS);
    edit(0);
    // Another browser renames the station and changes its camera type; a reload brings that here.
    const renamed = await registry.updateStation("tok", "st_1", { name: "Bench Prime", camera_type: "sensing_gmsl" });
    deliver(renamed);
    expect(editor("Bench Prime")).toBeTruthy();
    fireEvent.change(field("Bench Prime", "Right wrist"), { target: { value: "cam_right" } });
    fireEvent.click(button("Save"));
    await waitFor(() => expect(screen.queryByRole("group")).toBeNull());
    // Only the camera this editor assigned: the name and type it never touched are not sent as they were.
    expect(backend.updateStation).toHaveBeenCalledWith("tok", "st_1", { camera_roles: { wrist_right: "cam_right" } });
    expect(registry.listing.stations[0]).toMatchObject({ name: "Bench Prime", camera_type: "sensing_gmsl" });
    expect(screen.getByLabelText("Use Bench Prime")).toBeTruthy();
  });

  it("sends a single changed field alone, and can clear the camera type and a role", async () => {
    const { backend } = renderCard(TWO_STATIONS);
    edit(0);
    fireEvent.change(field("Bench A", "Camera type"), { target: { value: "" } });
    fireEvent.click(button("Save"));
    await waitFor(() => expect(screen.queryByRole("group")).toBeNull());
    expect(backend.updateStation).toHaveBeenLastCalledWith("tok", "st_1", { camera_type: null });
    edit(0);
    fireEvent.change(field("Bench A", "Left wrist"), { target: { value: "" } });
    fireEvent.click(button("Save"));
    await waitFor(() => expect(screen.queryByRole("group")).toBeNull());
    expect(backend.updateStation).toHaveBeenLastCalledWith("tok", "st_1", { camera_roles: { wrist_left: null } });
  });

  it("swaps the wrists", async () => {
    const { backend } = renderCard(TWO_STATIONS);
    edit(0);
    fireEvent.click(button("Swap wrists"));
    expect(field("Bench A", "Left wrist").value).toBe("");
    expect(field("Bench A", "Right wrist").value).toBe("cam_left");
    fireEvent.click(button("Save"));
    await waitFor(() => expect(screen.queryByRole("group")).toBeNull());
    expect(backend.updateStation).toHaveBeenCalledWith("tok", "st_1", { camera_roles: { wrist_left: null, wrist_right: "cam_left" } });
  });

  it("discards the draft on Cancel", () => {
    const { backend } = renderCard(TWO_STATIONS);
    edit(0);
    fireEvent.change(field("Bench A", "Name"), { target: { value: "Bench Z" } });
    fireEvent.click(button("Cancel"));
    expect(screen.queryByRole("group")).toBeNull();
    expect(backend.updateStation).not.toHaveBeenCalled();
    edit(0);
    expect(field("Bench A", "Name").value).toBe("Bench A");
  });

  it("saves on Enter in the name field, and with nothing changed just closes", async () => {
    const { backend } = renderCard(TWO_STATIONS);
    edit(1);
    fireEvent.keyDown(field("Cart B", "Name"), { key: "Enter" });
    expect(screen.queryByRole("group")).toBeNull();
    expect(backend.updateStation).not.toHaveBeenCalled();
    edit(1);
    fireEvent.change(field("Cart B", "Name"), { target: { value: "Cart Z" } });
    fireEvent.keyDown(field("Cart B", "Name"), { key: "Enter" });
    await waitFor(() => expect(screen.getByLabelText("Use Cart Z")).toBeTruthy());
    expect(backend.updateStation).toHaveBeenCalledWith("tok", "st_2", { name: "Cart Z" });
    expect(screen.queryByRole("group")).toBeNull();
  });

  it("does not save an empty name", () => {
    const { backend } = renderCard(TWO_STATIONS);
    edit(0);
    fireEvent.change(field("Bench A", "Name"), { target: { value: "   " } });
    expect(button("Save").disabled).toBe(true);
    fireEvent.keyDown(field("Bench A", "Name"), { key: "Enter" });
    expect(backend.updateStation).not.toHaveBeenCalled();
    expect(editor("Bench A")).toBeTruthy();
  });

  it("closes on Escape without letting the dialog see the key", () => {
    const { backend } = renderCard(TWO_STATIONS);
    const dialogSawKey = vi.fn();
    window.addEventListener("keydown", dialogSawKey);
    edit(0);
    fireEvent.change(field("Bench A", "Name"), { target: { value: "Bench Z" } });
    fireEvent.keyDown(field("Bench A", "Name"), { key: "Escape" });
    expect(screen.queryByRole("group")).toBeNull();
    edit(0);
    fireEvent.keyDown(field("Bench A", "Left wrist"), { key: "Escape" });
    expect(screen.queryByRole("group")).toBeNull();
    window.removeEventListener("keydown", dialogSawKey);
    expect(dialogSawKey).not.toHaveBeenCalled();
    expect(backend.updateStation).not.toHaveBeenCalled();
  });

  it("returns keyboard focus to the row's Edit button when the editor closes by Cancel, Escape or a save", async () => {
    renderCard(TWO_STATIONS);
    const editOf = (row: number) => within(rows()[row]).getByRole("button", { name: "Edit" });
    edit(0);
    fireEvent.click(button("Cancel"));
    expect(document.activeElement).toBe(editOf(0));
    edit(0);
    fireEvent.keyDown(field("Bench A", "Name"), { key: "Escape" });
    expect(document.activeElement).toBe(editOf(0));
    edit(0);
    fireEvent.change(field("Bench A", "Name"), { target: { value: "Bench Z" } });
    fireEvent.click(button("Save"));
    await waitFor(() => expect(screen.queryByRole("group")).toBeNull());
    expect(document.activeElement).toBe(editOf(0));
  });

  it("leaves focus in the editor just opened when it replaces another row's", () => {
    renderCard(TWO_STATIONS);
    edit(0);
    edit(1);
    expect(document.activeElement).toBe(field("Cart B", "Name"));
  });

  it("shows a refusal in the row and keeps the editor open with what was typed", async () => {
    const updateStation = vi.fn(async () => {
      throw new SettingsHttpError("a station named 'Cart B' already exists");
    });
    const { onListing } = renderCard(TWO_STATIONS, { backend: { updateStation } });
    edit(0);
    fireEvent.change(field("Bench A", "Name"), { target: { value: "cart b" } });
    fireEvent.change(field("Bench A", "Right wrist"), { target: { value: "cam_right" } });
    fireEvent.click(button("Save"));
    expect(await within(rows()[0]).findByText("Not saved: a station named 'Cart B' already exists")).toBeTruthy();
    expect(field("Bench A", "Name").value).toBe("cart b");
    expect(field("Bench A", "Right wrist").value).toBe("cam_right");
    expect(button("Save").disabled).toBe(false);
    expect(onListing).not.toHaveBeenCalled();
  });

  it("holds every other registry control while a save is on its way, and leaves the editor able to hold focus", async () => {
    const registry = fakeStationRegistry(TWO_STATIONS);
    let finish: () => void = () => undefined;
    const updateStation = vi.fn(async (...args: Parameters<StationsBackend["updateStation"]>) => {
      await new Promise<void>((resolve) => (finish = resolve));
      return registry.updateStation(...args);
    });
    const { backend } = renderCard(TWO_STATIONS, { backend: { updateStation } });
    fireEvent.change(screen.getByLabelText("New station name"), { target: { value: "Bench C" } });
    edit(0);
    fireEvent.change(field("Bench A", "Name"), { target: { value: "Bench Z" } });
    fireEvent.keyDown(field("Bench A", "Name"), { key: "Enter" });
    expect(button("Saving…")).toBeTruthy();
    // Each answer is a whole listing and the last one wins, so nothing else may change the registry meanwhile.
    const others = [screen.getByLabelText("Use Bench A"), screen.getByLabelText("Use Cart B"), screen.getByLabelText("Use no station"), button("Add station"), ...within(rows()[1]).getAllByRole("button")] as HTMLInputElement[];
    expect(others.map((control) => control.disabled)).toEqual([true, true, true, true, true, true]);
    // A disabled control gives up keyboard focus, and Escape would then close the dialog, not the editor.
    const name = field("Bench A", "Name");
    expect([name.disabled, name.readOnly, button("Saving…").disabled]).toEqual([false, true, false]);
    fireEvent.keyDown(name, { key: "Enter" });
    fireEvent.click(button("Saving…"));
    expect(backend.updateStation).toHaveBeenCalledTimes(1);
    await act(async () => finish());
    await waitFor(() => expect(screen.queryByRole("group")).toBeNull());
    expect(within(rows()[0]).getByText("Bench Z")).toBeTruthy();
    expect(others.map((control) => control.disabled)).toEqual([false, false, false, false, false, false]);
  });

  it("releases the other registry controls after a refused save", async () => {
    const updateStation = vi.fn(async () => {
      throw new SettingsHttpError("a station named 'Cart B' already exists");
    });
    renderCard(TWO_STATIONS, { backend: { updateStation } });
    edit(0);
    fireEvent.change(field("Bench A", "Name"), { target: { value: "cart b" } });
    fireEvent.click(button("Save"));
    await within(rows()[0]).findByText("Not saved: a station named 'Cart B' already exists");
    expect((screen.getByLabelText("Use Cart B") as HTMLInputElement).disabled).toBe(false);
    expect(field("Bench A", "Name").readOnly).toBe(false);
  });

  it("cannot save the station in use once a take starts, while its typed values stay", () => {
    const registry = fakeStationRegistry(TWO_STATIONS);
    const backend: StationsBackend = { addStation: vi.fn(), updateStation: vi.fn(registry.updateStation), deleteStation: vi.fn(), selectStation: vi.fn() };
    const props = { controlToken: "tok", backend, inherited: {}, cameraSources: ["cam_head"], headset: null, onListing: vi.fn() };
    const { rerender } = render(<StationsCard listing={TWO_STATIONS} {...props} />);
    edit(0);
    fireEvent.change(field("Bench A", "Name"), { target: { value: "Bench Z" } });
    rerender(<StationsCard listing={{ ...TWO_STATIONS, locked_reason: "finish or discard the current take before changing the station in use" }} {...props} />);
    expect(field("Bench A", "Name").value).toBe("Bench Z");
    expect(button("Save").disabled).toBe(true);
    fireEvent.keyDown(field("Bench A", "Name"), { key: "Enter" });
    expect(backend.updateStation).not.toHaveBeenCalled();
  });
});

describe("the headset address in the Stations card", () => {
  it("is the last line of the editor of the station in use, and of no other station", () => {
    renderCard(TWO_STATIONS, { headset: HEADSET });
    expect(screen.queryByText(/Headset connects to/)).toBeNull(); // only the editor shows it
    edit(0);
    expect(editor("Bench A").lastElementChild?.textContent).toBe("Headset connects to 10.0.0.42");
    expect(screen.queryByText(/This host's address changed/)).toBeNull();
    edit(1);
    expect(screen.queryByText(/Headset connects to/)).toBeNull();
  });

  it("warns under the station in use that the headset needs the new address, and beside the address while editing", () => {
    renderCard(TWO_STATIONS, { headset: HEADSET_MOVED });
    const [bench, cart] = rows();
    const closed = within(bench).getByText(MOVED_WARNING);
    expect(closed.className).toContain("tone-warning");
    expect(within(cart).queryByText(MOVED_WARNING)).toBeNull();
    edit(0);
    const lines = Array.from(editor("Bench A").children).slice(-2);
    expect(lines.map((line) => line.textContent)).toEqual(["Headset connects to 10.0.0.42", MOVED_WARNING]);
    expect(lines[1].className).toContain("tone-warning");
    expect(screen.getAllByText(MOVED_WARNING)).toHaveLength(1); // said once, not in the row and the editor both
    fireEvent.click(button("Cancel"));
    expect(within(rows()[0]).getByText(MOVED_WARNING)).toBeTruthy();
  });

  it("says nothing while the address is not known", () => {
    renderCard(TWO_STATIONS, { headset: { address: null, saved: "10.0.0.7", changed: true } });
    edit(0);
    expect(screen.queryByText(/Headset connects to/)).toBeNull();
    expect(screen.queryByText(/This host's address changed/)).toBeNull();
    cleanup();
    renderCard(TWO_STATIONS, { headset: null });
    edit(0);
    expect(screen.queryByText(/Headset connects to/)).toBeNull();
  });

  it("belongs to no row while no station is in use", () => {
    renderCard({ ...TWO_STATIONS, selected: null }, { headset: HEADSET_MOVED });
    expect(screen.queryByText(MOVED_WARNING)).toBeNull();
    edit(0);
    expect(screen.queryByText(/Headset connects to/)).toBeNull();
  });
});
