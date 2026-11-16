// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/** The Headset address card: which address it shows, remembering it per station, and warning when it is stale. */
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { HeadsetAddressCard, type HeadsetAddressBackend } from "./HeadsetAddressCard";
import { SettingsHttpError } from "./settingsApi";
import { HEADSET_ADDRESS_FIXTURE, STATIONS_FIXTURE } from "./settingsTestDoubles";
import type { HeadsetAddress, StationRecord } from "./stationsApi";

afterEach(cleanup);

const BENCH: StationRecord = {
  id: "st_1",
  name: "Bench A",
  camera_type: null,
  camera_roles: { ego: "cam_head", wrist_left: null, wrist_right: null },
  problems: [],
  headset_address: null,
};

const TWO_NETWORKS: HeadsetAddress = {
  candidates: [
    { address: "10.0.0.42", interface: "wlp2s0", kind: "wireless" },
    { address: "192.168.1.10", interface: "enp3s0", kind: "wired" },
  ],
  saved: null,
  changed: false,
};

function renderCard(
  answer: HeadsetAddress | Error = HEADSET_ADDRESS_FIXTURE,
  props: { controlToken?: string | null; station?: StationRecord | null } = {},
) {
  const backend: HeadsetAddressBackend = {
    loadHeadsetAddress: vi.fn(async () => {
      if (answer instanceof Error) throw answer;
      return answer;
    }),
    updateStation: vi.fn(async () => STATIONS_FIXTURE),
  };
  const onListing = vi.fn();
  const view = (station: StationRecord | null) => (
    <HeadsetAddressCard controlToken={props.controlToken === undefined ? "tok" : props.controlToken} station={station} backend={backend} onListing={onListing} />
  );
  const utils = render(view(props.station === undefined ? BENCH : props.station));
  return { backend, onListing, rerender: (station: StationRecord | null) => utils.rerender(view(station)) };
}

const address = () => screen.findByLabelText("Address to enter in the headset");
const mark = () => screen.getByRole("button", { name: "Mark as entered in the headset" }) as HTMLButtonElement;

describe("HeadsetAddressCard", () => {
  it("shows the host's address and where it comes from", async () => {
    const { backend } = renderCard();
    expect((await address()).textContent).toBe("10.0.0.42");
    expect(screen.getByText("10.0.0.42 (wlp2s0, wireless)")).toBeTruthy();
    expect(screen.queryByLabelText("Network to use")).toBeNull(); // one candidate: nothing to choose
    expect(backend.loadHeadsetAddress).toHaveBeenCalledWith("tok", expect.any(AbortSignal));
  });

  it("offers a choice of networks, wireless first, and saves the one entered with the station in use", async () => {
    const { backend, onListing } = renderCard(TWO_NETWORKS);
    expect((await address()).textContent).toBe("10.0.0.42");
    const chooser = screen.getByLabelText("Network to use") as HTMLSelectElement;
    expect(Array.from(chooser.options).map((option) => option.textContent)).toEqual(["10.0.0.42 (wlp2s0, wireless)", "192.168.1.10 (enp3s0, wired)"]);
    fireEvent.change(chooser, { target: { value: "192.168.1.10" } });
    expect((await address()).textContent).toBe("192.168.1.10");
    fireEvent.click(mark());
    await waitFor(() => expect(onListing).toHaveBeenCalledWith(STATIONS_FIXTURE));
    expect(backend.updateStation).toHaveBeenCalledWith("tok", "st_1", { headset_address: "192.168.1.10" });
  });

  it("confirms an address already entered, and has nothing left to mark", async () => {
    renderCard(TWO_NETWORKS, { station: { ...BENCH, headset_address: "10.0.0.42" } });
    expect(await screen.findByText("✓ Entered in the headset for Bench A.")).toBeTruthy();
    expect(mark().disabled).toBe(true);
    // Choosing the other network makes it markable again: the headset would need that address instead.
    fireEvent.change(screen.getByLabelText("Network to use"), { target: { value: "192.168.1.10" } });
    expect(mark().disabled).toBe(false);
    expect(screen.queryByText(/Entered in the headset for/)).toBeNull();
  });

  it("warns when the saved address is no longer one of the host's", async () => {
    renderCard({ ...HEADSET_ADDRESS_FIXTURE, saved: "10.0.0.7", changed: true }, { station: { ...BENCH, headset_address: "10.0.0.7" } });
    expect(await screen.findByText(/The address saved for Bench A \(10\.0\.0\.7\) is no longer one of this host's/)).toBeTruthy();
    expect(mark().disabled).toBe(false);
  });

  it("names a capture host on another machine as such", async () => {
    renderCard({ candidates: [{ address: "192.168.1.50", interface: null, kind: "capture_server" }], saved: null, changed: false });
    expect(await screen.findByText("192.168.1.50 (capture host)")).toBeTruthy();
  });

  it("says so when the host has no address, and looks again on request", async () => {
    const { backend } = renderCard({ candidates: [], saved: null, changed: false });
    expect(await screen.findByText(/No network address was found for this host/)).toBeTruthy();
    expect(mark().disabled).toBe(true);
    fireEvent.click(screen.getByRole("button", { name: "Detect again" }));
    await waitFor(() => expect(backend.loadHeadsetAddress).toHaveBeenCalledTimes(2));
  });

  it("cannot remember an address with no station in use", async () => {
    renderCard(HEADSET_ADDRESS_FIXTURE, { station: null });
    expect(await screen.findByText("Put a station in use to remember which address its headset was given.")).toBeTruthy();
    expect(mark().disabled).toBe(true);
  });

  it("reads the address again when the station's saved one changes, keeping the operator's choice", async () => {
    const { backend, rerender } = renderCard(TWO_NETWORKS);
    await address();
    fireEvent.change(screen.getByLabelText("Network to use"), { target: { value: "192.168.1.10" } });
    rerender({ ...BENCH, headset_address: "192.168.1.10" });
    await waitFor(() => expect(backend.loadHeadsetAddress).toHaveBeenCalledTimes(2));
    expect((await address()).textContent).toBe("192.168.1.10");
    expect(await screen.findByText("✓ Entered in the headset for Bench A.")).toBeTruthy();
  });

  it("asks for the control token instead of loading without it", () => {
    const { backend } = renderCard(HEADSET_ADDRESS_FIXTURE, { controlToken: null });
    expect(screen.getByText("Enter the control token in the recording panel to see this host's addresses.")).toBeTruthy();
    expect(backend.loadHeadsetAddress).not.toHaveBeenCalled();
  });

  it("shows why the addresses could not be read, and why a save was refused", async () => {
    renderCard(new SettingsHttpError("control token rejected"));
    expect(await screen.findByText("Addresses unavailable: control token rejected")).toBeTruthy();
    cleanup();
    const { backend } = renderCard();
    backend.updateStation = vi.fn(async () => {
      throw new SettingsHttpError("finish or discard the current take before changing the station in use");
    });
    await address();
    fireEvent.click(mark());
    expect(await screen.findByText("Not saved: finish or discard the current take before changing the station in use")).toBeTruthy();
  });
});
