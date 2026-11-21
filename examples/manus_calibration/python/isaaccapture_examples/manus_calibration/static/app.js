// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

"use strict";

// Every piece of text from the server is assigned with textContent, never as markup.

const STAGE_TEXT = {
  idle: "Ready.",
  starting: "Starting the calibration worker...",
  discovering: "Looking for the requested gloves...",
  awaiting_confirmation: "Waiting for your confirmation.",
  sampling: "The SDK is sampling. Follow the instruction below.",
  sampling_estimate:
    "The SDK started this step. Keep following the instruction; the bar is a time estimate, not a sign the step finished.",
  saving: "Saving the calibration files...",
  stopping: "Stopping...",
  finished: "Finished.",
};
const RESULT_HEADING = {
  completed: "Calibration saved",
  cancelled: "Calibration cancelled",
  failed: "Calibration failed",
  uncertain: "Outcome uncertain",
};

const $ = (id) => document.getElementById(id);

// One id per browser tab: it names the tab as the owner of an attempt, and survives a reload.
const clientId = (() => {
  let id = sessionStorage.getItem("manusCalibrationClient");
  if (!id) {
    const bytes = crypto.getRandomValues(new Uint8Array(16));
    id = Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
    sessionStorage.setItem("manusCalibrationClient", id);
  }
  return id;
})();

let state = null;
let requestInFlight = false;
let cancelInFlight = false;
let localError = "";

async function post(path, body) {
  const isCancel = path === "/api/cancel";
  requestInFlight = !isCancel || requestInFlight;
  cancelInFlight = isCancel || cancelInFlight;
  localError = "";
  render();
  try {
    const response = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Client-Id": clientId },
      body: JSON.stringify(body),
    });
    const result = await response.json();
    if (!result.ok) {
      localError = result.error || "the request was refused";
    }
  } catch (error) {
    localError = "cannot reach the server";
  } finally {
    if (isCancel) {
      cancelInFlight = false;
    } else {
      requestInFlight = false;
    }
    render();
  }
}

function selectedSides() {
  const value = document.querySelector('input[name="sides"]:checked').value;
  return value === "both" ? ["left", "right"] : [value];
}

// A continuous step the worker holds for the SDK's estimate after the step starts.
function windowContinuous(prompt) {
  return Boolean(prompt && prompt.continuous && state.continuous_pacing === "window");
}

function guidance(prompt) {
  if (prompt.mode === "motion") {
    return `Confirm, then perform the movement for ${prompt.duration_s.toFixed(1)} seconds.`;
  }
  if (windowContinuous(prompt)) {
    if (state.confirm_pending || requestInFlight) {
      // Do not make the operator wait for the SDK before moving: the call may take a while to return.
      return "Follow the instruction now. Waiting for the SDK to start the step.";
    }
    return `Confirm, then begin the instructed movement immediately and continue for about ${Math.round(prompt.estimated_s)} seconds.`;
  }
  if (prompt.continuous) {
    return `Get into the pose (or start the movement) for about ${Math.round(prompt.estimated_s)} seconds, then confirm.`;
  }
  return "Hold the pose first, then confirm to capture it.";
}

function confirmLabel(prompt) {
  if (state.confirm_pending || requestInFlight) {
    return "Starting...";
  }
  if (windowContinuous(prompt)) {
    return `Start step (~${Math.round(prompt.estimated_s)} s)`;
  }
  return prompt.mode === "motion" ? `Start motion (${Math.round(prompt.duration_s)} s)` : "Capture pose";
}

function renderStatus() {
  if (state.stage === "idle") {
    $("stage").textContent = "Ready.";
  } else if (state.active && !state.owner) {
    $("stage").textContent =
      "Another browser is running a calibration; this page is read-only until it finishes.";
  } else {
    const timedEstimate = state.stage === "sampling" && windowContinuous(state.prompt);
    $("stage").textContent = timedEstimate
      ? STAGE_TEXT.sampling_estimate
      : STAGE_TEXT[state.stage] || state.stage;
  }
  $("gloves").textContent = state.gloves.length
    ? "Gloves found: " + state.gloves.map((g) => `${g.side} (id ${g.glove_id})`).join(", ")
    : "";
  $("notice").textContent = localError || (state.stage === "finished" ? "" : state.notice);
}

function renderStep() {
  const prompt = state.prompt;
  $("step").hidden = !prompt;
  if (!prompt) {
    return;
  }
  const side = prompt.side.charAt(0).toUpperCase() + prompt.side.slice(1);
  $("step-heading").textContent = `${side} glove, step ${prompt.index + 1} of ${prompt.count}`;
  $("step-title").textContent = prompt.title;
  $("step-instruction").textContent = prompt.instruction;
  $("step-guidance").textContent = guidance(prompt);
  const sampling = state.stage === "sampling" && (prompt.mode === "motion" || windowContinuous(prompt));
  $("progress").hidden = !sampling;
  const window_s = prompt.continuous ? prompt.estimated_s : prompt.duration_s;
  if (sampling && window_s > 0 && state.elapsed_s !== null) {
    $("progress").value = Math.min(100, (100 * state.elapsed_s) / window_s);
  }
}

function renderControls() {
  const prompt = state.prompt;
  // Other browsers watch; only the owner gets controls.
  $("start").disabled = state.active || requestInFlight;
  for (const radio of document.querySelectorAll('input[name="sides"]')) {
    radio.disabled = state.active;
  }
  $("actions").hidden = !(state.active && state.owner);
  $("confirm").hidden = !prompt;
  if (prompt) {
    $("confirm").textContent = confirmLabel(prompt);
    $("confirm").disabled = !(
      state.stage === "awaiting_confirmation" && prompt.token && !state.confirm_pending && !requestInFlight
    );
  }
  // Cancel stays available while a confirm is in flight: that is when a stuck step needs it.
  $("cancel").disabled = state.stage === "stopping" || cancelInFlight;
}

function renderResult() {
  const outcome = state.outcome;
  $("result").hidden = !outcome || state.active;
  if (!outcome || state.active) {
    return;
  }
  $("result-heading").textContent = RESULT_HEADING[outcome.kind] || outcome.kind;
  $("result-message").textContent = outcome.message;
  $("result-dir").hidden = !outcome.directory;
  $("result-dir").textContent = outcome.directory ? `Saved on the server host in ${outcome.directory}:` : "";
  const list = $("result-files");
  list.replaceChildren(
    ...outcome.files.map((file) => {
      const item = document.createElement("li");
      // Same-origin download of the file the worker just wrote on the server host.
      const link = document.createElement("a");
      link.href = `/download/${encodeURIComponent(file.file)}`;
      link.download = file.file;
      link.textContent = file.file;
      item.append(`${file.side}: `, link, ` (${file.bytes} bytes)`);
      return item;
    }),
  );
  if (!outcome.directory) {
    const none = document.createElement("li");
    none.textContent = "No calibration files were saved.";
    list.replaceChildren(none);
  }
  $("result-log").hidden = outcome.log_tail.length === 0;
  $("result-log").textContent = outcome.log_tail.join("\n");
}

function render() {
  if (!state) {
    return;
  }
  $("out-dir").textContent = state.out_dir;
  renderStatus();
  renderStep();
  renderControls();
  renderResult();
}

$("start").addEventListener("click", () => post("/api/start", { sides: selectedSides() }));
$("confirm").addEventListener("click", () => {
  if (state && state.prompt && state.prompt.token) {
    post("/api/confirm", { token: state.prompt.token });
  }
});
$("cancel").addEventListener("click", () => post("/api/cancel", {}));

const events = new EventSource(`/events?client=${clientId}`);
let defaultApplied = false;
events.onmessage = (event) => {
  state = JSON.parse(event.data);
  if (!defaultApplied) {
    // The launch option preselects the gloves; the operator can still change it.
    const radio = document.querySelector(`input[name="sides"][value="${state.default_sides}"]`);
    if (radio) {
      radio.checked = true;
    }
    defaultApplied = true;
  }
  render();
};
events.onerror = () => {
  $("notice").textContent = "Connection lost; retrying...";
};
