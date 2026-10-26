// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/* Viser 1.1.1 event boundaries. Document mutations remain in the Python loop. */
(() => {
  "use strict";
  window.__tactileMetadataBridge?.dispose?.();
  const VERSION = "1.1.1";
  let settings, controls = new Map(), transforms = new Map();
  let sequence = 0, gestureCounter = 0, tail = Promise.resolve();
  let text = null, slider = null, transform = null, pointerDispatch = null;
  let stopped = false, cleaning = false;
  let announcedGeneration = -1;
  let navigatedGeneration = -1, navigating = false;
  let hovered = null;
  let pendingSignature = "";
  let restoringText = false, failureAlert = null;
  const baselines = new WeakMap();
  const boundElements = new Map();
  const incompatible = new Set();
  const listeners = new AbortController();
  const listen = (target, name, callback, capture = false) =>
    target.addEventListener(name, callback, { capture, signal: listeners.signal });
  let observer = null, wrappedPost = null;
  const fields = new WeakMap(), roots = new WeakMap();
  const originalPost = Worker.prototype.postMessage;
  const id = () => `${settings.client_id}:${++gestureCounter}`;
  const labelFor = (uuid) => Array.from(document.querySelectorAll("label[for]"))
    .filter((label) => label.htmlFor === uuid);
  const visibleInputs = (node) => Array.from(node.querySelectorAll("input,textarea"))
    .filter((input) => input.type !== "hidden");

  function send(event) {
    const frozen = { ...event, client_id: settings.client_id,
      token: settings.token, epoch: settings.epoch, seq: ++sequence };
    tail = tail.then(async () => {
      if (stopped) return;
      // A retry retains the exact sequence; an accepted message is never re-enqueued.
      frozen.revision ??= settings.revision;
      let response;
      for (let attempt = 0; attempt < 3; attempt++) {
        try {
          response = await fetch(settings.endpoint, {
            method: "POST", mode: "cors", cache: "no-store",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(frozen),
          });
          if (response.status === 503) continue;
          break;
        } catch (error) {
          if (attempt === 2) throw error;
        }
      }
      if (!response || !response.ok) throw new Error("Editor event was not acknowledged");
      const ack = await response.json();
      if (ack.seq !== frozen.seq) throw new Error("Editor acknowledgement is out of order");
      updateRevision(ack.revision, ack.generation);
      if (ack.error) console.error("Tactile editor:", ack.error);
      return ack;
    }).catch((error) => {
      if (stopped) return;
      stopped = true;
      console.error("Tactile editor bridge stopped:", error.message);
      const alert = document.createElement("div");
      alert.setAttribute("role", "alert");
      alert.textContent = "Tactile editor input is disabled because the local event bridge failed. Reload this page to reconnect.";
      Object.assign(alert.style, { position: "fixed", top: "0", left: "0", right: "0", zIndex: "2147483647",
        padding: "12px", background: "#8b2020", color: "white" });
      document.body.prepend(alert);
      failureAlert = alert;
      for (const item of controls.values()) {
        const element = document.getElementById(item.uuid);
        if (element) { element.setAttribute("disabled", "disabled"); element.style.pointerEvents = "none"; }
        const input = boundElements.get(item.uuid);
        if (input) input.disabled = true;
      }
    });
    return tail;
  }

  function message(control, phase, extra = {}) {
    return { control: control.uuid, generation: control.generation,
      phase, source: control.kind, ...extra };
  }

  function updateRevision(revision, generation) {
    if (!Number.isInteger(generation)) return;
    if (generation > (settings.revisionGeneration ?? -1)) {
      settings.revisionGeneration = generation;
      settings.revision = revision;
    } else if (generation === settings.revisionGeneration) {
      settings.revision = Math.max(settings.revision, revision);
    }
  }

  function bind() {
    const verified = new Set();
    const missing = [];
    const waitFor = (control, reason) => missing.push({ uuid: control.uuid, reason });
    for (const control of controls.values()) {
      if (control.kind === "action" || control.kind === "choice" || control.kind === "checkbox") {
        if (document.getElementById(control.uuid)) verified.add(control.uuid);
        else waitFor(control, "Registered native control has not mounted");
        continue;
      }
      if (control.kind === "slider") {
        const root = document.getElementById(control.uuid);
        if (!root) { waitFor(control, "Slider root UUID has not mounted"); continue; }
        const thumb = root.querySelector('[role="slider"]');
        const labels = labelFor(control.uuid);
        if (!thumb || labels.length !== 1) {
          waitFor(control, "Slider thumb or unique UUID label is unavailable"); continue;
        }
        let row = root;
        while (row && !row.contains(labels[0])) row = row.parentElement;
        if (!row) { waitFor(control, "Slider and UUID label have no shared row"); continue; }
        const number = visibleInputs(row).filter((node) => !root.contains(node));
        if (number.length !== 1) {
          reportIncompatible(control, "Slider companion input differs from Viser 1.1.1");
          continue;
        }
        roots.set(root, control);
        fields.set(number[0], { ...control, kind: "slider_number" });
        number[0].disabled = control.disabled;
        boundElements.set(control.uuid, number[0]);
        verified.add(control.uuid);
      } else {
        // TextInput.tsx omits id; common.tsx supplies the exact label htmlFor UUID.
        const labels = labelFor(control.uuid);
        if (labels.length !== 1) { waitFor(control, "Expected one UUID label; found " + labels.length); continue; }
        let row = labels[0].parentElement;
        while (row && visibleInputs(row).length === 0) row = row.parentElement;
        if (!row) { waitFor(control, "UUID label has no mounted input"); continue; }
        const inputs = visibleInputs(row);
        if (inputs.length !== 1 || row.querySelectorAll("label[for]").length !== 1) {
          reportIncompatible(control, "Registered text input has ambiguous DOM ownership");
          continue;
        }
        fields.set(inputs[0], control);
        boundElements.set(control.uuid, inputs[0]);
        verified.add(control.uuid);
      }
    }
    if (controls.size > 0 && verified.size === controls.size && announcedGeneration !== settings.generation
        && incompatible.size === 0) {
      const tab = Array.from(document.querySelectorAll('[role="tab"][id]'))
        .filter((element) => element.id.endsWith(`-tab-${settings.active_tab}`));
      if (tab.length !== 1) missing.push({ uuid: settings.active_tab, reason: "Registered tab has not uniquely mounted" });
      else if (navigatedGeneration !== settings.generation) {
        navigating = true;
        tab[0].click();
        navigating = false;
        navigatedGeneration = settings.generation;
      }
      if (tab.length === 1) {
        announcedGeneration = settings.generation;
        send({ phase: "controls_ready", generation: settings.generation });
      }
    }
    const signature = JSON.stringify([settings.generation, missing]);
    if (Number.isInteger(settings.generation) && signature !== pendingSignature) {
      pendingSignature = signature;
      send({ phase: "controls_pending", generation: settings.generation, missing });
    }
  }

  function reportIncompatible(control, reason) {
    if (incompatible.has(control.uuid)) return;
    incompatible.add(control.uuid);
    send(message(control, "incompatible", { reason }));
  }

  function setText(input, value) {
    const prototype = input instanceof HTMLTextAreaElement
      ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(prototype, "value").set.call(input, value);
    restoringText = true;
    try { input.dispatchEvent(new Event("input", { bubbles: true })); }
    finally { restoringText = false; }
  }

  function focusText(control, input, original = input.value) {
    baselines.set(input, original);
    text = { control, input, original, confirmed: original, id: id(), dirty: false };
    send(message(control, "focus", { gesture_id: text.id }));
  }

  function confirmText(retainFocus = false) {
    if (!text) return;
    const pending = text;
    text = null;
    const value = pending.input.value;
    if (value !== pending.confirmed) {
      const nextIdentity = retainFocus ? id() : null;
      const completion = send(message(pending.control, "commit", {
        gesture_id: pending.id, value, retain_focus: retainFocus,
        next_gesture_id: nextIdentity,
      }));
      if (retainFocus) {
        text = { ...pending, confirmed: value, id: nextIdentity, dirty: false };
        completion.then((ack) => {
          if (ack && !ack.error) {
            baselines.set(pending.input, value);
            if (text?.control.uuid === pending.control.uuid && text.control.generation === pending.control.generation) text.original = value;
          }
        });
      }
    } else if (retainFocus) {
      text = pending;
    } else {
      send(message(pending.control, "release", { gesture_id: pending.id }));
    }
  }

  function cancel(reason) {
    if (text) {
      const pending = text;
      text = null;
      setText(pending.input, pending.original);
      baselines.set(pending.input, pending.original);
      send(message(pending.control, "cancel", { gesture_id: pending.id, reason }));
    }
    if (slider) {
      send(message(slider.control, "cancel", { gesture_id: slider.id, reason }));
      slider = null;
    }
    if (transform) {
      const active = transform;
      active.cancelled = true;
      send(message(active.control, "cancel", { gesture_id: active.id, reason }));
      // Drei's pointerup restores camera controls; pointercancel only clears hover.
      cleaning = true;
      try {
        active.canvas.dispatchEvent(new PointerEvent("pointerup", {
          bubbles: true, pointerId: active.pointerId, pointerType: active.pointerType,
          clientX: active.clientX, clientY: active.clientY, button: 0, buttons: 0,
        }));
      } finally {
        cleaning = false;
        transform = null;
      }
    }
  }

  function findRoot(target) {
    for (let node = target; node instanceof Element; node = node.parentElement) {
      const control = roots.get(node);
      if (control) return { control, root: node };
    }
    return null;
  }

  function install() {
    listen(document, "focusin", (event) => {
      bind();
      const control = fields.get(event.target);
      if (!control || control.disabled || stopped) return;
      if (text && text.input !== event.target) confirmText();
      if (!text) focusText(control, event.target);
    }, true);
    listen(document, "input", (event) => {
      if (restoringText || stopped) return;
      if (!text) {
        const control = fields.get(event.target);
        if (control && !control.disabled) focusText(control, event.target, baselines.get(event.target) ?? event.target.value);
      }
      if (text && text.input === event.target && !text.dirty) {
        text.dirty = true;
        send(message(text.control, "dirty", { gesture_id: text.id }));
      }
    }, true);
    listen(document, "blur", (event) => {
      if (text && text.input === event.target) {
        if (document.hasFocus()) confirmText();
        else cancel("window-blur");
      }
    }, true);
    listen(document, "keydown", (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
        event.preventDefault(); event.stopImmediatePropagation();
        cancel("history");
        send({ phase: event.shiftKey ? "redo" : "undo" });
      } else if (event.key === "Escape") {
        event.preventDefault(); event.stopImmediatePropagation(); cancel("escape");
      } else if (event.key === "Enter" && text && !event.shiftKey) {
        event.preventDefault(); event.stopImmediatePropagation(); confirmText(true);
      } else if (["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End", "PageUp", "PageDown"].includes(event.key)) {
        const found = findRoot(event.target);
        if (found && !fields.has(event.target) && !slider) {
          confirmText();
          slider = { ...found, id: id(), key: event.key };
          send(message(found.control, "begin", { gesture_id: slider.id }));
        }
      }
    }, true);
    listen(document, "keyup", (event) => {
      if (slider?.key === event.key) finishSlider(slider);
    }, true);
    listen(document, "click", (event) => {
      const tab = event.target.closest?.('[role="tab"][id]');
      if (tab && !navigating) {
        const entry = Object.entries(settings.tabs || {}).find(([uuid]) => tab.id.endsWith(`-tab-${uuid}`));
        if (entry) { confirmText(); send({ phase: "tab", value: entry[1] }); }
      }
      const element = event.target.closest?.("button[id]");
      const control = element && controls.get(element.id);
      if (!control || control.kind !== "action") return;
      event.preventDefault(); event.stopImmediatePropagation();
      if (element.disabled || stopped) return;
      confirmText();
      if (slider || transform) cancel("action");
      send(message(control, "action"));
    }, true);
    listen(document, "pointerdown", (event) => {
      pointerDispatch = event;
      if (transform && event.pointerId !== transform.pointerId) {
        event.preventDefault(); event.stopImmediatePropagation(); return;
      }
      bind();
      if (fields.has(event.target)) return;
      const found = findRoot(event.target);
      if (!found || stopped) return;
      confirmText();
      slider = { ...found, id: id(), pointerId: event.pointerId };
      send(message(found.control, "begin", { gesture_id: slider.id }));
    }, true);
    listen(document, "pointermove", (event) => {
      if (transform && transform.pointerId === event.pointerId) {
        transform.clientX = event.clientX; transform.clientY = event.clientY;
      }
    }, true);
    listen(document, "pointerover", (event) => {
      const control = fields.get(event.target) || findRoot(event.target)?.control;
      const next = control?.hover ? control : null;
      if (hovered?.uuid === next?.uuid) return;
      if (hovered) send(message(hovered, "hover", { value: false }));
      hovered = next;
      if (hovered) send(message(hovered, "hover", { value: true }));
    }, true);
    listen(document, "pointerup", (event) => {
      if (!slider || slider.pointerId !== event.pointerId) return;
      const active = slider;
      finishSlider(active);
    }, true);
    listen(document, "pointercancel", () => cancel("pointercancel"), true);
    listen(window, "blur", () => cancel("window-blur"));
    listen(window, "pagehide", () => cancel("disconnect"));
    observer = new MutationObserver(bind);
    observer.observe(document.body, { childList: true, subtree: true });

    wrappedPost = function () {
      const packet = arguments[0], outgoing = packet?.type === "send" && packet.message;
      if (outgoing && !stopped) {
        if (outgoing.type === "GuiUpdateMessage") {
          const control = controls.get(outgoing.uuid);
          if (control && (control.kind === "choice" || control.kind === "checkbox")) {
            confirmText();
            send(message(control, "commit", { value: outgoing.updates.value }));
          } else if (control?.kind === "slider" && slider?.control.uuid === control.uuid) {
            send(message(control, "update", { gesture_id: slider.id, value: outgoing.updates.value }));
          }
        }
        const control = transforms.get(outgoing.name);
        if (control && outgoing.owner === String(settings.client_id)) {
          if (outgoing.type === "TransformControlsDragStartMessage") {
            const event = pointerDispatch;
            if (!event || event.type !== "pointerdown" || event.eventPhase === Event.NONE
                || !(event.target instanceof HTMLCanvasElement)) {
              send(message(control, "incompatible", { reason: "No synchronous canvas pointer for transform start" }));
            } else {
              confirmText();
              transform = { control, id: id(), pointerId: event.pointerId,
                pointerType: event.pointerType, canvas: event.target,
                clientX: event.clientX, clientY: event.clientY, cancelled: false, value: null };
              send(message(control, "begin", { gesture_id: transform.id }));
            }
          } else if (outgoing.type === "TransformControlsUpdateMessage" && transform?.control.uuid === control.uuid) {
            transform.value = [...outgoing.position, ...outgoing.wxyz.slice(1), outgoing.wxyz[0]];
            if (!transform.cancelled) send(message(control, "update", { gesture_id: transform.id, value: transform.value }));
          } else if (outgoing.type === "TransformControlsDragEndMessage" && transform?.control.uuid === control.uuid) {
            if (!transform.cancelled && !cleaning) send(message(control, "end", { gesture_id: transform.id, value: transform.value }));
            transform = null;
          }
        }
      }
      return Reflect.apply(originalPost, this, arguments);
    };
    Worker.prototype.postMessage = wrappedPost;
  }

  function finishSlider(active) {
    // Paint follows a known release/key-up; it never defines a gesture boundary.
    requestAnimationFrame(() => {
      if (slider !== active) return;
      const raw = active.root.querySelector('[role="slider"]')?.getAttribute("aria-valuenow");
      if (raw === null || raw === undefined || !Number.isFinite(Number(raw))) {
        cancel("slider-value-unavailable");
        return;
      }
      send(message(active.control, "end", { gesture_id: active.id, value: Number(raw) }));
      slider = null;
    });
  }

  window.__tactileMetadataBridge = {
    start(config) {
      if (settings || config.version !== VERSION) throw new Error("Unsupported tactile bridge version");
      settings = config;
      install(); send({ phase: "ready" });
    },
    configure(config) {
      if (!settings || stopped) return;
      if (config.generation < settings.generation) return;
      updateRevision(config.revision, config.generation);
      settings.epoch = config.epoch;
      if (settings.generation !== config.generation) {
        incompatible.clear();
        boundElements.clear();
        text = null;
      }
      settings.generation = config.generation;
      settings.tabs = config.tabs;
      settings.active_tab = config.active_tab;
      controls = new Map(config.controls.map((item) => [item.uuid, item]));
      transforms = new Map(config.transforms.map((item) => [item.name, item]));
      bind();
    },
    cancel() { cancel("server-refresh"); },
    dispose() {
      stopped = true;
      cancel("dispose");
      listeners.abort();
      observer?.disconnect();
      failureAlert?.remove();
      if (Worker.prototype.postMessage === wrappedPost) Worker.prototype.postMessage = originalPost;
    },
  };
})();
