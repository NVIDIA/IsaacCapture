// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
(() => {
  "use strict";

  const key = "__tactileMetadataEditorInteractions";
  globalThis[key]?.stop();

  let dragged = null;
  let nonce = 0;
  let observer = null;
  let infoInput = null;
  let infoElement = null;
  let raf = 0;

  const findBridge = (name) => {
    for (const input of document.querySelectorAll("input")) {
      let parent = input;
      for (let depth = 0; parent && depth < 5; ++depth, parent = parent.parentElement) {
        if (parent.textContent?.includes(name)) {
          input.style.position = "absolute";
          input.style.width = "1px";
          input.style.height = "1px";
          input.style.opacity = "0";
          input.style.pointerEvents = "none";
          input.tabIndex = -1;
          for (const label of parent.querySelectorAll("label")) {
            if (label.textContent?.includes(name)) {
              label.style.display = "none";
            }
          }
          return input;
        }
      }
    }
    return null;
  };

  const bridge = () => findBridge("__Tactile Reorder Event");

  const emit = (kind, source, target) => {
    const input = bridge();
    if (!input) return;
    const value = `${kind}:${source}:${target}:${++nonce}`;
    const setter = Object.getOwnPropertyDescriptor(
      HTMLInputElement.prototype,
      "value",
    )?.set;
    setter?.call(input, value);
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  };

  const alignBindingButtons = () => {
    const names = new Set([
      "Start Binding to Segments",
      "Select Segments",
      "Complete",
    ]);
    const buttons = [...document.querySelectorAll("button")].filter((button) =>
      names.has(button.textContent?.trim()),
    );
    if (buttons.length !== 3) return;
    let common = buttons[0].parentElement;
    while (common && !buttons.every((button) => common.contains(button))) {
      common = common.parentElement;
    }
    if (!common || common.querySelectorAll("button").length !== 3) return;
    const children = buttons.map((button) => {
      let child = button;
      while (child.parentElement !== common) child = child.parentElement;
      return child;
    });
    if (new Set(children).size !== 3) return;
    common.style.display = "grid";
    common.style.gridTemplateColumns = "repeat(3, minmax(0, 1fr))";
    common.style.gap = "6px";
    for (const child of common.children) {
      child.style.gridColumn = children.includes(child) ? "auto" : "1 / -1";
    }
    for (const button of buttons) button.style.width = "100%";
  };

  const syncInfo = () => {
    const parent = globalThis.__viserMutable?.canvas?.parentElement;
    if (!parent) return;
    if (!infoElement) {
      infoElement = document.createElement("div");
      Object.assign(infoElement.style, {
        position: "absolute",
        left: "12px",
        right: "12px",
        bottom: "12px",
        minHeight: "64px",
        maxHeight: "120px",
        overflow: "auto",
        zIndex: "20",
        padding: "10px 14px",
        border: "1px solid rgba(0,0,0,0.25)",
        borderRadius: "6px",
        background: "rgba(255,255,255,0.9)",
        color: "#202020",
        boxSizing: "border-box",
      });
    }
    if (infoElement.parentElement !== parent) parent.appendChild(infoElement);
    if (!infoInput?.isConnected) {
      infoInput = findBridge("__Tactile Info");
    }
    infoElement.innerHTML = `<strong>Info</strong><div>${infoInput?.value || "&nbsp;"}</div>`;
  };

  const frame = () => {
    syncInfo();
    raf = requestAnimationFrame(frame);
  };

  const bind = () => {
    bridge();
    findBridge("__Tactile Info");
    alignBindingButtons();
    for (const label of document.querySelectorAll("label")) {
      if (!label.textContent?.trim().startsWith("⚠")) continue;
      const field = label.parentElement?.querySelector("input, button");
      if (field) {
        field.style.outline = "1px solid #d32f2f";
        field.style.outlineOffset = "1px";
      }
    }
    for (const handle of document.querySelectorAll(".tm-sort-handle")) {
      if (handle.dataset.bound === "true") continue;
      handle.dataset.bound = "true";
      handle.style.cursor = "grab";
      handle.style.userSelect = "none";
      handle.addEventListener("dragstart", (event) => {
        dragged = {
          kind: handle.dataset.kind,
          row: Number(handle.dataset.row),
        };
        event.dataTransfer.effectAllowed = "move";
      });
      handle.addEventListener("dragover", (event) => {
        if (dragged?.kind === handle.dataset.kind) {
          event.preventDefault();
          event.dataTransfer.dropEffect = "move";
        }
      });
      handle.addEventListener("drop", (event) => {
        event.preventDefault();
        if (dragged?.kind === handle.dataset.kind) {
          emit(dragged.kind, dragged.row, Number(handle.dataset.row));
        }
        dragged = null;
      });
      handle.addEventListener("dragend", () => {
        dragged = null;
      });
    }
  };

  observer = new MutationObserver(bind);
  observer.observe(document.body, { childList: true, subtree: true });
  bind();
  frame();
  globalThis[key] = {
    stop() {
      observer?.disconnect();
      observer = null;
      cancelAnimationFrame(raf);
      infoElement?.remove();
      infoElement = null;
      infoInput = null;
      dragged = null;
    },
  };
})();
