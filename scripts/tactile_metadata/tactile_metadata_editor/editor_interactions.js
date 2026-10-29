// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
(() => {
  "use strict";

  const key = "__tactileMetadataEditorInteractions";
  globalThis[key]?.stop();

  let dragged = null;
  let nonce = 0;
  let observer = null;

  const bridge = () => {
    for (const input of document.querySelectorAll("input")) {
      let parent = input;
      for (let depth = 0; parent && depth < 5; ++depth, parent = parent.parentElement) {
        if (parent.textContent?.includes("__Tactile Reorder Event")) {
          if (parent.parentElement) parent.parentElement.style.display = "none";
          return input;
        }
      }
    }
    return null;
  };

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

  const bind = () => {
    bridge();
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
  globalThis[key] = {
    stop() {
      observer?.disconnect();
      observer = null;
      dragged = null;
    },
  };
})();
