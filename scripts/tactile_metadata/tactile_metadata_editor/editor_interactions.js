// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
(() => {
  "use strict";

  const key = "__tactileMetadataEditorInteractions";
  globalThis[key]?.stop();

  let observer = null;
  let infoInput = null;
  let infoElement = null;
  let raf = 0;
  const numberDrafts = new Map();
  const nativeInputValueSetter = Object.getOwnPropertyDescriptor(
    HTMLInputElement.prototype,
    "value",
  )?.set;

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

  // Place Viser's logo above the fixed bottom Info area.
  const positionViserLogo = () => {
    const logo = document.querySelector('a[title="About Viser"]');
    if (!logo) return;
    logo.style.top = "auto";
    logo.style.left = "16px";
    logo.style.bottom = "144px";
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
    for (const [input, draft] of numberDrafts) {
      if (!input.isConnected || document.activeElement !== input) {
        numberDrafts.delete(input);
      } else if (input.value !== draft) {
        nativeInputValueSetter?.call(input, draft);
        input.dispatchEvent(new Event("input", { bubbles: true }));
        input.setSelectionRange(draft.length, draft.length);
      }
    }
    syncInfo();
    raf = requestAnimationFrame(frame);
  };

  const boldChoiceLabels = () => {
    for (const label of document.querySelectorAll("label")) {
      if (["Edit", "Preview", "Editor", "Live"].includes(label.textContent?.trim())) {
        label.style.fontWeight = "700";
      }
    }
  };

  const folderFromLabel = (name) => {
    const label = [...document.querySelectorAll("div")].find(
      (element) => element.textContent?.trim() === name && element.querySelector("svg"),
    );
    return label?.parentElement || null;
  };

  const directChild = (parent, descendant) => {
    let child = descendant;
    while (child?.parentElement && child.parentElement !== parent) {
      child = child.parentElement;
    }
    return child?.parentElement === parent ? child : null;
  };

  const freezeMainMenu = () => {
    const general = folderFromLabel("General");
    const preview = folderFromLabel("Preview");
    const tabNames = new Set(["URDF", "Segments", "Taxels"]);
    const tabs = [...document.querySelectorAll("button")].filter((button) =>
      tabNames.has(button.textContent?.trim()),
    );
    if (!general || tabs.length !== 3) return;
    const tabGroup = tabs[0].closest("[data-dock-group]");
    if (!tabGroup) return;
    let root = general.parentElement;
    while (root && !root.contains(tabGroup)) root = root.parentElement;
    if (!root) return;
    const generalChild = directChild(root, general);
    const tabsChild = directChild(root, tabGroup);
    const previewChild = preview ? directChild(root, preview) : null;
    if (!generalChild || !tabsChild) return;

    Object.assign(root.style, {
      display: "flex",
      flexDirection: "column",
      height: "100%",
      minHeight: "0",
      overflow: "hidden",
    });
    Object.assign(generalChild.style, {
      flex: "0 0 auto",
      zIndex: "30",
    });
    Object.assign(tabsChild.style, {
      flex: "1 1 0",
      minHeight: "0",
      overflow: "hidden",
    });
    if (previewChild) {
      Object.assign(previewChild.style, {
        flex: "1 1 0",
        minHeight: "0",
        overflowY: "auto",
      });
    }
  };

  const preservePartialNumbers = () => {
    for (const input of document.querySelectorAll("input")) {
      if (input.dataset.tactileNumberDraft === "true") continue;
      if (!["decimal", "numeric"].includes(input.inputMode) && input.type !== "number") {
        continue;
      }
      input.dataset.tactileNumberDraft = "true";
      input.addEventListener("input", () => {
        const value = input.value;
        if (/^[+-]?(?:\d+)?\.$/.test(value)) {
          numberDrafts.set(input, value);
        } else {
          numberDrafts.delete(input);
        }
      });
      input.addEventListener("blur", () => numberDrafts.delete(input));
    }
  };

  const bind = () => {
    findBridge("__Tactile Info");
    positionViserLogo();
    boldChoiceLabels();
    freezeMainMenu();
    preservePartialNumbers();
    for (const label of document.querySelectorAll("label")) {
      if (!label.textContent?.trim().startsWith("⚠")) continue;
      const field = label.parentElement?.querySelector("input, button");
      if (field) {
        field.style.outline = "1px solid #d32f2f";
        field.style.outlineOffset = "1px";
      }
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
      numberDrafts.clear();
    },
  };
})();
