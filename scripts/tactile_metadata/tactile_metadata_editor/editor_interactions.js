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
    const label = [...document.querySelectorAll("label")].find(
      (candidate) => candidate.textContent?.trim() === name,
    );
    let input = label?.htmlFor ? document.getElementById(label.htmlFor) : null;
    let row = label?.parentElement;
    while (!input && row) {
      input = row.querySelector("input");
      row = row.parentElement;
    }
    if (!(input instanceof HTMLInputElement)) return null;
    input.style.position = "absolute";
    input.style.width = "1px";
    input.style.height = "1px";
    input.style.opacity = "0";
    input.style.pointerEvents = "none";
    input.tabIndex = -1;
    label.style.display = "none";
    return input;
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

  const directChild = (parent, descendant) => {
    let child = descendant;
    while (child?.parentElement && child.parentElement !== parent) {
      child = child.parentElement;
    }
    return child?.parentElement === parent ? child : null;
  };

  const layoutSelectedNames = () => {
    for (const label of document.querySelectorAll("label")) {
      if (label.textContent?.trim() !== "__Selected Object Names") continue;
      let row = label.parentElement;
      while (row && !row.querySelector("textarea")) row = row.parentElement;
      if (!row) continue;
      const textarea = row.querySelector("textarea");
      const labelColumn = directChild(row, label);
      const valueColumn = directChild(row, textarea);
      if (!labelColumn || !valueColumn) continue;
      labelColumn.style.display = "none";
      valueColumn.style.flex = "1 1 100%";
      valueColumn.style.width = "100%";
      row.style.width = "100%";
      if (textarea.dataset.tactileFixedRows !== "true") {
        const style = getComputedStyle(textarea);
        const fontSize = Number.parseFloat(style.fontSize);
        const lineHeight =
          Number.parseFloat(style.lineHeight) || (Number.isFinite(fontSize) ? fontSize * 1.5 : 20);
        const chrome = ["paddingTop", "paddingBottom", "borderTopWidth", "borderBottomWidth"]
          .map((name) => Number.parseFloat(style[name]) || 0)
          .reduce((sum, value) => sum + value, 0);
        textarea.rows = 2;
        textarea.style.height = `${Math.ceil(2 * lineHeight + chrome)}px`;
        textarea.style.overflowY = "auto";
        textarea.dataset.tactileFixedRows = "true";
      }
    }
  };

  const layoutTaxelBindings = () => {
    const names = [
      "Segment 0",
      "Segment 1",
      "Segment 2",
      "Segment 3",
      "Weight 0",
      "Weight 1",
      "Weight 2",
      "Weight 3",
    ];
    const clean = (label) => label.textContent?.trim().replace(/^⚠\s*/, "");
    for (const first of document.querySelectorAll("label")) {
      if (clean(first) !== names[0]) continue;
      let container = first.parentElement;
      let fields = [];
      while (container) {
        const labels = [...container.querySelectorAll("label")];
        fields = names.map((name) => labels.find((label) => clean(label) === name));
        if (fields.every(Boolean)) break;
        container = container.parentElement;
      }
      if (!container) continue;
      const children = fields.map((field) => directChild(container, field));
      if (children.some((child) => !child) || new Set(children).size !== names.length) {
        continue;
      }
      const title = container.querySelector("[data-tactile-binding-title]");
      const titleChild = directChild(container, title);
      if (!title || !titleChild) continue;
      const bindingChildren = new Set(children);
      Object.assign(container.style, {
        display: "grid",
        gridTemplateColumns: "6em repeat(4, minmax(0, 1fr))",
        gap: "0",
        alignItems: "end",
      });
      // Only bindings use the four columns; all other taxel controls stay full-width.
      for (const child of container.children) {
        if (child === titleChild) {
          child.style.gridColumn = "1";
          child.style.gridRow = "span 2";
          child.style.alignSelf = "center";
          child.style.margin = "0";
          child.style.padding = "0 0.5em";
        } else if (bindingChildren.has(child)) {
          child.style.gridColumn = "auto";
          child.style.minWidth = "0";
          child.style.margin = "0";
          child.style.padding = "0";
        } else {
          child.style.gridColumn = "1 / -1";
        }
      }
      Object.assign(title.style, {
        color: "var(--mantine-color-dimmed)",
        fontSize: "0.875em",
        fontWeight: "450",
      });
      for (const field of fields) {
        let row = field.parentElement;
        while (row && !row.querySelector("input, button")) row = row.parentElement;
        if (!row) continue;
        const control = row.querySelector("input, button");
        const labelColumn = directChild(row, field);
        const valueColumn = directChild(row, control);
        if (!labelColumn || !valueColumn) continue;
        // Hiding only the label text leaves Viser's fixed-width label column empty.
        labelColumn.style.display = "none";
        valueColumn.style.flex = "1 1 100%";
        valueColumn.style.width = "100%";
        valueColumn.style.minWidth = "0";
        row.style.width = "100%";
        row.style.margin = "0";
        control.style.borderRadius = "0";
      }
      if (title.dataset.tactileBindingTitle === "add") {
        const capture = [...container.querySelectorAll("button")].find(
          (button) => button.textContent?.trim() === "Capture URDF Surface Pose",
        );
        const captureChild = directChild(container, capture);
        if (captureChild) captureChild.style.marginTop = "0.5em";
      }
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
        const parsed = Number(value);
        if (
          value === "" ||
          !Number.isFinite(parsed) ||
          String(parsed) !== value
        ) {
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
    layoutSelectedNames();
    preservePartialNumbers();
    layoutTaxelBindings();
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
