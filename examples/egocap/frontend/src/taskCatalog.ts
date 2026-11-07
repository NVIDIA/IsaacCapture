// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * The task catalog the recording panel's picker offers: the operator settings name the catalog file and an
 * optional task to preselect, and the catalog itself comes from the task-file API. Loaded when the operator
 * opens the task editor, so an import or a changed preselection in Settings is seen on the next edit without a
 * page reload. A failure to load leaves the editor as plain free text -- the catalog is a convenience, never a
 * gate on recording.
 */
import { useEffect, useRef, useState } from "react";
import { fetchOperatorSettings, fetchTaskFile, type TaskDefinition } from "./settingsApi";

export interface TaskCatalog {
  fileName: string;
  tasks: TaskDefinition[];
  /** The task the editor starts on when the current task is unset, or null. */
  preselected: TaskDefinition | null;
}

export type LoadTaskCatalog = (signal: AbortSignal) => Promise<TaskCatalog | null>;

/** The catalog named by the operator settings, or null when none is configured. */
export async function loadTaskCatalog(signal: AbortSignal): Promise<TaskCatalog | null> {
  const operator = await fetchOperatorSettings(signal);
  if (operator.task_file === null) return null;
  const file = await fetchTaskFile(operator.task_file, signal);
  const preselected = operator.preselected_task === null ? null : (file.tasks.find((task) => task.name === operator.preselected_task) ?? null);
  return { fileName: file.name, tasks: file.tasks, preselected };
}

/** Loads the catalog each time `active` turns true; null while loading, when none is configured, or on failure. */
export function useTaskCatalog(active: boolean, load: LoadTaskCatalog = loadTaskCatalog): TaskCatalog | null {
  const [catalog, setCatalog] = useState<TaskCatalog | null>(null);
  // The loader is read through a ref so a caller passing a fresh function each render (the panel re-renders on
  // its elapsed-time ticker) does not restart, and abort, the load on every render.
  const loadRef = useRef(load);
  loadRef.current = load;
  useEffect(() => {
    if (!active) {
      setCatalog(null);
      return;
    }
    const controller = new AbortController();
    loadRef.current(controller.signal).then(
      (loaded) => {
        if (!controller.signal.aborted) setCatalog(loaded);
      },
      () => {
        // No catalog is a normal state (free text); the reason is not the operator's problem mid-edit.
        if (!controller.signal.aborted) setCatalog(null);
      },
    );
    return () => controller.abort();
  }, [active]);
  return catalog;
}
