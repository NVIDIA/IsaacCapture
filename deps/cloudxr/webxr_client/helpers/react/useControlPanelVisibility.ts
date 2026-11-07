/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect, useRef, useState } from 'react';

interface ControlPanelVisibilityOptions {
  isXRMode: boolean;
  panelHiddenAtStart: boolean;
  autoHideOnConnect: boolean;
  isStreaming: boolean;
}

export function useControlPanelVisibility({
  isXRMode,
  panelHiddenAtStart,
  autoHideOnConnect,
  isStreaming,
}: ControlPanelVisibilityOptions) {
  const [panelHidden, setPanelHidden] = useState(false);
  const prevXRMode = useRef(false);

  useEffect(() => {
    if (isXRMode && !prevXRMode.current && !autoHideOnConnect) {
      setPanelHidden(panelHiddenAtStart);
    }
    prevXRMode.current = isXRMode;
  }, [isXRMode, panelHiddenAtStart, autoHideOnConnect]);

  useEffect(() => {
    // Video viewers need connection feedback until the stream starts, even with a saved
    // hide-on-entry preference. XR entry and teleop Play do not indicate stream readiness.
    // Only connection transitions change visibility, so Show/Hide remains under user control.
    if (isXRMode && autoHideOnConnect) {
      setPanelHidden(isStreaming);
    }
  }, [isXRMode, autoHideOnConnect, isStreaming]);

  return [panelHidden, setPanelHidden] as const;
}
