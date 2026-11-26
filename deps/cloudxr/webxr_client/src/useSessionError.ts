/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { useCallback, useEffect, useState } from 'react';

import type { CloudXR2DUI } from './CloudXR2DUI';

const STORAGE_KEY = 'cxr.isaac.lastSessionError';

function readSessionError(): string {
  try {
    return sessionStorage.getItem(STORAGE_KEY) ?? '';
  } catch {
    return '';
  }
}

function saveSessionError(message: string): void {
  try {
    if (message) sessionStorage.setItem(STORAGE_KEY, message);
    else sessionStorage.removeItem(STORAGE_KEY);
  } catch {
    // Restricted storage must not prevent error reporting or session cleanup.
  }
}

export function useSessionError(ui: CloudXR2DUI | null): (message: string) => void {
  const [message, setMessage] = useState(readSessionError);

  useEffect(() => {
    ui?.setSessionError(message);
  }, [message, ui]);

  return useCallback(
    (error: string) => {
      saveSessionError(error);
      ui?.setSessionError(error);
      setMessage(error);
    },
    [ui]
  );
}
