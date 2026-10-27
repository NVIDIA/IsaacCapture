/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { useCallback, useEffect, useState } from 'react';

import type { CloudXR2DUI } from './CloudXR2DUI';
import { readSessionError, saveSessionError } from './sessionError';

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
