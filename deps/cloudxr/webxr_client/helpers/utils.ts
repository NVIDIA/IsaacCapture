/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * Parses URL parameters and returns them as an object
 * @param location - Optional location object (defaults to window.location)
 * @returns Object with URL parameters as key-value pairs
 */
export function getUrlParams(location: Location = window.location): Record<string, string> {
  const params: Record<string, string> = {};
  const queryString = location.search.substring(1);

  if (queryString) {
    const pairs = queryString.split('&');
    for (const pair of pairs) {
      const [key, value = ''] = pair.split('=');
      params[decodeURIComponent(key)] = decodeURIComponent(value);
    }
  }

  return params;
}

/**
 * Enables localStorage functionality for form elements
 * @param element - The HTML input or select element to enable localStorage for
 * @param key - The localStorage key to use for saving/loading the value
 */
export function enableLocalStorage(element: HTMLInputElement | HTMLSelectElement, key: string) {
  // Check if localStorage is already enabled for this element and key
  const localStorageKey = `__localStorageEnabled_${key}`;
  if ((element as any)[localStorageKey]) {
    console.warn(`localStorage already enabled for ${key}, skipping`);
    return;
  }

  // Checkboxes persist via `.checked` (as 'true'/'false'), everything else via `.value`.
  const isCheckbox = element instanceof HTMLInputElement && element.type === 'checkbox';

  // Load saved value from localStorage
  try {
    // Check if the key exists in localStorage, not just if it has values
    if (localStorage.hasOwnProperty(key)) {
      const savedValue = localStorage.getItem(key);
      if (isCheckbox) {
        (element as HTMLInputElement).checked = savedValue === 'true';
      } else {
        element.value = savedValue || '';
      }
      console.info(`Loaded saved ${key} from localStorage:`, savedValue);
    }
  } catch (error) {
    console.warn(`${key}: Failed to load saved value from localStorage:`, error);
  }

  // Set up event listener to save value when changed
  const changeHandler = () => {
    try {
      const toSave = isCheckbox ? String((element as HTMLInputElement).checked) : element.value;
      localStorage.setItem(key, toSave);
      console.info(`Saved ${key} to localStorage:`, JSON.stringify(toSave));
    } catch (error) {
      console.warn(`${key}: Failed to save to localStorage:`, error);
    }
  };

  element.addEventListener('change', changeHandler);

  // Mark this element as having localStorage enabled for this key
  (element as any)[localStorageKey] = true;
}

/**
 * Sets the select value only when the option exists.
 * @param select - The select element to update
 * @param value - The value to apply if an option matches
 */
export function setSelectValueIfAvailable(select: HTMLSelectElement, value: string): void {
  const hasOption = Array.from(select.options).some(option => option.value === value);
  if (hasOption) {
    select.value = value;
  }
}

/**
 * Strips protocol prefixes (http:// or https://) from a URL string
 * @param url - The URL string to clean
 * @returns The URL without protocol prefix
 */
function stripProtocol(url: string): string {
  return url.replace(/^https?:\/\//, '');
}

/**
 * Connection configuration object containing server connection details
 */
export interface ConnectionConfiguration {
  /** Server address (proxy address when using proxy, direct server address otherwise) */
  serverIP: string;

  /** Port number (443 for HTTPS proxy, user-provided for direct connections) */
  port: number;

  /** Whether the connection will use secure protocol (HTTPS/WSS) */
  useSecureConnection: boolean;

  /** Optional resource path for proxy routing (e.g., "/192.168.1.100" for target server) */
  resourcePath?: string;
}

/**
 * CloudXR configuration interface containing all streaming settings
 */
export interface CloudXRConfig {
  /** IP address of the CloudXR streaming server */
  serverIP: string;

  /** Port number for the CloudXR server connection */
  port: number;

  /** Whether to use secure connection (HTTPS/WSS) or insecure (HTTP/WS) */
  useSecureConnection: boolean;

  /** Width of each eye in pixels (must be multiple of 16) */
  perEyeWidth: number;

  /** Height of each eye in pixels (must be multiple of 16) */
  perEyeHeight: number;

  /** Depth reprojection mesh grid vertex columns (not cell columns). Undefined uses factor mode. */
  reprojectionGridCols?: number;

  /** Depth reprojection mesh grid vertex rows (not cell rows). Undefined uses factor mode. */
  reprojectionGridRows?: number;

  /** Target frame rate for the XR device in frames per second (FPS) */
  deviceFrameRate: number;

  /** Maximum streaming bitrate in Megabits per second (Mbps) */
  maxStreamingBitrateMbps: number;

  /** Preferred video codec used for streaming */
  codec?: 'h264' | 'h265' | 'av1';

  /** XR immersive mode: 'ar' for augmented reality, 'vr' for virtual reality */
  immersiveMode: 'ar' | 'vr';

  /** Optional device profile identifier used by the examples UI */
  deviceProfileId?: string;

  /** Type of server being connected to */
  serverType: string;

  /** Optional proxy URL for HTTPS routing (e.g., 'https://proxy.example.com/'); if empty, uses direct WSS connection */
  proxyUrl: string;

  /** Preferred XR reference space for tracking and positioning */
  referenceSpaceType: 'auto' | 'local-floor' | 'local' | 'viewer' | 'unbounded';

  /** XR reference space offset along X axis in meters (positive is right) */
  xrOffsetX?: number;
  /** XR reference space offset along Y axis in meters (positive is up) */
  xrOffsetY?: number;
  /** XR reference space offset along Z axis in meters (positive is backward) */
  xrOffsetZ?: number;

  /** Enable secondary position smoothing in CloudXR */
  enablePoseSmoothing?: boolean;
  /** Pose prediction factor for CloudXR (0.0-1.0) */
  posePredictionFactor?: number;
  /** Enable texSubImage2D optimization in CloudXR */
  enableTexSubImage2D?: boolean;
  /** Quest-specific color workaround in CloudXR */
  useQuestColorWorkaround?: boolean;
  /** Whether WebXR controller models should be hidden */
  hideControllerModel?: boolean;

  /** Media server address for WebRTC streaming (for NAT traversal) */
  mediaAddress?: string;

  /** Media server port for WebRTC streaming (use 0 for server-provided port) */
  mediaPort?: number;

  /**
   * Pre-stream network test: 'off' skips it, 'warn' measures and reports, 'block' also
   * refuses to stream when the link fails the gate.
   *
   * Defaults to 'off' for IsaacTeleop. The test holds the session in Connecting for its
   * whole window, and an operator connecting to a robot should not be gated on it.
   */
  streamTestMode?: 'off' | 'warn' | 'block';

  /** Length of the network test measurement window in seconds. Ignored when the test is off. */
  streamTestDurationSeconds?: number;
}

/**
 * Reads per-eye resolution from two number inputs. When a field is blank, uses that input's
 * HTML value attribute (default from the page). Returns 0 for any non-finite parsed value.
 */
export function getResolutionFromInputs(
  widthInput: HTMLInputElement,
  heightInput: HTMLInputElement
): { w: number; h: number } {
  const wRaw = widthInput.value.trim();
  const hRaw = heightInput.value.trim();
  const w =
    wRaw === ''
      ? parseInt(widthInput.getAttribute('value') ?? '', 10)
      : parseInt(widthInput.value, 10);
  const h =
    hRaw === ''
      ? parseInt(heightInput.getAttribute('value') ?? '', 10)
      : parseInt(heightInput.value, 10);
  return {
    w: Number.isFinite(w) ? w : 0,
    h: Number.isFinite(h) ? h : 0,
  };
}

/**
 * Reads reprojection grid values from two number inputs.
 * Blank means "use default factor mode", returned as undefined.
 */
export function getGridFromInputs(
  reprojectionGridColsInput: HTMLInputElement,
  reprojectionGridRowsInput: HTMLInputElement
): { reprojectionGridCols: number | undefined; reprojectionGridRows: number | undefined } {
  const colsRaw = reprojectionGridColsInput.value.trim();
  const rowsRaw = reprojectionGridRowsInput.value.trim();
  const cols = colsRaw === '' ? undefined : parseInt(reprojectionGridColsInput.value, 10);
  const rows = rowsRaw === '' ? undefined : parseInt(reprojectionGridRowsInput.value, 10);
  return {
    reprojectionGridCols: cols === undefined || Number.isFinite(cols) ? cols : NaN,
    reprojectionGridRows: rows === undefined || Number.isFinite(rows) ? rows : NaN,
  };
}

/**
 * Determines connection configuration based on protocol and user inputs
 * Supports both direct WSS connections and proxy routing for HTTPS
 *
 * @param serverIP - The user-provided server IP address
 * @param port - The user-provided port number
 * @param proxyUrl - Optional proxy URL for HTTPS routing (if provided, uses proxy routing; otherwise direct connection)
 * @param location - Optional location object (defaults to window.location)
 * @returns Object containing server IP, port, security settings, and optional resource path for proxy routing
 * @throws {Error} When proxy URL format is invalid (must start with https://)
 */
export function getConnectionConfig(
  serverIP: string,
  port: number,
  proxyUrl: string,
  location: Location = window.location
): ConnectionConfiguration {
  let finalServerIP = '';
  let finalPort = port;
  let finalUseSecureConnection = false;
  let resourcePath: string | undefined = undefined;

  // Determine if we should use secure connection based on page protocol
  if (location.protocol === 'https:') {
    console.info('Running on HTTPS protocol - using secure WebSocket (WSS)');
    finalUseSecureConnection = true;

    // Check if proxy URL is provided for routing
    const trimmedProxyUrl = proxyUrl?.trim();
    if (trimmedProxyUrl) {
      // Proxy routing mode
      console.info('Proxy URL provided - using proxy routing mode');

      if (!trimmedProxyUrl.startsWith('https://')) {
        throw new Error('Proxy URL must start with https://. Received: ' + trimmedProxyUrl);
      }

      // Parse proxy URL to separate hostname from path
      const proxyUrlObj = new URL(trimmedProxyUrl);
      finalServerIP = proxyUrlObj.hostname;

      // Use proxy port if specified, otherwise default to 443
      finalPort = proxyUrlObj.port ? parseInt(proxyUrlObj.port, 10) : 443;

      // Get proxy path (remove trailing slash)
      const proxyPath = proxyUrlObj.pathname.replace(/\/$/, '');

      // Route through proxy: combine proxy path with target server IP
      if (serverIP) {
        const cleanServerIP = stripProtocol(serverIP);
        // Combine proxy path with target server: /proxy/10.28.132.185
        resourcePath = proxyPath ? `${proxyPath}/${cleanServerIP}` : `/${cleanServerIP}`;
        console.info(
          `Using HTTPS proxy: ${finalServerIP}:${finalPort} with resource path: ${resourcePath}`
        );
      } else {
        // No target server, just use proxy path if any
        resourcePath = proxyPath || undefined;
        console.info(
          `Using HTTPS proxy: ${finalServerIP}:${finalPort} (no target server specified)`
        );
      }
    } else {
      // Direct WSS connection mode
      console.info('No proxy URL - using direct WSS connection');

      // Handle server IP input
      if (serverIP) {
        finalServerIP = stripProtocol(serverIP);
        console.info('Using user-provided server IP:', finalServerIP);
      } else {
        finalServerIP = new URL(location.href).hostname;
        console.info('Using default server IP from window location:', finalServerIP);
      }

      // Use user-provided port for direct WSS
      if (port && !isNaN(port)) {
        finalPort = port;
        console.info('Using user-provided port:', finalPort);
      }
    }
  } else {
    // HTTP protocol - direct WS connection
    console.info('Running on HTTP protocol - using insecure WebSocket (WS)');
    finalUseSecureConnection = false;

    // Handle server IP input
    if (serverIP) {
      finalServerIP = stripProtocol(serverIP);
      console.info('Using user-provided server IP:', finalServerIP);
    } else {
      finalServerIP = new URL(location.href).hostname;
      console.info('Using default server IP from window location:', finalServerIP);
    }

    // Handle port input
    if (port && !isNaN(port)) {
      finalPort = port;
      console.info('Using user-provided port:', finalPort);
    }
  }

  return {
    serverIP: finalServerIP,
    port: finalPort,
    useSecureConnection: finalUseSecureConnection,
    resourcePath,
  } as ConnectionConfiguration;
}

/**
 * Returns true when the hostname looks like a local/dev environment rather than
 * a cloud-hosted domain. Used to decide whether the cert acceptance link should
 * be shown automatically when no server IP is provided.
 */
export function isLocalServer(hostname: string): boolean {
  const isIPv4 = /^\d{1,3}(\.\d{1,3}){3}$/.test(hostname);
  const isIPv6 = hostname.includes(':');
  return hostname === 'localhost' || isIPv4 || isIPv6;
}

/**
 * Sets up certificate acceptance link for self-signed certificates in HTTPS mode
 * Shows a link to accept certificates when using direct WSS connection (no proxy)
 *
 * @param serverIpInput - Input element for server IP address
 * @param portInput - Input element for port number
 * @param proxyUrlInput - Input element for proxy URL
 * @param certAcceptanceLink - Container element for the certificate link
 * @param certLink - Anchor element for the certificate URL
 * @param location - Optional location object (defaults to window.location)
 * @returns Cleanup function that removes listeners created by this helper
 */
export function setupCertificateAcceptanceLink(
  serverIpInput: HTMLInputElement,
  portInput: HTMLInputElement,
  proxyUrlInput: HTMLInputElement,
  certAcceptanceLink: HTMLElement,
  certLink: HTMLAnchorElement,
  location: Location = window.location,
  fetchFn: typeof fetch = globalThis.fetch
): () => void {
  let abortController: AbortController | null = null;
  let accepted = false;
  let certRequired = false;
  let activeCertUrl: string | null = null;
  let pendingVerification: Promise<void> | null = null;
  let pendingVerificationUrl: string | null = null;

  function markAccepted(url: string): void {
    if (url !== activeCertUrl) {
      return;
    }
    if (!accepted) {
      console.warn('[CloudXR] Certificate accepted for %s', url);
    }
    accepted = true;
    certAcceptanceLink.classList.add('cert-accepted');
    certLink.textContent = `Certificate accepted (${url})`;
  }

  function markUnverified(url: string): void {
    if (url !== activeCertUrl) {
      return;
    }
    accepted = false;
    certAcceptanceLink.classList.remove('cert-accepted');
    certLink.textContent = `Click ${url} to accept cert`;
  }

  async function checkCert(url: string): Promise<void> {
    if (url !== activeCertUrl) {
      return;
    }
    // Skip polling while an XR session is active to avoid unnecessary network requests
    if (document.body.classList.contains('xr-mode')) {
      return;
    }
    if (abortController) {
      abortController.abort();
    }
    abortController = new AbortController();
    try {
      await fetchFn(url, { signal: abortController.signal, mode: 'no-cors' });
      markAccepted(url);
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') {
        return;
      }
      markUnverified(url);
      console.warn(
        '[CloudXR] Certificate not yet accepted — cert polling errors for %s are expected.',
        url
      );
    }
  }

  /**
   * Updates the certificate acceptance link based on current configuration
   * Shows link only when in HTTPS mode without proxy (direct WSS)
   */
  const updateCertLink = () => {
    const isHttps = location.protocol === 'https:';
    const hasProxy = proxyUrlInput.value.trim().length > 0;
    const portValue = parseInt(portInput.value, 10);
    const defaultPort = hasProxy ? 443 : 48322;
    const port = portValue || defaultPort;

    const serverIp = serverIpInput.value.trim();
    const serverIpPopulated = serverIp.length > 0;

    // Only show when we have a reasonable cert URL: either the user filled in
    // a server IP, or the page itself is on a local/dev host.
    certRequired = isHttps && !hasProxy && (serverIpPopulated || isLocalServer(location.hostname));
    if (certRequired) {
      const effectiveIp = serverIpPopulated ? serverIp : new URL(location.href).hostname;
      const url = `https://${effectiveIp}:${port}/`;
      activeCertUrl = url;
      certAcceptanceLink.style.display = 'block';
      certLink.href = url;
      markUnverified(url);
    } else {
      activeCertUrl = null;
      accepted = false;
      if (abortController) abortController.abort();
      certAcceptanceLink.classList.remove('cert-accepted');
      certAcceptanceLink.style.display = 'none';
    }
  };

  const onFocus = () => {
    if (certRequired && activeCertUrl) {
      void startVerification();
    }
  };

  const onInput = () => {
    updateCertLink();
  };
  const onCommittedChange = () => {
    void startVerification();
  };
  const onProxyCommittedChange = () => {
    updateCertLink();
    if (certRequired && activeCertUrl) {
      void startVerification();
    }
  };

  // Typing updates displayed URL/state; committed IP/port changes trigger probes.
  serverIpInput.addEventListener('input', onInput);
  portInput.addEventListener('input', onInput);
  proxyUrlInput.addEventListener('input', onInput);
  serverIpInput.addEventListener('change', onCommittedChange);
  serverIpInput.addEventListener('blur', onCommittedChange);
  portInput.addEventListener('change', onCommittedChange);
  portInput.addEventListener('blur', onCommittedChange);
  proxyUrlInput.addEventListener('change', onProxyCommittedChange);
  proxyUrlInput.addEventListener('blur', onProxyCommittedChange);
  window.addEventListener('focus', onFocus);

  // Run initial cert state after localStorage restoration.
  void startVerification();

  function startVerification(): Promise<void> {
    updateCertLink();
    const currentUrl = certRequired ? activeCertUrl : null;
    if (pendingVerification && pendingVerificationUrl === currentUrl) {
      return pendingVerification;
    }
    const run = (async () => {
      if (currentUrl) {
        await checkCert(currentUrl);
      }
    })();
    pendingVerification = run;
    pendingVerificationUrl = currentUrl;
    return run.finally(() => {
      if (pendingVerification === run) {
        pendingVerification = null;
        pendingVerificationUrl = null;
      }
    });
  }

  return () => {
    serverIpInput.removeEventListener('input', onInput);
    portInput.removeEventListener('input', onInput);
    proxyUrlInput.removeEventListener('input', onInput);
    serverIpInput.removeEventListener('change', onCommittedChange);
    serverIpInput.removeEventListener('blur', onCommittedChange);
    portInput.removeEventListener('change', onCommittedChange);
    portInput.removeEventListener('blur', onCommittedChange);
    proxyUrlInput.removeEventListener('change', onProxyCommittedChange);
    proxyUrlInput.removeEventListener('blur', onProxyCommittedChange);
    window.removeEventListener('focus', onFocus);
    if (abortController) abortController.abort();
  };
}
