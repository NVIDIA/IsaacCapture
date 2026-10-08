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
 * CloudXR2DUI.tsx - CloudXR 2D User Interface Management
 *
 * This class handles all the HTML form interactions, localStorage persistence,
 * and form validation for the CloudXR React example. It follows the same pattern
 * as the simple example's CloudXRWebUI class, providing a clean separation
 * between UI management and React component logic.
 *
 * Features:
 * - Form field management and localStorage persistence
 * - Proxy configuration based on protocol
 * - Form validation and default value handling
 * - Event listener management
 * - Error handling and logging
 */

import {
  getGridValidationError,
  getGridValidationMessageForConnect,
  getResolutionValidationError,
  getResolutionValidationMessageForConnect,
  validateDepthReprojectionGrid,
  validatePerEyeResolution,
} from '@nvidia/cloudxr';

import {
  detectDeviceProfileId,
  type DeviceProfileId,
  getDeviceProfile,
  resolveDeviceProfileId,
} from '@helpers/DeviceProfiles';
import {
  type AutoRefreshMode,
  loadPerProject,
  parseAutoRefreshMode,
  parseControlPanelPosition,
  parseStreamTestMode,
  ReactUIConfig,
  savePerProject,
} from '@helpers/react/utils';
import {
  DEFAULT_TELEOP_PATH,
  DROPDOWN_ENTRIES,
  getProjectBreadcrumb,
  getProjectSettings,
} from '@helpers/TeleopProjects';
import {
  CloudXRConfig,
  enableLocalStorage,
  getGridFromInputs,
  getResolutionFromInputs,
  setSelectValueIfAvailable,
  setupCertificateAcceptanceLink,
} from '@helpers/utils';

import { URL_PARAMS } from './config/params';
import { seedsFromParams } from './config/resolve';
import type { ReplayPacing } from './xrInputRecorder';

/** Full config: CloudXR connection settings + React UI options. */
type AppConfig = CloudXRConfig &
  ReactUIConfig & {
    showTrace: boolean;
    showRecordingControls: boolean;
    replayPacing: ReplayPacing;
  };

/**
 * localStorage key for the teleop-start countdown. Owned by the countdown feature in
 * App.tsx but defined here with the other storage keys so reset clears the same key the
 * feature writes. (Imported by App.tsx; App already depends on this module, so no cycle.)
 */
export const COUNTDOWN_STORAGE_KEY = 'cxr.react.countdownSeconds';

/**
 * 2D UI Management for CloudXR React Example
 * Handles the main user interface for CloudXR streaming, including form management,
 * localStorage persistence, and user interaction controls.
 */
export class CloudXR2DUI {
  /** Button to initiate XR streaming session */
  private startButton!: HTMLButtonElement;
  /** Input field for the CloudXR server IP address */
  private serverIpInput!: HTMLInputElement;
  /** Button to clear the prefilled server IP (re-enables the browser autocomplete dropdown) */
  private serverIpClearButton!: HTMLButtonElement;
  /** Input field for the CloudXR server port number */
  private portInput!: HTMLInputElement;
  /** Input field for proxy URL configuration */
  private proxyUrlInput!: HTMLInputElement;
  /** Dropdown to select between AR and VR immersive modes */
  private immersiveSelect!: HTMLSelectElement;
  /** Dropdown to select device frame rate (FPS) */
  private deviceFrameRateSelect!: HTMLSelectElement;
  /** Dropdown to select max streaming bitrate (Mbps) */
  private maxStreamingBitrateMbpsSelect!: HTMLSelectElement;
  /** Dropdown to select preferred streaming codec */
  private codecSelect!: HTMLSelectElement;
  /** Input field for per-eye width configuration */
  private perEyeWidthInput!: HTMLInputElement;
  /** Input field for per-eye height configuration */
  private perEyeHeightInput!: HTMLInputElement;
  /** Input field for reprojection mesh grid X (columns) */
  private reprojectionGridColsInput!: HTMLInputElement;
  /** Input field for reprojection mesh grid Y (rows) */
  private reprojectionGridRowsInput!: HTMLInputElement;
  /** Inline resolution validation under width input */
  private resolutionWidthValidationMessage: HTMLElement | null = null;
  /** Inline resolution validation under height input */
  private resolutionHeightValidationMessage: HTMLElement | null = null;
  /** Inline grid validation under reprojection grid columns input */
  private reprojectionGridColsValidationMessage: HTMLElement | null = null;
  /** Inline grid validation under reprojection grid rows input */
  private reprojectionGridRowsValidationMessage: HTMLElement | null = null;
  private validationMessageBox!: HTMLElement;
  private validationMessageText!: HTMLElement;
  /** Dropdown to enable pose smoothing */
  private enablePoseSmoothingSelect!: HTMLSelectElement;
  /** Pose prediction factor slider */
  private posePredictionFactorInput!: HTMLInputElement;
  /** Pose prediction factor value text */
  private posePredictionFactorValue!: HTMLElement;
  /** Dropdown to enable texSubImage2D optimization */
  private enableTexSubImage2DSelect!: HTMLSelectElement;
  /** Dropdown to enable Quest color workaround */
  private useQuestColorWorkaroundSelect!: HTMLSelectElement;
  /** Dropdown to select server backend type */
  private serverTypeSelect!: HTMLSelectElement;
  /** Dropdown to select device profile */
  private deviceProfileSelect!: HTMLSelectElement;
  /** Whether the control panel starts hidden when immersive XR begins */
  private panelHiddenAtStartSelect!: HTMLSelectElement;
  /** Dropdown to select reference space for XR tracking */
  private referenceSpaceSelect!: HTMLSelectElement;
  /** Input for XR reference space X offset (cm) */
  private xrOffsetXInput!: HTMLInputElement;
  /** Input for XR reference space Y offset (cm) */
  private xrOffsetYInput!: HTMLInputElement;
  /** Input for XR reference space Z offset (cm) */
  private xrOffsetZInput!: HTMLInputElement;
  /** Select for in-XR control panel start position (left / center / right) */
  private controlPanelPositionSelect!: HTMLSelectElement;
  /** In-XR control panel layout tuning: distance/height/angle overrides. */
  private controlPanelDistanceInput!: HTMLInputElement;
  private controlPanelHeightInput!: HTMLInputElement;
  private controlPanelAngleDegreesInput!: HTMLInputElement;
  private controlPanelTrackHeadsetInput!: HTMLInputElement;
  /** Text element displaying proxy configuration help */
  private proxyDefaultText!: HTMLElement;
  /** Device profile warning text */
  private deviceProfileWarning!: HTMLElement;
  /** Error message box element */
  private errorMessageBox!: HTMLElement;
  /** Error message text element */
  private errorMessageText!: HTMLElement;
  /** Certificate acceptance link container */
  private certAcceptanceLink!: HTMLElement;
  /** Certificate acceptance link anchor */
  private certLink!: HTMLAnchorElement;
  /** Input field for media server address */
  private mediaAddressInput!: HTMLInputElement;
  /** Input field for media server port */
  private mediaPortInput!: HTMLInputElement;
  /** Dropdown selecting the pre-stream network test mode (off / warn / block) */
  private streamTestModeSelect!: HTMLSelectElement;
  /** Input field for the network test window length in seconds */
  private streamTestDurationSecondsInput!: HTMLInputElement;
  /** Dropdown for controller model visibility (show / hide) */
  private controllerModelVisibilitySelect!: HTMLSelectElement;
  private showTraceInXRSelect!: HTMLSelectElement;
  private showRecordingControlsSelect!: HTMLSelectElement;
  private replayPacingSelect!: HTMLSelectElement;
  /** Skip client CloudXR `render` (headless: client blit off; tracking on) */
  private headlessInput!: HTMLInputElement;
  /** When to reload the page after the XR session ends (never / clean / any) */
  private autoRefreshModeSelect!: HTMLSelectElement;
  /** Retry a recoverable stream error instead of ending the session */
  private reconnectEnabledInput!: HTMLInputElement;
  /** Max retry attempts before giving up */
  private reconnectMaxAttemptsInput!: HTMLInputElement;
  /** Delay before each retry attempt, in milliseconds */
  private reconnectDelayMsInput!: HTMLInputElement;
  /** Base timeout for passthrough-only detection, in milliseconds - independent of retry. */
  private streamAttachTimeoutMsInput!: HTMLInputElement;
  /** Timeout for detecting no decoder warm-up signal at all since attach, in milliseconds. */
  private warmupBeginTimeoutMsInput!: HTMLInputElement;
  /** Timeout for detecting decoder warm-up that began but never finished, in milliseconds. */
  private warmupEndTimeoutMsInput!: HTMLInputElement;
  /** Button that clears stored settings and reloads to defaults. */
  private resetSettingsButton!: HTMLButtonElement;
  /** Container for the runtime-generated URL-parameter help list (optional in markup). */
  private urlParamsHelpList: HTMLElement | null = null;
  /** Breadcrumb subtitle in header (e.g. "for Real Robot › GEAR › Dexmate"). */
  private teleopModeSubtitle!: HTMLElement;
  /** Hierarchical project selector in header */
  private teleopProjectSelect!: HTMLSelectElement;
  /** Active teleop project path (a key path in `TELEOP_PROJECTS`, e.g. `real/gear/dexmate`). */
  private teleopPath: string = DEFAULT_TELEOP_PATH;
  /** Flag to track if the 2D UI has been initialized */
  private initialized: boolean = false;

  /** Current form configuration state */
  private currentConfiguration: AppConfig;
  /** Callback function for configuration changes */
  private onConfigurationChange: ((config: AppConfig) => void) | null = null;
  /** Connect button click handler for cleanup */
  private handleConnectClick: ((event: Event) => void) | null = null;
  /** Array to store all event listeners for proper cleanup */
  private eventListeners: Array<{
    element: HTMLElement;
    event: string;
    handler: EventListener;
  }> = [];
  /** Certificate link cleanup function */
  private certLinkCleanup: (() => void) | null = null;

  /**
   * Creates a new CloudXR2DUI instance
   * @param onConfigurationChange - Callback function called when configuration changes
   */
  constructor(onConfigurationChange?: (config: AppConfig) => void) {
    this.onConfigurationChange = onConfigurationChange || null;
    this.currentConfiguration = this.getDefaultConfiguration();
  }

  /**
   * Initializes the CloudXR2DUI with all necessary components and event handlers
   */
  public initialize(teleopPath?: string): void {
    if (this.initialized) {
      return;
    }

    try {
      this.initializeElements();
      this.decoratePerProjectLabels();
      this.setupLocalStorage();

      if (teleopPath) {
        this.teleopPath = teleopPath;
      }
      this.applyTeleopPath();

      // Before URL seeds so explicit params still win: on a fresh load, default the
      // device profile from the headset UA and apply its values.
      this.applyDefaultDeviceProfileFromUserAgent();
      // Re-apply a persisted non-custom profile so profile updates reach returning
      // clients: localStorage restores the values saved when the profile was picked,
      // which otherwise pins them forever. Safe because any manual edit of a
      // profile-bound field switches the persisted profile to 'custom'.
      const persistedProfileId = resolveDeviceProfileId(this.deviceProfileSelect.value);
      if (persistedProfileId !== 'custom') {
        this.applyDeviceProfileToForm(persistedProfileId);
        this.persistProfileFieldsToLocalStorage();
      }
      this.applyUrlSeeds();
      this.setupProxyConfiguration();
      this.renderUrlParamsHelp();
      this.setupEventListeners();
      this.restoreGroupExpandedState();
      // Set initial display value
      this.posePredictionFactorValue.textContent = this.posePredictionFactorInput.value;
      this.updateConfiguration();
      this.updateDeviceProfileWarning(resolveDeviceProfileId(this.deviceProfileSelect.value));
      this.updateConnectButtonState();
      this.initialized = true;
    } catch (error) {
      // Continue with default values if initialization fails
      this.showError(`Failed to initialize CloudXR2DUI: ${error}`);
    }
  }

  /** Renders the header breadcrumb and rebuilds the project dropdown from the registry. */
  private applyTeleopPath(): void {
    const breadcrumb = getProjectBreadcrumb(this.teleopPath);
    this.teleopModeSubtitle.replaceChildren();
    this.teleopModeSubtitle.appendChild(document.createTextNode('for '));
    breadcrumb.forEach((label, i) => {
      if (i > 0) {
        // aria-hidden so screen readers skip the chevron glyph.
        const sep = document.createElement('span');
        sep.setAttribute('aria-hidden', 'true');
        sep.textContent = ' \u203A ';
        this.teleopModeSubtitle.appendChild(sep);
      }
      this.teleopModeSubtitle.appendChild(document.createTextNode(label));
    });

    this.populateProjectDropdown();
    this.applyPerProjectSettings();
  }

  /** Loads per-project-path settings (falling back to registry defaults) into their form controls. */
  private applyPerProjectSettings(): void {
    const settings = getProjectSettings(this.teleopPath);
    const boolFromStorage = (raw: string) =>
      raw === 'true' ? true : raw === 'false' ? false : undefined;
    const panelHidden = loadPerProject<boolean>(
      'panelHiddenAtStart',
      this.teleopPath,
      boolFromStorage,
      settings.panelHiddenAtStart ?? false
    );
    this.panelHiddenAtStartSelect.value = String(panelHidden);
    const headless = loadPerProject<boolean>(
      'headless',
      this.teleopPath,
      boolFromStorage,
      settings.headless ?? false
    );
    this.headlessInput.checked = headless;
    this.applyHeadlessImmersiveDropdown();
  }

  /**
   * Headless forces immersive-vr: grey out AR/VR and show VR only ({@link CloudXR2DUI.updateConfiguration} still coerces immersiveMode).
   */
  private applyHeadlessImmersiveDropdown(): void {
    if (this.headlessInput.checked) {
      this.immersiveSelect.value = 'vr';
      this.immersiveSelect.disabled = true;
      this.immersiveSelect.title =
        'Headless requires VR (immersive-vr); AR passthrough is not available in this mode.';
    } else {
      this.immersiveSelect.disabled = false;
      this.immersiveSelect.title = '';
    }
  }

  /**
   * Appends a visible marker to the label of each setting that persists per
   * teleop application, so users can tell at a glance which fields switch when
   * they change the active project. No hover text: the UI runs on VR headsets
   * where hover is not reliable, so the marker itself must be self-describing.
   */
  private decoratePerProjectLabels(): void {
    const marker = ' (saved per teleop application)';
    for (const el of [this.panelHiddenAtStartSelect, this.headlessInput]) {
      const label = el.labels?.[0];
      if (!label || label.textContent?.includes(marker)) continue;
      label.appendChild(document.createTextNode(marker));
    }
  }

  /** Builds the hierarchical dropdown options from the project registry. */
  private populateProjectDropdown(): void {
    const select = this.teleopProjectSelect;
    select.replaceChildren();

    const INDENT = '\u00A0\u00A0\u00A0';
    const currentHash = `#/${this.teleopPath}`;

    // Static prompt when collapsed (the breadcrumb already shows the current path);
    // the active entry is suffixed and disabled so it reads as already selected.
    const prompt = document.createElement('option');
    prompt.value = '';
    prompt.textContent = 'Change teleop application';
    select.appendChild(prompt);

    for (const entry of DROPDOWN_ENTRIES) {
      const option = document.createElement('option');
      option.value = entry.hash;
      const isCurrent = entry.hash === currentHash;
      option.textContent =
        INDENT.repeat(entry.depth) + entry.label + (isCurrent ? '  (current)' : '');
      if (isCurrent) option.disabled = true;
      select.appendChild(option);
    }

    select.selectedIndex = 0;
  }

  /**
   * On a fresh load (no stored device-profile choice), default the Device Profile from the
   * headset user-agent and apply its values to the form. A stored 'deviceProfile' suppresses
   * this — it is written whenever the user picks a profile or edits any profile-linked field
   * (setProfileToCustomIfNeeded), so we never override an explicit choice. Not persisted, so
   * detection re-runs each fresh load until the user makes a choice. Runs before applyUrlSeeds()
   * so URL params still win.
   *
   * Note: the IWER emulator emulates a headset via the WebXR API but does NOT change
   * navigator.userAgent, so under the emulator this resolves to 'custom' (no flip).
   */
  private applyDefaultDeviceProfileFromUserAgent(): void {
    let stored: string | null = null;
    try {
      stored = localStorage.getItem('deviceProfile');
    } catch (_) {}
    if (stored != null) return;
    const detected = detectDeviceProfileId();
    if (detected === 'custom') return;
    this.deviceProfileSelect.value = detected;
    this.applyDeviceProfileToForm(detected);
  }

  /**
   * Override form controls from URL query params. Values are applied but not persisted;
   * called after setupLocalStorage() so URL params win over stored values for this load.
   */
  private applyUrlSeeds(): void {
    // Seed every form-backed URL_PARAMS control whose query param is present and valid.
    // URL values override but are not persisted (next load without the param
    // falls back to the stored/default value).
    const seeds = seedsFromParams(new URLSearchParams(window.location.search));
    for (const field of URL_PARAMS) {
      if (!field.elementId) continue;
      const raw = seeds.get(field.key);
      if (raw === undefined) continue;
      const el = document.getElementById(field.elementId) as
        HTMLInputElement | HTMLSelectElement | null;
      if (!el) continue;
      if (field.kind === 'checked') {
        (el as HTMLInputElement).checked = raw === 'true';
      } else {
        el.value = raw;
      }
    }
    if (seeds.has('headless')) {
      this.applyHeadlessImmersiveDropdown();
    }
  }

  /**
   * Initializes all DOM element references by their IDs
   * Throws an error if any required element is not found
   */
  private initializeElements(): void {
    this.startButton = this.getElement<HTMLButtonElement>('startButton');
    this.serverIpInput = this.getElement<HTMLInputElement>('serverIpInput');
    this.serverIpClearButton = this.getElement<HTMLButtonElement>('serverIpClearButton');
    this.portInput = this.getElement<HTMLInputElement>('portInput');
    this.proxyUrlInput = this.getElement<HTMLInputElement>('proxyUrl');
    this.immersiveSelect = this.getElement<HTMLSelectElement>('immersive');
    this.deviceFrameRateSelect = this.getElement<HTMLSelectElement>('deviceFrameRate');
    this.maxStreamingBitrateMbpsSelect =
      this.getElement<HTMLSelectElement>('maxStreamingBitrateMbps');
    this.codecSelect = this.getElement<HTMLSelectElement>('codec');
    this.perEyeWidthInput = this.getElement<HTMLInputElement>('perEyeWidth');
    this.perEyeHeightInput = this.getElement<HTMLInputElement>('perEyeHeight');
    this.reprojectionGridColsInput = this.getElement<HTMLInputElement>('reprojectionGridCols');
    this.reprojectionGridRowsInput = this.getElement<HTMLInputElement>('reprojectionGridRows');
    this.resolutionWidthValidationMessage = document.getElementById(
      'resolutionWidthValidationMessage'
    );
    this.resolutionHeightValidationMessage = document.getElementById(
      'resolutionHeightValidationMessage'
    );
    this.reprojectionGridColsValidationMessage = document.getElementById(
      'reprojectionGridColsValidationMessage'
    );
    this.reprojectionGridRowsValidationMessage = document.getElementById(
      'reprojectionGridRowsValidationMessage'
    );
    this.enablePoseSmoothingSelect = this.getElement<HTMLSelectElement>('enablePoseSmoothing');
    this.posePredictionFactorInput = this.getElement<HTMLInputElement>('posePredictionFactor');
    this.posePredictionFactorValue = this.getElement<HTMLElement>('posePredictionFactorValue');
    this.enableTexSubImage2DSelect = this.getElement<HTMLSelectElement>('enableTexSubImage2D');
    this.useQuestColorWorkaroundSelect =
      this.getElement<HTMLSelectElement>('useQuestColorWorkaround');
    this.serverTypeSelect = this.getElement<HTMLSelectElement>('serverType');
    this.deviceProfileSelect = this.getElement<HTMLSelectElement>('deviceProfile');
    this.panelHiddenAtStartSelect = this.getElement<HTMLSelectElement>('panelHiddenAtStart');
    this.referenceSpaceSelect = this.getElement<HTMLSelectElement>('referenceSpace');
    this.xrOffsetXInput = this.getElement<HTMLInputElement>('xrOffsetX');
    this.xrOffsetYInput = this.getElement<HTMLInputElement>('xrOffsetY');
    this.xrOffsetZInput = this.getElement<HTMLInputElement>('xrOffsetZ');
    this.controlPanelPositionSelect = this.getElement<HTMLSelectElement>('controlPanelPosition');
    this.controlPanelDistanceInput = this.getElement<HTMLInputElement>('controlPanelDistance');
    this.controlPanelHeightInput = this.getElement<HTMLInputElement>('controlPanelHeight');
    this.controlPanelAngleDegreesInput = this.getElement<HTMLInputElement>(
      'controlPanelAngleDegrees'
    );
    this.controlPanelTrackHeadsetInput = this.getElement<HTMLInputElement>(
      'controlPanelTrackHeadset'
    );
    this.proxyDefaultText = this.getElement<HTMLElement>('proxyDefaultText');
    this.deviceProfileWarning = this.getElement<HTMLElement>('deviceProfileWarning');
    this.errorMessageBox = this.getElement<HTMLElement>('errorMessageBox');
    this.errorMessageText = this.getElement<HTMLElement>('errorMessageText');
    this.validationMessageBox = this.getElement<HTMLElement>('validationMessageBox');
    this.validationMessageText = this.getElement<HTMLElement>('validationMessageText');
    this.certAcceptanceLink = this.getElement<HTMLElement>('certAcceptanceLink');
    this.certLink = this.getElement<HTMLAnchorElement>('certLink');
    this.mediaAddressInput = this.getElement<HTMLInputElement>('mediaAddress');
    this.mediaPortInput = this.getElement<HTMLInputElement>('mediaPort');
    this.streamTestModeSelect = this.getElement<HTMLSelectElement>('streamTestMode');
    this.streamTestDurationSecondsInput = this.getElement<HTMLInputElement>(
      'streamTestDurationSeconds'
    );
    this.controllerModelVisibilitySelect = this.getElement<HTMLSelectElement>(
      'controllerModelVisibility'
    );
    this.showTraceInXRSelect = this.getElement<HTMLSelectElement>('showTraceInXR');
    this.showRecordingControlsSelect = this.getElement<HTMLSelectElement>('showRecordingControls');
    this.replayPacingSelect = this.getElement<HTMLSelectElement>('replayPacing');
    this.headlessInput = this.getElement<HTMLInputElement>('cloudxrHeadless');
    this.autoRefreshModeSelect = this.getElement<HTMLSelectElement>('cloudxrAutoRefreshMode');
    this.reconnectEnabledInput = this.getElement<HTMLInputElement>('cloudxrReconnectEnabled');
    this.reconnectMaxAttemptsInput = this.getElement<HTMLInputElement>(
      'cloudxrReconnectMaxAttempts'
    );
    this.reconnectDelayMsInput = this.getElement<HTMLInputElement>('cloudxrReconnectDelayMs');
    this.streamAttachTimeoutMsInput = this.getElement<HTMLInputElement>(
      'cloudxrStreamAttachTimeoutMs'
    );
    this.warmupBeginTimeoutMsInput = this.getElement<HTMLInputElement>(
      'cloudxrWarmupBeginTimeoutMs'
    );
    this.warmupEndTimeoutMsInput = this.getElement<HTMLInputElement>('cloudxrWarmupEndTimeoutMs');
    this.teleopModeSubtitle = this.getElement<HTMLElement>('teleopModeSubtitle');
    this.teleopProjectSelect = this.getElement<HTMLSelectElement>('teleopProjectSelect');
    this.resetSettingsButton = this.getElement<HTMLButtonElement>('resetSettingsButton');
    // Optional: absent in trimmed builds; renderUrlParamsHelp() no-ops when null.
    this.urlParamsHelpList = document.getElementById('urlParamsHelpList');
  }

  /**
   * Gets a DOM element by ID with type safety
   * @param id - The element ID to find
   * @returns The found element with the specified type
   * @throws Error if element is not found
   */
  private getElement<T extends HTMLElement>(id: string): T {
    const element = document.getElementById(id) as T;
    if (!element) {
      throw new Error(`Element with id '${id}' not found`);
    }
    return element;
  }

  /**
   * Gets the default configuration values
   * @returns Default configuration object
   */
  private getDefaultConfiguration(): AppConfig {
    const useSecure = typeof window !== 'undefined' ? window.location.protocol === 'https:' : false;
    // Default port: HTTP → 49100, HTTPS without proxy → 48322, HTTPS with proxy → 443
    const defaultPort = useSecure ? 48322 : 49100;
    return {
      serverIP: (typeof window !== 'undefined' && window.location.hostname) || '127.0.0.1',
      port: defaultPort,
      useSecureConnection: useSecure,
      perEyeWidth: 2048,
      perEyeHeight: 1792,
      reprojectionGridCols: 0,
      reprojectionGridRows: 0,
      // Keep in sync with the Quest profile defaults (helpers/DeviceProfiles.ts)
      // and the 'selected' options in index.html.
      deviceFrameRate: 72,
      maxStreamingBitrateMbps: 25,
      codec: 'av1',
      immersiveMode: 'ar',
      deviceProfileId: 'custom',
      serverType: 'manual',
      panelHiddenAtStart: false,
      proxyUrl: '',
      referenceSpaceType: 'auto',
      controlPanelPosition: 'right',
      controlPanelDistance: 1.8,
      controlPanelHeight: 1.85,
      controlPanelAngleDegrees: 70,
      controlPanelTrackHeadset: false,
      enablePoseSmoothing: true,
      posePredictionFactor: 1.0,
      enableTexSubImage2D: false,
      useQuestColorWorkaround: false,
      hideControllerModel: false,
      showTrace: false,
      showRecordingControls: false,
      replayPacing: 'time',
      // Off by default: the test holds the session in Connecting for its whole window,
      // and a teleop operator connecting to a robot should not be gated on it. Opt in
      // via the settings panel or ?streamTestMode=warn when diagnosing a link.
      streamTestMode: 'off',
      streamTestDurationSeconds: 5,
      headless: false,
      autoRefreshMode: 'clean',
      reconnectEnabled: true,
      reconnectMaxAttempts: 3,
      reconnectDelayMs: 3000,
      streamAttachTimeoutMs: 120000,
      warmupBeginTimeoutMs: 10000,
      warmupEndTimeoutMs: 30000,
      teleopPath: DEFAULT_TELEOP_PATH,
    };
  }

  /**
   * Single source of truth for the global (non-per-project-path) localStorage-backed
   * controls: maps each form control to its storage key. Used both to wire persistence
   * ({@link CloudXR2DUI.setupLocalStorage}) and to clear it ({@link CloudXR2DUI.resetToDefaults}),
   * so adding a setting in one place can't leave the reset path stale.
   */
  private localStorageBindings(): Array<{
    el: HTMLInputElement | HTMLSelectElement;
    key: string;
  }> {
    return [
      { el: this.serverTypeSelect, key: 'serverType' },
      { el: this.serverIpInput, key: 'serverIp' },
      { el: this.portInput, key: 'port' },
      { el: this.perEyeWidthInput, key: 'perEyeWidth' },
      { el: this.perEyeHeightInput, key: 'perEyeHeight' },
      { el: this.reprojectionGridColsInput, key: 'reprojectionGridCols' },
      { el: this.reprojectionGridRowsInput, key: 'reprojectionGridRows' },
      { el: this.proxyUrlInput, key: 'proxyUrl' },
      { el: this.deviceFrameRateSelect, key: 'deviceFrameRate' },
      { el: this.maxStreamingBitrateMbpsSelect, key: 'maxStreamingBitrateMbps' },
      { el: this.codecSelect, key: 'codec' },
      { el: this.enablePoseSmoothingSelect, key: 'enablePoseSmoothing' },
      { el: this.posePredictionFactorInput, key: 'posePredictionFactor' },
      { el: this.enableTexSubImage2DSelect, key: 'enableTexSubImage2D' },
      { el: this.useQuestColorWorkaroundSelect, key: 'useQuestColorWorkaround' },
      { el: this.immersiveSelect, key: 'immersiveMode' },
      { el: this.deviceProfileSelect, key: 'deviceProfile' },
      { el: this.controlPanelPositionSelect, key: 'controlPanelPosition' },
      { el: this.controlPanelDistanceInput, key: 'controlPanelDistance' },
      { el: this.controlPanelHeightInput, key: 'controlPanelHeight' },
      { el: this.controlPanelAngleDegreesInput, key: 'controlPanelAngleDegrees' },
      { el: this.controlPanelTrackHeadsetInput, key: 'controlPanelTrackHeadset' },
      { el: this.referenceSpaceSelect, key: 'referenceSpace' },
      { el: this.xrOffsetXInput, key: 'xrOffsetX' },
      { el: this.xrOffsetYInput, key: 'xrOffsetY' },
      { el: this.xrOffsetZInput, key: 'xrOffsetZ' },
      { el: this.mediaAddressInput, key: 'mediaAddress' },
      { el: this.mediaPortInput, key: 'mediaPort' },
      { el: this.streamTestModeSelect, key: 'streamTestMode' },
      { el: this.streamTestDurationSecondsInput, key: 'streamTestDurationSeconds' },
      { el: this.controllerModelVisibilitySelect, key: 'controllerModelVisibility' },
      { el: this.showTraceInXRSelect, key: 'showTraceInXR' },
      { el: this.showRecordingControlsSelect, key: 'showRecordingControls' },
      { el: this.replayPacingSelect, key: 'replayPacing' },
      { el: this.autoRefreshModeSelect, key: 'autoRefreshMode' },
      { el: this.reconnectEnabledInput, key: 'reconnectEnabled' },
      { el: this.reconnectMaxAttemptsInput, key: 'reconnectMaxAttempts' },
      { el: this.reconnectDelayMsInput, key: 'reconnectDelayMs' },
      { el: this.streamAttachTimeoutMsInput, key: 'streamAttachTimeoutMs' },
      { el: this.warmupBeginTimeoutMsInput, key: 'warmupBeginTimeoutMs' },
      { el: this.warmupEndTimeoutMsInput, key: 'warmupEndTimeoutMs' },
    ];
  }

  /**
   * Wires up localStorage persistence for global (non-per-project-path) form
   * inputs. Per-project-path fields are handled separately via
   * loadPerProject/savePerProject around their own change listeners.
   */
  private setupLocalStorage(): void {
    for (const { el, key } of this.localStorageBindings()) {
      enableLocalStorage(el, key);
    }
  }

  /**
   * Clears stored settings and reloads to a clean default state. Removes the global
   * localStorage keys, the per-project debug settings for the active teleop application,
   * and the teleop-start countdown preference. URL params are handled below.
   */
  public resetToDefaults(): void {
    try {
      for (const { key } of this.localStorageBindings()) {
        localStorage.removeItem(key);
      }
      // Per-project debug settings persist under `cxr.isaac.<key>|<teleopPath>`
      // (see helpers/react/utils savePerProject); reset only the active application's.
      for (const key of CloudXR2DUI.PER_PROJECT_SETTING_KEYS) {
        localStorage.removeItem(`cxr.isaac.${key}|${this.teleopPath}`);
      }
      // Teleop-start countdown (owned by App.tsx's countdown feature).
      localStorage.removeItem(COUNTDOWN_STORAGE_KEY);
      // Advanced groups' expanded/collapsed state (cxr.group.<id>).
      for (let i = localStorage.length - 1; i >= 0; i--) {
        const k = localStorage.key(i);
        if (k && k.startsWith(CloudXR2DUI.GROUP_STATE_PREFIX)) {
          localStorage.removeItem(k);
        }
      }
    } catch (error) {
      console.warn('Failed to clear stored settings:', error);
    }

    // applyUrlSeeds() runs after setupLocalStorage() on load, so a form-backed query
    // param would immediately re-override the cleared value and the reset would look
    // like it did nothing. Strip those params from the address bar, but keep the direct
    // transport/OOB params (turnServer, controlToken, …, set by oob_teleop_env.py) and
    // the teleop path hash. Then reload explicitly: replaceState followed by reload()
    // always re-reads cleared storage, even when no params were present (a same-URL
    // location.replace can be a no-op in some browsers, which would leave the stale form).
    const url = new URL(window.location.href);
    for (const param of URL_PARAMS) {
      if (param.elementId) {
        url.searchParams.delete(param.url ?? param.key);
      }
    }
    window.history.replaceState(null, '', url.toString());
    window.location.reload();
  }

  /**
   * Populates the "URL parameters" help list from the param registry, listing only
   * params that opted in with a `description` (keeps secrets/transport internals out).
   * Reads from URL_PARAMS so the list can never drift from what the client accepts.
   */
  private renderUrlParamsHelp(): void {
    if (!this.urlParamsHelpList) return;
    this.urlParamsHelpList.replaceChildren();
    for (const param of URL_PARAMS) {
      if (!param.description) continue;
      const li = document.createElement('li');
      const code = document.createElement('code');
      code.textContent = param.url ?? param.key;
      li.appendChild(code);
      li.appendChild(document.createTextNode(` — ${param.description}`));
      this.urlParamsHelpList.appendChild(li);
    }
  }

  /** localStorage key prefix for each collapsible advanced group's open/closed state. */
  private static readonly GROUP_STATE_PREFIX = 'cxr.group.';

  /**
   * Settings persisted per teleop application under `cxr.isaac.<key>|<teleopPath>`
   * (see {@link applyPerProjectSettings} and helpers/react/utils savePerProject).
   * Centralized so resetToDefaults clears exactly the keys the per-project handlers write.
   */
  private static readonly PER_PROJECT_SETTING_KEYS = ['panelHiddenAtStart', 'headless'];

  /**
   * Restore each advanced group's expanded/collapsed state from localStorage and persist it on
   * toggle, so a user's "open" sections stay open across reloads. Keyed by the group's element id.
   */
  private restoreGroupExpandedState(): void {
    const groups = document.querySelectorAll<HTMLDetailsElement>('details.settings-group[id]');
    for (const group of Array.from(groups)) {
      const key = `${CloudXR2DUI.GROUP_STATE_PREFIX}${group.id}`;
      try {
        const saved = localStorage.getItem(key);
        if (saved === 'true') group.open = true;
        else if (saved === 'false') group.open = false;
      } catch (_) {}
      const handler = () => {
        try {
          localStorage.setItem(key, String(group.open));
        } catch (_) {}
      };
      group.addEventListener('toggle', handler);
      this.eventListeners.push({ element: group, event: 'toggle', handler });
    }
  }

  /**
   * Configures proxy settings based on the current protocol (HTTP/HTTPS)
   * Sets appropriate placeholders and help text for port and proxy URL inputs
   */
  private setupProxyConfiguration(): void {
    // Update port placeholder based on protocol
    if (window.location.protocol === 'https:') {
      this.portInput.placeholder = 'Port (default: 48322, or 443 if proxy URL set)';
    } else {
      this.portInput.placeholder = 'Port (default: 49100)';
    }

    // Set default text and placeholder based on protocol
    if (window.location.protocol === 'https:') {
      this.proxyDefaultText.textContent =
        'Optional: Leave empty for direct WSS connection, or provide URL for proxy routing (e.g., https://proxy.example.com/)';
      this.proxyUrlInput.placeholder = '';
    } else {
      this.proxyDefaultText.textContent = 'Not needed for HTTP - uses direct WS connection';
      this.proxyUrlInput.placeholder = '';
    }
  }

  /**
   * Sets up event listeners for form input changes
   * Handles both input and change events for better compatibility
   */
  private setupEventListeners(): void {
    // Update configuration when form inputs change
    const updateConfig = () => this.updateConfiguration();
    const onProfileLinkedChange = () => {
      this.setProfileToCustomIfNeeded();
      updateConfig();
    };

    // Helper function to add listeners and store them for cleanup
    const addListener = (element: HTMLElement, event: string, handler: EventListener) => {
      element.addEventListener(event, handler);
      this.eventListeners.push({ element, event, handler });
    };

    // Add event listeners for all form fields
    addListener(this.serverTypeSelect, 'change', updateConfig);
    addListener(this.serverIpInput, 'input', updateConfig);
    addListener(this.serverIpInput, 'change', updateConfig);

    // Show the clear ("x") button only while the server IP field has a value,
    // and clear the prefill (incl. localStorage) on click so the browser's
    // autocomplete dropdown of previously connected servers can show again.
    const updateServerIpClearButton = () => {
      this.serverIpClearButton.classList.toggle('visible', this.serverIpInput.value.length > 0);
    };
    addListener(this.serverIpInput, 'input', updateServerIpClearButton);
    addListener(this.serverIpClearButton, 'click', () => {
      this.serverIpInput.value = '';
      // Update the live config + clear-button state directly; 'change' persists
      // the now-empty value via the enableLocalStorage handler.
      updateServerIpClearButton();
      updateConfig();
      this.serverIpInput.dispatchEvent(new Event('change', { bubbles: true }));
      this.serverIpInput.focus();
    });
    updateServerIpClearButton();
    addListener(this.portInput, 'input', updateConfig);
    addListener(this.portInput, 'change', updateConfig);
    const updateResValidation = () => this.updateResolutionValidationMessage();
    addListener(this.perEyeWidthInput, 'input', onProfileLinkedChange);
    addListener(this.perEyeWidthInput, 'change', onProfileLinkedChange);
    addListener(this.perEyeWidthInput, 'blur', updateResValidation);
    addListener(this.perEyeWidthInput, 'keyup', updateResValidation);
    addListener(this.perEyeHeightInput, 'input', onProfileLinkedChange);
    addListener(this.perEyeHeightInput, 'change', onProfileLinkedChange);
    addListener(this.perEyeHeightInput, 'blur', updateResValidation);
    addListener(this.perEyeHeightInput, 'keyup', updateResValidation);
    this.updateResolutionValidationMessage();
    const updateGridValidation = () => this.updateGridValidationMessage();
    addListener(this.reprojectionGridColsInput, 'input', onProfileLinkedChange);
    addListener(this.reprojectionGridColsInput, 'change', onProfileLinkedChange);
    addListener(this.reprojectionGridColsInput, 'blur', updateGridValidation);
    addListener(this.reprojectionGridColsInput, 'keyup', updateGridValidation);
    addListener(this.reprojectionGridRowsInput, 'input', onProfileLinkedChange);
    addListener(this.reprojectionGridRowsInput, 'change', onProfileLinkedChange);
    addListener(this.reprojectionGridRowsInput, 'blur', updateGridValidation);
    addListener(this.reprojectionGridRowsInput, 'keyup', updateGridValidation);
    this.updateGridValidationMessage();
    addListener(this.deviceFrameRateSelect, 'change', onProfileLinkedChange);
    addListener(this.maxStreamingBitrateMbpsSelect, 'change', onProfileLinkedChange);
    addListener(this.codecSelect, 'change', onProfileLinkedChange);
    addListener(this.enablePoseSmoothingSelect, 'change', onProfileLinkedChange);
    addListener(this.posePredictionFactorInput, 'change', onProfileLinkedChange);
    addListener(this.posePredictionFactorInput, 'input', () => {
      this.setProfileToCustomIfNeeded();
      this.posePredictionFactorValue.textContent = this.posePredictionFactorInput.value;
      this.updateConfiguration();
    });
    addListener(this.enableTexSubImage2DSelect, 'change', onProfileLinkedChange);
    addListener(this.useQuestColorWorkaroundSelect, 'change', onProfileLinkedChange);
    addListener(this.immersiveSelect, 'change', updateConfig);
    addListener(this.panelHiddenAtStartSelect, 'change', () => {
      // Pass the raw select value string through savePerProject; the matching
      // loadPerProject parses `'true'`/`'false'` back into a boolean.
      savePerProject('panelHiddenAtStart', this.teleopPath, this.panelHiddenAtStartSelect.value);
      updateConfig();
    });
    addListener(this.referenceSpaceSelect, 'change', updateConfig);
    addListener(this.xrOffsetXInput, 'input', updateConfig);
    addListener(this.xrOffsetXInput, 'change', updateConfig);
    addListener(this.xrOffsetYInput, 'input', updateConfig);
    addListener(this.xrOffsetYInput, 'change', updateConfig);
    addListener(this.xrOffsetZInput, 'input', updateConfig);
    addListener(this.xrOffsetZInput, 'change', updateConfig);
    addListener(this.controlPanelPositionSelect, 'change', updateConfig);
    addListener(this.streamAttachTimeoutMsInput, 'input', updateConfig);
    addListener(this.streamAttachTimeoutMsInput, 'change', updateConfig);
    addListener(this.warmupBeginTimeoutMsInput, 'input', updateConfig);
    addListener(this.warmupBeginTimeoutMsInput, 'change', updateConfig);
    addListener(this.warmupEndTimeoutMsInput, 'input', updateConfig);
    addListener(this.warmupEndTimeoutMsInput, 'change', updateConfig);
    addListener(this.controlPanelDistanceInput, 'input', updateConfig);
    addListener(this.controlPanelDistanceInput, 'change', updateConfig);
    addListener(this.controlPanelHeightInput, 'input', updateConfig);
    addListener(this.controlPanelHeightInput, 'change', updateConfig);
    addListener(this.controlPanelAngleDegreesInput, 'input', updateConfig);
    addListener(this.controlPanelAngleDegreesInput, 'change', updateConfig);
    addListener(this.controlPanelTrackHeadsetInput, 'change', updateConfig);
    addListener(this.teleopProjectSelect, 'change', () => {
      const value = this.teleopProjectSelect.value;
      if (!value) return;
      // Reset to the prompt before navigating so if the reload is aborted the
      // control doesn't end up stuck on the just-picked value.
      this.teleopProjectSelect.selectedIndex = 0;
      window.location.hash = value.replace(/^#/, '');
    });
    addListener(this.proxyUrlInput, 'input', updateConfig);
    addListener(this.proxyUrlInput, 'change', updateConfig);
    addListener(this.mediaAddressInput, 'input', updateConfig);
    addListener(this.mediaAddressInput, 'change', updateConfig);
    addListener(this.mediaPortInput, 'input', updateConfig);
    addListener(this.mediaPortInput, 'change', updateConfig);
    addListener(this.streamTestModeSelect, 'change', updateConfig);
    addListener(this.streamTestDurationSecondsInput, 'input', updateConfig);
    addListener(this.streamTestDurationSecondsInput, 'change', updateConfig);
    addListener(this.controllerModelVisibilitySelect, 'change', updateConfig);
    addListener(this.showTraceInXRSelect, 'change', updateConfig);
    addListener(this.showRecordingControlsSelect, 'change', updateConfig);
    addListener(this.replayPacingSelect, 'change', updateConfig);
    addListener(this.headlessInput, 'change', () => {
      savePerProject('headless', this.teleopPath, this.headlessInput.checked ? 'true' : 'false');
      this.applyHeadlessImmersiveDropdown();
      this.updateConfiguration();
    });
    addListener(this.autoRefreshModeSelect, 'change', updateConfig);

    addListener(this.resetSettingsButton, 'click', () => {
      if (window.confirm('Reset all settings to their defaults? This reloads the page.')) {
        this.resetToDefaults();
      }
    });

    // Headset on-screen keyboards sometimes lack a minus key, so offsets can't be typed
    // negative. Each ± button (data-target = input id) flips its field's sign. Dispatch
    // 'change' so the existing offset listeners (updateConfiguration + localStorage) run.
    for (const btn of Array.from(document.querySelectorAll<HTMLButtonElement>('.input-sign-btn'))) {
      const targetId = btn.dataset.target;
      if (!targetId) continue;
      const input = document.getElementById(targetId) as HTMLInputElement | null;
      if (!input) continue;
      addListener(btn, 'click', () => {
        // Pure sign flip: a no-op on an empty/non-numeric field rather than inserting 0.
        const value = parseFloat(input.value);
        if (!Number.isFinite(value)) return;
        input.value = String(-value);
        input.dispatchEvent(new Event('change', { bubbles: true }));
      });
    }

    addListener(this.deviceProfileSelect, 'change', () => {
      this.applyDeviceProfileToForm(resolveDeviceProfileId(this.deviceProfileSelect.value));
      this.persistProfileFieldsToLocalStorage();
      this.updateConfiguration();
    });

    // Set up certificate acceptance link and store cleanup function
    this.certLinkCleanup = setupCertificateAcceptanceLink(
      this.serverIpInput,
      this.portInput,
      this.proxyUrlInput,
      this.certAcceptanceLink,
      this.certLink
    );
  }

  /** Update inline resolution validation under each input. */
  private updateResolutionValidationMessage(): void {
    const { w: wNum, h: hNum } = getResolutionFromInputs(
      this.perEyeWidthInput,
      this.perEyeHeightInput
    );
    const { widthError, heightError } = validatePerEyeResolution(wNum, hNum);
    if (this.resolutionWidthValidationMessage) {
      const showWidth = widthError ?? '';
      this.resolutionWidthValidationMessage.textContent = showWidth;
      this.resolutionWidthValidationMessage.className = showWidth
        ? 'config-text resolution-validation-error'
        : 'config-text';
    }
    if (this.resolutionHeightValidationMessage) {
      const showHeight = heightError ?? '';
      this.resolutionHeightValidationMessage.textContent = showHeight;
      this.resolutionHeightValidationMessage.className = showHeight
        ? 'config-text resolution-validation-error'
        : 'config-text';
    }
    this.updateConnectButtonState();
  }

  /** Update inline grid validation under each input. */
  private updateGridValidationMessage(): void {
    const { reprojectionGridCols, reprojectionGridRows } = getGridFromInputs(
      this.reprojectionGridColsInput,
      this.reprojectionGridRowsInput
    );
    const { reprojectionGridColsError, reprojectionGridRowsError } = validateDepthReprojectionGrid(
      reprojectionGridCols,
      reprojectionGridRows
    );
    if (this.reprojectionGridColsValidationMessage) {
      const showGridCols = reprojectionGridColsError ?? '';
      this.reprojectionGridColsValidationMessage.textContent = showGridCols;
      this.reprojectionGridColsValidationMessage.className = showGridCols
        ? 'config-text resolution-validation-error'
        : 'config-text';
    }
    if (this.reprojectionGridRowsValidationMessage) {
      const showGridRows = reprojectionGridRowsError ?? '';
      this.reprojectionGridRowsValidationMessage.textContent = showGridRows;
      this.reprojectionGridRowsValidationMessage.className = showGridRows
        ? 'config-text resolution-validation-error'
        : 'config-text';
    }
    this.updateConnectButtonState();
  }

  /** Disable Connect button and show validation error when resolution invalid; enable when valid. */
  public updateConnectButtonState(): void {
    const { w, h } = getResolutionFromInputs(this.perEyeWidthInput, this.perEyeHeightInput);
    const { reprojectionGridCols, reprojectionGridRows } = getGridFromInputs(
      this.reprojectionGridColsInput,
      this.reprojectionGridRowsInput
    );
    const resolutionError = getResolutionValidationError(w, h);
    const gridError = getGridValidationError(reprojectionGridCols, reprojectionGridRows);
    const connectMessage = getResolutionValidationMessageForConnect(w, h);
    const gridConnectMessage = getGridValidationMessageForConnect(
      reprojectionGridCols,
      reprojectionGridRows
    );
    const combinedConnectMessage = [connectMessage, gridConnectMessage].filter(Boolean).join('\n');
    if (combinedConnectMessage) {
      this.validationMessageText.textContent = combinedConnectMessage;
      this.validationMessageBox.className = 'validation-message-box show';
    } else {
      this.validationMessageText.textContent = '';
      this.validationMessageBox.className = 'validation-message-box';
    }
    // Only update button when idle (don't override "CONNECT (starting...)" or "CONNECT (XR session active)")
    if (this.startButton && this.startButton.innerHTML === 'CONNECT') {
      const shouldEnable = !resolutionError && !gridError;
      this.setStartButtonState(!shouldEnable, 'CONNECT');
    }
  }

  /**
   * Updates the current configuration from form values
   * Calls the configuration change callback if provided
   */
  private updateConfiguration(): void {
    const useSecure = this.getDefaultConfiguration().useSecureConnection;
    const portValue = parseInt(this.portInput.value);
    const hasProxy = this.proxyUrlInput.value.trim().length > 0;

    // Smart default port based on connection type and proxy usage
    let defaultPort = 49100; // HTTP default
    if (useSecure) {
      defaultPort = hasProxy ? 443 : 48322; // HTTPS with proxy → 443, HTTPS without → 48322
    }

    const { w: perEyeWidth, h: perEyeHeight } = getResolutionFromInputs(
      this.perEyeWidthInput,
      this.perEyeHeightInput
    );
    const { reprojectionGridCols, reprojectionGridRows } = getGridFromInputs(
      this.reprojectionGridColsInput,
      this.reprojectionGridRowsInput
    );
    const newConfiguration: AppConfig = {
      serverIP: this.serverIpInput.value || this.getDefaultConfiguration().serverIP,
      port: portValue || defaultPort,
      useSecureConnection: useSecure,
      perEyeWidth,
      perEyeHeight,
      reprojectionGridCols,
      reprojectionGridRows,
      deviceFrameRate:
        parseInt(this.deviceFrameRateSelect.value) ||
        this.getDefaultConfiguration().deviceFrameRate,
      maxStreamingBitrateMbps:
        parseInt(this.maxStreamingBitrateMbpsSelect.value) ||
        this.getDefaultConfiguration().maxStreamingBitrateMbps,
      codec:
        (this.codecSelect.value as 'h264' | 'h265' | 'av1') || this.getDefaultConfiguration().codec,
      // Headless mode turns off the client's CloudXR frame blit but keeps tracking; the WebXR
      // session must be immersive-vr. immersive-ar uses passthrough semantics that do not match
      // that pipeline, so we ignore the AR/VR dropdown whenever headless is checked.
      immersiveMode: this.headlessInput.checked
        ? 'vr'
        : (this.immersiveSelect.value as 'ar' | 'vr') ||
          this.getDefaultConfiguration().immersiveMode,
      deviceProfileId: resolveDeviceProfileId(this.deviceProfileSelect.value),
      serverType: this.serverTypeSelect.value || this.getDefaultConfiguration().serverType,
      proxyUrl: this.proxyUrlInput.value || this.getDefaultConfiguration().proxyUrl,
      referenceSpaceType:
        (this.referenceSpaceSelect.value as 'auto' | 'local-floor' | 'local' | 'viewer') ||
        this.getDefaultConfiguration().referenceSpaceType,
      enablePoseSmoothing: this.enablePoseSmoothingSelect.value === 'true',
      posePredictionFactor: parseFloat(this.posePredictionFactorInput.value),
      enableTexSubImage2D: this.enableTexSubImage2DSelect.value === 'true',
      useQuestColorWorkaround: this.useQuestColorWorkaroundSelect.value === 'true',
      // Convert cm from UI into meters for config (respect 0; if invalid, use 0)
      xrOffsetX: (() => {
        const v = parseFloat(this.xrOffsetXInput.value);
        return Number.isFinite(v) ? v / 100 : 0;
      })(),
      xrOffsetY: (() => {
        const v = parseFloat(this.xrOffsetYInput.value);
        return Number.isFinite(v) ? v / 100 : 0;
      })(),
      xrOffsetZ: (() => {
        const v = parseFloat(this.xrOffsetZInput.value);
        return Number.isFinite(v) ? v / 100 : 0;
      })(),
      controlPanelPosition: parseControlPanelPosition(
        this.controlPanelPositionSelect.value,
        this.getDefaultConfiguration().controlPanelPosition ?? 'center'
      ),
      controlPanelDistance: (() => {
        const v = parseFloat(this.controlPanelDistanceInput.value);
        // Zero or negative places the panel at or behind the viewer (see
        // worldPositionFromHeadOffset in CloudXRUI.tsx) - reject those like an invalid
        // (non-finite) value, not just NaN/Infinity.
        return Number.isFinite(v) && v > 0
          ? v
          : this.getDefaultConfiguration().controlPanelDistance;
      })(),
      controlPanelHeight: (() => {
        const v = parseFloat(this.controlPanelHeightInput.value);
        return Number.isFinite(v) ? v : this.getDefaultConfiguration().controlPanelHeight;
      })(),
      controlPanelAngleDegrees: (() => {
        const v = parseFloat(this.controlPanelAngleDegreesInput.value);
        return Number.isFinite(v) ? v : this.getDefaultConfiguration().controlPanelAngleDegrees;
      })(),
      controlPanelTrackHeadset: this.controlPanelTrackHeadsetInput.checked,
      // Parse media address and port if provided
      mediaAddress: this.mediaAddressInput.value.trim() || undefined,
      mediaPort: (() => {
        const v = parseInt(this.mediaPortInput.value, 10);
        return !isNaN(v) ? v : undefined;
      })(),
      streamTestMode: parseStreamTestMode(
        this.streamTestModeSelect.value,
        this.getDefaultConfiguration().streamTestMode ?? 'off'
      ),
      streamTestDurationSeconds: (() => {
        const v = parseInt(this.streamTestDurationSecondsInput.value, 10);
        return !isNaN(v) ? v : this.getDefaultConfiguration().streamTestDurationSeconds;
      })(),
      hideControllerModel: this.controllerModelVisibilitySelect.value === 'hide',
      showTrace: this.showTraceInXRSelect.value === 'true',
      showRecordingControls: this.showRecordingControlsSelect.value === 'true',
      replayPacing: this.replayPacingSelect.value === 'frame' ? 'frame' : 'time',
      // See immersiveMode above: when true, callers must start an immersive-vr WebXR session.
      headless: this.headlessInput.checked,
      autoRefreshMode: parseAutoRefreshMode(
        this.autoRefreshModeSelect.value,
        this.getDefaultConfiguration().autoRefreshMode ?? 'clean'
      ),
      reconnectEnabled: this.reconnectEnabledInput.checked,
      reconnectMaxAttempts: (() => {
        const v = parseInt(this.reconnectMaxAttemptsInput.value, 10);
        return !isNaN(v) && v >= 0 ? v : this.getDefaultConfiguration().reconnectMaxAttempts;
      })(),
      reconnectDelayMs: (() => {
        const v = parseInt(this.reconnectDelayMsInput.value, 10);
        return !isNaN(v) && v >= 0 ? v : this.getDefaultConfiguration().reconnectDelayMs;
      })(),
      streamAttachTimeoutMs: (() => {
        const v = parseInt(this.streamAttachTimeoutMsInput.value, 10);
        return !isNaN(v) && v >= 0 ? v : this.getDefaultConfiguration().streamAttachTimeoutMs;
      })(),
      warmupBeginTimeoutMs: (() => {
        const v = parseInt(this.warmupBeginTimeoutMsInput.value, 10);
        return !isNaN(v) && v >= 0 ? v : this.getDefaultConfiguration().warmupBeginTimeoutMs;
      })(),
      warmupEndTimeoutMs: (() => {
        const v = parseInt(this.warmupEndTimeoutMsInput.value, 10);
        return !isNaN(v) && v >= 0 ? v : this.getDefaultConfiguration().warmupEndTimeoutMs;
      })(),
      panelHiddenAtStart: this.panelHiddenAtStartSelect.value === 'true',
      teleopPath: this.teleopPath,
    };

    this.currentConfiguration = newConfiguration;

    // Call the configuration change callback if provided
    if (this.onConfigurationChange) {
      this.onConfigurationChange(newConfiguration);
    }
  }

  /**
   * Applies a device profile to the form: sets the CloudXR-related fields that the profile defines.
   * Profile values are defined in @helpers/DeviceProfiles (DEVICE_PROFILES, QUEST3_PROFILE, etc.).
   * The fields below are the only ones profiles change; editing any of them should switch the
   * device profile to Custom (see onProfileLinkedChange / setProfileToCustomIfNeeded).
   */
  private applyDeviceProfileToForm(profileId: DeviceProfileId): void {
    const profile = getDeviceProfile(profileId);
    const cloudxr = profile.cloudxr;
    this.updateDeviceProfileWarning(profileId);

    if (!cloudxr || profileId === 'custom') {
      return;
    }

    if (cloudxr.perEyeWidth !== undefined) {
      this.perEyeWidthInput.value = String(cloudxr.perEyeWidth);
    }
    if (cloudxr.perEyeHeight !== undefined) {
      this.perEyeHeightInput.value = String(cloudxr.perEyeHeight);
    }
    this.reprojectionGridColsInput.value =
      cloudxr.reprojectionGridCols !== undefined ? String(cloudxr.reprojectionGridCols) : '';
    this.reprojectionGridRowsInput.value =
      cloudxr.reprojectionGridRows !== undefined ? String(cloudxr.reprojectionGridRows) : '';
    if (cloudxr.deviceFrameRate !== undefined) {
      setSelectValueIfAvailable(this.deviceFrameRateSelect, String(cloudxr.deviceFrameRate));
    }
    if (cloudxr.maxStreamingBitrateKbps !== undefined) {
      const mbps = Math.round(cloudxr.maxStreamingBitrateKbps / 1000);
      setSelectValueIfAvailable(this.maxStreamingBitrateMbpsSelect, String(mbps));
    }
    if (cloudxr.codec) {
      setSelectValueIfAvailable(this.codecSelect, cloudxr.codec);
    }
    if (cloudxr.enablePoseSmoothing !== undefined) {
      this.enablePoseSmoothingSelect.value = String(cloudxr.enablePoseSmoothing);
    }
    if (cloudxr.posePredictionFactor !== undefined) {
      this.posePredictionFactorInput.value = String(cloudxr.posePredictionFactor);
      this.posePredictionFactorValue.textContent = this.posePredictionFactorInput.value;
    }
    if (cloudxr.enableTexSubImage2D !== undefined) {
      this.enableTexSubImage2DSelect.value = String(cloudxr.enableTexSubImage2D);
    }
    if (cloudxr.useQuestColorWorkaround !== undefined) {
      this.useQuestColorWorkaroundSelect.value = String(cloudxr.useQuestColorWorkaround);
    }
  }

  /** When user edits a profile-driven setting, switch device profile to Custom and persist. */
  private setProfileToCustomIfNeeded(): void {
    if (this.deviceProfileSelect.value === 'custom') return;
    this.deviceProfileSelect.value = 'custom';
    this.updateDeviceProfileWarning('custom');
    try {
      localStorage.setItem('deviceProfile', 'custom');
    } catch (_) {}
  }

  /** Persist profile-driven form fields to localStorage so they are restored on load. */
  private persistProfileFieldsToLocalStorage(): void {
    try {
      localStorage.setItem('perEyeWidth', this.perEyeWidthInput.value);
      localStorage.setItem('perEyeHeight', this.perEyeHeightInput.value);
      localStorage.setItem('reprojectionGridCols', this.reprojectionGridColsInput.value);
      localStorage.setItem('reprojectionGridRows', this.reprojectionGridRowsInput.value);
      localStorage.setItem('deviceFrameRate', this.deviceFrameRateSelect.value);
      localStorage.setItem('maxStreamingBitrateMbps', this.maxStreamingBitrateMbpsSelect.value);
      localStorage.setItem('codec', this.codecSelect.value);
      localStorage.setItem('enablePoseSmoothing', this.enablePoseSmoothingSelect.value);
      localStorage.setItem('posePredictionFactor', this.posePredictionFactorInput.value);
      localStorage.setItem('enableTexSubImage2D', this.enableTexSubImage2DSelect.value);
      localStorage.setItem('useQuestColorWorkaround', this.useQuestColorWorkaroundSelect.value);
    } catch (e) {
      console.warn('Failed to persist profile fields to localStorage:', e);
    }
  }

  private updateDeviceProfileWarning(profileId: DeviceProfileId): void {
    if (!this.deviceProfileWarning) return;
    const profile = getDeviceProfile(profileId);
    const needsHttps = profile.connection?.httpsRequired === true;
    const isHttp = window.location.protocol === 'http:';

    if (needsHttps && isHttp) {
      this.deviceProfileWarning.textContent = 'This device requires HTTPS mode.';
      this.deviceProfileWarning.style.display = 'block';
    } else {
      this.deviceProfileWarning.style.display = 'none';
      this.deviceProfileWarning.textContent = '';
    }
  }

  /**
   * Gets the current configuration
   * @returns Current configuration object
   */
  public getConfiguration(): AppConfig {
    return { ...this.currentConfiguration };
  }

  /**
   * Sets the start button state
   * @param disabled - Whether the button should be disabled
   * @param text - Text to display on the button
   */
  public setStartButtonState(disabled: boolean, text: string): void {
    if (this.startButton) {
      this.startButton.disabled = disabled;
      this.startButton.innerHTML = text;
    }
  }

  /**
   * Sets up the connect button click handler
   * @param onConnect - Function to call when connect button is clicked
   * @param onError - Function to call when an error occurs
   */
  public setupConnectButtonHandler(
    onConnect: () => Promise<void>,
    onError: (error: Error) => void
  ): void {
    if (this.startButton) {
      // Remove any existing listener
      if (this.handleConnectClick) {
        this.startButton.removeEventListener('click', this.handleConnectClick);
      }

      // Create new handler
      this.handleConnectClick = async () => {
        this.updateConnectButtonState();
        if (this.startButton?.disabled) {
          this.updateConnectButtonState();
          return;
        }
        const cfg = this.getConfiguration();
        const resolutionError = getResolutionValidationError(cfg.perEyeWidth, cfg.perEyeHeight);
        const gridError = getGridValidationError(
          cfg.reprojectionGridCols,
          cfg.reprojectionGridRows
        );
        if (resolutionError || gridError) {
          this.updateConnectButtonState();
          return;
        }
        this.setStartButtonState(true, 'CONNECT (starting XR session...)');

        try {
          await onConnect();
        } catch (error) {
          this.setStartButtonState(false, 'CONNECT');
          this.updateConnectButtonState();
          onError(error as Error);
        }
      };

      // Add the new listener
      this.startButton.addEventListener('click', this.handleConnectClick);
    }
  }

  /**
   * Shows a status message in the UI with a specific type
   * @param message - Message to display
   * @param type - Message type: 'success', 'error', or 'info'
   */
  public showStatus(message: string, type: 'success' | 'error' | 'info'): void {
    if (this.errorMessageText && this.errorMessageBox) {
      this.errorMessageText.textContent = message;
      this.errorMessageBox.className = `error-message-box show ${type}`;
    }
    console[type === 'error' ? 'error' : 'info'](message);
  }

  /**
   * Shows an error message in the UI
   * @param message - Error message to display
   */
  public showError(message: string): void {
    this.showStatus(message, 'error');
  }

  /**
   * Hides the error message
   */
  public hideError(): void {
    if (this.errorMessageBox) {
      this.errorMessageBox.classList.remove('show');
    }
  }

  /**
   * Hides the status box only if it still shows `message`.
   *
   * The box is a single shared slot: capability results, CloudXR errors, and
   * host workstation notices all write to it, and each overwrites the last. A
   * caller retracting its own message must not blank whatever replaced it --
   * for example a CloudXR error raised moments earlier, which is reported
   * imperatively and has no React state to re-render it afterwards.
   *
   * @param message - The exact text the caller previously displayed
   */
  public hideStatusIfShowing(message: string): void {
    if (this.errorMessageText?.textContent === message) {
      this.hideError();
    }
  }

  /**
   * Cleans up event listeners and resources
   * Should be called when the component unmounts
   */
  public cleanup(): void {
    // Remove all stored event listeners
    this.eventListeners.forEach(({ element, event, handler }) => {
      element.removeEventListener(event, handler);
    });
    this.eventListeners = [];

    // Remove CONNECT button listener
    if (this.startButton && this.handleConnectClick) {
      this.startButton.removeEventListener('click', this.handleConnectClick);
      this.handleConnectClick = null;
    }

    // Clean up certificate acceptance link listeners
    if (this.certLinkCleanup) {
      this.certLinkCleanup();
      this.certLinkCleanup = null;
    }
  }
}
