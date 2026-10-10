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
 * CloudXRUI.tsx - CloudXR User Interface Component
 *
 * This component renders the in-VR user interface for the CloudXR application using
 * React Three UIKit. It provides:
 * - Server connection information and status display
 * - Interactive control buttons (Start Teleop, Reset Teleop, Disconnect)
 * - Responsive button layout with hover effects
 * - Integration with parent component event handlers
 * - Configurable position and rotation in world space for flexible UI placement
 * - Draggable handle bar for repositioning the UI in 3D space
 * - Face-camera rotation for optimal viewing angle (Y-axis only)
 * - Panel depth: full control panel, compact (when "minimize on play" and teleop active), or hidden
 *   (semi-transparent Show + slim drag handle).
 *
 * The UI is positioned in 3D space and designed for VR/AR interaction with
 * visual feedback and clear button labeling. All interactions are passed
 * back to the parent component through callback props.
 */

import { ReadonlySignal } from '@preact/signals-react';
import { useFrame, useThree } from '@react-three/fiber';
import { Handle, HandleState, HandleTarget } from '@react-three/handle';
import { Container, Image, Text } from '@react-three/uikit';
import { Button } from '@react-three/uikit-default';
import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Camera,
  Color,
  Group,
  Mesh,
  MeshStandardMaterial,
  Object3D,
  Quaternion,
  Vector3,
} from 'three';
import { damp } from 'three/src/math/MathUtils.js';

import { PerformanceCanvasImage } from '@helpers/react/PerformanceCanvasImage';
import { useXRButton } from '@helpers/react/useXRButton';
import {
  ControlPanelLayoutOptions,
  ControlPanelPosition,
  getControlPanelPositionVector,
} from '@helpers/react/utils';

import arrowLeftStartOnRectangleSvg from './icons/arrow-left-start-on-rectangle.svg';
import arrowUturnLeftSvg from './icons/arrow-uturn-left.svg';
import playCircleSvg from './icons/play-circle.svg';
import { useRecorder } from './RecorderContext';

// Face-camera rotation constants
const FACE_CAMERA_DAMPING = 10; // Higher = faster rotation toward camera

const WORLD_FORWARD = new Vector3(0, 0, -1);
const WORLD_UP = new Vector3(0, 1, 0);
// Synthesized via CDP by oob_teleop_adb.py's _cdp_send_reset_panel_key - keep in sync.
const RESET_PANEL_KEY = 'r'; // compared against event.key.toLowerCase()

// Minimum panel movement (metres) before continuous head-tracking emits another pose log line -
// large enough to skip per-frame jitter, small enough that a real reposition is still caught.
const TRACKING_LOG_EPSILON_M = 0.02;

/** Display size for the Performance metrics slot (width and height passed to PerformanceCanvasImage and its container). */
const METRIC_SLOT_WIDTH = 512;
/** Tracks PerformanceCanvasImage's 1024x760 canvas: the session-quality card plus four metric cards. */
const METRIC_SLOT_HEIGHT = 380;

interface CloudXRUIProps {
  onStartTeleop?: () => void;
  onDisconnect?: () => void;
  onResetTeleop?: () => void;
  serverAddress?: string;
  sessionStatus?: string;
  playLabel?: string;
  playInProgress?: boolean;
  countdownSeconds?: number;
  onCountdownIncrease?: () => void;
  onCountdownDecrease?: () => void;
  countdownDisabled?: boolean;
  position?: [number, number, number];
  rotation?: [number, number, number];
  /** Same setting used for the initial world-space `position` above - reused by
   * resetPanelRelativeToHead so a reset reproduces the same left/center/right choice. */
  controlPanelPosition?: ControlPanelPosition;
  /** Same layout constants used to compute `position` above. */
  controlPanelLayout?: ControlPanelLayoutOptions;
  /** When true, the panel continuously follows the headset every frame (resetPanelRelativeToHead
   * re-runs each frame instead of once) instead of staying at a fixed room position. Dragging
   * still works while this is on - see handleApply, which converts the dragged world position
   * back into a head-relative offset instead of a fixed one. */
  trackHeadset?: boolean;
  /** Computed signal for render FPS text - updates without React re-render */
  renderFpsText?: ReadonlySignal<string>;
  /** Computed signal for pose send FPS text - the rate operator intent reaches the robot */
  poseSendFpsText?: ReadonlySignal<string>;
  /** Computed signal for streaming FPS text - updates without React re-render */
  streamingFpsText?: ReadonlySignal<string>;
  /** Computed signal for pose-to-render latency text - updates without React re-render */
  poseToRenderText?: ReadonlySignal<string>;
  /** Live session quality 0-4 ({@link CloudXR.QualityScore}); drives the HUD quality bars. */
  sessionQuality?: ReadonlySignal<number>;
  /** Network test status line; empty when no test is running or configured. */
  streamTestText?: ReadonlySignal<string>;
  /** Traffic-light color for {@link streamTestText}. */
  streamTestColor?: ReadonlySignal<string>;
  /** From settings: hide control panel when immersive XR begins. */
  panelHiddenAtStart?: boolean;
  /** Immersive XR active; used to apply panelHiddenAtStart on session enter. */
  isXRMode?: boolean;
  /** Show input recording controls in the XR panel. */
  showRecordingControls?: boolean;
  /** Computed signal for the workstation notice title; empty when no notice is active. */
  systemNoticeTitleText?: ReadonlySignal<string>;
  /** Computed signal for the workstation notice body (one unmet requirement per line). */
  systemNoticeBodyText?: ReadonlySignal<string>;
  /** Whether a workstation notice is currently active. */
  systemNoticeVisible?: boolean;
  /** Severity of the active notice; selects the banner palette. */
  systemNoticeLevel?: 'warning' | 'info';
  /** Dismiss the workstation notice. */
  onDismissSystemNotice?: () => void;
}

/**
 * Warning banner shown in-headset when the host reports that its workstation is
 * below the recommended teleop spec.
 *
 * Advisory only: it never blocks the session, so it is dismissible and also
 * times out on its own. Text comes in as signals so updates bypass React,
 * matching how the performance metrics are rendered.
 */
/** Palette per notice level, so an informational notice does not read as a warning. */
const SYSTEM_NOTICE_PALETTE = {
  warning: {
    border: 'rgba(255, 193, 7, 0.9)',
    background: 'rgba(80, 60, 0, 0.85)',
    title: 'rgba(255, 213, 79, 1)',
  },
  info: {
    border: 'rgba(66, 165, 245, 0.9)',
    background: 'rgba(10, 45, 80, 0.85)',
    title: 'rgba(144, 202, 249, 1)',
  },
} as const;

function SystemNoticeBanner({
  titleText,
  bodyText,
  level = 'warning',
  onDismiss,
}: {
  titleText?: ReadonlySignal<string>;
  bodyText?: ReadonlySignal<string>;
  level?: 'warning' | 'info';
  onDismiss?: () => void;
}) {
  const palette = SYSTEM_NOTICE_PALETTE[level];
  const xrButton = useXRButton();
  return (
    <Container
      flexDirection="row"
      gap={16}
      alignItems="center"
      justifyContent="space-between"
      padding={16}
      marginBottom={8}
      borderRadius={12}
      borderWidth={2}
      borderColor={palette.border}
      backgroundColor={palette.background}
    >
      <Container flexDirection="column" gap={6} flexGrow={1}>
        <Text fontSize={34} fontWeight="bold" color={palette.title}>
          {titleText}
        </Text>
        {/* whiteSpace="pre-line" is required, not cosmetic: uikit's default
            normalization collapses every run of whitespace -- newlines
            included -- into a single space, which would run the summary, each
            unmet requirement, and its fix together on one unreadable line. */}
        <Text fontSize={28} color="rgba(240, 240, 240, 1)" whiteSpace="pre-line">
          {bodyText}
        </Text>
      </Container>
      <Button
        {...xrButton('system-notice-dismiss', onDismiss ?? (() => {}))}
        variant="default"
        width={80}
        height={64}
        borderRadius={16}
        backgroundColor="rgba(220, 220, 220, 0.9)"
        hover={{ backgroundColor: 'rgba(100, 150, 255, 1)', borderColor: 'white', borderWidth: 2 }}
      >
        <Text fontSize={32} color="black" fontWeight="bold">
          X
        </Text>
      </Button>
    </Container>
  );
}

// Reusable objects for face-camera rotation (avoid allocations in render loop)
const cameraPositionHelper = new Vector3();
const uiPositionHelper = new Vector3();

// Reusable objects for the head-relative yaw math shared by worldPositionFromHeadOffset and
// handleApply below (avoid per-frame allocations while trackHeadset or a drag is active).
const headYawForwardHelper = new Vector3();
const headYawQuatHelper = new Quaternion();
const headYawHorizontalHelper = new Vector3();
const headOffsetWorldHelper = new Vector3();

// Handle hover colors (module-level to avoid per-render allocations)
const HANDLE_COLOR_DEFAULT = new Color('#666666');
const HANDLE_COLOR_HOVER = new Color('#aaaaaa');

function RecordingButton({
  id,
  label,
  onClick,
  disabled = false,
  active = false,
}: {
  id: string;
  label: string;
  onClick: () => void;
  disabled?: boolean;
  active?: boolean;
}) {
  const xrButton = useXRButton();
  return (
    <Button
      {...xrButton(id, onClick)}
      variant="default"
      width={140}
      height={72}
      borderRadius={20}
      disabled={disabled}
      backgroundColor={active ? 'rgba(220, 60, 60, 0.9)' : 'rgba(220, 220, 220, 0.9)'}
      hover={{ backgroundColor: 'rgba(100, 150, 255, 1)', borderColor: 'white', borderWidth: 2 }}
    >
      <Text fontSize={30} color="black" fontWeight="medium">
        {label}
      </Text>
    </Button>
  );
}

export default function CloudXR3DUI({
  onStartTeleop,
  onDisconnect,
  onResetTeleop,
  serverAddress = '127.0.0.1',
  sessionStatus = 'Disconnected',
  playLabel = 'Play',
  playInProgress = false,
  countdownSeconds,
  onCountdownIncrease,
  onCountdownDecrease,
  countdownDisabled = false,
  position = [1.8, 1.75, -1.3],
  controlPanelPosition = 'center',
  controlPanelLayout = { distance: 1.8, height: 1.85, angleDegrees: 70 },
  trackHeadset = false,
  rotation = [0, 0, 0], // Note: Y rotation is controlled by face-camera logic
  renderFpsText,
  poseSendFpsText,
  streamingFpsText,
  poseToRenderText,
  sessionQuality,
  streamTestText,
  streamTestColor,
  panelHiddenAtStart = false,
  isXRMode = false,
  showRecordingControls = false,
  systemNoticeTitleText,
  systemNoticeBodyText,
  systemNoticeVisible = false,
  systemNoticeLevel = 'warning',
  onDismissSystemNotice,
}: CloudXRUIProps) {
  const recorder = useRecorder();
  const MINIMIZE_ON_PLAY_KEY = 'cxr.isaac.minimizeOnPlay';

  const groupRef = useRef<Group>(null);
  const handleRef = useRef<Mesh>(null);
  const xrButton = useXRButton();
  // useState(initializer): React calls the fn once on mount to get the initial value; it returns [value, setter]. Setter used in onClick below.
  const [minimizeOnPlay, setMinimizeOnPlay] = useState(() => {
    try {
      const saved = localStorage.getItem(MINIMIZE_ON_PLAY_KEY);
      return saved === 'true';
    } catch {
      return false;
    }
  });

  /** Control panel hidden: small Show control (see settings to hide control panel on XR enter). */
  const [panelHidden, setPanelHidden] = useState(false);
  const prevXRMode = useRef(false);
  // Set on the same isXRMode false->true transition below, consumed (and cleared) by the
  // useFrame block further down - see resetPanelRelativeToHead's doc comment for why the
  // placement itself can't happen here, in a plain effect.
  const needsInitialPlacement = useRef(false);

  useEffect(() => {
    if (isXRMode && !prevXRMode.current) {
      setPanelHidden(panelHiddenAtStart);
      needsInitialPlacement.current = true;
    }
    prevXRMode.current = isXRMode;
  }, [isXRMode, panelHiddenAtStart]);

  // Only host-observable signal for the in-headset panel's visibility (a world-anchored WebXR
  // scene object, not a DOM element) - lets a host-side watchdog/test tell "panel hidden" apart
  // from "panel missing" instead of both looking like silence.
  useEffect(() => {
    console.debug(`[CloudXRUI] panel visibility: ${panelHidden ? 'hidden' : 'visible'}`);
  }, [panelHidden]);

  // Keep localStorage in sync when the user toggles the option.
  useEffect(() => {
    try {
      localStorage.setItem(MINIMIZE_ON_PLAY_KEY, String(minimizeOnPlay));
    } catch (_) {}
  }, [minimizeOnPlay]);

  useEffect(() => {
    if (groupRef.current) {
      groupRef.current.position.set(position[0], position[1], position[2]);
    }
  }, [position[0], position[1], position[2]]);

  const { camera } = useThree();

  /**
   * The head-relative offset applied every frame while trackHeadset is on: X/Z rotate with the
   * head's yaw (see worldPositionFromHeadOffset), Y is an absolute world height, not relative to
   * the head - same reasoning as worldPositionFromHeadOffset below. Starts at the
   * config-derived default and is overwritten by a completed drag (see handleApply) - so once
   * the operator repositions the panel, tracking continues from *that* spot for the rest of the
   * session instead of snapping back to the default.
   */
  const headOffsetRef = useRef(new Vector3());
  useEffect(() => {
    const [localX, panelHeight, localZ] = getControlPanelPositionVector(
      controlPanelPosition,
      controlPanelLayout
    );
    headOffsetRef.current.set(localX, panelHeight, localZ);
  }, [controlPanelPosition, controlPanelLayout]);

  /**
   * Projects *offset* (see headOffsetRef's doc comment) into world space relative to *cam*.
   *
   * *cam*'s yaw is used on the horizontal (XZ) plane only, matching the face-camera effect below
   * - using the raw camera quaternion (which includes pitch) would place the panel above or
   * below eye level, or tilted, whenever the operator's head isn't level.
   *
   * IMPORTANT: only call this with a camera whose transform is known-current for this frame
   * (i.e. from useFrame's `state.camera`, or - as in the keydown handler below - useThree()'s
   * camera when called well after session start, never from a plain mount-time effect). A WebXR
   * session can only produce a frame callback once it has a real tracked pose, so useFrame is the
   * only place that's guaranteed fresh; a plain useEffect keyed on isXRMode can fire before the
   * first tracked frame lands, reading a stale/default transform instead.
   */
  /** Writes into the shared headOffsetWorldHelper - callers must finish using the returned
   * reference before the next call (true for every caller here: all synchronous, single-frame
   * use). */
  const worldPositionFromHeadOffset = useCallback((cam: Camera, offset: Vector3): Vector3 => {
    const forward = headYawForwardHelper.copy(WORLD_FORWARD).applyQuaternion(cam.quaternion);
    forward.y = 0;
    forward.normalize();
    const yawQuat = headYawQuatHelper.setFromUnitVectors(WORLD_FORWARD, forward);
    const horizontal = headYawHorizontalHelper.set(offset.x, 0, offset.z).applyQuaternion(yawQuat);
    return headOffsetWorldHelper.set(
      cam.position.x + horizontal.x,
      offset.y,
      cam.position.z + horizontal.z
    );
  }, []);

  /**
   * The one place *offset* actually gets applied to groupRef.position from a head-relative
   * offset (a discrete event - reset, or a drag's final release - never the continuous
   * per-frame tracking in useFrame below, which logs on its own, throttled by movement rather
   * than every frame - see lastLoggedWorldRef). Logs all three poses in a single line so a test
   * (or an operator diagnosing tracking drift) can confirm the panel actually lands near the
   * headset instead of at a fixed world coordinate - see the console lines, not the scene graph,
   * since there's no other way to observe a Three.js object's world position from outside the
   * page.
   */
  const logPanelPose = useCallback((cam: Camera, offset: Vector3, target: Vector3) => {
    console.debug(
      `[CloudXRUI] headset=(${cam.position.x.toFixed(2)}, ${cam.position.y.toFixed(2)}, ${cam.position.z.toFixed(2)}) ` +
        `relative=(${offset.x.toFixed(2)}, ${offset.y.toFixed(2)}, ${offset.z.toFixed(2)}) ` +
        `world=(${target.x.toFixed(2)}, ${target.y.toFixed(2)}, ${target.z.toFixed(2)})`
    );
  }, []);

  /** Last world position a pose log line was emitted for - shared between applyPanelPosition's
   * discrete-event logging and the continuous-tracking logging in useFrame below, so the two
   * never double-log the same position. */
  const lastLoggedWorldRef = useRef<Vector3 | null>(null);

  const applyPanelPosition = useCallback(
    (cam: Camera, offset: Vector3): Vector3 => {
      const target = worldPositionFromHeadOffset(cam, offset);
      if (groupRef.current) {
        groupRef.current.position.copy(target);
      }
      logPanelPose(cam, offset, target);
      lastLoggedWorldRef.current = target.clone();
      return target;
    },
    [worldPositionFromHeadOffset, logPanelPose]
  );

  /**
   * Resets headOffsetRef back to the config-derived default (discarding any drag-derived
   * offset - a reset should mean "back to the configured position", not "keep whatever I last
   * dragged to"), and repositions it via worldPositionFromHeadOffset.
   *
   * *unhide* defaults to true: an explicit reset (R key, or the host's CDP
   * _cdp_send_reset_panel_key) is the fix for a panel hidden or dragged out of reach (see
   * CloudXR2DUI's panelHiddenAtStart docs) - the drag handle needed to recover it can itself be
   * unreachable - so the operator or host can always bring it back regardless of where it ended
   * up. The one caller that passes false is the initial-placement path in the useFrame block
   * below: it must still compute the config-derived position (so a later un-hide lands
   * correctly), but it must not override panelHiddenAtStart on the very frame that just set it.
   */
  const resetPanelRelativeToHead = useCallback(
    (cam: Camera, unhide: boolean = true) => {
      if (!groupRef.current) {
        return;
      }
      const [localX, panelHeight, localZ] = getControlPanelPositionVector(
        controlPanelPosition,
        controlPanelLayout
      );
      headOffsetRef.current.set(localX, panelHeight, localZ);
      applyPanelPosition(cam, headOffsetRef.current);
      if (unhide) {
        setPanelHidden(false);
      }
    },
    [controlPanelPosition, controlPanelLayout, applyPanelPosition]
  );

  /**
   * Handle's own apply, run at priority -1 (before this component's face-camera/tracking
   * useFrame below, which runs at the default priority 0) every frame a drag is active. Always
   * copies position (matching the original applyPositionSkipRotation it replaces - quaternion is
   * deliberately never copied, see below). While trackHeadset is on, also converts the dragged
   * world position back into a head-relative offset on every update, not just the last one: the
   * continuous tracking in useFrame recomputes world position from headOffsetRef using this same
   * frame's camera pose, so once this runs (first, priority -1) it reproduces the exact position
   * just set here rather than fighting it - no separate "drag in progress" flag needed to make
   * the two agree. Logging only happens on state.last, so an active multi-frame drag doesn't
   * spam the console on every intermediate update.
   *
   * Quaternion is skipped because of a @pmndrs/handle defaultApply quirk: defaultApply copies
   * state.current.quaternion to the target on every drag frame AND on drag release. With
   * rotate={false}, state.current.quaternion is always the drag-start quaternion, so it resets
   * face-camera rotation on every frame and wipes it entirely on drag release. Scale is
   * intentionally omitted too: scale={false} keeps it constant, so copying it would be a no-op.
   * If scale is ever enabled on this Handle, add it back here.
   */
  const handleApply = useCallback(
    (state: HandleState<unknown>, target: Object3D) => {
      target.position.copy(state.current.position);
      if (!trackHeadset) {
        return;
      }
      // Shares worldPositionFromHeadOffset's scratch objects - safe because both this and that
      // function fully overwrite them (via .copy()/.set()) before reading, every call, so
      // there's never a stale value left over from the other.
      const forward = headYawForwardHelper.copy(WORLD_FORWARD).applyQuaternion(camera.quaternion);
      forward.y = 0;
      forward.normalize();
      const yawQuat = headYawQuatHelper.setFromUnitVectors(WORLD_FORWARD, forward);
      const worldDelta = headYawHorizontalHelper
        .set(target.position.x - camera.position.x, 0, target.position.z - camera.position.z)
        .applyQuaternion(yawQuat.invert());
      headOffsetRef.current.set(worldDelta.x, target.position.y, worldDelta.z);
      if (state.last) {
        // Re-applies the same position target already holds (from state.current.position above)
        // - this call exists for its single log line, going through the same applyPanelPosition
        // every other offset change does, not because the position itself needs recomputing.
        applyPanelPosition(camera, headOffsetRef.current);
      }
    },
    [trackHeadset, camera, applyPanelPosition]
  );

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key.toLowerCase() !== RESET_PANEL_KEY) {
        return;
      }
      // Unlike a function key, a plain letter can be typed into any of the 2D settings form's
      // text inputs - don't reset the panel out from under someone typing a server IP.
      const target = event.target as HTMLElement | null;
      if (
        target?.tagName === 'INPUT' ||
        target?.tagName === 'TEXTAREA' ||
        target?.isContentEditable
      ) {
        return;
      }
      resetPanelRelativeToHead(camera);
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [camera, resetPanelRelativeToHead]);

  const isCompact = minimizeOnPlay && playInProgress;
  const isMinimizedLayout = isCompact || panelHidden;
  const handleWidth = panelHidden ? 0.12 : isCompact ? 0.28 : 1.0;
  const handleY = panelHidden ? -0.065 : isCompact ? -0.15 : -0.42;
  // The notice only floats when the panel is collapsed; the full panel renders
  // it inline. Clearing the top edge of a 2.33-unit-tall panel would put the
  // notice about a metre above the operator's head, out of view entirely --
  // floating is what keeps it visible when the panel is small, not a layout
  // that makes sense at full size.
  const noticeFloats = panelHidden || isCompact;
  const noticeY = panelHidden ? 0.42 : 0.6;

  // Face-camera rotation: smoothly rotate UI to face the user (Y-axis only)
  useFrame((state, dt) => {
    if (groupRef.current == null) {
      return;
    }
    // First real XR frame since session start (see resetPanelRelativeToHead's doc comment for
    // why it must happen here, not in the isXRMode effect that set the flag). unhide=false:
    // the isXRMode effect above just set panelHidden from panelHiddenAtStart - preserve that
    // choice instead of forcing the panel visible on its very first frame.
    if (needsInitialPlacement.current) {
      // WebXR explicitly permits getViewerPose() to return null on the first frame(s) while
      // tracking initializes (https://www.w3.org/TR/webxr/#dom-xrframe-getviewerpose) - three.js
      // still invokes this callback regardless, leaving state.camera at its stale/default
      // transform. Placing the panel from that and clearing the flag would be permanent (nothing
      // else re-triggers initial placement), so wait for a frame with a real pose instead.
      const xrFrame = state.gl.xr.getFrame() as XRFrame | null;
      const referenceSpace = state.gl.xr.getReferenceSpace();
      const viewerPose = xrFrame && referenceSpace ? xrFrame.getViewerPose(referenceSpace) : null;
      if (viewerPose) {
        resetPanelRelativeToHead(state.camera, false);
        needsInitialPlacement.current = false;
      }
    } else if (trackHeadset) {
      // Continuous version of the same reset: every frame instead of once, using whatever
      // headOffsetRef currently holds (the config default, or wherever the operator is currently
      // dragging to - see handleApply, which updates it every frame too, not just on release),
      // and without the un-hide side effect (calling resetPanelRelativeToHead here would force
      // the panel visible every frame, defeating the hide-panel button). Runs even mid-drag:
      // Handle's own apply already ran this frame (priority -1, before this useFrame) and updated
      // headOffsetRef from the same camera pose this frame will use, so recomputing world
      // position from it here reproduces what Handle just set instead of fighting it.
      const target = worldPositionFromHeadOffset(state.camera, headOffsetRef.current);
      groupRef.current.position.copy(target);
      // Logging every frame would spam the console, but logging nothing makes continuous
      // tracking unobservable from outside the page (no other way to read a Three.js object's
      // world position - see applyPanelPosition's doc comment). Log only once the panel has
      // actually moved a visible amount since the last log line, discrete or continuous.
      if (
        !lastLoggedWorldRef.current ||
        lastLoggedWorldRef.current.distanceTo(target) > TRACKING_LOG_EPSILON_M
      ) {
        logPanelPose(state.camera, headOffsetRef.current, target);
        lastLoggedWorldRef.current = target.clone();
      }
    }
    state.camera.getWorldPosition(cameraPositionHelper);
    groupRef.current.getWorldPosition(uiPositionHelper);

    // Project onto the horizontal plane (XZ) to get a pure yaw angle.
    // Using atan2 avoids the 3D-quaternion→Euler extraction that can give wrong
    // yaw when the camera has significant height offset relative to the panel.
    const dx = cameraPositionHelper.x - uiPositionHelper.x;
    const dz = cameraPositionHelper.z - uiPositionHelper.z;
    let targetY = Math.atan2(dx, dz);

    // Wrap the angular difference to [-π, π] so damp() always takes the
    // shortest path.  Without this, when the target crosses the ±π boundary
    // (camera near the side/behind the panel), damp interpolates the long way
    // around and the panel snaps to face away from the user.
    const currentY = groupRef.current.rotation.y;
    let diff = targetY - currentY;
    diff = diff - Math.round(diff / (2 * Math.PI)) * (2 * Math.PI);
    targetY = currentY + diff;

    groupRef.current.rotation.y = damp(currentY, targetY, FACE_CAMERA_DAMPING, dt);
  });

  return (
    <HandleTarget>
      <group
        ref={groupRef}
        position={position}
        rotation={rotation}
        pointerEventsType={{ deny: 'grab' }}
      >
        {/* Drag Handle Bar - grab to reposition the panel. Works the same whether trackHeadset is
            on or off; while it's on, handleApply also updates headOffsetRef on release so
            continuous tracking (useFrame below) keeps following from the new spot instead of
            snapping back to wherever it was configured. */}
        <Handle
          handleRef={handleRef}
          targetRef={groupRef}
          scale={false}
          multitouch={false}
          rotate={false}
          apply={handleApply}
        >
          <mesh
            ref={handleRef}
            position={[0, handleY, 0.01]}
            onPointerEnter={() => {
              const mat = handleRef.current?.material as MeshStandardMaterial | undefined;
              if (mat) {
                mat.color.copy(HANDLE_COLOR_HOVER);
                mat.opacity = panelHidden ? 0.55 : 0.9;
              }
            }}
            onPointerLeave={() => {
              const mat = handleRef.current?.material as MeshStandardMaterial | undefined;
              if (mat) {
                mat.color.copy(HANDLE_COLOR_DEFAULT);
                mat.opacity = panelHidden ? 0.35 : 0.6;
              }
            }}
          >
            <boxGeometry args={[handleWidth, panelHidden ? 0.035 : 0.05, 0.02]} />
            <meshStandardMaterial
              color="#666666"
              transparent
              opacity={panelHidden ? 0.35 : 0.6}
              roughness={0.5}
            />
          </mesh>
        </Handle>

        <Container
          pixelSize={0.001}
          width={panelHidden ? 128 : isCompact ? 520 : 2000}
          height={panelHidden ? 128 : isCompact ? 320 : 1400}
          alignItems="center"
          justifyContent="center"
          pointerEvents="auto"
          padding={panelHidden ? 0 : isCompact ? 24 : 40}
          sizeX={panelHidden ? 0.2 : isCompact ? 0.87 : 3.33}
          sizeY={panelHidden ? 0.2 : isCompact ? 0.53 : 2.33}
          flexDirection="column"
        >
          {panelHidden ? (
            <Button
              {...xrButton('show-panel', () => setPanelHidden(false))}
              variant="default"
              width={112}
              height={112}
              borderRadius={56}
              backgroundColor="rgba(90, 130, 210, 0.42)"
              hover={{
                backgroundColor: 'rgba(90, 130, 210, 0.72)',
                borderColor: 'rgba(255, 255, 255, 0.6)',
                borderWidth: 2,
              }}
            >
              <Text fontSize={26} color="rgba(255, 255, 255, 0.95)" fontWeight="bold">
                Show
              </Text>
            </Button>
          ) : isCompact ? (
            <Container
              width="100%"
              flexDirection="column"
              gap={16}
              alignItems="center"
              justifyContent="center"
              backgroundColor="rgba(40, 40, 40, 0.85)"
              borderRadius={20}
              padding={24}
            >
              <Button
                {...xrButton('start-min', onStartTeleop)}
                variant="default"
                width={400}
                height={80}
                borderRadius={24}
                backgroundColor="rgba(220, 220, 220, 0.9)"
                hover={{
                  backgroundColor: 'rgba(100, 150, 255, 1)',
                  borderColor: 'white',
                  borderWidth: 2,
                }}
                disabled={playInProgress}
              >
                <Container flexDirection="row" alignItems="center" gap={8}>
                  {playLabel === 'Play' && <Image src={playCircleSvg} width={40} height={40} />}
                  <Text fontSize={36} color="black" fontWeight="medium">
                    {playLabel}
                  </Text>
                </Container>
              </Button>
              <Container
                flexDirection="row"
                gap={14}
                alignItems="center"
                justifyContent="center"
                width="100%"
              >
                <Button
                  {...xrButton('reset-min', onResetTeleop)}
                  variant="default"
                  width={292}
                  height={80}
                  borderRadius={24}
                  backgroundColor="rgba(220, 220, 220, 0.9)"
                  hover={{
                    backgroundColor: 'rgba(100, 150, 255, 1)',
                    borderColor: 'white',
                    borderWidth: 2,
                  }}
                >
                  <Container flexDirection="row" alignItems="center" gap={8}>
                    <Image src={arrowUturnLeftSvg} width={40} height={40} />
                    <Text fontSize={36} color="black" fontWeight="medium">
                      Reset
                    </Text>
                  </Container>
                </Button>
                <Button
                  {...xrButton('hide-panel-compact', () => setPanelHidden(true))}
                  variant="default"
                  width={94}
                  height={80}
                  borderRadius={20}
                  backgroundColor="rgba(70, 75, 90, 0.55)"
                  hover={{
                    backgroundColor: 'rgba(90, 95, 115, 0.85)',
                    borderColor: 'rgba(255, 255, 255, 0.5)',
                    borderWidth: 2,
                  }}
                >
                  <Text fontSize={26} color="rgba(255, 255, 255, 0.92)" fontWeight="medium">
                    Hide
                  </Text>
                </Button>
              </Container>
            </Container>
          ) : (
            <Container
              width={1900}
              height={980}
              backgroundColor="rgba(40, 40, 40, 0.85)"
              borderRadius={20}
              padding={50}
              paddingLeft={50}
              paddingRight={50}
              alignItems="center"
              justifyContent="center"
              flexDirection="row"
              gap={36}
            >
              {/* Left Column - Performance Metrics */}
              <Container
                width={520}
                flexDirection="column"
                gap={24}
                alignItems="center"
                justifyContent="center"
              >
                <Container
                  width="100%"
                  flexDirection="column"
                  gap={20}
                  alignItems="center"
                  justifyContent="center"
                  backgroundColor="rgba(20, 20, 20, 0.6)"
                  borderRadius={20}
                  padding={36}
                >
                  <Text
                    fontSize={52}
                    fontWeight="bold"
                    color="white"
                    textAlign="center"
                    marginBottom={4}
                  >
                    Performance
                  </Text>

                  <Container
                    width={METRIC_SLOT_WIDTH}
                    height={METRIC_SLOT_HEIGHT}
                    alignItems="center"
                    justifyContent="center"
                  >
                    <PerformanceCanvasImage
                      width={METRIC_SLOT_WIDTH}
                      height={METRIC_SLOT_HEIGHT}
                      renderFpsText={renderFpsText}
                      poseSendFpsText={poseSendFpsText}
                      streamingFpsText={streamingFpsText}
                      poseToRenderText={poseToRenderText}
                      sessionQuality={sessionQuality}
                    />
                  </Container>
                </Container>

                <Container
                  flexDirection="row"
                  alignItems="center"
                  justifyContent="center"
                  gap={14}
                  marginTop={20}
                  cursor="pointer"
                  {...xrButton('minimize', () => setMinimizeOnPlay(v => !v))}
                >
                  <Container
                    width={48}
                    height={48}
                    borderRadius={8}
                    borderWidth={2}
                    borderColor="rgba(200, 200, 200, 1)"
                    backgroundColor="rgba(60, 60, 60, 0.8)"
                    alignItems="center"
                    justifyContent="center"
                    padding={8}
                  >
                    {minimizeOnPlay && (
                      <Container
                        width="100%"
                        height="100%"
                        borderRadius={4}
                        backgroundColor="rgba(100, 255, 100, 0.95)"
                      />
                    )}
                  </Container>
                  <Text fontSize={30} color="rgba(220, 220, 220, 1)">
                    Minimize on play (compact controls)
                  </Text>
                </Container>

                {showRecordingControls && (
                  <Container
                    width="100%"
                    flexDirection="column"
                    gap={12}
                    alignItems="center"
                    marginTop={16}
                  >
                    <Text fontSize={36} fontWeight="bold" color="rgba(220, 220, 220, 1)">
                      {recorder.mode === 'recording'
                        ? `REC ${recorder.recordedFrameCount} frames`
                        : recorder.mode === 'replaying'
                          ? recorder.replayNeedsCalibration
                            ? 'Replay paused'
                            : 'Replaying'
                          : 'Recording'}
                    </Text>
                    {recorder.replayNeedsCalibration && recorder.mode === 'replaying' && (
                      <Text fontSize={24} color="white" textAlign="center">
                        Return your headset to its recording-start position and heading, then
                        calibrate.
                      </Text>
                    )}
                    {recorder.mode === 'idle' && recorder.recordingInterrupted && (
                      <Text fontSize={24} color="white" textAlign="center">
                        Recording stopped because the tracking origin changed. Save it or start a
                        new recording.
                      </Text>
                    )}
                    {recorder.mode === 'idle' && !recorder.recordingInterrupted && (
                      <Text fontSize={24} color="white" textAlign="center">
                        Note your headset position and heading when starting a recording. Use the
                        same pose to calibrate replay in a new session.
                      </Text>
                    )}
                    <Container flexDirection="row" gap={12} justifyContent="center">
                      {recorder.mode === 'replaying' && recorder.replayNeedsCalibration && (
                        <RecordingButton
                          id="calibrate-replay"
                          label="Calibrate"
                          onClick={recorder.calibrateReplay}
                        />
                      )}
                      {recorder.mode !== 'replaying' && (
                        <RecordingButton
                          id="record-input"
                          label={recorder.mode === 'recording' ? 'Stop' : 'Rec'}
                          onClick={
                            recorder.mode === 'recording'
                              ? recorder.stopRecord
                              : recorder.startRecord
                          }
                          active={recorder.mode === 'recording'}
                        />
                      )}
                      {recorder.mode !== 'recording' && (
                        <RecordingButton
                          id="replay-input"
                          label={recorder.mode === 'replaying' ? 'Stop' : 'Play'}
                          onClick={
                            recorder.mode === 'replaying'
                              ? recorder.stopReplay
                              : recorder.startReplay
                          }
                          disabled={recorder.mode === 'idle' && !recorder.savedRecording}
                        />
                      )}
                      {recorder.mode === 'idle' && recorder.savedRecording && (
                        <RecordingButton
                          id="save-input"
                          label="Save"
                          onClick={recorder.onSaveRecording}
                        />
                      )}
                    </Container>
                  </Container>
                )}
              </Container>

              {/* Right Column - Controls */}
              <Container
                flexGrow={1}
                flexDirection="column"
                gap={20}
                alignItems="center"
                justifyContent="center"
              >
                {/* Title */}
                <Text fontSize={72} fontWeight="bold" color="white" textAlign="center">
                  Controls
                </Text>

                {/* Workstation advisory, inline while the panel is full size.
                    See the floating copy below for the collapsed states. */}
                {systemNoticeVisible && !noticeFloats && (
                  <SystemNoticeBanner
                    titleText={systemNoticeTitleText}
                    bodyText={systemNoticeBodyText}
                    level={systemNoticeLevel}
                    onDismiss={onDismissSystemNotice}
                  />
                )}

                {/* Server Info */}
                <Container
                  flexDirection="column"
                  gap={8}
                  alignItems="center"
                  marginTop={4}
                  marginBottom={4}
                >
                  <Text fontSize={38} color="rgba(200, 200, 200, 1)" textAlign="center">
                    Server: {serverAddress}
                  </Text>
                  <Text fontSize={38} color="rgba(200, 200, 200, 1)" textAlign="center">
                    Status: {sessionStatus}
                  </Text>
                  {/* Network test status. Signals drive the text and traffic-light color;
                      both are empty when the test is off, which is the default. */}
                  <Text fontSize={34} color={streamTestColor} textAlign="center">
                    {streamTestText}
                  </Text>
                </Container>

                {/* Countdown Config Row */}
                <Container
                  flexDirection="row"
                  gap={16}
                  alignItems="center"
                  justifyContent="center"
                  marginTop={12}
                >
                  <Text fontSize={36} color="white">
                    Countdown
                  </Text>
                  <Button
                    {...xrButton('countdown-dec', onCountdownDecrease)}
                    variant="default"
                    width={90}
                    height={90}
                    borderRadius={45}
                    backgroundColor="rgba(220, 220, 220, 0.9)"
                    disabled={countdownDisabled}
                  >
                    <Text fontSize={44} color="black" fontWeight="bold">
                      -
                    </Text>
                  </Button>
                  <Container
                    width={140}
                    height={90}
                    alignItems="center"
                    justifyContent="center"
                    backgroundColor="rgba(255,255,255,0.9)"
                    borderRadius={12}
                  >
                    <Text fontSize={48} color="black" fontWeight="bold">
                      {countdownSeconds}s
                    </Text>
                  </Container>
                  <Button
                    {...xrButton('countdown-inc', onCountdownIncrease)}
                    variant="default"
                    width={90}
                    height={90}
                    borderRadius={45}
                    backgroundColor="rgba(220, 220, 220, 0.9)"
                    disabled={countdownDisabled}
                  >
                    <Text fontSize={44} color="black" fontWeight="bold">
                      +
                    </Text>
                  </Button>
                </Container>

                {/* Button Grid */}
                <Container
                  flexDirection="column"
                  gap={20}
                  alignItems="center"
                  justifyContent="center"
                  width="100%"
                  marginTop={16}
                >
                  {/* Start/reset row*/}
                  <Container flexDirection="row" gap={24} justifyContent="center">
                    <Button
                      {...xrButton('start', onStartTeleop)}
                      variant="default"
                      width={420}
                      height={100}
                      borderRadius={32}
                      backgroundColor="rgba(220, 220, 220, 0.9)"
                      hover={{
                        backgroundColor: 'rgba(100, 150, 255, 1)',
                        borderColor: 'white',
                        borderWidth: 2,
                      }}
                      disabled={playInProgress}
                    >
                      <Container flexDirection="row" alignItems="center" gap={10}>
                        {playLabel === 'Play' && (
                          <Image src={playCircleSvg} width={50} height={50} />
                        )}
                        <Text fontSize={42} color="black" fontWeight="medium">
                          {playLabel}
                        </Text>
                      </Container>
                    </Button>

                    <Button
                      {...xrButton('reset', onResetTeleop)}
                      variant="default"
                      width={420}
                      height={100}
                      borderRadius={32}
                      backgroundColor="rgba(220, 220, 220, 0.9)"
                      hover={{
                        backgroundColor: 'rgba(100, 150, 255, 1)',
                        borderColor: 'white',
                        borderWidth: 2,
                      }}
                    >
                      <Container flexDirection="row" alignItems="center" gap={10}>
                        <Image src={arrowUturnLeftSvg} width={50} height={50} />
                        <Text fontSize={42} color="black" fontWeight="medium">
                          Reset
                        </Text>
                      </Container>
                    </Button>
                  </Container>

                  {/* Bottom Row */}
                  <Container
                    flexDirection="row"
                    justifyContent="center"
                    alignItems="center"
                    gap={18}
                  >
                    <Button
                      {...xrButton('disconnect', onDisconnect)}
                      variant="destructive"
                      width={320}
                      height={90}
                      borderRadius={28}
                      backgroundColor="rgba(255, 150, 150, 0.9)"
                      hover={{
                        backgroundColor: 'rgba(255, 50, 50, 1)',
                        borderColor: 'white',
                        borderWidth: 2,
                      }}
                    >
                      <Container flexDirection="row" alignItems="center" gap={10}>
                        <Image src={arrowLeftStartOnRectangleSvg} width={50} height={50} />
                        <Text fontSize={38} color="black" fontWeight="medium">
                          Disconnect
                        </Text>
                      </Container>
                    </Button>
                    <Button
                      {...xrButton('hide-panel-full', () => setPanelHidden(true))}
                      variant="default"
                      width={100}
                      height={90}
                      borderRadius={22}
                      backgroundColor="rgba(70, 75, 90, 0.55)"
                      hover={{
                        backgroundColor: 'rgba(90, 95, 115, 0.88)',
                        borderColor: 'rgba(255, 255, 255, 0.5)',
                        borderWidth: 2,
                      }}
                    >
                      <Text fontSize={28} color="rgba(255, 255, 255, 0.92)" fontWeight="medium">
                        Hide
                      </Text>
                    </Button>
                  </Container>
                </Container>
              </Container>
            </Container>
          )}
        </Container>

        {/* Workstation advisory from the host, in its floating form.
            The panel collapses to compact when "minimize on play" fires, and to
            a bare Show button when hidden -- either would swallow the warning at
            exactly the moment the operator is working. Rendering it as a sibling
            of the panel keeps it visible in those states, while it still rides
            the panel's drag position and face-camera rotation so it appears
            where the operator already put their UI.

            The full panel renders the notice inline instead: it is 2.33 units
            tall, so a floating notice clearing its top edge would sit about a
            metre above the operator's head. */}
        {systemNoticeVisible && noticeFloats && (
          <group position={[0, noticeY, 0]}>
            <Container
              pixelSize={0.001}
              width={1400}
              height={340}
              alignItems="center"
              justifyContent="center"
              pointerEvents="auto"
              sizeX={2.33}
              sizeY={0.57}
              flexDirection="column"
            >
              <SystemNoticeBanner
                titleText={systemNoticeTitleText}
                bodyText={systemNoticeBodyText}
                level={systemNoticeLevel}
                onDismiss={onDismissSystemNotice}
              />
            </Container>
          </group>
        )}
      </group>
    </HandleTarget>
  );
}
