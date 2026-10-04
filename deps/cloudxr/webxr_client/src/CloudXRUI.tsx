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
import { useFrame } from '@react-three/fiber';
import { Handle, HandleState, HandleTarget } from '@react-three/handle';
import { Container, Image, Text } from '@react-three/uikit';
import { Button } from '@react-three/uikit-default';
import React, { useEffect, useRef, useState } from 'react';
import { Color, Group, Mesh, MeshStandardMaterial, Object3D, Vector3 } from 'three';
import { damp } from 'three/src/math/MathUtils.js';

import { PerformanceCanvasImage } from '@helpers/react/PerformanceCanvasImage';
import { useXRButton } from '@helpers/react/useXRButton';

import arrowLeftStartOnRectangleSvg from './icons/arrow-left-start-on-rectangle.svg';
import arrowUturnLeftSvg from './icons/arrow-uturn-left.svg';
import playCircleSvg from './icons/play-circle.svg';
import { useRecorder } from './RecorderContext';

// Face-camera rotation constants
const FACE_CAMERA_DAMPING = 10; // Higher = faster rotation toward camera

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

// Handle hover colors (module-level to avoid per-render allocations)
const HANDLE_COLOR_DEFAULT = new Color('#666666');
const HANDLE_COLOR_HOVER = new Color('#aaaaaa');

// Workaround for @pmndrs/handle defaultApply behavior: defaultApply copies
// state.current.quaternion to the target on every drag frame AND on drag release.
// With rotate={false}, state.current.quaternion is always the drag-start quaternion,
// so it resets our face-camera rotation on every frame (priority -1 runs before
// face-camera priority 0) and wipes it entirely on drag release.
// By providing a custom apply that skips quaternion, face-camera owns rotation fully.
// Scale is intentionally omitted too: scale={false} keeps it constant, so copying
// it would be a no-op. If scale is ever enabled on this Handle, add it back here.
function applyPositionSkipRotation(state: HandleState<unknown>, target: Object3D): void {
  target.position.copy(state.current.position);
}

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

  useEffect(() => {
    if (isXRMode && !prevXRMode.current) {
      setPanelHidden(panelHiddenAtStart);
    }
    prevXRMode.current = isXRMode;
  }, [isXRMode, panelHiddenAtStart]);

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
        {/* Drag Handle Bar - grab to reposition the panel */}
        <Handle
          handleRef={handleRef}
          targetRef={groupRef}
          scale={false}
          multitouch={false}
          rotate={false}
          apply={applyPositionSkipRotation}
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
