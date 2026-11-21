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

import { useFrame } from '@react-three/fiber';
import { useEffect, useRef } from 'react';
import { BufferAttribute, BufferGeometry, Color, Points, PointsMaterial } from 'three';

import { useRecorder } from './RecorderContext';
import type { SerializedPose } from './xrInputRecorder';

const TRACE_LENGTH = 500;

class TraceBuffer {
  private readonly values = new Float32Array(TRACE_LENGTH * 3);
  private writeIndex = 0;
  private count = 0;

  push({ px, py, pz }: Exclude<SerializedPose, null>): void {
    const index = this.writeIndex * 3;
    this.values[index] = px;
    this.values[index + 1] = py;
    this.values[index + 2] = pz;
    this.writeIndex = (this.writeIndex + 1) % TRACE_LENGTH;
    this.count = Math.min(this.count + 1, TRACE_LENGTH);
  }

  copyTo(positions: Float32Array, colors: Float32Array, color: Color): number {
    const start = this.count < TRACE_LENGTH ? 0 : this.writeIndex;
    for (let index = 0; index < this.count; index++) {
      const source = ((start + index) % TRACE_LENGTH) * 3;
      const target = index * 3;
      const brightness = 0.1 + 0.9 * (this.count > 1 ? index / (this.count - 1) : 1);
      positions.set(this.values.subarray(source, source + 3), target);
      colors[target] = color.r * brightness;
      colors[target + 1] = color.g * brightness;
      colors[target + 2] = color.b * brightness;
    }
    return this.count;
  }

  clear(): void {
    this.writeIndex = 0;
    this.count = 0;
  }
}

type TraceChannel = {
  buffer: TraceBuffer;
  points: Points;
  positions: Float32Array;
  colors: Float32Array;
  color: Color;
};

function createChannel(color: string): TraceChannel {
  const positions = new Float32Array(TRACE_LENGTH * 3);
  const colors = new Float32Array(TRACE_LENGTH * 3);
  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new BufferAttribute(positions, 3));
  geometry.setAttribute('color', new BufferAttribute(colors, 3));
  geometry.setDrawRange(0, 0);

  const points = new Points(
    geometry,
    new PointsMaterial({ vertexColors: true, size: 6, sizeAttenuation: false })
  );
  points.visible = false;

  return {
    buffer: new TraceBuffer(),
    points,
    positions,
    colors,
    color: new Color(color),
  };
}

function updateChannel(channel: TraceChannel, pose: SerializedPose | undefined): void {
  if (!pose) return;
  channel.buffer.push(pose);
  const count = channel.buffer.copyTo(channel.positions, channel.colors, channel.color);
  const geometry = channel.points.geometry as BufferGeometry;
  geometry.setDrawRange(0, count);
  (geometry.attributes.position as BufferAttribute).needsUpdate = true;
  (geometry.attributes.color as BufferAttribute).needsUpdate = true;
  channel.points.visible = true;
}

export function TraceVisualization({ showTrace }: { showTrace: boolean }) {
  const { recorder, mode } = useRecorder();
  const channelsRef = useRef<TraceChannel[] | null>(null);
  if (!channelsRef.current) {
    channelsRef.current = [
      createChannel('#4488ff'),
      createChannel('#44ff88'),
      createChannel('#ff4422'),
      createChannel('#ff44cc'),
    ];
  }
  const channels = channelsRef.current;

  useEffect(() => {
    channels.forEach(channel => channel.buffer.clear());
  }, [channels, mode]);

  useEffect(
    () => () => {
      channels.forEach(({ points }) => {
        points.geometry.dispose();
        (points.material as PointsMaterial).dispose();
      });
    },
    [channels]
  );

  useFrame(() => {
    const frame = recorder.currentFrame;
    if (!showTrace || !frame) {
      channels.forEach(channel => {
        channel.points.visible = false;
      });
      return;
    }

    // Transform the whole trail so origin resets also move its earlier samples.
    const alignment = recorder.replaySceneAlignment;
    for (const { points } of channels) {
      points.position.set(alignment?.px ?? 0, alignment?.py ?? 0, alignment?.pz ?? 0);
      points.quaternion.set(
        alignment?.ox ?? 0,
        alignment?.oy ?? 0,
        alignment?.oz ?? 0,
        alignment?.ow ?? 1
      );
    }
    updateChannel(channels[0], frame.poses.leftGrip);
    updateChannel(channels[1], frame.poses.rightGrip);
    updateChannel(channels[2], frame.handJoints.left.wrist);
    updateChannel(channels[3], frame.handJoints.right.wrist);
  });

  return (
    <>
      {channels.map((channel, index) => (
        <primitive key={index} object={channel.points} />
      ))}
    </>
  );
}
