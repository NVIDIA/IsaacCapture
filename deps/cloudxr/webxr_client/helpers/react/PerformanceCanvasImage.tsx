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
 * PerformanceCanvasImage - Canvas-backed performance metrics display.
 *
 * Renders Render FPS, Streaming FPS, and Pose-to-Render in a single canvas texture
 * with rounded rectangles per line. Used as a uikit Image for efficient per-frame
 * updates without triggering layout.
 *
 * Why canvas + texture instead of uikit Text?
 * - Updating uikit Text from signals can trigger layout recalculations every frame.
 * - Drawing to a canvas and setting texture.needsUpdate = true updates the image
 *   without affecting the rest of the UI tree.
 *
 * The texture is assigned to the Image via ref (imageRef.current.texture.value), not
 * the `src` prop, because the uikit bridge would stringify a Texture object and
 * cause a 404 if passed as src.
 */

import { ReadonlySignal } from '@preact/signals-react';
import { useFrame } from '@react-three/fiber';
import { Image, type VanillaImage } from '@react-three/uikit';
import React, { useEffect, useRef } from 'react';
import { CanvasTexture } from 'three';

/** Canvas resolution (pixels). High values keep text sharp when the texture is scaled to the display size. */
const CANVAS_WIDTH = 1024;
const CANVAS_HEIGHT = 760;

// Layout constants (canvas space) — compact card style with label + value side-by-side.
// SQ card (200px) + 4 metric cards (110px) + 4 gaps (14px) = 696px, centred in 760px canvas (32px margin each side).
const LAYOUT = {
  fontSize: 52,
  sqCardHeight: 200, // session-quality bars card — must exceed tallest bar (160px) plus 10px padding
  cardHeight: 110, // metric text cards
  cardGap: 14,
  margin: 56,
  paddingLeft: 40,
  radius: 18,
  cardFillStyle: 'rgba(0, 0, 0, 0.5)',
  labelColor: 'rgba(180, 180, 180, 1)',
} as const;

/** RGBA fill colors for each session quality level (index = quality integer 0–4). */
const SESSION_QUALITY_COLORS = [
  'rgba(120, 120, 120, 1)', // NoData — grey
  'rgba(220, 50, 50, 1)', // Unsustainable — red
  'rgba(220, 160, 30, 1)', // Degraded — amber
  'rgba(80, 200, 80, 1)', // Good — green
  'rgba(0, 255, 128, 1)', // Excellent — bright green
] as const;

/** Dim fill used for unlit bars in the session quality indicator. */
const SESSION_QUALITY_BAR_DIM = 'rgba(60, 60, 60, 0.6)';

/** Heights (px, canvas space) of the 4 quality bars, shortest to tallest. */
const SESSION_QUALITY_BAR_HEIGHTS = [50, 80, 115, 160] as const;

const CARD_WIDTH = CANVAS_WIDTH - LAYOUT.margin * 2;

/** Draw a rounded rectangle path; caller must ctx.fill() or ctx.stroke() after. */
function drawRoundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number
): void {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

/** Props for {@link PerformanceCanvasImage}: sizing and per-frame metric/quality signals. */
export interface PerformanceCanvasImageProps {
  /** Display width of the metrics image (uikit units). */
  width?: number;
  /** Display height of the metrics image (uikit units). */
  height?: number;
  /** Signal for render FPS value (e.g. "72.0"). Label "Render FPS: " is drawn here. */
  renderFpsText?: ReadonlySignal<string>;
  /** Signal for pose send FPS value (e.g. "72.0"). Label "Pose Send FPS: " is drawn here. */
  poseSendFpsText?: ReadonlySignal<string>;
  /** Signal for streaming FPS value. Label "Streaming FPS: " is drawn here. */
  streamingFpsText?: ReadonlySignal<string>;
  /** Signal for pose-to-render latency (e.g. "12.3ms"). Label "Pose-to-Render: " is drawn here. */
  poseToRenderText?: ReadonlySignal<string>;
  /** Signal carrying live session quality (0–4); see {@link CloudXR.MetricsName.SessionQuality}. */
  sessionQuality?: ReadonlySignal<number>;
}

/**
 * Renders three performance metric lines on an offscreen canvas, uploads it to a
 * CanvasTexture, and displays it via a uikit Image. Redrawn every frame in useFrame
 * so values stay in sync without React re-renders.
 */
export function PerformanceCanvasImage({
  width = 512,
  height = 512,
  renderFpsText,
  poseSendFpsText,
  streamingFpsText,
  poseToRenderText,
  sessionQuality,
}: PerformanceCanvasImageProps) {
  /** Ref for the uikit Image; we set .texture.value on it to use our CanvasTexture. */
  const imageRef = useRef<VanillaImage | null>(null);
  /** Last image instance that owned our texture; React clears imageRef before effect cleanup. */
  const boundImageRef = useRef<VanillaImage | null>(null);
  /** Offscreen canvas we draw into each frame. */
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  /** Cached 2D context for the canvas (avoids getContext('2d') every frame). */
  const ctxRef = useRef<CanvasRenderingContext2D | null>(null);
  /** Three.js texture wrapping the canvas; needsUpdate = true each frame after drawing. */
  const textureRef = useRef<CanvasTexture | null>(null);
  /** Create the offscreen canvas and CanvasTexture once on mount; dispose on unmount. */
  useEffect(() => {
    const canvas = document.createElement('canvas');
    canvas.width = CANVAS_WIDTH;
    canvas.height = CANVAS_HEIGHT;
    canvasRef.current = canvas;
    ctxRef.current = canvas.getContext('2d');
    const tex = new CanvasTexture(canvas);
    tex.matrixAutoUpdate = false;
    textureRef.current = tex;
    return () => {
      if (boundImageRef.current?.texture.value === tex) {
        boundImageRef.current.texture.value = undefined;
      }
      boundImageRef.current = null;
      tex.dispose();
      textureRef.current = null;
      canvasRef.current = null;
      ctxRef.current = null;
    };
  }, []);

  /** Every frame: keep the image bound, then draw the quality and metric cards. */
  useFrame(() => {
    const canvas = canvasRef.current;
    const texture = textureRef.current;
    const ctx = ctxRef.current;
    if (!canvas || !texture || !ctx) return;
    // UIKit's asynchronous src loader can clear a texture assigned once in an
    // effect. Restore it when that happens, including after remounts.
    if (imageRef.current && imageRef.current.texture.value !== texture) {
      imageRef.current.texture.value = texture;
    }
    if (imageRef.current) boundImageRef.current = imageRef.current;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const {
      fontSize,
      sqCardHeight,
      cardHeight,
      cardGap,
      margin,
      paddingLeft,
      radius,
      cardFillStyle,
      labelColor,
    } = LAYOUT;
    // Each tuple: [label, current signal value (or em-dash fallback), value color].
    const metrics: [string, string, string][] = [
      ['Render FPS', renderFpsText?.value ?? '—', 'rgba(100, 255, 100, 1)'],
      ['Pose Send FPS', poseSendFpsText?.value ?? '—', 'rgba(180, 255, 140, 1)'],
      ['Streaming FPS', streamingFpsText?.value ?? '—', 'rgba(100, 200, 255, 1)'],
      ['Pose-to-Render', poseToRenderText?.value ?? '—', 'rgba(255, 200, 100, 1)'],
    ];
    // Vertically center: 1 SQ card + N metric cards + N gaps between each adjacent pair.
    const totalHeight = sqCardHeight + metrics.length * cardHeight + metrics.length * cardGap;
    let cardY = (canvas.height - totalHeight) / 2;

    ctx.font = `bold ${fontSize}px system-ui, sans-serif`;
    ctx.textBaseline = 'middle';

    // Session Quality card first — appears directly under the "Performance" label.
    // Clamp against COLORS (length 5, indices 0–4) not HEIGHTS (length 4) so Excellent (4) renders correctly.
    const qualityLevel = Math.max(
      0,
      Math.min(SESSION_QUALITY_COLORS.length - 1, Math.round(sessionQuality?.value ?? 0))
    );
    const qualityColor = SESSION_QUALITY_COLORS[qualityLevel];
    ctx.fillStyle = cardFillStyle;
    drawRoundRect(ctx, margin, cardY, CARD_WIDTH, sqCardHeight, radius);
    ctx.fill();

    // Draw 4 vertical bars of increasing height, bottom-aligned within the SQ card.
    const numBars = SESSION_QUALITY_BAR_HEIGHTS.length;
    const barWidth = 130;
    const barGap = 70;
    const barsTotal = numBars * barWidth + (numBars - 1) * barGap;
    const barBaseX = margin + (CARD_WIDTH - barsTotal) / 2;
    const barBottomY = cardY + sqCardHeight - 10;
    for (let i = 0; i < numBars; i++) {
      const barH = SESSION_QUALITY_BAR_HEIGHTS[i];
      const barX = barBaseX + i * (barWidth + barGap);
      // Bars 0..qualityLevel-1 are lit; the rest are dim. For NoData all are dim.
      ctx.fillStyle = qualityLevel > 0 && i < qualityLevel ? qualityColor : SESSION_QUALITY_BAR_DIM;
      drawRoundRect(ctx, barX, barBottomY - barH, barWidth, barH, 8);
      ctx.fill();
    }
    cardY += sqCardHeight + cardGap;

    // Three metric cards below the quality indicator.
    const centerY = cardHeight / 2;
    for (const [label, value, valueColor] of metrics) {
      // Draw the rounded card background.
      ctx.fillStyle = cardFillStyle;
      drawRoundRect(ctx, margin, cardY, CARD_WIDTH, cardHeight, radius);
      ctx.fill();

      const textY = cardY + centerY;

      // Draw the grey label (e.g. "Render FPS") on the left side of the card.
      ctx.textAlign = 'left';
      ctx.fillStyle = labelColor;
      ctx.fillText(label, margin + paddingLeft, textY);

      // Draw the colored value immediately after the label, with a small gap.
      const labelWidth = ctx.measureText(label).width;
      ctx.fillStyle = valueColor;
      ctx.fillText('  ' + value, margin + paddingLeft + labelWidth, textY);

      // Advance to the next card position.
      cardY += cardHeight + cardGap;
    }

    texture.needsUpdate = true;
  });

  /** Single uikit Image; texture is set via ref, width/height from props. */
  return (
    <Image ref={imageRef} width={width} height={height} objectFit="fill" keepAspectRatio={false} />
  );
}
