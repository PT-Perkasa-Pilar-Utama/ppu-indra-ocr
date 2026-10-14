// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama
// Ported from datalab-to/indra indra_ocr/rec_preproc.py (Apache-2.0, Copyright Datalab).

import { pyRound } from "./det-post.js";
import type { FloatImage, Gray, ImageOps } from "./image.js";
import { inkOf } from "./image.js";
import { item } from "./util.js";

export const LINE_HEIGHT = 32;
export const PAD_Y = 0.2;
export const PAD_X = 0.3;
const WIDTH_MULTIPLE = 8; // widths are padded to a multiple of this (timestep stride)

/** An axis-aligned box: x0, y0, x1, y1 in page pixels. */
export type Box = readonly [number, number, number, number];

/**
 * Crop one line for the recognizer: grow the box by PAD_Y x its height above
 * and below and PAD_X x its height left and right (white outside the page),
 * then scale it to `height` pixels high. Returns the float32 ink map.
 */
export function cropLine(
  gray: Gray,
  box: Box,
  height: number,
  ops: ImageOps,
  maxWidth = 4096
): FloatImage {
  const [x0, y0, x1, y1] = box;
  const h = Math.max(1, y1 - y0);
  const ix0 = Math.floor(x0 - PAD_X * h);
  const iy0 = Math.floor(y0 - PAD_Y * h);
  const ix1 = Math.ceil(x1 + PAD_X * h);
  const iy1 = Math.ceil(y1 + PAD_Y * h);
  const cw = Math.max(1, ix1 - ix0);
  const ch = Math.max(1, iy1 - iy0);
  const crop = new Uint8Array(cw * ch).fill(255);
  const sx0 = Math.max(0, ix0);
  const sy0 = Math.max(0, iy0);
  const sx1 = Math.min(gray.width, ix1);
  const sy1 = Math.min(gray.height, iy1);
  if (sx1 > sx0 && sy1 > sy0) {
    for (let y = sy0; y < sy1; y++) {
      const row = gray.data.subarray(y * gray.width + sx0, y * gray.width + sx1);
      crop.set(row, (y - iy0) * cw + (sx0 - ix0));
    }
  }
  const outW = Math.max(WIDTH_MULTIPLE, Math.min(maxWidth, pyRound((cw * height) / ch)));
  const resized = ops.resize(
    { data: crop, width: cw, height: ch },
    outW,
    height,
    ch > height ? "area" : "linear"
  );
  return inkOf(resized);
}

/** A recognizer batch: [B, 1, H, W] ink maps and each line's real width. */
export type PaddedBatch = {
  data: Float32Array;
  dims: [number, number, number, number];
  widths: BigInt64Array;
};

/** Right-pad ink maps to one width (a multiple of 8). */
export function padBatch(lines: readonly FloatImage[]): PaddedBatch {
  const h = item(lines, 0).height;
  const wm = Math.ceil(Math.max(...lines.map((l) => l.width)) / WIDTH_MULTIPLE) * WIDTH_MULTIPLE;
  const data = new Float32Array(lines.length * h * wm);
  const widths = new BigInt64Array(lines.length);
  for (const [i, line] of lines.entries()) {
    for (let y = 0; y < h; y++) {
      data.set(line.data.subarray(y * line.width, (y + 1) * line.width), (i * h + y) * wm);
    }
    widths[i] = BigInt(line.width);
  }
  return { data, dims: [lines.length, 1, h, wm], widths };
}

/**
 * Group line indices into batches of similar width, narrowest first. A batch
 * closes at `maxBatch` lines, when the next line is wider than `maxRatio`
 * times the batch's first (widths below `minWidth` count as `minWidth`), or
 * when lines x width would pass `maxArea`: ONNX Runtime keeps its largest
 * allocation, so one wide batch holds memory for the whole run.
 */
export function widthBatches(
  widths: readonly number[],
  maxBatch = 64,
  maxRatio = 1.3,
  minWidth = 64,
  maxArea: number = 64 * 640
): number[][] {
  // Stable sort, as Python's sorted().
  const order = widths.map((_, i) => i).toSorted((a, b) => item(widths, a) - item(widths, b));
  const batches: number[][] = [];
  for (const i of order) {
    const last = batches.at(-1);
    const wi = Math.max(item(widths, i), minWidth);
    if (
      last &&
      last.length < maxBatch &&
      wi <= maxRatio * Math.max(item(widths, item(last, 0)), minWidth) &&
      (last.length + 1) * wi <= maxArea
    ) {
      last.push(i);
    } else {
      batches.push([i]);
    }
  }
  return batches;
}
