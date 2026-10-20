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
 * numpy's linear percentiles (2, 50, 98 by default) of the uint8 pixels in
 * rows y0..y1 and columns x0..x1 of the page, from their histogram.
 */
function percentiles(
  gray: Gray,
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  qs: readonly number[] = [0.02, 0.5, 0.98]
): number[] {
  const cum = new Float64Array(256);
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) cum[gray.data[y * gray.width + x] ?? 0]++;
  }
  for (let v = 1; v < 256; v++) cum[v] = (cum[v] ?? 0) + (cum[v - 1] ?? 0);
  const n = (x1 - x0) * (y1 - y0);
  // np.searchsorted(cum, i, side="right"): the first value whose count passes i.
  const above = (i: number): number => {
    let v = 0;
    while (v < 256 && (cum[v] ?? 0) <= i) v++;
    return v;
  };
  return qs.map((q) => {
    const x = q * (n - 1);
    const i = Math.floor(x);
    const g = x - i;
    const lo = above(i);
    const hi = above(Math.min(i + 1, n - 1));
    const d = hi - lo;
    // numpy's lerp: from the nearer end
    return g >= 0.5 ? hi - d * (1 - g) : lo + d * g;
  });
}

/**
 * Light text on a dark ground (a banner, a dark table head, a slide): the
 * ground (the median of the line box) lies nearer the dark end than the text
 * does, so the bright tail is the longer one. The recognizers read dark text
 * on light paper, so such a line is read inverted.
 */
export function isDarkLine(gray: Gray, box: Box): boolean {
  const [x0, y0, x1, y1] = box;
  const bx0 = Math.max(0, Math.trunc(x0));
  const by0 = Math.max(0, Math.trunc(y0));
  const bx1 = Math.min(gray.width, Math.trunc(x1) + 1);
  const by1 = Math.min(gray.height, Math.trunc(y1) + 1);
  if (bx1 <= bx0 || by1 <= by0) return false;
  const [lo = 0, med = 0, hi = 0] = percentiles(gray, bx0, by0, bx1, by1);
  return hi - med > 2 * (med - lo) + 8;
}

/**
 * Crop one line for the recognizer: grow the box by PAD_Y x its height above
 * and below and PAD_X x its height left and right (paper outside the page),
 * then scale it to `height` pixels high. Returns the float32 ink map.
 *
 * A tall window (`window` > 1) keeps the text scale and adds the page above
 * and below the line: `window * height` pixels high, its middle band the
 * window-1 crop. A light-on-dark line (`dark`, decided by isDarkLine when not
 * given) is padded with black and not inverted.
 */
export function cropLine(
  gray: Gray,
  box: Box,
  height: number,
  ops: ImageOps,
  maxWidth = 4096,
  window = 1,
  dark?: boolean
): FloatImage {
  const padY = (window * (1 + 2 * PAD_Y) - 1) / 2;
  const outH = height * window;
  const [x0, y0, x1, y1] = box;
  const h = Math.max(1, y1 - y0);
  const ix0 = Math.floor(x0 - PAD_X * h);
  const iy0 = Math.floor(y0 - padY * h);
  const ix1 = Math.ceil(x1 + PAD_X * h);
  const iy1 = Math.ceil(y1 + padY * h);
  const cw = Math.max(1, ix1 - ix0);
  const ch = Math.max(1, iy1 - iy0);
  const isDark = dark ?? isDarkLine(gray, box);
  const crop = new Uint8Array(cw * ch).fill(isDark ? 0 : 255);
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
  const outW = Math.max(WIDTH_MULTIPLE, Math.min(maxWidth, pyRound((cw * outH) / ch)));
  const resized = ops.resize(
    { data: crop, width: cw, height: ch },
    outW,
    outH,
    ch > outH ? "area" : "linear"
  );
  return isDark ? paperOf(resized) : inkOf(resized);
}

/** A light-on-dark crop as an ink map: the pixel value itself, scaled to 0..1. */
function paperOf(img: Gray): FloatImage {
  const data = new Float32Array(img.data.length);
  for (let i = 0; i < data.length; i++) data[i] = Math.fround((img.data[i] ?? 0) / 255);
  return { data, width: img.width, height: img.height };
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
