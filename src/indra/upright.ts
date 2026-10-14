// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama
// Ported from datalab-to/indra indra_ocr/pipeline.py (Apache-2.0, Copyright Datalab).

import { pyRound } from "./det-post.js";
import type { FloatImage, Gray, ImageOps } from "./image.js";
import { inkOf, rotationMatrix } from "./image.js";
import { meanF32, sumF32 } from "./numeric.js";
import type { Box } from "./rec-preproc.js";

export const ORIENT_SIDE = 768; // orient.onnx input: page thumbnail, long side, padded square
/**
 * A turn is applied only above this probability: real pages are almost all
 * upright, and a wrong turn loses the page.
 */
export const ORIENT_MIN = 0.95;
const DESKEW_MAX = 5.0; // degrees of tilt searched
export const DESKEW_MIN = 0.3; // smaller tilts are left alone (the search step is 0.1)
const SKEW_SIDE = 800;

const f = Math.fround;

/** The page scaled so its long side is `side`, with INTER_AREA (any direction). */
function thumbnail(gray: Gray, side: number, ops: ImageOps): Gray {
  const s = side / Math.max(gray.height, gray.width);
  const w = Math.max(1, pyRound(gray.width * s));
  const h = Math.max(1, pyRound(gray.height * s));
  return ops.resize(gray, w, h, "area");
}

/** The orient.onnx input: [side, side] ink map, long side scaled to `side`, padded with 0. */
export function orientInput(gray: Gray, ops: ImageOps, side: number = ORIENT_SIDE): Float32Array {
  const small = inkOf(thumbnail(gray, side, ops));
  const out = new Float32Array(side * side);
  for (let y = 0; y < small.height; y++) {
    out.set(small.data.subarray(y * small.width, (y + 1) * small.width), y * side);
  }
  return out;
}

/**
 * Degrees (counter-clockwise) that straighten the text rows of a page: the
 * angle whose horizontal ink profile changes most sharply row to row, on a
 * page thumbnail, searched in 0.5 degree steps and then 0.1 degree steps.
 */
export function estimateSkew(gray: Gray, ops: ImageOps, side: number = SKEW_SIDE): number {
  const small = thumbnail(gray, side, ops);
  // (255 - g) / 255, not 1 - g / 255: the two differ in the last float32 bit.
  const data = new Float32Array(small.data.length);
  for (let i = 0; i < data.length; i++) data[i] = f(f(255 - (small.data[i] ?? 0)) / 255);
  if (meanF32(data) < 0.003) return 0; // a blank page has no rows to straighten
  const ink: FloatImage = { data, width: small.width, height: small.height };
  const cx = small.width / 2;
  const cy = small.height / 2;

  const sharpness = (a: number): number => {
    const r = ops.warpFloat(ink, rotationMatrix(cx, cy, a), small.width, small.height);
    const rows = new Float32Array(r.height);
    for (let y = 0; y < r.height; y++) rows[y] = sumF32(r.data, y * r.width, (y + 1) * r.width);
    const sq = new Float32Array(Math.max(0, r.height - 1));
    for (let y = 1; y < r.height; y++) {
      const d = f((rows[y] ?? 0) - (rows[y - 1] ?? 0));
      sq[y - 1] = f(d * d);
    }
    return sumF32(sq);
  };

  const best = argmax(arange(-DESKEW_MAX, DESKEW_MAX + 1e-6, 0.5), sharpness);
  return argmax(arange(best - 0.5, best + 0.5 + 1e-6, 0.1), sharpness);
}

/** `np.arange` for floats: start + i * delta, where delta is the rounded first step. */
function arange(start: number, stop: number, step: number): number[] {
  const n = Math.ceil((stop - start) / step);
  const delta = start + step - start;
  return Array.from({ length: Math.max(0, n) }, (_, i) => start + i * delta);
}

/** The first value with the highest key, as Python's max(values, key=...). */
function argmax(values: readonly number[], key: (v: number) => number): number {
  let best = values[0] ?? 0;
  let bestKey = -Infinity;
  for (const v of values) {
    const k = key(v);
    if (k > bestKey) {
      best = v;
      bestKey = k;
    }
  }
  return best;
}

/** The page rotated by `degrees` counter-clockwise about its center (same size, edges replicated). */
export function rotatePage(gray: Gray, degrees: number, ops: ImageOps): Gray {
  const m = rotationMatrix(gray.width / 2, gray.height / 2, degrees);
  return ops.warpReplicate(gray, m, gray.width, gray.height);
}

/** The page turned clockwise by `k` quarter turns (`np.rot90(gray, -k)`). */
export function rot90(gray: Gray, k: number): Gray {
  let out = gray;
  for (let t = 0; t < ((k % 4) + 4) % 4; t++) {
    const { width: w, height: h, data } = out;
    const next = new Uint8Array(w * h);
    // Clockwise: the new row y is the old column y read bottom to top.
    for (let y = 0; y < w; y++) {
      for (let x = 0; x < h; x++) next[y * h + x] = data[(h - 1 - x) * w + y] ?? 0;
    }
    out = { data: next, width: h, height: w };
  }
  return out;
}

/**
 * A box of the page that rotatePage(.., degrees) made, mapped back to the
 * page before it: the bounding box of the four mapped corners, clipped to it.
 */
export function unrotateBox(box: Box, degrees: number, width: number, height: number): Box {
  const [a, b, c, d, e, g] = rotationMatrix(width / 2, height / 2, -degrees);
  const [x0, y0, x1, y1] = box;
  const xs = [x0, x1, x0, x1];
  const ys = [y0, y0, y1, y1];
  const mx = xs.map((x, i) => a * x + b * (ys[i] ?? 0) + c);
  const my = xs.map((x, i) => d * x + e * (ys[i] ?? 0) + g);
  const clip = (v: number, hi: number): number => roundTo(Math.min(Math.max(v, 0), hi), 1);
  return [
    clip(Math.min(...mx), width),
    clip(Math.min(...my), height),
    clip(Math.max(...mx), width),
    clip(Math.max(...my), height),
  ];
}

/** Python's round(x, digits): nearest, exact ties to even. */
export function roundTo(x: number, digits: number): number {
  const p = 10 ** digits;
  const t = x * p;
  if (Math.abs(t - Math.trunc(t)) === 0.5) return pyRound(t) / p;
  return Number(x.toFixed(digits));
}
