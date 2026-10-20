// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama

import type { Affine, Gray, ImageOps } from "./image.js";

// ponytail: plain float math, not OpenCV's fixed-point tables; pixels differ
// from cv2 by a gray level or two. Use the opencv engine where parity matters.

const clampByte = (v: number): number => Math.min(255, Math.max(0, Math.round(v)));

/** Source coordinate weights for one output axis of a linear resize (half-pixel centers). */
type Taps = { i0: Int32Array; i1: Int32Array; w: Float64Array };

function linearTaps(out: number, size: number): Taps {
  const scale = size / out;
  const i0 = new Int32Array(out);
  const i1 = new Int32Array(out);
  const w = new Float64Array(out);
  for (let d = 0; d < out; d++) {
    const s = Math.min(Math.max((d + 0.5) * scale - 0.5, 0), size - 1);
    const f = Math.floor(s);
    i0[d] = f;
    i1[d] = Math.min(f + 1, size - 1);
    w[d] = s - f;
  }
  return { i0, i1, w };
}

function resizeLinear(src: Gray, width: number, height: number): Gray {
  const xs = linearTaps(width, src.width);
  const ys = linearTaps(height, src.height);
  const out = new Uint8Array(width * height);
  const s = src.data;
  for (let y = 0; y < height; y++) {
    const r0 = (ys.i0[y] ?? 0) * src.width;
    const r1 = (ys.i1[y] ?? 0) * src.width;
    const wy = ys.w[y] ?? 0;
    for (let x = 0; x < width; x++) {
      const a = xs.i0[x] ?? 0;
      const b = xs.i1[x] ?? 0;
      const wx = xs.w[x] ?? 0;
      const top = (s[r0 + a] ?? 0) * (1 - wx) + (s[r0 + b] ?? 0) * wx;
      const bot = (s[r1 + a] ?? 0) * (1 - wx) + (s[r1 + b] ?? 0) * wx;
      out[y * width + x] = clampByte(top * (1 - wy) + bot * wy);
    }
  }
  return { data: out, width, height };
}

/** Box-filter weights for one output axis: each output cell averages the source span it covers. */
type Span = { start: number; weights: number[] };

function areaSpans(out: number, size: number): Span[] {
  const scale = size / out;
  return Array.from({ length: out }, (_, d) => {
    const lo = d * scale;
    const hi = Math.min((d + 1) * scale, size);
    const start = Math.floor(lo);
    const weights: number[] = [];
    for (let i = start; i < hi; i++)
      weights.push((Math.min(i + 1, hi) - Math.max(i, lo)) / (hi - lo));
    return { start, weights };
  });
}

function resizeArea(src: Gray, width: number, height: number): Gray {
  // Shrinking averages areas; growing has no areas to average, so it is linear, as in OpenCV.
  if (width > src.width || height > src.height) return resizeLinear(src, width, height);
  const xs = areaSpans(width, src.width);
  const ys = areaSpans(height, src.height);
  const out = new Uint8Array(width * height);
  const row = new Float64Array(width);
  for (const [y, sy] of ys.entries()) {
    row.fill(0);
    for (const [k, wy] of sy.weights.entries()) {
      const base = (sy.start + k) * src.width;
      for (const [x, sx] of xs.entries()) {
        let acc = 0;
        for (const [j, wx] of sx.weights.entries())
          acc += (src.data[base + sx.start + j] ?? 0) * wx;
        row[x] = (row[x] ?? 0) + acc * wy;
      }
    }
    for (let x = 0; x < width; x++) out[y * width + x] = clampByte(row[x] ?? 0);
  }
  return { data: out, width, height };
}

/** The inverse of an affine map: destination pixel -> source pixel. */
function invert([a, b, c, d, e, f]: Affine): Affine {
  const det = a * e - b * d;
  const ia = e / det;
  const ib = -b / det;
  const id = -d / det;
  const ie = a / det;
  return [ia, ib, -(ia * c + ib * f), id, ie, -(id * c + ie * f)];
}

/**
 * Bilinear warp of 8-bit pixels; `edge` decides what a sample outside the
 * source reads. Writes rounded bytes straight into the output: the skew
 * search runs this some 80 times a page.
 */
function warp(
  src: Gray,
  m: Affine,
  width: number,
  height: number,
  edge: "replicate" | "zero"
): Gray {
  const [a, b, c, d, e, f] = invert(m);
  const out = new Uint8Array(width * height);
  const { width: sw, height: sh, data } = src;
  const read = (x: number, y: number): number => {
    if (x >= 0 && x < sw && y >= 0 && y < sh) return data[y * sw + x] ?? 0;
    if (edge === "zero") return 0;
    return data[Math.min(Math.max(y, 0), sh - 1) * sw + Math.min(Math.max(x, 0), sw - 1)] ?? 0;
  };
  for (let y = 0; y < height; y++) {
    const rowX = b * y + c;
    const rowY = e * y + f;
    const row = y * width;
    for (let x = 0; x < width; x++) {
      const sx = a * x + rowX;
      const sy = d * x + rowY;
      const x0 = Math.floor(sx);
      const y0 = Math.floor(sy);
      const wx = sx - x0;
      const wy = sy - y0;
      let v: number;
      if (x0 >= 0 && y0 >= 0 && x0 + 1 < sw && y0 + 1 < sh) {
        // Fast path: all four neighbours inside the source.
        const i = y0 * sw + x0;
        const top = (data[i] ?? 0) * (1 - wx) + (data[i + 1] ?? 0) * wx;
        const bot = (data[i + sw] ?? 0) * (1 - wx) + (data[i + sw + 1] ?? 0) * wx;
        v = top * (1 - wy) + bot * wy;
      } else if (edge === "zero" && (x0 + 1 < 0 || y0 + 1 < 0 || x0 >= sw || y0 >= sh)) {
        continue; // all four neighbours outside: zero
      } else {
        v =
          (read(x0, y0) * (1 - wx) + read(x0 + 1, y0) * wx) * (1 - wy) +
          (read(x0, y0 + 1) * (1 - wx) + read(x0 + 1, y0 + 1) * wx) * wy;
      }
      // A weighted mean of bytes: in [0, 255], so rounding half up is clampByte.
      out[row + x] = Math.round(v);
    }
  }
  return { data: out, width, height };
}

/** ImageOps in plain TypeScript: no OpenCV, the same code on every platform. */
export const tsImageOps: ImageOps = {
  resize(src, width, height, interpolation) {
    if (width === src.width && height === src.height) return { ...src, data: src.data.slice() };
    return interpolation === "area"
      ? resizeArea(src, width, height)
      : resizeLinear(src, width, height);
  },
  warpReplicate(src, m, width, height): Gray {
    return warp(src, m, width, height, "replicate");
  },
  warpZero(src, m, width, height): Gray {
    return warp(src, m, width, height, "zero");
  },
};
