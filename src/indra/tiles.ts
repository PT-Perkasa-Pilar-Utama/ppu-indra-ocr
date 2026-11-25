// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama
// Ported from datalab-to/indra indra_ocr/pipeline.py (Apache-2.0, Copyright Datalab).

import type { DetBox } from "./det-post.js";
import { OUT_STRIDE, pyRound } from "./det-post.js";
import type { Gray } from "./image.js";
import type { DetectPass } from "./types.js";
import { item } from "./util.js";

/**
 * Large pages (plans, posters): when the lines of a page whose long side is
 * over TILE_MIN_SIDE px are still under ADAPTIVE_MIN_H after the adaptive
 * pass, the page is detected again in overlapping tiles at TILE_SCALE (a
 * 200 dpi render -> 150 dpi), where the whole-page pass sees small text at
 * 30-60 dpi. Only tiles over text-bearing regions of the whole-page pass run
 * (its text probability over TILE_REGION_P).
 */
export const TILE_MIN_SIDE = 4096;
const TILE_SCALE = 0.75;
const TILE = 1024; // detector input side per tile
const TILE_OVERLAP = 128; // detector input px shared by neighbouring tiles
const TILE_REGION_P = 0.15;
/**
 * Line boxes in page pixels from overlapping TILE x TILE detector inputs at
 * TILE_SCALE. Each tile keeps the boxes whose centre lies in its own core
 * (the tile minus half the overlap on every inner side); a line cut by a
 * core edge comes back in pieces from two tiles, and pieces from different
 * tiles that share a text row and touch are joined.
 */
export async function detectTiled(
  gray: Gray,
  pass: DetectPass,
  detectAt: (gray: Gray, longSide: number) => Promise<DetectPass>
): Promise<DetBox[]> {
  const f = Math.fround;
  const { logit, scale: s } = pass;
  const region = dilate5(textRegion(logit.data, TILE_REGION_P), logit.width, logit.height);
  const { width: w, height: h } = gray;
  const k = TILE_SCALE;
  const step = TILE - TILE_OVERLAP;
  const span = TILE / k; // page px per tile
  const sh = Math.ceil(h * k);
  const sw = Math.ceil(w * k);
  const ys = range(0, Math.max(1, sh - TILE_OVERLAP), step);
  const xs = range(0, Math.max(1, sw - TILE_OVERLAP), step);
  // (tile, box, may continue in a neighbouring tile)
  const found: { tile: string; box: DetBox; edge: boolean }[] = [];
  for (const ty of ys) {
    for (const tx of xs) {
      // The page box of this tile, and the same box on the region map.
      const py0 = ty / k;
      const px0 = tx / k;
      const py1 = Math.min(h, py0 + span);
      const px1 = Math.min(w, px0 + span);
      const r0 = Math.trunc((py0 * s) / OUT_STRIDE);
      const r1 = Math.ceil((py1 * s) / OUT_STRIDE) + 1;
      const c0 = Math.trunc((px0 * s) / OUT_STRIDE);
      const c1 = Math.ceil((px1 * s) / OUT_STRIDE) + 1;
      if (!anyIn(region, logit.width, logit.height, r0, r1, c0, c1)) continue;
      const crop = cropGray(gray, Math.trunc(px0), Math.trunc(py0), Math.ceil(px1), Math.ceil(py1));
      const tile = await detectAt(crop, pyRound(Math.max(crop.width, crop.height) * k));
      if (!tile.boxes.length) continue;
      const cs = tile.scale;
      // The core: half the overlap in from every edge that has a neighbour.
      const m = TILE_OVERLAP / 2 / k;
      const cx0 = px0 + (tx > 0 ? m : -1e9);
      const cy0 = py0 + (ty > 0 ? m : -1e9);
      const cx1 = px1 - (tx + step < sw - TILE_OVERLAP ? m : -1e9);
      const cy1 = py1 - (ty + step < sh - TILE_OVERLAP ? m : -1e9);
      for (const [bx0, by0, bx1, by1, score] of tile.boxes) {
        // float32 boxes, and Python floats taken as float32 next to them (NumPy 2 promotion)
        const row: DetBox = [
          f(f(bx0 / cs) + px0),
          f(f(by0 / cs) + py0),
          f(f(bx1 / cs) + px0),
          f(f(by1 / cs) + py0),
          score,
        ];
        const cx = f(f(row[0] + row[2]) / 2);
        const cy = f(f(row[1] + row[3]) / 2);
        if (!(cx >= f(cx0) && cx < f(cx1) && cy >= f(cy0) && cy < f(cy1))) continue;
        const lh = f(row[3] - row[1]);
        const edge =
          row[0] < f(cx0 + lh) ||
          row[2] > f(cx1 - lh) ||
          row[1] < f(cy0 + lh) ||
          row[3] > f(cy1 - lh);
        found.push({ tile: `${ty},${tx}`, box: row, edge });
      }
    }
  }
  if (!found.length) return [];
  const boxes = found.map((b) => b.box);
  const alive = found.map(() => true);
  // Join pieces of one line from two tiles: a shared text row (vertical
  // overlap over half the lower box) and a horizontal gap under one line height.
  const edges = found.flatMap((b, i) => (b.edge ? [i] : []));
  for (const a of edges) {
    if (!alive[a]) continue;
    for (const c of edges) {
      if (c === a || !alive[c] || item(found, c).tile === item(found, a).tile) continue;
      const A = item(boxes, a);
      const C = item(boxes, c);
      const hgt = Math.min(f(A[3] - A[1]), f(C[3] - C[1]));
      const vov = f(Math.min(A[3], C[3]) - Math.max(A[1], C[1]));
      const gap = f(Math.max(A[0], C[0]) - Math.min(A[2], C[2]));
      if (hgt > 0 && vov > f(0.5 * hgt) && gap < hgt) {
        boxes[a] = [
          Math.min(A[0], C[0]),
          Math.min(A[1], C[1]),
          Math.max(A[2], C[2]),
          Math.max(A[3], C[3]),
          Math.max(A[4], C[4]),
        ];
        alive[c] = false;
      }
    }
  }
  const clip = (v: number, hi: number): number => Math.min(Math.max(v, 0), hi);
  return boxes
    .filter((_, i) => alive[i])
    .map(([x0, y0, x1, y1, score]) => [clip(x0, w), clip(y0, h), clip(x1, w), clip(y1, h), score]);
}

/** `range(start, stop, step)`. */
function range(start: number, stop: number, step: number): number[] {
  const out: number[] = [];
  for (let v = start; v < stop; v += step) out.push(v);
  return out;
}

/** Cells whose text probability is over `p`: the logit over log(p / (1 - p)). */
function textRegion(logit: Float32Array, p: number): Uint8Array {
  // float32 logits compared with the float64 threshold, as numpy compares them
  const cut = Math.log(p / (1 - p));
  const out = new Uint8Array(logit.length);
  for (const [i, v] of logit.entries()) out[i] = v > cut ? 1 : 0;
  return out;
}

/** `cv2.dilate` with a 5 x 5 square: the max over the window, clipped at the edges. */
function dilate5(src: Uint8Array, w: number, h: number): Uint8Array {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let v = 0;
      for (let dy = -2; dy <= 2 && !v; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= h) continue;
        for (let dx = -2; dx <= 2; dx++) {
          const xx = x + dx;
          if (xx >= 0 && xx < w && src[yy * w + xx]) {
            v = 1;
            break;
          }
        }
      }
      out[y * w + x] = v;
    }
  }
  return out;
}

/** Whether any cell of rows [r0, r1) and columns [c0, c1) is set (numpy slice bounds clip). */
function anyIn(
  map: Uint8Array,
  w: number,
  h: number,
  r0: number,
  r1: number,
  c0: number,
  c1: number
): boolean {
  for (let y = Math.max(0, r0); y < Math.min(h, r1); y++) {
    for (let x = Math.max(0, c0); x < Math.min(w, c1); x++) if (map[y * w + x]) return true;
  }
  return false;
}

/** `gray[y0:y1, x0:x1]` (bounds clipped as numpy clips them). */
function cropGray(gray: Gray, x0: number, y0: number, x1: number, y1: number): Gray {
  const a = Math.max(0, x0);
  const b = Math.min(gray.width, x1);
  const c = Math.max(0, y0);
  const d = Math.min(gray.height, y1);
  const width = Math.max(0, b - a);
  const height = Math.max(0, d - c);
  const data = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) {
    data.set(
      gray.data.subarray((c + y) * gray.width + a, (c + y) * gray.width + a + width),
      y * width
    );
  }
  return { data, width, height };
}
