// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama
// Ported from datalab-to/indra indra_ocr/det_post.py (Apache-2.0, Copyright Datalab).

import type { FloatImage, Gray, ImageOps } from "./image.js";
import { inkOf } from "./image.js";

const CORE_X_SHRINK = 0.15; // the core is shorter than the line by this fraction of its height, at each end
export const OUT_STRIDE = 2;
const INPUT_MULTIPLE = 32; // H and W must divide by this

/** The detector input (float32 ink map) and the scale from page to input pixels. */
export type DetInput = { input: FloatImage; scale: number };

/** Component labels (0 = background) and left, width, area per label. */
export type Components = { labels: Int32Array; count: number; stats: Int32Array };

/** A detected line: x0, y0, x1, y1 in pixels, and the mean core probability. */
export type DetBox = [number, number, number, number, number];

/**
 * Resize a page so its long side is `longSide` and pad it with white to a
 * multiple of 32. Returns the float32 ink map the detector reads, and the scale.
 */
export function pageToInput(gray: Gray, longSide: number, ops: ImageOps): DetInput {
  const { width: w, height: h } = gray;
  const scale = longSide / Math.max(h, w);
  const nh = Math.max(1, pyRound(h * scale));
  const nw = Math.max(1, pyRound(w * scale));
  let page = gray;
  if (nh !== h || nw !== w) page = ops.resize(gray, nw, nh, scale < 1 ? "area" : "linear");
  const ph = Math.ceil(nh / INPUT_MULTIPLE) * INPUT_MULTIPLE;
  const pw = Math.ceil(nw / INPUT_MULTIPLE) * INPUT_MULTIPLE;
  const ink = inkOf(page);
  const data = new Float32Array(ph * pw); // ink map: 0 = white
  for (let y = 0; y < nh; y++) data.set(ink.data.subarray(y * nw, (y + 1) * nw), y * pw);
  return { input: { data, width: pw, height: ph }, scale };
}

/**
 * Decode the detector output for one page. `out` is [3, h, w]: core logit,
 * distance to the line top, distance to the line bottom. Returns boxes in
 * detector-input pixels, in the order OpenCV labels their components.
 */
export function decode(
  out: Float32Array,
  h: number,
  w: number,
  thresh = 0.5,
  minArea = 2
): DetBox[] {
  const n = h * w;
  const prob = new Float32Array(n);
  const mask = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    prob[i] = Math.fround(1 / Math.fround(1 + Math.fround(Math.exp(-(out[i] ?? 0)))));
    mask[i] = (prob[i] ?? 0) > thresh ? 1 : 0;
  }
  const { labels, count, stats } = connectedComponents4(mask, h, w);
  if (count <= 1) return [];
  const s = OUT_STRIDE;
  const cnt = new Float64Array(count);
  const sumTop = new Float64Array(count);
  const sumBot = new Float64Array(count);
  const sumScore = new Float64Array(count);
  for (let y = 0; y < h; y++) {
    const ys = Math.fround((y + 0.5) * s);
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const k = labels[i] ?? 0;
      if (k === 0) continue;
      cnt[k] = (cnt[k] ?? 0) + 1;
      sumTop[k] = (sumTop[k] ?? 0) + Math.fround(ys - (out[n + i] ?? 0));
      sumBot[k] = (sumBot[k] ?? 0) + Math.fround(ys + (out[2 * n + i] ?? 0));
      sumScore[k] = (sumScore[k] ?? 0) + (prob[i] ?? 0);
    }
  }
  const boxes: DetBox[] = [];
  for (let k = 1; k < count; k++) {
    const area = stats[k * 3 + 2] ?? 0;
    if (area < minArea) continue;
    const c = Math.max(Math.fround(cnt[k] ?? 0), 1);
    const y0 = (sumTop[k] ?? 0) / c;
    const y1 = (sumBot[k] ?? 0) / c;
    const lh = Math.max(1, y1 - y0);
    const ex = Math.min(CORE_X_SHRINK * lh, 0.5 * lh);
    const left = stats[k * 3] ?? 0;
    const width = stats[k * 3 + 1] ?? 0;
    boxes.push([
      Math.fround(left * s - ex),
      Math.fround(y0),
      Math.fround((left + width) * s + ex),
      Math.fround(y1),
      Math.fround((sumScore[k] ?? 0) / c),
    ]);
  }
  return boxes;
}

/**
 * 4-connected component labels, numbered like OpenCV: in raster order of each
 * component's first pixel. Label 0 is the background. `stats` holds left,
 * width and area for each label.
 */
export function connectedComponents4(mask: Uint8Array, h: number, w: number): Components {
  // Union-find over provisional labels, then relabel by first appearance.
  const prov = new Int32Array(h * w);
  const parent: number[] = [0];
  const find = (a: number): number => {
    let r = a;
    while (parent[r] !== r) r = parent[r] ?? r;
    while (parent[a] !== r) {
      const next = parent[a] ?? r;
      parent[a] = r;
      a = next;
    }
    return r;
  };
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!mask[i]) continue;
      const up = y > 0 ? (prov[i - w] ?? 0) : 0;
      const left = x > 0 ? (prov[i - 1] ?? 0) : 0;
      if (!up && !left) {
        parent.push(parent.length);
        prov[i] = parent.length - 1;
      } else if (up && left) {
        const a = find(up);
        const b = find(left);
        if (a !== b) parent[Math.max(a, b)] = Math.min(a, b);
        prov[i] = Math.min(a, b);
      } else {
        prov[i] = up || left;
      }
    }
  }
  const final = new Int32Array(parent.length);
  let count = 1;
  const labels = new Int32Array(h * w);
  const lefts: number[] = [0];
  const rights: number[] = [0];
  const areas: number[] = [0];
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!prov[i]) continue;
      const root = find(prov[i] ?? 0);
      if (!final[root]) {
        final[root] = count++;
        lefts.push(x);
        rights.push(x);
        areas.push(0);
      }
      const k = final[root] ?? 0;
      labels[i] = k;
      if (x < (lefts[k] ?? x)) lefts[k] = x;
      if (x > (rights[k] ?? x)) rights[k] = x;
      areas[k] = (areas[k] ?? 0) + 1;
    }
  }
  const stats = new Int32Array(count * 3);
  for (let k = 1; k < count; k++) {
    const left = lefts[k] ?? 0;
    stats[k * 3] = left;
    stats[k * 3 + 1] = (rights[k] ?? left) - left + 1;
    stats[k * 3 + 2] = areas[k] ?? 0;
  }
  return { labels, count, stats };
}

/** Python's round(): half to even. */
export function pyRound(x: number): number {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}
