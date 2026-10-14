// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama
// Ported from datalab-to/indra indra_ocr/order_post.py (Apache-2.0, Copyright Datalab).
//
// Float32 steps are rounded with Math.fround where the reference computes in
// numpy float32, so near-equal scores break the same way.

import { pyRound } from "./det-post.js";
import { sumF32 } from "./numeric.js";
import type { Box } from "./rec-preproc.js";
import { item } from "./util.js";

const f = Math.fround;

export const N_FEATS = 16;

/** Block classes the layout heads predict for each line. */
export const BLOCK_CLASSES = [
  "Text",
  "Section-Header",
  "Page-Header",
  "Page-Footer",
  "List-Group",
  "Table",
  "Caption",
  "Footnote",
  "Table-Of-Contents",
  "Form",
] as const;

export type BlockLabel = (typeof BLOCK_CLASSES)[number];

/** Table, Table-Of-Contents and Form: their lines read row by row. */
const GRID = new Set([5, 8, 9]);

// Bullets, dashes and arrows that start a list item.
const BULLETS = String.fromCharCode(0x2022, 0x2d, 0x2013, 0x25aa, 0x25e6, 0x2a, 0xb7, 0x2192);
// Python's str.isdigit: decimal digits plus superscripts, subscripts and circled digits.
const DIGIT = new RegExp(
  "^[\\p{Nd}\\u00b2\\u00b3\\u00b9\\u2070\\u2074-\\u2079\\u2080-\\u2089\\u2460-\\u2468\\u2474-\\u247c" +
    "\\u2488-\\u2490\\u24ea\\u24f5-\\u24fd\\u24ff\\u2776-\\u277e\\u2780-\\u2788\\u278a-\\u2792]$",
  "u"
);
const UPPER = /^\p{Uppercase}$/u;

const isDigits = (s: string): boolean => s.length > 0 && [...s].every((c) => DIGIT.test(c));

/** numpy's float32 median. */
export function medianF32(values: readonly number[]): number {
  const v = values.toSorted((a, b) => a - b);
  const m = Math.floor(v.length / 2);
  return v.length % 2 ? item(v, m) : f(f(item(v, m - 1) + item(v, m)) / 2);
}

/** [N, 16] float32 features of the lines for order.onnx: 10 geometric, 6 from the text. */
export function lineFeatures(
  boxes: readonly Box[],
  width: number,
  height: number,
  texts?: readonly string[]
): Float32Array {
  const n = boxes.length;
  const out = new Float32Array(n * N_FEATS);
  if (n === 0) return out;
  const minSide = f(1e-3);
  const ws = boxes.map(([x0, , x1]) => Math.max(f(x1 - x0), minSide));
  const hs = boxes.map(([, y0, , y1]) => Math.max(f(y1 - y0), minSide));
  const medH = medianF32(hs);
  const scaleCol = f((medH / height) * 20);
  for (const [i, [x0, y0, x1, y1]] of boxes.entries()) {
    const w = item(ws, i);
    const h = item(hs, i);
    const row = out.subarray(i * N_FEATS, (i + 1) * N_FEATS);
    row[0] = f(x0 / width);
    row[1] = f(y0 / height);
    row[2] = f(x1 / width);
    row[3] = f(y1 / height);
    row[4] = f(f(x0 + x1) / (2 * width));
    row[5] = f(f(y0 + y1) / (2 * height));
    row[6] = f(w / width);
    row[7] = f(Math.log(f(h / medH)));
    row[8] = f(f(Math.log(f(w / h))) / 4);
    row[9] = scaleCol;
    const t = (texts?.[i] ?? "").trim();
    if (!t) continue;
    const cps = [...t];
    const first = item(cps, 0);
    const last = item(cps, cps.length - 1);
    row[10] = Math.log1p(cps.length) / 5;
    row[11] = cps.filter((c) => DIGIT.test(c)).length / cps.length;
    const numbered =
      isDigits(
        cps
          .slice(0, 3)
          .join("")
          .replace(/[.)]+$/, "")
      ) && cps.length > 3;
    row[12] = BULLETS.includes(first) || numbered ? 1 : 0;
    row[13] = t.endsWith("-") ? 1 : 0;
    row[14] = ".!?:;".includes(last) ? 1 : 0;
    row[15] = UPPER.test(first) ? 1 : 0;
  }
  return out;
}

/** numpy's float32 logaddexp(0, x). */
function logAddExp0(x: number): number {
  if (x === 0) return f(Math.LN2);
  const big = Math.max(0, x);
  return f(big + f(Math.log1p(f(Math.exp(-Math.abs(x))))));
}

/**
 * Order from a pairwise logit matrix [n, n] (a[i, j] > 0: i before j). Start
 * from the row-sum ranking, then insertion moves: take each item out and put
 * it back where it most raises the summed log-probability of the pairs.
 */
export function decodeOrder(a: Float32Array, n: number, sweeps = 4): number[] {
  if (n <= 1) return Array.from({ length: n }, (_, i) => i);
  const lp = new Float32Array(n * n); // log P(i before j)
  for (let i = 0; i < n * n; i++) lp[i] = f(-logAddExp0(f(-(a[i] ?? 0))));
  const at = (i: number, j: number): number => lp[i * n + j] ?? 0;
  // -rowsum (pairwise, as numpy sums a row) + colsum (row by row, as numpy reduces axis 0)
  const col = new Float32Array(n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) col[j] = f((col[j] ?? 0) + at(i, j));
  const score = Array.from({ length: n }, (_, i) =>
    f(-sumF32(lp, i * n, (i + 1) * n) + (col[i] ?? 0))
  );
  let order = score.map((_, i) => i).toSorted((p, q) => item(score, p) - item(score, q));

  for (let s = 0; s < sweeps; s++) {
    let improved = false;
    for (const x of order) {
      const rest = order.filter((v) => v !== x);
      const m = rest.length;
      // gains[k]: x after rest[0..k-1] and before rest[k..]
      const gains = new Float64Array(m + 1);
      let acc = 0;
      for (let k = 0; k < m; k++) {
        acc = f(acc + at(item(rest, k), x));
        gains[k + 1] = acc;
      }
      acc = 0;
      for (let k = m - 1; k >= 0; k--) {
        acc = f(acc + at(x, item(rest, k)));
        gains[k] = (gains[k] ?? 0) + acc;
      }
      let k = 0;
      for (let j = 1; j <= m; j++) if ((gains[j] ?? 0) > (gains[k] ?? 0)) k = j;
      const cur = order.indexOf(x);
      if ((gains[k] ?? 0) > (gains[cur] ?? 0) + 1e-6) {
        order = [...rest.slice(0, k), x, ...rest.slice(k)];
        improved = true;
      }
    }
    if (!improved) break;
  }
  return order;
}

/** Rows top to bottom (centers within half a line height share a row), left to right. */
export function orderTopDown(boxes: readonly Box[]): number[] {
  if (!boxes.length) return [];
  const cy = boxes.map(([, y0, , y1]) => f(f(y0 + y1) / 2));
  const h = medianF32(boxes.map(([, y0, , y1]) => f(y1 - y0)));
  const half = f(h * 0.5);
  const unit = half > 1e-3 ? half : 1e-3;
  const key = cy.map((c) => pyRound(f(c / unit)));
  return boxes
    .map((_, i) => i)
    .toSorted((p, q) => item(key, p) - item(key, q) || item(boxes, p)[0] - item(boxes, q)[0]);
}

type Cut = { gap: number; k: number; order: number[] };

/** The widest gap between boxes along one axis (1: rows, 0: columns), after sorting by start. */
function bestCut(boxes: readonly Box[], axis: 0 | 1): Cut {
  const lo = boxes.map((b) => (axis === 1 ? b[1] : b[0]));
  const hi = boxes.map((b) => (axis === 1 ? b[3] : b[2]));
  const order = lo.map((_, i) => i).toSorted((p, q) => item(lo, p) - item(lo, q));
  let end = item(hi, item(order, 0));
  let gap = 0;
  let k = -1;
  for (let j = 1; j < order.length; j++) {
    const o = item(order, j);
    const g = f(item(lo, o) - end);
    if (g > gap) {
      gap = g;
      k = j;
    }
    end = Math.max(end, item(hi, o));
  }
  return { gap, k, order };
}

/**
 * Recursive XY-cut: split at the widest whitespace gap on either axis. The
 * upper part goes first after a row cut, the left part after a column cut.
 * Without a gap, top-down order.
 */
export function orderXYCut(boxes: readonly Box[]): number[] {
  const rec = (ids: number[]): number[] => {
    if (ids.length <= 1) return ids;
    const b = ids.map((i) => item(boxes, i));
    const y = bestCut(b, 1);
    const x = bestCut(b, 0);
    if (y.k < 0 && x.k < 0) return orderTopDown(b).map((i) => item(ids, i));
    const { k, order } = y.gap >= x.gap ? y : x;
    return [
      ...rec(order.slice(0, k).map((i) => item(ids, i))),
      ...rec(order.slice(k).map((i) => item(ids, i))),
    ];
  };
  return rec(boxes.map((_, i) => i));
}

function argmaxRow(m: Float64Array, row: number, n: number): number {
  let best = 0;
  for (let j = 1; j < n; j++) if ((m[row * n + j] ?? 0) > (m[row * n + best] ?? 0)) best = j;
  return best;
}

/**
 * Average-linkage clustering of pairwise "same group" logits [n, n]: merge the
 * two groups with the highest mean logit between them while that mean is
 * positive. One wrong pair cannot join two groups the other pairs keep apart.
 */
export function cluster(logits: ArrayLike<number>, n: number): number[][] {
  if (n === 0) return [];
  const S = Float64Array.from({ length: n * n }, (_, i) =>
    i % (n + 1) === 0 ? 0 : (logits[i] ?? 0)
  );
  const size = new Float64Array(n).fill(1);
  const alive = Array.from({ length: n }, () => true);
  const members = Array.from({ length: n }, (_, i) => [i]);
  const M = Float64Array.from(S, (v, i) => (i % (n + 1) === 0 ? -Infinity : v));
  const best = Array.from({ length: n }, (_, i) => argmaxRow(M, i, n));
  const bestV = Float64Array.from(best, (b, i) => M[i * n + b] ?? -Infinity);
  for (let step = 0; step < n - 1; step++) {
    let a = 0;
    for (let i = 1; i < n; i++) if ((bestV[i] ?? 0) > (bestV[a] ?? 0)) a = i;
    let b = item(best, a);
    if ((bestV[a] ?? 0) <= 0) break;
    [a, b] = [Math.min(a, b), Math.max(a, b)];
    for (let j = 0; j < n; j++) S[a * n + j] = (S[a * n + j] ?? 0) + (S[b * n + j] ?? 0);
    for (let i = 0; i < n; i++) S[i * n + a] = (S[i * n + a] ?? 0) + (S[i * n + b] ?? 0);
    S[a * n + a] = 0;
    size[a] = (size[a] ?? 0) + (size[b] ?? 0);
    alive[b] = false;
    members[a] = [...item(members, a), ...item(members, b)];
    const row = new Float64Array(n);
    for (let j = 0; j < n; j++) {
      row[j] =
        alive[j] && j !== a ? (S[a * n + j] ?? 0) / ((size[a] ?? 1) * (size[j] ?? 1)) : -Infinity;
    }
    for (let j = 0; j < n; j++) {
      M[a * n + j] = row[j] ?? -Infinity;
      M[j * n + a] = row[j] ?? -Infinity;
    }
    for (let j = 0; j < n; j++) {
      M[b * n + j] = -Infinity;
      M[j * n + b] = -Infinity;
    }
    bestV[b] = -Infinity;
    best[a] = argmaxRow(row, 0, n);
    bestV[a] = row[item(best, a)] ?? -Infinity;
    // other rows: (j, a) changed and (j, b) is gone
    for (let j = 0; j < n; j++) {
      if (j === a || !alive[j] || (best[j] !== a && best[j] !== b)) continue;
      best[j] = argmaxRow(M, j, n);
      bestV[j] = M[j * n + item(best, j)] ?? -Infinity;
    }
    for (let j = 0; j < n; j++) {
      if (j !== a && alive[j] && (row[j] ?? -Infinity) > (bestV[j] ?? 0)) {
        best[j] = a;
        bestV[j] = row[j] ?? -Infinity;
      }
    }
  }
  return members.filter((_, i) => alive[i]).map((m) => m.toSorted((p, q) => p - q));
}

/** Index of the largest per-class sum of the lines' class logits [n, C]. */
export function blockClass(cls: Float32Array, lines: readonly number[]): number {
  const c = BLOCK_CLASSES.length;
  const sums = new Float32Array(c);
  for (const i of lines)
    for (let k = 0; k < c; k++) sums[k] = f((sums[k] ?? 0) + (cls[i * c + k] ?? 0));
  let best = 0;
  for (let k = 1; k < c; k++) if ((sums[k] ?? 0) > (sums[best] ?? 0)) best = k;
  return best;
}

/**
 * Reading order with the layout heads: blocks from clustering the same-block
 * logits, lines inside a block by XY-cut (row by row in grid blocks), blocks
 * by the mean pairwise order logit between them.
 */
export function decodeBlocksOrder(
  a: Float32Array,
  same: Float32Array,
  boxes: readonly Box[],
  cls: Float32Array
): number[] {
  const n = boxes.length;
  if (n <= 1) return boxes.map((_, i) => i);
  const blocks = cluster(same, n);
  const inner = blocks.map((b) => {
    const bx = b.map((i) => item(boxes, i));
    const local = GRID.has(blockClass(cls, b)) ? orderTopDown(bx) : orderXYCut(bx);
    return local.map((k) => item(b, k));
  });
  // Mean order logit between blocks p < q, antisymmetric.
  const m = blocks.length;
  const ab = new Float32Array(m * m);
  for (let p = 0; p < m; p++) {
    for (let q = p + 1; q < m; q++) {
      const bp = item(blocks, p);
      const bq = item(blocks, q);
      let sum = 0;
      for (const i of bp) for (const j of bq) sum += a[i * n + j] ?? 0;
      const mean = sum / (bp.length * bq.length);
      ab[p * m + q] = f(mean);
      ab[q * m + p] = f(-mean);
    }
  }
  return decodeOrder(ab, m).flatMap((k) => item(inner, k));
}
