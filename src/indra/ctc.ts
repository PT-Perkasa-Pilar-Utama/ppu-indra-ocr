// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama
// Ported from datalab-to/indra indra_ocr/ctc.py (Apache-2.0, Copyright Datalab).

import { sumF32 } from "./numeric.js";

/**
 * A recognizer exported with a "logits" output ([B, T, V]) ends at the greedy
 * ids, the lengths and the logits. ONNX Runtime runs every node whatever the
 * caller fetches, so the softmax and the top-k over all V classes are
 * computed here, only for the frames and lines that use them. The values are
 * the older graphs' outputs (the same ops in float32), so a line reads the same.
 */

const f = Math.fround;

/** Classes per step for the beam search (the older graphs' TopK). */
export const TOPK = 8;

/** The LM inputs of one line: blank log-probability [T], top-k values and classes [T, k]. */
export type LmInputs = { blank: Float32Array; topv: Float32Array; topi: Int32Array; k: number };

/** Row `t`'s max over `v` classes. */
function rowMax(logits: Float32Array, t: number, v: number): number {
  let m = -Infinity;
  for (let c = 0; c < v; c++) {
    const x = logits[t * v + c] ?? -Infinity;
    if (x > m) m = x;
  }
  return m;
}

/** `exp(logits[t] - max)` in float32, the softmax numerators numpy sums. */
function rowExp(logits: Float32Array, t: number, v: number, m: number, out: Float32Array): void {
  for (let c = 0; c < v; c++) out[c] = f(Math.exp(f((logits[t * v + c] ?? 0) - m)));
}

/** The probability of the best class at each of the `rows` steps (`steps` index rows of `logits`). */
export function peakProb(logits: Float32Array, v: number, steps: readonly number[]): Float32Array {
  const out = new Float32Array(steps.length);
  const e = new Float32Array(v);
  for (const [i, t] of steps.entries()) {
    rowExp(logits, t, v, rowMax(logits, t, v), e);
    out[i] = f(1 / sumF32(e));
  }
  return out;
}

/**
 * Steps [0, n) of one line's logits (row-major [T, V] from `offset`) -> the
 * blank log-probability and the k best classes' log-probabilities, best
 * first. Equal values keep the lower class first, as ONNX Runtime's TopK
 * does. With `floor` (a probability) only classes at or over it are kept;
 * the other places hold class 0 at -inf, which beamSearch skips, and its
 * candidates are the same as long as its own floor is at least this one.
 */
export function lmInputs(
  logits: Float32Array,
  offset: number,
  n: number,
  v: number,
  floor: number,
  k: number = TOPK
): LmInputs {
  const kk = Math.min(k, v);
  const blank = new Float32Array(n);
  const topv = new Float32Array(n * kk).fill(-Infinity);
  const topi = new Int32Array(n * kk);
  const row = logits.subarray(offset, offset + n * v);
  const e = new Float32Array(v);
  // beam_search compares in float64; the float32 slack catches rounding before the exact test.
  const lf = Math.log(floor);
  for (let t = 0; t < n; t++) {
    const m = rowMax(row, t, v);
    rowExp(row, t, v, m, e);
    const logs = f(Math.log(sumF32(e)));
    blank[t] = f(f((row[t * v] ?? 0) - m) - logs);
    const cut = floor > 0 ? f(logs + f(lf - 1e-3)) : -Infinity;
    const cand: [number, number][] = [];
    for (let c = 0; c < v; c++) {
      const d = f((row[t * v + c] ?? 0) - m);
      if (floor > 0 && d < cut) continue;
      const val = f(d - logs);
      if (floor > 0 && val < lf) continue;
      cand.push([val, c]);
    }
    if (floor <= 0 && cand.length > kk) {
      // The k-th best value; every class at or over it is a candidate.
      const kth = cand.map(([val]) => val).toSorted((a, b) => b - a)[kk - 1] ?? -Infinity;
      for (let i = cand.length - 1; i >= 0; i--) if ((cand[i]?.[0] ?? 0) < kth) cand.splice(i, 1);
    }
    cand.sort((a, b) => b[0] - a[0] || a[1] - b[1]);
    for (let i = 0; i < Math.min(kk, cand.length); i++) {
      const [val, c] = cand[i] ?? [-Infinity, 0];
      topv[t * kk + i] = val;
      topi[t * kk + i] = c;
    }
  }
  return { blank, topv, topi, k: kk };
}

/**
 * The blank's log-probability and the k most likely classes (log-probabilities,
 * best first) per frame, for frames [0, n) of each of B lines' logits
 * ([B, T, V] row-major): the text-layer check's view of a recognizer. Unlike
 * lmInputs, the log-softmax is `x - (max + log(sum(exp(x - max))))`, as the
 * reference computes it here.
 */
export function topKLogits(
  logits: Float32Array,
  B: number,
  T: number,
  v: number,
  n: number,
  k: number = TOPK
): LmInputs[] {
  const out: LmInputs[] = [];
  const e = new Float32Array(v);
  for (let b = 0; b < B; b++) {
    const blank = new Float32Array(n);
    const topv = new Float32Array(n * k);
    const topi = new Int32Array(n * k);
    for (let t = 0; t < n; t++) {
      const row = logits.subarray((b * T + t) * v, (b * T + t + 1) * v);
      const m = rowMax(row, 0, v);
      rowExp(row, 0, v, m, e);
      const lse = f(m + f(Math.log(sumF32(e))));
      blank[t] = f((row[0] ?? 0) - lse);
      // The k best classes, ties to the lower class.
      const best: number[] = [];
      for (let c = 0; c < v; c++) {
        const x = row[c] ?? -Infinity;
        if (best.length === k && x <= (row[best[k - 1] ?? 0] ?? -Infinity)) continue;
        let p = best.length;
        while (p > 0 && (row[best[p - 1] ?? 0] ?? -Infinity) < x) p--;
        best.splice(p, 0, c);
        if (best.length > k) best.pop();
      }
      for (const [i, c] of best.entries()) {
        topv[t * k + i] = f((row[c] ?? 0) - lse);
        topi[t * k + i] = c;
      }
    }
    out.push({ blank, topv, topi, k });
  }
  return out;
}
