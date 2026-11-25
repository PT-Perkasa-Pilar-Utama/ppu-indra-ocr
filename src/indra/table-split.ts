// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama
// Ported from datalab-to/indra indra_ocr/table_rel.py (Apache-2.0, Copyright Datalab).
//
// tsr_split.onnx (the TableNet separator model): row and column separator maps
// of a table crop, and the word features read from them.

import type { InferenceSession } from "onnxruntime-common";

import { pyRound } from "./det-post.js";
import type { Gray, ImageOps } from "./image.js";
import { inkOf } from "./image.js";
import type { TableBox } from "./table-features.js";
import type { TensorCtor } from "./types.js";
import { floats, item, output } from "./util.js";

const f = Math.fround;

export const N_SPLIT = 8; // split features, the last one = 1 when they are present
const SPLIT_TEXT_PX = 12; // tsr_split.onnx's line height (it expects text 8-16 px high)
const SPLIT_MAX_SIDE = 1024;
const SPLIT_PAD_TO = 32;
const SPLIT_STRIDE = 2; // map px per crop px: 1/2

/**
 * The model's row separator, column separator and header probability maps, each
 * [h, w] at 1/2 of the scaled crop, stacked: `prob` is [3, h, w]. `scale` is
 * map px per page px; `origin` the crop's top-left corner in page px.
 */
export type SplitMaps = {
  prob: Float32Array;
  h: number;
  w: number;
  scale: number;
  origin: readonly [number, number];
};

const ceilDiv = (a: number, b: number): number => -Math.floor(-a / b);

/** The model's input: the table crop scaled to text 12 px high, as ink, padded to a multiple of 32. */
export type SplitInput = {
  x: Float32Array;
  /** The padded size. */
  ph: number;
  pw: number;
  /** The scaled crop's size. */
  nh: number;
  nw: number;
  /** Scaled px per page px. */
  scale: number;
  origin: readonly [number, number];
};

/**
 * The tsr_split.onnx input for the table at `bbox` (page px) of `gray`, or null
 * for a crop too small to read. `lineH` is the median word height.
 */
export function splitInput(
  ops: ImageOps,
  gray: Gray,
  bbox: TableBox,
  lineH: number
): SplitInput | null {
  const { width: W, height: H } = gray;
  const m = 1.5 * lineH; // context around the lines (the model expects 0.3-2 text heights)
  const x0 = Math.max(Math.trunc(f(bbox[0] - f(m))), 0);
  const y0 = Math.max(Math.trunc(f(bbox[1] - f(m))), 0);
  const x1 = Math.min(Math.ceil(f(bbox[2] + f(m))), W);
  const y1 = Math.min(Math.ceil(f(bbox[3] + f(m))), H);
  const cw = x1 - x0;
  const ch = y1 - y0;
  if (cw <= 0 || ch <= 0) return null;
  let s = SPLIT_TEXT_PX / Math.max(lineH, 1);
  if (Math.max(cw, ch) * s > SPLIT_MAX_SIDE) s = SPLIT_MAX_SIDE / Math.max(cw, ch);
  const nh = Math.max(1, pyRound(ch * s));
  const nw = Math.max(1, pyRound(cw * s));
  if (Math.min(nh, nw) < 16) return null;
  const crop = new Uint8Array(cw * ch);
  for (let y = 0; y < ch; y++)
    crop.set(gray.data.subarray((y0 + y) * W + x0, (y0 + y) * W + x1), y * cw);
  const small = ops.resize(
    { data: crop, width: cw, height: ch },
    nw,
    nh,
    s < 1 ? "area" : "linear"
  );
  const ph = ceilDiv(nh, SPLIT_PAD_TO) * SPLIT_PAD_TO;
  const pw = ceilDiv(nw, SPLIT_PAD_TO) * SPLIT_PAD_TO;
  const x = new Float32Array(ph * pw);
  const ink = inkOf(small).data;
  for (let y = 0; y < nh; y++) x.set(ink.subarray(y * nw, (y + 1) * nw), y * pw);
  return { x, ph, pw, nh, nw, scale: s, origin: [x0, y0] };
}

/** tsr_split.onnx on the table at `bbox` (page px) of `gray`, or null for a crop too small to read. */
export async function splitMaps(
  session: InferenceSession,
  Tensor: TensorCtor,
  ops: ImageOps,
  gray: Gray,
  bbox: TableBox,
  lineH: number
): Promise<SplitMaps | null> {
  const input = splitInput(ops, gray, bbox, lineH);
  if (!input) return null;
  const { x, ph, pw, nh, nw, scale, origin } = input;
  const res = await session.run({ x: new Tensor("float32", x, [1, 1, ph, pw]) });
  const out = output(res, item(session.outputNames, 0));
  const logits = floats(out);
  const [, , h2 = 0, w2 = 0] = out.dims;
  const vh = ceilDiv(nh, SPLIT_STRIDE);
  const vw = ceilDiv(nw, SPLIT_STRIDE);
  if (vh > h2 || vw > w2)
    throw new RangeError(`tsr_split map ${h2}x${w2} is smaller than ${vh}x${vw}`);
  const prob = new Float32Array(3 * vh * vw);
  for (let c = 0; c < 3; c++)
    for (let y = 0; y < vh; y++)
      for (let xx = 0; xx < vw; xx++) {
        const v = item(logits, (c * h2 + y) * w2 + xx);
        prob[(c * vh + y) * vw + xx] = f(1 / f(1 + f(Math.exp(f(-v)))));
      }
  return { prob, h: vh, w: vw, scale: scale / SPLIT_STRIDE, origin };
}

/**
 * Per word: the highest mean of P [h, w] over columns [lo, hi) on any of 8 rows
 * spread over [a, b] (map px, float32 arrays); 0 where the area is off the map.
 */
function band(
  P: Float32Array,
  [h, w]: readonly [number, number],
  [a, b]: readonly [Float32Array, Float32Array],
  [lo, hi]: readonly [Float32Array, Float32Array]
): number[] {
  const k = 8;
  const cs = new Float32Array(h * (w + 1));
  for (let r = 0; r < h; r++)
    for (let c = 0; c < w; c++)
      cs[r * (w + 1) + c + 1] = item(cs, r * (w + 1) + c) + item(P, r * w + c); // sequential float32, as np.cumsum
  const clip = (v: number, top: number): number => Math.min(Math.max(v, 0), top);
  return Array.from(a, (a0, i) => {
    const c0 = clip(Math.floor(item(lo, i)), w);
    const c1 = Math.max(clip(Math.ceil(item(hi, i)), w), Math.min(c0 + 1, w));
    const span = f(item(b, i) - a0);
    let best = -Infinity;
    for (let j = 0; j < k; j++) {
      const r = Math.floor(f(a0 + f(span * ((j + 0.5) / k))));
      const rr = clip(r, h - 1);
      const v = f(item(cs, rr * (w + 1) + c1) - item(cs, rr * (w + 1) + c0)) / Math.max(c1 - c0, 1);
      best = Math.max(best, r >= 0 && r < h && c1 > c0 ? v : 0);
    }
    return best;
  });
}

/**
 * [N, N_SPLIT] features of word `boxes` (float32 [N, 4], page px) from
 * `splitMaps`: row separator just below / above the word, column separator just
 * left / right of it, a row (column) separator through it, its mean header
 * probability, and 1 (present). Zeros (absent) when `maps` is null.
 */
export function splitFeatures(boxes: Float32Array, maps: SplitMaps | null): Float32Array {
  const n = boxes.length / 4;
  const out = new Float32Array(n * N_SPLIT);
  if (!maps || n === 0) return out;
  const { prob, h: mh, w: mw, scale: s, origin } = maps;
  const plane = (c: number): Float32Array => prob.subarray(c * mh * mw, (c + 1) * mh * mw);
  const pr = plane(0);
  const pc = plane(1);
  const ph = plane(2);
  const pcT = new Float32Array(mw * mh); // the column map transposed: [w, h]
  for (let y = 0; y < mh; y++) for (let x = 0; x < mw; x++) pcT[x * mh + y] = item(pc, y * mw + x);
  const col = (j: number): Float32Array =>
    Float32Array.from({ length: n }, (_, i) => item(boxes, i * 4 + j));
  const scaled = (v: Float32Array, o: number): Float32Array => v.map((e) => f(f(e - f(o)) * f(s)));
  const [x0, x1] = [scaled(col(0), origin[0]), scaled(col(2), origin[0])];
  const [y0, y1] = [scaled(col(1), origin[1]), scaled(col(3), origin[1])];
  const h = y1.map((e, i) => Math.max(f(e - item(y0, i)), 1));
  // `base + c * h`: c a Python float, h a float32 array
  const edge = (base: Float32Array, c: number): Float32Array =>
    base.map((e, i) => f(e + f(f(c) * item(h, i))));
  const rows = [mh, mw] as const;
  const cols = [mw, mh] as const;
  const put = (j: number, v: number[]): void => {
    for (const [i, e] of v.entries()) out[i * N_SPLIT + j] = e;
  };
  put(0, band(pr, rows, [edge(y1, -0.1), edge(y1, 0.8)], [x0, x1]));
  put(1, band(pr, rows, [edge(y0, -0.8), edge(y0, 0.1)], [x0, x1]));
  put(2, band(pcT, cols, [edge(x0, -2), edge(x0, 0.1)], [y0, y1]));
  put(3, band(pcT, cols, [edge(x1, -0.1), edge(x1, 2)], [y0, y1]));
  put(4, band(pr, rows, [edge(y0, 0.3), edge(y1, -0.3)], [x0, x1]));
  put(5, band(pcT, cols, [edge(x0, 0.2), edge(x1, -0.2)], [y0, y1]));
  put(6, band(ph, rows, [y0, y1], [x0, x1]));
  for (let i = 0; i < n; i++) out[i * N_SPLIT + 7] = 1;
  return out;
}
