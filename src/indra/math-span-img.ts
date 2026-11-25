// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama
// Ported from datalab-to/indra indra_ocr/math_span_img.py (Apache-2.0, Copyright Datalab).

import type { InferenceSession } from "onnxruntime-common";

import type { FloatImage } from "./image.js";
import type { Box } from "./rec-preproc.js";
import { PAD_X, padBatch, widthBatches } from "./rec-preproc.js";
import type { TensorCtor, Tok } from "./types.js";
import { floats, item, output } from "./util.js";

/**
 * Inline math inside lines that MathRec does not read whole. A Cyrillic,
 * Korean, Arabic or Devanagari line with inline math stays with its script
 * recognizer, which reads the formula as plain characters. A small image
 * tagger marks the columns of such a line that hold math; MathRec reads each
 * marked run's crop, and a confident reading replaces the characters that the
 * recognizer read there.
 */

/** `math_span_img.json`. */
export type SpanImgMeta = {
  height: number;
  stride: number;
  threshold?: number;
  /** A shorter run is noise. */
  min_cols?: number;
  /** A shorter gap joins two runs. */
  merge_cols?: number;
  /**
   * A confident reading without structure ("$a = 0$") also replaces the run:
   * the script recognizer reads Latin letters as its own look-alikes.
   */
  accept_plain?: boolean;
};

/** A run of crop pixel columns [a, b). */
export type ColumnRun = readonly [a: number, b: number];

/** The image tagger of math columns (math_span_img.onnx). */
export class SpanImgTagger {
  public readonly height: number;
  public readonly acceptPlain: boolean;
  private readonly stride: number;
  private readonly threshold: number;
  private readonly minCols: number;
  private readonly mergeCols: number;

  public constructor(
    private readonly Tensor: TensorCtor,
    private readonly sess: InferenceSession,
    meta: SpanImgMeta
  ) {
    this.height = meta.height;
    this.stride = meta.stride;
    this.threshold = meta.threshold ?? 0.5;
    this.minCols = meta.min_cols ?? 2;
    this.mergeCols = meta.merge_cols ?? 1;
    this.acceptPlain = meta.accept_plain ?? false;
  }

  /** Column runs [a, b) at or above the threshold, short gaps joined. */
  public runs(prob: ArrayLike<number>, n: number): ColumnRun[] {
    const out: [number, number][] = [];
    let a: number | undefined;
    for (let i = 0; i <= n; i++) {
      const p = i < n ? (prob[i] ?? 0) : 0;
      if (p >= this.threshold && a === undefined) {
        a = i;
      } else if (p < this.threshold && a !== undefined) {
        const last = out.at(-1);
        if (last && a - last[1] <= this.mergeCols) last[1] = i;
        else out.push([a, i]);
        a = undefined;
      }
    }
    return out.filter(([x, y]) => y - x >= this.minCols);
  }

  /** Per ink map [height, W]: column runs that hold math, in crop pixels. */
  public async spans(crops: readonly FloatImage[], batch = 64): Promise<ColumnRun[][]> {
    const out: ColumnRun[][] = crops.map(() => []);
    for (const idx of widthBatches(
      crops.map((c) => c.width),
      batch
    )) {
      const { data, dims, widths } = padBatch(idx.map((j) => item(crops, j)));
      const res = await this.sess.run({
        x: new this.Tensor("float32", data, dims),
        widths: new this.Tensor("int64", widths, [idx.length]),
      });
      const prob = output(res, "prob");
      const c = prob.dims[1] ?? 0;
      const p = floats(prob);
      for (const [k, j] of idx.entries()) {
        const n = Math.floor(item(crops, j).width / this.stride);
        const row = p.subarray(k * c, k * c + Math.min(n, c));
        out[j] = this.runs(row, row.length).map(([a, b]) => [a * this.stride, b * this.stride]);
      }
    }
    return out;
  }
}

/** Crop pixel columns -> page x (the crop's padding and scale, as the recognizer's character positions). */
export function pageX([x0, y0, x1, y1]: Box, cropW: number, [a, b]: ColumnRun): [number, number] {
  const h = Math.max(1, y1 - y0);
  const ix0 = Math.floor(x0 - PAD_X * h);
  const f = (Math.ceil(x1 + PAD_X * h) - ix0) / cropW;
  return [ix0 + a * f, ix0 + b * f];
}

/** Whether a token is the formula of a splice. */
export function isFormula(t: Tok): boolean {
  return t[3]?.kind === "formula";
}

/**
 * The line's (char, x) tokens with those inside [x0, x1] replaced by one
 * formula token; a space token on each side keeps the formula a word. Each
 * replaced character stays as an empty token at its x: the gaps between
 * characters (cell and column cuts) are the same as before.
 */
export function splice(toks: readonly Tok[], x0: number, x1: number, latex: string): Tok[] {
  const xc = (x0 + x1) / 2;
  const left = toks.filter((t) => t[1] < x0);
  const right = toks.filter((t) => t[1] > x1);
  const inside = toks.filter((t) => x0 <= t[1] && t[1] <= x1);
  const pad = (s: string, x: number): Tok => [s, x, undefined, { kind: "pad" }];
  const held = inside.filter((t) => t[0] !== " ").map((t) => pad("", t[1]));
  // Stable, as Python's sorted: the empty tokens keep their order, the formula goes after equal xs.
  const middle = [
    ...held,
    [latex, xc, undefined, { kind: "formula", held: inside }] as const,
  ].toSorted((a, b) => a[1] - b[1]);
  let mid: Tok[] = [pad(" ", x0), ...middle, pad(" ", x1)];
  if (left.at(-1)?.[0] === " ") mid = mid.slice(1);
  if (right[0]?.[0] === " ") mid = mid.slice(0, -1);
  return [...left, ...mid, ...right];
}

/**
 * The tokens as the recognizer read them, before any splice (the table model
 * reads words and their characters: a formula word changes its structure).
 */
export function unspliced(toks: readonly Tok[]): readonly Tok[] {
  if (!toks.some(isFormula)) return toks;
  const out = toks.filter((t) => t[3] === undefined);
  for (const t of toks) if (t[3]?.kind === "formula") out.push(...t[3].held);
  return out.toSorted((a, b) => a[1] - b[1]);
}
