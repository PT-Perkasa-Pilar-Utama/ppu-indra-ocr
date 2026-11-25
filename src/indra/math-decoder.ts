// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama
// Ported from datalab-to/indra indra_ocr/math.py (Apache-2.0, Copyright Datalab).

import type { InferenceSession, Tensor } from "onnxruntime-common";

import { sumF32 } from "./numeric.js";
import type { TensorCtor } from "./types.js";
import { floats } from "./util.js";

/** Only the inputs a graph declares: exports drop unused ones (init does not read the mask). */
export function declared(
  sess: InferenceSession,
  feeds: Record<string, Tensor>
): Record<string, Tensor> {
  return Object.fromEntries(Object.entries(feeds).filter(([k]) => sess.inputNames.includes(k)));
}

/**
 * `t` with only the rows `rows` along `axis` (the batch axis), as numpy's
 * `x[rows]` / `x[:, rows]`: finished lines leave the decoding batch.
 */
export function takeRows(
  Tensor: TensorCtor,
  t: Tensor,
  rows: readonly number[],
  axis: number
): Tensor {
  const dims = [...t.dims];
  const outer = dims.slice(0, axis).reduce((a, b) => a * b, 1);
  const n = dims[axis] ?? 0;
  const inner = dims.slice(axis + 1).reduce((a, b) => a * b, 1);
  dims[axis] = rows.length;
  if (t.type === "float32") {
    const src = floats(t);
    const out = new Float32Array(outer * rows.length * inner);
    for (let o = 0; o < outer; o++) {
      for (const [r, row] of rows.entries()) {
        const from = (o * n + row) * inner;
        out.set(src.subarray(from, from + inner), (o * rows.length + r) * inner);
      }
    }
    return new Tensor("float32", out, dims);
  }
  if (t.type === "bool") {
    // SAFETY: bool tensors hold a Uint8Array (type checked above).
    const src = t.data as Uint8Array;
    const out = new Uint8Array(outer * rows.length * inner);
    for (let o = 0; o < outer; o++) {
      for (const [r, row] of rows.entries()) {
        const from = (o * n + row) * inner;
        out.set(src.subarray(from, from + inner), (o * rows.length + r) * inner);
      }
    }
    return new Tensor("bool", out, dims);
  }
  throw new Error(`cannot take rows of a ${t.type} tensor`);
}

const LETTER = /^\p{L}$/u;

/** A LaTeX command token: a backslash and a letter. */
export function isCommand(t: string): boolean {
  return t.length > 1 && t[0] === "\\" && LETTER.test(t[1] ?? "");
}

/** Per row: the argmax token and its softmax probability, in float32 as numpy computes it. */
export function topTokens(
  logits: Float32Array,
  rows: number,
  n: number
): { id: number; top: number }[] {
  const f = Math.fround;
  return Array.from({ length: rows }, (_, b) => {
    const row = logits.subarray(b * n, (b + 1) * n);
    let id = 0;
    for (let j = 1; j < n; j++) if ((row[j] ?? 0) > (row[id] ?? 0)) id = j;
    const max = row[id] ?? 0;
    const e = row.map((v) => f(Math.exp(f(v - max))));
    return { id, top: f(1 / sumF32(e)) };
  });
}
