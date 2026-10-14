// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama
// Ported from datalab-to/indra indra_ocr/pipeline.py (Apache-2.0, Copyright Datalab).

import type { InferenceSession } from "onnxruntime-common";

import type { FloatImage, Gray } from "./image.js";
import type { MathReader } from "./math.js";
import { MATH_MIN_CONF } from "./math.js";
import { meanF32 } from "./numeric.js";
import type { Box } from "./rec-preproc.js";
import { PAD_X, padBatch, widthBatches } from "./rec-preproc.js";
import type { RecLine, TensorCtor, Tok } from "./types.js";
import { floats, ints, item, output } from "./util.js";
import type { Vocab } from "./vocab.js";

/**
 * A recognized line under this confidence (mean CTC peak probability) gets no
 * text: on empty form fields, checkboxes and rules the recognizer reads "-",
 * "I" or "1" at low confidence. The line's box stays.
 */
export const MIN_CONF = 0.5;

/** Lines that go to one recognizer, and whether its tokens are characters (CJK) or Latin ids. */
export type RecGroup = { sess: InferenceSession; vocab: Vocab; sel: number[]; asChars: boolean };

/** One recognizer batch: text, confidence and character positions per crop. */
export async function recognizeBatch(
  Tensor: TensorCtor,
  { sess, vocab, asChars }: RecGroup,
  crops: readonly FloatImage[],
  boxes: readonly Box[]
): Promise<RecLine[]> {
  const { data, dims, widths } = padBatch(crops);
  const res = await sess.run({
    x: new Tensor("float32", data, dims),
    widths: new Tensor("int64", widths, [crops.length]),
  });
  const ids = output(res, "ids");
  const best = ints(ids);
  const pmax = floats(output(res, "conf"));
  const lengths = ints(output(res, "lengths"));
  const steps = ids.dims[1] ?? 0;
  const stride = dims[3] / Math.max(steps, 1);
  return crops.map((crop, k) => {
    const n = item(lengths, k);
    const seq = best.slice(k * steps, k * steps + n);
    const kept: number[] = [];
    for (let t = 0; t < n; t++) {
      if (seq[t] !== 0 && (t === 0 || seq[t] !== seq[t - 1])) kept.push(t);
    }
    // Page x of each kept CTC step, through the crop's padding and scale.
    const [x0, y0, x1, y1] = item(boxes, k);
    const h = Math.max(1, y1 - y0);
    const ix0 = Math.floor(x0 - PAD_X * h);
    const f = (Math.ceil(x1 + PAD_X * h) - ix0) / crop.width;
    const keptIds = kept.map((t) => item(seq, t));
    // The CJK recognizer's ids mean nothing in the Latin vocab: its tokens
    // are characters, like the text-layer tokens.
    const toks = kept.map((t, i): Tok => {
      const x = ix0 + (t + 0.5) * stride * f;
      const id = item(keptIds, i);
      return asChars ? [item(vocab.chars, id - 1), x] : [id, x];
    });
    const conf = kept.length ? meanF32(kept.map((t) => pmax[k * steps + t] ?? 0)) : 0;
    return conf >= MIN_CONF
      ? { text: vocab.decode(keptIds), conf, toks }
      : { text: "", conf, toks: [] };
  });
}

/** Per line crop: does it hold Chinese characters (script.onnx)? All false without the model. */
export async function hanziLines(
  Tensor: TensorCtor,
  script: InferenceSession | undefined,
  crops: readonly FloatImage[],
  batch = 64
): Promise<boolean[]> {
  const out = crops.map(() => false);
  if (!script) return out;
  for (const idx of widthBatches(
    crops.map((c) => c.width),
    batch
  )) {
    const { data, dims, widths } = padBatch(idx.map((j) => item(crops, j)));
    const res = await script.run({
      x: new Tensor("float32", data, dims),
      widths: new Tensor("int64", widths, [idx.length]),
    });
    const p = floats(output(res, "p"));
    for (const [k, j] of idx.entries()) out[j] = (p[k] ?? 0) > 0.5;
  }
  return out;
}

/**
 * Lines the math classifier flags are read by the math reader. A reading
 * replaces the line only when it holds math and is confident; one token
 * holds the whole line, so it is never split.
 */
export async function readMath(
  math: MathReader,
  gray: Gray,
  boxes: readonly Box[],
  crops: readonly FloatImage[],
  out: RecLine[]
): Promise<void> {
  const idx = (await math.flags(crops)).flatMap((flag, j) => (flag ? [j] : []));
  if (!idx.length) return;
  const readings = await math.read(
    gray,
    idx.map((j) => item(boxes, j))
  );
  for (const [k, j] of idx.entries()) {
    const { text, conf } = item(readings, k);
    if (!text.includes("$") || conf < MATH_MIN_CONF) continue;
    const [x0, , x1] = item(boxes, j);
    out[j] = { text, conf: 1, toks: [[text, Math.fround(x0 + x1) / 2]] };
  }
}
