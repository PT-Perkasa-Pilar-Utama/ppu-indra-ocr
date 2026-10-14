// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama
// Ported from datalab-to/indra indra_ocr/math.py (Apache-2.0, Copyright Datalab).

import type { InferenceSession, Tensor } from "onnxruntime-common";

import type { FloatImage, Gray, ImageOps } from "./image.js";
import { sumF32 } from "./numeric.js";
import type { Box } from "./rec-preproc.js";
import { cropLine, padBatch, widthBatches } from "./rec-preproc.js";
import type { TensorCtor } from "./types.js";
import { floats, item, output } from "./util.js";

const MATH_MIN = 0.5; // classifier probability above which a line is read as math
/**
 * A math reading replaces a line only at this mean token probability or more:
 * on print the model has not seen it writes confident-looking LaTeX that is
 * wrong, at low probability.
 */
export const MATH_MIN_CONF = 0.9;
/** A line this far below MATH_MIN_CONF after 8 tokens stops decoding: it would be rejected anyway. */
const EARLY_STOP_MARGIN = 0.1;
const BOS = 1;
const EOS = 2;
const N_SPECIAL = 4;

/** The math models: line classifier, encoder, and the decoder's init and step graphs. */
export type MathSessions = {
  cls: InferenceSession;
  enc: InferenceSession;
  init: InferenceSession;
  step: InferenceSession;
};

/** `math_meta.json`. */
export type MathMeta = {
  config: {
    height: number;
    max_tokens: number;
    dec_layers: number;
    dec_heads: number;
    channels: number;
  };
  tokens: string[];
};

/** A math reading: text with inline LaTeX, and the mean token probability. */
export type MathReading = { text: string; conf: number };

/** Tokens to text; inside math (between "$" tokens) they are joined by spaces. */
export function detokenize(tokens: readonly string[]): string {
  const out: string[] = [];
  let math: string[] = [];
  let inMath = false;
  for (const t of tokens) {
    if (t === "$") {
      if (inMath) {
        out.push(`$${math.join(" ")}$`);
        math = [];
      }
      inMath = !inMath;
    } else if (inMath) {
      math.push(t);
    } else {
      out.push(t);
    }
  }
  if (inMath && math.length) out.push(`$${math.join(" ")}$`);
  return out.join("").split(/\s+/u).filter(Boolean).join(" ");
}

/** Only the inputs a graph declares: exports drop unused ones (init does not read the mask). */
function declared(sess: InferenceSession, feeds: Record<string, Tensor>): Record<string, Tensor> {
  return Object.fromEntries(Object.entries(feeds).filter(([k]) => sess.inputNames.includes(k)));
}

/**
 * Math lines: the classifier finds lines that hold math, and an attention
 * decoder reads a taller crop of each one token per step, with its keys and
 * values cached.
 */
export class MathReader {
  private readonly itos: string[];
  private readonly headDim: number;

  public constructor(
    private readonly Tensor: TensorCtor,
    private readonly sessions: MathSessions,
    private readonly meta: MathMeta,
    private readonly ops: ImageOps
  ) {
    this.itos = ["", "", "", "", ...meta.tokens];
    this.headDim = meta.config.channels / meta.config.dec_heads;
  }

  /** Per line crop (the recognizer's crop): does it hold math? */
  public async flags(crops: readonly FloatImage[], batch = 64): Promise<boolean[]> {
    const out = crops.map(() => false);
    for (const idx of widthBatches(
      crops.map((c) => c.width),
      batch
    )) {
      const { data, dims, widths } = padBatch(idx.map((j) => item(crops, j)));
      const res = await this.sessions.cls.run({
        x: new this.Tensor("float32", data, dims),
        widths: new this.Tensor("int64", widths, [idx.length]),
      });
      const p = floats(output(res, "p"));
      for (const [k, j] of idx.entries()) out[j] = (p[k] ?? 0) > MATH_MIN;
    }
    return out;
  }

  /** Text with inline LaTeX and the mean token probability, for each box. */
  public async read(gray: Gray, boxes: readonly Box[], batch = 64): Promise<MathReading[]> {
    const { height } = this.meta.config;
    const crops = boxes.map((b) => cropLine(gray, b, height, this.ops));
    const out: MathReading[] = crops.map(() => ({ text: "", conf: 0 }));
    // Few, wide batches: each batch is one decoding loop of up to max_tokens
    // steps, and a step costs about the same for 8 lines as for 40.
    const widths = crops.map((c) => c.width);
    for (const idx of widthBatches(widths, batch, 4.0, 64, 16 * 64 * 640)) {
      const readings = await this.decodeBatch(idx.map((j) => item(crops, j)));
      for (const [k, j] of idx.entries()) out[j] = item(readings, k);
    }
    return out;
  }

  private async decodeBatch(crops: readonly FloatImage[]): Promise<MathReading[]> {
    const { enc, init, step } = this.sessions;
    const { dec_layers: layers, dec_heads: heads, max_tokens: maxTokens } = this.meta.config;
    const B = crops.length;
    const { data, dims, widths } = padBatch(crops);
    const encoded = await enc.run({
      x: new this.Tensor("float32", data, dims),
      widths: new this.Tensor("int64", widths, [B]),
    });
    const memory = output(encoded, "memory");
    const mask = output(encoded, "mask");
    const cross = await init.run(declared(init, { memory, mask }));
    const empty = new this.Tensor("float32", new Float32Array(0), [
      layers,
      B,
      heads,
      0,
      this.headDim,
    ]);
    let selfK: Tensor = empty;
    let selfV: Tensor = empty;
    let tok = new BigInt64Array(B).fill(BigInt(BOS));
    const seqs: number[][] = Array.from({ length: B }, () => []);
    const probs: number[][] = Array.from({ length: B }, () => []);
    const done = Array.from({ length: B }, () => false);
    for (let t = 0; t < maxTokens - 1; t++) {
      const res = await step.run(
        declared(step, {
          token: new this.Tensor("int64", tok, [B]),
          pos: new this.Tensor("int64", BigInt64Array.of(BigInt(t)), []),
          self_k: selfK,
          self_v: selfV,
          cross_k: output(cross, "cross_k"),
          cross_v: output(cross, "cross_v"),
          mask,
        })
      );
      selfK = output(res, "new_k");
      selfV = output(res, "new_v");
      const logits = output(res, "logits");
      const n = logits.dims[1] ?? 0;
      const ids = topTokens(floats(logits), B, n);
      tok = BigInt64Array.from(ids.map((p) => BigInt(p.id)));
      for (const [b, { id, top }] of ids.entries()) {
        if (done[b]) continue;
        const pb = item(probs, b);
        pb.push(top);
        if (id === EOS) {
          done[b] = true;
        } else if (pb.length >= 8 && meanOf(pb) < MATH_MIN_CONF - EARLY_STOP_MARGIN) {
          done[b] = true; // rejected: keeps a low mean probability
        } else if (id >= N_SPECIAL) {
          item(seqs, b).push(id);
        }
      }
      if (done.every(Boolean)) break;
    }
    return seqs.map((seq, b) => {
      const pb = item(probs, b);
      return {
        text: detokenize(seq.map((i) => item(this.itos, i))),
        conf: pb.length ? meanOf(pb) : 0,
      };
    });
  }
}

/** Per row: the argmax token and its softmax probability, in float32 as numpy computes it. */
function topTokens(logits: Float32Array, rows: number, n: number): { id: number; top: number }[] {
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

function meanOf(values: readonly number[]): number {
  return values.reduce((a, b) => a + b, 0) / values.length;
}
