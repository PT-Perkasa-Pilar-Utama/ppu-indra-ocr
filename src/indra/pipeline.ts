// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama
// Ported from datalab-to/indra indra_ocr/pipeline.py (Apache-2.0, Copyright Datalab).

import type { InferenceSession, Tensor } from "onnxruntime-common";

import type { DetBox } from "./det-post.js";
import { decode, pageToInput } from "./det-post.js";
import type { FloatImage, Gray, ImageOps } from "./image.js";
import { meanF32 } from "./numeric.js";
import type { Box } from "./rec-preproc.js";
import { PAD_X, cropLine, padBatch, widthBatches } from "./rec-preproc.js";
import { floats, ints, item, output } from "./util.js";
import { Vocab } from "./vocab.js";

const ADAPTIVE_MIN_H = 9.0; // median line height (input px) that triggers a second detection pass
const ADAPTIVE_TARGET_H = 12.0;
const ADAPTIVE_MAX_SIDE = 2048;

/**
 * A recognized line under this confidence (mean CTC peak probability) gets no
 * text: on empty form fields, checkboxes and rules the recognizer reads "-",
 * "I" or "1" at low confidence. The line's box stays.
 */
export const MIN_CONF = 0.5;

/**
 * One recognized character position: a class id (Latin recognizer) or a
 * character (CJK recognizer, PDF text layer, math), its page x, and for text
 * layer characters their index in the PDF content stream.
 */
export type Tok = readonly [id: number | string, x: number, contentIndex?: number];

/** Text, confidence and character positions of one detected line. */
export type RecLine = { text: string; conf: number; toks: readonly Tok[] };

/** One detector pass: boxes in detector-input pixels, and the page scale. */
export type DetectPass = { boxes: DetBox[]; scale: number };

/** The `meta.json` that ships with the models. */
export type IndraMeta = {
  vocab: string;
  vocab_cjk?: string;
  rec_height: number;
  det_long_side: number;
};

/** ONNX Runtime sessions of the Indra models. Optional ones switch a stage off. */
export type IndraSessions = {
  det: InferenceSession;
  rec: InferenceSession;
  order?: InferenceSession;
  orient?: InferenceSession;
  script?: InferenceSession;
  recCjk?: InferenceSession;
};

export type TensorCtor = typeof Tensor;

type RecGroup = { sess: InferenceSession; vocab: Vocab; sel: number[]; asChars: boolean };

/**
 * The OCR pipeline on one grayscale page. Each stage is its own method, so
 * callers can run the whole page or chain stages themselves.
 */
export class IndraPipeline {
  public readonly vocab: Vocab;
  public readonly vocabCjk: Vocab | undefined;
  public readonly longSide: number;
  public readonly height: number;

  public constructor(
    private readonly Tensor: TensorCtor,
    private readonly sessions: IndraSessions,
    meta: IndraMeta,
    private readonly ops: ImageOps,
    private readonly adaptive = true
  ) {
    this.vocab = Vocab.fromJson(meta.vocab);
    this.vocabCjk =
      sessions.script && sessions.recCjk && meta.vocab_cjk
        ? Vocab.fromJson(meta.vocab_cjk)
        : undefined;
    this.longSide = meta.det_long_side;
    this.height = meta.rec_height;
  }

  /** One detector pass with the long side at `longSide`. */
  public async detectAt(gray: Gray, longSide: number): Promise<DetectPass> {
    const { input, scale } = pageToInput(gray, longSide, this.ops);
    const x = new this.Tensor("float32", input.data, [1, 1, input.height, input.width]);
    const out = output(await this.sessions.det.run({ x }), "out");
    const [, , h = 0, w = 0] = out.dims;
    return { boxes: decode(floats(out), h, w), scale };
  }

  /**
   * Line boxes (x0, y0, x1, y1, score) in page pixels. Pages with small text
   * get a second pass at a larger size.
   */
  public async detect(gray: Gray): Promise<DetBox[]> {
    let { boxes, scale } = await this.detectAt(gray, this.longSide);
    if (this.adaptive && boxes.length >= 8) {
      const medH = medianF32(boxes.map((b) => Math.fround(b[3] - b[1])));
      if (medH < ADAPTIVE_MIN_H) {
        const target = (this.longSide * ADAPTIVE_TARGET_H) / Math.max(medH, 1);
        const ls = Math.floor(Math.trunc(Math.min(ADAPTIVE_MAX_SIDE, target)) / 32) * 32;
        if (ls > this.longSide) ({ boxes, scale } = await this.detectAt(gray, ls));
      }
    }
    // The reference divides its float32 boxes by the scale in float32.
    const s = Math.fround(scale);
    const clip = (v: number, hi: number): number => Math.min(Math.max(Math.fround(v / s), 0), hi);
    return boxes.map(([x0, y0, x1, y1, score]) => [
      clip(x0, gray.width),
      clip(y0, gray.height),
      clip(x1, gray.width),
      clip(y1, gray.height),
      score,
    ]);
  }

  /**
   * Text, confidence and character positions per line. Lines with Chinese
   * characters go to the CJK recognizer when the script model is loaded. A
   * line under MIN_CONF comes back empty.
   */
  public async recognize(gray: Gray, boxes: readonly Box[], batch = 64): Promise<RecLine[]> {
    const crops = boxes.map((b) => cropLine(gray, b, this.height, this.ops));
    const out: RecLine[] = crops.map(() => ({ text: "", conf: 0, toks: [] }));
    const all = crops.map((_, i) => i);
    let groups: RecGroup[] = [
      { sess: this.sessions.rec, vocab: this.vocab, sel: all, asChars: false },
    ];
    const { recCjk } = this.sessions;
    if (recCjk && this.vocabCjk && crops.length) {
      const hanzi = await this.hanziLines(crops, batch);
      groups = [
        {
          sess: this.sessions.rec,
          vocab: this.vocab,
          sel: all.filter((j) => !hanzi[j]),
          asChars: false,
        },
        { sess: recCjk, vocab: this.vocabCjk, sel: all.filter((j) => hanzi[j]), asChars: true },
      ];
    }
    for (const group of groups) {
      const widths = group.sel.map((j) => item(crops, j).width);
      for (const part of widthBatches(widths, batch)) {
        const idx = part.map((i) => item(group.sel, i));
        const lines = await this.recognizeBatch(
          group,
          idx.map((j) => item(crops, j)),
          idx.map((j) => item(boxes, j))
        );
        for (const [k, j] of idx.entries()) out[j] = item(lines, k);
      }
    }
    return out;
  }

  private async recognizeBatch(
    { sess, vocab, asChars }: RecGroup,
    crops: readonly FloatImage[],
    boxes: readonly Box[]
  ): Promise<RecLine[]> {
    const { data, dims, widths } = padBatch(crops);
    const res = await sess.run({
      x: new this.Tensor("float32", data, dims),
      widths: new this.Tensor("int64", widths, [crops.length]),
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

  /** Per line crop: does it hold Chinese characters (script.onnx)? */
  public async hanziLines(crops: readonly FloatImage[], batch = 64): Promise<boolean[]> {
    const out = crops.map(() => false);
    const script = this.sessions.script;
    if (!script) return out;
    for (const idx of widthBatches(
      crops.map((c) => c.width),
      batch
    )) {
      const { data, dims, widths } = padBatch(idx.map((j) => item(crops, j)));
      const res = await script.run({
        x: new this.Tensor("float32", data, dims),
        widths: new this.Tensor("int64", widths, [idx.length]),
      });
      const p = floats(output(res, "p"));
      for (const [k, j] of idx.entries()) out[j] = (p[k] ?? 0) > 0.5;
    }
    return out;
  }
}

/** numpy's median of float32 values (mean of the middle two in float32). */
function medianF32(values: readonly number[]): number {
  const v = values.toSorted((a, b) => a - b);
  const m = Math.floor(v.length / 2);
  return v.length % 2 ? item(v, m) : Math.fround(Math.fround(item(v, m - 1) + item(v, m)) / 2);
}
