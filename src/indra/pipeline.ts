// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama
// Ported from datalab-to/indra indra_ocr/pipeline.py (Apache-2.0, Copyright Datalab).

import type { InferenceSession } from "onnxruntime-common";

import type { DetBox } from "./det-post.js";
import { decode, pageToInput } from "./det-post.js";
import type { FloatImage, Gray, ImageOps } from "./image.js";
import { buildPage, splitParts } from "./assemble.js";
import { meanF32 } from "./numeric.js";
import {
  N_FEATS,
  decodeBlocksOrder,
  decodeOrder,
  lineFeatures,
  medianF32,
  orderXYCut,
} from "./order-post.js";
import type { Box } from "./rec-preproc.js";
import { PAD_X, cropLine, padBatch, widthBatches } from "./rec-preproc.js";
import { floats, ints, item, output } from "./util.js";
import {
  DESKEW_MIN,
  ORIENT_MIN,
  ORIENT_SIDE,
  estimateSkew,
  orientInput,
  rot90,
  rotatePage,
  roundTo,
  unrotateBox,
} from "./upright.js";
import type {
  DetectPass,
  IndraMeta,
  IndraSessions,
  Orientation,
  Page,
  ReadingOrder,
  RecLine,
  Straightened,
  TensorCtor,
  Tok,
} from "./types.js";
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
    private readonly furniture = false,
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

  /**
   * Quarter turns (counter-clockwise) the page content is turned by. 0 without
   * orient.onnx, or when no turn is more likely than ORIENT_MIN.
   */
  public async orient(gray: Gray): Promise<Orientation> {
    const orient = this.sessions.orient;
    if (!orient) return { turns: 0, probs: [] };
    const x = new this.Tensor("float32", orientInput(gray, this.ops), [
      1,
      1,
      ORIENT_SIDE,
      ORIENT_SIDE,
    ]);
    const probs = Array.from(floats(output(await orient.run({ x }), "p")));
    let k = 0;
    for (const [i, p] of probs.entries()) if (p > (probs[k] ?? 0)) k = i;
    return { turns: (probs[k] ?? 0) > ORIENT_MIN ? k : 0, probs };
  }

  /**
   * The page turned upright by `turns` quarter turns, then straightened when
   * its rows tilt by DESKEW_MIN degrees or more.
   */
  public straighten(gray: Gray, turns: number): Straightened {
    const image = turns ? rot90(gray, turns) : gray;
    const a = estimateSkew(image, this.ops);
    return Math.abs(a) >= DESKEW_MIN
      ? { image: rotatePage(image, a, this.ops), turns, skew: a }
      : { image, turns, skew: 0 };
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

  /**
   * OCR of one page image: turn upright (unless `turn` is false), deskew,
   * detect, recognize, order and lay out.
   */
  public async page(gray: Gray, turn = true): Promise<Page> {
    const t: Record<string, number> = {};
    let t0 = performance.now();
    const { turns } = turn ? await this.orient(gray) : { turns: 0 };
    const { image, skew } = this.straighten(gray, turns);
    t["orient_ms"] = performance.now() - t0;
    t0 = performance.now();
    const boxes = await this.detect(image);
    t["det_ms"] = performance.now() - t0;
    t0 = performance.now();
    const rec = await this.recognize(image, boxes.map(toBox));
    t["rec_ms"] = performance.now() - t0;
    const page = await this.assemble(image.width, image.height, boxes.map(toBox), rec, t);
    page.rotation = 90 * turns;
    page.skew = roundTo(skew, 2);
    if (skew) {
      // Boxes back on the page as turned, before the deskew.
      const back = (b: Box): Box => unrotateBox(b, skew, image.width, image.height);
      page.lines = page.lines.map((ln) => ({ ...ln, box: back(ln.box) }));
      page.blocks = page.blocks.map((b) => ({ ...b, bbox: back(b.bbox) }));
    }
    return page;
  }

  /**
   * Lines -> split at wide gaps -> reading order -> blocks and markdown. `rec`
   * holds the text, confidence and tokens of each box.
   */
  public async assemble(
    width: number,
    height: number,
    boxes: readonly Box[],
    rec: readonly RecLine[],
    t: Record<string, number> = {}
  ): Promise<Page> {
    const t0 = performance.now();
    const parts = splitParts(this.vocab, boxes, rec);
    const ro = await this.readingOrder(
      parts.map((p) => p.box),
      parts.map((p) => p.text),
      width,
      height
    );
    t["order_ms"] = performance.now() - t0;
    return buildPage(this.vocab, parts, ro, this.furniture, t);
  }

  /**
   * Reading order of the lines. With order.onnx's layout heads: blocks first,
   * then lines inside blocks, plus the class and same-block logits.
   */
  public async readingOrder(
    boxes: readonly Box[],
    texts: readonly string[],
    width: number,
    height: number
  ): Promise<ReadingOrder> {
    const order = this.sessions.order;
    const n = boxes.length;
    if (n < 2 || !order) return { order: n ? orderXYCut(boxes) : [] };
    const feats = lineFeatures(boxes, width, height, texts);
    const res = await order.run({
      feats: new this.Tensor("float32", feats, [1, n, N_FEATS]),
      mask: new this.Tensor("bool", new Uint8Array(n).fill(1), [1, n]),
    });
    const logits = floats(output(res, "a"));
    if (!res["cls"] || !res["same"]) return { order: decodeOrder(logits, n), logits };
    const cls = floats(output(res, "cls"));
    const same = floats(output(res, "same"));
    return { order: decodeBlocksOrder(logits, same, boxes, cls), cls, same, logits };
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

function toBox(b: DetBox): Box {
  return [b[0], b[1], b[2], b[3]];
}
