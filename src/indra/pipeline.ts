// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama
// Ported from datalab-to/indra indra_ocr/pipeline.py (Apache-2.0, Copyright Datalab).

import { buildPage, mirrorBox, pageIsRtl, splitParts, verticalOrder } from "./assemble.js";
import type { DetBox } from "./det-post.js";
import { decode, pageToInput } from "./det-post.js";
import type { Gray, ImageOps } from "./image.js";
import type { MathReader } from "./math.js";
import {
  N_FEATS,
  decodeBlocksOrder,
  decodeOrder,
  lineFeatures,
  medianF32,
  orderXYCut,
} from "./order-post.js";
import type { Box } from "./rec-preproc.js";
import { Recognizers } from "./recognizers.js";
import type {
  DetectPass,
  IndraMeta,
  IndraSessions,
  Orientation,
  Page,
  PipelineOptions,
  ReadingOrder,
  RecLine,
  Straightened,
  TensorCtor,
} from "./types.js";
import {
  DESKEW_MIN,
  ORIENT_MIN,
  ORIENT_SIDE,
  VERTICAL,
  estimateSkew,
  orientInput,
  preview,
  rot90,
  rotatePage,
  roundTo,
  unrotateBox,
} from "./upright.js";
import { floats, output } from "./util.js";
import type { Vocab } from "./vocab.js";

const ADAPTIVE_MIN_H = 9.0; // median line height (input px) that triggers a second detection pass
const ADAPTIVE_TARGET_H = 12.0;
const ADAPTIVE_MAX_SIDE = 2048;

/**
 * The OCR pipeline on one grayscale page. Each stage is its own method, so
 * callers can run the whole page or chain stages themselves.
 */
export class IndraPipeline {
  public readonly vocab: Vocab;
  public readonly longSide: number;
  public readonly height: number;

  private readonly furniture: boolean;
  private readonly adaptive: boolean;
  private readonly math: MathReader | undefined;
  private readonly recognizers: Recognizers;

  public constructor(
    private readonly Tensor: TensorCtor,
    private readonly sessions: IndraSessions,
    meta: IndraMeta,
    private readonly ops: ImageOps,
    options: PipelineOptions = {}
  ) {
    this.furniture = options.furniture ?? false;
    this.adaptive = options.adaptive ?? true;
    this.math = options.math;
    this.recognizers = new Recognizers(Tensor, sessions, meta, ops, options.lm ?? true, this.math);
    this.vocab = this.recognizers.vocab;
    this.longSide = meta.det_long_side;
    this.height = meta.rec_height;
  }

  /** Every recognizer's vocabulary: the main one first, then each script group's. */
  public get vocabs(): Vocab[] {
    return this.recognizers.vocabs;
  }

  /** The math reader, when math is on. */
  public get mathReader(): MathReader | undefined {
    return this.math;
  }

  /** The image operations of this pipeline's engine. */
  public get imageOps(): ImageOps {
    return this.ops;
  }

  /**
   * Quarter turns (counter-clockwise) the page content is turned by, or
   * VERTICAL for a page set in vertical columns. 0 without orient.onnx, or
   * when no class is more likely than ORIENT_MIN.
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
   * The page turned upright by `turns` quarter turns (VERTICAL: a quarter
   * turn counter-clockwise, columns become lines), then straightened when its
   * rows tilt by DESKEW_MIN degrees or more. `skew` is the tilt measured
   * before the turn; after a quarter turn the rows were columns, so the tilt
   * is measured again, on `small` (the page preview) turned when given.
   */
  public straighten(
    gray: Gray,
    turns: number,
    skew: number = estimateSkew(gray, this.ops),
    small?: Gray
  ): Straightened {
    // np.rot90(gray, 1) for VERTICAL is three clockwise quarter turns here.
    const cw = turns === VERTICAL ? 3 : turns;
    const image = cw ? rot90(gray, cw) : gray;
    let a = skew;
    if (turns === 1 || turns === 3 || turns === VERTICAL) {
      a = estimateSkew(small ? rot90(small, cw) : image, this.ops);
    }
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
   * Text, confidence and character positions per line. With script.onnx,
   * each line goes to the recognizer of its script group; a line the
   * classifier is unsure of is read by the main recognizer too, and the more
   * confident reading wins. `vertical`: the page is a vertical one read
   * turned (every line to the CJK recognizer). A line under MIN_CONF comes
   * back empty.
   */
  public async recognize(
    gray: Gray,
    boxes: readonly Box[],
    batch = 64,
    vertical = false
  ): Promise<RecLine[]> {
    return this.recognizers.read(gray, boxes, batch, vertical);
  }

  /**
   * OCR of one page image: measure the tilt and the turn on a preview (unless
   * `turn` is false), turn upright and deskew, detect, recognize, order and
   * lay out.
   */
  public async page(gray: Gray, turn = true): Promise<Page> {
    let t0 = performance.now();
    const small = preview(gray, this.ops);
    const a = estimateSkew(small, this.ops);
    // The orientation model reads the page straightened by its row angle (a
    // tilted page could read as turned).
    const { turns } = turn
      ? await this.orient(Math.abs(a) >= DESKEW_MIN ? rotatePage(small, a, this.ops) : small)
      : { turns: 0 };
    const orientMs = performance.now() - t0;
    const t: Record<string, number> = {};
    t0 = performance.now();
    const { image, skew } = this.straighten(gray, turns, a, small);
    t["orient_ms"] = performance.now() - t0;
    t0 = performance.now();
    const boxes = (await this.detect(image)).map(toBox);
    t["det_ms"] = performance.now() - t0;
    t0 = performance.now();
    const vertical = turns === VERTICAL;
    const rec = await this.recognize(image, boxes, 64, vertical);
    t["rec_ms"] = performance.now() - t0;
    const page = await this.assemble(image.width, image.height, boxes, rec, t, vertical);
    page.timings["orient_ms"] = (page.timings["orient_ms"] ?? 0) + orientMs;
    page.rotation = vertical ? 0 : 90 * turns;
    page.skew = roundTo(skew, 2);
    page.vertical = vertical;
    if (skew) {
      // Boxes back on the page as turned, before the deskew.
      const back = (b: Box): Box => unrotateBox(b, skew, image.width, image.height);
      page.lines = page.lines.map((ln) => ({ ...ln, box: back(ln.box) }));
      page.blocks = page.blocks.map((b) => ({ ...b, bbox: back(b.bbox) }));
    }
    if (vertical) {
      // Back on the page as it is: (x, y) -> (W - y, x).
      const w = image.height;
      const back = ([x0, y0, x1, y1]: Box): Box => [w - y1, x0, w - y0, x1];
      page.lines = page.lines.map((ln) => ({ ...ln, box: back(ln.box) }));
      page.blocks = page.blocks.map((b) => ({ ...b, bbox: back(b.bbox) }));
    }
    return page;
  }

  /**
   * Lines -> split at wide gaps -> tidy -> reading order -> blocks and
   * markdown. `rec` holds the text, confidence and tokens of each box. A
   * right-to-left page orders and lays out its lines mirrored, so the right
   * column comes first. `vertical`: a vertical page read turned, ordered by
   * XY-cut.
   */
  public async assemble(
    width: number,
    height: number,
    boxes: readonly Box[],
    rec: readonly RecLine[],
    t: Record<string, number> = {},
    vertical = false
  ): Promise<Page> {
    const t0 = performance.now();
    const pageRtl = pageIsRtl(rec);
    const parts = splitParts(this.vocab, boxes, rec, pageRtl);
    let ro = await this.readingOrder(
      parts.map((p) => (pageRtl ? mirrorBox(p.box, width) : p.box)),
      parts.map((p) => p.text),
      width,
      height
    );
    if (vertical) {
      ro = verticalOrder(
        parts.map((p) => p.box),
        ro
      );
    }
    t["order_ms"] = performance.now() - t0;
    return buildPage(
      this.vocab,
      parts,
      ro,
      vertical ? "order=xycut" : "order=model",
      this.furniture,
      t,
      pageRtl,
      width
    );
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
}

function toBox(b: DetBox): Box {
  return [b[0], b[1], b[2], b[3]];
}
