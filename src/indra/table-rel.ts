// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama
// Ported from datalab-to/indra indra_ocr/table_rel.py (Apache-2.0, Copyright Datalab).
//
// Table structure from word relations (tsr_rel.onnx).
//
// A table block's lines are cut into words (at the recognizer's space tokens).
// Each word gets a feature vector (`wordFeatures`): its box relative to the
// table, simple text classes, the gaps to its neighbours on its line, and the ink
// in thin strips just outside its box (ruling lines). A small transformer over
// the words predicts, for every pair, logits for "same cell", "same row" and
// "same column", and per word logits for "header" and "outside the table".
//
// With tsr_split.onnx (the TableNet separator model) and a tsr_rel.onnx that takes
// more than N_FEAT_BASE features, each word also gets `splitFeatures`: how
// strongly the model's row / column separator maps run next to and through the
// word, and its header map. tsr_rel can follow or overrule them. tsr_refine.onnx
// refines the word tags (header, outside the table) of tsr_rel.
//
// The steps live in table-features.ts (words and their features), table-split.ts
// (tsr_split), table-text.ts (word ids and text features), and table-decode.ts
// (cells, rows, columns and HTML); `TableReader` runs them.

import type { InferenceSession, Tensor } from "onnxruntime-common";

import type { Gray, ImageOps } from "./image.js";
import { medianF32 } from "./order-post.js";
import type { ColSpans } from "./table-decode.js";
import { decodeTable, structHtml } from "./table-decode.js";
import type { TableLine, Word, WordText } from "./table-features.js";
import {
  N_FEAT_BASE,
  boxesF32,
  normBoxes,
  unionBox,
  wordFeatures,
  wordsOf,
} from "./table-features.js";
import { N_SPLIT, splitFeatures, splitMaps } from "./table-split.js";
import { N_REFINE_TEXT, refineTextFeatures, wordIds } from "./table-text.js";
import type { TensorCtor } from "./types.js";
import { floats, item, output } from "./util.js";

const f = Math.fround;

/** The feature count the port builds for tsr_rel.onnx: the word features and the split features. */
export const N_FEAT: number = N_FEAT_BASE + N_SPLIT;

/** The key-value pairs of an ONNX model's `metadata_props` (ModelProto field 14), read from the file's bytes. */
export function onnxMetadata(bytes: Uint8Array): Map<string, string> {
  const text = new TextDecoder();
  const out = new Map<string, string>();
  /** The varint at `from` and the offset after it. */
  const varint = (from: number): [number, number] => {
    let v = 0;
    for (let p = from, shift = 0; p < bytes.length; p++, shift += 7) {
      const b = item(bytes, p);
      v += (b & 0x7f) * 2 ** shift;
      if (!(b & 0x80)) return [v, p + 1];
    }
    throw new RangeError("onnxMetadata: truncated varint");
  };
  /** Visit the length-delimited fields of the message in bytes[from, to). */
  const fields = (
    from: number,
    to: number,
    visit: (field: number, start: number, end: number) => void
  ): void => {
    let p = from;
    while (p < to) {
      const [tag, afterTag] = varint(p);
      p = afterTag;
      const wire = tag & 7;
      if (wire === 0) {
        p = varint(p)[1];
      } else if (wire === 1 || wire === 5) {
        p += wire === 1 ? 8 : 4;
      } else if (wire === 2) {
        const [len, start] = varint(p);
        if (start + len > to) throw new RangeError("onnxMetadata: a field runs past the end");
        visit(Math.floor(tag / 8), start, start + len);
        p = start + len;
      } else {
        throw new RangeError(`onnxMetadata: wire type ${wire}`);
      }
    }
  };
  fields(0, bytes.length, (field, start, end) => {
    if (field !== 14) return;
    const entry = { key: "", value: "" };
    fields(start, end, (inner, s, e) => {
      if (inner === 1) entry.key = text.decode(bytes.subarray(s, e));
      if (inner === 2) entry.value = text.decode(bytes.subarray(s, e));
    });
    out.set(entry.key, entry.value);
  });
  return out;
}

/** The feature count of a tsr_rel model and its word-id buckets (0 when it has no word-text input). */
export type TsrRelInfo = { nFeat: number; nVocab: number };

/**
 * What the reference reads from the session: the model's "feat" width, and the
 * "n_vocab" metadata when the model has a "wid" input. onnxruntime-node does not
 * expose model metadata, so the model file's bytes are read for it.
 */
export function tsrRelInfo(rel: InferenceSession, modelBytes: Uint8Array): TsrRelInfo {
  const meta: readonly InferenceSession.ValueMetadata[] = rel.inputMetadata ?? []; // absent on some runtimes
  const feat = meta.find((m) => m.name === "feat");
  const nFeat = feat?.isTensor ? feat.shape.at(-1) : undefined;
  if (typeof nFeat !== "number") throw new Error("tsr_rel.onnx: the feat input has no fixed width");
  const nVocab =
    rel.inputNames.length < 3 ? 0 : Number(onnxMetadata(modelBytes).get("n_vocab") ?? 0);
  return { nFeat, nVocab };
}

/** The tsr_rel session with what `tsrRelInfo` read from it, and the optional tsr_split and tsr_refine sessions. */
export type TsrSessions = TsrRelInfo & {
  rel: InferenceSession;
  split?: InferenceSession;
  refine?: InferenceSession;
};

/** One table's model inputs. */
type ModelInputs = { feat: Tensor; box: Tensor; texts: string[] };

/** Reads tables with tsr_rel.onnx; `colSpans` is meta.json's "tsr_rel_span" (default "geo"). */
export class TableReader {
  public constructor(
    private readonly Tensor: TensorCtor,
    private readonly sessions: TsrSessions,
    private readonly ops: ImageOps,
    private readonly colSpans: ColSpans = "geo"
  ) {}

  /**
   * HTML for one table block, or null when it has fewer than two words or two
   * rows and columns. `lines`: the block's lines on `gray`; `mirror`: a
   * right-to-left page, whose table is read mirrored, as a left-to-right one.
   */
  public async read(
    gray: Gray,
    lines: readonly TableLine[],
    text: WordText,
    mirror = false
  ): Promise<string | null> {
    const words = wordsOf(lines, text);
    if (words.length < 2) return null;
    const page = mirror ? mirrorWords(words, gray) : gray;
    const boxes = boxesF32(words);
    const inputs = await this.inputs(words, page, boxes);
    const { rel, refine } = this.sessions;
    const { nVocab } = this.sessions;
    const { feat, box, texts } = inputs;
    const ids = nVocab ? BigInt64Array.from(wordIds(texts, nVocab), BigInt) : undefined;
    const wid = ids && new this.Tensor("int64", ids, [1, words.length]);
    const res = await rel.run(wid ? { feat, box, wid } : { feat, box });
    const pair = output(res, "pair");
    let tag = output(res, "tag");
    if (refine) {
      // tsr_refine.onnx: a small relational model over the tsr_rel logits. It changes
      // only the word tags (header, outside the table). Rows, columns and cells
      // keep the tsr_rel relations.
      const tfeat = refineTextFeatures(inputs.texts);
      const out = await refine.run({
        pair,
        tag,
        box: inputs.box,
        feat: inputs.feat,
        tfeat: new this.Tensor("float32", tfeat, [1, words.length, N_REFINE_TEXT]),
      });
      tag = output(out, "tag_out");
    }
    const struct = decodeTable(floats(pair), floats(tag), boxes, this.colSpans, inputs.texts);
    return struct ? structHtml(struct, words) : null;
  }

  /** The feature and box inputs of tsr_rel.onnx: split features only for a model that takes more than the word features. */
  private async inputs(
    words: readonly Word[],
    page: Gray,
    boxes: Float32Array
  ): Promise<ModelInputs> {
    const n = words.length;
    const { nFeat, split } = this.sessions;
    if (nFeat > N_FEAT)
      throw new RangeError(`tsr_rel.onnx takes ${nFeat} features, the port builds ${N_FEAT}`);
    const bbox = unionBox(boxes);
    const base = wordFeatures(words, page, bbox);
    let feat = base;
    let width = N_FEAT_BASE;
    if (nFeat > N_FEAT_BASE) {
      const heights = Array.from({ length: n }, (_, i) =>
        f(item(boxes, i * 4 + 3) - item(boxes, i * 4 + 1))
      );
      const maps = split
        ? await splitMaps(split, this.Tensor, this.ops, page, bbox, medianF32(heights))
        : null;
      const extra = splitFeatures(boxes, maps);
      width = N_FEAT;
      feat = new Float32Array(n * width);
      for (let i = 0; i < n; i++) {
        feat.set(base.subarray(i * N_FEAT_BASE, (i + 1) * N_FEAT_BASE), i * width);
        feat.set(extra.subarray(i * N_SPLIT, (i + 1) * N_SPLIT), i * width + N_FEAT_BASE);
      }
    }
    if (nFeat < width) {
      // a model that takes fewer features
      const cut = new Float32Array(n * nFeat);
      for (let i = 0; i < n; i++) cut.set(feat.subarray(i * width, i * width + nFeat), i * nFeat);
      feat = cut;
      width = nFeat;
    }
    return {
      feat: new this.Tensor("float32", feat, [1, n, width]),
      box: new this.Tensor("float32", normBoxes(boxes), [1, n, 4]),
      texts: words.map((w) => w.text),
    };
  }
}

/**
 * A right-to-left table is read mirrored, as a left-to-right one: the page and
 * the word boxes are flipped before the features (positions, gaps, ink strips,
 * rules), so features and boxes agree. Flips the boxes of `words` and orders
 * each line left to right, in place; returns the flipped page.
 */
function mirrorWords(words: Word[], gray: Gray): Gray {
  const { width: W, height: H } = gray;
  const flipped = new Uint8Array(gray.data.length);
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) flipped[y * W + x] = item(gray.data, y * W + (W - 1 - x));
  for (const w of words) {
    const [x0, y0, x1, y1] = w.box;
    w.box = [W - x1, y0, W - x0, y1];
  }
  const sorted = words.toSorted((a, b) => a.line - b.line || a.box[0] - b.box[0]);
  words.splice(0, words.length, ...sorted);
  const byLine = new Map<number, Word[]>();
  for (const w of words) byLine.set(w.line, [...(byLine.get(w.line) ?? []), w]);
  for (const ws of byLine.values()) {
    for (const [k, w] of ws.entries()) {
      w.pos = k;
      w.n = ws.length;
    }
  }
  return { data: flipped, width: W, height: H };
}
