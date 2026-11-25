// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama
// Ported from datalab-to/indra indra_ocr/pipeline.py (Apache-2.0, Copyright Datalab).

import type { InferenceSession, Tensor } from "onnxruntime-common";

import type { LmInputs } from "./ctc.js";
import { lmInputs, peakProb } from "./ctc.js";
import type { FloatImage, Gray, ImageOps } from "./image.js";
import type { NgramLM } from "./lm.js";
import { beamSearch } from "./lm.js";
import type { SpanImgTagger } from "./math-span-img.js";
import { pageX, splice } from "./math-span-img.js";
import { MATH_MIN_CONF, hasMath, latinLine } from "./math-text.js";
import type { MathReader } from "./math.js";
import { isAlpha, pyStrip } from "./py-unicode.js";
import { meanF32 } from "./numeric.js";
import type { Box } from "./rec-preproc.js";
import { PAD_X, cropLine, padBatch, widthBatches } from "./rec-preproc.js";
import type { Lazy, RecLine, TensorCtor, Tok } from "./types.js";
import { floats, ints, item, output } from "./util.js";
import type { Vocab } from "./vocab.js";

/**
 * A recognized line under this confidence (mean CTC peak probability) gets no
 * text: on empty form fields, checkboxes and rules the recognizer reads "-",
 * "I" or "1" at low confidence. The line's box stays.
 */
export const MIN_CONF = 0.5;

/**
 * Language model re-reading: lines a recognizer reads with confidence in
 * [MIN_CONF, gate) are read again by a CTC beam search that scores readings
 * with the n-gram model (alpha weighs it, beta rewards each token).
 */
export type LmUse = { lm: Lazy<NgramLM>; alpha: number; beta: number; gate: number };

/** Lines that go to one recognizer, and whether its tokens are characters (group) or ids (main). */
export type RecGroup = {
  sess: InferenceSession;
  vocab: Vocab;
  sel: number[];
  asChars: boolean;
  lm?: LmUse;
  /** The recognizer reads lines widened by this factor (meta.json `rec_xscale_<group>`). */
  xscale: number;
};

export const LM_BEAM = 8;
/** A class under this probability at a step is not tried by the beam search. */
export const LM_CAND_FLOOR = 3e-3;
const OUT_GREEDY = ["ids", "conf", "lengths"];
const OUT_LM = ["blank", "topv", "topi"];
/** A newer recognizer ends at its logits: confidence and LM inputs come from ctc.ts. */
const OUT_LOGITS = ["ids", "lengths", "logits"];

/** Whether a recognizer ends at its logits, not at the CTC outputs. */
export function hasLogits(sess: InferenceSession): boolean {
  return sess.outputNames.includes("logits");
}

/** Whether a recognizer gives what the beam search reads: top-k outputs or logits. */
export function lmReady(sess: InferenceSession): boolean {
  return OUT_LM.every((o) => sess.outputNames.includes(o)) || hasLogits(sess);
}

/**
 * One recognizer batch: text, confidence and character positions per crop.
 * `crops` are the recognizer inputs (a tall window for rec.onnx when its
 * meta says so), already widened by the group's x scale; `bands` are the
 * line-band crops that map CTC steps back to page x, as the reference does
 * for both.
 */
export async function recognizeBatch(
  Tensor: TensorCtor,
  { sess, vocab, asChars, lm, xscale }: RecGroup,
  crops: readonly FloatImage[],
  bands: readonly FloatImage[],
  boxes: readonly Box[]
): Promise<RecLine[]> {
  const { data, dims, widths } = padBatch(crops);
  const logitsOut = hasLogits(sess);
  let fetch = lm ? [...OUT_GREEDY, ...OUT_LM] : OUT_GREEDY;
  if (logitsOut) fetch = OUT_LOGITS;
  const res = await sess.run(
    {
      x: new Tensor("float32", data, dims),
      widths: new Tensor("int64", widths, [crops.length]),
    },
    fetch
  );
  const ids = output(res, "ids");
  const best = ints(ids);
  const lengths = ints(output(res, "lengths"));
  const steps = ids.dims[1] ?? 0;
  const logits = logitsOut ? output(res, "logits") : undefined;
  const v = logits?.dims[2] ?? 0;
  const pmax = logits ? undefined : floats(output(res, "conf"));
  const stride = dims[3] / Math.max(steps, 1);
  const out: RecLine[] = [];
  for (const [k, band] of bands.entries()) {
    const n = item(lengths, k);
    const seq = best.slice(k * steps, k * steps + n);
    let kept: number[] = [];
    for (let t = 0; t < n; t++) {
      if (seq[t] !== 0 && (t === 0 || seq[t] !== seq[t - 1])) kept.push(t);
    }
    let keptIds = kept.map((t) => item(seq, t));
    const peaks = logits
      ? peakProb(
          floats(logits),
          v,
          kept.map((t) => k * steps + t)
        )
      : kept.map((t) => pmax?.[k * steps + t] ?? 0);
    const conf = kept.length ? meanF32(peaks) : 0;
    if (lm && MIN_CONF <= conf && conf < lm.gate) {
      const input = logits
        ? lmInputs(floats(logits), k * steps * v, n, v, LM_CAND_FLOOR)
        : topK(res, k, steps, n);
      const read = beamSearch(
        input.blank,
        input.topv,
        input.topi,
        n,
        input.k,
        await lm.lm(),
        lm.alpha,
        lm.beta,
        LM_BEAM,
        0.999,
        LM_CAND_FLOOR
      );
      keptIds = read.ids;
      kept = read.steps;
    }
    // Page x of each kept CTC step, through the band crop's padding and scale.
    const [x0, y0, x1, y1] = item(boxes, k);
    const h = Math.max(1, y1 - y0);
    const ix0 = Math.floor(x0 - PAD_X * h);
    const f = (Math.ceil(x1 + PAD_X * h) - ix0) / (band.width * xscale);
    // A group recognizer's ids mean nothing in the main vocab: its tokens
    // are characters, like the text-layer tokens.
    const toks = kept.map((t, i): Tok => {
      const x = ix0 + (t + 0.5) * stride * f;
      const id = item(keptIds, i);
      return asChars ? [item(vocab.chars, id - 1), x] : [id, x];
    });
    out.push(
      conf >= MIN_CONF ? { text: vocab.decode(keptIds), conf, toks } : { text: "", conf, toks: [] }
    );
  }
  return out;
}

/** Beam search inputs of one line, from the logits (ctc.ts) or the older graphs' top-k outputs. */
type BeamInputs = Omit<LmInputs, "topi"> & { topi: Int32Array | BigInt64Array };

/** The older graphs' top-k outputs of line `k` (steps [0, n)). */
function topK(
  res: InferenceSession.OnnxValueMapType,
  k: number,
  steps: number,
  n: number
): BeamInputs {
  const topv = output(res, "topv");
  const K = topv.dims[2] ?? 0;
  return {
    blank: floats(output(res, "blank")).subarray(k * steps, k * steps + n),
    topv: floats(topv).subarray(k * steps * K, (k * steps + n) * K),
    topi: topIds(output(res, "topi")).subarray(k * steps * K, (k * steps + n) * K),
    k: K,
  };
}

/** An int64 tensor's data (topi): a BigInt64Array, as onnxruntime returns it. */
function topIds(t: Tensor): BigInt64Array {
  if (t.type !== "int64") throw new Error(`expected an int64 tensor, got ${t.type}`);
  // SAFETY: int64 tensors hold a BigInt64Array (type checked above).
  return t.data as BigInt64Array;
}

/** Recognizer groups in script.onnx's class order, unless meta.json names them. */
export const GROUPS = ["base", "cjk", "rtl", "other"] as const;

/**
 * A line goes to one recognizer when the classifier is surer than this;
 * otherwise base reads it as well and the more confident reading wins.
 */
export const ROUTE_SURE = 0.95;

/** Per line crop: the script group whose recognizer reads it, and whether the classifier is sure. */
export type LineGroups = { names: string[]; sure: boolean[] };

/**
 * The group of each line crop (script.onnx), and whether the classifier is
 * sure of it (probability over ROUTE_SURE). An older script.onnx has one
 * output, the chance that the line holds Chinese characters. A group without
 * its recognizer (`known`) goes to base.
 */
export async function lineGroups(
  Tensor: TensorCtor,
  script: InferenceSession,
  crops: readonly FloatImage[],
  order: readonly string[],
  known: ReadonlySet<string>,
  batch = 64
): Promise<LineGroups> {
  const names = crops.map(() => "base");
  const sure = crops.map(() => true);
  for (const idx of widthBatches(
    crops.map((c) => c.width),
    batch
  )) {
    const { data, dims, widths } = padBatch(idx.map((j) => item(crops, j)));
    const res = await script.run({
      x: new Tensor("float32", data, dims),
      widths: new Tensor("int64", widths, [idx.length]),
    });
    const out = output(res, "p");
    const p = floats(out);
    const c = out.dims.length === 1 ? 1 : (out.dims[1] ?? 1);
    for (const [k, j] of idx.entries()) {
      let g: string;
      let pk: number;
      if (out.dims.length === 1) {
        const v = p[k] ?? 0;
        g = v > 0.5 ? "cjk" : "base";
        pk = Math.max(v, 1 - v);
      } else {
        let a = 0;
        for (let i = 1; i < c; i++) if ((p[k * c + i] ?? 0) > (p[k * c + a] ?? 0)) a = i;
        g = order[a] ?? "base";
        pk = p[k * c + a] ?? 0;
      }
      names[j] = known.has(g) ? g : "base";
      sure[j] = pk > ROUTE_SURE;
    }
  }
  return { names, sure };
}

/**
 * Lines the math classifier flags are read by the math reader. MathRec reads
 * a line whose reading is mostly Latin (a formula line may have gone to a
 * group recognizer: Greek letters to rec_world), or one whose letters its own
 * vocabulary writes; any other line keeps its reading. A reading replaces the
 * line only when the accept rules take it, with the recognizer's accents in
 * its prose; one token holds the whole line, so it is never split. The other
 * lines get their inline formulas (readSpanImg) unless the page is vertical.
 */
export async function readMath(
  math: MathReader,
  gray: Gray,
  boxes: readonly Box[],
  crops: readonly FloatImage[],
  out: RecLine[],
  vertical: boolean,
  ops: ImageOps,
  vocab: Vocab
): Promise<void> {
  const cand = out.flatMap((r, j) => (latinLine(r.text) || math.writes(r.text) ? [j] : []));
  const { flags, probs } = cand.length
    ? await math.flags(cand.map((j) => item(crops, j)))
    : { flags: [], probs: [] };
  let idx = cand.filter((_, n) => flags[n]);
  if (idx.length && math.hasGate) {
    const keep = math.gate(
      probs.filter((_, n) => flags[n]),
      idx.map((j) => item(out, j).text),
      cand.length,
      "ocr"
    );
    idx = idx.filter((_, n) => keep[n]);
  }
  if (idx.length) {
    const readings = await math.read(
      gray,
      idx.map((j) => item(boxes, j))
    );
    for (const [k, j] of idx.entries()) {
      const { text, conf } = item(readings, k);
      const plain = item(out, j);
      const merged = math.accept(text, conf, plain.text, plain.conf);
      if (merged === null) continue;
      const [x0, , x1] = item(boxes, j);
      out[j] = {
        text: merged,
        conf: 1,
        toks: [[merged, Math.fround(x0 + x1) / 2]],
        math: { plain, reading: text, conf, draft: plain.text, rconf: plain.conf },
      };
    }
  }
  if (math.spanImg && !vertical) {
    const seen = new Set(cand);
    const rest = out.flatMap((_, j) => (seen.has(j) ? [] : [j]));
    await readSpanImg(math, math.spanImg, gray, boxes, out, rest, ops, vocab);
  }
}

/**
 * Inline math in the lines `idx` (lines MathRec does not read whole): the
 * image tagger marks math columns, MathRec reads each run's crop, and a
 * confident reading of one formula replaces the recognizer's characters there.
 */
async function readSpanImg(
  math: MathReader,
  tagger: SpanImgTagger,
  gray: Gray,
  boxes: readonly Box[],
  out: RecLine[],
  idx: readonly number[],
  ops: ImageOps,
  vocab: Vocab
): Promise<void> {
  const lines = idx.filter((j) => {
    const r = item(out, j);
    return r.toks.length && Array.from(r.text).some(isAlpha);
  });
  if (!lines.length) return;
  const crops = lines.map((j) => cropLine(gray, item(boxes, j), tagger.height, ops));
  const spans = await tagger.spans(crops);
  const jobs = lines.flatMap((j, n) =>
    item(spans, n).map((xs) => ({ j, x: pageX(item(boxes, j), item(crops, n).width, xs) }))
  );
  if (!jobs.length) return;
  const f = Math.fround;
  const readings = await math.read(
    gray,
    jobs.map(({ j, x: [x0, x1] }): Box => {
      const b = item(boxes, j);
      return [f(x0), b[1], f(x1), b[3]];
    })
  );
  for (const [n, { j, x }] of jobs.entries()) {
    const { text, conf } = item(readings, n);
    const t = pyStrip(text);
    // A run's reading must be one formula: the line keeps its own prose.
    if (
      conf < MATH_MIN_CONF ||
      t.split("$").length - 1 !== 2 ||
      !t.startsWith("$") ||
      !t.endsWith("$")
    ) {
      continue;
    }
    if (!(tagger.acceptPlain || hasMath(t))) continue;
    const line = item(out, j);
    // Main recognizer tokens are vocab ids: characters from here on.
    const chars = line.toks.map((tk): Tok =>
      typeof tk[0] === "string" ? tk : [vocab.decode([tk[0]]) || " ", tk[1]]
    );
    const toks = splice(chars, x[0], x[1], t);
    out[j] = { text: toks.map((tk) => String(tk[0])).join(""), conf: line.conf, toks };
  }
}
