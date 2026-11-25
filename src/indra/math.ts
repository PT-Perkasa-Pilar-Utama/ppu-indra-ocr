// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama
// Ported from datalab-to/indra indra_ocr/math.py (Apache-2.0, Copyright Datalab).

import type { InferenceSession, Tensor } from "onnxruntime-common";

import type { FloatImage, Gray, ImageOps } from "./image.js";
import { declared, isCommand, takeRows, topTokens } from "./math-decoder.js";
import { meanF64 } from "./numeric.js";
import type { Box } from "./rec-preproc.js";
import { cropLine, padBatch, widthBatches } from "./rec-preproc.js";
import type { MathAcceptHead, MathGateHead } from "./math-text.js";
import { MATH_MIN_CONF, mathAccept, mathChars, mathGate, mathWrites } from "./math-text.js";
import type { SpanImgTagger } from "./math-span-img.js";
import type { TensorCtor } from "./types.js";
import { floats, item, output } from "./util.js";

const MATH_MIN = 0.5; // classifier probability above which a line is read as math
export { MATH_MIN_CONF };
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

/** The optional parts of math: the learned accept head and read gate, and the inline formula tagger. */
export type MathHeads = { accept?: MathAcceptHead; gate?: MathGateHead; spanImg?: SpanImgTagger };

/** The classifier's verdict per line crop: holds math, and its probability (for the read gate). */
export type MathFlags = { flags: boolean[]; probs: number[] };

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

/**
 * The decoder's caches, in the layout of its step graph: one tensor per layer
 * ([B, heads, P, d]) for math_dec_step_kv.onnx, or all layers stacked
 * ([layers, B, heads, P, d]) for the older math_dec_step.onnx.
 */
type Caches = { selfK: Tensor[]; selfV: Tensor[]; crossK: Tensor[]; crossV: Tensor[] };

/**
 * Math lines: the classifier finds lines that hold math, and an attention
 * decoder reads a taller crop of each one token per step, with its keys and
 * values cached.
 */
export class MathReader {
  /** The tagger of inline formulas in lines MathRec does not read whole (math_span_img.onnx). */
  public readonly spanImg: SpanImgTagger | undefined;
  private readonly itos: string[];
  private readonly headDim: number;
  /** The step graph takes one cache per layer (math_dec_step_kv.onnx). */
  private readonly splitKv: boolean;
  /** Single-character tokens: the letters MathRec can write. */
  private readonly chars: ReadonlySet<string>;
  private readonly acceptHead: MathAcceptHead | null;
  private readonly gateHead: MathGateHead | null;

  public constructor(
    private readonly Tensor: TensorCtor,
    private readonly sessions: MathSessions,
    private readonly meta: MathMeta,
    private readonly ops: ImageOps,
    heads: MathHeads = {}
  ) {
    this.itos = ["", "", "", "", ...meta.tokens];
    this.headDim = meta.config.channels / meta.config.dec_heads;
    this.splitKv = sessions.step.inputNames.includes("self_k0");
    this.chars = mathChars(meta.tokens);
    this.acceptHead = heads.accept ?? null;
    this.gateHead = heads.gate ?? null;
    this.spanImg = heads.spanImg;
  }

  /** The math crops' line height (MathRec reads a taller crop than the recognizers). */
  public get height(): number {
    return this.meta.config.height;
  }

  /** Whether a learned read gate decides which flagged lines MathRec reads. */
  public get hasGate(): boolean {
    return this.gateHead !== null;
  }

  /** Whether a learned accept head decides which readings replace a line (and knows table lines). */
  public get hasAcceptHead(): boolean {
    return this.acceptHead !== null;
  }

  /**
   * The line's text when MathRec's reading replaces the plain reading (the
   * recognizer's or the text layer's, `draft`, read with confidence `rconf`),
   * else null; `inTable`: the layout puts the line in a table.
   */
  public accept(
    text: string,
    conf: number,
    draft: string,
    rconf = 1.0,
    inTable = 0.0
  ): string | null {
    return mathAccept(this.acceptHead, text, conf, draft, rconf, inTable);
  }

  /** Which flagged lines (classifier probabilities, drafts) MathRec reads; `nCand`: lines the classifier saw. */
  public gate(
    probs: readonly number[],
    drafts: readonly string[],
    nCand: number,
    source: "ocr" | "text" | "hybrid"
  ): boolean[] {
    return mathGate(this.gateHead, probs, drafts, nCand, source);
  }

  /** Whether MathRec's vocabulary writes this line's letters (a MathRec with another script's letters may read it). */
  public writes(text: string): boolean {
    return mathWrites(this.chars, text);
  }

  /** Per line crop (the recognizer's crop): does it hold math, and the classifier's probability. */
  public async flags(crops: readonly FloatImage[], batch = 64): Promise<MathFlags> {
    const flags = crops.map(() => false);
    const probs = crops.map(() => 0);
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
      for (const [k, j] of idx.entries()) {
        probs[j] = p[k] ?? 0;
        flags[j] = (p[k] ?? 0) > MATH_MIN;
      }
    }
    return { flags, probs };
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

  /** The caches at the first step, from the init graph's cross keys and values ([layers, B, heads, T, d]). */
  private caches(ck: Tensor, cv: Tensor, B: number): Caches {
    const { dec_layers: layers, dec_heads: heads } = this.meta.config;
    if (!this.splitKv) {
      const empty = new this.Tensor("float32", new Float32Array(0), [
        layers,
        B,
        heads,
        0,
        this.headDim,
      ]);
      return { selfK: [empty], selfV: [empty], crossK: [ck], crossV: [cv] };
    }
    const empty = new this.Tensor("float32", new Float32Array(0), [B, heads, 0, this.headDim]);
    const all = Array.from({ length: layers }, (_, i) => i);
    const layer = (t: Tensor, i: number): Tensor => {
      const [, b = 0, h = 0, n = 0, d = 0] = t.dims;
      const size = b * h * n * d;
      return new this.Tensor("float32", floats(t).subarray(i * size, (i + 1) * size), [b, h, n, d]);
    };
    return {
      selfK: all.map(() => empty),
      selfV: all.map(() => empty),
      crossK: all.map((i) => layer(ck, i)),
      crossV: all.map((i) => layer(cv, i)),
    };
  }

  /** The caches of the decoder rows `rows` only. */
  private take(c: Caches, rows: readonly number[]): Caches {
    const axis = this.splitKv ? 0 : 1;
    const t = (xs: Tensor[]): Tensor[] => xs.map((x) => takeRows(this.Tensor, x, rows, axis));
    return { selfK: t(c.selfK), selfV: t(c.selfV), crossK: t(c.crossK), crossV: t(c.crossV) };
  }

  /** One decoder step: logits [B, V] and the caches with this token's keys and values appended. */
  private async step(
    tok: BigInt64Array,
    t: number,
    c: Caches,
    mask: Tensor
  ): Promise<{ logits: Tensor; c: Caches }> {
    const { step } = this.sessions;
    const base = {
      token: new this.Tensor("int64", tok, [tok.length]),
      pos: new this.Tensor("int64", BigInt64Array.of(BigInt(t)), []),
      mask,
    };
    if (!this.splitKv) {
      const res = await step.run(
        declared(step, {
          ...base,
          self_k: item(c.selfK, 0),
          self_v: item(c.selfV, 0),
          cross_k: item(c.crossK, 0),
          cross_v: item(c.crossV, 0),
        })
      );
      const next = { ...c, selfK: [output(res, "new_k")], selfV: [output(res, "new_v")] };
      return { logits: output(res, "logits"), c: next };
    }
    const caches = Object.fromEntries(
      c.selfK.flatMap((_, i) => [
        [`self_k${i}`, item(c.selfK, i)],
        [`self_v${i}`, item(c.selfV, i)],
        [`cross_k${i}`, item(c.crossK, i)],
        [`cross_v${i}`, item(c.crossV, i)],
      ])
    );
    const res = await step.run(declared(step, { ...base, ...caches }));
    const next = {
      ...c,
      selfK: c.selfK.map((_, i) => output(res, `new_k${i}`)),
      selfV: c.selfV.map((_, i) => output(res, `new_v${i}`)),
    };
    return { logits: output(res, "logits"), c: next };
  }

  private async decodeBatch(crops: readonly FloatImage[]): Promise<MathReading[]> {
    const { enc, init } = this.sessions;
    const { max_tokens: maxTokens } = this.meta.config;
    const B = crops.length;
    const { data, dims, widths } = padBatch(crops);
    const encoded = await enc.run({
      x: new this.Tensor("float32", data, dims),
      widths: new this.Tensor("int64", widths, [B]),
    });
    let mask = output(encoded, "mask");
    const cross = await init.run(declared(init, { memory: output(encoded, "memory"), mask }));
    let c = this.caches(output(cross, "cross_k"), output(cross, "cross_v"), B);
    let tok = new BigInt64Array(B).fill(BigInt(BOS));
    const seqs: number[][] = Array.from({ length: B }, () => []);
    const probs: number[][] = Array.from({ length: B }, () => []);
    let rows = Array.from({ length: B }, (_, i) => i); // the batch's lines still decoding, in step order
    const inMath = Array.from({ length: B }, () => false); // inside a $...$ span
    const textLike = Array.from({ length: B }, () => false); // the reading already fails acceptMath
    for (let t = 0; t < maxTokens - 1; t++) {
      const step = await this.step(tok, t, c, mask);
      c = step.c;
      const logits = step.logits;
      const ids = topTokens(floats(logits), rows.length, logits.dims[1] ?? 0);
      tok = BigInt64Array.from(ids.map((p) => BigInt(p.id)));
      const live = rows.map(() => true);
      for (const [r, b] of rows.entries()) {
        const { id, top } = item(ids, r);
        const pb = item(probs, b);
        pb.push(top);
        if (id === EOS) {
          live[r] = false;
        } else if (pb.length >= 8 && meanF64(pb) < MATH_MIN_CONF - EARLY_STOP_MARGIN) {
          live[r] = false; // rejected: keeps a low mean probability
        } else if (id >= N_SPECIAL) {
          item(seqs, b).push(id);
          const tk = item(this.itos, id);
          if (tk === "$") {
            inMath[b] = !inMath[b];
          } else if (!inMath[b] && isCommand(tk)) {
            // A LaTeX command outside the $...$ spans: acceptMath would reject it, stop here.
            textLike[b] = true;
            live[r] = false;
          }
        }
      }
      if (!live.some(Boolean)) break;
      if (!live.every(Boolean)) {
        // Finished lines leave the batch: a step costs about its number of
        // lines, and the longest line sets the step count.
        const keep = live.flatMap((l, r) => (l ? [r] : []));
        rows = keep.map((r) => item(rows, r));
        tok = BigInt64Array.from(keep.map((r) => tok[r] ?? 0n));
        mask = takeRows(this.Tensor, mask, keep, 0);
        c = this.take(c, keep);
      }
    }
    return seqs.map((seq, b) => {
      // A reading that wrote LaTeX outside its spans is not math.
      if (item(textLike, b)) return { text: "", conf: 0 };
      const pb = item(probs, b);
      return {
        text: detokenize(seq.map((i) => item(this.itos, i))),
        conf: pb.length ? meanF64(pb) : 0,
      };
    });
  }
}
