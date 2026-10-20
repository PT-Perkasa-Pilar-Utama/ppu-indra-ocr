// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama
// Ported from datalab-to/indra indra_ocr/pipeline.py (Apache-2.0, Copyright Datalab).

import type { InferenceSession } from "onnxruntime-common";

import type { FloatImage, Gray, ImageOps } from "./image.js";
import type { NgramLM } from "./lm.js";
import type { MathReader } from "./math.js";
import type { Box } from "./rec-preproc.js";
import { cropLine, isDarkLine, widthBatches } from "./rec-preproc.js";
import type { LmUse, RecGroup } from "./recognize.js";
import { GROUPS, lineGroups, readMath, recognizeBatch } from "./recognize.js";
import type { IndraMeta, IndraSessions, Lazy, LmParams, RecLine, TensorCtor } from "./types.js";
import { item } from "./util.js";
import { Vocab } from "./vocab.js";

// Beam search with a language model reads a recognizer's lines under the gate
// confidence; meta.json "lm_params" overrides these per group.
const LM_GATE = 0.9;
const LM_ALPHA = 0.5;
const LM_BETA = 1.5;
const LM_OUTPUTS = ["blank", "topv", "topi"];

/** A script group's recognizer: loaded on the first line routed to it. */
type GroupModel = { session: Lazy<InferenceSession>; vocab: Vocab };

/** A loader that runs once; a failed load is retried on the next call. */
function once<T>(load: Lazy<T>): Lazy<T> {
  let p: Promise<T> | undefined;
  return async () => {
    p ??= load();
    try {
      return await p;
    } catch (e) {
      p = undefined;
      throw e;
    }
  };
}

/**
 * The line recognizers of a model set: rec.onnx, and with script.onnx one
 * recognizer per script group (rec_<group>.onnx with meta["vocab_<group>"]),
 * each with its language model when there is one.
 */
export class Recognizers {
  public readonly vocab: Vocab;
  public readonly height: number;
  /** rec.onnx reads a window this many line heights tall. */
  public readonly window: number;

  private readonly groups = new Map<string, GroupModel>();
  private readonly lms = new Map<string, Lazy<NgramLM>>();
  private readonly lmParams: Record<string, LmParams>;
  private readonly scriptGroups: readonly string[];
  private readonly script: InferenceSession | undefined;

  public constructor(
    private readonly Tensor: TensorCtor,
    private readonly sessions: IndraSessions,
    meta: IndraMeta,
    private readonly ops: ImageOps,
    private readonly useLm: boolean,
    private readonly math: MathReader | undefined
  ) {
    this.vocab = Vocab.fromJson(meta.vocab);
    this.height = meta.rec_height;
    this.window = meta.rec_window ?? 1;
    // Groups in meta.json's key order.
    for (const key of Object.keys(meta)) {
      if (!key.startsWith("vocab_")) continue;
      const name = key.slice("vocab_".length);
      const load = sessions.groups?.[name];
      const json = meta[`vocab_${name}`];
      if (load && json) this.groups.set(name, { session: once(load), vocab: Vocab.fromJson(json) });
    }
    for (const [name, load] of Object.entries(sessions.lms ?? {})) this.lms.set(name, once(load));
    this.lmParams = meta.lm_params ?? {};
    this.scriptGroups = meta.script_groups ?? GROUPS;
    this.script = this.groups.size ? sessions.script : undefined;
  }

  /** Every recognizer's vocabulary: the main one first, then each group's. */
  public get vocabs(): Vocab[] {
    return [this.vocab, ...[...this.groups.values()].map((g) => g.vocab)];
  }

  /** The language model use of one recognizer, when its file is there and the model emits top-k outputs. */
  private lmUse(name: string, sess: InferenceSession): LmUse | undefined {
    const lm = this.lms.get(name);
    if (!this.useLm || !lm || !LM_OUTPUTS.every((o) => sess.outputNames.includes(o))) {
      return undefined;
    }
    const p = this.lmParams[name] ?? {};
    return { lm, alpha: p.alpha ?? LM_ALPHA, beta: p.beta ?? LM_BETA, gate: p.gate ?? LM_GATE };
  }

  private base(sel: number[]): RecGroup {
    const sess = this.sessions.rec;
    return { sess, vocab: this.vocab, sel, asChars: false, lm: this.lmUse("base", sess) };
  }

  private async group(name: string, sel: number[]): Promise<RecGroup> {
    const g = this.groups.get(name);
    if (!g) throw new Error(`no recognizer for script group "${name}"`);
    const sess = await g.session();
    return { sess, vocab: g.vocab, sel, asChars: true, lm: this.lmUse(name, sess) };
  }

  /**
   * Which recognizer reads which lines. With script.onnx, each line goes to
   * the recognizer of its script group, and a line the classifier is unsure
   * of goes to the main recognizer too. A vertical page's lines are CJK
   * columns, characters on their side: all go to the CJK recognizer.
   */
  private async plan(
    crops: readonly FloatImage[],
    vertical: boolean
  ): Promise<(() => Promise<RecGroup>)[]> {
    const all = crops.map((_, i) => i);
    if (vertical && this.groups.has("cjk")) return [() => this.group("cjk", all)];
    if (!this.script || !crops.length) return [async () => this.base(all)];
    const { names, sure } = await lineGroups(
      this.Tensor,
      this.script,
      crops,
      this.scriptGroups,
      new Set(this.groups.keys())
    );
    const plan = [async () => this.base(all.filter((j) => names[j] === "base" || !sure[j]))];
    for (const name of this.groups.keys()) {
      const sel = all.filter((j) => names[j] === name);
      if (sel.length) plan.push(() => this.group(name, sel));
    }
    return plan;
  }

  /**
   * Text, confidence and character positions per line; a line read twice
   * keeps the more confident reading. A line under MIN_CONF comes back empty.
   */
  public async read(
    gray: Gray,
    boxes: readonly Box[],
    batch: number,
    vertical: boolean
  ): Promise<RecLine[]> {
    // Light-on-dark is decided once per line box, for both crops.
    const dark = this.window > 1 ? boxes.map((b) => isDarkLine(gray, b)) : undefined;
    const crop = (b: Box, i: number, window: number): FloatImage =>
      cropLine(gray, b, this.height, this.ops, 4096, window, dark?.[i]);
    const crops = boxes.map((b, i) => crop(b, i, 1));
    // rec.onnx may read a tall window: the line and the page around it.
    const tall = this.window > 1 ? boxes.map((b, i) => crop(b, i, this.window)) : crops;
    const out: RecLine[] = crops.map(() => ({ text: "", conf: 0, toks: [] }));
    const bestConf = crops.map(() => -1);
    for (const next of await this.plan(crops, vertical)) {
      const g = await next();
      const src = g.sess === this.sessions.rec ? tall : crops;
      for (const part of widthBatches(
        g.sel.map((j) => item(crops, j).width),
        batch
      )) {
        const idx = part.map((i) => item(g.sel, i));
        const lines = await recognizeBatch(
          this.Tensor,
          g,
          idx.map((j) => item(src, j)),
          idx.map((j) => item(crops, j)),
          idx.map((j) => item(boxes, j))
        );
        for (const [k, j] of idx.entries()) {
          const line = item(lines, k);
          if (line.conf <= item(bestConf, j)) continue;
          bestConf[j] = line.conf;
          out[j] = line;
        }
      }
    }
    if (this.math && crops.length) await readMath(this.math, gray, boxes, crops, out);
    return out;
  }
}
