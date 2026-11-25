// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama
// Ported from datalab-to/indra indra_ocr/pipeline.py (Apache-2.0, Copyright Datalab).

import type { InferenceSession } from "onnxruntime-common";

import type { FloatImage, Gray, ImageOps } from "./image.js";
import type { NgramLM } from "./lm.js";
import type { MathReader } from "./math.js";
import type { Box } from "./rec-preproc.js";
import { cropLine, isDarkLine, padBatch, widen, widthBatches } from "./rec-preproc.js";
import type { LineGroups, LmUse, RecGroup } from "./recognize.js";
import { GROUPS, lineGroups, lmReady, readMath, recognizeBatch } from "./recognize.js";
import type { IndraMeta, IndraSessions, Lazy, LmParams, RecLine, TensorCtor } from "./types.js";
import { floats, item, output } from "./util.js";
import { Vocab } from "./vocab.js";

// Beam search with a language model reads a recognizer's lines under the gate
// confidence; meta.json "lm_params" overrides these per group.
const LM_GATE = 0.9;
const LM_ALPHA = 0.5;
const LM_BETA = 1.5;
/** hw.onnx: a base line goes to rec_hw.onnx at this handwritten probability or more. */
const HW_ROUTE_MIN = 0.5;

/** A script group's recognizer: loaded on the first line routed to it. */
type GroupModel = { session: Lazy<InferenceSession>; vocab: Vocab; xscale: number };

/** Which recognizer reads which lines: the main one, a script group's, or the handwriting one. */
type PlanEntry =
  | { kind: "base"; sel: number[] }
  | { kind: "group"; name: string; sel: number[] }
  | { kind: "hw"; sel: number[] };

/** The crop made for line `j`; every routed line has one. */
function cropOf(crops: ReadonlyMap<number, FloatImage>, j: number): FloatImage {
  const c = crops.get(j);
  if (!c) throw new Error(`no handwriting crop for line ${j}`);
  return c;
}

/** The line-band crops of some boxes, and rec.onnx's tall-window crops of them. */
export type LineCrops = { crops: FloatImage[]; tall: FloatImage[] };

/** One recognizer as the text-layer check reads with it. */
export type Reader = {
  name: string;
  vocab: Vocab;
  session: Lazy<InferenceSession>;
  /** Lines are widened by this factor for it. */
  xscale: number;
  /** rec.onnx, which reads the tall window. */
  main: boolean;
};

/** A loader that runs once; a failed load is retried on the next call. */
export function once<T>(load: Lazy<T>): Lazy<T> {
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
 * The line recognizers of a model set: rec.onnx, with script.onnx one
 * recognizer per script group (rec_<group>.onnx with meta["vocab_<group>"]),
 * and with hw.onnx the handwriting recognizer rec_hw.onnx; each with its
 * language model when there is one.
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
  private readonly hw: { cls: InferenceSession; rec: Lazy<InferenceSession> } | undefined;
  private readonly hwHeight: number;
  private readonly hwWindow: number;
  private readonly hwRouteMin: number;

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
      if (load && json) {
        const xscale = meta[`rec_xscale_${name}`] ?? 1;
        this.groups.set(name, { session: once(load), vocab: Vocab.fromJson(json), xscale });
      }
    }
    for (const [name, load] of Object.entries(sessions.lms ?? {})) this.lms.set(name, once(load));
    this.lmParams = meta.lm_params ?? {};
    this.scriptGroups = meta.script_groups ?? GROUPS;
    this.script = this.groups.size ? sessions.script : undefined;
    this.hw = sessions.hw ? { cls: sessions.hw.cls, rec: once(sessions.hw.rec) } : undefined;
    this.hwHeight = meta.rec_hw_height ?? this.height;
    this.hwWindow = meta.rec_hw_window ?? this.window;
    this.hwRouteMin = meta.hw_route_min ?? HW_ROUTE_MIN;
  }

  /** Every recognizer's vocabulary: the main one first, then each group's. */
  public get vocabs(): Vocab[] {
    return [this.vocab, ...[...this.groups.values()].map((g) => g.vocab)];
  }

  /**
   * The language model use of one recognizer, when its file is there and the
   * model gives top-k outputs or logits. rec_hw.onnx reads with the base model.
   */
  private lmUse(name: string, sess: InferenceSession): LmUse | undefined {
    const lm = this.lms.get(name);
    if (!this.useLm || !lm || !lmReady(sess)) return undefined;
    const p = this.lmParams[name] ?? {};
    return { lm, alpha: p.alpha ?? LM_ALPHA, beta: p.beta ?? LM_BETA, gate: p.gate ?? LM_GATE };
  }

  private async resolve(entry: PlanEntry): Promise<RecGroup> {
    const { sel } = entry;
    if (entry.kind === "base") {
      const sess = this.sessions.rec;
      return {
        sess,
        vocab: this.vocab,
        sel,
        asChars: false,
        lm: this.lmUse("base", sess),
        xscale: 1,
      };
    }
    if (entry.kind === "hw") {
      if (!this.hw) throw new Error("no handwriting recognizer");
      const sess = await this.hw.rec();
      return {
        sess,
        vocab: this.vocab,
        sel,
        asChars: false,
        lm: this.lmUse("base", sess),
        xscale: 1,
      };
    }
    const g = this.groups.get(entry.name);
    if (!g) throw new Error(`no recognizer for script group "${entry.name}"`);
    const sess = await g.session();
    const lm = this.lmUse(entry.name, sess);
    return { sess, vocab: g.vocab, sel, asChars: true, lm, xscale: g.xscale };
  }

  /**
   * Which recognizer reads which lines. With script.onnx, each line goes to
   * the recognizer of its script group, and a line the classifier is unsure
   * of goes to the main recognizer too. A vertical page's lines are CJK
   * columns, characters on their side: all go to the CJK recognizer.
   */
  private async plan(crops: readonly FloatImage[], vertical: boolean): Promise<PlanEntry[]> {
    const all = crops.map((_, i) => i);
    if (vertical && this.groups.has("cjk")) return [{ kind: "group", name: "cjk", sel: all }];
    if (!this.script || !crops.length) return [{ kind: "base", sel: all }];
    const { names, sure } = await lineGroups(
      this.Tensor,
      this.script,
      crops,
      this.scriptGroups,
      new Set(this.groups.keys())
    );
    const plan: PlanEntry[] = [
      { kind: "base", sel: all.filter((j) => names[j] === "base" || !sure[j]) },
    ];
    for (const name of this.groups.keys()) {
      const sel = all.filter((j) => names[j] === name);
      if (sel.length) plan.push({ kind: "group", name, sel });
    }
    return plan;
  }

  /** Per line crop (line band, base height): whether hw.onnx calls it handwritten. */
  private async hwLines(
    cls: InferenceSession,
    crops: readonly FloatImage[],
    batch: number
  ): Promise<boolean[]> {
    const out = crops.map(() => false);
    for (const idx of widthBatches(
      crops.map((c) => c.width),
      batch
    )) {
      const { data, dims, widths } = padBatch(idx.map((j) => item(crops, j)));
      const res = await cls.run({
        x: new this.Tensor("float32", data, dims),
        widths: new this.Tensor("int64", widths, [idx.length]),
      });
      const p = floats(output(res, "p"));
      for (const [k, j] of idx.entries()) out[j] = (p[k] ?? 0) >= this.hwRouteMin;
    }
    return out;
  }

  /** The main recognizer and each script group's, in meta.json order (`main`: rec.onnx). */
  public readers(): Reader[] {
    const main: Reader = {
      name: "base",
      vocab: this.vocab,
      session: async () => this.sessions.rec,
      xscale: 1,
      main: true,
    };
    return [
      main,
      ...[...this.groups].map(([name, g]): Reader => ({
        name,
        vocab: g.vocab,
        session: g.session,
        xscale: g.xscale,
        main: false,
      })),
    ];
  }

  /** The line-band crops of the boxes, and rec.onnx's tall-window crops, as `read` makes them. */
  public lineCrops(gray: Gray, boxes: readonly Box[]): LineCrops {
    const dark = this.window > 1 ? boxes.map((b) => isDarkLine(gray, b)) : undefined;
    const crop = (b: Box, i: number, window: number): FloatImage =>
      cropLine(gray, b, this.height, this.ops, 4096, window, dark?.[i]);
    const crops = boxes.map((b, i) => crop(b, i, 1));
    const tall = this.window > 1 ? boxes.map((b, i) => crop(b, i, this.window)) : crops;
    return { crops, tall };
  }

  /** The script group of each line crop and whether the classifier is sure; base and sure without script.onnx. */
  public async cropGroups(crops: readonly FloatImage[]): Promise<LineGroups> {
    if (!this.script) return { names: crops.map(() => "base"), sure: crops.map(() => true) };
    return lineGroups(
      this.Tensor,
      this.script,
      crops,
      this.scriptGroups,
      new Set(this.groups.keys())
    );
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
    // Light-on-dark is decided once per line box, for every crop of it.
    const dark = this.window > 1 ? boxes.map((b) => isDarkLine(gray, b)) : undefined;
    const crop = (b: Box, i: number, height: number, window: number): FloatImage =>
      cropLine(gray, b, height, this.ops, 4096, window, dark?.[i]);
    const crops = boxes.map((b, i) => crop(b, i, this.height, 1));
    // rec.onnx may read a tall window: the line and the page around it.
    const tall =
      this.window > 1 ? boxes.map((b, i) => crop(b, i, this.height, this.window)) : crops;
    const plan = await this.plan(crops, vertical);
    // Handwritten base lines go to rec_hw.onnx, with crops at its own height.
    const hwBands = new Map<number, FloatImage>();
    const hwTall = new Map<number, FloatImage>();
    const base = plan[0];
    if (this.hw && !vertical && crops.length && base?.kind === "base") {
      const hw = await this.hwLines(this.hw.cls, crops, batch);
      const moved = base.sel.filter((j) => hw[j]);
      if (moved.length) {
        base.sel = base.sel.filter((j) => !hw[j]);
        plan.push({ kind: "hw", sel: moved });
        for (const j of moved) {
          const band = crop(item(boxes, j), j, this.hwHeight, 1);
          hwBands.set(j, band);
          hwTall.set(
            j,
            this.hwWindow > 1 ? crop(item(boxes, j), j, this.hwHeight, this.hwWindow) : band
          );
        }
      }
    }
    const out: RecLine[] = crops.map(() => ({ text: "", conf: 0, toks: [] }));
    const bestConf = crops.map(() => -1);
    for (const entry of plan) {
      const g = await this.resolve(entry);
      const bandOf = (j: number): FloatImage =>
        entry.kind === "hw" ? cropOf(hwBands, j) : item(crops, j);
      const srcOf = (j: number): FloatImage => {
        if (entry.kind === "hw") return cropOf(hwTall, j);
        return g.sess === this.sessions.rec ? item(tall, j) : item(crops, j);
      };
      for (const part of widthBatches(
        g.sel.map((j) => bandOf(j).width),
        batch
      )) {
        const idx = part.map((i) => item(g.sel, i));
        const lines = await recognizeBatch(
          this.Tensor,
          g,
          widen(idx.map(srcOf), g.xscale, this.ops),
          idx.map(bandOf),
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
    if (this.math && crops.length) {
      await readMath(this.math, gray, boxes, crops, out, vertical, this.ops, this.vocab);
    }
    return out;
  }
}
