// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama
// Ported from datalab-to/indra indra_ocr/pipeline.py (Apache-2.0, Copyright Datalab).

import type { GapItem } from "./agree.js";
import { forcedGaps, layerText, targetIds, writes } from "./agree.js";
import type { LmInputs } from "./ctc.js";
import { topKLogits } from "./ctc.js";
import type { Gray, ImageOps } from "./image.js";
import { meanF32 } from "./numeric.js";
import type { Box } from "./rec-preproc.js";
import { padBatch, widen } from "./rec-preproc.js";
import { hasLogits, lmReady } from "./recognize.js";
import type { Recognizers } from "./recognizers.js";
import type { TensorCtor, Tok } from "./types.js";
import { floats, ints, item, output } from "./util.js";

/**
 * One checked line: the log-probability per token by which the line's
 * recognizer finds the layer's text less likely than its own best reading
 * (null for a line read as math: the recognizer writes LaTeX where the layer
 * holds Unicode symbols, so the two do not compare), that recognizer's
 * confidence in its own reading, and the recognizer ("base" or a group).
 */
export type LayerRead = { gap: number | null; conf: number; group: string };

/**
 * Per line box and its text-layer tokens: the gap, confidence and recognizer
 * (LayerRead). script.onnx routes each crop; a line the router is not sure of
 * is also read by each other recognizer that writes the layer's letters, and
 * the smallest gap counts.
 */
export async function layerGaps(
  Tensor: TensorCtor,
  recs: Recognizers,
  ops: ImageOps,
  gray: Gray,
  boxes: readonly Box[],
  tokens: readonly (readonly Tok[])[]
): Promise<LayerRead[]> {
  if (!boxes.length) return [];
  const readers = recs.readers();
  const main = item(readers, 0);
  if (!lmReady(await main.session())) {
    // A recognizer without top-k outputs or logits (the v3.1 models) gives no
    // gap: its lines drop out of the check, as math lines do (indra-ocr 1.2.0
    // fails the page here instead).
    return boxes.map(() => ({ gap: null, conf: 0, group: "base" }));
  }
  const { crops, tall } = recs.lineCrops(gray, boxes);
  const { names, sure } = await recs.cropGroups(crops);
  const texts = new Map(
    readers.map((r) => [r.name, tokens.map((t) => layerText(t, r.name === "rtl"))])
  );
  const textOf = (name: string, j: number): string => item(texts.get(name) ?? [], j);
  const cands = names.map((g, j) => {
    const c = new Set([g]);
    if (!sure[j]) for (const r of readers) if (writes(textOf(r.name, j), r.vocab)) c.add(r.name);
    return c;
  });
  const mathLines = new Set<number>();
  const todo: { j: number; conf: number; name: string; item: GapItem }[] = [];
  for (const r of readers) {
    const sel = cands.flatMap((c, j) => (c.has(r.name) ? [j] : []));
    if (!sel.length) continue;
    const sess = await r.session();
    const src = sel.map((j) => item(r.main ? tall : crops, j));
    const { data, dims, widths } = padBatch(widen(src, r.xscale, ops));
    const feed = {
      x: new Tensor("float32", data, dims),
      widths: new Tensor("int64", widths, [sel.length]),
    };
    const logitsOut = hasLogits(sess);
    const res = await sess.run(
      feed,
      logitsOut ? ["ids", "lengths", "logits"] : ["ids", "conf", "lengths", "blank", "topv", "topi"]
    );
    const ids = output(res, "ids");
    const best = ints(ids);
    const steps = ids.dims[1] ?? 0;
    const lengths = ints(output(res, "lengths"));
    let lines: LmInputs[];
    if (logitsOut) {
      const logits = output(res, "logits");
      const v = logits.dims[2] ?? 0;
      lines = topKLogits(floats(logits), sel.length, steps, v, Math.max(...lengths));
    } else {
      const topvT = output(res, "topv");
      const K = topvT.dims[2] ?? 0;
      const blank = floats(output(res, "blank"));
      const topv = floats(topvT);
      const topi = Int32Array.from(ints(output(res, "topi")));
      lines = sel.map((_, k) => ({
        blank: blank.subarray(k * steps, (k + 1) * steps),
        topv: topv.subarray(k * steps * K, (k + 1) * steps * K),
        topi: topi.subarray(k * steps * K, (k + 1) * steps * K),
        k: K,
      }));
    }
    const dollar = r.vocab.index.get("$"); // a math span opens: the reading is LaTeX
    for (const [k, j] of sel.entries()) {
      const n = item(lengths, k);
      const seq = best.slice(k * steps, k * steps + n);
      if (dollar !== undefined && seq.includes(dollar)) {
        mathLines.add(j);
        continue;
      }
      const { blank, topv, topi, k: K } = item(lines, k);
      const peaks: number[] = [];
      for (let t = 0; t < n; t++) {
        if (seq[t] === 0 || (t > 0 && seq[t] === seq[t - 1])) continue;
        peaks.push(Math.fround(Math.exp(Math.max(blank[t] ?? 0, topv[t * K] ?? 0))));
      }
      const conf = peaks.length ? meanF32(peaks) : 0;
      const target = targetIds(textOf(r.name, j), r.vocab, r.name === "rtl");
      const gapItem: GapItem = {
        blank: blank.subarray(0, n),
        topv: topv.subarray(0, n * K),
        topi: topi.subarray(0, n * K),
        k: K,
        ids: target,
      };
      todo.push({ j, conf, name: r.name, item: gapItem });
    }
  }
  const out: LayerRead[] = names.map((g) => ({ gap: Infinity, conf: 0, group: g }));
  const gaps = forcedGaps(todo.map((t) => t.item));
  for (const [i, { j, conf, name }] of todo.entries()) {
    const g = item(gaps, i);
    if (g < (item(out, j).gap ?? Infinity)) out[j] = { gap: g, conf, group: name };
  }
  for (const j of mathLines) out[j] = { gap: null, conf: 0, group: item(names, j) };
  return out;
}
