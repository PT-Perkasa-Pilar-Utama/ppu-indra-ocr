// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama
// Ported from datalab-to/indra indra_ocr/pipeline.py (Apache-2.0, Copyright Datalab).

import type { Part } from "./assemble.js";
import { buildPage, mirrorBox, pageIsRtl, splitParts, verticalOrder } from "./assemble.js";
import type { Gray } from "./image.js";
import { decodeToks } from "./lines.js";
import { isFormula, unspliced } from "./math-span-img.js";
import type { MathReader } from "./math.js";
import type { TableModels } from "./page-tables.js";
import { pageTables } from "./page-tables.js";
import { BLOCK_CLASSES } from "./order-post.js";
import type { Box } from "./rec-preproc.js";
import type { Page, ReadingOrder, RecLine } from "./types.js";
import type { Vocab } from "./vocab.js";

const TABLE = BLOCK_CLASSES.indexOf("Table");

/** What laying out a page needs from the pipeline. */
export type LayoutDeps = {
  vocab: Vocab;
  furniture: boolean;
  math: MathReader | undefined;
  tables: TableModels;
  readingOrder(
    boxes: readonly Box[],
    texts: readonly string[],
    width: number,
    height: number,
    gray?: Gray,
    imgBoxes?: readonly Box[]
  ): Promise<ReadingOrder>;
};

/**
 * The texts the order and layout model reads: a line with a spliced formula
 * as the recognizer read it, so blocks and tables stay as without the formula.
 */
function orderTexts(vocab: Vocab, parts: readonly Part[], pageRtl: boolean): string[] {
  return parts.map((p) =>
    p.toks.some(isFormula) ? decodeToks(vocab, unspliced(p.toks), pageRtl) : p.text
  );
}

/** Whether line `i` holds the Table class (argmax of its class logits). */
function inTable(cls: Float32Array, i: number): boolean {
  const c = BLOCK_CLASSES.length;
  let best = 0;
  for (let k = 1; k < c; k++) if ((cls[i * c + k] ?? 0) > (cls[i * c + best] ?? 0)) best = k;
  return best === TABLE;
}

/**
 * Lines -> split at wide gaps -> tidy -> reading order -> blocks and
 * markdown. `rec` holds the text, confidence and tokens of each box on `gray`.
 * A right-to-left page orders and lays out its lines mirrored, so the right
 * column comes first. `vertical`: a vertical page read turned, ordered by
 * XY-cut.
 */
export async function layoutPage(
  deps: LayoutDeps,
  gray: Gray,
  boxes: readonly Box[],
  rec: readonly RecLine[],
  t: Record<string, number>,
  vertical: boolean,
  fromLayer: boolean
): Promise<Page> {
  const t0 = performance.now();
  const { width, height } = gray;
  const pageRtl = pageIsRtl(rec);
  let parts = splitParts(deps.vocab, boxes, rec, pageRtl);
  const order = async (ps: readonly Part[]): Promise<ReadingOrder> =>
    deps.readingOrder(
      ps.map((p) => (pageRtl ? mirrorBox(p.box, width) : p.box)),
      orderTexts(deps.vocab, ps, pageRtl),
      width,
      height,
      gray,
      ps.map((p) => p.box)
    );
  let ro = await order(parts);
  const math = deps.math;
  const { cls } = ro;
  if (math?.hasAcceptHead && cls && parts.some((p) => p.math)) {
    // The accept head sees the table class only now: a reading on a table
    // line goes back to the plain line when the head rejects it there.
    let back = false;
    parts = parts.map((p, i) => {
      const q = p.math;
      if (!q || !inTable(cls, i)) return p;
      if (math.accept(q.reading, q.conf, q.draft, q.rconf, 1.0) !== null) return p;
      back = true;
      return { box: p.box, text: q.plain.text, toks: q.plain.toks, conf: q.plain.conf };
    });
    if (back) ro = await order(parts);
  }
  if (vertical) {
    ro = verticalOrder(
      parts.map((p) => p.box),
      ro
    );
  }
  t["order_ms"] = performance.now() - t0;
  const tab = vertical
    ? { ro, ms: 0 }
    : await pageTables(deps.tables, deps.vocab, gray, parts, ro, pageRtl, fromLayer);
  const page = buildPage(
    deps.vocab,
    parts,
    tab.ro,
    vertical ? "order=xycut" : "order=model",
    deps.furniture,
    t,
    pageRtl,
    width,
    tab.tables
  );
  // As the reference: the table time is not part of total_ms.
  if (tab.tables) page.timings["table_ms"] = tab.ms;
  return page;
}
