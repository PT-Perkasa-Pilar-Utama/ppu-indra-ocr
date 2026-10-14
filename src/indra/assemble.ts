// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama
// Ported from datalab-to/indra indra_ocr/pipeline.py (Apache-2.0, Copyright Datalab).

import type { Cell, LayoutBlock } from "./layout-post.js";
import { pageBlocks, pageMarkdown } from "./layout-post.js";
import { CELL_GAP, SPLIT_GAP, gutterCuts, splitLine } from "./lines.js";
import type { Box } from "./rec-preproc.js";
import type { Line, LinePart, Page, ReadingOrder, RecLine } from "./types.js";
import { pdfOrderCoherent } from "./textlayer.js";
import { roundTo } from "./upright.js";
import { item } from "./util.js";
import type { Vocab } from "./vocab.js";

/** A line part with the confidence of the line it came from. */
export type Part = LinePart & { conf: number };

/** A box rounded to float32, as the reference stores split boxes in a float32 array. */
function f32Box([x0, y0, x1, y1]: Box): Box {
  const f = Math.fround;
  return [f(x0), f(y0), f(x1), f(y1)];
}

/** Lines split at wide gaps and column gutters (SPLIT_GAP), the way text-layer labels split them. */
export function splitParts(vocab: Vocab, boxes: readonly Box[], rec: readonly RecLine[]): Part[] {
  const cuts = gutterCuts(vocab, boxes, rec);
  return boxes.flatMap((b, i) =>
    splitLine(vocab, b, item(rec, i).toks, SPLIT_GAP, item(cuts, i)).map((p) => ({
      ...p,
      box: f32Box(p.box),
      conf: item(rec, i).conf,
    }))
  );
}

/** A reading order and how it was chosen. */
export type ChosenOrder = { order: number[]; note: string };

/**
 * Keep the PDF content order when every part comes from the text layer and
 * that order is coherent (few upward jumps, the order model agrees with most
 * neighbours); otherwise the model's order.
 */
export function chooseOrder(
  parts: readonly Part[],
  ro: ReadingOrder,
  contentOrder: boolean
): ChosenOrder {
  const n = parts.length;
  if (contentOrder && n > 1) {
    const keys = parts.map((p) => {
      const idx = p.toks.flatMap((tk) => (tk[2] === undefined ? [] : [tk[2]]));
      return idx.length ? Math.min(...idx) : undefined;
    });
    if (keys.every((k) => k !== undefined)) {
      const pdf = keys.map((_, i) => i).toSorted((a, b) => (keys[a] ?? 0) - (keys[b] ?? 0));
      const logits = ro.logits;
      const sub = logits
        ? Float32Array.from(
            { length: n * n },
            (_, k) => logits[item(pdf, Math.floor(k / n)) * n + item(pdf, k % n)] ?? 0
          )
        : undefined;
      if (
        pdfOrderCoherent(
          pdf.map((i) => item(parts, i).box),
          sub
        )
      )
        return { order: pdf, note: "order=pdf" };
    }
  }
  return { order: ro.order, note: "order=model" };
}

/** The page from ordered line parts: output lines, blocks and markdown. */
export function buildPage(
  vocab: Vocab,
  parts: readonly Part[],
  { cls, same }: ReadingOrder,
  { order, note }: ChosenOrder,
  furniture: boolean,
  t: Record<string, number>
): Page {
  t["total_ms"] = Object.entries(t).reduce((sum, [k, v]) => (k === "total_ms" ? sum : sum + v), 0);
  const boxes = parts.map((p) => p.box);
  const texts = parts.map((p) => p.text);
  const lines = order.map((i): Line => {
    const p = item(parts, i);
    const [x0, y0, x1, y1] = p.box;
    return {
      box: [roundTo(x0, 1), roundTo(y0, 1), roundTo(x1, 1), roundTo(y1, 1)],
      text: p.text,
      conf: roundTo(p.conf, 4),
    };
  });
  let markdown: string | null = null;
  let blocks: LayoutBlock[] = [];
  if (cls && same) {
    blocks = pageBlocks(boxes, texts, order, cls, same);
    // Inside tables, lines split into cells at smaller gaps (CELL_GAP).
    const cells = new Map<number, Cell[]>(
      parts.map((p, i) => [
        i,
        splitLine(vocab, p.box, p.toks, CELL_GAP).map(({ box, text }) => ({ box, text })),
      ])
    );
    markdown = pageMarkdown(boxes, texts, order, cls, same, furniture, cells);
  } else if (parts.length < 2) {
    markdown = texts.join("\n");
  }
  return {
    lines,
    text: lines.map((l) => l.text).join("\n"),
    markdown,
    blocks,
    source: "ocr",
    note,
    rotation: 0,
    skew: 0,
    timings: t,
  };
}
