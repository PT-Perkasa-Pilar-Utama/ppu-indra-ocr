// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama
// Ported from datalab-to/indra indra_ocr/pipeline.py (Apache-2.0, Copyright Datalab).

import type { Cell, LayoutBlock } from "./layout-post.js";
import { pageBlocks, pageMarkdown } from "./layout-post.js";
import { CELL_GAP, SPLIT_GAP, gutterCuts, splitLine } from "./lines.js";
import { BLOCK_CLASSES, GRID_CLASSES, xycutBlocks } from "./order-post.js";
import type { Box } from "./rec-preproc.js";
import { isRtl } from "./rtl.js";
import type { Line, LinePart, Page, ReadingOrder, RecLine, Tok } from "./types.js";
import { roundTo } from "./upright.js";
import { isPySpace, item } from "./util.js";
import type { Vocab } from "./vocab.js";

const f = Math.fround;

/** A line part with the confidence of the line it came from. */
export type Part = LinePart & { conf: number };

/** A box rounded to float32, as the reference stores split boxes in a float32 array. */
function f32Box([x0, y0, x1, y1]: Box): Box {
  return [f(x0), f(y0), f(x1), f(y1)];
}

const LETTER = /^\p{L}$/u;

/**
 * Whether a page reads right to left: more right-to-left letters than other
 * letters in its recognized lines (letter counts do not depend on order).
 */
export function pageIsRtl(rec: readonly RecLine[]): boolean {
  let rtl = 0;
  let all = 0;
  for (const r of rec) {
    for (const c of r.text) {
      if (!LETTER.test(c)) continue;
      all++;
      if (isRtl(c)) rtl++;
    }
  }
  return rtl > all - rtl;
}

/**
 * Lines split at wide gaps and column gutters (SPLIT_GAP), the way text-layer
 * labels split them, then tidied (tidyLines).
 */
export function splitParts(
  vocab: Vocab,
  boxes: readonly Box[],
  rec: readonly RecLine[],
  pageRtl = false
): Part[] {
  const cuts = gutterCuts(vocab, boxes, rec);
  const parts = boxes.flatMap((b, i) =>
    splitLine(vocab, b, item(rec, i).toks, SPLIT_GAP, item(cuts, i), pageRtl).map((p) => ({
      ...p,
      box: f32Box(p.box),
      conf: item(rec, i).conf,
    }))
  );
  return tidyLines(parts, pageRtl, vocab.index.get(" "));
}

/** Python's str.strip() with no argument. */
function pyStrip(s: string): string {
  const cps = [...s];
  let a = 0;
  let b = cps.length;
  while (a < b && isPySpace(item(cps, a))) a++;
  while (b > a && isPySpace(item(cps, b - 1))) b--;
  return cps.slice(a, b).join("");
}

// Dot leaders and fill-in rules: . middle-dot ellipsis two-dot-leader _ - en-dash em-dash =
const LEADER_CHARS = new Set([
  ..."._-=",
  ...[0xb7, 0x2026, 0x2025, 0x2013, 0x2014].map((c) => String.fromCharCode(c)),
]);
// A lone short dash can be an empty table cell; other lone marks are not.
const DASHES = new Set(["-", ...[0x2013, 0x2014].map((c) => String.fromCharCode(c))]);
// List markers (not dashes: a "-" can be a table cell).
const BULLETS = new Set(
  [
    0x2022, 0x25e6, 0x25aa, 0x25ab, 0x25cf, 0x25cb, 0x25a0, 0x25a1, 0x2023, 0x2043, 0x2219, 0x2a,
  ].map((c) => String.fromCharCode(c))
);
// A numbered list marker: "1.", "12)", "a.", "B)", "(3)", "(b)", "(iv)".
const ENUMERATOR =
  /^(?:(?:\p{Nd}{1,3}|[A-Za-z])[.)]|\((?:\p{Nd}{1,3}|[A-Za-z]|[ivxlcIVXLC]{1,5})\))$/u;
// A list marker and its item's text are at most this many line heights apart.
const MARKER_GAP = 2.5;

/**
 * Two kinds of lines carry no text of their own:
 *
 * - a line of leader characters only (form fill-in dots and rules, specks) is
 *   dropped: 3 or more of them, a box 3 or more times wider than tall, or any
 *   dot or underscore alone. A short lone dash stays: it can be an empty cell;
 * - a lone list marker (a bullet, "1.", "(a)") joins the line that starts
 *   within MARKER_GAP line heights after it on the same row (on its left on a
 *   right-to-left page). On a left-to-right page the marker's tokens join too
 *   (`space`: the space token), so a table line still splits into cells.
 *
 * Other lines without letters or digits ("-", "/ /", "( )") stay.
 */
export function tidyLines(parts: readonly Part[], rtl = false, space?: number): Part[] {
  const keep = parts.map(() => true);
  const out: Part[] = [...parts];
  const h = parts.map((p) => f(p.box[3] - p.box[1]));
  const w = parts.map((p) => f(p.box[2] - p.box[0]));
  for (const [i, p] of parts.entries()) {
    const core = new Set(p.text.replaceAll(" ", ""));
    if (core.size && [...core].every((c) => LEADER_CHARS.has(c))) {
      const wide = item(w, i) >= f(3 * item(h, i));
      const n = [...p.text.replaceAll(" ", "")].length;
      if (n >= 3 || wide || ![...core].some((c) => DASHES.has(c))) keep[i] = false;
    }
  }
  for (const i of parts.keys()) {
    const cur = item(out, i);
    const t = pyStrip(cur.text);
    if (!keep[i] || !(BULLETS.has(t) || ENUMERATOR.test(t))) continue;
    const bi = cur.box;
    const hi = item(h, i);
    let best = -1;
    let bestGap = 0;
    for (const [j, other] of out.entries()) {
      if (j === i || !keep[j] || !pyStrip(other.text)) continue;
      const bj = other.box;
      const hj = item(h, j);
      const ov = f(Math.min(bi[3], bj[3]) - Math.max(bi[1], bj[1]));
      if (ov < f(0.5 * Math.min(hi, hj))) continue;
      const gap = f(rtl ? bi[0] - bj[2] : bj[0] - bi[2]);
      if (
        f(f(-0.3) * hi) <= gap &&
        gap <= f(MARKER_GAP * Math.max(hi, hj)) &&
        (best < 0 || gap < bestGap)
      ) {
        best = j;
        bestGap = gap;
      }
    }
    if (best < 0) continue;
    const target = item(out, best);
    let toks: readonly Tok[] = target.toks;
    const chars = toks.length > 0 && typeof item(toks, 0)[0] === "string";
    const sp = chars ? " " : space;
    // Same kind of tokens (text-layer chars or recognizer ids) only.
    if (!rtl && cur.toks.length && toks.length && sp !== undefined) {
      if ((typeof item(cur.toks, 0)[0] === "string") === chars) {
        const x = f(bi[2] + target.box[0]) / 2;
        const last = item(cur.toks, cur.toks.length - 1);
        const join: Tok = last.length > 2 ? [sp, x, last[2]] : [sp, x];
        toks = [...cur.toks, join, ...toks];
      }
    }
    out[best] = {
      box: [
        Math.min(bi[0], target.box[0]),
        Math.min(bi[1], target.box[1]),
        Math.max(bi[2], target.box[2]),
        Math.max(bi[3], target.box[3]),
      ],
      text: `${t} ${pyStrip(target.text)}`,
      toks,
      conf: target.conf,
    };
    keep[i] = false;
  }
  return out.filter((_, i) => keep[i]);
}

/** A box mirrored left to right on a page `width` wide, in float32. */
export function mirrorBox([x0, y0, x1, y1]: Box, width: number): Box {
  return [f(width - x1), f(y0), f(width - x0), f(y1)];
}

/**
 * Reading order and layout for a vertical page read turned: the order model
 * saw no vertical pages (turned, their columns of short lines look like one
 * table read row by row across tiers). Blocks and order come from XY-cut, and
 * no line gets a grid class (Table, Form, Table-Of-Contents).
 */
export function verticalOrder(boxes: readonly Box[], ro: ReadingOrder): ReadingOrder {
  if (!ro.cls) return ro;
  const n = boxes.length;
  const blocks = xycutBlocks(boxes);
  const same = new Float32Array(n * n).fill(-1);
  for (const b of blocks) for (const i of b) for (const j of b) same[i * n + j] = 1;
  const cls = ro.cls.slice();
  for (let i = 0; i < n; i++)
    for (const c of GRID_CLASSES) cls[i * BLOCK_CLASSES.length + c] = -1e4;
  return { order: blocks.flat(), cls, same, logits: ro.logits };
}

/**
 * The page from ordered line parts: output lines, blocks and markdown. On a
 * right-to-left page (`pageRtl`), blocks and markdown work on the page
 * mirrored left to right, so the first column is the right one; output boxes
 * stay on the page as it is.
 */
export function buildPage(
  vocab: Vocab,
  parts: readonly Part[],
  { order, cls, same }: ReadingOrder,
  note: string,
  furniture: boolean,
  t: Record<string, number>,
  pageRtl = false,
  width = 0
): Page {
  t["total_ms"] = Object.entries(t).reduce((sum, [k, v]) => (k === "total_ms" ? sum : sum + v), 0);
  const geo = (b: Box): Box => (pageRtl ? mirrorBox(b, width) : b);
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
    const mirrored = boxes.map(geo);
    blocks = pageBlocks(mirrored, texts, order, cls, same).map((b) => {
      const [x0, y0, x1, y1] = geo(b.bbox);
      return {
        ...b,
        bbox: [roundTo(x0, 1), roundTo(y0, 1), roundTo(x1, 1), roundTo(y1, 1)],
      };
    });
    // Inside tables, lines split into cells at smaller gaps (CELL_GAP).
    const cells = new Map<number, Cell[]>(
      parts.map((p, i) => [
        i,
        splitLine(vocab, p.box, p.toks, CELL_GAP, [], pageRtl).map(({ box, text }) => ({
          box: geo(f32Box(box)),
          text,
        })),
      ])
    );
    markdown = pageMarkdown(mirrored, texts, order, cls, same, furniture, cells);
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
    vertical: false,
    timings: t,
  };
}
