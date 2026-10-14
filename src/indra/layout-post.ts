// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama
// Ported from datalab-to/indra indra_ocr/layout_post.py (Apache-2.0, Copyright Datalab).
//
// Not ported: table_html_heads / heads_plausible (only with table_heads=True,
// which the reference's load() never sets) and balance_math (unused upstream).

import { meanF32 } from "./numeric.js";
import type { BlockLabel } from "./order-post.js";
import { BLOCK_CLASSES, blockClass, medianF32 } from "./order-post.js";
import type { Box } from "./rec-preproc.js";
import { roundTo } from "./upright.js";
import { item } from "./util.js";

const f = Math.fround;

const HEADER = 2;
const FOOTER = 3;
const SECTION = 1;
const LIST = 4;
const TABLE = 5;

const WHOLE_MATH = /^\$[^$]+\$$/; // a line that is one formula
// A list marker, followed by text or alone on its line (the detector often cuts
// a bullet off as a line of its own; the next line then joins its item).
const ITEM = /^\s*([•\-–▪◦*·→]|\(?\p{Nd}{1,3}[.)]|\(?[a-zA-Z][.)])(\s|$)/u;
const ALPHA = /^\p{L}$/u;
const LOWER = /^\p{Lowercase}$/u;

/** A layout block: its label, the union of its line boxes, and its lines' text. */
export type LayoutBlock = { label: BlockLabel; bbox: Box; text: string };

/** One line split into table cells: each cell's box and text. */
export type Cell = { box: Box; text: string };

/** Han, kana, CJK punctuation and fullwidth forms: written without spaces (Hangul is not). */
function isCjk(c: string): boolean {
  const o = c.codePointAt(0) ?? 0;
  return (
    (o >= 0x3000 && o <= 0x30ff) ||
    (o >= 0x3400 && o <= 0x9fff) ||
    (o >= 0xf900 && o <= 0xfaff) ||
    (o >= 0xff00 && o <= 0xffef) ||
    (o >= 0x20000 && o <= 0x2fa1f)
  );
}

/**
 * Join the lines of one paragraph. "tumori-" + "genesis" -> "tumorigenesis";
 * a line break between two CJK characters takes no space.
 */
export function joinLines(texts: readonly string[]): string {
  let out: string[] = [];
  for (const raw of texts) {
    const t = [...raw.trim()];
    if (!t.length) continue;
    const first = item(t, 0);
    if (
      out.length > 1 &&
      out.at(-1) === "-" &&
      ALPHA.test(out.at(-2) ?? "") &&
      ALPHA.test(first) &&
      LOWER.test(first)
    ) {
      out = [...out.slice(0, -1), ...t];
    } else if (out.length && isCjk(out.at(-1) ?? "") && isCjk(first)) {
      out.push(...t);
    } else {
      out = out.length ? [...out, " ", ...t] : t;
    }
  }
  return out.join("");
}

/** Runs of the reading order with a positive same-block logit between neighbours. */
export function splitBlocks(order: readonly number[], same: Float32Array, n: number): number[][] {
  const blocks: number[][] = [];
  for (const i of order) {
    const last = blocks.at(-1);
    if (last && (same[(last.at(-1) ?? 0) * n + i] ?? 0) > 0) last.push(i);
    else blocks.push([i]);
  }
  return blocks;
}

/** Cell lines grouped into rows by vertical center, top to bottom, left to right. */
function tableRows(boxes: readonly Box[]): number[][] {
  const cy = boxes.map(([, y0, , y1]) => f(f(y0 + y1) / 2));
  const h = medianF32(boxes.map(([, y0, , y1]) => f(y1 - y0)));
  const rows: number[][] = [];
  for (const i of cy.map((_, k) => k).toSorted((p, q) => item(cy, p) - item(cy, q))) {
    const last = rows.at(-1);
    if (last && Math.abs(f(item(cy, i) - meanF32(last.map((k) => item(cy, k))))) < 0.5 * h)
      last.push(i);
    else rows.push([i]);
  }
  return rows.map((r) => r.toSorted((p, q) => item(boxes, p)[0] - item(boxes, q)[0]));
}

/** numpy's median of float64 values. */
function median64(values: readonly number[]): number {
  const v = values.toSorted((a, b) => a - b);
  const m = Math.floor(v.length / 2);
  return v.length % 2 ? item(v, m) : (item(v, m - 1) + item(v, m)) / 2;
}

/**
 * Join the line rows of wrapped cells into one table row: a row joins the row
 * above when its gap is under half the median row gap and under 0.6 line
 * heights, and it has fewer lines than the row above.
 */
function mergeWrapped(rows: number[][], boxes: readonly Box[]): number[][] {
  if (rows.length < 3) return rows;
  const h = medianF32(boxes.map(([, y0, , y1]) => f(y1 - y0)));
  const gaps = rows.slice(0, -1).map((cur, k) => {
    const next = item(rows, k + 1);
    const top = Math.min(...next.map((i) => item(boxes, i)[1]));
    const bottom = Math.max(...cur.map((i) => item(boxes, i)[3]));
    return f(top - bottom);
  });
  const med = median64(gaps);
  const out = [[...item(rows, 0)]];
  let prev = item(rows, 0);
  for (const [k, g] of gaps.entries()) {
    const r = item(rows, k + 1);
    if (g < 0.5 * med && g < 0.6 * h && r.length < prev.length) out.at(-1)?.push(...r);
    else out.push([...r]);
    prev = r;
  }
  return out;
}

const escapeHtml = (s: string): string =>
  s.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");

/**
 * An HTML table from cell lines (one text line per cell). A first row that is
 * one line across most of the table is a title and goes before the table. The
 * first remaining row is the header (<th>).
 */
export function tableHtml(boxes: readonly Box[], texts: readonly string[]): string {
  let rows = tableRows(boxes);
  const spanX = f(Math.max(...boxes.map((b) => b[2])) - Math.min(...boxes.map((b) => b[0])));
  const top = item(item(rows, 0), 0);
  const [tx0, , tx1] = item(boxes, top);
  if (
    rows.length > 2 &&
    item(rows, 0).length === 1 &&
    Math.max(...rows.map((r) => r.length)) > 1 &&
    f(tx1 - tx0) > f(f(0.6) * spanX)
  ) {
    const keep = texts.map((_, k) => k).filter((k) => k !== top);
    return `${item(texts, top)}\n\n${tableHtml(
      keep.map((k) => item(boxes, k)),
      keep.map((k) => item(texts, k))
    )}`;
  }
  const width = Math.max(...rows.map((r) => r.length));
  const full = rows.filter((r) => r.length === width);
  rows = mergeWrapped(rows, boxes);
  // Column bands from the rows with the most cells: those rows rarely span.
  const bands = Array.from(
    { length: width },
    (_, c) =>
      [
        medianF32(full.map((r) => item(boxes, item(r, c))[0])),
        medianF32(full.map((r) => item(boxes, item(r, c))[2])),
      ] as const
  );
  // Widen each band to the midpoints between neighbours, so every x falls in one band.
  const edges = [
    -Infinity,
    ...bands.slice(0, -1).map((b, c) => f(f(b[1] + item(bands, c + 1)[0]) / 2)),
    Infinity,
  ];
  const bandOf = (x: number): number => edges.filter((e) => e <= x).length - 1;
  const out = ["<table>"];
  for (const [ri, r] of rows.entries()) {
    const tag = ri === 0 && rows.length > 1 ? "th" : "td";
    const cells = new Map<number, { span: number; text: string }>();
    const byPos = r.toSorted(
      (p, q) => item(boxes, p)[1] - item(boxes, q)[1] || item(boxes, p)[0] - item(boxes, q)[0]
    );
    for (const k of byPos) {
      const [x0, , x1] = item(boxes, k);
      const first = bandOf(x0);
      const last = bandOf(x1);
      // A cell spans every band it covers by more than half of the band.
      const span: number[] = [];
      for (let c = Math.max(first, 0); c <= Math.min(last, width - 1); c++) {
        const [b0, b1] = item(bands, c);
        if (f(Math.min(x1, b1) - Math.max(x0, b0)) > f(0.5 * f(b1 - b0))) span.push(c);
      }
      if (!span.length) span.push(Math.min(Math.max(first, 0), width - 1));
      const c0 = item(span, 0);
      const prev = cells.get(c0);
      cells.set(
        c0,
        prev
          ? { span: Math.max(prev.span, span.length), text: `${prev.text} ${item(texts, k)}` }
          : { span: span.length, text: item(texts, k) }
      );
    }
    let tds = "";
    for (let c = 0; c < width;) {
      const cell = cells.get(c);
      if (cell) {
        const attr = cell.span > 1 ? ` colspan="${cell.span}"` : "";
        tds += `<${tag}${attr}>${escapeHtml(cell.text)}</${tag}>`;
        c += cell.span;
      } else {
        tds += `<${tag}></${tag}>`;
        c += 1;
      }
    }
    out.push(`<tr>${tds}</tr>`);
  }
  out.push("</table>");
  return out.join("\n");
}

/** The layout blocks behind pageMarkdown, in reading order. */
export function pageBlocks(
  boxes: readonly Box[],
  texts: readonly string[],
  order: readonly number[],
  cls: Float32Array,
  same: Float32Array
): LayoutBlock[] {
  return splitBlocks(order, same, boxes.length).map((blk) => {
    const b = blk.map((i) => item(boxes, i));
    return {
      label: item(BLOCK_CLASSES, blockClass(cls, blk)),
      bbox: [
        roundTo(Math.min(...b.map((x) => x[0])), 1),
        roundTo(Math.min(...b.map((x) => x[1])), 1),
        roundTo(Math.max(...b.map((x) => x[2])), 1),
        roundTo(Math.max(...b.map((x) => x[3])), 1),
      ],
      text: blk.map((i) => item(texts, i)).join("\n"),
    };
  });
}

/**
 * Markdown for one page: section headers as "## ", paragraphs joined, lists one
 * item per line, tables as HTML from their cell lines, display math as $$...$$.
 * Page headers and footers are left out unless `keepFurniture`. `cells` maps a
 * line to its table cells when the line holds several.
 */
export function pageMarkdown(
  boxes: readonly Box[],
  texts: readonly string[],
  order: readonly number[],
  cls: Float32Array,
  same: Float32Array,
  keepFurniture = false,
  cells?: ReadonlyMap<number, readonly Cell[]>
): string {
  const parts: string[] = [];
  for (const blk of splitBlocks(order, same, boxes.length)) {
    const c = blockClass(cls, blk);
    const textsB = blk.map((i) => item(texts, i));
    if ((c === HEADER || c === FOOTER) && !keepFurniture) continue;
    if (textsB.every((t) => WHOLE_MATH.test(t.trim()))) {
      parts.push(textsB.map((t) => `$$${t.trim().slice(1, -1)}$$`).join("\n"));
      continue;
    }
    if (c === SECTION) {
      parts.push(`## ${joinLines(textsB)}`);
    } else if (c === TABLE && blk.length > 1) {
      const cb: Box[] = [];
      const ct: string[] = [];
      for (const i of blk) {
        const lineCells = cells?.get(i);
        for (const cell of lineCells?.length
          ? lineCells
          : [{ box: item(boxes, i), text: item(texts, i) }]) {
          const [x0, y0, x1, y1] = cell.box;
          cb.push([f(x0), f(y0), f(x1), f(y1)]);
          ct.push(cell.text);
        }
      }
      parts.push(tableHtml(cb, ct));
    } else if (c === LIST) {
      const items: string[][] = [];
      for (const t of textsB) {
        const last = items.at(-1);
        if (!last || ITEM.test(t)) items.push([t]);
        else last.push(t);
      }
      parts.push(items.map(joinLines).join("\n"));
    } else {
      parts.push(joinLines(textsB));
    }
  }
  return parts.filter((p) => p).join("\n\n");
}
