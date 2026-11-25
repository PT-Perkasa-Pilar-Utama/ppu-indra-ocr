// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama
// Ported from datalab-to/indra indra_ocr/layout_post.py (Apache-2.0, Copyright Datalab).
//
// Tables from the order model's table heads (`tab`: logits same row, same
// column, same cell for every pair of lines). The reference reads them only with
// load(table_heads=True); `blockPart` in layout-post.ts uses them when given `tab`.

import { meanF32 } from "./numeric.js";
import { cluster, medianF32 } from "./order-post.js";
import { pyHtmlEscape } from "./py-text.js";
import type { Box } from "./rec-preproc.js";
import { item } from "./util.js";

const f = Math.fround;

function mapAt<K, V>(m: ReadonlyMap<K, V>, key: K): V {
  const v = m.get(key);
  if (v === undefined) throw new RangeError(`no entry for ${String(key)}`);
  return v;
}

/** The N x N matrices (same row, column, cell) of the lines `blk`, from `tab` [3, n, n]. */
export function subTab(tab: Float32Array, n: number, blk: readonly number[]): Float32Array {
  if (tab.length !== 3 * n * n)
    throw new RangeError(`tab holds ${tab.length} values, want 3 * ${n} * ${n}`);
  const k = blk.length;
  const out = new Float32Array(3 * k * k);
  for (let m = 0; m < 3; m++)
    for (const [a, i] of blk.entries())
      for (const [b, j] of blk.entries()) out[(m * k + a) * k + b] = item(tab, (m * n + i) * n + j);
  return out;
}

/**
 * The lines of one cell are stacked (they overlap horizontally). When the cell
 * head joins side-by-side lines in more than 10% of its cells, it has failed on
 * this table and the geometric assembly is used instead. `tab` is [3, k, k] for
 * the k lines of `boxes`.
 */
export function headsPlausible(boxes: readonly Box[], tab: Float32Array): boolean {
  const k = boxes.length;
  const cells = cluster(tab.subarray(2 * k * k, 3 * k * k), k);
  let bad = 0;
  for (const c of cells) {
    if (c.length < 2) continue;
    const x0 = c.map((i) => item(boxes, i)[0]);
    const x1 = c.map((i) => item(boxes, i)[2]);
    let min = Infinity;
    for (const [a, p] of x1.entries())
      for (const [b, q] of x1.entries())
        min = Math.min(min, f(Math.min(p, q) - Math.max(item(x0, a), item(x0, b))));
    if (min < 0) bad++;
  }
  return bad <= 0.1 * cells.length;
}

/**
 * An HTML table from the order model's table heads. `tab` [3, k, k] holds the
 * logits same row, same column, same cell for the k lines of one table block.
 */
export function tableHtmlHeads(
  boxes: readonly Box[],
  texts: readonly string[],
  tab: Float32Array
): string {
  const k = boxes.length;
  const cells = cluster(tab.subarray(2 * k * k, 3 * k * k), k);
  const cb = cells.map((c): Box => [
    Math.min(...c.map((i) => item(boxes, i)[0])),
    Math.min(...c.map((i) => item(boxes, i)[1])),
    Math.max(...c.map((i) => item(boxes, i)[2])),
    Math.max(...c.map((i) => item(boxes, i)[3])),
  ]);
  const ctext = cells.map((c) =>
    c
      .toSorted(
        (p, q) => item(boxes, p)[1] - item(boxes, q)[1] || item(boxes, p)[0] - item(boxes, q)[0]
      )
      .map((i) => item(texts, i))
      .join(" ")
  );
  const m = cells.length;
  const cellLogits = (plane: number): Float32Array => {
    const out = new Float32Array(m * m);
    for (let a = 0; a < m; a++) {
      for (let b = a + 1; b < m; b++) {
        const vals: number[] = [];
        for (const i of item(cells, a))
          for (const j of item(cells, b)) vals.push(item(tab, (plane * k + i) * k + j));
        out[a * m + b] = out[b * m + a] = meanF32(vals);
      }
    }
    return out;
  };
  const byMin = (axis: 0 | 1) => (g: number[]) => Math.min(...g.map((i) => item(cb, i)[axis]));
  const rows = cluster(cellLogits(0), m).toSorted((p, q) => byMin(1)(p) - byMin(1)(q));
  const cols = cluster(cellLogits(1), m).toSorted((p, q) => byMin(0)(p) - byMin(0)(q));
  const rowOf = new Map(rows.flatMap((g, r) => g.map((c) => [c, r] as const)));
  const colOf = new Map(cols.flatMap((g, c) => g.map((i) => [i, c] as const)));
  const centers = cols.map((g) =>
    medianF32(g.map((i) => f(f(item(cb, i)[0] + item(cb, i)[2]) / 2)))
  );
  const h = medianF32(boxes.map(([, y0, , y1]) => f(y1 - y0)));
  const grid = new Map<string, { span: number; text: string }>();
  for (let c = 0; c < m; c++) {
    const c0 = mapAt(colOf, c);
    const [bx0, , bx2] = item(cb, c);
    let span = 1;
    while (
      c0 + span < cols.length &&
      f(bx0 - f(h)) <= item(centers, c0 + span) &&
      item(centers, c0 + span) <= f(bx2 + f(h))
    )
      span++;
    const key = `${mapAt(rowOf, c)},${c0}`;
    const prev = grid.get(key);
    grid.set(
      key,
      prev
        ? { span: Math.max(prev.span, span), text: `${prev.text} ${item(ctext, c)}` }
        : { span, text: item(ctext, c) }
    );
  }
  const out = ["<table>"];
  for (let r = 0; r < rows.length; r++) {
    const tag = r === 0 && rows.length > 1 ? "th" : "td";
    let tds = "";
    for (let c = 0; c < cols.length;) {
      const cell = grid.get(`${r},${c}`);
      if (cell) {
        const attr = cell.span > 1 ? ` colspan="${cell.span}"` : "";
        tds += `<${tag}${attr}>${pyHtmlEscape(cell.text)}</${tag}>`;
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
